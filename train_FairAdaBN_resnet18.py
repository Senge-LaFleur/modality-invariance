#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
os.environ['MPLBACKEND'] = 'Agg'
import random
import math
import json
import shutil
from pathlib import Path
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from models.models_losses import DualResNet18, get_layer_wise_lr_params
from models.evaluation import (
    fairness, save_results_csv, plot_confusion_matrix,
    plot_per_class_metrics, plot_fairness_metrics,
    plot_training_curves, plot_tsne, build_loaders, LABEL_NAMES
)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Fair Adaptive BatchNorm Layer (Paper Sec 3)
# ------------------------------------------------------------
class FairAdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive BatchNorm that maintains separate affine parameters (gamma, beta)
    and running statistics for each sensitive subgroup.
    task_idx is obtained from the parent model's current_task_idx attribute.
    """
    def __init__(self, num_features, num_groups=2, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super().__init__()
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = nn.Parameter(torch.zeros(num_groups, num_features))
            self.bias = nn.Parameter(torch.zeros(num_groups, num_features))
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_groups, num_features))
            self.register_buffer('running_var', torch.ones(num_groups, num_features))

    def set_root(self, root):
        """Store a (non-parameter) reference to the root model so forward() can
        read current_task_idx without any parent traversal hacks."""
        import weakref
        self._root_ref = weakref.ref(root)

    def forward(self, x):
        # Get task_idx from the root module (the model)
        task_idx = None
        module = self
        while module is not None:
            if hasattr(module, 'current_task_idx'):
                task_idx = module.current_task_idx
                break
            module = getattr(module, '_parent', None)
            if module is None:
                # Try to traverse up using children's _modules? Simpler: assume root has attribute.
                # For robustness, we can also check the global state.
                break
        if task_idx is None:
            # Fallback: try to get from a global variable or raise error
            raise RuntimeError("FairAdaptiveBatchNorm2d could not find current_task_idx in parent modules. "
                               "Set model.current_task_idx before forward.")
        if task_idx < 0 or task_idx >= self.num_groups:
            raise ValueError(f"task_idx={task_idx} out of range [0, {self.num_groups-1}]")

        if self.affine:
            weight = self.weight[task_idx]
            bias = self.bias[task_idx]
        else:
            weight = None
            bias = None

        if self.track_running_stats:
            running_mean = self.running_mean[task_idx]
            running_var = self.running_var[task_idx]
        else:
            running_mean = None
            running_var = None

        return F.batch_norm(
            x, running_mean, running_var, weight, bias,
            self.training, self.momentum, self.eps
        )

def replace_bn(model, num_groups=2):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, FairAdaBN2d(
                child.num_features, num_groups, child.eps, child.momentum,
                child.affine, child.track_running_stats
            ))
        else:
            convert_module_to_fair_adanbn(child, num_groups, skip_names)


class FairDualResNet18(DualResNet18):
    """
    Extension of DualResNet18 where all BatchNorm layers are replaced with FairAdaptiveBatchNorm2d.
    Adds a `current_task_idx` attribute that is used by the adaptive BN layers.
    """
    def __init__(self, embed_dim=512, num_classes=5, num_skin_types=6,
                 pretrained=True, use_projection=False, num_groups=2):
        super(FairDualResNet18, self).__init__(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_skin_types=num_skin_types,
            pretrained=pretrained,
            use_projection=use_projection
        )
        # Replace all BatchNorm2d in both backbones
        convert_module_to_fair_adanbn(self.clinical_backbone, num_groups=num_groups, skip_names=[])
        convert_module_to_fair_adanbn(self.dermoscopic_backbone, num_groups=num_groups, skip_names=[])
        self.num_groups = num_groups
        self.current_task_idx = None  # will be set before forward

    def forward(self, batch):
        """
        Args:
            batch: dict with keys 'clinical', 'dermoscopic', optionally 'skin_type'
        Returns:
            dict with 'logits', 'z'
        """
        clinical_img = batch['clinical']
        dermoscopic_img = batch['dermoscopic']
        
        # Set current_task_idx for all adaptive BN layers
        if self.current_task_idx is None:
            raise RuntimeError("Must set model.current_task_idx before forward()")
        
        # Forward through backbones
        clinical_feat = self.clinical_backbone(clinical_img)
        dermoscopic_feat = self.dermoscopic_backbone(dermoscopic_img)
        
        # Concatenate
        z = torch.cat([clinical_feat, dermoscopic_feat], dim=1)
        if self.use_projection:
            z = self.projection(z)
        logits = self.classifier(z)
        return {'logits': logits, 'z': z}


# ------------------------------------------------------------
# Statistical Disparity Loss (L_SD) - Paper Eq.4
# ------------------------------------------------------------
class StatisticalDisparityLoss(nn.Module):
    """L_SD = sum_{y} || P(Ŷ=y|A=0) - P(Ŷ=y|A=1) ||^2"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    def forward(self, preds0, preds1):
        if len(preds0)==0 or len(preds1)==0:
            return torch.tensor(0.0, device=preds0.device)
        p0 = torch.zeros(self.num_classes, device=preds0.device)
        p1 = torch.zeros(self.num_classes, device=preds1.device)
        p0.scatter_add_(0, preds0, torch.ones_like(preds0, dtype=torch.float))
        p1.scatter_add_(0, preds1, torch.ones_like(preds1, dtype=torch.float))
        p0 /= len(preds0)
        p1 /= len(preds1)
        return ((p0 - p1)**2).sum()

# ============================================================
# Training and validation functions
# ============================================================
def train_epoch(model, loader, optimizer, epoch, scaler, device, alpha, num_classes):
    model.train()
    total_loss = total_ce = total_spd = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    ce_loss = nn.CrossEntropyLoss()
    spd_loss = SPDLoss(num_classes)

    pbar = tqdm(loader, desc=f"Ep {epoch+1:3d} [train]", unit="batch", leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        sens = batch['skin_type']
        if sens.max() > 1:
            task_idx = (sens >= 3).long()
        else:
            task_idx = sens.long()

        idx0 = (task_idx == 0).nonzero(as_tuple=True)[0]
        idx1 = (task_idx == 1).nonzero(as_tuple=True)[0]

        loss0 = loss1 = torch.tensor(0.0, device=device)
        preds0 = preds1 = torch.tensor([], dtype=torch.long, device=device)

        if len(idx0) > 0:
            b0 = {k: v[idx0] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 0
            out0 = model(b0)
            loss0 = ce_loss(out0['logits'], b0['label'])
            preds0 = out0['logits'].argmax(dim=1)
            all_preds.append(preds0.cpu().numpy())
            all_labels.append(b0['label'].cpu().numpy())

        if len(idx1) > 0:
            b1 = {k: v[idx1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 1
            out1 = model(b1)
            loss1 = ce_loss(out1['logits'], b1['label'])
            preds1 = out1['logits'].argmax(dim=1)
            all_preds.append(preds1.cpu().numpy())
            all_labels.append(b1['label'].cpu().numpy())

        spd = spd_loss(preds0, preds1) if len(preds0)>0 and len(preds1)>0 else torch.tensor(0.0, device=device)
        loss = loss0 + loss1 + alpha * spd

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_ce += loss0.item() + loss1.item()
        total_spd += spd.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {"total": total_loss/n_batches, "ce": total_ce/n_batches, "spd": total_spd/n_batches, "acc": acc, "macro_f1": f1}

def validate(model, loader, device, num_classes, desc="Validation"):
    model.eval()
    all_preds, all_labels, all_sens = [], [], []
    all_losses = []
    ce_loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, unit="batch", leave=False)
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            sens = batch['skin_type']
            if sens.max() > 1:
                task_idx = (sens >= 3).long()
            else:
                task_idx = sens.long()
            all_logits, all_order = [], []
            for t in [0,1]:
                mask = (task_idx == t)
                if mask.sum() > 0:
                    bt = {k: v[mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    model.current_task_idx = t
                    out = model(bt)
                    all_logits.append(out['logits'])
                    all_order.append(mask.nonzero(as_tuple=True)[0].cpu())
            if not all_logits:
                continue
            logits = torch.cat(all_logits)
            order = torch.cat(all_order)
            _, inv = order.sort()
            logits = logits[inv]
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            all_sens.append(task_idx.cpu().numpy())
            all_losses.append(ce_loss(logits, batch['label']).item())
            pbar.set_postfix(loss=f"{np.mean(all_losses):.4f}")
    if not all_preds:
        return None
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_sens = np.concatenate(all_sens)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    return {
        'acc': acc, 'auroc': 0.0, 'macro_f1': macro_f1, 'weighted_f1': weighted_f1,
        'conf_mat': conf_mat, 'preds': all_preds, 'labels': all_labels, 'sens': all_sens,
        'preds_by_sens': {0: all_preds[all_sens==0], 1: all_preds[all_sens==1]},
        'labels_by_sens': {0: all_labels[all_sens==0], 1: all_labels[all_sens==1]},
        'loss': np.mean(all_losses)
    }

# ============================================================
# Configuration
# ============================================================
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR = WORK_ROOT / 'csvs'
IMAGE_ROOTS = {
    'hiba': Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'ham10000': Path('/kaggle/input/datasets/asosenge/ham10000/HAM10000'),
    'derm7pt': Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
}
CFG = {
    'csv_dir': CSV_DIR, 'image_roots': IMAGE_ROOTS,
    'ckpt_dir': WORK_ROOT / 'checkpoints_FairAdaBN_resnet18',
    'results_dir': WORK_ROOT / 'results_FairAdaBN_resnet18',
    'embed_dim': 512, 'num_classes': 5, 'num_skin_types': 6, 'num_groups': 2,
    'batch_size': 32, 'num_epochs': 1, 'lr': 1e-4, 'min_lr': 1e-6,
    'weight_decay': 1e-4, 'warmup_epochs': 1, 'alpha': 1.0,
}
CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

def main():
    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Instantiate base DualResNet18
    model = DualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=False,
    )
    # Replace all BatchNorm2d with FairAdaBN2d
    replace_bn(model, num_groups=CFG["num_groups"])
    # Add current_task_idx attribute (will be set before each forward)
    model.current_task_idx = None
    model = model.to(DEVICE)

    param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"], betas=(0.9,0.999), eps=1e-8)
    def lr_lambda(ep):
        if ep < CFG["warmup_epochs"]:
            return (ep+1)/CFG["warmup_epochs"]
        progress = (ep - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return max(CFG["min_lr"]/CFG["lr"], cos)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

    history = defaultdict(list)
    best_f1 = 0.0
    for epoch in range(CFG["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, scaler, DEVICE,
                                    CFG["alpha"], CFG["num_classes"])
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]
        for k, v in train_metrics.items():
            history[f"train_{k}"].append(v)
        if val_metrics:
            for k in ["acc", "macro_f1"]:
                history[f"val_{k}"].append(val_metrics[k])
        history["lr"].append(lr)
        print(f"Ep {epoch+1:3d} loss={train_metrics['total']:.4f} tr_acc={train_metrics['acc']:.4f} "
              f"val_acc={val_metrics['acc'] if val_metrics else 0.0:.4f} spd={train_metrics['spd']:.4f} lr={lr:.2e}")

        ckpt = {
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "history": dict(history)
        }
        torch.save(ckpt, CFG["ckpt_dir"] / "last_model.pt")
        if val_metrics and val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt", CFG["ckpt_dir"] / "best_model.pt")

    # Final evaluation
    best_ckpt = CFG["ckpt_dir"] / "best_model.pt"
    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"])
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Final Validation")
    if val_res:
        fair_metrics = fairness(val_res)
        save_results_csv(val_res, fair_metrics, "val", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(val_res["conf_mat"], LABEL_NAMES[:CFG["num_classes"]],
                              "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
        plot_per_class_metrics(val_res, LABEL_NAMES[:CFG["num_classes"]],
                               "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
        plot_fairness_metrics(fair_metrics, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")
    if test_loader:
        test_res = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        if test_res:
            fair_metrics = fairness(test_res)
            save_results_csv(test_res, fair_metrics, "test", CFG["results_dir"], LABEL_NAMES)
    plot_training_curves(history, "Training History (FairAdaBN)", CFG["results_dir"] / "training_curves.png")
    print(f"Done. Results saved to {CFG['results_dir']}")

if __name__ == "__main__":
    main()