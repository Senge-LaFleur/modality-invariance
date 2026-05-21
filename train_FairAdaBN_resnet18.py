#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FairAdaBN Training Script for Dual-Encoder ResNet-18
Based on paper: "FairAdaBN: Adaptive Batch Normalization for Fair Medical Image Classification"
Replaces standard BatchNorm with Adaptive BatchNorm per sensitive attribute.
Uses L_CE + alpha * L_SD loss where L_SD is statistical disparity loss.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import random
import math
import json
import shutil
from pathlib import Path
from collections import defaultdict
import warnings

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from models.models_losses import DualResNet18, get_layer_wise_lr_params
from models.evaluation import (
    fairness,
    save_results_csv,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_fairness_metrics,
    plot_training_curves,
    plot_tsne,
    build_loaders,
    LABEL_NAMES,
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
        super(FairAdaptiveBatchNorm2d, self).__init__()
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
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_groups, num_features))
            self.register_buffer('running_var', torch.ones(num_groups, num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_var', None)

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
                break
        if task_idx is None:
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

    def extra_repr(self):
        return (f'{self.num_features}, num_groups={self.num_groups}, '
                f'eps={self.eps}, momentum={self.momentum}, affine={self.affine}, '
                f'track_running_stats={self.track_running_stats}')


def convert_module_to_fair_adanbn(module, num_groups=2, skip_names=None):
    """Recursively replace nn.BatchNorm2d with FairAdaptiveBatchNorm2d."""
    if skip_names is None:
        skip_names = []
    for name, child in module.named_children():
        if any(skip in name for skip in skip_names):
            continue
        if isinstance(child, nn.BatchNorm2d):
            fair_bn = FairAdaptiveBatchNorm2d(
                num_features=child.num_features,
                num_groups=num_groups,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats
            )
            # Initialize with default values
            if child.affine:
                fair_bn.weight.data.fill_(1.0)
                fair_bn.bias.data.fill_(0.0)
            if child.track_running_stats:
                fair_bn.running_mean.data.fill_(0.0)
                fair_bn.running_var.data.fill_(1.0)
            setattr(module, name, fair_bn)
        else:
            convert_module_to_fair_adanbn(child, num_groups, skip_names)


class FairDualResNet18Wrapper(nn.Module):
    """
    Wrapper around a standard DualResNet18 where all BatchNorm layers have been replaced
    with FairAdaptiveBatchNorm2d. This wrapper holds a `current_task_idx` attribute
    that is used by the adaptive BN layers during forward.
    """
    def __init__(self, base_model, num_groups=2):
        super().__init__()
        # Replace all BN layers in the base model
        convert_module_to_fair_adanbn(base_model, num_groups=num_groups, skip_names=[])
        self.base_model = base_model
        self.num_groups = num_groups
        self.current_task_idx = None

    def forward(self, batch):
        """Forward pass. Requires self.current_task_idx to be set beforehand."""
        if self.current_task_idx is None:
            raise RuntimeError("Must set model.current_task_idx before forward()")
        return self.base_model(batch)


# ------------------------------------------------------------
# Statistical Disparity Loss (L_SD) - Paper Eq.4
# ------------------------------------------------------------
class StatisticalDisparityLoss(nn.Module):
    """L_SD = sum_{y} || P(Ŷ=y|A=0) - P(Ŷ=y|A=1) ||^2"""
    def __init__(self, num_classes):
        super(StatisticalDisparityLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, preds_group0, preds_group1):
        if len(preds_group0) == 0 or len(preds_group1) == 0:
            return torch.tensor(0.0, device=preds_group0.device)
        n0 = preds_group0.shape[0]
        n1 = preds_group1.shape[0]
        p0 = torch.zeros(self.num_classes, device=preds_group0.device)
        p1 = torch.zeros(self.num_classes, device=preds_group1.device)
        p0.scatter_add_(0, preds_group0, torch.ones_like(preds_group0, dtype=torch.float))
        p1.scatter_add_(0, preds_group1, torch.ones_like(preds_group1, dtype=torch.float))
        p0 = p0 / n0
        p1 = p1 / n1
        return torch.sum((p0 - p1) ** 2)


# ------------------------------------------------------------
# Training function with FairAdaBN and L_SD
# ------------------------------------------------------------
def train_epoch_fair(model, loader, optimizer, epoch, scaler, device,
                     alpha=1.0, num_classes=5):
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_spd_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    
    criterion_ce = nn.CrossEntropyLoss()
    spd_loss_fn = StatisticalDisparityLoss(num_classes)
    
    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch", dynamic_ncols=True, leave=False)
    for batch in pbar:
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        
        # Binarize sensitive attribute (skin_type)
        sens = batch['skin_type']
        if sens.max() > 1:
            # Fitzpatrick: 0-2 -> light (0), 3-5 -> dark (1)
            task_idx = (sens >= 3).long()
        else:
            task_idx = sens.long()
        
        idx_0 = (task_idx == 0).nonzero(as_tuple=True)[0]
        idx_1 = (task_idx == 1).nonzero(as_tuple=True)[0]
        
        loss_ce_0 = torch.tensor(0.0, device=device)
        loss_ce_1 = torch.tensor(0.0, device=device)
        preds_0 = torch.tensor([], dtype=torch.long, device=device)
        preds_1 = torch.tensor([], dtype=torch.long, device=device)
        
        # Subgroup A=0 (unprivileged)
        if len(idx_0) > 0:
            batch_0 = {k: v[idx_0] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 0
            out_0 = model(batch_0)
            loss_ce_0 = criterion_ce(out_0['logits'], batch_0['label'])
            preds_0 = out_0['logits'].argmax(dim=1)
            all_preds.append(preds_0.cpu().numpy())
            all_labels.append(batch_0['label'].cpu().numpy())
        
        # Subgroup A=1 (privileged)
        if len(idx_1) > 0:
            batch_1 = {k: v[idx_1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 1
            out_1 = model(batch_1)
            loss_ce_1 = criterion_ce(out_1['logits'], batch_1['label'])
            preds_1 = out_1['logits'].argmax(dim=1)
            all_preds.append(preds_1.cpu().numpy())
            all_labels.append(batch_1['label'].cpu().numpy())
        
        # Statistical disparity loss
        if len(preds_0) > 0 and len(preds_1) > 0:
            loss_spd = spd_loss_fn(preds_0, preds_1)
        else:
            loss_spd = torch.tensor(0.0, device=device)
        
        loss = loss_ce_0 + loss_ce_1 + alpha * loss_spd
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_ce_loss += (loss_ce_0.item() + loss_ce_1.item())
        total_spd_loss += loss_spd.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")
    
    pbar.close()
    avg_loss = total_loss / max(n_batches, 1)
    avg_ce = total_ce_loss / max(n_batches, 1)
    avg_spd = total_spd_loss / max(n_batches, 1)
    
    if len(all_preds) > 0:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = (all_preds == all_labels).mean()
        macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    else:
        acc = 0.0
        macro_f1 = 0.0
    
    return {"total": avg_loss, "ce": avg_ce, "spd": avg_spd, "acc": acc, "macro_f1": macro_f1}


# ------------------------------------------------------------
# Validation function for FairAdaBN
# ------------------------------------------------------------
def validate_fair(model, loader, device, num_classes, desc="Validation"):
    model.eval()
    all_preds = []
    all_labels = []
    all_sens = []
    all_losses = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
        for batch in pbar:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device, non_blocking=True)
            
            sens = batch['skin_type']
            if sens.max() > 1:
                task_idx = (sens >= 3).long()
            else:
                task_idx = sens.long()
            
            # Process each subgroup separately
            all_logits = []
            all_batch_order = []
            for t in [0, 1]:
                mask = (task_idx == t)
                if mask.sum() > 0:
                    batch_t = {k: v[mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    model.current_task_idx = t
                    out_t = model(batch_t)
                    all_logits.append(out_t['logits'])
                    all_batch_order.append(mask.nonzero(as_tuple=True)[0].cpu())
            if len(all_logits) == 0:
                continue
            logits = torch.cat(all_logits, dim=0)
            # Reorder to original batch order
            batch_order = torch.cat(all_batch_order, dim=0)
            _, orig_order = batch_order.sort()
            logits = logits[orig_order]
            
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['label'].cpu().numpy())
            all_sens.append(task_idx.cpu().numpy())
            loss = criterion(logits, batch['label'])
            all_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(all_losses):.4f}")
    
    pbar.close()
    if len(all_preds) == 0:
        return None
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_sens = np.concatenate(all_sens)
    
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    
    # For AUROC we would need probabilities; skip for simplicity (or compute later)
    auroc = 0.0
    
    preds_by_sens = {0: all_preds[all_sens==0], 1: all_preds[all_sens==1]}
    labels_by_sens = {0: all_labels[all_sens==0], 1: all_labels[all_sens==1]}
    
    return {
        'acc': acc,
        'auroc': auroc,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'conf_mat': conf_mat,
        'preds': all_preds,
        'labels': all_labels,
        'sens': all_sens,
        'preds_by_sens': preds_by_sens,
        'labels_by_sens': labels_by_sens,
        'loss': np.mean(all_losses) if all_losses else 0.0
    }


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR = WORK_ROOT / 'csvs'

IMAGE_ROOTS = {
    'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'ham10000':       Path('/kaggle/input/datasets/asosenge/ham10000/HAM10000'),
    'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
}

CFG = {
    'csv_dir':      CSV_DIR,
    'image_roots':  IMAGE_ROOTS,
    'ckpt_dir':     WORK_ROOT / 'checkpoints_FairAdaBN_resnet18',
    'results_dir':  WORK_ROOT / 'results_FairAdaBN_resnet18',
    
    'backbone': 'resnet18',
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 5,
    'num_skin_types': 6,
    'num_groups': 2,
    
    'batch_size': 32,
    'num_epochs': 1,       # Update as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,
    'aug_probability': 0.85,
    'alpha': 1.0,          # weight for L_SD loss
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")
    print("Image roots:")
    for name, root in CFG['image_roots'].items():
        print(f"  {name:<15}: {root}")

    # Build data loaders (same as baseline)
    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    # Instantiate base model (standard DualResNet18)
    base_model = DualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=False,
    ).to(DEVICE)

    # Wrap with FairAdaBN conversion
    model = FairDualResNet18Wrapper(base_model, num_groups=CFG["num_groups"])
    model = model.to(DEVICE)

    # Layer-wise learning rate (apply to the base model's parameters)
    param_groups = get_layer_wise_lr_params(model.base_model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"], betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(epoch):
        if epoch < CFG["warmup_epochs"]:
            return (epoch + 1) / CFG["warmup_epochs"]
        progress = (epoch - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        min_frac = CFG["min_lr"] / CFG["lr"]
        return max(min_frac, cos)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 0
    best_auroc = 0.0
    best_f1 = 0.0
    history = defaultdict(list)

    for epoch in range(start_epoch, CFG["num_epochs"]):
        train_metrics = train_epoch_fair(
            model, train_loader, optimizer, epoch, scaler, DEVICE,
            alpha=CFG["alpha"], num_classes=CFG["num_classes"]
        )
        scheduler.step()
        
        val_metrics = validate_fair(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        
        lr = optimizer.param_groups[0]["lr"]

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        if val_metrics:
            for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
                history[f"val_{k}"].append(float(val_metrics[k]))
        else:
            for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
                history[f"val_{k}"].append(0.0)
        history["lr"].append(float(lr))

        val_acc_str = f"{val_metrics['acc']:.4f}" if val_metrics else "N/A"
        val_f1_str = f"{val_metrics['macro_f1']:.4f}" if val_metrics else "N/A"
        print(f"Ep {epoch+1:3d}/{CFG['num_epochs']}  "
              f"loss={train_metrics['total']:.4f}  tr_acc={train_metrics['acc']:.4f}  "
              f"val_acc={val_acc_str}  val_f1={val_f1_str}  "
              f"spd_loss={train_metrics['spd']:.4f}  lr={lr:.2e}")

        # Save checkpoints
        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": dict(history),
        }
        torch.save(ckpt_state, CFG["ckpt_dir"] / "last_model.pt")
        if val_metrics and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt", CFG["ckpt_dir"] / "best_auroc_model.pt")
        if val_metrics and val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt", CFG["ckpt_dir"] / "best_f1_model.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt_state, CFG["ckpt_dir"] / f"checkpoint_ep{epoch+1:03d}.pt")

        with open(CFG["results_dir"] / "history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    print(f"Training complete. Best AUROC: {best_auroc:.4f}, Best F1: {best_f1:.4f}")

    # Load best model
    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # Final evaluation on validation set
    val_res = validate_fair(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    if val_res:
        val_fair = fairness(val_res)
        save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
        plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
        plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")

    # Test set evaluation
    if test_loader:
        test_res = validate_fair(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        if test_res:
            test_fair = fairness(test_res)
            save_results_csv(test_res, test_fair, "test", CFG["results_dir"], LABEL_NAMES)
            plot_confusion_matrix(test_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                                  "Confusion Matrix - Test", CFG["results_dir"] / "test_confusion.png")
            plot_per_class_metrics(test_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                                   "Per-Class Metrics - Test", CFG["results_dir"] / "test_per_class.png")
            plot_fairness_metrics(test_fair, "Fairness - Test", CFG["results_dir"] / "test_fairness.png")

    # Cross-dataset evaluation
    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        print(f"\nEvaluating on {ds_name}")
        res = validate_fair(model, loader, DEVICE, CFG["num_classes"], desc=f"Cross-eval: {ds_name}")
        if res:
            fair_metrics = fairness(res)
            save_results_csv(res, fair_metrics, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
            plot_confusion_matrix(res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                                  f"Confusion Matrix - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_confusion.png")
            plot_per_class_metrics(res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                                   f"Per-Class Metrics - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_per_class.png")
            plot_fairness_metrics(fair_metrics, f"Fairness - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_fairness.png")
            cross_results[ds_name] = {
                "accuracy": res["acc"],
                "auroc": res["auroc"],
                "macro_f1": res["macro_f1"],
                "EOM": fair_metrics.get("EOM", 0),
                "PQD": fair_metrics.get("PQD", 0),
                "DPM": fair_metrics.get("DPM", 0),
            }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    # Training curves
    plot_training_curves(history, "Training History (FairAdaBN ResNet-18)",
                         CFG["results_dir"] / "training_curves.png")

    # t-SNE visualization on test set
    if test_loader:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                sens = batch['skin_type']
                if sens.max() > 1:
                    task_idx = (sens >= 3).long()
                else:
                    task_idx = sens.long()
                embs_batch = []
                for t in [0, 1]:
                    mask = (task_idx == t)
                    if mask.sum() > 0:
                        batch_t = {k: v[mask] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                        model.current_task_idx = t
                        out_t = model(batch_t)
                        embs_batch.append(out_t["z"].cpu().numpy())
                if embs_batch:
                    embs = np.concatenate(embs_batch, axis=0)
                    all_embs.append(embs)
                    all_labels_tsne.append(batch['label'].cpu().numpy())
        if all_embs:
            embs = np.concatenate(all_embs)
            labels_tsne = np.concatenate(all_labels_tsne)
            plot_tsne(embs, labels_tsne, "t-SNE - Test Set (FairAdaBN)", CFG["results_dir"] / "tsne_test.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()