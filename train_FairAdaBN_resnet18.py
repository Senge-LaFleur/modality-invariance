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

# ----------------------------------------------------------------------
# Fair Adaptive BatchNorm (no class name conflict)
# ----------------------------------------------------------------------
class FairAdaBN2d(nn.Module):
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

    def forward(self, x):
        # Find model.current_task_idx from the root module
        module = self
        task_idx = None
        while module is not None:
            if hasattr(module, 'current_task_idx'):
                task_idx = module.current_task_idx
                break
            module = getattr(module, '_parent', None)
        if task_idx is None:
            raise RuntimeError("FairAdaBN2d: model.current_task_idx not set")
        if self.affine:
            w = self.weight[task_idx]
            b = self.bias[task_idx]
        else:
            w, b = None, None
        if self.track_running_stats:
            rm = self.running_mean[task_idx]
            rv = self.running_var[task_idx]
        else:
            rm, rv = None, None
        return F.batch_norm(x, rm, rv, w, b, self.training, self.momentum, self.eps)


def replace_bn_with_fair_ada(model, num_groups=2):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, FairAdaBN2d(
                child.num_features, num_groups, child.eps, child.momentum,
                child.affine, child.track_running_stats
            ))
        else:
            replace_bn_with_fair_ada(child, num_groups)

# ----------------------------------------------------------------------
# Statistical Disparity Loss
# ----------------------------------------------------------------------
class StatisticalDisparityLoss(nn.Module):
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
        return torch.sum((p0 - p1)**2)

# ----------------------------------------------------------------------
# Training function
# ----------------------------------------------------------------------
def train_epoch(model, loader, optimizer, epoch, scaler, device, alpha, num_classes):
    model.train()
    total_loss = total_ce = total_spd = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    ce_loss = nn.CrossEntropyLoss()
    spd_loss = StatisticalDisparityLoss(num_classes)

    pbar = tqdm(loader, desc=f"Ep {epoch+1:3d} [train]", unit="batch", leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        sens = batch['skin_type']
        task_idx = (sens >= 3).long() if sens.max() > 1 else sens.long()
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
        total_ce += (loss0.item() + loss1.item())
        total_spd += spd.item()
        n_batches += 1
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    avg_loss = total_loss / n_batches
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {"total": avg_loss, "ce": total_ce/n_batches, "spd": total_spd/n_batches, "acc": acc, "macro_f1": f1}

# ----------------------------------------------------------------------
# Validation function
# ----------------------------------------------------------------------
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
            task_idx = (sens >= 3).long() if sens.max() > 1 else sens.long()
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

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
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

    # Build base model, replace BN, and attach current_task_idx
    base_model = DualResNet18(embed_dim=CFG["embed_dim"], num_classes=CFG["num_classes"],
                              num_skin_types=CFG["num_skin_types"], pretrained=True, use_projection=False)
    replace_bn_with_fair_ada(base_model, num_groups=CFG["num_groups"])
    base_model.current_task_idx = None   # will be set before each forward
    base_model = base_model.to(DEVICE)

    param_groups = get_layer_wise_lr_params(base_model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"], betas=(0.9,0.999), eps=1e-8)
    def lr_lambda(ep):
        if ep < CFG["warmup_epochs"]: return (ep+1)/CFG["warmup_epochs"]
        progress = (ep - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        return max(CFG["min_lr"]/CFG["lr"], 0.5*(1+math.cos(math.pi*progress)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type=="cuda"))

    history = defaultdict(list)
    best_f1 = 0.0
    for epoch in range(CFG["num_epochs"]):
        train_metrics = train_epoch(base_model, train_loader, optimizer, epoch, scaler, DEVICE,
                                    CFG["alpha"], CFG["num_classes"])
        scheduler.step()
        val_metrics = validate(base_model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]
        for k,v in train_metrics.items(): history[f"train_{k}"].append(v)
        if val_metrics:
            for k in ["acc","macro_f1"]: history[f"val_{k}"].append(val_metrics[k])
        history["lr"].append(lr)
        print(f"Ep {epoch+1:3d} loss={train_metrics['total']:.4f} tr_acc={train_metrics['acc']:.4f} "
              f"val_acc={val_metrics['acc']:.4f if val_metrics else 'N/A'} spd={train_metrics['spd']:.4f} lr={lr:.2e}")

        torch.save({"epoch":epoch, "model":base_model.state_dict(), "optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(), "history":dict(history)},
                   CFG["ckpt_dir"]/"last_model.pt")
        if val_metrics and val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"]/"last_model.pt", CFG["ckpt_dir"]/"best_model.pt")

    # Final evaluation
    if (CFG["ckpt_dir"]/"best_model.pt").exists():
        ckpt = torch.load(CFG["ckpt_dir"]/"best_model.pt", map_location=DEVICE, weights_only=False)
        base_model.load_state_dict(ckpt["model"])
    val_res = validate(base_model, val_loader, DEVICE, CFG["num_classes"], desc="Final Validation")
    if val_res:
        fair_metrics = fairness(val_res)
        save_results_csv(val_res, fair_metrics, "val", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(val_res["conf_mat"], LABEL_NAMES[:CFG["num_classes"]], "Confusion Matrix",
                              CFG["results_dir"]/"val_confusion.png")
        plot_per_class_metrics(val_res, LABEL_NAMES[:CFG["num_classes"]], "Per-Class Metrics",
                               CFG["results_dir"]/"val_per_class.png")
        plot_fairness_metrics(fair_metrics, "Fairness", CFG["results_dir"]/"val_fairness.png")
    if test_loader:
        test_res = validate(base_model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        if test_res:
            fair_metrics = fairness(test_res)
            save_results_csv(test_res, fair_metrics, "test", CFG["results_dir"], LABEL_NAMES)
    plot_training_curves(history, "Training Curves", CFG["results_dir"]/"training_curves.png")
    print(f"Done. Results in {CFG['results_dir']}")

if __name__ == "__main__":
    main()