#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FairAdaBN Training Script for Dual-Encoder ResNet-18
Based on paper: "FairAdaBN: Adaptive Batch Normalization for Fair Medical Image Classification"
Replaces standard BatchNorm with Adaptive BatchNorm per sensitive attribute.
Uses L_CE + alpha * L_SD loss where L_SD is statistical disparity loss.

Now supports three training regimes: clin, derm, both.

All outputs saved in:
    - checkpoints_FairAdaBN_resnet18/{clin,derm,both}/
    - results_FairAdaBN_resnet18/{clin,derm,both}/
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import argparse
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

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="FairAdaBN ResNet-18 — modality ablation")
parser.add_argument(
    '--train_modality',
    choices=['clin', 'derm', 'both'],
    default='both',
    help="Training regime: 'clin' (clinical only), 'derm' (derm only), 'both' (dual encoder)."
)
args = parser.parse_args()
TRAIN_MODALITY = args.train_modality

# ============================================================
# Seeds
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device        : {DEVICE}")
print(f"Train modality: {TRAIN_MODALITY}")

# ============================================================
# PATH CONFIGURATION
# ============================================================
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
    'ckpt_dir':     WORK_ROOT / f'checkpoints_FairAdaBN_resnet18' / TRAIN_MODALITY,
    'results_dir':  WORK_ROOT / f'results_FairAdaBN_resnet18'     / TRAIN_MODALITY,

    'train_modality': TRAIN_MODALITY,
    'backbone': 'resnet18',
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 5,
    'num_skin_types': 6,
    'num_groups': 2,          # light vs dark skin

    'batch_size': 32,
    'num_epochs': 1,        # adjust as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,      # adjust as needed
    'aug_probability': 0.85,
    'alpha': 1.0,             # weight for L_SD loss
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Fair Adaptive BatchNorm Layer (Paper Sec 3)
# ------------------------------------------------------------
class FairAdaptiveBatchNorm2d(nn.Module):
    """
    Adaptive BatchNorm that maintains separate affine parameters (gamma, beta)
    and running statistics for each sensitive subgroup.
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
        self._root_ref = None

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

    def set_root(self, root):
        import weakref
        self._root_ref = weakref.ref(root)

    def forward(self, x):
        if self._root_ref is not None:
            root = self._root_ref()
            if root is not None and hasattr(root, 'current_task_idx'):
                task_idx = root.current_task_idx
            else:
                raise RuntimeError("FairAdaptiveBatchNorm2d: root model missing current_task_idx")
        else:
            raise RuntimeError("FairAdaptiveBatchNorm2d: root not set")

        if task_idx < 0 or task_idx >= self.num_groups:
            raise ValueError(f"task_idx={task_idx} out of range [0, {self.num_groups-1}]")

        weight = self.weight[task_idx] if self.affine else None
        bias   = self.bias[task_idx]   if self.affine else None
        running_mean = self.running_mean[task_idx] if self.track_running_stats else None
        running_var  = self.running_var[task_idx]  if self.track_running_stats else None

        return F.batch_norm(
            x, running_mean, running_var, weight, bias,
            self.training, self.momentum, self.eps
        )

    def extra_repr(self):
        return (f'{self.num_features}, num_groups={self.num_groups}, '
                f'eps={self.eps}, momentum={self.momentum}, affine={self.affine}, '
                f'track_running_stats={self.track_running_stats}')


def convert_module_to_fair_adanbn(module, num_groups=2, skip_names=None):
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
            if child.affine:
                fair_bn.weight.data.fill_(1.0)
                fair_bn.bias.data.fill_(0.0)
            if child.track_running_stats:
                fair_bn.running_mean.data.fill_(0.0)
                fair_bn.running_var.data.fill_(1.0)
            setattr(module, name, fair_bn)
        else:
            convert_module_to_fair_adanbn(child, num_groups, skip_names)


class FairDualResNet18(DualResNet18):
    """
    Extension of DualResNet18 where all BatchNorm layers are replaced with FairAdaptiveBatchNorm2d.
    Supports train_modality regimes.
    """
    def __init__(self, embed_dim=512, num_classes=5, num_skin_types=6,
                 pretrained=True, use_projection=False, num_groups=2,
                 train_modality='both'):
        super(FairDualResNet18, self).__init__(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_skin_types=num_skin_types,
            pretrained=pretrained,
            use_projection=use_projection,
            train_modality=train_modality   # pass to parent for weight-tying/freezing
        )
        self.num_groups = num_groups
        self.current_task_idx = None

        # Replace all BatchNorm2d in the entire model (both backbones)
        _SKIP = ['classifier', 'projection', 'fc', 'skin_clf']
        for name, child in self.named_children():
            if any(skip in name for skip in _SKIP):
                continue
            # Check if child contains any BN
            has_bn = any(isinstance(m, nn.BatchNorm2d) for m in child.modules())
            if has_bn:
                convert_module_to_fair_adanbn(child, num_groups=num_groups, skip_names=[])

        # Wire each BN layer back to this root
        self._wire_bn_roots()

    def _wire_bn_roots(self):
        for m in self.modules():
            if isinstance(m, FairAdaptiveBatchNorm2d):
                m.set_root(self)

    def forward(self, batch):
        if self.current_task_idx is None:
            raise RuntimeError("Must set model.current_task_idx before forward()")
        return super(FairDualResNet18, self).forward(batch)


# ------------------------------------------------------------
# Statistical Disparity Loss (L_SD) - Paper Eq.4
# ------------------------------------------------------------
class StatisticalDisparityLoss(nn.Module):
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
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Binarize sensitive attribute (Fitzpatrick: 0-2 light (0), 3-5 dark (1))
        sens = batch['skin_type']
        if sens.max() > 1:
            task_idx = (sens >= 3).long()
        else:
            task_idx = sens.long()

        idx_0 = (task_idx == 0).nonzero(as_tuple=True)[0]
        idx_1 = (task_idx == 1).nonzero(as_tuple=True)[0]

        loss_ce_0 = torch.tensor(0.0, device=device)
        loss_ce_1 = torch.tensor(0.0, device=device)
        preds_0 = torch.tensor([], dtype=torch.long, device=device)
        preds_1 = torch.tensor([], dtype=torch.long, device=device)

        if len(idx_0) > 0:
            batch_0 = {k: v[idx_0] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 0
            out_0 = model(batch_0)
            loss_ce_0 = criterion_ce(out_0['logits'], batch_0['label'])
            preds_0 = out_0['logits'].argmax(dim=1)
            all_preds.append(preds_0.cpu().numpy())
            all_labels.append(batch_0['label'].cpu().numpy())

        if len(idx_1) > 0:
            batch_1 = {k: v[idx_1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model.current_task_idx = 1
            out_1 = model(batch_1)
            loss_ce_1 = criterion_ce(out_1['logits'], batch_1['label'])
            preds_1 = out_1['logits'].argmax(dim=1)
            all_preds.append(preds_1.cpu().numpy())
            all_labels.append(batch_1['label'].cpu().numpy())

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
# Validation function for FairAdaBN (handles multi‑group forward)
# ------------------------------------------------------------
def validate_fair(model, loader, device, num_classes, desc="Validation"):
    model.eval()
    all_preds  = []
    all_probs  = []
    all_labels = []
    all_skins  = []
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

            idx_list   = []
            logit_list = []
            for t in [0, 1]:
                mask = (task_idx == t)
                if mask.sum() == 0:
                    continue
                orig_idx = mask.nonzero(as_tuple=True)[0]
                batch_t  = {k: v[orig_idx] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                model.current_task_idx = t
                out_t = model(batch_t)
                idx_list.append(orig_idx.cpu())
                logit_list.append(out_t['logits'].cpu())

            if len(logit_list) == 0:
                continue

            gathered_idx = torch.cat(idx_list, dim=0)
            restore = gathered_idx.argsort()
            logits = torch.cat(logit_list, dim=0)[restore].to(device)

            proc_labels = batch['label'][gathered_idx[restore]]
            proc_skins  = batch['skin_type'][gathered_idx[restore]]

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(proc_labels.cpu().numpy())
            all_skins.append(proc_skins.cpu().numpy())
            loss = criterion(logits, proc_labels)
            all_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(all_losses):.4f}")

    pbar.close()
    if len(all_preds) == 0:
        return None

    from sklearn.metrics import precision_score as _prec, recall_score as _rec
    from models.evaluation import robust_macro_auroc

    all_preds  = np.concatenate(all_preds)
    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_skins  = np.concatenate(all_skins)

    acc         = accuracy_score(all_labels, all_preds)
    macro_f1    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    micro_f1    = f1_score(all_labels, all_preds, average='micro',    zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    macro_prec  = _prec(all_labels, all_preds, average='macro',  zero_division=0)
    macro_rec   = _rec( all_labels, all_preds, average='macro',  zero_division=0)
    per_class_prec = _prec(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_rec  = _rec( all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    per_class_f1   = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=list(range(num_classes)))
    conf_mat = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
    auroc = robust_macro_auroc(all_probs, all_labels)
    if np.isnan(auroc):
        auroc = 0.0

    return {
        'acc': acc, 'auroc': auroc,
        'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1,
        'macro_prec': macro_prec, 'macro_rec': macro_rec,
        'per_class_prec': per_class_prec, 'per_class_rec': per_class_rec, 'per_class_f1': per_class_f1,
        'conf_mat': conf_mat,
        'preds': all_preds, 'probs': all_probs, 'labels': all_labels, 'skin': all_skins,
        'loss': np.mean(all_losses) if all_losses else 0.0,
    }


# ------------------------------------------------------------
# Evaluation helper for test loaders
# ------------------------------------------------------------
def evaluate_test_loaders(model, test_loaders, device, cfg, results_dir, label_names):
    summary = {}
    for mod_name, loader in test_loaders.items():
        split_tag = f"test_{mod_name}"
        print(f"\n── Test on {mod_name} images ──")
        res = validate_fair(model, loader, device, cfg["num_classes"], desc=f"Test [{mod_name}]")
        if res is None:
            continue
        fair = fairness(res)
        save_results_csv(res, fair, split_tag, results_dir, label_names)
        plot_confusion_matrix(
            res["conf_mat"],
            [label_names[i] for i in range(cfg["num_classes"])],
            f"Confusion Matrix — Test [{mod_name.upper()}]",
            results_dir / f"{split_tag}_confusion.png")
        plot_per_class_metrics(
            res,
            [label_names[i] for i in range(cfg["num_classes"])],
            f"Per-Class Metrics — Test [{mod_name.upper()}]",
            results_dir / f"{split_tag}_per_class.png")
        plot_fairness_metrics(
            fair,
            f"Fairness — Test [{mod_name.upper()}]",
            results_dir / f"{split_tag}_fairness.png")
        summary[mod_name] = {
            "accuracy":   res["acc"],
            "auroc":      res["auroc"],
            "macro_f1":   res["macro_f1"],
            "EOM":        fair["EOM"],
            "PQD":        fair["PQD"],
            "DPM":        fair["DPM"],
        }
    return summary


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")

    train_loader, val_loader, test_loaders, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches  : {len(val_loader)}")
    print(f"Test loaders : {list(test_loaders.keys())}")
    print(f"Cross-eval   : {list(eval_loaders.keys())}")

    model = FairDualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=False,
        num_groups=CFG["num_groups"],
        train_modality=TRAIN_MODALITY,
    ).to(DEVICE)

    param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"],
                                  betas=(0.9, 0.999), eps=1e-8)

    def lr_lambda(epoch):
        if epoch < CFG["warmup_epochs"]:
            return (epoch + 1) / CFG["warmup_epochs"]
        progress = (epoch - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        min_frac = CFG["min_lr"] / CFG["lr"]
        return max(min_frac, cos)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auroc = 0.0
    best_f1 = 0.0
    history = defaultdict(list)

    for epoch in range(CFG["num_epochs"]):
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

    print(f"\nTraining complete [{TRAIN_MODALITY}]. "
          f"Best AUROC: {best_auroc:.4f}  Best F1: {best_f1:.4f}")

    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # Validation final
    val_res = validate_fair(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    if val_res:
        val_fair = fairness(val_res)
        save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
        plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
        plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")

    # Test on both modalities
    test_summary = evaluate_test_loaders(
        model, test_loaders, DEVICE, CFG, CFG["results_dir"], LABEL_NAMES)
    print("\nTest summary:")
    for mod_name, metrics in test_summary.items():
        print(f"  [{mod_name}]  acc={metrics['accuracy']:.4f}  "
              f"auroc={metrics['auroc']:.4f}  f1={metrics['macro_f1']:.4f}  "
              f"EOM={metrics['EOM']:.4f}  PQD={metrics['PQD']:.4f}")
    if test_summary:
        pd.DataFrame(test_summary).T.to_csv(CFG["results_dir"] / "test_modality_summary.csv")

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
                "accuracy": res["acc"], "precision": res["macro_prec"], "recall": res["macro_rec"],
                "auroc": res["auroc"], "macro_f1": res["macro_f1"], "micro_f1": res["micro_f1"],
                "weighted_f1": res["weighted_f1"],
                "EOM": fair_metrics.get("EOM", 0), "PQD": fair_metrics.get("PQD", 0), "DPM": fair_metrics.get("DPM", 0),
            }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    plot_training_curves(history, f"Training History (FairAdaBN ResNet-18 [{TRAIN_MODALITY}])",
                         CFG["results_dir"] / "training_curves.png")

    # t‑SNE on clin test
    if 'clin' in test_loaders:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loaders['clin']:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                sens = batch['skin_type']
                if sens.max() > 1:
                    task_idx = (sens >= 3).long()
                else:
                    task_idx = sens.long()
                idx_list, emb_list = [], []
                for t in [0, 1]:
                    mask = (task_idx == t)
                    if mask.sum() == 0:
                        continue
                    orig_idx = mask.nonzero(as_tuple=True)[0]
                    batch_t = {k: v[orig_idx] for k, v in batch.items() if isinstance(v, torch.Tensor)}
                    model.current_task_idx = t
                    out_t = model(batch_t)
                    idx_list.append(orig_idx.cpu())
                    emb_list.append(out_t["z"].cpu())
                if emb_list:
                    gathered_idx = torch.cat(idx_list, dim=0)
                    restore = gathered_idx.argsort()
                    embs_ordered = torch.cat(emb_list, dim=0)[restore].numpy()
                    all_embs.append(embs_ordered)
                    all_labels_tsne.append(batch['label'][gathered_idx[restore]].cpu().numpy())
        if all_embs:
            embs = np.concatenate(all_embs)
            labels_tsne = np.concatenate(all_labels_tsne)
            plot_tsne(embs, labels_tsne,
                      f"t-SNE — Test [clin] (FairAdaBN ResNet-18, {TRAIN_MODALITY} trained)",
                      CFG["results_dir"] / "tsne_test_clin.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")

if __name__ == "__main__":
    main()