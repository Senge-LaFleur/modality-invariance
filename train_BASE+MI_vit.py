#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline + Modality Invariance (MI) with Dual ViT backbone.
Uses:
- Weighted + label‑smoothed cross‑entropy loss for disease classification (L_c)
- Modality invariance loss (mi_loss) on paired clinical/derm embeddings

Supports three training regimes: clin, derm, both.

All outputs saved in:
    - checkpoints_BASE+MI_vit/{clin,derm,both}/
    - results_BASE+MI_vit/{clin,derm,both}/
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

from sklearn.metrics import f1_score
from models.models_losses import (
    DualViT,
    mi_loss,
    cls_loss_fn,
    compute_class_weights,
    get_layer_wise_lr_params,
)
from models.evaluation import (
    validate,
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
parser = argparse.ArgumentParser(description="Baseline+MI ViT — modality ablation (weighted CE)")
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
# PATH CONFIGURATION — update for each environment
# ============================================================
WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR   = WORK_ROOT / 'csvs'

IMAGE_ROOTS = {
    'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'ham10000':       Path('/kaggle/input/datasets/asosenge/ham10000/HAM10000'),
    'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
}

CFG = {
    'csv_dir':        CSV_DIR,
    'image_roots':    IMAGE_ROOTS,
    'ckpt_dir':       WORK_ROOT / f'checkpoints_BASE+MI_vit' / TRAIN_MODALITY,
    'results_dir':    WORK_ROOT / f'results_BASE+MI_vit'     / TRAIN_MODALITY,

    'train_modality': TRAIN_MODALITY,
    'backbone':       'vit_small',
    'embed_dim':      512,
    'img_size':       224,
    'num_classes':    5,
    'num_skin_types': 6,

    'batch_size':     32,
    'num_epochs':     1,          # adjust as needed
    'lr':             1e-4,
    'min_lr':         1e-6,
    'weight_decay':   1e-4,
    'warmup_epochs':  1,           # adjust as needed
    'aug_probability': 0.85,

    'mi_weight':      0.5,
    'label_smoothing': 0.1,
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

# ============================================================
# Training function with weighted CE and MI loss
# ============================================================
def train_epoch(model, loader, optimizer, epoch, scaler, device, mi_weight, weight_tensor, label_smoothing):
    model.train()
    total_loss = 0.0
    total_loss_c = 0.0
    total_loss_mi = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)

            # Weighted + label‑smoothed cross‑entropy
            loss_c = cls_loss_fn(out["logits"], batch["label"], weight_tensor, label_smoothing)

            # MI loss only for paired samples
            loss_mi = 0.0
            if "z_c" in out and "z_d" in out:
                loss_mi = mi_loss(out["z_c"], out["z_d"])

            loss = loss_c + mi_weight * loss_mi

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_loss_c += loss_c.item()
        total_loss_mi += loss_mi.item() if torch.is_tensor(loss_mi) else loss_mi
        n_batches += 1

        with torch.no_grad():
            preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()
    avg_loss = total_loss / max(n_batches, 1)
    avg_loss_c = total_loss_c / max(n_batches, 1)
    avg_loss_mi = total_loss_mi / max(n_batches, 1)

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "total": avg_loss,
        "loss_c": avg_loss_c,
        "loss_mi": avg_loss_mi,
        "acc": acc,
        "macro_f1": macro_f1,
    }

# ============================================================
# Evaluation helper — runs on a dict of test loaders
# ============================================================
def evaluate_test_loaders(model, test_loaders, device, cfg, results_dir, label_names, prefix="test"):
    summary = {}
    for mod_name, loader in test_loaders.items():
        split_tag = f"{prefix}_{mod_name}"
        print(f"\n── Evaluating {prefix} on {mod_name} images ──")
        res  = validate(model, loader, device, cfg["num_classes"],
                        desc=f"{prefix.capitalize()} [{mod_name}]")
        fair = fairness(res)
        save_results_csv(res, fair, split_tag, results_dir, label_names)
        plot_confusion_matrix(
            res["conf_mat"],
            [label_names[i] for i in range(cfg["num_classes"])],
            f"Confusion Matrix — {prefix.capitalize()} [{mod_name.upper()}]",
            results_dir / f"{split_tag}_confusion.png")
        plot_per_class_metrics(
            res,
            [label_names[i] for i in range(cfg["num_classes"])],
            f"Per-Class Metrics — {prefix.capitalize()} [{mod_name.upper()}]",
            results_dir / f"{split_tag}_per_class.png")
        plot_fairness_metrics(
            fair,
            f"Fairness — {prefix.capitalize()} [{mod_name.upper()}]",
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

# ============================================================
# Main
# ============================================================
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")

    # Load all data
    train_loader, val_loader, test_loaders, paired_test_loaders, eval_loaders = build_loaders(CFG, seed=SEED)

    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches  : {len(val_loader)}")
    print(f"Test loaders (unpaired) : {list(test_loaders.keys())}")
    print(f"Paired test loaders     : {list(paired_test_loaders.keys())}")
    print(f"Cross-eval loaders      : {list(eval_loaders.keys())}")

    # Compute class weights for balanced loss
    class_weights = compute_class_weights(CFG["csv_dir"], CFG["num_classes"])
    if class_weights:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
        print(f"Class weights: {np.round(class_weights, 3)}")
    else:
        weight_tensor = None

    model = DualViT(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=True,
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
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, scaler, DEVICE,
            mi_weight=CFG["mi_weight"],
            weight_tensor=weight_tensor,
            label_smoothing=CFG["label_smoothing"]
        )
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
            history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))

        print(f"Ep {epoch+1:3d}/{CFG['num_epochs']}  "
              f"total_loss={train_metrics['total']:.4f}  tr_acc={train_metrics['acc']:.4f}  "
              f"val_acc={val_metrics['acc']:.4f}  val_auroc={val_metrics['auroc']:.4f}  "
              f"val_f1={val_metrics['macro_f1']:.4f}  lr={lr:.2e}")

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": dict(history),
        }
        torch.save(ckpt_state, CFG["ckpt_dir"] / "last_model.pt")
        if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt", CFG["ckpt_dir"] / "best_auroc_model.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt", CFG["ckpt_dir"] / "best_f1_model.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt_state, CFG["ckpt_dir"] / f"checkpoint_ep{epoch+1:03d}.pt")

        with open(CFG["results_dir"] / "history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    print(f"\nTraining complete [{TRAIN_MODALITY}]. "
          f"Best AUROC: {best_auroc:.4f}  Best F1: {best_f1:.4f}")

    # Load best model
    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # Validation (final)
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
    plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")

    # ── Paired test set (HIBASkinLesions) ───────────────────────────────
    paired_summary = evaluate_test_loaders(
        model, paired_test_loaders, DEVICE, CFG, CFG["results_dir"], LABEL_NAMES, prefix="paired_test")
    print("\nPaired test set (HIBASkinLesions) summary:")
    for mod_name, metrics in paired_summary.items():
        print(f"  [paired_{mod_name}]  acc={metrics['accuracy']:.4f}  "
              f"auroc={metrics['auroc']:.4f}  f1={metrics['macro_f1']:.4f}  "
              f"EOM={metrics['EOM']:.4f}  PQD={metrics['PQD']:.4f}")
    if paired_summary:
        pd.DataFrame(paired_summary).T.to_csv(
            CFG["results_dir"] / "paired_test_modality_summary.csv")

    # ── Cross‑dataset evaluation (derm7pt) ────────────────────────────────
    cross_summary = evaluate_test_loaders(
        model, eval_loaders, DEVICE, CFG, CFG["results_dir"], LABEL_NAMES, prefix="cross")
    print("\nCross‑dataset (Derm7pt) summary:")
    for mod_name, metrics in cross_summary.items():
        print(f"  {mod_name}  acc={metrics['accuracy']:.4f}  "
              f"auroc={metrics['auroc']:.4f}  f1={metrics['macro_f1']:.4f}  "
              f"EOM={metrics['EOM']:.4f}  PQD={metrics['PQD']:.4f}")
    if cross_summary:
        pd.DataFrame(cross_summary).T.to_csv(
            CFG["results_dir"] / "cross_dataset_summary.csv")

    plot_training_curves(history, f"Training History (Baseline+MI ViT [{TRAIN_MODALITY}])",
                         CFG["results_dir"] / "training_curves.png")

    if 'clin' in test_loaders:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loaders['clin']:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
        embs = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        plot_tsne(embs, labels_tsne,
                  f"t-SNE — Test [clin] (Baseline+MI ViT, {TRAIN_MODALITY} trained)",
                  CFG["results_dir"] / "tsne_test_clin.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")

if __name__ == "__main__":
    main()