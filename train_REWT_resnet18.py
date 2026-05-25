#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reweighting (REWT) for skin lesion classification.
Each sample gets weight = P_exp / P_obs to break correlation between skin type and condition.
Uses same DualResNet18 backbone as baseline.

All outputs saved in:
    - checkpoints_REWT_resnet18/
    - results_REWT_resnet18/
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import f1_score
from models.models_losses import DualResNet18, compute_class_weights, get_layer_wise_lr_params
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
    'padufes20':      Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),              # update path as needed
    'isic2019':       Path('/kaggle/input/datasets/sengenjih/isic2019'),                 # update path as needed
}

CFG = {
    'csv_dir':      CSV_DIR,
    'image_roots':  IMAGE_ROOTS,
    'ckpt_dir':     WORK_ROOT / 'checkpoints_REWT_resnet18',
    'results_dir':  WORK_ROOT / 'results_REWT_resnet18',

    'backbone': 'resnet18',
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 5,
    'num_skin_types': 6,

    'batch_size': 32,
    'num_epochs': 500,          # Update as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 100,        # Update as needed
    'aug_probability': 0.85,
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Helper: Compute group weights for REWT
# ------------------------------------------------------------
def compute_group_weights(dataset):
    """
    Compute w(s, y) = P_exp(s, y) / P_obs(s, y) for each (skin_type, label) group.
    Returns a dictionary mapping (s, y) -> weight.
    """
    total = len(dataset)
    joint_counts = defaultdict(int)
    skin_type_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for idx in range(total):
        sample = dataset[idx]
        s = sample['skin_type']
        y = sample['label']
        # Convert to scalar if tensor
        if torch.is_tensor(s):
            s = s.item()
        if torch.is_tensor(y):
            y = y.item()
        joint_counts[(s, y)] += 1
        skin_type_counts[s] += 1
        label_counts[y] += 1

    group_weight_map = {}
    for (s, y), cnt in joint_counts.items():
        p_obs = cnt / total
        p_exp = (skin_type_counts[s] / total) * (label_counts[y] / total)
        # Avoid division by zero (p_obs > 0)
        w = p_exp / p_obs
        group_weight_map[(s, y)] = w

    # Normalize weights to have mean 1
    w_vals = np.array(list(group_weight_map.values()))
    norm_factor = np.mean(w_vals)
    for k in group_weight_map:
        group_weight_map[k] /= norm_factor

    return group_weight_map


# ------------------------------------------------------------
# Training function with weighted cross‑entropy
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, epoch, scaler, device, group_weight_map):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    criterion = nn.CrossEntropyLoss(reduction='none')  # per‑sample loss

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        # Compute batch weights from group_weight_map
        batch_weights = []
        for s, y in zip(batch['skin_type'], batch['label']):
            s_val = s.item() if torch.is_tensor(s) else s
            y_val = y.item() if torch.is_tensor(y) else y
            batch_weights.append(group_weight_map[(s_val, y_val)])
        batch_weights = torch.tensor(batch_weights, device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)
            losses = criterion(out["logits"], batch["label"])
            loss = (losses * batch_weights).mean()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

        with torch.no_grad():
            preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()
    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {"total": avg_loss, "acc": acc, "macro_f1": macro_f1}


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

    # Build standard loaders (no resampling, just standard train loader)
    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    # Precompute group weights from the training dataset
    group_weight_map = compute_group_weights(train_loader.dataset)
    print(f"Computed weights for {len(group_weight_map)} (skin_type, label) groups")

    model = DualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=False,
    ).to(DEVICE)

    param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
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
    patience = 20
    patience_counter = 0
    history = defaultdict(list)

    for epoch in range(start_epoch, CFG["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, scaler, DEVICE, group_weight_map)
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]

        # ----- EARLY STOPPING (patience=20) -----
        # current_f1 = val_metrics["macro_f1"]
        # if current_f1 > best_f1:
        #     best_f1 = current_f1
        #     patience_counter = 0
        # else:
        #     patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs (no improvement in F1 for {patience} epochs).")
        #     break
        # -----------------------------------------

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
            history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))

        print(f"Ep {epoch+1:3d}/{CFG['num_epochs']}  "
              f"loss={train_metrics['total']:.4f}  tr_acc={train_metrics['acc']:.4f}  "
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

    # Evaluation (same as baseline)
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
    plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")

    if test_loader:
        test_res = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
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
        res = validate(model, loader, DEVICE, CFG["num_classes"], desc=f"Cross-eval: {ds_name}")
        fair = fairness(res)
        save_results_csv(res, fair, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              f"Confusion Matrix - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_confusion.png")
        plot_per_class_metrics(res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               f"Per-Class Metrics - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_per_class.png")
        plot_fairness_metrics(fair, f"Fairness - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_fairness.png")
        cross_results[ds_name] = {
            "accuracy": res["acc"],
            "precision": res["macro_prec"],
            "recall": res["macro_rec"],
            "auroc": res["auroc"],
            "macro_f1": res["macro_f1"],
            "micro_f1": res["micro_f1"],
            "weighted_f1": res["weighted_f1"],
            "EOM": fair["EOM"],
            "PQD": fair["PQD"],
            "DPM": fair["DPM"],
        }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    plot_training_curves(history, "Training History (REWT ResNet-18)",
                         CFG["results_dir"] / "training_curves.png")

    if test_loader:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
        embs = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        plot_tsne(embs, labels_tsne, "t-SNE - Test Set", CFG["results_dir"] / "tsne_test.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()