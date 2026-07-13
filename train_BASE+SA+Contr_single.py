#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_BASE+SA+Contr_single.py

Ablation study: BASE + SA (skin-attribute confusion/skin-type loss) +
Contr (supervised contrastive), SINGLE-STREAM encoder -- the domain-shift
ablation. Single-modality training, cross-modality + cross-dataset testing.

Trains ONE encoder (resnet18 or vit_small) on ONE modality (clinical or
derm), then evaluates the SAME trained weights on:
  - held-out same-modality test set        (in-distribution)
  - held-out other-modality test set        (the domain-shift condition)
  - existing cross-dataset eval sets (padufes20 / isic2019 / fitzpatrick17k)

Losses used: L_cls + L_conf + L_s + L_con, where L_con is the PLAIN
SupConLoss on out["z"] (single embedding per sample) -- NOT
cross_modal_supcon_loss, which requires a z_c/z_d split that this
single-stream model doesn't produce. L_MI is not computed: it requires
paired same-lesion clinical+derm samples, which by construction don't
exist in a single-modality training set.

All outputs saved in:
    - checkpoints_domain_shift_<modality>_<backbone>/
    - results_domain_shift_<modality>_<backbone>/

Usage:
    python "train_BASE+SA+Contr_single.py" <WORK_DIR> <WORK_ROOT> --train_modality clinical --backbone vit_small
    python "train_BASE+SA+Contr_single.py" <WORK_DIR> <WORK_ROOT> --train_modality derm     --backbone vit_small
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

from sklearn.metrics import f1_score

from models.models_losses import (
    SingleStreamModel,
    SupConLoss,
    confusion_loss,
    skin_type_loss,
    get_layer_wise_lr_params,
    cls_loss_fn,
    compute_class_weights,
)
from models.evaluation import (
    validate,
    fairness,
    fairness_binary,
    save_results_csv,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_fairness_metrics,
    plot_training_curves,
    plot_tsne,
    compute_knn_accuracy,
    build_domain_shift_loaders,
    LABEL_NAMES,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# CLI / Configuration
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("work_dir")
parser.add_argument("work_root")
parser.add_argument("--train_modality", choices=["clinical", "derm"], required=True)
parser.add_argument("--backbone", choices=["resnet18", "vit_small"], default="vit_small")
args = parser.parse_args()

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

WORK_ROOT = Path(args.work_root)
WORK_DIR = args.work_dir
CSV_DIR = WORK_ROOT / 'csvs'
TRAIN_MODALITY = args.train_modality
OTHER_MODALITY = "derm" if TRAIN_MODALITY == "clinical" else "clinical"
BACKBONE = args.backbone

IMAGE_ROOTS = {
    'hiba':           Path(WORK_DIR + '/data/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'fitzpatrick17k': Path(WORK_DIR + '/data/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'derm7pt':        Path(WORK_DIR + '/data/datasets/asosenge/derm7pt/release_v0/images'),
    'padufes20':      Path(WORK_DIR + '/data/datasets/mahdavi1202/skin-cancer'),
    'isic2019':       Path(WORK_DIR + '/data/datasets/sengenjih/isic2019'),
}

# WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
# CSV_DIR = WORK_ROOT / 'csvs'

# IMAGE_ROOTS = {
#     'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
#     'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
#     'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
#     'padufes20':      Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),              # update path as needed
#     'isic2019':       Path('/kaggle/input/datasets/sengenjih/isic2019'),
# }

RUN_TAG = f"domain_shift_{TRAIN_MODALITY}_{BACKBONE}"

CFG = {
    'csv_dir': CSV_DIR,
    'image_roots': IMAGE_ROOTS,
    'ckpt_dir': WORK_ROOT / f'checkpoints_{RUN_TAG}',
    'results_dir': WORK_ROOT / f'results_{RUN_TAG}',

    'backbone': BACKBONE,
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 3,
    'num_skin_types': 6,

    'batch_size': 32,
    'num_epochs': 100,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 10,

    'lambda_cls': 1.0,
    'lambda_conf': 0.3,
    'lambda_skin': 0.2,
    'lambda_con': 0.5,
    'temperature': 0.1,
    'label_smoothing': 0.01,
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

supcon_criterion = SupConLoss(temperature=CFG["temperature"])


# ------------------------------------------------------------
# Training function (single-stream, single-modality)
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, cfg, epoch, scaler, device, weight_tensor):
    model.train()
    total_loss = total_loss_c = total_loss_conf = total_loss_s = total_loss_con = 0.0
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

            loss_c = cls_loss_fn(out["logits"], batch["label"],
                                  weight_tensor=weight_tensor,
                                  smoothing=cfg["label_smoothing"])
            loss_conf = confusion_loss(out["skin_logits"])
            loss_s = skin_type_loss(out["skin_logits"].detach(), batch["skin_type"])
            # Plain SupCon on the single-stream embedding -- NOT
            # cross_modal_supcon_loss (that needs a z_c/z_d split, which
            # this single-encoder model never produces).
            loss_con = supcon_criterion(out["z"], batch["label"])

            loss = (cfg["lambda_cls"] * loss_c + cfg["lambda_conf"] * loss_conf
                    + cfg["lambda_skin"] * loss_s + cfg["lambda_con"] * loss_con)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_loss_c += loss_c.item()
        total_loss_conf += loss_conf.item()
        total_loss_s += loss_s.item()
        total_loss_con += loss_con.item()
        n_batches += 1

        with torch.no_grad():
            preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}")

    pbar.close()
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "total": total_loss / max(n_batches, 1),
        "loss_c": total_loss_c / max(n_batches, 1),
        "loss_conf": total_loss_conf / max(n_batches, 1),
        "loss_s": total_loss_s / max(n_batches, 1),
        "loss_con": total_loss_con / max(n_batches, 1),
        "acc": acc,
        "macro_f1": macro_f1,
    }


def _eval_and_report(model, loader, split_name, results_dir):
    """Run validate() + fairness + save + plot for one loader. Returns the res dict (or None)."""
    if loader is None:
        print(f"[SKIP] {split_name}: no data")
        return None
    res = validate(model, loader, DEVICE, CFG["num_classes"], desc=split_name)
    fair = fairness(res)
    fair_binary = fairness_binary(res)
    save_results_csv(res, fair, split_name, results_dir, LABEL_NAMES, fair_binary=fair_binary)
    plot_confusion_matrix(res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           f"Confusion Matrix - {split_name}", results_dir / f"{split_name}_confusion.png")
    plot_per_class_metrics(res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                            f"Per-Class Metrics - {split_name}", results_dir / f"{split_name}_per_class.png")
    plot_fairness_metrics(fair, f"Fairness - {split_name}", results_dir / f"{split_name}_fairness.png")
    print(f"\n[{split_name}] acc={res['acc']:.4f}  auroc={res['auroc']:.4f}  "
          f"macro_f1={res['macro_f1']:.4f}")
    print(f"  Acc_light={fair_binary['Acc_light']:.4f}  Acc_dark={fair_binary['Acc_dark']:.4f}  "
          f"Acc_gap={fair_binary['Acc_gap']:.4f}  DP_diff={fair_binary['DP_diff']:.4f}  "
          f"EOpp0={fair_binary['EOpp0']:.4f}  EOpp1={fair_binary['EOpp1']:.4f}  EOdd={fair_binary['EOdd']:.4f}")
    return res


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"Run tag       : {RUN_TAG}")
    print(f"Train modality: {TRAIN_MODALITY}  (cross-eval modality: {OTHER_MODALITY})")
    print(f"Backbone      : {BACKBONE}")
    print(f"CSV dir       : {CFG['csv_dir']}")
    print(f"Checkpoints   : {CFG['ckpt_dir']}")
    print(f"Results       : {CFG['results_dir']}")

    class_weights = compute_class_weights(CFG["csv_dir"], CFG["num_classes"])
    weight_tensor = (torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)
                      if class_weights else None)
    print(f"Class weights: {np.round(class_weights, 3) if class_weights else 'uniform'}")

    (train_loader, val_loader, test_loader_same, test_loader_cross,
     eval_loaders) = build_domain_shift_loaders(CFG, TRAIN_MODALITY, seed=SEED)
    print(f"Train batches: {len(train_loader)}"
          + (f", Val batches: {len(val_loader)}" if val_loader else ""))
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    model = SingleStreamModel(
        backbone=BACKBONE,
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=True,
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

    best_auroc, best_f1 = 0.0, 0.0
    history = defaultdict(list)

    for epoch in range(CFG["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, CFG, epoch, scaler, DEVICE, weight_tensor)
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation") if val_loader else None
        lr = optimizer.param_groups[0]["lr"]

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        if val_metrics:
            for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
                history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))

        msg = (f"Ep {epoch+1:3d}/{CFG['num_epochs']}  total_loss={train_metrics['total']:.4f}  "
               f"tr_acc={train_metrics['acc']:.4f}")
        if val_metrics:
            msg += (f"  val_acc={val_metrics['acc']:.4f}  val_auroc={val_metrics['auroc']:.4f}  "
                    f"val_f1={val_metrics['macro_f1']:.4f}")
        print(msg + f"  lr={lr:.2e}")

        ckpt_state = {"epoch": epoch, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "history": dict(history)}
        torch.save(ckpt_state, CFG["ckpt_dir"] / "last_model.pt")
        if val_metrics and not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
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

    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # --- Evaluation: same-modality test, cross-modality test, cross-dataset ---
    if val_loader:
        _eval_and_report(model, val_loader, "val", CFG["results_dir"])

    same_res = _eval_and_report(model, test_loader_same, f"test_{TRAIN_MODALITY}_same", CFG["results_dir"])
    cross_res = _eval_and_report(model, test_loader_cross, f"test_{OTHER_MODALITY}_shifted", CFG["results_dir"])

    # KNN + t-SNE on the same-modality test embeddings
    if test_loader_same is not None:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loader_same:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
        embs = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        knn_acc = compute_knn_accuracy(embs, labels_tsne, k=5)
        print(f"\n[{RUN_TAG}] Same-modality test KNN (k=5) accuracy: {knn_acc:.4f}")
        plot_tsne(embs, labels_tsne, f"t-SNE - Test ({RUN_TAG})",
                  CFG["results_dir"] / "tsne_test_same.png")

    # Domain-shift gap, front and center
    if same_res is not None and cross_res is not None:
        gap = same_res["acc"] - cross_res["acc"]
        auroc_gap = same_res["auroc"] - cross_res["auroc"]
        print(f"\n=== DOMAIN SHIFT GAP ({TRAIN_MODALITY} -> {OTHER_MODALITY}) ===")
        print(f"  Acc  same={same_res['acc']:.4f}  shifted={cross_res['acc']:.4f}  gap={gap:.4f}")
        print(f"  AUROC same={same_res['auroc']:.4f}  shifted={cross_res['auroc']:.4f}  gap={auroc_gap:.4f}")
        with open(CFG["results_dir"] / "domain_shift_summary.json", "w") as f:
            json.dump({
                "train_modality": TRAIN_MODALITY,
                "shifted_modality": OTHER_MODALITY,
                "backbone": BACKBONE,
                "acc_same": float(same_res["acc"]),
                "acc_shifted": float(cross_res["acc"]),
                "acc_gap": float(gap),
                "auroc_same": float(same_res["auroc"]),
                "auroc_shifted": float(cross_res["auroc"]),
                "auroc_gap": float(auroc_gap),
            }, f, indent=2)

    # Cross-dataset evaluation (padufes20 / isic2019 / fitzpatrick17k)
    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        res = _eval_and_report(model, loader, f"cross_{ds_name}", CFG["results_dir"])
        fair = fairness(res)
        cross_results[ds_name] = {
            "accuracy": res["acc"], "auroc": res["auroc"],
            "precision": res["macro_prec"], "recall": res["macro_rec"],
            "macro_f1": res["macro_f1"], "micro_f1": res["micro_f1"],
            "weighted_f1": res["weighted_f1"],
            "EOM": fair["EOM"], "PQD": fair["PQD"], "DPM": fair["DPM"],
        }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    plot_training_curves(history, f"Training History ({RUN_TAG})",
                          CFG["results_dir"] / "training_curves.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()