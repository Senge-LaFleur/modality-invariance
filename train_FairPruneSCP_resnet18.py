#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_FairPruneSCP_resnet18.py
Implements Sensitive Channel Pruning (SCP) using SNNL on dual-encoder ResNet-18.
Iteratively prunes sensitive channels, fine-tunes, and tracks fairness.

All outputs are saved under:
    checkpoints_SCP_resnet18/ and results_SCP_resnet18/
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

from sklearn.metrics import f1_score
from models.models_losses import DualResNet18, compute_class_weights
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
# Configuration – copy from train_BASE_resnet18.py, but with pruning parameters
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

# Paths (adjust to your environment)
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
    'ckpt_dir':     WORK_ROOT / 'checkpoints_FairPruneSCP_resnet18',
    'results_dir':  WORK_ROOT / 'results_FairPruneSCP_resnet18',

    'backbone': 'resnet18',
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 5,
    'num_skin_types': 6,

    'batch_size': 32,
    'num_epochs': 150,           # Update as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 30,         # Update as needed
    'aug_probability': 0.85,

    # SCP specific
    'prune_ratio': 0.02,         # fraction of channels pruned each iteration
    'max_prune_iters': 5,        # maximum number of pruning iterations
    'finetune_epochs_per_iter': 10,  # fine‑tune epochs after each prune
    'snnl_temperature': 1.0,
    'sensitive_attr': 'skin_type_binary',  # or 'gender'
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# SNNL computation (same as in SNNL_SCP_resnet18.py)
# ------------------------------------------------------------
class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, temperature=1.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        emb_norm = F.normalize(embeddings, dim=1)
        pairwise_cos = 1 - torch.mm(emb_norm, emb_norm.t())
        exp_neg = torch.exp(-pairwise_cos / self.temperature)

        same_group = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        eye = torch.eye(batch_size, device=embeddings.device)
        mask_same = same_group - eye
        mask_all = 1 - eye

        same_sum = (exp_neg * mask_same).sum(dim=1)
        all_sum = (exp_neg * mask_all).sum(dim=1) + self.eps
        loss = -torch.log((same_sum + self.eps) / all_sum).mean()
        return loss

def compute_channel_snnl(feature_maps, sensitive_labels, temperature=1.0):
    device = feature_maps.device
    batch, C, H, W = feature_maps.shape
    scores = []
    snnl_loss = SoftNearestNeighborLoss(temperature=temperature)
    for c in range(C):
        feat_c = feature_maps[:, c, :, :].reshape(batch, -1)
        if feat_c.shape[1] == 0:
            scores.append(0.0)
            continue
        loss_val = snnl_loss(feat_c, sensitive_labels)
        scores.append(loss_val.item())
    return torch.tensor(scores, device=device)

def get_last_conv_layer(model, encoder_name):
    encoder = getattr(model, encoder_name)
    return encoder.layer4[-1].conv2

def register_hook(layer, outputs_dict, key):
    def hook(module, input, output):
        outputs_dict[key] = output.detach()
    return layer.register_forward_hook(hook)

# ------------------------------------------------------------
# Pruning function (zero out entire output channels)
# ------------------------------------------------------------
def prune_channels_by_indices(layer, channel_indices):
    """
    Zero out the weights and biases of specified output channels.
    layer: nn.Conv2d
    channel_indices: list of ints (0‑based)
    """
    with torch.no_grad():
        # Zero weight filters
        for idx in channel_indices:
            layer.weight.data[idx] = 0.0
        if layer.bias is not None:
            for idx in channel_indices:
                layer.bias.data[idx] = 0.0
    print(f"Pruned {len(channel_indices)} channels from {layer}")

# ------------------------------------------------------------
# Training function (identical to base, but accepts a prefix for logging)
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, epoch, scaler, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)
            loss = criterion(out["logits"], batch["label"])

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

    avg_loss = total_loss / max(n_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"total": avg_loss, "acc": acc, "macro_f1": macro_f1}

# ------------------------------------------------------------
# Main pruning & training loop
# ------------------------------------------------------------
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")
    print(f"Prune ratio  : {CFG['prune_ratio']}, max iters: {CFG['max_prune_iters']}")

    # Build loaders
    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Baseline model (pretrained)
    model = DualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=False,
    ).to(DEVICE)

    # Optional: load a previously trained baseline checkpoint
    # baseline_ckpt = CFG['ckpt_dir'].parent / 'checkpoints_BASE_resnet18/best_f1_model.pt'
    # if baseline_ckpt.exists():
    #     ckpt = torch.load(baseline_ckpt, map_location=DEVICE)
    #     model.load_state_dict(ckpt['model'])
    #     print(f"Loaded baseline from {baseline_ckpt}")

    # Identify layers to prune
    clinical_conv = get_last_conv_layer(model, 'clinical_encoder')
    derm_conv = get_last_conv_layer(model, 'dermoscopic_encoder')
    print(f"Clinical conv out_channels: {clinical_conv.out_channels}")
    print(f"Dermoscopic conv out_channels: {derm_conv.out_channels}")

    # Storage for history and FATE
    prune_history = {
        'iteration': [],
        'val_acc': [],
        'val_f1': [],
        'EOpp': [],
        'EOdd': [],
        'FATE_EOpp': [],
        'FATE_EOdd': [],
        'n_pruned_clinical': [],
        'n_pruned_derm': [],
    }

    # Evaluate baseline fairness (needed for FATE)
    print("\n=== Baseline evaluation (before pruning) ===")
    baseline_val = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Baseline validation")
    baseline_fair = fairness(baseline_val)
    baseline_eopp = baseline_fair.get('EOpp', 0.0)
    baseline_eodd = baseline_fair.get('EOdd', 0.0)
    print(f"Baseline: acc={baseline_val['acc']:.4f}, f1={baseline_val['macro_f1']:.4f}, EOpp={baseline_eopp:.4f}, EOdd={baseline_eodd:.4f}")

    # Best metrics for early stopping
    best_fate_eopp = -np.inf
    best_fate_eodd = -np.inf
    patience_counter = 0
    patience = 2  # stop if FATE does not improve for 2 iterations

    # Pruning iterations
    for it in range(CFG['max_prune_iters']):
        print(f"\n{'='*50}")
        print(f"Pruning iteration {it+1}/{CFG['max_prune_iters']}")
        print(f"{'='*50}")

        # ---- Step 1: Compute SNNL scores on training set ----
        print("Computing SNNL scores...")
        clinical_output = {}
        derm_output = {}
        hook_clin = register_hook(clinical_conv, clinical_output, 'out')
        hook_derm = register_hook(derm_conv, derm_output, 'out')

        all_clin_scores = []
        all_derm_scores = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="SNNL batches"):
                clinical = batch['clinical'].to(DEVICE)
                dermoscopic = batch['dermoscopic'].to(DEVICE)
                sensitive = batch[CFG['sensitive_attr']].to(DEVICE)

                _ = model({'clinical': clinical, 'dermoscopic': dermoscopic})
                feat_clin = clinical_output['out']
                feat_derm = derm_output['out']

                scores_clin = compute_channel_snnl(feat_clin, sensitive, CFG['snnl_temperature'])
                scores_derm = compute_channel_snnl(feat_derm, sensitive, CFG['snnl_temperature'])

                all_clin_scores.append(scores_clin.cpu().numpy())
                all_derm_scores.append(scores_derm.cpu().numpy())

        hook_clin.remove()
        hook_derm.remove()

        avg_clin = np.mean(all_clin_scores, axis=0)
        avg_derm = np.mean(all_derm_scores, axis=0)

        # ---- Step 2: Select channels to prune (lowest scores) ----
        n_prune_clin = max(1, int(len(avg_clin) * CFG['prune_ratio']))
        n_prune_derm = max(1, int(len(avg_derm) * CFG['prune_ratio']))
        clin_sorted = np.argsort(avg_clin)
        derm_sorted = np.argsort(avg_derm)
        prune_clin = clin_sorted[:n_prune_clin].tolist()
        prune_derm = derm_sorted[:n_prune_derm].tolist()
        print(f"Pruning {len(prune_clin)} clinical channels, {len(prune_derm)} dermoscopic channels")

        # ---- Step 3: Apply pruning (zero out) ----
        prune_channels_by_indices(clinical_conv, prune_clin)
        prune_channels_by_indices(derm_conv, prune_derm)

        # ---- Step 4: Fine‑tune the model ----
        # Reinitialize optimizer and scheduler for fine‑tuning
        from models.models_losses import get_layer_wise_lr_params
        param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"], betas=(0.9, 0.999), eps=1e-8)

        def lr_lambda(epoch):
            if epoch < CFG["warmup_epochs"]:
                return (epoch + 1) / CFG["warmup_epochs"]
            progress = (epoch - CFG["warmup_epochs"]) / max(1, CFG["finetune_epochs_per_iter"] - CFG["warmup_epochs"])
            cos = 0.5 * (1 + math.cos(math.pi * progress))
            min_frac = CFG["min_lr"] / CFG["lr"]
            return max(min_frac, cos)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

        print(f"Fine‑tuning for {CFG['finetune_epochs_per_iter']} epochs...")
        best_val_f1 = 0.0
        for epoch in range(CFG['finetune_epochs_per_iter']):
            train_metrics = train_epoch(model, train_loader, optimizer, epoch, scaler, DEVICE)
            scheduler.step()
            val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
            if val_metrics['macro_f1'] > best_val_f1:
                best_val_f1 = val_metrics['macro_f1']
            print(f"  FT Ep{epoch+1}: tr_loss={train_metrics['total']:.4f}, val_acc={val_metrics['acc']:.4f}, val_f1={val_metrics['macro_f1']:.4f}")

        # ---- Step 5: Evaluate fairness after this iteration ----
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Post‑prune validation")
        fair = fairness(val_metrics)
        eopp = fair.get('EOpp', 0.0)
        eodd = fair.get('EOdd', 0.0)

        # Compute FATE (relative improvement over baseline)
        fate_eopp = (baseline_eopp - eopp) / (baseline_eopp + 1e-8)
        fate_eodd = (baseline_eodd - eodd) / (baseline_eodd + 1e-8)

        print(f"Iteration {it+1}: acc={val_metrics['acc']:.4f}, f1={val_metrics['macro_f1']:.4f}, EOpp={eopp:.4f} (FATE={fate_eopp:.4f}), EOdd={eodd:.4f} (FATE={fate_eodd:.4f})")

        # Save history
        prune_history['iteration'].append(it+1)
        prune_history['val_acc'].append(val_metrics['acc'])
        prune_history['val_f1'].append(val_metrics['macro_f1'])
        prune_history['EOpp'].append(eopp)
        prune_history['EOdd'].append(eodd)
        prune_history['FATE_EOpp'].append(fate_eopp)
        prune_history['FATE_EOdd'].append(fate_eodd)
        prune_history['n_pruned_clinical'].append(len(prune_clin))
        prune_history['n_pruned_derm'].append(len(prune_derm))

        # Save checkpoint after this iteration
        torch.save({
            'iteration': it+1,
            'model': model.state_dict(),
            'history': prune_history,
        }, CFG['ckpt_dir'] / f'iter_{it+1}_model.pt')

        # Early stopping based on FATE (if no improvement in either metric)
        current_best = max(fate_eopp, fate_eodd)
        if current_best > max(best_fate_eopp, best_fate_eodd):
            best_fate_eopp = fate_eopp
            best_fate_eodd = fate_eodd
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping: FATE did not improve for {patience} iterations.")
                break

    # ------------------------------------------------------------
    # Final evaluation on test and cross-datasets
    # ------------------------------------------------------------
    print("\n=== Final evaluation with best pruned model ===")
    # Load best iteration model (or last)
    final_ckpt = CFG['ckpt_dir'] / f'iter_{prune_history["iteration"][-1]}_model.pt'
    ckpt = torch.load(final_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])

    # Validation
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Final validation")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation", CFG["results_dir"] / "val_confusion.png")
    plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation", CFG["results_dir"] / "val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"] / "val_fairness.png")

    # Test
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
        fair_metrics = fairness(res)
        save_results_csv(res, fair_metrics, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              f"Confusion Matrix - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_confusion.png")
        plot_per_class_metrics(res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               f"Per-Class Metrics - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_per_class.png")
        plot_fairness_metrics(fair_metrics, f"Fairness - {ds_name}", CFG["results_dir"] / f"cross_{ds_name}_fairness.png")
        cross_results[ds_name] = {
            "accuracy": res["acc"],
            "precision": res["macro_prec"],
            "recall": res["macro_rec"],
            "auroc": res["auroc"],
            "macro_f1": res["macro_f1"],
            "micro_f1": res["micro_f1"],
            "weighted_f1": res["weighted_f1"],
            "EOM": fair_metrics["EOM"],
            "PQD": fair_metrics["PQD"],
            "DPM": fair_metrics["DPM"],
        }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    # Save pruning history
    pd.DataFrame(prune_history).to_csv(CFG["results_dir"] / "prune_history.csv", index=False)

    # (Optional) t‑SNE plot for test set
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
        plot_tsne(embs, labels_tsne, "t-SNE - Test Set (SCP)", CFG["results_dir"] / "tsne_test.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")

if __name__ == "__main__":
    main()