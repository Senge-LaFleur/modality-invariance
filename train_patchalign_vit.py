#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatchAlign — Dual ViT (vit_small_patch16_224) with Masked Graph Optimal Transport
==================================================================================
Extends train_BASE_vit.py with the three PatchAlign losses from:
  "Fair and Accurate Skin Disease Image Classification by Alignment with
   Clinical Labels" (PatchAlign, MICCAI 2024).

ViT is the recommended backbone in the paper.  The ViT naturally produces
patch tokens from its transformer blocks, making the alignment architecture
cleaner than the ResNet variant: no patch_proj is needed when using ViT-base
(hidden_size = 768 matches text_embed_dim = 768 directly).  For vit_small
(hidden_size = 384) a lightweight projection layer is used.

Added losses on top of the baseline cross-entropy:
  • Confusion loss  (L_conf)  — removes skin-type bias from representations
  • Skin-type CE    (L_s)     — keeps skin_clf calibrated
  • MGOT alignment  (L_got)   — aligns image patches with clinical text embeddings

Total loss (Eq. 3 of the paper):
  L = L_c + α·L_conf + L_s + β·L_got

New model outputs compared to the baseline:
  out["patches"]     — patch-level embeddings  (B, N_patches, text_embed_dim)
  out["mask"]        — learnable MGOT mask      (B, N_patches, num_text_labels)
  out["logits"]      — disease logits            (B, num_classes)               [same]
  out["skin_logits"] — skin-type logits          (B, num_skin_types)            [same]
  out["z"]           — fused CLS embedding       (B, embed_dim)                 [same]

All outputs (checkpoints, results) are saved in:
    - checkpoints_patchalign_vit/
    - results_patchalign_vit/
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

import timm

from sklearn.metrics import f1_score
from models.models_losses import compute_class_weights
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

from models.Masked_GOT_NewSinkhorn import (
    cost_matrix_batch_torch,
    IPOT_distance_torch_batch_uniform,
    GW_distance_uniform,
)

def got_loss(p, q, mask, lamb=0.9):
    """
    Combined Wasserstein + Gromov-Wasserstein loss (Eq. 1 of PatchAlign paper).
    p    : (B, N_patches, D)  image patch embeddings
    q    : (B, N_text,    D)  text label  embeddings
    mask : (B, N_patches, N_text)  learnable soft mask in [0, 1]
    lamb : weight on GWD term vs WD term
    """
    p_t = p.transpose(1, 2)
    q_t = q.transpose(1, 2)

    cos_distance = cost_matrix_batch_torch(p_t, q_t).transpose(1, 2)
    beta = 0.1
    threshold = cos_distance.min() + beta * (cos_distance.max() - cos_distance.min())
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)

    bs, n_p, n_t = cos_dist.size()
    wd, _T = IPOT_distance_torch_batch_uniform(cos_dist, mask, bs, n_p, n_t)
    gwd    = GW_distance_uniform(p_t, q_t, mask)
    return lamb * torch.mean(gwd) + (1.0 - lamb) * torch.mean(wd)

class Confusion_Loss(torch.nn.Module):
    """
    Confusion loss — encourages uniform skin-type predictions so the backbone
    cannot distinguish skin types.  Copied verbatim from models/got_losses.py
    to avoid that module's top-level `from transformers import ...` import
    which is not needed here and would crash if transformers is not installed.
    Source: https://www.repository.cam.ac.uk/handle/1810/309834
    """
    def __init__(self):
        super().__init__()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, output, label):
        prediction = self.softmax(output)
        log_prediction = torch.log(prediction)
        return -torch.mean(torch.mean(log_prediction, dim=1), dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# PatchAlign Dual ViT model
# ─────────────────────────────────────────────────────────────────────────────

_VIT_SMALL_FEAT_DIM = 384   # vit_small_patch16_224  (paper uses ViT-base = 768)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = 384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 1 and self.training:
            self.eval(); out = self.net(x); self.train()
        else:
            out = self.net(x)
        return F.normalize(out, dim=-1)


class PatchAlignViT(nn.Module):
    """
    Dual ViT-small encoder with PatchAlign MGOT alignment.

    ViT naturally provides per-patch token embeddings from last_hidden_state.
    For vit_small (D=384) we project patches to text_embed_dim (768) via
    patch_proj.  If you switch to vit_base (D=768) you can set
    text_embed_dim=768 and the projection becomes an identity.

    Number of patch tokens = (224/16)² = 196.  For paired samples, clinical
    and derm patches are concatenated → 392 patches per sample.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        num_skin_types: int = 6,
        num_text_labels: int = 6,       # disease classes + eudermic
        text_embed_dim: int = 768,      # dimensionality of saved text embeddings
        pretrained: bool = True,
        use_projection: bool = False,
        vit_model_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        # ── ViT backbones (output: last_hidden_state) ────────────────────────
        # We need the full hidden states, so we use the HuggingFace-style
        # forward that returns all patch tokens, not the timm pooled output.
        # timm's `forward_features` returns (B, N_patches+1, D) including CLS.
        self.clinical_vit = timm.create_model(
            vit_model_name, pretrained=pretrained, num_classes=0
        )
        self.derm_vit = timm.create_model(
            vit_model_name, pretrained=pretrained, num_classes=0
        )

        vit_feat_dim = _VIT_SMALL_FEAT_DIM   # 384 for vit_small

        # ── Projection head (CLS token → embed_dim) ──────────────────────────
        self.use_projection = use_projection
        if use_projection:
            self.proj_head = ProjectionHead(vit_feat_dim, 1024, embed_dim)
        else:
            self.proj_head = nn.Identity()
            embed_dim = vit_feat_dim

        # ── Disease classifier & skin-type classifier ────────────────────────
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf   = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types),
        )

        # ── PatchAlign additions ─────────────────────────────────────────────
        # Project ViT patch tokens (384-d) to text embedding space (768-d)
        self.patch_proj = nn.Sequential(
            nn.Linear(vit_feat_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )

        # Mask generator: flattened patch embeddings → soft mask (B, N_p, N_text)
        # N_p = 196 for vit_small_patch16_224; N_text = num_text_labels
        n_patches = 196
        self.num_text_labels = num_text_labels
        self.mask_net = nn.Sequential(
            nn.Linear(n_patches * text_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, n_patches * num_text_labels),
            nn.Sigmoid(),
        )
        self._n_patches      = n_patches
        self._text_embed_dim = text_embed_dim

    # ── Low-level helpers ────────────────────────────────────────────────────
    def _extract_features(self, x: torch.Tensor, modality: str):
        """
        Returns
          z       : (B, embed_dim)           CLS-derived global embedding
          patches : (B, 196, text_embed_dim) projected patch tokens
          mask    : (B, 196, num_text_labels) learnable MGOT mask
        """
        vit = self.clinical_vit if modality == "clinical" else self.derm_vit

        # timm's forward_features returns (B, N_patches+1, D)
        # where token 0 is the [CLS] token and tokens 1..196 are patch tokens
        hidden = vit.forward_features(x)           # (B, 197, 384)

        cls_token    = hidden[:, 0, :]             # (B, 384)  — CLS
        patch_tokens = hidden[:, 1:, :]            # (B, 196, 384) — patch tokens

        z       = self.proj_head(cls_token)        # (B, embed_dim)
        patches = self.patch_proj(patch_tokens)    # (B, 196, 768)

        B = x.size(0)
        mask_flat = self.mask_net(patches.view(B, -1))        # (B, 196*num_text)
        mask = mask_flat.view(B, self._n_patches, self.num_text_labels)  # (B, 196, 6)

        return z, patches, mask

    # ── Forward ─────────────────────────────────────────────────────────────
    def forward(self, batch: dict) -> dict:
        device     = batch["label"].device
        batch_size = len(batch["label"])
        embed_dim  = self.classifier[1].in_features
        embeddings = None

        paired_mask   = torch.tensor(batch["paired"], dtype=torch.bool, device=device)
        unpaired_mask = ~paired_mask

        out = {}
        patches_list, mask_list_idx = [], []

        # ── Paired samples ───────────────────────────────────────────────────
        if paired_mask.any() and "clinical" in batch and "derm" in batch:
            clin_t = batch["clinical"][paired_mask].to(device)
            derm_t = batch["derm"][paired_mask].to(device)

            z_c, p_c, m_c = self._extract_features(clin_t, "clinical")
            z_d, p_d, m_d = self._extract_features(derm_t, "derm")
            z_paired = (z_c + z_d) / 2

            if embeddings is None:
                embeddings = torch.zeros(batch_size, embed_dim, device=device,
                                         dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            out["z_c"] = z_c
            out["z_d"] = z_d

            # Fuse patches from both modalities along the patch dimension
            fused_patches = torch.cat([p_c, p_d], dim=1)   # (B_p, 392, 768)
            fused_mask    = torch.cat([m_c, m_d], dim=1)   # (B_p, 392,   6)
            patches_list.append((paired_mask, fused_patches, fused_mask))

        # ── Unpaired samples ─────────────────────────────────────────────────
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            z, patches, mask = self._extract_features(img_t, "clinical")

            if embeddings is None:
                embeddings = torch.zeros(batch_size, embed_dim, device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z
            patches_list.append((unpaired_mask, patches, mask))

        # Fallback
        if embeddings is None:
            embeddings = torch.zeros(batch_size, embed_dim, device=device)

        # ── Assemble patch/mask tensors at full batch size ───────────────────
        if patches_list:
            n_patches = patches_list[0][1].size(1)
            text_d    = patches_list[0][1].size(2)
            n_text    = patches_list[0][2].size(2)
            all_patches = torch.zeros(batch_size, n_patches, text_d,
                                      device=device, dtype=patches_list[0][1].dtype)
            all_masks   = torch.zeros(batch_size, n_patches, n_text,
                                      device=device, dtype=patches_list[0][2].dtype)
            for sel_mask, p, m in patches_list:
                all_patches[sel_mask] = p
                all_masks[sel_mask]   = m
            out["patches"] = all_patches
            out["mask"]    = all_masks

        out["z"]           = embeddings
        out["logits"]      = self.classifier(embeddings)
        out["skin_logits"] = self.skin_clf(embeddings)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Seeds & device
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# Path configuration — update for each environment
# ─────────────────────────────────────────────────────────────────────────────
WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR   = WORK_ROOT / 'csvs'

DATASET_ROOTS = {
    'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k'),
    'ham10000':       Path('/kaggle/input/datasets/asosenge/ham10000'),
    'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt'),
}

TEXT_EMBEDDINGS_PATH = WORK_ROOT / 'text_embeddings_3_large_consecutive_averaged.npy'

print("Checking configured paths:")
print(f"  WORK_ROOT          : {WORK_ROOT}  {'[OK]' if WORK_ROOT.exists() else '[MISSING]'}")
print(f"  CSV_DIR            : {CSV_DIR}  {'[OK]' if CSV_DIR.exists() else '[MISSING]'}")
print(f"  TEXT_EMBEDDINGS    : {TEXT_EMBEDDINGS_PATH}  "
      f"{'[OK]' if TEXT_EMBEDDINGS_PATH.exists() else '[MISSING — run Create_Embeddings.ipynb first]'}")
for name, root in DATASET_ROOTS.items():
    print(f"  {name:<15}: {root}  {'[OK]' if root.exists() else '[MISSING — update DATASET_ROOTS]'}")

CFG = {
    'csv_dir':       CSV_DIR,
    'dataset_roots': DATASET_ROOTS,
    'ckpt_dir':      WORK_ROOT / 'checkpoints_patchalign_vit',
    'results_dir':   WORK_ROOT / 'results_patchalign_vit',

    'vit_model':        'vit_small_patch16_224',
    'embed_dim':        384,
    'img_size':         224,
    'num_classes':      5,
    'num_skin_types':   6,
    'num_text_labels':  6,
    'text_embed_dim':   768,

    'batch_size':    32,
    'num_epochs':    20,         # paper trains for 20 epochs
    'lr':            3e-5,
    'min_lr':        1e-6,
    'weight_decay':  0.05,
    'warmup_epochs': 4,          # ≈ 1/5 × num_epochs
    'aug_probability': 0.85,

    # ── PatchAlign loss weights (Eq. 3 of the paper) ─────────────────────
    'alpha_conf': 0.5,
    'beta_got':   1.0,
    'lamb_got':   0.9,
}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, epoch, scaler, device, text_emb, cfg):
    model.train()
    total_loss    = 0.0
    loss_c_sum    = 0.0
    loss_conf_sum = 0.0
    loss_s_sum    = 0.0
    loss_got_sum  = 0.0
    all_preds, all_labels = [], []
    n_batches = 0

    criterion_cls  = nn.CrossEntropyLoss()
    criterion_skin = nn.CrossEntropyLoss(ignore_index=-1)
    confusion_loss = Confusion_Loss()

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch",
                dynamic_ncols=True, leave=False)

    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)

            # L_c
            loss_c = criterion_cls(out["logits"], batch["label"])

            # L_conf + L_s
            skin_labels = batch.get("fitzpatrick", None)
            if skin_labels is not None:
                skin_labels = skin_labels.long()
                loss_conf = confusion_loss(out["skin_logits"], skin_labels)
                loss_s    = criterion_skin(out["skin_logits"], skin_labels)
            else:
                loss_conf = out["skin_logits"].new_tensor(0.)
                loss_s    = out["skin_logits"].new_tensor(0.)

            # L_got
            loss_got = out["logits"].new_tensor(0.)
            if "patches" in out and text_emb is not None:
                B = out["patches"].size(0)
                text_batch = text_emb.unsqueeze(0).expand(B, -1, -1)
                loss_got = got_loss(
                    out["patches"],
                    text_batch,
                    out["mask"],
                    lamb=cfg['lamb_got'],
                )

            loss = (loss_c
                    + cfg['alpha_conf'] * loss_conf
                    + loss_s
                    + cfg['beta_got']   * loss_got)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss    += loss.item()
        loss_c_sum    += loss_c.item()
        loss_conf_sum += loss_conf.item()
        loss_s_sum    += loss_s.item()
        loss_got_sum  += loss_got.item() if isinstance(loss_got, torch.Tensor) else float(loss_got)
        n_batches += 1

        with torch.no_grad():
            preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())

        pbar.set_postfix(
            loss=f"{total_loss/n_batches:.4f}",
            lc=f"{loss_c_sum/n_batches:.3f}",
            lgot=f"{loss_got_sum/n_batches:.3f}",
        )

    pbar.close()
    nb = max(n_batches, 1)
    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "total":    total_loss    / nb,
        "cls":      loss_c_sum    / nb,
        "conf":     loss_conf_sum / nb,
        "skin":     loss_s_sum    / nb,
        "got":      loss_got_sum  / nb,
        "acc":      (all_preds == all_labels).mean(),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")
    print("Dataset roots:")
    for name, root in CFG['dataset_roots'].items():
        print(f"  {name:<15}: {root}")

    # ── Load text embeddings ─────────────────────────────────────────────────
    if TEXT_EMBEDDINGS_PATH.exists():
        text_emb_np = np.load(TEXT_EMBEDDINGS_PATH)
        text_emb    = torch.tensor(text_emb_np, dtype=torch.float32).to(DEVICE)
        print(f"Loaded text embeddings: {text_emb.shape}")
    else:
        print("[WARN] Text embeddings not found — GOT loss will be skipped.")
        print("       Run Create_Embeddings.ipynb first.")
        text_emb = None

    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = PatchAlignViT(
        embed_dim       = CFG["embed_dim"],
        num_classes     = CFG["num_classes"],
        num_skin_types  = CFG["num_skin_types"],
        num_text_labels = CFG["num_text_labels"],
        text_embed_dim  = CFG["text_embed_dim"],
        pretrained      = True,
        use_projection  = False,
        vit_model_name  = CFG["vit_model"],
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"],
        weight_decay=CFG["weight_decay"], betas=(0.9, 0.999), eps=1e-8,
    )

    def lr_lambda(epoch):
        if epoch < CFG["warmup_epochs"]:
            return (epoch + 1) / CFG["warmup_epochs"]
        progress = (epoch - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return max(CFG["min_lr"] / CFG["lr"], cos)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 0
    best_auroc  = 0.0
    best_f1     = 0.0
    history     = defaultdict(list)

    # ── Training loop ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, CFG["num_epochs"]):
        train_metrics = train_epoch(
            model, train_loader, optimizer, epoch, scaler, DEVICE, text_emb, CFG
        )
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
            history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))

        print(
            f"Ep {epoch+1:3d}/{CFG['num_epochs']}  "
            f"loss={train_metrics['total']:.4f}  "
            f"(cls={train_metrics['cls']:.3f} "
            f"conf={train_metrics['conf']:.3f} "
            f"skin={train_metrics['skin']:.3f} "
            f"got={train_metrics['got']:.3f})  "
            f"tr_acc={train_metrics['acc']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  "
            f"val_auroc={val_metrics['auroc']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}  "
            f"lr={lr:.2e}"
        )

        ckpt_state = {
            "epoch":     epoch,
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history":   dict(history),
        }
        torch.save(ckpt_state, CFG["ckpt_dir"] / "last_model.pt")
        if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt",
                        CFG["ckpt_dir"] / "best_auroc_model.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt",
                        CFG["ckpt_dir"] / "best_f1_model.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt_state, CFG["ckpt_dir"] / f"checkpoint_ep{epoch+1:03d}.pt")

        with open(CFG["results_dir"] / "history.json", "w") as f:
            json.dump({k: [float(x) for x in v] for k, v in history.items()}, f, indent=2)

    print(f"Training complete. Best AUROC: {best_auroc:.4f}, Best F1: {best_f1:.4f}")

    # ── Load best model ──────────────────────────────────────────────────────
    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # ── Evaluation ───────────────────────────────────────────────────────────
    val_res  = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"],
                          [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation",
                          CFG["results_dir"] / "val_confusion.png")
    plot_per_class_metrics(val_res,
                           [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation",
                           CFG["results_dir"] / "val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation",
                          CFG["results_dir"] / "val_fairness.png")

    if test_loader:
        test_res  = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        test_fair = fairness(test_res)
        save_results_csv(test_res, test_fair, "test", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(test_res["conf_mat"],
                              [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              "Confusion Matrix - Test",
                              CFG["results_dir"] / "test_confusion.png")
        plot_per_class_metrics(test_res,
                               [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               "Per-Class Metrics - Test",
                               CFG["results_dir"] / "test_per_class.png")
        plot_fairness_metrics(test_fair, "Fairness - Test",
                              CFG["results_dir"] / "test_fairness.png")

    # ── Cross-dataset evaluation ─────────────────────────────────────────────
    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        print(f"\nEvaluating on {ds_name}")
        res  = validate(model, loader, DEVICE, CFG["num_classes"],
                        desc=f"Cross-eval: {ds_name}")
        fair = fairness(res)
        save_results_csv(res, fair, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(res["conf_mat"],
                              [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              f"Confusion Matrix - {ds_name}",
                              CFG["results_dir"] / f"cross_{ds_name}_confusion.png")
        plot_per_class_metrics(res,
                               [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               f"Per-Class Metrics - {ds_name}",
                               CFG["results_dir"] / f"cross_{ds_name}_per_class.png")
        plot_fairness_metrics(fair, f"Fairness - {ds_name}",
                              CFG["results_dir"] / f"cross_{ds_name}_fairness.png")
        cross_results[ds_name] = {
            "accuracy": res["acc"],
            "auroc":    res["auroc"],
            "macro_f1": res["macro_f1"],
            "EOM":      fair["EOM"],
            "PQD":      fair["PQD"],
            "DPM":      fair["DPM"],
        }
    if cross_results:
        cross_df = pd.DataFrame(cross_results).T
        cross_df.to_csv(CFG["results_dir"] / "cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", cross_df)

    plot_training_curves(history, "Training History (PatchAlign ViT)",
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
        embs        = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        plot_tsne(embs, labels_tsne,
                  "t-SNE — Test Set (PatchAlign ViT)",
                  CFG["results_dir"] / "tsne_test.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()