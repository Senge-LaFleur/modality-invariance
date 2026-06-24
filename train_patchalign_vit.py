#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatchAlign — Dual ViT (vit_small_patch16_224) with Masked Graph Optimal Transport
==================================================================================

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
from models.models_losses import confusion_loss, skin_type_loss
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


# ─────────────────────────────────────────────────────────────────────────────
# Safe checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _free_bytes(path: Path) -> int:
    """Return free disk bytes on the filesystem that contains *path*."""
    import shutil as _shutil
    return _shutil.disk_usage(str(path)).free


def safe_torch_save(obj, path: Path, min_free_gb: float = 1.0) -> bool:
    """
    Write *obj* to *path* atomically via a sibling .tmp file.

    Atomic strategy: write to ``<path>.tmp``, then os.replace() — a single
    kernel syscall that is guaranteed to be atomic on POSIX/Linux (including
    Kaggle).  This means a failed save never corrupts the previous checkpoint.

    Returns True on success, False on failure (prints a warning).

    Parameters
    ----------
    min_free_gb : float
        Skip the save (with a warning) if less than this much free space
        remains on the target filesystem.  Defaults to 1.0 GB.
    """
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    # Pre-flight disk-space check
    try:
        free_gb = _free_bytes(path.parent) / 1e9
        if free_gb < min_free_gb:
            print(
                f"[WARN] safe_torch_save: only {free_gb:.2f} GB free — "
                f"skipping save to {path.name}"
            )
            return False
    except Exception:
        pass  # If we can't stat, proceed anyway

    # Remove stale .tmp from a previous crashed run
    tmp_path.unlink(missing_ok=True)

    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)   # atomic rename
        return True
    except (RuntimeError, OSError, IOError) as exc:
        print(f"[WARN] safe_torch_save failed for {path.name}: {exc}")
        tmp_path.unlink(missing_ok=True)   # clean up partial write
        return False


def prune_periodic_checkpoints(ckpt_dir: Path, keep: int = 2) -> None:
    """
    Keep only the *keep* most-recent ``checkpoint_epNNN.pt`` files,
    deleting older ones to reclaim disk space.
    """
    pattern = sorted(ckpt_dir.glob("checkpoint_ep*.pt"))
    for old in pattern[:-keep]:
        try:
            old.unlink()
            print(f"[INFO] Pruned old checkpoint: {old.name}")
        except OSError:
            pass


# ----------------------------- Original label list (115 items) --------------
ORIGINAL_LABELS = [
    'drug induced pigmentary changes', 'photodermatoses',
    'dermatofibroma', 'psoriasis', 'kaposi sarcoma',
    'neutrophilic dermatoses', 'granuloma annulare',
    'nematode infection', 'allergic contact dermatitis',
    'necrobiosis lipoidica', 'hidradenitis', 'melanoma',
    'acne vulgaris', 'sarcoidosis', 'xeroderma pigmentosum',
    'actinic keratosis', 'scleroderma', 'syringoma', 'folliculitis',
    'pityriasis lichenoides chronica', 'porphyria',
    'dyshidrotic eczema', 'seborrheic dermatitis', 'prurigo nodularis',
    'acne', 'neurofibromatosis', 'eczema', 'pediculosis lids',
    'basal cell carcinoma', 'pityriasis rubra pilaris',
    'pityriasis rosea', 'livedo reticularis',
    'stevens johnson syndrome', 'erythema multiforme',
    'acrodermatitis enteropathica', 'epidermolysis bullosa',
    'dermatomyositis', 'urticaria', 'basal cell carcinoma morpheiform',
    'vitiligo', 'erythema nodosum', 'lupus erythematosus',
    'lichen planus', 'sun damaged skin', 'drug eruption', 'scabies',
    'cheilitis', 'urticaria pigmentosa', 'behcets disease',
    'nevocytic nevus', 'mycosis fungoides',
    'superficial spreading melanoma ssm', 'porokeratosis of mibelli',
    'juvenile xanthogranuloma', 'milia', 'granuloma pyogenic',
    'papilomatosis confluentes and reticulate',
    'neurotic excoriations', 'epidermal nevus', 'naevus comedonicus',
    'erythema annulare centrifigum', 'pilar cyst',
    'pustular psoriasis', 'ichthyosis vulgaris', 'lyme disease',
    'striae', 'rhinophyma', 'calcinosis cutis', 'stasis edema',
    'neurodermatitis', 'congenital nevus', 'squamous cell carcinoma',
    'mucinosis', 'keratosis pilaris', 'keloid', 'tuberous sclerosis',
    'acquired autoimmune bullous diseaseherpes gestationis',
    'fixed eruptions', 'lentigo maligna', 'lichen simplex',
    'dariers disease', 'lymphangioma', 'pilomatricoma',
    'lupus subacute', 'perioral dermatitis',
    'disseminated actinic porokeratosis', 'erythema elevatum diutinum',
    'halo nevus', 'aplasia cutis', 'incontinentia pigmenti',
    'tick bite', 'fordyce spots', 'telangiectases',
    'solid cystic basal cell carcinoma', 'paronychia', 'becker nevus',
    'pyogenic granuloma', 'langerhans cell histiocytosis',
    'port wine stain', 'malignant melanoma', 'factitial dermatitis',
    'xanthomas', 'nevus sebaceous of jadassohn',
    'hailey hailey disease', 'scleromyxedema', 'porokeratosis actinic',
    'rosacea', 'acanthosis nigricans', 'myiasis',
    'seborrheic keratosis', 'mucous cyst', 'lichen amyloidosis',
    'ehlers danlos syndrome', 'tungiasis', 'eudermic'
]

CLASS_MAPPING = {
    "melanoma":                 "melanoma",
    "nevus":                    "nevocytic nevus",
    "basal cell carcinoma":     "basal cell carcinoma",
    # "actinic keratosis":        "actinic keratosis",
    # "squamous cell carcinoma":  "squamous cell carcinoma",
    "eudermic":                 "eudermic",
}

# ─────────────────────────────────────────────────────────────────────────────
# PatchAlign Dual ViT model
# ─────────────────────────────────────────────────────────────────────────────

_VIT_SMALL_FEAT_DIM = 384   # vit_small_patch16_224

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
    """
    def __init__(
        self,
        embed_dim: int = 384,
        num_classes: int = 5,
        num_skin_types: int = 6,
        num_text_labels: int = 6,
        text_embed_dim: int = 768,
        pretrained: bool = True,
        use_projection: bool = False,
        vit_model_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        self.clinical_vit = timm.create_model(
            vit_model_name, pretrained=pretrained, num_classes=0
        )
        self.derm_vit = timm.create_model(
            vit_model_name, pretrained=pretrained, num_classes=0
        )

        vit_feat_dim = _VIT_SMALL_FEAT_DIM

        self.use_projection = use_projection
        if use_projection:
            self.proj_head = ProjectionHead(vit_feat_dim, 1024, embed_dim)
        else:
            self.proj_head = nn.Identity()
            embed_dim = vit_feat_dim

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf   = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types),
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(vit_feat_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )

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

    def _extract_features(self, x: torch.Tensor, modality: str):
        vit = self.clinical_vit if modality == "clinical" else self.derm_vit
        hidden = vit.forward_features(x)           # (B, 197, 384)

        cls_token    = hidden[:, 0, :]             # (B, 384)
        patch_tokens = hidden[:, 1:, :]            # (B, 196, 384)

        z       = self.proj_head(cls_token)        # (B, embed_dim)
        patches = self.patch_proj(patch_tokens)    # (B, 196, 768)

        B = x.size(0)
        mask_flat = self.mask_net(patches.view(B, -1))
        mask = mask_flat.view(B, self._n_patches, self.num_text_labels)

        return z, patches, mask

    def forward(self, batch: dict) -> dict:
        device = batch["label"].device
        batch_size = len(batch["label"])

        if "clinical" in batch:
            clinical_t = batch["clinical"].to(device)
            z_clin, patches, mask = self._extract_features(clinical_t, "clinical")
            embeddings = z_clin
            out_patches = patches
            out_masks = mask
        else:
            img_t = batch["image"].to(device)
            z, patches, mask = self._extract_features(img_t, "clinical")
            embeddings = z
            out_patches = patches
            out_masks = mask

        paired_mask = torch.tensor(batch.get("paired", torch.zeros(batch_size, dtype=torch.bool)), dtype=torch.bool, device=device)
        if paired_mask.any() and "clinical" in batch and "derm" in batch:
            clin_t = batch["clinical"][paired_mask].to(device)
            derm_t = batch["derm"][paired_mask].to(device)
            z_c, _, _ = self._extract_features(clin_t, "clinical")
            z_d, _, _ = self._extract_features(derm_t, "derm")
            z_paired = (z_c + z_d) / 2
            embeddings[paired_mask] = z_paired

        if embeddings is None:
            embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device)

        out = {
            "z": embeddings,
            "logits": self.classifier(embeddings),
            "skin_logits": self.skin_clf(embeddings),
            "patches": out_patches,
            "mask": out_masks,
        }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Seeds & device
# ─────────────────────────────────────────────────────────────────────────────
SEED = 0
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
WORK_ROOT = Path(sys.argv[2])
WORK_DIR=sys.argv[1]
CSV_DIR = WORK_ROOT / 'csvs'

IMAGE_ROOTS = {
    'hiba':           Path(WORK_DIR+'/data/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'fitzpatrick17k': Path(WORK_DIR+'/data/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'derm7pt':        Path(WORK_DIR+'/data/datasets/asosenge/derm7pt/release_v0/images'),
    'padufes20':      Path(WORK_DIR+'/data/datasets/mahdavi1202/skin-cancer'),
    'isic2019':       Path(WORK_DIR+'/data/datasets/sengenjih/isic2019'),
}

# WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
# CSV_DIR = WORK_ROOT / 'csvs'

# IMAGE_ROOTS = {
#     'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
#     'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
#     'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
#     'padufes20':      Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),
#     'isic2019':       Path('/kaggle/input/datasets/sengenjih/isic2019'),
# }

FULL_EMBEDDINGS_PATH = WORK_ROOT / 'text_embeddings_3_large_consecutive_averaged.npy'

CFG = {
    'csv_dir':       CSV_DIR,
    'image_roots':   IMAGE_ROOTS,
    'ckpt_dir':      WORK_ROOT / 'checkpoints_PatchAlign_vit',
    'results_dir':   WORK_ROOT / 'results_PatchAlign_vit',

    'vit_model':        'vit_small_patch16_224',
    'embed_dim':        384,
    'img_size':         224,
    'num_classes':      3,
    'num_skin_types':   6,
    'num_text_labels':  4,   # must equal len(CLASS_MAPPING); was incorrectly 6
    'text_embed_dim':   768,

    'batch_size':    32,
    'num_epochs':    500,
    'lr':            3e-5,
    'min_lr':        1e-6,
    'weight_decay':  0.05,
    'warmup_epochs': 50,
    'aug_probability': 0.85,

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

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch",
                dynamic_ncols=True, leave=False)

    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)

            loss_c = criterion_cls(out["logits"], batch["label"])

            skin_labels = batch.get("fitzpatrick", None)
            if skin_labels is None:
                skin_labels = batch.get("skin_type", None)
            if skin_labels is None:
                skin_labels = batch.get("fitzpatrick_scale", None)
            if skin_labels is not None:
                skin_labels = skin_labels.long()
                if skin_labels.max() > 5:
                    skin_labels = skin_labels - 1
                skin_labels = torch.clamp(skin_labels, 0, 5)
                loss_conf = confusion_loss(out["skin_logits"])
                loss_s    = skin_type_loss(out["skin_logits"].detach(), skin_labels)
            else:
                loss_conf = out["skin_logits"].new_tensor(0.)
                loss_s    = out["skin_logits"].new_tensor(0.)
                if epoch == 0 and n_batches == 0:
                    print("[WARN] No skin type column found. L_conf and L_s are zero.")

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
    print("Image roots:")
    for name, root in CFG['image_roots'].items():
        print(f"  {name:<15}: {root}")

    # ── Load text embeddings ───────────────────────────────────────────────
    if FULL_EMBEDDINGS_PATH.exists():
        full_emb = np.load(FULL_EMBEDDINGS_PATH)
        print(f"Loaded full embedding matrix: {full_emb.shape}")

        label_to_idx = {label: idx for idx, label in enumerate(ORIGINAL_LABELS)}
        selected_indices = []
        for our_name, orig_name in CLASS_MAPPING.items():
            if orig_name not in label_to_idx:
                raise ValueError(f"Label '{orig_name}' not found in original label list")
            selected_indices.append(label_to_idx[orig_name])

        text_emb_np = full_emb[selected_indices]
        print(f"Selected embeddings: {text_emb_np.shape}")
        text_emb = torch.tensor(text_emb_np, dtype=torch.float32).to(DEVICE)
    else:
        print("[WARN] Full text embeddings not found — GOT loss will be skipped.")
        print("       Please ensure the file exists at:", FULL_EMBEDDINGS_PATH)
        text_emb = None

    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    # ── Build model ──────────────────────────────────────────────────────────
    # Derive num_text_labels from the loaded text_emb so CFG and mask_net
    # always agree, even if CLASS_MAPPING or the embedding file changes.
    num_text_labels = text_emb.shape[0] if text_emb is not None else CFG["num_text_labels"]
    model = PatchAlignViT(
        embed_dim       = CFG["embed_dim"],
        num_classes     = CFG["num_classes"],
        num_skin_types  = CFG["num_skin_types"],
        num_text_labels = num_text_labels,
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

        # --- atomic save of rolling last checkpoint ---
        last_path = CFG["ckpt_dir"] / "last_model.pt"
        saved_last = safe_torch_save(ckpt_state, last_path)

        if saved_last:
            if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
                best_auroc = val_metrics["auroc"]
                try:
                    shutil.copy(last_path, CFG["ckpt_dir"] / "best_auroc_model.pt")
                except (OSError, IOError) as exc:
                    print(f"[WARN] Could not copy best_auroc checkpoint: {exc}")

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                try:
                    shutil.copy(last_path, CFG["ckpt_dir"] / "best_f1_model.pt")
                except (OSError, IOError) as exc:
                    print(f"[WARN] Could not copy best_f1 checkpoint: {exc}")

        # --- periodic checkpoint every 5 epochs; prune old ones to save space ---
        if (epoch + 1) % 5 == 0:
            periodic_path = CFG["ckpt_dir"] / f"checkpoint_ep{epoch+1:03d}.pt"
            safe_torch_save(ckpt_state, periodic_path)
            prune_periodic_checkpoints(CFG["ckpt_dir"], keep=2)

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
    val_fair_binary = fairness_binary(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES, fair_binary=val_fair_binary)
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
    print("\nBinary fairness (Validation):")
    print(f"  Acc_light: {val_fair_binary['Acc_light']:.4f}  (FST I–III)")
    print(f"  Acc_dark : {val_fair_binary['Acc_dark']:.4f}  (FST IV–VI)")
    print(f"  Acc_gap  : {val_fair_binary['Acc_gap']:.4f}")
    print(f"  DP_diff  : {val_fair_binary['DP_diff']:.4f}")
    print(f"  EOpp0    : {val_fair_binary['EOpp0']:.4f}")
    print(f"  EOpp1    : {val_fair_binary['EOpp1']:.4f}")
    print(f"  EOdd     : {val_fair_binary['EOdd']:.4f}")

    if test_loader:
        test_res = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        test_fair = fairness(test_res)
        test_fair_binary = fairness_binary(test_res)

        # ---- Compute KNN accuracy on test embeddings ----
        model.eval()
        all_embs = []
        all_labels_tsne = []
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
        knn_acc = compute_knn_accuracy(embs, labels_tsne, k=5)
        print(f"\n[PatchAlign ViT] Test KNN (k=5) accuracy: {knn_acc:.4f}")

        save_results_csv(test_res, test_fair, "test", CFG["results_dir"], LABEL_NAMES,
                         knn_acc=knn_acc, fair_binary=test_fair_binary)
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
        print("\nBinary fairness (Test):")
        print(f"  Acc_light: {test_fair_binary['Acc_light']:.4f}  (FST I–III)")
        print(f"  Acc_dark : {test_fair_binary['Acc_dark']:.4f}  (FST IV–VI)")
        print(f"  Acc_gap  : {test_fair_binary['Acc_gap']:.4f}")
        print(f"  DP_diff  : {test_fair_binary['DP_diff']:.4f}")
        print(f"  EOpp0    : {test_fair_binary['EOpp0']:.4f}")
        print(f"  EOpp1    : {test_fair_binary['EOpp1']:.4f}")
        print(f"  EOdd     : {test_fair_binary['EOdd']:.4f}")

        # t-SNE plot
        plot_tsne(embs, labels_tsne, "t-SNE — Test Set (PatchAlign ViT)",
                  CFG["results_dir"] / "tsne_test.png")

    # ── Cross-dataset evaluation ─────────────────────────────────────────────
    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        print(f"\nEvaluating on {ds_name}")
        res  = validate(model, loader, DEVICE, CFG["num_classes"],
                        desc=f"Cross-eval: {ds_name}")
        fair = fairness(res)
        fair_binary = fairness_binary(res)
        save_results_csv(res, fair, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES, fair_binary=fair_binary)
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
        print(f"\nBinary fairness ({ds_name}):")
        print(f"  Acc_light: {fair_binary['Acc_light']:.4f}  (FST I–III)")
        print(f"  Acc_dark : {fair_binary['Acc_dark']:.4f}  (FST IV–VI)")
        print(f"  Acc_gap  : {fair_binary['Acc_gap']:.4f}")
        print(f"  DP_diff  : {fair_binary['DP_diff']:.4f}")
        print(f"  EOpp0    : {fair_binary['EOpp0']:.4f}")
        print(f"  EOpp1    : {fair_binary['EOpp1']:.4f}")
        print(f"  EOdd     : {fair_binary['EOdd']:.4f}")

        cross_results[ds_name] = {
            "accuracy": res["acc"],
            "auroc": res["auroc"],
            "precision": res["macro_prec"],
            "recall": res["macro_rec"],
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

    plot_training_curves(history, "Training History (PatchAlign ViT)",
                         CFG["results_dir"] / "training_curves.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()