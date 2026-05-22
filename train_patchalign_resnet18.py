#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatchAlign — Dual ResNet-18 with Masked Graph Optimal Transport (MGOT)
=======================================================================
Extracts text embeddings for the 5 disease classes + eudermic from the original
PatchAlign embedding file (115 clinical labels). All other components are
identical to the baseline dual ResNet-18 training.

Now supports three training regimes: clin, derm, both.

All outputs saved in:
    - checkpoints_patchalign_resnet18/{clin,derm,both}/
    - results_patchalign_resnet18/{clin,derm,both}/
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
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import f1_score
from models.models_losses import get_layer_wise_lr_params, confusion_loss, skin_type_loss
from models.evaluation import (
    validate, fairness, save_results_csv, plot_confusion_matrix,
    plot_per_class_metrics, plot_fairness_metrics, plot_training_curves,
    plot_tsne, build_loaders, LABEL_NAMES,
)
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
    p_t = p.transpose(1, 2)   # (B, D, N_patches)
    q_t = q.transpose(1, 2)   # (B, D, N_text)

    cos_distance = cost_matrix_batch_torch(p_t, q_t).transpose(1, 2)
    beta = 0.1
    threshold = cos_distance.min() + beta * (cos_distance.max() - cos_distance.min())
    cos_dist = torch.nn.functional.relu(cos_distance - threshold)

    bs, n_p, n_t = cos_dist.size()
    wd, _T = IPOT_distance_torch_batch_uniform(cos_dist, mask, bs, n_p, n_t)
    gwd    = GW_distance_uniform(p_t, q_t, mask)
    return lamb * torch.mean(gwd) + (1.0 - lamb) * torch.mean(wd)

warnings.filterwarnings("ignore")

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser(description="PatchAlign ResNet-18 — modality ablation")
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
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
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

FULL_EMBEDDINGS_PATH = WORK_ROOT / 'text_embeddings_3_large_consecutive_averaged.npy'

CFG = {
    'csv_dir': CSV_DIR,
    'image_roots': IMAGE_ROOTS,
    'ckpt_dir': WORK_ROOT / f'checkpoints_patchalign_resnet18' / TRAIN_MODALITY,
    'results_dir': WORK_ROOT / f'results_patchalign_resnet18' / TRAIN_MODALITY,

    'train_modality': TRAIN_MODALITY,
    'backbone': 'resnet18',
    'embed_dim': 512,
    'img_size': 224,
    'num_classes': 5,
    'num_skin_types': 6,
    'num_text_labels': 6,
    'text_embed_dim': 768,
    'batch_size': 32,
    'num_epochs': 1,        # adjust as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-4,
    'warmup_epochs': 1,      # adjust as needed
    'aug_probability': 0.85,
    'alpha_conf': 0.5,
    'beta_got': 1.0,
    'lamb_got': 0.9,
}
CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

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
    "actinic keratosis":        "actinic keratosis",
    "squamous cell carcinoma":  "squamous cell carcinoma",
    "eudermic":                 "eudermic",
}

# ----------------------------- Model Definition ------------------------------
_RESNET18_FEAT_DIM = 512

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=1024, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
    def forward(self, x):
        if x.size(0) == 1 and self.training:
            self.eval(); out = self.net(x); self.train()
        else:
            out = self.net(x)
        return F.normalize(out, dim=-1)

class PatchAlignResNet18(nn.Module):
    """
    Dual ResNet-18 with PatchAlign losses. Supports train_modality regimes.
    """
    def __init__(self, embed_dim=512, num_classes=5, num_skin_types=6,
                 num_text_labels=6, text_embed_dim=768, pretrained=True,
                 use_projection=False, train_modality='both'):
        super().__init__()
        self.train_modality = train_modality

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        def _make_backbone():
            net = resnet18(weights=weights)
            return nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
        self.clinical_backbone = _make_backbone()
        self.derm_backbone     = _make_backbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = _RESNET18_FEAT_DIM

        self.use_projection = use_projection
        if use_projection:
            self.proj_head = ProjectionHead(feat_dim, 1024, embed_dim)
        else:
            self.proj_head = nn.Identity()
            embed_dim = feat_dim

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types),
        )

        self.patch_proj = nn.Sequential(
            nn.Linear(feat_dim, text_embed_dim),
            nn.LayerNorm(text_embed_dim),
        )
        n_patches = 49
        self.num_text_labels = num_text_labels
        self.mask_net = nn.Sequential(
            nn.Linear(n_patches * text_embed_dim, 256), nn.ReLU(),
            nn.Linear(256, n_patches * num_text_labels), nn.Sigmoid(),
        )
        self._n_patches = n_patches
        self._text_embed_dim = text_embed_dim

        # --- Weight-tying + freezing for single-modality regimes ---
        if train_modality == 'clin':
            self.derm_backbone.load_state_dict(self.clinical_backbone.state_dict())
            for p in self.derm_backbone.parameters():
                p.requires_grad = False
        elif train_modality == 'derm':
            self.clinical_backbone.load_state_dict(self.derm_backbone.state_dict())
            for p in self.clinical_backbone.parameters():
                p.requires_grad = False

    def _extract_features(self, x, modality):
        backbone = self.clinical_backbone if modality == "clinical" else self.derm_backbone
        feat = backbone(x)
        pooled = self.global_pool(feat).flatten(1)
        z = self.proj_head(pooled)
        return z, feat

    def _patch_embeddings(self, feat):
        B, C, H, W = feat.shape
        patches_raw = feat.view(B, C, H*W).permute(0,2,1)   # (B,49,512)
        patches = self.patch_proj(patches_raw)              # (B,49,768)
        mask_flat = self.mask_net(patches.view(B, -1))
        mask = mask_flat.view(B, self._n_patches, self.num_text_labels)
        return patches, mask

    def forward(self, batch):
        device = batch["label"].device
        batch_size = len(batch["label"])

        # Single-modality regimes
        if self.train_modality in ('clin', 'derm'):
            if self.train_modality == 'clin':
                img_t = batch["clinical"].to(device)
                z, feat = self._extract_features(img_t, "clinical")
            else:
                img_t = batch["derm"].to(device)
                z, feat = self._extract_features(img_t, "derm")
            patches, mask = self._patch_embeddings(feat)
            embeddings = z
            out_patches = patches
            out_masks = mask
        else:  # both regime: original logic (use clinical for patches)
            # For paired averaging we still use clinical branch for patches
            if "clinical" in batch:
                clinical_t = batch["clinical"].to(device)
                z, feat = self._extract_features(clinical_t, "clinical")
                patches, mask = self._patch_embeddings(feat)
                embeddings = z
                out_patches = patches
                out_masks = mask
            else:
                # fallback
                img_t = batch["image"].to(device)
                z, feat = self._extract_features(img_t, "clinical")
                patches, mask = self._patch_embeddings(feat)
                embeddings = z
                out_patches = patches
                out_masks = mask

            # If paired, average embeddings from both modalities (but patches remain from clinical)
            paired_mask = torch.tensor(batch.get("paired", [False]*batch_size), dtype=torch.bool, device=device)
            if paired_mask.any() and "clinical" in batch and "derm" in batch:
                clin_t = batch["clinical"][paired_mask].to(device)
                derm_t = batch["derm"][paired_mask].to(device)
                z_c, _ = self._extract_features(clin_t, "clinical")
                z_d, _ = self._extract_features(derm_t, "derm")
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

# ----------------------------- Training Epoch --------------------------------
def train_epoch(model, loader, optimizer, epoch, scaler, device, text_emb, cfg):
    model.train()
    total_loss = loss_c_sum = loss_conf_sum = loss_s_sum = loss_got_sum = 0.0
    all_preds, all_labels = [], []
    n_batches = 0
    criterion_cls = nn.CrossEntropyLoss()

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)
            loss_c = criterion_cls(out["logits"], batch["label"])

            # Extract skin labels
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
                loss_conf = out["logits"].new_tensor(0.)
                loss_s    = out["logits"].new_tensor(0.)
                if epoch == 0 and n_batches == 0:
                    print("[WARN] No skin type column found. L_conf and L_s are zero.")

            # MGOT loss
            loss_got = out["logits"].new_tensor(0.)
            if out["patches"] is not None and text_emb is not None:
                B = out["patches"].size(0)
                text_batch = text_emb.unsqueeze(0).expand(B, -1, -1)
                loss_got = got_loss(out["patches"], text_batch, out["mask"], lamb=cfg['lamb_got'])

            loss = (loss_c
                    + cfg['alpha_conf'] * loss_conf
                    + loss_s
                    + cfg['beta_got'] * loss_got)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loss_c_sum += loss_c.item()
        loss_conf_sum += loss_conf.item()
        loss_s_sum += loss_s.item()
        loss_got_sum += loss_got.item() if isinstance(loss_got, torch.Tensor) else loss_got
        n_batches += 1

        preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())
        pbar.set_postfix(loss=f"{total_loss/n_batches:.4f}", lc=f"{loss_c_sum/n_batches:.3f}")

    pbar.close()
    nb = max(n_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        "total": total_loss/nb, "cls": loss_c_sum/nb, "conf": loss_conf_sum/nb,
        "skin": loss_s_sum/nb, "got": loss_got_sum/nb,
        "acc": (all_preds == all_labels).mean(),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

# ----------------------------- Evaluation helper -----------------------------
def evaluate_test_loaders(model, test_loaders, device, cfg, results_dir, label_names):
    summary = {}
    for mod_name, loader in test_loaders.items():
        split_tag = f"test_{mod_name}"
        print(f"\n── Test on {mod_name} images ──")
        res  = validate(model, loader, device, cfg["num_classes"], desc=f"Test [{mod_name}]")
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

# ----------------------------- Main ------------------------------------------
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoints  : {CFG['ckpt_dir']}")
    print(f"Results      : {CFG['results_dir']}")

    # Load text embeddings
    if FULL_EMBEDDINGS_PATH.exists():
        full_emb = np.load(FULL_EMBEDDINGS_PATH)
        print(f"Loaded full embedding matrix: {full_emb.shape}")
        label_to_idx = {label: idx for idx, label in enumerate(ORIGINAL_LABELS)}
        selected_indices = []
        for our_name, orig_name in CLASS_MAPPING.items():
            if orig_name not in label_to_idx:
                raise ValueError(f"Label '{orig_name}' not found")
            selected_indices.append(label_to_idx[orig_name])
        text_emb_np = full_emb[selected_indices]
        print(f"Selected embeddings: {text_emb_np.shape}")
        text_emb = torch.tensor(text_emb_np, dtype=torch.float32).to(DEVICE)
    else:
        print("[WARN] Text embeddings not found; GOT loss will be zero.")
        text_emb = None

    train_loader, val_loader, test_loaders, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches  : {len(val_loader)}")
    print(f"Test loaders : {list(test_loaders.keys())}")
    print(f"Cross-eval   : {list(eval_loaders.keys())}")

    model = PatchAlignResNet18(
        embed_dim=CFG["embed_dim"], num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"], num_text_labels=CFG["num_text_labels"],
        text_embed_dim=CFG["text_embed_dim"], pretrained=True, use_projection=False,
        train_modality=TRAIN_MODALITY,
    ).to(DEVICE)

    param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=CFG["weight_decay"], betas=(0.9,0.999), eps=1e-8)
    def lr_lambda(epoch):
        if epoch < CFG["warmup_epochs"]:
            return (epoch+1)/CFG["warmup_epochs"]
        progress = (epoch - CFG["warmup_epochs"]) / max(1, CFG["num_epochs"] - CFG["warmup_epochs"])
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return max(CFG["min_lr"]/CFG["lr"], cos)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_auroc = 0.0
    best_f1 = 0.0
    history = defaultdict(list)

    for epoch in range(CFG["num_epochs"]):
        train_metrics = train_epoch(model, train_loader, optimizer, epoch, scaler, DEVICE, text_emb, CFG)
        scheduler.step()
        val_metrics = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation")
        lr = optimizer.param_groups[0]["lr"]

        for k,v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k in ["acc","auroc","macro_f1","weighted_f1"]:
            history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))
        print(f"Ep {epoch+1:3d}/{CFG['num_epochs']}  loss={train_metrics['total']:.4f}  "
              f"(cls={train_metrics['cls']:.3f} conf={train_metrics['conf']:.3f} "
              f"skin={train_metrics['skin']:.3f} got={train_metrics['got']:.3f})  "
              f"tr_acc={train_metrics['acc']:.4f}  val_acc={val_metrics['acc']:.4f}  "
              f"val_auroc={val_metrics['auroc']:.4f}  val_f1={val_metrics['macro_f1']:.4f}  lr={lr:.2e}")

        ckpt_state = {"epoch":epoch, "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                      "scheduler":scheduler.state_dict(), "history":dict(history)}
        torch.save(ckpt_state, CFG["ckpt_dir"]/"last_model.pt")
        if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            shutil.copy(CFG["ckpt_dir"]/"last_model.pt", CFG["ckpt_dir"]/"best_auroc_model.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"]/"last_model.pt", CFG["ckpt_dir"]/"best_f1_model.pt")
        if (epoch+1)%5 ==0:
            torch.save(ckpt_state, CFG["ckpt_dir"]/f"checkpoint_ep{epoch+1:03d}.pt")
        with open(CFG["results_dir"]/"history.json","w") as f:
            json.dump({k:[float(x) for x in v] for k,v in history.items()}, f, indent=2)

    print(f"Training complete. Best AUROC: {best_auroc:.4f}, Best F1: {best_f1:.4f}")

    best_ckpt = CFG["ckpt_dir"]/"best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"]/"best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"]/"last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation", CFG["results_dir"]/"val_confusion.png")
    plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation", CFG["results_dir"]/"val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"]/"val_fairness.png")

    test_summary = evaluate_test_loaders(
        model, test_loaders, DEVICE, CFG, CFG["results_dir"], LABEL_NAMES)
    print("\nTest summary:")
    for mod_name, metrics in test_summary.items():
        print(f"  [{mod_name}]  acc={metrics['accuracy']:.4f}  "
              f"auroc={metrics['auroc']:.4f}  f1={metrics['macro_f1']:.4f}  "
              f"EOM={metrics['EOM']:.4f}  PQD={metrics['PQD']:.4f}")
    if test_summary:
        pd.DataFrame(test_summary).T.to_csv(CFG["results_dir"] / "test_modality_summary.csv")

    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        print(f"\nEvaluating on {ds_name}")
        res = validate(model, loader, DEVICE, CFG["num_classes"], desc=f"Cross-eval: {ds_name}")
        fair = fairness(res)
        save_results_csv(res, fair, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              f"Confusion Matrix - {ds_name}", CFG["results_dir"]/f"cross_{ds_name}_confusion.png")
        plot_per_class_metrics(res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               f"Per-Class Metrics - {ds_name}", CFG["results_dir"]/f"cross_{ds_name}_per_class.png")
        plot_fairness_metrics(fair, f"Fairness - {ds_name}", CFG["results_dir"]/f"cross_{ds_name}_fairness.png")
        cross_results[ds_name] = {
            "accuracy": res["acc"], "precision": res["macro_prec"], "recall": res["macro_rec"],
            "auroc": res["auroc"], "macro_f1": res["macro_f1"], "micro_f1": res["micro_f1"],
            "weighted_f1": res["weighted_f1"],
            "EOM": fair["EOM"], "PQD": fair["PQD"], "DPM": fair["DPM"],
        }
    if cross_results:
        pd.DataFrame(cross_results).T.to_csv(CFG["results_dir"]/"cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", pd.DataFrame(cross_results).T)

    plot_training_curves(history, f"Training History (PatchAlign ResNet-18 [{TRAIN_MODALITY}])",
                         CFG["results_dir"]/"training_curves.png")

    if 'clin' in test_loaders:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loaders['clin']:
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
        embs = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        plot_tsne(embs, labels_tsne,
                  f"t-SNE — Test [clin] (PatchAlign ResNet-18, {TRAIN_MODALITY} trained)",
                  CFG["results_dir"]/"tsne_test_clin.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")

if __name__ == "__main__":
    main()