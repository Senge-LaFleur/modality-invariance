#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PatchAlign — Dual ResNet-18 with Masked Graph Optimal Transport (MGOT)
=======================================================================
Extends train_BASE_resnet18.py with PatchAlign losses:
  L = L_c + α·L_conf + L_s + β·L_got
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
from torchvision.models import resnet18, ResNet18_Weights

from sklearn.metrics import f1_score
from models.models_losses import get_layer_wise_lr_params
from models.evaluation import (
    validate, fairness, save_results_csv, plot_confusion_matrix,
    plot_per_class_metrics, plot_fairness_metrics, plot_training_curves,
    plot_tsne, build_loaders, LABEL_NAMES,
)
from models.got_losses import Confusion_Loss
from Masked_GOT_NewSinkhorn import got_loss   # <-- import from your existing module

warnings.filterwarnings("ignore")

# ----------------------------- Configuration ---------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED); torch.backends.cudnn.deterministic = True
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
TEXT_EMBEDDINGS_PATH = WORK_ROOT / 'text_embeddings_3_large_consecutive_averaged.npy'

CFG = {
    'csv_dir': CSV_DIR, 'image_roots': IMAGE_ROOTS,
    'ckpt_dir': WORK_ROOT / 'checkpoints_patchalign_resnet18',
    'results_dir': WORK_ROOT / 'results_patchalign_resnet18',
    'backbone': 'resnet18', 'embed_dim': 512, 'img_size': 224,
    'num_classes': 5, 'num_skin_types': 6, 'num_text_labels': 6,
    'text_embed_dim': 768,
    'batch_size': 32, 'num_epochs': 20, 'lr': 1e-4, 'min_lr': 1e-6,
    'weight_decay': 1e-4, 'warmup_epochs': 4, 'aug_probability': 0.85,
    'alpha_conf': 0.5, 'beta_got': 1.0, 'lamb_got': 0.9,
}
CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)

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
    def __init__(self, embed_dim=512, num_classes=5, num_skin_types=6,
                 num_text_labels=6, text_embed_dim=768, pretrained=True, use_projection=False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        def _make_backbone():
            net = resnet18(weights=weights)
            return nn.Sequential(
                net.conv1, net.bn1, net.relu, net.maxpool,
                net.layer1, net.layer2, net.layer3, net.layer4,
            )
        self.clinical_backbone = _make_backbone()
        self.derm_backbone = _make_backbone()
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
        # PatchAlign components
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

    def _extract_features(self, x, modality):
        backbone = self.clinical_backbone if modality == "clinical" else self.derm_backbone
        feat = backbone(x)                     # (B,512,7,7)
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
        paired_mask = torch.tensor(batch["paired"], dtype=torch.bool, device=device)
        unpaired_mask = ~paired_mask
        embeddings = None
        patches_list = []

        # Paired samples
        if paired_mask.any() and "clinical" in batch and "derm" in batch:
            clin_t = batch["clinical"][paired_mask].to(device)
            derm_t = batch["derm"][paired_mask].to(device)
            z_c, feat_c = self._extract_features(clin_t, "clinical")
            z_d, feat_d = self._extract_features(derm_t, "derm")
            z_paired = (z_c + z_d) / 2
            if embeddings is None:
                embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device, dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            p_c, m_c = self._patch_embeddings(feat_c)
            p_d, m_d = self._patch_embeddings(feat_d)
            fused_patches = torch.cat([p_c, p_d], dim=1)
            fused_mask = torch.cat([m_c, m_d], dim=1)
            patches_list.append((paired_mask, fused_patches, fused_mask))

        # Unpaired samples
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            z, feat = self._extract_features(img_t, "clinical")
            if embeddings is None:
                embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z
            patches, mask = self._patch_embeddings(feat)
            patches_list.append((unpaired_mask, patches, mask))

        if embeddings is None:
            embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device)

        # Assemble full batch patches/masks
        if patches_list:
            n_patches = patches_list[0][1].size(1)
            text_d = patches_list[0][1].size(2)
            n_text = patches_list[0][2].size(2)
            all_patches = torch.zeros(batch_size, n_patches, text_d, device=device, dtype=patches_list[0][1].dtype)
            all_masks = torch.zeros(batch_size, n_patches, n_text, device=device, dtype=patches_list[0][2].dtype)
            for sel_mask, p, m in patches_list:
                all_patches[sel_mask] = p
                all_masks[sel_mask] = m
            out_patches = all_patches
            out_masks = all_masks
        else:
            out_patches = None
            out_masks = None

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
    criterion_skin = nn.CrossEntropyLoss(ignore_index=-1)
    confusion_loss = ConfusionLoss()   # from models.got_losses

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)
            loss_c = criterion_cls(out["logits"], batch["label"])
            skin_labels = batch.get("fitzpatrick", None)
            if skin_labels is not None:
                skin_labels = skin_labels.long()
                loss_conf = confusion_loss(out["skin_logits"], skin_labels)
                loss_s = criterion_skin(out["skin_logits"], skin_labels)
            else:
                loss_conf = out["logits"].new_tensor(0.)
                loss_s = out["logits"].new_tensor(0.)
            loss_got = out["logits"].new_tensor(0.)
            if out["patches"] is not None and text_emb is not None:
                B = out["patches"].size(0)
                text_batch = text_emb.unsqueeze(0).expand(B, -1, -1)
                loss_got = got_loss(out["patches"], text_batch, out["mask"], lamb=cfg['lamb_got'])
            loss = loss_c + cfg['alpha_conf'] * loss_conf + loss_s + cfg['beta_got'] * loss_got

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

# ----------------------------- Main ------------------------------------------
def main():
    print(f"CSV dir: {CFG['csv_dir']}")
    print(f"Checkpoints: {CFG['ckpt_dir']}")
    print(f"Results: {CFG['results_dir']}")

    # Load text embeddings
    if TEXT_EMBEDDINGS_PATH.exists():
        text_emb = torch.tensor(np.load(TEXT_EMBEDDINGS_PATH), dtype=torch.float32).to(DEVICE)
        print(f"Loaded text embeddings: {text_emb.shape}")
    else:
        print("[WARN] Text embeddings not found; GOT loss will be zero.")
        text_emb = None

    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    model = PatchAlignResNet18(
        embed_dim=CFG["embed_dim"], num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"], num_text_labels=CFG["num_text_labels"],
        text_embed_dim=CFG["text_embed_dim"], pretrained=True, use_projection=False,
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

    best_auroc = best_f1 = 0.0
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

    # Load best model and evaluate
    best_ckpt = CFG["ckpt_dir"]/"best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"]/"best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"]/"last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    # Standard validation, test, cross-dataset evaluation, plotting...
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"], desc="Validation (final)")
    val_fair = fairness(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                          "Confusion Matrix - Validation", CFG["results_dir"]/"val_confusion.png")
    plot_per_class_metrics(val_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                           "Per-Class Metrics - Validation", CFG["results_dir"]/"val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation", CFG["results_dir"]/"val_fairness.png")

    if test_loader:
        test_res = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        test_fair = fairness(test_res)
        save_results_csv(test_res, test_fair, "test", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(test_res["conf_mat"], [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                              "Confusion Matrix - Test", CFG["results_dir"]/"test_confusion.png")
        plot_per_class_metrics(test_res, [LABEL_NAMES[i] for i in range(CFG["num_classes"])],
                               "Per-Class Metrics - Test", CFG["results_dir"]/"test_per_class.png")
        plot_fairness_metrics(test_fair, "Fairness - Test", CFG["results_dir"]/"test_fairness.png")

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
        cross_results[ds_name] = {"accuracy": res["acc"], "auroc": res["auroc"], "macro_f1": res["macro_f1"],
                                  "EOM": fair["EOM"], "PQD": fair["PQD"], "DPM": fair["DPM"]}
    if cross_results:
        pd.DataFrame(cross_results).T.to_csv(CFG["results_dir"]/"cross_dataset_summary.csv")
        print("\nCross-dataset summary:\n", pd.DataFrame(cross_results).T)

    plot_training_curves(history, "Training History (PatchAlign ResNet-18)", CFG["results_dir"]/"training_curves.png")

    if test_loader:
        model.eval()
        all_embs, all_labels_tsne = [], []
        with torch.no_grad():
            for batch in test_loader:
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
        embs = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        plot_tsne(embs, labels_tsne, "t-SNE - Test Set (PatchAlign ResNet-18)", CFG["results_dir"]/"tsne_test.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")

if __name__ == "__main__":
    main()