"""
models_losses.py

Defines:
- Dual ResNet-18 encoder (with projection head for modality invariance)
- Dual ViT encoder (with projection head)
- Loss functions: SupConLoss, confusion_loss, skin_type_loss, mi_loss,
  cls_loss_fn (label-smoothed weighted CE), mixup_embeddings.
- Helper to compute class weights.
- get_layer_wise_lr_params (for differential learning rates)
"""

from torch.nn.modules import batchnorm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
import timm
from pathlib import Path


# ------------------------------------------------------------
# Shared constants and helpers
# ------------------------------------------------------------
_RESNET18_FEAT_DIM = 512
_VIT_SMALL_FEAT_DIM = 384  # vit_small_patch16_224


def compute_class_weights(csv_dir, num_classes=5):
    """Compute effective number class weights from all training CSVs."""
    try:
        csv_dir = Path(csv_dir)
        dfs = []
        for p in csv_dir.glob("*_train*.csv"):
            try:
                dfs.append(pd.read_csv(p, usecols=["label"]))
            except Exception:
                pass
        if not dfs:
            return None
        labels = pd.concat(dfs)["label"].astype(int)
        counts = labels.value_counts().sort_index()
        freqs = np.array([counts.get(i, 1) for i in range(num_classes)], dtype=float)
        beta = 0.9999
        eff_num = (1.0 - beta ** freqs) / (1.0 - beta)
        weights = 1.0 / eff_num
        weights /= weights.sum() / num_classes
        return weights.tolist()
    except Exception as e:
        print(f"[WARN] Could not compute class weights: {e}")
        return None


def get_layer_wise_lr_params(model, base_lr=1e-4, lr_decay=0.85):
    """
    Return parameter groups with decreasing learning rates for earlier layers.
    Decay is applied per sequential block.
    """
    params = []
    # Collect backbone parameters (clinical and derm share same structure)
    backbone_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'skin_clf' in name or 'proj_head' in name:
            continue
        backbone_params.append(param)

    # Group backbone layers by depth
    # This is a simplified version: all backbone get same LR = base_lr * lr_decay
    # For more granular decay, one could iterate over the block names.
    params.append({'params': backbone_params, 'lr': base_lr * lr_decay})

    # Classifier heads get full learning rate
    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'skin_clf' in name or 'proj_head' in name:
            classifier_params.append(param)
    params.append({'params': classifier_params, 'lr': base_lr})

    return params


# ------------------------------------------------------------
# Projection head (for contrastive learning)
# ------------------------------------------------------------
class ProjectionHead(nn.Module):
    """3-layer MLP projection head with BN + GELU."""
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
            self.eval()
            out = self.net(x)
            self.train()
        else:
            out = self.net(x)
        return F.normalize(out, dim=-1)


# ------------------------------------------------------------
# Dual ResNet-18 Model (with projection head)
# ------------------------------------------------------------
class DualResNet18(nn.Module):
    """
    Dual ResNet-18 encoder with optional projection head.
    For baseline (no contrastive loss), the projection head is still present
    but not used in loss. For FairDisCo, projection outputs are used in L_con.
    """
    def __init__(self, embed_dim, num_classes, num_skin_types, pretrained=True, use_projection=True):
        super().__init__()
        # Load ResNet-18 backbone without final FC
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.clinical_backbone = resnet18(weights=weights)
        self.derm_backbone = resnet18(weights=weights)
        self.clinical_backbone.fc = nn.Identity()
        self.derm_backbone.fc = nn.Identity()

        feat_dim = _RESNET18_FEAT_DIM
        self.use_projection = use_projection
        if use_projection:
            self.proj_head = ProjectionHead(feat_dim, 1024, embed_dim)
        else:
            self.proj_head = nn.Identity()
            embed_dim = feat_dim

        # Classifier and skin classifier
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types)
        )

    def encode(self, x, modality):
        if modality == "clinical":
            f = self.clinical_backbone(x)
        else:
            f = self.derm_backbone(x)
        return f, self.proj_head(f)

    def forward(self, batch):
        device = batch["label"].device
        batch_size = len(batch["label"])
        embed_dim = self.classifier[1].in_features
        # Initialize embeddings as None; we will create it later with correct dtype
        embeddings = None

        paired_mask = torch.tensor(batch["paired"], dtype=torch.bool, device=device)
        unpaired_mask = ~paired_mask

        out = {}

        # Paired samples
        if paired_mask.any() and "clinical" in batch and "derm" in batch:
            clin_t = batch["clinical"][paired_mask].to(device)
            derm_t = batch["derm"][paired_mask].to(device)
            _, z_c = self.encode(clin_t, "clinical")
            _, z_d = self.encode(derm_t, "derm")
            z_paired = (z_c + z_d) / 2

            # Create embeddings tensor using dtype from z_paired
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z_paired.size(-1), device=device, dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            out["z_c"] = z_c
            out["z_d"] = z_d

        # Unpaired samples
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            _, z = self.encode(img_t, "clinical")
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z.size(-1), device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z

        # Fallback (should never happen)
        if embeddings is None:
            embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device)

        out["z"] = embeddings
        out["logits"] = self.classifier(out["z"])
        out["skin_logits"] = self.skin_clf(out["z"])
        return out


# ------------------------------------------------------------
# Dual ViT Model (with projection head)
# ------------------------------------------------------------
class DualViT(nn.Module):
    """
    Dual ViT-small encoder with optional projection head.
    """
    def __init__(self, embed_dim, num_classes, num_skin_types, pretrained=True, use_projection=True):
        super().__init__()
        vit_name = "vit_small_patch16_224"
        self.clinical_vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.derm_vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        feat_dim = _VIT_SMALL_FEAT_DIM

        self.use_projection = use_projection
        if use_projection:
            self.proj_head = ProjectionHead(feat_dim, 1024, embed_dim)
        else:
            self.proj_head = nn.Identity()
            embed_dim = feat_dim

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types)
        )

    def encode(self, x, modality):
        if modality == "clinical":
            f = self.clinical_vit(x)
        else:
            f = self.derm_vit(x)
        return f, self.proj_head(f)

    def forward(self, batch):
        device = batch["label"].device
        batch_size = len(batch["label"])
        embed_dim = self.classifier[1].in_features
        # Initialize embeddings as None; we will create it later with correct dtype
        embeddings = None

        paired_mask = torch.tensor(batch["paired"], dtype=torch.bool, device=device)
        unpaired_mask = ~paired_mask

        out = {}

        # Paired samples
        if paired_mask.any() and "clinical" in batch and "derm" in batch:
            clin_t = batch["clinical"][paired_mask].to(device)
            derm_t = batch["derm"][paired_mask].to(device)
            _, z_c = self.encode(clin_t, "clinical")
            _, z_d = self.encode(derm_t, "derm")
            z_paired = (z_c + z_d) / 2

            # Create embeddings tensor using dtype from z_paired
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z_paired.size(-1), device=device, dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            out["z_c"] = z_c
            out["z_d"] = z_d

        # Unpaired samples
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            _, z = self.encode(img_t, "clinical")
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z.size(-1), device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z

        # Fallback (should never happen)
        if embeddings is None:
            embeddings = torch.zeros(batch_size, self.classifier[1].in_features, device=device)

        out["z"] = embeddings
        out["logits"] = self.classifier(out["z"])
        out["skin_logits"] = self.skin_clf(out["z"])
        return out


# ------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------
class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.T = temperature

    def forward(self, projections, targets):
        B = projections.size(0)
        if B < 2:
            return projections.new_tensor(0.)
        dot = torch.mm(projections, projections.T) / self.T
        dot_max, _ = dot.max(dim=1, keepdim=True)
        exp_dot = torch.exp(dot - dot_max.detach()) + 1e-5
        mask_pos = (targets.unsqueeze(1) == targets.unsqueeze(0))
        mask_no_diag = ~torch.eye(B, dtype=torch.bool, device=dot.device)
        mask_neg = (~mask_pos) & mask_no_diag
        mask_combined = mask_pos & mask_no_diag
        neg_sum = (exp_dot * mask_neg).sum(dim=1, keepdim=True)
        log_prob = (dot - dot_max.detach()) - torch.log(exp_dot + neg_sum + 1e-5)
        cardinality = mask_combined.sum(dim=1).float()
        has_pos = cardinality > 0
        if not has_pos.any():
            return projections.new_tensor(0.)
        per_anchor = (log_prob * mask_combined).sum(dim=1)
        loss = -(per_anchor[has_pos] / cardinality[has_pos]).mean()
        return loss


def confusion_loss(skin_logits):
    """Confusion loss: encourage uniform skin type predictions."""
    log_p = F.log_softmax(skin_logits, dim=1)
    return -log_p.mean()


def skin_type_loss(skin_logits_detached, skin_labels):
    """Standard CE on detached embeddings (only skin_clf is updated)."""
    valid = skin_labels >= 0
    if valid.sum() == 0:
        return skin_logits_detached.new_tensor(0.)
    return F.cross_entropy(skin_logits_detached[valid], skin_labels[valid])


def mi_loss(z_c, z_d):
    """Modality invariance loss: cosine + MSE."""
    cos_part = (1.0 - F.cosine_similarity(z_c, z_d, dim=-1)).mean()
    mse_part = F.mse_loss(z_c, z_d)
    return 0.7 * cos_part + 0.3 * mse_part


def mixup_embeddings(z, labels, alpha=0.4):
    """Manifold mixup on embeddings."""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1.0 - lam)
    idx = torch.randperm(z.size(0), device=z.device)
    return lam * z + (1.0 - lam) * z[idx], labels, labels[idx], lam


def cls_loss_fn(logits, targets, weight_tensor=None, smoothing=0.1):
    """Label-smoothed weighted cross-entropy."""
    n = logits.size(1)
    log_probs = F.log_softmax(logits, dim=-1)
    hard = F.nll_loss(log_probs, targets, weight=weight_tensor, reduction="mean")
    smooth_t = torch.full_like(log_probs, smoothing / (n - 1))
    smooth_t.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing)
    soft = -(smooth_t * log_probs).sum(dim=-1).mean()
    return (1.0 - smoothing) * hard + smoothing * soft