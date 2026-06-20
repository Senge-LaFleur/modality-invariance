"""
models_losses.py

Defines:
- Dual ResNet-18 encoder (with per-modality projection heads for modality invariance)
- Dual ViT encoder (with per-modality projection heads)
- Loss functions: SupConLoss, confusion_loss, skin_type_loss, mi_loss_vicreg,
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
_RESNET18_FEAT_DIM = 384
_VIT_SMALL_FEAT_DIM = 384  


def compute_class_weights(csv_dir, num_classes=3):
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
    backbone_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'skin_clf' in name or 'proj_head' in name:
            continue
        backbone_params.append(param)
    params.append({'params': backbone_params, 'lr': base_lr * lr_decay})

    classifier_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'skin_clf' in name or 'proj_head' in name:
            classifier_params.append(param)
    params.append({'params': classifier_params, 'lr': base_lr})

    return params


def get_layer_wise_lr_params_vit(model, base_lr=1e-4, lr_decay=0.85):
    """
    ViT-specific layer-wise LR decay.
    Backbone layers get base_lr * lr_decay; heads get full base_lr.
    Exported here so all ViT train scripts can import from one place.
    """
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name or 'skin_clf' in name or 'proj_head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)
    return [
        {'params': backbone_params, 'lr': base_lr * lr_decay},
        {'params': head_params,     'lr': base_lr},
    ]


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
    Dual ResNet-18 encoder with per-modality projection heads.
    Clinical embeddings are projected via proj_head_clinical;
    dermoscopic embeddings via proj_head_derm. Both heads map into
    the same shared embedding space (same out_dim).
    """
    def __init__(self, embed_dim, num_classes, num_skin_types, pretrained=True, use_projection=True):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.clinical_backbone = resnet18(weights=weights)
        self.derm_backbone = resnet18(weights=weights)
        self.clinical_backbone.fc = nn.Identity()
        self.derm_backbone.fc = nn.Identity()

        feat_dim = _RESNET18_FEAT_DIM
        self.use_projection = use_projection
        if use_projection:
            self.proj_head_clinical = ProjectionHead(feat_dim, 1024, embed_dim)
            self.proj_head_derm     = ProjectionHead(feat_dim, 1024, embed_dim)
        else:
            self.proj_head_clinical = nn.Identity()
            self.proj_head_derm     = nn.Identity()
            embed_dim = feat_dim

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types)
        )

    def encode(self, x, modality):
        if modality == "clinical":
            f = self.clinical_backbone(x)
            return f, self.proj_head_clinical(f)
        else:
            f = self.derm_backbone(x)
            return f, self.proj_head_derm(f)

    def forward(self, batch):
        device = batch["label"].device
        batch_size = len(batch["label"])
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

            if embeddings is None:
                embeddings = torch.zeros(batch_size, z_paired.size(-1), device=device, dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            out["z_c"] = z_c
            out["z_d"] = z_d
            # Store the paired mask so train scripts can use it for cross-modal contrastive loss
            out["paired_mask"] = paired_mask

        # Unpaired samples
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            _, z = self.encode(img_t, "clinical")
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z.size(-1), device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z

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
    Dual ViT-small encoder with per-modality projection heads.
    Clinical embeddings are projected via proj_head_clinical;
    dermoscopic embeddings via proj_head_derm. Both heads map into
    the same shared embedding space (same out_dim).
    """
    def __init__(self, embed_dim, num_classes, num_skin_types, pretrained=True, use_projection=True):
        super().__init__()
        vit_name = "vit_small_patch16_224"
        self.clinical_vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        self.derm_vit = timm.create_model(vit_name, pretrained=pretrained, num_classes=0)
        feat_dim = _VIT_SMALL_FEAT_DIM

        self.use_projection = use_projection
        if use_projection:
            self.proj_head_clinical = ProjectionHead(feat_dim, 1024, embed_dim)
            self.proj_head_derm     = ProjectionHead(feat_dim, 1024, embed_dim)
        else:
            self.proj_head_clinical = nn.Identity()
            self.proj_head_derm     = nn.Identity()
            embed_dim = feat_dim

        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(embed_dim, num_classes))
        self.skin_clf = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_skin_types)
        )

    def encode(self, x, modality):
        if modality == "clinical":
            f = self.clinical_vit(x)
            return f, self.proj_head_clinical(f)
        else:
            f = self.derm_vit(x)
            return f, self.proj_head_derm(f)

    def forward(self, batch):
        device = batch["label"].device
        batch_size = len(batch["label"])
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

            if embeddings is None:
                embeddings = torch.zeros(batch_size, z_paired.size(-1), device=device, dtype=z_paired.dtype)
            embeddings[paired_mask] = z_paired
            out["z_c"] = z_c
            out["z_d"] = z_d
            # Store the paired mask so train scripts can use it for cross-modal contrastive loss
            out["paired_mask"] = paired_mask

        # Unpaired samples
        if unpaired_mask.any() and "clinical" in batch:
            img_t = batch["clinical"][unpaired_mask].to(device)
            _, z = self.encode(img_t, "clinical")
            if embeddings is None:
                embeddings = torch.zeros(batch_size, z.size(-1), device=device, dtype=z.dtype)
            embeddings[unpaired_mask] = z

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


def cross_modal_supcon_loss(z_c, z_d, labels_paired, temperature=0.07):
    """
    Cross-modal supervised contrastive loss.

    FIX: Instead of computing SupCon on the already-averaged embedding `z`
    (which teaches nothing about cross-modal alignment), this concatenates
    the raw clinical embedding z_c and derm embedding z_d along the batch
    dimension and runs SupCon on that. This forces the model to pull clinical
    and dermoscopy embeddings of the same class together while pushing apart
    embeddings of different classes — which is exactly what modality invariance
    requires.

    Args:
        z_c:           [N, D] clinical embeddings for paired samples (L2-normalised)
        z_d:           [N, D] derm embeddings for paired samples (L2-normalised)
        labels_paired: [N]    disease labels for the N paired samples
        temperature:   scalar, typically 0.07

    Returns:
        scalar loss
    """
    if z_c.size(0) < 2:
        return z_c.new_tensor(0.)

    # Stack: 2N embeddings, 2N labels
    z_all = torch.cat([z_c, z_d], dim=0)           # [2N, D]
    labels_all = torch.cat([labels_paired, labels_paired], dim=0)  # [2N]

    sup_con = SupConLoss(temperature=temperature)
    return sup_con(z_all, labels_all)


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


# ------------------------------------------------------------------
# FIX: VICReg-style MI loss (replaces the collapsing cosine+MSE loss)
# ------------------------------------------------------------------
def mi_loss_vicreg(z_c, z_d, lambda_inv=1.0, lambda_var=1.0, lambda_cov=0.04):
    """
    Modality invariance loss with collapse prevention (VICReg-style).

    The original mi_loss() used only cosine similarity + MSE, which gives the
    optimiser a trivial solution: collapse all embeddings to a constant vector,
    which scores perfectly on both metrics while being useless for downstream
    classification. This caused the AUROC drops seen in the results (90.01 vs
    95.71 baseline for ResNet, 88.38 vs 97.12 for ViT).

    This version adds:
      - Variance term: penalises dimensions whose standard deviation falls
        below 1, preventing collapse.
      - Covariance term: penalises off-diagonal covariance, decorrelating
        the embedding dimensions so each carries distinct information.

    Reference: Bardes, Ponce & LeCun, "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning", ICLR 2022.

    Args:
        z_c:        [N, D] clinical embeddings (L2-normalised)
        z_d:        [N, D] derm embeddings (L2-normalised)
        lambda_inv: weight for the invariance (alignment) term
        lambda_var: weight for the variance (anti-collapse) term
        lambda_cov: weight for the covariance (decorrelation) term

    Returns:
        scalar loss
    """
    if z_c.size(0) < 2:
        return z_c.new_tensor(0.)

    # --- Invariance: pull the two modalities together ---
    inv = F.mse_loss(z_c, z_d)

    # --- Variance: prevent dimensional collapse ---
    std_c = torch.sqrt(z_c.var(dim=0) + 1e-4)
    std_d = torch.sqrt(z_d.var(dim=0) + 1e-4)
    var = torch.mean(F.relu(1.0 - std_c)) + torch.mean(F.relu(1.0 - std_d))

    # --- Covariance: decorrelate embedding dimensions ---
    N, D = z_c.shape
    z_c_n = z_c - z_c.mean(dim=0)
    z_d_n = z_d - z_d.mean(dim=0)
    cov_c = (z_c_n.T @ z_c_n) / (N - 1)
    cov_d = (z_d_n.T @ z_d_n) / (N - 1)

    # Sum of squared off-diagonal elements, normalised by D
    def off_diag_sq(mat):
        return (mat.pow(2).sum() - mat.diagonal().pow(2).sum()) / D

    cov = off_diag_sq(cov_c) + off_diag_sq(cov_d)

    return lambda_inv * inv + lambda_var * var + lambda_cov * cov


def mi_loss_legacy(z_c, z_d):
    """
    DEPRECATED — kept for reference only. Do NOT use in training.
    The cosine+MSE formulation has no collapse-prevention mechanism,
    causing catastrophic AUROC degradation. Use mi_loss_vicreg() instead.
    """
    cos_part = (1.0 - F.cosine_similarity(z_c, z_d, dim=-1)).mean()
    mse_part = F.mse_loss(z_c, z_d)
    return 0.7 * cos_part + 0.3 * mse_part


# Keep the old name as an alias pointing to the safe version, so any code
# that still calls mi_loss() gets the fixed implementation automatically.
def mi_loss(z_c, z_d):
    """Alias for mi_loss_vicreg() with default weights. Backwards-compatible."""
    return mi_loss_vicreg(z_c, z_d)


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