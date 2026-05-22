#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SNNL_SCP_resnet18.py
Compute Soft Nearest Neighbor Loss (SNNL) per channel for both encoders of DualResNet18.
Identifies sensitive channels (lowest SNNL) that will be pruned.

Usage:
    python SNNL_SCP_resnet18.py --checkpoint path/to/model.pt --prune_ratio 0.02

Outputs (JSON scores) are saved to:
    /kaggle/working/modality-invariance/process/process/outputs/snnl_scores/

Note: All model checkpoints (e.g., from train_FairPruneSCP_resnet18.py) should be placed in directories
that are listed in .gitignore to avoid committing large binary files.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from models.models_losses import DualResNet18
from models.evaluation import build_loaders

# ------------------------------------------------------------
# Fixed working root (same as in train_BASE_resnet18.py)
# ------------------------------------------------------------
WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
SNNL_OUTPUT_DIR = WORK_ROOT / 'snnl_scores'

# ------------------------------------------------------------
# SNNL computation (corrected from provided SNNL.py)
# ------------------------------------------------------------
class SoftNearestNeighborLoss(nn.Module):
    """SNNL for measuring entanglement of sensitive attributes."""
    def __init__(self, temperature=1.0, eps=1e-8):
        super().__init__()
        self.temperature = temperature
        self.eps = eps

    def forward(self, embeddings, labels):
        """
        embeddings: (batch_size, feature_dim)
        labels: (batch_size,) – sensitive attribute (0/1)
        """
        batch_size = embeddings.shape[0]
        # Cosine distance
        emb_norm = nn.functional.normalize(embeddings, dim=1)
        pairwise_cos = 1 - torch.mm(emb_norm, emb_norm.t())
        exp_neg = torch.exp(-pairwise_cos / self.temperature)

        # Mask for same sensitive group
        same_group = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # Exclude diagonal
        eye = torch.eye(batch_size, device=embeddings.device)
        mask_same = same_group - eye
        mask_all = 1 - eye

        same_sum = (exp_neg * mask_same).sum(dim=1)
        all_sum = (exp_neg * mask_all).sum(dim=1) + self.eps
        loss = -torch.log((same_sum + self.eps) / all_sum).mean()
        return loss

def compute_channel_snnl(feature_maps, sensitive_labels, temperature=1.0):
    """
    feature_maps: (batch, C, H, W)
    sensitive_labels: (batch,)
    Returns: scores (C,) – lower score = more sensitive (more biased)
    """
    device = feature_maps.device
    batch, C, H, W = feature_maps.shape
    scores = []
    snnl_loss = SoftNearestNeighborLoss(temperature=temperature)
    for c in range(C):
        # Take channel c, flatten spatial dims -> (batch, H*W)
        feat_c = feature_maps[:, c, :, :].reshape(batch, -1)
        if feat_c.shape[1] == 0:
            scores.append(0.0)
            continue
        loss_val = snnl_loss(feat_c, sensitive_labels)
        scores.append(loss_val.item())
    return torch.tensor(scores, device=device)

def get_last_conv_layer(model, encoder_name):
    """Retrieve the last convolutional layer of a ResNet encoder."""
    encoder = getattr(model, encoder_name)
    # ResNet: after layer4, before avgpool. Typically layer4[-1].conv2
    return encoder.layer4[-1].conv2

def register_hook(layer, outputs_dict, key):
    """Forward hook to store output of a layer."""
    def hook(module, input, output):
        outputs_dict[key] = output.detach()
    return layer.register_forward_hook(hook)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to baseline model .pt')
    parser.add_argument('--prune_ratio', type=float, default=0.02, help='Fraction of channels to prune (lowest SNNL)')
    parser.add_argument('--output_dir', type=str, default=str(SNNL_OUTPUT_DIR),
                        help=f'Where to save JSON files (default: {SNNL_OUTPUT_DIR})')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for SNNL')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for SNNL computation')
    parser.add_argument('--sensitive_attr', type=str, default='skin_type_binary',
                        help='Key in dataset for sensitive attribute (skin_type_binary or gender)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = DualResNet18(
        embed_dim=512,
        num_classes=5,
        num_skin_types=6,
        pretrained=False,
        use_projection=False,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Identify last conv layers
    clinical_conv = get_last_conv_layer(model, 'clinical_encoder')
    derm_conv = get_last_conv_layer(model, 'dermoscopic_encoder')
    print(f"Clinical last conv: {clinical_conv} (out_channels={clinical_conv.out_channels})")
    print(f"Dermoscopic last conv: {derm_conv} (out_channels={derm_conv.out_channels})")

    # Build data loaders (only need training set for SNNL)
    # Reuse CFG from base script – you may need to adjust paths
    from train_BASE_resnet18 import CFG  # or copy the config here
    CFG['batch_size'] = args.batch_size
    train_loader, _, _, _ = build_loaders(CFG, seed=42)  # only train_loader needed

    # Prepare storage
    clinical_scores = []
    derm_scores = []

    # Forward hooks to capture layer outputs
    clinical_output = {}
    derm_output = {}
    hook_clin = register_hook(clinical_conv, clinical_output, 'out')
    hook_derm = register_hook(derm_conv, derm_output, 'out')

    print("Computing SNNL scores over training set...")
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Batches"):
            # Move to device
            clinical = batch['clinical'].to(device)
            dermoscopic = batch['dermoscopic'].to(device)
            sensitive = batch[args.sensitive_attr].to(device)

            # Forward pass (hooks capture feature maps)
            _ = model({'clinical': clinical, 'dermoscopic': dermoscopic})

            feat_clin = clinical_output['out']   # (batch, C_clin, H, W)
            feat_derm = derm_output['out']       # (batch, C_derm, H, W)

            # Compute SNNL per channel
            scores_clin = compute_channel_snnl(feat_clin, sensitive, args.temperature)
            scores_derm = compute_channel_snnl(feat_derm, sensitive, args.temperature)

            clinical_scores.append(scores_clin.cpu().numpy())
            derm_scores.append(scores_derm.cpu().numpy())

    # Remove hooks
    hook_clin.remove()
    hook_derm.remove()

    # Average across batches
    avg_clin = np.mean(clinical_scores, axis=0)
    avg_derm = np.mean(derm_scores, axis=0)

    # Identify channels to prune (lowest scores)
    n_prune_clin = max(1, int(len(avg_clin) * args.prune_ratio))
    n_prune_derm = max(1, int(len(avg_derm) * args.prune_ratio))

    # Argsort ascending -> lower SNNL first
    clin_sorted_idx = np.argsort(avg_clin)
    derm_sorted_idx = np.argsort(avg_derm)
    prune_clin = clin_sorted_idx[:n_prune_clin].tolist()
    prune_derm = derm_sorted_idx[:n_prune_derm].tolist()

    print(f"Clinical: {len(avg_clin)} channels, pruning {n_prune_clin} (lowest SNNL indices: {prune_clin[:10]}...)")
    print(f"Dermoscopic: {len(avg_derm)} channels, pruning {n_prune_derm} (lowest SNNL indices: {prune_derm[:10]}...)")

    # Save results
    results = {
        'clinical_scores': avg_clin.tolist(),
        'dermoscopic_scores': avg_derm.tolist(),
        'prune_clinical_channels': prune_clin,
        'prune_dermoscopic_channels': prune_derm,
        'prune_ratio': args.prune_ratio,
        'temperature': args.temperature,
    }
    with open(out_dir / 'snnl_scores.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved SNNL scores to {out_dir / 'snnl_scores.json'}")

if __name__ == '__main__':
    main()