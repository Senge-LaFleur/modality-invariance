#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bootstrap resampling on the test set to estimate confidence intervals
for classification and fairness metrics.

Loads a pre‑trained model (e.g., best_f1_model.pt) and runs
bootstrap sampling on the test set predictions.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'
import sys
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import warnings

import torch

from models.models_losses import DualViT
from models.evaluation import build_loaders, validate, fairness, LABEL_NAMES

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Configuration (must match the trained model)
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR = WORK_ROOT / 'csvs'
CKPT_DIR = WORK_ROOT / 'checkpoints_Modality_Invariance_vit'   # directory containing best_f1_model.pt

IMAGE_ROOTS = {
    'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'padufes20':      Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),
    'isic2019':       Path('/kaggle/input/datasets/sengenjih/isic2019'),
}

CFG = {
    'csv_dir':       CSV_DIR,
    'image_roots':   IMAGE_ROOTS,
    'vit_model':     'vit_small_patch16_224',
    'embed_dim':     512,            # MUST match training
    'img_size':      224,
    'num_classes':   3,
    'num_skin_types': 6,
    'batch_size':    32,
}

# ------------------------------------------------------------
# Bootstrap function
# ------------------------------------------------------------
def bootstrap_metrics(model, test_loader, device, num_bootstrap=1000, num_classes=3, num_skin=6):
    """
    Returns a DataFrame with bootstrap estimates and confidence intervals.
    """
    # Get full test set predictions once
    full_res = validate(model, test_loader, device, num_classes=num_classes, desc="Test set evaluation")
    n_samples = len(full_res['labels'])
    metrics_list = []

    for _ in tqdm(range(num_bootstrap), desc="Bootstrapping"):
        # Resample indices with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        boot_labels = full_res['labels'][idx]
        boot_preds  = full_res['preds'][idx]
        boot_skins  = full_res['skin'][idx]
        boot_probs  = full_res['probs'][idx]

        # Compute metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
        acc = accuracy_score(boot_labels, boot_preds)
        # AUROC – handle cases with only one class in bootstrap sample
        try:
            auroc = roc_auc_score(boot_labels, boot_probs, multi_class='ovr', average='macro')
        except ValueError:
            auroc = float('nan')
        macro_f1 = f1_score(boot_labels, boot_preds, average='macro', zero_division=0)

        # Fairness metrics (using your existing function)
        boot_res = {
            'preds': boot_preds,
            'labels': boot_labels,
            'skin': boot_skins,
            'probs': boot_probs
        }
        fair = fairness(boot_res, K=num_skin)

        metrics_list.append({
            'accuracy': acc,
            'auroc': auroc,
            'macro_f1': macro_f1,
            'EOM': fair['EOM'],
            'PQD': fair['PQD'],
            'DPM': fair['DPM']
        })

    metrics_df = pd.DataFrame(metrics_list)
    ci_low = metrics_df.quantile(0.025)
    ci_high = metrics_df.quantile(0.975)
    mean_vals = metrics_df.mean()
    return metrics_df, mean_vals, (ci_low, ci_high)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print(f"CSV dir      : {CFG['csv_dir']}")
    print(f"Checkpoint dir: {CKPT_DIR}")

    # Build test loader
    _, _, test_loader, _ = build_loaders(CFG, seed=SEED)
    if test_loader is None:
        print("No test loader found. Ensure that test CSV files exist.")
        return

    # Load model with the same configuration used during training
    model = DualViT(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,          # architecture only; weights will be overwritten
        use_projection=True,      # MUST match training (projection head was used)
    ).to(DEVICE)

    ckpt_path = CKPT_DIR / "best_f1_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "best_auroc_model.pt"
    if not ckpt_path.exists():
        ckpt_path = CKPT_DIR / "last_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded model from {ckpt_path.name} (epoch {ckpt.get('epoch', -1)+1})")

    # Run bootstrap
    bootstrap_df, mean_vals, (ci_low, ci_high) = bootstrap_metrics(
        model, test_loader, DEVICE, num_bootstrap=1000,
        num_classes=CFG["num_classes"], num_skin=CFG["num_skin_types"]
    )

    print("\n=== Bootstrap Results (1000 resamples) ===")
    print("Mean (95% CI):")
    for col in bootstrap_df.columns:
        print(f"{col:10} : {mean_vals[col]:.4f}  [{ci_low[col]:.4f}, {ci_high[col]:.4f}]")

    # Save results
    results_dir = Path("bootstrap_results")
    results_dir.mkdir(exist_ok=True)
    bootstrap_df.to_csv(results_dir / "bootstrap_samples.csv", index=False)
    summary = pd.DataFrame({
        'metric': bootstrap_df.columns,
        'mean': mean_vals.values,
        'ci_lower': ci_low.values,
        'ci_upper': ci_high.values
    })
    summary.to_csv(results_dir / "bootstrap_summary.csv", index=False)
    print(f"\nResults saved to {results_dir}")

if __name__ == "__main__":
    main()