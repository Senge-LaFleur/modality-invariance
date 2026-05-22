"""
evaluation.py

Provides functions for:
- Loading data (paired/unpaired) with fast ID‑based image resolution (prebuilt maps)
- Running validation (collecting predictions, computing metrics)
- Fairness metrics (EOM, PQD, DPM, per-FST accuracy)
- Plotting: confusion matrix, per-class metrics, fairness summary, training curves, t‑SNE
- Saving results as CSV files (overall, per-class, per-FST)
- Building train/val/test/cross‑eval data loaders

build_loaders() now accepts train_modality in cfg:
    'clin'  → train/val on clin CSVs only;  test on clin_test + derm_test + paired_test (clinical/derm)
    'derm'  → train/val on derm CSVs only;  test on clin_test + derm_test + paired_test (clinical/derm)
    'both'  → train/val on paired+clin+derm; test on clin_test + derm_test + paired_test (clinical/derm)

Returns:
    train_loader, val_loader, test_loaders (unpaired clin/derm), 
    paired_test_loaders (unpaired clin/derm from paired_test.csv),
    eval_loaders (cross‑dataset unpaired clin/derm for derm7pt)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as sk_confusion_matrix,
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
from tqdm import tqdm


# ------------------------------------------------------------
# Helper: pre‑build image maps for all datasets
# ------------------------------------------------------------
def build_image_maps(dataset_roots):
    """
    Prebuild mapping from (dataset, image_id) -> full path for all datasets.
    Returns a dict: dataset_name -> {image_id_stem: Path}
    """
    image_maps = {}
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    for ds_name, root in dataset_roots.items():
        img_map = {}
        for ext in extensions:
            for p in root.rglob(f'*{ext}'):
                img_map[p.stem] = p
        image_maps[ds_name] = img_map
        print(f"[INFO] Built image map for {ds_name}: {len(img_map)} entries")
    return image_maps


# ------------------------------------------------------------
# Custom Dataset Classes (ID‑based image resolution using prebuilt maps)
# ------------------------------------------------------------
class UnpairedDataset(Dataset):
    """
    Dataset for unpaired images.
    CSV must contain an ID column (specified by `id_col`) and columns:
    'label', 'skin_type', 'dataset'
    """
    def __init__(self, df, image_maps, transform=None, id_col='image_id'):
        self.df = df.reset_index(drop=True)
        self.image_maps = image_maps
        self.transform = transform
        self.id_col = id_col

    def _resolve_path(self, row):
        ds = row['dataset']
        img_id = str(row[self.id_col])
        full_path = self.image_maps[ds].get(img_id)
        if full_path is None:
            raise FileNotFoundError(f"Image not found for {ds}: {img_id}")
        return full_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = self._resolve_path(row)
        img = Image.open(full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        modality = 'clinical' if self.id_col == 'clinical' else 'derm'
        return {
            'clinical': img,
            'derm': img,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'skin_type': torch.tensor(row['skin_type'], dtype=torch.long),
            'dataset': row['dataset'],
            'paired': False,
            'modality': modality,
        }


class PairedDataset(Dataset):
    """
    Dataset for paired clinical + dermoscopic images.
    CSV must contain columns: 'clinical', 'derm', 'label', 'skin_type', 'dataset'
    """
    def __init__(self, df, image_maps, transform=None,
                 clinical_col='clinical', derm_col='derm'):
        self.df = df.reset_index(drop=True)
        self.image_maps = image_maps
        self.transform = transform
        self.clinical_col = clinical_col
        self.derm_col = derm_col

    def _resolve_path(self, ds, img_id):
        full_path = self.image_maps[ds].get(img_id)
        if full_path is None:
            raise FileNotFoundError(f"Image not found for {ds}: {img_id}")
        return full_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ds = row['dataset']
        clinical_id = str(row[self.clinical_col])
        derm_id = str(row[self.derm_col])
        clinical_path = self._resolve_path(ds, clinical_id)
        derm_path = self._resolve_path(ds, derm_id)

        clinical_img = Image.open(clinical_path).convert('RGB')
        derm_img = Image.open(derm_path).convert('RGB')
        if self.transform:
            clinical_img = self.transform(clinical_img)
            derm_img = self.transform(derm_img)

        return {
            'clinical': clinical_img,
            'derm': derm_img,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'skin_type': torch.tensor(row['skin_type'], dtype=torch.long),
            'dataset': ds,
            'paired': True,
            'modality': 'paired',
        }


# ------------------------------------------------------------
# DataLoader Builder
# ------------------------------------------------------------
def build_loaders(cfg, seed=42):
    """
    Build train, val, test, paired-test, and cross-evaluation loaders.

    Returns
    -------
    train_loader : DataLoader
    val_loader   : DataLoader
    test_loaders : dict  {'clin': DataLoader, 'derm': DataLoader}  from clin_test/derm_test
    paired_test_loaders : dict  {'clin': DataLoader, 'derm': DataLoader}  from paired_test.csv (clinical/derm split)
    eval_loaders : dict  cross-dataset unpaired loaders, e.g. {'derm7pt_clin': DataLoader, 'derm7pt_derm': DataLoader}
    """
    csv_dir = Path(cfg['csv_dir'])
    if 'dataset_roots' in cfg:
        dataset_roots = cfg['dataset_roots']
    elif 'image_roots' in cfg:
        dataset_roots = cfg['image_roots']
    else:
        raise KeyError("Missing 'dataset_roots' or 'image_roots' in config")

    train_modality = cfg.get('train_modality', 'both')
    image_maps = build_image_maps(dataset_roots)

    batch_size = cfg['batch_size']
    img_size   = cfg.get('img_size', 224)

    # ── Transforms ──────────────────────────────────────────────────────
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ── Helper to safely load a CSV ──────────────────────────────────────
    def _load(name):
        p = csv_dir / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    # ── Build train / val datasets based on modality regime ─────────────
    if train_modality == 'clin':
        # Single-modality: clinical only
        clin_train = _load('clin_train.csv')
        clin_val   = _load('clin_val.csv')
        if clin_train.empty:
            raise FileNotFoundError("clin_train.csv not found or empty.")
        train_datasets = [UnpairedDataset(clin_train, image_maps,
                                          transform=train_transform, id_col='clinical')]
        val_datasets   = ([UnpairedDataset(clin_val, image_maps,
                                           transform=val_transform, id_col='clinical')]
                          if not clin_val.empty else [])
        print(f"[build_loaders] train_modality=clin  "
              f"| train={len(clin_train)}  val={len(clin_val)}")

    elif train_modality == 'derm':
        # Single-modality: dermoscopic only
        derm_train = _load('derm_train.csv')
        derm_val   = _load('derm_val.csv')
        if derm_train.empty:
            raise FileNotFoundError("derm_train.csv not found or empty.")
        train_datasets = [UnpairedDataset(derm_train, image_maps,
                                          transform=train_transform, id_col='derm')]
        val_datasets   = ([UnpairedDataset(derm_val, image_maps,
                                           transform=val_transform, id_col='derm')]
                          if not derm_val.empty else [])
        print(f"[build_loaders] train_modality=derm  "
              f"| train={len(derm_train)}  val={len(derm_val)}")

    else:  # 'both'
        # Dual regime: paired + unpaired clinical + unpaired derm
        paired_train = _load('paired_train.csv')
        clin_train   = _load('clin_train.csv')
        derm_train   = _load('derm_train.csv')
        paired_val   = _load('paired_val.csv')
        clin_val     = _load('clin_val.csv')
        derm_val     = _load('derm_val.csv')

        train_datasets = []
        if not paired_train.empty:
            train_datasets.append(PairedDataset(paired_train, image_maps,
                                                transform=train_transform))
        if not clin_train.empty:
            train_datasets.append(UnpairedDataset(clin_train, image_maps,
                                                  transform=train_transform, id_col='clinical'))
        if not derm_train.empty:
            train_datasets.append(UnpairedDataset(derm_train, image_maps,
                                                  transform=train_transform, id_col='derm'))
        if not train_datasets:
            raise FileNotFoundError("No training CSVs found for train_modality='both'.")

        val_datasets = []
        if not paired_val.empty:
            val_datasets.append(PairedDataset(paired_val, image_maps,
                                              transform=val_transform))
        if not clin_val.empty:
            val_datasets.append(UnpairedDataset(clin_val, image_maps,
                                                transform=val_transform, id_col='clinical'))
        if not derm_val.empty:
            val_datasets.append(UnpairedDataset(derm_val, image_maps,
                                                transform=val_transform, id_col='derm'))

        n_train = sum(len(d.df) for d in train_datasets)
        n_val   = sum(len(d.df) for d in val_datasets)
        print(f"[build_loaders] train_modality=both  "
              f"| train={n_train}  val={n_val}")

    if not train_datasets:
        raise FileNotFoundError("No training data found. Check CSV files in " + str(csv_dir))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    val_dataset   = (torch.utils.data.ConcatDataset(val_datasets)
                     if val_datasets else None)

    # ── Test loaders — always clin_test AND derm_test (unpaired) ────────
    clin_test_df = _load('clin_test.csv')
    derm_test_df = _load('derm_test.csv')

    test_loaders = {}
    if not clin_test_df.empty:
        clin_test_ds = UnpairedDataset(clin_test_df, image_maps,
                                       transform=val_transform, id_col='clinical')
        test_loaders['clin'] = DataLoader(clin_test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)
        print(f"[build_loaders] clin_test : {len(clin_test_df)} samples")
    if not derm_test_df.empty:
        derm_test_ds = UnpairedDataset(derm_test_df, image_maps,
                                       transform=val_transform, id_col='derm')
        test_loaders['derm'] = DataLoader(derm_test_ds, batch_size=batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)
        print(f"[build_loaders] derm_test : {len(derm_test_df)} samples")

    # ── Paired test loader (paired_test.csv) split into clinical and derm ──
    paired_test_df = _load('paired_test.csv')
    paired_test_loaders = {}
    if not paired_test_df.empty:
        # Clinical images from paired_test (use column 'clinical' as image_id)
        paired_clin_ds = UnpairedDataset(paired_test_df, image_maps,
                                         transform=val_transform, id_col='clinical')
        paired_test_loaders['clin'] = DataLoader(paired_clin_ds, batch_size=batch_size,
                                                 shuffle=False, num_workers=4, pin_memory=True)
        # Dermoscopic images from paired_test (use column 'derm' as image_id)
        paired_derm_ds = UnpairedDataset(paired_test_df, image_maps,
                                         transform=val_transform, id_col='derm')
        paired_test_loaders['derm'] = DataLoader(paired_derm_ds, batch_size=batch_size,
                                                 shuffle=False, num_workers=4, pin_memory=True)
        print(f"[build_loaders] paired_test (clinical) : {len(paired_test_df)} samples")
        print(f"[build_loaders] paired_test (derm)      : {len(paired_test_df)} samples")

    # ── Cross‑evaluation loaders: derm7pt (split into clinical and derm) ──
    eval_loaders = {}
    derm7pt_df = _load('eval_derm7pt.csv') if (csv_dir / 'eval_derm7pt.csv').exists() else pd.DataFrame()
    if not derm7pt_df.empty:
        # Clinical: use column 'clinical' as image_id
        derm7pt_clin_ds = UnpairedDataset(derm7pt_df, image_maps,
                                          transform=val_transform, id_col='clinical')
        eval_loaders['derm7pt_clin'] = DataLoader(derm7pt_clin_ds, batch_size=batch_size,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        # Derm: use column 'derm' as image_id
        derm7pt_derm_ds = UnpairedDataset(derm7pt_df, image_maps,
                                          transform=val_transform, id_col='derm')
        eval_loaders['derm7pt_derm'] = DataLoader(derm7pt_derm_ds, batch_size=batch_size,
                                                  shuffle=False, num_workers=4, pin_memory=True)
        print(f"[INFO] Loaded derm7pt clinical eval set: {len(derm7pt_df)} samples")
        print(f"[INFO] Loaded derm7pt derm eval set: {len(derm7pt_df)} samples")
    else:
        print("[WARN] eval_derm7pt.csv not found. Skipping cross‑eval.")

    # ── Weighted sampler for training ────────────────────────────────────
    labels = []
    for ds in train_datasets:
        labels.extend(ds.df['label'].tolist())
    class_counts   = np.bincount(labels, minlength=cfg['num_classes'])
    class_weights  = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(train_dataset),
                                    replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = (DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
                    if val_dataset else None)

    return train_loader, val_loader, test_loaders, paired_test_loaders, eval_loaders


# ------------------------------------------------------------
# Helper: robust macro AUROC
# ------------------------------------------------------------
def robust_macro_auroc(probs, labels):
    present = np.unique(labels)
    if len(present) < 2:
        return float("nan")
    aucs = []
    for c in range(probs.shape[1]):
        if c not in present:
            continue
        binary = (labels == c).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            continue
        try:
            auc = roc_auc_score(binary, probs[:, c])
            aucs.append(auc)
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


# ------------------------------------------------------------
# Validation function
# ------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, num_classes=5, desc="Validation"):
    model.eval()
    all_probs, all_labels, all_skins = [], [], []
    print(f"Running {desc}...")
    pbar = tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        out = model(batch)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].cpu().numpy())
        all_skins.append(batch["skin_type"].cpu().numpy())
    pbar.close()

    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    skins  = np.concatenate(all_skins)
    preds  = probs.argmax(axis=1)

    acc          = (preds == labels).mean()
    auroc        = robust_macro_auroc(probs, labels)
    macro_f1     = f1_score(labels, preds, average="macro",    zero_division=0)
    micro_f1     = f1_score(labels, preds, average="micro",    zero_division=0)
    weighted_f1  = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_prec   = precision_score(labels, preds, average="macro", zero_division=0)
    macro_rec    = recall_score(labels, preds,    average="macro", zero_division=0)
    per_class_prec = precision_score(labels, preds, average=None, zero_division=0,
                                     labels=list(range(num_classes)))
    per_class_rec  = recall_score(labels, preds, average=None, zero_division=0,
                                  labels=list(range(num_classes)))
    per_class_f1   = f1_score(labels, preds, average=None, zero_division=0,
                              labels=list(range(num_classes)))
    conf_mat = sk_confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {
        "acc": acc, "auroc": auroc,
        "macro_f1": macro_f1, "micro_f1": micro_f1, "weighted_f1": weighted_f1,
        "macro_prec": macro_prec, "macro_rec": macro_rec,
        "per_class_prec": per_class_prec,
        "per_class_rec":  per_class_rec,
        "per_class_f1":   per_class_f1,
        "conf_mat": conf_mat,
        "probs": probs, "preds": preds, "labels": labels, "skin": skins,
    }


# ------------------------------------------------------------
# Fairness metrics (unchanged)
# ------------------------------------------------------------
def pg_acc(preds, labels, groups, K=6):
    out = {}
    for g in range(K):
        mask = groups == g
        out[g] = float("nan") if mask.sum() == 0 else (preds[mask] == labels[mask]).mean()
    return out

def eom(preds, labels, groups, K=6):
    classes = np.unique(labels)
    valid_groups = [g for g in range(K) if (groups == g).sum() > 0]
    if len(valid_groups) < 2:
        return float("nan")
    ratios = []
    for cls in classes:
        tprs = []
        for g in valid_groups:
            mask = (groups == g) & (labels == cls)
            if mask.sum() == 0:
                continue
            tprs.append((preds[mask] == cls).mean())
        if len(tprs) < 2:
            continue
        mn, mx = min(tprs), max(tprs)
        ratios.append(1.0 if mx == 0 else mn / mx)
    return float(np.mean(ratios)) if ratios else float("nan")

def pqd(pg_acc_dict):
    vals = [v for v in pg_acc_dict.values() if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    mn, mx = min(vals), max(vals)
    return float("nan") if mx == 0 else mn / mx

def dpm(preds, labels, groups, K=6):
    classes = np.unique(labels)
    valid_groups = [g for g in range(K) if (groups == g).sum() > 0]
    if len(valid_groups) < 2:
        return float("nan")
    ratios = []
    for cls in classes:
        rates = []
        for g in valid_groups:
            mask = groups == g
            if mask.sum() == 0:
                continue
            rates.append((preds[mask] == cls).mean())
        if len(rates) < 2:
            continue
        mn, mx = min(rates), max(rates)
        ratios.append(1.0 if mx == 0 else mn / mx)
    return float(np.mean(ratios)) if ratios else float("nan")

def fairness(res, K=6):
    pg = pg_acc(res["preds"], res["labels"], res["skin"], K)
    return {
        "pg_acc": pg,
        "EOM": eom(res["preds"], res["labels"], res["skin"], K),
        "PQD": pqd(pg),
        "DPM": dpm(res["preds"], res["labels"], res["skin"], K),
    }


# ------------------------------------------------------------
# CSV saving functions (unchanged)
# ------------------------------------------------------------
def save_results_csv(res, fair, split_name, results_dir, label_names):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    overall = {
        "split": split_name,
        "accuracy": res["acc"], "auroc": res["auroc"],
        "macro_precision": res["macro_prec"], "macro_recall": res["macro_rec"],
        "macro_f1": res["macro_f1"], "micro_f1": res["micro_f1"],
        "weighted_f1": res["weighted_f1"],
        "EOM": fair["EOM"], "PQD": fair["PQD"], "DPM": fair["DPM"],
    }
    pd.DataFrame([overall]).to_csv(results_dir / f"{split_name}_overall.csv", index=False)

    per_class = []
    for i in range(len(res["per_class_prec"])):
        per_class.append({
            "split": split_name,
            "class": label_names.get(i, str(i)),
            "precision": res["per_class_prec"][i],
            "recall":    res["per_class_rec"][i],
            "f1":        res["per_class_f1"][i],
        })
    pd.DataFrame(per_class).to_csv(results_dir / f"{split_name}_per_class.csv", index=False)

    per_fst = []
    for fst_idx, acc in fair["pg_acc"].items():
        per_fst.append({
            "split": split_name,
            "fitzpatrick_type": f"FST {fst_idx+1}",
            "accuracy": acc if not np.isnan(acc) else None,
        })
    pd.DataFrame(per_fst).to_csv(results_dir / f"{split_name}_per_fst.csv", index=False)


# ------------------------------------------------------------
# Plotting functions (unchanged)
# ------------------------------------------------------------
def plot_confusion_matrix(conf_mat, class_names, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Raw Counts")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_xticklabels(class_names, rotation=40, ha="right")
    axes[0].set_yticklabels(class_names)

    row_sums = conf_mat.sum(axis=1, keepdims=True).clip(min=1)
    conf_norm = conf_mat.astype(float) / row_sums
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1],
                vmin=0, vmax=1, cbar=False)
    axes[1].set_title("Normalized (Recall)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_xticklabels(class_names, rotation=40, ha="right")
    axes[1].set_yticklabels(class_names)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_per_class_metrics(res, class_names, title, save_path):
    n_cls = len(class_names)
    x = np.arange(n_cls)
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, res["per_class_prec"], width, label="Precision", color="#1950A0")
    ax.bar(x,         res["per_class_rec"],  width, label="Recall",    color="#0096B4")
    ax.bar(x + width, res["per_class_f1"],   width, label="F1",        color="#DC641E")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_fairness_metrics(fair, title, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    fst_acc    = [fair["pg_acc"].get(i, float("nan")) for i in range(6)]
    fst_labels = [f"FST {i+1}" for i in range(6)]
    colors = ["#0096B4" if not np.isnan(v) else "#CCCCCC" for v in fst_acc]
    bars = axes[0].bar(fst_labels, [0 if np.isnan(v) else v for v in fst_acc],
                       color=colors, edgecolor="white")
    mean_acc = np.nanmean(fst_acc)
    axes[0].axhline(mean_acc, ls="--", color="#D1333B", label=f"Mean = {mean_acc:.3f}")
    for bar, v in zip(bars, fst_acc):
        if not np.isnan(v):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01,
                         f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-FST Accuracy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    fair_names = ["EOM (↑)", "PQD (↑)", "DPM (↑)"]
    fair_vals  = [fair["EOM"], fair["PQD"], fair["DPM"]]
    axes[1].bar(fair_names, [max(0, v) if not np.isnan(v) else 0 for v in fair_vals],
                color="#2ECC71", edgecolor="white")
    for i, v in enumerate(fair_vals):
        if not np.isnan(v):
            axes[1].text(i, max(0, v) + 0.01, f"{v:.4f}", ha="center", fontsize=10)
    axes[1].set_ylim(0, 1.15)
    axes[1].set_ylabel("Score")
    axes[1].set_title("Fairness Summary (EOM / PQD / DPM)")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_training_curves(history, title, save_path):
    epochs = list(range(1, len(history.get("train_total", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0]
    if "train_total" in history:
        ax.plot(epochs, history["train_total"], color="#1950A0", lw=2,
                marker="o", ms=4, label="Train Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    if "train_acc"  in history:
        ax.plot(epochs, history["train_acc"],  color="#1950A0", lw=2,
                marker="o", ms=4, label="Train Acc")
    if "val_acc"    in history:
        ax.plot(epochs, history["val_acc"],    color="#0096B4", lw=2,
                marker="s", ms=4, label="Val Acc",   linestyle="--")
    if "val_auroc"  in history:
        ax.plot(epochs, history["val_auroc"],  color="#DC641E", lw=2,
                marker="^", ms=4, label="Val AUROC", linestyle=":")
    ax.set_title("Accuracy & AUROC")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_tsne(embeddings, labels, title, save_path, perplexity=30):
    if embeddings.shape[0] < 5:
        print(f"[SKIP] t-SNE: too few samples ({embeddings.shape[0]})")
        return
    perp = min(perplexity, max(5, embeddings.shape[0] // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp,
                max_iter=1000, init="pca")
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1],
                          c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.colorbar(scatter, ticks=range(labels.max()+1), label="Class")
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

def print_full_report(res, fair, split_name, label_names):
    print(f"\n{'='*60}")
    print(f"  {split_name} — Full Evaluation Report")
    print(f"{'='*60}")
    print(f"\n{'─'*40}")
    print("  Overall Metrics")
    print(f"{'─'*40}")
    print(f"  Accuracy          : {res['acc']:.4f}")
    print(f"  AUC-ROC (macro)   : {res['auroc']:.4f}")
    print(f"  Precision (macro) : {res['macro_prec']:.4f}")
    print(f"  Recall    (macro) : {res['macro_rec']:.4f}")
    print(f"  F1        (macro) : {res['macro_f1']:.4f}")
    print(f"  F1        (micro) : {res['micro_f1']:.4f}")
    print(f"  F1     (weighted) : {res['weighted_f1']:.4f}")
    print(f"\n{'─'*40}")
    print("  Per-Class Metrics")
    print(f"{'─'*40}")
    print(f"  {'Class':<26} {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'─'*48}")
    for i in range(len(res["per_class_prec"])):
        name = label_names.get(i, f"class_{i}").title()
        print(f"  {name:<26} {res['per_class_prec'][i]:6.4f}  "
              f"{res['per_class_rec'][i]:6.4f}  {res['per_class_f1'][i]:6.4f}")
    print(f"\n{'─'*40}")
    print("  Confusion Matrix  (rows=True, cols=Pred)")
    print(f"{'─'*40}")
    short = [label_names.get(i, str(i))[:5].ljust(5)
             for i in range(len(res["per_class_prec"]))]
    print("       " + "  ".join(short))
    for i, row in enumerate(res["conf_mat"]):
        print(f"  {short[i]}  " + "  ".join(f"{v:5d}" for v in row))
    print(f"\n{'─'*40}")
    print("  Fairness Metrics")
    print(f"{'─'*40}")
    for m in ["EOM", "PQD", "DPM"]:
        print(f"  {m} (↑): {fair[m]:.4f}")
    print(f"\n{'─'*40}")
    print("  Per-Fitzpatrick-Skin-Type Accuracy")
    print(f"{'─'*40}")
    for g, v in fair["pg_acc"].items():
        n_grp = int((res["skin"] == g).sum())
        bar   = "█" * int((v if not np.isnan(v) else 0) * 20)
        note  = "" if not np.isnan(v) else f"  ← n={n_grp} (no samples)"
        print(f"  FST {g+1}  n={n_grp:>4}  {v:.4f}  {bar}{note}")
    print(f"\n{'='*60}\n")


# ------------------------------------------------------------
# Label mapping (shared with training scripts)
# ------------------------------------------------------------
LABEL_NAMES = {
    0: 'melanoma',
    1: 'nevus',
    2: 'basal cell carcinoma',
    3: 'actinic keratosis',
    4: 'squamous cell carcinoma',
}