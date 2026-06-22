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
    roc_curve,
    auc,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
from tqdm import tqdm


# ------------------------------------------------------------
# Label mapping (shared with training scripts) — 3-class
# ------------------------------------------------------------
LABEL_NAMES = {
    0: 'melanoma',
    1: 'nevus',
    2: 'basal cell carcinoma',
}


# ------------------------------------------------------------
# Helper: pre-build image maps for all datasets
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
# Custom Dataset Classes (ID-based image resolution using prebuilt maps)
# ------------------------------------------------------------
class UnpairedDataset(Dataset):
    def __init__(self, df, image_maps, transform=None,
                 id_col='image_id', modality='clinical'):
        self.df = df.reset_index(drop=True)
        self.image_maps = image_maps
        self.transform = transform
        self.id_col = id_col
        self.modality = modality  # 'clinical' or 'derm'

    def _resolve_path(self, row):
        ds = row['dataset']
        img_id = str(row[self.id_col])
        full_path = self.image_maps[ds].get(img_id)
        if full_path is None:
            raise FileNotFoundError(
                f"Image not found for dataset={ds}, id_col={self.id_col}, id={img_id}"
            )
        return full_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        full_path = self._resolve_path(row)
        img = Image.open(full_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'clinical': img,
            'derm': img,
            'label': torch.tensor(row['label'], dtype=torch.long),
            'skin_type': torch.tensor(row['skin_type'], dtype=torch.long),
            'dataset': row['dataset'],
            'paired': False,
            'modality': self.modality,
        }


class PairedDataset(Dataset):
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
# DataLoader Builder (ID-based, using prebuilt image maps)
# ------------------------------------------------------------
def build_loaders(cfg, seed=42):
    csv_dir = Path(cfg['csv_dir'])
    # Accept both 'dataset_roots' and 'image_roots' keys
    if 'dataset_roots' in cfg:
        dataset_roots = cfg['dataset_roots']
    elif 'image_roots' in cfg:
        dataset_roots = cfg['image_roots']
    else:
        raise KeyError("Missing 'dataset_roots' or 'image_roots' in config")

    # Prebuild image maps once for fast lookups
    image_maps = build_image_maps(dataset_roots)

    batch_size = cfg['batch_size']
    img_size = cfg.get('img_size', 224)

    # Transforms
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

    def _load_csv(fname):
        p = csv_dir / fname
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _load_paired_split(split):
        """
        Load paired data for a given split ('train' / 'val' / 'test').

        Preferred format: two separate, ID-correspondence files —
            paired_clin_<split>.csv  (columns include: lesion_id, clinical, label, skin_type, dataset)
            paired_derm_<split>.csv  (columns include: lesion_id, derm,     label, skin_type, dataset)
        joined on a shared id column (lesion_id, falling back to pair_id)
        so correspondence is guaranteed by ID rather than by row order.

        Falls back to the legacy single combined CSV
        (paired_train_update.csv / paired_val.csv / paired_test.csv) if the
        split files aren't present, so older preprocessing runs still work.
        """
        clin_p = csv_dir / f'paired_clin_{split}.csv'
        derm_p = csv_dir / f'paired_derm_{split}.csv'
        if clin_p.exists() and derm_p.exists():
            clin_df = pd.read_csv(clin_p)
            derm_df = pd.read_csv(derm_p)
            join_key = next(
                (k for k in ('lesion_id', 'pair_id') if k in clin_df.columns and k in derm_df.columns),
                None,
            )
            if join_key is None:
                raise KeyError(
                    f"paired_clin_{split}.csv / paired_derm_{split}.csv must share a "
                    "'lesion_id' (or 'pair_id') column to establish clinical<->derm correspondence."
                )
            merged = clin_df.merge(
                derm_df[[join_key, 'derm']], on=join_key, how='inner', validate='one_to_one'
            )
            dropped = len(clin_df) - len(merged)
            if dropped:
                print(f"[WARN] paired_{split}: {dropped} clinical rows had no derm match on '{join_key}' and were dropped")
            print(f"[INFO] paired_{split}: loaded {len(merged)} ID-matched pairs from split clin/derm CSVs")
            return merged

        # Legacy fallback: single combined file (back-compat with older runs)
        legacy_name = 'paired_train_update.csv' if split == 'train' else f'paired_{split}.csv'
        legacy = _load_csv(legacy_name)
        if legacy.empty and legacy_name != f'paired_{split}.csv':
            legacy = _load_csv(f'paired_{split}.csv')
        if not legacy.empty:
            print(f"[INFO] paired_{split}: loaded {len(legacy)} rows from legacy combined CSV (no split clin/derm files found)")
        return legacy

    # Load CSV files
    paired_train = _load_paired_split('train')
    clin_train   = _load_csv('clin_train.csv')
    derm_train   = _load_csv('derm_train.csv')

    paired_val = _load_paired_split('val')
    clin_val   = _load_csv('clin_val.csv')
    derm_val   = _load_csv('derm_val.csv')

    paired_test = _load_paired_split('test')
    clin_test   = _load_csv('clin_test.csv')
    derm_test   = _load_csv('derm_test.csv')

    # Build training datasets
    train_datasets = []
    if not paired_train.empty:
        train_datasets.append(PairedDataset(paired_train, image_maps, transform=train_transform))
    if not clin_train.empty:
        train_datasets.append(UnpairedDataset(clin_train, image_maps, transform=train_transform, id_col='clinical', modality='clinical'))
    if not derm_train.empty:
        train_datasets.append(UnpairedDataset(derm_train, image_maps, transform=train_transform, id_col='derm', modality='derm'))
    if not train_datasets:
        raise FileNotFoundError("No training data found. Check CSV files in " + str(csv_dir))

    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Validation datasets
    val_datasets = []
    if not paired_val.empty:
        val_datasets.append(PairedDataset(paired_val, image_maps, transform=val_transform))
    if not clin_val.empty:
        val_datasets.append(UnpairedDataset(clin_val, image_maps, transform=val_transform, id_col='clinical', modality='clinical'))
    if not derm_val.empty:
        val_datasets.append(UnpairedDataset(derm_val, image_maps, transform=val_transform, id_col='derm', modality='derm'))
    
    val_dataset = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None

    # Test datasets
    test_datasets = []
    if not paired_test.empty:
        test_datasets.append(PairedDataset(paired_test, image_maps, transform=val_transform))
    if not clin_test.empty:
        test_datasets.append(UnpairedDataset(clin_test, image_maps, transform=val_transform, id_col='clinical', modality='clinical'))
    if not derm_test.empty:
        test_datasets.append(UnpairedDataset(derm_test, image_maps, transform=val_transform, id_col='derm', modality='derm'))
    test_dataset = torch.utils.data.ConcatDataset(test_datasets) if test_datasets else None

    # Weighted sampler for training (handle class imbalance)
    labels = []
    for ds in train_datasets:
        labels.extend(ds.df['label'].tolist())
    class_counts = np.bincount(labels, minlength=cfg['num_classes'])
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[lbl] for lbl in labels]
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_dataset), replacement=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    ) if val_dataset else None
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    ) if test_dataset else None

    # ------------------------------------------------------------------
    # Cross-evaluation loaders
    # ------------------------------------------------------------------
    eval_loaders = {}

    # ── Fitzpatrick17k (clinical, image_id column) ─────────────────────
    _fitz_csv = csv_dir / 'eval_fitzpatrick17k.csv'
    if _fitz_csv.exists():
        fitz_df = pd.read_csv(_fitz_csv)
        has_cols = {'clinical', 'label', 'skin_type', 'dataset'}.issubset(fitz_df.columns)
        if has_cols:
            fitz_dataset = UnpairedDataset(
                fitz_df, image_maps, transform=val_transform,
                id_col='clinical', modality='clinical'
            )
            eval_loaders['fitzpatrick17k'] = DataLoader(
                fitz_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            print(f"[INFO] Loaded fitzpatrick17k eval set: {len(fitz_df)} samples")
        else:
            print(
                "[WARN] eval_fitzpatrick17k.csv must contain "
                "('clinical','label','skin_type','dataset'). Skipping."
            )
    else:
        print(
            f"[WARN] eval_fitzpatrick17k.csv not found at {_fitz_csv} — "
            "run data_preprocessing_v3 to generate it."
        )

    # ── PAD-UFES-20 (clinical, 'clinical' column) ───────────────────────
    _padufes_csv = csv_dir / 'eval_padufes20.csv'
    if _padufes_csv.exists():
        padufes_df = pd.read_csv(_padufes_csv)
        has_clinical = {'clinical', 'label', 'skin_type', 'dataset'}.issubset(
            padufes_df.columns
        )
        if has_clinical:
            padufes_dataset = UnpairedDataset(
                padufes_df, image_maps, transform=val_transform,
                id_col='clinical', modality='clinical'
            )
            eval_loaders['padufes20'] = DataLoader(
                padufes_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            print(f"[INFO] Loaded padufes20 eval set: {len(padufes_df)} samples")
        else:
            print(
                "[WARN] eval_padufes20.csv must contain "
                "('clinical','label','skin_type','dataset'). Skipping."
            )
    else:
        print(
            f"[WARN] eval_padufes20.csv not found at {_padufes_csv} — "
            "run data_preprocessing_v3 to generate it."
        )

    # ── ISIC2019 (dermoscopic, 'derm' column) ───────────────────────────
    _isic_csv = csv_dir / 'eval_isic2019.csv'
    if _isic_csv.exists():
        isic_df = pd.read_csv(_isic_csv)
        has_derm = {'derm', 'label', 'skin_type', 'dataset'}.issubset(isic_df.columns)
        if has_derm:
            isic_dataset = UnpairedDataset(
                isic_df, image_maps, transform=val_transform,
                id_col='derm', modality='derm'
            )
            eval_loaders['isic2019'] = DataLoader(
                isic_dataset, batch_size=batch_size, shuffle=False,
                num_workers=4, pin_memory=True
            )
            print(f"[INFO] Loaded isic2019 eval set: {len(isic_df)} samples")
        else:
            print(
                "[WARN] eval_isic2019.csv must contain "
                "('derm','label','skin_type','dataset'). Skipping."
            )
    else:
        print(
            f"[WARN] eval_isic2019.csv not found at {_isic_csv} — "
            "run data_preprocessing_v3 to generate it."
        )

    return train_loader, val_loader, test_loader, eval_loaders


# ------------------------------------------------------------
# Helper: robust macro AUROC and robust macro F1
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
            auc_score = roc_auc_score(binary, probs[:, c])
            aucs.append(auc_score)
        except Exception:
            continue
    return float(np.mean(aucs)) if aucs else float("nan")


def robust_macro_f1(labels, preds):
    """
    Compute macro F1 only over classes that actually appear in the ground truth.
    Ignores missing classes, avoiding the artificial 0.0 that standard macro F1 would add.
    """
    present_classes = np.unique(labels)
    if len(present_classes) < 2:   # need at least 2 classes for macro average to be meaningful
        return float("nan")
    
    f1_scores = []
    for c in present_classes:
        tp = np.sum((labels == c) & (preds == c))
        fp = np.sum((labels != c) & (preds == c))
        fn = np.sum((labels == c) & (preds != c))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)
    
    return float(np.mean(f1_scores))


# ------------------------------------------------------------
# KNN accuracy on embeddings (for class separation evaluation)
# ------------------------------------------------------------
def compute_knn_accuracy(embeddings, labels, k=3, device='cpu'):
    """
    Compute k-NN classification accuracy using leave-one-out cross-validation.

    WHY LOO: fitting and predicting on the same set inflates accuracy to
    near-100% regardless of embedding quality, because every point's own
    nearest neighbour is itself.  Leave-one-out excludes each query point
    from its own neighbour search, giving a meaningful measure of how well
    the embedding space separates classes for *unseen* queries.

    embeddings: numpy array of shape (n_samples, embed_dim)
    labels:     numpy array of shape (n_samples,)
    Returns:    accuracy (float)
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import LeaveOneOut, cross_val_score

    n = len(labels)
    # LOO is exact but O(n^2) — cap at 2000 samples with stratified subsampling
    # to keep evaluation fast while remaining statistically representative.
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        embeddings = embeddings[idx]
        labels = labels[idx]

    knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    loo = LeaveOneOut()
    scores = cross_val_score(knn, embeddings, labels, cv=loo, scoring='accuracy')
    return float(scores.mean())


# ------------------------------------------------------------
# Validation function (with descriptive progress bar)
# ------------------------------------------------------------
@torch.no_grad()
def validate(model, loader, device, num_classes=3, desc="Validation"):
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

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    skins = np.concatenate(all_skins)
    preds = probs.argmax(axis=1)

    acc = (preds == labels).mean()
    auroc = robust_macro_auroc(probs, labels)
    macro_f1 = robust_macro_f1(labels, preds)
    micro_f1 = f1_score(labels, preds, average="micro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    macro_prec = precision_score(labels, preds, average="macro", zero_division=0)
    macro_rec = recall_score(labels, preds, average="macro", zero_division=0)
    per_class_prec = precision_score(labels, preds, average=None, zero_division=0,
                                     labels=list(range(num_classes)))
    per_class_rec = recall_score(labels, preds, average=None, zero_division=0,
                                 labels=list(range(num_classes)))
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0,
                             labels=list(range(num_classes)))
    conf_mat = sk_confusion_matrix(labels, preds, labels=list(range(num_classes)))

    return {
        "acc": acc,
        "auroc": auroc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_prec": macro_prec,
        "macro_rec": macro_rec,
        "per_class_prec": per_class_prec,
        "per_class_rec": per_class_rec,
        "per_class_f1": per_class_f1,
        "conf_mat": conf_mat,
        "probs": probs,
        "preds": preds,
        "labels": labels,
        "skin": skins,
    }


# ------------------------------------------------------------
# Fairness metrics (original, 6‑class FST, ratio‑based)
# ------------------------------------------------------------
def pg_acc(preds, labels, groups, K=6):
    out = {}
    for g in range(K):
        mask = groups == g
        if mask.sum() == 0:
            out[g] = float("nan")
        else:
            out[g] = (preds[mask] == labels[mask]).mean()
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
# Binary fairness metrics (light = FST 0-2, dark = FST 3-5)
# Returns differences: (light - dark)
# ------------------------------------------------------------
def group_light_dark(skin_types):
    """
    Convert 6-class Fitzpatrick (0-5) to binary:
        Light (I-III): 0,1,2 -> 0
        Dark  (IV-VI): 3,4,5 -> 1
    Returns array of 0/1 (or -1 for unknown).
    """
    groups = np.full_like(skin_types, -1)
    light_mask = (skin_types >= 0) & (skin_types <= 2)
    dark_mask  = (skin_types >= 3) & (skin_types <= 5)
    groups[light_mask] = 0
    groups[dark_mask]  = 1
    return groups

def demographic_parity_diff(preds, labels, groups_binary):
    """
    Demographic Parity Difference = P(ŷ=1 | light) - P(ŷ=1 | dark)
    For multi-class, we average over classes.
    """
    classes = np.unique(labels)
    diffs = []
    for cls in classes:
        mask_light = (groups_binary == 0)
        mask_dark  = (groups_binary == 1)
        rate_light = (preds[mask_light] == cls).mean() if mask_light.sum() > 0 else np.nan
        rate_dark  = (preds[mask_dark]  == cls).mean() if mask_dark.sum()  > 0 else np.nan
        if not np.isnan(rate_light) and not np.isnan(rate_dark):
            diffs.append(abs(rate_light - rate_dark))
    return float(np.mean(diffs)) if diffs else float('nan')

def equal_opportunity_tpr(preds, labels, groups_binary):
    """
    Equal Opportunity (TPR) Difference = TPR_light - TPR_dark
    TPR = P(ŷ=1 | y=1) for binary classification; for multi-class, average over classes.
    """
    classes = np.unique(labels)
    diffs = []
    for cls in classes:
        mask_light = (groups_binary == 0) & (labels == cls)
        mask_dark  = (groups_binary == 1) & (labels == cls)
        tpr_light = (preds[mask_light] == cls).mean() if mask_light.sum() > 0 else np.nan
        tpr_dark  = (preds[mask_dark]  == cls).mean() if mask_dark.sum()  > 0 else np.nan
        if not np.isnan(tpr_light) and not np.isnan(tpr_dark):
            diffs.append(abs(tpr_light - tpr_dark))
    return float(np.mean(diffs)) if diffs else float('nan')

def equal_opportunity_tnr(preds, labels, groups_binary):
    """
    Equal Opportunity (TNR) Difference = TNR_light - TNR_dark
    TNR = P(ŷ≠1 | y≠1). For multi-class, we consider each class as positive and others as negative,
    then average over classes.
    """
    classes = np.unique(labels)
    diffs = []
    for cls in classes:
        # For TNR, "positive" = cls, "negative" = all other classes
        mask_light = (groups_binary == 0) & (labels != cls)
        mask_dark  = (groups_binary == 1) & (labels != cls)
        # Correct prediction for negative samples means prediction ≠ cls
        tnr_light = (preds[mask_light] != cls).mean() if mask_light.sum() > 0 else np.nan
        tnr_dark  = (preds[mask_dark]  != cls).mean() if mask_dark.sum()  > 0 else np.nan
        if not np.isnan(tnr_light) and not np.isnan(tnr_dark):
            diffs.append(abs(tnr_light - tnr_dark))
    return float(np.mean(diffs)) if diffs else float('nan')

def equalized_odds(preds, labels, groups_binary):
    """
    Equalized Odds = average of absolute TPR difference and absolute TNR difference.
    (Or max of the two; here we use average for simplicity.)
    """
    tpr_diff = equal_opportunity_tpr(preds, labels, groups_binary)
    tnr_diff = equal_opportunity_tnr(preds, labels, groups_binary)
    if np.isnan(tpr_diff) or np.isnan(tnr_diff):
        return float('nan')
    return (abs(tpr_diff) + abs(tnr_diff)) / 2.0

def fairness_binary(res):
    """
    Compute binary fairness metrics (differences) using light (FST I-III) vs dark (FST IV-VI).
    Returns dict with keys:
        DP_diff   (demographic parity difference)
        EOpp0     (equal opportunity TPR difference)
        EOpp1     (equal opportunity TNR difference)
        EOdd      (equalized odds – average of absolute TPR and TNR differences)
        Acc_gap   (accuracy difference: light - dark)
    """
    skin = res['skin']
    known = skin >= 0
    if known.sum() == 0:
        return {
            'DP_diff': float('nan'),
            'EOpp0': float('nan'),
            'EOpp1': float('nan'),
            'EOdd': float('nan'),
            'Acc_gap': float('nan')
        }
    groups = group_light_dark(skin[known])
    preds_known = res['preds'][known]
    labels_known = res['labels'][known]

    # Accuracy gap
    mask_light = groups == 0
    mask_dark  = groups == 1
    acc_light = (preds_known[mask_light] == labels_known[mask_light]).mean() if mask_light.sum() > 0 else np.nan
    acc_dark  = (preds_known[mask_dark]  == labels_known[mask_dark]).mean() if mask_dark.sum()  > 0 else np.nan
    acc_gap = (abs(acc_light - acc_dark)) if (not np.isnan(acc_light) and not np.isnan(acc_dark)) else float('nan')

    return {
        'DP_diff': demographic_parity_diff(preds_known, labels_known, groups),
        'EOpp0':   equal_opportunity_tpr(preds_known, labels_known, groups),
        'EOpp1':   equal_opportunity_tnr(preds_known, labels_known, groups),
        'EOdd':    equalized_odds(preds_known, labels_known, groups),
        'Acc_gap': acc_gap,
    }


# ------------------------------------------------------------
# CSV saving functions (updated with optional knn_acc)
# ------------------------------------------------------------
def save_results_csv(res, fair, split_name, results_dir, label_names, fair_binary=None, knn_acc=None):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Overall (original fairness metrics + optional KNN accuracy)
    overall = {
        "split": split_name,
        "accuracy": res["acc"],
        "auroc": res["auroc"],
        "macro_precision": res["macro_prec"],
        "macro_recall": res["macro_rec"],
        "macro_f1": res["macro_f1"],
        "micro_f1": res["micro_f1"],
        "weighted_f1": res["weighted_f1"],
        "EOM": fair["EOM"],
        "PQD": fair["PQD"],
        "DPM": fair["DPM"],
    }
    if knn_acc is not None:
        overall["knn_accuracy"] = knn_acc
    pd.DataFrame([overall]).to_csv(results_dir / f"{split_name}_overall.csv", index=False)

    # Binary fairness metrics (if provided)
    if fair_binary is not None:
        binary_metrics = {
            "split": split_name,
            "DP_diff": fair_binary["DP_diff"],
            "EOpp0": fair_binary["EOpp0"],
            "EOpp1": fair_binary["EOpp1"],
            "EOdd": fair_binary["EOdd"],
            "Acc_gap": fair_binary["Acc_gap"],
        }
        pd.DataFrame([binary_metrics]).to_csv(results_dir / f"{split_name}_binary_fairness.csv", index=False)

    # Per-class metrics (unchanged)
    per_class = []
    for i in range(len(res["per_class_prec"])):
        per_class.append({
            "split": split_name,
            "class": label_names.get(i, str(i)),
            "precision": res["per_class_prec"][i],
            "recall": res["per_class_rec"][i],
            "f1": res["per_class_f1"][i],
        })
    pd.DataFrame(per_class).to_csv(
        results_dir / f"{split_name}_per_class.csv", index=False
    )

    # Per-FST accuracy (unchanged)
    per_fst = []
    for fst_idx, acc in fair["pg_acc"].items():
        per_fst.append({
            "split": split_name,
            "fitzpatrick_type": f"FST {fst_idx+1}",
            "accuracy": acc if not np.isnan(acc) else None,
        })
    pd.DataFrame(per_fst).to_csv(
        results_dir / f"{split_name}_per_fst.csv", index=False
    )


# ------------------------------------------------------------
# Plotting functions
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
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues",
                ax=axes[1], vmin=0, vmax=1, cbar=False)
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
    ax.bar(x, res["per_class_rec"], width, label="Recall", color="#0096B4")
    ax.bar(x + width, res["per_class_f1"], width, label="F1", color="#DC641E")
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

    fst_acc = [fair["pg_acc"].get(i, float("nan")) for i in range(6)]
    fst_labels = [f"FST {i+1}" for i in range(6)]
    colors = ["#0096B4" if not np.isnan(v) else "#CCCCCC" for v in fst_acc]
    bars = axes[0].bar(fst_labels, [0 if np.isnan(v) else v for v in fst_acc],
                       color=colors, edgecolor="white")
    mean_acc = np.nanmean(fst_acc)
    axes[0].axhline(mean_acc, ls="--", color="#D1333B", label=f"Mean = {mean_acc:.3f}")
    for bar, v in zip(bars, fst_acc):
        if not np.isnan(v):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.01, f"{v:.3f}",
                         ha="center", fontsize=9)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Per-FST Accuracy")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines[["top", "right"]].set_visible(False)

    fair_names = ["EOM (↑)", "PQD (↑)", "DPM (↑)"]
    fair_vals = [fair["EOM"], fair["PQD"], fair["DPM"]]
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

def plot_roc_curve(y_true, y_probs, class_names, title, save_path):
    n_classes = len(class_names)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve((y_true == i).astype(int), y_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Macro-average ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)

    plt.figure(figsize=(10, 8))
    colors = plt.cm.get_cmap('tab10', n_classes)
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color=colors(i), lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    plt.plot(all_fpr, mean_tpr, color='black', lw=2, linestyle='--',
             label=f'Macro-average (AUC = {macro_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def plot_training_curves(history, title, save_path):
    epochs = list(range(1, len(history.get("train_total", [])) + 1))
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for ax, key, panel_title, color in [
        (axes[0, 0], "train_total", "Total Loss", "#1950A0"),
        (axes[0, 1], "train_mi",    "L_MI",       "#0096B4"),
        (axes[0, 2], "train_conf",  "L_conf",     "#D1333B"),
        (axes[0, 3], "train_con",   "L_con",      "#9B59B6"),
    ]:
        if history.get(key):
            ax.plot(epochs, history[key], color=color, lw=2, marker="o", ms=4)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    for ax, t_key, v_key, panel_title, color in [
        (axes[1, 0], "train_acc",      "val_acc",      "Accuracy",     "#1950A0"),
        (axes[1, 1], "train_auroc",    "val_auroc",    "AUROC",        "#0096B4"),
        (axes[1, 2], "train_macro_f1", "val_macro_f1", "Macro-F1",     "#D1333B"),
        (axes[1, 3], "lr",             None,            "Learning Rate", "#9AAABB"),
    ]:
        if history.get(t_key):
            ax.plot(epochs, history[t_key], color=color, lw=2, marker="o", ms=4,
                    label="Train", linestyle="-")
        if v_key and history.get(v_key):
            ax.plot(epochs, history[v_key], color=color, lw=2, marker="s", ms=4,
                    label="Val", linestyle="--", alpha=0.7)
        ax.set_title(panel_title, fontweight="bold")
        ax.set_xlabel("Epoch")
        if t_key != "lr":
            ax.set_ylim(0, 1.05)
        if v_key:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Palettes shared by t-SNE functions ─────────────────────────────────────
_CLS_COLORS = ["#1950A0", "#0096B4", "#DC641E"]   # 3 classes
_CLS_NAMES  = {0: "melanoma", 1: "nevus", 2: "basal cell ca."}
_FST_COLORS = ["#FFEDE0", "#F4C18C", "#D49060", "#A0522D", "#5C3317", "#2B1500"]
_FST_MAP    = {i: f"FST {i+1}" for i in range(6)}
_MOD_COLORS = {0: "#1950A0", 1: "#DC641E"}
_MOD_NAMES  = {0: "Clinical",  1: "Dermoscopic"}
_MOD_MARKERS = {0: "o", 1: "s"}
_MOD_SIZES   = {0: 20,  1: 16}


def _tsne_scatter_labeled(ax, xy, color_ids, palette, labels_map, title,
                          xlabel="t-SNE-1", ylabel="t-SNE-2", s=18, alpha=0.7):
    for cid in sorted(set(color_ids.tolist())):
        mask = color_ids == cid
        ax.scatter(xy[mask, 0], xy[mask, 1], c=palette[cid % len(palette)],
                   s=s, alpha=alpha, label=labels_map.get(cid, str(cid)),
                   edgecolors="none")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.legend(fontsize=7, markerscale=1.4, framealpha=0.6, loc="best", ncol=2)
    ax.spines[["top", "right"]].set_visible(False)


def _run_tsne(embeddings, perplexity=40, seed=42):
    n = embeddings.shape[0]
    if n < 5:
        return None
    perp = min(perplexity, max(5, n // 10))
    try:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp,
                    max_iter=1000, learning_rate='auto', init='pca')
        return tsne.fit_transform(embeddings)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perp,
                    n_iter=1000, learning_rate=200.0, init='pca')
        return tsne.fit_transform(embeddings)


def plot_tsne_class_fst(embeddings, labels, skins, title, save_path,
                        perplexity=40, seed=42):
    e2d = _run_tsne(embeddings, perplexity=perplexity, seed=seed)
    if e2d is None:
        print(f"[SKIP] plot_tsne_class_fst: too few samples ({embeddings.shape[0]})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    _tsne_scatter_labeled(axes[0], e2d, labels, _CLS_COLORS, _CLS_NAMES,
                          "By Disease Class (3 classes)")

    mk_known = skins >= 0
    if mk_known.any():
        _tsne_scatter_labeled(axes[1], e2d[mk_known], skins[mk_known],
                              _FST_COLORS, _FST_MAP, "By Fitzpatrick Skin Type")
    if (~mk_known).any():
        axes[1].scatter(e2d[~mk_known, 0], e2d[~mk_known, 1],
                        c="#CCCCCC", s=8, alpha=0.25, label="FST unknown",
                        edgecolors="none")
        axes[1].legend(fontsize=7, markerscale=1.4, framealpha=0.6, ncol=2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne_modality(embeddings, skins, modalities, title, save_path,
                       perplexity=40, seed=42):
    """
    modalities: array of ints, 0=clinical, 1=dermoscopic
    (paired samples are filtered out before calling this function)
    """
    from matplotlib.colors import ListedColormap as _LC
    from matplotlib.lines import Line2D
    e2d = _run_tsne(embeddings, perplexity=perplexity, seed=seed)
    if e2d is None:
        print(f"[SKIP] plot_tsne_modality: too few samples ({embeddings.shape[0]})")
        return

    fst_cmap = _LC(_FST_COLORS)
    has_multi_mod = len(set(modalities.tolist())) > 1

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    # Left: modality only (colour by modality)
    for mid in [0, 1]:
        mask = modalities == mid
        if mask.any():
            axes[0].scatter(e2d[mask, 0], e2d[mask, 1],
                            c=_MOD_COLORS[mid], s=18, alpha=0.65,
                            label=_MOD_NAMES[mid], edgecolors="none")
    axes[0].set_title(
        "By Modality\n(mixed clusters → modality-invariant)" if has_multi_mod
        else f"By Modality\n(single: {_MOD_NAMES[int(modalities[0])]})",
        fontweight="bold", fontsize=10)
    axes[0].set_xlabel("t-SNE-1"); axes[0].set_ylabel("t-SNE-2")
    axes[0].legend(fontsize=9, markerscale=1.5, framealpha=0.7)
    axes[0].spines[["top", "right"]].set_visible(False)

    # Right: colour = FST, shape = modality
    sc = None
    for mid in [0, 1]:
        m_mask = modalities == mid
        fst_sub = skins[m_mask]
        xy_sub = e2d[m_mask]
        known = fst_sub >= 0
        if known.any():
            sc = axes[1].scatter(xy_sub[known, 0], xy_sub[known, 1],
                                 c=fst_sub[known], cmap=fst_cmap, vmin=0, vmax=5,
                                 marker=_MOD_MARKERS[mid], s=_MOD_SIZES[mid],
                                 alpha=0.7, edgecolors="none")
        if (~known).any():
            axes[1].scatter(xy_sub[~known, 0], xy_sub[~known, 1],
                            c="#CCCCCC", marker=_MOD_MARKERS[mid],
                            s=_MOD_SIZES[mid], alpha=0.25, edgecolors="none")
    if sc is not None:
        plt.colorbar(sc, ax=axes[1], label="FST (0=I ... 5=VI)", shrink=0.85)
    legend_h = [
        Line2D([0], [0], marker="o", color="grey", ms=7, ls="none", label="Clinical"),
        Line2D([0], [0], marker="s", color="grey", ms=7, ls="none", label="Dermoscopic"),
    ]
    axes[1].legend(handles=legend_h, fontsize=8, title="Modality",
                   title_fontsize=8, framealpha=0.7)
    axes[1].set_title("By FST x Modality\n(interleaved → no skin-colour confounding)",
                      fontweight="bold", fontsize=10)
    axes[1].set_xlabel("t-SNE-1"); axes[1].set_ylabel("t-SNE-2")
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_tsne(embeddings, labels, title, save_path, perplexity=40, seed=42):
    """Legacy single-panel t-SNE coloured by class label."""
    e2d = _run_tsne(embeddings, perplexity=perplexity, seed=seed)
    if e2d is None:
        print(f"[SKIP] plot_tsne: too few samples ({embeddings.shape[0]})")
        return
    fig, ax = plt.subplots(figsize=(10, 8))
    _tsne_scatter_labeled(ax, e2d, labels, _CLS_COLORS, _CLS_NAMES, title)
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
        p = res["per_class_prec"][i]
        r = res["per_class_rec"][i]
        f = res["per_class_f1"][i]
        print(f"  {name:<26} {p:6.4f}  {r:6.4f}  {f:6.4f}")

    print(f"\n{'─'*40}")
    print("  Confusion Matrix  (rows=True, cols=Pred)")
    print(f"{'─'*40}")
    short = [label_names.get(i, str(i))[:5].ljust(5)
             for i in range(len(res["per_class_prec"]))]
    print("       " + "  ".join(short))
    for i, row in enumerate(res["conf_mat"]):
        row_vals = "  ".join(f"{v:5d}" for v in row)
        print(f"  {short[i]}  {row_vals}")

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
        bar = "█" * int((v if not np.isnan(v) else 0) * 20)
        note = "" if not np.isnan(v) else f"  ← n={n_grp} (no samples)"
        print(f"  FST {g+1}  n={n_grp:>4}  {v:.4f}  {bar}{note}")
    print(f"\n{'='*60}\n")