"""
evaluation.py

Provides functions for:
- Running validation (collecting predictions, computing metrics)
- Fairness metrics (EOM, PQD, DPM, per-FST accuracy)
- Plotting: confusion matrix, per-class metrics, fairness summary, training curves, t-SNE
- Saving results as CSV files (overall, per-class, per-FST)

All functions are designed to be imported and reused across different model training scripts.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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


# ------------------------------------------------------------
# Helper: robust macro AUROC
# ------------------------------------------------------------
def robust_macro_auroc(probs, labels):
    """Compute macro AUROC over classes that appear in labels."""
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
def validate(model, loader, device, num_classes=5):
    """
    Run model on loader and return a dictionary with all metrics.
    """
    model.eval()
    all_probs, all_labels, all_skins = [], [], []
    for batch in loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
        out = model(batch)
        probs = F.softmax(out["logits"], dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(batch["label"].cpu().numpy())
        all_skins.append(batch["skin_type"].cpu().numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    skins = np.concatenate(all_skins)
    preds = probs.argmax(axis=1)

    acc = (preds == labels).mean()
    auroc = robust_macro_auroc(probs, labels)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
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
# Fairness metrics (no FATE, as baseline does not need it)
# ------------------------------------------------------------
def pg_acc(preds, labels, groups, K=6):
    """Per-group accuracy."""
    out = {}
    for g in range(K):
        mask = groups == g
        if mask.sum() == 0:
            out[g] = float("nan")
        else:
            out[g] = (preds[mask] == labels[mask]).mean()
    return out


def eom(preds, labels, groups, K=6):
    """Equality of Opportunity Metric."""
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
    """Predictive Quality Disparity."""
    vals = [v for v in pg_acc_dict.values() if not np.isnan(v)]
    if len(vals) < 2:
        return float("nan")
    mn, mx = min(vals), max(vals)
    return float("nan") if mx == 0 else mn / mx


def dpm(preds, labels, groups, K=6):
    """Demographic Parity Metric."""
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
    """Compute all fairness metrics from validation result dict."""
    pg = pg_acc(res["preds"], res["labels"], res["skin"], K)
    return {
        "pg_acc": pg,
        "EOM": eom(res["preds"], res["labels"], res["skin"], K),
        "PQD": pqd(pg),
        "DPM": dpm(res["preds"], res["labels"], res["skin"], K),
    }


# ------------------------------------------------------------
# CSV saving functions
# ------------------------------------------------------------
def save_results_csv(res, fair, split_name, results_dir, label_names):
    """
    Save overall, per-class, and per-FST results to CSV files.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics
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
    pd.DataFrame([overall]).to_csv(results_dir / f"{split_name}_overall.csv", index=False)

    # Per-class metrics
    per_class = []
    for i in range(len(res["per_class_prec"])):
        per_class.append({
            "split": split_name,
            "class": label_names.get(i, str(i)),
            "precision": res["per_class_prec"][i],
            "recall": res["per_class_rec"][i],
            "f1": res["per_class_f1"][i],
        })
    pd.DataFrame(per_class).to_csv(results_dir / f"{split_name}_per_class.csv", index=False)

    # Per-FST accuracy
    per_fst = []
    for fst_idx, acc in fair["pg_acc"].items():
        per_fst.append({
            "split": split_name,
            "fitzpatrick_type": f"FST {fst_idx+1}",
            "accuracy": acc if not np.isnan(acc) else None,
        })
    pd.DataFrame(per_fst).to_csv(results_dir / f"{split_name}_per_fst.csv", index=False)


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
    sns.heatmap(conf_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[1], vmin=0, vmax=1, cbar=False)
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

    # Per-FST accuracy
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

    # Summary fairness
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


def plot_training_curves(history, title, save_path):
    """Plot training loss, accuracy, and AUROC curves."""
    epochs = list(range(1, len(history.get("train_total", [])) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # Loss
    ax = axes[0]
    if "train_total" in history:
        ax.plot(epochs, history["train_total"], color="#1950A0", lw=2, marker="o", ms=4, label="Train Loss")
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Accuracy & AUROC
    ax = axes[1]
    if "train_acc" in history:
        ax.plot(epochs, history["train_acc"], color="#1950A0", lw=2, marker="o", ms=4, label="Train Acc")
    if "val_acc" in history:
        ax.plot(epochs, history["val_acc"], color="#0096B4", lw=2, marker="s", ms=4, label="Val Acc", linestyle="--")
    if "val_auroc" in history:
        ax.plot(epochs, history["val_auroc"], color="#DC641E", lw=2, marker="^", ms=4, label="Val AUROC", linestyle=":")
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
    """t-SNE visualization of embeddings."""
    if embeddings.shape[0] < 5:
        print(f"[SKIP] t-SNE: too few samples ({embeddings.shape[0]})")
        return
    perp = min(perplexity, max(5, embeddings.shape[0] // 10))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp, n_iter=1000, init="pca")
    emb_2d = tsne.fit_transform(embeddings)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=labels, cmap="tab10", s=20, alpha=0.7)
    plt.colorbar(scatter, ticks=range(labels.max()+1), label="Class")
    plt.title(title)
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def print_full_report(res, fair, split_name, label_names):
    """Print a formatted evaluation report to console."""
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
    short = [label_names.get(i, str(i))[:5].ljust(5) for i in range(len(res["per_class_prec"]))]
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