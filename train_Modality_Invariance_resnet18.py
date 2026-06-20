#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modality-Invariant CNN (ResNet-18) for Skin Disease Classification
3-class setup: melanoma / nevus / basal cell carcinoma
Training data: HIBA (paired) + Derm7pt (paired)
Cross-eval:    Fitzpatrick17k (clinical) · PAD-UFES-20 (clinical) · ISIC2019 (derm)

Uses multi-objective losses: Lcls, Lconf, Lcon, LMI, Lkl.

All outputs (checkpoints, results, logs) are saved in:
    - checkpoints_Modality_Invariance_resnet18/
    - results_Modality_Invariance_resnet18/
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

from sklearn.metrics import f1_score
from models.models_losses import (
    DualResNet18,
    SupConLoss,
    cross_modal_supcon_loss,
    confusion_loss,
    skin_type_loss,
    mi_loss,
    symmetric_kl_loss,
    mixup_embeddings,
    cls_loss_fn,
    compute_class_weights,
    get_layer_wise_lr_params,
)

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
    plot_tsne_class_fst,
    plot_tsne_modality,
    plot_roc_curve,
    compute_knn_accuracy,
    build_loaders,
    LABEL_NAMES,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# ============================================================
# PATH CONFIGURATION  — update these for each environment
# ============================================================

WORK_ROOT = Path('/kaggle/working/modality-invariance/process/process/outputs')
CSV_DIR = WORK_ROOT / 'csvs'

IMAGE_ROOTS = {
    'hiba':           Path('/kaggle/input/datasets/asosenge/hibaskinlesionsdataset-main/HIBASkinLesionsDataset-main/images'),
    'derm7pt':        Path('/kaggle/input/datasets/asosenge/derm7pt/release_v0/images'),
    'fitzpatrick17k': Path('/kaggle/input/datasets/asosenge/fitzpatrick17k/fitzpatrick17k/data/finalfitz17k'),
    'padufes20':      Path('/kaggle/input/datasets/mahdavi1202/skin-cancer'),             
    'isic2019':       Path('/kaggle/input/datasets/sengenjih/isic2019'),
}

print("Checking configured paths:")
print(f"  WORK_ROOT : {WORK_ROOT}  {'[OK]' if WORK_ROOT.exists() else '[MISSING]'}")
print(f"  CSV_DIR   : {CSV_DIR}  {'[OK]' if CSV_DIR.exists() else '[MISSING]'}")
for name, root in IMAGE_ROOTS.items():
    print(f"  {name:<15}: {root}  {'[OK]' if root.exists() else '[MISSING — update IMAGE_ROOTS]'}")

CFG = {
    'csv_dir':      CSV_DIR,
    'image_roots':  IMAGE_ROOTS,
    'ckpt_dir':     WORK_ROOT / 'checkpoints_Modality_Invariance_resnet18',
    'results_dir':  WORK_ROOT / 'results_Modality_Invariance_resnet18',

    'backbone':        'resnet18',
    'embed_dim':       512,
    'img_size':        224,
    'num_classes':     3,
    'num_skin_types':  6,

    'batch_size':      32,
    'num_epochs':      50,     # update as needed
    'lr': 1e-4,
    'min_lr': 1e-6,
    # Applying the same overfitting-prevention philosophy proactively
    # (ported from the ViT run, which showed tr_acc=1.0 with declining
    # val_acc by epoch 100 — see chat history). ResNet18 has fewer
    # parameters than two ViT-Small backbones, so this is a moderate,
    # not aggressive, starting point — watch tr_acc vs val_acc on the
    # first run and adjust if ResNet18 doesn't show the same overfitting
    # severity (it may not, given its smaller capacity).
    'weight_decay': 5e-3,
    'warmup_epochs': 5,      # update as needed
    'aug_probability': 0.85,   # FIX: now actually applied (see evaluation.py
                               # build_loaders — was previously dead config)

    # Overfitting-prevention, CNN-appropriate equivalent of DualViT's
    # drop_path_rate: nn.Dropout2d inserted after each residual stage
    # (see DualResNet18 in models_losses.py for the full rationale).
    # classifier_dropout matches DualViT's value for consistency across
    # backbones.
    'drop_block_rate':    0.1,
    'classifier_dropout': 0.4,

    'lambda_cls':      1.0,
    'lambda_conf':     0.5,
    'lambda_skin':     0.3,
    'lambda_con':      1.0,
    'lambda_mi':       1.0,
    # Alignment-through-classification (symmetric KL between clinical/derm
    # classifier predictions for the same lesion) — see symmetric_kl_loss
    # docstring in models_losses.py. Kept below lambda_cls deliberately so
    # the optimizer can't satisfy Lkl via low-confidence collapse.
    'lambda_kl':       0.35,
    'temperature':     0.1,
    'label_smoothing': 0.01,
    'mixup_alpha':     0.4,

    'use_conf':  True,
    'use_con':   True,
    'use_mi':    True,
    'use_kl':    True,
    'use_mixup': False,

}

CFG["ckpt_dir"].mkdir(parents=True, exist_ok=True)
CFG["results_dir"].mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Training epoch function
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, cfg, epoch, scaler, class_weights, device):
    model.train()
    totals = dict(total=0., cls=0., conf=0., skin=0., con=0., mi=0., kl=0.)
    all_preds, all_labels = [], []
    n_batches = 0

    # FIX: replaced the old `paired_rate` (fraction of batches touching
    # >=1 pair, which read ~1.0 even when most images never got
    # cross-modal loss signal) with a metric that actually answers
    # "how many of this epoch's pair-eligible images had their sibling
    # available in the same batch to compute Lcon/LMI on". Tracked at the
    # image level, not the batch level.
    n_pairable_images_seen = 0   # images that came from a paired row (have a pair_id)
    n_pairable_images_matched = 0  # of those, how many found their sibling in-batch

    # Lightweight diagnostic for loss_kl specifically: of the matched
    # clinical/derm pairs seen this epoch, what fraction did the
    # classifier predict the SAME class for? Should trend toward 1.0 as
    # training progresses if loss_kl is doing its job.
    n_kl_pairs_seen = 0
    n_kl_pairs_agree = 0

    sup_con = SupConLoss(cfg["temperature"]).to(device)
    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights else None
    )

    pbar = tqdm(loader, desc=f"Ep {epoch+1:>3} [train]", unit="batch",
                dynamic_ncols=True, leave=False)
    for batch in pbar:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(batch)
            labels = batch["label"]
            skin_types = batch["skin_type"]

            use_mixup = cfg.get("use_mixup", True)
            if use_mixup and out["z"].requires_grad:
                z_mix, y_a, y_b, lam = mixup_embeddings(
                    out["z"], labels, alpha=cfg["mixup_alpha"]
                )
                logits_mix = model.classifier(z_mix)
                loss_cls = (
                    lam * cls_loss_fn(logits_mix, y_a, weight_tensor, cfg["label_smoothing"])
                    + (1.0 - lam) * cls_loss_fn(logits_mix, y_b, weight_tensor, cfg["label_smoothing"])
                )
            else:
                loss_cls = cls_loss_fn(
                    out["logits"], labels, weight_tensor, cfg["label_smoothing"]
                )

            loss_conf = confusion_loss(out["skin_logits"]) if cfg.get("use_conf") else 0.0
            loss_skin = (
                skin_type_loss(model.skin_clf(out["z"].detach()), skin_types)
                if cfg.get("use_conf") else 0.0
            )

            # ------------------------------------------------------------
            # FIX: model.forward() no longer special-cases pairing — every
            # sample (clinical or derm) gets its own row in out["z"].
            # To recover (z_c, z_d) for the SAME lesion for Lcon/LMI, we
            # regroup by `pair_id` + `modality` here instead. With
            # PairAwareBatchSampler guaranteeing both halves of every
            # sampled pair co-occur in this batch, `common_pids` below
            # should be non-empty whenever the batch contains any paired
            # data — but we don't assume that; we measure it directly via
            # n_pairable_images_seen / n_pairable_images_matched so a
            # regression in the sampler shows up immediately in logs
            # instead of silently degrading Lcon/LMI like before.
            # ------------------------------------------------------------
            pair_ids = batch["pair_id"]          # list, None for unpaired samples
            modality = batch["modality"]         # list of 'clinical' / 'derm'

            clin_idx_by_pid, derm_idx_by_pid = {}, {}
            for i, (pid, m) in enumerate(zip(pair_ids, modality)):
                if pid is None:
                    continue
                n_pairable_images_seen += 1
                if m == "clinical":
                    clin_idx_by_pid[pid] = i
                else:
                    derm_idx_by_pid[pid] = i
            common_pids = sorted(set(clin_idx_by_pid) & set(derm_idx_by_pid))
            n_pairable_images_matched += 2 * len(common_pids)

            z_c = z_d = paired_labels = None
            if common_pids:
                c_idx = torch.tensor([clin_idx_by_pid[p] for p in common_pids], device=device)
                d_idx = torch.tensor([derm_idx_by_pid[p] for p in common_pids], device=device)
                z_c = out["z"][c_idx]
                z_d = out["z"][d_idx]
                paired_labels = labels[c_idx]   # clinical/derm sibling share the same label

            loss_con = 0.0
            if cfg.get("use_con") and z_c is not None and z_c.size(0) > 1:
                loss_con = cross_modal_supcon_loss(z_c, z_d, paired_labels, cfg["temperature"])

            # --- VICReg-based MI loss (collapse-resistant) ---
            loss_mi = 0.0
            if cfg.get("use_mi") and z_c is not None and z_c.size(0) > 1:
                loss_mi = mi_loss(z_c, z_d)

            # ------------------------------------------------------------
            # Alignment-through-classification (symmetric KL). Reuses the
            # SAME c_idx/d_idx already computed above for loss_con/loss_mi.
            # Operates on out["logits"] (the classifier's actual
            # predictions), not raw z, so alignment pressure flows through
            # the same parameters that produce the classification
            # decision. See symmetric_kl_loss docstring in
            # models_losses.py for the collapse-mode analysis and why
            # lambda_kl must stay below lambda_cls.
            # ------------------------------------------------------------
            loss_kl = 0.0
            if cfg.get("use_kl") and z_c is not None and z_c.size(0) > 1:
                logits_c = out["logits"][c_idx]
                logits_d = out["logits"][d_idx]
                loss_kl = symmetric_kl_loss(logits_c, logits_d)
                with torch.no_grad():
                    pred_c = logits_c.argmax(dim=-1)
                    pred_d = logits_d.argmax(dim=-1)
                    n_kl_pairs_seen += pred_c.size(0)
                    n_kl_pairs_agree += (pred_c == pred_d).sum().item()

            total_loss = cfg["lambda_cls"] * loss_cls
            if cfg.get("use_conf"):
                total_loss += cfg["lambda_conf"] * loss_conf + cfg["lambda_skin"] * loss_skin   # FIX: explicit lambda
            if cfg.get("use_con"):
                total_loss += cfg["lambda_con"] * loss_con
            if cfg.get("use_mi") and loss_mi != 0:
                total_loss += cfg["lambda_mi"] * loss_mi
            if cfg.get("use_kl") and loss_kl != 0:
                total_loss += cfg["lambda_kl"] * loss_kl

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        totals["total"] += total_loss.item()
        totals["cls"]   += loss_cls.item()  if isinstance(loss_cls,  torch.Tensor) else loss_cls
        totals["conf"]  += loss_conf.item() if isinstance(loss_conf, torch.Tensor) else loss_conf
        totals["skin"]  += loss_skin.item() if isinstance(loss_skin, torch.Tensor) else loss_skin
        totals["con"]   += loss_con.item()  if isinstance(loss_con,  torch.Tensor) else loss_con
        totals["mi"]    += loss_mi.item()   if isinstance(loss_mi,   torch.Tensor) else loss_mi
        totals["kl"]    += loss_kl.item()   if isinstance(loss_kl,   torch.Tensor) else loss_kl
        n_batches += 1

        with torch.no_grad():
            preds = out["logits"].argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(batch["label"].cpu().numpy())

        pbar.set_postfix(loss=f"{totals['total']/n_batches:.4f}")

    pbar.close()
    for k in totals:
        totals[k] /= max(n_batches, 1)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    totals["acc"] = (all_preds == all_labels).mean()
    totals["macro_f1"] = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # FIX: image-level pair-match rate. 1.0 means every pairable image
    # found its clinical/derm sibling in the same batch this epoch (the
    # PairAwareBatchSampler invariant holding); anything less is a real
    # signal that Lcon/LMI are running on fewer pairs than they should.
    totals["pair_match_rate"] = n_pairable_images_matched / max(n_pairable_images_seen, 1)

    # Fraction of matched clinical/derm pairs the classifier agreed on
    # this epoch. Track across epochs — should trend up toward 1.0 if
    # loss_kl is actually aligning the two modalities through the
    # classifier.
    totals["kl_agreement_rate"] = n_kl_pairs_agree / max(n_kl_pairs_seen, 1)

    return totals


# ------------------------------------------------------------
# Main training and evaluation pipeline
# ------------------------------------------------------------
def main():
    print(f"CSV dir:        {CFG['csv_dir']}")
    print(f"Checkpoints:    {CFG['ckpt_dir']}")
    print(f"Results:        {CFG['results_dir']}")
    print(f"Num classes:    {CFG['num_classes']}  ({', '.join(LABEL_NAMES.values())})")
    print("Dataset roots:")
    for name, root in CFG['image_roots'].items():
        print(f"  {name:<15}: {root}")

    train_loader, val_loader, test_loader, eval_loaders = build_loaders(CFG, seed=SEED)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    if test_loader:
        print(f"Test batches: {len(test_loader)}")
    print(f"Cross-eval loaders: {list(eval_loaders.keys())}")

    class_weights = compute_class_weights(CFG["csv_dir"], CFG["num_classes"])
    if class_weights:
        print(f"Class weights: {np.round(class_weights, 3)}")
    else:
        class_weights = None

    model = DualResNet18(
        embed_dim=CFG["embed_dim"],
        num_classes=CFG["num_classes"],
        num_skin_types=CFG["num_skin_types"],
        pretrained=True,
        use_projection=True,
        drop_block_rate=CFG["drop_block_rate"],
        classifier_dropout=CFG["classifier_dropout"],
    ).to(DEVICE)

    param_groups = get_layer_wise_lr_params(model, base_lr=CFG["lr"], lr_decay=0.85)
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=CFG["weight_decay"], betas=(0.9, 0.999), eps=1e-8
    )

    def lr_lambda(epoch):
        if epoch < CFG["warmup_epochs"]:
            return (epoch + 1) / CFG["warmup_epochs"]
        progress = (epoch - CFG["warmup_epochs"]) / max(
            1, CFG["num_epochs"] - CFG["warmup_epochs"]
        )
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        min_frac = CFG["min_lr"] / CFG["lr"]
        return max(min_frac, cos)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    start_epoch = 0
    best_auroc = 0.0
    best_f1 = 0.0
    history = defaultdict(list)


    for epoch in range(start_epoch, CFG["num_epochs"]):
        train_metrics = train_epoch(
            model, train_loader, optimizer, CFG, epoch, scaler, class_weights, DEVICE
        )
        scheduler.step()
        val_metrics = validate(
            model, val_loader, DEVICE, CFG["num_classes"], desc="Validation"
        )
        lr = optimizer.param_groups[0]["lr"]

        for k, v in train_metrics.items():
            history[f"train_{k}"].append(float(v))
        for k in ["acc", "auroc", "macro_f1", "weighted_f1"]:
            history[f"val_{k}"].append(float(val_metrics[k]))
        history["lr"].append(float(lr))

        print(
            f"Ep {epoch+1:3d}/{CFG['num_epochs']}  "
            f"loss={train_metrics['total']:.4f}  tr_acc={train_metrics['acc']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  val_auroc={val_metrics['auroc']:.4f}  "
            f"val_f1={val_metrics['macro_f1']:.4f}  "
            f"pair_match={train_metrics['pair_match_rate']:.2f}  "
            f"kl_agree={train_metrics['kl_agreement_rate']:.2f}  lr={lr:.2e}"
        )

        ckpt_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": dict(history),
        }
        torch.save(ckpt_state, CFG["ckpt_dir"] / "last_model.pt")
        if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt",
                        CFG["ckpt_dir"] / "best_auroc_model.pt")
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            shutil.copy(CFG["ckpt_dir"] / "last_model.pt",
                        CFG["ckpt_dir"] / "best_f1_model.pt")
        if (epoch + 1) % 5 == 0:
            torch.save(ckpt_state, CFG["ckpt_dir"] / f"checkpoint_ep{epoch+1:03d}.pt")

        with open(CFG["results_dir"] / "history.json", "w") as f:
            json.dump(
                {k: [float(x) for x in v] for k, v in history.items()}, f, indent=2
            )

    print(f"Training complete. Best AUROC: {best_auroc:.4f}, Best F1: {best_f1:.4f}")

    # Load best model for final evaluation
    best_ckpt = CFG["ckpt_dir"] / "best_f1_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "best_auroc_model.pt"
    if not best_ckpt.exists():
        best_ckpt = CFG["ckpt_dir"] / "last_model.pt"
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded best model from {best_ckpt.name} (epoch {ckpt['epoch']+1})")

    class_names = [LABEL_NAMES[i] for i in range(CFG["num_classes"])]

    # ── Validation split ────────────────────────────────────────────────────
    val_res = validate(model, val_loader, DEVICE, CFG["num_classes"],
                       desc="Validation (final)")
    val_fair = fairness(val_res)
    val_fair_binary = fairness_binary(val_res)
    save_results_csv(val_res, val_fair, "val", CFG["results_dir"], LABEL_NAMES)
    plot_confusion_matrix(val_res["conf_mat"], class_names,
                          "Confusion Matrix - Validation",
                          CFG["results_dir"] / "val_confusion.png")
    plot_per_class_metrics(val_res, class_names,
                           "Per-Class Metrics - Validation",
                           CFG["results_dir"] / "val_per_class.png")
    plot_fairness_metrics(val_fair, "Fairness - Validation",
                          CFG["results_dir"] / "val_fairness.png")
    print("\nBinary fairness (Validation):")
    print(f"  DP_diff  : {val_fair_binary['DP_diff']:.4f}")
    print(f"  EOpp0    : {val_fair_binary['EOpp0']:.4f}")
    print(f"  EOpp1    : {val_fair_binary['EOpp1']:.4f}")
    print(f"  EOdd     : {val_fair_binary['EOdd']:.4f}")
    print(f"  Acc_gap  : {val_fair_binary['Acc_gap']:.4f}")
    plot_roc_curve(val_res["labels"], val_res["probs"], class_names,
                   "ROC Curves - Validation",
                   CFG["results_dir"] / "val_roc.png")

    # ── Internal test split ─────────────────────────────────────────────────
    if test_loader:
        test_res = validate(model, test_loader, DEVICE, CFG["num_classes"], desc="Test")
        test_fair = fairness(test_res)
        test_fair_binary = fairness_binary(test_res)

        # ---- Compute KNN accuracy on test embeddings ----
        model.eval()
        all_embs = []
        all_labels_tsne = []
        all_skins_tsne = []
        # FIX: batch["modality"] is now always 'clinical' or 'derm' for every
        # sample (paired rows used to tag samples 'paired' and silently fall
        # into the `else: -1 (exclude)` branch below, meaning this t-SNE/KNN
        # modality-invariance plot was previously built ONLY from the small
        # unpaired residual pools — never from HIBA/Derm7pt pairs at all).
        # Now every test sample, including ones from paired rows, is plotted.
        all_mods_tsne = []   # 0=clinical, 1=derm
        with torch.no_grad():
            for batch in test_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                out = model(batch)
                b = out["z"].size(0)
                all_embs.append(out["z"].cpu().numpy())
                all_labels_tsne.append(batch["label"].cpu().numpy())
                all_skins_tsne.append(batch["skin_type"].cpu().numpy())

                mod_list = [0 if m == "clinical" else 1
                            for m in batch.get("modality", ["clinical"] * b)]
                all_mods_tsne.append(np.array(mod_list, dtype=np.int64))

        embs        = np.concatenate(all_embs)
        labels_tsne = np.concatenate(all_labels_tsne)
        skins_tsne  = np.concatenate(all_skins_tsne)
        mods_tsne   = np.concatenate(all_mods_tsne)

        knn_acc = compute_knn_accuracy(embs, labels_tsne, k=5)
        print(f"\n[Modality-Invariant ResNet-18] Test KNN (k=5) accuracy: {knn_acc:.4f}")

        save_results_csv(test_res, test_fair, "test", CFG["results_dir"], LABEL_NAMES, knn_acc=knn_acc)
        plot_confusion_matrix(test_res["conf_mat"], class_names,
                              "Confusion Matrix - Test",
                              CFG["results_dir"] / "test_confusion.png")
        plot_per_class_metrics(test_res, class_names,
                               "Per-Class Metrics - Test",
                               CFG["results_dir"] / "test_per_class.png")
        plot_fairness_metrics(test_fair, "Fairness - Test",
                              CFG["results_dir"] / "test_fairness.png")
        print("\nBinary fairness (Test):")
        print(f"  DP_diff  : {test_fair_binary['DP_diff']:.4f}")
        print(f"  EOpp0    : {test_fair_binary['EOpp0']:.4f}")
        print(f"  EOpp1    : {test_fair_binary['EOpp1']:.4f}")
        print(f"  EOdd     : {test_fair_binary['EOdd']:.4f}")
        print(f"  Acc_gap  : {test_fair_binary['Acc_gap']:.4f}")
        plot_roc_curve(test_res["labels"], test_res["probs"], class_names,
                       "ROC Curves - Test",
                       CFG["results_dir"] / "test_roc.png")

        # t-SNE plots
        plot_tsne_class_fst(
            embs, labels_tsne, skins_tsne,
            title="t-SNE — Shared Embedding Space  [Internal Test]",
            save_path=CFG["results_dir"] / "tsne_test_class_fst.png",
        )

        mask_mod = mods_tsne >= 0
        if mask_mod.sum() > 1:
            plot_tsne_modality(
                embs[mask_mod], skins_tsne[mask_mod], mods_tsne[mask_mod],
                title="t-SNE — Modality-Invariance  [Internal Test]",
                save_path=CFG["results_dir"] / "tsne_test_modality_invariance.png",
            )
        else:
            print("[WARN] Not enough unpaired clinical/derm samples for modality t-SNE plot.")

    # ── Cross-dataset evaluation ────────────────────────────────────────────
    cross_results = {}
    for ds_name, loader in eval_loaders.items():
        print(f"\nEvaluating on {ds_name}")
        res = validate(model, loader, DEVICE, CFG["num_classes"],
                       desc=f"Cross-eval: {ds_name}")
        fair = fairness(res)
        fair_binary = fairness_binary(res)
        save_results_csv(res, fair, f"cross_{ds_name}", CFG["results_dir"], LABEL_NAMES)
        plot_confusion_matrix(res["conf_mat"], class_names,
                              f"Confusion Matrix - {ds_name}",
                              CFG["results_dir"] / f"cross_{ds_name}_confusion.png")
        plot_per_class_metrics(res, class_names,
                               f"Per-Class Metrics - {ds_name}",
                               CFG["results_dir"] / f"cross_{ds_name}_per_class.png")
        plot_fairness_metrics(fair, f"Fairness - {ds_name}",
                              CFG["results_dir"] / f"cross_{ds_name}_fairness.png")
        print(f"\nBinary fairness ({ds_name}):")
        print(f"  DP_diff  : {fair_binary['DP_diff']:.4f}")
        print(f"  EOpp0    : {fair_binary['EOpp0']:.4f}")
        print(f"  EOpp1    : {fair_binary['EOpp1']:.4f}")
        print(f"  EOdd     : {fair_binary['EOdd']:.4f}")
        print(f"  Acc_gap  : {fair_binary['Acc_gap']:.4f}")
        plot_roc_curve(res["labels"], res["probs"], class_names,
                       f"ROC Curves - {ds_name}",
                       CFG["results_dir"] / f"cross_{ds_name}_roc.png")
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

    # ── Training curves ─────────────────────────────────────────────────────
    plot_training_curves(history, "Training History (Modality-Invariant ResNet-18)",
                         CFG["results_dir"] / "training_curves.png")

    print(f"\nAll results saved to {CFG['results_dir']}")
    print(f"Checkpoints saved to {CFG['ckpt_dir']}")


if __name__ == "__main__":
    main()