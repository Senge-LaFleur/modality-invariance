# Modality-Invariant Representation Learning for Fair Skin Disease Classification

This repository contains the official implementation of **FairDisCo** (Fair and Modality-Invariant Representation Learning) for skin disease classification using both clinical and dermoscopic images. It includes baseline models (ResNet‑18, ViT) and their modality‑invariant counterparts with multi‑objective losses.

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Training Scripts](#training-scripts)
- [Evaluation & Visualization](#evaluation--visualization)
- [Results](#results)
- [License](#license)

## Overview

We address skin disease classification under domain shift (clinical vs. dermoscopic) and demographic bias (Fitzpatrick skin type). The proposed **Modality-Invariance** method learns a shared embedding space invariant to imaging modality and uncorrelated with skin type, while maintaining high classification accuracy. The framework uses:

- Dual‑encoder architecture (ResNet‑18 or ViT)
- Multi‑objective loss:  
  - Lcls (label‑smoothed weighted cross‑entropy)  
  - Lconf (confusion loss)  
  - Lcon (supervised contrastive loss)  
  - LMI (modality invariance loss)
- Stage‑wise learning rate decay and MixUp regularisation

## Datasets

We use four public datasets, harmonised to a unified 5‑class taxonomy:

| Dataset | Modality | Role | Official Download Link | Kaggle Dataset Link | 
|---------|----------|------|  |  |
| HIBA | Clinical + Dermoscopic (paired) | Paired training |  |  |
| Fitzpatrick17k | Clinical | Unpaired clinical training |  |  |
| HAM10000 | Dermoscopic | Unpaired dermoscopic training |  |  |
| Derm7pt | Clinical + Dermoscopic | Cross‑dataset evaluation |  |  |

The five classes are: **Melanoma, Nevus, Basal Cell Carcinoma, Actinic Keratosis, Squamous Cell Carcinoma**.

## Installation

```bash
### 1. Clone the repository
git clone https://github.com/Senge-LaFleur/modality-invariance.git

### 2. Create and activate a Conda environment
conda create -n modality-invariance python=3.9 -y
conda activate modality-invariance

cd modality-invariance   # Path to the modality-invariance folder

### 3. Install PyTorch with CUDA support (adjust for your system)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

### 4. Install remaining dependencies
pip install -r requirements.txt
```

## Data Preprocessing
1. Download the datasets from the respective sources.
2. Run the notebook "data_preprocessing.ipynb" to preprocess the datasets. Make sure to adjust the paths to match your system.

## Training Scripts
```bash
# Training Baselines
python train_baseline_resnet18.py
python train_baseline_vit.py

# Training Modality-Invariant Models
python train_modality_invariance_resnet18.py
python train_modality_invariance_vit.py

# Train the other scripts the same way and make sure to save them in the same directory structure
```

## Evaluation
Run the notebook `evaluation.ipynb` to evaluate the models. Make sure to update the paths in the notebook to point to the correct result directory depending on which model you are evaluating.
