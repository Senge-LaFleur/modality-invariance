# Modality-Invariant Skin Disease Classification

A PyTorch-based framework for training skin disease classifiers with modality invariance, incorporating clinical + dermoscopic data, skin-type fairness, and explainability (SHAP).

## Project Structure

```
modality-invariance/
├── .vscode/
├── .venv/
├── .gitignore
├── environment.yml
├── data/
│   └── datasets/
│       ├── asosenge/hibaskinlesionsdataset-main/
│       ├── asosenge/fitzpatrick17k/
│       ├── asosenge/ham10000/
│       └── shubhamgoel27/dermnet/
├── logs/
├── outputs/
│   ├── csvs/
│   │   └── ...
│   ├── checkpoints/
│   │   └── ...
│   └── results/
│       └── ...
├── modality-invariance-v1-1.ipynb  # Notebook for staging + training + evaluation
├── modality_invariance_v1-1.py   # Compiled script from notebook
├── README.md
└── job_modality-invariance.sh      # SLURM job script
```


Links to datasets:
https://www.kaggle.com/datasets/asosenge/hibaskinlesionsdataset-main
https://www.kaggle.com/datasets/asosenge/fitzpatrick17k
https://www.kaggle.com/datasets/asosenge/ham10000
https://www.kaggle.com/datasets/shubhamgoel27/dermnet
