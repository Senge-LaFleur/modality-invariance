#!/bin/bash
#SBATCH --job-name=train_No_MI_vit
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=bigpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jerry.lacmou.zeutouo@u-picardie.fr

# ── Environment ───────────────────────────────────────────────────────────────
set -euo pipefail
echo "========================================="
echo "Job:       $SLURM_JOB_NAME  ($SLURM_JOB_ID)"
echo "Node:      $(hostname)"
echo "Started:   $(date)"
echo "========================================="

module purge
module load cuda/12.6
#module load cudnn/8.9
module load python/3.11.7

WORK_DIR="20260623-process"

SCRIPT_NAME="../train_No_MI_vit.py"       

export DATA_ROOT="$WORK_DIR/data/datasets"

export WORK_ROOT="$WORK_DIR/outputs_No_MI_vit"

export BACKBONE_CACHE="$WORK_DIR/backbone_cache"
export TORCH_HOME="$BACKBONE_CACHE"
export HF_HOME="$BACKBONE_CACHE"
export HUGGINGFACE_HUB_CACHE="$BACKBONE_CACHE"

HF_TOKEN_FILE="$HOME/.hf_token"
if [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE")
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "[INFO] HF token loaded."
else
    echo "[INFO] No HF token found at $HF_TOKEN_FILE — downloads may be slow."
fi

# ── Kaggle credentials setup ──────────────────────────────────────────────────
if [ -f "$WORK_DIR/kaggle.json" ]; then
    # Use the kaggle.json file included in the project directory
    export KAGGLE_CONFIG_DIR="$WORK_DIR"
    chmod 600 "$WORK_DIR/kaggle.json"
    echo "[INFO] Kaggle token loaded from $WORK_DIR/kaggle.json."
else
    echo "[INFO] No kaggle.json found in $WORK_DIR. Relying on ~/.kaggle/ or env vars."
fi

# ── Create required directories and copy preprocessed data and helper scripts ──
mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_ROOT"

# Copy preprocessed CSVs and numpy embeddings from the repository to the job's outputs directory
cp -r "../csvs" "$WORK_ROOT/csvs"
if [ -f "../process/process/outputs/text_embeddings_3_large_consecutive_averaged.npy" ]; then
    cp "../process/process/outputs/text_embeddings_3_large_consecutive_averaged.npy" "$WORK_ROOT/"
fi

# Copy helper python models and scripts from the repository to the job's working directory
cp -r "../models" "$WORK_DIR/models"
if [ -f "../GOT.py" ]; then
    cp "../GOT.py" "$WORK_DIR/GOT.py"
fi

mkdir -p "$WORK_ROOT/checkpoints"
mkdir -p "$WORK_ROOT/results"
mkdir -p "$BACKBONE_CACHE"

cd "$WORK_DIR"

# ── Virtual environment setup ─────────────────────────────────────────────────
VENV_DIR="$WORK_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[SETUP] Creating virtual environment..."
    python -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[SETUP] Installing / verifying Python dependencies..."
pip install --upgrade pip --quiet
#pip uninstall -y torch torchvision

pip install --quiet \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu126

pip install --quiet \
    timm \
    einops \
    scikit-learn \
    umap-learn \
    matplotlib \
    seaborn \
    pandas \
    tqdm \
    Pillow \
    nbconvert \
    scikit-image \
    kaggle

# ── GPU diagnostics ───────────────────────────────────────────────────────────
echo ""
echo "── GPU info ──────────────────────────────"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "
import torch
print(f'PyTorch : {torch.__version__}')
print(f'CUDA    : {torch.version.cuda}')
print(f'Devices : {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU 0   : {torch.cuda.get_device_name(0)}')
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM    : {vram:.1f} GB')
"
echo "──────────────────────────────────────────"
echo ""

# ── Dataset path validation ───────────────────────────────────────────────────
# Dataset Links:
# - https://www.kaggle.com/datasets/asosenge/hibaskinlesionsdataset-main
# - https://www.kaggle.com/datasets/asosenge/fitzpatrick17k
# - https://www.kaggle.com/datasets/asosenge/derm7pt
# - https://www.kaggle.com/datasets/mahdavi1202/skin-cancer'  
# - https://www.kaggle.com/datasets/sengenjih/isic2019'  

echo "[CHECK] Validating dataset roots..."
for DSET in \
    "asosenge/hibaskinlesionsdataset-main" \
    "asosenge/fitzpatrick17k" \
    "asosenge/derm7pt" \
    "mahdavi1202/skin-cancer" \
    "sengenjih/isic2019"; do
    FULL_PATH="$DATA_ROOT/$DSET"
    if [ -d "$FULL_PATH" ]; then
        echo "  [OK]      $FULL_PATH"
    else
        echo "  [MISSING] $FULL_PATH  ← Attempting to download via Kaggle CLI..."
        mkdir -p "$FULL_PATH"
        if kaggle datasets download -d "$DSET" -p "$FULL_PATH" --unzip; then
            echo "  [DOWNLOADED] $FULL_PATH"
        else
            echo "  [ERROR] Failed to download $DSET. Please check your Kaggle credentials (~/.kaggle/kaggle.json)."
            exit 1
        fi
    fi
done
echo ""

# ── Convert notebook → Python script ─────────────────────────────────────────
echo "[CONVERT] Converting notebook to Python script..."
#jupyter nbconvert "$NOTEBOOK" \
#    --to python \
#    --output "$SCRIPT_NAME"

echo "[PATCH] Redirecting Kaggle paths → cluster paths..."

PATCH_SCRIPT="patch_paths.py"
cat > "$PATCH_SCRIPT" << 'PYEOF'
import re, os, sys

script_path = sys.argv[1]
with open(script_path) as f:
    src = f.read()

data_root  = os.environ.get("DATA_ROOT",  os.path.expanduser("~/modality-invariance/data/datasets"))
work_root  = os.environ.get("WORK_ROOT",  os.path.expanduser("~/modality-invariance/outputs"))
bb_cache   = os.environ.get("BACKBONE_CACHE", os.path.expanduser("~/modality-invariance/backbone_cache"))

# Replace hardcoded Kaggle paths
src = src.replace("'/kaggle/input/datasets'",  f"'{data_root}'")
src = src.replace("'/kaggle/working'",          f"'{work_root}'")
src = src.replace("'/kaggle/working/backbone_cache'", f"'{bb_cache}'")

# Suppress notebook-only widgets (tqdm.notebook → tqdm)
src = src.replace("from tqdm.notebook import tqdm", "from tqdm import tqdm")

with open(script_path, "w") as f:
    f.write(src)

print(f"  DATA_ROOT  → {data_root}")
print(f"  WORK_ROOT  → {work_root}")
print(f"  BB_CACHE   → {bb_cache}")
print("  Patch applied successfully.")
PYEOF

python "$PATCH_SCRIPT" "../$SCRIPT_NAME"

# ── Run the training script ───────────────────────────────────────────────────
echo ""
echo "[RUN] Starting training — $(date)"
echo "========================================="

pip install -q timm einops scikit-learn umap-learn \
               matplotlib seaborn pandas tqdm Pillow \
               shap captum

python "../$SCRIPT_NAME"

EXIT_CODE=$?

echo "========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"

# ── Post-run summary ──────────────────────────────────────────────────────────
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "[DONE] Training completed successfully."
    echo "Outputs:"
    echo "  Checkpoints → $WORK_ROOT/checkpoints/"
    echo "  Results     → $WORK_ROOT/results/"
    echo "  CSVs        → $WORK_ROOT/csvs/"
else
    echo ""
    echo "[ERROR] Training failed with exit code $EXIT_CODE."
    echo "Check the error log: logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.err"
fi

deactivate
exit $EXIT_CODE
