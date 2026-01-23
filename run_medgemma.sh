#!/usr/bin/env bash
#SBATCH --job-name=medgemma
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 00:45:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

# Environment setup
source ~/.bashrc
conda activate medgemma

# Debug info
python - <<'PY'
import torch, subprocess
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
subprocess.run(["nvidia-smi", "-L"])
PY

# Paths
REPO_DIR=/pathway/to/main/script
OUTPUT_DIR=/pathway/to/where/results/should/be/saved

cd "$REPO_DIR"

export VLLM_DISABLE_COMPILE_CACHE=1

python medgemma_main.py \
    --input_folder "$REPO_DIR" \
    --output_folder "$OUTPUT_DIR" \
    --batch_size 1
