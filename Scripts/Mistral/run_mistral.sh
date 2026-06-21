#!/usr/bin/env bash
#SBATCH --job-name=mistral
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL


# Environment setup
source ~/.bashrc
conda activate mistral-env


export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export VLLM_DISABLE_COMPILE=1
export VLLM_DISABLE_COMPILE_CACHE=1 #added from gpt to prevents vLLM writing/reading the torch-compile cache


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
REPO_DIR=/load/script/from/local/path 
OUTPUT_DIR=/pathway/to/save/output

cd "$REPO_DIR"

python /add/wanted/script.py\
    --input_folder "$REPO_DIR" \
    --output_folder "$OUTPUT_DIR" \
    --batch_size 1
