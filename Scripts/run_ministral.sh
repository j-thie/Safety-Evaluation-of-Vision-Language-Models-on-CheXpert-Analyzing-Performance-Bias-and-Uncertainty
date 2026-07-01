#!/usr/bin/env bash
#SBATCH --job-name=mistral
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH -t 01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=email

# Environment setup
source ~/.bashrc
conda activate mistral-env


export VLLM_DISABLE_COMPILE=1
export VLLM_DISABLE_COMPILE_CACHE=1 #added from gpt to prevents vLLM writing/reading the torch-compile cache


# Debug info (optional but useful)
python - <<'PY'
import torch, subprocess
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA runtime:", torch.version.cuda)
subprocess.run(["nvidia-smi", "-L"])
PY

# Paths
REPO_DIR=/pfs/data6/home/ul/ul_student/ul_sau95/MIRP_Benchmark
OUTPUT_DIR=/home/ul/ul_student/ul_sau95/results/mistral

cd "$REPO_DIR/2_inference_code"

python onepicthreetimes.py \
    --input_folder "$REPO_DIR" \
    --output_folder "$OUTPUT_DIR" \
    --batch_size 1
