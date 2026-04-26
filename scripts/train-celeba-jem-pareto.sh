#!/bin/bash
#SBATCH --job-name=toy_exp          # Job name
#SBATCH --partition=normal          # Partition name (adjust for your cluster)
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --cpus-per-task=24          # Number of CPU cores per task
#SBATCH --gres=gpu:4                # Number of GPUs
#SBATCH --mem=128G                  # Memory request
#SBATCH --time=4800:00:00           # Max runtime (DD-HH:MM:SS)
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SRC_DIR="${SRC_DIR:-${PROJECT_ROOT}/src}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/train_celeba_models}"
RUN_TAG="train-celeba-jem-pareto"
RUN_DIR="${LOG_ROOT}/${RUN_TAG}"

# Activate conda environment
source ~/.bashrc
conda init
conda activate diffusion

# Show job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Show GPU information
nvidia-smi

# Show environment information
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "CUDA available:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "CUDA device count:"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# Switch to working directory
cd "${SRC_DIR}"

# Run training script
echo "Starting training..."

# Create log directory
mkdir -p "${RUN_DIR}"

# Execute training
nohup python -u train_pareto_jem.py --gpus 4 \
                                       --proj_name experiment \
                                       --exp_name "${RUN_TAG}" \
                                       --strategy ddp \
                                       --save_dir "${RUN_DIR}" \
                                       --dataset celeba \
                                       --img_sigma 0.01 \
                                       --celeba_drop_infreq 0.13 \
                                       --dataset_transform \
                                       --mix_up \
                                       --epoch_num 200 \
                                       --batch_size 64 \
                                       --num_workers 4 \
                                       --lr 1e-4 \
                                       --beta1 0.0 \
                                       --beta2 0.999 \
                                       --clip_gradients \
                                       --gradient_clip_val 0.5 \
                                       --buffer_size 50000 \
                                       --replace_prob 0.05 \
                                       --buffer_transform \
                                       --sgld_steps 60 \
                                       --sgld_lr 1 \
                                       --img_size 64 \
                                       --deep \
                                       --multiscale \
                                       --self-attn \
                                       --sgld_sigma 0.001 \
                                       --kl_loss \
                                       --use_ema \
                                       --ema_eval \
                                       --ema_decay 0.9999
