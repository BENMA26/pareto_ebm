#!/bin/bash
#SBATCH --job-name=toy_exp          # 作业名称
#SBATCH --partition=normal                     # 分区名称 (请根据您的集群修改)
#SBATCH --nodes=1                             # 节点数量
#SBATCH --cpus-per-task=24                     # 每个任务的CPU核心数
#SBATCH --gres=gpu:4                          # GPU数量 (根据您的需求修改)
#SBATCH --mem=128G                             # 内存需求 (根据您的需求修改)
#SBATCH --time=480:00:00                       # 最大运行时间 (格式: DD-HH:MM:SS)
#SBATCH --output=/work/home/maben/project/blue_whale_lab/projects/PARETO_EBM/experiments/final_experiments/experiments/train_celeba_models/logs/2025-12-06-train-celeba-ebm-all.out
#SBATCH --error=/work/home/maben/project/blue_whale_lab/projects/PARETO_EBM/experiments/final_experiments/experiments/train_celeba_models/logs/2025-12-06-train-celeba-ebm-all.err

# 激活conda环境
source ~/.bashrc
conda init
conda activate diffusion

# 显示作业信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Working Directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# 显示GPU信息
nvidia-smi

# 显示环境信息
echo "Python version:"
python --version
echo "PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
echo "CUDA available:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "CUDA device count:"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# 切换到工作目录
cd /work/home/maben/project/blue_whale_lab/projects/PARETO_EBM/experiments/final_experiments/src

# 运行训练脚本
echo "Starting training..."

# 创建日志保存目录
mkdir -p /work/home/maben/project/blue_whale_lab/projects/PARETO_EBM/experiments/final_experiments/experiments/train_celeba_models/logs/$(date +"%Y%m%d")-2025-12-06-train-celeba-ebm-all

# 执行训练
nohup python -u train_ebm.py --gpus 4 \
                                       --proj_name experiment \
                                       --exp_name $(date +"%Y%m%d")-2025-12-06-train-celeba-ebm-all \
                                       --strategy ddp_find_unused_parameters_true \
                                       --save_dir /work/home/maben/project/blue_whale_lab/projects/PARETO_EBM/experiments/final_experiments/experiments/train_celeba_models/logs/$(date +"%Y%m%d")-2025-12-06-train-celeba-ebm-all \
                                       --dataset celeba \
                                       --img_sigma 0.01 \
                                       --celeba_drop_infreq 0.13 \
                                       --dataset_transform \
                                       --mix_up \
                                       --epoch_num 100 \
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
                                       --sgld_lr 100 \
                                       --img_size 64 \
                                       --deep \
                                       --multiscale \
                                       --self-attn \
                                       --sgld_sigma 0.001 \
                                       --kl_loss \
                                       --use_ema \
                                       --ema_eval \
                                       --ema_decay 0.9999