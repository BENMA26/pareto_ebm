# PARETO-EBM source code

This repository contains training and sampling code for multi-attribute CelebA energy-based models, including:
- Classifier
- EBM
- JEM
- Pareto Classifier
- Pareto JEM
- Gibbs JEM

## Project Layout

- `src/train_*.py`: training entrypoints
- `src/model/model.py`: core model and Lightning modules
- `src/model/data.py`: dataset loading and filtering
- `src/model/sampler.py`: replay buffer
- `scripts/*.sh`: SLURM training examples
- `scripts/01.generation-for-fid-pareto-jem.py`: conditional sampling/generation script

## Environment

Recommended: Python 3.10+ with CUDA-enabled PyTorch.

Example setup:

```bash
conda create -n pareto-ebm python=3.10 -y
conda activate pareto-ebm
pip install torch torchvision pytorch-lightning wandb kornia scikit-learn pillow tqdm numpy torchjd matplotlib
```

## Data Path

Set dataset root (used by `src/model/data.py`):

```bash
export PARETO_EBM_DATASET_ROOT=/path/to/datasets
```

## Quick Train

Run from repository root:

```bash
cd src
python train_classifier.py \
  --gpus 1 \
  --proj_name demo \
  --exp_name demo-classifier \
  --strategy auto \
  --save_dir ../logs/demo-classifier \
  --dataset celeba \
  --img_size 64 \
  --batch_size 64 \
  --epoch_num 10
```

Other entrypoints:
- `train_ebm.py`
- `train_jem.py`
- `train_pareto_classifier.py`
- `train_pareto_jem.py`
- `train_gibbs_jem.py`
- `train_ebm_subset.py`

## SLURM Examples

```bash
sbatch scripts/train-celeba-classifier.sh
sbatch scripts/train-celeba-ebm-all-1.sh
sbatch scripts/train-celeba-jem-pareto.sh
```

The scripts support these variables:
- `PROJECT_ROOT`
- `SRC_DIR`
- `LOG_ROOT`

## Generation Script

Optional metadata paths:

```bash
export PARETO_EBM_ATTR_NAMES_JSON=/path/to/attr_names.json
export PARETO_EBM_THRESHOLD_JSON=/path/to/classifier_threshold.json
```

Example:

```bash
python scripts/01.generation-for-fid-pareto-jem.py \
  --attr_name_1 Male --attr_positive_1 \
  --attr_name_2 Smiling --attr_positive_2 \
  --ckpt_dir /path/to/checkpoints \
  --buffer_path /path/to/buffer.pkl \
  --save_path ./outputs \
  --model_name pareto_jem \
  --num_rounds 5 \
  --device cuda:0
```
