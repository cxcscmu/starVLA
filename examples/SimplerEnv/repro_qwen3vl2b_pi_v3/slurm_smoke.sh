#!/bin/bash
# Quick model-load smoke test on 1 GPU. Edit partition/qos/account for your cluster.
#SBATCH --job-name=pi_v3_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=smoke-%j.out
#SBATCH --error=smoke-%j.err

set -euo pipefail

CONDA_SH=${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
STARVLA_ENV=${STARVLA_ENV:-starVLA}
STARVLA_REPO=${STARVLA_REPO:?set STARVLA_REPO=/path/to/starVLA}

source "${CONDA_SH}"
conda activate "${STARVLA_ENV}"
cd "${STARVLA_REPO}"

export WANDB_MODE=offline
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}

nvidia-smi
python -c "import torch; print('torch', torch.__version__, 'cuda available', torch.cuda.is_available())"

python examples/SimplerEnv/repro_qwen3vl2b_pi_v3/smoke_test.py \
    --config_yaml examples/SimplerEnv/repro_qwen3vl2b_pi_v3/train_config.yaml
