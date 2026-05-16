#!/bin/bash
# Example SLURM submit — edit partition/qos/account for your cluster.
#SBATCH --job-name=pi_v3_qwen3vl2b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# === edit these for your cluster ============================================
CONDA_SH=${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
STARVLA_ENV=${STARVLA_ENV:-starVLA}
STARVLA_REPO=${STARVLA_REPO:?set STARVLA_REPO=/path/to/starVLA}
export BASE_VLM=${BASE_VLM:?set BASE_VLM=/path/to/Qwen3-VL-2B-Instruct}
export DATA_ROOT=${DATA_ROOT:?set DATA_ROOT=/path/to/OXE_LEROBOT}
export RUN_ROOT_DIR=${RUN_ROOT_DIR:?set RUN_ROOT_DIR=/path/to/checkpoints/out}
# ============================================================================

source "${CONDA_SH}"
conda activate "${STARVLA_ENV}"
cd "${STARVLA_REPO}"

export WANDB_MODE=${WANDB_MODE:-online}
export HF_HOME=${HF_HOME:-${HOME}/.cache/huggingface}

nvidia-smi
echo "Launching training at $(date)"
bash examples/SimplerEnv/repro_qwen3vl2b_pi_v3/run_train.sh
echo "Training finished at $(date)"
