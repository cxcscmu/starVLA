#!/bin/bash
# Quick SimplerEnv smoke test — confirms Vulkan + sapien can build a WidowX env.
# Edit partition/qos/account for your cluster.
#SBATCH --job-name=simpler_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=simpler_smoke-%j.out
#SBATCH --error=simpler_smoke-%j.err

set +e

CONDA_SH=${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}
SIMPLER_ENV_NAME=${SIMPLER_ENV_NAME:-simpler_env}
STARVLA_REPO=${STARVLA_REPO:?set STARVLA_REPO=/path/to/starVLA}

source "${CONDA_SH}"
conda activate "${SIMPLER_ENV_NAME}"
cd "${STARVLA_REPO}"

# Conda libs first so we pick the env's libvulkan if it has one.
export LD_LIBRARY_PATH=$(dirname $(which python))/../lib:${LD_LIBRARY_PATH:-}
export DISPLAY=""

# If your cluster ships NVIDIA Vulkan ICD, no overrides are needed.
# If it doesn't, see the README's "Vulkan setup" section.

nvidia-smi
python -c "import sapien; import sapien.core; print('sapien import OK')"
python examples/SimplerEnv/eval_files/test_your_simplerEnv.py
