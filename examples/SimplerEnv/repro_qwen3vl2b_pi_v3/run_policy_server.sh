#!/bin/bash
# Policy server for the Qwen3-VL-2B-Instruct + QwenPI_v3 checkpoint.
# Adapted from examples/SimplerEnv/eval_files/run_policy_server.sh.

set -euo pipefail

############# environment — set via env vars ###################
cd ${STARVLA_REPO:?set STARVLA_REPO=/path/to/starVLA}
export star_vla_python=${STAR_VLA_PYTHON:-$(which python)}
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
port=${PORT:-6678}
gpu_id=${GPU_ID:-0}

# The checkpoint dir layout the policy server expects:
#   <run_dir>/config.yaml
#   <run_dir>/dataset_statistics.json
#   <run_dir>/checkpoints/steps_30000_pytorch_model.pt
your_ckpt=${CKPT:?set CKPT to .../<run_id>/checkpoints/steps_30000_pytorch_model.pt}
############# end ##############################################

ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"

CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"
