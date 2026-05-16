#!/bin/bash
# Launch StarVLA-PI_v3 training with Qwen3-VL-2B-Instruct backbone on Bridge+RT-1.
# Target: replicate SimplerEnv WidowX avg success ~62.5 at 30k steps.

# Single-node 8-GPU run: let NCCL auto-detect the interface (no bond0/mlx5 on flame).
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=1000

############# environment — override via env vars #################
Framework_name=QwenPI_v3
freeze_module_list=''
base_vlm=${BASE_VLM:?set BASE_VLM=/path/to/Qwen3-VL-2B-Instruct}
config_yaml=./examples/SimplerEnv/repro_qwen3vl2b_pi_v3/train_config.yaml
oxe_data_root=${DATA_ROOT:?set DATA_ROOT=/path/to/OXE_LEROBOT}
data_mix=bridge_rt_1
run_root_dir=${RUN_ROOT_DIR:?set RUN_ROOT_DIR=/path/to/checkpoints/dir}
run_id=qwen3vl2b_pi_v3_bridge_rt_1_30k
############# end environment #####################################

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" "${output_dir}/"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --datasets.vla_data.data_root_dir ${oxe_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size 16 \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starvla_yd \
  --wandb_entity cxcscmu
