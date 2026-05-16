#!/bin/bash
# WidowX 4-task evaluation client for SimplerEnv (Bridge).
# Adapted from examples/SimplerEnv/eval_files/start_simpler_env.sh.

set -euo pipefail

############# environment — set via env vars ###################
cd ${STARVLA_REPO:?set STARVLA_REPO=/path/to/starVLA}
export sim_python=${SIM_PYTHON:?set SIM_PYTHON=/path/to/simpler_env/bin/python}
export SimplerEnv_PATH=${SIMPLER_ENV_PATH:?set SIMPLER_ENV_PATH=/path/to/SimplerEnv}
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export LD_LIBRARY_PATH=$(dirname "${sim_python}")/../lib:${LD_LIBRARY_PATH:-}
port=${PORT:-6678}

your_ckpt=${1:?usage: $0 <ckpt_path> [port]}
port=${2:-${port}}
############# end ##############################################

ckpt_path=${your_ckpt}
ckpt_dir=$(dirname "${ckpt_path}")
ckpt_base=$(basename "${ckpt_path}")
ckpt_name="${ckpt_base%.*}"
output_eval_dir="${ckpt_dir}/output_eval"
mkdir -p "${output_eval_dir}"

TSET_NUM=1

scene_name=bridge_table_1_v1
robot=widowx
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
robot_init_x=0.147
robot_init_y=0.028

declare -a ENV_NAMES=(
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
)

for env in "${ENV_NAMES[@]}"; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "Launching [${env}] run#${run_idx} -> ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &
    sleep 6
  done
done

# V2 variant: PutEggplantInBasketScene-v0 uses the sink scene
scene_name=bridge_table_1_v2
robot=widowx_sink_camera_setup
rgb_overlay_path=${SimplerEnv_PATH}/ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
robot_init_x=0.127
robot_init_y=0.06

for env in PutEggplantInBasketScene-v0; do
  for ((run_idx=1; run_idx<=TSET_NUM; run_idx++)); do
    task_log="${output_eval_dir}/${ckpt_name}_${env}_run${run_idx}.log"
    echo "Launching V2 [${env}] run#${run_idx} -> ${task_log}"

    ${sim_python} examples/SimplerEnv/eval_files/start_simpler_env.py \
      --ckpt-path ${ckpt_path} \
      --port ${port} \
      --robot ${robot} \
      --policy-setup widowx_bridge \
      --control-freq 5 \
      --sim-freq 500 \
      --max-episode-steps 120 \
      --env-name "${env}" \
      --scene-name ${scene_name} \
      --rgb-overlay-path ${rgb_overlay_path} \
      --robot-init-x ${robot_init_x} ${robot_init_x} 1 \
      --robot-init-y ${robot_init_y} ${robot_init_y} 1 \
      --obj-variation-mode episode \
      --obj-episode-range 0 24 \
      --robot-init-rot-quat-center 0 0 0 1 \
      --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
      > "${task_log}" 2>&1 &
    sleep 6
  done
done

wait
echo "All 4 WidowX tasks done."
