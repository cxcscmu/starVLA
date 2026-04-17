#!/bin/bash

cd /data/user_data/yutengz/projects/starVLA
source ~/.bashrc
conda init
conda activate starvla2

###########################################################################################
export LIBERO_HOME=/data/user_data/yutengz/projects/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/libero
export LIBERO_Python=~/miniconda3/envs/libero/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME}
export PYTHONPATH=$(pwd):${PYTHONPATH}

host="172.16.1.89"
base_port=5694
your_ckpt=/data/user_data/yutengz/projects/starVLA/results/Checkpoints/0401_libero4in1_qwen35_0.8b_oft/checkpoints/steps_30000_pytorch_model.pt
# your_ckpt=/data/user_data/yutengz/projects/starVLA/results/Checkpoints/0401_libero4in1_qwen3oft/checkpoints/steps_10000_pytorch_model.pt
# your_ckpt=/data/user_data/yutengz/projects/starVLA/results/Checkpoints/0401_libero4in1_qwen3oft/checkpoints/steps_10000_pytorch_model.pt

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
###########################################################################################

num_trials_per_task=50
RESULT_LOG="results/eval_summary_$(date +"%Y%m%d_%H%M%S").log"
mkdir -p results

echo "=== Evaluation Summary ===" > ${RESULT_LOG}
echo "Checkpoint: ${your_ckpt}" >> ${RESULT_LOG}
echo "Date: $(date)" >> ${RESULT_LOG}
echo "" >> ${RESULT_LOG}

for task_suite_name in libero_spatial libero_object libero_goal libero_10; do
    echo "=========================================="
    echo "Running: ${task_suite_name}"
    echo "=========================================="

    video_out_path="results/${task_suite_name}/${folder_name}"
    tmp_log=$(mktemp)

    # Run eval, tee to both terminal and temp file
    ${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
        --args.pretrained-path ${your_ckpt} \
        --args.host "$host" \
        --args.port $base_port \
        --args.task-suite-name "$task_suite_name" \
        --args.num-trials-per-task "$num_trials_per_task" \
        --args.video-out-path "$video_out_path" 2>&1 | tee "$tmp_log" || true

    # Extract and log Total success rate
    rate=$(grep "Total success rate" "$tmp_log" | tail -1)
    episodes=$(grep "Total episodes" "$tmp_log" | tail -1)

    echo "[${task_suite_name}] ${rate}" | tee -a ${RESULT_LOG}
    echo "[${task_suite_name}] ${episodes}" | tee -a ${RESULT_LOG}
    echo "" >> ${RESULT_LOG}

    rm -f "$tmp_log"
done

echo "==========================================" | tee -a ${RESULT_LOG}
echo "All tasks done. Results saved to ${RESULT_LOG}"

# bash examples/LIBERO/eval_files/eval_libero_all.sh