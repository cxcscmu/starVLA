#!/bin/bash
# Copy the StarVLA per-dataset modality.json files into each LeRobot dataset's meta/ dir.
set -euo pipefail

REPO=${REPO:-$(cd "$(dirname "$0")/../../.." && pwd)}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT to the OXE_LEROBOT root}

mkdir -p ${DATA_ROOT}/bridge_orig_1.0.0_lerobot/meta
cp ${REPO}/examples/SimplerEnv/train_files/modality.json \
   ${DATA_ROOT}/bridge_orig_1.0.0_lerobot/meta/modality.json

mkdir -p ${DATA_ROOT}/fractal20220817_data_0.1.0_lerobot/meta
cp ${REPO}/examples/SimplerEnv/train_files/fractal_modality.json \
   ${DATA_ROOT}/fractal20220817_data_0.1.0_lerobot/meta/modality.json

echo "modality.json files installed."
ls -la ${DATA_ROOT}/*/meta/modality.json
