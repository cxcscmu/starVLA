# Reproducing Qwen3VL-2B + StarVLA-PI_v3 on SimplerEnv WidowX

This is a self-contained recipe for running (or just evaluating) the
**Qwen3-VL-2B-Instruct + QwenPI_v3 (layer-wise cross-DiT flow-matching)** policy
on the 4 SimplerEnv WidowX/Bridge tasks. Target: replicate StarVLA's reported
**62.5 average success** at 30k training steps.

> **For a coding agent on a different cluster:** if you only want to evaluate
> the released checkpoint, skip directly to [Evaluation](#3-evaluation). The
> checkpoint is on HuggingFace and the source cluster's training data is not
> required.

---

## Released artifacts

| Resource | Where |
|---|---|
| Trained checkpoint (5.4 GB) | https://huggingface.co/yiyangd/StarVLA-Qwen3VL2B-PI_v3-Bridge-RT_1 |
| Repo layout the policy server expects | `<run_dir>/{config.yaml, dataset_statistics.json, checkpoints/steps_30000_pytorch_model.pt}` |

The HF repo above already follows that layout — `hf download` keeps it intact.

## Recipe summary

- **Backbone:** Qwen3-VL-2B-Instruct (28 transformer layers, 2048 hidden dim)
- **Framework:** QwenPI_v3 (per-layer LayerNorm+Linear projector → cross-DiT flow-matching head, π₀.₅-style)
- **Action head:** 16-step chunk, action_dim=7, state_dim=7, `action_dit_hidden_dim=1024`
- **Data:** OXE `bridge_rt_1` mix = bridge_orig_lerobot + fractal20220817_data_lerobot
- **Training:** 30k steps, 8 GPUs single node, DeepSpeed ZeRO-2, bf16
- **Hyperparams:** LR base 1e-5 / qwen_vl 1e-5 / action_model 1e-4, cosine_with_min_lr, 1500 warmup
- **Total params:** ~2.6 B (qwen_vl 2.13 B + action_model 421 M + project_layers 59 M)

## Empirical notes from the source cluster run

- Single node × 8 × H100-80GB, batch_size 16/GPU
- Step rate ≈ 0.6 s/step → **30k steps in ~5h40m** wall clock
- `action_dit_loss` decreased from ~0.58 (step 100) to ~0.20-0.27 (step 25k)
- `wandb_entity: cxcscmu`, `wandb_project: starvla_yd`, `WANDB_MODE=online`

---

## 0. Repo layout

```
examples/SimplerEnv/repro_qwen3vl2b_pi_v3/
├── README.md              ← this file
├── train_config.yaml      ← OmegaConf YAML (overridden by CLI flags)
├── run_train.sh           ← accelerate-launch wrapper for the 8-GPU run
├── slurm_train.sh         ← SLURM submit script
├── smoke_test.py          ← load model + run forward / predict_action on fake data
├── slurm_smoke.sh         ← 1-GPU smoke test
├── download_datasets.py   ← idempotent OXE downloader (HF rate-limit aware)
├── setup_modality.sh      ← copy per-dataset modality.json into meta/
├── run_policy_server.sh   ← start the websocket policy server
├── start_simpler_env.sh   ← run the 4 WidowX SimplerEnv tasks against the server
└── slurm_simpler_smoke.sh ← quick check that sapien can build a Vulkan env
```

---

## 1. Environment setup

### 1a. `starVLA` env (training + policy server)

```bash
git clone https://github.com/starVLA/starVLA
cd starVLA
git checkout f                                      # this branch

conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install -e .

# flash-attn needs a wheel matching torch + CUDA. The pip build-from-source
# path hit a cross-device-link bug on our cluster; downloading the prebuilt
# wheel and installing from file is reliable:
WHL=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
wget "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${WHL}"
pip install "./${WHL}"
```

Versions that worked end-to-end:
`torch 2.6.0+cu124`, `transformers 4.57.0`, `accelerate 1.5.2`,
`deepspeed 0.16.9`, `flash-attn 2.7.4.post1`, NVIDIA driver `580.126.09`.

### 1b. `simpler_env` env (simulator)

```bash
conda create -n simpler_env python=3.10 -y
conda activate simpler_env
pip install "setuptools<70"      # sapien 2.2 imports pkg_resources
pip install numpy==1.24.4

git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv \
    /path/to/SimplerEnv
cd /path/to/SimplerEnv
pip install -e ManiSkill2_real2sim
pip install -e .

pip install tyro matplotlib mediapy websockets msgpack
pip install numpy==1.24.4        # reinforce after editable installs
```

### 1c. Vulkan setup (often the hard part)

`sapien.SapienRenderer` uses Vulkan. Confirm the cluster has the NVIDIA Vulkan
ICD installed:

```bash
ls /usr/share/vulkan/icd.d/nvidia_icd.json   # should exist
ls /dev/dri/                                  # renderD128 etc. should exist
vulkaninfo --summary                          # should list your H100/A100
```

If those are missing:
- **Best:** ask your cluster admin to install `libnvidia-gl-<driver_version>`.
- **DIY:** extract `libGLX_nvidia.so` from NVIDIA's `.run` installer for the
  **exact** kernel-driver version, point `VK_ICD_FILENAMES` at the bundled
  `nvidia_icd.json`, and add `LD_LIBRARY_PATH` to the lib dir. The user-space
  driver libs and kernel driver must agree on version, **and** the kernel
  module must expose graphics (not just compute) — if `/dev/dri/` is empty,
  no user-space patching can fix that.
- **CPU fallback:** Mesa lvp (`/usr/share/vulkan/icd.d/lvp_icd.x86_64.json`)
  lacks the extensions sapien wants on `sapien==2.2.2`, so this did **not**
  work in our testing. Skip.

Verify with the smoke test:

```bash
STARVLA_REPO=/path/to/starVLA sbatch examples/SimplerEnv/repro_qwen3vl2b_pi_v3/slurm_simpler_smoke.sh
```

Success looks like: `✅ Env built successfully`. If you see
`vk::PhysicalDevice::createDeviceUnique: ErrorExtensionNotPresent`, the
cluster's Vulkan / NVIDIA ICD chain is the issue.

---

## 2. Get the checkpoint

```bash
hf download yiyangd/StarVLA-Qwen3VL2B-PI_v3-Bridge-RT_1 \
    --local-dir /path/to/checkpoints/qwen3vl2b_pi_v3_bridge_rt_1_30k
```

After download you should have, exactly:

```
qwen3vl2b_pi_v3_bridge_rt_1_30k/
├── config.yaml
├── dataset_statistics.json
└── checkpoints/
    └── steps_30000_pytorch_model.pt
```

The policy server's loader walks `parents[1]` from the `.pt` file looking for
`config.yaml` and `dataset_statistics.json`, so keep this layout.

---

## 3. Evaluation

You'll run two processes on the same node, one per conda env. SimplerEnv's
eval client talks to the policy server over a localhost websocket.

### 3a. Start the policy server (terminal 1, `starVLA` env)

```bash
conda activate starVLA
export STARVLA_REPO=/path/to/starVLA
export CKPT=/path/to/checkpoints/qwen3vl2b_pi_v3_bridge_rt_1_30k/checkpoints/steps_30000_pytorch_model.pt
bash ${STARVLA_REPO}/examples/SimplerEnv/repro_qwen3vl2b_pi_v3/run_policy_server.sh
```

Wait for `server listening on 0.0.0.0:6678`.

### 3b. Run the 4 WidowX tasks (terminal 2, `simpler_env` env)

```bash
conda activate simpler_env
export STARVLA_REPO=/path/to/starVLA
export SIM_PYTHON=$(which python)
export SIMPLER_ENV_PATH=/path/to/SimplerEnv
bash ${STARVLA_REPO}/examples/SimplerEnv/repro_qwen3vl2b_pi_v3/start_simpler_env.sh \
    /path/to/checkpoints/qwen3vl2b_pi_v3_bridge_rt_1_30k/checkpoints/steps_30000_pytorch_model.pt
```

The 4 tasks (25 episodes each, episode 0–24):

1. `StackGreenCubeOnYellowCubeBakedTexInScene-v0`
2. `PutCarrotOnPlateInScene-v0`
3. `PutSpoonOnTableClothInScene-v0`
4. `PutEggplantInBasketScene-v0` (V2 sink scene)

Per-task `Average success` is printed at the end of each task log under
`<ckpt_dir>/output_eval/`. **Target average: 62.5.**

---

## 4. (Optional) Re-train from scratch

### 4a. Download data

```bash
export DATA_ROOT=/path/to/OXE_LEROBOT
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxx        # https://huggingface.co/settings/tokens
python examples/SimplerEnv/repro_qwen3vl2b_pi_v3/download_datasets.py
```

Downloads:
- `IPEC-COMMUNITY/bridge_orig_lerobot` → `${DATA_ROOT}/bridge_orig_1.0.0_lerobot/`
- `IPEC-COMMUNITY/fractal20220817_data_lerobot` → `${DATA_ROOT}/fractal20220817_data_0.1.0_lerobot/`

Filtered to only the camera views the data config uses (`image_0` for bridge,
`image` for fractal) — cuts file count ~4×. The script lists the remote file
set, downloads what's missing with 3 worker threads, exponential 429 backoff,
and verifies completeness; safe to ctrl-c and re-run. Total ~44 GB.

### 4b. Install per-dataset modality.json

```bash
export DATA_ROOT=/path/to/OXE_LEROBOT
bash examples/SimplerEnv/repro_qwen3vl2b_pi_v3/setup_modality.sh
```

### 4c. Smoke test the model build

```bash
export STARVLA_REPO=/path/to/starVLA
sbatch ${STARVLA_REPO}/examples/SimplerEnv/repro_qwen3vl2b_pi_v3/slurm_smoke.sh
```

Expect `Smoke test PASSED.` and a 2.6 B parameter breakdown.

### 4d. Train

```bash
export STARVLA_REPO=/path/to/starVLA
export BASE_VLM=/path/to/Qwen3-VL-2B-Instruct       # or HF hub id
export DATA_ROOT=/path/to/OXE_LEROBOT
export RUN_ROOT_DIR=/path/to/results/Checkpoints
sbatch ${STARVLA_REPO}/examples/SimplerEnv/repro_qwen3vl2b_pi_v3/slurm_train.sh
```

NCCL: the launcher does **not** set `NCCL_SOCKET_IFNAME` — it lets NCCL
auto-detect, which works for single-node runs. If you go multi-node, set the
right interface for your fabric.

W&B project / entity are set to `cxcscmu / starvla_yd` and online mode by
default; override with `WANDB_MODE=offline` or by editing
`train_config.yaml`.

---

## 5. Known issues / lessons from the first run

- **HF download rate limit.** The default 8-worker `huggingface-cli download`
  hammers the xet-token endpoint past the 5000 req / 5min auth cap and gets
  429'd. `download_datasets.py` here is intentionally 3 workers with explicit
  exp-backoff. Don't increase.
- **NCCL `bond0`/`mlx5_*`.** The original `run_oxe_train.sh` from upstream
  hardcoded HKUST-cluster interface names. We removed those — single-node
  runs work fine with auto-detect. Multi-node will need the right values
  for your fabric.
- **flash-attn build.** `pip install flash-attn --no-build-isolation` failed
  with `Invalid cross-device link` because pip's tmpdir landed on a
  different filesystem from the cache. Downloading the prebuilt wheel and
  installing it locally was reliable.
- **Vulkan / `libGLX_nvidia`.** Some compute-only clusters install the
  NVIDIA driver without the graphics user-space libs. See §1c.
