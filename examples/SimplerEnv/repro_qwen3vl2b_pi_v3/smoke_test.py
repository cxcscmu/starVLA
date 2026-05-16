"""Standalone smoke test: load Qwen3-VL-2B-Instruct + QwenPI_v3 + run forward / predict on fake data."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from PIL import Image

from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3


def print_model_size(m: nn.Module):
    total = sum(p.numel() for p in m.parameters())
    print(f"\n{'='*55}")
    print(f"{'Module':<35} {'Params':>12}  {'%':>6}")
    print(f"{'-'*55}")
    for name, child in m.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {name:<33} {n:>12,}  {100*n/total:>5.1f}%")
    print(f"{'-'*55}")
    print(f"  {'TOTAL':<33} {total:>12,}  100.0%")
    print(f"{'='*55}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="examples/SimplerEnv/repro_qwen3vl2b_pi_v3/train_config.yaml",
    )
    args = parser.parse_args()

    print(f"Loading config from {args.config_yaml}", flush=True)
    cfg = OmegaConf.load(args.config_yaml)
    print("Building Qwen_PI_v3 with", cfg.framework.qwenvl.base_vlm, flush=True)

    model = Qwen_PI_v3(cfg)
    print_model_size(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Moving model to {device}", flush=True)
    model = model.to(device)

    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16),
        "image": [image],
        "lang": "place the carrot on the plate.",
        "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16),
    }
    batch = [sample, sample]

    print("\n=== forward pass ===", flush=True)
    out = model(batch)
    print(f"action_loss = {out['action_loss'].item():.6f}", flush=True)

    print("\n=== predict_action ===", flush=True)
    pred = model.predict_action([sample])
    print(f"normalized_actions shape = {pred['normalized_actions'].shape}", flush=True)
    print(f"normalized_actions[0,0] = {pred['normalized_actions'][0,0]}", flush=True)

    print("\nSmoke test PASSED.", flush=True)


if __name__ == "__main__":
    main()
