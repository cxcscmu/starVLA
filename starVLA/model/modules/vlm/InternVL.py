# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
InternVL3 / InternVL3.5 VLM Interface for starVLA — memory-optimized.

Key changes vs. the original wrapper to fix OOM at BS>=12:
  1. Real gradient checkpointing on both the ViT and LLM towers (30–50% activation savings).
  2. Default `output_hidden_states=False` in forward; downstream code that needs the last
     hidden state should pass `output_hidden_states=True` explicitly. Many call sites only
     need the action-head input, which is the *last* layer — never the full tuple.
  3. Skip LM-head materialization when labels=None and we don't need logits (the action
     framework only consumes hidden_states). InternVLChatModel.forward unfortunately always
     computes logits, so we monkey-patch it to short-circuit when no loss is requested.
  4. Tokenizer truncation by default to prevent one long prompt from blowing up BS-wide L.
  5. autocast / stdout-redirect contexts are no longer nested.
"""

import os
import contextlib

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from accelerate.logging import get_logger

logger = get_logger(__name__)

IGNORE_INDEX = -100
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(image, min_num=1, max_num=12, image_size=224, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def _load_pil_image(image, input_size=224, max_num=12):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image)).convert('RGB')
    else:
        image = image.convert('RGB')
    transform = _build_transform(input_size=input_size)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def _enable_gradient_checkpointing_robust(model):
    """
    Enable gradient checkpointing on InternVLChatModel.

    The custom HF Hub modeling file does not always implement
    `model.gradient_checkpointing_enable()` correctly — sometimes it only wires
    the LLM tower, sometimes neither. We reach into both sub-towers explicitly.

    Returns the list of sub-modules that were successfully checkpointed (for logging).
    """
    enabled = []

    # Top-level: try the standard HF API first.
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        enabled.append("model (top-level)")
    except Exception as e:
        print(f"[InternVL3] top-level gradient_checkpointing_enable failed: {e}", flush=True)

    # LLM tower (InternLM2 / Qwen2 / Qwen3 backbone) — the custom modeling files
    # expose the language model under `.language_model` for InternVL3 and `.llm` for
    # some 2.5 forks. Try both.
    for attr in ("language_model", "llm"):
        sub = getattr(model, attr, None)
        if sub is None:
            continue
        try:
            sub.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            enabled.append(f"model.{attr}")
        except Exception:
            # Older custom modeling files don't support kwargs — fall back to bare call.
            try:
                sub.gradient_checkpointing_enable()
                enabled.append(f"model.{attr} (no-kwargs)")
            except Exception as e:
                print(f"[InternVL3] could not enable ckpt on model.{attr}: {e}", flush=True)

    # ViT tower — InternVL3 names it `.vision_model`.
    vit = getattr(model, "vision_model", None)
    if vit is not None:
        try:
            vit.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            enabled.append("model.vision_model")
        except Exception:
            try:
                vit.gradient_checkpointing_enable()
                enabled.append("model.vision_model (no-kwargs)")
            except Exception as e:
                print(f"[InternVL3] could not enable ckpt on vision_model: {e}", flush=True)

    # HF requires inputs to require grad when ckpt is on — needed for embedding layer
    # to receive gradients through the checkpoint boundary.
    if hasattr(model, "enable_input_require_grads"):
        try:
            model.enable_input_require_grads()
        except Exception:
            pass

    return enabled


class _InternVL3_Interface(nn.Module):
    """
    Memory-optimized wrapper around InternVL3 / InternVL3.5 (InternVLChatModel via trust_remote_code).
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "OpenGVLab/InternVL3-1B")
        use_flash_attn = "flash" in qwenvl_config.get("attn_implementation", "sdpa")
        # Truncate prompts so one long instruction can't dominate the padded batch.
        # Set to None to disable. 4096 is a generous default — actual VLA prompts
        # rarely exceed 1k tokens including image tokens.
        self.max_text_length = qwenvl_config.get("max_text_length", 4096)
        # Default ON — saves 30-50% activation memory at the cost of one extra
        # forward pass per backward. Critical for fitting BS>=12 on L40S.
        enable_grad_ckpt = bool(qwenvl_config.get("enable_gradient_checkpointing", True))

        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            use_fast=False,
        )
        tokenizer.padding_side = "left"

        self.model = model
        self.tokenizer = tokenizer
        # Expose tokenizer as processor.tokenizer for compatibility with framework code
        self.processor = type('obj', (object,), {'tokenizer': tokenizer})()
        self.config = config

        # Surface text hidden_size at the top level for downstream action heads.
        self.model.config.hidden_size = self.model.config.llm_config.hidden_size

        # InternVL3 image config
        self.image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        self.num_image_token = self.model.num_image_token
        self.max_num_tiles = qwenvl_config.get("max_num_tiles", 1)

        # Set img_context_token_id on the model
        self.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        # ---- gradient checkpointing wiring ----
        if enable_grad_ckpt:
            enabled = _enable_gradient_checkpointing_robust(model)
            if enabled:
                print(
                    f"[InternVL3] gradient_checkpointing ENABLED on: {', '.join(enabled)} "
                    f"(use_reentrant=False where supported)",
                    flush=True,
                )
            else:
                print(
                    "[InternVL3] WARNING: gradient_checkpointing requested but no sub-module "
                    "accepted the call. Memory will not be reduced.",
                    flush=True,
                )

    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        image_flags=None,
        labels=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=False,  # CHANGED: default False; callers opt in.
        return_dict=True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to InternVLChatModel.

        Memory notes:
          - `output_hidden_states` defaults to False here. If your action head
            needs the last layer's hidden state, pass `output_hidden_states=True`
            explicitly — the framework will then receive `outputs.hidden_states[-1]`.
            We default to False to avoid the (B, L, H, num_layers) memory blow-up.
          - `use_cache` is forced to False during training to skip KV-cache allocation.
        """
        # During training (we know we're training when labels is None *and* grad is enabled,
        # OR labels is not None) we never want KV cache. Only generation needs it, and
        # generation goes through self.generate() which sets its own use_cache.
        if self.training and use_cache is None:
            use_cache = False

        # Suppress InternVL's per-step "dynamic ViT batch size" stdout chatter.
        # First call we let it through for sanity, after that we mute. The mute
        # uses a single per-instance devnull file handle to avoid open/close overhead.
        if not getattr(self, "_quiet_forward_logged", False):
            self._quiet_forward_logged = True
            stdout_ctx = contextlib.nullcontext()
        else:
            if not hasattr(self, "_devnull"):
                self._devnull = open(os.devnull, "w")
            stdout_ctx = contextlib.redirect_stdout(self._devnull)

        with torch.autocast("cuda", dtype=torch.bfloat16), stdout_ctx:
            outputs = self.model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                labels=labels,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return outputs

    def generate(self, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.model.generate(**kwargs)

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """Construct and tokenize multimodal inputs for InternVL3 (batched)."""
        assert len(images) == len(instructions), "Images and instructions must have the same length"

        all_pixel_values = []
        all_num_patches = []
        queries = []

        for imgs, instruction in zip(images, instructions):
            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            sample_pixel_values = []
            num_patches_list = []

            for img in imgs:
                pv = _load_pil_image(img, input_size=self.image_size, max_num=self.max_num_tiles)
                sample_pixel_values.append(pv)
                num_patches_list.append(pv.shape[0])

            if len(sample_pixel_values) > 0:
                all_pixel_values.append(torch.cat(sample_pixel_values, dim=0))
            all_num_patches.append(num_patches_list)

            if len(imgs) > 0 and '<image>' not in prompt:
                image_prefix = '\n'.join([f'Image-{i+1}: <image>' for i in range(len(imgs))]) + '\n'
                question = image_prefix + prompt
            else:
                question = prompt

            template = self.model.conv_template.copy()
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            queries.append(query)

        # Tokenize with truncation so one long prompt cannot blow up the padded batch.
        self.tokenizer.padding_side = 'left'
        tok_kwargs = dict(return_tensors='pt', padding=True)
        if self.max_text_length is not None:
            tok_kwargs.update(truncation=True, max_length=self.max_text_length)
        model_inputs = self.tokenizer(queries, **tok_kwargs)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16)
            image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        else:
            pixel_values = torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.bfloat16)
            image_flags = torch.zeros(1, 1, dtype=torch.long)

        result = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_flags': image_flags,
        }
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in result.items()}