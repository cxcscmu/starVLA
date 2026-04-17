# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");

"""
InternVL3 / InternVL3.5 VLM Interface for starVLA.

Wraps OpenGVLab/InternVL3-* and InternVL3_5-* models into the same interface as Qwen VL wrappers.
Both InternVL3 and InternVL3.5 share the same InternVLChatModel architecture (ViT-MLP-LLM).
Uses trust_remote_code=True to load the custom InternVLChatModel.

Supported models:
  - OpenGVLab/InternVL3-1B, InternVL3-2B, InternVL3-8B, ...
  - OpenGVLab/InternVL3_5-1B, InternVL3_5-2B, InternVL3_5-8B, ...

Key differences from Qwen VL:
  - Uses AutoModel + AutoTokenizer (no AutoProcessor)
  - Image preprocessing: dynamic tiling (224×224) + ImageNet normalization
  - Uses conversation template from the model's own `get_conv_template`
  - image_flags tensor to distinguish real image tiles from padding
"""

import os
import contextlib

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List
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
    """Build the standard InternVL image transform."""
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
    """Split an image into dynamic tiles following InternVL's strategy."""
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
    """Preprocess a single PIL image into pixel_values tensor."""
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.array(image)).convert('RGB')
    else:
        image = image.convert('RGB')
    transform = _build_transform(input_size=input_size)
    # images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    images = _dynamic_preprocess(image, image_size=input_size, use_thumbnail=False, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class _InternVL3_Interface(nn.Module):
    """
    Lightweight wrapper around InternVL3 / InternVL3.5 (InternVLChatModel via trust_remote_code).
    Both model families share the same architecture and API.

    Provides the same interface as _QWen_VL_Interface:
      - forward() with output_hidden_states support
      - generate() for inference
      - build_qwenvl_inputs() for batch preprocessing
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        qwenvl_config = config.framework.get("qwenvl", {})
        model_id = qwenvl_config.get("base_vlm", "OpenGVLab/InternVL3-1B")
        use_flash_attn = "flash" in qwenvl_config.get("attn_implementation", "sdpa")

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

        # Set hidden_size from the LLM config for downstream action heads
        self.model.config.hidden_size = self.model.config.llm_config.hidden_size

        # InternVL3 image config
        self.image_size = self.model.config.force_image_size or self.model.config.vision_config.image_size
        self.num_image_token = self.model.num_image_token
        self.max_num_tiles = qwenvl_config.get("max_num_tiles", 1)

        # Set img_context_token_id on the model
        self.model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

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
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass delegating to InternVLChatModel."""

        # Suppress InternVL's per-step "dynamic ViT batch size" print to keep tqdm clean
        with open(os.devnull, 'w') as devnull, \
             contextlib.redirect_stdout(devnull), \
             torch.autocast("cuda", dtype=torch.bfloat16):
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
        """Generation interface."""
        with torch.autocast("cuda", dtype=torch.bfloat16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal inputs for InternVL3 (batched).

        Args:
            images: List[List[PIL.Image]] - B samples, each with a list of images.
            instructions: List[str] - B instruction strings.
            solutions: Optional, not used for InternVL3 OFT-style training.

        Returns:
            dict with keys: pixel_values, input_ids, attention_mask, image_flags
            moved to model device.
        """
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

            # Build question with <image> placeholders for each image
            if len(imgs) > 0 and '<image>' not in prompt:
                image_prefix = '\n'.join([f'Image-{i+1}: <image>' for i in range(len(imgs))]) + '\n'
                question = image_prefix + prompt
            else:
                question = prompt

            # Use the model's conversation template
            template = self.model.conv_template.copy()
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            # Replace <image> placeholders with actual image tokens
            for num_patches in num_patches_list:
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace('<image>', image_tokens, 1)

            queries.append(query)

        # Tokenize all queries with left padding
        self.tokenizer.padding_side = 'left'
        model_inputs = self.tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        # Stack all pixel values
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0).to(torch.bfloat16)
            image_flags = torch.ones(pixel_values.shape[0], 1, dtype=torch.long)
        else:
            # No images - create dummy pixel values
            pixel_values = torch.zeros(1, 3, self.image_size, self.image_size, dtype=torch.bfloat16)
            image_flags = torch.zeros(1, 1, dtype=torch.long)

        result = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'image_flags': image_flags,
        }

        # Move to model device
        device = next(self.model.parameters()).device
        result = {k: v.to(device) for k, v in result.items()}

        return result