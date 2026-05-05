# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by GitHub Copilot in 2025

from typing import Optional

import torch
from starVLA.training.trainer_utils import initialize_overwatch
from transformers import AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100
# InternVL3.5 token indices (based on Qwen3 tokenizer via llm_config)
IMAGE_TOKEN_INDEX = 151655  # Image token in Qwen3 tokenizer
VIDEO_TOKEN_INDEX = 151656  # Video token in Qwen3 tokenizer
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

# Action token range for InternVL3.5 (aligned with Qwen3 action token range)
# These should be adjusted after running special token addition scripts
_ACTION_TOKEN_MIN = 151669  # Starting range for action tokens
_ACTION_TOKEN_MAX = 153716  # Action token range (2048 tokens)


import torch.nn as nn


class _InternVL3_5_VL_Interface(nn.Module):
    """
    Lightweight wrapper around InternVL3.5 model.
    
    Purpose:
        - Unify interface with other VLM backends (CausalLM-like usage).
        - Centralize preprocessing (tokenization + multimodal packing).
        - Provide consistent forward / generate signatures.
    
    Notes:
        - InternVL3.5 is a vision-language model designed for efficient multimodal understanding.
        - Built on Qwen3 LLM backbone (from llm_config in config.json).
        - Supports dynamic image resolution processing.
        - Uses InternVLProcessor for handling images and text.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        """
        Initialize the InternVL3.5 wrapper.
        Following https://huggingface.co/OpenGVLab/InternVL3.5-1B
        
        Parameters:
            config: Configuration object with framework.qwenvl settings
                    Expected: config.framework.qwenvl.base_vlm = model_path or huggingface_id
            **kwargs: Reserved for future extensions
        """
        super().__init__()

        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "AutoModel import failed. Please ensure transformers is properly installed."
            ) from e

        # Use qwenvl config key for compatibility with framework conventions
        # InternVL3.5 is loaded through the same config path as Qwen models
        internvl_config = config.framework.get("qwenvl", {})
        model_id = internvl_config.get("base_vlm", "OpenGVLab/InternVL3.5-1B")
        attn_implementation = internvl_config.get("attn_implementation", "sdpa")

        # Fallback to sdpa if flash_attention_2 is requested but flash_attn is not installed
        if attn_implementation == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError:
                print("[WARNING] flash_attn not installed, falling back to sdpa")
                attn_implementation = "sdpa"

        # Load model - InternVL3.5 uses AutoModel with trust_remote_code
        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,  # InternVL models use custom code
        )
        
        # Load processor - InternVL3.5 has InternVLProcessor
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
        )
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        self.config = config
        self._ACTION_TOKEN_MIN = _ACTION_TOKEN_MIN
        self._ACTION_TOKEN_MAX = _ACTION_TOKEN_MAX

        # Extract hidden size from LLM config (InternVL3.5 uses llm_config)
        if hasattr(self.model, "config"):
            model_config = self.model.config
            # InternVL3.5 has llm_config nested in config
            if hasattr(model_config, "llm_config"):
                if isinstance(model_config.llm_config, dict):
                    self.hidden_size = model_config.llm_config.get("hidden_size", 1024)
                else:
                    self.hidden_size = getattr(model_config.llm_config, "hidden_size", 1024)
            elif hasattr(model_config, "hidden_size"):
                self.hidden_size = model_config.hidden_size
            else:
                self.hidden_size = 1024  # InternVL3.5-1B default
        else:
            self.hidden_size = 1024  # Default fallback
            
        logger.info(f"InternVL3.5 model initialized with hidden_size: {self.hidden_size}")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass delegating to underlying InternVL3.5 backbone.
        
        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            pixel_values: Vision input (handled by InternVL3.5 model)
            labels: Training labels [B, T]
            image_grid_thw: Image grid configuration (optional)
            inputs_embeds: Embedded inputs
            past_key_values: KV cache for generation
            use_cache: Whether to cache KV
            output_attentions: Return attention maps
            output_hidden_states: Return hidden states
            return_dict: Return as dict
            **kwargs: Additional arguments forwarded to model
            
        Returns:
            CausalLMOutputWithPast: Model outputs
        """
        # Prepare inputs for InternVL3.5
        model_inputs = {}
        if input_ids is not None:
            model_inputs["input_ids"] = input_ids
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if labels is not None:
            model_inputs["labels"] = labels
        if inputs_embeds is not None:
            model_inputs["inputs_embeds"] = inputs_embeds
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
        if use_cache is not None:
            model_inputs["use_cache"] = use_cache
        
        model_inputs["output_hidden_states"] = output_hidden_states
        model_inputs["output_attentions"] = output_attentions
        model_inputs["return_dict"] = return_dict
        
        model_inputs.update(kwargs)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**model_inputs)

        return outputs

    def generate(self, **kwargs):
        """
        High-level generation interface (auto-regressive decoding).
        
        Args:
            **kwargs: Arguments following model.generate() signature.
            
        Returns:
            Generated token IDs or generation output
        """
        with torch.autocast("cuda", dtype=torch.float16):
            generation_output = self.model.generate(**kwargs)
        return generation_output

    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal inputs for InternVL3.5 (batched).
        
        Overview:
            For each sample i:
                - Takes a list of PIL images: images[i] = [img_0, img_1, ...]
                - Takes a matching instruction string instructions[i]
                - Optionally formats instruction with CoT_prompt template
                - Applies processor to generate token IDs and vision tensors
                - Optionally masks labels for action tokens
        
        Parameters:
            images (List[List[PIL.Image.Image]]):
                Length B. Each element is a list of PIL images.
            instructions (List[str]):
                Length B, text instructions for VLA task.
            solutions (List[str], optional):
                Ground-truth action sequences for training.
            **kwargs:
                Reserved for future extensions.
        
        Returns:
            BatchFeature:
                Keys: input_ids, attention_mask, pixel_values, labels (if solutions)
                All tensors moved to model device.
        """
        # Prepare text prompts and collect images
        text_inputs = []
        image_inputs_per_sample = []  # Store images per sample

        assert len(images) == len(instructions), "Images and instructions must have the same length"

        for imgs, instruction in zip(images, instructions):
            # Apply CoT prompt if configured
            if "CoT_prompt" in self.config.datasets.vla_data:
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
            else:
                prompt = instruction

            # Store text and images for this sample
            text_inputs.append(prompt)
            image_inputs_per_sample.append(imgs)

        # Process with InternVL3.5 processor
        # For InternVL3.5, we flatten the image list
        image_inputs = []
        for imgs in image_inputs_per_sample:
            image_inputs.extend(imgs)
        
        # Create batch feature using processor
        if image_inputs:
            batch_inputs = self.processor(
                text=text_inputs,
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Handle text-only case
            batch_inputs = self.processor(
                text=text_inputs,
                padding=True,
                return_tensors="pt",
            )

        # Handle label masking for action tokens during training
        if solutions is not None:
            labels = batch_inputs["input_ids"].clone()
            
            for i in range(labels.size(0)):
                seq = labels[i]
                # Mask tokens before first action token
                mask_seq = (seq >= _ACTION_TOKEN_MIN) & (seq <= _ACTION_TOKEN_MAX)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # No action token found, mask entire sequence
                    seq[:] = IGNORE_INDEX
                    if i == 0:  # Only warn once per batch
                        logger.warning(
                            "No action tokens found in sequence. "
                            "Check starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md"
                        )
            
            # Mask padding tokens
            if hasattr(self.processor, "tokenizer"):
                batch_inputs["labels"] = labels
                batch_inputs["labels"][batch_inputs["labels"] == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

        return batch_inputs.to(self.model.device)


if __name__ == "__main__":
    import argparse
    import os

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="./starVLA/config/training/starvla_cotrain_oxe.yaml",
        help="Path to YAML config",
    )
    args, clipargs = parser.parse_known_args()

    if os.getenv("DEBUGPY_ENABLE", "0") == "1":
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/InternVL3_5-1B"
    
    vlm = _InternVL3_5_VL_Interface(cfg)
    print(f"Model loaded: {type(vlm.model).__name__}")
    print(f"Hidden size: {vlm.hidden_size}")
