"""LoRA utilities for training and merging"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
from typing import Dict, Any, Optional
import logging
import os

logger = logging.getLogger(__name__)


class LoRAManager:
    """Manage LoRA configuration, training, and merging"""

    @staticmethod
    def get_lora_config(
        r: int = 16,
        lora_alpha: int = 32,
        target_modules: Optional[list] = None,
        lora_dropout: float = 0.1,
        bias: str = "none",
        task_type: str = "CAUSAL_LM"
    ) -> LoraConfig:
        """Get LoRA configuration"""
        if target_modules is None:
            # Default target modules for common architectures
            target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj",  # Attention layers
                "gate_proj", "up_proj", "down_proj",      # MLP layers
                "lm_head"                                  # Output layer
            ]

        config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=TaskType.CAUSAL_LM if task_type == "CAUSAL_LM" else task_type,
        )

        return config

    @staticmethod
    def prepare_model_for_lora(
        model_name_or_path: str,
        lora_config: LoraConfig,
        use_8bit: bool = False,
        use_4bit: bool = False
    ) -> tuple:
        """Prepare model for LoRA training"""
        logger.info(f"Loading base model: {model_name_or_path}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif use_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16,
            )

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Prepare for k-bit training if using quantization
        if use_4bit or use_8bit:
            model = prepare_model_for_kbit_training(model)

        # Add LoRA adapters
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model, tokenizer

    @staticmethod
    def save_lora_checkpoint(
        model: PeftModel,
        output_dir: str,
        tokenizer: Optional[AutoTokenizer] = None
    ):
        """Save LoRA checkpoint"""
        logger.info(f"Saving LoRA checkpoint to: {output_dir}")

        # Save LoRA weights
        model.save_pretrained(output_dir)

        # Save tokenizer if provided
        if tokenizer:
            tokenizer.save_pretrained(output_dir)

        # Save adapter config
        model.peft_config.save_pretrained(output_dir)

        logger.info("LoRA checkpoint saved successfully")

    @staticmethod
    def merge_lora_to_base(
        base_model_path: str,
        lora_adapter_path: str,
        output_path: str,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None
    ) -> str:
        """Merge LoRA weights into base model"""
        logger.info(f"Merging LoRA adapter into base model...")
        logger.info(f"Base model: {base_model_path}")
        logger.info(f"LoRA adapter: {lora_adapter_path}")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path)

        # Merge LoRA weights
        logger.info("Merging LoRA weights...")
        model = model.merge_and_unload()

        # Save merged model
        logger.info(f"Saving merged model to: {output_path}")
        os.makedirs(output_path, exist_ok=True)
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)

        # Optionally push to hub
        if push_to_hub and hub_model_id:
            logger.info(f"Pushing to hub: {hub_model_id}")
            model.push_to_hub(hub_model_id)
            tokenizer.push_to_hub(hub_model_id)

        logger.info("Model merge completed successfully")
        return output_path

    @staticmethod
    def load_lora_model(
        base_model_path: str,
        lora_adapter_path: str,
        device_map: str = "auto"
    ) -> tuple:
        """Load model with LoRA adapter (without merging)"""
        logger.info(f"Loading model with LoRA adapter...")

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=device_map if torch.cuda.is_available() else None,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, lora_adapter_path)

        return model, tokenizer

    @staticmethod
    def sequential_merge(
        base_model_path: str,
        lora_checkpoints: list,
        output_base_dir: str
    ) -> str:
        """Sequentially merge multiple LoRA checkpoints"""
        current_model_path = base_model_path

        for i, lora_path in enumerate(lora_checkpoints):
            logger.info(f"Merging checkpoint {i+1}/{len(lora_checkpoints)}: {lora_path}")

            output_path = os.path.join(
                output_base_dir,
                f"merged_stage_{i+1}"
            )

            current_model_path = LoRAManager.merge_lora_to_base(
                base_model_path=current_model_path,
                lora_adapter_path=lora_path,
                output_path=output_path
            )

        logger.info(f"Sequential merge completed. Final model: {current_model_path}")
        return current_model_path