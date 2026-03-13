"""SFT Trainer Module"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class RyzeSFTTrainer:
    """Supervised Fine-Tuning Trainer"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model_name = self.config.get('model_name', 'microsoft/phi-2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self.config.get('output_dir', './sft_outputs')

        # Training parameters
        self.batch_size = self.config.get('batch_size', 8)
        self.learning_rate = self.config.get('learning_rate', 5e-5)
        self.num_epochs = self.config.get('num_epochs', 3)
        self.max_length = self.config.get('max_length', 2048)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        self.warmup_steps = self.config.get('warmup_steps', 100)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        logger.info(f"Model loaded on {self.device}")

    def train(self, train_data_path: str, val_data_path: Optional[str] = None) -> Dict[str, Any]:
        """Train the model using SFT"""
        if self.model is None:
            self.load_model()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(self.output_dir, f"sft_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        # Load datasets
        data_loaders = DatasetLoader.load_sft_dataset(
            train_path=train_data_path,
            val_path=val_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=self.warmup_steps,
            learning_rate=self.learning_rate,
            fp16=torch.cuda.is_available(),
            logging_dir=os.path.join(run_output_dir, 'logs'),
            logging_steps=10,
            eval_strategy="steps" if val_data_path else "no",
            eval_steps=500 if val_data_path else None,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True if val_data_path else False,
            report_to=["tensorboard"],
            push_to_hub=False,
        )

        # Create Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data_loaders['train'].dataset,
            eval_dataset=data_loaders.get('val', {}).dataset if val_data_path else None,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting SFT training...")
        train_result = trainer.train()

        # Save model
        model_save_path = os.path.join(run_output_dir, "final_model")
        trainer.save_model(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)

        # Save training results
        results = {
            'model_name': self.model_name,
            'train_data_path': train_data_path,
            'val_data_path': val_data_path,
            'run_output_dir': run_output_dir,
            'model_save_path': model_save_path,
            'training_loss': train_result.training_loss,
            'training_steps': train_result.global_step,
            'timestamp': timestamp,
            'config': self.config
        }

        results_path = os.path.join(run_output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Training completed. Model saved to: {model_save_path}")

        return results

    def generate_rl_dataset(self, sft_model_path: str, input_data_path: str, output_path: str) -> Dict[str, Any]:
        """Generate dataset for RL training using SFT model"""
        logger.info(f"Generating RL dataset using SFT model: {sft_model_path}")

        # Load SFT model
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()

        # Load input data
        with open(input_data_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)

        rl_dataset = []

        for item in input_data:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
            else:
                prompt = f"{instruction}\n\nResponse:"

            # Generate response using SFT model
            inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id
                )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text.split("Response:")[-1].strip()

            rl_dataset.append({
                'instruction': instruction,
                'input': input_text,
                'sft_output': response,
                'original_output': item.get('output', ''),
                'prompt': prompt
            })

        # Save RL dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(rl_dataset, f, ensure_ascii=False, indent=2)

        logger.info(f"Generated {len(rl_dataset)} samples for RL training")

        return {
            'sft_model_path': sft_model_path,
            'input_data_path': input_data_path,
            'output_path': output_path,
            'num_samples': len(rl_dataset)
        }
