"""SFT Trainer with LoRA support"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

from .dataset_loader import DatasetLoader
from .lora_utils import LoRAManager

if TYPE_CHECKING:
    from ..core.task import RyzeTask

logger = logging.getLogger(__name__)


class RyzeSFTLoRATrainer:
    """Supervised Fine-Tuning Trainer with LoRA"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.base_model_name = self.config.get('base_model_name', 'microsoft/phi-2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self.config.get('output_dir', './sft_lora_outputs')

        # Training parameters
        self.batch_size = self.config.get('batch_size', 8)
        self.learning_rate = self.config.get('learning_rate', 3e-4)
        self.num_epochs = self.config.get('num_epochs', 3)
        self.max_length = self.config.get('max_length', 2048)
        self.gradient_accumulation_steps = self.config.get('gradient_accumulation_steps', 4)
        self.max_steps = self.config.get('max_steps', -1)
        self.warmup_ratio = self.config.get('warmup_ratio', 0.03)
        self.weight_decay = self.config.get('weight_decay', 0.001)
        self.num_gpus = self.config.get('num_gpus', 1)

        # LoRA parameters
        self.lora_r = self.config.get('lora_r', 16)
        self.lora_alpha = self.config.get('lora_alpha', 32)
        self.lora_dropout = self.config.get('lora_dropout', 0.1)
        self.target_modules = self.config.get('target_modules', None)
        self.use_4bit = self.config.get('use_4bit', False)
        self.use_8bit = self.config.get('use_8bit', False)

        # Auto-merge option
        self.auto_merge = self.config.get('auto_merge', True)

        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.lora_config = None

    def prepare_model(self, model_name_or_path: Optional[str] = None):
        """Prepare model with LoRA"""
        if model_name_or_path is None:
            model_name_or_path = self.base_model_name

        # Get LoRA configuration
        self.lora_config = LoRAManager.get_lora_config(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout
        )

        # Prepare model with LoRA
        # When using multiple GPUs with HF Trainer DataParallel,
        # device_map must be None so Trainer can wrap the model.
        dm = None if self.num_gpus > 1 else "auto"
        self.model, self.tokenizer = LoRAManager.prepare_model_for_lora(
            model_name_or_path=model_name_or_path,
            lora_config=self.lora_config,
            use_8bit=self.use_8bit,
            use_4bit=self.use_4bit,
            device_map=dm,
        )

        logger.info(f"Model prepared with LoRA on {self.device}")

    def train(
        self,
        train_data_path: str,
        val_data_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """Train the model using SFT with LoRA"""
        if self.model is None:
            self.prepare_model()

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"sft_lora_{self.base_model_name.split('/')[-1]}_{timestamp}"
        run_output_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(run_output_dir, 'training_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Load datasets
        data_loaders = DatasetLoader.load_sft_dataset(
            train_path=train_data_path,
            val_path=val_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=run_output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            gradient_checkpointing=True,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            fp16=torch.cuda.is_available(),
            bf16=False,
            logging_dir=os.path.join(run_output_dir, 'logs'),
            logging_steps=10,
            eval_strategy="steps" if val_data_path else "no",
            eval_steps=100 if val_data_path else None,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True if val_data_path else False,
            metric_for_best_model="eval_loss" if val_data_path else None,
            greater_is_better=False if val_data_path else None,
            report_to=["tensorboard"],
            push_to_hub=False,
            optim="paged_adamw_8bit" if self.use_4bit or self.use_8bit else "adamw_torch",
            remove_unused_columns=False,
            max_steps=self.max_steps,
            run_name=run_name,
        )

        # Create Trainer
        trainer_kwargs = dict(
            model=self.model,
            args=training_args,
            train_dataset=data_loaders['train'].dataset,
            eval_dataset=data_loaders.get('val', {}).dataset if val_data_path else None,
            data_collator=data_collator,
        )
        # transformers >=4.46 renamed 'tokenizer' to 'processing_class'
        import inspect
        _trainer_params = inspect.signature(Trainer.__init__).parameters
        if 'processing_class' in _trainer_params:
            trainer_kwargs['processing_class'] = self.tokenizer
        else:
            trainer_kwargs['tokenizer'] = self.tokenizer

        trainer = Trainer(**trainer_kwargs)

        # Train
        import os as _os

        _cuda_visible = _os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        _n_gpu = training_args.n_gpu
        logger.info(
            "[GPU-TRACK] SFT Trainer — CUDA_VISIBLE_DEVICES=%s, "
            "n_gpu=%d, device=%s, per_device_batch=%d",
            _cuda_visible,
            _n_gpu,
            training_args.device,
            self.batch_size,
        )
        logger.info(
            "[GPU-TRACK] SFT Trainer — max_steps=%d, num_epochs=%d",
            self.max_steps,
            self.num_epochs,
        )
        logger.info("Starting SFT LoRA training...")
        if resume_from_checkpoint:
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            train_result = trainer.train()

        # Save LoRA adapter
        lora_save_path = os.path.join(run_output_dir, "lora_adapter")
        LoRAManager.save_lora_checkpoint(self.model, lora_save_path, self.tokenizer)

        # Optionally merge LoRA to base model
        merged_model_path = None
        if self.auto_merge:
            logger.info("Auto-merging LoRA weights to base model...")
            merged_model_path = os.path.join(run_output_dir, "merged_model")
            LoRAManager.merge_lora_to_base(
                base_model_path=self.base_model_name,
                lora_adapter_path=lora_save_path,
                output_path=merged_model_path
            )

        # Capture GPU info for tracking
        _gpu_info = {
            "cuda_visible_devices": _os.environ.get("CUDA_VISIBLE_DEVICES", "not set"),
            "n_gpu": int(_n_gpu),
            "device": str(training_args.device),
            "pid": _os.getpid(),
        }
        logger.info(
            "[GPU-TRACK] SFT completed — %d steps on %d GPUs (pid=%d, CUDA_VISIBLE_DEVICES=%s)",
            train_result.global_step,
            _gpu_info["n_gpu"],
            _gpu_info["pid"],
            _gpu_info["cuda_visible_devices"],
        )

        # Save training results
        results = {
            'base_model_name': self.base_model_name,
            'train_data_path': train_data_path,
            'val_data_path': val_data_path,
            'run_name': run_name,
            'run_output_dir': run_output_dir,
            'lora_adapter_path': lora_save_path,
            'merged_model_path': merged_model_path,
            'training_loss': train_result.training_loss,
            'training_steps': train_result.global_step,
            'gpu_info': _gpu_info,
            'timestamp': timestamp,
            'lora_config': {
                'r': self.lora_r,
                'alpha': self.lora_alpha,
                'dropout': self.lora_dropout
            },
            'config': self.config
        }

        results_path = os.path.join(run_output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"Training completed. Results saved to: {run_output_dir}")

        return results

    def generate_rl_dataset(
        self,
        model_path: str,
        input_data_path: str,
        output_path: str,
        use_merged_model: bool = True
    ) -> Dict[str, Any]:
        """Generate dataset for RL training using SFT model"""
        logger.info(f"Generating RL dataset using model: {model_path}")

        # Load model (either merged or with LoRA adapter)
        if use_merged_model:
            # Load merged model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        else:
            # Load with LoRA adapter
            base_model_path = self.config.get('base_model_for_generation', self.base_model_name)
            self.model, self.tokenizer = LoRAManager.load_lora_model(
                base_model_path=base_model_path,
                lora_adapter_path=model_path
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
            inputs = self.tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length
            )
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
            'model_path': model_path,
            'input_data_path': input_data_path,
            'output_path': output_path,
            'num_samples': len(rl_dataset),
            'use_merged_model': use_merged_model
        }

    def as_task(self) -> RyzeTask:
        """Create a RyzeTask wrapper for this trainer."""
        from ..core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType

        trainer = self

        class SFTTrainTask(RyzeTask):
            def __init__(self):
                super().__init__(
                    task_type=TaskType.SFT_TRAIN,
                    name="SFT LoRA Training",
                )

            def resource_requirements(self) -> ResourceRequirement:
                return ResourceRequirement(gpu_count=trainer.num_gpus, memory_gb=16.0, estimated_duration_s=3600)

            def validate_inputs(self) -> bool:
                return True

            def execute(self, inputs: dict) -> TaskResult:
                train_path = inputs.get("train_path", "")
                val_path = inputs.get("val_path")
                if not train_path:
                    return TaskResult(status=TaskStatus.FAILED, error="No train_path provided")
                results = trainer.train(train_path, val_path)
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    output=results,
                    metrics={"training_loss": results.get("training_loss", 0)},
                    artifacts=[
                        results.get("lora_adapter_path", ""),
                        results.get("merged_model_path", ""),
                    ],
                )

            def to_config(self) -> dict:
                """Return trainer class path and config for remote reconstruction."""
                return {
                    "trainer_class_path": "ryze.rl.sft_lora_trainer.RyzeSFTLoRATrainer",
                    "trainer_config": dict(trainer.config),
                }

        return SFTTrainTask()
