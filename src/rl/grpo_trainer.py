"""GRPO (Group Relative Policy Optimization) Trainer with LoRA"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from .dataset_loader import DatasetLoader
from .lora_utils import LoRAManager

logger = logging.getLogger(__name__)


class RyzeGRPOTrainer:
    """Group Relative Policy Optimization Trainer with LoRA"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self.config.get('output_dir', './grpo_outputs')

        # Model parameters
        self.base_model_path = None  # Will be set to SFT merged model

        # Training parameters
        self.batch_size = self.config.get('batch_size', 4)
        self.micro_batch_size = self.config.get('micro_batch_size', 1)
        self.learning_rate = self.config.get('learning_rate', 5e-5)
        self.num_epochs = self.config.get('num_epochs', 3)
        self.max_length = self.config.get('max_length', 1024)
        self.max_new_tokens = self.config.get('max_new_tokens', 256)

        # GRPO parameters
        self.num_samples_per_prompt = self.config.get('num_samples_per_prompt', 4)
        self.temperature = self.config.get('temperature', 0.8)
        self.kl_coef = self.config.get('kl_coef', 0.1)
        self.clip_range = self.config.get('clip_range', 0.2)
        self.value_clip_range = self.config.get('value_clip_range', 0.2)
        self.grpo_epochs = self.config.get('grpo_epochs', 4)

        # LoRA parameters for GRPO
        self.lora_r = self.config.get('lora_r', 8)
        self.lora_alpha = self.config.get('lora_alpha', 16)
        self.lora_dropout = self.config.get('lora_dropout', 0.1)
        self.target_modules = self.config.get('target_modules', None)
        self.use_4bit = self.config.get('use_4bit', False)
        self.use_8bit = self.config.get('use_8bit', False)

        # Auto-merge option
        self.auto_merge = self.config.get('auto_merge', True)

        # Initialize models
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.value_head = None
        self.optimizer = None

    def load_models(self, sft_model_path: str, ref_model_path: Optional[str] = None):
        """Load policy model with LoRA and reference model"""
        logger.info(f"Loading models for GRPO training...")

        # Use SFT model as base
        self.base_model_path = sft_model_path

        # Get LoRA configuration for GRPO
        lora_config = LoRAManager.get_lora_config(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout
        )

        # Prepare policy model with LoRA
        self.model, self.tokenizer = LoRAManager.prepare_model_for_lora(
            model_name_or_path=sft_model_path,
            lora_config=lora_config,
            use_8bit=self.use_8bit,
            use_4bit=self.use_4bit
        )

        # Load reference model (frozen, no LoRA)
        if ref_model_path:
            ref_path = ref_model_path
        else:
            ref_path = sft_model_path

        logger.info(f"Loading reference model from: {ref_path}")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            ref_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Initialize value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        ).to(self.device)

        # Setup optimizer for both LoRA and value head
        optimizer_params = [
            {'params': self.model.parameters(), 'lr': self.learning_rate},
            {'params': self.value_head.parameters(), 'lr': self.learning_rate * 2}
        ]
        self.optimizer = torch.optim.AdamW(optimizer_params)

        logger.info("Models loaded successfully for GRPO")

    def compute_rewards(self, responses: List[str], prompts: List[str]) -> torch.Tensor:
        """Compute rewards for generated responses"""
        rewards = []

        for response, prompt in zip(responses, prompts):
            # Basic reward components
            reward = 0.0

            # Length penalty/bonus
            response_length = len(response.split())
            if 20 < response_length < 200:
                reward += 0.2
            elif response_length > 300:
                reward -= 0.3
            elif response_length < 10:
                reward -= 0.5

            # Coherence check
            if response.strip() and not response.isspace():
                reward += 0.3

            # Diversity reward
            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                reward += unique_ratio * 0.3

            # Repetition penalty
            if len(words) > 3:
                bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
                repeated_bigrams = len(bigrams) - len(set(bigrams))
                if repeated_bigrams > 2:
                    reward -= 0.2 * repeated_bigrams

            # Response relevance (basic check)
            prompt_keywords = set(prompt.lower().split())
            response_keywords = set(response.lower().split())
            overlap = len(prompt_keywords & response_keywords)
            if overlap > 0:
                reward += min(0.3, overlap * 0.05)

            rewards.append(reward)

        return torch.tensor(rewards, device=self.device)

    def group_relative_rewards(self, rewards: torch.Tensor, groups: List[int]) -> torch.Tensor:
        """Compute group-relative rewards"""
        # Normalize rewards within each group
        normalized_rewards = torch.zeros_like(rewards)

        unique_groups = list(set(groups))
        for group_id in unique_groups:
            mask = torch.tensor([g == group_id for g in groups], device=self.device)
            group_rewards = rewards[mask]

            if len(group_rewards) > 1:
                # Normalize within group
                mean = group_rewards.mean()
                std = group_rewards.std() + 1e-8
                normalized_rewards[mask] = (group_rewards - mean) / std
            else:
                normalized_rewards[mask] = 0.0

        return normalized_rewards

    def generate_samples(self, prompts: List[str]) -> Tuple[List[List[str]], torch.Tensor]:
        """Generate multiple samples per prompt"""
        all_responses = []
        all_log_probs = []

        for prompt in prompts:
            prompt_responses = []
            prompt_log_probs = []

            # Generate multiple samples
            for _ in range(self.num_samples_per_prompt):
                inputs = self.tokenizer(
                    prompt,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature,
                        do_sample=True,
                        return_dict_in_generate=True,
                        output_scores=True,
                        pad_token_id=self.tokenizer.pad_token_id
                    )

                    # Extract response
                    prompt_length = inputs['input_ids'].shape[1]
                    response_ids = outputs.sequences[0][prompt_length:]
                    response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    prompt_responses.append(response)

                    # Calculate log probabilities
                    if hasattr(outputs, 'scores'):
                        scores = torch.stack(outputs.scores, dim=1)
                        log_probs = F.log_softmax(scores, dim=-1)
                        selected_log_probs = log_probs[0, torch.arange(len(response_ids)), response_ids]
                        prompt_log_probs.append(selected_log_probs.sum().item())
                    else:
                        prompt_log_probs.append(0.0)

            all_responses.append(prompt_responses)
            all_log_probs.append(torch.tensor(prompt_log_probs))

        return all_responses, torch.stack(all_log_probs)

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Compute advantages using GAE or simple advantage"""
        advantages = rewards - values
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def grpo_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one GRPO optimization step"""
        prompts = batch['prompt']

        # Generate multiple samples per prompt
        logger.info(f"Generating {self.num_samples_per_prompt} samples per prompt...")
        responses_per_prompt, old_log_probs = self.generate_samples(prompts)

        # Flatten responses and create group indices
        all_responses = []
        all_prompts = []
        groups = []

        for i, (prompt, responses) in enumerate(zip(prompts, responses_per_prompt)):
            for response in responses:
                all_responses.append(response)
                all_prompts.append(prompt)
                groups.append(i)

        # Compute rewards
        rewards = self.compute_rewards(all_responses, all_prompts)

        # Apply group-relative reward normalization
        normalized_rewards = self.group_relative_rewards(rewards, groups)

        # Get values from value head
        all_texts = [f"{p} {r}" for p, r in zip(all_prompts, all_responses)]
        inputs = self.tokenizer(
            all_texts,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_length,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            # Use the last token's hidden state
            last_token_hidden = last_hidden_states[:, -1, :]
            values = self.value_head(last_token_hidden).squeeze()

        # Compute advantages
        advantages = self.compute_advantages(normalized_rewards, values)

        # GRPO training loop
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_kl = 0

        for epoch in range(self.grpo_epochs):
            # Shuffle data
            indices = torch.randperm(len(all_responses))

            for i in range(0, len(indices), self.micro_batch_size):
                batch_indices = indices[i:i+self.micro_batch_size]

                # Get batch data
                batch_texts = [all_texts[idx] for idx in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards = normalized_rewards[batch_indices]
                batch_old_log_probs = old_log_probs.view(-1)[batch_indices]

                # Forward pass
                batch_inputs = self.tokenizer(
                    batch_texts,
                    return_tensors='pt',
                    truncation=True,
                    max_length=self.max_length,
                    padding=True
                ).to(self.device)

                outputs = self.model(**batch_inputs, output_hidden_states=True)

                # Get current values
                last_hidden_states = outputs.hidden_states[-1]
                last_token_hidden = last_hidden_states[:, -1, :]
                current_values = self.value_head(last_token_hidden).squeeze()

                # Calculate current log probs (simplified)
                logits = outputs.logits
                current_log_probs = F.log_softmax(logits, dim=-1).mean(dim=(1, 2))

                # Compute KL divergence with reference model
                with torch.no_grad():
                    ref_outputs = self.ref_model(**batch_inputs)
                    ref_logits = ref_outputs.logits
                    ref_log_probs = F.log_softmax(ref_logits, dim=-1).mean(dim=(1, 2))

                kl_div = (current_log_probs - ref_log_probs).mean()

                # Policy loss with clipping
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                policy_loss_1 = -batch_advantages * ratio
                policy_loss_2 = -batch_advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss with clipping
                value_loss = F.mse_loss(current_values, batch_rewards)

                # Total loss
                loss = policy_loss + 0.5 * value_loss + self.kl_coef * kl_div

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.value_head.parameters()),
                    1.0
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += kl_div.item()

        num_updates = (len(all_responses) // self.micro_batch_size) * self.grpo_epochs

        return {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'kl_divergence': total_kl / num_updates,
            'mean_reward': rewards.mean().item(),
            'mean_normalized_reward': normalized_rewards.mean().item()
        }

    def train(self, sft_model_path: str, grpo_data_path: str) -> Dict[str, Any]:
        """Train using GRPO"""
        # Load models
        self.load_models(sft_model_path)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"grpo_lora_{timestamp}"
        run_output_dir = os.path.join(self.output_dir, run_name)
        os.makedirs(run_output_dir, exist_ok=True)

        # Save config
        config_path = os.path.join(run_output_dir, 'grpo_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Load dataset
        data_loader = DatasetLoader.load_rl_dataset(
            data_path=grpo_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

        # Training loop
        logger.info("Starting GRPO training...")
        training_stats = []

        for epoch in range(self.num_epochs):
            epoch_stats = defaultdict(list)

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # GRPO step
                step_stats = self.grpo_step(batch)

                # Record stats
                for key, value in step_stats.items():
                    epoch_stats[key].append(value)

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_stats['loss']:.4f}",
                    'reward': f"{step_stats['mean_reward']:.4f}"
                })

                # Save checkpoint
                if (batch_idx + 1) % 100 == 0:
                    checkpoint_dir = os.path.join(run_output_dir, f"checkpoint-{epoch}-{batch_idx}")
                    self.save_checkpoint(checkpoint_dir)

            # Calculate epoch averages
            epoch_summary = {key: np.mean(values) for key, values in epoch_stats.items()}
            epoch_summary['epoch'] = epoch
            training_stats.append(epoch_summary)

            logger.info(f"Epoch {epoch+1} - Loss: {epoch_summary['loss']:.4f}, "
                       f"Reward: {epoch_summary['mean_reward']:.4f}")

        # Save final LoRA adapter
        final_lora_path = os.path.join(run_output_dir, "final_lora_adapter")
        LoRAManager.save_lora_checkpoint(self.model, final_lora_path, self.tokenizer)

        # Optionally merge LoRA weights
        merged_model_path = None
        if self.auto_merge:
            logger.info("Auto-merging GRPO LoRA weights to base model...")
            merged_model_path = os.path.join(run_output_dir, "merged_model")
            LoRAManager.merge_lora_to_base(
                base_model_path=sft_model_path,
                lora_adapter_path=final_lora_path,
                output_path=merged_model_path
            )

        # Save value head separately
        value_head_path = os.path.join(run_output_dir, 'value_head.pt')
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save training results
        results = {
            'sft_model_path': sft_model_path,
            'grpo_data_path': grpo_data_path,
            'run_name': run_name,
            'run_output_dir': run_output_dir,
            'final_lora_path': final_lora_path,
            'merged_model_path': merged_model_path,
            'value_head_path': value_head_path,
            'training_stats': training_stats,
            'timestamp': timestamp,
            'config': self.config
        }

        results_path = os.path.join(run_output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"GRPO training completed. Results saved to: {run_output_dir}")

        return results

    def save_checkpoint(self, checkpoint_dir: str):
        """Save training checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save LoRA adapter
        LoRAManager.save_lora_checkpoint(self.model, checkpoint_dir, self.tokenizer)

        # Save value head
        value_head_path = os.path.join(checkpoint_dir, 'value_head.pt')
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(self.optimizer.state_dict(), optimizer_path)

        logger.info(f"Checkpoint saved to: {checkpoint_dir}")