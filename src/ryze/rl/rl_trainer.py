"""RL Trainer Module using PPO"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)


class RyzeRLTrainer:
    """Reinforcement Learning Trainer using PPO"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self.config.get('output_dir', './rl_outputs')

        # Model parameters
        self.model_name = self.config.get('model_name', None)  # Will use SFT model
        self.ref_model_name = self.config.get('ref_model_name', None)  # Reference model

        # Training parameters
        self.batch_size = self.config.get('batch_size', 4)
        self.learning_rate = self.config.get('learning_rate', 1e-5)
        self.num_epochs = self.config.get('num_epochs', 3)
        self.max_length = self.config.get('max_length', 1024)

        # PPO parameters
        self.ppo_epochs = self.config.get('ppo_epochs', 4)
        self.chunk_size = self.config.get('chunk_size', 128)
        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)
        self.clip_ratio = self.config.get('clip_ratio', 0.2)
        self.value_clip = self.config.get('value_clip', 0.2)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.0)

        # Initialize models
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.value_head = None
        self.optimizer = None

    def load_models(self, sft_model_path: str, ref_model_path: Optional[str] = None):
        """Load policy model and reference model"""
        logger.info(f"Loading policy model from: {sft_model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load policy model (from SFT)
        self.model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )

        # Load reference model (frozen)
        if ref_model_path:
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                ref_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            # Use the same model as reference (frozen copy)
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                sft_model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

        # Initialize value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1).to(self.device)

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.value_head.parameters()),
            lr=self.learning_rate
        )

        logger.info("Models loaded successfully")

    def compute_rewards(self, responses: List[str], prompts: List[str]) -> torch.Tensor:
        """Compute rewards for generated responses"""
        # This is a simplified reward function
        # In practice, you might use a reward model or human feedback
        rewards = []

        for response, prompt in zip(responses, prompts):
            # Simple heuristics for rewards
            reward = 0.0

            # Length penalty/bonus
            response_length = len(response.split())
            if 10 < response_length < 200:
                reward += 0.1
            elif response_length > 200:
                reward -= 0.2

            # Check for coherence (very basic)
            if response.strip() and not response.isspace():
                reward += 0.2

            # Diversity bonus (avoid repetition)
            words = response.lower().split()
            if len(words) > 0:
                unique_ratio = len(set(words)) / len(words)
                reward += unique_ratio * 0.3

            rewards.append(reward)

        return torch.tensor(rewards, device=self.device)

    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> tuple:
        """Compute advantages using GAE"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.lam * last_advantage

        returns = advantages + values
        return advantages, returns

    def ppo_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one PPO optimization step"""
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        prompts = batch['prompt']

        # Generate responses
        self.model.eval()
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Extract only the generated part
            generated_texts = []
            for i, gen in enumerate(generated):
                prompt_length = input_ids[i].shape[0]
                response_ids = gen[prompt_length:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                generated_texts.append(response_text)

            # Compute rewards
            rewards = self.compute_rewards(generated_texts, prompts)

            # Get values from value head
            model_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = model_outputs.hidden_states[-1]
            values = self.value_head(last_hidden_state[:, -1, :]).squeeze()

            # Compute advantages
            advantages, returns = self.compute_advantages(rewards, values)

            # Get old log probs
            old_logits = self.model(input_ids=generated, attention_mask=torch.ones_like(generated)).logits
            old_log_probs = torch.log_softmax(old_logits, dim=-1)

        # PPO training
        self.model.train()
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0

        for _ in range(self.ppo_epochs):
            # Forward pass
            outputs = self.model(input_ids=generated, attention_mask=torch.ones_like(generated), output_hidden_states=True)
            logits = outputs.logits
            log_probs = torch.log_softmax(logits, dim=-1)

            # Get current values
            last_hidden_state = outputs.hidden_states[-1]
            current_values = self.value_head(last_hidden_state[:, -1, :]).squeeze()

            # Compute ratio
            ratio = torch.exp(log_probs - old_log_probs).mean(dim=-1).mean(dim=-1)

            # Policy loss
            policy_loss_1 = -advantages * ratio
            policy_loss_2 = -advantages * torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

            # Value loss
            value_loss = nn.functional.mse_loss(current_values, returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.value_head.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()

            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            'loss': total_loss / self.ppo_epochs,
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs,
            'mean_reward': rewards.mean().item()
        }

    def train(self, sft_model_path: str, rl_data_path: str) -> Dict[str, Any]:
        """Train using PPO"""
        # Load models
        self.load_models(sft_model_path)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = os.path.join(self.output_dir, f"rl_run_{timestamp}")
        os.makedirs(run_output_dir, exist_ok=True)

        # Load dataset
        data_loader = DatasetLoader.load_rl_dataset(
            data_path=rl_data_path,
            tokenizer=self.tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

        # Training loop
        logger.info("Starting RL training with PPO...")
        training_stats = []

        for epoch in range(self.num_epochs):
            epoch_stats = {
                'epoch': epoch,
                'losses': [],
                'policy_losses': [],
                'value_losses': [],
                'rewards': []
            }

            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}")

            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # PPO step
                step_stats = self.ppo_step(batch)

                # Record stats
                epoch_stats['losses'].append(step_stats['loss'])
                epoch_stats['policy_losses'].append(step_stats['policy_loss'])
                epoch_stats['value_losses'].append(step_stats['value_loss'])
                epoch_stats['rewards'].append(step_stats['mean_reward'])

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{step_stats['loss']:.4f}",
                    'reward': f"{step_stats['mean_reward']:.4f}"
                })

                # Save checkpoint every 100 steps
                if (batch_idx + 1) % 100 == 0:
                    checkpoint_dir = os.path.join(run_output_dir, f"checkpoint-{epoch}-{batch_idx}")
                    self.save_model(checkpoint_dir)

            # Calculate epoch averages
            epoch_stats['avg_loss'] = np.mean(epoch_stats['losses'])
            epoch_stats['avg_policy_loss'] = np.mean(epoch_stats['policy_losses'])
            epoch_stats['avg_value_loss'] = np.mean(epoch_stats['value_losses'])
            epoch_stats['avg_reward'] = np.mean(epoch_stats['rewards'])

            training_stats.append(epoch_stats)

            logger.info(f"Epoch {epoch+1} - Loss: {epoch_stats['avg_loss']:.4f}, "
                       f"Reward: {epoch_stats['avg_reward']:.4f}")

        # Save final model
        final_model_path = os.path.join(run_output_dir, "final_model")
        self.save_model(final_model_path)

        # Save training results
        results = {
            'sft_model_path': sft_model_path,
            'rl_data_path': rl_data_path,
            'run_output_dir': run_output_dir,
            'final_model_path': final_model_path,
            'training_stats': training_stats,
            'timestamp': timestamp,
            'config': self.config
        }

        results_path = os.path.join(run_output_dir, 'training_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"RL training completed. Model saved to: {final_model_path}")

        return results

    def save_model(self, output_dir: str):
        """Save model and value head"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save value head
        value_head_path = os.path.join(output_dir, 'value_head.pt')
        torch.save(self.value_head.state_dict(), value_head_path)

        # Save config
        config_path = os.path.join(output_dir, 'rl_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to: {output_dir}")
