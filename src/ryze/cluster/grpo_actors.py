"""Ray actors for distributed GRPO training (actor-learner pattern).

Splits GRPO's generation (sampling) and training (optimization) onto
separate GPU pools.  The generation actor explores by sampling responses
and computing advantages; the training actor runs the inner PPO-style
optimization loop.  Weight sync (LoRA + value head) happens after each
training step.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Ray is imported lazily inside each actor class method to avoid
# import errors when ray is not installed.


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Detach tensor and convert to numpy for Ray zero-copy transport."""
    return t.detach().cpu().numpy()


def _to_tensor(a: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert numpy array back to tensor on the given device."""
    # np.copy() ensures the array is writable (Ray zero-copy makes it read-only)
    return torch.from_numpy(np.copy(a)).to(device)


class GRPOGenerationActor:
    """Ray actor that holds policy (no grad) + ref (frozen) + value head.

    Performs the generation phase of ``grpo_step()``:
    1. Sample responses via ``generate_samples()``
    2. Compute rewards and group-relative normalization
    3. Forward pass for old_log_probs, values, ref_log_probs
    4. Compute advantages

    All returned tensors are serialized as numpy arrays.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._delegate = None  # RyzeGRPOTrainer used as helper
        self._model = None
        self._ref_model = None
        self._value_head = None
        self._tokenizer = None

    def get_gpu_info(self) -> Dict[str, Any]:
        """Return GPU assignment details for this actor."""
        import os

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return {
            "actor": "GRPOGenerationActor",
            "cuda_visible_devices": cuda_visible,
            "gpu_count": gpu_count,
            "device": str(self._device),
            "gpu_names": gpu_names,
            "pid": os.getpid(),
        }

    def load_models(self, sft_model_path: str) -> None:
        """Load policy (LoRA, no grad), ref model, value head, tokenizer."""
        from ..rl.grpo_trainer import RyzeGRPOTrainer

        self._delegate = RyzeGRPOTrainer(self._config)
        self._delegate.load_models(sft_model_path)

        # Keep references for convenience
        self._model = self._delegate.model
        self._ref_model = self._delegate.ref_model
        self._value_head = self._delegate.value_head
        self._tokenizer = self._delegate.tokenizer

        # Generation actor: inference mode, no optimizer needed
        self._model.requires_grad_(False)
        self._value_head.requires_grad_(False)
        # Detach optimizer to free memory
        self._delegate.optimizer = None
        logger.info("GRPOGenerationActor: models loaded on %s", self._device)

    def generate_and_prepare(self, prompts: List[str]) -> Dict[str, Any]:
        """Run the generation phase and return data for the training actor.

        Args:
            prompts: Batch of prompt strings.

        Returns:
            Dict with numpy arrays and scalar metrics.
        """
        d = self._delegate

        # 1. Generate samples
        responses_per_prompt, _ = d.generate_samples(prompts)

        # 2. Flatten
        all_responses: List[str] = []
        all_prompts: List[str] = []
        groups: List[int] = []
        for i, (prompt, responses) in enumerate(zip(prompts, responses_per_prompt)):
            for response in responses:
                all_responses.append(response)
                all_prompts.append(prompt)
                groups.append(i)

        # 3. Rewards + normalization
        rewards = d.compute_rewards(all_responses, all_prompts)
        normalized_rewards = d.group_relative_rewards(rewards, groups)

        # 4. Forward pass for old_log_probs, values, ref_log_probs
        all_texts = [f"{p} {r}" for p, r in zip(all_prompts, all_responses)]
        inputs = self._tokenizer(
            all_texts,
            return_tensors="pt",
            truncation=True,
            max_length=d.max_length,
            padding=True,
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, -1, :].float()
            values = self._value_head(last_hidden).squeeze()

            old_log_probs = d._per_token_log_probs(
                outputs.logits, inputs["input_ids"], inputs["attention_mask"]
            )

            ref_outputs = self._ref_model(**inputs)
            ref_log_probs_all = d._per_token_log_probs(
                ref_outputs.logits,
                inputs["input_ids"],
                inputs["attention_mask"],
            )

        # 5. Advantages
        advantages = d.compute_advantages(normalized_rewards, values)

        # 6. Pack as numpy for zero-copy Ray transport
        return {
            "all_texts": all_texts,
            "advantages": _to_numpy(advantages),
            "normalized_rewards": _to_numpy(normalized_rewards),
            "old_log_probs": _to_numpy(old_log_probs),
            "ref_log_probs_all": _to_numpy(ref_log_probs_all),
            "mean_reward": rewards.mean().item(),
            "mean_normalized_reward": normalized_rewards.mean().item(),
        }

    def update_weights(
        self,
        lora_state_dict: Dict[str, Any],
        value_head_state_dict: Dict[str, Any],
    ) -> None:
        """Sync weights from the training actor."""
        # Load LoRA weights
        current_sd = self._model.state_dict()
        for k, v in lora_state_dict.items():
            if k in current_sd:
                current_sd[k] = v
        self._model.load_state_dict(current_sd)

        # Load value head weights
        self._value_head.load_state_dict(value_head_state_dict)
        logger.info("GRPOGenerationActor: weights synced from training actor")

    def get_lora_weights(self) -> Dict[str, Any]:
        """Return LoRA adapter state dict (for initial sync)."""
        return {k: v.cpu() for k, v in self._model.state_dict().items() if "lora_" in k}


class GRPOTrainingActor:
    """Ray actor that holds policy (LoRA, train) + ref + value head + optimizer.

    Performs the training phase of ``grpo_step()``: inner GRPO epoch loop
    with micro-batches, forward pass, KL penalty, clipped policy loss,
    value loss, backward, optimizer step.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._delegate = None  # RyzeGRPOTrainer for helpers + state
        self._model = None
        self._ref_model = None
        self._value_head = None
        self._optimizer = None
        self._tokenizer = None

    def get_gpu_info(self) -> Dict[str, Any]:
        """Return GPU assignment details for this actor."""
        import os

        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
        gpu_count = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        return {
            "actor": "GRPOTrainingActor",
            "cuda_visible_devices": cuda_visible,
            "gpu_count": gpu_count,
            "device": str(self._device),
            "gpu_names": gpu_names,
            "pid": os.getpid(),
        }

    def load_models(self, sft_model_path: str) -> None:
        """Load policy (LoRA, train), ref, value head, optimizer."""
        from ..rl.grpo_trainer import RyzeGRPOTrainer

        self._delegate = RyzeGRPOTrainer(self._config)
        self._delegate.load_models(sft_model_path)

        self._model = self._delegate.model
        self._ref_model = self._delegate.ref_model
        self._value_head = self._delegate.value_head
        self._optimizer = self._delegate.optimizer
        self._tokenizer = self._delegate.tokenizer

        # Training actor: train mode
        self._model.train()
        self._value_head.train()
        logger.info("GRPOTrainingActor: models loaded on %s", self._device)

    def grpo_train_step(self, gen_data: Dict[str, Any]) -> Dict[str, float]:
        """Run the training phase using data from the generation actor.

        Args:
            gen_data: Dict from ``GRPOGenerationActor.generate_and_prepare()``.

        Returns:
            Step statistics dict.
        """
        import torch.nn.functional as F

        d = self._delegate
        all_texts = gen_data["all_texts"]
        advantages = _to_tensor(gen_data["advantages"], self._device).float()
        normalized_rewards = _to_tensor(gen_data["normalized_rewards"], self._device).float()
        old_log_probs = _to_tensor(gen_data["old_log_probs"], self._device)
        # ref_log_probs_all from gen actor not used here; training actor
        # recomputes KL from its own local ref model per micro-batch.

        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        for epoch in range(d.grpo_epochs):
            indices = torch.randperm(len(all_texts))

            for i in range(0, len(indices), d.micro_batch_size):
                batch_indices = indices[i : i + d.micro_batch_size]

                batch_texts = [all_texts[idx] for idx in batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_rewards = normalized_rewards[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]

                # Forward pass
                batch_inputs = self._tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=d.max_length,
                    padding=True,
                ).to(self._device)

                outputs = self._model(**batch_inputs, output_hidden_states=True)

                # Values
                last_hidden = outputs.hidden_states[-1][:, -1, :].float()
                current_values = self._value_head(last_hidden).squeeze(-1)

                # Per-token log probs
                current_log_probs = d._per_token_log_probs(
                    outputs.logits,
                    batch_inputs["input_ids"],
                    batch_inputs["attention_mask"],
                )

                # KL with local ref model
                with torch.no_grad():
                    ref_outputs = self._ref_model(**batch_inputs)
                    micro_ref_log_probs = d._per_token_log_probs(
                        ref_outputs.logits,
                        batch_inputs["input_ids"],
                        batch_inputs["attention_mask"],
                    )
                kl_div = (current_log_probs - micro_ref_log_probs).mean()

                # Clipped policy loss
                log_ratio = current_log_probs - batch_old_log_probs
                log_ratio = torch.clamp(log_ratio, -5.0, 5.0)
                ratio = torch.exp(log_ratio)

                policy_loss_1 = -batch_advantages * ratio
                policy_loss_2 = -batch_advantages * torch.clamp(
                    ratio, 1 - d.clip_range, 1 + d.clip_range
                )
                policy_loss = torch.max(policy_loss_1, policy_loss_2).mean()

                # Value loss
                value_loss = F.mse_loss(current_values.float(), batch_rewards)

                # Total loss
                loss = policy_loss + 0.5 * value_loss + d.kl_coef * kl_div

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(
                        "Skipping step with NaN/Inf loss (policy=%.4f, value=%.4f, kl=%.4f)",
                        policy_loss.item(),
                        value_loss.item(),
                        kl_div.item(),
                    )
                    continue

                self._optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self._model.parameters()) + list(self._value_head.parameters()),
                    1.0,
                )
                self._optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_kl += kl_div.item()
                num_updates += 1

        if num_updates == 0:
            num_updates = 1

        return {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "kl_divergence": total_kl / num_updates,
            "mean_reward": gen_data["mean_reward"],
            "mean_normalized_reward": gen_data["mean_normalized_reward"],
            "num_updates": num_updates,
        }

    def get_lora_weights(self) -> tuple:
        """Return (lora_state_dict, value_head_state_dict) for weight sync."""
        lora_sd = {k: v.cpu() for k, v in self._model.state_dict().items() if "lora_" in k}
        vh_sd = {k: v.cpu() for k, v in self._value_head.state_dict().items()}
        return lora_sd, vh_sd

    def save_checkpoint(self, path: str) -> None:
        """Save LoRA adapter + value head + optimizer state."""
        import os

        from ..rl.lora_utils import LoRAManager

        os.makedirs(path, exist_ok=True)
        LoRAManager.save_lora_checkpoint(self._model, path, self._tokenizer)
        torch.save(
            self._value_head.state_dict(),
            os.path.join(path, "value_head.pt"),
        )
        torch.save(
            self._optimizer.state_dict(),
            os.path.join(path, "optimizer.pt"),
        )
        logger.info("GRPOTrainingActor: checkpoint saved to %s", path)
