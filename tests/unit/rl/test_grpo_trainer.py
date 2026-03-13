"""Tests for ryze.rl.grpo_trainer module."""

from unittest.mock import patch

import torch


class TestRyzeGRPOTrainer:
    def test_init_defaults(self):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        assert trainer.num_samples_per_prompt == 4
        assert trainer.kl_coef == 0.1
        assert trainer.auto_merge is True

    @patch("ryze.rl.grpo_trainer.torch.cuda.is_available", return_value=False)
    def test_compute_rewards(self, mock_cuda):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        trainer.device = torch.device("cpu")
        rewards = trainer.compute_rewards(
            ["This is a reasonable response with good content."],
            ["Summarize the text:"]
        )
        assert rewards.shape == (1,)

    @patch("ryze.rl.grpo_trainer.torch.cuda.is_available", return_value=False)
    def test_group_relative_rewards(self, mock_cuda):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        trainer.device = torch.device("cpu")
        rewards = torch.tensor([1.0, 2.0, 3.0, 0.5, 1.5, 2.5])
        groups = [0, 0, 0, 1, 1, 1]
        normalized = trainer.group_relative_rewards(rewards, groups)
        assert normalized.shape == rewards.shape
        # Within each group, mean should be ~0
        group0 = normalized[:3]
        assert abs(group0.mean().item()) < 0.01

    @patch("ryze.rl.grpo_trainer.torch.cuda.is_available", return_value=False)
    def test_compute_advantages(self, mock_cuda):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        rewards = torch.tensor([1.0, 2.0, 3.0])
        values = torch.tensor([1.5, 1.5, 1.5])
        advantages = trainer.compute_advantages(rewards, values)
        assert advantages.shape == rewards.shape

    def test_as_task_creates_grpo_task(self):
        from ryze.core.task import TaskType
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        task = trainer.as_task()
        assert task.task_type == TaskType.GRPO_TRAIN
        assert task.resource_requirements().gpu_count == 1

    def test_as_task_execute_no_model_path(self):
        from ryze.core.task import TaskStatus
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        trainer = RyzeGRPOTrainer()
        task = trainer.as_task()
        result = task.execute({})
        assert result.status == TaskStatus.FAILED
