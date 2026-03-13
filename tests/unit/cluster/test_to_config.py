"""Tests for to_config() protocol on GPU task classes (spec §6.3)."""

import importlib

import pytest


class TestSFTTrainTaskConfig:
    def test_to_config_returns_correct_keys(self):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

        config = {"base_model_name": "test-model", "batch_size": 2}
        task = RyzeSFTLoRATrainer(config).as_task()
        result = task.to_config()
        assert result["trainer_class_path"] == "ryze.rl.sft_lora_trainer.RyzeSFTLoRATrainer"
        assert result["trainer_config"]["base_model_name"] == "test-model"
        assert result["trainer_config"]["batch_size"] == 2

    def test_remote_rebuild_produces_equivalent_task(self):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

        original_config = {"base_model_name": "test", "output_dir": "/tmp/test_sft"}
        task = RyzeSFTLoRATrainer(original_config).as_task()
        spec = task.to_config()

        # Simulate remote rebuild
        module_path, class_name = spec["trainer_class_path"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        rebuilt_task = trainer_class(spec["trainer_config"]).as_task()

        assert rebuilt_task.task_type == task.task_type
        assert rebuilt_task.name == task.name


class TestGRPOTrainTaskConfig:
    def test_to_config_returns_correct_keys(self):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer

        config = {"batch_size": 4, "learning_rate": 5e-5}
        task = RyzeGRPOTrainer(config).as_task()
        result = task.to_config()
        assert result["trainer_class_path"] == "ryze.rl.grpo_trainer.RyzeGRPOTrainer"
        assert result["trainer_config"]["batch_size"] == 4

    def test_remote_rebuild_produces_equivalent_task(self):
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer

        config = {"batch_size": 2, "output_dir": "/tmp/test_grpo"}
        task = RyzeGRPOTrainer(config).as_task()
        spec = task.to_config()

        module_path, class_name = spec["trainer_class_path"].rsplit(".", 1)
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        rebuilt_task = trainer_class(spec["trainer_config"]).as_task()

        assert rebuilt_task.task_type == task.task_type
        assert rebuilt_task.name == task.name


class TestEvaluationTaskConfig:
    def test_to_config_returns_correct_keys(self):
        from ryze.eval.evaluator import RyzeEvaluator

        config = {"max_new_tokens": 64}
        task = RyzeEvaluator(config).as_task()
        result = task.to_config()
        assert result["trainer_class_path"] == "ryze.eval.evaluator.RyzeEvaluator"
        assert result["trainer_config"]["max_new_tokens"] == 64
