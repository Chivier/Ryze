"""Tests for ryze.rl.sft_lora_trainer module."""

from unittest.mock import MagicMock, patch


class TestRyzeSFTLoRATrainer:
    def test_init_defaults(self):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        trainer = RyzeSFTLoRATrainer()
        assert trainer.base_model_name == "microsoft/phi-2"
        assert trainer.lora_r == 16
        assert trainer.auto_merge is True

    def test_init_custom_config(self):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        trainer = RyzeSFTLoRATrainer({"base_model_name": "test", "lora_r": 8, "num_epochs": 2})
        assert trainer.base_model_name == "test"
        assert trainer.lora_r == 8
        assert trainer.num_epochs == 2

    @patch("ryze.rl.sft_lora_trainer.LoRAManager")
    def test_prepare_model(self, mock_lora):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_lora.get_lora_config.return_value = MagicMock()
        mock_lora.prepare_model_for_lora.return_value = (mock_model, mock_tokenizer)

        trainer = RyzeSFTLoRATrainer({"base_model_name": "test"})
        trainer.prepare_model()
        assert trainer.model is mock_model
        assert trainer.tokenizer is mock_tokenizer

    @patch("ryze.rl.sft_lora_trainer.LoRAManager")
    @patch("ryze.rl.sft_lora_trainer.Trainer")
    @patch("ryze.rl.sft_lora_trainer.DatasetLoader")
    @patch("ryze.rl.sft_lora_trainer.DataCollatorForLanguageModeling")
    def test_train_returns_results(self, mock_collator, mock_loader, mock_hf_trainer, mock_lora, tmp_path, sample_sft_data):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_tokenizer = MagicMock()
        mock_lora.get_lora_config.return_value = MagicMock()
        mock_lora.prepare_model_for_lora.return_value = (mock_model, mock_tokenizer)

        mock_train_dataset = MagicMock()
        mock_loader.load_sft_dataset.return_value = {"train": MagicMock(dataset=mock_train_dataset)}

        mock_result = MagicMock()
        mock_result.training_loss = 0.5
        mock_result.global_step = 100
        mock_hf_trainer.return_value.train.return_value = mock_result

        trainer = RyzeSFTLoRATrainer({"output_dir": str(tmp_path), "auto_merge": False})
        results = trainer.train(sample_sft_data)
        assert results["training_loss"] == 0.5
        assert results["training_steps"] == 100

    def test_as_task_creates_sft_task(self):
        from ryze.core.task import TaskType
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        trainer = RyzeSFTLoRATrainer()
        task = trainer.as_task()
        assert task.task_type == TaskType.SFT_TRAIN
        reqs = task.resource_requirements()
        assert reqs.gpu_count == 1

    def test_as_task_validate_always_true(self):
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        trainer = RyzeSFTLoRATrainer()
        task = trainer.as_task()
        assert task.validate_inputs() is True

    def test_as_task_execute_no_train_path(self):
        from ryze.core.task import TaskStatus
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer
        trainer = RyzeSFTLoRATrainer()
        task = trainer.as_task()
        result = task.execute({})
        assert result.status == TaskStatus.FAILED
