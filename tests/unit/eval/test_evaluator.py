"""Tests for ryze.eval.evaluator module."""

from unittest.mock import MagicMock, patch


class TestRyzeEvaluator:
    def test_init_defaults(self):
        from ryze.eval.evaluator import RyzeEvaluator
        evaluator = RyzeEvaluator()
        assert evaluator.max_new_tokens == 256
        assert evaluator.temperature == 0.7

    @patch("ryze.eval.evaluator.AutoModelForCausalLM")
    @patch("ryze.eval.evaluator.AutoTokenizer")
    def test_generate_response(self, mock_tok, mock_model_cls):
        import torch

        from ryze.eval.evaluator import RyzeEvaluator

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token = "<eos>"
        inputs = {"input_ids": torch.ones(1, 5, dtype=torch.long), "attention_mask": torch.ones(1, 5, dtype=torch.long)}
        mock_tokenizer.return_value = inputs
        mock_tokenizer.decode.return_value = "Generated text"
        mock_tok.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model.generate.return_value = torch.ones(1, 10, dtype=torch.long)
        mock_model_cls.from_pretrained.return_value = mock_model

        evaluator = RyzeEvaluator()
        evaluator.model = mock_model
        evaluator.tokenizer = mock_tokenizer
        response = evaluator.generate_response("test prompt")
        assert isinstance(response, str)

    def test_as_task_creates_eval_task(self):
        from ryze.core.task import TaskType
        from ryze.eval.evaluator import RyzeEvaluator
        evaluator = RyzeEvaluator()
        task = evaluator.as_task(model_path="test_model")
        assert task.task_type == TaskType.EVALUATION
