"""Tests for ryze.rl.lora_utils module."""

from unittest.mock import MagicMock, patch


class TestLoRAManager:
    @patch("ryze.rl.lora_utils.LoraConfig")
    def test_get_lora_config_defaults(self, mock_config):
        from ryze.rl.lora_utils import LoRAManager
        LoRAManager.get_lora_config()
        mock_config.assert_called_once()
        call_kwargs = mock_config.call_args
        assert call_kwargs.kwargs["r"] == 16
        assert call_kwargs.kwargs["lora_alpha"] == 32

    @patch("ryze.rl.lora_utils.LoraConfig")
    def test_get_lora_config_custom(self, mock_config):
        from ryze.rl.lora_utils import LoRAManager
        LoRAManager.get_lora_config(r=8, lora_alpha=16, lora_dropout=0.05)
        call_kwargs = mock_config.call_args
        assert call_kwargs.kwargs["r"] == 8
        assert call_kwargs.kwargs["lora_dropout"] == 0.05

    @patch("ryze.rl.lora_utils.get_peft_model")
    @patch("ryze.rl.lora_utils.AutoModelForCausalLM")
    @patch("ryze.rl.lora_utils.AutoTokenizer")
    def test_prepare_model_for_lora(self, mock_tok, mock_model_cls, mock_peft):
        from ryze.rl.lora_utils import LoRAManager

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tok.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_peft.return_value = MagicMock()

        model, tokenizer = LoRAManager.prepare_model_for_lora("test-model", MagicMock())
        assert model is not None
        assert tokenizer.pad_token == "<eos>"

    @patch("ryze.rl.lora_utils.PeftModel")
    def test_save_lora_checkpoint(self, mock_peft):
        from ryze.rl.lora_utils import LoRAManager
        mock_model = MagicMock()
        LoRAManager.save_lora_checkpoint(mock_model, "/tmp/test")
        mock_model.save_pretrained.assert_called_once_with("/tmp/test")

    @patch("ryze.rl.lora_utils.PeftModel")
    @patch("ryze.rl.lora_utils.AutoModelForCausalLM")
    @patch("ryze.rl.lora_utils.AutoTokenizer")
    def test_merge_lora_to_base(self, mock_tok, mock_model_cls, mock_peft, tmp_path):
        from ryze.rl.lora_utils import LoRAManager

        mock_model = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model
        mock_tok.from_pretrained.return_value = MagicMock()

        mock_peft_model = MagicMock()
        mock_peft_model.merge_and_unload.return_value = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        output = str(tmp_path / "merged")
        result = LoRAManager.merge_lora_to_base("base", "adapter", output)
        assert result == output

    @patch("ryze.rl.lora_utils.LoRAManager.merge_lora_to_base")
    def test_sequential_merge(self, mock_merge, tmp_path):
        from ryze.rl.lora_utils import LoRAManager
        mock_merge.side_effect = lambda *args, **kw: args[2] if len(args) > 2 else kw.get("output_path", "")

        LoRAManager.sequential_merge("base_model", ["lora1", "lora2"], str(tmp_path))
        assert mock_merge.call_count == 2
