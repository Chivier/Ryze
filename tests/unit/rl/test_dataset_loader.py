"""Tests for ryze.rl.dataset_loader module."""

import json
from unittest.mock import MagicMock, patch

import torch


class TestSFTDataset:
    def test_loads_data(self, sample_sft_data):
        with patch("ryze.rl.dataset_loader.SFTDataset.__getitem__", return_value={"input_ids": torch.ones(10)}):
            from ryze.rl.dataset_loader import SFTDataset
            mock_tokenizer = MagicMock()
            mock_tokenizer.pad_token = "<pad>"
            mock_tokenizer.eos_token = "<eos>"
            ds = SFTDataset.__new__(SFTDataset)
            with open(sample_sft_data) as f:
                ds.data = json.load(f)
            ds.tokenizer = mock_tokenizer
            ds.max_length = 128
            assert len(ds) == 3

    def test_dataset_length(self, sample_sft_data):
        from ryze.rl.dataset_loader import SFTDataset
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        encoding = MagicMock()
        encoding.__getitem__ = lambda self, k: torch.ones(1, 10, dtype=torch.long)
        encoding.input_ids = torch.ones(1, 10, dtype=torch.long)
        encoding.attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = encoding
        ds = SFTDataset(sample_sft_data, mock_tokenizer, max_length=128)
        assert len(ds) == 3


class TestRLDataset:
    def test_loads_data(self, sample_rl_data):
        from ryze.rl.dataset_loader import RLDataset
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        encoding = MagicMock()
        encoding.__getitem__ = lambda self, k: torch.ones(1, 10, dtype=torch.long)
        encoding.input_ids = torch.ones(1, 10, dtype=torch.long)
        encoding.attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = encoding
        ds = RLDataset(sample_rl_data, mock_tokenizer, max_length=128)
        assert len(ds) == 2


class TestDatasetLoader:
    def test_load_sft_dataset(self, sample_sft_data):
        from ryze.rl.dataset_loader import DatasetLoader
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        encoding = MagicMock()
        encoding.__getitem__ = lambda self, k: torch.ones(1, 10, dtype=torch.long)
        encoding.input_ids = torch.ones(1, 10, dtype=torch.long)
        encoding.attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = encoding
        loaders = DatasetLoader.load_sft_dataset(sample_sft_data, None, mock_tokenizer, batch_size=2, num_workers=0)
        assert "train" in loaders
        assert "val" not in loaders

    def test_load_sft_with_val(self, sample_sft_data):
        from ryze.rl.dataset_loader import DatasetLoader
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        encoding = MagicMock()
        encoding.__getitem__ = lambda self, k: torch.ones(1, 10, dtype=torch.long)
        encoding.input_ids = torch.ones(1, 10, dtype=torch.long)
        encoding.attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = encoding
        loaders = DatasetLoader.load_sft_dataset(sample_sft_data, sample_sft_data, mock_tokenizer, batch_size=2, num_workers=0)
        assert "train" in loaders
        assert "val" in loaders

    def test_load_rl_dataset(self, sample_rl_data):
        from ryze.rl.dataset_loader import DatasetLoader
        mock_tokenizer = MagicMock()
        encoding = MagicMock()
        encoding.__getitem__ = lambda self, k: torch.ones(1, 10, dtype=torch.long)
        encoding.input_ids = torch.ones(1, 10, dtype=torch.long)
        encoding.attention_mask = torch.ones(1, 10, dtype=torch.long)
        mock_tokenizer.return_value = encoding
        loader = DatasetLoader.load_rl_dataset(sample_rl_data, mock_tokenizer, batch_size=2, num_workers=0)
        assert loader is not None
