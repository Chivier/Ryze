"""Dataset Loader for Training"""
import json
import logging
from typing import Dict, Optional

from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class SFTDataset(Dataset):
    """Dataset for Supervised Fine-Tuning"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples from {data_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Format the prompt
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')

        # Create full prompt
        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"{instruction}\n\nResponse:"

        full_text = f"{prompt} {output}"

        # Tokenize
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        # Create labels (mask the prompt part)
        labels = encoding['input_ids'].clone()
        prompt_encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        prompt_length = prompt_encoding['input_ids'].shape[1]
        labels[0, :prompt_length] = -100  # Mask prompt tokens

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }


class RLDataset(Dataset):
    """Dataset for Reinforcement Learning"""

    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        logger.info(f"Loaded {len(self.data)} samples for RL training")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # For RL, we typically just need the prompt
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')

        if input_text:
            prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
        else:
            prompt = f"{instruction}\n\nResponse:"

        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'prompt': prompt,
            'reference_output': item.get('output', '')
        }


class DatasetLoader:
    """Utility class to load datasets for training"""

    @staticmethod
    def load_sft_dataset(
        train_path: str,
        val_path: Optional[str],
        tokenizer,
        batch_size: int = 8,
        max_length: int = 2048,
        num_workers: int = 4
    ) -> Dict[str, DataLoader]:
        """Load SFT training datasets"""
        train_dataset = SFTDataset(train_path, tokenizer, max_length)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )

        loaders = {'train': train_loader}

        if val_path:
            val_dataset = SFTDataset(val_path, tokenizer, max_length)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
            loaders['val'] = val_loader

        return loaders

    @staticmethod
    def load_rl_dataset(
        data_path: str,
        tokenizer,
        batch_size: int = 4,
        max_length: int = 1024,
        num_workers: int = 4
    ) -> DataLoader:
        """Load RL training dataset"""
        dataset = RLDataset(data_path, tokenizer, max_length)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers
        )
