"""Shared test fixtures for Ryze test suite."""

import json
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory."""
    return tmp_path


@pytest.fixture
def sample_config():
    """Minimal Ryze config dict."""
    return {
        "data_processing": {
            "ocr": {"language": "eng", "dpi": 150, "use_gpu": False},
            "dataset": {
                "min_text_length": 10,
                "max_text_length": 512,
                "instruction_templates": ["Summarize:", "Extract key points:"],
            },
            "output_base": "./test_output",
        },
        "training": {
            "sft": {
                "base_model_name": "test-model",
                "batch_size": 2,
                "learning_rate": 1e-4,
                "num_epochs": 1,
                "max_length": 128,
                "gradient_accumulation_steps": 1,
                "warmup_ratio": 0.01,
                "weight_decay": 0.0,
                "output_dir": "./test_sft_output",
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": None, "use_4bit": False, "use_8bit": False},
                "auto_merge": False,
            },
            "rl": {
                "batch_size": 2,
                "micro_batch_size": 1,
                "learning_rate": 1e-5,
                "num_epochs": 1,
                "max_length": 128,
                "max_new_tokens": 32,
                "output_dir": "./test_grpo_output",
                "grpo": {"num_samples_per_prompt": 2, "temperature": 0.7, "kl_coef": 0.1, "clip_range": 0.2, "value_clip_range": 0.2, "grpo_epochs": 1},
                "lora": {"r": 4, "alpha": 8, "dropout": 0.0, "target_modules": None, "use_4bit": False, "use_8bit": False},
                "auto_merge": False,
            },
        },
        "evaluation": {
            "max_new_tokens": 32,
            "temperature": 0.5,
            "do_sample": False,
            "top_p": 0.9,
            "output_dir": "./test_eval_output",
            "benchmarks_dir": "./test_benchmarks",
        },
        "ui": {"title": "Test", "theme": "default", "server": {"host": "127.0.0.1", "port": 7860, "share": False, "debug": False}},
        "cluster": {"mode": "local", "ray_address": "auto", "ray_dashboard_url": "http://localhost:8265", "timeout_s": 10, "max_retries": 1},
    }


@pytest.fixture
def sample_sft_data(tmp_path):
    """Create a small SFT training dataset."""
    data = [
        {"instruction": "Summarize:", "input": "The quick brown fox jumps over the lazy dog.", "output": "A fox jumps over a dog."},
        {"instruction": "Key points:", "input": "Machine learning uses data to learn patterns.", "output": "ML learns from data."},
        {"instruction": "Topic:", "input": "Climate change affects global temperatures.", "output": "Climate change."},
    ]
    path = tmp_path / "train.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_rl_data(tmp_path):
    """Create a small RL training dataset."""
    data = [
        {"instruction": "Summarize:", "input": "Test input one.", "output": "Test output one.", "prompt": "Summarize:\n\nInput: Test input one.\n\nResponse:"},
        {"instruction": "Explain:", "input": "Test input two.", "output": "Test output two.", "prompt": "Explain:\n\nInput: Test input two.\n\nResponse:"},
    ]
    path = tmp_path / "rl_data.json"
    path.write_text(json.dumps(data))
    return str(path)


@pytest.fixture
def sample_markdown_dir(tmp_path):
    """Create a directory with sample markdown files."""
    md_dir = tmp_path / "markdown"
    md_dir.mkdir()
    (md_dir / "doc1.md").write_text("## Title\n\nThis is a sample document with enough text to be processed. It contains multiple sentences for testing purposes. The quick brown fox jumps over the lazy dog repeatedly.")
    (md_dir / "doc2.md").write_text("### Section\n\nAnother document for testing. Machine learning is a subset of AI. It uses algorithms and statistical models.")
    return str(md_dir)


@pytest.fixture
def sample_benchmark(tmp_path):
    """Create a sample benchmark file."""
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    data = {
        "test": [
            {"id": "t1", "instruction": "Summarize:", "input": "AI is intelligence by machines.", "expected_output": "AI is machine intelligence.", "category": "summary"},
            {"id": "t2", "instruction": "Topic:", "input": "Climate change affects weather.", "expected_output": "Climate change.", "category": "topic"},
        ],
        "validation": [
            {"id": "v1", "instruction": "Overview:", "input": "Quantum computing uses qubits.", "expected_output": "Quantum uses qubits.", "category": "overview"},
        ],
    }
    path = benchmarks_dir / "test_benchmark.json"
    path.write_text(json.dumps(data))
    return str(benchmarks_dir), "test_benchmark"


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<eos>"
    tokenizer.pad_token_id = 0

    def mock_call(text, **kwargs):
        result = MagicMock()
        import torch
        length = min(kwargs.get("max_length", 128), 20)
        result.__getitem__ = lambda self, key: torch.ones(1, length, dtype=torch.long)
        result.input_ids = torch.ones(1, length, dtype=torch.long)
        result.attention_mask = torch.ones(1, length, dtype=torch.long)
        result.to = lambda device: result
        result.keys = lambda: ["input_ids", "attention_mask"]
        result.items = lambda: [("input_ids", result.input_ids), ("attention_mask", result.attention_mask)]
        return result

    tokenizer.side_effect = mock_call
    tokenizer.__call__ = mock_call
    tokenizer.decode = MagicMock(return_value="Generated text response")
    return tokenizer


@pytest.fixture
def mock_ray():
    """Create a mock Ray module for testing RayManager."""
    ray = MagicMock()
    ray.is_initialized.return_value = False
    ray.init.return_value = None
    ray.available_resources.return_value = {"GPU": 4, "CPU": 16, "memory": 68719476736}
    ray.cluster_resources.return_value = {"GPU": 4, "CPU": 16, "memory": 68719476736}
    ray.nodes.return_value = [
        {"NodeID": "node1", "Alive": True, "Resources": {"GPU": 2}},
        {"NodeID": "node2", "Alive": True, "Resources": {"GPU": 2}},
    ]
    return ray


@pytest.fixture
def mock_ray_job_client():
    """Create a mock Ray Job Submission client."""
    client = MagicMock()
    client.submit_job.return_value = "raysubmit_test123"

    # Mock job info objects
    job_info = MagicMock()
    job_info.submission_id = "raysubmit_test123"
    job_info.job_id = "raysubmit_test123"
    job_info.status = MagicMock()
    job_info.status.value = "RUNNING"
    job_info.entrypoint = "python train.py"

    client.list_jobs.return_value = [job_info]
    client.stop_job.return_value = True
    return client
