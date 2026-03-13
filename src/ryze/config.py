"""Ryze configuration with Pydantic models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .exceptions import ConfigError


class OCRConfig(BaseModel):
    language: str = "eng+chi_sim"
    dpi: int = 300
    use_gpu: bool = False


class DatasetConfig(BaseModel):
    min_text_length: int = 50
    max_text_length: int = 2048
    instruction_templates: list[str] = Field(default_factory=lambda: [
        "Please summarize the following text:",
        "Extract the key points from this document:",
        "What is the main topic of this text?",
        "Identify the important information in this passage:",
        "Provide a brief overview of the following content:",
    ])


class DataProcessingConfig(BaseModel):
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    output_base: str = "./output"


class LoRAConfig(BaseModel):
    r: int = 16
    alpha: int = 32
    dropout: float = 0.1
    target_modules: Optional[list[str]] = None
    use_4bit: bool = False
    use_8bit: bool = False


class SFTConfig(BaseModel):
    base_model_name: str = "microsoft/phi-2"
    batch_size: int = 8
    learning_rate: float = 3e-4
    num_epochs: int = 3
    max_length: int = 2048
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    output_dir: str = "./sft_lora_outputs"
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    auto_merge: bool = True


class GRPOConfig(BaseModel):
    num_samples_per_prompt: int = 4
    temperature: float = 0.8
    kl_coef: float = 0.1
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    grpo_epochs: int = 4


class RLConfig(BaseModel):
    batch_size: int = 4
    micro_batch_size: int = 1
    learning_rate: float = 5e-5
    num_epochs: int = 3
    max_length: int = 1024
    max_new_tokens: int = 256
    output_dir: str = "./grpo_outputs"
    grpo: GRPOConfig = Field(default_factory=GRPOConfig)
    lora: LoRAConfig = Field(default_factory=lambda: LoRAConfig(r=8, alpha=16))
    auto_merge: bool = True


class TrainingConfig(BaseModel):
    sft: SFTConfig = Field(default_factory=SFTConfig)
    rl: RLConfig = Field(default_factory=RLConfig)


class EvaluationConfig(BaseModel):
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = False
    top_p: float = 0.95
    output_dir: str = "./eval_outputs"
    benchmarks_dir: str = "./benchmarks"


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 7860
    share: bool = False
    debug: bool = False


class UIConfig(BaseModel):
    title: str = "Ryze-ACL Pipeline v2"
    theme: str = "default"
    server: ServerConfig = Field(default_factory=ServerConfig)


class ClusterConfig(BaseModel):
    mode: str = "local"  # "local" or "ray"
    ray_address: str = "auto"
    ray_dashboard_url: str = "http://localhost:8265"
    timeout_s: int = 300
    max_retries: int = 3


class RyzeConfig(BaseModel):
    data_processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    cluster: ClusterConfig = Field(default_factory=ClusterConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> RyzeConfig:
        """Load configuration from JSON file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    @classmethod
    def from_legacy_json(cls, path: str | Path) -> RyzeConfig:
        """Load from legacy JSON config, migrating old fields."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Configuration file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate cluster config
        if "cluster" not in data:
            data["cluster"] = ClusterConfig().model_dump()
        else:
            cluster = data["cluster"]
            if "ray_address" not in cluster:
                cluster["ray_address"] = "auto"
            if "ray_dashboard_url" not in cluster:
                cluster["ray_dashboard_url"] = "http://localhost:8265"
        return cls.model_validate(data)

    def to_legacy_dict(self) -> dict[str, Any]:
        """Export as legacy dict format for backward compatibility."""
        return self.model_dump()
