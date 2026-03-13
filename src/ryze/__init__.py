# Ryze: LLM fine-tuning pipeline with cluster scheduling support
"""
Ryze Pipeline Components:
1. Data Module: PDF to Markdown OCR and SFT dataset generation
2. RL Module: SFT training and RL training pipeline
   - v1: Traditional SFT + PPO
   - v2: LoRA-based SFT + GRPO with stage-wise merging
3. Eval Module: Model evaluation framework
4. Core Module: Task abstraction and pipeline orchestration
5. Cluster Module: SwarmPilot/PyLet distributed execution
"""

__version__ = "0.3.0"

from .config import RyzeConfig
from .data import RyzeDataProcessor
from .eval import RyzeEvaluator
from .exceptions import RyzeError
from .rl import (
    LoRAManager,
    RyzeGRPOTrainer,
    RyzeRLTrainer,
    RyzeSFTLoRATrainer,
    RyzeSFTTrainer,
)

__all__ = [
    "RyzeDataProcessor",
    "RyzeSFTTrainer",
    "RyzeRLTrainer",
    "RyzeSFTLoRATrainer",
    "RyzeGRPOTrainer",
    "LoRAManager",
    "RyzeEvaluator",
    "RyzeConfig",
    "RyzeError",
]
