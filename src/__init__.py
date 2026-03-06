# Ryze-ACL: A clean pipeline for PDF OCR, SFT, RL training and evaluation
"""
Ryze-ACL Pipeline Components:
1. Ryze Data Module: PDF to Markdown OCR and SFT dataset generation
2. Ryze RL Module: SFT training and RL training pipeline
   - v1: Traditional SFT + PPO
   - v2: LoRA-based SFT + GRPO with stage-wise merging
3. Ryze Eval Module: Model evaluation framework
"""

__version__ = "0.2.0"

from .data import RyzeDataProcessor
from .rl import (
    RyzeSFTTrainer,
    RyzeRLTrainer,
    RyzeSFTLoRATrainer,
    RyzeGRPOTrainer,
    LoRAManager
)
from .eval import RyzeEvaluator

__all__ = [
    "RyzeDataProcessor",
    "RyzeSFTTrainer",
    "RyzeRLTrainer",
    "RyzeSFTLoRATrainer",
    "RyzeGRPOTrainer",
    "LoRAManager",
    "RyzeEvaluator"
]