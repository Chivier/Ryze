#!/usr/bin/env python3
"""
Smoke test: pull up full pipeline (Data → SFT LoRA → GRPO) with a tiny model.

Usage:
    python scripts/smoke_test_pipeline.py [--model MODEL_NAME]

Default model: HuggingFaceTB/SmolLM-135M (small Llama-arch, ~135M params)
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def step_data_processing(work_dir: str) -> dict:
    """Step 1: Generate SFT dataset from markdown files (skip OCR, use synthetic md)."""
    from ryze.data.dataset import SFTDatasetGenerator

    logger.info("=" * 60)
    logger.info("STEP 1: Data Processing (Markdown → SFT dataset)")
    logger.info("=" * 60)

    # Create synthetic markdown files
    md_dir = os.path.join(work_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)

    sample_texts = [
        (
            "## Introduction to Machine Learning\n\n"
            "Machine learning is a subset of artificial intelligence that focuses on building "
            "systems that learn from data. These systems improve their performance over time "
            "without being explicitly programmed. Supervised learning uses labeled data to train "
            "models. Unsupervised learning discovers hidden patterns in unlabeled data. "
            "Reinforcement learning trains agents through rewards and penalties. Deep learning "
            "uses neural networks with multiple layers to model complex patterns in data."
        ),
        (
            "## Natural Language Processing\n\n"
            "Natural language processing enables computers to understand human language. "
            "Tokenization splits text into meaningful units. Word embeddings represent words "
            "as dense vectors in a continuous space. Transformer models use self-attention "
            "mechanisms to capture long-range dependencies. Pre-trained language models like "
            "BERT and GPT have revolutionized NLP tasks including text classification, "
            "question answering, and machine translation."
        ),
        (
            "## Reinforcement Learning from Human Feedback\n\n"
            "RLHF is a technique used to align language models with human preferences. "
            "First a supervised fine-tuning step adapts the base model to follow instructions. "
            "Then a reward model is trained on human comparisons of model outputs. "
            "Finally policy optimization algorithms like PPO or GRPO refine the model. "
            "GRPO uses group-relative reward normalization to stabilize training. "
            "This approach has been key to creating helpful and harmless AI assistants."
        ),
    ]

    for i, text in enumerate(sample_texts):
        with open(os.path.join(md_dir, f"doc_{i}.md"), "w") as f:
            f.write(text)

    # Generate dataset
    generator = SFTDatasetGenerator({"min_text_length": 30, "max_text_length": 512})
    dataset_path = os.path.join(work_dir, "dataset", "sft.json")
    metadata = generator.create_dataset(md_dir, dataset_path)

    logger.info(f"  Train samples: {metadata['train_samples']}")
    logger.info(f"  Val samples:   {metadata['val_samples']}")
    logger.info(f"  Train path:    {metadata['train_path']}")

    # Also create RL dataset (prompts for GRPO)
    with open(metadata["train_path"], "r") as f:
        train_data = json.load(f)

    rl_data = []
    for item in train_data:
        rl_data.append({
            "instruction": item["instruction"],
            "input": item["input"],
            "output": item["output"],
            "prompt": f"{item['instruction']}\n\nInput: {item['input']}\n\nResponse:",
        })

    rl_data_path = os.path.join(work_dir, "dataset", "rl_data.json")
    with open(rl_data_path, "w") as f:
        json.dump(rl_data, f, indent=2)

    logger.info(f"  RL data path:  {rl_data_path} ({len(rl_data)} samples)")
    logger.info("STEP 1 DONE\n")

    return {
        "train_path": metadata["train_path"],
        "val_path": metadata["val_path"],
        "rl_data_path": rl_data_path,
    }


def step_sft_training(work_dir: str, train_path: str, val_path: str, model_name: str) -> dict:
    """Step 2: SFT LoRA training with a tiny model."""
    from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

    logger.info("=" * 60)
    logger.info("STEP 2: SFT LoRA Training")
    logger.info("=" * 60)

    sft_config = {
        "base_model_name": model_name,
        "batch_size": 2,
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "max_length": 256,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "output_dir": os.path.join(work_dir, "sft_output"),
        # LoRA
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],  # minimal for speed
        "auto_merge": True,
    }

    trainer = RyzeSFTLoRATrainer(sft_config)
    results = trainer.train(train_path, val_path)

    logger.info(f"  Training loss:      {results.get('training_loss', 'N/A')}")
    logger.info(f"  LoRA adapter path:  {results['lora_adapter_path']}")
    logger.info(f"  Merged model path:  {results['merged_model_path']}")
    logger.info("STEP 2 DONE\n")

    return results


def step_grpo_training(work_dir: str, sft_model_path: str, rl_data_path: str) -> dict:
    """Step 3: GRPO RL training on top of SFT model."""
    from ryze.rl.grpo_trainer import RyzeGRPOTrainer

    logger.info("=" * 60)
    logger.info("STEP 3: GRPO (RL) Training")
    logger.info("=" * 60)

    grpo_config = {
        "batch_size": 2,
        "micro_batch_size": 1,
        "learning_rate": 5e-5,
        "num_epochs": 1,
        "max_length": 256,
        "max_new_tokens": 32,
        # GRPO
        "num_samples_per_prompt": 2,
        "temperature": 0.8,
        "kl_coef": 0.1,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "grpo_epochs": 1,
        # LoRA
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "auto_merge": True,
        "output_dir": os.path.join(work_dir, "grpo_output"),
    }

    trainer = RyzeGRPOTrainer(grpo_config)
    results = trainer.train(sft_model_path, rl_data_path)

    final_stats = results.get("training_stats", [{}])[-1]
    logger.info(f"  Final loss:         {final_stats.get('loss', 'N/A')}")
    logger.info(f"  Mean reward:        {final_stats.get('mean_reward', 'N/A')}")
    logger.info(f"  LoRA adapter path:  {results['final_lora_path']}")
    logger.info(f"  Merged model path:  {results['merged_model_path']}")
    logger.info("STEP 3 DONE\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Ryze full pipeline smoke test")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="Base model for SFT (default: HuggingFaceTB/SmolLM-135M)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory (default: temp dir, auto-cleaned)",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Keep working directory after test",
    )
    args = parser.parse_args()

    # Setup working directory
    if args.work_dir:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp(prefix="ryze_smoke_")

    logger.info(f"Work directory: {work_dir}")
    logger.info(f"Model: {args.model}")
    logger.info("")

    try:
        # Step 1: Data processing
        data_result = step_data_processing(work_dir)

        # Step 2: SFT LoRA
        sft_result = step_sft_training(
            work_dir,
            data_result["train_path"],
            data_result["val_path"],
            args.model,
        )

        # Step 3: GRPO
        grpo_result = step_grpo_training(
            work_dir,
            sft_result["merged_model_path"],
            data_result["rl_data_path"],
        )

        # Summary
        logger.info("=" * 60)
        logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"  Data:  {data_result['train_path']}")
        logger.info(f"  SFT:   {sft_result['merged_model_path']}")
        logger.info(f"  GRPO:  {grpo_result['merged_model_path']}")
        logger.info(f"  Work:  {work_dir}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1

    finally:
        if not args.keep and not args.work_dir:
            logger.info(f"Cleaning up: {work_dir}")
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
