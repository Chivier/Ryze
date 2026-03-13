#!/usr/bin/env python3
"""Smoke test: full pipeline (Data -> SFT LoRA -> GRPO) via Task/Pipeline abstraction.

Usage:
    # Local mode (default)
    python scripts/smoke_test_pipeline.py [--model MODEL_NAME]

    # Distributed mode (requires Ray cluster)
    python scripts/smoke_test_pipeline.py --mode ray [--model MODEL_NAME]

Default model: HuggingFaceTB/SmolLM-135M (small Llama-arch, ~135M params)
"""

import argparse
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


SAMPLE_TEXTS = [
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


def create_sample_markdown(work_dir: str) -> None:
    """Create synthetic markdown files for smoke test."""
    md_dir = os.path.join(work_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)
    for i, text in enumerate(SAMPLE_TEXTS):
        with open(os.path.join(md_dir, f"doc_{i}.md"), "w") as f:
            f.write(text)
    logger.info("Created %d sample markdown files in %s", len(SAMPLE_TEXTS), md_dir)


def build_smoke_pipeline(work_dir: str, model_name: str):
    """Build three-stage smoke test pipeline using Task abstraction.

    Args:
        work_dir: Working directory for intermediate outputs.
        model_name: HuggingFace model name for training.

    Returns:
        Configured PipelineOrchestrator with Data -> SFT -> GRPO stages.
    """
    from ryze.core.pipeline import PipelineOrchestrator
    from ryze.data.dataset import SFTDatasetGenerator
    from ryze.rl.grpo_trainer import RyzeGRPOTrainer
    from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

    pipeline = PipelineOrchestrator()

    # Stage 1: Data Processing (CPU, local)
    data_task = SFTDatasetGenerator(
        {"min_text_length": 30, "max_text_length": 512}
    ).as_task(
        markdown_dir=os.path.join(work_dir, "markdown"),
        output_path=os.path.join(work_dir, "dataset", "sft.json"),
    )

    # Stage 2: SFT LoRA Training (GPU, remote in ray mode)
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
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "auto_merge": True,
    }
    sft_task = RyzeSFTLoRATrainer(sft_config).as_task()

    # Stage 3: GRPO Training (GPU, remote in ray mode)
    grpo_config = {
        "batch_size": 2,
        "micro_batch_size": 1,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "max_length": 256,
        "max_new_tokens": 32,
        "num_samples_per_prompt": 2,
        "temperature": 0.8,
        "kl_coef": 0.1,
        "clip_range": 0.2,
        "value_clip_range": 0.2,
        "grpo_epochs": 2,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "auto_merge": True,
        "output_dir": os.path.join(work_dir, "grpo_output"),
    }
    grpo_task = RyzeGRPOTrainer(grpo_config).as_task()

    data_id = pipeline.add_task(data_task)
    sft_id = pipeline.add_task(sft_task, depends_on=[data_id])
    pipeline.add_task(grpo_task, depends_on=[sft_id])

    return pipeline


def print_summary(results: dict) -> None:
    """Print pipeline execution summary."""
    logger.info("=" * 60)
    succeeded = sum(1 for r in results.values() if r.status.value == "completed")
    failed = sum(1 for r in results.values() if r.status.value == "failed")
    logger.info("Pipeline complete: %d succeeded, %d failed", succeeded, failed)
    for task_id, result in results.items():
        logger.info("  %s: %s", task_id, result.status.value)
        if result.output:
            for key in ("train_path", "merged_model_path", "final_lora_path", "grpo_data_path"):
                if key in result.output:
                    logger.info("    %s: %s", key, result.output[key])
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Ryze full pipeline smoke test")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="Base model for SFT (default: HuggingFaceTB/SmolLM-135M)",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "ray"],
        default="local",
        help="Execution mode: local (default) or ray (distributed)",
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

    if args.work_dir:
        work_dir = args.work_dir
        os.makedirs(work_dir, exist_ok=True)
    else:
        work_dir = tempfile.mkdtemp(prefix="ryze_smoke_")

    logger.info("Work directory: %s", work_dir)
    logger.info("Model: %s", args.model)
    logger.info("Mode: %s", args.mode)
    logger.info("")

    try:
        # 1. Create synthetic markdown
        create_sample_markdown(work_dir)

        # 2. Build pipeline
        pipeline = build_smoke_pipeline(work_dir, args.model)

        # 3. Select runner
        if args.mode == "ray":
            from ryze.cluster.ray_manager import RayManager
            from ryze.core.runner import DistributedRunner

            ray_manager = RayManager(address="auto")
            runner = DistributedRunner(ray_manager=ray_manager)
        else:
            from ryze.core.runner import LocalRunner

            runner = LocalRunner()

        # 4. Execute pipeline
        results = pipeline.run(runner=runner, fail_fast=True)

        # 5. Summary
        print_summary(results)

        all_ok = all(r.status.value in ("completed",) for r in results.values())
        return 0 if all_ok else 1

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        return 1

    finally:
        if not args.keep and not args.work_dir:
            logger.info("Cleaning up: %s", work_dir)
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
