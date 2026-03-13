#!/usr/bin/env python3
"""Multi-GPU smoke test: SFT (4 GPUs DataParallel) + distributed GRPO (2+2 actor-learner).

Generates 30 synthetic markdown docs (~81 train samples) to ensure 10+ training
steps per stage.  SFT uses HF Trainer with all GPUs.  GRPO splits generation
and training onto separate GPU pools via Ray actors.

Usage:
    # Default: 4 GPUs, SFT on all 4, GRPO gen=2 + train=2
    python scripts/smoke_test_multi_gpu.py

    # Custom GPU layout
    python scripts/smoke_test_multi_gpu.py --num-gpus 4 --gen-gpus 2 --train-gpus 2

    # With existing Ray cluster
    python scripts/smoke_test_multi_gpu.py --ray-address auto

    # Keep outputs for inspection
    python scripts/smoke_test_multi_gpu.py --keep --work-dir ./smoke_outputs
"""

import argparse
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---------------------------------------------------------------------------
# 30 diverse AI/ML topic markdown documents (~500 chars each, ~3 QA pairs/doc)
# ---------------------------------------------------------------------------
TOPICS = [
    (
        "Introduction to Machine Learning",
        "Machine learning is a subset of artificial intelligence that focuses on building "
        "systems that learn from data. These systems improve their performance over time "
        "without being explicitly programmed. Supervised learning uses labeled data to train "
        "models. Unsupervised learning discovers hidden patterns in unlabeled data. "
        "Reinforcement learning trains agents through rewards and penalties. Deep learning "
        "uses neural networks with multiple layers to model complex patterns in data.",
    ),
    (
        "Natural Language Processing",
        "Natural language processing enables computers to understand human language. "
        "Tokenization splits text into meaningful units. Word embeddings represent words "
        "as dense vectors in a continuous space. Transformer models use self-attention "
        "mechanisms to capture long-range dependencies. Pre-trained language models like "
        "BERT and GPT have revolutionized NLP tasks including text classification, "
        "question answering, and machine translation.",
    ),
    (
        "Reinforcement Learning from Human Feedback",
        "RLHF is a technique used to align language models with human preferences. "
        "First a supervised fine-tuning step adapts the base model to follow instructions. "
        "Then a reward model is trained on human comparisons of model outputs. "
        "Finally policy optimization algorithms like PPO or GRPO refine the model. "
        "GRPO uses group-relative reward normalization to stabilize training. "
        "This approach has been key to creating helpful and harmless AI assistants.",
    ),
    (
        "Convolutional Neural Networks",
        "Convolutional neural networks are specialized for processing grid-like data such as "
        "images. They use convolutional layers to detect local features like edges and textures. "
        "Pooling layers reduce spatial dimensions while preserving important features. "
        "Modern architectures like ResNet use skip connections to train very deep networks. "
        "CNNs have achieved superhuman performance on image classification benchmarks.",
    ),
    (
        "Recurrent Neural Networks",
        "Recurrent neural networks are designed for sequential data processing. "
        "They maintain a hidden state that captures information from previous time steps. "
        "Long Short-Term Memory networks solve the vanishing gradient problem with gating "
        "mechanisms. GRU cells offer a simpler alternative with comparable performance. "
        "RNNs are widely used for time series forecasting and speech recognition tasks.",
    ),
    (
        "Generative Adversarial Networks",
        "Generative adversarial networks consist of two competing neural networks. "
        "The generator creates synthetic data samples from random noise. "
        "The discriminator learns to distinguish real data from generated samples. "
        "Training alternates between improving the generator and discriminator. "
        "GANs have produced remarkable results in image synthesis and style transfer.",
    ),
    (
        "Transfer Learning Techniques",
        "Transfer learning leverages knowledge from pre-trained models for new tasks. "
        "Fine-tuning adapts a pre-trained model by continuing training on task-specific data. "
        "Feature extraction uses the pre-trained model as a fixed feature extractor. "
        "Domain adaptation addresses the shift between source and target data distributions. "
        "Transfer learning significantly reduces the data and compute needed for new tasks.",
    ),
    (
        "Attention Mechanisms in Deep Learning",
        "Attention mechanisms allow models to focus on relevant parts of the input. "
        "Self-attention computes relationships between all positions in a sequence. "
        "Multi-head attention runs multiple attention operations in parallel. "
        "Cross-attention relates elements from different sequences or modalities. "
        "The transformer architecture relies entirely on attention without recurrence.",
    ),
    (
        "Optimization Algorithms",
        "Stochastic gradient descent is the foundation of neural network optimization. "
        "Adam combines momentum and adaptive learning rates for faster convergence. "
        "Learning rate scheduling adjusts the step size during training. "
        "Gradient clipping prevents exploding gradients in deep networks. "
        "Weight decay adds regularization to prevent overfitting in large models.",
    ),
    (
        "Regularization Methods",
        "Regularization prevents overfitting by constraining model complexity. "
        "Dropout randomly deactivates neurons during training to improve generalization. "
        "Batch normalization normalizes layer inputs to stabilize training. "
        "Data augmentation increases effective dataset size through transformations. "
        "Early stopping monitors validation performance to halt training at the right time.",
    ),
    (
        "Model Evaluation Metrics",
        "Accuracy measures the proportion of correct predictions for classification tasks. "
        "Precision and recall capture different aspects of classification performance. "
        "F1 score provides a harmonic mean of precision and recall. "
        "BLEU and ROUGE scores evaluate text generation quality against references. "
        "Perplexity measures how well a language model predicts a held-out test set.",
    ),
    (
        "Data Preprocessing Pipelines",
        "Data preprocessing transforms raw data into a format suitable for training. "
        "Normalization scales features to a standard range to improve convergence. "
        "Tokenization converts text into numerical representations for language models. "
        "Missing value imputation replaces absent data with statistical estimates. "
        "Feature engineering creates new informative features from existing data columns.",
    ),
    (
        "Distributed Training Systems",
        "Distributed training parallelizes computation across multiple devices. "
        "Data parallelism replicates the model and splits data across workers. "
        "Model parallelism distributes different model layers across devices. "
        "Pipeline parallelism combines both approaches for very large models. "
        "Gradient synchronization ensures consistent updates across all workers.",
    ),
    (
        "Knowledge Distillation",
        "Knowledge distillation transfers knowledge from a large teacher to a small student. "
        "The student learns from the teacher's soft probability distributions over classes. "
        "Temperature scaling controls the softness of the teacher's output distribution. "
        "Self-distillation uses the model itself as both teacher and student. "
        "Distillation enables deploying efficient models on resource-constrained devices.",
    ),
    (
        "Prompt Engineering Strategies",
        "Prompt engineering designs effective inputs for large language models. "
        "Few-shot prompting provides examples to guide the model's output format. "
        "Chain-of-thought prompting encourages step-by-step reasoning. "
        "System prompts define the model's role and behavioral constraints. "
        "Prompt templates standardize input formatting for consistent results.",
    ),
    (
        "Federated Learning",
        "Federated learning trains models across decentralized data sources. "
        "Clients train locally and share only model updates with the server. "
        "Federated averaging aggregates local model updates into a global model. "
        "Differential privacy adds noise to protect individual data contributions. "
        "Federated learning enables privacy-preserving collaborative machine learning.",
    ),
    (
        "Graph Neural Networks",
        "Graph neural networks process data structured as graphs with nodes and edges. "
        "Message passing aggregates information from neighboring nodes iteratively. "
        "Graph convolution generalizes the convolution operation to irregular structures. "
        "Graph attention networks use attention to weight neighbor contributions. "
        "GNNs are applied to social networks, molecules, and recommendation systems.",
    ),
    (
        "AutoML and Neural Architecture Search",
        "AutoML automates the process of designing machine learning pipelines. "
        "Neural architecture search discovers optimal network architectures automatically. "
        "Hyperparameter optimization tunes training parameters for best performance. "
        "Bayesian optimization efficiently explores the hyperparameter search space. "
        "AutoML democratizes machine learning by reducing the need for expert knowledge.",
    ),
    (
        "Variational Autoencoders",
        "Variational autoencoders learn latent representations of input data. "
        "The encoder maps inputs to a distribution in latent space. "
        "The decoder reconstructs inputs from sampled latent vectors. "
        "The ELBO loss balances reconstruction quality with latent space regularity. "
        "VAEs enable smooth interpolation and generation of new data samples.",
    ),
    (
        "Contrastive Learning",
        "Contrastive learning trains models to distinguish similar from dissimilar pairs. "
        "Positive pairs are created through data augmentation of the same example. "
        "Negative pairs come from different examples in the training batch. "
        "SimCLR and MoCo are popular frameworks for self-supervised visual learning. "
        "Contrastive learning produces powerful representations without labeled data.",
    ),
    (
        "Mixture of Experts Models",
        "Mixture of experts routes inputs to specialized sub-networks. "
        "A gating network decides which experts process each input token. "
        "Sparse activation enables scaling model capacity without proportional compute. "
        "Load balancing losses ensure all experts receive training signal. "
        "MoE architectures power many of the largest language models today.",
    ),
    (
        "Quantization for Model Compression",
        "Quantization reduces model size by lowering numerical precision. "
        "INT8 quantization halves model memory compared to FP16 with minimal quality loss. "
        "Post-training quantization requires no retraining of the original model. "
        "Quantization-aware training simulates low precision during the training process. "
        "Mixed-precision inference uses different precisions for different operations.",
    ),
    (
        "Diffusion Models for Generation",
        "Diffusion models generate data by reversing a gradual noising process. "
        "The forward process adds Gaussian noise to data over many timesteps. "
        "The reverse process learns to denoise step by step to create samples. "
        "Classifier-free guidance balances quality and diversity in generation. "
        "Diffusion models achieve state-of-the-art results in image synthesis.",
    ),
    (
        "Reward Modeling for RLHF",
        "Reward models score language model outputs based on human preferences. "
        "Training data consists of human comparisons between pairs of outputs. "
        "The Bradley-Terry model converts pairwise preferences to reward scores. "
        "Reward model overoptimization occurs when the policy exploits reward artifacts. "
        "Careful reward model design is critical for safe and helpful AI systems.",
    ),
    (
        "Parameter-Efficient Fine-Tuning",
        "Parameter-efficient fine-tuning adapts large models by updating few parameters. "
        "LoRA adds low-rank trainable matrices to frozen pre-trained weights. "
        "Prefix tuning prepends learnable tokens to the model input. "
        "Adapters insert small trainable modules between frozen transformer layers. "
        "PEFT methods make fine-tuning feasible on consumer hardware.",
    ),
    (
        "Tokenization and Vocabulary Design",
        "Byte-pair encoding builds a vocabulary by iteratively merging frequent pairs. "
        "SentencePiece provides a language-independent tokenization framework. "
        "WordPiece tokenization is used by BERT and similar masked language models. "
        "Vocabulary size affects model capacity and computational efficiency. "
        "Multilingual tokenizers balance coverage across diverse writing systems.",
    ),
    (
        "Embedding Spaces and Representations",
        "Word embeddings map discrete tokens to continuous vector spaces. "
        "Sentence embeddings capture the meaning of variable-length text sequences. "
        "Contrastive pre-training aligns embeddings of semantically similar items. "
        "Embedding dimension affects the capacity and computational cost of models. "
        "Nearest neighbor search in embedding space enables semantic retrieval.",
    ),
    (
        "Curriculum Learning Strategies",
        "Curriculum learning presents training examples in order of increasing difficulty. "
        "Easy examples help the model learn basic patterns before encountering hard cases. "
        "Self-paced learning lets the model select which examples to learn from next. "
        "Anti-curriculum approaches train on hard examples first for faster convergence. "
        "Curriculum strategies can improve both training speed and final model quality.",
    ),
    (
        "Multi-Task Learning",
        "Multi-task learning trains a single model on multiple related tasks simultaneously. "
        "Shared representations capture common features across tasks. "
        "Task-specific heads produce outputs tailored to each task. "
        "Auxiliary tasks can provide regularization and improve main task performance. "
        "Multi-task learning is efficient and often improves generalization.",
    ),
    (
        "Safety and Alignment Research",
        "AI safety research aims to ensure AI systems behave as intended. "
        "Constitutional AI uses principles to guide model self-improvement. "
        "Red teaming probes models for harmful or unexpected behaviors. "
        "Interpretability techniques help understand model decision processes. "
        "Alignment research is essential for developing trustworthy AI systems.",
    ),
]


def create_sample_markdown(work_dir: str) -> int:
    """Create 30 synthetic markdown files for the smoke test.

    Returns:
        Number of files created.
    """
    md_dir = os.path.join(work_dir, "markdown")
    os.makedirs(md_dir, exist_ok=True)
    for i, (title, body) in enumerate(TOPICS):
        with open(os.path.join(md_dir, f"doc_{i:02d}.md"), "w") as f:
            f.write(f"## {title}\n\n{body}\n")
    logger.info("Created %d sample markdown files in %s", len(TOPICS), md_dir)
    return len(TOPICS)


def build_multi_gpu_pipeline(
    work_dir: str,
    model_name: str,
    num_gpus: int,
    gen_gpus: int,
    train_gpus: int,
    max_steps: int,
):
    """Build the Data -> SFT (multi-GPU) -> GRPO (distributed) pipeline.

    Args:
        work_dir: Working directory for intermediate outputs.
        model_name: HuggingFace model name.
        num_gpus: Total GPUs for SFT stage.
        gen_gpus: GPUs for GRPO generation actor.
        train_gpus: GPUs for GRPO training actor.
        max_steps: Minimum training steps target for SFT.

    Returns:
        Configured PipelineOrchestrator.
    """
    from ryze.core.pipeline import PipelineOrchestrator
    from ryze.data.dataset import SFTDatasetGenerator
    from ryze.rl.grpo_trainer import RyzeGRPOTrainer
    from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

    pipeline = PipelineOrchestrator()

    # Stage 1: Data Processing (CPU)
    data_task = SFTDatasetGenerator({"min_text_length": 30, "max_text_length": 512}).as_task(
        markdown_dir=os.path.join(work_dir, "markdown"),
        output_path=os.path.join(work_dir, "dataset", "sft.json"),
    )

    # Stage 2: SFT LoRA Training (all GPUs via HF Trainer DataParallel)
    sft_config = {
        "base_model_name": model_name,
        "batch_size": 2,
        "learning_rate": 3e-4,
        "num_epochs": 1,
        "max_length": 256,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "max_steps": max_steps,
        "num_gpus": num_gpus,
        "output_dir": os.path.join(work_dir, "sft_output"),
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj", "v_proj"],
        "auto_merge": True,
    }
    sft_task = RyzeSFTLoRATrainer(sft_config).as_task()

    # Stage 3: GRPO Training (distributed actor-learner via Ray)
    grpo_config = {
        "batch_size": 2,
        "micro_batch_size": 1,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "max_length": 256,
        "max_new_tokens": 32,
        "num_samples_per_prompt": 4,
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
        # Distributed GRPO settings
        "distributed_grpo": True,
        "gen_gpu_count": gen_gpus,
        "train_gpu_count": train_gpus,
    }
    grpo_task = RyzeGRPOTrainer(grpo_config).as_task()

    data_id = pipeline.add_task(data_task)
    sft_id = pipeline.add_task(sft_task, depends_on=[data_id])
    pipeline.add_task(grpo_task, depends_on=[sft_id, data_id])

    return pipeline


def print_summary(results: dict, min_steps: int) -> bool:
    """Print pipeline summary and verify step counts.

    Returns:
        True if all verification checks pass.
    """
    logger.info("=" * 60)
    succeeded = sum(1 for r in results.values() if r.status.value == "completed")
    failed = sum(1 for r in results.values() if r.status.value == "failed")
    logger.info("Pipeline complete: %d succeeded, %d failed", succeeded, failed)

    all_ok = True
    for task_id, result in results.items():
        logger.info("  %s: %s", task_id, result.status.value)
        if result.output:
            # SFT step count verification
            if "training_steps" in result.output:
                steps = result.output["training_steps"]
                logger.info("    SFT training_steps: %d", steps)
                if steps < min_steps:
                    logger.error("    FAIL: SFT steps %d < min %d", steps, min_steps)
                    all_ok = False
                else:
                    logger.info("    OK: SFT steps >= %d", min_steps)
                # Show GPU info if available
                gpu_info = result.output.get("gpu_info")
                if gpu_info:
                    logger.info(
                        "    [GPU-TRACK] SFT ran on %d GPUs "
                        "(CUDA_VISIBLE_DEVICES=%s, pid=%s)",
                        gpu_info.get("n_gpu", "?"),
                        gpu_info.get("cuda_visible_devices", "?"),
                        gpu_info.get("pid", "?"),
                    )

            # GRPO batch step verification
            if "total_batch_steps" in result.output:
                steps = result.output["total_batch_steps"]
                logger.info("    GRPO total_batch_steps: %d", steps)
                if steps < min_steps:
                    logger.error("    FAIL: GRPO steps %d < min %d", steps, min_steps)
                    all_ok = False
                else:
                    logger.info("    OK: GRPO steps >= %d", min_steps)
                # Show GPU info for both actors
                grpo_gpu_info = result.output.get("gpu_info", {})
                for actor_key in ("gen_actor", "train_actor"):
                    info = grpo_gpu_info.get(actor_key, {})
                    if info:
                        logger.info(
                            "    [GPU-TRACK] %s (pid=%s) — "
                            "CUDA_VISIBLE_DEVICES=%s, gpu_count=%s",
                            info.get("actor", actor_key),
                            info.get("pid", "?"),
                            info.get("cuda_visible_devices", "?"),
                            info.get("gpu_count", "?"),
                        )
                # Verify GPU separation
                gen_info = grpo_gpu_info.get("gen_actor", {})
                train_info = grpo_gpu_info.get("train_actor", {})
                if gen_info and train_info:
                    gen_set = set(gen_info.get("cuda_visible_devices", "").split(","))
                    train_set = set(train_info.get("cuda_visible_devices", "").split(","))
                    overlap = gen_set & train_set - {""}
                    if overlap:
                        logger.error(
                            "    FAIL: gen and train actors share GPUs: %s",
                            overlap,
                        )
                        all_ok = False
                    else:
                        logger.info(
                            "    OK: gen and train actors on SEPARATE GPUs "
                            "(gen=%s, train=%s)",
                            gen_info.get("cuda_visible_devices"),
                            train_info.get("cuda_visible_devices"),
                        )

            for key in (
                "train_path",
                "merged_model_path",
                "final_lora_path",
                "grpo_data_path",
            ):
                if key in result.output:
                    logger.info("    %s: %s", key, result.output[key])

        if result.status.value != "completed":
            all_ok = False

    logger.info("=" * 60)
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Ryze multi-GPU distributed smoke test")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM-135M",
        help="Base model (default: HuggingFaceTB/SmolLM-135M)",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Working directory (default: temp dir, auto-cleaned)",
    )
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray cluster address (default: start local Ray)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=4,
        help="Total GPUs for SFT (default: 4)",
    )
    parser.add_argument(
        "--gen-gpus",
        type=int,
        default=2,
        help="GPUs for GRPO generation actor (default: 2)",
    )
    parser.add_argument(
        "--train-gpus",
        type=int,
        default=2,
        help="GPUs for GRPO training actor (default: 2)",
    )
    parser.add_argument(
        "--min-steps",
        type=int,
        default=10,
        help="Minimum training steps to verify per stage (default: 10)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=12,
        help="SFT max_steps (default: 12)",
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
        work_dir = tempfile.mkdtemp(prefix="ryze_multigpu_smoke_")

    logger.info("Work directory: %s", work_dir)
    logger.info("Model: %s", args.model)
    logger.info(
        "GPU layout: SFT=%d, GRPO gen=%d + train=%d",
        args.num_gpus,
        args.gen_gpus,
        args.train_gpus,
    )
    logger.info("Min steps target: %d", args.min_steps)
    logger.info("")

    try:
        # 0. Log system GPU info
        import torch

        logger.info("=" * 60)
        logger.info("[GPU-TRACK] System GPU inventory:")
        for i in range(torch.cuda.device_count()):
            logger.info(
                "[GPU-TRACK]   GPU %d: %s (%.1f GB)",
                i,
                torch.cuda.get_device_name(i),
                torch.cuda.get_device_properties(i).total_memory / 1e9,
            )
        logger.info("=" * 60)

        # 1. Create 30 synthetic markdown docs
        num_docs = create_sample_markdown(work_dir)
        logger.info(
            "Expected ~%d train samples (30 docs x ~3 QA pairs x 0.9 split)",
            int(num_docs * 3 * 0.9),
        )

        # 2. Build pipeline
        logger.info("")
        logger.info("=" * 60)
        logger.info("[STAGE] Building pipeline: Data -> SFT (%d GPUs) -> GRPO (gen=%d + train=%d GPUs)",
                     args.num_gpus, args.gen_gpus, args.train_gpus)
        logger.info("=" * 60)
        pipeline = build_multi_gpu_pipeline(
            work_dir=work_dir,
            model_name=args.model,
            num_gpus=args.num_gpus,
            gen_gpus=args.gen_gpus,
            train_gpus=args.train_gpus,
            max_steps=args.max_steps,
        )

        # 3. Run with Ray distributed runner
        from ryze.cluster.ray_manager import RayManager
        from ryze.core.runner import DistributedRunner

        ray_manager = RayManager(address=args.ray_address)
        runner = DistributedRunner(ray_manager=ray_manager, timeout_s=1800)

        # 4. Execute
        logger.info("")
        logger.info("=" * 60)
        logger.info("[STAGE] Executing pipeline via DistributedRunner + Ray")
        logger.info("=" * 60)
        results = pipeline.run(runner=runner, fail_fast=True)

        # 5. Verify
        all_ok = print_summary(results, args.min_steps)
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
