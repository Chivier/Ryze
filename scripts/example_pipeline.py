#!/usr/bin/env python3
"""
Example script demonstrating the complete Ryze-ACL pipeline
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import RyzeDataProcessor
from rl import RyzeSFTTrainer, RyzeRLTrainer
from eval import RyzeEvaluator


def run_complete_pipeline():
    """Run the complete pipeline from PDF to evaluation"""

    print("=" * 60)
    print("Ryze-ACL Pipeline Example")
    print("=" * 60)

    # Step 1: Data Processing
    print("\n📄 Step 1: Processing PDF files...")

    # Configuration for data processing
    data_config = {
        'ocr': {
            'language': 'eng',
            'dpi': 300
        },
        'output_base': './example_output'
    }

    # Initialize data processor
    processor = RyzeDataProcessor(data_config)

    # Process a sample PDF (you need to provide a PDF file)
    # For demo, we'll skip this if no PDF is provided
    pdf_path = "./sample.pdf"  # Replace with your PDF

    if os.path.exists(pdf_path):
        result = processor.process_single_pdf(pdf_path)
        print(f"✅ PDF processed: {result['output_dir']}")

        # Get the generated dataset paths
        train_data_path = result['dataset_result']['train_path']
        val_data_path = result['dataset_result']['val_path']
    else:
        print("⚠️  No PDF file found. Creating sample dataset...")
        # Create a sample dataset for demo
        sample_data = [
            {
                "instruction": "Summarize the following text:",
                "input": "Artificial intelligence is transforming the world...",
                "output": "AI is revolutionizing various industries."
            }
        ]

        os.makedirs('./example_output/dataset', exist_ok=True)
        train_data_path = './example_output/dataset/sample_train.json'
        val_data_path = './example_output/dataset/sample_val.json'

        with open(train_data_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        with open(val_data_path, 'w') as f:
            json.dump(sample_data, f, indent=2)

    # Step 2: SFT Training
    print("\n🎯 Step 2: SFT Training...")

    sft_config = {
        'model_name': 'microsoft/phi-2',  # Using a small model for demo
        'num_epochs': 1,  # Quick demo
        'batch_size': 4,
        'learning_rate': 5e-5,
        'output_dir': './example_output/sft'
    }

    sft_trainer = RyzeSFTTrainer(sft_config)

    # Train the model
    print("Training SFT model...")
    sft_results = sft_trainer.train(train_data_path, val_data_path)
    sft_model_path = sft_results['model_save_path']
    print(f"✅ SFT training completed: {sft_model_path}")

    # Step 3: Generate RL Dataset
    print("\n🔄 Step 3: Generating RL dataset...")

    rl_dataset_path = './example_output/rl_dataset/rl_data.json'
    rl_gen_results = sft_trainer.generate_rl_dataset(
        sft_model_path,
        train_data_path,
        rl_dataset_path
    )
    print(f"✅ RL dataset generated: {rl_gen_results['output_path']}")

    # Step 4: RL Training
    print("\n🚀 Step 4: RL Training...")

    rl_config = {
        'num_epochs': 1,  # Quick demo
        'batch_size': 2,
        'learning_rate': 1e-5,
        'ppo_epochs': 2,
        'output_dir': './example_output/rl'
    }

    rl_trainer = RyzeRLTrainer(rl_config)

    print("Training with RL...")
    rl_results = rl_trainer.train(sft_model_path, rl_dataset_path)
    rl_model_path = rl_results['final_model_path']
    print(f"✅ RL training completed: {rl_model_path}")

    # Step 5: Evaluation
    print("\n📊 Step 5: Model Evaluation...")

    eval_config = {
        'temperature': 0.7,
        'do_sample': True,
        'max_new_tokens': 256,
        'output_dir': './example_output/eval'
    }

    evaluator = RyzeEvaluator(eval_config)

    # Evaluate both SFT and RL models
    print("Evaluating SFT model...")
    sft_eval_results = evaluator.evaluate_model(sft_model_path, 'general_qa')

    print("Evaluating RL model...")
    rl_eval_results = evaluator.evaluate_model(rl_model_path, 'general_qa')

    # Print comparison
    print("\n📈 Results Comparison:")
    print("-" * 40)

    for model_type, results in [("SFT", sft_eval_results), ("RL", rl_eval_results)]:
        test_metrics = results['results_by_split']['test']['metrics']
        print(f"\n{model_type} Model:")
        print(f"  BLEU Score: {test_metrics.get('bleu', 0):.4f}")
        print(f"  ROUGE-1: {test_metrics.get('rouge-1', 0):.4f}")
        print(f"  ROUGE-L: {test_metrics.get('rouge-l', 0):.4f}")
        print(f"  Exact Match: {test_metrics.get('exact_match', 0):.4f}")

    print("\n✅ Pipeline completed successfully!")
    print(f"All outputs saved in: ./example_output/")


def run_inference_example(model_path: str):
    """Example of using a trained model for inference"""

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    print("\n🔮 Running inference example...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    # Example prompts
    prompts = [
        "Summarize the following text: The quick brown fox jumps over the lazy dog.",
        "What is the main topic of this text? Climate change is affecting our planet.",
        "Extract key points from: Machine learning is a subset of artificial intelligence."
    ]

    print("\nGenerating responses:")
    print("-" * 40)

    for prompt in prompts:
        full_prompt = f"{prompt}\n\nResponse:"

        inputs = tokenizer(full_prompt, return_tensors='pt', truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("Response:")[-1].strip()

        print(f"\nPrompt: {prompt}")
        print(f"Response: {response}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Ryze-ACL Pipeline Example')
    parser.add_argument('--full', action='store_true', help='Run full pipeline')
    parser.add_argument('--inference', type=str, help='Run inference with model path')

    args = parser.parse_args()

    if args.full:
        run_complete_pipeline()
    elif args.inference:
        run_inference_example(args.inference)
    else:
        print("Usage:")
        print("  Run full pipeline: python example_pipeline.py --full")
        print("  Run inference: python example_pipeline.py --inference <model_path>")