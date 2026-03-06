"""Main Evaluator Module"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
from pathlib import Path
from tqdm import tqdm

from .metrics import MetricsCalculator
from .benchmark import BenchmarkRunner

logger = logging.getLogger(__name__)


class RyzeEvaluator:
    """Main evaluator for model performance assessment"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = self.config.get('output_dir', './eval_outputs')

        # Model parameters
        self.max_new_tokens = self.config.get('max_new_tokens', 256)
        self.temperature = self.config.get('temperature', 0.7)
        self.do_sample = self.config.get('do_sample', False)
        self.top_p = self.config.get('top_p', 0.95)

        # Initialize components
        self.metrics_calculator = MetricsCalculator()
        self.benchmark_runner = BenchmarkRunner(config)

        # Model and tokenizer
        self.model = None
        self.tokenizer = None

    def load_model(self, model_path: str):
        """Load model for evaluation"""
        logger.info(f"Loading model from: {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()

        logger.info(f"Model loaded on {self.device}")

    def generate_response(self, prompt: str) -> str:
        """Generate a single response"""
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # Extract only the generated part
        prompt_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][prompt_length:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response.strip()

    def evaluate_model(self, model_path: str, benchmark_name: str) -> Dict[str, Any]:
        """Evaluate model on a specific benchmark"""
        # Load model
        self.load_model(model_path)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = Path(model_path).name
        run_output_dir = os.path.join(
            self.output_dir,
            f"eval_{model_name}_{benchmark_name}_{timestamp}"
        )
        os.makedirs(run_output_dir, exist_ok=True)

        # Load benchmark
        benchmark_data = self.benchmark_runner.load_benchmark(benchmark_name)

        # Results storage
        all_results = {
            'model_path': model_path,
            'benchmark_name': benchmark_name,
            'timestamp': timestamp,
            'config': self.config,
            'results_by_split': {}
        }

        # Process each split
        for split_name, split_data in benchmark_data.items():
            logger.info(f"Evaluating on {split_name} split ({len(split_data)} examples)")

            # Prepare prompts
            prompts = self.benchmark_runner.prepare_prompts(split_data)

            # Generate responses
            model_outputs = {}
            references = []
            hypotheses = []

            for prompt_data in tqdm(prompts, desc=f"Generating {split_name}"):
                prompt = prompt_data['prompt']
                prompt_id = prompt_data['id']

                # Generate response
                response = self.generate_response(prompt)
                model_outputs[prompt_id] = response

                # Collect for metrics
                references.append(prompt_data['expected_output'])
                hypotheses.append(response)

            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all_metrics(references, hypotheses)
            diversity_metrics = self.metrics_calculator.calculate_diversity_metrics(hypotheses)

            # Combine metrics
            all_metrics = {**metrics, **diversity_metrics}

            # Run benchmark evaluation
            benchmark_results = self.benchmark_runner.run_benchmark(
                benchmark_name,
                model_outputs
            )

            # Store results for this split
            split_results = {
                'metrics': all_metrics,
                'num_examples': len(split_data),
                'detailed_results': benchmark_results['results_by_split'][split_name]
            }

            all_results['results_by_split'][split_name] = split_results

            # Save intermediate results
            split_output_path = os.path.join(run_output_dir, f"{split_name}_results.json")
            with open(split_output_path, 'w', encoding='utf-8') as f:
                json.dump(split_results, f, ensure_ascii=False, indent=2)

            logger.info(f"{split_name} metrics: {all_metrics}")

        # Save complete results
        final_results_path = os.path.join(run_output_dir, 'complete_evaluation_results.json')
        with open(final_results_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        # Generate summary report
        self._generate_summary_report(all_results, run_output_dir)

        logger.info(f"Evaluation completed. Results saved to: {run_output_dir}")

        return all_results

    def _generate_summary_report(self, results: Dict[str, Any], output_dir: str):
        """Generate a human-readable summary report"""
        report_lines = [
            "# Ryze Evaluation Report",
            "",
            f"**Model**: {results['model_path']}",
            f"**Benchmark**: {results['benchmark_name']}",
            f"**Timestamp**: {results['timestamp']}",
            "",
            "## Results Summary",
            ""
        ]

        # Add metrics for each split
        for split_name, split_results in results['results_by_split'].items():
            metrics = split_results['metrics']

            report_lines.extend([
                f"### {split_name.capitalize()} Split",
                f"- **Number of Examples**: {split_results['num_examples']}",
                f"- **BLEU Score**: {metrics.get('bleu', 0):.4f}",
                f"- **ROUGE-1**: {metrics.get('rouge-1', 0):.4f}",
                f"- **ROUGE-2**: {metrics.get('rouge-2', 0):.4f}",
                f"- **ROUGE-L**: {metrics.get('rouge-l', 0):.4f}",
                f"- **Exact Match**: {metrics.get('exact_match', 0):.4f}",
                f"- **Distinct-1**: {metrics.get('distinct-1', 0):.4f}",
                f"- **Distinct-2**: {metrics.get('distinct-2', 0):.4f}",
                f"- **Average Length**: {metrics.get('avg_length', 0):.2f} tokens",
                ""
            ])

            # Add some example outputs
            report_lines.extend([
                "#### Sample Outputs:",
                ""
            ])

            # Show first 3 examples
            for i, result in enumerate(split_results['detailed_results'][:3]):
                report_lines.extend([
                    f"**Example {i+1}**",
                    f"- **Instruction**: {result['instruction']}",
                    f"- **Input**: {result['input'][:100]}..." if len(result['input']) > 100 else f"- **Input**: {result['input']}",
                    f"- **Expected**: {result['expected_output']}",
                    f"- **Model Output**: {result['model_output']}",
                    ""
                ])

        # Save report
        report_path = os.path.join(output_dir, 'evaluation_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

    def compare_models(self, model_paths: List[str], benchmark_name: str) -> Dict[str, Any]:
        """Compare multiple models on the same benchmark"""
        comparison_results = {
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }

        for model_path in model_paths:
            logger.info(f"Evaluating model: {model_path}")
            results = self.evaluate_model(model_path, benchmark_name)

            model_name = Path(model_path).name
            comparison_results['models'][model_name] = results

        # Create comparison summary
        comparison_output_dir = os.path.join(
            self.output_dir,
            f"comparison_{benchmark_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(comparison_output_dir, exist_ok=True)

        # Save comparison results
        comparison_path = os.path.join(comparison_output_dir, 'model_comparison.json')
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, ensure_ascii=False, indent=2)

        # Generate comparison report
        self._generate_comparison_report(comparison_results, comparison_output_dir)

        return comparison_results

    def _generate_comparison_report(self, comparison_results: Dict[str, Any], output_dir: str):
        """Generate a comparison report for multiple models"""
        report_lines = [
            "# Model Comparison Report",
            "",
            f"**Benchmark**: {comparison_results['benchmark_name']}",
            f"**Timestamp**: {comparison_results['timestamp']}",
            "",
            "## Metrics Comparison",
            ""
        ]

        # Create comparison table
        models = list(comparison_results['models'].keys())
        if models:
            # Get metrics from first model to know what to compare
            first_model_results = comparison_results['models'][models[0]]
            splits = list(first_model_results['results_by_split'].keys())

            for split in splits:
                report_lines.extend([
                    f"### {split.capitalize()} Split",
                    "",
                    "| Model | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Exact Match |",
                    "|-------|------|---------|---------|---------|-------------|"
                ])

                for model_name in models:
                    model_results = comparison_results['models'][model_name]
                    metrics = model_results['results_by_split'][split]['metrics']

                    row = (
                        f"| {model_name} | "
                        f"{metrics.get('bleu', 0):.4f} | "
                        f"{metrics.get('rouge-1', 0):.4f} | "
                        f"{metrics.get('rouge-2', 0):.4f} | "
                        f"{metrics.get('rouge-l', 0):.4f} | "
                        f"{metrics.get('exact_match', 0):.4f} |"
                    )
                    report_lines.append(row)

                report_lines.append("")

        # Save report
        report_path = os.path.join(output_dir, 'comparison_report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        logger.info(f"Comparison report saved to: {report_path}")