"""Benchmark Runner for Model Evaluation"""
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Run standardized benchmarks for model evaluation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.benchmarks_dir = self.config.get('benchmarks_dir', './benchmarks')

    def load_benchmark(self, benchmark_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load a benchmark dataset"""
        benchmark_path = os.path.join(self.benchmarks_dir, f"{benchmark_name}.json")

        if not os.path.exists(benchmark_path):
            # Create a sample benchmark if it doesn't exist
            logger.info(f"Creating sample benchmark: {benchmark_name}")
            self._create_sample_benchmark(benchmark_path)

        with open(benchmark_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def _create_sample_benchmark(self, path: str):
        """Create a sample benchmark for testing"""
        sample_data = {
            "test": [
                {
                    "id": "sample_001",
                    "instruction": "Summarize the following text:",
                    "input": "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals. Leading AI textbooks define the field as the study of 'intelligent agents': any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.",
                    "expected_output": "AI is machine-demonstrated intelligence, contrasting with natural intelligence in humans and animals. It involves intelligent agents that perceive their environment and act to achieve goals.",
                    "category": "summarization"
                },
                {
                    "id": "sample_002",
                    "instruction": "What is the main topic of this text?",
                    "input": "Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver of climate change, primarily due to burning fossil fuels like coal, oil and gas.",
                    "expected_output": "The main topic is climate change, specifically its definition and the role of human activities as the primary driver since the 1800s.",
                    "category": "topic_identification"
                },
                {
                    "id": "sample_003",
                    "instruction": "Extract key points from this document:",
                    "input": "Machine learning is a subset of artificial intelligence that involves the use of algorithms and statistical models to enable computer systems to improve their performance on a specific task through experience. Unlike traditional programming where explicit instructions are provided, machine learning systems learn patterns from data.",
                    "expected_output": "Key points: 1) Machine learning is a subset of AI, 2) Uses algorithms and statistical models, 3) Systems improve through experience, 4) Learns patterns from data rather than explicit programming.",
                    "category": "key_points"
                }
            ],
            "validation": [
                {
                    "id": "sample_val_001",
                    "instruction": "Provide a brief overview of the following content:",
                    "input": "Quantum computing is a type of computation that harnesses the phenomena of quantum mechanics to process information. Unlike classical computers that use bits (0 or 1), quantum computers use quantum bits or qubits, which can exist in superposition of both states simultaneously.",
                    "expected_output": "Quantum computing uses quantum mechanics for computation, employing qubits that can exist in multiple states simultaneously, unlike classical bits that are either 0 or 1.",
                    "category": "overview"
                }
            ]
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)

    def prepare_prompts(self, benchmark_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Prepare prompts from benchmark data"""
        prompts = []

        for item in benchmark_data:
            instruction = item.get('instruction', '')
            input_text = item.get('input', '')

            if input_text:
                prompt = f"{instruction}\n\nInput: {input_text}\n\nResponse:"
            else:
                prompt = f"{instruction}\n\nResponse:"

            prompts.append({
                'id': item.get('id', ''),
                'prompt': prompt,
                'expected_output': item.get('expected_output', ''),
                'category': item.get('category', 'general'),
                'instruction': instruction,
                'input': input_text
            })

        return prompts

    def run_benchmark(self, benchmark_name: str, model_outputs: Dict[str, str]) -> Dict[str, Any]:
        """Run evaluation on a benchmark"""
        # Load benchmark
        benchmark_data = self.load_benchmark(benchmark_name)

        results = {
            'benchmark_name': benchmark_name,
            'timestamp': datetime.now().isoformat(),
            'results_by_split': {}
        }

        # Process each split (test, validation, etc.)
        for split_name, split_data in benchmark_data.items():
            split_results = []

            for item in split_data:
                item_id = item.get('id', '')
                expected = item.get('expected_output', '')
                model_output = model_outputs.get(item_id, '')

                result = {
                    'id': item_id,
                    'category': item.get('category', 'general'),
                    'expected_output': expected,
                    'model_output': model_output,
                    'instruction': item.get('instruction', ''),
                    'input': item.get('input', '')
                }

                split_results.append(result)

            results['results_by_split'][split_name] = split_results

        return results

    def get_available_benchmarks(self) -> List[str]:
        """Get list of available benchmarks"""
        if not os.path.exists(self.benchmarks_dir):
            os.makedirs(self.benchmarks_dir)
            # Create default benchmarks
            self._create_sample_benchmark(os.path.join(self.benchmarks_dir, "general_qa.json"))

        benchmarks = []
        for file in Path(self.benchmarks_dir).glob("*.json"):
            benchmarks.append(file.stem)

        return benchmarks

    def create_custom_benchmark(self, name: str, data: Dict[str, List[Dict[str, Any]]]):
        """Create a custom benchmark"""
        benchmark_path = os.path.join(self.benchmarks_dir, f"{name}.json")

        # Validate data structure
        required_fields = ['instruction', 'input', 'expected_output']
        for split_name, split_data in data.items():
            for item in split_data:
                for field in required_fields:
                    if field not in item:
                        raise ValueError(f"Missing required field '{field}' in item {item.get('id', 'unknown')}")

        # Save benchmark
        os.makedirs(self.benchmarks_dir, exist_ok=True)
        with open(benchmark_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Created custom benchmark: {name}")

        return benchmark_path