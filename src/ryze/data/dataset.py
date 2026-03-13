"""SFT Dataset Generator for Ryze Data Module"""
from __future__ import annotations

import json
import logging
import os
import random
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..core.task import RyzeTask

logger = logging.getLogger(__name__)


class SFTDatasetGenerator:
    """Generate SFT (Supervised Fine-Tuning) dataset from markdown OCR results"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.instruction_templates = self.config.get('instruction_templates', self._get_default_templates())
        self.min_text_length = self.config.get('min_text_length', 50)
        self.max_text_length = self.config.get('max_text_length', 2048)

    def _get_default_templates(self) -> List[str]:
        """Default instruction templates"""
        return [
            "Please summarize the following text:",
            "Extract the key points from this document:",
            "What is the main topic of this text?",
            "Identify the important information in this passage:",
            "Provide a brief overview of the following content:",
            "List the main conclusions from this text:",
            "What are the key findings mentioned in this document?",
            "Summarize the following in your own words:",
            "Extract the most important facts from this text:",
            "What is the central message of this document?"
        ]

    def chunk_text(self, text: str, chunk_size: int = 1024) -> List[str]:
        """Split text into chunks of appropriate size"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text) >= self.min_text_length:
                    chunks.append(chunk_text)
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text) >= self.min_text_length:
                chunks.append(chunk_text)

        return chunks

    def generate_qa_pairs(self, text: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from text"""
        qa_pairs = []
        chunks = self.chunk_text(text, self.max_text_length)

        for chunk in chunks:
            # Skip chunks that are too short
            if len(chunk) < self.min_text_length:
                continue

            # Generate instruction-response pairs
            instruction = random.choice(self.instruction_templates)

            # Create a simple response based on the chunk
            # In a real implementation, you might use an LLM to generate better responses
            response = self._generate_response(chunk, instruction)

            qa_pairs.append({
                "instruction": instruction,
                "input": chunk,
                "output": response
            })

        return qa_pairs

    def _generate_response(self, text: str, instruction: str) -> str:
        """Generate a response based on the instruction"""
        # This is a simplified version. In production, you might want to use
        # an LLM or more sophisticated NLP techniques

        sentences = text.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        if "summarize" in instruction.lower():
            # Take first and last sentences as summary
            if len(sentences) >= 2:
                return f"{sentences[0]}. {sentences[-1]}."
            else:
                return sentences[0] if sentences else text[:200]

        elif "key points" in instruction.lower() or "important information" in instruction.lower():
            # Extract first few sentences
            key_points = sentences[:3] if len(sentences) >= 3 else sentences
            return " ".join(key_points)

        elif "main topic" in instruction.lower():
            # Return first sentence as main topic
            return sentences[0] if sentences else text[:200]

        else:
            # Default: return a shortened version
            return text[:500] if len(text) > 500 else text

    def process_markdown_file(self, markdown_path: str) -> List[Dict[str, str]]:
        """Process a single markdown file to generate dataset"""
        with open(markdown_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Remove markdown formatting for cleaner text
        # Simple removal of common markdown elements
        content = content.replace('##', '').replace('###', '')
        content = content.replace('*', '').replace('_', '')
        content = content.replace('<!-- ', '').replace(' -->', '')

        return self.generate_qa_pairs(content)

    def create_dataset(self, markdown_dir: str, output_path: str) -> Dict[str, Any]:
        """Create SFT dataset from all markdown files in directory"""
        all_qa_pairs = []
        processed_files = 0

        markdown_files = list(Path(markdown_dir).glob("*.md"))

        for md_file in markdown_files:
            logger.info(f"Processing markdown file: {md_file}")
            qa_pairs = self.process_markdown_file(str(md_file))
            all_qa_pairs.extend(qa_pairs)
            processed_files += 1

        # Split into train/validation sets
        random.shuffle(all_qa_pairs)
        split_idx = int(len(all_qa_pairs) * 0.9)
        train_data = all_qa_pairs[:split_idx]
        val_data = all_qa_pairs[split_idx:]

        # Save datasets
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        train_path = output_path.replace('.json', '_train.json')
        val_path = output_path.replace('.json', '_val.json')

        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)

        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)

        metadata = {
            'processed_files': processed_files,
            'total_qa_pairs': len(all_qa_pairs),
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'timestamp': datetime.now().isoformat(),
            'train_path': train_path,
            'val_path': val_path
        }

        # Save metadata
        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return metadata

    def as_task(self, markdown_dir: str = "", output_path: str = "") -> RyzeTask:
        """Create a RyzeTask wrapper for this generator."""
        from ..core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType

        generator = self

        class DatasetGenTask(RyzeTask):
            def __init__(self, markdown_dir: str, output_path: str):
                super().__init__(
                    task_type=TaskType.DATASET_GEN,
                    inputs={"markdown_dir": markdown_dir, "output_path": output_path},
                    name="Dataset Generation",
                )

            def resource_requirements(self) -> ResourceRequirement:
                return ResourceRequirement(gpu_count=0, memory_gb=1.0, estimated_duration_s=30)

            def validate_inputs(self) -> bool:
                return True  # Inputs may come from upstream task

            def execute(self, inputs: dict) -> TaskResult:
                md_dir = inputs.get("markdown_dir") or inputs.get("output_path", "")
                if md_dir.endswith(".md"):
                    md_dir = str(Path(md_dir).parent)
                out = inputs.get("output_path") or os.path.join(md_dir, "dataset.json")
                metadata = generator.create_dataset(md_dir, out)
                return TaskResult(
                    status=TaskStatus.COMPLETED,
                    output=metadata,
                    artifacts=[metadata.get("train_path", ""), metadata.get("val_path", "")],
                )

        return DatasetGenTask(markdown_dir, output_path)
