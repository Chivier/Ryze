"""Main Data Processing Module"""
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json
from datetime import datetime

from .ocr import PDFOCRProcessor
from .dataset import SFTDatasetGenerator

logger = logging.getLogger(__name__)


class RyzeDataProcessor:
    """Main processor that orchestrates PDF OCR and dataset generation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.ocr_processor = PDFOCRProcessor(self.config.get('ocr', {}))
        self.dataset_generator = SFTDatasetGenerator(self.config.get('dataset', {}))
        self.output_base = self.config.get('output_base', './output')

    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file"""
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_name = Path(pdf_path).stem
        output_dir = os.path.join(self.output_base, f"{pdf_name}_{timestamp}")
        markdown_dir = os.path.join(output_dir, "markdown")
        dataset_dir = os.path.join(output_dir, "dataset")

        os.makedirs(markdown_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Step 1: PDF OCR to Markdown
        logger.info(f"Starting OCR for: {pdf_path}")
        ocr_result = self.ocr_processor.process_pdf(pdf_path, markdown_dir)

        if ocr_result['status'] != 'success':
            return {
                'status': 'failed',
                'error': 'OCR processing failed',
                'details': ocr_result
            }

        # Step 2: Generate SFT Dataset
        logger.info("Generating SFT dataset from markdown")
        dataset_path = os.path.join(dataset_dir, f"{pdf_name}_sft.json")
        dataset_result = self.dataset_generator.create_dataset(markdown_dir, dataset_path)

        # Combine results
        result = {
            'status': 'success',
            'pdf_path': pdf_path,
            'output_dir': output_dir,
            'ocr_result': ocr_result,
            'dataset_result': dataset_result,
            'timestamp': timestamp
        }

        # Save processing log
        log_path = os.path.join(output_dir, 'processing_log.json')
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def process_batch(self, pdf_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple PDF files"""
        results = []

        for pdf_path in pdf_paths:
            try:
                logger.info(f"Processing: {pdf_path}")
                result = self.process_single_pdf(pdf_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
                results.append({
                    'status': 'failed',
                    'pdf_path': pdf_path,
                    'error': str(e)
                })

        return results

    def process_directory(self, input_dir: str, pattern: str = "*.pdf") -> List[Dict[str, Any]]:
        """Process all PDF files in a directory"""
        pdf_files = list(Path(input_dir).glob(pattern))
        logger.info(f"Found {len(pdf_files)} PDF files to process")

        return self.process_batch([str(f) for f in pdf_files])