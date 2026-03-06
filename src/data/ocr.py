"""PDF OCR Processor for Ryze Data Module"""
import os
import json
from typing import List, Dict, Any, Optional
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class PDFOCRProcessor:
    """Process PDF files to extract text and convert to markdown format"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.dpi = self.config.get('dpi', 300)
        self.language = self.config.get('language', 'eng+chi_sim')
        self.use_gpu = self.config.get('use_gpu', False)

    def pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images"""
        images = []
        pdf_document = fitz.open(pdf_path)

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            mat = fitz.Matrix(self.dpi/72, self.dpi/72)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            images.append(img)

        pdf_document.close()
        return images

    def ocr_image(self, image: Image.Image) -> str:
        """Perform OCR on an image"""
        try:
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config='--psm 6'  # Uniform block of text
            )
            return text
        except Exception as e:
            logger.error(f"OCR failed: {str(e)}")
            return ""

    def text_to_markdown(self, text: str) -> str:
        """Convert plain text to markdown format"""
        # Basic markdown formatting
        lines = text.split('\n')
        markdown_lines = []

        for line in lines:
            line = line.strip()
            if not line:
                markdown_lines.append('')
                continue

            # Simple heuristics for headers
            if line.isupper() and len(line) < 50:
                markdown_lines.append(f"## {line}")
            elif line.endswith(':'):
                markdown_lines.append(f"### {line}")
            else:
                markdown_lines.append(line)

        return '\n'.join(markdown_lines)

    def process_pdf(self, pdf_path: str, output_dir: str) -> Dict[str, Any]:
        """Main processing function"""
        pdf_name = Path(pdf_path).stem
        output_path = os.path.join(output_dir, f"{pdf_name}.md")

        logger.info(f"Processing PDF: {pdf_path}")

        # Convert PDF to images
        images = self.pdf_to_images(pdf_path)

        # Perform OCR on each page
        all_text = []
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)}")
            text = self.ocr_image(image)
            all_text.append(f"<!-- Page {i+1} -->\n{text}")

        # Combine and convert to markdown
        combined_text = '\n\n'.join(all_text)
        markdown_text = self.text_to_markdown(combined_text)

        # Save markdown file
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)

        return {
            'pdf_path': pdf_path,
            'output_path': output_path,
            'page_count': len(images),
            'status': 'success'
        }