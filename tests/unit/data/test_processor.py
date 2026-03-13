"""Tests for ryze.data.processor module."""

from unittest.mock import patch


class TestRyzeDataProcessor:
    @patch("ryze.data.processor.PDFOCRProcessor")
    @patch("ryze.data.processor.SFTDatasetGenerator")
    def test_process_single_pdf(self, mock_gen_cls, mock_ocr_cls, tmp_path):
        from ryze.data.processor import RyzeDataProcessor

        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.process_pdf.return_value = {"status": "success", "output_path": str(tmp_path / "test.md")}
        mock_gen = mock_gen_cls.return_value
        mock_gen.create_dataset.return_value = {"processed_files": 1, "total_qa_pairs": 5, "train_samples": 4, "val_samples": 1, "train_path": "train.json", "val_path": "val.json"}

        processor = RyzeDataProcessor({"output_base": str(tmp_path)})
        result = processor.process_single_pdf("test.pdf")
        assert result["status"] == "success"

    @patch("ryze.data.processor.PDFOCRProcessor")
    @patch("ryze.data.processor.SFTDatasetGenerator")
    def test_process_batch_with_failure(self, mock_gen_cls, mock_ocr_cls, tmp_path):
        from ryze.data.processor import RyzeDataProcessor

        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.process_pdf.side_effect = RuntimeError("PDF corrupt")

        processor = RyzeDataProcessor({"output_base": str(tmp_path)})
        results = processor.process_batch(["bad.pdf"])
        assert len(results) == 1
        assert results[0]["status"] == "failed"

    @patch("ryze.data.processor.PDFOCRProcessor")
    @patch("ryze.data.processor.SFTDatasetGenerator")
    def test_process_directory(self, mock_gen_cls, mock_ocr_cls, tmp_path):
        from ryze.data.processor import RyzeDataProcessor

        # Create dummy PDFs
        (tmp_path / "doc1.pdf").write_bytes(b"fake")
        (tmp_path / "doc2.pdf").write_bytes(b"fake")

        mock_ocr = mock_ocr_cls.return_value
        mock_ocr.process_pdf.return_value = {"status": "success", "output_path": "out.md"}
        mock_gen = mock_gen_cls.return_value
        mock_gen.create_dataset.return_value = {"processed_files": 1, "total_qa_pairs": 3, "train_samples": 2, "val_samples": 1, "train_path": "t.json", "val_path": "v.json"}

        processor = RyzeDataProcessor({"output_base": str(tmp_path / "output")})
        results = processor.process_directory(str(tmp_path))
        assert len(results) == 2

    def test_as_task_creates_task(self):
        from ryze.core.task import TaskType
        from ryze.data.processor import RyzeDataProcessor

        processor = RyzeDataProcessor()
        task = processor.as_task(pdf_path="test.pdf")
        assert task.task_type == TaskType.OCR
        assert "test.pdf" in task.name
