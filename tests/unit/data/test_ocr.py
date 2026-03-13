"""Tests for ryze.data.ocr module."""

from unittest.mock import MagicMock, patch


class TestPDFOCRProcessor:
    @patch("ryze.data.ocr.fitz")
    def test_pdf_to_images(self, mock_fitz):
        from ryze.data.ocr import PDFOCRProcessor

        # Setup mock PDF
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = _make_png_bytes()
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__ = lambda self: 2
        mock_doc.__iter__ = lambda self: iter([mock_page, mock_page])
        mock_doc.__getitem__ = lambda self, i: mock_page
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix = MagicMock()

        processor = PDFOCRProcessor({"dpi": 150})
        images = processor.pdf_to_images("test.pdf")
        assert len(images) == 2
        mock_fitz.open.assert_called_once_with("test.pdf")

    @patch("ryze.data.ocr.pytesseract")
    def test_ocr_image(self, mock_tess):
        from ryze.data.ocr import PDFOCRProcessor

        mock_tess.image_to_string.return_value = "Hello world"
        processor = PDFOCRProcessor()
        result = processor.ocr_image(MagicMock())
        assert result == "Hello world"
        mock_tess.image_to_string.assert_called_once()

    @patch("ryze.data.ocr.pytesseract")
    def test_ocr_image_failure(self, mock_tess):
        from ryze.data.ocr import PDFOCRProcessor

        mock_tess.image_to_string.side_effect = RuntimeError("OCR failed")
        processor = PDFOCRProcessor()
        result = processor.ocr_image(MagicMock())
        assert result == ""

    def test_text_to_markdown_headers(self):
        from ryze.data.ocr import PDFOCRProcessor

        processor = PDFOCRProcessor()
        text = "INTRODUCTION\nSome content\nSection:"
        md = processor.text_to_markdown(text)
        assert "## INTRODUCTION" in md
        assert "### Section:" in md

    def test_text_to_markdown_empty_lines(self):
        from ryze.data.ocr import PDFOCRProcessor

        processor = PDFOCRProcessor()
        md = processor.text_to_markdown("line1\n\nline2")
        assert "line1" in md
        assert "line2" in md

    @patch("ryze.data.ocr.fitz")
    @patch("ryze.data.ocr.pytesseract")
    def test_process_pdf(self, mock_tess, mock_fitz, tmp_path):
        from ryze.data.ocr import PDFOCRProcessor

        # Mock PDF with 1 page
        mock_doc = MagicMock()
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = _make_png_bytes()
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__ = lambda self: 1
        mock_doc.__getitem__ = lambda self, i: mock_page
        mock_fitz.open.return_value = mock_doc
        mock_fitz.Matrix = MagicMock()
        mock_tess.image_to_string.return_value = "OCR text content"

        processor = PDFOCRProcessor({"dpi": 150})
        result = processor.process_pdf("test.pdf", str(tmp_path))
        assert result["status"] == "success"
        assert result["page_count"] == 1
        assert (tmp_path / "test.md").exists()


def _make_png_bytes():
    """Create minimal valid PNG bytes."""
    import struct
    import zlib

    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = zlib.compress(b"\x00\x00\x00\x00")
    idat = chunk(b"IDAT", raw)
    iend = chunk(b"IEND", b"")
    return sig + ihdr + idat + iend
