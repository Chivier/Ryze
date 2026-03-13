"""Tests for ryze.data.dataset module."""

from ryze.data.dataset import SFTDatasetGenerator


class TestSFTDatasetGenerator:
    def test_default_templates(self):
        gen = SFTDatasetGenerator()
        assert len(gen.instruction_templates) == 10

    def test_custom_templates(self):
        gen = SFTDatasetGenerator({"instruction_templates": ["Q1:", "Q2:"]})
        assert len(gen.instruction_templates) == 2

    def test_chunk_text_basic(self):
        gen = SFTDatasetGenerator({"min_text_length": 5})
        text = " ".join(["word"] * 100)
        chunks = gen.chunk_text(text, chunk_size=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) >= 5

    def test_chunk_text_short_input(self):
        gen = SFTDatasetGenerator({"min_text_length": 5})
        chunks = gen.chunk_text("short", chunk_size=100)
        assert len(chunks) == 1

    def test_chunk_text_below_min_length(self):
        gen = SFTDatasetGenerator({"min_text_length": 1000})
        chunks = gen.chunk_text("too short", chunk_size=100)
        assert len(chunks) == 0

    def test_generate_qa_pairs(self):
        gen = SFTDatasetGenerator({"min_text_length": 10})
        text = "This is a long enough document. It has multiple sentences. It discusses important topics."
        pairs = gen.generate_qa_pairs(text)
        assert len(pairs) > 0
        for pair in pairs:
            assert "instruction" in pair
            assert "input" in pair
            assert "output" in pair

    def test_create_dataset(self, sample_markdown_dir, tmp_path):
        gen = SFTDatasetGenerator({"min_text_length": 10})
        output_path = str(tmp_path / "dataset.json")
        metadata = gen.create_dataset(sample_markdown_dir, output_path)
        assert metadata["processed_files"] == 2
        assert metadata["total_qa_pairs"] > 0
        assert metadata["train_samples"] + metadata["val_samples"] == metadata["total_qa_pairs"]
