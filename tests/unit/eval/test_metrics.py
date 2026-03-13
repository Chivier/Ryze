"""Tests for ryze.eval.metrics module."""

from ryze.eval.metrics import MetricsCalculator


class TestBleu:
    def test_identical_strings(self):
        score = MetricsCalculator.calculate_bleu("hello world", "hello world")
        assert score > 0.9

    def test_empty_hypothesis(self):
        score = MetricsCalculator.calculate_bleu("hello world", "")
        assert score == 0.0

    def test_partial_overlap(self):
        # With 4-gram BLEU, short sentences may get 0 due to missing higher n-grams
        # Use unigram BLEU to test partial overlap
        score = MetricsCalculator.calculate_bleu("the cat sat on the mat", "the cat on the mat", n_gram=1)
        assert 0.0 < score < 1.0


class TestRouge:
    def test_identical_strings(self):
        scores = MetricsCalculator.calculate_rouge("hello world", "hello world")
        assert scores["rouge-1"] > 0.9
        assert scores["rouge-l"] > 0.9

    def test_empty_strings(self):
        scores = MetricsCalculator.calculate_rouge("", "")
        assert scores["rouge-1"] == 0.0

    def test_partial_overlap(self):
        scores = MetricsCalculator.calculate_rouge("the cat sat on the mat", "the cat is on a mat")
        assert 0.0 < scores["rouge-1"] < 1.0


class TestExactMatch:
    def test_exact_match(self):
        score = MetricsCalculator.calculate_exact_match("Hello World", "hello world")
        assert score == 1.0

    def test_no_match(self):
        score = MetricsCalculator.calculate_exact_match("hello", "world")
        assert score == 0.0

    def test_punctuation_ignored(self):
        score = MetricsCalculator.calculate_exact_match("hello, world!", "hello world")
        assert score == 1.0


class TestBatchMetrics:
    def test_calculate_all_metrics(self):
        refs = ["the cat sat", "hello world"]
        hyps = ["the cat sat", "hi world"]
        metrics = MetricsCalculator.calculate_all_metrics(refs, hyps)
        assert "bleu" in metrics
        assert "rouge-1" in metrics
        assert "exact_match" in metrics
        assert "bleu_std" in metrics


class TestDiversity:
    def test_diversity_metrics(self):
        texts = ["the cat sat on the mat", "dogs are friendly animals"]
        metrics = MetricsCalculator.calculate_diversity_metrics(texts)
        assert "distinct-1" in metrics
        assert "distinct-2" in metrics
        assert "avg_length" in metrics
        assert metrics["avg_length"] > 0
