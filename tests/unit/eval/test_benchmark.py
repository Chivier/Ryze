"""Tests for ryze.eval.benchmark module."""

from ryze.eval.benchmark import BenchmarkRunner


class TestBenchmarkRunner:
    def test_load_benchmark_creates_sample(self, tmp_path):
        runner = BenchmarkRunner({"benchmarks_dir": str(tmp_path)})
        data = runner.load_benchmark("general_qa")
        assert "test" in data
        assert len(data["test"]) > 0

    def test_prepare_prompts(self, sample_benchmark):
        benchmarks_dir, name = sample_benchmark
        runner = BenchmarkRunner({"benchmarks_dir": benchmarks_dir})
        data = runner.load_benchmark(name)
        prompts = runner.prepare_prompts(data["test"])
        assert len(prompts) == 2
        assert "prompt" in prompts[0]
        assert "Response:" in prompts[0]["prompt"]

    def test_run_benchmark(self, sample_benchmark):
        benchmarks_dir, name = sample_benchmark
        runner = BenchmarkRunner({"benchmarks_dir": benchmarks_dir})
        outputs = {"t1": "AI is intelligence.", "t2": "Climate."}
        results = runner.run_benchmark(name, outputs)
        assert "results_by_split" in results

    def test_get_available_benchmarks(self, sample_benchmark):
        benchmarks_dir, _ = sample_benchmark
        runner = BenchmarkRunner({"benchmarks_dir": benchmarks_dir})
        benchmarks = runner.get_available_benchmarks()
        assert "test_benchmark" in benchmarks

    def test_create_custom_benchmark(self, tmp_path):
        runner = BenchmarkRunner({"benchmarks_dir": str(tmp_path)})
        data = {
            "test": [{"id": "c1", "instruction": "Q:", "input": "text", "expected_output": "answer"}]
        }
        runner.create_custom_benchmark("custom", data)
        assert (tmp_path / "custom.json").exists()
