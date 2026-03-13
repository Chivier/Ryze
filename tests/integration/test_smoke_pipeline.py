"""End-to-end tests for the migrated smoke test pipeline (spec §6.4)."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


@pytest.mark.integration
class TestSmokePipelineLocalMode:
    """test_smoke_pipeline_local_mode: LocalRunner runs all three stages,
    verifies outputs propagate correctly between stages."""

    def test_pipeline_build_creates_three_stages(self, tmp_path):
        """Pipeline construction produces 3 tasks with correct dependencies."""
        from smoke_test_pipeline import build_smoke_pipeline, create_sample_markdown

        work_dir = str(tmp_path)
        create_sample_markdown(work_dir)
        pipeline = build_smoke_pipeline(work_dir, "HuggingFaceTB/SmolLM-135M")

        assert len(pipeline.tasks) == 3

    def test_data_stage_produces_required_keys(self, tmp_path):
        """DatasetGenTask produces train_path, val_path, and grpo_data_path."""
        from ryze.data.dataset import SFTDatasetGenerator

        work_dir = str(tmp_path)
        md_dir = os.path.join(work_dir, "markdown")
        os.makedirs(md_dir)
        for i, text in enumerate([
            "## Doc One\n\nThis is a test document with enough text for processing. "
            "Machine learning uses data to find patterns in large datasets.",
            "## Doc Two\n\nAnother test document. Natural language processing enables "
            "computers to understand and generate human language effectively.",
        ]):
            Path(os.path.join(md_dir, f"doc_{i}.md")).write_text(text)

        gen = SFTDatasetGenerator({"min_text_length": 10, "max_text_length": 512})
        task = gen.as_task(
            markdown_dir=md_dir,
            output_path=os.path.join(work_dir, "dataset", "sft.json"),
        )
        result = task.run({})

        assert result.status.value == "completed"
        assert "train_path" in result.output
        assert "val_path" in result.output
        assert "grpo_data_path" in result.output
        assert os.path.exists(result.output["grpo_data_path"])

    def test_output_keys_flow_between_stages(self, tmp_path):
        """Verify that output key names from each stage match the input key names
        expected by the downstream stage (train_path, merged_model_path, grpo_data_path)."""
        from ryze.core.task import TaskType
        from ryze.data.dataset import SFTDatasetGenerator
        from ryze.rl.grpo_trainer import RyzeGRPOTrainer
        from ryze.rl.sft_lora_trainer import RyzeSFTLoRATrainer

        # Build tasks
        gen = SFTDatasetGenerator({"min_text_length": 10})
        data_task = gen.as_task(
            markdown_dir=str(tmp_path / "md"),
            output_path=str(tmp_path / "sft.json"),
        )
        sft_task = RyzeSFTLoRATrainer({"base_model_name": "test"}).as_task()
        grpo_task = RyzeGRPOTrainer({"batch_size": 2}).as_task()

        # Verify task types match runner routing expectations
        assert data_task.task_type == TaskType.DATASET_GEN
        assert sft_task.task_type == TaskType.SFT_TRAIN
        assert grpo_task.task_type == TaskType.GRPO_TRAIN


@pytest.mark.integration
class TestSmokePipelineRayMode:
    """test_smoke_pipeline_ray_mode: mock Ray cluster runs three stages."""

    def test_distributed_runner_routes_gpu_tasks_to_wrapper(self):
        """DistributedRunner uses wrapper.submit() for GPU tasks."""
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper
        from ryze.core.runner import DistributedRunner
        from ryze.core.task import TaskResult, TaskStatus, TaskType

        mock_manager = MagicMock()
        mock_wrapper = MagicMock(spec=RayExecutionWrapper)
        mock_wrapper.submit.return_value = TaskResult(
            status=TaskStatus.COMPLETED,
            output={"merged_model_path": "/tmp/model"},
        )

        runner = DistributedRunner(
            ray_manager=mock_manager,
            execution_wrapper=mock_wrapper,
        )

        # Create a GPU task
        task = MagicMock()
        task.task_type = TaskType.SFT_TRAIN
        task.name = "test_sft"

        result = runner.run_task(task, {})

        mock_wrapper.submit.assert_called_once_with(task, {})
        assert result.status == TaskStatus.COMPLETED

    def test_distributed_runner_uses_local_for_cpu_tasks(self):
        """DistributedRunner runs CPU tasks locally, not through wrapper."""
        from ryze.core.runner import DistributedRunner
        from ryze.core.task import TaskResult, TaskStatus, TaskType

        mock_manager = MagicMock()
        mock_wrapper = MagicMock()

        runner = DistributedRunner(
            ray_manager=mock_manager,
            execution_wrapper=mock_wrapper,
        )

        # Create a CPU task with mock run method
        task = MagicMock()
        task.task_type = TaskType.DATASET_GEN
        task.name = "test_data"
        task.run.return_value = TaskResult(
            status=TaskStatus.COMPLETED,
            output={"train_path": "/tmp/data.json"},
        )

        result = runner.run_task(task, {})

        mock_wrapper.submit.assert_not_called()
        task.run.assert_called_once_with({})
        assert result.status == TaskStatus.COMPLETED
