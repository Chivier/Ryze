"""Tests for ryze.core.runner module."""

from unittest.mock import MagicMock

from ryze.core.runner import DistributedRunner, LocalRunner
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class SimpleTask(RyzeTask):
    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"done": True})


class TestLocalRunner:
    def test_run_task_completes(self):
        runner = LocalRunner()
        task = SimpleTask(TaskType.OCR)
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED

    def test_cancel_unknown_task(self):
        runner = LocalRunner()
        assert runner.cancel_task("nonexistent") is False


class TestDistributedRunner:
    def test_cpu_task_runs_locally(self):
        runner = DistributedRunner(ray_manager=MagicMock())
        task = SimpleTask(TaskType.OCR)  # OCR is not a GPU task
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED

    def test_gpu_task_falls_back_on_no_manager(self):
        runner = DistributedRunner(ray_manager=None)
        task = SimpleTask(TaskType.SFT_TRAIN)
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED


class TestDistributedRunnerWithWrapper:
    def test_gpu_task_dispatched_via_wrapper(self):
        """GPU task should be routed through wrapper.submit()."""
        mock_wrapper = MagicMock()
        expected = TaskResult(status=TaskStatus.COMPLETED, output={"remote": True})
        mock_wrapper.submit.return_value = expected

        runner = DistributedRunner(ray_manager=MagicMock(), execution_wrapper=mock_wrapper)
        task = SimpleTask(TaskType.SFT_TRAIN)
        result = runner.run_task(task, {"key": "val"})

        mock_wrapper.submit.assert_called_once_with(task, {"key": "val"})
        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"remote": True}

    def test_cpu_task_still_local_with_wrapper(self):
        """CPU task should bypass wrapper and run locally."""
        mock_wrapper = MagicMock()
        runner = DistributedRunner(ray_manager=MagicMock(), execution_wrapper=mock_wrapper)
        task = SimpleTask(TaskType.OCR)
        result = runner.run_task(task, {})

        mock_wrapper.submit.assert_not_called()
        assert result.status == TaskStatus.COMPLETED

    def test_fallback_on_wrapper_failure(self):
        """When wrapper.submit raises, should fallback to local execution."""
        mock_wrapper = MagicMock()
        mock_wrapper.submit.side_effect = RuntimeError("cluster down")

        runner = DistributedRunner(ray_manager=MagicMock(), execution_wrapper=mock_wrapper)
        task = SimpleTask(TaskType.SFT_TRAIN)
        result = runner.run_task(task, {})

        assert result.status == TaskStatus.COMPLETED  # fell back to local

    def test_no_wrapper_when_no_manager(self):
        """Without ray_manager, wrapper should be None, GPU falls back to local."""
        runner = DistributedRunner(ray_manager=None)
        assert runner._wrapper is None
        task = SimpleTask(TaskType.SFT_TRAIN)
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED
