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
        runner = DistributedRunner(pylet_manager=MagicMock())
        task = SimpleTask(TaskType.OCR)  # OCR is not a GPU task
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED

    def test_gpu_task_falls_back_on_no_manager(self):
        runner = DistributedRunner(pylet_manager=None)
        task = SimpleTask(TaskType.SFT_TRAIN)
        result = runner.run_task(task, {})
        assert result.status == TaskStatus.COMPLETED
