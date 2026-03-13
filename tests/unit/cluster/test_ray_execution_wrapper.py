"""Tests for ryze.cluster.ray_execution_wrapper module."""

from unittest.mock import MagicMock, patch

import pytest

from ryze.cluster.ray_manager import RayManager
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType
from ryze.exceptions import ClusterError


class FakeTaskWithConfig(RyzeTask):
    """Task that supports to_config() for remote execution."""

    def __init__(self):
        super().__init__(TaskType.SFT_TRAIN, name="FakeGPUTask")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1, memory_gb=16.0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"result": "ok"})

    def to_config(self):
        return {
            "trainer_class_path": "tests.unit.cluster.test_ray_execution_wrapper.FakeTrainer",
            "trainer_config": {"key": "value"},
        }


class FakeTaskWithoutConfig(RyzeTask):
    """Task that does NOT support to_config()."""

    def __init__(self):
        super().__init__(TaskType.SFT_TRAIN, name="NoConfigTask")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED)


class TestExtractTaskSpec:
    def test_extract_spec_success(self):
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        task = FakeTaskWithConfig()
        spec = RayExecutionWrapper._extract_task_spec(task)
        assert spec["trainer_class_path"] == "tests.unit.cluster.test_ray_execution_wrapper.FakeTrainer"
        assert spec["trainer_config"] == {"key": "value"}

    def test_extract_spec_missing_to_config(self):
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        task = FakeTaskWithoutConfig()
        with pytest.raises(ClusterError, match="does not implement to_config"):
            RayExecutionWrapper._extract_task_spec(task)


class TestSubmit:
    def test_submit_success(self):
        """Normal submit: mock ray.remote returns a successful TaskResult."""
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        mock_manager = MagicMock(spec=RayManager)
        mock_ray = MagicMock()
        expected_result = TaskResult(status=TaskStatus.COMPLETED, output={"done": True})
        mock_ref = MagicMock()
        mock_ray.get.return_value = expected_result

        wrapper = RayExecutionWrapper(mock_manager, max_retries=1, timeout_s=10)

        with patch("ryze.cluster.ray_execution_wrapper.ray", mock_ray):
            # Mock the remote function's .options().remote() chain
            mock_remote_fn = MagicMock()
            mock_remote_fn.options.return_value.remote.return_value = mock_ref
            with patch("ryze.cluster.ray_execution_wrapper._remote_fn", mock_remote_fn):
                task = FakeTaskWithConfig()
                result = wrapper.submit(task, {"input": "data"})

        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"done": True}

    def test_submit_retry_on_failure(self):
        """First attempt fails, second succeeds."""
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        mock_manager = MagicMock(spec=RayManager)
        mock_ray = MagicMock()
        expected_result = TaskResult(status=TaskStatus.COMPLETED, output={"retried": True})
        mock_ray.get.side_effect = [RuntimeError("worker died"), expected_result]

        wrapper = RayExecutionWrapper(mock_manager, max_retries=2, timeout_s=10)

        with patch("ryze.cluster.ray_execution_wrapper.ray", mock_ray):
            mock_remote_fn = MagicMock()
            mock_ref = MagicMock()
            mock_remote_fn.options.return_value.remote.return_value = mock_ref
            with patch("ryze.cluster.ray_execution_wrapper._remote_fn", mock_remote_fn):
                task = FakeTaskWithConfig()
                result = wrapper.submit(task, {})

        assert result.status == TaskStatus.COMPLETED
        assert result.output == {"retried": True}

    def test_submit_max_retries_exceeded(self):
        """All retries fail — should return FAILED TaskResult."""
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        mock_manager = MagicMock(spec=RayManager)
        mock_ray = MagicMock()
        mock_ray.get.side_effect = RuntimeError("persistent failure")

        wrapper = RayExecutionWrapper(mock_manager, max_retries=1, timeout_s=10)

        with patch("ryze.cluster.ray_execution_wrapper.ray", mock_ray):
            mock_remote_fn = MagicMock()
            mock_remote_fn.options.return_value.remote.return_value = MagicMock()
            with patch("ryze.cluster.ray_execution_wrapper._remote_fn", mock_remote_fn):
                task = FakeTaskWithConfig()
                result = wrapper.submit(task, {})

        assert result.status == TaskStatus.FAILED
        assert "persistent failure" in result.error


class TestCancel:
    def test_cancel_active_task(self):
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        mock_manager = MagicMock(spec=RayManager)
        wrapper = RayExecutionWrapper(mock_manager)
        mock_ref = MagicMock()
        wrapper._active_refs["task_123"] = mock_ref

        mock_ray = MagicMock()
        with patch("ryze.cluster.ray_execution_wrapper.ray", mock_ray):
            assert wrapper.cancel("task_123") is True
        mock_ray.cancel.assert_called_once_with(mock_ref)

    def test_cancel_unknown_task(self):
        from ryze.cluster.ray_execution_wrapper import RayExecutionWrapper

        mock_manager = MagicMock(spec=RayManager)
        wrapper = RayExecutionWrapper(mock_manager)
        assert wrapper.cancel("nonexistent") is False


class TestExecuteTaskRemote:
    def test_remote_rebuilds_and_executes(self):
        """_execute_task_remote should import trainer, call as_task(), run()."""
        from ryze.cluster.ray_execution_wrapper import _execute_task_remote_impl

        mock_task = MagicMock()
        expected = TaskResult(status=TaskStatus.COMPLETED, output={"rebuilt": True})
        mock_task.run.return_value = expected

        mock_trainer = MagicMock()
        mock_trainer.as_task.return_value = mock_task

        mock_class = MagicMock(return_value=mock_trainer)
        mock_module = MagicMock()
        setattr(mock_module, "FakeTrainer", mock_class)

        with patch("importlib.import_module", return_value=mock_module):
            result = _execute_task_remote_impl(
                "some.module.FakeTrainer",
                {"config_key": "val"},
                {"input_key": "data"},
            )

        mock_class.assert_called_once_with({"config_key": "val"})
        mock_trainer.as_task.assert_called_once()
        mock_task.run.assert_called_once_with({"input_key": "data"})
        assert result.status == TaskStatus.COMPLETED
