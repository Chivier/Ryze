"""Tests for ryze.core.task module."""

from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class ConcreteTask(RyzeTask):
    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1, memory_gb=8.0)

    def validate_inputs(self):
        return "required_key" in self.inputs

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"result": "ok"})


class FailingTask(RyzeTask):
    def resource_requirements(self):
        return ResourceRequirement()

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        raise RuntimeError("something broke")


class TestRyzeTask:
    def test_task_id_format(self):
        task = ConcreteTask(TaskType.SFT_TRAIN, inputs={"required_key": True})
        assert task.task_id.startswith("sft_train_")
        assert len(task.task_id) == len("sft_train_") + 8

    def test_status_lifecycle(self):
        task = ConcreteTask(TaskType.OCR, inputs={"required_key": True})
        assert task.status == TaskStatus.PENDING
        result = task.run()
        assert result.status == TaskStatus.COMPLETED
        assert task.status == TaskStatus.COMPLETED

    def test_validation_failure(self):
        task = ConcreteTask(TaskType.OCR, inputs={})
        result = task.run()
        assert result.status == TaskStatus.FAILED
        assert "validation failed" in result.error

    def test_execution_exception(self):
        task = FailingTask(TaskType.OCR)
        result = task.run()
        assert result.status == TaskStatus.FAILED
        assert "something broke" in result.error
        assert result.duration_s > 0

    def test_to_dict_serialization(self):
        task = ConcreteTask(TaskType.EVALUATION, inputs={"required_key": True}, name="test-eval")
        d = task.to_dict()
        assert d["task_type"] == "evaluation"
        assert d["name"] == "test-eval"
        assert d["status"] == "pending"
