"""Integration tests for task lifecycle."""

from ryze.core.runner import LocalRunner
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class EchoTask(RyzeTask):
    def resource_requirements(self):
        return ResourceRequirement()

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"echo": inputs.get("msg", "")})


class TestTaskLifecycle:
    def test_full_lifecycle_with_local_runner(self):
        runner = LocalRunner()
        task = EchoTask(TaskType.OCR, inputs={"msg": "hello"}, name="echo-test")
        assert task.status == TaskStatus.PENDING
        result = runner.run_task(task, {"msg": "hello"})
        assert result.status == TaskStatus.COMPLETED
        assert result.output["echo"] == "hello"

    def test_task_with_progress_callback(self):
        progress_log = []

        class ProgressTask(RyzeTask):
            def resource_requirements(self):
                return ResourceRequirement()

            def validate_inputs(self):
                return True

            def execute(self, inputs):
                self.report_progress(0.5, "halfway")
                self.report_progress(1.0, "done")
                return TaskResult(status=TaskStatus.COMPLETED)

        task = ProgressTask(TaskType.OCR)
        task.set_progress_callback(lambda p, m: progress_log.append((p, m)))
        task.run()
        assert len(progress_log) == 2
        assert progress_log[-1] == (1.0, "done")

    def test_multiple_tasks_sequential(self):
        runner = LocalRunner()
        tasks = [EchoTask(TaskType.OCR, inputs={"msg": f"task-{i}"}) for i in range(3)]
        results = [runner.run_task(t, t.inputs) for t in tasks]
        assert all(r.status == TaskStatus.COMPLETED for r in results)
