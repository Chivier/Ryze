"""Tests for ryze.core.pipeline module."""

import pytest

from ryze.core.pipeline import PipelineOrchestrator
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType
from ryze.exceptions import PipelineError


class DummyTask(RyzeTask):
    def __init__(self, name="dummy", should_fail=False):
        super().__init__(TaskType.OCR, name=name)
        self._should_fail = should_fail

    def resource_requirements(self):
        return ResourceRequirement()

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        if self._should_fail:
            return TaskResult(status=TaskStatus.FAILED, error="intentional failure")
        return TaskResult(status=TaskStatus.COMPLETED, output={"from": self.name})


class TestPipelineOrchestrator:
    def test_add_and_run_single_task(self):
        pipeline = PipelineOrchestrator()
        task = DummyTask("single")
        pipeline.add_task(task)
        results = pipeline.run()
        assert results[task.task_id].status == TaskStatus.COMPLETED

    def test_dependency_order(self):
        pipeline = PipelineOrchestrator()
        t1 = DummyTask("first")
        t2 = DummyTask("second")
        id1 = pipeline.add_task(t1)
        pipeline.add_task(t2, depends_on=[id1])
        results = pipeline.run()
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())

    def test_fail_fast_stops_pipeline(self):
        pipeline = PipelineOrchestrator()
        t1 = DummyTask("fail", should_fail=True)
        t2 = DummyTask("skip")
        id1 = pipeline.add_task(t1)
        pipeline.add_task(t2, depends_on=[id1])
        results = pipeline.run(fail_fast=True)
        assert results[t1.task_id].status == TaskStatus.FAILED
        assert results[t2.task_id].status == TaskStatus.CANCELLED

    def test_circular_dependency_detected(self):
        pipeline = PipelineOrchestrator()
        t1 = DummyTask("a")
        t2 = DummyTask("b")
        id1 = pipeline.add_task(t1)
        pipeline.add_task(t2, depends_on=[id1])
        pipeline._dependencies[t1.task_id].append(t2.task_id)
        with pytest.raises(PipelineError, match="Circular"):
            pipeline.run()
