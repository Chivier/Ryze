"""Integration tests for pipeline stages."""

from ryze.core.pipeline import PipelineOrchestrator
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class StageTask(RyzeTask):
    def __init__(self, task_type, name, output_key, output_value):
        super().__init__(task_type, name=name)
        self._output_key = output_key
        self._output_value = output_value

    def resource_requirements(self):
        return ResourceRequirement()

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(
            status=TaskStatus.COMPLETED,
            output={self._output_key: self._output_value, **inputs},
        )


class TestPipelineStages:
    def test_two_stage_pipeline(self):
        pipeline = PipelineOrchestrator()
        t1 = StageTask(TaskType.OCR, "ocr", "markdown_path", "/tmp/doc.md")
        t2 = StageTask(TaskType.DATASET_GEN, "dataset", "dataset_path", "/tmp/dataset.json")
        id1 = pipeline.add_task(t1)
        pipeline.add_task(t2, depends_on=[id1])
        results = pipeline.run()
        # t2 should have received t1's output
        assert results[t2.task_id].output["markdown_path"] == "/tmp/doc.md"

    def test_three_stage_chain(self):
        pipeline = PipelineOrchestrator()
        t1 = StageTask(TaskType.OCR, "step1", "step1_out", "v1")
        t2 = StageTask(TaskType.SFT_TRAIN, "step2", "step2_out", "v2")
        t3 = StageTask(TaskType.EVALUATION, "step3", "step3_out", "v3")
        id1 = pipeline.add_task(t1)
        id2 = pipeline.add_task(t2, depends_on=[id1])
        pipeline.add_task(t3, depends_on=[id2])
        results = pipeline.run()
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())

    def test_parallel_independent_tasks(self):
        pipeline = PipelineOrchestrator()
        t1 = StageTask(TaskType.OCR, "branch_a", "a", "1")
        t2 = StageTask(TaskType.OCR, "branch_b", "b", "2")
        t3 = StageTask(TaskType.DATASET_GEN, "merge", "c", "3")
        id1 = pipeline.add_task(t1)
        id2 = pipeline.add_task(t2)
        pipeline.add_task(t3, depends_on=[id1, id2])
        results = pipeline.run()
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())
