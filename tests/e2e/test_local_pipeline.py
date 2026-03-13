"""End-to-end test for local pipeline execution."""

from ryze.core.pipeline import PipelineOrchestrator
from ryze.core.runner import LocalRunner
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class MockOCRTask(RyzeTask):
    def __init__(self):
        super().__init__(TaskType.OCR, name="Mock OCR")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=0, memory_gb=1.0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"markdown_dir": "/tmp/md", "output_path": "/tmp/doc.md"})


class MockDatasetTask(RyzeTask):
    def __init__(self):
        super().__init__(TaskType.DATASET_GEN, name="Mock Dataset")

    def resource_requirements(self):
        return ResourceRequirement()

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"train_path": "/tmp/train.json", "val_path": "/tmp/val.json"})


class MockTrainTask(RyzeTask):
    def __init__(self, task_type, name):
        super().__init__(task_type, name=name)

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1, memory_gb=16.0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"merged_model_path": "/tmp/model"}, metrics={"loss": 0.1})


class MockEvalTask(RyzeTask):
    def __init__(self):
        super().__init__(TaskType.EVALUATION, name="Mock Eval")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1, memory_gb=8.0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"bleu": 0.85, "rouge-1": 0.90})


class TestLocalPipeline:
    def test_full_pipeline_with_mocks(self):
        pipeline = PipelineOrchestrator()
        runner = LocalRunner()

        ocr = MockOCRTask()
        dataset = MockDatasetTask()
        sft = MockTrainTask(TaskType.SFT_TRAIN, "Mock SFT")
        grpo = MockTrainTask(TaskType.GRPO_TRAIN, "Mock GRPO")
        evaluation = MockEvalTask()

        ocr_id = pipeline.add_task(ocr)
        ds_id = pipeline.add_task(dataset, depends_on=[ocr_id])
        sft_id = pipeline.add_task(sft, depends_on=[ds_id])
        grpo_id = pipeline.add_task(grpo, depends_on=[sft_id])
        pipeline.add_task(evaluation, depends_on=[grpo_id])

        results = pipeline.run(runner=runner)

        assert len(results) == 5
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())
        # Eval should have received model path from GRPO
        eval_result = results[evaluation.task_id]
        assert eval_result.output.get("bleu") == 0.85
