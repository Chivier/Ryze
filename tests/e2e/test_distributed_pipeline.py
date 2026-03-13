"""End-to-end test for distributed pipeline with mocked cluster."""

from unittest.mock import MagicMock

from ryze.core.pipeline import PipelineOrchestrator
from ryze.core.runner import DistributedRunner
from ryze.core.task import ResourceRequirement, RyzeTask, TaskResult, TaskStatus, TaskType


class CPUTask(RyzeTask):
    def __init__(self):
        super().__init__(TaskType.OCR, name="CPU Task")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"data": "processed"})


class GPUTask(RyzeTask):
    def __init__(self):
        super().__init__(TaskType.SFT_TRAIN, name="GPU Task")

    def resource_requirements(self):
        return ResourceRequirement(gpu_count=1, memory_gb=16.0)

    def validate_inputs(self):
        return True

    def execute(self, inputs):
        return TaskResult(status=TaskStatus.COMPLETED, output={"model": "trained"})


class TestDistributedPipeline:
    def test_mixed_cpu_gpu_pipeline(self):
        """CPU tasks run locally, GPU tasks attempt cluster then fallback."""
        mock_mgr = MagicMock()
        mock_mgr.acquire_instance = MagicMock(return_value=None)  # No cluster available

        runner = DistributedRunner(pylet_manager=mock_mgr)
        pipeline = PipelineOrchestrator()

        cpu_task = CPUTask()
        gpu_task = GPUTask()

        cpu_id = pipeline.add_task(cpu_task)
        pipeline.add_task(gpu_task, depends_on=[cpu_id])

        results = pipeline.run(runner=runner)
        assert all(r.status == TaskStatus.COMPLETED for r in results.values())
