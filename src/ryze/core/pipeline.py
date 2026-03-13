"""Pipeline orchestration for Ryze tasks."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from ..exceptions import PipelineError
from .task import RyzeTask, TaskResult, TaskStatus

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Manages a DAG of tasks with dependency-ordered execution."""

    def __init__(self) -> None:
        self._tasks: dict[str, RyzeTask] = {}
        self._dependencies: dict[str, list[str]] = defaultdict(list)
        self._results: dict[str, TaskResult] = {}

    def add_task(self, task: RyzeTask, depends_on: list[str] | None = None) -> str:
        """Add a task to the pipeline. Returns task_id."""
        self._tasks[task.task_id] = task
        if depends_on:
            for dep_id in depends_on:
                if dep_id not in self._tasks:
                    raise PipelineError(f"Dependency {dep_id} not found in pipeline")
                self._dependencies[task.task_id].append(dep_id)
        return task.task_id

    def _topological_sort(self) -> list[str]:
        """Return task IDs in dependency order."""
        in_degree: dict[str, int] = {tid: 0 for tid in self._tasks}
        for tid, deps in self._dependencies.items():
            in_degree[tid] = len(deps)

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        order: list[str] = []

        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for other_tid, deps in self._dependencies.items():
                if tid in deps:
                    in_degree[other_tid] -= 1
                    if in_degree[other_tid] == 0:
                        queue.append(other_tid)

        if len(order) != len(self._tasks):
            raise PipelineError("Circular dependency detected in pipeline")

        return order

    def run(self, runner: Any = None, fail_fast: bool = True) -> dict[str, TaskResult]:
        """Execute all tasks in dependency order."""
        execution_order = self._topological_sort()
        logger.info("Pipeline execution order: %s", [self._tasks[t].name for t in execution_order])

        for task_id in execution_order:
            task = self._tasks[task_id]

            # Check if dependencies succeeded
            for dep_id in self._dependencies.get(task_id, []):
                dep_result = self._results.get(dep_id)
                if dep_result and dep_result.status == TaskStatus.FAILED:
                    if fail_fast:
                        logger.error("Skipping %s: dependency %s failed", task.name, dep_id)
                        self._results[task_id] = TaskResult(
                            status=TaskStatus.CANCELLED,
                            error=f"Dependency {dep_id} failed",
                        )
                        continue

            # Collect outputs from dependencies as inputs
            dep_outputs: dict[str, Any] = {}
            for dep_id in self._dependencies.get(task_id, []):
                dep_result = self._results.get(dep_id)
                if dep_result and dep_result.output:
                    dep_outputs.update(dep_result.output)

            logger.info("Running task: %s", task.name)
            if runner:
                result = runner.run_task(task, dep_outputs)
            else:
                result = task.run(dep_outputs)
            self._results[task_id] = result
            logger.info("Task %s completed: %s", task.name, result.status.value)

            if fail_fast and result.status == TaskStatus.FAILED:
                logger.error("Pipeline halted: task %s failed: %s", task.name, result.error)
                # Mark remaining tasks as cancelled
                idx = execution_order.index(task_id)
                for remaining_id in execution_order[idx + 1:]:
                    if remaining_id not in self._results:
                        self._results[remaining_id] = TaskResult(
                            status=TaskStatus.CANCELLED,
                            error=f"Pipeline halted due to {task.name} failure",
                        )
                break

        return dict(self._results)

    @property
    def tasks(self) -> dict[str, RyzeTask]:
        return dict(self._tasks)

    @property
    def results(self) -> dict[str, TaskResult]:
        return dict(self._results)


def build_default_pipeline(pdf_path: str) -> PipelineOrchestrator:
    """Create the standard OCR -> Dataset -> SFT -> GRPO -> Eval pipeline."""
    # Defer imports to avoid circular deps and heavy module loading
    from ..data.dataset import SFTDatasetGenerator
    from ..data.ocr import PDFOCRProcessor
    from ..eval.evaluator import RyzeEvaluator
    from ..rl.grpo_trainer import RyzeGRPOTrainer
    from ..rl.sft_lora_trainer import RyzeSFTLoRATrainer

    pipeline = PipelineOrchestrator()

    ocr_task = PDFOCRProcessor().as_task(pdf_path=pdf_path)
    ocr_id = pipeline.add_task(ocr_task)

    dataset_task = SFTDatasetGenerator().as_task()
    dataset_id = pipeline.add_task(dataset_task, depends_on=[ocr_id])

    sft_task = RyzeSFTLoRATrainer().as_task()
    sft_id = pipeline.add_task(sft_task, depends_on=[dataset_id])

    grpo_task = RyzeGRPOTrainer().as_task()
    grpo_id = pipeline.add_task(grpo_task, depends_on=[sft_id])

    eval_task = RyzeEvaluator().as_task()
    pipeline.add_task(eval_task, depends_on=[grpo_id])

    return pipeline
