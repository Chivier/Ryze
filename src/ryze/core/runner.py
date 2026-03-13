"""Task runners for local and distributed execution."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from .task import RyzeTask, TaskResult, TaskStatus, TaskType

logger = logging.getLogger(__name__)


class TaskRunner(ABC):
    """Abstract base class for task runners."""

    @abstractmethod
    def run_task(self, task: RyzeTask, inputs: dict[str, Any]) -> TaskResult:
        """Run a task and return results."""

    @abstractmethod
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""


class LocalRunner(TaskRunner):
    """Runs tasks in the current process."""

    def __init__(self) -> None:
        self._running_tasks: dict[str, RyzeTask] = {}

    def run_task(self, task: RyzeTask, inputs: dict[str, Any]) -> TaskResult:
        self._running_tasks[task.task_id] = task
        try:
            result = task.run(inputs)
            return result
        finally:
            self._running_tasks.pop(task.task_id, None)

    def cancel_task(self, task_id: str) -> bool:
        task = self._running_tasks.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            return True
        return False


class DistributedRunner(TaskRunner):
    """Dispatches GPU tasks to PyLet instances, CPU tasks run locally."""

    # GPU-requiring task types dispatched to cluster
    _GPU_TASKS = {TaskType.SFT_TRAIN, TaskType.GRPO_TRAIN, TaskType.EVALUATION}

    def __init__(self, pylet_manager: Any = None) -> None:
        self._local_runner = LocalRunner()
        self._pylet_manager = pylet_manager

    def run_task(self, task: RyzeTask, inputs: dict[str, Any]) -> TaskResult:
        if task.task_type in self._GPU_TASKS and self._pylet_manager:
            return self._run_distributed(task, inputs)
        return self._local_runner.run_task(task, inputs)

    def _run_distributed(self, task: RyzeTask, inputs: dict[str, Any]) -> TaskResult:
        """Dispatch task to PyLet cluster."""
        logger.info("Dispatching task %s to cluster", task.name)
        try:
            reqs = task.resource_requirements()
            instance = self._pylet_manager.acquire_instance(task, reqs)
            if not instance:
                logger.warning("No cluster instance available, falling back to local")
                return self._local_runner.run_task(task, inputs)
            result = task.run(inputs)
            self._pylet_manager.release_instance(task.task_id)
            return result
        except Exception as e:
            logger.error("Distributed execution failed: %s, falling back to local", e)
            return self._local_runner.run_task(task, inputs)

    def cancel_task(self, task_id: str) -> bool:
        if self._pylet_manager:
            try:
                self._pylet_manager.release_instance(task_id)
                return True
            except Exception:
                pass
        return self._local_runner.cancel_task(task_id)
