"""Task abstraction for Ryze pipeline stages."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class TaskStatus(Enum):
    PENDING = "pending"
    PREPARING = "preparing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(Enum):
    OCR = "ocr"
    DATASET_GEN = "dataset_gen"
    SFT_TRAIN = "sft_train"
    GRPO_TRAIN = "grpo_train"
    EVALUATION = "evaluation"


@dataclass
class ResourceRequirement:
    gpu_count: int = 0
    memory_gb: float = 0.0
    estimated_duration_s: float = 0.0


@dataclass
class TaskResult:
    status: TaskStatus
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class RyzeTask(ABC):
    """Abstract base class for all Ryze pipeline tasks."""

    def __init__(
        self,
        task_type: TaskType,
        inputs: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        self.task_id = f"{task_type.value}_{uuid.uuid4().hex[:8]}"
        self.task_type = task_type
        self.inputs = inputs or {}
        self.name = name or self.task_id
        self.status = TaskStatus.PENDING
        self._progress: float = 0.0
        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._start_time: Optional[float] = None

    @abstractmethod
    def resource_requirements(self) -> ResourceRequirement:
        """Return resource requirements for this task."""

    @abstractmethod
    def validate_inputs(self) -> bool:
        """Validate that all required inputs are present and valid."""

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> TaskResult:
        """Execute the task with given inputs."""

    def report_progress(self, progress: float, message: str = "") -> None:
        """Report task progress (0.0 to 1.0)."""
        self._progress = min(max(progress, 0.0), 1.0)
        if self._progress_callback:
            self._progress_callback(self._progress, message)

    def set_progress_callback(self, callback: Callable[[float, str], None]) -> None:
        self._progress_callback = callback

    def run(self, inputs: dict[str, Any] | None = None) -> TaskResult:
        """Run the task with lifecycle management."""
        run_inputs = {**self.inputs, **(inputs or {})}
        self.status = TaskStatus.PREPARING
        self._start_time = time.time()

        if not self.validate_inputs():
            self.status = TaskStatus.FAILED
            return TaskResult(
                status=TaskStatus.FAILED,
                error=f"Input validation failed for task {self.task_id}",
            )

        self.status = TaskStatus.RUNNING
        try:
            result = self.execute(run_inputs)
            self.status = result.status
            result.duration_s = time.time() - self._start_time
            return result
        except Exception as e:
            self.status = TaskStatus.FAILED
            return TaskResult(
                status=TaskStatus.FAILED,
                error=str(e),
                duration_s=time.time() - self._start_time,
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize task metadata."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "name": self.name,
            "status": self.status.value,
            "progress": self._progress,
            "inputs": {k: str(v) for k, v in self.inputs.items()},
        }
