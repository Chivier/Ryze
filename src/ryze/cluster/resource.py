"""Resource tracking for cluster GPU and memory management."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    gpu_id: str
    name: str
    memory_total_gb: float
    memory_used_gb: float = 0.0
    assigned_task: str | None = None

    @property
    def memory_free_gb(self) -> float:
        return self.memory_total_gb - self.memory_used_gb

    @property
    def is_available(self) -> bool:
        return self.assigned_task is None


class ResourceTracker:
    """Track GPU and memory resources across the cluster."""

    def __init__(self) -> None:
        self._gpus: dict[str, GPUInfo] = {}

    def register_gpu(self, gpu: GPUInfo) -> None:
        self._gpus[gpu.gpu_id] = gpu

    def list_gpus(self) -> list[GPUInfo]:
        return list(self._gpus.values())

    def available_gpus(self) -> list[GPUInfo]:
        return [g for g in self._gpus.values() if g.is_available]

    def has_capacity(self, gpu_count: int, memory_gb: float = 0.0) -> bool:
        """Check if the cluster has enough free resources."""
        available = self.available_gpus()
        if len(available) < gpu_count:
            return False
        if memory_gb > 0:
            total_free = sum(g.memory_free_gb for g in available[:gpu_count])
            return total_free >= memory_gb
        return True

    def allocate(self, task_id: str, gpu_count: int) -> list[str]:
        """Allocate GPUs to a task. Returns list of GPU IDs."""
        available = self.available_gpus()
        if len(available) < gpu_count:
            return []
        allocated = available[:gpu_count]
        for gpu in allocated:
            gpu.assigned_task = task_id
        return [g.gpu_id for g in allocated]

    def release(self, task_id: str) -> int:
        """Release GPUs assigned to a task. Returns count released."""
        count = 0
        for gpu in self._gpus.values():
            if gpu.assigned_task == task_id:
                gpu.assigned_task = None
                count += 1
        return count

    def get_status(self) -> dict[str, Any]:
        """Get cluster resource status summary."""
        total = len(self._gpus)
        available = len(self.available_gpus())
        return {
            "total_gpus": total,
            "available_gpus": available,
            "used_gpus": total - available,
            "gpus": [
                {
                    "id": g.gpu_id,
                    "name": g.name,
                    "available": g.is_available,
                    "task": g.assigned_task,
                    "memory_total_gb": g.memory_total_gb,
                    "memory_free_gb": g.memory_free_gb,
                }
                for g in self._gpus.values()
            ],
        }
