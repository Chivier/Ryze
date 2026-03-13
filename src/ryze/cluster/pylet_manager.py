"""PyLet integration for GPU task management and switching."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..exceptions import ClusterError

logger = logging.getLogger(__name__)


class PyLetManager:
    """Manages GPU instances via PyLet for task switching and resource adjustment.

    This is the core integration point with SwarmPilot's PyLet subsystem.
    PyLet provides the ability to deploy, release, and switch GPU tasks
    across cluster nodes.
    """

    def __init__(self, head_url: str = "http://localhost:8000", timeout_s: int = 300) -> None:
        self._head_url = head_url
        self._timeout_s = timeout_s
        self._client: Any = None
        self._active_instances: dict[str, dict[str, Any]] = {}

    def _ensure_client(self) -> Any:
        """Lazily initialize the PyLet client."""
        if self._client is None:
            try:
                from swarmpilot.planner.pylet.client import PyLetClient
                self._client = PyLetClient(head_url=self._head_url)
            except ImportError:
                raise ClusterError(
                    "swarmpilot package not installed. "
                    "Install with: pip install ryze[cluster]"
                )
        return self._client

    async def acquire_instance(
        self,
        task: Any,
        reqs: Any,
    ) -> dict[str, Any]:
        """Deploy a training job via PyLet and return instance info."""
        client = self._ensure_client()
        try:
            instance = await asyncio.wait_for(
                client.deploy(
                    task_id=task.task_id,
                    gpu_count=reqs.gpu_count,
                    memory_gb=reqs.memory_gb,
                ),
                timeout=self._timeout_s,
            )
            instance_info = {
                "instance_id": instance.get("id", task.task_id),
                "task_id": task.task_id,
                "gpu_count": reqs.gpu_count,
                "status": "running",
            }
            self._active_instances[task.task_id] = instance_info
            logger.info("Acquired instance for task %s", task.task_id)
            return instance_info
        except asyncio.TimeoutError:
            raise ClusterError(f"Timeout acquiring instance for task {task.task_id}")
        except Exception as e:
            raise ClusterError(f"Failed to acquire instance: {e}")

    async def release_instance(self, task_id: str) -> bool:
        """Release/stop an instance."""
        client = self._ensure_client()
        instance = self._active_instances.pop(task_id, None)
        if not instance:
            logger.warning("No active instance for task %s", task_id)
            return False
        try:
            await client.cancel(instance_id=instance["instance_id"])
            logger.info("Released instance for task %s", task_id)
            return True
        except Exception as e:
            logger.error("Failed to release instance for task %s: %s", task_id, e)
            return False

    async def switch_task(
        self,
        from_task_id: str,
        to_task: Any,
    ) -> dict[str, Any]:
        """Core PyLet value: release GPU from one task and reassign to another.

        This enables dynamic resource reallocation between pipeline stages.
        """
        logger.info("Switching from task %s to %s", from_task_id, to_task.task_id)
        await self.release_instance(from_task_id)
        reqs = to_task.resource_requirements()
        return await self.acquire_instance(to_task, reqs)

    async def adjust_resources(self, task_id: str, gpu_count: int) -> bool:
        """Adjust GPU allocation for a running task."""
        client = self._ensure_client()
        instance = self._active_instances.get(task_id)
        if not instance:
            raise ClusterError(f"No active instance for task {task_id}")
        try:
            await client.adjust(
                instance_id=instance["instance_id"],
                gpu_count=gpu_count,
            )
            instance["gpu_count"] = gpu_count
            logger.info("Adjusted task %s to %d GPUs", task_id, gpu_count)
            return True
        except Exception as e:
            raise ClusterError(f"Failed to adjust resources: {e}")

    async def list_active(self) -> list[dict[str, Any]]:
        """List active instances."""
        return list(self._active_instances.values())

    async def health_check(self) -> dict[str, Any]:
        """Check cluster health."""
        client = self._ensure_client()
        try:
            status = await asyncio.wait_for(
                client.health(),
                timeout=10,
            )
            return {"healthy": True, "details": status}
        except Exception as e:
            return {"healthy": False, "error": str(e)}
