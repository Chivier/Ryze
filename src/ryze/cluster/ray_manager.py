"""Ray cluster integration for GPU task management.

Drop-in replacement for PyLetManager that uses Ray as the distributed
execution backend instead of SwarmPilot/PyLet.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ..exceptions import ClusterError

logger = logging.getLogger(__name__)


class RayManager:
    """Manages GPU instances via Ray for task scheduling and resource allocation.

    This is the Ray-based counterpart to PyLetManager.  It lazily connects to
    an existing Ray cluster (or starts a local one) and tracks GPU assignments
    for pipeline tasks.

    Parameters:
        address: Ray cluster address.  ``"auto"`` connects to an existing
            cluster or starts a local one.
        timeout_s: Default timeout in seconds for blocking operations.
    """

    def __init__(self, address: str | None = "auto", timeout_s: int = 300) -> None:
        """Initialise the RayManager.

        Args:
            address: Ray cluster address.  Use ``"auto"`` to connect to an
                existing cluster, or ``None`` to start a local Ray runtime.
            timeout_s: Default timeout in seconds for acquire / health-check
                operations.
        """
        self._address = address
        self._timeout_s = timeout_s
        self._ray: Any = None
        self._connected: bool = False
        self._active_instances: dict[str, dict[str, Any]] = {}

    def _ensure_connected(self) -> Any:
        """Lazily import ``ray`` and initialise the connection.

        The first call will import the ``ray`` package and invoke
        ``ray.init()``.  Subsequent calls are no-ops and return the
        cached module reference.

        Returns:
            The ``ray`` module object so callers can use ``ray.*`` APIs.

        Raises:
            ClusterError: If the ``ray`` package is not installed or if
                ``ray.init`` fails.
        """
        if self._connected:
            return self._ray

        try:
            # Disable Ray's uv-run runtime env hook BEFORE importing ray.
            # The hook auto-packages the working directory into an isolated
            # venv that lacks `ray` itself, crashing workers.  The constant
            # is evaluated at `import ray` time, so env must be set first.
            import os

            os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
            import ray  # noqa: WPS433 – intentional lazy import
        except ImportError:
            raise ClusterError(
                "ray package not installed. Install with: pip install ryze[cluster]"
            )

        try:
            ray.init(address=self._address, ignore_reinit_error=True)
        except Exception as exc:
            raise ClusterError(f"Failed to initialise Ray cluster: {exc}")

        self._ray = ray
        self._connected = True
        logger.info("Connected to Ray cluster at address=%s", self._address)
        return self._ray

    def acquire_instance(
        self,
        task: Any,
        reqs: Any,
    ) -> dict[str, Any]:
        """Reserve cluster resources for *task* according to *reqs*.

        Checks the available GPUs reported by Ray and, if sufficient, records
        an allocation entry in ``_active_instances``.

        Args:
            task: A ``RyzeTask`` instance (must expose ``.task_id``).
            reqs: A ``ResourceRequirement`` instance (must expose
                ``.gpu_count`` and ``.memory_gb``).

        Returns:
            A dict with keys ``instance_id``, ``task_id``, ``gpu_count``,
            and ``status``.

        Raises:
            ClusterError: If there are not enough GPUs or if Ray raises an
                unexpected error.
        """
        ray = self._ensure_connected()
        try:
            available = ray.available_resources()
            available_gpus = int(available.get("GPU", 0))

            if available_gpus < reqs.gpu_count:
                raise ClusterError(
                    f"Insufficient GPUs for task {task.task_id}: "
                    f"requested {reqs.gpu_count}, available {available_gpus}"
                )

            instance_id = f"ray-{uuid.uuid4().hex[:8]}"
            instance_info: dict[str, Any] = {
                "instance_id": instance_id,
                "task_id": task.task_id,
                "gpu_count": reqs.gpu_count,
                "status": "running",
            }
            self._active_instances[task.task_id] = instance_info
            logger.info(
                "Acquired Ray instance %s for task %s (%d GPUs)",
                instance_id,
                task.task_id,
                reqs.gpu_count,
            )
            return instance_info
        except ClusterError:
            raise
        except Exception as exc:
            raise ClusterError(f"Failed to acquire instance: {exc}")

    def release_instance(self, task_id: str) -> bool:
        """Release the resources associated with *task_id*.

        Removes the allocation record from ``_active_instances``.  If no
        allocation exists for the given *task_id*, the method returns
        ``False`` without raising.

        Args:
            task_id: The task identifier whose resources should be freed.

        Returns:
            ``True`` if an allocation was found and removed, ``False``
            otherwise.
        """
        instance = self._active_instances.pop(task_id, None)
        if instance is None:
            logger.warning("No active instance for task %s", task_id)
            return False

        logger.info("Released Ray instance %s for task %s", instance["instance_id"], task_id)
        return True

    def switch_task(
        self,
        from_task_id: str,
        to_task: Any,
    ) -> dict[str, Any]:
        """Release resources from one task and re-acquire for another.

        This is the core dynamic-reallocation primitive: the GPUs assigned to
        *from_task_id* are freed, and a fresh allocation is made for
        *to_task* based on its ``resource_requirements()``.

        Args:
            from_task_id: The task identifier to release.
            to_task: The ``RyzeTask`` whose ``resource_requirements()`` will
                be used for the new allocation.

        Returns:
            The newly-created instance info dict (same shape as
            ``acquire_instance`` output).
        """
        logger.info("Switching from task %s to %s", from_task_id, to_task.task_id)
        self.release_instance(from_task_id)
        reqs = to_task.resource_requirements()
        return self.acquire_instance(to_task, reqs)

    def list_active(self) -> list[dict[str, Any]]:
        """Return a snapshot of all currently-tracked allocations.

        Returns:
            A list of instance-info dicts (one per active task).
        """
        return list(self._active_instances.values())

    def health_check(self) -> dict[str, Any]:
        """Probe the Ray cluster and return a health summary.

        Queries ``ray.nodes()``, ``ray.cluster_resources()``, and
        ``ray.available_resources()`` to build a status dict.

        Returns:
            A dict with ``healthy`` (bool) and, on success, ``nodes``,
            ``cluster_resources``, and ``available_resources`` keys.
            On failure the dict contains ``healthy=False`` and an ``error``
            string.
        """
        try:
            ray = self._ensure_connected()
            nodes = ray.nodes()
            cluster_resources = ray.cluster_resources()
            available_resources = ray.available_resources()
            return {
                "healthy": True,
                "nodes": nodes,
                "cluster_resources": cluster_resources,
                "available_resources": available_resources,
            }
        except Exception as exc:
            return {"healthy": False, "error": str(exc)}
