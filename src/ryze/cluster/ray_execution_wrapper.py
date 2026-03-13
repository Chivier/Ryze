"""Ray remote execution wrapper for distributed task dispatch.

Bridges DistributedRunner and Ray workers via 'send trainer config,
rebuild remotely' pattern.  Trainer class path + config dict are
serialized, sent to a Ray worker, where the trainer is reconstructed
and .as_task() creates a fresh task for execution.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from ..core.task import RyzeTask, TaskResult, TaskStatus
from ..exceptions import ClusterError

logger = logging.getLogger(__name__)

# Lazy ray import — populated on first use via RayExecutionWrapper._ensure_ray()
ray: Any = None


def _execute_task_remote_impl(
    trainer_class_path: str,
    trainer_config: dict,
    inputs: dict,
) -> TaskResult:
    """Reconstruct and execute a RyzeTask on a Ray worker.

    Dynamically imports the trainer class, creates a task via .as_task(),
    and executes it.

    Args:
        trainer_class_path: Fully qualified trainer class path
            (e.g. "ryze.rl.sft_lora_trainer.RyzeSFTLoRATrainer").
        trainer_config: Serializable dict for the trainer constructor.
        inputs: Input dict for task.run().

    Returns:
        TaskResult from task execution.
    """
    module_path, class_name = trainer_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    trainer_class = getattr(module, class_name)

    trainer = trainer_class(trainer_config)
    task = trainer.as_task()
    return task.run(inputs)


# The actual @ray.remote function is created lazily (see _ensure_remote_fn)
# because ray may not be installed.
_remote_fn: Any = None


class RayExecutionWrapper:
    """Bridge between DistributedRunner and Ray workers.

    Extracts trainer class path and config from a task, submits to
    Ray worker for remote reconstruction and execution.

    Args:
        ray_manager: RayManager instance for cluster connection.
        max_retries: Maximum retry attempts on failure.
        timeout_s: Timeout in seconds for ray.get() calls.
    """

    def __init__(
        self,
        ray_manager: Any,
        max_retries: int = 3,
        timeout_s: int = 300,
    ) -> None:
        self._ray_manager = ray_manager
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._active_refs: dict[str, Any] = {}

    def _ensure_ray(self) -> Any:
        """Lazily import ray and create remote function."""
        global ray, _remote_fn
        if ray is not None:
            return ray

        # RayManager._ensure_connected() handles ray.init()
        ray_module = self._ray_manager._ensure_connected()
        ray = ray_module

        _remote_fn = ray.remote(_execute_task_remote_impl)
        return ray

    def submit(self, task: RyzeTask, inputs: dict[str, Any]) -> TaskResult:
        """Submit task to Ray worker for remote execution.

        Args:
            task: The RyzeTask to execute remotely.
            inputs: Input dict (from upstream task outputs).

        Returns:
            TaskResult from remote execution.
        """
        self._ensure_ray()
        spec = self._extract_task_spec(task)
        reqs = task.resource_requirements()

        for attempt in range(self._max_retries + 1):
            try:
                ref = _remote_fn.options(
                    num_gpus=reqs.gpu_count,
                    num_cpus=1,
                ).remote(spec["trainer_class_path"], spec["trainer_config"], inputs)
                self._active_refs[task.task_id] = ref
                result = ray.get(ref, timeout=self._timeout_s)
                self._active_refs.pop(task.task_id, None)
                return result
            except Exception as e:
                self._active_refs.pop(task.task_id, None)
                if attempt < self._max_retries:
                    logger.warning(
                        "Task %s failed (attempt %d/%d): %s, retrying...",
                        task.name,
                        attempt + 1,
                        self._max_retries + 1,
                        e,
                    )
                    continue
                logger.error(
                    "Task %s failed after %d attempts: %s",
                    task.name,
                    attempt + 1,
                    e,
                )
                return TaskResult(status=TaskStatus.FAILED, error=str(e))

        # Unreachable, but satisfy type checker
        return TaskResult(status=TaskStatus.FAILED, error="Unexpected retry loop exit")

    def cancel(self, task_id: str) -> bool:
        """Cancel an in-flight Ray task.

        Args:
            task_id: The task to cancel.

        Returns:
            True if cancellation was requested, False if task not found.
        """
        ref = self._active_refs.pop(task_id, None)
        if ref is not None and ray is not None:
            ray.cancel(ref)
            return True
        return False

    @staticmethod
    def _extract_task_spec(task: RyzeTask) -> dict[str, Any]:
        """Extract trainer class path and config from a task.

        Calls task.to_config() which must return a dict with keys:
        - "trainer_class_path": importable module-level trainer class
        - "trainer_config": dict of constructor arguments

        Args:
            task: The task to extract spec from.

        Returns:
            Dict with "trainer_class_path" and "trainer_config" keys.

        Raises:
            ClusterError: If task does not implement to_config().
        """
        if not hasattr(task, "to_config"):
            raise ClusterError(
                f"Task {task.name} does not implement to_config() — "
                f"cannot dispatch to Ray worker"
            )
        return task.to_config()
