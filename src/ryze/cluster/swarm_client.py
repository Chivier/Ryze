"""SwarmPilot SDK client wrapper for cluster operations."""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..exceptions import ClusterError

logger = logging.getLogger(__name__)


class SwarmClient:
    """Wraps swarmpilot.sdk.SwarmPilotClient for high-level cluster operations."""

    def __init__(self, head_url: str = "http://localhost:8000") -> None:
        self._head_url = head_url
        self._client: Any = None

    def _ensure_client(self) -> Any:
        """Lazily initialize the SwarmPilot SDK client."""
        if self._client is None:
            try:
                from swarmpilot.sdk import SwarmPilotClient
                self._client = SwarmPilotClient(endpoint=self._head_url)
            except ImportError:
                raise ClusterError(
                    "swarmpilot package not installed. "
                    "Install with: pip install ryze[cluster]"
                )
        return self._client

    async def serve_training_job(
        self,
        model_name: str,
        command: str,
        gpu_count: int = 1,
        name: Optional[str] = None,
    ) -> dict[str, Any]:
        """Deploy a training job via the Planner SDK."""
        client = self._ensure_client()
        try:
            result = await client.serve(
                model=model_name,
                command=command,
                gpu=gpu_count,
                name=name or f"ryze-{model_name}",
            )
            logger.info("Deployed training job: %s", result)
            return result
        except Exception as e:
            raise ClusterError(f"Failed to deploy training job: {e}")

    async def get_cluster_state(self) -> list[dict[str, Any]]:
        """List all instances in the cluster."""
        client = self._ensure_client()
        try:
            instances = await client.list_instances()
            return instances
        except Exception as e:
            raise ClusterError(f"Failed to get cluster state: {e}")

    async def scale_model(self, model_name: str, replicas: int) -> dict[str, Any]:
        """Adjust replica count for a model deployment."""
        client = self._ensure_client()
        try:
            result = await client.scale(model=model_name, replicas=replicas)
            logger.info("Scaled %s to %d replicas", model_name, replicas)
            return result
        except Exception as e:
            raise ClusterError(f"Failed to scale model: {e}")

    async def terminate(
        self,
        name: Optional[str] = None,
        model: Optional[str] = None,
        all_instances: bool = False,
    ) -> bool:
        """Terminate instances by name, model, or all."""
        client = self._ensure_client()
        try:
            if all_instances:
                await client.terminate(all=True)
            elif name:
                await client.terminate(name=name)
            elif model:
                await client.terminate(model=model)
            else:
                raise ClusterError("Must specify name, model, or all_instances=True")
            logger.info("Terminated instances (name=%s, model=%s, all=%s)", name, model, all_instances)
            return True
        except ClusterError:
            raise
        except Exception as e:
            raise ClusterError(f"Failed to terminate: {e}")
