"""Integration tests for cluster functionality."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ryze.cluster.pylet_manager import PyLetManager
from ryze.cluster.resource import GPUInfo, ResourceTracker
from ryze.core.task import ResourceRequirement


class TestClusterIntegration:
    @pytest.mark.asyncio
    async def test_deploy_and_switch(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client

        # Deploy first task
        task1 = MagicMock()
        task1.task_id = "sft-001"
        reqs1 = ResourceRequirement(gpu_count=1, memory_gb=16.0)
        await mgr.acquire_instance(task1, reqs1)
        assert "sft-001" in mgr._active_instances

        # Switch to second task
        task2 = MagicMock()
        task2.task_id = "grpo-001"
        task2.resource_requirements.return_value = ResourceRequirement(gpu_count=2, memory_gb=24.0)
        await mgr.switch_task("sft-001", task2)
        assert "sft-001" not in mgr._active_instances
        assert "grpo-001" in mgr._active_instances

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        mgr = PyLetManager()
        mgr._client = AsyncMock()
        mgr._client.health = AsyncMock(side_effect=ConnectionError("refused"))
        health = await mgr.health_check()
        assert health["healthy"] is False

    def test_resource_tracker_with_allocation(self):
        tracker = ResourceTracker()
        for i in range(4):
            tracker.register_gpu(GPUInfo(gpu_id=f"gpu-{i}", name="A100", memory_total_gb=80.0))

        assert tracker.has_capacity(4)
        tracker.allocate("task-big", 3)
        assert not tracker.has_capacity(2)
        tracker.release("task-big")
        assert tracker.has_capacity(4)
