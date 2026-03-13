"""Tests for ryze.cluster.ray_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from ryze.cluster.ray_manager import RayManager
from ryze.exceptions import ClusterError


class TestRayManager:
    """Unit tests for RayManager — Ray-based cluster resource manager."""

    def test_init_defaults(self):
        """Verify default constructor values: address='auto', timeout=300, not connected."""
        mgr = RayManager()
        assert mgr._address == "auto"
        assert mgr._timeout_s == 300
        assert mgr._connected is False
        assert mgr._ray is None
        assert mgr._active_instances == {}

    def test_ensure_connected_import_error(self):
        """When ray is not importable, _ensure_connected must raise ClusterError."""
        mgr = RayManager()
        with patch.dict("sys.modules", {"ray": None}):
            with pytest.raises(ClusterError, match="ray package not installed"):
                mgr._ensure_connected()

    def test_ensure_connected_success(self):
        """A successful connection should call ray.init and cache the module."""
        mock_ray = MagicMock()
        mgr = RayManager(address="ray://test:10001")
        with patch.dict("sys.modules", {"ray": mock_ray}):
            result = mgr._ensure_connected()

        mock_ray.init.assert_called_once_with(
            address="ray://test:10001", ignore_reinit_error=True
        )
        assert mgr._connected is True
        assert mgr._ray is mock_ray
        assert result is mock_ray

    def test_ensure_connected_idempotent(self):
        """Calling _ensure_connected twice should only invoke ray.init once."""
        mock_ray = MagicMock()
        mgr = RayManager()
        with patch.dict("sys.modules", {"ray": mock_ray}):
            mgr._ensure_connected()
            mgr._ensure_connected()

        mock_ray.init.assert_called_once()

    # ------------------------------------------------------------------
    # acquire_instance
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_acquire_instance_success(self):
        """Acquiring with sufficient GPUs should store instance info and return it."""
        mock_ray = MagicMock()
        mock_ray.available_resources.return_value = {"GPU": 4, "CPU": 16}

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True

        task = MagicMock()
        task.task_id = "sft_train_abc12345"
        reqs = MagicMock()
        reqs.gpu_count = 2
        reqs.memory_gb = 16.0

        result = await mgr.acquire_instance(task, reqs)

        assert result["task_id"] == "sft_train_abc12345"
        assert result["gpu_count"] == 2
        assert result["status"] == "running"
        assert result["instance_id"].startswith("ray-")
        assert "sft_train_abc12345" in mgr._active_instances

    @pytest.mark.asyncio
    async def test_acquire_instance_insufficient_gpus(self):
        """Requesting more GPUs than available should raise ClusterError."""
        mock_ray = MagicMock()
        mock_ray.available_resources.return_value = {"GPU": 2, "CPU": 16}

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True

        task = MagicMock()
        task.task_id = "grpo_train_xyz"
        reqs = MagicMock()
        reqs.gpu_count = 4

        with pytest.raises(ClusterError, match="Insufficient GPUs"):
            await mgr.acquire_instance(task, reqs)

    @pytest.mark.asyncio
    async def test_acquire_instance_exception_wrapping(self):
        """An unexpected RuntimeError from Ray should be wrapped in ClusterError."""
        mock_ray = MagicMock()
        mock_ray.available_resources.side_effect = RuntimeError("cluster gone")

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True

        task = MagicMock()
        task.task_id = "eval_task_001"
        reqs = MagicMock()
        reqs.gpu_count = 1

        with pytest.raises(ClusterError, match="Failed to acquire instance"):
            await mgr.acquire_instance(task, reqs)

    # ------------------------------------------------------------------
    # release_instance
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_release_instance_success(self):
        """Releasing a tracked instance should remove it and return True."""
        mgr = RayManager()
        mgr._active_instances["task-A"] = {
            "instance_id": "ray-aaa11111",
            "task_id": "task-A",
            "gpu_count": 1,
            "status": "running",
        }

        result = await mgr.release_instance("task-A")

        assert result is True
        assert "task-A" not in mgr._active_instances

    @pytest.mark.asyncio
    async def test_release_instance_not_found(self):
        """Releasing a non-existent task should return False without raising."""
        mgr = RayManager()
        result = await mgr.release_instance("nonexistent")
        assert result is False

    # ------------------------------------------------------------------
    # switch_task
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_switch_task(self):
        """switch_task should release old task, then acquire for the new one."""
        mock_ray = MagicMock()
        mock_ray.available_resources.return_value = {"GPU": 4}

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True
        mgr._active_instances["old-task"] = {
            "instance_id": "ray-old00000",
            "task_id": "old-task",
            "gpu_count": 2,
            "status": "running",
        }

        new_task = MagicMock()
        new_task.task_id = "new-task"
        new_reqs = MagicMock()
        new_reqs.gpu_count = 2
        new_reqs.memory_gb = 16.0
        new_task.resource_requirements.return_value = new_reqs

        result = await mgr.switch_task("old-task", new_task)

        assert result["task_id"] == "new-task"
        assert "old-task" not in mgr._active_instances
        assert "new-task" in mgr._active_instances

    # ------------------------------------------------------------------
    # list_active
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_active_empty(self):
        """An empty manager should return an empty list."""
        mgr = RayManager()
        active = await mgr.list_active()
        assert active == []

    @pytest.mark.asyncio
    async def test_list_active_with_instances(self):
        """list_active should return all tracked instances."""
        mgr = RayManager()
        mgr._active_instances["t1"] = {"task_id": "t1", "gpu_count": 1}
        mgr._active_instances["t2"] = {"task_id": "t2", "gpu_count": 2}

        active = await mgr.list_active()

        assert len(active) == 2
        task_ids = {inst["task_id"] for inst in active}
        assert task_ids == {"t1", "t2"}

    # ------------------------------------------------------------------
    # health_check
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """A healthy cluster should return healthy=True with resource details."""
        mock_ray = MagicMock()
        mock_ray.nodes.return_value = [{"NodeID": "node-1", "Alive": True}]
        mock_ray.cluster_resources.return_value = {"GPU": 8, "CPU": 32}
        mock_ray.available_resources.return_value = {"GPU": 4, "CPU": 16}

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True

        health = await mgr.health_check()

        assert health["healthy"] is True
        assert health["nodes"] == [{"NodeID": "node-1", "Alive": True}]
        assert health["cluster_resources"] == {"GPU": 8, "CPU": 32}
        assert health["available_resources"] == {"GPU": 4, "CPU": 16}

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """If Ray raises during health_check, the result should be healthy=False."""
        mock_ray = MagicMock()
        mock_ray.nodes.side_effect = RuntimeError("cluster unreachable")

        mgr = RayManager()
        mgr._ray = mock_ray
        mgr._connected = True

        health = await mgr.health_check()

        assert health["healthy"] is False
        assert "cluster unreachable" in health["error"]
