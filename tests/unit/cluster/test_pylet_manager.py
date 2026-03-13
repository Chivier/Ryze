"""Tests for ryze.cluster.pylet_manager module."""

from unittest.mock import MagicMock, patch

import pytest

from ryze.cluster.pylet_manager import PyLetManager
from ryze.exceptions import ClusterError


class TestPyLetManager:
    def test_init(self):
        mgr = PyLetManager(head_url="http://test:8000", timeout_s=60)
        assert mgr._head_url == "http://test:8000"
        assert mgr._timeout_s == 60
        assert mgr._client is None

    def test_ensure_client_import_error(self):
        mgr = PyLetManager()
        with patch.dict("sys.modules", {"swarmpilot": None, "swarmpilot.planner": None, "swarmpilot.planner.pylet": None, "swarmpilot.planner.pylet.client": None}):
            with pytest.raises(ClusterError, match="not installed"):
                mgr._ensure_client()

    @pytest.mark.asyncio
    async def test_acquire_instance(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        task = MagicMock()
        task.task_id = "test-001"
        reqs = MagicMock()
        reqs.gpu_count = 1
        reqs.memory_gb = 8.0
        result = await mgr.acquire_instance(task, reqs)
        assert result["task_id"] == "test-001"
        assert "test-001" in mgr._active_instances

    @pytest.mark.asyncio
    async def test_release_instance(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        mgr._active_instances["test-001"] = {"instance_id": "inst-001", "task_id": "test-001"}
        result = await mgr.release_instance("test-001")
        assert result is True
        assert "test-001" not in mgr._active_instances

    @pytest.mark.asyncio
    async def test_release_nonexistent_instance(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        result = await mgr.release_instance("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_switch_task(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        mgr._active_instances["old-task"] = {"instance_id": "inst-001", "task_id": "old-task"}

        new_task = MagicMock()
        new_task.task_id = "new-task"
        new_reqs = MagicMock()
        new_reqs.gpu_count = 2
        new_reqs.memory_gb = 16.0
        new_task.resource_requirements.return_value = new_reqs

        result = await mgr.switch_task("old-task", new_task)
        assert result["task_id"] == "new-task"
        assert "old-task" not in mgr._active_instances

    @pytest.mark.asyncio
    async def test_health_check(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        health = await mgr.health_check()
        assert health["healthy"] is True

    @pytest.mark.asyncio
    async def test_list_active(self, mock_pylet_client):
        mgr = PyLetManager()
        mgr._client = mock_pylet_client
        mgr._active_instances["t1"] = {"task_id": "t1", "gpu_count": 1}
        active = await mgr.list_active()
        assert len(active) == 1
