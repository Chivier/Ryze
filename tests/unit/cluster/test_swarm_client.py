"""Tests for ryze.cluster.swarm_client module."""

from unittest.mock import AsyncMock, patch

import pytest

from ryze.cluster.swarm_client import SwarmClient
from ryze.exceptions import ClusterError


class TestSwarmClient:
    def test_init(self):
        client = SwarmClient(head_url="http://test:9000")
        assert client._head_url == "http://test:9000"

    def test_ensure_client_import_error(self):
        client = SwarmClient()
        with patch.dict("sys.modules", {"swarmpilot": None, "swarmpilot.sdk": None}):
            with pytest.raises(ClusterError, match="not installed"):
                client._ensure_client()

    @pytest.mark.asyncio
    async def test_serve_training_job(self, mock_swarm_client):
        client = SwarmClient()
        client._client = mock_swarm_client
        result = await client.serve_training_job("test-model", "python train.py", gpu_count=2)
        assert result["status"] == "running"

    @pytest.mark.asyncio
    async def test_get_cluster_state(self, mock_swarm_client):
        client = SwarmClient()
        client._client = mock_swarm_client
        state = await client.get_cluster_state()
        assert len(state) == 1

    @pytest.mark.asyncio
    async def test_terminate_requires_arg(self):
        client = SwarmClient()
        client._client = AsyncMock()
        with pytest.raises(ClusterError, match="Must specify"):
            await client.terminate()
