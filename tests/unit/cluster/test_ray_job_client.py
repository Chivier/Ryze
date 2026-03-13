"""Unit tests for :class:`RayJobClient`.

All Ray imports are mocked so that the test suite can run without the ``ray``
package installed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ryze.cluster.ray_job_client import RayJobClient
from ryze.exceptions import ClusterError


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


class TestInit:
    """Tests for ``RayJobClient.__init__``."""

    def test_init_defaults(self) -> None:
        """Default dashboard URL is stored and client is ``None``."""
        client = RayJobClient()
        assert client._dashboard_url == "http://localhost:8265"
        assert client._client is None

    def test_init_custom_url(self) -> None:
        """A custom dashboard URL is persisted correctly."""
        client = RayJobClient(dashboard_url="http://ray:9999")
        assert client._dashboard_url == "http://ray:9999"


# ---------------------------------------------------------------------------
# _ensure_client
# ---------------------------------------------------------------------------


class TestEnsureClient:
    """Tests for the lazy client bootstrap."""

    def test_ensure_client_import_error(self) -> None:
        """``ClusterError`` is raised when ``ray`` is not installed."""
        client = RayJobClient()

        with patch.dict("sys.modules", {"ray": None, "ray.job_submission": None}):
            # Force re-import to trigger ImportError
            with patch(
                "ryze.cluster.ray_job_client.RayJobClient._ensure_client",
                wraps=client._ensure_client,
            ):
                # Directly patch the import inside _ensure_client
                with patch("builtins.__import__", side_effect=ImportError("no ray")):
                    with pytest.raises(ClusterError, match="Ray is not installed"):
                        client._ensure_client()

    def test_ensure_client_success(self) -> None:
        """Client is created and cached on first call."""
        mock_jsc_class = MagicMock()
        mock_jsc_instance = MagicMock()
        mock_jsc_class.return_value = mock_jsc_instance

        client = RayJobClient(dashboard_url="http://ray:8265")

        with patch.dict(
            "sys.modules",
            {
                "ray": MagicMock(),
                "ray.job_submission": MagicMock(
                    JobSubmissionClient=mock_jsc_class,
                ),
            },
        ):
            result = client._ensure_client()

        assert result is mock_jsc_instance
        mock_jsc_class.assert_called_once_with("http://ray:8265")
        # Subsequent call returns the cached client without re-importing.
        assert client._ensure_client() is mock_jsc_instance


# ---------------------------------------------------------------------------
# submit_training_job
# ---------------------------------------------------------------------------


class TestSubmitTrainingJob:
    """Tests for job submission."""

    def _make_client_with_mock(self) -> tuple[RayJobClient, MagicMock]:
        """Return a ``RayJobClient`` whose internal client is a mock."""
        client = RayJobClient()
        mock_inner = MagicMock()
        client._client = mock_inner
        return client, mock_inner

    def test_submit_training_job_success(self) -> None:
        """Successful submission returns a dict with job_id and status."""
        client, mock_inner = self._make_client_with_mock()
        mock_inner.submit_job.return_value = "raysubmit_123"

        result = client.submit_training_job(
            entrypoint="python train.py",
            gpu_count=2,
            name="my-job",
            runtime_env={"pip": ["torch"]},
        )

        assert result == {"job_id": "raysubmit_123", "status": "PENDING"}
        mock_inner.submit_job.assert_called_once_with(
            entrypoint="python train.py",
            entrypoint_num_gpus=2,
            submission_id="my-job",
            runtime_env={"pip": ["torch"]},
        )

    def test_submit_training_job_empty_entrypoint(self) -> None:
        """``ClusterError`` is raised when entrypoint is empty."""
        client, _ = self._make_client_with_mock()

        with pytest.raises(ClusterError, match="entrypoint must not be empty"):
            client.submit_training_job(entrypoint="")

        with pytest.raises(ClusterError, match="entrypoint must not be empty"):
            client.submit_training_job(entrypoint="   ")

    def test_submit_training_job_failure(self) -> None:
        """``ClusterError`` wraps unexpected submission failures."""
        client, mock_inner = self._make_client_with_mock()
        mock_inner.submit_job.side_effect = RuntimeError("connection refused")

        with pytest.raises(ClusterError, match="Failed to submit training job"):
            client.submit_training_job(entrypoint="python train.py")


# ---------------------------------------------------------------------------
# get_cluster_state
# ---------------------------------------------------------------------------


class TestGetClusterState:
    """Tests for cluster state retrieval."""

    def _make_client_with_mock(self) -> tuple[RayJobClient, MagicMock]:
        client = RayJobClient()
        mock_inner = MagicMock()
        client._client = mock_inner
        return client, mock_inner

    def test_get_cluster_state(self) -> None:
        """Jobs are transformed into plain dicts."""
        client, mock_inner = self._make_client_with_mock()

        job1 = MagicMock()
        job1.submission_id = "raysubmit_1"
        job1.status = MagicMock(value="RUNNING")
        job1.entrypoint = "python train.py"

        job2 = MagicMock()
        job2.submission_id = "raysubmit_2"
        job2.entrypoint = "python eval.py"
        # Use a plain string for status so hasattr(status, "value") is False,
        # exercising the non-.value code path.
        job2.status = "SUCCEEDED"

        mock_inner.list_jobs.return_value = [job1, job2]

        result = client.get_cluster_state()

        assert result == [
            {
                "job_id": "raysubmit_1",
                "status": "RUNNING",
                "entrypoint": "python train.py",
            },
            {
                "job_id": "raysubmit_2",
                "status": "SUCCEEDED",
                "entrypoint": "python eval.py",
            },
        ]

    def test_get_cluster_state_failure(self) -> None:
        """``ClusterError`` is raised on query failure."""
        client, mock_inner = self._make_client_with_mock()
        mock_inner.list_jobs.side_effect = RuntimeError("timeout")

        with pytest.raises(ClusterError, match="Failed to query cluster state"):
            client.get_cluster_state()


# ---------------------------------------------------------------------------
# stop_job
# ---------------------------------------------------------------------------


class TestStopJob:
    """Tests for job cancellation."""

    def _make_client_with_mock(self) -> tuple[RayJobClient, MagicMock]:
        client = RayJobClient()
        mock_inner = MagicMock()
        client._client = mock_inner
        return client, mock_inner

    def test_stop_job_success(self) -> None:
        """Returns ``True`` when the stop request succeeds."""
        client, mock_inner = self._make_client_with_mock()

        assert client.stop_job("raysubmit_1") is True
        mock_inner.stop_job.assert_called_once_with("raysubmit_1")

    def test_stop_job_failure(self) -> None:
        """Returns ``False`` when the stop request fails."""
        client, mock_inner = self._make_client_with_mock()
        mock_inner.stop_job.side_effect = RuntimeError("not found")

        assert client.stop_job("raysubmit_1") is False

    def test_stop_job_empty_id(self) -> None:
        """``ClusterError`` is raised when job_id is empty."""
        client, _ = self._make_client_with_mock()

        with pytest.raises(ClusterError, match="job_id must not be empty"):
            client.stop_job("")

        with pytest.raises(ClusterError, match="job_id must not be empty"):
            client.stop_job("   ")
