"""Ray Job Submission client.

Wraps the Ray Job Submission SDK so that the rest of the Ryze codebase can
schedule GPU training jobs on a Ray cluster without importing Ray directly.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from ..exceptions import ClusterError

logger = logging.getLogger(__name__)


class RayJobClient:
    """Thin wrapper around Ray's ``JobSubmissionClient``.

    Callers (e.g. ``DistributedRunner``) use this to submit and manage
    GPU training jobs on a Ray cluster.

    Parameters
    ----------
    dashboard_url:
        HTTP address of the Ray dashboard (default ``http://localhost:8265``).
    """

    def __init__(self, dashboard_url: str = "http://localhost:8265") -> None:
        """Initialise the client without connecting immediately.

        Parameters
        ----------
        dashboard_url:
            URL of the Ray dashboard used for job submission.
        """
        self._dashboard_url = dashboard_url
        self._client: Any = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_client(self) -> Any:
        """Lazily create the underlying ``JobSubmissionClient``.

        The Ray dependency is imported on first use so that the rest of Ryze
        can be imported even when Ray is not installed.

        Returns
        -------
        Any
            A live ``JobSubmissionClient`` instance.

        Raises
        ------
        ClusterError
            If the ``ray`` package is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            from ray.job_submission import JobSubmissionClient
        except ImportError as exc:
            raise ClusterError(
                "Ray is not installed. Install it with: "
                "pip install 'ray[default]'"
            ) from exc

        self._client = JobSubmissionClient(self._dashboard_url)
        logger.info("Connected to Ray dashboard at %s", self._dashboard_url)
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_training_job(
        self,
        entrypoint: str,
        gpu_count: int = 1,
        name: str | None = None,
        runtime_env: dict | None = None,
    ) -> dict[str, Any]:
        """Submit a training job to the Ray cluster.

        Parameters
        ----------
        entrypoint:
            Shell command that Ray will execute (e.g.
            ``"python train.py --epochs 5"``).
        gpu_count:
            Number of GPUs the job requires.
        name:
            Optional human-readable submission id.  When ``None`` Ray will
            generate one automatically.
        runtime_env:
            Optional Ray runtime-env dict (e.g. ``{"pip": ["torch"]}``).

        Returns
        -------
        dict[str, Any]
            ``{"job_id": <str>, "status": "PENDING"}``.

        Raises
        ------
        ClusterError
            If *entrypoint* is empty or if the submission fails.
        """
        if not entrypoint or not entrypoint.strip():
            raise ClusterError("entrypoint must not be empty")

        client = self._ensure_client()

        try:
            job_id = client.submit_job(
                entrypoint=entrypoint,
                entrypoint_num_gpus=gpu_count,
                submission_id=name,
                runtime_env=runtime_env,
            )
            logger.info("Submitted job %s (gpus=%d)", job_id, gpu_count)
            return {"job_id": job_id, "status": "PENDING"}
        except Exception as exc:
            raise ClusterError(
                f"Failed to submit training job: {exc}"
            ) from exc

    def get_cluster_state(self) -> list[dict[str, Any]]:
        """Return a snapshot of all jobs known to the Ray cluster.

        Returns
        -------
        list[dict[str, Any]]
            Each element is a dict with keys ``job_id``, ``status``, and
            ``entrypoint``.

        Raises
        ------
        ClusterError
            If the cluster cannot be queried.
        """
        client = self._ensure_client()

        try:
            jobs = client.list_jobs()
            result: list[dict[str, Any]] = []
            for job in jobs:
                status = job.status
                # Ray status objects expose a .value attribute for the
                # underlying string; fall back to str() otherwise.
                if hasattr(status, "value"):
                    status = status.value
                result.append(
                    {
                        "job_id": job.submission_id,
                        "status": status,
                        "entrypoint": job.entrypoint,
                    }
                )
            return result
        except ClusterError:
            raise
        except Exception as exc:
            raise ClusterError(
                f"Failed to query cluster state: {exc}"
            ) from exc

    def stop_job(self, job_id: str) -> bool:
        """Request cancellation of a running job.

        Parameters
        ----------
        job_id:
            The submission id of the job to stop.

        Returns
        -------
        bool
            ``True`` if the stop request was accepted, ``False`` on failure.

        Raises
        ------
        ClusterError
            If *job_id* is empty.
        """
        if not job_id or not job_id.strip():
            raise ClusterError("job_id must not be empty")

        client = self._ensure_client()

        try:
            client.stop_job(job_id)
            logger.info("Stop request sent for job %s", job_id)
            return True
        except Exception:
            logger.warning("Failed to stop job %s", job_id, exc_info=True)
            return False
