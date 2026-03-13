"""Ryze exception hierarchy."""


class RyzeError(Exception):
    """Base exception for all Ryze errors."""


class TaskError(RyzeError):
    """Error during task execution."""


class ClusterError(RyzeError):
    """Error communicating with the cluster."""


class ConfigError(RyzeError):
    """Invalid or missing configuration."""


class PipelineError(RyzeError):
    """Error in pipeline orchestration."""
