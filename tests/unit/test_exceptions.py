"""Tests for ryze.exceptions module."""

from ryze.exceptions import ClusterError, ConfigError, PipelineError, RyzeError, TaskError


class TestExceptionHierarchy:
    def test_all_inherit_from_ryze_error(self):
        for exc_cls in (TaskError, ClusterError, ConfigError, PipelineError):
            assert issubclass(exc_cls, RyzeError)

    def test_can_catch_with_base(self):
        try:
            raise TaskError("test")
        except RyzeError as e:
            assert str(e) == "test"

    def test_message_formatting(self):
        err = ClusterError("connection refused to http://localhost:8000")
        assert "connection refused" in str(err)
