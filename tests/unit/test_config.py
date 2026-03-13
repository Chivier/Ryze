"""Tests for ryze.config module."""

import json

import pytest

from ryze.config import ClusterConfig, RyzeConfig
from ryze.exceptions import ConfigError


class TestRyzeConfig:
    def test_default_config(self):
        config = RyzeConfig()
        assert config.cluster.mode == "local"
        assert config.training.sft.base_model_name == "microsoft/phi-2"
        assert config.evaluation.temperature == 0.7

    def test_from_json(self, tmp_path):
        data = RyzeConfig().model_dump()
        path = tmp_path / "config.json"
        path.write_text(json.dumps(data))
        loaded = RyzeConfig.from_json(path)
        assert loaded.cluster.mode == "local"
        assert loaded.training.sft.lora.r == 16

    def test_from_json_missing_file(self):
        with pytest.raises(ConfigError, match="not found"):
            RyzeConfig.from_json("/nonexistent/path.json")

    def test_from_legacy_json(self, tmp_path):
        """Legacy configs lack cluster section."""
        data = RyzeConfig().model_dump()
        del data["cluster"]
        path = tmp_path / "legacy.json"
        path.write_text(json.dumps(data))
        loaded = RyzeConfig.from_legacy_json(path)
        assert loaded.cluster.mode == "local"

    def test_to_legacy_dict(self):
        config = RyzeConfig()
        d = config.to_legacy_dict()
        assert "cluster" in d
        assert "training" in d
        assert d["training"]["sft"]["base_model_name"] == "microsoft/phi-2"

    def test_cluster_config_ray_fields(self):
        """Verify new Ray field defaults on ClusterConfig."""
        cluster = ClusterConfig()
        assert cluster.mode == "local"
        assert cluster.ray_address == "auto"
        assert cluster.ray_dashboard_url == "http://localhost:8265"
        assert cluster.timeout_s == 300
        assert cluster.max_retries == 3

    def test_from_legacy_json_migrates_pylet_url(self, tmp_path):
        """Legacy configs with pylet_head_url are migrated to Ray fields."""
        data = {
            "data_processing": RyzeConfig().data_processing.model_dump(),
            "training": RyzeConfig().training.model_dump(),
            "evaluation": RyzeConfig().evaluation.model_dump(),
            "ui": RyzeConfig().ui.model_dump(),
            "cluster": {
                "mode": "local",
                "pylet_head_url": "http://localhost:8000",
                "timeout_s": 300,
                "max_retries": 3,
            },
        }
        path = tmp_path / "legacy_pylet.json"
        path.write_text(json.dumps(data))
        loaded = RyzeConfig.from_legacy_json(path)
        assert loaded.cluster.ray_address == "auto"
        assert loaded.cluster.ray_dashboard_url == "http://localhost:8265"
        assert not hasattr(loaded.cluster, "pylet_head_url") or "pylet_head_url" not in loaded.cluster.model_dump()

    def test_cluster_config_validates(self):
        """Verify Pydantic validation of ClusterConfig with custom values."""
        cluster = ClusterConfig(
            mode="ray",
            ray_address="ray://192.168.1.100:10001",
            ray_dashboard_url="http://192.168.1.100:8265",
            timeout_s=600,
            max_retries=5,
        )
        assert cluster.mode == "ray"
        assert cluster.ray_address == "ray://192.168.1.100:10001"
        assert cluster.ray_dashboard_url == "http://192.168.1.100:8265"
        assert cluster.timeout_s == 600
        assert cluster.max_retries == 5
