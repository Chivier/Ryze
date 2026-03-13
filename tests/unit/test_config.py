"""Tests for ryze.config module."""

import json

import pytest

from ryze.config import RyzeConfig
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
