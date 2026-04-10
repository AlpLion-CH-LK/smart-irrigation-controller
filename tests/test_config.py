"""Tests for the controller configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from irrigation.config import (
    ActuatorConfig,
    ClimateConfig,
    ControllerConfig,
    RLConfig,
    SensorConfig,
)


class TestSensorConfig:
    def test_defaults(self):
        cfg = SensorConfig()
        assert cfg.soil_moisture_pin == 4
        assert cfg.read_interval_seconds == 300


class TestRLConfig:
    def test_exploration_rates(self):
        cfg = RLConfig()
        assert 0.0 < cfg.exploration_rate <= 1.0
        assert cfg.exploration_min < cfg.exploration_rate
        assert 0.0 < cfg.exploration_decay <= 1.0


class TestControllerConfig:
    def test_default_sri_lanka(self):
        cfg = ControllerConfig.default_sri_lanka()
        assert "sri lanka" in cfg.climate.region.lower()
        assert cfg.crop_profile == "chili"

    def test_default_switzerland(self):
        cfg = ControllerConfig.default_switzerland()
        assert "switzerland" in cfg.climate.region.lower()

    def test_yaml_round_trip(self):
        cfg = ControllerConfig.default_sri_lanka()
        cfg.simulation_mode = True

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            cfg.to_yaml(path)
            loaded = ControllerConfig.from_yaml(path)

        assert loaded.simulation_mode is True
        assert loaded.crop_profile == cfg.crop_profile
        assert loaded.climate.region == cfg.climate.region

    def test_simulation_mode_default_false(self):
        cfg = ControllerConfig()
        assert cfg.simulation_mode is False

    def test_dry_rainy_seasons_non_overlapping(self):
        cfg = ControllerConfig.default_sri_lanka()
        dry = set(cfg.climate.dry_season_months)
        rainy = set(cfg.climate.rainy_season_months)
        assert dry.isdisjoint(rainy), "Dry and rainy season months must not overlap"
