"""Tests for crop profiles."""

from __future__ import annotations

import pytest

from irrigation.crops import get_crop_profile, CROP_REGISTRY
from irrigation.crops.base import CropProfile, MoistureThresholds
from irrigation.crops.chili import ChiliProfile


class TestMoistureThresholds:
    def test_thresholds_are_ordered(self):
        chili = ChiliProfile()
        t = chili.moisture_thresholds
        assert t.wilting_point < t.stress_threshold < t.optimal_min < t.optimal_max < t.field_capacity


class TestChiliProfile:
    def setup_method(self):
        self.crop = ChiliProfile()

    def test_name(self):
        assert "chili" in self.crop.name.lower() or "pepper" in self.crop.name.lower()

    def test_stress_level_at_optimal(self):
        t = self.crop.moisture_thresholds
        mid = (t.optimal_min + t.optimal_max) / 2
        assert self.crop.stress_level(mid) == pytest.approx(0.0)

    def test_stress_level_at_wilting(self):
        t = self.crop.moisture_thresholds
        assert self.crop.stress_level(t.wilting_point) == pytest.approx(1.0)

    def test_stress_level_between_wilting_and_optimal(self):
        t = self.crop.moisture_thresholds
        mid = (t.wilting_point + t.optimal_min) / 2
        stress = self.crop.stress_level(mid)
        assert 0.0 < stress < 1.0

    def test_stress_above_optimal_is_zero(self):
        t = self.crop.moisture_thresholds
        assert self.crop.stress_level(t.optimal_max) == pytest.approx(0.0)
        assert self.crop.stress_level(t.field_capacity) == pytest.approx(0.0)

    def test_needs_irrigation_below_threshold(self):
        t = self.crop.moisture_thresholds
        assert self.crop.needs_irrigation(t.stress_threshold - 1.0)

    def test_no_irrigation_above_threshold(self):
        t = self.crop.moisture_thresholds
        assert not self.crop.needs_irrigation(t.optimal_min)

    def test_positive_water_demand(self):
        assert self.crop.peak_water_demand_mm_per_day > 0.0

    def test_positive_growing_season(self):
        assert self.crop.growing_season_days > 0


class TestCropRegistry:
    def test_chili_in_registry(self):
        assert "chili" in CROP_REGISTRY

    def test_get_crop_profile_chili(self):
        profile = get_crop_profile("chili")
        assert isinstance(profile, CropProfile)

    def test_get_crop_profile_case_insensitive(self):
        profile = get_crop_profile("CHILI")
        assert isinstance(profile, CropProfile)

    def test_get_unknown_crop_raises(self):
        with pytest.raises(ValueError, match="Unknown crop"):
            get_crop_profile("banana")
