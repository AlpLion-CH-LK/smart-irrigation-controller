"""Tests for sensor implementations."""

from __future__ import annotations

from datetime import datetime

import pytest

from irrigation.sensors.base import SensorReading
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor, SimulatedWeatherSensor


class TestSensorReading:
    def test_valid_reading(self):
        reading = SensorReading(
            soil_moisture_pct=50.0,
            temperature_celsius=25.0,
            humidity_pct=60.0,
        )
        assert reading.soil_moisture_pct == 50.0
        assert reading.temperature_celsius == 25.0

    def test_moisture_out_of_range_raises(self):
        with pytest.raises(ValueError, match="soil_moisture_pct"):
            SensorReading(soil_moisture_pct=110.0)

    def test_moisture_negative_raises(self):
        with pytest.raises(ValueError, match="soil_moisture_pct"):
            SensorReading(soil_moisture_pct=-1.0)

    def test_humidity_out_of_range_raises(self):
        with pytest.raises(ValueError, match="humidity_pct"):
            SensorReading(humidity_pct=101.0)

    def test_rainfall_negative_raises(self):
        with pytest.raises(ValueError, match="rainfall_mm"):
            SensorReading(rainfall_mm=-0.1)

    def test_timestamp_defaults_to_now(self):
        before = datetime.now()
        reading = SensorReading()
        after = datetime.now()
        assert before <= reading.timestamp <= after


class TestSimulatedSoilMoistureSensor:
    def test_initial_moisture(self):
        sensor = SimulatedSoilMoistureSensor(initial_moisture_pct=60.0, seed=42)
        reading = sensor.read()
        assert abs(reading.soil_moisture_pct - 60.0) < 2.0

    def test_irrigate_increases_moisture(self):
        sensor = SimulatedSoilMoistureSensor(initial_moisture_pct=30.0, seed=42)
        sensor.irrigate(20.0)
        assert sensor.moisture_pct == pytest.approx(50.0, abs=0.1)

    def test_irrigate_caps_at_100(self):
        sensor = SimulatedSoilMoistureSensor(initial_moisture_pct=90.0)
        sensor.irrigate(50.0)
        assert sensor.moisture_pct == pytest.approx(100.0)

    def test_apply_rain_increases_moisture(self):
        sensor = SimulatedSoilMoistureSensor(initial_moisture_pct=40.0)
        sensor.apply_rain(10.0)  # 10 mm → +5%
        assert sensor.moisture_pct == pytest.approx(45.0)

    def test_read_returns_sensor_reading(self):
        sensor = SimulatedSoilMoistureSensor(seed=42)
        reading = sensor.read()
        assert isinstance(reading, SensorReading)
        assert 0.0 <= reading.soil_moisture_pct <= 100.0


class TestSimulatedWeatherSensor:
    def test_read_returns_sensor_reading(self):
        sensor = SimulatedWeatherSensor(seed=42)
        reading = sensor.read()
        assert isinstance(reading, SensorReading)

    def test_temperature_in_plausible_range(self):
        sensor = SimulatedWeatherSensor(base_temp_celsius=28.0, temp_amplitude=5.0, seed=42)
        for _ in range(10):
            reading = sensor.read()
            assert 10.0 <= reading.temperature_celsius <= 45.0

    def test_humidity_in_valid_range(self):
        sensor = SimulatedWeatherSensor(base_humidity_pct=70.0, seed=42)
        for _ in range(10):
            reading = sensor.read()
            assert 0.0 <= reading.humidity_pct <= 100.0

    def test_set_rain_forces_raining(self):
        sensor = SimulatedWeatherSensor(seed=42)
        sensor.set_rain(True)
        readings = [sensor.read() for _ in range(5)]
        assert all(r.is_raining for r in readings)

    def test_set_rain_false_stops_rain(self):
        sensor = SimulatedWeatherSensor(seed=42)
        sensor.set_rain(False)
        # Without rainy season and rain forced off, rain probability is low.
        # With seed=42 over a few reads we should get at least some non-raining reads.
        readings = [sensor.read() for _ in range(20)]
        assert any(not r.is_raining for r in readings)
