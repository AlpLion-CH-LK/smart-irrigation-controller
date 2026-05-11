"""Sensor interfaces for the irrigation controller."""

from irrigation.sensors.base import SensorReading, SensorInterface
from irrigation.sensors.soil import SoilMoistureSensor
from irrigation.sensors.weather import WeatherSensor
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor, SimulatedWeatherSensor

__all__ = [
    "SensorReading",
    "SensorInterface",
    "SoilMoistureSensor",
    "WeatherSensor",
    "SimulatedSoilMoistureSensor",
    "SimulatedWeatherSensor",
]
