"""Configuration management for the irrigation controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SensorConfig(BaseModel):
    """Configuration for sensor hardware."""

    soil_moisture_pin: int = Field(default=4, description="GPIO pin for soil moisture sensor")
    dht_pin: int = Field(default=17, description="GPIO pin for DHT temperature/humidity sensor")
    rain_sensor_pin: int = Field(default=27, description="GPIO pin for rain sensor")
    adc_channel: int = Field(default=0, description="ADC channel for analog soil moisture sensor")
    read_interval_seconds: int = Field(default=300, description="Sensor reading interval in seconds")


class ActuatorConfig(BaseModel):
    """Configuration for actuator hardware."""

    valve_pin: int = Field(default=22, description="GPIO pin for main irrigation valve")
    pump_pin: int = Field(default=23, description="GPIO pin for water pump")
    max_irrigation_duration_seconds: int = Field(
        default=1800, description="Maximum irrigation duration in seconds"
    )
    min_irrigation_interval_seconds: int = Field(
        default=3600, description="Minimum time between irrigation cycles in seconds"
    )


class RLConfig(BaseModel):
    """Configuration for the reinforcement learning agent."""

    learning_rate: float = Field(default=0.1, description="Q-learning rate (alpha)")
    discount_factor: float = Field(default=0.95, description="Reward discount factor (gamma)")
    exploration_rate: float = Field(default=1.0, description="Initial exploration rate (epsilon)")
    exploration_min: float = Field(default=0.05, description="Minimum exploration rate")
    exploration_decay: float = Field(
        default=0.995, description="Exploration rate decay per episode"
    )
    n_soil_moisture_bins: int = Field(
        default=10, description="Number of discrete soil moisture levels"
    )
    n_temperature_bins: int = Field(default=5, description="Number of discrete temperature bins")
    n_time_bins: int = Field(default=8, description="Number of discrete time-of-day bins")
    model_path: str = Field(
        default="models/rl_agent.pkl", description="Path to save/load the trained model"
    )


class ClimateConfig(BaseModel):
    """Configuration for climate zone."""

    name: str = Field(default="tropical", description="Climate zone name")
    region: str = Field(default="Sri Lanka", description="Target region")
    avg_daily_temp_celsius: float = Field(
        default=28.0, description="Average daily temperature in Celsius"
    )
    avg_annual_rainfall_mm: float = Field(
        default=1750.0, description="Average annual rainfall in mm"
    )
    dry_season_months: list[int] = Field(
        default=[1, 2, 3, 7, 8], description="Months of the dry season (1=January)"
    )
    rainy_season_months: list[int] = Field(
        default=[4, 5, 6, 9, 10, 11, 12], description="Months of the rainy season"
    )


class ControllerConfig(BaseModel):
    """Top-level controller configuration."""

    sensor: SensorConfig = Field(default_factory=SensorConfig)
    actuator: ActuatorConfig = Field(default_factory=ActuatorConfig)
    rl: RLConfig = Field(default_factory=RLConfig)
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    crop_profile: str = Field(default="chili", description="Crop profile to use")
    simulation_mode: bool = Field(
        default=False, description="Run in simulation mode (no hardware required)"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    data_dir: str = Field(default="data", description="Directory for data storage")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ControllerConfig":
        """Load configuration from a YAML file."""
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def default_sri_lanka(cls) -> "ControllerConfig":
        """Return a default configuration for Sri Lanka conditions."""
        return cls(
            climate=ClimateConfig(
                name="tropical",
                region="Sri Lanka",
                avg_daily_temp_celsius=28.0,
                avg_annual_rainfall_mm=1750.0,
                dry_season_months=[1, 2, 3, 7, 8],
                rainy_season_months=[4, 5, 6, 9, 10, 11, 12],
            ),
            crop_profile="chili",
        )

    @classmethod
    def default_switzerland(cls) -> "ControllerConfig":
        """Return a default configuration for Switzerland conditions."""
        return cls(
            climate=ClimateConfig(
                name="temperate",
                region="Switzerland",
                avg_daily_temp_celsius=9.0,
                avg_annual_rainfall_mm=1060.0,
                dry_season_months=[7, 8],
                rainy_season_months=[1, 2, 3, 4, 5, 6, 9, 10, 11, 12],
            ),
            crop_profile="chili",
        )
