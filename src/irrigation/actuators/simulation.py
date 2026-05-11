"""Simulated actuator for testing and development.

Records the history of commands issued so that tests can assert irrigation
behaviour without physical hardware.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from irrigation.actuators.base import ActuatorInterface, IrrigationCommand


@dataclass
class IrrigationEvent:
    """Records a single irrigation event."""

    timestamp: datetime
    command: IrrigationCommand
    total_water_litres: float


class SimulatedActuator(ActuatorInterface):
    """An actuator that simulates irrigation without hardware.

    The actuator notifies an optional ``soil_sensor`` (a
    :class:`~irrigation.sensors.simulation.SimulatedSoilMoistureSensor`) so
    that the RL environment sees a realistic soil moisture response to
    irrigation commands.

    Args:
        soil_sensor: Optional simulated soil sensor to update on irrigation.
        moisture_per_litre: Soil moisture increase per litre of water applied.
    """

    def __init__(
        self,
        soil_sensor: object | None = None,
        moisture_per_litre: float = 1.5,
    ) -> None:
        self.soil_sensor = soil_sensor
        self.moisture_per_litre = moisture_per_litre
        self._active: bool = False
        self.history: list[IrrigationEvent] = []
        self.total_water_used_litres: float = 0.0

    def execute(self, command: IrrigationCommand) -> None:
        duration = command.effective_duration_seconds
        if duration <= 0:
            return

        water = command.water_used_litres
        self._active = True
        self.history.append(
            IrrigationEvent(
                timestamp=datetime.now(),
                command=command,
                total_water_litres=water,
            )
        )
        self.total_water_used_litres += water

        # Update the linked soil moisture sensor if provided.
        if self.soil_sensor is not None and hasattr(self.soil_sensor, "irrigate"):
            moisture_increase = water * self.moisture_per_litre
            self.soil_sensor.irrigate(moisture_increase)

        self._active = False

    def stop(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def irrigation_count(self) -> int:
        """Return the total number of irrigation events."""
        return len(self.history)

    def reset(self) -> None:
        """Reset the recorded history and water usage counter."""
        self.history.clear()
        self.total_water_used_litres = 0.0
