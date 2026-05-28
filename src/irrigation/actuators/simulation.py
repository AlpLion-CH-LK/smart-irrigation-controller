"""Simulated actuator for testing and development."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from irrigation.actuators.base import ActuatorInterface, IrrigationCommand


@dataclass
class IrrigationEvent:
    """Records a single irrigation event."""

    timestamp: datetime
    water_litres: float


class SimulatedActuator(ActuatorInterface):
    """An actuator that simulates irrigation without hardware.

    Notifies an optional soil sensor so the RL environment sees a realistic
    moisture response to irrigation commands.

    Args:
        soil_sensor: Optional simulated soil sensor to update on irrigation.
        moisture_per_litre: Soil moisture % increase per litre applied.
            Default 0.1 matches a 3m² zone with 0.3m root depth at 90% efficiency.
            Formula: (efficiency / (area_m2 × root_depth_m × 1000)) × 100
    """

    def __init__(
        self,
        soil_sensor: object | None = None,
        moisture_per_litre: float = 0.1,
    ) -> None:
        self.soil_sensor = soil_sensor
        self.moisture_per_litre = moisture_per_litre
        self._active: bool = False
        self.history: list[IrrigationEvent] = []
        self.total_water_used_litres: float = 0.0

    def execute(self, command: IrrigationCommand) -> None:
        if command.water_litres <= 0.0:
            return

        self._active = True
        self.history.append(
            IrrigationEvent(
                timestamp=datetime.now(),
                water_litres=command.water_litres,
            )
        )
        self.total_water_used_litres += command.water_litres

        if self.soil_sensor is not None and hasattr(self.soil_sensor, "irrigate"):
            self.soil_sensor.irrigate(command.water_litres * self.moisture_per_litre)

        self._active = False

    def stop(self) -> None:
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    def irrigation_count(self) -> int:
        """Total number of irrigation events recorded."""
        return len(self.history)

    def reset(self) -> None:
        """Reset recorded history and water usage counter."""
        self.history.clear()
        self.total_water_used_litres = 0.0
