"""Base actuator interfaces and command data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class IrrigationCommand:
    """A continuous irrigation command carrying the volume to apply.

    Attributes:
        water_litres: Volume of water to apply in litres. 0.0 = no irrigation.
    """

    water_litres: float


class ActuatorInterface(ABC):
    """Abstract base class for irrigation actuators."""

    @abstractmethod
    def execute(self, command: IrrigationCommand) -> None:
        """Execute the given irrigation command."""

    @abstractmethod
    def stop(self) -> None:
        """Immediately stop all irrigation."""

    @property
    @abstractmethod
    def is_active(self) -> bool:
        """Whether irrigation is currently active."""

    def close(self) -> None:
        """Release hardware resources."""
        self.stop()
