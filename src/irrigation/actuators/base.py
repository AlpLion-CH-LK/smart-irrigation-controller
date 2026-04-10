"""Base actuator interfaces and command data classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class IrrigationAction(Enum):
    """Discrete irrigation actions that the RL agent can choose."""

    NO_IRRIGATION = 0
    IRRIGATE_SHORT = 1    # 5 minutes
    IRRIGATE_MEDIUM = 2   # 15 minutes
    IRRIGATE_LONG = 3     # 30 minutes


# Duration in seconds for each action.
ACTION_DURATIONS: dict[IrrigationAction, int] = {
    IrrigationAction.NO_IRRIGATION: 0,
    IrrigationAction.IRRIGATE_SHORT: 300,
    IrrigationAction.IRRIGATE_MEDIUM: 900,
    IrrigationAction.IRRIGATE_LONG: 1800,
}

# Approximate water volume in litres per duration (assuming 2 L/min flow rate).
FLOW_RATE_LPM = 2.0


@dataclass
class IrrigationCommand:
    """A command sent to the actuator.

    Attributes:
        action: The discrete action to execute.
        duration_seconds: Override for irrigation duration (0 = stop / no action).
    """

    action: IrrigationAction
    duration_seconds: int | None = None

    @property
    def effective_duration_seconds(self) -> int:
        """Return the actual duration in seconds for this command."""
        if self.duration_seconds is not None:
            return self.duration_seconds
        return ACTION_DURATIONS[self.action]

    @property
    def water_used_litres(self) -> float:
        """Estimate water consumed in litres."""
        return self.effective_duration_seconds / 60.0 * FLOW_RATE_LPM


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
