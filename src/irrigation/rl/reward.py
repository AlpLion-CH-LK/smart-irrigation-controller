"""Reward function for the irrigation RL agent.

The reward signal balances two objectives:
1. **Plant health** – keep soil moisture in the optimal range for the crop.
2. **Water savings** – minimise irrigation water use.

A small bonus is applied when irrigation is correctly withheld during rain,
and a penalty is applied for over-watering (above field capacity).
"""

from __future__ import annotations

from irrigation.crops.base import CropProfile
from irrigation.actuators.base import IrrigationAction, IrrigationCommand


class RewardFunction:
    """Compute scalar rewards for the RL agent.

    Args:
        crop: The active crop profile providing moisture thresholds.
        water_penalty_weight: How strongly to penalise water use (default 0.3).
        stress_penalty_weight: How strongly to penalise plant stress (default 1.0).
        overwater_penalty: Penalty applied when soil is above field capacity (default 2.0).
        rain_bonus: Bonus for skipping irrigation when it is raining (default 0.5).
    """

    def __init__(
        self,
        crop: CropProfile,
        water_penalty_weight: float = 0.3,
        stress_penalty_weight: float = 1.0,
        overwater_penalty: float = 2.0,
        rain_bonus: float = 0.5,
    ) -> None:
        self.crop = crop
        self.water_penalty_weight = water_penalty_weight
        self.stress_penalty_weight = stress_penalty_weight
        self.overwater_penalty = overwater_penalty
        self.rain_bonus = rain_bonus

    def compute(
        self,
        soil_moisture_pct: float,
        command: IrrigationCommand,
        is_raining: bool = False,
    ) -> float:
        """Compute the reward for a single time-step.

        Args:
            soil_moisture_pct: Soil moisture *after* executing the command.
            command: The irrigation command that was executed.
            is_raining: Whether it was raining when the command was issued.

        Returns:
            A scalar reward value.  Positive rewards encourage good behaviour;
            negative rewards discourage bad behaviour.
        """
        reward = 0.0
        t = self.crop.moisture_thresholds

        # --- Plant health component ---
        stress = self.crop.stress_level(soil_moisture_pct)
        reward -= self.stress_penalty_weight * stress

        # Penalise over-watering (above field capacity → anaerobic root zone).
        if soil_moisture_pct > t.field_capacity:
            excess = soil_moisture_pct - t.field_capacity
            reward -= self.overwater_penalty * (excess / 10.0)

        # Small reward for maintaining optimal moisture.
        if t.optimal_min <= soil_moisture_pct <= t.optimal_max:
            reward += 0.5

        # --- Water conservation component ---
        water_used = command.water_used_litres
        reward -= self.water_penalty_weight * water_used

        # Bonus for not irrigating when it's raining.
        if is_raining and command.action == IrrigationAction.NO_IRRIGATION:
            reward += self.rain_bonus

        return round(reward, 4)
