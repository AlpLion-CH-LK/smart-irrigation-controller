"""Tests for actuator implementations."""

from __future__ import annotations

import pytest

from irrigation.actuators.base import IrrigationAction, IrrigationCommand, ACTION_DURATIONS
from irrigation.actuators.simulation import SimulatedActuator
from irrigation.sensors.simulation import SimulatedSoilMoistureSensor


class TestIrrigationCommand:
    def test_no_irrigation_zero_duration(self):
        cmd = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        assert cmd.effective_duration_seconds == 0
        assert cmd.water_used_litres == 0.0

    def test_short_irrigation_duration(self):
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT)
        assert cmd.effective_duration_seconds == ACTION_DURATIONS[IrrigationAction.IRRIGATE_SHORT]

    def test_medium_irrigation_water_usage(self):
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_MEDIUM)
        assert cmd.water_used_litres > 0.0

    def test_custom_duration_override(self):
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT, duration_seconds=60)
        assert cmd.effective_duration_seconds == 60

    def test_all_actions_have_durations(self):
        for action in IrrigationAction:
            cmd = IrrigationCommand(action=action)
            assert cmd.effective_duration_seconds >= 0


class TestSimulatedActuator:
    def test_no_op_on_no_irrigation(self):
        actuator = SimulatedActuator()
        cmd = IrrigationCommand(action=IrrigationAction.NO_IRRIGATION)
        actuator.execute(cmd)
        assert actuator.irrigation_count() == 0

    def test_irrigation_recorded(self):
        actuator = SimulatedActuator()
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT)
        actuator.execute(cmd)
        assert actuator.irrigation_count() == 1

    def test_water_tracking(self):
        actuator = SimulatedActuator()
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_MEDIUM)
        actuator.execute(cmd)
        assert actuator.total_water_used_litres == pytest.approx(cmd.water_used_litres)

    def test_multiple_irrigations_accumulate(self):
        actuator = SimulatedActuator()
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT)
        actuator.execute(cmd)
        actuator.execute(cmd)
        assert actuator.irrigation_count() == 2
        assert actuator.total_water_used_litres == pytest.approx(2 * cmd.water_used_litres)

    def test_reset_clears_history(self):
        actuator = SimulatedActuator()
        actuator.execute(IrrigationCommand(action=IrrigationAction.IRRIGATE_SHORT))
        actuator.reset()
        assert actuator.irrigation_count() == 0
        assert actuator.total_water_used_litres == 0.0

    def test_updates_soil_sensor(self):
        soil = SimulatedSoilMoistureSensor(initial_moisture_pct=30.0, seed=42)
        actuator = SimulatedActuator(soil_sensor=soil, moisture_per_litre=2.0)
        before = soil.moisture_pct
        cmd = IrrigationCommand(action=IrrigationAction.IRRIGATE_MEDIUM)
        actuator.execute(cmd)
        assert soil.moisture_pct > before

    def test_not_active_before_execute(self):
        actuator = SimulatedActuator()
        assert not actuator.is_active

    def test_stop_does_not_raise(self):
        actuator = SimulatedActuator()
        actuator.stop()  # Should be a no-op without error.
