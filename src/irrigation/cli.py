"""Command-line interface for the smart irrigation controller."""

from __future__ import annotations

import sys

import click

from irrigation.config import ControllerConfig


@click.group()
def main() -> None:
    """AlpLion Smart Irrigation Controller CLI."""


@main.command()
@click.option("--config", "-c", default=None, help="Path to a YAML configuration file.")
@click.option("--simulation", is_flag=True, help="Run in simulation mode (no hardware needed).")
@click.option("--crop", default=None, help="Crop profile to use (e.g. 'chili').")
@click.option("--region", default=None, type=click.Choice(["sri-lanka", "switzerland"]))
def run(
    config: str | None,
    simulation: bool,
    crop: str | None,
    region: str | None,
) -> None:
    """Start the irrigation controller."""
    from irrigation.controller import IrrigationController

    cfg = _load_config(config, region)
    if simulation:
        cfg.simulation_mode = True
    if crop:
        cfg.crop_profile = crop

    controller = IrrigationController(cfg)
    controller.run()


@main.command()
@click.option("--config", "-c", default=None, help="Path to a YAML configuration file.")
@click.option("--simulation", is_flag=True, default=True, show_default=True)
@click.option("--episodes", "-n", default=500, show_default=True, help="Training episodes.")
@click.option("--region", default=None, type=click.Choice(["sri-lanka", "switzerland"]))
def train(
    config: str | None,
    simulation: bool,
    episodes: int,
    region: str | None,
) -> None:
    """Train the RL agent."""
    from irrigation.controller import IrrigationController

    cfg = _load_config(config, region)
    cfg.simulation_mode = simulation

    controller = IrrigationController(cfg)
    rewards = controller.train(n_episodes=episodes)
    avg = sum(rewards[-100:]) / min(100, len(rewards))
    click.echo(f"Training complete. Average reward (last 100 episodes): {avg:.3f}")


@main.command("generate-config")
@click.option("--region", default="sri-lanka", type=click.Choice(["sri-lanka", "switzerland"]))
@click.option("--output", "-o", default="config.yaml", show_default=True)
def generate_config(region: str, output: str) -> None:
    """Generate a default configuration YAML file."""
    if region == "sri-lanka":
        cfg = ControllerConfig.default_sri_lanka()
    else:
        cfg = ControllerConfig.default_switzerland()
    cfg.to_yaml(output)
    click.echo(f"Configuration written to {output}")


def _load_config(path: str | None, region: str | None) -> ControllerConfig:
    if path:
        return ControllerConfig.from_yaml(path)
    if region == "switzerland":
        return ControllerConfig.default_switzerland()
    return ControllerConfig.default_sri_lanka()


if __name__ == "__main__":
    main()
