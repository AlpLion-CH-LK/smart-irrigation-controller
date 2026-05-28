"""Central configuration loader for the smart irrigation controller.

Loads config.yaml from the project root and provides it as a dict.
All modules read their parameters from here — one place to change everything.

Usage:
    from irrigation.config_loader import load_config
    cfg = load_config()
    learning_rate = cfg["ppo"]["learning_rate"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Path to config.yaml — always at project root regardless of where this file is
_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

# Cache — config is loaded once and reused across all imports
_cache: dict[str, Any] | None = None


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the project configuration from config.yaml.

    Args:
        path: Optional custom path to a YAML file.
              Defaults to config.yaml at project root.

    Returns:
        Dict containing all configuration sections:
        zone, crop, stages, reward, training, ppo
    """
    global _cache

    # Use custom path if provided (useful for testing)
    if path is not None:
        with open(path) as f:
            return yaml.safe_load(f)

    # Return cached config on repeated calls
    if _cache is not None:
        return _cache

    if not _CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {_CONFIG_PATH}.\n"
            f"Make sure you are running from the project root."
        )

    with open(_CONFIG_PATH) as f:
        _cache = yaml.safe_load(f)

    return _cache
