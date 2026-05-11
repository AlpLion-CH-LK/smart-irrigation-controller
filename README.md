# AlpLion Smart Irrigation Controller

Open-source, RL-powered irrigation for smallholder farmers in Sri Lanka and Switzerland.

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

## Overview

The AlpLion Smart Irrigation Controller is a fully open-source alternative to proprietary irrigation platforms. It runs on affordable Raspberry Pi hardware and uses **reinforcement learning (RL)** to optimise irrigation decisions based on real-time sensor data — minimising water use while keeping crops healthy.

| Field | Details |
| --- | --- |
| **Project Name** | AlpLion Smart Irrigation Controller |
| **Mission** | Open-source, RL-powered irrigation for smallholder farmers in Sri Lanka and Switzerland |
| **Differentiator** | Fully open-source alternative to proprietary platforms, using reinforcement learning on affordable hardware |
| **License** | Apache-2.0 |
| **Hardware Platform** | Raspberry Pi with low-cost soil/weather sensors |

## Scope & Design Decisions

| Topic | Decision |
| --- | --- |
| **Crop Scope** | Generic design supporting different crop types. First test crop: chilis. |
| **Evaluation Metrics** | Primary: water savings. Secondary (if feasible): harvest yield. |
| **Climate Context** | Generic design for different climate zones. First test region: Sri Lanka. |
| **Hardware Selection** | To be determined as part of Phase 1. |

## Features

- 🌱 **Crop-aware** — pluggable crop profiles with crop-specific moisture thresholds. Ships with a chili pepper profile.
- 🧠 **RL-powered decisions** — tabular Q-learning agent adapts irrigation scheduling to local conditions over time.
- 💧 **Water-saving reward function** — penalises unnecessary irrigation, rewards keeping soil in the optimal range.
- 🌦️ **Climate-aware** — built-in configurations for Sri Lanka (tropical) and Switzerland (temperate).
- 🖥️ **Hardware-optional** — full simulation mode lets you develop and train the agent without a Raspberry Pi.
- 📦 **Extensible** — add new crop profiles, sensor drivers, or actuator backends with minimal boilerplate.

## Quick Start

### Installation

```bash
pip install -e .
```

For hardware support on Raspberry Pi:

```bash
pip install -e ".[hardware]"
```

### Run in Simulation Mode

```bash
# Generate a default config for Sri Lanka
irrigation-controller generate-config --region sri-lanka --output config.yaml

# Train the RL agent (simulation, 500 episodes)
irrigation-controller train --simulation --episodes 500 --region sri-lanka

# Run the controller (makes irrigation decisions every 5 minutes)
irrigation-controller run --simulation --region sri-lanka
```

### Python API

```python
from irrigation.config import ControllerConfig
from irrigation.controller import IrrigationController

# Create a simulation-mode controller for Sri Lanka chili farming
config = ControllerConfig.default_sri_lanka()
config.simulation_mode = True

controller = IrrigationController(config)

# Train the RL agent
rewards = controller.train(n_episodes=500)
print(f"Training complete. Final avg reward: {sum(rewards[-100:])/100:.3f}")

# Run one decision cycle
controller.decide()
```

## Project Structure

```
smart-irrigation-controller/
├── src/irrigation/
│   ├── config.py              # Pydantic configuration models
│   ├── controller.py          # Top-level controller
│   ├── cli.py                 # Click CLI entry-point
│   ├── sensors/
│   │   ├── base.py            # SensorReading dataclass + abstract interface
│   │   ├── soil.py            # Raspberry Pi soil moisture sensor
│   │   ├── weather.py         # DHT22 temperature/humidity + rain sensor
│   │   └── simulation.py      # Simulated sensors for testing
│   ├── actuators/
│   │   ├── base.py            # IrrigationAction enum + abstract interface
│   │   ├── valve.py           # Raspberry Pi GPIO valve/pump driver
│   │   └── simulation.py      # Simulated actuator for testing
│   ├── rl/
│   │   ├── environment.py     # Gym-like RL environment
│   │   ├── agent.py           # Q-learning agent
│   │   └── reward.py          # Reward function (water savings + plant health)
│   └── crops/
│       ├── base.py            # CropProfile abstract base class
│       └── chili.py           # Chili pepper profile (first test crop)
├── tests/                     # pytest test suite (71 tests)
├── docs/
│   ├── architecture.md        # System architecture
│   └── hardware-setup.md      # Hardware wiring guide
└── pyproject.toml
```

## Reinforcement Learning Design

### State Space

The RL agent observes a **discrete state** at each decision step (every 30 minutes):

| Dimension | Values | Description |
| --- | --- | --- |
| Soil moisture bin | 0–9 (10 bins) | Discretised soil moisture % |
| Temperature bin | 0–4 (5 bins) | Discretised air temperature |
| Time of day bin | 0–7 (8 × 3 h windows) | Time of day |
| Is raining | 0 / 1 | Rain sensor reading |

### Action Space

| Action | Duration | Water used (2 L/min) |
| --- | --- | --- |
| `NO_IRRIGATION` | 0 s | 0 L |
| `IRRIGATE_SHORT` | 5 min | 10 L |
| `IRRIGATE_MEDIUM` | 15 min | 30 L |
| `IRRIGATE_LONG` | 30 min | 60 L |

### Reward Function

The reward at each step is:

```
R = +0.5 × (moisture in optimal range)
  − stress_weight × stress_level(moisture)
  − water_weight × water_used_litres
  − overwater_penalty × (moisture above field_capacity)
  + rain_bonus × (no irrigation while raining)
```

This directly incentivises the primary evaluation metric (**water savings**) while keeping the crop healthy.

## Adding a New Crop

1. Create `src/irrigation/crops/tomato.py` following the `CropProfile` interface.
2. Register it in `src/irrigation/crops/__init__.py`:
   ```python
   from irrigation.crops.tomato import TomatoProfile
   CROP_REGISTRY["tomato"] = TomatoProfile
   ```

## Hardware Setup

See [docs/hardware-setup.md](docs/hardware-setup.md) for detailed wiring instructions.

**Minimum bill of materials (Phase 1 candidate):**

| Component | Purpose | Est. cost |
| --- | --- | --- |
| Raspberry Pi Zero 2 W | Controller | ~$15 |
| Capacitive soil moisture sensor | Soil moisture reading | ~$3 |
| DHT22 sensor | Temperature & humidity | ~$4 |
| Rain detection module | Rain detection | ~$2 |
| 5 V solenoid valve | Water flow control | ~$8 |
| Relay module (1-channel) | Valve / pump switching | ~$2 |
| 12 V DC pump (optional) | Active pumping | ~$10 |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/ tests/
```

## License

[Apache License 2.0](LICENSE) — © AlpLion-CH-LK contributors.
