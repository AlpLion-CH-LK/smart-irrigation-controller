# System Architecture

## Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    IrrigationController                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────────┐   ┌───────────────┐  │
│  │  Sensor      │   │  RL Environment  │   │  RL Agent     │  │
│  │  Interface   │──▶│  (State / Reward)│──▶│  (Q-Learning) │  │
│  │              │   │                  │   │               │  │
│  │  SoilSensor  │   │  IrrigationEnv   │   │  QLearning    │  │
│  │  WeatherSens │   │  RewardFunction  │   │  Agent        │  │
│  └──────────────┘   └──────────────────┘   └───────┬───────┘  │
│                                                     │          │
│  ┌──────────────┐                                   │          │
│  │  Actuator    │◀──────────────────────────────────┘          │
│  │  Interface   │                                              │
│  │              │                                              │
│  │  Valve       │                                              │
│  │  Actuator    │                                              │
│  └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
[Sensors] ──read()──▶ SensorReading
                           │
                           ▼
              IrrigationEnvironment.observe()
                           │
                           ▼
                    IrrigationState
                    (discretised bins)
                           │
                           ▼
              QLearningAgent.choose_action()
                           │
                           ▼
                    IrrigationAction
                    (NO_IRRIGATION /
                     SHORT / MEDIUM / LONG)
                           │
                           ▼
              IrrigationEnvironment.step()
                    │              │
                    ▼              ▼
              [Actuator]     RewardFunction
              execute()      .compute()
                                   │
                                   ▼
              QLearningAgent.update()  ← Q-table update
```

## Module Descriptions

### `irrigation.config`
Pydantic v2 configuration models. All settings are validated on load and can
be serialised to / deserialised from YAML. Separate config sections cover:
- **SensorConfig** – GPIO pin assignments and read interval
- **ActuatorConfig** – GPIO pin assignments and irrigation limits
- **RLConfig** – Q-learning hyper-parameters
- **ClimateConfig** – Regional climate data used by simulated sensors

### `irrigation.sensors`
Hardware-agnostic sensor abstraction.

- **`SensorReading`** – validated dataclass holding one reading snapshot.
- **`SoilMoistureSensor`** – reads a capacitive/resistive sensor via an ADC (ADS1115 or similar). Falls back gracefully when hardware is absent.
- **`WeatherSensor`** – reads a DHT22 temperature/humidity sensor and a digital rain detector module.
- **`SimulatedSoilMoistureSensor`** / **`SimulatedWeatherSensor`** – physics-based simulation for offline development.

### `irrigation.actuators`
Controls the water valve (and optional pump).

- **`IrrigationAction`** – four-value enum representing the discrete action space.
- **`IrrigationCommand`** – binds an action to an optional duration override and calculates estimated water usage.
- **`ValveActuator`** – drives a GPIO relay to open a solenoid valve.
- **`SimulatedActuator`** – records irrigation events and optionally updates a `SimulatedSoilMoistureSensor`.

### `irrigation.crops`
Crop-specific parameters.

- **`CropProfile`** – abstract base with moisture thresholds, stress calculation, and irrigation trigger logic.
- **`ChiliProfile`** – concrete profile for *Capsicum annuum / frutescens* based on FAO Paper 56 recommendations.
- **`get_crop_profile(name)`** – factory function for look-up by name.

### `irrigation.rl`
Reinforcement learning components.

- **`IrrigationEnvironment`** – Gym-inspired environment. Discretises sensor readings into a compact state tuple and computes rewards.
- **`RewardFunction`** – balances plant health (stress penalty), water savings (water-use penalty), over-watering (field-capacity penalty), and weather intelligence (rain bonus).
- **`QLearningAgent`** – tabular Q-learning with ε-greedy exploration, ε decay, and pickle-based persistence.

### `irrigation.controller`
Top-level orchestrator. Wires all components together based on `ControllerConfig`. Exposes `train()`, `decide()`, and `run()` methods.

### `irrigation.cli`
Click command-line interface exposing `run`, `train`, and `generate-config` sub-commands.

## Design Principles

1. **Hardware independence** – Every hardware component has a simulated counterpart so the full stack can be tested without physical devices.
2. **Separation of concerns** – Sensors, actuators, RL logic, and configuration are cleanly separated; each can be swapped independently.
3. **Extensibility** – New crops, sensor types, and RL algorithms can be added without modifying existing code.
4. **Water savings first** – The reward function is designed to directly optimise the primary project metric.
