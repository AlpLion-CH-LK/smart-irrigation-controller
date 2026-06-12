# Codebase Reading Guide

This guide tells a new reader **where to start, what order to read the files, and what to understand at each step** before moving on.

The system is a **Reinforcement Learning based smart irrigation controller** for Jaffna chilli (*Capsicum annuum*). A PPO agent learns to decide how many litres of water to apply to a 3m² growing zone at each 30-minute step across a 150-day growing season.

---

## Step 1 — Understand the data structures first
**Read these before anything else. Everything else builds on them.**

### `src/irrigation/sensors/base.py`
- `SensorReading` — the raw data from hardware: soil moisture, temperature, humidity, rain
- `SensorInterface` — abstract class all sensors implement (one method: `read()`)

### `src/irrigation/actuators/base.py`
- `IrrigationCommand(water_litres)` — the only thing the agent sends to the hardware
- `ActuatorInterface` — abstract class: `execute(command)`, `stop()`

### `src/irrigation/zone_config.py`
- `ZoneConfig` — physical description of the field zone (area, irrigation type, root depth)
- Calculates `moisture_per_litre` from physics: how much % moisture 1 litre adds to this zone
- **Default = 3m² drip bed** matching Sri Lanka DoA Jaffna chilli recommendation

---

## Step 2 — Understand the crop knowledge
**The agent needs to know what "good" looks like for chilli at each growth stage.**

### `src/irrigation/crops/base.py`
- `MoistureThresholds` — wilting point, stress threshold, optimal range, field capacity
- `CropProfile` — abstract base: every crop must define thresholds, season length, stage boundaries
- Key methods: `stage_for_day(day)`, `moisture_thresholds_for_stage(stage)`, `stress_level_for_stage(moisture, stage)`

### `src/irrigation/crops/chili.py`
- `ChiliProfile` — concrete chilli implementation
- 5 growth stages with different moisture requirements:
  - Stage 0 Germination (day 0–20): optimal 55–70%
  - Stage 1 Vegetative (day 20–60): optimal 50–70%
  - Stage 2 Flowering (day 60–90): optimal 65–75% ← most critical
  - Stage 3 Fruit development (day 90–120): optimal 60–75%
  - Stage 4 Maturity (day 120–150): optimal 35–50%
- `optimal_litres_per_day(stage)` — ETc values from Sri Lanka DoA + FAO Paper 56

---

## Step 3 — Understand the simulation layer
**These replace real hardware during training. Read the simulated versions, not the real ones.**

### `src/irrigation/sensors/simulation.py`
- `SimulatedSoilMoistureSensor` — tracks moisture, applies ET loss per step (not wall-clock time)
- `SimulatedWeatherSensor` — generates temperature (diurnal cycle), humidity, rain stochastically
- `CombinedSimulatedSensor` — **this is what the environment uses**: merges both sensors into one `SensorReading`
- Key concept: `step_hours=0.5` means each `read()` advances 30 simulated minutes regardless of real time

### `src/irrigation/actuators/simulation.py`
- `SimulatedActuator` — records events, calls `soil_sensor.irrigate(water × moisture_per_litre)`
- `moisture_per_litre=0.1` default: calibrated for 3m² zone (1L water → 0.1% moisture increase)

---

## Step 4 — Understand the reward signal
**The reward tells the agent what "good" and "bad" behaviour looks like.**

### `src/irrigation/rl/reward.py`
- `RewardFunction.compute(soil_moisture_pct, command, is_raining, stage)`
- Returns a scalar reward each step. Components:
  - **Plant health** `-stress_level × 1.0` (0 if optimal, 1.0 if at wilting point)
  - **Over-watering** `-2.0 × excess / 10` if above field capacity
  - **Optimal moisture** `+0.5` if in stage-specific optimal range
  - **Water conservation** `-0.3 × (water_litres / 15.0)` (normalized, max penalty = -0.3)
  - **Rain bonus** `+0.5` for not irrigating when it's raining
  - **Emergency penalty** `-5.0` if soil ≤ wilting point and agent applies < 2L

---

## Step 5 — Understand the environment
**This is the core. It connects sensors, actuator, crop, and reward into one step loop.**

### `src/irrigation/rl/environment.py`

**`IrrigationState`** — what the agent observes each step:
```
soil_moisture_pct, temperature_celsius, humidity_pct,
hour, is_raining, growth_stage, current_day
```

**`IrrigationState.to_observation()`** — normalizes state to float32 array of shape (7,):
```
[moisture/100, temp/40, humidity/100, hour/24, is_raining, stage/4, day/150]
```

**`IrrigationEnvironment.step(water_litres)`** — one decision cycle:
```
1. Build IrrigationCommand(water_litres)
2. actuator.execute(command)  → valve opens, soil moisture increases
3. sensor.read()              → observe new state
4. reward_fn.compute(...)     → calculate reward
5. return (next_state, reward, done=False)
```

**Key attribute:** `_sim_day` — when set (by gym env during training), overrides real calendar so
the agent can simulate 150 days of growth in seconds.

---

## Step 6 — Understand the Gymnasium wrapper
**This adapts the environment to the interface PPO (Stable-Baselines3) expects.**

### `src/irrigation/rl/gym_env.py`

- `IrrigationGymEnv(gymnasium.Env)` wraps `IrrigationEnvironment`
- `observation_space = Box(0.0, 1.0, shape=(7,))` — 7 normalized floats
- `action_space = Box(0.0, 1.0, shape=(1,))` — agent outputs a fraction
  - `water_litres = action[0] × zone.max_litres_per_event (15L)`
- `reset()` — randomizes initial moisture (40–80%), resets sensors, sets day=0
- `step(action)` — maps fraction → litres, calls environment, returns Gymnasium-format tuple
- One episode = 150 days × 48 steps/day = **7,200 steps**

---

## Step 7 — Understand the training script
**Run this to actually train the agent.**

### `scripts/train.py`
```bash
python scripts/train.py                                    # default: drip, 3m², 500k steps
python scripts/train.py --area 6.0 --irrigation-type sprinkler
python scripts/train.py --timesteps 1000000 --n-envs 8
```
- Uses Stable-Baselines3 PPO with `MlpPolicy` (neural network)
- 4 parallel environments for faster training
- Saves best model to `models/best/`, checkpoints to `models/checkpoints/`
- TensorBoard logs to `models/logs/`

---

## Step 8 — Real hardware (deployment, not training)

### `src/irrigation/sensors/soil.py`
Real capacitive soil moisture sensor (I2C/ADC).

### `src/irrigation/actuators/valve.py`
- `ValveActuator` — controls GPIO relay to open/close valve
- `execute(command)`: opens valve for `water_litres / flow_rate_lpm × 60` seconds

---

## File dependency map

```
zone_config.py
    ↓
crops/base.py ← crops/chili.py
    ↓
sensors/base.py ← sensors/simulation.py
actuators/base.py ← actuators/simulation.py
                 ← actuators/valve.py
    ↓
rl/reward.py  (needs: crop, zone)
    ↓
rl/environment.py  (needs: sensor, actuator, crop, zone, reward)
    ↓
rl/gym_env.py  (needs: environment, zone)
    ↓
scripts/train.py  (needs: gym_env, zone)
```

---

## Key numbers to remember

| Parameter | Value | Why |
|---|---|---|
| Zone area | 3 m² | Sri Lanka DoA Jaffna chilli bed size |
| Max water/event | 15 L | Safety limit (prevents waterlogging) |
| Emergency min | 2 L | Minimum when near wilting point |
| moisture_per_litre | 0.1 %/L | Physics: 1L in 900L soil at 90% efficiency |
| Step duration | 30 min | Decision frequency |
| Episode length | 7,200 steps | Full 150-day season |
| Growing stages | 5 | Germination → Maturity |
| Observation size | 7 floats | All normalized to [0, 1] |
| Action size | 1 float | Fraction of 15L max |

---

## Where NOT to start

- `src/irrigation/rl/agent.py` — old Q-learning agent, no longer used (replaced by PPO)
- `src/irrigation/cli.py` — CLI wrapper, not relevant to RL logic
- `src/irrigation/controller.py` — rule-based controller, read only after understanding the RL system
- `src/irrigation/config.py` — configuration loader, not critical for understanding the system
