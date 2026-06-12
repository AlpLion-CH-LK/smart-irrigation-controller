# Smart Irrigation Controller — Progress Report
**Date:** 2026-05-16
**Project:** RL-Based Smart Irrigation Controller for Jaffna Chilli (*Capsicum annuum*)
**Location:** Uduvil, Jaffna, Sri Lanka (lat: 9.7432, lon: 80.0076)

---

## Project Overview

The goal of this project is to train a Reinforcement Learning (RL) agent to automatically decide **how much water to irrigate** a Jaffna chilli growing zone at every 30-minute interval across a 150-day growing season. The agent replaces manual irrigation decisions by learning from real local weather patterns and crop-specific water requirements.

---

## What Was Built This Week

### 1. Redesigned the RL Environment for PPO

**What changed:**
The original system used **Q-learning** — a method that requires all sensor inputs to be converted into discrete integer bins (e.g., soil moisture 0–100% split into 10 buckets). This approach loses information and cannot scale to a realistic number of states.

**Why it was changed:**
We switched to **Proximal Policy Optimisation (PPO)** — a modern deep RL algorithm that uses a neural network and accepts raw continuous sensor values directly. This is more suitable because:
- No information is lost from discretisation
- The state space can include more variables (growth stage, current day, humidity)
- PPO has proven performance in continuous control problems

**Domain reason:**
Chilli irrigation is a continuous problem. Soil moisture at 49.9% and 50.1% are nearly identical agronomically, but a bin boundary would treat them as completely different states. PPO handles this naturally.

**New observation vector (7 inputs to the neural network):**

| # | Sensor | Source | Normalization |
|---|---|---|---|
| 1 | Soil moisture | Capacitive sensor | ÷ 100 |
| 2 | Temperature | DHT22 | ÷ 40 (max 40°C Jaffna) |
| 3 | Humidity | DHT22 | ÷ 100 |
| 4 | Time of day | System clock | ÷ 24 |
| 5 | Is raining | Rain detector | 0 or 1 |
| 6 | Growth stage | Derived from planting date | ÷ 4 |
| 7 | Current day | System clock + planting date | ÷ 150 |

---

### 2. Stage-Aware Reward Function

**What changed:**
The reward function was updated to use **different moisture thresholds depending on the crop growth stage**, rather than a single fixed threshold for the entire 150-day season.

**Why it was changed:**
A critical bug was found: the code had two nearly identical methods (`moisture_thresholds_for_stages` and `moisture_thresholds_for_stage`) and the reward was calling the wrong one — meaning the stage-specific thresholds in `ChiliProfile` were never actually used. The reward always used the generic baseline threshold regardless of growth stage.

**Domain reason:**
Chilli has significantly different water requirements at each growth stage based on FAO Paper 56:

| Stage | Days | Optimal Moisture | Why |
|---|---|---|---|
| 0 Germination | 0–20 | 55–70% | Seeds need consistent moisture to sprout |
| 1 Vegetative | 20–60 | 50–70% | Root establishment, moderate demand |
| 2 Flowering | 60–90 | 65–75% | Most critical — water stress reduces yield significantly |
| 3 Fruit development | 90–120 | 60–75% | Fruit fill requires consistent supply |
| 4 Maturity | 120–150 | 35–50% | Reduced — dry conditions help ripening |

A single threshold cannot correctly reward the agent across all these stages.

---

### 3. Step-Based Simulation (Fixed Critical Training Bug)

**What changed:**
The soil moisture simulation was rewritten to use **fixed time steps** instead of real wall-clock time.

**Why it was changed:**
The original simulation calculated evapotranspiration (ET) using real elapsed time:
```
ET loss = et_rate × (real seconds elapsed)
```
During training, the agent runs thousands of steps per second. Real elapsed time between steps = microseconds. Therefore ET loss ≈ zero — soil moisture never dropped — and the agent learned nothing meaningful about irrigation timing.

**Domain reason:**
In reality, a chilli field loses soil moisture continuously due to evapotranspiration. The simulation must reflect this regardless of how fast the training loop runs. The fix: each simulation step always advances exactly 30 simulated minutes, applying the correct ET loss for that period.

**Additional fix:**
A new `CombinedSimulatedSensor` was added that merges soil moisture and weather into one reading — matching exactly what the real hardware configuration provides (one capacitive sensor + one DHT22 per zone).

---

### 4. Gymnasium Wrapper for PPO Training

**What changed:**
A new `IrrigationGymEnv` class was created that wraps the environment in the standard **Gymnasium** interface required by Stable-Baselines3 PPO.

**Why it was needed:**
The PPO library (Stable-Baselines3) requires a standardised environment interface with:
- Defined `observation_space` — what the agent sees
- Defined `action_space` — what the agent can do
- Standard `reset()` and `step()` methods returning specific formats

**Episode design:**
```
One episode = one full 150-day chilli growing season
Step size   = 30 simulated minutes
Steps/day   = 48
Total steps = 150 × 48 = 7,200 steps per episode
```

Each episode starts with randomised initial soil moisture (40–80%) to ensure the agent learns to handle diverse field conditions.

---

### 5. Continuous Action Space with Zone Calibration

**What changed:**
The agent's action was changed from **4 discrete choices** (no water / short / medium / long) to a **continuous value** representing any amount of water from 0L to 15L.

**Why it was changed:**
Fixed irrigation amounts cannot match the crop's actual water need at each stage. Chilli needs:
- ~5 L/day at germination
- ~12.7 L/day at flowering (peak)
- ~10.3 L/day at maturity

No fixed set of 4 buckets can satisfy all these different needs efficiently.

**Domain reason — Zone definition (Sri Lanka DoA standard):**
Based on the Sri Lanka Department of Agriculture recommendation for Jaffna chilli, one irrigation zone is defined as:

```
Zone size:    3m × 1m raised bed = 3 m²
Plants:       10–12 chilli plants (60cm × 45cm paired row spacing)
Valve:        One per zone (drip irrigation)
Sensor:       One capacitive soil moisture sensor per zone
Max water:    15L per irrigation event (safety limit)
Emergency:    Minimum 2L when soil approaches wilting point
```

**Critical calibration fix:**
A 15× calibration error was found and corrected in the simulation:

```
Before: 1 litre → 1.5% moisture increase  (completely unrealistic)
After:  1 litre → 0.1% moisture increase  (correct for 3m² zone)

Calculation:
  Soil volume = 3m² × 0.3m root depth × 1000 = 900 litres
  Drip efficiency = 90%
  1L applied → 0.9L reaches roots → 0.9/900 × 100 = 0.1% increase
```

**Water penalty redesign:**
The reward now penalises water use **relative to the zone maximum** rather than absolute litres:
```
penalty = 0.3 × (water_applied / 15L)
```
This ensures the penalty is always between 0 and 0.3 — never dominating the +0.5 reward for correct moisture maintenance.

**Emergency penalty:**
A strong −5.0 penalty was added when soil moisture falls to wilting point and the agent applies less than 2L — teaching the agent that critically dry soil must receive emergency irrigation.

---

### 6. Real Weather Data — NASA POWER (20 Years)

**What was done:**
Two Python scripts were written to download 20 years of real historical weather data for **Uduvil, Jaffna** from NASA POWER (NASA's global agrometeorological database).

**Why NASA POWER:**
- Specifically designed for agricultural applications (`community=AG`)
- Covers Jaffna with satellite + station verified data from 1984 onwards
- Free, no API key required
- Provides all variables needed for realistic irrigation simulation

**Daily data downloaded** (`data/weather/uduvil_weather_2004_2024.csv`):
- 7,671 days (2004–2024)
- Variables: T_max, T_min, humidity, rainfall, wind speed, solar radiation, ET₀

**Hourly data downloaded** (`data/weather/uduvil_per_hour/uduvil_hourly_2004_2024.csv`):
- 184,104 hourly records (2004–2024)
- Variables: temperature, humidity, rainfall, is_raining, wind speed, solar radiation, ET₀

**Key findings from real Jaffna data:**

| Metric | Value | Significance |
|---|---|---|
| Average temperature | 28.1°C | Consistently tropical — high ET demand |
| Max temperature | 35.1°C | Within DHT22 range, below 40°C model max |
| Average humidity | 77.6% | High coastal humidity reduces ET slightly |
| Average ET₀ | 6.09 mm/day | Higher than initial estimate — soil dries faster than expected |
| Max single-day rain | 141 mm | Northeast monsoon events confirmed |

**ET₀ computation:**
Since NASA POWER does not provide ET₀ directly at hourly resolution, it was computed using the **FAO-56 Penman-Monteith hourly equation** from temperature, humidity, wind speed and solar radiation — the same standard used in FAO Irrigation Paper 56 on which the chilli ETc values are based.

**How real weather data will be used in training:**
The hourly ET₀ values drive the soil moisture drain rate in the simulation — replacing the hardcoded `et_rate = 2%/hr` with real Jaffna-specific values that vary by season, time of day, and weather conditions. The agent will be trained on real Jaffna monsoon patterns, dry season heat, and typical rainfall events — not random Gaussian noise.

---

## Summary of All Files Changed or Created

| File | Status | Purpose |
|---|---|---|
| `src/irrigation/rl/environment.py` | Updated | Continuous observations, stage-aware, PPO-ready |
| `src/irrigation/rl/reward.py` | Updated | Stage-specific thresholds, zone-calibrated penalty |
| `src/irrigation/rl/gym_env.py` | New | Gymnasium wrapper, continuous action space |
| `src/irrigation/rl/__init__.py` | Updated | Removed Q-learning agent export |
| `src/irrigation/sensors/simulation.py` | Updated | Step-based simulation, CombinedSimulatedSensor |
| `src/irrigation/actuators/base.py` | Updated | Continuous IrrigationCommand(water_litres) |
| `src/irrigation/actuators/simulation.py` | Updated | Fixed moisture_per_litre calibration |
| `src/irrigation/actuators/valve.py` | Updated | Duration from water_litres ÷ flow_rate |
| `src/irrigation/crops/chili.py` | Updated | ETc data, stage_for_day(), optimal_litres_per_day() |
| `src/irrigation/crops/base.py` | Updated | Cleaned duplicate methods, added stage_for_day() |
| `src/irrigation/zone_config.py` | New | Zone physical parameters, moisture_per_litre |
| `scripts/train.py` | New | PPO training with zone arguments |
| `scripts/fetch_weather.py` | New | Downloads 20yr daily NASA POWER data |
| `scripts/fetch_weather_hourly.py` | New | Downloads 20yr hourly NASA POWER data |
| `data/weather/uduvil_weather_2004_2024.csv` | New | 7,671 days of real Jaffna weather |
| `data/weather/uduvil_per_hour/uduvil_hourly_2004_2024.csv` | New | 184,104 hourly Jaffna weather records |
| `CODEBASE_GUIDE.md` | New | 8-step reading guide for new developers |
| `requirements.txt` | Updated | Added gymnasium, stable-baselines3, requests, pandas |
| `tests/test_actuators.py` | Updated | Rewritten for continuous command API |
| `tests/test_rl_environment.py` | Updated | Rewritten for new IrrigationState |

---

## What Is Next

| Stage | Task | Reason |
|---|---|---|
| A | Build HistoricalWeatherSensor | Replace fake weather with real Jaffna data in training |
| B | Run PPO training | Train agent on real weather patterns |
| C | Evaluate vs rule-based baseline | Measure agent's improvement over simple threshold control |
