# ADR-001: Reinforcement Learning Approach Selection

**Project:** AlpLion Smart Irrigation Controller  
**Author:** Shriganeshan Nathiskar  
**Date:** 2025-05  
**Status:** pending

---

## Context

The AlpLion smart irrigation controller requires a reinforcement learning algorithm to learn an optimal irrigation policy for chilli pepper (Capsicum annuum) cultivation in Jaffna, Northern Sri Lanka.

### Hardware Constraints
*(Source: AlpLion Hardware Setup Guide, `doc/hardware/`)*

| Parameter | Value |
|---|---|
| Controller | Raspberry Pi Zero 2W |
| RAM | 512 MB |
| CPU | 1 GHz single-core ARM |
| GPU | None |
| Total system budget | ~$60 USD |

### Project Requirements

- Learn optimal irrigation decisions from sensor observations
- Handle multi-dimensional state space (soil moisture, temperature, humidity, time, growth stage, rain)
- Support 4 discrete irrigation actions (NO\_IRRIGATION / SHORT / MEDIUM / LONG)
- Deploy and run inference on Raspberry Pi Zero 2W
- Train within reasonable time using simulated environment
- Generalise across Jaffna tropical climate conditions (28°C average, semi-arid)

### Existing Codebase

Already implemented a Gym-inspired custom environment (`irrigation/rl/environment.py`) with:

```python
observe() → IrrigationState
step(action) → (next_state, reward, done)
state_shape → (soil_bins, temp_bins, time_bins, rain, stages)
```

This is **not a registered Gymnasium environment** — it is a custom implementation inspired by the Gym API. A standard Gymnasium wrapper will be added to enable compatibility with Stable-Baselines3.

---

## Benchmark — Existing Solutions

Three existing RL-based irrigation solutions were reviewed and benchmarked.

### Benchmark 1 — A2C Greenhouse (MDPI Agronomy, 2025)

| Parameter | Detail |
|---|---|
| Crop | Jalapeño (Capsicum annuum) |
| Algorithm | Advantage Actor-Critic (A2C) |
| Hardware | Raspberry Pi 4 + ESP32 |
| Environment | Custom real greenhouse sensors |
| State space | 3 soil moisture regions |
| Action space | Binary: irrigate / no irrigate |
| Reward | Gaussian curve centred at MAD (52.5%) |
| Water savings | **36.9%** vs time-based |
| RMSE | **2%** |

**Limitations:**
- Greenhouse in Mexico — not validated for tropical open-field
- No growth-stage-aware moisture thresholds
- A2C neural network — heavier than needed for small state space
- Binary action space — cannot modulate irrigation duration

---

### Benchmark 2 — DRLIC Deep RL in the Field (arXiv, 2023)

| Parameter | Detail |
|---|---|
| Crop | Almond trees (California) |
| Algorithm | PPO (neural network) |
| Hardware | ESP32 nodes + i5 server |
| Environment | Custom soil-water simulator |
| Node cost | $294.8 per node |
| Training | 1000 episodes (~4 hours on i5 CPU) |
| Water savings | **9.52%** vs ET-based |

**Limitations:**
- Tree crop only — not applicable to chilli
- Server-dependent — not edge-deployable
- Node cost $294.8 — too expensive for Sri Lanka smallholder farmers
- No growth-stage awareness

---

### Benchmark 3 — PPO Maize (MDPI Mathematics, 2025)

| Parameter | Detail |
|---|---|
| Crop | Maize (Nebraska, USA) |
| Algorithm | PPO (Stable-Baselines3) |
| Hardware | A100 GPU (supercomputer) |
| Environment | AquaCrop-OSPy + Gymnasium |
| State space | 26-dimensional continuous |
| Training | 2.5 million timesteps |
| Water savings | **29%** vs optimised threshold |

**Limitations:**
- Simulation only — no real hardware deployment
- Requires A100 GPU — not feasible on edge devices
- Maize only — not applicable to chilli
- No IoT hardware integration

---

## Gaps AlpLion Addresses

Based on the benchmark analysis, the following gaps exist in current solutions that AlpLion directly addresses:

| Gap | Existing Solutions | AlpLion |
|---|---|---|
| **Gap 1** — Chilli tropical climate | None validated for tropical dry-zone | Capsicum annuum, Jaffna (28°C, semi-arid) |
| **Gap 2** — Stage-aware thresholds | All use static thresholds | 5-stage Kc-based thresholds (FAO Paper 56) |
| **Gap 3** — High compute | A100 GPU / dedicated server needed | RPi Zero 2W (512 MB, no GPU) |
| **Gap 4** — High hardware cost | $294.8/node (DRLIC) | ~$60 total system |
| **Gap 5** — Tropical IoT deployment | None demonstrated | GPIO sensors + valve, Jaffna field deployment |

**AlpLion contribution:**

> AlpLion addresses all five gaps by implementing a PPO-based RL controller with growth-stage-aware chilli moisture thresholds on a Raspberry Pi Zero 2W, targeting Jaffna tropical climate conditions at ~$60 total system cost.

---

## Decision

**PPO (Proximal Policy Optimization) is selected as the RL algorithm for AlpLion.**

### Rationale

| Criteria | Assessment |
|---|---|
| Discrete action space | ✅ Native PPO support |
| Stability | ✅ Clipped objective prevents instability |
| RPi Zero inference | ✅ 64×64 network — fast, low RAM |
| Training time | ✅ Converges in hours on simulation |
| Literature support | ✅ Proven for irrigation (29–36.9% savings) |
| Stable-Baselines3 | ✅ Direct compatibility via Gymnasium wrapper |
| Hyperparameter sensitivity | ✅ Robust — fewer tuning issues than DQN/DDPG |

### Tabular Q-Learning (existing implementation)

Possible for  **curse of dimensionality**:

```
Current state space:  (10 × 5 × 8 × 2 × 5) = 4,000 states
With humidity + wind: (10 × 5 × 8 × 2 × 5 × 5 × 3) = 60,000+ states
```

As state space grows — Q-table too large, many states never visited, no generalisation across similar states. PPO neural network solves this.

### Why not other algorithms

| Algorithm | Reason Rejected |
|---|---|
| DQN | Overestimation of Q-values, unstable training |
| A2C | Less stable than PPO, requires eligibility traces |
| SAC | Replay buffer memory overhead on RPi Zero (see below) |
| DDPG / TD3 | Designed for continuous actions — ours are discrete |

---

## SAC — Future Evaluation

SAC (Soft Actor-Critic) is noted as a strong alternative for future comparison after PPO baseline is established:

| | PPO | SAC |
|---|---|---|
| Policy type | On-policy | Off-policy |
| Sample efficiency | Moderate | **Higher** |
| Stability | High | **Very high** |
| Exploration | Entropy bonus | **Automatic entropy** |
| Memory (RPi Zero) | Low | ~50 MB replay buffer |
| Irrigation literature | ✅ Proven | Limited |

**When to switch to SAC:** If PPO water savings fall below 20% target or convergence is slow after tuning.

---

## Gymnasium Environment

A **standard Gymnasium wrapper** will be added around the existing `IrrigationEnvironment` :

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class IrrigationGymEnv(gym.Env):
    """Standard Gymnasium wrapper around IrrigationEnvironment.
    Enables Stable-Baselines3 (PPO/SAC) without modifying existing code.
    """

    def __init__(self, irrigation_env: IrrigationEnvironment):
        super().__init__()
        self.env = irrigation_env
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(5,),
            dtype=np.float32,
        )

    def step(self, action: int):
        state, reward, done = self.env.step(IrrigationAction(action))
        return self._to_obs(state), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._to_obs(self.env.observe()), {}

    def _to_obs(self, state: IrrigationState) -> np.ndarray:
        return np.array([
            state.soil_moisture_bin / self.env.n_soil_bins,
            state.temperature_bin / self.env.n_temp_bins,
            state.time_bin / self.env.n_time_bins,
            float(state.is_raining),
            state.growth_stage_bin / self.env.n_stages,
        ], dtype=np.float32)
```

**Why wrapper over AquaCrop-OSPy (used in PPO Maize benchmark):**

| | AquaCrop-OSPy | Gymnasium Wrapper |
|---|---|---|
| Existing code | Replaces entirely | ✅ Kept intact |
| Real GPIO hardware | ✗ | ✅ Compatible |
| Simulated sensors | ✗ | ✅ Compatible |
| SB3 compatibility | ✅ | ✅ |
| Complexity | High | Low |

---

## Implementation Plan

```
Step 1: Add growth_stage_bin to IrrigationState        ← in progress
Step 2: Implement IrrigationGymEnv Gymnasium wrapper   ← new file
Step 3: Implement PPO agent via Stable-Baselines3      ← new file
Step 4: Train PPO in simulated environment             ← training
Step 5: Evaluate — water savings, convergence, RMSE    ← evaluation
Step 6: Deploy on Raspberry Pi Zero 2W                 ← hardware
Step 7: (Optional) Compare PPO vs SAC                  ← future
```

### PPO Hyperparameters (Starting Point)

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate | 3e-4 | SB3 default, well-tested |
| Discount factor (γ) | 0.99 | Long-term irrigation planning |
| n_steps | 2048 | One episode before update |
| Clip range (ε) | 0.2 | Standard PPO clipping |
| n_epochs | 10 | Update passes per rollout |
| Batch size | 64 | Fits RPi Zero RAM |
| Network | 64 × 64 | Lightweight for RPi Zero |

---

## Consequences

### Positive
- PPO generalises across unseen state combinations
- Gymnasium wrapper enables easy algorithm swap (PPO → SAC)
- Stable-Baselines3 provides battle-tested PPO implementation
- Stage-aware reward benefits from better generalisation
- Clear upgrade path: Q-Learning → PPO → SAC if needed

### Negative / Trade-offs
- More complex than tabular Q-Learning
- Gymnasium wrapper needs to be developed
- Neural network less interpretable than Q-table
- Training requires simulation environment (already exists)

---

## References

1. *Greenhouse Irrigation Control Based on Reinforcement Learning*, MDPI Agronomy 2025
2. *Optimizing Irrigation Efficiency using Deep Reinforcement Learning in the Field*, arXiv:2304.01435, 2023
3. *Optimizing Water Use in Maize Irrigation with Reinforcement Learning*, MDPI Mathematics 2025
4. *FAO Irrigation and Drainage Paper 56*, Allen et al., 1998
5. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017
6. Haarnoja et al., *Soft Actor-Critic*, arXiv:1801.01290, 2018
7. Stable-Baselines3 — https://stable-baselines3.readthedocs.io
