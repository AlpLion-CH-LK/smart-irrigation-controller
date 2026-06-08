# RL for Irrigation Research

**Project:** AlpLion Smart Irrigation Controller  
**Author:** Shriganeshan Nathiskar  
**Date:** 2025-05  
**Status:** Draft

---

## 1. Introduction

Agricultural irrigation accounts for approximately 70% of global freshwater withdrawals. Conventional irrigation strategies — fixed schedules or simple threshold rules — do not adapt to dynamic soil conditions, weather variability, or crop growth stages. This results in either water waste or crop stress.

Reinforcement learning (RL) offers a data-driven alternative. An RL agent learns an optimal irrigation policy by interacting with the environment, without requiring an explicit model of soil-water dynamics. The agent observes the field state (soil moisture, temperature, rainfall, growth stage), selects an irrigation action, and receives a reward signal that balances plant health against water savings.

This document reviews relevant research, benchmarks three existing RL-based irrigation solutions, identifies the gaps they leave, and justifies the RL approach selected for AlpLion.

---

## 2. Crop Domain Knowledge

### 2.1 Chilli (Capsicum annuum) — Water Requirements

AlpLion targets chilli pepper cultivation in Jaffna, Northern Sri Lanka. The crop is sensitive to both drought and waterlogging.

**Moisture thresholds (FAO Paper 56):**

| Parameter | Value |
|---|---|
| Permanent Wilting Point (PWP) | 20% VWC |
| Maximum Allowable Depletion (MAD) | 52.5% VWC |
| Optimal range | 50–75% VWC |
| Field Capacity (FC) | 85% VWC |

**Crop coefficient (Kc) per growth stage (FAO Paper 56):**

| Stage | Days | Kc | Water Need |
|---|---|---|---|
| 0 — Germination | 0–20 | 0.40 | Low |
| 1 — Vegetative | 20–60 | 0.70 | Medium |
| 2 — Flowering | 60–90 | 1.05 | **High (critical)** |
| 3 — Fruit development | 90–120 | 1.00 | **High (critical)** |
| 4 — Maturity / Ripening | 120–150 | 0.85 | Decreasing |

Peak water demand: **5.18 mm/day** (flowering/fruiting stage).  
Average water demand: **3.87 mm/day**.

Water stress during flowering and fruit development significantly reduces yield. Precise irrigation is critical during stages 2 and 3.

### 2.2 Jaffna Climate Data

| Parameter | Value |
|---|---|
| Temperature range | 25.5°C – 30.4°C |
| Average temperature | 28°C year-round |
| Average humidity | 75–79% |
| Peak rainy months | October, November, December |
| Minor rainy months | April, May |
| Dry months | January–March, June–September |
| Peak ETo | 5.18 mm/day |
| Average ETo | 3.87 mm/day |

Jaffna experiences a semi-arid tropical climate. Water scarcity during dry months (January–March, June–September) makes efficient irrigation critical. Inefficient irrigation directly impacts farm profitability.

### 2.3 Common Chilli Varieties in Jaffna

Commonly cultivated varieties in Jaffna District:

- Galkiriyagama
- Super Hybrid
- Vijaya F1 Hybrid

All are Capsicum annuum — consistent with the FAO Kc values and moisture thresholds used in this project.

---

## 3. Benchmark — Existing RL Irrigation Solutions

### 3.1 Benchmark 1 — A2C Greenhouse Irrigation (MDPI Agronomy, 2025)

**Reference:** *Greenhouse Irrigation Control Based on Reinforcement Learning*, MDPI Agronomy, December 2025.  
**Link:** https://www.mdpi.com/2073-4395/15/12/2781

**Summary:**  
This study implements an Advantage Actor-Critic (A2C) RL controller for closed-loop irrigation of jalapeño green pepper (Capsicum annuum) in a greenhouse in northern Mexico. The agent regulates soil moisture near the MAD threshold using a Gaussian reward function. Performance is compared against time-based and on-off threshold controllers.

**Environment:**

| Parameter | Value |
|---|---|
| Crop | Jalapeño (Capsicum annuum) |
| Hardware | Raspberry Pi 4 + ESP32 nodes |
| Sensors | ECH2O EC-5 soil moisture, DHT temp/humidity, rain |
| Communication | MQTT WiFi |
| Algorithm | A2C with eligibility traces |
| State space | 3 soil moisture regions (Gravitational / Available / Unavailable) |
| Action space | Binary: irrigate / no irrigate |
| Reward | Gaussian curve centred at MAD (52.5%) |

**Results:**

| Strategy | RMSE | Water savings |
|---|---|---|
| Time-based | 5% | baseline |
| On-Off threshold | 3% | moderate |
| A2C (RL) | **2%** | **36.9%** |

**Gaps / Limitations:**
- Designed for greenhouse in Mexico — not validated for tropical open-field conditions
- No growth-stage-aware moisture thresholds — single threshold used across entire season
- A2C requires neural network function approximation — computationally heavier than tabular methods
- Binary action space — cannot modulate irrigation duration

---

### 3.2 Benchmark 2 — DRLIC Deep RL in the Field (arXiv, 2023)

**Reference:** *Optimizing Irrigation Efficiency using Deep Reinforcement Learning in the Field*, Ding & Du, UC Merced, arXiv:2304.01435, 2023.  
**Link:** https://arxiv.org/abs/2304.01435

**Summary:**  
DRLIC is a practical DRL-based irrigation system deployed on a testbed of six almond trees in California. It uses a PPO-based neural network agent with a custom soil-water simulator and a safety mechanism that falls back to ET-based control when the RL agent produces unsafe actions.

**Environment:**

| Parameter | Value |
|---|---|
| Crop | Almond trees (California) |
| Hardware | ESP32 nodes + i5 server |
| Sensors | Decagon EC-5 moisture (2 depths), local weather station |
| Communication | IEEE 802.15.4 wireless + WiFi |
| Algorithm | PPO (neural network, 2 layers × 256 neurons) |
| State | Soil moisture + ET + weather data |
| Action | Continuous irrigation amount per valve |
| Training | 1000 episodes (~4 hours on i5 CPU) |
| Node cost | $294.8 per node |

**Results:**

| Comparison | Water savings |
|---|---|
| vs ET-based | **9.52%** |
| vs Sensor-based | **3.79%** |
| Full season simulation | **10.21%** vs ET-based |

**Gaps / Limitations:**
- Tree crop (almond) — not applicable to vegetable/chilli crops
- Server-dependent architecture — not edge-deployable
- Node cost $294.8 — prohibitively expensive for smallholder farmers in Sri Lanka
- Neural network policy — not interpretable, harder to debug
- No growth-stage awareness

---

### 3.3 Benchmark 3 — PPO Maize Irrigation (MDPI Mathematics, 2025)

**Reference:** *Optimizing Water Use in Maize Irrigation with Reinforcement Learning*, MDPI Mathematics, February 2025.  
**Link:** https://www.mdpi.com/2227-7390/13/4/595

**Summary:**  
This study uses PPO to optimise irrigation scheduling for maize using the AquaCrop-OSPy crop growth simulator within a Gymnasium environment. Hyperparameters are tuned using Optuna. The reward function penalises cumulative irrigation while providing a terminal yield-based reward at season end.

**Environment:**

| Parameter | Value |
|---|---|
| Crop | Maize (Nebraska, USA) |
| Hardware | A100 GPU (Aziz supercomputer) |
| Simulator | AquaCrop-OSPy + Gymnasium |
| Algorithm | PPO (Stable-Baselines3) |
| State | 26-dimensional (crop, soil, weather) |
| Action | Binary: 0 mm or 25 mm |
| Training | 2.5 million timesteps |
| Discount factor | γ = 0.98 |

**Results:**

| Strategy | Yield (t/ha) | Water (mm) | Water Efficiency | Profit (USD/ha) |
|---|---|---|---|---|
| PPO | 13.80 | 179.83 | 76.76 kg/ha/mm | $576.91 |
| Threshold (SMT) | 13.95 | 255.00 | 54.72 kg/ha/mm | $528.83 |
| Interval-based | 13.81 | 380.17 | 36.32 kg/ha/mm | $377.42 |

PPO achieved **29% water savings** vs optimised threshold strategy.

**Gaps / Limitations:**
- Simulation only — no real hardware deployment
- Requires A100 GPU and 2.5 million training timesteps — not feasible on edge devices
- Maize crop only — not applicable to chilli
- Binary action space — no irrigation duration modulation
- No IoT hardware integration

---

## 4. Comparison Table

| | A2C Greenhouse | DRLIC DRL | PPO Maize | **AlpLion** |
|---|---|---|---|---|
| Crop | Jalapeño ✅ | Almond ✗ | Maize ✗ | **Chilli (Jaffna)** ✅ |
| Algorithm | A2C (neural net) | PPO (neural net) | PPO (neural net) | **Q-Learning (tabular)** |
| Hardware | RPi 4 | ESP32 + Server | A100 GPU | **RPi Zero 2W** |
| Cost | Medium | $294.8/node | Very high | **~$60 total** |
| Edge deployment | Partial | No | No | **Yes** |
| Stage-aware | No | No | No | **Yes** |
| Tropical climate | No | No | No | **Yes (Jaffna)** |
| Real deployment | Yes | Yes | No | **Yes (planned)** |
| Water savings | 36.9% | 9.52% | 29% | **Target: 20–30%** |

---

## 5. Gaps AlpLion Addresses

Based on the benchmark analysis, the following gaps exist in current solutions:

**Gap 1 — No chilli-specific implementation for tropical climates**  
All three benchmarks use non-tropical crops (jalapeño in controlled greenhouse, almond in California, maize in Nebraska). None address Capsicum annuum cultivation under tropical dry-zone conditions (Jaffna: 28°C average, semi-arid).

**Gap 2 — No growth-stage-aware moisture thresholds**  
All benchmarks use static moisture thresholds. FAO Paper 56 clearly shows that Kc varies significantly across growth stages (0.40 to 1.05 for chilli). Stage-aware thresholds are needed to avoid stress during critical flowering and fruit development stages.


**Gap 3 — High hardware cost**  
DRLIC nodes cost $294.8 each — far beyond the budget of smallholder farmers in Northern Sri Lanka. AlpLion targets a complete system cost of ~$60.

**Gap 5 — No tropical IoT edge deployment**  
No existing solution combines RL with full GPIO-based IoT hardware deployment in a tropical smallholder farming context.

---

## 6. References

1. *Greenhouse Irrigation Control Based on Reinforcement Learning*, MDPI Agronomy 2025. https://www.mdpi.com/2073-4395/15/12/2781
2. *Optimizing Irrigation Efficiency using Deep Reinforcement Learning in the Field*, Ding & Du, arXiv:2304.01435, 2023. https://arxiv.org/abs/2304.01435
3. *Optimizing Water Use in Maize Irrigation with Reinforcement Learning*, MDPI Mathematics 2025. https://www.mdpi.com/2227-7390/13/4/595
4. *FAO Irrigation and Drainage Paper 56*, Allen et al., 1998.
5. *Chilli Pepper Production Guide*, Sri Lanka Department of Agriculture.
6. *Varietal Performance of Green Chilli under different irrigation systems in Jaffna District*, Open University Sri Lanka, 2018.
