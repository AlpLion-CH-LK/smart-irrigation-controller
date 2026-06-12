"""Custom training callbacks for the smart irrigation PPO agent.

IrrigationMonitorCallback collects per-episode stats and prints a detailed
summary every N episodes showing:
  - Plant survival rate and death causes
  - Growth stage reached at episode end
  - Plant health scores (optimal/stress/critical hours)
  - Water usage per season
  - TensorBoard logging of all metrics
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


STAGE_NAMES = {
    0: "Germination",
    1: "Vegetative",
    2: "Flowering",
    3: "Fruit dev",
    4: "Maturity",
}


class IrrigationMonitorCallback(BaseCallback):
    """Prints and logs detailed per-episode plant health and training stats.

    Args:
        log_interval: Print summary every this many episodes.
        verbose:      0 = silent, 1 = print summaries.
    """

    def __init__(self, log_interval: int = 10, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.log_interval   = log_interval
        self._n_episodes    = 0
        self._buffer: list[dict] = []

    def _on_step(self) -> bool:
        """Called every training step. Collects stats when episode ends."""
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                self._collect(info)
        return True

    def _collect(self, info: dict) -> None:
        """Store episode stats and trigger summary when interval reached."""
        self._n_episodes += 1

        # Only collect if terminal info keys are present
        if "overall_health_score" not in info:
            return

        self._buffer.append({
            "plant_dead":    info.get("plant_dead", False),
            "death_cause":   info.get("death_cause", ""),
            "death_stage":   info.get("death_stage", -1),
            "final_stage":   info.get("final_dynamic_stage", info.get("dynamic_stage", 0)),
            "health_score":  info.get("overall_health_score", 0.0),
            "optimal_hrs":   info.get("optimal_hours", 0),
            "stress_hrs":    info.get("stress_hours", 0),
            "critical_hrs":  info.get("critical_hours", 0),
            "waterlog_hrs":  info.get("waterlog_hours", 0),
            "water_litres":  info.get("total_water_litres", 0.0),
            "phase":         info.get("phase", 1),
        })

        if self._n_episodes % self.log_interval == 0:
            self._print_summary()
            self._log_tensorboard()
            self._buffer.clear()

    def _print_summary(self) -> None:
        n = len(self._buffer)
        if n == 0 or self.verbose == 0:
            return

        deaths   = [e for e in self._buffer if e["plant_dead"]]
        survived = [e for e in self._buffer if not e["plant_dead"]]
        phase    = self._buffer[-1]["phase"]
        phase_label = "Phase 1 — dry months" if phase == 1 else "Phase 2 — all months"

        # Stage distribution
        stage_counts: dict[int, int] = defaultdict(int)
        for e in self._buffer:
            stage_counts[e["final_stage"]] += 1

        # Death breakdown
        drought_deaths   = sum(1 for e in deaths if e["death_cause"] == "drought")
        waterlog_deaths  = sum(1 for e in deaths if e["death_cause"] == "waterlogging")

        # Averages
        avg_health   = np.mean([e["health_score"]  for e in self._buffer])
        avg_optimal  = np.mean([e["optimal_hrs"]   for e in self._buffer])
        avg_stress   = np.mean([e["stress_hrs"]    for e in self._buffer])
        avg_critical = np.mean([e["critical_hrs"]  for e in self._buffer])
        avg_waterlog = np.mean([e["waterlog_hrs"]  for e in self._buffer])
        avg_water    = np.mean([e["water_litres"]  for e in self._buffer])
        total_steps  = 3600   # 150 days × 24 hr/day

        border = "━" * 62
        print(f"\n{border}")
        print(
            f"  [TRAINING]  Episode {self._n_episodes:,}  |  "
            f"Timestep {self.num_timesteps:,}  |  {phase_label}"
        )
        print(border)

        # --- Survival ---
        print(f"\n  PLANT SURVIVAL  (last {n} episodes)")
        print(f"  {'Survived':<12}: {len(survived):>3} / {n}  ({len(survived)/n*100:.0f}%)")
        if deaths:
            print(
                f"  {'Died':<12}: {len(deaths):>3} / {n}  ({len(deaths)/n*100:.0f}%)"
                f"  —  drought: {drought_deaths}  waterlogging: {waterlog_deaths}"
            )
            # Stage where most deaths happened
            death_stages: dict[int, int] = defaultdict(int)
            for e in deaths:
                if e["death_stage"] >= 0:
                    death_stages[e["death_stage"]] += 1
            if death_stages:
                worst = max(death_stages, key=death_stages.__getitem__)
                print(f"  {'Most deaths at':<12}: Stage {worst} — {STAGE_NAMES[worst]}")
        else:
            print(f"  {'Died':<12}:   0 / {n}  (0%)  ✓ all survived")

        # --- Growth stage at episode end ---
        print(f"\n  GROWTH STAGE AT EPISODE END")
        row1, row2 = "", ""
        for s in range(5):
            label = f"Stage {s} {STAGE_NAMES[s]}"
            count = stage_counts.get(s, 0)
            entry = f"  {label:<22}: {count}"
            if s < 3:
                row1 += entry
            else:
                row2 += entry
        if row1: print(row1)
        if row2: print(row2)

        # --- Plant health ---
        pct_opt  = avg_optimal  / total_steps * 100
        pct_str  = avg_stress   / total_steps * 100
        pct_crit = avg_critical / total_steps * 100
        pct_wlog = avg_waterlog / total_steps * 100
        print(f"\n  PLANT HEALTH")
        print(f"  {'Health score':<22}: {avg_health:.3f}  (1.0 = perfect)")
        print(f"  {'Optimal moisture':<22}: {avg_optimal:>6.0f} hrs / {total_steps}  ({pct_opt:.0f}%)")
        print(f"  {'Stress hours':<22}: {avg_stress:>6.0f} hrs  ({pct_str:.0f}%)")
        print(f"  {'Critical hours':<22}: {avg_critical:>6.0f} hrs  ({pct_crit:.0f}%)")
        print(f"  {'Waterlog hours':<22}: {avg_waterlog:>6.0f} hrs  ({pct_wlog:.0f}%)")

        # --- Water usage ---
        avg_per_day = avg_water / 150
        print(f"\n  WATER USAGE")
        print(f"  {'Avg per season':<22}: {avg_water:>8.1f} L")
        print(f"  {'Avg per day':<22}: {avg_per_day:>8.1f} L/day")

        print(f"\n{border}\n")

    def _log_tensorboard(self) -> None:
        """Log episode metrics to TensorBoard."""
        n = len(self._buffer)
        if n == 0 or self.model is None:
            return

        deaths = [e for e in self._buffer if e["plant_dead"]]

        self.logger.record("episode/survival_rate",   1 - len(deaths) / n)
        self.logger.record("episode/death_rate",      len(deaths) / n)
        self.logger.record("episode/health_score",    np.mean([e["health_score"]  for e in self._buffer]))
        self.logger.record("episode/optimal_hours",   np.mean([e["optimal_hrs"]   for e in self._buffer]))
        self.logger.record("episode/stress_hours",    np.mean([e["stress_hrs"]    for e in self._buffer]))
        self.logger.record("episode/critical_hours",  np.mean([e["critical_hrs"]  for e in self._buffer]))
        self.logger.record("episode/waterlog_hours",  np.mean([e["waterlog_hrs"]  for e in self._buffer]))
        self.logger.record("episode/water_litres",    np.mean([e["water_litres"]  for e in self._buffer]))
        self.logger.record("episode/final_stage",     np.mean([e["final_stage"]   for e in self._buffer]))
        self.logger.dump(self.num_timesteps)
