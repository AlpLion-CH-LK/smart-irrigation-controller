"""PPO training script for the smart irrigation controller.

Curriculum Learning:
    Phase 1 (40% of timesteps) — Dry season months (Feb–Apr, Jaffna)
             Agent learns core skill: moisture drops → irrigate correctly
    Phase 2 (60% of timesteps) — All months randomly
             Agent generalises to monsoon, moderate, and dry conditions

Usage:
    python scripts/train.py
    python scripts/train.py --timesteps 1000000 --n-envs 4
    python scripts/train.py --output-dir runs/experiment_1
    python scripts/train.py --phase1-ratio 0.3
    python scripts/train.py --log-interval 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from callbacks import IrrigationMonitorCallback
from irrigation.config_loader import load_config
from irrigation.rl.gym_env import IrrigationGymEnv
from irrigation.zone_config import ZoneConfig

# Load all training parameters from config.yaml
# CLI arguments override these when provided
_cfg   = load_config()
_train = _cfg["training"]
_ppo   = _cfg["ppo"]


def _make_callbacks(
    output_path: Path,
    eval_env,
    log_interval: int,
) -> list:
    return [
        EvalCallback(
            eval_env,
            best_model_save_path=str(output_path / "best"),
            eval_freq=10_000,
            n_eval_episodes=5,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(output_path / "checkpoints"),
            name_prefix="ppo_irrigation",
        ),
        IrrigationMonitorCallback(log_interval=log_interval, verbose=1),
    ]


def train(
    total_timesteps: int = _train["total_timesteps"],
    n_envs: int           = _train["n_envs"],
    output_dir: str       = "models",
    area_m2: float        = _cfg["zone"]["area_m2"],
    irrigation_type: str  = _cfg["zone"]["irrigation_type"],
    phase1_ratio: float   = _train["phase1_ratio"],
    log_interval: int     = _train["log_interval"],
) -> None:
    zone = ZoneConfig(area_m2=area_m2, irrigation_type=irrigation_type)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    phase1_steps = int(total_timesteps * phase1_ratio)
    phase2_steps = total_timesteps - phase1_steps

    print("=" * 62)
    print("  PPO Smart Irrigation — Curriculum Learning")
    print("=" * 62)
    print(f"  Zone          : {area_m2}m²  {irrigation_type}  ({zone.efficiency*100:.0f}% efficiency)")
    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Phase 1 steps : {phase1_steps:,}  ({phase1_ratio*100:.0f}%)  dry months (Feb–Apr)")
    print(f"  Phase 2 steps : {phase2_steps:,}  ({(1-phase1_ratio)*100:.0f}%)  all months")
    print(f"  Episode length: 3,600 steps  (150 days × 24 hr/day)")
    print(f"  Parallel envs : {n_envs}")
    print(f"  Log interval  : every {log_interval} episodes")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Phase 1 — Dry season months (learn basic irrigation)
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Training on dry months (Feb–Apr Jaffna)...")

    phase1_kwargs = {"zone": zone, "training_phase": 1}
    train_env_p1 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase1_kwargs)
    eval_env_p1  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase1_kwargs)

    # PPO hyperparameters — all from config.yaml → ppo section
    model = PPO(
        "MlpPolicy",
        train_env_p1,
        learning_rate  = _ppo["learning_rate"],
        n_steps        = _ppo["n_steps"],
        batch_size     = _ppo["batch_size"],
        n_epochs       = _ppo["n_epochs"],
        gamma          = _ppo["gamma"],
        gae_lambda     = _ppo["gae_lambda"],
        clip_range     = _ppo["clip_range"],
        verbose        = 1,
        tensorboard_log= str(output_path / "logs"),
    )

    model.learn(
        total_timesteps=phase1_steps,
        callback=_make_callbacks(output_path / "phase1", eval_env_p1, log_interval),
        reset_num_timesteps=True,
    )

    phase1_save = output_path / "ppo_phase1_complete"
    model.save(str(phase1_save))
    print(f"\n[PHASE 1] Complete — model saved to {phase1_save}")

    # ------------------------------------------------------------------
    # Phase 2 — All months (generalise to monsoon + varied conditions)
    # ------------------------------------------------------------------
    print("\n[PHASE 2] Training on all months (dry + moderate + monsoon)...")

    phase2_kwargs = {"zone": zone, "training_phase": 2}
    train_env_p2 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase2_kwargs)
    eval_env_p2  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase2_kwargs)

    model.set_env(train_env_p2)
    model.learn(
        total_timesteps=phase2_steps,
        callback=_make_callbacks(output_path / "phase2", eval_env_p2, log_interval),
        reset_num_timesteps=False,
    )

    final_save = output_path / "ppo_irrigation_final"
    model.save(str(final_save))
    print(f"\n[PHASE 2] Complete — final model saved to {final_save}")
    print("\nTraining complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train PPO irrigation agent with curriculum learning."
    )
    parser.add_argument("--timesteps",       type=int,   default=500_000)
    parser.add_argument("--n-envs",          type=int,   default=4)
    parser.add_argument("--output-dir",      type=str,   default="models")
    parser.add_argument("--area",            type=float, default=3.0,
                        help="Zone area in m²")
    parser.add_argument("--irrigation-type", type=str,  default="drip",
                        choices=["drip", "sprinkler"])
    parser.add_argument("--phase1-ratio",    type=float, default=0.4,
                        help="Fraction of timesteps for Phase 1 (0.0–1.0)")
    parser.add_argument("--log-interval",    type=int,   default=10,
                        help="Print episode summary every N episodes")
    args = parser.parse_args()

    train(
        args.timesteps,
        args.n_envs,
        args.output_dir,
        args.area,
        args.irrigation_type,
        args.phase1_ratio,
        args.log_interval,
    )
