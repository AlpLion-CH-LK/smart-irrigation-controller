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

import wandb
from wandb.integration.sb3 import WandbCallback

import argparse
from datetime import datetime
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
        WandbCallback(verbose=0), 
    ]


def train(
    total_timesteps: int = _train["total_timesteps"],
    n_envs: int           = _train["n_envs"],
    output_dir: str       = "models",
    area_m2: float        = _cfg["zone"]["area_m2"],
    irrigation_type: str  = _cfg["zone"]["irrigation_type"],
    phase1_ratio: float   = _train["phase1_ratio"],
    phase2_ratio: float   = _train["phase2_ratio"],
    phase3_ratio: float   = _train["phase3_ratio"],
    log_interval: int     = _train["log_interval"],
) -> None:
    
    
    zone = ZoneConfig(area_m2=area_m2, irrigation_type=irrigation_type)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = Path(output_dir) / today
    output_path.mkdir(parents=True, exist_ok=True)

    run = wandb.init(
        project="smart-irrigation-controller",
        config={
            "total_timesteps": total_timesteps,
            "n_envs": n_envs,
            "phase1_ratio": phase1_ratio,
            "phase2_ratio": phase2_ratio,
            "phase3_ratio": phase3_ratio,
            "learning_rate": _ppo["learning_rate"],
            "gamma": _ppo["gamma"],
            "n_steps": _ppo["n_steps"],
            "batch_size": _ppo["batch_size"],
            "clip_range": _ppo["clip_range"],
            "area_m2": area_m2,
            "irrigation_type": irrigation_type,
        },
        sync_tensorboard=True,
    )

    phase1_steps = int(total_timesteps * phase1_ratio)
    phase2_steps = int(total_timesteps * phase2_ratio)
    phase3_steps = total_timesteps - phase1_steps - phase2_steps

    print("=" * 62)
    print("  PPO Smart Irrigation — 3-Phase Curriculum Learning")
    print("=" * 62)
    print(f"  Zone          : {area_m2}m²  {irrigation_type}  ({zone.efficiency*100:.0f}% efficiency)")
    print(f"  Total steps   : {total_timesteps:,}")
    print(f"  Phase 1 steps : {phase1_steps:,}  ({phase1_ratio*100:.0f}%)  Yala season (Jan–Mar start)")
    print(f"  Phase 2 steps : {phase2_steps:,}  ({phase2_ratio*100:.0f}%)  Maha season (Aug–Sep start)")
    print(f"  Phase 3 steps : {phase3_steps:,}  ({phase3_ratio*100:.0f}%)  Both seasons, random year/month")
    print(f"  Episode length: 3,600 steps  (150 days × 24 hr/day)")
    print(f"  Parallel envs : {n_envs}")
    print(f"  Log interval  : every {log_interval} episodes")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Phase 1 — Yala season (Jan/Feb/Mar start, fixed year, sequential months)
    # ------------------------------------------------------------------
    print("\n[PHASE 1] Yala season — dry conditions, Jan/Feb/Mar planting...")

    phase1_kwargs = {"zone": zone, "training_phase": 1}
    train_env_p1 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase1_kwargs)
    eval_env_p1  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase1_kwargs)

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

    phase1_save = output_path / "ppo_phase1_yala"
    model.save(str(phase1_save))
    print(f"\n[PHASE 1] Complete — model saved to {phase1_save}")

    # ------------------------------------------------------------------
    # Phase 2 — Maha season (Aug/Sep start, fixed year, sequential months)
    # ------------------------------------------------------------------
    print("\n[PHASE 2] Maha season — monsoon conditions, Aug/Sep planting...")

    phase2_kwargs = {"zone": zone, "training_phase": 2}
    train_env_p2 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase2_kwargs)
    eval_env_p2  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase2_kwargs)

    model.set_env(train_env_p2)
    model.learn(
        total_timesteps=phase2_steps,
        callback=_make_callbacks(output_path / "phase2", eval_env_p2, log_interval),
        reset_num_timesteps=False,
    )

    phase2_save = output_path / "ppo_phase2_maha"
    model.save(str(phase2_save))
    print(f"\n[PHASE 2] Complete — model saved to {phase2_save}")

    # ------------------------------------------------------------------
    # Phase 3 — Both seasons, random year per month (maximum variability)
    # ------------------------------------------------------------------
    print("\n[PHASE 3] Both seasons — random year per month, full variability...")

    phase3_kwargs = {"zone": zone, "training_phase": 3}
    train_env_p3 = make_vec_env(IrrigationGymEnv, n_envs=n_envs, env_kwargs=phase3_kwargs)
    eval_env_p3  = make_vec_env(IrrigationGymEnv, n_envs=1,      env_kwargs=phase3_kwargs)

    model.set_env(train_env_p3)
    model.learn(
        total_timesteps=phase3_steps,
        callback=_make_callbacks(output_path / "phase3", eval_env_p3, log_interval),
        reset_num_timesteps=False,
    )

    final_save = output_path / "ppo_irrigation_final"
    model.save(str(final_save))
    print(f"\n[PHASE 3] Complete — final model saved to {final_save}")
    print("\nTraining complete.")
    run.finish()


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
    parser.add_argument("--phase1-ratio",    type=float, default=0.25,
                        help="Fraction of timesteps for Phase 1 — Yala season")
    parser.add_argument("--phase2-ratio",    type=float, default=0.35,
                        help="Fraction of timesteps for Phase 2 — Maha season")
    parser.add_argument("--phase3-ratio",    type=float, default=0.40,
                        help="Fraction of timesteps for Phase 3 — both seasons")
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
        args.phase2_ratio,
        args.phase3_ratio,
        args.log_interval,
    )
