"""PPO training script for the smart irrigation controller.

Usage:
    python scripts/train.py
    python scripts/train.py --timesteps 1000000 --n-envs 8
    python scripts/train.py --output-dir runs/experiment_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from irrigation.rl.gym_env import IrrigationGymEnv


def train(
    total_timesteps: int = 500_000,
    n_envs: int = 4,
    output_dir: str = "models",
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO for {total_timesteps:,} timesteps with {n_envs} parallel envs.")
    print(f"One episode = 150-day chili season ({150 * 48:,} steps at 30 min/step).")

    train_env = make_vec_env(IrrigationGymEnv, n_envs=n_envs)
    eval_env = make_vec_env(IrrigationGymEnv, n_envs=1)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=str(output_path / "logs"),
    )

    callbacks = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(output_path / "best"),
            eval_freq=10_000,
            n_eval_episodes=3,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=50_000,
            save_path=str(output_path / "checkpoints"),
            name_prefix="ppo_irrigation",
        ),
    ]

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    save_path = output_path / "ppo_irrigation_final"
    model.save(str(save_path))
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO irrigation agent.")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--output-dir", type=str, default="models")
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.output_dir)
