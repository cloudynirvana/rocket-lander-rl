"""Train PPO on Lunar Lander under one reward variant and one seed."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs import make_env

# PPO hyperparameters for LunarLander from RL Baselines3 Zoo (Raffin, 2020).
PPO_KWARGS = dict(n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4, ent_coef=0.01)
N_ENVS = 16


def train(variant: str, seed: int, steps: int, out_dir: str | Path) -> dict:
    torch.set_num_threads(1)
    out = Path(out_dir)
    log_dir = out / "logs" / f"{variant}_s{seed}"

    def thunk(i):
        return lambda: make_env(variant, log_path=log_dir / f"env{i:02d}.csv", gamma=PPO_KWARGS["gamma"])

    vec = DummyVecEnv([thunk(i) for i in range(N_ENVS)])
    vec.seed(seed)
    model = PPO("MlpPolicy", vec, seed=seed, device="cpu", verbose=0, **PPO_KWARGS)
    t0 = time.time()
    model.learn(total_timesteps=steps)
    path = out / "models" / f"{variant}_s{seed}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)
    vec.close()
    return {"variant": variant, "seed": seed, "steps": steps, "seconds": time.time() - t0, "model": str(path)}
