"""Evaluation: fly a policy on the *built-in* reward only, over many fixed seeds."""
from __future__ import annotations

from typing import Callable

import numpy as np
from gymnasium.envs.box2d.lunar_lander import heuristic

from .envs import make_env

Policy = Callable[[np.ndarray, object], int]

# Dispersion conditions for robustness testing (Gymnasium LunarLander options).
CONDITIONS = {
    "nominal": {},
    "wind 10": {"enable_wind": True, "wind_power": 10.0, "turbulence_power": 1.0},
    "wind 20 (max)": {"enable_wind": True, "wind_power": 20.0, "turbulence_power": 2.0},
    "gravity -11.5 (heavier)": {"gravity": -11.5},
    "gravity -8 (lighter)": {"gravity": -8.0},
}


def sb3_policy(model) -> Policy:
    return lambda obs, env: int(model.predict(obs, deterministic=True)[0])


def heuristic_policy(obs, env) -> int:
    """Gymnasium's hand-written PID-style control law."""
    return int(heuristic(env.unwrapped, obs))


def random_policy(seed: int = 0) -> Policy:
    rng = np.random.default_rng(seed)
    return lambda obs, env: int(rng.integers(4))


def run_episodes(policy: Policy, seeds, **env_kwargs) -> list[dict]:
    """Run one episode per seed with no reward shaping and return per-episode outcomes."""
    env = make_env("builtin", **env_kwargs)
    rows = []
    for s in seeds:
        obs, _ = env.reset(seed=int(s))
        while True:
            obs, _, terminated, truncated, info = env.step(policy(obs, env))
            if terminated or truncated:
                rows.append({"seed": int(s), **info["episode_outcome"]})
                break
    env.close()
    return rows


def summarise(rows: list[dict]) -> dict:
    n = len(rows)
    outcomes = [r["outcome"] for r in rows]
    ret = np.array([r["true_return"] for r in rows])
    return {
        "n": n,
        "mean_return": float(ret.mean()),
        "solved_rate": float((ret >= 200).mean()),   # Gymnasium's per-episode "solved" bar
        "landed_on_pad": float(np.mean([r["on_pad"] for r in rows])),
        "landed": outcomes.count("landed") / n,
        "crashed": (outcomes.count("crashed") + outcomes.count("out_of_bounds")) / n,
        "timeout": outcomes.count("timeout") / n,
        "mean_length": float(np.mean([r["length"] for r in rows])),
        "mean_main_fires": float(np.mean([r["main_fires"] for r in rows])),
    }


def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n (k may be a rate)."""
    p = k / n if k > 1 else k
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(centre - half), float(centre + half)
