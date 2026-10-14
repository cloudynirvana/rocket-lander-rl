"""Lunar Lander environments with outcome tracking and reward-shaping variants.

Wrapper stack, innermost first::

    LunarLander-v3  ->  OutcomeTracker  ->  [shaping wrapper]  ->  EpisodeLogger

``OutcomeTracker`` sits *inside* the shaping so it always records the
environment's built-in ("true") reward, whatever the agent is being paid.
"""
from __future__ import annotations

import csv
from pathlib import Path

import gymnasium as gym
import numpy as np

# Observation indices (Gymnasium LunarLander): x, y, vx, vy, angle, angular velocity, left leg, right leg
X, Y, VX, VY, ANGLE = 0, 1, 2, 3, 4
PAD_HALF_WIDTH = 0.2  # landing pad spans |x| < 0.2 in observation units (between the flags)


def upright_and_slow(obs: np.ndarray, max_angle: float = 0.2, max_speed: float = 0.2) -> bool:
    """The 'good position' the hover-hack rule pays for, in observation units."""
    return abs(obs[ANGLE]) < max_angle and float(np.hypot(obs[VX], obs[VY])) < max_speed


class OutcomeTracker(gym.Wrapper):
    """Records how each episode ended, using the built-in reward and physics state.

    At episode end ``info["episode_outcome"]`` holds: true_return, length,
    outcome (landed / crashed / out_of_bounds / timeout), on_pad, and engine
    firings (main and side) as a fuel proxy.
    """

    def reset(self, **kwargs):
        self._ret, self._len, self._main, self._side = 0.0, 0, 0, 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ret += float(reward)
        self._len += 1
        a = int(action)
        self._main += a == 2
        self._side += a in (1, 3)
        if terminated or truncated:
            lunar = self.env.unwrapped
            if terminated:
                if lunar.game_over:
                    outcome = "crashed"
                elif abs(obs[X]) >= 1.0:
                    outcome = "out_of_bounds"
                else:
                    outcome = "landed"  # body came to rest
            else:
                outcome = "timeout"
            info["episode_outcome"] = {
                "true_return": self._ret, "length": self._len, "outcome": outcome,
                "on_pad": outcome == "landed" and abs(float(obs[X])) < PAD_HALF_WIDTH,
                "main_fires": self._main, "side_fires": self._side,
            }
        return obs, reward, terminated, truncated, info


class TimeFee(gym.Wrapper):
    """Charge a small fee every frame, so taking longer costs something."""

    def __init__(self, env, fee: float = 0.05):
        super().__init__(env)
        self.fee = fee

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward - self.fee, terminated, truncated, info


class HoverBonus(gym.Wrapper):
    """The flawed rule from the guide: +bonus for every frame spent upright and slow.

    Hovering for the full 1000-frame episode pays up to 1000 points, far more
    than landing (+100), so this is expected to invite reward hacking.
    """

    def __init__(self, env, bonus: float = 1.0):
        super().__init__(env)
        self.bonus = bonus

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward + self.bonus * upright_and_slow(obs), terminated, truncated, info


class PotentialHoverBonus(gym.Wrapper):
    """The fix: pay for *becoming* upright and slow, not for staying there, plus a time fee.

    Uses potential-based shaping F = gamma * phi(s') - phi(s) with
    phi(s) = bonus * upright_and_slow(s) and phi(terminal) = 0 (Ng, Harada &
    Russell, 1999), which cannot change which policy is optimal. A small
    per-frame fee then makes waiting costly.
    """

    def __init__(self, env, bonus: float = 1.0, gamma: float = 0.999, fee: float = 0.05):
        super().__init__(env)
        self.bonus, self.gamma, self.fee = bonus, gamma, fee

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = self.bonus * upright_and_slow(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = 0.0 if terminated else self.bonus * upright_and_slow(obs)
        shaping = self.gamma * phi_next - self._phi
        self._phi = phi_next
        return obs, reward + shaping - self.fee, terminated, truncated, info


class EpisodeLogger(gym.Wrapper):
    """Outermost wrapper: logs the shaped return the agent saw next to the true return."""

    FIELDS = ["episode", "env_steps", "shaped_return", "true_return", "length", "outcome", "on_pad",
              "main_fires", "side_fires"]

    def __init__(self, env, path: str | Path):
        super().__init__(env)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._file, self.FIELDS)
        self._writer.writeheader()
        self._episodes, self._steps = 0, 0

    def reset(self, **kwargs):
        self._shaped = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._shaped += float(reward)
        self._steps += 1
        if "episode_outcome" in info:
            self._episodes += 1
            self._writer.writerow({"episode": self._episodes, "env_steps": self._steps,
                                   "shaped_return": round(self._shaped, 3),
                                   **{k: info["episode_outcome"][k] for k in self.FIELDS[3:]}})
            self._file.flush()
        return obs, reward, terminated, truncated, info

    def close(self):
        self._file.close()
        super().close()


VARIANTS = {
    "builtin": "Built-in reward only",
    "time_fee": "Built-in + per-frame fee (0.05)",
    "hover_hack": "Built-in + 1 per frame upright & slow (the flawed rule)",
    "hover_fixed": "Potential-based upright & slow bonus + per-frame fee (the fix)",
}


def apply_variant(env: gym.Env, variant: str, gamma: float = 0.999) -> gym.Env:
    if variant == "builtin":
        return env
    if variant == "time_fee":
        return TimeFee(env)
    if variant == "hover_hack":
        return HoverBonus(env)
    if variant == "hover_fixed":
        return PotentialHoverBonus(env, gamma=gamma)
    raise ValueError(f"unknown variant {variant!r}")


def make_env(variant: str = "builtin", log_path: str | Path | None = None, gamma: float = 0.999,
             render_mode: str | None = None, **env_kwargs) -> gym.Env:
    env = OutcomeTracker(gym.make("LunarLander-v3", render_mode=render_mode, **env_kwargs))
    env = apply_variant(env, variant, gamma)
    if log_path is not None:
        env = EpisodeLogger(env, log_path)
    return env
