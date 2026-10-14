import numpy as np
import pytest

from lander.envs import (PotentialHoverBonus, make_env, upright_and_slow)
from lander.evaluate import heuristic_policy, random_policy, run_episodes, summarise, wilson


def _rollout(env, policy, seed=0):
    obs, _ = env.reset(seed=seed)
    rewards = []
    while True:
        obs, r, term, trunc, info = env.step(policy(obs, env))
        rewards.append(r)
        if term or trunc:
            return np.array(rewards), info


def test_outcome_tracker_records_builtin_return():
    env = make_env("builtin")
    rewards, info = _rollout(env, random_policy(0))
    out = info["episode_outcome"]
    assert out["true_return"] == pytest.approx(rewards.sum())
    assert out["length"] == len(rewards)
    assert out["outcome"] in {"landed", "crashed", "out_of_bounds", "timeout"}


@pytest.mark.parametrize("variant", ["time_fee", "hover_hack", "hover_fixed"])
def test_shaping_never_leaks_into_true_return(variant):
    # Same seed and same (open-loop) actions -> identical physics, so the true return must match.
    actions = np.random.default_rng(1).integers(0, 4, size=1000)
    results = {}
    for v in ("builtin", variant):
        env = make_env(v)
        env.reset(seed=3)
        shaped = 0.0
        for a in actions:
            _, r, term, trunc, info = env.step(int(a))
            shaped += r
            if term or trunc:
                break
        results[v] = (shaped, info["episode_outcome"]["true_return"])
    assert results[variant][1] == pytest.approx(results["builtin"][1])
    assert results[variant][0] != pytest.approx(results["builtin"][0])


def test_time_fee_charges_every_frame():
    actions = [0] * 30
    totals = {}
    for v in ("builtin", "time_fee"):
        env = make_env(v)
        env.reset(seed=5)
        totals[v] = sum(env.step(a)[1] for a in actions)
    assert totals["builtin"] - totals["time_fee"] == pytest.approx(0.05 * 30)


def test_potential_shaping_telescopes():
    """Sum of potential-based shaping over an episode = sum_t gamma^... ; with gamma=1 it is -phi(s0)."""
    env = PotentialHoverBonus(make_env("builtin"), gamma=1.0, fee=0.0)
    obs, _ = env.reset(seed=0)
    phi0 = float(upright_and_slow(obs))
    shaping_total, true_total = 0.0, 0.0
    while True:
        obs, r, term, trunc, info = env.step(heuristic_policy(obs, env))
        shaping_total += r
        if term or trunc:
            break
    true_total = info["episode_outcome"]["true_return"]
    if term:  # phi(terminal) = 0, so the shaping sums to exactly -phi(s0)
        assert shaping_total - true_total == pytest.approx(-phi0)


def test_heuristic_lands_most_of_the_time():
    s = summarise(run_episodes(heuristic_policy, range(20)))
    assert s["landed"] >= 0.8 and s["mean_return"] > 150


def test_random_policy_crashes():
    s = summarise(run_episodes(random_policy(0), range(10)))
    assert s["landed_on_pad"] == 0.0 and s["mean_return"] < 0


def test_wilson_interval():
    lo, hi = wilson(0.5, 100)
    assert lo < 0.5 < hi and hi - lo == pytest.approx(0.192, abs=0.005)
    lo, hi = wilson(0.0, 50)
    assert lo == pytest.approx(0.0, abs=1e-12) and 0 < hi < 0.1
