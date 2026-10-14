"""Full pipeline: train every reward variant on every seed, evaluate, plot, record GIFs.

    python experiments/run_all.py                      # everything (~40 min on 4 cores)
    python experiments/run_all.py --steps 100000 --seeds 2 --episodes 20   # quick smoke run
    python experiments/run_all.py --stages eval figures                    # reuse trained models

Evaluation always uses the environment's built-in reward with no shaping.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from lander.envs import VARIANTS  # noqa: E402
from lander.evaluate import (CONDITIONS, heuristic_policy, random_policy, run_episodes,  # noqa: E402
                             sb3_policy, summarise, wilson)
from lander.train import N_ENVS, train  # noqa: E402

RESULTS, FIGURES, MEDIA = ROOT / "results", ROOT / "figures", ROOT / "media"
EVAL_SEED0 = 100_000
COLORS = {"builtin": "#1f77b4", "time_fee": "#2ca02c", "hover_hack": "#d62728", "hover_fixed": "#9467bd",
          "heuristic": "#000000", "random": "#7f7f7f"}
LABELS = {**VARIANTS, "heuristic": "Hand-written control law (Gymnasium heuristic)", "random": "Random actions"}


def _train_job(args):
    return train(*args)


def _eval_job(args):
    variant, seed, n_eps, conditions = args
    from stable_baselines3 import PPO
    import torch
    torch.set_num_threads(1)
    model = PPO.load(RESULTS / "models" / f"{variant}_s{seed}.zip", device="cpu")
    seeds = range(EVAL_SEED0 + seed * n_eps, EVAL_SEED0 + (seed + 1) * n_eps)
    return {(variant, seed, c): run_episodes(sb3_policy(model), seeds, **CONDITIONS[c]) for c in conditions}


def _baseline_job(args):
    name, n_seeds, n_eps, cond = args
    seeds = range(EVAL_SEED0, EVAL_SEED0 + n_seeds * n_eps)
    policy = heuristic_policy if name == "heuristic" else random_policy(0)
    return (name, cond), run_episodes(policy, seeds, **CONDITIONS[cond])


# ----------------------------------------------------------------------------- stages

def stage_train(a, pool):
    jobs = [(v, s, a.steps, RESULTS) for v in VARIANTS for s in range(a.seeds)]
    meta = list(pool.map(_train_job, jobs))
    (RESULTS / "training_runs.json").write_text(json.dumps(meta, indent=2))


def stage_eval(a, pool):
    robust = ["builtin", "time_fee", "hover_fixed"]
    jobs = [(v, s, a.episodes, list(CONDITIONS) if v in robust else ["nominal"])
            for v in VARIANTS for s in range(a.seeds)]
    episodes = {}
    for res in pool.map(_eval_job, jobs):
        episodes.update(res)
    base_jobs = [("heuristic", a.seeds, a.episodes, c) for c in CONDITIONS] + [("random", a.seeds, a.episodes, "nominal")]
    for key, rows in pool.map(_baseline_job, base_jobs):
        episodes[key] = rows

    out = {"episodes_per_model": a.episodes, "seeds": a.seeds, "nominal": {}, "robustness": {}}
    for v in VARIANTS:
        pooled = [r for s in range(a.seeds) for r in episodes[(v, s, "nominal")]]
        per_seed = [summarise(episodes[(v, s, "nominal")]) for s in range(a.seeds)]
        out["nominal"][v] = _with_ci(summarise(pooled), len(pooled)) | {"per_seed": per_seed}
    for b in ("heuristic", "random"):
        out["nominal"][b] = _with_ci(summarise(episodes[(b, "nominal")]), len(episodes[(b, "nominal")]))
    for c in CONDITIONS:
        out["robustness"][c] = {}
        for v in robust:
            pooled = [r for s in range(a.seeds) for r in episodes[(v, s, c)]]
            out["robustness"][c][v] = _with_ci(summarise(pooled), len(pooled))
        out["robustness"][c]["heuristic"] = _with_ci(summarise(episodes[("heuristic", c)]), len(episodes[("heuristic", c)]))
    (RESULTS / "evaluation.json").write_text(json.dumps(out, indent=2))


def _with_ci(s, n):
    s["landed_on_pad_ci"] = wilson(s["landed_on_pad"], n)
    s["timeout_ci"] = wilson(s["timeout"], n)
    return s


def _load_logs(variant, seed, bin_steps):
    import csv
    rows = []
    for f in sorted((RESULTS / "logs" / f"{variant}_s{seed}").glob("env*.csv")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append((int(r["env_steps"]) * N_ENVS, float(r["true_return"]), float(r["shaped_return"]),
                             r["outcome"] == "timeout", r["on_pad"] == "True"))
    arr = np.array(rows, dtype=float)
    bins = (arr[:, 0] // bin_steps).astype(int)
    nb = bins.max() + 1
    out = np.full((nb, 4), np.nan)
    for b in range(nb):
        m = bins == b
        if m.any():
            out[b] = arr[m, 1:].mean(axis=0)
    return (np.arange(nb) + 0.5) * bin_steps, out


def stage_figures(a, pool):
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    bin_steps = max(a.steps // 40, 1000)

    # Learning curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for v in VARIANTS:
        curves = []
        for s in range(a.seeds):
            x, c = _load_logs(v, s, bin_steps)
            curves.append(c)
        n = min(len(c) for c in curves)
        C = np.stack([c[:n] for c in curves])  # seeds x bins x [true, shaped, timeout, on_pad]
        x = x[:n]
        for ax, k in ((axes[0], 0), (axes[2], 2)):
            m, sd = np.nanmean(C[:, :, k], 0), np.nanstd(C[:, :, k], 0)
            ax.plot(x, m, color=COLORS[v], label=VARIANTS[v])
            ax.fill_between(x, m - sd, m + sd, color=COLORS[v], alpha=0.15)
        if v == "hover_hack":
            axes[1].plot(x, np.nanmean(C[:, :, 1], 0), color=COLORS[v], label="Return the agent is paid (shaped)")
            axes[1].plot(x, np.nanmean(C[:, :, 0], 0), color=COLORS[v], ls="--", label="Built-in return (what we want)")
    axes[0].axhline(200, color="k", lw=0.8, ls=":")
    axes[0].set(title="True (built-in) episode return during training", xlabel="Environment steps", ylabel="Return")
    axes[1].set(title="Hover hack: paid score vs. real score", xlabel="Environment steps", ylabel="Return")
    axes[2].set(title="Share of episodes ending in a timeout (hovering)", xlabel="Environment steps", ylabel="Share")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=8)
    fig.suptitle(f"Training curves, mean ± s.d. over {a.seeds} seeds")
    fig.tight_layout(); fig.savefig(FIGURES / "learning_curves.png", dpi=150); plt.close(fig)

    # Outcomes (nominal)
    names = list(VARIANTS) + ["heuristic", "random"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    parts = [("landed on pad", "#2ca02c"), ("landed off pad", "#98df8a"), ("crashed / out of bounds", "#d62728"),
             ("timeout (hovering)", "#ffbf00")]
    left = np.zeros(len(names))
    for label, col in parts:
        vals = []
        for nm in names:
            s = ev["nominal"][nm]
            vals.append({"landed on pad": s["landed_on_pad"], "landed off pad": s["landed"] - s["landed_on_pad"],
                         "crashed / out of bounds": s["crashed"], "timeout (hovering)": s["timeout"]}[label])
        ax.barh([LABELS[n] for n in names], vals, left=left, color=col, label=label)
        left += np.array(vals)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("Share of evaluation episodes (built-in reward, no shaping)")
    ax.set_title("How episodes end, by reward design"); ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "outcomes.png", dpi=150); plt.close(fig)

    # Robustness
    conds = list(CONDITIONS)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    series = ["builtin", "time_fee", "hover_fixed", "heuristic"]
    w = 0.8 / len(series)
    for k, nm in enumerate(series):
        r = [ev["robustness"][c][nm]["landed_on_pad"] for c in conds]
        ci = np.array([ev["robustness"][c][nm]["landed_on_pad_ci"] for c in conds])
        ax.bar(np.arange(len(conds)) + (k - 1.5) * w, r, w, yerr=[np.array(r) - ci[:, 0], ci[:, 1] - np.array(r)],
               color=COLORS[nm], label=LABELS[nm], capsize=2, alpha=0.85)
    ax.set_xticks(range(len(conds)), conds); ax.set_ylim(0, 1.05)
    ax.set_ylabel("Landed on pad (95% Wilson CI)")
    ax.set_title("Robustness to wind and gravity dispersions (trained only in nominal conditions)")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(FIGURES / "robustness.png", dpi=150); plt.close(fig)


def stage_gifs(a, pool):
    import gymnasium as gym
    import imageio.v2 as imageio
    from stable_baselines3 import PPO
    from lander.envs import OutcomeTracker
    MEDIA.mkdir(exist_ok=True)
    ev = json.loads((RESULTS / "evaluation.json").read_text())
    picks = {}
    for v in ("builtin", "hover_hack", "hover_fixed"):
        per = ev["nominal"][v]["per_seed"]
        key = "timeout" if v == "hover_hack" else "landed_on_pad"
        picks[v] = int(np.argmax([p[key] for p in per]))
    for v, seed in picks.items():
        model = PPO.load(RESULTS / "models" / f"{v}_s{seed}.zip", device="cpu")
        env = OutcomeTracker(gym.make("LunarLander-v3", render_mode="rgb_array"))
        want = "timeout" if v == "hover_hack" else "landed"
        for ep_seed in range(EVAL_SEED0, EVAL_SEED0 + 50):
            frames, (obs, _) = [], env.reset(seed=ep_seed)
            while True:
                obs, _, term, trunc, info = env.step(int(model.predict(obs, deterministic=True)[0]))
                frames.append(env.render()[::2, ::2])
                if term or trunc:
                    break
            if info["episode_outcome"]["outcome"] == want:
                break
        stride = 4 if len(frames) > 400 else 2
        imageio.mimsave(MEDIA / f"{v}.gif", frames[::stride], duration=1000 * stride / 50, loop=0)
        env.close()
    (MEDIA / "gif_sources.json").write_text(json.dumps({v: {"seed": s} for v, s in picks.items()}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--stages", nargs="*", default=["train", "eval", "figures", "gifs"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    for d in (RESULTS, FIGURES, MEDIA):
        d.mkdir(exist_ok=True)
    stages = {"train": stage_train, "eval": stage_eval, "figures": stage_figures, "gifs": stage_gifs}
    with ProcessPoolExecutor(a.workers) as pool:
        for name in a.stages:
            stages[name](a, pool)
            print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    main()
