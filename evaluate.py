"""Evaluate every trained run with the deterministic policy (Stable-Baselines3 reference path).

    python evaluate.py

1. Nominal physics: EVAL_EPISODES episodes per run, seeds EVAL_SEED.. (never used in training).
2. Robustness sweep (norm and dr runs): body mass and ground friction scaled one at a time over MASS_GRID and
   FRICTION_GRID, SWEEP_EPISODES episodes per setting.
3. Selection score (norm and dr runs): SELECT_EPISODES validation episodes, each with its own mass and friction
   drawn from the randomization ranges; the same draws for every run. export_onnx.py deploys the best dr run.

Writes results/eval_episodes.csv, eval_runs.csv, robustness.csv, selection.csv and summary.json.
"""
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor

import gymnasium as gym
import numpy as np
import torch

from config import (CONDITIONS, DR_FRICTION, DR_MASS, ENV_ID, EVAL_EPISODES, EVAL_SEED, FRICTION_GRID, MASS_GRID,
                    RESULTS, RUNS, SELECT_EPISODES, SELECT_SEED, SWEEP_EPISODES)
from controller import Sb3Controller, load_run
from physics import FixedPhysics

MAX_STEPS = 1000


def episode(env, act, seed):
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(act(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return dict(ret=total, length=steps, fell=bool(terminated), distance=float(info["x_position"]))


def run_list():
    out = []
    for d in sorted(RUNS.iterdir()):
        m = re.fullmatch(r"(norm|raw|dr)_s(\d+)", d.name)
        if m and (d / "model.zip").exists():
            out.append((m.group(1), int(m.group(2)), d))
    return out


def validation_physics():
    """The SELECT_EPISODES (mass, friction) draws, identical for every run."""
    rng = np.random.default_rng(SELECT_SEED)
    return [(rng.uniform(*DR_MASS), rng.uniform(*DR_FRICTION)) for _ in range(SELECT_EPISODES)]


def evaluate_run(job):
    condition, seed, run_dir = job
    torch.set_num_threads(1)
    act = Sb3Controller(*load_run(run_dir))
    rows = [dict(condition=condition, seed=seed, episode=i, **episode(gym.make(ENV_ID), act, EVAL_SEED + i))
            for i in range(EVAL_EPISODES)]
    select = None
    if condition in ("norm", "dr"):
        eps = [episode(FixedPhysics(gym.make(ENV_ID), mass=m, friction=f), act, SELECT_SEED + i)
               for i, (m, f) in enumerate(validation_physics())]
        select = dict(condition=condition, seed=seed, select_return=round(float(np.mean([e["ret"] for e in eps])), 1),
                      select_survival=float(np.mean([not e["fell"] for e in eps])))
    sweep = []
    if condition in ("norm", "dr"):
        settings = [("mass", k) for k in MASS_GRID] + [("friction", k) for k in FRICTION_GRID]
        for param, k in settings:
            env = FixedPhysics(gym.make(ENV_ID), **{param: k})
            eps = [episode(env, act, EVAL_SEED + i) for i in range(SWEEP_EPISODES)]
            sweep.append(dict(condition=condition, seed=seed, param=param, scale=k,
                              mean_return=float(np.mean([e["ret"] for e in eps])),
                              survival=float(np.mean([not e["fell"] for e in eps]))))
    return rows, sweep, select


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    jobs = run_list()
    with ProcessPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(evaluate_run, jobs))
    episodes = [r for rows, _, _ in results for r in rows]
    sweep = [s for _, rows, _ in results for s in rows]
    selection = [sel for _, _, sel in results if sel]

    per_run = []
    for condition, seed, _ in jobs:
        eps = [e for e in episodes if e["condition"] == condition and e["seed"] == seed]
        per_run.append(dict(condition=condition, seed=seed,
                            mean_return=round(float(np.mean([e["ret"] for e in eps])), 1),
                            std_return=round(float(np.std([e["ret"] for e in eps])), 1),
                            falls=sum(e["fell"] for e in eps), episodes=len(eps),
                            mean_distance=round(float(np.mean([e["distance"] for e in eps])), 2)))

    summary = {}
    for c in CONDITIONS:
        runs = [r for r in per_run if r["condition"] == c]
        if not runs:
            continue
        means = np.array([r["mean_return"] for r in runs])
        summary[c] = dict(seeds=len(runs), mean_return=round(float(means.mean()), 1),
                          std_across_seeds=round(float(means.std(ddof=1)), 1) if len(runs) > 1 else None,
                          per_seed=[r["mean_return"] for r in runs],
                          falls=f"{sum(r['falls'] for r in runs)}/{sum(r['episodes'] for r in runs)}",
                          mean_distance=round(float(np.mean([r["mean_distance"] for r in runs])), 2))
    summary["protocol"] = dict(env=ENV_ID, eval_episodes=EVAL_EPISODES, eval_seeds=f"{EVAL_SEED}..",
                               policy="deterministic (Gaussian mean)", sweep_episodes=SWEEP_EPISODES)

    write_csv(RESULTS / "eval_episodes.csv", episodes)
    write_csv(RESULTS / "eval_runs.csv", per_run)
    if sweep:
        write_csv(RESULTS / "robustness.csv", sweep)
    if selection:
        write_csv(RESULTS / "selection.csv", selection)
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for c in CONDITIONS:
        if c in summary:
            s = summary[c]
            print(f"{c:5s} {s['mean_return']:8.1f} +/- {s['std_across_seeds']} over {s['seeds']} seeds  "
                  f"per seed {s['per_seed']}  falls {s['falls']}")


if __name__ == "__main__":
    main()
