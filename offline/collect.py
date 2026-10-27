"""Collect the offline datasets by running the PPO policies, and score a random policy for normalization.

    python -m offline.collect               # offline/data/{expert,medium}.npz + results/offline_data.json

Each file holds TRANSITIONS steps: obs, act (as executed), rew, next_obs, and `terminal`, which is True only when
the walker fell. Episodes cut off at the 1,000-step limit are not terminal: a TD target must keep bootstrapping
through a time limit, or the critic learns that walking well ends in zero future reward.
"""
import csv
import json

import gymnasium as gym
import numpy as np
import torch

from config import ENV_ID, EVAL_EPISODES, EVAL_SEED, RESULTS, RUNS
from controller import load_run

from .common import COLLECT_SEED, DATA, SOURCES, TRANSITIONS, evaluate


def collect(model, norm, n: int, seed: int) -> tuple[dict, list[float]]:
    """Run one PPO policy (actions sampled from its Gaussian) for n transitions."""
    torch.manual_seed(seed)                     # SB3 samples the Gaussian with torch's RNG
    env = gym.make(ENV_ID)
    buf = {k: [] for k in ("obs", "act", "rew", "next_obs", "terminal")}
    returns, ep, ep_ret = [], 0, 0.0
    obs, _ = env.reset(seed=COLLECT_SEED + seed * 10_000)
    while len(buf["obs"]) < n:
        act = model.predict(obs if norm is None else norm(obs), deterministic=False)[0]
        next_obs, rew, terminated, truncated, _ = env.step(act)
        for k, v in (("obs", obs), ("act", act), ("rew", rew), ("next_obs", next_obs), ("terminal", terminated)):
            buf[k].append(v)
        ep_ret += rew
        obs = next_obs
        if terminated or truncated:
            returns.append(ep_ret)
            ep, ep_ret = ep + 1, 0.0
            obs, _ = env.reset(seed=COLLECT_SEED + seed * 10_000 + ep)
    data = {k: np.asarray(v, dtype=np.float32 if k != "terminal" else bool) for k, v in buf.items()}
    return data, returns


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    stats = {}
    for i, (name, run) in enumerate(SOURCES.items()):
        data, returns = collect(*load_run(RUNS / run), TRANSITIONS, seed=i)
        np.savez_compressed(DATA / f"{name}.npz", **data)
        stats[name] = dict(source=run, transitions=len(data["obs"]), episodes=len(returns),
                           mean_episode_return=round(float(np.mean(returns)), 1),
                           falls=int(data["terminal"].sum()))
        print(f"{name:7s} from {run}: {stats[name]}")

    # Reference points for a normalized score: a uniform-random policy and the expert's deterministic return,
    # both on the evaluation seeds every offline policy is scored on.
    rng = np.random.default_rng(0)
    random_eps = evaluate(lambda obs: rng.uniform(-1, 1, 6).astype(np.float32), EVAL_EPISODES)
    expert_return = next(float(r["mean_return"]) for r in csv.DictReader(open(RESULTS / "eval_runs.csv"))
                         if f"{r['condition']}_s{r['seed']}" == SOURCES["expert"])
    stats["reference"] = dict(random_return=round(float(np.mean([e["ret"] for e in random_eps])), 1),
                              expert_return=expert_return, eval_seeds=f"{EVAL_SEED}..", episodes=EVAL_EPISODES)
    (RESULTS / "offline_data.json").write_text(json.dumps(stats, indent=2) + "\n")
    print("reference:", stats["reference"])


if __name__ == "__main__":
    main()
