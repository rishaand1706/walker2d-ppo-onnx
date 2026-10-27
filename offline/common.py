"""Offline RL on Walker2d from the PPO policies' rollouts: datasets, networks and evaluation, in plain PyTorch.

Stable-Baselines3 appears only in collect.py, to run the PPO policies that generate the data. Training (behaviour
cloning and TD3+BC, both in train.py) is written directly against torch.

Run every script from the repo root as a module, e.g. `python -m offline.collect`.
"""
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from config import ENV_ID, EVAL_SEED, ROOT, RUNS

OFFLINE = ROOT / "offline"
DATA = OFFLINE / "data"            # symlink outside iCloud; regenerate with `python -m offline.collect`
OUT = RUNS / "offline"             # one folder per (algorithm, dataset, seed)

# Data sources, D4RL-style. Expert: the strongest PPO policy on the nominal robot (norm_s0, 5,629, never falls).
# Medium: the trained policy closest to 40% of the expert's return (raw_s1, 2,409). Actions are sampled from each
# policy's own Gaussian, as D4RL does, so the data covers more than one deterministic trajectory.
SOURCES = {"expert": "norm_s0", "medium": "raw_s1"}
DATASETS = {"medium": ("medium",), "medium-expert": ("medium", "expert"), "expert": ("expert",)}
TRANSITIONS = 200_000
COLLECT_SEED = 40_000              # episode resets for data collection: disjoint from train / eval / validation seeds
VAL_SEED = 30_000                  # periodic evaluation during offline training (curves only, never for selection)

# TD3+BC (Fujimoto & Gu 2021), the paper's settings; BC uses the same actor, optimizer and batch.
HIDDEN = 256
BATCH = 256
LR = 3e-4
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2
ALPHA = 2.5


def load_dataset(name: str) -> dict:
    """Concatenate the source files that make up a dataset (medium-expert = medium + expert)."""
    parts = [np.load(DATA / f"{src}.npz") for src in DATASETS[name]]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("obs", "act", "rew", "next_obs", "terminal")}


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, act_dim), nn.Tanh())

    def forward(self, obs):
        return self.net(obs)


class Critic(nn.Module):
    """Two independent Q networks (TD3's clipped double-Q)."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()

        def q():
            return nn.Sequential(nn.Linear(obs_dim + act_dim, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

        self.q1, self.q2 = q(), q()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class StateNorm:
    """TD3+BC normalizes states with the dataset's mean and std; the same transform is applied at evaluation."""

    def __init__(self, obs: np.ndarray):
        self.mean = obs.mean(0)
        self.std = obs.std(0) + 1e-3

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return (obs - self.mean) / self.std


def episode(act_fn, seed: int) -> dict:
    env = gym.make(ENV_ID)
    obs, _ = env.reset(seed=seed)
    total, steps = 0.0, 0
    while True:
        obs, reward, terminated, truncated, info = env.step(act_fn(obs))
        total += reward
        steps += 1
        if terminated or truncated:
            return dict(ret=total, length=steps, fell=bool(terminated), distance=float(info["x_position"]))


def policy_fn(actor: Actor, norm: StateNorm):
    @torch.inference_mode()
    def act(obs):
        return actor(torch.as_tensor(norm(obs), dtype=torch.float32)[None])[0].numpy()
    return act


def evaluate(act_fn, episodes: int, base_seed: int = EVAL_SEED) -> list[dict]:
    return [episode(act_fn, base_seed + i) for i in range(episodes)]
