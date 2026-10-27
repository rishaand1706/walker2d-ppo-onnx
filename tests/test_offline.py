"""The offline datasets mark only real falls as terminal, and the offline policy is bounded.

Uses the committed policy in models/, so it runs from a fresh clone without the generated datasets.
"""
import numpy as np
import torch

from config import MODELS
from controller import load_run
from offline.collect import collect
from offline.common import Actor, StateNorm


def test_time_limit_is_not_terminal():
    data, returns = collect(*load_run(MODELS, "policy_sb3.zip"), n=3000, seed=0)
    assert data["obs"].shape == (3000, 17) and data["act"].shape == (3000, 6)
    assert data["terminal"].dtype == bool
    assert np.all(np.abs(data["act"]) <= 1.0)
    # Every episode that ran to the 1,000-step limit must end on a non-terminal transition.
    ends = np.flatnonzero(np.any(data["next_obs"][:-1] != data["obs"][1:], axis=1))   # where a reset happened
    assert len(ends) >= 2, "too few finished episodes to check"
    starts = np.concatenate([[0], ends + 1])
    for start, end in zip(starts, ends):
        if end - start + 1 == 1000:
            assert not data["terminal"][end]
        else:
            assert data["terminal"][end]               # an early end can only be a fall


def test_actor_output_is_bounded():
    actor = Actor(17, 6)
    out = actor(torch.randn(512, 17) * 1e3)
    assert out.shape == (512, 6) and out.abs().max() <= 1.0


def test_state_norm_centres_the_data():
    obs = np.random.default_rng(0).normal(5.0, 3.0, size=(10_000, 17))
    z = StateNorm(obs)(obs)
    assert np.allclose(z.mean(0), 0, atol=1e-6) and np.allclose(z.std(0), 1, atol=1e-3)
