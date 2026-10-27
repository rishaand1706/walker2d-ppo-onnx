"""The shipped ONNX controller must be the trained policy, bounded, and callable the way a robot calls it.

Needs models/ from export_onnx.py. Run with: pytest -q
"""
import pickle

import numpy as np
import pytest

from config import MODELS
from controller import OnnxController, Sb3Controller, load_run

FP32, INT8 = MODELS / "walker2d_ppo.onnx", MODELS / "walker2d_ppo_int8.onnx"


@pytest.fixture(scope="module")
def reference():
    model, norm = load_run(MODELS, "policy_sb3.zip")
    return Sb3Controller(model, norm), norm


def states(norm, n, spread, seed):
    """Synthetic states around the training distribution; spread 4 pushes many past the +/-10 clip."""
    rng = np.random.default_rng(seed)
    return norm.mean + np.sqrt(norm.var) * rng.normal(0, spread, size=(n, norm.mean.size))


def test_normalizer_is_vecnormalize_arithmetic(reference):
    _, norm = reference
    with open(MODELS / "vecnormalize.pkl", "rb") as f:
        vec_normalize = pickle.load(f)
    obs = states(norm, 500, 3.0, seed=0)
    # VecNormalize hands the policy float32, so agree to float32 precision.
    np.testing.assert_allclose(norm(obs), vec_normalize.normalize_obs(obs), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("spread", [1.0, 4.0])
def test_onnx_matches_sb3(reference, spread):
    sb3, norm = reference
    obs = states(norm, 1000, spread, seed=int(spread))
    expected = np.array([sb3(o) for o in obs])
    assert np.abs(OnnxController(FP32)(obs) - expected).max() < 1e-5


@pytest.mark.parametrize("path", [FP32, INT8], ids=["fp32", "int8"])
def test_actions_stay_in_bounds(path):
    obs = np.random.default_rng(1).normal(0, 1e4, size=(256, 17))  # wildly out-of-range sensor values
    actions = OnnxController(path)(obs)
    assert actions.min() >= -1.0 and actions.max() <= 1.0


@pytest.mark.parametrize("batch", [1, 7, 256])
def test_any_batch_size(batch):
    assert OnnxController(FP32)(np.zeros((batch, 17))).shape == (batch, 6)


def test_single_observation():
    assert OnnxController(FP32)(np.zeros(17)).shape == (6,)
