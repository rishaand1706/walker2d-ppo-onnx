"""The deployable controller: raw 17-dim state in, 6 joint torques out.

A trained run is two artifacts: the PPO policy (model.zip) and the running observation statistics VecNormalize
kept during training (vecnormalize.pkl). Shipping the policy without the statistics gives wrong actions, so the
exported controller carries them inside the graph:

    action = clip(actor(clip((obs - mean) / sqrt(var + eps), -c, c)), -1, 1)      c = 10 by default

This file holds that one definition three ways: the Stable-Baselines3 reference (`Sb3Controller`), a PyTorch
module for export (`TorchController`), and the ONNX Runtime version that a robot would run (`OnnxController`).
"""
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import FlattenExtractor


@dataclass
class Normalizer:
    """The observation half of a saved VecNormalize: running mean and variance, epsilon, clip."""
    mean: np.ndarray
    var: np.ndarray
    eps: float
    clip: float

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # Same arithmetic as VecNormalize._normalize_obs, in float64.
        return np.clip((obs - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)


def load_run(run_dir: Path, policy_file: str = "model.zip"):
    """Return (PPO model, Normalizer or None) for a run folder, or for models/ with policy_sb3.zip."""
    model = PPO.load(run_dir / policy_file, device="cpu")
    path = run_dir / "vecnormalize.pkl"
    if not path.exists():
        return model, None
    with open(path, "rb") as f:
        vec_normalize = pickle.load(f)  # VecNormalize pickles itself without its env
    rms = vec_normalize.obs_rms
    return model, Normalizer(rms.mean.copy(), rms.var.copy(), float(vec_normalize.epsilon), float(vec_normalize.clip_obs))


class Sb3Controller:
    """The reference: Stable-Baselines3's own deterministic predict() on normalized observations."""

    def __init__(self, model: PPO, norm: Normalizer | None):
        self.model, self.norm = model, norm

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.model.predict(obs if self.norm is None else self.norm(obs), deterministic=True)[0]


class TorchController(torch.nn.Module):
    """Normalizer + deterministic actor (the Gaussian's mean) + action clip, as one module to export."""

    def __init__(self, model: PPO, norm: Normalizer | None):
        super().__init__()
        policy = model.policy
        assert isinstance(policy.features_extractor, FlattenExtractor), "expects a flat MLP policy"
        self.actor = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net
        n = model.observation_space.shape[0]
        mean = np.zeros(n) if norm is None else norm.mean
        std = np.ones(n) if norm is None else np.sqrt(norm.var + norm.eps)
        clip = np.inf if norm is None else norm.clip
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer("obs_clip", torch.tensor(clip, dtype=torch.float32))
        self.register_buffer("low", torch.as_tensor(model.action_space.low, dtype=torch.float32))
        self.register_buffer("high", torch.as_tensor(model.action_space.high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = torch.clamp((obs - self.mean) / self.std, -self.obs_clip, self.obs_clip)
        return torch.clamp(self.action_net(self.actor(x)), self.low, self.high)


class OnnxController:
    """What a robot would run: one ONNX Runtime session, float32 observations in, actions out."""

    def __init__(self, path: Path, threads: int = 1):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        batch = np.asarray(obs, dtype=np.float32)
        single = batch.ndim == 1
        out = self.session.run(None, {self.input: batch[None] if single else batch})[0]
        return out[0] if single else out
