"""Shared settings for training, evaluation and export. One place, so every script agrees."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"          # symlink outside iCloud: one folder per training run
MODELS = ROOT / "models"      # the exported, deployable controller
RESULTS = ROOT / "results"    # JSON / CSV / figures that the README quotes

ENV_ID = "Walker2d-v5"        # MuJoCo planar biped: 17-dim state, 6 joint torques in [-1, 1], 1000-step episodes
TIMESTEPS = 1_000_000
SEEDS = (0, 1, 2, 3, 4)

# Three training conditions, each run with every seed:
#   norm  observation + reward normalization (VecNormalize)
#   raw   ablation: no normalization, otherwise identical
#   dr    norm + domain randomization of body mass and ground friction at every episode reset; the deployed
#         controller is the dr seed with the best validation score (export_onnx.py)
CONDITIONS = ("norm", "raw", "dr")

# PPO's MuJoCo settings from the paper (Schulman et al. 2017, Table 3): horizon 2048, Adam 3e-4, 10 epochs,
# minibatch 64, gamma 0.99, GAE lambda 0.95, clip 0.2. Policy and value nets are separate 64-64 tanh MLPs
# (the paper's architecture, also the Stable-Baselines3 default).
PPO_KWARGS = dict(
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
)
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))

# VecNormalize defaults, repeated here because the exported controller has to apply the same transform.
CLIP_OBS = 10.0
NORM_EPS = 1e-8

# Domain randomization: every body's mass and inertia scaled by one factor, every contact's friction by another,
# both drawn uniformly at each reset.
DR_MASS = (0.8, 1.2)
DR_FRICTION = (0.7, 1.3)

# Evaluation episodes use their own seeds, never used to reset a training episode.
EVAL_EPISODES = 20
EVAL_SEED = 10_000

# Model selection: every seed is scored on validation episodes whose physics is drawn from the randomization
# ranges (the conditions a deployed robot meets), seeds SELECT_SEED.., disjoint from training and evaluation.
SELECT_EPISODES = 20
SELECT_SEED = 20_000

# Robustness sweep: one physics parameter scaled at a time, inside and beyond the randomization range.
MASS_GRID = (0.6, 0.8, 1.0, 1.2, 1.4)
FRICTION_GRID = (0.4, 0.7, 1.0, 1.3, 1.6)
SWEEP_EPISODES = 10


def run_name(seed: int, condition: str, tag: str = "") -> str:
    return f"{condition}_s{seed}" + (f"_{tag}" if tag else "")
