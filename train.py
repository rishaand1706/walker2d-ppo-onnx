"""Train PPO on MuJoCo Walker2d-v5: one seed, one condition (see CONDITIONS in config.py).

    python train.py --seed 0 --condition norm
    python train.py --seed 0 --condition raw
    python train.py --seed 0 --condition dr
    python train.py --seed 0 --condition norm --timesteps 20480 --tag smoke

Writes runs/<condition>_s<seed>/: model.zip, vecnormalize.pkl (norm and dr only), train.monitor.csv (return
and length of every training episode), tb/ (TensorBoard) and config.json (settings, wall time, versions).
"""
import argparse
import json
import platform
import time

import gymnasium as gym
import mujoco
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import (CLIP_OBS, CONDITIONS, DR_FRICTION, DR_MASS, ENV_ID, NORM_EPS, POLICY_KWARGS, PPO_KWARGS,
                    RUNS, TIMESTEPS, run_name)
from physics import DomainRandomization


def make_env(condition: str, seed: int, run_dir):
    env = gym.make(ENV_ID)
    if condition == "dr":
        env = DomainRandomization(env, DR_MASS, DR_FRICTION, seed=seed)
    return Monitor(env, filename=str(run_dir / "train"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--timesteps", type=int, default=TIMESTEPS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    norm = args.condition in ("norm", "dr")

    # One thread per run, so several seeds can train side by side on a laptop CPU.
    torch.set_num_threads(1)
    run_dir = RUNS / run_name(args.seed, args.condition, args.tag)
    run_dir.mkdir(parents=True, exist_ok=True)

    venv = DummyVecEnv([lambda: make_env(args.condition, args.seed, run_dir)])
    if norm:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=CLIP_OBS,
                            gamma=PPO_KWARGS["gamma"], epsilon=NORM_EPS)

    model = PPO("MlpPolicy", venv, seed=args.seed, device="cpu", verbose=0,
                tensorboard_log=str(run_dir / "tb"), policy_kwargs=POLICY_KWARGS, **PPO_KWARGS)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps, tb_log_name="ppo")
    wall = time.time() - t0

    model.save(run_dir / "model.zip")
    if norm:
        venv.save(str(run_dir / "vecnormalize.pkl"))

    (run_dir / "config.json").write_text(json.dumps(dict(
        env_id=ENV_ID, seed=args.seed, condition=args.condition, normalize=norm, timesteps=args.timesteps,
        ppo=PPO_KWARGS, policy=POLICY_KWARGS, clip_obs=CLIP_OBS, norm_eps=NORM_EPS,
        domain_randomization=dict(mass=DR_MASS, friction=DR_FRICTION) if args.condition == "dr" else None,
        wall_seconds=round(wall, 1), steps_per_second=round(args.timesteps / wall, 1),
        versions=dict(python=platform.python_version(), torch=torch.__version__, sb3=stable_baselines3.__version__,
                      gymnasium=gym.__version__, mujoco=mujoco.__version__),
        machine=platform.machine(),
    ), indent=2) + "\n")
    print(f"{run_dir.name}: {args.timesteps} steps in {wall / 60:.1f} min ({args.timesteps / wall:.0f} steps/s)")


if __name__ == "__main__":
    main()
