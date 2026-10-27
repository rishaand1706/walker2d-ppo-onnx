"""Closed-loop and latency comparison of the exported controllers. Run on an idle machine, after export_onnx.py.

    python benchmark.py

1. Closed loop: the same EVAL_EPISODES seeds driven by the SB3 reference, the ONNX FP32 controller and the
   ONNX INT8 controller, under nominal physics.
2. Latency: one observation per call (how a control loop calls it), numpy in and numpy out, single thread,
   20,000 timed calls after 2,000 warm-up calls, on real walking states.

Writes results/benchmark.json.
"""
import json
import platform
import time

import gymnasium as gym
import numpy as np
import onnxruntime
import torch

from config import ENV_ID, EVAL_EPISODES, EVAL_SEED, MODELS, RESULTS
from controller import OnnxController, Sb3Controller, TorchController, load_run
from evaluate import episode
from export_onnx import FP32, INT8, rollout_observations

WARMUP, CALLS = 2_000, 20_000


class TorchPath:
    """The PyTorch module behind a numpy interface, so every path pays the same conversion costs."""

    def __init__(self, module):
        self.module = module

    @torch.inference_mode()
    def __call__(self, obs):
        return self.module(torch.from_numpy(np.asarray(obs, dtype=np.float32))[None])[0].numpy()


def latency_us(act, observations) -> dict:
    n = len(observations)
    for i in range(WARMUP):
        act(observations[i % n])
    times = np.empty(CALLS)
    for i in range(CALLS):
        obs = observations[i % n]
        t = time.perf_counter_ns()
        act(obs)
        times[i] = time.perf_counter_ns() - t
    times /= 1e3
    return dict(p50=round(float(np.percentile(times, 50)), 1), p99=round(float(np.percentile(times, 99)), 1),
                mean=round(float(times.mean()), 1))


def main():
    torch.set_num_threads(1)
    model, norm = load_run(MODELS, "policy_sb3.zip")  # the exported run, copied there by export_onnx.py
    controllers = {
        "SB3 predict()": Sb3Controller(model, norm),
        "PyTorch module": TorchPath(TorchController(model, norm).eval()),
        "ONNX Runtime FP32": OnnxController(FP32),
        "ONNX Runtime INT8": OnnxController(INT8),
    }

    closed = {}
    for name in ("SB3 predict()", "ONNX Runtime FP32", "ONNX Runtime INT8"):
        eps = [episode(gym.make(ENV_ID), controllers[name], EVAL_SEED + i) for i in range(EVAL_EPISODES)]
        closed[name] = dict(returns=[round(e["ret"], 1) for e in eps],
                            mean_return=round(float(np.mean([e["ret"] for e in eps])), 1),
                            std_return=round(float(np.std([e["ret"] for e in eps])), 1),
                            falls=sum(e["fell"] for e in eps))
    ref = np.array(closed["SB3 predict()"]["returns"])
    for name in ("ONNX Runtime FP32", "ONNX Runtime INT8"):
        closed[name]["max_abs_return_gap_vs_sb3"] = round(float(np.abs(np.array(closed[name]["returns"]) - ref).max()), 1)

    observations = rollout_observations(controllers["SB3 predict()"], episodes=3)
    latency = {name: latency_us(act, observations) for name, act in controllers.items()}

    dt = gym.make(ENV_ID).unwrapped.dt
    report = dict(closed_loop=closed, latency_us=latency, control_period_ms=round(dt * 1e3, 1),
                  machine=dict(cpu=platform.processor() or platform.machine(), python=platform.python_version(),
                               torch=torch.__version__, onnxruntime=onnxruntime.__version__, threads=1))
    (RESULTS / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    for name, c in closed.items():
        gap = f"  max |gap vs SB3| {c['max_abs_return_gap_vs_sb3']}" if "max_abs_return_gap_vs_sb3" in c else ""
        print(f"{name:18s} return {c['mean_return']:8.1f} +/- {c['std_return']:6.1f}  falls {c['falls']}{gap}")
    for name, l in latency.items():
        print(f"{name:18s} p50 {l['p50']:7.1f} us   p99 {l['p99']:7.1f} us")
    print(f"control period {report['control_period_ms']} ms")


if __name__ == "__main__":
    main()
