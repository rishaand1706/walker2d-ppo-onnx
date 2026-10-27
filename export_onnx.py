"""Export one trained run as a self-contained ONNX controller, check it against Stable-Baselines3, quantize it.

    python export_onnx.py                # the dr seed with the best validation score (results/selection.csv)
    python export_onnx.py --run norm_s3

Writes models/walker2d_ppo.onnx (FP32), models/walker2d_ppo_int8.onnx (dynamic INT8), copies of the source
policy and normalizer stats (models/policy_sb3.zip, models/vecnormalize.pkl) for the tests, and
results/export.json with the open-loop parity numbers.
"""
import argparse
import csv
import json
import logging
import shutil
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

from config import ENV_ID, EVAL_SEED, MODELS, RESULTS, RUNS
from controller import OnnxController, Sb3Controller, TorchController, load_run
from physics import FixedPhysics

logging.getLogger("torch.onnx").setLevel(logging.ERROR)
FP32, INT8 = MODELS / "walker2d_ppo.onnx", MODELS / "walker2d_ppo_int8.onnx"


def selected_run() -> str:
    """The domain-randomized seed with the best validation score; evaluation episodes play no part in the choice."""
    rows = [r for r in csv.DictReader(open(RESULTS / "selection.csv")) if r["condition"] == "dr"]
    return f"dr_s{max(rows, key=lambda r: float(r['select_return']))['seed']}"


def rollout_observations(act, episodes: int, mass: float = 1.0) -> np.ndarray:
    """Raw observations the controller actually meets, from closed-loop episodes."""
    env, out = FixedPhysics(gym.make(ENV_ID), mass=mass), []
    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + 500 + i)
        done = False
        while not done:
            out.append(obs)
            obs, _, terminated, truncated, _ = env.step(act(obs))
            done = terminated or truncated
    return np.array(out)


def export(ctrl: TorchController, path: Path):
    dummy = torch.zeros(1, ctrl.mean.shape[0])
    # external_data=False keeps the weights inside the .onnx file: one self-contained artifact to ship.
    torch.onnx.export(ctrl, (dummy,), str(path), input_names=["obs"], output_names=["action"],
                      dynamic_shapes={"obs": {0: torch.export.Dim("batch")}}, external_data=False, verbose=False)
    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    return model


def quantize(src: Path, dst: Path):
    # ORT's pre-processing re-runs shape inference first; without it the exporter's shape annotations clash.
    with tempfile.TemporaryDirectory() as tmp:
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(src), str(pre))
        quantize_dynamic(str(pre), str(dst), weight_type=QuantType.QInt8)


def diff(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(a - b)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    args = ap.parse_args()
    run = args.run or selected_run()
    run_dir = RUNS / run
    MODELS.mkdir(exist_ok=True)

    model, norm = load_run(run_dir)
    sb3 = Sb3Controller(model, norm)
    graph = export(TorchController(model, norm).eval(), FP32)
    quantize(FP32, INT8)
    shutil.copyfile(run_dir / "model.zip", MODELS / "policy_sb3.zip")
    shutil.copyfile(run_dir / "vecnormalize.pkl", MODELS / "vecnormalize.pkl")

    # Open-loop parity on three kinds of input: states from normal walking, states from an overloaded robot
    # (1.4x mass, outside anything seen in training), and synthetic extremes that hit the +/-10 clip.
    walk = rollout_observations(sb3, episodes=5)
    heavy = rollout_observations(sb3, episodes=5, mass=1.4)
    rng = np.random.default_rng(0)
    extreme = norm.mean + np.sqrt(norm.var) * rng.normal(0, 4, size=(2000, norm.mean.shape[0]))
    fp32, int8 = OnnxController(FP32), OnnxController(INT8)
    report = dict(run=run, opset=[o.version for o in graph.opset_import if o.domain == ""][0],
                  fp32_ops=[n.op_type for n in graph.graph.node],
                  int8_ops=[n.op_type for n in onnx.load(str(INT8)).graph.node],
                  bytes=dict(fp32=FP32.stat().st_size, int8=INT8.stat().st_size),
                  sidecar_files=sorted(f.name for f in MODELS.glob("*.data")), parity={})
    for name, obs in (("walking", walk), ("heavy_1.4x", heavy), ("extreme", extreme)):
        ref = np.array([sb3(o) for o in obs])
        report["parity"][name] = dict(n=len(obs), fp32_vs_sb3=diff(fp32(obs), ref), int8_vs_sb3=diff(int8(obs), ref))
    (RESULTS / "export.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"exported {run}: {FP32.name} {report['bytes']['fp32']} B, {INT8.name} {report['bytes']['int8']} B")
    print("FP32 graph:", " -> ".join(report["fp32_ops"]))
    for name, p in report["parity"].items():
        print(f"  {name:11s} n={p['n']:5d}  |fp32-sb3| max {p['fp32_vs_sb3']['max_abs']:.2e}"
              f"  |int8-sb3| max {p['int8_vs_sb3']['max_abs']:.2e} mean {p['int8_vs_sb3']['mean_abs']:.2e}")


if __name__ == "__main__":
    main()
