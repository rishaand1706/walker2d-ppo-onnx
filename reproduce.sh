#!/usr/bin/env bash
# Everything, in order, from an empty runs/ folder. About 35 min on an M2 MacBook Air (training is 30 of it).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
./run_sweep.sh                 # 3 conditions x 5 seeds x 1M steps -> runs/
$PY evaluate.py                # held-out eval, robustness sweep, validation scores -> results/*.csv, summary.json
$PY export_onnx.py             # best dr seed on validation -> models/*.onnx, results/export.json
$PY benchmark.py               # closed loop + latency -> results/benchmark.json (keep the machine idle)
$PY plot.py                    # results/learning_curves*.png, robustness*.png
$PY render.py                  # assets/walker.gif
$PY -m pytest -q               # the shipped ONNX file is the trained policy
