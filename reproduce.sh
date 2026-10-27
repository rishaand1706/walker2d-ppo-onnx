#!/usr/bin/env bash
# Everything, in order, from an empty runs/ folder. PPO part about 35 min on an M2 MacBook Air; the 30 offline runs take a few hours.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
./run_sweep.sh                 # 3 conditions x 5 seeds x 1M steps -> runs/
$PY evaluate.py                # held-out eval, robustness sweep, validation scores -> results/*.csv, summary.json
$PY export_onnx.py             # best dr seed on validation -> models/*.onnx, results/export.json
$PY benchmark.py               # closed loop + latency -> results/benchmark.json (keep the machine idle)
$PY plot.py                    # results/learning_curves*.png, robustness*.png
$PY render.py                  # assets/walker.gif
$PY -m offline.collect         # expert + medium datasets from the PPO policies -> offline/data/, results/offline_data.json
./offline/run_all.sh           # BC and TD3+BC, 3 datasets x 5 seeds, plain PyTorch -> runs/offline/
$PY -m offline.summarize       # results/offline_*.csv/json, offline_curves*.png
$PY -m pytest -q               # ONNX file is the trained policy; offline data marks only real falls as terminal
