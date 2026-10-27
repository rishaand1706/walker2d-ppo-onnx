#!/usr/bin/env bash
# Train every condition x seed, JOBS at a time (one CPU thread each). About 25 min on an M2 MacBook Air.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p runs
for c in norm raw dr; do for s in 0 1 2 3 4; do echo "$c $s"; done; done |
  xargs -P "${JOBS:-6}" -n 2 sh -c '.venv/bin/python train.py --condition "$0" --seed "$1" > "runs/log_$0_s$1.txt" 2>&1 && tail -1 "runs/log_$0_s$1.txt"'
