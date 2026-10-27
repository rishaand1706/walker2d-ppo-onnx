#!/usr/bin/env bash
# Every (algorithm, dataset, seed): 2 x 3 x 5 = 30 runs, JOBS at a time, TD3+BC first (it is ~3x slower than BC).
# A few hours on an M2 MacBook Air (a TD3+BC run is about an hour, six at a time). Run after `python -m offline.collect`.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs/offline
for a in td3bc bc; do for d in medium medium-expert expert; do for s in 0 1 2 3 4; do echo "$a $d $s"; done; done; done |
  xargs -P "${JOBS:-6}" -n 3 sh -c '.venv/bin/python -m offline.train --algo "$0" --dataset "$1" --seed "$2" > "runs/offline/log_$0_$1_s$2.txt" 2>&1 && tail -1 "runs/offline/log_$0_$1_s$2.txt"'
