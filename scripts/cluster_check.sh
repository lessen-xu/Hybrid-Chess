#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:?Compute allocation required}"
bash cpp/build.sh
python -m pytest tests -q --durations=10 "$@"
