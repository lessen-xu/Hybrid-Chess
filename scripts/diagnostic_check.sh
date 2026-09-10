#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:?Compute allocation required}"
bash cpp/build.sh
python -m pytest tests/test_diagnose_ai.py -q -x
python -m pytest tests -q --ignore=tests/test_diagnose_ai.py --durations=10
