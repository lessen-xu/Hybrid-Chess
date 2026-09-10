#!/usr/bin/env bash
set -euo pipefail
test -n "${SLURM_JOB_ID:?Compute allocation required}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
python -m pytest tests/test_general_ai.py tests/test_general_gpu.py tests/test_az_encoding.py tests/test_az_inference_server.py -q --durations=5 -x
python -m scripts.general_benchmark --device cuda --moves 32
