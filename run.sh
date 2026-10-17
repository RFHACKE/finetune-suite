#!/usr/bin/env bash
# Thin launcher: activates the venv created by setup.sh and forwards all
# arguments to the suite CLI. With no arguments it starts the interactive
# setup prompts.
set -euo pipefail
cd "$(dirname "$0")"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ -d .venv ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
exec python -m rank_suite.cli "$@"
