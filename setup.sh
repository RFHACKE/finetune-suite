#!/usr/bin/env bash
# Local virtualenv only. NVIDIA drivers belong to the Vast.ai host; PyTorch
# wheels supply the CUDA runtime. No CUDA toolkit/compiler or driver reinstall.
set -euo pipefail
cd "$(dirname "$0")"

CPU=0
PREFETCH=0
for arg in "$@"; do
    case "$arg" in
        --cpu) CPU=1 ;;
        --prefetch) PREFETCH=1 ;;
        --help|-h)
            echo 'Usage: ./setup.sh [--cpu] [--prefetch]'
            echo 'Default: torch 2.8.0 + CUDA 12.8, compatible with RTX 4090/5090.'
            echo '--cpu installs the preparation/analysis environment without CUDA wheels.'
            echo 'HF_TOKEN is read from the environment; Llama requires prior license acceptance.'
            exit 0 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

if [ "$CPU" -eq 0 ]; then
    if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null 2>&1; then
        echo 'No usable NVIDIA driver. Select a Vast.ai CUDA image, or use ./setup.sh --cpu.' >&2
        exit 1
    fi
    DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)"
    if [ "${DRIVER%%.*}" -lt 570 ]; then
        echo "Driver $DRIVER is too old for this CUDA 12.8 setup policy. Select a host with driver >=570." >&2
        exit 1
    fi
fi

supported_python() {
    "$1" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)' 2>/dev/null
}

if [ -d .venv ]; then
    if [ ! -x .venv/bin/python ] || ! supported_python .venv/bin/python \
        || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
        echo 'Existing .venv is incompatible. Rename it and rerun setup; it will not be deleted automatically.' >&2
        exit 1
    fi
else
    PY_CMD="${PYTHON:-}"
    if [ -n "$PY_CMD" ] && ! supported_python "$PY_CMD"; then
        echo 'PYTHON must name a working Python 3.10–3.12 interpreter.' >&2
        exit 1
    fi
    if [ -z "$PY_CMD" ]; then
        for candidate in python3.11 python3.12 python3.10 python3; do
            if command -v "$candidate" >/dev/null && supported_python "$candidate"; then
                PY_CMD="$candidate"
                break
            fi
        done
    fi
    if [ -n "$PY_CMD" ] && supported_python "$PY_CMD" && "$PY_CMD" -m venv .venv; then
        :
    else
        # Bootstrap a pinned uv wheel into this repository, without changing system pip.
        if ! command -v python3 >/dev/null; then
            echo 'Python 3 is required to bootstrap. Use an Ubuntu/PyTorch Vast.ai image.' >&2
            exit 1
        fi
        python3 -m pip install --disable-pip-version-check --target .bootstrap uv==0.9.5
        export UV_PYTHON_INSTALL_DIR="$PWD/.tools/python"
        # A failed stdlib venv can leave a partial directory created above.
        # This branch is reached only when .venv did not pre-exist this invocation.
        .bootstrap/bin/uv venv --allow-existing --python 3.11 --seed .venv
    fi
fi

PY_EXE="$PWD/.venv/bin/python"
"$PY_EXE" -m pip install --disable-pip-version-check --upgrade pip==25.3 wheel==0.45.1
install_packages() {
    if command -v uv >/dev/null 2>&1; then
        # uv downloads independent CUDA wheels concurrently; use it when supplied
        # by the image, retaining pip as a dependency-free installer fallback.
        UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}" \
            uv pip install --python "$PY_EXE" "$@"
    else
        "$PY_EXE" -m pip install "$@"
    fi
}
if [ "$CPU" -eq 1 ]; then
    TORCH_INDEX=https://download.pytorch.org/whl/cpu
    TORCH_BUILD=2.8.0+cpu
else
    TORCH_INDEX=https://download.pytorch.org/whl/cu128
    TORCH_BUILD=2.8.0+cu128
fi
# Fail on an incompatible wheel/driver instead of silently changing CUDA or torch versions.
install_packages "torch==$TORCH_BUILD" --index-url "$TORCH_INDEX"
install_packages -r requirements.txt
"$PY_EXE" -m pip check

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ "$CPU" -eq 1 ]; then
    "$PY_EXE" -m rank_suite.cli doctor --cpu
else
    "$PY_EXE" -m rank_suite.cli doctor
fi
"$PY_EXE" -m pip freeze > setup-freeze.txt
if [ "$PREFETCH" -eq 1 ]; then
    "$PY_EXE" -m rank_suite.cli prepare
fi
printf '\nSetup complete. Read docs/CLOUD_RUNBOOK.md before renting a long session.\n'
printf 'Next: ./run.sh plan\n      ./run.sh run --mode smoke --models qwen2.5-1.5b\n      ./run.sh run --mode pilot\n'
printf 'Llama requires license acceptance and HF_TOKEN, or .venv/bin/hf auth login.\n'
