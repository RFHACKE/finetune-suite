#!/usr/bin/env bash
# Resume the saved 512/1024 results with the existing budget guard and log.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

download_python="${MODAL_PYTHON:-}"
if [ -n "$download_python" ]; then
    if ! "$download_python" -c 'import modal' >/dev/null 2>&1; then
        echo 'MODAL_PYTHON must point to a Python interpreter with the Modal SDK installed.' >&2
        exit 1
    fi
else
    # The suite's scientific venv and the Modal CLI may use different Python
    # environments. Select one that actually has the cloud client installed.
    for download_candidate in "$PWD/.venv/bin/python" python3 /usr/bin/python3 python; do
        if command -v "$download_candidate" >/dev/null 2>&1 \
            && "$download_candidate" -c 'import modal' >/dev/null 2>&1; then
            download_python="$download_candidate"
            break
        fi
    done
    if [ -z "$download_python" ]; then
        echo 'Modal SDK not found. Install it in a Python environment, then set MODAL_PYTHON=/path/to/python.' >&2
        exit 1
    fi
fi

exec "$download_python" -u -m tools.watch_large_ranks --resume-download "$@"
