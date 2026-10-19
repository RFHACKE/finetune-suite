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

# A systemd service survives this shell and the Codex session. Foreground mode
# remains available for hosts without a running system manager.
download_foreground=0
download_help=0
download_args=()
download_root=Finetune-result/large-ranks
download_root_next=0
for download_arg in "$@"; do
    if [ "$download_root_next" -eq 1 ]; then
        download_root="$download_arg"
        download_root_next=0
    fi
    case "$download_arg" in
        --foreground) download_foreground=1 ;;
        --help|-h) download_help=1; download_args+=("$download_arg") ;;
        --root) download_root_next=1; download_args+=("$download_arg") ;;
        --root=*) download_root="${download_arg#--root=}"; download_args+=("$download_arg") ;;
        *) download_args+=("$download_arg") ;;
    esac
done
if [ "$download_help" -eq 1 ]; then
    echo 'start-download.sh: resume downloads in a background systemd service when available.'
    echo '--foreground runs in this terminal instead; MODAL_PYTHON overrides Python selection.'
    exec "$download_python" -u -m tools.watch_large_ranks --resume-download "${download_args[@]}"
fi
download_scope=()
download_manager=0
if [ "$EUID" -eq 0 ] && systemctl show-environment >/dev/null 2>&1; then
    download_manager=1
elif systemctl --user show-environment >/dev/null 2>&1; then
    download_manager=1
    download_scope=(--user)
fi
if [ "$download_foreground" -eq 0 ] && command -v systemd-run >/dev/null 2>&1 \
    && [ "$download_manager" -eq 1 ]; then
    download_unit=fyp-result-download
    if [ "$download_root" != Finetune-result/large-ranks ]; then
        download_digest="$(printf '%s' "$download_root" | sha256sum)"
        download_unit="fyp-result-download-${download_digest:0:8}"
    fi
    if systemctl "${download_scope[@]}" is-active --quiet "$download_unit.service"; then
        echo 'The download supervisor is already running.'
        echo "Status: systemctl ${download_scope[*]} status $download_unit"
        exit 0
    fi
    case "$download_root" in
        /*) download_execution="$download_root/execution" ;;
        *) download_execution="$PWD/$download_root/execution" ;;
    esac
    mkdir -p "$download_execution"
    systemd-run "${download_scope[@]}" --quiet --collect --no-ask-password \
        --unit="$download_unit" --working-directory="$PWD" \
        --property=Restart=on-failure --property=RestartSec=30 \
        --property=StartLimitIntervalSec=3600 --property=StartLimitBurst=3 \
        --property=KillMode=mixed --property=TimeoutStopSec=75 \
        --property="StandardOutput=append:$download_execution/download-supervisor.log" \
        --property="StandardError=append:$download_execution/download-supervisor.log" \
        "$download_python" -u -m tools.watch_large_ranks --resume-download "${download_args[@]}"
    echo 'Download supervisor started independently of this terminal.'
    echo "Progress: $download_execution/download-progress.log"
    echo "Supervisor: $download_execution/download-supervisor.log"
    echo "Stop: systemctl ${download_scope[*]} stop $download_unit"
    exit 0
fi
echo 'Running the download supervisor in this terminal.'
exec "$download_python" -u -m tools.watch_large_ranks --resume-download "${download_args[@]}"
