#!/usr/bin/env bash
# Convenience wrapper: runs the suite with Telegram start/done/failure
# notifications. Part of the notifier, NOT of the research suite — delete
# this file together with notify.py and the project is unaffected.
#
#   ./run_notify.sh run --models qwen2.5-1.5b --track both
#   ./run_notify.sh                       # interactive prompts, same as run.sh
#
# Unattended (survives SSH drop):
#   nohup ./run_notify.sh run --ranks 4 8 16 32 64 > train.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
exec python3 notify.py run -- ./run.sh "$@"
