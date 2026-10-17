#!/usr/bin/env bash
# One worker per GPU; complete model/seed blocks remain on the same shard.
set -euo pipefail
cd "$(dirname "$0")"
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo 'run_parallel.sh requires Bash >=5.1 (Ubuntu 22.04+); manual shard commands work on older Bash.' >&2
    exit 2
fi
if [ "$#" -lt 1 ]; then
    echo 'Usage: ./run_parallel.sh 0,1 [run options, e.g. --mode pilot --output results]' >&2
    exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$1"
shift
if [ "${#GPU_IDS[@]}" -eq 0 ]; then exit 2; fi
declare -A SEEN
for gpu_id in "${GPU_IDS[@]}"; do
    if ! [[ "$gpu_id" =~ ^[0-9]+$ ]] || [ -n "${SEEN[$gpu_id]:-}" ]; then
        echo "Invalid or duplicate GPU index: $gpu_id" >&2
        exit 2
    fi
    SEEN[$gpu_id]=1
done
LOG_DIR="${WORKER_LOG_DIR:-logs/parallel}"
mkdir -p "$LOG_DIR"
ACTIVE=()
stop_workers() {
    for pid in "${ACTIVE[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
trap 'stop_workers; exit 130' INT TERM
for i in "${!GPU_IDS[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS[$i]}" ./run.sh run "$@" \
        --num-shards "${#GPU_IDS[@]}" --shard-index "$i" \
        > "$LOG_DIR/worker-$i.log" 2>&1 &
    ACTIVE+=("$!")
    echo "GPU ${GPU_IDS[$i]}: worker $i PID $!, $LOG_DIR/worker-$i.log"
done
STATUS=0
while [ "${#ACTIVE[@]}" -gt 0 ]; do
    FINISHED_PID=''
    CODE=0
    wait -n -p FINISHED_PID "${ACTIVE[@]}" || CODE=$?
    if [ "$CODE" -eq 127 ] && [ -z "${FINISHED_PID:-}" ]; then
        # Bash may remove already-finished jobs from wait -n's eligible list.
        # Their recorded statuses remain available through an explicit wait.
        for pid in "${ACTIVE[@]}"; do
            CHILD_CODE=0
            wait "$pid" || CHILD_CODE=$?
            if [ "$CHILD_CODE" -ne 0 ] && [ "$CHILD_CODE" -ne 3 ]; then
                STATUS=$CHILD_CODE
            elif [ "$CHILD_CODE" -eq 3 ] && [ "$STATUS" -eq 0 ]; then
                STATUS=3
            fi
        done
        ACTIVE=()
        break
    fi
    NEXT=()
    for pid in "${ACTIVE[@]}"; do
        if [ "$pid" != "${FINISHED_PID:-}" ]; then NEXT+=("$pid"); fi
    done
    ACTIVE=("${NEXT[@]}")
    if [ "$CODE" -ne 0 ] && [ "$CODE" -ne 3 ]; then
        STATUS=$CODE
        stop_workers
        for pid in "${ACTIVE[@]}"; do wait "$pid" 2>/dev/null || true; done
        break
    elif [ "$CODE" -eq 3 ]; then
        STATUS=3
    fi
done
exit "$STATUS"
