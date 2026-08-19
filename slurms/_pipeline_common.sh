# Shared driver for the VLM slurm jobs.
#
# Sourced by vlm-qwen.slurm / vlm-llava.slurm once they have set MODEL, TAG and
# PIPELINE_DIR. Starts Ollama, then runs process_frames.py under a restart
# loop.
#
# Why the loop exists: a few hours after each model load the ollama instance
# stops emitting a stop token and answers every call with garbage, until
# something restarts it. The 2026-06 qwen run lost 2,972 of its 4,135 frames to
# this because nothing detected it and nothing restarted the server, and the
# run still reported success. vlm/health.py now detects it and aborts with
# EXIT_HEALTH_ABORT; this loop supplies the missing half — reload the model and
# resume. Detection takes ~60 VLM calls, so an episode costs minutes, not the
# 14-44 hours it used to.
#
# Only EXIT_HEALTH_ABORT is retried. Any other non-zero status (missing model,
# bad path, unreadable frames file) is a fault a restart cannot fix, so the job
# fails immediately instead of burning the walltime rediscovering that.

EXIT_HEALTH_ABORT=42                      # must match vlm/health.py
MAX_RESTARTS="${MAX_RESTARTS:-20}"        # ceiling on reload attempts per job
OLLAMA_PORT="${OLLAMA_PORT:-11435}"
RESULTS_DIR="${RESULTS_DIR:-${VLM_HPC_DATA_ROOT:?set VLM_HPC_DATA_ROOT in .env}/vlm/${TAG}/results}"

completed_frames() {
    ls -1 "$RESULTS_DIR"/*.json 2>/dev/null | grep -cv 'summary' || true
}

start_ollama() {
    ollama serve &
    SERVER_PID=$!
    until curl -sf "http://localhost:${OLLAMA_PORT}" > /dev/null; do sleep 2; done
    echo "[$(date)] Ollama ready (pid $SERVER_PID)."
}

stop_ollama() {
    [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
    # Wait for the port to actually close. A half-dead server still answers the
    # readiness check, so without this the next start_ollama would return
    # immediately and hand the pipeline back the instance we just tried to kill.
    for _ in $(seq 30); do
        curl -sf "http://localhost:${OLLAMA_PORT}" > /dev/null 2>&1 || return 0
        sleep 2
    done
    echo "[$(date)] Port ${OLLAMA_PORT} still open after SIGTERM — forcing."
    pkill -f "ollama serve" 2>/dev/null
    sleep 5
}

run_pipeline_with_restarts() {
    local restarts=0
    # Consecutive aborts that completed no frames at all. A server already
    # degraded at job start fails the canary and aborts before writing
    # anything, which is precisely what a reload fixes.
    #
    # The default of 2 was set when the monitor only checked between frames, so
    # an abort still left the in-flight frames on disk and zero progress really
    # did mean nothing was working. The monitor now aborts mid-call and those
    # frames are discarded, so a zero-progress abort is the *normal* shape of a
    # short-lived episode — with several workers sharing the good calls a fresh
    # load provides, no single frame need reach the end. Raise MAX_STUCK (and
    # drop PIPELINE_WORKERS to 1) when resuming a set of frames that reliably
    # degrades the server.
    local stuck=0
    local max_stuck="${MAX_STUCK:-2}"

    start_ollama
    echo "[$(date)] Warming up $MODEL..."
    ollama run "$MODEL" <<< "warmup"

    while true; do
        echo "[$(date)] Starting pipeline (tag=$TAG, restarts=$restarts/$MAX_RESTARTS)..."
        local before
        before=$(completed_frames)

        # Fewer workers concentrate the good calls a fresh load provides into
        # one frame instead of splitting them across several unfinished ones,
        # which is what turns a zero-progress abort back into progress.
        local worker_args=()
        [ -n "${PIPELINE_WORKERS:-}" ] && worker_args=(--workers "$PIPELINE_WORKERS")

        python process_frames.py --model "$MODEL" --tag "$TAG" --resume --hpc \
            "${worker_args[@]}"
        local status=$?

        local after
        after=$(completed_frames)

        if [ $status -eq 0 ]; then
            echo "[$(date)] Pipeline complete. $after frames on disk."
            return 0
        fi

        if [ $status -ne $EXIT_HEALTH_ABORT ]; then
            echo "[$(date)] Pipeline failed with status $status — not a health"
            echo "           abort, so a model reload would not fix it. Giving up."
            return $status
        fi

        restarts=$((restarts + 1))
        echo "[$(date)] Health abort. Progress this attempt: $((after - before)) frames (total $after)."

        if [ $restarts -gt $MAX_RESTARTS ]; then
            echo "[$(date)] Hit MAX_RESTARTS=$MAX_RESTARTS. Stopping so the job does"
            echo "           not spin on a server that never recovers."
            return $EXIT_HEALTH_ABORT
        fi

        if [ "$after" -eq "$before" ]; then
            stuck=$((stuck + 1))
            if [ $stuck -ge $max_stuck ]; then
                echo "[$(date)] $stuck aborts in a row completed no frames — the model"
                echo "           is not recovering on reload. Stopping."
                return $EXIT_HEALTH_ABORT
            fi
        else
            stuck=0
        fi

        echo "[$(date)] Reloading $MODEL..."
        stop_ollama
        start_ollama
        ollama run "$MODEL" <<< "warmup"
    done
}
