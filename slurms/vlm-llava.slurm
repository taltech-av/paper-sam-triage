#!/bin/bash
#SBATCH --job-name=vlm-pip
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --account=project_ito_llm
#SBATCH --output=/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline/logs/pipeline_%j.log
#SBATCH --error=/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline/logs/pipeline_%j.log

PIPELINE_DIR="/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline"
MODEL="llava:34b"
# The 2026-06 llava run came through the serving fault clean, so this tag is
# resumable as-is; a rerun is only needed if the prompts change.
TAG="llava_34b"

export PATH="$HOME/bin:$PATH"
export OLLAMA_HOST="0.0.0.0:11435"
export OLLAMA_CONTEXT_LENGTH=4096
export OLLAMA_MODELS="/gpfs/mariana/smbhome/totahv/ollama_models"
export OLLAMA_NUM_PARALLEL=2
export LD_LIBRARY_PATH="$PIPELINE_DIR/cuda/lib64:$LD_LIBRARY_PATH"

mkdir -p "$PIPELINE_DIR/logs"

# ── Ollama ───────────────────────────────────────
echo "[$(date)] Ollama version: $(ollama --version)"

# ── Venv ───────────────────────────────────
source "$PIPELINE_DIR/venv/bin/activate"

cd "$PIPELINE_DIR"
source "$PIPELINE_DIR/slurms/_pipeline_common.sh"

# ── Pull model if not already local ──────────────────────────────────────────
start_ollama
if ! ollama list | grep -q "$MODEL"; then
    echo "[$(date)] Pulling $MODEL (~20 GB)..."
    ollama pull "$MODEL"
fi
stop_ollama

# ── Run pipeline (restarts Ollama on a health abort, resumes each time) ──────
run_pipeline_with_restarts
status=$?

stop_ollama
echo "[$(date)] Job finished with status $status."
exit $status
