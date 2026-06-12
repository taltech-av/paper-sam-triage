#!/bin/bash
#SBATCH --job-name=vlm-pipeline
#SBATCH --gres=gpu:A100:1
#SBATCH --constraint=A100-80
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --account=project_ito_llm
#SBATCH --output=/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline/logs/pipeline_%j.log
#SBATCH --error=/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline/logs/pipeline_%j.log

PIPELINE_DIR="/gpfs/mariana/smbhome/totahv/paper-vlm-annotation-pipeline"
MODEL="qwen2.5vl:72b"

export PATH="$HOME/bin:$PATH"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_MODELS="/gpfs/mariana/smbhome/totahv/ollama_models"
export LD_LIBRARY_PATH="$PIPELINE_DIR/cuda/lib64:$LD_LIBRARY_PATH"

mkdir -p "$PIPELINE_DIR/logs"

# ── Ollama ───────────────────────────────────────
echo "[$(date)] Ollama version: $(ollama --version)"

# ── Venv ───────────────────────────────────
source "$PIPELINE_DIR/venv/bin/activate"

# ── Ollama server ─────────────────────────────────────────────────────────────
echo "[$(date)] Starting Ollama..."
ollama serve &
SERVER_PID=$!
until curl -sf "http://localhost:11434" > /dev/null; do sleep 2; done
echo "[$(date)] Ollama ready."

# ── Pull model if not already local ──────────────────────────────────────────
if ! ollama list | grep -q "$MODEL"; then
    echo "[$(date)] Pulling $MODEL (~45 GB)..."
    ollama pull "$MODEL"
fi

echo "[$(date)] Warming up $MODEL..."
ollama run "$MODEL" <<< "warmup"

# ── Run pipeline ──────────────────────────────────────────────────────────────
echo "[$(date)] Starting pipeline..."
cd "$PIPELINE_DIR"
python process_frames.py --model "$MODEL" --resume --hpc

echo "[$(date)] Pipeline complete."
kill $SERVER_PID
