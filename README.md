# SAM Annotation Triage Pipeline

Implementation for the paper:

> Deterministic Swin-Guided Multi-Signal Triage for Refining SAM Pseudo-Labels  
> Toomas Tahves, Mauro Bellone, Raivo Sell

| Resource | Link |
|---|---|
| SAM annotation generator (upstream) | https://github.com/taltech-av/paper-aim2026-zod-sam-generator |
| Fusion training framework (downstream) | https://github.com/taltech-av/paper-aim2026-fusion-trainer |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+, an [Ollama](https://ollama.com) server, and the
[fusion-training](https://github.com/taltech-av/paper-aim2026-fusion-trainer)
repo on the Python path (for the Swin quality model).

## Usage

**Test run (10 frames, no GPU needed):**
```bash
python process_frames.py --mock --limit 10
```

**Full run:**
```bash
python process_frames.py --model qwen2.5vl:72b --resume
python process_frames.py --model llava:34b --resume
```

**Merge two VLM runs into a consensus annotation:**
```bash
python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llava_34b
```

**Replay triage offline (no GPU, sweeps thresholds from stored JSON):**
```bash
python replay_triage.py --variant swin_only --swin-threshold 0.25 --tag qwen2.5vl_72b
python replay_triage.py --list-variants
```

**Analyse results:**
```bash
python analyze_results.py --tag qwen2.5vl_72b
```

## Models

| Model | VRAM (Q4) | Use |
|---|---|---|
| `qwen2.5vl:7b` | ~5 GB | local dev |
| `qwen2.5vl:72b` | ~40 GB | HPC run A |
| `llava:34b` | ~20 GB | HPC run B |

Pull before running:
```bash
OLLAMA_MODELS=/path/to/ollama_models ollama pull qwen2.5vl:72b
OLLAMA_MODELS=/path/to/ollama_models ollama pull llava:34b
```

## HPC Deployment (A100 80 GB)

```bash
sbatch slurms/vlm-qwen.slurm    # Qwen2.5-VL-72B  → tag: qwen2.5vl_72b
sbatch slurms/vlm-llava.slurm   # LLaVA-1.6-34B   → tag: llava_34b
```

Both scripts bind Ollama to port 11435 (`OLLAMA_HOST=0.0.0.0:11435`). After both
jobs complete, run `merge_annotations.py` for consensus labels.

## Key flags (`process_frames.py`)

| Flag | Description |
|---|---|
| `--model` | Ollama model name |
| `--tag` | Output namespace under `vlm/<tag>/` |
| `--resume` | Skip frames with existing results |
| `--no-swin` | Use VLM quality agent instead of Swin |
| `--no-discovery` | Disable missed-object discovery |
| `--diagnose` | Run failure-mode agent on rejected/review masks |
| `--hpc` | Switch all paths to HPC filesystem |
| `--limit N` | Process at most N frames |
| `--mock` | Deterministic mock client (no Ollama needed) |
