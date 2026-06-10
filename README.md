# VLM Annotation Refinement

Implementation for the paper:

> Language-Guided Pseudo-Label Refinement for Multi-Modal Semantic Segmentation  
> Toomas Tahves, Mauro Bellone, Raivo Sell

| Resource | Link |
|---|---|
| SAM annotation generator (upstream) | https://github.com/taltech-av/paper-aim2026-zod-sam-generator |
| Fusion training framework (downstream) | https://github.com/taltech-av/paper-aim2026-fusion-trainer |

## Overview

SAM-generated pseudo-labels for autonomous driving scenes contain systematic errors: boundary drift, hallucinated regions, and missed small objects — particularly for safety-critical classes (pedestrians, cyclists, traffic signs) that represent less than 1% of pixels.

This pipeline runs five specialized Visual-LLM agents over each SAM mask proposal. Their outputs are combined by a deterministic rule-based triage function to decide the fate of each mask:

| Decision | Meaning |
|---|---|
| `accept` | Mask is valid — keep as-is |
| `refine` | Mask is geometrically plausible but needs adjustment |
| `reject` | Mask is invalid — zero out pixels |
| `human_review` | Agents disagree — flag for manual inspection |

The result is a refined `annotation_vllm/` folder that is a drop-in replacement for `annotation_sam/` in the fusion trainer.

## Pipeline

```
annotation_sam/ + camera/ + lidar_png/
        │
        ▼ core/mask_extractor.py
  MaskProposal per connected component
        │
        ▼ core/bundle.py
  RGB crop + mask overlay + depth crop + metadata
        │
        ├─► BBoxAgent      → valid / invalid
        ├─► QualityAgent   → good / bad
        ├─► FailureModeAgent → boundary_drift / hallucination / occlusion_miss / fragmentation
        ├─► CorrectionAgent → refine / no_refine
        └─► ConsistencyAgent → pass / fail
                │
                ▼ core/triage.py (deterministic rules)
          accept / refine / reject / human_review
                │
        ┌───────┴────────┐
        ▼                ▼
annotation_vllm/    vllm_results/
  frame_N.png        frame_N.json + summary.json
```

## Class Encoding

| ID | Class |
|---|---|
| 0 | background |
| 1 | ignore |
| 2 | vehicle |
| 3 | sign |
| 4 | cyclist |
| 5 | pedestrian |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Requires Python 3.10+ and an [Ollama](https://ollama.com) server for production use.

## Data Layout

Expected input at `/run/media/tom/ml/zod_temp/` (configured in [config.py](config.py)):

```
zod_temp/
├── annotation_sam/        # SAM-generated masks (upstream input)
├── camera/                # RGB frames
└── lidar_png/             # LiDAR depth projections
```

Frame list lives in the repo at [frames/frames.txt](frames/frames.txt) — 2300 curated frames.

Outputs written to:

```
zod_temp/
├── annotation_vllm/                  # refined masks — same uint8 PNG format as annotation_sam
├── vllm_results/
│   ├── frame_XXXXXX.json             # per-mask agent decisions and triage outcome
│   └── summary.json                  # aggregate counts by class and triage decision
└── vllm_results_visualizations/      # camera image with green (accepted) / red (rejected) overlays
    └── frame_XXXXXX.png
```

## Usage

**Dry run with mock VLM (no GPU needed):**
```bash
python process_frames.py --mock --limit 10
python visualize_results.py
```

**Production run with Ollama:**
```bash
# Start Ollama with a vision model first:
ollama run qwen2.5vl:7b

python process_frames.py
python visualize_results.py
```

**Resume after interruption:**
```bash
python process_frames.py --resume
```

**All options:**
```
--model     Ollama model name                    (default: qwen2.5vl:7b)
--mock      Use deterministic mock client instead of Ollama
--resume    Skip frames with existing results in vllm_results/
--limit N   Process at most N frames (0 = all)
```

## Repository Structure

```
├── agents/
│   ├── base.py               # BaseAgent ABC: run(bundle) → str
│   ├── bbox_agent.py         # valid / invalid
│   ├── quality_agent.py      # good / bad
│   ├── failure_mode_agent.py # boundary_drift / hallucination / occlusion_miss / fragmentation
│   ├── correction_agent.py   # refine / no_refine
│   └── consistency_agent.py  # pass / fail
├── core/
│   ├── mask_extractor.py     # annotation_sam PNG → MaskProposal objects
│   ├── bundle.py             # build RGB/overlay/depth crops + metadata
│   └── triage.py             # deterministic accept/refine/reject/review rules
├── vlm/
│   ├── client.py             # VLMClient ABC
│   ├── ollama_client.py      # Ollama REST backend
│   └── mock_client.py        # deterministic mock for testing
├── output/
│   ├── annotation_writer.py  # write refined annotation PNG
│   └── results_writer.py     # write per-frame JSON and summary
├── frames/
│   └── frames.txt            # list of 2300 curated frames to process
├── process_frames.py         # main CLI entry point
├── visualize_results.py      # render green/red overlays from triage results
└── config.py                 # data paths, model settings, class constants
```

## Triage Rules

Implemented in [core/triage.py](core/triage.py) exactly as specified in the paper.

**Reject** (highest priority) if any of:
- BBox agent → `invalid`
- Quality agent → `bad`
- Failure-mode agent → `hallucination`

**Accept** if all of:
- BBox → `valid`, Quality → `good`, Consistency → `pass`

**Refine** if:
- Quality → `good`, Consistency → `fail`, Correction → `refine`

**Human review** — all other cases.

Early exit is applied: an `invalid` bbox skips all remaining agents; a `bad` quality result only additionally runs the failure-mode agent for diagnostics before stopping.

## Switching VLM Backends

The `VLMClient` ABC in [vlm/client.py](vlm/client.py) has one method:

```python
def query(self, images: list[PIL.Image], prompt: str) -> str: ...
```

Implement this interface for any other backend (Hugging Face Transformers, OpenAI-compatible API, etc.) and pass the client to `build_agents()` in [process_frames.py](process_frames.py).
