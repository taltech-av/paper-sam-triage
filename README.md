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

The result is a refined `vlm/annotation/` folder that is a drop-in replacement for `annotation_sam/` in the fusion trainer.

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
        ├─► ConsistencyAgent → pass / fail               (deterministic LiDAR support, no VLM)
        ├─► BBoxAgent        → valid / invalid /
        │                      background               (VLM, blind forced-choice on zoomed crop)
        ├─► QualityAgent     → good / bad               (VLM, overlay crop)
        ├─► CorrectionAgent  → refine / no_refine   (VLM, refine path only)
        └─► FailureModeAgent → boundary_drift / hallucination /
                               occlusion_miss / fragmentation   (--diagnose, analysis only)
                │
                ▼ core/triage.py (concordance rules)
          accept / refine / reject / human_review
                │
        ┌───────┴────────┐
        ▼                ▼
vlm/annotation/     vlm/results/
  frame_N.png        frame_N.json + summary.json
```

Every VLM judgment is made on a per-mask zoomed crop (≥224 px). Full-frame
multi-object queries are never used — a 7B VLM cannot resolve 10–30 px objects
or numbered box labels at full-frame scale, which causes mass false rejections.

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

Outputs written to a single `vlm/` folder:

```
zod_temp/
└── vlm/
    ├── annotation/                # refined masks — same uint8 PNG format as annotation_sam
    │   └── frame_XXXXXX.png
    ├── results/
    │   ├── frame_XXXXXX.json      # per-mask agent decisions and triage outcome
    │   └── summary.json           # aggregate counts by class and triage decision
    └── visualization/             # camera image with triage overlays
        └── frame_XXXXXX.png       # green=accept, yellow=refine, red=reject, blue=review
```

## Usage

**Dry run with mock VLM (no GPU needed):**
```bash
python process_frames.py --limit 10
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
--resume    Skip frames with existing results in vlm/results/
--limit N   Process at most N frames (0 = all)
--diagnose  Run the failure-mode agent on rejected/review masks (analysis only)
```

## HPC Deployment (A100 80 GB)

Recommended Ollama VLMs for running on an A100 80 GB GPU (e.g. HPC cluster node).

### Model Recommendations

| Model | VRAM (Q4) |
|---|---|
| `qwen2.5vl:72b` | ~45 GB |
| `llama3.2-vision:90b` | ~55 GB |

### HPC Usage

```bash
# Pull models on the compute node (requires internet or a local Ollama registry)
ollama pull qwen2.5vl:72b
ollama pull llama3.2-vision:90b

# Primary run — 72b model, all 2300 frames
ollama run qwen2.5vl:72b &
python process_frames.py --model qwen2.5vl:72b
```

On an A100 80 GB you can run `qwen2.5vl:72b` in Q4 (~45 GB) with headroom to spare. FP16 (~144 GB) requires two A100s with model sharding — Ollama handles this automatically when multiple GPUs are visible.

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

Implemented in [core/triage.py](core/triage.py). Rejection is destructive
(pixels are zeroed out), so it requires **two concordant negative signals**.
Rejecting on any single negative compounds the false-positive rates of the
individual judges (three judges at 10% FPR each would destroy ~27% of good
masks) — and since the SAM masks are seeded from ZOD ground-truth boxes, true
hallucinations are rare and single "reject" votes are mostly false positives.

**Reject** if at least two of:
- BBox agent → `invalid`
- Quality agent → `bad`
- Consistency → `fail`

**Reject** also when BBox agent → `background`: the model positively identified
a concrete non-object surface (vegetation, building, sky, snow) under the mask.
That is direct evidence of error and counts as two negatives on its own —
unlike `invalid` (wrong class, road, unclear), which is mere absence of
confirmation and needs corroboration.

The BBox agent asks a *blind forced-choice* question ("what does this crop
mainly show?") without revealing the proposed class — naming the expected class
in the prompt or image makes the model parrot it back, and yes/no presence
questions invite agreement. It sees only the zoomed crop: given the full frame,
a 7B VLM classifies the scene instead of the region ("highway → vehicle").

**Accept** if all of:
- BBox → `valid`, Quality → `good`, Consistency → `pass`

**Refine** if:
- BBox → `valid`, Quality → `good`, Consistency → `fail`, Correction → `refine`

**Human review** — all other cases (single negative signal). Review and refine
masks are kept in the output annotation; only rejects are zeroed.

The failure-mode agent is diagnostic only (`--diagnose`) and never affects the
decision. Early exit: when BBox → `invalid` and Consistency → `fail`, the
rejection is already decided and the quality call is skipped.

## Switching VLM Backends

The `VLMClient` ABC in [vlm/client.py](vlm/client.py) has one method:

```python
def query(self, images: list[PIL.Image], prompt: str) -> str: ...
```

Implement this interface for any other backend (Hugging Face Transformers, OpenAI-compatible API, etc.) and pass the client to `build_agents()` in [process_frames.py](process_frames.py).
