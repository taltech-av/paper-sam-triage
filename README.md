# SAM Annotation Triage Pipeline

Implementation for the paper:

> Swin–VLM Concordance Triage for Pseudo-Label Refinement in Autonomous Driving  
> Toomas Tahves, Mauro Bellone, Raivo Sell

| Resource | Link |
|---|---|
| SAM annotation generator (upstream) | https://github.com/taltech-av/paper-aim2026-zod-sam-generator |
| Fusion training framework (downstream) | https://github.com/taltech-av/paper-aim2026-fusion-trainer |

## Overview

SAM-generated pseudo-labels for autonomous driving contain systematic errors: boundary drift, hallucinated regions, and missed objects — particularly for safety-critical classes (cyclists, pedestrians, signs) that occupy a small fraction of pixels.

This pipeline triages each SAM mask proposal using three independent signals:

- **Swin quality agent** — domain-adapted segmentation model trained on clean ZOD frames; scores each mask by pixel-level agreement ratio α
- **BBox VLM agent** — blind forced-choice object classification on a zoomed crop; confirms or denies object presence
- **LiDAR consistency** — deterministic geometric check; flags masks with insufficient depth support

A **concordance-based triage rule** combines these signals: rejection requires at least two concordant negative signals, because a single negative from any one judge would compound false-positive rates across the pipeline. Review and refine masks are kept in the output; only rejects are zeroed.

A **discovery module** identifies objects SAM missed entirely: regions where Swin predicts a non-background class but SAM annotated background. VLM-confirmed candidates are added as new masks.

All five training-data variants are written in a single pipeline pass (see Output Layout). A separate `replay_triage.py` script can generate threshold-sweep variants offline from stored JSON outputs without re-running inference.

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
        ├─► ConsistencyAgent    → pass / fail         (deterministic LiDAR support)
        ├─► SwingQualityAgent   → good / bad           (Swin, once per frame)
        ├─► BBoxAgent           → valid / invalid /    (VLM, zoomed crop)
        │                         background
        ├─► CorrectionAgent     → refine / no_refine   (VLM, refine path only)
        └─► FailureModeAgent    → boundary_drift /     (--diagnose, analysis only)
                                  hallucination / ...
                │
                ▼ core/triage.py (concordance rules)
          accept / refine / reject / human_review
                │
        ┌───────┴───────────────────────────────────────┐
        │                                               │
        ▼ agents/discovery_agent.py                     ▼
  Swin-proposed missed objects → VLM confirm    All annotation variants written:
  → paint onto background pixels                annotation_full/  (triage + discovery)
                                                annotation_triage/
                                                annotation_raw_sam/
                                                annotation_swin_only/
                                                annotation_vlm_only/
```

Every VLM call is made on a per-mask zoomed crop (≥224 px, ≤512 px). Full-frame queries are never used — small objects (10–30 px) cannot be resolved at full-frame scale.

The Swin quality agent runs **once per frame**, not per mask. Its agreement score α is stored in the result JSON. A bypass threshold τ_skip (0.70 large / 0.40 small classes) marks masks where the BBox VLM call *could* be skipped; in normal runs all agents are always called so complete signal data is available for offline ablation via `replay_triage.py`.

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

Requires Python 3.10+, an [Ollama](https://ollama.com) server, and the
[fusion-training](https://github.com/taltech-av/paper-aim2026-fusion-trainer)
repo on the Python path (for the Swin quality model).

## Data Layout

Expected input (paths configured in [config.py](config.py)):

```
zod_temp/
├── annotation_sam/        # SAM-generated masks (upstream input)
├── camera/                # RGB frames (768px)
└── lidar_png/             # LiDAR depth projections (X/Y/Z encoded as RGB PNG)
```

Frame list: [frames/bad_frames.csv](frames/bad_frames.csv)

## Output Layout

All outputs live under `vlm/<tag>/` (tag = model name by default, override with `--tag`):

```
zod_temp/vlm/<tag>/
├── annotation_full/           # triage + exact discovery masks  →  CLFTv2 config D*
├── annotation_triage/         # triage only, no discovery       →  CLFTv2 config D
├── annotation_raw_sam/        # no triage, original SAM         →  CLFTv2 baseline
├── annotation_swin_only/      # Swin threshold only
├── annotation_vlm_only/       # BBox VLM + consistency only
├── results/
│   ├── frame_XXXXXX.json      # per-mask agent outputs, scores, triage decision
│   └── summary.json
└── visualization/
    └── frame_XXXXXX.png       # green=accept, yellow=refine, red=reject, blue=review
```

All five annotation folders are written during `process_frames.py` while pixel masks
are in memory. `annotation_full` is the only variant that contains exact
connected-component discovery masks; the others are triage-only.

After two HPC runs, use `merge_annotations.py` to produce a consensus annotation from
both VLM runs (intersection: keep mask only if both VLMs agreed).

## Usage

**Test run (10 frames, mock VLM):**
```bash
python process_frames.py --mock --limit 10
```

**Full run:**
```bash
python process_frames.py --model qwen2.5vl:72b --resume
python process_frames.py --model llama3.2-vision:90b --resume
```

**Fall back to VLM quality agent (no Swin):**
```bash
python process_frames.py --no-swin
```

**Analyse results:**
```bash
python analyze_results.py --tag qwen2.5vl_72b
```

**Replay triage with different thresholds (no GPU needed):**
```bash
python replay_triage.py --variant swin_only --swin-threshold 0.25 --tag qwen2.5vl_72b
python replay_triage.py --list-variants
```

**Merge two VLM runs into a consensus annotation:**
```bash
python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llava_34b
python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llava_34b --rule union
```

**All `process_frames.py` options:**
```
--model       Ollama model name                         (default: qwen2.5vl:7b)
--tag         Output namespace under vlm/<tag>/         (default: sanitised model name)
--no-swin     Use VLM quality agent instead of Swin
--no-discovery  Disable missed-object discovery
--mock        Use deterministic mock client (no Ollama needed)
--resume      Skip frames with existing results
--limit N     Process at most N frames
--workers N   Parallel frame workers
--frames      Override frame list CSV
--diagnose    Run failure-mode agent on rejected/review masks
--hpc         Switch all paths to HPC
```

## HPC Deployment (A100 80 GB)

SLURM script: [slurms/pipeline.slurm](slurms/pipeline.slurm)

The `--hpc` flag switches all data and model paths to the HPC filesystem. Two separate
runs are needed — one per VLM — so results are tagged separately:

```bash
# Run 1: Qwen-72B (model name becomes the tag automatically)
sbatch slurms/pipeline.slurm   # MODEL=qwen2.5vl:72b  → tag: qwen2.5vl_72b

# Run 2: edit MODEL in pipeline.slurm to llama3.2-vision:90b, then:
sbatch slurms/pipeline.slurm
```

**Prerequisite:** the [fusion-training](https://github.com/taltech-av/paper-aim2026-fusion-trainer)
repo must be present at the HPC path configured in `config.use_hpc()`. Sync it once:

```bash
rsync -avP --exclude='venv/' --exclude='logs/' --exclude='runs/' --exclude='__pycache__/' \
    /path/to/fusion-training/ \
    totahv@base.hpc.taltech.ee:/gpfs/mariana/smbhome/totahv/fusion-training/
```

| Model | VRAM (Q4) | Use |
|---|---|---|
| `qwen2.5vl:7b` | ~5 GB | local dev |
| `qwen2.5vl:72b` | ~40 GB | HPC run A |
| `llama3.2-vision:90b` | ~55 GB | HPC run B |

`OLLAMA_NUM_PARALLEL=2` in the SLURM script allows two concurrent requests on the
80 GB A100 while keeping both VLM copies in VRAM.

## Repository Structure

```
├── agents/
│   ├── base.py                # BaseAgent ABC: run(bundle) → str
│   ├── bbox_agent.py          # valid / invalid / background  (VLM)
│   ├── quality_agent.py       # good / bad  (VLM fallback, --no-swin)
│   ├── swin_quality_agent.py  # good / bad  (default; Swin segmentation model)
│   ├── failure_mode_agent.py  # boundary_drift / hallucination / ...  (--diagnose)
│   ├── correction_agent.py    # refine / no_refine  (VLM, refine path only)
│   ├── consistency_agent.py   # pass / fail  (deterministic)
│   └── discovery_agent.py     # find + VLM-confirm missed objects
├── core/
│   ├── mask_extractor.py      # annotation_sam PNG → MaskProposal objects
│   ├── bundle.py              # build RGB/overlay/depth crops + metadata
│   └── triage.py              # deterministic accept/refine/reject/review rules
├── vlm/
│   ├── client.py              # VLMClient ABC
│   ├── ollama_client.py       # Ollama REST backend
│   └── mock_client.py         # deterministic mock for testing
├── output/
│   ├── annotation_writer.py   # write annotation_full PNG (triage + discovery)
│   └── results_writer.py      # write per-frame JSON and summary
├── frames/
│   └── bad_frames.csv         # frame list to process
├── slurms/
│   └── pipeline.slurm         # SLURM job (A100, Qwen-72B)
├── process_frames.py          # main entry point — runs pipeline, writes all variants
├── replay_triage.py           # offline threshold/rule sweeps from stored JSON
├── merge_annotations.py       # consensus annotation from two VLM run tags
├── analyze_results.py         # aggregated stats + paper-ready LaTeX snippets
├── visualize_results.py       # render triage overlays from results
└── config.py                  # paths, thresholds, class constants
```

## Triage Rules

Implemented in [core/triage.py](core/triage.py). Rejection is **destructive**
(pixels are zeroed), so it requires two concordant negative signals.

| Outcome | Condition |
|---|---|
| **Reject** | ≥ 2 negative signals |
| **Reject** | `bbox=background` alone (VLM identified a non-object surface — direct evidence, not absence of confirmation) |
| **Reject** | `bbox=invalid` + `consistency=pass` (LiDAR confirms real content exists but VLM found no expected object) |
| **Accept** | `bbox=valid` + `quality=good` + `consistency=pass` |
| **Refine** | `bbox=valid` + `quality=good` + `consistency=fail` + `correction=refine` |
| **Human review** | All other cases (single disagreeing signal) |

Negative signals: `bbox=invalid`, `quality=bad`, `consistency=fail`.

The failure-mode agent (`--diagnose`) is diagnostic only and never affects the decision.

## Annotation Variants

`replay_triage.py` generates variant annotations from stored JSON without re-running
inference. All variants treat metadata-prefilter rejections (extreme aspect ratio,
sub-minimum pixel count) the same way.

| Variant | Description | Directory |
|---|---|---|
| `raw_sam` | No triage — original SAM | `annotation_raw_sam/` |
| `swin_only` | Swin threshold only | `annotation_swin_only/` |
| `vlm_only` | BBox VLM + consistency | `annotation_vlm_only/` |
| `triage` | Full concordance, no discovery | `annotation_triage/` |
| — | Triage + exact discovery masks | `annotation_full/` (process_frames only) |

## VLM Backend

The `VLMClient` ABC in [vlm/client.py](vlm/client.py) has one method:

```python
def query(self, images: list[PIL.Image], prompt: str) -> str: ...
```

Implement this for any backend (Hugging Face, OpenAI-compatible, etc.) and pass
the client to `build_agents()` in [process_frames.py](process_frames.py).
