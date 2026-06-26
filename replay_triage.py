#!/usr/bin/env python3
"""
Replay triage rules offline from saved result JSONs → write annotation PNGs.

Reads vlm/results/frame_*.json (raw agent outputs + scores saved during the
pipeline run), applies a triage variant, then writes refined annotation PNGs
to a named output folder — no GPU or Ollama required.

Each variant is a separate training dataset for CLFTv2 ablations:

    raw_sam      No triage — original SAM annotation unchanged (baseline B)
    swin_only    Swin agreement threshold only — no VLM signals
    vlm_only     BBox VLM + consistency — Swin quality ignored
    triage       Swin + VLM + consistency, no discovery (paper config D)
    with_bypass  Simulate bypass: skip VLM for masks where swin_bypass=True

For the full system (triage + discovery), use annotation_full/ written by
process_frames.py — it holds the exact connected-component discovery masks.

Usage:
    python replay_triage.py --variant raw_sam
    python replay_triage.py --variant swin_only
    python replay_triage.py --variant triage --out-suffix _v2
    python replay_triage.py --variant swin_only --swin-threshold 0.25
    python replay_triage.py --list-variants
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from core.triage import TRIAGE_REJECT, triage


# ── Triage variants ───────────────────────────────────────────────────────────

def _triage_raw_sam(_: dict, **__) -> str:
    """Accept all masks — original SAM annotation unchanged. Baseline for downstream training."""
    return "accept"


def _triage_full(mask: dict, swin_q: float, **_) -> str:
    """Current pipeline: Swin quality + BBox VLM + consistency, all signals always collected."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    return triage(ag.get("bbox"), quality_out, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_swin_only(mask: dict, swin_q: float, **_) -> str:
    """Reject based solely on Swin agreement — no VLM signals used."""
    swin = mask.get("scores", {}).get("swin_agreement")
    if swin is None:
        return "accept"  # no Swin data (e.g. class not in PIPELINE_TO_SWIN) → keep
    return TRIAGE_REJECT if swin < swin_q else "accept"


def _triage_vlm_only(mask: dict, **_) -> str:
    """BBox VLM + consistency only — Swin quality signal ignored."""
    ag = mask["agents"]
    return triage(ag.get("bbox"), None, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_with_bypass(mask: dict, swin_q: float, **_) -> str:
    """Simulate old bypass: masks where swin_bypass=True skip BBox VLM (treat as valid+good).
    Useful for measuring the accuracy vs. efficiency trade-off of the bypass optimization."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")

    if scores.get("swin_bypass", False):
        bbox_out = "valid"
        quality_out = "good"
    else:
        bbox_out = ag.get("bbox")
        quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")

    return triage(bbox_out, quality_out, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_swin_protected(mask: dict, swin_q: float, **_) -> str:
    """Full triage with Swin protection: bbox=invalid+lidar=pass only counts as
    two negatives when Swin quality also disagrees. Reduces VLM over-rejection
    of small/thin objects (signs) where pixel-level Swin agreement is high."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    return triage(ag.get("bbox"), quality_out, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_disjunctive_reject(mask: dict, swin_q: float, **_) -> str:
    """Ablation: reject if ANY single negative signal is present (instead of requiring ≥2).
    Shows the cost of the concordance rule — disjunctive rejection is more aggressive
    and expected to over-reject valid masks, especially for rare classes."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    bbox = ag.get("bbox")
    if bbox in ("invalid", "background") or quality_out == "bad" or ag.get("consistency") == "fail":
        return TRIAGE_REJECT
    return "accept"


def _triage_uniform_tau(mask: dict, swin_q: float, **_) -> str:
    """Ablation: uniform Swin threshold τ_q=0.30 for all classes (no class-aware overrides).
    Cyclists and pedestrians have per-class τ_q=0.15 in the full pipeline because Swin's
    recall on small objects is much lower. This variant applies 0.30 uniformly, which
    over-rejects small-object masks — shows the value of class-aware thresholds."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    # Apply threshold uniformly; do NOT fall back to stored quality (which has per-class
    # thresholds baked in from the pipeline run).
    if swin is not None:
        quality_out = "good" if swin >= swin_q else "bad"
    else:
        quality_out = ag.get("quality")
    return triage(ag.get("bbox"), quality_out, None, ag.get("correction"), ag.get("consistency")).decision


VARIANTS = {
    "raw_sam":            (_triage_raw_sam,            "No triage — original SAM annotation unchanged (baseline)"),
    "swin_only":          (_triage_swin_only,          "Swin agreement threshold only — no VLM"),
    "vlm_only":           (_triage_vlm_only,           "BBox VLM + consistency — Swin quality ignored"),
    "triage":             (_triage_full,               "Swin + VLM + consistency triage, no discovery"),
    "swin_protected":     (_triage_swin_protected,     "Full triage + Swin protects valid masks from VLM false negatives"),
    "with_bypass":        (_triage_with_bypass,        "Simulate bypass: skip VLM for masks where swin_bypass=True"),
    "disjunctive_reject": (_triage_disjunctive_reject, "Ablation: any single negative signal rejects (vs. concordance ≥2)"),
    "uniform_tau":        (_triage_uniform_tau,        "Ablation: uniform τ_q=0.30 for all classes (no class-aware thresholds)"),
}


# ── Annotation writer (mirrors annotation_writer.py logic) ───────────────────

def _add_discoveries(ann: np.ndarray, discovered: list, confirmed_only: bool) -> None:
    """Paint Swin-detected discovery objects onto ann in-place.

    bbox_orig format is [x1, y1, x2, y2] (col_start, row_start, col_end, row_end).
    Only background pixels (value 0) are overwritten so SAM proposals that survived
    triage are never displaced.  This is a bounding-box approximation of the
    connected-component masks used during the live pipeline run.
    """
    for obj in discovered:
        if confirmed_only and not obj.get("confirmed"):
            continue
        x1, y1, x2, y2 = obj["bbox_orig"]
        class_id = obj["class_id"]
        region = ann[y1:y2, x1:x2]
        region[region == 0] = class_id


def _write_annotation(
    frame_id: str,
    decisions: dict[int, str],
    out_dir: Path,
    discovered: list | None = None,
    discovery_confirmed_only: bool = True,
) -> None:
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann_path.exists():
        return
    ann = np.array(Image.open(ann_path))

    from core.mask_extractor import extract_proposals
    proposals = extract_proposals(ann_path, frame_id)

    for p in proposals:
        if decisions.get(p.mask_id) == TRIAGE_REJECT:
            ann[p.pixel_mask] = 0

    if discovered:
        _add_discoveries(ann, discovered, confirmed_only=discovery_confirmed_only)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ann.astype(np.uint8)).save(out_dir / f"{frame_id}.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"],
                        help="Triage variant to apply, or 'all' to generate every variant (default: all)")
    parser.add_argument("--tag", default=None,
                        help="run tag to read results from (vlm/<tag>/results/). "
                             "Must match the --tag used during process_frames.py.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to the output folder name, e.g. '_v2'")
    parser.add_argument("--swin-threshold", type=float, default=config.SWIN_AGREEMENT_THRESHOLD,
                        help=f"Swin quality threshold τ_q (default: {config.SWIN_AGREEMENT_THRESHOLD})")
    parser.add_argument("--swin-skip", type=float, default=config.SWIN_SKIP_BBOX_THRESHOLD,
                        help=f"Swin bypass threshold τ_skip (default: {config.SWIN_SKIP_BBOX_THRESHOLD})")
    parser.add_argument("--with-discovery", action="store_true",
                        help="Add VLM-confirmed discovery objects from stored results (mirrors annotation_swin_discovery)")
    parser.add_argument("--with-discovery-all", action="store_true",
                        help="Add ALL Swin-detected discovery candidates, skipping VLM confirmation "
                             "(ablation: isolates value of VLM confirmation in discovery)")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    args = parser.parse_args()

    if args.list_variants:
        for name, (_, desc) in VARIANTS.items():
            print(f"  {name:12s}  {desc}")
        return

    if args.with_discovery and args.with_discovery_all:
        parser.error("--with-discovery and --with-discovery-all are mutually exclusive")

    use_discovery = args.with_discovery or args.with_discovery_all
    discovery_confirmed_only = not args.with_discovery_all

    if args.hpc:
        config.use_hpc()

    if args.tag:
        config.set_run_tag(args.tag)

    result_files = sorted(config.RESULTS_DIR.glob("frame_*.json"))
    if not result_files:
        print(f"No results found in {config.RESULTS_DIR}")
        return

    variants_to_run = list(VARIANTS.items()) if args.variant == "all" else [(args.variant, VARIANTS[args.variant])]

    disc_label = ""
    if args.with_discovery:
        disc_label = "_discovery"
    elif args.with_discovery_all:
        disc_label = "_discovery_noVLM"

    for variant_name, (fn, desc) in variants_to_run:
        out_dir = config.OUTPUT_ROOT / f"annotation_{variant_name}{disc_label}{args.out_suffix}"
        print(f"\nVariant : {variant_name}{disc_label}  —  {desc}")
        print(f"τ_q     : {args.swin_threshold}   τ_skip: {args.swin_skip}")
        if use_discovery:
            mode = "VLM-confirmed only" if discovery_confirmed_only else "ALL Swin candidates (no VLM filter)"
            print(f"Discovery: {mode}")
        print(f"Output  : {out_dir}")

        totals: dict[str, int] = {}

        for rf in tqdm(result_files, unit="frame", desc=variant_name):
            data = json.loads(rf.read_text())
            frame_id = data["frame_id"]
            decisions: dict[int, str] = {}

            for mask in data["masks"]:
                decision = fn(mask, swin_q=args.swin_threshold, swin_skip=args.swin_skip)
                decisions[mask["mask_id"]] = decision
                totals[decision] = totals.get(decision, 0) + 1

            discovered = data.get("discovered", []) if use_discovery else None
            _write_annotation(frame_id, decisions, out_dir,
                              discovered=discovered,
                              discovery_confirmed_only=discovery_confirmed_only)

        total = sum(totals.values())
        for k, v in sorted(totals.items()):
            if v:
                print(f"  {k:15s}: {v:4d}  ({100*v/total:.0f}%)")

    print(f"\nDone. {len(result_files)} frames × {len(variants_to_run)} variants.")


if __name__ == "__main__":
    main()
