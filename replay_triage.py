#!/usr/bin/env python3
"""
Replay triage rules offline from saved result JSONs → write annotation PNGs.

Reads vlm/results/frame_*.json (raw agent outputs + scores saved during the
pipeline run), applies a triage variant, then writes refined annotation PNGs
to a named output folder — no GPU or Ollama required.

Each variant is a separate training dataset for ablation experiments:

    full        Current concordance rules  (bbox + Swin quality + consistency)
    swin_only   Swin agreement threshold only  (no VLM at all)
    vlm_only    bbox + consistency only  (as if Swin never ran)
    no_bypass   Full rules but τ_skip=1.0  (always calls bbox VLM, no fast path)

Usage:
    python replay_triage.py --variant swin_only
    python replay_triage.py --variant full --out-suffix _v2
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

def _triage_full(mask: dict, swin_q: float, swin_skip: float) -> str:
    """Current pipeline rules, replayed from stored agent outputs + scores."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")

    if swin is not None and swin >= swin_skip:
        bbox_out = "valid"
        quality_out = "good"
    else:
        bbox_out = ag.get("bbox")
        quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")

    return triage(bbox_out, quality_out, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_swin_only(mask: dict, swin_q: float, **_) -> str:
    """Reject based solely on Swin agreement — no VLM signals used."""
    swin = mask.get("scores", {}).get("swin_agreement")
    if swin is None:
        return "accept"  # no Swin data (e.g. class not in PIPELINE_TO_SWIN) → keep
    return TRIAGE_REJECT if swin < swin_q else "accept"


def _triage_vlm_only(mask: dict, **_) -> str:
    """bbox + consistency only — as if Swin quality agent never ran."""
    ag = mask["agents"]
    return triage(ag.get("bbox"), None, None, ag.get("correction"), ag.get("consistency")).decision


def _triage_no_bypass(mask: dict, swin_q: float, **_) -> str:
    """Full concordance rules but τ_skip=1.0 — Swin scores quality, bbox always called."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    return triage(ag.get("bbox"), quality_out, None, ag.get("correction"), ag.get("consistency")).decision


VARIANTS = {
    "full":       (_triage_full,      "Current pipeline: Swin quality + BBox VLM + consistency"),
    "swin_only":  (_triage_swin_only, "Swin agreement threshold only — no VLM"),
    "vlm_only":   (_triage_vlm_only,  "BBox VLM + consistency — no Swin quality"),
    "no_bypass":  (_triage_no_bypass, "Full rules with τ_skip=1.0 (Swin quality, bypass disabled)"),
}


# ── Annotation writer (mirrors annotation_writer.py logic) ───────────────────

def _write_annotation(frame_id: str, decisions: dict[int, str], out_dir: Path) -> None:
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann_path.exists():
        return
    ann = np.array(Image.open(ann_path))

    from core.mask_extractor import extract_proposals
    proposals = extract_proposals(ann_path, frame_id)

    for p in proposals:
        if decisions.get(p.mask_id) == TRIAGE_REJECT:
            ann[p.pixel_mask] = 0

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ann.astype(np.uint8)).save(out_dir / f"{frame_id}.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"],
                        help="Triage variant to apply, or 'all' to generate every variant (default: all)")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to the output folder name, e.g. '_v2'")
    parser.add_argument("--swin-threshold", type=float, default=config.SWIN_AGREEMENT_THRESHOLD,
                        help=f"Swin quality threshold τ_q (default: {config.SWIN_AGREEMENT_THRESHOLD})")
    parser.add_argument("--swin-skip", type=float, default=config.SWIN_SKIP_BBOX_THRESHOLD,
                        help=f"Swin bypass threshold τ_skip (default: {config.SWIN_SKIP_BBOX_THRESHOLD})")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    args = parser.parse_args()

    if args.list_variants:
        for name, (_, desc) in VARIANTS.items():
            print(f"  {name:12s}  {desc}")
        return

    if args.hpc:
        config.use_hpc()

    result_files = sorted(config.RESULTS_DIR.glob("frame_*.json"))
    if not result_files:
        print(f"No results found in {config.RESULTS_DIR}")
        return

    variants_to_run = list(VARIANTS.items()) if args.variant == "all" else [(args.variant, VARIANTS[args.variant])]

    for variant_name, (fn, desc) in variants_to_run:
        out_dir = config.OUTPUT_ROOT / f"annotation_{variant_name}{args.out_suffix}"
        print(f"\nVariant : {variant_name}  —  {desc}")
        print(f"τ_q     : {args.swin_threshold}   τ_skip: {args.swin_skip}")
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

            _write_annotation(frame_id, decisions, out_dir)

        total = sum(totals.values())
        for k, v in sorted(totals.items()):
            if v:
                print(f"  {k:15s}: {v:4d}  ({100*v/total:.0f}%)")

    print(f"\nDone. {len(result_files)} frames × {len(variants_to_run)} variants.")


if __name__ == "__main__":
    main()
