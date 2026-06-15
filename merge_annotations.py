#!/usr/bin/env python3
"""
Merge triage annotations from two VLM runs (Qwen-72B and Llama-90B) into a
consensus annotation set for downstream CLFTv2 training.

Reads vlm/<tag_a>/results/ and vlm/<tag_b>/results/, applies a consensus rule
per mask, and writes merged annotation PNGs.  Both runs must have processed the
same frame set (same annotation_sam source).

Consensus rules:
    intersection  Reject a mask if EITHER model rejected it (stricter, cleaner).
                  A mask survives only when both VLMs independently agreed to keep it.
    union         Reject a mask only if BOTH models rejected it (more permissive).

Usage:
    python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llama3.2_90b
    python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llama3.2_90b --rule union
    python merge_annotations.py --tag-a qwen2.5vl_72b --tag-b llama3.2_90b --hpc
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from core.mask_extractor import extract_proposals
from core.triage import TRIAGE_REJECT


def _load_decisions(results_dir: Path) -> dict[str, dict[int, str]]:
    """Return {frame_id: {mask_id: decision}} from a results directory."""
    out: dict[str, dict[int, str]] = {}
    for rf in sorted(results_dir.glob("frame_*.json")):
        data = json.loads(rf.read_text())
        out[data["frame_id"]] = {m["mask_id"]: m["triage"] for m in data["masks"]}
    return out


def _merge(dec_a: dict[int, str], dec_b: dict[int, str], rule: str) -> dict[int, str]:
    """Apply consensus rule across two per-frame decision dicts."""
    all_ids = set(dec_a) | set(dec_b)
    merged: dict[int, str] = {}
    for mid in all_ids:
        a = dec_a.get(mid, "accept")
        b = dec_b.get(mid, "accept")
        if rule == "intersection":
            # reject if either rejected
            merged[mid] = TRIAGE_REJECT if (a == TRIAGE_REJECT or b == TRIAGE_REJECT) else "accept"
        else:  # union
            # reject only if both rejected
            merged[mid] = TRIAGE_REJECT if (a == TRIAGE_REJECT and b == TRIAGE_REJECT) else "accept"
    return merged


def _write_annotation(frame_id: str, decisions: dict[int, str], out_dir: Path) -> None:
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann_path.exists():
        return
    ann = np.array(Image.open(ann_path))
    for p in extract_proposals(ann_path, frame_id):
        if decisions.get(p.mask_id) == TRIAGE_REJECT:
            ann[p.pixel_mask] = 0
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ann.astype(np.uint8)).save(out_dir / f"{frame_id}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-a", required=True, help="First run tag (e.g. qwen2.5vl_72b)")
    parser.add_argument("--tag-b", required=True, help="Second run tag (e.g. llama3.2_90b)")
    parser.add_argument("--rule", default="intersection", choices=["intersection", "union"],
                        help="Consensus rule (default: intersection)")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    dir_a = config.DATA_ROOT / "vlm" / args.tag_a / "results"
    dir_b = config.DATA_ROOT / "vlm" / args.tag_b / "results"
    out_dir = config.DATA_ROOT / "vlm" / f"merged_{args.tag_a}_{args.tag_b}" / f"annotation_{args.rule}"

    print(f"Run A  : {dir_a}")
    print(f"Run B  : {dir_b}")
    print(f"Rule   : {args.rule}")
    print(f"Output : {out_dir}")

    decs_a = _load_decisions(dir_a)
    decs_b = _load_decisions(dir_b)

    frame_ids = sorted(set(decs_a) & set(decs_b))
    only_a = set(decs_a) - set(decs_b)
    only_b = set(decs_b) - set(decs_a)
    if only_a:
        print(f"  WARNING: {len(only_a)} frames only in run A — skipped")
    if only_b:
        print(f"  WARNING: {len(only_b)} frames only in run B — skipped")
    print(f"Frames : {len(frame_ids)} in common")

    totals: dict[str, int] = {}
    for frame_id in tqdm(frame_ids, unit="frame"):
        merged = _merge(decs_a[frame_id], decs_b[frame_id], args.rule)
        _write_annotation(frame_id, merged, out_dir)
        for d in merged.values():
            totals[d] = totals.get(d, 0) + 1

    total = sum(totals.values())
    print(f"\nDone. {len(frame_ids)} frames written to {out_dir}")
    for k, v in sorted(totals.items()):
        print(f"  {k:15s}: {v:5d}  ({100*v/total:.1f}%)")


if __name__ == "__main__":
    main()
