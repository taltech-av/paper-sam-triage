#!/usr/bin/env python3
"""
Pixel accounting for discovery candidates, bucketed by geometry and human verdict.

`analyze_human_verification.py --only geometry` counts candidates. mIoU counts
pixels, and the two are not interchangeable: a rim on a bus and a rim on a distant
sign are one candidate each and differ by orders of magnitude in the loss. This
script therefore re-derives, for every answered candidate, the pixels it actually
contributed to the training label, and reports the same four-way taxonomy in those
units.

The painting is reproduced exactly as replay_triage._add_discoveries does it:
the 384x384 component map is upsampled NEAREST to full resolution and a candidate
claims only pixels the post-triage annotation left as background. Components are
disjoint, so per-candidate counts do not depend on paint order.

Two questions come out of it:

  1. What share of the pixels discovery added is pixels a human rejected? That is
     the quantity the downstream deficit responds to, and the count-based 52%
     `edge bleed` share does not estimate it.
  2. What share of object pixels sits in confirmed boundary growth? That bounds
     what any boundary-repair mechanism (SAM re-prompting, human redrawing) could
     recover before one is built.

Usage:
    python discovery_pixel_budget.py --export human_verified_output/verify_export.csv
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from analyze_human_verification import (BLEED, GROWTH, ALARM, FIND, CORRECT,
                                        fold_class, load_masks)

# Class ids that are objects rather than background or ignore.
OBJECT_IDS = (2, 3, 4, 5)


def geometry_bucket(row: dict, class_of: dict) -> str:
    """The four-way outcome, duplicated from the count-based report so the two agree."""
    try:
        neighbours = json.loads(row.get("touching_sam") or "[]")
    except (TypeError, ValueError):
        neighbours = []
    same = any(class_of.get((row["frame_id"], nid)) == fold_class(row["class"])
               for nid in neighbours)
    kept = row["human"] == CORRECT
    if same:
        return GROWTH if kept else BLEED
    return FIND if kept else ALARM


def frame_pixels(frame_id: str, candidates: list[dict], base_dir: Path,
                 masks_dir: Path) -> tuple[dict[int, int], int, int]:
    """
    Painted pixel count per candidate id, plus the frame's object and added totals.

    Returns ({mask_id: pixels}, object_pixels_in_base, total_added). A candidate
    whose component did not survive regeneration contributes zero rather than
    being dropped, so the denominator stays the set of candidates a human judged.
    """
    png = masks_dir / f"{frame_id}.png"
    base_path = base_dir / f"{frame_id}.png"
    if not (png.exists() and base_path.exists()):
        return {}, 0, 0

    base = np.array(Image.open(base_path))
    height, width = base.shape[:2]
    index_map = np.array(Image.open(png))
    index_full = np.array(Image.fromarray(index_map).resize((width, height), Image.NEAREST))

    background = base == 0
    object_pixels = int(np.isin(base, OBJECT_IDS).sum())

    # One pass over the frame: every candidate's paintable pixels at once, rather
    # than one full-resolution comparison per candidate.
    paintable = np.where(background, index_full, 0)
    counts = np.bincount(paintable.ravel())

    per_candidate = {}
    total = 0
    for row in candidates:
        index = row.get("candidate_index")
        if index is None:
            continue
        value = int(index) + 1
        pixels = int(counts[value]) if value < len(counts) else 0
        per_candidate[row["id"]] = pixels
        total += pixels
    return per_candidate, object_pixels, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--base", type=Path,
                        default=config.DATA_ROOT / "annotation_swin_only",
                        help="annotation the discovery variant was layered onto")
    parser.add_argument("--variant", type=Path,
                        default=config.DATA_ROOT / "annotation_swin_only_discovery_noVLM_ccm",
                        help="the painted discovery variant, for the reconciliation check")
    parser.add_argument("--masks", type=Path,
                        default=config.DATA_ROOT / "vlm" / "discovery_masks")
    parser.add_argument("--out", type=Path, help="write the report here")
    args = parser.parse_args()

    rows, _ = load_masks(args.export)
    class_of = {(row["frame_id"], row["id"]): fold_class(row.get("class")) for row in rows}
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("source") == "discovery" and row["human"]:
            by_frame[row["frame_id"]].append(row)

    pixels_by_bucket: Counter = Counter()
    counts_by_bucket: Counter = Counter()
    pixels_by_class: dict[str, Counter] = defaultdict(Counter)
    object_total = added_total = 0
    reconciled = reconciliation_gap = 0
    missing = 0

    for frame_id, candidates in tqdm(sorted(by_frame.items()), desc="Frames", unit="frame"):
        per_candidate, object_pixels, added = frame_pixels(
            frame_id, candidates, args.base, args.masks)
        if not per_candidate:
            missing += 1
            continue
        object_total += object_pixels
        added_total += added
        for row in candidates:
            bucket = geometry_bucket(row, class_of)
            pixels = per_candidate.get(row["id"], 0)
            pixels_by_bucket[bucket] += pixels
            counts_by_bucket[bucket] += 1
            pixels_by_class[fold_class(row["class"])][bucket] += pixels

        # The variant on disk is the ground truth for what was painted; if the
        # reproduction here drifts from it the accounting is wrong, so it is
        # checked rather than assumed.
        variant_path = args.variant / f"{frame_id}.png"
        base_path = args.base / f"{frame_id}.png"
        if variant_path.exists() and base_path.exists():
            base = np.array(Image.open(base_path))
            variant = np.array(Image.open(variant_path))
            reconciled += 1
            reconciliation_gap += abs(int((variant != base).sum()) - added)

    lines = []
    def emit(text: str = "") -> None:
        lines.append(text)
        print(text)

    emit(f"Frames: {len(by_frame):,} with answered discovery candidates"
         + (f" ({missing} skipped — no component map or base annotation)" if missing else ""))
    emit(f"Candidates: {sum(counts_by_bucket.values()):,}")
    if reconciled:
        emit(f"Reconciliation against {args.variant.name}: mean |reproduced - on-disk| = "
             f"{reconciliation_gap / reconciled:,.1f} px/frame over {reconciled:,} frames")
    emit()
    emit(f"Discovery added {added_total:,} pixels to a label set already holding "
         f"{object_total:,} object pixels ({100 * added_total / max(object_total, 1):.1f}%).")
    emit()

    emit(f"  {'outcome':<17}{'candidates':>12}{'% cand':>9}{'pixels':>14}{'% px':>8}{'px/cand':>10}")
    emit("  " + "-" * 70)
    for bucket in (BLEED, GROWTH, ALARM, FIND):
        n, px = counts_by_bucket[bucket], pixels_by_bucket[bucket]
        emit(f"  {bucket:<17}{n:>12,}{100 * n / max(sum(counts_by_bucket.values()), 1):>8.1f}%"
             f"{px:>14,}{100 * px / max(added_total, 1):>7.1f}%{px / max(n, 1):>10,.0f}")

    rejected_px = pixels_by_bucket[BLEED] + pixels_by_bucket[ALARM]
    emit()
    emit(f"  Pixels a human rejected: {rejected_px:,} of {added_total:,} added "
         f"({100 * rejected_px / max(added_total, 1):.1f}%)")
    emit(f"  Those are {100 * rejected_px / max(object_total, 1):.1f}% of the object pixels "
         f"in the label set they were added to.")
    emit()
    emit(f"  Boundary-repair ceiling: confirmed growth is {pixels_by_bucket[GROWTH]:,} px, "
         f"{100 * pixels_by_bucket[GROWTH] / max(object_total, 1):.2f}% of object pixels.")
    emit(f"  Genuine recoveries add {pixels_by_bucket[FIND]:,} px, "
         f"{100 * pixels_by_bucket[FIND] / max(object_total, 1):.2f}% of object pixels.")

    emit()
    emit("Per class (share of that class's added pixels):")
    for name in sorted(pixels_by_class):
        per = pixels_by_class[name]
        total = sum(per.values())
        parts = "  ".join(f"{bucket} {100 * per[bucket] / max(total, 1):5.1f}%"
                          for bucket in (BLEED, GROWTH, ALARM, FIND))
        emit(f"  {name:<10} {total:>12,} px   {parts}")

    if args.out:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nWritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
