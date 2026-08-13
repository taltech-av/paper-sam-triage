#!/usr/bin/env python3
"""
Write the human-verified annotation set — the clean arm of the training ladder.

Every other arm is a *rule* applied to agent outputs (`replay_triage.py`). This
one is not a rule: it applies the human verdicts themselves. A SAM proposal a
labeler called incorrect is erased from the annotation; everything else is kept
exactly as SAM drew it. The result is what the ladder's `human/human_verified`
config trains on, and the reference the dirty arms are measured against.

Everything the set consists of lands in one folder under the dataset root:

    vlm/human_verified/
        annotation/       class-id PNGs — the training input
        visualization/    camera + the kept masks, translucent, one per frame
        frames.csv        the frames written
        splits/           train / validation / test lists over exactly those frames
        report.txt        this run's counts

`splits/` is written by the same run, from the frames it just wrote, so the set
and its partition can never drift apart. Labelling is ongoing: every export is
larger than the last, so re-running this script re-cuts the splits over the new
frame count and every arm of the comparison picks the change up from one place.
The partition is *not* stable across runs — a frame in train today can be in
test at the next export — so a checkpoint is only comparable to others trained
against the same splits/ generation. Retrain the whole group after regenerating,
never one arm of it.

Frames are dealt out with `make_splits.pixel_balanced_splits`: weather is held
exactly and the per-class *pixel* mass is equalised across splits, because at
this size class content is what a random split gets wrong — cyclist is ~1% of
the pixels and lives in a few dozen frames, so presence-only stratification
leaves one split holding a third of the class. Pixel counts come from the clean
annotations written by this run, which is the reference every arm is scored on.

`annotation/` is what fusion-training reads. Its loader resolves an annotation by
string-replacing `camera` in the image path with the config's `annotation_path`
(`utils/helpers.get_annotation_path`), so the folder needs no index and no
naming of its own — the CLFT config points at it with

    "annotation_path": "vlm/human_verified/annotation"

and the PNGs must carry the *dataset* class ids, not training ids: 0 background,
2 vehicle, 3 sign, 4 cyclist, 5 pedestrian. `relabel_annotation` folds those into
the four trained classes at load time (4 and 5 merge into `human`), so writing
merged ids here would silently destroy the cyclist/pedestrian distinction the
verification job spent its labeler-hours on. The ids are checked before exit.

Only *complete* frames are written. A frame is complete when every region the
job asked about — SAM proposals and discovery candidates alike — has a resolved
verdict. A frame with one unanswered mask would silently keep whatever SAM put
there and enter training labelled "human-verified", which is the one failure
this set cannot afford, so those frames are skipped and counted instead.

Discovery is opt-in (`--discovery`), and off by default. A human confirmed that
a candidate *is* a real object of its class; they did not confirm its pixels.
Those pixels are a Swin connected component computed at 384x384 and upsampled to
camera space, so painting them in adds a real object with a coarse boundary —
worth having, but a different claim from cleaning, and a different training set.
Kinds are separable because they mean different things: `standalone` recovers an
object SAM missed, `fringe` only grows the rim of an object SAM already has.

Candidate pixels are reconstructed exactly as the labeler saw them
(`make_label_bundle.classify_candidates`): the candidate's own component minus
anything the full-resolution annotation already covers. Nothing a kept SAM mask
owns is ever overwritten.

A candidate proposed as `human` has no cyclist/pedestrian split — that is the
one thing the label job could not ask, since Swin proposes the class and only a
VLM confirmation resolves it. Those take the class id the runs resolved when
they confirmed; a candidate no run resolved is skipped and reported, never
guessed.

`visualization/` holds the annotation as it will be trained on and nothing else:
the camera frame with the kept masks filled in their class colour. No rejected
masks, no boxes, no before/after — anything drawn that is not in the label makes
the picture answer a different question than "is this label right".

Usage:
    python make_clean_annotations.py --export human_verified_output/job-<id>.csv
    python make_clean_annotations.py --export ... --discovery standalone
    python make_clean_annotations.py --export ... --no-vis        # PNGs only
    python make_clean_annotations.py --export ... --no-splits     # leave splits/ alone
"""

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from analyze_human_verification import CORRECT, Tee, load_masks, run_tags
from core.mask_extractor import extract_proposals
from make_splits import (WEATHER_CATEGORIES, analyze_frames, balance_table,
                         labelling_order, pixel_balanced_splits, write_report,
                         write_split)

# Swin proposes three classes; two map straight onto a pipeline class, the third
# is the split only a VLM confirmation resolves (see module docstring).
SWIN_TO_CLASS_ID = {1: 2, 2: 3}

# The dataset class ids fusion-training's `dataset_classes` declares. Anything
# else in a written PNG would be relabelled to background at load time.
DATASET_CLASS_IDS = frozenset({0, 1, 2, 3, 4, 5})

# BGR, for cv2 — the outcome colours frame a contact-sheet tile.
OUTCOME_COLORS = {
    "kept":    (60, 200, 60),
    "removed": (60, 60, 230),
    "added":   (0, 165, 255),
}
TILE = 240          # contact-sheet tile, px
TILE_COLUMNS = 6
CROP_PADDING = 24   # context around an instance's bbox, px


# --------------------------------------------------------------------------
# Building one clean frame
# --------------------------------------------------------------------------


def frame_rows(rows: list[dict]) -> dict[str, list[dict]]:
    """Export rows grouped by frame, in frame order."""
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_id"]].append(row)
    return dict(sorted(by_frame.items()))


def candidate_pixels(frame_id: str, shape: tuple[int, int], annotation: np.ndarray,
                     candidates: list[dict]) -> dict[int, np.ndarray]:
    """Each candidate's own pixels in camera space, keyed by candidate index.

    One nearest-neighbour resize of the whole index map: components are disjoint
    and carry `candidate_index + 1` as their value, so a single resize keeps
    every pixel's identity. Pixels the annotation already covers are dropped —
    the same subtraction the bundle made before showing the region to a human,
    which is what makes a `fringe` the rim it looked like rather than the whole
    object underneath it.
    """
    path = config.DATA_ROOT / "vlm" / "discovery_masks" / f"{frame_id}.png"
    if not path.exists():
        return {}
    height, width = shape
    index_map = cv2.resize(np.array(Image.open(path)), (width, height), interpolation=cv2.INTER_NEAREST)
    free = annotation == 0
    pixels = {}
    for row in candidates:
        index = row.get("candidate_index")
        if index is None:
            continue
        own = (index_map == index + 1) & free
        if own.any():
            pixels[index] = own
    return pixels


def candidate_class_id(row: dict, resolved: dict[tuple, int]) -> int | None:
    """Pipeline class id for a confirmed candidate, or None if nothing resolves it.

    Swin's vehicle and sign map straight across. Its `human` class does not: the
    labeler was shown "human" because that is all Swin claims, so cyclist vs
    pedestrian can only come from a run that confirmed the candidate and named
    it. When no run did, the object is real but unnamed, and a guess would put
    invented pixels of an invented class into a set whose whole value is that a
    human stands behind every pixel.
    """
    swin_class = row.get("swin_class")
    if swin_class in SWIN_TO_CLASS_ID:
        return SWIN_TO_CLASS_ID[swin_class]
    return resolved.get(tuple(json.loads(row["bbox_384"]))) if row.get("bbox_384") else None


def resolved_classes(frame_id: str, tags: list[str]) -> dict[tuple, int]:
    """bbox_384 -> class id, for candidates a run confirmed and named.

    Read from the runs' own results rather than the export: the bundle carries
    each run's confirmation flag but not the class that confirmation resolved.
    Runs that disagree resolve nothing — two models naming the same region
    differently is not a class, and the candidate is skipped.
    """
    named: dict[tuple, set[int]] = defaultdict(set)
    for tag in tags:
        path = config.DATA_ROOT / "vlm" / tag / "results" / f"{frame_id}.json"
        if not path.exists():
            continue
        for entry in json.loads(path.read_text()).get("discovered", []):
            if entry.get("confirmed"):
                named[tuple(entry["bbox_384"])].add(entry["class_id"])
    return {bbox: ids.pop() for bbox, ids in named.items() if len(ids) == 1}


def clean_frame(frame_id: str, rows: list[dict], kinds: set[str], tags: list[str]):
    """The cleaned annotation for one frame, plus every instance it decided on.

    Returns (annotation, instances, note). `note` is set when the frame cannot be
    written — a missing annotation, or proposals that no longer match the ones
    the job asked about, which means the SAM set was regenerated since the
    bundle was built and the ids in the export point at different pixels.
    """
    path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not path.exists():
        return None, [], "no annotation_sam PNG"
    annotation = np.array(Image.open(path))
    original = annotation.copy()

    proposals = {proposal.mask_id: proposal for proposal in extract_proposals(path, frame_id)}
    sam_rows = [row for row in rows if row.get("source", "sam") == "sam"]
    missing = [row["id"] for row in sam_rows if row["id"] not in proposals]
    if missing or len(proposals) != len(sam_rows):
        return None, [], f"{len(proposals)} proposals vs {len(sam_rows)} judged masks"

    instances = []
    for row in sam_rows:
        proposal = proposals[row["id"]]
        keep = row["human"] == CORRECT
        if not keep:
            annotation[proposal.pixel_mask] = 0
        instances.append({
            "frame_id": frame_id, "class": proposal.class_name, "class_id": proposal.class_id,
            "outcome": "kept" if keep else "removed", "bbox": proposal.bbox,
            "pixels": proposal.pixel_mask, "count": proposal.pixel_count,
        })

    if kinds:
        confirmed = [row for row in rows
                     if row.get("source") == "discovery" and row["human"] == CORRECT
                     and row.get("kind") in kinds]
        # Pixels come from the *original* annotation: a candidate covers what SAM
        # left free, not what the human's removals just freed up. Painting into a
        # region a rejected mask used to own would be adding pixels no labeler saw.
        pixels = candidate_pixels(frame_id, annotation.shape[:2], original, confirmed)
        resolved = resolved_classes(frame_id, tags) if any(
            row.get("swin_class") not in SWIN_TO_CLASS_ID for row in confirmed) else {}
        for row in confirmed:
            own = pixels.get(row.get("candidate_index"))
            class_id = candidate_class_id(row, resolved)
            if own is None or class_id is None:
                instances.append({
                    "frame_id": frame_id, "class": row.get("class", "other"), "class_id": class_id,
                    "outcome": "unresolved" if own is not None else "no pixels",
                    "bbox": None, "pixels": None, "count": 0,
                })
                continue
            annotation[own] = class_id
            ys, xs = np.nonzero(own)
            instances.append({
                "frame_id": frame_id, "class": config.CLASS_ID_TO_NAME[class_id], "class_id": class_id,
                "outcome": "added", "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "pixels": own, "count": int(own.sum()), "kind": row.get("kind"),
            })

    return annotation, instances, None


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------


def write_overlay(camera: np.ndarray, annotation: np.ndarray, path: Path) -> None:
    """The clean annotation, translucent over the camera frame.

    What is drawn is exactly what `annotation/` holds — the masks a human kept,
    plus whatever discovery added — so the picture and the training label cannot
    drift apart. Each region is filled at OVERLAY_ALPHA and outlined at full
    strength, because a 60-px sign at half alpha is invisible and the whole point
    is to be able to judge the small ones.
    """
    canvas = camera.copy()
    present = []
    for class_id, colour in config.CLASS_COLORS_BGR.items():
        mask = annotation == class_id
        if not mask.any():
            continue
        canvas[mask] = (canvas[mask] * (1 - config.OVERLAY_ALPHA)
                        + np.array(colour) * config.OVERLAY_ALPHA).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, 1)
        present.append((config.CLASS_ID_TO_NAME[class_id], colour, len(contours)))

    legend(canvas, present)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])


def legend(canvas: np.ndarray, present: list[tuple[str, tuple, int]]) -> None:
    """Class name, colour and region count along the top of an overlay."""
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (0, 0, 0), -1)
    x = 10
    for class_name, colour, count in present:
        text = f"{class_name} ({count})"
        cv2.rectangle(canvas, (x, 7), (x + 11, 18), colour, -1)
        cv2.putText(canvas, text, (x + 16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        x += 32 + 9 * len(text)


class Sheets:
    """Per-class, per-outcome tile reservoirs and the contact sheets they become.

    Off by default (`--class-sheets`): the per-frame overlays are the set as it
    will be trained on, while a sheet is a way to audit one class at a time —
    every rejected sign side by side is how you see whether the rejections were
    about the object or about its boundary.

    Tiles are rendered as they are met and sampled with Algorithm R, so a sheet
    is an unbiased sample of the whole set rather than the first N frames — the
    labelling order is chronological and the early frames are the ones where the
    labeler was still calibrating.
    """

    def __init__(self, per_class: int, seed: int):
        self.per_class = per_class
        self.rng = random.Random(seed)
        self.tiles: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        self.seen: Counter = Counter()

    def offer(self, camera: np.ndarray, instance: dict) -> None:
        key = (instance["class"], instance["outcome"])
        self.seen[key] += 1
        bucket = self.tiles[key]
        if len(bucket) < self.per_class:
            bucket.append(self.tile(camera, instance))
            return
        index = self.rng.randrange(self.seen[key])
        if index < self.per_class:
            bucket[index] = self.tile(camera, instance)

    def tile(self, camera: np.ndarray, instance: dict) -> np.ndarray:
        """One instance: its crop, its mask outlined, framed in its outcome colour."""
        height, width = camera.shape[:2]
        x1, y1, x2, y2 = instance["bbox"]
        x1, y1 = max(0, x1 - CROP_PADDING), max(0, y1 - CROP_PADDING)
        x2, y2 = min(width, x2 + CROP_PADDING), min(height, y2 + CROP_PADDING)
        crop = camera[y1:y2, x1:x2].copy()
        mask = instance["pixels"][y1:y2, x1:x2].astype(np.uint8)
        # White, not the class colour: the tile's border already carries the
        # outcome, and vehicle-red against removed-red would be unreadable. A
        # sheet is one class and one outcome throughout, so neither needs a hue.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, contours, -1, (255, 255, 255), 2)

        # Letterboxed, so a 20x300 sign and a 400x400 vehicle stay comparable.
        scale = min((TILE - 8) / crop.shape[1], (TILE - 8) / crop.shape[0])
        resized = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), max(1, int(crop.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        canvas = np.zeros((TILE, TILE, 3), dtype=np.uint8)
        top, left = (TILE - resized.shape[0]) // 2, (TILE - resized.shape[1]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.rectangle(canvas, (0, 0), (TILE - 1, TILE - 1), OUTCOME_COLORS[instance["outcome"]], 3)
        cv2.putText(canvas, f"{instance['frame_id'].removeprefix('frame_')} {instance['count']}px",
                    (6, TILE - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def write(self, out_dir: Path) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for (class_name, outcome), tiles in sorted(self.tiles.items()):
            if not tiles:
                continue
            rows = (len(tiles) + TILE_COLUMNS - 1) // TILE_COLUMNS
            sheet = np.zeros((rows * TILE + 34, TILE_COLUMNS * TILE, 3), dtype=np.uint8)
            # ASCII only: cv2's Hershey fonts have no glyph for anything else and
            # draw '???' where a dash would be.
            caption = (f"{class_name} - {outcome}   showing {len(tiles)} of "
                       f"{self.seen[(class_name, outcome)]:,}")
            cv2.putText(sheet, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        OUTCOME_COLORS[outcome], 1, cv2.LINE_AA)
            for index, tile in enumerate(tiles):
                top = 34 + (index // TILE_COLUMNS) * TILE
                left = (index % TILE_COLUMNS) * TILE
                sheet[top:top + TILE, left:left + TILE] = tile
            path = out_dir / f"{class_name}_{outcome}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
            written.append(path)
        return written


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report_classes(totals: dict[tuple[str, str], int], pixels: dict[tuple[str, str], int]) -> None:
    print("\nPer class — what the human verdicts did to the SAM set:")
    print(f"  {'class':<12} {'kept':>8} {'removed':>8} {'removed %':>10} {'added':>8}"
          f"   {'kept px':>12} {'removed px':>12} {'added px':>12}")
    # Only the outcomes that reached the annotation. A candidate whose class no
    # run resolved has no class column to sit in — it is counted below instead.
    classes = sorted({class_name for class_name, outcome in totals
                      if outcome in ("kept", "removed", "added")})
    for class_name in classes:
        kept, removed, added = (totals.get((class_name, outcome), 0)
                                for outcome in ("kept", "removed", "added"))
        share = f"{100 * removed / (kept + removed):.1f}%" if kept + removed else "--"
        print(f"  {class_name:<12} {kept:>8,} {removed:>8,} {share:>10} {added:>8,}   "
              + "".join(f"{pixels.get((class_name, outcome), 0):>12,} "
                        for outcome in ("kept", "removed", "added")))
    kept = sum(count for (_, outcome), count in totals.items() if outcome == "kept")
    removed = sum(count for (_, outcome), count in totals.items() if outcome == "removed")
    added = sum(count for (_, outcome), count in totals.items() if outcome == "added")
    print(f"  {'TOTAL':<12} {kept:>8,} {removed:>8,} "
          f"{100 * removed / (kept + removed):>9.1f}% {added:>8,}")


def write_splits(written: list[str], annotation_dir: Path, splits_dir: Path,
                 val_ratio: float, test_ratio: float, seed: int,
                 export: Path | None = None) -> None:
    """Cut the frames just written into train / validation / test.

    The per-weather `test_*.txt` files matter as much as `test.txt`: both
    `test_swin.py` and `dump_frame_metrics.py` enumerate the five weather names
    and evaluate each in turn, so a condition without a file is silently dropped
    from the score. At this set's size some of them are only a handful of frames
    — small enough that the weather-averaged mIoU those scripts print is noisy,
    which is why the comparison is meant to be read off the pooled per-frame
    dump (`dump_frame_metrics.py` → `bootstrap_miou.py`), where every test frame
    carries equal weight and the arms are paired on identical frames.

    Labelling order is balanced alongside the class pixel mass. The human
    reference is non-stationary — precision falls about 2 points per 100 frames
    judged and has not plateaued — so a split drawn mostly from early frames is
    graded against a measurably more permissive standard than one drawn from
    late frames. It happened to come out even last time; balancing it means that
    stops being luck as the export grows.
    """
    splits_dir.mkdir(parents=True, exist_ok=True)
    # make_splits works in bare ids ('011134'); this script carries the PNG stem
    # ('frame_011134'), and the two meet again in write_split's 'camera/frame_*'.
    frame_ids = [frame_id.removeprefix("frame_") for frame_id in written]
    # Pixel counts come from the clean annotations, not annotation_sam: the
    # split is balanced on the labels it will actually be trained and scored on.
    records = analyze_frames(frame_ids, annotation_dir)
    if export is not None:
        ranks = labelling_order(export)
        if all(rec["frame_id"] in ranks for rec in records):
            for rec in records:
                rec["era_rank"] = ranks[rec["frame_id"]]
        else:
            print("  NOTE: some frames carry no labelling order — era balancing skipped")
    train, val, test = pixel_balanced_splits(records, val_ratio, test_ratio, seed)

    print(f"\nSplits → {splits_dir}   "
          f"(weather held exactly, per-class pixel mass equalised, seed {seed})")
    write_split(train, splits_dir / "train.txt")
    write_split(val,   splits_dir / "validation.txt")
    write_split(test,  splits_dir / "test.txt")
    for weather in WEATHER_CATEGORIES:
        subset = [r for r in test if r["weather"] == weather]
        if subset:
            write_split(subset, splits_dir / f"test_{weather}.txt")
    write_split(records, splits_dir / "all.txt")
    (splits_dir / "frame_analysis.json").write_text(json.dumps(records, indent=2))
    write_report(records, train, val, test, splits_dir / "report.txt")
    print(balance_table(records, [("train", train), ("val", val), ("test", test)]))

    thin = [w for w in WEATHER_CATEGORIES
            if 0 < sum(1 for r in test if r["weather"] == w) < 8]
    if thin:
        print(f"\n  NOTE: {', '.join(thin)} have fewer than 8 test frames — read the "
              "comparison off the pooled bootstrap, not the weather-averaged mIoU.")


def report_training(out_root: Path, annotation_dir: Path, ids: Counter, frames: int) -> None:
    """What a CLFT config has to say to consume this, and the check that it can.

    The loader reaches an annotation by replacing `camera` in the image path, so
    the only thing it needs is the folder's path relative to the dataset root —
    printed here rather than left to be worked out from the docstring.
    """
    try:
        relative = annotation_dir.relative_to(config.DATA_ROOT)
    except ValueError:
        relative = annotation_dir
    print("\nFor a fusion-training config (config/vlm/clftv2-base/cleaning/...):")
    print(f'  "dataset_root":    "{config.DATA_ROOT}"')
    print(f'  "annotation_path": "{relative}"    (only the human arm; the automated arms')
    print( '                                      point at their own annotation dir)')
    print(f'  "train_split":     "{out_root / "splits" / "train.txt"}"')
    print(f'  "val_split":       "{out_root / "splits" / "validation.txt"}"')
    named = ", ".join(f"{class_id}={config.CLASS_ID_TO_NAME.get(class_id, '?')}: {count:,} px"
                      for class_id, count in sorted(ids.items()))
    print(f"\n  {frames:,} PNGs, dataset class ids present — {named}")
    unknown = set(ids) - DATASET_CLASS_IDS
    if unknown:
        print(f"  WARNING: ids {sorted(unknown)} are not in the config's dataset_classes and "
              "would be relabelled to background at load time")
    else:
        print("  ids check out: 4 and 5 stay separate here and merge into `human` "
              "only at load time, via train_classes.dataset_mapping")


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="visin job export CSV")
    parser.add_argument("--discovery", choices=["none", "standalone", "fringe", "all"], default="none",
                        help="also paint human-confirmed discovery candidates of this kind (default: none)")
    parser.add_argument("--out-root", type=Path,
                        help="one folder for the whole set "
                             "(default: <DATA_ROOT>/vlm/human_verified[_<kind>])")
    parser.add_argument("--no-vis", action="store_true", help="write the annotation PNGs only")
    parser.add_argument("--no-splits", action="store_true",
                        help="do not re-cut <out-root>/splits/ (leaves an existing partition, "
                             "and the checkpoints trained against it, valid)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="validation fraction (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.20,
                        help="test fraction (default: 0.20) — the held-out frames the "
                             "human-vs-automated mIoU contrast is read off")
    parser.add_argument("--class-sheets", action="store_true",
                        help="also write per-class contact sheets of kept/removed/added instances")
    parser.add_argument("--per-class", type=int, default=48, help="tiles per contact sheet")
    parser.add_argument("--tags", nargs="+", help="runs to resolve discovery classes from")
    parser.add_argument("--limit", type=int, help="stop after this many frames (for a quick check)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hpc", action="store_true")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    kinds = {"standalone", "fringe"} if args.discovery == "all" else (
        set() if args.discovery == "none" else {args.discovery})
    suffix = "" if args.discovery == "none" else f"_{args.discovery}"
    out_root = args.out_root or config.DATA_ROOT / "vlm" / f"human_verified{suffix}"
    out_root.mkdir(parents=True, exist_ok=True)

    with (out_root / "report.txt").open("w") as handle, redirect_stdout(Tee(sys.stdout, handle)):
        run(args, kinds, out_root)
        print(f"\nReport written to {out_root / 'report.txt'}")
    return 0


def run(args, kinds: set[str], out_root: Path) -> None:
    # Unresolved masks are loaded on purpose: a frame is only clean if nothing in
    # it is still unanswered, and that is a fact about the masks left out of the
    # scoring set.
    rows, stats = load_masks(args.export, include_unresolved=True)
    tags = args.tags or run_tags([row for row in rows if row["human"]])
    by_frame = frame_rows(rows)

    complete = {frame_id: frame for frame_id, frame in by_frame.items()
                if all(row["human"] is not None for row in frame)}
    if args.limit:
        complete = dict(list(complete.items())[:args.limit])

    annotation_dir = out_root / "annotation"
    vis_dir = out_root / "visualization"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    print(f"Export: {args.export}")
    print(f"Runs:   {', '.join(tags)}")
    print(f"Frames: {len(complete):,} complete of {len(by_frame):,} in the job "
          f"({len(by_frame) - len(complete):,} still hold an unanswered or tied mask)")
    print(f"Discovery: {'none — SAM masks only' if not kinds else ', '.join(sorted(kinds))}")
    print(f"Output:    {out_root}")

    sheets = Sheets(args.per_class, args.seed) if args.class_sheets else None
    totals: Counter = Counter()
    pixels: Counter = Counter()
    ids: Counter = Counter()
    skipped: Counter = Counter()
    written = []
    for frame_id, frame in tqdm(sorted(complete.items()), desc="frames", unit="frame"):
        annotation, instances, note = clean_frame(frame_id, frame, kinds, tags)
        if annotation is None:
            skipped[note] += 1
            continue
        Image.fromarray(annotation.astype(np.uint8)).save(annotation_dir / f"{frame_id}.png")
        written.append(frame_id)
        ids.update(dict(zip(*np.unique(annotation, return_counts=True))))

        for instance in instances:
            totals[(instance["class"], instance["outcome"])] += 1
            pixels[(instance["class"], instance["outcome"])] += instance["count"]

        if args.no_vis and sheets is None:
            continue
        camera_path = config.CAMERA_DIR / f"{frame_id}.png"
        if not camera_path.exists():
            skipped["no camera image (visualization skipped)"] += 1
            continue
        camera = cv2.imread(str(camera_path))
        if not args.no_vis:
            write_overlay(camera, annotation, vis_dir / f"{frame_id}.jpg")
        if sheets is not None:
            for instance in instances:
                if instance["pixels"] is not None:
                    sheets.offer(camera, instance)

    print(f"\nWrote {len(written):,} annotation PNGs → {annotation_dir}")
    if not args.no_vis:
        print(f"Wrote {len(written):,} overlays        → {vis_dir}   "
              "(camera + the kept masks, nothing else)")
    for note, count in skipped.items():
        print(f"  skipped {count:,} frames: {note}")

    report_classes(totals, pixels)
    unresolved = sum(count for (_, outcome), count in totals.items()
                     if outcome in ("unresolved", "no pixels"))
    if unresolved:
        print(f"\n  {unresolved:,} confirmed candidates were not painted — "
              "no run resolved their class, or their pixels are covered by a SAM mask")

    frames_csv = out_root / "frames.csv"
    frames_csv.write_text("".join(f"camera/{frame_id}.png\n" for frame_id in written))
    print(f"\nFrame list → {frames_csv}  ({len(written):,} frames)")

    if not args.no_splits:
        write_splits(written, annotation_dir, out_root / "splits",
                     args.val_ratio, args.test_ratio, args.seed, export=args.export)

    report_training(out_root, annotation_dir, ids, len(written))

    if sheets is not None:
        sheet_paths = sheets.write(out_root / "classes")
        print(f"\nContact sheets → {out_root / 'classes'}  ({len(sheet_paths)} sheets, "
              f"up to {args.per_class} instances each)")
        for path in sheet_paths:
            class_name, outcome = path.stem.rsplit("_", 1)
            print(f"    {path.name:<28} {sheets.seen[(class_name, outcome)]:,} instances")


if __name__ == "__main__":
    sys.exit(main())
