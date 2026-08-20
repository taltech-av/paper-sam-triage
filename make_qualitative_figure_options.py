#!/usr/bin/env python3
"""
Generate qualitative-figure options for a manuscript draft.

Every family below illustrates one claim the results already make in numbers,
using the human verdicts in human_verified_output/verify_export.csv as the
arbiter, so a caption can state what the reader is looking at without any new
inference:

  ladder     one frame through the annotation ladder of the variant ablation
             (raw SAM -> Swin threshold -> + VLM triage -> + discovery)
  signal     the free signal against the human: deleted/kept x right/wrong,
             the 2x2 behind the +3.9 mIoU row
  vlmcost    the same 2x2 for a VLM backend, including the masks it deletes on
             top of the Swin filter that the human called correct (the -3.0)
  flip       masks the two backends decide differently, arbitrated by the human
  queue      review targeting: the head of the alpha-sorted queue against the
             same number of masks drawn at random
  gate       the geometry gate on one frame: candidates proposed, then survivors
  inflation  the object-pixel budget: filtered labels, + all discovery,
             + gated discovery
  purity     the cleaning contrast at fixed size: raw SAM against the same
             frame with the human-rejected masks erased

Each family emits several *options* (different frames or samples), each option
as separate panels plus composites in two layouts, plus a browsable contact
sheet and a LaTeX snippet. Nothing is written into the manuscript until
--promote copies a pick into --figures-dir.

Palette is the one make_candidate_geometry_figure.py already uses -- grey for
what the pipeline already had, orange for material a human called not the object
(or a stage removes), teal for what a human kept -- so the two sets read as one
argument. Class-coloured panels (ladder) use the pipeline's own class colours
from config.

Frames already spent on another figure set are excluded by default: two
manuscripts drawing on one corpus must not share crops.

Usage:
    # everything, 4 options per family, into qualitative_candidates/options/
    python make_qualitative_figure_options.py

    # one family, more options to choose between
    python make_qualitative_figure_options.py --family ladder --per-family 8

    # what would be picked, without rendering
    python make_qualitative_figure_options.py --list

    # take a pick into the paper (panels + composite + provenance)
    python make_qualitative_figure_options.py --promote ladder_02 --as qual_ladder
"""

import argparse
import csv
import json
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import config
from core.mask_extractor import extract_proposals

REPO = Path(__file__).parent
EXPORT = REPO / "human_verified_output" / "verify_export.csv"

# ── Palette (BGR), shared with candidate_geometry.tex and the geom_* figures ──
GREY = (120, 120, 120)      # what the pipeline already had / what a stage keeps
ORANGE = (77, 168, 244)     # removed by a stage, or human said "not the object"
TEAL = (150, 160, 79)       # kept: survives the stage and the human agrees
WHITE = (255, 255, 255)
MASK_ALPHA = 0.5
# Full-frame panels print at a third of a column width, where a 0.5 tint on a
# dull frame disappears. The highlighted set gets a stronger tint and a heavier
# outline than a crop would need.
FULL_ALPHA = 0.6
FULL_OUTLINE = 2

# ── Geometry ─────────────────────────────────────────────────────────────────
# Full-frame panels: the source is 1363x768, and a 4-panel row at \textwidth
# gets under two inches each, so 1024 px wide is already generous.
FULL_W = 1024
# Crops reuse the size every existing qual_*/geom_* panel uses.
CROP_W, CROP_H = 480, 360
CROP_AR = CROP_W / CROP_H
CROP_PAD_FRAC = 0.55
QUEUE_W, QUEUE_H = 240, 180   # filmstrip thumbnails
MIN_QUEUE_PX = 400            # below this a mask is a smudge at thumbnail size

CORRECT, INCORRECT = "correct", "incorrect"
CLASS_FOLD = {"cyclist": "human", "pedestrian": "human"}

# ── Frames already spent elsewhere ───────────────────────────────────────────
# The 12 crops of the companion manuscript's qualitative set, recovered from the
# provenance file it ships beside its figures. Reusing one of these would put the
# same photograph in two simultaneous submissions, which is a self-plagiarism
# risk; reusing a crop within one manuscript is merely dull.
USED_COMPANION = [
    "frame_003514", "frame_045049", "frame_007408", "frame_029915",
    "frame_011472", "frame_011052", "frame_005713", "frame_065911",
    "frame_040869", "frame_035091", "frame_009025", "frame_011178",
]
# The four candidates behind the discovery-geometry figure that
# make_candidate_geometry_figure.py renders. That figure was dropped from the
# draft on 2026-08-20, so these are free again -- they stay listed so restoring
# it (the images are in git, and the script regenerates them) cannot collide with
# a figure picked in the meantime. Drop them from the list to reclaim them.
USED_GEOMETRY_SET = ["frame_007388", "frame_012457", "frame_023299", "frame_001072"]

# ── Ladder variants, mapped to the rows of tables/ablation.tex ───────────────
def ladder_dirs(tag: str) -> list[tuple[str, Path, str]]:
    """(slug, directory, caption fragment) per rung, in table order."""
    d = config.DATA_ROOT
    return [
        ("raw_sam", d / "annotation_raw_sam", "Raw SAM, no triage"),
        ("swin_only", d / "annotation_swin_only", "Swin agreement threshold"),
        ("triage", d / "vlm" / tag / "annotation_triage", "$+$ VLM triage"),
        ("full", d / "vlm" / tag / "annotation_full", "$+$ VLM triage $+$ discovery"),
    ]


DISCOVERY_UNGATED = "annotation_swin_only_discovery_noVLM_ccm"
DISCOVERY_GATED = "annotation_swin_only_discovery_noVLM_standalone_ccm"
HUMAN_CLEANED = "vlm/human_verified/annotation"


# ── Small image helpers ──────────────────────────────────────────────────────

def camera(frame_id: str) -> Optional[np.ndarray]:
    img = cv2.imread(str(config.CAMERA_DIR / f"{frame_id}.png"))
    return img


def label_png(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.array(Image.open(path))


def tint(img: np.ndarray, mask: np.ndarray, bgr, alpha: float = MASK_ALPHA) -> None:
    if not mask.any():
        return
    img[mask] = (np.array(bgr) * alpha + img[mask] * (1 - alpha)).astype(np.uint8)


def outline(img: np.ndarray, mask: np.ndarray, bgr, thickness: int = 2) -> None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, bgr, thickness, cv2.LINE_AA)


def paint_labels(img: np.ndarray, ann: np.ndarray, colours=None) -> np.ndarray:
    """Class-coloured overlay, exactly as the pipeline's own visualisations."""
    out = img.copy()
    colours = colours or config.CLASS_COLORS_BGR
    for class_id, bgr in colours.items():
        m = ann == class_id
        tint(out, m, bgr, config.OVERLAY_ALPHA)
        outline(out, m, bgr, 1)
    return out


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    interp = cv2.INTER_AREA if w > width else cv2.INTER_CUBIC
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=interp)


def crop_window(bbox, W: int, H: int, pad_frac: float = CROP_PAD_FRAC):
    """A 4:3 window around bbox, clamped into the frame without distorting."""
    x1, y1, x2, y2 = bbox
    pad = pad_frac * max(x2 - x1, y2 - y1)
    w = max((x2 - x1) + 2 * pad, 140.0)
    h = max((y2 - y1) + 2 * pad, 140.0 / CROP_AR)
    if w / h < CROP_AR:
        w = h * CROP_AR
    else:
        h = w / CROP_AR
    if w > W:
        w, h = W, W / CROP_AR
    if h > H:
        h, w = H, H * CROP_AR
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1 = int(round(min(max(cx - w / 2, 0), W - w)))
    ny1 = int(round(min(max(cy - h / 2, 0), H - h)))
    return nx1, ny1, nx1 + int(round(w)), ny1 + int(round(h))


def to_crop(img: np.ndarray, window, size=(CROP_W, CROP_H)) -> np.ndarray:
    x1, y1, x2, y2 = window
    piece = img[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if piece.shape[1] > size[0] else cv2.INTER_CUBIC
    return cv2.resize(piece, size, interpolation=interp)


def legible(crop_bgr: np.ndarray, min_brightness: float, min_contrast: float) -> bool:
    """Reject crops that print as a dark or flat rectangle."""
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return g.mean() >= min_brightness and g.std() >= min_contrast


def border(panel: np.ndarray, bgr, thickness: int = 6) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, out.shape[0] - 1), bgr, thickness * 2)
    return out


def caption_bar(panel: np.ndarray, text: str, height: int = 26) -> np.ndarray:
    """Burned-in label. Contact sheets only -- paper panels stay clean."""
    bar = np.full((height, panel.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text[:110], (6, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (0, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([panel, bar])


def compose(panels: list[np.ndarray], cols: int, gap: int = 10,
            bg: int = 255) -> np.ndarray:
    """Grid composite at one common panel width."""
    width = min(p.shape[1] for p in panels)
    scaled = [fit_width(p, width) for p in panels]
    height = max(p.shape[0] for p in scaled)
    scaled = [p if p.shape[0] == height else
              cv2.copyMakeBorder(p, 0, height - p.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(bg, bg, bg))
              for p in scaled]
    rows_n = (len(scaled) + cols - 1) // cols
    sheet = np.full((rows_n * height + (rows_n - 1) * gap,
                     cols * width + (cols - 1) * gap, 3), bg, np.uint8)
    for i, p in enumerate(scaled):
        r, c = divmod(i, cols)
        y, x = r * (height + gap), c * (width + gap)
        sheet[y:y + height, x:x + width] = p
    return sheet


# ── The verification export ──────────────────────────────────────────────────

def fold(name: str) -> str:
    return CLASS_FOLD.get(name, name)


def frame_id_of(row) -> str:
    return Path(row["frame"]).stem


def load_rows() -> list[dict]:
    with EXPORT.open() as fh:
        return list(csv.DictReader(fh))


def as_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_bbox(value) -> Optional[tuple]:
    try:
        x1, y1, x2, y2 = json.loads(value)
        return int(x1), int(y1), int(x2), int(y2)
    except (TypeError, ValueError):
        return None


def kept(triage: str) -> bool:
    """The paper's destructive/non-destructive split: refine and human_review
    both carry every pixel forward, only reject deletes."""
    return triage != "reject"


def sam_mask(frame_id: str, mask_id: int) -> Optional[np.ndarray]:
    ann = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann.exists():
        return None
    match = next((p for p in extract_proposals(ann, frame_id) if p.mask_id == mask_id), None)
    return None if match is None else match.pixel_mask


def candidate_pixels(frame_id: str, index: int, shape) -> Optional[np.ndarray]:
    png = config.DATA_ROOT / "vlm" / "discovery_masks" / f"{frame_id}.png"
    if not png.exists():
        return None
    h, w = shape
    pixel_map = cv2.resize(np.array(Image.open(png)), (w, h),
                           interpolation=cv2.INTER_NEAREST)
    return pixel_map == index + 1


# ── Options ──────────────────────────────────────────────────────────────────

@dataclass
class Panel:
    slug: str                 # file suffix, e.g. "a_raw_sam"
    image: np.ndarray
    caption: str              # LaTeX subcaption
    note: str = ""            # provenance line, not printed in the paper


@dataclass
class Option:
    family: str
    name: str                 # e.g. ladder_02
    panels: list[Panel]
    layouts: list[str]        # e.g. ["1x4", "2x2"]
    caption: str
    meta: dict = field(default_factory=dict)
    score: float = 0.0
    frames: list = field(default_factory=list)   # every frame the option spends


def layouts_for(n: int, wide_ok: bool = True) -> list[str]:
    if n == 2:
        return ["1x2", "2x1"] if wide_ok else ["2x1", "1x2"]
    if n == 3:
        return ["1x3", "3x1"]
    if n == 4:
        return ["2x2", "1x4"]
    return [f"1x{n}"]


def cols_of(layout: str) -> int:
    return int(layout.split("x")[1])


def env_of(layout: str, full_frame: bool) -> str:
    """figure* for anything that needs the page width to stay readable."""
    cols = cols_of(layout)
    if cols >= 3 or (cols == 2 and full_frame):
        return "figure*"
    return "figure"


# ── Family: ladder ───────────────────────────────────────────────────────────

def family_ladder(rows, args, excluded) -> list[Option]:
    """One frame down the rungs of tables/ablation.tex, class-coloured."""
    rungs = ladder_dirs(args.vlm)
    missing = [d for _, d, _ in rungs if not d.is_dir()]
    if missing:
        print(f"[ladder] skipped: missing {missing[0]}")
        return []

    pool = sorted(p.stem for p in (config.DATA_ROOT / "annotation_swin_only").glob("frame_*.png"))
    pool = [f for f in pool if f not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    scored = []
    for frame_id in pool[:args.scan_frames]:
        anns = [label_png(d / f"{frame_id}.png") for _, d, _ in rungs]
        if any(a is None for a in anns):
            continue
        obj = [(a > 1) for a in anns]
        counts = [int(m.sum()) for m in obj]
        if counts[0] < 12_000 or counts[0] > 220_000:
            continue
        # A rung that changes nothing makes a figure of four identical panels.
        removed = int((obj[0] & ~obj[1]).sum())          # the free signal deletes
        triaged = int((obj[1] & ~obj[2]).sum())          # triage deletes on top
        added = int((obj[3] & ~obj[2]).sum())            # discovery adds
        if removed < 1_500 or triaged < 800 or added < 6_000:
            continue
        img = camera(frame_id)
        if img is None or not legible(img, args.min_brightness, args.min_contrast):
            continue
        classes = len(set(np.unique(anns[0])) - {0, 1})
        # Prefer frames where every rung is visible and more than one class is.
        s = (min(removed / 12_000, 1) + min(triaged / 8_000, 1)
             + min(added / 40_000, 1) + 0.4 * classes)
        scored.append((s, frame_id, dict(removed=removed, triaged=triaged, added=added)))

    scored.sort(reverse=True, key=lambda t: t[0])
    options = []
    for n, (s, frame_id, stats) in enumerate(scored[:args.per_family]):
        img = camera(frame_id)
        panels = []
        for letter, (slug, d, label) in zip("abcd", rungs):
            ann = label_png(d / f"{frame_id}.png")
            panel = fit_width(paint_labels(img, ann), FULL_W)
            panels.append(Panel(f"{letter}_{slug}", panel, label,
                                note=f"{d.name}: {int((ann > 1).sum())} object px"))
        options.append(Option(
            family="ladder", name=f"ladder_{n:02d}", panels=panels,
            layouts=layouts_for(4), score=s,
            caption=("One frame down the annotation ladder of "
                     "Table~\\ref{tab:ablation}. Colours are the pipeline's class "
                     "colours. The free signal removes "
                     f"{stats['removed']:,} object pixels, triage a further "
                     f"{stats['triaged']:,}, and discovery adds {stats['added']:,} "
                     "-- the stage the downstream numbers penalise most."),
            frames=[frame_id],
            meta=dict(frame_id=frame_id, backend=args.vlm, **stats)))
    return options


# ── Families: 2x2 confusion matrices over crops ──────────────────────────────

SIGNAL_CELLS = {
    "true_delete": ("Deleted, human agrees", ORANGE),
    "false_delete": ("Deleted, human disagrees", ORANGE),
    "true_keep": ("Kept, human agrees", TEAL),
    "false_keep": ("Kept, human disagrees", TEAL),
}


def crop_candidate(frame_id: str, bbox, mask_getter, colour, args,
                   shape_cache: dict) -> Optional[np.ndarray]:
    img = camera(frame_id)
    if img is None:
        return None
    H, W = img.shape[:2]
    x1, y1, x2, y2 = bbox
    if x1 < args.border or y1 < args.border or x2 > W - args.border or y2 > H - args.border:
        return None
    mask = mask_getter()
    if mask is None or not mask.any():
        return None
    window = crop_window(bbox, W, H)
    if not legible(to_crop(img, window), args.min_brightness, args.min_contrast):
        return None
    canvas = img.copy()
    tint(canvas, mask, colour)
    outline(canvas, mask, colour, 2)
    return to_crop(canvas, window)


def crop_score(bbox, pixel_count: int, W: int, H: int, class_name: str) -> float:
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return -1.0
    area_frac = (bw * bh) / float(W * H)
    if not 0.002 <= area_frac <= 0.28:
        return -1.0
    ar = bw / bh
    if not 0.18 <= ar <= 5.5:
        return -1.0
    size = float(np.exp(-((np.log(area_frac) - np.log(0.03)) ** 2) / (2 * 1.1 ** 2)))
    fill = min(pixel_count / float(bw * bh), 1.0) if pixel_count else 0.5
    rare = 1.5 if fold(class_name) == "human" else (1.15 if class_name == "sign" else 1.0)
    return size * (0.5 + 0.5 * fill) * rare


def take(bucket: list, used_frames: set, used_classes: set):
    """Best candidate that renders, preferring a class this option lacks.

    Without the class preference a 2x2 comes out as four signs: signs are the
    most numerous class in the corpus and the most often wrong, so they sweep
    every ranked list. A figure of four signs illustrates signs, not the rule.
    """
    for require_new_class in (True, False):
        for i, cand in enumerate(bucket):
            if cand is None or cand["frame_id"] in used_frames:
                continue
            if require_new_class and cand["class"] in used_classes:
                continue
            panel = cand["render"]()
            if panel is None:
                bucket[i] = None            # unrenderable: drop on the next sweep
                continue
            bucket.pop(i)
            return cand, panel
        bucket[:] = [c for c in bucket if c is not None]
    return None, None


def _matrix_family(name, buckets, args, caption, meta_extra) -> list[Option]:
    """Assemble 2x2 options from four ranked cell lists.

    A cell can run dry -- for one backend the masks it deletes that the free
    signal had kept and the human called wrong number in the dozens, and most
    are unusable crops. An option with three of the four cells is still a
    figure, so the cell is dropped and named in the manifest rather than losing
    the option; the caption should then not describe it as a 2x2.
    """
    options = []
    used_frames = set()
    for n in range(args.per_family):
        panels, cells_meta, empty = [], {}, []
        used_classes = set()
        for cell, (label, colour) in SIGNAL_CELLS.items():
            cand, panel = take(buckets[cell], used_frames, used_classes)
            if cand is None:
                empty.append(cell)
                continue
            used_frames.add(cand["frame_id"])
            used_classes.add(cand["class"])
            letter = "abcd"[len(panels)]
            panels.append(Panel(f"{letter}_{cell}", panel,
                                f"{label} ({cand['sub']})", note=cand["note"]))
            cells_meta[cell] = {k: cand[k] for k in ("frame_id", "mask_id", "class")}
        if len(panels) < 3:
            break
        options.append(Option(family=name, name=f"{name}_{n:02d}", panels=panels,
                              layouts=layouts_for(len(panels), wide_ok=False),
                              caption=caption,
                              frames=[c["frame_id"] for c in cells_meta.values()],
                              meta={**meta_extra, "cells": cells_meta,
                                    **({"cells_unavailable": empty} if empty else {})}))
    return options


def family_signal(rows, args, excluded) -> list[Option]:
    """The dense agreement score against the human verdict, as photographs."""
    W = H = None
    buckets = {c: [] for c in SIGNAL_CELLS}
    for r in rows:
        if r["mask_source"] != "sam" or r["verdict"] not in (CORRECT, INCORRECT):
            continue
        quality = r["mask_quality_agent"]
        if quality not in ("good", "bad"):
            continue
        frame_id = frame_id_of(r)
        if frame_id in excluded:
            continue
        bbox = as_bbox(r["mask_bbox"])
        if bbox is None:
            continue
        if W is None:
            probe = camera(frame_id)
            if probe is None:
                continue
            H, W = probe.shape[:2]
        alpha = as_float(r["mask_swin_agreement"])
        pixels = int(as_float(r["mask_pixel_count"]) or 0)
        s = crop_score(bbox, pixels, W, H, r["class"])
        if s <= 0:
            continue
        deletes = quality == "bad"
        agrees = (r["verdict"] == INCORRECT) == deletes
        cell = f"{'true' if agrees else 'false'}_{'delete' if deletes else 'keep'}"
        colour = SIGNAL_CELLS[cell][1]
        mask_id = int(r["maskId"])
        # Sharpest examples first: an extreme score with the human against it is
        # the one worth printing.
        extreme = abs((alpha if alpha is not None else 0.5) - 0.3)
        buckets[cell].append(dict(
            frame_id=frame_id, mask_id=mask_id, **{"class": r["class"]},
            sub=(f"$\\alpha={alpha:.2f}$" if alpha is not None else "no score"),
            note=(f"{r['class']} mask {mask_id} in {frame_id}: swin agreement "
                  f"{alpha}, quality {quality}, human {r['verdict']}"),
            score=s * (1.0 + extreme),
            render=(lambda f=frame_id, b=bbox, m=mask_id, c=colour:
                    crop_candidate(f, b, lambda: sam_mask(f, m), c, args, {}))))

    for lst in buckets.values():
        lst.sort(key=lambda d: -d["score"])
    return _matrix_family(
        "signal", buckets, args,
        caption=("The free signal against the human reference. Orange masks the "
                 "Swin agreement threshold deletes, teal ones it keeps; the "
                 "columns split on whether the human agreed. This is the "
                 "filtering step worth $+3.9$ mIoU in Table~\\ref{tab:ablation}."),
        meta_extra={"signal": "swin_agreement threshold"})


def family_vlmcost(rows, args, excluded) -> list[Option]:
    """The same 2x2 for one VLM backend.

    Deletions are ranked so the *incremental* ones come first: masks the Swin
    threshold had already kept, which the backend then removed. Those are what
    the ladder's -3.0 mIoU is paid for -- everything else the backend deletes
    was going to be deleted anyway. There are few of them (the counts land in
    the option's meta), so --triage-only-incremental, which drops the rest
    entirely, will often leave a cell empty.
    """
    col = f"mask_triage_{args.vlm}"
    if col not in (rows[0] if rows else {}):
        print(f"[vlmcost] skipped: no {col} in the export")
        return []
    W = H = None
    buckets = {c: [] for c in SIGNAL_CELLS}
    incremental_tally = {CORRECT: 0, INCORRECT: 0}
    for r in rows:
        if r["mask_source"] != "sam" or r["verdict"] not in (CORRECT, INCORRECT):
            continue
        triage = r[col]
        if triage == "":
            continue
        frame_id = frame_id_of(r)
        if frame_id in excluded:
            continue
        deletes = not kept(triage)
        incremental = deletes and r["mask_quality_agent"] == "good"
        if incremental:
            incremental_tally[r["verdict"]] += 1
        if deletes and args.triage_only_incremental and not incremental:
            continue                      # already gone before the VLM saw it
        bbox = as_bbox(r["mask_bbox"])
        if bbox is None:
            continue
        if W is None:
            probe = camera(frame_id)
            if probe is None:
                continue
            H, W = probe.shape[:2]
        pixels = int(as_float(r["mask_pixel_count"]) or 0)
        s = crop_score(bbox, pixels, W, H, r["class"])
        if s <= 0:
            continue
        agrees = (r["verdict"] == INCORRECT) == deletes
        cell = f"{'true' if agrees else 'false'}_{'delete' if deletes else 'keep'}"
        colour = SIGNAL_CELLS[cell][1]
        mask_id = int(r["maskId"])
        alpha = as_float(r["mask_swin_agreement"])
        # Name the component that actually made the call. Triage is three things
        # in sequence -- an aspect-ratio pre-filter, the dense agreement score,
        # then the VLM -- and a mask can be deleted before any model runs. A
        # panel captioned "the backend deletes it" when no model was called is
        # simply false, and the export carries enough to tell the cases apart:
        # an empty bbox verdict with an empty quality means nothing ran.
        agent = r[f"mask_bbox_agent_{args.vlm}"]
        quality = r["mask_quality_agent"]
        model_decided = bool(agent)
        if not agent and not quality:
            decider = "aspect-ratio pre-filter, no model call"
        elif not agent:
            decider = f"dense signal only (quality {quality}), no VLM call"
        else:
            decider = f"VLM bbox {agent.replace('_', ' ')}, quality {quality or 'n/a'}"
        if deletes and args.triage_model_decided and not model_decided:
            continue
        sub = f"\\textsc{{{triage.replace('_', ' ')}}}: {decider}"
        buckets[cell].append(dict(
            frame_id=frame_id, mask_id=mask_id, **{"class": r["class"]},
            sub=sub + (", deleted by the VLM alone" if incremental else ""),
            note=(f"{r['class']} mask {mask_id} in {frame_id}: {args.vlm} "
                  f"triage={triage} by {decider} (alpha {alpha}), "
                  f"human {r['verdict']}"
                  + (" [incremental deletion]" if incremental else "")),
            score=s * (2.0 if incremental else 1.0),
            render=(lambda f=frame_id, b=bbox, m=mask_id, c=colour:
                    crop_candidate(f, b, lambda: sam_mask(f, m), c, args, {}))))

    for lst in buckets.values():
        lst.sort(key=lambda d: -d["score"])
    backend = args.vlm.replace("_", "\\_")
    total_incremental = sum(incremental_tally.values())
    share = (100 * incremental_tally[CORRECT] / total_incremental) if total_incremental else 0
    print(f"[vlmcost] {args.vlm} deletes {total_incremental} masks the free signal "
          f"kept; the human called {share:.0f}\\% of them correct")
    return _matrix_family(
        "vlmcost", buckets, args,
        caption=("The triage stage against the human reference "
                 f"({backend} backend). Orange masks triage deletes, teal ones "
                 "it keeps; the right-hand column is where triage and human part "
                 "company, and each subcaption names the component that decided "
                 "-- the aspect-ratio pre-filter, the dense agreement score, or "
                 f"the VLM. Of the {total_incremental} masks triage deletes on a "
                 "verdict from the model, after the free signal had kept them, "
                 f"the human called {share:.0f}\\,\\% correct; that is what the "
                 "$-3.0$ mIoU in Table~\\ref{tab:ablation} is spent on."),
        meta_extra={"backend": args.vlm,
                    "incremental_deletions": dict(incremental_tally),
                    "only_incremental": args.triage_only_incremental})


# ── Family: flip ─────────────────────────────────────────────────────────────

def family_flip(rows, args, excluded) -> list[Option]:
    """Masks the two backends decide differently, with the human as arbiter."""
    a_col, b_col = "mask_triage_llava_34b", "mask_triage_qwen2.5vl_72b_v2"
    W = H = None
    buckets = {"llava_right": [], "qwen_right": []}
    for r in rows:
        if r["mask_source"] != "sam" or r["verdict"] not in (CORRECT, INCORRECT):
            continue
        ta, tb = r[a_col], r[b_col]
        if not ta or not tb or kept(ta) == kept(tb):
            continue
        frame_id = frame_id_of(r)
        if frame_id in excluded:
            continue
        bbox = as_bbox(r["mask_bbox"])
        if bbox is None:
            continue
        if W is None:
            probe = camera(frame_id)
            if probe is None:
                continue
            H, W = probe.shape[:2]
        pixels = int(as_float(r["mask_pixel_count"]) or 0)
        s = crop_score(bbox, pixels, W, H, r["class"])
        if s <= 0:
            continue
        human_says_keep = r["verdict"] == CORRECT
        llava_right = kept(ta) == human_says_keep
        cell = "llava_right" if llava_right else "qwen_right"
        # Colour by what the human said, since that is the arbiter here.
        colour = TEAL if human_says_keep else ORANGE
        mask_id = int(r["maskId"])
        buckets[cell].append(dict(
            frame_id=frame_id, mask_id=mask_id, **{"class": r["class"]},
            sub=(f"LLaVA {'keeps' if kept(ta) else 'deletes'}, "
                 f"Qwen {'keeps' if kept(tb) else 'deletes'}; human: "
                 f"{'object' if human_says_keep else 'not the object'}"),
            note=(f"{r['class']} mask {mask_id} in {frame_id}: llava={ta}, "
                  f"qwen={tb}, shared swin agreement {r['mask_swin_agreement']}, "
                  f"human {r['verdict']}"),
            score=s,
            render=(lambda f=frame_id, b=bbox, m=mask_id, c=colour:
                    crop_candidate(f, b, lambda: sam_mask(f, m), c, args, {}))))

    for lst in buckets.values():
        lst.sort(key=lambda d: -d["score"])

    options, used = [], set()
    for n in range(args.per_family):
        panels, cells = [], {}
        used_classes = set()
        order = ["llava_right", "qwen_right", "llava_right", "qwen_right"]
        for letter, cell in zip("abcd", order):
            cand, panel = take(buckets[cell], used, used_classes)
            if cand is None:
                break
            used.add(cand["frame_id"])
            used_classes.add(cand["class"])
            panels.append(Panel(f"{letter}_{cell}", panel, cand["sub"], note=cand["note"]))
            cells[f"{letter}_{cell}"] = {k: cand[k] for k in ("frame_id", "mask_id", "class")}
        if len(panels) < 4:
            break
        options.append(Option(
            family="flip", name=f"flip_{n:02d}", panels=panels,
            layouts=layouts_for(4, wide_ok=False),
            frames=[c["frame_id"] for c in cells.values()],
            caption=("Masks the two backends decide differently, with the human "
                     "verdict as arbiter: teal where the human called the mask an "
                     "object, orange where they did not. Both backends see "
                     "identical crops and identical dense scores, so the "
                     "divergence is attributable to the model. These are "
                     "illustrations of the disagreement rate, not evidence that "
                     "either backend ranks above the other."),
            meta={"panels": cells}))
    return options


# ── Family: queue ────────────────────────────────────────────────────────────

def family_queue(rows, args, excluded) -> list[Option]:
    """Review targeting: the head of the alpha-sorted queue against random order."""
    pool = []
    for r in rows:
        if r["mask_source"] != "sam" or r["verdict"] not in (CORRECT, INCORRECT):
            continue
        frame_id = frame_id_of(r)
        if frame_id in excluded:
            continue
        alpha = as_float(r["mask_swin_agreement"])
        bbox = as_bbox(r["mask_bbox"])
        if alpha is None or bbox is None:
            continue
        pixels = int(as_float(r["mask_pixel_count"]) or 0)
        if pixels < MIN_QUEUE_PX:
            continue                      # unreadable at thumbnail size
        pool.append(dict(frame_id=frame_id, mask_id=int(r["maskId"]), bbox=bbox,
                         alpha=alpha, verdict=r["verdict"], cls=r["class"],
                         pixels=pixels))
    if len(pool) < 4 * args.queue_n:
        print("[queue] skipped: not enough verified masks")
        return []
    # The thumbnail-size floor is a restriction on both rows alike; the caption
    # quotes the restricted pool's own error rate so the random row can be read
    # against the right base rate rather than the corpus-wide one.
    pool_error = sum(1 for d in pool if d["verdict"] == INCORRECT) / float(len(pool))

    def thumb(item):
        colour = ORANGE if item["verdict"] == INCORRECT else TEAL
        panel = crop_candidate(item["frame_id"], item["bbox"],
                               lambda: sam_mask(item["frame_id"], item["mask_id"]),
                               colour, args, {})
        if panel is None:
            return None
        panel = cv2.resize(panel, (QUEUE_W, QUEUE_H), interpolation=cv2.INTER_AREA)
        return border(panel, colour)

    by_alpha = sorted(pool, key=lambda d: (d["alpha"], -d["pixels"]))
    options = []
    for n in range(args.per_family):
        rng = random.Random(args.seed + n)
        head, seen = [], set()
        for item in by_alpha:
            if len(head) >= args.queue_n:
                break
            if item["frame_id"] in seen:
                continue
            panel = thumb(item)
            if panel is None:
                continue
            seen.add(item["frame_id"])
            head.append((item, panel))
        # The random arm must stay a fair draw: sample first, render second, and
        # never re-draw because a thumbnail came out unflattering.
        shuffled = pool[:]
        rng.shuffle(shuffled)
        rand, seen_r = [], set(seen)
        for item in shuffled:
            if len(rand) >= args.queue_n:
                break
            if item["frame_id"] in seen_r:
                continue
            panel = thumb(item)
            if panel is None:
                continue
            seen_r.add(item["frame_id"])
            rand.append((item, panel))
        if len(head) < args.queue_n or len(rand) < args.queue_n:
            break

        wrong_head = sum(1 for i, _ in head if i["verdict"] == INCORRECT)
        wrong_rand = sum(1 for i, _ in rand if i["verdict"] == INCORRECT)
        strip_head = compose([p for _, p in head], cols=args.queue_n, gap=6)
        strip_rand = compose([p for _, p in rand], cols=args.queue_n, gap=6)
        panels = [
            Panel("a_alpha_sorted", strip_head,
                  f"First {args.queue_n} by dense agreement, lowest first "
                  f"({wrong_head} of {args.queue_n} wrong)",
                  note=", ".join(f"{i['frame_id']}:{i['mask_id']} a={i['alpha']:.3f} "
                                 f"{i['verdict']}" for i, _ in head)),
            Panel("b_random", strip_rand,
                  f"{args.queue_n} in random order "
                  f"({wrong_rand} of {args.queue_n} wrong)",
                  note=", ".join(f"{i['frame_id']}:{i['mask_id']} a={i['alpha']:.3f} "
                                 f"{i['verdict']}" for i, _ in rand)),
        ]
        options.append(Option(
            family="queue", name=f"queue_{n:02d}", panels=panels, layouts=["2x1"],
            caption=("Review targeting (Table~\\ref{tab:review_targeting}). Top: the "
                     "head of the queue when masks are sorted by dense agreement, "
                     "lowest first. Bottom: the same number drawn at random. "
                     "Orange borders mark masks the human called wrong. Both rows "
                     "are drawn from the same pool -- every verified mask above "
                     f"{MIN_QUEUE_PX} px, of which {100 * pool_error:.0f}\\,\\% are "
                     "wrong -- so the difference between the rows is the sorting. "
                     "Sorting concentrates the errors a reviewer has time to see."),
            frames=[i["frame_id"] for i, _ in head + rand],
            meta=dict(seed=args.seed + n, n=args.queue_n,
                      wrong_sorted=wrong_head, wrong_random=wrong_rand,
                      pool=len(pool), pool_error_rate=round(pool_error, 3))))
    return options


# ── Family: gate ─────────────────────────────────────────────────────────────

def geometry_of(row, class_of) -> str:
    """same / other / none, from the bundle's stored ring join."""
    try:
        neighbours = json.loads(row.get("mask_touching_sam") or "[]")
    except (TypeError, ValueError):
        neighbours = []
    if not neighbours:
        return "none"
    own = fold(row["class"])
    classes = [class_of.get((frame_id_of(row), n)) for n in neighbours]
    return "same" if any(c == own for c in classes if c) else "other"


def family_gate(rows, args, excluded) -> list[Option]:
    """One frame's discovery candidates, before and after the geometry gate."""
    class_of = {(frame_id_of(r), int(r["maskId"])): fold(r["class"])
                for r in rows if r["class"] and r["mask_source"] == "sam"}
    by_frame = {}
    for r in rows:
        if r["mask_source"] != "discovery" or r["mask_candidate_index"] in (None, ""):
            continue
        frame_id = frame_id_of(r)
        if frame_id in excluded:
            continue
        by_frame.setdefault(frame_id, []).append(r)

    scored = []
    for frame_id, frame_rows in by_frame.items():
        dropped = [r for r in frame_rows if geometry_of(r, class_of) == "same"]
        survives = [r for r in frame_rows if geometry_of(r, class_of) != "same"]
        # The gate discards 78.5\% of candidates corpus-wide, so a frame where
        # most of them survive misrepresents it. Require the frame to be on the
        # right side of that, and keep at least one survivor so the panel shows
        # a gate rather than a deletion.
        if len(dropped) < 3 or not survives or len(dropped) < 2 * len(survives):
            continue
        share = len(dropped) / float(len(dropped) + len(survives))
        scored.append((share * min(len(dropped), 12), frame_id, dropped, survives))
    scored.sort(reverse=True, key=lambda t: t[0])

    options, n = [], 0
    for s, frame_id, dropped, survives in scored:
        if n >= args.per_family:
            break
        img = camera(frame_id)
        if img is None or not legible(img, args.min_brightness, args.min_contrast):
            continue
        shape = img.shape[:2]
        sam_ann = label_png(config.ANNOTATION_SAM_DIR / f"{frame_id}.png")
        if sam_ann is None:
            continue

        def pixels_of(rs):
            acc = np.zeros(shape, bool)
            for r in rs:
                px = candidate_pixels(frame_id, int(float(r["mask_candidate_index"])), shape)
                if px is not None:
                    acc |= px
            return acc

        drop_px, keep_px = pixels_of(dropped), pixels_of(survives)
        if drop_px.sum() < 2_000 or keep_px.sum() < 400:
            continue

        def base_panel():
            out = img.copy()
            tint(out, sam_ann > 1, GREY, FULL_ALPHA)
            outline(out, sam_ann > 1, WHITE, FULL_OUTLINE)
            return out

        before = base_panel()
        tint(before, drop_px | keep_px, ORANGE, FULL_ALPHA)
        outline(before, drop_px | keep_px, ORANGE, FULL_OUTLINE)

        after = base_panel()
        tint(after, keep_px, TEAL, FULL_ALPHA)
        outline(after, keep_px, TEAL, FULL_OUTLINE)

        total = len(dropped) + len(survives)
        panels = [
            Panel("a_proposed", fit_width(before, FULL_W),
                  f"All {total} candidates proposed",
                  note=f"{int((drop_px | keep_px).sum())} candidate px"),
            Panel("b_gated", fit_width(after, FULL_W),
                  f"After the geometry gate: {len(survives)} of {total} left",
                  note=f"{int(keep_px.sum())} candidate px survive"),
        ]
        options.append(Option(
            family="gate", name=f"gate_{n:02d}", panels=panels,
            layouts=layouts_for(2), score=s,
            caption=("The geometry gate on one frame. Grey with a white outline is "
                     "SAM coverage the pipeline already had. Left: every discovery "
                     "candidate, orange. Right: what survives discarding any "
                     "candidate abutting a same-class SAM mask, teal. The gate "
                     "costs no model call and is worth $+5.6$ mIoU "
                     "(Section~\\ref{exp:geometry})."),
            frames=[frame_id],
            meta=dict(frame_id=frame_id, dropped=len(dropped), kept=len(survives))))
        n += 1
    return options


# ── Family: inflation ────────────────────────────────────────────────────────

RIM_BAND_PX = 12


def rim_fraction(base_px: np.ndarray, added: np.ndarray) -> float:
    """Share of added pixels lying within RIM_BAND_PX of an existing label."""
    if not added.any():
        return 0.0
    k = np.ones((2 * RIM_BAND_PX + 1,) * 2, np.uint8)
    near = cv2.dilate(base_px.astype(np.uint8), k).astype(bool)
    return float((added & near).sum()) / float(added.sum())


def family_inflation(rows, args, excluded) -> list[Option]:
    """What discovery adds to the object-pixel budget, gated and ungated."""
    base = config.DATA_ROOT / "annotation_swin_only"
    ung = config.DATA_ROOT / DISCOVERY_UNGATED
    gated = config.DATA_ROOT / DISCOVERY_GATED
    if not (ung.is_dir() and gated.is_dir()):
        print("[inflation] skipped: discovery variant folders missing")
        return []

    pool = [p.stem for p in sorted(base.glob("frame_*.png")) if p.stem not in excluded]
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    scored = []
    for frame_id in pool[:args.scan_frames]:
        a, b, c = (label_png(d / f"{frame_id}.png") for d in (base, ung, gated))
        if a is None or b is None or c is None:
            continue
        base_px, add_all, add_gated = (a > 1), (b > 1) & ~(a > 1), (c > 1) & ~(a > 1)
        n_base, n_all, n_gated = map(lambda m: int(m.sum()), (base_px, add_all, add_gated))
        if n_base < 10_000 or n_all < 12_000:
            continue
        if n_gated < 300 or n_gated > 0.5 * n_all:
            continue                      # want the gate to visibly bite
        # Four candidates in five are rims on objects the pipeline already had,
        # so a frame whose additions are all free-standing blobs would show the
        # reader the 6.1\% case and call it the stage. Measure how much of what
        # is added hugs an existing label, and prefer the frames that match.
        rim_share = rim_fraction(base_px, add_all)
        if rim_share < 0.45:
            continue
        img = camera(frame_id)
        if img is None or not legible(img, args.min_brightness, args.min_contrast):
            continue
        scored.append((rim_share * min(n_all / float(n_base), 1.5), frame_id, (a, b, c),
                       dict(base=n_base, added=n_all, added_gated=n_gated,
                            rim_share=round(rim_share, 3))))
    scored.sort(reverse=True, key=lambda t: t[0])

    options = []
    for n, (ratio, frame_id, (a, b, c), stats) in enumerate(scored[:args.per_family]):
        img = camera(frame_id)
        base_px = a > 1
        p_base = img.copy()
        tint(p_base, base_px, GREY, FULL_ALPHA)
        outline(p_base, base_px, WHITE, FULL_OUTLINE)

        p_all = p_base.copy()
        add_all = (b > 1) & ~base_px
        tint(p_all, add_all, ORANGE, FULL_ALPHA)
        outline(p_all, add_all, ORANGE, FULL_OUTLINE)

        p_gated = p_base.copy()
        add_gated = (c > 1) & ~base_px
        tint(p_gated, add_gated, TEAL, FULL_ALPHA)
        outline(p_gated, add_gated, TEAL, FULL_OUTLINE)

        panels = [
            Panel("a_filtered", fit_width(p_base, FULL_W),
                  f"Filtered labels ({stats['base']:,} object px)"),
            Panel("b_all_discovery", fit_width(p_all, FULL_W),
                  f"$+$ all discovered regions ($+{100 * stats['added'] / stats['base']:.0f}$\\,\\%)"),
            Panel("c_gated", fit_width(p_gated, FULL_W),
                  f"$+$ geometry-gated discovery ($+{100 * stats['added_gated'] / stats['base']:.0f}$\\,\\%)"),
        ]
        options.append(Option(
            family="inflation", name=f"inflation_{n:02d}", panels=panels,
            layouts=layouts_for(3), score=ratio,
            caption=("What discovery adds to the object-pixel budget the training "
                     "loss sees. Grey is the filtered label set; orange is every "
                     "discovered region added on top of it, teal only those the "
                     "geometry gate keeps. Across the judged set the ungated stage "
                     "inflates the budget by $37.5$\\,\\% and $61.7$\\,\\% of what "
                     "it adds is material a human marked as not the object."),
            frames=[frame_id], meta=dict(frame_id=frame_id, **stats)))
    return options


# ── Family: purity ───────────────────────────────────────────────────────────

def family_purity(rows, args, excluded) -> list[Option]:
    """The cleaning contrast at fixed training-set size, as pixels."""
    cleaned_dir = config.DATA_ROOT / HUMAN_CLEANED
    if not cleaned_dir.is_dir():
        print("[purity] skipped: human-cleaned annotations missing")
        return []
    raw_dir = config.DATA_ROOT / "annotation_raw_sam"

    rejected = {}
    for r in rows:
        if r["mask_source"] != "sam" or r["verdict"] != INCORRECT:
            continue
        rejected.setdefault(frame_id_of(r), []).append(int(r["maskId"]))

    pool = [p.stem for p in sorted(cleaned_dir.glob("frame_*.png"))
            if p.stem not in excluded and p.stem in rejected]
    rng = random.Random(args.seed)
    rng.shuffle(pool)

    scored = []
    for frame_id in pool[:args.scan_frames]:
        raw, clean = (label_png(d / f"{frame_id}.png") for d in (raw_dir, cleaned_dir))
        if raw is None or clean is None:
            continue
        raw_px = raw > 1
        clean_px = clean > 1
        removed = raw_px & ~clean_px
        n_raw, n_removed, n_clean = (int(raw_px.sum()), int(removed.sum()),
                                     int(clean_px.sum()))
        if n_raw < 10_000 or n_removed < 3_000 or n_clean < 8_000:
            continue
        share = n_removed / float(n_raw)
        # Both arms have to be visible: a frame where cleaning erases nearly
        # everything shows a deletion, not a contrast between two label sets.
        if not 0.08 <= share <= 0.6:
            continue
        img = camera(frame_id)
        if img is None or not legible(img, args.min_brightness, args.min_contrast):
            continue
        # Peak around a third erased, and prefer frames where more than one mask
        # went, since the experiment is about a class of deletions.
        balance = float(np.exp(-((share - 0.33) ** 2) / (2 * 0.18 ** 2)))
        scored.append((balance * min(len(rejected[frame_id]) / 3.0, 1.5),
                       frame_id, (raw, clean),
                       dict(raw=n_raw, removed=n_removed, clean=n_clean,
                            removed_share=round(share, 3),
                            masks_rejected=len(rejected[frame_id]))))
    scored.sort(reverse=True, key=lambda t: t[0])

    options = []
    for n, (share, frame_id, (raw, clean), stats) in enumerate(scored[:args.per_family]):
        img = camera(frame_id)
        raw_px, clean_px = raw > 1, clean > 1
        removed = raw_px & ~clean_px

        p_raw = img.copy()
        tint(p_raw, clean_px, GREY, FULL_ALPHA)
        outline(p_raw, clean_px, WHITE, FULL_OUTLINE)
        tint(p_raw, removed, ORANGE, FULL_ALPHA)
        outline(p_raw, removed, ORANGE, FULL_OUTLINE)

        p_clean = img.copy()
        tint(p_clean, clean_px, TEAL, FULL_ALPHA)
        outline(p_clean, clean_px, WHITE, FULL_OUTLINE)

        panels = [
            Panel("a_uncleaned", fit_width(p_raw, FULL_W),
                  f"Raw SAM: orange is what the human rejected "
                  f"({stats['masks_rejected']} masks, "
                  f"{100 * stats['removed'] / stats['raw']:.0f}\\,\\% of object px)"),
            Panel("b_cleaned", fit_width(p_clean, FULL_W),
                  "Human-cleaned: the same labels with those masks erased"),
        ]
        options.append(Option(
            family="purity", name=f"purity_{n:02d}", panels=panels,
            layouts=layouts_for(2), score=share,
            caption=("Label purity at fixed training-set size "
                     "(Table~\\ref{tab:cleaning}). The two arms differ only in "
                     "whether the masks a human called incorrect are erased; "
                     "cleaning only ever deletes, and deleting $5.3$\\,\\% of the "
                     "object pixels is worth $+2.5$ mIoU."),
            frames=[frame_id], meta=dict(frame_id=frame_id, **stats)))
    return options


# Order matters: each family's frames are excluded from the ones after it, so
# whatever must be an unbiased draw goes first. The queue figure *is* the head
# of the real queue -- if the hand-picked crop families run first they take the
# worst masks with them and the queue understates itself.
FAMILIES = {
    "queue": family_queue,
    "signal": family_signal,
    "vlmcost": family_vlmcost,
    "flip": family_flip,
    "ladder": family_ladder,
    "gate": family_gate,
    "inflation": family_inflation,
    "purity": family_purity,
}

FULL_FRAME_FAMILIES = {"ladder", "gate", "inflation", "purity", "queue"}


# ── Writing ──────────────────────────────────────────────────────────────────

def snippet(opt: Option, layout: str, prefix: str = "figures/") -> str:
    """A LaTeX block that uses the individual panels, not the composite: the
    composites exist to be looked at, the panels to be typeset.

    `prefix` is where the panels sit relative to the document: figures/ for the
    paper (which is where --promote puts them), the candidate tree for the
    preview build.
    """
    cols = cols_of(layout)
    env = env_of(layout, opt.family in FULL_FRAME_FAMILIES)
    width = "\\textwidth" if env == "figure*" else "\\columnwidth"
    frac = {1: 0.98, 2: 0.49, 3: 0.325, 4: 0.245}.get(cols, 0.245)
    lines = [f"% {opt.name} ({layout})",
             f"\\begin{{{env}}}[t]", "    \\centering"]
    for i, p in enumerate(opt.panels):
        lines += [f"    \\begin{{subfigure}}[b]{{{frac:.3f}{width}}}",
                  f"        \\includegraphics[width=\\linewidth]"
                  f"{{{prefix}{opt.name}_{p.slug}.png}}",
                  f"        \\caption{{{p.caption}}}",
                  "    \\end{subfigure}" + ("\\hfill" if (i + 1) % cols else "")]
        if (i + 1) % cols == 0 and i + 1 < len(opt.panels):
            lines.append("    \\vspace{3pt}")
    lines += [f"    \\caption{{{opt.caption}}}",
              f"    \\label{{fig:{opt.name}}}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def write_option(opt: Option, out_dir: Path) -> dict:
    fam_dir = out_dir / opt.family
    panels_dir, comp_dir = fam_dir / "panels", fam_dir / "composites"
    panels_dir.mkdir(parents=True, exist_ok=True)
    comp_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for p in opt.panels:
        rel = f"{opt.family}/panels/{opt.name}_{p.slug}.png"
        cv2.imwrite(str(out_dir / rel), p.image, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        files[p.slug] = rel
    composites = {}
    for layout in opt.layouts:
        sheet = compose([p.image for p in opt.panels], cols=cols_of(layout))
        rel = f"{opt.family}/composites/{opt.name}_{layout}.png"
        cv2.imwrite(str(out_dir / rel), fit_width(sheet, min(sheet.shape[1], 1800)),
                    [cv2.IMWRITE_PNG_COMPRESSION, 6])
        composites[layout] = rel
    return {
        "name": opt.name, "family": opt.family, "caption": opt.caption,
        "frames": opt.frames,
        "score": round(float(opt.score), 3), "meta": opt.meta,
        "panels": [{"slug": p.slug, "caption": p.caption, "note": p.note,
                    "file": files[p.slug]} for p in opt.panels],
        "composites": composites,
        "latex": {lay: snippet(opt, lay) for lay in opt.layouts},
        "latex_preview": {lay: snippet(opt, lay, f"{opt.family}/panels/")
                          for lay in opt.layouts},
    }


def sheet_for(family: str, options: list[Option], out_dir: Path) -> None:
    tiles = []
    for opt in options:
        thumb = compose([p.image for p in opt.panels], cols=cols_of(opt.layouts[0]))
        thumb = fit_width(thumb, 900)
        tiles.append(caption_bar(thumb, f"{opt.name}  {opt.meta.get('frame_id', '')}"))
    if not tiles:
        return
    width = min(t.shape[1] for t in tiles)
    stack = np.vstack([cv2.copyMakeBorder(fit_width(t, width), 0, 12, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for t in tiles])
    cv2.imwrite(str(out_dir / f"{family}_sheet.png"), stack,
                [cv2.IMWRITE_PNG_COMPRESSION, 6])


PREVIEW_HEAD = r"""\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage[compatibility=false]{caption}
\usepackage{subcaption}
\usepackage{xcolor}
% Cross-references to the paper's tables cannot resolve here; they print as ??
% and are the only warnings this document should raise.
\begin{document}
"""


def write_preview(manifest: dict, out_dir: Path, compile_it: bool) -> None:
    """Every option typeset in a two-column class, one per page.

    A composite PNG shows what the panels contain; only a two-column layout at
    the real column width shows whether it is readable at the size it prints.
    """
    def tex_safe(text: str) -> str:
        return str(text).replace("_", "\\_")

    body = [PREVIEW_HEAD]
    for entry in manifest["options"]:
        for layout, tex in entry.get("latex_preview", {}).items():
            meta = ", ".join(f"{k}={v}" for k, v in entry["meta"].items()
                             if not isinstance(v, (dict, list)))
            # A figure* is placed at the top of a later page, never beside the
            # text that introduces it, so the label goes underneath.
            body.append(tex)
            body.append(f"\\section*{{{tex_safe(entry['name'])} "
                        f"\\textemdash\\ {layout}}}")
            body.append(f"\\noindent\\texttt{{\\small {tex_safe(meta[:300])}}}\\par")
            body.append("\\clearpage")
    body.append("\\end{document}")
    tex_path = out_dir / "preview.tex"
    tex_path.write_text("\n".join(body))
    print(f"preview:  {tex_path}")
    if not compile_it:
        return
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
                            "preview.tex"], cwd=out_dir, capture_output=True, text=True)
    if r.returncode == 0:
        print(f"preview:  {out_dir / 'preview.pdf'}")
    else:
        tail = "\n".join(r.stdout.strip().splitlines()[-12:])
        print(f"preview: pdflatex failed\n{tail}")


# ── Promotion ────────────────────────────────────────────────────────────────

def promote(args) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path} -- render options first")
        return 1
    manifest = json.loads(manifest_path.read_text())
    entry = next((o for o in manifest["options"] if o["name"] == args.promote), None)
    if entry is None:
        print(f"unknown option '{args.promote}'")
        return 1

    stem = args.as_name
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for p in entry["panels"]:
        dst = args.figures_dir / f"{stem}_{p['slug']}.png"
        shutil.copyfile(args.out_dir / p["file"], dst)
        written.append(dst)

    prov_path = args.figures_dir / "qualitative_provenance.json"
    prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
    prov[stem] = {"option": entry["name"], "family": entry["family"],
                  "meta": entry["meta"],
                  "panels": [{"slug": p["slug"], "note": p["note"]} for p in entry["panels"]]}
    prov_path.write_text(json.dumps(prov, indent=2, sort_keys=True))

    for d in written:
        print(f"  wrote {d}")
    print(f"  provenance recorded in {prov_path}")
    print("\nLaTeX (rename figure paths already point at figures/):\n")
    layout = args.layout or next(iter(entry["latex"]))
    print(entry["latex"][layout].replace(entry["name"], stem))
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--family", action="append", choices=list(FAMILIES),
                    help="repeatable; default is every family")
    ap.add_argument("--per-family", type=int, default=4,
                    help="how many options to render per family (default 4)")
    ap.add_argument("--vlm", default="llava_34b", help="backend tag for ladder/vlmcost")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "qualitative_candidates" / "options")
    ap.add_argument("--figures-dir", type=Path, default=REPO / "figures",
                    help="where --promote copies panels; point it at the "
                         "manuscript's figures directory")
    ap.add_argument("--scan-frames", type=int, default=600,
                    help="frames to consider for the full-frame families")
    ap.add_argument("--queue-n", type=int, default=8, help="thumbnails per queue row")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude", action="append", default=[],
                    metavar="FRAME_ID", help="repeatable; skip this frame")
    ap.add_argument("--no-auto-exclude", action="store_true",
                    help="allow frames already spent on another figure set")
    ap.add_argument("--triage-model-decided", action="store_true",
                    help="vlmcost: only show deletions the VLM actually made, "
                         "excluding pre-filter and dense-signal-only rejects")
    ap.add_argument("--triage-only-incremental", action="store_true",
                    help="vlmcost: keep only deletions the free signal had not "
                         "already made (a sharper claim, but cells may run dry)")
    ap.add_argument("--border", type=int, default=6,
                    help="reject masks within this many px of the frame edge")
    ap.add_argument("--min-brightness", type=float, default=42.0)
    ap.add_argument("--min-contrast", type=float, default=18.0)
    ap.add_argument("--no-preview-pdf", action="store_true",
                    help="write preview.tex but do not run pdflatex on it")
    ap.add_argument("--list", action="store_true",
                    help="print what each family would pick, without rendering")
    ap.add_argument("--promote", metavar="NAME",
                    help="copy an option into --figures-dir")
    ap.add_argument("--as", dest="as_name", metavar="STEM",
                    help="figure basename to promote to, e.g. qual_ladder")
    ap.add_argument("--layout", help="promote: which layout's LaTeX to print")
    args = ap.parse_args()

    if args.promote:
        if not args.as_name:
            print("--promote requires --as")
            return 1
        return promote(args)

    excluded = set(args.exclude)
    if not args.no_auto_exclude:
        excluded |= set(USED_COMPANION) | set(USED_GEOMETRY_SET)
        prov = args.figures_dir / "qualitative_provenance.json"
        if prov.exists():
            for entry in json.loads(prov.read_text()).values():
                fid = (entry.get("meta") or {}).get("frame_id")
                if fid:
                    excluded.add(fid)
        print(f"excluding {len(excluded)} frames already spent on other figures")

    print(f"reading {EXPORT.name} ...")
    rows = load_rows()
    families = args.family or list(FAMILIES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"backend": args.vlm, "seed": args.seed,
                "excluded_frames": sorted(excluded), "options": []}
    snippets = []

    for family in families:
        options = FAMILIES[family](rows, args, excluded)
        if not options:
            print(f"[{family}] no options")
            continue
        if args.list:
            for opt in options:
                print(f"  {opt.name:16s} {opt.meta}")
            continue
        for opt in options:
            manifest["options"].append(write_option(opt, args.out_dir))
            snippets.append(manifest["options"][-1]["latex"][opt.layouts[0]])
        # No photograph twice in one paper: the families rank overlapping pools,
        # and left to themselves signal and vlmcost pick the same four crops.
        excluded |= {f for opt in options for f in opt.frames}
        sheet_for(family, options, args.out_dir)
        print(f"[{family}] {len(options)} options -> {args.out_dir / family}")

    if args.list:
        return 0

    # Merge, don't clobber: rendering one family must not drop the options a
    # previous run wrote for the others, since --promote resolves names against
    # whatever is in this manifest.
    man_path = args.out_dir / "manifest.json"
    if man_path.exists():
        previous = json.loads(man_path.read_text()).get("options", [])
        kept_previous = [o for o in previous if o["family"] not in families]
        manifest["options"] = kept_previous + manifest["options"]
        snippets = [o["latex"][next(iter(o["latex"]))] for o in kept_previous] + snippets
    man_path.write_text(json.dumps(manifest, indent=2))
    (args.out_dir / "snippets.tex").write_text("\n".join(snippets))
    write_preview(manifest, args.out_dir, compile_it=not args.no_preview_pdf)
    print(f"\nmanifest: {args.out_dir / 'manifest.json'}")
    print(f"snippets: {args.out_dir / 'snippets.tex'}")
    print("browse the *_sheet.png files and the composites/, then:")
    print("  python make_qualitative_figure_options.py --promote <option> --as <figure_stem>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
