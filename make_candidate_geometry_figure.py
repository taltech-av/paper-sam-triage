#!/usr/bin/env python3
"""
Render the four discovery-candidate outcomes as real crops, one per cell of the
candidate-geometry join: edge bleed, boundary growth, false alarm, new object.

This is the photographic counterpart to paper/ral/diagrams/candidate_geometry.tex
and deliberately shares its palette -- grey for a SAM mask the pipeline already
had, orange for a candidate the human rejected, teal for one they kept -- so the
schematic and the photographs read as one argument.

The cell a candidate belongs to is the join of two things:

  geometry   does the candidate's dilated ring contain a SAM mask of its own
             class? This is `abuts_same_class` in analyze_human_verification,
             computed here from the bundle's stored `touching_sam` so the figure
             cannot disagree with tables/discovery_geometry.
  verdict    what the human said about the candidate itself.

Candidates are scored for legibility rather than picked at random: a figure has
to show the mechanism, and a 30-pixel rim at the horizon does not. The chosen
frame ids are printed so the selection is reproducible and auditable.

Usage:
    python make_candidate_geometry_figure.py
    python make_candidate_geometry_figure.py --list          # show top picks
    python make_candidate_geometry_figure.py --pick edge_bleed=frame_012345:3
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import config

EXPORT = Path(__file__).parent / "human_verified_output" / "verify_export.csv"
OUT_DIR = Path(__file__).parent / "paper" / "ral" / "figures"

NEIGHBOUR_RADIUS = 9
CLASS_FOLD = {"cyclist": "human", "pedestrian": "human"}
FOLD_ID = {"vehicle": 1, "sign": 2, "human": 3}
CORRECT, INCORRECT = "correct", "incorrect"

# BGR, matching candidate_geometry.tex: black!13 / orange!45 / teal!40
SAM_BGR = (120, 120, 120)
BAD_BGR = (77, 168, 244)     # orange
GOOD_BGR = (150, 160, 79)    # teal
MASK_ALPHA = 0.5

CELLS = {
    "edge_bleed":      ("same",  INCORRECT, "Edge bleed"),
    "boundary_growth": ("same",  CORRECT,   "Boundary growth"),
    "false_alarm":     ("other", INCORRECT, "False alarm"),
    "new_object":      ("other", CORRECT,   "New object"),
}


def fold(name):
    return CLASS_FOLD.get(name, name)


def frame_id_of(row):
    return Path(row["frame"]).stem


def load_rows():
    with EXPORT.open() as fh:
        return list(csv.DictReader(fh))


def geometry_of(row, class_of):
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


def candidate_pixels(frame_id, index, shape):
    """The candidate's own component in camera space, or None."""
    png = config.DATA_ROOT / "vlm" / "discovery_masks" / f"{frame_id}.png"
    if not png.exists():
        return None
    height, width = shape
    pixel_map = cv2.resize(np.array(Image.open(png)), (width, height),
                           interpolation=cv2.INTER_NEAREST)
    return pixel_map == index + 1


def sam_class_map(frame_id):
    """Folded class id per pixel over the SAM proposals, as the bundle painted them."""
    from core.mask_extractor import extract_proposals
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann_path.exists():
        return None
    ann = np.array(Image.open(ann_path))
    out = np.zeros(ann.shape[:2], dtype=np.uint8)
    for p in extract_proposals(ann_path, frame_id):
        out[p.pixel_mask] = FOLD_ID[fold(p.class_name)]
    return out


def score(row, pixels):
    """Legibility: mid-sized candidates, not slivers and not whole-frame blobs."""
    area = int(pixels.sum())
    if not 900 <= area <= 90_000:
        return -1.0
    ys, xs = np.where(pixels)
    h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
    if min(h, w) < 18:
        return -1.0
    fill = area / float(h * w)          # solid blobs read better than scatter
    centre = 1.0 - abs((xs.mean() / pixels.shape[1]) - 0.5) * 2
    return fill * 0.6 + centre * 0.4


def overlay(img, mask, bgr):
    img[mask] = (np.array(bgr) * MASK_ALPHA + img[mask] * (1 - MASK_ALPHA)).astype(np.uint8)


def render(frame_id, index, verdict, own_class, out_path):
    camera = cv2.imread(str(config.CAMERA_DIR / f"{frame_id}.png"))
    if camera is None:
        return False
    cmap = sam_class_map(frame_id)
    pixels = candidate_pixels(frame_id, index, camera.shape[:2])
    if cmap is None or pixels is None or not pixels.any():
        return False

    canvas = camera.copy()
    # Only same-class SAM masks are drawn: the claim the figure makes is about
    # what the candidate is lying against, and a van behind a sign is noise here.
    same = cmap == FOLD_ID[fold(own_class)]
    overlay(canvas, same, SAM_BGR)
    cand_bgr = BAD_BGR if verdict == INCORRECT else GOOD_BGR
    overlay(canvas, pixels, cand_bgr)
    for m, colour, thick in ((same, (255, 255, 255), 2), (pixels, cand_bgr, 2)):
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thick, cv2.LINE_AA)

    ys, xs = np.where(pixels)
    pad = int(max(np.ptp(ys), np.ptp(xs)) * 0.7) + 26
    y1, y2 = max(0, ys.min() - pad), min(canvas.shape[0], ys.max() + pad)
    x1, x2 = max(0, xs.min() - pad), min(canvas.shape[1], xs.max() + pad)
    crop = canvas[y1:y2, x1:x2]
    # 4:3, so the four panels tile without the subfigure widths fighting.
    h, w = crop.shape[:2]
    want = 4 / 3
    if w / h > want:
        need = int(w / want) - h
        y1, y2 = max(0, y1 - need // 2), min(canvas.shape[0], y2 + need - need // 2)
    else:
        need = int(h * want) - w
        x1, x2 = max(0, x1 - need // 2), min(canvas.shape[1], x2 + need - need // 2)
    crop = canvas[y1:y2, x1:x2]
    crop = cv2.resize(crop, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print top picks per cell and exit")
    parser.add_argument("--pick", action="append", default=[],
                        help="force a cell: cell=frame_id:candidate_index")
    parser.add_argument("--top", type=int, default=6)
    args = parser.parse_args()

    forced = {}
    for spec in args.pick:
        cell, _, where = spec.partition("=")
        frame_id, _, index = where.partition(":")
        forced[cell] = (frame_id, int(index))

    rows = load_rows()
    class_of = {(frame_id_of(r), int(r["maskId"])): fold(r["class"])
                for r in rows if r["class"]}
    discovery = [r for r in rows
                 if r.get("mask_source") == "discovery"
                 and r["verdict"] in (CORRECT, INCORRECT)
                 and r.get("mask_candidate_index") not in (None, "")]

    buckets = {cell: [] for cell in CELLS}
    by_frame = {}
    for row in discovery:
        by_frame.setdefault(frame_id_of(row), []).append(row)

    for frame_id, frame_rows in by_frame.items():
        shape = None
        for row in frame_rows:
            geom = geometry_of(row, class_of)
            for cell, (want_geom, want_verdict, _) in CELLS.items():
                if geom != want_geom or row["verdict"] != want_verdict:
                    continue
                if shape is None:
                    ann = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
                    if not ann.exists():
                        shape = False
                        break
                    shape = np.array(Image.open(ann)).shape[:2]
                if shape is False:
                    break
                index = int(float(row["mask_candidate_index"]))
                pixels = candidate_pixels(frame_id, index, shape)
                if pixels is None or not pixels.any():
                    continue
                s = score(row, pixels)
                if s > 0:
                    buckets[cell].append((s, frame_id, index, row["class"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for cell, (_, verdict, label) in CELLS.items():
        picks = sorted(buckets[cell], reverse=True)[:args.top]
        if args.list:
            print(f"\n{label} ({cell}) — {len(buckets[cell])} eligible")
            for s, frame_id, index, cls in picks:
                print(f"   {s:.3f}  {frame_id}:{index}  {cls}")
            continue
        if cell in forced:
            frame_id, index = forced[cell]
            cls = next((c for _, f, i, c in buckets[cell]
                        if f == frame_id and i == index), "vehicle")
            chosen = (frame_id, index, cls)
        elif picks:
            _, frame_id, index, cls = picks[0]
            chosen = (frame_id, index, cls)
        else:
            print(f"{label}: no eligible candidate")
            continue
        out = OUT_DIR / f"geom_{cell}.png"
        ok = render(chosen[0], chosen[1], verdict, chosen[2], out)
        print(f"{label:18s} {'wrote' if ok else 'FAILED'} {out.name}  "
              f"from {chosen[0]}:{chosen[1]} ({chosen[2]})")


if __name__ == "__main__":
    main()
