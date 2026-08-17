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


NEIGHBOUR_VIEW_RADIUS = 40      # px; "near the candidate" for the figure, not the join


def neighbour_pixels(pixels, cmap, own_class):
    """Same-class SAM pixels lying near the candidate.

    The ring join that defines the cell uses a 9 px test, which can be satisfied
    by a mask far too small to see. This is the *visual* neighbourhood: what a
    reader would actually have to spot to accept that the candidate is a rim on
    an object the pipeline already had.
    """
    same = cmap == FOLD_ID[fold(own_class)]
    if not same.any():
        return np.zeros_like(same)
    k = np.ones((2 * NEIGHBOUR_VIEW_RADIUS + 1,) * 2, np.uint8)
    return same & cv2.dilate(pixels.astype(np.uint8), k).astype(bool)


def score(row, pixels, cmap=None, geom=None, own_class=None):
    """Legibility: mid-sized candidates, not slivers and not whole-frame blobs.

    For the two `same` cells the figure has to show a relationship, not just a
    blob: if the SAM mask the candidate is a rim of is a speck beside it, the
    panel demonstrates nothing. Those candidates are rejected outright, and the
    score prefers a neighbour of comparable size to the candidate.
    """
    area = int(pixels.sum())
    if not 900 <= area <= 90_000:
        return -1.0
    ys, xs = np.where(pixels)
    h, w = int(np.ptp(ys)) + 1, int(np.ptp(xs)) + 1
    if min(h, w) < 18:
        return -1.0
    fill = area / float(h * w)          # solid blobs read better than scatter
    centre = 1.0 - abs((xs.mean() / pixels.shape[1]) - 0.5) * 2

    relation = 0.0
    if geom == "same" and cmap is not None:
        near = int(neighbour_pixels(pixels, cmap, own_class).sum())
        if near < 600 or near < 0.25 * area:
            return -1.0                 # the object it abuts would be invisible
        ratio = near / float(area)
        # Best when the two are within about 3x of each other in either
        # direction: the rim reads as a rim on something, not as the object.
        relation = float(np.exp(-((np.log(ratio)) ** 2) / (2 * 1.0 ** 2)))

    return fill * 0.4 + centre * 0.2 + relation * 0.4


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


def contact_sheet(tiles, cols=3):
    """One browsable sheet per cell, labelled so a pick can be named from it."""
    tw, th, cap = 400, 300, 24
    rows_n = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows_n * (th + cap), cols * tw, 3), np.uint8)
    for i, (name, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = r * (th + cap), c * tw
        sheet[y:y + th, x:x + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, name, (x + 4, y + th + 17), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="print top picks per cell and exit")
    parser.add_argument("--pick", action="append", default=[],
                        help="force a cell: cell=frame_id:candidate_index")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--sheet", action="store_true",
                        help="render the top picks per cell as browsable panels + "
                             "contact sheets instead of writing the paper figures")
    parser.add_argument("--sheet-dir", type=Path,
                        default=Path(__file__).parent / "qualitative_candidates" / "ral_ready")
    parser.add_argument("--exclude-icaart", action="store_true",
                        help="skip frames already used in the ICAART figures, so the two "
                             "papers keep disjoint photo sets")
    args = parser.parse_args()

    forced = {}
    for spec in args.pick:
        cell, _, where = spec.partition("=")
        frame_id, _, index = where.partition(":")
        forced[cell] = (frame_id, int(index))

    excluded = set()
    if args.exclude_icaart:
        prov = Path(__file__).parent / "paper" / "icaart" / "figures" / "qualitative_provenance.json"
        if prov.exists():
            excluded = {e["frame_id"] for e in json.loads(prov.read_text()).values()
                        if e.get("frame_id")}
            print(f"skipping {len(excluded)} frames already used in the ICAART figures")

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
        if frame_id in excluded:
            continue
        shape = None
        cmap = None                     # SAM class map, built once per frame
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
                if cmap is None:
                    cmap = sam_class_map(frame_id)
                s = score(row, pixels, cmap, want_geom, row["class"])
                if s > 0:
                    buckets[cell].append((s, frame_id, index, row["class"]))

    if args.sheet:
        manifest = {}
        for cell, (_, verdict, label) in CELLS.items():
            picks = sorted(buckets[cell], reverse=True)[:args.top]
            cell_dir = args.sheet_dir / cell
            cell_dir.mkdir(parents=True, exist_ok=True)
            for old_png in cell_dir.glob("*.png"):
                old_png.unlink()
            tiles, entries = [], []
            for n, (s_, frame_id, index, cls) in enumerate(picks):
                name = f"{cell}_{n:02d}"
                out = cell_dir / f"{name}.png"
                if not render(frame_id, index, verdict, cls, out):
                    continue
                tiles.append((name, cv2.imread(str(out))))
                entries.append({"name": name, "cell": cell, "frame_id": frame_id,
                                "candidate_index": index, "class": cls,
                                "score": round(s_, 3),
                                "pick": f"{cell}={frame_id}:{index}"})
            if tiles:
                cv2.imwrite(str(args.sheet_dir / f"{cell}_sheet.png"), contact_sheet(tiles))
            manifest[cell] = entries
            print(f"{label:18s} {len(tiles)} panels -> {cell_dir}")
        (args.sheet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nbrowse {args.sheet_dir}/*_sheet.png, then re-run with e.g."
              f"\n  python make_candidate_geometry_figure.py --pick {manifest['edge_bleed'][0]['pick']}"
              if manifest.get("edge_bleed") else "")
        return

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
