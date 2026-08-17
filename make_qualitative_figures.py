#!/usr/bin/env python3
"""
Mine stored triage results for qualitative-figure candidates and render them at
the exact size and style the paper's figures use, so a pick can go straight into
paper/*/figures/ without re-cropping by hand.

This is the render half of select_qualitative_frames.py. That script narrows the
corpus down to contact sheets of thumbnails for browsing; this one produces
publication-ready panels: 480x360 (4:3, matching every existing qual_*.png),
cropped without distortion, with the paper's overlay conventions --

    camera panel     red translucent SAM mask under review
    LiDAR panel      cyan returns on black, green mask contour
    discovery panel  cyan Swin component, orange confirmed-region box

No new inference: everything comes from the per-frame JSON the runs already
wrote, plus the SAM annotation PNGs and the shared discovery component masks.

Selection is the point. Most masks in a category make a bad figure -- too small
to read at 0.19\\textwidth, clipped by the frame border, or shot at night where
the crop prints as a black rectangle. Candidates are filtered on those grounds
and then scored, so the top of each category is what is worth looking at.

Usage:
    # render 8 candidates per category into qualitative_candidates/paper_ready/
    python make_qualitative_figures.py --vlm llava_34b --per-category 8

    # one category, more options, including night frames
    python make_qualitative_figures.py --category discovery --per-category 16 --allow-dark

    # copy a chosen panel into the paper (keeps the manifest entry as provenance)
    python make_qualitative_figures.py --promote disagreement_02 --as qual_review3
"""
import argparse
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import config
from core.mask_extractor import extract_proposals

# ── Paper geometry ───────────────────────────────────────────────────────────
# Every existing figure is 480x360 placed at 0.19\textwidth in a figure*; the
# subfigures are stacked in pairs, so all panels must share one aspect ratio or
# the rows stop lining up.
PANEL_W, PANEL_H = 480, 360
PANEL_AR = PANEL_W / PANEL_H

# Context around the object, as a fraction of its longer side. Tight crops read
# as "trust me" — the reviewer needs to see what the mask did not cover.
CROP_PAD_FRAC = 0.45
CROP_MIN_PX = 120          # never crop a window smaller than this before upscaling

MASK_ALPHA = 0.45
COLOR_MASK = (0, 0, 255)       # BGR red — SAM mask under review
COLOR_CONTOUR = (0, 255, 0)    # BGR green — mask outline on the LiDAR panel
COLOR_DISCOVERY = (255, 255, 0)  # BGR cyan — Swin component
COLOR_DISC_BOX = (0, 165, 255)   # BGR orange — VLM-confirmed region

HUMAN_CLASSES = {"cyclist", "pedestrian"}

# ── Categories ───────────────────────────────────────────────────────────────
# One per decision path the paper illustrates. `lidar_panel` marks the ones
# whose story is geometric, where the camera crop alone shows nothing.
CATEGORIES = {
    "hallucination":   dict(lidar_panel=False),
    "consistency_fail": dict(lidar_panel=True),
    "depth_support":   dict(lidar_panel=True),
    "disagreement":    dict(lidar_panel=False),
    "discovery":       dict(lidar_panel=False),
    # --paired-with only: the same mask, kept by one backend and deleted by the
    # other. One panel per mask -- the pixels are identical by construction, so
    # a second panel would show nothing; the divergence lives in the verdicts,
    # which the manifest carries into the caption.
    "flip_a_keeps":    dict(lidar_panel=False),
    "flip_b_keeps":    dict(lidar_panel=False),
}
PAIRED_CATEGORIES = ["flip_a_keeps", "flip_b_keeps"]


@dataclass
class Candidate:
    category: str
    frame_id: str
    mask_id: Optional[int]        # None for discovery candidates
    class_name: str
    bbox: tuple                   # (x1, y1, x2, y2) in full-frame pixels
    pixel_count: int
    agents: dict
    scores: dict
    note: str                     # factual one-liner for the manifest
    disc_bbox_384: Optional[list] = None
    quality: float = 0.0          # suitability score, filled in later
    reasons: dict = field(default_factory=dict)
    paired: dict = field(default_factory=dict)  # per-run verdicts, --paired-with only


# ── Candidate mining ─────────────────────────────────────────────────────────

def iter_results(results_dir: Path):
    for f in sorted(results_dir.glob("frame_*.json")):
        d = json.loads(f.read_text())
        if "masks" in d:
            yield d


def mine(results_dir: Path, categories: list[str]) -> dict[str, list[Candidate]]:
    """Bucket every mask by the decision path it took. Cheap: JSON only."""
    out: dict[str, list[Candidate]] = {c: [] for c in categories}

    for d in iter_results(results_dir):
        fid = d["frame_id"]

        for m in d["masks"]:
            a, s = m["agents"], m.get("scores", {})
            common = dict(frame_id=fid, mask_id=m["mask_id"], class_name=m["class_name"],
                          bbox=tuple(m["bbox"]), pixel_count=m["pixel_count"],
                          agents=a, scores=s)
            alpha = s.get("swin_agreement")
            supp = s.get("lidar_support")

            if "hallucination" in out and a.get("bbox") == "background":
                out["hallucination"].append(Candidate(
                    category="hallucination", note=(
                        f"BBox agent returned background on a {m['class_name']} mask "
                        f"(Swin agreement {_pct(alpha)}, LiDAR support {_pct(supp)})"),
                    **common))

            # Everything looks right except the geometry: one negative, so the
            # mask is retained rather than deleted.
            if ("consistency_fail" in out and a.get("bbox") == "valid"
                    and a.get("quality") == "good" and a.get("consistency") == "fail"):
                out["consistency_fail"].append(Candidate(
                    category="consistency_fail", note=(
                        f"valid + good but LiDAR support {_pct(supp)} below tau; "
                        f"single negative, retained and flagged"),
                    **common))

            # Geometry as a *rejecting* signal: the same failure, but seconded.
            if ("depth_support" in out and a.get("consistency") == "fail"
                    and m["triage"] == "reject" and a.get("quality") == "bad"):
                out["depth_support"].append(Candidate(
                    category="depth_support", note=(
                        f"rejected on two negatives: quality bad (agreement {_pct(alpha)}) "
                        f"and LiDAR support {_pct(supp)}"),
                    **common))

            # Retained because the judges disagree — the downgrade rule at work.
            if "disagreement" in out and m["triage"] in ("human_review", "refine"):
                if a.get("bbox") == "invalid" and a.get("quality") == "good":
                    note = (f"BBox invalid vs Swin agreement {_pct(alpha)}: downgraded to one "
                            f"negative and retained")
                elif a.get("bbox") == "valid" and a.get("quality") == "bad":
                    note = (f"BBox valid vs Swin agreement {_pct(alpha)}: single negative, "
                            f"retained")
                else:
                    note = None
                if note:
                    out["disagreement"].append(Candidate(
                        category="disagreement", note=note, **common))

        for disc in d.get("discovered", []):
            if "discovery" not in out or not disc.get("confirmed"):
                continue
            out["discovery"].append(Candidate(
                category="discovery", frame_id=fid, mask_id=None,
                class_name=disc["class_name"], bbox=tuple(disc["bbox_orig"]),
                pixel_count=disc["pixel_count_384"], agents={}, scores={},
                disc_bbox_384=disc["bbox_384"],
                note=(f"SAM missed this {disc['class_name']}; Swin proposed it and the VLM "
                      f"answered '{disc['vlm_response']}'")))

    return out


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * float(v):.1f}%"


# ── Paired mining: where the two backends part company ───────────────────────

def reported(triage: str) -> str:
    """The paper's three outcomes. refine and human_review both keep every
    pixel, so they pool; only reject is destructive."""
    return "reject" if triage == "reject" else ("accept" if triage == "accept" else "retained")


def _index(results_dir: Path) -> dict:
    idx = {}
    for d in iter_results(results_dir):
        for m in d["masks"]:
            idx[(d["frame_id"], m["mask_id"])] = m
    return idx


def mine_paired(dir_a: Path, dir_b: Path, tag_a: str, tag_b: str) -> dict[str, list[Candidate]]:
    """Masks one backend keeps and the other deletes.

    This is the 18.8% of masks that carry different pixels into the two label
    sets, so it is the divergence a downstream model would actually inherit.
    Swin agreement and LiDAR support are identical across runs by construction,
    which is what makes the pair attributable to the VLM: the note records the
    shared numbers once and the differing verdicts twice.
    """
    a_idx, b_idx = _index(dir_a), _index(dir_b)
    out: dict[str, list[Candidate]] = {c: [] for c in PAIRED_CATEGORIES}

    for key, ma in a_idx.items():
        mb = b_idx.get(key)
        if mb is None:
            continue
        ra, rb = reported(ma["triage"]), reported(mb["triage"])
        if (ra == "reject") == (rb == "reject"):
            continue                      # same fate; nothing to show

        cat = "flip_b_keeps" if ra == "reject" else "flip_a_keeps"
        keeper, deleter = (tag_b, tag_a) if ra == "reject" else (tag_a, tag_b)
        mk, md = (mb, ma) if ra == "reject" else (ma, mb)
        s = ma.get("scores", {})
        note = (
            f"{keeper} kept it (bbox={mk['agents'].get('bbox')}, "
            f"quality={mk['agents'].get('quality')} -> {reported(mk['triage'])}); "
            f"{deleter} deleted it (bbox={md['agents'].get('bbox')}, "
            f"quality={md['agents'].get('quality')} -> reject). "
            f"Shared inputs: Swin agreement {_pct(s.get('swin_agreement'))}, "
            f"LiDAR support {_pct(s.get('lidar_support'))}")

        out[cat].append(Candidate(
            category=cat, frame_id=key[0], mask_id=key[1], class_name=ma["class_name"],
            bbox=tuple(ma["bbox"]), pixel_count=ma["pixel_count"],
            agents=ma["agents"], scores=s, note=note,
            paired={tag_a: {"agents": ma["agents"], "triage": ma["triage"],
                            "reported": ra},
                    tag_b: {"agents": mb["agents"], "triage": mb["triage"],
                            "reported": rb}}))

    return out


# ── Suitability ──────────────────────────────────────────────────────────────
# A figure panel has to survive being printed two inches wide in a column. These
# checks throw out what will not, before anything is rendered.

def suitability(c: Candidate, W: int, H: int, args) -> Optional[float]:
    x1, y1, x2, y2 = c.bbox
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    if args.classes and c.class_name not in args.classes:
        return None

    # High Swin agreement is a proxy for "the mask actually covers the object":
    # the class map fires on the pixels inside it. Low-alpha masks are often
    # leaked or misplaced, which makes them poor figures whatever the verdicts.
    alpha_val = c.scores.get("swin_agreement")
    if args.min_alpha > 0 and (alpha_val is None or float(alpha_val) < args.min_alpha):
        c.reasons["reject"] = f"swin agreement below {args.min_alpha}"
        return None

    # Clipped at the frame edge: the crop cannot show the object in context, and
    # a half-object reads as a rendering bug rather than an example.
    if x1 < args.border or y1 < args.border or x2 > W - args.border or y2 > H - args.border:
        c.reasons["reject"] = "touches frame border"
        return None

    area_frac = (bw * bh) / float(W * H)
    if not (args.min_area <= area_frac <= args.max_area):
        c.reasons["reject"] = f"area fraction {area_frac:.4f} outside band"
        return None

    ar = bw / bh
    if not (0.15 <= ar <= 6.0):
        c.reasons["reject"] = f"extreme aspect ratio {ar:.2f}"
        return None

    # Size sweet spot: large enough to read, small enough to keep surroundings.
    # Log-gaussian around 3% of the frame.
    size_term = float(np.exp(-((np.log(area_frac) - np.log(0.03)) ** 2) / (2 * 1.1 ** 2)))

    # A mask that fills its own box is legible as a shape; a sparse scatter is
    # not, whatever the verdict says about it. Discovery components are measured
    # in the 384x384 class-map space they were extracted from, and their density
    # matters most: a hollow component draws as a ring around the object.
    if c.mask_id is not None:
        fill = c.pixel_count / float(bw * bh)
    elif c.disc_bbox_384:
        dx1, dy1, dx2, dy2 = c.disc_bbox_384
        fill = c.pixel_count / float(max((dx2 - dx1) * (dy2 - dy1), 1))
    else:
        fill = 1.0
    fill_term = float(np.clip(fill / 0.5, 0, 1)) if fill < 0.5 else 1.0

    rare_term = 1.6 if c.class_name in HUMAN_CLASSES else (1.2 if c.class_name == "sign" else 1.0)

    # Category-specific: prefer the example that shows the mechanism most
    # sharply, not merely the biggest object.
    signal = 1.0
    supp = c.scores.get("lidar_support")
    alpha = c.scores.get("swin_agreement")
    if c.category == "depth_support" and supp is not None:
        signal = 1.0 + (1.0 - min(float(supp) / 0.1, 1.0))       # the emptier, the clearer
    elif c.category == "consistency_fail" and supp is not None:
        signal = 1.0 + (1.0 - min(float(supp) / 0.1, 1.0))
    elif c.category == "disagreement" and alpha is not None:
        signal = 1.0 + abs(float(alpha) - 0.5) * 2               # the starker the split
    elif c.category == "hallucination":
        signal = 1.0 + min(c.pixel_count / 20000.0, 1.0)
    elif c.category in PAIRED_CATEGORIES:
        # A unilateral deletion on a `background` verdict is the sharpest form
        # of the divergence: one model saw an object, the other saw a surface.
        verdicts = {r["agents"].get("bbox") for r in c.paired.values()}
        signal = 1.6 if "background" in verdicts else 1.0

    c.reasons.update(area_frac=round(area_frac, 5), fill=round(fill, 3))
    return size_term * fill_term * rare_term * signal


def legible(crop_bgr: np.ndarray, args) -> Optional[str]:
    """Reject crops that will print as a dark or flat rectangle."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    if not args.allow_dark and gray.mean() < args.min_brightness:
        return f"too dark (mean {gray.mean():.0f})"
    if gray.std() < args.min_contrast:
        return f"too flat (std {gray.std():.0f})"
    return None


# ── Rendering ────────────────────────────────────────────────────────────────

def crop_window(bbox, W: int, H: int) -> tuple[int, int, int, int]:
    """A 4:3 window around bbox, clamped into the frame without distorting."""
    x1, y1, x2, y2 = bbox
    pad = CROP_PAD_FRAC * max(x2 - x1, y2 - y1)
    cx1, cy1 = x1 - pad, y1 - pad
    cx2, cy2 = x2 + pad, y2 + pad

    w, h = max(cx2 - cx1, CROP_MIN_PX), max(cy2 - cy1, CROP_MIN_PX)
    if w / h < PANEL_AR:                 # too tall — widen
        w = h * PANEL_AR
    else:                                # too wide — heighten
        h = w / PANEL_AR

    # Shrink to fit the frame before shifting, so the aspect ratio survives.
    if w > W:
        w, h = W, W / PANEL_AR
    if h > H:
        h, w = H, H * PANEL_AR

    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    nx1 = int(round(min(max(cx - w / 2, 0), W - w)))
    ny1 = int(round(min(max(cy - h / 2, 0), H - h)))
    return nx1, ny1, nx1 + int(round(w)), ny1 + int(round(h))


def to_panel(img: np.ndarray, window) -> np.ndarray:
    x1, y1, x2, y2 = window
    crop = img[y1:y2, x1:x2]
    interp = cv2.INTER_AREA if crop.shape[1] > PANEL_W else cv2.INTER_CUBIC
    return cv2.resize(crop, (PANEL_W, PANEL_H), interpolation=interp)


def sam_mask(frame_id: str, mask_id: int) -> Optional[np.ndarray]:
    ann = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann.exists():
        return None
    match = next((p for p in extract_proposals(ann, frame_id) if p.mask_id == mask_id), None)
    return None if match is None else match.pixel_mask


def discovery_mask(frame_id: str, bbox_384: list, W: int, H: int) -> Optional[np.ndarray]:
    """The stored connected component for one discovery candidate, upscaled.

    Painting the bounding box instead would overstate what the agent confirmed —
    the same mistake regenerate_discovery_masks.py exists to undo.
    """
    base = config.DATA_ROOT / "vlm" / "discovery_masks"
    png, meta = base / f"{frame_id}.png", base / f"{frame_id}.json"
    if not (png.exists() and meta.exists()):
        return None
    entries = json.loads(meta.read_text())
    idx = next((i for i, e in enumerate(entries) if list(e["bbox_384"]) == list(bbox_384)), None)
    if idx is None:
        return None
    comp = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    if comp is None:
        return None
    return cv2.resize((comp == idx + 1).astype(np.uint8), (W, H),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes in a component, keeping its outer silhouette.

    Swin components are often hollow -- the class map fires on a pedestrian's
    limbs and coat edges but not the dark interior -- so the raw overlay reads
    as a cyan ring around the object instead of the object. Filling only the
    holes enclosed by the component's own outline keeps the silhouette honest
    while making the shape legible at figure size. The caption says the panels
    are filled.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def overlay(img: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    tinted = img.copy()
    tinted[mask] = color
    return cv2.addWeighted(tinted, MASK_ALPHA, img, 1 - MASK_ALPHA, 0)


def lidar_panel(frame_id: str, mask: Optional[np.ndarray], window) -> Optional[np.ndarray]:
    """Cyan returns on black with the mask outlined, as in the paper's panel (b)."""
    path = config.LIDAR_DIR / f"{frame_id}.png"
    if not path.exists():
        return None
    lid = cv2.imread(str(path))
    if lid is None:
        return None

    hit = lid.max(axis=2) > 0
    # Modulate by the depth channel so near and far returns stay distinguishable.
    depth = lid[:, :, 2].astype(np.float32)
    inten = np.zeros_like(depth)
    if hit.any():
        d = depth[hit]
        lo, hi = float(d.min()), float(d.max())
        inten[hit] = 90 + 165 * ((d - lo) / (hi - lo) if hi > lo else 1.0)

    canvas = np.zeros_like(lid)
    canvas[:, :, 0] = inten      # B
    canvas[:, :, 1] = inten      # G  -> cyan
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, COLOR_CONTOUR, 2)
    return to_panel(canvas, window)


def render(c: Candidate, args) -> Optional[dict]:
    cam_path = config.CAMERA_DIR / f"{c.frame_id}.png"
    img = cv2.imread(str(cam_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    mask = None
    if c.mask_id is not None:
        mask = sam_mask(c.frame_id, c.mask_id)
        if mask is None:
            return None
        canvas = overlay(img, mask, COLOR_MASK)
    else:
        mask = discovery_mask(c.frame_id, c.disc_bbox_384, W, H)
        if mask is None:
            return None
        if not args.no_fill_discovery:
            mask = fill_holes(mask)
        canvas = overlay(img, mask, COLOR_DISCOVERY)
        x1, y1, x2, y2 = c.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), COLOR_DISC_BOX,
                      max(2, int(round(max(W, H) / 400))))

    window = crop_window(c.bbox, W, H)
    panel = to_panel(canvas, window)

    why = legible(to_panel(img, window), args)
    if why:
        c.reasons["reject"] = why
        return None

    out = {"camera": panel}
    if CATEGORIES[c.category]["lidar_panel"]:
        lp = lidar_panel(c.frame_id, mask, window)
        if lp is not None:
            out["lidar"] = lp
    return out


# ── Diversity ────────────────────────────────────────────────────────────────

def pick(cands: list[Candidate], n: int) -> list[Candidate]:
    """Best per frame, then round-robin over classes so vehicles do not sweep."""
    best: dict[str, Candidate] = {}
    for c in cands:
        cur = best.get(c.frame_id)
        if cur is None or c.quality > cur.quality:
            best[c.frame_id] = c

    by_class: dict[str, list[Candidate]] = {}
    for c in best.values():
        by_class.setdefault(c.class_name, []).append(c)
    for lst in by_class.values():
        lst.sort(key=lambda c: c.quality, reverse=True)

    order = sorted(by_class, key=lambda k: -by_class[k][0].quality)
    picked, idx = [], {k: 0 for k in order}
    while len(picked) < n and any(idx[k] < len(by_class[k]) for k in order):
        for k in order:
            if len(picked) >= n:
                break
            if idx[k] < len(by_class[k]):
                picked.append(by_class[k][idx[k]])
                idx[k] += 1
    return picked


def contact_sheet(panels: list[tuple[str, np.ndarray]], cols: int = 4) -> np.ndarray:
    """One browsable sheet per category. Labels are burned in so a pick can be
    named from the sheet alone."""
    tw, th, cap = 320, 240, 22
    rows = (len(panels) + cols - 1) // cols
    sheet = np.zeros((rows * (th + cap), cols * tw, 3), np.uint8)
    for i, (name, p) in enumerate(panels):
        r, col = divmod(i, cols)
        y, x = r * (th + cap), col * tw
        sheet[y:y + th, x:x + tw] = cv2.resize(p, (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(sheet, name, (x + 4, y + th + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    return sheet


# ── Promotion ────────────────────────────────────────────────────────────────

def promote(args) -> int:
    manifest_path = args.out_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path} — render candidates first")
        return 1
    manifest = json.loads(manifest_path.read_text())
    entry = next((e for e in manifest["candidates"] if e["name"] == args.promote), None)
    if entry is None:
        print(f"unknown candidate '{args.promote}'")
        return 1

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for panel, src in entry["files"].items():
        suffix = "" if panel == "camera" else f"_{panel}"
        dst = args.figures_dir / f"{args.as_name}{suffix}.png"
        shutil.copyfile(args.out_dir / src, dst)
        copied.append(dst)

    used = args.figures_dir / "qualitative_provenance.json"
    log = json.loads(used.read_text()) if used.exists() else {}
    log[args.as_name] = {k: entry[k] for k in
                         ("category", "frame_id", "mask_id", "class", "note", "agents", "scores")}
    used.write_text(json.dumps(log, indent=2, sort_keys=True))

    for d in copied:
        print(f"  wrote {d}")
    print(f"  provenance recorded in {used}")
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--vlm", default="llava_34b", help="run tag under DATA_ROOT/vlm/")
    ap.add_argument("--paired-with", metavar="TAG",
                    help="second run tag; switches to paired mode and mines masks one "
                         "backend keeps and the other deletes")
    ap.add_argument("--category", choices=list(CATEGORIES), action="append",
                    help="repeatable; default is every category")
    ap.add_argument("--per-category", type=int, default=8)
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "qualitative_candidates" / "paper_ready")
    ap.add_argument("--figures-dir", type=Path,
                    default=Path(__file__).parent / "paper" / "icaart" / "figures")
    ap.add_argument("--exclude", action="append", default=[],
                    help="frame_id to skip; repeatable (e.g. frames already in the paper)")
    ap.add_argument("--no-auto-exclude", action="store_true",
                    help="do not skip frames recorded in figures/qualitative_provenance.json")
    ap.add_argument("--class", dest="classes", action="append", default=[],
                    metavar="NAME", help="keep only this class; repeatable")
    ap.add_argument("--min-alpha", type=float, default=0.0,
                    help="require Swin agreement at least this high (0-1); finds masks "
                         "that actually cover the object")
    ap.add_argument("--min-area", type=float, default=0.0015,
                    help="smallest bbox area as a fraction of the frame (default 0.0015)")
    ap.add_argument("--max-area", type=float, default=0.30)
    ap.add_argument("--border", type=int, default=6,
                    help="reject masks within this many px of the frame edge")
    ap.add_argument("--min-brightness", type=float, default=42.0)
    ap.add_argument("--min-contrast", type=float, default=18.0)
    ap.add_argument("--no-fill-discovery", action="store_true",
                    help="draw discovery components exactly as predicted, holes and all")
    ap.add_argument("--allow-dark", action="store_true",
                    help="keep night frames (useful for limitation figures)")
    ap.add_argument("--promote", metavar="NAME", help="copy a rendered candidate into the paper")
    ap.add_argument("--as", dest="as_name", metavar="STEM",
                    help="figure basename to promote to, e.g. qual_review3")
    args = ap.parse_args()

    if args.promote:
        if not args.as_name:
            print("--promote requires --as")
            return 1
        return promote(args)

    config.set_run_tag(args.vlm)
    results_a = config.RESULTS_DIR
    if args.paired_with:
        config.set_run_tag(args.paired_with)
        results_b = config.RESULTS_DIR
        config.set_run_tag(args.vlm)          # panels are rendered from shared inputs
        categories = [c for c in (args.category or PAIRED_CATEGORIES)
                      if c in PAIRED_CATEGORIES]
        print(f"pairing {results_a} against {results_b} ...")
        buckets = mine_paired(results_a, results_b, args.vlm, args.paired_with)
    else:
        categories = [c for c in (args.category or CATEGORIES) if c not in PAIRED_CATEGORIES]
        print(f"scanning {results_a} ...")
        buckets = mine(results_a, categories)

    # One frame size for the whole corpus; read it once for the filters.
    probe = next(iter(sorted(config.CAMERA_DIR.glob("frame_*.png"))), None)
    H, W = cv2.imread(str(probe)).shape[:2]

    # A frame already in the paper is not a new example. Every promoted figure is
    # logged, so the pool shrinks by itself as picks are made.
    excluded = set(args.exclude)
    if not args.no_auto_exclude:
        used = args.figures_dir / "qualitative_provenance.json"
        if used.exists():
            in_paper = {e["frame_id"] for e in json.loads(used.read_text()).values()
                        if e.get("frame_id")}
            excluded |= in_paper
            print(f"skipping {len(in_paper)} frames already used in {args.figures_dir.name}/")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"run": args.vlm, "panel_size": [PANEL_W, PANEL_H], "candidates": []}

    for cat in categories:
        cands = [c for c in buckets[cat] if c.frame_id not in excluded]
        scored = []
        for c in cands:
            q = suitability(c, W, H, args)
            if q is not None:
                c.quality = q
                scored.append(c)
        print(f"[{cat}] {len(cands)} in category, {len(scored)} pass the filters")

        cat_dir = args.out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for old in cat_dir.glob("*.png"):
            old.unlink()

        sheet, written = [], 0
        # Over-draw: some candidates only fail the brightness test once rendered.
        for c in pick(scored, args.per_category * 3):
            if written >= args.per_category:
                break
            panels = render(c, args)
            if panels is None:
                continue
            name = f"{cat}_{written:02d}"
            files = {}
            for panel, img in panels.items():
                suffix = "" if panel == "camera" else f"_{panel}"
                rel = f"{cat}/{name}{suffix}.png"
                cv2.imwrite(str(args.out_dir / rel), img)
                files[panel] = rel
            sheet.append((name, panels["camera"]))
            manifest["candidates"].append({
                "name": name, "category": cat, "frame_id": c.frame_id,
                "mask_id": c.mask_id, "class": c.class_name, "bbox": list(c.bbox),
                "pixel_count": c.pixel_count, "agents": c.agents, "scores": c.scores,
                "note": c.note, "score": round(c.quality, 3), "files": files,
                **({"paired": c.paired} if c.paired else {}),
            })
            written += 1

        if sheet:
            cv2.imwrite(str(args.out_dir / f"{cat}_sheet.png"), contact_sheet(sheet))
        print(f"[{cat}] wrote {written} panels -> {cat_dir}")

    # Merge, don't clobber: a paired run and a decision-path run write different
    # categories, and --promote resolves names against whatever is in here.
    man_path = args.out_dir / "manifest.json"
    if man_path.exists():
        kept = [c for c in json.loads(man_path.read_text()).get("candidates", [])
                if c["category"] not in categories]
        manifest["candidates"] = kept + manifest["candidates"]
    man_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest: {args.out_dir / 'manifest.json'}")
    print("browse the *_sheet.png files, then:")
    print("  python make_qualitative_figures.py --promote <name> --as <figure_stem>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
