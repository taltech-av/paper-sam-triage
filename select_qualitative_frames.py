#!/usr/bin/env python3
"""
Mine already-computed VLM-triage results (no new inference) to shortlist candidate
frames for the paper's qualitative-examples figure, and render a per-category
contact sheet of cropped, annotated candidates for fast visual review.

This does NOT pick the final figure for you — it narrows ~4,135 frames x tens of
masks each down to a a handful of strong candidates per category, using the
pipeline's own categorical agent outputs (agents/*.py decisions already stored in
vlm/<tag>/results/frame_*.json) rather than re-inspecting pixel data. You then look
at the contact sheets and pick the crops you like.

Categories mirror the four cases named in the paper's qualitative-analysis
paragraph, plus one bonus (discovery, a headline contribution):

  hallucination   - BBox agent returns 'background': SAM proposed a mask, the VLM
                    says nothing is there.
  boundary_drift  - mask is otherwise valid+good but fails the LiDAR consistency
                    check, so one negative signal cannot delete it.
  depth_support   - mask is rejected primarily because of LiDAR consistency failure
                    (sky/void leak), rather than retained on a single negative.
  human_review    - mask routed to human review from agent disagreement, including
                    the bbox=invalid/quality=good downgrade rule described in
                    Section III-D of the paper.
  discovery       - Swin-driven discovery objects confirmed by the VLM, weighted
                    toward the safety-critical human class (cyclist/pedestrian).

Usage:
    python select_qualitative_frames.py --vlm llava_34b --topk 16
    python select_qualitative_frames.py --vlm llava_34b --category boundary_drift --topk 24
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import config
from core.mask_extractor import extract_proposals

HUMAN_CLASSES = {"cyclist", "pedestrian"}
CATEGORIES = ["hallucination", "boundary_drift", "depth_support", "human_review", "discovery"]

CROP_PAD = 30
CROP_MIN_SIDE = 160
THUMB_SIZE = 260
GRID_COLS = 4
CAPTION_H = 46


@dataclass
class Candidate:
    category: str
    frame_id: str
    mask_id: Optional[int]
    class_name: str
    bbox: tuple
    score: float
    label: str


def iter_results(results_dir: Path):
    for f in sorted(results_dir.glob("frame_*.json")):
        if f.stem == "summary":
            continue
        d = json.loads(f.read_text())
        if "masks" not in d:
            continue
        yield d


def find_candidates(results_dir: Path) -> dict[str, list[Candidate]]:
    buckets: dict[str, list[Candidate]] = {c: [] for c in CATEGORIES}

    for d in iter_results(results_dir):
        fid = d["frame_id"]

        for m in d["masks"]:
            a = m["agents"]
            bbox = tuple(m["bbox"])
            px = m["pixel_count"]
            cls = m["class_name"]

            if a["bbox"] == "background":
                buckets["hallucination"].append(
                    Candidate("hallucination", fid, m["mask_id"], cls, bbox, px,
                               f"{cls}: BBox agent -> background"))

            # Selected from the signals rather than the decision label, so this
            # works on runs made before and after the CorrectionAgent removal
            # (archived runs label some of these "refine", new ones never do).
            if (a["bbox"] == "valid" and a.get("quality") == "good"
                    and a["consistency"] == "fail"):
                buckets["boundary_drift"].append(
                    Candidate("boundary_drift", fid, m["mask_id"], cls, bbox, px,
                               f"{cls}: valid+good, consistency fail -> retained"))

            if a["consistency"] == "fail" and m["triage"] == "reject":
                buckets["depth_support"].append(
                    Candidate("depth_support", fid, m["mask_id"], cls, bbox, px,
                               f"{cls}: LiDAR support fail -> reject"))

            if m["triage"] == "human_review":
                if a["bbox"] == "invalid" and a["quality"] == "good":
                    label = f"{cls}: bbox=invalid, quality=good (downgrade rule)"
                    weight = 2.0  # explicitly called out in the paper's triage rule
                elif a["bbox"] == "valid" and a["quality"] == "bad":
                    label = f"{cls}: bbox=valid, quality=bad"
                    weight = 1.0
                else:
                    label = f"{cls}: bbox={a['bbox']}, quality={a['quality']}, consistency={a['consistency']}"
                    weight = 1.0
                human_bonus = 2.0 if cls in HUMAN_CLASSES else 1.0
                buckets["human_review"].append(
                    Candidate("human_review", fid, m["mask_id"], cls, bbox,
                               weight * human_bonus * px, label))

        for disc in d.get("discovered", []):
            if not disc["confirmed"]:
                continue
            cls = disc["class_name"]
            weight = 3.0 if cls in HUMAN_CLASSES else 1.0
            buckets["discovery"].append(
                Candidate("discovery", fid, None, cls, tuple(disc["bbox_orig"]),
                           weight * disc["pixel_count_384"], f"discovered {cls}"))

    return buckets


def _diversify(cands: list[Candidate], topk: int) -> list[Candidate]:
    """Keep at most one candidate per frame, then round-robin across classes
    so one dominant class (e.g. vehicle) doesn't crowd out the rest."""
    best_per_frame: dict[str, Candidate] = {}
    for c in cands:
        cur = best_per_frame.get(c.frame_id)
        if cur is None or c.score > cur.score:
            best_per_frame[c.frame_id] = c

    by_class: dict[str, list[Candidate]] = {}
    for c in best_per_frame.values():
        by_class.setdefault(c.class_name, []).append(c)
    for lst in by_class.values():
        lst.sort(key=lambda c: c.score, reverse=True)

    ordered_classes = sorted(by_class, key=lambda k: -len(by_class[k]))
    picked, idx = [], {k: 0 for k in ordered_classes}
    while len(picked) < topk and any(idx[k] < len(by_class[k]) for k in ordered_classes):
        for k in ordered_classes:
            if idx[k] < len(by_class[k]):
                picked.append(by_class[k][idx[k]])
                idx[k] += 1
                if len(picked) >= topk:
                    break
    return picked


def render_crop(cand: Candidate, camera_dir: Path) -> Optional[Image.Image]:
    cam_path = camera_dir / f"{cand.frame_id}.png"
    if not cam_path.exists():
        return None
    img = cv2.imread(str(cam_path))
    if img is None:
        return None
    H, W = img.shape[:2]

    canvas = img.copy()
    if cand.mask_id is not None:
        # Re-extract SAM proposals to get the exact pixel mask for this mask_id.
        ann_path = config.ANNOTATION_SAM_DIR / f"{cand.frame_id}.png"
        if ann_path.exists():
            proposals = extract_proposals(ann_path, cand.frame_id)
            match = next((p for p in proposals if p.mask_id == cand.mask_id), None)
            if match is not None:
                overlay = canvas.copy()
                overlay[match.pixel_mask] = (0, 0, 255)
                canvas = cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0)
    else:
        # Discovery candidate: draw the confirmed bbox.
        x1, y1, x2, y2 = cand.bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 165, 255), 2)

    x1, y1, x2, y2 = cand.bbox
    cx1 = max(0, x1 - CROP_PAD)
    cy1 = max(0, y1 - CROP_PAD)
    cx2 = min(W, x2 + CROP_PAD)
    cy2 = min(H, y2 + CROP_PAD)
    crop = canvas[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < CROP_MIN_SIDE or w < CROP_MIN_SIDE:
        scale = max(CROP_MIN_SIDE / max(h, 1), CROP_MIN_SIDE / max(w, 1))
        crop = cv2.resize(crop, (max(int(w * scale), 1), max(int(h * scale), 1)),
                           interpolation=cv2.INTER_CUBIC)

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def build_contact_sheet(cands: list[Candidate], camera_dir: Path, out_path: Path) -> int:
    tiles = []
    for c in cands:
        crop = render_crop(c, camera_dir)
        if crop is None:
            continue
        # Fit into a square thumbnail without distortion.
        crop.thumbnail((THUMB_SIZE, THUMB_SIZE))
        tile = Image.new("RGB", (THUMB_SIZE, THUMB_SIZE + CAPTION_H), (30, 30, 30))
        ox = (THUMB_SIZE - crop.width) // 2
        oy = (THUMB_SIZE - crop.height) // 2
        tile.paste(crop, (ox, oy))
        draw = ImageDraw.Draw(tile)
        caption = f"{c.frame_id}#{c.mask_id if c.mask_id is not None else 'disc'}"
        draw.text((4, THUMB_SIZE + 2), caption, fill=(255, 255, 0))
        draw.text((4, THUMB_SIZE + 20), c.label[:34], fill=(255, 255, 255))
        tiles.append(tile)

    if not tiles:
        return 0

    rows = (len(tiles) + GRID_COLS - 1) // GRID_COLS
    sheet = Image.new("RGB", (GRID_COLS * THUMB_SIZE, rows * (THUMB_SIZE + CAPTION_H)), (0, 0, 0))
    for i, tile in enumerate(tiles):
        r, col = divmod(i, GRID_COLS)
        sheet.paste(tile, (col * THUMB_SIZE, r * (THUMB_SIZE + CAPTION_H)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return len(tiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vlm", default="llava_34b",
                        help="Run tag under DATA_ROOT/vlm/<tag>/ (default: llava_34b)")
    parser.add_argument("--topk", type=int, default=16,
                        help="Candidates per category to render (default: 16)")
    parser.add_argument("--category", choices=CATEGORIES, default=None,
                        help="Only process this category (default: all)")
    parser.add_argument("--out-dir", type=Path,
                        default=Path(__file__).parent / "qualitative_candidates",
                        help="Directory to write contact sheets + manifest into")
    args = parser.parse_args()

    config.set_run_tag(args.vlm)
    results_dir = config.RESULTS_DIR
    camera_dir = config.CAMERA_DIR
    print(f"Scanning results in {results_dir} ...")

    buckets = find_candidates(results_dir)
    manifest = {}
    categories = [args.category] if args.category else CATEGORIES

    for cat in categories:
        cands = buckets[cat]
        print(f"[{cat}] {len(cands)} raw candidates")
        picked = _diversify(cands, args.topk)
        sheet_path = args.out_dir / f"{cat}.png"
        n = build_contact_sheet(picked, camera_dir, sheet_path)
        print(f"[{cat}] wrote {n} tiles -> {sheet_path}")
        manifest[cat] = [
            {"frame_id": c.frame_id, "mask_id": c.mask_id, "class": c.class_name,
             "bbox": c.bbox, "score": round(c.score, 2), "label": c.label,
             "full_frame_viz": str(config.VIS_DIR / f"{c.frame_id}.png")}
            for c in picked
        ]

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
