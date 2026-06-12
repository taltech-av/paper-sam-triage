#!/usr/bin/env python3
"""
Generate per-frame visualizations of VLM triage results.

Re-extracts the mask proposals from annotation_sam (deterministic, same order
as during processing) and colors each one by its triage decision from
vlm/results/frame_XXXXXX.json:

    accept → green, refine → yellow, reject → red, human_review → blue

Output: vlm/visualization/frame_XXXXXX.png under DATA_ROOT
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

import config
from core.mask_extractor import extract_proposals

VIS_DIR = config.VIS_DIR

# BGRA overlay colors (transparent)
TRIAGE_COLORS = {
    "accept":       (0,   200,  0,   100),   # green
    "reject":       (0,   0,    220, 110),   # red
    "refine":       (0,   200,  220, 100),   # yellow
    "human_review": (200, 80,   0,   90),    # blue
}


def overlay_mask(canvas_bgr: np.ndarray, mask: np.ndarray, bgra_color: tuple) -> None:
    """Alpha-blend a colored mask onto canvas in-place."""
    b, g, r, a = bgra_color
    alpha = a / 255.0
    roi = canvas_bgr[mask]
    canvas_bgr[mask] = (roi * (1 - alpha) + np.array([b, g, r]) * alpha).astype(np.uint8)


def visualize_frame(frame_id: str, out_dir: Path) -> None:
    ann_sam_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    result_path  = config.RESULTS_DIR        / f"{frame_id}.json"
    cam_path     = config.CAMERA_DIR         / f"{frame_id}.png"

    if not cam_path.exists() or not ann_sam_path.exists() or not result_path.exists():
        return

    camera = cv2.imread(str(cam_path))
    result = json.loads(result_path.read_text())
    decisions = {m["mask_id"]: m["triage"] for m in result["masks"]}

    proposals = extract_proposals(ann_sam_path, frame_id)

    canvas = camera.copy()
    for p in proposals:
        decision = decisions.get(p.mask_id)
        color = TRIAGE_COLORS.get(decision)
        if color is not None:
            overlay_mask(canvas, p.pixel_mask, color)

    _draw_legend(canvas)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{frame_id}.png"), canvas)


def _draw_legend(img: np.ndarray) -> None:
    entries = [
        ("accepted", TRIAGE_COLORS["accept"]),
        ("refine",   TRIAGE_COLORS["refine"]),
        ("rejected", TRIAGE_COLORS["reject"]),
        ("review",   TRIAGE_COLORS["human_review"]),
    ]
    x, y0, pad = 12, 12, 4
    h_entry = 22
    box_w = 16

    for i, (label, (b, g, r, _)) in enumerate(entries):
        y = y0 + i * (h_entry + pad)
        # colored box
        cv2.rectangle(img, (x, y), (x + box_w, y + h_entry), (b, g, r), -1)
        cv2.rectangle(img, (x, y), (x + box_w, y + h_entry), (0, 0, 0), 1)
        # label
        cv2.putText(img, label, (x + box_w + 6, y + h_entry - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (x + box_w + 6, y + h_entry - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Max frames to visualize (0=all)")
    args = parser.parse_args()

    result_files = sorted((config.RESULTS_DIR).glob("frame_*.json"))
    if not result_files:
        print(f"No results found in {config.RESULTS_DIR}")
        return

    frame_ids = [f.stem for f in result_files]
    if args.limit:
        frame_ids = frame_ids[: args.limit]

    print(f"Visualizing {len(frame_ids)} frames → {VIS_DIR}")
    for frame_id in tqdm(frame_ids):
        visualize_frame(frame_id, VIS_DIR)

    print(f"Done. {len(frame_ids)} images written to {VIS_DIR}")


if __name__ == "__main__":
    main()
