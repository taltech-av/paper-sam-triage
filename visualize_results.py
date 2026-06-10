#!/usr/bin/env python3
"""
Generate per-frame visualizations of VLM triage results.

For each processed frame, overlays accepted masks (green) and rejected masks (red)
on top of the camera image. Refine = yellow, human_review = blue.

Output: /run/media/tom/ml/zod_temp/vllm_results_visualizations/frame_XXXXXX.png
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config

VIS_DIR = config.DATA_ROOT / "vllm_results_visualizations"

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
    ann_sam_path  = config.ANNOTATION_SAM_DIR  / f"{frame_id}.png"
    ann_vllm_path = config.ANNOTATION_OUT_DIR  / f"{frame_id}.png"
    cam_path      = config.CAMERA_DIR          / f"{frame_id}.png"

    if not cam_path.exists() or not ann_sam_path.exists() or not ann_vllm_path.exists():
        return

    camera = cv2.imread(str(cam_path))
    sam    = np.array(Image.open(ann_sam_path))
    vllm   = np.array(Image.open(ann_vllm_path))

    canvas = camera.copy()

    # Pixels present in SAM but removed in VLLM → rejected
    rejected_mask = (sam > 0) & (vllm == 0)
    # Pixels kept in VLLM → accepted / refine / review (we don't distinguish here)
    accepted_mask = vllm > 0

    overlay_mask(canvas, accepted_mask, TRIAGE_COLORS["accept"])
    overlay_mask(canvas, rejected_mask, TRIAGE_COLORS["reject"])

    # Legend
    _draw_legend(canvas)

    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{frame_id}.png"), canvas)


def _draw_legend(img: np.ndarray) -> None:
    entries = [
        ("accepted", TRIAGE_COLORS["accept"]),
        ("rejected", TRIAGE_COLORS["reject"]),
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
