#!/usr/bin/env python3
"""
Regenerate exact connected-component pixel masks for stored discovery
candidates by re-running the Swin Quality Agent per frame.

Why: the live pipeline computes discovery candidates as connected components
of the Swin class map, but stores only their bounding boxes in the results
JSON. replay_triage previously painted candidates as filled bounding-box
rectangles, inflating foreground (2.9% → 7.9% of pixels in the no-VLM
variant) and corrupting every replayed discovery training set. This script
recomputes the components (Swin inference is deterministic and VLM-free) and
matches them to the stored candidates by bbox IoU, so replay can paint the
true object pixels instead of rectangles.

Output (shared by all VLM tags — candidates are Swin-proposed and identical):
    <DATA_ROOT>/vlm/discovery_masks/<frame_id>.png   384×384 uint8,
        pixel value = candidate index + 1 (components are disjoint)
    <DATA_ROOT>/vlm/discovery_masks/<frame_id>.json  index-aligned list of
        {"bbox_384": stored bbox, "swin_class": c, "match_iou": x}

Usage:
    python regenerate_discovery_masks.py            # all frames, GPU
    python regenerate_discovery_masks.py --limit 10 # smoke test
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from agents.discovery_agent import _connected_components
from agents.swin_quality_agent import SWIN_SIZE, SwingQualityAgent

MATCH_IOU_MIN = 0.3


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return inter / (area_a + area_b - inter)


def compute_components(swin_pred: np.ndarray, ann: np.ndarray) -> list[dict]:
    """All discovery components ≥ min_pixels, as the live DiscoveryAgent computes them."""
    ann_384 = cv2.resize(ann, (SWIN_SIZE, SWIN_SIZE), interpolation=cv2.INTER_NEAREST)
    comps = []
    for swin_cls in (1, 2, 3):
        uncovered = (swin_pred == swin_cls) & (ann_384 == 0)
        if not np.any(uncovered):
            continue
        labels, n = _connected_components(uncovered)
        for comp_id in range(1, n + 1):
            comp_mask = labels == comp_id
            if int(comp_mask.sum()) < config.DISCOVERY_MIN_PIXELS:
                continue
            ys, xs = np.where(comp_mask)
            comps.append({
                "swin_class": swin_cls,
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "mask": comp_mask,
            })
    return comps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="llava_34b",
                        help="Results tag providing the stored candidate list "
                             "(candidate sets are identical across tags)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    parser.add_argument("--out", type=Path, default=None,
                        help="write masks here instead of <DATA_ROOT>/vlm/discovery_masks. "
                             "Use this to verify a regeneration against the published set "
                             "without overwriting it.")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    results_dir = config.DATA_ROOT / "vlm" / args.tag / "results"
    out_dir = args.out or (config.DATA_ROOT / "vlm" / "discovery_masks")
    out_dir.mkdir(parents=True, exist_ok=True)

    result_files = sorted(results_dir.glob("frame_*.json"))
    if args.limit:
        result_files = result_files[: args.limit]
    print(f"{len(result_files)} frames, output → {out_dir}")

    agent = SwingQualityAgent(threshold=config.SWIN_AGREEMENT_THRESHOLD)

    n_cand = n_matched = 0
    iou_sum = 0.0
    for rf in tqdm(result_files, unit="frame"):
        data = json.loads(rf.read_text())
        frame_id = data["frame_id"]
        discovered = data.get("discovered", [])
        if not discovered:
            continue

        cam_path = config.CAMERA_DIR / f"{frame_id}.png"
        lid_path = config.LIDAR_DIR / f"{frame_id}.png"
        ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
        if not (cam_path.exists() and lid_path.exists() and ann_path.exists()):
            print(f"  {frame_id}: missing input, skipped")
            continue

        camera = cv2.imread(str(cam_path))
        lidar = cv2.imread(str(lid_path))
        ann = np.array(Image.open(ann_path))

        swin_pred = agent.predict_frame(camera, lidar)
        comps = compute_components(swin_pred, ann)

        index_png = np.zeros((SWIN_SIZE, SWIN_SIZE), dtype=np.uint8)
        sidecar = []
        used = set()
        for obj in discovered:
            n_cand += 1
            entry = {"bbox_384": list(obj["bbox_384"]),
                     "swin_class": obj["swin_class"], "match_iou": 0.0}
            best_iou, best_j = 0.0, None
            for j, c in enumerate(comps):
                if j in used or c["swin_class"] != obj["swin_class"]:
                    continue
                iou = bbox_iou(obj["bbox_384"], c["bbox"])
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j is not None and best_iou >= MATCH_IOU_MIN:
                used.add(best_j)
                idx = len(sidecar) + 1
                index_png[comps[best_j]["mask"]] = idx
                entry["match_iou"] = round(best_iou, 4)
                n_matched += 1
                iou_sum += best_iou
            sidecar.append(entry)

        Image.fromarray(index_png).save(out_dir / f"{frame_id}.png")
        (out_dir / f"{frame_id}.json").write_text(json.dumps(sidecar))

    pct = 100 * n_matched / n_cand if n_cand else 0.0
    mean_iou = iou_sum / n_matched if n_matched else 0.0
    print(f"\nCandidates: {n_cand}   matched: {n_matched} ({pct:.1f}%)   "
          f"mean bbox IoU of matches: {mean_iou:.3f}")
    if pct < 95:
        print("WARNING: match rate below 95% — local Swin predictions may "
              "diverge from the run that produced the stored candidates.")


if __name__ == "__main__":
    main()
