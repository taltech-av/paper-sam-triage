#!/usr/bin/env python3
"""
Main entry point: runs the VLM multi-agent triage pipeline over frames/frames.txt.

Design goal: VLM validates the vast majority of masks so we can measure its contribution.
Only extreme-geometry cases are rejected/accepted by metadata alone.

Pipeline:
  1. Metadata pre-filter  — reject malformed shapes; accept only very large masks (>5000px)
  2. Batch bbox per class  — 1 VLM call per class per frame, all remaining masks together
  3. Fast quality check    — 1 VLM call per bbox-valid mask (overlay crop only)

Usage:
    python process_frames.py
    python process_frames.py --resume --limit 20
    python process_frames.py --mock --limit 3
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from agents.quality_agent import QualityAgent
from core.bundle import build_bundle, Bundle
from core.mask_extractor import MaskProposal, extract_proposals
from core.triage import (
    TRIAGE_ACCEPT, TRIAGE_REJECT, TRIAGE_REVIEW, TriageResult, triage,
)
from output.annotation_writer import write_annotation
from output.results_writer import write_frame_result, write_summary
from vlm.client import VLMClient, encode_image_b64


# ── helpers ──────────────────────────────────────────────────────────────────

def load_frames() -> list[str]:
    if not config.FRAMES_FILE.exists():
        sys.exit(f"Frames file not found: {config.FRAMES_FILE}")
    lines = config.FRAMES_FILE.read_text().splitlines()
    return [Path(line.split("/")[-1]).stem for line in lines if line.strip()]


def check_ollama(model: str) -> None:
    import requests
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = [m["name"] for m in r.json().get("models", [])]
        if model not in names:
            print(f"  WARNING: model '{model}' not found. Available: {names}")
        else:
            print(f"  Ollama reachable — model '{model}' found")
    except Exception as e:
        sys.exit(f"  ERROR: cannot reach Ollama at {config.OLLAMA_URL} — {e}")


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ── Stage 1: metadata pre-filter ─────────────────────────────────────────────

def metadata_verdict(proposal: MaskProposal, bundle: Bundle) -> str | None:
    """
    Return 'accept' or 'reject' only when the answer is unambiguous from geometry.
    Everything else returns None → VLM verification.

    VLM must see the vast majority of masks so we can measure its contribution.

    Tiers:
      1. Extreme aspect ratio  → reject  (mask leaked into adjacent region — VLM can't help)
      2. Below class minimum   → reject  (sub-threshold noise)
      3. Very large (>5000px)  → accept  (SAM cannot produce 5K+ px on a ZOD bbox by mistake)
      4. Everything else       → VLM
    """
    px = proposal.pixel_count
    ar = bundle.metadata["aspect_ratio"]
    min_px = config.MIN_OBJECT_PIXELS.get(proposal.class_id, 15)

    if ar > config.AUTO_REJECT_MAX_ASPECT or ar < config.AUTO_REJECT_MIN_ASPECT:
        return TRIAGE_REJECT

    if px < min_px:
        return TRIAGE_REJECT

    if px >= config.AUTO_ACCEPT_LARGE_PIXELS:
        return TRIAGE_ACCEPT

    return None  # → VLM


# ── Stage 2: batch bbox per class ────────────────────────────────────────────

def _draw_numbered_context(
    camera_img: np.ndarray,
    proposals: list[MaskProposal],
) -> np.ndarray:
    """Draw numbered bbox rectangles for a single class onto camera_img."""
    ctx = camera_img.copy()
    class_id = proposals[0].class_id
    color = config.CLASS_COLORS_BGR.get(class_id, (255, 255, 255))
    for i, p in enumerate(proposals):
        x1, y1, x2, y2 = p.bbox
        cv2.rectangle(ctx, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ctx, str(i + 1), (x1, max(y1 - 3, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return ctx


def _parse_number_list(text: str, max_n: int) -> set[int]:
    """Extract 1-based box numbers from a free-form model response."""
    import re
    nums = {int(m) for m in re.findall(r"\b(\d+)\b", text)}
    return {n for n in nums if 1 <= n <= max_n}


def batch_bbox_check(
    vlm: VLMClient,
    proposals: list[MaskProposal],
    camera_img: np.ndarray,
    class_name: str,
) -> set[int]:
    """
    One VLM call for all proposals of one class.
    Returns a set of 0-based proposal indices that are VALID.
    """
    ctx = _draw_numbered_context(camera_img, proposals)
    img = _bgr_to_pil(ctx)

    prompt = (
        f"This is an autonomous driving scene. {len(proposals)} bounding boxes are "
        f"numbered 1–{len(proposals)} and each is supposed to contain a {class_name}.\n\n"
        f"List only the box numbers where a real {class_name} is visible — even if "
        f"small, distant, or partially occluded.\n"
        f"Reply with only the valid box numbers separated by spaces. "
        f"If none are valid reply with '0'."
    )

    try:
        raw = vlm.query([img], prompt)
        valid_1based = _parse_number_list(raw, len(proposals))
        # fallback: if model returned nothing parseable, accept all (safe default)
        if not valid_1based and "0" not in raw:
            return set(range(len(proposals)))
        return {n - 1 for n in valid_1based}  # convert to 0-based
    except Exception:
        return set(range(len(proposals)))  # on error, accept all


# ── Stage 3: fast quality check ──────────────────────────────────────────────

def fast_quality_check(quality_agent: QualityAgent, bundle: Bundle) -> str:
    """Quality check using only the overlay crop — no full frame."""
    # Temporarily override select_images to avoid sending context/depth
    original = quality_agent.select_images
    quality_agent.select_images = lambda b: [b.overlay_crop]
    result = quality_agent.run(bundle)
    quality_agent.select_images = original
    return result


# ── Frame processor ──────────────────────────────────────────────────────────

def process_frame(
    frame_id: str,
    vlm: VLMClient,
    quality_agent: QualityAgent,
    ann_out_dir: Path,
    results_out_dir: Path,
    bar: tqdm,
) -> dict:
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    cam_path = config.CAMERA_DIR / f"{frame_id}.png"
    lid_path = config.LIDAR_DIR / f"{frame_id}.png"

    if not ann_path.exists():
        return {}

    original_ann = np.array(Image.open(ann_path))
    camera_img = cv2.imread(str(cam_path))
    lidar_img = cv2.imread(str(lid_path)) if lid_path.exists() else np.zeros_like(camera_img)

    proposals = extract_proposals(ann_path, frame_id)
    bar.set_postfix_str(f"{frame_id}  {len(proposals)} masks")

    if not proposals:
        write_annotation(frame_id, original_ann, [], [], ann_out_dir)
        return write_frame_result(frame_id, [], [], results_out_dir)

    # Build bundles
    bundles = [build_bundle(p, camera_img, lidar_img) for p in proposals]

    # ── Stage 1: metadata pre-filter ────────────────────────────────────────
    verdicts: dict[int, str] = {}  # mask_id → decision
    needs_vlm: list[int] = []      # indices into proposals list

    for i, (p, b) in enumerate(zip(proposals, bundles)):
        v = metadata_verdict(p, b)
        if v is not None:
            verdicts[i] = v
        else:
            needs_vlm.append(i)

    # ── Stage 2: batch bbox per class ────────────────────────────────────────
    t_bbox = time.time()
    by_class: dict[int, list[int]] = defaultdict(list)
    for i in needs_vlm:
        by_class[proposals[i].class_id].append(i)

    bbox_valid: set[int] = set()
    for class_id, indices in by_class.items():
        class_proposals = [proposals[i] for i in indices]
        valid_local = batch_bbox_check(vlm, class_proposals, camera_img,
                                       config.CLASS_ID_TO_NAME[class_id])
        for local_idx in valid_local:
            bbox_valid.add(indices[local_idx])
        for local_idx in range(len(indices)):
            if local_idx not in valid_local:
                verdicts[indices[local_idx]] = TRIAGE_REJECT

    n_classes = len(by_class)
    bar.write(
        f"  {frame_id}  bbox batch: {n_classes} class calls  "
        f"valid={len(bbox_valid)}/{len(needs_vlm)}  ({time.time()-t_bbox:.1f}s)"
    )

    # ── Stage 3: fast quality check per bbox-valid mask ──────────────────────
    t_qual = time.time()
    accepted = rejected_q = 0
    for i in bbox_valid:
        q = fast_quality_check(quality_agent, bundles[i])
        if q == "good":
            verdicts[i] = TRIAGE_ACCEPT
            accepted += 1
        else:
            verdicts[i] = TRIAGE_REJECT
            rejected_q += 1

    bar.write(
        f"  {frame_id}  quality: accept={accepted} reject={rejected_q}  "
        f"({time.time()-t_qual:.1f}s)"
    )

    # ── Build TriageResult objects ────────────────────────────────────────────
    triage_results = []
    for i, p in enumerate(proposals):
        decision = verdicts.get(i, TRIAGE_REVIEW)
        triage_results.append(TriageResult(
            decision=decision,
            bbox_out="valid" if i in bbox_valid else ("invalid" if i in by_class.get(p.class_id, []) else None),
            quality_out="good" if decision == TRIAGE_ACCEPT else ("bad" if i in bbox_valid else None),
            failure_mode_out=None,
            correction_out=None,
            consistency_out=None,
        ))

    n_accept = sum(1 for r in triage_results if r.decision == TRIAGE_ACCEPT)
    n_reject = sum(1 for r in triage_results if r.decision == TRIAGE_REJECT)
    bar.write(f"  {frame_id}  done  accept={n_accept}  reject={n_reject}  "
              f"auto={len(proposals)-len(needs_vlm)}/{len(proposals)} skipped VLM")

    write_annotation(frame_id, original_ann, proposals, triage_results, ann_out_dir)
    return write_frame_result(frame_id, proposals, triage_results, results_out_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.mock:
        from vlm.mock_client import MockClient
        vlm = MockClient()
        print("VLM: mock client")
    else:
        print("VLM: Ollama — checking connection...")
        check_ollama(args.model)
        from vlm.ollama_client import OllamaClient
        vlm = OllamaClient(model=args.model)

    quality_agent = QualityAgent(vlm)

    frame_ids = load_frames()
    print(f"Frames: {len(frame_ids)} loaded")
    if args.limit:
        frame_ids = frame_ids[: args.limit]
        print(f"Frames: limited to {args.limit}")

    ann_out_dir = config.ANNOTATION_OUT_DIR
    results_out_dir = config.RESULTS_DIR
    ann_out_dir.mkdir(parents=True, exist_ok=True)
    results_out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume:
        before = len(frame_ids)
        frame_ids = [f for f in frame_ids
                     if not (results_out_dir / f"{f}.json").exists()]
        print(f"Resume: {len(frame_ids)} remaining ({before - len(frame_ids)} done)")

    frame_records = []
    with tqdm(frame_ids, desc="Frames", unit="frame") as bar:
        for frame_id in bar:
            record = process_frame(frame_id, vlm, quality_agent,
                                   ann_out_dir, results_out_dir, bar)
            if record:
                frame_records.append(record)

    write_summary(frame_records, results_out_dir)
    print(f"\nDone. {len(frame_records)} frames processed.")
    print(f"  Annotations → {ann_out_dir}")
    print(f"  Results     → {results_out_dir}")


if __name__ == "__main__":
    main()
