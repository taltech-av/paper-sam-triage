#!/usr/bin/env python3
"""
Main entry point: runs the VLM multi-agent triage pipeline over frames/frames.txt.

Every mask is judged on its own zoomed crop — never via numbered boxes on the
full frame, which a 7B VLM cannot resolve for 10–30 px objects. Rejection is
destructive (pixels zeroed), so it requires two concordant negative signals;
a single negative routes to human review instead.

Pipeline per mask:
  1. Metadata pre-filter        — reject extreme geometry; accept very large masks
  2. Consistency check (free)   — deterministic LiDAR support threshold
  3. BBox agent (VLM)           — object presence on the zoomed crop
  4. Quality agent (VLM)        — mask/object alignment on the overlay crop
     (skipped when bbox=invalid AND consistency=fail — already 2 negatives)
  5. Correction agent (VLM)     — only on the refine path (good + fail)
  6. core/triage.py             — concordance rules → accept/refine/reject/review

Usage:
    python process_frames.py
    python process_frames.py --resume --limit 20
    python process_frames.py --mock --limit 3
    python process_frames.py --diagnose          # run failure-mode agent on negatives
    python process_frames.py --workers 4         # parallel frame workers (default 4)
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from agents.bbox_agent import BBoxAgent
from agents.consistency_agent import ConsistencyAgent
from agents.correction_agent import CorrectionAgent
from agents.failure_mode_agent import FailureModeAgent
from agents.quality_agent import QualityAgent
from core.bundle import build_bundle, Bundle
from core.mask_extractor import MaskProposal, extract_proposals
from core.triage import (
    TRIAGE_ACCEPT, TRIAGE_REJECT, TRIAGE_REVIEW, TriageResult, triage,
)
from output.annotation_writer import write_annotation
from output.results_writer import write_frame_result, write_summary
from vlm.client import VLMClient


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
            sys.exit(f"  ERROR: model '{model}' not loaded in Ollama. Available: {names}\n"
                     f"  Run: ollama pull {model}")
        print(f"  Ollama reachable — model '{model}' found")
    except Exception as e:
        sys.exit(f"  ERROR: cannot reach Ollama at {config.OLLAMA_URL} — {e}")


def build_agents(vlm: VLMClient) -> dict:
    return {
        "bbox": BBoxAgent(vlm),
        "quality": QualityAgent(vlm),
        "consistency": ConsistencyAgent(),       # deterministic, no VLM
        "correction": CorrectionAgent(vlm),
        "failure_mode": FailureModeAgent(vlm),   # diagnostic only
    }


# ── Stage 1: metadata pre-filter ─────────────────────────────────────────────

def metadata_verdict(proposal: MaskProposal, bundle: Bundle) -> str | None:
    """
    Return 'accept' or 'reject' only when the answer is unambiguous from geometry.
    Everything else returns None → VLM verification.

    Tiers:
      1. Extreme aspect ratio  → reject  (mask leaked into adjacent region — VLM can't help)
      2. Below class minimum   → reject  (sub-threshold noise)
      3. Everything else       → VLM    (large masks included: on flagged frames the
                                         biggest components are often leaked noise blobs)
    """
    px = proposal.pixel_count
    ar = bundle.metadata["aspect_ratio"]
    min_px = config.MIN_OBJECT_PIXELS.get(proposal.class_id, 15)

    if ar > config.AUTO_REJECT_MAX_ASPECT or ar < config.AUTO_REJECT_MIN_ASPECT:
        return TRIAGE_REJECT

    if px < min_px:
        return TRIAGE_REJECT

    return None  # → VLM


# ── Stage 2: per-mask agent triage ───────────────────────────────────────────

def triage_mask(agents: dict, bundle: Bundle, diagnose: bool) -> TriageResult:
    """Run the agent cascade on one mask and combine via the triage rules."""
    consistency_out = agents["consistency"].run(bundle)
    bbox_out = agents["bbox"].run(bundle)

    # Early exit: rejection already decided — skip the quality call.
    # "background" = the VLM positively identified a non-object surface.
    if bbox_out == "background" or (bbox_out == "invalid" and consistency_out == "fail"):
        quality_out = None
    else:
        quality_out = agents["quality"].run(bundle)

    correction_out = None
    if bbox_out == "valid" and quality_out == "good" and consistency_out == "fail":
        correction_out = agents["correction"].run(bundle)

    result = triage(bbox_out, quality_out, None, correction_out, consistency_out)

    # Failure-mode diagnosis for non-accepted masks (analysis only)
    if diagnose and result.decision in (TRIAGE_REJECT, TRIAGE_REVIEW):
        result.failure_mode_out = agents["failure_mode"].run(bundle)

    return result


# ── Frame processor ──────────────────────────────────────────────────────────

def process_frame(
    frame_id: str,
    vlm: VLMClient,
    ann_out_dir: Path,
    results_out_dir: Path,
    diagnose: bool,
) -> dict:
    # Build per-frame agents so BBoxAgent._expected_class is thread-local.
    agents = build_agents(vlm)

    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    cam_path = config.CAMERA_DIR / f"{frame_id}.png"
    lid_path = config.LIDAR_DIR / f"{frame_id}.png"

    if not ann_path.exists():
        return {}

    original_ann = np.array(Image.open(ann_path))
    camera_img = cv2.imread(str(cam_path))
    lidar_img = cv2.imread(str(lid_path)) if lid_path.exists() else np.zeros_like(camera_img)

    proposals = extract_proposals(ann_path, frame_id)

    if not proposals:
        write_annotation(frame_id, original_ann, [], [], ann_out_dir)
        return write_frame_result(frame_id, [], [], results_out_dir)

    t0 = time.time()
    triage_results: list[TriageResult] = []
    n_auto = 0

    for proposal in proposals:
        bundle = build_bundle(proposal, camera_img, lidar_img)

        verdict = metadata_verdict(proposal, bundle)
        if verdict is not None:
            n_auto += 1
            triage_results.append(TriageResult(
                decision=verdict,
                bbox_out=None, quality_out=None, failure_mode_out=None,
                correction_out=None, consistency_out=None,
            ))
            continue

        triage_results.append(triage_mask(agents, bundle, diagnose))

    counts = {d: sum(1 for r in triage_results if r.decision == d)
              for d in (TRIAGE_ACCEPT, "refine", TRIAGE_REJECT, TRIAGE_REVIEW)}
    tqdm.write(
        f"  {frame_id}  accept={counts['accept']} refine={counts['refine']} "
        f"reject={counts['reject']} review={counts['human_review']}  "
        f"auto={n_auto}/{len(proposals)}  ({time.time()-t0:.1f}s)"
    )

    write_annotation(frame_id, original_ann, proposals, triage_results, ann_out_dir)
    return write_frame_result(frame_id, proposals, triage_results, results_out_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--diagnose", action="store_true",
                        help="run the failure-mode agent on rejected/review masks")
    parser.add_argument("--hpc", action="store_true",
                        help="use HPC data paths (totahv@base.hpc.taltech.ee)")
    parser.add_argument("--workers", type=int, default=4,
                        help="parallel frame workers (default 4; reduce if GPU OOMs)")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    if args.mock:
        from vlm.mock_client import MockClient
        vlm = MockClient()
        print("VLM: mock client")
    else:
        print("VLM: Ollama — checking connection...")
        check_ollama(args.model)
        from vlm.ollama_client import OllamaClient
        vlm = OllamaClient(model=args.model)

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
    with tqdm(total=len(frame_ids), desc="Frames", unit="frame") as bar:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    process_frame, fid, vlm, ann_out_dir, results_out_dir, args.diagnose
                ): fid
                for fid in frame_ids
            }
            for fut in as_completed(futures):
                try:
                    record = fut.result()
                    if record:
                        frame_records.append(record)
                except Exception as e:
                    tqdm.write(f"  ERROR {futures[fut]}: {e}")
                finally:
                    bar.update(1)

    write_summary(frame_records, results_out_dir)
    print(f"\nDone. {len(frame_records)} frames processed.")
    print(f"  Annotations → {ann_out_dir}")
    print(f"  Results     → {results_out_dir}")


if __name__ == "__main__":
    main()
