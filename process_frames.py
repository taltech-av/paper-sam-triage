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
from agents.base import AgentOutcome
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
from core.triage import TRIAGE_REJECT as _TRIAGE_REJECT
from output.results_writer import write_frame_result, write_summary
from visualize_results import visualize_frame
from vlm.client import VLMClient
from vlm.health import EXIT_HEALTH_ABORT, VLMHealthError, VLMHealthMonitor, probe


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


def build_agents(vlm: VLMClient, swin_agent=None, monitor=None) -> dict:
    return {
        "bbox": BBoxAgent(vlm, monitor),
        "quality": swin_agent if swin_agent is not None else QualityAgent(vlm, monitor),
        "consistency": ConsistencyAgent(),       # deterministic, no VLM
        "correction": CorrectionAgent(vlm, monitor),
        "failure_mode": FailureModeAgent(vlm, monitor),   # diagnostic only
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

def _timed(fn) -> tuple:
    """Call fn(), return (result, elapsed_seconds)."""
    t = time.perf_counter()
    out = fn()
    return out, round(time.perf_counter() - t, 4)


def _outcome(agent, bundle: Bundle) -> AgentOutcome:
    """
    AgentOutcome for any agent, VLM-backed or deterministic.

    ConsistencyAgent and SwingQualityAgent compute their verdicts without a
    model call, so they have no run_outcome() and can never parse-fail.
    """
    run_outcome = getattr(agent, "run_outcome", None)
    if run_outcome is not None:
        return run_outcome(bundle)
    return AgentOutcome(agent.run(bundle))


def triage_mask(agents: dict, bundle: Bundle, diagnose: bool,
                swin_score: float | None = None) -> TriageResult:
    """Run the full agent cascade on one mask — all agents always called."""
    elapsed: dict[str, float] = {}
    vlm_calls = 0
    failures: dict[str, dict] = {}

    def verdict(name: str, outcome: AgentOutcome) -> str:
        """Unwrap an outcome, recording the substitution when there was one."""
        if outcome.parse_failed:
            failures[name] = {"degenerate": outcome.degenerate, "raw": outcome.raw_sample}
        return outcome.value

    consistency_out, elapsed["consistency"] = _timed(
        lambda: verdict("consistency", _outcome(agents["consistency"], bundle)))

    from agents.swin_quality_agent import SwingQualityAgent
    swin_bypass = False
    if isinstance(agents["quality"], SwingQualityAgent) and swin_score is not None:
        cls_id = bundle.metadata.get("class_id")
        swin_bypass = swin_score >= config.swin_skip_threshold(cls_id)

    bbox_out, elapsed["bbox"] = _timed(
        lambda: verdict("bbox", _outcome(agents["bbox"], bundle)))
    vlm_calls += 1

    quality_out, elapsed["quality"] = _timed(
        lambda: verdict("quality", _outcome(agents["quality"], bundle)))

    correction_out = None
    if bbox_out == "valid" and quality_out == "good" and consistency_out == "fail":
        correction_out, elapsed["correction"] = _timed(
            lambda: verdict("correction", _outcome(agents["correction"], bundle)))
        vlm_calls += 1

    result = triage(bbox_out, quality_out, None, correction_out, consistency_out)
    result.swin_score = swin_score
    result.swin_bypass = swin_bypass
    result.agent_elapsed = elapsed
    result.vlm_calls = vlm_calls

    if diagnose and result.decision in (TRIAGE_REJECT, TRIAGE_REVIEW):
        result.failure_mode_out, elapsed["failure_mode"] = _timed(
            lambda: verdict("failure_mode", _outcome(agents["failure_mode"], bundle)))
        vlm_calls += 1

    result.parse_failed = failures or None
    return result


# ── Variant annotation writers ───────────────────────────────────────────────

def _compute_variant_decisions(
    proposals: list, triage_results: list
) -> dict[str, dict[int, str]]:
    """Derive per-mask decisions for the training-data variants from in-memory results.

    Variants produced — one per row of the downstream ablation, no more. Each
    adds exactly one pipeline stage to the one before it, so a change in
    downstream score is attributable to that stage:

        raw_sam   — no triage, accept everything (true SAM baseline)
        swin_only — Swin agreement threshold only, per-class τ_q
        triage    — full concordance triage, no discovery

    `annotation_full` (triage + discovery) is written separately by
    write_annotation, completing the ladder.

    Anything else — `vlm_only`, alternative triage rules, discovery on a
    different base — is reachable offline through replay_triage.py from the
    stored results, so it costs nothing to leave out of every run. Writing a
    variant here means writing it for every frame of every run forever.
    """
    from core.triage import triage as _triage
    raw_sam, swin_only, triage_no_disc = {}, {}, {}
    for proposal, result in zip(proposals, triage_results):
        mid = proposal.mask_id
        cls_id = proposal.class_id

        raw_sam[mid] = "accept"

        swin_only[mid] = (
            _TRIAGE_REJECT
            if (result.swin_score is not None
                and result.swin_score < config.swin_quality_threshold(cls_id))
            else "accept"
        )

        triage_no_disc[mid] = result.decision

    return {"raw_sam": raw_sam, "swin_only": swin_only, "triage": triage_no_disc}


def _write_variant(
    frame_id: str, original_ann: np.ndarray, proposals: list,
    decisions: dict[int, str], out_dir: Path,
    discovered=None,
) -> None:
    refined = original_ann.copy()
    for p in proposals:
        if decisions.get(p.mask_id) == _TRIAGE_REJECT:
            refined[p.pixel_mask] = 0
    if discovered:
        H, W = refined.shape[:2]
        for disc in discovered:
            if not disc.confirmed:
                continue
            mask_orig = cv2.resize(
                disc.pixel_mask_384.astype(np.uint8),
                (W, H), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            write_pixels = mask_orig & (refined == 0)
            refined[write_pixels] = disc.class_id
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(refined.astype(np.uint8)).save(out_dir / f"{frame_id}.png")


# ── Frame processor ──────────────────────────────────────────────────────────

def process_frame(
    frame_id: str,
    vlm: VLMClient,
    ann_out_dir: Path,
    results_out_dir: Path,
    diagnose: bool,
    swin_agent=None,
    discovery_agent=None,
    monitor: VLMHealthMonitor | None = None,
) -> dict:
    # Build per-frame agents so BBoxAgent._expected_class is thread-local.
    # The monitor is deliberately shared — degradation is a property of the
    # server, so every worker has to count into the same window.
    agents = build_agents(vlm, swin_agent=swin_agent, monitor=monitor)

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
        # All variants are identical to annotation_full when there are no masks to triage
        for variant in ("raw_sam", "swin_only", "triage"):
            _write_variant(frame_id, original_ann, [],
                           {}, config.variant_dir(variant, ann_out_dir.parent))
        return write_frame_result(frame_id, [], [], results_out_dir)

    import datetime
    frame_start = time.perf_counter()
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    triage_results: list[TriageResult] = []
    mask_elapsed: list[float] = []
    n_auto = 0

    swin_pred = None
    swin_elapsed = 0.0
    if swin_agent is not None:
        t_swin = time.perf_counter()
        swin_pred = swin_agent.predict_frame(camera_img, lidar_img)
        swin_elapsed = round(time.perf_counter() - t_swin, 4)

    n_masks = len(proposals)
    for i, proposal in enumerate(proposals):
        cls = config.CLASS_ID_TO_NAME.get(proposal.class_id, str(proposal.class_id))
        tqdm.write(f"  {frame_id}  mask {i+1}/{n_masks}  {cls}  px={proposal.pixel_count}")

        bundle = build_bundle(proposal, camera_img, lidar_img)
        bundle.swin_pred = swin_pred

        swin_score = swin_agent.agreement(bundle) if swin_agent is not None else None
        lidar_support = bundle.metadata.get("lidar_support_ratio")

        t_mask = time.perf_counter()
        verdict = metadata_verdict(proposal, bundle)
        if verdict is not None:
            n_auto += 1
            triage_results.append(TriageResult(
                decision=verdict,
                bbox_out=None, quality_out=None, failure_mode_out=None,
                correction_out=None, consistency_out=None,
                swin_score=swin_score, lidar_support=lidar_support,
            ))
            mask_elapsed.append(round(time.perf_counter() - t_mask, 4))
            continue

        result = triage_mask(agents, bundle, diagnose, swin_score=swin_score)
        result.lidar_support = lidar_support
        triage_results.append(result)
        mask_elapsed.append(round(time.perf_counter() - t_mask, 4))

    triage_elapsed = round(time.perf_counter() - frame_start, 4)
    counts = {d: sum(1 for r in triage_results if r.decision == d)
              for d in (TRIAGE_ACCEPT, "refine", TRIAGE_REJECT, TRIAGE_REVIEW)}
    tqdm.write(
        f"  {frame_id}  accept={counts['accept']} refine={counts['refine']} "
        f"reject={counts['reject']} review={counts['human_review']}  "
        f"auto={n_auto}/{len(proposals)}  ({triage_elapsed:.1f}s)"
    )

    discovered = []
    discovery_elapsed = 0.0
    if discovery_agent is not None and swin_pred is not None:
        t_disc = time.perf_counter()
        try:
            discovered = discovery_agent.run(swin_pred, original_ann, camera_img)
            if discovered:
                n_confirmed = sum(1 for d in discovered if d.confirmed)
                tqdm.write(f"  {frame_id}  discovery: {n_confirmed}/{len(discovered)} confirmed")
        except VLMHealthError:
            # A degraded server is not a per-frame discovery failure — skipping
            # it here would write the frame with every candidate unconfirmed.
            raise
        except Exception as e:
            tqdm.write(f"  {frame_id}  discovery ERROR (skipped): {e}")
        discovery_elapsed = round(time.perf_counter() - t_disc, 4)

    total_vlm_calls = sum(r.vlm_calls for r in triage_results)
    n_bypass = sum(1 for r in triage_results if r.swin_bypass)
    run_info = {
        "model": vlm.model if hasattr(vlm, "model") else str(vlm),
        "timestamp": timestamp,
        "elapsed_seconds": round(time.perf_counter() - frame_start, 2),
        "triage_elapsed_seconds": triage_elapsed,
        "swin_elapsed_seconds": swin_elapsed,
        "discovery_elapsed_seconds": discovery_elapsed,
        "n_masks": len(proposals),
        "n_auto_rejected": n_auto,
        "n_vlm_calls": total_vlm_calls,
        "n_swin_bypass": n_bypass,
        "workers": config.WORKERS,
        # Per-frame degradation counts, so a corrupted stretch is visible in the
        # results themselves rather than only in the run's stdout.
        "n_parse_failed": sum(1 for r in triage_results if r.degraded),
        "n_discovery_parse_failed": sum(1 for d in discovered if d.parse_failed),
        **({"vlm_health": monitor.snapshot().as_dict()} if monitor else {}),
    }

    write_annotation(frame_id, original_ann, proposals, triage_results, ann_out_dir, discovered)

    # Write all training-data variants in one pass while pixel masks are in memory
    variants = _compute_variant_decisions(proposals, triage_results)
    for variant_name, decisions in variants.items():
        _write_variant(frame_id, original_ann, proposals, decisions,
                       config.variant_dir(variant_name, ann_out_dir.parent))

    # No *_discovery variants are written here. `annotation_full` above already
    # carries triage + discovery, which is the discovery row the ablation uses;
    # the discovery controls (`swin_only` with all Swin candidates, or with
    # VLM-confirmed ones) are replay_triage.py's job, since they are derivable
    # from the stored `discovered[]` without re-running anything.

    result = write_frame_result(
        frame_id, proposals, triage_results, results_out_dir, discovered,
        mask_elapsed=mask_elapsed, run_info=run_info,
    )
    visualize_frame(frame_id, config.VIS_DIR)

    # Backstop only: the monitor already raises from `record`, mid-frame, so a
    # saturated window never gets this far. This catches a swallowed abort, at
    # a point where the frame is safely on disk and the run stays resumable.
    if monitor is not None:
        monitor.check()
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--tag", default=None,
                        help="output namespace under vlm/<tag>/ (default: sanitized model name). "
                             "Use this to keep two HPC runs separate, e.g. --tag qwen72b vs --tag llama90b")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--frames", type=Path, default=None,
                        help="override frames CSV (default: config.FRAMES_FILE)")
    parser.add_argument("--diagnose", action="store_true",
                        help="run the failure-mode agent on rejected/review masks")
    parser.add_argument("--hpc", action="store_true",
                        help="use HPC data paths (totahv@base.hpc.taltech.ee)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel frame workers (default: config.WORKERS — 1 local, 4 HPC)")
    parser.add_argument("--no-swin", action="store_true",
                        help="use VLM quality agent instead of Swin segmentation model")
    parser.add_argument("--no-discovery", action="store_true",
                        help="disable Swin-based object discovery (missed object recovery)")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    # Namespace outputs by model name so two HPC runs never overwrite each other.
    tag = args.tag or args.model.replace(":", "_").replace("/", "_")
    config.set_run_tag(tag)
    print(f"Output tag: {tag}  →  {config.OUTPUT_ROOT}")

    if args.frames:
        config.FRAMES_FILE = args.frames

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

    swin_agent = None
    if not args.no_swin:
        from agents.swin_quality_agent import SwingQualityAgent
        print(f"Swin: loading model on {config.SWIN_DEVICE}...")
        swin_agent = SwingQualityAgent(
            threshold=config.SWIN_AGREEMENT_THRESHOLD,
            device=config.SWIN_DEVICE,
        )
        _ = swin_agent.model  # eagerly load so workers share one copy
        print("Swin: model ready")

    # One monitor for the whole run: the fault this guards against is a
    # property of the server, not of any frame.
    monitor = VLMHealthMonitor()
    if not args.mock:
        healthy, raw = probe(vlm)
        if not healthy:
            # Same exit status as an in-run abort: both mean "the server needs
            # reloading", and a batch loop should treat them identically.
            print(f"  ERROR: VLM canary failed before the run started — "
                  f"response was {raw!r}. Reload the model and retry.")
            sys.exit(EXIT_HEALTH_ABORT)
        print(f"  VLM canary OK ({raw.strip()[:40]!r})")

    discovery_agent = None
    if swin_agent is not None and not args.no_discovery:
        from agents.discovery_agent import DiscoveryAgent
        discovery_agent = DiscoveryAgent(vlm, monitor=monitor)
        print(f"Discovery: enabled (min_pixels={config.DISCOVERY_MIN_PIXELS})")

    frame_records = []
    aborted = None
    with tqdm(total=len(frame_ids), desc="Frames", unit="frame") as bar:
        workers = args.workers if args.workers is not None else config.WORKERS
        print(f"Workers: {workers}")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_frame, fid, vlm, ann_out_dir, results_out_dir,
                    args.diagnose, swin_agent, discovery_agent, monitor,
                ): fid
                for fid in frame_ids
            }
            for fut in as_completed(futures):
                try:
                    record = fut.result()
                    if record:
                        frame_records.append(record)
                except VLMHealthError as e:
                    # Not a per-frame error: every remaining frame would be
                    # written with substituted defaults. Stop the run instead.
                    aborted = str(e)
                    for pending in futures:
                        pending.cancel()
                    break
                except Exception as e:
                    tqdm.write(f"  ERROR {futures[fut]}: {e}")
                finally:
                    bar.update(1)

    health = monitor.snapshot()
    if health.degenerate:
        print(f"\n  WARNING: {health.degenerate}/{health.calls} VLM responses "
              f"({health.overall_rate:.1%}) carried no usable content. Masks and "
              f"candidates affected are flagged `parse_failed` in the results.")

    if aborted:
        # Deliberately no write_summary: the in-flight workers raised after
        # writing their frames, so frame_records is short of what is on disk and
        # summarising it would overwrite a good summary with an empty one.
        on_disk = len(list(results_out_dir.glob("*.json")))
        print(f"\nABORTED.\n  {aborted}\n"
              f"  {on_disk} frame results are on disk and intact — reload the model, "
              f"then rerun the same command with --resume to continue.")
        sys.exit(EXIT_HEALTH_ABORT)

    write_summary(frame_records, results_out_dir)
    print(f"\nDone. {len(frame_records)} frames processed.")
    print(f"  Annotations    → {ann_out_dir}  (full system: triage + discovery)")
    print(f"  Variants       → annotation_{{raw_sam,swin_only,triage}} (+ annotation_full)")
    print(f"  Results        → {results_out_dir}")
    print(f"  Visualizations → {config.VIS_DIR}")


if __name__ == "__main__":
    main()
