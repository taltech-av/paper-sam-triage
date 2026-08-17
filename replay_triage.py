#!/usr/bin/env python3
"""
Replay triage rules offline from saved result JSONs → write annotation PNGs.

Reads vlm/results/frame_*.json (raw agent outputs + scores saved during the
pipeline run), applies a triage variant, then writes refined annotation PNGs
to a named output folder — no GPU or Ollama required.

Each variant is a separate training dataset for CLFTv2 ablations:

    raw_sam      No triage — original SAM annotation unchanged (baseline B)
    swin_only    Swin agreement threshold only — no VLM signals
    vlm_only     BBox VLM + consistency — Swin quality ignored
    triage       Swin + VLM + consistency, no discovery (paper config D)
    with_bypass  Simulate bypass: skip VLM for masks where swin_bypass=True

For the full system (triage + discovery), use annotation_full/ written by
process_frames.py — it holds the exact connected-component discovery masks.

Usage:
    python replay_triage.py --variant raw_sam
    python replay_triage.py --variant swin_only
    python replay_triage.py --variant triage --out-suffix _v2
    python replay_triage.py --variant swin_only --swin-threshold 0.25
    python replay_triage.py --list-variants
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from core.triage import TRIAGE_REJECT, triage

# ── Discovery geometry gate ──────────────────────────────────────────────────
# A candidate component is formed by subtracting SAM's coverage from the Swin
# class map, so a candidate lying on an object the pipeline already segmented is
# only the rim left over. Whether it is such a rim is decidable from geometry
# alone — does its dilated ring contain a SAM mask of its own class — with no VLM
# call and no human label, which is why the gate applies to all 4,110 frames and
# not only to the verified ones.
#
# These two constants must match make_label_bundle's, because the same test was
# what the human-verification job used to classify candidates and what
# analyze_human_verification's geometry decomposition is measured on. A
# recomputation of that join from pixels alone agrees with the bundle's stored
# `touching_sam` on 2,047/2,047 candidates over 120 export frames.
NEIGHBOUR_RADIUS = 9
# Dataset class id → the coarse class the trainer folds it to. cyclist and
# pedestrian merge, so a `human` candidate abutting a cyclist mask is a rim, not
# a find: the labels do not distinguish them downstream.
COARSE_CLASS = {2: 1, 3: 2, 4: 3, 5: 3}

# Class-aware LiDAR support thresholds for the lidar_class_aware variant.
# Thin/sparse structures (bike frames, sign posts, LED matrices, near-field
# partial humans) reflect far fewer beams than solid vehicles, so the uniform
# τ=0.10 drives consistency=fail on real objects (paper Fig. limitations c/d).
LIDAR_TAU_DEFAULT = 0.10
LIDAR_TAU_BY_CLASS = {
    "vehicle": 0.10,
    "sign": 0.05,
    "cyclist": 0.05,
    "pedestrian": 0.05,
}


# ── Triage variants ───────────────────────────────────────────────────────────

def _triage_raw_sam(_: dict, **__) -> str:
    """Accept all masks — original SAM annotation unchanged. Baseline for downstream training."""
    return "accept"


def _triage_full(mask: dict, swin_q: float, **_) -> str:
    """Current pipeline: Swin quality + BBox VLM + consistency, all signals always collected."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    return triage(ag.get("bbox"), quality_out, ag.get("consistency")).decision


def _triage_swin_only(mask: dict, swin_q: float, **_) -> str:
    """Reject based solely on Swin agreement — no VLM signals used."""
    swin = mask.get("scores", {}).get("swin_agreement")
    if swin is None:
        return "accept"  # no Swin data (e.g. class not in PIPELINE_TO_SWIN) → keep
    return TRIAGE_REJECT if swin < swin_q else "accept"


def _triage_vlm_only(mask: dict, **_) -> str:
    """BBox VLM + consistency only — Swin quality signal ignored."""
    ag = mask["agents"]
    return triage(ag.get("bbox"), None, ag.get("consistency")).decision


def _triage_with_bypass(mask: dict, swin_q: float, **_) -> str:
    """Simulate old bypass: masks where swin_bypass=True skip BBox VLM (treat as valid+good).
    Useful for measuring the accuracy vs. efficiency trade-off of the bypass optimization."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")

    if scores.get("swin_bypass", False):
        bbox_out = "valid"
        quality_out = "good"
    else:
        bbox_out = ag.get("bbox")
        quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")

    return triage(bbox_out, quality_out, ag.get("consistency")).decision


def _triage_swin_protected(mask: dict, swin_q: float, **_) -> str:
    """Full triage with Swin protection: bbox=invalid+lidar=pass only counts as
    two negatives when Swin quality also disagrees. Reduces VLM over-rejection
    of small/thin objects (signs) where pixel-level Swin agreement is high."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    return triage(ag.get("bbox"), quality_out, ag.get("consistency")).decision


def _triage_disjunctive_reject(mask: dict, swin_q: float, **_) -> str:
    """Ablation: reject if ANY single negative signal is present (instead of requiring ≥2).
    Shows the cost of the concordance rule — disjunctive rejection is more aggressive
    and expected to over-reject valid masks, especially for rare classes."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    quality_out = "good" if (swin is not None and swin >= swin_q) else ag.get("quality")
    bbox = ag.get("bbox")
    if bbox in ("invalid", "background") or quality_out == "bad" or ag.get("consistency") == "fail":
        return TRIAGE_REJECT
    return "accept"


def _triage_uniform_tau(mask: dict, swin_q: float, **_) -> str:
    """Ablation: uniform Swin threshold τ_q=0.30 for all classes (no class-aware overrides).
    Cyclists and pedestrians have per-class τ_q=0.15 in the full pipeline because Swin's
    recall on small objects is much lower. This variant applies 0.30 uniformly, which
    over-rejects small-object masks — shows the value of class-aware thresholds."""
    ag = mask["agents"]
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    # Apply threshold uniformly; do NOT fall back to stored quality (which has per-class
    # thresholds baked in from the pipeline run).
    if swin is not None:
        quality_out = "good" if swin >= swin_q else "bad"
    else:
        quality_out = ag.get("quality")
    return triage(ag.get("bbox"), quality_out, ag.get("consistency")).decision


def _quality_out(mask: dict, swin_q: float) -> str | None:
    """Swin quality verdict: threshold on stored agreement, falling back to the
    stored per-class verdict (same logic as _triage_full)."""
    swin = mask.get("scores", {}).get("swin_agreement")
    if swin is not None:
        return "good" if swin >= swin_q else mask["agents"].get("quality")
    return mask["agents"].get("quality")


def _consistency_class_aware(mask: dict) -> str | None:
    """Recompute the LiDAR consistency verdict from the stored support fraction
    using class-aware thresholds instead of the uniform τ=0.10."""
    support = mask.get("scores", {}).get("lidar_support")
    if support is None:
        return mask["agents"].get("consistency")
    tau = LIDAR_TAU_BY_CLASS.get(mask.get("class_name"), LIDAR_TAU_DEFAULT)
    return "pass" if support >= tau else "fail"


def _protected_triage(bbox: str | None, quality: str | None,
                      consistency: str | None) -> str:
    """Concordance triage with the background verdict protected by Swin.

    Identical to core.triage.triage except bbox=background counts as two
    negatives only when Swin quality also disagrees; when Swin says good, it is
    downgraded to a single negative and the mask routes to human review instead
    of being rejected outright (paper Fig. limitations a/b: a single VLM
    misread of a dark or blurred crop deletes a real object unilaterally).
    """
    background_confirmed = bbox == "background" and quality != "good"
    bbox_invalid_confirmed = (
        bbox == "invalid" and consistency == "pass" and quality != "good"
    )
    negatives = sum([
        2 * background_confirmed,
        bbox == "background" and not background_confirmed,
        2 * bbox_invalid_confirmed,
        bbox == "invalid" and not bbox_invalid_confirmed,
        quality == "bad",
        consistency == "fail",
    ])
    if negatives >= 2:
        return TRIAGE_REJECT
    if bbox == "valid" and quality == "good" and consistency == "pass":
        return "accept"
    return "human_review"


def _triage_background_protected(mask: dict, swin_q: float, **_) -> str:
    """Limitation fix (a): background verdict no longer rejects unilaterally
    when Swin quality is good — downgraded to one negative, routed to review."""
    ag = mask["agents"]
    return _protected_triage(ag.get("bbox"), _quality_out(mask, swin_q),
                             ag.get("consistency"))


def _triage_lidar_class_aware(mask: dict, swin_q: float, **_) -> str:
    """Limitation fix (b): class-aware LiDAR support thresholds — sparse-object
    classes use τ=0.05 so thin structures no longer fail consistency and Swin
    simultaneously for the same physical reason (correlated failure)."""
    ag = mask["agents"]
    return triage(ag.get("bbox"), _quality_out(mask, swin_q),
                  _consistency_class_aware(mask)).decision


def _triage_limits_fixed(mask: dict, swin_q: float, **_) -> str:
    """Both limitation fixes combined: Swin-protected background verdict +
    class-aware LiDAR thresholds."""
    ag = mask["agents"]
    return _protected_triage(ag.get("bbox"), _quality_out(mask, swin_q),
                             _consistency_class_aware(mask))


VARIANTS = {
    "raw_sam":            (_triage_raw_sam,            "No triage — original SAM annotation unchanged (baseline)"),
    "swin_only":          (_triage_swin_only,          "Swin agreement threshold only — no VLM"),
    "vlm_only":           (_triage_vlm_only,           "BBox VLM + consistency — Swin quality ignored"),
    "triage":             (_triage_full,               "Swin + VLM + consistency triage, no discovery"),
    "swin_protected":     (_triage_swin_protected,     "Full triage + Swin protects valid masks from VLM false negatives"),
    "with_bypass":        (_triage_with_bypass,        "Simulate bypass: skip VLM for masks where swin_bypass=True"),
    "disjunctive_reject": (_triage_disjunctive_reject, "Ablation: any single negative signal rejects (vs. concordance ≥2)"),
    "uniform_tau":        (_triage_uniform_tau,        "Ablation: uniform τ_q=0.30 for all classes (no class-aware thresholds)"),
    "background_protected": (_triage_background_protected, "Limitation fix (a): Swin-protected background verdict"),
    "lidar_class_aware":  (_triage_lidar_class_aware,  "Limitation fix (b): class-aware LiDAR support thresholds"),
    "limits_fixed":       (_triage_limits_fixed,       "Both limitation fixes: (a) + (b)"),
}

# What `--variant all` writes: exactly the variants a fusion training config
# consumes. Every other entry in VARIANTS stays reachable by name for one-off
# analysis, but is no longer materialised as a training set by default.
#
# The ablation is a ladder — raw_sam -> swin_only -> triage -> full — where each
# step adds one pipeline stage, so a change in downstream score is attributable
# to that stage. The rule variations that used to be written here
# (`vlm_only`, `disjunctive_reject`, `uniform_tau`, `limits_fixed`,
# `with_bypass`, `swin_protected`, and the two halves of `limits_fixed`) all
# landed inside the same 39-46 mIoU band and re-confirmed the same result, at
# the cost of two GPU trainings each. Generate them by name if a specific
# question needs one.
PAPER_VARIANTS = ("raw_sam", "swin_only", "triage")


# ── Annotation writer (mirrors annotation_writer.py logic) ───────────────────

_BBOX_FALLBACKS = 0
_GATE_DROPPED = 0
_GATE_NO_MASK = 0
_GATE_PAINTED = 0


def _load_discovery_masks(frame_id: str):
    """Exact connected-component masks from regenerate_discovery_masks.py, or None."""
    masks_dir = config.DATA_ROOT / "vlm" / "discovery_masks"
    png = masks_dir / f"{frame_id}.png"
    sidecar = masks_dir / f"{frame_id}.json"
    if not (png.exists() and sidecar.exists()):
        return None
    index_map = np.array(Image.open(png))
    entries = json.loads(sidecar.read_text())
    # regenerate_discovery_masks.py assigns PNG index = sidecar position + 1
    by_bbox = {tuple(e["bbox_384"]): i + 1
               for i, e in enumerate(entries) if e["match_iou"] > 0}
    return index_map, by_bbox


def _window_around(bbox_384, shape, margin: int):
    """A camera-space crop containing a candidate's 384-space bbox, plus margin.

    Mirrors make_label_bundle.window_around: the ring test is local, so dilating
    a full-resolution frame per candidate would be wasted work.
    """
    height, width = shape
    x1, y1, x2, y2 = bbox_384
    return (slice(max(0, int(y1 * height / 384) - margin), min(height, int((y2 + 1) * height / 384) + margin)),
            slice(max(0, int(x1 * width / 384) - margin), min(width, int((x2 + 1) * width / 384) + margin)))


def _abuts_same_class(coarse_map: np.ndarray, index_map_full: np.ndarray,
                      idx: int, bbox_384, own_class: int) -> bool:
    """Is this candidate the rim of a SAM mask of its own class?

    `coarse_map` carries the *raw* SAM proposals, before triage deleted any of
    them. That is deliberate and it is the same reference the measurement used:
    the candidate pool was formed against SAM's full coverage, so whether a
    candidate is a rim is a fact about what SAM proposed there, not about what a
    triage rule later chose to keep. Testing against the post-triage labels would
    reclassify a rim as a find precisely when triage had just deleted the object
    it is a rim of.
    """
    rows_slice, cols_slice = _window_around(bbox_384, coarse_map.shape, NEIGHBOUR_RADIUS + 2)
    window_cls = coarse_map[rows_slice, cols_slice]
    pixels = (index_map_full[rows_slice, cols_slice] == idx) & (window_cls == 0)
    if not pixels.any():
        return False
    kernel = np.ones((NEIGHBOUR_RADIUS, NEIGHBOUR_RADIUS), np.uint8)
    ring = cv2.dilate(pixels.astype(np.uint8), kernel).astype(bool) & ~pixels
    return bool(own_class in (set(np.unique(window_cls[ring]).tolist()) - {0}))


def _add_discoveries(ann: np.ndarray, discovered: list, confirmed_only: bool,
                     frame_id: str | None = None,
                     coarse_map: np.ndarray | None = None) -> None:
    """Paint Swin-detected discovery objects onto ann in-place.

    Uses the exact connected-component pixel masks regenerated by
    regenerate_discovery_masks.py when available; falls back to the filled
    bounding-box approximation otherwise (counted and reported, since the
    rectangle fallback inflates foreground and corrupts training labels).
    Only background pixels (value 0) are overwritten so SAM proposals that
    survived triage are never displaced.

    `coarse_map` turns the geometry gate on: candidates that are the rim of a
    same-class SAM mask are dropped instead of painted. The gate needs the exact
    component masks — a bounding box has no rim to test — so a candidate that
    fell back to its rectangle is dropped rather than painted ungated, which
    keeps the variant's definition honest at the cost of a handful of regions.
    """
    global _BBOX_FALLBACKS, _GATE_DROPPED, _GATE_NO_MASK, _GATE_PAINTED
    exact = _load_discovery_masks(frame_id) if frame_id else None
    H, W = ann.shape[:2]
    if exact is not None:
        index_map_full = np.array(Image.fromarray(exact[0]).resize((W, H), Image.NEAREST))

    for obj in discovered:
        if confirmed_only and not obj.get("confirmed"):
            continue
        class_id = obj["class_id"]
        idx = exact[1].get(tuple(obj["bbox_384"])) if exact is not None else None
        if idx is not None:
            if coarse_map is not None and _abuts_same_class(
                    coarse_map, index_map_full, idx, obj["bbox_384"],
                    COARSE_CLASS.get(class_id, 0)):
                _GATE_DROPPED += 1
                continue
            if coarse_map is not None:
                _GATE_PAINTED += 1
            region_mask = (index_map_full == idx) & (ann == 0)
            ann[region_mask] = class_id
        elif coarse_map is not None:
            _GATE_NO_MASK += 1
        else:
            _BBOX_FALLBACKS += 1
            x1, y1, x2, y2 = obj["bbox_orig"]
            region = ann[y1:y2, x1:x2]
            region[region == 0] = class_id


def _write_annotation(
    frame_id: str,
    decisions: dict[int, str],
    out_dir: Path,
    discovered: list | None = None,
    discovery_confirmed_only: bool = True,
    discovery_geometry_gate: bool = False,
) -> None:
    ann_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
    if not ann_path.exists():
        return
    ann = np.array(Image.open(ann_path))

    from core.mask_extractor import extract_proposals
    proposals = extract_proposals(ann_path, frame_id)

    # Built from the proposals rather than from the PNG: a sub-threshold
    # component is not a proposal, was never shown to a labeler, and must not
    # count as a mask a candidate is the rim of.
    coarse_map = None
    if discovery_geometry_gate:
        coarse_map = np.zeros(ann.shape[:2], dtype=np.uint8)
        for p in proposals:
            coarse_map[p.pixel_mask] = COARSE_CLASS.get(p.class_id, 0)

    for p in proposals:
        if decisions.get(p.mask_id) == TRIAGE_REJECT:
            ann[p.pixel_mask] = 0

    if discovered:
        _add_discoveries(ann, discovered, confirmed_only=discovery_confirmed_only,
                         frame_id=frame_id, coarse_map=coarse_map)

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ann.astype(np.uint8)).save(out_dir / f"{frame_id}.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all", choices=list(VARIANTS) + ["all"],
                        help="Triage variant to apply, or 'all' to generate every variant (default: all)")
    parser.add_argument("--tag", default=None,
                        help="run tag to read results from (vlm/<tag>/results/). "
                             "Must match the --tag used during process_frames.py.")
    parser.add_argument("--out-suffix", default="",
                        help="Suffix appended to the output folder name, e.g. '_v2'")
    parser.add_argument("--swin-threshold", type=float, default=config.SWIN_AGREEMENT_THRESHOLD,
                        help=f"Swin quality threshold τ_q (default: {config.SWIN_AGREEMENT_THRESHOLD})")
    parser.add_argument("--swin-skip", type=float, default=config.SWIN_SKIP_BBOX_THRESHOLD,
                        help=f"Swin bypass threshold τ_skip (default: {config.SWIN_SKIP_BBOX_THRESHOLD})")
    parser.add_argument("--with-discovery", action="store_true",
                        help="Add VLM-confirmed discovery objects from stored results (mirrors annotation_swin_discovery)")
    parser.add_argument("--with-discovery-all", action="store_true",
                        help="Add ALL Swin-detected discovery candidates, skipping VLM confirmation "
                             "(ablation: isolates value of VLM confirmation in discovery)")
    parser.add_argument("--discovery-geometry-gate", action="store_true",
                        help="Paint only candidates that are NOT the rim of a same-class SAM mask. "
                             "Pure geometry — no VLM call, no human label — so it applies to every "
                             "frame. Appends '_standalone' to the output folder name.")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    parser.add_argument("--compare-to", default=None, choices=list(VARIANTS),
                        help="Also compute this baseline variant and print per-class "
                             "decision transitions (e.g. masks rescued from rejection)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip annotation PNG writing — print decision statistics only")
    parser.add_argument("--list-variants", action="store_true", help="List available variants and exit")
    args = parser.parse_args()

    if args.list_variants:
        for name, (_, desc) in VARIANTS.items():
            print(f"  {name:12s}  {desc}")
        return

    if args.with_discovery and args.with_discovery_all:
        parser.error("--with-discovery and --with-discovery-all are mutually exclusive")
    if args.discovery_geometry_gate and not (args.with_discovery or args.with_discovery_all):
        parser.error("--discovery-geometry-gate needs --with-discovery or --with-discovery-all")

    use_discovery = args.with_discovery or args.with_discovery_all
    discovery_confirmed_only = not args.with_discovery_all

    if args.hpc:
        config.use_hpc()

    if args.tag:
        config.set_run_tag(args.tag)

    result_files = sorted(config.RESULTS_DIR.glob("frame_*.json"))
    if not result_files:
        print(f"No results found in {config.RESULTS_DIR}")
        return

    if args.variant == "all":
        variants_to_run = [(n, VARIANTS[n]) for n in PAPER_VARIANTS]
    else:
        variants_to_run = [(args.variant, VARIANTS[args.variant])]

    disc_label = ""
    if args.with_discovery:
        disc_label = "_discovery"
    elif args.with_discovery_all:
        disc_label = "_discovery_noVLM"
    if args.discovery_geometry_gate:
        disc_label += "_standalone"

    for variant_name, (fn, desc) in variants_to_run:
        # A replayed variant is VLM-independent only if its triage rule reads no
        # VLM field *and* any discovery it adds is unfiltered (`--with-discovery-all`).
        # `--with-discovery` confirms candidates with the VLM, which makes even a
        # swin_only base model-specific.
        vlm_free = (variant_name in config.VLM_INDEPENDENT_VARIANTS
                    and not args.with_discovery)
        root = config.DATA_ROOT if vlm_free else config.OUTPUT_ROOT
        out_dir = root / f"annotation_{variant_name}{disc_label}{args.out_suffix}"
        print(f"\nVariant : {variant_name}{disc_label}  —  {desc}")
        print(f"τ_q     : {args.swin_threshold}   τ_skip: {args.swin_skip}")
        if use_discovery:
            mode = "VLM-confirmed only" if discovery_confirmed_only else "ALL Swin candidates (no VLM filter)"
            print(f"Discovery: {mode}")
        print(f"Output  : {out_dir}")

        totals: dict[str, int] = {}
        base_fn = VARIANTS[args.compare_to][0] if args.compare_to else None
        # (baseline_decision, variant_decision) → class_name → count
        transitions: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        transition_pixels: dict[tuple, int] = defaultdict(int)

        for rf in tqdm(result_files, unit="frame", desc=variant_name):
            data = json.loads(rf.read_text())
            frame_id = data["frame_id"]
            decisions: dict[int, str] = {}

            for mask in data["masks"]:
                decision = fn(mask, swin_q=args.swin_threshold, swin_skip=args.swin_skip)
                decisions[mask["mask_id"]] = decision
                totals[decision] = totals.get(decision, 0) + 1
                if base_fn is not None:
                    base = base_fn(mask, swin_q=args.swin_threshold, swin_skip=args.swin_skip)
                    if base != decision:
                        key = (base, decision)
                        transitions[key][mask["class_name"]] += 1
                        transition_pixels[key] += mask["pixel_count"]

            if not args.stats_only:
                discovered = data.get("discovered", []) if use_discovery else None
                _write_annotation(frame_id, decisions, out_dir,
                                  discovered=discovered,
                                  discovery_confirmed_only=discovery_confirmed_only,
                                  discovery_geometry_gate=args.discovery_geometry_gate)

        total = sum(totals.values())
        for k, v in sorted(totals.items()):
            if v:
                print(f"  {k:15s}: {v:4d}  ({100*v/total:.0f}%)")

        if transitions:
            print(f"\n  Decision changes vs '{args.compare_to}':")
            for (base, new), by_class in sorted(transitions.items()):
                n = sum(by_class.values())
                per_class = ", ".join(f"{c}: {v}" for c, v in sorted(by_class.items(), key=lambda kv: -kv[1]))
                print(f"    {base} → {new}: {n} masks, {transition_pixels[(base, new)]:,} px  ({per_class})")
        elif args.compare_to:
            print(f"\n  No decision changes vs '{args.compare_to}'.")

    if use_discovery:
        if _BBOX_FALLBACKS:
            print(f"\nWARNING: {_BBOX_FALLBACKS} discovery objects painted as bounding-box "
                  f"rectangles (no exact mask found — run regenerate_discovery_masks.py).")
        else:
            print("\nAll discovery objects painted with exact connected-component masks.")

    if args.discovery_geometry_gate:
        painted = _GATE_PAINTED
        seen = painted + _GATE_DROPPED
        print(f"\nGeometry gate: {_GATE_DROPPED:,} of {seen:,} candidates dropped as rims of a "
              f"same-class SAM mask ({100*_GATE_DROPPED/max(seen,1):.1f}%); {painted:,} painted.")
        if _GATE_NO_MASK:
            print(f"  {_GATE_NO_MASK:,} dropped for having no exact mask to test "
                  f"(a bounding box has no rim).")

    print(f"\nDone. {len(result_files)} frames × {len(variants_to_run)} variants.")


if __name__ == "__main__":
    main()
