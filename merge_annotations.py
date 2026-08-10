#!/usr/bin/env python3
"""
Two-VLM consensus annotations from stored results — no VLM inference needed.

Joins the per-mask agent outputs of two runs (e.g. llava_34b + qwen2.5vl_72b_v2,
same frames, same SAM masks), recomputes triage per model under the CURRENT
deterministic rule (stored `triage` fields may predate rule changes), applies
a consensus rule, and writes annotation PNGs for downstream CLFTv2 training.

Triage consensus rules:
    union         Reject only if BOTH models' triage rejects (high-precision
                  deletion — the concordance principle applied across models).
    intersection  Reject if EITHER model's triage rejects (high-recall).
    swin_only     VLM-independent Swin filtering (identical for both runs);
                  use together with --discovery both for "Swin + disc
                  (both-confirm)".

Discovery consensus (Swin proposes identical candidates for both runs; joined
on (frame_id, bbox_384)):
    both      add a candidate only if both VLMs confirmed it with the same class
    either    add if either VLM confirmed (class from LLaVA-side run if both)
    none      no discovery objects

Usage:
    python merge_annotations.py --tag-a llava_34b --tag-b qwen2.5vl_72b_v2 \\
        --triage swin_only --discovery both
    python merge_annotations.py --tag-a llava_34b --tag-b qwen2.5vl_72b_v2 \\
        --triage union --discovery both --stats-only
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config
from core.triage import TRIAGE_REJECT
from replay_triage import VARIANTS, _write_annotation

_triage_full = VARIANTS["triage"][0]
_triage_swin_only = VARIANTS["swin_only"][0]


def _load_results(tag: str) -> dict[str, dict]:
    results_dir = config.DATA_ROOT / "vlm" / tag / "results"
    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No results found in {results_dir}")
    return {rec["frame_id"]: rec for rec in (json.loads(f.read_text()) for f in files)}


def _consensus_decisions(masks_a: list[dict], masks_b: list[dict],
                         rule: str, swin_q: float) -> dict[int, str]:
    """Per-mask consensus decision. Non-rejected masks keep run A's decision
    (accept/review/refine all retain pixels, so only reject/keep matters for
    the annotation)."""
    by_id_b = {m["mask_id"]: m for m in masks_b}
    decisions: dict[int, str] = {}
    for ma in masks_a:
        mid = ma["mask_id"]
        mb = by_id_b.get(mid)
        if rule == "swin_only":
            decisions[mid] = _triage_swin_only(ma, swin_q=swin_q)
            continue
        da = _triage_full(ma, swin_q=swin_q)
        db = _triage_full(mb, swin_q=swin_q) if mb is not None else da
        if rule == "union":
            rejected = da == TRIAGE_REJECT and db == TRIAGE_REJECT
        else:  # intersection
            rejected = da == TRIAGE_REJECT or db == TRIAGE_REJECT
        if rejected:
            decisions[mid] = TRIAGE_REJECT
        elif da == TRIAGE_REJECT:
            # rejected by A only — cross-model disagreement, downgrade to review
            decisions[mid] = "human_review"
        else:
            decisions[mid] = da
    return decisions


def _consensus_discoveries(disc_a: list[dict], disc_b: list[dict],
                           mode: str, stats: dict) -> list[dict]:
    """Merged discovery list with `confirmed` set by the consensus mode."""
    by_bbox_b = {tuple(d["bbox_384"]): d for d in disc_b}
    merged: list[dict] = []
    for da in disc_a:
        db = by_bbox_b.get(tuple(da["bbox_384"]))
        conf_a = bool(da.get("confirmed"))
        conf_b = bool(db.get("confirmed")) if db is not None else False
        same_class = db is not None and da["class_id"] == db["class_id"]
        obj = dict(da)
        if mode == "both":
            obj["confirmed"] = conf_a and conf_b and same_class
            if conf_a and conf_b and not same_class:
                stats["disc_class_disagreement"] += 1
        else:  # either
            obj["confirmed"] = conf_a or conf_b
            if not conf_a and conf_b:
                obj["class_id"] = db["class_id"]
                obj["class_name"] = db["class_name"]
        if obj["confirmed"]:
            stats[f"disc_added_{obj['class_name']}"] += 1
        merged.append(obj)
    stats["disc_candidates"] += len(disc_a)
    stats["disc_confirmed"] += sum(1 for o in merged if o["confirmed"])
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag-a", default="llava_34b", help="First run tag")
    parser.add_argument("--tag-b", default="qwen2.5vl_72b_v2", help="Second run tag")
    parser.add_argument("--triage", default="union",
                        choices=["union", "intersection", "swin_only"],
                        help="Triage consensus rule (default: union)")
    parser.add_argument("--discovery", default="none",
                        choices=["both", "either", "none"],
                        help="Discovery consensus mode (default: none)")
    parser.add_argument("--swin-threshold", type=float, default=config.SWIN_AGREEMENT_THRESHOLD)
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip annotation PNG writing — print statistics only")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    runs_a = _load_results(args.tag_a)
    runs_b = _load_results(args.tag_b)
    frame_ids = sorted(set(runs_a) & set(runs_b))
    skipped = (len(runs_a) - len(frame_ids), len(runs_b) - len(frame_ids))
    if any(skipped):
        print(f"WARNING: {skipped[0]} frames only in {args.tag_a}, "
              f"{skipped[1]} only in {args.tag_b} — skipped")

    disc_label = f"_disc_{args.discovery}" if args.discovery != "none" else ""
    out_dir = (config.DATA_ROOT / "vlm" / f"merged_{args.tag_a}_{args.tag_b}"
               / f"annotation_{args.triage}{disc_label}")
    print(f"Triage    : {args.triage}   Discovery: {args.discovery}")
    print(f"Frames    : {len(frame_ids)} in common")
    print(f"Output    : {out_dir}")

    totals: dict[str, int] = defaultdict(int)
    stats: dict[str, int] = defaultdict(int)

    for frame_id in tqdm(frame_ids, unit="frame"):
        rec_a, rec_b = runs_a[frame_id], runs_b[frame_id]
        decisions = _consensus_decisions(rec_a["masks"], rec_b["masks"],
                                         args.triage, args.swin_threshold)
        for d in decisions.values():
            totals[d] += 1

        discovered = None
        if args.discovery != "none":
            discovered = _consensus_discoveries(
                rec_a.get("discovered", []), rec_b.get("discovered", []),
                args.discovery, stats)

        if not args.stats_only:
            _write_annotation(frame_id, decisions, out_dir,
                              discovered=discovered,
                              discovery_confirmed_only=True)

    total = sum(totals.values())
    print("\nTriage consensus outcomes:")
    for k, v in sorted(totals.items()):
        print(f"  {k:15s}: {v:6d}  ({100*v/total:.1f}%)")
    if args.discovery != "none":
        print("Discovery consensus:")
        n, c = stats["disc_candidates"], stats["disc_confirmed"]
        print(f"  candidates      : {n}")
        print(f"  added           : {c}  ({100*c/n:.1f}%)" if n else "  added           : 0")
        for k in sorted(stats):
            if k.startswith("disc_added_"):
                print(f"    {k.removeprefix('disc_added_'):13s}: {stats[k]}")
        if stats["disc_class_disagreement"]:
            print(f"  class disagreement (both confirmed, dropped): "
                  f"{stats['disc_class_disagreement']}")


if __name__ == "__main__":
    main()
