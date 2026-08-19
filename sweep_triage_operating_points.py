"""
Operating points of the triage rule, swept, scored against the human verdicts.

Answers two questions the paper's Table~II cannot, because that table reports one
setting per rule:

  * does moving a threshold change the operating point, or only the labels?
  * can any setting of the three-signal rule reach the corner the dense
    agreement threshold reaches on its own?

Everything is recomputed from the closed verification export, so the rule is
applied to the same masks the human judged. The main rule reproduces the shipped
`triage` row exactly (54.9 / 14.1 at the shipped thresholds), which is the check
that the reimplementation below is faithful.

Why not reuse `analyze_human_verification.py --variant`: its variant replay takes
a single scalar Swin threshold and, below it, falls back to the *stored* quality
verdict rather than calling the mask bad. That is fine for the variants it was
written for but it makes a swept threshold mean two things at once, so the
disjunctive row it prints (12.6 / 47.1) is not the same rule as the one described
in the paper (11.6 / 47.5 here).

    python sweep_triage_operating_points.py --export human_verified_output/verify_export.csv
"""

import argparse
import csv
from pathlib import Path

LARGE_CLASSES = {"vehicle", "sign"}      # class-aware dense-agreement thresholds
TAU_LARGE, TAU_SMALL = 0.30, 0.15
TAU_LIDAR = 0.10


def load(export: Path) -> list[dict]:
    """SAM proposals that carry a human verdict and both stored scores."""
    with export.open(newline="") as handle:
        return [row for row in csv.DictReader(handle)
                if row.get("mask_kind") == ""          # '' is a SAM proposal
                and row.get("verdict")
                and row.get("mask_swin_agreement")
                and row.get("mask_lidar_support")]


def signals(row: dict, tau_lidar: float, tau_large: float, tau_small: float, run: str):
    agreement = float(row["mask_swin_agreement"])
    support = float(row["mask_lidar_support"])
    bar = tau_large if row["class"] in LARGE_CLASSES else tau_small
    return (row[f"mask_bbox_agent_{run}"],
            "good" if agreement >= bar else "bad",
            "pass" if support >= tau_lidar else "fail")


def concordance(bbox: str, quality: str, consistency: str) -> bool:
    """The shipped rule: delete on two negative signals. Returns True to delete."""
    confirmed = bbox == "invalid" and consistency == "pass" and quality != "good"
    negatives = (2 * confirmed
                 + (bbox == "invalid" and not confirmed)
                 + 2 * (bbox == "background")
                 + (quality == "bad")
                 + (consistency == "fail"))
    return negatives >= 2


def single_negative(bbox: str, quality: str, consistency: str) -> bool:
    """Delete as soon as any one signal is negative."""
    return bbox in ("invalid", "background") or quality == "bad" or consistency == "fail"


def rates(rows, delete, **kwargs) -> tuple[float, float]:
    """(bad masks kept %, good masks deleted %) against the human verdicts."""
    kept_bad = lost_good = bad = good = 0
    for row in rows:
        gone = delete(*signals(row, **kwargs))
        if row["verdict"] == "correct":
            good += 1
            lost_good += gone
        else:
            bad += 1
            kept_bad += not gone
    return 100 * kept_bad / bad, 100 * lost_good / good


# ── Review targeting ─────────────────────────────────────────────────────────
# A reviewer who cannot look at every mask wants the wrong ones early. Ranking by
# the free dense-agreement score puts them there; ranking whole frames does not,
# because almost every frame holds at least one wrong mask.


def targeting(rows, run: str) -> None:
    """Errors found in the first fifth of a queue, and the queue needed for half."""
    from pathlib import Path as _P
    from collections import defaultdict

    bad = sum(1 for row in rows if row["verdict"] == "incorrect")

    def curve(order):
        found = half = 0
        at_fifth = None
        for i, hit in enumerate(order, 1):
            found += hit
            if not half and found >= bad / 2:
                half = 100 * i / len(order)
            if i == round(0.2 * len(order)):
                at_fifth = 100 * found / bad
        return at_fifth, half

    def by(key):
        return curve([row["verdict"] == "incorrect"
                      for row in sorted(rows, key=key)])

    agreement = lambda row: float(row["mask_swin_agreement"])
    combined = lambda row: (float(row["mask_swin_agreement"])
                            - 0.15 * (row["mask_consistency"] == "fail")
                            - 0.20 * (row[f"mask_bbox_agent_{run}"] == "invalid"))
    crop_only = lambda row: row[f"mask_bbox_agent_{run}"] != "invalid"

    frames = defaultdict(list)
    for row in rows:
        frames[_P(row["frame"]).stem].append(row)
    frame_order = sorted(frames.values(),
                         key=lambda masks: min(float(m["mask_swin_agreement"]) for m in masks))
    found = half = 0
    at_fifth = None
    for i, masks in enumerate(frame_order, 1):
        found += sum(1 for m in masks if m["verdict"] == "incorrect")
        if not half and found >= bad / 2:
            half = 100 * i / len(frame_order)
        if i == round(0.2 * len(frame_order)):
            at_fifth = 100 * found / bad

    print("\nreview targeting        found in first 20%   queue needed for half")
    print(f"  random order          {20.0:14.1f}   {50.0:19.1f}")
    for name, key in (("dense agreement", agreement), ("  + crop verdict", combined),
                      ("crop verdict only", crop_only)):
        a, h = by(key)
        print(f"  {name:20}{a:14.1f}   {h:19.1f}")
    print(f"  whole frames         {at_fifth:15.1f}   {half:19.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path,
                        default=Path("human_verified_output/verify_export.csv"))
    parser.add_argument("--run", default="llava_34b",
                        help="which stored BBox verdict to replay")
    args = parser.parse_args()

    rows = load(args.export)
    fixed = dict(tau_large=TAU_LARGE, tau_small=TAU_SMALL, run=args.run)
    print(f"{len(rows):,} judged SAM proposals, BBox verdicts from {args.run}\n")

    print("three-signal rule, LiDAR threshold swept")
    print("  tau     bad kept %   good deleted %")
    for tau in (0.05, 0.10, 0.20, 0.30, 0.50, 0.70):
        bad, good = rates(rows, concordance, tau_lidar=tau, **fixed)
        mark = "   <- shipped" if tau == TAU_LIDAR else ""
        print(f"  {tau:4.2f}    {bad:8.1f}   {good:12.1f}{mark}")

    bad, good = rates(rows, single_negative, tau_lidar=TAU_LIDAR, **fixed)
    print(f"\nany one negative signal deletes:  {bad:.1f} / {good:.1f}")

    print("\ndense agreement threshold alone (no VLM, no LiDAR)")
    print("  thr         bad kept %   good deleted %")
    for scale in (0.20, 0.30, 0.40, 0.60):
        b, g = rates(rows, lambda _b, q, _c: q == "bad",
                     tau_lidar=TAU_LIDAR, tau_large=scale, tau_small=scale / 2,
                     run=args.run)
        print(f"  {scale:4.2f}/{scale / 2:4.2f}    {b:8.1f}   {g:12.1f}")

    targeting(rows, args.run)


if __name__ == "__main__":
    main()
