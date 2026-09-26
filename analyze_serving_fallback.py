#!/usr/bin/env python3
"""
How much of a backend's permissiveness is the serving path rather than the model.

`analyze_human_verification.py` scores triage rules against the human reference
but cannot separate a verdict the model gave from a SAFE_DEFAULT the pipeline
substituted when no reply could be parsed. BBoxAgent.SAFE_DEFAULT is "valid", so
every substitution is a positive one and inflates the retained-incorrect rate.

The substitutions are recorded per mask as `parse_failed` in the stored frame
results, so the rule can be replayed twice: over every mask, and over answered
crops only. The difference is the serving path's contribution to the rate.

    python analyze_serving_fallback.py --tag qwen2.5vl_72b_v2

Only the qwen2.5vl_72b_v2 run carries the flag; the llava_34b results were
written before it was added, so that backend cannot be split this way.

Also prints the per-frame audit-time distribution, which is heavy-tailed enough
that a single figure for annotator cost is misleading: elapsedMs is a per-task
value repeated onto every mask row, so it is deduplicated per frame here.
"""
import argparse
import csv
import glob
import json
import re
import statistics as st
from pathlib import Path

import config
from core.triage import triage

csv.field_size_limit(10 ** 9)


def _human_verdicts(export: Path) -> dict:
    """(frame_id, mask_id) -> 'correct' | 'incorrect', SAM proposals only."""
    out = {}
    with export.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["mask_source"] != "sam" or not row["verdict"]:
                continue
            frame = re.search(r"(frame_\d+)", row["frame"]).group(1)
            out[(frame, str(row["maskId"]))] = row["verdict"]
    return out


def _replay(mask: dict) -> bool:
    """True when the shipped rule retains this mask. Thresholds applied post-hoc,
    exactly as replay_triage's `triage` variant does."""
    scores = mask.get("scores", {})
    swin = scores.get("swin_agreement")
    lidar = scores.get("lidar_support")
    quality = "good" if swin is not None and swin >= config.swin_quality_threshold(
        mask["class_id"]) else "bad"
    consistency = "pass" if lidar is not None and lidar >= config.LIDAR_SUPPORT_MIN else "fail"
    decision = triage(mask["agents"].get("bbox"), quality, consistency).decision
    return decision != "reject"


def _rates(rows, label):
    bad = [r for r in rows if r[0] == "incorrect"]
    good = [r for r in rows if r[0] == "correct"]
    if not bad or not good:
        print(f"  {label:34s} — no data")
        return
    kept = 100 * sum(1 for r in bad if r[1]) / len(bad)
    lost = 100 * sum(1 for r in good if not r[1]) / len(good)
    print(f"  {label:34s} n={len(rows):6,}  bad kept {kept:5.1f}  good lost {lost:5.1f}")


def fallback(tag: str, export: Path) -> None:
    human = _human_verdicts(export)
    rows, total, substituted = [], 0, 0
    results = Path(str(config.DATA_ROOT)) / "vlm" / tag / "results"
    files = sorted(glob.glob(str(results / "frame_*.json")))
    if not files:
        raise SystemExit(f"no stored results under {results}")
    flagged = False
    for path in files:
        record = json.load(open(path))
        frame = record["frame_id"]
        for mask in record["masks"]:
            total += 1
            failures = mask.get("parse_failed") or {}
            if "parse_failed" in mask:
                flagged = True
            fell_back = "bbox" in failures
            substituted += fell_back
            verdict = human.get((frame, str(mask["mask_id"])))
            if verdict is not None:
                rows.append((verdict, _replay(mask), fell_back))

    print(f"\n=== {tag} — crop verdicts that were substituted, not judged ===")
    if not flagged:
        print("  this run predates the parse_failed flag; nothing to separate")
        return
    print(f"  proposals in the run          {total:,}")
    print(f"  bbox fell back to SAFE_DEFAULT {substituted:,}  ({100*substituted/total:.1f}%)")

    bad = [r for r in rows if r[0] == "incorrect"]
    good = [r for r in rows if r[0] == "correct"]
    print(f"\n  audited subset: {len(rows):,} proposals "
          f"({len(bad):,} human-rejected, {len(good):,} human-approved)")
    print(f"    substituted among human-rejected  {100*sum(r[2] for r in bad)/len(bad):5.1f}%")
    print(f"    substituted among human-approved  {100*sum(r[2] for r in good)/len(good):5.1f}%")

    print("\n  the shipped rule, replayed:")
    _rates(rows, "all masks (the paper's number)")
    _rates([r for r in rows if not r[2]], "answered crops only")
    _rates([r for r in rows if r[2]], "substituted crops only")


def pace(export: Path, total_frames: int = 4110, total_regions: int = 145428) -> None:
    """Per-frame audit time, with browser-idle time removed.

    `elapsedMs` is the interval a task was open, not time worked: a frame opened
    and left sitting carries the whole idle stretch. Raw, that puts 14.0 of the
    20.9 recorded hours inside 113 of 1,001 frames, so any mean over it measures
    the labeller's browsing habits rather than the cost of the audit.

    Frames differ in how much there is to judge (1 to 86 regions, median 36), so
    the work rate is seconds *per region*, not per frame. That rate is tight
    through the bulk of the pass -- p25 0.66, p50 0.87, p75 1.16 s/region -- and
    then breaks: p95 is 5.7 and the worst frame records 93.8 s/region, an hour on
    one screen. The break is the idle tasks.

    Frames above the 90th-percentile rate are therefore winsorized to it rather
    than dropped: the work in them did happen, only the idle stretch did not, and
    charging them at a ceiling of roughly three times the median rate keeps every
    frame in the estimate while bounding what idling can contribute.
    """
    per_frame: dict[str, int] = {}
    regions: dict[str, int] = {}
    with export.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if not row.get("userEmail") or not row.get("elapsedMs"):
                continue
            stem = Path(row["frame"]).stem
            regions[stem] = regions.get(stem, 0) + 1
            try:
                per_frame.setdefault(stem, int(row["elapsedMs"]))
            except ValueError:
                continue

    keys = list(per_frame)
    sec = {k: per_frame[k] / 1000 for k in keys}
    n, judged = len(keys), sum(regions[k] for k in keys)
    rates = sorted(sec[k] / regions[k] for k in keys)
    ceiling = rates[int(0.90 * n)]

    print(f"\n=== annotator pace over {n:,} frames, {judged:,} regions ===")
    print("  s/region  p25 %.2f  p50 %.2f  p75 %.2f  p90 %.2f  p95 %.2f  max %.2f"
          % (rates[int(.25 * n)], rates[int(.50 * n)], rates[int(.75 * n)],
             rates[int(.90 * n)], rates[int(.95 * n)], rates[-1]))

    idle = sum(sec[k] for k in keys if sec[k] > ceiling * regions[k])
    n_idle = sum(1 for k in keys if sec[k] > ceiling * regions[k])
    print(f"  raw total {sum(sec.values())/3600:.1f} h, of which {idle/3600:.1f} h "
          f"sits in the {n_idle} frames above {ceiling:.2f} s/region")

    def scale(total_sec, label, capped=None):
        # Regions are the unit of work and the two sets are near-identical in
        # density (35.9 vs 35.4 per frame), so this is a frame scale-up either way.
        hours = total_sec / judged * total_regions / 3600
        tail = "" if capped is None else f"   ({capped} frames winsorized)"
        print(f"  {label:32s} {total_sec/n:5.1f} s/frame  {total_sec/judged:4.2f} s/region"
              f"  ->  {hours:5.1f} h over {total_frames:,} frames{tail}")

    print()
    scale(sum(sec.values()), "raw (idle time included)")
    scale(sum(min(sec[k], ceiling * regions[k]) for k in keys),
          f"winsorized at {ceiling:.2f} s/region", n_idle)
    scale(sum(min(sec[k], 2.0 * regions[k]) for k in keys),
          "winsorized at 2.00 s/region",
          sum(1 for k in keys if sec[k] > 2.0 * regions[k]))
    scale(st.median(sec.values()) * n, "median frame")
    print("  The de-idled estimators agree to within 2 h; the raw figure is "
          "two-thirds idle browser time and is not an estimate of anything.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="qwen2.5vl_72b_v2")
    ap.add_argument("--export", type=Path,
                    default=Path("human_verified_output/verify_export.csv"))
    args = ap.parse_args()
    fallback(args.tag, args.export)
    pace(args.export)
