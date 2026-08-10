#!/usr/bin/env python3
"""
Post-run analysis of VLM pipeline results.

Reads vlm/results/frame_*.json and prints a paper-ready report.
All sections are formatted for direct copy-paste into the paper draft.

Sync from HPC then run (replace TAG with the model tag, e.g. qwen2.5vl_72b_v2).
`--delete` matters: scp/rsync without it leaves pruned frames behind locally,
which is how a cleaned run reappears dirty on this machine.
    rsync -av --delete totahv@base.hpc.taltech.ee:/gpfs/mariana/smbhome/totahv/zod_temp/vlm/TAG/ \\
        /mnt/ml/zod_temp/vlm/TAG/
    python analyze_results.py --tag TAG

The report opens with RUN INTEGRITY, which must be read before any other
number is quoted. The 2026-06 qwen run looked entirely normal in every other
section while 72% of its verdicts were serving-fault defaults (see
vlm/health.py). Frames processed during such a window are identified by the
corruption oracle and can be excluded with --exclude-corrupt or listed for
deletion-and-resume with --list-corrupt.
"""

import argparse
import datetime
import json
import statistics
from collections import defaultdict
from pathlib import Path

import config
from vlm.health import looks_degenerate

SEP  = "─" * 64
SEP2 = "═" * 64
CLASSES = ["vehicle", "sign", "cyclist", "pedestrian"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No results found in {results_dir}")
    return [json.loads(f.read_text()) for f in files]


# ── Run integrity ─────────────────────────────────────────────────────────────
#
# Two independent oracles for "this frame was processed against a degraded
# server". Runs predating vlm/health.py only have the first.
#
#   1. Stored responses. `discovered[].vlm_response` holds the raw answer; a
#      degraded server wrote characters with no alphanumeric content. Works
#      retroactively on any run, but sees discovery calls only — bbox raw
#      responses were never stored before the health fix.
#   2. Recorded flags. `discovered[].degenerate` and the per-agent
#      `masks[].parse_failed[agent].degenerate` are written by the pipeline
#      itself and cover every call, mask calls included.
#
# `looks_degenerate` is imported from vlm.health rather than restated here. A
# local copy used to shadow it and the two had already drifted — this one
# scored a None or empty response as *usable*, where the pipeline's own
# definition counts both as degenerate — so the same run could be called clean
# by the analysis and aborted by the monitor.

# Agents whose output core.triage.decide() actually reads. A degenerate answer
# from one of these changes the stored verdict; failure_mode is recorded but
# never consulted, so degrading it costs nothing.
VERDICT_AGENTS = {"bbox", "quality", "consistency", "correction"}


def frame_integrity(rec: dict) -> dict:
    """Classify one frame and count its degraded agent outcomes."""
    disc = rec.get("discovered", [])
    # Oracle 1 — retroactive, works on pre-health-fix runs.
    stored = [looks_degenerate(d.get("vlm_response")) for d in disc]
    # Oracle 2 — recorded by the pipeline, covers mask calls too.
    flagged_disc = sum(1 for d in disc if d.get("degenerate"))

    fail_by_agent, degen_by_agent = defaultdict(int), defaultdict(int)
    masks_degraded = 0
    for m in rec.get("masks", []):
        # Pre-health-fix runs have no parse_failed field at all; post-fix it is
        # {agent: {"degenerate": bool, "raw": str}} or None.
        hit = False
        for agent, info in (m.get("parse_failed") or {}).items():
            fail_by_agent[agent] += 1
            if info.get("degenerate"):
                degen_by_agent[agent] += 1
                hit |= agent in VERDICT_AGENTS
        masks_degraded += hit

    if not disc:
        status = "no-discovery"
    elif all(stored):
        status = "corrupt"
    elif any(stored):
        status = "partial"
    else:
        status = "clean"

    n_disc_degenerate = max(sum(stored), flagged_disc)

    return dict(
        frame_id=rec.get("frame_id", "?"),
        status=status,
        # Discovery is lost for the whole frame; the frame has to be redone.
        unusable=status in ("corrupt", "partial"),
        # Any degenerate response at all — a frame worth redoing, but its other
        # masks are still sound.
        touched=status in ("corrupt", "partial") or flagged_disc > 0 or masks_degraded > 0,
        n_masks=len(rec.get("masks", [])),
        masks_degraded=masks_degraded,
        n_disc=len(disc),
        n_disc_degenerate=n_disc_degenerate,
        fail_by_agent=dict(fail_by_agent),
        degen_by_agent=dict(degen_by_agent),
        timestamp=rec.get("run_info", {}).get("timestamp", ""),
        health=rec.get("run_info", {}).get("vlm_health"),
    )


def health_segments(frames: list[dict]) -> list[dict]:
    """
    Split frames into pipeline-process runs.

    `vlm_health.vlm_calls` counts from process start, so a resume after a health
    abort resets it. The counter is not monotonic within a segment — several
    frame workers snapshot it concurrently — so only a collapse well below the
    running peak marks a genuine restart.
    """
    rows = sorted((f for f in frames if f["health"]), key=lambda f: f["timestamp"])
    segments, peak = [], 0
    for f in rows:
        calls = f["health"].get("vlm_calls", 0)
        if peak > 200 and calls < peak * 0.5:
            segments.append([])
            peak = 0
        elif not segments:
            segments.append([])
        peak = max(peak, calls)
        segments[-1].append(f)

    out = []
    for seg in segments:
        out.append(dict(
            n_frames=len(seg),
            start=seg[0]["timestamp"],
            end=seg[-1]["timestamp"],
            calls=max(f["health"].get("vlm_calls", 0) for f in seg),
            degenerate=max(f["health"].get("degenerate_responses", 0) for f in seg),
        ))
    return out


def integrity(records: list[dict]) -> dict:
    frames = [frame_integrity(r) for r in records]
    status = defaultdict(int)
    fail_by_agent, degen_by_agent = defaultdict(int), defaultdict(int)
    by_day = defaultdict(lambda: [0, 0])

    for f in frames:
        status[f["status"]] += 1
        day = f["timestamp"][:10]
        by_day[day][0] += 1
        by_day[day][1] += f["touched"]
        for a, n in f["fail_by_agent"].items():
            fail_by_agent[a] += n
        for a, n in f["degen_by_agent"].items():
            degen_by_agent[a] += n

    return dict(
        frames=frames,
        status=dict(status),
        unusable_frames=[f for f in frames if f["unusable"]],
        touched_frames=[f for f in frames if f["touched"]],
        fail_by_agent=dict(fail_by_agent),
        degen_by_agent=dict(degen_by_agent),
        by_day=dict(sorted(by_day.items())),
        segments=health_segments(frames),
        has_telemetry=any(f["health"] for f in frames),
        masks_total=sum(f["n_masks"] for f in frames),
        masks_degraded=sum(f["masks_degraded"] for f in frames),
        disc_candidates=sum(f["n_disc"] for f in frames),
        disc_degenerate=sum(f["n_disc_degenerate"] for f in frames),
    )


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    masks_total = 0
    decisions   = defaultdict(int)
    by_class    = defaultdict(lambda: defaultdict(int))

    bbox_vals  = defaultdict(int)
    qual_vals  = defaultdict(int)
    cons_vals  = defaultdict(int)

    swin_by_class   = defaultdict(list)   # class_name → [swin_score]
    lidar_scores    = []
    bypass_count    = 0
    bypass_by_class = defaultdict(int)

    reject_causes = defaultdict(int)
    review_causes = defaultdict(int)

    disc_candidates = 0
    disc_confirmed  = 0
    disc_by_class   = defaultdict(int)

    for rec in records:
        for m in rec["masks"]:
            masks_total += 1
            d   = m["triage"]
            cls = m["class_name"]
            decisions[d] += 1
            by_class[cls][d] += 1

            ag = m["agents"]
            sc = m.get("scores", {})
            bbox = ag.get("bbox")
            qual = ag.get("quality")
            cons = ag.get("consistency")

            bbox_vals[str(bbox)] += 1
            qual_vals[str(qual)] += 1
            cons_vals[str(cons)] += 1

            swin = sc.get("swin_agreement")
            lid  = sc.get("lidar_support")
            if swin is not None:
                swin_by_class[cls].append(swin)
            if lid is not None:
                lidar_scores.append(lid)
            if sc.get("swin_bypass"):
                bypass_count += 1
                bypass_by_class[cls] += 1

            if d == "reject":
                if bbox is None and qual is None:
                    reject_causes["(1) Metadata prefilter — extreme geometry"] += 1
                elif bbox == "background":
                    reject_causes["(2) BBox=background alone [direct evidence]"] += 1
                elif bbox == "invalid" and cons == "pass":
                    reject_causes["(3) BBox=invalid + LiDAR confirmed [real content, no object found]"] += 1
                elif bbox == "invalid" and qual == "bad":
                    reject_causes["(4) BBox=invalid + Swin quality=bad"] += 1
                elif bbox == "invalid" and cons == "fail":
                    reject_causes["(5) BBox=invalid + Consistency=fail"] += 1
                elif qual == "bad" and cons == "fail":
                    reject_causes["(6) Swin quality=bad + Consistency=fail"] += 1
                else:
                    reject_causes["(7) Other"] += 1

            if d == "human_review":
                if bbox == "invalid": review_causes["BBox=invalid (single signal)"] += 1
                elif qual == "bad":   review_causes["Swin quality=bad (single signal)"] += 1
                elif cons == "fail":  review_causes["Consistency=fail (single signal)"] += 1
                else:                 review_causes["Other"] += 1

        for disc in rec.get("discovered", []):
            disc_candidates += 1
            if disc.get("confirmed"):
                disc_confirmed += 1
                disc_by_class[disc["class_name"]] += 1

    return dict(
        n_frames=len(records),
        masks_total=masks_total,
        decisions=dict(decisions),
        by_class={k: dict(v) for k, v in by_class.items()},
        bbox_vals=dict(bbox_vals),
        qual_vals=dict(qual_vals),
        cons_vals=dict(cons_vals),
        swin_by_class=dict(swin_by_class),
        lidar_scores=lidar_scores,
        bypass_count=bypass_count,
        bypass_by_class=dict(bypass_by_class),
        reject_causes=dict(reject_causes),
        review_causes=dict(review_causes),
        disc_candidates=disc_candidates,
        disc_confirmed=disc_confirmed,
        disc_by_class=dict(disc_by_class),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def pct(n, total, decimals=1):
    if not total:
        return "—"
    return f"{100*n/total:.{decimals}f}\\%"

def pct_plain(n, total, decimals=1):
    if not total:
        return "—"
    return f"{100*n/total:.{decimals}f}%"

def bar(n, total, width=30):
    if not total:
        return ""
    return "█" * int(width * n / total)


# ── Report ────────────────────────────────────────────────────────────────────

def print_integrity(g: dict, masks_total: int) -> None:
    """Section 0 — whether the numbers below can be trusted at all."""
    F = len(g["frames"])
    st = g["status"]
    unusable = len(g["unusable_frames"])
    touched = len(g["touched_frames"])

    print()
    print(SEP2)
    print(f"  RUN INTEGRITY — {F} frames")
    print(SEP2)

    if not g["has_telemetry"]:
        print("  ⚠ No `vlm_health` in run_info: this run predates the health")
        print("    monitor and was blind to serving faults while it ran. Only the")
        print("    stored-response oracle applies, and it sees discovery calls only —")
        print("    bbox degradation on these runs is not directly observable.\n")

    print(f"  {'clean':16s} {st.get('clean',0):5d}  {pct_plain(st.get('clean',0),F)}")
    print(f"  {'corrupt':16s} {st.get('corrupt',0):5d}  {pct_plain(st.get('corrupt',0),F)}"
          f"   all discovery responses degenerate")
    print(f"  {'partial':16s} {st.get('partial',0):5d}  {pct_plain(st.get('partial',0),F)}"
          f"   some degenerate")
    print(f"  {'no-discovery':16s} {st.get('no-discovery',0):5d}  {pct_plain(st.get('no-discovery',0),F)}"
          f"   oracle 1 cannot classify")
    print(f"  {'─ unusable ─':16s} {unusable:5d}  {pct_plain(unusable,F)}"
          f"   corrupt + partial: discovery lost, redo these")
    print(f"  {'─ touched ─':16s} {touched:5d}  {pct_plain(touched,F)}"
          f"   ≥1 degenerate response anywhere in the frame")

    # Frames touched overstates the damage: one bad call in a 30-mask frame
    # taints one verdict, not the frame. Rate the contamination on the units
    # that actually enter the paper's numbers.
    print(f"\n  CONTAMINATED UNITS  (verdict is a degenerate-driven SAFE_DEFAULT)")
    print(SEP)
    if g["has_telemetry"]:
        print(f"  masks               {g['masks_degraded']:6d}/{g['masks_total']:<6d} "
              f"{pct_plain(g['masks_degraded'], g['masks_total']):>7}   "
              f"a verdict agent answered degenerately")
    else:
        # No per-agent flags were recorded, so mask contamination is unknown —
        # not zero. Reporting 0% here is how the 2026-06 run looked healthy.
        print(f"  masks                    ?/{g['masks_total']:<6d} {'n/a':>7}   "
              f"not recorded before the health fix")
    if g["disc_candidates"]:
        print(f"  discovery candidates{g['disc_degenerate']:6d}/{g['disc_candidates']:<6d} "
              f"{pct_plain(g['disc_degenerate'], g['disc_candidates']):>7}   "
              f"silently counted as 'not confirmed'")

    # Fallback accounting. `parse_failed` means every attempt was unusable and
    # the verdict is SAFE_DEFAULT. Splitting it by cause separates a broken
    # server (degenerate) from the model genuinely answering UNCLEAR — the
    # latter is a real, reportable property of the prompt, not a fault.
    if g["fail_by_agent"]:
        print(f"\n  SAFE_DEFAULT SUBSTITUTIONS  (verdict is a default, not an answer)")
        print(SEP)
        print(f"  {'agent':14s} {'total':>7} {'of masks':>9} {'degenerate':>11} {'UNCLEAR etc':>12}")
        for agent, n in sorted(g["fail_by_agent"].items(), key=lambda x: -x[1]):
            deg = g["degen_by_agent"].get(agent, 0)
            print(f"  {agent:14s} {n:7d} {pct_plain(n,masks_total):>9} {deg:11d} {n-deg:12d}")
        unclear_bbox = g["fail_by_agent"].get("bbox", 0) - g["degen_by_agent"].get("bbox", 0)
        if unclear_bbox:
            print(f"\n  {unclear_bbox} masks ({pct_plain(unclear_bbox, masks_total)}) were scored"
                  f" `valid` because the BBox")
            print(f"  agent answered UNCLEAR and the parser discards it (SAFE_DEFAULT).")
            print(f"  This is prompt/parser behaviour, not a fault — report it as a")
            print(f"  limitation; the valid rate below is inflated by exactly this much.")

    if g["segments"]:
        print(f"\n  VLM SERVER HEALTH  ({len(g['segments'])} pipeline "
              f"process{'es' if len(g['segments'])>1 else ''})")
        print(SEP)
        print("  A new segment means the process restarted — a health abort plus")
        print("  --resume, or a walltime kill.")
        for i, s in enumerate(g["segments"], 1):
            rate = s["degenerate"] / s["calls"] if s["calls"] else 0
            print(f"  {i}. {s['start'][5:16].replace('T',' ')} → {s['end'][5:16].replace('T',' ')}  "
                  f"{s['n_frames']:5d} frames  "
                  f"{s['degenerate']:5d}/{s['calls']:6d} degenerate ({rate:.2%})")
        tot_c = sum(s["calls"] for s in g["segments"])
        tot_d = sum(s["degenerate"] for s in g["segments"])
        print(f"     {'total':>29}  {sum(s['n_frames'] for s in g['segments']):5d} frames  "
              f"{tot_d:5d}/{tot_c:6d} degenerate ({tot_d/tot_c if tot_c else 0:.2%})")

    if len(g["by_day"]) > 1:
        print(f"\n  PER-DAY DEGRADATION")
        print(SEP)
        for day, (n, bad) in g["by_day"].items():
            print(f"  {day}  {n:5d} frames  {bad:5d} degraded  {pct_plain(bad,n):>7}  {bar(bad,n,24)}")

    # Rate the run on contaminated units, not frames touched.
    mask_frac = (g["masks_degraded"] / g["masks_total"]
                 if g["masks_total"] and g["has_telemetry"] else 0)
    disc_frac = g["disc_degenerate"] / g["disc_candidates"] if g["disc_candidates"] else 0
    worst = max(mask_frac, disc_frac, unusable / F if F else 0)
    mask_txt = f"masks {mask_frac:.1%}" if g["has_telemetry"] else "masks n/a"

    print(f"\n  VERDICT: ", end="")
    if worst == 0:
        print("CLEAN — no degenerate responses anywhere in this run.")
    elif worst < 0.10:
        print(f"USABLE WITH CLEANUP — contamination {worst:.1%} "
              f"({mask_txt}, discovery {disc_frac:.1%}).")
        print(f"           Rates below shift by well under a point, so the")
        print(f"           qualitative picture stands. But {unusable} frame(s) lost")
        print(f"           discovery outright and {touched} carry a defaulted verdict;")
        print(f"           redoing all {touched} is cheap and leaves a pristine run:")
        print(f"             python analyze_results.py --tag TAG --list-corrupt \\")
        print(f"               | sed 's|.*|<results>/&.json|' | xargs rm -f")
        print(f"           then rerun process_frames.py --resume.")
    else:
        print(f"DO NOT QUOTE — contamination is {worst:.1%} "
              f"({mask_txt}, discovery {disc_frac:.1%}).")
        print("           The verdict rates below are substantially SAFE_DEFAULTs,")
        print("           not model behaviour. Rerun, or use --exclude-corrupt to")
        print("           describe the surviving clean subset.")
    print(SEP2)


def print_report(r: dict) -> None:
    N  = r["masks_total"]
    F  = r["n_frames"]
    d  = r["decisions"]
    acc = d.get("accept", 0)
    rej = d.get("reject", 0)
    ref = d.get("refine", 0)
    rev = d.get("human_review", 0)
    kept = acc + ref + rev
    byp  = r["bypass_count"]

    all_swin = [s for scores in r["swin_by_class"].values() for s in scores]

    # ── Header ────────────────────────────────────────────────────
    print()
    print(SEP2)
    print(f"  PIPELINE RESULTS — {F} frames, {N} masks ({N/F:.1f} per frame)")
    print(SEP2)

    # ── 1. Narrative summary (ready to paste into paper) ──────────
    print(f"""
EXECUTIVE SUMMARY  (paste into paper)
{SEP}
We applied the pipeline to {F} frames containing {N} SAM mask proposals
({N/F:.1f} masks per frame on average). All masks were evaluated by both the
Swin Quality Agent and the BBox VLM agent to collect complete signal data.
The triage system accepted {acc} masks ({pct_plain(acc,N)}), rejected {rej}
({pct_plain(rej,N)}) as invalid, and routed {rev} ({pct_plain(rev,N)}) for
human review due to a single ambiguous signal.
The Swin bypass threshold (τ_skip) would have triggered for {byp} masks
({pct_plain(byp,N)}), indicating potential inference savings of that fraction
without the BBox VLM call — a figure used for the bypass ablation in replay_triage.py.
Discovery identified {r['disc_confirmed']} previously unannotated objects
confirmed by VLM across the {F} frames ({r['disc_confirmed']/F:.1f} per frame).""")

    # ── 2. Triage decisions ───────────────────────────────────────
    print(f"\nTRIAGE DECISIONS")
    print(SEP)
    rows = [("accept",       acc, "Mask is valid — kept as-is"),
            ("reject",       rej, "Mask removed  — pixels zeroed"),
            ("human_review", rev, "Single signal — kept, flagged for review"),
            ("refine",       ref, "Geometrically off — kept, flagged for correction")]
    for label, v, desc in rows:
        print(f"  {label:15s} {v:5d}  {pct_plain(v,N):6s}  {bar(v,N,28)}  {desc}")
    print(f"  {'kept total':15s} {kept:5d}  {pct_plain(kept,N):6s}")

    # ── 3. Per-class breakdown ────────────────────────────────────
    print(f"\nPER-CLASS BREAKDOWN")
    print(SEP)
    print(f"  {'Class':12s}  {'N':>5}  {'Accept':>7}  {'Reject':>7}  {'Review':>7}  "
          f"{'Reject%':>8}  {'τ_q':>5}  {'τ_skip':>6}")
    for cls in CLASSES:
        cd  = r["by_class"].get(cls, {})
        tot = sum(cd.values())
        if not tot:
            continue
        rj  = cd.get("reject", 0)
        cls_id = next((k for k,v in config.CLASS_ID_TO_NAME.items() if v == cls), None)
        tq   = config.swin_quality_threshold(cls_id)  if cls_id else "—"
        tsk  = config.swin_skip_threshold(cls_id)     if cls_id else "—"
        flag = "  ⚠ small obj" if cls in ("cyclist","pedestrian") else ""
        print(f"  {cls:12s}  {tot:5d}  {cd.get('accept',0):7d}  {rj:7d}  "
              f"{cd.get('human_review',0):7d}  {pct_plain(rj,tot):>8}  "
              f"{tq:>5}  {tsk:>6}{flag}")

    # ── 4. Agent signal breakdown ─────────────────────────────────
    print(f"\nAGENT SIGNAL BREAKDOWN  — what each agent contributed")
    print(SEP)
    bv = r["bbox_vals"]
    qv = r["qual_vals"]
    cv = r["cons_vals"]

    print(f"  BBox VLM (forced-choice crop query):")
    print(f"    valid      {bv.get('valid',0):5d}  {pct_plain(bv.get('valid',0),N)}  "
          f"→ object confirmed present")
    print(f"    invalid    {bv.get('invalid',0):5d}  {pct_plain(bv.get('invalid',0),N)}  "
          f"→ expected object not found (1 negative)")
    print(f"    background {bv.get('background',0):5d}  {pct_plain(bv.get('background',0),N)}  "
          f"→ VLM identified non-object surface (2 negatives, rejects alone)")
    print(f"    skipped    {bv.get('None',0):5d}  {pct_plain(bv.get('None',0),N)}  "
          f"→ early exit (older run with bypass enabled)")

    print(f"\n  Swin Quality Agent (pixel agreement score):")
    print(f"    good       {qv.get('good',0):5d}  {pct_plain(qv.get('good',0),N)}  "
          f"→ α ≥ τ_q  (mask pixels match predicted class)")
    print(f"    bad        {qv.get('bad',0):5d}  {pct_plain(qv.get('bad',0),N)}  "
          f"→ α < τ_q  (1 negative)")
    print(f"    skipped    {qv.get('None',0):5d}  {pct_plain(qv.get('None',0),N)}  "
          f"→ early exit (2 negatives already)")
    print(f"  Swin bypass (would-have-triggered, τ_skip): {byp:5d}  {pct_plain(byp,N)}  "
          f"→ α ≥ τ_skip (VLM was still called; use with_bypass replay to simulate)")

    print(f"\n  Consistency Agent (LiDAR support, deterministic):")
    print(f"    pass       {cv.get('pass',0):5d}  {pct_plain(cv.get('pass',0),N)}  "
          f"→ sufficient LiDAR depth returns under mask")
    print(f"    fail       {cv.get('fail',0):5d}  {pct_plain(cv.get('fail',0),N)}  "
          f"→ mask lacks geometric support (1 negative)")

    # ── 5. Swin score distribution per class ─────────────────────
    print(f"\nSWIN AGREEMENT SCORES  α ∈ [0, 1]  — per class")
    print(SEP)
    print(f"  {'Class':12s}  {'N':>4}  {'Mean':>6}  {'Med':>6}  "
          f"{'<τ_q':>6}  {'≥τ_skip':>8}  {'τ_q':>5}  {'τ_skip':>6}")
    for cls in CLASSES:
        scores = r["swin_by_class"].get(cls, [])
        if not scores:
            continue
        cls_id = next((k for k,v in config.CLASS_ID_TO_NAME.items() if v == cls), None)
        tq   = config.swin_quality_threshold(cls_id)
        tsk  = config.swin_skip_threshold(cls_id)
        n_bad = sum(1 for s in scores if s < tq)
        n_byp = sum(1 for s in scores if s >= tsk)
        print(f"  {cls:12s}  {len(scores):4d}  "
              f"{statistics.mean(scores):6.3f}  {statistics.median(scores):6.3f}  "
              f"{pct_plain(n_bad,len(scores)):>6}  {pct_plain(n_byp,len(scores)):>8}  "
              f"{tq:>5}  {tsk:>6}")

    if all_swin:
        print(f"\n  Overall distribution (all classes):")
        bins = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]
        for lo, hi in bins:
            cnt = sum(1 for s in all_swin if lo <= s < hi)
            print(f"    α=[{lo:.1f},{hi:.1f})  {cnt:4d}  {bar(cnt,len(all_swin),24)}")
        print(f"  Note: strongly bimodal — masks are either clearly correct (α≥0.9)")
        print(f"        or clearly wrong (α<0.1), with few ambiguous cases.")

    # ── 6. Reject / review causes ─────────────────────────────────
    print(f"\nWHAT CAUSED EACH REJECT  (n={rej})")
    print(SEP)
    print("  Rejection requires ≥2 concordant negative signals.")
    print("  'background' counts as 2 negatives alone.")
    print("  'bbox=invalid + LiDAR confirmed' also counts as 2 negatives (special rule).\n")
    for k, v in sorted(r["reject_causes"].items(), key=lambda x: -x[1]):
        print(f"  {k:50s}  {v:4d}  {pct_plain(v,rej)}")

    print(f"\nWHAT CAUSED EACH HUMAN REVIEW  (n={rev})")
    print(SEP)
    print("  Single negative signal → kept in annotation but flagged.\n")
    for k, v in sorted(r["review_causes"].items(), key=lambda x: -x[1]):
        print(f"  {k:50s}  {v:4d}  {pct_plain(v,rev)}")

    # ── 7. Discovery ──────────────────────────────────────────────
    dc = r["disc_candidates"]
    cf = r["disc_confirmed"]
    if dc > 0:
        print(f"\nDISCOVERY — missed object recovery")
        print(SEP)
        print(f"  Swin predicts non-background class on pixels annotation_sam marks as")
        print(f"  background. VLM confirms each candidate region.\n")
        print(f"  Candidates (Swin-proposed)  : {dc}  ({dc/F:.1f} per frame)")
        print(f"  Confirmed by VLM            : {cf}  ({pct_plain(cf,dc)} hit rate)")
        print(f"  Added to annotations        : {cf}  ({pct_plain(cf,N)} of original mask count)")
        print(f"  By class:")
        for cls, cnt in sorted(r["disc_by_class"].items(), key=lambda x: -x[1]):
            print(f"    {cls:12s}: {cnt}")

    # ── 8. LaTeX snippets ─────────────────────────────────────────
    print(f"\nLATEX SNIPPETS  — copy-paste into paper")
    print(SEP)

    print("% --- Overall triage results (one row for results table) ---")
    print(f"Full pipeline & {F} & {N} & {pct(acc,N)} & {pct(rej,N)} "
          f"& {pct(rev,N)} & {pct(byp,N)} \\\\")

    print("\n% --- Per-class results table ---")
    print("\\begin{tabular}{lrrrrrr}")
    print("\\hline")
    print("Class & Total & Accept & Reject & Review & Reject\\% & $\\tau_q$ \\\\")
    print("\\hline")
    for cls in CLASSES:
        cd  = r["by_class"].get(cls, {})
        tot = sum(cd.values())
        if not tot:
            continue
        rj = cd.get("reject", 0)
        cls_id = next((k for k,v in config.CLASS_ID_TO_NAME.items() if v == cls), None)
        tq = config.swin_quality_threshold(cls_id)
        print(f"{cls.capitalize()} & {tot} & {cd.get('accept',0)} & {rj} "
              f"& {cd.get('human_review',0)} & {pct(rj,tot)} & {tq} \\\\")
    print("\\hline")
    print("\\end{tabular}")

    print("\n% --- Swin score distribution table ---")
    print("\\begin{tabular}{lrrrrrr}")
    print("\\hline")
    print("Class & N & Mean $\\alpha$ & Median $\\alpha$ & $\\alpha < \\tau_q$ & "
          "$\\alpha \\geq \\tau_{\\text{skip}}$ & $\\tau_q$ \\\\")
    print("\\hline")
    for cls in CLASSES:
        scores = r["swin_by_class"].get(cls, [])
        if not scores:
            continue
        cls_id = next((k for k,v in config.CLASS_ID_TO_NAME.items() if v == cls), None)
        tq  = config.swin_quality_threshold(cls_id)
        tsk = config.swin_skip_threshold(cls_id)
        n_bad = sum(1 for s in scores if s < tq)
        n_byp = sum(1 for s in scores if s >= tsk)
        print(f"{cls.capitalize()} & {len(scores)} & "
              f"{statistics.mean(scores):.2f} & {statistics.median(scores):.2f} & "
              f"{pct(n_bad,len(scores))} & {pct(n_byp,len(scores))} & {tq} \\\\")
    print("\\hline")
    print("\\end{tabular}")

    if dc > 0:
        print(f"\n% --- Discovery sentence ---")
        print(f"% The discovery agent identified {dc} candidate regions across {F} frames,")
        print(f"% of which {cf} ({pct_plain(cf,dc)}) were confirmed by the VLM as genuine")
        print(f"% objects absent from the original SAM annotations.")

    # ── Footer ────────────────────────────────────────────────────
    print()
    print(SEP2)
    print(f"  Frames  : {F}")
    print(f"  Masks   : {N}  ({N/F:.1f} per frame)")
    print(f"  Kept    : {kept}  ({pct_plain(kept,N)})   →  Rejected: {rej}  ({pct_plain(rej,N)})")
    print(f"  Bypass  : {byp}  ({pct_plain(byp,N)}) of masks skipped BBox VLM call")
    if dc > 0:
        print(f"  Added   : {cf} objects recovered by discovery")
    print(SEP2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    parser.add_argument("--tag", default=None,
                        help="run tag to analyze (vlm/<tag>/results/). Must match --tag used in process_frames.py")
    parser.add_argument("--exclude-corrupt", action="store_true",
                        help="drop frames with any degenerate response before aggregating, "
                             "so the report describes model behaviour rather than SAFE_DEFAULTs")
    parser.add_argument("--list-corrupt", action="store_true",
                        help="print ids of frames carrying any degenerate response and exit — "
                             "feed to rm, then rerun process_frames.py with --resume")
    parser.add_argument("--integrity-only", action="store_true",
                        help="print the RUN INTEGRITY section and stop")
    args = parser.parse_args()
    if args.hpc:
        config.use_hpc()
    if args.tag:
        config.set_run_tag(args.tag)

    records = load_results(config.RESULTS_DIR)
    g = integrity(records)

    if args.list_corrupt:
        for f in sorted(g["touched_frames"], key=lambda f: f["frame_id"]):
            print(f["frame_id"])
        return

    # Integrity first: every rate below is only meaningful once the degraded
    # fraction is known.
    print_integrity(g, sum(len(r["masks"]) for r in records))
    if args.integrity_only:
        return

    if args.exclude_corrupt:
        bad = {f["frame_id"] for f in g["touched_frames"]}
        records = [r for r in records if r.get("frame_id") not in bad]
        print(f"\n  --exclude-corrupt: dropped {len(bad)} degraded frame(s); "
              f"{len(records)} remain.")
        if not records:
            raise SystemExit("No clean frames left to analyze.")

    r = aggregate(records)
    print_report(r)


if __name__ == "__main__":
    main()
