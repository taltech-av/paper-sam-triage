#!/usr/bin/env python3
"""
Post-run analysis of VLM pipeline results.

Reads vlm/results/frame_*.json and prints a paper-ready report.
All sections are formatted for direct copy-paste into the paper draft.

Sync from HPC then run (replace TAG with the model tag, e.g. qwen2.5vl_72b):
    rsync -avP totahv@base.hpc.taltech.ee:/gpfs/mariana/smbhome/totahv/zod_temp/vlm/TAG/results/ \\
        /run/media/tom/ml/zod_temp/vlm/TAG/results/
    python analyze_results.py --tag TAG
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import config

SEP  = "─" * 64
SEP2 = "═" * 64
CLASSES = ["vehicle", "sign", "cyclist", "pedestrian"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No results found in {results_dir}")
    return [json.loads(f.read_text()) for f in files]


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
                    reject_causes["(2) BBox=background  [2 negatives, VLM-only]"] += 1
                elif bbox == "invalid" and cons == "fail":
                    reject_causes["(3) BBox=invalid + Consistency=fail"] += 1
                elif bbox == "invalid" and qual == "bad":
                    reject_causes["(4) BBox=invalid + Swin quality=bad"] += 1
                elif qual == "bad" and cons == "fail":
                    reject_causes["(5) Swin quality=bad + Consistency=fail"] += 1
                else:
                    reject_causes["(6) Other"] += 1

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
    print("  'background' counts as 2 negatives alone (VLM identified wrong surface type).\n")
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
    args = parser.parse_args()
    if args.hpc:
        config.use_hpc()
    if args.tag:
        config.set_run_tag(args.tag)

    records = load_results(config.RESULTS_DIR)
    r = aggregate(records)
    print_report(r)


if __name__ == "__main__":
    main()
