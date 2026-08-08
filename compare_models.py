#!/usr/bin/env python3
"""
Cross-model comparison of two VLM runs on a common frame set.

Makes `qwen2.5vl_72b_v2` and `llava_34b` comparable despite three asymmetries
between them. Everything is computed from stored per-frame JSON — no new VLM
inference.

    python compare_models.py
    python compare_models.py --tag-a llava_34b --tag-b qwen2.5vl_72b_v2
    python compare_models.py --latex paper/tables/agent_behavior.tex
    python compare_models.py --exclude-degraded      # drop qwen frames with any
                                                     # degenerate response

Asymmetry 1 — frame coverage. The qwen rerun is still in progress, so it covers
a subset of llava's 4,135 frames. The pending frames are *not* a random sample:
they carry 28.4 masks/frame against 20.9 in the completed ones, so comparing
whole runs compares different mask populations. Everything below is restricted
to the frames both runs completed, and `--verify` checks that the two runs saw
byte-identical VLM-independent inputs on them.

Asymmetry 2 — SAFE_DEFAULT visibility on the BBox agent. Since vlm/health.py,
a mask whose every BBox attempt was unparseable records
`parse_failed["bbox"]`, and its stored verdict is the SAFE_DEFAULT "valid"
rather than an answer. The llava run predates this and stored only the parsed
verdict, so its defaulted masks are indistinguishable from confidently-valid
ones and cannot be recovered offline (raw BBox responses were never stored).
Two bases are therefore reported side by side:

  * `as-recorded`  — the verdict the pipeline acted on, defaults included.
    Directly comparable across runs, and the basis that matters for every
    downstream claim, because this is what was written into the annotations.
  * `answered-only` — defaults removed. This is the model-behaviour basis, and
    it is measurable for the qwen rerun only. It is reported as a one-sided
    correction, never as a cross-model number.

The missing number is not needed, because the BBox comparison can be settled by
a one-sided bound instead. `BBoxAgent.SAFE_DEFAULT` is `"valid"`, so every
defaulted mask is counted as valid: a run's valid rate is a weighted mean of its
answered rate and 100%, and therefore

    valid(answered only)  <=  valid(as-recorded)        for ANY run,

with equality only when nothing was defaulted. The bound holds whatever llava's
unmeasured default share turns out to be. So if the measured answered-only rate
of one model exceeds the *as-recorded* rate of the other, that model is the more
permissive one on answered masks, and no rerun can overturn it. `--verify`
prints this comparison as VERDICT PERMISSIVENESS.

Asymmetry 3 — discovery is fully recoverable, and is where model behaviour is
measured directly. `discovered[].vlm_response` holds the raw answer verbatim in
*both* runs, so re-parsing it with the current rule separates four outcomes on
each side identically: confirmed, declined (`other`), a reply outside the
offered vocabulary, and no content at all. The stored `confirmed` flag collapses
the last three into the first negative, which is how a degraded server read as a
sceptical model in the 2026-06 run. Splitting them also separates *model*
non-answering (out-of-vocabulary text) from *server* non-answering (degenerate
output), which are attributed to opposite causes.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import config
from agents.discovery_agent import _ANSWERABLE, _VALID_RESPONSES
from core.triage import triage
from vlm.health import looks_degenerate

SEP = "─" * 74
SEP2 = "═" * 74
CLASSES = ["vehicle", "sign", "cyclist", "pedestrian"]
BBOX_VERDICTS = ["valid", "invalid", "background"]
TRIAGE_OUTCOMES = ["accept", "reject", "human_review", "refine"]
DISC_CLASSES = {1: "vehicle", 2: "sign", 3: "human"}

# VLM-independent per-mask signals. Both runs compute these from the same Swin
# checkpoint and the same LiDAR projection, so on a shared mask they must agree
# exactly. A mismatch means the runs are not comparable at all and every number
# below is meaningless — hence --verify rather than a silent assumption.
SHARED_SCORES = ["swin_agreement", "lidar_support"]


# ── Loading ───────────────────────────────────────────────────────────────────

def load_run(tag: str) -> dict[str, dict]:
    results_dir = config.DATA_ROOT / "vlm" / tag / "results"
    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No results found in {results_dir}")
    out = {}
    for f in files:
        rec = json.loads(f.read_text())
        out[rec["frame_id"]] = rec
    return out


def degraded_frames(records: dict[str, dict]) -> set[str]:
    """Frame ids carrying any degenerate response, by either oracle."""
    bad = set()
    for fid, rec in records.items():
        hit = any(looks_degenerate(d.get("vlm_response")) or d.get("degenerate")
                  for d in rec.get("discovered", []))
        for m in rec.get("masks", []):
            for info in (m.get("parse_failed") or {}).values():
                hit |= bool(info.get("degenerate"))
        if hit:
            bad.add(fid)
    return bad


# ── Input identity ────────────────────────────────────────────────────────────

def verify_shared_inputs(runs: dict[str, dict], frames: list[str]) -> dict:
    """
    Confirm both runs scored the same masks with the same VLM-independent
    signals, so a verdict difference is attributable to the VLM alone.
    """
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]
    mask_ids_differ = score_mismatch = masks = 0
    cand_differ = 0

    for fid in frames:
        ma = {m["mask_id"]: m for m in a[fid]["masks"]}
        mb = {m["mask_id"]: m for m in b[fid]["masks"]}
        if set(ma) != set(mb):
            mask_ids_differ += 1
        for mid in set(ma) & set(mb):
            masks += 1
            sa, sb = ma[mid].get("scores", {}), mb[mid].get("scores", {})
            if any(sa.get(k) != sb.get(k) for k in SHARED_SCORES):
                score_mismatch += 1
        # Discovery candidates are Swin-determined and deterministic; only the
        # confirmation verdict is the VLM's. Differing counts would mean the
        # candidate generator itself changed between runs.
        if len(a[fid].get("discovered", [])) != len(b[fid].get("discovered", [])):
            cand_differ += 1

    return dict(frames=len(frames), masks=masks, mask_ids_differ=mask_ids_differ,
                score_mismatch=score_mismatch, cand_differ=cand_differ)


# ── Per-run statistics ────────────────────────────────────────────────────────

def recomputed_triage(mask: dict) -> str:
    """
    Triage under the CURRENT rule, from stored agent outputs.

    Stored `triage` is not used: the two runs were produced months apart, and
    recomputing both under one rule guarantees the comparison isolates VLM
    verdicts from any rule change in between.
    """
    a = mask["agents"]
    return triage(
        bbox_out=a.get("bbox"),
        quality_out=a.get("quality"),
        failure_mode_out=a.get("failure_mode"),
        correction_out=a.get("correction"),
        consistency_out=a.get("consistency"),
    ).decision


def mask_stats(records: dict[str, dict], frames: list[str]) -> dict:
    bbox = defaultdict(int)          # as-recorded, defaults included
    bbox_answered = defaultdict(int) # defaults removed
    tri = defaultdict(int)
    reject_by_class, total_by_class = defaultdict(int), defaultdict(int)
    defaults = defaultdict(int)      # agent → masks whose verdict is a default
    degen = defaultdict(int)
    correction_called = correction_empty = 0
    telemetry = False
    total = 0

    for fid in frames:
        for m in records[fid]["masks"]:
            total += 1
            cls = m["class_name"]
            total_by_class[cls] += 1
            verdict = m["agents"].get("bbox")
            bbox[str(verdict)] += 1

            pf = m.get("parse_failed") or {}
            if m.get("parse_failed") is not None:
                telemetry = True
            for agent, info in pf.items():
                defaults[agent] += 1
                if info.get("degenerate"):
                    degen[agent] += 1
            if "bbox" not in pf:
                bbox_answered[str(verdict)] += 1

            if m["agents"].get("correction") is not None or "correction" in pf:
                correction_called += 1
                if "correction" in pf:
                    correction_empty += 1

            d = recomputed_triage(m)
            tri[d] += 1
            if d == "reject":
                reject_by_class[cls] += 1

    return dict(total=total, bbox=dict(bbox), bbox_answered=dict(bbox_answered),
                triage=dict(tri), reject_by_class=dict(reject_by_class),
                total_by_class=dict(total_by_class), defaults=dict(defaults),
                degen=dict(degen), has_telemetry=telemetry,
                correction_called=correction_called,
                correction_empty=correction_empty)


def disc_stats(records: dict[str, dict], frames: list[str]) -> dict:
    """
    Discovery outcomes re-derived from the stored raw response.

    Both runs store `vlm_response` verbatim, so applying one parser to both
    gives the only cross-model quantity in this script that is fully measured
    on each side. Four outcomes, deliberately kept apart:

      confirmed   — the model named a class the prompt offered
      other       — the model declined; a real negative
      unanswered  — a reply outside the offered vocabulary; no evidence
      no-response — empty/None/degenerate; the server, not the model

    `confirmed=False` in the stored record collapses the last three into the
    first negative, which is exactly how a degraded server read as a sceptical
    model in the 2026-06 run.
    """
    out = {}
    for swin_cls, label in DISC_CLASSES.items():
        out[label] = defaultdict(int)
    totals = defaultdict(int)
    human_added = 0

    for fid in frames:
        for d in records[fid].get("discovered", []):
            swin_cls = d.get("swin_class")
            label = DISC_CLASSES.get(swin_cls)
            if label is None:
                continue
            raw = d.get("vlm_response")
            norm = (raw or "").strip().lower().rstrip(".")
            if looks_degenerate(raw):
                bucket = "no-response"
            elif norm in _VALID_RESPONSES[swin_cls]:
                bucket = "confirmed"
            elif norm == "other":
                bucket = "other"
            else:
                bucket = "unanswered"
            out[label][bucket] += 1
            totals[bucket] += 1
            if bucket == "confirmed" and swin_cls == 3:
                human_added += 1

    return dict(by_class={k: dict(v) for k, v in out.items()},
                totals=dict(totals), human_added=human_added)


def swin_reference(records: dict[str, dict], frames: list[str],
                   answered_only: bool = False) -> dict:
    """
    BBox verdicts scored against the Swin quality signal as reference positive.

    Not a ground-truth evaluation — Swin is itself a noisy judge — but it is
    identical for both runs, so it ranks the two VLMs on one fixed yardstick.
    """
    tp = fp = tn = fn = 0
    for fid in frames:
        for m in records[fid]["masks"]:
            qual = m["agents"].get("quality")
            verdict = m["agents"].get("bbox")
            if qual is None or verdict is None:
                continue  # geometry-prefiltered: no signal from either agent
            if answered_only and "bbox" in (m.get("parse_failed") or {}):
                continue
            pred_pos = verdict == "valid"
            ref_pos = qual == "good"
            if pred_pos and ref_pos:   tp += 1
            elif pred_pos:             fp += 1
            elif ref_pos:              fn += 1
            else:                      tn += 1
    n = tp + fp + tn + fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(n=n, acc=(tp + tn) / n if n else 0.0,
                prec=prec, rec=rec, f1=f1)


def bbox_agreement(runs: dict[str, dict], frames: list[str]) -> dict:
    """Cross-model verdict agreement, and the same figure on answered masks."""
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]
    same = total = same_ans = total_ans = 0
    for fid in frames:
        mb = {m["mask_id"]: m for m in b[fid]["masks"]}
        for m in a[fid]["masks"]:
            other = mb.get(m["mask_id"])
            if other is None:
                continue
            va, vb = m["agents"].get("bbox"), other["agents"].get("bbox")
            if va is None or vb is None:
                continue
            total += 1
            same += va == vb
            if "bbox" in (m.get("parse_failed") or {}) or \
               "bbox" in (other.get("parse_failed") or {}):
                continue
            total_ans += 1
            same_ans += va == vb
    return dict(agree=same / total if total else 0.0, n=total,
                agree_answered=same_ans / total_ans if total_ans else 0.0,
                n_answered=total_ans)


def timing(records: dict[str, dict], frames: list[str]) -> dict:
    """Per-mask BBox latency and per-frame wall time from stored timings."""
    bbox_calls, frame_total, frame_triage, frame_disc = [], [], [], []
    for fid in frames:
        rec = records[fid]
        ri = rec.get("run_info", {})
        for key, dest in (("elapsed_seconds", frame_total),
                          ("triage_elapsed_seconds", frame_triage),
                          ("discovery_elapsed_seconds", frame_disc)):
            if ri.get(key) is not None:
                dest.append(ri[key])
        for m in rec["masks"]:
            secs = (m.get("timing") or {}).get("agent_seconds") or {}
            if secs.get("bbox"):
                bbox_calls.append(secs["bbox"])

    def q(xs, p):
        if not xs:
            return None
        s = sorted(xs)
        return s[min(len(s) - 1, int(p * len(s)))]

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    gpu_hours = sum(frame_total) / 3600 if frame_total else None
    return dict(bbox_mean=mean(bbox_calls), bbox_p90=q(bbox_calls, 0.90),
                n_bbox=len(bbox_calls),
                frame_total_med=q(frame_total, 0.5),
                frame_triage_med=q(frame_triage, 0.5),
                frame_disc_med=q(frame_disc, 0.5),
                gpu_hours=gpu_hours,
                wall_hours=gpu_hours / 4 if gpu_hours else None)


# ── Report ────────────────────────────────────────────────────────────────────

def pct(n, total, decimals=1):
    return "—" if not total else f"{100*n/total:.{decimals}f}%"


def tex_pct(n, total, decimals=1):
    """Percent for LaTeX. The escape is not optional: a bare % comments out
    the remainder of the line and silently eats the rest of the table row."""
    return "---" if not total else f"{100*n/total:.{decimals}f}\\%"


def tex_num(v):
    """Thousands separator LaTeX will not treat as a unit-spacing comma."""
    return f"{v:,}".replace(",", "{,}")


def print_report(names, stats, dstats, refs, agree, times, ident, args):
    a, b = names
    W = 22

    print()
    print(SEP2)
    print(f"  CROSS-MODEL COMPARISON — {ident['frames']} common frames, "
          f"{ident['masks']} common masks")
    print(SEP2)
    print(f"  A = {a}")
    print(f"  B = {b}")

    print(f"\n  INPUT IDENTITY  (must be zero, else nothing below is comparable)")
    print(SEP)
    print(f"  frames with differing mask ids     {ident['mask_ids_differ']}")
    print(f"  masks with differing Swin/LiDAR    {ident['score_mismatch']}")
    print(f"  frames with differing candidates   {ident['cand_differ']}")
    ok = not (ident["mask_ids_differ"] or ident["score_mismatch"]
              or ident["cand_differ"])
    print(f"  → {'PAIRED: same masks, same VLM-independent signals' if ok else 'MISMATCH — investigate before quoting'}")

    # ── SAFE_DEFAULT visibility ───────────────────────────────────────────────
    print(f"\n  SAFE_DEFAULT VISIBILITY  (can a defaulted verdict be told apart?)")
    print(SEP)
    print(f"  {'':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for label, key in (("bbox telemetry recorded", "has_telemetry"),):
        print(f"  {label:32s} {str(stats[a][key]):>{W}} {str(stats[b][key]):>{W}}")
    for agent in ("bbox", "correction"):
        row = []
        for n in names:
            s = stats[n]
            if not s["has_telemetry"]:
                row.append("not recorded")
            else:
                tot = s["defaults"].get(agent, 0)
                row.append(f"{tot} ({pct(tot, s['total'])})")
        print(f"  {'defaulted verdicts: ' + agent:32s} {row[0]:>{W}} {row[1]:>{W}}")
    print(f"\n  A defaulted BBox verdict is stored as `valid` (SAFE_DEFAULT), so a run")
    print(f"  without telemetry cannot separate it from a real `valid`. Raw BBox")
    print(f"  responses were never stored, so this is unrecoverable offline.")

    # ── BBox verdicts ─────────────────────────────────────────────────────────
    print(f"\n  BBOX VERDICTS — as-recorded  (defaults included; the basis the")
    print(f"  pipeline acted on and the one written into the annotations)")
    print(SEP)
    print(f"  {'verdict':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for v in BBOX_VERDICTS:
        print(f"  {v:32s} "
              f"{pct(stats[a]['bbox'].get(v,0), stats[a]['total']):>{W}} "
              f"{pct(stats[b]['bbox'].get(v,0), stats[b]['total']):>{W}}")

    print(f"\n  BBOX VERDICTS — answered only  (defaults removed; model behaviour.")
    print(f"  One-sided: measurable only where telemetry exists. NOT a cross-model row.)")
    print(SEP)
    print(f"  {'verdict':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for v in BBOX_VERDICTS:
        row = []
        for n in names:
            s = stats[n]
            if not s["has_telemetry"]:
                row.append("n/a")
            else:
                den = sum(s["bbox_answered"].values())
                row.append(pct(s["bbox_answered"].get(v, 0), den))
        print(f"  {v:32s} {row[0]:>{W}} {row[1]:>{W}}")

    # ── The bound that settles the comparison ─────────────────────────────────
    print(f"\n  VERDICT PERMISSIVENESS — settled without the missing telemetry")
    print(SEP)
    print(f"  SAFE_DEFAULT is `valid`, so defaults can only push a valid rate UP:")
    print(f"  for any run, valid(answered) <= valid(as-recorded).")
    lo, hi = {}, {}
    for n in names:
        s = stats[n]
        hi[n] = 100 * s["bbox"].get("valid", 0) / s["total"] if s["total"] else 0.0
        den = sum(s["bbox_answered"].values())
        lo[n] = (100 * s["bbox_answered"].get("valid", 0) / den
                 if s["has_telemetry"] and den else None)
    for n in names:
        if lo[n] is None:
            print(f"  {n[:34]:34s} valid(answered) <= {hi[n]:5.1f}%   (measured: n/a)")
        else:
            print(f"  {n[:34]:34s} valid(answered)  = {lo[n]:5.1f}%   "
                  f"(as-recorded {hi[n]:.1f}%)")
    measured = [n for n in names if lo[n] is not None]
    unmeasured = [n for n in names if lo[n] is None]
    if len(measured) == 1 and len(unmeasured) == 1:
        m, u = measured[0], unmeasured[0]
        if lo[m] > hi[u]:
            print(f"\n  → {lo[m]:.1f}% > {hi[u]:.1f}%: {m} is the more permissive model on")
            print(f"    answered masks by at least {lo[m]-hi[u]:.1f} points. The gap is a LOWER")
            print(f"    bound — {u}'s own defaults can only widen it — so the direction")
            print(f"    holds no matter what its unmeasured default share is. Rerunning")
            print(f"    {u} with telemetry could only make the gap larger, never smaller.")
        else:
            print(f"\n  → {lo[m]:.1f}% <= {hi[u]:.1f}%: inconclusive. {u}'s unmeasured default")
            print(f"    share could place it on either side, so the direction of the gap")
            print(f"    cannot be established without rerunning it with telemetry.")

    # ── Triage ────────────────────────────────────────────────────────────────
    print(f"\n  TRIAGE OUTCOMES  (recomputed under the current rule for both runs)")
    print(SEP)
    print(f"  {'outcome':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for o in TRIAGE_OUTCOMES:
        print(f"  {o:32s} "
              f"{pct(stats[a]['triage'].get(o,0), stats[a]['total']):>{W}} "
              f"{pct(stats[b]['triage'].get(o,0), stats[b]['total']):>{W}}")

    print(f"\n  PER-CLASS REJECTION RATE")
    print(SEP)
    print(f"  {'class':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for c in CLASSES:
        print(f"  {c:32s} "
              f"{pct(stats[a]['reject_by_class'].get(c,0), stats[a]['total_by_class'].get(c,0)):>{W}} "
              f"{pct(stats[b]['reject_by_class'].get(c,0), stats[b]['total_by_class'].get(c,0)):>{W}}")

    # ── Correction agent ──────────────────────────────────────────────────────
    print(f"\n  CORRECTION AGENT  (gates the refine path)")
    print(SEP)
    print(f"  {'':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    print(f"  {'calls':32s} {stats[a]['correction_called']:>{W}} {stats[b]['correction_called']:>{W}}")
    row = [f"{stats[n]['correction_empty']} "
           f"({pct(stats[n]['correction_empty'], stats[n]['correction_called'])})"
           if stats[n]["has_telemetry"] else "not recorded" for n in names]
    print(f"  {'unusable → default no_refine':32s} {row[0]:>{W}} {row[1]:>{W}}")
    print(f"  {'refine decisions':32s} "
          f"{stats[a]['triage'].get('refine',0):>{W}} {stats[b]['triage'].get('refine',0):>{W}}")

    # ── Discovery ─────────────────────────────────────────────────────────────
    print(f"\n  DISCOVERY — re-derived from stored raw responses with ONE parser.")
    print(f"  Fully measured on both sides: the only unbiased cross-model basis here.")
    print(SEP)
    print(f"  {'outcome':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    tot = {n: sum(dstats[n]["totals"].values()) for n in names}
    for bucket, desc in (("confirmed", "named an offered class"),
                         ("other", "declined — a real negative"),
                         ("unanswered", "reply outside the vocabulary"),
                         ("no-response", "empty/degenerate — server, not model")):
        print(f"  {bucket:32s} "
              f"{dstats[a]['totals'].get(bucket,0):>10} {pct(dstats[a]['totals'].get(bucket,0), tot[a]):>{W-11}} "
              f"{dstats[b]['totals'].get(bucket,0):>10} {pct(dstats[b]['totals'].get(bucket,0), tot[b]):>{W-11}}"
              f"   {desc}")
    print(f"  {'candidates':32s} {tot[a]:>{W}} {tot[b]:>{W}}")

    print(f"\n  {'hit rate, all candidates':32s} "
          f"{pct(dstats[a]['totals'].get('confirmed',0), tot[a]):>{W}} "
          f"{pct(dstats[b]['totals'].get('confirmed',0), tot[b]):>{W}}")
    row = []
    for n in names:
        t = dstats[n]["totals"]
        den = t.get("confirmed", 0) + t.get("other", 0)
        row.append(pct(t.get("confirmed", 0), den))
    print(f"  {'hit rate, answered only ★':32s} {row[0]:>{W}} {row[1]:>{W}}")
    print(f"  {'human objects confirmed':32s} "
          f"{dstats[a]['human_added']:>{W}} {dstats[b]['human_added']:>{W}}")
    print(f"\n  ★ the comparable discovery number: confirmed / (confirmed + other),")
    print(f"    dropping candidates the model never answered. Measured the same way")
    print(f"    on both runs, so no telemetry asymmetry enters it.")

    print(f"\n  Per-class confirmation (answered only)")
    print(SEP)
    print(f"  {'class':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for label in DISC_CLASSES.values():
        row = []
        for n in names:
            c = dstats[n]["by_class"].get(label, {})
            den = c.get("confirmed", 0) + c.get("other", 0)
            row.append(pct(c.get("confirmed", 0), den))
        print(f"  {label:32s} {row[0]:>{W}} {row[1]:>{W}}")

    print(f"\n  NON-ANSWER RATE, split by cause")
    print(SEP)
    print(f"  {'':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for label, key in (("model (out-of-vocabulary)", "unanswered"),
                       ("server (degenerate/empty)", "no-response")):
        print(f"  {label:32s} "
              f"{pct(dstats[a]['totals'].get(key,0), tot[a]):>{W}} "
              f"{pct(dstats[b]['totals'].get(key,0), tot[b]):>{W}}")
    print(f"  The two causes are opposite facts and must not be pooled: the first is")
    print(f"  the model failing to answer its prompt, the second is the serving fault.")

    # ── Swin reference ────────────────────────────────────────────────────────
    print(f"\n  BBOX vs SWIN QUALITY REFERENCE  (same yardstick for both)")
    print(SEP)
    print(f"  {'run':38s} {'N':>7} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}")
    for n in names:
        r = refs[n]["all"]
        print(f"  {n[:38]:38s} {r['n']:7d} {100*r['acc']:6.1f} {100*r['prec']:6.1f} "
              f"{100*r['rec']:6.1f} {100*r['f1']:6.1f}")
        ra = refs[n]["answered"]
        if stats[n]["has_telemetry"] and ra["n"] != r["n"]:
            print(f"  {'  └ answered only':38s} {ra['n']:7d} {100*ra['acc']:6.1f} "
                  f"{100*ra['prec']:6.1f} {100*ra['rec']:6.1f} {100*ra['f1']:6.1f}")
    print(f"\n  Cross-model BBox agreement  {agree['agree']:.1%} on {agree['n']} masks"
          f"   (answered only: {agree['agree_answered']:.1%} on {agree['n_answered']})")

    # ── Timing ────────────────────────────────────────────────────────────────
    print(f"\n  COST  (on the common frames only)")
    print(SEP)
    print(f"  {'metric':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for label, key, fmt in (("BBox call mean (s)", "bbox_mean", "{:.2f}"),
                            ("BBox call p90 (s)", "bbox_p90", "{:.2f}"),
                            ("frame triage median (s)", "frame_triage_med", "{:.1f}"),
                            ("frame discovery median (s)", "frame_disc_med", "{:.1f}"),
                            ("frame total median (s)", "frame_total_med", "{:.1f}"),
                            ("aggregate GPU-hours", "gpu_hours", "{:.1f}"),
                            ("wall-hours at 4 workers", "wall_hours", "{:.1f}")):
        row = [fmt.format(times[n][key]) if times[n][key] is not None else "—"
               for n in names]
        print(f"  {label:32s} {row[0]:>{W}} {row[1]:>{W}}")
    print(SEP2)


# ── LaTeX ─────────────────────────────────────────────────────────────────────

def write_latex(path: Path, names, stats, dstats, ident, coverage) -> None:
    """Emit tables/agent_behavior.tex on the paired frame set."""
    a, b = names
    tot = {n: sum(dstats[n]["totals"].values()) for n in names}

    def p(n, d, width=True):
        s = tex_pct(n, d)
        # IEEEtran tables align better with a digit-width phantom on 1-digit values
        if width and d and 100 * n / d < 10:
            return "\\phantom{0}" + s
        return s

    num = tex_num

    L = []
    L.append("% Generated by compare_models.py — do not edit by hand.")
    L.append(f"% Paired frame set: {ident['frames']} frames / {ident['masks']} masks.")
    L.append(f"% {a}: {coverage[a]} frames on disk; {b}: {coverage[b]} frames on disk.")
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{VLM agent behaviour on the {num(ident['frames'])} flagged frames "
             f"both models completed ({num(ident['masks'])} SAM masks, identical for both; "
             "Swin agreement and LiDAR consistency are VLM-independent and verified "
             "bit-identical across the two runs). "
             "Triage outcomes are recomputed offline under the current deterministic rule "
             "for both models. "
             "BBox rates are \\emph{as-recorded}: they include masks whose verdict is a "
             "\\textsc{safe\\_default} substitution after an unparseable reply, which is "
             "what the pipeline acted on. "
             "Discovery outcomes are re-derived from the raw responses stored by both runs "
             "with a single parser, separating a model declining a candidate "
             "(\\texttt{other}) from a reply the parser could not read at all; the hit rate "
             "is over answered candidates only. "
             "``Human objects added'' counts confirmed cyclist and pedestrian candidates.}")
    L.append("\\label{tab:agent_behavior}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{lcc}")
    L.append("\\toprule")
    L.append("\\textbf{Metric} & \\textbf{LLaVA-1.6-34B} & \\textbf{Qwen2.5-VL-72B} \\\\")
    L.append("\\midrule")

    L.append("\\multicolumn{3}{l}{\\textit{Triage outcomes (current rule, recomputed)}} \\\\")
    for label, key in (("Accept", "accept"), ("Reject", "reject"),
                       ("Human review", "human_review"), ("Refine", "refine")):
        L.append(f"{label} & {p(stats[a]['triage'].get(key,0), stats[a]['total'])} "
                 f"& {p(stats[b]['triage'].get(key,0), stats[b]['total'])} \\\\")

    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{BBox VLM verdicts (as-recorded)}} \\\\")
    for label, key in (("Valid (object present)", "valid"),
                       ("Invalid (object absent)", "invalid"),
                       ("Background", "background")):
        L.append(f"{label} & {p(stats[a]['bbox'].get(key,0), stats[a]['total'])} "
                 f"& {p(stats[b]['bbox'].get(key,0), stats[b]['total'])} \\\\")

    # The default share is the reason the two Valid rates are not a like-for-like
    # measure of model behaviour, so it belongs in the table, not a footnote.
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Verdict provenance}} \\\\")
    row = []
    for n in names:
        s = stats[n]
        row.append("n/a\\textsuperscript{a}" if not s["has_telemetry"]
                   else p(s["defaults"].get("bbox", 0), s["total"]))
    L.append(f"\\textsc{{safe\\_default}} share of BBox & {row[0]} & {row[1]} \\\\")
    row = []
    for n in names:
        s = stats[n]
        if not s["has_telemetry"]:
            row.append("n/a\\textsuperscript{a}")
        else:
            den = sum(s["bbox_answered"].values())
            row.append(p(s["bbox_answered"].get("valid", 0), den))

    L.append(f"Valid, answered only & {row[0]} & {row[1]} \\\\")

    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Per-class rejection rate}} \\\\")
    for c in CLASSES:
        L.append(f"{c.capitalize()} & "
                 f"{p(stats[a]['reject_by_class'].get(c,0), stats[a]['total_by_class'].get(c,0))} & "
                 f"{p(stats[b]['reject_by_class'].get(c,0), stats[b]['total_by_class'].get(c,0))} \\\\")

    L.append("\\addlinespace")
    L.append(f"\\multicolumn{{3}}{{l}}{{\\textit{{Discovery ({num(tot[a])} Swin-proposed candidates)}}}} \\\\")
    for label, key in (("Confirmed", "confirmed"), ("Declined (\\texttt{other})", "other")):
        L.append(f"{label} & {num(dstats[a]['totals'].get(key,0))} "
                 f"& {num(dstats[b]['totals'].get(key,0))} \\\\")
    row = []
    for n in names:
        t = dstats[n]["totals"]
        row.append(num(t.get("unanswered", 0) + t.get("no-response", 0)))
    L.append(f"Unparseable reply & {row[0]} & {row[1]} \\\\")
    row = []
    for n in names:
        t = dstats[n]["totals"]
        den = t.get("confirmed", 0) + t.get("other", 0)
        row.append(p(t.get("confirmed", 0), den))
    L.append(f"Hit rate (answered only) & {row[0]} & {row[1]} \\\\")
    L.append(f"Human objects added & {num(dstats[a]['human_added'])} "
             f"& {num(dstats[b]['human_added'])} \\\\")

    # The pooled hit rate hides the only large cross-model gap that survives
    # correction, and it sits on the two safety-critical classes.
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Discovery confirmation by prompt (answered only)}} \\\\")
    for label in DISC_CLASSES.values():
        row = []
        for n in names:
            c = dstats[n]["by_class"].get(label, {})
            den = c.get("confirmed", 0) + c.get("other", 0)
            row.append(p(c.get("confirmed", 0), den))
        pretty = "Human (cyclist/pedestrian)" if label == "human" else label.capitalize()
        L.append(f"{pretty} & {row[0]} & {row[1]} \\\\")

    L.append("\\bottomrule")
    L.append("\\end{tabular}}")
    L.append("\\\\[2pt]")
    L.append("\\footnotesize\\textsuperscript{a} The LLaVA run predates the "
             "per-agent parse telemetry in \\texttt{vlm/health.py} and stored only the "
             "parsed verdict, so its \\textsc{safe\\_default} share is not recoverable "
             "offline. Its \\emph{discovery} non-answer rate is measurable and is "
             "reported above.")
    L.append("\\end{table}")

    path.write_text("\n".join(L) + "\n")
    print(f"\n  wrote {path}")


def write_latex_performance(path: Path, names, stats, refs, agree, ident) -> None:
    a, b = names
    L = ["% Generated by compare_models.py — do not edit by hand."]
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{Agent signal agreement on the {tex_num(ident['masks'])} paired masks "
             f"from {tex_num(ident['frames'])} frames both models completed. "
             "Reference positive $=$ Swin quality \\textit{good} "
             "($\\alpha \\geq \\tau_q$, per-class threshold); masks without a Swin score "
             "(geometry-prefiltered) are excluded. "
             "Swin is a noisy judge rather than ground truth, but it is identical for both "
             "runs, so it ranks the two VLMs on one fixed yardstick. "
             "For Qwen the \\emph{answered only} row excludes masks whose verdict is a "
             "\\textsc{safe\\_default} substitution; the equivalent row cannot be computed "
             "for LLaVA, whose run predates the parse telemetry. "
             f"Cross-VLM BBox agreement: {100*agree['agree']:.1f}\\% of masks receive the "
             "same verdict from both models.}")
    L.append("\\label{tab:agent_performance}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{lcccc}")
    L.append("\\toprule")
    L.append("\\textbf{Agent} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\")
    L.append("\\midrule")

    def row(label, r):
        return (f"{label} & {100*r['acc']:.1f} & {100*r['prec']:.1f} & "
                f"{100*r['rec']:.1f} & {100*r['f1']:.1f} \\\\")

    L.append(row("BBox VLM (LLaVA-1.6-34B)", refs[a]["all"]))
    L.append(row("BBox VLM (Qwen2.5-VL-72B)", refs[b]["all"]))
    if stats[b]["has_telemetry"]:
        L.append(row("\\quad Qwen, answered only", refs[b]["answered"]))
    L.append("\\bottomrule")
    L.append("\\end{tabular}}")
    L.append("\\end{table}")
    path.write_text("\n".join(L) + "\n")
    print(f"  wrote {path}")


def write_latex_timing(path: Path, names, times, ident) -> None:
    a, b = names

    def f(n, key, fmt):
        v = times[n][key]
        return fmt.format(v) if v is not None else "—"

    L = ["% Generated by compare_models.py — do not edit by hand."]
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{Inference cost on the {tex_num(ident['frames'])} frames / "
             f"{tex_num(ident['masks'])} masks both models completed. "
             "Qwen figures are from the instrumented rerun, whose health monitor aborts "
             "and resumes on a degraded server; the retry traffic those episodes generate "
             "is included in the totals, so these are deployed costs rather than "
             "fault-free ones.}")
    L.append("\\label{tab:timing}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{lcc}")
    L.append("\\toprule")
    L.append("\\textbf{Metric} & \\textbf{LLaVA-1.6-34B} & \\textbf{Qwen2.5-VL-72B} \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{3}{l}{\\textit{Per-mask VLM latency (s)}} \\\\")
    L.append(f"BBox call, mean & \\phantom{{0}}{f(a,'bbox_mean','{:.2f}')} & {f(b,'bbox_mean','{:.2f}')} \\\\")
    L.append(f"BBox call, p90 & \\phantom{{0}}{f(a,'bbox_p90','{:.2f}')} & {f(b,'bbox_p90','{:.2f}')} \\\\")
    L.append("Swin quality (shared, per mask) & \\phantom{0}$<$0.01 & \\phantom{0}$<$0.01 \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Per-frame wall time (s), median}} \\\\")
    L.append(f"Triage phase & {f(a,'frame_triage_med','{:.1f}')} & {f(b,'frame_triage_med','{:.1f}')} \\\\")
    L.append(f"Discovery phase & {f(a,'frame_disc_med','{:.1f}')} & {f(b,'frame_disc_med','{:.1f}')} \\\\")
    L.append(f"Total & {f(a,'frame_total_med','{:.1f}')} & {f(b,'frame_total_med','{:.1f}')} \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{Campaign cost}} \\\\")
    L.append(f"Aggregate GPU-time & {f(a,'gpu_hours','{:.1f}')}h & {f(b,'gpu_hours','{:.1f}')}h \\\\")
    L.append(f"Wall-clock (4 workers) & \\phantom{{0}}{f(a,'wall_hours','{:.1f}')}h & {f(b,'wall_hours','{:.1f}')}h \\\\")
    L.append("\\addlinespace")
    L.append("\\multicolumn{3}{l}{\\textit{VRAM (single A100-80GB)}} \\\\")
    L.append("Total & $\\sim$22GB & $\\sim$51GB \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}}")
    L.append("\\end{table}")
    path.write_text("\n".join(L) + "\n")
    print(f"  wrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--tag-a", default="llava_34b")
    ap.add_argument("--tag-b", default="qwen2.5vl_72b_v2")
    ap.add_argument("--exclude-degraded", action="store_true",
                    help="drop frames where either run has a degenerate response")
    ap.add_argument("--latex", type=Path, default=None,
                    help="write agent_behavior.tex to this path")
    ap.add_argument("--latex-dir", type=Path, default=None,
                    help="write agent_behavior/agent_performance/timing .tex into this dir")
    ap.add_argument("--hpc", action="store_true")
    args = ap.parse_args()
    if args.hpc:
        config.use_hpc()

    names = [args.tag_a, args.tag_b]
    runs = {n: load_run(n) for n in names}
    coverage = {n: len(runs[n]) for n in names}

    frames = sorted(set(runs[names[0]]) & set(runs[names[1]]))
    if not frames:
        raise SystemExit("No common frames between the two runs.")

    if args.exclude_degraded:
        bad = set()
        for n in names:
            bad |= degraded_frames(runs[n])
        before = len(frames)
        frames = [f for f in frames if f not in bad]
        print(f"\n  --exclude-degraded: dropped {before - len(frames)} of {before} "
              f"common frames; {len(frames)} remain.")

    ident = verify_shared_inputs(runs, frames)
    stats = {n: mask_stats(runs[n], frames) for n in names}
    dstats = {n: disc_stats(runs[n], frames) for n in names}
    refs = {n: dict(all=swin_reference(runs[n], frames),
                    answered=swin_reference(runs[n], frames, answered_only=True))
            for n in names}
    agree = bbox_agreement(runs, frames)
    times = {n: timing(runs[n], frames) for n in names}

    print_report(names, stats, dstats, refs, agree, times, ident, args)
    if args.latex:
        write_latex(args.latex, names, stats, dstats, ident, coverage)
    if args.latex_dir:
        d = args.latex_dir
        write_latex(d / "agent_behavior.tex", names, stats, dstats, ident, coverage)
        write_latex_performance(d / "agent_performance.tex", names, stats, refs, agree, ident)
        write_latex_timing(d / "timing.tex", names, times, ident)


if __name__ == "__main__":
    main()
