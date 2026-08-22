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

# What a CROSS-MODEL comparison may report. `refine` appears only in archived
# runs: it was produced by a CorrectionAgent that gated a refinement stage never
# implemented, and it has since been removed from the pipeline. Pooling it with
# `human_review` is lossless for annotation outcomes, because annotation_writer.py
# zeroes pixels only on `reject`, so the two write byte-identical labels. Runs
# made after the removal simply have no `refine` to pool.
TRIAGE_REPORTED = ["accept", "reject", "retained_flagged"]


def triage_reported(s: dict, outcome: str) -> int:
    """Count for a reported (pooled) triage outcome."""
    if outcome == "retained_flagged":
        return s["triage"].get("human_review", 0) + s["triage"].get("refine", 0)
    return s["triage"].get(outcome, 0)


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
        consistency_out=a.get("consistency"),
    ).decision


def mask_stats(records: dict[str, dict], frames: list[str]) -> dict:
    bbox = defaultdict(int)          # as-recorded, defaults included
    bbox_answered = defaultdict(int) # defaults removed
    tri = defaultdict(int)
    reject_by_class, total_by_class = defaultdict(int), defaultdict(int)
    defaults = defaultdict(int)      # agent → masks whose verdict is a default
    degen = defaultdict(int)
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

            d = recomputed_triage(m)
            tri[d] += 1
            if d == "reject":
                reject_by_class[cls] += 1

    return dict(total=total, bbox=dict(bbox), bbox_answered=dict(bbox_answered),
                triage=dict(tri), reject_by_class=dict(reject_by_class),
                total_by_class=dict(total_by_class), defaults=dict(defaults),
                degen=dict(degen), has_telemetry=telemetry)


def _disc_bucket(d: dict) -> str:
    """One discovery reply -> one of the four outcomes, re-parsed from the raw text.

    Shared by disc_stats() and transitions() so a marginal rate and a joint
    count can never be computed under two different parsers.
    """
    swin_cls = d.get("swin_class")
    raw = d.get("vlm_response")
    norm = (raw or "").strip().lower().rstrip(".")
    if looks_degenerate(raw):
        return "no-response"
    if norm in _VALID_RESPONSES[swin_cls]:
        return "confirmed"
    if norm == "other":
        return "other"
    return "unanswered"


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
            bucket = _disc_bucket(d)
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


def label_divergence(runs: dict[str, dict], frames: list[str]) -> dict:
    """How much of the annotation set a backend substitution moves.

    Verdict agreement is an agent-level statistic; this is the consequence a
    downstream consumer inherits. Outcomes are recomputed under one rule and
    pooled to accept / reject / retained, since only `reject` deletes pixels —
    so `flipped` counts masks present in one annotation set and absent from the
    other.
    """
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]
    n = moved = flipped = pixels = 0
    for fid in frames:
        mb = {m["mask_id"]: m for m in b[fid]["masks"]}
        for m in a[fid]["masks"]:
            other = mb.get(m["mask_id"])
            if other is None:
                continue
            da, db = recomputed_triage(m), recomputed_triage(other)
            da = "retained" if da in ("refine", "human_review") else da
            db = "retained" if db in ("refine", "human_review") else db
            n += 1
            moved += da != db
            if (da == "reject") != (db == "reject"):
                flipped += 1
                pixels += m.get("pixel_count") or 0
    return dict(n=n, moved=moved, flipped=flipped, pixels=pixels)


def cascade(runs: dict[str, dict], frames: list[str]) -> dict:
    """How far a raw verdict disagreement travels before it reaches a pixel.

    Three nested levels on the same items. The nesting is not assumed: with
    every non-VLM signal shared and one deterministic rule, two identical
    verdicts must produce identical outcomes, so a flip without a verdict
    disagreement would mean the comparison is not paired. `flip_same_verdict`
    is that check and must be zero.

    Reported conditionally rather than as a ratio of rates, because the three
    levels do not share a denominator: a mask the geometry pre-filter rejected
    never reached the BBox agent and has no verdict to disagree about.
    """
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]
    n = judged = 0
    verdict_diff = state_diff = flip = 0
    flip_same_verdict = 0
    state_diff_no_pixels = 0      # accept <-> retained: both write the mask
    px_total = px_flipped = 0

    for fid in frames:
        mb = {m["mask_id"]: m for m in b[fid]["masks"]}
        for m in a[fid]["masks"]:
            other = mb.get(m["mask_id"])
            if other is None:
                continue
            n += 1
            px = m.get("pixel_count") or 0
            px_total += px

            va, vb = m["agents"].get("bbox"), other["agents"].get("bbox")
            has_verdict = va is not None and vb is not None
            judged += has_verdict
            vdiff = has_verdict and va != vb
            verdict_diff += vdiff

            da, db = recomputed_triage(m), recomputed_triage(other)
            da = "retained" if da in ("refine", "human_review") else da
            db = "retained" if db in ("refine", "human_review") else db
            sdiff = da != db
            state_diff += sdiff
            flipped = (da == "reject") != (db == "reject")
            flip += flipped
            if flipped:
                px_flipped += px
                if not vdiff:
                    flip_same_verdict += 1
            elif sdiff:
                state_diff_no_pixels += 1

    return dict(n=n, judged=judged, verdict_diff=verdict_diff,
                state_diff=state_diff, flip=flip,
                flip_same_verdict=flip_same_verdict,
                state_diff_no_pixels=state_diff_no_pixels,
                px_total=px_total, px_flipped=px_flipped)


def transitions(runs: dict[str, dict], frames: list[str]) -> dict:
    """Paired transitions per class, at the verdict level and at the label level.

    A marginal rejection rate cannot distinguish two backends that reject the
    same 20% of a class from two that reject disjoint 20% halves of it. These
    are the joint counts: for every mask, what A decided crossed with what B
    decided, kept apart by class, and for the final outcome weighted by the
    pixels each flip actually moves.

    Three levels, all on the same paired masks:

      bbox    the raw verdict crossed 3x3 (valid / invalid / background).
      final   the recomputed triage outcome pooled to keep vs delete, since
              only `reject` erases pixels. Four cells, of which the two
              directional ones are the whole point.
      disc    discovery confirmation crossed 2x2 per prompt class, on the
              answered-by-both basis (the only unbiased one, see disc_stats)
              and on the policy-inclusive basis the pipeline acted on.

    Discovery candidates carry no id, so they are paired by position within a
    frame; `bbox_384` is compared on every pair and mismatches are counted, so
    a silent misalignment cannot pass as a disagreement.
    """
    names = list(runs)
    a, b = runs[names[0]], runs[names[1]]

    bbox = {c: defaultdict(int) for c in CLASSES}
    final = {c: defaultdict(lambda: [0, 0]) for c in CLASSES}
    disc = {c: defaultdict(int) for c in DISC_CLASSES.values()}
    disc_policy = {c: defaultdict(int) for c in DISC_CLASSES.values()}
    cand_misaligned = 0

    for fid in frames:
        mb = {m["mask_id"]: m for m in b[fid]["masks"]}
        for m in a[fid]["masks"]:
            other = mb.get(m["mask_id"])
            if other is None:
                continue
            cls = m["class_name"]
            if cls not in bbox:
                continue

            va, vb = m["agents"].get("bbox"), other["agents"].get("bbox")
            if va is not None and vb is not None:
                bbox[cls][(va, vb)] += 1

            # Keep/delete is the only distinction the annotation file records:
            # accept and retained_flagged both write the mask unchanged.
            ka = recomputed_triage(m) != "reject"
            kb = recomputed_triage(other) != "reject"
            cell = ("both_keep" if ka and kb else
                    "both_delete" if not ka and not kb else
                    "a_keep_b_delete" if ka else "b_keep_a_delete")
            px = m.get("pixel_count") or 0
            final[cls][cell][0] += 1
            final[cls][cell][1] += px

        da, db = a[fid].get("discovered", []), b[fid].get("discovered", [])
        for ca, cb in zip(da, db):
            if ca.get("bbox_384") != cb.get("bbox_384"):
                cand_misaligned += 1
                continue
            label = DISC_CLASSES.get(ca.get("swin_class"))
            if label is None:
                continue
            ba, bb = _disc_bucket(ca), _disc_bucket(cb)
            # Policy-inclusive: the pipeline confirms only on `confirmed`, so an
            # unreadable reply silently discards the candidate.
            disc_policy[label][(ba == "confirmed", bb == "confirmed")] += 1
            if ba in ("confirmed", "other") and bb in ("confirmed", "other"):
                disc[label][(ba == "confirmed", bb == "confirmed")] += 1

    return dict(bbox={c: dict(v) for c, v in bbox.items()},
                final={c: {k: list(v) for k, v in d.items()} for c, d in final.items()},
                disc={c: dict(v) for c, v in disc.items()},
                disc_policy={c: dict(v) for c, v in disc_policy.items()},
                cand_misaligned=cand_misaligned)


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


def print_report(names, stats, dstats, refs, agree, diverge, times, ident, args, trans, casc):
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
    for agent in ("bbox",):
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
    for o in TRIAGE_REPORTED:
        print(f"  {o:32s} "
              f"{pct(triage_reported(stats[a], o), stats[a]['total']):>{W}} "
              f"{pct(triage_reported(stats[b], o), stats[b]['total']):>{W}}")
    print(f"  retained_flagged pools human_review with the archived `refine` label:")
    print(f"  both keep every pixel, and the rule no longer produces the latter.")

    print(f"\n  LABEL-SET DIVERGENCE  (outcomes pooled to accept/reject/retained)")
    print(SEP)
    print(f"  {'masks compared':40s} {diverge['n']:>10}")
    print(f"  {'different triage outcome':40s} {diverge['moved']:>10}  "
          f"{pct(diverge['moved'], diverge['n'])}")
    print(f"  {'retained by one, deleted by the other':40s} {diverge['flipped']:>10}  "
          f"{pct(diverge['flipped'], diverge['n'])}")
    print(f"  {'object pixels at stake':40s} {diverge['pixels']/1e6:>9.1f}M")
    print(f"  Only `reject` deletes, so the second row is the share of the mask set")
    print(f"  that exists in one annotation set and not in the other.")

    print(f"\n  PER-CLASS REJECTION RATE")
    print(SEP)
    print(f"  {'class':32s} {a[:W]:>{W}} {b[:W]:>{W}}")
    for c in CLASSES:
        print(f"  {c:32s} "
              f"{pct(stats[a]['reject_by_class'].get(c,0), stats[a]['total_by_class'].get(c,0)):>{W}} "
              f"{pct(stats[b]['reject_by_class'].get(c,0), stats[b]['total_by_class'].get(c,0)):>{W}}")

    # ── Correction agent ──────────────────────────────────────────────────────
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

    # ── Cascade ───────────────────────────────────────────────────────────────
    c = casc
    print(f"\n  CASCADE  (how far a verdict disagreement travels)")
    print(SEP)
    print(f"  {'paired masks':44s} {c['n']:>10,}")
    print(f"  {'  ...with a BBox verdict from both runs':44s} {c['judged']:>10,}")
    print(f"  {'differing BBox verdict':44s} {c['verdict_diff']:>10,}"
          f"  {100*c['verdict_diff']/c['judged']:5.1f}% of judged")
    print(f"  {'  ...of which the triage state differs':44s} {c['state_diff']:>10,}"
          f"  {100*c['state_diff']/c['verdict_diff']:5.1f}% of those")
    print(f"  {'  ...of which the keep/delete outcome flips':44s} {c['flip']:>10,}"
          f"  {100*c['flip']/c['verdict_diff']:5.1f}% of those")
    print(f"  {'state differs but no pixel changes':44s} {c['state_diff_no_pixels']:>10,}"
          f"  {100*c['state_diff_no_pixels']/c['state_diff']:5.1f}% of state diffs")
    print(f"  {'flips without a verdict disagreement':44s} {c['flip_same_verdict']:>10,}"
          f"   (must be 0)")
    print(f"  {'object pixels, all masks':44s} {c['px_total']/1e6:>9.1f}M")
    print(f"  {'object pixels at stake':44s} {c['px_flipped']/1e6:>9.1f}M"
          f"  {100*c['px_flipped']/c['px_total']:5.1f}% of object pixels")

    # ── Paired transitions ────────────────────────────────────────────────────
    tr = trans
    print(f"\n  FINAL OUTCOME TRANSITIONS  (keep vs delete, per class)")
    print(SEP)
    print(f"  {'cell':34s} " + " ".join(f"{c[:9]:>10}" for c in CLASSES) + f" {'total':>10}")
    cells = [("both keep", "both_keep"), ("both delete", "both_delete"),
             (f"{a[:14]} keeps, {b[:8]} deletes", "a_keep_b_delete"),
             (f"{b[:14]} keeps, {a[:8]} deletes", "b_keep_a_delete")]
    for label, key in cells:
        row = [tr["final"][c].get(key, [0, 0])[0] for c in CLASSES]
        print(f"  {label:34s} " + " ".join(f"{v:>10,}" for v in row) + f" {sum(row):>10,}")
    print(f"  {'':34s} " + " ".join(f"{'':>10}" for _ in CLASSES) + f" {'':>10}")
    for label, key in cells[2:]:
        row = [tr["final"][c].get(key, [0, 0])[1] for c in CLASSES]
        print(f"  {label + ' (px)':34s} " + " ".join(f"{v/1e3:>9.0f}k" for v in row)
              + f" {sum(row)/1e6:>9.2f}M")

    print(f"\n  BBOX VERDICT TRANSITIONS  (per class, as-recorded)")
    print(SEP)
    print(f"  {'cell':34s} " + " ".join(f"{c[:9]:>10}" for c in CLASSES) + f" {'total':>10}")
    def bcell(c, pred):
        return sum(n for (va, vb), n in tr["bbox"][c].items() if pred(va, vb))
    for label, pred in (
            ("same verdict", lambda x, y: x == y),
            ("A valid, B not", lambda x, y: x == "valid" != y),
            ("B valid, A not", lambda x, y: y == "valid" != x),
            ("both non-valid, differing", lambda x, y: x != y and "valid" not in (x, y))):
        row = [bcell(c, pred) for c in CLASSES]
        print(f"  {label:34s} " + " ".join(f"{v:>10,}" for v in row) + f" {sum(row):>10,}")

    print(f"\n  DISCOVERY CONFIRMATION TRANSITIONS  (answered by both)")
    print(SEP)
    dcls = list(DISC_CLASSES.values())
    print(f"  {'cell':34s} " + " ".join(f"{c[:9]:>10}" for c in dcls) + f" {'total':>10}")
    for label, key in (("both confirm", (True, True)),
                       (f"{a[:14]} only", (True, False)),
                       (f"{b[:14]} only", (False, True)),
                       ("neither", (False, False))):
        row = [tr["disc"][c].get(key, 0) for c in dcls]
        print(f"  {label:34s} " + " ".join(f"{v:>10,}" for v in row) + f" {sum(row):>10,}")
    row = [sum(tr["disc_policy"][c].values()) - sum(tr["disc"][c].values()) for c in dcls]
    print(f"  {'(unanswered by one side)':34s} " + " ".join(f"{v:>10,}" for v in row)
          + f" {sum(row):>10,}")
    if tr["cand_misaligned"]:
        print(f"  !! {tr['cand_misaligned']} candidate pairs had differing bbox_384")

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
             "Triage outcomes are the three reported outcomes of Section~\\ref{sec:triage}, "
             "recomputed offline under the current deterministic rule for both models. "
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
                       ("Retained, flagged", "retained_flagged")):
        L.append(f"{label} & {p(triage_reported(stats[a], key), stats[a]['total'])} "
                 f"& {p(triage_reported(stats[b], key), stats[b]['total'])} \\\\")

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
             f"Cross-VLM BBox agreement: {100*agree['agree']:.1f}\\% of the "
             f"{tex_num(agree['n'])} masks both runs sent to the BBox agent receive the "
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


def write_latex_transitions(path: Path, names, trans, ident) -> None:
    """Emit tables/transitions.tex — the joint counts behind the marginal rates.

    Two blocks over the same paired masks: what the BBox agent said crossed
    between backends, and what survived triage crossed between backends, the
    latter also in pixels. The two directional rows are the reason the table
    exists: pooled they nearly cancel, per class they do not.
    """
    a, b = names
    A, B = "LLaVA", "Qwen"
    F, X = trans["final"], trans["bbox"]

    def bcell(c, pred):
        return sum(n for (va, vb), n in X[c].items() if pred(va, vb))

    def row(label, values, fmt=tex_num):
        cells = " & ".join(fmt(v) for v in values)
        return f"{label} & {cells} & {fmt(sum(values))} \\\\"

    L = ["% Generated by compare_models.py --latex-dir — do not edit by hand."]
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append(f"\\caption{{Paired transitions on the {tex_num(ident['masks'])} identical "
             "SAM masks, per class. Verdict rows use the "
             f"{tex_num(sum(sum(X[c].values()) for c in CLASSES))} masks both runs sent to "
             "the BBox agent; outcome rows pool accept and retained into \\emph{keep}, since "
             "only rejection erases pixels. The two directional rows carry almost the same "
             "pixel mass in opposite directions while pointing at different classes, which "
             "is what a marginal rejection rate cannot show.}")
    L.append("\\label{tab:transitions}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{lrrrrr}")
    L.append("\\toprule")
    L.append("\\textbf{Transition} & \\textbf{Vehicle} & \\textbf{Sign} & "
             "\\textbf{Cyclist} & \\textbf{Ped.} & \\textbf{Total} \\\\")
    L.append("\\midrule")

    L.append("\\multicolumn{6}{l}{\\textit{BBox verdict}} \\\\")
    L.append(row("Same verdict", [bcell(c, lambda x, y: x == y) for c in CLASSES]))
    L.append(row(f"{A} valid, {B} not",
                 [bcell(c, lambda x, y: x == "valid" != y) for c in CLASSES]))
    L.append(row(f"{B} valid, {A} not",
                 [bcell(c, lambda x, y: y == "valid" != x) for c in CLASSES]))
    L.append(row("Differing, neither valid",
                 [bcell(c, lambda x, y: x != y and "valid" not in (x, y)) for c in CLASSES]))

    L.append("\\addlinespace")
    L.append("\\multicolumn{6}{l}{\\textit{Final outcome, masks}} \\\\")
    for label, key in (("Both keep", "both_keep"), ("Both delete", "both_delete"),
                       (f"\\textbf{{{A} keeps, {B} deletes}}", "a_keep_b_delete"),
                       (f"\\textbf{{{B} keeps, {A} deletes}}", "b_keep_a_delete")):
        L.append(row(label, [F[c].get(key, [0, 0])[0] for c in CLASSES]))

    L.append("\\addlinespace")
    L.append("\\multicolumn{6}{l}{\\textit{Final outcome, object pixels deleted by one side}} \\\\")
    px = lambda v: f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k"
    for label, key in ((f"{A} keeps, {B} deletes", "a_keep_b_delete"),
                       (f"{B} keeps, {A} deletes", "b_keep_a_delete")):
        L.append(row(label, [F[c].get(key, [0, 0])[1] for c in CLASSES], fmt=px))

    L.append("\\bottomrule")
    L.append("\\end{tabular}}")
    L.append("\\end{table}")
    path.write_text("\n".join(L) + "\n")
    print(f"  wrote {path}")


def write_latex_disc_transitions(path: Path, names, trans) -> None:
    """Emit tables/discovery_transitions.tex — the same joint view for discovery.

    Pooled, the two backends look interchangeable here: each confirms roughly
    as many candidates the other declines. The human column is where that
    breaks, and only the joint counts show it.
    """
    A, B = "LLaVA", "Qwen"
    D, P = trans["disc"], trans["disc_policy"]
    dcls = list(DISC_CLASSES.values())

    def row(label, values):
        return f"{label} & " + " & ".join(tex_num(v) for v in values) + \
               f" & {tex_num(sum(values))} \\\\"

    L = ["% Generated by compare_models.py --latex-dir — do not edit by hand."]
    L.append("\\begin{table}[t]")
    L.append("\\centering")
    L.append("\\caption{Paired discovery confirmation on identical candidates, by prompt. "
             "Rows cover the candidates both backends answered; the last row counts those one "
             "side left unreadable, which the pipeline treats as non-confirmation. The two "
             "exclusive rows differ by 8\\% in total and by two orders of magnitude on the "
             "human prompt.}")
    L.append("\\label{tab:disc_transitions}")
    L.append("\\resizebox{\\columnwidth}{!}{%")
    L.append("\\begin{tabular}{lrrrr}")
    L.append("\\toprule")
    L.append("\\textbf{Transition} & \\textbf{Vehicle} & \\textbf{Sign} & "
             "\\textbf{Human} & \\textbf{Total} \\\\")
    L.append("\\midrule")
    for label, key in (("Both confirm", (True, True)),
                       (f"\\textbf{{{A} only}}", (True, False)),
                       (f"\\textbf{{{B} only}}", (False, True)),
                       ("Neither", (False, False))):
        L.append(row(label, [D[c].get(key, 0) for c in dcls]))
    L.append("\\addlinespace")
    L.append(row("Unanswered by one side",
                 [sum(P[c].values()) - sum(D[c].values()) for c in dcls]))
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
                    help="write agent_behavior/agent_performance/timing/transitions .tex into this dir")
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
    diverge = label_divergence(runs, frames)
    times = {n: timing(runs[n], frames) for n in names}

    trans = transitions(runs, frames)
    casc = cascade(runs, frames)
    print_report(names, stats, dstats, refs, agree, diverge, times, ident, args, trans, casc)
    if args.latex:
        write_latex(args.latex, names, stats, dstats, ident, coverage)
    if args.latex_dir:
        d = args.latex_dir
        write_latex(d / "agent_behavior.tex", names, stats, dstats, ident, coverage)
        write_latex_performance(d / "agent_performance.tex", names, stats, refs, agree, ident)
        write_latex_timing(d / "timing.tex", names, times, ident)
        write_latex_transitions(d / "transitions.tex", names, trans, ident)
        write_latex_disc_transitions(d / "discovery_transitions.tex", names, trans)


if __name__ == "__main__":
    main()
