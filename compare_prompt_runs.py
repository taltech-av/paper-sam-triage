#!/usr/bin/env python3
"""
Within-model prompt-sensitivity analysis.

Compares two runs of the SAME VLM under different prompts on the same frames
(qwen2.5vl_72b_old_prompt vs qwen2.5vl_72b, 1,000 common frames), with an
optional second-model reference run (llava_34b) on the same subset. Everything
is computed from stored per-frame JSON — no new VLM inference or training.

The prompt revision (commit 6770195) also changed the triage rule, so stored
`triage` fields are NOT comparable across runs. This script recomputes triage
for every run with the CURRENT rule (core.triage.triage) from stored agent
outputs, isolating the prompt effect from the rule change.

Discovery note: the commit changed the sign and human confirm-prompts but left
the vehicle confirm-prompt untouched, so vehicle discovery acts as a control —
its confirmation rate should be stable across runs while sign/human shift.

Usage:
    python compare_prompt_runs.py
    python compare_prompt_runs.py --tag-old qwen2.5vl_72b_old_prompt --tag-new qwen2.5vl_72b
    python compare_prompt_runs.py --latex paper/tables/prompt_sensitivity.tex
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import config
from core.triage import triage

SEP  = "─" * 72
SEP2 = "═" * 72
CLASSES = ["vehicle", "sign", "cyclist", "pedestrian"]
BBOX_VERDICTS = ["valid", "invalid", "background"]
TRIAGE_OUTCOMES = ["accept", "reject", "human_review", "refine"]
DISC_CLASSES = {1: "vehicle", 2: "sign", 3: "human"}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_run(tag: str) -> dict[str, dict]:
    """Return {frame_id: record} for a run tag."""
    results_dir = config.DATA_ROOT / "vlm" / tag / "results"
    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No results found in {results_dir}")
    records = {}
    for f in files:
        rec = json.loads(f.read_text())
        records[rec["frame_id"]] = rec
    return records


def join_masks(runs: dict[str, dict[str, dict]], frames: list[str]) -> dict:
    """Return {(frame_id, mask_id): {run_name: mask_dict}} on common frames."""
    joined: dict = defaultdict(dict)
    for name, records in runs.items():
        for fid in frames:
            for m in records[fid]["masks"]:
                joined[(fid, m["mask_id"])][name] = m
    return joined


def join_discoveries(runs: dict[str, dict[str, dict]], frames: list[str]) -> dict:
    """Return {(frame_id, bbox_384): {run_name: disc_dict}}.

    Candidates are Swin-proposed and deterministic given the shared Swin model,
    so (frame_id, bbox_384) identifies the same candidate across runs.
    """
    joined: dict = defaultdict(dict)
    for name, records in runs.items():
        for fid in frames:
            for d in records[fid].get("discovered", []):
                joined[(fid, tuple(d["bbox_384"]))][name] = d
    return joined


# ── Recomputed triage ─────────────────────────────────────────────────────────

def recomputed_triage(mask: dict) -> str:
    a = mask["agents"]
    return triage(
        bbox_out=a.get("bbox"),
        quality_out=a.get("quality"),
        failure_mode_out=a.get("failure_mode"),
        correction_out=a.get("correction"),
        consistency_out=a.get("consistency"),
    ).decision


# ── Per-run statistics on the common mask set ────────────────────────────────

def mask_stats(joined: dict, run_name: str) -> dict:
    bbox_counts = defaultdict(int)
    triage_counts = defaultdict(int)
    reject_by_class = defaultdict(int)
    total_by_class = defaultdict(int)
    pixels_total = 0
    pixels_rejected = 0
    n = 0

    for masks in joined.values():
        m = masks.get(run_name)
        if m is None:
            continue
        n += 1
        cls = m["class_name"]
        total_by_class[cls] += 1
        pixels_total += m["pixel_count"]

        verdict = m["agents"].get("bbox")
        if verdict is not None:
            bbox_counts[verdict] += 1

        decision = recomputed_triage(m)
        triage_counts[decision] += 1
        if decision == "reject":
            reject_by_class[cls] += 1
            pixels_rejected += m["pixel_count"]

    return {
        "n": n,
        "bbox": bbox_counts,
        "triage": triage_counts,
        "reject_by_class": reject_by_class,
        "total_by_class": total_by_class,
        "pixels_total": pixels_total,
        "pixels_rejected": pixels_rejected,
    }


def disc_stats(joined_disc: dict, run_name: str) -> dict:
    total = defaultdict(int)
    confirmed = defaultdict(int)
    human_added = 0
    for cands in joined_disc.values():
        d = cands.get(run_name)
        if d is None:
            continue
        grp = DISC_CLASSES.get(d["swin_class"], "other")
        total[grp] += 1
        if d["confirmed"]:
            confirmed[grp] += 1
            if d["class_name"] in ("cyclist", "pedestrian"):
                human_added += 1
    return {"total": total, "confirmed": confirmed, "human_added": human_added}


# ── Pairwise comparisons ──────────────────────────────────────────────────────

def transition_matrix(joined: dict, run_a: str, run_b: str, field: str) -> dict:
    """field: 'bbox' (agent verdict) or 'triage' (recomputed)."""
    matrix = defaultdict(int)
    for masks in joined.values():
        ma, mb = masks.get(run_a), masks.get(run_b)
        if ma is None or mb is None:
            continue
        if field == "bbox":
            va, vb = ma["agents"].get("bbox"), mb["agents"].get("bbox")
        else:
            va, vb = recomputed_triage(ma), recomputed_triage(mb)
        if va is None or vb is None:
            continue
        matrix[(va, vb)] += 1
    return matrix


def agreement(matrix: dict) -> float:
    total = sum(matrix.values())
    same = sum(v for (a, b), v in matrix.items() if a == b)
    return 100.0 * same / total if total else 0.0


def sanity_check(joined: dict, run_a: str, run_b: str) -> tuple[int, int]:
    """Swin/LiDAR signals are VLM-independent and must match across runs."""
    checked = mismatched = 0
    for masks in joined.values():
        ma, mb = masks.get(run_a), masks.get(run_b)
        if ma is None or mb is None:
            continue
        checked += 1
        if (abs(ma["scores"]["swin_agreement"] - mb["scores"]["swin_agreement"]) > 1e-6
                or ma["agents"].get("consistency") != mb["agents"].get("consistency")):
            mismatched += 1
    return checked, mismatched


# ── Report ────────────────────────────────────────────────────────────────────

def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:5.1f}%" if total else "    —"


def print_distribution_table(stats: dict[str, dict], run_names: list[str]) -> None:
    header = f"{'':28s}" + "".join(f"{name:>18s}" for name in run_names)
    print(header)
    print(SEP)
    print("BBox verdicts")
    for v in BBOX_VERDICTS:
        row = f"  {v:26s}"
        for name in run_names:
            s = stats[name]
            row += f"{pct(s['bbox'][v], sum(s['bbox'].values())):>18s}"
        print(row)
    print("Triage outcomes (current rule, recomputed)")
    for t in TRIAGE_OUTCOMES:
        row = f"  {t:26s}"
        for name in run_names:
            s = stats[name]
            row += f"{pct(s['triage'][t], sum(s['triage'].values())):>18s}"
        print(row)
    print("Per-class rejection rate")
    for c in CLASSES:
        row = f"  {c:26s}"
        for name in run_names:
            s = stats[name]
            row += f"{pct(s['reject_by_class'][c], s['total_by_class'][c]):>18s}"
        print(row)
    print("Annotation effect")
    row = f"  {'pixels deleted':26s}"
    for name in run_names:
        s = stats[name]
        row += f"{pct(s['pixels_rejected'], s['pixels_total']):>18s}"
    print(row)


def print_transition(matrix: dict, labels: list[str], title: str) -> None:
    total = sum(matrix.values())
    print(f"\n{title}  (rows = old, cols = new, n = {total})")
    print(f"{'':14s}" + "".join(f"{l:>12s}" for l in labels))
    for a in labels:
        row = f"{a:14s}"
        for b in labels:
            row += f"{matrix[(a, b)]:>12d}"
        print(row)
    print(f"  verdict unchanged: {agreement(matrix):.1f}%")


def print_discovery(dstats: dict[str, dict], run_names: list[str]) -> None:
    print(f"{'':28s}" + "".join(f"{name:>18s}" for name in run_names))
    print(SEP)
    for grp in ("vehicle", "sign", "human"):
        note = "  [control: prompt unchanged]" if grp == "vehicle" else ""
        row = f"  {grp + ' confirmed':26s}"
        for name in run_names:
            d = dstats[name]
            row += f"{pct(d['confirmed'][grp], d['total'][grp]):>18s}"
        print(row + note)
    row = f"  {'human objects added':26s}"
    for name in run_names:
        row += f"{dstats[name]['human_added']:>18d}"
    print(row)


# ── LaTeX output ──────────────────────────────────────────────────────────────

def write_latex(path: Path, stats: dict, dstats: dict, run_names: list[str],
                col_labels: dict[str, str], n_frames: int, bbox_flip: float) -> None:
    cols = "l" + "c" * len(run_names)

    def row(label: str, values: list[str]) -> str:
        return label + " & " + " & ".join(values) + r" \\"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Within-model prompt sensitivity on the " f"{n_frames}" r"-frame"
        r" overlap subset. The revised prompts (Section~\ref{sec:methodology})"
        r" are applied to the same Qwen2.5-VL-72B model on identical frames;"
        r" LLaVA-1.6-34B (revised prompts) is shown on the same subset for"
        r" reference. Triage outcomes are recomputed offline under the current"
        r" deterministic rule for all columns, so differences reflect prompt"
        r" changes only. The vehicle discovery prompt was not revised and serves"
        r" as a control.}",
        r"\label{tab:prompt_sensitivity}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
        row(r"\textbf{Metric}",
            [r"\textbf{" + col_labels[n] + "}" for n in run_names]),
        r"\midrule",
        r"\multicolumn{" + str(len(run_names) + 1) + r"}{l}{\textit{BBox VLM verdicts}} \\",
    ]
    for v, label in [("valid", "Valid"), ("invalid", "Invalid"), ("background", "Background")]:
        lines.append(row(label, [
            pct(stats[n]["bbox"][v], sum(stats[n]["bbox"].values())).strip().replace("%", r"\%")
            for n in run_names]))
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(len(run_names) + 1)
                 + r"}{l}{\textit{Triage outcomes (current rule, recomputed)}} \\")
    for t, label in [("accept", "Accept"), ("reject", "Reject"), ("human_review", "Human review")]:
        lines.append(row(label, [
            pct(stats[n]["triage"][t], sum(stats[n]["triage"].values())).strip().replace("%", r"\%")
            for n in run_names]))
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(len(run_names) + 1)
                 + r"}{l}{\textit{Per-class rejection rate}} \\")
    for c in CLASSES:
        lines.append(row(c.capitalize(), [
            pct(stats[n]["reject_by_class"][c], stats[n]["total_by_class"][c]).strip().replace("%", r"\%")
            for n in run_names]))
    lines.append(r"\addlinespace")
    lines.append(r"\multicolumn{" + str(len(run_names) + 1)
                 + r"}{l}{\textit{Discovery confirmation rate}} \\")
    for grp, label in [("vehicle", "Vehicle (control)"), ("sign", "Sign"), ("human", "Human")]:
        lines.append(row(label, [
            pct(dstats[n]["confirmed"][grp], dstats[n]["total"][grp]).strip().replace("%", r"\%")
            for n in run_names]))
    lines.append(row("Human objects added",
                     [str(dstats[n]["human_added"]) for n in run_names]))
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    path.write_text("\n".join(lines) + "\n")
    print(f"\nLaTeX table written to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--tag-old", default="qwen2.5vl_72b_old_prompt",
                        help="Run with the original prompts")
    parser.add_argument("--tag-new", default="qwen2.5vl_72b",
                        help="Run with the revised prompts (same model)")
    parser.add_argument("--ref-tag", default="llava_34b",
                        help="Second-model reference run on the same frames ('' to disable)")
    parser.add_argument("--latex", type=Path, default=None,
                        help="Write a LaTeX table to this path")
    parser.add_argument("--hpc", action="store_true", help="Use HPC data paths")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    runs = {"old": load_run(args.tag_old), "new": load_run(args.tag_new)}
    col_labels = {"old": "Qwen (original)", "new": "Qwen (revised)"}
    if args.ref_tag:
        runs["ref"] = load_run(args.ref_tag)
        col_labels["ref"] = "LLaVA (revised)"

    frames = sorted(set.intersection(*(set(r) for r in runs.values())))
    run_names = list(runs)

    print(SEP2)
    print("Prompt-sensitivity comparison")
    for name in run_names:
        info = runs[name][frames[0]]["run_info"]
        print(f"  {name:4s}: {info['model']:16s} tag={args.__dict__['tag_' + name] if name != 'ref' else args.ref_tag:28s}"
              f" run={info['timestamp'][:10]}")
    print(f"  common frames: {len(frames)}")
    print(SEP2)

    joined = join_masks(runs, frames)
    joined_disc = join_discoveries(runs, frames)

    # Sanity: VLM-independent signals must be identical between the Qwen runs
    checked, mismatched = sanity_check(joined, "old", "new")
    print(f"\nSanity check (Swin agreement + LiDAR consistency, old vs new): "
          f"{checked} masks joined, {mismatched} mismatched")
    only_old = sum(1 for m in joined.values() if "old" in m and "new" not in m)
    only_new = sum(1 for m in joined.values() if "new" in m and "old" not in m)
    if only_old or only_new:
        print(f"  WARNING: {only_old} masks only in old run, {only_new} only in new run")

    stats = {name: mask_stats(joined, name) for name in run_names}
    dstats = {name: disc_stats(joined_disc, name) for name in run_names}

    print(f"\n{SEP2}\nPer-run distributions on {stats['new']['n']} common masks\n{SEP2}")
    print_distribution_table(stats, run_names)

    bbox_matrix = transition_matrix(joined, "old", "new", "bbox")
    print_transition(bbox_matrix, BBOX_VERDICTS,
                     "BBox verdict transitions, Qwen original → revised prompt")
    triage_matrix = transition_matrix(joined, "old", "new", "triage")
    print_transition(triage_matrix, TRIAGE_OUTCOMES,
                     "Recomputed triage transitions, Qwen original → revised prompt")

    print(f"\n{SEP2}\nDiscovery confirmation on common Swin candidates\n{SEP2}")
    print_discovery(dstats, run_names)

    # Effect-size framing: within-model prompt effect vs between-model effect
    within = 100.0 - agreement(bbox_matrix)
    print(f"\n{SEP2}\nEffect size\n{SEP2}")
    print(f"  Within-model (Qwen, prompt revision): {within:.1f}% of BBox verdicts flip")
    if "ref" in runs:
        for qwen_run, label in [("old", "original"), ("new", "revised")]:
            m = transition_matrix(joined, qwen_run, "ref", "bbox")
            print(f"  Between-model (Qwen {label} vs LLaVA):  {100.0 - agreement(m):.1f}% disagree")

    if args.latex:
        write_latex(args.latex, stats, dstats, run_names, col_labels,
                    len(frames), within)


if __name__ == "__main__":
    main()
