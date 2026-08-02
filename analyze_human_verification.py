#!/usr/bin/env python3
"""
Score the pipeline's agents against human verdicts from the visin labeling platform.

Input is a job export (CSV, long format: one row per user per mask) plus the
`.sidecar.json` that `make_label_bundle.py` wrote next to the uploaded zip. The
platform's export carries only id/class/verdict, so every agent signal comes from
the sidecar, rejoined on (frame, mask id).

The sidecar describes the whole uploaded corpus while a job covers a slice of it,
so most sidecar masks having no verdict is normal — they belong to masks this job
never asked about. It is what makes the population counts below trustworthy: they
are the real totals, not the sample's.

  --job triage     Replaces Table agent_performance's proxy reference. That table
                   scores each agent against *Swin quality*, so it measures
                   agreement between two agents, not accuracy: it cannot say
                   whether Qwen's 95.5% recall means Qwen is right or that Swin
                   and Qwen are wrong together. Here the reference is human, so
                   Swin is scored on the same footing as the VLMs, and the masks
                   where the two runs' triage disagrees are adjudicated.

  --job discovery  Precision per confirmation stratum, and the estimate that
                   follows from it: how many of the 45,882 candidates LLaVA
                   confirmed and the 9,878 Qwen confirmed are real objects.
                   The downstream mIoU comparison cannot answer this, because the
                   curated reference is itself SAM-derived and cannot credit a
                   genuinely recovered object (see the paper's Evaluation
                   Protocol); a human verdict on the candidate itself can.

All confidence intervals resample *frames*, not masks: masks in one frame share a
scene, an exposure and a labeler's calibration, so a per-mask bootstrap would
treat ~21 correlated judgements as independent and report intervals that are far
too tight.

Usage:
    python analyze_human_verification.py --job triage \\
        --export triage_export.csv --sidecar /path/to/triage.sidecar.json
    python analyze_human_verification.py --job discovery \\
        --export discovery_export.csv --sidecar /path/to/discovery.sidecar.json \\
        --latex-out paper/tables/human_discovery.tex
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

CORRECT, INCORRECT = "correct", "incorrect"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_verdicts(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    """(frame_id, mask_id) -> {labeler email: verdict}."""
    verdicts: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if "maskId" not in row:
                raise SystemExit("Export has no maskId column — this is a single_choice job export")
            frame_id = Path(row["frame"]).stem
            verdicts[(frame_id, int(row["maskId"]))][row["userEmail"]] = row["verdict"]
    if not verdicts:
        raise SystemExit(f"No verdict rows in {path}")
    return verdicts


def consensus(labels: dict[str, str]) -> str | None:
    """Majority human verdict; None on a tie (an even split is not evidence)."""
    counts = Counter(labels.values()).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def join(sidecar: dict, verdicts: dict) -> tuple[list[dict], dict]:
    """
    Sidecar masks with a human verdict attached, plus join diagnostics.

    Masks whose labelers split evenly keep `human=None` rather than being
    dropped here: they carry no reference verdict, but they are exactly the
    masks inter-labeler agreement has to see, and silently discarding them
    would report perfect agreement by construction.
    """
    rows, stats = [], Counter()
    for frame_id, frame in sidecar["frames"].items():
        for mask in frame["masks"]:
            labels = verdicts.get((frame_id, mask["id"]))
            if not labels:
                stats["unlabeled"] += 1
                continue
            verdict = consensus(labels)
            stats["tied" if verdict is None else "scored"] += 1
            rows.append({**mask, "frame_id": frame_id, "frame_stratum": frame["stratum"],
                         "human": verdict, "labelers": labels})
    labeled_keys = {key for key in verdicts}
    sidecar_keys = {(f, m["id"]) for f, frame in sidecar["frames"].items() for m in frame["masks"]}
    stats["exported_not_in_sidecar"] = len(labeled_keys - sidecar_keys)
    if not rows:
        raise SystemExit("No mask matched between export and sidecar — mismatched job/bundle?")
    return rows, stats


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def bootstrap_ci(rows: list[dict], statistic, iterations: int, seed: int) -> tuple[float, float] | None:
    """Percentile CI over frames resampled with replacement (frame = cluster)."""
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_id"]].append(row)
    frames = list(by_frame.values())
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        picked = [frames[i] for i in rng.integers(0, len(frames), len(frames))]
        value = statistic([row for frame in picked for row in frame])
        if value is not None:
            samples.append(value)
    if len(samples) < iterations // 2:
        return None
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def rate(rows: list[dict], positive) -> float | None:
    return sum(1 for row in rows if positive(row)) / len(rows) if rows else None


def classification(rows: list[dict], predicts_correct) -> dict[str, float] | None:
    """Accuracy/precision/recall/F1 with positive = 'the human called this mask correct'."""
    tp = fp = fn = tn = 0
    for row in rows:
        predicted, actual = predicts_correct(row), row["human"] == CORRECT
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1
    total = tp + fp + fn + tn
    if total == 0 or tp + fp == 0 or tp + fn == 0:
        return None
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    if precision + recall == 0:
        return None
    return {
        "acc": 100 * (tp + tn) / total,
        "prec": 100 * precision,
        "rec": 100 * recall,
        "f1": 100 * 2 * precision * recall / (precision + recall),
        "n": total,
    }


def cohen_kappa(rows: list[dict]) -> tuple[float, int] | None:
    """Chance-corrected agreement, over masks two labelers both judged."""
    pairs = [tuple(row["labelers"].values()) for row in rows if len(row["labelers"]) == 2]
    if len(pairs) < 2:
        return None
    observed = sum(1 for a, b in pairs if a == b) / len(pairs)
    first = Counter(a for a, _ in pairs)
    second = Counter(b for _, b in pairs)
    expected = sum((first[label] / len(pairs)) * (second[label] / len(pairs)) for label in set(first) | set(second))
    if expected >= 1:
        return None
    return (observed - expected) / (1 - expected), len(pairs)


def show(label: str, metrics: dict | None, ci: tuple[float, float] | None = None, key: str = "f1") -> None:
    if metrics is None:
        print(f"  {label:<34} (not estimable)")
        return
    interval = f"  [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else ""
    print(f"  {label:<34} acc {metrics['acc']:5.1f}  prec {metrics['prec']:5.1f}  "
          f"rec {metrics['rec']:5.1f}  F1 {metrics[key]:5.1f}{interval}  (n={metrics['n']})")


# --------------------------------------------------------------------------
# Triage job
# --------------------------------------------------------------------------


def signals(tags: list[str]) -> dict[str, callable]:
    """Each agent's binary verdict, as 'this mask is correct'."""
    checks = {
        "Swin quality (good)": lambda row: row.get("quality_agent") == "good",
        "LiDAR consistency (pass)": lambda row: row.get("consistency") == "pass",
    }
    for tag in tags:
        checks[f"BBox VLM ({tag})"] = lambda row, tag=tag: row.get(f"bbox_agent_{tag}") == "valid"
    for tag in tags:
        checks[f"Triage retains ({tag})"] = lambda row, tag=tag: row.get(f"triage_{tag}") != "reject"
    return checks


def report_triage(all_rows: list[dict], sidecar: dict, args) -> None:
    tags = sidecar["tags"]
    rows = [row for row in all_rows if row["human"]]
    human_correct = rate(rows, lambda row: row["human"] == CORRECT)
    print(f"\n{len(rows)} masks over {len({row['frame_id'] for row in rows})} frames; "
          f"humans called {100 * human_correct:.1f}% correct")

    kappa = cohen_kappa(all_rows)
    if kappa:
        print(f"Inter-labeler agreement (Cohen's kappa): {kappa[0]:.3f} over {kappa[1]} doubly-labeled masks")
    else:
        print("Inter-labeler agreement: not computable (needs redundancy >= 2)")

    print("\nAgent accuracy vs human reference (positive = mask is correct):")
    for label, predicts in signals(tags).items():
        metrics = classification(rows, predicts)
        ci = bootstrap_ci(rows, lambda sub, p=predicts: (classification(sub, p) or {}).get("f1"),
                          args.iterations, args.seed) if metrics else None
        show(label, metrics, ci)

    print("\nPer class (F1 of each run's BBox verdict):")
    for class_name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == class_name]
        line = f"  {class_name:<12} n={len(subset):<5}"
        for tag in tags:
            metrics = classification(subset, lambda row, tag=tag: row.get(f"bbox_agent_{tag}") == "valid")
            line += f"  {tag}: {metrics['f1']:.1f}" if metrics else f"  {tag}: --"
        print(line)

    if len(tags) == 2:
        a, b = tags
        contested = [row for row in rows if row[f"triage_{a}"] != row[f"triage_{b}"]]
        print(f"\nCross-run adjudication on {len(contested)} masks the runs triage differently:")
        for tag in tags:
            agreed = sum(1 for row in contested
                         if (row[f"triage_{tag}"] != "reject") == (row["human"] == CORRECT))
            share = f"{100 * agreed / len(contested):.1f}%" if contested else "--"
            print(f"  {tag} sides with the human on {agreed}/{len(contested)} ({share})")

    for tag in tags:
        wrongly_rejected = [row for row in rows
                            if row[f"triage_{tag}"] == "reject" and row["human"] == CORRECT]
        by_class = Counter(row["class"] for row in wrongly_rejected)
        print(f"\n{tag}: {len(wrongly_rejected)} masks rejected that the human kept "
              f"({dict(by_class)})")


# --------------------------------------------------------------------------
# Discovery job
# --------------------------------------------------------------------------


def report_discovery(all_rows: list[dict], sidecar: dict, args) -> list[dict]:
    tags = sidecar["tags"]
    rows = [row for row in all_rows if row["human"]]
    population = sidecar.get("population", {})
    print(f"\n{len(rows)} candidates over {len({row['frame_id'] for row in rows})} frames")

    kappa = cohen_kappa(all_rows)
    if kappa:
        print(f"Inter-labeler agreement (Cohen's kappa): {kappa[0]:.3f} over {kappa[1]} doubly-labeled candidates")

    print("\nPrecision per confirmation stratum (share the human judged a real object):")
    table = []
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        precision = rate(subset, lambda row: row["human"] == CORRECT)
        ci = bootstrap_ci(subset, lambda sub: rate(sub, lambda row: row["human"] == CORRECT),
                          args.iterations, args.seed)
        size = population.get(stratum)
        entry = {"stratum": stratum, "n": len(subset), "precision": 100 * precision,
                 "ci": ci, "population": size,
                 "real": size * precision if size else None}
        table.append(entry)
        extra = f"  → ~{entry['real']:,.0f} real of {size:,}" if size else ""
        interval = f"  [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""
        print(f"  {stratum:<24} n={len(subset):<4} precision {100 * precision:5.1f}%{interval}{extra}")

    # Each run's confirmed set is 'both' plus its own exclusive stratum, so the
    # per-stratum precisions compose into the number the paper actually needs:
    # how much of a 45,882-candidate confirmed set is real.
    print("\nConfirmed-set precision per run (composed from the strata above):")
    for tag in tags:
        parts = [entry for entry in table if entry["stratum"] in ("both", f"{tag}_only")]
        subset = [row for row in rows if row["stratum"] in ("both", f"{tag}_only")]
        sizes = [entry["population"] for entry in parts]
        if not subset or any(size is None for size in sizes):
            print(f"  {tag}: population sizes missing from sidecar")
            continue
        total = sum(sizes)
        # Strata were sampled to equal size, not proportionally, so a pooled rate
        # would over-weight the rare stratum: weight each by its population.
        weighted = sum(entry["population"] * entry["precision"] / 100 for entry in parts) / total
        print(f"  {tag:<24} {100 * weighted:5.1f}% of {total:,} confirmed  "
              f"→ ~{total * weighted:,.0f} real objects added")

    missed = next((entry for entry in table if entry["stratum"] == "neither"), None)
    if missed and missed["real"]:
        print(f"\n  Candidates neither run confirmed still contain ~{missed['real']:,.0f} real objects "
              f"({missed['precision']:.1f}% of {missed['population']:,}) — the recall cost of VLM confirmation.")
    return table


def write_latex(table: list[dict], sidecar: dict, path: Path) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Human verification of Swin-proposed discovery candidates. "
        r"Candidates are sampled uniformly within each confirmation stratum; "
        r"precision is the share a human labeler judged a real object of the proposed class. "
        r"95\% CIs resample frames with replacement. "
        r"``Real'' extrapolates the stratum's precision to its full population.}",
        r"\label{tab:human_discovery}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Stratum} & \textbf{Labeled} & \textbf{Precision (\%)} & \textbf{Real / total} \\",
        r"\midrule",
    ]
    for entry in table:
        ci = f" [{100 * entry['ci'][0]:.0f}, {100 * entry['ci'][1]:.0f}]" if entry["ci"] else ""
        real = f"{entry['real']:,.0f} / {entry['population']:,}" if entry["real"] else "--"
        stratum = entry["stratum"].replace("_", r"\_")
        lines.append(f"{stratum} & {entry['n']} & {entry['precision']:.1f}{ci} & {real} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))
    print(f"\nWrote {path}")


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="job export CSV from label-front")
    parser.add_argument("--sidecar", type=Path, required=True, help=".sidecar.json from make_label_bundle.py")
    parser.add_argument("--job", choices=["triage", "discovery"], help="default: whatever the sidecar says")
    parser.add_argument("--iterations", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latex-out", type=Path, help="discovery: write the paper table here")
    args = parser.parse_args()

    sidecar = json.loads(args.sidecar.read_text())
    job = args.job or sidecar["job"]
    rows, stats = join(sidecar, load_verdicts(args.export))

    print(f"Job: {job}   tags: {', '.join(sidecar['tags'])}")
    print(f"Joined {stats['scored']} masks "
          f"({stats['unlabeled']} in the bundle but not in this job, {stats['tied']} tied, "
          f"{stats['exported_not_in_sidecar']} exported rows with no sidecar entry)")

    if job == "triage":
        report_triage(rows, sidecar, args)
    else:
        table = report_discovery(rows, sidecar, args)
        if args.latex_out:
            write_latex(table, sidecar, args.latex_out)


if __name__ == "__main__":
    main()
