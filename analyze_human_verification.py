#!/usr/bin/env python3
"""
Score the pipeline's agents against human verdicts from the visin labeling platform.

Input is the export of the single `verify` job — CSV, long format, one row per
labeler per mask. That is the only required input: label-service carries every
field the bundle put in `<frame>.masks.json` back out as a `mask_<field>` column,
so a verdict arrives with its own provenance attached and nothing has to be
re-joined against the uploaded zip.

The human was never asked which run produced what. Attribution is done here: the
export says what each run decided about each mask, so a human "this one is wrong"
lands on whichever run kept it, and a human "this one is fine" lands on whichever
run threw it away. One pass of labeling, both runs scored.

Population totals — what a measured rate scales to — come from the job's manifest
(`--manifest`, the platform's third export format), which pairs each field value
with how many masks the bundle holds against how many the job asked about. Pass
it whenever the job sampled rather than covering everything; without it the
per-stratum precisions are still reported, just not extrapolated.

The job holds two kinds of region, split apart by `mask_source` and reported
separately:

  sam          Are the SAM masks the pipeline triaged actually correct? Replaces
               Table agent_performance's proxy reference. That table scores each
               agent against *Swin quality*, so it measures agreement between two
               agents, not accuracy: it cannot say whether Qwen's 95.5% recall
               means Qwen is right or that Swin and Qwen are wrong together. Here
               the reference is human, so Swin is scored on the same footing as
               the VLMs, and the masks where the two runs' triage disagrees are
               adjudicated.

  discovery    Precision per confirmation stratum, and the estimate that follows
               from it: how many of the 45,882 candidates LLaVA confirmed and the
               9,878 Qwen confirmed are real objects. The downstream mIoU
               comparison cannot answer this, because the curated reference is
               itself SAM-derived and cannot credit a genuinely recovered object
               (see the paper's Evaluation Protocol); a human verdict on the
               candidate itself can.

               Reported per `kind`, never pooled across it. Roughly two thirds of
               candidates are *fringe* — the rim left over on an object SAM had
               already segmented — and only a *standalone* candidate is an object
               the pipeline would otherwise have missed. A pooled precision would
               read as recall gain while being mostly boundary growth, so the two
               get separate tables and separate extrapolations.

A mask_toggle labeler clicks only what is wrong, but the export is still
exhaustive: label-service writes one row per mask per answer and reads a mask's
verdict as `incorrect` iff the labeler clicked it (`exportService.ts`,
`maskVerdict`). So silence in the *workbench* is already resolved into `correct`
in the *export*. A row with no labeler on it is the other case the export
distinguishes — a task nobody has reached yet — and those are counted as pending
rather than scored.

All confidence intervals resample *frames*, not masks: masks in one frame share a
scene, an exposure and a labeler's calibration, so a per-mask bootstrap would
treat ~35 correlated judgements as independent and report intervals that are far
too tight.

Usage:
    python analyze_human_verification.py --export verify_export.csv \\
        --manifest job-<id>.manifest.json \\
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


def scalar(text: str):
    """CSV is all strings; put the bundle's own types back."""
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


def load_masks(path: Path) -> tuple[list[dict], dict]:
    """
    One entry per mask, carrying its bundle metadata and every labeler's verdict.

    The export is long — one row per labeler per mask — so rows are folded back
    into masks here. `mask_<field>` columns are the bundle's own metadata coming
    home; the prefix exists so a mask field named `stratum` cannot collide with
    the task-level column of the same name, and it is stripped again here.

    Masks nobody has judged yet keep an empty `labelers` and are reported as
    pending: label-service emits a row for an unanswered task precisely so the
    denominator survives, and silently dropping them would report progress as
    completeness.
    """
    masks: dict[tuple[str, int], dict] = {}
    stats = Counter()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "maskId" not in reader.fieldnames:
            raise SystemExit(f"{path} has no maskId column — this is a single_choice job export")
        for row in reader:
            key = (Path(row["frame"]).stem, int(row["maskId"]))
            if key not in masks:
                masks[key] = {
                    "frame_id": key[0],
                    "id": key[1],
                    "class": row.get("class"),
                    "labelers": {},
                    **{
                        field.removeprefix("mask_"): scalar(value)
                        for field, value in row.items()
                        if field.startswith("mask_") and value != ""
                    },
                }
            if row.get("userEmail"):
                masks[key]["labelers"][row["userEmail"]] = row["verdict"]
    if not masks:
        raise SystemExit(f"No mask rows in {path}")

    rows = []
    for mask in masks.values():
        if not mask["labelers"]:
            stats["pending"] += 1
            continue
        verdict = consensus(mask["labelers"])
        stats["tied" if verdict is None else "scored"] += 1
        rows.append({**mask, "human": verdict})
    if not rows:
        raise SystemExit("Every mask in the export is still unanswered")
    return rows, stats


def consensus(labels: dict[str, str]) -> str | None:
    """Majority human verdict; None on a tie (an even split is not evidence)."""
    counts = Counter(labels.values()).most_common()
    if len(counts) > 1 and counts[0][1] == counts[1][1]:
        return None
    return counts[0][0]


def run_tags(rows: list[dict]) -> list[str]:
    """The VLM runs the bundle shipped decisions for, read off the columns.

    Every SAM mask carries a `triage_<tag>` per run, so the run names are in the
    export rather than needing to be told to this script — and a bundle built
    with a third run is picked up without a flag.
    """
    return sorted({
        field.removeprefix("triage_") for row in rows for field in row if field.startswith("triage_")
    })


def stratum_population(manifest: dict | None) -> dict[str, int]:
    """
    Bundle-wide candidate count per confirmation stratum, for extrapolation.

    The manifest tallies each field independently, so it holds a stratum total
    and a kind total but not the joint. That is enough here only because the
    bundle ships one kind of candidate (`make_label_bundle.py --candidates`), and
    SAM masks carry no `stratum` at all — so the stratum marginal *is* the
    per-kind total. A bundle carrying both kinds makes that false, and this
    returns nothing rather than a number that would overstate every stratum.
    """
    if not manifest or "masks" not in manifest:
        return {}
    if len(manifest["masks"].get("kind", {})) > 1:
        return {}
    return {value: counts["bundle"] for value, counts in manifest["masks"].get("stratum", {}).items()}


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


def report_triage(all_rows: list[dict], tags: list[str], args) -> None:
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

    # Where a human "wrong" landed. The labeler judged the mask, not the run, so
    # both error directions are attributed here rather than asked about there.
    print("\nAttribution of the human's verdicts, per run:")
    incorrect = [row for row in rows if row["human"] == INCORRECT]
    print(f"  {len(incorrect)} masks the human called incorrect")
    for tag in tags:
        retained = [row for row in incorrect if row[f"triage_{tag}"] != "reject"]
        by_class = Counter(row["class"] for row in retained)
        share = f"{100 * len(retained) / len(incorrect):.1f}%" if incorrect else "--"
        print(f"    {tag}: kept {len(retained)} of them ({share}) {dict(by_class)}")
    for tag in tags:
        wrongly_rejected = [row for row in rows
                            if row[f"triage_{tag}"] == "reject" and row["human"] == CORRECT]
        by_class = Counter(row["class"] for row in wrongly_rejected)
        print(f"    {tag}: rejected {len(wrongly_rejected)} masks the human kept ({dict(by_class)})")

    if len(tags) == 2:
        a, b = tags
        only = {tag: [row for row in incorrect
                      if (row[f"triage_{tag}"] != "reject") and (row[f"triage_{other}"] == "reject")]
                for tag, other in ((a, b), (b, a))}
        both = [row for row in incorrect
                if row[f"triage_{a}"] != "reject" and row[f"triage_{b}"] != "reject"]
        neither = len(incorrect) - len(both) - sum(len(value) for value in only.values())
        print(f"  Of those: {len(both)} survived both runs, {neither} caught by both, "
              + ", ".join(f"{len(value)} let through by {tag} alone" for tag, value in only.items()))


# --------------------------------------------------------------------------
# Discovery job
# --------------------------------------------------------------------------


def report_discovery(all_rows: list[dict], tags: list[str], population: dict[str, int],
                     reason_no_population: str, args) -> list[dict]:
    """
    Precision per confirmation stratum, within one kind of candidate.

    Kind is not a nuisance variable to pool over. A *standalone* candidate is a
    region SAM missed entirely, so a human "yes" means an object was recovered; a
    *fringe* is the leftover rim of an object SAM already segmented, so a human
    "yes" means the boundary grew. Pooling them reports the second as if it were
    the first, and since fringes outnumber standalones roughly 2:1 the pooled
    number would be mostly boundary growth wearing the label of recall.
    """
    rows = [row for row in all_rows if row["human"]]
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
            print(f"  {tag}: not extrapolated — {reason_no_population}")
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


def write_latex(table: list[dict], path: Path, kind: str) -> None:
    scope = {
        "standalone": "Candidates are restricted to those covering image regions SAM segmented "
                      "no object in, so a positive verdict is an object recovered rather than a "
                      "boundary extended.",
        "fringe": "Candidates here abut an existing SAM mask, so a positive verdict extends an "
                  "object's boundary rather than recovering a missed object.",
    }[kind]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Human verification of Swin-proposed discovery candidates. " + scope + " "
        r"Precision is the share a human labeler judged a real object of the proposed class; "
        r"labelers saw each candidate's segmented pixels, not its bounding box. "
        r"95\% CIs resample frames with replacement. "
        r"``Real'' extrapolates the stratum's precision to its full population.}",
        rf"\label{{tab:human_discovery_{kind}}}",
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
    parser.add_argument("--manifest", type=Path,
                        help="job manifest JSON from label-front — bundle-wide totals, for extrapolation")
    parser.add_argument("--only", choices=["sam", "discovery"], help="report one source only")
    parser.add_argument("--iterations", type=int, default=10000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latex-out", type=Path, help="write the discovery paper table here")
    parser.add_argument("--latex-kind", choices=["standalone", "fringe"], default="standalone",
                        help="which kind of candidate the paper table reports (default: standalone)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    rows, stats = load_masks(args.export)
    tags = run_tags(rows)

    if manifest:
        job = manifest["job"]
        print(f"Job: {job['name']}   K={job['redundancy']}   "
              f"{manifest['progress']['completed']}/{manifest['progress']['tasks']} frames complete")
    print(f"Runs: {', '.join(tags) or '(none named in the export)'}")
    print(f"Scored {stats['scored']} masks ({stats['tied']} tied, {stats['pending']} not yet answered)")

    # One job, two questions: `source` says which masks answer which, so the
    # split happens here rather than in the labeling.
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("source", "sam")].append(row)

    if args.only is None or args.only == "sam":
        if by_source["sam"]:
            print("\n" + "=" * 70 + "\nSAM proposals — triage accuracy\n" + "=" * 70)
            report_triage(by_source["sam"], tags, args)
    if args.only is None or args.only == "discovery":
        # Reported one kind at a time: a standalone find and a boundary fringe
        # are different claims and do not average into anything meaningful.
        by_kind = defaultdict(list)
        for row in by_source["discovery"]:
            by_kind[row.get("kind", "standalone")].append(row)
        # A stratum total covers every candidate in the bundle, so it is only the
        # total *for a kind* when the bundle carries one kind. With both present
        # the totals are refused rather than misattributed — build the bundle
        # with `--candidates standalone` to get the extrapolation back.
        population = stratum_population(manifest) if len(by_kind) == 1 else {}
        reason = (
            "pass --manifest for the bundle-wide totals" if not manifest
            else "the job mixes standalone and fringe candidates, whose stratum totals the manifest "
                 "does not separate" if len(by_kind) > 1
            else "the manifest carries no stratum totals"
        )
        for kind, rows_of_kind in sorted(by_kind.items()):
            print("\n" + "=" * 70 + f"\nDiscovery candidates ({kind}) — precision per stratum\n" + "=" * 70)
            table = report_discovery(rows_of_kind, tags, population, reason, args)
            if args.latex_out and kind == args.latex_kind:
                write_latex(table, args.latex_out, kind)


if __name__ == "__main__":
    main()
