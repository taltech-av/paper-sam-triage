#!/usr/bin/env python3
"""
Score the pipeline against human verdicts from the visin labeling platform.

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

               Beyond the agents as shipped, *every* triage variant is replayed
               over these masks and scored (the VARIANTS section). A variant is a
               deterministic function of stored agent outputs, so it can be
               replayed offline and ranked directly against the human reference —
               no training run, no GPU. Downstream mIoU ranks the same variants
               only indirectly, at one GPU job each, and returns intervals wide
               enough that most variants are indistinguishable.

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

Read F1 here with care, and prefer the error budget printed beside it. A retained
wrong mask injects wrong pixels into a training label; a dropped good mask only
omits them. F1 weights the two equally, so it rewards the rule that keeps
everything whenever the base rate is high — which is why `raw_sam` tops the F1
column while losing the measured mIoU ladder. `bad kept` (share of
human-rejected masks a rule retains) and `good lost` (share of human-accepted
masks it deletes) separate the two error directions and are the columns to rank
variants on.

A mask_toggle labeler clicks only what is wrong, but the export is still
exhaustive: label-service writes one row per mask per answer and reads a mask's
verdict as `incorrect` iff the labeler clicked it (`exportService.ts`,
`maskVerdict`). So silence in the *workbench* is already resolved into `correct`
in the *export*. A row with no labeler on it is the other case the export
distinguishes — a task nobody has reached yet — and those are counted as pending
rather than scored.

Partial exports are the intended case, not a degraded one: labelling a few frames
first confirms the export joins, the rules evaluate and the numbers are sane
before committing to the full set. Coverage and labelling pace are therefore
reported before any score, and every rate carries the mask count it rests on.

All confidence intervals resample *frames*, not masks: masks in one frame share a
scene, an exposure and a labeler's calibration, so a per-mask bootstrap would
treat ~35 correlated judgements as independent and report intervals that are far
too tight.

Usage:
    python analyze_human_verification.py --export verify_export.csv
    python analyze_human_verification.py --export verify_export.csv \\
        --manifest job-<id>.manifest.json \\
        --latex-out paper/tables/human_discovery.tex

The whole report is written to a file as well as the terminal — next to the
export by default, or wherever `--out` says.
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import config
from replay_triage import VARIANTS

CORRECT, INCORRECT = "correct", "incorrect"
TRIAGE_REJECT = "reject"

# Variants whose decision needs state the bundle sidecar does not carry —
# correction-agent output, Swin bypass, class id. They are still scored; the join
# to `results/` supplies it, and they are the reason that join exists.
NEEDS_FULL_STATE = {"with_bypass", "lidar_class_aware", "limits_fixed", "background_protected"}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


_NUMERIC_START = frozenset("0123456789+-.")


def scalar(text: str):
    """CSV is all strings; put the bundle's own types back.

    The export is ~3M field values and most are words (`valid`, `pass`, `sam`) or
    bracketed bboxes, so trying int() then float() on everything means two raised
    exceptions per value. Only a leading digit, sign or point can begin a number,
    so that check short-circuits the common case.
    """
    if text in ("true", "True"):
        return True
    if text in ("false", "False"):
        return False
    if text and text[0] in _NUMERIC_START:
        for cast in (int, float):
            try:
                return cast(text)
            except ValueError:
                pass
    return text


def load_masks(path: Path, include_unresolved: bool = False) -> tuple[list[dict], dict]:
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

    Scoring wants only the resolved masks, so those are what is returned by
    default. `include_unresolved` keeps the pending and tied ones too, with
    `human` set to None — which is what a *writer* needs: an annotation is only
    clean if every mask in the frame was judged, and that question cannot be
    answered from the resolved masks alone.
    """
    masks: dict[tuple[str, int], dict] = {}
    stats = Counter()
    # A few hundred frame paths repeat across ~150k rows; Path().stem on each is
    # pure overhead, so the stems are memoised.
    stems: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "maskId" not in reader.fieldnames:
            raise SystemExit(f"{path} has no maskId column — this is a single_choice job export")
        for row in reader:
            frame = row["frame"]
            stem = stems.get(frame)
            if stem is None:
                stem = stems[frame] = Path(frame).stem
            key = (stem, int(row["maskId"]))
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

    # Frames the job holds at all — the denominator coverage is measured against,
    # so it comes from the export rather than a constant that can drift.
    stats["frames_total"] = len({key[0] for key in masks})

    rows = []
    for mask in masks.values():
        if not mask["labelers"]:
            stats["pending"] += 1
            if include_unresolved:
                rows.append({**mask, "human": None})
            continue
        verdict = consensus(mask["labelers"])
        stats["tied" if verdict is None else "scored"] += 1
        if verdict is not None or include_unresolved:
            rows.append({**mask, "human": verdict})
    if not stats["scored"]:
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


def load_run_masks(tag: str, frame_ids: set[str]) -> dict[tuple[str, int], dict]:
    """Per-(frame, mask_id) pipeline state for one run, for the labelled frames only.

    The export alone cannot drive a variant replay: the bundle sidecar carries
    what a labeller might be asked to group by, not the full agent state. The
    results JSON is the authority for what each agent actually returned.
    """
    results = config.DATA_ROOT / "vlm" / tag / "results"
    if not results.is_dir():
        raise SystemExit(f"No results for tag '{tag}': {results}")
    out: dict[tuple[str, int], dict] = {}
    for frame_id in frame_ids:
        path = results / f"{frame_id}.json"
        if not path.exists():
            continue
        for mask in json.loads(path.read_text()).get("masks", []):
            out[(frame_id, mask["mask_id"])] = mask
    return out


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


def bootstrap_ci(frame_ids, terms, statistic, iterations: int, seed: int,
                 block: int = 1024) -> tuple[float, float] | None:
    """Percentile CI over frames resampled with replacement (frame = cluster).

    Every statistic reported here — F1, a precision, a rate — is a function of
    sums that are *additive over rows*: tp/fp/fn counts, positives, denominators.
    So a resample never has to touch a row. The per-row terms are summed once
    into a (frame x term) table, and a resample is then a vector of how many
    times each frame was drawn, times that table. Ten thousand resamples become
    one matrix product instead of ten thousand passes over the masks, which is
    the difference between half an hour and a few milliseconds per interval.

    Drawing frame counts from a multinomial is the same experiment as drawing
    `n_frames` frame indices with replacement and tallying them; it just skips
    materialising the indices.

    `terms` is (n_rows, k). `statistic` maps resampled sums, shape (m, k), to m
    values, NaN where the statistic is undefined on that resample — those are
    dropped, and an interval is refused if fewer than half the resamples define
    it, matching the old row-wise contract of returning None.
    """
    codes, n_frames = _frame_codes(frame_ids)
    terms = np.asarray(terms, dtype=np.float64)
    per_frame = np.column_stack([
        np.bincount(codes, weights=terms[:, column], minlength=n_frames)
        for column in range(terms.shape[1])
    ])

    rng = np.random.default_rng(seed)
    probabilities = np.full(n_frames, 1.0 / n_frames)
    samples = []
    for start in range(0, iterations, block):
        draws = rng.multinomial(n_frames, probabilities, size=min(block, iterations - start))
        samples.append(statistic(draws @ per_frame))
    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    if values.size < iterations // 2:
        return None
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _frame_codes(frame_ids) -> tuple[np.ndarray, int]:
    """Frame ids to contiguous integer codes, for use as bincount cluster labels."""
    _, codes = np.unique(np.asarray(frame_ids), return_inverse=True)
    return codes, int(codes.max()) + 1 if codes.size else 0


def f1_ci(rows: list[dict], predicted, iterations: int, seed: int) -> tuple[float, float] | None:
    """95% CI on F1 (percent) for a binary prediction over `rows`."""
    if not iterations:
        return None
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray([row["human"] == CORRECT for row in rows], dtype=bool)
    terms = np.column_stack([predicted & actual, predicted & ~actual, ~predicted & actual])
    return bootstrap_ci([row["frame_id"] for row in rows], terms, _f1_of, iterations, seed)


def rate_ci(rows: list[dict], positive, iterations: int, seed: int) -> tuple[float, float] | None:
    """95% CI on the share of rows satisfying `positive` (a fraction, as `rate`)."""
    if not iterations:
        return None
    flags = np.asarray([bool(positive(row)) for row in rows], dtype=float)
    terms = np.column_stack([flags, np.ones_like(flags)])
    return bootstrap_ci([row["frame_id"] for row in rows], terms, _rate_of, iterations, seed)


def paired_delta_ci(rows: list[dict], predicted_a, predicted_b,
                    iterations: int, seed: int) -> tuple[float, float] | None:
    """95% CI on F1(a) − F1(b) over the same masks, resampled by frame.

    The two runs decide on identical masks, so the comparison is paired and the
    difference is far better determined than the gap between two independent
    intervals suggests. Both F1s are recomputed inside each resample, which is
    what makes the interval paired rather than a difference of marginals.
    """
    if not iterations:
        return None
    actual = np.asarray([row["human"] == CORRECT for row in rows], dtype=bool)
    a, b = np.asarray(predicted_a, dtype=bool), np.asarray(predicted_b, dtype=bool)
    terms = np.column_stack([
        a & actual, a & ~actual, ~a & actual,
        b & actual, b & ~actual, ~b & actual,
    ])
    return bootstrap_ci([row["frame_id"] for row in rows], terms,
                        lambda sums: _f1_of(sums[:, :3]) - _f1_of(sums[:, 3:]),
                        iterations, seed)


def _f1_of(sums: np.ndarray) -> np.ndarray:
    """F1 in percent from summed (tp, fp, fn); NaN where the old code returned None.

    tp == 0 is exactly that case: either a denominator is empty, or precision and
    recall are both zero and their harmonic mean is undefined.
    """
    tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(tp > 0, 100 * 2 * tp / (2 * tp + fp + fn), np.nan)


def _rate_of(sums: np.ndarray) -> np.ndarray:
    """Share of positives from summed (positives, rows)."""
    total = sums[:, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, sums[:, 0] / total, np.nan)


def rate(rows: list[dict], positive) -> float | None:
    return sum(1 for row in rows if positive(row)) / len(rows) if rows else None


def classification(rows: list[dict], predicts_correct) -> dict[str, float] | None:
    """Accuracy/precision/recall/F1 with positive = 'the human called this mask correct'."""
    return metrics([predicts_correct(row) for row in rows],
                   [row["human"] == CORRECT for row in rows])


def metrics(predicted, actual) -> dict[str, float] | None:
    """The same scores from two aligned boolean sequences, for callers holding arrays.

    `bad_kept` and `good_lost` are the error budget: the share of human-rejected
    masks the rule retains, and the share of human-accepted masks it deletes.
    They are the two error directions F1 pools, and they are not interchangeable
    downstream — a kept bad mask paints wrong pixels into the training label,
    a lost good mask only leaves them unlabelled.
    """
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    total = int(predicted.size)
    tp = int(np.count_nonzero(predicted & actual))
    fp = int(np.count_nonzero(predicted & ~actual))
    fn = int(np.count_nonzero(~predicted & actual))
    tn = total - tp - fp - fn
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
        "bad_kept": 100 * fp / (fp + tn) if fp + tn else float("nan"),
        "good_lost": 100 * fn / (tp + fn),
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


def show(label: str, scores: dict | None, ci: tuple[float, float] | None = None, note: str = "") -> None:
    """One scored rule per line: accuracy block, F1 with its CI, then the error budget."""
    suffix = f"  {note}" if note else ""
    if scores is None:
        print(f"  {label:<32} (not estimable){suffix}")
        return
    interval = f" [{ci[0]:5.1f},{ci[1]:5.1f}]" if ci else " " * 14
    print(f"  {label:<32} n={scores['n']:>6}  acc {scores['acc']:5.1f}  prec {scores['prec']:5.1f}  "
          f"rec {scores['rec']:5.1f}  F1 {scores['f1']:5.1f}{interval}  "
          f"bad kept {scores['bad_kept']:5.1f}  good lost {scores['good_lost']:5.1f}{suffix}")


def header(title: str) -> None:
    print("\n" + "=" * 116)
    print(f"  {title}")
    print("=" * 116)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------


def report_coverage(export: Path, rows: list[dict], stats: dict, tags: list[str]) -> None:
    """What has actually been labelled, before any score is quoted."""
    sam_rows = [row for row in rows if row.get("source", "sam") == "sam"]
    disc_rows = [row for row in rows if row.get("source") == "discovery"]
    frames = {row["frame_id"] for row in rows}

    header("COVERAGE")
    print(f"  runs in the export       {', '.join(tags) or '(none named)'}")
    print(f"  frames labelled          {len(frames):,} of {stats['frames_total']:,} "
          f"({100 * len(frames) / stats['frames_total']:.1f}%)")
    print(f"  masks scored             {stats['scored']:,}")
    if stats["pending"]:
        print(f"  masks still unanswered   {stats['pending']:,}  (excluded)")
    if stats["tied"]:
        print(f"  masks tied between labellers {stats['tied']:,}  (excluded — a split is not evidence)")
    print(f"    SAM proposals          {len(sam_rows):,}")
    print(f"    discovery candidates   {len(disc_rows):,}")
    if disc_rows:
        print(f"      by kind              {dict(Counter(row.get('kind') for row in disc_rows))}")
    progress(export, stats["frames_total"])
    if len(frames) < 25:
        print("\n  NOTE: few frames labelled. Treat the numbers below as a pipeline check,"
              "\n        not a result — frame-clustered CIs need many more frames to be"
              "\n        informative, and rules that differ on rare masks may not differ here.")


def progress(export: Path, total_frames: int) -> None:
    """Labelling pace and time-to-completion, from the export's own timings.

    The unit of work is a *frame*, not a region: a task is one frame, the
    labeller toggles all of its masks in one sitting, and `elapsedMs` is the time
    on that task repeated onto every mask row of the frame. Averaging it per row
    therefore overstates the remaining work by the number of masks per frame —
    roughly 35x on this corpus. Pace is deduplicated per frame here.

    Two paces are reported: over every frame answered, and over the most recent
    quarter. They differ a lot early on, since the first frames carry interface
    learning, and the recent figure is the one that predicts what is left.
    """
    per_frame: dict[str, tuple[str, int]] = {}
    with export.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("userEmail") or not row.get("elapsedMs"):
                continue
            try:
                ms = int(row["elapsedMs"])
            except ValueError:
                continue
            per_frame.setdefault(Path(row["frame"]).stem, (row.get("answeredAt") or "", ms))
    if not per_frame:
        return
    ordered = [ms for _, (_, ms) in sorted(per_frame.items(), key=lambda item: item[1][0])]
    recent = ordered[-max(len(ordered) // 4, 1):]
    med = sorted(ordered)[len(ordered) // 2] / 1000
    med_recent = sorted(recent)[len(recent) // 2] / 1000
    done = len(per_frame)

    print(f"\n  PROGRESS  {done:,} frames answered   {sum(ordered) / 3.6e6:.1f} h spent")
    print(f"    pace, all frames   {med:5.1f} s/frame")
    print(f"    pace, recent 25%   {med_recent:5.1f} s/frame"
          + ("   (speeding up)" if med_recent < med * 0.9 else ""))
    for label, per in (("all", med), ("recent", med_recent)):
        hrs = (total_frames - done) * per / 3600
        print(f"    ETA at {label:6} pace  {hrs:6.1f} h remaining"
              f"   = {hrs / 7:4.1f} h/day for a week"
              f"   |  {hrs / 14:4.1f} h/day for two")


# --------------------------------------------------------------------------
# SAM proposals — the agents as shipped
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
    print(f"\n{len(rows):,} masks over {len({row['frame_id'] for row in rows}):,} frames; "
          f"humans called {100 * human_correct:.1f}% correct "
          f"({sum(1 for row in rows if row['human'] == INCORRECT):,} rejected)")

    kappa = cohen_kappa(all_rows)
    if kappa:
        print(f"Inter-labeler agreement (Cohen's kappa): {kappa[0]:.3f} over {kappa[1]} doubly-labeled masks")
    else:
        print("Inter-labeler agreement: not computable (needs redundancy >= 2)")

    print("\nAgent accuracy vs human reference (positive = mask is correct):")
    for label, predicts in signals(tags).items():
        predicted = [bool(predicts(row)) for row in rows]
        scores = metrics(predicted, [row["human"] == CORRECT for row in rows])
        ci = f1_ci(rows, predicted, args.bootstrap, args.seed) if scores else None
        show(label, scores, ci)

    print("\nPer class (F1 of each run's BBox verdict):")
    for class_name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == class_name]
        correct = rate(subset, lambda row: row["human"] == CORRECT)
        line = f"  {class_name:<12} n={len(subset):<6} human-correct {100 * correct:5.1f}%"
        for tag in tags:
            scores = classification(subset, lambda row, tag=tag: row.get(f"bbox_agent_{tag}") == "valid")
            line += f"   {tag}: {scores['f1']:5.1f}" if scores else f"   {tag}: --"
        print(line)

    if len(tags) == 2:
        report_cross_run(rows, tags, args)

    # Where a human "wrong" landed. The labeler judged the mask, not the run, so
    # both error directions are attributed here rather than asked about there.
    print("\nAttribution of the human's verdicts, per run:")
    incorrect = [row for row in rows if row["human"] == INCORRECT]
    print(f"  {len(incorrect):,} masks the human called incorrect")
    for tag in tags:
        retained = [row for row in incorrect if row[f"triage_{tag}"] != "reject"]
        by_class = Counter(row["class"] for row in retained)
        share = f"{100 * len(retained) / len(incorrect):.1f}%" if incorrect else "--"
        print(f"    {tag}: kept {len(retained):,} of them ({share}) {dict(by_class)}")
    for tag in tags:
        wrongly_rejected = [row for row in rows
                            if row[f"triage_{tag}"] == "reject" and row["human"] == CORRECT]
        by_class = Counter(row["class"] for row in wrongly_rejected)
        print(f"    {tag}: rejected {len(wrongly_rejected):,} masks the human kept ({dict(by_class)})")

    if len(tags) == 2:
        a, b = tags
        only = {tag: [row for row in incorrect
                      if (row[f"triage_{tag}"] != "reject") and (row[f"triage_{other}"] == "reject")]
                for tag, other in ((a, b), (b, a))}
        both = [row for row in incorrect
                if row[f"triage_{a}"] != "reject" and row[f"triage_{b}"] != "reject"]
        neither = len(incorrect) - len(both) - sum(len(value) for value in only.values())
        print(f"  Of those: {len(both):,} survived both runs, {neither:,} caught by both, "
              + ", ".join(f"{len(value):,} let through by {tag} alone" for tag, value in only.items()))


def report_cross_run(rows: list[dict], tags: list[str], args) -> None:
    """Which run's triage is closer to the human — overall, and per class.

    The overall gap is quoted as a paired interval because the runs decide on the
    same masks. It is also split by class: a corpus whose composition is dominated
    by one class can hand the win to whichever model happens to be good at that
    class, and the split says whether the ordering is a competence ordering or a
    mix artifact.
    """
    a, b = tags
    contested = [row for row in rows if row[f"triage_{a}"] != row[f"triage_{b}"]]
    print(f"\nCross-run adjudication on {len(contested):,} masks the runs triage differently:")
    for tag in tags:
        agreed = sum(1 for row in contested
                     if (row[f"triage_{tag}"] != "reject") == (row["human"] == CORRECT))
        share = f"{100 * agreed / len(contested):.1f}%" if contested else "--"
        print(f"  {tag} sides with the human on {agreed:,}/{len(contested):,} ({share})")

    retains = {tag: [row[f"triage_{tag}"] != "reject" for row in rows] for tag in tags}
    delta = paired_delta_ci(rows, retains[a], retains[b], args.bootstrap, args.seed)
    if delta:
        print(f"  paired F1 difference {a} − {b}: [{delta[0]:+.1f}, {delta[1]:+.1f}] "
              f"({'separable' if delta[0] * delta[1] > 0 else 'not separable — interval spans 0'})")

    print("  the same difference, per class (a class-heavy corpus can invent the ordering):")
    for class_name in sorted({row["class"] for row in rows}):
        subset = [row for row in rows if row["class"] == class_name]
        share_of_corpus = 100 * len(subset) / len(rows)
        scores = {tag: classification(subset, lambda row, tag=tag: row[f"triage_{tag}"] != "reject")
                  for tag in tags}
        if not all(scores.values()):
            print(f"    {class_name:<12} n={len(subset):<6} ({share_of_corpus:4.1f}% of masks)  not estimable")
            continue
        delta = paired_delta_ci(subset,
                                [row[f"triage_{a}"] != "reject" for row in subset],
                                [row[f"triage_{b}"] != "reject" for row in subset],
                                args.bootstrap, args.seed)
        band = f"  Δ [{delta[0]:+.1f}, {delta[1]:+.1f}]" if delta else ""
        print(f"    {class_name:<12} n={len(subset):<6} ({share_of_corpus:4.1f}% of masks)  "
              + "  ".join(f"{tag} F1 {scores[tag]['f1']:5.1f}" for tag in tags) + band)


# --------------------------------------------------------------------------
# SAM proposals — every triage variant, replayed
# --------------------------------------------------------------------------


def report_variants(sam_rows: list[dict], tags: list[str], args) -> None:
    """Replay each rule in `replay_triage.VARIANTS` over the judged masks.

    The rules come from the same module that writes the training annotations, so
    a variant scored here is the variant that would be trained on. A rule that is
    not separable here will not be separable downstream either.
    """
    header("TRIAGE VARIANTS vs HUMAN VERDICT   (positive = human called the mask correct)")
    print("  A rule 'predicts correct' when it retains the mask. `bad kept` is the share of")
    print("  human-rejected masks it retains — wrong pixels entering a training label; `good")
    print("  lost` is the share of human-accepted masks it deletes. Rank on those, not F1:")
    print("  F1 pools the two, so it favours keeping everything whenever the base rate is high.")

    runs = {tag: load_run_masks(tag, {row["frame_id"] for row in sam_rows}) for tag in tags}
    print("\n  join to stored pipeline state (needed to replay a rule):")
    joined = {}
    for tag in tags:
        joined[tag] = sum(1 for row in sam_rows if (row["frame_id"], row["id"]) in runs[tag])
        pct = 100 * joined[tag] / len(sam_rows) if sam_rows else 0
        flag = "" if pct > 99 else "   <-- INCOMPLETE, scores below are on the joined subset"
        print(f"    {tag:26} {joined[tag]:,}/{len(sam_rows):,} ({pct:.1f}%){flag}")
    if sam_rows and max(joined.values()) == 0:
        raise SystemExit("\nNo exported mask joined to stored results — check --tags and the export's frame ids.")

    for tag in tags:
        state = runs[tag]
        pairs = [(row, state[(row["frame_id"], row["id"])]) for row in sam_rows
                 if (row["frame_id"], row["id"]) in state]
        if not pairs:
            print(f"\n  {tag}: no joined masks")
            continue
        print(f"\n  {tag}  ({len(pairs):,} masks)")
        for name, (rule, _description) in VARIANTS.items():
            # One evaluation of the rule per mask, kept as a boolean vector: the
            # verdict does not change between bootstrap resamples, so scoring and
            # every resample below read this vector rather than re-running the rule.
            scored, retains = [], []
            for row, mask in pairs:
                try:
                    keeps = rule(mask, swin_q=args.swin_threshold) != TRIAGE_REJECT
                except Exception:
                    continue
                scored.append(row)
                retains.append(keeps)
            if not scored:
                print(f"  {name:<26} (rule could not be evaluated)")
                continue
            scores = metrics(retains, [row["human"] == CORRECT for row in scored])
            ci = f1_ci(scored, retains, args.bootstrap, args.seed) if scores else None
            show(name, scores, ci, "(needs joined state)" if name in NEEDS_FULL_STATE else "")


# --------------------------------------------------------------------------
# Discovery candidates
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
    base = rate(rows, lambda row: row["human"] == CORRECT)
    print(f"\n{len(rows):,} candidates over {len({row['frame_id'] for row in rows}):,} frames; "
          f"{100 * base:.1f}% are real objects before any confirmation")

    kappa = cohen_kappa(all_rows)
    if kappa:
        print(f"Inter-labeler agreement (Cohen's kappa): {kappa[0]:.3f} over {kappa[1]} doubly-labeled candidates")

    print("\nPrecision per confirmation stratum (share the human judged a real object):")
    table = []
    for stratum in sorted({row["stratum"] for row in rows}):
        subset = [row for row in rows if row["stratum"] == stratum]
        precision = rate(subset, lambda row: row["human"] == CORRECT)
        ci = rate_ci(subset, lambda row: row["human"] == CORRECT, args.bootstrap, args.seed)
        size = population.get(stratum)
        entry = {"stratum": stratum, "n": len(subset), "precision": 100 * precision,
                 "ci": ci, "population": size,
                 "real": size * precision if size else None}
        table.append(entry)
        extra = f"  → ~{entry['real']:,.0f} real of {size:,}" if size else ""
        interval = f"  [{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]" if ci else ""
        print(f"  {stratum:<24} n={len(subset):<5} precision {100 * precision:5.1f}%{interval}{extra}")

    # A confirmation flag read as a *classifier* of "is this a real object": the
    # stratum table says how precise each pool is, this says whether confirming
    # carries information at all.
    print("\nEach run's confirmation, scored as a predictor of 'real object':")
    for tag in tags:
        field = f"confirmed_{tag}"
        have = [row for row in rows if field in row]
        if not have:
            print(f"  {tag:<32} (no {field} column in the export)")
            continue
        confirms = [bool(row.get(field)) for row in have]
        scores = metrics(confirms, [row["human"] == CORRECT for row in have])
        # What confirmation buys: the precision of the confirmed pool against the
        # precision of the pool before anyone confirmed anything. A lift near zero
        # means the flag carries no information about whether the region is real.
        lift = ""
        if scores:
            confirmed_rate = rate([row for row in have if row.get(field)],
                                  lambda row: row["human"] == CORRECT)
            lift = f"lift {100 * (confirmed_rate - base):+.1f}pp on this kind's {100 * base:.1f}% base rate"
        show(f"confirmed by {tag}", scores,
             f1_ci(have, confirms, args.bootstrap, args.seed) if scores else None, lift)

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


# --------------------------------------------------------------------------
# Discovery geometry: what a candidate was sitting on, against what a human said
# --------------------------------------------------------------------------

# The four outcomes a discovery candidate can have once its geometry is joined to
# the human verdict. Ordered worst-to-best as training labels.
BLEED, GROWTH, ALARM, FIND = "edge bleed", "boundary growth", "false alarm", "new object"

# SAM proposals carry ZOD's four classes; discovery candidates carry Swin's three,
# in which cyclist and pedestrian are one `human` class (make_label_bundle's
# SWIN_CLASS_NAMES — an unconfirmed candidate has no finer class to show a
# labeler). Comparing the two raw would make every human-class candidate look as
# though it abuts nothing, so both sides are folded to Swin's vocabulary first.
# This is the same fold the downstream mIoU uses.
CLASS_FOLD = {"cyclist": "human", "pedestrian": "human"}


def fold_class(name: str | None) -> str | None:
    return CLASS_FOLD.get(name, name)


GEOMETRY_GLOSS = {
    BLEED:  "rim on a same-class SAM mask, human rejected it — paint on background beside a "
            "correct object",
    GROWTH: "rim on a same-class SAM mask, human kept it — SAM under-segmented and the rim is "
            "really the object",
    ALARM:  "not on a same-class mask, human rejected it — a region Swin invented",
    FIND:   "not on a same-class mask, human kept it — an object SAM missed entirely",
}


def abuts_same_class(row: dict, class_of: dict[tuple[str, int], str]) -> tuple[str, int]:
    """
    What the candidate is lying against: `same`, `other`, or `none`.

    `touching_sam` is the list of SAM mask ids whose pixels fall in the candidate's
    dilated ring, written by make_label_bundle at bundle time. It survives the CSV
    round-trip as a bracketed string because `scalar` only casts things that start
    with a digit, so it is parsed back here.

    The class test is what separates the two mechanisms. A rim against a SAM mask
    of the *same* class is the pipeline finding the edge of an object it already
    has; a rim against a mask of a different class, or against nothing, is a claim
    about an object that is not already in the labels.
    """
    try:
        neighbours = json.loads(row.get("touching_sam") or "[]")
    except (TypeError, ValueError):
        neighbours = []
    classes = [class_of.get((row["frame_id"], nid)) for nid in neighbours]
    resolved = [name for name in classes if name is not None]
    if not neighbours:
        return "none", 0
    if any(name == fold_class(row["class"]) for name in resolved):
        return "same", len(classes) - len(resolved)
    return "other", len(classes) - len(resolved)


def geometry_of(row: dict, class_of: dict) -> str:
    """The four-way outcome. `same` decides the mechanism, the verdict decides the sign."""
    against, _ = abuts_same_class(row, class_of)
    kept = row["human"] == CORRECT
    if against == "same":
        return GROWTH if kept else BLEED
    return FIND if kept else ALARM


def labelling_order(export: Path) -> dict[str, int]:
    """
    Frame stem → 1-based rank in labelling order, from the first verdict in each frame.

    The reference is non-stationary (about 2 points of precision per 100 frames),
    so any absolute share below is also reported over the final frames, which is
    the calibration in force at the end of the pass.
    """
    first: dict[str, str] = {}
    with export.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("verdict"):
                continue
            stem, stamp = Path(row["frame"]).stem, row.get("answeredAt") or ""
            if stem not in first or stamp < first[stem]:
                first[stem] = stamp
    return {stem: rank + 1 for rank, (stem, _) in
            enumerate(sorted(first.items(), key=lambda item: item[1]))}


def geometry_table(rows: list[dict], class_of: dict, args, total: int | None = None) -> None:
    """One block of the four-way taxonomy: counts, shares, and a CI on each share."""
    total = total if total is not None else len(rows)
    if not rows:
        print("    (no candidates)")
        return
    tagged = [(geometry_of(row, class_of), row) for row in rows]
    for outcome in (BLEED, GROWTH, ALARM, FIND):
        subset = [row for kind, row in tagged if kind == outcome]
        share = len(subset) / total if total else 0.0
        ci = rate_ci(rows, lambda row, o=outcome: geometry_of(row, class_of) == o,
                     args.bootstrap, args.seed)
        interval = f"  [{100 * ci[0]:4.1f}, {100 * ci[1]:4.1f}]" if ci else ""
        print(f"    {outcome:<16} n={len(subset):<6} {100 * share:5.1f}%{interval}   {GEOMETRY_GLOSS[outcome]}")


def report_discovery_geometry(all_rows: list[dict], export: Path, args) -> None:
    """
    Why discovery costs downstream mIoU: what each candidate was sitting on.

    The precision tables above ask whether a candidate is a real object. This asks
    a different question — whether the pipeline was finding an object or finding
    the edge of one it already had — because the two have opposite implications.
    A rejected rim on an object SAM already segmented is not a missed find; it is
    paint applied beside a correct mask, and since candidate components are
    SAM-subtracted before they are formed, every one of those pixels lands outside
    the existing mask rather than harmlessly on top of it.

    `kind` (the ring-share test written at bundle time) and the class test here
    disagree often enough to report both: a candidate can clear the ring-share
    threshold and still be lying against a same-class mask, so `standalone` is not
    a synonym for "SAM had nothing here".
    """
    class_of = {(row["frame_id"], row["id"]): fold_class(row.get("class")) for row in all_rows}
    rows = [row for row in all_rows if row.get("source") == "discovery" and row["human"]]
    if not rows:
        print("\n  No answered discovery candidates in this export.")
        return
    unresolved = sum(abuts_same_class(row, class_of)[1] for row in rows)
    if unresolved:
        print(f"\n  NOTE: {unresolved} touching-mask ids did not resolve to a mask in the export; "
              f"those neighbours are ignored in the class test.")

    print(f"\n{len(rows):,} answered discovery candidates over "
          f"{len({row['frame_id'] for row in rows}):,} frames.")

    print("\nAs the bundle classified them (ring-share geometry) against the human verdict:")
    print(f"  {'':<12}{'human removed':>16}{'human kept':>14}{'total':>10}")
    for kind in ("fringe", "standalone"):
        subset = [row for row in rows if row.get("kind") == kind]
        removed = sum(1 for row in subset if row["human"] == INCORRECT)
        kept = len(subset) - removed
        print(f"  {kind:<12}{removed:>10,} {100 * removed / len(rows):4.1f}%"
              f"{kept:>8,} {100 * kept / len(rows):4.1f}%{len(subset):>10,}")

    print("\nWith the class test — is the mask it abuts the same class as the candidate?")
    geometry_table(rows, class_of, args)

    print("\n  The ring-share test and the class test disagree on:")
    for kind in ("fringe", "standalone"):
        subset = [row for row in rows if row.get("kind") == kind]
        if not subset:
            continue
        same = sum(1 for row in subset if abuts_same_class(row, class_of)[0] == "same")
        print(f"    {kind:<12} {same:,} of {len(subset):,} ({100 * same / len(subset):.1f}%) "
              f"abut a same-class SAM mask")

    print("\nPer class (share of that class's candidates):")
    for name in sorted({row["class"] for row in rows if row.get("class")}):
        subset = [row for row in rows if row.get("class") == name]
        counts = Counter(geometry_of(row, class_of) for row in subset)
        parts = "  ".join(f"{outcome} {100 * counts[outcome] / len(subset):5.1f}%"
                          for outcome in (BLEED, GROWTH, ALARM, FIND))
        print(f"  {name:<12} n={len(subset):<6} {parts}")

    # The reference drifts, so the shares above are a mixture over a moving
    # standard. The ratio between outcomes is far more stable than any absolute
    # rate, but where an absolute share carries an argument the paper's own rule is
    # to give the final-frames value too.
    order = labelling_order(export)
    have_order = [row for row in rows if row["frame_id"] in order]
    if len(have_order) < len(rows):
        print(f"\n  NOTE: {len(rows) - len(have_order)} candidates carry no labelling order; "
              f"the drift check below covers the rest.")
    for window in (200, 400):
        cutoff = max(order.values(), default=0) - window
        recent = [row for row in have_order if order[row["frame_id"]] > cutoff]
        if not recent:
            continue
        print(f"\nFinal {window} frames only ({len(recent):,} candidates) — the calibration in force "
              f"at the end of the pass:")
        geometry_table(recent, class_of, args)


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


class Tee:
    """Write the report to the terminal and to a file in one pass."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def report(args) -> None:
    manifest = json.loads(args.manifest.read_text()) if args.manifest else None
    rows, stats = load_masks(args.export)
    tags = args.tags or run_tags(rows)

    if manifest:
        job = manifest["job"]
        print(f"Job: {job['name']}   K={job['redundancy']}   "
              f"{manifest['progress']['completed']}/{manifest['progress']['tasks']} frames complete")
    print(f"Export: {args.export}")
    report_coverage(args.export, rows, stats, tags)

    # One job, two questions: `source` says which masks answer which, so the
    # split happens here rather than in the labeling.
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("source", "sam")].append(row)

    if args.only in (None, "sam") and by_source["sam"]:
        header("SAM PROPOSALS — triage accuracy of the agents as shipped")
        report_triage(by_source["sam"], tags, args)
    if args.only in (None, "variants") and by_source["sam"]:
        report_variants([row for row in by_source["sam"] if row["human"]], tags, args)
    if args.only in (None, "discovery"):
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
            header(f"DISCOVERY CANDIDATES ({kind}) — precision per stratum")
            table = report_discovery(rows_of_kind, tags, population, reason, args)
            if args.latex_out and kind == args.latex_kind:
                write_latex(table, args.latex_out, kind)

    if args.only in (None, "geometry") and by_source["discovery"]:
        header("DISCOVERY GEOMETRY — what the candidate was sitting on")
        report_discovery_geometry(rows, args.export, args)

    header("READING THIS REPORT")
    print("  Every rule above is scored on the same masks, so differences are paired.")
    print("  Rank triage variants on `bad kept` (wrong pixels entering the label) against")
    print("  `good lost` (objects omitted) — F1 pools the two and rewards keeping everything.")
    print("  A variant not separable here will not be separable in a downstream mIoU run either.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="job export CSV from label-front")
    parser.add_argument("--manifest", type=Path,
                        help="job manifest JSON from label-front — bundle-wide totals, for extrapolation")
    parser.add_argument("--tags", nargs="+",
                        help="runs to score (default: every run named in the export)")
    parser.add_argument("--only", choices=["sam", "variants", "discovery", "geometry"],
                        help="report one section only")
    parser.add_argument("--bootstrap", type=int, default=10000,
                        help="frame-clustered resamples for each 95%% CI (0 = skip)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--swin-threshold", type=float, default=config.SWIN_AGREEMENT_THRESHOLD)
    parser.add_argument("--hpc", action="store_true", help="read results from the HPC data root")
    parser.add_argument("--out", type=Path,
                        help="write the report here (default: <export>.report.txt)")
    parser.add_argument("--latex-out", type=Path, help="write the discovery paper table here")
    parser.add_argument("--latex-kind", choices=["standalone", "fringe"], default="standalone",
                        help="which kind of candidate the paper table reports (default: standalone)")
    args = parser.parse_args()

    if args.hpc:
        config.use_hpc()

    out = args.out or args.export.with_suffix(".report.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle, redirect_stdout(Tee(sys.stdout, handle)):
        report(args)
        print(f"\nReport written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
