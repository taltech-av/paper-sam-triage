"""
Each verification signal on its own, scored against the human verdicts.

The paper's agreement table scores the VLM crop verdict only inside the vote, so
a reader cannot see what the VLM contributes by itself or where its errors sit.
This script answers that from the closed export, without new model calls:

  * every signal applied alone (dense agreement, LiDAR support, crop verdict);
  * the crop verdict's answer mix on correct and on incorrect masks;
  * per-class error rates, where each verifier's bias shows;
  * overlap of the dense and crop decisions on the incorrect masks;
  * the vote's input patterns, i.e. which combination of signals lets the
    incorrect masks through.

"Crop verdict alone" keeps a mask only when the VLM confirms the target class,
the same definition as the "BBox VLM" rows of analyze_human_verification.py
(44.0 / 32.4 for LLaVA, 51.2 / 24.1 for Qwen), so the two reports agree.

    python analyze_single_signals.py --export human_verified_output/verify_export.csv
"""

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from sweep_triage_operating_points import (TAU_LARGE, TAU_LIDAR, TAU_SMALL,
                                           concordance, load, signals)

RUNS = ("llava_34b", "qwen2.5vl_72b_v2")
CLASSES = ("vehicle", "sign", "cyclist", "pedestrian")
CROP_STATE = {"valid": "confirmed", "invalid": "uncertain", "background": "contradiction"}


def score(rows, keep) -> dict:
    """FA / FR and pooled precision, recall, F1 (positive = human called it correct)."""
    tp = fp = fn = tn = 0
    for row in rows:
        kept = keep(row)
        if row["verdict"] == "correct":
            tp += kept
            fn += not kept
        else:
            fp += kept
            tn += not kept
    prec = 100 * tp / (tp + fp) if tp + fp else 0.0
    rec = 100 * tp / (tp + fn) if tp + fn else 0.0
    return dict(n=len(rows), fa=100 * fp / (fp + tn), fr=100 * fn / (tp + fn),
                prec=prec, rec=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0)


def line(name, s) -> str:
    return (f"  {name:34} n={s['n']:6,}  FA {s['fa']:5.1f}  FR {s['fr']:5.1f}"
            f"  prec {s['prec']:5.1f}  rec {s['rec']:5.1f}  F1 {s['f1']:5.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path,
                        default=Path("human_verified_output/verify_export.csv"))
    args = parser.parse_args()

    rows = load(args.export)
    bad = sum(row["verdict"] == "incorrect" for row in rows)
    print(f"{len(rows):,} judged SAM proposals, {bad:,} rejected by the annotator\n")

    def sig(row, run="llava_34b"):
        return signals(row, tau_lidar=TAU_LIDAR, tau_large=TAU_LARGE,
                       tau_small=TAU_SMALL, run=run)

    # The trained dense-only variant (replay_triage swin_only) applies one scalar
    # threshold, 0.30, to every class; the class-aware T_c is used inside the vote.
    # Scoring it that way reproduces the paper's 23.5 / 26.7 row.
    dense = lambda row: float(row["mask_swin_agreement"]) >= TAU_LARGE
    lidar = lambda row: row["mask_consistency"] == "pass"    # stored verdict, as in the paper
    crop = {run: (lambda row, run=run: row[f"mask_bbox_agent_{run}"] == "valid") for run in RUNS}
    crop_contra = {run: (lambda row, run=run: row[f"mask_bbox_agent_{run}"] != "background")
                   for run in RUNS}
    vote = {run: (lambda row, run=run: not concordance(*sig(row, run))) for run in RUNS}

    print("SIGNALS ALONE (FA = incorrect masks kept, FR = correct masks deleted)")
    print(line("keep every proposal", score(rows, lambda _row: True)))
    print(line("dense agreement alone", score(rows, dense)))
    print(line("LiDAR support alone", score(rows, lidar)))
    for run in RUNS:
        print(line(f"crop verdict alone, {run}", score(rows, crop[run])))
    for run in RUNS:
        print(line(f"delete on contradiction, {run}", score(rows, crop_contra[run])))
    for run in RUNS:
        print(line(f"three-signal vote, {run}", score(rows, vote[run])))

    print("\nCROP VERDICT MIX (share of masks in each human group)")
    for run in RUNS:
        for verdict in ("correct", "incorrect"):
            group = [row for row in rows if row["verdict"] == verdict]
            mix = Counter(CROP_STATE.get(row[f"mask_bbox_agent_{run}"], "missing") for row in group)
            parts = "  ".join(f"{k} {100 * mix[k] / len(group):5.1f}"
                              for k in ("confirmed", "uncertain", "contradiction", "missing"))
            print(f"  {run:18} {verdict:9}  {parts}")

    print("\nPER CLASS  FA / FR")
    header = "  ".join(f"{c:>13}" for c in CLASSES)
    print(f"  {'':34}{header}")
    verifiers = [("dense agreement alone", dense)] + [
        (f"crop verdict alone, {run}", crop[run]) for run in RUNS] + [
        (f"three-signal vote, {run}", vote[run]) for run in RUNS]
    for name, keep in verifiers:
        cells = []
        for cls in CLASSES:
            s = score([row for row in rows if row["class"] == cls], keep)
            cells.append(f"{s['fa']:5.1f} / {s['fr']:5.1f}")
        print(f"  {name:34}" + "  ".join(f"{c:>13}" for c in cells))
    shares = {cls: 100 * sum(r["verdict"] == "incorrect" for r in rows if r["class"] == cls)
              / sum(r["class"] == cls for r in rows) for cls in CLASSES}
    print("  annotator error rate              " + "  ".join(f"{shares[c]:13.1f}" for c in CLASSES))

    print("\nDENSE vs CROP ON THE INCORRECT MASKS (who deletes them)")
    wrong = [row for row in rows if row["verdict"] == "incorrect"]
    right = [row for row in rows if row["verdict"] == "correct"]
    for run in RUNS:
        c = Counter((dense(row), crop[run](row)) for row in wrong)
        n = len(wrong)
        print(f"  {run:18} both delete {100 * c[(False, False)] / n:5.1f}"
              f"  dense only {100 * c[(False, True)] / n:5.1f}"
              f"  crop only {100 * c[(True, False)] / n:5.1f}"
              f"  neither {100 * c[(True, True)] / n:5.1f}")
        c = Counter((dense(row), crop[run](row)) for row in right)
        n = len(right)
        print(f"  {'':18} on correct masks: both delete {100 * c[(False, False)] / n:5.1f}"
              f"  dense only {100 * c[(False, True)] / n:5.1f}"
              f"  crop only {100 * c[(True, False)] / n:5.1f}")
    both = Counter((crop[RUNS[0]](row), crop[RUNS[1]](row)) for row in wrong)
    print(f"  incorrect masks confirmed by both VLMs {100 * both[(True, True)] / len(wrong):.1f}"
          f", by neither {100 * both[(False, False)] / len(wrong):.1f}")

    print("\nVOTE INPUT PATTERNS (dense, LiDAR, crop) -> vote decision")
    for run in RUNS:
        cells = defaultdict(lambda: [0, 0])
        for row in rows:
            _, quality, consistency = sig(row, run)
            state = CROP_STATE.get(row[f"mask_bbox_agent_{run}"], "missing")
            key = (quality, consistency, state, "keep" if vote[run](row) else "delete")
            cells[key][row["verdict"] == "incorrect"] += 1
        kept_bad = sum(v[1] for k, v in cells.items() if k[3] == "keep")
        print(f"  {run}  (share of the {kept_bad:,} incorrect masks the vote keeps)")
        for key, (good, badn) in sorted(cells.items(), key=lambda kv: -kv[1][1]):
            share = f"{100 * badn / kept_bad:5.1f}" if key[3] == "keep" else "    -"
            print(f"    dense {key[0]:4}  lidar {key[1]:4}  crop {key[2]:13} -> {key[3]:6}"
                  f"  n={good + badn:6,}  incorrect {100 * badn / (good + badn):5.1f}%"
                  f"  of kept-incorrect {share}")


if __name__ == "__main__":
    main()
