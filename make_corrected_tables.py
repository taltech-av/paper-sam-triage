#!/usr/bin/env python3
"""
Generate the paper's downstream-result tables from common-reference
per-frame metric dumps (fusion-training/dump_frame_metrics.py output).

All variants are evaluated on the clean-partition test split against the
manually curated annotations — the single shared reference — with paired,
weather-stratified bootstrap CIs on mIoU. Rerun this after new checkpoints
are dumped (seed runs, limitation-fix / consensus variants); rows whose dump
JSON is missing are skipped with a warning, so the tables stay consistent.

Usage:
    python make_corrected_tables.py \\
        --metrics-dir /run/media/tom/ml/projects/fusion-training/logs/vlm/frame_metrics/common_ref \\
        --out-dir paper/tables
"""

import argparse
from pathlib import Path

import numpy as np

from bootstrap_miou import bootstrap, ci, load_variant, point_estimates

WEATHER_ORDER = ["day_fair", "day_rain", "night_fair", "night_rain", "snow"]
WEATHER_HEADERS = ["Day-fair", "Day-rain", "Night-fair", "Night-rain", "Snow"]

# Reference row for the paired differences. Swin-only is the natural baseline
# for this table: it is the strongest variant and uses no VLM at all, so every
# other row reads as "what the VLM stage added", which is the question the
# ablation exists to answer.
REFERENCE_STEM = "shared_swin_only_fusion"
REFERENCE_LABEL = "Swin only"

# No superseded rows: every run in ABLATION_ROWS trains on the corrected
# 4,110-frame annotations. Kept as a hook in case a legacy dump is ever
# re-added alongside a fresh one.
SUPERSEDED_STEMS: set[str] = set()

# (dump-file stem, variant label, description, VLM label)
ABLATION_ROWS = [
    ("baseline", None, None, None),
    ("human_verified_fusion", "Human-verified", "All 4,110 flagged frames manually verified; incorrect masks removed", r"\textemdash"),
    ("shared_gt_fusion", "SAM (curated)", "Manually reviewed clean-partition annotations", r"\textemdash"),
    ("vlm_independent", None, None, None),
    ("shared_raw_sam_fusion", "Raw SAM", "No triage; SAM pseudo-labels unchanged", r"\textemdash"),
    ("shared_swin_only_fusion", "Swin only", "Swin agreement threshold; no VLM", r"\textemdash"),
    ("shared_swin_discovery_noVLM_ccm_fusion", "Swin + disc.\\ (no VLM)",
     "Swin filtering + all Swin-detected discoveries", r"\textemdash"),
    ("vlm_dependent", None, None, None),
    ("llava_triage_fusion", "Triage", "Swin + VLM + consistency", "LLaVA"),
    ("qwen_triage_fusion", "Triage", None, "Qwen"),
    ("llava_full_fusion", "Triage + disc.", "Full pipeline", "LLaVA"),
    ("qwen_full_fusion", "Triage + disc.", None, "Qwen"),
]

SECTION_HEADERS = {
    "baseline": r"\multicolumn{9}{l}{\textit{Human reference}} \\",
    "vlm_independent": r"\multicolumn{9}{l}{\textit{VLM-independent — trained on 4{,}110 flagged-partition frames}} \\",
    "vlm_dependent": r"\multicolumn{9}{l}{\textit{VLM-dependent — trained on 4{,}110 flagged-partition frames}} \\",
}

WEATHER_ROWS = [
    ("human_verified_fusion", "Human-verified", "--"),
    ("shared_raw_sam_fusion", "Raw SAM", "--"),
    ("shared_swin_only_fusion", "Swin only", "--"),
    ("shared_swin_discovery_noVLM_ccm_fusion", "Swin + disc.\\ (no VLM)", "--"),
    ("llava_full_fusion", "Triage + disc.", "LLaVA-1.6-34B"),
    ("qwen_full_fusion", "Triage + disc.", "Qwen2.5-VL-72B"),
]

MODALITY_GROUPS = [
    ("SAM (curated) baseline — clean partition", "--",
     [("shared_gt_rgb", "RGB"), ("shared_gt_lidar", "LiDAR"), ("shared_gt_fusion", "CrossFusion")]),
]


def fmt(x, bold=False):
    s = f"{100 * x:.1f}"
    return rf"\textbf{{{s}}}" if bold else s


def condition_miou(variant, cond):
    c = variant["conditions"][cond]
    return float(np.mean(c["overlap"].sum(0) / (c["union"].sum(0) + 1e-6)))


def write_ablation(variants, boots, out: Path):
    available = {k for k in variants}
    # bold = best value per metric column among available rows
    stems = [s for s, label, *_ in ABLATION_ROWS if label and s in available]
    pes = {s: point_estimates(variants[s]) for s in stems}
    # A planned run with no dump yet is rendered as a pending row rather than
    # dropped, so the table always shows the full experiment and a reader can
    # see what is outstanding. With no dumps at all the whole table is pending.
    best = {
        "veh": max((pes[s]["per_class"][0] for s in stems), default=None),
        "sign": max((pes[s]["per_class"][1] for s in stems), default=None),
        "human": max((pes[s]["per_class"][2] for s in stems), default=None),
        "miou": max((pes[s]["miou"] for s in stems), default=None),
        "fw": max((pes[s]["fw_iou"] for s in stems), default=None),
    }
    n_frames = (sum(len(variants[stems[0]]["conditions"][c]["frames"])
                    for c in variants[stems[0]]["conditions"]) if stems else 0)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Annotation variant ablation on CLFTv2-Base (RGB--LiDAR CrossFusion).",
        r"         All variants are evaluated on the same " + (str(n_frames) if n_frames else "held-out")
        + r" test frames against the human-verified reference,",
        r"         so scores are directly comparable across rows.",
        r"         mIoU averages vehicle, sign, and human (cyclist + pedestrian combined) and is the mean over the five weather-condition mIoUs;",
        r"         fw-IoU weights each class by its pixel frequency.",
        r"         The last column reports the \emph{paired} difference in mIoU against the "
        + REFERENCE_LABEL + r" variant, with a 95\% CI from a bootstrap over test frames",
        r"         (10{,}000 resamples, stratified by weather condition; the same resample indices are applied to every variant).",
        r"         A paired difference is the statistic this design supports: all variants are evaluated on the same frames,",
        r"         so per-variant marginal CIs overlap far more than the differences themselves and understate the separation.",
        r"         An interval excluding zero means the variant differs significantly from " + REFERENCE_LABEL + r".",
        r"         Bold marks the best result in each metric column.",
        r"         Rows without numbers are planned runs whose checkpoints do not exist yet.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llc ccccc l}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Description} & \textbf{VLM} &",
        r"  \textbf{Veh.} & \textbf{Sign} & \textbf{Human} & \textbf{mIoU} & \textbf{fw-IoU} & "
        r"\textbf{$\Delta$mIoU vs " + REFERENCE_LABEL + r" (95\% CI)} \\",
        r"\midrule",
    ]
    # drop section headers whose rows are all missing
    section_has_rows = {}
    current = None
    for stem, label, *_ in ABLATION_ROWS:
        if label is None:
            current = stem
            section_has_rows.setdefault(current, False)
        elif stem in available:
            section_has_rows[current] = True

    pending_space = False
    for stem, label, desc, vlm in ABLATION_ROWS:
        if label is None:
            if stem in SECTION_HEADERS and section_has_rows.get(stem):
                if pending_space:
                    lines.append(r"\addlinespace")
                lines.append(SECTION_HEADERS[stem])
                pending_space = False
            continue
        if stem not in available:
            print(f"  (ablation) no dump yet, row rendered as pending: {stem}")
            lines.append(f"{label} & {desc or ''} & {vlm} & "
                         + " & ".join([r"\textemdash"] * 5)
                         + r" & \textit{pending} \\")
            pending_space = True
            continue
        pe = pes[stem]
        if stem == REFERENCE_STEM:
            delta_cell = r"\textemdash\ (reference)"
        elif REFERENCE_STEM in boots:
            d = boots[stem] - boots[REFERENCE_STEM]
            lo, hi = ci(d)
            point = 100 * (pe["miou"] - pes[REFERENCE_STEM]["miou"])
            delta_cell = f"${point:+.1f}$ [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        else:
            # No reference dump — fall back to the marginal CI rather than
            # silently printing a difference against nothing.
            lo, hi = ci(boots[stem])
            delta_cell = f"[{100 * lo:.1f}, {100 * hi:.1f}] (marginal)"
        cells = [
            fmt(pe["per_class"][0], pe["per_class"][0] == best["veh"]),
            fmt(pe["per_class"][1], pe["per_class"][1] == best["sign"]),
            fmt(pe["per_class"][2], pe["per_class"][2] == best["human"]),
            fmt(pe["miou"], pe["miou"] == best["miou"]),
            fmt(pe["fw_iou"], pe["fw_iou"] == best["fw"]),
            delta_cell,
        ]
        desc_cell = desc if desc is not None else ""
        mark = r"\textsuperscript{\dag}" if stem in SUPERSEDED_STEMS else ""
        lines.append(f"{label}{mark} & {desc_cell} & {vlm} & " + " & ".join(cells) + r" \\")
        pending_space = True
    lines += [r"\bottomrule", r"\end{tabular}}"]
    if any(s in available for s in SUPERSEDED_STEMS):
        lines += [r"\\[2pt]", r"\footnotesize\textsuperscript{\dag} Superseded --- see caption."]
    lines += [r"\end{table*}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def _pending_table(out: Path, label: str, caption: str) -> None:
    """Overwrite a table with an explicit placeholder.

    Written rather than skipped on purpose: leaving the previous .tex in place
    would keep numbers from checkpoints that no longer exist, and the paper
    would still compile and still look finished.
    """
    out.write_text("\n".join([
        r"\begin{table}[t]", r"\centering",
        r"\caption{" + caption + r"}",
        r"\label{" + label + r"}",
        r"\begin{tabular}{l}", r"\toprule",
        r"\textit{Pending — awaiting retrained checkpoints.} \\",
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ]))
    print(f"wrote {out} (pending placeholder)")


def write_weather(variants, out: Path):
    rows = [(s, l, v) for s, l, v in WEATHER_ROWS if s in variants]
    if not rows:
        return _pending_table(out, "tab:weather_miou",
                              "Per-weather mIoU and fw-IoU on CLFTv2-Base CrossFusion.")
    n = {c: len(variants[rows[0][0]]["conditions"][c]["frames"]) for c in WEATHER_ORDER}
    best_cond = {c: max(condition_miou(variants[s], c) for s, *_ in rows) for c in WEATHER_ORDER}
    best_miou = max(point_estimates(variants[s])["miou"] for s, *_ in rows)
    best_fw = max(point_estimates(variants[s])["fw_iou"] for s, *_ in rows)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Per-weather mIoU and frequency-weighted IoU (fw-IoU) on CLFTv2-Base CrossFusion for key annotation variants.",
        r"         All variants are evaluated on the same clean-partition test frames against the curated annotations",
        r"         (" + ", ".join(f"{h.lower()}: {n[c]}" for c, h in zip(WEATHER_ORDER, WEATHER_HEADERS)) + r" frames).",
        r"         fw-IoU weights each class by its pixel frequency, so it is dominated by vehicles.",
        r"         Bold marks the best result per column.}",
        r"\label{tab:weather_miou}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llccccc cc}",
        r"\toprule",
        r"\textbf{Annotation} & \textbf{VLM} & " + " & ".join(rf"\textbf{{{h}}}" for h in WEATHER_HEADERS)
        + r" & \textbf{mIoU} & \textbf{fw-IoU} \\",
        r"\midrule",
    ]
    for stem, label, vlm in rows:
        pe = point_estimates(variants[stem])
        cells = [fmt(condition_miou(variants[stem], c),
                     condition_miou(variants[stem], c) == best_cond[c]) for c in WEATHER_ORDER]
        cells.append(fmt(pe["miou"], pe["miou"] == best_miou))
        cells.append(fmt(pe["fw_iou"], pe["fw_iou"] == best_fw))
        lines.append(f"{label} & {vlm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def write_modality(variants, out: Path):
    if not any(s in variants for _, _, g in MODALITY_GROUPS for s, _ in g):
        return _pending_table(out, "tab:modality_ablation",
                              "Modality ablation (RGB, LiDAR, CrossFusion) on CLFTv2-Base.")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Modality ablation on CLFTv2-Base using Swin + disc.\ as the annotation variant.",
        r"         RGB uses camera only; LiDAR uses depth projection only; CrossFusion uses both.",
        r"         All rows are evaluated on the same clean-partition test frames against the curated annotations.",
        r"         mIoU averages vehicle, sign, and human; fw-IoU weights each class by pixel frequency.}",
        r"\label{tab:modality_ablation}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textbf{Modality} & \textbf{VLM} & \textbf{Veh.} & \textbf{Sign} & \textbf{Human} & \textbf{mIoU} & \textbf{fw-IoU} \\",
        r"\midrule",
    ]
    first = True
    for title, vlm, group in MODALITY_GROUPS:
        if not first:
            lines.append(r"\addlinespace")
        first = False
        lines.append(rf"\multicolumn{{7}}{{l}}{{\textit{{{title}}}}} \\")
        for stem, modality in group:
            if stem not in variants:
                print(f"  (modality) missing dump, row skipped: {stem}")
                continue
            pe = point_estimates(variants[stem])
            cells = [fmt(pe["per_class"][0]), fmt(pe["per_class"][1]), fmt(pe["per_class"][2]),
                     fmt(pe["miou"]), fmt(pe["fw_iou"])]
            lines.append(f"{modality} & {vlm} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("paper/tables"))
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()

    variants = {}
    for f in sorted(args.metrics_dir.glob("*.json")):
        variants[f.stem] = load_variant(f)
    print(f"{len(variants)} dumps loaded from {args.metrics_dir}")

    # resolve "preferred|fallback" stems in the row specs to whichever dump exists
    def resolve(spec: str) -> str:
        for alt in spec.split("|"):
            if alt in variants:
                return alt
        return spec.split("|")[0]

    for rows in (ABLATION_ROWS, WEATHER_ROWS):
        for i, row in enumerate(rows):
            if row[1] is not None and "|" in row[0]:
                rows[i] = (resolve(row[0]),) + tuple(row[1:])
    for gi, (title, vlm, group) in enumerate(MODALITY_GROUPS):
        MODALITY_GROUPS[gi] = (title, vlm, [
            (resolve(stem) if "|" in stem else stem, modality) for stem, modality in group
        ])

    # With no dumps on disk every row is pending, and there is nothing to
    # resample — bootstrap() indexes the first variant's conditions.
    boots = bootstrap(list(variants.values()), args.n_boot, seed=0) if variants else {}

    write_ablation(variants, boots, args.out_dir / "ablation.tex")
    write_weather(variants, args.out_dir / "weather_miou.tex")
    write_modality(variants, args.out_dir / "modality_ablation.tex")


if __name__ == "__main__":
    main()
