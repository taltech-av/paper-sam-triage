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

# (dump-file stem, variant label, description, VLM label)
ABLATION_ROWS = [
    ("baseline", None, None, None),
    ("shared_gt_fusion", "SAM (curated)", "Manually reviewed clean-partition annotations", r"\textemdash"),
    ("vlm_independent", None, None, None),
    ("shared_raw_sam_fusion", "Raw SAM", "No triage; SAM pseudo-labels unchanged", r"\textemdash"),
    ("shared_swin_only_fusion", "Swin only", "Swin agreement threshold; no VLM", r"\textemdash"),
    # exact-CC-mask retrain preferred over the legacy bbox-rectangle run
    ("shared_swin_discovery_noVLM_ccm_fusion|shared_swin_discovery_noVLM_fusion",
     "Swin + disc.\\ (no VLM)", "Swin filtering + all Swin-detected discoveries", r"\textemdash"),
    ("vlm_dependent", None, None, None),
    ("qwen_vlm_only_fusion", "VLM only", "BBox VLM + consistency; Swin quality ignored", "Qwen"),
    ("llava_vlm_only_fusion", "VLM only", None, "LLaVA"),
    ("qwen_triage_fusion", "Triage", "Swin + VLM + consistency", "Qwen"),
    ("llava_triage_fusion", "Triage", None, "LLaVA"),
    ("qwen_swin_discovery_ccm_fusion|qwen_swin_discovery_fusion",
     "Swin + disc.", "Swin filtering + VLM-confirmed discovery", "Qwen"),
    ("llava_swin_discovery_ccm_fusion|llava_swin_discovery_fusion",
     "Swin + disc.", None, "LLaVA"),
    ("qwen_full_fusion", "Triage + disc.", "Full pipeline", "Qwen"),
    ("llava_full_fusion", "Triage + disc.", None, "LLaVA"),
    ("rule_ablations", None, None, None),
    ("qwen_disjunctive_reject_fusion", "Disjunctive reject", "Any single negative signal rejects", "Qwen"),
    ("llava_disjunctive_reject_fusion", "Disjunctive reject", None, "LLaVA"),
    ("qwen_uniform_tau_fusion", "Uniform $\\tau_q$", "Single threshold across all classes", "Qwen"),
    ("llava_uniform_tau_fusion", "Uniform $\\tau_q$", None, "LLaVA"),
    ("new_variants", None, None, None),
    ("llava_limits_fixed_discovery_fusion", "Triage (fixed) + disc.", "Triage with both limitation fixes + confirmed discovery", "LLaVA"),
    ("merged_consensus_union_disc_both_fusion", "Consensus triage + disc.", "Reject / add discovery only when both VLMs agree", "Both"),
    ("merged_consensus_swin_disc_both_fusion", "Swin + consensus disc.", "Swin filtering + discovery confirmed by both VLMs", "Both"),
]

SECTION_HEADERS = {
    "baseline": r"\multicolumn{9}{l}{\textit{Baseline — trained on 2{,}319 curated clean-partition frames}} \\",
    "vlm_independent": r"\multicolumn{9}{l}{\textit{VLM-independent — trained on 4{,}135 flagged-partition frames}} \\",
    "vlm_dependent": r"\multicolumn{9}{l}{\textit{VLM-dependent variants — trained on 4{,}135 flagged-partition frames}} \\",
    "rule_ablations": r"\multicolumn{9}{l}{\textit{Triage rule ablations (offline replay)}} \\",
    "new_variants": r"\multicolumn{9}{l}{\textit{Limitation fixes and two-VLM consensus (offline replay)}} \\",
}

WEATHER_ROWS = [
    ("shared_gt_fusion", "SAM (curated)", "--"),
    ("shared_raw_sam_fusion", "Raw SAM", "--"),
    ("shared_swin_only_fusion", "Swin only", "--"),
    ("shared_swin_discovery_noVLM_ccm_fusion|shared_swin_discovery_noVLM_fusion",
     "Swin + disc.\\ (no VLM)", "--"),
    ("qwen_swin_discovery_ccm_fusion|qwen_swin_discovery_fusion", "Swin + disc.", "Qwen2.5-VL-72B"),
    ("llava_swin_discovery_ccm_fusion|llava_swin_discovery_fusion", "Swin + disc.", "LLaVA-1.6-34B"),
    ("qwen_full_fusion", "Triage + disc.", "Qwen2.5-VL-72B"),
    ("llava_full_fusion", "Triage + disc.", "LLaVA-1.6-34B"),
]

MODALITY_GROUPS = [
    ("SAM (curated) baseline — clean partition", "--",
     [("shared_gt_rgb", "RGB"), ("shared_gt_lidar", "LiDAR"), ("shared_gt_fusion", "CrossFusion")]),
    # RGB/LiDAR sub-rows for the VLM discovery variants were trained on
    # bbox-rectangle-corrupted annotations and have been dropped rather than
    # retrained (see eval-protocol memory); only CrossFusion (exact-mask
    # retrain pending) remains per VLM.
    ("Swin + disc.\\ — Qwen2.5-VL-72B", "Qwen2.5-VL-72B",
     [("qwen_swin_discovery_ccm_fusion|qwen_swin_discovery_fusion", "CrossFusion")]),
    ("Swin + disc.\\ — LLaVA-1.6-34B", "LLaVA-1.6-34B",
     [("llava_swin_discovery_ccm_fusion|llava_swin_discovery_fusion", "CrossFusion")]),
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
    best = {
        "veh": max(pes[s]["per_class"][0] for s in stems),
        "sign": max(pes[s]["per_class"][1] for s in stems),
        "human": max(pes[s]["per_class"][2] for s in stems),
        "miou": max(pes[s]["miou"] for s in stems),
        "fw": max(pes[s]["fw_iou"] for s in stems),
    }
    n_frames = sum(len(variants[stems[0]]["conditions"][c]["frames"])
                   for c in variants[stems[0]]["conditions"])

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Annotation variant ablation on CLFTv2-Base (RGB--LiDAR CrossFusion).",
        r"         All variants are evaluated on the same " + str(n_frames)
        + r" clean-partition test frames against the manually curated annotations,",
        r"         so scores are directly comparable across rows.",
        r"         mIoU averages vehicle, sign, and human (cyclist + pedestrian combined) and is the mean over the five weather-condition mIoUs;",
        r"         fw-IoU weights each class by its pixel frequency.",
        r"         95\% CIs are from a paired bootstrap over test frames (10{,}000 resamples, stratified by weather condition).",
        r"         Bold marks the best result in each metric column.}",
        r"\label{tab:ablation}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{llc ccccc l}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{Description} & \textbf{VLM} &",
        r"  \textbf{Veh.} & \textbf{Sign} & \textbf{Human} & \textbf{mIoU} & \textbf{fw-IoU} & \textbf{95\% CI (mIoU)} \\",
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
            print(f"  (ablation) missing dump, row skipped: {stem}")
            continue
        pe = pes[stem]
        lo, hi = ci(boots[stem])
        cells = [
            fmt(pe["per_class"][0], pe["per_class"][0] == best["veh"]),
            fmt(pe["per_class"][1], pe["per_class"][1] == best["sign"]),
            fmt(pe["per_class"][2], pe["per_class"][2] == best["human"]),
            fmt(pe["miou"], pe["miou"] == best["miou"]),
            fmt(pe["fw_iou"], pe["fw_iou"] == best["fw"]),
            f"[{100 * lo:.1f}, {100 * hi:.1f}]",
        ]
        desc_cell = desc if desc is not None else ""
        lines.append(f"{label} & {desc_cell} & {vlm} & " + " & ".join(cells) + r" \\")
        pending_space = True
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")


def write_weather(variants, out: Path):
    rows = [(s, l, v) for s, l, v in WEATHER_ROWS if s in variants]
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

    boots = bootstrap(list(variants.values()), args.n_boot, seed=0)

    write_ablation(variants, boots, args.out_dir / "ablation.tex")
    write_weather(variants, args.out_dir / "weather_miou.tex")
    write_modality(variants, args.out_dir / "modality_ablation.tex")


if __name__ == "__main__":
    main()
