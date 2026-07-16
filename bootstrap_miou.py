#!/usr/bin/env python3
"""
Paired, weather-stratified bootstrap CIs for downstream mIoU.

Reads per-frame IoU count dumps produced by fusion-training/dump_frame_metrics.py
(one JSON per annotation variant, all evaluated on the SAME test frames against
the SAME reference annotation) and computes:

  1. Point estimates per variant: per-class IoU, mIoU (mean over weather-
     condition mIoUs, matching the paper protocol), and fw-IoU.
  2. Bootstrap 95% CIs: test frames are resampled with replacement WITHIN each
     weather condition (preserving the test composition); identical resample
     indices are used for every variant, so paired deltas between variants are
     tight (shared frame-sampling noise cancels).
  3. Paired deltas for requested variant pairs with CI and P(Δ ≤ 0).

Usage:
    python bootstrap_miou.py --metrics-dir /path/to/frame_metrics/common_ref
    python bootstrap_miou.py --metrics-dir ... \\
        --pair shared_swin_discovery_noVLM_fusion:llava_swin_discovery_fusion
"""

import argparse
import json
from pathlib import Path

import numpy as np

EPS = 1e-6


def load_variant(path: Path) -> dict:
    d = json.loads(path.read_text())
    out = {"name": path.stem, "classes": d["eval_classes"], "conditions": {}}
    for cond, frames in d["conditions"].items():
        out["conditions"][cond] = {
            "frames": [f["frame"] for f in frames],
            "overlap": np.array([f["overlap"] for f in frames]),
            "union": np.array([f["union"] for f in frames]),
            "label": np.array([f["label"] for f in frames]),
        }
    return out


def check_aligned(variants: list[dict]) -> None:
    ref = variants[0]
    for v in variants[1:]:
        if set(v["conditions"]) != set(ref["conditions"]):
            raise SystemExit(f"{v['name']}: condition set differs from {ref['name']}")
        for cond in ref["conditions"]:
            if v["conditions"][cond]["frames"] != ref["conditions"][cond]["frames"]:
                raise SystemExit(f"{v['name']}/{cond}: frame list differs from "
                                 f"{ref['name']} — dumps are not paired")


def miou_from_counts(overlap_sum: np.ndarray, union_sum: np.ndarray) -> float:
    return float(np.mean(overlap_sum / (union_sum + EPS)))


def point_estimates(v: dict) -> dict:
    per_cond = {}
    overlap_all = union_all = label_all = 0
    for cond, c in v["conditions"].items():
        per_cond[cond] = miou_from_counts(c["overlap"].sum(0), c["union"].sum(0))
        overlap_all = overlap_all + c["overlap"].sum(0)
        union_all = union_all + c["union"].sum(0)
        label_all = label_all + c["label"].sum(0)
    iou_all = overlap_all / (union_all + EPS)
    freq = label_all / label_all.sum()
    return {
        "per_class": iou_all,
        "miou": float(np.mean(list(per_cond.values()))),
        "fw_iou": float(np.sum(freq * iou_all)),
        "per_condition": per_cond,
    }


def bootstrap(variants: list[dict], n_boot: int, seed: int) -> dict[str, np.ndarray]:
    """Return {variant_name: (n_boot,) array of overall mIoU} using shared
    per-condition resample indices across variants."""
    rng = np.random.default_rng(seed)
    conds = list(variants[0]["conditions"])
    n_per_cond = {c: len(variants[0]["conditions"][c]["frames"]) for c in conds}
    # (n_boot, n_frames) index arrays, one per condition, shared by all variants
    idx = {c: rng.integers(0, n, size=(n_boot, n)) for c, n in n_per_cond.items()}

    result = {}
    for v in variants:
        cond_mious = np.empty((n_boot, len(conds)))
        for j, c in enumerate(conds):
            ov, un = v["conditions"][c]["overlap"], v["conditions"][c]["union"]
            # (n_boot, n_frames, n_classes) gather → sum over frames
            ov_s = ov[idx[c]].sum(axis=1)
            un_s = un[idx[c]].sum(axis=1)
            cond_mious[:, j] = (ov_s / (un_s + EPS)).mean(axis=1)
        result[v["name"]] = cond_mious.mean(axis=1)
    return result


def ci(samples: np.ndarray, level: float = 95.0) -> tuple[float, float]:
    lo, hi = np.percentile(samples, [(100 - level) / 2, 100 - (100 - level) / 2])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True,
                        help="Directory of dump_frame_metrics.py JSONs (one per variant)")
    parser.add_argument("--pair", action="append", default=[],
                        help="Variant pair 'a:b' for paired delta (a minus b); repeatable")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.metrics_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No JSONs in {args.metrics_dir}")
    variants = [load_variant(f) for f in files]
    check_aligned(variants)
    classes = variants[0]["classes"]
    conds = list(variants[0]["conditions"])
    n_frames = sum(len(variants[0]["conditions"][c]["frames"]) for c in conds)
    print(f"{len(variants)} variants, {n_frames} common test frames, "
          f"{args.n_boot} bootstrap resamples (stratified by {len(conds)} conditions)\n")

    boot = bootstrap(variants, args.n_boot, args.seed)

    header = (f"{'variant':42s} " + "".join(f"{c[:4]:>7s}" for c in classes)
              + f"{'mIoU':>7s}{'fw-IoU':>8s}   95% CI (mIoU)")
    print(header)
    print("─" * len(header))
    order = sorted(variants, key=lambda v: -point_estimates(v)["miou"])
    for v in order:
        pe = point_estimates(v)
        lo, hi = ci(boot[v["name"]])
        row = f"{v['name']:42s} "
        row += "".join(f"{100*x:7.1f}" for x in pe["per_class"])
        row += f"{100*pe['miou']:7.1f}{100*pe['fw_iou']:8.1f}"
        row += f"   [{100*lo:.1f}, {100*hi:.1f}]"
        print(row)

    if args.pair:
        print("\nPaired deltas (a − b, shared resamples):")
        names = {v["name"] for v in variants}
        for pair in args.pair:
            a, b = pair.split(":")
            if a not in names or b not in names:
                print(f"  {pair}: unknown variant name(s), available: {sorted(names)}")
                continue
            delta = boot[a] - boot[b]
            lo, hi = ci(delta)
            p_le0 = float((delta <= 0).mean())
            print(f"  {a} − {b}:")
            print(f"    Δ mIoU = {100*delta.mean():+.2f}  [{100*lo:+.2f}, {100*hi:+.2f}]"
                  f"   P(Δ≤0) = {p_le0:.4f}")


if __name__ == "__main__":
    main()
