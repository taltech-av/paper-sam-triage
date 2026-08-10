#!/usr/bin/env python3
"""
Create stratified train / val / test splits balanced by weather condition
and rare-class pixel presence (cyclist + pedestrian).

Reads a frame-list CSV (one 'camera/frame_XXXXXX.png' path per line), looks up
ZOD metadata for weather, counts class pixels from an annotation directory, then
allocates frames so that every split mirrors the overall weather × rare-class
distribution as closely as possible.

Usage:
    # Good frames (manually inspected clean partition)
    python make_splits.py --frames frames/good_frames.csv \\
        --out-dir /run/media/tom/ml/zod_temp/splits_good

    # Flagged frames (VLM-annotated partition)
    python make_splits.py --frames frames/bad_frames.csv \\
        --out-dir /run/media/tom/ml/zod_temp/splits_flagged

    # Custom ratios / annotation dir / seed
    python make_splits.py --frames frames/good_frames.csv \\
        --out-dir /run/media/tom/ml/zod_temp/splits_good \\
        --annotation-dir /run/media/tom/ml/zod_temp/annotation_sam \\
        --val-ratio 0.15 --test-ratio 0.15 --seed 42

Outputs (written to --out-dir):
    train.txt               training split
    val.txt                 validation split
    test.txt                combined test split
    test_day_fair.txt       per-weather test subsets
    test_day_rain.txt
    test_night_fair.txt
    test_night_rain.txt
    test_snow.txt
    frame_analysis.json     per-frame metadata cache (weather, pixel counts)
    report.txt              human-readable distribution summary
"""

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import config

# ── Paths ─────────────────────────────────────────────────────────────────────

# Taken from config rather than restated. These were hardcoded copies of the
# data root, and when the disk moved they silently pointed at nothing: the
# weather lookup then failed for every frame and fell back to "day_fair", which
# collapses the weather stratification and the per-weather test subsets without
# raising. The run still "succeeded".
ZOD_META_ROOT   = config.ZOD_DATA_ROOT
DEFAULT_ANN_DIR = config.ANNOTATION_SAM_DIR

# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES   = {0: "background", 1: "ignore", 2: "vehicle", 3: "sign", 4: "cyclist", 5: "pedestrian"}
OBJECT_CLASSES = [2, 3, 4, 5]
RARE_CLASSES   = [4, 5]   # cyclist, pedestrian

# Minimum pixels for a class to be considered "present" in a frame
RARE_PIXEL_THRESHOLD = 50

# ── Weather mapping ───────────────────────────────────────────────────────────

WEATHER_CATEGORIES = ["day_fair", "day_rain", "night_fair", "night_rain", "snow"]

# scraped_weather value → canonical category (None = resolve via time_of_day)
_WEATHER_MAP = {
    "clear-day":           "day_fair",
    "partly-cloudy-day":   "day_fair",
    "cloudy":              "day_fair",
    "wind":                "day_fair",
    "fog":                 "day_fair",
    "clear-night":         "night_fair",
    "partly-cloudy-night": "night_fair",
    "snow":                "snow",
    "rain":                None,   # → day_rain or night_rain depending on time_of_day
}


# ── Frame parsing ─────────────────────────────────────────────────────────────

def parse_frame_id(line: str) -> str:
    """'camera/frame_099985.png'  →  '099985'"""
    return line.strip().replace("camera/frame_", "").replace(".png", "")


def frame_to_path(frame_id: str) -> str:
    return f"camera/frame_{frame_id}.png"


# ── Per-frame analysis ────────────────────────────────────────────────────────

def get_weather(frame_id: str) -> str:
    meta_path = ZOD_META_ROOT / frame_id / "metadata.json"
    if not meta_path.exists():
        return "day_fair"
    meta = json.loads(meta_path.read_text())
    raw = meta.get("scraped_weather", "")
    tod = meta.get("time_of_day", "day")
    cat = _WEATHER_MAP.get(raw)
    if cat is None:  # rain
        cat = "day_rain" if tod == "day" else "night_rain"
    return cat


def count_pixels(frame_id: str, ann_dir: Path) -> dict[int, int]:
    ann_path = ann_dir / f"frame_{frame_id}.png"
    if not ann_path.exists():
        return {}
    arr = np.array(Image.open(ann_path))
    return {cls: int((arr == cls).sum()) for cls in range(6) if (arr == cls).any()}


def analyze_frames(frame_ids: list[str], ann_dir: Path) -> list[dict]:
    records = []
    for fid in tqdm(frame_ids, desc="Analyzing frames", unit="frame"):
        pixels = count_pixels(fid, ann_dir)
        weather = get_weather(fid)
        rare_pixels = sum(pixels.get(c, 0) for c in RARE_CLASSES)
        records.append({
            "frame_id": fid,
            "weather":  weather,
            "pixel_counts": pixels,
            "has_rare":  rare_pixels >= RARE_PIXEL_THRESHOLD,
            "rare_pixels": rare_pixels,
            "object_pixels": sum(pixels.get(c, 0) for c in OBJECT_CLASSES),
            "total_pixels":  sum(pixels.values()),
        })
    return records


# ── Stratified splitting ──────────────────────────────────────────────────────

def _stratum(rec: dict) -> str:
    return f"{rec['weather']}__{'rare' if rec['has_rare'] else 'common'}"


def make_splits(
    records: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_test_per_weather: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Stratify by (weather × has_rare_class) and allocate test, val, train
    proportionally within each stratum so every split mirrors the full distribution.

    min_test_per_weather: ensure at least this many frames per weather category
    end up in the test set (raises the effective test ratio for small categories).
    """
    rng = random.Random(seed)

    # Pre-compute per-weather minimum test counts so small conditions get enough
    # frames for reliable mIoU evaluation, regardless of the global test_ratio.
    weather_test_floor: dict[str, int] = defaultdict(int)
    if min_test_per_weather > 0:
        by_weather: dict[str, int] = defaultdict(int)
        for r in records:
            by_weather[r["weather"]] += 1
        for w, n in by_weather.items():
            needed = max(round(n * test_ratio), min_test_per_weather)
            weather_test_floor[w] = min(needed, n - 2)  # always keep ≥2 for train

    strata: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        strata[_stratum(rec)].append(rec)

    # First pass: allocate test frames per weather, respecting the floor
    weather_test_allocated: dict[str, int] = defaultdict(int)
    train, val, test = [], [], []

    for key, group in sorted(strata.items()):
        rng.shuffle(group)
        n = len(group)
        weather = group[0]["weather"]

        # Effective test ratio: use global ratio but ensure the weather floor is met
        floor = weather_test_floor.get(weather, 0)
        n_test = max(round(n * test_ratio),
                     floor - weather_test_allocated[weather])
        n_test = min(n_test, n - 2)  # always leave at least 2 frames for train+val
        n_test = max(1, n_test)
        weather_test_allocated[weather] += n_test

        n_val = max(1, round(n * val_ratio))

        # Guard: ensure at least 1 frame remains for train
        while n_test + n_val >= n and n_test + n_val > 1:
            if n_test >= n_val:
                n_test = max(1, n_test - 1)
            else:
                n_val = max(1, n_val - 1)

        test  += group[:n_test]
        val   += group[n_test:n_test + n_val]
        train += group[n_test + n_val:]

    # Shuffle each split so order doesn't reflect stratification
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ── Reporting ─────────────────────────────────────────────────────────────────

def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


def distribution_table(records: list[dict], label: str) -> str:
    lines = [f"\n{label}  (n={len(records)})"]
    lines.append("  Weather distribution:")
    by_weather: dict[str, int] = defaultdict(int)
    for r in records:
        by_weather[r["weather"]] += 1
    for w in WEATHER_CATEGORIES:
        n = by_weather.get(w, 0)
        lines.append(f"    {w:15s}  {n:5d}  {_pct(n, len(records)):>6}")

    lines.append("  Rare-class presence:")
    n_rare = sum(1 for r in records if r["has_rare"])
    n_total = len(records)
    lines.append(f"    has cyclist/ped   {n_rare:5d}  {_pct(n_rare, n_total):>6}")
    lines.append(f"    none              {n_total - n_rare:5d}  {_pct(n_total - n_rare, n_total):>6}")

    lines.append("  Pixel distribution (object classes only, mean % per frame):")
    for cls in OBJECT_CLASSES:
        pcts = [100 * r["pixel_counts"].get(cls, 0) / r["total_pixels"]
                for r in records if r["total_pixels"]]
        mean = sum(pcts) / len(pcts) if pcts else 0
        present = sum(1 for r in records if r["pixel_counts"].get(cls, 0) >= RARE_PIXEL_THRESHOLD)
        lines.append(f"    {CLASS_NAMES[cls]:12s}  mean={mean:.3f}%  present={present}/{n_total}")

    return "\n".join(lines)


def write_report(records: list[dict], train: list[dict], val: list[dict], test: list[dict],
                 out_path: Path) -> None:
    sep = "=" * 72
    lines = [sep, "SPLIT DISTRIBUTION REPORT", sep]
    lines.append(distribution_table(records, "ALL FRAMES"))
    lines.append(distribution_table(train,   "TRAIN"))
    lines.append(distribution_table(val,     "VAL"))
    lines.append(distribution_table(test,    "TEST"))

    lines.append(f"\n{sep}")
    lines.append("PER-WEATHER TEST SUBSETS")
    lines.append(sep)
    for w in WEATHER_CATEGORIES:
        subset = [r for r in test if r["weather"] == w]
        lines.append(f"  test_{w:15s}  {len(subset):5d} frames")

    lines.append(f"\n{sep}")
    lines.append("STRATA DETAIL  (weather × rare-class presence)")
    lines.append(sep)
    all_strata: dict[str, dict] = defaultdict(lambda: {"total": 0, "train": 0, "val": 0, "test": 0})
    for split_name, split in [("total", records), ("train", train), ("val", val), ("test", test)]:
        for r in split:
            all_strata[_stratum(r)][split_name] += 1
    for key in sorted(all_strata):
        s = all_strata[key]
        lines.append(f"  {key:30s}  total={s['total']:4d}  "
                     f"train={s['train']:4d}  val={s['val']:4d}  test={s['test']:4d}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Report → {out_path}")


# ── Output writers ────────────────────────────────────────────────────────────

def write_split(records: list[dict], path: Path) -> None:
    path.write_text("\n".join(frame_to_path(r["frame_id"]) for r in records) + "\n")
    print(f"  {path.name:30s}  {len(records):5d} frames")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frames", required=True,
                        help="Frame-list CSV (one 'camera/frame_XXXXXX.png' per line)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for split files and analysis")
    parser.add_argument("--annotation-dir", default=str(DEFAULT_ANN_DIR),
                        help=f"Annotation PNG directory for pixel counting (default: {DEFAULT_ANN_DIR})")
    parser.add_argument("--val-ratio",  type=float, default=0.15, help="Val fraction (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test fraction (default: 0.15)")
    parser.add_argument("--min-test-per-weather", type=int, default=40,
                        help="Minimum test frames per weather category (default: 40). "
                             "Raises the effective test ratio for small conditions so "
                             "per-weather mIoU evaluation is statistically meaningful.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cache", action="store_true",
                        help="Load frame_analysis.json if it exists (skip re-counting pixels)")
    args = parser.parse_args()

    out_dir  = Path(args.out_dir)
    ann_dir  = Path(args.annotation_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load frame list
    frame_lines = Path(args.frames).read_text().splitlines()
    frame_ids   = [parse_frame_id(l) for l in frame_lines if l.strip()]
    print(f"Frames    : {len(frame_ids)} from {args.frames}")
    print(f"Annotation: {ann_dir}")
    print(f"Output    : {out_dir}")
    print(f"Split     : train={1-args.val_ratio-args.test_ratio:.0%}  "
          f"val={args.val_ratio:.0%}  test={args.test_ratio:.0%}  "
          f"min_test_per_weather={args.min_test_per_weather}  seed={args.seed}")

    # Analyze frames (or load cache)
    cache_path = out_dir / "frame_analysis.json"
    if args.cache and cache_path.exists():
        print("Loading cached frame analysis...")
        cached = json.loads(cache_path.read_text())
        by_id  = {r["frame_id"]: r for r in cached}
        records = [by_id[fid] for fid in frame_ids if fid in by_id]
        missing = [fid for fid in frame_ids if fid not in by_id]
        if missing:
            print(f"  {len(missing)} frames not in cache — analyzing...")
            records += analyze_frames(missing, ann_dir)
    else:
        records = analyze_frames(frame_ids, ann_dir)

    # Save analysis cache
    cache_path.write_text(json.dumps(records, indent=2))
    print(f"  Cached  → {cache_path}")

    # Weather coverage check
    missing_meta = sum(1 for r in records if not (ZOD_META_ROOT / r["frame_id"] / "metadata.json").exists())
    if missing_meta:
        print(f"  WARNING: {missing_meta} frames missing ZOD metadata — assigned 'day_fair'")

    # Create stratified splits
    train, val, test = make_splits(records, args.val_ratio, args.test_ratio, args.seed,
                                   min_test_per_weather=args.min_test_per_weather)
    print(f"\nSplit sizes:  train={len(train)}  val={len(val)}  test={len(test)}")

    # Write split files
    print("\nWriting splits:")
    write_split(train, out_dir / "train.txt")
    write_split(val,   out_dir / "validation.txt")
    write_split(test,  out_dir / "test.txt")

    for weather in WEATHER_CATEGORIES:
        subset = [r for r in test if r["weather"] == weather]
        if subset:
            write_split(subset, out_dir / f"test_{weather}.txt")

    # all.txt — every frame in the dataset
    write_split(records, out_dir / "all.txt")

    # visualizations.txt — 2 frames per weather condition (one rare-class, one common)
    vis: list[dict] = []
    for weather in WEATHER_CATEGORIES:
        pool = [r for r in records if r["weather"] == weather]
        rare   = [r for r in pool if r["has_rare"]]
        common = [r for r in pool if not r["has_rare"]]
        if rare:
            vis.append(rare[0])
        if common:
            vis.append(common[0])
    write_split(vis, out_dir / "visualizations.txt")

    # Write report
    write_report(records, train, val, test, out_dir / "report.txt")

    # Print quick summary
    print(f"\nWeather × rare-class strata:")
    strata_counts: dict[str, int] = defaultdict(int)
    for r in records:
        strata_counts[_stratum(r)] += 1
    for key in sorted(strata_counts):
        print(f"  {key:35s}  {strata_counts[key]:5d}")


if __name__ == "__main__":
    main()
