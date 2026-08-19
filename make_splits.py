#!/usr/bin/env python3
"""
Create stratified train / val / test splits balanced by weather condition
and class content.

Reads a frame-list CSV (one 'camera/frame_XXXXXX.png' path per line), looks up
ZOD metadata for weather, counts class pixels from an annotation directory, then
allocates frames so that every split mirrors the overall distribution.

Two allocators, chosen with --balance:

  weather  (default, unchanged)  strata are weather × rare-class-presence and
           frames are dealt out at random inside each stratum. Rare-class
           *presence* is a coarse proxy for content: two frames both "have a
           pedestrian" whether that is 60 px or 60,000 px, so a split can match
           on presence and still hold a third of a class's pixels.

  pixels   weather is held exactly (allocation runs inside each condition) and
           frames are then dealt out to equalise, per class, the *pixel mass*
           each split receives as well as the frame count. Use this whenever
           the split is small enough that one big instance moves a class's
           share — an 800-frame set has ~25 snow frames, and which side of the
           split the one large snow-scene bus lands on is otherwise luck.
           IoU is a pixel-level metric, so the pixels are what has to be even.

Usage:
    # Good frames (manually inspected clean partition)
    python make_splits.py --frames frames/good_frames.csv \\
        --out-dir /run/media/tom/ml/zod_temp/splits_good

    # Flagged frames (VLM-annotated partition)
    python make_splits.py --frames frames/bad_frames.csv \\
        --out-dir /run/media/tom/ml/zod_temp/splits_flagged

    # Small set — balance the class pixel mass, not just presence
    python make_splits.py --frames "$VLM_DATA_ROOT"/vlm/human_verified/frames.csv \\
        --out-dir "$VLM_DATA_ROOT"/vlm/human_verified/splits \\
        --annotation-dir "$VLM_DATA_ROOT"/vlm/human_verified/annotation \\
        --balance pixels --val-ratio 0.15 --test-ratio 0.20

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
import csv
import json
import math
import random
import re
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


def labelling_order(export_path: Path) -> dict[str, int]:
    """frame_id → 1-based rank of when the labeller first answered that frame.

    The human reference is non-stationary: precision falls about 2 points per
    100 frames labelled and has not plateaued, so *when* a frame was judged is a
    property of its labels, not of the scene. Balancing it keeps a split from
    being graded against a systematically stricter standard than it was trained
    on. Rank rather than timestamp, so idle gaps between sessions carry no weight.
    """
    first: dict[str, str] = {}
    with open(export_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("verdict"):
                continue
            match = re.search(r"frame_(\d+)", row["frame"])
            if not match:
                continue
            fid, stamp = match.group(1), row["answeredAt"]
            if fid not in first or stamp < first[fid]:
                first[fid] = stamp
    return {fid: rank + 1 for rank, (fid, _) in
            enumerate(sorted(first.items(), key=lambda kv: kv[1]))}


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


# ── Class-pixel-balanced splitting ────────────────────────────────────────────


def _class_mass(records: list[dict]) -> dict[int, int]:
    return {cls: sum(r["pixel_counts"].get(cls, 0) for r in records) for cls in OBJECT_CLASSES}


def _balance_matrix(records: list[dict]) -> np.ndarray:
    """Per-frame contribution to each balanced quantity, one column per quantity.

    Every column is divided by its own total over `records`, so a column's units
    do not matter and a 720k-pixel pedestrian corpus gets the same say as a 19M
    -pixel vehicle one. Columns are the object classes' pixel mass, plus — when
    the frames carry an `era_rank` (see `labelling_order`) — one more for the
    labelling order. Frame counts per split are fixed before the search runs, so
    equalising each split's share of the summed rank is equalising its *mean*
    labelling order, which is the quantity that matters.
    """
    mass = _class_mass(records)
    columns = [[rec["pixel_counts"].get(cls, 0) / mass[cls] for rec in records]
               for cls in OBJECT_CLASSES if mass[cls] > 0]
    era_total = sum(rec.get("era_rank", 0) for rec in records)
    if era_total and all("era_rank" in rec for rec in records):
        columns.append([rec["era_rank"] / era_total for rec in records])
    return np.array(columns, dtype=float).T.reshape(len(records), len(columns))


def _apportion(total: int, ratios: list[float]) -> list[int]:
    """Largest-remainder apportionment — integer counts summing exactly to total."""
    exact = [total * r for r in ratios]
    counts = [int(x) for x in exact]
    for index in sorted(range(len(ratios)), key=lambda i: -(exact[i] - counts[i]))[:total - sum(counts)]:
        counts[index] += 1
    return counts


# A local-search pass is ~1 ms per condition; the cap only exists so a pathological
# corpus cannot spin. Convergence is normally reached in far fewer moves than this.
MAX_SWAPS = 2000


def _allocate_condition(records: list[dict], ratios: dict[str, float],
                        rng: random.Random) -> dict[str, list[dict]]:
    """Deal one weather condition's frames out, matching class pixel mass.

    Frame counts are fixed up front by largest-remainder apportionment, so the
    only free choice is *which* frames each split gets. That choice is made by
    local search on the imbalance itself:

        J = Σ_split Σ_class (split's share of that class's pixels − its ratio)²

    starting from a proportional deal and repeatedly applying the single
    best-improving swap of two frames between two splits until no swap helps.

    Optimising J directly is the point. Both greedy alternatives tried here
    failed, in opposite directions, for the same reason — a one-pass placement
    rule cannot see what it will be handed later, so the frame *order* decides
    the outcome:

      · class-at-a-time (iterative stratification) placed pedestrian frames by
        pedestrian deficit alone, which committed most of the cyclist mass
        before cyclist was ever scored: 46% of the cyclist pixels in a 65% train
        split. Cyclists ride next to pedestrians; the classes are not separable.
      · joint scoring, biggest-frame-first, sent every heavy frame to whichever
        split had the steepest gradient while all three were still empty, then
        filled the rest with crumbs — 73% of the pixels in that same 65% split.

    A swap changes J by an amount that depends only on the two frames and the
    two splits' current errors, so the best move over all pairs is one
    vectorised expression and the search converges in a few dozen moves.

    Shares, not pixels, are what is squared: each class is divided by its own
    total, so a 720k-pixel pedestrian corpus gets exactly the same say as a 19M
    -pixel vehicle one. Do not additionally divide by the split's ratio to
    "weight small splits more" — that was tried too, and it makes val's gradient
    44x train's, which is how the second failure above got its heavy frames.
    """
    names = list(ratios)
    total = len(records)
    quota = _apportion(total, [ratios[s] for s in names])

    pixels = _balance_matrix(records)

    # Start from a proportional deal of the influence-ordered frames — the
    # heaviest frame of the rarest class first, spread across the splits — so
    # the search begins near a solution rather than at a random one.
    influence = pixels.max(axis=1) if pixels.shape[1] else np.zeros(total)
    order = sorted(range(total), key=lambda i: (-influence[i], records[i]["frame_id"]))
    labels = np.empty(total, dtype=int)
    filled = [0] * len(names)
    for index in order:
        k = min(range(len(names)),
                key=lambda k: ((filled[k] + 0.5) / quota[k] if quota[k] else math.inf, k))
        labels[index] = k
        filled[k] += 1

    target = np.array([ratios[s] for s in names])
    share = np.array([pixels[labels == k].sum(axis=0) for k in range(len(names))])
    for _ in range(MAX_SWAPS):
        error = share - target[:, None]
        best_delta, best_move = -1e-12, None
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                rows, cols = np.flatnonzero(labels == a), np.flatnonzero(labels == b)
                if not len(rows) or not len(cols):
                    continue
                # Swapping i∈a with j∈b moves d = P[j] − P[i] from b into a:
                #   ΔJ = 2 · Σ_c d_c · (error[a,c] − error[b,c] + d_c)
                diff = pixels[cols][None, :, :] - pixels[rows][:, None, :]
                delta = 2 * ((diff * (error[a] - error[b])).sum(-1) + (diff * diff).sum(-1))
                flat = int(np.argmin(delta))
                i, j = divmod(flat, len(cols))
                if delta[i, j] < best_delta:
                    best_delta, best_move = delta[i, j], (a, b, rows[i], cols[j])
        if best_move is None:
            break
        a, b, i, j = best_move
        labels[i], labels[j] = b, a
        share[a] += pixels[j] - pixels[i]
        share[b] += pixels[i] - pixels[j]

    assigned = {s: [records[i] for i in np.flatnonzero(labels == k)] for k, s in enumerate(names)}
    for split in assigned.values():
        rng.shuffle(split)
    return assigned


def _deal(records: list[dict], ratios: dict[str, float],
          rng: random.Random) -> dict[str, list[dict]]:
    """Allocate every frame to one of `ratios`' keys, weather held exactly.

    Weather is exact rather than balanced: each condition is allocated on its
    own, so every part's weather profile is the corpus profile by construction
    and only the class content is left to the local search.
    """
    if min(ratios.values()) <= 0:
        raise ValueError(f"ratios must all be > 0, got {ratios}")

    by_weather: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_weather[rec["weather"]].append(rec)

    out: dict[str, list[dict]] = {name: [] for name in ratios}
    for weather in sorted(by_weather):
        group = sorted(by_weather[weather], key=lambda r: r["frame_id"])
        for name, part in _allocate_condition(group, ratios, rng).items():
            out[name] += part

    for split in out.values():
        rng.shuffle(split)
    return out


def pixel_balanced_splits(
    records: list[dict],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Splits that match on weather, frame count *and* per-class pixel mass."""
    out = _deal(records,
                {"train": 1.0 - val_ratio - test_ratio, "val": val_ratio, "test": test_ratio},
                random.Random(seed))
    return out["train"], out["val"], out["test"]


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


def balance_table(records: list[dict], splits: list[tuple[str, list[dict]]]) -> str:
    """Each split's share of the corpus, per class, in pixels and in frames.

    The column to read is the gap between a class's pixel share and the split's
    frame share: equal means the split holds its proportional slice of that
    class, and a class sitting several points off is one whose downstream IoU is
    measured on a differently-sized sample than it was trained on.
    """
    total_frames = len(records)
    mass = _class_mass(records)
    lines = ["", "  Per-class pixel share vs frame share (a split is even when they match):",
             f"    {'split':8} {'frames':>8} {'frames %':>9}"
             + "".join(f"{CLASS_NAMES[c] + ' %':>13}" for c in OBJECT_CLASSES)]
    for name, split in splits:
        shares = "".join(
            f"{100 * sum(r['pixel_counts'].get(c, 0) for r in split) / mass[c]:12.1f}%"
            if mass[c] else f"{'—':>13}"
            for c in OBJECT_CLASSES)
        lines.append(f"    {name:8} {len(split):8d} {_pct(len(split), total_frames):>9}{shares}")
    lines.append(f"    {'TOTAL':8} {total_frames:8d} {'100.0%':>9}"
                 + "".join(f"{m:12,d}p" for m in (mass[c] for c in OBJECT_CLASSES)))

    # Mean labelling order, when the frames carry one. The reference drifts by
    # roughly 2 points of precision per 100 frames labelled, so two splits whose
    # means are far apart are being held to different standards, and a
    # difference between them is partly the annotator changing their mind.
    if all("era_rank" in r for r in records):
        lines += ["", "  Mean labelling order (drift check — these should be close):"]
        for name, split in splits:
            if split:
                mean = sum(r["era_rank"] for r in split) / len(split)
                lines.append(f"    {name:8} {mean:8.0f}  of {total_frames}")
    return "\n".join(lines)


def write_report(records: list[dict], train: list[dict], val: list[dict], test: list[dict],
                 out_path: Path) -> None:
    sep = "=" * 72
    lines = [sep, "SPLIT DISTRIBUTION REPORT", sep]
    lines.append(balance_table(records, [("train", train), ("val", val), ("test", test)]))
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


def write_split_set(out_dir: Path, records: list[dict], train: list[dict],
                    val: list[dict], test: list[dict]) -> None:
    """One complete split directory: the three lists, per-weather test subsets,
    all.txt, a visualisation sample and the distribution report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split(train, out_dir / "train.txt")
    write_split(val,   out_dir / "validation.txt")
    write_split(test,  out_dir / "test.txt")

    for weather in WEATHER_CATEGORIES:
        subset = [r for r in test if r["weather"] == weather]
        if subset:
            write_split(subset, out_dir / f"test_{weather}.txt")

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

    write_report(records, train, val, test, out_dir / "report.txt")


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
    parser.add_argument("--balance", choices=["weather", "pixels"], default="weather",
                        help="weather: strata are weather × rare-class presence, random inside "
                             "(default, reproduces the existing splits). pixels: weather held "
                             "exactly, frames dealt to equalise per-class pixel mass — use on "
                             "small sets, where presence alone leaves classes lopsided.")
    parser.add_argument("--min-test-per-weather", type=int, default=40,
                        help="Minimum test frames per weather category (default: 40). "
                             "Raises the effective test ratio for small conditions so "
                             "per-weather mIoU evaluation is statistically meaningful.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cache", action="store_true",
                        help="Load frame_analysis.json if it exists (skip re-counting pixels)")
    parser.add_argument("--test-frames",
                        help="Frame-list CSV naming the frames that must form the test split. "
                             "The rest are dealt into train/val by --val-ratio. Use this when "
                             "the test set is decided by something other than chance — e.g. "
                             "the frames a human has verified are the only ones with an "
                             "admissible reference, so they are the test set and nothing else "
                             "can be. --test-ratio is ignored.")
    parser.add_argument("--era-csv", type=Path,
                        help="Verification export CSV. Adds each frame's labelling order to "
                             "the balanced quantities, so no split is graded against a "
                             "systematically stricter reference than another. Only affects "
                             "--balance pixels; frames absent from the export are dropped "
                             "from the era term.")
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
    balance = args.balance
    if args.test_frames:
        print(f"Split     : test fixed by {args.test_frames}  val={args.val_ratio:.0%} of the "
              f"remainder  balance={balance}  seed={args.seed}")
    else:
        print(f"Split     : train={1-args.val_ratio-args.test_ratio:.0%}  "
              f"val={args.val_ratio:.0%}  test={args.test_ratio:.0%}  "
              f"balance={balance}  "
              f"min_test_per_weather={args.min_test_per_weather if balance == 'weather' else 'n/a'}  "
              f"seed={args.seed}")

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

    # Labelling order, when the reference's drift has to be balanced away
    if args.era_csv:
        ranks = labelling_order(args.era_csv)
        hit = 0
        for rec in records:
            if rec["frame_id"] in ranks:
                rec["era_rank"] = ranks[rec["frame_id"]]
                hit += 1
        print(f"  Labelling order from {args.era_csv}: {hit}/{len(records)} frames matched")
        if hit != len(records):
            print("  NOTE: not every frame carries an order — the era term is skipped "
                  "(it is only applied when the whole set has one)")

    # Create stratified splits
    if args.test_frames:
        forced = {parse_frame_id(l) for l in Path(args.test_frames).read_text().splitlines()
                  if l.strip()}
        test = [r for r in records if r["frame_id"] in forced]
        rest = [r for r in records if r["frame_id"] not in forced]
        unseen = forced - {r["frame_id"] for r in records}
        print(f"\nTest fixed at {len(test)} frames from {args.test_frames}"
              + (f"  ({len(unseen)} of them are not in --frames and were ignored)" if unseen else ""))
        if not rest:
            raise SystemExit("--test-frames covers every frame; nothing left to train on")
        parts = _deal(rest, {"train": 1.0 - args.val_ratio, "val": args.val_ratio},
                      random.Random(args.seed))
        train, val = parts["train"], parts["val"]
    elif balance == "pixels":
        train, val, test = pixel_balanced_splits(records, args.val_ratio, args.test_ratio, args.seed)
    else:
        train, val, test = make_splits(records, args.val_ratio, args.test_ratio, args.seed,
                                       min_test_per_weather=args.min_test_per_weather)
    print(f"\nSplit sizes:  train={len(train)}  val={len(val)}  test={len(test)}")

    print("\nWriting splits:")
    write_split_set(out_dir, records, train, val, test)

    # Print quick summary
    print(f"\nWeather × rare-class strata:")
    strata_counts: dict[str, int] = defaultdict(int)
    for r in records:
        strata_counts[_stratum(r)] += 1
    for key in sorted(strata_counts):
        print(f"  {key:35s}  {strata_counts[key]:5d}")


if __name__ == "__main__":
    main()
