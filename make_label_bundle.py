#!/usr/bin/env python3
"""
Build a visin label-bundle zip from stored pipeline outputs — no VLM, no GPU.

One command, one zip, one annotation set per job — upload it once and create a
job per set in label-front.

This ships the **whole corpus**: every frame, every mask, every candidate, each
row carrying the metadata a job might want to slice on. Narrowing to what a
labeling session can actually get through is the platform's job, not this
script's — label-front's wizard samples frames (`sampleN`) and caps masks per
metadata value, so one upload backs any number of differently-scoped jobs and a
re-scope costs a job, not a rebuild and re-upload. The `--frames` /
`--candidates-per-stratum` flags remain for smoke-testing a small zip.

Both jobs are `mask_toggle` (the labeler sees the full frame and clicks every
highlighted region that is wrong):

  --job triage     Are the SAM masks the pipeline triaged actually correct?
                   Both runs score the *same* proposals — `extract_proposals()`
                   is deterministic and each run recorded 90,761 masks over the
                   same frames — so a human verdict on a mask is run-independent
                   and one annotation set (`sam`) serves both. masks.json carries
                   each run's triage decision, class, Swin verdict and LiDAR
                   verdict, so a job can be scoped to any of them — e.g. 200
                   masks per triage outcome for a balanced agent-accuracy sample.

  --job discovery  Are the Swin-proposed discovery candidates real objects?
                   Here the runs *do* differ: each candidate falls in one of four
                   confirmation strata (both VLMs, one only, neither), and the
                   whole point is measuring precision per stratum. Every
                   candidate ships with its `stratum` (both / <tag>_only /
                   neither), and a job capped per stratum draws a simple random
                   sample of each — so a precision estimate per stratum needs no
                   inclusion weights, and the rare strata (Qwen-only is ~2% of
                   candidates, scattered one per frame) are reachable because the
                   cap is global rather than per frame.

The platform wants, per frame: the image, a per-pixel *instance* id map, and a
JSON list describing those instances. The pipeline stores the first and the third
but never wrote an id map: results JSONs carry `mask_id`, and annotation PNGs
carry *class* ids. Both come from `extract_proposals()` over
`annotation_sam/<frame>.png`, which is deterministic (connected components, fixed
class order, MIN_OBJECT_PIXELS filter), so this script re-derives the same
proposals and paints `mask_id + 1` into a grayscale PNG. Every frame is verified
against the stored results before it is written — a frame whose regenerated ids
drift from what the run recorded is skipped, never silently mislabeled.

Layout produced (see label-front's "How to upload a bundle"):

    frames/<frame_id>.jpg                     camera image, 768px space
    annotations/sam/<frame_id>.png            mask overlay (RGBA, transparent bg)
    annotations/sam/<frame_id>.ids.png        grayscale, pixel = mask id + 1
    annotations/sam/<frame_id>.masks.json     [{id, class, bbox, ...}]
    annotations/discovery/<frame_id>.*        the same three, candidate instances
    manifest.csv                              single-set bundles only

The two sets do not cover quite the same frames — a frame Swin proposed no
candidate in gets no `discovery` entry — so `frames/` is their union and each set
is painted only where it applies. Create each job with "All frames / sample":
label-service scopes a mask_toggle job to the frames its own set carries.

Alongside the zip (never inside it) a `.sidecar.json` records every mask's full
provenance — triage per run, confirmation per run, stratum. The platform's export
returns only id/class/verdict, so `analyze_human_verification.py` rejoins the two
on (frame, mask id). Keep the sidecar with the zip; without it an export cannot
be scored.

Frames stay in the 768px `camera/` space on purpose: bboxes, id maps and the
stored mask geometry are all in that space, so the 4K ZOD originals would not
line up.

Usage:
    python make_label_bundle.py                                   # full corpus, both sets
    python make_label_bundle.py --job discovery                   # one set only
    python make_label_bundle.py --limit 400 --frames 4 --candidates-per-stratum 3 \\
        --out /tmp/smoke.zip                                      # small zip to smoke-test with
"""

import argparse
import csv
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

import config
from core.mask_extractor import extract_proposals

# Class → RGBA overlay colour (config's are BGR, for OpenCV).
CLASS_COLORS_RGBA = {
    class_id: (bgr[2], bgr[1], bgr[0], 255) for class_id, bgr in config.CLASS_COLORS_BGR.items()
}
FALLBACK_COLOR = (255, 255, 255, 255)

# Discovery candidates are proposed by raw Swin class, before any VLM names them:
# a human candidate is only resolved into cyclist/pedestrian on confirmation, so
# an unconfirmed one has no finer class to show the labeler than "human".
SWIN_CLASS_NAMES = {1: "vehicle", 2: "sign", 3: "human"}
SWIN_CLASS_COLORS = {1: CLASS_COLORS_RGBA[2], 2: CLASS_COLORS_RGBA[3], 3: CLASS_COLORS_RGBA[4]}

# The id map is a grayscale PNG read as `pixel - 1`, so ids must fit in a byte.
MAX_MASKS_PER_FRAME = 254

TRIAGE_SET = "sam"
DISCOVERY_SET = "discovery"


def results_dir(tag: str) -> Path:
    return config.DATA_ROOT / "vlm" / tag / "results"


def read_record(tag: str, frame_id: str) -> dict:
    return json.loads((results_dir(tag) / f"{frame_id}.json").read_text())


def frame_ids_for(tags: list[str]) -> list[str]:
    """Frames every requested run actually produced results for."""
    common: set[str] | None = None
    for tag in tags:
        ids = {path.stem for path in results_dir(tag).glob("frame_*.json")}
        common = ids if common is None else (common & ids)
    return sorted(common or [])


def sample_per_stratum(items: list, key, per_stratum: int | None, seed: int) -> list:
    """Seeded simple random sample of `per_stratum` items from each stratum."""
    if not per_stratum:
        return items
    buckets: dict[str, list] = defaultdict(list)
    for item in items:
        buckets[key(item)].append(item)
    rng = random.Random(seed)
    sampled: list = []
    for stratum in sorted(buckets):
        bucket = buckets[stratum]
        sampled.extend(bucket if len(bucket) <= per_stratum else rng.sample(bucket, per_stratum))
    return sampled


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def jpeg_bytes(path: Path, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.open(path).convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Triage job
# --------------------------------------------------------------------------


def verify(proposals, record: dict) -> str | None:
    """Regenerated proposals must match what the run stored, or we skip the frame."""
    stored = record["masks"]
    if len(stored) != len(proposals):
        return f"{len(proposals)} regenerated masks vs {len(stored)} stored"
    for proposal, mask in zip(proposals, stored):
        if proposal.mask_id != mask["mask_id"]:
            return f"mask_id {proposal.mask_id} != stored {mask['mask_id']}"
        if list(proposal.bbox) != list(mask["bbox"]):
            return f"bbox drift on mask {proposal.mask_id}"
    return None


def frame_stratum(records: dict[str, dict]) -> str:
    """Rare-class presence — a reporting label, not a sampling control.

    Nothing is stratified at frame level, because no frame-level stratum on this
    data controls anything: at ~21.6 masks per frame every outcome co-occurs in
    nearly every frame (measured over 600 random frames, 94% contain a reject and
    97% contain a mask the two runs triage differently, so severity and
    run-agreement strata are both ~95% one bucket). The masks carry the strata,
    and slicing on them is what the platform's per-value mask cap does.
    Cyclist/pedestrian presence rides along because per-class rejection rates
    diverge most there (Table agent_behavior: cyclist 47.4% vs 8.1%), so it is
    worth having on the task for a per-stratum read of job progress.
    """
    classes = {mask["class_name"] for record in records.values() for mask in record["masks"]}
    return "rare" if classes & {"cyclist", "pedestrian"} else "common"


def triage_mask_rows(records: dict[str, dict]) -> list[dict]:
    """One row per SAM proposal, with every run's decision on it.

    Swin agreement, LiDAR support and the quality verdict are VLM-independent and
    identical across runs, so they are stored once; only the BBox verdict and the
    resulting triage outcome are per run.
    """
    tags = list(records)
    rows = []
    for index, mask in enumerate(records[tags[0]]["masks"]):
        row = {
            "id": mask["mask_id"],
            "class": mask["class_name"],
            "bbox": mask["bbox"],
            "source": "sam",
            "pixel_count": mask["pixel_count"],
            "swin_agreement": mask["scores"].get("swin_agreement"),
            "lidar_support": mask["scores"].get("lidar_support"),
            "quality_agent": mask["agents"].get("quality"),
            "consistency": mask["agents"].get("consistency"),
        }
        for tag in tags:
            other = records[tag]["masks"][index]
            row[f"triage_{tag}"] = other["triage"]
            row[f"bbox_agent_{tag}"] = other["agents"].get("bbox")
        rows.append(row)
    return rows


def build_triage_frames(args, report: dict) -> Iterator[dict]:
    """
    Yield one painted frame at a time.

    A generator rather than a list on purpose: a frame's id map and RGBA overlay
    are ~5 MB together, so holding the whole corpus would want tens of GB. The
    writer consumes each frame, keeps only its metadata for the sidecar, and lets
    the arrays go.
    """
    frame_ids = frame_ids_for(args.tags)
    if args.limit:
        frame_ids = frame_ids[: args.limit]
    if not frame_ids:
        raise SystemExit(f"No frames common to tags {args.tags}")

    report["meta"] = {"framesAvailable": len(frame_ids)}
    rng = random.Random(args.seed)
    if args.frames and args.frames < len(frame_ids):
        frame_ids = sorted(rng.sample(frame_ids, args.frames))
    print(f"{len(frame_ids)} frames")

    skipped = report["skipped"]
    for frame_id in frame_ids:
        records = {tag: read_record(tag, frame_id) for tag in args.tags}
        annotation_path = config.ANNOTATION_SAM_DIR / f"{frame_id}.png"
        camera_path = config.CAMERA_DIR / f"{frame_id}.png"
        if not annotation_path.exists() or not camera_path.exists():
            skipped.append((frame_id, "missing camera or annotation_sam image"))
            continue

        proposals = extract_proposals(annotation_path, frame_id)
        problem = next(filter(None, (verify(proposals, record) for record in records.values())), None)
        if problem:
            skipped.append((frame_id, problem))
            continue
        if len(proposals) > MAX_MASKS_PER_FRAME:
            skipped.append((frame_id, f">{MAX_MASKS_PER_FRAME} masks — id map is 8-bit"))
            continue
        if not proposals:
            skipped.append((frame_id, "no masks"))
            continue

        shape = np.array(Image.open(annotation_path)).shape[:2]
        id_map = np.zeros(shape, dtype=np.uint8)
        overlay = np.zeros((*shape, 4), dtype=np.uint8)
        for proposal in proposals:
            id_map[proposal.pixel_mask] = proposal.mask_id + 1
            overlay[proposal.pixel_mask] = CLASS_COLORS_RGBA.get(proposal.class_id, FALLBACK_COLOR)

        yield {
            "frame_id": frame_id,
            "camera_path": camera_path,
            "stratum": frame_stratum(records),
            "id_map": id_map,
            "overlay": overlay,
            "masks": triage_mask_rows(records),
        }


# --------------------------------------------------------------------------
# Discovery job
# --------------------------------------------------------------------------


def candidate_table(frame_id: str, records: dict[str, dict]) -> list[dict] | None:
    """
    Every Swin-proposed candidate in one frame, with each run's confirmation.

    The candidates' pixels live in vlm/discovery_masks/ (384x384, value =
    candidate index + 1); results JSONs store only bboxes, so the two are matched
    on the stored `bbox_384` that regenerate_discovery_masks.py copied across.
    """
    base = config.DATA_ROOT / "vlm" / "discovery_masks" / frame_id
    png_path, json_path = base.with_suffix(".png"), base.with_suffix(".json")
    if not png_path.exists() or not json_path.exists():
        return None

    by_bbox = {
        tag: {tuple(entry["bbox_384"]): entry for entry in record.get("discovered", [])}
        for tag, record in records.items()
    }

    rows = []
    for index, candidate in enumerate(json.loads(json_path.read_text())):
        bbox_384 = tuple(candidate["bbox_384"])
        entries = {tag: by_bbox[tag].get(bbox_384) for tag in records}
        if any(entry is None for entry in entries.values()):
            continue  # candidate the runs never saw — nothing to score it against
        confirmed = {tag: bool(entry["confirmed"]) for tag, entry in entries.items()}
        swin_class = candidate["swin_class"]
        rows.append({
            "frame_id": frame_id,
            "candidate_index": index,
            "class": SWIN_CLASS_NAMES.get(swin_class, "other"),
            "swin_class": swin_class,
            "bbox": list(next(iter(entries.values()))["bbox_orig"]),
            "bbox_384": list(bbox_384),
            "pixel_count_384": next(iter(entries.values()))["pixel_count_384"],
            "source": "discovery",
            "stratum": confirmation_stratum(confirmed),
            **{f"confirmed_{tag}": confirmed[tag] for tag in records},
            **{f"vlm_response_{tag}": entries[tag].get("vlm_response") for tag in records},
        })
    return rows


def confirmation_stratum(confirmed: dict[str, bool]) -> str:
    """both / <tag>_only / neither — the four groups discovery precision splits by."""
    yes = sorted(tag for tag, value in confirmed.items() if value)
    if not yes:
        return "neither"
    if len(yes) == len(confirmed):
        return "both"
    return f"{yes[0]}_only"


def build_discovery_frames(args, report: dict) -> Iterator[dict]:
    """Yield one painted frame at a time — see `build_triage_frames` on why."""
    frame_ids = frame_ids_for(args.tags)
    if args.limit:
        frame_ids = frame_ids[: args.limit]
    if not frame_ids:
        raise SystemExit(f"No frames common to tags {args.tags}")

    print(f"Scanning {len(frame_ids)} frames for discovery candidates...")
    universe: list[dict] = []
    for frame_id in frame_ids:
        records = {tag: read_record(tag, frame_id) for tag in args.tags}
        rows = candidate_table(frame_id, records)
        if rows:
            universe.extend(rows)
    if not universe:
        raise SystemExit("No discovery candidates found — is vlm/discovery_masks/ populated?")
    stratum_sizes = Counter(row["stratum"] for row in universe)
    report["meta"] = {"population": dict(stratum_sizes)}
    print(f"{len(universe)} candidates: {dict(stratum_sizes)}")

    selected = sample_per_stratum(universe, lambda row: row["stratum"], args.candidates_per_stratum, args.seed)
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in selected:
        by_frame[row["frame_id"]].append(row)
    verb = "Sampled" if args.candidates_per_stratum else "Including all"
    print(f"{verb} {len(selected)} candidates across {len(by_frame)} frames: "
          f"{dict(Counter(row['stratum'] for row in selected))}")

    skipped = report["skipped"]
    for frame_id in sorted(by_frame):
        camera_path = config.CAMERA_DIR / f"{frame_id}.png"
        if not camera_path.exists():
            skipped.append((frame_id, "missing camera image"))
            continue

        shape = np.array(Image.open(camera_path)).shape[:2]
        candidate_map = np.array(Image.open(config.DATA_ROOT / "vlm" / "discovery_masks" / f"{frame_id}.png"))
        id_map = np.zeros(shape, dtype=np.uint8)
        overlay = np.zeros((*shape, 4), dtype=np.uint8)

        masks = []
        for row in sorted(by_frame[frame_id], key=lambda r: r["candidate_index"]):
            small = (candidate_map == row["candidate_index"] + 1).astype(np.uint8)
            pixels = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if not pixels.any():
                skipped.append((f"{frame_id}#{row['candidate_index']}", "candidate has no pixels"))
                continue
            mask_id = len(masks)
            id_map[pixels] = mask_id + 1
            overlay[pixels] = SWIN_CLASS_COLORS.get(row["swin_class"], FALLBACK_COLOR)
            masks.append({"id": mask_id, **{k: v for k, v in row.items() if k != "frame_id"}})

        if not masks:
            continue
        # Rarest stratum present, so the platform's per-stratum job stats surface
        # the enrichment rather than being swamped by the majority group.
        rarest = min(masks, key=lambda m: stratum_sizes[m["stratum"]])["stratum"]
        yield {
            "frame_id": frame_id,
            "camera_path": camera_path,
            "stratum": rarest,
            "id_map": id_map,
            "overlay": overlay,
            "masks": masks,
        }


# --------------------------------------------------------------------------


def write_bundle(groups: dict[str, dict], args) -> None:
    """
    One zip, one annotation set per job.

    Each set is painted only on the frames it applies to, and `frames/` is their
    union — a frame both sets cover is stored once. label-service scopes a
    mask_toggle job to the frames its own set carries, so "All frames / sample"
    on set `sam` and on set `discovery` yields the two jobs off this one upload.

    manifest.csv is written only for a single-set bundle: a manifest is one frame
    list, and two sets over different frames have no single list that fits both.

    Frames arrive as a stream and their pixels are dropped as soon as they are
    written; only each frame's metadata is kept, for the sidecar this returns.
    """
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    kept: dict[str, list[dict]] = {}
    position = 0

    # Stored, not deflated: JPEG/PNG payloads do not compress further.
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for annotation_set, group in groups.items():
            kept[annotation_set] = []
            for frame in group["frames"]:
                frame_id = frame["frame_id"]
                if frame_id not in written:
                    bundle.writestr(f"frames/{frame_id}.jpg", jpeg_bytes(frame["camera_path"], args.jpeg_quality))
                    written.add(frame_id)
                prefix = f"annotations/{annotation_set}/{frame_id}"
                bundle.writestr(f"{prefix}.png", png_bytes(Image.fromarray(frame["overlay"], mode="RGBA")))
                bundle.writestr(f"{prefix}.ids.png", png_bytes(Image.fromarray(frame["id_map"], mode="L")))
                bundle.writestr(f"{prefix}.masks.json", json.dumps(frame["masks"]))
                kept[annotation_set].append(
                    {"frame_id": frame_id, "stratum": frame["stratum"], "masks": frame["masks"]}
                )
                position += 1
                if position % 100 == 0:
                    print(f"  {position} frame-sets written", flush=True)

        if len(groups) == 1:
            manifest = io.StringIO()
            writer = csv.writer(manifest)
            writer.writerow(["filename", "stratum"])
            writer.writerows(
                (f"{frame['frame_id']}.jpg", frame["stratum"]) for frame in next(iter(kept.values()))
            )
            bundle.writestr("manifest.csv", manifest.getvalue())
    return kept


def write_sidecar(annotation_set: str, job: str, frames: list[dict], meta: dict, args) -> Path:
    path = args.out.with_suffix(f".{annotation_set}.sidecar.json")
    path.write_text(json.dumps({
        "job": job,
        "annotationSet": annotation_set,
        "tags": args.tags,
        "seed": args.seed,
        "sampleSize": args.frames if job == "triage" else args.candidates_per_stratum,
        # Population sizes the sample was drawn from: a per-stratum rate measured
        # on the sample scales to the whole run only with these.
        **meta,
        "frames": {
            frame["frame_id"]: {"stratum": frame["stratum"], "masks": frame["masks"]}
            for frame in frames
        },
    }, indent=1))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", choices=["both", "triage", "discovery"], default="both",
                        help="both (default) puts one annotation set per job in a single bundle")
    parser.add_argument("--tags", nargs="+", default=["llava_34b", "qwen2.5vl_72b"],
                        help="VLM run tags whose decisions ride along in masks.json")
    parser.add_argument("--out", type=Path, help="default: <DATA_ROOT>/label_bundles/<job>.zip")
    parser.add_argument("--limit", type=int, help="only the first N frames (smoke test)")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=0,
                        help="triage: sample this many frames instead of all of them (smoke tests)")
    parser.add_argument("--candidates-per-stratum", type=int, default=0,
                        help="discovery: sample this many candidates per stratum instead of all")
    args = parser.parse_args()

    if args.out is None:
        args.out = config.DATA_ROOT / "label_bundles" / f"{args.job}.zip"

    builders = {TRIAGE_SET: ("triage", build_triage_frames), DISCOVERY_SET: ("discovery", build_discovery_frames)}
    wanted = list(builders) if args.job == "both" else [
        TRIAGE_SET if args.job == "triage" else DISCOVERY_SET
    ]

    groups, reports = {}, {}
    for annotation_set in wanted:
        job, build = builders[annotation_set]
        print(f"\n=== {annotation_set} ===")
        reports[annotation_set] = {"skipped": [], "meta": {}}
        groups[annotation_set] = {"job": job, "frames": build(args, reports[annotation_set])}

    kept = write_bundle(groups, args)
    if not any(kept.values()):
        raise SystemExit("Nothing to write — every frame was skipped")

    print()
    for annotation_set, frames in kept.items():
        if not frames:
            continue
        job = groups[annotation_set]["job"]
        sidecar = write_sidecar(annotation_set, job, frames, reports[annotation_set]["meta"], args)
        masks = sum(len(frame["masks"]) for frame in frames)
        print(f"set '{annotation_set}': {len(frames)} frames / {masks} masks "
              f"→ one {job} job.  Provenance: {sidecar.name}")

    skipped = [
        (f"{annotation_set}/{name}", reason)
        for annotation_set, report in reports.items()
        for name, reason in report["skipped"]
    ]
    size_mb = args.out.stat().st_size / 1024**2
    frames_total = len({frame["frame_id"] for frames in kept.values() for frame in frames})
    print(f"\nWrote {frames_total} frames ({size_mb:.0f} MB) to {args.out}")
    if len(kept) > 1:
        print("Upload once, then create one job per annotation set with "
              "'All frames / sample' — the manifest option needs a single-set bundle.")
    if skipped:
        print(f"Skipped {len(skipped)}:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  (+{len(skipped) - 10} more)")


if __name__ == "__main__":
    main()
