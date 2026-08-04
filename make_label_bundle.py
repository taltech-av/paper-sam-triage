#!/usr/bin/env python3
"""
Build a visin label-bundle zip from stored pipeline outputs — no VLM, no GPU.

**One upload, one annotation set, one job.** The labeler opens a frame once and
clicks every highlighted region that is wrong — SAM proposals and discovery
candidates side by side, in the same pass. Splitting them into a triage job and a
discovery job made the human open the same 4,000 frames twice and re-read the
same scene twice to answer two questions that a single look already answers.

Nothing is asked of the labeler about which VLM run produced what, because
nothing needs to be: `extract_proposals()` is deterministic and both runs scored
the *same* proposals, so a verdict on a mask is run-independent. Attribution
happens afterwards, in software — the sidecar records, per mask, what each run
decided (triage accept/reject, candidate confirmed or not), so
`analyze_human_verification.py` rejoins a human "this one is wrong" onto whichever
run kept it and whichever run threw it away. The human judges pixels; the runs are
scored against that judgement.

**Both sources are painted as pixels — never as boxes.**

A SAM proposal is judged on its own geometry, so its own pixels are the right
thing to show: filled, in its class colour, opaque.

A discovery candidate is `(swin_pred == class) & (annotation_sam == 0)`, one
connected component of it — the Swin-class pixels SAM did not already cover.
Those pixels are stored, at 384x384, by `regenerate_discovery_masks.py`, and they
are what gets painted here: upsampled to camera space (median 442 px, and no
candidate in a 60-frame probe lost more than its boundary to the finer-grained
full-resolution annotation), semi-transparent, with a white contour so the
labeler can see where the candidate ends and the SAM mask beside it begins.

The bounding box is not used, and that is the point. `bbox_orig` is the padded
box the VLM was shown, and a box drawn around a wheel arch beside a parked van
contains the arch, the van, road and sky — it looks arbitrary because it *is*
arbitrary with respect to object shape, and a human asked about it would be
judging a rectangle rather than a segment.

**But a candidate is often not a whole object, and that is a property of the
data, not of the drawing.** Because SAM's coverage is subtracted first, a
candidate on an object SAM already found is only the fringe left over. Measured
on 1,611 candidates: 69% are *fringe* — the ring of pixels around them is largely
SAM-covered — and 31% are *standalone*, an object region SAM missed entirely, for
which the candidate's pixels genuinely are the whole object as Swin sees it.

That distinction is computed here and shipped as `kind`, and it is what makes the
question well posed for both:

  standalone  fill is the object; "is this a real <class>?" has a plain answer.
  fringe      the fill is one part of a larger object. The overlay draws a
              contour around the *union* of the candidate and the SAM masks it
              touches, so the labeler sees the whole object and judges whether
              the highlighted part belongs to a real one — while the id map keeps
              the click on the candidate's own pixels alone.

The two are worth reporting separately and `analyze_human_verification.py` does:
precision on standalone candidates is what discovery *recovers*, and precision on
fringes is boundary completion on objects the pipeline already had. A single
pooled number over both would read as recall gain and be mostly the latter.

**By default the job carries standalone candidates only** (`--candidates all`
includes fringes). `kind` is pure geometry, so nothing is lost by deciding it
without a human: the fringe share is measurable either way and lands in the
sidecar regardless. What a labeler's time buys is the answer to the question only
a human can settle — is this thing SAM missed entirely a real object — and
spending two thirds of the discovery budget on rims around objects the pipeline
already had would buy an answer to a question the paper does not ask.

masks.json carries a mask's **whole** provenance — triage per run, confirmation
per run, stratum, kind, raw scores — because label-service hands every field back
in the job export as a `mask_<field>` column. So an export is scoreable on its
own and there is nothing to keep beside the zip. The platform's "Group by" picker
only offers low-cardinality scalars, so raw VLM replies and continuous scores
ride along without crowding out the fields a job would actually be scoped on.

Layout produced (see label-front's "How to upload a bundle"):

    frames/<frame_id>.jpg                       camera image, 1363x768 space
    annotations/verify/<frame_id>.png           overlay (RGBA, transparent bg)
    annotations/verify/<frame_id>.ids.png       grayscale, pixel = mask id + 1
    annotations/verify/<frame_id>.masks.json    [{id, class, source, ...}]

Create the job with "All frames" — one `mask_toggle` job over the whole set.

Frames stay in the 1363x768 `camera/` space on purpose: bboxes, id maps and the
stored mask geometry are all in that space, so the 4K ZOD originals would not
line up.

Usage:
    python make_label_bundle.py                       # every frame both runs scored
    python make_label_bundle.py --frames 40 --out /tmp/smoke.zip
"""

import argparse
import io
import json
import zipfile
from collections import Counter
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
# SAM proposals and candidates share that budget: ~35 per frame combined, 82 at
# the worst frame sampled, so the cap only ever fires on something pathological.
MAX_MASKS_PER_FRAME = 254

# A candidate is filled like a SAM mask but semi-transparent and white-edged, so
# a fringe sitting against the mask it belongs to still reads as its own region.
CANDIDATE_ALPHA = 160
CONTOUR_PX = 2

# Radius, in camera-space pixels, of the ring searched around a candidate for the
# SAM masks it belongs to. The candidate map is 384x384 upsampled to 1363x768, so
# one source pixel is ~3.5x2 here: a 9-px ring is a couple of source pixels, wide
# enough to cross the seam left by the upsample and narrow enough not to reach a
# different object.
NEIGHBOUR_RADIUS = 9

# Share of a candidate's ring that has to be SAM-covered before it counts as a
# fringe of an existing mask rather than a standalone find. The measured
# distribution is bimodal around it: the median candidate sits at 0.35, the 10th
# percentile at 0.0.
FRINGE_RING_SHARE = 0.2

ANNOTATION_SET = "verify"





def band(value: float | None, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    """Bucket a continuous score, so it survives the picker's cardinality cap."""
    if value is None:
        return "unknown"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


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


def select_frames(frame_ids: list[str], wanted: int) -> list[str]:
    """
    Thin the corpus to `wanted` frames by even stride, deterministically.

    The default is not to thin at all — every frame both runs scored goes in the
    job. When a smaller job is wanted, a stride rather than a head slice or a
    shuffle: frame ids sort by sequence, so the first N would be one end of the
    collection, while a stride spans the whole of it and repeats exactly on a
    rebuild — the sidecar and the zip must agree, and a re-run after a crash must
    produce the same job.
    """
    if wanted <= 0 or wanted >= len(frame_ids):
        return frame_ids
    step = len(frame_ids) / wanted
    return [frame_ids[int(index * step)] for index in range(wanted)]


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def jpeg_bytes(path: Path, quality: int) -> bytes:
    buffer = io.BytesIO()
    Image.open(path).convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


# --------------------------------------------------------------------------
# SAM proposals (the triage question)
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


def sam_mask_rows(records: dict[str, dict]) -> list[dict]:
    """One row per SAM proposal, with every run's decision on it.

    Swin agreement, LiDAR support and the quality verdict are VLM-independent and
    identical across runs, so they are stored once; only the BBox verdict and the
    resulting triage outcome are per run.

    The two scores are continuous — thousands of distinct values over 90,761
    masks — so they ship as bands as well: the raw float is unusable as a job
    scope and the platform drops it from the picker anyway. `size` bands the
    pixel count at roughly its own quartiles (61/111/248/708 at the 10/25/50/75th
    percentile), the one axis not already covered by some agent's verdict.
    """
    tags = list(records)
    rows = []
    for index, mask in enumerate(records[tags[0]]["masks"]):
        swin = mask["scores"].get("swin_agreement")
        lidar = mask["scores"].get("lidar_support")
        row = {
            "id": mask["mask_id"],
            "class": mask["class_name"],
            "bbox": mask["bbox"],
            "source": "sam",
            "pixel_count": mask["pixel_count"],
            "size": band(mask["pixel_count"], (100, 300, 1000), ("tiny", "small", "medium", "large")),
            "swin_agreement": swin,
            # Bimodal: 30% of masks sit at exactly 0.0 and 22% at exactly 1.0, so
            # both ends get a band of their own rather than being lumped in.
            "swin_agreement_band": band(swin, (0.01, 0.5, 0.99), ("none", "low", "high", "full")),
            "lidar_support": lidar,
            "lidar_support_band": band(lidar, (0.01, 0.4, 0.8), ("none", "low", "medium", "high")),
            "quality_agent": mask["agents"].get("quality"),
            "consistency": mask["agents"].get("consistency"),
        }
        for tag in tags:
            other = records[tag]["masks"][index]
            row[f"triage_{tag}"] = other["triage"]
            row[f"bbox_agent_{tag}"] = other["agents"].get("bbox")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Discovery candidates (the recall question)
# --------------------------------------------------------------------------


def confirmation_stratum(confirmed: dict[str, bool]) -> str:
    """both / <tag>_only / neither — the four groups discovery precision splits by."""
    yes = sorted(tag for tag, value in confirmed.items() if value)
    if not yes:
        return "neither"
    if len(yes) == len(confirmed):
        return "both"
    return f"{yes[0]}_only"


def candidate_rows(frame_id: str, records: dict[str, dict], first_id: int) -> list[dict]:
    """
    Every Swin-proposed candidate in one frame, with each run's confirmation.

    Ids continue past the frame's SAM proposals: one id map holds both sources,
    so one id space has to cover both.

    Results JSONs store only bboxes, so candidates are matched across runs on the
    stored `bbox_384` that regenerate_discovery_masks.py wrote; the component
    pixels painted for the labeler come from the index PNG beside it, and
    `candidate_index` is the row's position in both.

    `bbox_orig` rides along unpainted, as provenance: it is the crop the VLM was
    actually shown, and `fill_ratio` — how little of it the candidate's own
    pixels fill — is how much of that crop was something else. A confirmation on
    a 0.05-fill crop is a weaker claim about the candidate than one on a solid
    blob, and that is worth reading off afterwards rather than showing a human.
    """
    json_path = (config.DATA_ROOT / "vlm" / "discovery_masks" / frame_id).with_suffix(".json")
    if not json_path.exists():
        return []

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
        pixels_384 = next(iter(entries.values()))["pixel_count_384"]
        x1, y1, x2, y2 = bbox_384
        fill = pixels_384 / ((x2 - x1 + 1) * (y2 - y1 + 1))
        rows.append({
            "id": first_id + len(rows),
            "candidate_index": index,
            "class": SWIN_CLASS_NAMES.get(swin_class, "other"),
            "swin_class": swin_class,
            "bbox": list(next(iter(entries.values()))["bbox_orig"]),
            "bbox_384": list(bbox_384),
            "pixel_count_384": pixels_384,
            "size": band(pixels_384, (50, 150, 500), ("tiny", "small", "medium", "large")),
            "fill_ratio": round(fill, 3),
            "fill_ratio_band": band(fill, (0.15, 0.35, 0.6), ("sliver", "thin", "solid", "blob")),
            "source": "discovery",
            "stratum": confirmation_stratum(confirmed),
            **{f"confirmed_{tag}": confirmed[tag] for tag in records},
            **{f"vlm_response_{tag}": entries[tag].get("vlm_response") for tag in records},
        })
    return rows


# --------------------------------------------------------------------------
# Painting
# --------------------------------------------------------------------------


def candidate_pixel_map(frame_id: str, shape: tuple[int, int]) -> np.ndarray | None:
    """
    The stored 384x384 candidate components, upsampled to camera space.

    One nearest-neighbour resize of the whole index map, not one per candidate:
    components are disjoint and their values are candidate index + 1, so a single
    resize keeps them disjoint and keeps every pixel's identity.
    """
    path = config.DATA_ROOT / "vlm" / "discovery_masks" / f"{frame_id}.png"
    if not path.exists():
        return None
    height, width = shape
    return cv2.resize(np.array(Image.open(path)), (width, height), interpolation=cv2.INTER_NEAREST)


def window_around(bbox_384: list[int], shape: tuple[int, int], margin: int) -> tuple[slice, slice]:
    """A camera-space crop containing a candidate's 384-space bbox, plus margin."""
    height, width = shape
    x1, y1, x2, y2 = bbox_384
    return (
        slice(max(0, int(y1 * height / 384) - margin), min(height, int((y2 + 1) * height / 384) + margin)),
        slice(max(0, int(x1 * width / 384) - margin), min(width, int((x2 + 1) * width / 384) + margin)),
    )


def paint_sam(shape: tuple[int, int], proposals, sam_rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """SAM proposals, opaque and in class colour — they own every pixel they cover."""
    id_map = np.zeros(shape, dtype=np.uint8)
    overlay = np.zeros((*shape, 4), dtype=np.uint8)
    for proposal, row in zip(proposals, sam_rows):
        id_map[proposal.pixel_mask] = row["id"] + 1
        overlay[proposal.pixel_mask] = CLASS_COLORS_RGBA.get(proposal.class_id, FALLBACK_COLOR)
    return id_map, overlay


def classify_candidates(id_map: np.ndarray, sam_ids: set[int], candidates: list[dict],
                        pixel_map: np.ndarray) -> list[dict]:
    """
    Attach each candidate's pixels and its `kind`, without painting anything yet.

    A candidate takes its own component's pixels, minus any the full-resolution
    annotation turns out to cover after the upsample — a boundary's worth, never
    the whole component. Components are disjoint, so no two candidates compete
    and the order among them does not matter.

    `kind` comes from what surrounds those pixels: a ring `NEIGHBOUR_RADIUS` wide
    that is more than `FRINGE_RING_SHARE` SAM-covered means the candidate is a
    fringe of those masks rather than something SAM missed. Nothing about this
    needs a human — it is geometry — which is why it can decide what a human is
    asked about rather than being something a human is asked.

    Returns the candidates that have pixels of their own; the rest cannot be
    shown at all and are reported as skipped.
    """
    kernel = np.ones((NEIGHBOUR_RADIUS, NEIGHBOUR_RADIUS), np.uint8)
    sam_id_list = list(sam_ids)
    resolved = []
    for row in candidates:
        rows_slice, cols_slice = window_around(row["bbox_384"], id_map.shape, NEIGHBOUR_RADIUS + 2)
        window_ids = id_map[rows_slice, cols_slice]
        pixels = (pixel_map[rows_slice, cols_slice] == row["candidate_index"] + 1) & (window_ids == 0)
        if not pixels.any():
            continue

        ring = cv2.dilate(pixels.astype(np.uint8), kernel).astype(bool) & ~pixels
        neighbours = sorted(set(np.unique(window_ids[ring]).tolist()) & sam_ids)
        share = float(np.isin(window_ids[ring], sam_id_list).mean()) if ring.any() else 0.0
        resolved.append({
            **row,
            "touching_sam": [neighbour - 1 for neighbour in neighbours],
            "ring_sam_share": round(share, 3),
            "kind": "fringe" if share > FRINGE_RING_SHARE else "standalone",
            "_window": (rows_slice, cols_slice),
            "_pixels": pixels,
            "_neighbours": neighbours,
        })
    return resolved


def paint_candidates(id_map: np.ndarray, overlay: np.ndarray, candidates: list[dict]) -> None:
    """
    Draw the candidates that made it into the job, in place.

    Semi-transparent fill with a white contour, so a candidate is never mistaken
    for the opaque SAM mask it may be lying against. A fringe additionally gets a
    class-coloured contour around the union of itself and the masks it touches,
    so the labeler sees the whole object it is part of — drawn in the overlay
    only, since the id map must keep that object's clicks with the SAM mask that
    owns it.
    """
    for row in candidates:
        window_ids, window_overlay = id_map[row["_window"]], overlay[row["_window"]]
        pixels = row["_pixels"]
        window_ids[pixels] = row["id"] + 1

        color = SWIN_CLASS_COLORS.get(row["swin_class"], FALLBACK_COLOR)
        window_overlay[pixels] = (*color[:3], CANDIDATE_ALPHA)
        outlines = [(pixels, FALLBACK_COLOR)]
        if row["kind"] == "fringe" and row["_neighbours"]:
            outlines.append((pixels | np.isin(window_ids, row["_neighbours"]), color))
        for region, stroke in outlines:
            contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(window_overlay, contours, -1, stroke, CONTOUR_PX)


def build_frames(args, report: dict) -> Iterator[dict]:
    """
    Yield one painted frame at a time.

    A generator rather than a list on purpose: a frame's id map and RGBA overlay
    are ~5 MB together, so holding the whole 4,135-frame corpus would want tens
    of GB. The writer consumes each frame, keeps only its metadata for the
    sidecar, and lets the arrays go.
    """
    available = frame_ids_for(args.tags)
    if not available:
        raise SystemExit(f"No frames common to tags {args.tags}")
    frame_ids = select_frames(available, args.frames)

    report["meta"] = {"framesAvailable": len(available), "framesSelected": len(frame_ids)}
    print(f"{len(frame_ids)} of {len(available)} frames selected")

    skipped = report["skipped"]
    strata: Counter = Counter()
    kinds: Counter = Counter()
    sources: Counter = Counter()
    for position, frame_id in enumerate(frame_ids, 1):
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

        shape = np.array(Image.open(annotation_path)).shape[:2]
        sam_rows = sam_mask_rows(records)
        next_id = max((row["id"] for row in sam_rows), default=-1) + 1

        id_map, overlay = paint_sam(shape, proposals, sam_rows)

        # No stored component pixels means no honest way to paint a candidate —
        # the box is not an option — so the frame ships with its SAM masks alone.
        pixel_map = candidate_pixel_map(frame_id, shape)
        candidates = []
        if pixel_map is None:
            skipped.append((frame_id, "no discovery_masks PNG — SAM masks only"))
        else:
            proposed = candidate_rows(frame_id, records, next_id)
            candidates = classify_candidates(id_map, {row["id"] + 1 for row in sam_rows}, proposed, pixel_map)
            for row in proposed:
                if row["id"] not in {kept_row["id"] for kept_row in candidates}:
                    skipped.append((f"{frame_id}#{row['id']}", "candidate has no pixels SAM left free"))
            # Counted before the job filter: how much of the candidate pool is
            # boundary fringe is a result in itself, and it is measured on the
            # whole pool whether or not a human is asked about all of it.
            kinds.update((row["kind"], row["stratum"]) for row in candidates)
            if args.candidates == "standalone":
                candidates = [row for row in candidates if row["kind"] == "standalone"]

        masks = sam_rows + candidates
        if len(masks) > MAX_MASKS_PER_FRAME:
            skipped.append((frame_id, f"{len(masks)} regions — id map is 8-bit"))
            continue
        if not masks:
            skipped.append((frame_id, "no masks and no candidates"))
            continue

        paint_candidates(id_map, overlay, candidates)
        # The pixel arrays a candidate carried between the two phases stay out of
        # the sidecar: they are working state, and not JSON in any case.
        masks = [{key: value for key, value in row.items() if not key.startswith("_")} for row in masks]
        strata.update(row["stratum"] for row in masks if row["source"] == "discovery")
        sources.update(row["source"] for row in masks)
        if position % 250 == 0:
            print(f"  {position}/{len(frame_ids)} frames painted", flush=True)

        yield {
            "frame_id": frame_id,
            "camera_path": camera_path,
            "id_map": id_map,
            "overlay": overlay,
            "masks": masks,
        }

    # Reported at the end of a build, not shipped: the platform tallies these
    # itself once the zip is imported, and its job manifest is what an analysis
    # extrapolates from. These are here so a build says what it just made.
    report["meta"]["population"] = dict(strata)
    report["meta"]["populationByKind"] = {
        kind: {stratum: count for (other, stratum), count in kinds.items() if other == kind}
        for kind in {kind for kind, _ in kinds}
    }
    report["meta"]["sources"] = dict(sources)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_bundle(frames: Iterator[dict], args) -> list[dict]:
    """
    One zip, one annotation set, one job.

    Frames arrive as a stream and their pixels are dropped as soon as they are
    written; only each frame's mask rows are kept, for the summary this returns.
    """
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept: list[dict] = []

    # Stored, not deflated: JPEG/PNG payloads do not compress further.
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED, allowZip64=True) as bundle:
        for frame in frames:
            frame_id = frame["frame_id"]
            bundle.writestr(f"frames/{frame_id}.jpg", jpeg_bytes(frame["camera_path"], args.jpeg_quality))
            prefix = f"annotations/{ANNOTATION_SET}/{frame_id}"
            bundle.writestr(f"{prefix}.png", png_bytes(Image.fromarray(frame["overlay"], mode="RGBA")))
            bundle.writestr(f"{prefix}.ids.png", png_bytes(Image.fromarray(frame["id_map"], mode="L")))
            bundle.writestr(f"{prefix}.masks.json", json.dumps(frame["masks"]))
            kept.append({"frame_id": frame_id, "masks": frame["masks"]})
    return kept


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tags", nargs="+", default=["llava_34b", "qwen2.5vl_72b"],
                        help="VLM run tags whose decisions ride along in masks.json")
    parser.add_argument("--out", type=Path, help=f"default: <DATA_ROOT>/label_bundles/{ANNOTATION_SET}.zip")
    parser.add_argument("--frames", type=int, default=0,
                        help="frames in the job; the default takes the whole corpus, and a smaller "
                             "number thins it by even stride")
    parser.add_argument("--candidates", choices=["standalone", "all"], default="standalone",
                        help="standalone (default) leaves boundary fringes out of the job — see the "
                             "module docstring; either way both kinds are counted and reported")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    if args.out is None:
        args.out = config.DATA_ROOT / "label_bundles" / f"{ANNOTATION_SET}.zip"

    report: dict = {"skipped": [], "meta": {}}
    kept = write_bundle(build_frames(args, report), args)
    if not kept:
        raise SystemExit("Nothing to write — every frame was skipped")

    counts = Counter(mask["source"] for frame in kept for mask in frame["masks"])
    pool = report["meta"]["populationByKind"]
    totals = {kind: sum(strata.values()) for kind, strata in pool.items()}
    size_mb = args.out.stat().st_size / 1024**2

    print(f"\nWrote {len(kept)} frames ({size_mb:.0f} MB) to {args.out}")
    print(f"  {sum(counts.values())} regions to verify: "
          f"{counts['sam']} SAM masks + {counts['discovery']} discovery candidates "
          f"({sum(counts.values()) / len(kept):.1f} per frame)")
    print(f"  candidate pool on these frames: {totals.get('standalone', 0)} standalone, "
          f"{totals.get('fringe', 0)} fringes of an existing SAM mask"
          + (" — fringes left out of the job" if args.candidates == "standalone" else ""))
    print("\nUpload once, then create a single mask_toggle job on annotation set "
          f"'{ANNOTATION_SET}' with 'All frames'. Every field above rides in masks.json and comes "
          "back out of the job export, so the zip is the only artefact to keep.")

    skipped = report["skipped"]
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for name, reason in skipped[:10]:
            print(f"  {name}: {reason}")
        if len(skipped) > 10:
            print(f"  (+{len(skipped) - 10} more)")


if __name__ == "__main__":
    main()
