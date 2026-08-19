"""
Package the Zenodo artifact bundle: everything needed to reproduce the paper's
tables without a GPU, without ZOD, and without calling a model.

    python release/make_bundle.py --out ~/zenodo_bundle

Writes one tar.gz per component plus MANIFEST.txt (sha256 + row/file counts), so
a reproducer can download only the piece they need and verify it.

Two things this script exists to get right:

  * The human export ships with the annotator's name and email in every row.
    Those columns are dropped and replaced by a stable pseudonym before anything
    leaves the machine. This is not optional and there is no flag to skip it.
  * Checksums are computed over the *anonymised* file, so the manifest describes
    what was actually published rather than what was on disk.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Columns naming a real person. Replaced with a stable pseudonym rather than
# blanked: `userEmail` is also the field that marks a row as answered and
# identifies which labeller answered it (analyze_human_verification.py keys the
# per-labeller verdict map on it), so emptying it turns all 145,428 rows into
# unanswered ones and every downstream number silently disappears. Only
# non-empty values are rewritten, so a genuinely unanswered row stays empty.
PII_COLUMNS = {
    "userEmail": "annotator_1@example.invalid",
    "userName": "annotator_1",
}

PUBLISHED_TAGS = ("llava_34b", "qwen2.5vl_72b_v2")

# Replayed annotation variants the offline analysis reads. An allowlist rather
# than a glob: DATA_ROOT also holds annotation_sam and the other-modality sets,
# which run to gigabytes and are inputs to the pipeline rather than its output.
# discovery_pixel_budget.py needs the first two — without them it silently
# reports zero candidates instead of failing.
ANNOTATION_VARIANTS = (
    "annotation_swin_only",
    "annotation_swin_only_discovery_noVLM_ccm",
    "annotation_swin_only_discovery_noVLM_standalone_ccm",
    "annotation_raw_sam",
)


README = """\
VLM Annotation Triage Pipeline - artifact bundle
(Tallinn University of Technology)
DOI: 10.5281/zenodo.22010998

SAM pseudo-label triage, object discovery, and a human verification pass over
4,110 Zenseact Open Dataset frames. These files let the whole analysis be
re-derived offline: every model response is stored, so no GPU or model call is
needed to reproduce the reported rates.

Code, schemas, and the command behind every reported number:
https://github.com/taltech-av/paper-vlm-annotation-pipeline
  README.md    start here
  REPRODUCE.md reported number -> exact command
  DATA.md      field-by-field schemas

FILES  (sha256 for all of them in MANIFEST.txt)

  human_verification.tar.gz    the human reference: 145,428 regions, 35,984 judged
  responses_llava_34b.tar.gz   every stored LLaVA-1.6-34B response and score
  responses_qwen2.5vl_72b_v2   the same for Qwen2.5-VL-72B
  discovery_masks.tar.gz       discovery candidate masks (GPU needed to rebuild)
  annotations.tar.gz           the four replayed annotation variants (PNG)
  splits_and_frames.tar.gz     frame lists + splits; use frames/vlm_frames.csv
  ---- the two below are only for re-running the pipeline or training ----
  annotation_sam.tar.gz        the SAM pseudo-labels the paper audits
  best.pth, config_9.json      the CLFTv2/Swin quality model (not archives)

USE

  Offline, no GPU, no ZOD, 59 MB: unpack the six files above the line into one
  directory and set VLM_DATA_ROOT to it. That reproduces every rate measured
  against the human reference. Re-running the pipeline adds 2.8 GB and needs the
  Zenseact Open Dataset (https://zod.zenseact.com/), which is NOT redistributed
  here - this bundle holds decisions about ZOD frames, not ZOD imagery.

Artifacts CC-BY-4.0. Code MIT. ZOD imagery under Zenseact's own terms.
"""


def write_readme(out_dir: Path) -> Path:
    """Orientation file for the published record: what each file is, and the
    minimum needed per tier. MANIFEST.txt carries the checksums."""
    dest = out_dir / "README.txt"
    dest.write_text(README)
    return dest


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def anonymise_export(src: Path) -> tuple[bytes, int]:
    """Return the export with PII columns rewritten, and the row count."""
    with src.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = PII_COLUMNS.keys() - set(fields)
        if missing:
            print(f"  note: no {sorted(missing)} column to anonymise")
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        rows = 0
        for row in reader:
            for column, replacement in PII_COLUMNS.items():
                if column in row and row[column]:
                    row[column] = replacement
            writer.writerow(row)
            rows += 1
    return buf.getvalue().encode("utf-8"), rows


def add_tree(tar: tarfile.TarFile, src: Path, arcname: str) -> int:
    """Add a directory, returning the file count. Sorted for a stable archive."""
    count = 0
    for path in sorted(src.rglob("*")):
        if path.is_file():
            tar.add(path, arcname=f"{arcname}/{path.relative_to(src)}")
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="output directory")
    ap.add_argument("--export", type=Path,
                    default=Path("human_verified_output/verify_export.csv"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []

    # 1. the human reference ---------------------------------------------------
    print("human verification export")
    if not args.export.is_file():
        print(f"  MISSING: {args.export}", file=sys.stderr)
        return 1
    data, rows = anonymise_export(args.export)
    dest = args.out / "human_verification.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        info = tarfile.TarInfo("human_verification/verify_export.csv")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    print(f"  {rows:,} rows, PII columns replaced -> {dest.name}")
    manifest.append(f"human_verification.tar.gz  {sha256(dest)}  {rows} rows")

    # 2. stored model responses ------------------------------------------------
    for tag in PUBLISHED_TAGS:
        results = config.DATA_ROOT / "vlm" / tag / "results"
        print(f"stored responses: {tag}")
        if not results.is_dir():
            print(f"  MISSING: {results} — skipped", file=sys.stderr)
            continue
        dest = args.out / f"responses_{tag}.tar.gz"
        with tarfile.open(dest, "w:gz") as tar:
            n = add_tree(tar, results, f"vlm/{tag}/results")
        print(f"  {n:,} files -> {dest.name}")
        manifest.append(f"responses_{tag}.tar.gz  {sha256(dest)}  {n} files")

    # 3. discovery component masks ---------------------------------------------
    # Not regenerable offline: regenerate_discovery_masks.py re-runs the Swin
    # model to recover them, so a reproducer without a GPU cannot rebuild these.
    # discovery_pixel_budget.py needs them for the pixel columns of the candidate
    # geometry table, which is why they are a required component and not optional.
    print("discovery component masks")
    masks = config.DATA_ROOT / "vlm" / "discovery_masks"
    if masks.is_dir():
        dest = args.out / "discovery_masks.tar.gz"
        with tarfile.open(dest, "w:gz") as tar:
            n = add_tree(tar, masks, "vlm/discovery_masks")
        print(f"  {n:,} files -> {dest.name}")
        manifest.append(f"discovery_masks.tar.gz  {sha256(dest)}  {n} files")
    else:
        print(f"  MISSING: {masks} — skipped", file=sys.stderr)

    # 4. splits and frame lists ------------------------------------------------
    print("splits and frame lists")
    dest = args.out / "splits_and_frames.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        n = add_tree(tar, Path("frames"), "frames")
        # splits_vlm_4110_humantest is the one every downstream experiment uses
        # (test = the 1,001 human-verified frames). splits_good is the clean
        # partition the quality signal was trained on, published so the claim
        # that it never saw the partition it scores can be checked rather than
        # taken on trust.
        for name in ("splits_vlm_4110_humantest", "splits_good", "splits_vlm_4110"):
            splits = config.DATA_ROOT / name
            if splits.is_dir():
                n += add_tree(tar, splits, f"splits/{name}")
            else:
                print(f"  MISSING: {splits} — skipped", file=sys.stderr)
    print(f"  {n:,} files -> {dest.name}")
    manifest.append(f"splits_and_frames.tar.gz  {sha256(dest)}  {n} files")

    # 5. the SAM pseudo-labels being audited ------------------------------------
    # The input the whole paper is about. Tier 1 does not need it (the replayed
    # variants are shipped pre-built), but replay_triage.py reads it to write any
    # variant, so Tier 2/3 cannot start without it.
    print("SAM pseudo-labels (annotation_sam)")
    sam = config.ANNOTATION_SAM_DIR
    if sam.is_dir():
        dest = args.out / "annotation_sam.tar.gz"
        with tarfile.open(dest, "w:gz") as tar:
            n = add_tree(tar, sam, "annotation_sam")
        print(f"  {n:,} files -> {dest.name}")
        manifest.append(f"annotation_sam.tar.gz  {sha256(dest)}  {n} files")
    else:
        print(f"  MISSING: {sam} — skipped", file=sys.stderr)

    # 6. the quality-signal checkpoint ------------------------------------------
    # Shipped uncompressed and untarred: it is 2.5 GB of float weights that gzip
    # barely touches, and a reproducer who only wants the model should not have
    # to unpack an archive to get it. Without this file the pipeline cannot run
    # at all and the paper's one positive filtering result cannot be recomputed.
    print("Swin quality-model checkpoint")
    for src, label in ((config.SWIN_CKPT_PATH, "best.pth"),
                       (config.SWIN_CFG_PATH, "config_9.json")):
        if src.is_file():
            dest = args.out / label
            dest.write_bytes(src.read_bytes())
            size = dest.stat().st_size
            print(f"  {size / 1e6:,.0f} MB -> {label}")
            manifest.append(f"{label}  {sha256(dest)}  {size} bytes")
        else:
            print(f"  MISSING: {src} — skipped", file=sys.stderr)

    # 7. replayed annotation variants ------------------------------------------
    print("annotation variants")
    dest = args.out / "annotations.tar.gz"
    with tarfile.open(dest, "w:gz") as tar:
        n = 0
        for name in ANNOTATION_VARIANTS:
            variant = config.DATA_ROOT / name
            if variant.is_dir():
                n += add_tree(tar, variant, name)
            else:
                print(f"  MISSING: {variant} — skipped", file=sys.stderr)
    print(f"  {n:,} files -> {dest.name}")
    manifest.append(f"annotations.tar.gz  {sha256(dest)}  {n} files")

    manifest_path = args.out / "MANIFEST.txt"
    manifest_path.write_text(
        "VLM Annotation Triage Pipeline - artifact bundle.\n"
        "DOI: 10.5281/zenodo.22010998\n\n"
        "Code: https://github.com/taltech-av/paper-vlm-annotation-pipeline\n"
        "Schemas and unpacking instructions: DATA.md in that repository.\n\n"
        "sha256:\n" + "\n".join(f"  {line}" for line in manifest) + "\n")
    readme_path = write_readme(args.out)
    print(f"\nwrote {manifest_path} and {readme_path.name}")
    total = sum(f.stat().st_size for f in args.out.iterdir()
                if f.is_file() and f.name != "MANIFEST.txt")
    print(f"bundle total: {total / 1e6:.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
