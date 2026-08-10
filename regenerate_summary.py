"""
Rebuild a run tag's summary.json from the per-frame results on disk.

`process_frames.py` deliberately skips write_summary when a run ends in a health
abort, because the in-memory frame_records are short of what reached disk and
summarising them would replace a good summary with a partial one. The cost is
that a tag whose *last* attempt was short ends up with a summary describing only
that attempt — `qwen2.5vl_72b_v2` sat at `total_frames: 1` after the 2026-08-10
rerun loop, against 4,110 frames of real results.

Reading the frames back off disk sidesteps that: the results directory is the
authoritative record, so the summary always matches it.

    python regenerate_summary.py --tag qwen2.5vl_72b_v2

Aggregation goes through output.results_writer.write_summary — the same function
the pipeline uses — so a regenerated summary is byte-identical to one the
pipeline would have written for the same frames.
"""

import argparse
import json
import sys
from pathlib import Path

import config
from output.results_writer import write_summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run tag under vlm/<tag>/")
    ap.add_argument("--hpc", action="store_true", help="use HPC data paths")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the summary instead of writing it")
    args = ap.parse_args()

    if args.hpc:
        config.use_hpc()
    config.set_run_tag(args.tag)

    results_dir = config.RESULTS_DIR
    if not results_dir.is_dir():
        sys.exit(f"No results directory: {results_dir}")

    records = []
    skipped = []
    for path in sorted(results_dir.glob("frame_*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            skipped.append((path.name, str(e)))
            continue
        # A record with no masks key is not a frame result — refuse to count it
        # rather than let a stray file quietly shrink the totals.
        if "masks" not in record:
            skipped.append((path.name, "no 'masks' key"))
            continue
        records.append(record)

    if not records:
        sys.exit(f"No usable frame results in {results_dir}")

    old = results_dir / "summary.json"
    previous = None
    if old.exists():
        try:
            previous = json.loads(old.read_text()).get("total_frames")
        except (OSError, json.JSONDecodeError):
            previous = None

    if args.dry_run:
        # Same aggregation, written somewhere harmless so the real file is not
        # touched by a dry run.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            write_summary(records, Path(tmp))
            print((Path(tmp) / "summary.json").read_text())
    else:
        write_summary(records, results_dir)

    print(f"{results_dir}")
    print(f"  frames aggregated  {len(records):,}"
          + (f"   (summary previously claimed {previous})" if previous is not None else ""))
    print(f"  masks              {sum(len(r['masks']) for r in records):,}")
    if skipped:
        print(f"  skipped            {len(skipped)} file(s):")
        for name, why in skipped[:10]:
            print(f"    {name}: {why}")
    print("  dry run — nothing written" if args.dry_run else f"  wrote {old}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
