"""
Delete frame results that contain degenerate VLM responses, so `--resume` redoes them.

The health monitor aborts a run once the recent window saturates, but it can
only do so *after* the window fills: up to `window * max_rate` garbage responses
land before the trip, and a frame that completes inside that gap is written with
a few substituted verdicts in it. Those frames look finished to `--resume`.

So the rerun is a loop, not a single pass:

    python prune_degenerate_results.py --tag qwen2.5vl_72b_v2 --hpc --delete
    sbatch slurms/vlm-qwen.slurm          # --resume redoes exactly what was pruned
    # ...repeat until the prune reports 0

Each pass strictly shrinks the dirty set, because a frame is only kept when
every response in it carried real content.

Detection matches vlm/health.py exactly — `looks_degenerate` over both signals
that reach disk:
  * masks:      parse_failed[<agent>].degenerate  (the raw text is kept too)
  * discovered: `degenerate`, or no alphanumerics in `vlm_response`

CorrectionAgent failures are excluded by default. Every one of its calls fails
under qwen2.5vl:72b (raw=None, an exception rather than server garbage), so
counting them would condemn ~800 otherwise-clean frames on a fault a rerun
cannot fix. Pass --include-correction to see them anyway.
"""

import argparse
import json
import sys
from pathlib import Path

import config
from vlm.health import looks_degenerate


def degenerate_counts(result: dict, include_correction: bool) -> dict[str, int]:
    """Per-signal count of degenerate responses in one frame result."""
    counts = {"bbox": 0, "other_agents": 0, "correction": 0, "discovery": 0}

    for mask in result.get("masks") or []:
        for agent, info in (mask.get("parse_failed") or {}).items():
            # Older runs stored a bare list of agent names with no raw text; a
            # frame like that cannot be judged, so it is left alone.
            if not isinstance(info, dict) or not info.get("degenerate"):
                continue
            if agent == "correction":
                counts["correction"] += 1
            elif agent == "bbox":
                counts["bbox"] += 1
            else:
                counts["other_agents"] += 1

    for cand in result.get("discovered") or []:
        if cand.get("degenerate") or looks_degenerate(cand.get("vlm_response")):
            counts["discovery"] += 1

    if not include_correction:
        counts.pop("correction")
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True, help="run tag under vlm/<tag>/")
    ap.add_argument("--hpc", action="store_true", help="use HPC data paths")
    ap.add_argument("--delete", action="store_true",
                    help="actually delete; without it this only reports")
    ap.add_argument("--include-correction", action="store_true",
                    help="also condemn frames whose only failure is CorrectionAgent")
    ap.add_argument("--list", type=Path, default=None,
                    help="write the affected frame ids here")
    args = ap.parse_args()

    if args.hpc:
        config.use_hpc()
    config.set_run_tag(args.tag)

    results_dir = config.RESULTS_DIR
    if not results_dir.is_dir():
        sys.exit(f"No results directory: {results_dir}")

    files = sorted(results_dir.glob("frame_*.json"))
    if not files:
        sys.exit(f"No frame results in {results_dir}")

    dirty: list[tuple[str, dict[str, int]]] = []
    totals: dict[str, int] = {}
    unreadable = []
    for path in files:
        try:
            result = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            # A truncated write is not clean data either — redo it.
            unreadable.append((path, str(e)))
            continue
        counts = degenerate_counts(result, args.include_correction)
        if sum(counts.values()):
            dirty.append((path.stem, counts))
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v

    print(f"{results_dir}")
    print(f"  scanned          {len(files)} frames")
    print(f"  degenerate       {len(dirty)} frames  ({len(dirty)/len(files):.1%})")
    if unreadable:
        print(f"  unreadable       {len(unreadable)} frames")
    for k, v in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"    {k:14} {v:,} responses")
    if not args.include_correction:
        print("  (CorrectionAgent failures ignored — see --include-correction)")

    if args.list:
        args.list.write_text("\n".join(sorted(f for f, _ in dirty)) + "\n")
        print(f"  wrote {args.list}")

    if not dirty and not unreadable:
        print("\nClean — nothing to prune.")
        return 0

    if not args.delete:
        print(f"\nDry run. Re-run with --delete to remove these "
              f"{len(dirty) + len(unreadable)} results so --resume redoes them.")
        return 0

    for frame_id, _ in dirty:
        (results_dir / f"{frame_id}.json").unlink()
    for path, _ in unreadable:
        path.unlink()
    print(f"\nDeleted {len(dirty) + len(unreadable)} results. "
          f"Rerun the pipeline with --resume to regenerate them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
