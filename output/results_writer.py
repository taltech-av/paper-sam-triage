import json
from pathlib import Path

from core.mask_extractor import MaskProposal
from core.triage import TriageResult


def write_frame_result(
    frame_id: str,
    proposals: list[MaskProposal],
    triage_results: list[TriageResult],
    out_dir: Path,
    discovered: list | None = None,
) -> dict:
    """Write per-frame JSON and return the dict for summary aggregation."""
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = []
    for proposal, result in zip(proposals, triage_results):
        masks.append({
            "mask_id": proposal.mask_id,
            "class_id": proposal.class_id,
            "class_name": proposal.class_name,
            "bbox": list(proposal.bbox),
            "pixel_count": proposal.pixel_count,
            "agents": {
                "bbox": result.bbox_out,
                "quality": result.quality_out,
                "failure_mode": result.failure_mode_out,
                "correction": result.correction_out,
                "consistency": result.consistency_out,
            },
            "scores": {
                "swin_agreement": round(result.swin_score, 4) if result.swin_score is not None else None,
                "lidar_support": round(result.lidar_support, 4) if result.lidar_support is not None else None,
                "swin_bypass": result.swin_bypass,
            },
            "triage": result.decision,
            "metadata": {k: v for k, v in proposal.metadata.items()
                         if k not in ("pixel_mask",)},
        })

    discoveries = []
    if discovered:
        for disc in discovered:
            discoveries.append({
                "swin_class": disc.swin_class,
                "class_id": disc.class_id,
                "class_name": disc.class_name,
                "bbox_384": list(disc.bbox_384),
                "bbox_orig": list(disc.bbox_orig),
                "pixel_count_384": disc.pixel_count_384,
                "vlm_response": disc.vlm_response,
                "confirmed": disc.confirmed,
            })

    frame_record = {"frame_id": frame_id, "masks": masks, "discovered": discoveries}
    out_path = out_dir / f"{frame_id}.json"
    out_path.write_text(json.dumps(frame_record, indent=2))
    return frame_record


def write_summary(frame_records: list[dict], out_dir: Path) -> None:
    """Aggregate counts across all frames and write summary.json."""
    from collections import defaultdict

    triage_by_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    triage_totals: dict[str, int] = defaultdict(int)
    total_masks = 0
    total_frames = len(frame_records)

    for record in frame_records:
        for mask in record["masks"]:
            cls = mask["class_name"]
            decision = mask["triage"]
            triage_by_class[cls][decision] += 1
            triage_totals[decision] += 1
            total_masks += 1

    summary = {
        "total_frames": total_frames,
        "total_masks": total_masks,
        "triage_totals": dict(triage_totals),
        "triage_by_class": {cls: dict(counts) for cls, counts in triage_by_class.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
