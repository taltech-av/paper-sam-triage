from pathlib import Path

import numpy as np
from PIL import Image

from core.mask_extractor import MaskProposal
from core.triage import TRIAGE_REJECT, TriageResult


def write_annotation(
    frame_id: str,
    original_ann: np.ndarray,
    proposals: list[MaskProposal],
    triage_results: list[TriageResult],
    out_dir: Path,
) -> Path:
    """
    Produce a refined annotation PNG.

    Starts from original_ann (annotation_sam), then zeros out all pixels
    belonging to rejected proposals. Accept / refine / review are kept as-is.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    refined = original_ann.copy()

    for proposal, result in zip(proposals, triage_results):
        if result.decision == TRIAGE_REJECT:
            refined[proposal.pixel_mask] = 0  # set to background

    out_path = out_dir / f"{frame_id}.png"
    Image.fromarray(refined.astype(np.uint8)).save(out_path)
    return out_path
