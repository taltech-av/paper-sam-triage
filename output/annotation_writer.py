from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from agents.swin_quality_agent import SWIN_SIZE
from core.mask_extractor import MaskProposal
from core.triage import TRIAGE_REJECT, TriageResult


def write_annotation(
    frame_id: str,
    original_ann: np.ndarray,
    proposals: list[MaskProposal],
    triage_results: list[TriageResult],
    out_dir: Path,
    discovered: list | None = None,
) -> Path:
    """
    Produce a refined annotation PNG.

    Starts from original_ann (annotation_sam), zeros out rejected proposals,
    then paints confirmed discovered masks onto background pixels.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    refined = original_ann.copy()

    for proposal, result in zip(proposals, triage_results):
        if result.decision == TRIAGE_REJECT:
            refined[proposal.pixel_mask] = 0

    if discovered:
        H, W = refined.shape[:2]
        for disc in discovered:
            if not disc.confirmed:
                continue
            mask_orig = cv2.resize(
                disc.pixel_mask_384.astype(np.uint8),
                (W, H), interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            # Only write to background pixels — never overwrite existing masks
            write_pixels = mask_orig & (refined == 0)
            refined[write_pixels] = disc.class_id

    out_path = out_dir / f"{frame_id}.png"
    Image.fromarray(refined.astype(np.uint8)).save(out_path)
    return out_path
