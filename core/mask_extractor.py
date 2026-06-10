from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from config import CLASS_ID_TO_NAME, MIN_OBJECT_PIXELS


@dataclass
class MaskProposal:
    frame_id: str
    mask_id: int
    class_id: int
    class_name: str
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords
    pixel_mask: np.ndarray  # full-frame boolean mask, shape (H, W)
    pixel_count: int
    metadata: dict = field(default_factory=dict)


def extract_proposals(annotation_path: Path, frame_id: str) -> list[MaskProposal]:
    """
    Extract per-object mask proposals from a SAM annotation PNG.

    Each connected component in each non-background class becomes one proposal.
    Components smaller than MIN_OBJECT_PIXELS[class_id] are dropped.
    """
    ann = np.array(Image.open(annotation_path))
    proposals = []
    mask_id = 0

    for class_id in [2, 3, 4, 5]:
        class_mask = (ann == class_id).astype(np.uint8)
        if not class_mask.any():
            continue

        min_px = MIN_OBJECT_PIXELS.get(class_id, 15)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(class_mask, connectivity=8)

        for label_idx in range(1, n_labels):  # 0 = background component
            pixel_count = int(stats[label_idx, cv2.CC_STAT_AREA])
            if pixel_count < min_px:
                continue

            x1 = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y1 = int(stats[label_idx, cv2.CC_STAT_TOP])
            w  = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h  = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            x2, y2 = x1 + w, y1 + h

            component_mask = labels == label_idx
            aspect_ratio = round(w / h, 3) if h > 0 else 0.0

            proposals.append(MaskProposal(
                frame_id=frame_id,
                mask_id=mask_id,
                class_id=class_id,
                class_name=CLASS_ID_TO_NAME[class_id],
                bbox=(x1, y1, x2, y2),
                pixel_mask=component_mask,
                pixel_count=pixel_count,
                metadata={
                    "aspect_ratio": aspect_ratio,
                    "bbox_width": w,
                    "bbox_height": h,
                },
            ))
            mask_id += 1

    return proposals
