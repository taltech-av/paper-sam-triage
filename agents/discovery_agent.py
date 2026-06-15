"""
Discovery agent: finds objects Swin detects but SAM annotation missed.

Compares Swin's per-pixel class map against annotation_sam. Pixels predicted
as a non-background class but currently annotated as background are candidate
missed objects. Connected-component analysis and size filtering produce crops
that a VLM then confirms and (for "human" class) disambiguates into cyclist
or pedestrian.
"""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

import config
from agents.swin_quality_agent import SWIN_SIZE
from vlm.client import VLMClient

# Swin output class → pipeline class_id (None = needs VLM disambiguation)
SWIN_TO_PIPELINE: dict[int, int | None] = {
    1: 2,    # vehicle
    2: 3,    # sign
    3: None, # human → VLM picks cyclist(4) or pedestrian(5)
}

_CONFIRM_PROMPTS = {
    1: (
        "You are looking at a crop from a front-facing camera on an autonomous vehicle.\n"
        "Does this crop mainly show a vehicle (car, truck, bus, van, or motorcycle)?\n"
        "Answer with exactly one word: vehicle or other."
    ),
    2: (
        "You are looking at a crop from a front-facing camera on an autonomous vehicle.\n"
        "Does this crop mainly show a traffic sign, road sign, or information sign?\n"
        "Answer with exactly one word: sign or other."
    ),
    3: (
        "You are looking at a crop from a front-facing camera on an autonomous vehicle.\n"
        "Does this crop show a person on a bicycle (cyclist) or a person walking (pedestrian), or neither?\n"
        "Answer with exactly one word: cyclist, pedestrian, or other."
    ),
}

_VALID_RESPONSES: dict[int, set[str]] = {
    1: {"vehicle"},
    2: {"sign"},
    3: {"cyclist", "pedestrian"},
}


@dataclass
class DiscoveredMask:
    swin_class: int            # raw Swin class (1=vehicle, 2=sign, 3=human)
    class_id: int              # pipeline class_id assigned
    class_name: str
    bbox_384: tuple            # (x1, y1, x2, y2) in Swin 384-space
    bbox_orig: tuple           # (x1, y1, x2, y2) in original camera image space
    pixel_count_384: int
    pixel_mask_384: np.ndarray # bool [384, 384] — used by annotation writer
    vlm_response: str
    confirmed: bool


class DiscoveryAgent:
    """
    Frame-level agent: returns newly discovered objects as DiscoveredMask list.

    Call once per frame after the per-mask triage loop, passing the same
    swin_pred that was used for quality scoring.
    """

    def __init__(
        self,
        vlm: VLMClient,
        min_pixels: int = config.DISCOVERY_MIN_PIXELS,
        max_candidates: int = config.DISCOVERY_MAX_CANDIDATES,
    ):
        self.vlm = vlm
        self.min_pixels = min_pixels
        self.max_candidates = max_candidates

    def run(
        self,
        swin_pred: np.ndarray,      # [384, 384] uint8 Swin class predictions
        annotation_sam: np.ndarray, # [H, W] uint8 original annotation
        camera_bgr: np.ndarray,     # [H, W, 3] BGR camera image
    ) -> list[DiscoveredMask]:
        H, W = annotation_sam.shape[:2]

        # Resize annotation to Swin space for pixel-level comparison
        ann_384 = cv2.resize(annotation_sam, (SWIN_SIZE, SWIN_SIZE),
                             interpolation=cv2.INTER_NEAREST)

        # Collect all candidates across classes, then cap by size
        candidates = []
        for swin_cls in (1, 2, 3):
            uncovered = (swin_pred == swin_cls) & (ann_384 == 0)
            if not np.any(uncovered):
                continue
            labels, n = _connected_components(uncovered)
            for comp_id in range(1, n + 1):
                comp_mask = labels == comp_id
                pixel_count = int(np.sum(comp_mask))
                if pixel_count >= self.min_pixels:
                    candidates.append((pixel_count, swin_cls, comp_mask))

        # Largest candidates first; cap VLM calls per frame
        candidates.sort(key=lambda x: -x[0])
        candidates = candidates[: self.max_candidates]

        results = []
        for pixel_count, swin_cls, comp_mask in candidates:
            ys, xs = np.where(comp_mask)
            bbox_384 = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

            crop, bbox_orig = _extract_crop(camera_bgr, bbox_384, H, W)
            response = self.vlm.query([crop], _CONFIRM_PROMPTS[swin_cls]).strip().lower().rstrip(".")

            confirmed = response in _VALID_RESPONSES[swin_cls]

            if confirmed and swin_cls == 3:
                class_id = 4 if response == "cyclist" else 5
            elif confirmed:
                class_id = SWIN_TO_PIPELINE[swin_cls]
            else:
                class_id = SWIN_TO_PIPELINE.get(swin_cls) or 4

            results.append(DiscoveredMask(
                swin_class=swin_cls,
                class_id=class_id,
                class_name=config.CLASS_ID_TO_NAME.get(class_id, "other") if confirmed else "other",
                bbox_384=bbox_384,
                bbox_orig=bbox_orig,
                pixel_count_384=pixel_count,
                pixel_mask_384=comp_mask,
                vlm_response=response,
                confirmed=confirmed,
            ))

        return results


# ── helpers ───────────────────────────────────────────────────────────────────

def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, n - 1  # subtract background label 0


def _extract_crop(
    camera_bgr: np.ndarray, bbox_384: tuple, H: int, W: int
) -> tuple[Image.Image, tuple]:
    """Return (PIL crop, bbox_orig) where bbox_orig is (x1,y1,x2,y2) in original image space."""
    x1, y1, x2, y2 = bbox_384
    sx, sy = W / SWIN_SIZE, H / SWIN_SIZE
    ox1 = max(0, int(x1 * sx) - config.CROP_PADDING)
    oy1 = max(0, int(y1 * sy) - config.CROP_PADDING)
    ox2 = min(W, int(x2 * sx) + config.CROP_PADDING)
    oy2 = min(H, int(y2 * sy) + config.CROP_PADDING)
    bbox_orig = (ox1, oy1, ox2, oy2)
    crop = camera_bgr[oy1:oy2, ox1:ox2]
    h, w = crop.shape[:2]
    if h < config.MIN_CROP_SIZE or w < config.MIN_CROP_SIZE:
        scale = max(config.MIN_CROP_SIZE / h, config.MIN_CROP_SIZE / w)
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), bbox_orig
