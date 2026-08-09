"""
Discovery agent: recovers objects Swin detects that SAM annotation missed.

Swin proposes candidates: pixels predicted as non-background but annotated as
background in annotation_sam. Connected-component analysis and size filtering
produce crops. A VLM then confirms each crop (yes/no) and disambiguates the
"human" class into cyclist or pedestrian. Swin drives detection; VLM is the
confirmation gate only.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import config
from agents.swin_quality_agent import SWIN_SIZE
from vlm.client import VLMClient
from vlm.health import VLMHealthError, VLMHealthMonitor, looks_degenerate

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
        "The object of interest may be small or distant and does not need to fill the frame.\n"
        "Does this crop contain a traffic sign, road sign, information sign, variable message "
        "board, or pole-mounted road lighting equipment (traffic lights, street lights)?\n"
        "Answer with exactly one word: sign or other."
    ),
    3: (
        "You are looking at a crop from a front-facing camera on an autonomous vehicle.\n"
        "Cyclist means any person on a bicycle, electric scooter, motorcycle, moped, or cargo bike.\n"
        "Pedestrian means any person walking, running, standing, or pushing a pram, stroller, "
        "baby wagon, or wheelchair.\n"
        "Does this crop show a cyclist, a pedestrian, or neither?\n"
        "Answer with exactly one word: cyclist, pedestrian, or other."
    ),
}

_VALID_RESPONSES: dict[int, set[str]] = {
    1: {"vehicle"},
    2: {"sign"},
    3: {"cyclist", "pedestrian"},
}

# Everything the prompt actually offers, confirmations and rejections alike.
# An answer outside this set is one the model never mapped onto the question —
# distinct from a degenerate response (no content at all) and from a clean
# "other" (a real negative). All three used to collapse into confirmed=False.
_ANSWERABLE: dict[int, set[str]] = {
    cls: valid | {"other"} for cls, valid in _VALID_RESPONSES.items()
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
    # `confirmed` is False both when the model said "other" and when it said
    # nothing usable, and those are opposite facts: one is evidence the
    # candidate is not an object, the other is no evidence at all. Without this
    # flag a degraded server reads as a model rejecting every candidate — which
    # is how the 2026-06 qwen run reported 17% confirmation against llava's 69%.
    parse_failed: bool = False
    degenerate: bool = False


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
        monitor: Optional[VLMHealthMonitor] = None,
    ):
        self.vlm = vlm
        self.min_pixels = min_pixels
        self.max_candidates = max_candidates
        self.monitor = monitor

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
            response = None
            for attempt in range(config.VLM_MAX_RETRIES + 1):
                try:
                    response = self.vlm.query([crop], _CONFIRM_PROMPTS[swin_cls]).strip().lower().rstrip(".")
                    if self.monitor is not None:
                        self.monitor.record(response)
                    break
                except VLMHealthError:
                    raise
                except Exception:
                    if self.monitor is not None:
                        self.monitor.record(None)
                    if attempt == config.VLM_MAX_RETRIES:
                        response = None

            # A candidate whose call failed is kept, not dropped. Dropping it
            # silently shrinks the denominator of any recall claim; keeping it
            # flagged lets the analysis exclude it explicitly.
            degenerate = looks_degenerate(response)
            parse_failed = degenerate or response not in _ANSWERABLE[swin_cls]
            confirmed = (not degenerate) and response in _VALID_RESPONSES[swin_cls]

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
                parse_failed=parse_failed,
                degenerate=degenerate,
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
