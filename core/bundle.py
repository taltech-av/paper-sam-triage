from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from config import (
    CLASS_COLORS_BGR, CROP_PADDING, MIN_CROP_SIZE, OVERLAY_ALPHA,
    ZOD_DATA_ROOT,
)
from core.mask_extractor import MaskProposal


@dataclass
class Bundle:
    rgb_crop: Image.Image        # tight bbox crop of camera image
    overlay_crop: Image.Image    # same crop with mask rendered on top
    depth_crop: Image.Image      # same crop from lidar_png (768px, resized to match)
    metadata: dict               # structured text fields for the VLM prompt


def build_bundle(
    proposal: MaskProposal,
    camera_img: np.ndarray,   # H×W×3 BGR (768px, from cv2.imread)
    lidar_img: np.ndarray,    # H×W×3 BGR (768px)
) -> Bundle:
    # Try to load the original 4K image; fall back to the 768px one
    original_4k = _load_original(proposal.frame_id)
    if original_4k is not None:
        src_img = original_4k
        scale_x = original_4k.shape[1] / camera_img.shape[1]
        scale_y = original_4k.shape[0] / camera_img.shape[0]
    else:
        src_img = camera_img
        scale_x = scale_y = 1.0

    H_src, W_src = src_img.shape[:2]
    H_768, W_768 = camera_img.shape[:2]

    x1, y1, x2, y2 = proposal.bbox
    pad = CROP_PADDING

    # Scale bbox and padding to source image space
    sx1 = max(0, int((x1 - pad) * scale_x))
    sy1 = max(0, int((y1 - pad) * scale_y))
    sx2 = min(W_src, int((x2 + pad) * scale_x))
    sy2 = min(H_src, int((y2 + pad) * scale_y))

    rgb_crop_bgr = src_img[sy1:sy2, sx1:sx2].copy()

    # Scale mask to source image space for overlay
    mask_scaled = cv2.resize(
        proposal.pixel_mask.astype(np.uint8),
        (W_src, H_src),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    overlay_bgr = rgb_crop_bgr.copy()
    color = CLASS_COLORS_BGR.get(proposal.class_id, (255, 255, 255))
    local_mask = mask_scaled[sy1:sy2, sx1:sx2]
    overlay_bgr[local_mask] = (
        overlay_bgr[local_mask] * (1 - OVERLAY_ALPHA)
        + np.array(color) * OVERLAY_ALPHA
    ).astype(np.uint8)

    # Lidar stays at 768px — crop the same region then resize to match RGB crop
    lx1 = max(0, x1 - pad)
    ly1 = max(0, y1 - pad)
    lx2 = min(W_768, x2 + pad)
    ly2 = min(H_768, y2 + pad)
    lidar_crop_bgr = lidar_img[ly1:ly2, lx1:lx2].copy()
    if lidar_crop_bgr.shape[:2] != rgb_crop_bgr.shape[:2]:
        lidar_crop_bgr = cv2.resize(
            lidar_crop_bgr,
            (rgb_crop_bgr.shape[1], rgb_crop_bgr.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    lidar_support_ratio = _compute_lidar_support(proposal.pixel_mask, lidar_img)

    # Upscale all crops if smaller than MIN_CROP_SIZE
    rgb_crop_bgr   = _ensure_min_size(rgb_crop_bgr)
    overlay_bgr    = _ensure_min_size(overlay_bgr)
    lidar_crop_bgr = _ensure_min_size(lidar_crop_bgr)

    source = "4k" if original_4k is not None else "768px"
    w_out, h_out = rgb_crop_bgr.shape[1], rgb_crop_bgr.shape[0]

    metadata = {
        **proposal.metadata,
        "class_id": proposal.class_id,
        "class_name": proposal.class_name,
        "pixel_count": proposal.pixel_count,
        "lidar_support_ratio": round(lidar_support_ratio, 3),
        "bbox": list(proposal.bbox),
        "image_height": H_768,
        "image_width": W_768,
        "crop_size": f"{w_out}x{h_out}",
        "image_source": source,
    }

    return Bundle(
        rgb_crop=_bgr_to_pil(rgb_crop_bgr),
        overlay_crop=_bgr_to_pil(overlay_bgr),
        depth_crop=_bgr_to_pil(lidar_crop_bgr),
        metadata=metadata,
    )


def _load_original(frame_id: str) -> Optional[np.ndarray]:
    """Load full-resolution ZOD camera image if ZOD_DATA_ROOT is configured."""
    if ZOD_DATA_ROOT is None:
        return None
    frame_num = frame_id.replace("frame_", "")  # 'frame_000004' → '000004'
    folder = ZOD_DATA_ROOT / frame_num / "camera_front_blur"
    if not folder.exists():
        return None
    imgs = list(folder.glob("*.jpg"))
    if not imgs:
        return None
    return cv2.imread(str(imgs[0]))


def _ensure_min_size(img_bgr: np.ndarray) -> np.ndarray:
    """Upscale if either dimension is below MIN_CROP_SIZE."""
    h, w = img_bgr.shape[:2]
    if h >= MIN_CROP_SIZE and w >= MIN_CROP_SIZE:
        return img_bgr
    scale = max(MIN_CROP_SIZE / w, MIN_CROP_SIZE / h)
    new_w, new_h = max(int(w * scale), MIN_CROP_SIZE), max(int(h * scale), MIN_CROP_SIZE)
    return cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _compute_lidar_support(pixel_mask: np.ndarray, lidar_img: np.ndarray) -> float:
    if not pixel_mask.any():
        return 0.0
    lidar_gray = lidar_img[:, :, 0]
    lidar_nonzero = lidar_gray > 0
    support = np.logical_and(pixel_mask, lidar_nonzero).sum()
    return float(support) / float(pixel_mask.sum())


def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
