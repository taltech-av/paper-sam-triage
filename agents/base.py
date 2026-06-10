from abc import ABC, abstractmethod
from typing import Optional

from config import VLM_MAX_RETRIES
from core.bundle import Bundle
from vlm.client import VLMClient

# Set to True to print raw VLM responses — useful for diagnosing unexpected rejections
DEBUG_RAW_RESPONSES = True


class BaseAgent(ABC):
    VALID_OUTPUTS: tuple[str, ...]
    SAFE_DEFAULT: str

    def __init__(self, client: VLMClient):
        self.client = client

    def run(self, bundle: Bundle) -> str:
        prompt = self.build_prompt(bundle)
        images = self.select_images(bundle)

        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                raw = self.client.query(images, prompt)
                parsed = self.parse(raw)

                if DEBUG_RAW_RESPONSES:
                    tag = self.__class__.__name__.replace("Agent", "").lower()
                    status = parsed if parsed else f"PARSE_FAIL → {self.SAFE_DEFAULT}"
                    print(f"    [{tag}] raw={repr(raw[:120])}  →  {status}")

                if parsed is not None:
                    return parsed
            except Exception as e:
                if DEBUG_RAW_RESPONSES:
                    tag = self.__class__.__name__.replace("Agent", "").lower()
                    print(f"    [{tag}] ERROR: {e}")

        return self.SAFE_DEFAULT

    @abstractmethod
    def build_prompt(self, bundle: Bundle) -> str:
        pass

    def select_images(self, bundle: Bundle) -> list:
        return [bundle.rgb_crop, bundle.overlay_crop, bundle.depth_crop]

    def parse(self, raw: str) -> Optional[str]:
        lower = raw.lower().strip()
        for token in self.VALID_OUTPUTS:
            if token in lower:
                return token
        return None


def _meta_text(bundle: Bundle) -> str:
    m = bundle.metadata
    return (
        f"Class: {m['class_name']} (id={m['class_id']})\n"
        f"Mask pixels: {m['pixel_count']}\n"
        f"Bounding box (x1,y1,x2,y2): {m['bbox']}\n"
        f"Aspect ratio (w/h): {m['aspect_ratio']}\n"
        f"LiDAR support ratio: {m['lidar_support_ratio']}\n"
        f"Image size: {m['image_width']}×{m['image_height']}"
    )
