from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from config import VLM_MAX_RETRIES
from core.bundle import Bundle
from vlm.client import VLMClient
from vlm.health import VLMHealthMonitor, looks_degenerate

# Set to True to print raw VLM responses — useful for diagnosing unexpected rejections
DEBUG_RAW_RESPONSES = False

# Raw text kept on a failed outcome, so a run can be audited afterwards without
# having had full debug logging on.
RAW_SAMPLE_CHARS = 120


@dataclass(frozen=True)
class AgentOutcome:
    """
    What an agent produced, as distinct from what triage will read.

    `value` is always safe to feed to triage. The flags say how much to trust
    it: `parse_failed` means every attempt was unusable and `value` is only
    SAFE_DEFAULT, `degenerate` means the server returned no content at all
    rather than an answer the parser merely disagreed with.

    Keeping these apart matters because SAFE_DEFAULT is a *verdict-shaped*
    value — "valid" for bbox — so a substituted default is indistinguishable
    from a confident one downstream unless it is flagged here. That is exactly
    how the 2026-06 qwen run wrote ~40k defaulted verdicts without noticing.
    """

    value: str
    parse_failed: bool = False
    degenerate: bool = False
    raw_sample: Optional[str] = None


class BaseAgent(ABC):
    VALID_OUTPUTS: tuple[str, ...]
    SAFE_DEFAULT: str

    def __init__(self, client: VLMClient, monitor: Optional[VLMHealthMonitor] = None):
        self.client = client
        self.monitor = monitor

    def run(self, bundle: Bundle) -> str:
        """Verdict only. Prefer run_outcome() where the failure flags matter."""
        return self.run_outcome(bundle).value

    def run_outcome(self, bundle: Bundle) -> AgentOutcome:
        prompt = self.build_prompt(bundle)
        images = self.select_images(bundle)

        last_raw: Optional[str] = None
        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                raw = self.client.query(images, prompt)
                last_raw = raw
                if self.monitor is not None:
                    self.monitor.record(raw)
                parsed = self.parse(raw)

                if DEBUG_RAW_RESPONSES:
                    tag = self.__class__.__name__.replace("Agent", "").lower()
                    status = parsed if parsed else f"PARSE_FAIL → {self.SAFE_DEFAULT}"
                    print(f"    [{tag}] raw={repr(raw[:120])}  →  {status}")

                if parsed is not None:
                    return AgentOutcome(parsed)
            except Exception as e:
                last_raw = None
                if self.monitor is not None:
                    self.monitor.record(None)
                if DEBUG_RAW_RESPONSES:
                    tag = self.__class__.__name__.replace("Agent", "").lower()
                    print(f"    [{tag}] ERROR: {e}")

        # Every attempt failed. The value below is a substitution, not a
        # judgement, and is recorded as such so these masks can be excluded
        # from an analysis instead of silently counting as SAFE_DEFAULT.
        return AgentOutcome(
            self.SAFE_DEFAULT,
            parse_failed=True,
            degenerate=looks_degenerate(last_raw),
            raw_sample=None if last_raw is None else last_raw[:RAW_SAMPLE_CHARS],
        )

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
