from agents.base import BaseAgent, _meta_text
from core.bundle import Bundle


class FailureModeAgent(BaseAgent):
    VALID_OUTPUTS = ("boundary_drift", "hallucination", "occlusion_miss", "fragmentation")
    SAFE_DEFAULT = "boundary_drift"  # diagnostic only — never affects triage

    def build_prompt(self, bundle: Bundle) -> str:
        return (
            "You are diagnosing the dominant failure mode of a segmentation mask in a driving scene.\n"
            "You are given three images:\n"
            "  1. RGB camera crop\n"
            "  2. The same crop with the segmentation mask highlighted\n"
            "  3. LiDAR depth projection of the same region\n\n"
            f"{_meta_text(bundle)}\n\n"
            "Select the single best-matching error type:\n"
            "  BOUNDARY_DRIFT   – mask edges are offset or bleed into adjacent areas\n"
            "  HALLUCINATION    – mask covers a region with no real object of this class\n"
            "  OCCLUSION_MISS   – the mask misses part of the object hidden behind another\n"
            "  FRAGMENTATION    – the object is split into disconnected pieces\n\n"
            "Reply with exactly one word: BOUNDARY_DRIFT, HALLUCINATION, OCCLUSION_MISS, or FRAGMENTATION"
        )
