from agents.base import BaseAgent, _meta_text
from core.bundle import Bundle


class ConsistencyAgent(BaseAgent):
    VALID_OUTPUTS = ("pass", "fail")
    SAFE_DEFAULT = "fail"  # conservative: routes to refine/review rather than accept

    def build_prompt(self, bundle: Bundle) -> str:
        lidar_ratio = bundle.metadata.get("lidar_support_ratio", 0.0)
        return (
            "You are checking whether a segmentation mask is geometrically consistent "
            "with LiDAR depth data in a driving scene.\n"
            "You are given three images:\n"
            "  1. RGB camera crop\n"
            "  2. The same crop with the segmentation mask highlighted\n"
            "  3. LiDAR depth projection of the same region\n\n"
            f"{_meta_text(bundle)}\n"
            f"Pre-computed LiDAR support ratio: {lidar_ratio:.2f} "
            "(fraction of mask pixels with LiDAR returns)\n\n"
            "Does the mask region align with solid LiDAR returns in the depth image?\n"
            "PASS if LiDAR points are present and roughly match the mask shape.\n"
            "FAIL if LiDAR returns are absent, mis-aligned, or contradict the mask location.\n\n"
            "Reply with exactly one word: PASS or FAIL"
        )
