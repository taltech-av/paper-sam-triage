from agents.base import BaseAgent, _meta_text
from core.bundle import Bundle


class CorrectionAgent(BaseAgent):
    VALID_OUTPUTS = ("no_refine", "refine")  # no_refine first — "no_refine" contains "refine"
    SAFE_DEFAULT = "no_refine"

    def build_prompt(self, bundle: Bundle) -> str:
        return (
            "You are deciding whether a segmentation mask can be meaningfully improved.\n"
            "You are given three images:\n"
            "  1. RGB camera crop\n"
            "  2. The same crop with the segmentation mask highlighted\n"
            "  3. LiDAR depth projection of the same region\n\n"
            f"{_meta_text(bundle)}\n\n"
            "The mask quality appears acceptable but LiDAR depth cues suggest a discrepancy.\n"
            "Can the mask be improved by a simple geometric adjustment "
            "(e.g. tightening the boundary, removing a leaked region)?\n"
            "Reply REFINE if a clear correction step exists.\n"
            "Reply NO_REFINE if the mask is already as good as can be determined from the data.\n\n"
            "Reply with exactly one word: REFINE or NO_REFINE"
        )
