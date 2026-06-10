from agents.base import BaseAgent, _meta_text
from core.bundle import Bundle


class BBoxAgent(BaseAgent):
    VALID_OUTPUTS = ("invalid", "valid")  # invalid first — "invalid" contains "valid"
    SAFE_DEFAULT = "valid"  # benefit of the doubt — ZOD GT already confirmed object presence

    def select_images(self, bundle: Bundle) -> list:
        # Full frame context first so the model sees the driving scene,
        # then the tight crop so it can inspect the specific region
        return [bundle.context_image, bundle.rgb_crop]

    def build_prompt(self, bundle: Bundle) -> str:
        return (
            "You are checking a bounding-box proposal from an autonomous driving scene.\n\n"
            "Image 1: full camera frame with the bounding box marked.\n"
            "Image 2: zoomed crop of that bounding box region.\n\n"
            f"{_meta_text(bundle)}\n\n"
            "Reply VALID if any real object of the stated class is visible inside the "
            "marked box — even if small, distant, or partially occluded.\n\n"
            "Reply INVALID only if the marked box clearly contains nothing but road "
            "surface, sky, or flat background with absolutely no object of the stated class.\n\n"
            "Reply with exactly one word: VALID or INVALID"
        )
