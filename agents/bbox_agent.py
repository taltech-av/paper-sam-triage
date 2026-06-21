import re
from typing import Optional

from agents.base import BaseAgent
from core.bundle import Bundle

# Forced-choice vocabulary: the four annotation classes plus the surfaces SAM
# typically leaks onto. Asking "what is this?" makes the model commit to an
# answer; a yes/no presence question invites agreement and accepts noise blobs.
CLASSES = ("vehicle", "sign", "cyclist", "pedestrian", "traffic_light")
# Concrete non-object surfaces: answering one of these is positive evidence
# that the mask covers background, strong enough to reject on its own.
STRONG_DISTRACTORS = ("vegetation", "building", "sky", "snow")
# Weak distractors: absence of confirmation, needs corroboration. ROAD is the
# dominant background of every crop, so it is the most error-prone answer for
# tiny objects and must never reject alone.
WEAK_DISTRACTORS = ("road", "other", "unclear")


class BBoxAgent(BaseAgent):
    VALID_OUTPUTS = ("valid", "invalid", "background")
    SAFE_DEFAULT = "valid"  # benefit of the doubt — concordance rule guards rejection

    def select_images(self, bundle: Bundle) -> list:
        # Crop only. The full-frame context image makes a 7B VLM classify the
        # scene instead of the marked region ("highway -> vehicle").
        return [bundle.rgb_crop]

    def build_prompt(self, bundle: Bundle) -> str:
        # Blind classification: the prompt must not mention the proposed class —
        # telling the model what to expect makes it parrot that answer back.
        self._expected_class = bundle.metadata["class_name"]
        options = ", ".join(
            t.upper() for t in CLASSES + STRONG_DISTRACTORS + WEAK_DISTRACTORS
        )
        return (
            "This is a zoomed crop from an autonomous driving camera.\n"
            "The object of interest may be small, distant, or partially occluded — "
            "it does not need to fill the frame.\n"
            f"What class does the masked object belong to? Choose one: {options}\n"
            "Note: traffic lights, traffic signals, illuminated road signs, variable "
            "message boards, and any pole-mounted road lighting equipment count as TRAFFIC_LIGHT.\n"
            "Note: trees, bushes, shrubs, hedges, or roadside vegetation count as VEGETATION.\n"
            "Note: CYCLIST includes any person on a bicycle, electric scooter, motorcycle, "
            "moped, or cargo bike. PEDESTRIAN includes any person walking, running, standing, "
            "or pushing a pram, stroller, baby wagon, or wheelchair.\n"
            "If the masked region covers only background with no identifiable object, "
            "reply UNCLEAR.\n"
            "Reply with exactly one word from the list."
        )

    def parse(self, raw: str) -> Optional[str]:
        lower = raw.lower()
        tokens = CLASSES + STRONG_DISTRACTORS + WEAK_DISTRACTORS
        found = {t for t in tokens if re.search(rf"\b{t}s?\b", lower)}
        if not found or "unclear" in found:
            return None  # unparseable or ambiguous → retry, then SAFE_DEFAULT
        if self._expected_class in found:
            return "valid"
        # traffic_light is a valid instance of the "sign" class in ZOD
        if "traffic_light" in found and self._expected_class == "sign":
            return "valid"
        if found & set(STRONG_DISTRACTORS):
            return "background"
        return "invalid"
