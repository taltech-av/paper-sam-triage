"""
Deterministic mock client for testing without a running VLM.
Returns the first valid token from each agent's vocabulary based on a hash
of the image content, so the same input always gives the same output.
"""
import hashlib

from PIL import Image

from vlm.client import VLMClient


class MockClient(VLMClient):
    def __init__(self, seed: int = 42):
        self.seed = seed

    def query(self, images: list[Image.Image], prompt: str) -> str:
        # Hash first image bytes + prompt to get a deterministic "random" choice
        buf = images[0].tobytes()[:256] if images else b""
        digest = hashlib.md5(buf + prompt.encode()).hexdigest()
        idx = int(digest[:4], 16) % 4  # 0-3

        # Pick response from valid vocab based on which agent is being called
        if "one word from the list" in prompt:  # bbox forced-choice classification
            return ["vehicle", "vegetation", "sign", "pedestrian"][idx]
        if "GOOD or BAD" in prompt:
            return ["good", "bad", "good", "good"][idx]
        if "BOUNDARY_DRIFT" in prompt:
            return ["boundary_drift", "hallucination", "occlusion_miss", "fragmentation"][idx]
        if "REFINE or NO_REFINE" in prompt:
            return ["refine", "no_refine", "refine", "no_refine"][idx]
        if "PASS or FAIL" in prompt:
            return ["pass", "fail", "pass", "pass"][idx]
        return "unknown"
