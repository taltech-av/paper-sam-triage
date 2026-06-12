from typing import Optional

from config import LIDAR_SUPPORT_MIN
from core.bundle import Bundle
from vlm.client import VLMClient


class ConsistencyAgent:
    """
    Deterministic LiDAR-support check — no VLM call.

    A 7B VLM reading sparse depth projections is unreliable, while the
    lidar_support_ratio already computed in the bundle separates real masks
    (median ~0.63 on ZOD) from sky/void leaks cleanly. The check is therefore
    a fixed threshold, which also makes it free and exactly reproducible.
    """

    VALID_OUTPUTS = ("pass", "fail")

    def __init__(self, client: Optional[VLMClient] = None):
        # client accepted for interface compatibility with the other agents
        self.client = client

    def run(self, bundle: Bundle) -> str:
        ratio = bundle.metadata.get("lidar_support_ratio", 0.0)
        return "pass" if ratio >= LIDAR_SUPPORT_MIN else "fail"
