from dataclasses import dataclass
from typing import Optional


TRIAGE_ACCEPT = "accept"
TRIAGE_REFINE = "refine"
TRIAGE_REJECT = "reject"
TRIAGE_REVIEW = "human_review"


@dataclass
class TriageResult:
    decision: str
    bbox_out: Optional[str]
    quality_out: Optional[str]
    failure_mode_out: Optional[str]
    correction_out: Optional[str]
    consistency_out: Optional[str]
    # Raw scores for offline triage replay (thresholds applied post-hoc)
    swin_score: Optional[float] = None
    lidar_support: Optional[float] = None
    swin_bypass: bool = False
    # Timing: seconds spent in each agent call (only populated agents present)
    agent_elapsed: Optional[dict] = None
    # Number of actual VLM HTTP calls made for this mask
    vlm_calls: int = 0


def triage(
    bbox_out: Optional[str],
    quality_out: Optional[str],
    failure_mode_out: Optional[str],
    correction_out: Optional[str],
    consistency_out: Optional[str],
) -> TriageResult:
    """
    Deterministic concordance-based triage.

    Rejection is destructive (pixels are zeroed), so it requires at least two
    concordant negative signals among {bbox invalid, quality bad, consistency
    fail}. A single negative signal from any one judge routes to human review
    instead — with several noisy binary judges, rejecting on any single
    negative compounds their false-positive rates and destroys valid masks.

    Exception: bbox_out == "background" means the VLM positively identified a
    non-object surface (vegetation, building, sky, snow) under the mask. That
    is direct evidence of error, not mere absence of confirmation, and counts
    as two negatives — i.e. it rejects on its own.

    The failure-mode agent is diagnostic only and never affects the decision.
    """
    result = TriageResult(
        decision=TRIAGE_REVIEW,
        bbox_out=bbox_out,
        quality_out=quality_out,
        failure_mode_out=failure_mode_out,
        correction_out=correction_out,
        consistency_out=consistency_out,
    )

    # bbox=invalid + consistency=pass counts as two negatives only when the Swin
    # quality signal also disagrees (quality_out != "good"). If Swin says the
    # mask pixels DO match the expected class, the VLM crop verdict is likely
    # a false negative (common for small/thin objects like signs where the
    # cropped region is ambiguous) — route to human_review instead of rejecting.
    bbox_invalid_confirmed = (
        bbox_out == "invalid"
        and consistency_out == "pass"
        and quality_out != "good"
    )

    negatives = sum([
        2 * bbox_invalid_confirmed,
        bbox_out == "invalid" and not bbox_invalid_confirmed,
        2 * (bbox_out == "background"),
        quality_out == "bad",
        consistency_out == "fail",
    ])

    # Reject: two or more concordant negative signals
    if negatives >= 2:
        result.decision = TRIAGE_REJECT
        return result

    # Accept: all three signals positive
    if bbox_out == "valid" and quality_out == "good" and consistency_out == "pass":
        result.decision = TRIAGE_ACCEPT
        return result

    # Refine: visually good but geometrically unsupported, and a concrete
    # correction step exists
    if bbox_out == "valid" and quality_out == "good" and consistency_out == "fail":
        if correction_out == "refine":
            result.decision = TRIAGE_REFINE
        return result

    # Single negative signal or missing outputs → human review (non-destructive)
    return result
