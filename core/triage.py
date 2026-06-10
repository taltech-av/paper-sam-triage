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


def triage(
    bbox_out: Optional[str],
    quality_out: Optional[str],
    failure_mode_out: Optional[str],
    correction_out: Optional[str],
    consistency_out: Optional[str],
) -> TriageResult:
    """
    Deterministic triage rules as specified in the paper.

    Reject takes priority over all other rules.
    Accept and Refine are checked next.
    Anything that falls through → human_review.
    """
    result = TriageResult(
        decision=TRIAGE_REVIEW,
        bbox_out=bbox_out,
        quality_out=quality_out,
        failure_mode_out=failure_mode_out,
        correction_out=correction_out,
        consistency_out=consistency_out,
    )

    # Reject conditions
    if bbox_out == "invalid":
        result.decision = TRIAGE_REJECT
        return result
    if quality_out == "bad":
        result.decision = TRIAGE_REJECT
        return result
    if failure_mode_out == "hallucination":
        result.decision = TRIAGE_REJECT
        return result

    # Accept condition
    if bbox_out == "valid" and quality_out == "good" and consistency_out == "pass":
        result.decision = TRIAGE_ACCEPT
        return result

    # Refine condition
    if quality_out == "good" and consistency_out == "fail" and correction_out == "refine":
        result.decision = TRIAGE_REFINE
        return result

    # Everything else → human review
    return result
