"""
Liveness accounting for the VLM server.

The 2026-06 qwen2.5vl:72b run lost most of its frames to a serving fault that
was invisible from inside the pipeline. A few hours after each model load the
ollama instance stopped emitting a stop token and answered every call with ten
garbage characters, until something restarted it — 0% corrupt, then 100% for
14-44 hours at a stretch, four times over six days.

Nothing raised. `BaseAgent` treats an unparseable answer as a retry and then
substitutes `SAFE_DEFAULT`, and `DiscoveryAgent` reads it as "not confirmed",
so a total outage looked exactly like a model being cautious. 72% of frames
were affected and the run finished "successfully".

This module makes that mode loud. Every VLM answer is classified as usable or
degenerate; when the recent degenerate rate crosses a threshold the run aborts
instead of writing thousands more silently-defaulted verdicts.

Degeneracy is deliberately narrower than "unparseable": a model replying
"unclear" to the bbox prompt is a real answer the pipeline knows how to handle,
and must not count toward the health budget. Only answers carrying no
information at all do.
"""

import threading
from collections import deque
from dataclasses import dataclass
from typing import Optional

from PIL import Image

# A degraded server answered every call with `num_predict` copies of one
# unrenderable character. Ten `?` is what reached disk; the general test is
# "no alphanumeric content", which no legitimate one-word verdict can fail.
def looks_degenerate(raw: Optional[str]) -> bool:
    """True when a response carries no usable content at all."""
    if raw is None:
        return True
    stripped = raw.strip()
    if not stripped:
        return True
    return not any(ch.isalnum() for ch in stripped)


class VLMHealthError(RuntimeError):
    """Raised when the VLM server is returning degenerate output."""


# Exit status used when a run aborts on degenerate output. Distinct from 1 so a
# batch script can restart the model and resume on this alone, and still fail
# fast on an ordinary error (missing model, bad path) that a restart cannot fix.
# Kept in sync with slurms/_pipeline_common.sh.
EXIT_HEALTH_ABORT = 42


@dataclass(frozen=True)
class HealthSnapshot:
    calls: int
    degenerate: int
    window_degenerate: int
    window_size: int

    @property
    def window_rate(self) -> float:
        return self.degenerate / self.calls if self.calls else 0.0

    def as_dict(self) -> dict:
        return {
            "vlm_calls": self.calls,
            "degenerate_responses": self.degenerate,
            "degenerate_rate": round(self.window_rate, 4),
        }


class VLMHealthMonitor:
    """
    Sliding-window degeneracy tracker, shared across frame workers.

    Thread-safe: `process_frames` runs several frames concurrently against one
    client, so every agent records into the same monitor.

    `max_rate` is deliberately loose. The failure this guards against saturates
    at 100% within minutes, so a threshold anywhere below that catches it, while
    a loose one cannot be tripped by an unlucky run of genuinely hard crops.
    """

    def __init__(self, window: int = 200, max_rate: float = 0.5, min_samples: int = 60):
        self.window = window
        self.max_rate = max_rate
        self.min_samples = min_samples
        self._recent: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._calls = 0
        self._degenerate = 0
        self._tripped = False

    def record(self, raw: Optional[str]) -> bool:
        """Record one VLM answer. Returns True if it was degenerate."""
        bad = looks_degenerate(raw)
        with self._lock:
            self._calls += 1
            self._degenerate += bad
            self._recent.append(bad)
        return bad

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            return HealthSnapshot(
                calls=self._calls,
                degenerate=self._degenerate,
                window_degenerate=sum(self._recent),
                window_size=len(self._recent),
            )

    def check(self) -> None:
        """Abort the run if the recent window is mostly degenerate."""
        with self._lock:
            if self._tripped or len(self._recent) < self.min_samples:
                return
            bad, total = sum(self._recent), len(self._recent)
            if bad / total < self.max_rate:
                return
            self._tripped = True
            rate = bad / total
        raise VLMHealthError(
            f"VLM returning degenerate output: {bad}/{total} of the last responses "
            f"carry no usable content ({rate:.0%}). The server has most likely "
            f"degraded and needs reloading — see vlm/health.py. Aborting rather "
            f"than writing defaulted verdicts; rerun with --resume once it is back."
        )


# ── canary ────────────────────────────────────────────────────────────────────

_CANARY_PROMPT = (
    "Reply with exactly one word: ok"
)


def probe(client, monitor: Optional[VLMHealthMonitor] = None) -> tuple[bool, str]:
    """
    Ask the server a question whose answer needs no vision, to tell a degraded
    instance from a hard image. Returns (healthy, raw_response).

    Called once at startup. The fault appears hours into a model load, so this
    would never have caught the 2026-06 outage on its own — during the run that
    job belongs to VLMHealthMonitor, which watches real traffic. What the probe
    is for is the other end: refusing to start against a server that is already
    degraded, so a restart-and-resume loop cannot spin through the remaining
    frames writing defaulted verdicts.
    """
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    try:
        raw = client.query([image], _CANARY_PROMPT)
    except Exception as exc:  # noqa: BLE001 — any transport failure is a failed probe
        return False, f"<error: {exc}>"
    if monitor is not None:
        monitor.record(raw)
    return not looks_degenerate(raw), raw
