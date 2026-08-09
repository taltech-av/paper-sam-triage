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
    def overall_rate(self) -> float:
        """Degenerate share of every call since the process started."""
        return self.degenerate / self.calls if self.calls else 0.0

    @property
    def window_rate(self) -> float:
        """Degenerate share of the recent window — what the trip rule reads."""
        return self.window_degenerate / self.window_size if self.window_size else 0.0

    def as_dict(self) -> dict:
        # `degenerate_rate` is cumulative and has always been; it is kept under
        # that name so runs stay comparable. The window rate — the one the trip
        # rule acts on, and the one that moves first when a server degrades —
        # is reported alongside it rather than conflated with it.
        return {
            "vlm_calls": self.calls,
            "degenerate_responses": self.degenerate,
            "degenerate_rate": round(self.overall_rate, 4),
            "window_degenerate_rate": round(self.window_rate, 4),
            "window_size": self.window_size,
        }


class VLMHealthMonitor:
    """
    Sliding-window degeneracy tracker, shared across frame workers.

    Thread-safe: `process_frames` runs several frames concurrently against one
    client, so every agent records into the same monitor.

    `max_rate` stays well clear of the baseline without waiting for saturation.
    Legitimate answers are never degenerate — a one-word verdict always carries
    alphanumerics — so the only floor is transport errors and agent exceptions,
    which sit at 1-3% of calls in practice. A 200-call window at 35% is an order
    of magnitude above that, while the fault itself runs to 90-100%.

    The check runs on *every* recorded answer, not once per frame. Frames take
    300-800 s and issue 20-60 calls each across 4 workers, so a per-frame check
    let 300+ garbage responses reach disk per episode during the 2026-08-09
    rerun. Checking per call bounds the leak at roughly one window.
    """

    def __init__(self, window: int = 200, max_rate: float = 0.35, min_samples: int = 60):
        self.window = window
        self.max_rate = max_rate
        self.min_samples = min_samples
        self._recent: deque[bool] = deque(maxlen=window)
        self._lock = threading.Lock()
        self._calls = 0
        self._degenerate = 0
        self._tripped = False
        self._trip_stats: Optional[tuple[int, int]] = None

    def _saturation_locked(self) -> Optional[tuple[int, int]]:
        """
        (degenerate, total) for the window once it has gone bad, else None.

        Once tripped this keeps returning the same stats rather than latching
        silent: a caller that swallows the abort must hit it again on its next
        call, otherwise one stray `except Exception` disarms the monitor for the
        rest of the run.
        """
        if self._tripped:
            return self._trip_stats
        if len(self._recent) < self.min_samples:
            return None
        bad, total = sum(self._recent), len(self._recent)
        if bad / total < self.max_rate:
            return None
        self._tripped = True
        self._trip_stats = (bad, total)
        return self._trip_stats

    def _error(self, bad: int, total: int) -> "VLMHealthError":
        return VLMHealthError(
            f"VLM returning degenerate output: {bad}/{total} of the last responses "
            f"carry no usable content ({bad / total:.0%}). The server has most likely "
            f"degraded and needs reloading — see vlm/health.py. Aborting rather "
            f"than writing defaulted verdicts; rerun with --resume once it is back."
        )

    def record(self, raw: Optional[str]) -> bool:
        """
        Record one VLM answer. Returns True if it was degenerate.

        Raises VLMHealthError as soon as the window is saturated, so the run
        stops mid-frame instead of finishing the frame with garbage verdicts.
        The partial frame is simply not written; `--resume` picks it up.
        """
        bad = looks_degenerate(raw)
        with self._lock:
            self._calls += 1
            self._degenerate += bad
            self._recent.append(bad)
            stats = self._saturation_locked()
        if stats is not None:
            raise self._error(*stats)
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
        """
        Abort the run if the recent window is mostly degenerate.

        `record` already raises at the moment of saturation. This remains as an
        end-of-frame backstop for the case where the raise was swallowed.
        """
        with self._lock:
            stats = self._saturation_locked()
        if stats is not None:
            raise self._error(*stats)


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
