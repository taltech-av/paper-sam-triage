"""
Checks for the degradation guard added after the 2026-06 qwen2.5vl:72b run.

The last test is the one that matters: it replays the real corrupted responses
from that run through the new detector and asserts it would have caught them,
with no false positives against the clean llava run over the same candidates.

Run: python3 test_health_fix.py
"""

import glob
import json
import pathlib
import re
import sys

import config
from vlm.health import (
    EXIT_HEALTH_ABORT, VLMHealthError, VLMHealthMonitor, looks_degenerate,
)

failures = []


def check(name, cond, detail=""):
    print(("  ok   " if cond else "  FAIL ") + name + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        failures.append(name)


print("looks_degenerate")
check("ten question marks", looks_degenerate("??????????"))
check("empty", looks_degenerate(""))
check("whitespace", looks_degenerate("   \n "))
check("none", looks_degenerate(None))
check("punctuation only", looks_degenerate("!!!---"))
check("'sign' is fine", not looks_degenerate("sign"))
check("'other' is fine", not looks_degenerate("other"))
check("'unclear' is fine — a real answer", not looks_degenerate("unclear"))
check("verbose answer is fine", not looks_degenerate("This crop shows a sign."))

print("\nVLMHealthMonitor")
m = VLMHealthMonitor(window=100, max_rate=0.5, min_samples=60)
for _ in range(59):
    m.record("??????????")
try:
    m.check()
    check("stays quiet below min_samples", True)
except VLMHealthError:
    check("stays quiet below min_samples", False)

try:
    m.record("??????????")          # the 60th — min_samples is now satisfied
    check("record() raises the moment the window saturates", False)
except VLMHealthError as e:
    check("record() raises the moment the window saturates", True)
    check("message names the cause", "degenerate" in str(e).lower())

# A swallowed abort must not disarm the monitor: every later call raises too.
try:
    m.record("sign")
    check("stays tripped after a swallowed abort", False)
except VLMHealthError:
    check("stays tripped after a swallowed abort", True)
try:
    m.check()
    check("check() still reports a tripped monitor", False)
except VLMHealthError:
    check("check() still reports a tripped monitor", True)

clean = VLMHealthMonitor(window=100, max_rate=0.5, min_samples=60)
for i in range(200):
    clean.record("sign" if i % 10 else "unclear")
try:
    clean.check()
    check("healthy traffic never trips", True)
except VLMHealthError:
    check("healthy traffic never trips", False)

snap = clean.snapshot()
check("snapshot counts calls", snap.calls == 200, f"got {snap.calls}")
check("snapshot counts zero degenerate", snap.degenerate == 0, f"got {snap.degenerate}")

print("\nBaseAgent falls back but flags it")
from agents.base import AgentOutcome, BaseAgent


class _StubAgent(BaseAgent):
    # Non-overlapping on purpose: BaseAgent.parse matches by substring, so a
    # vocabulary like ("valid", "invalid") would resolve "invalid" to "valid".
    # The live agents avoid this — BBoxAgent overrides parse() with word-boundary
    # regex, CorrectionAgent orders "no_refine" first — but a stub must not
    # smuggle that hazard into a test of something else.
    VALID_OUTPUTS = ("alpha", "beta")
    SAFE_DEFAULT = "alpha"

    def build_prompt(self, bundle):
        return "prompt"

    def select_images(self, bundle):
        return []


class _Client:
    def __init__(self, reply):
        self.reply = reply

    def query(self, images, prompt):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


mon = VLMHealthMonitor()
out = _StubAgent(_Client("??????????"), mon).run_outcome(None)
check("degenerate reply -> SAFE_DEFAULT value", out.value == "alpha")
check("degenerate reply -> parse_failed", out.parse_failed)
check("degenerate reply -> degenerate", out.degenerate)
check("degenerate reply -> raw kept", out.raw_sample == "??????????")
check("monitor saw the calls", mon.snapshot().degenerate > 0)

out = _StubAgent(_Client("beta"), VLMHealthMonitor()).run_outcome(None)
check("good reply -> value", out.value == "beta")
check("good reply -> not flagged", not out.parse_failed and not out.degenerate)

out = _StubAgent(_Client("banana"), VLMHealthMonitor()).run_outcome(None)
check("off-menu reply -> parse_failed", out.parse_failed)
check("off-menu reply -> NOT degenerate", not out.degenerate)

out = _StubAgent(_Client(RuntimeError("boom")), VLMHealthMonitor()).run_outcome(None)
check("transport error -> parse_failed", out.parse_failed)
check("transport error -> degenerate", out.degenerate)

check("run() still returns a bare string", _StubAgent(_Client("alpha"), None).run(None) == "alpha")

print("\nReplay: the real 2026-06 run")
runs = {}
for tag in ("qwen2.5vl_72b", "llava_34b"):
    # Via config so the replay follows the project's data root instead of
    # carrying a second copy of it that silently goes stale when the disk moves.
    files = sorted(glob.glob(str(config.DATA_ROOT / "vlm" / tag / "results" / "*.json")))
    if not files:
        print(f"  skip {tag} — results not on this machine")
        continue
    caught = missed = false_pos = clean_n = 0
    for path in files:
        for entry in json.loads(open(path).read()).get("discovered", []):
            raw = entry.get("vlm_response")
            known_bad = raw is not None and set(str(raw)) == {"?"}
            flagged = looks_degenerate(raw)
            if known_bad:
                caught += flagged
                missed += not flagged
            else:
                clean_n += 1
                false_pos += flagged
    runs[tag] = (caught, missed, clean_n, false_pos)
    print(f"  {tag}: corrupted={caught + missed} detected={caught} missed={missed} "
          f"| clean={clean_n} false_positives={false_pos}")

if "qwen2.5vl_72b" in runs:
    caught, missed, clean_n, false_pos = runs["qwen2.5vl_72b"]
    check("detects every corrupted qwen response", missed == 0, f"{missed} missed")
    check("no false positives on qwen's clean responses", false_pos == 0, f"{false_pos}")
    check("the incident is actually large", caught > 40000, f"{caught}")
if "llava_34b" in runs:
    _, _, clean_n, false_pos = runs["llava_34b"]
    check("no false positives across the whole llava run", false_pos == 0, f"{false_pos}")

print()
print("Abort status is shared with the batch script")

# slurms/_pipeline_common.sh retries this status and only this one. If the two
# constants drift apart the loop stops restarting on a degraded server and the
# fault this whole module exists to catch goes back to running unattended.
_common_sh = pathlib.Path(__file__).parent / "slurms" / "_pipeline_common.sh"
_declared = re.search(r"^EXIT_HEALTH_ABORT=(\d+)", _common_sh.read_text(), re.M)
check("_pipeline_common.sh declares the status", _declared is not None)
if _declared:
    check("shell and python agree on the status",
          int(_declared.group(1)) == EXIT_HEALTH_ABORT,
          f"shell={_declared.group(1)} python={EXIT_HEALTH_ABORT}")
check("status is distinct from a generic failure", EXIT_HEALTH_ABORT != 1)

print()
if failures:
    print(f"FAILED ({len(failures)}): " + ", ".join(failures))
    sys.exit(1)
print("all checks passed")
