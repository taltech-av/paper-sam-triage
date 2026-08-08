# Qwen Run Corruption — Analysis & Rerun Plan (2026-08-04)

## TL;DR

The June 2026 `qwen2.5vl_72b` annotation run was corrupted by the Ollama serving
fault documented in `vlm/health.py`: **2,972 of 4,135 frames (72%) received
degenerate VLM responses** that were silently substituted with safe defaults.
The `llava_34b` run is fully clean. Critically, the paper's central cross-model
claims (Qwen's "near-unconditional acceptance" and "conservative discovery")
are **artifacts of the corruption, not real model behavior** — on clean frames
Qwen behaves much like LLaVA.

Plan: keep the same model and prompts (the design is sound; only serving
failed), commit + deploy the already-written health-monitor fix, harden the
slurm script with a restart-and-resume loop, rerun qwen fully into a fresh tag
(~2 × 48h A100 jobs), then regenerate only the tainted downstream artifacts.

---

## 1. What happened

From `vlm/health.py` (written after the incident): a few hours after each model
load, the Ollama instance stopped emitting a stop token and answered every call
with ten garbage characters (`"??????????"`), for 14–44 hours at a stretch,
four times over six days. Nothing raised:

- `BaseAgent` treated the unparseable answer as a retry, then substituted
  `SAFE_DEFAULT` — for the BBox agent that is **`"valid"`**.
- `DiscoveryAgent` read it as **"not confirmed"**.

So a total outage looked exactly like a cautious model, and the run finished
"successfully".

### Corruption oracle

The degenerate responses are stored verbatim in each result JSON under
`discovered[].vlm_response`. A frame whose discovery responses are *all*
non-alphanumeric was processed during an outage window. This classifies every
frame:

| Run | Frames | Fully corrupt | Partial | Clean | No discovery calls |
|---|---|---|---|---|---|
| `qwen2.5vl_72b` | 4,135 | **2,970** | 2 | 1,125 | 38 (all on clean day) |
| `llava_34b` | 4,135 | **0** | 0 | 4,097 | 38 |
| `qwen2.5vl_72b_old_prompt` | 1,000 | **319** | 0 | 681 | 0 |

### Qwen corruption timeline (per day)

| Day (2026-06) | Frames | Degenerate |
|---|---|---|
| 23 | 615 | 0% |
| 24 | 911 | 61% |
| 25 | 552 | 94% |
| 26 | 705 | 100% |
| 27 | 690 | 80% |
| 28 | 632 | 100% |
| 29 | 30 | 100% |

Both full runs predate the health fix: their `run_info` has no
`vlm_health` / `n_parse_failed` fields — the pipeline was blind while they ran.

---

## 2. Impact on the paper — the headline finding does not survive

Splitting qwen's stored verdicts by the corruption oracle:

| Metric | Paper claims | Corrupt frames | **Clean frames only** | LLaVA (clean) |
|---|---|---|---|---|
| BBox `valid` rate | 93.3% | 99.9% (= SAFE_DEFAULT) | **73.4%** | 62.7% |
| Discovery confirm rate | 17.8% | 0.0% | **83.3%** | 82.5% |
| Old-prompt `valid` rate | 64.4% | 99.9% | **41.7%** | — |

The arithmetic closes exactly: mixing 73.4% clean with 99.9% defaulted over the
mask counts reproduces the published 93.3%; 83.3% × the clean fraction of
candidates reproduces the published 17.8%. Conclusions:

- **"Qwen's near-unconditional acceptance"** is mostly the safe-default
  substitution, not Qwen. The real cross-model gap is ~73% vs ~63%, not
  93% vs 62%.
- **"Qwen's conservative discovery confirmation" does not exist.** On clean
  frames Qwen confirms at essentially LLaVA's rate (83.3% vs 82.5%). The
  downstream "protective scepticism" result (Swin+disc 45.1 vs 40.5 mIoU) and
  the both-confirmed consensus story (8,703 objects ≈ clean-qwen ∩ llava) sit
  on top of the artifact and should be **expected not to reproduce**. After the
  rerun, Qwen's confirmed set will likely grow ~4× (toward ~46k objects) and
  its discovery-variant mIoU will likely drop toward LLaVA's.
- **The prompt-sensitivity finding survives and gets stronger**: on clean
  frames the within-model effect is 41.7% → 73.4% valid — larger than the
  published (corruption-diluted) 64.4% → 85.2%.

### Tainted artifacts checklist

| Artifact | Status |
|---|---|
| `qwen2.5vl_72b` results + all its annotation variants | corrupt → rerun |
| `merged_llava_34b_qwen2.5vl_72b` consensus | derived from corrupt → re-merge |
| `qwen2.5vl_72b_old_prompt` (1,000 frames) | **deleted 2026-08-04**; 681 of its frames were clean and are the only surviving basis for the prompt-sensitivity section — see Step 4 |
| `llava_34b` results + variants | **clean — keep** |
| `raw_sam`, `swin_only` variants (VLM-independent) | **clean — keep** |
| Fusion checkpoints trained on qwen/merged variants | retrain |
| Fusion checkpoints for llava / swin_only / raw_sam variants | **keep** |
| Paper tables: `agent_behavior`, `ablation` (qwen + consensus rows), `prompt_sensitivity`, `timing` | regenerate |
| Paper text: abstract, cross-model narrative (ll. 53, 306–307, 325–328, 484), consensus-discovery section | rewrite after new numbers |

---

## 3. Decision: keep model, keep prompts

**Do not switch models or revise prompts for the rerun.**

- The failure was in the serving layer, not the model. Clean-frame qwen
  behavior is reasonable and interesting.
- The paper's design — identical fixed prompts across two open VLMs — is
  exactly what makes the corrected comparison publishable. Switching either
  model or prompt now would invalidate the clean llava run and restart the
  entire experimental matrix.
- **`BBoxAgent.parse()` maps `UNCLEAR` to retry → `SAFE_DEFAULT "valid"`, and
  this must be left alone.** The prompt invites UNCLEAR ("if the masked region
  covers only background... reply UNCLEAR") and the parser then discards it,
  which reads backwards. It is tempting to fix. Do not, because:
  - Raw BBox responses were never stored (only the parsed verdict), so the
    llava run cannot be re-parsed offline — changing the rule would force a
    full **llava rerun too** (~33 GPU-hours), and the paper's core design
    claim is identical prompts *and* parsing across both models.
  - The `UNCLEAR → None` branch was introduced in commit `6770195`, i.e. it is
    part of the "stricter response-parsing protocol" that the prompt-
    sensitivity experiment measures. Changing it now silently redefines the
    thing being studied.

  The health fix already makes this measurable without changing semantics: an
  UNCLEAR-driven fallback is now flagged `parse_failed=True, degenerate=False`,
  so the v2 run can *report* how often it fires. Document that rate as a
  limitation instead of changing the rule.

---

## 4. Execution plan

### Step 1 — Health fix — DONE (committed 2026-08-04)

Detection side, in `vlm/health.py` + wiring: every response classified
usable/degenerate; sliding window (200 calls, trip at ≥50% over ≥60 samples)
aborts the run instead of writing defaulted verdicts; per-frame
`n_parse_failed` + `vlm_health` in `run_info`; startup canary probe; abort is
resumable via `--resume`.

Validated by `test_health_fix.py`, which replays the real incident: **43,784
corrupted responses detected, 0 missed, 0 false positives** across both the
qwen and llava runs.

Still to do: **pull on HPC** — the deployed checkout is what actually matters,
and it is still running the blind version until it is updated.

### Step 2 — Slurm restart loop — DONE (committed 2026-08-04)

Detection is only half a fix; the run still has to recover. `process_frames.py`
now exits `EXIT_HEALTH_ABORT` (42) — distinct from 1 — and
`slurms/_pipeline_common.sh` (shared by both slurm scripts) retries *only* that
status: reload Ollama, resume from the frames on disk. Any other non-zero exit
(missing model, bad path) fails the job immediately, because a reload cannot
fix it.

Two guards stop the loop spinning on a server that never recovers:
`MAX_RESTARTS` (default 20), and a consecutive-no-progress counter that gives
up after two aborts in a row that completed zero frames. One zero-progress
abort is expected and recoverable — a server already degraded at job start
fails the canary before writing anything, which is exactly what a reload fixes
— so the loop always attempts at least one reload before concluding failure.

The monitor detects saturation within ~60 VLM calls, so an episode now costs
minutes rather than the 14–44 hours it used to.

Optional, not implemented: a proactive Ollama restart every ~4h (the fault
appears hours after load), and upgrading the Ollama build on HPC — this looks
like a known runner/stop-token bug class with large models. Worth doing if the
fault recurs during the rerun.

### Step 3 — Rerun qwen (the long pole — start first)

Full rerun of all 4,135 frames into a fresh tag. `slurms/vlm-qwen.slurm` now
sets `TAG="qwen2.5vl_72b_v2"` itself, so this needs no extra flags:

```bash
jid=$(sbatch --parsable slurms/vlm-qwen.slurm)
sbatch --dependency=afterany:$jid slurms/vlm-qwen.slurm   # chain job 2
```

`afterany` (not `afterok`) is deliberate: job 1 is expected to hit the 48h
walltime and be killed, which is not an `ok` exit. Each job resumes from the
results already on disk.

- Cost estimate (from clean-frame pace, mean 298 s/frame): ~343 frame-hours
  ≈ **86 wall-hours at 4 workers ≈ 2 × 48h A100 jobs** chained with `--resume`.
- Full rerun chosen over selective (delete 2,972 corrupt results + `--resume`):
  selective saves only ~25% (~62 wall-h) and costs clean provenance, health
  telemetry on all frames, and carries boundary-frame risk (bbox calls corrupt
  while discovery calls clean at a window transition — the oracle can't see
  bbox responses because old runs stored only verdicts).
- **Keep the old `qwen2.5vl_72b` tag** — it is the incident dataset.

### Step 4 — Prompt-sensitivity: recover the deleted run, or rerun it

**The `qwen2.5vl_72b_old_prompt` results were deleted on 2026-08-04.** This is
the only data behind Table `prompt_sensitivity`, and 681 of its 1,000 frames
were clean — the corruption-free subset where the finding actually gets
*stronger* (41.7% → 73.4% valid, versus the published, corruption-diluted
64.4% → 85.2%). It is one of the few results that survives this incident, so
losing the data would cost a section.

Recovery options, cheapest first:

1. **Restore from trash** (~0 cost). Still present as of 2026-08-04:
   `/run/media/tom/ml/.Trash-1000/files/qwen2.5vl_72b_old_prompt`
   ```bash
   mv /run/media/tom/ml/.Trash-1000/files/qwen2.5vl_72b_old_prompt \
      /run/media/tom/ml/zod_temp/vlm/
   ```
   Then restrict the analysis to the 681 clean frames as originally planned.
   Verify the restore with `test_health_fix.py`-style classification before
   trusting it.
2. **Check the HPC copy** — the run was produced there; `zod_temp/vlm/` on
   gpfs may still hold it.
3. **Rerun the old prompts** (~83 frame-hours ≈ 21 wall-hours at 4 workers for
   1,000 frames). The old prompt is recoverable from git: commit `6770195`
   ("feat: imrpove ablation study and prompts") is the revision, so its parent
   `6770195^` holds the original `agents/bbox_agent.py` and
   `agents/discovery_agent.py`. Note the revision changed **both** the prompt
   wording and the parse rule (the `UNCLEAR → None` branch was added in
   `6770195`), which is what the paper means by "plus a stricter
   response-parsing protocol" — a rerun must restore both to be a fair
   comparison.

If none of these are taken, the prompt-sensitivity section has to be dropped;
its published numbers cannot be defended, since they are corruption-diluted.

### Step 5 — Regenerate downstream (only what's tainted)

1. Re-merge consensus: `merge_annotations.py --tag-a qwen2.5vl_72b_v2 --tag-b llava_34b` (offline, cheap).
2. Retrain fusion **only** on: qwen `vlm_only`, `triage`, `swin_discovery`,
   `raw_sam_discovery`, and the merged-consensus variants.
   - Use `--seed` (pre-2026-07-14 checkpoints were unseeded).
   - Evaluate with the corrected per-variant-reference protocol
     (`dump_frame_metrics.py` + `bootstrap_miou.py`), not the old
     self-referencing one.
3. Do **not** retrain llava / `swin_only` / `raw_sam` variants.

### Step 6 — Paper updates

- Regenerate tables: `agent_behavior`, `ablation` (qwen + consensus rows),
  `prompt_sensitivity`, `timing`.
- Rewrite the cross-model narrative expecting the divergence to shrink
  dramatically; the "protective scepticism" and "consensus recovers parity"
  results will likely change qualitatively.
- **Consider turning the incident into a contribution**: "a VLM serving fault
  silently defaulted 72% of verdicts and produced a plausible-looking
  cross-model finding; here is the monitoring that catches it" is an honest,
  reviewer-friendly operational lesson that annotation-pipeline papers rarely
  report — and the corrupted run is a ready-made case study.

### Order of operations

~~commit fix~~ → ~~harden slurm~~ → **rescue the deleted old-prompt data**
(Step 4 — trash entries do not survive forever) → pull on HPC →
**launch qwen rerun** (long pole) → prompt-sensitivity recompute while it runs
→ re-merge → retrain tainted variants → regenerate tables → rewrite paper.

---

## Appendix: verification commands

Per-frame corruption oracle (run against any tag's `results/`):

```python
# a frame is corrupt when every discovered[].vlm_response has no alphanumeric chars
def degenerate(s):
    st = (s or "").strip()
    return bool(st) and not any(c.isalnum() for c in st)
```

Timing reference (from stored `run_info`): llava mean 116 s/frame
(133 frame-hours total); qwen clean-frame mean 298 s/frame, corrupt-frame mean
480 s/frame (timeout/retry overhead — corruption made the run *slower*, not
faster). Total VLM calls per run: ~91k.
