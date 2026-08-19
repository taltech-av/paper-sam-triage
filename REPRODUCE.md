# Reproducing the published results

Every reported number traces to a command here. The map is grouped by what
you need to run it: most of the paper reproduces **offline on a laptop**, because
the pipeline stores every model response and the analysis replays them.

| Tier | Needs | Reproduces |
|---|---|---|
| 1. Offline replay | ~1 GB disk, no GPU | the verification-coverage, per-rule agreement, operating-point, per-class backend, cost, and review-targeting tables; every human-reference number in Sections V-A, V-B, V-E, V-F |
| 2. Pipeline re-run | ZOD + GPU + Ollama + 2.8 GB of bundle | The stored responses themselves (`vlm/<tag>/results/`) |
| 3. Downstream training | Tier 2 output + [paper-sam-triage-training](https://github.com/taltech-av/paper-sam-triage-training) | the downstream ladder and the cleaning contrast; the mIoU ladder in Sections V-C, V-D |

Start by unpacking the artifact bundle and pointing `.env` at it — see [DATA.md](DATA.md).

```bash
cp .env.example .env      # set VLM_DATA_ROOT to the unpacked bundle
pip install -r requirements.txt
```

---

## Tier 1 — offline, no GPU

These read `human_verified_output/verify_export.csv` (the human reference) and
`$VLM_DATA_ROOT/vlm/<tag>/results/*.json` (the stored responses). No model is
called and no image is decoded, so a full pass takes minutes.

### The human reference and the triage rules

```bash
python analyze_human_verification.py --export human_verified_output/verify_export.csv
```

One pass writes a full report to `human_verified_output/verify_export.report.txt`
and produces, in these blocks:

| Paper item | Block in the report |
|---|---|
| **Verification coverage** (regions judged, per-class rejection rates) | `SAM PROPOSALS`, lower block |
| **Per-rule agreement** (bad kept / good deleted) | `TRIAGE VARIANTS` — the *replayed* rules, not the stored per-run decisions, which differ by ~1 point |
| **Per-class backend comparison** (F1 per backend) | `CROSS-RUN ADJUDICATION` |
| **Verification cost** (human time, 20.9 h over 1,001 frames) | `PROGRESS` |
| Annotator drift (−2.3 pp/100 frames, r = −0.90) | `PROGRESS` |
| "About a fifth of proposals are wrong" (20.4 %) | `SAM PROPOSALS` |

Restrict to one section with `--only {sam,variants,discovery,geometry}`, and add
`--bootstrap` for CIs on the rates.

### Operating points, and review targeting

```bash
python sweep_triage_operating_points.py --export human_verified_output/verify_export.csv
```

Prints the LiDAR-threshold sweep, the disjunctive rule, the dense-agreement-only
corner, and the review-ordering block. The shipped setting (τ = 0.10 → 54.9 /
14.1) must reproduce the `triage` row of the per-rule agreement table exactly — that equality is the check that the replay is faithful.

### Discovery candidate geometry

Candidate counts and pixel shares come from two scripts, because a candidate
count and a pixel count are not interchangeable:

```bash
# candidate columns (78.6 % rims, 6.1 % new objects)
python analyze_human_verification.py --export human_verified_output/verify_export.csv --only geometry

# pixel columns (+37.5 % object-pixel budget, 61.7 % of it rejected material)
python discovery_pixel_budget.py --export human_verified_output/verify_export.csv \
    --out human_verified_output/discovery_pixel_budget.txt
```

`discovery_pixel_budget.py` re-derives what each candidate actually painted by
repeating `replay_triage._add_discoveries` (384 px component map upsampled
NEAREST, claiming only pixels triage left as background). It reconciles with the
shipped annotation to ~111 px/frame against ~10,800 px/frame added — about 1 %
— and prints that figure as it runs. A much larger gap means `--base`/`--variant`
point at the wrong annotation directories.

### Cross-backend agreement (57.0 %)

```bash
python compare_models.py --tag-a llava_34b --tag-b qwen2.5vl_72b_v2
```

Makes the two runs comparable despite their asymmetries and reports the share of
crops on which both backends return the same verdict.

### Rebuilding the annotation variants

Every training set in the ladder is a rule replayed over the stored responses:

```bash
python replay_triage.py --list-variants                 # all 11 rules
python replay_triage.py --variant all                   # the 3 paper variants
python replay_triage.py --variant triage --with-discovery
python replay_triage.py --variant swin_only --with-discovery --discovery-geometry-gate
```

The human-verified (clean) arm is not a rule and has its own writer:

```bash
python make_clean_annotations.py --export human_verified_output/verify_export.csv
```

---

## Tier 2 — re-running the pipeline

Needs the ZOD release (from Zenseact, not redistributed here) plus three things
the artifact bundle ships: `annotation_sam.tar.gz`, `best.pth`, and
`config_9.json`. Also an Ollama server. The SAM annotations are the output of the
[upstream generator](https://github.com/taltech-av/paper-aim2026-zod-sam-generator);
they are republished here so this record is self-sufficient.

```bash
# smoke test: deterministic mock backend, no GPU, no Ollama
python process_frames.py --mock --limit 10 --tag smoke

# the published runs
python process_frames.py --model qwen2.5vl:72b --tag qwen2.5vl_72b_v2 --resume
python process_frames.py --model llava:34b     --tag llava_34b        --resume
```

On a cluster, `sbatch slurms/vlm-qwen.slurm` (see the HPC block of
`.env.example`). Both slurm jobs wrap the run in a restart loop — read the
serving-fault note below before trusting a completed run.

Rebuilding the human labelling job from pipeline output:

```bash
python make_label_bundle.py            # one zip, one job: proposals + candidates in one pass
```

---

## Tier 3 — downstream training

The downstream ladder and the cleaning contrast are trained in the
[paper-sam-triage-training](https://github.com/taltech-av/paper-sam-triage-training)
repo, not here. This repo produces their inputs and consumes their outputs.

```bash
# 1. splits (here) — the 1,001 human-verified frames are the test set
python make_splits.py --frames "$VLM_DATA_ROOT"/vlm/human_verified/frames.csv \
    --out-dir "$VLM_DATA_ROOT"/vlm/human_verified/splits \
    --annotation-dir "$VLM_DATA_ROOT"/vlm/human_verified/annotation

# 2. train each variant (paper-sam-triage-training), then dump per-frame metrics
#    → dump_frame_metrics.py, one JSON per variant on the SAME test frames

# 3. paired, weather-stratified bootstrap CIs (here)
python bootstrap_miou.py --metrics-dir <dumps> --pair swin_only raw_sam

# 4. the paper's LaTeX tables (here)
python make_corrected_tables.py
```

`bootstrap_miou.py` resamples test frames with replacement *within* each weather
condition and uses identical resample indices for every variant, so deltas
between variants are paired. **The printed Δ is the resample mean, not the
difference of the two point estimates** — they agree to a rounding step
everywhere except the cleaning contrast.

The cleaning contrast is two configs under `config/vlm/clftv2-base/cleaning/`
that differ in exactly one field, `annotation_path`; splits, architecture, and
hyperparameters are identical.

---

## Figures

| Figure | Command |
|---|---|
| Pipeline overview | `paper/ral/diagrams/architecture_overview.tex` (TikZ, no script) |
| Discovery candidate cases (edge bleed / growth / false alarm / new object) | `python make_candidate_geometry_figure.py` |
| Qualitative triage examples | `python select_qualitative_frames.py` → browse the contact sheets → `python make_qualitative_figures.py` |

`select_qualitative_frames.py` narrows the corpus to contact sheets; it does not
pick the final panels. The published figures were chosen by hand from its output.

---

## Things that will trip you up

These are the failure modes we actually hit. Each one silently produces
plausible-looking wrong numbers.

**The frame set is 4,110, not 4,135.** Use `frames/vlm_frames.csv`. The 25-frame
difference is the frames written off to the serving fault below; every published
figure is computed on the 4,110.

**`qwen2.5vl_72b` (v1) is not a usable run.** A few hours after each model load,
the Ollama instance stopped emitting a stop token and answered every call with
garbage. The 2026-06 qwen run lost 2,972 of 4,135 frames to this and *still
reported success*. The published run is the `_v2` tag; v1 is kept only as the
record of the fault. `vlm/health.py` now detects the degeneracy and aborts with
exit 42, and the slurm loop reloads the model and resumes.

**The `refine` bucket is an artifact, not a result.** Every CorrectionAgent call
fails under `qwen2.5vl:72b`, so anything that lands in `refine` for that backend
is a failure mode, not a decision. Cross-model comparisons pool `refine` into
human review; splitting them flips a sign.

**`DISCOVERY_MAX_CANDIDATES` must stay 20.** It is not a tuning knob — it fixes
which objects can be discovered at all. Both published runs used 20, giving the
candidate set the paper reports. It is deliberately *not* set by `use_hpc()`, so
a run cannot silently differ by which machine it landed on.

**Absolute human rates are a mixture over a moving standard.** The labeller's
acceptance rate falls 2.3 pp per hundred frames across the pass (84.3 % → 71.4 %)
while the incoming frames stay flat, so the standard tightened rather than the
frames getting harder. Every comparison in the paper is paired on the same masks,
which the drift affects equally; do not read a single absolute rate as a
population estimate. The report prints final-200 and final-400 blocks so you can
see the end-of-pass calibration separately.

**Discovery masks regenerate to within a boundary pixel, not to the byte.**
`regenerate_discovery_masks.py` re-runs the segmentation model, and floating-point
kernels differ across GPU architectures. Regenerating the published set on an
RTX 5070 Ti against the A100 that produced it gives 93.2 % byte-identical frames,
880 differing pixels out of 6.6M (0.013 %), and an identical candidate count of
55,256. Every published share in the candidate-geometry table is unchanged except the
object-pixel inflation, which moves 37.5 % → 37.4 %. The bundle therefore ships the original
masks rather than asking you to regenerate them; use `--out` if you want to
regenerate and diff without overwriting the published copy.

**One checkpoint directory per training run.** Two runs sharing a logdir will
silently select a checkpoint trained on the current test frames.

**Ladder rows are single runs.** Each interval covers test-frame resampling, not
training stochasticity. The geometry gate is one trained variant at one
threshold, not a swept curve.
