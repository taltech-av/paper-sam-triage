# Data and artifacts

The code in this repository is small; the artifacts it consumes are not. This
file describes what is published, where it lives, and what every field means.

## What is published

| Component | Size | Contents |
|---|---|---|
| `human_verification.tar.gz` | 4.3 MB | `verify_export.csv` — the human reference, 145,428 rows |
| `responses_llava_34b.tar.gz` | 6.6 MB | 4,111 per-frame JSONs: every LLaVA-1.6-34B response and score |
| `responses_qwen2.5vl_72b_v2.tar.gz` | 7.0 MB | 4,111 per-frame JSONs: the same for Qwen2.5-VL-72B |
| `discovery_masks.tar.gz` | 5.3 MB | 8,194 per-frame discovery component masks — needs a GPU to regenerate, and only to within a boundary pixel |
| `splits_and_frames.tar.gz` | 0.2 MB | frame lists and the train/val/test splits |
| `annotation_sam.tar.gz` | 286 MB | the 99,293 SAM pseudo-label masks the paper audits — the input, needed for Tier 2/3 |
| `best.pth` | 2.5 GB | the CLFTv2/Swin quality-model checkpoint, uncompressed |
| `config_9.json` | 2.4 KB | the config that checkpoint was trained under |
| `annotations.tar.gz` | 35 MB | the four replayed annotation variants; the base for the pixel accounting |

**59 MB for the offline reproduction path** (everything above `annotation_sam`):
the 215 MB of stored JSON compresses about sixteenfold, and that tier needs no
GPU, no ZOD download, and no checkpoint. The last three rows add ~3.2 GB and are
needed only to re-run the pipeline or retrain, and total 2.8 GB.

The checkpoint alone exceeds GitHub's 2 GB per-asset limit, which is why the
record lives on Zenodo; the 59 MB offline tier may also be mirrored as GitHub
release assets for convenience.

Published on Zenodo under **CC-BY-4.0**; the code in this repository is MIT.
Neither covers the underlying Zenseact Open Dataset imagery, which stays under
[ZOD's own licence](https://zod.zenseact.com/) — no ZOD pixels are redistributed
here. What is published is *decisions about* ZOD frames, keyed by frame id.

> **DOI:** [10.5281/zenodo.22010998](https://doi.org/10.5281/zenodo.22010998) — Zenodo record *VLM data analysis*.
> Cite it alongside the paper if you use the human verification data.

Rebuild the bundle from a working checkout with:

```bash
python release/make_bundle.py --out ~/zenodo_bundle
```

That script strips the annotator's name and email from the export before
archiving and writes `MANIFEST.txt` with sha256 sums over the published files.

## Unpacking

```bash
mkdir -p ~/zod_temp && cd ~/zod_temp
for f in ~/zenodo_bundle/*.tar.gz; do tar xzf "$f"; done
```

`best.pth` and `config_9.json` are not archives — point `VLM_SWIN_CKPT` and
`VLM_SWIN_CFG` at them directly, or drop them where `VLM_FUSION_DIR` expects them.

Then set `VLM_DATA_ROOT=~/zod_temp` in `.env`. The human export is read by path,
so put it where the commands in [REPRODUCE.md](REPRODUCE.md) expect it:

```bash
mkdir -p human_verified_output
cp ~/zod_temp/human_verification/verify_export.csv human_verified_output/
```

---

## `verify_export.csv` — the human reference

One row per region shown to the labeller, long format. 145,428 rows over 4,110
frames; the 35,984 rows with a non-empty `verdict` are the 1,001 frames the job
closed at. **Every automated decision travels with the region**, which is what
makes attribution a software problem rather than something the labeller was
asked about — they were never told which model produced what.

### Job and verdict

| Column | Meaning |
|---|---|
| `taskId` | labelling-platform task id |
| `frame` | ZOD frame id, e.g. `frame_000012` |
| `stratum` | sampling stratum the frame came from |
| `maskId` | region id, unique within the frame |
| `class` | annotated class: `vehicle`, `sign`, `cyclist`, `pedestrian` |
| `verdict` | `correct`, `incorrect`, or empty when never answered |
| `elapsedMs` | time the labeller spent on this region |
| `answeredAt` | timestamp — the ordering that reveals the calibration drift |
| `userEmail`, `userName` | annotator identity; **replaced with `annotator_1` in the published copy** |

### Region geometry

| Column | Meaning |
|---|---|
| `mask_kind` | `standalone` (no same-class SAM mask nearby) or `fringe` (a rim) |
| `mask_source` | which stage proposed it: SAM proposal or discovery candidate |
| `mask_bbox`, `mask_bbox_384` | bounding box in 768 px and 384 px space |
| `mask_pixel_count`, `mask_pixel_count_384` | region area in each space |
| `mask_size`, `mask_fill_ratio`, `mask_fill_ratio_band` | size bucket, box fill, and its band |
| `mask_ring_sam_share` | share of the 9 px ring around the region already covered by SAM |
| `mask_touching_sam` | whether that ring holds a same-class SAM mask — the geometry test behind Table IX |
| `mask_candidate_index` | rank among the frame's discovery candidates (largest first) |

### What each automated signal said

| Column | Meaning |
|---|---|
| `mask_lidar_support`, `mask_lidar_support_band` | fraction of mask pixels with LiDAR returns, and its band |
| `mask_swin_agreement`, `mask_swin_agreement_band` | α, the share of pixels the class map assigns to the annotated class |
| `mask_swin_class` | class the dense map actually predicted |
| `mask_consistency` | deterministic LiDAR check verdict |
| `mask_quality_agent` | quality verdict for the region |
| `mask_bbox_agent_<tag>` | that backend's crop verdict |
| `mask_vlm_response_<tag>` | the backend's raw reply, verbatim |
| `mask_triage_<tag>` | the triage outcome that run recorded |
| `mask_confirmed_<tag>` | for discovery candidates, whether that backend confirmed it |

`<tag>` is `llava_34b` or `qwen2.5vl_72b_v2`. Because both runs scored the *same*
deterministically extracted proposals, a verdict on a region is run-independent
and applies to both.

> The stored `mask_triage_<tag>` fields may predate later rule changes. The paper's
> numbers come from *replaying* the current rule over the stored signals, which is
> what `analyze_human_verification.py` does; the two differ by about a point.

---

## `vlm/<tag>/results/frame_*.json` — stored model responses

One JSON per frame. This is what makes the paper reproducible offline: every
model call's input state, raw reply, and derived verdict is on disk, so any rule
can be re-evaluated without re-running inference.

```
frame_id                     "frame_000012"
run_info                     model, timestamp, elapsed_seconds, n_masks,
                             n_vlm_calls, n_swin_bypass, workers
masks[]                      one entry per SAM proposal
  mask_id, class_id, class_name, bbox, pixel_count
  agents                     bbox | quality | failure_mode | correction | consistency
  scores                     swin_agreement, lidar_support, swin_bypass,
                             swin_q_threshold, swin_skip_threshold
  triage                     accept | reject | refine | human_review
  timing                     elapsed_seconds, agent_seconds, vlm_calls
  metadata                   aspect_ratio, bbox_width, ...
discoveries[]                one entry per discovery candidate
```

Class ids follow `config.CLASS_ID_TO_NAME`: `0` background, `1` ignore,
`2` vehicle, `3` sign, `4` cyclist, `5` pedestrian.

**`triage: refine` is an artifact for `qwen2.5vl_72b_v2`.** Every CorrectionAgent
call fails under that backend, so the bucket records a failure, not a decision.
Pool it into human review for any cross-model comparison.

---

## Annotation PNGs

Single-channel uint8, one pixel per class id, same resolution as the 768 px
camera images. A rejected mask's pixels revert to `0` (background) — the
annotation records what a rule *kept*, not what it deleted, so a false reject and
a genuine background pixel are indistinguishable after the fact. That is why the
human reference is scored on regions rather than on annotation diffs.

Directory names encode the rule: `annotation_raw_sam` (no triage),
`annotation_swin_only` (dense agreement threshold), `annotation_triage`
(the three-signal rule), with `_discovery` and `_ccm` suffixes for discovery
variants. Regenerate any of them with `replay_triage.py`.

## Frame lists

| File | Rows | Meaning |
|---|---|---|
| `frames/vlm_frames.csv` | 4,110 | **the canonical set** — every published figure uses this |
| `frames/bad_frames.csv` | 4,135 | the original flagged partition, before 25 frames were written off to the serving fault |
| `frames/good_frames.csv` | 2,319 | the clean partition that trains the quality signal |

Use `vlm_frames.csv`. See the troubleshooting section of
[REPRODUCE.md](REPRODUCE.md) for why the other two differ.
