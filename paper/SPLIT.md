# Paper plan: one full paper, with a split held in reserve

**Decision (2026-08-14): `paper/full/main.tex` is the primary artifact — a single, standalone, full
paper.** It is no longer framed as a technical report backing two shorter submissions. The split
below stays on disk as a fallback in case a venue's length limit forces it, but neither half is the
thing we are submitting.

| # | Artifact | Path | Role | Length | Status |
|---|----------|------|------|--------|--------|
| 1 | **Full paper** | `paper/full/main.tex` | **primary submission** | 18 pp | needs number refresh |
| 2 | Audit (split half) | `paper/ral/` | fallback | 9 pp — 1 over an 8 pp budget | headline resolved |
| 3 | System + portability (split half) | `paper/icaart/` | fallback | 6 pp | complete |

The full paper is self-contained and cites neither half. The two split halves cross-cite each other
and the full paper (`\cite{audit}`, `\cite{companion}`, `\cite{techreport}`) — those placeholders in
`references.bib` only matter if the split is revived.

Everything below documents the split: what each half owns, and the distinctness check. It is
accurate but currently dormant.

**Each paper directory is self-contained.** `paper/full/`, `paper/ral/` and `paper/icaart/` each
carry their own `main.tex`, `figures/`, `diagrams/`, `tables/` and `references.bib`, and none reads
a file from outside its own directory — verified against the `main.fls` input list, not just by
compiling. Any one of the three can be zipped and uploaded to a venue as-is. The cost is ~7 MB of
duplicated PNGs, which is the right trade for not shipping a submission with a broken image path.
`paper/` itself now holds only this file and the three directories.

**`paper/full/` is a strict superset of the other two.** Verified, not assumed: every table, every
diagram and every figure referenced by `ral/main.tex` or `icaart/main.tex` is also referenced by
`full/main.tex`, and each split's distinctive claims (57.0 % agreement, the safe-default bound, the
serving fault, the 41-point discovery collapse, 20.4 % wrong proposals, the 11:1 exchange rate,
23.1 % contaminated pixels, review targeting) all appear in it. The full paper additionally carries
the iSEAuto transfer and the dataset/splits tables, which neither split took.

---

## Who owns what

**RA-L — "Label Purity, Not Verification"**

The measurement and the verdict. Given a human reference over every proposal and every candidate:
verification is a purity–volume trade-off, F1 ranks rules backwards, the free deterministic signal
dominates zero-shot VLM triage on the error direction that matters, discovery is a boundary editor
contaminating 23.1 % of object pixels, and human deletion at fixed training-set size is the best buy.

Owns: the human reference construction · the error-budget criterion · the downstream ladder
(+3.9 / −9.4 / +2.5) · the discovery geometry decomposition and pixel accounting · cost against
human effort · review targeting.

**ICAART — "Fixed Prompts Do Not Transfer"**

The system and what happens when its backend changes. Five agents, byte-identical prompts, two
open-weight VLMs, verified-identical non-VLM inputs across 4,110 frames. Portability fails per-agent
and per-class rather than uniformly: 57.0 % verdict agreement, a class-redistributed rather than
shifted rejection profile, a 41-point collapse on the one discovery prompt carrying a three-way
discrimination, and one agent whose prompt does not transfer at all.

Owns: the five-agent architecture and output contracts · the safe-default policy and the
`valid_answered ≤ valid_as-recorded` bound · the controlled cross-backend comparison · response
provenance and the serving-fault disclosure · deployment cost · the two structural weaknesses of the
triage rule.

---

## Distinctness

Asset sets are disjoint by construction — no table and no diagram appears in both.

| | RA-L | ICAART |
|---|---|---|
| **Tables** | `manual_verification` (×2), `ablation`, `discovery_geometry`, `cleaning_contrast`, `review_targeting` | `agent_behavior`, `agent_performance`, `timing` |
| **Diagrams** | `candidate_geometry` (new, purpose-built; redrawn 2026-08-14) | `architecture_overview`, `pipeline_end_to_end` |
| **Photo figures** | 4 (`geom_*`, purpose-built) | all 14 (`qual_*` ×10, `limit_*` ×4) |

**Updated 2026-08-14.** The original plan sent all 14 PNGs to the arXiv report alone, on the
reasoning that both papers could cite it rather than carry figures. That reasoning died when
`main.tex` became a standalone full paper rather than a shared substrate, so ICAART now carries the
qualitative material itself: `fig:qualitative` (five agent decision paths) opens its Results, and
`fig:limitations` sits under the two-weaknesses subsection whose prose already narrated those exact
four images. Cost: 6 pp → 7 pp. Sharing figures with our own preprint is standard; the disjointness
that matters — RA-L against ICAART — is untouched.

**RA-L now has its own**, and it is disjoint from ICAART's by construction: `fig:geometry_cases`,
a 2×2 of real crops mirroring `candidate_geometry` cell for cell — edge bleed, false alarm, boundary
growth, new object — sharing the schematic's palette so the two read as one argument. Built by
`make_candidate_geometry_figure.py` from the `verify_export.csv` geometry join, `vlm/discovery_masks`
and the camera frames; the chosen frame ids are printed on every run, so the selection is auditable.
It cost no extra page — RA-L is still 9 pp.

Placing it exposed and fixed three pre-existing overflow bugs: `discovery_geometry` was 56 pt too
wide for its column and had been silently spilling its "Final 200" column into the right-hand
column, and `review_targeting` and `cleaning_contrast` were each ~19 pt over. All three are now
wrapped in `\resizebox{\columnwidth}`. RA-L compiles with **zero** overfull boxes.

`candidate_geometry` was redrawn because the first version had colliding column headers, a colliding
legend, and a rim glyph that read as a blob beside a blob rather than a rim. It now draws the
candidate as the annulus it geometrically is, since SAM coverage is subtracted before components are
formed.

**Measured prose overlap.** Excluding preamble and funding boilerplate, the two papers share
**21 of ~3,579 eight-gram shingles — 0.59 %**. Every one is a parameter specification that *should*
be identical because it describes the same system: the LiDAR threshold `τ=0.1`, the quality
thresholds `0.30 / 0.15`, the negative-signal set `{invalid, bad, fail}`, the `384×384` comparison
resolution, and the serving configuration. No shared sentence of argument or exposition remains.

Reproduce the check with the shingle script in the session log, or:

```bash
cd paper && python3 -c "..."   # 8-gram shingle intersection, preamble stripped
```

RA-L overlaps the arXiv report at ~41 % and ICAART at ~7 %. The former is expected and unproblematic
— that is what a preprint of your own work is.

---

## Submission order

Dormant while `paper/full/` is the plan. If the split is revived:

1. **ICAART** needs nothing pending except format: it is drafted in two-column IEEEtran so it
   compiles; ICAART is SCITEPRESS (single column, apalike). Download the 2027 author kit, swap
   `\documentclass` and the bib style, re-fit. **Confirm the page limit** — SCITEPRESS regular
   papers are normally longer than 7 pp, so there may be more room than budgeted.
2. **Then** post `paper/full/` to arXiv and fill `techreport`'s eprint ID in `references.bib`.
3. **RA-L** — the central comparison is resolved; remaining work is the 1-page length cut and
   (optionally) the two `Triage + disc.` rows. Rolling submission, so no deadline pressure.

---

## Blockers

**Resolved (2026-08-14): the headline comparison is closed.** Both triage rows landed and the answer
is negative — VLM triage *subtracts* from the free Swin signal.

| Variant | Veh | Sign | Human | mIoU | fw-IoU | Δ vs Swin only (95 % CI) |
|---|---|---|---|---|---|---|
| Raw SAM | 67.5 | 31.1 | 25.3 | 40.3 | 58.9 | −3.9 [−5.3, −2.5] |
| **Swin only** | **69.6** | **37.8** | **28.4** | **44.2** | **61.8** | *(reference)* |
| Swin + disc. (no VLM) | 60.6 | 25.3 | 21.7 | 34.8 | 52.4 | −9.4 [−10.4, −8.4] |
| Triage (LLaVA) | 67.5 | 35.4 | 26.8 | 41.1 | 59.6 | **−3.0 [−4.5, −1.6]** |
| Triage (Qwen) | 68.5 | 33.3 | 28.0 | 41.5 | 60.1 | **−2.7 [−4.1, −1.5]** |

Neither triage variant separates from unfiltered raw SAM (+0.8 CI [−1.0, +2.6] and +1.2 CI
[−0.8, +2.9]), and the two backends do not separate from each other (paired CI [−0.9, +1.4]) despite
the 3.2× cost gap. Source: `fusion-training/logs/vlm/frame_metrics/ladder`, via `bootstrap_miou.py`.

**Still pending: the two `Triage + disc.` rows** (`config/vlm/clftv2-base/{llava,qwen}/config_full_fusion.json`).
They are no longer a blocker — they measure the triage×discovery interaction, not the headline.
Still marked `\pending{}` (renders blue) and one `\TODO`. When they land:

```latex
\renewcommand{\pending}[1]{#1}
\renewcommand{\pendingnote}{}
```

**New blocker: length.** RA-L was at exactly 8 pp with zero slack; the triage result pushes it to
9 pp. About 23 lines of bibliography spill onto page 9 — roughly 0.4 of a column of body text to
recover. The `\TODO` block reclaims ~2 lines at submission; the rest needs an editorial cut.

**Highest-value extra run: the geometry-gated discovery variant.** One training run. Currently the
paper says the pixel contamination is a *sufficient* explanation for the −9.4 mIoU but stops short of
calling it the *operative* one. That run closes the gap, and it needs no VLM call and no human label,
so it applies to all 4,110 frames. It converts the paper's one actionable recommendation from
inference to demonstration. Worth more than any additional VLM row.

Not needed by either paper: the modality ablation and the iSEAuto convergence figure. Both were
weak, both are pending, and neither survived the cut. They belong in the arXiv report.

---

## ⚠ Stale numbers in `paper/full/main.tex`

`compare_models.py` regenerated the tables on the full **4,110-frame / 90,176-mask** paired set, but
the prose in `paper/full/main.tex` still cites the superseded **3,535-frame / 73,708-mask** subset. The
ICAART paper uses the corrected table values throughout; the arXiv report must be refreshed before
posting.

| Claim in prose | Stale | Current (tables) |
|---|---|---|
| Paired set | 3,535 fr / 73,708 masks | **4,110 fr / 90,176 masks** |
| Masks with Swin scores | 72,989 | **90,176** |
| Discovery candidates | 45,669 | **55,256** |
| Cross-VLM BBox agreement | 56.8 % | **57.0 %** |
| Qwen safe_default share | 13.3 % | **11.3 %** |
| Qwen valid, answered only | 67.5 % | **67.2 %** |
| Qwen discovery hit rate | 82.5 % | **82.4 %** |
| GPU-hours (LLaVA / Qwen) | 108.6 / 356.2 | **132.3 / 426.9** |
| Wall-clock | 27.1 / 89.0 h | **33.1 / 106.7 h** |
| Cost multiple | 3.3× | **3.2×** |
| Per-frame time | 110.6 / 362.7 s | **109.0 / 352.2 s** |
| BBox call mean (Qwen) | 11.63 s | **11.39 s** |
| Per-class reject, sign | 26.7 / 10.8 | **26.0 / 10.8** |
| Per-class reject, cyclist | 48.1 / 33.8 | **46.5 / 33.3** |
| Per-class reject, vehicle | 5.0 / 16.2 | **5.1 / 16.3** |
| Per-class reject, pedestrian | 17.3 / 35.1 | **16.9 / 35.4** |
| Discovery confirm, vehicle | 83.3 / 87.5 | **83.0 / 87.6** |
| Discovery confirm, sign | 84.2 / 89.1 | **84.2 / 89.3** |
| LLaVA prec/rec/F1 | 71.8 / 71.4 / 71.6 | **72.0 / 71.1 / 71.5** |
| Qwen prec/rec/F1 | 78.5 / 68.3 / 73.0 | **69.0 / 77.8 / 73.1** ⚠ |
| Qwen answered, prec/acc | 75.1 / 67.3 | **75.6 / 67.5** |

⚠ The Qwen precision/recall pair is **transposed** in the prose, not merely stale — the text reads
"higher-recall, lower-precision" with the numbers the wrong way round. The characterisation is
correct; the figures behind it were swapped.

Two further items:

- **Correction Agent counts.** The "1,449 Qwen calls / 1,311 LLaVA calls" figures were computed on
  the old 3,535-frame subset. Regenerate on 4,110. The qualitative null result (Qwen returns no
  usable content; refine path never fires) is unaffected and is carried in ICAART with a `\TODO`.
- **Candidate-count mismatch — RESOLVED (2026-08-14).** `agent_behavior` reports 55,256 candidates,
  `discovery_geometry` reports 55,252. The four are candidates with no exact connected-component
  mask, so they have no geometry to test and drop out of any measurement that needs their pixels.
  `agent_behavior` counts the whole pool; `discovery_geometry` counts the testable pool. Both are
  right; say which is which if either is challenged. Confirmed independently by the geometry-gate
  replay, which saw 55,252 testable candidates and reported exactly 4 without a mask.
