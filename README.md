# VLM Annotation Triage Pipeline

A multi-signal pipeline that triages [SAM](https://segment-anything.com)
pseudo-labels on the Zenseact Open Dataset — a LiDAR support check, a dense
camera–LiDAR class map, and a zero-shot vision–language judge — together with the
human verification pass and the analysis used to audit it.

Toomas Tahves, Mauro Bellone, Raivo Sell (Tallinn University of Technology).
Manuscripts describing this work are under review; this repository and its
artifacts stand on their own and are cited from them.

Automated checks for [SAM](https://segment-anything.com) pseudo-labels are
usually validated against another automated signal, which cannot say which model
is correct or whether the labels train better models. This repository is the
pipeline that was audited, plus the analysis that audits it against **a human
who judged every proposal and every discovery candidate** in 1,001 Zenseact Open
Dataset frames — 35,984 verified regions, independent of every automated signal.

What the audit found, each reproducible from the commands in
[REPRODUCE.md](REPRODUCE.md):

- About a fifth of SAM proposals are wrong, so every rejection rule buys purity
  by deleting correct labels — and ranking rules on a single accuracy score
  **reverses** their downstream order.
- The only filter that helps is a per-frame class map the pipeline already
  computes: **+3.9 mIoU**. Adding either of two open VLM backends gives most of
  that back, at 4–13× the time a human needs to judge the same frames by hand.
- Object discovery is a boundary editor, not an object finder: 78.6 % of its
  candidates are rims around objects SAM already segmented.
- Deleting the 5.3 % of object pixels the human rejected, at fixed training-set
  size, is worth **+2.5 mIoU**. Label purity limits the model, not label volume.

## Related repositories

| Stage | Repository |
|---|---|
| Upstream — generates the SAM annotations audited here | [paper-aim2026-zod-sam-generator](https://github.com/taltech-av/paper-aim2026-zod-sam-generator) |
| **This repository** — triage, discovery, human verification, analysis | — |
| Downstream — trains CLFTv2 on each annotation variant | [paper-sam-triage-training](https://github.com/taltech-av/paper-sam-triage-training) |

## Quick start

Most of the paper reproduces **offline, on a laptop, with no GPU and no ZOD
download**: the pipeline stored every model response, so the analysis replays
them instead of calling a model. That tier is a 59 MB download — no GPU, no
ZOD account, no checkpoint.

```bash
git clone https://github.com/taltech-av/paper-vlm-annotation-pipeline
cd paper-vlm-annotation-pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # set VLM_DATA_ROOT to the unpacked bundle
```

Download the artifact bundle — Zenodo [10.5281/zenodo.22010998](https://doi.org/10.5281/zenodo.22010998), unpacking
instructions in [DATA.md](DATA.md) — then:

```bash
# every human-reference number in the paper, one pass
python analyze_human_verification.py --export human_verified_output/verify_export.csv

# the operating-point sweep and the review-targeting table
python sweep_triage_operating_points.py --export human_verified_output/verify_export.csv
```

No artifacts and no GPU? The pipeline runs against a deterministic mock backend:

```bash
python process_frames.py --mock --limit 10 --tag smoke
```

## Running the pipeline

Needs ZOD, the upstream SAM annotations, the CLFTv2/Swin checkpoint, and an
[Ollama](https://ollama.com) server.

```bash
python process_frames.py --model qwen2.5vl:72b --tag qwen2.5vl_72b_v2 --resume
python process_frames.py --model llava:34b     --tag llava_34b        --resume
```

| Model | VRAM (Q4) | Role |
|---|---|---|
| `qwen2.5vl:72b` | ~40 GB | published run A |
| `llava:34b` | ~20 GB | published run B |
| `qwen2.5vl:7b` | ~5 GB | local development |

Key flags for `process_frames.py`:

| Flag | Effect |
|---|---|
| `--model` / `--tag` | Ollama model; output namespace under `vlm/<tag>/` |
| `--resume` | skip frames that already have results |
| `--mock` | deterministic mock client, no Ollama |
| `--no-swin` / `--no-discovery` | drop the dense quality signal / the discovery stage |
| `--hpc` | use the `VLM_HPC_*` paths from `.env` |
| `--limit N` | process at most N frames |

On a cluster: `sbatch slurms/vlm-qwen.slurm`. Both jobs wrap the run in a
restart loop that reloads the model when `vlm/health.py` detects the serving
degeneracy described in [REPRODUCE.md](REPRODUCE.md#things-that-will-trip-you-up)
— read that section before trusting a completed run.

## Layout

```
config.py                        all thresholds, vocabularies, and paths (via .env)
process_frames.py                pipeline entry point
agents/                          one module per judge
  consistency_agent.py             deterministic LiDAR support check (free)
  swin_quality_agent.py            dense class-map agreement (free, per frame)
  bbox_agent.py                    zero-shot VLM crop recognition (one call per mask)
  discovery_agent.py               proposes objects SAM missed
core/
  mask_extractor.py                deterministic proposal extraction — the reason
                                   two runs score identically comparable regions
  triage.py                        the concordance rule: two negatives to delete
  bundle.py                        crop/context assembly for a model call
vlm/                             Ollama client, mock client, degeneracy detector
output/                          annotation PNG and results JSON writers
release/make_bundle.py           packages the published artifact bundle
```

Analysis and replay scripts live at the top level; [REPRODUCE.md](REPRODUCE.md)
maps each one to the table or figure it produces.

## Documentation

| File | Contents |
|---|---|
| [REPRODUCE.md](REPRODUCE.md) | every paper number → the exact command, and the failure modes to avoid |
| [DATA.md](DATA.md) | artifact bundle contents, full schema of every published file |
| [.env.example](.env.example) | the machine-specific paths, all of them |

## Citing

See [CITATION.cff](CITATION.cff). Please cite both the paper and the artifact
DOI ([10.5281/zenodo.22010998](https://doi.org/10.5281/zenodo.22010998)) if you use the human verification data.

## License

Code MIT ([LICENSE](LICENSE)); published artifacts CC-BY-4.0. Neither covers the
underlying [Zenseact Open Dataset](https://zod.zenseact.com/) imagery — no ZOD
pixels are redistributed here, only decisions about ZOD frames.
