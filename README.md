# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

A lightweight image classifier for embedded and low-resource devices, built on **EuroSAT**
(27,000 Sentinel-2 RGB tiles, 10 land-use classes, natively 64×64).

Two custom architectures — a standard CNN and a depthwise separable variant under 100,000
parameters — are compared against each other and against two fine-tuned lightweight SOTA
backbones, on accuracy, memory footprint and computational cost.

| | |
|---|---|
| Brief | [`docs/EN3150_Assignment_03_In23.pdf`](docs/EN3150_Assignment_03_In23.pdf) |
| Contributor guide | [`AGENTS.md`](AGENTS.md) |
| Frozen interfaces | [`docs/interfaces.md`](docs/interfaces.md) |
| Board | [Issues](https://github.com/dpiyumal2319/pattern-recognition-cnn-edge/issues) · 6 milestones |
| Deadline | 02 October 2026 |

---

## Quickstart — local

```bash
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt && pip install -e .
```

Prove the pipeline works without downloading anything (about 3 seconds):

```bash
python scripts/run_experiment.py --config configs/model_b.yaml --smoke
```

Then the real thing:

```bash
python scripts/prepare_data.py --config configs/base.yaml
python scripts/run_experiment.py --config configs/model_b.yaml
python scripts/build_tables.py --all
```

Data preparation caches RGB channel means and population standard deviations in
`data.norm_stats` (default `data/norm_stats.json`). They come exclusively from the
manifest's training images, using native-resolution pixels scaled to `[0, 1]`
before resizing or augmentation. The same statistics therefore serve both 64
and 224 input sizes. Cache validation checks training paths, file sizes and
modification times without decoding images; a changed source or malformed cache
triggers recomputation. The cache's `_source` field is internal metadata;
`compute_norm_stats(cfg)` returns only `mean` and `std`.

The image transform helper resizes to `data.input_size` using bilinear
interpolation, converts RGB pixels to a float tensor, and applies those training
statistics. With `data.augment: true`, training images additionally receive
independent horizontal/vertical flips (each with probability 0.5) and a uniform
choice of 0, 90, 180 or 270 degree rotation. These preserve overhead land-cover
labels. Validation/test transforms are deterministic; `augment: false` makes
training preprocessing deterministic too. Random transforms use PyTorch's RNG
and the project's existing seed setup. DataLoader factory integration follows
in the next subtask of #3.

## Quickstart — Colab

Colab is the canonical compute environment: every member gets the same free T4 regardless of their
laptop, which matters because *training time per epoch* is a graded column and only comparable
across identical hardware. `notebooks/colab_run.ipynb` (issue #7) clones the repo, installs
dependencies, caches the dataset to Drive and calls the same scripts.

## Running the tests

```bash
pytest
```

Unimplemented modules have skipped placeholder tests naming the issue that owns them, so a green
run with skips is the expected state until the board is complete.

---

## Layout

```
src/pr_edge/        package: config, device, data, models, train, evaluate, profile, plots, tables
scripts/            CLI entrypoints
configs/            one YAML per experiment; the filename stem IS the run_id
results/<run_id>/   config.yaml, history.json, metrics.json, profile.json, figures/
report/             LaTeX; tables/ and figures/ are generated, never hand-edited
tests/              one file per module, mapped to owning issues in docs/interfaces.md
docs/slides/        EN3150 lecture decks — the syllabus anchor for every methodological choice
```

### How a number reaches the report

```
results/*/[metrics|profile].json → scripts/build_tables.py → report/tables/*.tex → \input{}
```

Never hand-type a result into LaTeX. Rerun an experiment and the report updates itself; type a
number and it silently goes stale.

---

## Notes that will cost you a day if you miss them

- **`torchvision>=0.20` is mandatory.** Earlier versions download EuroSAT from `madm.dfki.de`,
  which is dead. 0.20+ uses the HuggingFace `torchgeo/eurosat` mirror.
- **EuroSAT is natively 64×64**, so the brief's "downscale to a maximum of 64×64" is satisfied by
  construction. The report must state this explicitly rather than skipping the requirement.
- **Never call `.cuda()`.** Always `get_device()` — members run on CUDA, MPS and CPU.
- **Runs that share a table must share a device.** Epoch time is otherwise meaningless.
