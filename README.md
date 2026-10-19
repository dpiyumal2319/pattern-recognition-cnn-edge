# EN3150 Assignment 03 — Resource-Constrained CNN for Edge Image Classification

A lightweight image classifier for embedded and low-resource devices, built on **EuroSAT**
(27,000 Sentinel-2 RGB tiles, 10 land-use classes, natively 64×64).

Two custom architectures — a standard CNN and a depthwise separable variant under 100,000
parameters — are compared against each other and against two fine-tuned lightweight SOTA
backbones, on accuracy, memory footprint and computational cost.

The SOTA harness builds MobileNetV2 and EfficientNet-B0 through torchvision's
model registry, replaces their final linear prediction layer for EuroSAT's ten
classes, and applies `model.freeze_backbone`. A frozen feature-extraction phase
keeps the backbone in evaluation mode while training the new classifier; a
second config with `freeze_backbone: false` can restore that checkpoint and
fine-tune all parameters. Primary comparisons use 64×64 inputs, while the
supplementary MobileNetV2 config uses 224×224 through the same data pipeline.

| | |
|---|---|
| Brief | [`docs/EN3150_Assignment_03_In23.pdf`](docs/EN3150_Assignment_03_In23.pdf) |
| Contributor guide | [`AGENTS.md`](AGENTS.md) |
| Frozen interfaces | [`docs/interfaces.md`](docs/interfaces.md) |
| Board | [Issues](https://github.com/dpiyumal2319/pattern-recognition-cnn-edge/issues) · 6 milestones |
| Deadline | 02 October 2026 |

---

## Quickstart — local

Requires Python 3.10 or newer and Git. Clone the repository, then create an
isolated environment from the repository root:

```bash
git clone https://github.com/dpiyumal2319/pattern-recognition-cnn-edge.git
cd pattern-recognition-cnn-edge
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on macOS/Linux or
`.\.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the project:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Prove the pipeline works without downloading anything (about 3 seconds):

```bash
python scripts/run_experiment.py --config configs/model_b.yaml --smoke
```

Prepare EuroSAT once, then run an experiment and generate report inputs:

```bash
python scripts/prepare_data.py --config configs/base.yaml
python scripts/run_experiment.py --config configs/model_b.yaml
python scripts/build_tables.py --all
python scripts/build_figures.py --all
python scripts/build_report.py          # needs latexmk (MiKTeX or TeX Live)
```

The report is `report/main.tex`. Each task's prose is in `report/sections/`,
and any generated table or figure that does not exist yet renders as a red
"Pending" box naming the issue that owes it, so the PDF always builds.

The Task 5 explanation of the noisy frozen EfficientNet-B0 curves rests on a
diagnostic that retrains only the prediction layer on fixed backbone features.
It needs the prepared dataset and about ten minutes on a CPU, and writes
`results/frozen_head_diagnostic/diagnostic.json`:

```bash
python scripts/diagnose_frozen_head.py
```

Each experiment writes its resolved configuration, history, test metrics,
resource profile, checkpoint selection, and best checkpoint to
`results/<run_id>/`. The dataset and checkpoints are intentionally ignored by
Git; JSON results and generated figures are committed for the report.

Data preparation caches RGB channel means and population standard deviations in
`data.norm_stats` (default `data/norm_stats.json`). They come exclusively from the
manifest's training images, using native-resolution pixels scaled to `[0, 1]`
before resizing or augmentation. The same statistics therefore serve both 64
and 224 input sizes. Cache validation checks relative training paths, labels
and file sizes without decoding images; a changed source or malformed cache
triggers recomputation. The cache's `_source` field is internal metadata;
`compute_norm_stats(cfg)` returns only `mean` and `std`.

The image transform helper resizes to `data.input_size` using bilinear
interpolation, converts RGB pixels to a float tensor, and applies those training
statistics. With `data.augment: true`, training images additionally receive
independent horizontal/vertical flips (each with probability 0.5) and a uniform
choice of 0, 90, 180 or 270 degree rotation. These preserve overhead land-cover
labels. Validation/test transforms are deterministic; `augment: false` makes
training preprocessing deterministic too. Random transforms use PyTorch's RNG
and the project's existing seed setup.

`get_dataloaders(cfg)` returns `(train_loader, val_loader, test_loader)`, using
`data.batch_size` and `data.num_workers` for every split. Training shuffles each
epoch; validation/test preserve manifest order. Every sample is retained,
including incomplete final batches, and tensors stay on CPU until the training
or evaluation loop moves them to its device. Call `set_seed(cfg.seed)` once at
the beginning of a run, as the experiment CLI already does. Loader generators
seed sampling and workers independently per split; main-process augmentation
(`num_workers: 0`) uses that run-level seed. Reproducibility assumes the same
worker settings. On Windows, call loaders from a script guarded by
`if __name__ == "__main__":` when using workers.

The preparation command also writes a fixed-seed original/augmented sample grid
to `report/figures/data_augmentation_grid.png`. Its matching LaTeX methodology
text is in `report/sections/data_preparation.tex` for the final report assembly.

## Quickstart — Colab

Colab is the canonical compute environment: every member gets the same free T4 regardless of their
laptop, which matters because *training time per epoch* is a graded column and only comparable
across identical hardware. The canonical driver is
[`notebooks/colab_run.ipynb`](notebooks/colab_run.ipynb).

1. Open the notebook in Colab. While the repository is private, upload the
   notebook or authorize Colab's GitHub browser.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. Create a fine-grained GitHub token with read access to this repository. Add
   it to Colab's **Secrets** panel as `GITHUB_TOKEN`, enable notebook access,
   and never paste it into a code cell.
4. Review `BRANCH`, `CONFIG_PATH`, `RUN_PREPARE_DATA`, and `SMOKE` in the first
   code cell, then choose **Runtime → Run all**.
5. Approve the Drive mount. The first real run downloads EuroSAT; later
   runtimes reuse `MyDrive/EN3150_A03/data/eurosat/`.

The notebook calls `scripts/prepare_data.py` and `scripts/run_experiment.py`;
it contains no separate model or training implementation. Completed run
directories are copied to `MyDrive/EN3150_A03/results/<run_id>/`, including the
best checkpoint, so a Colab disconnect does not lose the artifacts. The local
`profile.json` also records the exact device name used for timing comparisons.

To test the complete wiring without downloading EuroSAT or using a GPU, set
`SMOKE = True`; the notebook then produces `_smoke_<run_id>` artifacts.

### Notebook hygiene

CI rejects notebooks with saved outputs or execution counts. Install the
repository's pre-commit hook once in each development clone:

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

The pinned `nbstripout` hook strips notebook outputs automatically before each
commit. This supplements the CI check and prevents output-only merge conflicts.

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
