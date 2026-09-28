# Interfaces

**Frozen contract.** Every signature, schema and CLI flag below is fixed by issue #1 before any
implementation starts, so that 25 parallel issues code against the same target instead of inventing
25 variations.

Changing anything here requires its own issue and a PR that updates every call site. Do not drift
from it quietly — a renamed argument that looks harmless in your branch breaks three other people
on merge.

---

## 1. Config schema

One YAML per experiment in `configs/`. The filename stem **is** the `run_id`, so
`configs/model_b_sgdm.yaml` writes to `results/model_b_sgdm/`.

```yaml
run_id: model_b            # must match the filename stem
seed: 42
device: auto               # auto | cpu | cuda | mps

data:
  root: data/
  manifest: data/splits.csv
  input_size: 64           # 64 for custom models and primary SOTA runs; 224 for the supplementary run
  batch_size: 128
  num_workers: 2
  augment: true            # train split only, always false for val and test
  norm_stats: data/norm_stats.json

model:
  name: model_b            # model_a | model_b | mobilenet_v2 | efficientnet_b0
  num_classes: 10
  channels: [32, 64, 128]  # custom models only
  head: gap                # gap | flatten  (custom models only)
  dropout: 0.2
  pretrained: false        # SOTA only
  freeze_backbone: false   # SOTA only

optimizer:
  name: adam               # adam | sgd
  lr: 0.001
  momentum: 0.0            # sgd only; ignored by adam
  weight_decay: 0.0

training:
  epochs: 30               # brief requires >= 20 for the custom models
  smoke: false
```

Unknown keys are an error, not a warning. A typo'd key that silently does nothing is how you lose a
day wondering why momentum had no effect.

---

## 2. Module and function map

Stubbed by #1 with `raise NotImplementedError`, then filled by the issue named in the comment.

### `src/pr_edge/config.py`

```python
@dataclass(frozen=True) class DataConfig:      root, manifest, input_size, batch_size, num_workers, augment, norm_stats
@dataclass(frozen=True) class ModelConfig:     name, num_classes, channels, head, dropout, pretrained, freeze_backbone
@dataclass(frozen=True) class OptimizerConfig: name, lr, momentum, weight_decay
@dataclass(frozen=True) class TrainingConfig:  epochs, smoke
@dataclass(frozen=True) class Config:          run_id, seed, device, data, model, optimizer, training

def load_config(path: str | Path, overrides: dict | None = None) -> Config
def dump_config(cfg: Config, path: str | Path) -> None       # the resolved config, into the run dir
```

### `src/pr_edge/device.py`

```python
def get_device(pref: str = "auto") -> torch.device           # auto -> cuda, else mps, else cpu
def set_seed(seed: int) -> None                              # random, numpy, torch, torch.cuda
def device_name(device: torch.device) -> str                 # e.g. "NVIDIA GeForce RTX 3060 Laptop GPU"
```

### `src/pr_edge/paths.py`

```python
PROJECT_ROOT: Path
DATA_DIR: Path        # data/
RESULTS_DIR: Path     # results/
REPORT_DIR: Path      # report/

def run_dir(run_id: str, smoke: bool = False) -> Path        # results/<run_id>/ or results/_smoke_<run_id>/
def resolve_path(path: str | Path) -> Path                   # config path -> repo-root-relative  [#2]
```

Config files hold relative paths, so every one of them must go through `resolve_path` rather than
being used as-is. Resolving against the process CWD instead would make `scripts/prepare_data.py`
create `scripts/data/` when launched from inside `scripts/`, and do the same in a Colab notebook
that `cd`s after cloning. Absolute paths pass through untouched.

### `src/pr_edge/schemas.py`

```python
HISTORY_FIELDS, METRICS_FIELDS, PER_CLASS_FIELDS, PROFILE_FIELDS, SELECTION_FIELDS: dict[str, tuple]

def validate_history(data: Any) -> list[str]        # violations; an empty list means valid
def validate_metrics(data: Any) -> list[str]
def validate_profile(data: Any) -> list[str]
def validate_selection(data: Any) -> list[str]
def check_selection_matches_history(selection: Any, history: Any) -> list[str]   # best_epoch = argmin val_loss
def validate_file(path: Path) -> list[str]          # dispatches on filename
def validate_run_dir(directory: Path) -> list[str]  # also reports missing files; cross-checks selection vs history
```

The machine-checkable form of section 3 below - prose cannot fail a build. Deliberately
dependency-free: the schemas are simple enough that a short validator is clearer than
`jsonschema` or `pydantic`, and one fewer pinned package is one fewer thing to break on a Colab
runtime. `tests/test_schemas.py` runs these against every committed results file on every PR.

### `src/pr_edge/data.py`

```python
CLASS_NAMES: list[str]    # the 10 EuroSAT classes, fixed order, the ONLY source of class ordering
SPLIT_FRACTIONS: dict[str, float]    # {"train": 0.70, "val": 0.15, "test": 0.15}
class DataError(RuntimeError)        # dataset on disk or manifest is not what we require

def prepare_eurosat(cfg: Config, force: bool = False) -> Path                    # -> manifest path   [#2]
def read_manifest(manifest: str | Path) -> list[dict[str, str]]                  # rows of the CSV    [#2]
def validate_manifest(rows: list[dict[str, str]]) -> None                        # raises DataError   [#2]
def summarise_manifest(rows: list[dict[str, str]]) -> str                        # printable table    [#2]
def compute_norm_stats(cfg: Config) -> dict[str, list[float]]                    # {"mean":[...], "std":[...]}  [#3]
def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]    # train, val, test   [#3]
def get_synthetic_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]   # smoke mode  [#1]
```

`compute_norm_stats` uses the **training split only**. `get_dataloaders` applies augmentation to the
training loader only; val and test are deterministic.

**The split manifest** `data/splits.csv` is committed and is the partition — not a cache of it.
Three columns, no more:

| Column | Value |
|---|---|
| `filepath` | relative to `<data.root>/eurosat/2750/`, forward slashes, e.g. `AnnualCrop/AnnualCrop_1.jpg` |
| `label` | the class **name**; the index comes from `CLASS_NAMES.index()`, never from the file |
| `split` | `train` \| `val` \| `test` |

Rows are sorted by `(class index, filepath)` and written with an explicit `\n` terminator, so the
file is byte-identical on Windows, Linux and Colab. Nothing regenerates it implicitly:
`prepare_eurosat` returns an existing manifest untouched unless called with `force=True`.

### `src/pr_edge/models/`

```python
# __init__.py
def build_model(cfg: Config) -> nn.Module        # registry dispatch on cfg.model.name    [#1 registry, #9/#10/#19 entries]

# model_a.py   [#9]
class ModelA(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None
    def forward(self, x: Tensor) -> Tensor       # (N,3,H,W) -> (N, num_classes) logits

# model_b.py   [#10]
class ModelB(nn.Module): ...                     # same signature as ModelA
class DepthwiseSeparableBlock(nn.Module):        # depthwise 3x3 then pointwise 1x1, separately inspectable
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None

# sota.py      [#19]
def build_sota(cfg: Config) -> nn.Module         # torchvision backbone, replaced head, freeze policy applied
```

All models return **raw logits**, never softmax. Loss is `CrossEntropyLoss`, which expects logits.

### `src/pr_edge/train.py`

```python
def train(model: nn.Module, cfg: Config, train_loader: DataLoader, val_loader: DataLoader,
          device: torch.device, out_dir: Path) -> list[dict]
    # the history records; also writes history.json, model.pt and selection.json   [#6]
def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer                          [#6]
```

Best checkpoint is selected by **lowest validation loss** and saved to `model.pt`. Ties go to the
earlier epoch; non-finite losses are never selected. Before returning, `train` **restores the
selected weights into `model`**, so the caller's `evaluate` and `profile_model` describe the same
weights as `model.pt`. `history.json` is rewritten every epoch and `model.pt` on each improvement,
both atomically, so an interrupted run keeps its best checkpoint. The test set is touched exactly
once, after training, by `evaluate`.

### `src/pr_edge/evaluate.py`

```python
def evaluate(model: nn.Module, loader: DataLoader, cfg: Config, split: str = "test") -> dict    # writes metrics.json  [#4]
```

### `src/pr_edge/profile.py`

```python
def count_parameters(model: nn.Module) -> tuple[int, int]            # (total, trainable)           [#5]
def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> tuple[int, dict[str, int]]        # [#5]
def state_dict_bytes(model: nn.Module) -> int                        # measured from an actual save [#5]
def profile_model(model: nn.Module, cfg: Config, mean_epoch_seconds: float) -> dict                 # [#5]
```

### `src/pr_edge/plots.py`

```python
def plot_loss_curves(run_ids: list[str], out: Path, title: str | None = None) -> Path     [#17]
def plot_confusion_matrix(run_id: str, out: Path, normalize: bool = True) -> Path         [#17]
```

### `src/pr_edge/tables.py`

```python
def build_dataset_table(manifest: Path, out: Path) -> Path                [#24]
def build_custom_comparison(run_ids: list[str], out: Path) -> Path        [#18]
def build_sota_comparison(run_ids: list[str], out: Path) -> Path          [#22]
def build_final_comparison(run_ids: list[str], out: Path) -> Path         [#25]
```

Each emits a LaTeX `tabular` fragment for `\input{}`. No `\begin{table}` wrapper — the caption and
placement belong to the report, not the generator.

---

## 3. Result file schemas

Exact. CI validates against these, so a missing or renamed key fails the build.

### `history.json` — a JSON array, one object per epoch

```json
[
  {
    "epoch": 1,
    "train_loss": 2.1934,
    "train_acc": 0.2148,
    "val_loss": 2.0311,
    "val_acc": 0.2722,
    "epoch_seconds": 12.41,
    "lr": 0.001
  }
]
```

`epoch` is 1-indexed. `epoch_seconds` is **training only** — it excludes the validation pass, so the
"training time per epoch" column in the report means what it says.

### `metrics.json`

```json
{
  "run_id": "model_b",
  "split": "test",
  "num_samples": 4050,
  "class_names": ["AnnualCrop", "Forest", "..."],
  "accuracy": 0.8123,
  "macro_precision": 0.8044,
  "macro_recall": 0.8011,
  "macro_f1": 0.8021,
  "per_class": {
    "AnnualCrop": {"precision": 0.83, "recall": 0.79, "f1": 0.81, "support": 450}
  },
  "confusion_matrix": [[0, 0]],
  "confusion_matrix_convention": "rows=true, cols=predicted, order=class_names"
}
```

`class_names` is always written alongside the matrix so a figure can never be mislabelled.
`per_class` is keyed by **class name**, not index.

### `profile.json`

```json
{
  "run_id": "model_b",
  "total_params": 84210,
  "trainable_params": 84210,
  "disk_size_bytes": 340112,
  "disk_size_kb": 332.14,
  "disk_size_mb": 0.32,
  "macs": 7231488,
  "macs_per_layer": {"features.0.depthwise": 110592},
  "input_shape": [1, 3, 64, 64],
  "mean_epoch_seconds": 12.41,
  "device": "cuda",
  "device_name": "NVIDIA GeForce RTX 3060 Laptop GPU"
}
```

Definitions, fixed so two people cannot report different numbers for the same model:

- **`macs`** — multiply-accumulates for **one forward pass of one image** (batch size 1). Bias
  additions excluded. A `k x k` conv producing `H x W x C_out` from `C_in` channels costs
  `k * k * C_in * C_out * H * W`.
- **`disk_size_bytes`** — measured from an actual `torch.save(model.state_dict())`, not estimated
  from the parameter count.
- **`disk_size_kb`** — `bytes / 1024`. **`disk_size_mb`** — `bytes / 1024**2`. The brief wants KB for
  the custom models and MB for the SOTA models; both are emitted so the report picks per table.
- **`mean_epoch_seconds`** — mean of `history.json`'s `epoch_seconds`.
- **`device_name`** — recorded because epoch time is a graded column and is meaningless without it.

### `selection.json`

Which epoch `model.pt` holds, so the report can cite it rather than implying the model trained
monotonically to its best result.

```json
{
  "criterion": "val_loss",
  "best_epoch": 14,
  "best_val_loss": 0.3121,
  "best_val_acc": 0.9012,
  "epochs_trained": 30
}
```

`best_val_loss` and `best_val_acc` are copied from that epoch's `history.json` record. `validate_run_dir`
checks that `best_epoch` is the first epoch holding the lowest finite `val_loss` in `history.json`.

### Run directory layout

```
results/<run_id>/
  config.yaml      resolved config, including the seed
  history.json
  metrics.json
  profile.json
  selection.json   which epoch model.pt holds
  model.pt         gitignored
  figures/*.png
```

---

## 4. CLI surface

```bash
python scripts/prepare_data.py  --config configs/base.yaml [--force]
python scripts/run_experiment.py --config configs/model_b.yaml [--smoke] [--epochs N] [--device D] [--run-id ID]
python scripts/build_tables.py  [--all | --table custom|sota|dataset|final]
```

Command-line flags override the config file; the **resolved** values are what gets dumped to
`config.yaml` in the run directory.

### `--smoke` contract

Frozen now because CI depends on it and retrofitting it after five modules exist is painful.

- Synthetic random tensors, **no download, no disk dataset**: 200 train / 50 val / 50 test, 10 classes
- 2 epochs, batch size 16, CPU
- Writes to `results/_smoke_<run_id>/`, which is **gitignored**
- Still produces all four contract files, fully schema-conformant
- Must complete in **under 60 seconds** on a CPU-only GitHub runner

The point is not to test accuracy. It is to prove that the wiring from config to run directory is
intact, on every pull request, in under a minute.

---

## 5. Test file map

Each file is created by #1 as a skipped placeholder and turned green by the issue named. An issue is
not done until its test file passes.

| File | Turned green by | Notable assertion |
|---|---|---|
| `tests/test_config.py` | #1 | Round-trip load/dump; unknown keys raise |
| `tests/test_device.py` | #1 | `set_seed` makes two runs bit-identical |
| `tests/test_schemas.py` | #1 | Every committed `results/*/[history\|metrics\|profile\|selection].json` validates |
| `tests/test_smoke.py` | #1, #6 | `run_experiment.py --smoke` writes a conformant run dir in under 60s |
| `tests/test_train.py` | #6 | `model.pt` and the returned model hold the lowest-val-loss epoch, not the last |
| `tests/test_data.py` | #2, #3 | Split is stratified, 70/15/15, deterministic; norm stats are train-only |
| `tests/test_metrics.py` | #4 | Accuracy, precision, recall match a hand-computed example |
| `tests/test_profile.py` | #5 | Reproduces the `CNNs.pdf` example: ~590k ops standard vs ~67.8k separable |
| `tests/test_model_a.py` | #9 | Forward pass shape `(N,3,64,64) -> (N,10)` |
| `tests/test_model_b.py` | #10 | **`trainable_params <= 100_000`** — the brief's hard constraint |
| `tests/test_sota.py` | #19 | Head replaced for 10 classes; freeze policy applied |
| `tests/test_tables.py` | #18, #22 | Generated LaTeX contains every required column |

---

## 6. Conventions that prevent silent divergence

- `CLASS_NAMES` in `data.py` is the **only** source of class ordering. Never re-derive it from a
  directory listing — filesystem order is not guaranteed across platforms.
- Models return logits. Softmax lives in the loss or in the reporting, never in `forward`.
- Anything reported in the same table must come from runs on the **same device**. Epoch time is
  otherwise incomparable.
- Generated files (`report/tables/*.tex`, `report/figures/*.png`) are never merged on conflict.
  Rerun the generator.
