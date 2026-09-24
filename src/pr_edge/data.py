"""EuroSAT data pipeline.

EuroSAT is 27,000 Sentinel-2 RGB tiles across 10 land-use classes, natively
64x64 - so the brief's "downscale to a maximum of 64x64" is satisfied by
construction rather than by a resize step. The report must say this explicitly.

The split produced here is written to a committed manifest (``data/splits.csv``)
and is never recomputed at training time. That is deliberate: the brief requires
the fine-tuned SOTA models to use the same partition as the custom models, and
the only way to *prove* that across four machines is to ship the partition
itself rather than the code that would regenerate it.

Loader construction, normalisation statistics and augmentation belong to #3,
whose signatures are frozen below.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .config import Config
from .paths import REPORT_DIR, resolve_path

# The single source of class ordering for the whole project. Never re-derive
# this from a directory listing: filesystem order is not guaranteed across
# platforms, and a different order on one member's machine would silently
# transpose every confusion matrix they produce.
CLASS_NAMES: list[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Smoke-mode dataset sizes. Small enough to finish in seconds on a CPU runner,
# large enough that every class appears in every split.
SMOKE_SIZES = {"train": 200, "val": 50, "test": 50}

# The brief's 70/15/15. Insertion order matters: it is the tie-break order used
# when the largest-remainder allocation below has a whole image left to place.
SPLIT_FRACTIONS: dict[str, float] = {"train": 0.70, "val": 0.15, "test": 0.15}

MANIFEST_COLUMNS = ("filepath", "label", "split")

# torchvision extracts the archive to <root>/eurosat/2750/<ClassName>/*.jpg.
# Manifest paths are relative to that directory rather than to <root>, so the
# manifest stays valid for a member who keeps the dataset on another drive.
EUROSAT_IMAGE_SUBDIR = Path("eurosat") / "2750"

EXPECTED_TOTAL = 27_000


class DataError(RuntimeError):
    """Raised when the dataset on disk or the manifest is not what we require."""


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def _allocate(n: int) -> dict[str, int]:
    """Split ``n`` items into 70/15/15 by the largest-remainder method.

    Naive rounding does not sum back to ``n``: a class of 13 gives quotas of
    9.1, 1.95 and 1.95, and three independent ``round`` calls happen to work
    there but not in general. Largest remainder floors every quota and then
    hands the leftover items to the splits that were rounded down hardest, so
    the parts always sum to the whole and no image is dropped or duplicated.

    EuroSAT's own class sizes (3000, 2500, 2000) all divide exactly into
    70/15/15, so on the real dataset this reduces to plain division - but the
    code must not depend on a property of one dataset.
    """
    exact = {name: n * frac for name, frac in SPLIT_FRACTIONS.items()}
    counts = {name: math.floor(value) for name, value in exact.items()}

    leftover = n - sum(counts.values())
    if leftover:
        # Largest fractional part first; ties fall back to SPLIT_FRACTIONS order
        # (train, val, test) so the result is a pure function of n.
        order = sorted(
            exact,
            key=lambda name: (
                -(exact[name] - counts[name]),
                list(SPLIT_FRACTIONS).index(name),
            ),
        )
        for name in order[:leftover]:
            counts[name] += 1

    return counts


def _stratified_split(labels: Sequence[int], seed: int) -> list[str]:
    """Assign every index to "train", "val" or "test", stratified by label.

    Returns a list parallel to ``labels``.

    Each class draws from its own generator seeded with ``seed + label``, so
    adding or removing one class cannot reshuffle the others. A single shared
    stream would make the whole partition sensitive to a change in any one
    class - grow one class by ten images and every later class repartitions.

    **This function is positional.** An item's split depends on its rank within
    its class in the order given, so passing the same files in a different
    order yields a different partition. Making the split a property of the
    *files* rather than of the enumeration order is the caller's job:
    :func:`prepare_eurosat` sorts by ``(class index, filepath)`` first, which
    is what keeps Windows, Linux and a Colab runtime in agreement.
    """
    by_label: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        by_label[int(label)].append(index)

    assignment: list[str | None] = [None] * len(labels)

    for label in sorted(by_label):
        indices = np.array(sorted(by_label[label]))
        order = np.random.default_rng(seed + label).permutation(len(indices))
        shuffled = indices[order]

        cursor = 0
        for split, count in _allocate(len(indices)).items():
            for index in shuffled[cursor : cursor + count]:
                assignment[int(index)] = split
            cursor += count

    if any(value is None for value in assignment):  # pragma: no cover - guards a logic error
        raise DataError("internal error: an index was left unassigned by the split")
    return [str(value) for value in assignment]


# ---------------------------------------------------------------------------
# The manifest
# ---------------------------------------------------------------------------


def _write_manifest(rows: list[dict[str, str]], manifest: Path) -> None:
    r"""Write the manifest so four machines produce byte-identical files.

    ``newline=""`` plus an explicit ``\n`` terminator is not a style choice: the
    csv module emits CRLF by default, and on Windows the file object would then
    translate those again. A manifest that differed by platform would show up as
    a 27,000-line diff the first time somebody on another OS reran the script.
    """
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest: str | Path) -> list[dict[str, str]]:
    """Read the split manifest. Used by #3's loaders and by #24's dataset table."""
    manifest = resolve_path(manifest)
    if not manifest.exists():
        raise DataError(
            f"split manifest not found: {manifest}. Run scripts/prepare_data.py first."
        )
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_manifest(rows: list[dict[str, str]]) -> None:
    """Raise :class:`DataError` unless the manifest is usable and stratified.

    Checked on every load, not only on generation. The manifest is committed,
    which means a bad merge can corrupt it - and a silently broken partition is
    the one failure in this project that invalidates every number downstream of
    it instead of crashing.
    """
    if not rows:
        raise DataError("manifest is empty")

    missing = set(MANIFEST_COLUMNS) - set(rows[0])
    if missing:
        raise DataError(f"manifest is missing column(s): {sorted(missing)}")

    paths = [row["filepath"] for row in rows]
    if len(set(paths)) != len(paths):
        raise DataError(
            f"manifest lists {len(paths) - len(set(paths))} duplicate path(s) - "
            "an image in two splits leaks test data into training"
        )

    unknown_labels = {row["label"] for row in rows} - set(CLASS_NAMES)
    if unknown_labels:
        raise DataError(f"manifest holds label(s) outside CLASS_NAMES: {sorted(unknown_labels)}")

    unknown_splits = {row["split"] for row in rows} - set(SPLIT_FRACTIONS)
    if unknown_splits:
        raise DataError(f"manifest holds unknown split name(s): {sorted(unknown_splits)}")

    # Stratification: every class must hold its own 70/15/15, not merely the
    # totals. A manifest can be 70/15/15 overall while one whole class sits in
    # the test split.
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(SPLIT_FRACTIONS, 0))
    for row in rows:
        per_class[row["label"]][row["split"]] += 1

    for label in sorted(per_class):
        actual = per_class[label]
        expected = _allocate(sum(actual.values()))
        if actual != expected:
            raise DataError(
                f"class '{label}' is split {actual}, expected {expected}. "
                "Regenerate with: python scripts/prepare_data.py --force"
            )


def summarise_manifest(rows: list[dict[str, str]]) -> str:
    """Per-class counts per split, as a plain-text table.

    This is the printed summary the issue asks for, and the same numbers #24
    renders as the report's dataset table.
    """
    counts: dict[str, dict[str, int]] = {
        name: dict.fromkeys(SPLIT_FRACTIONS, 0) for name in CLASS_NAMES
    }
    for row in rows:
        counts[row["label"]][row["split"]] += 1

    width = max(len(name) for name in CLASS_NAMES)
    header = f"{'class':<{width}}  {'train':>7} {'val':>7} {'test':>7} {'total':>7}"
    lines = [header, "-" * len(header)]

    totals = dict.fromkeys(SPLIT_FRACTIONS, 0)
    for name in CLASS_NAMES:
        row_counts = counts[name]
        for split, value in row_counts.items():
            totals[split] += value
        lines.append(
            f"{name:<{width}}  {row_counts['train']:>7} {row_counts['val']:>7} "
            f"{row_counts['test']:>7} {sum(row_counts.values()):>7}"
        )

    grand = sum(totals.values())
    lines.append("-" * len(header))
    lines.append(
        f"{'total':<{width}}  {totals['train']:>7} {totals['val']:>7} "
        f"{totals['test']:>7} {grand:>7}"
    )
    if grand:
        shares = " ".join(f"{100 * totals[s] / grand:>6.2f}%" for s in ("train", "val", "test"))
        lines.append(f"{'share':<{width}}  {shares} {100.0:>6.2f}%")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Manifest-backed image dataset
# ---------------------------------------------------------------------------


class _EuroSATDataset(Dataset):
    """One manifest split, decoded lazily with an optional image transform.

    This is internal plumbing for #3's statistics and loaders; the frozen
    public interface remains unchanged. With no transform it returns RGB PIL
    images, so statistics can use the original pixels. Each instance owns its
    transform, allowing train augmentation and deterministic evaluation to
    share the same image-loading code.
    """

    def __init__(
        self,
        cfg: Config,
        split: str,
        transform: Callable[[Image.Image], Image.Image | torch.Tensor] | None = None,
    ) -> None:
        if split not in SPLIT_FRACTIONS:
            raise DataError(f"unknown dataset split: {split!r}")

        # Validate the whole partition before filtering: validating only one
        # split would both miss cross-split duplicates and fail stratification.
        rows = read_manifest(cfg.data.manifest)
        validate_manifest(rows)
        image_root = resolve_path(cfg.data.root) / EUROSAT_IMAGE_SUBDIR
        self.image_root = image_root
        self.samples = [
            (image_root / row["filepath"], CLASS_NAMES.index(row["label"]))
            for row in rows
            if row["split"] == split
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image | torch.Tensor, int]:
        path, label = self.samples[index]
        image: Image.Image | torch.Tensor
        try:
            # Conversion loads the pixels into an independent RGB image. The
            # source file closes before transforms run, including in workers.
            with Image.open(path) as source:
                image = source.convert("RGB")
        except OSError as exc:
            raise DataError(f"cannot read EuroSAT image: {path}") from exc

        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


class _RandomQuarterTurn:
    """Rotate a square RGB image by a uniformly chosen multiple of 90 degrees.

    PIL transpose permutes pixels exactly, without interpolation or padding.
    Use torch's RNG so the existing set_seed and future DataLoader worker seeds
    control augmentation. A module-level callable is picklable on Windows.
    """

    def __call__(self, image: Image.Image) -> Image.Image:
        turns = int(torch.randint(4, ()).item())
        if turns == 0:
            return image
        rotations = (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270)
        return image.transpose(rotations[turns - 1])


def _build_transform(
    cfg: Config, split: str, stats: dict[str, list[float]],
) -> Callable[[Image.Image], torch.Tensor]:
    """Resize, optionally augment training images, then scale and normalise.

    The caller supplies compute_norm_stats(cfg)'s training-only statistics once
    for all three splits. This helper never reads data or estimates statistics.
    The frozen augment flag enables the fixed flip/quarter-turn policy; no
    additional configuration keys or public entry points are introduced.
    """
    from torchvision import transforms

    if split not in SPLIT_FRACTIONS:
        raise DataError(f"unknown transform split: {split!r}")
    size = cfg.data.input_size
    if type(size) is not int or size <= 0:
        raise DataError("data.input_size must be a positive integer")

    operations = [transforms.Resize(
        (size, size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True,
    )]
    if split == "train" and cfg.data.augment:
        # Overhead land-cover labels are preserved by flips and quarter-turns.
        # Each flip independently has probability 0.5 (torchvision's default).
        operations.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            _RandomQuarterTurn(),
        ])
    operations.extend([
        transforms.ToTensor(),  # RGB uint8 -> float32 CHW in [0, 1]
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])
    return transforms.Compose(operations)


def _write_sample_grid(
    cfg: Config,
    out: str | Path = REPORT_DIR / "figures" / "data_augmentation_grid.png",
) -> Path:
    """Write a reproducible original/augmented pair for every manifest class.

    Images are selected only from the training partition. Both columns pass
    through the production resize and normalisation transforms; tensors are
    denormalised solely for display. The local RNG context keeps report figure
    generation from changing a subsequent experiment's random state.
    """
    import matplotlib.pyplot as plt

    stats = compute_norm_stats(cfg)
    original = _EuroSATDataset(cfg, "train", transform=_build_transform(cfg, "val", stats))
    augmented = _EuroSATDataset(cfg, "train", transform=_build_transform(cfg, "train", stats))
    first_by_label: dict[int, int] = {}
    for index, (_, label) in enumerate(original.samples):
        first_by_label.setdefault(label, index)
    if not first_by_label:
        raise DataError("cannot create sample grid: training split is empty")

    labels = sorted(first_by_label)
    pairs_per_row = 2
    rows = math.ceil(len(labels) / pairs_per_row)
    figure, axes = plt.subplots(
        rows, pairs_per_row * 2, figsize=(10, 2.45 * rows), squeeze=False,
    )
    mean = torch.tensor(stats["mean"], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(stats["std"], dtype=torch.float32).view(3, 1, 1)

    def display(tensor: torch.Tensor) -> np.ndarray:
        return (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # fork_rng is CPU-only here: transforms use the CPU torch generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(cfg.seed)
        for position, label in enumerate(labels):
            row, pair = divmod(position, pairs_per_row)
            index = first_by_label[label]
            plain, _ = original[index]
            changed, _ = augmented[index]
            for column_offset, (image, heading) in enumerate(
                ((plain, "Original"), (changed, "Augmented"))
            ):
                axis = axes[row, pair * 2 + column_offset]
                axis.imshow(display(image))
                axis.set_title(f"{CLASS_NAMES[label]}\n{heading}", fontsize=9)
                axis.axis("off")

    # An incomplete final pair leaves two intentionally blank cells.
    for position in range(len(labels) * 2, rows * pairs_per_row * 2):
        axes.flat[position].axis("off")
    figure.suptitle("EuroSAT training samples: original and online augmentation", fontsize=13)
    figure.tight_layout(rect=(0, 0, 1, 0.975))
    out = resolve_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def prepare_eurosat(cfg: Config, force: bool = False) -> Path:
    """Download EuroSAT and write the stratified 70/15/15 split manifest.

    Returns the manifest path. Re-running is a no-op: an existing manifest is
    validated and returned without touching the download, because regenerating
    a split that experiments have already been run against would silently
    invalidate every result in ``results/``. ``force=True`` is the explicit
    opt-in to that.
    """
    root = resolve_path(cfg.data.root)
    manifest = resolve_path(cfg.data.manifest)

    if manifest.exists() and not force:
        rows = read_manifest(manifest)
        validate_manifest(rows)
        print(f"[data] reusing existing manifest ({len(rows)} images): {manifest}")
        print(summarise_manifest(rows))
        return manifest

    # Imported here rather than at module scope: torchvision's dataset machinery
    # is only needed on the one code path that actually downloads, so keeping it
    # out of the import graph leaves CLASS_NAMES cheap to import from anywhere.
    from torchvision.datasets import EuroSAT

    image_root = root / EUROSAT_IMAGE_SUBDIR
    print(f"[data] dataset root: {root}")
    if not image_root.exists():
        print("[data] downloading EuroSAT (~94 MB) from the torchgeo HuggingFace mirror ...")

    dataset = EuroSAT(root=str(root), download=True)

    # The divergence guard from docs/interfaces.md section 6. ImageFolder derives
    # its class order from a sorted directory listing; CLASS_NAMES is the
    # project's declared order. They agree today, and this is what makes it a
    # loud error rather than a silent relabelling if they ever stop agreeing.
    if list(dataset.classes) != CLASS_NAMES:
        raise DataError(
            "on-disk class order does not match CLASS_NAMES.\n"
            f"  on disk:     {list(dataset.classes)}\n"
            f"  CLASS_NAMES: {CLASS_NAMES}"
        )

    records = [
        (Path(path).relative_to(image_root).as_posix(), int(label))
        for path, label in dataset.samples
    ]
    # Sort before splitting so the seed, not the filesystem, decides the split.
    records.sort(key=lambda record: (record[1], record[0]))

    splits = _stratified_split([label for _, label in records], cfg.seed)
    rows = [
        {"filepath": path, "label": CLASS_NAMES[label], "split": split}
        for (path, label), split in zip(records, splits)
    ]

    validate_manifest(rows)
    _write_manifest(rows, manifest)

    if len(rows) != EXPECTED_TOTAL:
        # Not fatal - the offline test fixture is deliberately small - but on a
        # real run it means the extraction was incomplete.
        print(f"[data] NOTE: found {len(rows)} images, expected {EXPECTED_TOTAL} for full EuroSAT")

    print(f"[data] wrote {len(rows)} rows (seed {cfg.seed}) to {manifest}")
    print(summarise_manifest(rows))
    return manifest


def _norm_fingerprint(dataset: _EuroSATDataset) -> str:
    """Identify local training files without decoding pixels on cache hits.

    A changed partition, label or file size invalidates the cache. Relative
    paths deliberately make the fingerprint portable between local machines
    and Colab. EuroSAT files are immutable inputs, so hashing their full content
    on every cache hit would repeat nearly all of the expensive image I/O.
    """
    digest = hashlib.sha256(b"eurosat-rgb-native-population-v1\n")
    for path, label in dataset.samples:
        try:
            stat = path.stat()
        except OSError as exc:
            raise DataError(f"cannot inspect EuroSAT training image: {path}") from exc
        record = (path.relative_to(dataset.image_root).as_posix(), label, stat.st_size)
        digest.update(json.dumps(record).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _read_norm_cache(path: Path, fingerprint: str) -> dict[str, list[float]] | None:
    """Ignore stale, incomplete or malformed caches and recompute them."""
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(cached, dict) or cached.get("_source") != fingerprint:
        return None
    for name in ("mean", "std"):
        values = cached.get(name)
        if not isinstance(values, list) or len(values) != 3:
            return None
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            return None
        if any(not 0 <= value <= 1 for value in values):
            return None
        if name == "std" and any(value == 0 for value in values):
            return None
    # Metadata stays on disk; callers get exactly the frozen mean/std contract.
    return {name: [float(value) for value in cached[name]] for name in ("mean", "std")}


def compute_norm_stats(cfg: Config) -> dict[str, list[float]]:
    """Cache RGB population mean/std from native, unaugmented training pixels.

    Statistics describe pixels scaled to [0, 1]. They are independent of the
    requested input size, batch size, augmentation and seed: all experiments
    use the same manifest and native-image preprocessing statistics. Only
    training files are inspected or decoded; val/test never contribute.

    The JSON cache contains mean/std and a private ``_source`` fingerprint.
    Changed training file metadata or a changed partition triggers a rescan.
    """
    dataset = _EuroSATDataset(cfg, "train")
    if not len(dataset):
        raise DataError("cannot compute normalisation statistics: training split is empty")
    cache = resolve_path(cfg.data.norm_stats)
    fingerprint = _norm_fingerprint(dataset)
    cached = _read_norm_cache(cache, fingerprint)
    if cached is not None:
        return cached

    total = np.zeros(3, dtype=np.float64)
    squared_total = np.zeros(3, dtype=np.float64)
    count = 0
    for image, _ in dataset:
        # Stream one image at a time. Pool pixel sums, not per-image standard
        # deviations, and weight images by their actual number of pixels.
        pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)
        total += pixels.sum(axis=0)
        squared_total += np.square(pixels).sum(axis=0)
        count += len(pixels)

    # Accumulating raw integer-valued pixels in float64 avoids early rounding.
    # Dividing here gives the same units as converting each pixel with /255.
    mean = total / count
    variance = np.maximum(squared_total / count - np.square(mean), 0)
    std = np.sqrt(variance) / 255.0
    if np.any(std == 0):
        raise DataError("training images have a zero-variance RGB channel; cannot normalise")
    stats = {"mean": (mean / 255.0).tolist(), "std": std.tolist()}

    # Do not publish statistics for files that changed during the scan.
    if _norm_fingerprint(dataset) != fingerprint:
        raise DataError("training images changed while computing normalisation statistics; retry")
    cache.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically so an interrupted write cannot leave a partial cache.
    # Close the temporary file before replace (required on Windows).
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=cache.parent, prefix=f".{cache.name}.",
        suffix=".tmp", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump({**stats, "_source": fingerprint}, handle, indent=2, allow_nan=False)
            handle.write("\n")
        except BaseException:
            handle.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, cache)
    finally:
        temporary.unlink(missing_ok=True)
    return stats


def _seed_worker(worker_id: int) -> None:
    """Extend PyTorch's per-worker seed to Python and NumPy on every platform."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return train, validation and test loaders from the committed manifest.

    All splits share training-derived statistics; augmentation and shuffling
    apply only to training. No samples are dropped, including the last batch.
    Images stay on CPU for the training/evaluation loop to move to its device.

    Each loader has its own seeded generator for sampling and worker seeds.
    As in run_experiment.py, call set_seed(cfg.seed) once before starting a run:
    with num_workers=0, augmentation uses the main process's torch RNG. Do not
    reseed here, since constructing loaders must not reset model randomness.
    """
    if type(cfg.data.batch_size) is not int or cfg.data.batch_size <= 0:
        raise DataError("data.batch_size must be a positive integer")
    if type(cfg.data.num_workers) is not int or cfg.data.num_workers < 0:
        raise DataError("data.num_workers must be a non-negative integer")

    stats = compute_norm_stats(cfg)
    loaders = []
    for offset, split in enumerate(("train", "val", "test")):
        dataset = _EuroSATDataset(cfg, split, transform=_build_transform(cfg, split, stats))
        # Separate generators keep validation/test iterator creation from
        # advancing the next training shuffle or training worker seeds.
        generator = torch.Generator().manual_seed(cfg.seed + offset)
        loaders.append(DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            shuffle=(split == "train"),
            drop_last=False,
            generator=generator,
            worker_init_fn=_seed_worker,
        ))
    return loaders[0], loaders[1], loaders[2]


def get_synthetic_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Random-tensor loaders for smoke mode - no download, no disk dataset.

    This exists so CI can prove on every pull request that the wiring from
    config to run directory is intact, in under a minute on a CPU runner. The
    labels are random, so nothing about accuracy here is meaningful; the point
    is purely that the pipeline executes and emits conformant artifacts.
    """
    size = cfg.data.input_size
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = []

    for split in ("train", "val", "test"):
        n = SMOKE_SIZES[split]
        images = torch.rand(n, 3, size, size, generator=generator)
        # Cycle through the classes so every split covers all 10 - a confusion
        # matrix with an absent class would fail schema validation for reasons
        # that have nothing to do with the code under test.
        labels = torch.arange(n, dtype=torch.long) % len(CLASS_NAMES)
        loaders.append(
            DataLoader(
                TensorDataset(images, labels),
                batch_size=cfg.data.batch_size,
                shuffle=(split == "train"),
                num_workers=0,  # spawning workers for 200 tensors costs more than it saves
            )
        )

    return tuple(loaders)  # type: ignore[return-value]
