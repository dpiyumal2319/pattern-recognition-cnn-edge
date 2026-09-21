"""EuroSAT download, split and loaders.

Three tiers, none of which touch the network:

1. The allocation and split arithmetic, over synthetic labels.
2. ``prepare_eurosat`` end to end against a fake ImageFolder tree. ``ImageFolder``
   enumerates by file extension without decoding, and ``EuroSAT._check_exists()``
   short-circuits ``download=True`` when ``2750/`` already exists - so an
   offline fixture exercises the real function rather than a mock.
3. Property tests over the committed ``data/splits.csv``. Once #2 merges, that
   file is in the repository, so these run on every future PR and permanently
   guard the partition every result in ``results/`` descends from.

#3 owns the three loader tests still skipped at the bottom.
See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from pr_edge.config import Config, DataConfig
from pr_edge.data import (
    CLASS_NAMES,
    EUROSAT_IMAGE_SUBDIR,
    EXPECTED_TOTAL,
    SPLIT_FRACTIONS,
    DataError,
    _allocate,
    _stratified_split,
    prepare_eurosat,
    read_manifest,
    validate_manifest,
)
from pr_edge.paths import DATA_DIR

# Deliberately awkward: 7 and 13 do not divide into 70/15/15, 100 does. If the
# allocator only ever saw EuroSAT's own class sizes it would never be tested,
# because 3000, 2500 and 2000 all divide exactly.
AWKWARD_SIZES = {0: 7, 1: 13, 2: 100}

FILES_PER_FAKE_CLASS = 20


def _labels(sizes: dict[int, int]) -> list[int]:
    return [label for label, count in sorted(sizes.items()) for _ in range(count)]


@pytest.fixture
def fake_eurosat(tmp_path: Path) -> Path:
    """A minimal EuroSAT tree: correct layout, empty files, no download."""
    image_root = tmp_path / EUROSAT_IMAGE_SUBDIR
    for name in CLASS_NAMES:
        class_dir = image_root / name
        class_dir.mkdir(parents=True)
        for i in range(FILES_PER_FAKE_CLASS):
            (class_dir / f"{name}_{i + 1}.jpg").touch()
    return tmp_path


def _config(root: Path, manifest: Path, seed: int = 42) -> Config:
    return Config(
        run_id="base",
        seed=seed,
        data=DataConfig(root=str(root), manifest=str(manifest)),
    )


# ---------------------------------------------------------------------------
# 1. The split arithmetic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 13, 100, 2000, 2500, 3000, 27000])
def test_allocation_always_sums_back_to_the_whole(n: int):
    """Largest remainder must not drop or duplicate an image."""
    assert sum(_allocate(n).values()) == n


@pytest.mark.parametrize("n,expected", [
    (2000, {"train": 1400, "val": 300, "test": 300}),
    (2500, {"train": 1750, "val": 375, "test": 375}),
    (3000, {"train": 2100, "val": 450, "test": 450}),
])
def test_allocation_is_exact_for_eurosat_class_sizes(n: int, expected: dict[str, int]):
    """EuroSAT's three distinct class sizes all divide exactly into 70/15/15."""
    assert _allocate(n) == expected


def test_split_is_seventy_fifteen_fifteen():
    """Totals must be 70/15/15 across the whole dataset."""
    labels = _labels({label: 1000 for label in range(10)})
    splits = _stratified_split(labels, seed=42)

    assert splits.count("train") == 7000
    assert splits.count("val") == 1500
    assert splits.count("test") == 1500


def test_split_is_stratified_per_class():
    """Each class must hold its 70/15/15 proportion, not just the total."""
    labels = _labels(AWKWARD_SIZES)
    splits = _stratified_split(labels, seed=42)

    for label, size in AWKWARD_SIZES.items():
        actual = {name: 0 for name in SPLIT_FRACTIONS}
        for value, split in zip(labels, splits):
            if value == label:
                actual[split] += 1
        assert actual == _allocate(size), f"class {label} is not stratified"


def test_split_is_deterministic_from_the_seed():
    """Re-running prepare_data must not reshuffle the manifest."""
    labels = _labels(AWKWARD_SIZES)

    assert _stratified_split(labels, seed=42) == _stratified_split(labels, seed=42)
    assert _stratified_split(labels, seed=42) != _stratified_split(labels, seed=7)


def test_split_is_independent_of_the_order_labels_arrive_in():
    """Directory order differs across platforms; the seed must be the only input."""
    labels = _labels(AWKWARD_SIZES)
    reference = _stratified_split(labels, seed=42)

    # Interleave the classes instead of grouping them. Each index keeps its own
    # label, so a correct implementation assigns each one the same split.
    order = sorted(range(len(labels)), key=lambda i: (i % 7, i))
    shuffled = [labels[i] for i in order]
    shuffled_splits = _stratified_split(shuffled, seed=42)

    per_class_reference = {label: sorted(
        split for value, split in zip(labels, reference) if value == label
    ) for label in AWKWARD_SIZES}
    per_class_shuffled = {label: sorted(
        split for value, split in zip(shuffled, shuffled_splits) if value == label
    ) for label in AWKWARD_SIZES}

    assert per_class_reference == per_class_shuffled


def test_changing_one_class_does_not_reshuffle_the_others():
    """Per-class generators mean a dataset change is locally contained."""
    base = _stratified_split(_labels({0: 50, 1: 50}), seed=42)
    grown = _stratified_split(_labels({0: 50, 1: 60}), seed=42)

    assert base[:50] == grown[:50]


# ---------------------------------------------------------------------------
# 2. prepare_eurosat, end to end and offline
# ---------------------------------------------------------------------------


def test_prepare_eurosat_writes_a_valid_manifest(fake_eurosat: Path):
    manifest_path = fake_eurosat / "splits.csv"
    written = prepare_eurosat(_config(fake_eurosat, manifest_path))

    assert written == manifest_path
    rows = read_manifest(written)
    validate_manifest(rows)
    assert len(rows) == len(CLASS_NAMES) * FILES_PER_FAKE_CLASS
    assert {row["label"] for row in rows} == set(CLASS_NAMES)


def test_manifest_covers_every_image_exactly_once(fake_eurosat: Path):
    """No image may appear in two splits - that would leak test data."""
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    paths = [row["filepath"] for row in rows]
    on_disk = sorted(
        p.relative_to(fake_eurosat / EUROSAT_IMAGE_SUBDIR).as_posix()
        for p in (fake_eurosat / EUROSAT_IMAGE_SUBDIR).rglob("*.jpg")
    )

    assert len(set(paths)) == len(paths)
    assert sorted(paths) == on_disk


def test_manifest_paths_are_relative_and_posix(fake_eurosat: Path):
    """Backslashes would make a manifest written on Windows unusable on Colab."""
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    for row in rows:
        assert "\\" not in row["filepath"]
        assert not Path(row["filepath"]).is_absolute()
        assert row["filepath"].startswith(f"{row['label']}/")


def test_manifest_is_written_with_unix_line_endings(fake_eurosat: Path):
    """A CRLF manifest on Windows would diff against every other machine's."""
    manifest_path = fake_eurosat / "splits.csv"
    prepare_eurosat(_config(fake_eurosat, manifest_path))

    assert b"\r\n" not in manifest_path.read_bytes()


def test_rerunning_does_not_change_the_manifest(fake_eurosat: Path):
    """Regenerating a split would silently invalidate every existing result."""
    manifest_path = fake_eurosat / "splits.csv"
    cfg = _config(fake_eurosat, manifest_path)

    prepare_eurosat(cfg)
    first = manifest_path.read_bytes()

    # A different seed on the second call must still be ignored: the existing
    # manifest is the partition, and only --force may replace it.
    prepare_eurosat(_config(fake_eurosat, manifest_path, seed=1234))
    assert manifest_path.read_bytes() == first

    prepare_eurosat(cfg, force=True)
    assert manifest_path.read_bytes() == first, "same seed must reproduce the same split"


def test_force_with_a_different_seed_regenerates(fake_eurosat: Path):
    manifest_path = fake_eurosat / "splits.csv"

    prepare_eurosat(_config(fake_eurosat, manifest_path))
    first = manifest_path.read_bytes()

    prepare_eurosat(_config(fake_eurosat, manifest_path, seed=1234), force=True)
    assert manifest_path.read_bytes() != first


def test_class_order_divergence_is_an_error(fake_eurosat: Path):
    """CLASS_NAMES is the only source of ordering; disagreement must be loud."""
    (fake_eurosat / EUROSAT_IMAGE_SUBDIR / "Zebra").mkdir()
    (fake_eurosat / EUROSAT_IMAGE_SUBDIR / "Zebra" / "Zebra_1.jpg").touch()

    with pytest.raises(DataError, match="CLASS_NAMES"):
        prepare_eurosat(_config(fake_eurosat, fake_eurosat / "splits.csv"))


def test_validate_manifest_rejects_a_duplicated_image(fake_eurosat: Path):
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    train = next(row for row in rows if row["split"] == "train")
    rows.append({**train, "split": "test"})

    with pytest.raises(DataError, match="duplicate"):
        validate_manifest(rows)


def test_validate_manifest_rejects_an_unstratified_split(fake_eurosat: Path):
    """The failure a totals-only check would miss: one class entirely in test."""
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    for row in rows:
        if row["label"] == CLASS_NAMES[0]:
            row["split"] = "test"

    with pytest.raises(DataError, match=CLASS_NAMES[0]):
        validate_manifest(rows)


# ---------------------------------------------------------------------------
# 3. The committed manifest
# ---------------------------------------------------------------------------

COMMITTED_MANIFEST = DATA_DIR / "splits.csv"
needs_manifest = pytest.mark.skipif(
    not COMMITTED_MANIFEST.exists(),
    reason="data/splits.csv not generated yet - run scripts/prepare_data.py",
)


@pytest.fixture(scope="module")
def committed_rows() -> list[dict[str, str]]:
    if not COMMITTED_MANIFEST.exists():
        pytest.skip("data/splits.csv not generated yet")
    with COMMITTED_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


@needs_manifest
def test_committed_manifest_is_the_whole_of_eurosat(committed_rows):
    assert len(committed_rows) == EXPECTED_TOTAL
    assert {row["label"] for row in committed_rows} == set(CLASS_NAMES)


@needs_manifest
def test_committed_manifest_is_valid_and_stratified(committed_rows):
    validate_manifest(committed_rows)


@needs_manifest
def test_committed_manifest_totals_are_seventy_fifteen_fifteen(committed_rows):
    counts = {name: 0 for name in SPLIT_FRACTIONS}
    for row in committed_rows:
        counts[row["split"]] += 1

    total = sum(counts.values())
    for split, fraction in SPLIT_FRACTIONS.items():
        # Within one image: exact here, but the tolerance is what keeps this
        # test honest if the dataset ever changes size.
        assert abs(counts[split] - fraction * total) <= 1, f"{split} is off target"


@needs_manifest
def test_committed_manifest_has_unix_line_endings():
    """Guards against a member with core.autocrlf=false recommitting CRLF."""
    assert b"\r\n" not in COMMITTED_MANIFEST.read_bytes()


# ---------------------------------------------------------------------------
# Owned by #3 - loaders, normalisation and augmentation
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="not implemented yet - owned by #3")
def test_norm_stats_use_the_training_split_only():
    """Computing them over the full dataset leaks test statistics."""


@pytest.mark.skip(reason="not implemented yet - owned by #3")
def test_augmentation_applies_to_train_only():
    """Val and test must see deterministic transforms."""


@pytest.mark.skip(reason="not implemented yet - owned by #3")
def test_input_size_switches_to_224_without_a_second_pipeline():
    """The SOTA harness (#19) needs this for the resolution ablation."""
