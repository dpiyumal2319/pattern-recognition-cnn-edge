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

#3 covers manifest-backed images, statistics and transforms below; the loader
# integration check remains skipped until its factory is implemented.
See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pr_edge.config import Config, DataConfig
from pr_edge.data import (
    CLASS_NAMES,
    EUROSAT_IMAGE_SUBDIR,
    EXPECTED_TOTAL,
    SPLIT_FRACTIONS,
    DataError,
    _EuroSATDataset,
    _RandomQuarterTurn,
    _allocate,
    _build_transform,
    _stratified_split,
    compute_norm_stats,
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
        data=DataConfig(
            root=str(root), manifest=str(manifest), norm_stats=str(root / "norm_stats.json"),
        ),
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


def test_split_is_a_property_of_the_files_not_the_directory_order(fake_eurosat: Path):
    """The guarantee that makes four machines agree.

    ``_stratified_split`` is positional - an item's split depends on its rank
    within its class. So the partition is only reproducible elsewhere if the
    ordering fed to it is derivable from the files themselves, which is what
    the sort in ``prepare_eurosat`` provides. This recomputes the mapping from
    nothing but the sorted filenames and the seed, and demands an exact match.
    """
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    by_class: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_class[row["label"]].append(row["filepath"])

    records = [
        (path, CLASS_NAMES.index(name))
        for name in CLASS_NAMES
        for path in sorted(by_class[name])
    ]
    records.sort(key=lambda record: (record[1], record[0]))
    expected = dict(
        zip(
            [path for path, _ in records],
            _stratified_split([label for _, label in records], seed=42),
        )
    )

    assert {row["filepath"]: row["split"] for row in rows} == expected


def test_manifest_rows_are_sorted(fake_eurosat: Path):
    """A stable row order keeps the committed file diffable."""
    manifest_path = fake_eurosat / "splits.csv"
    rows = read_manifest(prepare_eurosat(_config(fake_eurosat, manifest_path)))

    keys = [(CLASS_NAMES.index(row["label"]), row["filepath"]) for row in rows]
    assert keys == sorted(keys)


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


@pytest.fixture
def image_manifest(tmp_path: Path) -> Config:
    """Decodable images with known pixels and an explicit 70/15/15 manifest.

    Use nonconsecutive class indices in reverse order to catch accidental
    relabelling or directory-based sorting. PNG keeps pixel assertions exact.
    """
    rows = []
    for name in ("River", "Forest"):
        directory = tmp_path / EUROSAT_IMAGE_SUBDIR / name
        directory.mkdir(parents=True)
        for index in range(20):
            relative = f"{name}/{index}.png"
            Image.new("RGB", (8, 8), (index, CLASS_NAMES.index(name), 100)).save(
                tmp_path / EUROSAT_IMAGE_SUBDIR / relative
            )
            rows.append({
                "filepath": relative,
                "label": name,
                "split": "train" if index < 14 else "val" if index < 17 else "test",
            })
    # Files absent from the manifest must never enter a split implicitly.
    Image.new("RGB", (8, 8)).save(directory / "unlisted.png")
    manifest = tmp_path / "splits.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filepath", "label", "split"))
        writer.writeheader()
        writer.writerows(rows)
    return _config(tmp_path, manifest)


@pytest.mark.parametrize("split,indices", [
    ("train", range(14)), ("val", range(14, 17)), ("test", range(17, 20)),
])
def test_dataset_loads_exact_manifest_members_and_labels(image_manifest, split, indices):
    manifest = Path(image_manifest.data.manifest)
    original = manifest.read_bytes()
    dataset = _EuroSATDataset(image_manifest, split)
    expected = [
        (index, CLASS_NAMES.index(name))
        for name in ("River", "Forest")
        for index in indices
    ]
    assert len(dataset) == len(expected)
    for position, (pixel, class_index) in enumerate(expected):
        image, label = dataset[position]
        assert image.mode == "RGB"
        assert image.size == (8, 8)
        assert image.getpixel((0, 0)) == (pixel, class_index, 100)
        assert label == class_index
    assert manifest.read_bytes() == original


def test_dataset_resolves_config_paths_from_project_root(image_manifest, monkeypatch, tmp_path):
    from pr_edge import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = _config(Path("."), Path("splits.csv"))
    image, label = _EuroSATDataset(cfg, "val")[0]
    assert image.getpixel((0, 0)) == (14, CLASS_NAMES.index("River"), 100)
    assert label == CLASS_NAMES.index("River")


def test_dataset_converts_to_rgb_before_applying_transform(image_manifest):
    from torchvision.transforms import ToTensor

    path = Path(image_manifest.data.root) / EUROSAT_IMAGE_SUBDIR / "River/0.png"
    Image.new("L", (8, 8), 255).save(path)
    image, label = _EuroSATDataset(image_manifest, "train", transform=ToTensor())[0]
    assert tuple(image.shape) == (3, 8, 8)
    assert image.eq(1).all()
    assert label == CLASS_NAMES.index("River")


def test_dataset_rejects_duplicates_across_splits(image_manifest):
    manifest = Path(image_manifest.data.manifest)
    with manifest.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(("River/0.png", "River", "test"))
    with pytest.raises(DataError, match="duplicate"):
        _EuroSATDataset(image_manifest, "train")


def test_dataset_rejects_unknown_split(image_manifest):
    with pytest.raises(DataError, match="unknown dataset split"):
        _EuroSATDataset(image_manifest, "validation")


@pytest.mark.parametrize("contents", [None, b"not an image"])
def test_dataset_reports_missing_or_corrupt_images_on_access(image_manifest, contents):
    path = Path(image_manifest.data.root) / EUROSAT_IMAGE_SUBDIR / "River/0.png"
    if contents is None:
        path.unlink()
    else:
        path.write_bytes(contents)
    # Construction reads only the manifest; images are opened on demand.
    dataset = _EuroSATDataset(image_manifest, "train")
    with pytest.raises(DataError, match="River[/\\\\]0.png"):
        dataset[0]


@pytest.fixture
def norm_config(image_manifest: Config) -> Config:
    """Equal counts of two colours give hand-computable population statistics."""
    for index, (path, _) in enumerate(_EuroSATDataset(image_manifest, "train").samples):
        colour = (0, 64, 255) if index % 2 == 0 else (255, 192, 0)
        Image.new("RGB", (8, 8), colour).save(path)
    return image_manifest


def test_norm_stats_use_the_training_split_only(norm_config):
    # Holdout images need not even be decodable to compute training statistics.
    for split in ("val", "test"):
        for path, _ in _EuroSATDataset(norm_config, split).samples:
            path.write_bytes(b"held-out pixels must not be read")
    stats = compute_norm_stats(norm_config)
    assert set(stats) == {"mean", "std"}
    assert stats["mean"] == pytest.approx([0.5, 128 / 255, 0.5])
    assert stats["std"] == pytest.approx([0.5, 64 / 255, 0.5])


def test_norm_stats_weight_pixels_instead_of_images(norm_config):
    dataset = _EuroSATDataset(norm_config, "train")
    Image.new("RGB", (16, 8), (0, 64, 255)).save(dataset.samples[0][0])
    # The first colour now occupies 15 image-equivalents, the other 14.
    stats = compute_norm_stats(norm_config)
    assert stats["mean"] == pytest.approx([14 / 29, (15 * 64 + 14 * 192) / (29 * 255), 15 / 29])
    spread = np.sqrt(15 * 14) / 29
    assert stats["std"] == pytest.approx([spread, spread * 128 / 255, spread])


def test_norm_stats_cache_avoids_image_decoding_and_rewriting(norm_config, monkeypatch):
    first = compute_norm_stats(norm_config)
    cache = Path(norm_config.data.norm_stats)
    original = cache.read_bytes()
    modified = cache.stat().st_mtime_ns

    def no_decode(*args, **kwargs):
        pytest.fail("cache hit must not decode any image")

    monkeypatch.setattr(Image, "open", no_decode)
    cfg = replace(norm_config, seed=123, data=replace(
        norm_config.data, input_size=224, batch_size=1, augment=False,
    ))
    assert compute_norm_stats(cfg) == first
    assert cache.read_bytes() == original
    assert cache.stat().st_mtime_ns == modified


def test_norm_stats_refresh_after_training_image_changes(norm_config):
    before = compute_norm_stats(norm_config)
    path = _EuroSATDataset(norm_config, "train").samples[0][0]
    previous = path.stat()
    Image.new("RGB", (8, 8), (255, 192, 0)).save(path)
    # Explicit time change avoids depending on the filesystem clock resolution.
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000_000))
    after = compute_norm_stats(norm_config)
    assert after["mean"][0] == pytest.approx(15 / 28)
    assert after != before


def test_norm_stats_refresh_after_manifest_reassigns_training_images(norm_config):
    before = compute_norm_stats(norm_config)
    rows = read_manifest(norm_config.data.manifest)
    train = next(row for row in rows if row["label"] == "River" and row["split"] == "train")
    val = next(row for row in rows if row["label"] == "River" and row["split"] == "val")
    train["split"], val["split"] = "val", "train"
    with Path(norm_config.data.manifest).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("filepath", "label", "split"))
        writer.writeheader()
        writer.writerows(rows)
    assert compute_norm_stats(norm_config) != before


@pytest.mark.parametrize("bad_cache", [
    "{incomplete", "[]", '{"mean": [0, 0, 0], "std": [1, 1, 1]}',
])
def test_norm_stats_rebuild_malformed_or_unversioned_caches(norm_config, bad_cache):
    cache = Path(norm_config.data.norm_stats)
    cache.write_text(bad_cache, encoding="utf-8")
    assert compute_norm_stats(norm_config)["mean"] == pytest.approx([0.5, 128 / 255, 0.5])
    assert "_source" in json.loads(cache.read_text(encoding="utf-8"))


@pytest.mark.parametrize("field,value", [
    ("mean", [float("nan"), 0.5, 0.5]), ("mean", [True, 0.5, 0.5]),
    ("mean", [2, 0.5, 0.5]), ("std", [0, 0.5, 0.5]), ("std", [0.5]),
])
def test_norm_stats_rebuild_invalid_cached_values(norm_config, field, value):
    expected = compute_norm_stats(norm_config)
    cache = Path(norm_config.data.norm_stats)
    cached = json.loads(cache.read_text(encoding="utf-8"))
    cached[field] = value
    cache.write_text(json.dumps(cached), encoding="utf-8")
    assert compute_norm_stats(norm_config) == expected


def test_norm_stats_reject_zero_variance_without_writing_cache(norm_config):
    for path, _ in _EuroSATDataset(norm_config, "train").samples:
        Image.new("RGB", (8, 8), (64, 64, 64)).save(path)
    with pytest.raises(DataError, match="zero-variance"):
        compute_norm_stats(norm_config)
    assert not Path(norm_config.data.norm_stats).exists()


def test_norm_stats_resolve_nested_cache_relative_to_project(norm_config, tmp_path, monkeypatch):
    from pr_edge import paths

    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    cfg = replace(norm_config, data=replace(
        norm_config.data, root=".", manifest="splits.csv", norm_stats="cache/stats.json",
    ))
    compute_norm_stats(cfg)
    assert (tmp_path / "cache/stats.json").is_file()
    assert not (elsewhere / "cache").exists()


def test_prepare_data_cli_writes_and_reuses_norm_cache(norm_config, tmp_path):
    from pr_edge.config import dump_config
    from scripts.prepare_data import main

    config_file = tmp_path / "base.yaml"
    dump_config(norm_config, config_file)
    manifest = Path(norm_config.data.manifest)
    original = manifest.read_bytes()
    assert main(["--config", str(config_file)]) == 0
    cache = Path(norm_config.data.norm_stats)
    cached = cache.read_bytes()
    assert main(["--config", str(config_file)]) == 0
    assert cache.read_bytes() == cached
    assert manifest.read_bytes() == original


@pytest.fixture
def patterned_image() -> Image.Image:
    """An asymmetric image exposes rotations/flips that flat colours hide."""
    return Image.fromarray(np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3))


@pytest.fixture
def transform_config() -> Config:
    return Config(run_id="base", data=DataConfig(input_size=8))


IDENTITY_STATS = {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]}


def test_augmentation_applies_to_train_only(transform_config, patterned_image):
    train = _build_transform(transform_config, "train", IDENTITY_STATS)
    val = _build_transform(transform_config, "val", IDENTITY_STATS)
    test = _build_transform(transform_config, "test", IDENTITY_STATS)
    original = val(patterned_image)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        outputs = [train(patterned_image) for _ in range(24)]
        assert any(not torch.equal(output, original) for output in outputs)
        for _ in range(4):
            state = torch.get_rng_state().clone()
            assert torch.equal(val(patterned_image), original)
            assert torch.equal(test(patterned_image), original)
            assert torch.equal(torch.get_rng_state(), state)


def test_disabling_augmentation_matches_evaluation(transform_config, patterned_image):
    cfg = replace(transform_config, data=replace(transform_config.data, augment=False))
    train = _build_transform(cfg, "train", IDENTITY_STATS)
    expected = _build_transform(cfg, "val", IDENTITY_STATS)(patterned_image)
    state = torch.get_rng_state().clone()
    for _ in range(4):
        assert torch.equal(train(patterned_image), expected)
    assert torch.equal(torch.get_rng_state(), state)


@pytest.mark.parametrize("turns", [0, 1, 2, 3])
def test_quarter_turns_are_exact_pixel_permutations(patterned_image, monkeypatch, turns):
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(turns))
    before = np.array(patterned_image)
    actual = _RandomQuarterTurn()(patterned_image)
    np.testing.assert_array_equal(np.asarray(actual), np.rot90(before, turns))
    np.testing.assert_array_equal(np.asarray(patterned_image), before)


@pytest.mark.parametrize("horizontal,vertical", [(True, False), (False, True), (True, True)])
def test_training_flips_preserve_pixels(transform_config, patterned_image, monkeypatch, horizontal, vertical):
    # Force each flip independently and suppress rotation to check direction.
    draws = iter([0.0 if horizontal else 1.0, 0.0 if vertical else 1.0])
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor([next(draws)]))
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(0))
    result = _build_transform(transform_config, "train", IDENTITY_STATS)(patterned_image)
    expected = np.array(patterned_image)
    if horizontal:
        expected = np.flip(expected, axis=1)
    if vertical:
        expected = np.flip(expected, axis=0)
    np.testing.assert_allclose(result.numpy(), expected.transpose(2, 0, 1) / 255, atol=1e-7)


def test_augmentation_sequence_repeats_with_seed(transform_config, patterned_image):
    transform = _build_transform(transform_config, "train", IDENTITY_STATS)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = [transform(patterned_image) for _ in range(8)]
        torch.manual_seed(42)
        second = [transform(patterned_image) for _ in range(8)]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert any(not torch.equal(first[0], image) for image in first[1:])


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("size", [64, 224])
def test_transforms_resize_and_use_supplied_training_statistics(norm_config, split, size):
    stats = compute_norm_stats(norm_config)
    cfg = replace(norm_config, data=replace(norm_config.data, input_size=size))
    image = Image.new("RGB", (8, 8), (0, 64, 255))
    with torch.random.fork_rng(devices=[]):
        actual = _build_transform(cfg, split, stats)(image)
    assert actual.shape == (3, size, size)
    assert actual.dtype == torch.float32
    # For this training fixture: (colour / 255 - mean) / std = [-1, -1, 1].
    expected = torch.tensor([-1.0, -1.0, 1.0]).view(3, 1, 1).expand_as(actual)
    torch.testing.assert_close(actual, expected)


def test_transform_works_with_manifest_dataset(norm_config):
    stats = compute_norm_stats(norm_config)
    transform = _build_transform(norm_config, "val", stats)
    dataset = _EuroSATDataset(norm_config, "val", transform=transform)
    first, label = dataset[0]
    second, repeated_label = dataset[0]
    assert first.shape == (3, norm_config.data.input_size, norm_config.data.input_size)
    assert torch.equal(first, second)
    assert label == repeated_label == CLASS_NAMES.index("River")


def test_transform_is_picklable_for_windows_workers(transform_config, patterned_image):
    import pickle

    transform = _build_transform(transform_config, "train", IDENTITY_STATS)
    restored = pickle.loads(pickle.dumps(transform))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        expected = transform(patterned_image)
        torch.manual_seed(42)
        assert torch.equal(restored(patterned_image), expected)


@pytest.mark.parametrize("size", [0, -1, 64.5, True])
def test_transforms_reject_invalid_input_size(transform_config, size):
    cfg = replace(transform_config, data=replace(transform_config.data, input_size=size))
    with pytest.raises(DataError, match="input_size"):
        _build_transform(cfg, "train", IDENTITY_STATS)


def test_transforms_reject_unknown_split(transform_config):
    with pytest.raises(DataError, match="unknown transform split"):
        _build_transform(transform_config, "validation", IDENTITY_STATS)


@pytest.mark.skip(reason="not implemented yet - owned by #3")
def test_input_size_switches_to_224_without_a_second_pipeline():
    """The SOTA harness (#19) needs this for the resolution ablation."""
