"""Static checks for the canonical Colab driver added by issue #7.

Google authentication and a GPU runtime cannot be exercised in unit tests,
but the committed notebook can still be held to its important repository
contracts: valid Python, no outputs or secrets, CLI-only orchestration, and a
Drive-backed EuroSAT cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from pr_edge.paths import PROJECT_ROOT

NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "colab_run.ipynb"
if not NOTEBOOK_PATH.is_file():
    # The submission ZIP places notebooks beside raw_code/ for direct upload.
    NOTEBOOK_PATH = PROJECT_ROOT.parent / "colab_run.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))


def _code(notebook: dict) -> str:
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )


def test_notebook_is_valid_v4_and_has_no_saved_outputs(notebook):
    assert notebook["nbformat"] == 4
    assert notebook["cells"]
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []


def test_every_code_cell_is_plain_valid_python(notebook):
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")


def test_notebook_is_a_cli_driver_not_a_second_training_implementation(notebook):
    code = _code(notebook)
    assert "scripts/prepare_data.py" in code
    assert "scripts/run_experiment.py" in code
    for forbidden in ("nn.Module", "DataLoader(", ".backward(", "optimizer.step("):
        assert forbidden not in code


def test_notebook_caches_eurosat_and_results_in_drive(notebook):
    code = _code(notebook)
    assert 'drive.mount("/content/drive")' in code
    assert 'DRIVE_ROOT / "data" / "eurosat"' in code
    assert 'image_root = REPO_EUROSAT / "2750"' in code
    assert 'cached_archive = CACHE_EUROSAT / "EuroSAT.zip"' in code
    assert "shutil.copy2(cached_archive, local_archive)" in code
    assert 'EuroSAT(root=str(REPO_DIR / "data"), download=True)' in code
    assert 'extract_archive(str(local_archive), str(REPO_EUROSAT))' in code
    assert "shutil.copy2(local_archive, cached_archive)" in code
    assert 'DRIVE_ROOT / "results" / run_name' in code


def test_notebook_reads_a_secret_without_embedding_a_github_token(notebook):
    code = _code(notebook)
    assert 'userdata.get("GITHUB_TOKEN")' in code
    assert "http.extraHeader=AUTHORIZATION" in code
    assert not re.search(r"gh[pousr]_[A-Za-z0-9_]{20,}", code)
