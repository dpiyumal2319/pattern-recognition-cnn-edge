"""Submission layout, verbatim notebook sources and offline reproducibility."""

import json
import zipfile
from pathlib import Path

import pytest

from scripts import build_submission as submission


def test_notebooks_are_deterministic_valid_and_outputs_are_stripped(tmp_path):
    paths = submission.generate_notebooks(destination=tmp_path)
    original = [path.read_bytes() for path in paths]
    assert len(paths) == 5
    submission.generate_notebooks(destination=tmp_path)
    assert [path.read_bytes() for path in paths] == original
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert (notebook["nbformat"], notebook["nbformat_minor"]) == (4, 5)
        assert len({cell["id"] for cell in notebook["cells"]}) == len(notebook["cells"])
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell["outputs"] == []
                assert cell["execution_count"] is None
                source = "".join(cell["source"])
                if source.startswith("%%writefile "):
                    source = source.split("\n", 1)[1]
                compile(source, str(path), "exec")


def test_source_cells_are_verbatim_and_cover_the_entire_package(tmp_path):
    paths = submission.generate_notebooks(destination=tmp_path)
    exposed = set()
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            source = "".join(cell["source"])
            if source.startswith("%%writefile "):
                header, body = source.split("\n", 1)
                name = header.removeprefix("%%writefile ")
                assert body == (submission.ROOT / name).read_text(encoding="utf-8")
                exposed.add(name)
    assert {p.relative_to(submission.ROOT).as_posix()
            for p in (submission.ROOT / "src").rglob("*.py")} <= exposed


def test_committed_notebooks_match_generator(tmp_path):
    for generated in submission.generate_notebooks(destination=tmp_path):
        committed = submission.NOTEBOOK_DIR / generated.name
        if not committed.is_file():
            committed = submission.ROOT.parent / generated.name
        assert generated.read_bytes() == committed.read_bytes()


def test_sota_notebook_exposes_report_cited_diagnostic_without_rerunning_it(tmp_path):
    paths = submission.generate_notebooks(destination=tmp_path)
    notebook = json.loads(next(path for path in paths if path.name.startswith("04_")).read_text(encoding="utf-8"))
    code = "\n".join("".join(cell["source"]) for cell in notebook["cells"]
                     if cell["cell_type"] == "code")
    assert "%%writefile scripts/diagnose_frozen_head.py" in code
    assert 'Path("results/frozen_head_diagnostic/diagnostic.json").read_text()' in code
    assert 'head["val_loss_sd_mean"]' in code
    assert "RUN_DIAGNOSTIC = False" in code
    assert "results/rerun_frozen_head_diagnostic/diagnostic.json" in code


@pytest.mark.parametrize("relative", ["", "raw_code", "raw_code/src", "notebooks/submission"])
def test_setup_finds_source_from_extracted_or_repository_layout(tmp_path, monkeypatch, relative):
    # Both the extracted archive layout and Git checkout layout are supported.
    project = tmp_path / "raw_code" if relative.startswith("raw_code") or not relative else tmp_path
    (project / "src/pr_edge").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "configs/base.yaml").write_text("seed: 42\n")
    start = tmp_path / relative
    start.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(start)
    import sys
    monkeypatch.setattr(sys, "path", list(sys.path))
    namespace = {}
    exec(submission.SETUP, namespace)
    assert namespace["PROJECT"] == project
    assert Path.cwd() == project


def test_setup_reports_missing_extraction(tmp_path, monkeypatch):
    # Use a nonexistent layout; no dependency installation should be attempted.
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="Extract the complete"):
        exec(submission.SETUP, {})


def test_package_layout_exclusions_and_verbatim_pdf(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    inputs = ["src/pr_edge/example.py", "scripts/run.py", "tests/test_example.py",
              "configs/base.yaml", "results/model_a/metrics.json", "results/model_a/config.yaml",
              "results/model_a/figures/loss.png", "data/splits.csv", "data/norm_stats.json",
              "requirements.txt", "pyproject.toml", "report/sections/dataset.tex"]
    excluded = ["AGENTS.md", "CLAUDE.md", "src/AGENTS.md", "src/claude.md",
                "data/eurosat/image.jpg", "results/model_a/model.pt",
                "results/_smoke_model_a/metrics.json", "report/main.aux", "tmp/private.py",
                ".git/config", ".venv/secret.py"]
    for name in inputs + excluded:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    notebook = tmp_path / "01_example.ipynb"
    notebook.write_text('{"nbformat":4}')
    original_colab = root / "notebooks/colab_run.ipynb"
    original_colab.parent.mkdir()
    original_colab.write_text('{"nbformat":4}')
    pdf = tmp_path / "main.pdf"
    pdf.write_bytes(b"%PDF-1.7\nunchanged final report\n")
    final, archive = submission.build_package(root, [notebook], pdf, tmp_path / "submission")
    assert final.read_bytes() == pdf.read_bytes()
    assert final.name == "9_A03_EN3150.pdf"
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        names = set(bundle.namelist())
        assert names == {"README.md", notebook.name, "colab_run.ipynb",
                         *("raw_code/" + name for name in inputs)}
        for name in inputs:
            assert bundle.read("raw_code/" + name) == (root / name).read_bytes()


@pytest.mark.parametrize("content", [None, b"not a PDF"])
def test_package_requires_valid_pdf_before_creating_archive(tmp_path, content):
    pdf = tmp_path / "main.pdf"
    if content is not None:
        pdf.write_bytes(content)
    with pytest.raises(ValueError, match="valid final report PDF"):
        submission.build_package(tmp_path, [], pdf, tmp_path / "output")
    assert not (tmp_path / "output").exists()
