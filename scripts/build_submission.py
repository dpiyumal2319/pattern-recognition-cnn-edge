#!/usr/bin/env python
"""Generate source notebooks and the separate PDF/code submission deliverables.

No training, downloads or PDF compilation occurs here. Build report/main.pdf
first, then run this script. Only allowlisted project files enter the ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "submission"
STEM = "9_A03_EN3150"

SETUP = '''from pathlib import Path
import os
import subprocess
import sys

# Extract the entire ZIP first. This also works from notebooks/submission in Git.
start = Path.cwd().resolve()
candidates = [start, *start.parents]
PROJECT = next((base for parent in candidates
                for base in (parent / "raw_code", parent)
                if (base / "src/pr_edge").is_dir()
                and (base / "configs/base.yaml").is_file()), None)
if PROJECT is None:
    raise FileNotFoundError("Extract the complete code ZIP, including raw_code, first.")
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT))

# Enable once in a fresh Python 3.10+ / Colab environment (requires internet).
INSTALL_DEPENDENCIES = False
if INSTALL_DEPENDENCIES:
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)

def run_script(script, *args):
    subprocess.run([sys.executable, str(PROJECT / script), *map(str, args)],
                   cwd=PROJECT, check=True)

print("Project:", PROJECT)
'''

SPECS = [
    ("01_data_preparation", "Data preparation and shared configuration", "Task 1",
     ["src/pr_edge/__init__.py", "src/pr_edge/paths.py", "src/pr_edge/config.py",
      "src/pr_edge/device.py", "src/pr_edge/data.py", "scripts/prepare_data.py"],
     '''import csv
import json
from collections import Counter

with open("data/splits.csv", newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
print("Recorded split counts:", dict(Counter(row["split"] for row in rows)))
print("Training-only normalization:", json.loads(Path("data/norm_stats.json").read_text()))

# EuroSAT images are not shipped. Enable to download/prepare them once.
PREPARE_DATA = False
if PREPARE_DATA:
    run_script("scripts/prepare_data.py", "--config", "configs/base.yaml")
''',
     "EuroSAT is natively 64 x 64. The committed manifest preserves the same "
     "70/15/15 partition for every model. Preparation downloads images only when enabled."),
    ("02_custom_architectures", "Custom CNN architectures and parameter counts", "Task 2",
     ["src/pr_edge/models/model_a.py", "src/pr_edge/models/model_b.py",
      "src/pr_edge/models/_smoke.py", "src/pr_edge/models/__init__.py"],
     '''import torch
from pr_edge.config import load_config
from pr_edge.models import build_model
from pr_edge.device import set_seed

for name in ("model_a", "model_b"):
    cfg = load_config(PROJECT / "configs" / f"{name}.yaml")
    set_seed(cfg.seed)
    model = build_model(cfg).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 64))
    count = sum(parameter.numel() for parameter in model.parameters())
    print(name, "parameters:", count, "output shape:", tuple(output.shape))
    if name == "model_b":
        assert count <= 100_000
''',
     "The source cells expose the standard convolutions, depthwise/pointwise blocks, "
     "activation functions and pooling heads used in the report. The final cell "
     "constructs the actual architectures, without training or downloads."),
    ("03_training_and_optimizers", "Training, optimizer comparison and momentum", "Tasks 3 and 4",
     ["src/pr_edge/train.py", "scripts/run_experiment.py",
      "scripts/run_optimizer_sweep.py", "scripts/run_momentum_ablation.py"],
     '''# Offline integration check: synthetic data and a tiny stand-in network.
# These smoke results are NOT the reported architecture accuracies.
run_script("scripts/run_experiment.py", "--config", "configs/model_b.yaml", "--smoke")

# Real runs require prepared EuroSAT and may take hours. Prefer a GPU runtime.
# Prefix outputs so the supplied benchmark records remain intact.
RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    for name in ("model_a", "model_b", "model_b_sgd", "model_b_sgdm",
                 "model_a_100", "model_b_100"):
        run_script("scripts/run_experiment.py", "--config", f"configs/{name}.yaml",
                   "--device", "auto", "--run-id", f"rerun_{name}")
''',
     "The training loop saves the lowest-validation-loss checkpoint. Source also "
     "includes the controlled Adam/SGD/SGD+momentum and momentum-ablation drivers. "
     "The default workflow only runs a two-epoch offline smoke check; opt in to "
     "full experiments after notebook 01 prepares the images. Existing results "
     "under results/ are the report's recorded runs, not new notebook execution outputs."),
    ("04_pretrained_models", "MobileNetV2 and EfficientNet-B0", "Task 5",
     ["src/pr_edge/models/sota.py", "scripts/diagnose_frozen_head.py"],
     '''import json

names = ("mobilenetv2_64", "mobilenetv2_64_ft",
         "efficientnetb0_64", "efficientnetb0_64_ft",
         "mobilenetv2_224", "mobilenetv2_224_ft_b128",
         "efficientnetb0_224", "efficientnetb0_224_ft_b128")
for name in names:
    metrics = json.loads((PROJECT / "results" / name / "metrics.json").read_text())
    print(name, "recorded accuracy:", metrics["accuracy"])

# Section 5 also cites this supplementary fixed-feature diagnostic.
# Inspect the recorded three-seed summaries without recomputing features.
diagnostic = json.loads(Path("results/frozen_head_diagnostic/diagnostic.json").read_text())
print("Diagnostic protocol:", diagnostic["protocol"])
for name, result in diagnostic["runs"].items():
    print(name, "recorded feature statistics:", result["features"])
    for head in result["heads"]:
        print("lr:", head["lr"], "standardised:", head["standardised"],
              "mean validation-loss SD:", head["val_loss_sd_mean"])

# Requires the data from notebook 01 and internet for ImageNet weights.
# Each config reproduces its independent initialization and training policy.
RUN_FULL_TRAINING = False
if RUN_FULL_TRAINING:
    for name in names:
        run_script("scripts/run_experiment.py", "--config", f"configs/{name}.yaml",
                   "--device", "auto", "--run-id", f"rerun_{name}")

# Optional diagnostic rerun: fixed, unaugmented features and head-only training.
# This differs from the main protocol and does not isolate activation causality.
RUN_DIAGNOSTIC = False
if RUN_DIAGNOSTIC:
    run_script("scripts/diagnose_frozen_head.py", "--device", "auto",
               "--out", "results/rerun_frozen_head_diagnostic/diagnostic.json")
''',
     "Primary comparisons use 64 x 64; 224 x 224 runs are supplementary. Frozen "
     "and fine-tuned policies are defined by their YAML configs. Saved weights "
     "and the EuroSAT images are excluded from the submission. Full GPU runs "
     "are optional and are not executed by default. The supplementary diagnostic "
     "source and recorded feature/head summaries support Section 5's discussion; "
     "its separate rerun is also opt-in."),
    ("05_evaluation_and_report", "Evaluation, profiling and report artifacts", "Tasks 4 and 6",
     ["src/pr_edge/evaluate.py", "src/pr_edge/profile.py", "src/pr_edge/schemas.py",
      "src/pr_edge/plots.py", "src/pr_edge/tables.py", "scripts/build_tables.py",
      "scripts/build_figures.py", "scripts/build_report.py"],
     '''import json
from IPython.display import Image, display
from pr_edge.schemas import validate_run_dir

for name in ("model_a", "model_a_100", "model_b", "model_b_100",
             "model_b_sgd", "model_b_sgdm", "mobilenetv2_64_ft", "efficientnetb0_64_ft"):
    folder = PROJECT / "results" / name
    errors = validate_run_dir(folder)
    assert not errors, (name, errors)
    metrics = json.loads((folder / "metrics.json").read_text())
    profile = json.loads((folder / "profile.json").read_text())
    print(name, "accuracy:", metrics["accuracy"],
          "parameters:", profile["total_params"], "MACs:", profile["macs"])

# Rebuild from the supplied recorded JSON; no training or dataset download.
run_script("scripts/build_tables.py", "--all")
run_script("scripts/build_figures.py", "--all")
display(Image(filename=str(PROJECT / "report/figures/custom_loss_curves.png")))
display(Image(filename=str(PROJECT / "report/figures/efficientnetb0_64_ft_confusion_matrix.png")))

# Optional PDF rebuild requires a local LaTeX installation and latexmk.
# The final PDF is supplied separately from this code ZIP.
BUILD_PDF = False
if BUILD_PDF:
    run_script("scripts/build_report.py")
''',
     "This notebook exposes the evaluation, MAC/parameter profiling, plotting and "
     "LaTeX table generators. Its workflow validates recorded artifacts and regenerates "
     "tables and figures. It reads the supplied canonical results; fresh rerun_* "
     "experiments are kept separate until deliberately selected for a new comparison."),
]


def cell(kind: str, source: str, index: int) -> dict:
    """Create stable notebook-v4 cells with no saved outputs."""
    value = {"cell_type": kind, "metadata": {},
             "id": hashlib.sha256(f"{index}:{source}".encode()).hexdigest()[:12],
             "source": source.splitlines(keepends=True)}
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    return value


def generate_notebooks(root: Path = ROOT, destination: Path = NOTEBOOK_DIR) -> list[Path]:
    """Copy reviewed source verbatim into runnable Jupyter writefile cells."""
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for slug, title, task, sources, workflow, explanation in SPECS:
        contents = [("markdown", f"# {title}\n\nEN3150 Assignment 03 - Group 9. {task}.\n\n{explanation}\n\n"
                     "## Running this notebook\n\nUse Python 3.10+ with the project dependencies. "
                     "Extract the whole ZIP and open this notebook beside raw_code/. "
                     "On Colab, upload and extract the complete ZIP before running. "
                     "Set INSTALL_DEPENDENCIES = True in the setup cell for a fresh runtime. "
                     "Run cells in order with a fresh kernel for each notebook.\n\n"
                     "The writefile cells below contain the original Python source, including "
                     "comments. They save that code into the bundled package so relative imports "
                     "and CLI entry points behave exactly as in the repository. "
                     "The complete package is already included: the notebooks can be run independently.\n"),
                    ("code", SETUP)]
        for source in sources:
            text = (root / source).read_text(encoding="utf-8")
            contents.extend([("markdown", f"## Source: `{source}`\n"),
                             ("code", f"%%writefile {source}\n{text}")])
        contents.extend([("markdown", "## Run / inspect results\n"), ("code", workflow)])
        notebook = {
            "cells": [cell(kind, source, index) for index, (kind, source) in enumerate(contents)],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.10"}},
            "nbformat": 4, "nbformat_minor": 5,
        }
        path = destination / f"{slug}.ipynb"
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def submission_files(root: Path) -> list[Path]:
    """Allowlist sources and reproducibility inputs; never traverse local caches."""
    files = []
    allowed = {"src": {".py"}, "scripts": {".py"}, "tests": {".py"},
               "configs": {".yaml"}, "results": {".json", ".yaml", ".png"},
               "report": {".tex", ".bib", ".sty", ".png"}}
    for directory, suffixes in allowed.items():
        for path in sorted((root / directory).rglob("*")):
            parts = path.relative_to(root).parts
            if (path.is_file() and path.suffix in suffixes
                    and not any(part.startswith(("_smoke_", "__pycache__")) for part in parts)
                    and path.name.lower() not in {"agents.md", "claude.md"}):
                files.append(path)
    for name in ("requirements.txt", "pyproject.toml", "README.md", "LICENSE",
                 "data/splits.csv", "data/norm_stats.json", "docs/interfaces.md", "docs/course-scope.md"):
        if (root / name).is_file():
            files.append(root / name)
    return files


PACKAGE_README = '''# EN3150 Assignment 03 - Group 9 code submission

The report PDF is submitted separately. Extract this entire ZIP before use.
Open the five top-level notebooks in numeric order, or run each independently
with a fresh Python kernel. Each contains the original commented Python source
plus a workflow. Source cells write the corresponding files in raw_code/.
The additional colab_run.ipynb is the original optional Drive-backed runner;
it clones the public repository and is not needed for the five task notebooks.

Python 3.10+ is required. For local Jupyter, install with:
    python -m pip install -r raw_code/requirements.txt jupyterlab
    python -m jupyterlab
On Google Colab, upload this ZIP, extract it into /content, open a notebook,
and enable INSTALL_DEPENDENCIES in its setup cell. A GPU is recommended for
full experiments. Notebook 01 prepares EuroSAT when PREPARE_DATA is enabled.

Default workflows inspect recorded results, construct the actual custom models,
run a tiny offline smoke test, and regenerate report tables/figures. Full
training is opt-in and writes rerun_* directories rather than overwriting the
recorded benchmarks. The smoke test uses synthetic data and a small stand-in
network, and does not reproduce the report's accuracies.

raw_code/ includes the Python package, CLI scripts, tests, YAML configs,
recorded JSON/figures, split manifest, normalization statistics and LaTeX
report sources. Notebook 04 also includes the frozen-head diagnostic cited
in Section 5, inspects its recorded summaries, and offers an opt-in rerun.

No dataset images, pretrained downloads, checkpoints, environment, Git history,
AGENTS.md or CLAUDE.md are included. EuroSAT and ImageNet weights download only
when the relevant full preparation/training workflow is enabled. Recorded
results are preserved as evidence; notebook outputs are deliberately cleared.

To run the original CLI directly:
    cd raw_code
    python scripts/run_experiment.py --config configs/model_b.yaml --smoke
    python -m pytest

Project: https://github.com/dpiyumal2319/pattern-recognition-cnn-edge
EuroSAT dataset: https://github.com/phelber/EuroSAT
'''


def build_package(root: Path, notebooks: list[Path], pdf: Path, output: Path) -> tuple[Path, Path]:
    """Package notebooks at ZIP root with supporting files under raw_code/."""
    if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
        raise ValueError(f"Build a valid final report PDF first: {pdf}")
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f"{STEM}_code.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("README.md", PACKAGE_README)
        for notebook in notebooks:
            bundle.write(notebook, notebook.name)
        original_colab = root / "notebooks/colab_run.ipynb"
        if original_colab.is_file():
            bundle.write(original_colab, original_colab.name)
        for path in submission_files(root):
            bundle.write(path, "raw_code/" + path.relative_to(root).as_posix())
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("Submission archive failed its CRC check")
        if any(Path(name).name.lower() in {"agents.md", "claude.md"} for name in bundle.namelist()):
            raise ValueError("Contributor instructions must not enter the submission")
    final_pdf = output / f"{STEM}.pdf"
    shutil.copy2(pdf, final_pdf)
    return final_pdf, archive


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebooks-only", action="store_true", help="refresh committed notebooks without packaging")
    parser.add_argument("--pdf", type=Path, default=ROOT / "report/main.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "submission")
    args = parser.parse_args(argv)
    notebooks = generate_notebooks()
    for path in notebooks:
        print(f"[notebook] {path}")
    if not args.notebooks_only:
        for path in build_package(ROOT, notebooks, args.pdf, args.output):
            print(f"[submission] {path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
