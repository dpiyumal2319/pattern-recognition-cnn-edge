"""Report skeleton wiring. Owned by #23.

CI's "report compiles" job builds the PDF itself. These tests catch the same
class of breakage in seconds and without a LaTeX install: a section file that
is never \\input, an \\input or \\includegraphics pointing at a file that does
not exist, or a generated-artifact hole that points outside tables/ and figures/.
"""

from __future__ import annotations

import re

import pytest

from pr_edge.paths import REPORT_DIR

MAIN = REPORT_DIR / "main.tex"
SECTIONS = REPORT_DIR / "sections"

# The stubs #23's acceptance criteria require, in report order.
REQUIRED_SECTIONS = [
    "dataset", "data_preparation", "architecture", "activations",
    "optimizer", "custom_results", "sota_results", "discussion",
]


def _source(path) -> str:
    """LaTeX source with comments removed (an unescaped % to end of line)."""
    text = path.read_text(encoding="utf-8")
    return re.sub(r"(?<!\\)%.*", "", text)


def _all_sources() -> dict[str, str]:
    files = {"main.tex": _source(MAIN)}
    files.update({f"sections/{p.name}": _source(p) for p in sorted(SECTIONS.glob("*.tex"))})
    return files


def _inputs(text: str) -> list[str]:
    return re.findall(r"\\input\{([^}#]+)\}", text)


def _graphics(text: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}#]+)\}", text)


def _generated(text: str) -> list[str]:
    tables = re.findall(r"\\generatedtable\{([^}#]+)\}", text)
    figures = re.findall(r"\\generatedfigure(?:\[[^\]]*\])?\{([^}#]+)\}", text)
    return tables + figures


def test_main_inputs_every_required_section_in_task_order():
    inputs = [p for p in _inputs(_source(MAIN)) if p.startswith("sections/")]
    assert [p.removeprefix("sections/") for p in inputs] == REQUIRED_SECTIONS


def test_no_section_file_is_orphaned():
    """A section nobody \\inputs silently never reaches the marker."""
    inputs = {p.removeprefix("sections/") for p in _inputs(_source(MAIN))}
    on_disk = {p.stem for p in SECTIONS.glob("*.tex")}
    assert on_disk <= inputs, f"not included by main.tex: {sorted(on_disk - inputs)}"


@pytest.mark.parametrize("name", list(_all_sources()))
def test_every_plain_reference_exists(name):
    """Plain \\input and \\includegraphics must resolve: they are not guarded.

    Paths are relative to report/, where latexmk runs. Anything that may not
    exist yet must go through \\generatedtable or \\generatedfigure instead.
    """
    text = _all_sources()[name]
    for target in _inputs(text):
        path = REPORT_DIR / target
        assert path.exists() or path.with_suffix(".tex").exists(), f"{name}: \\input{{{target}}}"
    for target in _graphics(text):
        assert (REPORT_DIR / target).exists(), f"{name}: \\includegraphics{{{target}}}"


def test_generated_holes_point_at_generator_output_directories():
    """Every pending hole names a path a generator writes, so it fills itself in."""
    holes = [path for text in _all_sources().values() for path in _generated(text)]
    assert holes, "expected the skeleton to declare its pending generated artifacts"
    for path in holes:
        assert (path.startswith("tables/") and path.endswith(".tex")) or (
            path.startswith("figures/") and path.endswith(".png")
        ), path


@pytest.mark.parametrize("name", list(_all_sources()))
def test_no_placeholder_or_pending_box_reaches_the_submitted_pdf(name):
    """#26: the submitted PDF carries no red placeholder or Pending box.

    The macros stay defined in main.tex; only their uses are forbidden. A
    \\generatedtable or \\generatedfigure renders a Pending box only when its
    file is missing, which test_every_generated_artifact_exists covers.
    """
    uses = [
        line for line in _all_sources()[name].splitlines()
        if re.search(r"\\(placeholder|pending)\{", line) and "\\newcommand" not in line
        and not line.lstrip().startswith("{\\pending")  # the macros' own bodies
    ]
    assert not uses, f"{name}: {uses}"


def test_every_generated_artifact_exists():
    """With every issue merged, no generated hole may fall back to Pending."""
    holes = [path for text in _all_sources().values() for path in _generated(text)]
    missing = [path for path in holes if not (REPORT_DIR / path).exists()]
    assert not missing, f"would render as Pending boxes: {missing}"


def test_title_page_names_the_group_members_and_repository():
    """The brief requires every member's name and index number, plus a GitHub link."""
    text = _source(MAIN)
    assert re.search(r"\\newcommand\{\\groupno\}\{\d+\}", text)
    title = text[text.index("\\begin{titlepage}"):text.index("\\end{titlepage}")]
    members = re.findall(r"^\s*(\S[^&\n]*?)\s*&\s*(\d{6}[A-Z])\s*\\\\", title, re.MULTILINE)
    assert len(members) == 4, members
    assert len({index for _, index in members}) == 4, "duplicate index number"
    assert "https://github.com/dpiyumal2319/pattern-recognition-cnn-edge" in text


def test_build_command_fails_loudly_and_runs_from_the_report_directory(monkeypatch):
    from scripts import build_report

    assert build_report.latexmk_command() == [
        "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex",
    ]
    assert build_report.latexmk_command(clean=True) == ["latexmk", "-c", "main.tex"]

    calls = []

    class Done:
        returncode = 0

    monkeypatch.setattr(build_report.shutil, "which", lambda _: "latexmk")
    monkeypatch.setattr(build_report.subprocess, "run",
                        lambda cmd, cwd: calls.append((cmd, cwd)) or Done())
    assert build_report.main([]) == 0
    assert calls == [(build_report.latexmk_command(), REPORT_DIR)]


def test_build_reports_a_missing_latex_install(monkeypatch, capsys):
    from scripts import build_report

    monkeypatch.setattr(build_report.shutil, "which", lambda _: None)
    assert build_report.main([]) == 1
    assert "latexmk not found" in capsys.readouterr().err


def test_build_propagates_a_latex_failure(monkeypatch):
    from scripts import build_report

    class Failed:
        returncode = 12

    monkeypatch.setattr(build_report.shutil, "which", lambda _: "latexmk")
    monkeypatch.setattr(build_report.subprocess, "run", lambda cmd, cwd: Failed())
    assert build_report.main([]) == 12
