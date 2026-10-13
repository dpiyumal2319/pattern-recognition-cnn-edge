#!/usr/bin/env python
"""Build the report PDF from report/main.tex in one command.

    python scripts/build_report.py            # -> report/main.pdf
    python scripts/build_report.py --clean    # remove LaTeX build artifacts

Runs latexmk, which reruns pdflatex until every cross-reference resolves.
Works the same on Windows (MiKTeX), Linux and macOS (TeX Live); CI compiles
the same file with xu-cheng/latex-action, so a report that builds here builds
there. The tables and figures it includes are not regenerated here - run
scripts/build_tables.py and scripts/build_figures.py first if results changed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_edge.paths import REPORT_DIR  # noqa: E402

MAIN_TEX = REPORT_DIR / "main.tex"


def latexmk_command(clean: bool = False) -> list[str]:
    """The latexmk invocation, run from inside report/.

    Running from report/ matters: main.tex and every section file use paths
    relative to it (tables/..., figures/...). -halt-on-error makes a broken
    \\input fail the build loudly instead of producing a half-built PDF.
    """
    if clean:
        return ["latexmk", "-c", MAIN_TEX.name]
    return ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", MAIN_TEX.name]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clean", action="store_true", help="remove build artifacts, keep main.pdf")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if shutil.which("latexmk") is None:
        print("[report] latexmk not found. Install a LaTeX distribution "
              "(MiKTeX on Windows, TeX Live on Linux/macOS) and retry.", file=sys.stderr)
        return 1

    result = subprocess.run(latexmk_command(args.clean), cwd=REPORT_DIR)
    if result.returncode != 0:
        print(f"[report] latexmk failed - see {REPORT_DIR / 'main.log'}", file=sys.stderr)
        return result.returncode
    if not args.clean:
        print(f"[report] built {REPORT_DIR / 'main.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
