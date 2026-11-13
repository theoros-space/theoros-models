# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Enforce per-path line and branch coverage floors from pyproject.toml.

Reads the JSON report written by pytest --cov --cov-report=json and the
table [tool.theoros.coverage-floors]. Each file is assigned to the
longest matching path prefix; line and branch coverage are checked
separately for every prefix that matches at least one file.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _package_path(filename: str) -> str:
    parts = Path(filename).parts
    return "/".join(parts[parts.index("theoros_models") :])


def main() -> int:
    """Check the report and return 0 when every floor is met."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="?", default=ROOT / "coverage.json")
    args = parser.parse_args()

    floors: dict[str, float] = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"][
        "theoros"
    ]["coverage-floors"]
    files = json.loads(args.report.read_text())["files"]

    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for filename, data in files.items():
        path = _package_path(filename)
        prefix = max((p for p in floors if path.startswith(p)), key=len)
        summary = data["summary"]
        for key in (prefix, ""):
            total = totals[key]
            total[0] += summary["covered_lines"]
            total[1] += summary["num_statements"]
            total[2] += summary["covered_branches"]
            total[3] += summary["num_branches"]

    failed = False
    for prefix, (lines, statements, branches, num_branches) in sorted(totals.items()):
        line_pct = 100.0 * lines / statements if statements else 100.0
        branch_pct = 100.0 * branches / num_branches if num_branches else 100.0
        floor = floors[prefix]
        ok = line_pct >= floor and branch_pct >= floor
        failed |= not ok
        label = prefix or "theoros_models (all)"
        print(
            f"{'ok  ' if ok else 'FAIL'} {label:<40} line {line_pct:6.2f}  "
            f"branch {branch_pct:6.2f}  floor {floor:.0f}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
