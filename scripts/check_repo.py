# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0
# Copyright (c) 2026 Theoros
"""Repository hygiene checks run in CI.

* Every Python and YAML file under version control starts with the SPDX
  licence line and the copyright line.
* No tracked text file contains en dashes, em dashes or emoji.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = (
    "# SPDX-License-Identifier: LicenseRef-Theoros-Open-1.0",
    "# Copyright (c) 2026 Theoros",
)
HEADER_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".sh", ".def"}
BINARY_SUFFIXES = {".png", ".jpg", ".pdf", ".onnx", ".pt", ".blob", ".bin"}
DASHES = {chr(0x2013): "en dash", chr(0x2014): "em dash"}
EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),
    (0x2600, 0x27BF),
    (0x2B00, 0x2BFF),
    (0xFE00, 0xFE0F),
)


def _tracked() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return [ROOT / name for name in out.split("\0") if name]


def _header_ok(lines: list[str]) -> bool:
    start = 1 if lines and lines[0].startswith("#!") else 0
    return tuple(lines[start : start + 2]) == HEADER


def _is_emoji(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in EMOJI_RANGES)


def main() -> int:
    """Run all checks and return 0 when the repository is clean."""
    problems: list[str] = []
    for path in _tracked():
        if path.suffix in BINARY_SUFFIXES or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        if path.suffix in HEADER_SUFFIXES and text.strip() and not _header_ok(text.splitlines()):
            problems.append(f"{relative}: missing SPDX and copyright header")
        for number, line in enumerate(text.splitlines(), start=1):
            for char, name in DASHES.items():
                if char in line:
                    problems.append(f"{relative}:{number}: {name}")
            if any(_is_emoji(char) for char in line):
                problems.append(f"{relative}:{number}: emoji")
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
