#!/usr/bin/env python3
"""Generate a text file containing all source code paths and contents in banqi/.

Usage:
  python generate_banqi_source_dump.py [output_path]

Defaults to banqi_source_dump.txt in the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BANQI_DIR = ROOT / "banqi"
DEFAULT_OUTPUT = ROOT / "banqi_source_dump.txt"

# Treat these as source code extensions. Add more if needed.
SOURCE_EXTS = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}

EXCLUDE_DIRS = {"__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"}


def is_text_file(path: Path) -> bool:
    if path.suffix.lower() in SOURCE_EXTS:
        return True
    return False


def iter_source_files(base_dir: Path) -> list[Path]:
    files: list[Path] = []
    for root, dirs, filenames in os.walk(base_dir):
        # prune excluded dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in filenames:
            path = Path(root) / name
            if is_text_file(path):
                files.append(path)
    return sorted(files)


def main() -> None:
    output_path = Path(os.sys.argv[1]) if len(os.sys.argv) > 1 else DEFAULT_OUTPUT
    files = iter_source_files(BANQI_DIR)

    with output_path.open("w", encoding="utf-8") as f:
        for path in files:
            rel = path.relative_to(ROOT)
            f.write(f"=== {rel} ===\n")
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                # fallback: read as bytes and decode with replacement
                content = path.read_bytes().decode("utf-8", errors="replace")
            f.write(content)
            if not content.endswith("\n"):
                f.write("\n")
            f.write("\n")

    print(f"Wrote {len(files)} files to {output_path}")


if __name__ == "__main__":
    main()
