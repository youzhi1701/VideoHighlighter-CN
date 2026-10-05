#!/usr/bin/env python3
"""Verify that applying the localization catalog recreates the checked-in CN files."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--localized-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--catalog", type=Path, required=True)
    args = p.parse_args()

    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    files = sorted({r["file"] for r in catalog["rules"]})
    files.append("packaging/installer/ChineseSimplified.isl")

    bad: list[str] = []
    for rel in files:
        a = args.localized_root / rel
        b = args.reference_root / rel
        if not a.exists() or not b.exists():
            bad.append(f"{rel}: missing after replay or in reference")
            continue
        if a.read_bytes() != b.read_bytes():
            bad.append(f"{rel}: replay differs from checked-in CN file")

    if bad:
        print("Localization replay mismatch:")
        for item in bad:
            print(" -", item)
        return 1

    print(f"Localization replay verified for {len(files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
