#!/usr/bin/env python3
"""Rebuild zh-CN patch catalog and translation memory from a known upstream ref.

Typical maintenance flow:
  1. update/sync upstream source,
  2. apply existing zh-CN layer,
  3. manually resolve genuinely new/changed UI strings,
  4. run this tool to make the new state replayable.

Only source/UI files are included. Localization infrastructure itself is never
folded back into the catalog.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
ROOT = PLUGIN_DIR.parent
CATALOG = PLUGIN_DIR / "数据" / "汉化规则.json"
MEMORY = PLUGIN_DIR / "数据" / "翻译记忆库.json"

INCLUDE_SUFFIXES = {".py", ".ts", ".tsx", ".js", ".jsx", ".iss"}
EXCLUDE_PREFIXES = (
    "localization/",
    ".github/",
    "tests/",
    "test/",
    "docs/",
)
CJK = re.compile(r"[\u3400-\u9fff]")
EN = re.compile(r"[A-Za-z]")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8")


def keep_file(path: str) -> bool:
    if path.startswith(EXCLUDE_PREFIXES):
        return False
    return Path(path).suffix.lower() in INCLUDE_SUFFIXES


def parse_patch(patch: str, file: str) -> list[dict]:
    lines = patch.splitlines()
    rules: list[dict] = []
    src: list[str] = []
    dst: list[str] = []
    before: str | None = None
    last_context: str | None = None
    hunk = ""

    def flush(after: str | None) -> None:
        nonlocal src, dst, before
        if not src and not dst:
            return
        rules.append({
            "id": f"{file}#{len(rules)+1}",
            "file": file,
            "source": "\n".join(src),
            "target": "\n".join(dst),
            "before": before,
            "after": after,
            "hunk": hunk,
        })
        src, dst, before = [], [], None

    for line in lines:
        if line.startswith("@@"):
            flush(None)
            hunk = line
            last_context = None
            continue
        if line.startswith(" "):
            ctx = line[1:]
            flush(ctx)
            last_context = ctx
            continue
        if line.startswith("-") and not line.startswith("---"):
            if not src and not dst:
                before = last_context
            src.append(line[1:])
            continue
        if line.startswith("+") and not line.startswith("+++"):
            if not src and not dst:
                before = last_context
            dst.append(line[1:])
            continue
    flush(None)
    return [r for r in rules if r["source"] != r["target"]]


def extract_tm(rules: list[dict]) -> list[dict]:
    entries: dict[tuple[str, str], set[str]] = defaultdict(set)
    quoted = re.compile(r"""(["'`])((?:\\.|(?!\1).)*?)\1""")
    jsx = re.compile(r">([^<>{}]+)<")

    def add(source: str, target: str, file: str) -> None:
        source, target = source.strip(), target.strip()
        if not source or not target or source == target:
            return
        if not EN.search(source) or not CJK.search(target):
            return
        entries[(source, target)].add(file)

    for rule in rules:
        a_lines = rule["source"].splitlines()
        b_lines = rule["target"].splitlines()
        if len(a_lines) != len(b_lines):
            continue
        for a, b in zip(a_lines, b_lines):
            qa = [m.group(2) for m in quoted.finditer(a)]
            qb = [m.group(2) for m in quoted.finditer(b)]
            if len(qa) == len(qb):
                for sa, sb in zip(qa, qb):
                    add(sa, sb, rule["file"])
            ma, mb = jsx.search(a), jsx.search(b)
            if ma and mb:
                add(ma.group(1), mb.group(1), rule["file"])

    out = [
        {"source": s, "target": t, "files": sorted(files)}
        for (s, t), files in entries.items()
    ]
    out.sort(key=lambda x: (x["source"].casefold(), x["target"]))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--upstream-ref", required=True, help="官方基线，例如 upstream/main 或具体 commit")
    p.add_argument("--current-ref", default="HEAD")
    args = p.parse_args()

    baseline = git("rev-parse", args.upstream_ref).strip()
    current = git("rev-parse", args.current_ref).strip()
    names = git("diff", "--name-only", baseline, current).splitlines()
    files = [x for x in names if keep_file(x)]

    rules: list[dict] = []
    for file in files:
        patch = git("diff", "--unified=3", baseline, current, "--", file)
        rules.extend(parse_patch(patch, file))

    # Same source + same target in one file can be consumed deterministically
    # from top to bottom by the replay engine.
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rules:
        if r["source"]:
            groups[(r["file"], r["source"])].append(r)
    for rs in groups.values():
        if len(rs) > 1 and len({r["target"] for r in rs}) == 1:
            for r in rs:
                r["safe_duplicate"] = True

    catalog = {
        "schema_version": 1,
        "locale": "zh-CN",
        "upstream": {
            "repository": "Aseiel/VideoHighlighter",
            "baseline_commit": baseline,
        },
        "generated_from": {
            "repository": "youzhi1701/VideoHighlighter-CN",
            "head_commit": current,
        },
        "strategy": "file-scoped exact block replacement with contextual insertion anchors; never global replace",
        "rules": rules,
    }
    memory = {
        "schema_version": 1,
        "locale": "zh-CN",
        "entries": extract_tm(rules),
    }

    CATALOG.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    MEMORY.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"baseline: {baseline}")
    print(f"localized files: {len(files)}")
    print(f"rules: {len(rules)}")
    print(f"translation-memory entries: {len(memory['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
