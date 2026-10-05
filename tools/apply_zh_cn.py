#!/usr/bin/env python3
"""Replay the Simplified-Chinese localization layer onto an upstream checkout.

Design goals:
- file-scoped rules only; never perform repository-wide string replacement;
- exact block matching first, contextual insertions second;
- never silently guess when upstream changed;
- classify every rule as applied/already_applied/changed/conflict/missing;
- keep a machine-readable and human-readable report.

The catalog is generated from the known-good CN tree and can be regenerated
without changing this engine.
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "localization" / "zh_CN" / "patches.json"
ASSET_DIR = ROOT / "localization" / "zh_CN" / "assets"
DEFAULT_REPORT_DIR = ROOT / "localization" / "reports"


def _read_text(path: Path) -> tuple[str, bool]:
    raw = path.read_text(encoding="utf-8")
    return raw.replace("\r\n", "\n"), raw.endswith(("\n", "\r\n"))


def _write_text(path: Path, text: str, had_final_newline: bool) -> None:
    if had_final_newline and not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _count(text: str, needle: str) -> int:
    return 0 if not needle else text.count(needle)


def _best_similarity(source: str, text: str) -> dict[str, Any] | None:
    if not source or not text:
        return None
    src_lines = source.splitlines()
    hay = text.splitlines()
    width = max(1, len(src_lines))
    # For very large files/rules, sample around lines sharing the first
    # non-empty token to avoid quadratic scans.
    first = next((ln.strip() for ln in src_lines if ln.strip()), "")
    candidates: list[int] = []
    if first:
        token = first[:80]
        candidates = [i for i, ln in enumerate(hay) if token[:24] in ln][:80]
    if not candidates:
        step = max(1, len(hay) // 80)
        candidates = list(range(0, max(1, len(hay)), step))[:80]

    best: tuple[float, str, int] | None = None
    for i in candidates:
        for delta in (-2, -1, 0, 1, 2):
            n = max(1, width + delta)
            chunk = "\n".join(hay[i:i+n])
            if not chunk:
                continue
            score = difflib.SequenceMatcher(None, source, chunk).ratio()
            if best is None or score > best[0]:
                best = (score, chunk, i + 1)
    if best is None:
        return None
    return {"score": round(best[0], 3), "line": best[2], "candidate": best[1][:800]}


def _apply_rule(text: str, rule: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
    source = rule.get("source", "")
    target = rule.get("target", "")
    before = rule.get("before")
    after = rule.get("after")
    meta: dict[str, Any] = {}

    if target and target in text:
        return text, "already_applied", meta

    if source:
        occurrences = _count(text, source)
        if occurrences == 1:
            return text.replace(source, target, 1), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta
        sim = _best_similarity(source, text)
        if sim and sim["score"] >= 0.72:
            meta["similar"] = sim
            return text, "changed", meta
        if sim:
            meta["similar"] = sim
        return text, "missing", meta

    # Pure insertion. Require a unique contextual anchor; never guess.
    if not target:
        return text, "already_applied", meta

    if before is not None and after is not None:
        anchor = before + "\n" + after
        occurrences = _count(text, anchor)
        if occurrences == 1:
            repl = before + "\n" + target + "\n" + after
            return text.replace(anchor, repl, 1), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta

    if before is not None:
        occurrences = _count(text, before)
        if occurrences == 1:
            return text.replace(before, before + "\n" + target, 1), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta

    if after is not None:
        occurrences = _count(text, after)
        if occurrences == 1:
            return text.replace(after, target + "\n" + after, 1), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta

    return text, "missing", meta


def _copy_assets(root: Path, dry_run: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not ASSET_DIR.exists():
        return results
    mapping = {
        "ChineseSimplified.isl": Path("packaging/installer/ChineseSimplified.isl"),
    }
    for name, rel in mapping.items():
        src = ASSET_DIR / name
        dst = root / rel
        if not src.exists():
            results.append({"file": str(rel), "status": "missing_asset"})
            continue
        wanted = src.read_bytes()
        if dst.exists() and dst.read_bytes() == wanted:
            results.append({"file": str(rel), "status": "already_applied"})
            continue
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        results.append({"file": str(rel), "status": "applied"})
    return results


def apply_catalog(root: Path, catalog_path: Path, dry_run: bool) -> dict[str, Any]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in catalog["rules"]:
        by_file[rule["file"]].append(rule)

    details: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for rel, rules in sorted(by_file.items()):
        path = root / rel
        if not path.exists():
            for rule in rules:
                entry = {"id": rule["id"], "file": rel, "status": "missing_file"}
                details.append(entry)
                counts["missing_file"] += 1
            continue

        text, final_nl = _read_text(path)
        changed_file = False
        for rule in rules:
            text2, status, meta = _apply_rule(text, rule)
            entry = {"id": rule["id"], "file": rel, "status": status, **meta}
            details.append(entry)
            counts[status] += 1
            if text2 != text:
                changed_file = True
                text = text2

        if changed_file and not dry_run:
            _write_text(path, text, final_nl)

    assets = _copy_assets(root, dry_run)
    for item in assets:
        counts["asset_" + item["status"]] += 1

    return {
        "schema_version": 1,
        "locale": catalog.get("locale", "zh-CN"),
        "upstream": catalog.get("upstream", {}),
        "root": str(root.resolve()),
        "dry_run": dry_run,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": dict(sorted(counts.items())),
        "assets": assets,
        "details": details,
    }


def write_report(report: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "apply_report.json"
    md_path = out_dir / "apply_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = report["summary"]
    lines = [
        "# VideoHighlighter 中文注入报告",
        "",
        f"- 时间：{report['generated_at']}",
        f"- 目标：`{report['root']}`",
        f"- 模式：{'仅检查' if report['dry_run'] else '已写入'}",
        "",
        "## 汇总",
        "",
    ]
    for key in ("applied", "already_applied", "changed", "conflict", "missing", "missing_file"):
        lines.append(f"- {key}: **{summary.get(key, 0)}**")

    problem = [x for x in report["details"] if x["status"] in {"changed", "conflict", "missing", "missing_file"}]
    if problem:
        lines += ["", "## 需要关注", ""]
        for item in problem[:200]:
            extra = ""
            if "similar" in item:
                s = item["similar"]
                extra = f" — 相似度 {s['score']}，约第 {s['line']} 行"
            lines.append(f"- `{item['file']}` · {item['status']} · `{item['id']}`{extra}")
    else:
        lines += ["", "没有发现冲突或失效规则。"]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> int:
    p = argparse.ArgumentParser(description="将 zh-CN 本地化层安全注入 VideoHighlighter 源码")
    p.add_argument("--root", type=Path, default=ROOT, help="目标源码根目录")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--dry-run", action="store_true", help="只检查，不写文件")
    p.add_argument("--strict", action="store_true", help="出现 changed/conflict/missing 时返回非零状态")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = p.parse_args()

    report = apply_catalog(args.root, args.catalog, args.dry_run)
    jp, mp = write_report(report, args.report_dir)
    s = report["summary"]
    print(
        "zh-CN localization: "
        f"applied={s.get('applied', 0)}, "
        f"already={s.get('already_applied', 0)}, "
        f"changed={s.get('changed', 0)}, "
        f"conflict={s.get('conflict', 0)}, "
        f"missing={s.get('missing', 0)}, "
        f"missing_file={s.get('missing_file', 0)}"
    )
    print(f"report: {mp}")
    bad = sum(s.get(k, 0) for k in ("changed", "conflict", "missing", "missing_file"))
    return 2 if args.strict and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
