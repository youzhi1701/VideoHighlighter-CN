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

PLUGIN_DIR = Path(__file__).resolve().parent
ROOT = PLUGIN_DIR.parent
DEFAULT_CATALOG = PLUGIN_DIR / "内部" / "汉化规则.json"
ASSET_DIR = PLUGIN_DIR / "内部" / "资源"
DEFAULT_REPORT_DIR = PLUGIN_DIR / "内部" / "报告"


def _read_text(path: Path) -> tuple[str, bool]:
    raw = path.read_text(encoding="utf-8")
    return raw.replace("\r\n", "\n"), raw.endswith(("\n", "\r\n"))


def _write_text(path: Path, text: str, had_final_newline: bool) -> None:
    if had_final_newline and not text.endswith("\n"):
        text += "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _block_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Return occurrences that match complete diff lines/blocks, not substrings."""
    if not needle:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        j = i + len(needle)
        left_ok = i == 0 or text[i - 1] == "\n"
        right_ok = j == len(text) or text[j] == "\n"
        if left_ok and right_ok:
            spans.append((i, j))
        start = i + 1
    return spans


def _count(text: str, needle: str) -> int:
    return len(_block_spans(text, needle))


def _replace_once(text: str, needle: str, replacement: str) -> str:
    spans = _block_spans(text, needle)
    if not spans:
        return text
    i, j = spans[0]
    return text[:i] + replacement + text[j:]


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

    if source:
        # Always prefer an exact source occurrence over a target found
        # somewhere else in the file. Repeated labels often share the same
        # Chinese target; a global "target exists" check would incorrectly
        # skip later untranslated occurrences.
        occurrences = _count(text, source)
        if occurrences == 1:
            return _replace_once(text, source, target), "applied", meta
        if occurrences > 1:
            # The same short source line can legitimately appear several times
            # (for example "Cancel" or repeated labels). Use the diff context
            # captured with the rule to disambiguate before declaring conflict.
            contextual: list[tuple[str, str]] = []
            if before is not None and after is not None:
                contextual.append((
                    before + "\n" + source + "\n" + after,
                    before + "\n" + target + "\n" + after,
                ))
            if before is not None:
                contextual.append((
                    before + "\n" + source,
                    before + "\n" + target,
                ))
            if after is not None:
                contextual.append((
                    source + "\n" + after,
                    target + "\n" + after,
                ))
            for needle, replacement in contextual:
                if _count(text, needle) == 1:
                    return _replace_once(text, needle, replacement), "applied", meta
            if rule.get("safe_duplicate"):
                # Same file + same source + same target appears more than once.
                # Replacing the first remaining exact block is deterministic;
                # later rules consume the remaining occurrences in source order.
                return _replace_once(text, source, target), "applied", meta
            meta["occurrences"] = occurrences
            return text, "conflict", meta
        # Source is gone. Only now may an exact target block mean this
        # rule was already applied.
        if target and _count(text, target) > 0:
            return text, "already_applied", meta
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
            return _replace_once(text, anchor, repl), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta

    if before is not None:
        occurrences = _count(text, before)
        if occurrences == 1:
            return _replace_once(text, before, before + "\n" + target), "applied", meta
        if occurrences > 1:
            meta["occurrences"] = occurrences
            return text, "conflict", meta

    if after is not None:
        occurrences = _count(text, after)
        if occurrences == 1:
            return _replace_once(text, after, target + "\n" + after), "applied", meta
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



def _detect_target_kind(root: Path) -> str:
    """Classify the target before touching files."""
    source_markers = [
        root / "main.py",
        root / "frontend",
        root / "modules",
    ]
    exe_markers = list(root.glob("*.exe"))
    internal_markers = [
        root / "_internal",
        root / "internal",
    ]

    source_score = sum(1 for p in source_markers if p.exists())
    packaged = bool(exe_markers) and any(p.exists() for p in internal_markers)

    if source_score >= 2:
        return "source"
    if packaged:
        return "portable"
    if exe_markers and not (root / "main.py").exists():
        return "packaged"
    return "unknown"


def _print_target_error(root: Path, kind: str) -> None:
    print("")
    print("VideoHighlighter 中文汉化插件：目标目录检查失败")
    print(f"目标目录: {root.resolve()}")
    print("")
    if kind in {"portable", "packaged"}:
        print("检测到的是已经打包好的 Windows 便携版 / EXE。")
        print("当前“一键植入中文”是源码级汉化器，不能直接修改已冻结进 EXE 的界面文字。")
        print("未修改任何文件。")
        print("")
        print("正确做法：")
        print("1. 对 VideoHighlighter 官方源码执行汉化；")
        print("2. 汉化完成后重新打包生成中文便携版 / 中文安装包。")
    else:
        print("没有检测到完整的 VideoHighlighter 源码结构。")
        print("至少应包含 main.py、frontend/、modules/ 等源码内容。")
        print("未修改任何文件。")


def main() -> int:
    p = argparse.ArgumentParser(description="将 zh-CN 本地化层安全注入 VideoHighlighter 源码")
    p.add_argument("--root", type=Path, default=ROOT, help="目标源码根目录")
    p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    p.add_argument("--dry-run", action="store_true", help="只检查，不写文件")
    p.add_argument("--strict", action="store_true", help="出现 changed/conflict/missing 时返回非零状态")
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = p.parse_args()

    kind = _detect_target_kind(args.root)
    if kind != "source":
        _print_target_error(args.root, kind)
        return 3

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
    problems = [item for item in report.get("details", [])
                if item.get("status") in {"changed", "conflict", "missing", "missing_file"}]
    if problems:
        print("localization problems:")
        for item in problems[:40]:
            extra = ""
            if item.get("occurrences") is not None:
                extra = f" occurrences={item['occurrences']}"
            print(
                f"  {item.get('status')}: {item.get('file')} "
                f"source={item.get('source', '')[:160]!r}{extra}"
            )
        if len(problems) > 40:
            print(f"  ... 另外还有 {len(problems) - 40} 条，请查看报告。")
    bad = sum(s.get(k, 0) for k in ("changed", "conflict", "missing", "missing_file"))
    return 2 if args.strict and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
