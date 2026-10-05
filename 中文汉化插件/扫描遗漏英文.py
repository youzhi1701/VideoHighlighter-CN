#!/usr/bin/env python3
"""Find likely user-visible English that is not yet Chinese-localized.

This is deliberately conservative: it reports candidates instead of modifying
source. False positives can be suppressed in localization/zh_CN/allowlist.json.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent
ROOT = PLUGIN_DIR.parent
ALLOW = PLUGIN_DIR / "数据" / "保留英文白名单.json"
REPORT = PLUGIN_DIR / "报告" / "遗漏英文扫描.json"

UI_CALLS = {
    "QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QGroupBox", "QAction", "QMenu",
    "setWindowTitle", "setText", "setToolTip", "setStatusTip", "setPlaceholderText",
    "addTab", "addAction", "addMenu", "showMessage", "information", "warning", "critical",
    "question", "getText", "getItem", "getOpenFileName", "getSaveFileName",
}
TECH_RE = re.compile(r"^(?:VideoHighlighter|AI|CPU|GPU|CUDA|CLIP|ONNX|OpenVINO|FFmpeg|GGUF|Ollama|PyTorch|YOLOX|AGPLv3|FCPXML|EDL|CSV|JSON|HTTP|HTTPS|NVIDIA|Intel|AMD)(?:\b|$)")
ENGLISH_RE = re.compile(r"[A-Za-z]{2,}")
CJK_RE = re.compile(r"[\u3400-\u9fff]")


@dataclass
class Hit:
    file: str
    line: int
    kind: str
    text: str


def call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def string_value(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                parts.append("{…}")
        return "".join(parts)
    return None


def visible_candidate(text: str, allow: set[str]) -> bool:
    t = " ".join(text.split())
    if not t or t in allow or CJK_RE.search(t) or not ENGLISH_RE.search(t):
        return False
    # JSX/source-code fragments occasionally look like prose to a regex.
    if any(tok in t for tok in ("=>", "===", "!==", "&&", "||", "?.", "??", "Number.isFinite", "Math.", "return ")):
        return False
    if TECH_RE.match(t) and len(t.split()) <= 3:
        return False
    if re.fullmatch(r"[A-Za-z0-9_.:/{}<>+*=-]+", t):
        return False
    return True


def scan_python(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    hits: list[Hit] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return hits
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = call_name(node.func)
        if name not in UI_CALLS:
            continue
        for arg in node.args[:3]:
            value = string_value(arg)
            if value and visible_candidate(value, allow):
                hits.append(Hit(rel, getattr(arg, "lineno", getattr(node, "lineno", 0)), "python-ui", value))
    return hits


def scan_tsx(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[Hit] = []
    patterns = [
        ("tsx-text", re.compile(r">([^<>{}\n]*[A-Za-z][^<>{}\n]*)<")),
        ("tsx-attr", re.compile(r"\b(?:title|placeholder|aria-label|alt)=([\"'])(.*?)\1")),
    ]
    for kind, pattern in patterns:
        for m in pattern.finditer(text):
            value = m.group(1) if kind == "tsx-text" else m.group(2)
            if visible_candidate(value, allow):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(Hit(rel, line, kind, value.strip()))
    return hits


def scan_iss(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[Hit] = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(";"):
            continue
        if not any(key in line for key in ("Description:", "MsgBox(", "CreateDownloadPage(", "ItemCaption", "SuppressibleMsgBox(")):
            continue
        for m in re.finditer(r"['\"]([^'\"]*[A-Za-z][^'\"]*)['\"]", line):
            value = m.group(1)
            if visible_candidate(value, allow):
                hits.append(Hit(rel, i, "installer-ui", value))
    return hits


def main() -> int:
    p = argparse.ArgumentParser(description="扫描可能遗漏的用户可见英文")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--report", type=Path, default=REPORT)
    args = p.parse_args()

    allow_data = json.loads(ALLOW.read_text(encoding="utf-8")) if ALLOW.exists() else {"exact": []}
    allow = set(allow_data.get("exact", []))
    hits: list[Hit] = []

    for path in args.root.rglob("*.py"):
        rel_parts = path.relative_to(args.root).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        if any(part in {"venv", "dist", "build", "node_modules", "tests", "test", "docs", "localization"} for part in rel_parts):
            continue
        rel = path.relative_to(args.root).as_posix()
        hits.extend(scan_python(path, rel, allow))

    front = args.root / "frontend" / "src"
    if front.exists():
        for path in front.rglob("*"):
            if path.suffix.lower() not in {".ts", ".tsx", ".js", ".jsx"}:
                continue
            rel_parts = path.relative_to(args.root).parts
            if any(part.startswith(".") for part in rel_parts):
                continue
            if any(part in {"tests", "test", "node_modules", "dist", "build"} for part in rel_parts):
                continue
            rel = path.relative_to(args.root).as_posix()
            hits.extend(scan_tsx(path, rel, allow))

    installer = args.root / "packaging" / "installer"
    if installer.exists():
        for path in installer.glob("*.iss"):
            hits.extend(scan_iss(path, path.relative_to(args.root).as_posix(), allow))

    # Stable, de-duplicated output.
    uniq = {(h.file, h.line, h.kind, h.text): h for h in hits}
    hits = [uniq[k] for k in sorted(uniq)]
    payload = {"count": len(hits), "hits": [asdict(h) for h in hits]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"untranslated candidates: {len(hits)}")
    print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
