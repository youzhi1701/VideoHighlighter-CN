#!/usr/bin/env python3
"""Audit likely user-visible English across the whole VideoHighlighter source tree.

This scanner is intentionally broader than the first-pass localization scanner.
It never edits source. It classifies candidates so maintainers can translate
real UI text while leaving protocols, model IDs and internal values untouched.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

INTERNAL_DIR = Path(__file__).resolve().parent
PLUGIN_DIR = INTERNAL_DIR.parent
ROOT = PLUGIN_DIR.parent
ALLOW = INTERNAL_DIR / "保留英文白名单.json"
REPORT = INTERNAL_DIR / "报告" / "遗漏英文扫描.json"

CJK_RE = re.compile(r"[\u3400-\u9fff]")
EN_RE = re.compile(r"[A-Za-z]{2,}")
STYLE_RE = re.compile(
    r"(?:Q[A-Za-z0-9_]+(?:[:#][A-Za-z0-9_]+)?\s*\{|"
    r"(?:background(?:-color)?|color|border(?:-[a-z]+)?|padding|margin|"
    r"font(?:-[a-z]+)?|min-width|max-width|min-height|max-height|"
    r"selection-color|selection-background-color)\s*:)",
    re.I,
)
TECH_RE = re.compile(
    r"^(?:Video ?Highlighter|AI|CPU|GPU|CUDA|CLIP|ONNX|OpenVINO|FFmpeg|GGUF|Ollama|"
    r"PyTorch|YOLOX|AGPLv3|FCPXML|EDL|CSV|JSON|HTTP|HTTPS|NVIDIA|Intel|AMD|"
    r"Python|Qt|PySide6|Hugging ?Face|Whisper|SigLIP|RTMPose)(?:\b|$)",
    re.I,
)

TK_WIDGET_CALLS = {
    "Label", "Button", "Checkbutton", "Radiobutton", "LabelFrame",
    "Menubutton", "Menu",
}
TK_MESSAGE_CALLS = {
    "showerror", "showwarning", "showinfo", "askyesno", "askokcancel",
    "askretrycancel", "askquestion",
}
TK_FILE_CALLS = {"askopenfilename", "asksaveasfilename", "askdirectory"}


UI_CALLS = {
    "QLabel", "QPushButton", "QCheckBox", "QRadioButton", "QGroupBox", "QAction",
    "QMenu", "QTabWidget", "QMessageBox", "QToolButton", "QCommandLinkButton",
    "QListWidgetItem", "QTableWidgetItem", "QTreeWidgetItem", "QStandardItem",
    "setWindowTitle", "setText",
    "setToolTip", "setStatusTip", "setPlaceholderText", "setWhatsThis",
    "setAccessibleName", "setAccessibleDescription", "addTab", "insertTab",
    "addAction", "addMenu", "showMessage", "information", "warning", "critical",
    "question", "about", "getText", "getItem", "getOpenFileName",
    "getSaveFileName", "getExistingDirectory", "setHeaderLabels",
    "setHorizontalHeaderLabels", "setVerticalHeaderLabels", "setTitle",
    "setLabelText", "setCancelButtonText", "setOkButtonText", "setItemText",
    "setTabText", "setHeaderData", "addItems", "insertItems", "setPrefix",
    "setSuffix", "setSpecialValueText",
}

# Backend callbacks that feed the desktop UI's log/progress panes.  These used
# to be invisible to the audit, which is how whole runtime paths such as object
# detection, transcription, diarization and auto-segmentation could stay
# English even after every static button/label looked translated.
USER_TEXT_CALLBACKS = {
    "log_fn", "progress_fn", "status_fn", "message_fn", "detail_fn",
    "log", "progress", "progress_cb", "status_cb", "message_cb", "detail_cb",
}
UI_NAME_RE = re.compile(
    r"(?:text|title|label|button|btn|tooltip|tip|status|message|msg|caption|"
    r"description|desc|placeholder|prompt|heading|header|menu|action|empty|"
    r"warning|error|success|progress|help|hint|option|options|choice|choices|"
    r"item|items|column|columns|tab|tabs)$",
    re.I,
)
UI_KEY_RE = re.compile(
    r"^(?:text|title|label|buttonText|tooltip|tip|status|message|caption|"
    r"description|placeholder|prompt|heading|header|emptyText|help|hint)$",
    re.I,
)

EXCLUDE_PARTS = {
    ".git", ".github", ".venv", "venv", "dist", "build", "node_modules",
    "tests", "test", "docs", "中文汉化插件", "__pycache__",
}

# Developer-only probes/build helpers are not application localization surfaces.
# Their output is useful to maintainers, but counting it as untranslated desktop
# UI creates pressure to change protocol-ish diagnostics that users never see.
DEV_ONLY_PATHS = {
    "modules/system/smoke_test.py",
    "tools/simulate_macos_bundle.py",
    "tools/export_frame_encoder.py",
    "tools/scrape_listing_probe.py",
    "tools/strip_bundle.py",
}
DEV_ONLY_PREFIXES = (
    "tools/teach_lab/",
)


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    kind: str
    priority: str
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
        out: list[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out.append(v.value)
            else:
                out.append("{…}")
        return "".join(out)
    if isinstance(node, (ast.List, ast.Tuple)):
        vals = [string_value(x) for x in node.elts]
        vals = [x for x in vals if x]
        return " | ".join(vals) if vals else None
    return None


def clean(text: str) -> str:
    return " ".join(text.replace("\\n", " ").split())


def visible_candidate(text: str, allow: set[str]) -> bool:
    t = clean(text)
    # Template expressions are runtime values, not literal English UI text.
    # Without stripping them, strings such as `✖ ${e.message}` are falsely
    # classified as untranslated because the variable name contains letters.
    t = re.sub(r"\$\{[^{}]*\}", "{…}", t)
    # Rich-text markup is presentation structure, not untranslated wording.
    # Strip tags before testing for English so fragments such as </span><br>
    # do not become false positives merely because HTML tag names use letters.
    t = re.sub(r"<[^>]+>", " ", t)
    t = clean(t)
    if not t or t in allow or CJK_RE.search(t) or not EN_RE.search(t):
        return False
    if t.startswith(("http://", "https://")):
        return False
    if STYLE_RE.search(t):
        return False
    # Keep plain alphabetic words.  Single-word labels such as "Cancel",
    # "Save", "Search" and "Preview" are common UI text and were previously
    # filtered out here by mistake.  Only suppress identifier/path-like tokens
    # that contain structural punctuation or digits.
    if re.fullmatch(r"[A-Za-z0-9_.:/{}<>+*=@%#\\-]+", t):
        if re.search(r"[_.:/{}<>+*=@%#\\-]", t) or any(ch.isdigit() for ch in t):
            return False
    if t.lower() in {"true", "false", "none", "null", "utf8", "utf-8", "rb", "wb"}:
        return False
    # Inno Setup expressions and symbolic task IDs are implementation details,
    # even though they appear inside quoted script fragments.
    if re.fullmatch(r"\+\s*[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*\+", t):
        return False
    if t.lower() in {"desktopicon", "quicklaunchicon"}:
        return False
    if TECH_RE.match(t) and len(t.split()) <= 4:
        return False
    if any(tok in t for tok in ("=>", "===", "!==", "&&", "||", "?.", "??", "return ", "lambda ")):
        return False
    return True


def add(hits: list[Hit], rel: str, node: ast.AST, kind: str, priority: str, value: str | None, allow: set[str]) -> None:
    if value and visible_candidate(value, allow):
        hits.append(Hit(rel, getattr(node, "lineno", 0), kind, priority, clean(value)))


def assignment_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for x in node.elts:
            out.extend(assignment_names(x))
        return out
    return []


def scan_python(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    hits: list[Hit] = []
    try:
        source_text = path.read_text(encoding="utf-8")
        tree = ast.parse(source_text)
        parent_of: dict[int, ast.AST] = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_of[id(child)] = parent
    except SyntaxError as exc:
        # A broken localized string is more serious than untranslated text.
        # Never silently skip the file: surface the parser error and make CI fail.
        return [Hit(
            rel,
            int(exc.lineno or 0),
            "python-syntax-error",
            "critical",
            clean(f"{exc.msg}（列 {exc.offset or 0}）"),
        )]
    except UnicodeDecodeError as exc:
        return [Hit(
            rel,
            0,
            "python-decode-error",
            "critical",
            clean(str(exc)),
        )]

    tkinter_file = ("import tkinter" in source_text or "from tkinter" in source_text)

    docstring_nodes: set[int] = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if owner.body and isinstance(owner.body[0], ast.Expr):
                value = owner.body[0].value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    docstring_nodes.add(id(value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = call_name(node.func)
            if name in UI_CALLS:
                for arg in node.args[:6]:
                    add(hits, rel, arg, "python-ui-call", "high", string_value(arg), allow)
                for kw in node.keywords:
                    if kw.arg and UI_KEY_RE.search(kw.arg):
                        add(hits, rel, kw.value, "python-ui-keyword", "high", string_value(kw.value), allow)
            elif tkinter_file and name in TK_WIDGET_CALLS:
                for kw in node.keywords:
                    if kw.arg == "text":
                        add(hits, rel, kw.value, "python-tk-ui", "high",
                            string_value(kw.value), allow)
            elif tkinter_file and name in TK_MESSAGE_CALLS:
                for arg in node.args[:3]:
                    add(hits, rel, arg, "python-tk-dialog", "high",
                        string_value(arg), allow)
            elif tkinter_file and name in TK_FILE_CALLS:
                for kw in node.keywords:
                    if kw.arg in {"title", "filetypes"}:
                        add(hits, rel, kw.value, "python-tk-file-dialog", "high",
                            string_value(kw.value), allow)
            elif tkinter_file and name == "StringVar":
                for kw in node.keywords:
                    if kw.arg == "value":
                        add(hits, rel, kw.value, "python-tk-ui", "high",
                            string_value(kw.value), allow)
            elif tkinter_file and name in {"title", "set"}:
                for arg in node.args[:1]:
                    add(hits, rel, arg, "python-tk-runtime", "high",
                        string_value(arg), allow)
            elif tkinter_file and name in {"config", "configure"}:
                for kw in node.keywords:
                    if kw.arg == "text":
                        add(hits, rel, kw.value, "python-tk-runtime", "high",
                            string_value(kw.value), allow)
            elif name in USER_TEXT_CALLBACKS:
                # These callbacks are wired into the visible task/log panels.
                # Scan every string argument because progress_fn commonly uses
                # (current, total, task_name, detail), not text as arg 0.
                for arg in node.args:
                    add(hits, rel, arg, "python-user-runtime-callback", "high",
                        string_value(arg), allow)
                for kw in node.keywords:
                    add(hits, rel, kw.value, "python-user-runtime-callback",
                        "high", string_value(kw.value), allow)
            elif name == "print":
                # Standalone tools and worker processes still surface these
                # lines directly in the console/log pane.
                for arg in node.args:
                    add(hits, rel, arg, "python-console-text", "medium",
                        string_value(arg), allow)
            elif name in {"info", "warning", "error", "critical", "exception"}:
                # Logger output is visible in the desktop debug/log surfaces and
                # in packaged diagnostic consoles. Keep this medium priority to
                # avoid treating internal debug chatter like primary UI.
                for arg in node.args[:2]:
                    add(hits, rel, arg, "python-log-text", "medium",
                        string_value(arg), allow)
            elif name in {
                "ValueError", "RuntimeError", "FileNotFoundError",
                "PermissionError", "TimeoutError",
            }:
                # These exception messages are frequently propagated unchanged
                # through API responses into visible error banners/dialogs.
                for arg in node.args[:1]:
                    add(hits, rel, arg, "python-exception-text", "medium",
                        string_value(arg), allow)
            elif name == "ArgumentParser":
                for kw in node.keywords:
                    if kw.arg == "description":
                        add(hits, rel, kw.value, "python-cli-help", "medium",
                            string_value(kw.value), allow)
            elif name == "add_argument":
                for kw in node.keywords:
                    if kw.arg == "help":
                        add(hits, rel, kw.value, "python-cli-help", "medium",
                            string_value(kw.value), allow)

        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value_node = node.value
            if value_node is None:
                continue
            names = [n for t in targets for n in assignment_names(t)]
            if any(UI_NAME_RE.search(n) for n in names):
                add(hits, rel, value_node, "python-ui-assignment", "medium", string_value(value_node), allow)

        elif isinstance(node, ast.Dict):
            for key, val in zip(node.keys, node.values):
                k = string_value(key) if key else None
                if k and UI_KEY_RE.match(k):
                    add(hits, rel, val, "python-ui-dict", "medium", string_value(val), allow)

    # Deep pass: runtime/user-facing strings emitted indirectly from GUI files.
    # This catches status/error/progress signals and helper-returned text that
    # does not sit directly inside a QLabel/QPushButton constructor.
    gui_file = ("PySide6" in source_text or "PyQt" in source_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            # Only text-bearing setters belong in the deep runtime pass.
            # Generic set* previously treated setObjectName("splashCard"),
            # setProperty("vrMode", ...) and stylesheet identifiers as visible UI.
            runtime_text_setters = {
                "setText", "setTitle", "setWindowTitle", "setToolTip",
                "setStatusTip", "setPlaceholderText", "setWhatsThis",
                "setAccessibleName", "setAccessibleDescription", "setLabelText",
                "setCancelButtonText", "setOkButtonText", "setTabText",
                "setPrefix", "setSuffix", "setSpecialValueText",
            }
            if attr == "emit" or attr in runtime_text_setters:
                for arg in node.args[:4]:
                    value = string_value(arg)
                    if value and visible_candidate(value, allow):
                        priority = "high" if gui_file else "medium"
                        hits.append(Hit(rel, getattr(arg, "lineno", getattr(node, "lineno", 0)),
                                        "python-runtime-text", priority, clean(value)))
        elif gui_file and isinstance(node, (ast.Constant, ast.JoinedStr)):
            if id(node) in docstring_nodes:
                continue
            value = string_value(node)
            cleaned = clean(value or "")
            if value and len(cleaned) >= 4 and visible_candidate(value, allow):
                # Broad safety net for GUI-local strings. Bare identifier-like
                # tokens are usually internal protocol/status/font/dict values
                # (for example success, filepath, prev, Arial, encode). Real
                # one-word UI labels are already covered by UI_CALLS, setters,
                # UI assignments and runtime emit() handling above.
                if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", cleaned):
                    continue

                # Do not flag structural literals that participate in program
                # behaviour. Common examples are dict protocol values, enum/
                # status comparisons and keyword tables such as
                # _VISION_KEYWORDS. Translating these would change behaviour,
                # which the localization audit must actively prevent.
                parent = parent_of.get(id(node))
                if isinstance(parent, (ast.Dict, ast.Compare, ast.Subscript)):
                    continue

                # Prompt/instruction/task templates are model-control text, not
                # application UI. Translating them would change model behaviour,
                # which is explicitly outside localization scope. Direct UI
                # setters/calls are already captured before this broad pass.
                assign_parent = parent
                while isinstance(assign_parent, (ast.JoinedStr, ast.List, ast.Tuple, ast.Set)):
                    assign_parent = parent_of.get(id(assign_parent))
                prompt_names = []
                if isinstance(assign_parent, ast.Assign):
                    prompt_names = [n for t in assign_parent.targets for n in assignment_names(t)]
                elif isinstance(assign_parent, ast.AnnAssign):
                    prompt_names = assignment_names(assign_parent.target)
                if any(re.search(r"(?:prompt|system|instruction|template|task)$", n, re.I)
                       for n in prompt_names):
                    continue

                if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
                    grand = parent_of.get(id(parent))
                    targets = []
                    if isinstance(grand, ast.Assign):
                        targets = [n for t in grand.targets for n in assignment_names(t)]
                    elif isinstance(grand, ast.AnnAssign):
                        targets = assignment_names(grand.target)
                    if targets and not any(UI_NAME_RE.search(n) for n in targets):
                        continue

                # File globs/cache names and generated output filenames are
                # implementation data, not labels. They may contain English
                # suffixes but translating them would break file discovery.
                if re.search(r"(?:\*\.|\.(?:json|txt|srt|onnx|pth|gguf|xml|yaml|yml))(?:$|\W)", cleaned, re.I):
                    continue

                hits.append(Hit(rel, getattr(node, "lineno", 0),
                                "python-gui-string", "medium", cleaned))

    return hits


TSX_PATTERNS = [
    ("tsx-text", "high", re.compile(r">\s*([^<>{}\n]*[A-Za-z][^<>{}\n]*)\s*<")),
    ("tsx-expression-text", "high", re.compile(r">\s*\{\s*([\"'])([^\"']*[A-Za-z][^\"']*)\1\s*\}\s*<")),
    ("tsx-attr", "high", re.compile(r"\b(?:title|placeholder|aria-label|alt|label|tooltip|description)=([\"'])(.*?)\1", re.I)),
    ("tsx-object-ui", "medium", re.compile(r"\b(?:label|title|description|text|placeholder|tooltip|message|emptyText|help|hint|caption|name)\s*:\s*([\"'\x60])(.*?)\1", re.I | re.S)),
    ("tsx-template-ui", "medium", re.compile(r"\b(?:setStatus|setMessage|setError|setTitle|setLabel|setHint|appendLog|alert|confirm)\(\s*(\x60)(.*?)\1", re.I | re.S)),
]


def scan_ts(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[Hit] = []
    runtime_patterns = [
        ("tsx-toast", "high", re.compile(r"""\btoast\.(?:error|success|warning|info)\(\s*(["'])(.*?)\1""", re.I)),
        ("tsx-log", "high", re.compile(r"""\bappendLog\(\s*(["'`])(.*?)\1""", re.I)),
        ("tsx-runtime-call", "medium", re.compile(r"""\b(?:setStatus|setMessage|setError|setTitle|setLabel|setHint)\(\s*(["'])(.*?)\1""", re.I)),
        ("tsx-option", "medium", re.compile(r"""\b(?:name|label|description|help|hint)\s*:\s*(["'])(.*?)\1""", re.I)),
    ]
    for kind, priority, pattern in TSX_PATTERNS + runtime_patterns:
        for m in pattern.finditer(text):
            value = m.group(1) if kind == "tsx-text" else m.group(2)
            cleaned = clean(value)
            # JSX ternary/control-flow fragments can sit between tag delimiters
            # and superficially look like English text (e.g. ") : paused ? (").
            if kind == "tsx-text" and re.fullmatch(r"[()?:\sA-Za-z_$.-]+", cleaned):
                if any(tok in cleaned for tok in (" ? ", ") :", " : ", "? (")):
                    continue
            if visible_candidate(value, allow):
                hits.append(Hit(rel, text.count("\n", 0, m.start()) + 1, kind, priority, cleaned))
    return hits


ISS_HINTS = (
    "Description:", "MsgBox(", "SuppressibleMsgBox(", "CreateDownloadPage(",
    "ItemCaption", "Button", "Caption", "StatusLabel", "WelcomeLabel", "FinishedLabel",
)


QML_PATTERNS = [
    ("qml-text", "high", re.compile(r"""\b(?:text|title|placeholderText|toolTip|accessibleName|accessibleDescription)\s*:\s*(["'])(.*?)\1""", re.I)),
    ("qml-menu", "high", re.compile(r"""\b(?:label|name|description|message|hint)\s*:\s*(["'])(.*?)\1""", re.I)),
    ("qml-qstr", "high", re.compile(r"""\b(?:qsTr|qsTranslate)\(\s*(["'])(.*?)\1""", re.I)),
    ("qml-runtime", "medium", re.compile(r"""\b(?:showMessage|setText|setTitle)\(\s*(["'])(.*?)\1""", re.I)),
]


def scan_qml(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[Hit] = []
    for kind, priority, pattern in QML_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(2)
            if visible_candidate(value, allow):
                hits.append(Hit(
                    rel,
                    text.count("\n", 0, m.start()) + 1,
                    kind,
                    priority,
                    clean(value),
                ))
    return hits


def scan_iss(path: Path, rel: str, allow: set[str]) -> list[Hit]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits: list[Hit] = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith(";") or not any(k in line for k in ISS_HINTS):
            continue
        for m in re.finditer(r"['\"]([^'\"]*[A-Za-z][^'\"]*)['\"]", line):
            value = m.group(1)
            if visible_candidate(value, allow):
                hits.append(Hit(rel, i, "installer-ui", "high", clean(value)))
    return hits


def excluded(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root).as_posix()
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    if rel in DEV_ONLY_PATHS or any(rel.startswith(p) for p in DEV_ONLY_PREFIXES):
        return True
    return any(p in EXCLUDE_PARTS or p.startswith(".") for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="全量扫描可能遗漏的用户可见英文")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    allow_data = json.loads(ALLOW.read_text(encoding="utf-8")) if ALLOW.exists() else {"exact": []}
    allow = set(allow_data.get("exact", []))
    hits: list[Hit] = []

    for path in args.root.rglob("*.py"):
        if excluded(path, args.root):
            continue
        hits.extend(scan_python(path, path.relative_to(args.root).as_posix(), allow))

    front = args.root / "frontend" / "src"
    if front.exists():
        for path in front.rglob("*"):
            if path.suffix.lower() not in {".ts", ".tsx", ".js", ".jsx"} or excluded(path, args.root):
                continue
            hits.extend(scan_ts(path, path.relative_to(args.root).as_posix(), allow))

    for path in args.root.rglob("*.qml"):
        if excluded(path, args.root):
            continue
        hits.extend(scan_qml(path, path.relative_to(args.root).as_posix(), allow))

    installer = args.root / "packaging" / "installer"
    if installer.exists():
        for path in installer.glob("*.iss"):
            hits.extend(scan_iss(path, path.relative_to(args.root).as_posix(), allow))

    uniq = {(h.file, h.line, h.kind, h.priority, h.text): h for h in hits}
    hits = [uniq[k] for k in sorted(uniq)]
    counts = Counter(h.file for h in hits)
    priorities = Counter(h.priority for h in hits)
    syntax_errors = [h for h in hits if h.kind in {"python-syntax-error", "python-decode-error"}]

    payload = {
        "count": len(hits),
        "syntax_error_count": len(syntax_errors),
        "priority_counts": dict(priorities),
        "files_with_candidates": len(counts),
        "top_files": counts.most_common(30),
        "hits": [asdict(h) for h in hits],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"untranslated candidates: {len(hits)}")
    print(f"files with candidates: {len(counts)}")
    print(f"priority: {dict(priorities)}")
    print("top files:")
    for file, count in counts.most_common(30):
        print(f"  {count:4d}  {file}")
    print("high-priority samples:")
    for h in [x for x in hits if x.priority == "high"][:500]:
        print(f"  {h.file}:{h.line} [{h.kind}] {h.text}")
    print("medium-priority samples from top files:")
    for file, _count in counts.most_common(15):
        samples = [x for x in hits if x.file == file and x.priority == "medium"][:12]
        if not samples:
            continue
        print(f"  -- {file} --")
        for h in samples:
            print(f"  {h.file}:{h.line} [{h.kind}] {h.text}")
    print(f"report: {args.report}")
    if syntax_errors:
        print(f"Python syntax/decode errors: {len(syntax_errors)}")
        for h in syntax_errors:
            print(f"  {h.file}:{h.line} [{h.kind}] {h.text}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
