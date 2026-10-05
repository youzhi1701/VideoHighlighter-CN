"""PySide6 UI: a publish wizard and a community model browser.

    from model_hub.gui import PublishWizard, ModelBrowserDialog
    PublishWizard(self, model_path=onnx, draft=draft).exec()   # after training
    ModelBrowserDialog(self).exec()

When the Train tab opens the wizard for a model it just trained, ``draft``
already holds every technical field — task, labels, input format, what the
model measured — and those rows stay hidden. The person says what the model
is, picks where it belongs, and confirms the checklist. That is the whole job.
"""
from __future__ import annotations

import re
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QWizard, QWizardPage,
)

from . import hub
from .manifest import (
    CATEGORY_RE, COMPLIANCE_ITEMS, COLOR_ORDERS, LAYOUTS, LICENSES, NORMALIZATIONS, SLUG_RE,
    TASKS, USABLE_TASKS, InputSpec, Manifest,
)
from .package import build_package

LICENSE_NAMES = {
    "apache-2.0": "Apache 2.0（推荐）",
    "mit": "MIT",
    "cc-by-4.0": "CC BY 4.0",
    "cc0-1.0": "CC0（无附加条件）",
}

LAYOUT_NAMES_ZH = {
    "NCHW": "NCHW（通道优先）",
    "NHWC": "NHWC（通道最后）",
}
COLOR_NAMES_ZH = {
    "RGB": "RGB（红绿蓝）",
    "BGR": "BGR（蓝绿红）",
}
NORMALIZATION_NAMES_ZH = {
    "0-1": "0-1（归一化）",
    "0-255": "0-255（原始像素范围）",
    "imagenet": "ImageNet 标准化",
    "minus1-1": "-1 到 1",
}
OUTPUT_FORMAT_NAMES_ZH = {
    "yolox": "YOLOX 检测输出",
    "logits": "Logits（原始分数）",
    "probabilities": "概率",
}


# ------------------------------------------------------------------ worker
class _Worker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            result = self._fn(*self._args, progress=self.progress.emit, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc) or traceback.format_exc())
        else:
            self.finished.emit(result)


def run_in_thread(owner: QObject, fn, *args, on_progress=None, on_done=None, on_error=None,
                  **kwargs) -> QThread:
    thread = QThread(owner)
    worker = _Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_done:
        worker.finished.connect(on_done)
    if on_error:
        worker.failed.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread._worker = worker  # keep a reference
    thread.start()
    return thread


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:64].strip("-")


def _known_categories(progress=None) -> list[str]:
    return hub.categories_in(hub.search_catalog(limit=500))


# ------------------------------------------------------------ wizard pages
class DetailsPage(QWizardPage):
    def __init__(self, model_path: str = "", draft: Manifest | None = None):
        super().__init__()
        self._draft = draft
        self.setTitle("描述你的模型")
        self.setSubTitle("说明模型能识别什么，以及它属于哪个分类。用户会按分类浏览社区模型。")

        self.model_path = QLineEdit(model_path)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse)
        self._model_row = self._hbox(self.model_path, browse)

        self.display_name = QLineEdit(placeholderText="模型识别内容，例如：直升机检测器")
        self.name = QLineEdit(placeholderText="例如：直升机检测器")
        self._name_edited = False
        self.name.textEdited.connect(lambda _t: setattr(self, "_name_edited", True))
        self.display_name.textChanged.connect(self._sync_name)
        self.description = QPlainTextEdit()
        self.description.setPlaceholderText("说明它识别什么，以及适合哪类视频素材。")
        self.description.setFixedHeight(70)

        self.category = QLineEdit(placeholderText="例如：动物/马")
        self.category.setToolTip("模型在社区中的分类路径：使用小写英文，最多三级。"
                                 "例如“体育/网球”对应 sports/tennis。")
        self.category_hint = QLabel("")
        self.category_hint.setStyleSheet("color:#999;")
        self.category.textChanged.connect(self._category_changed)

        self.author = QLineEdit(placeholderText="你的名称或昵称，将显示在模型页面")
        self.game = QLineEdit(placeholderText="可选，仅用于描述（不要使用 Logo）")
        self.content_type = QLineEdit(placeholderText="可选：游戏、体育、音乐视频等")
        self.license = QComboBox()
        for lic in LICENSES:
            self.license.addItem(LICENSE_NAMES.get(lic, lic), lic)

        # -- technical: filled in from training, shown only for a hand-picked file
        self.task = QComboBox()
        for key in sorted(USABLE_TASKS):
            self.task.addItem(TASK_NAMES_ZH.get(key, TASKS[key][0]), key)
        self.output_format = QComboBox()
        self.task.currentIndexChanged.connect(self._task_changed)
        self.labels = QLineEdit(placeholderText="例如：直升机、飞机")
        self.width_ = QSpinBox(minimum=16, maximum=2048, value=416)
        self.height_ = QSpinBox(minimum=16, maximum=2048, value=416)
        self.frames = QSpinBox(minimum=1, maximum=128, value=1)
        self.layout_ = QComboBox()
        for value in sorted(LAYOUTS):
            self.layout_.addItem(LAYOUT_NAMES_ZH.get(value, value), value)
        self.color = QComboBox()
        for value in sorted(COLOR_ORDERS):
            self.color.addItem(COLOR_NAMES_ZH.get(value, value), value)
        self.normalize = QComboBox()
        for value in sorted(NORMALIZATIONS):
            self.normalize.addItem(NORMALIZATION_NAMES_ZH.get(value, value), value)
        self.threshold = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.05, value=0.3)
        size = QHBoxLayout()
        for w, label in ((self.width_, "宽"), (self.height_, "高"), (self.frames, "帧数")):
            size.addWidget(QLabel(label))
            size.addWidget(w)
        size_box = QWidget()
        size_box.setLayout(size)

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)          # extra height below the form, not between rows
        self._form = form
        form.addRow("模型名称", self.display_name)
        form.addRow("描述", self.description)
        form.addRow("分类", self.category)
        form.addRow("", self.category_hint)
        form.addRow("作者", self.author)
        form.addRow("许可证", self.license)
        form.addRow("游戏", self.game)
        form.addRow("内容类型", self.content_type)

        self.measured = QLabel("")
        self.measured.setWordWrap(True)
        self.show_technical = QCheckBox("显示技术详情")
        form.addRow("", self.measured)
        form.addRow("", self.show_technical)
        self._technical_rows = []
        for label, widget in (("ONNX 模型", self._model_row), ("简称", self.name),
                              ("任务", self.task), ("输出格式", self.output_format),
                              ("标签（逗号分隔）", self.labels), ("输入尺寸", size_box),
                              ("布局 / 颜色 / 缩放",
                               self._hbox(self.layout_, self.color, self.normalize)),
                              ("默认置信度", self.threshold)):
            form.addRow(label, widget)
            self._technical_rows.append(widget)
        self.show_technical.toggled.connect(self._set_technical_visible)

        self._task_changed()
        if draft is not None:
            self._apply_draft(draft)
        self._set_technical_visible(draft is None)
        self.show_technical.setVisible(draft is not None)

        for w in (self.model_path, self.name, self.display_name, self.author, self.labels,
                  self.category):
            w.textChanged.connect(self.completeChanged)
        self.description.textChanged.connect(self.completeChanged)

        # Suggest categories people already use; typing a new one is fine too.
        run_in_thread(self, _known_categories, on_done=self._set_categories,
                      on_error=lambda _m: None)

    @staticmethod
    def _hbox(*widgets):
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            lay.addWidget(w)
        return box

    def _apply_draft(self, d: Manifest) -> None:
        idx = self.task.findData(d.task)
        if idx >= 0:
            self.task.setCurrentIndex(idx)
        self._task_changed()
        idx = self.output_format.findData(d.output_format)
        if idx >= 0:
            self.output_format.setCurrentIndex(idx)
        self.labels.setText(", ".join(d.labels))
        self.width_.setValue(d.input.width)
        self.height_.setValue(d.input.height)
        idx = self.layout_.findData(d.input.layout)
        if idx >= 0:
            self.layout_.setCurrentIndex(idx)
        idx = self.color.findData(d.input.color)
        if idx >= 0:
            self.color.setCurrentIndex(idx)
        idx = self.normalize.findData(d.input.normalize)
        if idx >= 0:
            self.normalize.setCurrentIndex(idx)
        self.threshold.setValue(d.confidence_threshold)
        if d.author:
            self.author.setText(d.author)
        found = d.found_sentence()
        self.measured.setText(
            f"识别内容：{', '.join(d.labels)}。"
            + (f" {found}——该结果会显示在模型页面。" if found else ""))

    def _set_technical_visible(self, visible: bool) -> None:
        for w in self._technical_rows:
            if hasattr(self._form, "setRowVisible"):      # Qt 6.4+: no gap left behind
                self._form.setRowVisible(w, visible)
            else:
                w.setVisible(visible)
                label = self._form.labelForField(w)
                if label is not None:
                    label.setVisible(visible)

    def _sync_name(self, text: str) -> None:
        if not self._name_edited:
            self.name.setText(slugify(text))

    def _category_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.category_hint.setText("使用 / 分隔的小写英文，例如 animals/horses")
        elif CATEGORY_RE.match(text):
            self.category_hint.setText(f"分类位置：{' › '.join(text.split('/'))}")
        else:
            self.category_hint.setText("请使用小写字母、数字和连字符，"
                                       "最多三级，并使用 / 分隔")

    def _set_categories(self, categories: list[str]) -> None:
        completer = QCompleter(categories, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.category.setCompleter(completer)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 ONNX 模型", "", "ONNX 模型 (*.onnx)")
        if path:
            self.model_path.setText(path)
            if not self.display_name.text():
                self.display_name.setText(Path(path).stem.replace("_", " "))

    def _task_changed(self):
        task = self.task.currentData()
        self.output_format.clear()
        for value in sorted(TASKS[task][1]):
            self.output_format.addItem(OUTPUT_FORMAT_NAMES_ZH.get(value, value), value)
        is_action = task == "action_recognition"
        self.frames.setEnabled(is_action)
        self.frames.setValue(16 if is_action else 1)

    def isComplete(self) -> bool:
        return (Path(self.model_path.text()).is_file()
                and bool(SLUG_RE.match(self.name.text()))
                and bool(CATEGORY_RE.match(self.category.text().strip()))
                and all(w.text().strip() for w in (self.display_name, self.author, self.labels))
                and bool(self.description.toPlainText().strip()))

    def manifest(self) -> Manifest:
        labels = [x.strip() for x in self.labels.text().split(",") if x.strip()]
        return Manifest(
            name=self.name.text().strip(),
            display_name=self.display_name.text().strip(),
            description=self.description.toPlainText().strip(),
            task=self.task.currentData(),
            labels=labels,
            author=self.author.text().strip(),
            license=self.license.currentData(),
            category=self.category.text().strip(),
            input=InputSpec(width=self.width_.value(), height=self.height_.value(),
                            layout=self.layout_.currentData(), color=self.color.currentData(),
                            normalize=self.normalize.currentData(), frames=self.frames.value()),
            output_format=self.output_format.currentData(),
            confidence_threshold=round(self.threshold.value(), 3),
            game=self.game.text().strip(),
            content_type=self.content_type.text().strip(),
            metrics=dict(self._draft.metrics) if self._draft is not None else {},
        )


TASK_NAMES_ZH = {
    "object_detection": "物体检测",
    "image_classification": "画面分类",
    "action_recognition": "动作识别",
}


COMPLIANCE_ZH = {
    "terms_checked": "我已检查所有用于训练的游戏或视频来源条款，确认其未禁止 AI/ML 训练。",
    "own_footage": "我只使用自己有权使用的素材进行训练（例如自己的录制内容）。",
    "no_training_data": "模型包中不包含视频片段、截图、音频或其他训练数据。",
    "non_generative": "该模型只用于检测或分类，不会生成内容。",
}


class ChecklistPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("分享前确认")
        self.setSubTitle("共享的模型将公开发布。请逐项确认；全部确认前无法继续分享。")
        lay = QVBoxLayout(self)
        self.boxes: dict[str, QCheckBox] = {}
        for key, text in COMPLIANCE_ITEMS.items():
            box = QCheckBox(COMPLIANCE_ZH.get(key, text))
            box.setStyleSheet("QCheckBox { padding: 4px 0; }")
            box.toggled.connect(self.completeChanged)
            lay.addWidget(box)
            self.boxes[key] = box
        note = QLabel(
            "只会分享模型，绝不会上传你的视频、视频帧或音频。部分发行商会在服务条款中"
            "禁止使用其游戏内容进行 AI 训练；如果不确定，请不要分享。"
            "此处仅为一般性说明，不构成法律建议。")
        note.setWordWrap(True)
        lay.addSpacing(8)
        lay.addWidget(note)
        lay.addStretch()

    def isComplete(self) -> bool:
        return all(b.isChecked() for b in self.boxes.values())

    def compliance(self) -> dict[str, bool]:
        return {k: b.isChecked() for k, b in self.boxes.items()}


class CheckPage(QWizardPage):
    def __init__(self, wizard: "PublishWizard"):
        super().__init__()
        self._wiz = wizard
        self.setTitle("检查模型")
        self.setSubTitle("VideoHighlighter 会生成模型包，并在你的 CPU 上执行一次测试推理。")
        self.output = QPlainTextEdit(readOnly=True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.output)
        self.package_dir: Path | None = None
        self._ok = False

    def initializePage(self):
        self._ok = False
        self.output.setPlainText("检查中…")
        manifest = self._wiz.details.manifest()
        manifest.compliance = self._wiz.checklist.compliance()
        self._tmp = tempfile.TemporaryDirectory(prefix="vh-package-")
        out = Path(self._tmp.name) / manifest.name
        try:
            self.package_dir, report = build_package(self._wiz.details.model_path.text(), manifest, out)
        except Exception as exc:  # noqa: BLE001
            self.output.setPlainText(f"✖ {exc}")
            self.completeChanged.emit()
            return
        self._ok = report.ok
        verdict = ("可以分享。" if report.ok
                   else "请先修复上方问题，然后返回重试。")
        self.output.setPlainText(report.text() + "\n\n" + verdict)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._ok


class PublishPage(QWizardPage):
    def __init__(self, wizard: "PublishWizard"):
        super().__init__()
        self._wiz = wizard
        self.setTitle("分享到 Hugging Face")
        self.setSubTitle("模型将保存到你自己的免费 Hugging Face 账号中，"
                         "并列入 VideoHighlighter 社区模型列表。")
        self.token = QLineEdit(echoMode=QLineEdit.Password)
        self.token.setPlaceholderText("hf_…（具有写入权限的令牌）")
        saved = hub.get_token()
        if saved:
            self.token.setPlaceholderText("将使用已保存的令牌")
        self.remember = QCheckBox("在系统凭据存储中记住令牌")
        self.remember.setChecked(True)
        get_token = QLabel('还没有账号？<a href="https://huggingface.co/join">创建账号</a>，'
                           '然后<a href="https://huggingface.co/settings/tokens">创建令牌</a>，'
                           '并授予写入权限。')
        get_token.setOpenExternalLinks(True)
        get_token.setWordWrap(True)
        self.repo = QLineEdit()
        self.publish_btn = QPushButton("分享")
        self.publish_btn.clicked.connect(self._publish)
        self.log = QPlainTextEdit(readOnly=True)
        self.url: str | None = None

        form = QFormLayout()
        form.addRow("访问令牌", self.token)
        form.addRow("", get_token)
        form.addRow("", self.remember)
        form.addRow("仓库名称", self.repo)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.publish_btn, alignment=Qt.AlignLeft)
        lay.addWidget(self.log)

    def initializePage(self):
        self.repo.setText(self._wiz.details.name.text())
        self.url = None
        self.log.clear()

    def _publish(self):
        token = self.token.text().strip() or hub.get_token()
        if not token:
            QMessageBox.warning(self, "需要令牌", "请粘贴具有写入权限的 Hugging Face 访问令牌。")
            return
        if self.token.text().strip() and self.remember.isChecked():
            if not hub.save_token(token):
                self.log.appendPlainText("⚠ 无法保存令牌，本次会话仍会使用。")
        self.publish_btn.setEnabled(False)
        run_in_thread(self, hub.publish, self._wiz.checkpage.package_dir,
                      repo_name=self.repo.text().strip() or None, token=token,
                      on_progress=self.log.appendPlainText,
                      on_done=self._done, on_error=self._error)

    def _done(self, url: str):
        self.url = url
        self.log.appendPlainText(f"\n已分享：{url}")
        self.log.appendPlainText("几分钟后即可在社区模型中看到它，感谢分享！")
        self.completeChanged.emit()

    def _error(self, message: str):
        self.log.appendPlainText(f"\n✖ {message}")
        self.publish_btn.setEnabled(True)

    def isComplete(self) -> bool:
        return self.url is not None


class PublishWizard(QWizard):
    def __init__(self, parent=None, model_path: str = "", draft: Manifest | None = None):
        super().__init__(parent)
        self.setWindowTitle("分享模型")
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(720, 640)
        self.details = DetailsPage(model_path=model_path, draft=draft)
        self.checklist = ChecklistPage()
        self.checkpage = CheckPage(self)
        self.publishpage = PublishPage(self)
        for page in (self.details, self.checklist, self.checkpage, self.publishpage):
            self.addPage(page)
        self.setButtonText(QWizard.FinishButton, "打开模型页面")
        self.finished.connect(self._open_page)

    def _open_page(self, result: int):
        if result == QDialog.Accepted and self.publishpage.url:
            QDesktopServices.openUrl(QUrl(self.publishpage.url))


# ------------------------------------------------------------------ browser
class ModelBrowserDialog(QDialog):
    """搜索、安装和移除社区模型。"""

    COLUMNS = ["模型", "分类", "任务", "状态", "下载量", "更新时间"]
    installed = Signal(object)          # InstalledModel, so the host can refresh its model list

    def __init__(self, parent=None, models_dir: str | Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("社区模型")
        self.resize(900, 540)
        self.models_dir = models_dir
        self.entries: list[hub.CatalogEntry] = []
        self._all: list[hub.CatalogEntry] = []

        self.query = QLineEdit(placeholderText="搜索模型，例如“直升机”")
        self.query.returnPressed.connect(self.refresh)
        self.category = QComboBox()
        self.category.addItem("全部分类", "")
        self.category.currentIndexChanged.connect(self._filter)
        search = QPushButton("搜索")
        search.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(self.query, 1)
        top.addWidget(self.category)
        top.addWidget(search)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._open_page)

        self.status = QLabel("社区模型由其他用户制作。只有标记为“已验证”的模型"
                             "标记为“已验证”才表示经过 VideoHighlighter 团队审核。")
        self.status.setWordWrap(True)

        self.install_btn = QPushButton("安装")
        self.install_btn.clicked.connect(self._install)
        self.page_btn = QPushButton("打开模型页面")
        self.page_btn.clicked.connect(self._open_page)
        self.report_btn = QPushButton("报告问题")
        self.report_btn.clicked.connect(self._report)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Close).setText("关闭")
        buttons.rejected.connect(self.reject)
        actions = QHBoxLayout()
        for b in (self.install_btn, self.page_btn, self.report_btn):
            actions.addWidget(b)
        actions.addStretch()
        actions.addWidget(buttons)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.status)
        lay.addLayout(actions)
        self._selection_changed()
        self.refresh()

    # ---------------------------------------------------------------- data
    def _installed_ids(self) -> set[str]:
        return {m.repo_id for m in hub.list_installed(self.models_dir)}

    def refresh(self):
        self.status.setText("正在加载社区模型…")
        run_in_thread(self, _search_with_progress, self.query.text().strip(),
                      on_done=self._loaded, on_error=self._load_failed)

    def _load_failed(self, message: str):
        self.status.setText(f"无法加载社区模型，请检查网络连接。（{message}）")

    def _loaded(self, entries: list[hub.CatalogEntry]):
        self._all = entries
        current = self.category.currentData() or ""
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("全部分类", "")
        for cat in hub.categories_in(entries):
            depth = cat.count("/")
            self.category.addItem("    " * depth + cat.split("/")[-1], cat)
        idx = self.category.findData(current)
        self.category.setCurrentIndex(max(0, idx))
        self.category.blockSignals(False)
        self._filter()

    def _filter(self, *_):
        cat = self.category.currentData() or ""
        shown = [e for e in self._all
                 if not cat or e.category == cat or e.category.startswith(cat + "/")]
        self._show(shown)

    def _show(self, entries: list[hub.CatalogEntry]):
        self.entries = entries
        installed = self._installed_ids()
        self.table.setRowCount(len(entries))
        for row, e in enumerate(entries):
            parts = []
            if e.verified:
                parts.append("已验证")
            if e.repo_id in installed:
                parts.append("已安装")
            if not e.usable:
                parts.append("需要更新版本的软件")
            status = "，".join(parts) or "社区模型"
            values = [e.repo_id, e.category or "—", TASK_NAMES_ZH.get(e.task, TASKS.get(e.task, ("—",))[0]), status,
                      str(e.downloads), e.last_modified[:10]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)
        if entries:
            self.status.setText(f"共 {len(entries)} 个模型。")
        elif not self._all:
            self.status.setText("暂无社区模型。可在“训练”页训练一个并成为第一个分享模型的人。")
        else:
            self.status.setText("没有匹配的模型，请尝试其他关键词或分类。")
        self._selection_changed()

    def _current(self) -> hub.CatalogEntry | None:
        rows = self.table.selectionModel().selectedRows()
        return self.entries[rows[0].row()] if rows else None

    def _selection_changed(self):
        e = self._current()
        for b in (self.page_btn, self.report_btn):
            b.setEnabled(e is not None)
        self.install_btn.setEnabled(e is not None and e.usable)

    # ------------------------------------------------------------- actions
    def _install(self):
        e = self._current()
        if not e:
            return
        if not e.verified:
            answer = QMessageBox.question(
                self, "安装社区模型",
                f"{e.repo_id} 由其他用户制作，尚未经过审核。\n\n"
                "VideoHighlighter 会在使用前检查文件，并且只通过 "
                "ONNX Runtime 或 OpenVINO 运行。是否安装？")
            if answer != QMessageBox.Yes:
                return
        self.install_btn.setEnabled(False)
        run_in_thread(self, hub.install, e.repo_id, e.sha or None, self.models_dir,
                      on_progress=self.status.setText,
                      on_done=self._installed, on_error=self._install_failed)

    def _installed(self, result):
        model, report = result
        if model is None:
            QMessageBox.warning(self, "模型未安装",
                                "模型未通过安全性和兼容性检查：\n\n" + report.text())
            self.status.setText("已取消安装：模型未通过检查。")
        else:
            self.status.setText(f"已安装 {model.manifest.display_name}。"
                                "可在“高级 → 物体模型”中选择该模型。")
            self.installed.emit(model)
        self._filter()

    def _install_failed(self, message: str):
        self.status.setText(f"安装失败：{message}")
        self._selection_changed()

    def _open_page(self, *_):
        e = self._current()
        if e:
            QDesktopServices.openUrl(QUrl(e.url))

    def _report(self):
        e = self._current()
        if e:
            QDesktopServices.openUrl(QUrl(f"{e.url}/discussions/new"))


def _search_with_progress(query: str, progress=None):
    return hub.search_catalog(query=query)
