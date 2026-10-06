# COMPAT_SOURCE_MARKER_RUN_HIGHLIGHTER: QPushButton("Run Highlighter")
import os
import sys

# Capture every print/warning/traceback from the very first import: the
# packaged exe is --windowed (no stdout), so modules/system/debug_console tees all
# output into debug.log next to the exe and can mirror it to a live console
# window. Must run before the heavy imports below — some of them print
# warnings worth keeping.
from modules.system import debug_console
debug_console.install()

# A frozen build starts every multiprocessing child — the object-detection
# workers, their Manager, the thumbnail decoder — by re-running this exe, and
# the child becomes a child only when it reaches freeze_support(). Left at the
# bottom of the file, each one first loaded everything below: cv2, Qt,
# OpenVINO, transformers, the assistant. Object detection starts six of them;
# on a Mac that filled the memory until the machine had to be restarted.
# Here a child costs the log tee and nothing else. In the parent (and in any
# run from source) this is a no-op.
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

# Every relative path in the app — `./cache` above all — resolves against the
# working directory, and a packaged app does not get to choose what that is.
# macOS starts an .app in `/`, which is read-only, so the first cache write
# failed with "[Errno 30] Read-only file system: 'cache'". Do this before
# anything opens a file.
from modules.system.app_paths import use_writable_cwd
print(f"📂 工作目录：{use_writable_cwd()}")

# Interface size, if the user set one. Qt reads QT_SCALE_FACTOR when the
# QApplication is constructed and never again, so this has to happen before the
# Qt imports below — on a 55" 4K panel the OS scale is right for a television
# and far too large for an app at desk distance.
from modules.system import ui_scale
ui_scale.apply()

# Progress reporting for the launch itself. Imported here, before the heavy
# imports below, because in a frozen build *they* are the slow part — several
# seconds of decompressing and initialising cv2/OpenVINO/transformers before
# any window can exist. The bootloader's splash covers that stretch with the
# logo; these stage() calls are what make a slow launch readable afterwards in
# debug.log, and they would drive the splash text too if the build ever moves
# to a .spec (see modules/system/startup_splash.py on why the CLI flag cannot).
# This module deliberately pulls in no Qt, so it cannot disturb the import
# order below, which matters on Windows.
from modules.system import startup_splash
from modules.system import compute_backend
startup_splash.stage("正在加载视频引擎…")

import cv2
import json
import subprocess
import threading
import time
import yaml
import multiprocessing

from PySide6.QtWidgets import (
    QApplication, QCompleter, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QFormLayout, QProgressBar, QCheckBox,
    QComboBox, QTabWidget, QListWidget, QSplitter, QStackedWidget,
    QDialog, QDialogButtonBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QGridLayout, QSlider, QSizePolicy, QToolButton, QMenu,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMetaObject, Q_ARG, Slot, QStringListModel
from downloader import download_videos_with_immediate_processing, extract_video_links, DownloadError, reset_duration_method_cache
startup_splash.stage("正在加载智能助手…")
from llm.llm_chat_widget import LLMChatWidget
from modules.media.video_cache import VideoAnalysisCache, CachedAnalysisData, build_analysis_cache_params
from modules.report import analysis_stats
from modules.ui import icons as _ui_icons, theme as _ui_theme
from modules.segments.simple_run import apply_simple_run
from modules.ui.simple_start import (
    SimpleStartPage, persist_simple_start, simple_start_enabled,
)
# The five classes the expression scan can report. Imported for the Basic
# tab's picker; the module itself loads no model until something asks it to scan.
from modules.vision.face_emotions import EMOTION_LABELS

startup_splash.stage("正在加载检测运行环境…")
try:
    import openvino  # registers OpenVINO's DLL dir on Windows
except Exception:
    pass

from modules.system.app_paths import resource_path as _resource_path, data_file as _data_file, config_path
from modules.system.app_paths import action_model_file as _action_model_file
from version import __version__, __edition__

# --- Contact / support details shown in the About tab ---
SUPPORT_EMAIL = "przkreft@gmail.com"
WEBSITE_URL = "https://aseiel.github.io/VideoHighlighter-site/"
DISCORD_URL = "https://discord.gg/cUPJqPAMmm"
REPO_URL = "https://github.com/Aseiel/VideoHighlighter"

# User-editable config: lives next to the exe when frozen (so saves persist),
# seeded from the bundled default; just the project-root file when run from source.
CONFIG_FILE = config_path("config.yaml")

YOLO_OBJECTS_LABELS_FILE = _resource_path("yolo_objects_labels.json")
KINETICS_400_LABELS_FILE = _resource_path("kinetics_400_labels.json")
# Trained action models live in models/actions/ (the flat root locations stay
# as a fallback) — see app_paths.action_model_file().
INTEL_CUSTOM_LABELS_FILE = _action_model_file("intel_finetuned_classifier_3d_mapping.json")
R3D_CUSTOM_LABELS_FILE = _action_model_file("r3d_finetuned_mapping.json")

class LabelSelectorDialog(QDialog):
    """Dialog with search/filter and multi-select for labels."""

    def __init__(self, title, labels, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(480, 520)
        self.all_labels = sorted(labels)
        self.current_selection = set(current_selection or [])

        layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("筛选："))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("输入文字筛选标签…")
        self.search_input.textChanged.connect(self._filter_labels)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        self.info_label = QLabel(f"{len(self.all_labels)} 个标签可用")
        self.info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.info_label)

        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._populate_list(self.all_labels)
        layout.addWidget(self.label_list)

        quick_layout = QHBoxLayout()
        select_all_btn = QPushButton("选择全部可见项")
        select_all_btn.clicked.connect(self._select_all_visible)
        deselect_all_btn = QPushButton("取消全选")
        deselect_all_btn.clicked.connect(self._deselect_all)
        quick_layout.addWidget(select_all_btn)
        quick_layout.addWidget(deselect_all_btn)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)

        self.selection_label = QLabel("0 个已选择")
        self.selection_label.setStyleSheet("font-weight: bold; color: #2f81f7;")
        layout.addWidget(self.selection_label)
        self.label_list.itemSelectionChanged.connect(self._update_selection_count)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self._preselect_current()

    def _populate_list(self, labels):
        self.label_list.clear()
        for label in labels:
            self.label_list.addItem(label)

    def _preselect_current(self):
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.text() in self.current_selection:
                item.setSelected(True)
        self._update_selection_count()

    def _filter_labels(self, text):
        text = text.strip().lower()
        filtered = [l for l in self.all_labels if text in l.lower()] if text else self.all_labels
        self._populate_list(filtered)
        self.info_label.setText(f"已显示 {len(filtered)} / {len(self.all_labels)} 个标签")
        self._preselect_current()

    def _select_all_visible(self):
        for i in range(self.label_list.count()):
            self.label_list.item(i).setSelected(True)
        self._update_selection_count()

    def _deselect_all(self):
        self.label_list.clearSelection()
        self._update_selection_count()

    def _update_selection_count(self):
        self.selection_label.setText(f"已选择 {len(self.label_list.selectedItems())} 个")

    def get_selected_labels(self):
        return [item.text() for item in self.label_list.selectedItems()]

class NoAnalysisWarningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("没有分析数据")
        self.setFixedWidth(420)
        
        layout = QVBoxLayout()
        
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("font-size: 32px;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        msg = QLabel(
            "未找到该视频的分析缓存。\n\n"
            "你仍然可以使用时间线查看器浏览视频并与大模型对话，\n"
            "但运动、音频、物体和动作等分析信号将不可用。\n\n"
            "请先运行完整分析流程，以获得全部信号数据。"
        )
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg)
        
        self.dont_show_chk = QCheckBox("不再显示此警告")
        self.dont_show_chk.setStyleSheet("color: #666;")
        layout.addWidget(self.dont_show_chk)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("仍然打开")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

class MultiCompleter(QCompleter):
    """QCompleter that works on comma-separated fields, completing only the current token.
    Matches labels where any word starts with the typed text."""

    def __init__(self, labels=None, parent=None):
        super().__init__(parent)
        self._all_labels = labels or []
        self._source_model = QStringListModel(self._all_labels)
        self.setModel(self._source_model)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchContains)

    def setLabels(self, labels):
        """Update the full label list."""
        self._all_labels = labels
        self._source_model.setStringList(labels)

    def pathFromIndex(self, index):
        completion = super().pathFromIndex(index)
        widget = self.widget()
        if not widget:
            return completion
        text = widget.text()
        cursor = widget.cursorPosition()
        before = text[:cursor]
        last_comma = before.rfind(",")
        prefix = text[:last_comma + 1] + " " if last_comma >= 0 else ""
        after_cursor = text[cursor:]
        next_comma = after_cursor.find(",")
        suffix = after_cursor[next_comma:] if next_comma >= 0 else ""
        return prefix + completion + suffix

    def splitPath(self, path):
        widget = self.widget()
        if not widget:
            return [path.strip()]
        cursor = widget.cursorPosition()
        before = path[:cursor]
        last_comma = before.rfind(",")
        current_token = before[last_comma + 1:].strip().lower()

        # Filter: any word in label starts with typed text
        if current_token:
            filtered = [l for l in self._all_labels
                        if any(w.startswith(current_token) for w in l.lower().split())]
        else:
            filtered = self._all_labels
        self._source_model.setStringList(filtered)
        return [current_token]

class DownloadWorker(QThread):
    """
    Worker thread for downloading videos (with optional immediate processing after each file).
    
    Emits signals for:
    - progress updates
    - logging
    - finished list of downloaded paths
    - cancellation
    - individual video processed (when immediate processing is active)
    """
    finished = Signal(list)              # List of downloaded file paths
    progress = Signal(int, int, str, str)  # current, total, status, message
    log = Signal(str)                    # log messages
    cancelled = Signal()                 # emitted when cancelled
    video_processed = Signal(str, dict)  # filepath, processing result dict
    add_to_file_list = Signal(str)       # emits filepath to be added

    def __init__(self, url, save_dir, pattern, time_range=None, download_full=True,
                 use_percentages=False, immediate_processing=False, max_concurrent=1,
                 process_callback=None, video_urls=None):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.pattern = pattern
        self.time_range = time_range                  # (start, end) seconds or percentages
        self.download_full = download_full
        self.use_percentages = use_percentages
        self.immediate_processing = immediate_processing
        self.max_concurrent = max_concurrent
        self.process_callback = process_callback      # called after each download if immediate_processing
        self.video_urls = video_urls                  # explicit selection from the picker (skips scrape)
        self._cancelled = False
        self._is_running = False
        self._download_results = []                   # store all download metadata

    def run(self):
        try:
            self._is_running = True
            self.log.emit(f"🚀 开始下载：{self.url}")

            def log_fn(message):
                self.log.emit(message)

            def progress_fn(current, total, status, message):
                self.progress.emit(current, total, status, message)

            # Wraps the GUI-supplied callback. Emits video_processed so the GUI
            # can react per file. Only used when immediate_processing is on
            # AND a real callback was provided; otherwise the downloader runs
            # without per-video processing.
            def wrapped_process_callback(filepath, metadata):
                if self._cancelled:
                    return {'cancelled': True}
                self.log.emit(f"🔧 正在处理：{os.path.basename(filepath)}")
                try:
                    result = self.process_callback(filepath, metadata)
                    self.log.emit(f"✅ 已处理：{os.path.basename(filepath)}")
                    self.video_processed.emit(filepath, result)
                    return result
                except Exception as e:
                    self.log.emit(f"❌ 处理失败：{e}")
                    return {'error': str(e)}

            callback = (wrapped_process_callback
                        if (self.immediate_processing and self.process_callback)
                        else None)

            results = download_videos_with_immediate_processing(
                search_url=self.url,
                save_dir=self.save_dir,
                pattern=self.pattern,
                log_fn=log_fn,
                progress_fn=progress_fn,
                process_callback=callback,
                cancel_flag=self,
                time_range=self.time_range,
                download_full=self.download_full,
                use_percentages=self.use_percentages,
                max_workers=self.max_concurrent,
                video_urls=self.video_urls,
            )

            # Collect downloaded files
            downloaded_files = []
            for result in results:
                if result.get('success') and result.get('filepath'):
                    downloaded_files.append(result['filepath'])
                    self._download_results.append(result)

            if self._cancelled:
                self.log.emit("⏹️ 下载已取消")
                self.cancelled.emit()
                self.finished.emit([])
            else:
                self.finished.emit(downloaded_files)

        except Exception as e:
            self.log.emit(f"❌ 下载线程错误：{e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit([])
        finally:
            self._is_running = False

    def cancel(self):
        """Request cancellation – called from GUI.

        Non-blocking: just trip the flag and return. run() unwinds and emits
        cancelled/finished, which drive the UI cleanup. (Previously this called
        self.wait()/terminate() on the GUI thread, which froze the UI and — on
        timeout — killed the thread before it could emit its signals, leaving the
        Download button stuck disabled. force_download_cleanup is the safety net
        for a worker genuinely stuck in a non-cancellable subprocess.)"""
        if self._is_running:
            self.log.emit("⏹️ 已请求取消，正在停止下载…")
            self._cancelled = True

    def is_cancelled(self):
        """Public method used by downloader module to check cancellation"""
        return self._cancelled

    def is_set(self):
        """Compatibility alias – matches threading.Event.is_set()"""
        return self._cancelled
    
class DetectionPreviewWindow(QWidget):
    """Standalone window showing live detection frames during processing.

    Supports pause (freezes the pipeline) and rewind (scrub back through a ring
    buffer of recently shown frames).
    """

    closed = Signal()

    BUFFER_SIZE = 250  # rewind history (~30s at 8 fps); ~125MB of pixmaps

    def __init__(self, parent=None):
        super().__init__(parent)
        from collections import deque
        self.setWindowTitle("🔍 实时检测预览")
        self.setMinimumSize(560, 400)
        self.resize(720, 540)

        self._frames = deque(maxlen=self.BUFFER_SIZE)  # (pixmap, caption)
        self._paused = False
        self._view_index = -1  # -1 = follow live (latest)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.image_label = QLabel("等待检测阶段（物体 / 动作）…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { background:#101010; color:#8c8c8c; border:1px solid #333; }"
        )
        layout.addWidget(self.image_label, 1)

        self.caption = QLabel("")
        self.caption.setStyleSheet("color:#9aa; font-size:10pt;")
        self.caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.caption)

        # ── Controls: pause/resume, step, scrub slider ──
        controls = QHBoxLayout()
        self.pause_btn = QPushButton("⏸ 冻结")
        self.pause_btn.setFixedWidth(90)
        self.pause_btn.setToolTip("冻结预览以便检查当前帧。\n"
                                  "处理会继续在后台运行。")
        self.pause_btn.clicked.connect(self._toggle_pause)
        controls.addWidget(self.pause_btn)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        controls.addWidget(self.prev_btn)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setMinimum(0)
        self.scrub.setMaximum(0)
        self.scrub.sliderPressed.connect(self._on_scrub_pressed)
        self.scrub.valueChanged.connect(self._on_scrub_moved)
        controls.addWidget(self.scrub, 1)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(36)
        self.next_btn.clicked.connect(lambda: self._step(1))
        controls.addWidget(self.next_btn)

        self.live_btn = QPushButton("⏭ 实时")
        self.live_btn.setFixedWidth(70)
        self.live_btn.setToolTip("跳回实时画面并继续跟随")
        self.live_btn.clicked.connect(self._go_live)
        controls.addWidget(self.live_btn)

        layout.addLayout(controls)
        self._update_controls_enabled()

    # ── public: called from the GUI when a new frame arrives ──
    def set_frame(self, pixmap, caption=""):
        # Background processing keeps feeding frames even while frozen.
        was_full = len(self._frames) == self._frames.maxlen
        self._frames.append((pixmap, caption))
        # If frozen and the buffer just dropped its oldest frame, shift the view
        # index down by one so we keep looking at the SAME content.
        if self._paused and was_full and self._view_index > 0:
            self._view_index -= 1

        self.scrub.blockSignals(True)
        self.scrub.setMaximum(len(self._frames) - 1)
        self.scrub.setValue(self._view_index if self._paused else len(self._frames) - 1)
        self.scrub.blockSignals(False)

        if not self._paused:
            self._view_index = len(self._frames) - 1
            self._render_current()
        self._update_controls_enabled()

    # ── internals ──
    def _render_current(self):
        if not self._frames:
            return
        idx = self._view_index if self._view_index >= 0 else len(self._frames) - 1
        idx = max(0, min(idx, len(self._frames) - 1))
        pix, cap = self._frames[idx]
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        live_tag = "  • 实时" if (not self._paused and idx == len(self._frames) - 1) else \
                   f"  • 已冻结 {idx + 1}/{len(self._frames)}"
        self.caption.setText((cap or "") + live_tag)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("▶ 继续" if self._paused else "⏸ 冻结")
        if not self._paused:
            # Un-freeze → follow live again
            self._view_index = len(self._frames) - 1
            self._render_current()
        self._update_controls_enabled()

    def _on_scrub_pressed(self):
        # Touching the slider implies you want to review → auto-pause
        if not self._paused:
            self._toggle_pause()

    def _on_scrub_moved(self, value):
        if self._paused:
            self._view_index = value
            self._render_current()

    def _step(self, delta):
        if not self._paused:
            self._toggle_pause()
        self._view_index = max(0, min(self._view_index + delta, len(self._frames) - 1))
        self.scrub.blockSignals(True)
        self.scrub.setValue(self._view_index)
        self.scrub.blockSignals(False)
        self._render_current()

    def _go_live(self):
        if self._paused:
            self._toggle_pause()  # resume → follows live
        else:
            self._view_index = len(self._frames) - 1
            self._render_current()

    def _update_controls_enabled(self):
        has = len(self._frames) > 0
        self.prev_btn.setEnabled(has)
        self.next_btn.setEnabled(has)
        self.scrub.setEnabled(has)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class Worker(QThread):
    finished = Signal(object)
    progress = Signal(int, int, str, str)
    log = Signal(str)
    cancelled = Signal()
    preview = Signal(object, object, int)   # frame_bgr (ndarray), boxes (list), sec
    timeline_requested = Signal(str, object)  # video_path, analysis_data

    def __init__(self, video_path, gui_config=None):
        super().__init__()
        self.video_path = video_path
        self.gui_config = gui_config
        self._cancel_flag = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts unpaused
        self._is_running = False
        self.preview_enabled = False

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def is_paused(self):
        return not self._pause_event.is_set()

    def run(self):
        from pipeline import run_highlighter
        try:
            self._is_running = True

            def pausing_progress(cur, tot, task, det):
                self._pause_event.wait()  # blocks while paused
                if not self._cancel_flag.is_set():
                    self.progress.emit(cur, tot, task, det)

            # Check if single or multiple files
            if isinstance(self.video_path, list):
                self.log.emit(f"🚀 开始批量处理 {len(self.video_path)} 个视频…")
            else:
                self.log.emit("🚀 正在启动视频高光处理流程…")

            # Gate the preview emit on the live flag, checked per call so the
            # checkbox works mid-run. (The detector only builds/resizes a frame
            # ~8x/sec, negligible next to inference.)
            def preview_gate(frame, boxes, sec):
                if self.preview_enabled and not self._cancel_flag.is_set():
                    self.preview.emit(frame, boxes, sec)

            output = run_highlighter(
                self.video_path,
                gui_config=self.gui_config,
                log_fn=self.log.emit,
                progress_fn=pausing_progress,
                cancel_flag=self._cancel_flag,
                preview_fn=preview_gate,
                # Qt widgets may only be built on the main thread; emitting
                # hands the request off to the GUI's reuse-aware handler.
                timeline_fn=lambda path, data: self.timeline_requested.emit(str(path), data),
            )

            if self._cancel_flag.is_set():
                self.log.emit("⏹️ 处理流程已取消")
                self.cancelled.emit()
                self.finished.emit("")
            else:
                self.finished.emit(output or "")

        except Exception as e:
            self.log.emit(f"❌ 工作线程错误：{e}")
            import traceback
            self.log.emit(f"完整错误堆栈：{traceback.format_exc()}")
            self.finished.emit("")
        finally:
            self._is_running = False

    def cancel(self):
        if self._is_running:
            self.log.emit("⏹️ 已请求取消，正在停止处理流程…")
            self._cancel_flag.set()
            if not self.wait(5000):
                self.log.emit("⚠️ 正在强制终止线程…")
                self.terminate()
                self.wait()

    def is_cancelled(self):
        return self._cancel_flag.is_set()


class SignalRunWorker(QThread):
    """Run ONE analysis signal按需分析，共 a list of videos, folding each
    result into that video's cache (leaving the other signals intact).

    This is the main-window twin of the timeline viewer's "Analyze" panel: same
    engine (`modules.report.analysis_ondemand`), same fold-into-cache behaviour, just
    looped over the whole file list instead of one loaded video. It never cuts
    highlights — it only produces the standalone `.srt`/`.txt` (subtitles /
    transcript) and/or warms the cache for a later highlight run or the viewer.
    """
    finished = Signal(str)                  # short summary, or "" on hard error
    progress = Signal(int, int, str, str)   # current, total, task, message
    log = Signal(str)
    cancelled = Signal()
    preview = Signal(object, object, int)   # frame_bgr (ndarray), boxes, sec

    def __init__(self, kind, video_paths, params=None):
        super().__init__()
        self.kind = kind
        self.video_paths = list(video_paths)
        self.params = params or {}
        self._cancel_flag = threading.Event()
        self._is_running = False
        self.preview_enabled = False

    def run(self):
        from modules.report import analysis_ondemand as aod
        self._is_running = True
        n = len(self.video_paths)
        done = 0
        kind_label = {"motion": "运动", "audio": "音频", "objects": "物体",
                      "actions": "动作", "transcript": "转录", "subtitles": "字幕"}.get(
                          self.kind, self.kind)
        try:
            self.log.emit(f"🚀 {kind_label}按需分析，共 {n} 个视频…")
            for i, vp in enumerate(self.video_paths):
                if self._cancel_flag.is_set():
                    break
                name = os.path.basename(vp)
                self.log.emit(f"▶️ {kind_label} [{i+1}/{n}]：{name}")

                def progress(cur, tot, task, det, _i=i, _name=name):
                    frac = (cur / tot) if tot else 0.0
                    overall = int(((_i + frac) / n) * 100)
                    self.progress.emit(overall, 100, kind_label, f"{_name}：{det}")

                try:
                    patch = self._run_one(aod, vp, progress)
                except aod._Cancelled:
                    break
                except Exception as e:
                    self.log.emit(f"❌ {name}: {e}")
                    continue

                if patch:
                    aod.merge_into_cache(vp, patch, log=self.log.emit)
                    done += 1

            if self._cancel_flag.is_set():
                self.log.emit("⏹️ 按需分析已取消")
                self.cancelled.emit()
                self.finished.emit("")
            else:
                self.finished.emit(f"{kind_label}：已完成 {done}/{n}")
        except Exception as e:
            import traceback
            self.log.emit(f"❌ {kind_label}运行错误：{e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit("")
        finally:
            self._is_running = False

    def _run_one(self, aod, video_path, progress):
        """Dispatch to the matching on-demand runner and shape the cache patch.
        Mirrors the timeline viewer's per-kind cache keys."""
        c = self._cancel_flag
        p = self.params

        # An on-demand object/action run detects over the whole video exactly as
        # the pipeline's stage does — same detector, same length — so it feeds
        # the preview window from the same checkbox. Without this the window
        # opened and stayed on its placeholder for the entire run, which is the
        # one place the wait is longest and the reassurance worth most.
        # Checked per call so the checkbox still works mid-run.
        def preview_fn(frame, boxes, sec):
            if self.preview_enabled and not self._cancel_flag.is_set():
                self.preview.emit(frame, boxes, sec)

        if self.kind == "motion":
            return aod.run_motion(video_path, progress=progress, cancel=c, log=self.log.emit)
        if self.kind == "audio":
            return aod.run_audio(video_path, progress=progress, cancel=c, log=self.log.emit)
        if self.kind == "objects":
            result = aod.run_objects(video_path, p.get("objects") or [],
                                     progress=progress, cancel=c, log=self.log.emit,
                                     preview_fn=preview_fn)
            return {"objects": result}
        if self.kind == "actions":
            result = aod.run_actions(video_path, interesting_actions=p.get("actions") or [],
                                     progress=progress, cancel=c, log=self.log.emit,
                                     preview_fn=preview_fn)
            return {"actions": result, "actions_all": result}
        if self.kind == "transcript":
            result = aod.run_transcript(video_path, language=p.get("language"),
                                        progress=progress, cancel=c, log=self.log.emit)
            return {"transcript": result}
        if self.kind == "subtitles":
            result = aod.run_subtitles(video_path, language=p.get("language"),
                                       source_lang=p.get("source_lang"),
                                       target_lang=p.get("target_lang"),
                                       progress=progress, cancel=c, log=self.log.emit)
            return {"transcript": result}
        raise ValueError(f"未知信号类型：{self.kind}")

    def cancel(self):
        if self._is_running:
            self.log.emit("⏹️ 已请求取消——正在完成当前视频…")
            self._cancel_flag.set()

    def is_cancelled(self):
        return self._cancel_flag.is_set()


class FaceScanWorker(QThread):
    """Offline identity pass over a video to populate the face bank with everyone
    who appears, so they show up in the Avoid list (the 'dry run')."""
    log = Signal(str)
    done = Signal(int)   # identity count after scan, or -1 on error

    def __init__(self, video_path, db_path):
        super().__init__()
        self.video_path = video_path
        self.db_path = db_path

    def run(self):
            try:
                from video_ai_editor.face_identity import FaceIdentityBank
                from modules.segments.compute_forbidden import build_tracking_model, tag_entries

                bank = FaceIdentityBank(db_path=self.db_path)
                model = build_tracking_model("n", log_fn=self.log.emit)
                self.log.emit(f"🔍 正在扫描 {os.path.basename(self.video_path)} 中的人脸…")
                # tag_entries caches the per-frame tagging so the pipeline's avoid step
                # reuses this same pass instead of re-running face recognition.
                tag_entries(
                    self.video_path, bank,
                    yolo_model=model,
                    model_size="n",
                    face_every=15,
                    vid_stride=3,
                    save_bank=True,
                    log_fn=self.log.emit,
                )
                self.done.emit(len(bank))
            except Exception as e:
                self.log.emit(f"❌ 人脸扫描失败：{e}")
                self.done.emit(-1)

class UpdateCheckWorker(QThread):
    """Ask the update manifest whether a newer build exists.

    Off the GUI thread because a network round trip on it would freeze the
    window for the whole timeout on a bad connection — at the exact moment the
    user is trying to start work. Nothing here touches a widget; the answer
    travels back as a signal, and silence means "nothing to say".
    """

    found = Signal(object)   # modules.update.update_check.UpdateInfo
    nothing = Signal(str)    # only for an explicit "check now": why it found nothing

    def __init__(self, force=False, parent=None):
        super().__init__(parent)
        self.force = force

    def run(self):
        try:
            from modules.update import update_check
            info = update_check.check_for_update(force=self.force)
        except Exception as e:
            # An update check must never be the reason anything goes wrong.
            print(f"update_check: check failed ({type(e).__name__}: {e})")
            if self.force:
                self.nothing.emit("当前无法检查更新。")
            return
        if info:
            self.found.emit(info)
        elif self.force:
            # The automatic check says nothing when there is nothing; a user who
            # pressed a button is owed an answer either way.
            self.nothing.emit(f"当前已是最新版本（{__version__}).")


class UpdateInstallWorker(QThread):
    """Download and install a release, off the GUI thread.

    All the logic lives in modules/update/update_install; this only marshals progress
    and the result back to the window.
    """

    progress = Signal(str, int, int, str)   # phase, done, total, detail
    finished_with = Signal(object)          # update_install.InstallResult

    def __init__(self, manifest_url, root, parent=None):
        super().__init__(parent)
        self.manifest_url = manifest_url
        self.root = root
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from modules.update import update_install
        try:
            result = update_install.install_update(
                self.manifest_url, self.root,
                progress=lambda *a: self.progress.emit(*a),
                should_cancel=lambda: self._cancel,
            )
        except Exception as e:
            print(f"update_install: unexpected failure ({type(e).__name__}: {e})")
            result = update_install.InstallResult(
                ok=False, message=f"更新失败：{e}")
        self.finished_with.emit(result)


class RangeSlider(QWidget):
    """Single slider with two handles for selecting a range"""
    startChanged = Signal(int)
    endChanged = Signal(int)

    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._start = minimum
        self._end = maximum
        self._dragging = None  # 'start', 'end', or None
        self.setFixedHeight(32)
        self.setMinimumWidth(200)
        self.setCursor(Qt.PointingHandCursor)

    def start(self):
        return self._start

    def end(self):
        return self._end

    def setStart(self, val):
        val = max(self._min, min(val, self._end - 1))
        if val != self._start:
            self._start = val
            self.startChanged.emit(val)
            self.update()

    def setEnd(self, val):
        val = min(self._max, max(val, self._start + 1))
        if val != self._end:
            self._end = val
            self.endChanged.emit(val)
            self.update()

    def setRangeValues(self, start, end):
        """Set both handles at once.

        setStart()/setEnd() clamp against the *current* opposite handle, so
        calling them in sequence fails when the whole window moves past the old
        range (e.g. switching from 'first 5min' to 'last 5min' clamps the new
        start to the old end). Setting both together avoids that cross-clamp.
        """
        start = max(self._min, min(int(start), self._max))
        end = max(self._min, min(int(end), self._max))
        if start > end:
            start, end = end, start
        if end <= start:
            end = min(self._max, start + 1)
        changed_start = (start != self._start)
        changed_end = (end != self._end)
        self._start = start
        self._end = end
        if changed_start:
            self.startChanged.emit(start)
        if changed_end:
            self.endChanged.emit(end)
        if changed_start or changed_end:
            self.update()

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self._start = max(self._start, minimum)
        self._end = min(self._end, maximum)
        self.update()

    def _val_to_x(self, val):
        inset = 8
        w = self.width() - 2 * inset
        if self._max == self._min:
            return inset
        return inset + int((val - self._min) / (self._max - self._min) * w)

    def _x_to_val(self, x):
        inset = 8
        w = self.width() - 2 * inset
        if w <= 0:
            return self._min
        ratio = max(0.0, min(1.0, (x - inset) / w))
        return int(self._min + ratio * (self._max - self._min))

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        x0 = self._val_to_x(self._start)
        x1 = self._val_to_x(self._end)
        track_y = self.height() // 2 - 3
        track_h = 6

        # Full track background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 60, 60))
        inset = 8
        p.drawRoundedRect(inset, track_y, self.width() - 2 * inset, track_h, 3, 3)

        # Selected range
        p.setBrush(QColor(47, 129, 247))
        p.drawRoundedRect(x0, track_y, max(2, x1 - x0), track_h, 3, 3)

        # Start handle
        p.setBrush(QColor(222, 222, 222))
        p.setPen(QColor(47, 129, 247))
        p.drawEllipse(x0 - 7, self.height() // 2 - 7, 14, 14)

        # End handle
        p.drawEllipse(x1 - 7, self.height() // 2 - 7, 14, 14)

        p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.position().toPoint().x()
        x0 = self._val_to_x(self._start)
        x1 = self._val_to_x(self._end)

        dist_start = abs(x - x0)
        dist_end = abs(x - x1)

        if dist_start <= dist_end and dist_start < 20:
            self._dragging = 'start'
        elif dist_end < 20:
            self._dragging = 'end'
        elif x0 < x < x1:
            # Click between handles — move nearest
            self._dragging = 'start' if dist_start < dist_end else 'end'

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        val = self._x_to_val(event.position().toPoint().x())
        if self._dragging == 'start':
            self.setStart(val)
        else:
            self.setEnd(val)

    def mouseReleaseEvent(self, event):
        self._dragging = None

class VideoHighlighterGUI(QWidget):
    #: A detection frame on its way to the preview window. The Run button has
    #: `Worker.preview` for this, but the download-and-process path calls
    #: `run_highlighter` straight from the download worker's thread and has no
    #: Worker to borrow a signal from — so the window owns one, and the hop to
    #: the GUI thread happens here rather than in each caller.
    preview_frame = Signal(object, object, int)   # frame_bgr, boxes, sec

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Highlighter v{__version__} {__edition__}")
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1000, screen.width() - 20)
        # Open as tall as the screen comfortably allows. The fixed sections
        # above the tabs (input list, output name, time range) cost ~410px
        # before a single tab row is drawn, so an 800px window spent half its
        # height before the settings even started. The cap only binds on very
        # tall screens; everywhere else the available height decides, and the
        # window can still shrink to its ~794px minimum.
        h = min(1200, screen.height() - 20)
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w) // 2, screen.y())

        # Coalesced, because a drag-resize fires dozens of these a second and a
        # log full of intermediate sizes hides the one that matters — the last
        # size before a crash that leaves no traceback of its own.
        self._size_log_timer = QTimer(self)
        self._size_log_timer.setSingleShot(True)
        self._size_log_timer.setInterval(400)
        self._size_log_timer.timeout.connect(self._log_size)

        self.worker = None

        self.config_data = self.load_config()

        # Publish the saved DirectML choice into the environment before anything
        # probes a device, so worker processes — which inherit the environment
        # and nothing else — make the same choice the GUI shows.
        compute_backend.apply(self.config_data)

        # Window root: update banner + a stack of (Simple start | full UI).
        # Simple start is the first-run alternative for issue #20; every
        # existing widget still lives on full_page — nothing is removed.
        root = QVBoxLayout()
        root.setContentsMargins(16, 8, 16, 8)
        root.setSpacing(6)

        # --- Update notice (hidden unless there is actually a newer build) ---
        # Costs no vertical space while hidden, which is the whole reason it is
        # a banner and not a startup dialog: nothing interrupts a launch, and
        # nothing is permanently occupying a row on a small screen.
        self.update_banner = self._build_update_banner()
        root.addWidget(self.update_banner)

        # --- Pro offer (hidden unless the user just hit something only Pro
        # does; see modules/ui/pro_offer.py for when that is) ---
        self.pro_banner = self._build_pro_banner()
        root.addWidget(self.pro_banner)

        self.view_stack = QStackedWidget()
        root.addWidget(self.view_stack, 1)

        self.full_page = QWidget()
        layout = QVBoxLayout(self.full_page)
        # A little breathing room, but tight enough that the ~8 stacked sections
        # don't add up to a screenful of gaps (that empty space pushed the tabs
        # and Run row down). Trimmed from the original 20/16/14.
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Store video duration
        self.current_video_duration = 0

        # --- File picker ---
        file_group = QGroupBox("输入视频")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(12, 6, 12, 6)
        file_layout.setSpacing(6)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("添加视频")
        self.browse_btn.clicked.connect(self.browse_files)
        self.remove_btn = QPushButton("移除所选")
        self.remove_btn.clicked.connect(self.remove_selected_file)
        self.clear_btn = QPushButton("全部清空")
        self.clear_btn.clicked.connect(self.clear_files)
        
        btn_layout.addWidget(self.browse_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()  # Push buttons to the left

        file_layout.addLayout(btn_layout)

        # File list — compact; it scrolls when there are more files. The tall
        # white box was the biggest chunk of empty space up top.
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(46)
        file_layout.addWidget(self.file_list)

        saved_paths = self.config_data.get("video", {}).get("paths", [])
        if saved_paths:
            for path in saved_paths:
                if os.path.exists(path):
                    self.file_list.addItem(path)
        
        file_group.setLayout(file_layout)
        # Vertical Maximum: the group takes only its natural height and never gets
        # stretched by slack. Without this, extra window height inflated this box
        # into a tall mostly-empty white panel up top.
        file_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(file_group)

        # --- Output filename ---
        # Blank keeps <video>_highlight.mp4 (what a run writes today). Any other
        # text is the mp4's base name and is passed through as output_base.
        out_layout = QHBoxLayout()
        self.output_input = QLineEdit(self.config_data.get("highlights", {}).get("output", "highlight.mp4"))
        self.output_input.setPlaceholderText("留空 = <视频名>_highlight.mp4")
        self.output_input.setToolTip(
            "高光 MP4 的基础文件名，输出到源视频旁边。\n"
            "留空时使用 <视频名>_highlight.mp4。\n"
            "单个视频：该名称直接作为文件名（需要时自动补 .mp4）。\n"
            "多个视频或下载后处理：使用 <视频名>_<名称>.mp4，避免互相覆盖。"
        )
        out_layout.addWidget(QLabel("输出基础名称："))
        out_layout.addWidget(self.output_input)
        layout.addLayout(out_layout)

        # Named snapshots beside config.yaml (presets/<name>.yaml). The live
        # file is still config.yaml; loading a preset copies it onto the
        # controls and save_config writes that back.
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("预设："))
        self.preset_name_input = QLineEdit()
        self.preset_name_input.setPlaceholderText("名称")
        self.preset_name_input.setToolTip(
            "以此名称保存当前设置。预设将保存为 presets/<名称>.yaml，\n"
            "位于 config.yaml 旁边；在加载预设前不会修改 config.yaml。"
        )
        self.preset_combo = QComboBox()
        self.preset_combo.setMinimumWidth(160)
        self.preset_combo.setToolTip("设置预设保存在 config.yaml 同目录。")
        self.preset_save_btn = QPushButton("保存")
        self.preset_save_btn.setToolTip("使用左侧名称保存当前设置。")
        self.preset_save_btn.clicked.connect(self.save_named_preset)
        self.preset_load_btn = QPushButton("加载")
        self.preset_load_btn.setToolTip(
            "将所选预设应用到当前控件，并写入 config.yaml。"
        )
        self.preset_load_btn.clicked.connect(self.load_named_preset)
        self.preset_delete_btn = QPushButton("删除")
        self.preset_delete_btn.setToolTip("删除所选预设文件，不会修改 config.yaml。")
        self.preset_delete_btn.clicked.connect(self.delete_named_preset)
        preset_layout.addWidget(self.preset_name_input)
        preset_layout.addWidget(self.preset_combo, stretch=1)
        preset_layout.addWidget(self.preset_save_btn)
        preset_layout.addWidget(self.preset_load_btn)
        preset_layout.addWidget(self.preset_delete_btn)
        layout.addLayout(preset_layout)
        self._refresh_presets()

        highlights_cfg = self.config_data.get("highlights", {})
        scoring_cfg = self.config_data.get("scoring", {})

        # --- Time Range Selection with Slider ---
        time_range_group = QGroupBox("处理时间范围")
        time_range_layout = QVBoxLayout()

        # Enable/disable checkbox
        self.use_time_range_chk = QCheckBox("仅处理指定时间范围")
        self.use_time_range_chk.setChecked(highlights_cfg.get("use_time_range", False))
        self.use_time_range_chk.toggled.connect(self.on_time_range_toggle)
        time_range_layout.addWidget(self.use_time_range_chk)

        # Everything below the checkbox lives in a body that is only shown while
        # "仅处理指定时间范围" is ticked. Off (the default), this
        # whole group collapses to one line — the slider, %-labels, selection
        # text and preset buttons no longer take a chunk of the window.
        self.time_range_body = QWidget()
        time_range_body_layout = QVBoxLayout(self.time_range_body)
        time_range_body_layout.setContentsMargins(0, 0, 0, 0)
        time_range_body_layout.setSpacing(4)

        # Video duration label
        self.video_duration_label = QLabel("按百分比设置时间范围（0-100%）— 选择视频后会加载实际时间")
        self.video_duration_label.setStyleSheet("color: #666; font-style: italic;")
        time_range_body_layout.addWidget(self.video_duration_label)

        # Range slider container
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)

        # Range slider (single bar with two handles)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("开始："))
        self.range_slider = RangeSlider(0, 100)
        self.range_slider.setStart(highlights_cfg.get("range_start_pct", 0))
        self.range_slider.setEnd(highlights_cfg.get("range_end_pct", 100))
        self.range_slider.setEnabled(False)
        self.range_slider.startChanged.connect(self.on_slider_changed)
        self.range_slider.endChanged.connect(self.on_slider_changed)
        range_row.addWidget(self.range_slider, stretch=1)
        range_row.addWidget(QLabel("结束"))

        self.start_time_label = QLabel("0%")
        self.start_time_label.setMinimumWidth(80)
        self.start_time_label.setStyleSheet("font-weight: bold;")

        self.end_time_label = QLabel("100%")
        self.end_time_label.setMinimumWidth(80)
        self.end_time_label.setStyleSheet("font-weight: bold;")

        labels_row = QHBoxLayout()
        labels_row.addWidget(self.start_time_label)
        labels_row.addStretch()
        labels_row.addWidget(self.end_time_label)

        slider_layout.addLayout(range_row)
        slider_layout.addLayout(labels_row)

        slider_container.setLayout(slider_layout)
        time_range_body_layout.addWidget(slider_container)

        # Selection info
        self.selection_info_label = QLabel("选择：完整视频")
        self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")
        time_range_body_layout.addWidget(self.selection_info_label)

        # Quick presets
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("快速预设："))
        self.first_5min_btn = QPushButton("前 5 分钟")
        self.first_5min_btn.clicked.connect(lambda: self.set_slider_preset("first_5"))
        self.first_5min_btn.setEnabled(False)
        self.last_5min_btn = QPushButton("后 5 分钟")
        self.last_5min_btn.clicked.connect(lambda: self.set_slider_preset("last_5"))
        self.last_5min_btn.setEnabled(False)
        self.last_10min_btn = QPushButton("后 10 分钟")
        self.last_10min_btn.clicked.connect(lambda: self.set_slider_preset("last_10"))
        self.last_10min_btn.setEnabled(False)
        self.middle_btn = QPushButton("中间")
        self.middle_btn.clicked.connect(lambda: self.set_slider_preset("middle"))
        self.middle_btn.setEnabled(False)
        self.full_video_btn = QPushButton("完整视频")
        self.full_video_btn.clicked.connect(lambda: self.set_slider_preset("full"))
        self.full_video_btn.setEnabled(False)
        presets_layout.addWidget(self.first_5min_btn)
        presets_layout.addWidget(self.last_5min_btn)
        presets_layout.addWidget(self.last_10min_btn)
        presets_layout.addWidget(self.middle_btn)
        presets_layout.addWidget(self.full_video_btn)
        presets_layout.addStretch()
        time_range_body_layout.addLayout(presets_layout)

        time_range_layout.addWidget(self.time_range_body)
        # Collapsed unless the box is already ticked from saved config.
        self.time_range_body.setVisible(self.use_time_range_chk.isChecked())

        time_range_group.setLayout(time_range_layout)
        time_range_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(time_range_group)

        # Enable slider if checkbox was already checked from config
        if self.use_time_range_chk.isChecked():
            self.range_slider.setEnabled(True)

        # Initialize the selection info display with saved values
        self.update_selection_info()

        # Load duration from first saved video
        if self.file_list.count() > 0:
            first_path = self.file_list.item(0).text()
            if os.path.exists(first_path):
                self.update_video_duration(first_path)

        # --- Live detection preview (opens a separate window) ---
        # Added before the progress group so that expanding the progress bars
        # (when the pipeline starts) does not push these controls off-screen.
        self.live_preview_checkbox = QCheckBox("实时检测预览（独立窗口）")
        self.live_preview_checkbox.setToolTip(
            "处理流程运行时打开窗口，实时显示视频帧和检测到的物体框。\n"
            "预览会限频并缩小显示，不会明显拖慢处理速度。"
        )
        self.live_preview_checkbox.toggled.connect(self._on_live_preview_toggled)
        layout.addWidget(self.live_preview_checkbox)
        self.preview_window = None  # DetectionPreviewWindow, created on demand
        # Read from detection threads, so it mirrors the checkbox rather than
        # being queried across threads (same reason Worker keeps its own copy).
        self._preview_enabled = False
        self.preview_frame.connect(self.on_preview_frame)

        # Force reprocess — the live preview only shows frames while detection
        # actually runs. If results are cached, detection is skipped and the
        # preview stays blank. Tick this to ignore the cache and re-run.
        self.force_reprocess_checkbox = QCheckBox("强制重新处理（忽略缓存）")
        self.force_reprocess_checkbox.setToolTip(
            "即使已有缓存结果，也重新运行分析。\n"
            "对于已处理过的视频，如需查看实时检测预览，必须启用此项。"
        )
        layout.addWidget(self.force_reprocess_checkbox)

        # --- Progress Section (hidden when idle) ---
        self.progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(4, 4, 4, 4)
        progress_layout.setSpacing(2)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setVisible(False)
        self.download_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.download_progress_bar)

        # Batch counter as text, not a bar. The whole group is hidden when idle
        # and shown while running, so every row in it is height the window gains
        # at the moment a run starts — and on a screen where the window is
        # already at its limit, that pushes the buttons under the taskbar. The
        # label below said "Video 1/1" anyway, so the bar was a second copy of
        # the same fact costing a row.
        self.batch_label = QLabel()
        self.batch_label.setVisible(False)
        self.batch_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.batch_label)

        self.process_progress_bar = QProgressBar()
        self.process_progress_bar.setVisible(False)
        self.process_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.process_progress_bar)

        self.task_label = QLabel("就绪")
        self.task_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.task_label)

        self.progress_group.setLayout(progress_layout)
        # Hidden when idle; sits here near the top (above the tabs) — its original
        # spot. It only appears while a download or pipeline/analysis runs.
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        # --- Tabs ---
        # Kept on self so features elsewhere can bring a tab forward — the
        # advisor hands a run to the LLM Chat tab rather than to a window.
        self.tabs = tabs = QTabWidget()

        # --- Tab 0: Download ---
        download_tab = QWidget()
        download_layout = QVBoxLayout()

        download_group = QGroupBox("从网站下载视频")
        download_form = QVBoxLayout()

        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("页面网址："))
        self.download_url_input = QLineEdit()
        self.download_url_input.setText(self.config_data.get("download", {}).get("last_url", ""))
        self.download_url_input.setPlaceholderText("https://example.com/videos")
        url_layout.addWidget(self.download_url_input)
        download_form.addLayout(url_layout)

        # Link pattern is auto-detected from the listing page (see
        # downloader.detect_link_pattern), so there's no manual field.

        # Save directory
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(QLabel("保存目录："))
        self.download_save_dir_input = QLineEdit()
        self.download_save_dir_input.setText(self.config_data.get("download", {}).get("save_dir", "D:\\movies"))
        save_dir_layout.addWidget(self.download_save_dir_input)
        self.browse_save_dir_btn = QPushButton("浏览…")
        self.browse_save_dir_btn.clicked.connect(self.browse_save_directory)
        save_dir_layout.addWidget(self.browse_save_dir_btn)
        download_form.addLayout(save_dir_layout)

        # Time range selection for downloads. One mode picker instead of two
        # overlapping checkboxes — the modes are mutually exclusive.
        time_range_group = QGroupBox("下载时间范围")
        time_range_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("下载："))
        self.download_mode_combo = QComboBox()
        self.download_mode_combo.addItem("完整视频", "full")
        self.download_mode_combo.addItem("与处理范围相同", "same")
        self.download_mode_combo.addItem("指定范围（秒）", "specific")
        self.download_mode_combo.setToolTip(
            "完整视频——下载全部内容。\n"
            "与处理范围相同——复用上方“处理时间范围”。\n"
            "指定范围——只下载下方设置的时间段。"
        )
        mode_row.addWidget(self.download_mode_combo)
        mode_row.addStretch()
        time_range_layout.addLayout(mode_row)

        # Manual seconds range — shown only in "Specific range" mode.
        self.download_range_widget = QWidget()
        range_col = QVBoxLayout(self.download_range_widget)
        range_col.setContentsMargins(0, 0, 0, 0)
        time_input_layout = QHBoxLayout()
        time_input_layout.addWidget(QLabel("开始时间（秒）："))
        self.download_start_input = QSpinBox()
        self.download_start_input.setRange(0, 86400)  # 0 to 24 hours
        self.download_start_input.setValue(0)
        time_input_layout.addWidget(self.download_start_input)

        time_input_layout.addWidget(QLabel("结束时间（秒）："))
        self.download_end_input = QSpinBox()
        self.download_end_input.setRange(1, 86400)  # 1 second to 24 hours
        self.download_end_input.setValue(300)  # Default: 5 minutes
        time_input_layout.addWidget(self.download_end_input)
        time_input_layout.addStretch()
        range_col.addLayout(time_input_layout)

        self.download_duration_label = QLabel("时长：300 秒（5:00）")
        range_col.addWidget(self.download_duration_label)
        time_range_layout.addWidget(self.download_range_widget)

        # Connect signals
        self.download_start_input.valueChanged.connect(self.update_download_duration)
        self.download_end_input.valueChanged.connect(self.update_download_duration)
        self.download_mode_combo.currentIndexChanged.connect(self.on_download_mode_changed)

        time_range_group.setLayout(time_range_layout)
        download_form.addWidget(time_range_group)

        # Options
        self.auto_add_downloaded_chk = QCheckBox("下载后自动添加到视频列表")
        self.auto_add_downloaded_chk.setChecked(self.config_data.get("download", {}).get("auto_add", True))
        download_form.addWidget(self.auto_add_downloaded_chk)

        # After-download processing: one mode picker instead of two overlapping
        # (and half-dead) checkboxes.
        process_row = QHBoxLayout()
        process_row.addWidget(QLabel("下载后："))
        self.process_mode_combo = QComboBox()
        self.process_mode_combo.addItem("不处理，仅下载", "none")
        self.process_mode_combo.addItem("每个视频下载后立即处理", "immediate")
        self.process_mode_combo.addItem("全部下载完成后统一处理", "batch")
        self.process_mode_combo.setToolTip(
            "不处理——只下载视频。\n"
            "边下载边处理——每个视频下载完成后立即开始分析，与剩余下载并行。\n"
            "全部下载后处理——先完成全部下载，再统一处理视频列表。"
        )
        process_row.addWidget(self.process_mode_combo)
        process_row.addStretch()
        download_form.addLayout(process_row)

        # Concurrent downloads — only meaningful while processing overlaps
        # downloads (the "immediate" mode).
        concurrent_layout = QHBoxLayout()
        concurrent_layout.addWidget(QLabel("并发下载数："))
        self.concurrent_spinbox = QSpinBox()
        self.concurrent_spinbox.setRange(1, 10)
        self.concurrent_spinbox.setValue(self.config_data.get("download", {}).get("concurrent_downloads", 1))
        self.concurrent_spinbox.setToolTip("同时下载的视频数量（越高越快，但占用更多资源）")
        concurrent_layout.addWidget(self.concurrent_spinbox)
        concurrent_layout.addStretch()
        download_form.addLayout(concurrent_layout)

        self.process_mode_combo.currentIndexChanged.connect(self.on_process_mode_changed)
        self.on_process_mode_changed()  # sync spinner enabled state

        # Download buttons. The pair is "choose some" vs "take everything", so
        # the labels say which is which, and only the second gets accent fill.
        download_btn_layout = QHBoxLayout()
        self.browse_select_btn = QPushButton("从页面选择视频…")
        self.browse_select_btn.setIcon(_ui_icons.picker())
        # No inline style: this is a plain secondary button, so it inherits the
        # theme's default QPushButton and stays in step if the palette changes.
        self.browse_select_btn.setToolTip("以缩略图网格显示页面视频，并选择要下载的内容")
        self.browse_select_btn.clicked.connect(self.browse_and_select_videos)

        self.download_btn = QPushButton("全部下载")
        self.download_btn.setIcon(_ui_icons.download())
        self.download_btn.setToolTip("无需选择，下载页面中找到的全部视频")
        # Accent fill marks the primary action. The disabled rule matters: this
        # button is switched off for the whole download, and without it the fill
        # stays bright blue and keeps inviting clicks that do nothing.
        _p = _ui_theme.DARK
        self.download_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_p.accent}; color: {_p.on_accent};"
            f" font-weight: bold; padding: 8px; border: none; border-radius: {_p.radius}px; }}"
            f"QPushButton:hover {{ background-color: {_p.accent_hover}; }}"
            f"QPushButton:pressed {{ background-color: {_p.accent_press}; }}"
            f"QPushButton:disabled {{ background-color: {_p.surface}; color: {_p.text_mute}; }}"
        )
        # lambda so the clicked(bool) arg isn't passed as start_download's video_urls
        self.download_btn.clicked.connect(lambda: self.start_download())
        download_btn_layout.addStretch()
        download_btn_layout.addWidget(self.browse_select_btn)
        download_btn_layout.addWidget(self.download_btn)
        download_form.addLayout(download_btn_layout)

        download_group.setLayout(download_form)
        download_layout.addWidget(download_group)
        download_layout.addStretch()
        download_tab.setLayout(download_layout)
        tabs.addTab(self._scrollable(download_tab), "下载")

        # --- Tab 1: Basic Settings ---
        basic_tab = QWidget()
        # Grid so the two tall groups (Scoring Points / Duration) sit side by side
        # and use horizontal space instead of stacking into one tall column that
        # overflows the window. Mirrors the Advanced tab layout.
        basic_layout = QGridLayout()

        # On-demand run buttons sit right next to the scoring row they run. Each
        # runs that one signal over every video in the list and folds the result
        # into each cache — no highlights are cut. Registered here so a full
        # pipeline run (or another on-demand run) can grey them out.
        self._analyze_buttons = {}
        self._signal_worker = None

        # ── Group 1: Scoring Points ──
        points_box = QGroupBox("评分项")
        points_layout = QVBoxLayout()
        points_layout.setSpacing(6)

        self.spin_scene_points = QSpinBox(); self.spin_scene_points.setRange(0,100); self.spin_scene_points.setValue(scoring_cfg.get("scene_points", 0))
        self.spin_scene_points.setToolTip(
            "场景切换开始时（帧差突然增大）对对应秒数加分。"
            "每次切换只记一次，不会持续给后续场景加分。")

        self.spin_motion_event_points = QSpinBox(); self.spin_motion_event_points.setRange(0,100); self.spin_motion_event_points.setValue(scoring_cfg.get("motion_event_points", 0))
        self.spin_motion_event_points.setToolTip(
            "采样帧检测到运动时对对应秒数加分（轮廓面积超过检测阈值）。"
            "这不是运动峰值；运动峰值用于检测突然增强后又降低的运动。")

        self.spin_motion_peak = QSpinBox(); self.spin_motion_peak.setRange(0,100); self.spin_motion_peak.setValue(scoring_cfg.get("motion_peak_points", 3))
        self.spin_motion_peak.setToolTip(
            "当场景内运动强度高于该场景平均值，随后约 4 秒采样保持较低时加分。")

        self.spin_audio_peak = QSpinBox(); self.spin_audio_peak.setRange(0,100); self.spin_audio_peak.setValue(scoring_cfg.get("audio_peak_points", 0))
        self.spin_audio_peak.setToolTip(
            "局部波形峰值高于 -20 dB 时对对应秒数加分。"
            "它不同于响度突增；响度突增会把每个时刻与其附近片段比较。")

        self.spin_loudness_burst = QSpinBox(); self.spin_loudness_burst.setRange(0,100)
        self.spin_loudness_burst.setValue(scoring_cfg.get("loudness_burst_points", 0))
        self.spin_loudness_burst.setToolTip(
            "当音频相对其自身局部平均响度明显升高时加分，而不是使用固定阈值。"
            "\n\n如果精彩时刻只是相对于当前片段更响，而非绝对音量更大，建议使用此项；"
            "它会自动适配不同响度的视频。"
            "\n\n该功能用于查找短暂突出的声音；如果整段始终很响，则不会被视为局部突发。")

        self.spin_keyword_points = QSpinBox(); self.spin_keyword_points.setRange(0,100); self.spin_keyword_points.setValue(scoring_cfg.get("keyword_points", 2))
        self.spin_keyword_points.setToolTip(
            "转录片段中出现搜索关键词时，对该片段覆盖的每一秒加分。未启用转录时忽略。")
        # Keyword scoring only works with a transcript — grey it out until then.
        self.spin_keyword_points.setEnabled(self.config_data.get("transcript", {}).get("enabled", False))

        self.spin_transcript_points = QSpinBox(); self.spin_transcript_points.setRange(0,100); self.spin_transcript_points.setValue(scoring_cfg.get("transcript_points", 2))
        self.spin_transcript_points.setToolTip(
            "转录结果会随本次分析保存，但此项本身不会直接给某一秒加分。"
            "语音关键词由“关键词加分”标记；仅在启用转录时参与评分有效性检查。")

        self.spin_object = QSpinBox(); self.spin_object.setRange(0,100); self.spin_object.setValue(scoring_cfg.get("object_points", 1))
        self.spin_object.setToolTip(
            "检测到“物体检测”字段中的类别时，对对应秒数加分。字段为空时不计此项。")

        self.spin_action = QSpinBox(); self.spin_action.setRange(0,1000); self.spin_action.setValue(scoring_cfg.get("action_points", 10))
        self.spin_action.setToolTip(
            "识别到“动作关键词”字段中的动作时加分；分数会按该动作在当前视频中的"
            "置信度缩放（0.5 倍、1 倍或 1.5 倍）。字段为空时不计此项。")

        self.spin_face_expression = QSpinBox(); self.spin_face_expression.setRange(0,100)
        self.spin_face_expression.setValue(scoring_cfg.get("face_expression_points", 0))
        self.spin_face_expression.setToolTip(
            "当画面中最明显的人脸被识别为你选择的表情之一时，对对应秒数加分。\n\n"
            "只有分数大于 0 且至少选择一种表情时才会运行扫描。\n\n"
            "这里显示的是五分类模型对人脸外观的识别结果，不代表人物真实情绪；"
            "侧脸、遮挡等情况会降低准确性，也无法判断表情是表演还是真实感受。")
        self._face_label_actions = {}
        self.btn_face_labels = QToolButton()
        self.btn_face_labels.setPopupMode(QToolButton.InstantPopup)
        self.btn_face_labels.setToolTip(
            "选择哪些表情可以获得上方分数。若全部选择，则只要出现人脸的每一秒都会加分，"
            "失去区分意义；建议只选择真正代表目标时刻的表情。")
        face_menu = QMenu(self.btn_face_labels)
        chosen = {str(x).lower()
                  for x in (scoring_cfg.get("face_expression_labels") or [])}
        for _label in EMOTION_LABELS:
            act = face_menu.addAction(_label)
            act.setCheckable(True)
            act.setChecked(_label in chosen)
            act.toggled.connect(self._update_face_labels_button)
            self._face_label_actions[_label] = act
        self.btn_face_labels.setMenu(face_menu)
        self._update_face_labels_button()

        self.spin_beginning_seconds = QSpinBox(); self.spin_beginning_seconds.setRange(0,3600); self.spin_beginning_seconds.setSuffix(" 秒"); self.spin_beginning_seconds.setValue(scoring_cfg.get("beginning_seconds", 60))
        self.spin_beginning_seconds.setToolTip("从视频开头计算多少秒作为片头时间窗")

        self.spin_beginning_points = QSpinBox(); self.spin_beginning_points.setRange(0,100); self.spin_beginning_points.setSuffix(" 分"); self.spin_beginning_points.setValue(scoring_cfg.get("beginning_points", 0))
        self.spin_beginning_points.setToolTip("片头时间窗内每秒增加的分数；提高可让片头更容易被选为高光，设为 0 则与普通片段同等评分")

        self.spin_ending_seconds = QSpinBox(); self.spin_ending_seconds.setRange(0,3600); self.spin_ending_seconds.setSuffix(" 秒"); self.spin_ending_seconds.setValue(scoring_cfg.get("ending_seconds", 120))
        self.spin_ending_seconds.setToolTip("视频结束前多少秒作为片尾时间窗")

        self.spin_ending_points = QSpinBox(); self.spin_ending_points.setRange(0,100); self.spin_ending_points.setSuffix(" 分"); self.spin_ending_points.setValue(scoring_cfg.get("ending_points", 0))
        self.spin_ending_points.setToolTip("片尾时间窗内每秒增加的分数；提高可让片尾更容易被选为高光，设为 0 则与普通片段同等评分")

        intro_row = QHBoxLayout()
        intro_row.addWidget(self.spin_beginning_seconds)
        intro_row.addWidget(self.spin_beginning_points)
        intro_row.addStretch(1)
        intro_widget = QWidget(); intro_widget.setLayout(intro_row)

        outro_row = QHBoxLayout()
        outro_row.addWidget(self.spin_ending_seconds)
        outro_row.addWidget(self.spin_ending_points)
        outro_row.addStretch(1)
        outro_widget = QWidget(); outro_widget.setLayout(outro_row)

        # One box per signal rather than one list of eleven rows. The labels
        # only mean anything once the reader knows which detector a row belongs
        # to, and in a flat column that has to be inferred from the wording of
        # each label — which is why the shortest ones ("Scene points") were the
        # hardest to place. A box answers it before the row is read.
        face_row = QWidget()
        face_h = QHBoxLayout(face_row)
        face_h.setContentsMargins(0, 0, 0, 0)
        face_h.setSpacing(6)
        face_h.addWidget(self.spin_face_expression)
        face_h.addWidget(self.btn_face_labels)
        face_h.addStretch(1)

        # Scene / motion-event / motion-peak all come from ONE detector pass, so
        # the button lives on the first of the three and runs all three.
        # Keyword + transcript points likewise share one transcription pass.
        groups = (
            ("运动与场景", (
                ("场景切换加分：", self._points_row_with_button(
                    self.spin_scene_points, "motion", "运动与场景",
                    "检测列表中每个视频的场景切换与运动，并写入缓存。"
                    "一次分析同时覆盖场景切换、运动事件和运动峰值，不会剪切高光。")),
                ("运动事件加分：", self.spin_motion_event_points),
                ("运动峰值加分：", self.spin_motion_peak),
            )),
            ("音频", (
                ("音频峰值加分：", self._points_row_with_button(
                    self.spin_audio_peak, "audio", "音频",
                    "检测列表中每个视频的音频峰值并写入缓存，不会剪切高光。")),
                ("响度突增加分（相对局部响度）：",
                 self.spin_loudness_burst),
            )),
            # Composition earns a row here rather than living only beside its
            # editor in Advanced. Rules built from signal conditions are an
            # analysis pass like any other on this page — they measure the file
            # and cache a result — and they are the one kind that needs no
            # previous run, so requiring a trip to another tab to start them put
            # the cheapest signal behind the most navigation.
            ("构图规则", (
                ("应用已保存规则：", self._rules_run_row()),
            )),
            ("语音", (
                ("关键词加分（转录中的关键词）：",
                 self.spin_keyword_points),
                ("转录加分（全部语音）：", self._points_row_with_button(
                    self.spin_transcript_points, "transcript", "转录",
                    "转录列表中的每个视频，将结果保存为 _transcript.txt 并写入缓存。"
                    "使用“转录”标签页设置的模型和语言，不会剪切高光。")),
            )),
            ("物体与动作", (
                ("物体加分：", self._points_row_with_button(
                    self.spin_object, "objects", "物体",
                    "在列表中的每个视频中检测下方“物体检测”字段指定的类别并写入缓存。"
                    "不会剪切高光。")),
                ("动作加分：", self._points_row_with_button(
                    self.spin_action, "actions", "动作",
                    "在列表中的每个视频中检测下方“动作关键词”指定的动作并写入缓存。"
                    "留空表示检测全部动作；不会剪切高光。")),
            )),
            ("人脸表情", (
                ("加分及表情选择：", face_row),
            )),
            ("视频位置", (
                ("片头（时间窗、加分）：", intro_widget),
                ("片尾（时间窗、加分）：", outro_widget),
            )),
            ("语音", (
                ("关键词加分（转录中的关键词）：",
                 self.spin_keyword_points),
                ("转录加分（全部语音）：", self._points_row_with_button(
                    self.spin_transcript_points, "transcript", "转录",
                    "转录列表中的每个视频，将结果保存为 _transcript.txt 并写入缓存。"
                    "使用“转录”标签页设置的模型和语言，不会剪切高光。")),
            )),
            ("物体与动作", (
                ("物体加分：", self._points_row_with_button(
                    self.spin_object, "objects", "物体",
                    "在列表中的每个视频中检测下方“物体检测”字段指定的类别并写入缓存。"
                    "不会剪切高光。")),
                ("动作加分：", self._points_row_with_button(
                    self.spin_action, "actions", "动作",
                    "在列表中的每个视频中检测下方“动作关键词”指定的动作并写入缓存。"
                    "留空表示检测全部动作；不会剪切高光。")),
            )),
            ("人脸表情", (
                ("加分及表情选择：", face_row),
            )),
            ("视频位置", (
                ("片头（时间窗、加分）：", intro_widget),
                ("片尾（时间窗、加分）：", outro_widget),
            )),
        )
        for title, rows in groups:
            points_layout.addWidget(self._points_group(title, rows))
        points_layout.addStretch(1)

        points_box.setLayout(points_layout)
        basic_layout.addWidget(points_box, 0, 0, Qt.AlignTop)

        # ── Group 2: Duration & Cutting ──
        duration_box = QGroupBox("时长与剪切")
        duration_layout = QVBoxLayout()

        # Main duration controls (always visible)
        duration_form = QFormLayout()

        self.spin_max_duration = QSpinBox(); self.spin_max_duration.setRange(1,3600); self.spin_max_duration.setValue(highlights_cfg.get("max_duration", 420))
        self.spin_max_duration.setToolTip(
            "高光最长不超过此秒数；若有效内容不足，结果可能更短。"
            "当“精确时长”大于 0 时忽略此项。")
        self.spin_exact_duration = QSpinBox(); self.spin_exact_duration.setRange(0,3600); self.spin_exact_duration.setValue(highlights_cfg.get("exact_duration", 0))
        self.spin_exact_duration.setToolTip(
            "大于 0 时，高光固定为此秒数，并用较低分片段补足。"
            "设为 0 时由“最大高光时长”控制，结果可以更短。")
        self.spin_clip_time = QSpinBox(); self.spin_clip_time.setRange(0,300); self.spin_clip_time.setValue(highlights_cfg.get("clip_time", 10))
        self.spin_clip_time.setToolTip(
            "每个固定窗口围绕得分秒数展开，单位为秒。"
            "设为 0 时使用自动分段（由下方最短/最长片段和合并间隔控制）。")

        duration_form.addRow("最大高光时长（秒）：", self.spin_max_duration)
        duration_form.addRow("精确时长（秒，0 = 关闭）：", self.spin_exact_duration)
        duration_form.addRow("片段时长（秒，0 = 自动）：", self.spin_clip_time)

        # ── Best moments <-> Full story ──
        # How far the cut is allowed to follow the score. Left, it takes the
        # highest-scoring moments wherever they are, which on a video whose
        # action is concentrated can mean the whole cut comes from one stretch.
        # Right, every part of the video contributes.
        self.slider_coverage = QSlider(Qt.Horizontal)
        self.slider_coverage.setRange(0, 100)
        self.slider_coverage.setValue(int(round(float(highlights_cfg.get("coverage", 0.0)) * 100)))
        self.slider_coverage.setTickPosition(QSlider.TicksBelow)
        self.slider_coverage.setTickInterval(25)
        self.slider_coverage.setToolTip(
            "0 表示只选择最高分片段，不考虑它们位于视频何处；"
            "100 表示让视频各部分都获得一定占比；中间值会混合两种策略。"
            "仅在“片段时长”大于 0 时生效，剩余时长仍会由最佳片段补足。")

        coverage_row = QVBoxLayout()
        coverage_row.addWidget(self.slider_coverage)
        self.coverage_hint_label = QLabel("")
        self.coverage_hint_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.coverage_hint_label.setWordWrap(True)
        coverage_row.addWidget(self.coverage_hint_label)

        def on_coverage_changed(value):
            if value <= 5:
                hint = "最佳时刻——只选择最高分部分，不限制它们在视频中的位置。"
            elif value >= 95:
                hint = "完整叙事——视频各部分都会有所体现。"
            else:
                hint = f"完整叙事倾向 {value}%——在保留最佳时刻的同时，让片段分布更均衡。"
            self.coverage_hint_label.setText(hint)

        self.slider_coverage.valueChanged.connect(on_coverage_changed)
        on_coverage_changed(self.slider_coverage.value())

        duration_form.addRow("最佳时刻 ↔ 完整叙事：", coverage_row)

        duration_layout.addLayout(duration_form)

        # Auto-segmentation info label (always visible, updates dynamically)
        self.auto_seg_info_label = QLabel("")
        self.auto_seg_info_label.setStyleSheet("color: #2f81f7; font-style: italic; padding: 4px;")
        self.auto_seg_info_label.setWordWrap(True)
        duration_layout.addWidget(self.auto_seg_info_label)

        # ── Auto-segmentation controls (shown only when clip_time = 0) ──
        self.auto_seg_group = QGroupBox("自动分段设置")
        auto_seg_layout = QFormLayout()

        self.spin_auto_min_clip = QSpinBox()
        self.spin_auto_min_clip.setRange(1, 30)
        self.spin_auto_min_clip.setValue(highlights_cfg.get("auto_min_clip", 2))
        self.spin_auto_min_clip.setToolTip(
            "自动片段的最短时长（秒）。更短的区域会被扩展。"
            "仅在“片段时长”为 0 时生效。")

        self.spin_auto_max_clip = QSpinBox()
        self.spin_auto_max_clip.setRange(3, 120)
        self.spin_auto_max_clip.setValue(highlights_cfg.get("auto_max_clip", 30))
        self.spin_auto_max_clip.setToolTip(
            "自动片段的最长时长（秒）。更长区域会按该时长拆成连续片段。"
            "仅在“片段时长”为 0 时生效。")

        self.spin_auto_merge_gap = QSpinBox()
        self.spin_auto_merge_gap.setRange(0, 10)
        self.spin_auto_merge_gap.setValue(highlights_cfg.get("auto_merge_gap", 2))
        self.spin_auto_merge_gap.setToolTip(
            "间隔小于此秒数的兴趣区域会先合并，再应用最短/最长片段限制。"
            "仅在“片段时长”为 0 时生效。")

        auto_seg_layout.addRow("最短片段（秒）：", self.spin_auto_min_clip)
        auto_seg_layout.addRow("最长片段（秒）：", self.spin_auto_max_clip)
        auto_seg_layout.addRow("合并间隔（秒）：", self.spin_auto_merge_gap)

        self.auto_seg_group.setLayout(auto_seg_layout)
        duration_layout.addWidget(self.auto_seg_group)

        duration_box.setLayout(duration_layout)
        basic_layout.addWidget(duration_box, 0, 1, Qt.AlignTop)

        # ── Connect clip_time spinner to show/hide auto-seg controls ──
        def on_clip_time_changed(value):
            is_auto = (value == 0)
            self.auto_seg_group.setVisible(is_auto)
            if is_auto:
                self.auto_seg_info_label.setText(
                    "🔧 自动模式：程序会根据动作持续时间、场景切换、关键词时间、物体聚集、音频/运动峰值等信号结构自动确定片段边界。"
                )
            else:
                self.auto_seg_info_label.setText(
                    f"✂️ 固定模式：每个高光片段长度为 {value} 秒。"
                )

        self.spin_clip_time.valueChanged.connect(on_clip_time_changed)
        # Trigger once to set initial state
        on_clip_time_changed(self.spin_clip_time.value())

        # Highlight object classes
        obj_layout = QHBoxLayout()
        self.objects_input = QLineEdit(",".join(self.config_data.get("objects", {}).get("interesting", [])))
        self.objects_input.setPlaceholderText("人物,玻璃杯,酒杯,球类")
        obj_layout.addWidget(QLabel("物体检测："))
        obj_layout.addWidget(self.objects_input)
        self.load_objects_btn = QPushButton("加载标签")
        self.load_objects_btn.setToolTip("从 yolo_objects_labels.json 加载标签")
        self.load_objects_btn.clicked.connect(self.open_object_label_selector)
        obj_layout.addWidget(self.load_objects_btn)
        basic_layout.addLayout(obj_layout, 1, 0, 1, 2)

        # Action keywords
        action_kw_layout = QHBoxLayout()
        self.actions_input = QLineEdit(",".join(self.config_data.get("actions", {}).get("interesting", [])))
        self.actions_input.setPlaceholderText("跳高,高踢,射箭")
        action_kw_layout.addWidget(QLabel("动作关键词："))
        action_kw_layout.addWidget(self.actions_input)
        self.load_actions_btn = QPushButton("加载标签")
        self.load_actions_btn.setToolTip("从 kinetics_400_labels.json（或自定义 Intel 模型）加载标签")
        self.load_actions_btn.clicked.connect(self.open_action_label_selector)
        action_kw_layout.addWidget(self.load_actions_btn)
        basic_layout.addLayout(action_kw_layout, 2, 0, 1, 2)

        # Transcript search keywords — moved here from the Transcript tab: it's a
        # common highlight signal (score moments where these words are spoken).
        # Greyed out unless transcript processing is enabled (nothing to search
        # otherwise); tracks the transcript toggle via on_transcript_toggle.
        _kw_enabled = self.config_data.get("transcript", {}).get("enabled", False)
        kw_layout = QHBoxLayout()
        self.search_keywords_input = QLineEdit(",".join(self.config_data.get("transcript", {}).get("search_keywords", [])))
        self.search_keywords_input.setPlaceholderText("进球,得分,获胜")
        self.search_keywords_input.setToolTip("为出现这些语音关键词的时刻加分（需要启用转录）")
        self.search_keywords_input.setEnabled(_kw_enabled)
        self.search_keywords_label = QLabel("转录关键词：")
        self.search_keywords_label.setEnabled(_kw_enabled)
        kw_layout.addWidget(self.search_keywords_label)
        kw_layout.addWidget(self.search_keywords_input)
        basic_layout.addLayout(kw_layout, 3, 0, 1, 2)

        # Conditional action scoring checkbox
        self.actions_require_objects_chk = QCheckBox("仅在检测到物体时给动作加分")
        self.actions_require_objects_chk.setChecked(self.config_data.get("actions", {}).get("require_objects", False))
        self.actions_require_objects_chk.setToolTip("只有同一时间段也检测到物体时，动作才会获得加分")
        basic_layout.addWidget(self.actions_require_objects_chk, 4, 0, 1, 2)

        # (The old "Skip highlights" checkbox is gone. Producing a transcript or
        # subtitles without cutting highlights is now a per-signal run button in
        # the Scoring Points panel above — Transcribe/Objects/Actions/etc. — so
        # scores never have to be zeroed by hand.)

        # Combine every processed video's highlights into one master video.
        # Config key stays under "download" (auto_combine) so saved configs load.
        self.auto_combine_chk = QCheckBox("将所有已处理视频的高光合并为一个视频")
        self.auto_combine_chk.setChecked(self.config_data.get("download", {}).get("auto_combine", True))
        self.auto_combine_chk.setToolTip("启用后，会将所有已处理视频的高光片段合并为一个总视频")
        basic_layout.addWidget(self.auto_combine_chk, 5, 0, 1, 2)

        # Equal-width columns; trailing stretch row keeps groups packed at the top.
        basic_layout.setColumnStretch(0, 1)
        basic_layout.setColumnStretch(1, 1)
        basic_layout.setRowStretch(6, 1)

        basic_tab.setLayout(basic_layout)
        tabs.addTab(self._scrollable(basic_tab), "基础设置")

        # --- Tab 2: Transcript & Subtitles ---
        transcript_cfg = self.config_data.get("transcript", {})
        subtitles_cfg = self.config_data.get("subtitles", {})

        transcript_tab = QWidget()
        transcript_layout = QVBoxLayout()

        transcript_group = QGroupBox("转录设置")
        transcript_form = QFormLayout()
        self.transcript_checkbox = QCheckBox("启用转录处理")
        self.transcript_checkbox.setChecked(transcript_cfg.get("enabled", False))
        self.transcript_checkbox.toggled.connect(self.on_transcript_toggle)
        transcript_form.addRow("启用转录：", self.transcript_checkbox)

        # Source language for transcription
        self.transcript_source_lang = QComboBox()
        self.transcript_source_lang.addItems(["auto","en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.transcript_source_lang.setCurrentText(transcript_cfg.get("source_lang", "en"))
        self.transcript_source_lang.setEnabled(transcript_cfg.get("enabled", False))
        transcript_form.addRow("源语言：", self.transcript_source_lang)

        self.transcript_model_combo = QComboBox()
        self.transcript_model_combo.addItems(["tiny","base","small","medium","large"])
        self.transcript_model_combo.setCurrentText(transcript_cfg.get("model", "base"))
        self.transcript_model_combo.setEnabled(transcript_cfg.get("enabled", False))
        transcript_form.addRow("Whisper 模型：", self.transcript_model_combo)

        # (Search keywords moved to the Basic Settings tab — a common highlight
        # signal, editable without opening this tab.)
        transcript_group.setLayout(transcript_form)
        transcript_layout.addWidget(transcript_group)

        subtitle_group = QGroupBox("字幕设置")
        subtitle_form = QFormLayout()
        self.subtitles_checkbox = QCheckBox("生成字幕（.srt）")
        self.subtitles_checkbox.setChecked(subtitles_cfg.get("enabled", False))
        self.subtitles_checkbox.toggled.connect(self.on_subtitles_toggle)
        # Disable subtitle checkbox if transcript is not enabled
        self.subtitles_checkbox.setEnabled(transcript_cfg.get("enabled", False))
        subtitle_form.addRow("生成字幕：", self.subtitles_checkbox)

        # No "source language" here. What is spoken is a property of the video,
        # it is already declared in Transcript Settings above (which is what
        # Whisper is actually told), and asking twice only let the two answers
        # disagree — a subtitle box saying "en" never made Russian audio English,
        # it just mislabelled the translation and named the file wrong. When a
        # cached transcript is reused, its own recorded language is used.
        self.subtitle_target_lang = QComboBox()
        self.subtitle_target_lang.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.subtitle_target_lang.setCurrentText(subtitles_cfg.get("target_lang", "pl"))
        self.subtitle_target_lang.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("目标语言：", self.subtitle_target_lang)
        _sub_run = self._make_analyze_button(
            "subtitles", "生成字幕",
            "转录列表中的每个视频，并在视频旁生成 .srt 字幕文件；"
            "当目标语言不同时会自动翻译。此操作不会剪切高光片段。")
        subtitle_form.addRow("", _sub_run)
        subtitle_group.setLayout(subtitle_form)
        transcript_layout.addWidget(subtitle_group)

        transcript_tab.setLayout(transcript_layout)
        tabs.addTab(self._scrollable(transcript_tab), "转录与字幕")

        # --- Tab 3: Advanced Tab ---
        advanced_cfg = self.config_data.get("advanced", {})
        visualization_cfg = self.config_data.get("visualization", {})

        advanced_tab = QWidget()
        # Grid so the small groups sit side by side and use horizontal space
        # (especially when maximized) instead of one tall scrolling column.
        advanced_layout = QGridLayout()

        # ── Group 1: Motion Recognition ──
        motion_box = QGroupBox("运动识别")
        motion_layout = QFormLayout()

        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 30)
        self.frame_skip_spin.setValue(advanced_cfg.get("frame_skip", 5))
        self.frame_skip_spin.setToolTip(
            "运动和场景检测每隔 N 帧读取一次。"
            "数值越大速度越快，但可能漏掉短暂切镜或瞬间运动。")

        motion_layout.addRow("帧间隔：", self.frame_skip_spin)
        self.vr_mode_chk = QCheckBox("VR 左右并排优化")
        self.vr_mode_chk.setChecked(bool(advanced_cfg.get("vr_mode", False)))
        self.vr_mode_chk.setToolTip(
            "对于左右并排的 VR/3D 视频，仅分析画面左半部分。"
        )
        motion_layout.addRow("", self.vr_mode_chk)
        motion_box.setLayout(motion_layout)
        advanced_layout.addWidget(motion_box, 1, 0)

        # ── Group 2: Object Recognition ──
        object_box = QGroupBox("物体识别")
        object_layout = QFormLayout()

        self.obj_frame_skip_spin = QSpinBox()
        self.obj_frame_skip_spin.setRange(1, 60)
        self.obj_frame_skip_spin.setValue(advanced_cfg.get("object_frame_skip", 10))
        self.obj_frame_skip_spin.setToolTip(
            "物体检测每隔 N 帧采样一次。"
            "数值越大速度越快，但可能漏掉只短暂出现在画面中的物体。")

        self.yolo_type_combo = QComboBox()
        self.yolo_type_combo.addItem("标准 YOLOX（80 类物体）", "standard")

        # Custom keypoint models are unsupported: their only trainer was AGPL.
        self._custom_pose_model = None

        self.yolo_model_combo = QComboBox()

        # Object model selector: standard COCO / Custom / Mixed, auto-discovered
        # from models/custom/. Run by the YOLOX runtime; class names come from
        # each model's metadata or the labels.json beside it.
        self.object_model_combo = QComboBox()
        self.object_model_combo.setToolTip(
            "标准——80 类 COCO 物体\n"
            "自定义——你训练的模型（自动从 models/custom/ 检测）\n"
            "混合——同时使用标准检测器和你的自定义模型")

        import_obj_btn = QPushButton("导入模型…")
        import_obj_btn.setToolTip("将训练好的模型（.onnx / OpenVINO .xml）复制到 models/custom/")
        obj_model_row = QHBoxLayout()
        obj_model_row.setContentsMargins(0, 0, 0, 0)
        obj_model_row.addWidget(self.object_model_combo, 1)
        obj_model_row.addWidget(import_obj_btn)
        community_btn = QPushButton("社区模型…")
        community_btn.setToolTip(
            "浏览并安装其他用户训练并分享的小型检测模型。"
            "模型托管在 Hugging Face，安装前会进行安全与兼容性检查。")
        obj_model_row.addWidget(community_btn)
        self.object_model_widget = QWidget()
        self.object_model_widget.setLayout(obj_model_row)

        def _populate_object_models(select_type=None, select_path=""):
            """Rebuild the combo from discovery. Each entry's data is
            (yolo_type, path), matching what the pipeline consumes."""
            from modules.system.app_paths import discover_object_models
            self.object_model_combo.blockSignals(True)
            self.object_model_combo.clear()
            self.object_model_combo.addItem("标准（80 类物体）", ("standard", ""))
            models = []
            try:
                models = discover_object_models()
            except Exception as e:
                print(f"⚠️ object model discovery failed: {e}")
            for m in models:
                n = len(m["classes"])
                kind = "社区" if m.get("community") else "自定义"
                self.object_model_combo.addItem(
                    f"{kind} — {m['name']}（{n} 个类别）", ("custom", m["path"]))
            for m in models:
                n = len(m["classes"])
                self.object_model_combo.addItem(
                    f"混合——标准 + {m['name']}（80 + {n}）", ("custom_mixed", m["path"]))

            # Restore selection by (type, path); fall back to standard.
            target = (select_type or "standard", select_path or "")
            idx = next((i for i in range(self.object_model_combo.count())
                        if self.object_model_combo.itemData(i) == target), 0)
            self.object_model_combo.setCurrentIndex(idx)
            self.object_model_combo.blockSignals(False)

        def _import_object_model():
            from modules.system.app_paths import import_object_model
            src, _ = QFileDialog.getOpenFileName(
                self, "导入物体检测模型", "",
                "检测模型 (*.onnx *.xml);;所有文件 (*)")
            if not src:
                return
            try:
                dst = import_object_model(src)
                _populate_object_models(select_type="custom", select_path=dst)
                self.append_log(f"✅ 已导入物体模型：{os.path.basename(dst)}")
            except Exception as e:
                self.append_log(f"⚠️ 物体模型导入失败：{e}")

        import_obj_btn.clicked.connect(_import_object_model)

        def _browse_community_models():
            try:
                from model_hub.gui import ModelBrowserDialog
            except Exception as e:
                self.append_log(f"⚠️ 社区模型不可用：{e}")
                return
            dialog = ModelBrowserDialog(self)

            def _on_installed(model):
                _populate_object_models(select_type="custom", select_path=str(model.model_path))
                self.append_log(f"✅ 已安装社区模型：{model.manifest.display_name}")

            dialog.installed.connect(_on_installed)
            dialog.exec()

        community_btn.clicked.connect(_browse_community_models)

        def on_object_model_changed(index=0):
            yolo_type = self.object_detector_choice()[0]
            prev_size = self.yolo_model_combo.currentData()
            self.yolo_model_combo.blockSignals(True)
            self.yolo_model_combo.clear()

            # Mixed still runs the standard detector, so the size stays live;
            # only custom-only makes it moot.
            custom_only = (yolo_type == "custom")

            if custom_only:
                # Size applies to the standard detector, which isn't used here
                self.yolo_model_combo.addItem("（自定义模型——尺寸未知）", "n")
                self.yolo_model_combo.setEnabled(False)
            else:
                self.yolo_model_combo.addItem("Nano（最快，精度最低）", "n")
                self.yolo_model_combo.addItem("Small（较快，平衡良好）", "s")
                self.yolo_model_combo.addItem("Medium（均衡）", "m")
                self.yolo_model_combo.addItem("Large（更准确，较慢）", "l")
                self.yolo_model_combo.addItem("Extra-Large（最准确，最慢）", "x")
                self.yolo_model_combo.setEnabled(True)

            restore_idx = self.yolo_model_combo.findData(prev_size)
            if restore_idx >= 0:
                self.yolo_model_combo.setCurrentIndex(restore_idx)
            self.yolo_model_combo.blockSignals(False)

        _populate_object_models(
            select_type=advanced_cfg.get("yolo_type", "standard"),
            select_path=advanced_cfg.get("yolo_custom_model_path", "") or "",
        )
        self.object_model_combo.currentIndexChanged.connect(on_object_model_changed)

        current_model = advanced_cfg.get("yolo_model_size", "n")
        on_object_model_changed()
        idx = self.yolo_model_combo.findData(current_model)
        self.yolo_model_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.obj_confidence_spin = QSpinBox()
        self.obj_confidence_spin.setRange(5, 95)
        self.obj_confidence_spin.setSuffix("%")
        self.obj_confidence_spin.setValue(int(self.config_data.get("objects", {}).get("confidence", 30)))
        self.obj_confidence_spin.setToolTip("物体检测最低置信度阈值（越低检测越多，但误报也越多）")

        object_layout.addRow("帧间隔：", self.obj_frame_skip_spin)
        object_layout.addRow("检测器类型：", self.yolo_type_combo)
        object_layout.addRow("检测模型大小：", self.yolo_model_combo)
        object_layout.addRow("物体模型：", self.object_model_widget)
        object_layout.addRow("置信度阈值：", self.obj_confidence_spin)

        object_box.setLayout(object_layout)
        advanced_layout.addWidget(object_box, 2, 0)

        # ── Group 3: Action Recognition ──
        action_box = QGroupBox("动作识别")
        action_layout = QFormLayout()

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1, 30)
        self.sample_rate_spin.setValue(advanced_cfg.get("sample_rate", 5))
        self.sample_rate_spin.setToolTip("动作识别片段每隔 N 帧采样一次")

        self.action_backend_combo = QComboBox()
        # "Auto" has picked DirectML since R3D learned to run through ONNX
        # Runtime; the old label predated that and named three of the four.
        self.action_backend_combo.addItem(
            "自动（优先使用已训练动作头，否则使用旧模型）", "auto")
        self.action_backend_combo.addItem(
            "SigLIP2 + 你训练的动作头", "siglip2")
        # The Intel and R3D paths are being retired in favour of SigLIP2.
        self.action_backend_combo.addItem(
            "OpenVINO Intel 模型（已弃用）", "openvino")
        self.action_backend_combo.addItem("R3D + CUDA（已弃用）", "r3d_cuda")
        self.action_backend_combo.addItem("R3D + DirectML（已弃用）", "r3d_dml")
        self.action_backend_combo.addItem("R3D + CPU（已弃用）", "r3d_cpu")
        current_backend = advanced_cfg.get("action_backend", "auto")
        idx_ab = self.action_backend_combo.findData(current_backend)
        self.action_backend_combo.setCurrentIndex(idx_ab if idx_ab >= 0 else 0)

        self._intel_count = len(self.load_labels_from_json(KINETICS_400_LABELS_FILE)) if os.path.exists(KINETICS_400_LABELS_FILE) else 0
        self._custom_ov_count = len(self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)) if os.path.exists(INTEL_CUSTOM_LABELS_FILE) else 0
        self._r3d_custom_count = len(self.load_labels_from_json(R3D_CUSTOM_LABELS_FILE)) if os.path.exists(R3D_CUSTOM_LABELS_FILE) else 0

        self.action_models_combo = QComboBox()

        import_action_btn = QPushButton("导入模型…")
        import_action_btn.setToolTip(
            "将训练好的自定义动作模型复制到应用的自定义模型位置：\n"
            "  • OpenVINO 解码器（.xml + .bin）\n"
            "  • R3D 微调权重（.pth）\n"
            "如果旁边存在同名 .json（标签 / 映射）会自动读取；否则会提示你选择。")
        action_model_row = QHBoxLayout()
        action_model_row.addWidget(self.action_models_combo, 1)
        action_model_row.addWidget(import_action_btn)
        action_model_widget = QWidget()
        action_model_widget.setLayout(action_model_row)

        def _import_action_model():
            src, _ = QFileDialog.getOpenFileName(
                self, "导入自定义动作模型", "",
                "动作模型 (*.xml *.pth);;OpenVINO IR (*.xml);;"
                "R3D 权重 (*.pth);;所有文件 (*)")
            if not src:
                return
            is_r3d = src.lower().endswith(".pth")
            labels_src = ""
            if not os.path.exists(os.path.splitext(src)[0] + ".json"):
                prompt = ("R3D 映射文件（idx_to_label + 元数据 JSON）" if is_r3d
                          else "该解码器的标签文件（idx_to_label JSON）")
                labels_src, _ = QFileDialog.getOpenFileName(
                    self, prompt, "", "JSON (*.json);;所有文件 (*)")
            try:
                # Fresh re-resolution (not the frozen *_LABELS_FILE constants) so
                # the newly imported model's class count shows up immediately,
                # without requiring an app restart.
                if is_r3d:
                    from modules.system.app_paths import (
                        import_r3d_action_model, r3d_custom_action_paths)
                    n_classes, variant = import_r3d_action_model(src, labels_src)
                    if n_classes == 0:
                        print("⚠️ R3D model imported without a mapping file — it won't "
                              "be usable until one is provided")
                    elif not variant:
                        print("⚠️ R3D mapping has no model_variant — the loader will use "
                              "the 'R3D model variant' dropdown selection")
                    fresh = r3d_custom_action_paths()[1]
                    self._r3d_custom_count = (
                        len(self.load_labels_from_json(fresh)) if os.path.exists(fresh) else 0)
                    select_mode = "r3d_custom_only"
                else:
                    from modules.system.app_paths import (
                        import_custom_action_model, custom_action_decoder_paths)
                    n_classes = import_custom_action_model(src, labels_src)
                    if n_classes == 0:
                        print("⚠️ Custom action decoder imported without a labels file "
                              "— it won't be usable until one is provided")
                    fresh = custom_action_decoder_paths()[2]
                    self._custom_ov_count = (
                        len(self.load_labels_from_json(fresh)) if os.path.exists(fresh) else 0)
                    select_mode = "custom_only"

                on_action_backend_changed(0)
                idx = self.action_models_combo.findData(select_mode)
                if idx < 0 and n_classes:
                    # The mode isn't offered under the current Backend, so switch to
                    # one that enables the just-imported model — preferring GPU —
                    # instead of leaving the user to hunt through the dropdown:
                    #   • R3D custom needs an R3D backend. "Auto" would *disable* R3D
                    #     on a non-CUDA machine (see pipeline.py), which would make
                    #     r3d_custom_only fail to load — so pick r3d_cuda when an
                    #     NVIDIA GPU is present, else r3d_cpu (slow, but it runs).
                    #   • OpenVINO custom → "Auto" (lists it, and uses the Intel
                    #     GPU / CPU at runtime).
                    if select_mode == "r3d_custom_only":
                        try:
                            from modules.system.device_utils import detect_best_device
                            has_cuda = detect_best_device(
                                log_fn=lambda *a, **k: None).pytorch_device == "cuda"
                        except Exception:
                            has_cuda = False
                        target_backend = "r3d_cuda" if has_cuda else "r3d_cpu"
                    else:
                        target_backend = "auto"
                    # Setting the combo fires on_action_backend_changed, which
                    # rebuilds the models list, so re-query the index afterward.
                    ab_idx = self.action_backend_combo.findData(target_backend)
                    if ab_idx >= 0:
                        self.action_backend_combo.setCurrentIndex(ab_idx)
                        idx = self.action_models_combo.findData(select_mode)
                if idx >= 0:
                    self.action_models_combo.setCurrentIndex(idx)
            except Exception as e:
                print(f"⚠️ action model import failed: {e}")

        import_action_btn.clicked.connect(_import_action_model)

        self.r3d_model_combo = QComboBox()
        self.r3d_model_combo.addItem("R3D-18（最快）", "r3d_18")
        self.r3d_model_combo.addItem("MC3-18（混合卷积）", "mc3_18")
        self.r3d_model_combo.addItem("R(2+1)D-18（最准确）", "r2plus1d_18")
        current_r3d = advanced_cfg.get("r3d_model", "r3d_18")
        idx_r3d = self.r3d_model_combo.findData(current_r3d)
        self.r3d_model_combo.setCurrentIndex(idx_r3d if idx_r3d >= 0 else 0)

        def on_action_backend_changed(index):
            backend = self.action_backend_combo.currentData()
            self.r3d_model_combo.setEnabled(backend in ("auto", "r3d_cuda", "r3d_cpu"))

            prev_data = self.action_models_combo.currentData()
            self.action_models_combo.blockSignals(True)
            self.action_models_combo.clear()
            self.action_models_combo.setEnabled(backend != "siglip2")

            if backend == "siglip2":
                from modules.vision import action_siglip
                head = action_siglip.installed_head_classes()
                self.action_models_combo.addItem(
                    f"动作头：{head[0]}（{len(head[1])} 个类别）" if head
                    else "未安装动作头——请先训练一个", "siglip2")
            elif backend in ("openvino",):
                if self._intel_count:
                    self.action_models_combo.addItem(f"Intel Kinetics-400（{self._intel_count} 个类别）", "intel_only")
                if self._custom_ov_count:
                    self.action_models_combo.addItem(f"自定义 OpenVINO（{self._custom_ov_count} 个类别）", "custom_only")
                if self._intel_count and self._custom_ov_count:
                    total = self._intel_count + self._custom_ov_count
                    self.action_models_combo.addItem(f"混合——两个解码器（{total} 个类别）", "mixed")
            elif backend in ("r3d_cuda", "r3d_cpu"):
                if self._intel_count:
                    self.action_models_combo.addItem(f"R3D Kinetics-400 预训练（{self._intel_count} 个类别）", "intel_only")
                if self._r3d_custom_count:
                    self.action_models_combo.addItem(f"R3D 微调模型（{self._r3d_custom_count} 个类别）", "r3d_custom_only")
                if self._intel_count and self._r3d_custom_count:
                    total = self._intel_count + self._r3d_custom_count
                    self.action_models_combo.addItem(f"混合——两个 R3D 模型（{total} 个类别）", "mixed")
            else:
                if self._intel_count:
                    self.action_models_combo.addItem(f"Intel Kinetics-400（{self._intel_count} 个类别）", "intel_only")
                if self._custom_ov_count:
                    self.action_models_combo.addItem(f"自定义 OpenVINO（{self._custom_ov_count} 个类别）", "custom_only")
                if self._r3d_custom_count:
                    self.action_models_combo.addItem(f"R3D 微调模型（{self._r3d_custom_count} 个类别）", "r3d_custom_only")
                available = sum(1 for c in [self._intel_count, self._custom_ov_count, self._r3d_custom_count] if c > 0)
                if available >= 2:
                    total = self._intel_count + self._custom_ov_count + self._r3d_custom_count
                    self.action_models_combo.addItem(f"混合——全部模型（{total} 个类别）", "mixed")

            restore_idx = self.action_models_combo.findData(prev_data)
            if restore_idx >= 0:
                self.action_models_combo.setCurrentIndex(restore_idx)
            self.action_models_combo.blockSignals(False)
            self.update_actions_completer()

        self.action_backend_combo.currentIndexChanged.connect(on_action_backend_changed)
        self.action_models_combo.currentIndexChanged.connect(lambda: self.update_actions_completer())
        on_action_backend_changed(0)
        current_action_models = advanced_cfg.get("action_models", "mixed")
        restore_idx = self.action_models_combo.findData(current_action_models)
        if restore_idx >= 0:
            self.action_models_combo.setCurrentIndex(restore_idx)

        action_layout.addRow("帧间隔：", self.sample_rate_spin)
        action_layout.addRow("运行后端：", self.action_backend_combo)
        action_layout.addRow("模型：", action_model_widget)
        action_layout.addRow("R3D 模型变体：", self.r3d_model_combo)

        action_box.setLayout(action_layout)
        advanced_layout.addWidget(action_box, 2, 1)

        # ── Group 4: Bounding Box Visualization ──
        # ── Group 4: Composition Rules ──
        comp_box = QGroupBox("构图规则")
        comp_outer = QVBoxLayout()

        comp_info = QLabel(
            "根据检测物体之间的空间关系组合更高层级的事件。"
            "例如：物体 A 在区域 B 内出现达到一定次数后触发事件 X。"
            "每一行代表一个条件；同一事件名称下的多行条件必须全部满足（AND 逻辑）。"
            "“窗口”表示用于平滑检测结果的时间长度，可减少闪烁；"
            "“保持”表示 YOLO 暂时丢失目标后继续保留该目标的时间，用于处理遮挡。"
            "规则保存在应用旁边的 composition_rules.yaml。"
        )
        comp_info.setWordWrap(True)
        comp_info.setStyleSheet("color: #888; font-size: 9pt;")
        comp_outer.addWidget(comp_info)

        # Table: Event Name | Label | Source | Region | Min | Max | Window | Persist | [Del]
        # Two kinds of condition share this table. A spatial one is geometry
        # (this class inside that class); a signal one is a threshold on a
        # per-second measurement. They need different fields, and the earlier
        # version had no row shape for the second kind — so signal rules were
        # invisible here and could only be edited as YAML.
        self.COMP_MIN_UNSET = -9999.0
        self.COMP_MAX_UNSET = 9999.0
        # Relation and Outline sit before the delete button; named so no
        # lookup below depends on remembering where they went.
        COMP_REL_COL, COMP_OUTLINE_COL, COMP_DEL_COL = 12, 13, 14
        self.comp_table = QTableWidget(0, 15)
        self.comp_table.setHorizontalHeaderLabels([
            "启用", "类型", "事件名称", "显示名称",
            "物体 / 信号", "区域 / 等于",
            "最小", "最大", "持续（秒）", "范围（秒）",
            "窗口（秒）", "保持（秒）", "关系", "轮廓", "",
        ])
        # chr(10) rather than an escape: this block is generated, and a
        # literal backslash-n did not survive the round trip intact.
        self.comp_table.horizontalHeader().setToolTip(chr(10).join([
            "启用：取消勾选可保留规则，但停止执行",
            "类型 - 空间：物体必须位于指定区域内。",
            "       信号：测量值必须位于最小值和最大值之间。",
            "最小/最大：空间规则表示次数，信号规则表示阈值；",
            "       保持极端默认值时表示无上下界，不会写入配置。",
            "区域 / 等于：空间规则中的容器类别，或信号规则必须等于的标签。",
            "持续：条件必须连续满足多少秒（信号规则）。",
            "范围：允许条件在前后多少秒内满足，用于不同采样时刻的信号。",
            "窗口：用于平滑帧结果的秒数，可减少闪烁。",
            "保持：来源消失后继续保留多少秒。",
            "关系（空间）：inside = 中心点位于区域内；",
            "       overlaps = 大部分区域重叠；touches = 两者接触。",
            "轮廓（空间）：使用检测框内的真实形状进行判断。",
            "       首次运行较慢，结果会缓存供后续使用。",
        ]))
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.comp_table.horizontalHeader().setStretchLastSection(False)
        for _c, _w in ((0, 34), (1, 78), (2, 140), (3, 140), (4, 150), (5, 120),
                       (6, 70), (7, 70), (8, 80), (9, 80), (10, 80), (11, 80),
                       (COMP_REL_COL, 92), (COMP_OUTLINE_COL, 60), (COMP_DEL_COL, 32)):
            self.comp_table.setColumnWidth(_c, _w)
        self.comp_table.setMinimumHeight(160)
        self.comp_table.setMaximumHeight(280)
        self.comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        comp_outer.addWidget(self.comp_table)

        comp_btn_row = QHBoxLayout()
        comp_add_btn = QPushButton("+ 添加空间条件")
        comp_add_btn.setToolTip("添加物体几何条件，例如：某个类别位于另一个类别内部")
        comp_add_signal_btn = QPushButton("+ 添加信号条件")
        comp_add_signal_btn.setToolTip(
            "添加基于每秒测量值的条件，例如 vocal_density_pct 或 "
            "waveform_peak_density。此类规则不需要物体检测。")
        comp_save_btn = QPushButton("保存规则")
        comp_save_btn.setToolTip("将构图规则保存到 composition_rules.yaml")
        comp_btn_row.addWidget(comp_add_btn)
        comp_btn_row.addWidget(comp_add_signal_btn)
        comp_btn_row.addStretch()
        # Run lives here, beside the editor, because this is where rules are
        # changed. Applying them is seconds against what is already cached; the
        # alternative was a full pipeline run just to see a rule edit.
        comp_btn_row.addWidget(self._make_analyze_button(
            "composition", "运行规则",
            "对列表中的每个视频运行已勾选的规则。\n\n"
            "如果规则所需的数据尚未缓存，会自动补充："
            "缺少指定物体类别时会启动检测；信号规则会直接测量音频。\n"
            "这些结果都会缓存，因此修改阈值后再次运行通常只需几秒。\n\n"
            "请先保存规则。未勾选的规则会跳过且不产生额外开销。"
            "可以安全重复运行；旧结果会被替换，而不是重复叠加。"))
        comp_btn_row.addWidget(comp_save_btn)
        comp_outer.addLayout(comp_btn_row)

        comp_box.setLayout(comp_outer)
        advanced_layout.addWidget(comp_box, 4, 0, 1, 2)

        # ---- load existing rules into table ----
        def _comp_load_rules():
            from modules.system.app_paths import composition_rules_path, user_data_dir
            path = composition_rules_path()
            events = []
            self._comp_top = {}
            if path:
                try:
                    with open(path, encoding='utf-8') as _f:
                        _raw = yaml.safe_load(_f) or {}
                    events = _raw.get('events', [])
                    # Top-level settings (the outliner) have no place in the
                    # table either; kept and written back as they were.
                    from modules.rules.rules_file import top_level_fields
                    self._comp_top = top_level_fields(_raw)
                except Exception:
                    pass
            self.comp_table.setRowCount(0)
            # Only an event with neither kind of condition is unrepresentable
            # now. Kept aside and written back untouched on save, because the
            # table rebuilds this file from its own rows and would otherwise
            # delete it silently.
            self._comp_passthrough = [ev for ev in events
                                      if not ev.get('rules') and not ev.get('signals')]
            # The whole original entry, by name. Saving rebuilds this file from
            # the table's rows, so any field without a column is lost — which is
            # how `min_duration_secs` and `ignore_edges_secs` disappeared the
            # first time somebody pressed Save after they were added. Keeping
            # the original and overwriting only what the table owns means a
            # field added later survives without needing a column first.
            self._comp_original = {str(ev.get('名称', '')): dict(ev)
                                   for ev in events if ev.get('名称')}
            for ev in events:
                common = dict(
                    ev_name=ev.get('名称', ''),
                    ev_label=ev.get('label', ev.get('名称', '')),
                    window=ev.get('window_secs', 0.75),
                    persist=ev.get('persist_secs', 0.5),
                    enabled=bool(ev.get('enabled', True)),
                )
                for rule in ev.get('rules', []) or []:
                    _comp_add_table_row(
                        kind='Spatial',
                        source=rule.get('source', ''),
                        region=rule.get('region', ''),
                        min_c=rule.get('min_count', 1),
                        max_c=rule.get('max_count', 999),
                        relation=rule.get('relation', 'inside'),
                        outline=bool(rule.get('outline', False)),
                        **common)
                for cond in ev.get('signals', []) or []:
                    equals = cond.get('equals')
                    if equals is None and cond.get('any_of'):
                        # any_of has no column of its own; showing the first
                        # value would quietly drop the rest, so the row is
                        # rendered read-only-ish by leaving Equals blank and the
                        # condition is preserved through passthrough instead.
                        equals = None
                    _comp_add_table_row(
                        kind='Signal',
                        source=str(cond.get('signal', '')),
                        region='' if equals is None else str(equals),
                        min_c=cond.get('min'),
                        max_c=cond.get('max'),
                        sustain=cond.get('sustained_secs', 0),
                        within=cond.get('within_secs', 0),
                        **common)
            if self._comp_passthrough:
                names = ', '.join(str(ev.get('名称', '?'))
                                  for ev in self._comp_passthrough)
                # print(), not append_log(): this runs from __init__, before the
                # log pane exists, and calling it there took the whole
                # application down before its first window.
                print(f"构图规则：{len(self._comp_passthrough)} 条规则没有条件（{names}）；保存时会原样保留。")


        def _comp_kind_of(row):
            combo = self.comp_table.cellWidget(row, 1)
            return str((combo.currentData() if combo and combo.currentData() is not None else (combo.currentText() if combo else "Spatial"))).strip()

        def _comp_apply_kind(row):
            """Grey the cells the chosen kind does not use.

            Left editable they would look like fields that simply had not been
            filled in, and a spatial rule carrying a Sustain value that is
            silently dropped on save is worse than one that never offered it.
            """
            signal = _comp_kind_of(row) == "Signal"
            for col in (8, 9):                       # Sustain / Within
                w = self.comp_table.cellWidget(row, col)
                if w:
                    w.setEnabled(signal)
            for col in (10, 11):                     # Window / Persist
                w = self.comp_table.cellWidget(row, col)
                if w:
                    w.setEnabled(True)
            for col in (COMP_REL_COL, COMP_OUTLINE_COL):   # shapes: spatial only
                w = self.comp_table.cellWidget(row, col)
                if w:
                    w.setEnabled(not signal)
            head = self.comp_table.horizontalHeaderItem(4)
            if head:
                head.setText("物体 / 信号")

        def _comp_add_table_row(ev_name='', ev_label='', source='', region='',
                                min_c=None, max_c=None, window=0.75, persist=0.5,
                                enabled=True, kind='Spatial',
                                sustain=0, within=0, relation='inside',
                                outline=False):
            r = self.comp_table.rowCount()
            self.comp_table.insertRow(r)

            # Enabled is a property of the *event*, and one event can occupy
            # several rows. Toggling any of them moves the rest, so the table
            # cannot be left saying a rule is both on and off.
            on_chk = QCheckBox()
            on_chk.setChecked(bool(enabled))
            on_chk.setToolTip("运行此规则。取消勾选后仍会保留在文件中，但不会再参与匹配。")
            def _sync(state, box=on_chk):
                row = next((i for i in range(self.comp_table.rowCount())
                            if self.comp_table.cellWidget(i, 0) is box), None)
                if row is None:
                    return
                item = self.comp_table.item(row, 2)
                name = item.text().strip() if item else ''
                if not name:
                    return
                for i in range(self.comp_table.rowCount()):
                    other = self.comp_table.cellWidget(i, 0)
                    twin = self.comp_table.item(i, 2)
                    if other is None or other is box or twin is None:
                        continue
                    if twin.text().strip() == name and other.isChecked() != box.isChecked():
                        other.blockSignals(True)
                        other.setChecked(box.isChecked())
                        other.blockSignals(False)
            on_chk.stateChanged.connect(_sync)
            # In the cell directly, not inside a centring wrapper: every lookup
            # finds it with cellWidget(row, 0), and a wrapper would return the
            # wrapper instead.
            self.comp_table.setCellWidget(r, 0, on_chk)

            kind_combo = QComboBox()
            kind_combo.addItem("空间", "Spatial")
            kind_combo.addItem("信号", "Signal")
            _kind_value = "Signal" if str(kind) == "Signal" else "Spatial"
            kind_combo.setCurrentIndex(max(0, kind_combo.findData(_kind_value)))
            def _on_kind(_i, box=kind_combo):
                row = next((i for i in range(self.comp_table.rowCount())
                            if self.comp_table.cellWidget(i, 1) is box), None)
                if row is not None:
                    _comp_apply_kind(row)
            kind_combo.currentIndexChanged.connect(_on_kind)
            self.comp_table.setCellWidget(r, 1, kind_combo)

            self.comp_table.setItem(r, 2, QTableWidgetItem(ev_name))
            self.comp_table.setItem(r, 3, QTableWidgetItem(ev_label))
            self.comp_table.setItem(r, 4, QTableWidgetItem(source))
            self.comp_table.setItem(r, 5, QTableWidgetItem(region))

            # One widget type for both kinds. Counts are whole numbers and
            # thresholds are not, and a spin box per kind would have to be
            # rebuilt every time the Kind cell changed; the save path rounds
            # counts back to integers instead.
            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(2)
            min_spin.setRange(self.COMP_MIN_UNSET, self.COMP_MAX_UNSET)
            min_spin.setValue(self.COMP_MIN_UNSET if min_c is None else float(min_c))
            min_spin.setToolTip("位于最小值时表示“不设下限”，保存时不会写入该限制。")
            self.comp_table.setCellWidget(r, 6, min_spin)

            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(2)
            max_spin.setRange(self.COMP_MIN_UNSET, self.COMP_MAX_UNSET)
            max_spin.setValue(self.COMP_MAX_UNSET if max_c is None else float(max_c))
            max_spin.setToolTip("位于最大值时表示“不设上限”，保存时不会写入该限制。")
            self.comp_table.setCellWidget(r, 7, max_spin)

            sus_spin = QSpinBox()
            sus_spin.setRange(0, 600)
            sus_spin.setValue(int(sustain or 0))
            sus_spin.setToolTip("0 = 非必需")
            self.comp_table.setCellWidget(r, 8, sus_spin)

            win_secs_spin = QSpinBox()
            win_secs_spin.setRange(0, 600)
            win_secs_spin.setValue(int(within or 0))
            win_secs_spin.setToolTip("0 = 必须完全同时发生")
            self.comp_table.setCellWidget(r, 9, win_secs_spin)

            win_spin = QDoubleSpinBox()
            win_spin.setRange(0.0, 10.0)
            win_spin.setSingleStep(0.25)
            win_spin.setValue(float(window))
            self.comp_table.setCellWidget(r, 10, win_spin)

            per_spin = QDoubleSpinBox()
            per_spin.setRange(0.0, 10.0)
            per_spin.setSingleStep(0.25)
            per_spin.setValue(float(persist))
            self.comp_table.setCellWidget(r, 11, per_spin)

            rel_combo = QComboBox()
            rel_combo.addItem("内部", "inside")
            rel_combo.addItem("重叠", "overlaps")
            rel_combo.addItem("接触", "touches")
            # Normalised the way the engine reads it, so `Touches` in the file
            # shows as touches instead of falling back to inside and being
            # dropped on the next save.
            _rel = str(relation or 'inside').strip().lower()
            if rel_combo.findData(_rel) < 0:
                rel_combo.addItem(_rel, _rel)
            rel_combo.setCurrentIndex(max(0, rel_combo.findData(_rel)))
            rel_combo.setToolTip("内部：中心位于区域内\n"
                                 "重叠：大部分区域位于目标区域内\n"
                                 "接触：两者边界或区域相接")
            self.comp_table.setCellWidget(r, COMP_REL_COL, rel_combo)

            outline_chk = QCheckBox()
            outline_chk.setChecked(bool(outline))
            outline_chk.setToolTip("根据检测框内描绘出的真实轮廓判断形状，而不是仅按矩形检测框判断。")
            self.comp_table.setCellWidget(r, COMP_OUTLINE_COL, outline_chk)

            del_btn = QPushButton()
            del_btn.setIcon(_ui_icons.cross())
            del_btn.setToolTip("删除此条件")
            del_btn.setFixedWidth(28)
            del_btn.setFlat(True)
            del_btn.setStyleSheet("border: none;")
            def _make_del(btn):
                def _del():
                    for i in range(self.comp_table.rowCount()):
                        if self.comp_table.cellWidget(i, COMP_DEL_COL) is btn:
                            self.comp_table.removeRow(i)
                            return
                return _del
            del_btn.clicked.connect(_make_del(del_btn))
            self.comp_table.setCellWidget(r, COMP_DEL_COL, del_btn)

            _comp_apply_kind(r)


        def _comp_collect_events():
            """The table as the rules file's ``events`` list."""
            # Group rows by event name (preserving order of first appearance).
            # A row is one *condition*; several rows can belong to one event,
            # and they may now be of either kind, so an event can carry spatial
            # and signal conditions together.
            events_ordered = []
            events_map = {}
            for r in range(self.comp_table.rowCount()):
                def _txt(c):
                    it = self.comp_table.item(r, c)
                    return it.text().strip() if it else ''
                kind     = _comp_kind_of(r)
                ev_name  = _txt(2)
                ev_label = _txt(3)
                first    = _txt(4)
                second   = _txt(5)
                min_v    = self.comp_table.cellWidget(r, 6).value()
                max_v    = self.comp_table.cellWidget(r, 7).value()
                sustain  = self.comp_table.cellWidget(r, 8).value()
                within   = self.comp_table.cellWidget(r, 9).value()
                window   = self.comp_table.cellWidget(r, 10).value()
                persist  = self.comp_table.cellWidget(r, 11).value()
                on_box   = self.comp_table.cellWidget(r, 0)
                enabled  = True if on_box is None else bool(on_box.isChecked())

                if not ev_name or not first:
                    continue
                if kind == "Spatial" and not second:
                    continue

                if ev_name not in events_map:
                    # Start from what was read, so fields this table has no
                    # column for are carried across untouched, then overwrite
                    # the ones it does own. `rules` and `signals` are rebuilt
                    # from the rows below and must not survive from the
                    # original, or a deleted condition would come back.
                    entry = dict(getattr(self, '_comp_original', {}).get(ev_name, {}))
                    entry.pop('rules', None)
                    entry.pop('signals', None)
                    entry.update({
                        '名称': ev_name,
                        'label': ev_label or ev_name,
                        'enabled': enabled,
                        'window_secs': window,
                        'persist_secs': persist,
                    })
                    events_map[ev_name] = entry
                    events_ordered.append(entry)
                entry = events_map[ev_name]

                if kind == "Signal":
                    cond = {'signal': first}
                    # The extremes mean "no bound". Writing them out would turn
                    # an open-ended condition into one clamped at an arbitrary
                    # number that happens to be this widget's range.
                    if min_v > self.COMP_MIN_UNSET:
                        cond['min'] = round(float(min_v), 4)
                    if max_v < self.COMP_MAX_UNSET:
                        cond['max'] = round(float(max_v), 4)
                    if second:
                        cond['equals'] = second
                    if int(sustain) > 0:
                        cond['sustained_secs'] = int(sustain)
                    if int(within) > 0:
                        cond['within_secs'] = int(within)
                    entry.setdefault('signals', []).append(cond)
                else:
                    rule = {
                        'source': first,
                        'region': second,
                        # Counts are whole numbers; the shared spin box carries
                        # decimals for the signal case.
                        'min_count': int(round(min_v)) if min_v > self.COMP_MIN_UNSET else 0,
                        'max_count': int(round(max_v)) if max_v < self.COMP_MAX_UNSET else 999,
                    }
                    rel_w = self.comp_table.cellWidget(r, COMP_REL_COL)
                    if rel_w is not None:
                        relation_value = str(rel_w.currentData() or rel_w.currentText()).strip()
                        if relation_value != 'inside':
                            rule['relation'] = relation_value
                    out_w = self.comp_table.cellWidget(r, COMP_OUTLINE_COL)
                    if out_w is not None and out_w.isChecked():
                        rule['outline'] = True
                    entry.setdefault('rules', []).append(rule)

            # Rule fields without a column (relation, outline, ...) carried
            # over from each rule's original, matched by its source and region.
            from modules.rules.rules_file import TABLE_RULE_KEYS, carry_rule_fields
            originals = getattr(self, '_comp_original', {})
            owned = TABLE_RULE_KEYS | {'relation', 'outline'}
            for entry in events_ordered:
                if entry.get('rules'):
                    entry['rules'] = carry_rule_fields(
                        (originals.get(entry['名称']) or {}).get('rules'),
                        entry['rules'], owned=owned)

            # Anything the table still cannot represent — an event with neither
            # kind of condition — is written back as it was read. The table
            # rebuilds this file from its own rows, so without this such an
            # entry would be deleted with nothing on screen to show it going.
            events_ordered.extend(getattr(self, '_comp_passthrough', []) or [])
            return events_ordered

        def _comp_save_rules(quiet=False):
            """Write the table to the rules file. Returns True if it wrote.

            ``quiet`` skips the write when nothing changed and reports only
            failures — for the automatic saves (on Run, on close), where a log
            line per close is noise and a silent loss of a ticked box is not.
            """
            from modules.system.app_paths import user_data_dir
            import os as _os
            out = {**(getattr(self, '_comp_top', None) or {}),
                   'events': _comp_collect_events()}
            if quiet and out == getattr(self, '_comp_saved_state', None):
                return False
            save_path = _os.path.join(user_data_dir(), 'composition_rules.yaml')
            try:
                with open(save_path, 'w', encoding='utf-8') as _f:
                    yaml.dump(out, _f, allow_unicode=True, sort_keys=False, default_flow_style=False)
                self._comp_saved_state = out
                if not quiet:
                    self.append_log(f"✅ 构图规则已保存 → {save_path}")
                return True
            except Exception as _e:
                # Never quiet: this is the one outcome the user has to know
                # about, and on close it is their last chance to.
                self.append_log(f"❌ 无法保存构图规则：{_e}")
                return False


        comp_add_btn.clicked.connect(
            lambda: _comp_add_table_row(kind='Spatial'))
        comp_add_signal_btn.clicked.connect(
            lambda: _comp_add_table_row(kind='Signal'))
        comp_save_btn.clicked.connect(_comp_save_rules)
        _comp_load_rules()

        # ── Group 5: Bounding Box Visualization ──
        bbox_box = QGroupBox("检测框可视化")
        bbox_layout = QVBoxLayout()

        info_label = QLabel("ℹ️ 启用检测框后会生成带 _annotated.mp4 后缀的新文件，用于调试")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        bbox_layout.addWidget(info_label)

        self.bbox_objects_chk = QCheckBox("绘制物体检测框")
        self.bbox_objects_chk.setChecked(visualization_cfg.get("draw_object_boxes", False))
        self.bbox_objects_chk.setToolTip("用带标签的检测框显示已检测物体")
        bbox_layout.addWidget(self.bbox_objects_chk)

        self.bbox_actions_chk = QCheckBox("显示动作识别标签")
        self.bbox_actions_chk.setChecked(visualization_cfg.get("draw_action_labels", False))
        self.bbox_actions_chk.setToolTip("在画面上显示检测到的动作名称")
        bbox_layout.addWidget(self.bbox_actions_chk)

        bbox_box.setLayout(bbox_layout)
        advanced_layout.addWidget(bbox_box, 1, 1)

        # ── Group 6: Highlight Report ──
        # Its own group, and not a corner of the bounding-box one. The report is
        # the answer to "why these moments", which is the thing this app is for;
        # filed under a debugging switch that writes an _annotated.mp4, it read
        # as a developer option nobody was meant to turn on.
        report_box = QGroupBox("高光报告")
        report_layout = QVBoxLayout()

        report_info = QLabel(
            "ℹ️ 解释每个时刻为什么被保留：会在高光视频旁生成 HTML 报告，"
            "包含缩略图、评分以及差一点被选中的时刻。")
        report_info.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        report_info.setWordWrap(True)
        report_layout.addWidget(report_info)

        # On by default: it costs one frame grab per kept segment and answers the
        # question every user asks first — why these moments and not others.
        self.why_report_chk = QCheckBox("生成高光报告")
        self.why_report_chk.setChecked(
            visualization_cfg.get("write_highlight_report", True))
        self.why_report_chk.setToolTip(
            "在高光视频旁生成 <output>_why.html：列出每个保留片段的评分构成、\n"
            "触发它的物体与动作，以及得分较高但最终未入选的时刻。\n\n"
            "报告为单个自包含文件，缩略图已嵌入，可直接用浏览器打开或发送给他人。\n"
            "同时生成对应的 .json 文件保存相同数据。")
        report_layout.addWidget(self.why_report_chk)

        # The narration passes, which used to be reachable only from the
        # AI-summary menu after the run had already finished. On by default: a
        # clip on footage nobody speaks over has nothing on its card about what
        # is in the picture, and a chapter is the same question one scale up.
        self.narrate_clips_chk = QCheckBox("…并描述每个片段")
        self.narrate_clips_chk.setChecked(
            visualization_cfg.get("narrate_clips", True))
        self.narrate_clips_chk.setToolTip(
            "处理结束时，让所选视觉模型根据画面描述每个保留片段。\n\n"
            "每个片段需要调用模型一次，因此会增加一些处理时间，并要求模型支持视觉。\n"
            "启用后，片段卡片会包含画面内容说明；否则只显示实际测量到的数据。")
        report_layout.addWidget(self.narrate_clips_chk)

        # The label carries the warning the default cannot: this is the slowest
        # thing the report does, so the user needs to know what it costs at the
        # moment they could turn it off, not after the run has spent it.
        self.narrate_chapters_chk = QCheckBox("…并生成各章节说明（较慢）")
        self.narrate_chapters_chk.setChecked(
            visualization_cfg.get("narrate_chapters", True))
        self.narrate_chapters_chk.setToolTip(
            "处理结束时，让所选模型为视频的每个章节生成说明。\n\n"
            "每个章节需要调用模型一次，这是报告流程中最慢的步骤之一。\n"
            "如果素材已有充分的语音转录，额外收益可能较小。")
        report_layout.addWidget(self.narrate_chapters_chk)

        for _chk in (self.narrate_clips_chk, self.narrate_chapters_chk):
            # Both narrate the report, so neither means anything without one.
            self.why_report_chk.toggled.connect(_chk.setEnabled)
            _chk.setEnabled(self.why_report_chk.isChecked())

        # Where the output folder is reachable over the network. A report read
        # on a phone has dead players — a browser cannot reach a sibling file
        # from a `content://` origin, nor seek one without HTTP range requests —
        # and every figure on the page stays true, which is what makes that
        # confusing rather than obviously broken. Given this, the page carries
        # the address where it can be played.
        self.serve_base_input = QLineEdit(
            str(visualization_cfg.get("report_serve_base", "") or ""))
        self.serve_base_input.setPlaceholderText("http://192.168.0.10:8000/")
        self.serve_base_input.setToolTip(
            "可选。如果你通过 HTTP 提供输出文件夹，请在这里填写基础地址，\n"
            "每份报告都会获得一个可直接播放的链接。\n\n"
            "留空则保持默认行为。此处只添加链接，不会启动服务器；\n"
            "如需支持拖动播放进度的服务器，可使用 tools/serve_report.py。")
        self.why_report_chk.toggled.connect(self.serve_base_input.setEnabled)
        self.serve_base_input.setEnabled(self.why_report_chk.isChecked())
        report_layout.addWidget(QLabel("服务地址（可选）："))
        report_layout.addWidget(self.serve_base_input)

        # The other half, and the one that survives with nothing running: where
        # the *footage* lives on a share. A browser cannot play `smb://` — no
        # browser implements the scheme — but tapping a link to one hands it to
        # a player app, which is enough to check a moment. A share is mounted
        # all day where an ad-hoc web server is not, so this is the fallback the
        # HTTP link needs rather than a duplicate of it.
        self.media_base_input = QLineEdit(
            str(visualization_cfg.get("report_media_base", "") or ""))
        self.media_base_input.setPlaceholderText("smb://192.168.0.10/movies/")
        self.media_base_input.setToolTip(
            "可选。填写其他设备可访问视频文件夹的位置，通常是 SMB 共享地址。\n"
            "这样每个片段都会带有一个可在播放器中打开源视频的链接。\n\n"
            "跨应用打开时无法保留播放位置，因此视频会从开头打开，链接会注明需要跳转的时间点。\n"
            "如果希望点击后直接落在对应时刻，请优先使用上方 HTTP 地址。")
        self.why_report_chk.toggled.connect(self.media_base_input.setEnabled)
        self.media_base_input.setEnabled(self.why_report_chk.isChecked())
        report_layout.addWidget(QLabel("视频访问地址（可选）："))
        report_layout.addWidget(self.media_base_input)

        report_box.setLayout(report_layout)
        # Full width, below the four detector groups: it is about the run as a
        # whole rather than about one detector, and the two URL fields need the
        # room to show an address without eliding it.
        advanced_layout.addWidget(report_box, 3, 0, 1, 2)

        # ── Group: Video Output ──
        # How the final highlight is re-encoded. CPU (libx265) is VR-safe but slow;
        # GPU is fast but its HEVC may not play in some VR players. Placed at the top
        # of Advanced so it's easy to find when a VR player rejects a render.
        output_box = QGroupBox("视频输出")
        output_layout = QFormLayout()
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("CPU x265（VR 兼容，较慢）", "cpu")
        self.render_mode_combo.addItem("GPU（快速，部分 VR 播放器可能不兼容）", "gpu")
        self.render_mode_combo.setToolTip(
            "高光视频的编码方式：\n"
            "CPU x265 — 使用 CPU 和 libx265（HEVC）重新编码，更符合 VR 源视频的常见编码方式；\n"
            "兼容性更好，但处理 6K 视频时较慢。\n"
            "GPU — 使用硬件编码器重新编码，速度更快，但生成的 HEVC 视频\n"
            "可能无法在 HereSphere 等部分 VR 播放器中播放。"
        )
        _saved_render_mode = highlights_cfg.get("render_mode", "cpu")
        _rm_idx = self.render_mode_combo.findData(_saved_render_mode)
        if _rm_idx >= 0:
            self.render_mode_combo.setCurrentIndex(_rm_idx)
        output_layout.addRow("剪切 / 编码：", self.render_mode_combo)
        output_box.setLayout(output_layout)
        advanced_layout.addWidget(output_box, 0, 0)

        # ── Group: Compute ──
        # DirectML had a switch (VH_DIRECTML) and no way to reach it: the
        # packaged app is started from a shortcut, and an environment variable
        # exported in a console is not inherited by one. Anybody without a
        # terminal therefore could not try the backend that exists for them.
        compute_box = QGroupBox("计算")
        compute_layout = QFormLayout()
        self.backend_combo = QComboBox()
        # Only what this platform has: Apple's GPU on a Mac, the PC backends
        # everywhere else (compute_backend.choices).
        for _backend, _label in compute_backend.choices():
            self.backend_combo.addItem(_label, _backend)
        if sys.platform == "darwin":
            self.backend_combo.setToolTip(
                "选择本次运行使用的计算加速后端。\n\n"
                "自动模式会优先使用可用的 Apple GPU / Core ML，否则使用处理器。"
                "只使用处理器可用于对比没有 GPU 加速时的速度。\n\n"
                "Apple GPU（Core ML）会把物体检测和动作识别交给 GPU / 神经网络引擎"
                "（实验功能）。每次运行都会在日志中显示实际使用的后端。"
            )
        else:
            self.backend_combo.setToolTip(
                "选择本次运行使用的计算加速后端。\n\n"
                "自动模式会优先选择当前电脑最快的可用后端：CUDA、Intel、DirectML，"
                "最后才使用处理器。你也可以手动指定某一种后端进行对比测试。\n\n"
                "如果选择了当前电脑不可用的后端，程序会自动回退到自动模式，"
                "并在日志中说明；每次运行都会显示最终实际使用的后端。\n\n"
                "DirectML 可用于物体检测和动作识别；其他部分的支持取决于当前安装环境。"
            )
        _saved_backend = (compute_backend.from_config(self.config_data)
                          or compute_backend.configured()
                          or compute_backend.AUTO)
        _backend_idx = self.backend_combo.findData(_saved_backend)
        if _backend_idx >= 0:
            self.backend_combo.setCurrentIndex(_backend_idx)
        # Applied immediately as well as saved: the next run reads the
        # environment, and waiting for a restart to try a backend is the kind of
        # friction that stops anybody trying it.
        self.backend_combo.currentIndexChanged.connect(
            lambda: compute_backend.set_now(self.backend_combo.currentData(),
                                            log=self.append_log))
        compute_layout.addRow("优先使用：", self.backend_combo)
        compute_box.setLayout(compute_layout)
        advanced_layout.addWidget(compute_box, 0, 1)

        # Equal column widths; let the row below the composition table absorb slack
        advanced_layout.setColumnStretch(0, 1)
        advanced_layout.setColumnStretch(1, 1)
        advanced_layout.setRowStretch(5, 1)

        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        _adv_container = QWidget()
        _adv_container.setLayout(advanced_layout)
        advanced_scroll.setWidget(_adv_container)
        advanced_tab.setLayout(QVBoxLayout())
        advanced_tab.layout().setContentsMargins(0, 0, 0, 0)
        advanced_tab.layout().addWidget(advanced_scroll)
        tabs.addTab(self._scrollable(advanced_tab), "高级")

        content_splitter = QSplitter(Qt.Vertical)
        # A floor, not the old behaviour. Wrapping the pages in scroll areas
        # dropped the tab widget's minimum height to almost nothing, which is
        # what let the window finally shrink — but it also left nothing
        # resisting the splitter, so the tabs collapsed to a couple of rows
        # while the empty log pane kept its share. Small enough that the window
        # still fits a 1080p screen, big enough to show a form without folding.
        tabs.setMinimumHeight(280)
        content_splitter.addWidget(tabs)
        self.content_splitter = content_splitter
        layout.addWidget(content_splitter)

        # --- Tab 4: LLM Chat ---
        llm_tab = QWidget()
        self.llm_tab_layout = QVBoxLayout()
        self.llm_chat = LLMChatWidget(parent=self)
        self.llm_tab_layout.addWidget(self.llm_chat)
        # Kept so the tab can reclaim the panel after the Simple view has
        # borrowed it (see set_simple_start).
        llm_tab.setLayout(self.llm_tab_layout)
        tabs.addTab(self._scrollable(llm_tab), "大模型对话")

        # --- Tab: Train ---
        # Assembling a dataset, fine-tuning a detector and exporting it for the
        # app were three scripts and a Python prompt. The panel drives the same
        # functions the tests do; nothing about the sequencing lives in it.
        try:
            from modules.ui.training_panel import TrainingPanel
            train_tab = QWidget()
            train_layout = QVBoxLayout()
            self.training_panel = TrainingPanel(parent=self)
            self.training_panel.model_installed.connect(self._on_model_installed)
            train_layout.addWidget(self.training_panel)
            train_tab.setLayout(train_layout)
            tabs.addTab(self._scrollable(train_tab), "训练")
        except Exception as e:
            self.append_log(f"⚠️ 训练面板不可用：{e}")

        # --- Tab 5: Avoid ---
        avoid_tab = QWidget()
        avoid_layout = QVBoxLayout()

        avoid_group = QGroupBox("排除人物")
        avoid_group_layout = QVBoxLayout()

        self.avoid_face_recognition_chk = QCheckBox("启用人脸识别")
        self.avoid_face_recognition_chk.setChecked(self.config_data.get("avoid", {}).get("face_recognition_enabled", False))
        self.avoid_face_recognition_chk.setToolTip(
            "启用后，处理流程会运行人脸识别，定位需要排除的人物，并跳过或裁掉其出现的画面。\n"
            "关闭后将完全跳过人脸识别步骤（速度更快，但不会执行人物排除）。"
        )
        avoid_group_layout.addWidget(self.avoid_face_recognition_chk)

        avoid_info = QLabel(
            "你在时间线查看器中命名的人物（右键点击人脸 → 命名）"
            "会显示在这里。勾选人物后，其出现片段将从生成的高光中排除。"
        )
        avoid_info.setWordWrap(True)
        avoid_info.setStyleSheet("color: #666; font-size: 9pt;")
        avoid_group_layout.addWidget(avoid_info)
        avoid_method_row = QHBoxLayout()
        avoid_method_row.addWidget(QLabel("识别到时："))
        self.avoid_method_combo = QComboBox()
        self.avoid_method_combo.addItem("跳过这些时刻", "skip")
        self.avoid_method_combo.addItem("裁掉这些内容（实验性）", "crop")
        self.avoid_method_combo.currentIndexChanged.connect(
            lambda: setattr(self, "_avoid_method", self.avoid_method_combo.currentData()))
        avoid_method_row.addWidget(self.avoid_method_combo)
        avoid_method_row.addStretch()
        avoid_group_layout.addLayout(avoid_method_row)

        avoid_row = QHBoxLayout()
        self.avoid_refresh_btn = QPushButton("🔄 从人脸库刷新")
        self.avoid_refresh_btn.clicked.connect(self.refresh_avoid_list)
        avoid_row.addWidget(self.avoid_refresh_btn)
        self.avoid_scan_btn = QPushButton("🔍 扫描视频人脸")
        self.avoid_scan_btn.setToolTip("对列表中的第一个视频运行人脸识别，收集出现的所有人物，然后勾选需要排除的人。")
        self.avoid_scan_btn.clicked.connect(self._on_scan_faces)
        avoid_row.addWidget(self.avoid_scan_btn)
        self.avoid_count_label = QLabel("")
        self.avoid_count_label.setStyleSheet("color: #2f81f7; font-weight: bold;")
        avoid_row.addWidget(self.avoid_count_label)
        avoid_row.addStretch()
        avoid_group_layout.addLayout(avoid_row)
        self.avoid_clear_btn = QPushButton("🗑 清空人脸")
        self.avoid_clear_btn.setToolTip("从人脸库移除扫描得到的人脸（保留已命名/已排除人物）。")
        self.avoid_clear_btn.clicked.connect(self._on_clear_faces)
        avoid_row.addWidget(self.avoid_clear_btn)

        self.avoid_scroll = QScrollArea()
        self.avoid_scroll.setWidgetResizable(True)
        self.avoid_list_container = QWidget()
        self.avoid_list_layout = QVBoxLayout(self.avoid_list_container)
        self.avoid_list_layout.addStretch()
        self.avoid_scroll.setWidget(self.avoid_list_container)
        avoid_group_layout.addWidget(self.avoid_scroll)

        avoid_group.setLayout(avoid_group_layout)
        avoid_layout.addWidget(avoid_group, 1)
        avoid_tab.setLayout(avoid_layout)
        tabs.addTab(self._scrollable(avoid_tab), "排除")

        # --- Tab: About & Contact ---
        tabs.addTab(self._scrollable(self._build_about_tab()), "关于")

        # Defer first populate until after __init__ finishes (so log_output exists)
        QTimer.singleShot(0, self.refresh_avoid_list)
        # Let the window finish painting first — the check is never urgent, and
        # its own throttle means most launches do no network at all.
        QTimer.singleShot(3000, self._start_update_check)
        # Clear whatever the last update displaced. It can only be deleted now,
        # on a launch after the process that had those files open has exited.
        QTimer.singleShot(1500, self._sweep_updated_files)

        # --- Run / Cancel Controls ---
        ctrl_layout = QHBoxLayout()
        self.keep_temp_chk = QPushButton("保留临时片段：开" if highlights_cfg.get("keep_temp", False) else "保留临时片段：关")
        self.keep_temp_chk.setCheckable(True)
        self.keep_temp_chk.setChecked(highlights_cfg.get("keep_temp", False))
        self.keep_temp_chk.clicked.connect(lambda: self.keep_temp_chk.setText(
            "保留临时片段：开" if self.keep_temp_chk.isChecked() else "保留临时片段：关"))

        _export_on = bool(highlights_cfg.get("export_separate_clips", False))
        self.export_clips_chk = QPushButton(
            "单独导出片段：开" if _export_on else "单独导出片段：关")
        self.export_clips_chk.setCheckable(True)
        self.export_clips_chk.setChecked(_export_on)
        self.export_clips_chk.clicked.connect(lambda: self.export_clips_chk.setText(
            "单独导出片段：开" if self.export_clips_chk.isChecked() else "单独导出片段：关"))
        self.export_clips_chk.setToolTip(
            "除了合并后的高光视频外，还会将每个得分片段分别导出到\n"
            "<video>_clips/ 文件夹中。")

        self.timeline_btn = QPushButton("时间线查看器")
        self.timeline_btn.setStyleSheet("QPushButton { background-color: #2f81f7; color: white; font-weight: bold; padding: 8px; }")
        self.timeline_btn.clicked.connect(self.open_timeline_viewer)

        self.why_report_btn = QPushButton("高光报告")
        self.why_report_btn.setToolTip(
            "打开解释每个高光为什么被选中的报告。\n"
            "报告默认写在高光视频旁，可在“高级”标签页中控制是否生成。")
        self.why_report_btn.clicked.connect(self.open_why_report)
        # "AI 摘要" writes a few plain-language sentences into that same
        # report. Separate button because it costs a model run and tens of
        # seconds — the report itself must stay instant.
        self.ai_summary_btn = QPushButton("AI 摘要")
        self.ai_summary_btn.setToolTip(
            "在高光报告中加入一段简短易懂的 AI 摘要：\n"
            "说明本次剪辑主要受哪些因素影响，以及最值得优先调整的一项设置。\n"
            "该功能会运行本地模型，因此需要一些时间；即使不生成 AI 摘要，\n"
            "报告中的基础分析结果仍会正常保留。")
        self.ai_summary_btn.clicked.connect(self.write_ai_summary)

        # The wheel keeps the choices that most users never touch out of sight.
        # A drawn icon, not a "⚙" glyph: the packaged build has no guarantee of
        # a font carrying it, and a blank button is what you get when it does
        # not.
        self.ai_summary_opts_btn = QPushButton()
        self.ai_summary_opts_btn.setIcon(_ui_icons.gear())
        self.ai_summary_opts_btn.setFixedWidth(28)
        self.ai_summary_opts_btn.setToolTip("摘要选项 — 提问、"
                                            "在聊天中讨论、选择模型")
        self.ai_summary_opts_btn.clicked.connect(self.show_ai_summary_menu)

        self.simple_start_btn = QPushButton("简洁视图")
        self.simple_start_btn.setToolTip(
            "一键式工作区：添加视频后直接分析即可。\n"
            "需要更多控制时仍可返回详细设置。")
        self.simple_start_btn.clicked.connect(lambda: self.set_simple_start(True))

        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #ff4444; color: white; font-weight: bold; }")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)

        # Scores and reports without encoding. Detection is cached, so this is
        # seconds once a video has been analysed — cheap enough to use as the
        # normal way of trying a setting.
        self.report_only_btn = QPushButton("仅生成报告")
        self.report_only_btn.setToolTip(
            "使用当前设置重新评分并生成高光报告，但不渲染视频。\n"
            "对于已经分析过的视频，这个过程通常很快。")
        self.report_only_btn.clicked.connect(lambda: self.run_pipeline(report_only=True))

        self.run_btn = QPushButton("运行高光分析")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.clicked.connect(self.toggle_run)

        ctrl_layout.addWidget(self.simple_start_btn)
        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.keep_temp_chk)
        ctrl_layout.addWidget(self.export_clips_chk)
        ctrl_layout.addWidget(self.timeline_btn)
        ctrl_layout.addWidget(self.why_report_btn)
        ctrl_layout.addWidget(self.ai_summary_btn)
        ctrl_layout.addWidget(self.ai_summary_opts_btn)
        self.debug_console_chk = QCheckBox("调试日志")
        self.debug_console_chk.setChecked(debug_console.is_console_visible())
        self.debug_console_chk.setToolTip(
            "打开实时调试窗口，显示程序的全部输出。\n"
            "近期输出也会重新显示，因此发生错误后再打开也能查看。\n"
            f"所有日志始终保存到：\n{debug_console.log_file_path()}"
        )
        self.debug_console_chk.toggled.connect(debug_console.set_console_visible)
        debug_console.register_checkbox(self.debug_console_chk)
        ctrl_layout.addWidget(self.debug_console_chk)
        self.session_analyzed_count = 0
        self.analyzed_counter_label = QLabel()
        self.analyzed_counter_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.analyzed_counter_label.setToolTip(
            "处理流程已成功分析的视频数量。\n"
            f"累计统计保存在：\n{analysis_stats.stats_path()}"
        )
        self.update_analyzed_counter()
        ctrl_layout.addWidget(self.analyzed_counter_label)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.report_only_btn)
        ctrl_layout.addWidget(self.run_btn)
        layout.addLayout(ctrl_layout)

        # --- Log view (inside splitter) ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(80)
        self.log_output.setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 9pt; }")
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("日志输出："))
        log_layout.addWidget(self.log_output)
        content_splitter.addWidget(log_widget)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        # Sized against the splitter's own height, not the window's. Asking for
        # window-height-derived sizes here requests more than the splitter
        # actually receives (the input list and time range are above it), and
        # QSplitter then scales BOTH panes down proportionally — which is what
        # squeezed the tabs while the log kept its 80px minimum.
        QTimer.singleShot(0, self._balance_content_splitter)

        # The Simple page grows when its chat section is unfolded, so it scrolls
        # rather than forcing a window taller than the screen. The stack holds
        # the scroll area; set_simple_start switches to that, not to the page.
        self.simple_page = SimpleStartPage(self)
        self.simple_host = self._scrollable(self.simple_page)
        self.view_stack.addWidget(self.simple_host)
        self.view_stack.addWidget(self.full_page)
        self.set_simple_start(simple_start_enabled(default=True), persist=False)

        self.setLayout(root)

        self.setup_label_completers()
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_worker_status)

        # Load download time range settings (AFTER all widgets are created)
        download_cfg = self.config_data.get("download", {})
        self.download_start_input.setValue(download_cfg.get("time_range_start", 0))
        self.download_end_input.setValue(download_cfg.get("time_range_end", 300))

        # Restore the download mode. Fall back to the old two-checkbox keys so
        # existing configs keep working: use_same_time_range -> "same",
        # download_full -> "full", else "specific".
        mode = download_cfg.get("download_mode")
        if mode is None:
            if download_cfg.get("use_same_time_range", False):
                mode = "same"
            elif download_cfg.get("download_full", False):
                mode = "full"
            else:
                mode = "full"
        idx = self.download_mode_combo.findData(mode)
        self.download_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.on_download_mode_changed()  # sync visibility

        # Restore the after-download processing mode, with fall-back from the old
        # keys: immediate_processing -> "immediate", auto_process -> "batch",
        # else "none".
        pmode = download_cfg.get("process_mode")
        if pmode is None:
            if download_cfg.get("immediate_processing", False):
                pmode = "immediate"
            elif download_cfg.get("auto_process", False):
                pmode = "batch"
            else:
                pmode = "none"
        pidx = self.process_mode_combo.findData(pmode)
        self.process_mode_combo.setCurrentIndex(pidx if pidx >= 0 else 0)
        self.on_process_mode_changed()  # sync spinner enabled

    def resizeEvent(self, event):
        """Record where a resize settled.

        Resizing to fill a large, heavily scaled display has been reported to
        kill the process, and a crash below the Python frame writes no
        traceback — so the last line in debug.log is the evidence. See
        modules/system/display_info.py.
        """
        super().resizeEvent(event)
        timer = getattr(self, "_size_log_timer", None)
        if timer is not None:
            timer.start()

    def _log_size(self):
        from modules.system import display_info
        display_info.log_window_size(self, "Main window")

    # --- About / Contact tab ---
    @staticmethod
    def _scrollable(page):
        """Wrap a tab page so it can shrink, and scroll instead of clipping.

        QTabWidget takes its minimum height from its tallest page, so one big
        tab set the floor for the entire window — 1049px, more than a 1080p
        screen has once the taskbar is accounted for. The window then could not
        shrink to fit, and the row of buttons at the bottom ended up under the
        taskbar, which is where this started.
        """
        from PySide6.QtWidgets import QScrollArea

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(page)
        return area

    def set_simple_start(self, on: bool, persist: bool = True):
        """Simple view vs Detailed settings. Neither one is a wizard step;
        the last chosen workspace is remembered. The detailed page is never
        destroyed."""
        page = getattr(self, "simple_page", None)
        if page is None or not hasattr(self, "view_stack"):
            return
        self.view_stack.setCurrentWidget(self.simple_host if on else self.full_page)
        if persist:
            persist_simple_start(bool(on))
        # One chat panel, moved to whichever view is on screen. Building a second
        # one would mean two model connections and two analysis caches claiming
        # to be the answer for the same video.
        chat = getattr(self, "llm_chat", None)
        if chat is not None:
            if on and not page.owns_chat(chat):
                # Moving to Simple view: attach chat
                page.attach_chat(chat)
            elif hasattr(self, "llm_tab_layout"):
                # Moving to Detailed view: ensure chat is in LLM tab layout
                if page.owns_chat(chat):
                    page.release_chat(chat)
                if self.llm_tab_layout.indexOf(chat) < 0:
                    self.llm_tab_layout.addWidget(chat)
        if on:
            page.refresh_files()
            page.sync_run_chrome()

    def _sync_simple_start(self):
        page = getattr(self, "simple_page", None)
        if page is not None:
            page.sync_run_chrome()

    def _balance_content_splitter(self):
        """Give the log a fixed slice and the tabs everything else.

        Runs after the first layout pass, when the splitter knows how tall it
        actually is. The log is a status pane — it wants a readable few lines,
        not a proportional share of the window.
        """
        splitter = getattr(self, "content_splitter", None)
        if splitter is None:
            return
        available = splitter.height()
        log_height = 150 if available >= 460 else 110
        splitter.setSizes([max(280, available - log_height), log_height])

    # --- Update notice ---
    def _build_update_banner(self):
        """The hidden-by-default "a newer version exists" strip.

        Built once at startup and only ever shown/hidden, so the check that
        fills it in never has to construct widgets from its own thread.
        """
        banner = QWidget()
        banner.setVisible(False)
        banner.setStyleSheet(
            "QWidget { background: #2d4a63; border-radius: 4px; }"
            "QLabel { color: #e8f1f8; background: transparent; }"
        )
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)

        self.update_label = QLabel()
        self.update_label.setWordWrap(True)
        row.addWidget(self.update_label, 1)

        # Shown only while an install is running.
        self.update_progress = QProgressBar()
        self.update_progress.setVisible(False)
        self.update_progress.setMaximumWidth(220)
        row.addWidget(self.update_progress)

        self.update_install_btn = QPushButton("下载并安装")
        self.update_install_btn.clicked.connect(self._install_update)
        self.update_install_btn.setVisible(False)
        row.addWidget(self.update_install_btn)

        self.update_get_btn = QPushButton("获取")
        self.update_get_btn.clicked.connect(self._open_update_download)
        row.addWidget(self.update_get_btn)

        self.update_skip_btn = QPushButton("跳过此版本")
        self.update_skip_btn.clicked.connect(self._skip_update)
        row.addWidget(self.update_skip_btn)

        self.update_close_btn = QPushButton("✕")
        self.update_close_btn.setFixedWidth(28)
        self.update_close_btn.setToolTip("隐藏到下次检查")
        self.update_close_btn.clicked.connect(
            lambda: self.update_banner.setVisible(False))
        row.addWidget(self.update_close_btn)

        banner.setLayout(row)
        return banner

    # --- Pro offer ---
    def _build_pro_banner(self):
        """The hidden-by-default "Pro can do this" strip.

        Same shape as the update banner and for the same reason: it costs no
        space while hidden and never interrupts. What it says, and when it is
        allowed to say it, is decided in :mod:`modules.ui.pro_offer`.
        """
        banner = QWidget()
        banner.setVisible(False)
        banner.setStyleSheet(
            "QWidget { background: #3b3552; border-radius: 4px; }"
            "QLabel { color: #eeeaf6; background: transparent; }"
        )
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)

        self.pro_label = QLabel()
        self.pro_label.setWordWrap(True)
        row.addWidget(self.pro_label, 1)

        try_btn = QPushButton("免费试用 Pro")
        try_btn.clicked.connect(self._open_pro_page)
        row.addWidget(try_btn)

        never_btn = QPushButton("不再推荐 Pro")
        never_btn.setToolTip("关闭此类推荐。“关于”页"
                             "可重新开启这些推荐。")
        never_btn.clicked.connect(self._silence_pro_offer)
        row.addWidget(never_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedWidth(28)
        close_btn.setToolTip("暂不")
        close_btn.clicked.connect(lambda: self.pro_banner.setVisible(False))
        row.addWidget(close_btn)

        banner.setLayout(row)
        return banner

    def _show_pro_offer(self, offer):
        """Put an offer on screen, if there is one. Returns whether it showed."""
        if offer is None or not hasattr(self, "pro_banner"):
            return False
        from modules.ui import pro_offer

        if getattr(offer, "moment", "") == "rule_unbuildable":
            pro_text = (
                "如果你要找的内容不在当前视频的检测类别中，构图规则无法新增类别；"
                "规则只能组合已有检测结果。你可以在播放器中右键目标并选择"
                "“训练模型”来教会程序识别。<b>VideoHighlighter Pro</b> 也可以直接按名称查找，"
                "无需训练，并提供免费试用。"
            )
        elif getattr(offer, "moment", "") == "report_unmeasured":
            pro_text = (
                "这份报告包含尚未被实际测量的内容。<b>VideoHighlighter Pro</b> "
                "提供当前版本没有的额外分析方式来补充这些测量，并提供免费试用。"
            )
        else:
            pro_text = offer.text
        self.pro_label.setText(pro_text)
        self.pro_banner.setVisible(True)
        pro_offer.mark_shown(offer.moment)
        print(f"pro_offer: showed '{offer.moment}'")
        return True

    def _open_pro_page(self):
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from modules.ui import pro_offer

        QDesktopServices.openUrl(QUrl(pro_offer.PRO_URL))
        self.pro_banner.setVisible(False)

    def _silence_pro_offer(self):
        from modules.ui import pro_offer

        pro_offer.set_enabled(False)
        self.pro_banner.setVisible(False)
        chk = getattr(self, "pro_offer_chk", None)
        if chk is not None:
            chk.blockSignals(True)
            chk.setChecked(False)
            chk.blockSignals(False)

    def _start_update_check(self, force=False):
        """Kick off the manifest check in the background.

        Deliberately after the window is up: startup must not wait on the
        network, and a user who never sees a newer build should never know
        this ran.
        """
        self.update_worker = UpdateCheckWorker(force=force, parent=self)
        self.update_worker.found.connect(self._on_update_available)
        self.update_worker.nothing.connect(self._on_update_check_quiet)
        self.update_worker.start()

    def _on_update_available(self, info):
        """Show the banner. Runs on the GUI thread (queued signal)."""
        self._pending_update = info
        text = f"<b>发现新版本 {info.version}（当前版本 {__version__}）</b>"
        if info.notes:
            text += f"<br>{info.notes}"
        self.update_label.setText(text)

        # Installing in place is only offered to a packaged build. From source
        # the "install root" is the git checkout, and an update would overwrite
        # working files with release ones — so a dev build gets the download
        # link like any release published before the updater existed.
        can_install = bool(info.can_self_install) and getattr(sys, "frozen", False)
        self.update_install_btn.setVisible(can_install)
        self.update_get_btn.setVisible(not can_install)

        self.update_banner.setVisible(True)
        print(f"update_check: {info.version} available (running {__version__})"
              f"{' [self-install]' if can_install else ''}")

    def _sweep_updated_files(self):
        from modules.update import update_apply

        try:
            freed = update_apply.sweep_old(update_apply.install_root())
        except Exception as e:
            print(f"update_apply: sweep failed ({e})")
            return
        if freed:
            print(f"update_apply: reclaimed {freed / (1024 ** 2):.1f} MB "
                  "from the previous update")

    def _install_update(self):
        """Download and apply the pending release."""
        info = getattr(self, "_pending_update", None)
        if not info or not info.manifest_url:
            return
        from modules.update import update_apply

        self.update_install_btn.setEnabled(False)
        self.update_skip_btn.setVisible(False)
        self.update_close_btn.setVisible(False)
        self.update_progress.setVisible(True)
        self.update_progress.setRange(0, 0)     # indeterminate until sizes known
        self.update_label.setText("<b>正在准备更新…</b>")

        self.update_installer = UpdateInstallWorker(
            info.manifest_url, update_apply.install_root(), parent=self)
        self.update_installer.progress.connect(self._on_install_progress)
        self.update_installer.finished_with.connect(self._on_install_finished)
        self.update_installer.start()

    def _on_install_progress(self, phase, done, total, detail):
        from modules.update import update_install

        if phase == update_install.DOWNLOADING and total:
            self.update_progress.setRange(0, total)
            self.update_progress.setValue(done)
            mb_done, mb_total = done / (1024 ** 2), total / (1024 ** 2)
            self.update_label.setText(
                f"<b>正在下载 {mb_done:.1f} / {mb_total:.1f} MB</b><br>{detail}")
        else:
            self.update_progress.setRange(0, 0)
            self.update_label.setText(f"<b>{detail or phase}</b>")

    def _on_install_finished(self, result):
        from PySide6.QtWidgets import QMessageBox

        self.update_progress.setVisible(False)
        self.update_install_btn.setEnabled(True)
        self.update_close_btn.setVisible(True)

        if not result.ok:
            self.update_skip_btn.setVisible(True)
            # Whatever stopped it (a refused release, a dead connection), the
            # download page still works, so it is always one click away.
            self.update_get_btn.setVisible(True)
            self.update_label.setText(f"<b>{result.message}</b>")
            self.append_log(f"⚠️ 更新：{result.message}")
            return

        self.update_label.setText(f"<b>{result.message}</b>")
        self.append_log(f"✅ 更新：{result.message}")
        if not result.restart_required:
            return

        answer = QMessageBox.question(
            self, "立即重启？",
            f"{result.message}\n\n是否立即重启 Video Highlighter？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._restart_for_update()

    def _restart_for_update(self):
        """Relaunch the (now updated) app and quit this process.

        The new files are already in place; this process is still running the
        copies that were moved aside, which is why a restart is what actually
        switches versions. The displaced files are swept on the next launch,
        once nothing holds them open.
        """
        import subprocess
        from modules.update import update_apply

        try:
            subprocess.Popen(update_apply.relaunch_command(),
                             cwd=update_apply.install_root(), close_fds=True)
        except Exception as e:
            print(f"update_install: could not relaunch ({e})")
            return
        QApplication.quit()

    def _on_update_check_quiet(self, message):
        """Answer an explicit "check now" that turned up nothing."""
        if hasattr(self, "update_status_label"):
            self.update_status_label.setText(message)

    def _open_update_download(self):
        info = getattr(self, "_pending_update", None)
        if not info or not info.download_url:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(info.download_url))

    def _skip_update(self):
        info = getattr(self, "_pending_update", None)
        if info:
            from modules.update import update_check
            update_check.skip_version(info.version)
        self.update_banner.setVisible(False)

    def _build_about_tab(self):
        """A read-only About & Contact panel: version, support links, licensing."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        scroll.setWidget(content)

        # Header
        title = QLabel(f"🎬 Video Highlighter ({__edition__})")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(f"版本 {__version__} — 免费开源（AGPLv3）")
        subtitle.setStyleSheet("color: #888;")
        subtitle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(subtitle)

        # --- Updates ---
        from modules.update import update_check as _update_check

        upd_group = QGroupBox("更新")
        upd_layout = QVBoxLayout(upd_group)

        upd_auto = QCheckBox("自动检查新版本")
        upd_auto.setChecked(_update_check.is_enabled())
        upd_auto.setToolTip(
            "每天最多检查一次，只会下载一个记录最新版本号的小型文本文件。"
            "不会上传任何与你或这台电脑有关的信息。"
        )
        upd_auto.toggled.connect(_update_check.set_enabled)
        upd_layout.addWidget(upd_auto)

        upd_row = QHBoxLayout()
        upd_now_btn = QPushButton("立即检查")
        upd_now_btn.clicked.connect(lambda: self._start_update_check(force=True))
        upd_row.addWidget(upd_now_btn)
        self.update_status_label = QLabel("")
        self.update_status_label.setStyleSheet("color: #888;")
        upd_row.addWidget(self.update_status_label)
        upd_row.addStretch()
        upd_layout.addLayout(upd_row)
        layout.addWidget(upd_group)

        # --- Upgrade to Pro ---
        from modules.ui import pro_offer as _pro_offer

        pro_group = QGroupBox("VideoHighlighter Pro")
        pro_layout = QVBoxLayout(pro_group)
        pro_line = QLabel(
            "你当前使用的是免费开源版本——人脸身份、表情识别、分析报告和 AI 助手均可使用。"
            "<b>Pro</b> 版进一步支持：用你自己的示例帧训练专属类别、按示例搜索、"
            "开放词汇检测、实时叠加显示以及商业许可。"
            f"可免费试用 {_pro_offer.TRIAL_DAYS} 天。<br>"
            f'👉 <a href="{_pro_offer.PRO_URL}">了解更多 / 开始试用</a>'
        )
        pro_line.setOpenExternalLinks(True)
        pro_line.setTextInteractionFlags(Qt.TextBrowserInteraction)
        pro_line.setWordWrap(True)
        pro_layout.addWidget(pro_line)

        self.pro_offer_chk = QCheckBox(
            "当遇到仅 Pro 版支持的功能时提示我")
        self.pro_offer_chk.setChecked(_pro_offer.is_enabled())
        self.pro_offer_chk.setToolTip(
            "例如，当当前视频已有类别无法构建某条规则时，会显示一行 Pro 提示。"
            "启动软件时不会弹出，也不会写入报告。")
        self.pro_offer_chk.toggled.connect(_pro_offer.set_enabled)
        pro_layout.addWidget(self.pro_offer_chk)
        layout.addWidget(pro_group)

        # --- Contact & support ---
        support_group = QGroupBox("联系与支持")
        support_layout = QVBoxLayout(support_group)
        intro = QLabel("需要帮助、发现问题或有功能建议？可通过以下方式联系我们：")
        intro.setWordWrap(True)
        support_layout.addWidget(intro)

        links = QLabel(
            f'📧 邮箱：<a href="mailto:{SUPPORT_EMAIL}?subject=VideoHighlighter%20support">{SUPPORT_EMAIL}</a><br>'
            f'💬 Discord: <a href="{DISCORD_URL}">{DISCORD_URL}</a><br>'
            f'🌐 官网：<a href="{WEBSITE_URL}">{WEBSITE_URL}</a><br>'
            f'⭐ 源代码：<a href="{REPO_URL}">{REPO_URL}</a>'
        )
        links.setOpenExternalLinks(True)
        links.setTextInteractionFlags(Qt.TextBrowserInteraction)
        links.setWordWrap(True)
        support_layout.addWidget(links)

        tip = QLabel(
            "💡 提交问题时，请同时提供操作系统信息和调试日志"
            "（可在“运行高光分析”旁开启“调试日志”），这样能更快定位问题。"
        )
        tip.setStyleSheet("color: #888; font-size: 9pt;")
        tip.setWordWrap(True)
        support_layout.addWidget(tip)
        layout.addWidget(support_group)

        # --- Legal ---
        legal_group = QGroupBox("法律信息")
        legal_layout = QVBoxLayout(legal_group)
        legal = QLabel(
            "© 2026 Przemysław Kreft 和 Meric Donmezer。<br>"
            "VideoHighlighter 是依据 "
            f'<a href="{REPO_URL}/blob/main/LICENSE">GNU AGPLv3</a> 许可发布的自由软件。'
            f'代码贡献依据 <a href="{REPO_URL}/blob/main/CLA.md">CLA</a> 接收。<br>'
            "软件包含 PySide6、FFmpeg 等第三方组件，各自遵循对应许可证。"
        )
        legal.setOpenExternalLinks(True)
        legal.setTextInteractionFlags(Qt.TextBrowserInteraction)
        legal.setWordWrap(True)
        legal_layout.addWidget(legal)
        layout.addWidget(legal_group)

        layout.addStretch()
        return outer

    # --- Avoid methods ---
    def _get_face_bank(self):
        """Lazily create / reload the shared face identity bank."""
        try:
            from video_ai_editor.face_identity import FaceIdentityBank
        except ImportError as e:
            if hasattr(self, "log_output"):
                self.append_log(f"⚠️ 人脸库不可用：{e}")
            return None
        if getattr(self, "_face_bank", None) is None:
            self._face_bank = FaceIdentityBank(db_path="./cache/face_db.json")
        else:
            self._face_bank.load()   # pick up names/avoids set in the timeline viewer
        return self._face_bank

    def refresh_avoid_list(self):
        """Rebuild the people rows from the face database."""
        import base64
        from PySide6.QtGui import QPixmap

        # clear existing rows (keep the trailing stretch)
        while self.avoid_list_layout.count() > 1:
            item = self.avoid_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        bank = self._get_face_bank()
        if bank is None:
            self.avoid_count_label.setText("人脸库不可用")
            return

        identities = bank.all_identities()
        identities.sort(key=lambda i: (i["名称"] is None, -(i.get("count") or 0)))

        named = 0
        for ident in identities:
            r = QWidget()
            rl = QHBoxLayout(r)
            rl.setContentsMargins(4, 2, 4, 2)

            thumb = QLabel()
            thumb.setFixedSize(48, 48)
            if ident.get("thumb"):
                pix = QPixmap()
                pix.loadFromData(base64.b64decode(ident["thumb"]), "JPEG")
                if not pix.isNull():
                    thumb.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))
            rl.addWidget(thumb)

            display = ident["名称"] or f"人物 {ident['id'][:8]}"
            if ident["名称"]:
                named += 1
            name_label = QLabel(
                f"<b>{display}</b><br>"
                f"<span style='color:#888;font-size:8pt;'>出现 {ident.get('count', 0)} 次</span>"
            )
            rl.addWidget(name_label, 1)

            chk = QCheckBox("排除")
            chk.setChecked(bool(ident.get("avoid", False)))
            chk.toggled.connect(lambda checked, iid=ident["id"]: self._on_avoid_toggled(iid, checked))
            rl.addWidget(chk)

            rm = QPushButton("✕")
            rm.setFixedWidth(28)
            rm.setToolTip("从人脸库移除此人物")
            rm.clicked.connect(lambda _=False, iid=ident["id"]: self._on_remove_identity(iid))
            rl.addWidget(rm)

            self.avoid_list_layout.insertWidget(self.avoid_list_layout.count() - 1, r)

        self.avoid_count_label.setText(
            f"{len(identities)} 人 · {named} 个已命名 · {len(bank.avoided_ids())} 个已排除"
        )

    def _on_avoid_toggled(self, identity_id, checked):
        """Persist an avoid toggle to the face database."""
        bank = getattr(self, "_face_bank", None)
        if bank is None:
            return
        bank.set_avoid(identity_id, checked)
        bank.save()
        name = bank.name_for(identity_id)
        self.append_log(f"{'🚫 正在排除' if checked else '✅ 已允许'} {name} "
                        f"（当前已排除 {len(bank.avoided_ids())} 人）")
        self.avoid_count_label.setText(
            f"{len(bank.all_identities())} 人 · "
            f"{sum(1 for i in bank.all_identities() if i['name'])} 个已命名 · "
            f"{len(bank.avoided_ids())} 个已排除"
        )

    def _on_scan_faces(self):
        videos = self.get_file_list()
        if not videos:
            self.append_log("⚠️ 请先添加视频，再扫描其中的人脸。")
            return
        video = videos[0]
        if not os.path.exists(video):
            self.append_log(f"⚠️ 找不到视频：{video}")
            return
        self.avoid_scan_btn.setEnabled(False)
        self.avoid_scan_btn.setText("🔍 扫描中…")
        self._scan_worker = FaceScanWorker(video, "./cache/face_db.json")
        self._scan_worker.log.connect(self.append_log)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_remove_identity(self, identity_id):
            bank = self._get_face_bank()
            if bank is None:
                return
            if bank.remove(identity_id):
                bank.save()
                self.append_log("🗑 已从人脸库移除 1 人")
            self.refresh_avoid_list()

    def _on_clear_faces(self):
            from PySide6.QtWidgets import QMessageBox
            bank = self._get_face_bank()
            if not bank or len(bank) == 0:
                self.append_log("ℹ️ 人脸库已经为空。")
                return
            box = QMessageBox(self)
            box.setWindowTitle("清空人脸")
            box.setText(f"清空人脸库中的 {len(bank)} 个身份？")
            box.setInformativeText("请选择要清除的内容。")
            btn_all   = box.addButton("全部清除", QMessageBox.ButtonRole.DestructiveRole)
            btn_keep  = box.addButton("保留已命名 / 已排除", QMessageBox.ButtonRole.AcceptRole)
            btn_cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is btn_cancel:
                return
            kept = bank.clear(keep_named=(clicked is btn_keep))
            bank.save()
            self.append_log(f"🗑 人脸库已清理——保留 {kept} 个身份")
            self.refresh_avoid_list()

    def _on_scan_done(self, n):
        self.avoid_scan_btn.setEnabled(True)
        self.avoid_scan_btn.setText("🔍 扫描视频人脸")
        if n >= 0:
            self.append_log(f"✅ 人脸扫描完成——人脸库中共有 {n} 个身份")
        self.refresh_avoid_list()

    # --- Downloader methods ---
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "选择保存目录", self.download_save_dir_input.text()
        )
        if directory:
            self.download_save_dir_input.setText(directory)

    def browse_and_select_videos(self):
        """Open the thumbnail picker for the listing URL, then download the chosen videos."""
        url = self.download_url_input.text().strip()
        if not url.startswith(("http://", "https://")):
            self.append_log("⚠️ 请先输入列表页面网址（http:// 或 https://）")
            return
        try:
            from video_picker_dialog import VideoPickerDialog
        except Exception as e:
            self.append_log(f"❌ 视频选择器不可用：{e}")
            return
        dlg = VideoPickerDialog(url, pattern="auto", use_browser="auto", parent=self)
        if dlg.exec():
            urls = [e["url"] for e in dlg.selected_entries()]
            if not urls:
                self.append_log("未选择任何视频。")
                return
            self.append_log(f"🗂 已从选择器中选择 {len(urls)} 个视频")
            self.start_download(video_urls=urls)

    def start_download(self, video_urls=None):
        """Start the download process. If video_urls is given (from the picker),
        those exact URLs are downloaded instead of scraping the listing."""
        url = self.download_url_input.text().strip()
        save_dir = self.download_save_dir_input.text().strip()
        pattern = "auto"  # link pattern is auto-detected from the listing page

        # After-download processing mode: none / immediate / batch.
        process_mode = self.process_mode_combo.currentData()
        immediate_processing = (process_mode == "immediate")
        max_concurrent = self.concurrent_spinbox.value() if immediate_processing else 1
        
        # Get time range settings from the download-mode picker.
        mode = self.download_mode_combo.currentData()
        time_range = None
        use_percentages = False
        download_full = False

        if mode == "same":
            # Reuse the processing range. Selecting this mode auto-enables the
            # processing checkbox (see on_download_mode_changed); guard anyway.
            if not self.use_time_range_chk.isChecked():
                self.use_time_range_chk.setChecked(True)
            start_pct = self.range_slider.start()
            end_pct = self.range_slider.end()
            if end_pct <= start_pct:
                self.append_log("⚠️ 时间范围无效：结束时间必须晚于开始时间")
                return
            time_range = (float(start_pct), float(end_pct))
            use_percentages = True
            self.append_log(f"⏱️ 正在下载百分比范围：{start_pct}% - {end_pct}%")
        elif mode == "specific":
            start_s = self.download_start_input.value()
            end_s = self.download_end_input.value()
            if end_s <= start_s:
                self.append_log("⚠️ 范围无效：结束位置必须大于开始位置")
                return
            time_range = (float(start_s), float(end_s))
            use_percentages = False
            self.append_log(f"⏱️ 正在下载时间范围：{start_s} 秒 - {end_s} 秒")
        else:  # "full"
            download_full = True
            self.append_log("📥 正在下载完整视频")
        
        # Validation
        if not url:
            self.append_log("⚠️ 请输入网址")
            return
        
        if not save_dir:
            self.append_log("⚠️ 请选择保存目录")
            return
        
        # Check if URL is valid
        if not url.startswith(("http://", "https://")):
            self.append_log("⚠️ 网址必须以 http:// 或 https:// 开头")
            return
        
        # Check if already running
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.isRunning():
            self.append_log("⚠️ 下载任务正在进行中！")
            return
        
        # Clear log and start
        self.log_output.clear()
        self._show_progress(True)
        self.append_log("=== 开始下载视频 ===")
        self.append_log(f"🌐 网址：{url}")
        self.append_log(f"📁 保存目录：{save_dir}")
        self.append_log("🔍 链接模式：自动检测")
        
        if immediate_processing:
            self.append_log("⚡ 模式：每个视频下载完成后立即处理")
            self.append_log(f"   并发下载数：{max_concurrent}")
        else:
            self.append_log("📦 模式：批量下载（全部下载完成后统一处理）")
        
        # (Range already logged per-mode above.)

        self.append_log("")
        
        # UI state changes
        self.download_progress_bar.setVisible(True)
        self.download_progress_bar.setRange(0, 100)
        self.download_progress_bar.setValue(0)
        self.process_progress_bar.setVisible(False)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.task_label.setText("🌐 正在提取视频链接…")
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Define processing callback for immediate processing
        def process_video_callback(filepath, metadata):
            """Process video immediately after download using the pipeline.
            Skips processing if the highlight mp4 already exists next to the file.
            """
            try:
                filename = os.path.basename(filepath)

                # Same naming as Run. A download batch is several files, so a
                # custom base is prefixed with the source stem.
                from modules.media.output_name import highlight_output_path
                output_file = highlight_output_path(
                    filepath, self.output_input.text().strip(), multiple=True)

                # Decide whether to skip existing highlights
                # If you later add a checkbox like self.skip_existing_highlights_chk, this will pick it up.
                skip_existing = True
                if hasattr(self, "skip_existing_highlights_chk"):
                    skip_existing = self.skip_existing_highlights_chk.isChecked()

                # Header in log
                self.append_log(f"\n{'='*60}")
                self.append_log(f"🎬 立即处理：{filename}")
                self.append_log(f"{'='*60}")

                # Auto-add downloaded video to file list (GUI-thread safe)
                if self.auto_add_downloaded_chk.isChecked():
                    existing = self.get_file_list()
                    if filepath not in existing:
                        QMetaObject.invokeMethod(
                            self.file_list, "addItem",
                            Qt.QueuedConnection,
                            Q_ARG(str, filepath)
                        )
                        self.append_log(f"📋 已添加到文件列表：{filename}")

                # --- SKIP if highlight already exists ---
                if skip_existing and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    self.append_log(f"⏭️ Skipping processing (highlight exists): {os.path.basename(output_file)}")
                    self.append_log(f"{'='*60}\n")

                    return {
                        'processed_at': time.time(),
                        'filename': filename,
                        'highlight_file': output_file,
                        'success': True,
                        'skipped': True
                    }

                # Build config for this single video
                config = self.build_pipeline_config()
                config['output_file'] = output_file

                self.append_log(f"📁 Output will be: {os.path.basename(output_file)}")
                self.append_log("")

                # Run pipeline synchronously (this blocks the download worker thread by design)
                try:
                    from pipeline import run_highlighter
                    cancel_flag = threading.Event()

                    # Show indeterminate processing state in GUI
                    QMetaObject.invokeMethod(
                        self, "set_process_busy",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"🔧 正在处理：{filename} | Initializing…")
                    )

                    # Thread-safe logging back to GUI
                    def log_fn(msg):
                        QMetaObject.invokeMethod(
                            self, "append_log",
                            Qt.QueuedConnection,
                            Q_ARG(str, f"  [{filename}] {msg}")
                        )

                    # Thread-safe progress updates back to GUI
                    def progress_fn(current, total, task, details):
                        QMetaObject.invokeMethod(
                            self, "update_process_progress",
                            Qt.QueuedConnection,
                            Q_ARG(int, int(current)),
                            Q_ARG(int, int(total)),
                            Q_ARG(str, f"{filename} | {task}"),
                            Q_ARG(str, str(details))
                        )

                    # Feed the live preview window, exactly as the Run button
                    # does. Without this the checkbox looks like it applies to
                    # every run, but a downloaded video processed here handed
                    # the pipeline no preview_fn at all — so the window opened,
                    # said "Waiting for the detection stage", and stayed on that
                    # for the whole run. The signal makes the thread hop; the
                    # flag is read instead of the checkbox so this stays off the
                    # GUI thread's widgets.
                    def preview_fn(frame, boxes, sec):
                        if self._preview_enabled and not cancel_flag.is_set():
                            self.preview_frame.emit(frame, boxes, sec)

                    result = run_highlighter(
                        filepath,
                        gui_config=config,
                        log_fn=log_fn,
                        progress_fn=progress_fn,
                        cancel_flag=cancel_flag,
                        preview_fn=preview_fn,
                    )

                    # If pipeline returns a path, use it; otherwise fall back to our expected output_file
                    highlight_path = result or output_file

                    if highlight_path and os.path.exists(highlight_path) and os.path.getsize(highlight_path) > 0:
                        self.append_log(f"✅ Highlight created: {os.path.basename(highlight_path)}")
                        self.append_log(f"{'='*60}\n")

                        return {
                            'processed_at': time.time(),
                            'filename': filename,
                            'highlight_file': highlight_path,
                            'success': True,
                            'skipped': False
                        }

                    self.append_log("⚠️ Processing completed but no highlight generated (or file missing/empty)")
                    self.append_log(f"{'='*60}\n")
                    return {'success': False, 'error': 'No highlight generated'}

                except Exception as e:
                    self.append_log(f"❌ Processing error: {e}")
                    import traceback
                    self.append_log(f"错误堆栈：\n{traceback.format_exc()}")
                    self.append_log(f"{'='*60}\n")
                    return {'success': False, 'error': str(e)}

            except Exception as e:
                self.append_log(f"❌ Callback setup error: {e}")
                import traceback
                self.append_log(f"Traceback:\n{traceback.format_exc()}")
                return {'success': False, 'error': str(e)}
            
        # Videos processed straight off the download feed the preview window
        # too, so the flag those runs read has to match the checkbox before the
        # first one starts.
        self._preview_enabled = self.live_preview_checkbox.isChecked()

        # Create download worker with processing callback
        self.download_worker = DownloadWorker(
            url, save_dir, pattern,
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            immediate_processing=immediate_processing,
            max_concurrent=max_concurrent,
            process_callback=process_video_callback if immediate_processing else None,
            video_urls=video_urls
        )
        
        # Connect signals
        self.download_worker.log.connect(self.append_log)
        self.download_worker.progress.connect(self.update_download_progress)
        self.download_worker.finished.connect(self.download_done)
        self.download_worker.cancelled.connect(self.download_cancelled)
        if immediate_processing:
            self.download_worker.video_processed.connect(self.on_video_processed)
        
        self.status_timer.start(100)
        self.download_worker.start()

    def build_pipeline_config(self):
        """Build pipeline configuration from GUI settings"""
        
        def get_list_from_input(input_field):
            text = input_field.text().strip()
            if not text:
                return None
            items = [s.strip() for s in text.split(",") if s.strip()]
            return items if items else None
        
        highlight_objects = get_list_from_input(self.objects_input)
        interesting_actions = get_list_from_input(self.actions_input)
        use_transcript = self.transcript_checkbox.isChecked()
        search_keywords = get_list_from_input(self.search_keywords_input) if use_transcript else []
        
        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None
        
        config = {
            "scene_points": int(self.spin_scene_points.value()),
            "motion_event_points": int(self.spin_motion_event_points.value()),
            "motion_peak_points": int(self.spin_motion_peak.value()),
            "audio_peak_points": int(self.spin_audio_peak.value()),
            "loudness_burst_points": int(self.spin_loudness_burst.value()),
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": int(self.spin_beginning_points.value()),
            "ending_points": int(self.spin_ending_points.value()),
            "beginning_seconds": int(self.spin_beginning_seconds.value()),
            "ending_seconds": int(self.spin_ending_seconds.value()),
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "face_expression_points": int(self.spin_face_expression.value()),
            "face_expression_labels": self.selected_face_labels(),
            "clip_time": int(self.spin_clip_time.value()),
            "coverage": self.slider_coverage.value() / 100.0,
            "report_only": bool(getattr(self, "_report_only", False)),
            "max_duration": int(self.spin_max_duration.value()),
            "exact_duration": exact_duration,
            "multi_signal_boost": 1.2,
            "min_signals_for_boost": 2,
            "keep_temp": self.keep_temp_chk.isChecked(),
            "export_separate_clips": self.export_clips_chk.isChecked(),
            "render_mode": self.render_mode_combo.currentData(),
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "transcript_source_lang": self.transcript_source_lang.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            # The spoken language has one home: Transcript Settings.
            "source_lang": self.transcript_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "vr_mode": self.vr_mode_chk.isChecked(),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_type": self.object_detector_choice()[0],
            "yolo_model_size": self.yolo_model_combo.currentData(),
            "yolo_custom_model_path": self.object_detector_choice()[1] or getattr(self, "_custom_pose_model", None),
            "sample_rate": int(self.sample_rate_spin.value()),
            "auto_min_clip": float(self.spin_auto_min_clip.value()),
            "auto_max_clip": float(self.spin_auto_max_clip.value()),
            "auto_merge_gap": float(self.spin_auto_merge_gap.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "write_highlight_report": self.why_report_chk.isChecked(),
            **self._report_config(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
            "action_backend": self.action_backend_combo.currentData(),
            "r3d_model": self.r3d_model_combo.currentData(),
            "action_models": self.action_models_combo.currentData(),
            "object_confidence": self.obj_confidence_spin.value() / 100.0,
            "force_reprocess": self.force_reprocess_checkbox.isChecked(),
        }
      
        # Add time range if enabled
        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.range_slider.start() / 100
            end_pct = self.range_slider.end() / 100
            config["use_time_range"] = True
            config["range_start"] = int(start_pct * self.current_video_duration)
            config["range_end"] = int(end_pct * self.current_video_duration)
        else:
            config["use_time_range"] = False
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}


    def on_video_processed(self, filepath, result):
        """Handle when a video is processed immediately after download"""
        filename = os.path.basename(filepath)
        if result.get('success'):
            self.append_log(f"✅ {filename} 下载并处理成功")
        else:
            self.append_log(f"⚠️ {filename} 已下载，但处理失败")


    def on_process_mode_changed(self):
        """Concurrent downloads only matter while processing overlaps downloads
        (the 'immediate' mode); grey the spinner otherwise."""
        self.concurrent_spinbox.setEnabled(
            self.process_mode_combo.currentData() == "immediate"
        )

    def on_download_mode_changed(self):
        """Show the seconds inputs only for 'specific', and make 'same' pull in
        a processing range to reuse (auto-enable 'Process only specific time
        range' so there's an actual range instead of the whole video)."""
        mode = self.download_mode_combo.currentData()
        if hasattr(self, "download_range_widget"):
            self.download_range_widget.setVisible(mode == "specific")
        if mode == "same" and not self.use_time_range_chk.isChecked():
            self.use_time_range_chk.setChecked(True)
        if mode == "specific":
            self.update_download_duration()

    def update_download_duration(self):
        """Update the duration label for the specific-range mode."""
        if self.download_mode_combo.currentData() != "specific":
            return

        start = self.download_start_input.value()
        end = self.download_end_input.value()
        
        # Ensure end is after start
        if end <= start:
            end = start + 1
            self.download_end_input.setValue(end)
        
        duration = end - start
        minutes = duration // 60
        seconds = duration % 60
        
        self.download_duration_label.setText(
            f"时长：{duration} 秒（{minutes}:{seconds:02d}）"
        )

    def download_done(self, downloaded_files):
        """Handle download completion with immediate processing support"""
        self.status_timer.stop()
        
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.is_cancelled():
            self.append_log("\n⏹️ === 下载已取消 ===")
            self.task_label.setText("⏹️ 已取消")
            self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.download_cleanup()
            return
        
        if downloaded_files:
            self.append_log(f"\n✅ === 下载完成 ===")
            self.append_log(f"📊 已成功下载 {len(downloaded_files)} 个视频")
            
            # Check if immediate processing was enabled
            if self.process_mode_combo.currentData() == "immediate":
                # Count successful processing
                if hasattr(self.download_worker, '_download_results'):
                    processed_count = sum(1 for r in self.download_worker._download_results 
                                        if r.get('processed', False))
                    self.append_log(f"🎬 已成功处理 {processed_count}/{len(downloaded_files)} 个视频")
                    
                    # List all results
                    for result in self.download_worker._download_results:
                        if result.get('success') and result.get('processed'):
                            highlight = result.get('process_result', {}).get('highlight_file')
                            if highlight:
                                self.append_log(f"  ✅ {os.path.basename(highlight)}")
                
                # Combine highlights if enabled and we have multiple
                if self.auto_combine_chk.isChecked() and len(downloaded_files) > 1:
                    self.append_log("\n🎬 正在合并全部高光…")
                    highlight_files = []
                    
                    if hasattr(self.download_worker, '_download_results'):
                        for result in self.download_worker._download_results:
                            highlight = result.get('process_result', {}).get('highlight_file')
                            if highlight and os.path.exists(highlight):
                                highlight_files.append(highlight)
                    
                    if len(highlight_files) > 1:
                        first_video_dir = os.path.dirname(highlight_files[0])
                        combined_output = os.path.join(first_video_dir, "all_highlights_combined.mp4")
                        combined_file = self.combine_highlights(highlight_files, combined_output)
                        
                        if combined_file:
                            self.append_log(f"🎉 高光合并完成：{combined_file}")
            
            self.task_label.setText("✅ 已完成！")
            self.task_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.append_log("\n⚠️ === 下载完成，但没有获得文件 ===")
            self.task_label.setText("❌ 下载失败")
            self.task_label.setStyleSheet("color: #f44336; font-weight: bold;")

        # Batch mode: downloads are done, now run the pipeline over them. Add the
        # files to the list first (batch needs them there regardless of the
        # auto-add toggle), then hand off to the pipeline.
        if self.process_mode_combo.currentData() == "batch" and downloaded_files:
            existing = self.get_file_list()
            for f in downloaded_files:
                if f not in existing and os.path.exists(f):
                    self.file_list.addItem(f)
            if self.file_list.count() > 0:
                self.append_log("\n▶️ 正在批量处理已下载的视频…")
                self.auto_start_pipeline()
                return

        self.download_cleanup()
        self._show_progress(False)

    def auto_start_pipeline(self):
        """Automatically start pipeline processing after download"""
        # Clean up download state
        self.download_cleanup()
        
        # Small delay to ensure UI updates
        QApplication.processEvents()
        
        # Now start the pipeline
        self.run_pipeline()

    def download_cancelled(self):
        """Handle download cancellation"""
        self.status_timer.stop()
        self.append_log("\n⏹️ === 用户已取消下载 ===")
        self.task_label.setText("⏹️ 下载已取消")
        self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.download_cleanup()

    def download_cleanup(self):
        """Clean up UI state after download completion/cancellation"""
        # Hide progress bar only if not auto-processing
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            self.download_progress_bar.setVisible(False)
            # If you're not auto-processing, also hide processing bar
            self.process_progress_bar.setVisible(False)

        
        # Re-enable controls
        self.download_btn.setEnabled(True)
        
        # Only re-enable cancel if not auto-processing
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("取消")
        
        # Reset task label style after 5 seconds (only if not auto-processing)
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            QTimer.singleShot(5000, lambda: self.task_label.setStyleSheet("color: #666; font-weight: bold;"))
        
        # Clean up worker
        if hasattr(self, 'download_worker') and self.download_worker:
            if self.download_worker.isRunning():
                self.download_worker.wait(1000)
            self.download_worker = None

    # --- Multi-file support methods ---
    def browse_files(self):
        """Add one or more video files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择视频", "", "视频 (*.mp4 *.mov *.avi *.mkv)"
        )
        existing = self.get_file_list()
        for path in file_paths:
            if path not in existing:
                self.file_list.addItem(path)
        
        # Auto-set output filename based on first video if output is empty or default
        if file_paths and (not self.output_input.text().strip() or 
                        self.output_input.text().strip() == "highlight.mp4"):
            first_video = file_paths[0]
            base_name = os.path.splitext(os.path.basename(first_video))[0]
            self.output_input.setText(f"{base_name}_highlight.mp4")
        
        # Update video duration for time range slider (use first video)
        if file_paths:
            self.update_video_duration(file_paths[0])
        self._sync_simple_start()

    def remove_selected_file(self):
        """Remove selected file from the list"""
        current_row = self.file_list.currentRow()
        if current_row >= 0:
            self.file_list.takeItem(current_row)
            self._sync_simple_start()

    def clear_files(self):
        """Clear all files from the list and reset output name"""
        self.file_list.clear()
        self.output_input.setText("highlight.mp4")
        # Reset video duration info
        self.current_video_duration = 0
        self.video_duration_label.setText("选择视频后即可启用时间范围控制")
        self.video_duration_label.setStyleSheet("color: #666; font-style: italic;")
        self.update_selection_info()
        self._sync_simple_start()

    def get_file_list(self):
        """Get list of all files in the list widget"""
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]
    
    def combine_highlights(self, highlight_files, output_path):
        """Combine multiple highlight videos into one.

        Thin delegate to modules.media.combine_videos.combine_videos (the same engine
        the sidecar drives), keeping this method's original contract for the Qt
        callers: None when there is nothing to combine, the lone file passed
        straight through when there is only one, otherwise the combined output
        path. All engine logging is routed through append_log."""
        if not highlight_files:
            self.append_log("⚠️ 没有可合并的高光文件")
            return None

        # Filter out None values and non-existent files
        valid_files = [f for f in highlight_files if f and os.path.exists(f)]

        if not valid_files:
            self.append_log("⚠️ 未找到有效的高光文件")
            return None

        if len(valid_files) == 1:
            self.append_log("ℹ️ 只有一个高光文件，无需合并")
            return valid_files[0]

        try:
            from modules.media.combine_videos import combine_videos

            return combine_videos(
                valid_files, output_path, log_fn=self.append_log,
            )
        except Exception as e:
            self.append_log(f"❌ 合并高光失败：{e}")
            import traceback
            self.append_log(f"Traceback:\n{traceback.format_exc()}")
            return None
            
    # --- Settings presets (presets/<name>.yaml next to config.yaml) ---
    def _set_preset_controls_enabled(self, enabled: bool):
        for widget in (self.preset_name_input, self.preset_combo,
                       self.preset_save_btn, self.preset_load_btn,
                       self.preset_delete_btn):
            widget.setEnabled(enabled)

    def _refresh_presets(self, select: str | None = None):
        from modules.system.presets import list_presets
        names = list_presets(CONFIG_FILE)
        self.preset_combo.blockSignals(True)
        self.preset_combo.clear()
        self.preset_combo.addItems(names)
        if select and select in names:
            self.preset_combo.setCurrentText(select)
        self.preset_combo.blockSignals(False)

    def _set_combo_data(self, combo, value):
        if value is None:
            return
        idx = combo.findData(value)
        if idx < 0:
            idx = combo.findText(str(value))
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _select_object_model(self, yolo_type, path):
        target = (yolo_type or "standard", path or "")
        for i in range(self.object_model_combo.count()):
            if self.object_model_combo.itemData(i) == target:
                self.object_model_combo.setCurrentIndex(i)
                return

    def _apply_settings_to_widgets(self, data: dict):
        """Push a config.yaml-shaped dict onto the controls. Missing keys stay."""
        def has(section, key):
            return isinstance(data.get(section), dict) and key in data[section]

        def get(section, key, default=None):
            block = data.get(section) or {}
            return block.get(key, default)

        video = data.get("video") or {}
        if "paths" in video and isinstance(video["paths"], list):
            self.file_list.clear()
            for path in video["paths"]:
                self.file_list.addItem(str(path))

        dl = data.get("download") or {}
        if "last_url" in dl:
            self.download_url_input.setText(str(dl.get("last_url") or ""))
        if "save_dir" in dl:
            self.download_save_dir_input.setText(str(dl.get("save_dir") or ""))
        if "auto_add" in dl:
            self.auto_add_downloaded_chk.setChecked(bool(dl["auto_add"]))
        if "auto_combine" in dl:
            self.auto_combine_chk.setChecked(bool(dl["auto_combine"]))
        self._set_combo_data(self.download_mode_combo, dl.get("download_mode"))
        self._set_combo_data(self.process_mode_combo, dl.get("process_mode"))
        if "concurrent_downloads" in dl:
            self.concurrent_spinbox.setValue(int(dl["concurrent_downloads"]))
        if "time_range_start" in dl:
            self.download_start_input.setValue(int(dl["time_range_start"]))
        if "time_range_end" in dl:
            self.download_end_input.setValue(int(dl["time_range_end"]))

        if has("highlights", "output"):
            self.output_input.setText(str(get("highlights", "output") or ""))
        for key, spin in (
            ("clip_time", self.spin_clip_time),
            ("max_duration", self.spin_max_duration),
            ("exact_duration", self.spin_exact_duration),
            ("auto_min_clip", self.spin_auto_min_clip),
            ("auto_max_clip", self.spin_auto_max_clip),
            ("auto_merge_gap", self.spin_auto_merge_gap),
        ):
            if has("highlights", key):
                spin.setValue(int(get("highlights", key) or 0))
        if has("highlights", "coverage"):
            self.slider_coverage.setValue(int(round(float(get("highlights", "coverage") or 0) * 100)))
        if has("highlights", "keep_temp"):
            self.keep_temp_chk.setChecked(bool(get("highlights", "keep_temp")))
            self.keep_temp_chk.setText(
                "保留临时片段：开" if self.keep_temp_chk.isChecked() else "保留临时片段：关")
        if has("highlights", "export_separate_clips"):
            self.export_clips_chk.setChecked(bool(get("highlights", "export_separate_clips")))
            self.export_clips_chk.setText(
                "Export clips: ON" if self.export_clips_chk.isChecked() else "Export clips: OFF")
        self._set_combo_data(self.render_mode_combo, get("highlights", "render_mode") if has("highlights", "render_mode") else None)
        if has("highlights", "use_time_range"):
            self.use_time_range_chk.setChecked(bool(get("highlights", "use_time_range")))
        if has("highlights", "range_start_pct"):
            self.range_slider.setStart(int(get("highlights", "range_start_pct") or 0))
        if has("highlights", "range_end_pct"):
            self.range_slider.setEnd(int(get("highlights", "range_end_pct") or 100))

        scoring_spins = (
            ("scene_points", self.spin_scene_points),
            ("motion_event_points", self.spin_motion_event_points),
            ("motion_peak_points", self.spin_motion_peak),
            ("audio_peak_points", self.spin_audio_peak),
            ("loudness_burst_points", self.spin_loudness_burst),
            ("keyword_points", self.spin_keyword_points),
            ("transcript_points", self.spin_transcript_points),
            ("object_points", self.spin_object),
            ("action_points", self.spin_action),
            ("face_expression_points", self.spin_face_expression),
            ("beginning_points", self.spin_beginning_points),
            ("ending_points", self.spin_ending_points),
            ("beginning_seconds", self.spin_beginning_seconds),
            ("ending_seconds", self.spin_ending_seconds),
        )
        for key, spin in scoring_spins:
            if has("scoring", key):
                spin.setValue(int(get("scoring", key) or 0))
        if has("scoring", "face_expression_labels"):
            chosen = {str(x).lower() for x in (get("scoring", "face_expression_labels") or [])}
            for name, act in self._face_label_actions.items():
                act.setChecked(name.lower() in chosen)
            self._update_face_labels_button()

        def as_csv(section, key):
            items = get(section, key) or []
            if isinstance(items, str):
                return items
            return ", ".join(str(x) for x in items)

        if has("actions", "interesting"):
            self.actions_input.setText(as_csv("actions", "interesting"))
        if has("actions", "require_objects"):
            self.actions_require_objects_chk.setChecked(bool(get("actions", "require_objects")))
        if has("objects", "interesting"):
            self.objects_input.setText(as_csv("objects", "interesting"))
        if has("objects", "confidence"):
            self.obj_confidence_spin.setValue(int(get("objects", "confidence") or 0))
        keywords = data.get("keywords") or {}
        transcript = data.get("transcript") or {}
        kw = transcript.get("search_keywords", keywords.get("interesting"))
        if kw is not None and ("search_keywords" in transcript or "interesting" in keywords):
            text = kw if isinstance(kw, str) else ", ".join(str(x) for x in kw)
            self.search_keywords_input.setText(text)
        if "enabled" in transcript:
            self.transcript_checkbox.setChecked(bool(transcript["enabled"]))
        if "model" in transcript:
            self.transcript_model_combo.setCurrentText(str(transcript["model"]))
        if "source_lang" in transcript:
            self.transcript_source_lang.setCurrentText(str(transcript["source_lang"]))
        sub = data.get("subtitles") or {}
        if "enabled" in sub:
            self.subtitles_checkbox.setChecked(bool(sub["enabled"]))
        if "target_lang" in sub:
            self.subtitle_target_lang.setCurrentText(str(sub["target_lang"]))

        adv = data.get("advanced") or {}
        if "frame_skip" in adv:
            self.frame_skip_spin.setValue(int(adv["frame_skip"]))
        if "vr_mode" in adv:
            self.vr_mode_chk.setChecked(bool(adv["vr_mode"]))
        if "object_frame_skip" in adv:
            self.obj_frame_skip_spin.setValue(int(adv["object_frame_skip"]))
        if "sample_rate" in adv:
            self.sample_rate_spin.setValue(int(adv["sample_rate"]))
        if "yolo_type" in adv or "yolo_custom_model_path" in adv:
            self._select_object_model(adv.get("yolo_type", "standard"),
                                      adv.get("yolo_custom_model_path") or "")
        if "yolo_model_size" in adv:
            self._set_combo_data(self.yolo_model_combo, adv["yolo_model_size"])
        if "action_backend" in adv:
            self._set_combo_data(self.action_backend_combo, adv["action_backend"])
        if "action_models" in adv:
            self._set_combo_data(self.action_models_combo, adv["action_models"])
        if "r3d_model" in adv:
            self._set_combo_data(self.r3d_model_combo, adv["r3d_model"])

        comp = data.get("compute") or {}
        if "backend" in comp:
            self._set_combo_data(self.backend_combo, comp["backend"])

        vis = data.get("visualization") or {}
        if "draw_object_boxes" in vis:
            self.bbox_objects_chk.setChecked(bool(vis["draw_object_boxes"]))
        if "draw_action_labels" in vis:
            self.bbox_actions_chk.setChecked(bool(vis["draw_action_labels"]))
        if "write_highlight_report" in vis:
            self.why_report_chk.setChecked(bool(vis["write_highlight_report"]))
        if "narrate_clips" in vis:
            self.narrate_clips_chk.setChecked(bool(vis["narrate_clips"]))
        if "narrate_chapters" in vis:
            self.narrate_chapters_chk.setChecked(bool(vis["narrate_chapters"]))
        if "report_serve_base" in vis:
            self.serve_base_input.setText(str(vis.get("report_serve_base") or ""))
        if "report_media_base" in vis:
            self.media_base_input.setText(str(vis.get("report_media_base") or ""))

        avoid = data.get("avoid") or {}
        if "face_recognition_enabled" in avoid:
            self.avoid_face_recognition_chk.setChecked(bool(avoid["face_recognition_enabled"]))
        self.update_selection_info()
        self._sync_simple_start()

    def save_named_preset(self):
        from modules.system.presets import safe_preset_name, save_preset
        name = self.preset_name_input.text().strip() or self.preset_combo.currentText().strip()
        safe = safe_preset_name(name)
        if not safe:
            self.append_log("⚠️ Preset name is empty or not usable as a filename.")
            return
        data = self.save_config()
        path = save_preset(safe, data, CONFIG_FILE)
        self.preset_name_input.clear()
        self._refresh_presets(select=safe)
        self.append_log(f"💾 Saved preset '{safe}' ({path})")

    def load_named_preset(self):
        from modules.system.presets import load_preset
        name = self.preset_combo.currentText().strip()
        if not name:
            self.append_log("⚠️ No preset selected.")
            return
        try:
            data = load_preset(name, CONFIG_FILE)
        except Exception as exc:
            self.append_log(f"⚠️ Could not load preset '{name}': {exc}")
            return
        # Carried keys with no widget (loudness_bursts, ui) come from here
        # when save_config writes config.yaml back.
        self.config_data = data
        self._apply_settings_to_widgets(data)
        self.config_data = self.save_config()
        self.append_log(f"📂 Loaded preset '{name}' into the controls and config.yaml")

    def delete_named_preset(self):
        from modules.system.presets import delete_preset
        name = self.preset_combo.currentText().strip()
        if not name:
            self.append_log("⚠️ No preset selected.")
            return
        if delete_preset(name, CONFIG_FILE):
            self._refresh_presets()
            self.append_log(f"🗑 Deleted preset '{name}'. config.yaml was not changed.")
        else:
            self.append_log(f"⚠️ Preset '{name}' is not on disk.")

    # --- Config persistence ---
    def load_config(self):

        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_config(self):
        # Helper function to get non-empty text or empty list
        def get_text_list(input_field):
            text = input_field.text().strip()
            if not text:
                return []
            return [s.strip() for s in text.split(",") if s.strip()]

        data = {
            "video": {"paths": self.get_file_list()},
            "download": {
                "last_url": self.download_url_input.text().strip(),
                "save_dir": self.download_save_dir_input.text().strip(),
                "auto_add": self.auto_add_downloaded_chk.isChecked(),
                "auto_combine": self.auto_combine_chk.isChecked(),
                "download_mode": self.download_mode_combo.currentData(),
                "process_mode": self.process_mode_combo.currentData(),
                "concurrent_downloads": self.concurrent_spinbox.value(),
                "time_range_start": self.download_start_input.value(),
                "time_range_end": self.download_end_input.value(),
            },
            "highlights": {
                "clip_time": int(self.spin_clip_time.value()),
                "coverage": self.slider_coverage.value() / 100.0,
                "output": self.output_input.text().strip(),
                "max_duration": int(self.spin_max_duration.value()),
                "exact_duration": int(self.spin_exact_duration.value()),
                "keep_temp": self.keep_temp_chk.isChecked(),
                "export_separate_clips": self.export_clips_chk.isChecked(),
                "render_mode": self.render_mode_combo.currentData(),
                "auto_min_clip": int(self.spin_auto_min_clip.value()),
                "auto_max_clip": int(self.spin_auto_max_clip.value()),
                "auto_merge_gap": int(self.spin_auto_merge_gap.value()),
                "use_time_range": self.use_time_range_chk.isChecked(),
                "range_start_pct": self.range_slider.start(),
                "range_end_pct": self.range_slider.end(),
            },
            "scoring": {
                "scene_points": int(self.spin_scene_points.value()),
                "motion_event_points": int(self.spin_motion_event_points.value()),
                "motion_peak_points": int(self.spin_motion_peak.value()),
                "audio_peak_points": int(self.spin_audio_peak.value()),
                "loudness_burst_points": int(self.spin_loudness_burst.value()),
                "keyword_points": int(self.spin_keyword_points.value()),
                "transcript_points": int(self.spin_transcript_points.value()),
                "object_points": int(self.spin_object.value()),
                "action_points": int(self.spin_action.value()),
                "face_expression_points": int(self.spin_face_expression.value()),
                "face_expression_labels": self.selected_face_labels(),
                "beginning_points": int(self.spin_beginning_points.value()),
                "ending_points": int(self.spin_ending_points.value()),
                "beginning_seconds": int(self.spin_beginning_seconds.value()),
                "ending_seconds": int(self.spin_ending_seconds.value()),
                "multi_signal_boost": 1.2,
                "min_signals_for_boost": 2,
            },
            "actions": {
                "interesting": get_text_list(self.actions_input),
                "require_objects": self.actions_require_objects_chk.isChecked()
            },
            "objects": {
                "interesting": get_text_list(self.objects_input),
                "confidence": self.obj_confidence_spin.value(),
            },
            "keywords": {
                "transcript_file": "transcript.txt",
                "interesting": get_text_list(self.search_keywords_input),
            },
            "transcript": {
                "enabled": self.transcript_checkbox.isChecked(),
                "model": self.transcript_model_combo.currentText(),
                "source_lang": self.transcript_source_lang.currentText(),
                "search_keywords": get_text_list(self.search_keywords_input),
            },
            "subtitles": {
                "enabled": self.subtitles_checkbox.isChecked(),
                # Mirrors transcript.source_lang so an older build (and anything
                # still reading subtitles.source_lang) sees one answer, not two.
                "source_lang": self.transcript_source_lang.currentText(),
                "target_lang": self.subtitle_target_lang.currentText(),
            },
            # Detector knobs with no widget of their own. Carried through from
            # whatever is on disk rather than omitted, because this dict is
            # written whole - anything missing here is deleted from config.yaml
            # the first time the user saves settings.
            "loudness_bursts": self.config_data.get("loudness_bursts", {}),
            "advanced": {
                "frame_skip": int(self.frame_skip_spin.value()),
                "vr_mode": self.vr_mode_chk.isChecked(),
                "object_frame_skip": int(self.obj_frame_skip_spin.value()),
                "sample_rate": int(self.sample_rate_spin.value()),
                "yolo_type": self.object_detector_choice()[0],
                "yolo_model_size": self.yolo_model_combo.currentData(),
                "yolo_custom_model_path": self.object_detector_choice()[1],
                "action_backend": self.action_backend_combo.currentData(),
                "r3d_model": self.r3d_model_combo.currentData(),
                "action_models": self.action_models_combo.currentData(),
            },
            "compute": {
                "backend": self.backend_combo.currentData(),
            },
            "visualization": {
                "draw_object_boxes": self.bbox_objects_chk.isChecked(),
                "write_highlight_report": self.why_report_chk.isChecked(),
                **self._report_config(),
                "draw_action_labels": self.bbox_actions_chk.isChecked(),
            },
            "avoid": {
                "face_recognition_enabled": self.avoid_face_recognition_chk.isChecked(),
            },
            "ui": {
                "suppress_no_cache_warning": self.config_data.get("ui", {}).get("suppress_no_cache_warning", False),
            },
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
        return data
            
    def closeEvent(self, event):
        self.save_config()
        event.accept()
        # Hard-kill on main-window close. We can't rely on app.exec() returning:
        # the timeline viewer window is kept alive (hidden) for reuse, and a
        # lingering hidden window can stop Qt from quitting. Killing here, the
        # moment the user closes the main GUI, guarantees the process dies even
        # if native FFmpeg/onnxruntime threads are stuck (which deadlock the
        # normal os._exit/ExitProcess path on Windows).
        _hard_exit(0)

    def check_worker_status(self):
        """Periodic check of worker status for UI responsiveness"""
        if self.worker and not self.worker.isRunning():
            self.status_timer.stop()

    def on_transcript_toggle(self, checked):
        """Handle transcript checkbox toggle"""
        self.transcript_source_lang.setEnabled(checked)
        self.transcript_model_combo.setEnabled(checked)
        # Keyword scoring controls live in Basic Settings but only work with a
        # transcript, so they grey out with it.
        self.search_keywords_input.setEnabled(checked)
        self.search_keywords_label.setEnabled(checked)
        self.spin_keyword_points.setEnabled(checked)
        self.subtitles_checkbox.setEnabled(checked)
        
        # If transcript is disabled, also disable subtitles
        if not checked:
            self.subtitles_checkbox.setChecked(False)
            self.on_subtitles_toggle(False)

    def on_subtitles_toggle(self, checked):
        """Handle subtitles checkbox toggle"""
        # Subtitles can only be enabled if transcript is enabled
        transcript_enabled = self.transcript_checkbox.isChecked()
        final_state = checked and transcript_enabled
        
        self.subtitle_target_lang.setEnabled(final_state)

    # --- Labels ---
    def load_labels_from_json(self, filepath):
            """Load label list from a JSON file. Handles list, dict, and nested dict formats."""
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [str(item) for item in data]
                elif isinstance(data, dict):
                    # Intel custom: has "label_to_idx" key
                    if "label_to_idx" in data:
                        return list(data["label_to_idx"].keys())
                    # Intel custom alt: has "idx_to_label" key
                    if "idx_to_label" in data:
                        return list(data["idx_to_label"].values())
                    # YOLO: has "class" key with {index: label}
                    if "class" in data:
                        return list(data["class"].values())
                    # Flat dict: {index: label} or {label: index}
                    values = list(data.values())
                    if values and isinstance(values[0], str):
                        return list(data.values())
                    else:
                        return list(data.keys())
                else:
                    self.append_log(f"⚠️ Unexpected JSON format in {filepath}")
                    return []
            except Exception as e:
                self.append_log(f"❌ Failed to load labels from {filepath}: {e}")
                return []

    def object_detector_choice(self):
        """The Advanced tab's object-model selection as (yolo_type, path) —
        exactly the pair the pipeline consumes. ("standard", "") when nothing
        custom is selected."""
        data = self.object_model_combo.currentData()
        if not data:
            return ("standard", "")
        return (data[0] or "standard", data[1] or "")

    def custom_object_class_names(self):
        """Class names of the selected custom object detector, read from the
        model's own metadata. Empty for keypoint/pose models and when no custom
        model is selected — callers fall back to the pose sidecars."""
        path = self.object_detector_choice()[1]
        if not path or not os.path.exists(path):
            return []
        from modules.system.app_paths import object_model_names
        return object_model_names(path)

    def open_object_label_selector(self):
        """Open label selector. For the custom model this offers your trained
        class names; for 'mixed' it merges those with the COCO objects;
        otherwise the standard YOLO objects."""
        yolo_type = self.object_detector_choice()[0]

        labels = []
        if "custom" in yolo_type:
            # A custom *detector* carries its class names in the model itself;
            # only fall back to the pose sidecars when it's a keypoint model.
            labels = self.custom_object_class_names()
            if not labels:
                try:
                    from modules.system.app_paths import custom_keypoint_names
                    labels = custom_keypoint_names()
                except Exception:
                    labels = []
            if not labels:
                self.append_log("⚠️ 未找到自定义类别名称（请选择模型或检查标签文件）。")

        if yolo_type != "custom":  # standard or mixed -> include COCO objects
            if os.path.exists(YOLO_OBJECTS_LABELS_FILE):
                labels = labels + self.load_labels_from_json(YOLO_OBJECTS_LABELS_FILE)

        if not labels:
            self.append_log("⚠️ 所选模型没有可用标签。")
            return

        current = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
        title = ("选择物体标签（自定义 + YOLO）" if yolo_type == "custom_mixed"
                 else "选择标签（自定义模型）" if yolo_type == "custom"
                 else "选择物体标签（YOLO）")
        dlg = LabelSelectorDialog(title, labels, current, self)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_labels()
            self.objects_input.setText(", ".join(selected))
            self.append_log(f"✅ 已加载 {len(selected)} 个物体标签")

    def _siglip_action_labels(self):
        """``(head name, classes)`` when the action backend resolves to SigLIP2
        and a head is installed, else None. With that backend the actions to
        look for are the user's own classes, not a built-in label list."""
        backend = self.action_backend_combo.currentData()
        if backend not in ("siglip2", "auto"):
            return None
        from modules.vision import action_siglip
        if backend == "auto" and not action_siglip.available():
            return None
        return action_siglip.installed_head_classes()

    def open_action_label_selector(self):
        """Open label selector based on current backend and action models settings."""
        siglip = self._siglip_action_labels()
        if siglip is not None:
            name, labels = siglip
            current = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
            dlg = LabelSelectorDialog(
                f"选择动作标签（动作头 {name}——{len(labels)} 个类别）",
                sorted(labels), current, self)
            if dlg.exec() == QDialog.Accepted:
                selected = dlg.get_selected_labels()
                self.actions_input.setText(", ".join(selected))
                self.append_log(f"✅ 已从 {name} 加载 {len(selected)} 个动作标签")
            return
        backend = self.action_backend_combo.currentData()
        action_models = self.action_models_combo.currentData()

        # R3D-only always uses Kinetics-400
        if backend in ("r3d_cuda", "r3d_cpu"):
            action_models = "intel_only"

        if action_models == "custom_only":
            label_file = INTEL_CUSTOM_LABELS_FILE
            title = f"选择动作标签（自定义微调——{self._custom_ov_count} 个类别）"
        elif action_models == "intel_only":
            label_file = KINETICS_400_LABELS_FILE
            title = "选择动作标签（Intel Kinetics-400——400 个类别）"
        elif action_models == "r3d_custom_only":
            label_file = R3D_CUSTOM_LABELS_FILE
            title = "选择动作标签（R3D 微调模型）"
        elif action_models == "mixed":
            # Show labels tagged with source model
            custom_labels = []
            intel_labels = []
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                custom_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
            if os.path.exists(KINETICS_400_LABELS_FILE):
                intel_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)

            tagged = []
            custom_set = set(l.lower() for l in custom_labels)
            intel_set = set(l.lower() for l in intel_labels)
            # Labels in both → show tagged versions
            overlap = custom_set & intel_set
            for label in sorted(custom_labels):
                if label.lower() in overlap:
                    tagged.append(f"{label} [custom]")
                else:
                    tagged.append(label)
            for label in sorted(intel_labels):
                if label.lower() in overlap:
                    tagged.append(f"{label} [intel]")
                else:
                    if label.lower() not in custom_set:  # avoid duplicates for non-overlap
                        tagged.append(label)
            tagged.sort()

            if not tagged:
                self.append_log("⚠️ 未找到标签文件")
                return
            current = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
            overlap_count = len(overlap)
            dlg = LabelSelectorDialog(
                f"选择动作标签（混合——{len(tagged)} 个标签，{overlap_count} 个重复标签）",
                tagged, current, self)
            if dlg.exec() == QDialog.Accepted:
                selected = dlg.get_selected_labels()
                self.actions_input.setText(", ".join(selected))
                self.append_log(f"✅ 已加载 {len(selected)} 个动作标签（混合）")
            return
        else:
            label_file = KINETICS_400_LABELS_FILE
            title = "选择动作标签"

        if not os.path.exists(label_file):
            self.append_log(f"⚠️ 找不到标签文件：{label_file}")
            return

        labels = self.load_labels_from_json(label_file)
        if not labels:
            self.append_log(f"⚠️ {label_file} 中没有找到标签")
            return

        current = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        dlg = LabelSelectorDialog(title, labels, current, self)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_labels()
            self.actions_input.setText(", ".join(selected))
            self.append_log(f"✅ 已从 {os.path.basename(label_file)} 加载 {len(selected)} 个动作标签")

    def setup_label_completers(self):
        if os.path.exists(YOLO_OBJECTS_LABELS_FILE):
            obj_labels = self.load_labels_from_json(YOLO_OBJECTS_LABELS_FILE)
            if obj_labels:
                completer = MultiCompleter(obj_labels, self)
                completer.setMaxVisibleItems(10)
                self.objects_input.setCompleter(completer)

    def update_actions_completer(self):
        """Update actions auto-complete labels based on selected backend and action models.

        Called from several places that can fire in one cascade (a backend change
        repopulates the model combo, which re-emits currentIndexChanged), so it
        no-ops when the selection resolves to the labels already installed."""
        backend = self.action_backend_combo.currentData()
        action_models = self.action_models_combo.currentData()

        # R3D-only always uses Kinetics-400
        if backend in ("r3d_cuda", "r3d_cpu"):
            action_models = "intel_only"

        siglip = self._siglip_action_labels()
        if siglip is not None:
            action_models = ("siglip2",) + tuple(siglip[1])

        if action_models == getattr(self, "_actions_completer_models", -1):
            return
        self._actions_completer_models = action_models

        action_labels = []
        source = None

        if siglip is not None:
            action_labels = sorted(siglip[1])
            source = f"action head {siglip[0]} ({len(action_labels)} classes)"
        elif action_models == "custom_only":
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                action_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
                source = f"Custom fine-tuned ({self._custom_ov_count} classes)"
        elif action_models == "intel_only":
            if os.path.exists(KINETICS_400_LABELS_FILE):
                action_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)
                source = "Intel Kinetics-400 (400 classes)"
        elif action_models == "r3d_custom_only":
            if os.path.exists(R3D_CUSTOM_LABELS_FILE):
                action_labels = self.load_labels_from_json(R3D_CUSTOM_LABELS_FILE)
                source = f"R3D fine-tuned ({len(action_labels)} classes)"
        elif action_models == "mixed":
            custom_labels = []
            intel_labels = []
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                custom_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
            if os.path.exists(KINETICS_400_LABELS_FILE):
                intel_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)
            # Build tagged list for overlapping labels
            custom_set = set(l.lower() for l in custom_labels)
            intel_set = set(l.lower() for l in intel_labels)
            overlap = custom_set & intel_set
            tagged = []
            for label in custom_labels:
                tagged.append(f"{label} [custom]" if label.lower() in overlap else label)
            for label in intel_labels:
                if label.lower() in overlap:
                    tagged.append(f"{label} [intel]")
                elif label.lower() not in custom_set:
                    tagged.append(label)
            action_labels = sorted(set(tagged))
            source = f"Mixed ({len(custom_labels)} custom + {len(intel_labels)} Kinetics-400, {len(overlap)} shared, {len(action_labels)} total)"

        if action_labels:
            completer = MultiCompleter(action_labels, self)
            completer.setMaxVisibleItems(10)
            self.actions_input.setCompleter(completer)
            print(f"🔤 Actions auto-complete: {source}")
        else:
            self.actions_input.setCompleter(None)

    @Slot(str)
    def append_log(self, text: str):
        """Thread-safe log append (always executes on GUI thread)."""
        app = QApplication.instance()
        gui_thread = app.thread() if app else None

        if gui_thread and QThread.currentThread() != gui_thread:
            QMetaObject.invokeMethod(
                self, "append_log",
                Qt.QueuedConnection,
                Q_ARG(str, text)
            )
            return

        # --- GUI thread only below ---
        # Insert through a standalone cursor rather than QTextEdit.append(),
        # which moves the widget's own cursor and clears the user's selection —
        # that made text impossible to select/copy while logs were streaming.
        from PySide6.QtGui import QTextCursor
        scrollbar = self.log_output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        cursor = QTextCursor(self.log_output.document())
        cursor.movePosition(QTextCursor.End)
        if not self.log_output.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text)

        # Only follow the tail if the user was already at the bottom; don't yank
        # them down (and away from a selection) while they scroll back through it.
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

        page = getattr(self, "simple_page", None)
        if page is not None:
            page.append_log(text)

    def _show_progress(self, visible=True):
        # Show/hide the whole progress box. Hidden when idle so it doesn't sit
        # there empty; the tabs+log splitter above absorbs the size change.
        # The bars are made visible again by update_download/process_progress().
        self.progress_group.setVisible(visible)
        if not visible:
            self.download_progress_bar.setVisible(False)
            self.process_progress_bar.setVisible(False)
            self.hide_batch_progress()
            self.task_label.setText("就绪")
        self._sync_simple_start()

    @Slot(int, int, str, str)
    def update_pipeline_progress(self, current: int, total: int, task_name: str, details: str = ""):
        """Split the batch counter off to its own row. Everything else the
        pipeline emits is a stage of the video currently being worked on."""
        if task_name.lower().startswith("batch"):
            self.update_batch_progress(current, total, task_name, details)
        else:
            self.update_process_progress(current, total, task_name, details)

    @Slot(int, int, str, str)
    def update_batch_progress(self, current: int, total: int, task_name: str, details: str = ""):
        """Videos finished out of total, kept visible for the whole batch run."""
        # Only worth stating when there is more than one video: "1 / 1" tells
        # nobody anything they cannot see from the file list.
        counter = f"{max(0, min(current, total))}/{total}  " if total > 1 else ""
        self.batch_label.setText(f"📦 {counter}{details}")
        self.batch_label.setVisible(True)
        QApplication.processEvents()

    def hide_batch_progress(self):
        self.batch_label.setVisible(False)

    @Slot(str)
    def set_process_busy(self, text: str):
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 0)  # indeterminate
        self.task_label.setText(text)
        self._sync_simple_start()

    @Slot(int, int, str, str)
    def update_download_progress(self, current: int, total: int, task_name: str, details: str = ""):
        if total > 0:
            self.download_progress_bar.setRange(0, 100)
            pct = min(100, max(0, int((current / total) * 100)))
            self.download_progress_bar.setValue(pct)
            self.download_progress_bar.setVisible(True)
            self.task_label.setText(f"⬇️ {task_name}: {pct}% - {details}")
        else:
            self.download_progress_bar.setVisible(True)
            self.download_progress_bar.setRange(0, 0)
            self.task_label.setText(f"⬇️ {task_name} - {details}")

        self._sync_simple_start()
        QApplication.processEvents()

    @Slot(int, int, str, str)
    def update_process_progress(self, current: int, total: int, task_name: str, details: str = ""):
        if total > 0:
            self.process_progress_bar.setRange(0, 100)
            pct = min(100, max(0, int((current / total) * 100)))
            self.process_progress_bar.setValue(pct)
            self.process_progress_bar.setVisible(True)
            self.task_label.setText(f"🔧 {task_name}: {pct}% - {details}")
        else:
            self.process_progress_bar.setVisible(True)
            self.process_progress_bar.setRange(0, 0)
            self.task_label.setText(f"🔧 {task_name} - {details}")

        self._sync_simple_start()
        # Keep UI responsive
        QApplication.processEvents()

    def format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def on_time_range_toggle(self, checked):
        """Show/enable the time-range controls only while the box is ticked;
        collapse the group to a single line otherwise."""
        if hasattr(self, "time_range_body"):
            self.time_range_body.setVisible(checked)
        # Always enable sliders when checkbox is checked, even without video
        self.range_slider.setEnabled(checked)
        
        # Preset buttons only work when video duration is known
        has_duration = self.current_video_duration > 0
        self.first_5min_btn.setEnabled(checked and has_duration)
        self.last_5min_btn.setEnabled(checked and has_duration)
        self.last_10min_btn.setEnabled(checked and has_duration)
        self.middle_btn.setEnabled(checked and has_duration)
        self.full_video_btn.setEnabled(checked and has_duration)
        
        self.update_selection_info()

    def on_slider_changed(self):
        self.update_selection_info()

    def update_selection_info(self):
        """Update the selection information labels"""
        start_pct = self.range_slider.start()
        end_pct = self.range_slider.end()
        
        if self.current_video_duration == 0:
            # No video loaded - show percentages
            self.start_time_label.setText(f"{start_pct}%")
            self.end_time_label.setText(f"{end_pct}%")
            
            if self.use_time_range_chk.isChecked():
                range_pct = end_pct - start_pct
                self.selection_info_label.setText(
                    f"Selection: {start_pct}% to {end_pct}% ({range_pct}% of video)"
                )
                self.selection_info_label.setStyleSheet("color: #2f81f7; font-weight: bold; font-size: 10pt;")
            else:
                self.selection_info_label.setText("选择：完整视频")
                self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")
            return
        
        # Calculate actual times when video is loaded
        start_seconds = int((start_pct / 100) * self.current_video_duration)
        end_seconds = int((end_pct / 100) * self.current_video_duration)
        duration = end_seconds - start_seconds
        
        # Update labels with time and percentage
        self.start_time_label.setText(f"{self.format_time(start_seconds)} ({start_pct}%)")
        self.end_time_label.setText(f"{self.format_time(end_seconds)} ({end_pct}%)")
        
        # Update selection info
        percentage = end_pct - start_pct
        
        if self.use_time_range_chk.isChecked():
            self.selection_info_label.setText(
                f"Selection: {self.format_time(duration)} ({percentage}% of video)"
            )
            self.selection_info_label.setStyleSheet("color: #2f81f7; font-weight: bold; font-size: 10pt;")
        else:
            self.selection_info_label.setText("选择：完整视频")
            self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")

    def update_video_duration(self, video_path):
        """Update slider ranges based on video duration"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = int(total_frames / fps) if fps else 0
            cap.release()
            
            if duration > 0:
                self.current_video_duration = duration
                
                # Update sliders with 100 steps (0-100 representing 0%-100% of video)
                self.range_slider.setRange(0, 100)
                
                # Keep existing slider values (don't reset user's choice)
                # Only update the display labels
                
                # Update labels
                self.video_duration_label.setText(
                    f"Video duration: {self.format_time(duration)} ({duration}s)"
                )
                self.video_duration_label.setStyleSheet("color: #4CAF50; font-style: italic;")
                
                # Enable controls if checkbox is checked
                if self.use_time_range_chk.isChecked():
                    self.range_slider.setEnabled(True)
                    self.first_5min_btn.setEnabled(True)
                    self.last_5min_btn.setEnabled(True)
                    self.last_10min_btn.setEnabled(True)
                    self.middle_btn.setEnabled(True)
                    self.full_video_btn.setEnabled(True)
                
                self.update_selection_info()
                return True
            else:
                self.current_video_duration = 0
                self.video_duration_label.setText("无法确定视频时长")
                self.video_duration_label.setStyleSheet("color: #f44336; font-style: italic;")
                return False
                
        except Exception as e:
            self.current_video_duration = 0
            self.video_duration_label.setText(f"读取视频时出错：{e}")
            self.video_duration_label.setStyleSheet("color: #f44336; font-style: italic;")
            return False

    def set_slider_preset(self, preset_type):
        """Set quick preset time ranges using sliders"""
        if self.current_video_duration == 0:
            self.append_log("⚠️ No video loaded")
            return
        
        duration = self.current_video_duration
        
        if preset_type == "first_5":
            # First 5 minutes or entire video if shorter
            end_seconds = min(300, duration)
            start_pct = 0
            end_pct = int((end_seconds / duration) * 100)
        elif preset_type == "last_5":
            # Last 5 minutes
            start_seconds = max(0, duration - 300)
            start_pct = int((start_seconds / duration) * 100)
            end_pct = 100
        elif preset_type == "last_10":
            # Last 10 minutes
            start_seconds = max(0, duration - 600)
            start_pct = int((start_seconds / duration) * 100)
            end_pct = 100
        elif preset_type == "middle":
            # Middle third of video
            third = duration / 3
            start_pct = int((third / duration) * 100)
            end_pct = int((2 * third / duration) * 100)
        elif preset_type == "full":
            start_pct = 0
            end_pct = 100
        else:
            return
        
        self.range_slider.setRangeValues(start_pct, end_pct)

        start_time = int((start_pct / 100) * duration)
        end_time = int((end_pct / 100) * duration)
        self.append_log(f"✅ Preset '{preset_type}': {self.format_time(start_time)} to {self.format_time(end_time)}")


    def _position_preview_window(self):
        """Place the preview window just to the right of the main GUI."""
        if self.preview_window is None:
            return
        try:
            g = self.frameGeometry()
            x = g.x() + g.width() + 8
            y = g.y()
            # Keep it on-screen: if it would overflow the screen, clamp.
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                pw = self.preview_window.width() or 720
                if x + pw > avail.right():
                    x = max(avail.left(), avail.right() - pw)
            self.preview_window.move(x, y)
        except Exception:
            pass

    def _on_live_preview_toggled(self, checked):
        """Open/close the separate preview window. Applies live to a running job."""
        if checked:
            if self.preview_window is None:
                # Top-level window (no parent) so it's freely movable and not
                # clipped to the main window; we position it ourselves.
                self.preview_window = DetectionPreviewWindow()
                self.preview_window.closed.connect(
                    lambda: self.live_preview_checkbox.setChecked(False)
                )
            self._position_preview_window()
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
        else:
            if self.preview_window is not None:
                self.preview_window.hide()
        self._preview_enabled = checked
        # Every kind of run that can be in flight: the pipeline, an on-demand
        # Analyze run, and an Analyze run started in an open timeline viewer.
        # Ticking the box mid-run starts showing frames in any of them without
        # waiting for the next one.
        for attr in ("worker", "_signal_worker", "timeline_window"):
            running = getattr(self, attr, None)
            if running is not None:
                try:
                    running.preview_enabled = checked
                except (RuntimeError, AttributeError):
                    pass   # viewer's C++ side already deleted, or no such worker

    @Slot(object, object, int)
    def on_preview_frame(self, frame_bgr, boxes, sec):
        """Draw a live detection frame (BGR ndarray + normalised boxes)."""
        if not self.live_preview_checkbox.isChecked() or self.preview_window is None:
            return
        try:
            from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap
            import numpy as np

            # Ensure a contiguous uint8 BGR array, then convert to RGB
            frame_bgr = np.ascontiguousarray(frame_bgr)
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 230, 90), 2)
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            for item in boxes or []:
                name, nx, ny, nw, nh, conf = item
                rx, ry, rw, rh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
                painter.setPen(pen)
                painter.drawRect(rx, ry, rw, rh)
                label = f"{name} {conf:.2f}"
                painter.fillRect(rx, max(0, ry - 14), 8 + len(label) * 6, 14, QColor(0, 0, 0, 160))
                painter.setPen(QColor(0, 255, 120))
                painter.drawText(rx + 3, max(10, ry - 3), label)
            painter.end()

            n = len(boxes or [])
            cap = f"t={sec//60:d}:{sec%60:02d}"
            if n:
                cap += f"  •  {n} object{'s' if n != 1 else ''}"
            self.preview_window.set_frame(pix, caption=cap)
        except Exception as e:
            print(f"⚠️ preview draw error: {e}")

    def run_pipeline(self, report_only: bool = False, simple: bool = False):
        from pipeline import run_highlighter
        """Start the pipeline processing (UPDATED for multi-file).

        ``report_only`` scores and reports without encoding anything. Tuning
        weights is cheap — detection is cached — but re-rendering a highlight
        to find out what the new weights did is not, and that cost is what
        makes trying a setting feel expensive.

        ``simple`` is the one-button workspace: built-in defaults for that run
        only. It does not rewrite the Detailed settings knobs.
        """
        self._report_only = bool(report_only)
        self._simple_run = bool(simple)
        video_paths = self.get_file_list()
        
        if not video_paths:
            self.append_log("⚠️ No videos selected!")
            return

        # Check if all files exist
        missing_files = [p for p in video_paths if not os.path.exists(p)]
        if missing_files:
            self.append_log(f"⚠️ Video file(s) not found:")
            for f in missing_files:
                self.append_log(f"  - {f}")
            return

        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Pipeline already running!")
            return
        
        # --- Validate scoring points ---
        scene_points = int(self.spin_scene_points.value())
        motion_event_points = int(self.spin_motion_event_points.value())
        motion_peak_points = int(self.spin_motion_peak.value())
        audio_peak_points = int(self.spin_audio_peak.value())
        loudness_burst_points = int(self.spin_loudness_burst.value())
        
        # Object points only count if objects are configured
        highlight_objects = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
        object_points = int(self.spin_object.value()) if highlight_objects else 0
        
        # Action points only count if actions are configured
        interesting_actions = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        action_points = int(self.spin_action.value()) if interesting_actions else 0
        
        # Transcript and keyword points only count if transcript is enabled
        use_transcript = self.transcript_checkbox.isChecked()
        keyword_points = int(self.spin_keyword_points.value()) if use_transcript else 0
        transcript_points = int(self.spin_transcript_points.value()) if use_transcript else 0
        
        beginning_points = int(self.spin_beginning_points.value())
        ending_points = int(self.spin_ending_points.value())
        
        # Expressions only count when a class is chosen, for the same reason
        # objects need a class list: the scan is skipped otherwise, so counting
        # the weight would promise points nothing can earn.
        face_points = (int(self.spin_face_expression.value())
                       if self.selected_face_labels() else 0)

        total_points = (scene_points + motion_event_points + motion_peak_points + 
                       audio_peak_points + loudness_burst_points +
                       keyword_points + transcript_points + 
                       beginning_points + ending_points + object_points + action_points
                       + face_points)
        
        if total_points == 0 and not self._simple_run:
            self.append_log("❌ ERROR: All scoring points are set to 0!")
            self.append_log("")
            self.append_log("Please configure at least one scoring point:")
            self.append_log("  • Scene points")
            self.append_log("  • Motion event points")
            self.append_log("  • Motion peak points")
            self.append_log("  • Audio peak points")
            self.append_log("  • Object points")
            self.append_log("  • Action points")
            if use_transcript:
                self.append_log("  • Keyword points (transcript enabled)")
                self.append_log("  • Transcript points (transcript enabled)")
            else:
                self.append_log("")
                self.append_log("Note: Transcript is disabled - keyword and transcript")
                self.append_log("points are not counted. Enable transcript to use them.")
            return

        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None
        
        # Blank → <video>_highlight.mp4. Otherwise the text is the base name.
        # The worker always hands run_highlighter a list, so the name has to
        # travel as output_base; the batch loop used to ignore output_file.
        from modules.media.output_name import highlight_output_path
        output_base = self.output_input.text().strip()
        output_multiple = len(video_paths) != 1
        output_file = highlight_output_path(
            video_paths[0], output_base, multiple=output_multiple)

        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None

        # Helper function to get non-empty lists
        def get_list_from_input(input_field):
            text = input_field.text().strip()
            if not text:
                return None
            items = [s.strip() for s in text.split(",") if s.strip()]
            return items if items else None
        
        highlight_objects = get_list_from_input(self.objects_input)
        interesting_actions = get_list_from_input(self.actions_input)
        use_transcript = self.transcript_checkbox.isChecked()
        search_keywords = get_list_from_input(self.search_keywords_input) if use_transcript else []
        # Avoid: pull flagged identities from the shared face bank
        avoid_bank = self._get_face_bank()
        avoid_ids = avoid_bank.avoided_ids() if avoid_bank else []

        config = {
            "scene_points": int(self.spin_scene_points.value()),
            "motion_event_points": int(self.spin_motion_event_points.value()),
            "motion_peak_points": int(self.spin_motion_peak.value()),
            "audio_peak_points": int(self.spin_audio_peak.value()),
            "loudness_burst_points": int(self.spin_loudness_burst.value()),
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": int(self.spin_beginning_points.value()),
            "ending_points": int(self.spin_ending_points.value()),
            "beginning_seconds": int(self.spin_beginning_seconds.value()),
            "ending_seconds": int(self.spin_ending_seconds.value()),
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "face_expression_points": int(self.spin_face_expression.value()),
            "face_expression_labels": self.selected_face_labels(),
            "clip_time": int(self.spin_clip_time.value()),
            "coverage": self.slider_coverage.value() / 100.0,
            "report_only": bool(getattr(self, "_report_only", False)),
            "max_duration": int(self.spin_max_duration.value()),
            "exact_duration": exact_duration,
            "multi_signal_boost": 1.2,
            "min_signals_for_boost": 2,
            "keep_temp": self.keep_temp_chk.isChecked(),
            "export_separate_clips": self.export_clips_chk.isChecked(),
            "render_mode": self.render_mode_combo.currentData(),
            "output_file": output_file,
            "output_base": output_base,
            "output_multiple": output_multiple,
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "transcript_source_lang": self.transcript_source_lang.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            # The spoken language has one home: Transcript Settings.
            "source_lang": self.transcript_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_type": self.object_detector_choice()[0],
            "yolo_model_size": self.yolo_model_combo.currentData(),
            "yolo_custom_model_path": self.object_detector_choice()[1] or getattr(self, "_custom_pose_model", None),
            "sample_rate": int(self.sample_rate_spin.value()),
            "auto_min_clip": float(self.spin_auto_min_clip.value()),
            "auto_max_clip": float(self.spin_auto_max_clip.value()),
            "auto_merge_gap": float(self.spin_auto_merge_gap.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "write_highlight_report": self.why_report_chk.isChecked(),
            **self._report_config(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
            "action_backend": self.action_backend_combo.currentData(),
            "r3d_model": self.r3d_model_combo.currentData(),
            "avoid_enabled": self.avoid_face_recognition_chk.isChecked() and bool(avoid_ids),
            "avoid_method": getattr(self, "_avoid_method", "skip"),
            "avoid_identity_ids": avoid_ids,
            "avoid_manual_ranges": self._get_manual_avoid_ranges(),
            "face_db_path": "./cache/face_db.json",
            "force_reprocess": self.force_reprocess_checkbox.isChecked(),
        }

        if self._simple_run:
            length = "medium"
            page = getattr(self, "simple_page", None)
            if page is not None:
                length = page.length_key()
            apply_simple_run(config, length)

        # Remove None values
        config = {k: v for k,v in config.items() if v is not None}

        # Clear previous logs
        self.log_output.clear()
        self._show_progress(True)
        self.append_log("=== Starting Video Highlighter Pipeline ===")
        if self._simple_run:
            self.append_log("Simple view: default scoring (motion peaks + loudness), "
                            "reel + separate clips, highlight length from this page. "
                            "Detailed knobs unchanged.")
        self.append_log(f"📁 Input: {video_paths}")
        self.append_log(f"📁 Output: {config.get('output_file', 'highlight.mp4')}")
        if config.get('draw_object_boxes') or config.get('draw_action_labels'):
            self.append_log("🎨 Bounding box visualization enabled for temp files")
        self.append_log("")

        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.range_slider.start() / 100
            end_pct = self.range_slider.end() / 100
            config["use_time_range"] = True
            config["range_start"] = int(start_pct * self.current_video_duration)
            config["range_end"] = int(end_pct * self.current_video_duration)
        else:
            config["use_time_range"] = False

        # UI state changes
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(False)
        # The pipeline reveals this again on the first batch update; a single-file
        # run must not inherit the last batch's counter.
        self.hide_batch_progress()
        self.task_label.setText("🚀 正在初始化…")
        self.run_btn.setText("⏸ 暂停")
        self.run_btn.setStyleSheet("QPushButton { background-color: #ff8c00; color: white; font-weight: bold; padding: 8px; }")
        self._set_analyze_buttons_enabled(False)   # no on-demand run while a pipeline runs
        self.cancel_btn.setEnabled(True)

        # Disable form inputs during processing
        self.file_list.setEnabled(False)
        self.output_input.setEnabled(False)
        self._set_preset_controls_enabled(False)
        self.browse_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)
        self._sync_simple_start()

        # Create and start worker
        self.worker = Worker(video_paths, config)
        self._preview_enabled = self.live_preview_checkbox.isChecked()
        self.worker.preview_enabled = self._preview_enabled
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_pipeline_progress)
        self.worker.finished.connect(self.pipeline_done)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        self.worker.preview.connect(self.on_preview_frame)
        self.worker.timeline_requested.connect(self.on_timeline_requested)
        
        # Start status checking timer
        self.status_timer.start(100)  # Check every 100ms
        
        self.worker.start()

    def cancel_pipeline(self):
        """Cancel the running pipeline or download"""
        # Check if download is running
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.isRunning():
            self.append_log("\n⏹️ === 已请求取消 ===")
            self.append_log("⏹️ Stopping download...")
            self.task_label.setText("⏹️ 正在取消下载…")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("正在取消…")
            worker = self.download_worker
            worker.cancel()
            QTimer.singleShot(10000, lambda: self.force_download_cleanup(worker))
            return
        
        # Check if pipeline is running
        if self.worker and self.worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.append_log("⏹️ Stopping pipeline...")
            self.task_label.setText("⏹️ 正在取消处理流程…")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("正在取消…")
            self.worker.cancel()
            QTimer.singleShot(10000, self.force_worker_cleanup)
            return

        # Check if an on-demand signal run is going
        if self._signal_worker and self._signal_worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.append_log("⏹️ 正在停止按需任务…")
            self.task_label.setText("⏹️ 正在取消按需任务…")
            self.cancel_btn.setText("正在取消…")
            self._signal_worker.cancel()
            return

        # Nothing is running
        self.append_log("⚠️ 当前没有正在运行的任务可取消")

    def _make_analyze_button(self, kind, label, tooltip):
        """A small 'run this one signal on demand' button, registered so the
        pipeline can grey it out while a full run (or another on-demand run) is
        active."""
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda _=False, k=kind: self.start_signal_run(k))
        # A list, not one button per kind. Composition has two Run buttons — one
        # beside the rules editor where rules are changed, one in the signals
        # list beside every other on-demand run — and keying by kind alone let
        # the second registration drop the first, leaving a live button during a
        # run that is meant to disable them all.
        self._analyze_buttons.setdefault(kind, []).append(btn)
        return btn

    def _points_group(self, title, rows):
        """One signal's scoring rows, in their own titled box.

        ``rows`` is ``((label, field), ...)`` — the same pairs a form layout
        takes, so a row moves between groups by moving one tuple.
        """
        box = QGroupBox(title)
        form = QFormLayout()
        form.setContentsMargins(8, 4, 8, 4)
        form.setSpacing(4)
        for label, field in rows:
            tip = field.toolTip() if hasattr(field, "toolTip") else ""
            if isinstance(label, str) and tip:
                lab = QLabel(label)
                lab.setToolTip(tip)
                form.addRow(lab, field)
            else:
                form.addRow(label, field)
        box.setLayout(form)
        return box

    def selected_face_labels(self):
        """The expression classes chosen to score, lowercased for the pipeline."""
        return [name for name, act in
                getattr(self, "_face_label_actions", {}).items()
                if act.isChecked()]

    def _update_face_labels_button(self, *_args):
        """Keep the button reading as what it will actually score.

        Named rather than counted while the list is short: "happy, surprise" is
        the setting itself, where "2 selected" makes the user open the menu to
        find out what they chose. The empty case has to say the consequence —
        points with nothing selected score nothing at all, and a button reading
        "none" would look like a valid state.
        """
        chosen = self.selected_face_labels()
        self.btn_face_labels.setText(
            ", ".join(chosen) if chosen else "选择表情…")

    def _rules_run_row(self):
        """The composition Run button plus a word on what it will actually do.

        No points spinbox: composed events are not scored the way the signals
        above are, so the row carries the button and a note instead of a number
        nobody would set.
        """
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(self._make_analyze_button(
            "composition", "应用规则", chr(10).join([
                "对列表中的每个视频运行已保存的构图规则，",
                "并缓存生成的事件。",
                "",
                "自动补充已勾选规则所需、但缓存中尚不存在的数据：",
                "信号规则直接分析文件；空间规则则会",
                "针对规则中指定的类别启动一次检测。两者都会",
                "写入缓存，因此修改阈值后再次运行通常只需几秒。",
                "",
                "可以安全重复运行：这些规则之前的结果会被",
                "替换，而不是重复叠加。",
            ])))
        note = QLabel("可在“高级 → 构图规则”中编辑")
        note.setStyleSheet("color: #888; font-size: 9pt;")
        h.addWidget(note)
        h.addStretch(1)
        return w

    def _points_row_with_button(self, spin, kind, label, tooltip):
        """Wrap a scoring-point spinbox and its on-demand Run button into one
        form-row field: [spinbox] [Run button] [stretch]."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(spin)
        h.addWidget(self._make_analyze_button(kind, label, tooltip))
        h.addStretch(1)
        return w

    def _set_analyze_buttons_enabled(self, enabled):
        for btn in [b for group in getattr(self, "_analyze_buttons", {}).values()
                    for b in group]:
            btn.setEnabled(enabled)

    def start_signal_run(self, kind):
        """Run one analysis signal (objects / actions / transcript / subtitles /
        motion / audio) over every video in the list, folding each result into
        that video's cache. No highlights are cut — this is the main-window twin
        of the timeline viewer's Analyze panel."""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ 完整处理流程正在运行，请先等待其结束。")
            return
        if self._signal_worker and self._signal_worker.isRunning():
            self.append_log("⚠️ 已有按需任务正在运行。")
            return

        video_paths = self.get_file_list()
        if not video_paths:
            self.append_log("⚠️ 视频列表为空。")
            return
        missing = [p for p in video_paths if not os.path.exists(p)]
        if missing:
            self.append_log("⚠️ 找不到以下视频文件：")
            for f in missing:
                self.append_log(f"  - {f}")
            return

        # Per-kind params + validation.
        params = {}
        if kind == "objects":
            objs = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
            if not objs:
                self.append_log("⚠️ 请先输入至少一个物体类别（例如 person、car）。")
                return
            params["objects"] = objs
        elif kind == "actions":
            # Blank = detect every action (same as the timeline viewer).
            params["actions"] = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        elif kind == "transcript":
            params["language"] = self.transcript_source_lang.currentText()
        elif kind == "subtitles":
            # One spoken language, from Transcript Settings. run_subtitles takes
            # the .srt's source from the transcript it actually used, so a reused
            # cached one is labelled with its own language rather than this.
            params["language"] = self.transcript_source_lang.currentText()
            params["target_lang"] = self.subtitle_target_lang.currentText()

        # UI state — reuse the pipeline's progress row.
        self.log_output.clear()
        self._show_progress(True)
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(False)
        self.hide_batch_progress()
        self.task_label.setText(f"🚀 {kind.title()}（按需运行）…")
        self._set_analyze_buttons_enabled(False)
        self.run_btn.setEnabled(False)   # no full run while an on-demand run goes
        self.cancel_btn.setEnabled(True)

        self._signal_run_paths = list(video_paths)   # for live-refreshing an open viewer
        self._signal_worker = SignalRunWorker(kind, video_paths, params)
        self._preview_enabled = self.live_preview_checkbox.isChecked()
        self._signal_worker.preview_enabled = self._preview_enabled
        self._signal_worker.log.connect(self.append_log)
        self._signal_worker.progress.connect(self.update_pipeline_progress)
        self._signal_worker.finished.connect(self._signal_run_finished)
        self._signal_worker.preview.connect(self.on_preview_frame)
        self._signal_worker.start()

    @Slot(str)
    def _signal_run_finished(self, summary):
        if summary:
            self.append_log(f"✅ {summary}")
        self._set_analyze_buttons_enabled(True)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("取消")
        self.task_label.setText("就绪")
        self.process_progress_bar.setValue(100)
        self._signal_worker = None

        # If a timeline viewer is open for one of the videos we just ran, refresh
        # it live so the new signal appears without reopening.
        tw = getattr(self, 'timeline_window', None)
        if tw is not None:
            try:
                if getattr(tw, 'video_path', None) in getattr(self, '_signal_run_paths', []):
                    tw.refresh_from_disk()
                    self.append_log("🔄 已刷新当前打开的时间线查看器。")
            except RuntimeError:
                self.timeline_window = None   # underlying window was destroyed
            except Exception as e:
                self.append_log(f"⚠️ 无法刷新时间线查看器：{e}")

    def _on_model_installed(self, exported):
        """Say, in the user-facing log, that a model of their own is now installed.

        The panel already reports the result in its own status line, but that
        line is on a tab they are about to leave. The detector will use this
        model on the next scan, which is a change to how the app behaves and
        therefore belongs where the user reads about what the app did.
        """
        try:
            names = ", ".join(getattr(exported, "class_names", []) or [])
            self.append_log(
                f"✅ 你的自定义检测器已安装（{names}）。"
                f"可在“高级 → 物体模型”中选择它。")
        except Exception as e:                     # pragma: no cover - defensive
            print(f"⚠️ Could not report the installed model: {e}")

    def toggle_run(self, *args, simple=False):
        """Run / Pause / Resume - single button.

        ``simple=True`` is the one-button workspace (built-in defaults).
        Extra *args absorb QPushButton.clicked(bool).
        """
        # Not running → start pipeline
        if not self.worker or not self.worker._is_running:
            self.run_pipeline(report_only=False, simple=simple)
            return

        # Running and not paused → pause
        if not self.worker.is_paused():
            self.worker.pause()
            self.run_btn.setText("▶ 继续")
            self.run_btn.setStyleSheet("QPushButton { background-color: #2f81f7; color: white; font-weight: bold; padding: 8px; }")
            self.task_label.setText("⏸ 已暂停")
            self.task_label.setStyleSheet("color: #ff8c00; font-weight: bold;")
            self.append_log("⏸ 处理流程已暂停")
            self._sync_simple_start()
            return

        # Paused → resume
        self.worker.resume()
        self.run_btn.setText("⏸ 暂停")
        self.run_btn.setStyleSheet("QPushButton { background-color: #ff8c00; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.setEnabled(True)  # keep enabled for pause
        self._sync_simple_start()
    def force_download_cleanup(self, worker=None):
        """Safety net (fires ~10s after a cancel request) in case the worker
        never emitted its finished/cancelled signal — e.g. it's stuck in a
        non-cancellable subprocess. Runs download_cleanup() unconditionally so
        the Download button always comes back."""
        worker = worker or getattr(self, 'download_worker', None)
        # A newer download may have replaced this worker in the meantime; don't
        # touch it — the new download owns the UI now.
        if worker is not getattr(self, 'download_worker', None):
            return
        if worker and worker.isRunning():
            self.append_log("⚠️ 正在强制终止下载任务…")
            worker.terminate()
            worker.wait(3000)
        self.download_cleanup()

    def force_worker_cleanup(self):
        """Force cleanup if worker doesn't stop gracefully"""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Forcing pipeline termination...")
            self.worker.terminate()
            self.worker.wait(3000)  # Wait up to 3 seconds
            self.pipeline_cleanup()
            self._show_progress(False)

    def update_analyzed_counter(self):
        """Refresh the analyzed-videos counter label (lifetime + this session)."""
        total = analysis_stats.get_analyzed_count()
        self.analyzed_counter_label.setText(
            f"📈 Analyzed videos: {total} (session: {self.session_analyzed_count})"
        )

    def pipeline_done(self, output_file):
        """Handle pipeline completion"""
        self.status_timer.stop()
        was_cancelled = bool(self.worker and self.worker.is_cancelled())
        
        if output_file and not was_cancelled:
            self.append_log(f"\n✅ === PIPELINE COMPLETED SUCCESSFULLY ===")
            
            # Handle both single file (string) and multiple files (list of tuples)
            if isinstance(output_file, list):
                self.append_log(f"🎬 Processed {len(output_file)} videos:")
                
                highlight_files = []  # Track valid highlight files
                
                for item in output_file:
                    # Handle tuple format: (input_path, output_path)
                    if isinstance(item, tuple):
                        input_path, result_path = item
                        file = result_path
                    else:
                        file = item
                    
                    if file:
                        self.append_log(f"   • {file}")
                        highlight_files.append(file)  # Add to list for combining
                        
                        # Check for additional files for each video
                        base_name = os.path.splitext(file)[0]
                        srt_file = f"{base_name}_{self.subtitle_target_lang.currentText()}.srt"
                        transcript_file = f"{base_name}_transcript.txt"
                        
                        if os.path.exists(srt_file): 
                            self.append_log(f"     📝 Subtitle: {srt_file}")
                        if os.path.exists(transcript_file): 
                            self.append_log(f"     📄 Transcript: {transcript_file}")
                    else:
                        self.append_log(f"   ❌ Failed to process")
                
                # Combine highlights if enabled and we have multiple files
                if len(highlight_files) > 1 and self.auto_combine_chk.isChecked():
                    self.append_log("")
                    self.append_log("=" * 60)
                    
                    # Auto-generate combined output name in same directory as first highlight
                    first_video_dir = os.path.dirname(highlight_files[0])
                    combined_output = os.path.join(first_video_dir, "all_highlights_combined.mp4")
                    
                    # Call the combine method
                    combined_file = self.combine_highlights(highlight_files, combined_output)
                    
                    if combined_file:
                        self.append_log(f"🎉 All highlights combined into: {combined_file}")
                        
                        # Calculate and display total duration
                        try:
                            cap = cv2.VideoCapture(combined_file)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            duration = total_frames / fps if fps else 0
                            cap.release()
                            self.append_log(f"   Total duration: {int(duration//60)}:{int(duration%60):02d} ({duration:.1f}s)")
                        except Exception as e:
                            self.append_log(f"   (Could not determine duration: {e})")
                    
                    self.append_log("=" * 60)
                
            else:
                # Single file
                self.append_log(f"🎬 Output saved to: {output_file}")
                
                # Check for additional files
                base_name = os.path.splitext(output_file)[0]
                srt_file = f"{base_name}_{self.subtitle_target_lang.currentText()}.srt"
                transcript_file = f"{base_name}_transcript.txt"
                
                if os.path.exists(srt_file): 
                    self.append_log(f"📝 Subtitle file: {srt_file}")
                if os.path.exists(transcript_file): 
                    self.append_log(f"📄 Transcript file: {transcript_file}")
                
            newly_analyzed = len(highlight_files) if isinstance(output_file, list) else 1
            if newly_analyzed:
                self.session_analyzed_count += newly_analyzed
                total_analyzed = analysis_stats.increment_analyzed(newly_analyzed)
                self.update_analyzed_counter()
                self.append_log(
                    f"📈 Analyzed videos: +{newly_analyzed} this run — lifetime total: {total_analyzed}"
                )

            self.task_label.setText("✅ 已完成！")
            self.task_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif not was_cancelled:
            self.append_log("\n⚠️ === PIPELINE COMPLETED WITH ERRORS ===")
            self.append_log("❌ No output file was generated. Check the log for errors.")
            self.task_label.setText("❌ 失败")
            self.task_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        # Feed analysis data to LLM chat
        if hasattr(self, 'llm_chat'):
            try:
                from modules.media.video_cache import VideoAnalysisCache
                cache = VideoAnalysisCache()
                video_path = self.get_file_list()[0] if self.get_file_list() else ""
                
                # Try loading from cache
                config = self.build_pipeline_config()
                cache_data = cache.load(video_path, params=None)  # load latest
                
                if cache_data:
                    self.llm_chat.set_analysis_data(cache_data, video_path)
                    self.append_log("🤖 LLM chat context updated with analysis data")
            except Exception as e:
                self.append_log(f"⚠️ Could not update LLM context: {e}")

        # feed cache to bot after finished pipeline
        if hasattr(self, 'llm_chat') and output_file:
            try:
                video_paths = self.get_file_list()
                video_path = video_paths[0] if video_paths else ""
                if video_path and os.path.exists(video_path):
                    config = self.build_pipeline_config()
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    video_duration = total_frames / fps if fps else 0
                    cap.release()
                    cfg_data = {}
                    if os.path.exists(CONFIG_FILE):
                        with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
                            cfg_data = yaml.safe_load(_f) or {}
                    analysis_params = build_analysis_cache_params(
                        gui_config=config, config=cfg_data,
                        sample_rate=int(self.sample_rate_spin.value()),
                        video_duration=video_duration,
                    )
                    cache = VideoAnalysisCache(cache_dir=config.get("cache_dir", "./cache"))
                    cache_data = cache.load(video_path, params=analysis_params)
                    if cache_data:
                        self.llm_chat.set_analysis_data(cache_data, video_path)
                        self.append_log("🤖 LLM chat context updated with analysis data")
            except Exception as e:
                self.append_log(f"⚠️ Could not update LLM context: {e}")

        self.pipeline_cleanup()

    def pipeline_cancelled(self):
        """Handle pipeline cancellation"""
        self.status_timer.stop()
        self.append_log("\n⏹️ === PIPELINE CANCELLED ===")
        self.task_label.setText("⏹️ 已取消")
        self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.pipeline_cleanup()

    def pipeline_cleanup(self):
        """Clean up UI state after pipeline completion/cancellation"""
        # Hide progress bar
        self.process_progress_bar.setVisible(False)
        # (Optional) keep download bar hidden too
        self.download_progress_bar.setVisible(False)

        
        # Re-enable controls
        self.run_btn.setText("运行高光分析")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.setEnabled(True)
        self._set_analyze_buttons_enabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("取消")

        # Re-enable file inputs
        self.file_list.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.output_input.setEnabled(True)
        self._set_preset_controls_enabled(True)
        self._sync_simple_start()

        # Reset task label style
        QTimer.singleShot(5000, lambda: self.task_label.setStyleSheet("color: #666; font-weight: bold;"))
        
        # Clean up worker
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait(1000)  # Wait up to 1 second
            self.worker = None

    def _get_manual_avoid_ranges(self):
        """Manual avoid ranges marked on the timeline.

        Prefer the live window (same process, always current). Fall back to the
        shared store so ranges still apply after the viewer is closed, or when
        they were marked in an earlier session. Safe if neither exists."""
        tw = getattr(self, "timeline_window", None)
        if tw is not None and hasattr(tw, "get_avoid_ranges"):
            try:
                return tw.get_avoid_ranges()
            except Exception:
                pass
        try:
            from modules.segments.manual_avoid import load_ranges
            paths = self.get_file_list()
            if paths:
                return [list(r) for r in load_ranges(paths[0])]
        except Exception:
            pass
        return []

    def on_timeline_requested(self, video_path, analysis_data):
        """Open the timeline viewer at the pipeline's request.

        Runs on the main thread (the worker emits timeline_requested), so it is
        safe to build Qt widgets here. Reuses an already-open window for the same
        video rather than constructing a second one — each SignalTimelineWindow
        pins itself in memory and can't be torn down, so a fresh one per run
        leaks ~2.5GB. See open_timeline_viewer() for the same guard.
        """
        try:
            existing = getattr(self, 'timeline_window', None)
            if existing is not None:
                try:
                    if getattr(existing, 'video_path', None) == video_path:
                        existing.show()
                        existing.raise_()
                        existing.activateWindow()
                        self.append_log("📊 Reusing open timeline viewer.")
                        return
                except RuntimeError:
                    # Underlying C++ object was deleted — fall through.
                    self.timeline_window = None

            from signal_timeline_viewer import SignalTimelineWindow
            self.append_log(f"📊 Opening timeline viewer for: {os.path.basename(video_path)}")
            self.timeline_window = SignalTimelineWindow(video_path, analysis_data)
            self.timeline_window.show()
            self.llm_chat.set_timeline_window(self.timeline_window)
            self.llm_chat.set_video_path(video_path)
            self.llm_chat.load_cache_for_video(video_path)
        except Exception as e:
            self.append_log(f"❌ Failed to open timeline viewer: {e}")
    def _why_report_candidates(self) -> list:
        """Where a report for the current selection could be, newest first.

        Mirrors pipeline.py's own naming (`os.path.splitext(OUTPUT_FILE)[0] +
        "_why.html"`, falling back to the source video's stem) rather than
        guessing, so the button and the writer cannot disagree about the path.
        Several candidates because the output name is resolved differently for a
        single file than for a batch.
        """
        out = []
        video_paths = self.get_file_list()
        from modules.media.output_name import highlight_output_path
        output_base = self.output_input.text().strip()
        multiple = len(video_paths) != 1

        for vp in video_paths:
            source_dir = os.path.dirname(vp)
            named = highlight_output_path(vp, output_base, multiple=multiple)
            out.append(os.path.splitext(named)[0] + "_why.html")
            # Older runs always wrote <stem>_highlight, whatever the field said.
            base = os.path.splitext(os.path.basename(vp))[0]
            out.append(os.path.join(source_dir, f"{base}_highlight_why.html"))
            out.append(os.path.splitext(vp)[0] + "_why.html")

        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return [p for p in uniq if os.path.exists(p)]

    def open_why_report(self):
        """Open the newest "why these moments" report for the current selection."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        found = self._why_report_candidates()
        if not found:
            if not self.get_file_list():
                self.append_log("⚠️ Add a video first — the report sits next to its highlight.")
                return
            self.append_log(
                "⚠️ No report found yet. Run the highlighter with “Write a highlight "
                "report” enabled (Advanced tab) — it is written before the video is "
                "encoded, so it appears early in the run.")
            return

        newest = max(found, key=lambda p: os.path.getmtime(p))
        if QDesktopServices.openUrl(QUrl.fromLocalFile(newest)):
            self.append_log(f"📄 Opened report: {os.path.basename(newest)}")
        else:
            # No browser association is plausible on a stripped Windows install;
            # the path is more useful than a silent failure.
            self.append_log(f"⚠️ Could not open a browser. The report is at: {newest}")
        self._offer_pro_for_report(os.path.splitext(newest)[0] + ".json")

    def _offer_pro_for_report(self, json_path):
        """After a report is opened: say so if only Pro could measure its gaps."""
        import json

        from modules.ui import pro_offer

        if not pro_offer.may_offer("report_unmeasured"):
            return
        try:
            with open(json_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except (OSError, ValueError):
            return
        self._show_pro_offer(pro_offer.for_report(report))

    # ── AI summary of the highlight report ─────────────────────────────
    def _newest_why_report_json(self):
        """The JSON beside the newest report, or None with a logged reason."""
        found = self._why_report_candidates()
        if not found:
            self.append_log("⚠️ No highlight report yet — run the highlighter "
                            "first, the summary is written into that report.")
            return None
        newest = max(found, key=lambda p: os.path.getmtime(p))
        json_path = os.path.splitext(newest)[0] + ".json"
        if not os.path.exists(json_path):
            self.append_log(f"⚠️ {os.path.basename(newest)} has no .json beside "
                            "it, so there is nothing to summarise from.")
            return None
        return json_path

    def _ai_summary_settings(self):
        """The model a report is written with, as ``(backend, name-or-path)``."""
        entry = self._active_llm_model()
        if not entry:
            return ("ollama", "llama3")
        return (entry["backend"], entry["model"])

    def _report_config(self):
        """What the run needs to write the report the way this window is set up.

        The model goes into the run's config rather than being read from
        settings inside the pipeline, so that a run narrates with the model that
        was chosen when it started — the menu can be used to pick a different
        one while a long analysis is still going, and a run that changed model
        halfway would be very hard to explain from the report afterwards.

        Built in one place because three call sites assemble a config dict, and
        three copies of these keys is three chances for one to be forgotten and
        for the setting to appear to do nothing.
        """
        return {
            "narrate_clips": self.narrate_clips_chk.isChecked(),
            "narrate_chapters": self.narrate_chapters_chk.isChecked(),
            "narration_model": self._active_llm_model() or {},
            "report_serve_base": self.serve_base_input.text().strip(),
            "report_media_base": self.media_base_input.text().strip(),
        }

    def _llm_models(self):
        """Every configured model, oldest single-model setting folded in."""
        from PySide6.QtCore import QSettings
        from modules.narration.llm_models import migrate, parse

        s = QSettings("VideoHighlighter", "Pro")
        models = parse(s.value("advisor/models"))
        if not models:
            models = migrate(models, s.value("advisor/backend"),
                             s.value("advisor/model"))
        return models

    def _save_llm_models(self, models, chosen=None):
        from PySide6.QtCore import QSettings
        from modules.narration.llm_models import label_for, serialise

        s = QSettings("VideoHighlighter", "Pro")
        s.setValue("advisor/models", serialise(models))
        if chosen is not None:
            s.setValue("advisor/model_chosen", label_for(chosen))

    def _active_llm_model(self):
        from PySide6.QtCore import QSettings
        from modules.narration.llm_models import active

        s = QSettings("VideoHighlighter", "Pro")
        return active(self._llm_models(), s.value("advisor/model_chosen"))

    def write_ai_summary(self, question=None, reading=False, model=None):
        """Generate the summary and put it in the report, then open it.

        ``reading`` swaps the task: the default asks what to change about the
        run, this asks what the footage looks like it is doing. Two different
        questions, kept in two fields on the report so a reader can always tell
        which one they are looking at.
        """
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        from modules.report import advisor
        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        model = (entry or {}).get("model", "llama3")
        mmproj = (entry or {}).get("mmproj")
        self.append_log(
            f"🤖 Asking {backend}/{model} to "
            f"{'read what happens in this cut' if reading else 'summarise the report'}… "
            "this takes a moment.")
        # The call blocks; without this the window looks hung rather than busy.
        self.ai_summary_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, model, mmproj=mmproj)
            if llm is None:
                self.append_log(
                    f"⚠️ Could not reach {backend}/{model}. The report's findings "
                    "are there without it — only the summary needs a model.")
                return
            from modules.narration.llm_models import label_for
            text = advisor.summarise_report_file(
                json_path, llm=llm, question=question or None, reading=reading,
                model_name=label_for(entry))
        except Exception as exc:
            self.append_log(f"⚠️ Summary failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.ai_summary_btn.setEnabled(True)

        if not text:
            self.append_log("⚠️ The model returned nothing; report unchanged.")
            return
        self.append_log(f"💡 {text}")
        html_path = os.path.splitext(json_path)[0] + ".html"
        if os.path.exists(html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def write_chapter_story(self, model=None):
        """Narrate every chapter of the newest report, then open it.

        One model call per chapter, so this is minutes rather than seconds and
        the log has to show progress — a silent wait of that length reads as a
        hang. The projector is asked for here, unlike everywhere else in the
        report: this is the one narration that sends pictures, and a model that
        can see the footage is the difference between describing what was said
        and describing what happened.
        """
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        import json

        from modules.report import advisor

        from modules.narration import chapter_story
        from modules.narration.llm_models import label_for

        try:
            with open(json_path, encoding="utf-8") as fh:
                chapters = json.load(fh).get("chapters") or []
        except Exception as exc:
            self.append_log(f"⚠️ Could not read the report: {exc}")
            return
        if not chapters:
            self.append_log("⚠️ This report has no chapters to tell.")
            return

        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        name = (entry or {}).get("model", "llama3")
        mmproj = (entry or {}).get("mmproj")
        self.append_log(
            f"📖 Asking {backend}/{name} to tell {len(chapters)} chapters — "
            "one call each, so this takes minutes, not seconds.")
        self.ai_summary_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, name, mmproj=mmproj, vision=True)
            if llm is None:
                self.append_log(
                    f"⚠️ Could not reach {backend}/{name}. The chapters keep "
                    "their measurements — only the telling needs a model.")
                return

            def progress(line):
                # Straight to the user's pane rather than the debug log: this
                # is the only thing moving for the next several minutes.
                self.append_log(line)
                QApplication.processEvents()

            told = chapter_story.tell_report_file(
                json_path, llm=llm, model_name=label_for(entry),
                log_fn=progress)
        except Exception as exc:
            self.append_log(f"⚠️ Telling the chapters failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.ai_summary_btn.setEnabled(True)

        if not told:
            self.append_log("⚠️ The model returned nothing; report unchanged.")
            return
        self.append_log(f"📖 Told {told} of {len(chapters)} chapters.")
        html_path = os.path.splitext(json_path)[0] + ".html"
        if os.path.exists(html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def propose_composition_rule(self, model=None):
        """Draft a rule that would let the next run check something that was said.

        The one place in this app where a model's output becomes configuration,
        so the sequence is fixed: it proposes, `rule_proposal` rejects anything
        naming a class this video has no detections for, the user reads the YAML
        and says yes, and only then is the file written. There is no path that
        skips the middle two.
        """
        import json

        from PySide6.QtWidgets import (QApplication, QInputDialog, QMessageBox)

        from modules.report import advisor

        from modules.rules import rule_proposal
        from modules.system.app_paths import composition_rules_path
        from modules.narration.llm_models import label_for

        json_path = self._newest_why_report_json()
        if not json_path:
            return
        try:
            with open(json_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception as exc:
            self.append_log(f"⚠️ Could not read the report: {exc}")
            return

        vocabulary = report.get("vocabulary") or {}
        classes = vocabulary.get("classes") or []
        if not classes:
            self.append_log(
                "⚠️ This report has no detections to build a rule from. Run "
                "object detection with a transcript first.")
            return

        # What the user wants tested. Seeded from the strongest gap so the
        # common case is a keypress, and editable because the gap is a
        # candidate rather than a question.
        # Seeded with the longest line among the gaps rather than the most
        # distinctive one. Keyness ranks "Mm-hmm." top on real footage — it is
        # genuinely concentrated and genuinely not a claim — and a dialog that
        # opens with it reads as the feature being broken. Length is a crude
        # proxy for "contains an assertion" and beats the alternative.
        gaps = vocabulary.get("gaps") or []
        seed = max((str(where.get("quote") or "")
                    for gap in gaps for where in (gap.get("chapters") or [])),
                   key=len, default="")
        claim, ok = QInputDialog.getMultiLineText(
            self, "核对视频中说过的内容",
            "下一次运行需要核对哪一句话？\n"
            "直接使用转录文本中的一句话效果最好，规则会尝试确认或反驳它。\n"
            f"当前视频可用类别：{', '.join(classes)}",
            seed)
        if not ok or not claim.strip():
            return

        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        name = (entry or {}).get("model", "llama3")
        rules_path = composition_rules_path()
        self.append_log(f"🧩 正在让 {backend}/{name} 生成用于核对该内容的规则…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, name)
            if llm is None:
                self.append_log(f"⚠️ 无法连接 {backend}/{name}。")
                return
            proposal = rule_proposal.propose(
                claim.strip(), classes, llm=llm,
                existing=rule_proposal.existing_rules(rules_path),
                gaps=gaps, claim_at=self._claim_second(report, claim),
                model_name=label_for(entry))
        except Exception as exc:
            self.append_log(f"⚠️ 规则生成失败：{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if proposal is None:
            self.append_log(
                "⚠️ No usable rule came back. Either the claim cannot be "
                "expressed with the classes this video has, or the model named "
                "one it does not have — the debug log says which.")
            from modules.ui import pro_offer
            self._show_pro_offer(pro_offer.for_unbuildable_rule())
            return

        answer = QMessageBox.question(
            self, "添加这条规则？",
            f"<b>{proposal.label}</b><br><br>"
            f"{proposal.why}<br><br>"
            f"<pre>{proposal.as_yaml()}</pre>"
            f"要将它添加到构图规则中吗？<br>"
            f"<small>{rules_path}<br>当前规则文件会先自动备份。"
            f"添加后必须重新运行物体检测，这条规则才能生效。</small>",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self.append_log("ℹ️ 未添加规则。")
            return

        try:
            rule_proposal.apply(rules_path, proposal,
                                video_path=(report.get("video") or {}).get("path"))
        except Exception as exc:
            self.append_log(f"⚠️ 无法写入规则：{exc}")
            return
        self.append_log(
            f"✅ Added '{proposal.name}' to {os.path.basename(rules_path)}. "
            "Re-run with object detection forced (a cached detection pass "
            "skips the composition engine), then tell the chapters again — the "
            "report will say whether it fired.")

    @staticmethod
    def _claim_second(report, claim):
        """Where in the video a claim was said, if it is a line of transcript.

        Matched on the stored text so the check can be filed under the chapter
        the sentence belongs to. Returns None when the user typed a question of
        their own rather than pasting a line, which is a perfectly good way to
        use this and simply carries no timestamp.
        """
        wanted = " ".join(str(claim or "").split()).lower()
        if not wanted:
            return None
        for chapter in (report.get("chapters") or []):
            for line in ((chapter.get("dialogue") or [])
                         + (chapter.get("quotes") or [])):
                text = " ".join(str(line.get("text") or "").split()).lower()
                if text and (text in wanted or wanted in text):
                    return float(line.get("start") or 0.0)
        return None

    def show_ai_summary_menu(self):
        from PySide6.QtWidgets import QMenu

        from modules.narration.llm_models import label_for

        menu = QMenu(self)
        # First, because it is the one that answers "what is in this video"
        # rather than "what should I change" — and the one people reach for.
        act_read = menu.addAction("读取此片段发生的内容…")
        # The chapter walk-through is the slow one — a call per chapter rather
        # than one for the report — so it says so on the menu rather than in a
        # log line the user reads after committing to the wait.
        act_story = menu.addAction("按章节讲述故事…（较慢）")
        # Closes the loop the other two open: they describe what was said, this
        # is how the next run gets a signal that can check it.
        act_rule = menu.addAction("核对说过的内容…")
        act_wrong = menu.addAction("这个片段有问题…")
        act_ask = menu.addAction("询问此片段…")
        act_chat = menu.addAction("在大模型对话中讨论")

        # Which model writes it. A submenu rather than a setting to go and
        # change, because the choice belongs to the run: reading a scene and
        # advising on weights suit different models, and picking one here is the
        # difference between switching and going to look for where switching
        # lives.
        models = self._llm_models()
        active_model = self._active_llm_model()
        read_with = {}
        if len(models) > 1:
            menu.addSeparator()
            sub = menu.addMenu("使用以下模型读取…")
            for entry in models:
                item = sub.addAction(label_for(entry))
                item.setCheckable(True)
                item.setChecked(entry == active_model)
                item.setToolTip(f"{entry['backend']} · {entry['model']}")
                read_with[item] = entry

        menu.addSeparator()
        act_model = menu.addAction(
            f"模型：{label_for(active_model)}…" if models else "添加模型…")

        chosen = menu.exec(self.ai_summary_opts_btn.mapToGlobal(
            self.ai_summary_opts_btn.rect().bottomLeft()))
        if chosen in read_with:
            entry = read_with[chosen]
            self._save_llm_models(models, chosen=entry)
            self.write_ai_summary(reading=True, model=entry)
        elif chosen is act_read:
            self.write_ai_summary(reading=True)
        elif chosen is act_story:
            self.write_chapter_story()
        elif chosen is act_rule:
            self.propose_composition_rule()
        elif chosen is act_wrong:
            self._report_what_is_wrong()
        elif chosen is act_ask:
            self._ask_ai_summary_question()
        elif chosen is act_chat:
            self._discuss_report_in_chat()
        elif chosen is act_model:
            self._choose_ai_summary_model()

    def _report_what_is_wrong(self):
        """Ask what disappointed the user, then answer that.

        Without this the advisor can only list everything it noticed. Naming
        the complaint is what turns "give another signal a weight" into which
        one, and why that one.
        """
        import json

        from PySide6.QtWidgets import QInputDialog
        from modules.report.highlight_advice import CONCERNS, attach_advice

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        labels = list(CONCERNS.values())
        picked, ok = QInputDialog.getItem(
            self, "这个高光片段哪里不满意？",
            "请选择最接近的问题，报告会按照这个方向重新分析：",
            labels, 0, False)
        if not ok:
            return
        concern = next(k for k, v in CONCERNS.items() if v == picked)

        try:
            with open(json_path, encoding="utf-8") as fh:
                report = json.load(fh)
            attach_advice(report, concern=concern)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)

            from modules.report.highlight_report import render_html
            html_path = os.path.splitext(json_path)[0] + ".html"
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(render_html(report))
        except Exception as exc:
            self.append_log(f"⚠️ Could not re-read the report: {exc}")
            return

        findings = report.get("advice") or []
        self.append_log(f"💡 Re-read with '{picked}' in mind — "
                        f"{len(findings)} suggestion(s):")
        for finding in findings[:3]:
            self.append_log(f"   • {finding.get('title', '')}")

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def _ask_ai_summary_question(self):
        """A typed question is about the footage, and goes to the reader.

        It went to the advisor, whose system prompt opens "you help someone tune
        a video highlight tool" and whose rules push every answer toward a weight
        to change. So a question about what happens in a video was answered by
        the persona hired to talk about settings, and came back sounding like it
        had refused — when it had simply been asked by the wrong one of the two.

        Tuning questions have their own item on this menu.
        """
        from PySide6.QtWidgets import QInputDialog

        question, ok = QInputDialog.getText(
            self, "询问这个剪辑",
            "你想了解这个视频的什么内容？\n"
            "模型会根据本次运行测量到的标记、顺序以及重复频率来回答。",
            text="这些片段整体呈现出什么规律？")
        if ok and question.strip():
            self.write_ai_summary(question.strip(), reading=True)

    def _choose_ai_summary_model(self):
        """The report's models, on one screen — the chat panel's form, listed.

        Was a chain of four prompts with no view of what was already configured
        and no way back from the second one.
        """
        from PySide6.QtWidgets import QApplication

        from modules.narration.llm_models import label_for
        from modules.ui.model_dialog import ModelDialog

        models = self._llm_models()
        active = self._active_llm_model()
        # Building it asks the Ollama server what it holds, so the dialog can
        # offer the names instead of asking the user to remember them. That is
        # a request with a timeout, and a server that is not running spends all
        # of it — once per session, since the answer is cached, but the first
        # time it should look like waiting rather than like a hang.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dialog = ModelDialog(self, models=models,
                                 chosen=label_for(active) if active else None)
        finally:
            QApplication.restoreOverrideCursor()
        dialog.exec()

        self._save_llm_models(dialog.models)
        if dialog.chosen:
            from PySide6.QtCore import QSettings
            QSettings("VideoHighlighter", "Pro").setValue(
                "advisor/model_chosen", dialog.chosen)
            self.append_log(f"🤖 The report will be written with {dialog.chosen}.")
        elif not dialog.models:
            self.append_log("🤖 No model configured for the report.")

    def _discuss_report_in_chat(self):
        """Open the LLM chat with this run's findings already in front of it."""
        json_path = self._newest_why_report_json()
        if not json_path:
            return
        widget = self._open_llm_chat_widget()
        if widget is None:
            self.append_log("⚠️ The LLM chat window is not available in this build.")
            return
        try:
            widget.seed_from_report(json_path)
        except Exception as exc:
            self.append_log(f"⚠️ Could not hand the report to the chat: {exc}")

    def _open_llm_chat_widget(self):
        """The LLM Chat tab, brought to the front."""
        widget = getattr(self, "llm_chat", None)
        if widget is None:
            return None
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            # The chat is a tab, not a window: handing it a report without
            # showing it would look like nothing happened.
            index = tabs.indexOf(widget)
            if index == -1 and widget.parentWidget() is not None:
                index = tabs.indexOf(widget.parentWidget())
            if index != -1:
                tabs.setCurrentIndex(index)
        return widget

    def open_timeline_viewer(self):
        """Open timeline viewer for the selected video"""
        video_paths = self.get_file_list()
        
        if not video_paths:
            self.append_log("⚠️ No video selected. Please add a video first.")
            return
        
        # Use the first video in the list
        video_path = video_paths[0]
        
        if not os.path.exists(video_path):
            self.append_log(f"⚠️ Video file not found: {video_path}")
            return
        
        try:
            from signal_timeline_viewer import SignalTimelineWindow

            # Reuse an existing timeline window for the same video instead of
            # building a new one. The timeline window pins itself in memory
            # (it installs an app-wide event filter, and is referenced by the
            # LLM chat), and its 4K players can't be torn down without blocking,
            # so creating a fresh one each open leaks ~2.5GB per cycle. Re-show
            # the existing one when the video matches.
            existing = getattr(self, 'timeline_window', None)
            if existing is not None:
                try:
                    same_video = (getattr(existing, 'video_path', None) == video_path)
                    if same_video:
                        # Pick up any signals added on demand since it was opened
                        # (per-signal Run buttons fold into the cache on disk).
                        #
                        # Reusing the window is not the instant path it looks
                        # like: the refresh re-ingests every signal and redraws
                        # the timeline, which on a long video takes as long as
                        # building the window did — and it blocks the GUI
                        # thread, with the old view still on screen. Without the
                        # splash, reopening looked like the app had frozen.
                        startup_splash.begin("Reopening timeline viewer",
                                             os.path.basename(video_path),
                                             steps=4, parent=self)
                        try:
                            existing.refresh_from_disk()
                        except Exception as e:
                            self.append_log(f"⚠️ Could not refresh timeline cache: {e}")
                        finally:
                            startup_splash.finish(existing)
                        # Un-mute (close() muted the audio outputs) and re-show
                        for ao_attr, obj in (('audio_output', existing),
                                             ('_audio', getattr(existing, 'realtime_preview', None))):
                            ao = getattr(obj, ao_attr, None) if obj is not None else None
                            if ao is not None:
                                try:
                                    ao.setMuted(False)
                                except Exception:
                                    pass
                        existing.show()
                        existing.raise_()
                        existing.activateWindow()
                        self.append_log("📊 Reusing open timeline viewer.")
                        return
                except RuntimeError:
                    # Underlying C++ object was deleted — fall through to recreate
                    self.timeline_window = None

            # Check if cache exists - use the same parameters as in pipeline
            from modules.media.video_cache import VideoAnalysisCache, build_analysis_cache_params
            
            # Build the same parameters that were used when processing
            # We need to recreate the analysis_params that were used
            # Let's get the current config from GUI
            config = self.build_pipeline_config()
            
            # Get video duration for parameter building
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration = total_frames / fps if fps else 0
            cap.release()
            
            # Build analysis params that match what was used
            sample_rate = int(self.sample_rate_spin.value())
            
            # Load config.yaml defaults
            cfg_data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
            
            analysis_params = build_analysis_cache_params(
                gui_config=config,
                config=cfg_data,
                sample_rate=sample_rate,
                video_duration=video_duration
            )
            
            # Try to load with these params first
            cache = VideoAnalysisCache()
            cache_data = cache.load(video_path, params=analysis_params)
            
            if not cache_data:
                import json
                from pathlib import Path
                
                video_hash = cache._get_video_hash(video_path)
                cache_dir = Path("./cache")
                matching_files = list(cache_dir.glob(f"{video_hash}*.cache.json"))
                
                if matching_files:
                    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        cache_data = json.load(f)
                    self.append_log(f"✅ Loaded cache: {latest_file.name}")
                else:
                    # Check if user suppressed this warning
                    suppress = self.config_data.get("ui", {}).get("suppress_no_cache_warning", False)
                    
                    if not suppress:
                        dlg = NoAnalysisWarningDialog(self)
                        if dlg.exec() != QDialog.Accepted:
                            return  # User clicked Cancel
                        
                        if dlg.dont_show_chk.isChecked():
                            # Persist the preference
                            if "ui" not in self.config_data:
                                self.config_data["ui"] = {}
                            self.config_data["ui"]["suppress_no_cache_warning"] = True
                            self.save_config()
                    
                    self.append_log("⚠️ Opening timeline without signal data — run pipeline to populate signals.")
                    cache_data = {}

            
            self.append_log(f"📊 Opening timeline viewer for: {os.path.basename(video_path)}")

            # Building this window takes several seconds on a real analysis —
            # the signal timeline and the assistant panel are most of it — and
            # it blocks the GUI thread, so without the splash the app just
            # appears to hang. The window reports its own stages (see
            # signal_timeline_viewer.init_ui).
            startup_splash.begin("Opening timeline viewer",
                                 os.path.basename(video_path), steps=6,
                                 parent=self)
            startup_splash.stage("Reading the analysis cache…")
            window = None
            try:
                # Create and show the timeline window
                window = SignalTimelineWindow(video_path, cache_data)
                # An Analyze run started over there detects over the whole
                # video just as a pipeline stage does, so it feeds the preview
                # window this side owns. Queued (the frames come off the
                # viewer's analysis thread), and gated on the checkbox by the
                # emitting end.
                window.preview_frame.connect(self.on_preview_frame)
                window.preview_enabled = self._preview_enabled
                self.timeline_window = window
                window.show()
            finally:
                # finally: a viewer that fails half-way must not leave an
                # always-on-top splash stranded over the app with no window
                # behind it to explain itself. `window` stays None in that
                # case, so a *previous* viewer is never raised by mistake.
                startup_splash.finish(window)
            # Connect LLM chat to timeline and video
            self.llm_chat.set_timeline_window(self.timeline_window)
            self.llm_chat.set_video_path(video_path)
            self.llm_chat.load_cache_for_video(video_path)

        except ImportError as e:
            self.append_log(f"❌ Failed to import timeline viewer: {e}")
            self.append_log("   Make sure signal_timeline_viewer.py is in the same directory.")
        except Exception as e:
            self.append_log(f"❌ Failed to open timeline viewer: {e}")
            import traceback
            self.append_log(traceback.format_exc())

def _hard_exit(exit_code: int = 0):
    """Terminate the process immediately, bypassing slow/hanging native teardown.

    os._exit() is NOT safe enough on Windows — it calls ExitProcess, which runs
    DLL detach and tries to terminate threads cleanly. If a native thread is
    stuck (FFmpeg 4K decoder, onnxruntime/InsightFace mid-inference), ExitProcess
    deadlocks and the process never dies. TerminateProcess on our own process is
    the hardest kill available: it terminates every thread immediately with no
    cleanup, so a stuck decoder/inference thread can't block exit.

    Config is saved in the main window's closeEvent, so nothing is left to
    persist by the time this runs.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if sys.platform == "win32":
        # On Windows, os.kill() with any signal other than CTRL_C_EVENT /
        # CTRL_BREAK_EVENT unconditionally calls TerminateProcess on our own
        # process — the hardest kill available, and (unlike a raw ctypes
        # TerminateProcess call) Python handles the process handle correctly,
        # so it can't be silently truncated/failed on 64-bit.
        try:
            import signal
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        # Fallback: raw TerminateProcess with correct 64-bit handle types.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), exit_code & 0xFFFF)
        except Exception:
            pass
    os._exit(exit_code)


if __name__ == "__main__":
    # freeze_support() runs at the top of the file, before the heavy imports.
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    reset_duration_method_cache()
    # Nobody should have to install ffmpeg: pip already brought one
    # (imageio-ffmpeg). Give it its plain name on PATH before anything runs it.
    from modules.media.ffmpeg_tools import ensure_ffmpeg_on_path
    ensure_ffmpeg_on_path()

    # `--smoke-test <video>` runs the packaged app's riskiest paths with no
    # window and exits with the result; CI runs it on the built mac .app (see
    # modules/system/smoke_test.py). Here, after every import above, so a crash
    # on import still fails it — and before QApplication, which it does not need.
    if "--smoke-test" in sys.argv:
        from modules.system import smoke_test
        _smoke_code = smoke_test.main(sys.argv)
        sys.stdout.flush()
        # Not _hard_exit: on Windows it kills the process with SIGTERM, and
        # the exit code is the whole answer the workflow reads.
        os._exit(_smoke_code)

    # Disable D3D11VA hardware acceleration in Qt multimedia's FFmpeg backend.
    # On some Windows systems D3D11VA initialisation fails for H.264, causing
    # noisy warnings even though playback still works via software decoding.
    os.environ.setdefault("QT_FFMPEG_DECODING_HWACCEL", "none")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.ffmpeg=false")

    # Give Windows an explicit AppUserModelID so the taskbar groups this app
    # under our own icon instead of the generic python.exe host. Must run
    # before any window is created.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "VideoHighlighter.App"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)

    # What we are drawing on, written down before anything draws. A window that
    # dies while being resized on a large scaled display leaves no traceback, so
    # the screen geometry and scale factors are the evidence.
    from modules.system import display_info
    display_info.log(app)

    # Central theme: one graphite + accent stylesheet for all base widgets.
    # Additive — screens with their own inline styles still override it.
    _ui_theme.apply(app)
    # The settings screens are tall scrolling columns of spin boxes; Qt's
    # default hands the wheel to whichever one is under the cursor, so scrolling
    # past the Scoring Points panel quietly re-scores the next run. Wheel scrolls
    # the panel instead, everywhere in the app.
    from modules.ui.wheel_guard import install as _install_wheel_guard
    _install_wheel_guard(app)
    # One icon for every window the app opens (main window, timeline viewer,
    # dialogs). Set on the QApplication so nothing has to remember to do it.
    # The .ico carries every size from 16 to 256px, all of them the same logo
    # — 16-48px have their brightness and contrast lifted, because at taskbar
    # size the artwork's thin outlines and dark glass otherwise disappear into a
    # dark taskbar. It previously carried a separate simplified mark at those
    # sizes, which was drawn from an older logo and so kept shipping it long
    # after the artwork changed.
    _icon_path = _resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(_icon_path):
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(_icon_path))

    # `--timeline <video>` opens just the Signal Timeline viewer for one video
    # instead of the full GUI. The packaged build is a single exe, so this is how
    # another process (the web UI's sidecar) asks for the viewer — without it,
    # the only way in was to launch the whole application.
    #
    # Placed after the theme, wheel guard and window icon are installed on the
    # QApplication: the viewer is a window like any other, and launching it
    # ahead of that setup shipped it unthemed and carrying the old mark.
    if "--timeline" in sys.argv:
        idx = sys.argv.index("--timeline")
        video = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else ""
        if not video or not os.path.exists(video):
            print(f"--timeline needs an existing video path (got {video!r})")
            _hard_exit(2)
        from signal_timeline_viewer import SignalTimelineWindow
        win = SignalTimelineWindow(video)
        win.show()
        _hard_exit(app.exec())

    # Hand over from the bootloader's splash to the Qt one, which can keep
    # reporting through the window build (the remaining seconds) and follows
    # the app's theme. begin() closes the native splash once this is painted.
    startup_splash.begin(f"VideoHighlighter {__edition__}",
                         f"Version {__version__}", steps=2)

    # Reopen the live debug-log window if it was on last session (needs the
    # QApplication, hence here and not earlier).
    debug_console.restore_console_preference()

    startup_splash.stage("Building the workspace…")
    gui = VideoHighlighterGUI()
    gui.show()
    startup_splash.finish(gui)
    exit_code = app.exec()

    # Backup hard-exit in case app.exec() does return (main closeEvent already
    # hard-exits, so this is belt-and-suspenders).
    _hard_exit(exit_code)