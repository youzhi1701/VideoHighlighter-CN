"""The panel that turns labelled examples into a model, without a command line.

Everything this drives already exists and is tested without Qt:
``modules.vision.label_store`` assembles a COCO dataset, ``training.train_yolox_run``
fine-tunes on whatever device is present, and ``training.export_yolox`` converts
the result into the IR the app's detector loads. Until now the only way to reach
any of it was a Python prompt, which meant in practice it was run by whoever
wrote it.

So this widget holds no logic of its own. It picks paths, starts a worker, shows
what the worker says, and stops it when asked. Every number it displays comes
from a callback the training loop already emitted.

**Progress is reported in the user's terms.** Before the run: how long it will
take, on which hardware, and whether that number was measured on this computer
(``training.train_estimate``). During it: which stage it is in, time 已用 and
left, and after every round a sentence about what the model now finds on frames
it was not trained on (``modules.vision.training_preview``) — with a "Watch it learn"
window for anyone who wants to see it. Loss values go to the debug log — they
are the right diagnostic and the wrong progress indicator, because nobody
outside this file can say whether 2.48 is good.

Placement: currently a tab, which is the interim home. The design in
``docs/CUSTOM-MODEL-TRAINING.md`` puts this in the dock beside the video, as a
list of things being taught, each row offering exactly one next action. Nothing
in this widget depends on where it lives, so that move is a change of parent.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from modules.ui.collapsible import CollapsibleSection
from modules.ui.theme import DARK as THEME


class ObjectTrainingWorker(QObject):
    """Assemble, train, export — off the GUI thread.

    One worker for the whole chain rather than three, because the user asked
    for a model and the intermediate artifacts are not decisions they made.
    A failure anywhere surfaces as one error with the stage named.
    """

    progress = Signal(int, str)        # percent, human-readable status
    stage = Signal(int)                # index into STAGES
    round_done = Signal(object)        # modules.vision.training_preview.RoundSnapshot
    finished = Signal(object)          # ExportResult
    error = Signal(str)

    STAGES = ("收集帧", "准备模型", "训练中", "保存")

    def __init__(self, store_path: str, work_dir: str, dest_dir: str,
                 epochs: int, batch_size: int, size: str):
        super().__init__()
        self._store_path = store_path
        self._work_dir = work_dir
        self._dest_dir = dest_dir
        self._epochs = epochs
        self._batch_size = batch_size
        self._size = size
        self._stop = False
        # Set from the GUI thread when the live view opens or closes; a plain
        # bool read between rounds, so no lock is needed.
        self.draw_rounds = False

    def cancel(self) -> None:
        """Thread-safe: the loop polls this between steps."""
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    @Slot()
    def run(self) -> None:
        stage = "starting"
        try:
            from modules.vision.label_store import LabelStore, build_dataset
            from training.train_yolox_run import Cancelled, train
            from training.export_yolox import install

            stage = "reading the labels"
            store = LabelStore(self._store_path).load()
            counts = store.counts()
            if not counts:
                raise ValueError(
                    "No accepted labels in this store. Mark some examples and "
                    "accept them before training.")

            import time
            run_started = time.perf_counter()
            stage = "collecting the frames"
            self.stage.emit(0)
            self.progress.emit(0, "收集帧 from your videos...")
            dataset_dir = os.path.join(self._work_dir, "dataset")
            summary = build_dataset(
                store, dataset_dir,
                progress=lambda done, total: self.progress.emit(
                    int(5 * done / max(1, total)),
                    f"收集帧... {done} of {total}"),
            )
            if self._should_stop():
                raise Cancelled("stopped before training")

            trained = summary["splits"]["train"]["images"]
            checked = summary["splits"].get("val", {}).get("images", 0)
            extract_per_frame = ((time.perf_counter() - run_started)
                                 / max(1, trained + checked))
            if trained == 0:
                raise ValueError("No frames could be read from your videos.")

            stage = "preparing the model"
            self.stage.emit(1)
            from training.train_yolox_run import pretrained_path
            first_time = not os.path.exists(pretrained_path(self._size))
            self.progress.emit(5, "准备模型..." + (
                " 首次运行需要下载初始权重（20–70 MB）。"
                if first_time else ""))

            from modules.vision.training_preview import pick_frames, snapshot
            preview_frames = pick_frames(dataset_dir)
            history: list = []
            learning_started = [False]

            def on_epoch(report):
                snap = snapshot(report, preview_frames, history, draw=self.draw_rounds)
                print(f"[train] round {report.epoch}: found {snap.found}/{snap.expected}, "
                      f"{snap.false_alarms} false alarm(s), train loss "
                      f"{report.train_loss:.4f}, val loss {report.val_loss:.4f}")
                self.round_done.emit(snap)

            stage = "training"

            def on_progress(update):
                if not learning_started[0]:
                    learning_started[0] = True
                    self.stage.emit(2)
                # 5% was the frame collection; the rest is the training run.
                percent = 5 + int(93 * update.fraction)
                left = _friendly_time(update.eta)
                self.progress.emit(percent, (
                    f"训练中... round {update.epoch} of {update.total_epochs}"
                    + (f"，预计剩余 {left}" if left else "")))

            result = train(
                dataset_dir=dataset_dir,
                output_dir=os.path.join(self._work_dir, "checkpoint"),
                epochs=self._epochs,
                batch_size=self._batch_size,
                size=self._size,
                progress=on_progress,
                should_stop=self._should_stop,
                on_epoch=on_epoch,
            )
            export_started = time.perf_counter()

            stage = "saving the model"
            self.stage.emit(3)
            self.progress.emit(98, "保存 the model...")
            exported = install(result.weights_path, dest_dir=self._dest_dir)
            _record_speed(
                result, self._size, extract_per_frame,
                # setup + export; skipped on a first run, whose download would skew it
                None if first_time else
                (result.seconds - result.loop_seconds) + (time.perf_counter() - export_started))
            exported.trained_on = trained          # for the finished message
            exported.checked_on = checked
            exported.best_val_loss = result.best_val_loss
            exported.last_round = history[-1] if history else None
            exported.device = result.device
            self.progress.emit(100, "完成。")
            self.finished.emit(exported)

        except Exception as exc:                   # noqa: BLE001 - never crash the GUI
            name = type(exc).__name__
            if name == "Cancelled":
                self.error.emit("已停止。")
                return
            import traceback
            traceback.print_exc()
            self.error.emit(f"在“{stage}”阶段失败：{exc}")


def _record_speed(result, size: str, extract_per_frame=None, fixed_seconds=None) -> None:
    """Remember how fast this computer actually trained, so the next estimate
    is measured rather than guessed. Never fails a run."""
    try:
        from training.train_estimate import ThroughputStore, default_store_path, device_kind
        from training.train_yolox_run import DEFAULT_IMAGE_SIZE
        store = ThroughputStore(default_store_path()).load()
        store.record(device_kind(result.device), size, DEFAULT_IMAGE_SIZE,
                     result.train_images_per_second, result.val_images_per_second)
        store.record_overheads(extract_per_frame, fixed_seconds)
        store.save()
        print(f"[train] measured {result.train_images_per_second:.1f} img/s training, "
              f"{result.val_images_per_second:.1f} img/s validating on {result.device}")
    except Exception as exc:                    # noqa: BLE001
        print(f"[train] could not record training speed: {exc}")


def _probe_training_device() -> tuple:
    """(torch device string, human name) the run will use. Imports torch, so
    it is called off the GUI thread."""
    from training.train_yolox_run import resolve_device
    device = resolve_device("AUTO")
    name = ""
    try:
        import torch
        if device.startswith("xpu"):
            name = torch.xpu.get_device_name(0)
        elif device.startswith("cuda"):
            name = torch.cuda.get_device_name(0)
    except Exception:                           # noqa: BLE001
        pass
    return device, name


def _elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _parse_friendly(text: str) -> float:
    """Inverse of ``_friendly_time``, for the clock between progress events."""
    m = re.match(r"([\d.]+) (second|minute|hour)", text or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"second": 1, "minute": 60, "hour": 3600}[m.group(2)]


def _friendly_time(seconds: float) -> str:
    """"about 3 minutes left" beats "eta 184.2s" for someone deciding whether
    to go and make tea. Empty string when there is no estimate yet, so the
    caller can leave the phrase out rather than print "about 0 seconds"."""
    seconds = int(seconds or 0)
    if seconds <= 0:
        return ""
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


class ObjectTrainingSection(QWidget):
    """Pick a set of labels, train a detector, install it.

    Objects are taught from **boxes in frames**: where a thing is, in a still.
    That is a different kind of example from an action, which is why this and
    :class:`ActionTrainingSection` are separate rather than one form with a
    mode switch — they take different data and produce different models.
    """

    model_installed = Signal(object)     # ExportResult, for the host to react to
    _device_found = Signal(str, str)     # from the probe thread: device, name

    # Deliberately modest defaults. A first run should finish while somebody is
    # still interested in it; the advanced section is there for the second one.
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH = 8

    def __init__(self, parent=None, store_path: str = ""):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[TrainingWorker] = None
        self._store_path = store_path
        self._frames = (0, 0)                # (train, val) the store will produce
        self._device: Optional[tuple] = None  # (device, name) once probed
        self._probing = False
        self._started_at = 0.0
        self._time_left = 0.0
        self._estimate_seconds = 0.0
        self._preview = None                 # TrainingPreviewWindow, when open
        self._rounds: list = []              # every RoundSnapshot of this run
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_clock)
        self._device_found.connect(self._on_device_found)
        self._build_ui()
        if store_path:
            self._load_store(store_path)

    # ── layout ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        explain = QLabel(
            "让应用学会识别你自己的物体。先标记示例，程序会训练一个小型检测模型，"
            "用于在每个视频中寻找这些物体。\n"
            "初次训练可能无法识别全部目标；随着你继续添加示例，模型会逐步改进。"
            ""
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color:#999;")
        root.addWidget(explain)

        # -- where the labels come from --
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("示例："))
        self.store_label = QLabel("尚未选择")
        self.store_label.setStyleSheet("font-style:italic;color:#999;")
        source_row.addWidget(self.store_label, 1)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse_store)
        source_row.addWidget(browse)
        self.import_btn = QPushButton("从标注工具导入…")
        self.import_btn.setToolTip(
            "读取 tools/labeler.py 导出的数据。点标注会被转换为固定大小的边界框，"
            "因此导入后需要先检查确认，而不会直接作为已接受标注。")
        self.import_btn.clicked.connect(self._import_labeler)
        source_row.addWidget(self.import_btn)
        root.addLayout(source_row)

        self.counts_label = QLabel("")
        self.counts_label.setWordWrap(True)
        root.addWidget(self.counts_label)

        # -- advanced, folded: the point is that nobody has to open it --
        advanced = CollapsibleSection("高级", settings_key="training/advanced")
        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.DEFAULT_EPOCHS)
        self.epochs_spin.valueChanged.connect(self._refresh_estimate)
        form.addRow("训练轮数：", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(self.DEFAULT_BATCH)
        form.addRow("每批帧数：", self.batch_spin)

        self.size_combo = QComboBox()
        for size, hint in (("nano", "最小、最快"),
                           ("tiny", "推荐"),
                           ("s", "较慢，但略微更准确")):
            self.size_combo.addItem(f"{size} - {hint}", size)
        self.size_combo.setCurrentIndex(1)
        self.size_combo.currentIndexChanged.connect(self._refresh_estimate)
        form.addRow("模型大小：", self.size_combo)
        advanced.setContentLayout(form)
        root.addWidget(advanced)

        # -- how long, said before the button is pressed --
        self.estimate_label = QLabel("")
        self.estimate_label.setWordWrap(True)
        root.addWidget(self.estimate_label)

        # -- the one button --
        self.train_btn = QPushButton("训练模型")
        self.train_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.success};color:white;"
            f"font-weight:bold;padding:10px 18px;}}")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start)
        root.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("停止")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        root.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.stage_label = QLabel("")
        self.stage_label.setTextFormat(Qt.RichText)
        self.stage_label.setVisible(False)
        root.addWidget(self.stage_label)

        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet("color:#999;")
        self.clock_label.setVisible(False)
        root.addWidget(self.clock_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # "How is it going" — one sentence per finished round.
        self.round_label = QLabel("")
        self.round_label.setWordWrap(True)
        self.round_label.setVisible(False)
        root.addWidget(self.round_label)

        self.watch_btn = QPushButton("👁 查看训练过程")
        self.watch_btn.setToolTip(
            "打开实时预览：每轮结束后，模型都会查看同一组未参与训练的帧，"
            "你可以直接看到它识别到了什么。")
        self.watch_btn.clicked.connect(self._open_preview)
        self.watch_btn.setVisible(False)
        root.addWidget(self.watch_btn)

        # Asked once the model has worked for the person, never before, and
        # never automatically: sharing is their decision, made in the wizard.
        self.share_box = QWidget()
        share_layout = QVBoxLayout(self.share_box)
        share_layout.setContentsMargins(0, 8, 0, 0)
        share_note = QLabel(
            "模型已经可以使用。是否愿意分享它，让其他人也能在自己的视频中识别"
            "相同目标？只会分享模型，绝不会上传你的视频、视频帧或音频。")
        share_note.setWordWrap(True)
        share_layout.addWidget(share_note)
        self.share_btn = QPushButton("将此模型分享到社区…")
        self.share_btn.clicked.connect(self._share)
        share_layout.addWidget(self.share_btn)
        self.share_box.setVisible(False)
        root.addWidget(self.share_box)
        self._last_export = None

        root.addStretch()
        self.setLayout(root)

    # ── choosing labels ──────────────────────────────────────────────────

    def _browse_store(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择一组示例", "", "标注文件 (*.json);;所有文件 (*)")
        if path:
            self._load_store(path)

    def _load_store(self, path: str) -> None:
        try:
            from modules.vision.label_store import LabelStore
            store = LabelStore(path).load()
        except Exception as exc:
            self._say(f"无法读取该文件：{exc}", THEME.danger)
            return

        self._store_path = path
        self.store_label.setText(os.path.basename(path))
        self.store_label.setStyleSheet("")
        counts = store.counts()
        pending = len(store.pending())

        if counts:
            described = ", ".join(f"{name} ({n})" for name, n in sorted(counts.items()))
            note = f"可以开始训练：{described}。"
            if pending:
                note += f"  还有 {pending} 个示例需要检查。"
            self.counts_label.setText(note)
            self.counts_label.setStyleSheet("")
            self.train_btn.setEnabled(True)
            try:
                from training.train_estimate import frames_in_store
                self._frames = frames_in_store(store)
            except Exception as exc:            # noqa: BLE001
                print(f"[training] could not count frames: {exc}")
                self._frames = (0, 0)
            self._refresh_estimate()
        else:
            self.counts_label.setText(
                f"目前还没有已接受的示例"
                + (f" —— 有 {pending} 个示例等待检查。"
                   if pending else "。"))
            self.counts_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            self._frames = (0, 0)
            self.estimate_label.setText("")

    def _import_labeler(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "导入标注工具数据", "", "标注文件 (*.json);;所有文件 (*)")
        if not path:
            return
        try:
            from modules.vision.label_store import LabelStore, from_labeler_export
            imported = from_labeler_export(path)
        except Exception as exc:
            self._say(f"无法导入该数据：{exc}", THEME.danger)
            return
        if not imported:
            self._say("该导出文件中没有已标注的点。", THEME.warning)
            return

        target = self._store_path or os.path.splitext(path)[0] + ".examples.json"
        store = LabelStore(target).load()
        store.extend(imported)
        store.save()
        self._load_store(target)
        self._say(
            f"已导入 {len(imported)} 个示例。训练前需要先检查这些示例，"
            f"因为标注工具记录的是点，而不是目标的实际大小。", THEME.warning)

    # ── the estimate ─────────────────────────────────────────────────────

    def _refresh_estimate(self, *_args) -> None:
        if sum(self._frames) == 0:
            return
        if self._device is None:
            self.estimate_label.setText("正在估算训练所需时间…")
            self.estimate_label.setStyleSheet("color:#999;")
            if not self._probing:
                self._probing = True
                import threading

                def probe():
                    try:
                        device, name = _probe_training_device()
                    except Exception as exc:    # noqa: BLE001
                        print(f"[training] device probe failed: {exc}")
                        device, name = "cpu", ""
                    self._device_found.emit(device, name)

                threading.Thread(target=probe, daemon=True).start()
            return
        try:
            from training.train_estimate import (
                ThroughputStore, default_store_path, estimate, friendly_device,
                device_kind)
            from training.train_yolox_run import DEFAULT_IMAGE_SIZE, pretrained_path
            device, name = self._device
            size = self.size_combo.currentData()
            est = estimate(
                self._frames[0], self._frames[1], self.epochs_spin.value(), size,
                device, DEFAULT_IMAGE_SIZE,
                store=ThroughputStore(default_store_path()).load(),
                pretrained_cached=os.path.exists(pretrained_path(size)))
        except Exception as exc:                # noqa: BLE001
            print(f"[training] estimate failed: {exc}")
            self.estimate_label.setText("")
            return
        self._estimate_seconds = est.seconds
        text = est.sentence(friendly_device(device, name))
        if device_kind(device) == "cpu":
            text += (" 使用显卡通常可以快数倍；"
                     "选择更小的模型也会更快。")
        self.estimate_label.setText(text)
        self.estimate_label.setStyleSheet("" if est.seconds < 1800 else f"color:{THEME.warning};")

    @Slot(str, str)
    def _on_device_found(self, device: str, name: str) -> None:
        self._device = (device, name)
        self._probing = False
        self._refresh_estimate()

    # ── the live parts of a run ──────────────────────────────────────────

    def _show_stage(self, index: int) -> None:
        stages = ObjectTrainingWorker.STAGES
        parts = []
        for i, label in enumerate(stages):
            if i < index:
                parts.append(f"<span style='color:{THEME.success};'>✓ {label}</span>")
            elif i == index:
                parts.append(f"<b>▶ {label}</b>")
            else:
                parts.append(f"<span style='color:#777;'>{label}</span>")
        self.stage_label.setText("&nbsp;&nbsp;→&nbsp;&nbsp;".join(parts))

    def _update_clock(self) -> None:
        import time
        spent = time.monotonic() - self._started_at
        text = f"已用 {_elapsed(spent)}"
        if self._time_left > 0:
            text += f" · 预计剩余 {_friendly_time(self._time_left)}"
        elif self._estimate_seconds > 0:
            text += f" · 预计还需 {_friendly_time(max(0.0, self._estimate_seconds - spent)) or '片刻'}"
        self.clock_label.setText(text)

    @Slot(object)
    def _on_round(self, snap) -> None:
        self._rounds.append(snap)
        self.round_label.setText(snap.sentence())
        self.round_label.setVisible(True)
        if self._preview is not None:
            self._preview.add_round(snap)

    def _share(self) -> None:
        exported = self._last_export
        onnx_path = getattr(exported, "onnx_path", "") if exported is not None else ""
        if not onnx_path or not os.path.exists(onnx_path):
            self._say("训练后的 ONNX 模型文件已不存在，因此无法分享。请重新训练后再分享。", THEME.warning)
            return
        try:
            from model_hub.gui import PublishWizard
            from model_hub.package import draft_for_trained_detector
        except Exception as exc:                # noqa: BLE001
            self._say(f"分享功能不可用：{exc}", THEME.danger)
            return
        last = getattr(exported, "last_round", None)
        metrics = {
            "heldout_found": last[1] if last else None,
            "heldout_expected": last[2] if last else None,
            "rounds": len(self._rounds) or None,
            "train_frames": getattr(exported, "trained_on", None),
        }
        if self._rounds:
            metrics["false_alarms"] = self._rounds[-1].false_alarms
        draft = draft_for_trained_detector(onnx_path, metrics=metrics)
        PublishWizard(self, model_path=onnx_path, draft=draft).exec()

    def _open_preview(self) -> None:
        from modules.ui.training_preview import TrainingPreviewWindow
        if self._preview is None:
            self._preview = TrainingPreviewWindow(self)
            self._preview.closed.connect(self._on_preview_closed)
            for snap in self._rounds:           # rounds before it opened, as numbers
                self._preview.add_round(snap)
        if self._worker is not None:
            self._worker.draw_rounds = True
        self._preview.show()
        self._preview.raise_()
        self._preview.activateWindow()

    def _on_preview_closed(self) -> None:
        if self._worker is not None:
            self._worker.draw_rounds = False
        self._preview = None

    # ── the run ──────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread is not None:
            return

        # The dataset and checkpoints sit beside the user's own labels, not in
        # the install directory: they are working files about their footage,
        # they are large, and they belong wherever that footage is organised.
        work_dir = os.path.join(os.path.dirname(self._store_path), "training_run")
        # The model, by contrast, goes where the detector looks for it.
        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        dest_dir = os.path.join(repo_root, "models", "custom")

        self._worker = ObjectTrainingWorker(
            store_path=self._store_path,
            work_dir=work_dir,
            dest_dir=dest_dir,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            size=self.size_combo.currentData(),
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._show_stage)
        self._worker.round_done.connect(self._on_round)
        self._worker.draw_rounds = self._preview is not None
        self._rounds = []
        if self._preview is not None:
            self._preview.reset()
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._set_running(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._say("将在当前步骤完成后停止…", THEME.warning)

    def _set_running(self, running: bool) -> None:
        import time
        self.train_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress_bar.setVisible(running)
        self.import_btn.setEnabled(not running)
        self.stage_label.setVisible(running)
        self.clock_label.setVisible(running)
        self.estimate_label.setVisible(not running)
        self.watch_btn.setVisible(running or bool(self._rounds))
        if running:
            self.share_box.setVisible(False)
        if running:
            self.progress_bar.setValue(0)
            self.round_label.setText("")
            self.round_label.setVisible(False)
            self._started_at = time.monotonic()
            self._time_left = 0.0
            self._show_stage(0)
            self._update_clock()
            self._tick.start()
        else:
            self._tick.stop()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        # The loop's own estimate, once it has one, replaces the up-front guess.
        match = re.search(r"about (.+) left", message)
        self._time_left = _parse_friendly(match.group(1)) if match else self._time_left
        self._say(re.sub(r", about .+ left", "", message), "")

    @Slot(object)
    def _on_finished(self, exported) -> None:
        import time
        took = time.monotonic() - self._started_at
        self._teardown()
        trained = getattr(exported, "trained_on", 0)
        checked = getattr(exported, "checked_on", 0)
        names = ", ".join(exported.class_names)
        message = (f"模型已准备就绪。它从 {trained} 个训练帧中学习了 {names}")
        if checked:
            message += f"，并使用 {checked} 个未见过的帧进行了验证"
        last = getattr(exported, "last_round", None)
        if last and last[2]:
            message += (f"。在未参与训练的帧中，它识别出 {last[1]} / {last[2]} 个目标")
        message += f"。耗时 {_elapsed(took)}。"
        message += "\nIt is installed and will be used when you run a scan."
        self._say(message, THEME.success)
        self._last_export = exported
        self.share_box.setVisible(bool(getattr(exported, "onnx_path", "")))
        self._device = None                     # re-probe: speeds were just measured
        self._refresh_estimate()
        self.model_installed.emit(exported)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._teardown()
        self._say(message, THEME.danger)
        if not message.startswith("已停止"):
            QMessageBox.warning(self, "训练", message)

    def _teardown(self) -> None:
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    def _say(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{colour};" if colour else "")

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        """A training run must not outlive its window."""
        self._cancel()
        self._teardown()
        super().closeEvent(event)


class ActionTrainingWorker(QObject):
    """Drive the R3D trainer in a child process, reporting what it prints.

    A subprocess rather than an import, for one reason: ``model_training.r3d``
    is an existing, working training script with its own configuration, cache
    handling and ONNX export. Refactoring it to take progress callbacks would
    be the larger and riskier change, and it would be a change to code that is
    not broken. Reading its output costs a parser and leaves it alone.

    It also means cancelling is a terminated process rather than a cooperative
    flag, which for a run holding a large clip cache is the more reliable stop.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    # "Epoch 3/30" from the trainer's own progress bar, and the per-epoch
    # summary it prints afterwards. Everything else it says goes to the debug
    # log unchanged.
    _EPOCH = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    _VAL = re.compile(r"Val\s+Loss:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)")

    def __init__(self, data_path: str, epochs: int, batch_size: int,
                 pipeline: str, variant: str, device: str):
        super().__init__()
        self._data_path = data_path
        self._epochs = epochs
        self._batch_size = batch_size
        self._pipeline = pipeline
        self._variant = variant
        self._device = device
        self._process = None
        self._stop = False

    def cancel(self) -> None:
        self._stop = True
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:                       # pragma: no cover - defensive
                pass

    @Slot()
    def run(self) -> None:
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        command = self._command(sys.executable)
        try:
            self.progress.emit(
                0, "正在准备片段——第一次处理会比较慢…")
            creation = 0
            if sys.platform.startswith("win"):
                creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            # The trainers print emoji, and a Windows console here is cp1250:
            # without this the child dies on UnicodeEncodeError at its first
            # line of output, long before it touches the user's data. Inside
            # the app the same prints survive because debug_console tees them
            # through a UTF-8 stream; a subprocess gets the raw console.
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            self._process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=creation, env=env,
            )
            note = ""
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[actions] {line}")   # the debug log keeps everything
                    note = self._read(line) or note
            code = self._process.wait()

            if self._stop:
                self.error.emit("已停止。")
                return
            if code != 0:
                self.error.emit(
                    f"训练进程异常结束，退出代码 {code}。调试日志中保留了训练器的完整输出。")
                return
            self.progress.emit(100, "完成。")
            self.finished.emit(note)
        except Exception as exc:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error.emit(f"无法运行动作训练器：{exc}")

    def _command(self, python: str) -> list:
        """The trainer to run, and its flags.

        Two pipelines, because the hardware genuinely differs:

        ``intel`` runs Intel's action-recognition encoder under OpenVINO and
        trains only a decoder on top. The encoder is frozen, so its output is
        cached once and every later epoch is fast. This is the path that
        produced the classifier the app already ships.

        ``r3d`` fine-tunes a 3D CNN end to end. More capable and far heavier,
        and it wants a CUDA card.

        Their flags are not the same: the Intel trainer picks its own device
        and takes a decoder type, while the R3D one takes a device and a model
        variant. So this builds each command rather than sharing one.
        """
        common = [
            python, "-u", "-m", f"model_training.{self._pipeline}.train",
            "--data-path", self._data_path,
            "--epochs", str(self._epochs),
            "--batch-size", str(self._batch_size),
            "--no-viz",
        ]
        if self._pipeline == "intel":
            # No --device: intel/config.py already resolves XPU itself, and
            # the encoder half runs under OpenVINO regardless.
            return common
        return common + ["--model", self._variant, "--device", self._device]

    def _read(self, line: str):
        """Turn one line of the trainer's output into a progress update."""
        epoch = self._EPOCH.search(line)
        if epoch:
            done, total = int(epoch.group(1)), max(1, int(epoch.group(2)))
            self.progress.emit(int(100 * done / total),
                               f"训练中… 第 {done}/{total} 轮")
            return None
        val = self._VAL.search(line)
        if val:
            # Accuracy is the one number here worth showing: unlike a loss, a
            # person can read it without knowing the model.
            share = float(val.group(2)) * 100
            return f"在未见过的片段中识别正确率约为 {share:.0f}%"
        return None


class ActionTrainingSection(QWidget):
    """Train the app to recognise an action of the user's own.

    Actions are taught from **whole clips**, not boxes: one folder per action,
    videos inside it. That is why this is its own section rather than a mode of
    the object one - the example a user has to supply is a different kind of
    thing, and a shared form would ask for the wrong input.
    """

    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH = 4          # 3D clips are far heavier than stills
    # The trainer's own minimums: below these it skips the class.
    MIN_TRAIN_CLIPS = 5
    MIN_VAL_CLIPS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ActionTrainingWorker] = None
        self._data_path = ""
        self._device = "cpu"
        self._pipeline = "intel"
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        explain = QLabel(
            "让应用学会识别随时间发生的动作，"
            "而不是只识别单帧中可见的物体。\n"
            "每个动作准备一个文件夹，并在其中放入若干短视频片段。"
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color:#999;")
        root.addWidget(explain)

        row = QHBoxLayout()
        row.addWidget(QLabel("片段文件夹："))
        self.folder_label = QLabel("尚未选择")
        self.folder_label.setStyleSheet("font-style:italic;color:#999;")
        row.addWidget(self.folder_label, 1)
        browse = QPushButton("选择…")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        root.addLayout(row)

        self.classes_label = QLabel("")
        self.classes_label.setWordWrap(True)
        root.addWidget(self.classes_label)

        advanced = CollapsibleSection(
            "高级", settings_key="training/actions_advanced")
        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(self.DEFAULT_EPOCHS)
        form.addRow("训练轮数：", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(self.DEFAULT_BATCH)
        form.addRow("每批片段数：", self.batch_spin)

        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("自动（匹配当前硬件）", "auto")
        self.pipeline_combo.addItem("Intel - OpenVINO 编码器", "intel")
        self.pipeline_combo.addItem("NVIDIA - 3D CNN", "r3d")
        self.pipeline_combo.currentIndexChanged.connect(self._choose_pipeline)
        form.addRow("训练方式：", self.pipeline_combo)

        self.variant_combo = QComboBox()
        for variant, hint in (("r3d_18", "推荐"),
                              ("mc3_18", "更轻量"),
                              ("r2plus1d_18", "较慢，通常效果更好")):
            self.variant_combo.addItem(f"{variant} - {hint}", variant)
        form.addRow("3D CNN 模型：", self.variant_combo)
        # Held so the row can be hidden: it belongs to the 3D CNN only, and a
        # visible-but-irrelevant control reads as a setting that was ignored.
        self._advanced_form = form
        advanced.setContentLayout(form)
        root.addWidget(advanced)

        self.device_label = QLabel("")
        self.device_label.setWordWrap(True)
        root.addWidget(self.device_label)
        self._choose_pipeline()

        self.train_btn = QPushButton("训练动作模型")
        self.train_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.success};color:white;"
            f"font-weight:bold;padding:10px 18px;}}")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start)
        root.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("停止")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        root.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        root.addStretch()
        self.setLayout(root)

    def _choose_pipeline(self) -> None:
        """Decide which trainer to use, and say so before anything is started.

        **Hardware detection goes through `modules.system.device_utils`, not a torch
        probe.** That module is the app's single source of truth and it knows
        something a torch probe cannot: the released build ships a *CUDA* torch
        wheel, on which `torch.xpu` exists but reports `is_available()` False —
        so on a packaged app an Intel Arc looks like no GPU at all. Its own
        comment says so. `device_utils` falls through to asking OpenVINO, which
        still sees the card. Probing torch here reproduced exactly that bug:
        "No GPU found" on a machine with an A750 in it.

        **Intel is not a fallback.** ``model_training/intel`` is a purpose-built
        pipeline — Intel's action-recognition encoder run under OpenVINO with a
        decoder trained on top — and it produced the classifier this app ships.
        On an Intel machine it is the right answer, not a consolation for
        lacking CUDA. The end-to-end 3D CNN is the better tool only where there
        is an NVIDIA card to run it on.
        """
        backend, gpu_present = "CPU", False
        try:
            from modules.system.device_utils import detect_best_device
            info = detect_best_device(log_fn=lambda *a, **k: None)
            backend = str(getattr(info, "backend_name", "CPU"))
            gpu_present = bool(getattr(info, "gpu_available", False))
        except Exception as exc:                    # pragma: no cover - defensive
            print(f"[training] device detection failed: {exc}")

        has_cuda = backend.upper().startswith("CUDA")

        # What torch itself can train on. Separate from the question above,
        # because device_utils answers for the *inference* pipeline — where
        # Intel deliberately goes through OpenVINO and its `pytorch_device`
        # stays "cpu" — while training is the other case.
        self._device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch, "xpu", None) and torch.xpu.is_available():
                self._device = "xpu"
        except Exception:
            pass

        chosen = (self.pipeline_combo.currentData()
                  if hasattr(self, "pipeline_combo") else "auto")
        self._pipeline = ("r3d" if has_cuda else "intel") if chosen == "auto" else chosen

        colour = "#999"
        if self._pipeline == "intel":
            where = {"cuda": "你的 NVIDIA GPU", "xpu": "你的 Intel GPU"}.get(
                self._device, "处理器")
            note = (f"Intel 方式，使用 {backend}。编码器通过 OpenVINO 运行，"
                    f"仅训练解码器（{where}），因此第一次处理较慢，后续轮次会更快。")
            if not gpu_present:
                note += " 未检测到 GPU，第一次处理可能需要较长时间。"
                colour = THEME.warning
        else:
            where = {"cuda": "your NVIDIA GPU", "xpu": "your Intel GPU"}.get(
                self._device, "the processor")
            note = f"3D CNN：在{where}上训练所有层。"
            if self._device == "cpu":
                note += (" 在处理器上训练可能需要数小时；这种情况下通常更适合使用 Intel 方式。")
                colour = THEME.warning
        self.device_label.setText(note)
        self.device_label.setStyleSheet(f"color:{colour};")

        # The 3D CNN variants mean nothing to the Intel method, which trains a
        # decoder on a fixed encoder. Leaving the row on screen invites someone
        # to pick r2plus1d_18 and then wonder why nothing about the run changed.
        form = getattr(self, "_advanced_form", None)
        if form is not None:
            form.setRowVisible(self.variant_combo, self._pipeline == "r3d")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择片段文件夹")
        if path:
            self._load_folder(path)

    def _load_folder(self, path: str) -> None:
        """Check the folder is laid out the way the trainer reads it.

        The layout is ``<folder>/train/<action>/*.mp4`` and ``<folder>/val/...``
        — *not* a folder per action at the top level, which is the arrangement
        that looks natural and silently yields "No training samples found".
        The minimums are the trainer's own: below them it skips a class, so
        they are worth stating here rather than after a wasted run.
        """
        train_root = os.path.join(path, "train")
        val_root = os.path.join(path, "val")
        if not os.path.isdir(train_root):
            self._data_path = ""
            self.folder_label.setText(os.path.basename(path.rstrip(os.sep)) or path)
            self.folder_label.setStyleSheet("")
            self.classes_label.setText(
                "此文件夹中需要包含 train 文件夹，其中每个动作对应一个子文件夹；"
                "同时还需要相同结构的 val 文件夹。")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            return

        def count(root: str) -> dict:
            out = {}
            if not os.path.isdir(root):
                return out
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):
                    continue
                clips = [f for f in os.listdir(folder)
                         if f.lower().endswith((".mp4", ".avi", ".mov"))]
                if clips:
                    out[name] = len(clips)
            return out

        train_counts, val_counts = count(train_root), count(val_root)
        self._data_path = path
        self.folder_label.setText(os.path.basename(path.rstrip(os.sep)) or path)
        self.folder_label.setStyleSheet("")

        if len(train_counts) < 2:
            # One class cannot be learned: a classifier needs something to tell
            # its class apart from, and a single folder trains a model that
            # answers "yes" to everything it is ever shown.
            self.classes_label.setText(
                "至少需要两个动作，每个动作各有一个片段文件夹。只有一个类别时，模型无法学会区分。")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            return

        short = [name for name, n in train_counts.items()
                 if n < self.MIN_TRAIN_CLIPS or val_counts.get(name, 0) < self.MIN_VAL_CLIPS]
        described = ", ".join(
            f"{name} ({n} + {val_counts.get(name, 0)})" for name, n in train_counts.items())
        if short:
            self.classes_label.setText(
                f"{described}。以下动作因片段太少将被跳过：{', '.join(short)}。"
                f"每个动作至少需要 {self.MIN_TRAIN_CLIPS} 个训练片段和 "
                f"{self.MIN_VAL_CLIPS} 个验证片段。")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
        else:
            self.classes_label.setText(f"可以开始训练：{described}。")
            self.classes_label.setStyleSheet("")
        self.train_btn.setEnabled(len(train_counts) - len(short) >= 2)

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._worker = ActionTrainingWorker(
            data_path=self._data_path,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            pipeline=self._pipeline,
            variant=self.variant_combo.currentData(),
            device=self._device,
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._set_running(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._say("正在停止…", THEME.warning)

    def _set_running(self, running: bool) -> None:
        self.train_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress_bar.setVisible(running)
        if running:
            self.progress_bar.setValue(0)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self._say(message, "")

    @Slot(str)
    def _on_finished(self, note: str) -> None:
        self._teardown()
        message = "动作模型已准备就绪。"
        if note:
            message += f" {note}。"
        self._say(message, THEME.success)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._teardown()
        self._say(message, THEME.danger)

    def _teardown(self) -> None:
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    def _say(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{colour};" if colour else "")

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        """A training run must not outlive its window."""
        self._cancel()
        self._teardown()
        super().closeEvent(event)


class TrainingPanel(QWidget):
    """The two kinds of training, side by side.

    Separate tabs rather than one form, because the *example* differs: an
    object is a box in a frame, an action is a clip that runs over time. They
    take different data from disk, train different models with different
    scripts, and share nothing but the word "training" - so a combined form
    would only hide which inputs each one needs.
    """

    model_installed = Signal(object)

    def __init__(self, parent=None, store_path: str = ""):
        super().__init__(parent)
        from PySide6.QtWidgets import QTabWidget

        self.objects = ObjectTrainingSection(store_path=store_path)
        self.objects.model_installed.connect(self.model_installed)
        self.actions = ActionTrainingSection()

        tabs = QTabWidget()
        # First: the automated loop (modules/teach), cutting, sorting and
        # labelling by itself from example clips and videos. The two tabs after
        # it train from data that is already labelled.
        try:
            from modules.teach.teach_panel import TeachPanel
            self.teach = TeachPanel()
            tabs.addTab(self.teach, "从视频学习")
        except Exception as exc:                # pragma: no cover - never cost the rest
            self.teach = None
            print(f"[training] teach panel unavailable: {exc}")
        tabs.addTab(self.objects, "物体")
        tabs.addTab(self.actions, "动作")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hardware_label())
        layout.addWidget(tabs)
        self.setLayout(layout)

    @staticmethod
    def _hardware_label() -> QLabel:
        """Name the hardware, above both kinds of training.

        Shown so somebody can confirm a run is about to use the card they think
        it is, before committing hours to it. Above the tabs rather than inside
        them because it is the same machine either way, and a fact repeated in
        two places is a fact that can disagree with itself.
        """
        label = QLabel()
        label.setWordWrap(True)
        try:
            from modules.system.device_utils import describe_devices
            devices = describe_devices()
        except Exception as exc:                # pragma: no cover - defensive
            devices = []
            print(f"[training] could not list devices: {exc}")

        if devices:
            label.setText("训练 hardware: " + "; ".join(devices))
            label.setStyleSheet("color:#999;")
        else:
            label.setText(
                "训练 hardware: no GPU found - training will use the "
                "处理器训练，速度会明显较慢。")
            label.setStyleSheet(f"color:{THEME.warning};")
        return label
