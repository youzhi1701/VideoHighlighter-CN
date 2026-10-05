"""Qt layer over pack_manager: ask, download with a progress dialog, restart.

All the logic lives in pack_manager; this only asks the user, marshals progress
off the worker thread, and offers the restart a PyTorch pack needs.

Call from the GUI thread, *before* starting the work that needs the pack:

    from modules.packs import pack_ui
    if not pack_ui.ensure_pack(self, pack_ui.CLIP_PACK,
                               why="视觉搜索需要 CLIP 模型。"):
        return
"""
from __future__ import annotations

import subprocess

from PySide6.QtCore import QEventLoop, Qt, QThread, Signal
from PySide6.QtWidgets import QApplication, QMessageBox, QProgressDialog

from modules.packs import pack_manager
from modules.packs.pack_manager import CLIP_PACK, CUDA_PACK, FRAME_ENCODER_PACK  # noqa: F401  (re-exported)

_TITLES = {
    CUDA_PACK: "NVIDIA GPU 加速",
    CLIP_PACK: "视觉搜索模型",
    FRAME_ENCODER_PACK: "动作模型编码器",
}


class PackInstallWorker(QThread):
    # object, not int: byte counts pass 2**31 and a Qt int signal would wrap.
    progress = Signal(str, object, object, str)   # phase, done, total, detail
    finished_with = Signal(object)                # pack_manager.PackResult

    def __init__(self, name, parent=None):
        super().__init__(parent)
        self.name = name
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            result = pack_manager.install_pack(
                self.name,
                progress=lambda *a: self.progress.emit(*a),
                should_cancel=lambda: self._cancel,
            )
        except Exception as e:  # install_pack does not raise; belt and braces
            print(f"pack_install: unexpected failure ({type(e).__name__}: {e})")
            result = pack_manager.PackResult(False, f"The download failed: {e}", self.name)
        self.finished_with.emit(result)


def _mb(n) -> str:
    return f"{int(n) / 2**20:,.0f} MB"


def ensure_pack(parent, name: str, *, why: str = "") -> bool:
    """True when ``name`` is usable now. Otherwise offers to download it, and
    returns True only if it was installed and needs no restart."""
    lock = pack_manager.load_lock()
    pack = lock.get(name)
    title = _TITLES.get(name, name)
    if pack is None:
        # Running from source, or a build without packs: nothing to offer.
        return False
    state = pack_manager.status(pack)
    if state == pack_manager.INSTALLED:
        return True
    if state == pack_manager.PENDING:
        _offer_restart(parent, title)
        return False

    if name == CUDA_PACK:
        advice = pack_manager.cuda_pack_advice(pack_manager.probe_nvidia())
        if advice:
            QMessageBox.information(parent, title, advice)
            return False

    reason = pack_manager.incompatibility(pack)
    if reason:
        QMessageBox.warning(parent, title, reason)
        return False

    after = ("VideoHighlighter 会重启一次以切换到该组件。"
             if not pack.is_model else "安装完成后立即生效。")
    text = (f"{why}\n\n" if why else "") + (
        f"现在下载吗？下载量 {_mb(pack.bytes)}，安装后占用 {_mb(pack.bytes_installed)} "
        f"磁盘空间。只需下载一次，软件更新时会保留。{after}")
    if QMessageBox.question(parent, title, text,
                            QMessageBox.Yes | QMessageBox.No,
                            QMessageBox.Yes) != QMessageBox.Yes:
        return False

    result = _run_with_progress(parent, name, title)
    if result.cancelled:
        return False
    if not result.ok:
        QMessageBox.warning(parent, title, result.message)
        return False
    if result.restart_required:
        _offer_restart(parent, title)
        return False
    return True


def _run_with_progress(parent, name, title):
    dialog = QProgressDialog(f"正在下载 {title}…", "暂停", 0, 1000, parent)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(0)
    dialog.setAutoReset(False)
    dialog.setAutoClose(False)
    dialog.setValue(0)

    worker = PackInstallWorker(name, parent)
    loop = QEventLoop()
    holder = {}

    def on_progress(phase, done, total, detail):
        if phase == pack_manager.DOWNLOADING and total:
            dialog.setRange(0, 1000)
            dialog.setValue(min(1000, int(done * 1000 / total)))
            dialog.setLabelText(f"正在下载 {title}…\n{_mb(done)} / {_mb(total)}")
        elif phase == pack_manager.VERIFYING:
            dialog.setLabelText("正在检查下载文件…")
        elif phase == pack_manager.INSTALLING:
            dialog.setRange(0, 0)          # busy: 7-Zip reports no progress
            dialog.setCancelButton(None)   # unpacking is not interruptible
            dialog.setLabelText(f"正在解压 {title}… 可能需要几分钟。")

    def on_done(result):
        holder["result"] = result
        loop.quit()

    worker.progress.connect(on_progress)
    worker.finished_with.connect(on_done)
    dialog.canceled.connect(worker.cancel)
    worker.start()
    loop.exec()
    worker.wait()
    dialog.close()
    return holder["result"]


def _offer_restart(parent, title):
    answer = QMessageBox.question(
        parent, title,
        f"{title} 已安装，重启后生效。现在重启吗？",
        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if answer != QMessageBox.Yes:
        return
    from modules.update import update_apply
    try:
        subprocess.Popen(update_apply.relaunch_command(),
                         cwd=update_apply.install_root(), close_fds=True)
    except Exception as e:
        print(f"pack_install: could not relaunch ({e})")
        return
    QApplication.quit()
