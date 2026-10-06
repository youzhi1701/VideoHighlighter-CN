"""Write down what the app is being asked to draw on.

A window that dies when it is resized to fill a large, heavily scaled display
leaves nothing behind: the failure is in Qt or the driver, below the Python
frame, so `debug.log` ends mid-sentence and the traceback that would name the
cause never exists. What can be recorded is the *shape of the problem* — how
many screens, how big, at what device pixel ratio, with which scale factor, and
what size the window had reached when the log stopped.

That is what this module is for. It proves or refutes the obvious theory (a
4K panel at 200% asking for a surface twice the size anyone tested) without
anybody having to reproduce the crash on hardware they do not own.
"""

from __future__ import annotations

import os
import sys


def _screen_line(screen) -> str:
    geometry = screen.geometry()
    available = screen.availableGeometry()
    return (f"   {screen.name() or '屏幕'}: "
            f"{geometry.width()}x{geometry.height()}，位置 "
            f"({geometry.x()},{geometry.y()})，"
            f"可用区域 {available.width()}x{available.height()}，"
            f"像素比 {screen.devicePixelRatio():g}，"
            f"逻辑 DPI {screen.logicalDotsPerInch():.0f}，"
            f"物理 DPI {screen.physicalDotsPerInch():.0f}")


def describe(app) -> list:
    """Lines describing every screen, plus the scale factors in play."""
    lines = []
    try:
        screens = list(app.screens())
        primary = app.primaryScreen()
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        return [f"   （无法读取显示器信息：{type(e).__name__}：{e}）"]

    for screen in screens:
        try:
            mark = "（主显示器）" if screen is primary else ""
            lines.append(_screen_line(screen) + mark)
        except Exception as e:  # noqa: BLE001
            lines.append(f"   （无法读取此显示器详情：{e}）")

    factor = os.environ.get("QT_SCALE_FACTOR")
    rounding = os.environ.get("QT_SCALE_FACTOR_ROUNDING_POLICY")
    lines.append(f"   QT_SCALE_FACTOR={factor or '（未设置）'}, "
                 f"rounding={rounding or '（默认）'}, "
                 f"platform={sys.platform}")
    return lines


def log(app, log_fn=print) -> None:
    """Report the display setup once, at startup."""
    log_fn("🖥️ 显示器：")
    for line in describe(app):
        log_fn(line)


def log_window_size(window, label: str, log_fn=print) -> None:
    """Report a window's size, in the units that matter for a crash.

    Both numbers are here on purpose: Qt lays out in logical pixels, the driver
    allocates in physical ones, and a surface that is fine at 1920 wide may not
    be at 3840. A log that carries only one of them cannot tell those apart.
    """
    try:
        size = window.size()
        ratio = window.devicePixelRatioF()
        state = "最大化" if window.isMaximized() else (
            "全屏" if window.isFullScreen() else "窗口")
        log_fn(f"🖥️ {label}：逻辑尺寸 {size.width()}x{size.height()}，"
               f"物理尺寸 {int(size.width() * ratio)}x{int(size.height() * ratio)}，"
               f"状态：{state}")
    except Exception as e:  # noqa: BLE001 - diagnostics must not raise
        log_fn(f"🖥️ {label}：无法读取尺寸（{type(e).__name__}：{e}）")
