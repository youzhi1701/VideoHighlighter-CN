"""
llm_chat_widget.py - Embeddable Qt chat panel for VideoHighlighter.

Key feature: AUTO-LOADS the most recent cache file from ./cache/ on startup,
so the LLM always has video context even after restarting the app.

Now with VideoSeekAnalyzer integration for visual search and seeking capabilities.

Usage:
    chat = LLMChatWidget(parent=self, cache_dir="./cache")
    layout.addWidget(chat)
"""

from __future__ import annotations

import os
import json
import time as _time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QLineEdit, QComboBox,
    QFileDialog, QApplication, QCheckBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSettings, QObject
from PySide6.QtGui import QTextCursor

from modules.ui.collapsible import CollapsibleSection
from modules.ui.fit import fit_width
from modules.ui.theme import DARK as THEME
from modules.ui import icons as ui_icons

from .llm_module import (
    LLMModule, VideoContextBuilder, get_available_backends, get_ollama_models,
    VideoSeekAnalyzer, CancellationToken, GenerationCancelled,
)
from .llm_reasoning import ReasoningLLMIntegration
from .ollama_host import (
    DEFAULT_BASE_URL as OLLAMA_DEFAULT_URL, is_remote as ollama_is_remote,
    remember as remember_ollama_host, resolve as resolve_ollama_host,
)

# timeline bridge (only available when timeline viewer is present)
try:
    from .llm_timeline_bridge import TimelineBridge
    HAS_TIMELINE_BRIDGE = True
except ImportError:
    HAS_TIMELINE_BRIDGE = False


# ---------------------------------------------------------------------------
# Worker thread for LLM queries
# ---------------------------------------------------------------------------
class _LLMWorker(QObject):
    """Runs LLM.query() off the GUI thread with token-by-token streaming.

    Uses the QObject + moveToThread() pattern. CancellationToken supports
    mid-generation cancellation; blocking image-decode calls still have to
    finish, but the worker emits `finished` cleanly via the event loop and
    the `finished → deleteLater` chain handles teardown.
    """

    token_received = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, llm: LLMModule, message: str,
                 analysis_data: dict | None = None,
                 video_path: str = "",
                 timeline_context: str = "",
                 frame_base64: str | None = None,
                 free_chat_mode: bool = False):
        super().__init__()
        self.llm = llm
        self.message = message
        self.analysis_data = analysis_data
        self.video_path = video_path
        self.timeline_context = timeline_context
        self.frame_base64 = frame_base64
        self.free_chat_mode = free_chat_mode
        self._cancel_token = CancellationToken()

    def cancel(self):
        """Request cancellation — works even for blocking GGUF vision calls."""
        self._cancel_token.cancel()

    @Slot()
    def run(self):
        try:
            def _stream_callback(token: str):
                if self._cancel_token.is_cancelled:
                    raise GenerationCancelled()
                self.token_received.emit(token)

            full_response = self.llm.query(
                user_message=self.message,
                analysis_data=self.analysis_data,
                video_path=self.video_path,
                timeline_context=self.timeline_context,
                frame_base64=self.frame_base64,
                stream_callback=_stream_callback,
                cancellation_token=self._cancel_token,
                free_chat_mode=self.free_chat_mode,
            )
            if self._cancel_token.is_cancelled:
                self.finished.emit(full_response + "\n[已停止]")
            else:
                self.finished.emit(full_response)
        except GenerationCancelled:
            self.finished.emit("[用户已停止]")
        except Exception as e:
            if self._cancel_token.is_cancelled:
                self.finished.emit("[用户已停止]")
            else:
                self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Worker thread for visual search
# ---------------------------------------------------------------------------
class _VisualSearchWorker(QObject):
    """Runs VideoSeekAnalyzer visual search in background.

    Uses the QObject + moveToThread() pattern. CancellationToken interrupts
    per-frame LLM calls; the image decode itself is a ~20s blocking C call,
    so cancellation takes effect at the next frame boundary.
    """

    progress = Signal(int, int, float, str)  # current, total, timestamp, preview
    frame_analyzed = Signal(float, str, str, bool, float)  # timestamp, timestamp_str, response, contains_target, score(0-1)
    found = Signal(float, str, str)  # timestamp, timestamp_str, analysis
    finished = Signal(list)  # all results
    error = Signal(str)

    def __init__(self, analyzer: VideoSeekAnalyzer, target: str,
                 interval: float = 1.0, max_seeks: int = 100,
                 stop_on_first_match: bool = True,
                 start_time: float = 0.0,
                 scene_threshold: float = 8.0,
                 mode: str = "llm", top_k: int = 30, clip_device: str = "AUTO"):
        super().__init__()
        self.analyzer = analyzer
        self.target = target
        self.interval = interval
        self.max_seeks = max_seeks
        self.stop_on_first_match = stop_on_first_match
        self.start_time = start_time
        # Engine: "llm" (VLM brute force), "clip" (fast GPU ranker only), or
        # "clip_llm" (CLIP ranks all frames on the GPU, VLM confirms top_k).
        self.mode = mode
        self.top_k = top_k
        self.clip_device = clip_device
        self._clip = None  # lazily created ClipEmbedder
        self._clip_labels = None      # embedded [positive, *negatives] for this query
        self._clip_memo = None        # ClipFrameIndex: frame embeddings, reused across scans
        self._clip_memo_path = None
        self._clip_memo_added = 0     # frames encoded this scan (vs served from memo)
        # Content-aware sampling: skip a frame when its mean per-pixel change from
        # the last *analyzed* frame is below this (0-255 scale). 0 disables it.
        # Higher = skip more aggressively. This only avoids redundant ~3s vision
        # encodes on visually-static stretches; it never alters what the model is
        # asked, and the downscale used for the metric never reaches the model.
        self.scene_threshold = scene_threshold
        self._prev_small = None  # downscaled grayscale of the last analyzed frame
        self._cancel_token = CancellationToken()

    def cancel(self):
        self._cancel_token.cancel()

    @Slot()
    def run(self):
        """Dispatch to the selected engine. Each path emits finished/error."""
        try:
            if self.mode == "clip":
                self._run_clip()
            elif self.mode == "clip_llm":
                self._run_clip_llm()
            else:
                self._run_llm()
        except GenerationCancelled:
            self.finished.emit([])
        except Exception as e:
            import traceback
            traceback.print_exc()
            # The windowed exe has no console, so the printed traceback is
            # invisible — put the origin (deepest frames) in the chat message.
            frames = traceback.extract_tb(e.__traceback__)[-3:]
            where = " < ".join(
                f"{os.path.basename(f.filename)}:{f.lineno}" for f in reversed(frames)
            )
            self.error.emit(f"{e} [位置：{where}]")

    # ------------------------------------------------------------------ shared
    def _vlm_analyze(self, timestamp, frame, stage_totals, scene_diff=-1.0):
        """Two-phase VLM presence check on one decoded frame. Emits
        frame_analyzed and returns the result dict (or None if cancelled).
        Frames go to the model at full resolution — downscaling wrecks accuracy.
        """
        import time
        frame_t0 = time.perf_counter()
        try:
            h, w = frame.shape[:2]
            frame_dims = f"{w}x{h}"
        except Exception:
            frame_dims = "?"

        t0 = time.perf_counter()
        frame_b64 = self.analyzer.frame_to_base64(frame)
        t_encode = time.perf_counter() - t0
        stage_totals['encode'].append(t_encode)
        b64_kb = len(frame_b64) / 1024
        if self._cancel_token.is_cancelled:
            return None

        # one-word YES/NO presence check (cheap)
        t0 = time.perf_counter()
        response = self.analyzer.llm.query(
            user_message=f"Does this frame contain a {self.target}? Answer with only one word: YES or NO.",
            frame_base64=frame_b64,
            system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
            temperature=0.0, max_tokens=5, cancellation_token=self._cancel_token,
        )
        t_llm = time.perf_counter() - t0
        stage_totals['llm'].append(t_llm)
        starts_with_yes = response.strip().lower().startswith("yes")

        # describe only on a positive hit
        analysis_text = response
        t_confirm = 0.0
        if starts_with_yes:
            t0 = time.perf_counter()
            analysis_text = self.analyzer.llm.query(
                user_message=f"This frame contains a {self.target}. Briefly describe what you see.",
                frame_base64=frame_b64,
                system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
                temperature=0.0, max_tokens=150, cancellation_token=self._cancel_token,
            )
            t_confirm = time.perf_counter() - t0
            stage_totals['confirm'].append(t_confirm)

        result = {
            "timestamp": timestamp,
            "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
            "analysis": analysis_text,
            "contains_target": starts_with_yes,
        }
        # The VLM answers yes/no, so there is no graded score — 1.0 is honest here.
        self.frame_analyzed.emit(timestamp, result["timestamp_str"],
                                 analysis_text, starts_with_yes, 1.0)
        t_total = time.perf_counter() - frame_t0
        print(f"[t={timestamp:6.1f}s] {frame_dims} {b64_kb:5.0f}KB Δ={scene_diff:5.1f} | "
              f"encode={t_encode*1000:5.0f}ms  llm={t_llm*1000:6.0f}ms  "
              f"confirm={t_confirm*1000:6.0f}ms  TOTAL={t_total*1000:6.0f}ms "
              f"({'YES' if starts_with_yes else 'no '})")
        return result

    def _print_summary(self, stage_totals, n_skipped):
        if not stage_totals.get('llm'):
            return
        print("\n" + "=" * 70)
        print("性能统计汇总")
        print("=" * 70)
        print(f"{'阶段':<10} {'次数':>5} {'平均':>8} {'中位数':>8} {'最小':>8} {'最大':>8} {'合计':>8}")
        for stage in ('encode', 'llm', 'confirm'):
            vals = stage_totals.get(stage) or []
            if not vals:
                continue
            vs = sorted(vals)
            stage_display = {"encode": "编码", "llm": "大模型", "confirm": "确认"}.get(stage, stage)
            print(f"{stage_display:<10} {len(vals):>5} {sum(vals)/len(vals)*1000:>7.0f}ms "
                  f"{vs[len(vs)//2]*1000:>7.0f}ms {vs[0]*1000:>7.0f}ms {vs[-1]*1000:>7.0f}ms "
                  f"{sum(vals):>7.1f}s")
        print(f"\n已分析画面：{len(stage_totals['llm'])}   因场景稳定跳过：{n_skipped}")
        print("=" * 70 + "\n")

    # ----------------------------------------------------------------- engines
    def _run_llm(self):
        """LLM-only: walk every interval frame and ask the VLM (brute force)."""
        import time
        import numpy as np
        from collections import defaultdict

        stage_totals = defaultdict(list)
        n_skipped = 0
        results = []
        num = min(int(self.analyzer.duration / self.interval) + 1, self.max_seeks)
        timestamps = [self.start_time + i * self.interval for i in range(num)]

        for i, timestamp in enumerate(timestamps):
            if self._cancel_token.is_cancelled or timestamp > self.analyzer.duration + 0.1:
                break
            self.progress.emit(i + 1, len(timestamps), timestamp, f"正在分析 {timestamp:.1f} 秒")
            self.analyzer.current_time = timestamp
            frame = self.analyzer.seek_to_time(timestamp)
            if frame is None:
                print(f"[t={timestamp:6.1f}s] 定位失败")
                continue

            # content-aware skip (cheap downscaled diff; never sent to the model)
            scene_diff = -1.0
            if self.scene_threshold > 0:
                small = frame[::8, ::8]
                small = small.mean(axis=2).astype('float32') if small.ndim == 3 else small.astype('float32')
                if self._prev_small is not None and small.shape == self._prev_small.shape:
                    scene_diff = float(np.abs(small - self._prev_small).mean())
                    if scene_diff < self.scene_threshold:
                        n_skipped += 1
                        print(f"[t={timestamp:6.1f}s] 已跳过（场景变化 Δ={scene_diff:4.1f} < {self.scene_threshold}）")
                        continue
                self._prev_small = small

            if self._cancel_token.is_cancelled:
                break
            try:
                result = self._vlm_analyze(timestamp, frame, stage_totals, scene_diff)
            except GenerationCancelled:
                break
            except Exception as e:
                print(f"{timestamp:.1f} 秒处出错：{e}")
                continue
            if result is None:
                break
            results.append(result)
            if result["contains_target"]:
                self.found.emit(timestamp, result["timestamp_str"], result["analysis"])
                if self.stop_on_first_match:
                    break

        self._print_summary(stage_totals, n_skipped)
        self.finished.emit(results)

    def _clip_scan(self):
        """Score every interval frame with CLIP on the GPU. Returns a list of
        (timestamp, score) sorted by score desc, or None if CLIP is unavailable."""
        import time
        try:
            from .clip_prefilter import ClipFramePrefilter
            from .clip_index import ClipEmbedder
        except Exception as e:
            self.error.emit(f"CLIP 预筛选模块导入失败：{e}")
            return None
        reason = ClipFramePrefilter.import_error(self.clip_device)
        if reason is not None:
            self.error.emit(f"CLIP 不可用——{reason}")
            return None

        if self._clip is None:
            self.progress.emit(0, 1, self.start_time,
                               "正在 GPU 上加载 CLIP（首次运行需要下载模型）…")
            self._clip = ClipEmbedder(device=self.clip_device)
            self._clip.load()
        self._clip.set_query(self.target)

        # Embed the query once, then score frames by dot product against it.
        # The frames themselves are memoised across scans (see _clip_memo), so a
        # reworded search re-encodes nothing it has already seen.
        self._clip_labels = self._clip.embed_query_labels()
        self._open_clip_memo()

        timestamps = self._clip_timestamps()

        scored = []
        BATCH = 16
        t0 = time.perf_counter()

        def grab(missing):
            # Seeks are per-frame, so only the memo's misses cost anything.
            pairs = []
            for ts in missing:
                if self._cancel_token.is_cancelled:
                    break
                self.analyzer.current_time = ts
                frame = self.analyzer.seek_to_time(ts)
                if frame is not None:
                    pairs.append((ts, frame))
            return pairs

        for c in range(0, len(timestamps), BATCH):
            if self._cancel_token.is_cancelled:
                break
            chunk = timestamps[c:c + BATCH]
            batch, _ = self._embed_and_score(chunk, grab)
            for ts in chunk:
                if ts in batch:
                    scored.append((ts, batch[ts]))
            self.progress.emit(min(c + BATCH, len(timestamps)), len(timestamps),
                               chunk[-1], f"CLIP 扫描到 {chunk[-1]:.0f} 秒")

        elapsed = time.perf_counter() - t0
        encoded = self._clip_memo_added
        print(f"\nCLIP 扫描：{len(scored)} 帧，用时 {elapsed:.1f} 秒 "
              f"（{elapsed/max(1,len(scored))*1000:.1f} 毫秒/帧），设备 {self._clip.device}，"
              f"缓存命中 {len(scored) - encoded} 帧，新编码 {encoded} 帧")
        self._save_clip_memo()
        scored.sort(key=lambda x: -x[1])
        return scored

    # -- embedding memo ---------------------------------------------------

    def _clip_timestamps(self):
        """Sample on a fixed lattice: multiples of `interval` measured from zero,
        starting at the first one at/after start_time.

        NOT `start_time + i*interval`. start_time follows wherever the user last
        clicked the timeline, so that grid shifts by a fractional offset every
        search — 12.37, 12.87, ... shares no timestamp with 0.0, 0.5, ... The
        memo is keyed by timestamp, so it would hit nothing and re-encode the
        whole video every time. A lattice also means a coarser interval reuses a
        finer interval's frames, since its timestamps are a subset.
        """
        import math

        step = self.interval
        if step <= 0:
            return []
        first = int(math.ceil((self.start_time - 1e-6) / step))
        limit = self.analyzer.duration + 0.1
        out = []
        for i in range(first, first + self.max_seeks):
            # Round at the source so keys are exact, not float-drifted.
            ts = round(i * step, 3)
            if ts > limit:
                break
            out.append(ts)
        return out

    def _open_clip_memo(self):
        """Load this video's embedding memo (or start an empty one).

        Keyed per timestamp, not per scan: a cancelled search still banks what
        it encoded, and the next query pays only for gaps.
        """
        from .clip_index import open_memo

        self._clip_memo_added = 0     # per scan, not cumulative
        if self._clip_memo is not None:
            return
        try:
            self._clip_memo, self._clip_memo_path = open_memo(
                self.analyzer.video_path, self._clip.model_id,
            )
        except Exception as e:
            # A memo is an optimisation; never let it break search.
            print(f"⚠️ CLIP 记忆缓存不可用（{e}）；将不使用缓存继续扫描")
            self._clip_memo, self._clip_memo_path = None, None

    def _save_clip_memo(self):
        # `is None`, not truthiness: ClipFrameIndex has __len__, so an empty
        # memo is falsy even though it's a perfectly good object.
        if (self._clip_memo is None or self._clip_memo_path is None
                or not self._clip_memo_added):
            return
        try:
            self._clip_memo.save(self._clip_memo_path)
            print(f"🧠 CLIP 记忆缓存：已保存 {len(self._clip_memo)} 帧")
        except Exception as e:
            print(f"⚠️ CLIP 记忆缓存保存失败（{e}）")

    def _embed_and_score(self, timestamps, grab):
        """Score `timestamps`, encoding only the ones the memo doesn't hold.

        `grab(missing)` returns [(ts, frame_bgr)] for the timestamps it could
        decode, or None if its extraction path is unusable. Returns
        (scores_by_ts, grab_failed).
        """
        from .clip_index import score_embeddings

        memo = self._clip_memo
        if memo is None:      # memo unavailable: encode everything, cache nothing
            pairs = grab(list(timestamps))
            if pairs is None:
                return {}, True
            if not pairs:
                return {}, False
            embs = self._clip.embed_frames_bgr([f for _, f in pairs])
            scores = score_embeddings(embs, self._clip_labels, self._clip.logit_scale)
            return {ts: float(s) for (ts, _), s in zip(pairs, scores)}, False

        _, _, missing = memo.lookup(timestamps)
        if missing:
            pairs = grab(missing)
            if pairs is None:
                return {}, True
            if pairs:
                embs = self._clip.embed_frames_bgr([f for _, f in pairs])
                self._clip_memo_added += memo.extend([ts for ts, _ in pairs], embs)

        rows, positions, _ = memo.lookup(timestamps)
        if not rows:
            return {}, False
        scores = score_embeddings(memo.embeddings[rows], self._clip_labels,
                                  self._clip.logit_scale)
        return {timestamps[p]: float(s) for p, s in zip(positions, scores)}, False

    def _run_clip(self):
        """CLIP-only: return the top_k frames ranked by similarity (no VLM)."""
        scored = self._clip_scan()
        if scored is None:
            return
        top = scored[:self.top_k]
        print(f"CLIP: top {len(top)} of {len(scored)} frames "
              f"(scores {top[0][1]:.2f}..{top[-1][1]:.2f})" if top else "CLIP: no frames")
        results = []
        for ts, score in sorted(top, key=lambda x: x[0]):  # chronological for playback
            tstr = f"{int(ts)//60}:{int(ts)%60:02d}"
            analysis = f"CLIP 匹配（相似度 {score:.2f}）"
            results.append({"timestamp": ts, "timestamp_str": tstr,
                            "analysis": analysis, "contains_target": True,
                            "clip_score": score})
            self.frame_analyzed.emit(ts, tstr, analysis, True, float(score))
            self.found.emit(ts, tstr, analysis)
        self.finished.emit(results)

    def _run_clip_llm(self):
        """CLIP ranks all frames on the GPU; the VLM confirms only the top_k."""
        from collections import defaultdict
        scored = self._clip_scan()
        if scored is None:
            return
        candidates = scored[:self.top_k]
        print(f"CLIP+LLM：正在使用视觉模型确认排名前 {len(candidates)} 个候选画面")
        # Mark the handoff in the UI too. Without it the funnel is invisible:
        # the scan's progress and the confirms' progress read identically, and
        # a Top-K at or above the number of sampled frames (short video, coarse
        # interval) silently degrades to "the model sees everything" — which is
        # the same run, but worth being able to tell apart.
        self.progress.emit(0, max(1, len(candidates)),
                           float(candidates[0][0]) if candidates else 0.0,
                           f"CLIP 已排序 {len(scored)} 帧 → 视觉模型确认前 {len(candidates)} 帧")

        stage_totals = defaultdict(list)
        results = []
        for rank, (ts, score) in enumerate(candidates):  # strongest first
            if self._cancel_token.is_cancelled:
                break
            self.progress.emit(rank + 1, len(candidates), ts,
                               f"视觉模型确认 {ts:.0f} 秒（CLIP {score:.2f}）")
            self.analyzer.current_time = ts
            frame = self.analyzer.seek_to_time(ts)
            if frame is None:
                continue
            try:
                result = self._vlm_analyze(ts, frame, stage_totals)
            except GenerationCancelled:
                break
            except Exception as e:
                print(f"{ts:.1f} 秒处出错：{e}")
                continue
            if result is None:
                break
            result["clip_score"] = score
            results.append(result)
            if result["contains_target"]:
                self.found.emit(ts, result["timestamp_str"], result["analysis"])
                if self.stop_on_first_match:
                    break

        self._print_summary(stage_totals, 0)
        self.finished.emit(results)

# Keywords that indicate the user wants visual/frame analysis
_VISION_KEYWORDS = (
    "see", "look", "show me", "what is this", "describe frame",
    "describe image", "what's in the frame", "what do you see",
    "visual", "screenshot", "current frame", "this frame",
    "what's happening here", "what is happening here",
)

# ---------------------------------------------------------------------------
# Time spinbox — displays seconds as mm:ss, accepts both formats as input
# ---------------------------------------------------------------------------
class TimeSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox that stores a float seconds value but displays it as mm:ss.
    Accepts input in either form:  '90'  or  '1:30'  or  '1:30.5'
    """

    def textFromValue(self, value: float) -> str:
        total = max(0.0, value)
        minutes = int(total // 60)
        secs = total - minutes * 60
        if abs(secs - round(secs)) < 0.01:
            return f"{minutes}:{int(round(secs)):02d}"
        return f"{minutes}:{secs:05.2f}"  # e.g. 1:05.50

    def valueFromText(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0
        if ':' in text:
            parts = text.split(':')
            try:
                minutes = int(parts[0]) if parts[0] else 0
                seconds = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                return minutes * 60.0 + seconds
            except ValueError:
                return self.value()
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        from PySide6.QtGui import QValidator
        import re
        stripped = text.strip()
        if not stripped:
            return (QValidator.Intermediate, text, pos)
        if ':' in stripped:
            if re.fullmatch(r'\d{0,3}:\d{0,2}(\.\d*)?', stripped):
                return (QValidator.Acceptable, text, pos)
            if re.fullmatch(r'\d{0,3}:', stripped):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)
        return super().validate(text, pos)

class LLMChatWidget(QWidget):
    """
    Self-contained chat panel. Auto-loads latest cache from disk on startup.
    Now with VideoSeekAnalyzer integration for visual search and seeking.
    """
    MAX_RECENT_GGUF = 5
    SETTINGS_KEY = "VideoHighlighter/LLMChat"

    llm_replied = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False,
                 cache_dir: str = "./cache", video_path: str = ""):
        super().__init__(parent)
        self._llm: Optional[LLMModule] = None
        self._analyzer: Optional[VideoSeekAnalyzer] = None
        self._analysis_data: Optional[dict] = None
        self._video_path: str = video_path
        self._chat_history: list[dict] = []
        self._worker: Optional[_LLMWorker] = None
        self._worker_thread: Optional[QThread] = None
        self._search_worker: Optional[_VisualSearchWorker] = None
        self._search_worker_thread: Optional[QThread] = None
        self._compact = compact
        self._cache_dir = cache_dir
        self._timeline_bridge: Optional['TimelineBridge'] = None
        self._preview_window = None
        self.reasoning_engine = None
        self.reasoning_enabled = True
        # Found visual-search matches, for ◀ ▶ navigation
        self._search_results: list[dict] = []
        self._search_result_idx = -1
        if HAS_TIMELINE_BRIDGE:
            self._timeline_bridge = TimelineBridge()

        self._build_ui()
        self._auto_load_latest_cache()
        
        # Initialize analyzer if video path provided
        if self._video_path and os.path.exists(self._video_path):
            self._init_analyzer()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)

        # --- Settings section (foldable: once connected it's set-and-forget,
        # so folding it gives its height back to the chat/search below) ---
        settings_group = CollapsibleSection("大模型设置", settings_key="llm/settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(4)

        # Row 1: backend + model + connect + status (one row to save height)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("运行后端："))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Ollama（本地服务）", "ollama")
        self.backend_combo.addItem("llama-cpp（GGUF 文件）", "llama-cpp")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row1.addWidget(self.backend_combo)

        row1.addWidget(QLabel("模型："))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(180)
        row1.addWidget(self.model_combo)

        self.refresh_btn = QPushButton("刷新")
        fit_width(self.refresh_btn)
        self.refresh_btn.clicked.connect(self._refresh_models)
        row1.addWidget(self.refresh_btn)

        # Connect + status share this row (was a separate row) to save vertical space
        self.connect_btn = QPushButton("连接")
        self.connect_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.success};color:white;font-weight:bold;padding:6px 16px;}}"
        )
        self.connect_btn.clicked.connect(self._connect_llm)
        row1.addWidget(self.connect_btn)

        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color:#999;font-style:italic;")
        row1.addWidget(self.status_label)
        row1.addStretch()
        settings_layout.addLayout(row1)

        # Row 1b: which Ollama server. Hidden for llama-cpp, which has no
        # server to point anywhere. It sits next to Backend rather than in a
        # preferences dialog because this is the row where "连接到哪里"
        # is already being answered, and because a run that fails with "model
        # not found" needs the machine that was asked to be on screen.
        self.ollama_row_widget = QWidget()
        ollama_inner = QHBoxLayout()
        ollama_inner.setContentsMargins(0, 0, 0, 0)
        ollama_inner.addWidget(QLabel("Ollama 地址："))
        self.ollama_host_input = QLineEdit()
        self.ollama_host_input.setPlaceholderText(OLLAMA_DEFAULT_URL)
        self.ollama_host_input.setToolTip(
            "Ollama 服务地址。留空时使用 " + OLLAMA_DEFAULT_URL + "。\n"
            "也可以连接局域网中的其他电脑，例如 192.168.1.50 或\n"
            "http://box.lan:11434。远程 Ollama 需要以 OLLAMA_HOST=0.0.0.0 启动，\n"
            "才能被其他设备访问。聊天、报告、旁白和顾问功能都会使用这里的地址。"
        )
        # editingFinished, not textChanged: normalising while somebody is still
        # typing an address rewrites the field under the cursor.
        self.ollama_host_input.editingFinished.connect(self._on_ollama_host_edited)
        ollama_inner.addWidget(self.ollama_host_input)
        self.ollama_row_widget.setLayout(ollama_inner)
        settings_layout.addWidget(self.ollama_row_widget)

        # Row 2: GGUF path (hidden by default)
        self.gguf_row_widget = QWidget()
        gguf_inner = QHBoxLayout()
        gguf_inner.setContentsMargins(0, 0, 0, 0)
        gguf_inner.addWidget(QLabel("GGUF 路径："))
        self.gguf_path_input = QLineEdit()
        self.gguf_path_input.setPlaceholderText("请选择 model.gguf")
        gguf_inner.addWidget(self.gguf_path_input)
        self.gguf_browse_btn = QPushButton("浏览…")
        self.gguf_browse_btn.clicked.connect(self._browse_gguf)
        gguf_inner.addWidget(self.gguf_browse_btn)
        self.gguf_row_widget.setLayout(gguf_inner)
        self.gguf_row_widget.setVisible(False)
        settings_layout.addWidget(self.gguf_row_widget)

        # Row 3: mmproj path (for vision models, hidden by default)
        self.mmproj_row_widget = QWidget()
        mmproj_inner = QHBoxLayout()
        mmproj_inner.setContentsMargins(0, 0, 0, 0)
        mmproj_inner.addWidget(QLabel("mmproj 路径："))
        self.mmproj_path_input = QLineEdit()
        self.mmproj_path_input.setPlaceholderText("请选择 mmproj-model.gguf（可选，用于视觉模型）")
        mmproj_inner.addWidget(self.mmproj_path_input)
        self.mmproj_browse_btn = QPushButton("浏览…")
        self.mmproj_browse_btn.clicked.connect(self._browse_mmproj)
        mmproj_inner.addWidget(self.mmproj_browse_btn)
        self.mmproj_row_widget.setLayout(mmproj_inner)
        self.mmproj_row_widget.setVisible(False)
        settings_layout.addWidget(self.mmproj_row_widget)
        # Restore last-used GGUF paths
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        self.gguf_path_input.setText(settings.value("last_gguf_path", ""))
        self.mmproj_path_input.setText(settings.value("last_mmproj_path", ""))
        # Shows the server that would actually be used, including one inherited
        # from OLLAMA_HOST in the environment - a field that reads "localhost"
        # while the run goes somewhere else is worse than no field.
        self.ollama_host_input.setText(resolve_ollama_host())


        # Row 4: context indicator + reasoning controls + Load Cache + Show Context.
        # The reasoning toggle/buttons share this row (rather than a dedicated row
        # of their own) so the compact LLM panel doesn't overflow and clip its
        # lower controls below the panel's action bar when there's no context.
        row4 = QHBoxLayout()
        self.context_label = QLabel("没有视频上下文")
        self.context_label.setStyleSheet("color:#f44336;font-size:9pt;font-weight:bold;")
        row4.addWidget(self.context_label)

        self.reasoning_checkbox = QCheckBox("启用推理")
        self.reasoning_checkbox.setChecked(True)
        self.reasoning_checkbox.setToolTip(
            "启用后，大模型会推断检测到的物体与动作之间的关系。\n"
            "例如：“人物正在击打人物”“人物正在用杯子喝水”“多人正在交谈”"
        )
        row4.addWidget(self.reasoning_checkbox)

        self.reasoning_stats_btn = QPushButton("统计")
        fit_width(self.reasoning_stats_btn)
        self.reasoning_stats_btn.setToolTip("显示推理统计")
        self.reasoning_stats_btn.clicked.connect(self._show_reasoning_stats)
        row4.addWidget(self.reasoning_stats_btn)

        self.reasoning_save_btn = QPushButton("保存")
        fit_width(self.reasoning_save_btn)
        self.reasoning_save_btn.setToolTip("将推理结果保存到缓存")
        self.reasoning_save_btn.clicked.connect(self._save_reasoning_facts)
        row4.addWidget(self.reasoning_save_btn)

        row4.addStretch()

        self.load_cache_btn = QPushButton("加载缓存")
        self.load_cache_btn.setToolTip("手动选择 .cache.json 文件")
        self.load_cache_btn.clicked.connect(self._load_cache_from_file)
        row4.addWidget(self.load_cache_btn)

        self.show_context_btn = QPushButton("显示上下文")
        self.show_context_btn.setToolTip("查看大模型实际接收到的完整文本")
        self.show_context_btn.clicked.connect(self._show_context_debug)
        row4.addWidget(self.show_context_btn)

        settings_layout.addLayout(row4)
        
        # Visual search — its own foldable section, sibling of the settings
        # (it used to be nested inside them, so it vanished with them too).
        search_group = CollapsibleSection("视觉搜索", settings_key="llm/visual-search")
        search_layout = QHBoxLayout()
        
        search_layout.addWidget(QLabel("搜索内容："))
        self.search_target = QLineEdit()
        self.search_target.setPlaceholderText("爆炸、人物、汽车等")
        search_layout.addWidget(self.search_target)
        
        search_layout.addWidget(QLabel("从："))
        self.search_start_time = TimeSpinBox()
        self.search_start_time.setRange(0, 99999)
        self.search_start_time.setValue(0)
        self.search_start_time.setSingleStep(10)
        self.search_start_time.setDecimals(2)
        self.search_start_time.setToolTip(
            "从此时间点开始搜索。可输入秒数（90）或 mm:ss（1:30）。"
        )
        search_layout.addWidget(self.search_start_time)

        search_layout.addWidget(QLabel("间隔："))
        self.search_interval = TimeSpinBox()
        self.search_interval.setRange(0.5, 600.0)  # up to 10 minutes per step
        self.search_interval.setValue(1.0)
        self.search_interval.setSingleStep(1.0)
        self.search_interval.setDecimals(2)
        self.search_interval.setToolTip(
            "采样帧之间的时间间隔。可输入秒数（60）或 mm:ss（1:00）。"
        )
        search_layout.addWidget(self.search_interval)
        
        # Engine selector: CLIP (fast GPU ranker) / LLM (VLM) / CLIP+LLM (funnel)
        search_layout.addWidget(QLabel("引擎："))
        self.search_engine_combo = QComboBox()
        self.search_engine_combo.addItem("CLIP + 大模型", "clip_llm")
        self.search_engine_combo.addItem("仅 CLIP", "clip")
        self.search_engine_combo.addItem("仅大模型", "llm")
        self.search_engine_combo.setToolTip(
            "仅 CLIP：使用 GPU 快速进行相似度排序，适合宽泛概念。\n"
            "仅大模型：视觉模型逐帧检查，速度较慢，但可以进行推理。\n"
            "CLIP + 大模型：CLIP 先在 GPU 上对全部帧排序，再由大模型确认 Top-K 候选。"
        )
        self.search_engine_combo.currentIndexChanged.connect(self._on_search_engine_changed)
        search_layout.addWidget(self.search_engine_combo)

        self.search_topk_label = QLabel("Top-K：")
        search_layout.addWidget(self.search_topk_label)
        self.search_topk = QSpinBox()
        self.search_topk.setRange(1, 1000)
        self.search_topk.setValue(30)
        self.search_topk.setToolTip(
            "保留多少个 CLIP 高分候选（仅 CLIP），或发送给大模型"
            "进行确认（CLIP + 大模型）。数值越高召回率越高，但速度更慢。"
        )
        search_layout.addWidget(self.search_topk)
        self.search_topk.valueChanged.connect(self._save_search_prefs)

        # Restore last-used engine + Top-K (first run defaults to CLIP only —
        # frictionless, needs no connected model).
        _ss = QSettings(self.SETTINGS_KEY, "LLMChat")
        _eidx = self.search_engine_combo.findData(_ss.value("search_engine", "clip"))
        if _eidx >= 0:
            self.search_engine_combo.setCurrentIndex(_eidx)
        try:
            self.search_topk.setValue(int(_ss.value("search_topk", 30)))
        except (TypeError, ValueError):
            pass
        self._on_search_engine_changed()  # sync Top-K visibility to restored engine

        # Add "stop on first match" checkbox
        self.stop_on_match_cb = QCheckBox("找到后停止")
        # Off by default: a full scan reports every match and, since frame
        # embeddings are memoised, it also leaves the video fully indexed — so
        # every later search on it is instant. Stopping early saves time once
        # and forfeits that.
        self.stop_on_match_cb.setChecked(False)
        self.stop_on_match_cb.setToolTip(
            "勾选后，搜索会在第一个匹配处停止并跳转。\n"
            "取消勾选时，将扫描完整视频并报告所有匹配。"
        )
        search_layout.addWidget(self.stop_on_match_cb)

        self.search_btn = QPushButton("搜索")
        self.search_btn.setIcon(ui_icons.search())
        self.search_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.warning};color:white;font-weight:bold;padding:4px 10px;border-radius:4px;}}"
        )
        self.search_btn.clicked.connect(self._start_visual_search)
        search_layout.addWidget(self.search_btn)

        self.stop_search_btn = QPushButton("停止")
        self.stop_search_btn.setIcon(ui_icons.stop())
        self.stop_search_btn.setStyleSheet(
            "QPushButton{background:#8a2a2a;color:white;font-weight:bold;padding:4px 10px;border-radius:4px;}"
        )
        self.stop_search_btn.clicked.connect(self._stop_visual_search)
        self.stop_search_btn.setEnabled(False)
        search_layout.addWidget(self.stop_search_btn)

        # Match navigation: step through found timestamps with arrows
        self.search_prev_btn = QPushButton("◀")
        self.search_prev_btn.setFixedWidth(30)
        self.search_prev_btn.setToolTip("上一个匹配")
        self.search_prev_btn.clicked.connect(lambda: self._step_search_result(-1))
        self.search_prev_btn.setEnabled(False)
        search_layout.addWidget(self.search_prev_btn)

        self.search_result_label = QLabel("0/0")
        self.search_result_label.setStyleSheet("color:#999;min-width:36px;")
        self.search_result_label.setAlignment(Qt.AlignCenter)
        search_layout.addWidget(self.search_result_label)

        self.search_next_btn = QPushButton("▶")
        self.search_next_btn.setFixedWidth(30)
        self.search_next_btn.setToolTip("下一个匹配")
        self.search_next_btn.clicked.connect(lambda: self._step_search_result(1))
        self.search_next_btn.setEnabled(False)
        search_layout.addWidget(self.search_next_btn)

        # Search progress lives inside the section, under the controls row
        search_vbox = QVBoxLayout()
        search_vbox.setSpacing(4)
        search_vbox.addLayout(search_layout)

        self.search_progress = QLabel("")
        self.search_progress.setStyleSheet("color:#2f81f7;font-style:italic;font-size:9pt;")
        search_vbox.addWidget(self.search_progress)

        search_group.setContentLayout(search_vbox)

        settings_group.setContentLayout(settings_layout)
        self._settings_section = settings_group
        root.addWidget(settings_group)
        root.addWidget(search_group)

        # --- Chat display ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#ddd;"
            "font-family:'Segoe UI','SF Pro',sans-serif;font-size:10pt;"
            "border:1px solid #333;border-radius:4px;padding:6px;}"
        )
        self.chat_display.setMinimumHeight(150 if self._compact else 250)
        root.addWidget(self.chat_display, stretch=1)

        # --- Input bar ---
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("询问视频分析结果，或进行视觉搜索…")
        self.input_field.setStyleSheet(
            "QLineEdit{padding:4px 8px;font-size:10pt;border:1px solid #555;border-radius:4px;}"
        )
        self.input_field.returnPressed.connect(self._send_message)
        self.input_field.setEnabled(False)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("发送")
        self.send_btn.setStyleSheet(
            "QPushButton{background:#2f81f7;color:white;font-weight:bold;"
            "padding:4px 14px;border-radius:4px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setEnabled(False)
        input_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#8a2a2a;color:white;font-weight:bold;"
            "padding:4px 12px;border-radius:4px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        input_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_layout.addWidget(self.clear_btn)

        self.free_chat_chk = QCheckBox("自由对话")
        self.free_chat_chk.setToolTip(
            "勾选后，大模型可以自由回答，不仅限于视频数据"
        )
        input_layout.addWidget(self.free_chat_chk)

        root.addLayout(input_layout)
        self.setLayout(root)
        self._refresh_models()

    # --------------------------------------------------------- Public API

    def set_analysis_data(self, data: dict, video_path: str = ""):
        """将视频分析缓存提供给大模型作为上下文。"""
        self._analysis_data = data
        self._video_path = video_path
        self._update_context_label()
        
        # Initialize analyzer if we have video path
        if video_path and os.path.exists(video_path):
            self._init_analyzer()
        
        # Initialize reasoning engine if enabled. set_analysis_data() can be
        # called twice for the same cache (widget auto-load, then the timeline
        # feeding its own copy), so skip re-announcing identical stats.
        if hasattr(self, 'reasoning_checkbox') and self.reasoning_checkbox.isChecked():
            sig = self._analysis_signature(data)
            if sig == getattr(self, '_reasoning_sig', None):
                return
            self._reasoning_sig = sig
            try:
                from .llm_reasoning import ReasoningLLMIntegration
                llm = self._llm if (self._llm and self._llm.is_loaded()) else None
                self.reasoning_engine = ReasoningLLMIntegration(
                    llm, data, video_path
                )
                stats = self.reasoning_engine.reasoning_engine.get_action_statistics()
                self._append_system(
                    f"🧠 推理引擎已初始化，共分析 {stats['total_actions']} 个动作"
                )
                if not llm:
                    self._append_system(
                        "ℹ️ 当前已可查看统计；连接模型后可进一步回答推理类问题。"
                    )
            except Exception as e:
                self._append_system(f"⚠️ 无法初始化推理引擎：{e}")
                self.reasoning_engine = None

    @staticmethod
    def _analysis_signature(data: dict) -> tuple:
        """Lightweight fingerprint of a cache's contents, used to suppress
        duplicate reasoning-init messages when the same data is fed twice."""
        if not isinstance(data, dict):
            return (0, 0, 0, 0, 0)
        t_data = data.get("transcript", {})
        n_trans = len(t_data.get("segments", [])) if isinstance(t_data, dict) else 0
        meta = data.get("video_metadata", {})
        return (
            int(meta.get("duration", 0) or 0),
            len(data.get("objects", [])),
            len(data.get("actions", [])),
            n_trans,
            len(data.get("scenes", [])),
        )

    def set_preview_window(self, preview):
        """Connect to a VideoPreviewWindow for seek sync."""
        self._preview_window = preview

    def set_timeline_window(self, window):
        """Connect to a SignalTimelineWindow for timeline control."""
        if self._timeline_bridge:
            self._timeline_bridge.set_timeline_window(window)
            self._timeline_bridge.set_scan_callback(self._trigger_visual_scan)
            self._update_context_label()
            self._append_system(
                "时间线已连接！现在可以通过对话编辑时间线。\n"
                "例如：添加 0:10 到 0:15 的片段、删除第 2 个片段、"
                "只显示人物检测、播放 0:30 处的片段。"
            )

        # Always grab video path from timeline window
        if hasattr(window, 'video_path') and window.video_path:
            self._video_path = window.video_path
            self._update_context_label()

        # Sync search start time with timeline clicks
        if hasattr(window, 'signal_scene'):
            window.signal_scene.time_clicked.connect(self._update_search_start_time)

    def _update_search_start_time(self, time):
        """Update visual search start time when user clicks timeline."""
        if hasattr(self, 'search_start_time'):
            self.search_start_time.setValue(time)

    def get_llm_module(self) -> Optional[LLMModule]:
        return self._llm

    def get_analyzer(self) -> Optional[VideoSeekAnalyzer]:
        return self._analyzer

    def load_cache_for_video(self, video_path: str) -> bool:
        """Try to find and load a cache file for a specific video."""
        if not video_path or not os.path.exists(video_path):
            return False
        cache_dir = Path(self._cache_dir)
        if not cache_dir.exists():
            return False

        video_stem = Path(video_path).stem.lower()
        all_caches = sorted(
            cache_dir.glob("*.cache.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for cf in all_caches:
            if video_stem in cf.stem.lower():
                success = self._load_cache_file(str(cf), video_path)
                if success:
                    self._init_analyzer(video_path)
                return success

        try:
            from modules.media.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache(cache_dir=self._cache_dir)
            vhash = cache._get_video_hash(video_path)
            matching = sorted(
                cache_dir.glob(f"{vhash}*.cache.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matching:
                success = self._load_cache_file(str(matching[0]), video_path)
                if success:
                    self._init_analyzer(video_path)
                return success
        except Exception:
            pass

        return False

    def set_video_path(self, video_path: str):
        """Set the video path and initialize analyzer."""
        self._video_path = video_path
        if video_path and os.path.exists(video_path):
            self._init_analyzer(video_path)

    # ------------------------------------------------ Analyzer initialization

    def _init_analyzer(self, video_path: str = None, require_llm: bool = True):
        """Initialize VideoSeekAnalyzer for visual operations.

        require_llm=False allows CLIP-only search, which never touches the VLM.
        """
        path = video_path or self._video_path
        if not path or not os.path.exists(path):
            return False

        if require_llm and (not self._llm or not self._llm.is_loaded()):
            self._append_system("⚠️ 大模型未连接。此引擎需要已连接的视觉模型。")
            return False
        
        try:
            if self._analyzer:
                self._analyzer.close()
            
            self._analyzer = VideoSeekAnalyzer(path, self._llm, verbose=False)
            self._append_system(f"✅ 视频分析器已就绪：{os.path.basename(path)}")
            return True
        except Exception as e:
            self._append_system(f"❌ 初始化分析器失败：{e}")
            return False

    # ------------------------------------------------ Visual search

    def _on_search_engine_changed(self):
        """Top-K only applies to the CLIP-based engines; hide it for LLM-only."""
        engine = self.search_engine_combo.currentData()
        is_clip = engine in ("clip", "clip_llm")
        self.search_topk_label.setVisible(is_clip)
        self.search_topk.setVisible(is_clip)
        self._save_search_prefs()

    def _save_search_prefs(self):
        """Persist the engine + Top-K so they're remembered next session."""
        try:
            s = QSettings(self.SETTINGS_KEY, "LLMChat")
            s.setValue("search_engine", self.search_engine_combo.currentData())
            s.setValue("search_topk", self.search_topk.value())
        except Exception:
            pass

    def _search_thread_running(self) -> bool:
        """Whether the visual-search worker thread is alive and running.

        The QThread is deleteLater'd when it finishes (finished→deleteLater), so
        the Python attribute can outlive its C++ object. Touching .isRunning() on
        that dead wrapper raises RuntimeError ('Internal C++ object already
        deleted') — which is exactly the crash on a second search. Treat that as
        'not running' and drop the stale refs so the next run starts clean."""
        t = self._search_worker_thread
        if t is None:
            return False
        try:
            return t.isRunning()
        except RuntimeError:
            self._search_worker_thread = None
            self._search_worker = None
            return False

    def _llm_thread_running(self) -> bool:
        """Whether the LLM chat worker thread is alive and running.

        Same lifecycle trap as _search_thread_running: the QThread is
        deleteLater'd on finish, so its Python wrapper can outlive the C++
        object and .isRunning() then raises. Treat that as 'not running' and
        drop the stale refs."""
        t = self._worker_thread
        if t is None:
            return False
        try:
            return t.isRunning()
        except RuntimeError:
            self._worker_thread = None
            self._worker = None
            return False

    def _start_visual_search(self):
        """Start visual search for target in video."""
        target = self.search_target.text().strip()
        if not target:
            self._append_system("❌ 请输入要搜索的内容")
            return

        engine = self.search_engine_combo.currentData()  # "clip_llm" | "clip" | "llm"
        top_k = self.search_topk.value()

        if not self._analyzer:
            if not self._init_analyzer(require_llm=(engine != "clip")):
                self._append_system("❌ 无法开始搜索：尚未加载视频或分析器未就绪")
                return

        # Engines that use the VLM need a live model. The analyzer may have been
        # created earlier by a CLIP-only run (with no LLM), so sync the current one.
        if engine != "clip":
            if not self._llm or not self._llm.is_loaded():
                self._append_system("⚠️ 此引擎需要连接视觉模型。请先连接模型，或使用“仅 CLIP”。")
                return
            self._analyzer.llm = self._llm
        
        if self._search_thread_running():
            # Cancel the previous worker. The QObject+moveToThread pattern
            # handles cleanup via the finished→deleteLater chain — old worker
            # finishes its current frame, emits finished, thread quits, both
            # objects schedule themselves for deletion through the event loop.
            # Qt's parent-child ownership keeps the old thread alive until then,
            # even though we're about to reassign self._search_worker_thread.
            if self._search_worker:
                self._search_worker.cancel()
                try:
                    self._search_worker.disconnect()
                except (RuntimeError, TypeError):
                    pass
            self._append_system(
                "⏹ 正在取消上一次搜索（当前帧会在后台处理完成）…"
            )

        interval = self.search_interval.value()
        start_from = self.search_start_time.value()
        remaining = max(0, self._analyzer.duration - start_from)
        max_seeks = int(remaining / interval) + 1
        stop_on_match = self.stop_on_match_cb.isChecked()
        
        mode_str = "找到第一个匹配后停止" if stop_on_match else "扫描完整视频"
        start_str = f"{int(start_from)//60}:{int(start_from)%60:02d}"

        # Reset match navigation for this fresh search.
        self._reset_search_results()

        # Generate a unique scan ID and clear any previous findings for this query
        # so the timeline shows fresh results.
        self._current_scan_id = f"scan_{int(_time.time())}"
        if (self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'signal_scene')):
            scene = self._timeline_bridge._window.signal_scene
            if hasattr(scene, 'clear_visual_findings'):
                scene.clear_visual_findings(query=target)

        engine_label = {"clip_llm": "CLIP+LLM", "clip": "CLIP", "llm": "LLM"}.get(engine, engine)
        extra = f", top-{top_k}" if engine in ("clip", "clip_llm") else ""
        self._append_system(
            f"🔍 [{engine_label}{extra}] 正在从 {start_str} 开始搜索“{target}” "
            f"每 {interval} 秒采样一次（{mode_str}）…"
        )
        self.search_progress.setText(f"正在搜索“{target}” [{engine_label}]…")
        self.search_btn.setEnabled(False)
        self.stop_search_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)  # Also enable main Stop button
        
        self._search_worker_thread = QThread(self)  # parent=self for Qt ownership
        self._search_worker = _VisualSearchWorker(
            analyzer=self._analyzer,
            target=target,
            interval=interval,
            max_seeks=max_seeks,
            stop_on_first_match=stop_on_match,
            start_time=start_from,
            mode=engine,
            top_k=top_k,
        )
        self._search_worker.moveToThread(self._search_worker_thread)

        # Lifecycle
        self._search_worker_thread.started.connect(self._search_worker.run)
        self._search_worker.finished.connect(self._search_worker_thread.quit)
        self._search_worker.finished.connect(self._search_worker.deleteLater)
        self._search_worker.error.connect(self._search_worker_thread.quit)
        self._search_worker_thread.finished.connect(self._search_worker_thread.deleteLater)

        # Our handlers
        self._search_worker.progress.connect(self._on_search_progress)
        self._search_worker.frame_analyzed.connect(self._on_frame_analyzed)
        self._search_worker.found.connect(self._on_search_found)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)

        self._search_worker_thread.start()

    def _stop_visual_search(self):
        """Stop ongoing visual search."""
        if self._search_thread_running():
            self._search_worker.cancel()
            self._append_system(
                "⏹ 已请求停止，将在当前帧处理完成后结束。\n"
                "   （GGUF 图像解码无法在单帧处理中途停止）"
            )
            self.search_progress.setText("当前帧结束后停止…")
            self.search_btn.setEnabled(True)
            self.stop_search_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def _trigger_visual_scan(self, target: str, interval: float):
        """Called by TimelineBridge when LLM generates [CMD:visual_scan]."""
        self.search_target.setText(target)
        self.search_interval.setValue(interval)
        self._start_visual_search()

    @Slot(int, int, float, str)
    def _on_search_progress(self, current: int, total: int, timestamp: float, preview: str):
        """Update search progress.

        `preview` names the *phase* ("CLIP scan 240s", "VLM confirm 240s
        (CLIP 0.42)"). Dropping it made the two-stage engines unreadable: the
        CLIP+LLM funnel scans every frame first and only then sends the top-K to
        the model, but the label showed the same "Searching: n/total" for both,
        so a slow scan looked like the model running on every frame.
        """
        percent = (current / total * 100) if total else 0.0
        head = f"{preview} — " if preview else "正在搜索："
        self.search_progress.setText(
            f"{head}{current}/{total} ({percent:.1f}%)"
        )

    @Slot(float, str, str, bool, float)
    def _on_frame_analyzed(self, timestamp: float, timestamp_str: str, response: str,
                           contains_target: bool, score: float = 1.0):
        """Show each frame's YES/NO result in chat and update preview for ALL frames."""
        icon = "✅" if contains_target else "❌"
        short = response[:120] + "..." if len(response) > 120 else response
        self._append_system(f"  {icon} [{timestamp_str}] {short}")

        # ── Live push YES hits onto the signal timeline ──
        if (contains_target
                and self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'add_visual_findings')):
            window = self._timeline_bridge._window

            # Resolve model name for metadata
            model_name = ''
            if self._llm:
                model_name = (getattr(self._llm, 'model', None)
                            or getattr(self._llm, 'model_path', None) or '')
                if model_name and ('/' in model_name or '\\' in model_name):
                    model_name = os.path.basename(model_name)

            finding = {
                'timestamp':  timestamp,
                'query':      self.search_target.text().strip(),
                # The engine's real score (CLIP similarity, 0-1); the VLM's binary
                # yes/no legitimately reports 1.0. Feeds the timeline's
                # visual-confidence filter and ranking.
                'confidence': float(score),
                'model':      model_name,
                'scan_id':    getattr(self, '_current_scan_id', ''),
                'analysis':   (response or '')[:200],
            }
            try:
                # save=False — defer disk write until scan finishes (see D3)
                window.add_visual_findings([finding], save=False)
            except Exception as e:
                print(f"⚠️ 无法将视觉搜索结果写入时间线：{e}")
        
        # Update preview window directly (QMediaPlayer has its own decoder,
        # so this is safe). Do NOT call _seek_to_timestamp() here — that
        # would re-seek the analyzer's cv2.VideoCapture from the GUI thread
        # while the worker thread is still using it, causing the FFmpeg
        # assertion: "!dst->progress failed at libavcodec/utils.c:954"
        if hasattr(self, '_preview_window') and self._preview_window:
            self._preview_window.seek_to_time(timestamp)
            self._preview_window.force_frame_update()
            
            if hasattr(self._preview_window, 'show_frame_analysis_status'):
                self._preview_window.show_frame_analysis_status(timestamp, contains_target)
            
            QApplication.processEvents()
        
        # Update timeline playhead (no decoder involved, just UI)
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            self._timeline_bridge._cmd_seek({'time': str(timestamp)})
        
        self.search_progress.setText(f"正在分析 {timestamp_str} 的画面…")

    @Slot(float, str, str)
    # ---- match navigation (◀ ▶) -------------------------------------------
    def _reset_search_results(self):
        self._search_results = []
        self._search_result_idx = -1
        self._update_search_nav()

    def _add_search_result(self, timestamp, timestamp_str, analysis):
        self._search_results.append({
            "timestamp": timestamp, "timestamp_str": timestamp_str, "analysis": analysis,
        })
        self._search_results.sort(key=lambda r: r["timestamp"])
        if self._search_result_idx < 0:
            self._search_result_idx = 0
        self._update_search_nav()

    def _update_search_nav(self):
        n = len(self._search_results)
        i = self._search_result_idx
        self.search_result_label.setText(f"{i + 1}/{n}" if n else "0/0")
        self.search_prev_btn.setEnabled(n > 0 and i > 0)
        self.search_next_btn.setEnabled(n > 0 and i < n - 1)

    def _step_search_result(self, delta: int):
        if not self._search_results:
            return
        self._search_result_idx = max(
            0, min(len(self._search_results) - 1, self._search_result_idx + delta)
        )
        r = self._search_results[self._search_result_idx]
        self._seek_to_timestamp(r["timestamp"])
        self._append_system(
            f"➡️ 匹配结果 {self._search_result_idx + 1}/{len(self._search_results)} "
            f"at {r['timestamp_str']}"
        )
        self._update_search_nav()

    def _on_search_found(self, timestamp: float, timestamp_str: str, analysis: str):
        """Handle found target — auto-seek preview and timeline to the found timestamp."""
        self._add_search_result(timestamp, timestamp_str, analysis)
        self._append_system(
            f"🎯 在 {timestamp_str} 找到匹配！\n"
            f"   {analysis[:150]}..."
        )
        
        # Only update preview + timeline, NOT the analyzer (worker may still be running)
        if hasattr(self, '_preview_window') and self._preview_window:
            self._preview_window.seek_to_time(timestamp)
            self._preview_window.force_frame_update()
        
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            self._timeline_bridge._cmd_seek({'time': str(timestamp)})

    @Slot(list)
    def _on_search_finished(self, results: list):
        """Handle search completion — worker is done, safe to seek analyzer."""
        found_results = [r for r in results if r.get("contains_target", False)]
        found_count = len(found_results)

        # Rebuild the authoritative match list for ◀ ▶ navigation.
        self._search_results = sorted(
            ({"timestamp": r["timestamp"], "timestamp_str": r["timestamp_str"],
              "analysis": r.get("analysis", "")} for r in found_results),
            key=lambda r: r["timestamp"],
        )
        self._search_result_idx = 0 if self._search_results else -1
        self._update_search_nav()

        if found_count > 0:
            ts_list = ", ".join(r["timestamp_str"] for r in found_results)
            self._append_system(
                f"✅ 搜索完成。在 {found_count} 个时间点找到“{self.search_target.text()}”：{ts_list}"
            )
            
            # Worker is finished so _seek_to_timestamp (which touches the
            # analyzer's cv2.VideoCapture) is safe now — no race condition.
            self._seek_to_timestamp(found_results[0]["timestamp"])

        else:
            self._append_system(
                f"❌ 搜索完成。视频中未找到“{self.search_target.text()}”。"
            )

        # ── Persist findings (added live during scan) to disk once ──
        if (self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'save_visual_findings_to_cache')):
            saved = self._timeline_bridge._window.save_visual_findings_to_cache()
            if saved and found_count > 0:
                self._append_system(
                    f"💾 已保存 {found_count} 条“{self.search_target.text()}”"
                    f"搜索结果到缓存（下次会话会自动重新加载）"
                )
        
        self.search_progress.setText("")
        self.search_btn.setEnabled(True)
        self.stop_search_btn.setEnabled(False)
        # Disable main stop button too (unless a chat query is still running)
        if not self._llm_thread_running():
            self.stop_btn.setEnabled(False)

    @Slot(str)
    def _on_search_error(self, error: str):
        """Handle search error."""
        self._append_system(f"❌ 搜索出错：{error}")
        self.search_progress.setText("搜索失败")
        self.search_btn.setEnabled(True)
        self.stop_search_btn.setEnabled(False)
        if not self._llm_thread_running():
            self.stop_btn.setEnabled(False)

    def _seek_to_timestamp(self, seconds: float):
        """Seek the timeline AND preview to a specific timestamp and ensure frame is displayed."""
        ts_str = f"{int(seconds)//60}:{int(seconds)%60:02d}"
        
        # Update analyzer position
        if self._analyzer:
            self._analyzer.current_time = seconds
            # Actually seek and read a frame to ensure it's 已加载
            frame = self._analyzer.seek_to_time(seconds)
            if frame is not None:
                # Optionally cache the frame for later use
                if hasattr(self, '_current_frame'):
                    self._current_frame = frame
        
        # Update the preview window with forced frame update
        if hasattr(self, '_preview_window') and self._preview_window:
            # First seek to the time
            self._preview_window.seek_to_time(seconds)
            
            # Then force a frame update
            self._preview_window.force_frame_update()
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # For debugging: check if frame was captured
            if hasattr(self._preview_window, 'capture_current_frame'):
                frame_image = self._preview_window.capture_current_frame()
                if frame_image:
                    self._append_system(f"📸 已截取 {ts_str} 的画面")
                else:
                    self._append_system(f"⚠️ 无法截取 {ts_str} 的画面")
        
        # Also use timeline bridge if available
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            result = self._timeline_bridge._cmd_seek({'time': str(seconds)})
            self._append_system(result)
        
        return True

    # ------------------------------------------------ Cache loading

    def _auto_load_latest_cache(self):
        """On startup, find and load the most recent cache file from disk."""
        cache_dir = Path(self._cache_dir)
        if not cache_dir.exists():
            self._append_system(
                f"缓存目录不存在：‘{self._cache_dir}’。"
                "请先运行处理流水线，或使用“加载缓存”。"
            )
            return

        all_caches = sorted(
            cache_dir.glob("*.cache.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not all_caches:
            self._append_system(
                "未找到缓存文件。请先对视频运行处理流程。"
            )
            return

        latest = all_caches[0]
        age_sec = _time.time() - latest.stat().st_mtime
        if age_sec < 60:
            age_str = f"{age_sec:.0f} 秒前"
        elif age_sec < 3600:
            age_str = f"{age_sec/60:.0f} 分钟前"
        elif age_sec < 86400:
            age_str = f"{age_sec/3600:.1f} 小时前"
        else:
            age_str = f"{age_sec/86400:.1f} 天前"

        self._append_system(f"正在自动加载最新缓存：{latest.name}（{age_str}）")
        self._load_cache_file(str(latest))

    def _load_cache_file(self, filepath: str, video_path: str = "") -> bool:
        """Load a .cache.json file and set it as analysis context."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self._append_system(f"缓存格式无效：应为字典结构，实际为 {type(data).__name__}")
                return False

            if not video_path:
                video_path = os.path.basename(filepath)

            self.set_analysis_data(data, video_path)

            # Summary
            meta = data.get("video_metadata", {})
            dur = meta.get("duration", 0)
            n_obj = len(data.get("objects", []))
            n_act = len(data.get("actions", []))
            t_data = data.get("transcript", {})
            n_trans = len(t_data.get("segments", [])) if isinstance(t_data, dict) else 0
            n_scenes = len(data.get("scenes", []))
            n_motion = len(data.get("motion_events", []))
            n_peaks = len(data.get("motion_peaks", []))
            audio = data.get("audio", {})
            n_audio = len(audio.get("peaks", [])) if isinstance(audio, dict) else 0

            self._append_system(
                f"缓存已加载：{os.path.basename(filepath)}\n"
                f"  时长：{int(dur)} 秒（{int(dur)//60} 分 {int(dur)%60:02d} 秒） | "
                f"物体：{n_obj} | 动作：{n_act} | "
                f"转录：{n_trans} 段 | 场景：{n_scenes}\n"
                f"  运动事件：{n_motion} | 运动峰值：{n_peaks} | 音频峰值：{n_audio}"
            )
            return True

        except json.JSONDecodeError as e:
            self._append_system(f"{os.path.basename(filepath)} 中的 JSON 无效：{e}")
            return False
        except Exception as e:
            self._append_system(f"加载缓存失败：{e}")
            return False

    def _load_cache_from_file(self):
        """Manual cache file picker dialog."""
        start_dir = self._cache_dir if os.path.isdir(self._cache_dir) else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "选择缓存文件", start_dir,
            "缓存文件 (*.cache.json);;JSON 文件 (*.json);;所有文件 (*)"
        )
        if path:
            self._load_cache_file(path)

    def _show_context_debug(self):
        """Show exactly what context text would be sent to the LLM."""
        if not self._analysis_data:
            self._append_system(
                "未加载分析数据！\n"
                "这会导致大模型缺少依据并产生幻觉。\n"
                "请先“加载缓存”，或先运行完整分析流程。"
            )
            return

        context_text = VideoContextBuilder.build(self._analysis_data, self._video_path)
        ctx_chars = len(context_text)
        ctx_lines = context_text.count("\n") + 1

        dlg = QDialog(self)
        dlg.setWindowTitle("大模型上下文调试 - 查看模型实际输入")
        dlg.setMinimumSize(700, 500)
        layout = QVBoxLayout()

        stats = QLabel(
            f"上下文：{ctx_chars:,} 字符 | {ctx_lines} 行 | "
            f"约 {ctx_chars // 4:,} 个词元（估算）\n"
            f"视频：{self._video_path}\n"
            f"数据字段：{', '.join(sorted(self._analysis_data.keys()))}"
        )
        stats.setStyleSheet("font-weight:bold;padding:4px;")
        layout.addWidget(stats)

        if ctx_chars > 8000:
            warn = QLabel(
                "警告：上下文较大！小模型（3B）可能忽略部分内容。"
                "建议使用 8B 及以上模型以获得更好效果。"
            )
            warn.setStyleSheet("color:#ff9800;font-weight:bold;padding:4px;")
            layout.addWidget(warn)

        text_view = QTextEdit()
        text_view.setReadOnly(True)
        text_view.setPlainText(context_text)
        text_view.setStyleSheet(
            "QTextEdit{font-family:'Consolas','Courier New',monospace;font-size:9pt;}"
        )
        layout.addWidget(text_view, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dlg.close)
        layout.addWidget(btn_box)

        dlg.setLayout(layout)
        dlg.exec()

    def _update_context_label(self):
        if not self._analysis_data:
            self.context_label.setText("没有视频上下文 — 大模型可能产生幻觉！")
            self.context_label.setStyleSheet("color:#f44336;font-size:9pt;font-weight:bold;")
            return

        meta = self._analysis_data.get("video_metadata", {})
        dur = meta.get("duration", 0)
        n_obj = len(self._analysis_data.get("objects", []))
        n_act = len(self._analysis_data.get("actions", []))
        t_data = self._analysis_data.get("transcript", {})
        n_trans = len(t_data.get("segments", [])) if isinstance(t_data, dict) else 0

        vname = os.path.basename(self._video_path) if self._video_path else "已加载"

        analyzer_status = " | 分析器：就绪" if self._analyzer else ""

        tl_status = ""
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            tl_status = " | 时间线：已连接"

        self.context_label.setText(
            f"上下文：{vname} | {int(dur)} 秒 | "
            f"{n_obj} 个物体 | {n_act} 个动作 | {n_trans} 条转录{tl_status}{analyzer_status}"
        )
        self.context_label.setStyleSheet("color:#4CAF50;font-size:9pt;font-weight:bold;")

    def _show_reasoning_stats(self):
        """Show reasoning engine statistics.
        
        Properly scoped — no more referencing `lines` from a different 
        branch, and handles missing attributes gracefully.
        """
        if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
            if self._analysis_data:
                self._append_system("⚠️ 推理引擎尚未初始化，请尝试切换“启用推理”。")
            else:
                self._append_system("⚠️ 尚未加载分析数据，请先加载缓存文件。")
            return
        
        try:
            # Try to get action summary first (newer API)
            summary = self.reasoning_engine.get_action_summary()
            self._append_system(summary)
        except AttributeError:
            # Fallback to older method
            try:
                stats = self.reasoning_engine.reasoning_engine.get_action_statistics()
                lines = [
                    "📊 **动作统计**",
                    f"• 动作总数：{stats['total_actions']}",
                    f"• 动作类型数：{stats['unique_actions']}",
                    f"• 包含动作的时间点：{stats['timestamps_with_actions']}",
                    f"• 动作聚类数：{stats['action_clusters']}",
                    "\n最常见动作："
                ]
                for action, count in stats['most_common'][:5]:
                    lines.append(f"  • {action}：{count} 次")
                
                # This block was previously OUTSIDE the except, referencing
                # `lines` that only existed inside. Now it's properly scoped.
                if hasattr(self.reasoning_engine, 'reasoning_engine') and \
                   hasattr(self.reasoning_engine.reasoning_engine, 'action_sequences') and \
                   self.reasoning_engine.reasoning_engine.action_sequences:
                    lines.append("\n🎬 **检测到的动作序列：**")
                    for seq in self.reasoning_engine.reasoning_engine.action_sequences[:3]:
                        lines.append(
                            f"  • {seq.description} "
                            f"({int(seq.start_time)//60}:{int(seq.start_time)%60:02d} - "
                            f"{int(seq.end_time)//60}:{int(seq.end_time)%60:02d})"
                        )
                
                self._append_system("\n".join(lines))
            except Exception as e:
                self._append_system(f"⚠️ 无法获取统计信息：{e}")

    def _save_reasoning_facts(self):
        """Save inferred facts to cache."""
        if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
            self._append_system("⚠️ 没有可保存的推理引擎。")
            return
        
        try:
            saved_path = self.reasoning_engine.save_analysis(self._cache_dir)
            self._append_system(f"💾 推理结果已保存到：{os.path.basename(saved_path)}")
        except Exception as e:
            self._append_system(f"❌ 保存失败：{e}")

    # ------------------------------------------------ Handlers

    def _on_backend_changed(self, _index):
        backend = self.backend_combo.currentData()
        is_gguf = backend == "llama-cpp"
        self.ollama_row_widget.setVisible(not is_gguf)
        self.gguf_row_widget.setVisible(is_gguf)
        self.mmproj_row_widget.setVisible(is_gguf)
        self.refresh_btn.setVisible(not is_gguf)
        self.model_combo.setEnabled(not is_gguf)

        if is_gguf:
            self._populate_recent_gguf()
        else:
            self._refresh_models()

    def _on_ollama_host_edited(self):
        """Store the new server and ask *it* what models it has.

        Storing here rather than at Connect is deliberate: the model dropdown
        beside this field is filled from the server, so a host typed and left
        alone must already be the one being listed, or the user picks a tag from
        the old machine and connects to the new one.
        """
        url = remember_ollama_host(self.ollama_host_input.text())
        self.ollama_host_input.setText(url or resolve_ollama_host())
        try:
            from modules.narration.llm_discovery import forget_ollama_models
            forget_ollama_models()
        except Exception:                       # pragma: no cover - defensive
            pass
        if self.backend_combo.currentData() == "ollama":
            self._refresh_models()

    def _populate_recent_gguf(self):
        """Fill model combo with recently used GGUF files."""
        self.model_combo.clear()
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        recent = settings.value("recent_gguf_paths", [])
        # QSettings may return a string instead of list if only 1 item
        if isinstance(recent, str):
            recent = [recent] if recent else []

        if recent:
            for path in recent:
                # Show just filename in dropdown, store full path as data
                self.model_combo.addItem(os.path.basename(path), path)
            # Auto-fill the path input when user picks from dropdown
            self.model_combo.currentIndexChanged.connect(self._on_recent_gguf_selected)
            # Select the most recent one
            self.model_combo.setCurrentIndex(0)
            self._on_recent_gguf_selected(0)
        else:
            self.model_combo.addItem("（没有最近使用的模型 — 请使用“浏览”）")

    def _on_recent_gguf_selected(self, index):
        """When user picks a recent GGUF from dropdown, fill the path input."""
        path = self.model_combo.currentData()
        if path and os.path.exists(path):
            self.gguf_path_input.setText(path)

    def _refresh_models(self):
        self.model_combo.clear()
        backend = self.backend_combo.currentData()
        
        if backend == "llama-cpp":
            self.model_combo.addItem("（请在下方选择 GGUF 文件）")
            return
            
        if backend == "ollama":
            host = resolve_ollama_host()
            # Named only when it is not the default: on localhost the URL is
            # noise, and on another machine it is the whole answer.
            where = f"（{host}）" if ollama_is_remote(host) else ""
            models = get_ollama_models(host)
            if models:
                for m in models:
                    self.model_combo.addItem(m)
                self.status_label.setText(
                    f"找到 {len(models)} 个 Ollama 模型{where}")
                self.status_label.setStyleSheet("color:#4CAF50;font-style:italic;")
            else:
                for m in ["llama3.2", "llama3.2-vision", "llava", "bakllava", "llava-llama3"]:
                    self.model_combo.addItem(m)
                self.status_label.setText(
                    f"未检测到可用的 Ollama 服务{where or '（本机）'}，正在显示"
                    "默认模型（推荐使用视觉模型）")
                self.status_label.setStyleSheet("color:#ff9800;font-style:italic;")

    def _browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 GGUF 模型", "", "GGUF 模型 (*.gguf);;所有文件 (*)"
        )
        if path:
            self.gguf_path_input.setText(path)

    def _browse_mmproj(self):
        """Browse for mmproj file (for vision models)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 mmproj 文件", "", "GGUF 模型 (*.gguf);;所有文件 (*)"
        )
        if path:
            self.mmproj_path_input.setText(path)

    def _connect_llm(self):
        backend = self.backend_combo.currentData()
        model = self.model_combo.currentText().strip()
        gguf_path = ""
        mmproj_path = None

        self.status_label.setText("正在连接…")
        self.status_label.setStyleSheet("color:#2f81f7;font-style:italic;")
        self.connect_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if backend == "ollama":
                self._llm = LLMModule(backend="ollama", model=model,
                                      base_url=resolve_ollama_host(),
                                      log_fn=self._log)
            elif backend == "llama-cpp":
                gguf_path = self.gguf_path_input.text().strip()
                if not gguf_path:
                    raise ValueError("请先选择 GGUF 模型文件")
                
                mmproj_path = self.mmproj_path_input.text().strip() or None
                
                self._llm = LLMModule(
                    backend="llama-cpp", 
                    model_path=gguf_path,
                    mmproj_path=mmproj_path,
                    log_fn=self._log
                )
                
                model_name = os.path.basename(gguf_path)
                self.status_label.setText(f"正在加载 {model_name}…")
                QApplication.processEvents()
            else:
                raise ValueError(f"未知运行后端：{backend}")

            self._llm.load()

            self._save_gguf_to_recent(gguf_path)
            settings = QSettings(self.SETTINGS_KEY, "LLMChat")
            settings.setValue("last_gguf_path", gguf_path)
            if mmproj_path:
                settings.setValue("last_mmproj_path", mmproj_path)

            if backend == "llama-cpp":
                model_name = os.path.basename(gguf_path)
                if mmproj_path:
                    model_name += "（支持视觉）"
                self.status_label.setText(f"已连接：{model_name}")
            else:
                self.status_label.setText(f"已连接：{model}")
                
            self.status_label.setStyleSheet(f"color:{THEME.success};font-weight:bold;")
            # Folded section header mirrors the connection state
            self._settings_section.set_hint(self.status_label.text())
            self.connect_btn.setText("重新连接")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)

            if self._video_path and os.path.exists(self._video_path):
                self._init_analyzer()

            # Update reasoning engine with now-connected LLM
            if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
                self.reasoning_engine.llm = self._llm
                self._append_system("🧠 推理引擎已连接大模型，现在可以回答“为什么”类问题")

            if self._analysis_data:
                n_obj = len(self._analysis_data.get("objects", []))
                n_act = len(self._analysis_data.get("actions", []))
                self._append_system(
                    f"已连接到 {model if backend=='ollama' else os.path.basename(gguf_path)}。"
                    f"上下文已就绪：{n_obj} 条物体记录，{n_act} 条动作记录。"
                )
            else:
                self._append_system(
                    f"已连接到 {model if backend=='ollama' else os.path.basename(gguf_path)}。"
                    f"警告：没有视频上下文！请先使用“加载缓存”。"
                )

        except Exception as e:
            self.status_label.setText(f"错误：{e}")
            self.status_label.setStyleSheet("color:#f44336;font-style:italic;")
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
        finally:
            self.connect_btn.setEnabled(True)

    def _save_gguf_to_recent(self, path: str):
        """Add a GGUF path to the recent list (most recent first, no duplicates)."""
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        recent = settings.value("recent_gguf_paths", [])
        if isinstance(recent, str):
            recent = [recent] if recent else []

        # Remove if already present, then prepend
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)

        # Cap the list
        recent = recent[:self.MAX_RECENT_GGUF]
        settings.setValue("recent_gguf_paths", recent)

        # Refresh dropdown if currently showing GGUF mode
        if self.backend_combo.currentData() == "llama-cpp":
            self._populate_recent_gguf()

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        if not self._llm or not self._llm.is_loaded():
            self._append_system("尚未连接，请先点击“连接”。")
            return
        if self._llm_thread_running():
            self._append_system("仍在生成中，请稍候。")
            return
        
        # Special handling for reasoning questions
        if (hasattr(self, 'reasoning_engine') and self.reasoning_engine and 
            text.lower().startswith(('why ', 'how do you know', 'explain '))):
            
            current_time = 0
            if self._timeline_bridge and self._timeline_bridge._window:
                current_time = getattr(self._timeline_bridge._window, 'current_time', 0)
            
            answer = self.reasoning_engine.answer_why_question(text, current_time)
            if answer:
                self._append_user(text)
                self._append_html(
                    f'<div style="color:#FFD700;margin:8px 0;padding:8px;'
                    f'background:#2a2a2a;border-left:4px solid #FFD700;'
                    f'font-family:monospace;">'
                    f'🔍 {answer}</div>'
                )
                self.input_field.clear()
                self._chat_history.append({"role": "user", "content": text})
                self._chat_history.append({"role": "assistant", "content": answer})

                MAX_HISTORY_PAIRS = 6
                if len(self._chat_history) > MAX_HISTORY_PAIRS * 2:
                    self._chat_history = self._chat_history[-(MAX_HISTORY_PAIRS * 2):]

                return

        # Parse search requests from chat (e.g. "search for explosion every 60s")
        if self._try_parse_chat_search(text):
            self._append_user(text)
            self.input_field.clear()
            return

        # Check for mode commands
        force_visual = False
        force_text = False
        actual_message = text
        
        if text.startswith('!visual'):
            force_visual = True
            actual_message = text[7:].strip()
            self._append_system("🎯 已强制切换到视觉模式——将截取并分析当前画面")
        elif text.startswith('!text'):
            force_text = True
            actual_message = text[5:].strip()
            self._append_system("📝 已强制切换到文本模式——忽略视觉关键词，仅使用分析数据")

        # Handle seek commands (still useful)
        if self._handle_seek_command(actual_message):
            self.input_field.clear()
            return

        if not self._analysis_data:
            self._append_system(
                "⚠️ 未加载分析数据，大模型可能缺少依据并产生幻觉。\n"
                "请先使用“加载缓存”选择缓存文件。"
            )

        self._append_user(actual_message)
        self.input_field.clear()
        self._chat_history.append({"role": "user", "content": actual_message})

        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("...")
        self.stop_btn.setEnabled(True)

        self._append_html(
            '<div style="color:#8BC34A;margin-top:8px;"><b>助手：</b></div>'
        )

        # Build timeline context if connected
        timeline_ctx = ""
        has_timeline = self._timeline_bridge and self._timeline_bridge.is_connected
        if has_timeline:
            timeline_ctx = (
                self._timeline_bridge.get_timeline_state() + "\n" +
                self._timeline_bridge.get_available_commands_text()
            )

        if has_timeline:
            timeline_ctx = self._timeline_bridge.get_timeline_state()
            # Truncate clip list for small models
            lines = timeline_ctx.split('\n')
            if len(lines) > 15:
                timeline_ctx = '\n'.join(lines[:15]) + f'\n…（另有 {len(lines)-15} 行）'
            timeline_ctx += '\n' + self._timeline_bridge.get_available_commands_text()

        # A report handed over by seed_from_report rides along with every
        # message, not just the first — the model is stateless between turns,
        # so a follow-up would otherwise be answered about nothing.
        advisor_ctx = getattr(self, "_advisor_context", "")
        if advisor_ctx:
            timeline_ctx = (advisor_ctx + "\n\n" + timeline_ctx
                            if timeline_ctx else advisor_ctx)

        # Capture frame based on mode
        frame_b64 = None
        _text_lower = actual_message.lower()
        
        if force_visual:
            if self._timeline_bridge and self._timeline_bridge._window:
                window = self._timeline_bridge._window
                if hasattr(window, 'capture_current_frame_base64'):
                    frame_b64 = window.capture_current_frame_base64()
                    if frame_b64:
                        self._append_system(
                            f"📷 已截取 {window.current_time:.1f} 秒处画面"
                            f"（{len(frame_b64)//1024} KB）"
                        )
                        if has_timeline:
                            self._append_system(
                                "ℹ️ 正在结合画面分析与时间线上下文…"
                            )
                    else:
                        self._append_system("⚠️ 画面截取失败")
            else:
                self._append_system("⚠️ 无法截取画面：尚未连接时间线窗口")
        
        elif force_text:
            frame_b64 = None
            self._append_system("ℹ️ 文本模式已启用：仅使用分析数据，不读取画面")
        
        else:
            _wants_vision = any(kw in _text_lower for kw in _VISION_KEYWORDS)
            _asks_about_current = any(phrase in _text_lower for phrase in 
                                    ["current frame", "this frame", "what do you see", 
                                    "what's happening now", "describe this frame"])
            
            if (_wants_vision or _asks_about_current) and self._timeline_bridge and self._timeline_bridge._window:
                window = self._timeline_bridge._window
                if hasattr(window, 'capture_current_frame_base64'):
                    frame_b64 = window.capture_current_frame_base64()
                    if frame_b64:
                        self._append_system(
                            f"📷 已截取 {window.current_time:.1f} 秒处画面"
                            f"（{len(frame_b64)//1024} KB）"
                        )
                        if has_timeline:
                            self._append_system(
                                "ℹ️ 正在结合画面分析与时间线上下文…"
                            )
                    else:
                        self._append_system("⚠️ 画面截取失败")

        # Create worker and host thread
        _free_chat = self.free_chat_chk.isChecked()
        self._worker_thread = QThread(self)
        self._worker = _LLMWorker(
            llm=self._llm,
            message=actual_message,
            analysis_data=self._analysis_data if not _free_chat else None,
            video_path=self._video_path,
            timeline_context=timeline_ctx,
            frame_base64=frame_b64,
            free_chat_mode=_free_chat,
        )
        self._worker.moveToThread(self._worker_thread)

        # Lifecycle — worker.finished triggers thread.quit + deleteLater on both
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        # Our handlers
        self._worker.token_received.connect(self._on_token)
        self._worker.finished.connect(self._on_response_done)
        self._worker.error.connect(self._on_response_error)

        self._worker_thread.start()

    def _try_parse_chat_search(self, text: str) -> bool:
        """Parse visual search requests typed in the chat input.
        
        Handles natural language like:
            "search for explosion every 60s"
            "seek through video every 60s until you find a goal"
            "please seek every 60s until you will find X"
            "go through video every 30s looking for X"
            "scan every 10s for fire"
            "find person every 30s"
            "search for car" (uses current interval)
        
        Returns True if the message was handled as a search command.
        """
        import re
        text_lower = text.lower().strip()
        
        # Quick check: does this look like a search/seek request at all?
        # Must contain at least one action keyword AND either "every" or "search/find for"
        action_words = ('seek', 'search', 'scan', 'find', 'look for', 'looking for',
                        'go through', 'scrub through')
        has_action = any(w in text_lower for w in action_words)
        if not has_action:
            return False
        
        # --- Extract interval (if present) ---
        interval = None
        interval_match = re.search(r'every\s+(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?', text_lower)
        if interval_match:
            interval = float(interval_match.group(1))
        
        # --- Extract target ---
        target = None
        
        # Pattern group 1: "until you [will] find <target>"  /  "until <target> is found"
        # Covers: "seek every 60s until you find X", "seek until you will find X"
        m = re.search(r'until\s+(?:you\s+)?(?:will\s+)?(?:find|see|spot|locate)\s+(.+)', text_lower)
        if m:
            target = m.group(1).strip()
        
        if not target:
            m = re.search(r'until\s+(.+?)\s+is\s+(?:found|seen|spotted|located|detected)', text_lower)
            if m:
                target = m.group(1).strip()

        # Pattern group 2: "looking for <target>"  /  "for <target>" at end
        if not target:
            m = re.search(r'(?:looking|searching|scanning)\s+for\s+(.+)', text_lower)
            if m:
                target = m.group(1).strip()
        
        # Pattern group 3: "search/scan/find for <target> [every Ns]"
        if not target:
            m = re.search(r'(?:search|scan|find|look)\s+(?:for|the)\s+(.+?)(?:\s+every\s+|$)', text_lower)
            if m:
                target = m.group(1).strip()
        
        # Pattern group 3b: "find <target> every Ns" (no "for")
        if not target:
            m = re.search(r'(?:find|seek)\s+(.+?)\s+every\s+', text_lower)
            if m:
                target = m.group(1).strip()
                # Don't match filler like "find through video every"
                filler = {'through', 'the', 'video', 'in', 'a', 'an', 'this', 'my'}
                if target in filler or all(w in filler for w in target.split()):
                    target = None
        
        # Pattern group 4: "every Ns for <target>"  (interval before target)
        if not target:
            m = re.search(r'every\s+\d+(?:\.\d+)?\s*s(?:ec(?:ond)?s?)?\s+(?:for|to find|looking for)\s+(.+)', text_lower)
            if m:
                target = m.group(1).strip()
        
        if not target:
            return False
        
        # --- Clean up target ---
        # Remove common trailing filler words and punctuation
        target = re.sub(
            r'\s+(?:in\s+(?:the\s+)?(?:video|frame|frames|clip)|every\s+\d+.*|please|thanks?)\.?$',
            '', target
        ).strip()
        target = target.rstrip('.,!?')
        
        # If we still have "every Ns" embedded in the target, remove it
        target = re.sub(r'\s*every\s+\d+(?:\.\d+)?\s*s(?:ec(?:ond)?s?)?\s*', ' ', target).strip()
        
        if not target or len(target) < 2:
            return False
        
        # --- Apply ---
        self.search_target.setText(target)
        if interval is not None:
            self.search_interval.setValue(interval)
            self._append_system(f"🔍 已解析搜索：‘{target}’，每 {interval} 秒采样一次")
        else:
            self._append_system(
                f"🔍 已解析搜索：‘{target}’"
                f"（使用当前间隔：{self.search_interval.value()} 秒）"
            )
        self._start_visual_search()
        return True

    def _handle_seek_command(self, text: str) -> bool:
        """Handle direct seek commands without going through LLM."""
        text_lower = text.lower()
        
        if text_lower.startswith(('seek ', 'go to ', 'jump to ')):
            parts = text.split()
            if len(parts) >= 2:
                time_str = parts[-1]
                try:
                    if ':' in time_str:
                        minutes, seconds = map(int, time_str.split(':'))
                        seconds = minutes * 60 + seconds
                    else:
                        seconds = float(time_str)
                    
                    return self._seek_to_timestamp(seconds)
                except ValueError:
                    pass
        
        return False
  
    def _stop_generation(self):
        """Stop the current LLM generation or visual search.
        
        Note: For GGUF vision models, the image encoding/decoding step is a
        single blocking C call (~20s) that cannot be interrupted from Python.
        Stop takes effect after the current frame finishes processing.
        """
        stopped_something = False
        if self._llm_thread_running():
            if self._worker:
                self._worker.cancel()
            stopped_something = True
        if self._search_thread_running():
            if self._search_worker:
                self._search_worker.cancel()
            self.search_btn.setEnabled(True)
            self.stop_search_btn.setEnabled(False)
            self.search_progress.setText("当前帧结束后停止…")
            stopped_something = True
        if stopped_something:
            self.stop_btn.setEnabled(False)
            self._append_system(
                "⏹ 已请求停止——将在当前帧处理完成后结束。\n"
                "   （GGUF 图像解码单帧可能需要约 20 秒，处理中无法强制中断）"
            )

    @Slot(str)
    def _on_token(self, token: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    @Slot(str)
    def _on_response_done(self, full_text: str):
        self._chat_history.append({"role": "assistant", "content": full_text})

        if self._timeline_bridge and self._timeline_bridge.is_connected:
            try:
                from .llm_timeline_bridge import parse_commands
                commands = parse_commands(full_text)
                if commands:
                    cursor = self.chat_display.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    cursor.insertText("\n")
                    self.chat_display.setTextCursor(cursor)

                    _, results = self._timeline_bridge.process_response(full_text)
                    for result in results:
                        self._append_html(
                            f'<div style="color:#FFD700;margin:2px 0;font-style:italic;">'
                            f'{result}</div>'
                        )
            except Exception as e:
                self._append_html(
                    f'<div style="color:#ff9800;">命令错误：{e}</div>'
                )

        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.chat_display.setTextCursor(cursor)

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.stop_btn.setEnabled(False)
        self.input_field.setFocus()

    @Slot(str)
    def _on_response_error(self, error_msg: str):
        self._append_html(
            f'<div style="color:#f44336;margin-left:12px;">错误：{error_msg}</div><br>'
        )
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("发送")
        self.stop_btn.setEnabled(False)

    def _clear_chat(self):
        self.chat_display.clear()
        self._chat_history.clear()
        if self._llm and self._llm.is_loaded():
            self._append_system("对话已清空，可以开始新的问题。")

    # ------------------------------------------------ Display helpers

    def _append_html(self, html: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        # Start each fragment in its own block. Otherwise Qt's insertHtml()
        # merges consecutive block elements into the trailing block, so status
        # messages render as one run-together wall of text. length() == 1 means
        # the current block is empty, so we don't add a leading blank line.
        if cursor.block().length() > 1:
            cursor.insertBlock()
        cursor.insertHtml(html)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_user(self, text: str):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self._append_html(
            f'<div style="color:#2f81f7;margin-top:8px;"><b>你：</b></div>'
            f'<div style="color:#ccc;margin-left:12px;">{safe}</div><br>'
        )

    def _append_system(self, text: str):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self._append_html(
            f'<div style="color:#aaa;font-style:italic;margin:4px 0;">{safe}</div>'
        )

    def seed_from_report(self, json_path: str):
        """Put a highlight report's findings in front of the chat.

        The findings are shown, not sent: the user reads what was actually
        measured and then asks their own question, rather than a canned one
        being answered before they have said what they care about.

        The findings are also kept as pending context, so the next message
        carries them — otherwise the model would be asked "why?" about a run it
        has never been told anything about.
        """
        import json as _json

        from modules.report.advisor import build_prompt, format_findings
        from modules.report.highlight_advice import diagnose

        with open(json_path, encoding="utf-8") as fh:
            report = _json.load(fh)

        findings = diagnose(report)
        self._advisor_context = build_prompt(report, findings, question=" ")
        self._append_system(
            f"已加载 {os.path.basename(json_path)} —— "
            f"本次运行发现 {len(findings)} 项问题：")
        self._append_system(format_findings(findings))
        self._append_system(
            "你可以询问这次剪辑的任何问题。回答将依据这些分析结果和"
            "顾问文档，而不是直接读取视频本身。")
        if hasattr(self, "input_field"):
            self.input_field.setPlaceholderText(
                "例如：为什么每个片段看起来都很相似？")
            self.input_field.setFocus()

    def _log(self, msg: str):
        self.status_label.setText(msg)

    def closeEvent(self, event):
        """Clean up resources on close.

        Wait for running threads to finish — destroying a QThread while
        its run-loop is still active triggers qFatal. Cancellation sets a
        flag; uninterruptible C calls (GGUF image decode) need to finish
        naturally before the worker can emit finished.
        """
        # LLM worker
        if self._llm_thread_running():
            self._worker.cancel()
            if not self._worker_thread.wait(5000):
                print("⚠️ 大模型线程未能及时停止，正在强制终止")
                self._worker_thread.terminate()
                self._worker_thread.wait(2000)

        # Visual search worker
        if self._search_thread_running():
            self._search_worker.cancel()
            if not self._search_worker_thread.wait(5000):
                print("⚠️ 搜索线程未能及时停止，正在强制终止")
                self._search_worker_thread.terminate()
                self._search_worker_thread.wait(2000)

        if self._analyzer:
            self._analyzer.close()
        super().closeEvent(event)

# ---------------------------------------------------------------------------
# Standalone window for testing
# ---------------------------------------------------------------------------
class LLMChatWindow(QWidget):
    def __init__(self, cache_dir: str = "./cache", video_path: str = ""):
        super().__init__()
        self.setWindowTitle("VideoHighlighter - 大模型对话与视觉搜索")
        self.setMinimumSize(700, 600)
        layout = QVBoxLayout()
        self.chat = LLMChatWidget(parent=self, cache_dir=cache_dir, video_path=video_path)
        layout.addWidget(self.chat)
        self.setLayout(layout)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="大模型对话与视觉搜索")
    parser.add_argument("--video", type=str, help="视频文件路径")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="缓存目录")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    win = LLMChatWindow(cache_dir=args.cache_dir, video_path=args.video or "")
    win.show()
    sys.exit(app.exec())