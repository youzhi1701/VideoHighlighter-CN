"""
llm_module.py — Local LLM interface for VideoHighlighter.

Supports two backends:
  1. ollama   — requires `ollama` running locally (easiest setup)
  2. llama-cpp — requires `llama-cpp-python` + a GGUF model file

The module builds rich context from your video analysis cache so the LLM
can reason about detected objects, actions, transcript, scores, etc.

Usage:
    from llm_module import LLMModule, VideoSeekAnalyzer

    llm = LLMModule(backend="ollama", model="llama3.2")
    llm.load()
    
    # Analyze video every 1 second
    analyzer = VideoSeekAnalyzer("video.mp4", llm)
    results = analyzer.analyze_every_1_second()
"""

from __future__ import annotations

import base64
import math
import json
import os
import re
import time
import threading
from typing import Optional, Callable

# Try to import OpenCV for video analysis (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ 未安装 OpenCV，VideoSeekAnalyzer 无法工作。请运行：pip install opencv-python")

# Which Ollama server to talk to is decided in one place, so the chat panel, the
# report and the advisor cannot end up pointed at two different machines. The
# fallback is for running this file directly, which the header still documents.
try:
    from llm.ollama_host import resolve as resolve_ollama_host
except ImportError:  # pragma: no cover - standalone/dev fallback
    from ollama_host import resolve as resolve_ollama_host


# ---------------------------------------------------------------------------
# Cancellation token — allows external code to abort generation mid-stream
# ---------------------------------------------------------------------------
class CancellationToken:
    """Thread-safe cancellation flag that backends check during generation."""
    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def reset(self):
        self._cancelled.clear()


class GenerationCancelled(Exception):
    """Raised when generation is cancelled via CancellationToken."""
    pass


# ---------------------------------------------------------------------------
# Response sanitizer — strips leaked role tokens and self-conversation
# ---------------------------------------------------------------------------
# Patterns that indicate the model started role-playing a conversation
_SELF_TALK_PATTERNS = re.compile(
    r'(?:\n|^)\s*(?:'
    r'(?:USER|User|user|HUMAN|Human|human)\s*:\s*'   # "USER:" turn markers
    r'|(?:ASSISTANT|Assistant|assistant)\s*:\s*'       # "ASSISTANT:" markers
    r'|⏹\s*Stopping generation'                       # leaked stop tokens
    r'|\[SYSTEM INSTRUCTIONS\]'                        # leaked system wrapping
    r'|\[END INSTRUCTIONS\]'
    r'|--- VIDEO ANALYSIS DATA ---'                    # new context markers
    r'|--- TIMELINE CONTROL ---'
    r'|--- END ---'
    r'|```python'
    r'|\[CMD:\w+[^\]]*\]'                              # ANY [CMD:...] block
    r'|#{1,3}\s+Additional\s+analysis'                 # self-generated headers
    r'|#{1,3}\s+Further\s+'
    r'|#{1,3}\s+Next\s+steps'
    r'|(?:Clip\s*#\d+.*\n){3,}'           # hallucinated clip lists
    r')',
    re.IGNORECASE
)

# Stop sequences to prevent generation past the assistant's turn
STOP_SEQUENCES = [
    "\nUSER:", "\nUser:", "\nuser:",
    "\nHUMAN:", "\nHuman:", "\nhuman:",
    "\nASSISTANT:", "\nAssistant:",
    "\n## ", "\n===",  # Don't regenerate context markers
    "\n---",           # Don't regenerate context separators
    "[CMD:",           # Don't generate commands
    "```",             # Don't generate code blocks
    "⏹",
]

def sanitize_response(text: str) -> str:
    """
    Clean up LLM output: strip self-conversation, leaked markers, and 
    truncate at the first sign of role-playing.
    """
    if not text:
        return text
    
    # Find the first occurrence of a self-talk pattern
    match = _SELF_TALK_PATTERNS.search(text)
    if match:
        # Truncate everything from the first leaked marker onward
        text = text[:match.start()].rstrip()
    
    # Also strip any trailing partial role markers that didn't fully match
    # e.g. the model outputting "USER" right at the end
    for marker in ["USER", "User", "ASSISTANT", "Assistant", "HUMAN", "Human"]:
        if text.rstrip().endswith(marker):
            text = text[:text.rfind(marker)].rstrip()
    
    # Strip trailing whitespace and dangling punctuation
    text = text.rstrip()
    
    return text


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------
class _LLMBackend:
    """Abstract backend."""

    def load(self, **kwargs):
        raise NotImplementedError

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                 images: list[str] | None = None,
                 cancellation_token: Optional[CancellationToken] = None) -> str:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return False

    def unload(self):
        pass

    @staticmethod
    def available() -> bool:
        return False


# ---------------------------------------------------------------------------
# Helper: check cancellation inside any streaming loop
# ---------------------------------------------------------------------------
def _check_cancel(cancel_token: Optional[CancellationToken]):
    """Raise GenerationCancelled if the token is set."""
    if cancel_token and cancel_token.is_cancelled:
        raise GenerationCancelled("Generation cancelled by user")


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
# What to give a model that reasons before it answers: no cap at all.
#
# Ollama reads a negative `num_predict` as "generate until the model stops", so
# the reasoning ends where the model ends it rather than where a budget cut it
# off. That is deliberate. The reasoning is not overhead to be minimised - a
# narrator's job is to connect what the run measured to what the frames show,
# and a model that works that through before writing is doing the thing that
# was wanted. Capping it buys a faster run by making it dumber.
#
# The cost is real and is the user's to accept: measured against the clip brief
# on `qwen3-vl:8b`, reasoning runs into thousands of tokens for a two-sentence
# paragraph, so a run on such a model takes minutes per clip.
THINKING_BUDGET = -1


def _unreachable_hint(base_url: str) -> str:
    """What to try next when nothing answered at ``base_url``.

    Two different pieces of advice, because the usual cause differs: locally the
    server is simply not started, while a remote one is nearly always running
    and bound to its own localhost - the default - where the network cannot see
    it. Telling a user to start a server they can watch running is the kind of
    help that makes them stop reading the message.
    """
    try:
        from llm.ollama_host import is_remote
    except ImportError:  # pragma: no cover - standalone/dev fallback
        from ollama_host import is_remote
    if is_remote(base_url):
        return ("On that machine, start it as OLLAMA_HOST=0.0.0.0 ollama serve "
                "and let port 11434 through its firewall.")
    return "Start with: ollama serve"


class _OllamaBackend(_LLMBackend):
    """Talks to an Ollama server - localhost by default, wherever it was set.

    ``base_url=None`` means "ask :mod:`llm.ollama_host`", which is how a server
    on another machine reaches every caller at once instead of one at a time.
    """

    def __init__(self, model: str = "llama3.2", base_url: Optional[str] = None):
        self.model = model
        self.base_url = resolve_ollama_host(base_url)
        self._loaded = False
        # Whether this model reasons before it answers. Read from the server at
        # load() where the server will say, and otherwise discovered by the
        # first call that comes back as reasoning only.
        self._thinks = False

    @staticmethod
    def available() -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, **kwargs):
        """Verify the Ollama server is reachable and the model exists."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            matched = any(self.model in m for m in models)
            if not matched:
                raise RuntimeError(
                    f"Ollama 中未找到模型 '{self.model}'。"
                    f"可用模型：{models}\n"
                    f"请运行：ollama pull {self.model}"
                )
            self._loaded = True

            # Ollama lists `thinking` among a model's capabilities, so the
            # budget can be right on the first call instead of after one spent
            # discovering it. That wasted call is not free: a reasoning-only
            # reply costs its whole budget, and on a CPU-bound server that is
            # over a minute of a run buying nothing but a log line.
            #
            # Best-effort: an older server, a proxy, or a model without the
            # field leaves the flag alone and the retry below covers it. Only
            # the asking is guarded - wrapping the logging too would report a
            # console that cannot encode the message as a failure to read the
            # capability, which is a true sentence about the wrong thing.
            capabilities = None
            try:
                shown = requests.post(f"{self.base_url}/api/show",
                                      json={"model": self.model}, timeout=5)
                if shown.ok:
                    capabilities = shown.json().get("capabilities") or []
            except Exception as exc:
                print(f"   （无法读取 {self.model} 的能力信息：{exc}）")

            if capabilities is not None:
                self._thinks = "thinking" in capabilities
                if self._thinks:
                    print(f"[思考模型] '{self.model}' 会先进行推理再回答；"
                          f"当前不限制输出预算，因此耗时会更长。")
        except requests.ConnectionError:
            # A remote server that refuses the connection has almost always
            # been started bound to localhost, which is the default and is
            # invisible from any other machine. "Is it running?" sends the
            # user there to check the one thing that is already true.
            raise RuntimeError(
                "无法连接到 Ollama："
                f"{self.base_url}。请检查服务是否正在运行。\n" + _unreachable_hint(self.base_url)
            )

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                images: list[str] | None = None,
                cancellation_token: Optional[CancellationToken] = None) -> str:
        import requests
        import json as _json
        import time as _time

        # Discovered once, applied for the rest of the run: a model that
        # reasons gets the uncapped budget on every later call, instead of
        # every clip paying a wasted call to find out again.
        if self._thinks:
            max_tokens = THINKING_BUDGET

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": True,  # Always stream internally so we can cancel + capture timing
            # Reasoning tags think by default. Ollama ignores this on models
            # that cannot think - and, measured on `qwen3-vl:8b`, on some that
            # can, which is what the retry below exists for.
            "think": False,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "repeat_last_n": 128,
                "stop": STOP_SEQUENCES,
            },
        }
        if images:
            payload["images"] = images

        # ── Phase 1: serialize JSON body ──
        t0 = _time.perf_counter()
        body = _json.dumps(payload).encode("utf-8")
        t_serialize_ms = (_time.perf_counter() - t0) * 1000
        body_kb = len(body) / 1024

        # ── Phase 2: send request and stream response ──
        full_text = []
        thought_text = []
        final_chunk = None
        t_send_start = _time.perf_counter()
        t_headers_received = None
        t_first_byte = None

        with requests.post(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            stream=True,
            timeout=120,
        ) as resp:
            t_headers_received = _time.perf_counter()
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                if t_first_byte is None:
                    t_first_byte = _time.perf_counter()
                _check_cancel(cancellation_token)
                chunk = _json.loads(line)
                # A thinking model streams its reasoning in a separate field
                # and leaves `response` empty until it is done. Collected only
                # to tell "spent the budget thinking" apart from "had nothing
                # to say" below; it is never part of the answer.
                thinking = chunk.get("thinking", "")
                if thinking:
                    thought_text.append(thinking)
                token = chunk.get("response", "")
                if token:
                    full_text.append(token)
                    current_text = "".join(full_text)
                    if _SELF_TALK_PATTERNS.search(current_text):
                        break
                    if stream_callback:
                        stream_callback(token)
                if chunk.get("done", False):
                    final_chunk = chunk
                    break

        t_end = _time.perf_counter()

        # ── Phase 3: report timing ──
        if final_chunk:
            headers_ms = ((t_headers_received - t_send_start) * 1000
                        if t_headers_received else 0)
            ttfb_ms    = ((t_first_byte - t_send_start) * 1000
                        if t_first_byte else 0)
            wall_ms    = (t_end - t_send_start) * 1000

            load_ms = final_chunk.get("load_duration", 0)         / 1e6
            pe_ms   = final_chunk.get("prompt_eval_duration", 0)  / 1e6
            ev_ms   = final_chunk.get("eval_duration", 0)         / 1e6
            total_ms = final_chunk.get("total_duration", 0)       / 1e6
            pe_n = final_chunk.get("prompt_eval_count", 0)
            ev_n = final_chunk.get("eval_count", 0)
            pe_rate = (pe_n / (pe_ms / 1000)) if pe_ms > 0 else 0
            ev_rate = (ev_n / (ev_ms / 1000)) if ev_ms > 0 else 0

            # If ttfb > total, Ollama did unreported work (likely image preprocessing)
            server_unaccounted_ms = ttfb_ms - total_ms

            print(
                f"   ⏱  请求体={body_kb:5.0f}KB  "
                f"序列化={t_serialize_ms:5.0f}ms  "
                f"收到响应头={headers_ms:5.0f}ms  "
                f"ttfb={ttfb_ms:5.0f}ms  "
                f"总耗时={wall_ms:5.0f}ms"
            )
            print(
                f"      Ollama：加载={load_ms:4.0f}ms  "
                f"提示词计算={pe_ms:5.0f}ms（{pe_n}t @ {pe_rate:5.1f}t/s）  "
                f"生成={ev_ms:4.0f}ms（{ev_n}t @ {ev_rate:5.1f}t/s）  "
                f"总计={total_ms:5.0f}ms"
            )
            print(
                f"      服务端未计入耗时（ttfb − total）={server_unaccounted_ms:5.0f}ms  "
                f"← 若明显大于 0，通常来自图像预处理"
            )

        raw = "".join(full_text)

        # `think: False` is a request, not a guarantee. Measured on
        # `qwen3-vl:8b` against Ollama 0.17.1: the flag is ignored, `/no_think`
        # in the prompt and in the system message are both ignored, and the
        # model reasons for thousands of tokens before it begins to answer.
        # Under a narration-sized budget `response` is therefore empty on every
        # call - which reaches the caller as a clip that simply had no
        # paragraph, and a whole run of them as a report with nothing in it to
        # summarise, after paying for every call.
        if not raw.strip() and thought_text:
            thought_n = len("".join(thought_text))

            if max_tokens != THINKING_BUDGET:
                self._thinks = True
                print(f"[思考模型] '{self.model}' 已将 {max_tokens} 个 token 全部用于推理；"
                      f"正在取消上限后重试，本轮后续调用也将保持不限额。")
                # Terminates: the retry is made with THINKING_BUDGET, so this
                # branch cannot be taken twice for the same call.
                return self.generate(
                    prompt, system=system, max_tokens=THINKING_BUDGET,
                    temperature=temperature,
                    stream_callback=stream_callback, images=images,
                    cancellation_token=cancellation_token)

            raise RuntimeError(
                f"'{self.model}' 在未限制预算的情况下推理了 {thought_n} 个字符，但没有给出答案；"
                f"这说明耗尽的是上下文而不是输出预算。请缩短要求，让模型为最终回答保留空间。")

        return sanitize_response(raw)

    def unload(self):
        self._loaded = False


# ---------------------------------------------------------------------------
# OpenVINO backend (Intel GPU / NPU / CPU, in-process)
# ---------------------------------------------------------------------------
# Why this exists beside Ollama, measured on an Arc A750 over the same frames
# and the same brief: 41 tok/s against 4.9 with a picture in the context, and
# an image encode of about a second against forty-six. The encode is the
# structural half — the vision tower is an OpenVINO graph and runs on the GPU,
# where llama.cpp falls back to the CPU for it whichever backend is chosen.
# See `docs/INTEL-GPU.md`.
#
# The cost is that models arrive as OpenVINO IR rather than GGUF, so there is
# no `pull`: the path below is a directory the user had to obtain or convert.
# That is a setup step this backend cannot hide, only explain.

# A thinking model's reasoning comes back inline here, where Ollama puts it in
# a field of its own. It is terminated by this and — because the chat template
# opens the block in the prompt — is NOT preceded by an opening tag, so the
# obvious `<think>.*?</think>` strip matches nothing and would publish the
# model's private deliberation as the description. Split on the last close.
THINK_CLOSE = "</think>"


class _OpenVINOBackend(_LLMBackend):
    """Runs a vision-language model in-process through OpenVINO GenAI."""

    def __init__(self, model_path: str, device: str = "GPU"):
        self.model_path = model_path
        self.device = device
        self._pipe = None
        # Read off the chat template at load, rather than discovered by a
        # wasted call: a build whose template carries the think tags reasons
        # before it answers, and one whose template does not, does not.
        self._thinks = False

    @staticmethod
    def available() -> bool:
        try:
            import openvino_genai  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def devices() -> list:
        """Which OpenVINO devices this machine offers, e.g. ['CPU', 'GPU']."""
        try:
            import openvino as ov
            return list(ov.Core().available_devices)
        except Exception:
            return []

    def load(self, **kwargs):
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(
                f"在 {self.model_path} 未找到 OpenVINO 模型目录。"
                f"此后端需要已转换的 OpenVINO IR 模型，不支持 GGUF 或 Ollama 标签；"
                f"请参阅 docs/INTEL-GPU.md。")
        import openvino_genai as ov_genai

        template = os.path.join(self.model_path, "chat_template.jinja")
        try:
            with open(template, encoding="utf-8") as fh:
                self._thinks = THINK_CLOSE in fh.read()
        except OSError:
            self._thinks = False

        print(f"[OpenVINO] 正在将 {os.path.basename(self.model_path)} 加载到 {self.device}"
              + ("（回答前会先推理）" if self._thinks else ""))
        self._pipe = ov_genai.VLMPipeline(self.model_path, self.device)

    def is_loaded(self) -> bool:
        return self._pipe is not None

    def accepts_images(self) -> bool:
        return True

    def unload(self):
        self._pipe = None

    def _tensors(self, images):
        """base64 strings, as the rest of this module passes them, to Tensors.

        Returns before importing anything when there are no frames. Pillow and
        numpy are only needed to turn a picture into a tensor, so a text-only
        call must not require them to be installed - CI has neither, and an
        unconditional import there failed ten tests that never passed an image.
        """
        frames = [raw for raw in (images or []) if raw]
        if not frames:
            return []

        import base64 as _b64
        import io as _io

        import numpy as _np
        import openvino as ov
        from PIL import Image

        out = []
        for raw in frames:
            img = Image.open(_io.BytesIO(_b64.b64decode(raw))).convert("RGB")
            out.append(ov.Tensor(_np.array(img)[None]))
        return out

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                 images: list[str] | None = None,
                 cancellation_token: Optional[CancellationToken] = None,
                 seed: Optional[int] = None) -> str:
        if self._pipe is None:
            raise RuntimeError("OpenVINO 后端尚未加载，请先调用 load()")
        import openvino_genai as ov_genai

        config = ov_genai.GenerationConfig()
        # Left at its default, which is unbounded, whenever the model reasons -
        # the same decision as the Ollama backend and for the same reason: a cap
        # is spent on the reasoning first and the answer never arrives.
        if not self._thinks and max_tokens and max_tokens > 0:
            config.max_new_tokens = int(max_tokens)
        config.do_sample = bool(temperature and temperature > 0)
        if config.do_sample:
            config.temperature = float(temperature)
        if seed is not None:
            config.rng_seed = int(seed)

        full = []
        answering = [not self._thinks]

        def _streamer(chunk):
            """Forward the answer only, and stop early when cancelled.

            The reasoning is deliberately not streamed. Ollama's `thinking`
            never reached a caller either, and a live pane filling with the
            model talking to itself about rule four is not what this is for.
            """
            full.append(chunk)
            if cancellation_token and cancellation_token.is_cancelled:
                return True
            if stream_callback:
                if answering[0]:
                    stream_callback(chunk)
                elif THINK_CLOSE in "".join(full):
                    answering[0] = True
                    tail = "".join(full).rsplit(THINK_CLOSE, 1)[-1]
                    if tail:
                        stream_callback(tail)
            return False

        tensors = self._tensors(images)
        kwargs = {"generation_config": config, "streamer": _streamer}
        if tensors:
            kwargs["images"] = tensors
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        self._pipe.generate(full_prompt, **kwargs)

        _check_cancel(cancellation_token)

        text = "".join(full)
        if THINK_CLOSE in text:
            answer = text.rsplit(THINK_CLOSE, 1)[-1].strip()
        elif self._thinks:
            # A model that reasons and never closed the block was still
            # reasoning when it stopped, so every character of this is
            # deliberation. Falling through to `text` here would publish it as
            # the description - the exact failure this split exists to prevent.
            answer = ""
        else:
            answer = text.strip()

        if not answer and self._thinks and text.strip():
            raise RuntimeError(
                f"'{os.path.basename(self.model_path)}' 推理了 {len(text)} 个字符但没有给出答案。"
                f"请缩短要求，为最终回答保留足够上下文空间。")

        return sanitize_response(answer)


# ---------------------------------------------------------------------------
# llama-cpp-python backend with vision support
# ---------------------------------------------------------------------------
class _LlamaCppBackend(_LLMBackend):
    """Uses llama-cpp-python to run a GGUF model directly (no server)."""

    def __init__(self, model_path: str, mmproj_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        self._chat_handler = None

    @staticmethod
    def available() -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, **kwargs):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"未找到 GGUF 模型：{self.model_path}")
        
        from llama_cpp import Llama
        
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            print(f"📷 正在加载视觉模型，mmproj：{self.mmproj_path}")
            
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                
                self._chat_handler = Llava15ChatHandler(
                    clip_model_path=self.mmproj_path,
                    verbose=False
                )
                
                print("✅ 已创建 Llava15ChatHandler 以支持视觉输入")
                
                self._model = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                    n_threads=None,
                )
            except ImportError:
                print("⚠️ Llava15ChatHandler 不可用，正在回退到基础视觉模式")
                self._model = Llama(
                    model_path=self.model_path,
                    clip_model_path=self.mmproj_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                    n_threads=None,
                )
        else:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_threads=None,
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                images: list[str] | None = None,
                cancellation_token: Optional[CancellationToken] = None) -> str:
        if not self._model:
            raise RuntimeError("模型尚未加载，请先调用 load()")

        if images and self.mmproj_path:
            return self._generate_vision(prompt, system, images, max_tokens, 
                                        temperature, stream_callback,
                                        cancellation_token)
        else:
            return self._generate_text(prompt, system, max_tokens, 
                                      temperature, stream_callback,
                                      cancellation_token)

    def _generate_text(self, prompt: str, system: str = "", max_tokens: int = 1024,
                      temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                      cancellation_token: Optional[CancellationToken] = None) -> str:
        """Handle text-only generation with anti-hallucination measures."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        gen_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": 1.3,
            "stop": STOP_SEQUENCES,
        }

        # ALWAYS use streaming for llama-cpp so we can check cancellation
        full_text = []
        for chunk in self._model.create_chat_completion(**gen_kwargs, stream=True):
            _check_cancel(cancellation_token)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_text.append(token)
                joined = "".join(full_text)
                if _SELF_TALK_PATTERNS.search(joined):
                    break
                if stream_callback:
                    stream_callback(token)
        raw = "".join(full_text)
        return sanitize_response(raw)

    def _generate_vision(self, prompt: str, system: str = "", images: list[str] = None,
                        max_tokens: int = 1024, temperature: float = 0.7,
                        stream_callback: Optional[Callable] = None,
                        cancellation_token: Optional[CancellationToken] = None) -> str:
        """Handle vision generation with proper chat handler.
        
        Always streams internally so cancellation_token can interrupt
        even when no stream_callback is provided.
        """
        try:
            image_urls = []
            for img_b64 in images:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                image_urls.append(f"data:image/jpeg;base64,{img_b64}")
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            
            user_content = []
            user_content.append({"type": "text", "text": prompt})
            
            for img_url in image_urls:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": img_url}
                })
            
            messages.append({"role": "user", "content": user_content})
            
            print(f"📤 正在发送视觉请求，共 {len(images)} 张图片")

            gen_kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "stop": STOP_SEQUENCES,
            }

            # ALWAYS stream internally so we can check cancellation_token
            # between tokens. Previously the non-streaming path was a single
            # blocking C call with no way to interrupt.
            full_text = []
            for chunk in self._model.create_chat_completion(**gen_kwargs, stream=True):
                _check_cancel(cancellation_token)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_text.append(token)
                        joined = "".join(full_text)
                        if _SELF_TALK_PATTERNS.search(joined):
                            break
                        if stream_callback:
                            stream_callback(token)
            raw = "".join(full_text)
            result = sanitize_response(raw)
            print(f"✅ 视觉生成完成：{len(result)} 个字符")
            return result
                    
        except GenerationCancelled:
            # Re-raise so callers see the cancellation
            raw = "".join(full_text) if 'full_text' in dir() else ""
            return sanitize_response(raw)
        except Exception as e:
            print(f"❌ 视觉生成出错：{e}")
            import traceback
            traceback.print_exc()
            
            print("⚠️ 正在回退到原始提示词格式…")
            return self._generate_vision_fallback(prompt, system, images, max_tokens, 
                                                  temperature, stream_callback,
                                                  cancellation_token)

    def _generate_vision_fallback(self, prompt: str, system: str = "", images: list[str] = None,
                                 max_tokens: int = 1024, temperature: float = 0.7,
                                 stream_callback: Optional[Callable] = None,
                                 cancellation_token: Optional[CancellationToken] = None) -> str:
        """Fallback method using raw prompt formatting (LLaVA style)."""
        try:
            prompt_parts = []
            
            if system:
                prompt_parts.append(system)
                prompt_parts.append("")
            
            for _ in images:
                prompt_parts.append("<image>")
            
            prompt_parts.append("")
            prompt_parts.append(f"Question: {prompt}")
            prompt_parts.append("Answer:")
            
            full_prompt = "\n".join(prompt_parts)
            
            print(f"📤 正在使用备用提示词格式，共 {len(images)} 张图片")
            
            image_bytes = []
            for img_b64 in images:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                image_bytes.append(base64.b64decode(img_b64))
            
            gen_kwargs = {
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "image_data": image_bytes,
                "repeat_penalty": 1.3,
                "stop": STOP_SEQUENCES + ["\nQuestion:", "\nQ:"],
            }

            # Always stream for cancellation support
            full_text = []
            for chunk in self._model.create_completion(**gen_kwargs, stream=True):
                _check_cancel(cancellation_token)
                token = chunk["choices"][0].get("text", "")
                if token:
                    full_text.append(token)
                    joined = "".join(full_text)
                    if _SELF_TALK_PATTERNS.search(joined):
                        break
                    if stream_callback:
                        stream_callback(token)
            raw = "".join(full_text)
            return sanitize_response(raw)
                
        except GenerationCancelled:
            raw = "".join(full_text) if 'full_text' in dir() else ""
            return sanitize_response(raw)
        except Exception as e:
            print(f"❌ 备用方案同样失败：{e}")
            return f"处理图片时出错：{e}"

    def unload(self):
        self._model = None
        self._chat_handler = None

# ---------------------------------------------------------------------------
# Context builder — turns analysis cache into LLM-readable text
# ---------------------------------------------------------------------------
class VideoContextBuilder:
    """
    Converts a VideoHighlighter analysis cache dict into a concise text
    summary that fits in the LLM context window.
    """

    @staticmethod
    def build(analysis_data: dict, video_path: str = "", max_items: int = 30) -> str:
        """
        Build a context string from analysis data.

        Args:
            analysis_data: The cache dict (same format as collect_analysis_data output)
            video_path: Optional video filename for reference
            max_items: Cap per section to avoid blowing up context

        Returns:
            A formatted context string
        """
        parts = []

        # --- Video metadata ---
        meta = analysis_data.get("video_metadata", {})
        duration = meta.get("duration", 0)
        fps = meta.get("fps", 0)
        if video_path:
            parts.append(f"## Video: {os.path.basename(video_path)}")
        parts.append(f"Duration: {int(duration)}s ({int(duration)//60}m{int(duration)%60:02d}s), FPS: {fps:.1f}")
        parts.append("")

        # --- Detected objects ---
        objects_raw = analysis_data.get("objects", [])
        if objects_raw:
            obj_counts: dict[str, int] = {}
            obj_timestamps: dict[str, list] = {}
            for entry in objects_raw:
                ts = entry.get("timestamp", 0)
                for obj_name in entry.get("objects", []):
                    obj_counts[obj_name] = obj_counts.get(obj_name, 0) + 1
                    obj_timestamps.setdefault(obj_name, []).append(ts)

            parts.append(f"## Detected Objects ({len(objects_raw)} seconds with detections)")
            for obj, count in sorted(obj_counts.items(), key=lambda x: -x[1]):
                timestamps = obj_timestamps[obj]
                sample_ts = timestamps[:5]
                ts_str = ", ".join(f"{t//60}:{t%60:02d}" for t in sample_ts)
                if len(timestamps) > 5:
                    ts_str += f" ... (+{len(timestamps)-5} more)"
                parts.append(f"  - {obj}: {count} detections (at {ts_str})")
            parts.append("")

        # --- Detected actions ---
        actions_raw = analysis_data.get("actions", [])
        if actions_raw:
            action_groups: dict[str, list] = {}
            for act in actions_raw:
                name = act.get("action_name", "unknown")
                action_groups.setdefault(name, []).append(act)

            parts.append(f"## Detected Actions ({len(actions_raw)} total detections)")
            for name, detections in sorted(action_groups.items(), key=lambda x: -len(x[1])):
                confidences = [d.get("confidence", 0) for d in detections]
                timestamps = [d.get("timestamp", 0) for d in detections]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                max_conf = max(confidences) if confidences else 0
                sample_ts = sorted(timestamps)[:5]
                ts_str = ", ".join(f"{int(t)//60}:{int(t)%60:02d}" for t in sample_ts)

                parts.append(
                    f"  - {name}: {len(detections)} detections, "
                    f"avg conf={avg_conf:.2f}, max conf={max_conf:.2f} "
                    f"(at {ts_str}{'...' if len(timestamps) > 5 else ''})"
                )
            parts.append("")

        # --- Scenes ---
        scenes = analysis_data.get("scenes", [])
        if scenes:
            parts.append(f"## Scene Changes ({len(scenes)})")
            for sc in scenes[:max_items]:
                s, e = sc.get("start", 0), sc.get("end", 0)
                parts.append(f"  - {int(s)//60}:{int(s)%60:02d} → {int(e)//60}:{int(e)%60:02d}")
            if len(scenes) > max_items:
                parts.append(f"  ... and {len(scenes) - max_items} more")
            parts.append("")

        # --- Motion events/peaks ---
        motion_events = analysis_data.get("motion_events", [])
        motion_peaks = analysis_data.get("motion_peaks", [])
        if motion_events or motion_peaks:
            parts.append(f"## Motion: {len(motion_events)} events, {len(motion_peaks)} peaks")
            if motion_peaks:
                sample = sorted(motion_peaks)[:10]
                parts.append(f"  Peak timestamps: {', '.join(f'{int(t)//60}:{int(t)%60:02d}' for t in sample)}")
            parts.append("")

        # --- Audio peaks ---
        audio_data = analysis_data.get("audio", {})
        audio_peaks = audio_data.get("peaks", []) if isinstance(audio_data, dict) else analysis_data.get("audio_peaks", [])
        if audio_peaks:
            parts.append(f"## Audio Peaks ({len(audio_peaks)})")
            sample = sorted(audio_peaks)[:10]
            parts.append(f"  Timestamps: {', '.join(f'{int(t)//60}:{int(t)%60:02d}' for t in sample)}")
            parts.append("")

        # --- Transcript snippets ---
        transcript = analysis_data.get("transcript", {})
        segments = transcript.get("segments", [])
        lang = transcript.get("language", "unknown")
        if segments:
            parts.append(f"## Transcript ({len(segments)} segments, language: {lang})")
            show_count = min(max_items, len(segments))
            for seg in segments[:show_count]:
                start = seg.get("start", 0)
                text = seg.get("text", "").strip()
                if text:
                    parts.append(f"  [{int(start)//60}:{int(start)%60:02d}] {text[:120]}")
            if len(segments) > show_count:
                parts.append(f"  ... ({len(segments) - show_count} more segments)")
            parts.append("")

        # --- Keyword matches ---
        keyword_matches = analysis_data.get("keyword_matches", [])
        if keyword_matches:
            parts.append(f"## Keyword Matches ({len(keyword_matches)})")
            for km in keyword_matches[:max_items]:
                kw = km.get("keyword", "?")
                seg = km.get("main_segment", {})
                start = seg.get("start", 0)
                text = seg.get("text", "")[:80]
                parts.append(f"  - '{kw}' at {int(start)//60}:{int(start)%60:02d}: \"{text}\"")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def build_action_learning_context(
        action_name: str,
        object_name: str,
        existing_labels: list[str],
        clip_analysis: list[dict] | None = None,
    ) -> str:
        """
        Build context specifically for the auto-learn pipeline (Step 5).
        """
        parts = [
            f"## Action Learning Task",
            f"Target action: '{action_name}'",
            f"Expected object: '{object_name}'",
            f"",
            f"The system is trying to learn to recognize '{action_name}' from video clips.",
            f"Currently known actions ({len(existing_labels)}): "
            + ", ".join(existing_labels[:20])
            + ("..." if len(existing_labels) > 20 else ""),
            "",
        ]

        if clip_analysis:
            parts.append("## Clip Analysis Results")
            for i, clip in enumerate(clip_analysis):
                parts.append(f"### Clip {i+1}")
                if "objects" in clip:
                    parts.append(f"  Objects detected: {', '.join(clip['objects'])}")
                if "actions" in clip:
                    for act in clip["actions"]:
                        parts.append(
                            f"  Action: {act.get('name', '?')} "
                            f"(confidence: {act.get('confidence', 0):.2f})"
                        )
                if "motion_level" in clip:
                    parts.append(f"  Motion level: {clip['motion_level']}")
                parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main LLM Module
# ---------------------------------------------------------------------------
class LLMModule:
    """
    High-level interface to a local LLM for VideoHighlighter.

    Examples:
        # Ollama (easiest)
        llm = LLMModule(backend="ollama", model="llama3.2")
        llm.load()
        answer = llm.query("Summarize the video", analysis_data=cache)

        # llama-cpp-python (no server)
        llm = LLMModule(backend="llama-cpp",
                         model_path="/models/llama-3.2-3b.Q4_K_M.gguf")
        llm.load()
    """

    SYSTEM_PROMPT = (
        "You are a video analysis assistant. You answer questions about video content "
        "using ONLY the analysis data provided below.\n\n"
        "RULES:\n"
        "1. Use ONLY the provided analysis data to answer. Do not guess or invent details.\n"
        "2. Quote specific timestamps from the data when relevant (e.g. 'at 1:23').\n"
        "3. If the data does not contain enough information, say so clearly.\n"
        "4. Give ONE concise answer. Do not repeat yourself.\n"
        "5. NEVER output commands, code, system messages, or status lines.\n"
        "6. NEVER pretend to scan, seek, or process the video yourself.\n"
        "7. STOP after answering. Do not add follow-up questions or next steps.\n"
    )

    SYSTEM_PROMPT_ACTION_LEARNING = (
        "You are an AI assistant helping to learn new action categories from video clips. "
        "Given detected objects, motion features, and existing action classifications, "
        "determine whether a video clip likely contains the target action. "
        "Respond with a confidence score (0.0 to 1.0) and brief reasoning. "
        "Format: SCORE: 0.XX\nREASON: ...\n"
        "Give ONE response and STOP. Do not generate any follow-up."
    )

    SYSTEM_PROMPT_TIMELINE = (
        "You are a timeline assistant. You help the user understand what is on their "
        "edit timeline using the provided data.\n\n"
        "RULES:\n"
        "1. Answer questions about the timeline using ONLY the provided data.\n"
        "2. Reference specific timestamps and clip numbers from the data.\n"
        "3. Give ONE concise answer.\n"
        "4. NEVER output commands, code blocks, or system messages.\n"
        "5. NEVER pretend to scan or analyze video.\n"
        "6. STOP after answering.\n"
    )

    SYSTEM_PROMPT_VISION = (
        "Describe what you see in the image.\n"
        "Focus on: people (count, poses, clothing, actions), objects, scene setting, "
        "colors, and lighting.\n"
        "If additional analysis data is provided, use it as context.\n"
        "Be specific and detailed in your description.\n\n"
        "RULES:\n"
        "1. Give ONE description and STOP.\n"
        "2. NEVER generate commands, code blocks, or system messages.\n"
        "3. NEVER generate follow-up questions or fake conversation.\n"
        "4. NEVER pretend to run searches, scans, or tools.\n"
        "5. STOP after your description. Do not add anything else.\n"
    )

    SYSTEM_PROMPT_VISUAL_SEARCH = (
        "The user will ask if something specific is present in the image.\n\n"
        "RULES:\n"
        "1. Start with YES or NO.\n"
        "2. Follow with ONE sentence explaining what you see.\n"
        "3. STOP after that single sentence.\n"
        "4. NEVER describe the full scene.\n"
        "5. NEVER generate commands, code, follow-up questions, or fake conversation.\n"
    )

    SYSTEM_PROMPT_GENERAL = (
    "You are a helpful assistant. Answer the user's questions directly and concisely.\n"
    "1. Give ONE answer and STOP.\n"
    "2. NEVER output commands, code blocks, or system messages.\n"
    "3. NEVER generate follow-up questions or fake conversation.\n"
    )


    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.2",
        model_path: str = "",
        mmproj_path: str = None,
        base_url: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        log_fn: Callable = print,
        device: str = "GPU",
    ):
        self.backend_name = backend
        self.log_fn = log_fn
        self._backend: _LLMBackend

        if backend == "ollama":
            self._backend = _OllamaBackend(model=model, base_url=base_url)
        elif backend == "llama-cpp":
            self._backend = _LlamaCppBackend(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        elif backend == "openvino":
            # `model_path` is a converted-model directory here, not a file, and
            # `device` is an OpenVINO device string rather than a layer count.
            self._backend = _OpenVINOBackend(model_path=model_path, device=device)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'llama-cpp'.")

    def load(self):
        """Load/verify the model. Raises RuntimeError on failure."""
        self.log_fn(f"🤖 正在加载大模型后端：{self.backend_name}")
        start = time.time()
        self._backend.load()
        elapsed = time.time() - start
        self.log_fn(f"✅ 大模型已就绪（{elapsed:.1f} 秒)")

    def is_loaded(self) -> bool:
        return self._backend.is_loaded()

    def unload(self):
        self._backend.unload()
        self.log_fn("🤖 大模型已卸载")

    def query(
        self,
        user_message: str,
        analysis_data: dict | None = None,
        free_chat_mode: bool = False,
        video_path: str = "",
        system_prompt: str | None = None,
        timeline_context: str = "",
        frame_base64: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream_callback: Optional[Callable[[str], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """
        Send a query to the LLM with optional video analysis context.
        
        Args:
            cancellation_token: Optional CancellationToken that, when cancelled,
                will interrupt generation even mid-token for GGUF vision models.
        """
        if not self._backend.is_loaded():
            raise RuntimeError("LLM 尚未加载，请先调用 load()")
        
        print("🔍 已调用 query()：")
        print(f"   frame_base64：{'有（' + str(len(frame_base64)) + ' 个字符）' if frame_base64 else '无'}")
        print(f"   timeline_context：{'有' if timeline_context else '无'}")
        print(f"   analysis_data：{'有' if analysis_data else '无'}")

        # ===== VISION MODE =====
        if frame_base64:
            prompt_parts = []
            
            if timeline_context:
                prompt_parts.append(
                    "--- TIMELINE CONTEXT ---\n"
                    f"{timeline_context}\n"
                    "--- END ---\n"
                )
            
            if analysis_data and not free_chat_mode:
                context = VideoContextBuilder.build(analysis_data, video_path)
                prompt_parts.append(
                    "--- VIDEO ANALYSIS DATA ---\n"
                    f"{context}\n"
                    "--- END ---\n"
                )
            
            prompt_parts.append(user_message)
            full_prompt = "\n".join(prompt_parts)
            
            system = system_prompt or (self.SYSTEM_PROMPT_GENERAL if free_chat_mode else self.SYSTEM_PROMPT_VISION)
            
            return self._backend.generate(
                prompt=full_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                stream_callback=stream_callback,
                images=[frame_base64],
                cancellation_token=cancellation_token,
            )
        
        # ===== TEXT MODE =====
        prompt_parts = []
        
        if analysis_data and not free_chat_mode:
            context = VideoContextBuilder.build(analysis_data, video_path)
            prompt_parts.append(
                "--- VIDEO ANALYSIS DATA ---\n"
                f"{context}\n"
                "--- END ---\n"
            )
        
        if timeline_context:
            prompt_parts.append(
                "--- TIMELINE CONTROL ---\n"
                f"{timeline_context}\n"
                "--- END ---\n"
            )
        
        prompt_parts.append(user_message)
        full_prompt = "\n".join(prompt_parts)
        
        if free_chat_mode:
            system = system_prompt or self.SYSTEM_PROMPT_GENERAL
        elif system_prompt:
            system = system_prompt
        elif timeline_context:
            system = self.SYSTEM_PROMPT_TIMELINE
        else:
            system = self.SYSTEM_PROMPT
        
        return self._backend.generate(
            prompt=full_prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_callback=stream_callback,
            cancellation_token=cancellation_token,
        )

    def verify_action_clip(
        self,
        action_name: str,
        object_name: str,
        clip_analysis: list[dict],
        existing_labels: list[str],
    ) -> tuple[float, str]:
        """
        Step 5 of auto-learn pipeline: ask LLM whether a clip contains the target action.
        """
        context = VideoContextBuilder.build_action_learning_context(
            action_name=action_name,
            object_name=object_name,
            existing_labels=existing_labels,
            clip_analysis=clip_analysis,
        )

        prompt = (
            f"{context}\n\n"
            f"Question: Based on the detected objects and action features above, "
            f"does this clip likely show the action '{action_name}'?\n"
            f"Respond with:\n"
            f"SCORE: <0.0 to 1.0>\n"
            f"REASON: <brief explanation>"
        )

        response = self._backend.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT_ACTION_LEARNING,
            max_tokens=256,
            temperature=0.3,
        )

        score = 0.5
        reason = response.strip()
        for line in response.strip().split("\n"):
            line_clean = line.strip().upper()
            if line_clean.startswith("SCORE:"):
                try:
                    score = float(line_clean.split(":", 1)[1].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line_clean.startswith("REASON:"):
                reason = line.strip().split(":", 1)[1].strip()

        return score, reason

    def query_async(
        self,
        user_message: str,
        callback: Callable[[str], None],
        error_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> threading.Thread:
        """Non-blocking query — runs in a background thread."""
        def _run():
            try:
                result = self.query(user_message, **kwargs)
                callback(result)
            except Exception as e:
                if error_callback:
                    error_callback(str(e))
                else:
                    self.log_fn(f"❌ 大模型查询出错：{e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# Convenience: check what's available
# ---------------------------------------------------------------------------
def get_available_backends() -> list[str]:
    """Return list of available backend names."""
    backends = []
    if _OllamaBackend.available():
        backends.append("ollama")
    if _LlamaCppBackend.available():
        backends.append("llama-cpp")
    return backends


def get_ollama_models(base_url: Optional[str] = None) -> list[str]:
    """Query Ollama for available models. Returns empty list on failure.

    ``None`` means the configured server (:mod:`llm.ollama_host`), which is
    localhost unless the user has pointed the app somewhere else.
    """
    try:
        import requests
        resp = requests.get(f"{resolve_ollama_host(base_url)}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Video Seek Analyzer
# ---------------------------------------------------------------------------
class VideoSeekAnalyzer:
    """
    Analyzes video frames at regular intervals using LLM vision.
    
    Requires OpenCV: pip install opencv-python
    
    Example:
        llm = LLMModule(backend="ollama", model="llava")
        llm.load()
        
        analyzer = VideoSeekAnalyzer("video.mp4", llm)
        results = analyzer.analyze_every_1_second()
        
        for r in results:
            print(f"[{r['timestamp_str']}] {r['analysis'][:100]}...")
        
        analyzer.close()
    """
    
    def __init__(self, video_path: str, llm: LLMModule, verbose: bool = False):
        self.verbose = verbose

        if not HAS_CV2:
            raise ImportError(
                "OpenCV (cv2) is required for VideoSeekAnalyzer. "
                "Install with: pip install opencv-python"
            )
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"未找到视频文件：{video_path}")
        
        self.video_path = video_path
        self.llm = llm
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_time = 0
        self.lock = threading.Lock()
        self.running = False
        self.analysis_cache = []
        
        print(f"📹 视频已加载：{os.path.basename(video_path)}")
        print(f"   时长：{int(self.duration)//60}分{int(self.duration)%60:02d}秒")
        print(f"   分辨率：{self.width}x{self.height}")
        print(f"   FPS：{self.fps:.2f}")
    
    def seek_to_time(self, timestamp_seconds: float):
        timestamp_seconds = max(0, min(timestamp_seconds, self.duration))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        self.cap.read()  # flush
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def frame_to_base64(self, frame, quality: int = 100) -> str:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_current_frame(
        self, 
        user_query: str = "What do you see in this frame? Describe the scene, people, objects, and actions.",
        custom_prompt: str = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> dict:
        frame = self.seek_to_time(self.current_time)
        if frame is None:
            return {
                "error": "Could not read frame",
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
        
        frame_b64 = self.frame_to_base64(frame)
        system = custom_prompt if custom_prompt else None
        
        try:
            response = self.llm.query(
                user_message=user_query,
                frame_base64=frame_b64,
                system_prompt=system,
                temperature=0.3,
                max_tokens=500,
                cancellation_token=cancellation_token,
            )
            
            return {
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}",
                "analysis": response,
                "frame_info": {
                    "width": self.width,
                    "height": self.height
                }
            }
        except GenerationCancelled:
            return {
                "cancelled": True,
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
    
    def analyze_every_n_seconds(self, interval: float = 1.0, callback=None, save_to_file=None,
                                cancellation_token: Optional[CancellationToken] = None):
        results = []
        num_analyses = int(self.duration / interval) + 1
        timestamps = [i * interval for i in range(num_analyses)]
        
        print(f"\n📊 每隔 {interval} 秒进行一次定位分析（共 {len(timestamps)} 帧）")
        print(f"   视频时长：{int(self.duration)//60}分{int(self.duration)%60:02d}秒")
        
        progress_interval = max(1, len(timestamps) // 10)
        
        for i, timestamp in enumerate(timestamps):
            # Check cancellation between frames
            if cancellation_token and cancellation_token.is_cancelled:
                print(f"\n⏹ 分析已在 {timestamp:.1f} 秒处取消")
                break

            if timestamp > self.duration + 0.1:
                break
                
            self.current_time = timestamp
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                if i % progress_interval == 0:
                    print(f"⚠️ 无法读取 {timestamp:.1f} 秒处的视频帧")
                continue
            
            frame_b64 = self.frame_to_base64(frame)
            
            try:
                response = self.llm.query(
                    user_message="What do you see in this frame? Describe the scene, people, objects, and actions.",
                    frame_base64=frame_b64,
                    temperature=0.3,
                    max_tokens=500,
                    cancellation_token=cancellation_token,
                )
                
                result = {
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "analysis": response,
                    "frame_number": i
                }
                
                results.append(result)
                self.analysis_cache.append(result)
                
                if callback:
                    callback(result)
                
                if i % progress_interval == 0 or i == len(timestamps) - 1:
                    print(f"  [{i+1}/{len(timestamps)}] {timestamp:.1f} 秒：已分析")
                    
                if self.verbose and len(response) > 0 and i % progress_interval == 0:
                    preview = response[:50] + "..." if len(response) > 50 else response
                    print(f"     ↪ {preview}")
                
            except GenerationCancelled:
                print(f"\n⏹ 生成已在 {timestamp:.1f} 秒处取消")
                break
            except Exception as e:
                if i % progress_interval == 0:
                    print(f"❌ {timestamp:.1f} 秒处出错：{e}")
                results.append({
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "error": str(e),
                    "frame_number": i
                })
        
        print(f"\n✅ 完成，成功分析 {len(results)} 帧")
        
        if save_to_file:
            self.save_results(results, save_to_file)
        
        return results

    def analyze_with_seeking(self, interval: float = 1.0, target_description: str = "explosion",
                            max_seeks: int = 100,
                            cancellation_token: Optional[CancellationToken] = None):
        results = []
        start_time = 0.0
        current_time = start_time
        
        print(f"\n🔍 每隔 {interval} 秒搜索一次：{target_description}")
        print("=" * 60)
        
        for seek_num in range(max_seeks):
            if cancellation_token and cancellation_token.is_cancelled:
                print(f"\n⏹ 搜索已在第 {seek_num} 次定位时取消")
                break

            timestamp = current_time + (seek_num * interval)
            
            if timestamp >= self.duration:
                print(f"\n🏁 已在 {timestamp:.1f} 秒处到达视频末尾")
                break
            
            print(f"\n⏩ 正在定位到 {timestamp:.1f} 秒（{int(timestamp)//60}:{int(timestamp)%60:02d}）")
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                print(f"⚠️ 无法读取 {timestamp:.1f} 秒处的视频帧")
                continue
            
            frame_b64 = self.frame_to_base64(frame)
            
            try:
                response = self.llm.query(
                    user_message=f"Does this frame contain a {target_description}? Answer with YES or NO, and briefly explain what you see.",
                    frame_base64=frame_b64,
                    system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
                    temperature=0.1,
                    max_tokens=150,
                    cancellation_token=cancellation_token,
                )
                
                result = {
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "analysis": response,
                    "contains_target": response.strip().lower().startswith("yes")
                }
                
                results.append(result)
                print(f"📝 分析：{response[:100]}…")
                
                if result["contains_target"]:
                    print(f"\n🎯 在 {timestamp:.1f} 秒处找到 {target_description.upper()}！")
                    print(f"完整分析：{response}")
                    break
                
            except GenerationCancelled:
                print(f"\n⏹ 搜索已在 {timestamp:.1f} 秒处取消")
                break
            except Exception as e:
                print(f"❌ {timestamp:.1f} 秒处出错：{e}")
            
            time.sleep(0.5)
        
        print(f"\n✅ 定位分析完成，共分析 {len(results)} 帧")
        return results

    def save_results(self, results: list, filepath: str):
        output = {
            "video_path": self.video_path,
            "video_info": {
                "duration": self.duration,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "total_frames": self.total_frames
            },
            "analysis_interval": 1,
            "total_analyses": len(results),
            "analyses": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=1, ensure_ascii=False)
        
        print(f"💾 结果已保存到：{filepath}")
    
    def search_analyses(self, query: str, results: list = None) -> list:
        search_results = []
        analyses = results if results is not None else self.analysis_cache
        query_lower = query.lower()
        
        for r in analyses:
            if "analysis" in r and query_lower in r["analysis"].lower():
                search_results.append({
                    "timestamp": r["timestamp"],
                    "timestamp_str": r["timestamp_str"],
                    "context": r["analysis"][:200] + "..." if len(r["analysis"]) > 200 else r["analysis"]
                })
        
        return search_results
    
    def interactive_mode(self, interval: int = 1):
        self.running = True
        
        def analysis_loop():
            consecutive_errors = 0
            while self.running:
                try:
                    with self.lock:
                        current_pos = self.current_time
                    
                    result = self.analyze_current_frame()
                    
                    if "error" in result:
                        consecutive_errors += 1
                        if consecutive_errors > 3:
                            print("\n❌ 错误过多，已停止分析")
                            break
                    else:
                        consecutive_errors = 0
                        self.analysis_cache.append(result)
                        
                        print(f"\n{'='*60}")
                        print(f"📹 [{result['timestamp_str']}] 分析：")
                        print(f"{'='*60}")
                        analysis = result['analysis']
                        if len(analysis) > 300:
                            print(analysis[:300] + "...")
                        else:
                            print(analysis)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"\n⚠️ 分析出错：{e}")
                    time.sleep(interval)
        
        thread = threading.Thread(target=analysis_loop, daemon=True)
        thread.start()
        self._show_help()
        
        while self.running:
            try:
                cmd = input("\n🎮 Command: ").strip().lower()
                if not cmd:
                    continue
                
                parts = cmd.split()
                
                if parts[0] == 's' and len(parts) > 1:
                    self._handle_seek_command(parts[1])
                elif parts[0] == 'p':
                    print(f"📌 当前位置：{int(self.current_time)//60}:{int(self.current_time)%60:02d}")
                elif parts[0] == 'f' and len(parts) > 1:
                    self._handle_forward_command(parts[1])
                elif parts[0] == 'b' and len(parts) > 1:
                    self._handle_backward_command(parts[1])
                elif parts[0] == 'h':
                    self._show_help()
                elif parts[0] == 'q':
                    print("\n👋 正在停止分析…")
                    self.running = False
                    break
                else:
                    print("❌ 未知命令。输入 'h' 查看帮助。")
                    
            except KeyboardInterrupt:
                print("\n\n👋 已中断，正在停止…")
                self.running = False
                break
            except Exception as e:
                print(f"❌ 命令执行出错：{e}")
    
    def _show_help(self):
        print("\n" + "="*50)
        print("🎬 交互式视频分析模式")
        print("="*50)
        print("命令：")
        print("  s <秒数>      - 定位到指定时间（例如：'s 30'）")
        print("  s <mm:ss>     - 定位到指定时间（例如：'s 1:30'）")
        print("  p             - 显示当前位置")
        print("  f <秒数>      - 向前移动 N 秒")
        print("  b <秒数>      - 向后移动 N 秒")
        print("  h             - 显示此帮助")
        print("  q             - 退出")
        print("="*50)
    
    def _handle_seek_command(self, arg: str):
        try:
            if ':' in arg:
                minutes, seconds = map(int, arg.split(':'))
                seek_to = minutes * 60 + seconds
            else:
                seek_to = int(arg)
            
            seek_to = max(0, min(seek_to, int(self.duration)))
            with self.lock:
                self.current_time = seek_to
            
            if self.verbose:
                print(f"⏩ 正在定位到 {self.current_time//60}:{self.current_time%60:02d}")
        except ValueError:
            print("❌ 时间格式无效，请使用秒数或 mm:ss")
    
    def _handle_forward_command(self, arg: str):
        try:
            forward = int(arg)
            with self.lock:
                self.current_time = min(self.current_time + forward, self.duration)
            print(f"⏩ 向前 {forward} 秒，到 {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ 向前移动的数值无效")
    
    def _handle_backward_command(self, arg: str):
        try:
            backward = int(arg)
            with self.lock:
                self.current_time = max(self.current_time - backward, 0)
            print(f"⏪ 向后 {backward} 秒，到 {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ 向后移动的数值无效")
    
    def close(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print("📹 视频捕获资源已释放")


# ---------------------------------------------------------------------------
# Example usage when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 带视频定位分析器的 LLM 模块")
    print("=" * 50)
    print("本模块提供：")
    print("  - LLMModule：本地 LLM 接口")
    print("  - VideoSeekAnalyzer：按时间间隔分析视频帧")
    print("\n示例用法：")
    print("  from llm_module import LLMModule, VideoSeekAnalyzer")
    print("  llm = LLMModule(backend='ollama', model='llava')")
    print("  llm.load()")
    print("  analyzer = VideoSeekAnalyzer('video.mp4', llm)")
    print("  results = analyzer.analyze_every_1_second()")
    print("  analyzer.close()")