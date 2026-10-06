"""报告 CLIP 实际使用的后端，并验证该后端是否真正正确可用。

CLIP 会根据本机硬件选择后端（NVIDIA -> torch/CUDA，Intel -> OpenVINO，
否则使用 CPU）。最需要防范的是“静默回退”：选错后端仍可能得到看似正常的
分数，只是速度明显变慢。因此本工具回答三个问题：

  1. CLIP 最终解析到哪个后端/设备，依据是什么？
  2. 在该设备上真实前向计算是否正常，嵌入结果是否合理？
  3. 与 CPU 参考结果是否一致，而且是否确实更快？

    python -m tools.check_clip_device
    python -m tools.check_clip_device --frames 96        # 更长的基准测试
    python -m tools.check_clip_device --device CPU       # 强制指定后端
    python -m tools.check_clip_device --skip-reference   # 快速模式，不比较 CPU

如果解析出的后端错误、结果包含非有限值，或与参考结果不一致，程序会以非零
状态退出，因此它既可作为诊断输出，也可作为冒烟测试。

仅需要 torch + transformers + pillow + opencv-python + numpy。
CUDA 路径不需要 optimum-intel；OpenVINO 路径需要。
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm.clip_prefilter as cp  # noqa: E402
from llm.clip_index import ClipEmbedder  # noqa: E402

# Cosine below this between the tested backend and an fp32 CPU reference means
# the plumbing mangled something. fp16 noise lands ~1e-6 away, so this is loose
# enough to never flake and tight enough to catch a real fault.
AGREEMENT_MIN = 0.999


def _hr(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def report_env() -> None:
    _hr("环境")
    print(f"python       {platform.python_version()} ({platform.machine()})")
    print(f"操作系统     {platform.system()} {platform.release()}")

    try:
        import torch

        print(f"torch        {torch.__version__}")
        print(f"  CUDA 构建  {torch.version.cuda or '（无——这是 CPU/XPU wheel）'}")
        avail = torch.cuda.is_available()
        print(f"  是否可用   {avail}")
        if avail:
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                print(f"  设备 {i}    {p.name} ({p.total_memory / 1024**3:.1f} GB, "
                      f"sm_{p.major}{p.minor})")
    except Exception as e:
        print(f"torch        不可用——{type(e).__name__}：{e}")

    for mod in ("transformers", "optimum.intel", "openvino"):
        try:
            m = __import__(mod, fromlist=["__version__"])
            print(f"{mod:<12} {getattr(m, '__version__', '（无 __version__）')}")
        except Exception as e:
            note = "  （CUDA 不需要）" if mod.startswith("optimum") else ""
            print(f"{mod:<12} 无法导入——{type(e).__name__}{note}")


def report_resolution(requested: str) -> tuple[str, str]:
    _hr("后端解析")
    probe = cp.cuda_device()
    print(f"CUDA 设备探测结果 → {probe!r}")
    for req in ("AUTO", "GPU", "CUDA", "CPU"):
        print(f"  {req:<5} -> {cp.resolve_device(req)}")

    backend, device = cp.resolve_device(requested)
    print(f"\n请求 {requested!r} -> 后端={backend!r} 设备={device!r}")
    err = cp.ClipFramePrefilter.import_error(requested)
    print(f"依赖导入检查（{requested!r}）→ {err or '无（依赖栈导入正常）'}")
    return backend, device


def make_frames(n: int) -> list[np.ndarray]:
    """n distinct BGR frames, as OpenCV would hand them over. Structured rather
    than pure noise so CLIP produces embeddings that actually differ."""
    rng = np.random.default_rng(0)
    frames = []
    for i in range(n):
        f = np.zeros((224, 224, 3), np.uint8)
        f[:, :, i % 3] = 60 + (i * 37) % 190          # varying dominant channel
        f[40:180, 40:180] = rng.integers(0, 255, (140, 140, 3), dtype=np.uint8)
        frames.append(f)
    return frames


def cpu_reference(model_id: str) -> ClipEmbedder | None:
    """An fp32 CLIP on the CPU via plain torch — the thing we trust."""
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        ref = ClipEmbedder(device="CPU")
        ref.backend, ref.device, ref._dtype = "torch", "cpu", torch.float32
        ref._model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float32).eval()
        ref._processor = CLIPProcessor.from_pretrained(cp._bundled_ov_dir() or model_id)
        return ref
    except Exception as e:
        print(f"⚠️  无法建立 CPU 参考模型（{type(e).__name__}：{e}）")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="AUTO", help="AUTO/GPU/CUDA/cuda:N/CPU/...")
    ap.add_argument("--frames", type=int, default=48, help="用于基准测试的帧数")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--expect", default=None, choices=["torch", "openvino"],
                    help="如果未选择此后端则失败（用于 CI/自动化）")
    ap.add_argument("--skip-reference", action="store_true",
                    help="跳过 CPU 一致性检查（可省去第二次模型加载）")
    args = ap.parse_args()

    report_env()
    backend, device = report_resolution(args.device)

    err = cp.ClipFramePrefilter.import_error(args.device)
    if err is not None:
        # Stop here rather than let load() raise: a missing dep is a setup
        # problem with a known fix, and a traceback buries it.
        print(f"\n❌ 失败——{backend!r} 后端的依赖栈不完整：\n   {err}")
        if backend == "torch":
            print("\n   请安装 CUDA 版 torch：\n"
                  "     pip install torch --index-url https://download.pytorch.org/whl/cu128")
        else:
            print('\n   请安装 OpenVINO 依赖栈：\n'
                  '     pip install "optimum[openvino]" optimum-intel')
        print("   另外还需：pip install transformers pillow opencv-python numpy")
        return 1

    failures: list[str] = []
    if args.expect and backend != args.expect:
        failures.append(f"期望后端 {args.expect!r}，实际解析为 {backend!r}")

    _hr("加载")
    emb = ClipEmbedder(device=args.device)
    emb.load()
    print(f"已加载：后端={emb.backend!r} 设备={emb.device!r} dtype={emb._dtype}")
    if args.expect and emb.backend != args.expect:
        failures.append(f"期望加载到 {args.expect!r}，实际为 {emb.backend!r} "
                        f"（发生了回退，原因请查看上方警告）")

    _hr("正确性")
    frames = make_frames(args.frames)
    img = emb.embed_frames_bgr(frames[:4])
    print(f"图像嵌入     shape={img.shape} dtype={img.dtype}")
    if img.shape != (4, 512):
        failures.append(f"嵌入形状异常 {img.shape}")
    if not np.all(np.isfinite(img)):
        failures.append("图像嵌入中存在非有限值")
    norms = np.linalg.norm(img, axis=1)
    print(f"单位范数      {np.round(norms, 5)}")
    if not np.allclose(norms, 1.0, atol=1e-3):
        failures.append(f"嵌入不是单位范数：{norms}")

    txt = emb.embed_texts(["a red photo", "a blue photo"])
    print(f"文本嵌入     shape={txt.shape}")
    print(f"logit_scale   {emb.logit_scale:.2f}（CLIP 公布值为 100.0）")

    # The calibrated-score path, which is what the app thresholds on.
    emb.set_query("a red photo", negatives=["a blue photo"])
    scores = emb.score_frames_bgr(frames[:4])
    print(f"分数          {[round(s, 4) for s in scores]}")
    if not all(np.isfinite(scores)):
        failures.append("score_frames_bgr 返回了非有限分数")

    if not args.skip_reference:
        _hr("与 fp32 CPU 参考结果的一致性")
        ref = cpu_reference(emb.model_id)
        if ref is not None:
            ref_img = ref.embed_frames_bgr(frames[:4])
            agree = np.abs((img * ref_img).sum(1))
            print(f"每帧余弦值   {np.round(agree, 6)}")
            worst = float(agree.min())
            print(f"最差值       {worst:.6f}（阈值 {AGREEMENT_MIN}）")
            if worst < AGREEMENT_MIN:
                failures.append(f"后端与 CPU 参考结果不一致（最差 {worst:.6f}），"
                                f"嵌入结果会不正确，且与其他机器生成的缓存索引不兼容")
            else:
                print("→ 与 CPU 结果一致：索引可在不同机器之间通用。")

    _hr(f"基准测试（{args.frames} 帧，批大小 {args.batch}）")
    emb.embed_frames_bgr(frames[:args.batch])          # warm up kernels/caches
    t0 = time.perf_counter()
    for i in range(0, len(frames), args.batch):
        emb.embed_frames_bgr(frames[i:i + args.batch])
    elapsed = time.perf_counter() - t0
    per = elapsed / len(frames) * 1000
    print(f"{emb.device}：总计 {elapsed:.2f} 秒，{per:.1f} 毫秒/帧，"
          f"{len(frames) / elapsed:.1f} 帧/秒")
    print("（这里只测试编码；实际扫描还会包含视频解码）")

    _hr("结论")
    if failures:
        print(f"❌ 失败（{len(failures)} 项）")
        for f in failures:
            print(f"   - {f}")
        return 1
    print(f"✅ 通过——CLIP 正在 {emb.device!r} 上通过 {emb.backend!r} 后端运行，"
          f"速度为 {per:.1f} 毫秒/帧。")
    return 0


if __name__ == "__main__":
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    raise SystemExit(main())
