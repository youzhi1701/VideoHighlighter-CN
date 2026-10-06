"""Train a taught-action head on a dataset of sorted clips.

    python -m model_training.action_head.train --data-path <dataset>
        [--out <folder>] [--name taught-actions] [--frames 4] [--folds 5]
        [--steps 750,1500,3000] [--min-clips 5] [--precision 0.7]
        [--backend auto|intel|directml|cpu] [--cache <file.npz>]
        [--aliases <file.json>] [--teach-test] [--min-videos 3]

The dataset is the app's layout: ``train/``, ``val/`` (and ``test/``) holding
one folder per class, clips directly inside. A folder named ``a_b`` shows both
``a`` and ``b``: the head scores every action on its own, so such clips teach
it that two actions can be present together. It only learns the
combinations it is shown; measured, taught combinations are found far more
often than ones it never saw.

What it does:

1. **Encodes every clip once** (4 frames across it, ``features.py``), cached.
2. **Scores on unseen source videos.** The clips are split by source video
   (the clip name before ``_temp``/``_highlight``), five ways: neighbouring
   clips of one video share scene, people and light, so a split by clip
   rewards remembering the scene. Each fold trains a head on the other videos
   and scores its own. The out-of-fold scores choose the training length
   (``--steps``) and give every number this prints.
3. **Sets each action's trust threshold** from those scores
   (``trust.trust_thresholds``): the score above which "this action is
   present" is right ``--precision`` of the time, at 80 % confidence, with
   hits from ``--min-videos`` source videos or more. An action that never
   gets there is only ever a suggestion. Each pair of actions taught in 5 or
   more clips gets a threshold of its own, on the lower of its two scores.
4. **Trains the saved head on every clip** with the chosen length.
5. **Scores ``test/``.** Without ``--teach-test`` it is not trained on: each
   clip is scored by the fold heads that never saw its source video. With
   ``--teach-test`` its clips join the folds (scored the same honest way) and
   the saved head learns from them too.
6. **Scores ``val/`` as the dataset defines it** (trained on ``train/`` only),
   next to how many of its clips share a source video with ``train/``: that
   share is how much of the score is remembering the scene.

Writes ``head.onnx`` and ``head.json`` (encoder id, frames, classes,
thresholds, held-out scores) into the output folder. Nothing in them names a
file or holds a frame.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

HEAD_FORMAT = 1
HEAD_FILE = "head.onnx"
META_FILE = "head.json"
DEFAULT_NAME = "taught-actions"
POOL_SPLITS = ("train", "val")


def _utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a stream without reconfigure is fine as is
            pass


def default_cache(data_path: str, encoder_id: str, frames: int) -> str:
    from modules.system import app_paths
    digest = hashlib.sha1(os.path.abspath(data_path).lower().encode("utf-8")).hexdigest()[:12]
    return os.path.join(app_paths.user_data_dir(), "cache", "action_head",
                        f"{digest}-{encoder_id}-{frames}f.npz")


def default_out(name: str) -> str:
    from modules.system import app_paths
    return os.path.join(app_paths.action_models_dir(), name)


def select_clips(clips, min_clips: int, splits=POOL_SPLITS):
    """The clips to train on, from ``splits``: every clip whose actions are
    all classes with ``min_clips`` or more single-action clips. Returns
    ``(kept, classes, notes)``, notes being sentences for the log."""
    notes = []
    pool = [c for c in clips if c.split in splits]
    counts = Counter(c.labels[0] for c in pool if len(c.labels) == 1)
    classes = sorted(k for k, v in counts.items() if v >= min_clips)
    known = set(classes)
    small = sorted(k for k, v in counts.items() if v < min_clips)
    if small:
        notes.append(f"{len(small)} 个类别的单动作片段少于 {min_clips} 个，已排除："
                     + ", ".join(f"{k} ({counts[k]})" for k in small))
    kept = [c for c in pool if c.labels and set(c.labels) <= known]
    multi = sum(len(c.labels) > 1 for c in kept)
    if multi:
        notes.append(f"{multi} 个片段包含两个或更多动作，将一起用于示教")
    unknown = [c for c in pool if not set(c.labels) <= known]
    if unknown:
        notes.append(f"{len(unknown)} 个片段包含当前类别列表中不存在的动作，已排除")
    return kept, classes, notes


def group_folds(strat: np.ndarray, groups: np.ndarray, folds: int, seed: int) -> list:
    """``folds`` (train, held-out) index pairs; no source video on both sides."""
    from sklearn.model_selection import StratifiedGroupKFold
    folds = max(2, min(folds, len(set(groups.tolist()))))
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(strat)), strat, groups))


def out_of_fold(x, targets, groups, splits, steps, seed, log,
                x_extra=None, groups_extra=None):
    """Held-out scores for every clip, and for ``x_extra`` (test clips) the
    mean over the fold heads that never saw the clip's source video (NaN when
    none qualifies)."""
    from model_training.action_head import head as H
    n_classes = targets.shape[1]
    scores = np.zeros((len(targets), n_classes), np.float32)
    n_extra = 0 if x_extra is None else len(x_extra)
    extra_sum = np.zeros((n_extra, n_classes), np.float32)
    extra_n = np.zeros(n_extra)
    for i, (tr, te) in enumerate(splits, 1):
        model = H.train_head(x[tr], targets[tr], n_classes, steps=steps, seed=seed)
        scores[te] = H.predict_proba(model, x[te])
        if n_extra:
            unseen = ~np.isin(groups_extra, groups[tr])
            if unseen.any():
                extra_sum[unseen] += H.predict_proba(model, x_extra[unseen])
                extra_n[unseen] += 1
        one = targets[te].sum(1) == 1
        acc = np.mean(scores[te][one].argmax(1) == targets[te][one].argmax(1)) if one.any() else 0.0
        log(f"    折 {i}/{len(splits)}：单动作片段 {int(one.sum())} 个，准确率 {acc:.3f}")
    extra = extra_sum / np.maximum(extra_n, 1)[:, None]
    extra[extra_n == 0] = np.nan
    return scores, extra


def _log_scores(log, title: str, s: dict) -> None:
    if "single" in s:
        t = s["single"]
        log(f"{title}，单动作（{t['clips']} 个片段）：准确率 {t['accuracy']:.3f}，"
            f"平衡准确率 {t['balanced_accuracy']:.3f}，Top-3 {t['top3']:.3f}")
        if t["trusted_precision"] is not None:
            log(f"  可信结果：覆盖 {t['trusted_share']:.0%}，其中正确率 {t['trusted_precision']:.0%}")
    if "two_actions" in s:
        t = s["two_actions"]
        log(f"{title}，双动作（{t['clips']} 个片段）：两个都进入 Top 2 {t['both_in_top2']:.0%}，"
            f"两个都进入 Top 5 {t['both_in_top5']:.0%}，首位命中其中一个 {t['top1_is_one_of_them']:.0%}")
        log(f"  可信结果：两个均检测 {t['both_detected']:.0%}，仅一个 {t['one_detected']:.0%}，"
            f"同时误检其他动作 {t['wrong_detected']:.0%}")


def main(argv=None) -> int:
    _utf8_stdout()
    ap = argparse.ArgumentParser(description="使用帧编码器训练已示教的动作分类头")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--out", default=None, help="输出文件夹（默认：models/actions/<name>）")
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", default="750,1500,3000",
                    help="在留出视频上比较的训练步数")
    ap.add_argument("--min-clips", type=int, default=5)
    ap.add_argument("--precision", type=float, default=0.7,
                    help="动作被视为可信所需达到的留出集精确率")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default=None, help="编码器计算路线（compute.backend 的值）")
    ap.add_argument("--cache", default=None, help="特征缓存文件（.npz）")
    ap.add_argument("--aliases", default=None,
                    help='文件夹或类别名 -> 类别名的 JSON 映射（"" 表示排除），与示教模块一致')
    ap.add_argument("--min-videos", type=int, default=3,
                    help="可信动作在留出集中的命中至少需要来自多少个源视频")
    ap.add_argument("--teach-test", action="store_true",
                    help="同时从 test/ 学习（评分仍只使用从未见过对应视频的分类头）")
    args = ap.parse_args(argv)
    log = print

    from model_training.action_head import features as Fx
    from model_training.action_head import head as H
    from model_training.action_head import trust
    from modules.teach.benchmark import load_aliases, read_dataset
    from modules.vision import frame_encoder

    started = time.time()
    data = read_dataset(args.data_path, aliases=load_aliases(args.aliases))
    splits_used = POOL_SPLITS + (("test",) if args.teach_test else ())
    clips, classes, notes = select_clips(data["clips"], args.min_clips, splits_used)
    for note in notes:
        log(f"ℹ️ {note}")
    if not classes:
        log("❌ 没有任何类别拥有足够的单动作片段可用于训练")
        return 1
    known = set(classes)
    extra = ([] if args.teach_test else
             [c for c in data["clips"] if c.split == "test" and c.labels and set(c.labels) <= known])

    encoder = frame_encoder.load(args.backend, log=log)
    if encoder is None:
        log("❌ 帧编码器不可用，请查看上方日志")
        return 1
    cache = Fx.FeatureCache(args.cache or default_cache(args.data_path, encoder.encoder_id, args.frames),
                            encoder.encoder_id, args.frames, encoder.dims)
    n = len(clips)
    x_all, ok = Fx.encode_clips([c.path for c in clips + extra], args.data_path, encoder,
                                cache, log=log)
    x, x_extra = x_all[:n][ok[:n]], x_all[n:][ok[n:]]
    clips = [c for c, good in zip(clips, ok[:n]) if good]
    extra = [c for c, good in zip(extra, ok[n:]) if good]

    index = {c: k for k, c in enumerate(classes)}
    label_sets = [{index[lb] for lb in c.labels} for c in clips]
    targets = np.zeros((len(clips), len(classes)), np.float32)
    for i, s in enumerate(label_sets):
        targets[i, list(s)] = 1
    strat = np.array([min(s) for s in label_sets])
    groups = np.array([c.group for c in clips])
    g_extra = np.array([c.group for c in extra])
    n_videos = len(set(groups.tolist()))
    log(f"\n{len(clips)} 个片段（{int((targets.sum(1) > 1).sum())} 个包含两个或更多动作），"
        f"{len(classes)} 个类别，{n_videos} 个源视频")

    folds = group_folds(strat, groups, args.folds, args.seed)
    log(f"正在未见过的源视频上评分（{len(folds)} 折）")
    single = targets.sum(1) == 1
    best = None
    for steps in [int(s) for s in str(args.steps).split(",") if s.strip()]:
        log(f"  {steps} 步")
        scores, scores_extra = out_of_fold(x, targets, groups, folds, steps, args.seed, log,
                                           x_extra=x_extra if extra else None,
                                           groups_extra=g_extra)
        acc = float(np.mean(scores[single].argmax(1) == targets[single].argmax(1)))
        log(f"  {steps} 步：留出集单动作准确率 {acc:.3f}")
        if best is None or acc > best[1] + 1e-9:
            best = (steps, acc, scores, scores_extra)
    steps, acc, scores, scores_extra = best
    thresholds = trust.trust_thresholds(scores, targets, target=args.precision,
                                        groups=groups, min_videos=args.min_videos)
    pairs = trust.taught_pairs(targets)
    pair_th = trust.pair_thresholds(scores, targets, pairs, target=args.precision,
                                    groups=groups, min_videos=args.min_videos)

    is_test = np.array([c.split == "test" for c in clips], bool)
    heldout = trust.score_clips(scores[~is_test],
                                [s for s, t in zip(label_sets, is_test) if not t], thresholds,
                                pairs, pair_th)
    if args.teach_test:
        test_scores = (trust.score_clips(scores[is_test],
                                         [s for s, t in zip(label_sets, is_test) if t], thresholds,
                                         pairs, pair_th)
                       if is_test.any() else {})
    elif extra:
        scored = ~np.isnan(scores_extra).any(1)
        test_scores = trust.score_clips(scores_extra[scored],
                                        [{index[lb] for lb in c.labels}
                                         for c, s in zip(extra, scored) if s], thresholds,
                                        pairs, pair_th)
    else:
        test_scores = {}

    in_train = np.array([c.split == "train" for c in clips], bool)
    in_val = np.array([c.split == "val" for c in clips], bool) & single
    val_scores = None
    if in_train.any() and in_val.any():
        log("\n正在按数据集定义对 val/ 评分（仅使用 train/ 训练）")
        model = H.train_head(x[in_train], targets[in_train], len(classes), steps=steps,
                             seed=args.seed)
        val_pred = H.predict_proba(model, x[in_val]).argmax(1)
        shared = np.isin(groups[in_val], groups[in_train])
        val_scores = {"clips": int(in_val.sum()),
                      "accuracy": round(float(np.mean(val_pred == targets[in_val].argmax(1))), 4),
                      "clips_sharing_a_video_with_train": int(shared.sum())}

    log(f"\n正在使用全部 {len(clips)} 个片段训练最终保存的分类头（{steps} 步）")
    model = H.train_head(x, targets, len(classes), steps=steps, seed=args.seed)
    out = args.out or default_out(args.name)
    os.makedirs(out, exist_ok=True)
    head_path = os.path.join(out, HEAD_FILE)
    H.export_onnx(model, head_path, args.frames)
    check = H.onnx_proba(H.load_onnx_session(head_path), x[:64])
    drift = float(np.abs(check - H.predict_proba(model, x[:64])).max())
    if drift > 1e-4:
        log(f"❌ 导出的分类头与训练结果不一致（最大偏差 {drift:.2e}）")
        return 1

    found = trust.detected(scores, thresholds, pairs, pair_th)
    pred = scores.argmax(1)
    per_class = {}
    for k, name in enumerate(classes):
        shows, said = targets[:, k] > 0, found[:, k]
        mine = shows & single
        per_class[name] = {
            "clips": int(mine.sum()),
            "clips_with_another_action": int((shows & ~single).sum()),
            "videos": int(len(set(groups[shows].tolist()))),
            "heldout_recall": round(float(np.mean(pred[mine] == k)), 3) if mine.any() else None,
            "heldout_detected_precision": (round(float(np.mean(shows[said])), 3)
                                           if said.any() else None),
            "trust_threshold": (None if thresholds[k] is None else round(thresholds[k], 4)),
        }
    meta = {
        "format": HEAD_FORMAT,
        "kind": "action-head",
        "encoder": encoder.encoder_id,
        "frames": args.frames,
        "input": f"features [N, frames, {encoder.dims}]: frame encoder vectors, "
                 f"frames evenly across the clip",
        "output": "logits [N, classes]; a sigmoid gives each action's own score, and an "
                  "action is detected when its score reaches its trust threshold",
        "activation": "sigmoid",
        "classes": classes,
        "trust_thresholds": [None if t is None else round(t, 4) for t in thresholds],
        "pair_thresholds": [{"actions": [classes[a], classes[b]],
                             "threshold": None if t is None else round(t, 4)}
                            for (a, b), t in zip(pairs, pair_th)],
        "trust_precision": args.precision,
        "trust_min_videos": args.min_videos,
        "steps": steps,
        "heldout": {"how": f"{len(folds)} folds by source video", "clips": len(clips),
                    "videos": n_videos, **heldout},
        "test": {"taught": bool(args.teach_test), **test_scores},
        "val_folder": val_scores,
        "per_class": per_class,
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    with open(os.path.join(out, META_FILE), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)

    log("")
    _log_scores(log, "留出集（完整源视频从未参与训练）", heldout)
    if val_scores:
        log(f"按数据集定义的 val/（仅使用 train/ 训练）：准确率 {val_scores['accuracy']:.3f}，"
            f"共 {val_scores['clips']} 个片段，其中 {val_scores['clips_sharing_a_video_with_train']} 个"
            f"与 train/ 共享源视频")
    if test_scores:
        _log_scores(log, "test/" + ("（参与示教；由从未见过对应视频的分类头评分）"
                                    if args.teach_test else "（未参与示教）"), test_scores)
    log(f"可信动作：{sum(t is not None for t in thresholds)}/{len(classes)}；"
        f"可信动作对：{sum(t is not None for t in pair_th)}/{len(pairs)} 个已示教动作对")
    width = max(len(c) for c in classes)
    log(f"\n{'类别':{width}}  片段 +动作对 视频数 召回率 精确率   阈值")
    for name, row in sorted(per_class.items(), key=lambda kv: -kv[1]["clips"]):
        rec = "-" if row["heldout_recall"] is None else f"{row['heldout_recall']:.2f}"
        prec = ("-" if row["heldout_detected_precision"] is None
                else f"{row['heldout_detected_precision']:.2f}")
        th = "不可信" if row["trust_threshold"] is None else f"{row['trust_threshold']:.2f}"
        log(f"{name:{width}}  {row['clips']:5} {row['clips_with_another_action']:5} "
            f"{row['videos']:6} {rec:>6} {prec:>9} {th:>11}")
    log(f"\n✅ 已保存 {head_path} 和 {META_FILE}（{time.time() - started:.0f} 秒）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
