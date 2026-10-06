"""What to call a class, decided from its examples rather than guessed.

A class name does three jobs, and a bad one fails all of them quietly:

* it is the **folder** a sample is sorted into and the **label** a model
  reports, so it must be safe as a path and stable across rounds;
* it is the **prompt** CLIP sorts with before the first model exists, so a
  code like ``cls1`` sorts nothing — it needs words that describe the thing;
* it is what the **person** reads in the timeline months later.

So there are rules (``check_name``), and there is a way to let a few samples
suggest a name (``suggest_names``): embed them, and rank the labels the app's
stock models already know — Kinetics-400 for actions, COCO for objects, the
two label files in the repo root — by how well each describes them. The answer
is advice, never a decision:

* a stock label that fits well is worth reusing, because the stock model can
  then help pre-sort, and the name means the same thing everywhere;
* nothing fitting well is the normal case for something new, and says only
  that the name has to come from the person (or the agent looking at a
  contact sheet of the examples) — plus a description, since CLIP will lean on
  it;
* examples that do not agree with each other (``consistency``) usually mean
  two things under one name. Splitting them is cheaper now than after a
  hundred samples are sorted.

Conventions, which keep names comparable with the stock vocabularies:
lowercase, words separated by single spaces; actions as what is being done
("riding a bike" style), objects as a singular noun. Mechanism only: the
vocabulary is whatever label files the app ships, and nothing here holds an
opinion about what a class should be.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MAX_LENGTH = 40

# Names that say nothing about what they name. Allowed, with advice: CLIP can
# not sort by them, and a timeline full of "thing" helps nobody.
_VAGUE = {"thing", "things", "object", "objects", "action", "actions", "stuff",
          "other", "misc", "item", "items", "class", "category", "unknown",
          "good", "bad", "yes", "no", "test"}

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
VOCAB_FILES = {
    "actions": os.path.join(_REPO, "kinetics_400_labels.json"),
    "objects": os.path.join(_REPO, "yolo_objects_labels.json"),
}

# How a vocabulary label is worded for CLIP. Scores are only compared within
# one vocabulary, so the template just has to be the same for every label.
PROMPTS = {
    "actions": "a video frame of a person {}",
    "objects": "a photo of a {}",
}


@dataclass
class Problem:
    code: str
    message: str
    blocking: bool = False

    def to_json(self) -> dict:
        return {"code": self.code, "message": self.message, "blocking": self.blocking}


def normalize_name(name: str) -> str:
    """``"  Kick_Flip-Fast "`` -> ``"kick flip fast"``."""
    text = unicodedata.normalize("NFKC", str(name or "")).strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _plain(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name)


def _singular(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def check_name(name: str, existing: Sequence[str] = (),
               task: Optional[str] = None,
               vocabulary: Optional[Sequence[str]] = None) -> list:
    """Everything wrong with ``name`` as a class. Blocking problems stop it.

    ``name`` is taken as given; normalise first. ``vocabulary`` defaults to the
    stock labels for ``task`` when that is given.
    """
    problems = []
    if not name:
        return [Problem("empty", "类别需要一个名称。", True)]
    if name != normalize_name(name):
        problems.append(Problem("format", f"请使用 {normalize_name(name)!r}：使用小写字母，"
                                          "单词之间只保留一个空格。", True))
    if len(name) > MAX_LENGTH:
        problems.append(Problem("long", f"名称请控制在 {MAX_LENGTH} 个字符以内；"
                                        "更多细节请写在描述中。", True))
    if name.startswith("_") or name in {".", ".."}:
        problems.append(Problem("reserved", "以下划线 _ 开头的名称保留给排序器内部文件夹使用。", True))
    if re.search(r'[/\\:*?"<>|]', name):
        problems.append(Problem("path", "类别名称会用于文件夹名，请不要包含 / \\ : * ? \" < > |。",
                                True))
    if re.fullmatch(r"[\d\s.]+", name):
        problems.append(Problem("numeric", "纯数字不能作为类别名称，请说明它是什么。", True))

    for other in existing:
        if other == name:
            problems.append(Problem("duplicate", f"已经存在类别 {name!r}。", True))
        elif (_plain(other) == _plain(name)
              or " ".join(map(_singular, other.split())) == " ".join(map(_singular, name.split()))):
            problems.append(Problem("near_duplicate",
                                    f"{name!r} 与 {other!r} 看起来是同一个类别。请合并为一个类别，"
                                    "或在描述中明确两者区别。"))

    words = name.split()
    if len(words) == 1 and name in _VAGUE:
        problems.append(Problem("vague", f"{name!r} 无法明确说明目标是什么。CLIP 会根据名称排序，"
                                         "请使用更具描述性的词语。"))
    if re.search(r"\d", name) and not re.search(r"[a-z]{3,}", name):
        problems.append(Problem("code", "这个名称看起来像代码。CLIP 无法仅靠代码进行排序；"
                                        "请添加描述或改用自然语言。"))

    if task == "objects" and words and _singular(words[-1]) != words[-1]:
        problems.append(Problem("plural", f"物体类别请使用单数形式，例如 "
                                          f"{' '.join(words[:-1] + [_singular(words[-1])])!r}；"
                                          "每个检测框对应一个物体。"))

    vocab = vocabulary if vocabulary is not None else (load_vocabulary(task) if task else [])
    if name in vocab:
        problems.append(Problem("stock", f"{name!r} 已是内置"
                                         f"{'动作' if task == 'actions' else '物体'}模型认识的标签。"
                                         "可以继续使用；这样它在各处含义一致。"))
    return problems


def load_vocabulary(task: Optional[str]) -> list:
    """The stock labels for ``task``, from the label files the app ships."""
    path = VOCAB_FILES.get(task or "")
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, dict) and isinstance(data.get("class"), dict):
        data = data["class"]
    if isinstance(data, dict):
        return [str(data[k]) for k in sorted(data, key=lambda k: int(k))]
    return [str(x) for x in data]


def _unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(norms, 1e-8)


def consistency(vectors: np.ndarray) -> float:
    """Mean pairwise cosine of the examples; 1.0 for a single one."""
    v = _unit(vectors)
    n = len(v)
    if n < 2:
        return 1.0
    sims = v @ v.T
    return float((sims.sum() - n) / (n * (n - 1)))


def split_hint(vectors: np.ndarray, threshold: float = 0.1) -> Optional[list]:
    """Two groups of example indices if the examples look like two things.

    A plain 2-means on the unit sphere, seeded with the least similar pair. The
    groups are returned only when each is clearly tighter than the whole —
    ``threshold`` is how much tighter, in mean cosine.
    """
    v = _unit(vectors)
    if len(v) < 4:
        return None
    sims = v @ v.T
    i, j = np.unravel_index(np.argmin(sims), sims.shape)
    centres = v[[i, j]]
    for _ in range(10):
        groups = np.argmax(v @ centres.T, axis=1)
        if len(set(groups.tolist())) < 2:
            return None
        centres = _unit(np.stack([v[groups == g].mean(axis=0) for g in (0, 1)]))
    whole = consistency(v)
    parts = [consistency(v[groups == g]) for g in (0, 1)]
    if min(parts) - whole < threshold or min(np.bincount(groups)) < 2:
        return None
    return [np.flatnonzero(groups == g).tolist() for g in (0, 1)]


def suggest_names(example_vectors: np.ndarray, labels: Sequence[str],
                  label_vectors: np.ndarray, top_k: int = 5) -> dict:
    """Rank stock labels by how well they describe the examples.

    ``example_vectors`` are unit image embeddings (one per example sample),
    ``label_vectors`` the unit text embeddings of ``labels`` in the task's
    prompt. Returns ``{"suggestions": [{name, score, fit}], "consistency",
    "split"}``.

    ``fit`` puts each score in words relative to the rest of the vocabulary:
    CLIP's image-text cosines sit in a narrow band (~0.15-0.35), so an absolute
    number reads as meaningless to anyone but this file. A label far above the
    vocabulary's typical score is ``"good"``; merely the best of a poor lot is
    ``"weak"``.
    """
    if len(labels) == 0:
        return {"suggestions": [], "consistency": consistency(example_vectors),
                "split": split_hint(example_vectors)}
    centre = _unit(_unit(example_vectors).mean(axis=0))
    scores = _unit(label_vectors) @ centre
    mean, std = float(scores.mean()), float(scores.std()) or 1e-6
    order = np.argsort(-scores)[:top_k]
    suggestions = []
    for idx in order:
        z = (float(scores[idx]) - mean) / std
        fit = "good" if z >= 3.0 else "possible" if z >= 2.0 else "weak"
        suggestions.append({"name": labels[idx], "score": round(float(scores[idx]), 4),
                            "fit": fit})
    return {"suggestions": suggestions,
            "consistency": round(consistency(example_vectors), 4),
            "split": split_hint(example_vectors)}
