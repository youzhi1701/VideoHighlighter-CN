"""Where a project stands, and the one thing to do next.

The whole pipeline is steps with files between them, so at any moment exactly
one step is the useful next one. ``status`` works it out and hands back the
command for it. A person reads it as a to-do line; an agent runs the command,
calls ``status`` again, and repeats: that loop is all an LLM needs to take a
project from "I want to find X" to a trained model, stopping only where a
judgement is needed (naming, reviewing).

Each step is described with ``who``: ``"auto"`` steps can run unattended;
``"judge"`` steps need someone to look (a person, or an agent that can see
the contact sheet), because every safeguard downstream rests on it.
"""
from __future__ import annotations

import shlex

from modules.teach.project import (
    MIN_TO_TRAIN, OBJECTS, PENDING, Project,
)


def _cmd(project: Project, *args) -> str:
    return " ".join(["python -m modules.teach", "--project",
                     shlex.quote(project.root)] + [shlex.quote(str(a)) for a in args])


def _step(project, who: str, why: str, *args) -> dict:
    return {"who": who, "why": why, "command": _cmd(project, *args) if args else "",
            "args": list(args)}


def next_step(project: Project) -> dict:
    counts = project.counts()
    names = project.class_names()
    pending = [s for s in project.samples if s.verdict == PENDING]
    scored = [s for s in project.samples if s.scores]

    if not names:
        return _step(project, "judge", "请先说明要查找什么：添加一个类别（名称 + 几个描述词）。如果不确定如何命名，"
                     "可先运行 `check-name`。",
                     "add-class", "<name>", "--description", "<what it looks like>")
    if not project.sources:
        return _step(project, "judge", "请提供素材：加入几个会出现这些类别的视频。URL 会自动下载，本地文件则直接使用。",
                     "add-video", "<path or url>")
    if any(not s.cut for s in project.sources):
        return _step(project, "auto", "将新素材切分为样本。", "cut")
    if (project.task == "actions" and project.settings.focus
            and any(not s.focus_tried for s in project.samples)):
        return _step(project, "auto", "将样本裁剪到其中的人物区域。", "focus")
    # An object class someone drew a box around is found region by region
    # (``find``): its samples are decided by answering its boxes, so the
    # whole-frame sort and the sample review are only for classes in words.
    drawn = set()
    if project.task == OBJECTS:
        from modules.teach.boxes import seeded, store
        drawn = seeded(store(project))
    by_words = [n for n in names if n not in drawn]

    unsorted = [s for s in project.samples if not s.scores and not s.unreadable]
    if by_words and (not scored or unsorted):
        return _step(project, "auto", "对每个样本与每个类别进行评分。", "sort")
    if drawn:
        from modules.teach.find import needed
        if needed(project):
            return _step(project, "auto", "在每个样本中查找已示教的目标。", "find")

    no_examples = [n for n in by_words if counts[n]["accepted"] == 0 and not
                   project.get_class(n).examples]
    if (no_examples and len(pending)
            and all(counts[n]["accepted"] == 0 for n in by_words)):
        # Words alone sort weakly. One reviewed sheet turns into examples, and
        # every sort after it uses them.
        return _step(project, "judge", "检查第一批模型判断：接受的样本会成为示例，使后续排序更准确。"
                     "也可以用 `add-example` 添加已有示例片段。", "review")

    short = [n for n in by_words if counts[n]["accepted"] < counts[n]["target"]]
    ready = [n for n in names if counts[n]["accepted"] >= MIN_TO_TRAIN]
    if short and pending:
        worst = min(short, key=lambda n: counts[n]["accepted"] / max(counts[n]["target"], 1))
        c = counts[worst]
        if project.rounds and len(ready) == len(names):
            pass    # enough to train again; reviewing more is optional below
        else:
            return _step(project, "judge",
                         f"检查模型判断：{worst!r} 已接受 {c['accepted']}/{c['target']} 个样本。"
                         "检查几批后重新运行 `sort`，已接受样本会帮助下一轮判断更准确。", "review")

    if project.task == OBJECTS:
        from modules.teach.boxes import labeler_worklist, retryable, store
        labels = store(project)
        if labels.pending():
            return _step(project, "judge", f"检查 {len(labels.pending())} 个候选检测框。",
                         "boxes", "review")
        todo = labeler_worklist(project)
        if any(not s.boxes_tried for s in project.accepted()):
            return _step(project, "auto", "为已接受样本生成候选检测框。",
                         "boxes", "propose")
        retry = retryable(project, labels)
        if retry:
            return _step(project, "auto", f"重新处理 {len(retry)} 个检测框曾被拒绝的画面，并参考已接受检测框的外观进行匹配。",
                         "boxes", "propose")
        if len(todo) > len(project.accepted()) // 2:
            return _step(project, "judge", f"{len(todo)} 个已接受样本仍没有合适的检测框。"
                         "请在 tools/labeler.py 中标注，然后导入导出结果。",
                         "boxes", "worklist")

    if len(ready) < len(names):
        missing = [n for n in names if n not in ready]
        more = (" 也可以从不同角度或其他视频使用 `seed` 再添加一个检测框；"
                "每增加一个示例都能扩大可查找范围。" if drawn & set(missing) else "")
        return _step(project, "judge", f"{', '.join(repr(n) for n in missing)} 至少需要 {MIN_TO_TRAIN} 个已接受样本。"
                     "请添加更多包含这些目标的素材，或继续检查现有样本。" + more,
                     "add-video", "<path or url>")

    from modules.teach import autolabel
    audits = {n: autolabel.audits_needed(project, n) for n in names + ["_none"]}
    owed = {n: k for n, k in audits.items() if k}
    if owed:
        listed = "，".join(f"{n!r} 还需 {k} 个" for n, k in owed.items())
        return _step(project, "judge", f"训练前请抽查自动接受的样本（{listed}）；检查批次中会包含这些样本。", "review")

    from modules.teach.build import built_signature, dataset_signature

    signature = dataset_signature(project)
    if built_signature(project) != signature:
        return _step(project, "auto", "根据已接受的样本构建数据集。", "build")
    if not any(r.get("dataset") == signature for r in project.rounds):
        return _step(project, "auto", "开始一轮训练（GPU 通常需要数分钟到数小时，可无人值守运行；"
                     "只有效果优于上一版时才会安装新模型）。",
                     "train")
    return _step(project, "judge", "已使用全部接受样本完成训练。若要继续提升：添加模型没见过的视频，"
                 "再进行切分、排序和检查；模型在新素材上的错误最有价值。", "add-video", "<path or url>")


def report(project: Project) -> dict:
    counts = project.counts()
    verdicts = {}
    for sample in project.samples:
        verdicts[sample.verdict] = verdicts.get(sample.verdict, 0) + 1
    last = project.rounds[-1] if project.rounds else None
    return {
        "project": project.root,
        "name": project.name,
        "task": project.task,
        "classes": counts,
        "sources": len(project.sources),
        "samples": len(project.samples),
        "verdicts": verdicts,
        "rounds": len(project.rounds),
        "last_round": ({k: last[k] for k in ("round", "metrics", "installed",
                                             "better_than_installed") if k in last}
                       if last else None),
        "next": next_step(project),
    }
