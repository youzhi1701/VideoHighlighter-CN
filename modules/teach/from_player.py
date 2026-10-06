"""Teach a model from the player: draw a box, give it a name, done.

The one thing asked of the person is to point at the thing once. That box
becomes a seed (``seed``) in a project named after the thing, under the
app's user data, with ``background`` on: from then on, while the app is
idle, the project cuts the video, finds the thing in it, accepts what it is
sure of, trains, and installs the model only if it beats the last one
(``background``). The few boxes it is unsure of wait under Train -> From
videos -> Check guesses. Drawing another box with the same name, on another
frame or video, shows it from another side.
"""
from __future__ import annotations

from modules.teach.cli import resolve_root
from modules.teach.project import Project
from modules.teach.seed import seed


def teach(video: str, moment: float, roi, name: str) -> dict:
    """Seed ``name`` from a normalised ``roi`` at ``moment`` of ``video``."""
    name = " ".join(str(name).split())
    if not name:
        raise ValueError("请先为这个对象命名")
    root = resolve_root(name)
    result = seed(root, video, moment, tuple(roi), name)
    project = Project.load(root)
    if not project.settings.background:
        project.settings.background = True
        project.save()
    return {**result, "root": root}


def message(result: dict) -> str:
    """What the person is told after drawing the box."""
    if result["seeds"] > 1:
        return (f"已添加“{result['class']}”的另一个视角（目前共 {result['seeds']} 个）。"
                "不同视角会帮助模型识别更多情况。")
    return (f"正在学习识别“{result['class']}”。应用空闲时会在此视频中继续寻找它，"
            "自动接收高置信度结果并训练模型；只有新模型表现更好时才会启用。"
            "少量不确定结果会保留在“训练 → 从视频学习 → 检查判断”中供你确认。"
            "你也可以用相同名称再画一个框，补充另一个视角。")
