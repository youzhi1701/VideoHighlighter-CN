"""Fetch the geometry a rule set asks for, beyond the detector's boxes.

Two on-demand passes, both run only where a rule could use them and both
cached with the boxes: outlines (``outline: true``, modules/vision/outlines.py)
and keypoints (rules naming a body part, modules/vision/keypoints.py). One
function, so the two places rules run (``compose_events.apply_rules`` in the
pipeline, ``analysis_ondemand.run_composition`` for Re-apply) cannot drift.
"""
from __future__ import annotations

from typing import Callable, Optional


def default_pose_estimator():
    from modules.vision.pose_backend import build_pose_estimator
    return build_pose_estimator(auto_install=True)


def trace_for_rules(video_path: str, boxes: list, engine, *,
                    outliner=None, pose_factory: Optional[Callable] = None,
                    cancel=None, log=print) -> dict:
    """Add outlines and keypoints to ``boxes`` (in place) as ``engine`` needs.

    Returns ``{"outlines": stats | None, "keypoints": stats | None}``. A pass
    with nothing to do costs nothing; a missing pose model is logged and the
    body-part rules simply find no parts.
    """
    report = {"outlines": None, "keypoints": None}
    if not video_path or not boxes:
        return report
    if engine.outline_pairs:
        from modules.vision.outlines import add_outlines, make_outliner
        stats = add_outlines(video_path, boxes, engine.outline_pairs,
                             outliner=outliner or make_outliner(engine.outliner),
                             cancel=cancel)
        report["outlines"] = stats
        if stats["frames"]:
            log(f"✏️ 轮廓（{stats['outliner']}）：在 {stats['frames']} 帧中描绘 {stats['traced']} 个，"
                f"{stats['no_outline']} 个保留为边界框")
    if engine.keypoint_pairs and not (cancel is not None and cancel.is_set()):
        from modules.vision.keypoints import add_keypoints
        from modules.vision.outlines import wanted
        # Only load the pose model when some frame actually needs it.
        if any(wanted(entry, engine.keypoint_pairs) for entry in boxes):
            estimator = (pose_factory or default_pose_estimator)()
            if estimator is None:
                log("⚠️ 规则引用了身体部位，但当前没有可用的姿态模型；"
                    "这些规则无法触发。")
            else:
                stats = add_keypoints(video_path, boxes, engine.keypoint_pairs,
                                      estimator, cancel=cancel)
                report["keypoints"] = stats
                if stats["frames"]:
                    log(f"🦴 姿态：为身体部位规则在 {stats['frames']} 帧中检测到 "
                        f"{stats['people']} 人")
    return report
