"""What happened, in what order, how far apart — and what the run cannot say.

Composition rules test *state*: whether something holds at a second. That is
most of what a report needs and it cannot express the other part, which is
**order and interval**. "A was present, then B, four seconds later" is not a
rule about a second; it is a relation between two of them, and nothing here
could state it before this module.

Order matters wherever a sequence is the subject rather than a tally. The
interval between two observations is frequently the whole question, and it is
exactly what is tedious to recover by hand from a report that lists moments
independently.

The second half is the part that took a session to learn to want.

Why a "not established" list ships with the findings
----------------------------------------------------

A report that lists what it found, and stops, invites the reader to treat its
silence as absence. It is not: it is the boundary of what ran. Worse, a model
asked to summarise such a report will fill the silence, confidently and in the
same voice as the measurements — which is how a run that measured six things
becomes a paragraph asserting a dozen.

So every result carries the questions this run **structurally cannot answer**,
stated as plainly as the ones it can. Some of those are permanent:

- *what anyone knew, intended, perceived or felt.* Nothing observable from
  outside a person establishes it, and a classifier's label for an appearance
  is not a reading of a mind.
- *whether a thing is what it looks like.* A detector reports appearance. What
  something actually was is a fact about the world, not about the pixels.
- *why one thing followed another.* Two observations in order are two
  observations in order. Sequence is not cause, and this module reports the
  interval precisely so that the reader supplies the rest.
- *anything outside the frame or outside the analysed span.*

Others are conditional on what actually ran, and are computed rather than
recited: a detector with no activity establishes nothing about its subject, and
saying which ones were silent is more useful than an empty section.

The voiceover check
-------------------

Speech is treated as evidence about the material, which is wrong when the audio
is *about* the material instead of from within it. Narration, commentary and
dubbing all read to a transcript exactly like dialogue. The heuristic is
unglamorous and works: one speaker holding nearly the whole runtime, at an even
rate, is far more likely to be describing the footage than to be in it. When
that trips, the quotes are reported as description of the material rather than
as something that happened in it, and "what was said within the material"
joins the not-established list.

That distinction changes what a report means. A narrated clip can assert
outcomes, names and conclusions that no detector saw and that the footage does
not show, and without the check those sentences sit beside the measurements
looking like more of them.

Scope
-----

Built from the report's kept moments, so the sequence covers the analysed
selection rather than every second of the source. Where coverage is partial the
result says so, because "first seen at" means "first seen in what was kept".
"""
from __future__ import annotations

import json
from typing import Mapping, Optional, Sequence

# One speaker holding at least this share of the runtime, with no other speaker
# of substance, reads as narration rather than as dialogue.
_VOICEOVER_SHARE = 90.0
_VOICEOVER_MIN_SECONDS = 20.0

# Claims no arrangement of these signals supports. Kept as text rather than as
# rules because the point is to be read by a person, and because a mechanism
# that could decide them would not need the list.
_ALWAYS_UNESTABLISHED = (
    "任何人知道、意图、感知或感受到什么——没有检测器能够观察人的内在状态，"
    "对外观的标签也不能等同于读取心理状态",
    "画面中的事物是否确实如其外观所示——检测器只能报告外观，"
    "事物真实是什么属于现实事实，不能仅由像素确认",
    "为什么一个观察结果会跟在另一个之后——这里测量的是顺序和间隔，"
    "两者都不能自动推导出因果关系",
    "画面范围之外，或本次分析时间范围之外的任何内容",
)


def _windows(report: Mapping) -> list:
    """(start, end, labels) for every kept moment, in time order."""
    out = []
    for entry in (report.get("segments") or []):
        try:
            start = float(entry.get("start"))
            end = float(entry.get("end"))
        except (TypeError, ValueError):
            continue
        labels = set()
        for obj in (entry.get("objects") or []):
            labels.add(("object", str(obj)))
        for ev in (entry.get("events") or []):
            labels.add(("event", str(ev)))
        for act in (entry.get("actions") or []):
            name = str((act or {}).get("name") or "").strip()
            if name:
                labels.add(("action", name))
        for sig in (entry.get("signals_present") or []):
            labels.add(("signal", str(sig)))
        out.append((start, end, labels))
    out.sort(key=lambda w: w[0])
    return out


def _timestamp(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _voiceover(report: Mapping) -> Optional[dict]:
    """The speech, when it reads as narration about the material. Else ``None``."""
    speech = report.get("speech") or {}
    speakers = speech.get("speakers") or []
    share = float(speech.get("speech_share_pct") or 0.0)
    seconds = float(speech.get("speech_seconds") or 0.0)
    if len(speakers) != 1 or share < _VOICEOVER_SHARE or seconds < _VOICEOVER_MIN_SECONDS:
        return None
    only = speakers[0] or {}
    return {
        "speaker": str(only.get("speaker") or "one speaker"),
        "share_pct": round(share, 1),
        "seconds": round(seconds, 1),
        "words": int(only.get("words") or speech.get("words") or 0),
    }


def findings(report: Mapping, *, kinds: Sequence[str] = ("object", "event", "action")) -> dict:
    """Conditions in order of first appearance, with intervals and limits.

    ``kinds`` selects which label families count as conditions. Signals are
    excluded by default: "motion_peak was present" is a statement about the
    scoring, not about the material, and mixing the two produces a sequence in
    which the detector's own behaviour appears as an event.
    """
    windows = _windows(report)
    seen: dict = {}
    for start, end, labels in windows:
        for kind, name in labels:
            if kind not in kinds:
                continue
            key = (kind, name)
            rec = seen.setdefault(key, {"kind": kind, "name": name,
                                        "first": start, "last": end, "windows": 0})
            rec["first"] = min(rec["first"], start)
            rec["last"] = max(rec["last"], end)
            rec["windows"] += 1

    ordered = sorted(seen.values(), key=lambda r: r["first"])
    for i, rec in enumerate(ordered):
        rec["at"] = _timestamp(rec["first"])
        rec["until"] = _timestamp(rec["last"])
        rec["since_previous_s"] = (None if i == 0 else
                                   round(rec["first"] - ordered[i - 1]["first"], 2))

    # Coverage, so "first seen at" is read as "first seen in what was kept".
    duration = float((report.get("video") or {}).get("duration") or 0.0)
    kept = sum(max(0.0, e - s) for s, e, _ in windows)
    coverage = round(kept / duration * 100, 1) if duration else 0.0

    unestablished = list(_ALWAYS_UNESTABLISHED)

    speech = report.get("speech") or {}
    narration = _voiceover(report)
    if narration:
        unestablished.insert(0,
            "素材中说了什么——音频中单一说话者覆盖了 "
            f"{narration['share_pct']}% 的时长，更像是在对素材进行旁白说明，"
            "而不是素材现场内的声音。其陈述应视为说话者对素材的描述，"
            "不是本次分析直接观察到的事实")
    elif not speech.get("segments"):
        unestablished.insert(0, "说了什么——本次没有转录到语音")

    activity = ((report.get("settings") or {}).get("detector_activity") or {})
    silent = sorted(k for k, v in activity.items() if not v)
    if silent:
        unestablished.append(
            "未产生有效结果的检测器本应覆盖的内容：" + "，".join(silent)
            + "。这些检测器可能未运行、权重为零或没有结果；这并不能证明不存在相应内容")

    if coverage < 99.0:
        unestablished.append(
            f"源视频中未保留的 {round(100 - coverage, 1)}% 内容；"
            "下方顺序仅代表已保留片段内部的顺序")

    return {
        "conditions": ordered,
        "coverage_pct": coverage,
        "narration": narration,
        "not_established": unestablished,
    }


def summarise(result: Mapping) -> str:
    """The findings as text, limits included rather than appended."""
    if not result:
        return "序列分析：未测量。"
    lines = []
    conditions = result.get("conditions") or []
    if not conditions:
        lines.append("保留片段中没有检测到已标注条件，因此没有可报告的事件序列。")
    else:
        lines.append(f"保留片段中共检测到 {len(conditions)} 个条件，按首次出现顺序排列"
                     f"（保留源视频的 {result.get('coverage_pct', 0)}%）：")
        for rec in conditions:
            gap = ("" if rec["since_previous_s"] is None
                   else f"  （距上一个 +{rec['since_previous_s']:g} 秒）")
            lines.append(f"  {rec['at']}–{rec['until']}  {rec['kind']}: "
                         f"{rec['name']}  ×{rec['windows']}{gap}")
    narration = result.get("narration")
    if narration:
        lines.append("")
        lines.append(f"音频更像旁白：单一说话者覆盖 {narration['share_pct']}% 的时长"
                     f"（{narration['words']} 个词）。这些陈述应视为说话者对素材的描述，"
                     f"而不是本次分析直接得出的结论。")
    lines.append("")
    lines.append("本次分析无法确认：")
    for item in (result.get("not_established") or []):
        lines.append(f"  - {item}")
    return "\n".join(lines)


def _quotes(report: Mapping, limit: int = 12) -> list:
    """Every transcribed line in the kept moments, in time order, deduplicated."""
    seen, out = set(), []
    for entry in (report.get("segments") or []):
        for line in ((entry.get("speech") or {}).get("lines") or []):
            text = " ".join(str(line.get("text") or "").split())
            if not text or text in seen:
                continue
            seen.add(text)
            out.append({"at": str(line.get("timestamp") or ""),
                        "start": float(line.get("start") or 0.0),
                        "speaker": str(line.get("speaker") or ""),
                        "text": text})
    out.sort(key=lambda q: q["start"])
    return out[:limit]


def closing_summary(report: Mapping) -> dict:
    """What the run established, what was merely asserted, and what neither.

    Three parts kept apart on purpose. A summary that runs them together is how
    a sentence somebody said becomes a sentence the report appears to stand
    behind — and on narrated material almost everything of consequence is in
    the second part while almost nothing is in the first.

    It states no conclusion. The interval between two observations is reported
    because a reader needs it; what it implies is theirs, and no arrangement of
    what is measured here reaches a judgement about anyone's conduct.
    """
    found = findings(report)
    quotes = _quotes(report)
    narration = found.get("narration")

    established = []
    for rec in found.get("conditions") or []:
        gap = ("" if rec["since_previous_s"] is None
               else f"，距上一个 {rec['since_previous_s']:g} 秒")
        established.append(
            f"{rec['name']}（{rec['kind']}），时间 {rec['at']}–{rec['until']}，"
            f"在保留片段中出现 {rec['windows']} 次{gap}")

    if established:
        first = "本次分析观察到：" + "；".join(established) + "。"
    else:
        first = "本次分析在保留片段中没有观察到可由现有标签描述的内容。"

    if quotes and narration:
        second = (f"下方其他内容来自一位覆盖 {narration['share_pct']}% 视频时长的说话者。"
                  f"这是该说话者对素材的描述，不是本次分析直接观察到的事实，"
                  f"也可能描述画面之外的事件。")
    elif quotes:
        second = "以下内容来自素材语音转录。它代表“说了什么”，并不等同于“发生了什么”。"
    else:
        second = "没有转录到语音内容。"

    return {
        "observed": established,
        "observed_sentence": first,
        "quoted_note": second,
        "quotes": quotes,
        "not_established": found.get("not_established") or [],
        "coverage_pct": found.get("coverage_pct", 0.0),
        "narration": narration,
    }


def attach(report: dict) -> dict:
    """Compute the findings and the closing summary, and store both."""
    report["sequence_findings"] = findings(report)
    report["closing_summary"] = closing_summary(report)
    return report["sequence_findings"]


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        prog="python -m modules.report.sequence_findings",
        description=__doc__.splitlines()[0])
    ap.add_argument("report", help="剪辑文件旁生成的 *_why.json 报告")
    ap.add_argument("--attach", action="store_true",
                    help="将分析结果写回报告文件")
    args = ap.parse_args(argv)

    with open(args.report, encoding="utf-8") as fh:
        report = json.load(fh)
    result = findings(report)
    print(summarise(result))

    if args.attach:
        report["sequence_findings"] = result
        with open(args.report, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n已写回报告：{args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
