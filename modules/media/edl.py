"""
edl.py — the cut list: which piece of which file, in what order, joined how.

Why this exists
===============
:mod:`modules.segments.script_plan` says what the film should *contain* ("three action
beats of about eight seconds"). It is an intention, and the engine answers it
by scoring. That is the right shape for the first pass and the wrong shape for
the second, because once you have watched the result the note you have is never
"make beat two more actiony" — it is "that third clip should start two seconds
later, and lose the one after it".

An EDL is that second shape: explicit sources and explicit timestamps. It is
what the automatic pass emits, so the machine's answer arrives as an editable
document rather than as a finished file you can only accept or re-roll. Edit
the numbers, render again, and only the changed part of your intent moves.

Timestamps
==========
Written the way people write them — ``8``, ``0:08``, ``1:23.5``, ``01:02:03.5``
— and always emitted as ``M:SS.s`` (or ``H:MM:SS.s`` past an hour), because a
list of raw seconds is unreadable at exactly the moment you most need to read
it. Parsing is strict about nonsense (``1:2:3:4``, ``abc``, negative times) and
liberal about form.

The transition on a cut is the one *leaving* it, toward the next cut. The last
cut's transition is therefore meaningless and is ignored rather than rejected —
deleting the final clip of a reel should not also require you to remember to
clear a field on the one before it.

Round trip
==========
    run the pipeline -> edl_from_clips(...) -> film.edl.yaml   (edit it)
                     -> load_edl(...) -> render_edl(...)       -> film.mp4

``render_edl`` cuts each source at its timestamps and hands the pieces to
:mod:`modules.media.transitions`, so everything that module knows about blending,
clamping and delivery size applies unchanged.

Public API
==========
    parse_time(text) -> float          format_time(seconds) -> str
    Cut / Edl
    parse_edl(text) / load_edl(path) / save_edl(edl, path)
    edl_from_clips(paths, ...) -> Edl
    validate_edl(edl) -> list[str]
    render_edl(edl, output, ...) -> str
    class EdlError(ValueError)
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field

EDL_VERSION = 1

_TOP_KEYS = {
    "version", "title", "music", "music_mode", "music_volume",
    "width", "height", "fps", "crf", "fill", "cuts",
}
_CUT_KEYS = {"source", "in", "out", "transition", "transition_duration",
             "easing", "feather", "motion", "label", "text"}

# Accepts S, S.s, M:SS, M:SS.s, H:MM:SS, H:MM:SS.s — and nothing else.
_TIME = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d+)?)$")


class EdlError(ValueError):
    """A cut list that cannot be acted on. Always names the cut and the field,
    because an EDL is a list of near-identical entries and "invalid duration"
    on its own sends you reading all of them."""


def parse_time(value) -> float:
    """Seconds from ``8``, ``0:08``, ``1:23.5`` or ``01:02:03.5``.

    Numbers pass through, so an EDL written by a machine (plain seconds) and
    one written by a person (clock time) both load.
    """
    if isinstance(value, bool):
        raise EdlError(f"需要时间戳，实际得到 {value!r}")
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds < 0 or seconds != seconds or seconds in (float("inf"), float("-inf")):
            raise EdlError(f"时间戳必须是有限的非负数，实际得到 {value!r}")
        return seconds
    match = _TIME.match(str(value).strip())
    if not match:
        raise EdlError(
            f"无法读取时间戳 {value!r}——请使用秒数（8.5）、"
            f"M:SS（1:23.5）或 H:MM:SS（1:02:03）格式")
    first, second, last = match.groups()
    parts = [p for p in (first, second) if p is not None]
    total = float(last)
    if len(parts) == 1:
        total += int(parts[0]) * 60
    elif len(parts) == 2:
        total += int(parts[0]) * 3600 + int(parts[1]) * 60
    return total


def format_time(seconds: float) -> str:
    """``M:SS.sss``, or ``H:MM:SS.sss`` once there is an hour to show.

    Millisecond precision with the trailing zeros trimmed, so a round number
    still reads as ``0:08`` while an awkward one keeps its digits.

    The precision is not decoration. A bar at 66 BPM is 3.63578 s; written to
    one decimal it becomes 3.6, and that 36 ms error compounds — twenty cuts
    later the reel is 1.7 s off the beat, which is half a bar. This function
    round-tripping a duration is what beat alignment rests on, and a cut list
    is a file that gets loaded back, not just displayed.
    """
    seconds = max(0.0, float(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{secs:06.3f}".rstrip("0").rstrip(".")
    if len(text.split(".")[0]) < 2:
        text = "0" + text
    if hours >= 1:
        return f"{int(hours)}:{int(minutes):02d}:{text}"
    return f"{int(minutes)}:{text}"


@dataclass
class Cut:
    """One piece of one source file, and how the reel leaves it."""
    source: str
    start: float = 0.0
    end: float = 0.0
    transition: str = "cut"
    transition_duration: float = 0.5
    # How the blend moves. Linear is what xfade does on its own; anything else
    # is what makes it look designed rather than applied.
    easing: str = "linear"
    # How soft the transition's edge is, 0..1 of its length. Only means
    # anything for a transition that has an edge — a wipe, an iris, blinds. 0
    # is the hard edge ffmpeg draws by default.
    feather: float = 0.0
    # A move on the ends of this cut — a punch, a shake, a roll. Unlike a
    # transition, which lives on the join, this belongs to the clip.
    motion: str = "none"
    label: str = ""
    # Burnt into the picture for the length of this cut. Short-form video is
    # mostly watched muted, so the opening line has to be readable rather than
    # spoken — which makes this part of the edit, not a decoration on it.
    text: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Edl:
    title: str = "Untitled"
    cuts: list[Cut] = field(default_factory=list)
    music: str = ""
    music_mode: str = "replace"
    music_volume: float = 0.8
    width: int = 0
    height: int = 0
    fps: int = 0
    crf: int = 20
    # "pad" keeps the whole frame; "crop" fills the canvas. Vertical Reels
    # want crop, or 16:9 footage becomes a strip in a black screen.
    fill: str = "pad"

    @property
    def source_duration(self) -> float:
        """Total length of the pieces, before transitions absorb any of it."""
        return sum(c.duration for c in self.cuts)

    @property
    def duration(self) -> float:
        """What the finished reel will actually run to.

        Every transition overlaps two clips, so the reel is shorter than the
        sum of its cuts. Reporting the sum instead is how a 90-second target
        quietly delivers 84.

        The overlap is the *clamped* length, not the requested one, because
        that is what the renderer will use: a transition may not eat more than
        a third of either cut it joins, and one shorter than a couple of frames
        degrades to a hard cut with no overlap at all. Subtracting the raw
        request instead reports a two-second dissolve between two one-second
        shots as taking two seconds out of the reel, when the renderer will
        only ever take a third of a second — which made a reel of long
        transitions read as several seconds shorter than it renders.
        """
        from modules.media.transitions import MAX_CLIP_FRACTION, MIN_DURATION

        overlap = 0.0
        for cut, following in zip(self.cuts, self.cuts[1:]):
            if cut.transition == "cut":
                continue
            room = min(cut.duration, following.duration) * MAX_CLIP_FRACTION
            held = min(float(cut.transition_duration), room)
            if held >= MIN_DURATION:
                overlap += held
        return max(0.0, self.source_duration - overlap)


def _require_mapping(data, what):
    if not isinstance(data, dict):
        raise EdlError(f"{what} 必须是映射对象，实际得到 {type(data).__name__}")


def _reject_unknown(mapping, allowed, what):
    """A misspelled key that is silently ignored is the failure mode that costs
    an afternoon: the render succeeds and quietly ignores half the edit."""
    import difflib

    for key in mapping:
        if key not in allowed:
            near = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f"——你是不是想写 {near[0]!r}？" if near else ""
            raise EdlError(f"{what} 中存在未知字段 {key!r}{hint}")


def parse_edl(data) -> Edl:
    """Build an :class:`Edl` from YAML text or an already-parsed mapping."""
    if isinstance(data, str):
        import yaml
        try:
            data = yaml.safe_load(data)
        except Exception as exc:
            raise EdlError(f"剪辑列表不是有效的 YAML：{exc}") from exc
    if data is None:
        raise EdlError("剪辑列表为空——请先创建剪辑内容")
    _require_mapping(data, "剪辑列表")
    _reject_unknown(data, _TOP_KEYS, "顶层")

    version = data.get("version", EDL_VERSION)
    if int(version) != EDL_VERSION:
        raise EdlError(f"不支持剪辑列表版本 {version!r} "
                       f"（当前版本支持 {EDL_VERSION}）")

    raw_cuts = data.get("cuts")
    if raw_cuts is None:
        raise EdlError("剪辑列表必须包含 'cuts:' 列表——没有镜头就无法渲染")
    if not isinstance(raw_cuts, list):
        raise EdlError(f"'cuts' 必须是列表，实际得到 {type(raw_cuts).__name__}")
    if not raw_cuts:
        raise EdlError("剪辑列表至少需要一个镜头")

    from modules.media.motion import normalise_motion
    from modules.media.transitions import normalise_feather, normalise_kind

    cuts: list[Cut] = []
    for i, entry in enumerate(raw_cuts, start=1):
        if not isinstance(entry, dict):
            raise EdlError(f"镜头 {i} 必须是映射对象，并至少包含 source "
                           f"和 out 时间，实际得到 {entry!r}")
        _reject_unknown(entry, _CUT_KEYS, f"镜头 {i}")

        source = entry.get("source")
        if not source or not str(source).strip():
            raise EdlError(f"镜头 {i} 没有源文件")

        try:
            start = parse_time(entry.get("in", 0))
            end = parse_time(entry["out"]) if "out" in entry else 0.0
        except EdlError as exc:
            raise EdlError(f"镜头 {i}（{source}）：{exc}") from None
        if "out" not in entry:
            raise EdlError(f"镜头 {i}（{source}）没有 'out' 时间")
        if end <= start:
            raise EdlError(
                f"镜头 {i}（{source}）范围为 {format_time(start)}..{format_time(end)}，"
                f"'out' 必须晚于 'in'")

        kind = entry.get("transition", "cut")
        try:
            kind = normalise_kind(kind)
        except ValueError as exc:
            raise EdlError(f"镜头 {i}（{source}）：{exc}") from None

        try:
            hold = float(entry.get("transition_duration", 0.5))
        except (TypeError, ValueError):
            raise EdlError(f"镜头 {i}（{source}）：transition_duration 必须是秒数，实际得到 "
                           f"{entry.get('transition_duration')!r}") from None
        if hold < 0 or hold != hold:
            raise EdlError(f"镜头 {i}（{source}）：transition_duration 必须是有限的非负秒数")

        try:
            soft = normalise_feather(entry.get("feather", 0.0))
        except ValueError as exc:
            raise EdlError(f"镜头 {i}（{source}）：{exc}") from None

        try:
            move = normalise_motion(entry.get("motion", "none"))
        except ValueError as exc:
            raise EdlError(f"镜头 {i}（{source}）：{exc}") from None

        cuts.append(Cut(source=str(source), start=start, end=end,
                        transition=kind, transition_duration=hold,
                        easing=str(entry.get("easing", "linear") or "linear"),
                        feather=soft, motion=move,
                        label=str(entry.get("label", "") or ""),
                        text=str(entry.get("text", "") or "")))

    def _int(key):
        try:
            return int(data.get(key, 0) or 0)
        except (TypeError, ValueError):
            raise EdlError(f"{key} 必须是整数，实际得到 {data.get(key)!r}") from None

    try:
        volume = float(data.get("music_volume", 0.8))
    except (TypeError, ValueError):
        raise EdlError(f"music_volume 必须是 0 到 1 之间的数字，实际得到 "
                       f"{data.get('music_volume')!r}") from None

    return Edl(
        title=str(data.get("title", "未命名") or "未命名"),
        cuts=cuts,
        music=str(data.get("music", "") or ""),
        music_mode=str(data.get("music_mode", "replace") or "replace"),
        music_volume=min(1.0, max(0.0, volume)),
        width=_int("width"), height=_int("height"), fps=_int("fps"),
        crf=_int("crf") or 20,
    )


def load_edl(path: str) -> Edl:
    """Read a cut list from disk.

    Every failure arrives as :class:`EdlError` — including a file that is not
    UTF-8, which would otherwise escape as ``UnicodeDecodeError`` and lose the
    file name from the message.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except UnicodeDecodeError as exc:
        raise EdlError(f"无法读取剪辑列表 {os.path.basename(path)}："
                       f"文件不是 UTF-8 文本（{exc.reason}）") from None
    except OSError as exc:
        raise EdlError(f"无法读取剪辑列表 {path}：{exc}") from None
    try:
        return parse_edl(text)
    except EdlError as exc:
        raise EdlError(f"{os.path.basename(path)}: {exc}") from None


def save_edl(edl: Edl, path: str) -> str:
    """Write a cut list a person can edit and return its path.

    Hand-written by design rather than dumped through yaml: the timestamps must
    come out as ``1:23.5`` rather than as floats, defaults are omitted so the
    interesting fields are the visible ones, and the header explains what the
    numbers do — this file is meant to be opened.
    """
    lines = [
        f"# {edl.title} — cut list",
        "#",
        "# Each entry is one piece of one file. 'in' and 'out' are times in that",
        "# source; 'transition' is how the reel leaves this cut for the next one.",
        "# Times: 8, 0:08, 1:23.5 or 1:02:03. Edit and render again.",
        "#",
        f"# {len(edl.cuts)} cut(s), {format_time(edl.source_duration)} of footage,",
        f"# {format_time(edl.duration)} after transitions.",
        "",
        f"version: {EDL_VERSION}",
        f"title: {edl.title}",
    ]
    if edl.music:
        lines.append(f"music: {edl.music}")
        lines.append(f"music_mode: {edl.music_mode}")
        lines.append(f"music_volume: {edl.music_volume:g}")
    if edl.width and edl.height:
        lines.append(f"width: {edl.width}")
        lines.append(f"height: {edl.height}")
    if edl.fps:
        lines.append(f"fps: {edl.fps}")
    if edl.crf and edl.crf != 20:
        lines.append(f"crf: {edl.crf}")
    if edl.fill != "pad":
        lines.append(f"fill: {edl.fill}")

    lines.append("")
    lines.append("cuts:")
    for i, cut in enumerate(edl.cuts):
        last = i == len(edl.cuts) - 1
        lines.append(f"  - source: {cut.source}")
        lines.append(f"    in: {format_time(cut.start)}")
        lines.append(f"    out: {format_time(cut.end)}")
        if not last:
            lines.append(f"    transition: {cut.transition}")
            if cut.transition != "cut":
                lines.append(f"    transition_duration: {cut.transition_duration:g}")
                if cut.easing != "linear":
                    lines.append(f"    easing: {cut.easing}")
                if cut.feather:
                    lines.append(f"    feather: {cut.feather:g}")
        if cut.motion and cut.motion != "none":
            lines.append(f"    motion: {cut.motion}")
        if cut.label:
            lines.append(f"    label: {cut.label}")
        if cut.text:
            # Quoted: a caption routinely contains a colon, which YAML would
            # otherwise read as the start of a nested mapping.
            escaped = cut.text.replace('"', '\\"')
            lines.append(f'    text: "{escaped}"')
    text = "\n".join(lines) + "\n"

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def edl_from_clips(paths, *, title: str = "Untitled", transition: str = "cut",
                   transition_duration: float = 0.5, music: str = "",
                   probe: bool = True) -> Edl:
    """Turn finished clips into an editable cut list.

    This is the round trip: the pipeline produces clip files, and this describes
    them as cuts so the next render can be adjusted rather than re-rolled. Each
    clip becomes a whole-file cut (``in`` 0 to its duration), which is exactly
    what it is at that point.
    """
    from modules.media.transitions import normalise_kind

    kind = normalise_kind(transition)
    cuts: list[Cut] = []
    for path in paths or []:
        end = 0.0
        if probe:
            try:
                from modules.media.video_probe import probe_video
                end = float(probe_video(path)["duration"])
            except Exception:
                end = 0.0
        cuts.append(Cut(source=str(path), start=0.0, end=end, transition=kind,
                        transition_duration=transition_duration,
                        label=os.path.splitext(os.path.basename(str(path)))[0]))
    if cuts:
        cuts[-1].transition = "cut"
    return Edl(title=title, cuts=cuts, music=music)


def quantise_to_music(edl: Edl, analysis, *, unit: str = "bar",
                      min_units: int = 1, transition_bars: float = 0.0,
                      log_fn=print) -> Edl:
    """Round every cut to a whole number of bars (or beats) of ``analysis``.

    This is what actually makes an edit land on the music, and it is worth
    being precise about why it works on *durations* rather than on positions.
    A reel is played back to back, so the time a cut happens is the sum of
    every clip before it. Snapping each clip's start to the nearest beat of the
    track would therefore snap nothing — the clip does not begin where its
    source timestamp says, it begins wherever the previous clips ended. What
    has to be a whole number of bars is the *advance*: how far the reel moves
    on per clip.

    With hard cuts the advance is the clip length and one bar per clip is
    right. With a transition it is not, because the two clips overlap: the
    advance is ``duration - transition``, so a clip of exactly one bar advances
    the reel by less than a bar and every join after the first drifts further
    off. Measured on a real render — 3.64 s bars, 0.6 s crossfades — the third
    join landed 1.8 s off the beat, which is half a bar and audible as being
    simply wrong. Each clip therefore gets its own transition added back on
    top, so ``duration - transition`` is the bar and every transition *starts*
    exactly on a downbeat.

    A cut is never lengthened past the end of its source. When the nearest
    whole number of units would run off the end, the *next lower* whole number
    is used rather than the leftover footage: a 6.03 s clip against a 3.64 s
    bar rounds to two bars, cannot have them, and becomes one bar — not 6.03 s,
    which is what simply clamping gives and is not on the beat at all. Losing
    two seconds of a clip is the price of the cut landing where the music says.
    Only a clip with less than one whole unit available keeps its own length
    and breaks the pattern, because the alternative is a clip too short to see.

    ``transition_bars`` also sets each transition's length from the bar, since a
    blend that is a musical fraction is the other half of the same idea. Left
    at 0 the existing durations are untouched.

    Returns a new :class:`Edl`; the input is not modified.
    """
    interval = float(getattr(analysis, "beat_interval", 0.0) or 0.0)
    meter = int(getattr(analysis, "meter", 4) or 4)
    if interval <= 0:
        log_fn("⚠️ 没有可用的节拍信息——保留原始剪切位置")
        return Edl(**{**edl.__dict__, "cuts": list(edl.cuts)})

    step = interval * meter if unit == "bar" else interval
    if unit not in ("bar", "beat"):
        raise EdlError(f"未知节拍对齐单位 {unit!r}（应为 'bar' 或 'beat'）")

    durations: dict[str, float] = {}

    def available(path: str) -> float:
        if path not in durations:
            try:
                from modules.media.video_probe import probe_video
                durations[path] = float(probe_video(path)["duration"])
            except Exception:
                durations[path] = 0.0
        return durations[path]

    hold = max(0.0, step * float(transition_bars)) if transition_bars else 0.0
    cuts: list[Cut] = []
    trimmed = 0
    unblended = 0
    last = len(edl.cuts) - 1
    for index, cut in enumerate(edl.cuts):
        blend = hold or cut.transition_duration
        kind = cut.transition
        # Nothing follows the last clip, and a hard cut has no overlap, so
        # neither gets the extra.
        overlap = blend if (index < last and kind != "cut") else 0.0

        limit = available(cut.source)
        room = max(0.0, limit - cut.start) if limit else 0.0
        units = max(int(min_units), round((cut.duration - overlap) / step)) if step else 0

        if room:
            # Drop whole units until it fits, rather than clamping to the
            # leftover: a clamped length is off the grid, which is the one
            # thing this function exists to prevent.
            while units > 1 and units * step + overlap > room + 0.01:
                units -= 1
            if overlap and units * step + overlap > room + 0.01 \
                    and units * step <= room + 0.01:
                # Room for the bar but not the blend on top. Cutting hard is
                # better than either breaking the grid or losing a whole bar:
                # this is exactly the case of a 4 s clip against a 3.64 s bar,
                # and letting it keep its raw length puts every later cut off
                # the beat.
                kind, overlap = "cut", 0.0
                unblended += 1

        target = units * step + overlap
        if room and target > room + 0.01:
            # Not even one whole unit. Keep the footage — a clip shorter than
            # a bar is still better than no clip.
            target = 0.0
        end = cut.start + (target if target > 0 else cut.duration)
        if abs(end - cut.end) > 0.01:
            trimmed += 1
        cuts.append(Cut(source=cut.source, start=cut.start, end=end,
                        transition=kind,
                        transition_duration=(hold or cut.transition_duration)
                        if kind != "cut" else cut.transition_duration,
                        easing=cut.easing, label=cut.label, text=cut.text))
    if unblended:
        log_fn(f"✂️ 有 {unblended} 个片段没有足够空间叠加转场，"
               f"为保持节拍网格已改为直接切换")

    log_fn(f"🎼 已将 {trimmed}/{len(cuts)} 个剪切对齐到 "
           f"{step:.2f} 秒 {'小节' if unit == 'bar' else '节拍'}（{60.0 / interval:.1f} BPM）")
    return Edl(**{**edl.__dict__, "cuts": cuts})


def validate_edl(edl: Edl) -> list[str]:
    """Non-fatal warnings — things worth saying before a long render.

    Kept separate from parsing because none of these make the cut list
    unrenderable, and refusing to render over a missing file the user is about
    to plug in would be obnoxious.
    """
    warnings: list[str] = []
    for i, cut in enumerate(edl.cuts, start=1):
        if not os.path.exists(cut.source):
            warnings.append(f"镜头 {i}：磁盘上找不到 {os.path.basename(cut.source)}")
            continue
        if cut.duration < 0.5:
            warnings.append(
                f"镜头 {i}：时长 {format_time(cut.duration)} 不足半秒，画面几乎无法辨认")
        try:
            from modules.media.video_probe import probe_video
            available = float(probe_video(cut.source)["duration"])
        except Exception:
            continue
        if available and cut.end > available + 0.05:
            warnings.append(
                f"镜头 {i}：out 时间 {format_time(cut.end)} 超过了 "
                f"{os.path.basename(cut.source)} 的结尾（{format_time(available)}）")
    if edl.music and not os.path.exists(edl.music):
        warnings.append(f"磁盘上找不到音乐文件：{edl.music}")
    return warnings


def _draw_overlays(edl: Edl, built: str, overlays, track: str, temp_dir: str,
                   *, log_fn=print, cancel_check=None) -> str:
    """Draw the named graphics over a finished reel, in place.

    Every failure costs the graphics and keeps the reel: the expensive work is
    all behind us by this point, and throwing away a finished render because a
    GPX file was malformed would be absurd.
    """
    try:
        from modules.media.overlay import build_scene, burn_overlay, make_elements
        from modules.segments.shot_place import locate, read_track
        from modules.media.video_probe import probe_video

        gps = read_track(track, log_fn=log_fn) if track else None
        places = locate([c.source for c in edl.cuts], track=gps, log_fn=log_fn)
        # The delivery size and frame rate are whatever the build settled on,
        # not what the cut list asked for — it may have left them at 0.
        info = probe_video(built)
        scene = build_scene(edl, gps, places,
                            width=int(info.get("width") or 0),
                            height=int(info.get("height") or 0))
        elements = make_elements(overlays, scene, gps)
        if not elements:
            log_fn("⚠️ 当前素材无法绘制所选图形")
            return built

        drawn = os.path.join(temp_dir, "overlaid.mp4")
        burn_overlay(built, drawn, scene, elements,
                     fps=int(round(float(info.get("fps") or 30))) or 30,
                     log_fn=log_fn, cancel_check=cancel_check)
        if os.path.exists(drawn) and os.path.getsize(drawn) > 0:
            shutil.move(drawn, built)
    except Exception as exc:  # noqa: BLE001
        log_fn(f"⚠️ 无法绘制图形（{exc}）；"
               f"成片将不包含这些图形")
    return built


def render_edl(edl: Edl, output: str, *, mode: str = "gpu",
               music_optional: bool = False, overlays=None, track: str = "",
               log_fn=print, progress_fn=None, cancel_check=None) -> str:
    """Cut every entry from its source and join them into ``output``.

    The pieces are extracted into a temp directory and removed afterwards: an
    EDL render is reproducible from the cut list plus the sources, so keeping
    the intermediates costs disk for something that can always be rebuilt.

    ``overlays`` names the graphics to draw over the finished reel (see
    :mod:`modules.media.overlay`) and ``track`` is an optional GPX file they read
    from. They are applied *after* the reel is assembled rather than per clip,
    because their timing is expressed in reel seconds — an elevation profile
    that fills as the reel plays has to know how long the reel turned out to
    be, which is only settled once the transitions have taken their share.
    """
    if not edl.cuts:
        raise EdlError("没有可渲染内容——剪辑列表为空")
    missing = [c.source for c in edl.cuts if not os.path.exists(c.source)]
    if missing:
        raise EdlError("无法渲染，以下源文件缺失："
                       + "，".join(os.path.basename(m) for m in dict.fromkeys(missing)))

    from modules.media.transitions import ReelCancelled, Transition, build_reel
    from modules.media.video_cutter import cut_video

    temp_dir = tempfile.mkdtemp(prefix="vh_edl_")
    try:
        pieces: list[str] = []
        total = len(edl.cuts)
        for i, cut in enumerate(edl.cuts):
            if cancel_check is not None and cancel_check():
                raise ReelCancelled("已取消")
            if progress_fn:
                try:
                    progress_fn(i, total + 1, "正在剪切", f"镜头 {i + 1}/{total}")
                except Exception:
                    pass
            piece = os.path.join(temp_dir, f"cut{i:03d}.mp4")
            log_fn(f"✂️ 剪切 {i + 1}/{total}：{os.path.basename(cut.source)} "
                   f"{format_time(cut.start)}–{format_time(cut.end)}")
            cut_video(cut.source, cut.start, cut.end, piece, mode=mode)
            pieces.append(piece)

        transitions = [
            Transition(index=i, kind=c.transition,
                       duration=c.transition_duration, easing=c.easing,
                       feather=c.feather)
            for i, c in enumerate(edl.cuts[:-1])
        ]
        music = ({"path": edl.music, "mode": edl.music_mode,
                  "volume": edl.music_volume} if edl.music else None)

        texts = {i: c.text for i, c in enumerate(edl.cuts) if c.text.strip()}
        # A motion belongs to a clip rather than to a join, so it is passed
        # per index rather than as one setting for the reel.
        motions = {i: c.motion for i, c in enumerate(edl.cuts)
                   if c.motion and c.motion != "none"}
        built = build_reel(pieces, output, transitions=transitions,
                           motions=motions or None,
                           width=edl.width, height=edl.height, fps=edl.fps,
                           crf=edl.crf, music=music,
                           music_optional=music_optional, texts=texts,
                           fill=edl.fill,
                           log_fn=log_fn, progress_fn=progress_fn,
                           cancel_check=cancel_check)
        if overlays:
            built = _draw_overlays(edl, built, overlays, track, temp_dir,
                                   log_fn=log_fn, cancel_check=cancel_check)
        return built
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
