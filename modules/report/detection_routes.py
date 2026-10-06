"""What it would take to measure something this run had no signal for.

:mod:`modules.report.vocabulary_gap` and :mod:`modules.report.uncovered_claims` both end in
the same place: *this was said, and nothing here was watching for it*. The
question a user asks next is not "which weight do I change" — no weight helps —
but **"so how would I detect it, and is it worth the trouble?"** That question
has a small number of real answers in this application, they differ by an order
of magnitude in cost, and choosing badly between them is expensive: teaching a
category takes minutes and training a class takes an afternoon, and people
reliably reach for the second when the first would have done.

So the answers are enumerated here, with what each one *gives* you and what it
*costs*, and two of them are picked: the cheapest thing that would actually
measure it, and the most reliable thing available. Two rather than a list,
because a list of six routes is a decision handed back to the person who asked.

What this module does not know
------------------------------

**What was said.** It is never given the claim, and could not use it — deciding
that a spoken word describes an object rather than a movement is a judgement
about subject matter, and this repo ships no vocabulary to make it with (see
CLAUDE.md, and :mod:`modules.report.vocabulary_gap` for the same refusal). So a route
is not selected by what the thing *is*. Each one instead carries the condition
it holds under, in the user's own terms — "if you can point at frames showing
it", "if it fills a good part of the frame" — and the reader, who can see their
own footage, settles in a second what no lexicon here could settle at all.

**Whether it will work.** Every figure below is a cost, not a promise. The one
route whose success is genuinely unpredictable — an open-vocabulary query, which
is excellent on ordinary things and can be nearly blind on specialised subject
matter — is therefore offered as a *probe* rather than as a recommendation: five
minutes with a control query says whether it can see your subject at all, and
that answer is worth having before committing to an afternoon of labelling.

What it does know is which routes this particular run has the prerequisites for,
and that is read from the record: how many classes the detector produced (a
composition rule needs at least two to arrange), whether a CLIP index was ever
built (chapters say so — their `method` is "visual" when one was), whether a
face scan ran. A route offered without its prerequisite is a route the user
discovers is unavailable after deciding on it.

And which engines this *build* has at all. Each route names the module it needs
and is dropped when that module is not importable, which is what lets one file
serve both editions honestly: the editions ship different engines, and a route
recommending something this build cannot run is the same failure as one
recommending a class the detector never emits — it costs the user a decision
before they find out. Detected rather than declared, so no edition flag has to
be threaded here and nothing goes stale when the boundary moves.

The costs quoted here come from ``docs/DETECTION-GUIDE.md`` and are measured,
not estimated. When they change, change them there and here together.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Sequence

# Effort and confidence are ordinals, not scores. They exist to be *compared* —
# "cheaper than", "more reliable than" — and any arithmetic on them would be
# inventing a precision that measuring a route's cost in minutes does not have.
EFFORT_INSTANT, EFFORT_MINUTES, EFFORT_HOURS, EFFORT_SESSION = 0, 1, 2, 3
CONFIDENCE_PROXY, CONFIDENCE_UNEVEN, CONFIDENCE_GOOD, CONFIDENCE_EXACT = 0, 1, 2, 3


@dataclass
class Route:
    """One way to get a signal the run does not have.

    ``holds_when`` and ``fails_when`` are the load-bearing fields. The cost of a
    route is a fact and the same for everybody; whether it applies depends on
    what the user is looking for, which only they can see, so every route states
    its condition instead of pretending the choice was made for them.
    """
    id: str
    name: str
    gives: str                 # what you get out of it
    effort: str                # what it costs, in time and what it needs
    effort_rank: int
    confidence: str            # how much the answer is worth
    confidence_rank: int
    holds_when: str            # when this is the right route
    fails_when: str            # when it is not, said before they spend the time
    repeat: str                # cost of asking a second question
    topic: str                 # page in docs/advisor that explains it
    needs: str = ""            # prerequisite in this run, "" when always usable
    module: str = ""           # engine this build must have, "" when always
    # The cheap test this route can be turned into, when it has one: not a way
    # to measure the thing, a way to find out in five minutes whether the
    # expensive route is unavoidable.
    probe: Optional[dict] = None

    def as_dict(self) -> dict:
        return asdict(self)


# Ordered as a person would work through them: cheapest first. `pick` does not
# rely on this order — it sorts — but a reader of `all` gets the sequence that
# the recommendation is drawn from rather than an arbitrary one.
ROUTES: tuple = (
    Route(
        id="compose",
        name="用本视频已经检测出的类别组合规则",
        gives="按秒生成标签和精确计数，还能表达评分无法表达的情况——某个对象并不存在",
        effort="编写规则只需几分钟，但需要重新运行一次物体检测；仅复用检测缓存会跳过构图引擎，新规则无法触发",
        effort_rank=EFFORT_MINUTES,
        confidence="适用时结果精确——检测框计数是确定性计算，不是相似度，因此无需调阈值",
        confidence_rank=CONFIDENCE_EXACT,
        holds_when="所描述的是检测器已能识别对象之间的组合关系，例如包含、同时出现或全部未出现",
        fails_when="所描述的是当前类别完全覆盖不到的新对象。规则不能凭空创造类别，只能组合已有类别",
        repeat="即时——规则直接读取已经缓存的检测结果",
        topic="composition",
        needs="本视频至少已检测到两个类别",
    ),
    Route(
        id="clip_search",
        name="按说出的文字内容搜索视频",
        gives="为每个采样帧给出与输入短语的视觉相似度评分；不生成检测框，因此不能直接计数",
        effort="通常几分钟。文字已在转录中；视频只需建立一次嵌入索引，此后的查询只是小数组运算，几乎没有额外成本",
        effort_rank=EFFORT_MINUTES,
        confidence="效果不均匀，因为它依赖文字表达方式。模型熟悉的描述通常效果好，陌生表达可能只得到没有意义的低分",
        confidence_rank=CONFIDENCE_UNEVEN,
        holds_when="目标能够用自然语言清楚描述；对于视频里口头提到的内容，这是最值得优先尝试的方法",
        fails_when="目标在画面中太小，或描述方式很特殊。嵌入描述的是整幅画面，若模型本身无法表示该目标，调阈值也无法补救",
        repeat="无需额外分析——索引只构建一次，之后可反复查询",
        topic="training",
        module="llm.clip_index",
        probe={
            "why": ("在投入时间做标注前，先用几分钟确认是否真的需要训练。"
                    "用视频中说出的文字进行搜索，同时再用一个你确定画面里存在的普通对象作为对照。"
                    "如果对照能找到而目标短语找不到，调阈值通常也无济于事，更适合训练专用类别；"
                    "如果两者都能找到，就可以省掉训练。"),
            "how": ('python -m llm.clip_index --video "your.mp4" --interval 2 '
                    '--query "your thing" --query "a close-up" --topk 10'),
        },
    ),
    Route(
        id="example_category",
        name="用示例画面教会一个类别",
        gives="为每个采样秒给出与示例画面的相似度评分；不生成检测框，因此不能直接计数",
        effort="通常几分钟。选几帧示例并命名即可，无需完整数据集、逐框标注或 GPU；首次搜索建立嵌入索引，之后查询几乎无额外成本",
        effort_rank=EFFORT_MINUTES,
        confidence="当目标在画面中占比较明显时通常效果较好；评分经过校准，低分表示匹配较弱，而不是量纲偏差",
        confidence_rank=CONFIDENCE_GOOD,
        holds_when="你能指出哪些画面展示了目标。对于“看得出来但很难准确用文字描述”的内容尤其适用",
        fails_when="目标在画面中占比太小。整图嵌入容易被周围内容淹没，这种情况更适合使用检测器",
        repeat="无需额外分析——索引只构建一次，之后可反复查询",
        topic="training",
        module="llm.clip_categories",
    ),
    Route(
        id="open_vocabulary",
        name="把目标名称输入开放词汇检测器",
        gives="无需训练即可得到真实检测框和可计数结果",
        effort="短时间窗口测试通常只需几分钟；CPU 上每帧约需数秒，因此扫描完整视频可能需要数小时",
        effort_rank=EFFORT_HOURS,
        confidence="效果差异较大。常见物体通常表现很好，但对专业或小众对象可能几乎无法识别，调阈值也无法补救",
        confidence_rank=CONFIDENCE_UNEVEN,
        holds_when="目标是边界明确的常见物体，并且能够用普通文字直接命名",
        fails_when="目标过于专业，或本质上是事件而不是物体。此检测器擅长找有边界的对象，而事件没有固定边界",
        repeat="每组新查询都需要重新完整运行一次",
        topic="training",
        module="llm.owl_detect",
        probe={
            "why": ("在投入时间做标注前，先用短时间窗口测试几分钟。"
                    "同时输入目标查询和一个你确定画面里存在的普通对象作为对照。"
                    "如果对照得分正常而目标始终很低，调阈值通常无法解决，更适合训练专用类别；"
                    "如果两者都能检测到，就可以省掉训练。"),
            "how": ('python -m llm.owl_detect --video "your.mp4" '
                    '--query "your thing" --query sofa --interval 10 '
                    '--start 600 --end 720'),
        },
    ),
    Route(
        id="action_model",
        name="使用动作识别模型",
        gives="对一段连续时间给出动作标签，而不是只判断单帧画面",
        effort="内置已认识的常见动作无需额外设置；如果要训练自定义动作类别，则需要准备示例片段并完成一次训练",
        effort_rank=EFFORT_HOURS,
        confidence="对由运动过程定义的内容效果较好——这是这些方案中真正利用时间连续性的引擎",
        confidence_rank=CONFIDENCE_GOOD,
        holds_when="所描述的是一个动作过程，只有跨越数秒才能成立，单独看任意一帧都无法判断",
        fails_when="需要区分的两个动作仅发生位置不同。模型输入的是裁剪区域，位置差异在判断前已被弱化；更适合先训练同一类别，再用构图规则拆分",
        repeat="需要完整重新运行",
        topic="training",
    ),
    Route(
        id="face_category",
        name="用示例人脸裁剪教会一个人脸类别",
        gives="针对每张人脸给出类别评分，可用于你定义的任意人脸特征类别",
        effort="通常几分钟。选择少量人脸裁剪作为示例即可；人脸本身已由本次扫描定位",
        effort_rank=EFFORT_MINUTES,
        confidence="通常表现较好，而且不像内置表情分类器那样受固定七类限制",
        confidence_rank=CONFIDENCE_GOOD,
        holds_when="所描述的内容确实体现在人脸上，例如视线方向、光照状态或面部特征动作",
        fails_when="目标特征与人脸无关，或人脸太小、偏转过大，以至于前置扫描根本没有找到",
        repeat="视频完成一次人脸扫描后，可反复使用而无需再次扫描",
        topic="training",
        needs="本次运行已完成人脸扫描",
        module="modules.vision.face_examples",
    ),
    Route(
        id="trained_class",
        name="训练自己的专用类别",
        gives="按帧生成可计数检测框，可用于构图规则和实时检测，并能复用于之后分析的所有视频",
        effort="需要一次完整训练流程并建议使用 GPU：收集容易失败的场景、完成标注、训练并导出。数据集中应包含足够的负样本，尤其是容易与目标混淆但不应框选的画面",
        effort_rank=EFFORT_SESSION,
        confidence="这些方案中通常可靠性最高，也是最适合长期稳定驱动规则的一种",
        confidence_rank=CONFIDENCE_EXACT,
        holds_when="目标足够重要，值得投入标注成本，或者更轻量的方法都无法可靠识别",
        fails_when="训练数据没有覆盖模型容易失败的场景。补充少量真正会误判/漏判的样本，往往比继续堆积已经识别正常的样本更有效",
        repeat="无需重复训练——训练后的类别可用于之后每次分析",
        topic="training",
    ),
    Route(
        id="spoken_marker",
        name="给转录中谈到它的时刻加分",
        gives="找出转录文本谈到该内容的时间点；这只是“谈到它”，不能当成目标本身出现在画面中的证据",
        effort="即时——调整转录关键词权重并重新评分即可，无需重新分析视频",
        effort_rank=EFFORT_INSTANT,
        confidence="只是一种较弱的代理信号，测量的是“是否谈到”。内容可能在发生很久后才被提及，也可能发生时无人说话",
        confidence_rank=CONFIDENCE_PROXY,
        holds_when="你现在只想先找到与它相关的讨论时刻，同时判断是否值得投入时间使用上面的更强方案",
        fails_when="你需要准确知道目标本身何时真正出现在画面中",
        repeat="无需额外分析",
        topic="weights",
        needs="已有转录文本",
    ),
)

BY_ID = {route.id: route for route in ROUTES}


def capabilities(report: Mapping) -> dict:
    """What this run has, as the prerequisites the routes are stated in.

    Read from the record rather than from configuration, for the reason
    :func:`modules.report.vocabulary_gap.observed_classes` gives: what a detector
    *could* emit and what it emitted in this file are different lists, and only
    the second one supports a recommendation.
    """
    settings = report.get("settings") or {}
    activity = settings.get("detector_activity") or {}
    vocabulary = report.get("vocabulary") or {}
    chapters = report.get("chapters") or []
    return {
        "classes": [str(c) for c in (vocabulary.get("classes") or [])],
        "events": [str(e) for e in (vocabulary.get("events") or [])],
        # A CLIP index is what makes the example route instant rather than a
        # pass over the video, and the chapters record whether one existed:
        # they are cut on it when it is there and fall back to shot length
        # when it is not.
        "clip_index": any(str(ch.get("method") or "") == "visual"
                          for ch in chapters),
        "faces": int(activity.get("face") or 0) > 0,
        "actions": int(activity.get("action") or 0) > 0,
        "transcript": bool(report.get("speech")),
        "engines": [route.id for route in ROUTES if _installed(route.module)],
    }


# {module path: importable}. Answered once per process -- the answer cannot
# change while the app is running, and `find_spec` walks the path every call.
_present: dict = {}


def _installed(module: str) -> bool:
    """Whether this build actually ships the engine a route needs.

    Detected, not declared. The two editions ship different engines and the
    boundary between them moves; an edition flag threaded through here would be
    one more thing to remember to update, and the failure when it went stale
    would be a recommendation the user cannot act on.
    """
    if not module:
        return True
    if module not in _present:
        try:
            from importlib.util import find_spec
            _present[module] = find_spec(module) is not None
        except (ImportError, ValueError):           # pragma: no cover - defensive
            _present[module] = False
    return _present[module]


def available(caps: Mapping, installed: Optional[Callable] = None) -> list:
    """The routes this run can actually offer, cheapest first.

    ``installed`` decides whether a build has a given engine; it defaults to a
    real import check and is passed in by tests, which have to be able to
    describe a build other than the one they are running on — the same file
    ships in both editions and has to be right in each.
    """
    installed = installed or _installed
    classes = list(caps.get("classes") or []) + list(caps.get("events") or [])
    out = []
    for route in ROUTES:
        if not installed(route.module):
            continue
        if route.id == "compose" and len(set(classes)) < 2:
            continue
        if route.id == "face_category" and not caps.get("faces"):
            continue
        if route.id == "spoken_marker" and not caps.get("transcript"):
            continue
        out.append(route)
    return out


def pick(report: Mapping, installed: Optional[Callable] = None) -> dict:
    """The two routes worth naming, plus the free checks that come before them.

    *Fastest* is the least effort that would genuinely measure the thing, and
    *strongest* is the most reliable answer available. They are occasionally the
    same route, and when they are, one is returned rather than one pretending
    to be two — an advisor that always lists exactly two options is one that has
    padded to a number.

    Two routes are deliberately not in that contest.

    **Composing it from existing classes** is not a peer of the others, because
    whether it is possible at all is not a matter of effort: either what was
    said is an arrangement of classes this video already produced, or no rule
    can express it. That question is already answered for free elsewhere —
    :mod:`modules.rules.rule_proposal` asks a model for the rule and reports back that
    it cannot be built — so this is returned as the thing to try *first*, before
    spending anything.

    **The proxy route** is never picked as either. It measures the talking, and
    offering it as "the fast way to measure this" would be the specific
    dishonesty this whole section exists to prevent; it is returned separately,
    labelled as the stopgap it is.
    """
    caps = capabilities(report)
    routes = available(caps, installed)
    ordering = {route.id: index for index, route in enumerate(ROUTES)}
    measuring = [r for r in routes
                 if r.confidence_rank > CONFIDENCE_PROXY and r.id != "compose"]
    out: dict = {"capabilities": caps,
                 "all": [r.as_dict() for r in routes]}
    if BY_ID["compose"] in routes:
        out["first"] = BY_ID["compose"].as_dict()
    if not measuring:
        return out

    # Ties are broken by the catalogue's own order, which is the sequence a
    # person would work through — not by whichever route the dataclass happened
    # to sort next to.
    fastest = min(measuring,
                  key=lambda r: (r.effort_rank, -r.confidence_rank,
                                 ordering[r.id]))
    strongest = max(measuring,
                    key=lambda r: (r.confidence_rank, -r.effort_rank,
                                   -ordering[r.id]))
    out["fastest"] = fastest.as_dict()
    if strongest.id != fastest.id:
        out["strongest"] = strongest.as_dict()

    # The control test, and it is only worth naming when the answer it gives
    # would change what the user does. If the strongest route on offer already
    # costs minutes, spending five of them finding out whether an even cheaper
    # one might work saves nothing.
    #
    # Which test, of the ones this build has, is the *last* route carrying one:
    # the catalogue runs cheapest to dearest, and the dearest zero-training
    # engine is the one whose silence best predicts that training is
    # unavoidable. A detector that cannot see the thing settles the question in
    # a way a whole-frame similarity score cannot.
    probing = [r for r in routes if r.probe]
    if strongest.effort_rank >= EFFORT_SESSION and probing:
        chosen = probing[-1]
        out["probe"] = dict(chosen.probe, route=chosen.id)
        # On a build where the cheapest route is also the best test, the probe
        # is not a separate errand — it is how you find out whether the route
        # you were going to take anyway is working. Saying "first, do this"
        # about the thing already recommended two lines down reads as a page
        # that has lost track of itself.
        if chosen.id == fastest.id:
            out["probe"]["same_as_fastest"] = True

    interim = next((r for r in routes if r.confidence_rank == CONFIDENCE_PROXY),
                   None)
    if interim is not None:
        out["interim"] = interim.as_dict()
    return out


def describe(picked: Mapping) -> list:
    """The picks as lines a page or a prompt can print, in reading order.

    One rendering, used by both, so the page and the narration cannot end up
    recommending different routes for the same run.
    """
    lines = []
    first = picked.get("first")
    if first:
        lines.append(f"Costs nothing to rule out: {first['name']} — "
                     f"{first['holds_when']}. Ask the advisor to draft the "
                     f"rule; it says so when the claim cannot be built from "
                     f"the classes this video has, and that answer is free.")
    probe = picked.get("probe")
    if probe:
        lines.append(("How to tell in five minutes whether it is working: "
                      if probe.get("same_as_fastest")
                      else "Then, before committing: ") + probe["why"])
    for key, lead in (("fastest", "Fastest"), ("strongest", "Most reliable"),
                      ("interim", "Meanwhile")):
        route = picked.get(key)
        if not route:
            continue
        lines.append(f"{lead}: {route['name']} — {route['effort']}. "
                     f"Gives you {route['gives']}. "
                     f"Right route when {route['holds_when']}; "
                     f"not when {route['fails_when']}.")
    return lines
