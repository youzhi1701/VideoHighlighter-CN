// Make a short-form reel from footage that has already been analysed.
//
// The Auto tab produces a film: every good moment, at whatever length they
// came out. A reel is a different object — fifteen to thirty seconds, a cut
// every second or two, and a shape (hook, context, escalation, payoff) that
// the highest-scoring seconds do not have on their own.
//
// So this tab asks for three things only: how long, how fast, and what the
// opening line says. Everything else is derived, and the plan is shown before
// anything is rendered, because the useful moment to discover that 24 seconds
// of calm pacing is impossible is before the encode rather than after it.

import { useCallback, useEffect, useMemo, useState } from "react"
import {
  Clapperboard,
  Loader2,
  Music,
  Play,
  RefreshCw,
  MapPin,
  Scissors,
  Shapes,
  Spline,
  Type,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Checkbox } from "@/components/ui/checkbox"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  getReelOptions,
  planReel,
  renderReel,
  type ReelOptions,
  type ReelPace,
  type ReelPlan,
  type ReelRequest,
} from "@/lib/api"
import {
  basename,
  pickAudioFile,
  pickDirectory,
  pickTrackFile,
} from "@/lib/files"

const SHAPES = [
  { key: "vertical", label: "竖屏 9:16", width: 1080, height: 1920, fill: "crop" },
  { key: "square", label: "方形 1:1", width: 1080, height: 1080, fill: "crop" },
  { key: "wide", label: "横屏 16:9", width: 1920, height: 1080, fill: "pad" },
]

// Transition names are the renderer's, which are terse by design. These are
// what a person picking one would call it; anything without an entry falls
// back to its own name with the underscores taken out.
const TRANSITION_LABELS: Record<string, string> = {
  cut: "Cut",
  crossfade: "Crossfade",
  dissolve: "Dissolve",
  dip_to_black: "Dip to black",
  dip_to_white: "Dip to white",
  fade_grays: "Fade through grey",
  iris_open: "Iris open",
  iris_close: "Iris close",
  circle_open: "Circle open",
  circle_close: "Circle close",
  diamond_open: "Diamond open",
  diamond_close: "Diamond close",
  box_open: "Box open",
  box_close: "Box close",
  barn_open: "Barn doors open",
  barn_close: "Barn doors close",
  barn_up: "Barn doors up",
  barn_down: "Barn doors down",
  clock: "Clock sweep",
  clock_back: "Clock sweep, back",
  blinds: "Blinds",
  blinds_fine: "Blinds, fine",
  blinds_v: "Blinds, vertical",
  blinds_v_fine: "Blinds, vertical fine",
  checker: "Checkerboard",
  grain: "Film grain",
  grain_iris: "Film grain, from the middle",
  ripple: "Ripple",
  spiral: "Spiral",
}

const prettyName = (key: string) =>
  TRANSITION_LABELS[key] ??
  key.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase())

// What each move does. These are not transitions — they happen on the ends of
// a shot rather than on the join, so they stack with whatever the join is.
const MOTION_HINTS: Record<string, string> = {
  punch: "The frame grows into the cut and settles after it. The most useful one.",
  pull: "The reverse — the frame is largest away from the cut and eases back.",
  shake: "A short kick either side of the join. Good on a beat.",
  roll: "A few degrees of tilt into the cut and out of it.",
  glitch: "Shake with the colour channels split for the length of the move.",
}

// What each curve does, in the terms someone choosing one is thinking in.
const EASING_HINTS: Record<string, string> = {
  linear: "Even throughout — what ffmpeg does unaided",
  ease_in: "Slow to start, then full speed",
  ease_out: "Fast away, gentle landing — the safe choice",
  ease_in_out: "Slow at both ends; reads as deliberate",
  smooth: "Gentler than the others; closest to a hand-drawn fade",
  snap: "Most of the move at once, then it settles",
}

// The four sections a viewer actually experiences, and what each is for. Only
// the hook is filled in by most people, which is why it is first and explained.
const TEXT_FIELDS = [
  {
    section: "Hook",
    label: "开场文案",
    hint: "The first two seconds. Most people watch muted, so this is what they read.",
    placeholder: "例如：我差点在第 38 英里放弃",
  },
  {
    section: "Context",
    label: "背景信息",
    hint: "Where, what, what is at stake. Keep it to a few words.",
    placeholder: "例如：50 英里，暴雨，没有备用计划。",
  },
  {
    section: "Payoff",
    label: "结尾文案",
    hint: "The result, the lesson, or a question worth answering.",
    placeholder: "例如：答案其实是慢下来",
  },
]

interface Props {
  running: boolean
  onCancel: () => void
  /** Project folder the last automatic run used, if any. */
  suggestedRoot?: string
}

export function ReelTab({ running, onCancel, suggestedRoot }: Props) {
  const [root, setRoot] = useState("")
  const [duration, setDuration] = useState(24)
  const [pace, setPace] = useState("energetic")
  const [shape, setShape] = useState(0)
  const [music, setMusic] = useState("")
  const [texts, setTexts] = useState<Record<string, string>>({})
  const [paces, setPaces] = useState<ReelPace[]>([])
  const [lengths, setLengths] = useState<{ seconds: number; reason: string }[]>([])
  const [plan, setPlan] = useState<ReelPlan | null>(null)
  const [planning, setPlanning] = useState(false)
  const [status, setStatus] = useState("")

  // How the reel joins its shots.
  const [options, setOptions] = useState<ReelOptions | null>(null)
  const [transition, setTransition] = useState("cut")
  const [transitionSeconds, setTransitionSeconds] = useState(0.35)
  const [easing, setEasing] = useState("ease_out")
  const [feather, setFeather] = useState(0)
  // A move on the ends of each shot — the half of a transition a mask cannot do.
  const [motion, setMotion] = useState("none")
  const [showEveryTransition, setShowEveryTransition] = useState(false)
  // Whether a shot may start later than the top of its clip.
  const [settle, setSettle] = useState(true)
  // Whether the reel avoids showing the same spot, or the same picture, twice.
  const [spread, setSpread] = useState(true)
  const [track, setTrack] = useState("")
  // Graphics drawn over the finished reel.
  const [overlays, setOverlays] = useState<string[]>([])

  useEffect(() => {
    void getReelOptions().then((r) => {
      if (!r.ok) return
      setOptions(r)
      setPaces(r.paces ?? [])
      setLengths(r.lengths ?? [])
    })
  }, [])

  // Only a transition with an edge can have that edge softened; a crossfade
  // has no edge and a slide cannot be masked at all. Rather than let the
  // slider sit there doing nothing, it is disabled and says why.
  const chosen = useMemo(
    () => options?.transitions?.find((t) => t.key === transition),
    [options, transition],
  )
  const softenable = Boolean(chosen?.maskable)
  const isCut = transition === "cut"

  useEffect(() => {
    if (suggestedRoot && !root) setRoot(suggestedRoot)
  }, [suggestedRoot, root])

  const band = useMemo(() => paces.find((p) => p.key === pace), [paces, pace])

  const request = useCallback(
    (): ReelRequest => ({
      dest_root: root,
      duration,
      pace,
      title: "短片",
      music,
      transition,
      transition_duration: transitionSeconds,
      easing,
      feather: softenable ? feather : 0,
      motion,
      settle,
      spread,
      track,
      overlays,
      width: SHAPES[shape].width,
      height: SHAPES[shape].height,
      fill: SHAPES[shape].fill,
      texts,
    }),
    [
      root,
      duration,
      pace,
      shape,
      music,
      texts,
      transition,
      transitionSeconds,
      easing,
      feather,
      softenable,
      motion,
      settle,
      spread,
      track,
      overlays,
    ],
  )

  // Re-plan whenever a choice changes. Planning only probes durations, so it is
  // cheap enough to run on every keystroke of the length field.
  useEffect(() => {
    if (!root) {
      setPlan(null)
      return
    }
    let stale = false
    setPlanning(true)
    void planReel(request())
      .then((p) => {
        if (!stale) setPlan(p)
      })
      .finally(() => {
        if (!stale) setPlanning(false)
      })
    return () => {
      stale = true
    }
  }, [request, root])

  const tooShort = Boolean(band && duration < band.minimum_duration - 0.5)

  const render = async () => {
    const res = await renderReel(request())
    if (!res.ok) setStatus(res.error ?? "无法开始渲染")
    else setStatus(`Rendering ${res.shots} shots to ${basename(res.output ?? "")}…`)
  }

  return (
    <div className="space-y-5">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Clapperboard className="size-4" /> 成片
          </CardTitle>
          {plan?.ok && (
            <div className="flex items-center gap-2">
              <Badge>{plan.duration?.toFixed(0)} 秒</Badge>
              <Badge variant="secondary">{plan.shots} 个镜头</Badge>
              <Badge variant="outline">{plan.cuts_per_minute} 次剪切/分钟</Badge>
            </div>
          )}
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <Label className="text-xs">素材</Label>
            <div className="flex gap-2">
              <Input
                value={root}
                onChange={(e) => setRoot(e.target.value)}
                placeholder="“自动”标签页写入的项目文件夹"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  const dir = await pickDirectory()
                  if (dir) setRoot(dir)
                }}
              >
                Browse
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              直接使用之前运行时已经找到的高光结果，因此不会重复分析。
            </p>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label className="text-xs">时长</Label>
            <div className="flex flex-wrap items-center gap-2">
              {lengths.map((l) => (
                <Button
                  key={l.seconds}
                  variant={duration === l.seconds ? "default" : "outline"}
                  size="sm"
                  onClick={() => setDuration(l.seconds)}
                  title={l.reason}
                >
                  {l.seconds}s
                </Button>
              ))}
              <Input
                type="number"
                min={5}
                max={180}
                value={duration}
                onChange={(e) => setDuration(Number(e.target.value) || 24)}
                className="w-20"
              />
              <span className="text-xs text-muted-foreground">
                {lengths.find((l) => l.seconds === duration)?.reason ?? "seconds"}
              </span>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <Label className="text-xs">节奏</Label>
              <Select value={pace} onValueChange={setPace}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {paces.map((p) => (
                    <SelectItem key={p.key} value={p.key}>
                      {p.label} — {p.min_shot}–{p.max_shot}s shots
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {band && (
                <p className="text-xs text-muted-foreground">
                  {band.cuts_per_minute[0]}–{band.cuts_per_minute[1]} cuts per
                  minute.
                </p>
              )}
              {tooShort && band && (
                <p className="text-xs text-amber-500">
                  A {band.label.toLowerCase()} story needs about{" "}
                  {band.minimum_duration.toFixed(0)}s. Pick a faster pace or a
                  longer reel.
                </p>
              )}
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">画幅</Label>
              <Select value={String(shape)} onValueChange={(v) => setShape(Number(v))}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {SHAPES.map((s, i) => (
                    <SelectItem key={s.key} value={String(i)}>
                      {s.label} ({s.width}×{s.height})
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {SHAPES[shape].fill === "crop"
                  ? "Fills the frame — the sides of a wide shot are cropped away."
                  : "Keeps the whole frame, with bars where it does not fit."}
              </p>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="flex items-center gap-1.5 text-xs">
              <Music className="size-3.5" /> 音乐
            </Label>
            <div className="flex gap-2">
              <Input
                value={music ? basename(music) : ""}
                readOnly
                placeholder="可选——镜头会自动对齐节拍"
              />
              <Button
                variant="outline"
                size="sm"
                onClick={async () => {
                  const p = await pickAudioFile()
                  if (p) setMusic(p)
                }}
              >
                Pick
              </Button>
              {music && (
                <Button variant="ghost" size="sm" onClick={() => setMusic("")}>
                  清除
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm font-medium">
            <Shapes className="size-4" /> 拼接方式
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs">转场</Label>
                <button
                  type="button"
                  className="text-[10px] text-muted-foreground underline-offset-2 hover:underline"
                  onClick={() => setShowEveryTransition((v) => !v)}
                >
                  {showEveryTransition ? "show the usual ones" : "show all of them"}
                </button>
              </div>
              <Select value={transition} onValueChange={setTransition}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-80">
                  {showEveryTransition
                    ? (options?.families ?? []).map((family) => (
                        <SelectGroup key={family.name}>
                          <SelectLabel>{family.name}</SelectLabel>
                          {family.items.map((t) => (
                            <SelectItem key={t.key} value={t.key}>
                              {prettyName(t.key)}
                            </SelectItem>
                          ))}
                        </SelectGroup>
                      ))
                    : (options?.curated ?? ["cut"]).map((key) => (
                        <SelectItem key={key} value={key}>
                          {prettyName(key)}
                        </SelectItem>
                      ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {isCut
                  ? "Every shot butts straight against the next. Fastest, and never wrong."
                  : softenable
                    ? "Has an edge, so it can be softened below."
                    : "Blends or slides the whole frame — there is no edge to soften."}
              </p>
            </div>

            <div className="space-y-1.5">
              <Label className="text-xs">运动</Label>
              <Select value={easing} onValueChange={setEasing} disabled={isCut}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {(options?.easings ?? ["linear"]).map((key) => (
                    <SelectItem key={key} value={key}>
                      {prettyName(key)}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {isCut
                  ? "A cut has no length to move over."
                  : (EASING_HINTS[easing] ?? "How the blend moves across its length.")}
              </p>
            </div>
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">切点运动</Label>
            <Select value={motion} onValueChange={setMotion}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {(options?.motions ?? [{ key: "none", label: "静止" }]).map((m) => (
                  <SelectItem key={m.key} value={m.key}>
                    {m.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="max-w-prose text-xs text-muted-foreground">
              {motion === "none"
                ? "Every transition above is a shape moving over two still pictures, which is why they can start to feel alike. This moves the picture itself on the ends of each shot."
                : (MOTION_HINTS[motion] ??
                  "Applied to the ends of each shot, so it stacks with the transition.")}
            </p>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs">时长</Label>
                <span className="font-mono text-xs text-muted-foreground">
                  {transitionSeconds.toFixed(2)}s
                </span>
              </div>
              <Slider
                min={0.08}
                max={1.5}
                step={0.01}
                value={[transitionSeconds]}
                onValueChange={([v]) => setTransitionSeconds(v)}
                disabled={isCut}
              />
              <p className="text-xs text-muted-foreground">
                如果镜头太短无法容纳，会自动缩短。
              </p>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <Label className="text-xs">柔边</Label>
                <span className="font-mono text-xs text-muted-foreground">
                  {softenable && !isCut ? `${Math.round(feather * 100)}%` : "—"}
                </span>
              </div>
              <Slider
                min={0}
                max={0.8}
                step={0.01}
                value={[feather]}
                onValueChange={([v]) => setFeather(v)}
                disabled={isCut || !softenable}
              />
              <p className="text-xs text-muted-foreground">
                {isCut || !softenable
                  ? "Pick a wipe, an iris or blinds to soften its edge."
                  : feather === 0
                    ? "A hard edge, the way ffmpeg draws it."
                    : "Feathers the edge instead of flipping each pixel at once."}
              </p>
            </div>
          </div>

          <Separator />

          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <Label className="flex items-center gap-1.5 text-xs">
                <Scissors className="size-3.5" /> 跳过摆放相机的画面
              </Label>
              <p className="max-w-prose text-xs text-muted-foreground">
                有些片段开头仍处于举起相机、转动相机或刚从口袋取出设备的过程。
                开启后，会优先从每个片段中选择更稳定的一段，而不是直接从第一个画面开始。
                {plan?.ok && settle && (plan.trimmed ?? 0) > 0 && (
                  <>
                    {" "}
                    <span className="text-foreground">
                      已调整 {plan.trimmed}/{plan.shots} 个镜头，最晚的镜头从
                      {plan.trimmed_max?.toFixed(1)} 秒后开始。
                    </span>
                  </>
                )}
                {plan?.ok && settle && (plan.trimmed ?? 0) === 0 && (
                  <> 该素材开头已经干净，无需额外处理。</>
                )}
              </p>
            </div>
            <Switch checked={settle} onCheckedChange={setSettle} />
          </div>

          <Separator />

          <div className="flex items-start justify-between gap-4">
            <div className="space-y-1">
              <Label className="flex items-center gap-1.5 text-xs">
                <MapPin className="size-3.5" /> 不要连续显示相同视角
              </Label>
              <p className="max-w-prose text-xs text-muted-foreground">
                在同一地点停留并连续拍摄多次，是成片内容重复最常见的原因。
                系统会按拍摄地点和时间对片段分组，优先让每个地点出现一次，再考虑重复使用。
                {plan?.ok && spread && (plan.clips_seen ?? 0) > 0 && (
                  <>
                    {" "}
                    <span className="text-foreground">
                      已使用 {plan.clips_seen} 个片段中的 {plan.places} 个地点，优先来自不同位置。
                    </span>
                  </>
                )}
              </p>
            </div>
            <Switch checked={spread} onCheckedChange={setSpread} />
          </div>

          <div className="space-y-1.5">
            <Label className="text-xs">GPS 轨迹（可选）</Label>
            <div className="flex gap-2">
              <Input
                value={track ? basename(track) : ""}
                readOnly
                placeholder="来自手表或手机的 .gpx 文件"
                disabled={!spread}
              />
              <Button
                variant="outline"
                size="sm"
                disabled={!spread}
                onClick={async () => {
                  const p = await pickTrackFile()
                  if (p) setTrack(p)
                }}
              >
                Pick
              </Button>
              {track && (
                <Button variant="ghost" size="sm" onClick={() => setTrack("")}>
                  清除
                </Button>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              仅当素材本身没有 GPS 信息时才需要。即使没有轨迹，成片也会根据拍摄时间尽量分散镜头。
              提供轨迹后，还可以用于绘制下方的图形效果。
            </p>
          </div>

          <Separator />

          <div className="space-y-2">
            <Label className="flex items-center gap-1.5 text-xs">
              <Spline className="size-3.5" /> 图形
            </Label>
            <div className="grid gap-2 sm:grid-cols-2">
              {(options?.overlays ?? []).map((item) => {
                const blocked = item.needs_track && !track
                return (
                  <label
                    key={item.key}
                    className={`flex items-center gap-2 text-xs ${
                      blocked ? "opacity-50" : "cursor-pointer"
                    }`}
                    title={blocked ? "需要 GPS 轨迹" : undefined}
                  >
                    <Checkbox
                      checked={overlays.includes(item.key)}
                      disabled={blocked}
                      onCheckedChange={(on) =>
                        setOverlays((current) =>
                          on
                            ? [...current, item.key]
                            : current.filter((k) => k !== item.key),
                        )
                      }
                    />
                    {item.label}
                  </label>
                )
              })}
            </div>
            <p className="text-xs text-muted-foreground">
              Drawn over the finished reel. The profile and the route come from
              the track, and the marker jumps to wherever each shot was filmed.
            </p>
          </div>
        </CardContent>
      </Card>

      <div className="grid min-w-0 gap-5 lg:grid-cols-2 [&>*]:min-w-0">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-sm font-medium">
              <Type className="size-4" /> 屏幕文字
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {TEXT_FIELDS.map((field) => (
              <div key={field.section} className="space-y-1">
                <Label className="text-xs">{field.label}</Label>
                <Input
                  value={texts[field.section] ?? ""}
                  onChange={(e) =>
                    setTexts((t) => ({ ...t, [field.section]: e.target.value }))
                  }
                  placeholder={field.placeholder}
                />
                <p className="text-xs text-muted-foreground">{field.hint}</p>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">剪辑计划</CardTitle>
            {planning && <Loader2 className="size-4 animate-spin text-muted-foreground" />}
          </CardHeader>
          <CardContent className="space-y-3">
            {!root && (
              <p className="text-xs text-muted-foreground">
                请选择素材文件夹以查看镜头计划。
              </p>
            )}
            {plan && !plan.ok && (
              <p className="text-xs text-destructive">{plan.error}</p>
            )}
            {plan?.ok && (
              <>
                <pre className="overflow-x-auto rounded-md border bg-muted/30 p-2 font-mono text-xs leading-relaxed">
                  {plan.summary}
                </pre>
                <ShotStrip plan={plan} />
              </>
            )}

            <div className="flex items-center gap-2 pt-1">
              {running ? (
                <Button variant="destructive" size="sm" onClick={onCancel}>
                  取消
                </Button>
              ) : (
                <Button
                  size="sm"
                  onClick={() => void render()}
                  disabled={!plan?.ok}
                >
                  <Play className="size-3.5" /> 生成短片
                </Button>
              )}
              <Button
                variant="ghost"
                size="sm"
                onClick={() => void planReel(request()).then(setPlan)}
                disabled={!root}
              >
                <RefreshCw className="size-3.5" /> 重新规划
              </Button>
            </div>
            {status && <p className="text-xs text-muted-foreground">{status}</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

/** The shots as proportional blocks, coloured by section. Reading the four
 *  sections as four bands is the quickest way to see whether the shape is
 *  right — a payoff the same width as the body is the usual mistake.
 *
 *  A shot taken from partway into its clip is marked, because an in-point
 *  nobody chose is the kind of thing that reads as a bug until you can see
 *  that it was deliberate and how far in it went. */
function ShotStrip({ plan }: { plan: ReelPlan }) {
  const cuts = plan.cuts ?? []
  const total = plan.duration ?? 0
  if (!cuts.length || !total) return null

  const tint: Record<string, string> = {
    Hook: "bg-rose-500/50",
    Context: "bg-amber-500/50",
    Escalation: "bg-sky-500/50",
    Payoff: "bg-emerald-500/50",
  }

  return (
    <div className="space-y-1">
      <div className="flex h-8 w-full overflow-hidden rounded-md border">
        {cuts.map((cut, i) => {
          const moved = cut.start > 0.25
          return (
            <div
              key={`${cut.source}-${i}`}
              className={`${tint[cut.label ?? ""] ?? "bg-muted"} relative border-r border-background/40`}
              style={{ width: `${((cut.duration ?? 0) / total) * 100}%` }}
              title={
                `${cut.label} — ${(cut.duration ?? 0).toFixed(1)}s\n` +
                `${basename(cut.source)}，从 ${cut.start.toFixed(2)} 秒开始` +
                (moved ? " （已跳过放置相机的片段）" : "") +
                (cut.transition && cut.transition !== "cut"
                  ? `\n${prettyName(cut.transition)} ${cut.transition_duration.toFixed(2)}s`
                  : "") +
                (cut.text ? `\n"${cut.text}"` : "")
              }
            >
              {moved && (
                <span className="absolute inset-x-0 bottom-0 h-1 bg-foreground/50" />
              )}
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap items-center gap-2">
        {Object.entries(tint).map(([name, cls]) => (
          <span key={name} className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <span className={`inline-block size-2 rounded-sm ${cls}`} /> {name}
          </span>
        ))}
        {(plan.trimmed ?? 0) > 0 && (
          <span className="flex items-center gap-1 text-[10px] text-muted-foreground">
            <span className="inline-block h-1 w-3 rounded-sm bg-foreground/50" />
            开始时间晚于对应片段
          </span>
        )}
      </div>
    </div>
  )
}
