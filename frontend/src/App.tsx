import { useEffect, useRef, useState } from "react"
import {
  Film,
  Moon,
  Sun,
  Plus,
  Trash2,
  Play,
  Pause,
  Square,
  Sparkles,
  MonitorPlay,
  FileText,
  FolderOpen,
  ChevronDown,
  ChevronUp,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Toaster } from "@/components/ui/sonner"
import { toast } from "sonner"
import { useTheme } from "@/lib/theme"
import { pickVideos, pickDirectory, pickAudioFile, basename } from "@/lib/files"
import {
  DEFAULT_CONFIG,
  toGuiConfig,
  totalPoints,
  fromConfigFile,
  toConfigFile,
  MUSIC_MODES,
} from "@/lib/config"
import type { HighlighterConfig } from "@/lib/config"
import {
  startRun,
  startDownload,
  cancelRun,
  pauseRun,
  resumeRun,
  openEventSocket,
  getHealth,
  getStats,
  getObjectLabels,
  getActionLabels,
  getConfigFile,
  saveConfigFile,
  getVideoInfo,
  getAvoidRanges,
  openEditor,
  revealLog,
  revealOutput,
  scanFolder,
  combineVideos,
  startAuto,
  type AutoStageName,
  type AutoStageStatus,
  type RunEvent,
} from "@/lib/api"
import { VideoCard } from "@/components/VideoCard"
import { SelectField } from "@/components/SelectField"
import { Slider } from "@/components/ui/slider"
import { TimeRange, DEFAULT_TIME_RANGE, type TimeRangeState } from "@/components/TimeRange"
import {
  DetectionPreview,
  MAX_FRAMES,
  type PreviewFrame,
} from "@/components/DetectionPreview"
import { setPreview } from "@/lib/api"
import { AutoTab } from "@/components/tabs/AutoTab"
import { ReelTab } from "@/components/tabs/ReelTab"
import { TimelineTab } from "@/components/tabs/TimelineTab"
import { BasicTab } from "@/components/tabs/BasicTab"
import { TranscriptTab } from "@/components/tabs/TranscriptTab"
import { AdvancedTab } from "@/components/tabs/AdvancedTab"
import { AvoidTab } from "@/components/tabs/AvoidTab"
import { LlmChatTab } from "@/components/tabs/LlmChatTab"
import { VisionSearchTab } from "@/components/tabs/VisionSearchTab"
import { AboutTab } from "@/components/tabs/AboutTab"
import type { VisionResult } from "@/lib/api"
import {
  DownloadTab,
  DEFAULT_DOWNLOAD,
  type DownloadSettings,
} from "@/components/tabs/DownloadTab"

type LogLine = { text: string; kind: "info" | "err" | "ok" }

export default function App() {
  const { theme, toggle } = useTheme()
  const [videos, setVideos] = useState<string[]>([])
  const [output, setOutput] = useState("highlight.mp4")
  const [cfg, setCfg] = useState<HighlighterConfig>(DEFAULT_CONFIG)
  const [dl, setDl] = useState<DownloadSettings>(DEFAULT_DOWNLOAD)
  const [avoidIds, setAvoidIds] = useState<string[]>([])
  const [avoidRanges, setAvoidRanges] = useState<[number, number][]>([])
  const [objectLabels, setObjectLabels] = useState<string[]>([])
  const [actionLabels, setActionLabels] = useState<string[]>([])
  const [timeRange, setTimeRange] = useState<TimeRangeState>(DEFAULT_TIME_RANGE)
  const [duration, setDuration] = useState(0)
  const [running, setRunning] = useState(false)
  const [paused, setPaused] = useState(false)
  const [progress, setProgress] = useState(0)
  const [task, setTask] = useState("")
  const [log, setLog] = useState<LogLine[]>([])
  const [online, setOnline] = useState<boolean | null>(null)
  const [analyzed, setAnalyzed] = useState<number | null>(null)
  const [sessionCount, setSessionCount] = useState(0)
  const [faceRefresh, setFaceRefresh] = useState(0)
  const [loaded, setLoaded] = useState(false)
  const [livePreview, setLivePreview] = useState(false)
  const [frames, setFrames] = useState<PreviewFrame[]>([])
  // Output dock starts closed: settings need the room until there's a run.
  const [logOpen, setLogOpen] = useState(false)
  // Set when the engine says it reused cached detections — the one case where
  // the preview legitimately has nothing to show, so the panel can say so
  // instead of waiting forever.
  const [usedCache, setUsedCache] = useState(false)
  // Last finished run's output file, so the user can jump to the video they
  // just made instead of hunting for it.
  const [lastOutput, setLastOutput] = useState("")
  // Shared by the LLM Chat and Visual Search tabs.
  const [llmBackend, setLlmBackend] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [visionResults, setVisionResults] = useState<VisionResult[]>([])
  const [lastEdl, setLastEdl] = useState("")
  const [lastRoot, setLastRoot] = useState("")
  const [autoStages, setAutoStages] = useState<
    Partial<Record<AutoStageName, { status: AutoStageStatus; detail: string }>>
  >({})
  const wsRef = useRef<WebSocket | null>(null)
  const logEndRef = useRef<HTMLDivElement | null>(null)
  // Read inside WS callbacks, which close over the mount-time value otherwise.
  const dlRef = useRef(dl)
  dlRef.current = dl
  // Reel chaining: when a multi-video run finishes, its outputs are stashed here
  // and the `done` handler kicks off a /combine. Cleared before that POST so a
  // combine run can never re-trigger itself. Read cfg/output through refs so the
  // WS callback (closed over mount-time values) sees the current settings.
  const pendingReelRef = useRef<string[] | null>(null)
  const cfgRef = useRef(cfg)
  cfgRef.current = cfg
  const outputRef = useRef(output)
  outputRef.current = output

  const set = <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) =>
    setCfg((c) => ({ ...c, [k]: v }))

  // Health poll so the UI shows whether the Python engine is reachable.
  useEffect(() => {
    let alive = true
    const check = () =>
      getHealth()
        .then((h) => {
          if (!alive) return
          setOnline(true)
          setRunning(h.running)
          setPaused(h.paused)
        })
        .catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, 4000)
    return () => {
      alive = false
      clearInterval(id)
    }
  }, [])

  // Restore settings from config.yaml — the same file the Qt GUI reads/writes.
  useEffect(() => {
    void (async () => {
      const res = await getConfigFile()
      if (res.ok) {
        setCfg((c) => ({ ...c, ...fromConfigFile(res.config) }))
        const h = res.config.highlights ?? {}
        setTimeRange({
          enabled: Boolean(h.use_time_range),
          startPct: h.range_start_pct ?? 0,
          endPct: h.range_end_pct ?? 100,
        })
        if (h.output) setOutput(h.output)
        const paths: string[] = res.config.video?.paths ?? []
        if (paths.length) setVideos(paths)
        const d = res.config.download ?? {}
        setDl((s) => ({
          ...s,
          url: d.last_url ?? s.url,
          saveDir: d.save_dir ?? s.saveDir,
          downloadFull: d.download_full ?? s.downloadFull,
          rangeStart: d.time_range_start ?? s.rangeStart,
          rangeEnd: d.time_range_end ?? s.rangeEnd,
          concurrent: d.concurrent_downloads ?? s.concurrent,
          autoAdd: d.auto_add ?? s.autoAdd,
        }))
      }
      setLoaded(true)
    })()
  }, [])

  // Persist settings whenever they settle, mirroring the Qt app's save-on-close.
  // Debounced so typing doesn't thrash the file; gated on `loaded` so we never
  // write defaults over a real config before it has been read.
  useEffect(() => {
    if (!loaded) return
    const id = setTimeout(() => {
      void saveConfigFile(
        toConfigFile(cfg, {
          videoPaths: videos,
          output,
          timeRange,
          download: {
            last_url: dl.url,
            save_dir: dl.saveDir,
            auto_add: dl.autoAdd,
            download_full: dl.downloadFull,
            time_range_start: dl.rangeStart,
            time_range_end: dl.rangeEnd,
            concurrent_downloads: dl.concurrent,
          },
        }),
      )
    }, 800)
    return () => clearTimeout(id)
  }, [cfg, videos, output, timeRange, dl, loaded])

  // Label vocabularies depend on the detector/backend selection, same as Qt.
  useEffect(() => {
    void getObjectLabels(cfg.yolo_type).then(setObjectLabels)
  }, [cfg.yolo_type])

  useEffect(() => {
    void getActionLabels(cfg.action_backend, cfg.action_models).then(setActionLabels)
  }, [cfg.action_backend, cfg.action_models])

  // Real duration for the first video drives the time-range slider.
  useEffect(() => {
    if (!videos.length) {
      setDuration(0)
      return
    }
    void getVideoInfo(videos[0]).then((r) =>
      setDuration(r.ok ? r.duration : 0),
    )
  }, [videos])

  /** Ranges the user marked in the native Timeline Viewer, via the shared store.
   *  Refreshed on video change and whenever the window regains focus, so ranges
   *  marked in the viewer land here without a manual reload. */
  const refreshAvoidRanges = () => {
    if (!videos.length) {
      setAvoidRanges([])
      return
    }
    void getAvoidRanges(videos[0]).then((r) =>
      setAvoidRanges(r.ok ? r.ranges : []),
    )
  }

  useEffect(() => {
    refreshAvoidRanges()
    window.addEventListener("focus", refreshAvoidRanges)
    return () => window.removeEventListener("focus", refreshAvoidRanges)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videos])

  // Lifetime analyzed counter (shared stats file with the Qt GUI).
  useEffect(() => {
    void getStats().then((r) => r.ok && setAnalyzed(r.analyzed))
  }, [sessionCount])

  // Preview toggle applies mid-run, matching the Qt checkbox.
  useEffect(() => {
    void setPreview(livePreview)
  }, [livePreview])

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [log])

  const appendLog = (text: string, kind: LogLine["kind"] = "info") =>
    setLog((l) => [...l, { text, kind }])

  const handleEvent = (e: RunEvent) => {
    switch (e.type) {
      case "started":
        appendLog("=== 已开始 ===", "ok")
        break
      case "log":
        appendLog(e.message)
        break
      case "cache_used":
        setUsedCache(true)
        break
      case "progress":
        if (e.total > 0) setProgress(Math.round((e.current / e.total) * 100))
        setTask(e.detail ? `${e.task} — ${e.detail}` : e.task)
        break
      case "downloaded":
        if (dlRef.current.autoAdd && e.paths.length) {
          setVideos((v) => [...new Set([...v, ...e.paths])])
          appendLog(`➕ 已添加 ${e.paths.length} 个下载的视频`, "ok")
        }
        break
      case "faces_scanned":
        appendLog(`👤 找到 ${e.count} 个身份`, "ok")
        setFaceRefresh((n) => n + 1)
        break
      case "preview":
        setFrames((f) => {
          const next = [...f, { jpeg: e.jpeg, boxes: e.boxes, sec: e.sec }]
          return next.length > MAX_FRAMES ? next.slice(-MAX_FRAMES) : next
        })
        break
      case "vision_hit":
        appendLog(`🔎 在 ${e.timestamp.toFixed(1)} 秒找到匹配`, "ok")
        break
      case "vision_results":
        setVisionResults(e.results)
        break
      case "stage":
        setAutoStages((s) => ({
          ...s,
          [e.stage]: { status: e.status, detail: e.detail },
        }))
        break
      case "finished":
        appendLog(`✔ 已完成：${e.output || "（无输出）"}`, "ok")
        setSessionCount((n) => n + 1)
        // Downloads and face scans reuse this event for a summary ("3 个文件"),
        // so only keep an output that's actually a file we can reveal.
        if (/\.[a-z0-9]{2,4}$/i.test(e.output)) setLastOutput(e.output)
        // Stash produced highlights so `done` can combine them into a reel.
        if (e.outputs && e.outputs.length > 1) pendingReelRef.current = e.outputs
        // The cut list is what makes the run editable rather than final.
        if (e.edl) setLastEdl(e.edl)
        toast.success("完成")
        break
      case "cancelled":
        appendLog("⏹ 已取消", "err")
        toast("已取消")
        break
      case "error":
        appendLog(`✖ ${e.message}`, "err")
        toast.error("发生错误——请查看日志")
        break
      case "done": {
        wsRef.current?.close()
        wsRef.current = null
        // A finished multi-video run with the reel toggle on: combine its outputs
        // into one video. Firing from `done` (not `finished`) means the run's
        // child process has already exited, so /combine won't hit "a run is
        // already in progress". Clear the stash first — the combine run emits its
        // own `done`, and a null stash there stops it re-combining itself.
        const reel = pendingReelRef.current
        pendingReelRef.current = null
        if (reel && cfgRef.current.combine_reel) {
          void startReelCombine(reel)
        } else {
          setRunning(false)
          setTask("")
        }
        break
      }
    }
  }

  /** Directory of a path, honoring whichever separator it uses. */
  const pathDir = (p: string) => {
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"))
    return i === -1 ? "" : p.slice(0, i + 1) // keep the trailing separator
  }

  /** Combine finished highlights into one reel, reusing the event socket. */
  const startReelCombine = async (files: string[]) => {
    const c = cfgRef.current
    const dir = pathDir(files[0])
    const stem = (outputRef.current || "highlight.mp4").replace(/\.[^.]+$/, "")
    const out = `${dir}${stem}_reel.mp4`
    appendLog(`🎬 正在将 ${files.length} 个高光合并为短片…`, "ok")
    setTask("正在合并成片")
    wsRef.current = openEventSocket(handleEvent)
    await new Promise((r) => setTimeout(r, 150))
    const res = await combineVideos({
      files,
      output: out,
      ...(c.music_path
        ? {
            music_path: c.music_path,
            music_mode: c.music_mode,
            music_volume: c.music_volume / 100,
          }
        : {}),
    })
    if (!res.ok) {
      appendLog(`✖ 短片合并失败：${res.error ?? "未知错误"}`, "err")
      toast.error(res.error ?? "短片合并失败")
      setRunning(false)
      setTask("")
      wsRef.current?.close()
      wsRef.current = null
    }
  }

  /** Open the events socket and give it a tick to connect before work starts. */
  const beginRun = async () => {
    setLog([])
    setFrames([])
    setUsedCache(false)
    // Only a highlight run arms the reel; clear it so a download/scan can't chain.
    pendingReelRef.current = null
    setProgress(0)
    setRunning(true)
    // Starting a run is exactly when the output matters.
    setLogOpen(true)
    wsRef.current = openEventSocket(handleEvent)
    await new Promise((r) => setTimeout(r, 150))
  }

  const failRun = (msg: string) => {
    appendLog(`✖ ${msg}`, "err")
    toast.error(msg)
    setRunning(false)
    wsRef.current?.close()
  }

  const onRun = async () => {
    if (!videos.length) return toast.error("请至少添加一个视频")
    if (totalPoints(cfg) === 0 && !cfg.skip_highlights)
      return toast.error("请至少设置一个评分项")

    // When combining, the reel gets the music once; don't bake it per-clip.
    const willCombine = cfg.combine_reel && videos.length > 1
    await beginRun()
    const res = await startRun(
      videos,
      toGuiConfig(cfg, output, videos, {
        avoidIds,
        avoidRanges,
        timeRange,
        duration,
        willCombine,
      }),
    )
    if (!res.ok) failRun(res.error ?? "启动失败")
  }

  /** Run -> Pause -> Resume, matching the Qt toggle_run tri-state. */
  const onToggleRun = async () => {
    if (!running) return onRun()
    if (paused) {
      await resumeRun()
      setPaused(false)
      appendLog("▶ 已继续")
    } else {
      await pauseRun()
      setPaused(true)
      appendLog("⏸ 处理流程已暂停")
    }
  }

  /** urls set = download exactly those (from the picker); otherwise scrape. */
  const onDownload = async (urls?: string[]) => {
    await beginRun()
    const res = await startDownload({
      url: dl.url,
      save_dir: dl.saveDir,
      download_full: dl.downloadFull,
      time_range_start: dl.rangeStart,
      time_range_end: dl.rangeEnd,
      concurrent: dl.concurrent,
      ...(urls?.length ? { video_urls: urls } : {}),
    })
    if (!res.ok) failRun(res.error ?? "启动下载失败")
  }

  const onCancel = async () => {
    await cancelRun()
    appendLog("⏹ 已请求取消…", "err")
  }

  /** Card-to-film in one job. The engine config comes from the other tabs, so
   *  the scoring the user already set up is what the automatic run uses. */
  const onAutoStart = async (opts: Parameters<typeof startAuto>[0]) => {
    setAutoStages({})
    await beginRun()
    // No video paths yet — the pipeline discovers them by copying the card, and
    // it names the film itself, so only the scoring settings carry over.
    setLastRoot(opts.dest_root)
    const res = await startAuto({
      ...opts,
      config: toGuiConfig(cfgRef.current, opts.output_name ?? "film.mp4", [], {
        avoidIds,
      }),
    })
    if (!res.ok) failRun(res.error ?? "启动处理流程失败")
  }

  const addVideos = async () => {
    const picked = await pickVideos()
    if (picked.length) setVideos((v) => [...new Set([...v, ...picked])])
  }

  const addFolder = async () => {
    const dir = await pickDirectory()
    if (!dir) return
    const res = await scanFolder(dir, true)
    if (!res.ok) return toast.error(res.error ?? "无法扫描文件夹")
    if (!res.files.length) return toast("该文件夹中未找到视频")
    let added = 0
    setVideos((v) => {
      const merged = [...new Set([...v, ...res.files])]
      added = merged.length - v.length
      return merged
    })
    toast.success(`已添加 ${added} 个视频`)
  }

  const pickMusic = async () => {
    const path = await pickAudioFile()
    if (path) set("music_path", path)
  }

  const launchEditor = async () => {
    if (!videos.length) return toast.error("请先添加视频")
    // The viewer is a separate Qt process and takes ~10s to appear, so say so —
    // otherwise the click looks like it did nothing.
    toast("正在打开时间线查看器，可能需要几秒钟…")
    appendLog(`📊 正在为 ${basename(videos[0])} 打开时间线查看器…`)
    const res = await openEditor(videos[0])
    if (!res.ok) {
      toast.error(res.error ?? "无法打开时间线查看器")
      appendLog(`✖ ${res.error}`, "err")
    }
  }

  return (
    // App shell: fixed header, one scrolling column, pinned action bar. A tool
    // window shouldn't scroll as a document -- the primary action and the log
    // have to stay reachable no matter how long the settings get.
    <div className="flex h-screen flex-col overflow-hidden">
      {/* Header */}
      <header className="flex shrink-0 items-center justify-between border-b px-5 py-2.5">
        <div className="flex items-center gap-2.5">
          <div className="grid size-7 place-items-center rounded bg-primary/15 text-primary">
            <Film className="size-4" />
          </div>
          <h1 className="text-sm font-semibold tracking-tight">
            Video Highlighter
          </h1>
          <span
            className="flex items-center gap-1.5 text-xs text-muted-foreground"
            title={online ? "Python 引擎连接正常" : "Python 引擎无响应"}
          >
            <span
              className={`size-1.5 rounded-full ${
                online === null
                  ? "bg-muted-foreground"
                  : online
                  ? "bg-[color:var(--success)]"
                  : "bg-destructive"
              }`}
            />
            {online === null ? "connecting" : online ? "引擎就绪" : "引擎离线"}
          </span>
        </div>
        <Button variant="ghost" size="icon" onClick={toggle} title="切换主题">
          {theme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
        </Button>
      </header>

      {/* The only scrolling region. */}
      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-5xl flex-col gap-4 p-5">

      {/* Input videos */}
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle className="text-sm font-medium">输入视频</CardTitle>
          <div className="flex gap-2">
            {/* Inputs lock during a run, same as the Qt GUI. */}
            <Button size="sm" variant="secondary" onClick={addVideos} disabled={running}>
              <Plus className="size-4" /> 添加
            </Button>
            <Button size="sm" variant="secondary" onClick={addFolder} disabled={running}>
              <FolderOpen className="size-4" /> 添加文件夹
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setVideos([])}
              disabled={!videos.length || running}
            >
              <Trash2 className="size-4" /> 清空
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {videos.length === 0 ? (
            <p className="rounded-md border border-dashed py-6 text-center text-sm text-muted-foreground">
              暂未添加视频
            </p>
          ) : (
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4">
              {videos.map((v) => (
                <VideoCard
                  key={v}
                  path={v}
                  disabled={running}
                  onRemove={() => setVideos((l) => l.filter((x) => x !== v))}
                />
              ))}
            </div>
          )}
          <Separator className="my-4" />
          <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">输出名称</Label>
            <Input
              value={output}
              onChange={(e) => setOutput(e.target.value)}
              disabled={running}
              className="h-8 w-full"
            />
          </div>

          <Separator className="my-4" />

          {/* Reel + music: turn many highlights into one soundtracked video. */}
          <div className="space-y-3">
            {videos.length > 1 && (
              <label className="flex items-center gap-2 text-sm">
                <Checkbox
                  checked={cfg.combine_reel}
                  disabled={running}
                  onCheckedChange={(v) => set("combine_reel", Boolean(v))}
                />
                合并为一个短片
              </label>
            )}
            <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-3">
              <Label className="text-sm text-muted-foreground">音乐</Label>
              <span
                className="min-w-0 truncate text-sm"
                title={cfg.music_path || undefined}
              >
                {cfg.music_path ? basename(cfg.music_path) : (
                  <span className="text-muted-foreground">无音乐</span>
                )}
              </span>
              <div className="flex gap-1">
                <Button size="sm" variant="secondary" onClick={pickMusic} disabled={running}>
                  Pick
                </Button>
                {cfg.music_path && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => set("music_path", "")}
                    disabled={running}
                  >
                    Clear
                  </Button>
                )}
              </div>
            </div>
            {cfg.music_path && (
              <div className="space-y-3">
                <SelectField
                  label="混合"
                  value={cfg.music_mode}
                  onChange={(v) => set("music_mode", v)}
                  options={MUSIC_MODES}
                  disabled={running}
                />
                <div className="flex min-w-0 items-center gap-3">
                  <Label className="text-sm font-normal text-muted-foreground">音量</Label>
                  <Slider
                    min={0}
                    max={100}
                    step={1}
                    value={[cfg.music_volume]}
                    onValueChange={([v]) => set("music_volume", v)}
                    disabled={running}
                    className="flex-1"
                  />
                  <span className="w-10 text-right text-sm tabular-nums text-muted-foreground">
                    {cfg.music_volume}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <TimeRange state={timeRange} onChange={setTimeRange} duration={duration} />

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={livePreview}
            onCheckedChange={(v) => {
              const on = Boolean(v)
              setLivePreview(on)
              // Frames only exist while detection is actually running. On an
              // already-analysed video the pipeline serves cached detections and
              // never calls preview_fn, so the panel would sit on "Waiting for
              // the detection stage" forever. Asking to watch detection means
              // asking for detection to happen.
              if (on && !cfg.force_reprocess) {
                set("force_reprocess", true)
                toast("已开启强制重新处理，以便显示检测帧")
              }
            }}
          />
          实时检测预览
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={cfg.force_reprocess}
            onCheckedChange={(v) => set("force_reprocess", Boolean(v))}
          />
          强制重新处理（忽略缓存）
        </label>
      </div>

      {livePreview && (
        <DetectionPreview
          frames={frames}
          running={running}
          cached={usedCache}
        />
      )}

      {/* Tabs */}
      <Tabs defaultValue="basic" className="min-w-0">
        <TabsList>
          <TabsTrigger value="auto">自动</TabsTrigger>
          <TabsTrigger value="reel">成片</TabsTrigger>
          <TabsTrigger value="timeline">时间线</TabsTrigger>
          <TabsTrigger value="download">下载</TabsTrigger>
          <TabsTrigger value="basic">基础</TabsTrigger>
          <TabsTrigger value="transcript">转录</TabsTrigger>
          <TabsTrigger value="advanced">高级</TabsTrigger>
          <TabsTrigger value="llm">大模型对话</TabsTrigger>
          <TabsTrigger value="search">视觉搜索</TabsTrigger>
          <TabsTrigger value="avoid">排除</TabsTrigger>
          <TabsTrigger value="about">关于</TabsTrigger>
        </TabsList>

        <TabsContent value="auto" className="mt-4">
          <AutoTab
            running={running}
            stages={autoStages}
            onStart={(o) => void onAutoStart(o)}
            onCancel={() => void onCancel()}
          />
        </TabsContent>
        <TabsContent value="reel" className="mt-4">
          <ReelTab
            running={running}
            onCancel={() => void onCancel()}
            suggestedRoot={lastRoot}
          />
        </TabsContent>
        <TabsContent value="timeline" className="mt-4">
          <TimelineTab
            running={running}
            onCancel={() => void onCancel()}
            suggestedPath={lastEdl}
          />
        </TabsContent>
        <TabsContent value="download" className="mt-4">
          <DownloadTab
            settings={dl}
            onChange={setDl}
            onDownload={() => onDownload()}
            onDownloadUrls={(urls) => onDownload(urls)}
            running={running}
          />
        </TabsContent>
        <TabsContent value="basic" className="mt-4">
          <BasicTab
            cfg={cfg}
            set={set}
            objectLabels={objectLabels}
            actionLabels={actionLabels}
          />
        </TabsContent>
        <TabsContent value="transcript" className="mt-4">
          <TranscriptTab cfg={cfg} set={set} />
        </TabsContent>
        <TabsContent value="advanced" className="mt-4">
          <AdvancedTab cfg={cfg} set={set} />
        </TabsContent>
        <TabsContent value="llm" className="mt-4">
          <LlmChatTab
            videoPath={videos[0]}
            backend={llmBackend}
            model={llmModel}
            onBackendChange={setLlmBackend}
            onModelChange={setLlmModel}
          />
        </TabsContent>
        <TabsContent value="search" className="mt-4">
          <VisionSearchTab
            videoPath={videos[0]}
            backend={llmBackend || "ollama"}
            model={llmModel || "llava"}
            running={running}
            results={visionResults}
            onStart={async () => {
              setVisionResults([])
              await beginRun()
            }}
          />
        </TabsContent>
        <TabsContent value="avoid" className="mt-4">
          <AvoidTab
            cfg={cfg}
            set={set}
            onAvoidIdsChange={setAvoidIds}
            videoPath={videos[0]}
            running={running}
            refreshKey={faceRefresh}
            avoidRanges={avoidRanges}
            onAvoidRangesChange={refreshAvoidRanges}
          />
        </TabsContent>
        <TabsContent value="about" className="mt-4">
          <AboutTab />
        </TabsContent>
      </Tabs>

        </div>
      </main>

      {/* Log dock: collapsible, and it only claims height when it has something
          to show. An empty panel holding 180px hostage is worse than no panel. */}
      {logOpen && log.length > 0 && (
        <div className="shrink-0 border-t bg-card/40">
          <div className="mx-auto w-full max-w-5xl">
            <ScrollArea className="h-44 px-5 py-3">
              <pre className="whitespace-pre-wrap font-mono text-xs leading-relaxed">
                {(
                  log.map((l, i) => (
                    <div
                      key={i}
                      className={
                        l.kind === "err"
                          ? "text-destructive"
                          : l.kind === "ok"
                          ? "text-[color:var(--success)]"
                          : "text-muted-foreground"
                      }
                    >
                      {l.text}
                    </div>
                  ))
                )}
                <div ref={logEndRef} />
              </pre>
            </ScrollArea>
          </div>
        </div>
      )}

      {/* Action bar — pinned. Everything the Qt bottom bar has: Cancel, keep
          temp, Timeline Viewer, debug log, the analyzed counter, and Run. */}
      <footer className="shrink-0 border-t bg-card/60 px-5 py-2.5">
        <div className="mx-auto flex w-full max-w-5xl items-center gap-3">
          <Button
            size="sm"
            onClick={onToggleRun}
            disabled={online === false}
            className={
              running
                ? paused
                  ? "gap-1.5 bg-primary text-primary-foreground hover:opacity-90"
                  : "gap-1.5 bg-[color:var(--warning)] text-black hover:opacity-90"
                : "gap-1.5 bg-[color:var(--success)] text-black hover:opacity-90"
            }
          >
            {!running ? (
              <>
                <Sparkles className="size-3.5" /> 开始生成高光
              </>
            ) : paused ? (
              <>
                <Play className="size-3.5" /> Resume
              </>
            ) : (
              <>
                <Pause className="size-3.5" /> Pause
              </>
            )}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={onCancel}
            disabled={!running}
            className="gap-1.5 text-destructive hover:text-destructive disabled:opacity-40"
          >
            <Square className="size-3.5" /> Cancel
          </Button>

          {/* Progress owns the middle: it's the only thing that changes while a
              run is going, so it gets the space rather than a row of buttons. */}
          <div className="min-w-0 flex-1">
            <div className="mb-1 flex justify-between gap-3 text-[11px] text-muted-foreground">
              <span className="truncate">{task || (running ? "处理中…" : "空闲")}</span>
              <span className="tabular-nums">{progress}%</span>
            </div>
            <Progress value={progress} className="h-1" />
          </div>

          <div className="flex shrink-0 items-center gap-1">
            <label
              className="flex cursor-pointer items-center gap-1.5 px-1 text-xs text-muted-foreground"
              title="合并完成后保留中间片段，而不是删除"
            >
              <Checkbox
                checked={cfg.keep_temp}
                onCheckedChange={(v) => set("keep_temp", Boolean(v))}
              />
              保留临时文件
            </label>
            <Button
              size="sm"
              variant="ghost"
              onClick={launchEditor}
              disabled={!videos.length}
              title={
                videos.length
                  ? "打开第一个视频的原生时间线查看器"
                  : "请先添加视频"
              }
              className="gap-1.5"
            >
              <MonitorPlay className="size-3.5" /> 时间线
            </Button>
            {lastOutput && (
              <Button
                size="sm"
                variant="ghost"
                onClick={async () => {
                  const res = await revealOutput(lastOutput)
                  if (!res.ok) toast.error(res.error ?? "无法显示输出文件")
                }}
                title={`在文件管理器中显示 ${lastOutput}`}
                className="gap-1.5"
              >
                <FolderOpen className="size-3.5" /> 显示输出文件
              </Button>
            )}
            <Button
              size="sm"
              variant="ghost"
              onClick={async () => {
                const res = await revealLog()
                if (!res.ok) toast.error(res.error ?? "暂无日志可显示")
              }}
              title="在文件管理器中显示 debug.log"
              className="gap-1.5"
            >
              <FileText className="size-3.5" /> 日志文件
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={() => setLogOpen((v) => !v)}
              disabled={!log.length}
              title={
                !log.length
                  ? "尚无输出"
                  : logOpen
                  ? "隐藏输出面板"
                  : "显示输出面板"
              }
              className="gap-1.5"
            >
              {logOpen ? (
                <ChevronDown className="size-3.5" />
              ) : (
                <ChevronUp className="size-3.5" />
              )}
              Output
              {log.length > 0 && (
                <span className="tabular-nums opacity-60">{log.length}</span>
              )}
            </Button>
            {analyzed !== null && (
              <span
                className="ml-1 border-l pl-2.5 text-xs tabular-nums text-muted-foreground"
                title="已成功分析的视频。累计总数会跨会话保留。"
              >
                {analyzed} analyzed
                {sessionCount > 0 && ` · 本次运行 ${sessionCount} 个`}
              </span>
            )}
          </div>
        </div>
      </footer>

      <Toaster richColors position="top-right" />
    </div>
  )
}
