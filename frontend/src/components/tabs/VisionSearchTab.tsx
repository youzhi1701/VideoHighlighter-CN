import { useEffect, useState } from "react"
import { Search, Square, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { NumberField } from "@/components/NumberField"
import { SelectField } from "@/components/SelectField"
import { basename } from "@/lib/files"
import {
  getClipStatus,
  visionSearch,
  cancelRun,
  type VisionResult,
} from "@/lib/api"
import { toast } from "sonner"

const MODES = [
  { value: "clip", label: "仅 CLIP（快速，GPU 排序）" },
  { value: "clip_llm", label: "CLIP + 视觉模型（复核最佳匹配）" },
  { value: "llm", label: "仅视觉模型（较慢，能力最强）" },
]
const DEVICES = [
  { value: "GPU", label: "GPU（Intel/OpenVINO）" },
  { value: "CPU", label: "CPU" },
]

interface Props {
  videoPath?: string
  backend: string
  model: string
  running: boolean
  results: VisionResult[]
  onStart: () => void
}

const fmt = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

export function VisionSearchTab({
  videoPath,
  backend,
  model,
  running,
  results,
  onStart,
}: Props) {
  const [query, setQuery] = useState("")
  const [mode, setMode] = useState<"clip" | "clip_llm" | "llm">("clip")
  const [interval, setIntervalS] = useState(1)
  const [topK, setTopK] = useState(30)
  const [threshold, setThreshold] = useState(0.5)
  const [device, setDevice] = useState("GPU")
  const [clipOk, setClipOk] = useState<boolean | null>(null)
  const [clipErr, setClipErr] = useState<string | null>(null)

  useEffect(() => {
    void getClipStatus().then((r) => {
      setClipOk(r.available)
      setClipErr(r.error ?? null)
    })
  }, [])

  const needsClip = mode !== "llm"
  const blocked = needsClip && clipOk === false

  const start = async () => {
    if (!videoPath) return toast.error("请先添加视频")
    if (!query.trim()) return toast.error("请输入要搜索的内容")
    onStart()
    const res = await visionSearch({
      video_path: videoPath,
      query: query.trim(),
      mode,
      interval,
      top_k: topK,
      threshold,
      clip_device: device,
      backend,
      model,
    })
    if (!res.ok) toast.error(res.error ?? "无法启动搜索")
  }

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Search className="size-4" /> 视觉搜索
          {videoPath && (
            <Badge variant="secondary" className="font-normal">
              {basename(videoPath)}
            </Badge>
          )}
        </CardTitle>
        {running && (
          <Button size="sm" variant="destructive" onClick={() => cancelRun()}>
            <Square className="size-4" /> 取消
          </Button>
        )}
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          用文字描述想找的画面。CLIP 会对采样帧与描述进行匹配评分，视觉模型还可以进一步确认最符合的结果。
        </p>

        <div className="flex gap-2">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && !running && start()}
            placeholder="一个骑自行车的人"
            disabled={running}
          />
          <Button onClick={start} disabled={running || blocked} className="gap-2">
            <Sparkles className="size-4" /> 搜索
          </Button>
        </div>

        <div className="grid gap-2.5 md:grid-cols-2">
          <SelectField
            label="搜索引擎"
            value={mode}
            options={MODES}
            onChange={(v) => setMode(v as typeof mode)}
          />
          {needsClip && (
            <SelectField
              label="CLIP 设备"
              value={device}
              options={DEVICES}
              onChange={setDevice}
            />
          )}
          <NumberField
            label="采样间隔"
            hint="（秒）"
            value={interval}
            step={0.5}
            min={0.1}
            onChange={setIntervalS}
          />
          <NumberField label="最大结果数" value={topK} min={1} onChange={setTopK} />
          {mode === "clip" && (
            <NumberField
              label="分数阈值"
              hint="（0-1）"
              value={threshold}
              step={0.05}
              onChange={setThreshold}
            />
          )}
        </div>

        {blocked && (
          <p className="text-xs text-destructive">
            CLIP 当前不可用：{clipErr}。请选择“仅视觉模型”搜索引擎，或安装 OpenVINO CLIP 组件。
          </p>
        )}

        <div className="rounded-md border">
          {results.length === 0 ? (
            <p className="p-6 text-center text-sm text-muted-foreground">
              {running ? "正在搜索…" : "暂无搜索结果。"}
            </p>
          ) : (
            <ul className="divide-y">
              {results.map((r, i) => (
                <li key={i} className="flex gap-3 p-2 text-sm">
                  {r.thumb && (
                    <img
                      src={`data:image/jpeg;base64,${r.thumb}`}
                      alt=""
                      className="h-16 w-28 shrink-0 rounded object-cover"
                    />
                  )}
                  <div className="min-w-0 flex-1">
                    <p className="flex items-center gap-2">
                      <span className="font-medium tabular-nums">
                        {fmt(r.timestamp)}
                      </span>
                      {r.score > 0 && (
                        <Badge variant="secondary" className="tabular-nums">
                          {(r.score * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </p>
                    {r.analysis && (
                      <p className="line-clamp-3 text-xs text-muted-foreground">
                        {r.analysis}
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
        {results.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {results.length} match(es) — timestamps are seconds into the video.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
