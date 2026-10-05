import { useEffect, useState } from "react"
import { RefreshCw, UserX, ExternalLink, ScanFace, Trash2, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { SelectField } from "@/components/SelectField"
import { AVOID_METHODS, type HighlighterConfig } from "@/lib/config"
import {
  getFaces,
  setFaceAvoid,
  removeFace,
  nameFace,
  clearFaces,
  scanFaces,
  saveAvoidRanges,
  openEditor,
  type FaceIdentity,
} from "@/lib/api"
import { toast } from "sonner"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
  onAvoidIdsChange: (ids: string[]) => void
  /** First input video — the scan target, matching the Qt single-video rule. */
  videoPath?: string
  running: boolean
  /** Bumped by App when a faces_scanned event arrives, to trigger a refresh. */
  refreshKey?: number
  /** Ranges marked in the native Timeline Viewer, via the shared store. */
  avoidRanges: [number, number][]
  onAvoidRangesChange: () => void
}

const fmtT = (s: number) =>
  `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`

/** "1:30" or "90" -> seconds; null if it isn't either. Mirrors the formats
 *  manual_avoid._parse_time_token accepts, so what the field takes matches
 *  what the engine stores. */
const parseTime = (raw: string): number | null => {
  const t = raw.trim()
  if (!t) return null
  const parts = t.split(":")
  if (parts.length > 2) return null
  const nums = parts.map(Number)
  if (nums.some((n) => !Number.isFinite(n) || n < 0)) return null
  const secs = parts.length === 2 ? nums[0] * 60 + nums[1] : nums[0]
  return secs < 0 ? null : secs
}

export function AvoidTab({
  cfg,
  set,
  onAvoidIdsChange,
  videoPath,
  running,
  refreshKey,
  avoidRanges,
  onAvoidRangesChange,
}: Props) {
  const [faces, setFaces] = useState<FaceIdentity[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clearOpen, setClearOpen] = useState(false)
  // Manual range entry — "1:30" or "90" both work (parsed server-side).
  const [rangeStart, setRangeStart] = useState("")
  const [rangeEnd, setRangeEnd] = useState("")

  /** Persist `next` and refresh; the engine merges overlaps and validates. */
  const writeRanges = async (next: [number, number][], okMsg: string) => {
    if (!videoPath) return
    const res = await saveAvoidRanges(videoPath, next)
    if (!res.ok) return toast.error(res.error ?? "无法保存区间")
    onAvoidRangesChange()
    toast.success(okMsg)
  }

  const addRange = async () => {
    const a = parseTime(rangeStart)
    const b = parseTime(rangeEnd)
    if (a === null || b === null) return toast.error("请输入 mm:ss 或秒数")
    if (b <= a) return toast.error("结束时间必须晚于开始时间")
    await writeRanges([...avoidRanges, [a, b]], "已添加避开时间段")
    setRangeStart("")
    setRangeEnd("")
  }

  const refresh = async () => {
    setLoading(true)
    const res = await getFaces()
    setLoading(false)
    if (!res.ok) {
      setError(res.error ?? "人脸库不可用")
      setFaces([])
      return
    }
    setError(null)
    setFaces(res.identities)
    onAvoidIdsChange(res.identities.filter((f) => f.avoid).map((f) => f.id))
  }

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey])

  const syncAvoidIds = (next: FaceIdentity[]) =>
    onAvoidIdsChange(next.filter((x) => x.avoid).map((x) => x.id))

  const toggle = async (f: FaceIdentity, next: boolean) => {
    // Optimistic: reflect immediately, revert if the bank write fails.
    const optimistic = faces.map((x) =>
      x.id === f.id ? { ...x, avoid: next } : x,
    )
    setFaces(optimistic)
    syncAvoidIds(optimistic)
    const res = await setFaceAvoid(f.id, next)
    if (!res.ok) {
      toast.error(res.error ?? "无法更新人脸库")
      void refresh()
    }
  }

  const rename = async (f: FaceIdentity) => {
    const name = window.prompt("为此人命名：", f.name)
    if (name === null) return
    const res = await nameFace(f.id, name)
    if (!res.ok) return toast.error(res.error ?? "无法设置名称")
    if (res.merged_into) toast.success("已合并到现有人物")
    void refresh()
  }

  const remove = async (f: FaceIdentity) => {
    const res = await removeFace(f.id)
    if (!res.ok) return toast.error("无法移除")
    void refresh()
  }

  const doClear = async (keepNamed: boolean) => {
    setClearOpen(false)
    const res = await clearFaces(keepNamed)
    if (!res.ok) return toast.error(res.error ?? "无法清除")
    toast.success(`已清除——保留 ${res.remaining} 项`)
    void refresh()
  }

  const scan = async () => {
    if (!videoPath) return toast.error("请先添加视频")
    const res = await scanFaces(videoPath)
    if (!res.ok) toast.error(res.error ?? "无法开始扫描")
    else toast("正在扫描人脸，请查看下方日志")
  }

  const avoidCount = faces.filter((f) => f.avoid).length
  const namedCount = faces.filter((f) => f.name).length

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between space-y-0">
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <UserX className="size-4" /> 避开人物
          {avoidCount > 0 && <Badge>{avoidCount} 个已避开</Badge>}
        </CardTitle>
        <div className="flex gap-2">
          <Button size="sm" variant="secondary" onClick={refresh} disabled={loading}>
            <RefreshCw className={loading ? "size-4 animate-spin" : "size-4"} />
            刷新
          </Button>
          <Button size="sm" variant="secondary" onClick={scan} disabled={running}>
            <ScanFace className="size-4" /> 扫描视频
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setClearOpen(true)}
            disabled={!faces.length}
          >
            <Trash2 className="size-4" /> 清空
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <label className="flex items-center gap-2 text-sm">
          <Checkbox
            checked={cfg.avoid_enabled}
            onCheckedChange={(v) => set("avoid_enabled", Boolean(v))}
          />
          启用人脸识别
        </label>
        <p className="text-xs text-muted-foreground">
          扫描视频以收集出现过的人物，然后勾选需要排除的人。
          你也可以在这里或时间线查看器中为人物命名。
        </p>

        <SelectField
          label="识别到时"
          value={cfg.avoid_method}
          options={AVOID_METHODS}
          onChange={(v) => set("avoid_method", v)}
          disabled={!cfg.avoid_enabled}
        />

        <div className="rounded-md border">
          {error ? (
            <p className="p-4 text-center text-sm text-destructive">{error}</p>
          ) : faces.length === 0 ? (
            <div className="space-y-3 p-6 text-center">
              <p className="text-sm text-muted-foreground">
                人脸库中暂无人物。请先扫描视频，或在时间线查看器中为人物命名。
              </p>
              <Button
                size="sm"
                variant="outline"
                disabled={!videoPath}
                onClick={async () => {
                  // Separate Qt process; it takes a few seconds to show up.
                  toast("正在打开时间线查看器，可能需要几秒钟…")
                  const res = await openEditor(videoPath)
                  if (!res.ok) toast.error(res.error ?? "无法打开编辑器")
                }}
              >
                <ExternalLink className="size-4" /> 打开时间线查看器
              </Button>
            </div>
          ) : (
            <ul className="divide-y">
              {faces.map((f) => (
                <li key={f.id} className="flex items-center gap-3 px-3 py-2 text-sm">
                  <Checkbox
                    checked={f.avoid}
                    onCheckedChange={(v) => toggle(f, Boolean(v))}
                    disabled={!cfg.avoid_enabled}
                  />
                  {f.thumb ? (
                    <img
                      src={`data:image/jpeg;base64,${f.thumb}`}
                      alt=""
                      className="size-10 shrink-0 rounded object-cover"
                    />
                  ) : (
                    <div className="size-10 shrink-0 rounded bg-muted" />
                  )}
                  <button
                    className="min-w-0 flex-1 truncate text-left hover:underline"
                    onClick={() => rename(f)}
                    title="点击命名"
                  >
                    <span className="font-medium">{f.label}</span>
                  </button>
                  <span className="shrink-0 text-xs text-muted-foreground">
                    出现 {f.count} 次
                  </span>
                  <button
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    onClick={() => remove(f)}
                    title="移除"
                  >
                    <X className="size-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
        {faces.length > 0 && (
          <p className="text-xs text-muted-foreground">
            {faces.length} 人 · {namedCount} 个已命名 · {avoidCount} 个已避开
          </p>
        )}

        {/* Time ranges marked in the native Timeline Viewer. They're stored per
            video, so they apply to runs started here too. */}
        <div className="space-y-2 border-t pt-3">
          <div className="flex items-center justify-between">
            <p className="text-sm font-medium">
              已避开的时间范围
              {avoidRanges.length > 0 && (
                <span className="ml-2 text-xs font-normal text-muted-foreground">
                  {avoidRanges.length} 个区间
                </span>
              )}
            </p>
            {avoidRanges.length > 0 && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => writeRanges([], "已清空排除区间")}
              >
                全部清除
              </Button>
            )}
          </div>
          {avoidRanges.length === 0 ? (
            <p className="text-xs text-muted-foreground">
              暂无区间。你可以在下方添加，或在时间线查看器中拖动选择一个范围；两种方式都会应用到后续运行。
            </p>
          ) : (
            <ul className="flex flex-wrap gap-2">
              {avoidRanges.map(([a, b], i) => (
                <li
                  key={i}
                  className="flex items-center gap-1.5 rounded-md bg-muted px-2 py-1 text-xs tabular-nums"
                >
                  {fmtT(a)} – {fmtT(b)}
                  <button
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() =>
                      writeRanges(
                        avoidRanges.filter((_, j) => j !== i),
                        "已移除排除区间",
                      )
                    }
                    title="移除此区间"
                  >
                    <X className="size-3" />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <div className="flex items-end gap-2">
            <Input
              value={rangeStart}
              onChange={(e) => setRangeStart(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addRange()}
              placeholder="1:30"
              aria-label="区间开始"
              disabled={!videoPath}
              className="h-8 w-24 text-xs tabular-nums"
            />
            <span className="pb-1.5 text-xs text-muted-foreground">至</span>
            <Input
              value={rangeEnd}
              onChange={(e) => setRangeEnd(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addRange()}
              placeholder="2:00"
              aria-label="区间结束"
              disabled={!videoPath}
              className="h-8 w-24 text-xs tabular-nums"
            />
            <Button
              size="sm"
              variant="secondary"
              onClick={addRange}
              disabled={!videoPath || !rangeStart || !rangeEnd}
              title={videoPath ? "排除此时间段" : "请先添加视频"}
            >
              添加
            </Button>
          </div>
        </div>
      </CardContent>

      <Dialog open={clearOpen} onOpenChange={setClearOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>清空人脸库（{faces.length} 个身份）？</DialogTitle>
            <DialogDescription>选择要排除的内容。</DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="ghost" onClick={() => setClearOpen(false)}>
              取消
            </Button>
            <Button variant="secondary" onClick={() => doClear(true)}>
              保留已命名 / 已避开项
            </Button>
            <Button variant="destructive" onClick={() => doClear(false)}>
              清除全部
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </Card>
  )
}
