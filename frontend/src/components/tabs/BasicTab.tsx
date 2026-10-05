import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { NumberField } from "@/components/NumberField"
import { totalPoints, type HighlighterConfig } from "@/lib/config"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
  objectLabels: string[]
  actionLabels: string[]
}

export function BasicTab({ cfg, set, objectLabels, actionLabels }: Props) {
  return (
    <div className="space-y-5">
      <div className="grid min-w-0 gap-5 md:grid-cols-2 [&>*]:min-w-0">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0">
            <CardTitle className="text-sm font-medium">评分项</CardTitle>
            <Badge variant={totalPoints(cfg) ? "default" : "secondary"}>
              total {totalPoints(cfg)}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <NumberField label="场景" value={cfg.scene_points} onChange={(v) => set("scene_points", v)} />
            <NumberField label="运动事件" value={cfg.motion_event_points} onChange={(v) => set("motion_event_points", v)} />
            <NumberField label="运动峰值" value={cfg.motion_peak_points} onChange={(v) => set("motion_peak_points", v)} />
            <NumberField label="音频峰值" value={cfg.audio_peak_points} onChange={(v) => set("audio_peak_points", v)} />
            <NumberField label="物体" value={cfg.object_points} onChange={(v) => set("object_points", v)} />
            <NumberField label="动作" value={cfg.action_points} onChange={(v) => set("action_points", v)} />
            <p className="pt-1 text-xs text-muted-foreground">
              关键词和转录文本加分位于“转录”页，
              只有启用转录后才会生效。
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium">时长与剪切</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2.5">
            <NumberField label="高光片段最长时长" hint="(s)" value={cfg.max_duration} onChange={(v) => set("max_duration", v)} />
            <NumberField label="固定时长" hint="（0=关闭）" value={cfg.exact_duration} onChange={(v) => set("exact_duration", v)} />
            <NumberField label="片段时长" hint="（0=自动）" value={cfg.clip_time} onChange={(v) => set("clip_time", v)} />
            <Separator className="my-1" />
            <p className="text-xs text-muted-foreground">
              {cfg.clip_time === 0
                ? "自动模式：片段边界根据动作、场景切换和峰值等信号结构确定。"
                : `固定模式：每个片段时长均为 ${cfg.clip_time} 秒。`}
            </p>
            <NumberField label="自动片段最短时长" hint="(s)" value={cfg.auto_min_clip} step={0.5} onChange={(v) => set("auto_min_clip", v)} />
            <NumberField label="自动片段最长时长" hint="(s)" value={cfg.auto_max_clip} step={0.5} onChange={(v) => set("auto_max_clip", v)} />
            <NumberField label="合并间隔" hint="(s)" value={cfg.auto_merge_gap} step={0.5} onChange={(v) => set("auto_merge_gap", v)} />
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">检测目标</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="grid min-w-0 grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">物体</Label>
            <Input
              list="object-labels"
              value={cfg.highlight_objects}
              onChange={(e) => set("highlight_objects", e.target.value)}
              placeholder="person, sports ball, dog"
              className="h-8 w-full"
            />
            <datalist id="object-labels">
              {objectLabels.map((l) => (
                <option key={l} value={l} />
              ))}
            </datalist>
          </div>
          <div className="grid min-w-0 grid-cols-[4.5rem_minmax(0,1fr)] items-center gap-3">
            <Label className="text-sm text-muted-foreground">动作</Label>
            <Input
              list="action-labels"
              value={cfg.interesting_actions}
              onChange={(e) => set("interesting_actions", e.target.value)}
              placeholder="high jump, high kick, archery"
              className="h-8 w-full"
            />
            <datalist id="action-labels">
              {actionLabels.map((l) => (
                <option key={l} value={l} />
              ))}
            </datalist>
          </div>
          <div className="flex flex-wrap gap-5 pt-1">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.actions_require_objects}
                onCheckedChange={(v) => set("actions_require_objects", Boolean(v))}
              />
              仅在检测到物体时给动作评分
            </label>
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.keep_temp}
                onCheckedChange={(v) => set("keep_temp", Boolean(v))}
              />
              保留临时片段
            </label>
            {/* Force reprocess lives on the main screen next to Live preview,
                where the Qt app puts it and where it's actually needed. */}
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={cfg.skip_highlights}
                onCheckedChange={(v) => set("skip_highlights", Boolean(v))}
              />
              跳过高光生成
            </label>
            <label
              className="flex items-center gap-2 text-sm"
              title="根据拉普拉斯方差评估清晰度；模糊片段会被降权，在兴趣度相同时优先选择更清晰的片段。"
            >
              <Checkbox
                checked={cfg.quality_gate}
                onCheckedChange={(v) => set("quality_gate", Boolean(v))}
              />
              降低模糊片段评分
            </label>
          </div>
          {cfg.quality_gate && (
            <div className="max-w-xs pt-1">
              <NumberField
                label="清晰度阈值"
                hint="（越低越严格）"
                value={cfg.quality_threshold}
                onChange={(v) => set("quality_threshold", v)}
              />
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
