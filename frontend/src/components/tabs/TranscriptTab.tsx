import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SelectField } from "@/components/SelectField"
import { NumberField } from "@/components/NumberField"
import {
  SUBTITLE_LANGS,
  TRANSCRIPT_LANGS,
  WHISPER_MODELS,
  type HighlighterConfig,
} from "@/lib/config"

interface Props {
  cfg: HighlighterConfig
  set: <K extends keyof HighlighterConfig>(k: K, v: HighlighterConfig[K]) => void
}

export function TranscriptTab({ cfg, set }: Props) {
  // Subtitles require a transcript, and the keyword/transcript scores only
  // count when transcript runs — same gating as the Qt tab.
  const on = cfg.use_transcript
  return (
    <div className="grid min-w-0 gap-5 md:grid-cols-2 [&>*]:min-w-0">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">转录</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.use_transcript}
              onCheckedChange={(v) => set("use_transcript", Boolean(v))}
            />
            启用语音转写处理（Whisper）
          </label>
          <SelectField
            label="源语言"
            value={cfg.transcript_source_lang}
            options={TRANSCRIPT_LANGS}
            onChange={(v) => set("transcript_source_lang", v)}
            disabled={!on}
          />
          <SelectField
            label="Whisper 模型"
            value={cfg.transcript_model}
            options={WHISPER_MODELS}
            onChange={(v) => set("transcript_model", v)}
            disabled={!on}
          />
          <div className="grid min-w-0 grid-cols-[minmax(0,1fr)_14rem] items-center gap-3">
            <Label className="min-w-0 truncate text-sm font-normal text-muted-foreground">
              搜索关键词
            </Label>
            <Input
              value={cfg.search_keywords}
              onChange={(e) => set("search_keywords", e.target.value)}
              placeholder="进球、得分、获胜"
              className="h-8 w-full"
              disabled={!on}
            />
          </div>
          <div className="space-y-2.5 border-t pt-3">
            <p className="text-xs text-muted-foreground">
              仅在启用语音转写时才计入这些分数。
            </p>
            <NumberField
              label="关键词加分"
              value={cfg.keyword_points}
              onChange={(v) => set("keyword_points", v)}
            />
            <NumberField
              label="转录文本加分"
              hint="（所有词）"
              value={cfg.transcript_points}
              onChange={(v) => set("transcript_points", v)}
            />
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">字幕</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <label className="flex items-center gap-2 text-sm">
            <Checkbox
              checked={cfg.create_subtitles}
              onCheckedChange={(v) => set("create_subtitles", Boolean(v))}
              disabled={!on}
            />
            生成字幕（.srt）
          </label>
          {!on && (
            <p className="text-xs text-muted-foreground">
              请先启用语音转写，然后再生成字幕。
            </p>
          )}
          <SelectField
            label="源语言"
            value={cfg.source_lang}
            options={SUBTITLE_LANGS}
            onChange={(v) => set("source_lang", v)}
            disabled={!on || !cfg.create_subtitles}
          />
          <SelectField
            label="目标语言"
            value={cfg.target_lang}
            options={SUBTITLE_LANGS}
            onChange={(v) => set("target_lang", v)}
            disabled={!on || !cfg.create_subtitles}
          />
        </CardContent>
      </Card>
    </div>
  )
}
