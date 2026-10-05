import { useEffect, useState } from "react"
import { Plus, Save, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { getCompositionRules, saveCompositionRules, type CompRule } from "@/lib/api"
import { toast } from "sonner"

const BLANK: CompRule = {
  name: "",
  label: "",
  source: "",
  region: "",
  min_count: 1,
  max_count: 999,
  relation: "inside",
  outline: false,
  window_secs: 0.75,
  persist_secs: 0.5,
}

export function CompositionRules() {
  const [rules, setRules] = useState<CompRule[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    void getCompositionRules().then((r) => r.ok && setRules(r.rules))
  }, [])

  const upd = (i: number, patch: Partial<CompRule>) =>
    setRules((rs) => rs.map((r, j) => (j === i ? { ...r, ...patch } : r)))

  const save = async () => {
    setLoading(true)
    const res = await saveCompositionRules(rules)
    setLoading(false)
    if (res.ok) toast.success(`已将 ${res.events} 个事件保存到 composition_rules.yaml`)
    else toast.error(res.error ?? "无法保存规则")
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">构图规则</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-xs text-muted-foreground">
          通过检测物体之间的空间关系组合出更高层级的动作。例如，当物体 A 多次出现在区域 B 内时触发动作 X。同一“事件名称”的多行条件必须同时满足（AND）。关系：内部（中心位于区域内）、重叠（大部分位于区域内）、接触（边界相交）。轮廓选项会根据框内真实形状判断；person.hand 之类的来源可使用身体部位。时间窗口可平滑短暂抖动；持续时间可让物体在短暂遮挡时继续有效。规则保存在 composition_rules.yaml。
        </p>

        <div className="overflow-x-auto rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="min-w-28">事件名称</TableHead>
                <TableHead className="min-w-28">显示名称</TableHead>
                <TableHead className="min-w-24">来源</TableHead>
                <TableHead className="min-w-24">区域</TableHead>
                <TableHead className="w-20">最小</TableHead>
                <TableHead className="w-20">最大</TableHead>
                <TableHead className="w-28">关系</TableHead>
                <TableHead className="w-16">轮廓</TableHead>
                <TableHead className="w-24">时间窗口（秒）</TableHead>
                <TableHead className="w-24">持续（秒）</TableHead>
                <TableHead className="w-10" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {rules.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={11}
                    className="text-center text-sm text-muted-foreground"
                  >
                    No rules yet
                  </TableCell>
                </TableRow>
              ) : (
                rules.map((r, i) => (
                  <TableRow key={i}>
                    {(["name", "label", "source", "region"] as const).map((f) => (
                      <TableCell key={f} className="p-1">
                        <Input
                          value={r[f]}
                          onChange={(e) => upd(i, { [f]: e.target.value })}
                          className="h-7"
                        />
                      </TableCell>
                    ))}
                    {(
                      [
                        ["min_count", 1],
                        ["max_count", 1],
                        ["window_secs", 0.25],
                        ["persist_secs", 0.25],
                      ] as const
                    ).map(([f, step]) => (
                      <TableCell key={f} className="p-1">
                        <Input
                          type="number"
                          step={step}
                          value={r[f]}
                          onChange={(e) => upd(i, { [f]: Number(e.target.value) })}
                          className="h-7 text-right tabular-nums"
                        />
                      </TableCell>
                    ))}
                    <TableCell className="p-1">
                      <select
                        value={r.relation ?? "inside"}
                        onChange={(e) =>
                          upd(i, { relation: e.target.value as CompRule["relation"] })
                        }
                        className="h-7 w-full rounded-md border bg-transparent px-1 text-sm"
                        title="内部：中心位于区域内 · 重叠：大部分位于区域内 · 接触：边界相交"
                      >
                        <option value="inside">内部</option>
                        <option value="overlaps">重叠</option>
                        <option value="touches">接触</option>
                      </select>
                    </TableCell>
                    <TableCell className="p-1 text-center">
                      <input
                        type="checkbox"
                        checked={Boolean(r.outline)}
                        onChange={(e) => upd(i, { outline: e.target.checked })}
                        title="根据检测框内的真实轮廓判断形状"
                      />
                    </TableCell>
                    <TableCell className="p-1">
                      <button
                        className="text-destructive hover:opacity-70"
                        onClick={() =>
                          setRules((rs) => rs.filter((_, j) => j !== i))
                        }
                        title="删除规则"
                      >
                        <X className="size-4" />
                      </button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex justify-between">
          <Button
            size="sm"
            variant="secondary"
            onClick={() => setRules((rs) => [...rs, { ...BLANK }])}
          >
            <Plus className="size-4" /> Add Rule
          </Button>
          <Button size="sm" onClick={save} disabled={loading}>
            <Save className="size-4" /> Save Rules
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
