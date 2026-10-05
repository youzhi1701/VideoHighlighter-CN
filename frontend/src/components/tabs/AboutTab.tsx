import { useEffect, useState } from "react"
import { Film } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { getAbout, type AboutInfo } from "@/lib/api"

const Link = ({ href, children }: { href: string; children: React.ReactNode }) => (
  <a
    href={href}
    target="_blank"
    rel="noreferrer"
    className="text-primary hover:underline"
  >
    {children}
  </a>
)

export function AboutTab() {
  const [info, setInfo] = useState<AboutInfo | null>(null)

  useEffect(() => {
    void getAbout().then((r) => r.ok && setInfo(r))
  }, [])

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <div className="grid size-12 place-items-center rounded-xl bg-primary/15 text-primary">
          <Film className="size-6" />
        </div>
        <div>
          <h2 className="text-lg font-semibold">
            Video Highlighter {info?.edition && `(${info.edition})`}
          </h2>
          <p className="text-sm text-muted-foreground">
            {info ? `版本 ${info.version} — 免费开源（AGPLv3）` : "…"}
          </p>
        </div>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">VideoHighlighter Pro</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p>
            你正在使用免费开源版本——人物身份识别、
            表情识别、报告和助手功能都已包含。{" "}
            <strong>Pro</strong> 还能让程序学习你自己的识别词汇：
            可从你的示例画面学习类别、按示例搜索、
            开放词汇检测、实时叠加显示，以及商业许可证。
          </p>
          <p>
            {info ? (
              <Link href={info.website}>了解更多 / 获取 Pro</Link>
            ) : (
              <Link href="https://aseiel.github.io/VideoHighlighter-site/">
                了解更多 / 获取 Pro
              </Link>
            )}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">联系与支持</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          {info && (
            <>
              <p>
                邮箱：{" "}
                <Link
                  href={`mailto:${info.support_email}?subject=VideoHighlighter%20support`}
                >
                  {info.support_email}
                </Link>
              </p>
              <p>
                Discord：<Link href={info.discord}>加入社区</Link>
              </p>
              <p>
                网站：<Link href={info.website}>{info.website}</Link>
              </p>
              <p>
                源代码：<Link href={info.repo}>{info.repo}</Link>
              </p>
              {info.log_path && (
                <p className="pt-1 text-xs text-muted-foreground">
                  报告问题时，请附上调试日志：{info.log_path}
                </p>
              )}
            </>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">法律信息</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm text-muted-foreground">
          <p>© 2026 Przemysław Kreft 及贡献者</p>
          <p>
            采用以下许可证：{" "}
            <Link href="https://www.gnu.org/licenses/agpl-3.0.html">AGPLv3</Link>.
          </p>
          <p className="text-xs">
            第三方组件包括 PySide6（Qt）和 FFmpeg，分别遵循其
            各自的许可证。
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
