<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter-CN" width="160">
</p>

# VideoHighlighter-CN

**VideoHighlighter 简体中文本地化版本**

本仓库基于上游开源项目 **Aseiel/VideoHighlighter** 持续维护。目标不是做一次性汉化，而是建立一套可以长期跟随官方更新的中文本地化版本。

<p align="center">
  <a href="https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/一键下载-当前中文版源码_ZIP-brightgreen?style=for-the-badge" alt="一键下载中文版">
  </a>
  &nbsp;
  <a href="https://github.com/youzhi1701/VideoHighlighter-CN/tree/main/%E4%B8%AD%E6%96%87%E6%B1%89%E5%8C%96%E6%8F%92%E4%BB%B6">
    <img src="https://img.shields.io/badge/中文汉化插件-打开目录-blue?style=for-the-badge" alt="中文汉化插件">
  </a>
</p>

> **一键下载说明：** 点击上方“当前中文版源码 ZIP”即可直接下载整个中文版项目。  
> Windows 安装版后续发布到 GitHub Releases 后，会在这里增加固定的“一键下载安装版”按钮。

---

## Windows 电脑端一键下载并运行

打开 **PowerShell**，复制下面这一整行并回车：

```powershell
$u='https://raw.githubusercontent.com/youzhi1701/VideoHighlighter-CN/main/Windows%E4%B8%80%E9%94%AE%E4%B8%8B%E8%BD%BD%E5%B9%B6%E8%BF%90%E8%A1%8C.ps1'; $p="$env:TEMP\VideoHighlighter-CN.ps1"; Invoke-WebRequest $u -OutFile $p -UseBasicParsing; powershell -ExecutionPolicy Bypass -File $p
```

它会自动完成：

- 下载最新版中文版源码
- 解压到 `%LOCALAPPDATA%\VideoHighlighter-CN`
- 检查并安装 Python 3.12（缺失时使用 winget）
- 创建独立虚拟环境
- 安装运行依赖
- 启动 VideoHighlighter-CN

> 首次安装包含 AI / 视频处理依赖，下载和安装时间会比较长；后续直接从本地环境启动即可。

[查看 Windows 一键运行脚本](./Windows一键下载并运行.ps1)

---

## 这是哪个版本？

这是 **VideoHighlighter 的简体中文维护版**。

- 上游官方项目：Aseiel/VideoHighlighter
- 中文维护仓库：youzhi1701/VideoHighlighter-CN
- 核心功能、算法与主要架构来自上游项目
- 本仓库主要维护简体中文界面、本地化工具、Windows 中文安装体验以及上游更新兼容

如果你需要查看官方原始英文说明：

[查看官方原版自述文件](./README_官方原版.md)

---

## 中文版主要内容

- 主程序界面简体中文化
- 时间线与视频预览界面中文化
- AI 检测、动作识别、物体识别界面中文化
- 模型中心与训练模块中文化
- LLM 对话与视觉搜索中文化
- 转录、字幕、下载、设置等页面中文化
- React / TypeScript 前端中文化
- Windows 安装程序简体中文化
- 弹窗、状态栏、工具提示、错误提示等用户可见文字中文化
- 保留内部枚举、API、配置键和协议字段，避免汉化影响程序逻辑

---

## 中文汉化插件一键下载

> 已兼容 Windows PowerShell 5.1：下载脚本内部使用 ASCII 源码，避免 UTF-8 无 BOM 导致中文脚本解析错误。

如果你只需要 **中文汉化插件**，不想下载整个项目，打开 PowerShell，复制下面这一整行并回车：

```powershell
$u='https://raw.githubusercontent.com/youzhi1701/VideoHighlighter-CN/main/Windows%E4%B8%80%E9%94%AE%E4%B8%8B%E8%BD%BD%E4%B8%AD%E6%96%87%E6%B1%89%E5%8C%96%E6%8F%92%E4%BB%B6.ps1'; $p="$env:TEMP\VideoHighlighter-CN-Plugin.ps1"; Invoke-WebRequest $u -OutFile $p -UseBasicParsing; powershell -ExecutionPolicy Bypass -File $p
```

它会自动：

- 下载当前最新版仓库
- 只提取 `中文汉化插件/`
- 保存到：

```text
下载\VideoHighlighter-中文汉化插件
```

- 如果本地已经有旧版，会先保留为 `VideoHighlighter-中文汉化插件-旧版`
- 下载完成后自动打开插件文件夹

[查看 Windows 一键下载中文汉化插件脚本](./Windows一键下载中文汉化插件.ps1)

---

## 中文汉化插件

仓库根目录提供了独立的：

```text
中文汉化插件/
```

完整结构：

```text
中文汉化插件/
├─ 一键植入中文.py
├─ 扫描遗漏英文.py
├─ 验证汉化完整性.py
├─ 重建汉化规则库.py
├─ 汉化日志.md
├─ 数据/
│  ├─ 汉化规则.json
│  ├─ 翻译记忆库.json
│  └─ 保留英文白名单.json
├─ 资源/
│  └─ ChineseSimplified.isl
└─ README.md
```

### 一键植入中文

在项目根目录运行：

```powershell
python "中文汉化插件/一键植入中文.py" --root . --strict
```

### 扫描遗漏英文

```powershell
python "中文汉化插件/扫描遗漏英文.py" --root .
```

---

## 汉化维护逻辑

以后官方项目更新时，不需要重新人工汉化整个软件。

```text
官方新版本
   ↓
应用已有汉化规则
   ↓
复用翻译记忆
   ↓
扫描新增 / 变化英文
   ↓
只处理 changed / missing
   ↓
重建汉化规则库
   ↓
生成新的中文版
```

当前汉化系统采用：

- 文件级精确匹配
- 上下文判断
- 翻译记忆库
- 冲突检测
- 新增英文扫描
- CI 自动重放验证

不会使用简单的全局字符串替换，以避免误改程序内部值。

---

## 汉化验证

目前汉化层已经验证可以从干净的官方源码重新植入中文。

验证流程包括：

- 从官方基线重新创建源码
- 自动运行中文注入器
- 检查冲突与缺失
- 对比重建后的中文版文件
- 扫描可能遗漏的用户可见英文
- 运行项目现有测试

GitHub Actions 中可以看到：

```text
中文汉化层验证
```

---

## 汉化日志

中文版自己的维护记录：

[查看汉化日志](./中文汉化插件/汉化日志.md)

这里仅记录：

- 汉化新增内容
- 翻译修正
- 本地化规则变化
- 翻译记忆库变化
- Windows 中文安装器变化
- 上游更新兼容情况
- 汉化验证结果

不会把官方原项目开发日志混入中文版汉化日志。

---

## 关于上游

VideoHighlighter 是原作者持续维护的开源项目。

中文版会尽量保持：

```text
官方功能不变
+
程序逻辑不变
+
内部协议不变
+
用户可见界面中文化
```

当官方发布新版本时，本仓库会通过中文汉化插件尽可能自动继承已有翻译，并集中处理新增或变化的界面内容。

---

## 许可证

本仓库继续遵循上游项目原有许可证。

请同时保留和遵守仓库中的：

```text
LICENSE
COPYRIGHT
```

中文本地化不改变原项目的许可证和版权归属。
