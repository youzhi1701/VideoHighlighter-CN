<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter-CN" width="128">
</p>

# VideoHighlighter-CN

> VideoHighlighter 的简体中文维护版，面向 Windows 中文用户提供可直接安装、运行和持续更新的正式桌面版本。

| 项目 | 信息 |
| --- | --- |
| 当前版本 | **v0.13.1-cn.2** |
| 项目类型 | 视频高光分析 / 桌面应用 |
| 主要平台 | Windows |
| 当前状态 | 正式发布 / 持续维护 |

---

## 项目概览

本仓库基于上游 **Aseiel/VideoHighlighter** 持续维护。

中文维护原则是：**完整处理用户可见界面，同时保留内部枚举、API、配置键、模型字段和协议标识，避免为了汉化破坏程序逻辑。**

正式版以封装后的 Windows EXE 为主要用户入口，不要求普通用户下载源码、打开 CMD 或手动创建 Python 环境。

---

## 核心能力

VideoHighlighter-CN 保留上游完整的视频分析能力，并持续维护以下用户可见模块：

- 视频导入与预览
- 时间线
- 高光分析
- 信号时间线
- 字幕
- 下载
- 设置
- 模型中心
- 训练
- LLM
- 视觉搜索
- Windows 安装与更新

中文维护覆盖：
- 主界面与二级页面
- 弹窗、菜单、状态栏和工具提示
- 动态生成文案与错误信息
- Windows 安装程序
- 用户可见英文残留扫描
- 本地化重建验证

---

## 快速开始

普通用户请直接进入 GitHub Releases 下载最新版：

```text
00-VideoHighlighter-CN-v0.13.1-cn.2-Windows-Setup.exe
```

双击 EXE 安装即可。

正式发布版封装运行环境与所需组件，普通用户不需要：
- 下载源码 ZIP
- 打开 CMD
- 手动创建 Python 虚拟环境
- 额外安装“中文汉化补丁”

> 根目录 `一键安装运行.cmd` 仅用于源码开发、调试与维护，不作为正式发行入口。

---

## 从源码运行

仓库包含：

```text
requirements.txt
```

建议在独立虚拟环境中安装依赖后运行主程序。

仓库同时保留上游测试、训练、模型、前端和打包结构，方便后续继续同步。

---

## 中文维护工具

根目录：

```text
中文汉化插件/
```

主要入口：

```text
中文汉化插件/
├─ 一键汉化.py
├─ README.md
└─ 内部/
```

重新套用本地化规则：

```powershell
python "中文汉化插件/一键汉化.py" --root . --strict
```

维护工具结合翻译规则、翻译记忆、英文残留扫描、保留英文白名单和重建验证，减少重复查漏。

---

## 构建与发布

GitHub Actions 已包含：
- 中文汉化验证
- 自动测试
- 安装程序检查
- Build & Release
- 更新包发布
- 本地化目录重建

发布流程以 `version.py` 为版本单一来源，当前：

```text
0.13.1-cn.2
```

正式发布会构建并验证 Windows 安装版；Release 中的 EXE 才是普通用户的正式安装入口。

---

## 项目结构

```text
VideoHighlighter-CN/
├─ assets/
├─ docs/
├─ tests/
├─ tools/
├─ training/
├─ video_ai_editor/
├─ sidecar/
├─ 中文汉化插件/
├─ signal_timeline_viewer.py
├─ video_picker_dialog.py
├─ requirements.txt
├─ version.py
├─ 一键安装运行.cmd
└─ README.md
```

实际目录会随上游版本演进继续变化。

---

## 汉化边界

以下内容不以“全部翻成中文”为目标：
- Python / TypeScript 内部变量
- 配置键
- 模型名
- API 字段
- 枚举值
- 协议字段
- 技术格式名
- 为兼容第三方依赖必须保留的英文

判断标准不是“源码里还能不能搜到英文”，而是 **用户实际使用时不应被无意义的英文界面打断，同时内部结构必须保持稳定。**

---

## 上游与许可证

- 上游项目：`Aseiel/VideoHighlighter`
- 中文维护：`youzhi1701/VideoHighlighter-CN`
- 核心算法与主体架构来自上游
- 中文版重点维护界面、Windows 使用体验、安装流程和本地化验证

本仓库继续遵循上游项目原有许可证和版权声明，请保留：

```text
LICENSE
COPYRIGHT
```

---

## 发布与维护

当前正式版本：**v0.13.1-cn.2**

后续正式版本必须保持：
- 应用内版本与 Release 版本一致
- 安装包可直接运行
- Windows 安装后烟雾测试通过
- 中文用户可见界面完成验证
- 不以 CMD/BAT 作为正式发行入口