<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter-CN" width="150">
</p>

# VideoHighlighter-CN

VideoHighlighter 的简体中文维护版。

本仓库基于上游 **Aseiel/VideoHighlighter** 持续维护，目标是让 Windows 中文用户可以直接安装、运行和使用，而不需要另外给英文版打补丁。

中文维护原则：**完整处理用户可见界面，同时保留内部枚举、API、配置键、模型字段和协议标识，避免汉化破坏程序逻辑。**

<p align="center">
  <a href="https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/下载-当前中文版源码_ZIP-brightgreen?style=for-the-badge" alt="下载中文版">
  </a>
</p>

---

## 项目能力

VideoHighlighter 本身是一套视频高光分析与辅助处理工具。当前中文版保留上游完整项目结构，并持续维护以下用户可见模块：

- 视频导入与预览
- 时间线
- 高光分析
- 信号时间线
- 字幕相关界面
- 下载相关界面
- 设置
- 模型中心
- 训练
- LLM
- 视觉搜索
- React / TypeScript 前端
- Windows 安装与运行流程

中文维护还覆盖：
- 主程序界面
- 二级页面
- 弹窗
- 菜单
- 状态栏
- 工具提示
- 错误提示
- 动态生成文案
- 安装程序

---

## Windows 最简单的安装方式

普通用户只需要三步：

1. 下载当前仓库 ZIP
2. 解压到本地
3. 双击根目录的 **`一键安装运行.cmd`**

第一次运行会自动：

- 检查 Python 3.12
- 缺少 Python 时尝试通过 Windows `winget` 安装
- 创建项目独立 `.venv`
- 安装 `requirements.txt` 依赖
- 安装 YOLOX 运行包
- 启动 VideoHighlighter-CN

以后再次双击同一个脚本，会直接复用已有环境。

> 普通用户不需要运行“中文汉化插件”，也不需要再给英文便携版打汉化补丁。当前仓库源码本身就是中文版。

如果环境安装异常，可删除项目目录中的 `.venv` 后重新运行安装脚本。

---

## 从源码运行

项目包含：

```text
requirements.txt
```

建议在独立虚拟环境中安装依赖后运行主程序。

仓库同时保留上游的测试、训练、模型、前端和打包结构，方便后续继续与官方版本同步。

---

## 中文汉化维护工具

根目录：

```text
中文汉化插件/
```

该目录主要面向维护者，不是普通用户运行软件所必需。

主要入口：

```text
中文汉化插件/
├─ 一键汉化.py
├─ README.md
└─ 内部/
```

当上游更新后，需要重新套用本地化规则时：

```powershell
python "中文汉化插件/一键汉化.py" --root . --strict
```

维护工具会结合：
- 翻译规则
- 翻译记忆
- 残留英文扫描
- 保留英文白名单
- 重建验证

尽量减少人工重复查漏。

---

## 自动验证

仓库 GitHub Actions 中包含多类构建与验证流程，例如：

- 中文汉化验证
- 测试
- 安装程序检查
- 构建 Release
- 发布更新包
- 本地化目录重建

汉化验证重点检查：

- 能否从官方基线重新应用中文化
- 是否出现翻译冲突或规则失效
- 重建后的中文版是否与当前源码一致
- 是否仍存在可能遗漏的用户可见英文
- 是否误翻译内部技术字段

---

## 项目结构

仓库保留上游完整的大型项目结构，核心目录与文件包括：

```text
VideoHighlighter-CN/
├─ assets/                 # 图标与静态资源
├─ docs/                   # 项目文档
├─ tests/                  # 测试
├─ tools/                  # 工具
├─ training/               # 训练相关
├─ video_ai_editor/        # 视频 AI 编辑相关
├─ sidecar/                # Sidecar 相关
├─ 中文汉化插件/           # 中文维护工具
├─ signal_timeline_viewer.py
├─ video_picker_dialog.py
├─ requirements.txt
├─ version.py
├─ 一键安装运行.cmd
└─ README.md
```

实际目录会随上游版本演进继续变化。

---

## 与官方项目的关系

- 官方上游：`Aseiel/VideoHighlighter`
- 中文维护：`youzhi1701/VideoHighlighter-CN`
- 核心算法与主体架构来自上游
- 中文版重点维护中文界面、Windows 使用体验、安装流程和本地化验证
- 上游更新后继续通过自动化本地化工具跟进

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

判断标准不是“源码里还能不能搜到英文”，而是 **用户实际使用时不应该被无意义的英文界面打断，同时程序内部结构必须保持稳定。**

---

## 原项目文档

上游 README、docs、测试、打包和模型工具继续保留，以便后续同步与维护。

---

## 许可证

本仓库继续遵循上游项目原有许可证和版权声明。

请保留仓库中的：

```text
LICENSE
COPYRIGHT
```
