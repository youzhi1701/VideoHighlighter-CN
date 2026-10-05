<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter-CN" width="150">
</p>

# VideoHighlighter-CN

VideoHighlighter 简体中文维护版。

本仓库基于上游 **Aseiel/VideoHighlighter** 持续维护，主要提供中文界面和中文本地化维护能力。核心功能、算法和项目结构仍以官方项目为基础。

<p align="center">
  <a href="https://github.com/youzhi1701/VideoHighlighter-CN/archive/refs/heads/main.zip">
    <img src="https://img.shields.io/badge/下载-当前中文版源码_ZIP-brightgreen?style=for-the-badge" alt="下载中文版">
  </a>
</p>

## 最简单的使用方式

如果你只是要使用中文版：

1. 点击上面的 **“下载当前中文版源码 ZIP”**
2. 解压项目
3. 按项目原有方式安装依赖并运行

不需要单独下载汉化脚本，也不需要对英文便携版打补丁。

---

## 中文版包含

- 主程序界面中文化
- 时间线、视频预览、下载、字幕、设置等界面中文化
- 模型中心、训练、LLM、视觉搜索等模块中文化
- React / TypeScript 前端中文化
- Windows 安装程序中文化
- 弹窗、状态栏、工具提示、错误提示等用户可见文字中文化
- 保留内部枚举、API、配置键和协议字段，避免汉化破坏程序逻辑

---

## 中文汉化插件

根目录只保留一套：

```text
中文汉化插件/
```

它是给**维护者**用的，不是普通用户运行软件所必需。

你平时只需要认识：

```text
中文汉化插件/
├─ 一键汉化.py
├─ README.md
└─ 内部/
```

如果以后需要把新的官方源码重新植入现有中文：

```powershell
python "中文汉化插件/一键汉化.py" --root . --strict
```

其余扫描器、规则库、翻译记忆和验证工具都放在 `内部/`，平时不用管。

---

## 与官方项目的关系

- 官方上游：Aseiel/VideoHighlighter
- 中文维护：youzhi1701/VideoHighlighter-CN
- 官方发布新版本后，中文版通过汉化规则和翻译记忆继续跟进
- 如果上游修改了原有界面文字，汉化工具会报告变化，而不是强行替换

---

## 自动验证

仓库保留一个后台 CI：

```text
中文汉化验证
```

它会检查：

- 是否可以从官方基线重新应用汉化
- 是否出现冲突或缺失
- 重建后的中文版是否与当前源码一致
- 是否存在可能遗漏的用户可见英文

普通用户无需操作这个流程。

---

## 原项目文档

上游原有的其他语言 README、docs、测试、打包和模型工具继续保留，避免破坏项目结构和后续同步。

---

## 许可证

本仓库继续遵循上游项目原有许可证和版权声明。

请保留：

```text
LICENSE
COPYRIGHT
```
