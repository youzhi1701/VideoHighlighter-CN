# VideoHighlighter zh-CN Localization Layer

这个目录是中文版的**可重放本地化层**，目的是把汉化从“直接散落修改源码”升级成可持续维护的补丁系统。

## 设计原则

1. **不做全局字符串替换**：每条规则都绑定具体文件和具体源码块。
2. **内部协议保持英文**：枚举、API 值、配置键、模型标签、文件格式字段不因为界面汉化而改变。
3. **上游变化不静默猜测**：原文被上游修改后，规则进入 `changed` / `missing`，不会强行套旧翻译。
4. **翻译记忆集中保存**：`translation_memory.json` 保存已知英文→中文映射及出现文件。
5. **安装器资源独立保存**：Windows 简体中文 Inno Setup 语言文件位于 `assets/`。
6. **扫描与注入分离**：注入器负责重放已确认翻译；扫描器只负责发现新增英文。

## 目录

```text
localization/zh_CN/
├─ patches.json               # 精确文件级补丁规则
├─ translation_memory.json    # 翻译记忆库
├─ allowlist.json             # 合理保留英文/专有名词
└─ assets/
   └─ ChineseSimplified.isl   # Windows 安装器中文语言资源
```

工具：

```text
tools/apply_zh_cn.py          # 安全注入中文
tools/scan_untranslated.py    # 扫描新增/遗漏英文
```

## 对新的官方源码执行汉化

建议先从官方 `Aseiel/VideoHighlighter` 的新版本建立干净分支，再运行：

```powershell
python tools/apply_zh_cn.py --root . --strict
python tools/scan_untranslated.py --root .
```

注入报告输出到：

```text
localization/reports/apply_report.md
localization/reports/apply_report.json
localization/reports/untranslated.json
```

### 状态含义

- `applied`：找到已知官方源码，成功植入中文。
- `already_applied`：已经是中文版，无需重复修改。
- `changed`：官方改写了附近源码，找到相似位置但没有自动替换。
- `conflict`：同一源码块出现多次，无法安全判断应该修改哪处。
- `missing`：旧源码块已经不存在，需要人工确认官方新实现。
- `missing_file`：官方移动或删除了文件。

`--strict` 下，只要出现 changed/conflict/missing/missing_file 就返回非 0，适合 CI 阻止错误汉化进入正式中文版。

## 为什么不直接只用 Qt Linguist

Qt 原生 `lupdate/lrelease/QTranslator` 对纯 Qt 项目非常成熟，但 VideoHighlighter 同时存在：

- Python / PySide6
- React / TypeScript / TSX
- Inno Setup
- 运行时动态字符串

因此本项目采用统一的“本地化层 + 各技术栈扫描器”结构。后续如果上游把 PySide6 字符串逐步改成 `tr()`，可以继续接入 Qt Linguist，而不需要推翻当前翻译记忆库。
