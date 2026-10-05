# VideoHighlighter 中文汉化工具

以后你主要看这个目录，不用记英文文件名。

## 直接使用

### 1. 一键植入中文
```powershell
python "中文汉化工具/一键植入中文.py" --root . --strict
```

### 2. 扫描遗漏英文
```powershell
python "中文汉化工具/扫描遗漏英文.py" --root .
```

### 3. 验证汉化完整性
这个脚本主要给自动验证流程使用。

### 4. 重建汉化规则库
官方更新并补完新增翻译后运行：

```powershell
python "中文汉化工具/重建汉化规则库.py" --upstream-ref upstream/main
```

## 对应关系

- 一键植入中文.py → tools/apply_zh_cn.py
- 扫描遗漏英文.py → tools/scan_untranslated.py
- 验证汉化完整性.py → tools/verify_localization_replay.py
- 重建汉化规则库.py → tools/rebuild_localization_catalog.py

英文底层脚本保留，是为了保证 GitHub Actions、自动化和后续上游同步稳定；你平时直接使用这里的中文入口即可。
