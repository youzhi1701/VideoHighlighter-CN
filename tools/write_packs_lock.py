"""Write packs.lock.json into a build: the packs this app build may download.

    python tools/write_packs_lock.py --base "$PACKS_BASE" \
        --out dist/VideoHighlighter/packs.lock.json

Reads ``<base>/packs.json`` (or ``--packs-json`` for a local copy), checks every
row has what the app needs to verify a download, and writes it back with the
base URL added. modules/packs/pack_manager.py reads it at runtime; nothing about
packs is looked up online after the build, so a build always installs exactly
the packs it was tested against.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request

REQUIRED = ("name", "asset", "bytes", "sha256")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True,
                    help="资源包标签的 Release 下载 URL，例如 "
                         "https://github.com/<owner>/<repo>/releases/download/packs-torch-2.7.1")
    ap.add_argument("--packs-json", help="本地 packs.json（默认：<base>/packs.json）")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    base = args.base.rstrip("/")
    if args.packs_json:
        with open(args.packs_json, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    else:
        with urllib.request.urlopen(base + "/packs.json", timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8-sig"))

    packs = data.get("packs") or []
    if not packs:
        print("packs.json 中没有资源包", file=sys.stderr)
        return 1
    for row in packs:
        missing = [k for k in REQUIRED if not row.get(k)]
        if missing:
            print(f"{row.get('name', '?')}：缺少 {', '.join(missing)}", file=sys.stderr)
            return 1
        if len(str(row["sha256"])) != 64:
            print(f"{row['name']}：sha256 不是有效的 SHA-256", file=sys.stderr)
            return 1

    lock = {"base": base, "tag": data.get("tag") or base.rsplit("/", 1)[-1], "packs": packs}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=2)
        fh.write("\n")
    print(f"{args.out}：{', '.join(p['name'] for p in packs)}，来源标签 {lock['tag']}")
    return 0


if __name__ == "__main__":
    # Keep this release helper self-contained. When Python executes a script by
    # path (python tools/write_packs_lock.py), sys.path[0] is tools/, so importing
    # the app's top-level modules package is not reliable on clean CI runners.
    # Configure UTF-8 directly instead of depending on application code.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
