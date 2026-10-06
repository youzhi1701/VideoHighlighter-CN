"""Build 00-VideoHighlighter-Windows-Setup.zip for GitHub Release attachment.

The ``00-`` prefix is deliberate: GitHub sorts release Assets alphabetically,
so without it the tiny installer sank below the multi-GB ``.7z`` parts and the
``.dmg``, and people downloaded the wrong file.

The zip is tiny (~10 KB): double-click Install-VideoHighlighter.bat on Windows,
and the script downloads both split 7z volumes plus extracts them.

The tag defaults to version.py, so bumping the app bumps what the installer
asks for. It used to be typed in by hand in two places -- the CI argument and
the committed config.json -- and the committed copy simply went stale: it still
named 0.9.0 several releases later, which is the version the installer falls
back to whenever the GitHub API cannot be reached.

Usage::

    python tools/build_bootstrap_zip.py --edition free
    python tools/build_bootstrap_zip.py --edition pro --tag 0.9.0-Pro
    python tools/build_bootstrap_zip.py --edition free --write-config
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = ROOT / "packaging" / "bootstrap"
ZIP_MEMBERS = (
    "Install-VideoHighlighter.bat",
    "Install-VideoHighlighter.ps1",
    "config.json",
)

FREE_REPO = "youzhi1701/VideoHighlighter-CN"
PRO_REPO = "Aseiel/VideoHighlighter-pro"

# The committed configs, one per edition. These are what someone gets when they
# run the installer straight from a checkout, and what the offline fallback in a
# shipped zip is copied from -- so they have to track version.py rather than be
# remembered.
CONFIG_PATHS = {
    "free": BOOTSTRAP / "config.json",
    "pro": BOOTSTRAP / "config.pro.example.json",
}


def default_tag(edition: str) -> str:
    """The tag this checkout would release under, per version.py.

    Mirrors the slug the release workflow computes: the Free tag is the bare
    version, Pro appends its edition. Deriving it here means the installer and
    the app can no longer disagree about which release is current.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import version

    return (f"{version.__version__}-Pro" if edition.lower() == "pro"
            else version.__version__)


def _windows_assets(tag: str, *, pro: bool) -> tuple[str, ...]:
    if pro:
        return (
            f"VideoHighlighter-Windows-{tag}.7z.001",
            f"VideoHighlighter-Windows-{tag}.7z.002",
        )
    return (
        f"VideoHighlighter-CN-v{tag}-Windows.7z.001",
        f"VideoHighlighter-CN-v{tag}-Windows.7z.002",
    )


def make_config(*, edition: str, tag: str) -> dict:
    pro = edition.lower() == "pro"
    repo = PRO_REPO if pro else FREE_REPO
    assets = list(_windows_assets(tag, pro=pro))
    return {
        "product_name": "VideoHighlighter-CN 中文版",
        "edition": "Pro" if pro else "Free",
        "repo": repo,
        "use_latest": not pro,
        "asset_pattern": r"^VideoHighlighter-CN-v.*-Windows\.7z\.\d{3}$",
        "tag": tag,
        "assets": assets,
        "base_url": f"https://github.com/{repo}/releases/download/{tag}",
        "notes": (
            "Pro: private repo — anonymous download URLs need auth; "
            "customers install from Lemon Squeezy my-orders (single .7z). "
            "This zip is for maintainers with local volumes or GitHub access."
            if pro
            else "use_latest asks the GitHub API for the current release."
        ),
    }


def write_config(*, edition: str, tag: str) -> Path:
    """Rewrite the committed config for `edition` at the given tag.

    Only the version-bearing fields are regenerated. The hand-written `notes`
    is kept as it is -- the Pro one carries the command for running the
    installer against that config, which no generator knows about.
    """
    path = CONFIG_PATHS[edition.lower()]
    if not path.exists():
        raise SystemExit(f"{edition} 没有已提交的配置：{path}")

    config = make_config(edition=edition, tag=tag)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if existing.get("notes"):
        config["notes"] = existing["notes"]

    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"完成 {path}（标签={tag}）")
    return path


def build_zip(*, edition: str, tag: str, out: Path) -> Path:
    if not BOOTSTRAP.is_dir():
        raise SystemExit(f"缺少引导安装目录：{BOOTSTRAP}")

    missing = [name for name in ZIP_MEMBERS[:2] if not (BOOTSTRAP / name).exists()]
    if missing:
        raise SystemExit(f"缺少引导安装文件：{', '.join(missing)}")

    config = make_config(edition=edition, tag=tag)
    staging = out.parent / f".bootstrap-staging-{out.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for name in ZIP_MEMBERS[:2]:
            (staging / name).write_bytes((BOOTSTRAP / name).read_bytes())
        (staging / "config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging.iterdir()):
                zf.write(path, arcname=path.name)
    finally:
        for child in staging.iterdir():
            child.unlink(missing_ok=True)
        staging.rmdir()

    size_kb = out.stat().st_size / 1024
    print(f"完成 {out}（{size_kb:.1f} KB）")
    print(f"   版本类型={edition} 标签={tag} 仓库={config['repo']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--edition",
        choices=("free", "pro"),
        required=True,
        help="Free（公开 GitHub）或 Pro（固定标签；面向客户）。",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release 标签，例如 0.9.0 或 0.9.0-Pro；默认取自 version.py。",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="重写该版本类型已提交的引导配置，而不是构建 zip；更新 version.py 后运行。",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "packaging" / "bootstrap" / "00-VideoHighlighter-Windows-Setup.zip",
        help="输出 zip 路径（00- 前缀可使其排在 Release 资源列表首位）。",
    )
    args = parser.parse_args(argv)
    tag = args.tag or default_tag(args.edition)
    if args.write_config:
        write_config(edition=args.edition, tag=tag)
    else:
        build_zip(edition=args.edition, tag=tag, out=args.out.resolve())
    return 0


if __name__ == "__main__":
    # ROOT only reaches sys.path inside default_tag(), which has not run yet.
    sys.path.insert(0, str(ROOT))
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    raise SystemExit(main())
