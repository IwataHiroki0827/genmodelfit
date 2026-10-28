# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""genmodelfit の出力を端末の見た目の PNG にする(英語版 <名前>.png と日本語版 <名前>.ja.png)。

使い方(リポジトリの一番上で):
    uvx --from . --with pillow python tools/screenshots.py            # README の画像(docs/images)
    uvx --from . --with pillow python tools/screenshots.py --all DIR  # note などに使う全部を DIR へ
    uvx --from . --with pillow python tools/screenshots.py --readme   # README の「例」のブロックも同じ GPU で

出力に色を付けて受け取り、HTML にして Edge か Chrome の headless で撮り、余白を切る。
全角の字は 2 桁の枠に置くので、字体によらず表の列がそろう。
--vram / --ram は量だけを変えるので、GPU の世代(使える精度)は撮る PC のものになってしまう(docs/hardware.md)。
そこで画像では GPU の検出を決まった架空の GPU(Ada 世代: bf16・fp8 とも使える)に差し替え、撮る PC の GPU が
写らないようにする(SIM_GPU)。「この PC そのもの」の画像(--all の *-this-pc)だけ本物を使う。
Pillow は撮るときだけ使う(genmodelfit の依存には入れない)。
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops
from rich.cells import cell_len
from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

ROOT = Path(__file__).resolve().parent.parent
FIX = ["--vram", "8", "--ram", "32"]
README = [("recommend", ["recommend", *FIX, "-n", "2"])]
MORE = [
    ("recommend-image", ["recommend", *FIX, "-m", "image", "-n", "8"]),
    ("recommend-video", ["recommend", *FIX, "-m", "video", "-n", "8"]),
    ("recommend-music", ["recommend", *FIX, "-m", "music", "-n", "8"]),
    ("recommend-sfx", ["recommend", *FIX, "-m", "sfx", "-n", "8"]),
    ("recommend-tts", ["recommend", *FIX, "-m", "tts", "-n", "8"]),
    ("recommend-3d", ["recommend", *FIX, "-m", "3d", "-n", "8"]),
    ("recommend-image-edit", ["recommend", *FIX, "-m", "image", "--task", "edit", "-n", "8"]),
    ("recommend-vram6-ram16", ["recommend", "--vram", "6", "--ram", "16", "-n", "2"]),
    ("recommend-vram12-ram32", ["recommend", "--vram", "12", "--ram", "32", "-n", "2"]),
    ("recommend-vram16-ram64", ["recommend", "--vram", "16", "--ram", "64", "-n", "2"]),
    ("recommend-vram24-ram64", ["recommend", "--vram", "24", "--ram", "64", "-n", "2"]),
    ("fit-video", ["fit", *FIX, "-m", "video", "-n", "12"]),
    ("search-wan", ["search", "wan", *FIX]),
    ("info-sdxl", ["info", "image_sdxl_simple", *FIX]),
    ("types", ["types"]),
    # 置き場所を決めないとユーザのフォルダのパスが出るので、決まった場所を渡す
    ("download-list", ["download", "SDXL1.0 Base", "--list", *FIX, "--output-dir", r"C:\ComfyUI\models"]),
    ("help", ["--help"]),
    # この PC そのもの(PC ごとに中身が変わる)
    ("system-this-pc", ["system"]),
    ("recommend-this-pc", ["recommend", "-n", "2"]),
]
COLS = 100
BG, FG = "#1e1e1e", "#d4d4d4"
THEME = TerminalTheme(
    (0x1E, 0x1E, 0x1E),
    (0xD4, 0xD4, 0xD4),
    [
        (0x00, 0x00, 0x00), (0xF1, 0x4C, 0x4C), (0x23, 0xD1, 0x8B), (0xF5, 0xF5, 0x43),
        (0x3B, 0x8E, 0xEA), (0xD6, 0x70, 0xD6), (0x29, 0xB8, 0xDB), (0xE5, 0xE5, 0xE5),
    ],
    [
        (0x66, 0x66, 0x66), (0xF1, 0x4C, 0x4C), (0x23, 0xD1, 0x8B), (0xF5, 0xF5, 0x43),
        (0x3B, 0x8E, 0xEA), (0xD6, 0x70, 0xD6), (0x29, 0xB8, 0xDB), (0xFF, 0xFF, 0xFF),
    ],
)  # fmt: skip
CSS = f"""body{{margin:0;background:{BG}}}
pre{{margin:0;padding:24px 28px;color:{FG};background:{BG};font-size:16px;line-height:1.45;
font-family:"Cascadia Mono",Consolas,"BIZ UDGothic","MS Gothic",monospace;white-space:pre}}
.c1{{display:inline-block;width:1ch;text-align:center;overflow:hidden;vertical-align:bottom}}
.c2{{display:inline-block;width:2ch;text-align:center;vertical-align:bottom;
font-family:"BIZ UDGothic","MS Gothic","Yu Gothic",monospace}}
.cmd{{color:#9cdcfe}}"""
# Windows でパイプに出すと rich が古い端末とみなして色を外すので、その判定だけ切って CLI を動かす
# 撮る PC の GPU を写さないための架空の GPU(世代だけ決めた汎用の名前。ComfyUI が Ada で選ぶ bf16 の計算)
SIM_GPU = (
    "import genmodelfit.hardware as h; from genmodelfit.models import Gpu; "
    "h.detect_gpus = lambda *a, **k: ([Gpu(name='Simulated GPU (Ada)', vendor='nvidia', vram_total_gb=8.0, "
    "backend='cuda', compute_capability='8.9', compute_capability_source='name', arch='ada', "
    "compute_dtype='bf16', bf16=True, fp8=True)], []); "
)
BOOT = (
    "import sys, runpy, rich.console as c; c.detect_legacy_windows = lambda: False; "
    "{sim}sys.argv = ['genmodelfit', *sys.argv[1:]]; runpy.run_module('genmodelfit', run_name='__main__')"
)


def browser() -> str:
    for p in (
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ):
        if os.path.exists(p):
            return p
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("Edge か Chrome が見つかりません")


def run(args: list[str], lang: str, simulate: bool = True) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "GENMODELFIT_COMFYUI_DIR")}
    env.update(FORCE_COLOR="1", COLUMNS=str(COLS), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env["GENMODELFIT_LANG"] = lang
    p = subprocess.run(
        [sys.executable, "-c", BOOT.replace("{sim}", SIM_GPU if simulate else ""), *args],
        capture_output=True,
        env=env,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
    )
    if p.returncode != 0:
        raise RuntimeError(f"exit {p.returncode}: {p.stderr.strip()[-500:]}")
    return p.stdout.rstrip("\n")


def to_html(cmd: str, ansi: str) -> str:
    with open(os.devnull, "w", encoding="utf-8") as null:
        con = Console(width=400, force_terminal=True, color_system="truecolor", file=null)
        lines = [f'<span class="cmd">&gt; {html.escape(cmd)}</span>', ""]
        for line in Text.from_ansi(ansi).split("\n", allow_blank=True):
            out = []
            for seg in line.render(con):
                css = seg.style.get_html_style(THEME) if seg.style else ""
                body = "".join(
                    html.escape(ch)
                    if ord(ch) < 128
                    else f'<span class="c{2 if cell_len(ch) == 2 else 1}">{html.escape(ch)}</span>'
                    for ch in seg.text
                )
                out.append(f'<span style="{css}">{body}</span>' if css else body)
            lines.append("".join(out))
    return f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><body><pre>{chr(10).join(lines)}</pre>'


def shoot(exe: str, page: Path, png: Path, n_lines: int, tmp: Path) -> None:
    height = int((n_lines + 2) * 16 * 1.45 * 1.3 + 200)  # 折り返しを見込んで多めに取り、あとで切る
    subprocess.run(
        [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--user-data-dir={tmp / 'profile'}",
            f"--window-size=1400,{height}",
            f"--screenshot={png}",
            page.as_uri(),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if not png.exists():
        raise RuntimeError("ブラウザが画像を作りませんでした")
    with Image.open(png) as raw:
        im = raw.convert("RGB")
    box = ImageChops.difference(im, Image.new("RGB", im.size, (0x1E, 0x1E, 0x1E))).getbbox()
    if box is None:
        raise RuntimeError("画像が空です")
    left, top, right, bottom = box
    if bottom >= im.height - 8 or right >= im.width - 8:
        raise RuntimeError("内容が窓からはみ出しました(切れています)")
    pad = 48
    crop = (max(left - pad, 0), max(top - pad, 0), min(right + pad, im.width), min(bottom + pad, im.height))
    im.crop(crop).save(png, optimize=True)


def update_readme_blocks() -> int:
    """README の「例」の `$ genmodelfit …` のブロックを、架空の GPU(SIM_GPU)の出力で書き直す。
    コマンドの行(`$ …`)は今のまま使う(--lang ja は言語の指定として読む)。元のブロックが `…` で終わるときは、
    元の行数だけ残して `…` を付ける。"""
    fence = "`" * 3
    pat = re.compile(fence + r"\n(\$ genmodelfit [^\n]*)\n(.*?)" + fence, re.S)
    for name, default_lang in (("README.md", "en"), ("README.ja.md", "ja")):
        path = ROOT / name
        raw = path.read_text(encoding="utf-8")
        crlf = "\r\n" in raw
        text = raw.replace("\r\n", "\n")

        def repl(m: re.Match[str], default_lang: str = default_lang) -> str:
            cmd, old = m.group(1), m.group(2)
            lang = default_lang
            argv = shlex.split(cmd[len("$ genmodelfit ") :])
            if "--lang" in argv:
                i = argv.index("--lang")
                lang = argv[i + 1]
                del argv[i : i + 2]
            if "--vram" not in argv:
                return m.group(0)
            out = Text.from_ansi(run(argv, lang)).plain
            lines = [ln.rstrip() for ln in out.split("\n")]
            old_lines = old.rstrip("\n").split("\n")
            if old_lines[-1] == "…":
                lines = [*lines[: len(old_lines) - 1], "…"]
            return f"{fence}\n{cmd}\n" + "\n".join(lines) + f"\n{fence}"

        new = pat.sub(repl, text)
        if new != text:
            path.write_text(new.replace("\n", "\r\n") if crlf else new, encoding="utf-8", newline="")
        print("OK  ", name, "(書き直した)" if new != text else "(変わらない)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="genmodelfit の出力を PNG にする(英語版と日本語版)")
    ap.add_argument("--all", metavar="DIR", help="README の画像に加えて、note などに使う全部を DIR に撮る")
    ap.add_argument("--readme", action="store_true", help="README の「例」のコードブロックだけを作り直す")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.readme:
        return update_readme_blocks()
    shots = [*README, *MORE] if a.all else README
    out = Path(a.all).resolve() if a.all else ROOT / "docs" / "images"
    out.mkdir(parents=True, exist_ok=True)
    exe = browser()
    # 写ってはいけない語(ユーザ名・PC 名)。見つけたらその画像は作らない
    secrets = {
        v for k in ("USERNAME", "USER", "COMPUTERNAME", "HOSTNAME") if len(v := os.environ.get(k, "")) >= 3
    }
    failed = []
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, args in shots:
            for lang, suffix in (("en", ""), ("ja", ".ja")):
                png = out / f"{name}{suffix}.png"
                try:
                    text = run(args, lang, simulate=not name.endswith("-this-pc"))
                    plain = Text.from_ansi(text).plain.lower()
                    hit = sorted(w for w in secrets if w.lower() in plain)
                    if hit:
                        raise RuntimeError(f"ユーザ名か PC 名が写っています: {hit}")
                    shown = " ".join(f'"{x}"' if " " in x else x for x in args)
                    page = tmp / "page.html"
                    page.write_text(to_html(f"genmodelfit {shown}", text), encoding="utf-8")
                    png.unlink(missing_ok=True)
                    shoot(exe, page, png, text.count("\n") + 1, tmp)
                    print("OK  ", png)
                except Exception as e:  # 1 枚失敗しても残りを撮る
                    png.unlink(missing_ok=True)
                    failed.append(f"{png.name}: {e}")
                    print("FAIL", png.name, e)
    print(f"{len(shots) * 2 - len(failed)} 枚 OK、{len(failed)} 枚 失敗")
    for f in failed:
        print("  ", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
