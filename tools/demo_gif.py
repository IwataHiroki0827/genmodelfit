# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ruff: noqa: E501  (HTML と CSS の文字列は 1 行が長い)

"""README の冒頭の GIF(search → recommend)を作る(英語版 demo.gif と日本語版 demo.ja.gif)。

使い方(リポジトリの一番上で。ffmpeg が PATH に要る。--ffmpeg でも渡せる):
    uvx --from . --with pillow python tools/demo_gif.py
    uvx --from . --with pillow python tools/demo_gif.py --out DIR --keep-frames

本物の CLI の出力(GPU は tools/screenshots.py と同じ架空の GPU(SIM_GPU)、メモリも決めた量)を、
コマの状態(打ち込み・字幕・強調・拡大・送り)ごとに HTML にし、Edge か Chrome の headless で撮る。
下に進み具合のバーを足し、ffmpeg の palettegen / paletteuse で GIF にする(3 MB 以下を確かめる)。
字幕の数字(Mem%・VRAM)とモデルの名前は、撮った出力の表から読む(手で書かない)。
Pillow・ffmpeg は作るときだけ使う(genmodelfit の依存には入れない)。
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from pathlib import Path

from PIL import Image, ImageDraw
from rich.cells import cell_len
from rich.console import Console
from rich.text import Text

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screenshots import BOOT, SIM_GPU, THEME, browser

ROOT = Path(__file__).resolve().parent.parent
W, H = 880, 496  # GIF の大きさ(px)
FPS = 10
COLS = 100
FONT_PX, LINE_PX, PAD_X, PAD_Y = 14, 19, 18, 12
BAR_PX = 6  # 進み具合のバー
MAX_ZOOM = 1.55
MAX_BYTES = 3 * 1024 * 1024
# 撮る PC のメモリを写さないための決まった量(32 GiB、空き 20 GiB)
SIM_MEM = (
    "import genmodelfit.hardware as hw; "
    "hw.read_memory = lambda: hw.Memory(total=32 * 1024**3, available=20 * 1024**3); "
)

TEXT = {
    "en": {
        "search": "uvx genmodelfit search flux",
        "recommend": "uvx genmodelfit recommend",
        "judge": "Judges every model for this PC: {levels}",
        "auto": "No specs to type: <em>it reads this PC</em>",
        "labels": ("GPU", "VRAM", "RAM"),
        "marginal": "{model} uses <em>{mem}</em> of the VRAM: it just fits",
        "too_tight": "{model} needs <em>{vram} GiB</em>: too big for this PC",
        "rec": "Or <em>the best models for this PC</em>, per type",
        "types": "<em>Image · Video · Music</em> · <em>Sound effects · Speech · 3D</em>",
        "joiner": " · ",
    },
    "ja": {
        "search": "uvx genmodelfit search flux --lang ja",
        "recommend": "uvx genmodelfit recommend --lang ja",
        "judge": "この PC で {levels} のどれかを判定",
        "auto": "スペックを入力しなくても、<em>この PC を自動で読み取る</em>",
        "labels": ("GPU", "VRAM", "メモリ"),
        "marginal": "{model} は VRAM の <em>{mem}</em>: ぎりぎり動く",
        "too_tight": "{model} は <em>{vram} GiB</em> 必要: この PC には大きすぎる",
        "rec": "この PC での<em>おすすめ</em>も一度に出せる",
        "types": "<em>画像・動画・音楽</em>・<em>効果音・読み上げ・3D</em>",
        "joiner": "・",
    },
}

CSS = f"""*{{box-sizing:border-box}}
html,body{{margin:0;width:{W}px;height:{H}px;overflow:hidden;background:#1e1e1e}}
#view{{position:absolute;left:0;top:0;width:{W}px;height:{H - BAR_PX}px;overflow:hidden}}
#term{{position:absolute;left:0;top:0;transform-origin:0 0}}
pre{{position:relative;margin:0;padding:{PAD_Y}px {PAD_X}px;color:#d4d4d4;font-size:{FONT_PX}px;line-height:{LINE_PX}px;
font-family:"Cascadia Mono",Consolas,"BIZ UDGothic","MS Gothic",monospace;white-space:pre}}
.ln{{height:{LINE_PX}px;transition:none}}
.dim{{opacity:.28}}
.c1{{display:inline-block;width:1ch;text-align:center;overflow:hidden;vertical-align:bottom}}
.c2{{display:inline-block;width:2ch;text-align:center;vertical-align:bottom}}
.c2 i{{font-style:normal;font-family:"BIZ UDGothic","MS Gothic","Yu Gothic",monospace}}
.cmd{{color:#9cdcfe}}
.cur{{display:inline-block;width:1ch;background:#d4d4d4;color:#1e1e1e}}
.box{{position:absolute;border:2px solid #f5c542;border-radius:4px;box-shadow:0 0 10px rgba(245,197,66,.55)}}
.tag{{position:absolute;padding:1px 8px;border-radius:9px;background:#f5c542;color:#1e1e1e;font:700 11px "Segoe UI","Yu Gothic UI",sans-serif}}
.cap{{position:absolute;left:50%;bottom:{BAR_PX + 22}px;transform:translateX(-50%);white-space:nowrap;
padding:8px 22px;border-radius:24px;background:#f6f8fa;color:#1f2328;font:700 19px "Segoe UI","Yu Gothic UI","Meiryo",sans-serif;
box-shadow:0 2px 14px rgba(0,0,0,.45)}}
.cap em{{font-style:normal;color:#1f6feb}}
"""


@dataclass
class State:
    lines: list[str]  # 1 行ずつの HTML
    scroll: float = 0.0  # 上に送った行の数
    zoom: float = 0.0  # 拡大の進み(0 = 等倍、1 = fit_cols の桁が画面の右に収まる大きさ。上限 MAX_ZOOM)
    fit_cols: int = 0
    dim_except: tuple[int, ...] | None = None  # これ以外の行を暗くする(None は暗くしない)
    boxes: list[tuple[int, int, int, str]] = field(default_factory=list)  # (行, 桁, 桁の数, 札)
    caption: str = ""


def run_cli(args: list[str], lang: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("NO_COLOR", "GENMODELFIT_COMFYUI_DIR")}
    env.update(
        FORCE_COLOR="1", COLUMNS=str(COLS), PYTHONIOENCODING="utf-8", PYTHONUTF8="1", GENMODELFIT_LANG=lang
    )
    p = subprocess.run(
        [sys.executable, "-c", BOOT.replace("{sim}", SIM_GPU + SIM_MEM), *args],
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


def ansi_lines(ansi: str) -> tuple[list[str], list[str]]:
    """(1 行ずつの HTML, 1 行ずつの文字だけ)。全角は 2 桁の枠に置く。枠の幅(ch)は端末の字体で測るよう、
    日本語の字体は枠の中の <i> にだけ付ける(枠に付けると ch が日本語の字体の幅になり、囲みの位置がずれる)。"""
    out, plain = [], []
    with open(os.devnull, "w", encoding="utf-8") as null:
        con = Console(width=400, force_terminal=True, color_system="truecolor", file=null)
        for line in Text.from_ansi(ansi).split("\n", allow_blank=True):
            parts = []
            for seg in line.render(con):
                css = seg.style.get_html_style(THEME) if seg.style else ""
                body = "".join(
                    html.escape(ch)
                    if ord(ch) < 128
                    else f'<span class="c2"><i>{html.escape(ch)}</i></span>'
                    if cell_len(ch) == 2
                    else f'<span class="c1">{html.escape(ch)}</span>'
                    for ch in seg.text
                )
                parts.append(f'<span style="{css}">{body}</span>' if css else body)
            out.append("".join(parts))
            plain.append(line.plain)
    return out, plain


def prompt(cmd: str, cursor: bool) -> str:
    return f'<span class="cmd">&gt; {html.escape(cmd)}</span>' + (
        '<span class="cur"> </span>' if cursor else ""
    )


def col_of(text: str, sub: str) -> tuple[int, int]:
    """sub の桁の位置と幅(全角は 2 桁)。"""
    i = text.index(sub)
    return cell_len(text[:i]), cell_len(sub)


@dataclass
class Row:
    index: int
    level: str
    model: str
    vram: str
    mem: str


def table_rows(plain: list[str]) -> list[Row]:
    """search の表の行(段階・モデル・VRAM・Mem%)を、見出しの行の桁で切り出す。"""
    head = next(i for i, ln in enumerate(plain) if ln.startswith("Type ") and "Model" in ln)
    h = plain[head]
    i_model, i_use, i_vram, i_mem = h.index("Model"), h.index("Use"), h.index("VRAM"), h.index("Mem%")
    rows = []
    for i, ln in enumerate(plain[head + 1 :], head + 1):
        if not re.match(r"\w+\s+● ", ln):
            break
        status = ln[ln.index("●") + 1 : h.index("Score")].strip()
        rows.append(
            Row(
                i,
                status,
                ln[i_model:i_use].strip(),
                ln[i_vram - 2 : i_vram + 4].strip(),
                ln[i_mem - 1 : i_mem + 4].strip(),
            )
        )
    return rows


def visible_lines() -> int:
    return (H - BAR_PX - 2 * PAD_Y) // LINE_PX


def page(s: State) -> str:
    lines = []
    for i, ln in enumerate(s.lines):
        dim = s.dim_except is not None and i not in s.dim_except
        lines.append(f'<div class="ln{" dim" if dim else ""}">{ln or " "}</div>')
    boxes = []
    for row, col, width, tag in s.boxes:
        top = PAD_Y + row * LINE_PX - 2
        boxes.append(
            f'<div class="box" style="left:calc({PAD_X - 4}px + {col}ch);top:{top}px;width:calc({width}ch + 8px);height:{LINE_PX + 4}px"></div>'
        )
        if tag:
            # 桁(ch)は端末の字体で測るので、札の字体は内側の span にだけ付ける
            boxes.append(
                f'<div style="position:absolute;left:calc({PAD_X}px + {col}ch);top:{top + LINE_PX + 6}px">'
                f'<span class="tag" style="position:static;display:inline-block">{html.escape(tag)}</span></div>'
            )
    probe = f'<span id="probe" style="position:absolute;left:{PAD_X}px;top:0;width:{s.fit_cols}ch"></span>'
    # 拡大の大きさは、描いた字の幅で決める(字体の幅を前もって決め打ちしない)
    script = (
        "<script>const t=document.getElementById('term'),p=document.getElementById('probe');"
        f"const z=Math.min({MAX_ZOOM},({W}-16)/p.getBoundingClientRect().right);"
        f"t.style.transform='translateY({-s.scroll * LINE_PX:.1f}px) scale('+(1+(z-1)*{s.zoom:.3f})+')';</script>"
    )
    cap = f'<div class="cap">{s.caption}</div>' if s.caption else ""
    return (
        f'<!doctype html><meta charset="utf-8"><style>{CSS}</style><body><div id="view">'
        f'<div id="term"><pre>'
        + "".join(lines)
        + "".join(boxes)
        + probe
        + f"</pre></div></div>{cap}{script}</body>"
    )


def build(lang: str) -> list[tuple[State, int]]:
    """(コマの状態, 何コマ続けるか) の並び。"""
    t = TEXT[lang]
    seq: list[tuple[State, int]] = []

    def typing(cmd: str, before: list[str]) -> None:
        for n in range(0, len(cmd) + 1, 2):
            seq.append((State([*before, prompt(cmd[:n], True)]), 1))
        seq.append((State([*before, prompt(cmd, True)]), 4))

    # 1. search を打つ → 表
    typing(t["search"], [])
    out, plain = ansi_lines(run_cli(["search", "flux"], lang))
    plain = ["> " + t["search"], *plain]
    lines = [prompt(t["search"], False), *out]
    rows = table_rows(plain)
    levels = []
    for r in rows:
        if r.level not in levels:
            levels.append(r.level)
    shown = t["joiner"].join(f"<em>{html.escape(x)}</em>" for x in levels)
    base = State(lines, caption=t["judge"].format(levels=shown))
    seq.append((base, 26))

    # 2. PC の行を拡大して、GPU・VRAM・メモリを囲む
    pc = next(i for i, ln in enumerate(plain) if ln.startswith("PC:"))
    pc_text = plain[pc]
    gpu_name = re.search(r"PC: (.+?) VRAM", pc_text).group(1)
    vram = re.search(r"[\d.]+ GiB VRAM|VRAM [\d.]+ GiB", pc_text).group(0)
    ram = re.search(r"(?:(?<![A-Z])RAM|メモリ) [\d.]+ GiB", pc_text).group(0)  # VRAM の中の RAM は取らない
    marks = [
        (pc, *col_of(pc_text, x), tag) for x, tag in zip((gpu_name, vram, ram), t["labels"], strict=True)
    ]
    # 一番右の印(メモリ)まで画面に収まる大きさまで拡大する
    zbase = replace(base, fit_cols=max(col + width for _row, col, width, _tag in marks) + 1, caption="")
    for k in range(1, 5):
        seq.append((replace(zbase, zoom=k / 4), 1))
    seq.append((replace(zbase, zoom=1.0, dim_except=(pc,), boxes=marks, caption=t["auto"]), 32))
    for k in range(3, -1, -1):
        seq.append((replace(zbase, zoom=k / 4), 1))

    # 3. ぎりぎり動く行(Marginal)と、大きすぎる行(Too tight)
    marginal = next((r for r in rows if r.level == "Marginal"), None)
    tight = next((r for r in rows if r.level == "Too tight" and re.fullmatch(r"[\d.]+", r.vram)), None)
    if marginal:
        cap = t["marginal"].format(model=html.escape(marginal.model), mem=html.escape(marginal.mem))
        seq.append(
            (
                replace(
                    base, dim_except=(marginal.index,), boxes=[(marginal.index, 0, COLS - 4, "")], caption=cap
                ),
                26,
            )
        )
    if tight:
        cap = t["too_tight"].format(model=html.escape(tight.model), vram=html.escape(tight.vram))
        seq.append(
            (
                replace(base, dim_except=(tight.index,), boxes=[(tight.index, 0, COLS - 4, "")], caption=cap),
                26,
            )
        )

    # 4. recommend を打つ → 種類ごとの表を送って見せる
    typing(t["recommend"], [])
    out, _plain = ansi_lines(run_cli(["recommend", "-n", "3"], lang))
    lines = [prompt(t["recommend"], False), *out]
    rec = State(lines, caption=t["rec"])
    seq.append((rec, 22))
    last = max(0, len(lines) - visible_lines() + 1)
    steps = 16  # 送りは 1 行ずつより粗く(コマごとに画面が全部変わり、GIF が大きくなるため)
    for k in range(1, steps + 1):
        seq.append((replace(rec, scroll=last * k / steps, caption=t["types"]), 2))
    seq.append((replace(rec, scroll=last, caption=t["types"]), 24))
    seq.append((replace(rec, scroll=last, caption=""), 14))
    return seq


def shoot(exe: str, html_text: str, png: Path, profile: Path) -> None:
    page_path = png.with_suffix(".html")
    page_path.write_text(html_text, encoding="utf-8")
    subprocess.run(
        [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=1",
            f"--user-data-dir={profile}",
            f"--window-size={W},{H}",
            f"--screenshot={png}",
            page_path.as_uri(),
        ],
        capture_output=True,
        timeout=120,
        check=False,
    )
    if not png.exists():
        raise RuntimeError(f"ブラウザが画像を作りませんでした: {png.name}")


def find_ffmpeg(given: str | None) -> str:
    exe = given or shutil.which("ffmpeg")
    if not exe:
        sys.exit("ffmpeg が見つかりません(PATH に置くか --ffmpeg で渡す)")
    return exe


def make_gif(lang: str, out: Path, ffmpeg: str, work: Path) -> int:
    seq = build(lang)
    exe = browser()
    unique: dict[str, int] = {}
    order = []
    for st, n in seq:
        key = page(st)
        idx = unique.setdefault(key, len(unique))
        order.append((idx, n))
    shots = work / "shots"
    shots.mkdir(parents=True, exist_ok=True)
    pages = list(unique)

    def one(i: int) -> None:
        shoot(exe, pages[i], shots / f"s{i:04d}.png", work / f"profile{i % 6}")

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(one, range(len(pages))))
    total = sum(n for _i, n in order)
    frames = work / "frames"
    frames.mkdir(exist_ok=True)
    f = 0
    for idx, n in order:
        with Image.open(shots / f"s{idx:04d}.png") as raw:
            im = raw.convert("RGB").crop((0, 0, W, H))
        for _ in range(n):
            fr = im.copy()
            d = ImageDraw.Draw(fr)
            d.rectangle((0, H - BAR_PX, W, H), fill=(0x2D, 0x33, 0x3B))
            d.rectangle((0, H - BAR_PX, round(W * (f + 1) / total), H), fill=(0x3B, 0x82, 0xF6))
            fr.save(frames / f"f{f:05d}.png")
            f += 1
    gif = out / ("demo.gif" if lang == "en" else f"demo.{lang}.gif")
    pattern = str(frames / "f%05d.png")
    subprocess.run(
        [
            ffmpeg, "-v", "error", "-y", "-framerate", str(FPS), "-i", pattern,
            "-filter_complex",
            "split[a][b];[a]palettegen=max_colors=64:stats_mode=full[p];[b][p]paletteuse=dither=none:diff_mode=rectangle",
            "-loop", "0", str(gif),
        ],
        check=True,
    )  # fmt: skip
    size = gif.stat().st_size
    print(
        f"{'OK  ' if size <= MAX_BYTES else 'FAIL'} {gif} ({total} コマ、{total / FPS:.1f} 秒、{size / 1024 / 1024:.2f} MB)"
    )
    return size


def main() -> int:
    ap = argparse.ArgumentParser(description="README の冒頭の GIF を作る(英語版と日本語版)")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"), help="置く場所(既定 docs/images)")
    ap.add_argument("--ffmpeg", help="ffmpeg の場所(既定は PATH から探す)")
    ap.add_argument("--lang", choices=["en", "ja"], action="append", help="作る言語(既定は両方)")
    ap.add_argument("--keep-frames", metavar="DIR", help="コマの PNG を DIR に残す(確かめる用)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ffmpeg = find_ffmpeg(a.ffmpeg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    too_big = []
    for lang in a.lang or ["en", "ja"]:
        with tempfile.TemporaryDirectory() as t:
            work = Path(t)
            if make_gif(lang, out, ffmpeg, work) > MAX_BYTES:
                too_big.append(lang)
            if a.keep_frames:
                dest = Path(a.keep_frames) / lang
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(work / "frames", dest)
    if too_big:
        print("3 MB を超えた:", ", ".join(too_big))
    return 1 if too_big else 0


if __name__ == "__main__":
    sys.exit(main())
