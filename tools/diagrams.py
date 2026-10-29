# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# ruff: noqa: E501  (HTML と CSS の文字列は 1 行が長い)

"""README の図(段階・share の流れ)を PNG にする(英語版 <名前>.png と日本語版 <名前>.ja.png)。

使い方(リポジトリの一番上で):
    uvx --from . --with pillow python tools/diagrams.py           # docs/images に 4 枚
    uvx --from . --with pillow python tools/diagrams.py --out DIR

図は HTML(文字の色と背景を決めた不透明の板。GitHub の明るい表示でも暗い表示でも読める)で書き、
tools/screenshots.py と同じく Edge か Chrome の headless で撮って余白を切る。
段階の帯(60・85・98%)と動かし方ごとの上限は、手で書かずにコード(estimate.FIT_BANDS・gpu_choice・cpu_top)から取る。
Pillow は撮るときだけ使う(genmodelfit の依存には入れない)。
"""

from __future__ import annotations

import argparse
import html
import importlib
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parent))
from screenshots import browser

import genmodelfit as gmf

# gmf.estimate は関数なので、モジュールは importlib で取る
estimate = importlib.import_module("genmodelfit.estimate")

ROOT = Path(__file__).resolve().parent.parent
BG = "#ffffff"
LEVELS = ("perfect", "good", "marginal", "too_tight")
NAME = {"perfect": "Perfect", "good": "Good", "marginal": "Marginal", "too_tight": "Too tight"}
COLOR = {"perfect": "#1a7f37", "good": "#4ea25a", "marginal": "#bf8700", "too_tight": "#cf222e"}

TEXT = {
    "en": {
        "fit_title": "Fit level = share of the memory the mode uses",
        "fit_sub": "VRAM on the GPU, RAM on the CPU. The table's Mem% column.",
        "axis": "memory used",
        "mode": "Mode",
        "modes": {
            "full": ("GPU", "all on the GPU"),
            "seq": ("GPU~", "parts swapped in turn"),
            "offload": ("GPU+RAM", "part offloaded to RAM (slow)"),
            "cpu": ("CPU", "generation on the CPU (slow)"),
        },
        "cap": "at most {level}",
        "nocap": "no cap",
        "cpu_note": "On the CPU, speech built for real-time CPU use and one-pass auxiliary tools: at most Good.",
        "share_title": "genmodelfit share",
        "steps": [
            ("Find ComfyUI", "the running one, else\nstart the one it finds"),
            ("Show the plan", "models, about 5 min;\nasks once [Y/n]"),
            ("Measure", "one by one; results\nsaved on this PC"),
            ("Show every file", "the whole content and\nthe privacy checks"),
        ],
        "choose": "How to send?",
        "choices": [
            ("1) Pull request", "sign in to GitHub;\na PR from your account"),
            ("2) Issue", "no sign-in; the form opens,\na bot makes the PR"),
            ("q) Not now:", "the results stay saved on this PC; send them later with share --no-measure"),
        ],
        "after": [
            ("Maintainer merges", "a check runs\nwithin minutes"),
            ("Next version uses it", "for everyone with\nthe same GPU"),
        ],
    },
    "ja": {
        "fit_title": "段階 = 動かし方で使うメモリの使用率",
        "fit_sub": "GPU なら VRAM、CPU ならメモリ。表の Mem% の列。",
        "axis": "使用率",
        "mode": "動かし方",
        "modes": {
            "full": ("GPU", "全部 GPU に載せる"),
            "seq": ("GPU~", "部品を順に載せ替える"),
            "offload": ("GPU+RAM", "一部をメモリへ逃がす(遅い)"),
            "cpu": ("CPU", "CPU で生成する(遅い)"),
        },
        "cap": "{level} まで",
        "nocap": "上限なし",
        "cpu_note": "CPU で実時間に動く読み上げと、1 回の計算で終わる補助ツールは、CPU でも Good まで。",
        "share_title": "genmodelfit share",
        "steps": [
            ("ComfyUI を見つける", "起動中のもの。無ければ\n見つけたものを起動"),
            ("計画を見せる", "モデルと時間(5 分ほど)。\n1 回だけ尋ねる [Y/n]"),
            ("測る", "1 つずつ。\n結果はこの PC に保存"),
            ("送る中身を全部見せる", "ファイルのすべてと、\n個人の情報の点検"),
        ],
        "choose": "送り方を選ぶ",
        "choices": [
            ("1) PR", "GitHub にログイン。\nあなたのアカウントから PR"),
            ("2) Issue", "ログインなし。フォームが\n開き、ボットが PR を作る"),
            ("q) 送らない:", "結果はこの PC に残る。あとで share --no-measure で送れる"),
        ],
        "after": [
            ("メンテナがマージ", "数分で検査が動く"),
            ("次の版から使われる", "同じ GPU を使う人\nみんなの判定に"),
        ],
    },
}

CSS = f"""*{{box-sizing:border-box}}
body{{margin:0;background:{BG};font-family:"Segoe UI","Yu Gothic UI","Meiryo",sans-serif;color:#1f2328}}
.card{{display:inline-block;margin:24px;padding:28px 32px;background:{BG};border:2px solid #d0d7de;border-radius:0}}
h1{{font-size:26px;margin:0 0 4px}} .sub{{color:#57606a;font-size:17px;margin:0 0 20px}}
.mono{{font-family:"Cascadia Mono",Consolas,monospace}}
"""


def caps() -> dict[str, str]:
    """動かし方ごとの段階の上限を、判定の関数に聞いて決める(手で書かない)。"""
    tiny = 0.01
    out = {
        "full": estimate.gpu_choice(tiny, None, tiny, 1.0)[0],
        "seq": estimate.gpu_choice(9.0, tiny, 9.0, 1.0)[0],
        "offload": estimate.gpu_choice(9.0, 9.0, tiny, 1.0, ram_ratio=tiny)[0],
    }
    db = gmf.load_db(use_cache=False, user_measurements=False)
    # CPU で生成するモデル(手順を何度も回すもの)の上限 = CPU で動く系統のうち一番低い上限
    out["cpu"] = max((estimate.cpu_top(m) for m in db.models if estimate.cpu_family(m)), key=LEVELS.index)
    tts = next(m for m in db.models if estimate.cpu_realtime_tts(m) is not None)
    out["cpu_tts"] = estimate.cpu_top(tts)
    return out


def bands() -> list[tuple[float, float, str]]:
    """(始まり %, 終わり %, 段階)。FIT_BANDS から作る。"""
    out, lo = [], 0.0
    for limit, level in estimate.FIT_BANDS:
        out.append((lo, limit * 100, level))
        lo = limit * 100
    out.append((lo, 100.0, "too_tight"))
    return out


def worse(level: str, top: str) -> str:
    return LEVELS[max(LEVELS.index(level), LEVELS.index(top))]


def fit_html(lang: str) -> str:
    t = TEXT[lang]
    c = caps()
    width = 640  # 帯の幅(px)。100% を 100% 幅の外に出さないよう、最後の帯は小さくても文字を外に置く
    seg_css = (
        ".row{display:flex;align-items:center;margin:10px 0}"
        ".lab{width:250px;flex:none;font-size:17px}"
        ".lab b{display:inline-block;min-width:84px;font-family:'Cascadia Mono',Consolas,monospace}"
        ".lab span{color:#57606a;font-size:14px;display:block}"
        f".bar{{position:relative;width:{width}px;height:38px;flex:none}}"
        ".seg{position:absolute;top:0;height:38px;color:#fff;font-size:15px;font-weight:600;"
        "display:flex;align-items:center;justify-content:center;white-space:nowrap;border-right:2px solid #fff}"
        ".capnote{margin-left:14px;font-size:15px;color:#1f2328;white-space:nowrap}"
        ".axis{position:relative;width:"
        + str(width)
        + "px;height:26px;margin-left:250px;font-size:15px;color:#57606a}"
        ".tick{position:absolute;top:0;transform:translateX(-50%)}"
        ".hdr{display:flex;align-items:flex-end;margin-bottom:2px}"
        ".note{margin-top:14px;font-size:15px;color:#57606a}"
    )

    def bar(top: str | None) -> str:
        parts = []
        for lo, hi, level in bands():
            shown = worse(level, top) if top else level
            label = NAME[shown] if hi - lo >= 10 else ""
            parts.append(
                f'<div class="seg" style="left:{lo / 100 * width:.1f}px;width:{(hi - lo) / 100 * width:.1f}px;'
                f'background:{COLOR[shown]}">{html.escape(label)}</div>'
            )
        return '<div class="bar">' + "".join(parts) + "</div>"

    ticks = [0.0, *(limit * 100 for limit, _level in estimate.FIT_BANDS)]  # 100% は 98% と重なるので書かない
    axis = "".join(f'<div class="tick" style="left:{x / 100 * width:.1f}px">{x:g}%</div>' for x in ticks)
    rows = [
        f'<div class="row"><div class="lab"><b>{html.escape(t["axis"])}</b></div>{bar(None)}'
        f'<div class="capnote">&gt; {bands()[-1][0]:g}%: {NAME["too_tight"]}</div></div>',
        f'<div class="axis">{axis}</div>',
        f'<div class="hdr"><div class="lab" style="color:#57606a">{html.escape(t["mode"])}</div></div>',
    ]
    for key in ("full", "seq", "offload", "cpu"):
        mark, desc = t["modes"][key]
        cap_text = t["nocap"] if c[key] == LEVELS[0] else t["cap"].format(level=NAME[c[key]])
        rows.append(
            f'<div class="row"><div class="lab"><b>{html.escape(mark)}</b><span>{html.escape(desc)}</span></div>'
            f'{bar(c[key])}<div class="capnote">{html.escape(cap_text)}</div></div>'
        )
    note = t["cpu_note"].replace("Good", NAME[c["cpu_tts"]])
    return (
        f'<!doctype html><meta charset="utf-8"><style>{CSS}{seg_css}</style><body><div class="card">'
        f"<h1>{html.escape(t['fit_title'])}</h1><p class=sub>{html.escape(t['fit_sub'])}</p>"
        + "".join(rows)
        + f'<div class="note">{html.escape(note)}</div></div>'
    )


def lines(text: str) -> str:
    """改行で行を分ける(箱の中で 1 字だけの行ができないよう、折り返しは手で決める)。"""
    return "<br>".join(html.escape(x) for x in text.split("\n"))


def share_html(lang: str) -> str:
    t = TEXT[lang]
    w, gap = 230, 34  # 箱の幅と矢印の幅(下の折り返しの線の位置もこれで決める)
    css = (
        ".flow{display:flex;align-items:stretch}"
        f".box,.choice,.done{{width:{w}px;flex:none;padding:12px 14px;border:2px solid #0969da;"
        "background:#ddf4ff;border-radius:8px}"
        ".box b,.choice b,.done b{display:block;font-size:17px;margin-bottom:4px}"
        ".box span,.choice span,.done span,.later span{font-size:14px;color:#424a53;white-space:nowrap}"
        ".choice,.done{border-color:#1a7f37;background:#dafbe1}"
        ".n{display:inline-block;width:24px;height:24px;border-radius:12px;background:#0969da;color:#fff;"
        "text-align:center;font-size:14px;line-height:24px;margin-right:6px;font-weight:600}"
        f".arr{{width:{gap}px;flex:none;display:flex;align-items:center;justify-content:center;font-size:24px;"
        "color:#57606a}"
        ".q{font-size:17px;font-weight:600;margin:0 0 8px}"
        ".or{width:24px;flex:none;display:flex;align-items:center;justify-content:center;color:#57606a}"
        ".later{margin-top:12px;font-size:15px}.later b{font-family:'Cascadia Mono',Consolas,monospace}"
    )
    steps = []
    for i, (title, desc) in enumerate(t["steps"], 1):
        if i > 1:
            steps.append('<div class="arr">&#8594;</div>')
        steps.append(
            f'<div class="box"><b><span class="n">{i}</span>{html.escape(title)}</b>'
            f"<span>{lines(desc)}</span></div>"
        )
    (a1, b1), (a2, b2), (aq, bq) = t["choices"]
    choices = (
        f'<div class="choice"><b>{html.escape(a1)}</b><span>{lines(b1)}</span></div>'
        '<div class="or">/</div>'
        f'<div class="choice"><b>{html.escape(a2)}</b><span>{lines(b2)}</span></div>'
    )
    after = "".join(
        f'<div class="arr">&#8594;</div><div class="done"><b>{html.escape(a)}</b><span>{lines(b)}</span></div>'
        for a, b in t["after"]
    )
    # 4 の箱の下から、5 の行の頭へ折り返す線
    x4 = 3 * (w + gap) + w / 2
    total = 4 * w + 3 * gap
    turn = (
        f'<svg width="{total}" height="44" style="display:block">'
        '<defs><marker id="h" markerWidth="10" markerHeight="10" refX="5" refY="8" orient="auto">'
        '<path d="M0,0 L10,0 L5,9 z" fill="#57606a" transform="rotate(0)"/></marker></defs>'
        f'<path d="M{x4},2 V20 H{w / 2} V36" fill="none" stroke="#57606a" stroke-width="2"/>'
        f'<path d="M{w / 2 - 6},34 L{w / 2 + 6},34 L{w / 2},43 z" fill="#57606a"/></svg>'
    )
    return (
        f'<!doctype html><meta charset="utf-8"><style>{CSS}{css}</style><body><div class="card">'
        f'<h1 class="mono">{html.escape(t["share_title"])}</h1><p class=sub></p>'
        f'<div class="flow">{"".join(steps)}</div>{turn}'
        f'<div class="q"><span class="n">5</span>{html.escape(t["choose"])}</div>'
        f'<div class="flow">{choices}{after}</div>'
        f'<div class="later"><b>{html.escape(aq)}</b> <span>{html.escape(bq)}</span></div>'
        "</div>"
    )


def shoot(exe: str, page: Path, png: Path, tmp: Path) -> None:
    subprocess.run(
        [
            exe,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--force-device-scale-factor=2",
            f"--user-data-dir={tmp / 'profile'}",
            "--window-size=1500,900",
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
    box = ImageChops.difference(im, Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if box is None:
        raise RuntimeError("画像が空です")
    left, top, right, bottom = box
    if bottom >= im.height - 4 or right >= im.width - 4:
        raise RuntimeError("内容が窓からはみ出しました(切れています)")
    # 板の枠(2px の線)で切る。外は不透明の白のまま残さない
    im = im.crop((left, top, right, bottom))
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS) if im.width > 1800 else im
    im.save(png, optimize=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="README の図(段階・share の流れ)を PNG にする")
    ap.add_argument("--out", default=str(ROOT / "docs" / "images"), help="置く場所(既定 docs/images)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    exe = browser()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        for name, make in (("fit-levels", fit_html), ("share-flow", share_html)):
            for lang, suffix in (("en", ""), ("ja", ".ja")):
                page = tmp / f"{name}{suffix}.html"
                page.write_text(make(lang), encoding="utf-8")
                png = out / f"{name}{suffix}.png"
                png.unlink(missing_ok=True)
                shoot(exe, page, png, tmp)
                print("OK  ", png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
