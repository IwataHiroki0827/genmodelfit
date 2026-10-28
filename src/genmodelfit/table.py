# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""端末向けの表(Issue #7、#17)。llmfit の `fit` の表にならう。

- 先頭に PC と「何件中何件を出したか・隠した件数と理由」、末尾に印の説明
- 幅 100 文字の端末で崩れない。モデル名は残りの幅を全部使い、長い名前は真ん中を省略記号で詰める
  (狭い端末では Disk → Mem% → License → Prec → Use の順に列を外す)
- 段階は色と「●」で分ける。色の無い端末でも文字で分かる(NO_COLOR やパイプ出力では色を付けない)。
  絵文字は Windows の端末(cp932)で幅が崩れるので使わない
- VRAM は判定に使った値(m = 実測)。Mode は GPU / GPU~(部品を順に載せ替え)/ GPU+RAM(逃がす)/ CPU / -(動かない)
- 自動のおすすめ(gate。JSON の auto)の列(Verified)は外した(#84)。info と gate で出す
- 「?」は分からない(精度・ライセンス・大きさ = 判定していない)、「-」は当てはまらない
  (動かないので Mode が無い、CPU で動くので VRAM が無い、GPU が無いので Mem% が無い、など)。
  #30 で使い分けをそろえた
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text

from . import i18n
from .groups import precision_label
from .i18n import tr
from .recommend import FIT_LABEL, _ram_short, mode_label, usage_ratio, vram_need
from .tasks import MODALITY_LABEL, USES

LEVEL_STYLE = {
    "perfect": "bold green",
    "good": "cyan",
    "marginal": "yellow",
    "too_tight": "red",
    "unknown": "dim",
}
LEVEL_LABEL = FIT_LABEL
MARK = "●"  # 段階の印(絵文字ではない記号)
# 隠した理由の表示名(recommend.HIDDEN_KEYS)
HIDDEN_LABEL = {
    "modality": "other types",
    "uncensored": "uncensored",
    "sfw": "sfw",
    "tools": "tools",
    "license": "license",
    "license_unknown": "license unknown",
    "size_unknown": "size unknown",
    "too_tight": "too tight",
    "below_min_fit": "below --min-fit",
    "other_fit": "other levels (--fit)",
    "held": "held",
    "task": "other uses",
    "same_model": "merged into their models",
}
# 日本語の表示名(#76)
HIDDEN_LABEL_JA = {
    "modality": "ほかの種類",
    "uncensored": "無検閲",
    "sfw": "sfw",
    "tools": "補助ツール",
    "license": "ライセンス",
    "license_unknown": "ライセンス不明",
    "size_unknown": "大きさ不明",
    "too_tight": "動かない",
    "below_min_fit": "--min-fit 未満",
    "other_fit": "ほかの段階(--fit)",
    "held": "保留",
    "task": "ほかの用途",
    "same_model": "モデルにまとめた",
}


def hidden_label(key: str) -> str:
    """隠した理由の表示名(今の言語)。"""
    return tr(HIDDEN_LABEL_JA.get(key, key), HIDDEN_LABEL.get(key, key))


MAX_WIDTH = 100
MIN_NAME_WIDTH = 24  # モデル名がこれより狭くなるなら列を外す
# 列の幅(並びもこの順。Model は Score の右)と、狭いときに外す順。Type・Status・Score・Model は外さない
# (0.1.1 で Type を左端、Score を Model の左にした。#28 の利用者の指定)。
# Censored は内容の印の列(0.6.0。名前の前の「[U] 」の代わり。Y = 検閲あり(sfw)/ N = 無検閲で、空にしない)。
# 無検閲の行が出うるとき(--content all|uncensored)だけ出し、外さない
CONTENT_COLUMN = "Censored"
LEFT_COLUMNS = ("Type", "Status", "Score", CONTENT_COLUMN)
COLUMN_WIDTH = {
    "Type": 5,
    "Status": 11,
    "Score": 5,
    CONTENT_COLUMN: 8,
    "Use": 9,  # 「gen,edit」と、一番長い用途 1 つ +「,…」(「lipsync,…」)が収まる幅(#84)
    "Prec": 4,
    "VRAM": 5,
    "Mem%": 4,
    "Mode": 7,
    "Disk": 4,
    "License": 10,
}
DROP_ORDER = ("Disk", "Mem%", "License", "Prec", "Use")


def output_width(file=None) -> int:
    """出力の幅(桁)。COLUMNS → 端末の幅 → 100 の順に見て、100 を超えない。"""
    out = file or sys.stdout
    cols = os.environ.get("COLUMNS", "")  # パイプの先でも幅を指定できるように(shutil と同じ考え方)
    if cols.isdigit() and int(cols) > 0:
        return min(int(cols), MAX_WIDTH)
    try:
        return min(os.get_terminal_size(out.fileno()).columns, MAX_WIDTH)
    except (OSError, AttributeError, ValueError):
        return MAX_WIDTH


# 折り返しの禁則(#76): 行の頭に置かない文字と、行の終わりに置かない文字
_NO_LINE_START = frozenset("、。,.)」』）〕】・:;!?：；！？ー")
_NO_LINE_END = frozenset("(「『（〔【")
# 数のあとの単位(「3.95 GB」を数と単位で分けない)
_UNITS = frozenset({"GB", "GiB", "MB", "MiB", "KB", "TB", "px", "s", "%", "B"})
_NUMBER = re.compile(r"[(（]?[+-]?\d[\d.,]*$")


# ASCII の「(」の前でも分ける(日本語の文は「ファイル名(5.25 GB、…)」のように半角の括弧を続けて書く)。
# 「(s)」(model(s) など)は分けない
# 「<フォルダ>/<GPU>.json」のように < > で囲んだ日本語の入る印は、ASCII の並びの一部として切らない
_ATOM = re.compile(r" +|\(s\)|\(*(?:<[^<>\s]*>|[!-'\)-~])+|\(+|[^ !-~]")


def _atoms(text: str) -> list[str]:
    """空白の並び・ASCII の語(空白を含まない ASCII の並び。「(」の前で分ける)・ASCII でない 1 文字に
    分ける。"""
    return _ATOM.findall(text)


def _chunks(text: str) -> list[tuple[str, str]]:
    """折り返せない塊 (前の空白, 塊) の並び。塊の間だけで折り返す(空白は折り返しの所で落とす)。

    ASCII の語(ファイル名・URL・--uncensored・数)は塊の途中で切らない。数と単位(3.95 GB)はつなぐ。
    行の頭に置かない文字(、。)」 など)は前に、行の終わりに置かない文字((「 など)は後ろにつなぐ。"""
    chunks: list[list[str]] = []  # [前の空白, 塊]
    space = ""
    for a in _atoms(text):
        if a[0] == " ":
            space += a
            continue
        if chunks:
            prev = chunks[-1][1]
            glue = (
                (space and _NUMBER.search(prev) and a.rstrip(")）,.、。") in _UNITS)
                or (not space and (a[0] in _NO_LINE_START or prev[-1] in _NO_LINE_END))
                or (not space and a == "(s)")
            )
            if glue:
                chunks[-1][1] = prev + space + a
                space = ""
                continue
        chunks.append([space, a])
        space = ""
    return [(s, c) for s, c in chunks]


def _hard_split(chunk: str, room: int) -> tuple[str, str]:
    """幅より長い塊を、幅に入るところで切る(ASCII の語の途中では切らない。日本語の文字の間で切る)。"""
    used, cut = 0, 0
    for i, ch in enumerate(chunk[:-1]):
        used += cell_len(ch)
        if used > room:
            break
        nxt = chunk[i + 1]
        if not (ch.isascii() and nxt.isascii()) and nxt not in _NO_LINE_START and ch not in _NO_LINE_END:
            cut = i + 1
    return chunk[:cut], chunk[cut:]  # 切れなければ ("", 塊)


def wrap_text(text: str, width: int, first: str = "", rest: str = "", space_room: bool = True) -> list[str]:
    """first + text を幅(全角は 2 桁)に収まるように折り返す。続きの行の頭は rest。

    英数字の並び(ファイル名・URL・オプション・単位つきの数)の途中では折り返さない(1 つで幅より長ければ
    その行だけはみ出す)。行の頭に 、。) などを、行の終わりに ( 「 などを置かない(#76)。
    space_room: 後ろに空白で続く行は、その空白の分も幅に入れる(表・info の前からの折り返しと同じ位置。
    False は textwrap と同じく幅いっぱいまで使う: --help)。ただし、その空白の分だけで溢れるときも、
    行の前の部分が幅の半分までなら同じ行に置く(前の版と同じ。短い見出しの後ろの長いパスを次の行へ送らない)。"""
    out: list[str] = []
    line, prefix = "", first
    chunks = _chunks(text)
    for k, (space, chunk) in enumerate(chunks):
        # 後ろに空白で続くときは、その空白まで幅に入れる(前からの英語の折り返しと同じ位置にする)
        after = 1 if space_room and k + 1 < len(chunks) and chunks[k + 1][0] else 0
        while True:
            sep = space if line else ""
            need = cell_len(prefix + line + sep + chunk)
            # 空白の分だけで溢れる(need == width)ときは、前の部分が幅の半分までなら動かさない
            if need + after <= width or (need <= width and cell_len(prefix + line) <= width // 2):
                line += sep + chunk
                break
            if line:  # 次の行へ
                out.append(prefix + line)
                line, prefix, space = "", rest, ""
                continue
            if cell_len(prefix + chunk) <= width:  # 行の頭なら、後ろの空白は数えない
                line = chunk
                break
            # 1 つの塊だけで幅を超える: 日本語の文字の間で切れるなら切る、切れなければはみ出す
            head, remainder = _hard_split(chunk, width - cell_len(prefix))
            if not head:
                line = chunk
                break
            out.append(prefix + head)
            prefix, chunk = rest, remainder
    out.append(prefix + line)
    return out


def wrap_line(text: str, width: int) -> list[str]:
    """1 行を幅に収まるように折り返す(全角は 2 桁)。続きの行は元の字下げ + 4 桁。"""
    if cell_len(text) <= width:
        return [text]
    lead = len(text) - len(text.lstrip(" "))
    pad = " " * min(lead + 4, max(0, width // 2))
    head = lead + 2 if text[lead:].startswith("- ") else lead  # 箇条書きの「- 」は次の語と離さない
    return [part.rstrip(" ") for part in wrap_text(text[head:], width, text[:head], pad)]


class _Console(Console):
    def on_broken_pipe(self) -> None:
        # rich の既定は終了コード 1 で終わる(Linux)。CLI の BrokenPipeError の扱いにそろえる
        self.quiet = True
        raise BrokenPipeError


def make_console(file=None, width: int | None = None) -> Console:
    out = file or sys.stdout
    width = output_width(out) if width is None else min(width, MAX_WIDTH)
    return _Console(file=out, width=width, soft_wrap=False, highlight=False)


def middle_ellipsis(text: str, width: int) -> str:
    """真ん中を省略記号にする。頭が共通の名前(video_wan21_scail2_… など)も末尾で区別できるように。"""
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"[:width]
    head = (width - 1) * 11 // 20  # 頭を少し多めに残す
    tail = width - 1 - head
    return text[:head] + "…" + (text[-tail:] if tail else "")


def _window(text: str, width: int, head: int) -> str:
    return text[:head] + "…" + (text[len(text) - (width - 1 - head) :] if width - 1 - head > 0 else "")


def distinct_names(names: Sequence[str], width: int) -> dict[str, str]:
    """表に出す名前を決める。真ん中を省略して重なったら、違う部分が見える切り方を探す
    (7b / 3b のように真ん中だけが違う名前を区別するため)。どうしても区別できなければそのまま。"""
    shown = {n: middle_ellipsis(n, width) for n in names}
    for _ in range(3):
        groups: dict[str, list[str]] = {}
        for n, s in shown.items():
            groups.setdefault(s, []).append(n)
        clashes = [g for g in groups.values() if len(g) > 1]
        if not clashes:
            break
        taken = set(shown.values())
        for group in clashes:
            for n in group[1:]:
                for head in range(width - 1, -1, -1):
                    cand = _window(n, width, head)
                    if cand not in taken:
                        taken.add(cand)
                        shown[n] = cand
                        break
    return shown


def model_width(width: int, cols: Sequence[str]) -> int:
    used = sum(COLUMN_WIDTH[c] for c in cols) + len(cols)  # 列の間は 1 文字(Model の列を入れて len(cols) 個)
    return max(12, width - used)


def columns_for(width: int, with_type: bool = True, with_content: bool = False) -> list[str]:
    """幅に収まる列(Model 以外。並びは COLUMN_WIDTH の順)。with_type=False なら Type を出さない。
    with_content=True なら Censored(内容の印)の列を出す。"""
    cols = [c for c in COLUMN_WIDTH if (with_type or c != "Type") and (with_content or c != CONTENT_COLUMN)]
    for drop in DROP_ORDER:
        if model_width(width, cols) >= MIN_NAME_WIDTH:
            break
        if drop in cols:
            cols.remove(drop)
    return cols


def _gb(v: float) -> str:
    """GiB の値を 4 桁に収める(100 以上は整数、1000 以上は 999+)。"""
    if v < 99.95:
        return f"{v:.1f}"
    return f"{v:.0f}" if v < 999.5 else "999+"


def _vram_cell(e, system) -> Text:
    if e.unknown_size:
        return Text("?")
    need = vram_need(e, system)
    if need is None:
        if e.fit == "too_tight":  # GPU が無くて動かない: 最低これだけの VRAM が要る
            return Text(_gb(e.vram_offload_gb), style="red")
        return Text("-", style="dim")
    if e.basis == "measured":
        # c = ほかの利用者の実測(件数は info・JSON の sample_count)、m = 作者・この PC の実測
        mark = "c" if e.basis_detail == "measured_community" else "m"
        return Text.assemble(_gb(need), (mark, "bold"))
    return Text(_gb(need))


def _mem_cell(e, system) -> Text:
    if e.unknown_size:
        return Text("?")
    if _ram_short(e, system):  # メモリが理由(VRAM の割合だけだと足りているように見えるので)
        return Text("RAM", style="red" if e.fit == "too_tight" else "yellow")
    # その動かし方で一番きつい資源の使用率 r(Score の入力と同じ。#95、docs/scoring.md「Score」)
    r = usage_ratio(e, system)
    if r is None:
        return Text("-", style="dim")
    pct = 100 * r
    # 足りないのに 100% と出さない。載るのに 101% とも出さない
    shown = max(round(pct), 101) if r > 1 else min(round(pct), 100)
    style = "red" if pct > 100 else ("yellow" if pct > 90 else "")
    return Text(f"{shown}%" if shown <= 999 else ">999", style=style)


def mode_cell(e, system) -> str:
    """表の Mode。JSON の run_mode_label(mode_label)に加えて、動かない理由を短く出す(0.7.0、#22):
    RAM = メインメモリが足りない、GPU req = 使える GPU が無いので動かない(GPU が要る)。"""
    label = mode_label(e, system)
    if label != "-" or e.fit != "too_tight" or e.unknown_size:
        return label
    if _ram_short(e, system):
        return "RAM"
    if system is not None and system.vram_gb <= 0:
        return "GPU req"
    return label


def row_uses(f, grouped: bool = True) -> list[str]:
    """表の Use に出す用途(#84)。先頭は行自身の用途(f.use: --task で合った用途か代表の用途。幅が足りなくても
    見えるように)、そのあとに仲間のワークフロー(fit.workflows。絞り込みで隠した仲間も)の用途を重複なく
    USES の順に。--all-workflows(grouped=False)はその行の用途だけ。分からなければ「?」。"""
    own = getattr(f, "use", "") or "?"
    if not grouped:
        return [own]
    rest = {w.get("use") or "?" for w in getattr(f, "workflows", []) or []} - {own}
    order = [*USES, "?"]
    return [own, *sorted(rest, key=lambda u: (order.index(u) if u in order else len(order), u))]


def use_text(uses: Sequence[str], width: int) -> str:
    """用途を「gen,edit」のように並べる。幅を超えるときは、収まるところまでの用途の後ろに「…」を付ける
    (用途の途中では切らない。全部は info か --all-workflows で見る)。"""
    text = ",".join(uses)
    if cell_len(text) <= width:
        return text
    out = uses[0]
    for u in uses[1:]:
        if cell_len(f"{out},{u},…") > width:
            break
        out = f"{out},{u}"
    return f"{out},…" if cell_len(f"{out},…") <= width else f"{out[: width - 1]}…"


def _use_cell(f, grouped: bool) -> Text:
    return Text(use_text(row_uses(f, grouped), COLUMN_WIDTH["Use"]))


def content_column_wanted(fits: Sequence, content: str | None = None) -> bool:
    """Censored の列を出すか。content(--content の値)があればそれで決め(all / uncensored なら出す)、
    無ければ(Python から表を作るとき)無検閲の行があるときだけ出す。"""
    if content is not None:
        return content in ("all", "uncensored")
    return any(f.model.content == "uncensored" for f in fits)


def build_table(
    fits: Sequence,
    width: int = MAX_WIDTH,
    system=None,
    with_type: bool | None = None,
    grouped: bool = True,
    with_content: bool | None = None,
) -> Table:
    """with_content: Censored の列を出すか(None は content_column_wanted(fits) で決める)。"""
    if with_type is None:
        with_type = True  # 0.1.1 から、種類の列は常に左端に出す(表は 1 つ。#28)
    if with_content is None:
        with_content = content_column_wanted(fits)
    cols = columns_for(width, with_type, with_content)
    name_w = model_width(width, cols)
    left = [c for c in cols if c in LEFT_COLUMNS]
    right = [c for c in cols if c not in LEFT_COLUMNS]
    t = Table(
        header_style="bold",
        expand=False,
        pad_edge=False,
        box=None,
        show_edge=False,
        padding=(0, 1, 0, 0),
    )
    numeric = {"Score", "VRAM", "Mem%", "Disk"}

    def add(c: str) -> None:
        t.add_column(
            c,
            justify="right" if c in numeric else "left",
            no_wrap=True,
            overflow="ellipsis",
            width=COLUMN_WIDTH[c],
        )

    for c in left:  # Type | Status | Score | Censored
        add(c)
    # モデル名は他の列の残りを全部使う。長い名前は真ん中を省略記号にする(middle_ellipsis)
    t.add_column("Model", no_wrap=True, overflow="ellipsis", width=name_w)
    for c in right:
        add(c)

    # まとめた表はモデルの表示名(一覧の中で重ならない)、--all-workflows はワークフローの ID
    def label(f) -> str:
        return (getattr(f, "model_name", "") or f.model.id) if grouped else f.model.id

    shown = distinct_names([label(f) for f in fits], name_w)
    for f in fits:
        e, m = f.estimate, f.model
        name = Text(shown[label(f)])
        cells = {
            "Type": Text(m.modality),
            "Status": Text(f"{MARK} {LEVEL_LABEL.get(e.fit, e.fit)}", style=LEVEL_STYLE.get(e.fit, "")),
            "Use": _use_cell(f, grouped),
            "Prec": Text(precision_label(m.precision)),
            "Score": Text(f"{f.score:.0f}"),
            CONTENT_COLUMN: Text("N", style="magenta") if m.content == "uncensored" else Text("Y"),
            "VRAM": _vram_cell(e, system),
            "Mem%": _mem_cell(e, system),
            "Mode": Text(mode_cell(e, system), style="red" if e.fit == "too_tight" else ""),
            "Disk": Text(_gb(e.disk_gb) if e.disk_gb else "?"),
            "License": Text(m.license or "?"),
        }
        t.add_row(*(cells[c] for c in left), name, *(cells[c] for c in right))
    return t


BACKEND_LABEL = {"cuda": "CUDA", "rocm": "ROCm", "xpu": "XPU", "mps": "MPS"}


def _unusable_gpus(system) -> list[str]:
    """判定に使えなかった GPU の説明(「Intel(R) UHD Graphics 770 (integrated; not usable for generation →
    judged as CPU)」)。NVIDIA は nvidia-smi で見つけたもの、ほかは画面の部品の一覧から。"""
    out: list[str] = []
    seen: set[str] = set()
    for g in system.gpus:
        if g.name not in seen:
            seen.add(g.name)
            out.append(
                tr(
                    f"{g.name}(生成に使えない → CPU として判定)",
                    f"{g.name} (not usable for generation → judged as CPU)",
                )
            )
    for a in getattr(system, "other_gpus", []) or []:
        if (
            a.name in seen or a.vendor == "unknown"
        ):  # 会社の分からない画面(Microsoft の基本の画面など)は出さない
            continue
        seen.add(a.name)
        kind = tr("内蔵。" if a.integrated else "", "integrated; " if a.integrated else "")
        if getattr(a, "needs", None):  # OS の版だけで断った GPU は、使えるようにする条件を出す
            out.append(
                tr(
                    f"{a.name}({kind}生成には {a.needs} が要る → CPU として判定)",
                    f"{a.name} ({kind}needs {a.needs} for generation → judged as CPU)",
                )
            )
        else:
            out.append(
                tr(
                    f"{a.name}({kind}生成に使えない → CPU として判定)",
                    f"{a.name} ({kind}not usable for generation → judged as CPU)",
                )
            )
    return out


def pc_line(system) -> str:
    """この PC の 1 行。調べた値なら空きも出す。--vram / --ram で上書きしたら assumed(仮定)。
    使える GPU が無ければ、見つけた GPU とその扱い(CPU として判定した)を出す(0.7.0、#22)。"""
    g = system.best_gpu
    if g and g.vendor != "nvidia":
        backend = BACKEND_LABEL.get(g.backend, g.backend)
        memory = getattr(g, "memory", "dedicated")
        size = f"{g.vram_total_gb:.1f} GiB"
        if memory == "unified":
            gpu = tr(
                f"{g.name}(ユニファイドメモリ。GPU に {size}、{backend}、推定)",
                f"{g.name} (unified memory; {size} for the GPU, {backend}, estimated)",
            )
        elif memory == "shared":
            gpu = tr(
                f"{g.name}(内蔵。メモリと共有で {size}、{backend}、推定)",
                f"{g.name} (integrated; {size} shared with RAM, {backend}, estimated)",
            )
        else:
            gpu = tr(f"{g.name} VRAM {size}({backend}、推定)", f"{g.name} {size} VRAM ({backend}, estimated)")
    elif g:
        if g.vram_free_gb is not None:
            free = tr(f"(空き {g.vram_free_gb:.1f})", f" ({g.vram_free_gb:.1f} free)")
        else:
            free = ""
        gpu = tr(
            f"{g.name} VRAM {g.vram_total_gb:.1f} GiB{free}", f"{g.name} {g.vram_total_gb:.1f} GiB VRAM{free}"
        )
    else:
        found = _unusable_gpus(system)
        gpu = tr("、", "; ").join(found) if found else tr("GPU なし(CPU だけ)", "no GPU (CPU only)")
    if "vram" in system.overrides:
        if system.vram_gb > 0:
            gpu = tr(
                f"VRAM {system.vram_gb:.1f} GiB と仮定(--vram)",
                f"assumed {system.vram_gb:.1f} GiB VRAM (--vram)",
            )
        else:
            gpu = tr("GPU なしと仮定(--vram 0)", "assumed no GPU (--vram 0)")
    if "ram" in system.overrides:
        ram = tr(f"メモリ {system.ram_total_gb:.1f} GiB(--ram)", f"RAM {system.ram_total_gb:.1f} GiB (--ram)")
    else:
        ram = tr(
            f"メモリ {system.ram_total_gb:.1f} GiB(空き {system.ram_available_gb:.1f})",
            f"RAM {system.ram_total_gb:.1f} GiB ({system.ram_available_gb:.1f} available)",
        )
    return tr(f"PC: {gpu}、{ram}", f"PC: {gpu}, {ram}")


def no_gpu_line(system, hidden: Mapping[str, int], fits: Sequence = ()) -> str:
    """使える GPU が無いときの 1 行(0.7.0、#22)。GPU があれば空。"""
    if system is None or system.vram_gb > 0:
        return ""
    tools = hidden.get("tools", 0)
    if tools:
        tail = tr(
            f"(補助ツール {tools} 件は隠した: --include-tools)", f" ({tools} tools hidden: --include-tools)"
        )
    else:
        tail = ""
    if any(f.estimate.fit == "too_tight" for f in fits):
        return tr(
            f"この PC には使える GPU がありません: Mode が GPU req のモデルは GPU が要ります{tail}",
            f"This PC has no usable GPU: models marked GPU req need one{tail}",
        )
    return tr(
        f"この PC には使える GPU がありません: CPU で動くモデルだけを出します{tail}",
        f"This PC has no usable GPU: only models that run on CPU are shown{tail}",
    )


def hidden_text(hidden: Mapping[str, int]) -> str:
    """「(N hidden: uncensored 39, tools 20; 24 more workflows merged into their models)」。
    何も無ければ空。"""
    merged = hidden.get("same_model", 0)
    rest = {k: v for k, v in hidden.items() if v and k != "same_model"}
    parts = []
    if rest:
        total = sum(rest.values())
        parts.append(
            tr(
                f"隠した {total} 件: " + "・".join(f"{hidden_label(k)} {v}" for k, v in rest.items()),
                f"{total} hidden: " + ", ".join(f"{hidden_label(k)} {v}" for k, v in rest.items()),
            )
        )
    if merged:
        parts.append(
            tr(
                f"ほかに {merged} 件のワークフローをモデルにまとめた",
                f"{merged} more workflows merged into their models",
            )
        )
    if not parts:
        return ""
    return tr(f"({'。'.join(parts)})", f"({'; '.join(parts)})")


# 表の下の「Order:」の説明(並べ方ごと。recommend は auto)
ORDER_TEXT = {
    "auto": "Order: runs, Perfect/Good, Marginal (faster Mode first), official, popularity, Score.",
    "type": "Order: Type, Status, official, popularity, Score (--sort).",
    "score": "Order: runs, Perfect/Good, Marginal, Score, popularity (all types).",
    "vram": "Order: runs, VRAM (least first).",
    "disk": "Order: runs, Disk (smallest first).",
    "name": "Order: runs, Model name.",
}
# 日本語(#76。列の名前・段階の名前・Mode の印は表と同じ英語のまま)
ORDER_TEXT_JA = {
    "auto": "並び: 動く → Perfect/Good → Marginal(速い Mode が先)→ 公式 → 人気 → Score。",
    "type": "並び: Type → Status → 公式 → 人気 → Score(--sort)。",
    "score": "並び: 動く → Perfect/Good → Marginal → Score → 人気(種類をまたいで)。",
    "vram": "並び: 動くもの → VRAM の少ない順。",
    "disk": "並び: 動くもの → Disk の小さい順。",
    "name": "並び: 動くもの → モデルの名前。",
}


def legend(
    fits: Sequence,
    hidden: Mapping[str, int],
    grouped: bool = True,
    order: str = "auto",
    with_content: bool | None = None,
    system=None,
    json_first: bool | None = None,
) -> list[str]:
    """表の下の説明。json_first: 「--json は自動のおすすめを先に並べる」の行を出すか(None は order == "auto"。
    --auto-only では表と JSON の並びが同じなので CLI は False を渡す。#84)。"""
    if json_first is None:
        json_first = order == "auto"
    system_marks = any(mode_cell(f.estimate, system) in ("GPU req", "RAM") for f in fits)
    best = system.best_gpu if system is not None else None
    non_nvidia = best is not None and best.vendor != "nvidia"
    community = any(getattr(f.estimate, "basis_detail", "") == "measured_community" for f in fits)
    shown = content_column_wanted(fits) if with_content is None else with_content
    if i18n.get_language() == "ja":
        return _legend_ja(
            fits, hidden, grouped, order, shown, system_marks, non_nvidia, community, json_first
        )
    out = [  # 1 項目 1 行(幅 100 に収める。狭い端末では字下げして折り返す)
        ORDER_TEXT.get(order, ORDER_TEXT["auto"]),
        "Score 0-100 = memory headroom (Mem% <= 70% is 100; not speed). VRAM: GiB needed, m = measured.",
        *(["VRAM: c = measured by other users (how many: genmodelfit info <id>)."] if community else []),
        "Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.",
        *(
            ["Mode for no run: GPU req = needs a GPU (none usable here), RAM = not enough system RAM."]
            if system_marks
            else []
        ),
        "Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).",
        *(
            [  # 実測を送ってもらう案内(1 行だけ。#56 の後の 0.9.0)
                "Level caps: AMD / Intel / Apple GPUs are estimates, at most Good until measured"
                " (CONTRIBUTING.md)."
            ]
            if non_nvidia
            else []
        ),
        "Mem%: share of the tightest memory (VRAM; RAM for CPU, or if larger for GPU+RAM), RAM = RAM short.",
    ]
    if grouped:
        out.append(
            "One row per model: Prec = best precision that fits, Use = workflow uses (info, --all-workflows)."
        )
    out.append(
        "Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md)."
    )
    if json_first:  # recommend の --json だけ並びが表と違う(#84)
        out.append("--json lists auto-recommendable models first (schema.md).")
    if shown:
        out.append("Censored: N = safety filter removed / adult model")
    if hidden.get("uncensored"):
        out.append(f"Note: {hidden['uncensored']} uncensored model(s) hidden; add --uncensored to show them.")
    if order != "auto" and any(hidden.get(k) for k in ("tools", "too_tight", "size_unknown", "other_fit")):
        out.append("Note: add --all to also show tools, models that do not run and models not judged (?).")
    if hidden.get("license_unknown"):
        out.append(
            f"Note: {hidden['license_unknown']} model(s) with an unknown license hidden;"
            " add --any-license to show them."
        )
    out.append("? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).")
    if any(f.estimate.unknown_size for f in fits):
        out.append("Status ? = size unknown, not judged (after the levels that run, before Too tight).")
    return out


def _legend_ja(
    fits: Sequence,
    hidden: Mapping[str, int],
    grouped: bool,
    order: str,
    shown: bool,
    system_marks: bool,
    non_nvidia: bool,
    community: bool,
    json_first: bool = False,
) -> list[str]:
    """表の下の説明の日本語(#76)。英語と同じ項目を同じ順に、1 項目 1 行で。"""
    out = [
        ORDER_TEXT_JA.get(order, ORDER_TEXT_JA["auto"]),
        "Score 0〜100 = メモリの余裕(Mem% 70% 以下で 100。速さではない)。VRAM = 要る GiB(m = 実測)。",
        *(["VRAM の c = ほかの利用者の実測(件数は genmodelfit info <id>)。"] if community else []),
        "Mode: GPU 全部 GPU、GPU~ 部品を順に載せ替え、GPU+RAM メモリへ逃がす(遅い)、CPU(遅い)、- 動かない。",
        *(
            ["動かないときの Mode: GPU req = GPU が要る(ここでは使えない)、RAM = メインメモリが足りない。"]
            if system_marks
            else []
        ),
        "段階の上限: GPU~ は Good まで、GPU+RAM と CPU は Marginal まで(Mem% が低くても)。",
        *(
            [
                "段階の上限: AMD / Intel / Apple の GPU は推定なので、実測が入るまで Good まで"
                "(CONTRIBUTING.md)。"
            ]
            if non_nvidia
            else []
        ),
        "Mem% = 一番きついメモリの使用率(VRAM。CPU はメモリ、GPU+RAM は大きいほう)、RAM = メモリ不足。",
    ]
    if grouped:
        out.append(
            "1 行に 1 モデル: Prec = 載る中で一番高い精度、Use = ワークフローの用途(info、--all-workflows)。"
        )
    out.append("m の無い見積もりは目安。詳しくは genmodelfit info <名前か ID>。アプリは --json(schema.md)。")
    if json_first:
        out.append("--json は自動でおすすめしてよいものを先に並べる(schema.md)。")
    if shown:
        out.append("Censored: N = 安全のための絞り込みを外したモデル・成人向けのモデル")
    if hidden.get("uncensored"):
        out.append(f"注意: 無検閲のモデル {hidden['uncensored']} 件を隠しました。出すなら --uncensored。")
    if order != "auto" and any(hidden.get(k) for k in ("tools", "too_tight", "size_unknown", "other_fit")):
        out.append("注意: 補助ツール・動かないモデル・判定していないモデル(?)も出すなら --all。")
    if hidden.get("license_unknown"):
        out.append(
            f"注意: ライセンスの分からないモデル {hidden['license_unknown']} 件を隠しました。"
            "出すなら --any-license。"
        )
    out.append("? = 分からない(精度・ライセンス・大きさ。判定していない)、- = 当てはまらない(動かないなど)。")
    if any(f.estimate.unknown_size for f in fits):
        out.append("Status の ? = 大きさが分からず判定していない(動く段階の後、Too tight の前に並ぶ)。")
    return out


def print_fit_table(
    fits,
    title: str = "",
    system=None,
    console: Console | None = None,
    notes: Sequence[str] = (),
    matched: int | None = None,
    hidden: Mapping[str, int] | None = None,
    groups: Sequence[Mapping[str, Any]] | None = None,
    grouped: bool = True,
    order: str = "auto",
    content: str | None = None,
    json_first: bool | None = None,
) -> None:
    """PC の行 → 見出し(何件中何件・隠した理由)→ 表 → 説明。notes は末尾に足す一行の注意。

    content は絞り込みの --content(sfw / all / uncensored)。all・uncensored なら Censored の列を出す
    (None は無検閲の行があるときだけ)。節に分けても列はそろえる。

    groups を渡すと種類ごとの節に分ける(見出し 1 行 + 表。動くものが無い種類は 1 行で知らせる)。"""
    con = console or make_console()
    hidden = dict(hidden or {})
    kw = {"grouped": grouped, "with_content": content_column_wanted(fits, content)}
    matched = len(fits) if matched is None else matched

    def say(line: str, style: str = "") -> None:
        for part in wrap_line(line, con.width):
            con.print(Text(part, style=style), overflow="ellipsis", no_wrap=True)

    if system is not None:
        say(pc_line(system))
        line = no_gpu_line(system, hidden, fits)
        if line:
            say(line)
    if matched != len(fits):
        count = tr(f"{matched} 件中 {len(fits)} 件", f"{len(fits)} of {matched}")
    else:
        count = tr(f"{len(fits)} 件", str(len(fits)))
    head = f"{title}: {count}" if title else count
    say(tr("", " ").join(x for x in (head, hidden_text(hidden)) if x), "bold")
    if groups is not None:
        for g in groups:
            mod = g["modality"]
            _en, ja = MODALITY_LABEL.get(mod, (mod, mod))
            # 機械の名前(-m に書く名前)も見せる。英語は種類の名前だけ(1 行に 2 つの言語を並べない。#73)
            head = tr(f"{ja} ({mod})", mod)
            rows = [f for f in fits if f.model.modality == mod]
            if not g["matched"]:
                say(
                    tr(
                        f"== {head}: この条件でこの PC で動くモデルはありません",
                        f"== {head}: no model runs on this PC with these filters",
                    ),
                    "bold",
                )
                continue
            say(
                tr(
                    f"== {head}: {g['matched']} 件中 上位 {g['shown']} 件(続きは -m {mod})",
                    f"== {head}: top {g['shown']} of {g['matched']}  (-m {mod} for more)",
                ),
                "bold",
            )
            con.print(build_table(rows, width=con.width, system=system, **kw))
    elif not fits:
        say(tr("(該当なし)", "(no match)"))
    else:
        con.print(build_table(fits, width=con.width, system=system, **kw))
    for n in [
        *(legend(fits, hidden, grouped, order, kw["with_content"], system, json_first) if fits else []),
        *notes,
    ]:
        say(n, "dim")
