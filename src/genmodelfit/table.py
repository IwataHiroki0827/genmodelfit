# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""端末向けの表(Issue #7、#17)。llmfit の `fit` の表にならう。

- 先頭に PC と「何件中何件を出したか・隠した件数と理由」、末尾に印の説明
- 幅 100 文字の端末で崩れない。モデル名は残りの幅を全部使い、長い名前は真ん中を省略記号で詰める
  (狭い端末では License → Disk → Auto → Mem% → Type → Score の順に列を外す)
- 段階は色と「●」で分ける。色の無い端末でも文字で分かる(NO_COLOR やパイプ出力では色を付けない)。
  絵文字は Windows の端末(cp932)で幅が崩れるので使わない
- VRAM は判定に使った値(m = 実測)。Mode は GPU / GPU~(部品を順に載せ替え)/ GPU+RAM(逃がす)/ CPU / -(動かない)
- Auto は自動でおすすめに出せれば yes、出せなければ最初に落ちた条件の短い名前
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .groups import precision_label
from .recommend import FIT_LABEL, _ram_short, memory_use, mode_label, vram_need
from .tasks import MODALITY_LABEL

LEVEL_STYLE = {
    "perfect": "bold green",
    "good": "cyan",
    "marginal": "yellow",
    "too_tight": "red",
    "unknown": "dim",
}
LEVEL_LABEL = FIT_LABEL
MARK = "●"  # 段階の印(絵文字ではない記号)
# 落ちた条件の id(gate.py)を、表に収まる短い名前にする
CHECK_LABEL = {
    "license": "license",
    "gated": "gated",
    "official_workflow": "workflow",
    "custom_nodes": "nodes",
    "comfyui_version": "comfyui",
    "url_reachable": "url",
    "sha256": "sha256",
    "content": "content",
    "size_known": "size",
}
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
    "held": "held",
    "task": "other uses",
    "same_model": "merged into their models",
}
MAX_WIDTH = 100
MIN_NAME_WIDTH = 24  # モデル名がこれより狭くなるなら列を外す
# 列の幅(並びもこの順)と、狭いときに外す順。Type は種類が混ざる表(fit・search など)だけ
COLUMN_WIDTH = {
    "Status": 11,
    "Type": 5,
    "Use": 7,
    "Prec": 4,
    "Score": 5,
    "VRAM": 5,
    "Mem%": 4,
    "Mode": 7,
    "Disk": 4,
    "License": 10,
    "Auto": 8,
}
DROP_ORDER = ("Disk", "Mem%", "License", "Auto", "Prec", "Type", "Use", "Score")


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


def wrap_line(text: str, width: int) -> list[str]:
    """1 行を幅に収まるように折り返す(全角は 2 桁)。続きの行は元の字下げ + 4 桁。"""
    if cell_len(text) <= width:
        return [text]
    lead = len(text) - len(text.lstrip(" "))
    pad = " " * min(lead + 4, max(0, width // 2))
    out: list[str] = []
    rest = text
    while rest:
        prefix = pad if out else ""
        room = width - cell_len(prefix)
        n, used = 0, 0
        while n < len(rest) and used + cell_len(rest[n]) <= room:
            used += cell_len(rest[n])
            n += 1
        if n < len(rest):
            cut = rest.rfind(" ", 0, n)  # 英数字の語の途中で切らない(空白があれば)
            if cut > len(prefix.strip()) and cut > n // 2:
                n = cut + 1
        n = max(n, 1)
        out.append(prefix + rest[:n].rstrip(" "))
        rest = rest[n:].lstrip(" ")
    return out


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


def columns_for(width: int, with_type: bool = True) -> list[str]:
    """幅に収まる列(Model 以外。並びは COLUMN_WIDTH の順)。Type は種類が混ざる表だけ。"""
    cols = [c for c in COLUMN_WIDTH if with_type or c != "Type"]
    for drop in DROP_ORDER:
        if model_width(width, cols) >= MIN_NAME_WIDTH:
            break
        if drop in cols:
            cols.remove(drop)
    return cols


def _auto_cell(gate: dict[str, Any] | None) -> Text:
    if not gate:
        return Text("-", style="dim")
    if gate.get("auto"):
        return Text("yes", style="green")
    failed = [c.get("id", "?") for c in gate.get("checks", []) if not c.get("passed")]
    if not failed:
        return Text("held", style="yellow")
    return Text(CHECK_LABEL.get(failed[0], failed[0]), style="yellow")


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
        return Text.assemble(_gb(need), ("m", "bold"))
    return Text(_gb(need))


def _mem_cell(e, system) -> Text:
    if _ram_short(e, system):  # メモリが理由(VRAM の割合だけだと足りているように見えるので)
        return Text("RAM", style="red" if e.fit == "too_tight" else "yellow")
    _, need, avail = memory_use(e, system)
    if need is None or not avail:
        return Text("-", style="dim")
    pct = 100 * need / avail
    # 足りないのに 100% と出さない。載るのに 101% とも出さない
    shown = max(round(pct), 101) if need > avail else min(round(pct), 100)
    style = "red" if pct > 100 else ("yellow" if pct > 90 else "")
    return Text(f"{shown}%" if shown <= 999 else ">999", style=style)


def _use_cell(f, grouped: bool) -> Text:
    use = getattr(f, "use", "") or "-"
    n = len(getattr(f, "workflows", []) or []) if grouped else 0
    return Text.assemble(use, (f"+{n}", "dim") if n else "")


def build_table(
    fits: Sequence, width: int = MAX_WIDTH, system=None, with_type: bool | None = None, grouped: bool = True
) -> Table:
    if with_type is None:
        with_type = len({f.model.modality for f in fits}) > 1
    cols = columns_for(width, with_type)
    name_w = model_width(width, cols)
    t = Table(
        header_style="bold",
        expand=False,
        pad_edge=False,
        box=None,
        show_edge=False,
        padding=(0, 1, 0, 0),
    )
    right = {"Score", "VRAM", "Mem%", "Disk"}
    t.add_column("Status", no_wrap=True, width=COLUMN_WIDTH["Status"])
    # モデル名は他の列の残りを全部使う。長い名前は真ん中を省略記号にする(middle_ellipsis)
    t.add_column("Model", no_wrap=True, overflow="ellipsis", width=name_w)
    for c in cols[1:]:
        t.add_column(
            c,
            justify="right" if c in right else "left",
            no_wrap=True,
            overflow="ellipsis",
            width=COLUMN_WIDTH[c],
        )

    # まとめた表はモデルの表示名(一覧の中で重ならない)、--all-workflows はワークフローの ID
    def label(f) -> str:
        return (getattr(f, "model_name", "") or f.model.id) if grouped else f.model.id

    shown = distinct_names([label(f) for f in fits if f.model.content != "uncensored"], name_w)
    shown_u = distinct_names([label(f) for f in fits if f.model.content == "uncensored"], name_w - 4)
    for f in fits:
        e, m = f.estimate, f.model
        if m.content == "uncensored":
            name = Text.assemble(("[U] ", "magenta"), shown_u[label(f)])
        else:
            name = Text(shown[label(f)])
        cells = {
            "Type": Text(m.modality),
            "Use": _use_cell(f, grouped),
            "Prec": Text(precision_label(m.precision)),
            "Score": Text(f"{f.score:.0f}"),
            "VRAM": _vram_cell(e, system),
            "Mem%": _mem_cell(e, system),
            "Mode": Text(mode_label(e, system), style="red" if e.fit == "too_tight" else ""),
            "Disk": Text(_gb(e.disk_gb) if e.disk_gb else "?"),
            "License": Text(m.license or "?"),
            "Auto": _auto_cell(f.gate),
        }
        status = Text(f"{MARK} {LEVEL_LABEL.get(e.fit, e.fit)}", style=LEVEL_STYLE.get(e.fit, ""))
        t.add_row(status, name, *(cells[c] for c in cols[1:]))
    return t


def pc_line(system) -> str:
    """この PC の 1 行。調べた値なら空きも出す。--vram / --ram で上書きしたら assumed。"""
    g = system.best_gpu
    if g:
        free = f" ({g.vram_free_gb:.1f} free)" if g.vram_free_gb is not None else ""
        gpu = f"{g.name} {g.vram_total_gb:.1f} GiB VRAM{free}"
    elif system.gpus:
        gpu = f"no usable GPU ({system.gpus[0].name}: see `genmodelfit system`) (CPU only)"
    else:
        gpu = "no GPU (CPU only)"
    if "vram" in system.overrides:
        gpu = (
            f"assumed {system.vram_gb:.1f} GiB VRAM (--vram)"
            if system.vram_gb > 0
            else "assumed no GPU (--vram 0)"
        )
    if "ram" in system.overrides:
        ram = f"RAM {system.ram_total_gb:.1f} GiB (--ram)"
    else:
        ram = f"RAM {system.ram_total_gb:.1f} GiB ({system.ram_available_gb:.1f} available)"
    return f"PC: {gpu}, {ram}"


def hidden_text(hidden: Mapping[str, int]) -> str:
    """「(N hidden: uncensored 39, tools 20; 24 more workflows merged into their models)」。
    何も無ければ空。"""
    merged = hidden.get("same_model", 0)
    rest = {k: v for k, v in hidden.items() if v and k != "same_model"}
    parts = []
    if rest:
        parts.append(
            f"{sum(rest.values())} hidden: "
            + ", ".join(f"{HIDDEN_LABEL.get(k, k)} {v}" for k, v in rest.items())
        )
    if merged:
        parts.append(f"{merged} more workflows merged into their models")
    return f"({'; '.join(parts)})" if parts else ""


def legend(fits: Sequence, hidden: Mapping[str, int], grouped: bool = True) -> list[str]:
    out = [  # 1 項目 1 行(幅 100 に収める。狭い端末では字下げして折り返す)
        "Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.",
        "Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.",
        "Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.",
    ]
    if grouped:
        out.append(
            "One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows)."
        )
    out.append(
        "Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md)."
    )
    if any(f.model.content == "uncensored" for f in fits):
        out.append("[U] = uncensored")
    if hidden.get("uncensored"):
        out.append(
            f"Note: {hidden['uncensored']} uncensored model(s) hidden; add --content all to show them."
        )
    if hidden.get("license_unknown"):
        out.append(
            f"Note: {hidden['license_unknown']} model(s) with an unknown license hidden;"
            " add --include-unknown-license to show them."
        )
    if any(f.estimate.unknown_size for f in fits):
        out.append("? = size unknown, not judged (listed last).")
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
) -> None:
    """PC の行 → 見出し(何件中何件・隠した理由)→ 表 → 説明。notes は末尾に足す一行の注意。

    groups を渡すと種類ごとの節に分ける(見出し 1 行 + 表。動くものが無い種類は 1 行で知らせる)。"""
    con = console or make_console()
    hidden = dict(hidden or {})
    matched = len(fits) if matched is None else matched

    def say(line: str, style: str = "") -> None:
        for part in wrap_line(line, con.width):
            con.print(Text(part, style=style), overflow="ellipsis", no_wrap=True)

    if system is not None:
        say(pc_line(system))
    count = f"{len(fits)} of {matched}" if matched != len(fits) else str(len(fits))
    head = f"{title}: {count}" if title else count
    say(" ".join(x for x in (head, hidden_text(hidden)) if x), "bold")
    if groups is not None:
        for g in groups:
            mod = g["modality"]
            _en, ja = MODALITY_LABEL.get(mod, (mod, mod))
            rows = [f for f in fits if f.model.modality == mod]
            if not g["matched"]:
                say(f"== {mod} ({ja}): no model runs on this PC with these filters", "bold")
                continue
            say(f"== {mod} ({ja}): top {g['shown']} of {g['matched']}  (-m {mod} for more)", "bold")
            con.print(build_table(rows, width=con.width, system=system, with_type=False, grouped=grouped))
    elif not fits:
        say("(no match / 該当なし)")
    else:
        con.print(build_table(fits, width=con.width, system=system, grouped=grouped))
    for n in [*(legend(fits, hidden, grouped) if fits else []), *notes]:
        say(n, "dim")
