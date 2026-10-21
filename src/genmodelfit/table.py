# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""端末向けの表(Issue #7、#17)。llmfit の `fit` の表にならう。

- 先頭に PC と「何件中何件を出したか・隠した件数と理由」、末尾に印の説明
- 幅 100 文字の端末で崩れない。モデル名は残りの幅を全部使い、長い名前は真ん中を省略記号で詰める
  (狭い端末では Disk → Mem% → License → Verified → Prec → Use の順に列を外す)
- 段階は色と「●」で分ける。色の無い端末でも文字で分かる(NO_COLOR やパイプ出力では色を付けない)。
  絵文字は Windows の端末(cp932)で幅が崩れるので使わない
- VRAM は判定に使った値(m = 実測)。Mode は GPU / GPU~(部品を順に載せ替え)/ GPU+RAM(逃がす)/ CPU / -(動かない)
- Verified(0.5.0 までは Auto。JSON の auto)は自動でおすすめに出せれば yes、出せなければ
  最初に落ちた条件の短い名前
- 「?」は分からない(精度・ライセンス・大きさ = 判定していない)、「-」は当てはまらない
  (動かないので Mode が無い、CPU で動くので VRAM が無い、GPU が無いので Mem% が無い、など)。
  #30 で使い分けをそろえた
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
    "app_runtime": "runtime",
    "reviewed": "review",
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
    "other_fit": "other levels (--fit)",
    "held": "held",
    "task": "other uses",
    "same_model": "merged into their models",
}
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
    "Use": 7,
    "Prec": 4,
    "VRAM": 5,
    "Mem%": 4,
    "Mode": 7,
    "Disk": 4,
    "License": 10,
    "Verified": 8,  # 0.6.0 で Auto から名前を変えた(JSON の auto・held_by はそのまま)
}
DROP_ORDER = ("Disk", "Mem%", "License", "Verified", "Prec", "Use")


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
    if e.unknown_size:
        return Text("?")
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


def _use_cell(f, grouped: bool) -> Text:
    use = getattr(f, "use", "") or "?"  # 用途が分からない
    n = len(getattr(f, "workflows", []) or []) if grouped else 0
    return Text.assemble(use, (f"+{n}", "dim") if n else "")


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
            "Verified": _auto_cell(f.gate),
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
            out.append(f"{g.name} (not usable for generation → judged as CPU)")
    for a in getattr(system, "other_gpus", []) or []:
        if (
            a.name in seen or a.vendor == "unknown"
        ):  # 会社の分からない画面(Microsoft の基本の画面など)は出さない
            continue
        seen.add(a.name)
        kind = "integrated; " if a.integrated else ""
        out.append(f"{a.name} ({kind}not usable for generation → judged as CPU)")
    return out


def pc_line(system) -> str:
    """この PC の 1 行。調べた値なら空きも出す。--vram / --ram で上書きしたら assumed。
    使える GPU が無ければ、見つけた GPU とその扱い(CPU として判定した)を出す(0.7.0、#22)。"""
    g = system.best_gpu
    if g and g.vendor != "nvidia":
        backend = BACKEND_LABEL.get(g.backend, g.backend)
        memory = getattr(g, "memory", "dedicated")
        if memory == "unified":
            gpu = f"{g.name} (unified memory; {g.vram_total_gb:.1f} GiB for the GPU, {backend}, estimated)"
        elif memory == "shared":
            gpu = f"{g.name} (integrated; {g.vram_total_gb:.1f} GiB shared with RAM, {backend}, estimated)"
        else:
            gpu = f"{g.name} {g.vram_total_gb:.1f} GiB VRAM ({backend}, estimated)"
    elif g:
        free = f" ({g.vram_free_gb:.1f} free)" if g.vram_free_gb is not None else ""
        gpu = f"{g.name} {g.vram_total_gb:.1f} GiB VRAM{free}"
    else:
        found = _unusable_gpus(system)
        gpu = "; ".join(found) if found else "no GPU (CPU only)"
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


def no_gpu_line(system, hidden: Mapping[str, int], fits: Sequence = ()) -> str:
    """使える GPU が無いときの 1 行(0.7.0、#22)。GPU があれば空。"""
    if system is None or system.vram_gb > 0:
        return ""
    tools = hidden.get("tools", 0)
    tail = f" ({tools} tools hidden: --include-tools)" if tools else ""
    if any(f.estimate.fit == "too_tight" for f in fits):
        return f"This PC has no usable GPU: models marked GPU req need one{tail}"
    return f"This PC has no usable GPU: only models that run on CPU are shown{tail}"


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


# 表の下の「Order:」の説明(並べ方ごと。recommend は auto)
ORDER_TEXT = {
    "auto": "Order: runs, Verified=yes, Perfect/Good, Marginal, Score.",
    "type": "Order: Type, Status, Score (--sort).",
    "score": "Order: Status, Score (all types).",
    "vram": "Order: runs, VRAM (least first).",
    "disk": "Order: runs, Disk (smallest first).",
    "name": "Order: runs, Model name.",
}


def legend(
    fits: Sequence,
    hidden: Mapping[str, int],
    grouped: bool = True,
    order: str = "auto",
    with_content: bool | None = None,
    system=None,
) -> list[str]:
    system_marks = any(mode_cell(f.estimate, system) in ("GPU req", "RAM") for f in fits)
    best = system.best_gpu if system is not None else None
    non_nvidia = best is not None and best.vendor != "nvidia"
    out = [  # 1 項目 1 行(幅 100 に収める。狭い端末では字下げして折り返す)
        f"{ORDER_TEXT.get(order, ORDER_TEXT['auto'])} Score 0-100. VRAM: GiB needed, m = measured.",
        "Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.",
        *(
            ["Mode for no run: GPU req = needs a GPU (none usable here), RAM = not enough system RAM."]
            if system_marks
            else []
        ),
        "Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).",
        *(
            ["Level caps: AMD / Intel / Apple GPUs are estimates (no measurements), at most Good."]
            if non_nvidia
            else []
        ),
        "Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Verified: yes, or first failed check.",
    ]
    if grouped:
        out.append(
            "One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows)."
        )
    out.append(
        "Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md)."
    )
    shown = content_column_wanted(fits) if with_content is None else with_content
    if shown:
        out.append("Censored: N = safety filter removed / adult model")
    if hidden.get("uncensored"):
        out.append(
            f"Note: {hidden['uncensored']} uncensored model(s) hidden; add --content all to show them."
        )
    if order != "auto" and any(hidden.get(k) for k in ("tools", "too_tight", "size_unknown", "other_fit")):
        out.append("Note: add --all to also show tools, models that do not run and models not judged (?).")
    if hidden.get("license_unknown"):
        out.append(
            f"Note: {hidden['license_unknown']} model(s) with an unknown license hidden;"
            " add --include-unknown-license to show them."
        )
    out.append("? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).")
    if any(f.estimate.unknown_size for f in fits):
        out.append("Status ? = size unknown, not judged (after the levels that run, before Too tight).")
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
            con.print(build_table(rows, width=con.width, system=system, **kw))
    elif not fits:
        say("(no match / 該当なし)")
    else:
        con.print(build_table(fits, width=con.width, system=system, **kw))
    for n in [*(legend(fits, hidden, grouped, order, kw["with_content"], system) if fits else []), *notes]:
        say(n, "dim")
