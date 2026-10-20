"""端末向けの表(Issue #7)。llmfit の `fit` の表にならう。

- 幅 100 文字の端末で崩れない。モデル名は残りの幅を全部使い、長い名前だけ省略記号で詰める
  (88 未満では License 列、70 未満では Auto 列も外す)
- 段階は色分けするが、色の無い端末でも文字で分かる(NO_COLOR やパイプ出力では色を付けない)
- VRAM は判定に使った値を出す(~ = 部品を順に載せ替える、* = メインメモリへ逃がす、
  RAM = メインメモリが足りないのが理由、GPU = GPU が要るのに無い、? = 大きさ不明、cpu = CPU で動く)
- 自動でおすすめに出せないものは「held:<落ちた条件>」と短く出す
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.text import Text

LEVEL_STYLE = {"perfect": "bold green", "good": "cyan", "marginal": "yellow", "too_tight": "red"}
LEVEL_LABEL = {"perfect": "Perfect", "good": "Good", "marginal": "Marginal", "too_tight": "Too tight"}
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
    "files_complete": "size",
}
MAX_WIDTH = 100
FULL_WIDTH = 88  # これ未満では License 列を外す
MIN_AUTO_WIDTH = 70  # これ未満では Auto 列も外す


def make_console(file=None, width: int | None = None) -> Console:
    out = file or sys.stdout
    if width is None:
        try:
            width = os.get_terminal_size(out.fileno()).columns
        except (OSError, AttributeError, ValueError):
            width = MAX_WIDTH
    return Console(file=out, width=min(width, MAX_WIDTH), soft_wrap=False, highlight=False)


def _auto_cell(gate: dict[str, Any] | None) -> Text:
    if not gate:
        return Text("-", style="dim")
    if gate.get("auto"):
        return Text("yes", style="green")
    failed = [c.get("id", "?") for c in gate.get("checks", []) if not c.get("passed")]
    if not failed:
        return Text("held", style="yellow")
    return Text("held:" + CHECK_LABEL.get(failed[0], failed[0]), style="yellow")


def _ram_short(e, system) -> bool:
    return bool(system is not None and system.ram_total_gb and e.ram_gb > system.ram_total_gb)


def _need(e, vram: float, system=None) -> tuple[float | None, str]:
    """判定の根拠にした VRAM と印。"""
    if getattr(e, "unknown_size", False):
        return None, "?"
    if e.run_mode == "cpu":
        return None, "cpu"
    if e.fit == "too_tight" and vram <= 0:
        return None, "GPU"
    seq = getattr(e, "vram_sequential_gb", None)
    if e.fit == "too_tight":
        # VRAM は足りていてメモリが理由なら、VRAM の側は判定に使った載せ方の値を出す
        if _ram_short(e, system) and e.vram_offload_gb <= vram:
            if e.vram_gb <= vram:
                return e.vram_gb, ""
            if seq is not None and seq <= vram:
                return seq, "~"
            return e.vram_offload_gb, "*"
        return e.vram_offload_gb, "*"  # 動かないときは「最低これだけ要る」を出す
    if e.run_mode == "offload":
        return e.vram_offload_gb, "*"
    if e.run_mode == "gpu" and vram and e.vram_gb > vram and seq is not None and seq <= vram:
        return seq, "~"
    return e.vram_gb, ""


def _need_cell(e, vram: float, system=None) -> str:
    need, mark = _need(e, vram, system)
    return mark if need is None else f"{round(need, 1):.1f}{mark}"


def _room_cell(e, vram: float, system=None) -> Text:
    if vram <= 0:
        return Text("-", style="dim")
    need, _ = _need(e, vram, system)
    if need is None:
        return Text("-", style="dim")
    if _ram_short(e, system):  # メモリが理由なら VRAM の余裕を出さない(足りているように見えるので)
        return Text("RAM", style="red" if e.fit == "too_tight" else "yellow")
    diff = round(round(vram, 1) - round(need, 1), 1)  # 表に出した値どうしで引く
    if diff == 0:
        diff = 0.0  # -0.0 を出さない
    if e.fit == "too_tight":
        style = "red"
    elif e.fit == "marginal":
        style = "yellow"
    else:
        style = "green" if diff >= vram * 0.1 else "yellow"
    return Text(f"{diff:+.1f}", style=style)


COLUMN_WIDTH = {"Fit": 9, "Type": 5, "VRAM": 6, "Room": 5, "Basis": 5, "License": 10, "Auto": 13}


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


def model_width(width: int, show_license: bool, show_auto: bool) -> int:
    cols = (
        ["Fit", "Type", "VRAM", "Room", "Basis"]
        + (["License"] if show_license else [])
        + (["Auto"] if show_auto else [])
    )
    used = sum(COLUMN_WIDTH[c] for c in cols) + 2 * len(cols)  # 列の間は 2 文字
    return max(12, width - used)


def build_table(
    fits: Sequence, vram_gb: float, width: int = MAX_WIDTH, title: str | None = None, system=None
) -> Table:
    show_license = width >= FULL_WIDTH
    show_auto = width >= MIN_AUTO_WIDTH
    name_w = model_width(width, show_license, show_auto)
    t = Table(
        title=title,
        title_justify="left",
        header_style="bold",
        expand=False,
        pad_edge=False,
        box=None,
        show_edge=False,
    )
    t.add_column("Fit", no_wrap=True, width=COLUMN_WIDTH["Fit"])
    # モデル名は他の列の残りを全部使う。長い名前は真ん中を省略記号にする(middle_ellipsis)
    t.add_column("Model", no_wrap=True, overflow="ellipsis", width=name_w)
    t.add_column("Type", no_wrap=True, width=COLUMN_WIDTH["Type"])
    t.add_column("VRAM", justify="right", no_wrap=True, width=COLUMN_WIDTH["VRAM"])
    t.add_column("Room", justify="right", no_wrap=True, width=COLUMN_WIDTH["Room"])
    t.add_column("Basis", no_wrap=True, width=COLUMN_WIDTH["Basis"])
    if show_license:
        t.add_column("License", no_wrap=True, overflow="ellipsis", width=COLUMN_WIDTH["License"])
    if show_auto:
        t.add_column("Auto", no_wrap=True, overflow="ellipsis", width=COLUMN_WIDTH["Auto"])
    shown = distinct_names([f.model.id for f in fits if f.model.content != "uncensored"], name_w)
    shown_u = distinct_names([f.model.id for f in fits if f.model.content == "uncensored"], name_w - 4)
    for f in fits:
        e, m = f.estimate, f.model
        if m.content == "uncensored":
            name = Text.assemble(("[U] ", "magenta"), shown_u[m.id])
        else:
            name = Text(shown[m.id])
        row = [
            Text(LEVEL_LABEL.get(e.fit, e.fit), style=LEVEL_STYLE.get(e.fit, "")),
            name,
            m.modality,
            _need_cell(e, vram_gb, system),
            _room_cell(e, vram_gb, system),
            Text("meas.", style="bold") if e.basis == "measured" else Text("est.", style="dim"),
        ]
        if show_license:
            row.append(m.license or "?")
        if show_auto:
            row.append(_auto_cell(f.gate))
        t.add_row(*row)
    return t


def print_fit_table(fits, title: str = "", system=None, console: Console | None = None) -> None:
    con = console or make_console()
    vram = system.vram_gb if system is not None else 0.0
    head = f"{title} ({len(fits)})" if title else None
    if not fits:
        con.print(f"{head}: (該当なし)" if head else "(該当なし)")
        return
    con.print(build_table(fits, vram, width=con.width, title=head, system=system))
    notes = [
        "VRAM: GiB needed. ~ parts one at a time, * offload to RAM, ? size unknown. Basis: meas. = measured",
        "Room: this PC's VRAM minus need (GiB). RAM = not enough system RAM",
    ]
    if any(f.model.content == "uncensored" for f in fits):
        notes.append("[U] = uncensored")
    for n in notes:
        con.print(Text(n, style="dim"), overflow="ellipsis", no_wrap=True)
    if system is not None:
        g = system.best_gpu
        gpu = f"{g.name} {g.vram_total_gb:.1f} GiB" if g else "no GPU (CPU only)"
        if "vram" in system.overrides:
            gpu = f"assumed {system.vram_gb:.1f} GiB VRAM (--vram)"
        ram = f"RAM {system.ram_total_gb:.1f} GiB" + (" (--ram)" if "ram" in system.overrides else "")
        con.print(Text(f"PC: {gpu}, {ram}", style="dim"), overflow="ellipsis", no_wrap=True)
