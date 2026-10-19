"""端末向けの表(Issue #7)。llmfit の `fit` の表にならう。

- 幅 100 文字の端末で崩れない(モデル名は省略記号で詰める。80 未満では列を減らす)
- 段階は色分けするが、色の無い端末でも文字で分かる(NO_COLOR やパイプ出力では色を付けない)
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
MODALITY_LABEL = {
    "image": "image",
    "video": "video",
    "music": "music",
    "sfx": "sfx",
    "tts": "tts",
    "3d": "3d",
}
MAX_WIDTH = 100


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
    return Text("held:" + (failed[0] if failed else "?"), style="yellow")


def _need(e, vram: float) -> tuple[float | None, str]:
    """判定の根拠にした VRAM と印。* = メインメモリへ逃がす、~ = 部品を順に載せ替える。"""
    if e.run_mode == "cpu":
        return None, "cpu"
    if e.run_mode == "offload" or e.fit == "too_tight":  # 動かないときは「最低これだけ要る」を出す
        return e.vram_offload_gb, "*"
    seq = getattr(e, "vram_sequential_gb", None)
    if e.run_mode == "gpu" and vram and e.vram_gb > vram and seq is not None and seq <= vram:
        return seq, "~"
    return e.vram_gb, ""


def _need_cell(e, vram: float) -> str:
    need, mark = _need(e, vram)
    return mark if need is None else f"{need:.1f}{mark}"


def _headroom_cell(e, vram: float) -> Text:
    if vram <= 0:
        return Text("-", style="dim")
    need, _ = _need(e, vram)
    if need is None:
        return Text("-", style="dim")
    diff = vram - need
    style = "green" if diff >= vram * 0.1 else ("yellow" if diff >= 0 else "red")
    return Text(f"{diff:+.1f}", style=style)


def build_table(fits: Sequence, vram_gb: float, width: int = MAX_WIDTH, title: str | None = None) -> Table:
    compact = width < 80
    t = Table(
        title=title,
        title_justify="left",
        header_style="bold",
        expand=False,
        pad_edge=False,
        box=None,
        show_edge=False,
    )
    t.add_column("Fit", no_wrap=True, min_width=9)
    # 他の列の幅を引いた残りをモデル名に使う(長い名前は省略記号で詰める)
    model_max = max(16, width - (40 if compact else 64))
    t.add_column("Model", no_wrap=True, overflow="ellipsis", max_width=model_max)
    t.add_column("Type", no_wrap=True)
    t.add_column("VRAM", justify="right", no_wrap=True)
    t.add_column("Room", justify="right", no_wrap=True)
    t.add_column("Basis", no_wrap=True)
    if not compact:
        t.add_column("License", no_wrap=True, overflow="ellipsis", max_width=12)
        t.add_column("Auto", no_wrap=True, overflow="ellipsis", max_width=16)
    for f in fits:
        e, m = f.estimate, f.model
        name = Text(m.id)
        if m.content == "uncensored":
            name = Text.assemble(("[U] ", "magenta"), m.id)
        row = [
            Text(LEVEL_LABEL.get(e.fit, e.fit), style=LEVEL_STYLE.get(e.fit, "")),
            name,
            MODALITY_LABEL.get(m.modality, m.modality),
            _need_cell(e, vram_gb),
            _headroom_cell(e, vram_gb),
            Text("measured", style="bold") if e.basis == "measured" else Text("est.", style="dim"),
        ]
        if not compact:
            row += [m.license or "?", _auto_cell(f.gate)]
        t.add_row(*row)
    return t


def print_fit_table(fits, title: str = "", system=None, console: Console | None = None) -> None:
    con = console or make_console()
    vram = system.vram_gb if system is not None else 0.0
    head = f"{title} ({len(fits)})" if title else None
    if not fits:
        con.print(f"{head}: (該当なし)" if head else "(該当なし)")
        return
    con.print(build_table(fits, vram, width=con.width, title=head))
    notes = [
        "VRAM = GB needed (~ = parts loaded one at a time, * = offload to RAM)",
        "Room = this PC's VRAM minus need",
    ]
    if any(f.model.content == "uncensored" for f in fits):
        notes.append("[U] = uncensored")
    for n in notes:
        con.print(Text(n, style="dim"))
    if system is not None:
        g = system.best_gpu
        gpu = f"{g.name} {g.vram_total_gb:g} GB" if g else "no GPU (CPU only)"
        con.print(Text(f"PC: {gpu}, RAM {system.ram_total_gb:.0f} GB", style="dim"))
