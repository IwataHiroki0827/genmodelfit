# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NVIDIA 以外の GPU の使用率・使用中のメモリを各社の道具で読む
(bench の「GPU が空いているか」と「GPU 全体の増分」。NVIDIA は bench.py が nvidia-smi で読む)。

標準ライブラリと subprocess だけで動かす。道具の出力を読む関数は純粋な関数にして、公式の文書・ソースの出力例で
試験する(tests/test_gpustat.py)。道具(上から順に試す)と、使う値:

- AMD・Linux: `amd-smi metric --usage --mem-usage --json`(usage.gfx_activity %・mem_usage.used_vram MB)
- AMD・Linux: `rocm-smi --showuse --showmeminfo vram --json`
  (「GPU use (%)」・「VRAM Total Used Memory (B)」)
- Intel・Linux: `xpu-smi dump -d -1 -m 0,18 -n 1`(-1 = 全部の GPU。項目 0 = 全エンジンの平均 %・
  項目 18 = 使用中 MiB。GPU が 2 枚以上で ComfyUI の GPU を決められなければ、読めないものとして扱う)
- AMD・Intel・Windows: `typeperf` の性能カウンタ GPU Engine・GPU Adapter Memory
  (エンジンの種類ごとの合計のうち最大 = タスク マネージャーと同じ・Dedicated Usage バイト)
- Apple・macOS: `ioreg -r -d 1 -w 0 -c IOAccelerator`
  (「Device Utilization %」・「In use system memory」バイト)

出典(2026-10-22 に読んだ):
- amd-smi: ROCm/amdsmi(amd-staging)の amdsmi_cli/amdsmi_commands.py。ROCm 7 から `{"gpu_data": [...]}`、
  6.x は配列。値は `{"value", "unit"}` か "N/A"
- rocm-smi: ROCm/rocm_smi_lib の python_smi_tools/rocm_smi.py(showGpuUse・showMemInfo の項目名)
- xpu-smi: intel/xpumanager の doc/smi_user_guide.md(項目の番号と dump の CSV)
- 性能カウンタ: Microsoft の DirectX blog「GPUs in the Task Manager」
  (https://devblogs.microsoft.com/directx/gpus-in-the-task-manager/ 。いちばん忙しいエンジンの使用率を
  GPU の使用率にする。WDDM 2.0 以上のドライバ)
- ioreg: exelban/stats の Modules/GPU/reader.swift(「Device Utilization %」、無ければ「GPU Activity(%)」)。
  powermetrics は sudo が要るので使わない

個人の情報: 性能カウンタの名前には PC 名・プロセス ID・アダプタの LUID が入り、xpu-smi の行には時刻が入る。
どれも数を集めるのに使うだけで、残さない(返すのは数だけ)。
"""

from __future__ import annotations

import csv
import io
import json
import platform
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from . import hardware

MIB = 1024**2
TOOL_TIMEOUT = 8.0  # 1 回の読み取りの時間制限(秒)。typeperf は 1 秒の標本を取るので約 2〜3 秒かかる
# ComfyUI の VRAM の総量と道具の総量がこの割合の内なら、同じ GPU とみなす
TOTAL_MATCH = 0.10


@dataclass
class Reading:
    """GPU 1 枚の 1 回の読み取り。分からない値は None。"""

    util_pct: float | None = None
    used_mib: float | None = None
    total_mib: float | None = None


def _num(x: Any) -> float | None:
    """数・"12"・"12 %"・{"value": 12, "unit": "%"} を数にする。"N/A" などは None。"""
    if isinstance(x, dict):
        x = x.get("value")
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        m = re.fullmatch(r"\s*(-?\d+(?:\.\d+)?)\s*%?\s*", x)
        return float(m.group(1)) if m else None
    return None


def _keep(r: Reading) -> bool:
    return r.util_pct is not None or r.used_mib is not None


# --- 各社の道具の出力を読む(純粋な関数) -----------------------------------------------------------


def parse_amd_smi(text: str) -> list[Reading]:
    """`amd-smi metric --usage --mem-usage --json`。ROCm 6.x は配列、7.x は {"gpu_data": [...]}。"""
    try:
        data = json.loads(text.strip() or "null")
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("gpu_data")
    out = []
    for d in data if isinstance(data, list) else []:
        if not isinstance(d, dict):
            continue
        usage = d.get("usage") if isinstance(d.get("usage"), dict) else {}
        mem = d.get("mem_usage") if isinstance(d.get("mem_usage"), dict) else {}
        r = Reading(_num(usage.get("gfx_activity")), _num(mem.get("used_vram")), _num(mem.get("total_vram")))
        if _keep(r):
            out.append(r)
    return out


def parse_rocm_smi(text: str) -> list[Reading]:
    """`rocm-smi --showuse --showmeminfo vram --json`。{"card0": {"GPU use (%)": "0", ... (B): "バイト"}}。"""
    try:
        data = json.loads(text.strip() or "null")
    except ValueError:
        return []
    out = []
    if not isinstance(data, dict):
        return out
    for key in sorted(k for k in data if re.fullmatch(r"card\d+", str(k))):
        d = data[key]
        if not isinstance(d, dict):
            continue
        used = _num(d.get("VRAM Total Used Memory (B)"))
        total = _num(d.get("VRAM Total Memory (B)"))
        r = Reading(
            _num(d.get("GPU use (%)")),
            used / MIB if used is not None else None,
            total / MIB if total is not None else None,
        )
        if _keep(r):
            out.append(r)
    return out


def parse_xpu_smi_dump(text: str) -> list[Reading]:
    """xpu-smi dump(`-d -1 -m 0,18 -n 1`)の CSV(「Timestamp, DeviceId,
    Average % utilization of all GPU Engines, GPU Memory Used (MiB)」)。
    列は見出しの文字で探す。時刻は読まない。GPU(DeviceId)ごとに最後の行を 1 つ返す。"""
    lines = [x for x in text.splitlines() if x.strip()]
    head = next((i for i, x in enumerate(lines) if x.lstrip().lower().startswith("timestamp")), None)
    if head is None:
        return []
    cols = [c.strip().lower() for c in lines[head].split(",")]
    ui = next((i for i, c in enumerate(cols) if "utilization" in c and "memory" not in c), None)
    mi = next((i for i, c in enumerate(cols) if "memory used" in c), None)
    di = next((i for i, c in enumerate(cols) if c.replace(" ", "") == "deviceid"), None)
    if ui is None and mi is None:
        return []
    by_dev: dict[str, Reading] = {}
    for line in lines[head + 1 :]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != len(cols):
            continue
        r = Reading(_num(parts[ui]) if ui is not None else None, _num(parts[mi]) if mi is not None else None)
        if _keep(r):
            by_dev[parts[di] if di is not None else "0"] = r  # 同じ GPU は最後の行(-n 1 でも念のため)
    return list(by_dev.values())


_PDH_INSTANCE = re.compile(r"\(([^()]*)\)\\[^\\]*$")
_ENGINE = re.compile(r"luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_(\d+)_eng_\d+_engtype_(\w+)", re.I)
_ADAPTER = re.compile(r"^luid_(0x[0-9a-f]+_0x[0-9a-f]+)_phys_(\d+)$", re.I)


def parse_typeperf(text: str) -> list[Reading]:
    """Windows の typeperf(GPU Engine の Utilization Percentage と GPU Adapter Memory の
    Dedicated Usage、`-sc 1`)の CSV。アダプタ(LUID)ごとに、エンジンの種類(3D・Compute・Copy …)ごとに
    プロセスの使用率を足し、いちばん大きい種類を使用率にする(タスク マネージャーと同じ。100 で切る)。
    使用中のメモリは Dedicated Usage。PC 名・プロセス ID・LUID・時刻は読み捨てる。
    何も使っていない仮想のアダプタは除く。"""
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    head = next((i for i, r in enumerate(rows) if r[0].startswith("(PDH-CSV")), None)
    if head is None or head + 1 >= len(rows):
        return []
    names = rows[head]
    values = next((r for r in reversed(rows[head + 1 :]) if len(r) == len(names)), None)
    if values is None:
        return []
    engines: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    used: dict[str, float] = {}
    for name, value in zip(names[1:], values[1:], strict=True):
        m = _PDH_INSTANCE.search(name)
        v = _num(value)
        if not m or v is None:
            continue
        inst = m.group(1)
        e = _ENGINE.search(inst)
        if e:
            engines[f"{e.group(1)}_{e.group(2)}".lower()][e.group(3).lower()] += v
            continue
        a = _ADAPTER.match(inst)
        if a:
            used[f"{a.group(1)}_{a.group(2)}".lower()] = v / MIB
    out = []
    for luid in sorted(set(engines) | set(used)):
        util = min(100.0, max(engines[luid].values())) if engines.get(luid) else None
        r = Reading(util, used.get(luid))
        if (r.used_mib or 0) > 0 or (r.util_pct or 0) > 0:
            out.append(r)
    return out


_PERF_BLOCK = re.compile(r'"PerformanceStatistics"\s*=\s*\{([^{}]*)\}')


def parse_ioreg(text: str) -> list[Reading]:
    """macOS の `ioreg -r -d 1 -w 0 -c IOAccelerator`。GPU ごとの PerformanceStatistics の
    「Device Utilization %」(無ければ「GPU Activity(%)」)と「In use system memory」(バイト)。"""
    out = []
    for block in _PERF_BLOCK.findall(text):
        vals = {k: float(v) for k, v in re.findall(r'"([^"]+)"\s*=\s*(-?\d+(?:\.\d+)?)', block)}
        util = vals.get("Device Utilization %", vals.get("GPU Activity(%)"))
        used = vals.get("In use system memory")
        r = Reading(util, used / MIB if used is not None else None)
        if _keep(r):
            out.append(r)
    return out


# --- 道具を選んで読む ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str  # 利用者に見せる名前
    args: tuple[str, ...]
    parse: Any  # (text) -> list[Reading]
    # GPU が複数で ComfyUI の GPU を決められないとき、まとめた値(使用率は最大・メモリは合計)を使うか。
    # False なら読めないものとして扱う(xpu-smi は総量を出さず、どれが ComfyUI の GPU か決められないため)
    combine: bool = True


AMD_SMI = Tool("amd-smi", ("amd-smi", "metric", "--usage", "--mem-usage", "--json"), parse_amd_smi)
ROCM_SMI = Tool("rocm-smi", ("rocm-smi", "--showuse", "--showmeminfo", "vram", "--json"), parse_rocm_smi)
XPU_SMI = Tool(
    "xpu-smi", ("xpu-smi", "dump", "-d", "-1", "-m", "0,18", "-n", "1"), parse_xpu_smi_dump, combine=False
)
TYPEPERF = Tool(
    "Windows の性能カウンタ(typeperf)",
    (
        "typeperf",
        r"\GPU Engine(*)\Utilization Percentage",
        r"\GPU Adapter Memory(*)\Dedicated Usage",
        "-sc",
        "1",
    ),
    parse_typeperf,
)
IOREG = Tool("ioreg", ("ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"), parse_ioreg)


def tools_for(vendor: str, system: str | None = None) -> list[Tool]:
    """その会社・OS で試す道具(上から順)。無ければ空。
    Windows の AMD は amd-smi の Windows 版を確かめられなかった(AMD の文書は Linux の動きだけ)ので、
    会社によらない性能カウンタを使う。Intel の xpu-smi の Windows 版はデータセンター向けの GPU だけ
    なので同じにする。"""
    system = system or platform.system()
    if system == "Windows" and vendor in ("amd", "intel"):
        return [TYPEPERF]
    if system == "Linux" and vendor == "amd":
        return [AMD_SMI, ROCM_SMI]
    if system == "Linux" and vendor == "intel":
        return [XPU_SMI]
    if system == "Darwin" and vendor == "apple":
        return [IOREG]
    return []


def read_tool(tool: Tool) -> list[Reading]:
    """道具を 1 回動かして読む。無い・失敗・時間切れ・読めない出力は空。"""
    r = hardware._run(list(tool.args), TOOL_TIMEOUT)
    if r is None or r[0] != 0:
        return []
    try:
        return tool.parse(r[1])
    except (ValueError, TypeError, KeyError, IndexError, csv.Error):
        return []


def pick(readings: list[Reading], total_mib: float | None) -> tuple[Reading | None, bool]:
    """ComfyUI の GPU の読み取りを選ぶ。(読み取り, 複数をまとめたか)。
    1 枚ならそれ。総量が ComfyUI の総量と近いものが 1 枚ならそれ。使用中のメモリのあるものが 1 枚ならそれ。
    選べなければ、使用率は最大・メモリは合計にまとめる(ほかの GPU の仕事でも断る側に倒す)。"""
    if not readings:
        return None, False
    if len(readings) == 1:
        return readings[0], False
    if total_mib:
        near = [
            r for r in readings if r.total_mib and abs(r.total_mib - total_mib) <= total_mib * TOTAL_MATCH
        ]
        if len(near) == 1:
            return near[0], False
    busy = [r for r in readings if (r.used_mib or 0) > 0]
    if len(busy) == 1 and all(r.total_mib is None for r in readings):
        return busy[0], False
    utils = [r.util_pct for r in readings if r.util_pct is not None]
    mems = [r.used_mib for r in readings if r.used_mib is not None]
    return Reading(max(utils) if utils else None, sum(mems) if mems else None), True


class Probe:
    """bench の 1 回の測定で使う読み取り役。最初に読めた道具を覚えて、以後はそれだけを使う。"""

    def __init__(self, vendor: str, total_mib: float | None = None, system: str | None = None):
        self.tools = tools_for(vendor, system)
        self.total_mib = total_mib
        self.tool: Tool | None = None
        self.combined = False

    @property
    def tool_names(self) -> str:
        return "・".join(t.name for t in self.tools) or "なし"

    def read(self) -> Reading | None:
        for tool in [self.tool] if self.tool else self.tools:
            got, combined = pick(read_tool(tool), self.total_mib)
            if combined and not tool.combine:
                continue  # 複数の GPU のどれが ComfyUI のか決められない(道具で読めないのと同じ)
            if got is not None:
                self.tool = tool
                self.combined = self.combined or combined
                return got
        return None

    def used_mib(self) -> float | None:
        r = self.read()
        return r.used_mib if r else None
