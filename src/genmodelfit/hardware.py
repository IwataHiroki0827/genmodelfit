# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PC の検出。詳しい決まりと出典は docs/hardware.md。

- 判定に使う GPU は nvidia-smi で確かめた NVIDIA の GPU だけ(レジストリには取り外した GPU も残るため)。
  そのうち、今の PyTorch が動かない世代(Kepler 以前など)と、どの PyTorch も動かない古いドライバの GPU は
  backend を none にして CPU として判定する(理由は Gpu.notes と SystemInfo.notes)
- NVIDIA 以外の GPU(Windows は WMI + レジストリ、Linux は lspci)は知らせるだけで、判定には使わない。
  NVIDIA の GPU が見つかったときは既定では調べない(PowerShell の起動に約 1 秒かかるため。
  `system --all-gpus` で調べる)
- メモリは物理メモリと仮想メモリ(Windows はページファイルの設定とコミット、Linux はスワップ)
- PC 名・ユーザ名・GPU の UUID・PCI の場所や個体の ID は読まない・残さない
- 失敗しても例外を出さない(分からない項目は None にして、理由を notes に書く)
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from .models import Adapter, Gpu, SystemInfo

GIB = 1024**3
MIB = 1024**2

NVSMI_TIMEOUT = 10.0  # nvidia-smi(普段は 0.1〜0.5 秒。ドライバが固まっているときに待ち続けない)
OTHER_GPU_TIMEOUT = 8.0  # PowerShell の WMI・lspci(PowerShell の起動だけで約 1 秒)

# nvidia-smi で聞く項目。compute_cap を知らない古い nvidia-smi には QUERY_OLD で聞き直す
QUERY = ("name", "memory.total", "memory.free", "memory.used", "driver_version", "compute_cap")
QUERY_OLD = ("name", "memory.total", "memory.free", "memory.used", "driver_version")


# --- 外のコマンド・レジストリ(試験ではここを差し替える) -----------------------------------------


def _run(args: list[str], timeout: float) -> tuple[int, str] | None:
    """コマンドを動かして (終了コード, 標準出力) を返す。見つからない・時間切れ・失敗は None。
    Windows では窓を出さない(GUI のアプリから呼ばれても黒い窓が開かないように)。"""
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        r = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
            **kw,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return r.returncode, r.stdout.decode("utf-8", errors="replace")


def _winreg():
    try:
        import winreg

        return winreg
    except ImportError:  # Windows 以外
        return None


_DISPLAY_CLASS = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
_MEMORY_MANAGEMENT = r"SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management"


def _reg_display_entries() -> list[dict] | None:
    """画面の部品のクラスキー(0000, 0001 …)の DriverDesc・MatchingDeviceId・メモリの大きさ。
    取り外した GPU の項目も残っているので、WMI で今ある部品と突き合わせてから使う。"""
    winreg = _winreg()
    if winreg is None:
        return None
    out: list[dict] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS)
    except OSError:
        return None
    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not re.fullmatch(r"\d{4}", sub):
                continue
            try:
                key = winreg.OpenKey(root, sub)
            except OSError:  # Properties など読めないキー
                continue
            item: dict = {}
            with key:
                for name in (
                    "DriverDesc",
                    "MatchingDeviceId",
                    "HardwareInformation.qwMemorySize",
                    "HardwareInformation.MemorySize",
                ):
                    with contextlib.suppress(OSError):
                        item[name] = winreg.QueryValueEx(key, name)[0]
            out.append(item)
    return out


def _reg_paging_files() -> list[str] | None:
    """ページファイルの設定(Memory Management の PagingFiles。REG_MULTI_SZ)。読めなければ None。"""
    winreg = _winreg()
    if winreg is None:
        return None
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _MEMORY_MANAGEMENT) as key:
            value = winreg.QueryValueEx(key, "PagingFiles")[0]
    except OSError:
        return None
    if isinstance(value, str):
        value = [value]
    return [str(v) for v in value] if isinstance(value, list) else None


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _global_memory_status() -> dict[str, int] | None:
    """Windows の GlobalMemoryStatusEx。ullTotalPageFile は今のコミットの上限(物理メモリ + ページファイル)、
    ullAvailPageFile はそのうちの空き。Windows 以外・失敗は None。"""
    if os.name != "nt":
        return None
    try:
        st = _MemoryStatusEx()
        st.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
    except (OSError, AttributeError):
        return None
    return {
        "total_phys": int(st.ullTotalPhys),
        "avail_phys": int(st.ullAvailPhys),
        "commit_limit": int(st.ullTotalPageFile),
        "commit_avail": int(st.ullAvailPageFile),
    }


def _read_text(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


# --- 数の読み方 ------------------------------------------------------------------------------------


def _clean(value: str) -> str | None:
    """nvidia-smi の値。`[N/A]`・`[Not Supported]`・`N/A`・空は None。"""
    v = value.strip()
    if not v or v.startswith("[") or v.upper() in ("N/A", "NA", "NOT SUPPORTED", "UNKNOWN ERROR"):
        return None
    return v


def _number(value: str) -> float | None:
    v = _clean(value)
    if v is None:
        return None
    v = re.sub(r"\s*(MiB|MB)$", "", v, flags=re.I).replace(",", ".")  # 単位付き・小数点がカンマの出力も読む
    try:
        x = float(v)
    except ValueError:
        return None
    return x if x >= 0 and x == x and x != float("inf") else None


def parse_version(text: str | None) -> tuple[int, ...] | None:
    """ドライバの版("566.03" / "535.183.01")を数の組にする。読めなければ None。"""
    if not text:
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", text)
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def parse_compute_cap(text: str | None) -> tuple[int, int] | None:
    v = _clean(text or "")
    if v is None:
        return None
    m = re.fullmatch(r"(\d{1,2})[.,](\d)", v)
    return (int(m.group(1)), int(m.group(2))) if m else None


# --- GPU の世代 ----------------------------------------------------------------------------------


def arch_of(cc: tuple[int, int]) -> str:
    """compute capability → 世代の名前(NVIDIA CUDA GPUs の一覧による)。"""
    major, minor = cc
    if major < 3:
        return "fermi"  # Fermi 以前
    if major == 3:
        return "kepler"
    if major == 5:
        return "maxwell"
    if major == 6:
        return "pascal"
    if major == 7:
        return "volta" if minor < 5 else "turing"
    if major == 8:
        return "ada" if minor >= 9 else "ampere"
    if major == 9:
        return "hopper"
    if major in (10, 11, 12):
        return "blackwell"
    return "unknown"


# compute_cap を返さない古い nvidia-smi のための、GPU 名 → compute capability の表(上から順に試す)。
# 迷う名前(同じ型番で世代が混ざるノート用の GT/GTX 7xx・8xx など)は入れない(None = 分からない)
_NAME_CC: tuple[tuple[str, tuple[int, int]], ...] = (
    # Blackwell
    (r"\brtx\s*50[5-9]0\b|\brtx pro \d{4}\b.*\bblackwell\b", (12, 0)),
    (r"\bgb10\b", (12, 1)),
    (r"\bg?b300\b", (10, 3)),
    (r"\bg?b[12]00\b", (10, 0)),
    # Hopper
    (r"\bg?h(100|200|800|20)\b", (9, 0)),
    # Ada Lovelace
    (r"\brtx\s*40[5-9]0\b|\brtx\s*\d{4}\s*ada\b|\bl40s?\b|\bl4\b|\bl20\b", (8, 9)),
    # Ampere(RTX 2050 のノート用は GA107 で Ampere)
    (r"\ba100\b|\ba800\b|\ba30\b", (8, 0)),
    (
        r"\brtx\s*30[5-9]0\b|\brtx\s*2050\b|\brtx\s*a\d{3,4}\b|\ba10g?\b|\ba16\b|\ba40\b|\ba2\b|\bmx570\b",
        (8, 6),
    ),
    # Turing
    (
        r"\brtx\s*20[6-8]0\b|\btitan rtx\b|\bquadro rtx\b|\bgtx\s*16[3-6]0\b|\bt4\b"
        r"|\bt(400|500|550|600|1000|1200|2000)\b|\bmx(450|550)\b",
        (7, 5),
    ),
    # Volta
    (r"\bv100\b|\btitan v\b|\bgv100\b", (7, 0)),
    # Pascal
    (r"\bp100\b|\bgp100\b", (6, 0)),
    (
        r"\bgtx\s*10[5-8]0\b|\bgt\s*1030\b|\btitan xp\b|(?<!gtx )\btitan x\b|\bquadro p\d{3,4}\b"
        r"|\btesla p(4|40)\b|\bmx(150|230|250|330|350)\b",
        (6, 1),
    ),
    # Maxwell(Quadro K620 / K1200 / K2200 は名前は K だが Maxwell の GM107)。
    # 「GeForce GTX TITAN X」は Maxwell、「NVIDIA TITAN X」は Pascal(NVIDIA の一覧どおり)
    (
        r"\bgtx\s*(745|750)\b|\bquadro k(620|1200|2200)\b|\bquadro m(520|1200)\b"
        r"|\bmx1[13]0\b|\b940mx\b|\btesla m10\b",
        (5, 0),
    ),
    (
        r"\bgtx\s*(950|960|970|980)\b|\bgtx titan x\b|\bquadro m(620|2000|2200|4000|5000|6000)\b"
        r"|\btesla m(4|6|40|60)\b",
        (5, 2),
    ),
    # Kepler(デスクトップの GTX 6xx・7xx、GTX TITAN / Black / Z、Quadro K・Tesla K)
    (r"\btesla k80\b", (3, 7)),
    (
        r"\bgtx\s*780\b|\bgtx titan( black| z)?$|\btesla k(20|40)\w*\b|\bgt\s*(710|720)\b"
        r"|\bquadro k(5200|6000)\b",
        (3, 5),
    ),
    (r"\bgtx\s*(6[3-9]0|760|770)\b|\bquadro k\d{3,4}\b", (3, 0)),
)


def cc_from_name(name: str) -> tuple[int, int] | None:
    n = re.sub(r"\s+", " ", name.lower().replace("nvidia", "").replace("geforce", "")).strip()
    n = n.replace("(tm)", "").replace("(r)", "")
    for pat, cc in _NAME_CC:
        if re.search(pat, n):
            return cc
    return None


# --- 今の PyTorch が対応する世代とドライバ ---------------------------------------------------------

# PyTorch の CUDA 版のビルドが入れている compute capability と、要るドライバ(docs/hardware.md の表)。
# 今の版(2.14.1、2026-09-30)は cu126 / cu130 / cu132。cu128(2.7〜2.11)・cu129(2.8〜2.13)は
# 今の版には無いが、配布先(download.pytorch.org)に残っていて、ComfyUI の同梱版や
# AI生成キットのように版を固定して使う側があるので数える(ドライバ 580 未満の Blackwell が動く道)。
# 出典: 各版の torch/cuda/__init__.py の PYTORCH_RELEASES_CODE_CC と
# .ci/pytorch/windows/internal/cuda_config.bat・build_env_setup.py。Windows の cu126 は 6.1 も入れている
# (Linux は 6.0 の機械語が 6.1 でも動く)。最低のドライバは NVIDIA CUDA Toolkit のリリースノートの
# 表 3「CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility」
# (12.x は 525 以上、13.x は 580 以上。OS で分けていない)。世代ごとの最低は ARCH_MIN_DRIVER
TORCH_RELEASE = "2.14.1"


@dataclass(frozen=True)
class CudaBuild:
    tag: str  # cu132 など
    cuda: str
    ccs: dict[str, frozenset[int]]  # 機械("x86_64" / "aarch64")ごとの compute capability(86 = 8.6)
    min_driver_windows: tuple[int, ...]
    min_driver_linux: tuple[int, ...]
    torch: str  # このビルドがある PyTorch の版


_CU13_X86 = frozenset({75, 80, 86, 90, 100, 120})
_CU13_ARM = frozenset({80, 90, 100, 110, 120})
_CU12_WIN = (525,)
_CU12_LINUX = (525,)
TORCH_CUDA_BUILDS: tuple[CudaBuild, ...] = (
    CudaBuild("cu132", "13.2", {"x86_64": _CU13_X86, "aarch64": _CU13_ARM}, (580,), (580,), "2.12〜2.14"),
    CudaBuild("cu130", "13.0", {"x86_64": _CU13_X86, "aarch64": _CU13_ARM}, (580,), (580,), "2.9〜2.14"),
    CudaBuild(
        "cu129",
        "12.9",
        {"x86_64": _CU13_X86, "aarch64": frozenset({80, 90, 100, 120})},
        _CU12_WIN,
        _CU12_LINUX,
        "2.8〜2.13",
    ),
    CudaBuild("cu128", "12.8", {"x86_64": _CU13_X86}, _CU12_WIN, _CU12_LINUX, "2.7〜2.11"),
    CudaBuild(
        "cu126",
        "12.6",
        {"x86_64": frozenset({50, 60, 61, 70, 75, 80, 86, 90}), "aarch64": frozenset({80, 90})},
        _CU12_WIN,
        _CU12_LINUX,
        "2.6〜2.14",
    ),
)

# 世代ごとの最低のドライバ。Blackwell(10.x・12.x)の機械語が入るのは CUDA 12.8 からで(PyTorch の cu128 以降。
# 上の表)、CUDA 12.8 GA のドライバは Linux 570.26 / Windows 570.65(CUDA のリリースノートの表 4)。
# それより古いドライバは Blackwell を知らない(nvidia-smi に出ること自体がまれ)
ARCH_MIN_DRIVER: dict[str, tuple[tuple[int, ...], str]] = {
    "blackwell": ((570,), "CUDA 12.8 GA(Linux 570.26 / Windows 570.65)"),
}

# Jetson など、同じ世代の機械語が動かない compute capability(PyTorch の DEVICE_REQUIREMENT と同じ)
_SOLO_CC = {53, 62, 72, 87, 101, 110}


def _code_runs(code: int, cc: tuple[int, int]) -> bool:
    """compute capability X.y 向けの機械語は、同じ X で y 以上の GPU で動く
    (CUDA C++ Programming Guide「Binary Compatibility」)。"""
    dev = cc[0] * 10 + cc[1]
    if dev in _SOLO_CC or code in _SOLO_CC:
        return dev == code or {dev, code} == {101, 110}
    return code // 10 == cc[0] and code % 10 <= cc[1]


def _machine() -> str:
    m = platform.machine().lower()
    return "aarch64" if m in ("aarch64", "arm64") else "x86_64"


def _fmt_ver(v: tuple[int, ...]) -> str:
    return ".".join(f"{p:02d}" if i == 1 and len(v) == 2 else str(p) for i, p in enumerate(v))


@dataclass
class Support:
    builds: list[str]  # 世代とドライバの両方で動く CUDA 版(新しい順)
    arch_builds: list[str]  # 世代だけで見て動く CUDA 版
    usable: bool
    notes: list[str] = field(default_factory=list)


def torch_support(
    cc: tuple[int, int] | None, driver: str | None, *, windows_driver: bool, machine: str = "x86_64"
) -> Support:
    """PyTorch の CUDA 版のビルド(TORCH_CUDA_BUILDS)のうち、この GPU(compute capability)と
    ドライバで動くもの。分からない項目(cc・ドライバ)では断らない(分からないことを notes に書く)。"""
    notes: list[str] = []
    if cc is None:
        arch_builds = [b for b in TORCH_CUDA_BUILDS if b.ccs.get(machine)]
        notes.append("GPU の世代(compute capability)が分からないので、世代では断っていない")
    else:
        arch_builds = [b for b in TORCH_CUDA_BUILDS if any(_code_runs(c, cc) for c in b.ccs.get(machine, ()))]
        if not arch_builds:
            arch = arch_of(cc)
            label = {"fermi": "Fermi 以前の", "kepler": "Kepler ", "unknown": "この"}.get(
                arch, arch.capitalize() + " "
            )
            notes.append(
                f"{label}世代(compute capability {cc[0]}.{cc[1]})は PyTorch の CUDA 版のビルドが"
                "どれも対応しないので、CPU として判定した(ドライバを更新しても GPU では動かない)"
            )
            return Support([], [], False, notes)
    ver = parse_version(driver)
    if ver is None:
        notes.append("ドライバの版が読めないので、ドライバでは断っていない")
        return Support([b.tag for b in arch_builds], [b.tag for b in arch_builds], True, notes)

    floor = ARCH_MIN_DRIVER.get(arch_of(cc)) if cc else None
    if floor and ver < floor[0]:
        notes.append(
            f"ドライバ {driver} はこの世代({arch_of(cc).capitalize()})に対応しない"
            f"({_fmt_ver(floor[0])} 以上。"
            f"{floor[1]})ので、CPU として判定した(ドライバを更新すれば GPU で動く)"
        )
        return Support([], [b.tag for b in arch_builds], False, notes)

    def need(b: CudaBuild) -> tuple[int, ...]:
        return b.min_driver_windows if windows_driver else b.min_driver_linux

    ok = [b for b in arch_builds if ver >= need(b)]
    if not ok:
        easiest = min(arch_builds, key=need)
        notes.append(
            f"ドライバ {driver} は古く、この GPU で動く PyTorch(CUDA {easiest.cuda} 版など)には "
            f"{_fmt_ver(need(easiest))} 以上が要るので、CPU として判定した(ドライバを更新すれば GPU で動く)"
        )
        return Support([], [b.tag for b in arch_builds], False, notes)
    if len(ok) < len(arch_builds):
        bad = [b for b in arch_builds if b not in ok]
        tags = "・".join(b.tag for b in reversed(bad))
        notes.append(
            f"ドライバ {driver} では CUDA {bad[-1].cuda} 版以降の PyTorch({tags})は動かない"
            f"({_fmt_ver(min(need(b) for b in bad))} 以上が要る)。CUDA {ok[0].cuda} 版({ok[0].tag})なら動く"
        )
    return Support([b.tag for b in ok], [b.tag for b in arch_builds], True, notes)


# --- ComfyUI が選ぶ精度 --------------------------------------------------------------------------

# ComfyUI の comfy/model_management.py(should_use_fp16)の名前の一覧と同じもの(docs/hardware.md の版)。
# Pascal のこれらは Windows では fp16 で動かし、Linux では fp32 で計算する。
# 下の一覧(GTX 16 系など)は fp16 が壊れているとして fp32 で計算する(こちらは大文字・小文字を区別する)
_COMFY_FP16_PASCAL = (
    "1080",
    "1070",
    "titan x",
    "p3000",
    "p3200",
    "p4000",
    "p4200",
    "p5000",
    "p5200",
    "p6000",
    "1060",
    "1050",
    "p40",
    "p100",
    "p6",
    "p4",
)
_COMFY_FP16_BROKEN = (
    "1660",
    "1650",
    "1630",
    "T500",
    "T550",
    "T600",
    "MX550",
    "MX450",
    "CMP 30HX",
    "T2000",
    "T1000",
    "T1200",
)


def comfy_precision(
    cc: tuple[int, int] | None, name: str, *, windows: bool
) -> tuple[str | None, bool | None, bool | None]:
    """(計算の精度, bf16 の計算に対応, fp8 の計算に対応)。ComfyUI の should_use_fp16 / should_use_bf16 /
    supports_fp8_compute を、起動の引数を付けない既定の動きでなぞったもの。cc が分からなければ全部 None。
    計算の精度: fp16(半精度で計算できる)/ bf16 / fp32(半精度を使わない。作業用メモリが倍になる)。"""
    if cc is None:
        return None, None, None
    major, minor = cc
    if major >= 8:
        fp16 = True
    elif major < 6:
        fp16 = False
    elif any(x in name.lower() for x in _COMFY_FP16_PASCAL):
        fp16 = windows  # Linux では fp32 のほうが速いとして使わない
    elif major < 7 or any(x in name for x in _COMFY_FP16_BROKEN):
        fp16 = False
    else:
        fp16 = True
    bf16 = major >= 8
    fp8 = (major, minor) >= (8, 9)
    return ("fp16" if fp16 else "bf16" if bf16 else "fp32"), bf16, fp8


# --- nvidia-smi ----------------------------------------------------------------------------------


def nvidia_smi_candidates() -> list[str]:
    """nvidia-smi を探す場所。PATH → Windows の System32 → 古いドライバの NVSMI →
    Linux・WSL の決まった場所。"""
    out: list[str] = []
    found = shutil.which("nvidia-smi")
    if found:
        out.append(found)
    if os.name == "nt":
        root = os.environ.get("SYSTEMROOT") or os.environ.get("WINDIR") or r"C:\Windows"
        out.append(str(Path(root) / "System32" / "nvidia-smi.exe"))
        for env in ("ProgramW6432", "ProgramFiles"):
            pf = os.environ.get(env)
            if pf:
                out.append(str(Path(pf) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"))
        out.append(r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe")
    else:
        out += ["/usr/bin/nvidia-smi", "/usr/lib/wsl/lib/nvidia-smi", "/usr/local/bin/nvidia-smi"]
    seen: set[str] = set()
    return [p for p in out if not (p.lower() in seen or seen.add(p.lower()))]


def find_nvidia_smi() -> str | None:
    for p in nvidia_smi_candidates():
        if os.path.isfile(p):
            return p
    return None


def parse_nvidia_smi(text: str, fields: tuple[str, ...] = QUERY) -> list[Gpu]:
    """`nvidia-smi --query-gpu=<fields> --format=csv,noheader,nounits` の出力を読む。
    1 行 1 枚。項目の数が合わない行(エラーの文言など)は飛ばす。`[N/A]` などの値は None にする。
    世代・ドライバの判断は classify_gpu で行う(ここでは読むだけ)。"""
    gpus: list[Gpu] = []
    idx = {f: i for i, f in enumerate(fields)}
    for raw in text.splitlines():
        line = raw.strip().lstrip("\ufeff")
        if not line:
            continue
        parts = [p.strip() for p in line.split(", ")]
        if len(parts) != len(fields):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != len(fields):
                continue

        def get(key: str, parts: list[str] = parts) -> str:
            return parts[idx[key]] if key in idx else ""

        name = _clean(get("name")) or "NVIDIA GPU"
        total = _number(get("memory.total"))
        free = _number(get("memory.free"))
        used = _number(get("memory.used"))
        if total is None and free is None and used is None and _clean(get("driver_version")) is None:
            continue  # 数が 1 つも読めない行は GPU の行ではない
        if free is None and total is not None and used is not None:
            free = max(0.0, total - used)
        notes: list[str] = []
        if total is None or total <= 0:
            notes.append(
                "nvidia-smi が VRAM の量を返さない(メモリを CPU と共有する GPU など)ので、判定に使わない"
            )
        cc = parse_compute_cap(get("compute_cap")) if "compute_cap" in idx else None
        gpus.append(
            Gpu(
                name=name,
                vendor="nvidia",
                vram_total_gb=round((total or 0.0) / 1024, 2),
                vram_free_gb=round(free / 1024, 2) if free is not None else None,
                vram_used_gb=round(used / 1024, 2) if used is not None else None,
                driver=_clean(get("driver_version")),
                backend="cuda" if total and total > 0 else "none",
                compute_capability=f"{cc[0]}.{cc[1]}" if cc else None,
                compute_capability_source="nvidia-smi" if cc else None,
                notes=notes,
            )
        )
    return gpus


def classify_gpu(g: Gpu, *, windows_driver: bool, comfy_windows: bool, machine: str = "x86_64") -> Gpu:
    """世代(compute capability)・今の PyTorch の対応・ドライバ・ComfyUI の精度を決めて、g に書き込む。
    windows_driver: ドライバの版を Windows の表で見るか(WSL も Windows のドライバ)。
    comfy_windows: ComfyUI が Windows で動くか(Pascal の fp16 の扱いが変わる。WSL は Linux)。"""
    cc = parse_compute_cap(g.compute_capability)
    if cc is None:
        cc = cc_from_name(g.name)
        if cc:
            g.compute_capability = f"{cc[0]}.{cc[1]}"
            g.compute_capability_source = "name"
            g.notes.append(
                f"nvidia-smi が世代を返さないので、GPU 名から compute capability {cc[0]}.{cc[1]} と推した"
            )
    g.arch = arch_of(cc) if cc else None
    sup = torch_support(cc, g.driver, windows_driver=windows_driver, machine=machine)
    g.torch_cuda = sup.builds
    g.notes.extend(sup.notes)
    if not sup.usable:
        g.backend = "none"
    if g.backend == "cuda":  # 判定に使わない GPU には ComfyUI の精度を出さない
        g.compute_dtype, g.bf16, g.fp8 = comfy_precision(cc, g.name, windows=comfy_windows)
    if g.compute_dtype == "fp32":
        g.notes.append(
            "ComfyUI はこの GPU では半精度(fp16)を使わず fp32 で計算するので、"
            "作業用メモリを 2 倍で見積もる(遅い)"
        )
    return g


def _ask(exe: str, fields: tuple[str, ...]) -> tuple[int, str] | None:
    return _run([exe, f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits"], NVSMI_TIMEOUT)


def _partial_notes(result: tuple[int, str]) -> list[str]:
    """読めた行があるのに失敗した(終了コードが 0 でない・読めない GPU がある)ときの注意。
    エラーの文言には PCI の差し場所が入るので写さない。"""
    code, out = result
    bad = sum(1 for line in out.splitlines() if "Unable to determine" in line or "Unknown Error" in line)
    if bad:
        return [
            f"nvidia-smi が {bad} 枚の GPU の情報を読めなかった(ドライバ・GPU の不調)。"
            "読めた GPU だけで判定した"
        ]
    if code != 0:
        return [f"nvidia-smi が終了コード {code} を返した。読めた GPU だけで判定した"]
    return []


def query_nvidia_smi(exe: str) -> tuple[list[Gpu], list[str]]:
    """nvidia-smi に聞く。読めなければ compute_cap を除いた QUERY_OLD で聞き直す
    (compute_cap を知らない古い nvidia-smi は「not a valid field」で失敗する)。(GPU, 分かったこと)。"""
    first = _ask(exe, QUERY)
    if first is None:
        return [], ["nvidia-smi が動かない・時間内に終わらないので、GPU は無いものとして判定した"]
    gpus = parse_nvidia_smi(first[1], QUERY)
    if gpus:  # 1 枚だけ読めない(「Unable to determine the device handle」)ときも、読めた行は使う
        return gpus, _partial_notes(first)
    second = _ask(exe, QUERY_OLD)
    if second is not None:
        gpus = parse_nvidia_smi(second[1], QUERY_OLD)
        if gpus:
            return gpus, [
                "古い nvidia-smi で compute_cap を聞けなかったので、GPU 名から世代を推した",
                *_partial_notes(second),
            ]
    text = first[1] + (second[1] if second else "")
    if "No devices were found" in text:
        return [], ["nvidia-smi は動いたが NVIDIA の GPU が見つからない"]
    if first[0] != 0:
        return [], [
            f"nvidia-smi が失敗した(終了コード {first[0]})。ドライバが動いていない可能性があるので、"
            "GPU は無いものとして判定した"
        ]
    return [], ["nvidia-smi の出力が読めないので、GPU は無いものとして判定した"]


def detect_gpus(
    *, windows_driver: bool | None = None, comfy_windows: bool | None = None
) -> tuple[list[Gpu], list[str]]:
    """nvidia-smi で NVIDIA の GPU を調べ、世代・ドライバ・精度を決める。(GPU, 分かったこと)。"""
    exe = find_nvidia_smi()
    if not exe:
        return [], [
            "nvidia-smi が見つからない(NVIDIA のドライバが入っていない)ので、GPU は無いものとして判定した"
        ]
    gpus, notes = query_nvidia_smi(exe)
    wd = (os.name == "nt" or is_wsl()) if windows_driver is None else windows_driver
    cw = (os.name == "nt") if comfy_windows is None else comfy_windows
    machine = _machine()
    for g in gpus:
        classify_gpu(g, windows_driver=wd, comfy_windows=cw, machine=machine)
    return gpus, notes


# --- NVIDIA 以外の GPU(知らせるだけ) --------------------------------------------------------------

_VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}
# WMI の AdapterRAM は 32 ビットなので、4 GiB 以上の GPU はこの値(またはそれに近い値)で頭打ちになる
_ADAPTER_RAM_CAP = 0xFFF00000
_MIN_PLAUSIBLE_VRAM = 64 * MIB  # これより小さい「VRAM」は単位の違い(MB で書いたドライバ)とみなして使わない

_PS_WMI = (
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "$a=@(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | "
    "Select-Object Name,AdapterRAM,PNPDeviceID,ConfigManagerErrorCode);"
    "ConvertTo-Json -InputObject $a -Compress"
)


def _norm_name(name: str) -> str:
    n = name.lower().replace("(tm)", " ").replace("(r)", " ")
    return " ".join(re.sub(r"[^0-9a-z]+", " ", n).split())


def _hwid(pnp: str) -> str | None:
    """PNPDeviceID の、型番を表す前半("pci\\ven_10de&dev_2488&subsys_…&rev_a1")。
    後半(個体・差し場所の ID)は捨てる。PCI でないもの(仮想の画面など)は None。"""
    first = pnp.split("\\")
    if len(first) < 2 or first[0].upper() != "PCI":
        return None
    return f"pci\\{first[1].lower()}"


def _reg_vram(entry: dict) -> int | None:
    for key in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
        v = entry.get(key)
        if isinstance(v, (bytes, bytearray)):
            v = int.from_bytes(bytes(v[:8]), "little") if len(v) >= 4 else None
        if isinstance(v, int) and v < 0:
            v += 1 << 32
        if isinstance(v, int) and v >= _MIN_PLAUSIBLE_VRAM:
            if key == "HardwareInformation.MemorySize" and v >= _ADAPTER_RAM_CAP:
                continue  # 32 ビットで頭打ちの値
            return v
    return None


def _match_registry(hwid: str, name: str, registry: list[dict]) -> int | None:
    """WMI の部品に当たるレジストリの VRAM。型番の ID(MatchingDeviceId)で当て、無ければ名前で当てる。
    当たる項目が複数で値が違う(取り外した同名の GPU が残っている)ときは使わない。"""
    by_id = [
        e
        for e in registry
        if isinstance(e.get("MatchingDeviceId"), str) and hwid.startswith(e["MatchingDeviceId"].lower())
    ]
    cands = by_id or [e for e in registry if _norm_name(str(e.get("DriverDesc") or "")) == _norm_name(name)]
    values = {v for v in (_reg_vram(e) for e in cands) if v}
    return values.pop() if len(values) == 1 else None


def parse_wmi_video(text: str, registry: list[dict] | None = None) -> tuple[list[Adapter], list[str]]:
    """PowerShell の Win32_VideoController(JSON)を読む。PCI の部品だけを残す
    (仮想の画面・リモートの画面は捨てる)。VRAM はレジストリの qwMemorySize を使い、
    無ければ 4 GiB 未満の AdapterRAM だけを使う。(部品, 分かったこと)。"""
    try:
        data = json.loads(text.strip().lstrip("\ufeff") or "null")
    except ValueError:
        return [], []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return [], []
    out: list[Adapter] = []
    notes: list[str] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        hwid = _hwid(str(d.get("PNPDeviceID") or ""))
        if hwid is None:
            continue
        m = re.search(r"ven_([0-9a-f]{4})", hwid)
        vendor = _VENDOR_IDS.get(m.group(1), "unknown") if m else "unknown"
        name = str(d.get("Name") or "").strip() or f"{vendor} GPU"
        vram, src = None, None
        reg = _match_registry(hwid, name, registry or [])
        if reg:
            vram, src = reg, "registry"
        else:
            raw = d.get("AdapterRAM")
            if isinstance(raw, int) and _MIN_PLAUSIBLE_VRAM <= raw < _ADAPTER_RAM_CAP:
                vram, src = raw, "wmi"
        err = d.get("ConfigManagerErrorCode")
        if vendor == "nvidia" and "microsoft basic" in name.lower():
            notes.append(
                "NVIDIA の GPU があるが、NVIDIA のドライバが入っていない(標準の画面ドライバで動いている)。"
                "ドライバを入れると GPU で判定できる"
            )
        elif isinstance(err, int) and err != 0:
            notes.append(f"{name} は Windows のデバイスマネージャーで問題がある状態(コード {err})")
        out.append(
            Adapter(
                name=name,
                vendor=vendor,
                vram_gb=round(vram / GIB, 2) if vram else None,
                vram_source=src,
                source="wmi",
            )
        )
    return out, notes


_LSPCI_LINE = re.compile(
    r"^(?P<slot>[0-9a-f:.]+)\s+(?P<cls>[^\[]*?)\s*\[(?P<code>03[0-9a-f]{2})\]:\s*(?P<rest>.*)$", re.I
)


def parse_lspci(text: str, vram_of=None) -> list[Adapter]:
    """`lspci -Dnn` の出力から画面の部品(クラス 03xx)を読む。vram_of(slot) で sysfs の VRAM を引ける。
    差し場所(slot)は VRAM を引くのに使うだけで残さない。"""
    out: list[Adapter] = []
    for line in text.splitlines():
        m = _LSPCI_LINE.match(line.strip())
        if not m:
            continue
        rest = re.sub(r"\s*\(rev [0-9a-f]+\)\s*$", "", m.group("rest"), flags=re.I)
        ids = list(re.finditer(r"\[([0-9a-f]{4}):([0-9a-f]{4})\]", rest, flags=re.I))
        vendor = _VENDOR_IDS.get(ids[-1].group(1).lower(), "unknown") if ids else "unknown"
        desc = rest[: ids[-1].start()].strip() if ids else rest.strip()
        desc = re.sub(
            r"^(NVIDIA Corporation|Advanced Micro Devices, Inc\. \[AMD/ATI\]|Intel Corporation)\s*", "", desc
        )
        product = re.search(r"\[([^\]]+)\]\s*$", desc)  # "GA104 [GeForce RTX 3070]" の括弧の中が製品名
        label = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}.get(vendor, "")
        name = " ".join(x for x in (label, product.group(1) if product else desc) if x) or "GPU"
        vram = vram_of(m.group("slot")) if vram_of else None
        out.append(
            Adapter(
                name=name,
                vendor=vendor,
                vram_gb=round(vram / GIB, 2) if vram else None,
                vram_source="sysfs" if vram else None,
                source="lspci",
            )
        )
    return out


def _sysfs_vram(slot: str) -> int | None:
    """Linux の amdgpu が出す VRAM の量(mem_info_vram_total)。"""
    if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", slot, flags=re.I):
        return None
    text = _read_text(f"/sys/bus/pci/devices/{slot}/mem_info_vram_total")
    try:
        v = int((text or "").strip())
    except ValueError:
        return None
    return v if v >= _MIN_PLAUSIBLE_VRAM else None


def detect_other_gpus() -> tuple[list[Adapter], list[str]]:
    """NVIDIA 以外も含めた画面の部品。Windows は PowerShell の WMI + レジストリ、Linux は lspci。
    失敗・時間切れは黙って飛ばす(空のまま)。"""
    if os.name == "nt":
        r = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_WMI], OTHER_GPU_TIMEOUT)
        if r is None or r[0] != 0:
            return [], []
        return parse_wmi_video(r[1], _reg_display_entries() or [])
    if platform.system() == "Linux":
        r = _run(["lspci", "-Dnn"], OTHER_GPU_TIMEOUT)
        if r is None or r[0] != 0:
            return [], []
        return parse_lspci(r[1], _sysfs_vram), []
    return [], []


def _other_gpu_notes(adapters: list[Adapter], judged_gpu: bool) -> list[str]:
    notes = []
    for a in adapters:
        if a.vendor == "nvidia":
            if (
                not judged_gpu and a.source == "lspci"
            ):  # Windows はドライバの無いときの注意を WMI の読みで出す
                notes.append(
                    f"{a.name} があるが、nvidia-smi で確かめられないので CPU として判定した"
                    "(NVIDIA のドライバを入れると GPU で判定できる)"
                )
            continue
        vram = f"(VRAM {a.vram_gb:g} GB)" if a.vram_gb else ""
        label = {"amd": "AMD", "intel": "Intel"}.get(a.vendor, "NVIDIA 以外")
        if judged_gpu:
            notes.append(f"{a.name}{vram} を見つけた。{label} の GPU は判定に使わない(v0.1 は NVIDIA だけ)")
        else:
            notes.append(
                f"{a.name}{vram} を見つけた。v0.1 は NVIDIA の GPU だけを判定するので、CPU として判定した"
            )
    return notes


# --- メモリ ----------------------------------------------------------------------------------------


@dataclass
class Memory:
    total: int  # バイト
    available: int
    swap_total: int | None = None  # 今のページファイル・スワップの大きさ
    swap_max: int | None = None  # 伸びる上限(None は分からない・システム管理で伸びる)
    swap_auto: bool | None = None
    commit_limit: int | None = None  # Windows のコミットの上限(物理メモリ + ページファイル)
    commit_used: int | None = None
    notes: list[str] = field(default_factory=list)


def parse_paging_files(values: list[str]) -> tuple[int | None, bool | None]:
    """Memory Management の PagingFiles を読む。(上限のバイト数, システム管理か)。
    - 空 → ページファイル無し (0, False)
    - "?:\\pagefile.sys"(すべてのドライブで自動)・"C:\\pagefile.sys 0 0"・大きさの無い行
      → システム管理 (None, True)
    - "C:\\pagefile.sys 1024 4096" → 最大 4096 MB の合計 (バイト, False)"""
    lines = [v.strip() for v in values if v and v.strip()]
    if not lines:
        return 0, False
    total = 0
    for line in lines:
        if line.startswith("?:"):
            return None, True
        m = re.match(r"^(.*?\.sys)\s*(\d+)?\s*(\d+)?\s*$", line, flags=re.I)
        if not m:
            return None, None
        lo, hi = m.group(2), m.group(3)
        if lo is None or hi is None or (int(lo) == 0 and int(hi) == 0):
            return None, True
        total += int(hi) * MIB
    return total, False


def _linux_swaps(text: str) -> int:
    """/proc/swaps の合計(バイト)。zram(メモリを圧縮して使うもの)は ComfyUI と同じく数えない。"""
    total = 0
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        if os.path.basename(parts[0]).startswith("zram"):
            continue
        with contextlib.suppress(ValueError):
            total += int(parts[2]) * 1024  # KiB
    return total


def _cgroup_limit() -> int | None:
    """コンテナ(cgroup)のメモリ上限。無ければ None。"""
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        text = _read_text(path)
        if text and text.strip().isdigit():
            return int(text.strip())
    return None


def is_wsl() -> bool:
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


def read_memory() -> Memory:
    vm = psutil.virtual_memory()
    mem = Memory(total=int(vm.total), available=int(vm.available))
    if os.name == "nt":
        st = _global_memory_status()
        if st:
            mem.commit_limit = st["commit_limit"]
            mem.commit_used = max(0, st["commit_limit"] - st["commit_avail"])
            mem.swap_total = max(0, st["commit_limit"] - st["total_phys"])
        paging = _reg_paging_files()
        if paging is None:
            mem.notes.append("ページファイルの設定が読めないので、仮想メモリは目安の倍率で見た")
        else:
            mem.swap_max, mem.swap_auto = parse_paging_files(paging)
            if mem.swap_max is None and mem.swap_auto is None:
                mem.notes.append("ページファイルの設定が読めないので、仮想メモリは目安の倍率で見た")
            elif mem.swap_max == 0:
                mem.notes.append("ページファイルが無いので、メモリが足りないモデルは動かないと判定する")
    elif platform.system() == "Linux":
        text = _read_text("/proc/swaps")
        if text is not None:
            mem.swap_total = mem.swap_max = _linux_swaps(text)
            mem.swap_auto = False
        limit = _cgroup_limit()
        if limit and limit < mem.total:
            mem.notes.append(f"コンテナのメモリ上限 {limit / GIB:.1f} GB を、搭載量の代わりに使った")
            mem.total = limit
            mem.available = min(mem.available, limit)
        if is_wsl():
            mem.notes.append(
                "WSL の中で動いている: メモリは WSL に割り当てた分(.wslconfig の memory)で判定した"
            )
    else:  # macOS はスワップが自動で伸びる
        with contextlib.suppress(Exception):
            mem.swap_total = int(psutil.swap_memory().total)
        mem.swap_auto = True
    return mem


# --- まとめ ----------------------------------------------------------------------------------------


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if os.name == "nt":  # platform.processor() は「Intel64 Family 6 ...」になるので製品名を読む
        winreg = _winreg()
        if winreg is not None:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
                ) as key:
                    return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
            except OSError:
                return name
        return name
    text = _read_text("/proc/cpuinfo")
    for line in (text or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return name


def _gb(v: int | None, digits: int = 2) -> float | None:
    return round(v / GIB, digits) if v is not None else None


WINDOWS_11_BUILD = 22000  # Windows 11 の最初のビルド(21H2)。これより前は Windows 10


def os_name(system: str | None = None, release: str | None = None, version: str | None = None) -> str:
    """OS の名前(例 "Windows 11"、"Linux 6.8.0")。Python 3.10 / 3.11 の platform.release() は Windows 11 でも
    "10" を返す(3.12 から直った)ので、Windows はビルド番号(platform.version() の 3 つ目、22000 以上で
    Windows 11)で決める。"""
    system = platform.system() if system is None else system
    release = platform.release() if release is None else release
    if system == "Windows":
        version = platform.version() if version is None else version
        parts = (version or "").split(".")
        if len(parts) >= 3 and parts[2].isdigit() and release in ("10", "11"):
            release = "11" if int(parts[2]) >= WINDOWS_11_BUILD else "10"
    return f"{system} {release}".strip()


def detect(
    disk_path: str | os.PathLike[str] | None = None,
    vram_gb: float | None = None,
    ram_gb: float | None = None,
    gpus: list[Gpu] | None = None,
    *,
    other_gpus: bool | None = None,
    memory: Memory | None = None,
) -> SystemInfo:
    """PC を調べる。vram_gb / ram_gb を渡すと、その値で上書きする(別の PC を想定した判定用)。

    gpus を渡すと nvidia-smi を使わない(試験・組み込み用)。other_gpus は NVIDIA 以外の GPU を調べるか:
    None(既定)は nvidia-smi で NVIDIA の GPU が見つからず、VRAM を上書きしていないときだけ調べる。
    memory を渡すとメモリを調べない(試験用)。"""
    notes: list[str] = []
    if gpus is None:
        found, gpu_notes = detect_gpus()
        notes += gpu_notes
    else:
        found = list(gpus)
    for g in found:
        notes += [f"{g.name}: {n}" for n in g.notes]
    usable_names = {g.name for g in found if g.backend == "cuda"}
    if len(usable_names) > 1:
        notes.append(
            "GPU が複数ある。判定は VRAM が最大の 1 枚で行った。ComfyUI は既定で CUDA の 0 番"
            "(CUDA_DEVICE_ORDER の"
            "既定 FASTEST_FIRST で速い順の先頭)を使うので、違う GPU で動くときは ComfyUI を --cuda-device で"
            "その GPU に向ける"
        )
    mem = memory or read_memory()
    notes += mem.notes

    overrides: list[str] = []
    if vram_gb is not None:
        overrides.append("vram")
        usable = [g for g in found if g.backend == "cuda"]
        if usable:  # この PC の GPU の世代のまま、VRAM だけを変える
            g = max(usable, key=lambda x: x.vram_total_gb)
            found = [
                Gpu(
                    name=g.name,
                    vendor=g.vendor,
                    vram_total_gb=vram_gb,
                    vram_free_gb=None,
                    driver=g.driver,
                    backend="cuda" if vram_gb > 0 else "none",
                    compute_capability=g.compute_capability,
                    compute_capability_source=g.compute_capability_source,
                    arch=g.arch,
                    torch_cuda=list(g.torch_cuda),
                    compute_dtype=g.compute_dtype,
                    bf16=g.bf16,
                    fp8=g.fp8,
                )
            ]
        elif vram_gb > 0:  # GPU が無い・動かない世代: 世代の分からない仮の GPU で判定する
            found = [
                Gpu(name=f"(仮定) {vram_gb:g} GB GPU", vendor="nvidia", vram_total_gb=vram_gb, backend="cuda")
            ]
        else:
            found = []

    adapters: list[Adapter] = []
    has_nvidia = any(g.vendor == "nvidia" for g in found)
    want_other = (
        other_gpus if other_gpus is not None else (gpus is None and not has_nvidia and vram_gb is None)
    )
    if want_other:
        adapters, other_notes = detect_other_gpus()
        notes += other_notes
        judged = any(g.backend == "cuda" for g in found)
        notes += _other_gpu_notes(adapters, judged)
    if platform.system() == "Darwin" and not found:
        notes.append("macOS の GPU(Apple)は v0.1 では判定しないので、CPU として判定した")

    ram_total = mem.total / GIB
    ram_avail = mem.available / GIB
    swap = dict(
        swap_total_gb=_gb(mem.swap_total),
        swap_max_gb=_gb(mem.swap_max),
        swap_auto=mem.swap_auto,
        commit_limit_gb=_gb(mem.commit_limit),
        commit_used_gb=_gb(mem.commit_used),
    )
    if ram_gb is not None:
        overrides.append("ram")
        ram_total = ram_avail = ram_gb
        swap = dict.fromkeys(swap)  # 別の PC を想定するので、この PC の仮想メモリは使わない
    disk_free = None
    with contextlib.suppress(OSError):
        disk_free = shutil.disk_usage(disk_path or Path.home()).free / GIB
    return SystemInfo(
        os=os_name(),
        cpu=_cpu_name(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        gpus=found,
        disk_free_gb=round(disk_free, 1) if disk_free is not None else None,
        overrides=overrides,
        other_gpus=adapters,
        other_gpus_checked=bool(want_other),
        notes=notes,
        **swap,
    )
