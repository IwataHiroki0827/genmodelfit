# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PC の検出。詳しい決まりと出典は docs/hardware.md。

- 判定に使う GPU は nvidia-smi で確かめた NVIDIA の GPU だけ(レジストリには取り外した GPU も残るため)。
  そのうち、今の PyTorch が動かない世代(Kepler 以前など)と、どの PyTorch も動かない古いドライバの GPU は
  backend を none にして CPU として判定する(理由は Gpu.notes と SystemInfo.notes)
- NVIDIA 以外の GPU(Windows は WMI + レジストリ、Linux は lspci + sysfs、macOS は system_profiler)は、
  NVIDIA の GPU が使えないときだけ、一次資料の対応表(AMD の ROCm・Intel の XPU・Apple の MPS)で
  ComfyUI + PyTorch が動く組み合わせなら判定に使う(0.7.0、#22)。
  対応しないものは理由を付けて CPU として判定する。
  NVIDIA の GPU が見つかったときは既定では調べない(PowerShell の起動に約 1 秒かかるため。
  `system --all-gpus` で調べる)
- メモリは物理メモリと仮想メモリ(Windows はページファイルの設定とコミット、Linux はスワップ)
- PC 名・ユーザ名・GPU の UUID・PCI の場所や個体の ID は読まない・残さない
- 失敗しても例外を出さない(分からない項目は None にして、理由を notes に書く)
"""

from __future__ import annotations

import configparser
import contextlib
import ctypes
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import psutil

from .i18n import tr
from .models import USABLE_BACKENDS, Adapter, Gpu, SystemInfo

GIB = 1024**3
MIB = 1024**2

NVSMI_TIMEOUT = 10.0  # nvidia-smi(普段は 0.1〜0.5 秒。ドライバが固まっているときに待ち続けない)
OTHER_GPU_TIMEOUT = 8.0  # PowerShell の WMI・lspci(PowerShell の起動だけで約 1 秒)

# nvidia-smi で聞く項目。読めなければ mig.mode.current を除いた QUERY_CC、compute_cap を知らない古い
# nvidia-smi には QUERY_OLD で聞き直す(知らない項目が 1 つあると問い合わせ全体が失敗するため。#101)
QUERY = (
    "name",
    "memory.total",
    "memory.free",
    "memory.used",
    "driver_version",
    "compute_cap",
    "mig.mode.current",
)
QUERY_CC = QUERY[:-1]
QUERY_OLD = ("name", "memory.total", "memory.free", "memory.used", "driver_version")
_OPTIONAL_LAST = "mig.mode.current"  # 行に無くても読む(QUERY_CC の行・試験の 6 項目の行)


# --- 外のコマンド・レジストリ(試験ではここを差し替える) -----------------------------------------


# 読むファイルの根(試験では一時ディレクトリの木に差し替える。/proc・/sys・/dev・/mnt・/etc・/opt を読むとき)
_ROOT = ""


def _p(path: str) -> str:
    return _ROOT + path


def _windows() -> bool:
    """Windows か(試験で Linux・macOS の読み方を Windows の上でも確かめるために差し替える)。"""
    return os.name == "nt"


def _kill_group(proc: subprocess.Popen) -> None:
    """時間切れのコマンドを止める。POSIX は組(セッション)ごと止める(孫のプロセスが出力の口を持ったまま
    残って待ち続けないように。#101)。Windows は子だけ止める(孫は残りうる: docs/hardware.md「既知の限界」)。"""
    with contextlib.suppress(OSError, ValueError):
        if os.name != "nt" and hasattr(os, "killpg"):
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()


def _run(args: list[str], timeout: float) -> tuple[int, str] | None:
    """コマンドを動かして (終了コード, 標準出力) を返す。見つからない・時間切れ・失敗は None。
    Windows では窓を出さない(GUI のアプリから呼ばれても黒い窓が開かないように)。
    時間切れのときは組ごと止め、さらに 1 秒だけ待って返す(出力の口が閉じなくても待ち続けない)。"""
    kw: dict = {}
    if os.name == "nt":
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    else:
        kw["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, **kw
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    try:
        out, _ = proc.communicate(timeout=timeout)
    except (subprocess.SubprocessError, OSError, ValueError):
        _kill_group(proc)
        # 口を閉じない(Windows では読みかけの口を閉じると、孫が終わるまで止まる)。
        # 読む糸は daemon なので残っても終了を妨げない
        with contextlib.suppress(subprocess.SubprocessError, OSError, ValueError):
            proc.communicate(timeout=1.0)
        return None
    return proc.returncode, (out or b"").decode("utf-8", errors="replace")


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
    # 業務用は「RTX 4000 Ada Generation」「RTX 4000 SFF Ada Generation」「RTX 2000E Ada Generation」
    (r"\brtx\s*40[5-9]0\b|\brtx\s*\d{4}e?\s*(?:sff\s*)?ada\b|\bl40s?\b|\bl4\b|\bl20\b", (8, 9)),
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


# NVIDIA の GPU 名の形(#101。docs/hardware.md「GPU 名の形」)。名前の表(_NAME_CC)は古いドライバのときの世代の
# 推定だけに使い、提出の GPU 名は表でなく形で見る(llmfit と同じく、断るのは名前の代わりの値と汎用の名前だけ。
# 形に合わない名前はメンテナが確かめる)。大文字小文字は区別しない。語は 10 個まで:
#   NAME := [NVIDIA] BRAND* CORE SUFFIX*
# 新しい系列の語が出たら、製品ごとではなく語を 1 つ足す
_NAME_BRANDS = frozenset(
    [
        "geforce",
        "quadro",
        "tesla",
        "grid",
        "titan",
        "rtx",
        "pro",
        "gtx",
        "gts",
        "gt",
        "mx",
        "cmp",
        "nvs",
        "fx",
        "dgx",
    ]
)
# 型番の頭の文字は NVIDIA の系列の頭だけ(A100・B200・H100・L40・T4・P100・V100・K80・M60・GB10・GH200・GA102・
# GP100・GV100・GK110・GM200、vGPU の RTX6000-24Q)。「TY1985」「KEN1990」「tom99」は型番にならない。
# 後ろのハイフンの部分は載せ方・大きさ・vGPU の型(-SXM2・-PCIE・-DGXS・-FHHL・-16GB・-40C・-Ada・-100・-3e)
_NAME_CORE = re.compile(
    r"(?:a|b|h|l|t|p|v|k|m|mx|gb|gh|ga|gp|gv|gk|gm|rtx)?\d{1,4}[a-z]{0,3}"
    r"(?:-(?:sxm\d?|pcie|dgxs|fhhl|ada|\d{1,3}gb|\d{1,3}[abcqe]|\d{1,3}))*"
)
_NAME_TITAN_CORE = frozenset({"v", "x", "xp", "z", "black", "rtx"})  # TITAN の直後だけ
_NAME_SUFFIX_WORDS = frozenset(
    [
        "ti",
        "super",
        "d",
        "v2",
        "ultra",
        "gtx",
        "gts",
        "gt",
        "gx2",
        "oem",
        "laptop",
        "gpu",
        "max-q",
        "with",
        "design",
        "workstation",
        "server",
        "edition",
        "generation",
        "ada",
        "blackwell",
        "embedded",
        "sff",
        "active",
        "pcie",
        "nvl",
        "a",
        "ac",
    ]
)
# 数字だけの語(「RTX 4090 A 1985」の 1985)は後ろの語にしない。括弧は世代の名前だけ(「(Pascal)」)
_NAME_SUFFIX = re.compile(
    r"sxm\d?|hbm\d+e?|\d{1,4}(?:gb|mb)|\((?:pascal|maxwell|kepler|volta|turing|ampere|ada|hopper|blackwell)\)"
)
_NAME_MAX_WORDS = 10


def _suffixes_ok(words: list[str]) -> bool:
    if len(set(words)) != len(words):  # 同じ語の繰り返し(「Ti Ti」「8GB 8GB」)は実在しない
        return False
    return all(w in _NAME_SUFFIX_WORDS or _NAME_SUFFIX.fullmatch(w) for w in words)


def nvidia_name_well_formed(name: str | None) -> bool:
    """NVIDIA の GPU 名が名前の形(上の文法)に合うか(#101)。「NVIDIA GeForce RTX 4090 D」
    「NVIDIA TITAN X (Pascal)」「GRID A100-40C」「NVIDIA A800 40GB Active」は合い、
    「NVIDIA GeForce RTX 3070 taro yamada」「RTX 4090 ABC123 XYZ99」は合わない。合わない名前も提出を断らない
    (contrib の確認 gpu_name_form)。"""
    if not isinstance(name, str):
        return False
    n = name.lower()
    for mark in ("(tm)", "(r)", "™", "®"):
        n = n.replace(mark, " ")
    words = n.split()
    if words[:1] == ["nvidia"]:
        words = words[1:]
    if not words or len(words) > _NAME_MAX_WORDS:
        return False
    for i, w in enumerate(words):
        head = words[:i]
        if not all(x in _NAME_BRANDS for x in head):
            break
        core = _NAME_CORE.fullmatch(w) or (head[-1:] == ["titan"] and w in _NAME_TITAN_CORE)
        if core and _suffixes_ok(words[i + 1 :]):
            return True
    # 「GeForce GTX TITAN」(型番の語の無い初代の TITAN)
    return all(x in _NAME_BRANDS for x in words) and words[-1] == "titan"


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
    # cu128 の aarch64 は v2.11.0 の .ci/manywheel/build_cuda.sh(12.8 に 12.0 を足し、aarch64 は 8.0 未満と
    # 8.6 を除く)。download.pytorch.org/whl/cu128/torch/ に 2.7.0〜2.11.0 の manylinux_2_28_aarch64 がある
    # (2026-10-27 に確かめた。DGX Spark の GB10 などの道。#101)
    CudaBuild(
        "cu128",
        "12.8",
        {"x86_64": _CU13_X86, "aarch64": frozenset({80, 90, 100, 120})},
        _CU12_WIN,
        _CU12_LINUX,
        "2.7〜2.11",
    ),
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
    """CPU の種類("x86_64" / "aarch64"。ほかは小文字の生の値で、表に無いので CUDA 版が無いと判定する。
    #101)。"""
    m = platform.machine().lower()
    if m in ("aarch64", "arm64"):
        return "aarch64"
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    return m or "unknown"


def torch_platform(windows: bool, machine: str) -> str:
    """PyTorch の CUDA 版の表(TORCH_CUDA_BUILDS の ccs)の鍵。表は (OS, 機械) で引く: Windows の Arm64 向けの
    CUDA 版は無いので、Windows の x86_64 以外は表に無い鍵("windows-aarch64" など)にする(#101)。"""
    return machine if not windows or machine == "x86_64" else f"windows-{machine}"


def python_is_32bit() -> bool:
    return struct.calcsize("P") == 4


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
    if not any(b.ccs.get(machine) for b in TORCH_CUDA_BUILDS):
        notes.append(
            tr(
                f"この機械({machine})向けの PyTorch の CUDA 版は無いので、CPU として判定した",
                f"There is no CUDA build of PyTorch for this machine ({machine}), so it is judged as CPU",
            )
        )
        return Support([], [], False, notes)
    if cc is None:
        arch_builds = [b for b in TORCH_CUDA_BUILDS if b.ccs.get(machine)]
        notes.append(
            tr(
                "GPU の世代(compute capability)が分からないので、世代では断っていない",
                "The GPU generation (compute capability) is unknown, so it is not refused by generation",
            )
        )
    else:
        arch_builds = [b for b in TORCH_CUDA_BUILDS if any(_code_runs(c, cc) for c in b.ccs.get(machine, ()))]
        newest = max(c for b in TORCH_CUDA_BUILDS for c in b.ccs.get(machine, ()))
        if not arch_builds and cc[0] * 10 + cc[1] > newest:
            # 表より新しい世代は断らない(世代が分からないときと同じ。断るのは表より古い世代だけ。#101)
            notes.append(
                tr(
                    f"compute capability {cc[0]}.{cc[1]} は PyTorch の CUDA 版の表"
                    f"(PyTorch {TORCH_RELEASE}"
                    " まで)より新しい世代なので、断っていない。最新の PyTorch を使う",
                    f"Compute capability {cc[0]}.{cc[1]} is newer than the table of PyTorch CUDA builds"
                    f" (up to PyTorch {TORCH_RELEASE}), so it is not refused. Use the latest PyTorch",
                )
            )
            return Support([], [], True, notes)
        if not arch_builds:
            arch = arch_of(cc)
            label = {"fermi": "Fermi 以前の", "kepler": "Kepler ", "unknown": "この"}.get(
                arch, arch.capitalize() + " "
            )
            label_en = {"fermi": "Fermi or older", "kepler": "Kepler", "unknown": "This"}.get(
                arch, arch.capitalize()
            )
            notes.append(
                tr(
                    f"{label}世代(compute capability {cc[0]}.{cc[1]})は PyTorch の CUDA 版のビルドが"
                    "どれも対応しないので、CPU として判定した(ドライバを更新しても GPU では動かない)",
                    f"{label_en} generation (compute capability {cc[0]}.{cc[1]}) is supported by no CUDA "
                    "build"
                    " of PyTorch, so it is judged as CPU (updating the driver does not make it run on the "
                    "GPU)",
                )
            )
            return Support([], [], False, notes)
    ver = parse_version(driver)
    if ver is None:
        notes.append(
            tr(
                "ドライバの版が読めないので、ドライバでは断っていない",
                "The driver version cannot be read, so it is not refused by driver",
            )
        )
        return Support([b.tag for b in arch_builds], [b.tag for b in arch_builds], True, notes)

    floor = ARCH_MIN_DRIVER.get(arch_of(cc)) if cc else None
    if floor and ver < floor[0]:
        notes.append(
            tr(
                f"ドライバ {driver} はこの世代({arch_of(cc).capitalize()})に対応しない"
                f"({_fmt_ver(floor[0])} 以上。"
                f"{floor[1]})ので、CPU として判定した(ドライバを更新すれば GPU で動く)",
                f"Driver {driver} does not support this generation ({arch_of(cc).capitalize()})"
                f" ({_fmt_ver(floor[0])} or later; {floor[1].replace('(', ' (')}),"
                " so it is judged as CPU (it runs on the GPU after a driver update)",
            )
        )
        return Support([], [b.tag for b in arch_builds], False, notes)

    def need(b: CudaBuild) -> tuple[int, ...]:
        return b.min_driver_windows if windows_driver else b.min_driver_linux

    ok = [b for b in arch_builds if ver >= need(b)]
    if not ok:
        easiest = min(arch_builds, key=need)
        notes.append(
            tr(
                f"ドライバ {driver} は古く、この GPU で動く PyTorch(CUDA {easiest.cuda} 版など)には "
                f"{_fmt_ver(need(easiest))} 以上が要るので、CPU として判定した(ドライバを更新すれば GPU "
                "で動く)",
                f"Driver {driver} is too old: PyTorch for this GPU (CUDA {easiest.cuda} build etc.) needs "
                f"{_fmt_ver(need(easiest))} or later, so it is judged as CPU (it runs on the GPU after a "
                "driver update)",
            )
        )
        return Support([], [b.tag for b in arch_builds], False, notes)
    if len(ok) < len(arch_builds):
        bad = [b for b in arch_builds if b not in ok]
        tags = tr("・", ", ").join(b.tag for b in reversed(bad))
        notes.append(
            tr(
                f"ドライバ {driver} では CUDA {bad[-1].cuda} 版以降の PyTorch({tags})は動かない"
                f"({_fmt_ver(min(need(b) for b in bad))} 以上が要る)。CUDA {ok[0].cuda} 版({ok[0].tag})"
                "なら動く",
                f"With driver {driver}, PyTorch builds for CUDA {bad[-1].cuda} and later ({tags}) do not run"
                f" ({_fmt_ver(min(need(b) for b in bad))} or later needed). The CUDA {ok[0].cuda} build "
                f"({ok[0].tag}) runs",
            )
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
        n_ok = {len(fields)} | ({len(fields) - 1} if fields[-1:] == (_OPTIONAL_LAST,) else set())
        parts = [p.strip() for p in line.split(", ")]
        if len(parts) not in n_ok:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) not in n_ok:
                continue

        def get(key: str, parts: list[str] = parts) -> str:
            return parts[idx[key]] if key in idx and idx[key] < len(parts) else ""

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
            notes.append(_no_vram_note())
        if (_clean(get(_OPTIONAL_LAST)) or "").lower() == "enabled":
            # MIG の区画は nvidia-smi の総量より小さい。実機で確かめていないので値は変えない(#101)
            notes.append(
                tr(
                    "MIG が有効: ComfyUI が使えるのは区画(MIG"
                    " のインスタンス)の大きさで、nvidia-smi の VRAM は GPU "
                    "全体の量。区画の大きさを --vram で指定する",
                    "MIG is enabled: ComfyUI can use only the size of its partition (MIG instance), while "
                    "nvidia-smi reports the whole GPU. Give the partition size with --vram",
                )
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


def _no_vram_note() -> str:
    return tr(
        "nvidia-smi が VRAM の量を返さない(メモリを CPU と共有する GPU など)ので、判定に使わない",
        "nvidia-smi returns no VRAM amount (e.g. a GPU sharing memory with the CPU), so it is not used",
    )


# DGX Spark(GB10)。nvidia-smi の名前がこれと完全に一致するときだけ、統合メモリの GPU として判定に使う
# (#101。docs/hardware.md「DGX Spark(GB10)」。ASUS・Dell・MSI などの GB10 機も同じ名前)
GB10_NAME = "NVIDIA GB10"
_JETSON = re.compile(r"orin|thor|xavier|nvgpu|tegra|jetson", re.I)


def apply_gb10(g: Gpu, *, system: str, machine: str, meminfo: dict[str, int] | None) -> bool:
    """VRAM の量を返さない NVIDIA の GPU のうち、DGX Spark(GB10)を統合メモリの GPU にする(#101)。
    条件: 名前が GB10_NAME と完全に一致・memory.total が [N/A] / Not Supported・Linux の aarch64。
    総量は /proc/meminfo の MemTotal(割り引かない。メインメモリと共有で、二重に数えない)、空きは
    NVIDIA の known-issues の例どおり MemAvailable + SwapFree(HugePages を使っていれば空きの HugePages)で、
    今の空きの注意(vram_free_short)だけに使う。ほかの共有メモリの NVIDIA は理由を書いて使わない。"""
    if g.vram_total_gb > 0:
        return False
    with contextlib.suppress(ValueError):
        g.notes.remove(_no_vram_note())
    if g.name.strip() != GB10_NAME:
        if _JETSON.search(g.name):
            g.notes.append(
                tr(
                    "Jetson は PyTorch が JetPack 専用の版(jp6"
                    " の cu126 など)で、ここの PyTorch の CUDA 版の表と"
                    "別の系統。メモリの扱いも確かめていないので判定に使わない(対象外。CPU として判定した)",
                    "Jetson uses JetPack-only PyTorch builds (jp6"
                    " cu126 etc.), separate from the PyTorch CUDA "
                    "build table here, and its memory handling is"
                    " unverified, so it is not used (out of scope;"
                    " judged as CPU)",
                )
            )
        else:
            g.notes.append(
                tr(
                    "nvidia-smi が VRAM の量を返さない(メモリを"
                    " CPU と共有する GPU など)。共有メモリの NVIDIA の"
                    " GPU は、ComfyUI の公式の手順を確かめた DGX Spark(GB10)だけを判定に使う",
                    "nvidia-smi returns no VRAM amount (e.g. a GPU"
                    " sharing memory with the CPU). Of the NVIDIA"
                    " GPUs with shared memory, only DGX Spark (GB10), which has an official ComfyUI guide, is"
                    " used",
                )
            )
        return False
    if system != "Linux" or machine != "aarch64":
        g.notes.append(
            tr(
                "GB10 は Linux の aarch64(DGX OS など)でだけ判定に使う",
                "GB10 is used only on aarch64 Linux (DGX OS etc.)",
            )
        )
        return False
    total = (meminfo or {}).get("MemTotal")
    if not total:
        g.notes.append(
            tr(
                "GB10 の統合メモリの量(/proc/meminfo の MemTotal)が読めないので、判定に使わない",
                "The unified memory of GB10 (MemTotal in /proc/meminfo) cannot be read, so it is not used",
            )
        )
        return False
    free = unified_free(meminfo or {})
    g.vram_total_gb = round(total / GIB, 2)
    g.vram_free_gb = round(free / GIB, 2) if free is not None else None
    g.vram_used_gb = None
    g.backend = "cuda"
    g.memory = "shared"
    g.carveout_gb = 0.0
    g.notes += [
        tr(
            "DGX Spark(GB10)は統合メモリ: メインメモリ"
            f" {g.vram_total_gb:g} GB(MemTotal)を GPU と共有するとみなした"
            "(GPU だけの上限は NVIDIA"
            " の文書に無いので割り引かない。逃がす動かし方は無い)。推定なので段階は good まで",
            "DGX Spark (GB10) has unified memory: the GPU is"
            f" taken to share the {g.vram_total_gb:g} GB main memory"
            " (MemTotal) (NVIDIA documents no GPU-only limit, so"
            " it is not discounted; there is no offload mode)."
            " Estimated, so the level is capped at good",
        ),
        tr(
            "sm_121(GB10)では CUTLASS・xformers の"
            " fmha のカーネルが失敗した報告があり、直ったかは確かめていない",
            "CUTLASS / xformers fmha kernels were reported to fail on sm_121 (GB10); whether this is fixed is"
            " unverified",
        ),
    ]
    return True


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
                tr(
                    f"nvidia-smi が世代を返さないので、GPU 名から compute capability {cc[0]}.{cc[1]} "
                    "と推した",
                    f"nvidia-smi returns no generation, so compute capability {cc[0]}.{cc[1]} was inferred "
                    "from the GPU name",
                )
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
            tr(
                "ComfyUI はこの GPU では半精度(fp16)を使わず fp32 で計算するので、"
                "作業用メモリを 2 倍で見積もる(遅い)",
                "ComfyUI computes in fp32 on this GPU (no half precision, fp16),"
                " so the working memory is doubled in the estimate (slow)",
            )
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
            tr(
                f"nvidia-smi が {bad} 枚の GPU の情報を読めなかった(ドライバ・GPU の不調)。"
                "読めた GPU だけで判定した",
                f"nvidia-smi could not read {bad} GPU(s) (driver or GPU trouble)."
                " Judged with the GPUs that could be read",
            )
        ]
    if code != 0:
        return [
            tr(
                f"nvidia-smi が終了コード {code} を返した。読めた GPU だけで判定した",
                f"nvidia-smi returned exit code {code}. Judged with the GPUs that could be read",
            )
        ]
    return []


def query_nvidia_smi(exe: str) -> tuple[list[Gpu], list[str]]:
    """nvidia-smi に聞く。読めなければ compute_cap を除いた QUERY_OLD で聞き直す
    (compute_cap を知らない古い nvidia-smi は「not a valid field」で失敗する)。(GPU, 分かったこと)。"""
    first = _ask(exe, QUERY)
    if first is None:
        return [], [
            tr(
                "nvidia-smi が動かない・時間内に終わらないので、GPU は無いものとして判定した",
                "nvidia-smi does not run or did not finish in time, so judged as having no GPU",
            )
        ]
    gpus = parse_nvidia_smi(first[1], QUERY)
    if gpus:  # 1 枚だけ読めない(「Unable to determine the device handle」)ときも、読めた行は使う
        return gpus, _partial_notes(first)
    # mig.mode.current を知らない nvidia-smi(「Field "mig.mode.current" is not a valid field」と名指しする)
    middle = _ask(exe, QUERY_CC) if _OPTIONAL_LAST in first[1] else None
    if middle is not None:
        gpus = parse_nvidia_smi(middle[1], QUERY_CC)
        if gpus:
            return gpus, _partial_notes(middle)
    second = _ask(exe, QUERY_OLD)
    if second is not None:
        gpus = parse_nvidia_smi(second[1], QUERY_OLD)
        if gpus:
            return gpus, [
                tr(
                    "古い nvidia-smi で compute_cap を聞けなかったので、GPU 名から世代を推した",
                    "An old nvidia-smi could not report compute_cap, so the generation was inferred from the "
                    "GPU name",
                ),
                *_partial_notes(second),
            ]
    text = first[1] + (middle[1] if middle else "") + (second[1] if second else "")
    if "No devices were found" in text:
        return [], [
            tr("nvidia-smi は動いたが NVIDIA の GPU が見つからない", "nvidia-smi ran but found no NVIDIA GPU")
        ]
    if first[0] != 0:
        return [], [
            tr(
                f"nvidia-smi が失敗した(終了コード {first[0]})。ドライバが動いていない可能性があるので、"
                "GPU は無いものとして判定した",
                f"nvidia-smi failed (exit code {first[0]}). The driver may not be running,"
                " so judged as having no GPU",
            )
        ]
    return [], [
        tr(
            "nvidia-smi の出力が読めないので、GPU は無いものとして判定した",
            "The nvidia-smi output cannot be read, so judged as having no GPU",
        )
    ]


def detect_gpus(
    *, windows_driver: bool | None = None, comfy_windows: bool | None = None
) -> tuple[list[Gpu], list[str]]:
    """nvidia-smi で NVIDIA の GPU を調べ、世代・ドライバ・精度を決める。(GPU, 分かったこと)。"""
    exe = find_nvidia_smi()
    if not exe:
        return [], [
            tr(
                "nvidia-smi が見つからない(NVIDIA のドライバが入っていない)ので、GPU は無いものとして判定し"
                "た",
                "nvidia-smi not found (no NVIDIA driver installed), so judged as having no GPU",
            )
        ]
    gpus, notes = query_nvidia_smi(exe)
    wd = (_windows() or is_wsl()) if windows_driver is None else windows_driver
    cw = (_windows()) if comfy_windows is None else comfy_windows
    machine = _machine()
    system = platform.system()
    meminfo = read_meminfo() if any(g.vram_total_gb <= 0 for g in gpus) else None
    for g in gpus:
        apply_gb10(g, system=system, machine=machine, meminfo=meminfo)
        classify_gpu(g, windows_driver=wd, comfy_windows=cw, machine=torch_platform(_windows(), machine))
    return gpus, notes


# --- NVIDIA 以外の GPU(知らせるだけ) --------------------------------------------------------------

_VENDOR_IDS = {"10de": "nvidia", "1002": "amd", "1022": "amd", "8086": "intel"}
# WMI の AdapterRAM は 32 ビットなので、4 GiB 以上の GPU はこの値(またはそれに近い値)で頭打ちになる
_ADAPTER_RAM_CAP = 0xFFF00000
_MIN_PLAUSIBLE_VRAM = 64 * MIB  # これより小さい「VRAM」は単位の違い(MB で書いたドライバ)とみなして使わない

_PS_WMI = (
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "$a=@(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | "
    "Select-Object Name,AdapterRAM,PNPDeviceID,ConfigManagerErrorCode,DriverVersion);"
    "ConvertTo-Json -InputObject $a -Compress"
)
# WSL2 から Windows 側の GPU を読む台本(#101)。WMI は _PS_WMI と同じ項目、レジストリは画面の部品のクラスキー
# (0000, 0001 …)の DriverDesc・MatchingDeviceId・qwMemorySize・MemorySize(REG_BINARY はバイト列から変換)
# だけを読み、{"video": [...], "registry": [...]} の JSON で返す。PC 名などは読まない
_PS_WSL = (
    "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
    "$v=@(Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop | "
    "Select-Object Name,AdapterRAM,PNPDeviceID,ConfigManagerErrorCode,DriverVersion);"
    "$k='HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}';"
    "$r=@(Get-ChildItem -LiteralPath $k -ErrorAction SilentlyContinue | "
    "Where-Object {$_.PSChildName -match '^[0-9]{4}$'} | ForEach-Object {"
    "$q=Get-ItemProperty -LiteralPath $_.PSPath -ErrorAction SilentlyContinue;"
    "$m=$q.'HardwareInformation.MemorySize';"
    "if($m -is [byte[]]){if($m.Length -ge 4){$m=[BitConverter]::ToUInt32($m,0)}else{$m=$null}};"
    "[pscustomobject]@{DriverDesc=$q.DriverDesc;MatchingDeviceId=$q.MatchingDeviceId;"
    "'HardwareInformation.qwMemorySize'=$q.'HardwareInformation.qwMemorySize';"
    "'HardwareInformation.MemorySize'=$m}});"
    "ConvertTo-Json -InputObject @{video=$v;registry=$r} -Compress -Depth 3"
)
# デバイス マネージャーのエラー コード(この 3 つの部品は判定に使わない: REFUSED_CM_ERRORS。
# Microsoft「Device Manager error messages」
# https://learn.microsoft.com/en-us/windows-hardware/drivers/install/device-manager-error-messages)
_CM_ERRORS = {
    22: ("無効にしてある(デバイス マネージャーで有効にする)", "is disabled (enable it in Device Manager)"),
    28: ("ドライバが入っていない", "has no driver installed"),
    43: (
        "問題を報告して Windows が止めた(ドライバ・GPU の不調)",
        "was stopped by Windows because it reported problems (driver or GPU trouble)",
    ),
}


REFUSED_CM_ERRORS = frozenset(_CM_ERRORS)


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
    except (ValueError, RecursionError):
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
        # ドライバの無い NVIDIA の GPU は、名前(言語で変わる「Microsoft 基本ディスプレイ アダプター」)に
        # 頼らず _other_gpu_notes が知らせる(#101)
        # AMD・Intel の部品の 22・28・43 は classify_adapter が理由にして断る(注意を 2 回出さない)
        refused_later = vendor in ("amd", "intel") and isinstance(err, int) and err in REFUSED_CM_ERRORS
        if isinstance(err, int) and err != 0 and not refused_later:
            ja, en = _CM_ERRORS.get(err, ("問題がある状態", "has a problem"))
            notes.append(
                tr(
                    f"{name} は Windows のデバイス マネージャーで{ja}(コード {err})",
                    f"{name} {en} in Windows Device Manager (code {err})",
                )
            )
        driver = d.get("DriverVersion")
        code = err if isinstance(err, int) and not isinstance(err, bool) and err != 0 else None
        ids = re.search(r"ven_([0-9a-f]{4})&dev_([0-9a-f]{4})", hwid)
        out.append(
            Adapter(
                name=name,
                vendor=vendor,
                vram_gb=round(vram / GIB, 2) if vram else None,
                vram_source=src,
                source="wmi",
                pci_id=f"{ids.group(1)}:{ids.group(2)}" if ids else None,
                driver=driver.strip() if isinstance(driver, str) and driver.strip() else None,
                config_error=code,
            )
        )
    return out, notes


_LSPCI_LINE = re.compile(
    r"^(?P<slot>[0-9a-f:.]+)\s+(?P<cls>[^\[]*?)\s*\[(?P<code>03[0-9a-f]{2})\]:\s*(?P<rest>.*)$", re.I
)


def parse_lspci(text: str, vram_of=None, gtt_of=None) -> list[Adapter]:
    """`lspci -Dnn` の出力から画面の部品(クラス 03xx)を読む。vram_of(slot) で sysfs の VRAM を、
    gtt_of(slot) で amdgpu の GTT(メインメモリから使える分)を引ける。
    差し場所(slot)は sysfs を引くのに使うだけで残さない。"""
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
        if desc == "Device" and ids:
            # pci.ids が古いと、lspci -nn は「<会社> Device [1002:7551]」と出す(pciutils lib/names.c の
            # format_name_pair)。型番の ID を名前に残し(「AMD Device 7551」)、classify_adapter が
            # PCI_DEVICE_NAMES で引く。型番の ID は製品の種類を表すだけで、個体・差し場所は表さない
            desc = f"Device {ids[-1].group(2).lower()}"
        product = re.search(r"\[([^\]]+)\]\s*$", desc)  # "GA104 [GeForce RTX 3070]" の括弧の中が製品名
        label = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}.get(vendor, "")
        prod = product.group(1) if product else desc
        if label and prod.lower().startswith(label.lower() + " "):  # 「Meteor Lake-P [Intel Arc Graphics]」
            label = ""
        name = " ".join(x for x in (label, prod) if x) or "GPU"
        vram = vram_of(m.group("slot")) if vram_of else None
        gtt = gtt_of(m.group("slot")) if gtt_of else None
        out.append(
            Adapter(
                name=name,
                vendor=vendor,
                vram_gb=round(vram / GIB, 2) if vram else None,
                vram_source="sysfs" if vram else None,
                source="lspci",
                shared_gb=round(gtt / GIB, 2) if gtt else None,
                pci_id=f"{ids[-1].group(1).lower()}:{ids[-1].group(2).lower()}" if ids else None,
            )
        )
    return out


_SLOT = re.compile(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", re.I)


def _sysfs_text(slot: str, leaf: str) -> str | None:
    """sysfs の PCI の部品の値。差し場所(slot)は形を確かめて読むのに使うだけで、残さない・出さない。"""
    if not _SLOT.fullmatch(slot):
        return None
    text = _read_text(_p(f"/sys/bus/pci/devices/{slot}/{leaf}"))
    return text.strip() if text is not None else None


def _sysfs_value(slot: str, leaf: str) -> int | None:
    try:
        v = int(_sysfs_text(slot, leaf) or "")
    except ValueError:
        return None
    return v if v >= _MIN_PLAUSIBLE_VRAM else None


def _sysfs_vram(slot: str) -> int | None:
    """Linux の amdgpu が出す VRAM の量(mem_info_vram_total。APU は BIOS で取り分けた分)。"""
    return _sysfs_value(slot, "mem_info_vram_total")


def _sysfs_gtt(slot: str) -> int | None:
    """Linux の amdgpu の GTT(GPU がメインメモリから使える分。mem_info_gtt_total)。"""
    return _sysfs_value(slot, "mem_info_gtt_total")


def _sysfs_slots() -> list[str]:
    """/sys/bus/pci/devices の部品の差し場所(試験では conftest が塞ぐ)。読めなければ空。"""
    try:
        names = sorted(os.listdir(_p("/sys/bus/pci/devices")))
    except OSError:
        return []
    return [n for n in names if _SLOT.fullmatch(n)]


def sysfs_display_adapters() -> list[Adapter]:
    """lspci が無いときの画面の部品(#101): /sys/bus/pci/devices/*/class が 0x03xxxx のもの。名前は lspci の
    pci.ids が古いときと同じ「<会社> Device <型番の ID>」にし、classify_adapter が PCI_DEVICE_NAMES で引く。
    差し場所は読むのに使うだけで Adapter に入れない。"""
    out: list[Adapter] = []
    for slot in _sysfs_slots():
        cls = (_sysfs_text(slot, "class") or "").lower()
        ven = (_sysfs_text(slot, "vendor") or "").lower().removeprefix("0x")
        dev = (_sysfs_text(slot, "device") or "").lower().removeprefix("0x")
        if (
            not re.fullmatch(r"0x03[0-9a-f]{4}", cls)
            or not re.fullmatch(r"[0-9a-f]{4}", ven + "")
            or not re.fullmatch(r"[0-9a-f]{4}", dev)
        ):
            continue
        vendor = _VENDOR_IDS.get(ven, "unknown")
        label = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel"}.get(vendor)
        vram, gtt = _sysfs_vram(slot), _sysfs_gtt(slot)
        out.append(
            Adapter(
                name=f"{label} Device {dev}" if label else f"Device {ven}:{dev}",
                vendor=vendor,
                vram_gb=round(vram / GIB, 2) if vram else None,
                vram_source="sysfs" if vram else None,
                source="sysfs",
                shared_gb=round(gtt / GIB, 2) if gtt else None,
                pci_id=f"{ven}:{dev}",
            )
        )
    return out


_KFD_NODES = "/sys/class/kfd/kfd/topology/nodes"


def gfx_from_target_version(v: int) -> str | None:
    """amdkfd の gfx_target_version(10 進で 主×10000 + 下位×100 + stepping)→ gfx の名前。stepping は
    16 進 1 文字(90010 → gfx90a、110000 → gfx1100、120001 → gfx1201)。0(CPU のノード)は None。"""
    if v <= 0:
        return None
    major, minor, step = v // 10000, (v // 100) % 100, v % 100
    if minor > 9 or step > 15:
        return None
    return f"gfx{major}{minor}{step:x}"


def kfd_gfx_targets() -> dict[str, str]:
    """Linux の amdkfd が出す GPU の世代(#101): /sys/class/kfd/kfd/topology/nodes/*/properties の
    gfx_target_version を、ノードの vendor_id・device_id(10 進)で PCI の型番 ID("1002:744c")に当てる。
    amdkfd(drivers/gpu/drm/amd/amdkfd/kfd_topology.c)は amdgpu の GPU にだけ出す。読めなければ空。"""
    out: dict[str, str] = {}
    try:
        nodes = sorted(os.listdir(_p(_KFD_NODES)))
    except OSError:
        return out
    for node in nodes:
        if not node.isdigit():
            continue
        props: dict[str, int] = {}
        for line in (_read_text(_p(f"{_KFD_NODES}/{node}/properties")) or "").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1].isascii() and parts[1].isdigit():
                props[parts[0]] = int(parts[1])
        gfx = gfx_from_target_version(props.get("gfx_target_version", 0))
        ven, dev = props.get("vendor_id"), props.get("device_id")
        if gfx and ven and dev is not None and 0 <= ven < 0x10000 and 0 <= dev < 0x10000:
            out[f"{ven:04x}:{dev:04x}"] = gfx
    return out


# --- WSL2 の AMD・Intel(#101。docs/hardware.md「WSL2」) ---------------------------------------------


@dataclass(frozen=True)
class WslState:
    interop: bool  # Windows のプログラムを動かせるか(binfmt_misc の WSLInterop・WSLInterop-late)
    container: bool  # WSL の上のコンテナの中か(/.dockerenv・/run/.containerenv)


def _exists(path: str) -> bool:
    try:
        return os.path.exists(_p(path))
    except (OSError, ValueError):
        return False


def wsl_state() -> WslState | None:
    """WSL2 の中なら、相互運用とコンテナの状態(試験では conftest が塞ぐ)。WSL2 でなければ None。
    WSL2 = /dev/dxg(GPU の準仮想化の口)があり、カーネルの版に microsoft がある。プロセスは起こさない。"""
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        return None
    if not _exists("/dev/dxg"):
        return None
    return WslState(
        interop=_exists("/proc/sys/fs/binfmt_misc/WSLInterop")
        or _exists("/proc/sys/fs/binfmt_misc/WSLInterop-late"),
        container=_exists("/.dockerenv") or _exists("/run/.containerenv"),
    )


def _wsl_powershell() -> str | None:
    """WSL から呼ぶ powershell.exe。PATH に無ければ、wsl.conf の automount の根(既定 /mnt/)の下の
    C ドライブ。"""
    found = shutil.which("powershell.exe")
    if found:
        return found
    root = "/mnt/"
    text = _read_text(_p("/etc/wsl.conf"))
    if text:
        cp = configparser.ConfigParser(strict=False, interpolation=None)
        with contextlib.suppress(configparser.Error):
            cp.read_string(text)
            root = cp.get("automount", "root", fallback=root).strip().strip('"') or root
    path = root.rstrip("/") + "/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    return path if _exists(path) else None


def _find_tool(name: str, extra: tuple[str, ...] = ()) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for d in extra:
        if _exists(f"{d}/{name}"):
            return _p(f"{d}/{name}")
    return None


def parse_rocminfo(text: str) -> list[tuple[str, str]]:
    """rocminfo の GPU のエージェント([(gfx, Marketing Name)])。読むのは行の頭で照合した `Name: gfx…`・
    `Marketing Name:`・`Device Type:` だけで、ほかの行(UUID など)は読まない・残さない(#101)。"""
    out: list[tuple[str, str]] = []
    gfx = market = kind = None

    def flush() -> None:
        if gfx and kind == "GPU":
            out.append((gfx, market or ""))

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("*******") or line.startswith("Agent "):
            flush()
            gfx = market = kind = None
            continue
        m = re.match(r"^Name:\s+(gfx[0-9a-f]{3,5})\s*$", line)
        if m:
            gfx = m.group(1)
            continue
        m = re.match(r"^Marketing Name:\s+([A-Za-z0-9 ()._-]{1,64})\s*$", line)
        if m:
            market = m.group(1).strip()
            continue
        m = re.match(r"^Device Type:\s+(GPU|CPU|DSP)\s*$", line)
        if m:
            kind = m.group(1)
    flush()
    return out


def parse_clinfo_devices(text: str) -> list[str]:
    """`clinfo -l` の Intel の GPU の名前(「`-- Device #0: Intel(R) Arc(TM) A770 Graphics」の行)。"""
    out = []
    for line in text.splitlines():
        m = re.match(r"^\s*[`|+\\-]*\s*Device #\d+:\s*(.{1,80}?)\s*$", line)
        if (
            m
            and re.search(r"intel", m.group(1), re.I)
            and not re.search(r"\bcpu\b|core\(tm\) i|xeon", m.group(1), re.I)
        ):
            out.append(m.group(1))
    return out


def _wsl_note_unreadable(why_ja: str, why_en: str) -> str:
    return tr(
        f"WSL2 の中: {why_ja}ので、Windows 側の GPU(AMD・Intel)を読めない",
        f"Inside WSL2: {why_en}, so the Windows-side GPUs (AMD, Intel) cannot be read",
    )


def wsl_windows_adapters(state: WslState) -> tuple[list[Adapter], list[str]]:
    """WSL2 の中から、相互運用で Windows 側の GPU を読み(WMI + レジストリ。Windows と同じ読み方)、
    AMD は rocminfo、Intel は clinfo -l で WSL の中の準備を確かめる(#101)。失敗は例外にせず注意にする。"""
    if state.container:
        return [], [_wsl_note_unreadable("WSL の上のコンテナの中な", "this is a container on WSL")]
    if not state.interop:
        return [], [
            _wsl_note_unreadable(
                "Windows のプログラムを動かす相互運用(wsl.conf の [interop] enabled)が無効な",
                "interop for running Windows programs ([interop] enabled in wsl.conf) is disabled",
            )
        ]
    exe = _wsl_powershell()
    r = _run([exe, "-NoProfile", "-NonInteractive", "-Command", _PS_WSL], OTHER_GPU_TIMEOUT) if exe else None
    data = None
    if r is not None and r[0] == 0:
        with contextlib.suppress(ValueError, RecursionError):
            data = json.loads(r[1].strip().lstrip("\ufeff") or "null")
    if not isinstance(data, dict):
        return [], [
            _wsl_note_unreadable(
                "powershell.exe が無い・失敗した・時間内に終わらない",
                "powershell.exe is missing, failed or did not finish in time",
            )
        ]
    # 台本の出力の形を確かめる(項目の型が違えば読まない。例外にしない)
    reg_raw, video = data.get("registry"), data.get("video")
    registry = [e for e in reg_raw if isinstance(e, dict)] if isinstance(reg_raw, list) else []
    if isinstance(video, dict):
        video = [video]
    video = [v for v in video if isinstance(v, dict)] if isinstance(video, list) else []
    adapters, notes = parse_wmi_video(json.dumps(video), registry)
    for a in adapters:
        a.source = "wsl"
    if any(a.vendor == "amd" for a in adapters):
        tool = _find_tool("rocminfo", ("/opt/rocm/bin",))
        rr = _run([tool], OTHER_GPU_TIMEOUT) if tool else None
        agents = parse_rocminfo(rr[1]) if rr is not None and rr[0] == 0 else []
        for a in adapters:
            if a.vendor != "amd":
                continue
            a.wsl_runtime = bool(agents)
            mine = [g for g, mk in agents if mk and _norm_name(mk) == _norm_name(a.name)]
            by_name = amd_gfx(a.name)
            same = [g for g, _ in agents if by_name and g == by_name[0]]
            pick = mine or same or ([agents[0][0]] if len(agents) == 1 else [])
            a.gfx_target = pick[0] if pick else None
    if any(a.vendor == "intel" for a in adapters):
        tool = _find_tool("clinfo")
        rr = _run([tool, "-l"], OTHER_GPU_TIMEOUT) if tool else None
        ok = bool(parse_clinfo_devices(rr[1])) if rr is not None and rr[0] == 0 else False
        for a in adapters:
            if a.vendor == "intel":
                a.wsl_runtime = ok
    return adapters, notes


# Apple silicon のチップ名(system_profiler の sppci_model。例 "Apple M2 Pro")。bench の実測の GPU 名に使うのは
# この形に合うものだけ(チップ名以外の値・シリアルなどを入れない。0.9.0)
APPLE_CHIP_NAME = re.compile(r"Apple [MA]\d{1,2}(?: (?:Pro|Max|Ultra))?")


def apple_chip_name() -> str | None:
    """この PC の Apple silicon のチップ名(macOS の system_profiler で読む。ほかの OS・読めない・形が合わない
    ときは None)。bench が ComfyUI の MPS の GPU 名("mps")の代わりに使う。"""
    if platform.system() != "Darwin":
        return None
    r = _run(["system_profiler", "SPDisplaysDataType", "-json"], OTHER_GPU_TIMEOUT)
    if r is None or r[0] != 0:
        return None
    for a in parse_system_profiler(r[1]):
        name = (a.name or "").strip()
        if a.vendor == "apple" and APPLE_CHIP_NAME.fullmatch(name):
            return name
    return None


def local_pci_id(vendor: str, name: str) -> str | None:
    """この PC の、ComfyUI の GPU(会社と名前)の PCI の型番 ID("1002:744c")。bench が ComfyUI と同じ PC の
    ときだけ使う。AMD・Intel だけ(NVIDIA は名前がどこでも同じで、判定の側も名前で比べる)。Windows は WMI の
    PNPDeviceID の VEN・DEV、Linux は lspci -nn。名前が合う部品が 1 つならそれ、その会社の部品が 1 つだけなら
    それ。決められなければ None。型番 ID は機種の番号で、個体・差し場所(PCI のバス番号)は表さない。"""
    if vendor not in ("amd", "intel"):
        return None
    from .estimate import gpu_key

    adapters, _ = detect_other_gpus()
    cands = [a for a in adapters if a.vendor == vendor and a.pci_id]
    named = [a for a in cands if gpu_key(a.name) == gpu_key(name)]
    pick = named if len(named) == 1 else cands if len(cands) == 1 else []
    return pick[0].pci_id if pick else None


def parse_system_profiler(text: str) -> list[Adapter]:
    """macOS の `system_profiler SPDisplaysDataType -json` を読む。Apple silicon の GPU は名前が
    「Apple M2 Pro」などで、メモリは統合メモリ(大きさは psutil の搭載量 = sysctl hw.memsize で見る)。"""
    try:
        data = json.loads(text.strip() or "null")
    except (ValueError, RecursionError):
        return []
    items = data.get("SPDisplaysDataType") if isinstance(data, dict) else None
    out: list[Adapter] = []
    for d in items if isinstance(items, list) else []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("sppci_model") or d.get("_name") or "").strip()
        if not name:
            continue
        vendor_raw = str(d.get("spdisplays_vendor") or d.get("sppci_vendor") or "").lower()
        low = name.lower()
        if "apple" in vendor_raw or re.match(r"apple [ma]\d", low):
            vendor = "apple"
        elif "amd" in vendor_raw or "ati" in vendor_raw or "radeon" in low:
            vendor = "amd"
        elif "intel" in vendor_raw or "intel" in low:
            vendor = "intel"
        elif "nvidia" in vendor_raw or "nvidia" in low:
            vendor = "nvidia"
        else:
            vendor = "unknown"
        out.append(Adapter(name=name, vendor=vendor, source="system_profiler", integrated=vendor == "apple"))
    return out


def _unreadable(how: str) -> str:
    return tr(
        f"画面の部品の一覧を読めなかった({how} が無い・失敗した・時間内に終わらない)。"
        "NVIDIA 以外の GPU は判定に使っていない",
        f"The display adapter list could not be read ({how} is missing, failed or did not finish in time)."
        " GPUs other than NVIDIA are not used",
    )


def detect_other_gpus(*, wsl_windows: bool = True) -> tuple[list[Adapter], list[str]]:
    """NVIDIA 以外も含めた画面の部品。Windows は PowerShell の WMI + レジストリ、Linux は lspci(無ければ
    sysfs)+ sysfs・amdkfd、macOS は system_profiler。WSL2 では wsl_windows のとき Windows 側の GPU も読む
    (使える NVIDIA の GPU があるときは呼ぶ側が False にする)。読めなかったときは注意を返す(#101)。"""
    if _windows():
        r = _run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", _PS_WMI], OTHER_GPU_TIMEOUT)
        if r is None or r[0] != 0:
            return [], [_unreadable("PowerShell")]
        return parse_wmi_video(r[1], _reg_display_entries() or [])
    if platform.system() == "Linux":
        notes: list[str] = []
        r = _run(["lspci", "-Dnn"], OTHER_GPU_TIMEOUT)
        listed = r is not None and r[0] == 0
        if listed:
            adapters = parse_lspci(r[1], _sysfs_vram, _sysfs_gtt)
        else:
            listed = bool(_sysfs_slots())  # sysfs の一覧が読めれば、画面の部品が 0 個でも「読めた」
            adapters = sysfs_display_adapters()
            if adapters:
                notes.append(
                    tr(
                        "lspci が使えないので、画面の部品を sysfs(/sys/bus/pci/devices)から読んだ"
                        "(製品名は型番の ID から引く)",
                        "lspci is unavailable, so the display adapters were read from sysfs"
                        " (/sys/bus/pci/devices; product names come from the device IDs)",
                    )
                )
        kfd = kfd_gfx_targets()
        for a in adapters:
            if a.vendor == "amd" and a.pci_id in kfd:
                a.gfx_target = kfd[a.pci_id]
        state = wsl_state()
        if state is not None:
            # WSL2 の lspci・sysfs に見える部品は GPU の準仮想化の口(Microsoft 1414:008e)で、
            # 計算の GPU ではない
            adapters = [a for a in adapters if not (a.pci_id or "").startswith("1414:")]
            if wsl_windows:
                win, win_notes = wsl_windows_adapters(state)
                adapters += win
                notes += win_notes
        elif not listed:
            notes.append(_unreadable("lspci・sysfs"))
        return adapters, notes
    if platform.system() == "Darwin":
        r = _run(["system_profiler", "SPDisplaysDataType", "-json"], OTHER_GPU_TIMEOUT)
        if r is None or r[0] != 0:
            return [], [_unreadable("system_profiler")]
        return parse_system_profiler(r[1]), []
    return [], []


# --- NVIDIA 以外の GPU の対応表(docs/hardware.md「NVIDIA 以外の GPU」。2026-10-21 に一次資料で確かめた)
#
# ComfyUI(Comfy-Org/ComfyUI 426f5ee の README「AMD GPUs (Linux)」「AMD GPUs (Windows, ROCm 10.0)」
# 「Intel GPUs (Windows and Linux)」「Apple Mac silicon」)が案内する PyTorch で動く組み合わせだけを使う。
# 実測は 0 件なので、段階は推定で、印は「推定(<会社>・実測なし)」。段階は good までにする(estimate.py)。

# AMD: ComfyUI の Windows の案内は「RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4」(AMD の multi-arch 版 PyTorch、
# Windows 11)。TheRock の SUPPORTED_GPUS.md(4639f91)は Linux・Windows とも RDNA 1〜4 を Release Ready にして
# いるが、ComfyUI の案内に合わせて RDNA 2 以降(と Linux の Instinct)だけを使う。
# pytorch.org の ROCm 版(Linux だけ。ComfyUI の Linux の案内)は v2.14.1 の .ci/docker/manywheel/build.sh の
# PYTORCH_ROCM_ARCH の gfx だけを入れる。それ以外の RDNA 2(gfx1031〜1036 など)は TheRock の版を使う、
# と notes に書く
AMD_ARCH: dict[str, str] = {
    **dict.fromkeys(("gfx1200", "gfx1201"), "RDNA 4"),
    **dict.fromkeys(("gfx1150", "gfx1151", "gfx1152", "gfx1153"), "RDNA 3.5"),
    **dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1103"), "RDNA 3"),
    **dict.fromkeys(("gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036"), "RDNA 2"),
    **dict.fromkeys(("gfx1010", "gfx1011", "gfx1012"), "RDNA 1"),
    **dict.fromkeys(("gfx908", "gfx90a", "gfx942", "gfx950"), "CDNA"),
    **dict.fromkeys(("gfx900", "gfx906", "gfx90c"), "GCN 5"),
}
AMD_SUPPORTED_ARCH = ("RDNA 2", "RDNA 3", "RDNA 3.5", "RDNA 4")  # Windows・Linux
# 対象の世代でも使わない gfx: 画面を出すための 2 CU の内蔵 GPU(Raphael・Granite Ridge = Ryzen 7000 / 9000 の
# デスクトップ、Mendocino の 610M)。生成では CPU より速い見込みが無い(判断。実測なし)。
# Van Gogh(Steam Deck、gfx1033)は 8 CU(Valve の仕様 https://www.steamdeck.com/en/tech 「8 RDNA 2 CUs」)で、
# TheRock の SUPPORTED_GPUS.md が Linux・Windows とも Release Ready にしているので、ほかの RDNA 2 の内蔵 GPU
# (680M / 660M)と同じく GPU として使う(0.9.0 で直した。以前は 2 CU と誤ってここに入れていた)
AMD_DISPLAY_ONLY_GFX = frozenset({"gfx1036"})
AMD_LINUX_ONLY_ARCH = ("CDNA",)  # Instinct は Linux だけ(ComfyUI の Windows の案内は Radeon だけ)
PYTORCH_ROCM_LINUX_GFX = frozenset(
    {
        "gfx900",
        "gfx906",
        "gfx908",
        "gfx90a",
        "gfx942",
        "gfx950",
        "gfx1030",
        "gfx1100",
        "gfx1101",
        "gfx1102",
        "gfx1103",
        "gfx1200",
        "gfx1201",
        "gfx1150",
        "gfx1151",
    }
)
# RDNA 2 以前は bf16 を持たない。ComfyUI は fp16 を持たない系統でも bf16 で計算する(should_use_bf16 の
# manual_cast。model_management.py の AMD_RDNA2_AND_OLDER_ARCH)ので、作業用メモリは倍にならない。VAE は fp32
AMD_NO_BF16_ARCH = ("RDNA 2", "RDNA 1", "GCN 5")

# AMD の GPU 名(Windows の WMI・Linux の lspci の製品名・開発名)→ (gfx, 内蔵か)。上から順に試す。
# 製品名と gfx は TheRock の RELEASES.md「GFX target lookup table」(4639f91)による
_AMD_NAMES: tuple[tuple[str, str, bool], ...] = (
    (r"rx\s*9070|ai pro r9[67]00|navi\s*48", "gfx1201", False),
    (r"rx\s*90[456]0|navi\s*44", "gfx1200", False),  # RX 9060 / 9060 XT は Navi 44
    (r"8060s|8050s|8040s|strix halo", "gfx1151", True),
    (r"890m|880m|strix(?! halo)", "gfx1150", True),
    (r"860m|840m|krackan", "gfx1152", True),
    (r"820m", "gfx1153", True),
    (r"rx\s*7[67]00[sm]\b|rx\s*7600m xt", "gfx1102", False),  # ノート用の RX 7700S / 7600S / 7600M は Navi 33
    (r"rx\s*7900|pro w7[89]00|navi\s*31", "gfx1100", False),
    (r"rx\s*7[78]00|pro w7700|pro v710|navi\s*32", "gfx1101", False),
    (r"rx\s*76[05]0|pro w7[56]00|navi\s*33", "gfx1102", False),
    (r"780m|760m|740m|phoenix|hawk ?point", "gfx1103", True),
    # ノート用(M・S)は同じ番号でも石が違う: RX 6850M XT / 6800M / 6700M は Navi 22、RX 6800S / 6700S / 6650M /
    # 6600M / 6600S は Navi 23、RX 6550M / 6500M / 6450M / 6300M は Navi 24
    (r"rx\s*6[78][05]0m\b", "gfx1031", False),
    (r"rx\s*6[678]00s\b|rx\s*66[05]0m\b", "gfx1032", False),
    (r"rx\s*6[345][05]0m\b", "gfx1034", False),
    (r"rx\s*6[89]\d0|pro w6800|pro v620|navi\s*21", "gfx1030", False),
    (r"rx\s*67\d0|navi\s*22", "gfx1031", False),
    (r"rx\s*66\d0|pro w6600|navi\s*23", "gfx1032", False),
    # デスクトップの RX 6300 も Navi 24(pci.ids「7424 Navi 24 [Radeon RX 6300]」、Pro W6300 / W6400 も同じ)
    (r"rx\s*6[345]00|pro w6[345]00|navi\s*24", "gfx1034", False),
    (r"680m|660m|rembrandt", "gfx1035", True),
    (r"610m|raphael|granite ridge|mendocino", "gfx1036", True),
    # Steam Deck(Van Gogh、OLED 版の Sephiroth も gfx1033)。Windows のドライバの名前は「AMD Custom GPU 0405」
    # (pci.ids「163f VanGogh [AMD Custom GPU 0405]」「1435 Sephiroth [AMD Custom GPU 0405]」)
    (r"van ?gogh|sephiroth|custom gpu 0405", "gfx1033", True),
    (r"rx\s*5[67]\d0|navi\s*10", "gfx1010", False),
    (r"rx\s*5[35]00|pro w5500|navi\s*14", "gfx1012", False),
    (r"mi35\dx?", "gfx950", False),
    (r"mi3[02]\dx?|mi300a", "gfx942", False),
    (r"mi2[15]0x?", "gfx90a", False),
    (r"mi100", "gfx908", False),
    (r"radeon vii|radeon pro vii|mi[56]0\b|vega\s*20", "gfx906", False),
)
# 名前が「AMD Radeon(TM) Graphics」だけの内蔵 GPU は、CPU の名前で決める(上から順に試す。迷うものは入れない)
_AMD_CPU_IGPU: tuple[tuple[str, str], ...] = (
    (r"ryzen ai max", "gfx1151"),
    (r"ryzen ai 9 (hx )?3[67]\d", "gfx1150"),
    (r"ryzen ai [57] (pro )?3[3-5]\d", "gfx1152"),
    # Dragon Range(7945HX・7940HX・7840HX・7745HX・7645HX など)・Fire Range(9955HX・9850HX など)は
    # Radeon 610M(2 CU)の内蔵 GPU(gfx1036、AMD_DISPLAY_ONLY_GFX)。AMD の製品ページの「Graphics Model:
    # AMD Radeon 610M」「Graphics Core Count: 2」(2026-10-22 に読んだ。例
    # https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hx.html)
    (r"ryzen [3579] (pro )?7\d4[05]hx3?d?\b|ryzen [3579] (pro )?9\d5[05]hx3?d?\b", "gfx1036"),
    # 携帯機の Ryzen Z1 Extreme(12 CU)・Z1(4 CU)は RDNA 3 の「AMD Radeon Graphics」(AMD の製品ページ
    # https://www.amd.com/en/products/processors/handhelds/ryzen-z-series/z1-series/z1-extreme.html
    # ・…/z1.html)。12 CU・Zen 4 8 個は Phoenix1(780M)、4 CU・Zen 4 2 個 + Zen 4c 4 個は
    # Phoenix2(740M)と同じ石とみなした(推定)。
    # どちらも gfx1103(Linux の amdkfd kfd_device.c: GC 11.0.1・11.0.4 → gfx_target_version 110003)
    (r"ryzen z1\b", "gfx1103"),
    (r"ryzen [3579] (pro )?[78]\d4\d(u|hs|h)\b|ryzen [3579] (pro )?8[5-7]00g", "gfx1103"),
    (r"ryzen [3579] (pro )?(6[89]\d0|7[37]35)(u|hs|h|hx)\b", "gfx1035"),
)

# Intel: PyTorch 2.14 の「Getting Started on Intel GPU」(https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)
# の Client GPUs(Windows 11・Ubuntu 24.04/26.04・WSL2)の検証済みの部品だけ。
# UHD・Iris Xe・「Intel(R) Graphics」
# (Arrow Lake-S などの Xe-LPG)は入らない。(名前の形, 系統, 内蔵か, 名前から推す VRAM の下限 GiB)
_INTEL_XPU: tuple[tuple[str, str, bool, float | None], ...] = (
    (r"arc.*\ba770\b", "Arc A-Series (Alchemist)", False, 8.0),  # 8 GB 版と 16 GB 版があるので少ないほう
    (r"arc.*\ba750\b|arc.*\ba580\b", "Arc A-Series (Alchemist)", False, 8.0),
    (r"arc.*\ba380\b", "Arc A-Series (Alchemist)", False, 6.0),
    (r"arc.*\ba310\b", "Arc A-Series (Alchemist)", False, 4.0),
    # ノート用も Alchemist の外付け。公称の VRAM は Intel の表
    # (https://www.intel.com/content/www/us/en/support/articles/000091109.html: A770M 16GB・A730M 12GB・
    # A550M 8GB・A370M 4GB・A350M 4GB)。Linux の xe・i915 は VRAM の量を sysfs に出さない
    # (docs/hardware.md)ので、Linux ではこの公称の値を使う。A570M・A530M・Pro の M・組み込み用(E)は
    # Intel の資料で確かめられなかったので値を持たない
    (r"arc.*\ba770m\b", "Arc A-Series (Alchemist)", False, 16.0),
    (r"arc.*\ba730m\b", "Arc A-Series (Alchemist)", False, 12.0),
    (r"arc.*\ba550m\b", "Arc A-Series (Alchemist)", False, 8.0),
    (r"arc.*\ba3[57]0m\b", "Arc A-Series (Alchemist)", False, 4.0),
    (r"arc.*\bpro a\d{2}m?\b|arc.*\ba\d{3}[me]\b|\bdg2\b", "Arc A-Series (Alchemist)", False, None),
    (r"arc.*\bb580\b", "Arc B-Series (Battlemage)", False, 12.0),
    (r"arc.*\bb570\b", "Arc B-Series (Battlemage)", False, 10.0),
    (r"arc.*\bpro b60\b", "Arc B-Series (Battlemage)", False, 24.0),
    (r"arc.*\bpro b50\b", "Arc B-Series (Battlemage)", False, 16.0),
    (r"battlemage|\bbmg\b", "Arc B-Series (Battlemage)", False, None),
    (r"arc.*\bb3\d0\b|panther lake", "Core Ultra Series 3 (Panther Lake)", True, None),
    (r"arc.*\b1[34]0v\b|lunar lake", "Core Ultra Series 2 (Lunar Lake)", True, None),
    (r"arc.*\b1[34]0t\b|arrow lake-[hp]", "Core Ultra Series 2 (Arrow Lake-H)", True, None),
    (r"^(intel )?arc graphics$|meteor lake", "Core Ultra (Meteor Lake-H)", True, None),
)

# Linux の pci.ids が古く、lspci が「AMD Device 7551」のように型番の ID しか出さないときの表
# ((会社の ID, 型番の ID) → 上の名前の表で引ける名前)。pci.ids 2026.10.22(pciutils/pciids 86e7077e4b79、
# https://raw.githubusercontent.com/pciutils/pciids/master/pci.ids)の行から、判定に使う GPU と、使わないと
# 決めている内蔵 GPU(610M など)だけを写した。石の名前(Navi 31 など)で引くので、ノート用・Pro の違いは
# gfx に効かない。
# 「Intel Graphics」とだけある ID(Arc でない内蔵 GPU など)は入れない
PCI_DEVICE_NAMES: dict[tuple[str, str], str] = {
    **{("1002", d): "Navi 48" for d in ("7550", "7551")},
    ("1002", "7590"): "Navi 44",
    **{("1002", d): "Navi 31" for d in ("744c", "7448", "7449", "744a", "744b", "745e")},
    **{("1002", d): "Navi 32" for d in ("7460", "7461", "7470", "747e")},
    **{
        ("1002", d): "Navi 33"
        for d in ("7480", "7481", "7483", "7487", "7489", "748b", "7499", "749f", "73f0")
    },
    **{("1002", d): "Navi 21" for d in ("73a1", "73a2", "73a3", "73a5", "73ab", "73ae", "73af", "73bf")},
    **{("1002", d): "Navi 22" for d in ("73c3", "73df")},
    **{("1002", d): "Navi 23" for d in ("73e0", "73e1", "73e3", "73ef", "73ff")},
    **{("1002", d): "Navi 24" for d in ("7421", "7422", "7423", "7424", "743f")},
    **{("1002", d): "Phoenix" for d in ("15bf", "15c8")},
    **{("1002", d): "Hawk Point" for d in ("1900", "1901")},
    ("1002", "1681"): "Rembrandt",
    ("1002", "164e"): "Raphael",
    ("1002", "13c0"): "Granite Ridge",
    ("1002", "1506"): "Mendocino",
    ("1002", "163f"): "Van Gogh",
    ("1002", "1435"): "Van Gogh (Sephiroth)",
    ("1002", "150e"): "Strix",
    ("1002", "1586"): "Strix Halo",
    ("1002", "1114"): "Krackan",
    ("8086", "56a0"): "Arc A770",
    ("8086", "56a1"): "Arc A750",
    ("8086", "56a2"): "Arc A580",
    ("8086", "56a5"): "Arc A380",
    ("8086", "56a6"): "Arc A310",
    ("8086", "5690"): "Arc A770M",
    ("8086", "5691"): "Arc A730M",
    ("8086", "5692"): "Arc A550M",
    ("8086", "5693"): "Arc A370M",
    ("8086", "5694"): "Arc A350M",
    ("8086", "5696"): "Arc A570M",
    ("8086", "5697"): "Arc A530M",
    ("8086", "56b0"): "Arc Pro A30M",
    ("8086", "56b1"): "Arc Pro A40",
    ("8086", "56b2"): "Arc Pro A60M",
    ("8086", "56b3"): "Arc Pro A60",
    ("8086", "e20b"): "Arc B580",
    ("8086", "e20c"): "Arc B570",
    ("8086", "e211"): "Arc Pro B60",
    ("8086", "e212"): "Arc Pro B50",
    **{
        ("8086", d): "Battlemage"
        for d in ("e202", "e20d", "e210", "e215", "e216", "e220", "e221", "e222", "e223")
    },
    ("8086", "7d55"): "Arc Graphics",  # Meteor Lake-P [Intel Arc Graphics]
    ("8086", "7d51"): "Arc 140T",  # Arrow Lake-P [Arc Pro 130T/140T]
    ("8086", "64a0"): "Arc 140V",  # Lunar Lake の Arc Graphics 130V/140V
    **{("8086", d): "Arc B390" for d in ("b080", "b082", "b084", "b086")},  # Panther Lake
    **{("8086", d): "Arc B370" for d in ("b081", "b083", "b085", "b087")},
}
_VENDOR_PCI = {"amd": "1002", "intel": "8086"}


def pci_device_name(vendor: str, name: str) -> tuple[str, str] | None:
    """「AMD Device 7551」のような名前 → (型番の ID, 表の名前。表に無ければ空)。
    その形でない名前(製品名があるとき)は None。"""
    if vendor not in _VENDOR_PCI:  # 型番の ID で引くのは AMD・Intel だけ(#101)
        return None
    m = re.fullmatch(r"(amd |intel )?device ([0-9a-f]{4})", _low(name))
    if not m:
        return None
    dev = m.group(2)
    found = PCI_DEVICE_NAMES.get((_VENDOR_PCI.get(vendor, ""), dev))
    return (dev, found) if found else (dev, "")


def pci_name_conflict(vendor: str, pci_id: str, name: str) -> bool | None:
    """提出の PCI の型番 ID と GPU 名が食い違うか(0.9.0)。ID が PCI_DEVICE_NAMES に無ければ None。
    AMD は ID の石と名前の gfx、Intel は名前の表(_INTEL_XPU)の世代と公称の VRAM で比べる。名前から決まらない
    (「AMD Radeon(TM) Graphics」など)ときは食い違いとしない(False)。"""
    ven, _, dev = (pci_id or "").partition(":")
    known = PCI_DEVICE_NAMES.get((ven, dev))
    if not known:
        return None
    if vendor == "amd":
        a, b = amd_gfx(known), amd_gfx(name)
        return bool(a and b and a[0] != b[0])
    if vendor == "intel":
        a, b = intel_xpu(f"intel {known}"), intel_xpu(name)
        if not a or not b:
            return False
        return a[0] != b[0] or (a[2] is not None and b[2] is not None and a[2] != b[2])
    return False


# 内蔵 GPU・APU(Apple 以外)が使えるメインメモリの割合。Linux は TTM が GPU に貸すページの上限を
# 「約 50% of the total system memory」にする(torvalds/linux a804c57 drivers/gpu/drm/ttm/ttm_device.c:88〜92。
# amdgpu の GTT の既定はこの値: docs.kernel.org の amdgpu の gttsize「-1 = Use value specified by TTM」)。
# Windows の共有 GPU メモリも物理メモリの約半分(docs/hardware.md)。Linux の amdgpu は sysfs の
# mem_info_gtt_total が読めればそれを使う
SHARED_MEMORY_SHARE = 0.5
# Apple silicon の統合メモリのうち GPU に使うとみなす割合。ComfyUI は MPS では搭載量の全部を
# GPU のメモリとして数える(model_management.py の get_total_memory・vram_state SHARED)が、
# macOS は GPU が性能を落とさずに使える量
# (Metal の recommendedMaxWorkingSetSize)を搭載量より小さくし、OS・ComfyUI 自身もメモリを使うので、
# 3/4 とした(判断で決めた値で、実測していない。docs/accuracy.md)
UNIFIED_MEMORY_SHARE = 0.75
MACOS_MIN_FOR_MPS = (14, 0)  # Apple「Accelerated PyTorch training on Mac」: macOS 14.0 or later(PyTorch 2.11)

VENDOR_LABEL = {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}

# 計算に使う GPU ではない画面の部品の会社(PCI の会社の ID → 名前。pci.ids の会社名)。仮想マシンの画面と、
# サーバーの管理用(BMC)の画面(#101)
DISPLAY_ONLY_VENDORS = {
    "1234": "QEMU",
    "1af4": "virtio",
    "15ad": "VMware",
    "80ee": "VirtualBox",
    "1414": "Microsoft Hyper-V",
    "1013": "Cirrus Logic",
    "1b36": "QEMU (Red Hat)",
    "1a03": "ASPEED",
    "102b": "Matrox",
}

# WSL2 で使える AMD の GPU(AMD「WSL compatibility」の ROCm 7.2.1 の表の製品名。#101。
# https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html
# 2026-10-27 に読んだ)。名前は _wsl_amd_key の形(会社名・Radeon・(TM) を除いた小文字)。APU は表に無い
WSL_AMD_SUPPORTED = frozenset(
    {
        "rx 9070",
        "rx 9070 xt",
        "rx 9070 gre",
        "ai pro r9700",
        "ai pro r9600d",
        "rx 9060",
        "rx 9060 xt",
        "rx 7900 xtx",
        "rx 7900 xt",
        "rx 7900 gre",
        "pro w7900",
        "pro w7900 dual slot",
        "pro w7800",
        "pro w7800 48gb",
        "pro w7700",
        "rx 7800 xt",
        "rx 7700",
        "rx 7700 xt",
    }
)
# Intel の WSL2 の Windows のドライバの下限(Intel「PyTorch Prerequisites for Intel GPUs」2.14 の Windows の節
# 「32.0.101.8801 or higher」。版の比べ方は Intel support 000005654: 3 つ目と 4 つ目の欄をつないだ 7 桁)
INTEL_WSL_MIN_BUILD = 1018801


def _wsl_amd_key(name: str) -> str:
    n = _low(name).replace("amd ", " ").replace("radeon ", " ")
    return " ".join(n.split())


def intel_driver_build(version: str | None) -> int | None:
    """Intel の画面ドライバの版("32.0.101.8801")→ 7 桁の build 番号(1018801)。読めなければ None。"""
    parts = (version or "").strip().split(".")
    if len(parts) != 4 or not all(x.isascii() and x.isdigit() for x in parts):
        return None
    return int(f"{int(parts[2])}{int(parts[3]):04d}")


@dataclass(frozen=True)
class Host:
    """GPU の対応を決めるのに使う、この PC の OS と CPU(試験では差し替える)。"""

    system: str  # Windows / Linux / Darwin
    windows_build: int | None = None
    mac_version: tuple[int, ...] | None = None
    cpu: str = ""


def host_info() -> Host:
    system = platform.system()
    build = None
    mac = None
    if system == "Windows":
        parts = platform.version().split(".")
        build = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
    elif system == "Darwin":
        mac = parse_version(platform.mac_ver()[0]) or None
        if mac == (10, 16):
            # 古い SDK で作った Python は macOS 11 以降でも 10.16 を返す(互換の版)。本当の版を読み直す(#101)
            r = _run(
                ["/usr/bin/env", "SYSTEM_VERSION_COMPAT=0", "sw_vers", "-productVersion"], OTHER_GPU_TIMEOUT
            )
            again = parse_version(r[1]) if r is not None and r[0] == 0 else None
            mac = again if again and again >= (11,) else None
    return Host(system=system, windows_build=build, mac_version=mac, cpu=_cpu_name())


def _low(name: str) -> str:
    n = name.lower()
    for mark in ("(tm)", "(r)", "™", "®"):
        n = n.replace(mark, " ")
    return " ".join(n.split())


def amd_gfx(name: str, cpu: str = "") -> tuple[str, bool] | None:
    """AMD の GPU 名 → (gfx, 内蔵か)。
    名前が「AMD Radeon(TM) Graphics」だけの内蔵 GPU は CPU の名前で決める。"""
    n = _low(name)
    for pat, gfx, integrated in _AMD_NAMES:
        if re.search(pat, n):
            return gfx, integrated
    if re.fullmatch(r"(amd )?radeon graphics", n):
        c = _low(cpu)
        for pat, gfx in _AMD_CPU_IGPU:
            if re.search(pat, c):
                return gfx, True
    return None


def intel_xpu(name: str) -> tuple[str, bool, float | None] | None:
    """Intel の GPU 名 → (系統, 内蔵か, 名前から推す VRAM の下限)。PyTorch の XPU 版の対象でなければ None。"""
    n = _low(name)
    for pat, label, integrated, floor in _INTEL_XPU:
        if re.search(pat, n):
            return label, integrated, floor
    return None


def _windows_ok(host: Host) -> bool:
    """Windows 11 と分かるか(版が分からなければ使わない側にする)。"""
    return host.windows_build is not None and host.windows_build >= WINDOWS_11_BUILD


def _shared_pool(a: Adapter, ram_gb: float) -> tuple[float, str]:
    """内蔵 GPU・APU が使えるメモリ(GiB)と、その説明。取り分けた VRAM(BIOS の UMA)+ 共有の分。"""
    dedicated = a.vram_gb or 0.0
    if a.shared_gb:
        return dedicated + a.shared_gb, tr(
            f"取り分けた VRAM {dedicated:g} GB + GTT {a.shared_gb:g} GB(sysfs)",
            f"reserved VRAM {dedicated:g} GB + GTT {a.shared_gb:g} GB (sysfs)",
        )
    share = ram_gb * SHARED_MEMORY_SHARE
    return dedicated + share, tr(
        f"取り分けた VRAM {dedicated:g} GB + メインメモリの {SHARED_MEMORY_SHARE:.0%}({share:.1f} GB)",
        f"reserved VRAM {dedicated:g} GB + {SHARED_MEMORY_SHARE:.0%} of main memory ({share:.1f} GB)",
    )


def classify_adapter(a: Adapter, host: Host, ram_gb: float) -> Gpu | None:
    """NVIDIA 以外の画面の部品を対応表で判定し、a.usable・a.integrated・a.reason を書く。
    ComfyUI + PyTorch が動く組み合わせなら判定に使う Gpu を返す(推定で、段階は good まで)。
    ram_gb は内蔵 GPU・Apple の共有の分を決めるメインメモリ(--ram で上書きしたらその値)。"""
    if a.vendor == "nvidia":
        return None  # NVIDIA は nvidia-smi で決める
    a.needs = None  # 判定し直すたびに消す(同じ部品を別の PC の条件で判定し直すことがある)
    label = VENDOR_LABEL.get(a.vendor, "")
    if label:
        tag = tr(f"推定({label}・実測なし)", f"Estimated ({label}, no measurement)")
    else:
        tag = tr("推定(実測なし)", "Estimated (no measurement)")

    def refuse(why: str, why_en: str) -> None:
        a.usable = False
        a.reason = tr(why + "ので、CPU として判定した", why_en + ", so it is judged as CPU")

    if host.system == "Darwin" and a.vendor != "apple":
        # 項目 7: macOS の Apple 以外の GPU は、世代を引く前に対象外と言う(Apple の Metal の PyTorch のページ:
        # Apple silicon と macOS 14 以降)
        refuse(
            f"macOS の Apple 以外の GPU({a.name})は PyTorch の MPS 版の対象ではない(Apple silicon だけ)",
            f"GPUs other than Apple on macOS ({a.name}) are not covered by the MPS build of PyTorch"
            " (Apple silicon only)",
        )
        return None
    if a.config_error in REFUSED_CM_ERRORS:
        # S6: デバイス マネージャーで無効・ドライバ無し・止まった部品は動かない(WSL2 から読んだ部品も同じ)
        ja, en = _CM_ERRORS[a.config_error]
        refuse(
            f"{a.name} は Windows のデバイス マネージャーで{ja}(コード {a.config_error})",
            f"{a.name} {en} in Windows Device Manager (code {a.config_error})",
        )
        return None
    ven_id = (a.pci_id or "").partition(":")[0]
    if a.vendor not in VENDOR_LABEL and ven_id in DISPLAY_ONLY_VENDORS:
        refuse(
            f"{a.name} は計算に使う GPU ではない(仮想マシン・管理用の画面: {DISPLAY_ONLY_VENDORS[ven_id]})",
            f"{a.name} is not a GPU for computing (a virtual machine or management display:"
            f" {DISPLAY_ONLY_VENDORS[ven_id]})",
        )
        return None
    if a.source == "wsl":
        return _classify_wsl(a, ram_gb, tag, refuse)

    gfx = None
    arch = None
    # Linux の pci.ids が古いと「AMD Device 7551」「Intel Device e20b」になる。
    # 型番の ID を表(PCI_DEVICE_NAMES)で引く
    name = a.name
    pci_note = None
    pci = pci_device_name(a.vendor, a.name)
    if pci is not None and not (pci[1] or a.gfx_target):  # amdkfd の世代があれば名前は要らない
        if a.source == "sysfs":
            refuse(
                f"{a.name} は製品名が分からない(lspci が無く、型番の ID が表に無い。lspci を入れると分かる)",
                f"The product name of {a.name} is unknown (no lspci, and the device ID is not in the table;"
                " installing lspci shows it)",
            )
        else:
            refuse(
                f"{a.name} は製品名が分からない(lspci の pci.ids が古い。update-pciids で新しくすると分かる)",
                f"The product name of {a.name} is unknown (the pci.ids of lspci is old; update-pciids fixes "
                "it)",
            )
        return None
    if pci is not None and pci[1]:
        dev, known = pci
        name = known
        if a.source == "sysfs":
            pci_note = tr(
                f"lspci が無いので、型番の ID({dev})から {known} とみなした",
                f"There is no lspci, so it was taken as {known} from the device ID ({dev})",
            )
        else:
            pci_note = tr(
                f"lspci の pci.ids が古く製品名が出ないので、型番の ID({dev})から {known} とみなした"
                "(update-pciids で新しくすると製品名が出る)",
                f"The pci.ids of lspci is old and shows no product name, so it was taken as {known}"
                f" from the device ID ({dev}) (update-pciids shows the product name)",
            )
    if a.vendor == "amd":
        found = amd_gfx(name, host.cpu)
        if a.gfx_target:  # 項目 10: amdkfd の gfx_target_version を名前より先に使う
            gfx, a.integrated = a.gfx_target, (found[1] if found else _guess_integrated(a))
            arch = AMD_ARCH.get(gfx) or _new_amd_arch(gfx)
        elif found is None:
            generic = bool(re.fullmatch(r"(amd )?radeon graphics", _low(a.name)))
            if generic:
                a.integrated = True  # CPU の名前でも世代が決まらない内蔵 GPU
                refuse(
                    f"{a.name} は世代(gfx)が名前から分からない",
                    f"The generation (gfx) of {a.name} is unknown from its name",
                )
            else:  # 項目 11: 表に無い新しい GPU
                refuse(
                    f"{a.name} は名前の表に無い GPU(新しい GPU?)で、世代(gfx)が確かめられない",
                    f"{a.name} is not in the name table (a new GPU?),"
                    " so its generation (gfx) cannot be checked",
                )
                a.reason += _vram_hint()
            return None
        else:
            gfx, a.integrated = found
            arch = AMD_ARCH.get(gfx, "?")
        if host.system not in ("Windows", "Linux"):
            refuse(
                f"{host.system} の ROCm 版 PyTorch は無い",
                f"There is no ROCm build of PyTorch for {host.system}",
            )
            return None
        if not (
            arch in AMD_SUPPORTED_ARCH
            or arch == AMD_NEW_ARCH
            or (host.system == "Linux" and arch in AMD_LINUX_ONLY_ARCH)
        ):
            refuse(
                f"{a.name}({arch}・{gfx})は ComfyUI が案内する ROCm 版 PyTorch の対象(RDNA 2 以降)に入らない",
                f"{a.name} ({arch}, {gfx}) is not covered by the ROCm build of PyTorch that ComfyUI "
                "recommends (RDNA 2 and later)",
            )
            return None
        if gfx in AMD_DISPLAY_ONLY_GFX:
            refuse(  # Radeon 610M(Dragon Range・Raphael など)
                f"{a.name}({gfx})は画面を出すための 2 CU の小さな内蔵 GPU で、"
                "生成では CPU より速い見込みが無い",
                f"{a.name} ({gfx}) is a small 2-CU integrated GPU for the display"
                " and is not expected to generate faster than the CPU",
            )
            return None
        if host.system == "Windows" and not _windows_ok(host):
            if host.windows_build is not None:  # 版が分からないときは条件を言わない
                a.needs = "Windows 11"
            refuse(
                "AMD の GPU の Windows 版 PyTorch(ROCm)は Windows 11 が要る"
                "(ComfyUI の案内。版が分からなくても使わない)",
                "PyTorch for AMD GPUs on Windows (ROCm) needs Windows 11"
                " (per ComfyUI; not used when the version is unknown either)",
            )
            return None
        notes = [
            tr(
                f"{tag}: ROCm 版 PyTorch({gfx}、{arch})で動く組み合わせとして判定した。段階は good まで",
                f"{tag}: judged as a combination that runs with the ROCm build of PyTorch ({gfx}, {arch})."
                " Level up to good",
            )
        ]
        if arch == AMD_NEW_ARCH:
            notes.append(
                tr(
                    f"{gfx} は表に無い新しい世代。RDNA 2 以降の決まりに照らして使える側とみなした(推定)",
                    f"{gfx} is a newer generation than the table; taken as usable under the RDNA 2-or-later"
                    " rule (estimate)",
                )
            )
        if a.gfx_target and host.system == "Linux":
            notes.append(
                tr(
                    f"世代は Linux の amdkfd の gfx_target_version({gfx})で決めた",
                    f"The generation was taken from gfx_target_version of the Linux amdkfd ({gfx})",
                )
            )
        if host.system == "Linux" and gfx not in PYTORCH_ROCM_LINUX_GFX:
            notes.append(
                tr(
                    f"{gfx} は pytorch.org の ROCm 版には入っていないので、"
                    "AMD の multi-arch 版(TheRock)の PyTorch を使う",
                    f"{gfx} is not in the ROCm build on pytorch.org,"
                    " so use the AMD multi-arch (TheRock) build of PyTorch",
                )
            )
        if host.system == "Windows":
            notes.append(
                tr(
                    "Windows では AMD の multi-arch 版 PyTorch(ROCm 10)か ComfyUI の AMD 版 portable を使う",
                    "On Windows, use the AMD multi-arch build of PyTorch (ROCm 10) or the AMD portable of "
                    "ComfyUI",
                )
            )
        backend, bf16 = "rocm", arch not in AMD_NO_BF16_ARCH
    elif a.vendor == "intel":
        found_i = intel_xpu(name)
        if found_i is None and re.search(r"\barc\b", _low(name)):  # 項目 11: 表に無い新しい Arc
            refuse(
                f"{a.name} は名前の表に無い Arc(新しい GPU?)で、PyTorch の XPU 版の対象か確かめられない",
                f"{a.name} is an Arc not in the name table (a new GPU?), so whether the XPU build of PyTorch"
                " covers it cannot be checked",
            )
            a.reason += _vram_hint()
            return None
        if found_i is None:
            if re.search(r"uhd|iris|hd graphics|^intel graphics$", _low(name)):
                a.integrated = True
            refuse(
                f"{a.name} は PyTorch の XPU 版が対象にする Intel の GPU"
                "(Arc A / B シリーズ、Core Ultra の Arc Graphics)に入らない",
                f"{a.name} is not one of the Intel GPUs covered by the XPU build of PyTorch"
                " (Arc A / B series, Arc Graphics of Core Ultra)",
            )
            return None
        series, a.integrated, floor = found_i
        if host.system not in ("Windows", "Linux"):
            refuse(
                f"{host.system} の XPU 版 PyTorch は無い",
                f"There is no XPU build of PyTorch for {host.system}",
            )
            return None
        if host.system == "Windows" and not _windows_ok(host):
            if host.windows_build is not None:  # 版が分からないときは条件を言わない
                a.needs = "Windows 11"
            refuse(
                "Intel の GPU の XPU 版 PyTorch は Windows 11 で確かめられている"
                "(Windows 10・版が分からないときは対象外)",
                "The XPU build of PyTorch for Intel GPUs is verified on Windows 11"
                " (not covered on Windows 10 or an unknown version)",
            )
            return None
        notes = [
            tr(
                f"{tag}: XPU 版 PyTorch({series})で動く組み合わせとして判定した。段階は good まで",
                f"{tag}: judged as a combination that runs with the XPU build of PyTorch ({series})."
                " Level up to good",
            )
        ]
        if not a.integrated and not a.vram_gb and floor:
            a.vram_gb, a.vram_source = floor, "name"
            notes.append(
                tr(
                    f"VRAM の量が読めないので、名前から公称の {floor:g} GB"
                    "(量の違う版があれば少ないほう)とみなした",
                    f"The VRAM amount cannot be read, so the nominal {floor:g} GB was taken from the name"
                    " (the smaller one if versions differ)",
                )
            )
        backend, bf16 = "xpu", True
    elif a.vendor == "apple":
        a.integrated = True
        if host.system != "Darwin":
            refuse("Apple の GPU は macOS でだけ使える", "Apple GPUs can be used only on macOS")
            return None
        if not APPLE_CHIP_NAME.fullmatch(a.name.strip()):
            # 項目 2: macOS の仮想マシンの「Apple Paravirtual device」など。PyTorch #111576「[MPS] Skip
            # virtualized devices」(この部品では最初の確保が失敗する)
            refuse(
                f"{a.name} は Apple silicon のチップ名でない(仮想マシンの画面など)。MPS で動かない",
                f"{a.name} is not an Apple silicon chip name (e.g."
                " a virtual machine display) and does not run"
                " MPS",
            )
            return None
        if host.mac_version is None or host.mac_version < MACOS_MIN_FOR_MPS:
            refuse(
                "PyTorch の MPS 版は macOS 14.0 以降が要る(この PC の macOS は 14 より前か、版が分からない)",
                "The MPS build of PyTorch needs macOS 14.0 or later (this macOS is older than 14 or unknown)",
            )
            return None
        notes = [
            tr(
                f"{tag}: MPS 版 PyTorch で動く組み合わせとして判定した。段階は good まで",
                f"{tag}: judged as a combination that runs with the MPS build of PyTorch. Level up to good",
            )
        ]
        backend, bf16 = "mps", True
    else:
        refuse(f"{a.name} は会社が分からない", f"The vendor of {a.name} is unknown")
        return None
    if pci_note:
        notes.insert(1, pci_note)
    return _usable_gpu(a, ram_gb, notes, backend, bf16, arch, gfx)


AMD_NEW_ARCH = "RDNA (new)"


def _vram_hint() -> str:
    return tr(
        "(--vram で VRAM を指定すれば判定できる。名前の表は毎週の更新で足す)",
        " (give the VRAM with --vram to judge; the name table is extended by the weekly update)",
    )


def _new_amd_arch(gfx: str) -> str:
    """表(AMD_ARCH)に無い gfx の世代。gfx10.3 以降(RDNA 2 以降)と主番号 13 以上は新しい世代
    (AMD_NEW_ARCH。今の決まり「RDNA 2 以降」に照らして断らない)、ほかは分からない("?")。"""
    m = re.fullmatch(r"gfx(\d{1,2})(\d)([0-9a-f])", gfx)
    if not m:
        return "?"
    major, minor = int(m.group(1)), int(m.group(2))
    return AMD_NEW_ARCH if (major, minor) >= (10, 3) else "?"


def _guess_integrated(a: Adapter) -> bool:
    """名前で決まらない AMD の GPU が内蔵 GPU か: sysfs の VRAM が 2 GiB 未満で GTT が読めたとき
    内蔵とみなす。"""
    return bool(a.shared_gb) and (a.vram_gb or 0.0) < 2.0


def _classify_wsl(a: Adapter, ram_gb: float, tag: str, refuse) -> Gpu | None:
    """WSL2 の中から読んだ Windows 側の AMD・Intel の GPU(#101。docs/hardware.md「WSL2」)。
    使うのは、WSL の中の
    準備(rocminfo / clinfo)ができていて、公式の WSL の対応表にあるときだけ。段階は推定で good まで。"""
    vram_note = tr(
        "WSL の中では、Windows 側の VRAM より空きが約 3 GB 少ないという報告がある(値は変えていない)",
        "Inside WSL, free VRAM is reported to be about 3 GB less than on Windows (the value is not changed)",
    )
    if a.vendor == "amd":
        if _wsl_amd_key(a.name) not in WSL_AMD_SUPPORTED:
            refuse(
                f"{a.name} は AMD の WSL の対応表(ROCm 7.2.1)に無い",
                f"{a.name} is not in AMD's WSL support table (ROCm 7.2.1)",
            )
            return None
        if not a.wsl_runtime or not a.gfx_target:
            refuse(
                f"Windows 側に {a.name} があるが、WSL の中の rocminfo で GPU が見えない。WSL2 で使うには、"
                "Windows に AMD Software: Adrenalin Edition for WSL2 を、WSL の中に ROCm の WSL 版を入れる",
                f"{a.name} is on the Windows side, but rocminfo"
                " inside WSL does not see the GPU. To use it in WSL2,"
                " install AMD Software: Adrenalin Edition for WSL2"
                " on Windows and the WSL build of ROCm inside WSL",
            )
            return None
        if not a.vram_gb:
            refuse(f"{a.name} の VRAM の量が読めない", f"The VRAM amount of {a.name} cannot be read")
            return None
        a.integrated = False
        gfx = a.gfx_target
        arch = AMD_ARCH.get(gfx) or _new_amd_arch(gfx)
        notes = [
            tr(
                f"{tag}: WSL2 の ROCm 版"
                f" PyTorch({gfx}、{arch})で動く組み合わせとして判定した。段階は good まで",
                f"{tag}: judged as a combination that runs with the ROCm build of PyTorch in WSL2"
                f" ({gfx}, {arch}). Level up to good",
            ),
            vram_note,
        ]
        return _usable_gpu(a, 0.0, notes, "rocm", arch not in AMD_NO_BF16_ARCH, arch, gfx)
    if a.vendor == "intel":
        found = intel_xpu(a.name)
        if found is None:
            refuse(
                f"{a.name} は PyTorch の XPU 版が WSL2 で確かめた Intel の GPU"
                "(Arc A / B シリーズ、Core Ultra の Arc Graphics)に入らない",
                f"{a.name} is not one of the Intel GPUs the XPU build of PyTorch verifies on WSL2"
                " (Arc A / B series, Arc Graphics of Core Ultra)",
            )
            return None
        build = intel_driver_build(a.driver)
        if build is not None and build < INTEL_WSL_MIN_BUILD:
            refuse(
                f"Windows の Intel のドライバ {a.driver} は古い(WSL2"
                " の XPU 版 PyTorch は 32.0.101.8801 以上)",
                f"The Windows Intel driver {a.driver} is too old (the XPU build of PyTorch on WSL2 needs"
                " 32.0.101.8801 or later)",
            )
            return None
        if not a.wsl_runtime:
            refuse(
                f"Windows 側に {a.name} があるが、WSL の中の clinfo"
                " -l に Intel の GPU が出ない。WSL の中に Intel の"
                " compute runtime(libze1・intel-opencl-icd など)を入れる",
                f"{a.name} is on the Windows side, but clinfo -l"
                " inside WSL lists no Intel GPU. Install the Intel"
                " compute runtime (libze1, intel-opencl-icd etc.) inside WSL",
            )
            return None
        series, a.integrated, floor = found
        notes = [
            tr(
                f"{tag}: WSL2 の XPU 版 PyTorch({series})で動く組み合わせとして判定した。段階は good まで",
                f"{tag}: judged as a combination that runs with the XPU build of PyTorch in WSL2 ({series})."
                " Level up to good",
            )
        ]
        if build is None:
            notes.append(
                tr(
                    "Windows の Intel"
                    " のドライバの版が読めないので、ドライバでは断っていない(32.0.101.8801 以上が要る)",
                    "The Windows Intel driver version cannot be read, so it is not refused by driver"
                    " (32.0.101.8801 or later is needed)",
                )
            )
        if a.integrated:
            notes.append(
                tr(
                    "WSL の中のメモリ(Windows の搭載量以下)から共有の分を見積もった(少なめの見積もり)",
                    "The shared share was estimated from the memory inside WSL (at most the Windows amount;"
                    " a low estimate)",
                )
            )
        elif not a.vram_gb and floor:
            a.vram_gb, a.vram_source = floor, "name"
        notes.append(vram_note)
        return _usable_gpu(a, ram_gb, notes, "xpu", True, None, None)
    refuse(
        f"{a.name} は WSL2 で判定に使う GPU(AMD・Intel)ではない",
        f"{a.name} is not a GPU used in WSL2 (AMD, Intel)",
    )
    return None


def _usable_gpu(
    a: Adapter, ram_gb: float, notes: list[str], backend: str, bf16: bool, arch: str | None, gfx: str | None
) -> Gpu | None:
    def refuse(why: str, why_en: str) -> None:
        a.usable = False
        a.reason = tr(why + "ので、CPU として判定した", why_en + ", so it is judged as CPU")

    if a.vendor == "apple":
        pool, memory = ram_gb * UNIFIED_MEMORY_SHARE, "unified"
        notes.append(
            tr(
                f"統合メモリ {ram_gb:.1f} GB のうち {UNIFIED_MEMORY_SHARE:.0%}({pool:.1f} GB)を"
                "GPU が使えるとみなした"
                "(残りは OS・ComfyUI 自身の分。逃がす先も同じメモリなので、逃がす動かし方は無い)",
                f"The GPU is taken to use {UNIFIED_MEMORY_SHARE:.0%} ({pool:.1f} GB) of the {ram_gb:.1f} GB"
                " unified memory (the rest is for the OS and ComfyUI itself; offloading would go to the same"
                " memory, so there is no offload mode)",
            )
        )
    elif a.integrated:
        pool, how = _shared_pool(a, ram_gb)
        memory = "shared"
        notes.append(
            tr(
                f"内蔵 GPU・APU なので、GPU が使えるメモリを {pool:.1f} GB とみなした({how})。"
                "逃がす先も同じメモリなので、逃がす動かし方は無い",
                f"Integrated GPU / APU, so the GPU is taken to use {pool:.1f} GB ({how})."
                " Offloading would go to the same memory, so there is no offload mode",
            )
        )
    else:
        if not a.vram_gb:
            refuse(f"{a.name} の VRAM の量が読めない", f"The VRAM amount of {a.name} cannot be read")
            return None
        pool, memory = a.vram_gb, "dedicated"
    if pool <= 0:
        refuse(f"{a.name} が使えるメモリの量が分からない", f"The memory {a.name} can use is unknown")
        return None
    a.usable = True
    a.reason = notes[0]
    return Gpu(
        name=a.name,
        vendor=a.vendor,
        vram_total_gb=round(pool, 2),
        backend=backend,
        arch=arch,
        compute_dtype="fp16",
        bf16=bf16,
        fp8=False,
        notes=notes,
        memory=memory,
        gfx_target=gfx,
        carveout_gb=round(a.vram_gb or 0.0, 2) if memory == "shared" else None,
        pci_id=a.pci_id,
    )


def _other_gpu_notes(adapters: list[Adapter], judged_gpu: bool) -> list[str]:
    """画面の部品の注意。judged_gpu は NVIDIA の GPU(cuda)で判定するか。"""
    notes = []
    for a in adapters:
        if a.vendor == "nvidia":
            # 項目 8: どの読み方(WMI・lspci・sysfs・WSL)でも、名前(言語で変わる「Microsoft 基本ディスプレイ
            # アダプター」など)に頼らず、型番の会社 ID が NVIDIA の部品で知らせる
            if not judged_gpu:
                named = "nvidia" in a.name.lower()
                seen = a.name if named else f"NVIDIA の GPU({a.name} として見えている)"
                seen_en = a.name if named else f"An NVIDIA GPU (seen as {a.name})"
                notes.append(
                    tr(
                        f"{seen} があるが、nvidia-smi で確かめられないので CPU として判定した"
                        "(NVIDIA のドライバを入れると GPU で判定できる)",
                        f"{seen_en} is present but cannot be checked with nvidia-smi, so judged as CPU"
                        " (install the NVIDIA driver to judge with the GPU)",
                    )
                )
            continue
        if a.vram_gb and a.vram_source != "name":
            vram = tr(f"(VRAM {a.vram_gb:g} GB)", f" (VRAM {a.vram_gb:g} GB)")
        else:
            vram = ""
        if judged_gpu:
            notes.append(
                tr(
                    f"{a.name}{vram} を見つけた。NVIDIA の GPU があるので判定に使わない",
                    f"Found {a.name}{vram}. Not used because there is an NVIDIA GPU",
                )
            )
        elif a.usable is False:
            notes.append(f"{a.name}{vram}: {a.reason}")
    return notes


# --- メモリ ----------------------------------------------------------------------------------------


@dataclass
class Memory:
    total: int | None  # バイト。None は読めない(#101: psutil も OS の読み方も失敗)
    available: int | None
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


def read_meminfo() -> dict[str, int] | None:
    """/proc/meminfo(バイト。HugePages_* は数そのもの)。読めなければ None。"""
    text = _read_text(_p("/proc/meminfo"))
    if text is None:
        return None
    out: dict[str, int] = {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_()]+):\s+([0-9]+)(\s+kB)?\s*$", line.strip())
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if m.group(3) else 1)
    return out or None


def meminfo_available(mi: dict[str, int]) -> int | None:
    """空きのメインメモリ。MemAvailable の無い古いカーネルは MemFree + Cached(#101)。"""
    if "MemAvailable" in mi:
        return mi["MemAvailable"]
    if "MemFree" in mi:
        return mi["MemFree"] + mi.get("Cached", 0)
    return None


def unified_free(mi: dict[str, int]) -> int | None:
    """統合メモリの GPU(GB10)が今使える量。NVIDIA の DGX Spark の known-issues「Guidance for reporting memory
    resources with unified memory architecture」の例どおり: HugePages を使っていれば空きの HugePages × 大きさ
    (swap は数えない)、使っていなければ MemAvailable + SwapFree。"""
    if mi.get("HugePages_Total", 0) > 0 and mi.get("Hugepagesize"):
        return mi.get("HugePages_Free", 0) * mi["Hugepagesize"]
    avail = meminfo_available(mi)
    return None if avail is None else avail + mi.get("SwapFree", 0)


_CGROUP_V2 = "/sys/fs/cgroup"
_CGROUP_V1 = "/sys/fs/cgroup/memory"


def _cgroup_chain(base: str, path: str) -> list[str]:
    """cgroup のパスから根までのフォルダ(自分 → 親 → … → 根)。"." ".." を含むパスは根だけを見る。"""
    parts = [x for x in path.split("/") if x]
    if any(x in (".", "..") for x in parts):
        return [base]
    return [base + "".join("/" + x for x in parts[:i]) for i in range(len(parts), -1, -1)]


def _cgroup_number(path: str) -> int | None:
    text = (_read_text(_p(path)) or "").strip()
    return int(text) if text.isascii() and text.isdigit() else None  # v2 の "max"(上限なし)は None


def cgroup_memory() -> tuple[int | None, int | None]:
    """このプロセスの cgroup のメモリ上限と今の使用量(バイト。無ければ None)。/proc/self/cgroup の自分の
    cgroup から根へたどり、上限の一番小さいものを使う(#101。cgroup v2: `0::/path` の memory.max・
    memory.current、v1: `N:memory:/path` の memory.limit_in_bytes・memory.usage_in_bytes。Linux の
    Documentation/admin-guide/cgroup-v2.rst・cgroup-v1/memory.rst)。名前空間の中ではパスが `/` になり、
    根の値だけを読む。/proc/self/cgroup が読めなければ根の値だけを読む(以前の読み方)。"""
    text = _read_text(_p("/proc/self/cgroup"))
    v2: str | None = None
    v1: str | None = None
    for line in (text or "").splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        if parts[0] == "0" and parts[1] == "":
            v2 = parts[2]
        elif "memory" in parts[1].split(","):
            v1 = parts[2]
    if text is None:
        v2 = v1 = "/"
    limits: list[int] = []
    current: int | None = None
    for base, path, max_leaf, cur_leaf in (
        (_CGROUP_V2, v2, "memory.max", "memory.current"),
        (_CGROUP_V1, v1, "memory.limit_in_bytes", "memory.usage_in_bytes"),
    ):
        if path is None:
            continue
        chain = _cgroup_chain(base, path)
        limits += [v for v in (_cgroup_number(f"{d}/{max_leaf}") for d in chain) if v]
        if current is None:
            current = _cgroup_number(f"{chain[0]}/{cur_leaf}")
    return (min(limits) if limits else None), current


def is_wsl() -> bool:
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


def _fallback_memory() -> tuple[int | None, int | None, str]:
    """psutil が失敗したときの搭載量・空き(#101)。Windows は GlobalMemoryStatusEx、Linux は /proc/meminfo、
    macOS は sysctl hw.memsize(空きは分からない)。(搭載量, 空き, 読んだもの)。"""
    if _windows():
        st = _global_memory_status()
        if st:
            return st["total_phys"], st["avail_phys"], "GlobalMemoryStatusEx"
        return None, None, ""
    if platform.system() == "Linux":
        mi = read_meminfo()
        if mi and mi.get("MemTotal"):
            return mi["MemTotal"], meminfo_available(mi), "/proc/meminfo"
        return None, None, ""
    if platform.system() == "Darwin":
        r = _run(["sysctl", "-n", "hw.memsize"], OTHER_GPU_TIMEOUT)
        text = (r[1] if r and r[0] == 0 else "").strip()
        if text.isdigit():
            return int(text), None, "sysctl hw.memsize"
    return None, None, ""


def read_memory() -> Memory:
    try:
        vm = psutil.virtual_memory()
        mem = Memory(total=int(vm.total), available=int(vm.available))
    except Exception:
        total, avail, how = _fallback_memory()
        mem = Memory(total=total, available=avail)
        if total is None:
            mem.notes.append(
                tr(
                    "メモリの量が読めない(psutil が失敗し、OS の読み方でも読めない)。--ram で指定する",
                    "The memory amount cannot be read (psutil failed"
                    " and so did the OS reading). Give it with "
                    "--ram",
                )
            )
        else:
            mem.notes.append(
                tr(
                    f"psutil でメモリが読めなかったので、{how} で読んだ",
                    f"psutil could not read the memory, so it was read with {how}",
                )
            )
    if _windows():
        st = _global_memory_status()
        if st:
            mem.commit_limit = st["commit_limit"]
            mem.commit_used = max(0, st["commit_limit"] - st["commit_avail"])
            mem.swap_total = max(0, st["commit_limit"] - st["total_phys"])
        paging = _reg_paging_files()
        unread = tr(
            "ページファイルの設定が読めないので、仮想メモリは目安の倍率で見た",
            "The page file settings cannot be read, so virtual memory uses a rough factor",
        )
        if paging is None:
            mem.notes.append(unread)
        else:
            mem.swap_max, mem.swap_auto = parse_paging_files(paging)
            if mem.swap_max is None and mem.swap_auto is None:
                mem.notes.append(unread)
            elif mem.swap_max == 0:
                mem.notes.append(
                    tr(
                        "ページファイルが無いので、メモリが足りないモデルは動かないと判定する",
                        "No page file, so models that need more memory are judged not to run",
                    )
                )
    elif platform.system() == "Linux":
        text = _read_text(_p("/proc/swaps"))
        if text is not None:
            mem.swap_total = mem.swap_max = _linux_swaps(text)
            mem.swap_auto = False
        limit, current = cgroup_memory()
        if limit and (mem.total is None or limit < mem.total):
            mem.notes.append(
                tr(
                    f"コンテナのメモリ上限 {limit / GIB:.1f} GB を、搭載量の代わりに使った",
                    f"Used the container memory limit {limit / GIB:.1f} GB instead of the installed amount",
                )
            )
            mem.total = limit
            room = max(0, limit - current) if current is not None else limit
            mem.available = room if mem.available is None else min(mem.available, room)
        if is_wsl():
            mem.notes.append(
                tr(
                    "WSL の中で動いている: メモリは WSL に割り当てた分(.wslconfig の memory)で判定した",
                    "Running inside WSL: memory is judged by the amount given to WSL (memory in .wslconfig)",
                )
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
    text = _read_text(_p("/proc/cpuinfo"))
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
    host: Host | None = None,
) -> SystemInfo:
    """PC を調べる。vram_gb / ram_gb を渡すと、その値で上書きする(別の PC を想定した判定用)。

    gpus を渡すと nvidia-smi を使わない(試験・組み込み用)。other_gpus は NVIDIA 以外の GPU を調べるか:
    None(既定)は nvidia-smi で NVIDIA の GPU が見つからず、VRAM を上書きしていないときだけ調べる。
    判定に使える NVIDIA の GPU が無く VRAM を上書きしていなければ、見つけた AMD・Intel・Apple の GPU を対応表
    (classify_adapter)で判定に使う。memory を渡すとメモリを、host を渡すと OS・CPU を調べない(試験用)。"""
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
            tr(
                "GPU が複数ある。判定は VRAM が最大の 1 枚で行った。ComfyUI は既定で CUDA の 0 番"
                "(CUDA_DEVICE_ORDER の"
                "既定 FASTEST_FIRST で速い順の先頭)を使うので、違う GPU で動くときは ComfyUI を "
                "--cuda-device "
                "で"
                "その GPU に向ける",
                "There are several GPUs. Judged with the one with the most VRAM. ComfyUI uses CUDA device 0"
                " by default (the fastest, with the default CUDA_DEVICE_ORDER FASTEST_FIRST), so if it runs "
                "on"
                " a different GPU, point ComfyUI at that GPU with --cuda-device",
            )
        )
    mem = memory or read_memory()
    notes += mem.notes
    if python_is_32bit():
        notes.append(
            tr(
                "32 ビットの Python で動いている。PyTorch は 64 ビットの Python にしか無い",
                "Running on 32-bit Python. PyTorch exists only for 64-bit Python",
            )
        )

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
                Gpu(
                    name=tr(f"(仮定) {vram_gb:g} GB GPU", f"(assumed) {vram_gb:g} GB GPU"),
                    vendor="nvidia",
                    vram_total_gb=vram_gb,
                    backend="cuda",
                )
            ]
        else:
            found = []

    # メモリが読めない(#101)ときは 0 として判定する(notes に「--ram で指定する」とある)
    ram_total = (mem.total or 0) / GIB
    if ram_gb is not None:
        ram_total = ram_gb

    adapters: list[Adapter] = []
    # NVIDIA の GPU が使えない(見つからない・Kepler・古いドライバ)ときは既定で調べる
    # (AMD・Intel を判定に使うため)
    usable_nvidia = any(g.backend == "cuda" for g in found)
    want_other = (
        other_gpus if other_gpus is not None else (gpus is None and not usable_nvidia and vram_gb is None)
    )
    if want_other:
        # 使える NVIDIA の GPU があれば、WSL2 で Windows 側(powershell.exe・rocminfo・clinfo)は読まない
        adapters, other_notes = detect_other_gpus(wsl_windows=False) if usable_nvidia else detect_other_gpus()
        notes += other_notes
        judged = any(g.backend == "cuda" for g in found)
        if (
            not judged and vram_gb is None
        ):  # NVIDIA の GPU が使えないときだけ、AMD・Intel・Apple を対応表で見る
            h = host or host_info()
            for a in adapters:
                g = classify_adapter(a, h, ram_total)
                if g is not None:
                    found.append(g)
            usable = [g for g in found if g.backend in USABLE_BACKENDS]
            if usable:
                best = max(usable, key=lambda x: (x.memory == "dedicated", x.vram_total_gb))
                notes += [f"{best.name}: {n}" for n in best.notes]
                if len(usable) > 1:
                    notes.append(
                        tr(
                            f"使える GPU が複数ある。判定は {best.name} で行った"
                            "(専用の VRAM の GPU を先に、その中で最大)",
                            f"There are several usable GPUs. Judged with {best.name}"
                            " (GPUs with dedicated VRAM first, then the largest)",
                        )
                    )
        notes += _other_gpu_notes(adapters, judged)

    ram_total = (mem.total or 0) / GIB
    ram_avail = (mem.available if mem.available is not None else mem.total or 0) / GIB
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
