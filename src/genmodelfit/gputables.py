# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU の登録の表(#103。docs/gpu-inventory.md が一覧、docs/hardware.md が決まり)。

OS が世代を教えないものだけ、名前・型番 ID → 世代の表を手で持つ: Windows の AMD(名前 → gfx)、Intel(XPU の
対象)、NVIDIA の古いドライバ(名前 → compute capability。抜けても断らない)、PyTorch の CUDA 版の表(版ごと)。
表の 1 行は `Rule`(鍵・値・出典・確かめた日・足した版・注意)。出典は `SOURCES` の名前で書き、URL と読んだ日は
そちらに 1 回だけ書く。

新しい GPU を足すとき(毎週の更新の「GPU 表の見張り」が知らせる。docs/release.md「毎週の一覧の更新」):
1. 一次資料で値を確かめ、行に出典と確かめた日を付けて足す(`added` は次の版)
2. tests/fixtures/gpu_official の公式の一覧を `python -m genmodelfit.gpuwatch --save-fixtures` で取り直す
3. `python tools/gpu_inventory.py` で docs/gpu-inventory.md / .ja.md を作り直す(試験が一致を見る)

hardware.py は下の表から今までと同じ形(`_AMD_NAMES` など)を作って使う。表は判定にだけ使い、
ネットワークには触らない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# 表を確かめた日(この Issue の点検。公式の一覧は tests/fixtures/gpu_official に固定した)
AUDIT = "2026-10-27"


@dataclass(frozen=True)
class Source:
    title: str
    url: str
    read: str  # 読んだ日(YYYY-MM-DD)


SOURCES: dict[str, Source] = {
    # NVIDIA
    "nvidia-cuda-gpus": Source(
        "NVIDIA CUDA GPUs (compute capability)", "https://developer.nvidia.com/cuda-gpus", AUDIT
    ),
    "nvidia-cuda-legacy": Source(
        "NVIDIA Legacy CUDA GPUs (compute capability)", "https://developer.nvidia.com/cuda-legacy-gpus", AUDIT
    ),
    "cuda-rn": Source(
        "CUDA Toolkit release notes, Table 3 (minimum driver for minor version compatibility) and Table 4",
        "https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html",
        AUDIT,
    ),
    "torch-cuda-init": Source(
        "PyTorch v2.14.1 torch/cuda/__init__.py (PYTORCH_RELEASES_CODE_CC, DEVICE_REQUIREMENT)",
        "https://github.com/pytorch/pytorch/blob/v2.14.1/torch/cuda/__init__.py",
        AUDIT,
    ),
    "torch-build-cuda": Source(
        "PyTorch v2.11.0 .ci/manywheel/build_cuda.sh (TORCH_CUDA_ARCH_LIST of cu128 / cu129)",
        "https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh",
        AUDIT,
    ),
    "torch-win-cuda": Source(
        "PyTorch .ci/pytorch/windows/internal/cuda_config.bat / build_env_setup.py (Windows cu126 adds 6.1)",
        "https://github.com/pytorch/pytorch/tree/v2.14.1/.ci/pytorch/windows/internal",
        "2026-10-21",
    ),
    "torch-whl-index": Source(
        "download.pytorch.org torch wheel index (PEP 503)", "https://download.pytorch.org/whl/torch/", AUDIT
    ),
    "torch-build-rocm": Source(
        "PyTorch v2.14.1 .ci/docker/manywheel/build.sh (PYTORCH_ROCM_ARCH)",
        "https://github.com/pytorch/pytorch/blob/v2.14.1/.ci/docker/manywheel/build.sh",
        AUDIT,
    ),
    "comfyui-mm": Source(
        "ComfyUI comfy/model_management.py (should_use_fp16, AMD_RDNA2_AND_OLDER_ARCH)",
        "https://github.com/Comfy-Org/ComfyUI/blob/426f5ee/comfy/model_management.py",
        AUDIT,
    ),
    "comfyui-readme": Source(
        "ComfyUI README: AMD GPUs (Windows, ROCm 10.0) — RDNA 2, RDNA 3, RDNA 3.5, RDNA 4; Windows 11",
        "https://github.com/Comfy-Org/ComfyUI/blob/426f5ee/README.md",
        "2026-10-21",
    ),
    # AMD
    "therock-supported": Source(
        "ROCm/TheRock SUPPORTED_GPUS.md (Build Passing / Sanity Tested / Release Ready)",
        "https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md",
        AUDIT,
    ),
    "therock-releases": Source(
        "ROCm/TheRock RELEASES.md (GFX target lookup table)",
        "https://github.com/ROCm/TheRock/blob/main/RELEASES.md",
        AUDIT,
    ),
    "rocm-hw-specs": Source(
        "ROCm GPU hardware specifications (LLVM target name)",
        "https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html",
        AUDIT,
    ),
    "radeon-win-compat": Source(
        "AMD Radeon: Windows compatibility (ROCm 7.2.1)",
        "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/"
        "windows_compatibility.html",
        AUDIT,
    ),
    "ryzen-win-compat": Source(
        "AMD Ryzen: Windows compatibility (ROCm 7.2.1)",
        "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/"
        "windows_compatibility.html",
        AUDIT,
    ),
    "radeon-linux-compat": Source(
        "AMD Radeon: Linux compatibility (ROCm 7.2.1)",
        "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/"
        "native_linux_compatibility.html",
        AUDIT,
    ),
    "ryzen-linux-compat": Source(
        "AMD Ryzen: Linux compatibility (ROCm 7.2.1)",
        "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/"
        "native_linux_compatibility.html",
        AUDIT,
    ),
    "amd-wsl-compat": Source(
        "AMD Radeon: WSL compatibility (ROCm 7.2.1)",
        "https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/"
        "wsl_compatibility.html",
        AUDIT,
    ),
    "amd-product-pages": Source(
        "AMD product pages (Graphics Model / Graphics Core Count), e.g. Ryzen 9 7940HX, Ryzen Z1"
        " / Z1 Extreme",
        "https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hx.html",
        "2026-10-22",
    ),
    "linux-amdkfd": Source(
        "Linux drivers/gpu/drm/amd/amdkfd/kfd_device.c (GC IP version → gfx_target_version)",
        "https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_device.c",
        "2026-10-22",
    ),
    "steam-deck-spec": Source(
        "Valve Steam Deck tech specs (8 RDNA 2 CUs)", "https://www.steamdeck.com/en/tech", "2026-10-21"
    ),
    "pci-ids": Source(
        "pci.ids (pciutils/pciids, version 2026.10.27)",
        "https://raw.githubusercontent.com/pciutils/pciids/master/pci.ids",
        AUDIT,
    ),
    "llvm-amdgpu": Source(
        "LLVM AMDGPUUsage, processors table (gfx1153 / gfx1170 / gfx1171 / gfx1250: products TBA)",
        "https://llvm.org/docs/AMDGPUUsage.html",
        AUDIT,
    ),
    # Intel
    "pytorch-xpu": Source(
        "PyTorch 2.14 Getting Started on Intel GPU (Hardware Prerequisite)",
        "https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html",
        AUDIT,
    ),
    "intel-arc-mobile": Source(
        "Intel support 000091109 (Arc A-series mobile graphics memory)",
        "https://www.intel.com/content/www/us/en/support/articles/000091109.html",
        "2026-10-21",
    ),
    "intel-arc-naming": Source(
        "Intel support 000102144: Intel Arc Graphics appears as Intel Graphics (dual-channel memory needed)",
        "https://www.intel.com/content/www/us/en/support/articles/000102144/graphics.html",
        AUDIT,
    ),
    "intel-arc-desktop": Source(
        "Intel Arc A / B-series desktop and Arc Pro B50 / B60 product pages (graphics memory)",
        "https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html",
        "2026-10-21",
    ),
}


@dataclass(frozen=True)
class Rule:
    """表の 1 行。key は名前の正規表現・gfx・PCI の型番 ID("1002:7499")など、value は表ごとに決まる値。"""

    key: str
    value: Any
    source: tuple[str, ...]  # SOURCES の名前
    verified: str = AUDIT  # 一次資料で確かめた日
    added: str = ""  # 足した版(空 = 0.10 以前)
    note: tuple[str, str] = ("", "")  # (日本語, 英語)。判定の注意(判定の理由 notes にも出す行がある)


def _r(key: str, value: Any, *source: str, verified: str = AUDIT, added: str = "", note=("", "")) -> Rule:
    return Rule(key, value, tuple(source), verified, added, note)


@dataclass(frozen=True)
class ValueSet:
    """値の集まりの表(gfx の集まり・名前の一覧など)と、その出典。"""

    name: str
    values: frozenset | tuple
    source: tuple[str, ...]
    verified: str = AUDIT
    note: tuple[str, str] = ("", "")


# --- NVIDIA: GPU 名 → compute capability(古いドライバ用) -------------------------------------------------
# compute_cap を返さない古い nvidia-smi のための表(上から順に試す)。迷う名前(同じ型番で世代が混ざるノート用の
# GT/GTX 7xx・8xx など)は入れない(None = 分からない。断らない)。NVIDIA の一覧の 459 名で照合して誤り 0
# (tests/test_gpu_tables.py)。
_NV = ("nvidia-cuda-gpus", "nvidia-cuda-legacy")
NAME_CC_RULES: tuple[Rule, ...] = (
    # Blackwell
    _r(r"\brtx\s*50[5-9]0\b|\brtx pro \d{4}\b.*\bblackwell\b", (12, 0), *_NV),
    _r(r"\bgb10\b", (12, 1), *_NV),
    _r(r"\bg?b300\b", (10, 3), *_NV),
    _r(r"\bg?b[12]00\b", (10, 0), *_NV),
    # Hopper
    _r(r"\bg?h(100|200|800|20)\b", (9, 0), *_NV),
    # Ada Lovelace。業務用は「RTX 4000 Ada Generation」「RTX 4000 SFF Ada Generation」
    # 「RTX 2000E Ada Generation」
    _r(r"\brtx\s*40[5-9]0\b|\brtx\s*\d{4}e?\s*(?:sff\s*)?ada\b|\bl40s?\b|\bl4\b|\bl20\b", (8, 9), *_NV),
    # Ampere(RTX 2050 のノート用は GA107 で Ampere)
    _r(r"\ba100\b|\ba800\b|\ba30\b", (8, 0), *_NV),
    _r(
        r"\brtx\s*30[5-9]0\b|\brtx\s*2050\b|\brtx\s*a\d{3,4}\b|\ba10g?\b|\ba16\b|\ba40\b|\ba2\b|\bmx570\b",
        (8, 6),
        *_NV,
    ),
    # Turing
    _r(
        r"\brtx\s*20[6-8]0\b|\btitan rtx\b|\bquadro rtx\b|\bgtx\s*16[3-6]0\b|\bt4\b"
        r"|\bt(400|500|550|600|1000|1200|2000)\b|\bmx(450|550)\b",
        (7, 5),
        *_NV,
    ),
    # Volta
    _r(r"\bv100\b|\btitan v\b|\bgv100\b", (7, 0), *_NV),
    # Pascal
    _r(r"\bp100\b|\bgp100\b", (6, 0), *_NV),
    _r(
        r"\bgtx\s*10[5-8]0\b|\bgt\s*1030\b|\btitan xp\b|(?<!gtx )\btitan x\b|\bquadro p\d{3,4}\b"
        r"|\btesla p(4|40)\b|\bmx(150|230|250|330|350)\b",
        (6, 1),
        *_NV,
    ),
    # Maxwell(Quadro K620 / K1200 / K2200 は名前は K だが Maxwell の GM107)。
    # 「GeForce GTX TITAN X」は Maxwell、「NVIDIA TITAN X」は Pascal(NVIDIA の一覧どおり)
    _r(
        r"\bgtx\s*(745|750)\b|\bquadro k(620|1200|2200)\b|\bquadro m(520|1200)\b"
        r"|\bmx1[13]0\b|\b940mx\b|\btesla m10\b",
        (5, 0),
        *_NV,
    ),
    _r(
        r"\bgtx\s*(950|960|970|980)\b|\bgtx titan x\b|\bquadro m(620|2000|2200|4000|5000|6000)\b"
        r"|\btesla m(4|6|40|60)\b",
        (5, 2),
        *_NV,
    ),
    # Kepler(デスクトップの GTX 6xx・7xx、GTX TITAN / Black / Z、Quadro K・Tesla K)
    _r(r"\btesla k80\b", (3, 7), *_NV),
    _r(
        r"\bgtx\s*780\b|\bgtx titan( black| z)?$|\btesla k(20|40)\w*\b|\bgt\s*(710|720)\b"
        r"|\bquadro k(5200|6000)\b",
        (3, 5),
        *_NV,
    ),
    _r(r"\bgtx\s*(6[3-9]0|760|770)\b|\bquadro k\d{3,4}\b", (3, 0), *_NV),
)


# --- NVIDIA: PyTorch の CUDA 版のビルド ------------------------------------------------------------------
TORCH_RELEASE = "2.14.1"


@dataclass(frozen=True)
class CudaBuild:
    tag: str  # cu132 など
    cuda: str
    ccs: dict[str, frozenset[int]]  # 機械("x86_64" / "aarch64")ごとの compute capability(86 = 8.6)
    min_driver_windows: tuple[int, ...]
    min_driver_linux: tuple[int, ...]
    torch: str  # このビルドがある PyTorch の版(案内の文言。版ごとの wheel は download.pytorch.org)
    source: tuple[str, ...] = ("torch-cuda-init", "torch-whl-index", "cuda-rn")
    verified: str = AUDIT
    added: str = ""
    note: tuple[str, str] = field(default=("", ""))


_CU13_X86 = frozenset({75, 80, 86, 90, 100, 120})
_CU13_ARM = frozenset({80, 90, 100, 110, 120})
_CU12_WIN = (525,)
_CU12_LINUX = (525,)
# 今の版(2.14.1、2026-09-30)は cu126 / cu130 / cu132。cu128(2.7〜2.11)・cu129(2.8〜2.13)は今の版には無いが、
# 配布先(download.pytorch.org)に残っていて、ComfyUI の同梱版や AI生成キットのように版を固定して使う側がある
# ので数える(ドライバ 580 未満の Blackwell が動く道)。Windows の cu126 は 6.1 も入れている(Linux は 6.0 の
# 機械語が 6.1 でも動く)。最低のドライバは CUDA のリリースノートの表 3(12.x は 525 以上、13.x は 580 以上。
# OS で分けていない)
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
        source=("torch-build-cuda", "torch-whl-index", "cuda-rn"),
        note=(
            "Windows の cu129 の wheel は 2.8.0・2.9.0 だけ(2.10 以降は Linux だけ)",
            "Windows cu129 wheels exist only for 2.8.0 and 2.9.0 (2.10 and later are Linux only)",
        ),
    ),
    # cu128 の aarch64 は v2.11.0 の .ci/manywheel/build_cuda.sh(12.8 に 12.0 を足し、aarch64 は 8.0 未満と
    # 8.6 を除く)。download.pytorch.org/whl/cu128/torch/ に 2.7.0・2.7.1・2.9.0〜2.11.0 の
    # manylinux_2_28_aarch64 がある(2.8 は無い。DGX Spark の GB10 などの道。#101)
    CudaBuild(
        "cu128",
        "12.8",
        {"x86_64": _CU13_X86, "aarch64": frozenset({80, 90, 100, 120})},
        _CU12_WIN,
        _CU12_LINUX,
        "2.7〜2.11",
        source=("torch-build-cuda", "torch-whl-index", "cuda-rn"),
        note=(
            "aarch64 の wheel は 2.7・2.9〜2.11(2.8 は無い)",
            "aarch64 wheels exist for 2.7 and 2.9 to 2.11 (no 2.8)",
        ),
    ),
    CudaBuild(
        "cu126",
        "12.6",
        {"x86_64": frozenset({50, 60, 61, 70, 75, 80, 86, 90}), "aarch64": frozenset({80, 90})},
        _CU12_WIN,
        _CU12_LINUX,
        "2.6〜2.14",
        source=("torch-cuda-init", "torch-win-cuda", "torch-whl-index", "cuda-rn"),
    ),
)

# 世代ごとの最低のドライバ。Blackwell(10.x・12.x)の機械語が入るのは CUDA 12.8 からで(PyTorch の cu128 以降)、
# CUDA 12.8 GA のドライバは Linux 570.26 / Windows 570.65(CUDA のリリースノートの表 4)
ARCH_MIN_DRIVER_RULES: tuple[Rule, ...] = (
    _r("blackwell", ((570,), "CUDA 12.8 GA(Linux 570.26 / Windows 570.65)"), "cuda-rn"),
)

# Jetson など、同じ世代の機械語が動かない compute capability(PyTorch の DEVICE_REQUIREMENT と同じ)。
# PyTorch v2.14.1 は CUDA 13.2 以降のビルドで 8.0 / 8.6 の機械語を 8.7(Jetson Orin)でも動くとする
# (DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION)が、genmodelfit は Jetson を判定に使わない
# (JetPack 専用の版・統合メモリ。docs/hardware.md「NVIDIA の統合メモリ」)ので、この表は変えない(#103)
SOLO_CC = ValueSet(
    "solo_cc",
    frozenset({53, 62, 72, 87, 101, 110}),
    ("torch-cuda-init",),
    note=(
        "CUDA 13.2 以降は 8.7 で 8.0 / 8.6 の機械語が動く(PyTorch v2.14.1)が、Jetson は対象外なので変えない",
        "From CUDA 13.2, 8.0 / 8.6 code runs on 8.7 (PyTorch v2.14.1), but Jetson is out of scope,"
        " so unchanged",
    ),
)

# ComfyUI の should_use_fp16 の名前の一覧と同じもの(comfyui-mm)。Pascal のこれらは Windows では fp16 で
# 動かし、Linux では fp32 で計算する。下の一覧(GTX 16 系など)は fp16 が壊れているとして fp32 で計算する
# (こちらは大文字・小文字を区別する)
COMFY_FP16_PASCAL = ValueSet(
    "nvidia_10_series",
    (
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
    ),
    ("comfyui-mm",),
)
COMFY_FP16_BROKEN = ValueSet(
    "nvidia_16_series",
    ("1660", "1650", "1630", "T500", "T550", "T600", "MX550", "MX450", "CMP 30HX", "T2000", "T1000", "T1200"),
    ("comfyui-mm",),
)


# --- AMD: gfx → 世代 ---------------------------------------------------------------------------------------
# 世代の名前は TheRock の SUPPORTED_GPUS.md と ROCm の hw specs の Architecture の欄による
_TR = ("therock-supported",)
AMD_ARCH_RULES: tuple[Rule, ...] = (
    *(_r(g, "RDNA 4", *_TR, "rocm-hw-specs") for g in ("gfx1200", "gfx1201")),
    # gfx1170 / gfx1171(RDNA 4 系の APU)は TheRock で Build Passing だけ(Sanity Tested・Release Ready で
    # ない)。LLVM の製品名は TBA。名前は世代の表示に使い、判定は AMD_UNVERIFIED_GFX で断る(#103)
    *(_r(g, "RDNA 4", *_TR, "llvm-amdgpu", "comfyui-mm", added="0.11.0") for g in ("gfx1170", "gfx1171")),
    *(_r(g, "RDNA 3.5", *_TR, "rocm-hw-specs") for g in ("gfx1150", "gfx1151", "gfx1152")),
    _r("gfx1153", "RDNA 3.5", *_TR, "therock-releases"),
    *(_r(g, "RDNA 3", *_TR, "rocm-hw-specs") for g in ("gfx1100", "gfx1101", "gfx1102", "gfx1103")),
    *(
        _r(g, "RDNA 2", *_TR)
        for g in ("gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035", "gfx1036")
    ),
    *(_r(g, "RDNA 1", *_TR) for g in ("gfx1010", "gfx1011", "gfx1012")),
    *(_r(g, "CDNA", *_TR, "rocm-hw-specs") for g in ("gfx908", "gfx90a", "gfx942", "gfx950")),
    # gfx1250(CDNA 5、Instinct)は TheRock の Linux で Sanity Tested まで(Release Ready でない)。判定は
    # AMD_UNVERIFIED_GFX で断る(#103)
    _r("gfx1250", "CDNA", *_TR, "torch-build-rocm", "llvm-amdgpu", added="0.11.0"),
    *(_r(g, "GCN 5", *_TR, "rocm-hw-specs") for g in ("gfx900", "gfx906", "gfx90c")),
    # Polaris(Instinct MI6 / MI8・RX 400 / 500 系)。ComfyUI の AMD_RDNA2_AND_OLDER_ARCH にあり、hw specs は
    # gfx803。対象外の世代として表示に使うだけ(#103)
    _r("gfx803", "GCN 4", "rocm-hw-specs", "comfyui-mm", added="0.11.0"),
)
AMD_SUPPORTED_ARCH = ("RDNA 2", "RDNA 3", "RDNA 3.5", "RDNA 4")  # Windows・Linux
AMD_LINUX_ONLY_ARCH = ("CDNA",)  # Instinct は Linux だけ(ComfyUI の Windows の案内は Radeon だけ)
# RDNA 2 以前は bf16 を持たない。ComfyUI は fp16 を持たない系統でも bf16 で計算する(should_use_bf16 の
# manual_cast。model_management.py の AMD_RDNA2_AND_OLDER_ARCH)ので、作業用メモリは倍にならない。VAE は fp32
AMD_NO_BF16_ARCH = ("RDNA 2", "RDNA 1", "GCN 5", "GCN 4")

# 対象の世代でも使わない gfx と、その理由(判定の文言は hardware.classify_adapter)
AMD_DISPLAY_ONLY = ValueSet(
    "display_only_gfx",
    # 画面を出すための 2 CU の内蔵 GPU(Raphael・Granite Ridge = Ryzen 7000 / 9000 のデスクトップ、Mendocino の
    # 610M)。生成では CPU より速い見込みが無い(判断。実測なし)。Van Gogh(Steam Deck、gfx1033)は 8 CU で
    # GPU として使う(0.9.0 で直した)
    frozenset({"gfx1036"}),
    ("amd-product-pages", "therock-supported"),
    verified="2026-10-22",
    note=(
        "2 CU の画面用の内蔵 GPU(判断。実測なし)",
        "2-CU display-only integrated GPU (a judgment; no measurement)",
    ),
)
AMD_UNVERIFIED = ValueSet(
    "unverified_gfx",
    # TheRock の SUPPORTED_GPUS.md で Release Ready でない gfx。gfx117x は Linux・Windows とも Build Passing
    # だけ、gfx1250 は Linux で Sanity Tested まで。TheRock は「Sanity Tested が付くまでは pip で入っても
    # 動くとは限らない」と書く。Release Ready になれば外す(毎週の見張りが知らせる。#103)
    frozenset({"gfx1170", "gfx1171", "gfx1250"}),
    ("therock-supported",),
    note=(
        "TheRock で Release Ready でない(gfx117x は Build Passing だけ、gfx1250 は Sanity Tested まで)",
        "Not Release Ready in TheRock (gfx117x: Build Passing only; gfx1250: up to Sanity Tested)",
    ),
)
# pytorch.org の ROCm 版(Linux だけ。ComfyUI の Linux の案内)が入れる gfx(v2.14.1 の PYTORCH_ROCM_ARCH)。
# それ以外の RDNA 2・3.5(gfx1031〜1036・gfx1152・1153 など)は TheRock の版を使う、と notes に書く
PYTORCH_ROCM_LINUX = ValueSet(
    "pytorch_rocm_linux_gfx",
    frozenset(
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
    ),
    ("torch-build-rocm",),
)


# --- AMD: GPU 名 → (gfx, 内蔵か) --------------------------------------------------------------------------
# Windows の WMI・Linux の lspci の製品名・開発名で引く(上から順に試す)。Linux は amdkfd の gfx_target_version
# が先(#101)なので、名前が要るのは主に Windows
_TRR = ("therock-releases", "rocm-hw-specs")
_PCI = ("pci-ids",)
AMD_NAME_RULES: tuple[Rule, ...] = (
    _r(r"rx\s*9070|ai pro r9[67]00|navi\s*48\b", ("gfx1201", False), *_TRR),
    _r(r"rx\s*90[456]0|navi\s*44\b", ("gfx1200", False), *_TRR),  # RX 9060 / 9060 XT は Navi 44
    # Radeon 8065S(Ryzen AI Max+ PRO 495)は hw specs で gfx1151(#103)
    _r(r"8065s|8060s|8050s|8040s|strix halo", ("gfx1151", True), *_TRR, added="0.11.0"),
    _r(r"890m|880m|strix(?! halo)", ("gfx1150", True), *_TRR),
    # 820M は公式の資料どうしで食い違う: TheRock の RELEASES.md は gfx1153、ROCm の hw specs は
    # Ryzen AI 5 330 の Radeon 820M を gfx1152。pci.ids は 1902「Krackan2 [Radeon 820M / 840M]」で、
    # 別の石(Krackan2)に 820M と 840M が載る(820M・840M の gfx は名前だけでは決まらない)。
    # ROCm の wheel を選ぶ TheRock に合わせて gfx1153 にし、注意を出す。どちらも RDNA 3.5 で、
    # 判定(使う・pytorch.org の ROCm 版に無い)は同じ。
    # Linux は amdkfd の値が先。1902 は型番 ID の表に入れない。新しい pci.ids の lspci の名前「Radeon 820M /
    # 840M」(1902)もこの行に当たるよう、860M / 840M の行より前に置く(#103)
    _r(
        r"820m",
        ("gfx1153", True),
        "therock-releases",
        "rocm-hw-specs",
        "pci-ids",
        note=(
            "820M の gfx は公式の資料どうしで食い違う(TheRock の RELEASES.md は gfx1153、ROCm の hw specs は"
            " gfx1152)。TheRock の wheel が合わなければ gfx1152 の版を試す",
            "The gfx of the 820M differs between official sources (TheRock RELEASES.md: gfx1153;"
            " ROCm hw specs: gfx1152). If the TheRock wheel does not work, try the gfx1152 one",
        ),
    ),
    _r(r"860m|840m|krackan", ("gfx1152", True), *_TRR),
    _r(
        r"rx\s*7[67]00[sm]\b|rx\s*7600m xt", ("gfx1102", False), *_PCI
    ),  # ノート用の RX 7700S / 7600S / 7600M は Navi 33
    _r(r"rx\s*7900|pro w7[89]00|navi\s*31\b", ("gfx1100", False), *_TRR),
    _r(r"rx\s*7[78]00|pro w7700|pro v710|navi\s*32\b", ("gfx1101", False), *_TRR),
    # RX 7400 / 7300 / 7500・PRO W7400 も Navi 33(pci.ids 7499「Navi 33 [Radeon RX 7400/7300/Pro W7400]」・
    # 749f「Navi 33 [Radeon RX 7500]」。#103)
    _r(r"rx\s*7[3-5]00\b|pro w7400", ("gfx1102", False), *_PCI, added="0.11.0"),
    _r(r"rx\s*76[05]0|pro w7[56]00|navi\s*33\b", ("gfx1102", False), *_TRR, *_PCI),
    # Valve の Steam Machine の GPU(pci.ids 7481「Navi 33 [Steam Machine]」。#103)
    _r(r"steam machine", ("gfx1102", False), *_PCI, added="0.11.0"),
    _r(r"780m|760m|740m|phoenix|hawk ?point", ("gfx1103", True), *_TRR),
    # ノート用(M・S)は同じ番号でも石が違う: RX 6850M XT / 6800M / 6700M は Navi 22、RX 6800S / 6700S / 6650M /
    # 6600M / 6600S は Navi 23、RX 6550M / 6500M / 6450M / 6300M は Navi 24
    _r(r"rx\s*6[78][05]0m\b", ("gfx1031", False), *_PCI, verified="2026-10-22"),
    _r(r"rx\s*6[678]00s\b|rx\s*66[05]0m\b", ("gfx1032", False), *_PCI, verified="2026-10-22"),
    _r(r"rx\s*6[345][05]0m\b", ("gfx1034", False), *_PCI, verified="2026-10-22"),
    _r(r"rx\s*6[89]\d0|pro w6800|pro w6900x|pro v620|navi\s*21\b", ("gfx1030", False), *_TRR, added="0.11.0"),
    _r(r"rx\s*67\d0|navi\s*22\b", ("gfx1031", False), *_TRR),
    _r(r"rx\s*66\d0|pro w6600|navi\s*23\b", ("gfx1032", False), *_TRR),
    # デスクトップの RX 6300 も Navi 24(pci.ids「7424 Navi 24 [Radeon RX 6300]」、Pro W6300 / W6400 も同じ)
    _r(r"rx\s*6[345]00|pro w6[345]00|navi\s*24\b", ("gfx1034", False), *_PCI, verified="2026-10-22"),
    _r(r"680m|660m|rembrandt", ("gfx1035", True), "therock-supported", *_PCI, verified="2026-10-22"),
    _r(
        r"610m|raphael|granite ridge|mendocino",
        ("gfx1036", True),
        "amd-product-pages",
        *_PCI,
        verified="2026-10-22",
    ),
    # Steam Deck(Van Gogh、OLED 版の Sephiroth も gfx1033)。Windows のドライバの名前は「AMD Custom GPU 0405」
    # (pci.ids「163f VanGogh [AMD Custom GPU 0405]」「1435 Sephiroth [AMD Custom GPU 0405]」)
    _r(
        r"van ?gogh|sephiroth|custom gpu 0405",
        ("gfx1033", True),
        "therock-releases",
        "steam-deck-spec",
        *_PCI,
    ),
    _r(r"rx\s*5[67]\d0|navi\s*10\b", ("gfx1010", False), *_PCI, verified="2026-10-21"),
    _r(r"pro v520|navi\s*12\b", ("gfx1011", False), "therock-releases", *_PCI, added="0.11.0"),
    _r(r"rx\s*5[35]00|pro w5500|navi\s*14\b", ("gfx1012", False), *_PCI, verified="2026-10-21"),
    _r(r"mi35\dx?", ("gfx950", False), *_TRR),
    _r(r"mi3[02]\dx?|mi300a", ("gfx942", False), *_TRR),
    _r(r"mi2[15]0x?", ("gfx90a", False), *_TRR),
    _r(r"mi100", ("gfx908", False), *_TRR),
    _r(r"radeon vii|radeon pro vii|mi[56]0\b|vega\s*20", ("gfx906", False), *_TRR),
    # 対象外の世代の Instinct(表示のため。#103)
    _r(r"mi25\b", ("gfx900", False), *_TRR, added="0.11.0"),
    _r(r"\bmi[68]\b", ("gfx803", False), "rocm-hw-specs", added="0.11.0"),
)

# 名前が「AMD Radeon(TM) Graphics」だけの内蔵 GPU は、CPU の名前で決める(上から順に試す。迷うものは入れない)
AMD_CPU_IGPU_RULES: tuple[Rule, ...] = (
    _r(r"ryzen ai max", "gfx1151", "rocm-hw-specs"),
    _r(r"ryzen ai 9 (hx )?(pro )?[34][67]\d", "gfx1150", "rocm-hw-specs", added="0.11.0"),
    # Ryzen AI 5 330(Radeon 820M)は名前の表と同じく gfx1153(上の 820M の行の注意。#103)
    _r(
        r"ryzen ai 5 (pro )?330\b",
        "gfx1153",
        "therock-releases",
        "rocm-hw-specs",
        added="0.11.0",
        note=(
            "Ryzen AI 5 330 の 820M の gfx は公式の資料どうしで食い違う(TheRock は gfx1153、"
            "ROCm の hw specs は gfx1152)",
            "The gfx of the 820M in the Ryzen AI 5 330 differs between official sources (TheRock: gfx1153;"
            " ROCm hw specs: gfx1152)",
        ),
    ),
    _r(r"ryzen ai [57] (pro )?[34][3-5]\d", "gfx1152", "rocm-hw-specs", added="0.11.0"),
    # Dragon Range(7945HX・7940HX・7840HX・7745HX・7645HX など)・Fire Range(9955HX・9850HX など)は
    # Radeon 610M(2 CU)の内蔵 GPU(gfx1036、AMD_DISPLAY_ONLY)。AMD の製品ページの「Graphics Model:
    # AMD Radeon 610M」「Graphics Core Count: 2」
    _r(
        r"ryzen [3579] (pro )?7\d4[05]hx3?d?\b|ryzen [3579] (pro )?9\d5[05]hx3?d?\b",
        "gfx1036",
        "amd-product-pages",
        verified="2026-10-22",
    ),
    # 携帯機の Ryzen Z1 Extreme(12 CU)・Z1(4 CU)は RDNA 3 の「AMD Radeon Graphics」(AMD の製品ページ)。
    # 12 CU・Zen 4 8 個は Phoenix1(780M)、4 CU・Zen 4 2 個 + Zen 4c 4 個は Phoenix2(740M)と同じ石とみなした
    # (推定)。どちらも gfx1103(Linux の amdkfd kfd_device.c: GC 11.0.1・11.0.4 → gfx_target_version 110003)
    _r(r"ryzen z1\b", "gfx1103", "amd-product-pages", "linux-amdkfd", verified="2026-10-22"),
    # Ryzen 9 270(Hawk Point の 200 系)は TheRock の RELEASES.md で gfx1103(#103。ほかの 200 系は資料で
    # 確かめられなかったので入れない)
    _r(
        r"ryzen [3579] (pro )?[78]\d4\d(u|hs|h)\b|ryzen [3579] (pro )?8[5-7]00g|ryzen 9 (pro )?270\b",
        "gfx1103",
        "therock-releases",
        "amd-product-pages",
        added="0.11.0",
    ),
    _r(
        r"ryzen [3579] (pro )?(6[89]\d0|7[37]35)(u|hs|h|hx)\b",
        "gfx1035",
        "therock-supported",
        verified="2026-10-22",
    ),
)


# --- Intel: GPU 名 → (系統, 内蔵か, 名前から推す VRAM の下限 GiB) -----------------------------------------
# PyTorch 2.14 の「Getting Started on Intel GPU」の Client GPUs(Windows 11・Ubuntu 24.04/26.04・WSL2)と
# Intel Data Center GPU(Linux だけ)の検証済みの部品だけ。UHD・Iris Xe・「Intel(R) Graphics」(Arrow Lake-S
# などの Xe-LPG)は入らない
_XPU = ("pytorch-xpu",)
INTEL_DATACENTER = "Data Center GPU Max (Ponte Vecchio)"
INTEL_XPU_RULES: tuple[Rule, ...] = (
    _r(
        r"arc.*\ba770\b", ("Arc A-Series (Alchemist)", False, 8.0), *_XPU, "intel-arc-desktop"
    ),  # 8 GB / 16 GB
    _r(r"arc.*\ba750\b|arc.*\ba580\b", ("Arc A-Series (Alchemist)", False, 8.0), *_XPU, "intel-arc-desktop"),
    _r(r"arc.*\ba380\b", ("Arc A-Series (Alchemist)", False, 6.0), *_XPU, "intel-arc-desktop"),
    _r(r"arc.*\ba310\b", ("Arc A-Series (Alchemist)", False, 4.0), *_XPU, "intel-arc-desktop"),
    # ノート用も Alchemist の外付け。公称の VRAM は Intel の表(A770M 16GB・A730M 12GB・A550M 8GB・A370M 4GB・
    # A350M 4GB)。A570M・A530M・Pro の M・組み込み用(E)は Intel の資料で確かめられなかったので値を持たない
    _r(r"arc.*\ba770m\b", ("Arc A-Series (Alchemist)", False, 16.0), *_XPU, "intel-arc-mobile"),
    _r(r"arc.*\ba730m\b", ("Arc A-Series (Alchemist)", False, 12.0), *_XPU, "intel-arc-mobile"),
    _r(r"arc.*\ba550m\b", ("Arc A-Series (Alchemist)", False, 8.0), *_XPU, "intel-arc-mobile"),
    _r(r"arc.*\ba3[57]0m\b", ("Arc A-Series (Alchemist)", False, 4.0), *_XPU, "intel-arc-mobile"),
    _r(
        r"arc.*\bpro a\d{2}m?\b|arc.*\ba\d{3}[me]\b|\bdg2\b",
        ("Arc A-Series (Alchemist)", False, None),
        *_XPU,
        *_PCI,
    ),
    _r(r"arc.*\bb580\b", ("Arc B-Series (Battlemage)", False, 12.0), *_XPU, "intel-arc-desktop"),
    _r(r"arc.*\bb570\b", ("Arc B-Series (Battlemage)", False, 10.0), *_XPU, "intel-arc-desktop"),
    _r(r"arc.*\bpro b60\b", ("Arc B-Series (Battlemage)", False, 24.0), *_XPU, "intel-arc-desktop"),
    _r(r"arc.*\bpro b50\b", ("Arc B-Series (Battlemage)", False, 16.0), *_XPU, "intel-arc-desktop"),
    # Arc Pro B65 / B70(Battlemage G31、pci.ids e222・e223。#103)。公称の VRAM は Intel の資料で
    # 確かめられなかったので持たない
    _r(r"arc.*\bpro b[67][05]\b", ("Arc B-Series (Battlemage)", False, None), *_XPU, *_PCI, added="0.11.0"),
    _r(r"battlemage|\bbmg\b", ("Arc B-Series (Battlemage)", False, None), *_XPU, *_PCI),
    _r(r"arc.*\bb3\d0\b|panther lake", ("Core Ultra Series 3 (Panther Lake)", True, None), *_XPU, *_PCI),
    _r(r"arc.*\b1[34]0v\b|lunar lake", ("Core Ultra Series 2 (Lunar Lake)", True, None), *_XPU, *_PCI),
    _r(r"arc.*\b1[34]0t\b|arrow lake-[hp]", ("Core Ultra Series 2 (Arrow Lake-H)", True, None), *_XPU, *_PCI),
    _r(r"^(intel )?arc graphics$|meteor lake", ("Core Ultra (Meteor Lake-H)", True, None), *_XPU, *_PCI),
    # Data Center GPU Max(Ponte Vecchio)は Linux だけ(PyTorch の表: RHEL 9.2・SLES 15 SP5・
    # Ubuntu Server 22.04)。公称の VRAM は持たない(#103)
    _r(
        r"data center gpu max|ponte vecchio|\bpvc\b",
        (INTEL_DATACENTER, False, None),
        *_XPU,
        *_PCI,
        added="0.11.0",
        note=("Linux だけ", "Linux only"),
    ),
)
INTEL_LINUX_ONLY = frozenset({INTEL_DATACENTER})


# --- PCI の型番 ID → 名前(古い pci.ids・lspci が無いとき) ---------------------------------------------------
# lspci が「AMD Device 7551」のように型番の ID しか出さないとき、または製品名が汎用(「Radeon Graphics」
# 「Intel Graphics」)のときに引く。pci.ids 2026.10.27 の行から、判定に使う GPU と、使わないと決めている
# 内蔵 GPU(610M など)だけを写した。石の名前(Navi 31 など)で引くので、ノート用・Pro の違いは gfx に効かない。
# 「Intel Graphics」とだけある ID(Arc でない内蔵 GPU など)は入れない
def _pci(names: dict[str, str], *, verified: str = AUDIT, added: str = "") -> tuple[Rule, ...]:
    return tuple(_r(k, v, *_PCI, verified=verified, added=added) for k, v in names.items())


PCI_DEVICE_RULES: tuple[Rule, ...] = (
    *_pci({f"1002:{d}": "Navi 48" for d in ("7550", "7551")}),
    *_pci({"1002:7590": "Navi 44"}),
    *_pci({f"1002:{d}": "Navi 31" for d in ("744c", "7448", "7449", "744a", "744b", "745e")}),
    *_pci({f"1002:{d}": "Navi 32" for d in ("7460", "7461", "7470", "747e")}),
    *_pci(
        {
            f"1002:{d}": "Navi 33"
            for d in ("7480", "7481", "7483", "7487", "7489", "748b", "7499", "749f", "73f0")
        }
    ),
    *_pci({f"1002:{d}": "Navi 21" for d in ("73a1", "73a2", "73a3", "73a5", "73ab", "73ae", "73af", "73bf")}),
    *_pci({f"1002:{d}": "Navi 22" for d in ("73c3", "73df")}),
    *_pci({f"1002:{d}": "Navi 23" for d in ("73e0", "73e1", "73e3", "73ef", "73ff")}),
    *_pci({f"1002:{d}": "Navi 24" for d in ("7421", "7422", "7423", "7424", "743f")}),
    *_pci({f"1002:{d}": "Phoenix" for d in ("15bf", "15c8")}),
    *_pci({f"1002:{d}": "Hawk Point" for d in ("1900", "1901")}),
    *_pci(
        {
            "1002:1681": "Rembrandt",
            "1002:164e": "Raphael",
            "1002:13c0": "Granite Ridge",
            "1002:1506": "Mendocino",
            "1002:163f": "Van Gogh",
            "1002:1435": "Van Gogh (Sephiroth)",
            "1002:150e": "Strix",
            "1002:1586": "Strix Halo",
            "1002:1114": "Krackan",
        }
    ),
    *_pci(
        {
            "8086:56a0": "Arc A770",
            "8086:56a1": "Arc A750",
            "8086:56a2": "Arc A580",
            "8086:56a5": "Arc A380",
            "8086:56a6": "Arc A310",
            "8086:5690": "Arc A770M",
            "8086:5691": "Arc A730M",
            "8086:5692": "Arc A550M",
            "8086:5693": "Arc A370M",
            "8086:5694": "Arc A350M",
            "8086:5696": "Arc A570M",
            "8086:5697": "Arc A530M",
            "8086:56b0": "Arc Pro A30M",
            "8086:56b1": "Arc Pro A40",
            "8086:56b2": "Arc Pro A60M",
            "8086:56b3": "Arc Pro A60",
            "8086:e20b": "Arc B580",
            "8086:e20c": "Arc B570",
            "8086:e211": "Arc Pro B60",
            "8086:e212": "Arc Pro B50",
        }
    ),
    *_pci({f"8086:{d}": "Battlemage" for d in ("e202", "e20d", "e210", "e215", "e216", "e220", "e221")}),
    *_pci({"8086:e222": "Arc Pro B65", "8086:e223": "Arc Pro B70"}, added="0.11.0"),
    # Core Ultra の内蔵の Arc。メモリが 1 チャンネルだと Windows でも「Intel(R) Graphics」と出る
    # (intel-arc-naming)ので、汎用の名前のときも型番 ID で引く(#103。理由に 1 チャンネルは遅いと書く)
    *(
        _r(k, v, *_PCI, "intel-arc-naming")
        for k, v in {
            "8086:7d55": "Arc Graphics",  # Meteor Lake-P [Intel Arc Graphics]
            "8086:7d51": "Arc 140T",  # Arrow Lake-P [Arc Pro 130T/140T]
            "8086:64a0": "Arc 140V",  # Lunar Lake の Arc Graphics 130V/140V
        }.items()
    ),
    *_pci({f"8086:{d}": "Arc B390" for d in ("b080", "b082", "b084", "b086")}),  # Panther Lake
    *_pci({f"8086:{d}": "Arc B370" for d in ("b081", "b083", "b085", "b087")}),
    *_pci(
        {
            f"8086:{d}": "Data Center GPU Max"
            for d in ("0b69", "0b6e", "0bd0", "0bd4", "0bd5", "0bd6", "0bd7", "0bd8", "0bd9", "0bda", "0bdb")
        },
        added="0.11.0",
    ),
)


# --- WSL2 で使える AMD の GPU(AMD の WSL の対応表の製品名) ------------------------------------------------
# 名前は hardware._wsl_amd_key の形(会社名・Radeon・(TM) を除いた小文字)。APU は表に無い
WSL_AMD = ValueSet(
    "wsl_amd_supported",
    frozenset(
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
    ),
    ("amd-wsl-compat",),
)


# 公式の一覧にあるが、わざと表で扱わない・値が違うもの(試験と見張りが候補にしない。理由を書く)
KNOWN_EXCEPTIONS: dict[str, tuple[str, str]] = {
    # Windows の cu126 は 6.1 も入れている(torch-win-cuda)。PYTORCH_RELEASES_CODE_CC は Linux の一覧
    "torch:cu126 x86_64": (
        "Windows の cu126 は 6.1 も入れている(.ci/pytorch/windows/internal/cuda_config.bat)",
        "The Windows cu126 build also has 6.1 (.ci/pytorch/windows/internal/cuda_config.bat)",
    ),
    # ROCm の hw specs の Ryzen AI 5 330 / Radeon 820M = gfx1152 は、TheRock の RELEASES.md(gfx1153)と食い違う
    "amd:Radeon 820M=gfx1152": (
        "820M は TheRock の gfx1153 に合わせた(公式の資料どうしの食い違い。820M の行の注意)",
        "820M follows TheRock's gfx1153 (official sources disagree; see the note on the 820M row)",
    ),
}


def rules_by_table() -> dict[str, tuple[Rule, ...]]:
    """一覧(docs/gpu-inventory.md)と見張りが使う、表の名前 → 行。"""
    return {
        "NAME_CC_RULES": NAME_CC_RULES,
        "ARCH_MIN_DRIVER_RULES": ARCH_MIN_DRIVER_RULES,
        "AMD_ARCH_RULES": AMD_ARCH_RULES,
        "AMD_NAME_RULES": AMD_NAME_RULES,
        "AMD_CPU_IGPU_RULES": AMD_CPU_IGPU_RULES,
        "INTEL_XPU_RULES": INTEL_XPU_RULES,
        "PCI_DEVICE_RULES": PCI_DEVICE_RULES,
    }


def value_sets() -> tuple[ValueSet, ...]:
    return (
        SOLO_CC,
        COMFY_FP16_PASCAL,
        COMFY_FP16_BROKEN,
        AMD_DISPLAY_ONLY,
        AMD_UNVERIFIED,
        PYTORCH_ROCM_LINUX,
        WSL_AMD,
    )
