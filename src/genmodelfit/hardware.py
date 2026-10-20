# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""PC の検出。GPU は nvidia-smi で確かめた値だけを使う(レジストリには取り外した GPU も残るため)。"""

from __future__ import annotations

import contextlib
import os
import platform
import shutil
import subprocess
from pathlib import Path

import psutil

from .models import Gpu, SystemInfo

GIB = 1024**3


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    if os.name == "nt":  # PATH に無くても System32 に置かれることが多い
        cand = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        if cand.exists():
            return str(cand)
    return None


def parse_nvidia_smi(text: str) -> list[Gpu]:
    """`nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader,nounits`
    の出力を読む。"""
    gpus: list[Gpu] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        name, total, free, driver = parts[:4]
        try:
            total_gb = float(total) / 1024
            free_gb = float(free) / 1024
        except ValueError:
            continue
        gpus.append(
            Gpu(
                name=name,
                vendor="nvidia",
                vram_total_gb=round(total_gb, 2),
                vram_free_gb=round(free_gb, 2),
                driver=driver,
                backend="cuda",
            )
        )
    return gpus


def detect_gpus() -> list[Gpu]:
    exe = _nvidia_smi()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [
                exe,
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


def _cpu_name() -> str:
    name = platform.processor() or platform.machine()
    if os.name == "nt":  # platform.processor() は「Intel64 Family 6 ...」になるので製品名を読む
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return name
    if os.name != "nt":
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name


def detect(
    disk_path: str | os.PathLike[str] | None = None,
    vram_gb: float | None = None,
    ram_gb: float | None = None,
    gpus: list[Gpu] | None = None,
) -> SystemInfo:
    """PC を調べる。vram_gb / ram_gb を渡すと、その値で上書きする(別の PC を想定した判定用)。"""
    vm = psutil.virtual_memory()
    found = detect_gpus() if gpus is None else gpus
    overrides: list[str] = []
    if vram_gb is not None:
        overrides.append("vram")
        if found:
            g = max(found, key=lambda x: x.vram_total_gb)
            found = [
                Gpu(
                    name=g.name,
                    vendor=g.vendor,
                    vram_total_gb=vram_gb,
                    vram_free_gb=None,
                    driver=g.driver,
                    backend="cuda",
                )
            ]
        elif vram_gb > 0:
            found = [
                Gpu(name=f"(仮定) {vram_gb:g} GB GPU", vendor="nvidia", vram_total_gb=vram_gb, backend="cuda")
            ]
    ram_total = vm.total / GIB
    ram_avail = vm.available / GIB
    if ram_gb is not None:
        overrides.append("ram")
        ram_total = ram_avail = ram_gb
    disk_free = None
    with contextlib.suppress(OSError):
        disk_free = shutil.disk_usage(disk_path or Path.home()).free / GIB
    return SystemInfo(
        os=f"{platform.system()} {platform.release()}",
        cpu=_cpu_name(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(ram_total, 2),
        ram_available_gb=round(ram_avail, 2),
        gpus=found,
        disk_free_gb=round(disk_free, 1) if disk_free is not None else None,
        overrides=overrides,
    )
