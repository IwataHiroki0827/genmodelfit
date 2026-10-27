# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""すべての環境で検出する(#101。docs/hardware.md「すべての環境」)の試験。

外のコマンド(nvidia-smi・PowerShell・rocminfo・clinfo・sysctl・sw_vers)は hardware._run の差し替え、
/proc・/sys・/dev・/mnt・/opt は一時ディレクトリの木(hardware._ROOT)で再現する。本物の機械は読まない
(conftest の _no_real_hardware。sysfs・amdkfd・WSL の状態を読む元の関数は、この試験の中でだけ戻す)。
"""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import psutil
import pytest

from genmodelfit import hardware
from genmodelfit.estimate import Coefficients, estimate
from genmodelfit.hardware import (
    GIB,
    Host,
    Memory,
    classify_adapter,
    classify_gpu,
    detect,
    parse_lspci,
    parse_nvidia_smi,
    parse_system_profiler,
    parse_wmi_video,
)
from genmodelfit.models import Gpu, ModelEntry, ModelFile, SystemInfo

FIX = Path(__file__).parent / "fixtures" / "hardware"
GB = 2**30
MEM = Memory(total=32 * GB, available=24 * GB)
LINUX = Host("Linux", cpu="AMD Ryzen 9 7950X")
WIN11 = Host("Windows", 26100, cpu="AMD Ryzen 9 7950X")
MAC15 = Host("Darwin", mac_version=(15, 1))
UUID = "7f3a9c2e4b1d6a58"  # rocminfo のフィクスチャの GPU の Uuid

# conftest が塞ぐ前の元の関数(この試験の中でだけ戻す)
REAL_RUN = hardware._run
REAL_SYSFS_SLOTS = hardware._sysfs_slots
REAL_KFD = hardware.kfd_gfx_targets
REAL_WSL_STATE = hardware.wsl_state


def fx(name: str) -> str:
    return (FIX / name).read_bytes().decode("utf-8")


def tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel.lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    return root


class Router:
    """_run の差し替え: コマンドの名前の一部 → (終了コード, 出力)。呼ばれたコマンドを覚える。"""

    def __init__(self, table: dict[str, tuple[int, str] | None]):
        self.table = table
        self.calls: list[list[str]] = []

    def __call__(self, args, timeout):
        assert timeout > 0
        self.calls.append(list(args))
        head = " ".join(str(a) for a in args)
        for key, out in self.table.items():
            if key in head:
                return out
        return None


def linux(monkeypatch, root: Path | None = None):
    monkeypatch.setattr(hardware, "_windows", lambda: False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    if root is not None:
        monkeypatch.setattr(hardware, "_ROOT", str(root))


def entry(gb: float = 6.0) -> ModelEntry:
    files = [
        ModelFile(name="unet_fp16.safetensors", size_bytes=int(gb * GB), directory="diffusion_models"),
        ModelFile(name="clip_l.safetensors", size_bytes=int(0.25 * GB), directory="text_encoders"),
        ModelFile(name="vae.safetensors", size_bytes=int(0.3 * GB), directory="vae"),
    ]
    return ModelEntry(id="t", name="t", modality="image", files=files, precision="fp16")


# --- 1. メモリ(psutil の失敗・古いカーネル) ------------------------------------------------------


def _psutil_fails(monkeypatch):
    def boom():
        raise OSError("psutil failed")

    monkeypatch.setattr(psutil, "virtual_memory", boom)


def test_psutil_failure_falls_back_to_proc_meminfo(monkeypatch, tmp_path):
    _psutil_fails(monkeypatch)
    linux(monkeypatch, tree(tmp_path, {"/proc/meminfo": fx("meminfo_gb10.txt")}))
    m = hardware.read_memory()
    assert m.total == 128452404 * 1024 and m.available == 119876544 * 1024
    assert any("/proc/meminfo" in n for n in m.notes)


def test_psutil_failure_on_windows_uses_global_memory_status(monkeypatch):
    _psutil_fails(monkeypatch)
    monkeypatch.setattr(hardware, "_windows", lambda: True)
    monkeypatch.setattr(
        hardware,
        "_global_memory_status",
        lambda: {
            "total_phys": 32 * GB,
            "avail_phys": 8 * GB,
            "commit_limit": 40 * GB,
            "commit_avail": 9 * GB,
        },
    )
    m = hardware.read_memory()
    assert (m.total, m.available) == (32 * GB, 8 * GB)


def test_psutil_failure_on_macos_uses_sysctl(monkeypatch):
    _psutil_fails(monkeypatch)
    monkeypatch.setattr(hardware, "_windows", lambda: False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware, "_run", Router({"sysctl": (0, "17179869184\n")}))
    m = hardware.read_memory()
    assert m.total == 16 * GB and m.available is None


def test_memory_unreadable_everywhere_does_not_raise(monkeypatch, tmp_path):
    _psutil_fails(monkeypatch)
    linux(monkeypatch, tmp_path)  # /proc/meminfo が無い
    m = hardware.read_memory()
    assert m.total is None and any("--ram" in n for n in m.notes)
    s = detect(memory=m, gpus=[])  # 例外にしない
    assert s.ram_total_gb == 0 and any("--ram" in n for n in s.notes)


def test_old_kernel_without_memavailable(monkeypatch, tmp_path):
    linux(monkeypatch, tree(tmp_path, {"/proc/meminfo": fx("meminfo_old_kernel.txt")}))
    mi = hardware.read_meminfo()
    assert hardware.meminfo_available(mi) == (1048576 + 2097152) * 1024  # MemFree + Cached


# --- 2. Apple の仮想の画面・7. macOS の Apple 以外 --------------------------------------------------


def test_apple_paravirtual_is_cpu():
    a = parse_system_profiler(fx("system_profiler_paravirtual.json"))[0]
    assert a.vendor == "apple"
    assert classify_adapter(a, MAC15, 16.0) is None
    assert a.usable is False and "チップ名でない" in a.reason


def test_macos_non_apple_gpus_are_out_of_scope_first():
    for a in parse_system_profiler(fx("system_profiler_intel_mac_amd.json")):
        assert classify_adapter(a, MAC15, 32.0) is None
        assert "Apple 以外" in a.reason, a.reason
        assert "gfx" not in a.reason


# --- 3. 表より新しい世代・12. 機械の種類・13. MIG ------------------------------------------------------


def _smi(monkeypatch, *results):
    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: "nvidia-smi")
    seq = list(results)
    calls: list[list[str]] = []

    def run(args, timeout):
        calls.append(list(args))
        return seq.pop(0) if seq else None

    monkeypatch.setattr(hardware, "_run", run)
    return calls


def test_future_cc_is_used_with_a_note(monkeypatch):
    _smi(monkeypatch, (0, fx("nvsmi_future_cc13.csv")))
    gpus, _ = hardware.detect_gpus(windows_driver=True, comfy_windows=True)
    g = gpus[0]
    assert g.backend == "cuda" and g.compute_capability == "13.0"
    assert any("新しい世代" in n and "最新の PyTorch" in n for n in g.notes)


def test_older_than_table_is_still_refused():
    g = Gpu(
        name="NVIDIA GeForce GTX 780",
        vendor="nvidia",
        vram_total_gb=3,
        driver="474.44",
        backend="cuda",
        compute_capability="3.5",
    )
    assert classify_gpu(g, windows_driver=True, comfy_windows=True).backend == "none"


@pytest.mark.parametrize("raw", ["ppc64le", "riscv64", "armv7l", "i686"])
def test_unknown_machine_is_not_x86(monkeypatch, raw):
    monkeypatch.setattr(hardware.platform, "machine", lambda: raw)
    m = hardware._machine()
    assert m == raw
    g = Gpu(
        name="NVIDIA GeForce RTX 3070",
        vendor="nvidia",
        vram_total_gb=8,
        driver="580.65",
        backend="cuda",
        compute_capability="8.6",
    )
    classify_gpu(g, windows_driver=False, comfy_windows=False, machine=m)
    assert g.backend == "none" and any("この機械" in n for n in g.notes)


@pytest.mark.parametrize(("raw", "want"), [("AMD64", "x86_64"), ("x86_64", "x86_64"), ("ARM64", "aarch64")])
def test_known_machines(monkeypatch, raw, want):
    monkeypatch.setattr(hardware.platform, "machine", lambda: raw)
    assert hardware._machine() == want


def test_windows_arm64_has_no_cuda_build():
    key = hardware.torch_platform(True, "aarch64")
    assert key == "windows-aarch64" and hardware.torch_platform(False, "aarch64") == "aarch64"
    g = Gpu(
        name="NVIDIA GeForce RTX 4060",
        vendor="nvidia",
        vram_total_gb=8,
        driver="581.57",
        backend="cuda",
        compute_capability="8.9",
    )
    classify_gpu(g, windows_driver=True, comfy_windows=True, machine=key)
    assert g.backend == "none" and any("windows-aarch64" in n for n in g.notes)


def test_32bit_python_is_noted(monkeypatch):
    monkeypatch.setattr(hardware, "python_is_32bit", lambda: True)
    assert any("32 ビット" in n for n in detect(memory=MEM, gpus=[]).notes)


def test_mig_enabled_is_noted_but_value_unchanged():
    g = parse_nvidia_smi(fx("nvsmi_mig_enabled.csv"))[0]
    assert g.vram_total_gb == 80.0 and any("MIG" in n and "--vram" in n for n in g.notes)


def test_nvidia_smi_without_mig_field_retries_without_it(monkeypatch):
    calls = _smi(
        monkeypatch,
        (2, fx("nvsmi_no_mig_field.txt")),
        (0, "NVIDIA GeForce RTX 3070, 8192, 7000, 1192, 470.10, 8.6\n"),
    )
    gpus, _ = hardware.query_nvidia_smi("nvidia-smi")
    assert gpus[0].compute_capability == "8.6"
    assert "mig.mode.current" in calls[0][1] and "mig.mode.current" not in calls[1][1]
    assert "compute_cap" in calls[1][1]


def test_compute_cap_unknown_still_goes_to_the_old_query(monkeypatch):
    """compute_cap を知らない古い nvidia-smi は、mig を外した問い合わせを飛ばして QUERY_OLD へ。"""
    calls = _smi(
        monkeypatch,
        (2, 'Field "compute_cap" is not a valid field to query.\n'),
        (0, "NVIDIA GeForce GTX 1080, 8192, 7000, 1192, 472.12\n"),
    )
    gpus, _ = hardware.query_nvidia_smi("nvidia-smi")
    assert gpus[0].name == "NVIDIA GeForce GTX 1080" and len(calls) == 2


# --- 5.1 DGX Spark(GB10) ---------------------------------------------------------------------------


def _gb10(
    monkeypatch,
    tmp_path,
    smi: str = "nvsmi_gb10.csv",
    meminfo: str = "meminfo_gb10.txt",
    machine: str = "aarch64",
):
    _smi(monkeypatch, (0, fx(smi)))
    linux(monkeypatch, tree(tmp_path, {"/proc/meminfo": fx(meminfo)}))
    monkeypatch.setattr(hardware, "_machine", lambda: machine)
    gpus, _ = hardware.detect_gpus(windows_driver=False, comfy_windows=False)
    return gpus[0]


def test_gb10_is_used_as_unified_memory(monkeypatch, tmp_path):
    g = _gb10(monkeypatch, tmp_path)
    assert (g.backend, g.memory, g.carveout_gb) == ("cuda", "shared", 0.0)
    assert g.vram_total_gb == round(128452404 / 2**20, 2)  # MemTotal を割り引かない
    assert g.vram_free_gb == round((119876544 + 16777212) / 2**20, 2)  # MemAvailable + SwapFree
    assert g.compute_capability == "12.1" and g.arch == "blackwell"
    assert g.torch_cuda == ["cu132", "cu130", "cu129", "cu128"]  # sm_121 は sm_120 の機械語で動く
    assert any("GB10" in n and "good まで" in n for n in g.notes)
    assert any("sm_121" in n for n in g.notes)
    assert not any("判定に使わない" in n for n in g.notes)


def test_gb10_free_with_hugepages(monkeypatch, tmp_path):
    g = _gb10(monkeypatch, tmp_path, meminfo="meminfo_hugepages.txt")
    assert g.vram_free_gb == 80.0  # 空きの HugePages 40960 × 2 MiB、swap は数えない


def test_gb10_level_is_capped_at_good(monkeypatch, tmp_path):
    g = _gb10(monkeypatch, tmp_path)
    system = SystemInfo(
        os="Linux", cpu="-", cpu_cores=20, ram_total_gb=g.vram_total_gb, ram_available_gb=110.0, gpus=[g]
    )
    e = estimate(entry(6.0), system, coefficients=Coefficients())
    assert e.fit == "good" and e.run_mode == "gpu"
    assert any("good までにした" in r for r in e.reasons)


@pytest.mark.parametrize(
    ("smi", "machine", "why"),
    [
        ("nvsmi_gb10.csv", "x86_64", "aarch64"),
        ("nvsmi_jetson_orin.csv", "aarch64", "Jetson"),
    ],
)
def test_other_shared_memory_nvidia_is_not_used(monkeypatch, tmp_path, smi, machine, why):
    g = _gb10(monkeypatch, tmp_path, smi=smi, machine=machine)
    assert g.backend == "none" and g.memory == "dedicated"
    assert any(why in n for n in g.notes)


def test_gb10_name_must_match_exactly(monkeypatch, tmp_path):
    for name in ("NVIDIA GB20", "NVIDIA GB10 Engineering Sample", "nvgpu"):
        _smi(monkeypatch, (0, f"{name}, [N/A], [N/A], [N/A], 580.95.05, 12.1, [N/A]\n"))
        linux(monkeypatch, tree(tmp_path, {"/proc/meminfo": fx("meminfo_gb10.txt")}))
        monkeypatch.setattr(hardware, "_machine", lambda: "aarch64")
        g = hardware.detect_gpus(windows_driver=False, comfy_windows=False)[0][0]
        assert g.backend == "none", name


# --- 5. 画面の部品の一覧が読めない・sysfs・Flatpak・SELinux ---------------------------------------------


def test_adapter_list_unreadable_is_noted(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "_windows", lambda: True)
    assert any("PowerShell" in n for n in hardware.detect_other_gpus()[1])
    linux(monkeypatch, tmp_path)  # lspci なし・sysfs も空(Flatpak・Snap の閉じた環境と同じ)
    monkeypatch.setattr(hardware, "_sysfs_slots", REAL_SYSFS_SLOTS)  # /sys/bus/pci/devices が無い
    adapters, notes = hardware.detect_other_gpus()
    assert adapters == [] and any("画面の部品の一覧を読めなかった" in n for n in notes)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    assert any("system_profiler" in n for n in hardware.detect_other_gpus()[1])


SLOT = "0000:03:00.0"


SYSFS = {
    f"/sys/bus/pci/devices/{SLOT}/class": "0x030000\n",
    f"/sys/bus/pci/devices/{SLOT}/vendor": "0x1002\n",
    f"/sys/bus/pci/devices/{SLOT}/device": "0x744c\n",
    f"/sys/bus/pci/devices/{SLOT}/mem_info_vram_total": "25753026560\n",
    "/sys/bus/pci/devices/0000:00:1f.3/class": "0x040300\n",  # 音の部品は読まない
    "/sys/bus/pci/devices/0000:00:1f.3/vendor": "0x8086\n",
    "/sys/bus/pci/devices/0000:00:1f.3/device": "0x7a50\n",
}


def test_sysfs_is_read_when_lspci_is_missing(monkeypatch):
    """差し場所の名前(「:」を含む)は Windows のフォルダ名にできないので、ファイルの読みを表で差し替える。"""
    linux(monkeypatch)
    monkeypatch.setattr(hardware, "_read_text", lambda path: SYSFS.get(path.replace("\\", "/")))
    monkeypatch.setattr(hardware, "_sysfs_slots", lambda: [SLOT, "0000:00:1f.3", "../../etc"])
    adapters, notes = hardware.detect_other_gpus()
    assert [(a.name, a.vendor, a.pci_id, a.source, a.vram_gb) for a in adapters] == [
        ("AMD Device 744c", "amd", "1002:744c", "sysfs", 23.98)
    ]
    assert any("sysfs" in n for n in notes)
    g = classify_adapter(adapters[0], LINUX, 32.0)
    assert g is not None and (g.backend, g.gfx_target) == ("rocm", "gfx1100")
    assert any("lspci が無いので" in n for n in g.notes)
    assert SLOT not in json.dumps([dataclasses.asdict(a) for a in adapters]) + json.dumps(notes)


@pytest.mark.skipif(
    sys.platform == "win32", reason="差し場所の名前のフォルダは Windows に作れない(Linux・WSL で流す)"
)
def test_sysfs_slots_lists_only_pci_addresses(monkeypatch, tmp_path):
    root = tmp_path / "sys" / "bus" / "pci" / "devices"
    for name in (SLOT, "0000:00:1f.3", "not-a-slot"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(hardware, "_ROOT", str(tmp_path))
    assert REAL_SYSFS_SLOTS() == ["0000:00:1f.3", SLOT]


def test_unreadable_sysfs_files_are_none(monkeypatch, tmp_path):
    """SELinux などで読めないファイルは None(例外にしない)。"""
    linux(monkeypatch, tmp_path)

    def denied(self, *a, **k):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", denied)
    assert hardware._sysfs_text(SLOT, "class") is None
    assert hardware.read_meminfo() is None


# --- 6. 計算に使わない画面の部品・8. ドライバの無い NVIDIA・S6 のエラー コード ------------------------------


def test_virtual_and_bmc_displays_are_not_gpus():
    adapters = parse_lspci(fx("lspci_virtual.txt"))
    assert len(adapters) == 6
    for a in adapters:
        assert classify_adapter(a, LINUX, 32.0) is None
        assert "計算に使う GPU ではない" in a.reason, (a.name, a.reason)
        assert "pci.ids" not in a.reason
    wmi, _ = parse_wmi_video(fx("wmi_virtual_vmware.json"))
    assert classify_adapter(wmi[0], WIN11, 32.0) is None and "VMware" in wmi[0].reason


def test_device_id_lookup_is_only_for_amd_and_intel():
    assert hardware.pci_device_name("unknown", "Device 1111") is None
    assert hardware.pci_device_name("amd", "AMD Device 744c") == ("744c", "Navi 31")


def test_nvidia_without_driver_is_noted_whatever_the_name():
    adapters, _ = parse_wmi_video(fx("wmi_basic_display_ja.json"))
    assert adapters[0].vendor == "nvidia"
    notes = hardware._other_gpu_notes(adapters, judged_gpu=False)
    assert any("nvidia-smi で確かめられない" in n for n in notes)
    assert hardware._other_gpu_notes(adapters, judged_gpu=True) == []


def test_device_manager_error_codes():
    adapters, notes = parse_wmi_video(fx("wmi_error_codes.json"))
    joined = "\n".join(notes)
    assert "無効にしてある" in joined and "(コード 22)" in joined
    assert "ドライバが入っていない" in joined and "(コード 28)" in joined
    assert "Windows が止めた" in joined and "(コード 43)" in joined
    assert adapters[0].driver == "32.0.21025.1024" and adapters[1].driver is None


# --- 9・S7. コンテナのメモリ上限 ----------------------------------------------------------------------


def _cgroup(monkeypatch, tmp_path, files):
    linux(monkeypatch, tree(tmp_path, files))
    return hardware.cgroup_memory()


def test_cgroup_v2_walks_up_to_the_smallest_limit(monkeypatch, tmp_path):
    got = _cgroup(
        monkeypatch,
        tmp_path,
        {
            "/proc/self/cgroup": "0::/user.slice/app.scope\n",
            "/sys/fs/cgroup/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/memory.max": f"{8 * GB}\n",
            "/sys/fs/cgroup/user.slice/app.scope/memory.max": "max\n",
            "/sys/fs/cgroup/user.slice/app.scope/memory.current": f"{2 * GB}\n",
        },
    )
    assert got == (8 * GB, 2 * GB)
    monkeypatch.setattr(
        psutil, "virtual_memory", lambda: type("VM", (), {"total": 32 * GB, "available": 30 * GB})()
    )
    m = hardware.read_memory()
    assert (m.total, m.available) == (8 * GB, 6 * GB)  # 空き = 上限 - memory.current


def test_cgroup_v1(monkeypatch, tmp_path):
    got = _cgroup(
        monkeypatch,
        tmp_path,
        {
            "/proc/self/cgroup": "12:cpu,cpuacct:/docker/abc\n4:memory:/docker/abc\n",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes": "9223372036854771712\n",
            "/sys/fs/cgroup/memory/docker/abc/memory.limit_in_bytes": f"{4 * GB}\n",
            "/sys/fs/cgroup/memory/docker/abc/memory.usage_in_bytes": f"{1 * GB}\n",
        },
    )
    assert got == (4 * GB, 1 * GB)


def test_cgroup_namespace_root_and_no_limit(monkeypatch, tmp_path):
    assert _cgroup(
        monkeypatch, tmp_path, {"/proc/self/cgroup": "0::/\n", "/sys/fs/cgroup/memory.max": f"{2 * GB}\n"}
    ) == (2 * GB, None)
    other = tmp_path / "b"
    assert _cgroup(
        monkeypatch, other, {"/proc/self/cgroup": "0::/a\n", "/sys/fs/cgroup/memory.max": "max\n"}
    ) == (None, None)
    third = tmp_path / "c"
    assert _cgroup(monkeypatch, third, {"/proc/self/cgroup": "0::/../../etc\n"}) == (None, None)


# --- 10・S8. amdkfd の世代 -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("v", "gfx"),
    [
        (110000, "gfx1100"),
        (120001, "gfx1201"),
        (110003, "gfx1103"),
        (90010, "gfx90a"),
        (90402, "gfx942"),
        (100300, "gfx1030"),
        (0, None),
    ],
)
def test_gfx_from_target_version(v, gfx):
    assert hardware.gfx_from_target_version(v) == gfx


def _kfd(tmp_path, nodes: dict[int, str]) -> dict[str, str]:
    return {f"/sys/class/kfd/kfd/topology/nodes/{n}/properties": text for n, text in nodes.items()}


def test_kfd_gfx_is_used_before_the_name(monkeypatch, tmp_path):
    files = _kfd(
        tmp_path,
        {
            0: "cpu_cores_count 16\ngfx_target_version 0\nvendor_id 0\ndevice_id 0\n",
            1: "simd_count 192\ngfx_target_version 130000\nvendor_id 4098\ndevice_id 30208\n",  # 0x7600
        },
    )
    linux(monkeypatch, tree(tmp_path, files))
    monkeypatch.setattr(hardware, "kfd_gfx_targets", REAL_KFD)
    assert hardware.kfd_gfx_targets() == {"1002:7600": "gfx1300"}
    lspci = (
        "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI]"
        " Device [1002:7600]\n"
    )
    monkeypatch.setattr(hardware, "_run", Router({"lspci": (0, lspci)}))
    monkeypatch.setattr(hardware, "_sysfs_vram", lambda slot: 17163091968)
    adapters, _ = hardware.detect_other_gpus()
    assert adapters[0].gfx_target == "gfx1300"
    g = classify_adapter(adapters[0], LINUX, 32.0)
    assert g is not None and g.backend == "rocm" and g.gfx_target == "gfx1300"
    assert any("新しい世代" in n for n in g.notes) and any("amdkfd" in n for n in g.notes)


# --- 11. Windows の表に無い新しい AMD・Intel ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "pnp"),
    [
        ("AMD Radeon RX 10070 XT", "PCI\\VEN_1002&DEV_7600&SUBSYS_0E3B1002&REV_C0\\6&1"),
        ("Intel(R) Arc(TM) C770 Graphics", "PCI\\VEN_8086&DEV_E2FF&SUBSYS_0&REV_00\\6&1"),
    ],
)
def test_new_windows_gpu_reason_suggests_vram(name, pnp):
    data = [{"Name": name, "AdapterRAM": 4293918720, "PNPDeviceID": pnp, "ConfigManagerErrorCode": 0}]
    a = parse_wmi_video(json.dumps(data))[0][0]
    assert classify_adapter(a, WIN11, 32.0) is None
    assert "名前の表に無い" in a.reason and "--vram" in a.reason


# --- 5.2 WSL2 の AMD・Intel ----------------------------------------------------------------------------

PS_PATH = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"


def _wsl(monkeypatch, tmp_path, *, interop=True, container=False, rocm=True, files=None, table=None):
    t = {"/dev/dxg": "", PS_PATH: ""}
    if interop:
        t["/proc/sys/fs/binfmt_misc/WSLInterop"] = "enabled\n"
    if container:
        t["/.dockerenv"] = ""
    if rocm:
        t["/opt/rocm/bin/rocminfo"] = ""
    t.update(files or {})
    linux(monkeypatch, tree(tmp_path, t))
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setattr(hardware, "wsl_state", REAL_WSL_STATE)
    monkeypatch.setattr(hardware.shutil, "which", lambda name, *a, **k: None)
    router = Router(table or {})
    monkeypatch.setattr(hardware, "_run", router)
    return router


def test_wsl_state_needs_dxg_and_microsoft_kernel(monkeypatch, tmp_path):
    _wsl(monkeypatch, tmp_path)
    assert hardware.wsl_state() == hardware.WslState(interop=True, container=False)
    monkeypatch.setattr(hardware.platform, "release", lambda: "6.8.0-45-generic")
    assert hardware.wsl_state() is None


def test_wsl_amd_ready_is_used(monkeypatch, tmp_path):
    router = _wsl(
        monkeypatch,
        tmp_path,
        table={
            "powershell.exe": (0, fx("wsl_ps_rx7900xtx.json")),
            "rocminfo": (0, fx("rocminfo_wsl_7900xtx.txt")),
        },
    )
    adapters, _ = hardware.detect_other_gpus()
    assert [(a.name, a.source, a.gfx_target, a.wsl_runtime, a.vram_gb) for a in adapters] == [
        ("AMD Radeon RX 7900 XTX", "wsl", "gfx1100", True, 23.98)
    ]
    ps = next(c for c in router.calls if "powershell" in c[0])
    assert ps[0].endswith(PS_PATH) and ps[1:3] == ["-NoProfile", "-NonInteractive"]
    g = classify_adapter(adapters[0], LINUX, 32.0)
    assert g is not None and (g.backend, g.gfx_target, g.memory, g.vram_total_gb) == (
        "rocm",
        "gfx1100",
        "dedicated",
        23.98,
    )
    assert any("約 3 GB" in n for n in g.notes)


def test_wsl_uuid_never_leaks(monkeypatch, tmp_path, caplog):
    _wsl(
        monkeypatch,
        tmp_path,
        table={
            "powershell.exe": (0, fx("wsl_ps_rx7900xtx.json")),
            "rocminfo": (0, fx("rocminfo_wsl_7900xtx.txt")),
        },
    )
    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: None)
    assert UUID in fx("rocminfo_wsl_7900xtx.txt")
    with caplog.at_level(logging.DEBUG):
        s = detect(memory=MEM, host=LINUX)
    assert s.best_gpu is not None and s.best_gpu.backend == "rocm"
    out = json.dumps(s.to_dict(), ensure_ascii=False) + caplog.text
    assert UUID not in out and "Uuid" not in out


@pytest.mark.parametrize(
    ("kw", "table", "why"),
    [
        ({"interop": False}, {}, "相互運用"),
        ({"container": True}, {}, "コンテナ"),
        ({}, {"powershell.exe": None}, "powershell.exe"),
    ],
)
def test_wsl_windows_side_unreadable(monkeypatch, tmp_path, kw, table, why):
    router = _wsl(monkeypatch, tmp_path, table=table, **kw)
    adapters, notes = hardware.detect_other_gpus()
    assert notes
    assert adapters == [] and any(why in n and "Windows 側の GPU" in n for n in notes), notes
    if kw:
        assert router.calls == [["lspci", "-Dnn"]]  # Windows のプログラムは起こさない


def test_wsl_amd_without_runtime(monkeypatch, tmp_path):
    _wsl(monkeypatch, tmp_path, rocm=False, table={"powershell.exe": (0, fx("wsl_ps_rx7900xtx.json"))})
    a = hardware.detect_other_gpus()[0][0]
    assert a.wsl_runtime is False and classify_adapter(a, LINUX, 32.0) is None
    assert "Adrenalin" in a.reason and "ROCm" in a.reason


def test_wsl_amd_not_in_table(monkeypatch, tmp_path):
    _wsl(
        monkeypatch,
        tmp_path,
        table={
            "powershell.exe": (0, fx("wsl_ps_rx6800.json")),
            "rocminfo": (0, fx("rocminfo_wsl_7900xtx.txt")),
        },
    )
    a = hardware.detect_other_gpus()[0][0]
    assert classify_adapter(a, LINUX, 32.0) is None and "WSL の対応表" in a.reason


def _arc(driver: str) -> str:
    d = json.loads(fx("wsl_ps_arc_b580.json"))
    d["video"][0]["DriverVersion"] = driver
    return json.dumps(d)


@pytest.mark.parametrize(
    ("driver", "clinfo", "ok", "why"),
    [
        ("32.0.101.8801", "clinfo_l_arc.txt", True, ""),
        ("32.0.101.6913", "clinfo_l_arc.txt", False, "32.0.101.8801"),
        ("32.0.101.8801", "clinfo_l_cpu_only.txt", False, "compute runtime"),
    ],
)
def test_wsl_intel(monkeypatch, tmp_path, driver, clinfo, ok, why):
    _wsl(
        monkeypatch,
        tmp_path,
        files={"/opt/rocm/bin/clinfo": ""},
        table={"powershell.exe": (0, _arc(driver)), "clinfo": (0, fx(clinfo))},
    )
    monkeypatch.setattr(hardware, "_find_tool", lambda name, extra=(): name)
    a = hardware.detect_other_gpus()[0][0]
    g = classify_adapter(a, LINUX, 32.0)
    assert (g is not None) is ok, a.reason
    if ok:
        assert g.backend == "xpu" and g.vram_total_gb == 12.0
    else:
        assert why in a.reason


def test_intel_driver_build():
    assert hardware.intel_driver_build("32.0.101.8801") == 1018801
    assert hardware.intel_driver_build("30.0.101.2031") == 1012031  # Intel support 000005654 の例
    assert hardware.intel_driver_build("x") is None


def test_wsl_with_usable_nvidia_does_not_start_windows_programs(monkeypatch, tmp_path):
    router = _wsl(monkeypatch, tmp_path, table={"powershell.exe": (0, fx("wsl_ps_rx7900xtx.json"))})
    g = Gpu(
        name="NVIDIA GeForce RTX 3070",
        vendor="nvidia",
        vram_total_gb=8,
        driver="581.57",
        backend="cuda",
        compute_capability="8.6",
    )
    s = detect(memory=MEM, gpus=[g], other_gpus=True, host=LINUX)
    assert s.best_gpu.name == "NVIDIA GeForce RTX 3070"
    assert not any("powershell" in c[0] for c in router.calls)


def test_wsl_basic_render_driver_is_dropped(monkeypatch, tmp_path):
    lspci = "0000:00:00.0 3D controller [0302]: Microsoft Corporation Basic Render Driver [1414:008e]\n"
    _wsl(monkeypatch, tmp_path, interop=False, table={"lspci": (0, lspci)})
    adapters, _ = hardware.detect_other_gpus()
    assert adapters == []


def test_rocminfo_reads_only_three_fields():
    agents = hardware.parse_rocminfo(fx("rocminfo_wsl_7900xtx.txt"))
    assert agents == [("gfx1100", "AMD Radeon RX 7900 XTX")]


def test_clinfo_lists_only_intel_gpus():
    assert hardware.parse_clinfo_devices(fx("clinfo_l_arc.txt")) == ["Intel(R) Arc(TM) B580 Graphics"]
    assert hardware.parse_clinfo_devices(fx("clinfo_l_cpu_only.txt")) == []


# --- 4. macOS の版が 10.16 -----------------------------------------------------------------------------


def test_macos_10_16_is_read_again(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(hardware.platform, "mac_ver", lambda: ("10.16", ("", "", ""), "arm64"))
    monkeypatch.setattr(hardware, "_cpu_name", lambda: "Apple M1")
    router = Router({"sw_vers": (0, "15.1\n")})
    monkeypatch.setattr(hardware, "_run", router)
    h = hardware.host_info()
    assert h.mac_version == (15, 1)
    assert "SYSTEM_VERSION_COMPAT=0" in router.calls[0]
    a = parse_system_profiler(
        json.dumps(
            {"SPDisplaysDataType": [{"sppci_model": "Apple M1", "spdisplays_vendor": "sppci_vendor_Apple"}]}
        )
    )[0]
    assert classify_adapter(a, h, 16.0).backend == "mps"
    monkeypatch.setattr(hardware, "_run", Router({}))  # 読み直せなければ版は分からない(使わない側)
    assert hardware.host_info().mac_version is None


# --- 複数の会社(NVIDIA が先) ------------------------------------------------------------------------


def test_nvidia_first_with_other_vendors(monkeypatch):
    lspci = (
        "0000:01:00.0 VGA compatible controller [0300]: NVIDIA Corporation GA104 [GeForce RTX 3070]"
        " [10de:2484]\n"
        "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 "
        "[Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c] (rev c8)\n"
    )
    monkeypatch.setattr(
        hardware, "detect_other_gpus", lambda **_: (parse_lspci(lspci, lambda s: 24 * GB), [])
    )
    g = Gpu(
        name="NVIDIA GeForce RTX 3070",
        vendor="nvidia",
        vram_total_gb=8,
        driver="580.65",
        backend="cuda",
        compute_capability="8.6",
    )
    s = detect(memory=MEM, gpus=[g], other_gpus=True, host=LINUX)
    assert s.best_gpu.name == "NVIDIA GeForce RTX 3070" and [x.backend for x in s.gpus] == ["cuda"]


# --- S2. 時間切れのコマンドを組ごと止める ------------------------------------------------------------------


def test_run_timeout_does_not_hang_on_grandchildren():
    """孫のプロセスが出力の口を持ったまま残っても、時間切れから数秒で返る(POSIX は組ごと止める。
    Windows は子だけ止め、さらに 1 秒待って口を閉じて返す: docs/hardware.md「既知の限界」)。"""
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(6)'])\n"
        "time.sleep(6)\n"
    )
    t0 = time.monotonic()
    assert REAL_RUN([sys.executable, "-c", code], 0.5) is None
    assert time.monotonic() - t0 < 4.0


def test_run_returns_output():
    assert REAL_RUN([sys.executable, "-c", "print('ok')"], 30.0) == (
        0,
        "ok" + ("\r\n" if sys.platform == "win32" else "\n"),
    )


def test_gb_constant():
    assert GIB == GB
