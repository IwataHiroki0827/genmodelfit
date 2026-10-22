"""NVIDIA 以外の GPU の使用率・使用中のメモリを読む道具(gpustat.py)の試験。

この PC(RTX 5080)では AMD・Intel・Apple の道具を動かせないので、出力は公式の文書・ソースの例で試す。
本物の道具は呼ばない(conftest が hardware._run を塞ぐ)。Windows の性能カウンタだけは、この PC の実際の
typeperf の出力(PC 名・プロセス ID・LUID・時刻を置き換えた)を使う(GPU Engine の数え方は会社によらない)。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genmodelfit import gpustat, hardware
from genmodelfit.gpustat import (
    Probe,
    Reading,
    parse_amd_smi,
    parse_ioreg,
    parse_rocm_smi,
    parse_typeperf,
    parse_xpu_smi_dump,
    pick,
    tools_for,
)

FIX = Path(__file__).parent / "fixtures" / "hardware"

# amd-smi(ROCm 7 の形)。値は ROCm 7.1 の changelog の文字の例(TOTAL_VRAM 196592 MB・USED_VRAM 283 MB)、
# 形は ROCm/amdsmi amd-staging の amdsmi_cli/amdsmi_commands.py(metric の usage・mem_usage)による
AMD_SMI_7 = json.dumps(
    {
        "gpu_data": [
            {
                "gpu": 0,
                "usage": {
                    "gfx_activity": {"value": 12, "unit": "%"},
                    "umc_activity": {"value": 0, "unit": "%"},
                    "mm_activity": "N/A",
                    "vcn_activity": [{"value": 0, "unit": "%"}],
                },
                "mem_usage": {
                    "total_vram": {"value": 196592, "unit": "MB"},
                    "used_vram": {"value": 283, "unit": "MB"},
                    "free_vram": {"value": 196309, "unit": "MB"},
                },
            }
        ]
    }
)
# ROCm 6.x は配列で返す(同じソースの 6.1〜6.4 の版)
AMD_SMI_6 = json.dumps(
    [
        {"gpu": 0, "usage": {"gfx_activity": {"value": 3, "unit": "%"}}, "mem_usage": {"used_vram": 512}},
        {"gpu": 1, "usage": {"gfx_activity": "N/A"}, "mem_usage": "N/A"},
    ]
)
# rocm-smi --showuse --showmeminfo vram --json(項目名は ROCm/rocm_smi_lib の python_smi_tools/rocm_smi.py の
# showGpuUse「GPU use (%)」・showMemInfo「VRAM Total Memory (B)」「VRAM Total Used Memory (B)」。値は文字列)
ROCM_SMI = json.dumps(
    {
        "card0": {
            "GPU use (%)": "45",
            "VRAM Total Memory (B)": "25753026560",
            "VRAM Total Used Memory (B)": "1073741824",
        },
        "card1": {
            "GPU use (%)": "0",
            "VRAM Total Memory (B)": "536870912",
            "VRAM Total Used Memory (B)": "0",
        },
        "system": {"Driver version": "6.8.5"},
    }
)
# xpu-smi dump(intel/xpumanager doc/smi_user_guide.md の dump の CSV の形。項目 0 と 18 の見出しも同じ文書)
XPU_SMI = (
    "Timestamp, DeviceId, Average % utilization of all GPU Engines, GPU Memory Used (MiB)\n"
    "06:14:46.000,    0, 41.50,  2048\n"
)
# ioreg -r -d 1 -w 0 -c IOAccelerator の PerformanceStatistics
# (https://www.heatware.net/linux/how-to-monitor-gpu/ の例)
IOREG = (
    "+-o AGXAcceleratorG14X  <class AGXAcceleratorG14X, id 0x100000000, registered, matched, active>\n"
    "    {\n"
    '      "PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=6117801984,'
    '"Tiler Utilization %"=82,"recoveryCount"=0,"lastRecoveryTime"=0,"Renderer Utilization %"=82,'
    '"TiledSceneBytes"=1638400,"Device Utilization %"=82,"SplitSceneCount"=0,'
    '"Allocated PB Size"=184287232,"In use system memory"=2968010752}\n'
    "    }\n"
)


def test_amd_smi_rocm7_and_rocm6_shapes():
    assert parse_amd_smi(AMD_SMI_7) == [Reading(12.0, 283.0, 196592.0)]
    assert parse_amd_smi(AMD_SMI_6) == [Reading(3.0, 512.0, None)]  # "N/A" だけの GPU は捨てる


def test_rocm_smi_reads_bytes_as_mib():
    r = parse_rocm_smi(ROCM_SMI)
    assert r[0] == Reading(45.0, 1024.0, 25753026560 / 2**20) and r[1].util_pct == 0.0


def test_xpu_smi_dump_finds_columns_by_header():
    assert parse_xpu_smi_dump(XPU_SMI) == [Reading(41.5, 2048.0)]
    # 項目の順が違っても見出しで探す
    swapped = (
        "Timestamp, DeviceId, GPU Memory Used (MiB), Average % utilization of all GPU Engines\n1, 0, 10, 5\n"
    )
    assert parse_xpu_smi_dump(swapped) == [Reading(5.0, 10.0)]


def test_typeperf_real_output_takes_busiest_engine_and_dedicated_memory():
    text = (FIX / "typeperf_rtx5080.csv").read_bytes().decode("utf-8")
    got = parse_typeperf(text)
    # 実物は使っていない仮想のアダプタも出すが、使用率も使用中のメモリも 0 なので除く
    assert len(got) == 1
    assert 2.0 < got[0].util_pct < 3.0 and 2300 < got[0].used_mib < 2330  # 同じ時の nvidia-smi は 2309 MiB


def test_typeperf_sums_processes_per_engine_and_clamps():
    head = (
        '"(PDH-CSV 4.0)",'
        '"\\\\HOST\\GPU Engine(pid_1_luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3D)'
        '\\Utilization Percentage",'
        '"\\\\HOST\\GPU Engine(pid_2_luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3D)'
        '\\Utilization Percentage",'
        '"\\\\HOST\\GPU Engine(pid_2_luid_0x00000000_0x00000001_phys_0_eng_1_engtype_Compute_0)'
        '\\Utilization Percentage",'
        '"\\\\HOST\\GPU Adapter Memory(luid_0x00000000_0x00000001_phys_0)\\Dedicated Usage"\n'
    )
    assert parse_typeperf(head + '"t","20","15","30","1073741824"\n') == [Reading(35.0, 1024.0)]
    assert parse_typeperf(head + '"t","80","70","0","0"\n')[0].util_pct == 100.0  # 100 で切る


def test_ioreg_device_utilization_and_fallback_key():
    assert parse_ioreg(IOREG) == [Reading(82.0, 2968010752 / 2**20)]
    old = '"PerformanceStatistics" = {"GPU Activity(%)"=7,"In use system memory"=1048576}'
    assert parse_ioreg(old) == [Reading(7.0, 1.0)]


@pytest.mark.parametrize(
    "parse", [parse_amd_smi, parse_rocm_smi, parse_xpu_smi_dump, parse_typeperf, parse_ioreg]
)
@pytest.mark.parametrize(
    "text",
    [
        "",
        "not json at all",
        "{",
        "[1, 2, 3]",
        '{"gpu_data": "N/A"}',
        '{"card0": "x"}',
        "Timestamp, DeviceId\n",
        '"(PDH-CSV 4.0)","x"\n',
        "ERROR: No AMD GPUs specified",
        "\x00\xff garbage",
    ],
)
def test_broken_output_gives_nothing(parse, text):
    assert parse(text) == []


def test_tools_per_vendor_and_os():
    assert [t.name for t in tools_for("amd", "Linux")] == ["amd-smi", "rocm-smi"]
    assert tools_for("intel", "Linux") == [gpustat.XPU_SMI]
    assert tools_for("amd", "Windows") == [gpustat.TYPEPERF] == tools_for("intel", "Windows")
    assert tools_for("apple", "Darwin") == [gpustat.IOREG]
    assert tools_for("amd", "Darwin") == [] and tools_for("apple", "Linux") == []


def test_pick_matches_comfyui_gpu_by_total():
    a, b = Reading(5.0, 100.0, 24576.0), Reading(90.0, 400.0, 512.0)
    assert pick([a, b], 24000.0) == (a, False)
    # 選べなければ、使用率は最大・メモリは合計(ほかの GPU の仕事でも断る側)
    assert pick([a, b], None) == (Reading(90.0, 500.0), True)
    assert pick([], 1.0) == (None, False)
    # 総量の無い道具(性能カウンタ)は、メモリを使っている 1 枚
    c, d = Reading(1.0, 2000.0), Reading(0.0, 0.0)
    assert pick([c, d], 8000.0) == (c, False)


def test_tool_missing_failing_or_timing_out_reads_nothing(monkeypatch):
    calls = []

    def run(args, timeout):
        calls.append((args[0], timeout))
        return None  # 見つからない・時間切れ(hardware._run は時間切れも None)

    monkeypatch.setattr(hardware, "_run", run)
    p = Probe("amd", system="Linux")
    assert p.read() is None and p.tool is None
    assert calls == [("amd-smi", gpustat.TOOL_TIMEOUT), ("rocm-smi", gpustat.TOOL_TIMEOUT)]
    monkeypatch.setattr(hardware, "_run", lambda args, timeout: (2, AMD_SMI_7))  # 終了コードが 0 でない
    assert Probe("amd", system="Linux").read() is None


def test_probe_falls_back_and_remembers_tool(monkeypatch):
    seen = []

    def run(args, timeout):
        seen.append(args[0])
        return (0, "garbage") if args[0] == "amd-smi" else (0, ROCM_SMI)

    monkeypatch.setattr(hardware, "_run", run)
    p = Probe("amd", total_mib=24560.0, system="Linux")
    assert p.read() == Reading(45.0, 1024.0, 25753026560 / 2**20) and p.tool is gpustat.ROCM_SMI
    assert p.used_mib() == 1024.0
    assert seen == ["amd-smi", "rocm-smi", "rocm-smi"]  # 2 回目からは読めた道具だけ


def test_probe_has_no_tool_for_unknown_combination():
    p = Probe("amd", system="Darwin")
    assert p.read() is None and p.tool_names == "なし"
