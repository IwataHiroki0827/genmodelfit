"""NVIDIA 以外の GPU(#22)・CPU で動く系統(#39)・同じ GPU で成功した実測の段階(#26)。

本物の PowerShell・lspci・system_profiler は呼ばない(conftest が塞ぐ)。出力は tests/fixtures/hardware の
フィクスチャ(個体の ID は 0 に置き換えた)で差し替え、OS・CPU は hardware.Host で決める。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import genmodelfit as gmf
from genmodelfit import hardware
from genmodelfit.estimate import Coefficients, estimate
from genmodelfit.hardware import (
    Host,
    Memory,
    classify_adapter,
    classify_gpu,
    detect,
    parse_lspci,
    parse_system_profiler,
    parse_wmi_video,
)
from genmodelfit.models import Adapter, Gpu, ModelEntry, ModelFile, SystemInfo
from genmodelfit.recommend import judge, select
from genmodelfit.table import make_console, mode_cell, no_gpu_line, pc_line, print_fit_table

FIX = Path(__file__).parent / "fixtures" / "hardware"
GB = 2**30
ORDER = ["too_tight", "marginal", "good", "perfect"]

WIN11 = Host("Windows", 26100, cpu="13th Gen Intel(R) Core(TM) i7-13700KF")
WIN10 = Host("Windows", 19045)
LINUX = Host("Linux", cpu="AMD Ryzen 9 7950X 16-Core Processor")
MAC = Host("Darwin", mac_version=(15, 1), cpu="Apple M2 Pro")


def fx(name: str) -> str:
    return (FIX / name).read_bytes().decode("utf-8")


def reg(name: str) -> list[dict]:
    return json.loads(fx(name))


@pytest.fixture(scope="module")
def db():
    return gmf.load_db()


def _no_nvidia(monkeypatch, adapters: list[Adapter]):
    monkeypatch.setattr(hardware, "find_nvidia_smi", lambda: None)
    monkeypatch.setattr(hardware, "detect_other_gpus", lambda: (adapters, []))


def system_of(monkeypatch, adapters, host: Host, ram_gb: float = 32.0, **kw) -> SystemInfo:
    _no_nvidia(monkeypatch, adapters)
    return detect(memory=Memory(total=int(ram_gb * GB), available=int(ram_gb * GB)), host=host, **kw)


# --- 検出と対応表 ---------------------------------------------------------------------------------


def rx7900xtx_linux(monkeypatch, ram_gb=64.0):
    adapters = parse_lspci(fx("lspci_rx7900xtx.txt"), lambda slot: 24 * GB)
    return system_of(monkeypatch, adapters, LINUX, ram_gb)


def test_rx7900xtx_linux_rocm(monkeypatch):
    s = rx7900xtx_linux(monkeypatch)
    g = s.best_gpu
    assert (g.vendor, g.backend, g.gfx_target, g.arch, g.memory, g.vram_total_gb) == (
        "amd",
        "rocm",
        "gfx1100",
        "RDNA 3",
        "dedicated",
        24.0,
    )
    assert g.bf16 is True and g.fp8 is False and g.compute_dtype == "fp16"
    assert any("推定(AMD・実測なし)" in n for n in s.notes)
    # gfx1100 は pytorch.org の ROCm 版に入っているので、TheRock の注意は出さない
    assert not any("TheRock" in n for n in s.notes)
    assert pc_line(s).startswith(
        "PC: AMD Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M 24.0 GiB VRAM (ROCm, estimated)"
    )


def test_rx6600_windows_rdna2(monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_rx6600.json"), reg("registry_rx6600.json"))
    s = system_of(monkeypatch, adapters, WIN11)
    g = s.best_gpu
    assert (g.backend, g.gfx_target, g.arch, g.bf16, round(g.vram_total_gb)) == (
        "rocm",
        "gfx1032",
        "RDNA 2",
        False,
        8,
    )
    # RDNA 2 は bf16 を持たないが、ComfyUI は fp16 を持たない系統も bf16 で計算する(作業用メモリは倍にしない)
    e = ModelEntry(
        id="t",
        name="t",
        modality="image",
        precision="bf16",
        files=[
            ModelFile(name="qwen_image_bf16.safetensors", size_bytes=2 * GB, directory="diffusion_models")
        ],
    )
    est = estimate(e, s, coefficients=Coefficients())
    assert any("bf16 で動かす" in r and "RDNA 2" in r for r in est.reasons)
    # Windows 10 は対象外(ComfyUI の案内は Windows 11)
    adapters, _ = parse_wmi_video(fx("wmi_rx6600.json"), reg("registry_rx6600.json"))
    s10 = system_of(monkeypatch, adapters, WIN10)
    assert s10.best_gpu is None and any("Windows 11 が要る" in n for n in s10.notes)


def test_arc_a770_windows_xpu(monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_arc_a770.json"), reg("registry_arc_a770.json"))
    s = system_of(monkeypatch, adapters, WIN11)
    g = s.best_gpu
    assert (g.vendor, g.backend, g.memory, round(g.vram_total_gb, 1)) == ("intel", "xpu", "dedicated", 15.9)
    assert pc_line(s).startswith("PC: Intel(R) Arc(TM) A770 Graphics 15.9 GiB VRAM (XPU, estimated)")
    # Linux の lspci では VRAM が読めないので、名前から少ないほうの 8 GB とみなす
    a = Adapter(name="Intel Arc A770", vendor="intel", source="lspci")
    g2 = classify_adapter(a, LINUX, 32.0)
    assert g2.vram_total_gb == 8.0 and a.vram_source == "name"


def test_core_ultra_155h_igpu_shares_half_of_ram(monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_core_ultra_155h.json"), [])
    s = system_of(monkeypatch, adapters, WIN11, ram_gb=32.0)
    g = s.best_gpu
    assert (g.backend, g.memory) == ("xpu", "shared")
    assert g.vram_total_gb == pytest.approx(0.12 + 16.0, abs=0.01)  # 取り分け 128 MB + RAM の半分
    assert "(integrated; 16.1 GiB shared with RAM, XPU, estimated)" in pc_line(s)


def test_uhd770_is_cpu_with_a_reason(monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_uhd770.json"), [])
    s = system_of(monkeypatch, adapters, WIN11, ram_gb=15.7)
    assert s.best_gpu is None and s.vram_gb == 0
    a = s.other_gpus[0]
    assert a.usable is False and a.integrated is True and "XPU 版" in a.reason
    line = pc_line(s)
    assert line.startswith(
        "PC: Intel(R) UHD Graphics 770 (integrated; not usable for generation → judged as CPU), RAM 15.7 GiB"
    )


def test_m2_pro_unified_memory(monkeypatch):
    adapters = parse_system_profiler(fx("system_profiler_m2pro.json"))
    assert [(a.name, a.vendor) for a in adapters] == [("Apple M2 Pro", "apple")]
    s = system_of(monkeypatch, adapters, MAC, ram_gb=32.0)
    g = s.best_gpu
    assert (g.backend, g.memory, g.vram_total_gb) == ("mps", "unified", 24.0)
    assert pc_line(s).startswith("PC: Apple M2 Pro (unified memory; 24.0 GiB for the GPU, MPS, estimated)")
    # macOS 13 は MPS 版 PyTorch の対象外
    adapters = parse_system_profiler(fx("system_profiler_m2pro.json"))
    s13 = system_of(monkeypatch, adapters, Host("Darwin", mac_version=(13, 6)), ram_gb=32.0)
    assert s13.best_gpu is None and any("macOS 14.0 以降" in n for n in s13.notes)


def test_ryzen_ai_max_395_apu(monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_ryzen_ai_max_395.json"), reg("registry_ryzen_ai_max_395.json"))
    host = Host("Windows", 26100, cpu="AMD RYZEN AI MAX+ 395 w/ Radeon 8060S")
    s = system_of(monkeypatch, adapters, host, ram_gb=64.0)
    g = s.best_gpu
    assert (g.backend, g.gfx_target, g.memory) == ("rocm", "gfx1151", "shared")
    assert g.vram_total_gb == pytest.approx(64.0 + 32.0)  # BIOS で取り分けた 64 GB + 見えている RAM の半分


def test_generic_radeon_graphics_resolved_by_cpu():
    a = Adapter(name="AMD Radeon(TM) Graphics", vendor="amd", vram_gb=0.5)
    g = classify_adapter(a, Host("Windows", 26100, cpu="AMD Ryzen 7 8845HS w/ Radeon 780M Graphics"), 32.0)
    assert g is not None and g.gfx_target == "gfx1103" and g.memory == "shared"
    b = Adapter(name="AMD Radeon(TM) Graphics", vendor="amd", vram_gb=0.5)
    assert classify_adapter(b, Host("Windows", 26100, cpu="AMD Ryzen 7 5800X 8-Core Processor"), 32.0) is None
    assert b.integrated is True and "世代(gfx)が名前から分からない" in b.reason


def test_rdna1_and_raphael_are_cpu():
    a = Adapter(name="AMD Radeon RX 5700 XT", vendor="amd", vram_gb=8.0)
    assert classify_adapter(a, LINUX, 32.0) is None and "RDNA 2 以降" in a.reason
    r = Adapter(name="AMD Raphael", vendor="amd")
    assert classify_adapter(r, LINUX, 32.0) is None and r.integrated is True


def test_rdna2_not_in_pytorch_org_mentions_therock():
    a = Adapter(name="AMD Radeon RX 6600", vendor="amd", vram_gb=8.0)
    g = classify_adapter(a, LINUX, 32.0)
    assert g.gfx_target == "gfx1032" and any("TheRock" in n for n in g.notes)


def test_nvidia_present_keeps_nvidia(monkeypatch):
    """NVIDIA の判定は変えない: NVIDIA の GPU があれば AMD・Intel は判定に使わない。"""
    g = classify_gpu(
        Gpu(
            name="NVIDIA GeForce RTX 3070",
            vendor="nvidia",
            vram_total_gb=8.0,
            driver="566.03",
            backend="cuda",
            compute_capability="8.6",
        ),
        windows_driver=True,
        comfy_windows=True,
    )
    adapters, _ = parse_wmi_video(fx("wmi_arc_a770.json"), reg("registry_arc_a770.json"))
    monkeypatch.setattr(hardware, "detect_other_gpus", lambda: (adapters, []))
    s = detect(gpus=[g], memory=Memory(total=32 * GB, available=32 * GB), other_gpus=True, host=WIN11)
    assert s.best_gpu.name == "NVIDIA GeForce RTX 3070" and [x.backend for x in s.gpus] == ["cuda"]
    assert any("判定に使わない" in n for n in s.notes)


# --- 判定(推定・good まで・共有メモリ) ---------------------------------------------------------------


def small_image(main_gb: float = 2.0) -> ModelEntry:
    return ModelEntry(
        id="t",
        name="t",
        modality="image",
        precision="fp16",
        files=[
            ModelFile(
                name="flux1-dev_fp16.safetensors", size_bytes=int(main_gb * GB), directory="diffusion_models"
            ),
            ModelFile(name="ae.safetensors", size_bytes=int(0.3 * GB), directory="vae"),
        ],
    )


def vendor_system(kind: str, vram: float, ram: float = 32.0) -> SystemInfo:
    if kind == "amd":
        g = Gpu(
            name="AMD Radeon RX 7900 XTX",
            vendor="amd",
            vram_total_gb=vram,
            backend="rocm",
            compute_dtype="fp16",
            bf16=True,
            fp8=False,
            gfx_target="gfx1100",
        )
    elif kind == "intel":
        g = Gpu(
            name="Intel(R) Arc(TM) A770 Graphics",
            vendor="intel",
            vram_total_gb=vram,
            backend="xpu",
            compute_dtype="fp16",
            bf16=True,
            fp8=False,
        )
    elif kind == "apple":
        g = Gpu(
            name="Apple M2 Pro",
            vendor="apple",
            vram_total_gb=ram * hardware.UNIFIED_MEMORY_SHARE,
            backend="mps",
            compute_dtype="fp16",
            bf16=True,
            fp8=False,
            memory="unified",
        )
    elif kind == "apu":
        g = Gpu(
            name="AMD Radeon(TM) 8060S Graphics",
            vendor="amd",
            vram_total_gb=vram + ram * hardware.SHARED_MEMORY_SHARE,
            backend="rocm",
            compute_dtype="fp16",
            bf16=True,
            fp8=False,
            memory="shared",
            gfx_target="gfx1151",
        )
    else:
        g = classify_gpu(
            Gpu(
                name="NVIDIA GeForce RTX 4090",
                vendor="nvidia",
                vram_total_gb=vram,
                driver="581.57",
                backend="cuda",
                compute_capability="8.9",
            ),
            windows_driver=True,
            comfy_windows=True,
        )
    return SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[g])


@pytest.mark.parametrize("kind", ["amd", "intel", "apple", "apu"])
def test_non_nvidia_is_capped_at_good_and_marked_estimated(kind):
    e = estimate(small_image(1.0), vendor_system(kind, 48.0, 64.0), coefficients=Coefficients())
    assert e.fit == "good" and e.basis == "estimated"
    label = {"amd": "AMD", "apu": "AMD", "intel": "Intel", "apple": "Apple"}[kind]
    assert any(f"推定({label}・実測なし)" in r for r in e.reasons)
    nv = estimate(small_image(1.0), vendor_system("nvidia", 48.0, 64.0), coefficients=Coefficients())
    assert nv.fit == "perfect"  # NVIDIA は変えない


@pytest.mark.parametrize("kind", ["apple", "apu"])
def test_shared_memory_never_offloads(kind):
    big = small_image(20.0)
    e = estimate(big, vendor_system(kind, 0.5, 32.0), coefficients=Coefficients())
    assert e.run_mode != "offload"


@pytest.mark.parametrize("kind", ["nvidia", "amd", "intel", "apu"])
def test_fit_monotonic_in_vram_across_vendors(kind):
    for e in (small_image(2.0), small_image(6.0), small_image(14.0)):
        prev = -1
        for vram in (2, 4, 6, 8, 12, 16, 24, 32, 48):
            fit = estimate(e, vendor_system(kind, vram), coefficients=Coefficients()).fit
            assert ORDER.index(fit) >= prev, (kind, vram, fit)
            prev = ORDER.index(fit)


@pytest.mark.parametrize("kind", ["apple", "apu", "amd"])
def test_fit_monotonic_in_ram_across_vendors(kind):
    for e in (small_image(2.0), small_image(6.0), small_image(14.0)):
        prev = -1
        for ram in (8, 12, 16, 24, 32, 48, 64, 96):
            fit = estimate(e, vendor_system(kind, 8.0, ram), coefficients=Coefficients()).fit
            assert ORDER.index(fit) >= prev, (kind, ram, fit)
            prev = ORDER.index(fit)


def test_measurement_from_another_vendor_is_calibration_only(db):
    s = vendor_system("amd", 24.0, 64.0)
    e = judge(db.get("image_sdxl_simple"), s, db).estimate
    assert e.basis == "estimated" and e.basis_detail == "calibrated" and e.fit_from == "estimate"
    assert any("AMD の GPU での実測は無い" in r for r in e.reasons)


# --- #39 CPU で動く系統 -------------------------------------------------------------------------------


def no_gpu(ram: float) -> SystemInfo:
    return SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[])


@pytest.mark.parametrize(
    "model_id",
    [
        "image_sdxl_simple",
        "hf:stabilityai/stable-audio-3-small-sfx",
        "manual:stable-audio-3-small-music",
        "hf:stabilityai/stable-audio-open-1.0",
    ],
)
@pytest.mark.parametrize("ram", [16, 32])
def test_sd_and_stable_audio_run_on_cpu(db, model_id, ram):
    e = judge(db.get(model_id), no_gpu(ram), db).estimate
    assert (e.fit, e.run_mode) == ("marginal", "cpu"), (model_id, ram, e.reasons[:2])
    assert any("CPU で動く系統" in r for r in e.reasons) and any("CPU で動く根拠" in r for r in e.reasons)


def test_sdxl_cpu_ram_counts_fp32_weights(db):
    e = judge(db.get("image_sdxl_simple"), no_gpu(32), db).estimate
    assert 12.0 < e.ram_gb < 20.0  # 本体 4.8 GB と VAE を fp32 に広げた分 + 作業用メモリ × 2 + 固定分
    assert any("1 枚に数分〜十数分" in r for r in e.reasons)
    tiny = judge(db.get("image_sdxl_simple"), no_gpu(4), db).estimate
    assert tiny.fit == "too_tight"  # メモリが足りなければ動かない


def test_cpu_family_excludes_untested_versions(db):
    from genmodelfit.estimate import cpu_family

    sa = [m for m in db.models if "medium" in m.id.lower() and (m.comfy_family == "stable_audio")]
    assert all(cpu_family(m) is None for m in sa)
    assert cpu_family(db.get("utility_image_upscale_supir")) is None


def test_cpu_judgement_monotonic_in_ram(db):
    for mid in ("image_sdxl_simple", "manual:stable-audio-3-small-music"):
        prev = -1
        for ram in (4, 8, 12, 16, 24, 32, 64):
            fit = judge(db.get(mid), no_gpu(ram), db).estimate.fit
            assert ORDER.index(fit) >= prev
            prev = ORDER.index(fit)


# --- 表の先頭の行と Mode ---------------------------------------------------------------------------------


def test_no_gpu_header_lines(db, monkeypatch):
    adapters, _ = parse_wmi_video(fx("wmi_uhd770.json"), [])
    s = system_of(monkeypatch, adapters, WIN11, ram_gb=15.7)
    sel = select(db, s, min_fit="marginal", include_tools=False, drop_unknown_size=True)
    console = make_console(width=100)
    with console.capture() as cap:
        print_fit_table(
            sel.fits, title="Recommended", system=s, console=console, matched=sel.matched, hidden=sel.hidden
        )
    lines = cap.get().splitlines()
    assert lines[0].startswith(
        "PC: Intel(R) UHD Graphics 770 (integrated; not usable for generation → judged"
    )
    tools = sel.hidden["tools"]
    assert any(
        "This PC has no usable GPU: only models that run on CPU are shown"
        f" ({tools} tools hidden: --include-tools)" in x
        for x in lines[:3]
    )
    assert all(len(x) <= 100 for x in lines)


def test_mode_cell_gpu_req_and_ram(db):
    e = judge(db.get("image_z_image_turbo"), no_gpu(32), db).estimate
    assert e.fit == "too_tight" and mode_cell(e, no_gpu(32)) == "GPU req"
    tiny = judge(db.get("image_sdxl_simple"), no_gpu(4), db).estimate
    assert mode_cell(tiny, no_gpu(4)) == "RAM"
    assert no_gpu_line(vendor_system("amd", 24.0), {}) == ""


# --- #26 同じ GPU で成功した実測 ---------------------------------------------------------------------


def nvidia(name: str, vram: float, ram: float = 32.0) -> SystemInfo:
    g = Gpu(name=name, vendor="nvidia", vram_total_gb=vram, driver="610.88", backend="cuda")
    classify_gpu(g, windows_driver=True, comfy_windows=True)
    return SystemInfo(os="Windows", cpu="", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[g])


def test_issue_26_measured_success_on_same_gpu_is_good(db):
    """Issue #26 の再現: RTX 5080 16GB で 15.8 GB・56 秒で成功した Wan 2.2 5B が
    marginal / offload だった。"""
    e = judge(db.get("video_wan2_2_5B_ti2v"), nvidia("NVIDIA GeForce RTX 5080", 15.9), db).estimate
    assert (e.fit, e.run_mode, e.basis, e.fit_from) == ("good", "gpu", "measured", "measurement")
    assert e.measured_peak_vram_gb == 15.8
    assert any("段階は実測で決めた" in r for r in e.reasons)
    # 表でも GPU・実測の VRAM(m)を出す
    from genmodelfit.recommend import mode_label, vram_need

    s = nvidia("NVIDIA GeForce RTX 5080", 15.9)
    assert mode_label(e, s) == "GPU" and vram_need(e, s) == 15.8


def test_issue_26_smaller_gpu_stays_estimated(db):
    e = judge(db.get("video_wan2_2_5B_ti2v"), nvidia("NVIDIA GeForce RTX 3060", 12.0), db).estimate
    assert (e.fit, e.run_mode, e.fit_from) == ("marginal", "offload", "estimate")
    assert any("段階は見積もりで決めた" in r for r in e.reasons)


def test_issue_26_pinned_measurement_is_not_perfect(db):
    """張り付いた実測(動的 VRAM が空きを使い切った)は、大きな GPU でも perfect の根拠にしない。"""
    e = judge(db.get("video_wan2_2_5B_ti2v"), nvidia("NVIDIA GeForce RTX 5090", 31.8), db).estimate
    assert e.fit in ("good", "perfect") and e.run_mode == "gpu"
    if e.fit_from == "measurement":
        assert e.fit == "good"


def test_issue_26_monotonic_in_vram(db):
    m = db.get("video_wan2_2_5B_ti2v")
    prev = -1
    for vram in (8, 12, 15.9, 16, 20, 24, 32, 48):
        fit = judge(m, nvidia("NVIDIA GeForce RTX 5080", vram), db).estimate.fit
        assert ORDER.index(fit) >= prev, (vram, fit)
        prev = ORDER.index(fit)


def test_issue_26_ram_still_limits(db):
    e = judge(db.get("video_wan2_2_5B_ti2v"), nvidia("NVIDIA GeForce RTX 5080", 15.9, ram=4.0), db).estimate
    assert e.fit in ("marginal", "too_tight")
