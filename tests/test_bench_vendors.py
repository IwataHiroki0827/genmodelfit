"""bench を NVIDIA 以外の GPU(AMD・Intel・Apple)で使う(0.9.0)。
空きの確かめ方・道具が無いとき・提出の会社と backend。

ComfyUI は test_bench の偽のサーバー。道具の出力は test_gpustat の公式の出力例で差し替える
(本物の道具は呼ばない)。"""

from __future__ import annotations

import json
import threading

import pytest
from test_bench import GIB, FakeComfy, _opts, _quiet
from test_gpustat import IOREG, ROCM_SMI, XPU_SMI

from genmodelfit import bench, contrib, gpustat, hardware
from genmodelfit.bench import BenchError, device_vendor
from genmodelfit.models import Measurement, Settings

AMD_DEV = ("cuda:0 AMD Radeon RX 7900 XTX : native", "cuda")
INTEL_DEV = ("xpu:0 Intel(R) Arc(TM) B580 Graphics", "xpu")
APPLE_DEV = ("mps", "mps")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))

    def no_smi():
        raise AssertionError("NVIDIA 以外の GPU で nvidia-smi を呼んだ")

    monkeypatch.setattr(bench, "smi_gpus", no_smi)
    monkeypatch.setattr(bench, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(bench, "find_server_process", lambda url: None)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)


def _os(monkeypatch, system: str, outputs: dict[str, str]):
    """道具の名前 → 出力。無い道具は「見つからない」(None)。"""
    monkeypatch.setattr(gpustat.platform, "system", lambda: system)
    monkeypatch.setattr(
        hardware, "_run", lambda args, timeout: (0, outputs[args[0]]) if args[0] in outputs else None
    )


def _amd(fake: FakeComfy) -> FakeComfy:
    fake.device = AMD_DEV
    fake.pytorch_version = "2.9.1+rocm7.1"
    return fake


def test_device_vendor_from_system_stats():
    def stats(v="2.9.0+cu128"):
        return {"system": {"pytorch_version": v}}

    assert device_vendor({"name": AMD_DEV[0], "type": "cuda"}, stats()) == "amd"
    assert (
        device_vendor({"name": "cuda:0 Radeon 8060S Graphics : native", "type": "cuda"}, stats("2.9+rocm"))
        == "amd"
    )
    assert (
        device_vendor({"name": "cuda:0 NVIDIA GeForce RTX 3070 : cudaMallocAsync", "type": "cuda"}, stats())
        == "nvidia"
    )
    assert device_vendor({"name": INTEL_DEV[0], "type": "xpu"}, stats()) == "intel"
    assert device_vendor({"name": "mps", "type": "mps"}, {}) == "apple"


def test_amd_linux_checks_with_rocm_smi_and_records_backend(monkeypatch):
    idle = json.dumps({"card0": {"GPU use (%)": "3", "VRAM Total Used Memory (B)": str(GIB)}})
    _os(monkeypatch, "Linux", {"rocm-smi": idle})  # amd-smi は無い
    with FakeComfy() as fake:
        res = bench.run(_opts(_amd(fake).url), log=_quiet)
    m = res.measurement
    assert m.outcome == "ok" and m.standard and m.gpu_backend == "rocm"
    assert "rocm-smi" in m.note and "確かめていない" not in m.note
    assert not any("--force" in w for w in res.warnings)


def test_amd_busy_gpu_is_refused_before_any_write(monkeypatch):
    _os(monkeypatch, "Linux", {"rocm-smi": ROCM_SMI})  # card0 の使用率 45%
    with FakeComfy() as fake:
        with pytest.raises(BenchError, match=r"rocm-smi.*45%"):
            bench.run(_opts(_amd(fake).url, free_before=True), log=_quiet)
        assert fake.mutations == []
        res = bench.run(_opts(fake.url, force=True), log=_quiet)
    assert any("45%" in w and "--force" in w for w in res.warnings)


def test_intel_linux_xpu_smi(monkeypatch):
    _os(monkeypatch, "Linux", {"xpu-smi": XPU_SMI.replace("41.50", "2.00")})
    with FakeComfy() as fake:
        fake.device = INTEL_DEV
        res = bench.run(_opts(fake.url), log=_quiet)
    assert res.measurement.gpu_backend == "xpu" and "xpu-smi" in res.measurement.note


def test_intel_windows_uses_performance_counters(monkeypatch):
    head = (
        '"(PDH-CSV 4.0)",'
        '"\\\\HOST\\GPU Engine(pid_1_luid_0x00000000_0x00000001_phys_0_eng_0_engtype_3D)'
        '\\Utilization Percentage",'
        '"\\\\HOST\\GPU Adapter Memory(luid_0x00000000_0x00000001_phys_0)\\Dedicated Usage"\n'
    )
    _os(monkeypatch, "Windows", {"typeperf": head + '"t","1.5","1073741824"\n'})
    with FakeComfy() as fake:
        fake.device = INTEL_DEV
        res = bench.run(_opts(fake.url), log=_quiet)
    assert "性能カウンタ" in res.measurement.note and res.measurement.standard


def test_apple_ioreg_busy_is_refused(monkeypatch):
    _os(monkeypatch, "Darwin", {"ioreg": IOREG})  # Device Utilization 82%
    with FakeComfy() as fake:
        fake.device = APPLE_DEV
        with pytest.raises(BenchError, match="82%"):
            bench.run(_opts(fake.url), log=_quiet)


@pytest.mark.parametrize("output", [None, "garbage"], ids=["missing_or_timeout", "broken"])
def test_no_tool_proceeds_with_queue_check_and_notes(monkeypatch, output):
    _os(monkeypatch, "Linux", {} if output is None else {"amd-smi": output, "rocm-smi": output})
    with FakeComfy() as fake:
        res = bench.run(_opts(_amd(fake).url), log=_quiet)
    m = res.measurement
    # 止めずに測る。standard は変えない(--force と同じ扱い。docs/accuracy.md)
    assert m.outcome == "ok" and m.standard and m.gpu_backend == "rocm"
    assert "使用率は確かめていない" in m.note and "amd-smi・rocm-smi" in m.note
    assert any("ほかのアプリが GPU を使っていても分からない" in w for w in res.warnings)


def test_no_tool_refuses_when_other_apps_hold_most_vram(monkeypatch):
    _os(monkeypatch, "Linux", {})
    with FakeComfy() as fake:
        _amd(fake).other_apps = 6 * GIB  # 8 GB のうち 6 GB をほかのアプリが使っている
        with pytest.raises(BenchError, match="ComfyUI 以外"):
            bench.run(_opts(fake.url), log=_quiet)
        assert fake.mutations == []
        res = bench.run(_opts(fake.url, force=True), log=_quiet)
    assert any("ComfyUI 以外" in w for w in res.warnings)


def test_no_tool_apple_does_not_use_vram_share(monkeypatch):
    _os(monkeypatch, "Darwin", {})  # 統合メモリはメインメモリと同じなので、ほかのアプリの分を見ない
    with FakeComfy() as fake:
        fake.device = APPLE_DEV
        fake.other_apps = 7 * GIB
        res = bench.run(_opts(fake.url), log=_quiet)
    assert res.measurement.gpu_backend == "mps"


def test_remote_non_nvidia_still_needs_force(monkeypatch):
    _os(monkeypatch, "Linux", {"rocm-smi": ROCM_SMI})
    monkeypatch.setattr(bench, "is_local", lambda url: False)
    with FakeComfy() as fake:
        with pytest.raises(BenchError, match=r"別の PC.*--force"):
            bench.run(_opts(_amd(fake).url), log=_quiet)
        res = bench.run(_opts(fake.url, force=True), log=_quiet)
    assert "確かめていない(--force)" in res.measurement.note


def test_monitor_reads_other_vendor_peak():
    class Stats:
        def system_stats(self):
            return {"devices": [{"torch_vram_total": GIB, "vram_total": 8 * GIB, "vram_free": 6 * GIB}]}

    values = iter([1000.0, None, 3000.0])
    seen = threading.Event()

    def read():
        v = next(values, 2000.0)
        if v == 2000.0:
            seen.set()
        if v is None:
            raise OSError("道具が一度失敗しても測定は止めない")
        return v

    mon = bench.Monitor(Stats(), None, None, interval=0.01, use_smi=False, other_used=read)
    real_wait = mon._stop.wait
    mon._stop.wait = lambda t=None: real_wait(0.001)  # 1 秒待たずに次を読む
    mon.start()
    assert seen.wait(5)
    peaks = mon.stop()
    assert peaks.gpu_used_mib == 3000.0 and peaks.smi_samples >= 3


# --- 提出の会社・backend -------------------------------------------------------------------------


def _m(name: str, backend: str | None) -> Measurement:
    return Measurement(
        model_id="image_sdxl_simple",
        gpu_name=name,
        gpu_vram_gb=24.0,
        peak_vram_gb=7.0,
        settings=Settings(width=1024, height=1024, steps=20),
        seconds=30.0,
        comfyui_version="0.40.0",
        date="2026-10-21",
        outcome="ok",
        vram_metric="torch_reserved",
        gpu_backend=backend,
    )


def test_submission_records_vendor_and_backend():
    data = contrib.build_submission(_m("AMD Radeon RX 7900 XTX", "rocm"))
    gpu = data["system"]["gpu"]
    assert gpu["vendor"] == "amd" and gpu["backend"] == "rocm"
    assert gpu["arch"] is None and gpu["compute_capability"] is None
    r = contrib.validate_submission(data, check_filename=False)
    assert r.ok, r.errors
    assert contrib.to_measurement(data).gpu_backend == "rocm"
    # 記録が無い古い bench の結果からは、会社・backend を入れない(これまでと同じ形)
    old_m = _m("NVIDIA GeForce RTX 3070", None)
    old_m.gpu_vram_gb = 8.0  # RTX 3070 は 8 GB(名前の表と違う VRAM は誤り。#90)
    old = contrib.build_submission(old_m)
    assert "vendor" not in old["system"]["gpu"] and contrib.validate_submission(old, check_filename=False).ok


def test_submission_keeps_privacy_checklist_known_fields():
    data = contrib.build_submission(_m("Intel Arc B580 Graphics", "xpu"))
    items = {c["id"]: c for c in contrib.privacy_checklist(data)}
    assert items["known_fields"]["ok"] is True


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"vendor": "amd", "backend": "cuda"}, "gpu_inconsistent"),
        ({"vendor": "amd"}, "gpu_inconsistent"),  # backend が無い
        ({"vendor": "matrox", "backend": "rocm"}, "value"),
        ({"vendor": "amd", "backend": "vulkan"}, "value"),
        ({"vendor": "nvidia", "backend": "cuda"}, "gpu_inconsistent"),  # 名前は AMD
        ({"compute_capability": "8.6"}, "gpu_inconsistent"),  # NVIDIA 以外に compute capability
    ],
)
def test_submission_vendor_backend_checks(change, code):
    data = contrib.build_submission(_m("AMD Radeon RX 7900 XTX", "rocm"))
    data["system"]["gpu"].update(change)
    if "vendor" in change and "backend" not in change:
        data["system"]["gpu"].pop("backend")
    r = contrib.validate_submission(data, check_filename=False)
    assert any(e["code"] == code for e in r.errors), r.errors


def test_contrib_show_lists_vendor_and_backend(tmp_path, capsys):
    from genmodelfit import cli

    data = contrib.build_submission(_m("AMD Radeon RX 7900 XTX", "rocm"))
    p = contrib.write_submission(data, tmp_path / "data" / "contrib")
    cli.main(["contrib", "show", str(p)])
    out = capsys.readouterr().out
    assert "会社 amd" in out and "backend rocm" in out


# --- Apple: ComfyUI の MPS の GPU 名("mps")の代わりに、この PC のチップ名を使う(0.9.0) ----------------


def _sp(name: str = "Apple M2 Pro") -> str:
    return json.dumps(
        {
            "SPDisplaysDataType": [
                {"_name": name, "sppci_model": name, "spdisplays_vendor": "sppci_vendor_Apple"}
            ]
        }
    )


def test_apple_bench_uses_the_chip_name(monkeypatch, tmp_path):
    _os(monkeypatch, "Darwin", {"system_profiler": _sp()})
    with FakeComfy() as fake:
        fake.device = APPLE_DEV
        res = bench.run(_opts(fake.url), log=_quiet)
    m = res.measurement
    assert (m.gpu_name, m.gpu_backend) == ("Apple M2 Pro", "mps")
    m.seconds = 30.0  # 偽の ComfyUI はすぐ終わる(0.0 秒は提出の範囲の外。test_contrib の番兵の試験と同じ)
    data = contrib.build_submission(m)
    assert data["system"]["gpu"]["name"] == "Apple M2 Pro" and data["system"]["gpu"]["vendor"] == "apple"
    r = contrib.validate_submission(data, check_filename=False)
    assert r.ok, r.errors
    assert all(c["ok"] is not False for c in contrib.privacy_checklist(data)), contrib.privacy_checklist(data)
    assert contrib.suggested_path(data).startswith("data/contrib/apple-m2-pro/")


@pytest.mark.parametrize(
    "output",
    [None, _sp("Apple M2 Pro C02XK0ABCDEF"), _sp("Apple M2 Pro (bob's MacBook)")],
    ids=["unreadable", "serial_like", "free_text"],
)
def test_apple_bench_keeps_mps_without_a_plain_chip_name(monkeypatch, output):
    """チップ名の形(Apple M2 Pro など)に合わない値は入れない(シリアル・名前などを持ち込まない)。"""
    _os(monkeypatch, "Darwin", {} if output is None else {"system_profiler": output})
    with FakeComfy() as fake:
        fake.device = APPLE_DEV
        res = bench.run(_opts(fake.url), log=_quiet)
    assert res.measurement.gpu_name == "mps"
    assert any("チップ名が分からない" in w for w in res.warnings)


def test_apple_bench_remote_keeps_mps(monkeypatch):
    """ComfyUI が別の PC なら、この PC のチップ名は使わない(--force が要るのは今までどおり)。"""
    calls = []
    _os(monkeypatch, "Darwin", {"system_profiler": _sp()})
    monkeypatch.setattr(hardware, "apple_chip_name", lambda: calls.append(1) or "Apple M2 Pro")
    monkeypatch.setattr(bench, "is_local", lambda url: False)
    with FakeComfy() as fake:
        fake.device = APPLE_DEV
        with pytest.raises(BenchError, match="--force"):
            bench.run(_opts(fake.url), log=_quiet)
        res = bench.run(_opts(fake.url, force=True), log=_quiet)
    assert res.measurement.gpu_name == "mps" and not calls


@pytest.mark.parametrize(
    ("name", "ok"), [("Apple M2 Pro", True), ("mps", True), ("Apple M2 Pro X1234", False)]
)
def test_submission_apple_gpu_name_is_chip_or_mps(name, ok):
    data = contrib.build_submission(_m(name, "mps"))
    r = contrib.validate_submission(data, check_filename=False)
    assert r.ok is ok, r.errors
    if not ok:
        assert any(e["code"] == "gpu_inconsistent" for e in r.errors)


def test_apple_measurement_matches_this_pc_by_chip_name():
    """判定は、この PC の GPU 名(system_profiler のチップ名)と実測の GPU 名で同じ GPU を見る。
    古い "mps" の実測は会社 apple としてだけ使う(同じ GPU の実測にはならない)。"""
    from genmodelfit.db import load
    from genmodelfit.estimate import _measured_band, gpu_precision, measurement_vendor, same_gpu_name
    from genmodelfit.hardware import Host, classify_adapter, parse_system_profiler
    from genmodelfit.models import SystemInfo

    (a,) = parse_system_profiler(_sp())
    g = classify_adapter(a, Host("Darwin", None, (15, 0)), 32.0)
    pc = SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g])
    db = load(use_cache=False)
    entry = db.get("image_sdxl_simple")
    base = next(m for m in db.measurements if m.model_id == entry.id)
    s = base.settings
    kw = {
        **vars(base),
        "settings": s,
        "gpu_vram_gb": g.vram_total_gb,
        "peak_vram_gb": 6.0,
        "vram_metric": "torch_reserved",
        "origin": "community",
        "n_used": 1,
        "n_oom": 0,
    }
    chip = Measurement(**{**kw, "gpu_name": "Apple M2 Pro", "gpu_backend": "mps"})
    assert same_gpu_name(chip.gpu_name, g.name)
    assert _measured_band(chip, entry, s, pc, gpu_precision(entry, pc, s)) is not None
    old = Measurement(**{**kw, "gpu_name": "mps", "gpu_backend": None})
    assert measurement_vendor(old) == "apple"
    assert _measured_band(old, entry, s, pc, gpu_precision(entry, pc, s)) is None


# --- PCI の型番 ID(0.9.0): ComfyUI と同じ PC なら、検出した部品から vendor:device を記録する ----------


def _wmi(*items: tuple[str, str]) -> str:
    return json.dumps(
        [
            {"Name": n, "PNPDeviceID": f"PCI\\VEN_{v}&DEV_{d}&SUBSYS_0E3A1002&REV_C8\6&1&0&0"}
            for n, (v, d) in items
        ]
    )


@pytest.mark.skipif(
    not hasattr(hardware.os, "name") or hardware.os.name != "nt", reason="WMI は Windows だけ"
)
def test_bench_records_pci_id_of_the_same_pc(monkeypatch):
    wmi = _wmi(("AMD Radeon RX 7900 XTX", ("1002", "744C")), ("Intel(R) UHD Graphics 770", ("8086", "4680")))
    _os(monkeypatch, "Windows", {"powershell.exe": wmi})
    with FakeComfy() as fake:
        res = bench.run(_opts(_amd(fake).url), log=_quiet)
    m = res.measurement
    assert m.gpu_pci_id == "1002:744c"
    m.seconds = 30.0
    data = contrib.build_submission(m)
    assert data["system"]["gpu"]["pci_id"] == "1002:744c"
    assert contrib.validate_submission(data, check_filename=False).ok
    # 同じ会社の部品が 2 つで名前も合わなければ決めない
    two = _wmi(("AMD Radeon RX 6600", ("1002", "73FF")), ("AMD Radeon 780M Graphics", ("1002", "15BF")))
    _os(monkeypatch, "Windows", {"powershell.exe": two})
    with FakeComfy() as fake:
        assert bench.run(_opts(_amd(fake).url), log=_quiet).measurement.gpu_pci_id is None


def test_bench_remote_does_not_record_pci_id(monkeypatch):
    calls = []
    monkeypatch.setattr(hardware, "local_pci_id", lambda v, n: calls.append(1) or "1002:744c")
    _os(monkeypatch, "Linux", {"rocm-smi": ROCM_SMI})
    monkeypatch.setattr(bench, "is_local", lambda url: False)
    with FakeComfy() as fake:
        res = bench.run(_opts(_amd(fake).url, force=True), log=_quiet)
    assert res.measurement.gpu_pci_id is None and not calls


def test_lspci_and_wmi_keep_the_pci_model_id_only():
    """型番 ID は vendor:device だけ(差し場所のバス番号・SUBSYS・個体の番号は入れない)。"""
    adapters, _ = hardware.parse_wmi_video(_wmi(("AMD Radeon RX 7900 XTX", ("1002", "744C"))))
    assert adapters[0].pci_id == "1002:744c"
    lspci = (
        "0000:03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] "
        "Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] [1002:744c] (rev c8)\n"
    )
    (a,) = hardware.parse_lspci(lspci, vram_of=lambda slot: 24 * 2**30)
    assert a.pci_id == "1002:744c" and "0000:03" not in json.dumps(vars(a))
    g = hardware.classify_adapter(a, hardware.Host("Linux", None, cpu="AMD Ryzen 7 7700"), 32.0)
    assert g is not None and g.pci_id == "1002:744c"
