# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""measure・share を NVIDIA 以外の GPU(AMD・Intel・Apple)で使う(0.13.1。0.13.0 で誤って止めていた)。

この PC は偽物(Windows 11 の RDNA 3 の ROCm・Intel Arc の XPU・Apple の MPS)。ComfyUI は test_bench の偽の
サーバー、GitHub は test_share_pr の偽物。本物の GPU・道具(amd-smi・typeperf・ioreg など)・nvidia-smi は
呼ばない(呼んだら落とす)。"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bench import FakeComfy
from test_share_pr import TOKEN, FakeGitHub

from genmodelfit import bench, cli, gpustat, hardware, measure
from genmodelfit.db import load as load_db
from genmodelfit.models import Gpu, SystemInfo

ROOT = Path(__file__).resolve().parents[1]

# 偽の ComfyUI の VRAM は 8 GB なので、8 GB の GPU にする
VENDORS = {
    "amd": {
        "gpu": Gpu(
            name="AMD Radeon RX 7600",
            vendor="amd",
            vram_total_gb=8.0,
            backend="rocm",
            pci_id="1002:7480",
            gfx_target="gfx1102",
        ),
        "os": "Windows 11",
        "platform": "Windows",
        "device": ("cuda:0 AMD Radeon RX 7600 : native", "cuda"),
        "torch": "2.9.1+rocm7.1",
        "pci_id": "1002:7480",
        "backend": "rocm",
        "name": "AMD Radeon RX 7600",
    },
    "intel": {
        "gpu": Gpu(
            name="Intel(R) Arc(TM) A750 Graphics",
            vendor="intel",
            vram_total_gb=8.0,
            backend="xpu",
            pci_id="8086:56a1",
        ),
        "os": "Windows 11",
        "platform": "Windows",
        "device": ("xpu:0 Intel(R) Arc(TM) A750 Graphics", "xpu"),
        "torch": "2.9.0+xpu",
        "pci_id": "8086:56a1",
        "backend": "xpu",
        "name": "Intel(R) Arc(TM) A750 Graphics",
    },
    "apple": {
        "gpu": Gpu(name="Apple M2", vendor="apple", vram_total_gb=8.0, backend="mps", memory="unified"),
        "os": "macOS 15",
        "platform": "Darwin",
        "device": ("mps", "mps"),
        "torch": "2.9.0",
        "pci_id": None,
        "backend": "mps",
        "name": "Apple M2",
    },
}


def _never(what: str):
    def fail(*a, **k):
        raise AssertionError(f"NVIDIA 以外の GPU で {what} を呼んだ")

    return fail


@pytest.fixture(params=sorted(VENDORS))
def vendor_pc(request, monkeypatch):
    v = {**VENDORS[request.param], "vendor": request.param}
    s = SystemInfo(os=v["os"], cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[v["gpu"]])
    monkeypatch.setattr(cli, "_system", lambda args: s)
    monkeypatch.setattr(cli, "detect", lambda *a, **k: s)
    # 道具は無い(hardware._run は conftest が塞ぐ)。OS だけ合わせる
    monkeypatch.setattr(gpustat.platform, "system", lambda: v["platform"])
    monkeypatch.setattr(hardware, "local_pci_id", lambda vendor, name: v["pci_id"])
    monkeypatch.setattr(hardware, "apple_chip_name", lambda: "Apple M2")
    # nvidia-smi に頼るもの(GPU の一覧・空き・測っているあいだのプロセスの見張り)は使わない
    monkeypatch.setattr(bench, "smi_gpus", _never("nvidia-smi の GPU の一覧"))
    monkeypatch.setattr(bench, "vram_in_use", _never("nvidia-smi の VRAM"))
    monkeypatch.setattr(bench, "_gpu_app_pids", _never("nvidia-smi のプロセスの表"))
    monkeypatch.setattr(bench, "GpuAppWatch", _never("プロセスの見張り"))
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    real_run = bench.run

    def run(*a, **k):  # 偽の ComfyUI はすぐ終わるので、検査に通る時間にする(test_measure_share と同じ)
        res = real_run(*a, **k)
        res.measurement.seconds = 2.0
        return res

    monkeypatch.setattr(bench, "run", run)
    return v


def _comfy(fake: FakeComfy, v: dict) -> FakeComfy:
    fake.device = v["device"]
    fake.pytorch_version = v["torch"]
    return fake


def test_measure_suggests_on_non_nvidia_gpus(capsys, vendor_pc):
    assert cli.main(["measure", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["action"] == "suggest" and "error" not in d
    assert d["suggestions"], "NVIDIA 以外の GPU で提案が空"


def test_measure_measures_and_the_submission_validates(capsys, vendor_pc):
    v = vendor_pc
    with FakeComfy() as fake:
        argv = ["measure", "image_sdxl_simple", "--json", "--yes", "--free", "--url", _comfy(fake, v).url]
        assert cli.main(argv) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["measurement"]["outcome"] == "ok" and d["submission"], d
    sub = json.loads(Path(d["submission"]["saved_to"]).read_text(encoding="utf-8"))
    gpu = sub["system"]["gpu"]
    assert gpu["vendor"] == v["vendor"] and gpu["backend"] == v["backend"] and gpu["name"] == v["name"]
    assert "compute_capability" not in gpu or gpu["compute_capability"] is None
    assert gpu.get("pci_id") == v["pci_id"]
    assert cli._submission_refusal(sub, cli.build_parser().parse_args(["share"])) == []
    note = d["measurement"]["note"]
    # 確かめられなかったことは note に残る(測っているあいだの見張り・Intel と Apple の /system_stats)
    assert bench.GPU_APPS_UNCHECKED in note and bench.GPU_APPS_UNCHECKED in sub["measurement"]["note"]
    assert (bench.OTHERS_VRAM_UNCHECKED in note) == (v["vendor"] != "amd")


def test_share_does_not_stop_on_non_nvidia_gpus(capsys, vendor_pc):
    """GPU で止めない(0.13.0 は「NVIDIA だけ」で error にした)。--url の ComfyUI はフォルダが分からないので、
    入っているか分からないものは計画に入らない(GPU の会社によらない)。"""
    with FakeComfy() as fake:
        assert cli.main(["share", "--dry-run", "--json", "--url", _comfy(fake, vendor_pc).url]) == 0
    assert "error" not in json.loads(capsys.readouterr().out)


def test_share_measures_and_sends_on_non_nvidia_gpus(capsys, monkeypatch, vendor_pc):
    gh = FakeGitHub()
    monkeypatch.setattr(cli, "_share_session", lambda: gh)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    db = load_db(use_cache=False)
    e = next(m for m in db.models if m.id == "image_sdxl_simple")
    one = measure.Suggestion(
        model_id=e.id,
        model_name=e.name,
        name=e.name,
        modality=e.modality,
        precision=e.precision,
        fit_level="good",
        installed=measure.Installed("yes", 0, 0, 0),
        seconds=60,
        seconds_basis="typical",
        group_key=e.id,
        fit=SimpleNamespace(model=e),
    )
    monkeypatch.setattr(measure, "suggestions", lambda *a, **k: [one])
    with FakeComfy() as fake:
        assert cli.main(["share", "--yes", "--json", "--url", _comfy(fake, vendor_pc).url]) == 0
    d = json.loads(capsys.readouterr().out)
    assert [x["model_id"] for x in d["measured"]] == ["image_sdxl_simple"] and not d.get("failed"), d
    assert len(gh.put) == 1, "送っていない"
    sent = json.loads(base64.b64decode(next(iter(gh.put.values()))["content"]))
    assert sent["system"]["gpu"]["vendor"] == vendor_pc["vendor"]
    assert sent["system"]["gpu"]["backend"] == vendor_pc["backend"]


def test_no_text_says_only_nvidia_is_measured():
    """0.13.0 の「今は NVIDIA の GPU だけで測ります」を、コードにも文書にも残さない。"""
    pat = re.compile(r"only NVIDIA GPUs are measured|NVIDIA の GPU だけで測")
    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "docs").rglob("*.md")]
    files += [p for p in ROOT.glob("*.md") if p.name != "CHANGELOG.md"]  # CHANGELOG は直したことを引く
    hits = [str(p.relative_to(ROOT)) for p in files if pat.search(p.read_text(encoding="utf-8"))]
    assert hits == []


def test_start_check_does_not_read_nvidia_smi_for_other_vendors(vendor_pc):
    """ComfyUI を起動する前の nvidia-smi の VRAM の確かめは NVIDIA だけ(ほかの会社は、起動して空けたあとに
    bench が /system_stats で見る)。この PC に使えない NVIDIA の GPU があっても、それを数えない。"""
    assert cli._vram_busy_before_start(vendor_pc["gpu"]) is None  # vram_in_use は呼ぶと落ちる
