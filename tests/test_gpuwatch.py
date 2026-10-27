# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU の表の見張り(gpuwatch。#103)。

上流の応答は tests/fixtures/gpu_watch_raw に置いた小さな写し(公式のページと同じ形で、中身は試験のために
変えた: 新しい型番 ID・新しい CUDA の主番号・Release Ready に変わった gfx など)。ネットワークには触らない。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from genmodelfit import gpuwatch

RAW = Path(__file__).parent / "fixtures" / "gpu_watch_raw"
OFFICIAL = Path(__file__).parent / "fixtures" / "gpu_official"


def raw(name: str) -> str:
    return (RAW / name).read_text(encoding="utf-8")


class Fake:
    """URL → 応答。無い URL・raise に挙げた URL は失敗する(読めなかった出典になる)。"""

    def __init__(self, broken: tuple[str, ...] = (), garbage: tuple[str, ...] = ()):
        self.inner = gpuwatch._RawDir(RAW)
        self.broken = broken
        self.garbage = garbage
        self.urls: list[str] = []

    def _name(self, url: str) -> str:
        return self.inner._file(url).stem

    def get_text(self, url: str) -> str:
        self.urls.append(url)
        if self._name(url) in self.broken:
            raise ConnectionError("timed out")
        if self._name(url) in self.garbage:
            return "<html>maintenance</html>"
        return self.inner.get_text(url)

    def get_json(self, url: str):
        return json.loads(self.get_text(url))


# --- 読み方(見出しで読む) --------------------------------------------------------------------------------


def test_parse_pci_ids_keeps_only_gpu_lines():
    p = gpuwatch.parse_pci_ids(raw("pci_ids.ids"))
    assert p["version"] == "2026.11.03"
    assert "1002:12b8" not in p["devices"]  # Data Fabric は GPU ではない
    assert "8086:a0ed" not in p["devices"]
    assert p["devices"]["1002:75b0"] == "Navi 50 [Radeon RX 10070 XT]"
    assert p["devices"]["8086:e2ff"] == "Celestial [Arc C770]"


def test_parse_hw_specs_by_header_not_position():
    rows = gpuwatch.parse_rocm_hw_specs(raw("rocm_hw_specs.html"))
    assert {"name": "MI355X", "gfx": "gfx950"} in rows
    assert {"name": "Ryzen AI 5 330", "gfx": "gfx1152", "graphics_model": "Radeon 820M"} in rows
    # 列の順番が変わっても読める
    moved = (
        "<table><tr><th>LLVM target name</th><th>VRAM</th><th>Name</th></tr>"
        "<tr><td>gfx1201</td><td>16</td><td>Radeon RX 9070</td></tr></table>"
    )
    assert gpuwatch.parse_rocm_hw_specs(moved) == [{"name": "Radeon RX 9070", "gfx": "gfx1201"}]
    with pytest.raises(ValueError):
        gpuwatch.parse_rocm_hw_specs("<table><tr><th>Model</th></tr></table>")


def test_parse_compat_splits_products():
    rows = gpuwatch.parse_compat(raw("radeon_win_compat.html"))
    assert rows == [
        {
            "rocm": "7.2.1",
            "gfx": ["gfx1201", "gfx1200"],
            "products": ["AMD Radeon RX 9070", "AMD Radeon RX 9070 XT", "AMD Radeon RX 9060 XT"],
        }
    ]


def test_parse_therock():
    sup = gpuwatch.parse_therock_supported(raw("therock_supported.md"))
    assert {k: sup[-1][k] for k in ("os", "gfx")} == {"os": "windows", "gfx": "gfx1201"}
    assert any(r["gfx"] == "gfx117x" and r["release"] and r["os"] == "linux" for r in sup)
    rel = gpuwatch.parse_therock_releases(raw("therock_releases.md"))
    assert rel == [
        {"product": "AMD Radeon RX 9070 / XT", "gfx": "gfx1201"},
        {"product": "AMD Radeon 820M iGPU", "gfx": "gfx1153"},
    ]


def test_parse_intel_nvidia_cuda_torch_comfy():
    intel = gpuwatch.parse_intel_xpu(raw("intel_xpu.html"))
    assert "Intel Arc C-Series Graphics (CodeName: Celestial)" in intel["client"][0]["hardware"]
    assert intel["datacenter"][0]["os"] == ["Red Hat Enterprise Linux 9.2", "Ubuntu Server 22.04"]
    nv = gpuwatch.parse_nvidia(raw("nvidia_cuda_gpus.html"), "nvidia_cuda_gpus")
    assert {
        "source": "nvidia_cuda_gpus",
        "segment": "Workstation/Consumer",
        "cc": "13.0",
        "name": "NVIDIA RTX 6090",
    } in nv
    assert gpuwatch.parse_cuda_rn(raw("cuda_rn.html")) == {"14": 600, "13": 580, "12": 525}
    init = gpuwatch.parse_torch_cuda_init(raw("torch_cuda_init.txt"))
    assert init["solo"] == [87, 110] and init["post_jetson"] is True
    assert init["code_cc"]["13.4"]["aarch64"] == [80, 90, 100, 110, 120]
    assert "gfx900" in gpuwatch.parse_rocm_build(raw("torch_build_rocm.sh"))
    idx = gpuwatch.parse_torch_index(raw("torch_index.html"))
    assert idx == {
        "cu129": {"linux_x86_64": ["2.13.0"], "win_amd64": ["2.9.0"]},
        "cu134": {"linux_aarch64": ["2.15.0"]},
    }
    mm = gpuwatch.parse_comfy_mm(raw("comfyui_mm.txt"))
    assert mm["lists"]["nvidia_16_series"][-1] == "T300" and "gfx1300" in mm["gfx"]


def test_comfyui_is_read_without_importing_it():
    """ComfyUI(GPL-3.0)のコードは import しない。ast で読むだけ(書き換えられたコードも実行されない)。"""
    text = "import os\nos.system('echo bad')\nnvidia_10_series = ['1080']\n"
    assert gpuwatch.parse_comfy_mm(text)["lists"]["nvidia_10_series"] == ["1080"]


# --- 照らす ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def result():
    return gpuwatch.watch(Fake(), baseline=OFFICIAL)


def _has(res, source: str, kind: str, item: str) -> bool:
    return any(c.source == source and c.kind == kind and item in c.item for c in res.candidates)


def test_watch_finds_new_and_different(result):
    assert result.unreadable == []
    assert _has(result, "pci_ids", "not in tables", "1002:75b0")  # 表に無い新しい AMD の型番 ID
    assert _has(result, "pci_ids", "not in tables", "8086:e2ff")  # 新しい Intel の世代
    assert _has(result, "rocm_hw_specs", "not in tables", "Radeon 990M")
    assert _has(result, "therock_supported", "differs", "gfx117x (linux)")  # Release Ready に変わった
    assert _has(result, "cuda_rn", "not in tables", "CUDA 14.x")
    assert _has(result, "torch", "not in tables", "CUDA 13.4 (cu134)")  # wheel が出た新しい CUDA 版
    assert _has(result, "torch", "differs", "PYTORCH_ROCM_ARCH")
    assert _has(result, "torch", "differs", "PyTorch")
    assert _has(result, "comfyui", "differs", "nvidia_16_series")
    assert _has(result, "comfyui", "not in tables", "gfx1300")
    assert _has(result, "nvidia", "differs", "GeForce RTX 3070")  # 名前の表と違う cc
    assert _has(result, "nvidia", "not in tables", "NVIDIA RTX 6090")
    assert _has(result, "intel_xpu", "not in tables", "Celestial")
    assert _has(result, "torch_index", "not in tables", "cu134")


def test_fixed_official_lists_give_no_candidates():
    snap = gpuwatch.load_snapshot(OFFICIAL)
    res = gpuwatch.watch(snap=snap, baseline=OFFICIAL)
    assert res.candidates == [] and res.unreadable == []


def test_unreadable_sources_are_listed_and_never_raise():
    res = gpuwatch.watch(
        Fake(broken=("pci_ids", "torch_release"), garbage=("rocm_hw_specs",)), baseline=OFFICIAL
    )
    bad = {u["source"] for u in res.unreadable}
    assert bad == {"pci_ids", "torch_release", "rocm_hw_specs"}
    assert all(u["url"].startswith("https://") and u["error"] for u in res.unreadable)
    assert "pci_ids" not in res.read and "torch" not in res.read and "nvidia" in res.read


def test_watch_survives_a_bug(monkeypatch):
    def boom(_snap):
        raise RuntimeError("bug")

    monkeypatch.setattr(gpuwatch, "check_tables", boom)
    res = gpuwatch.watch(snap={}, baseline=None)
    assert res.unreadable[0]["source"] == "gpuwatch" and "bug" in res.unreadable[0]["error"]


def test_default_fetcher_uses_the_shared_session(monkeypatch):
    """取るのは fetch の HttpFetcher(fetch.make_session: User-Agent・~/.netrc を送らない)。"""
    from genmodelfit import fetch

    f = gpuwatch.default_fetcher()
    assert isinstance(f, fetch.HttpFetcher) and f._timeout == gpuwatch.TIMEOUT


def test_every_url_is_https_and_known():
    for name, url in gpuwatch.URLS.items():
        assert url.startswith("https://"), name
    f = Fake()
    gpuwatch.collect(f)
    assert all(u.startswith("https://") for u in f.urls)
    assert len(f.urls) == len(gpuwatch.URLS)


def test_read_only_commands_never_load_the_watch(tmp_path):
    """読むだけのコマンドと `import genmodelfit` は gpuwatch を読み込まない(見張りは weekly と手で動かす
    `python -m genmodelfit.gpuwatch` だけ。#103)。"""
    import subprocess
    import sys
    import textwrap

    from clihelp import child_env

    contrib_dir = Path(__file__).parent / "fixtures" / "contrib"
    script = textwrap.dedent(
        f"""
        import contextlib, io, sys
        import genmodelfit as gmf
        from genmodelfit import cli
        hw = ["--vram", "8", "--ram", "32"]
        for argv in (["system", "--json", *hw], ["recommend", "--json", *hw], ["info", "image_sdxl_simple"],
                     ["contrib", "validate", {str(contrib_dir)!r}]):
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                cli.main(argv)
            assert "genmodelfit.gpuwatch" not in sys.modules, argv
        sys.stdout.write("OK")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=child_env(tmp_path / "cache"),
        timeout=180,
        check=False,
    )
    assert r.stdout.decode().endswith("OK"), r.stderr.decode(errors="replace")[-2000:]
