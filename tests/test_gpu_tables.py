# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU の登録の表(gputables.py)と公式の一覧の照合(#103)。

公式の一覧は tests/fixtures/gpu_official に固定した(2026-10-27 に取った。`python -m genmodelfit.gpuwatch
--save-fixtures` で作る)。ネットワークには触らない。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from genmodelfit import gputables, gpuwatch, hardware, i18n
from genmodelfit.hardware import Host, classify_adapter, torch_support
from genmodelfit.models import Adapter

OFFICIAL = Path(__file__).parent / "fixtures" / "gpu_official"
WIN11 = Host("Windows", 26100, cpu="AMD Ryzen AI Max+ PRO 495 w/ Radeon 8065S")
LINUX = Host("Linux", cpu="AMD Ryzen 9 7950X 16-Core Processor")


@pytest.fixture(scope="module")
def official():
    return gpuwatch.load_snapshot(OFFICIAL)


# --- 表と公式の一覧 --------------------------------------------------------------------------------------


def test_fixture_has_every_source(official):
    assert set(official) == set(gpuwatch.SNAPSHOTS)


def test_tables_agree_with_the_official_lists(official):
    """見張りの「表と照らす」が、固定した公式の一覧では候補を出さない = 公式の製品はどれも表の行に当たり、
    gfx・compute capability が一致する(わざと違うものは KNOWN_EXCEPTIONS に理由つきで書く)。"""
    assert gpuwatch.check_tables(official) == []


def test_every_amd_product_resolves_to_the_official_gfx(official):
    n = 0
    for row in official["rocm_hw_specs"] + official["therock_releases"]:
        name = row.get("name") or row["product"]
        gm = row.get("graphics_model", "")
        if f"amd:{gm or name}={row['gfx']}" in gputables.KNOWN_EXCEPTIONS:
            continue
        assert gpuwatch.amd_product_gfx(name, gm) == row["gfx"], name
        n += 1
    assert n > 100


def test_amd_names_match_exactly_one_gfx(official):
    """製品名に当たる行が複数あっても、gfx は 1 つに決まる(順番に頼った食い違いが無い)。"""
    for row in official["rocm_hw_specs"]:
        if row.get("graphics_model"):
            continue
        name = hardware._low(("AMD Instinct " if re.match(r"MI\d", row["name"]) else "AMD ") + row["name"])
        hits = {r.value[0] for r in gputables.AMD_NAME_RULES if re.search(r.key, name)}
        assert len(hits) == 1, (row["name"], hits)


def test_nvidia_names_never_give_a_wrong_cc(official):
    hit = 0
    for row in official["nvidia"]:
        got = hardware.cc_from_name(row["name"].rstrip("*"))
        if got is None:
            continue  # 分からない名前は断らない(古いドライバのときだけ使う表)
        assert f"{got[0]}.{got[1]}" == row["cc"], row
        hit += 1
    assert len(official["nvidia"]) >= 450 and hit >= 150


def test_every_gfx_in_the_official_lists_is_in_amd_arch(official):
    gfx = {r["gfx"] for r in official["therock_supported"] if not r["gfx"].endswith("x")}
    gfx |= {r["gfx"] for r in official["rocm_hw_specs"]} | set(official["torch"]["rocm_arch"])
    assert gfx <= set(hardware.AMD_ARCH)


def test_unverified_gfx_follow_therock(official):
    """Release Ready でない gfx を断る(AMD_UNVERIFIED)。TheRock の状態と食い違えば試験が落ちる。"""
    ready = {r["gfx"] for r in official["therock_supported"] if r["release"]}
    known = {r["gfx"] for r in official["therock_supported"]}
    for g in gputables.AMD_UNVERIFIED.values:
        assert g not in ready
        assert g in known or g[:-1] + "x" in known


def test_pytorch_tables_follow_the_release(official):
    torch = official["torch"]
    assert torch["tag"] == f"v{gputables.TORCH_RELEASE}"
    assert set(torch["rocm_arch"]) == set(gputables.PYTORCH_ROCM_LINUX.values)
    assert set(torch["solo"]) == set(gputables.SOLO_CC.values)
    assert torch["post_jetson"] is True  # 8.7 は CUDA 13.2 以降で 8.0 / 8.6 が動く(採らない理由は gputables)
    idx = official["torch_index"]
    assert idx["cu129"]["win_amd64"] == ["2.8.0", "2.9.0"]  # cu129 の文言(Windows は 2.8・2.9 だけ)
    assert "2.8.0" not in idx["cu128"]["linux_aarch64"]
    assert official["cuda_rn"]["12"] == 525 and official["cuda_rn"]["13"] == 580


def test_comfyui_lists_match(official):
    lists = official["comfyui"]["lists"]
    assert lists["nvidia_10_series"] == list(gputables.COMFY_FP16_PASCAL.values)
    assert lists["nvidia_16_series"] == list(gputables.COMFY_FP16_BROKEN.values)


def test_every_rule_names_known_sources_and_dates():
    for table, rules in gputables.rules_by_table().items():
        for r in rules:
            assert r.source and all(s in gputables.SOURCES for s in r.source), (table, r.key)
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.verified), (table, r.key)
            assert r.added in ("", "0.11.0"), (table, r.key)
            if r.note[0]:
                assert r.note[1], (table, r.key)
    for vs in gputables.value_sets():
        assert all(s in gputables.SOURCES for s in vs.source), vs.name
    for b in gputables.TORCH_CUDA_BUILDS:
        assert all(s in gputables.SOURCES for s in b.source), b.tag
    for s in gputables.SOURCES.values():
        assert s.url.startswith("https://") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", s.read)


# --- 直した抜けと誤り(#103) ------------------------------------------------------------------------------


def _judge(name: str, vendor: str, host: Host, **kw) -> Adapter:
    a = Adapter(name=name, vendor=vendor, **kw)
    classify_adapter(a, host, 64.0)
    return a


@pytest.mark.parametrize(
    "name,gfx",
    [
        ("AMD Radeon 8065S Graphics", "gfx1151"),
        ("AMD Radeon RX 7400", "gfx1102"),
        ("AMD Radeon RX 7300", "gfx1102"),
        ("AMD Radeon RX 7500", "gfx1102"),
        ("AMD Radeon PRO W7400", "gfx1102"),
        ("AMD Radeon RX 7400/7300/Pro W7400", "gfx1102"),  # 新しい pci.ids の lspci の名前
        ("AMD Steam Machine", "gfx1102"),
        ("AMD Radeon Pro W6900X", "gfx1030"),
    ],
)
def test_new_amd_names(name, gfx):
    assert hardware.amd_gfx(name)[0] == gfx


@pytest.mark.parametrize("pci", ["1002:7487", "1002:748b"])
def test_generic_name_uses_the_pci_table_on_linux(pci):
    """新しい pci.ids では「Navi 33 [Radeon Graphics]」の lspci の名前が「AMD Radeon Graphics」になる。
    amdkfd が無いときは型番の ID の表を引く(古い pci.ids では使え、新しい pci.ids では断る、を無くす)。"""
    with i18n.forced("ja"):
        a = _judge("AMD Radeon Graphics", "amd", LINUX, vram_gb=8.0, source="lspci", pci_id=pci)
    assert a.usable, a.reason
    g = classify_adapter(a, LINUX, 64.0)
    assert g.gfx_target == "gfx1102"
    assert any("製品名が汎用" in n for n in g.notes)


@pytest.mark.parametrize("pci,name", [("8086:e220", "Intel Graphics"), ("8086:e222", "Intel Arc Pro B65")])
def test_battlemage_g31_on_linux(pci, name):
    a = _judge(name, "intel", LINUX, vram_gb=24.0, source="lspci", pci_id=pci)
    assert a.usable, a.reason
    assert hardware.intel_xpu("Intel Arc Pro B70")[0] == "Arc B-Series (Battlemage)"


def test_data_center_gpu_max_is_linux_only():
    a = _judge("Intel Data Center GPU Max 1100", "intel", LINUX, vram_gb=48.0, source="lspci")
    assert a.usable, a.reason
    w = Host("Windows", 26100, cpu="Intel Xeon")
    a = _judge("Intel Data Center GPU Max 1100", "intel", w, vram_gb=48.0, source="wmi")
    assert not a.usable and "Linux" in a.reason


@pytest.mark.parametrize("gfx", ["gfx1170", "gfx1171", "gfx1250"])
def test_gfx_not_release_ready_is_refused(gfx):
    with i18n.forced("ja"):
        a = _judge(
            "AMD Radeon Graphics", "amd", LINUX, vram_gb=0.5, shared_gb=16.0, source="lspci", gfx_target=gfx
        )
    assert not a.usable and "Release Ready でない" in a.reason


def test_820m_follows_therock_with_a_note():
    with i18n.forced("ja"):
        g = classify_adapter(Adapter(name="AMD Radeon 820M Graphics", vendor="amd", vram_gb=0.5), WIN11, 32.0)
    assert g.gfx_target == "gfx1153"
    assert any("食い違う" in n for n in g.notes)
    assert (
        hardware.amd_gfx("AMD Radeon(TM) Graphics", cpu="AMD Ryzen AI 5 330 w/ Radeon 820M")[0] == "gfx1153"
    )


@pytest.mark.parametrize(
    "cpu,gfx",
    [
        ("AMD Ryzen AI Max+ PRO 495 w/ Radeon 8065S", "gfx1151"),
        ("AMD Ryzen AI 9 HX 475 w/ Radeon 890M", "gfx1150"),
        ("AMD Ryzen AI 9 PRO 465", "gfx1150"),
        ("AMD Ryzen AI 7 450 w/ Radeon 860M", "gfx1152"),
        ("AMD Ryzen AI 5 PRO 435", "gfx1152"),
        ("AMD Ryzen 9 270 w/ Radeon 780M Graphics", "gfx1103"),
    ],
)
def test_cpu_names_of_new_apus(cpu, gfx):
    assert hardware.amd_gfx("AMD Radeon(TM) Graphics", cpu=cpu)[0] == gfx


def test_old_driver_without_cc_does_not_promise_kepler():
    """世代が分からない古いドライバ: Kepler・Fermi は更新しても動かない、と書く(#103)。"""
    with i18n.forced("ja"):
        s = torch_support(None, "470.82", windows_driver=True)
    assert not s.usable
    assert any("Kepler・Fermi 以前は更新しても動かない" in n for n in s.notes)
    with i18n.forced("ja"):
        s = torch_support((7, 5), "470.82", windows_driver=True)
    assert any("ドライバを更新すれば GPU で動く)" in n and "Kepler" not in n for n in s.notes)


def test_cu129_text_says_windows_is_only_28_29():
    b = next(b for b in gputables.TORCH_CUDA_BUILDS if b.tag == "cu129")
    assert "2.8.0・2.9.0 だけ" in b.note[0] and b.torch == "2.8〜2.13"


def test_windows_intel_graphics_with_an_arc_device_id_is_used():
    """1 チャンネルのメモリの Core Ultra は Arc Graphics を「Intel(R) Graphics」と出す
    (Intel support 000102144)。型番 ID が Arc の内蔵 GPU なら使い、1 チャンネルは遅いと書く(#103)。
    Arc でない ID は今までどおり CPU。"""
    with i18n.forced("ja"):
        a = _judge("Intel(R) Graphics", "intel", WIN11, source="wmi", pci_id="8086:7d55", vram_gb=0.1)
        g = classify_adapter(a, WIN11, 32.0)
    assert a.usable and g.backend == "xpu" and g.memory == "shared"
    assert any("1 チャンネル" in n and "000102144" in n for n in g.notes)
    b = _judge("Intel(R) Graphics", "intel", WIN11, source="wmi", pci_id="8086:7d67", vram_gb=0.1)
    assert not b.usable


@pytest.mark.parametrize(
    "name,gfx",
    [
        ("AMD Oberon/Navi12Lite", None),  # ゲーム機の石(Navi 12 ではない)
        ("AMD Ariel/Navi10Lite", None),
        ("AMD Radeon 820M / 840M Graphics", "gfx1153"),  # 新しい pci.ids の 1902。820M の決まりが先
        ("AMD Radeon 840M / 860M Graphics", "gfx1152"),
    ],
)
def test_pci_names_that_need_care(name, gfx):
    got = hardware.amd_gfx(name)
    assert (got[0] if got else None) == gfx
