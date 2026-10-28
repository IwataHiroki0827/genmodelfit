# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""組み込み用の公開 API の追加(#96)。AI生成キットが内部の関数で行っていた 6 つを、公開の関数で同じ答えにする。

1. AMD・Intel の GPU を system_from / judge_model で判定する(hardware.classify_adapter の代わり)
2. GPU 名の表(hardware.amd_gfx / intel_xpu の代わりの gmf.gpu_family)
3. 同梱の一覧と同梱の実測だけを読む・出どころを指定する(load_db の user_measurements / source)
4. 絞り込みの判定(ModelEntry.license_unknown / is_tool)
5. 日時(Database.generated_time / DataStatus.generated_time)と同梱物の確認(gmf.self_check)
6. safetensors の見出し(gmf.parse_safetensors_header / HeaderError)

それぞれ、アプリの今の呼び方(内部の関数)を写したものと、公開の関数が同じ答えを返すことを確かめる。
ネット・本物のハードウェア(nvidia-smi・PowerShell・lspci・レジストリの画面の部品)には触らない。
"""

from __future__ import annotations

import json
import platform
import struct

import pytest

import genmodelfit as gmf
from genmodelfit import appapi, hardware, headers, signing
from genmodelfit.db import COMMUNITY_FILE, bundled_text, parse_db, parse_measurements
from genmodelfit.models import Adapter, SystemInfo


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def bundled():
    """アプリの読み方(同梱の db.json だけ)。"""
    return parse_db(json.loads(bundled_text("db.json")), "bundled")


# --- 1. AMD・Intel の GPU を system_from で判定する ---------------------------------------------------

RYZEN = "AMD Ryzen 7 7700X 8-Core Processor"
# (名前, 会社, VRAM GB(None は分からない), CPU の名前)。使える・断る(世代・小さな内蔵 GPU・表に無い)の両方
OTHER_GPUS = [
    ("AMD Radeon RX 7800 XT", "amd", 16.0, RYZEN),
    ("AMD Radeon RX 7900 XTX", "amd", 24.0, RYZEN),
    ("AMD Radeon RX 9070 XT", "amd", 16.0, RYZEN),
    ("AMD Radeon RX 6600", "amd", 8.0, RYZEN),
    ("AMD Radeon RX 6700 XT", "amd", 12.0, RYZEN),
    ("AMD Radeon RX 5700 XT", "amd", 8.0, RYZEN),  # RDNA 1: 断る
    ("AMD Radeon 780M Graphics", "amd", 0.5, "AMD Ryzen 7 8700G w/ Radeon 780M Graphics"),
    ("AMD Radeon(TM) Graphics", "amd", 0.5, "AMD Ryzen AI Max+ 395 w/ Radeon 8060S"),  # CPU の名前で決める
    ("AMD Radeon(TM) Graphics", "amd", 0.5, "AMD Ryzen 9 7945HX with Radeon Graphics"),  # 610M: 断る
    ("AMD Radeon(TM) Graphics", "amd", 0.5, RYZEN),  # 世代が決まらない: 断る
    ("AMD Radeon RX 9999 XT", "amd", 16.0, RYZEN),  # 表に無い: 断る
    ("Intel(R) Arc(TM) B580 Graphics", "intel", 12.0, ""),
    ("Intel(R) Arc(TM) B580 Graphics", "intel", None, ""),  # VRAM は名前の公称
    ("Intel(R) Arc(TM) A770 Graphics", "intel", 16.0, ""),
    ("Intel(R) Arc(TM) A380 Graphics", "intel", None, ""),
    ("Intel(R) Arc(TM) 140V GPU (16GB)", "intel", None, ""),  # Lunar Lake の内蔵 GPU
    ("Intel(R) Arc(TM) Graphics", "intel", 0.1, ""),  # Meteor Lake の内蔵 GPU
    ("Intel(R) UHD Graphics 770", "intel", 0.1, ""),  # XPU の対象でない: 断る
    ("Intel(R) Iris(R) Xe Graphics", "intel", 0.1, ""),
    ("Intel(R) Arc(TM) Z999 Graphics", "intel", 16.0, ""),  # 表に無い Arc: 断る
    ("Some GPU", "unknown", 8.0, ""),
]
# (system_from の windows, windows_build) → アプリが作っていた Host
HOSTS = [(True, 22000), (True, 26100), (True, 19045)]
RAM = 64.0


def _app_classify(name, vendor, vram, cpu, build, ram=RAM, system="Windows"):
    """AI生成キットの今の呼び方(内部の hardware.classify_adapter)を写したもの。"""
    a = Adapter(name=name, vendor=vendor, vram_gb=vram, vram_source="registry", source="wmi")
    g = hardware.classify_adapter(a, hardware.Host(system, windows_build=build, cpu=cpu), ram)
    return a, g


def _public(name, vendor, vram, cpu, **host):
    spec = {"name": name, "vendor": vendor}
    if vram is not None:
        spec["vram_gb"] = vram
    return gmf.system_from([spec], ram_gb=RAM, cpu=cpu, **host)


@pytest.mark.parametrize(("windows", "build"), HOSTS)
@pytest.mark.parametrize(("name", "vendor", "vram", "cpu"), OTHER_GPUS)
def test_system_from_classifies_like_the_app(name, vendor, vram, cpu, windows, build):
    """system_from の判定 = アプリが classify_adapter で作った Gpu と理由(全部の欄が同じ)。"""
    a, want = _app_classify(name, vendor, vram, cpu, build)
    s = _public(name, vendor, vram, cpu, windows=windows, windows_build=build)
    assert s.best_gpu == want
    assert s.gpus == ([want] if want is not None else [])
    (got,) = s.other_gpus
    for k in ("usable", "reason", "integrated", "needs"):
        assert getattr(got, k) == getattr(a, k), k
    assert s.cpu == cpu
    if want is None:
        assert got.usable is False and got.reason  # 断った理由が分かる
        assert any(got.reason in n for n in s.notes)


@pytest.mark.parametrize(("name", "vendor", "vram", "cpu"), OTHER_GPUS)
def test_system_from_linux_classifies_like_the_app(name, vendor, vram, cpu, monkeypatch):
    """windows=False は Linux の対応表(この PC が Windows でも)。"""
    monkeypatch.setattr(appapi.platform, "system", lambda: "Windows")
    a, want = _app_classify(name, vendor, vram, cpu, None, system="Linux")
    s = _public(name, vendor, vram, cpu, windows=False)
    assert s.best_gpu == want
    assert s.other_gpus[0].reason == a.reason


def test_system_from_judges_like_the_app(bundled):
    """判定(judge_model)も、アプリが classify_adapter の Gpu で SystemInfo を作ったときと同じ。"""
    for name, vendor, vram, cpu in OTHER_GPUS[:4] + OTHER_GPUS[11:13]:
        _a, g = _app_classify(name, vendor, vram, cpu, 22000, ram=1024.0)
        s = gmf.system_from(
            [{"name": name, "vendor": vendor, "vram_gb": vram}],
            ram_gb=1024,
            gpu_usable=True,  # NVIDIA 以外には効かない(対応表で決める)
            windows=True,
            windows_build=22000,
            cpu=cpu,
        )
        app = SystemInfo(
            os=s.os,
            cpu=cpu,
            cpu_cores=s.cpu_cores,
            ram_total_gb=1024.0,
            ram_available_gb=1024.0,
            gpus=[g],
            disk_free_gb=s.disk_free_gb,
        )
        for mid in ("image_sdxl_simple", "image_z_image_turbo", "video_wan2_2_5B_ti2v"):
            mine = gmf.judge_model(mid, system_info=s, db=bundled)
            theirs = gmf.judge_model(mid, system_info=app, db=bundled)
            assert mine.estimate == theirs.estimate, (name, mid)
            assert mine.to_dict()["fit_level"] == theirs.to_dict()["fit_level"]
        # judge_model の GPU の引数でも同じ
        one = gmf.judge_model(
            "image_sdxl_simple",
            gpu_name=name,
            gpu_vendor=vendor,
            vram_gb=vram or 0,
            ram_gb=1024,
            windows=True,
            windows_build=22000,
            cpu=cpu,
            db=bundled,
        )
        assert one.estimate == gmf.judge_model("image_sdxl_simple", system_info=app, db=bundled).estimate


def test_system_from_windows_10_says_what_is_needed():
    s = _public("AMD Radeon RX 7800 XT", "amd", 16.0, RYZEN, windows=True, windows_build=19045)
    assert s.best_gpu is None and s.vram_gb == 0
    assert s.other_gpus[0].needs == "Windows 11" and "Windows 11" in s.other_gpus[0].reason


def test_system_from_unknown_windows_build_is_not_used(monkeypatch):
    """Windows の版が分からない(この PC が Windows でなく windows_build も無い)なら AMD・Intel は使わない。"""
    monkeypatch.setattr(appapi.platform, "system", lambda: "Linux")
    s = _public("Intel(R) Arc(TM) B580 Graphics", "intel", 12.0, "", windows=True)
    assert s.best_gpu is None and s.other_gpus[0].usable is False and s.other_gpus[0].needs is None


def test_system_from_nvidia_wins_over_other_vendors():
    """使える NVIDIA があれば、AMD・Intel は判定に使わない(検出と同じ。理由は notes)。"""
    s = gmf.system_from(
        [
            {"name": "AMD Radeon RX 7900 XTX", "vendor": "amd", "vram_gb": 24},
            {"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12, "driver": "560.94"},
        ],
        ram_gb=32,
        windows=True,
        windows_build=22631,
        cpu=RYZEN,
    )
    assert s.best_gpu.name == "NVIDIA GeForce RTX 3060" and s.vram_gb == 12
    assert [g.backend for g in s.gpus] == ["cuda"]
    assert any("AMD Radeon RX 7900 XTX" in n and "NVIDIA" in n for n in s.notes)


def test_system_from_gpu_usable_false_keeps_everything_on_cpu():
    s = _public("AMD Radeon RX 7800 XT", "amd", 16.0, RYZEN, windows=True, windows_build=22631)
    assert s.vram_gb == 16
    off = gmf.system_from(
        [{"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vram_gb": 16}],
        ram_gb=RAM,
        gpu_usable=False,
        windows=True,
        windows_build=22631,
    )
    assert off.best_gpu is None and off.vram_gb == 0 and off.other_gpus == []


def test_system_from_generic_name_uses_pci_id():
    """汎用の名前は pci_id(型番の ID)で世代を引く(検出と同じ)。"""
    s = gmf.system_from(
        [{"name": "AMD Radeon Graphics", "vendor": "amd", "vram_gb": 16, "pci_id": "1002:744C"}],
        ram_gb=RAM,
        windows=True,
        windows_build=22631,
        cpu=RYZEN,
    )
    assert s.best_gpu is not None and s.best_gpu.gfx_target == "gfx1100" and s.best_gpu.pci_id == "1002:744c"


def test_system_from_does_not_probe_hardware(monkeypatch):
    """渡した GPU だけを使う(画面の部品の一覧・nvidia-smi・WSL の Windows 側を読まない)。"""

    def boom(*a, **k):
        raise AssertionError("呼ばない")

    monkeypatch.setattr(hardware, "detect_other_gpus", boom)
    monkeypatch.setattr(hardware, "detect_gpus", boom)
    monkeypatch.setattr(hardware, "host_info", boom)
    s = _public("Intel(R) Arc(TM) B580 Graphics", "intel", None, "", windows=True, windows_build=26100)
    assert s.best_gpu.backend == "xpu" and s.best_gpu.vram_total_gb == 12.0


@pytest.mark.parametrize(
    ("spec", "kw"),
    [
        ({"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vram_gb": -1}, {}),
        ({"name": "AMD Radeon RX 7800 XT", "vendor": 3, "vram_gb": 16}, {}),
        ({"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "pci_id": "744c"}, {}),
        ({"name": "AMD Radeon RX 7800 XT", "vendor": "amd"}, {"windows": True, "windows_build": 0}),
        ({"name": "AMD Radeon RX 7800 XT", "vendor": "amd"}, {"windows": True, "windows_build": True}),
        ({"name": "AMD Radeon RX 7800 XT", "vendor": "amd"}, {"windows": False, "windows_build": 22000}),
        ({"name": "NVIDIA GeForce RTX 3060"}, {}),  # NVIDIA の VRAM は省けない
    ],
)
def test_system_from_rejects_bad_other_vendor_input(spec, kw):
    with pytest.raises(ValueError):
        gmf.system_from([spec], ram_gb=16, **kw)


def test_system_from_rejects_a_non_string_cpu():
    with pytest.raises(TypeError):
        gmf.system_from([], ram_gb=16, cpu=7)


def test_judge_model_gpu_vendor_rules(bundled):
    with pytest.raises(ValueError):
        gmf.judge_model("image_sdxl_simple", cpu_only=True, gpu_vendor="amd", db=bundled)
    with pytest.raises(ValueError):
        gmf.judge_model("image_sdxl_simple", gpu_vendor="amd", ram_gb=16, db=bundled)


def test_privacy_no_pc_identity_in_other_gpus():
    """渡したもの以外(PC 名・ユーザ名・差し場所)は出ない。"""
    s = _public("AMD Radeon RX 7800 XT", "amd", 16.0, RYZEN, windows=True, windows_build=22631)
    text = json.dumps(s.to_dict(), ensure_ascii=False)
    for secret in (platform.node(),):
        if secret:
            assert secret not in text
    assert set(s.other_gpus[0].__dict__) >= {"name", "vendor", "reason"}


# --- 2. GPU 名の表 -----------------------------------------------------------------------------------

FAMILY_NAMES = [
    *[(n, v, RYZEN) for n, v, _vram, _cpu in OTHER_GPUS if v in ("amd", "intel")],
    *[(n, "amd", cpu) for n, _v, _vram, cpu in OTHER_GPUS if n == "AMD Radeon(TM) Graphics"],
    # 型番の ID の表の名前(検出が名前の表で引ける形にした名前)
    *[(f"AMD {n}", "amd", "") for (ven, _d), n in hardware.PCI_DEVICE_NAMES.items() if ven == "1002"],
    *[(f"Intel {n}", "intel", "") for (ven, _d), n in hardware.PCI_DEVICE_NAMES.items() if ven == "8086"],
    ("AMD Radeon RX 6800M", "amd", ""),
    ("AMD Radeon 890M", "amd", ""),
    ("AMD Radeon Graphics", "amd", "AMD Ryzen AI 9 HX 370 w/ Radeon 890M"),
    ("Intel(R) Arc(TM) Pro B60 Graphics", "intel", ""),
    ("Intel(R) Arc(TM) A770M Graphics", "intel", ""),
    ("Intel(R) Data Center GPU Max 1550", "intel", ""),
    ("", "amd", ""),
]


@pytest.mark.parametrize(("name", "vendor", "cpu"), FAMILY_NAMES)
def test_gpu_family_matches_the_internal_tables(name, vendor, cpu):
    """gpu_family = アプリが使っていた hardware.amd_gfx / intel_xpu(表に無ければ両方 None)。"""
    got = gmf.gpu_family(name, vendor, cpu=cpu)
    if vendor == "amd":
        want = hardware.amd_gfx(name, cpu)
        assert (got is None) == (want is None)
        if want:
            assert (got["arch"], got["integrated"]) == want
            assert got["backend"] == "rocm" and got["vram_floor_gb"] is None
            assert got["family"] == hardware.AMD_ARCH.get(want[0], "?")
    else:
        want = hardware.intel_xpu(name)
        assert (got is None) == (want is None)
        if want:
            assert (got["family"], got["integrated"], got["vram_floor_gb"]) == want
            assert got["backend"] == "xpu" and got["arch"] is None
    if got:
        assert set(got) == {"vendor", "backend", "arch", "family", "integrated", "vram_floor_gb"}
        assert got["vendor"] == vendor


def test_gpu_family_examples_and_errors():
    assert gmf.gpu_family("AMD Radeon RX 7800 XT", "AMD")["arch"] == "gfx1101"
    b580 = gmf.gpu_family("Intel(R) Arc(TM) B580 Graphics", "intel")
    assert b580["backend"] == "xpu" and b580["vram_floor_gb"] == 12.0 and not b580["integrated"]
    assert gmf.gpu_family("Intel(R) UHD Graphics 770", "intel") is None
    for bad in ("nvidia", "apple", "", None):
        with pytest.raises(ValueError):
            gmf.gpu_family("NVIDIA GeForce RTX 3060", bad)
    with pytest.raises(TypeError):
        gmf.gpu_family(None, "amd")


# --- 3. 同梱の一覧と同梱の実測だけ ---------------------------------------------------------------------


def _write_user_measurement(tmp_path):
    """この PC の bench の実測(キャッシュ)を 1 件置く。"""
    m = json.loads(bundled_text("measurements.json"))["measurements"][0]
    m = {**m, "model_id": "image_sdxl_simple", "gpu_name": "NVIDIA GeForce RTX 9999 (this PC)"}
    d = tmp_path / "cache" / "measurements"
    d.mkdir(parents=True)
    (d / "bench.json").write_text(json.dumps({"measurements": [m]}), encoding="utf-8")


def test_load_db_without_user_measurements_is_the_bundled_data(tmp_path, bundled):
    _write_user_measurement(tmp_path)
    mixed = gmf.load_db(use_cache=False)
    only = gmf.load_db(use_cache=False, user_measurements=False)
    assert len(mixed.measurements) == len(only.measurements) + 1
    assert all("this PC" not in (m.gpu_name or "") for m in only.measurements)
    # 中身は同梱のファイルそのもの(アプリの読み方: db.json と、同梱の実測 2 つ)
    want = parse_measurements(json.loads(bundled_text("measurements.json")))
    want += parse_measurements(json.loads(bundled_text(COMMUNITY_FILE)))
    assert only.measurements == want
    assert only.models == bundled.models and only.origin == "bundled"


def test_load_db_source_release(tmp_path):
    p = tmp_path / "db.json"
    p.write_text(bundled_text("db.json"), encoding="utf-8")
    db = gmf.load_db(p, source="release", user_measurements=False)
    assert db.origin == "release" and gmf.data_status(db).source == "release"
    assert gmf.load_db(p).origin == "path"
    with pytest.raises(ValueError):
        gmf.load_db(source="release")
    with pytest.raises(ValueError):
        gmf.load_db(p, source="bundled")


# --- 4. 絞り込みの判定 ---------------------------------------------------------------------------------


def test_entry_filters_match_the_internal_functions(bundled):
    from genmodelfit.recommend import license_unknown
    from genmodelfit.tasks import is_tool

    assert any(m.license_unknown for m in bundled.models) and any(m.is_tool for m in bundled.models)
    for m in bundled.models:
        assert m.license_unknown is license_unknown(m), m.id
        assert m.is_tool is is_tool(m), m.id
    assert "license_unknown" not in bundled.models[0].to_dict()  # 一覧の形は変えない


def test_entry_filters_are_the_default_filters_of_fit_list(bundled):
    s = gmf.system_from([], ram_gb=32)
    shown = {
        f.model.id
        for f in gmf.fit_list(fit="all", content="all", all_workflows=True, system_info=s, db=bundled)
    }
    assert shown == {m.id for m in bundled.models if not m.is_tool and not m.license_unknown}


# --- 5. 日時と同梱物 -----------------------------------------------------------------------------------


def test_generated_time_is_parse_time(bundled):
    from genmodelfit.remote import parse_time

    assert bundled.generated_time == parse_time(bundled.generated_at)
    assert gmf.data_status(bundled).generated_time == bundled.generated_time
    newer = gmf.Database(generated_at="2099-01-01T00:00:00.1234567Z")
    assert newer.generated_time > bundled.generated_time  # 取り込んだ一覧が同梱より新しいか
    for bad in (None, "", "2026-10-21T06:20:18", "yesterday"):
        assert gmf.Database(generated_at=bad).generated_time is None


def test_self_check_ok_and_matches_the_internal_keys(capfd):
    r = gmf.self_check()
    assert r["ok"] is True and r["version"] == gmf.__version__
    assert set(r["checks"]) == {"db", "measurements", "calibration", "cryptography", "public_keys"}
    assert r["checks"]["public_keys"]["detail"].startswith(f"{len(signing.trusted_public_keys())} ")
    assert capfd.readouterr() == ("", "")


def test_self_check_reports_missing_keys_without_raising(monkeypatch):
    monkeypatch.setattr(signing, "embedded_public_keys_pem", lambda: [])
    r = gmf.self_check()
    assert r["ok"] is False and r["checks"]["public_keys"]["ok"] is False
    assert "SigningError" in r["checks"]["public_keys"]["detail"]
    assert r["checks"]["db"]["ok"] is True


# --- 6. safetensors の見出し ---------------------------------------------------------------------------


def test_parse_safetensors_header_is_public():
    assert gmf.parse_safetensors_header is headers.parse_safetensors_header
    assert gmf.HeaderError is headers.HeaderError and issubclass(gmf.HeaderError, ValueError)
    body = json.dumps(
        {
            "__metadata__": {"format": "pt"},
            "a": {"dtype": "BF16", "shape": [1024, 1024], "data_offsets": [0, 2097152]},
            "b": {"dtype": "F32", "shape": [10], "data_offsets": [2097152, 2097192]},
        }
    ).encode()
    blob = struct.pack("<Q", len(body)) + body  # ファイルの先頭: 8 バイトの長さ + 見出し
    (n,) = struct.unpack("<Q", blob[:8])
    info = gmf.parse_safetensors_header(blob[8 : 8 + n])
    assert isinstance(info, gmf.HeaderInfo)
    assert info.precision == "bf16" and info.params == 1024 * 1024 + 10
    for bad in (b"not json", b"[]", json.dumps({"a": {"dtype": "F16", "shape": [-1]}}).encode()):
        with pytest.raises(gmf.HeaderError):
            gmf.parse_safetensors_header(bad)


def test_new_names_are_public():
    for name in (
        "gpu_family",
        "self_check",
        "parse_safetensors_header",
        "HeaderError",
        "HeaderInfo",
    ):
        assert name in gmf.__all__ and hasattr(gmf, name), name
