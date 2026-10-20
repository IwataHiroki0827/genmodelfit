"""利用者が出す実測(#34): 提出用の JSON・検査・まとめ・根拠の段階・校正への取り込み・CLI の形。

tests/fixtures/contrib の 4 件は試験用に作った架空の値(note に SYNTHETIC TEST FIXTURE と書いた)。
data/contrib には何も足さない(本物の提出だけを置く)。
"""

from __future__ import annotations

import copy
import json
import shutil
from datetime import date
from pathlib import Path

import pytest

from genmodelfit import cli, contrib
from genmodelfit.calibrate import build_check_rows, build_rows, calibrate
from genmodelfit.db import COMMUNITY_FILE, Database, bundled_text, load, parse_measurements
from genmodelfit.estimate import estimate, pick_measurement, usable
from genmodelfit.models import BASIS_DETAILS, Gpu, Measurement, Settings, SystemInfo
from genmodelfit.recommend import score_components

FIX = Path(__file__).parent / "fixtures" / "contrib"
TODAY = date(2026, 10, 21)
GOOD = sorted(FIX.rglob("*.json"))
SDXL_3070 = [p for p in GOOD if p.parent.name == "rtx-3070"]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def db() -> Database:
    return load(use_cache=False)


def good() -> dict:
    return json.loads(SDXL_3070[0].read_text(encoding="utf-8"))


def pc(vram: float, name: str = "NVIDIA GeForce RTX 3070") -> SystemInfo:
    g = Gpu(name=name, vendor="nvidia", vram_total_gb=vram, backend="cuda")
    return SystemInfo(os="test", cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g])


# --- 提出用の JSON --------------------------------------------------------------------------------


def test_fixtures_are_synthetic_and_valid(db):
    assert len(GOOD) == 4
    for p in GOOD:
        assert "SYNTHETIC TEST FIXTURE" in json.loads(p.read_text(encoding="utf-8"))["measurement"]["note"]
    reports = contrib.validate_files(GOOD, db, today=TODAY)
    assert all(r.ok for r in reports), [r.errors for r in reports]
    # 名前は <GPU>/<日付>-<内容の sha256 の先頭 8 桁>.json
    for p, r in zip(GOOD, reports, strict=True):
        assert r.suggested_path.endswith(f"{p.parent.name}/{p.name}")
    assert not (Path(__file__).parent.parent / "data" / "contrib").exists() or not any(
        "SYNTHETIC" in p.read_text(encoding="utf-8")
        for p in (Path(__file__).parent.parent / "data" / "contrib").rglob("*.json")
    )


def test_build_submission_fields_and_hash():
    m = Measurement(
        model_id="image_sdxl_simple",
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=8.0,
        peak_vram_gb=6.9,
        settings=Settings(width=832, height=832, steps=20),
        seconds=41.2,
        driver="566.03",
        comfyui_version="0.40.0",
        offload="dynamic",
        source="genmodelfit bench, my own words",
        date="2026-10-20",
        note="x",
        vram_metric="torch_reserved",
    )
    sub = contrib.build_submission(m)  # PC の情報なし(ComfyUI が別の PC)
    assert set(sub) == contrib.TOP_KEYS
    assert sub["system"]["os_family"] is None and sub["system"]["ram_total_gb"] is None
    assert sub["system"]["gpu"]["arch"] == "ampere" and sub["system"]["gpu"]["compute_capability"] == "8.6"
    assert "source" not in json.dumps(sub)  # 自由に書ける source は入れない
    assert sub["measurement"]["settings"] == {"width": 832, "height": 832, "steps": 20}
    # hash は内容だけで決まる(キーの順・字下げによらない)
    shuffled = json.loads(json.dumps(sub, sort_keys=True))
    assert contrib.content_hash(shuffled) == contrib.content_hash(sub)
    assert (
        contrib.suggested_path(sub)
        == f"data/contrib/rtx-3070/2026-10-20-{contrib.content_hash(sub)[:8]}.json"
    )


def test_build_submission_rounds_ram_and_keeps_only_os_family():
    s = SystemInfo(os="Windows 11", cpu="SECRET CPU", cpu_cores=8, ram_total_gb=31.87, ram_available_gb=3)
    m = Measurement(model_id="x", gpu_name="NVIDIA GeForce RTX 4090", gpu_vram_gb=23.99, peak_vram_gb=1.0)
    sub = contrib.build_submission(m, s)
    assert sub["system"]["os_family"] == "windows" and sub["system"]["ram_total_gb"] == 32
    assert "SECRET CPU" not in json.dumps(sub) and "Windows 11" not in json.dumps(sub)
    s.overrides = ["ram"]  # --ram で上書きした値は、その PC の値ではないので入れない
    assert contrib.build_submission(m, s)["system"]["ram_total_gb"] is None


def test_scrub_text_removes_identifying_strings(monkeypatch):
    monkeypatch.setattr(contrib, "_identity_words", lambda: ["MYHOST-1234", "alice42"])
    text = (
        r"失敗: FileNotFoundError: C:\Users\alice42\ComfyUI\models\x.safetensors が無い。"
        "/home/alice42/comfy/x.py:12 と \\\\nas01\\share\\m を見た。MYHOST-1234 で alice42 が測った。"
        " GPU-0f3e1a2b-1111-2222-3333-444455556666 0000:01:00.0 http://192.168.1.20:8188/prompt"
        " me@example.com 2026-10-21T10:11:12Z 10:11:12 DESKTOP-AB12CDE"
    )
    out = contrib.scrub_text(text)
    for bad in (
        "alice42",
        "MYHOST",
        "Users",
        "/home",
        "nas01",
        "0f3e1a2b",
        "0000:01",
        "192.168",
        "example.com",
        "T10:11",
        "10:11:12",
        "DESKTOP-AB12CDE",
    ):
        assert bad not in out, (bad, out)
    assert not contrib.privacy_findings({"note": out}), out
    assert "FileNotFoundError" in out  # 理由の文そのものは残す


# --- 個人の情報(番兵の値): bench --submission ----------------------------------------------------

from test_privacy_pc import SENT_HOST, SENT_UUID, outbound, sentinel_pc  # noqa: E402, F401


def test_bench_submission_has_no_pc_identity(sentinel_pc, outbound, tmp_path, capsys, db):  # noqa: F811
    from test_bench import FakeComfy

    forbidden, home = sentinel_pc
    secret_note = (
        f"{SENT_HOST} で {forbidden[4]} が {home}\\ComfyUI を使い {SENT_UUID} 10.0.0.7 a@b.example 12:34:56"
    )
    out_dir = tmp_path / "fork" / "data" / "contrib"
    out_dir.mkdir(parents=True)
    with FakeComfy() as fake:
        args = [
            "bench",
            "image_sdxl_simple",
            "--url",
            fake.url,
            "--submission",
            str(out_dir),
            "--note",
            secret_note,
        ]
        assert cli.main([*args, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    path = Path(data["submission"]["saved_to"])
    assert data["submission"]["path_in_repo"].startswith("data/contrib/rtx-3070/")
    assert path.parent.name == "rtx-3070" and path.parent.parent == out_dir
    text = path.read_text(encoding="utf-8")
    for word in [*forbidden, SENT_UUID, "10.0.0.7", "a@b.example", "12:34:56", "ComfyUI\\"]:
        assert word not in text, word
    sub = json.loads(text)
    assert sub["system"]["ram_total_gb"] == 31  # 番兵の 31.4159 を丸めた値だけ
    assert sub["system"]["gpu"]["name"] == "NVIDIA GeForce RTX 3070"  # 測った GPU(ComfyUI が返した名前)
    assert sub["system"]["gpu"]["arch"] == "ampere"
    r = contrib.validate_files([path], db)[0]
    # 偽の ComfyUI はすぐ終わる(0 秒)ので、時間の値だけが範囲の外になる。名前・個人の情報などは通る
    assert sub["measurement"]["seconds"] == 0.0
    assert [e["code"] for e in r.errors] == ["value"] and "seconds" in r.errors[0]["message"], r.errors
    checks = contrib.privacy_checklist(sub, text)
    assert all(c["ok"] is not False for c in checks), checks


# --- 検査 -----------------------------------------------------------------------------------------


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    if value is _DEL:
        del d[keys[-1]]
    else:
        d[keys[-1]] = value


_DEL = object()

BAD_CASES = [
    ("system.hostname", "DESKTOP-1", "unknown_field"),
    ("measurement.serial", "ABC", "unknown_field"),
    ("measurement.note", r"C:\Users\bob\ComfyUI", "privacy"),
    ("measurement.note", "/home/bob/ComfyUI", "privacy"),
    ("measurement.note", "mail me: bob@example.org", "privacy"),
    ("measurement.note", "GPU-12345678-aaaa-bbbb-cccc-1234567890ab", "privacy"),
    ("measurement.note", "server 192.168.0.12", "privacy"),
    ("measurement.note", "on DESKTOP-ABC1234", "privacy"),
    ("measurement.note", "pci 0000:01:00.0", "privacy"),
    ("measurement.note", "http://example.org/x", "privacy"),
    ("date", "2026-10-20T10:11:12Z", "date"),
    ("date", "2027-01-01", "date"),
    ("contrib_schema", 2, "schema"),
    ("measurement.peak_vram_gb", 9.5, "value"),
    ("measurement.peak_vram_gb", -1, "value"),
    ("measurement.seconds", 0, "value"),
    ("measurement.seconds", _DEL, "value"),
    ("measurement.outcome", "crashed", "value"),
    ("measurement.peak_ram_gb", 64.0, "value"),
    ("measurement.offload", "magic", "value"),
    ("measurement.vram_metric", "nvidia_smi_total", "value"),
    ("measurement.settings.width", 10, "settings"),
    ("measurement.settings.steps", 2.5, "settings"),
    ("measurement.settings.prompt", "a cat", "unknown_field"),
    ("system.gpu.compute_capability", "7.5", "gpu_inconsistent"),
    ("system.gpu.arch", "turing", "gpu_inconsistent"),
    ("system.gpu.vram_gb", 0, "value"),
    ("system.gpu.driver", "latest", "value"),
    ("system.os_family", "Windows 11 26100", "value"),
    ("system.ram_total_gb", -4, "value"),
    ("measurement.model_id", "", "value"),
    ("kind", "other", "schema"),
    ("tool.name", "llmfit", "schema"),
]


@pytest.mark.parametrize(("key", "value", "code"), BAD_CASES, ids=[f"{k}={v!r}" for k, v, _ in BAD_CASES])
def test_validate_catches_bad_value(key, value, code, db):
    d = good()
    _set(d, key, value)
    r = contrib.validate_submission(d, db, today=TODAY)
    assert not r.ok
    assert code in {e["code"] for e in r.errors}, r.errors


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("measurement.model_id", "image_not_in_db", "unknown_model"),
        ("system.gpu.name", "AMD Radeon RX 7900 XTX", "unknown_gpu"),
        ("measurement.standard", False, "nonstandard"),
        ("system.ram_total_gb", 31.4, "ram_not_rounded"),
        ("system.gpu.driver", None, "driver_missing"),
        ("measurement.settings.frames", 49, "settings_modality"),
    ],
)
def test_validate_warnings_do_not_fail(key, value, code, db):
    d = good()
    _set(d, key, value)
    if key == "system.gpu.name":  # 表に無い GPU は世代・compute の照合もしない
        d["system"]["gpu"]["arch"] = d["system"]["gpu"]["compute_capability"] = None
    r = contrib.validate_submission(d, db, today=TODAY)
    assert r.ok, r.errors
    assert code in {w["code"] for w in r.warnings}


def test_validate_filename_duplicates_and_unreadable(tmp_path, db):
    d = good()
    good_name = contrib.suggested_path(d).split("/", 2)[2]
    ok = tmp_path / good_name
    ok.parent.mkdir(parents=True)
    ok.write_text(contrib.dumps(d), encoding="utf-8")
    dup = tmp_path / "copy" / good_name.split("/")[1]  # 同じ内容(字下げが違っても同じ hash)
    dup.parent.mkdir()
    dup.write_text(json.dumps(d, sort_keys=True), encoding="utf-8")
    wrong_hash = ok.parent / "2026-10-20-00000000.json"
    wrong_hash.write_text(contrib.dumps(d), encoding="utf-8")
    broken = ok.parent / "2026-10-20-11111111.json"
    broken.write_text("{not json", encoding="utf-8")
    reports = contrib.validate_files([ok, dup, wrong_hash, broken], db, today=TODAY)
    codes = [{e["code"] for e in r.errors} for r in reports]
    assert codes[0] == set()
    assert "duplicate" in codes[1] and "filename" in codes[1]  # フォルダ名も GPU と違う
    assert "filename" in codes[2] and "duplicate" in codes[2]
    assert codes[3] == {"read"}
    assert reports[1].errors[0]["message"].endswith("と同じ内容です") or any(
        e["code"] == "duplicate" for e in reports[1].errors
    )


def test_privacy_scan_reads_raw_file_too(db):
    d = good()
    raw = contrib.dumps(d).replace('"ok"', '"ok"', 1) + "\n// C:\\Users\\bob\n"
    r = contrib.validate_submission(d, db, raw=raw, today=TODAY)
    assert any(e["code"] == "privacy" for e in r.errors)


# --- まとめる -------------------------------------------------------------------------------------


def _sub(
    peak: float, *, gpu: str = "NVIDIA GeForce RTX 3070", outcome: str = "ok", standard: bool = True, w=832
):
    d = good()
    d["system"]["gpu"]["name"] = gpu
    d["measurement"].update(peak_vram_gb=peak, outcome=outcome, standard=standard)
    d["measurement"]["settings"]["width"] = w
    if outcome == "oom":
        d["measurement"]["offload"] = "oom"
    return contrib.Submission(path="-", sha256=contrib.content_hash(d), data=d)


def test_robust_filter():
    assert contrib.robust_filter([5.0, 9.0]) == ([0, 1], [], 2.0)  # 2 件では外れ値を見分けない
    keep, drop, _ = contrib.robust_filter([6.9, 7.0, 7.02, 7.05, 3.1])
    assert drop == [4] and keep == [0, 1, 2, 3]
    keep, drop, _ = contrib.robust_filter([7.0, 7.0, 7.0, 7.1])  # MAD 0 でも小さな差は外さない(下限)
    assert drop == []


def test_aggregate_outliers_counts_and_determinism():
    subs = [_sub(6.9), _sub(7.02), _sub(7.05), _sub(3.1), _sub(6.0, standard=False), _sub(7.9, outcome="oom")]
    subs += [_sub(5.0, gpu="NVIDIA GeForce RTX 4060")]  # 別の GPU は別の組
    subs += [_sub(7.3, outcome="oom", w=1024), _sub(7.4, outcome="oom", w=1024)]  # OOM だけの組
    rows = contrib.aggregate(subs)
    assert rows == contrib.aggregate(list(reversed(subs)))  # 並びによらない
    assert json.dumps(rows) == json.dumps(contrib.aggregate(subs[3:] + subs[:3]))
    by = {(r["gpu_name"], r["settings"]["width"], r["outcome"]): r for r in rows}
    main = by[("NVIDIA GeForce RTX 3070", 832, "ok")]
    assert main["n_submissions"] == 5  # 標準でない 1 件は数えない。OOM は数える
    assert (main["n_used"], main["n_outliers"], main["n_oom"]) == (3, 1, 1)
    assert main["peak_vram_gb"] == 7.02 and main["consistent"] is True
    assert main["outlier_submissions"] == [_sub(3.1).sha256[:8]]
    assert main["origin"] == "community" and main["source"].startswith("community")
    one = by[("NVIDIA GeForce RTX 4060", 832, "ok")]
    assert one["n_used"] == 1 and one["consistent"] is False  # 1 件だけは使わない
    oom = by[("NVIDIA GeForce RTX 3070", 1024, "oom")]
    assert oom["n_oom"] == 2 and oom["consistent"] is False
    ms = parse_measurements(contrib.community_document(rows))
    assert [usable(m) for m in ms].count(True) == 1  # まとめた行のうち見積もりに使うのは 1 行


def test_wide_spread_is_not_consistent():
    rows = contrib.aggregate([_sub(5.0), _sub(6.5)])  # 2 件だが幅が 26%
    assert rows[0]["n_used"] == 2 and rows[0]["consistent"] is False


def test_bundled_community_file_is_regenerated_from_data_contrib(db):
    """同梱の measurements_community.json は data/contrib から作り直したものと一致する
    (提出を足したら
    `genmodelfit contrib aggregate --output src/genmodelfit/data/measurements_community.json`)。"""
    root = Path(__file__).parent.parent / "data" / "contrib"
    subs, reports = contrib.load_dir(root, db)
    assert all(r.ok for r in reports)
    assert json.loads(bundled_text(COMMUNITY_FILE)) == contrib.community_document(contrib.aggregate(subs))


# --- 根拠の段階 -----------------------------------------------------------------------------------


def _community(m: Measurement, **kw) -> Measurement:
    d = {**vars(m), "settings": m.settings, "origin": "community", "n_submissions": 3, "consistent": True}
    d.update(kw)
    return Measurement(**d)


def test_basis_ordering(db):
    entry = db.get("image_sdxl_simple")
    maint = [m for m in db.measurements if m.model_id == entry.id and m.origin is None and usable(m)]
    assert len(maint) == 1 and maint[0].settings.width == 832
    s640 = Settings(width=640, height=640)
    comm = _community(maint[0], gpu_name="NVIDIA GeForce RTX 3070", gpu_vram_gb=8.0, peak_vram_gb=5.0)
    comm.settings = Settings(width=640, height=640, steps=20)
    system = pc(12)

    def est(ms, s):
        return estimate(entry, system, s, ms)

    e = est([], s640)
    assert (e.basis, e.basis_detail) == ("estimated", "estimated")
    e = est(maint, s640)
    assert (e.basis, e.basis_detail) == ("estimated", "calibrated")
    e = est([*maint, comm], s640)  # 同じ設定のほかの利用者の実測が、設定の違う作者の実測より先
    assert (e.basis, e.basis_detail) == ("measured", "measured_community")
    assert e.measured_on.startswith("NVIDIA GeForce RTX 3070") and any("中央値" in r for r in e.reasons)
    assert score_components(entry, e, system)["basis"] == 80.0
    same_maint = Measurement(**{**vars(maint[0]), "settings": Settings(width=640, height=640)})
    e = est([comm, same_maint], s640)  # 同じ設定なら作者の実測が先
    assert (e.basis, e.basis_detail) == ("measured", "measured")
    assert score_components(entry, e, system)["basis"] == 100.0
    loose = _community(comm, consistent=False)
    e = est([*maint, loose], s640)  # そろっていない提出は使わない
    assert e.basis_detail == "calibrated" and e.measured_on is None
    assert pick_measurement([loose], entry, s640) is None
    assert set(BASIS_DETAILS) == {"measured", "measured_community", "calibrated", "estimated"}
    # JSON は後方互換: basis は 2 値のまま、basis_detail を足しただけ
    d = e.to_dict()
    assert d["basis"] in ("measured", "estimated") and d["basis_detail"] in BASIS_DETAILS


def test_cli_info_json_has_basis_detail(capsys):
    assert cli.main(["info", "image_sdxl_simple", "--vram", "16", "--ram", "32", "--no-cache", "--json"]) == 0
    e = json.loads(capsys.readouterr().out)["estimate"]
    assert e["basis"] == "measured" and e["basis_detail"] == "measured"


# --- 校正への取り込み ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture_report(db):
    subs, _ = contrib.load_dir(FIX, db)
    rows = contrib.aggregate(subs)
    return rows, contrib.accuracy_report(contrib.report_db(db, rows), threshold_pct=10.0)


def test_report_on_fixtures(fixture_report):
    _rows, rep = fixture_report
    check = [x for x in rep["check"] if x["origin"] == "community"]
    assert [(x["model_id"], x["gpu_name"]) for x in check] == [
        ("image_sdxl_simple", "NVIDIA GeForce RTX 3070")
    ]
    assert check[0]["n_submissions"] == 3 and check[0]["measured_gb"] == 6.96
    assert rep["n_fit"] == 9  # torch_reserved の提出は係数の合わせに入れない(照合だけ)
    # 閾値 10% にすると、ほかの利用者の照合(+16%)が引っかかる。作者の照合(RTX 3070 の 2 件)は判定に入れない
    assert rep["breach"] is True
    assert {(b["kind"], b["model_id"]) for b in rep["breaches"]} == {("community_check", "image_sdxl_simple")}
    assert set(rep["by_modality"]) >= {"image", "3d"}
    assert rep["by_modality"]["image"]["evaluable"] is True


def test_calibration_includes_consistent_community_rows_of_the_same_metric(db):
    maint = parse_measurements(json.loads(bundled_text("measurements.json")))
    base = next(m for m in maint if m.model_id == "image_sdxl_simple" and m.vram_metric is None)
    same = _community(
        base, gpu_name="NVIDIA GeForce RTX 4090", gpu_vram_gb=24, peak_vram_gb=8.3, vram_metric=None
    )
    same.settings = Settings(width=1024, height=1024)
    loose = _community(same, consistent=False, peak_vram_gb=30.0)
    other = _community(same, vram_metric="torch_reserved")
    test_db = Database(models=db.models, measurements=[*maint, same, loose, other])
    rows = build_rows(test_db)
    assert len(rows) == 10 and sum(r.m.origin == "community" for r in rows) == 1
    assert sum(r.m.origin == "community" for r in build_check_rows(test_db)) == 1
    result = calibrate(test_db, date="2026-10-21")
    assert result["n"] == 10
    assert [x["origin"] for x in result["loo"]].count("community") == 1
    assert {x["origin"] for x in result["check"]} == {"maintainer", "community"}


# --- CLI の形 -------------------------------------------------------------------------------------


def test_cli_validate_json_and_exit_codes(tmp_path, capsys):
    assert cli.main(["contrib", "validate", str(FIX), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d) == {
        "genmodelfit",
        "command",
        "action",
        "ok",
        "files",
        "valid",
        "invalid",
        "warnings",
        "reports",
    }
    assert d["command"] == "contrib" and d["action"] == "validate" and d["files"] == 4 and d["ok"] is True
    assert set(d["reports"][0]) == {
        "path",
        "ok",
        "errors",
        "warnings",
        "sha256",
        "suggested_path",
        "model_id",
        "gpu_name",
    }
    bad = tmp_path / "rtx-3070" / "x.json"
    bad.parent.mkdir()
    b = good()
    b["measurement"]["note"] = r"C:\Users\bob"
    bad.write_text(json.dumps(b), encoding="utf-8")
    assert cli.main(["contrib", "validate", str(bad)]) == 1
    out = capsys.readouterr().out
    assert "NG" in out and "privacy" in out and "filename" in out
    assert cli.main(["contrib", "validate", str(FIX), "--markdown"]) == 0
    assert "| 結果 | ファイル |" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:  # ファイルを渡さないのは使い方の誤り
        cli.main(["contrib", "validate"])
    assert e.value.code == 2


def test_cli_validate_strict_turns_warnings_into_failure(tmp_path, capsys):
    d = good()
    d["measurement"]["standard"] = False
    p = tmp_path / contrib.suggested_path(d).split("/", 2)[2]
    p.parent.mkdir(parents=True)
    p.write_text(contrib.dumps(d), encoding="utf-8")
    assert cli.main(["contrib", "validate", str(p)]) == 0
    assert cli.main(["contrib", "validate", str(p), "--strict"]) == 1
    capsys.readouterr()


def test_cli_show(capsys):
    assert cli.main(["contrib", "show", str(SDXL_3070[0])]) == 0
    out = capsys.readouterr().out
    assert "これがファイルの中身のすべてです" in out and "Privacy checklist" in out
    assert "NVIDIA GeForce RTX 3070" in out and "[NG]" not in out
    assert cli.main(["contrib", "show", str(SDXL_3070[0]), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert set(d) == {"genmodelfit", "command", "action", "file", "ok", "submission", "checklist", "report"}
    assert [c["id"] for c in d["checklist"]] == [
        "known_fields",
        "paths",
        "host_user",
        "device_ids",
        "network",
        "date_only",
        "note_reviewed",
    ]


def test_cli_show_flags_problems(tmp_path, capsys):
    d = good()
    d["system"]["hostname"] = "DESKTOP-ABC1234"
    p = tmp_path / "x.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert cli.main(["contrib", "show", str(p)]) == 1
    out = capsys.readouterr().out
    assert "[NG] 決まった項目だけ" in out and "[NG] PC の名前" in out


def test_cli_make_from_bench_file(tmp_path, capsys):
    m = Measurement(
        model_id="image_sdxl_simple",
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=8.0,
        peak_vram_gb=6.5,
        settings=Settings(width=832, height=832, steps=20),
        seconds=40.0,
        driver="566.03",
        comfyui_version="0.40.0",
        date="2026-10-20",
        vram_metric="torch_reserved",
        note=r"保存先 C:\Users\carol\out",
    )
    src = tmp_path / "bench.json"
    src.write_text(json.dumps({"measurements": [vars(m) | {"settings": vars(m.settings)}]}), encoding="utf-8")
    out = tmp_path / "data" / "contrib"
    assert (
        cli.main(["contrib", "make", "--from", str(src), "--output", str(out), "--no-pc-info", "--json"]) == 0
    )
    d = json.loads(capsys.readouterr().out)
    assert d["action"] == "make" and len(d["submissions"]) == 1
    s = d["submissions"][0]
    assert set(s) == {"saved_to", "path_in_repo", "submission"}
    saved = Path(s["saved_to"])
    assert saved.relative_to(out).as_posix() == s["path_in_repo"].split("/", 2)[2]
    assert "carol" not in saved.read_text(encoding="utf-8")
    assert cli.main(["contrib", "validate", str(out)]) == 0
    capsys.readouterr()
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["contrib", "make", "--from", str(bad)]) == 1


def test_cli_aggregate_json_report_and_breach(tmp_path, capsys):
    out = tmp_path / "community.json"
    args = ["contrib", "aggregate", str(FIX), "--output", str(out), "--json"]
    assert cli.main(args) == 0
    d = json.loads(capsys.readouterr().out)
    assert {"action", "dir", "files", "valid", "invalid", "submissions_used", "groups", "consistent"} <= set(
        d
    )
    assert (d["files"], d["groups"], d["consistent"]) == (4, 2, 1)
    assert json.loads(out.read_text(encoding="utf-8")) == d["community"]
    assert "report" not in d
    assert (
        cli.main(["contrib", "aggregate", str(FIX), "--report", "--threshold", "10", "--fail-on-breach"]) == 3
    )
    text = capsys.readouterr().out
    assert "警告: 誤差が閾値 10% を超えた" in text and "| 種類 | LOO 件数 |" in text
    assert cli.main(["contrib", "aggregate", str(FIX), "--report", "--json"]) == 0  # 既定の 25% は超えない
    rep = json.loads(capsys.readouterr().out)["report"]
    assert rep["breach"] is False and rep["threshold_pct"] == 25.0
    assert {"breach", "breaches", "by_modality", "summary", "loo", "check", "n_fit"} <= set(rep)


def test_cli_aggregate_missing_dir_is_empty(tmp_path, capsys):
    assert cli.main(["contrib", "aggregate", str(tmp_path / "none"), "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["groups"] == 0 and d["community"]["measurements"] == []


# --- ライブラリは何も表示しない ----------------------------------------------------------------------


def test_library_writes_nothing(tmp_path, capfd, db):
    subs, _reports = contrib.load_dir(FIX, db)
    rows = contrib.aggregate(subs)
    contrib.write_community(rows, tmp_path / "c.json")
    contrib.validate_files([*GOOD, tmp_path / "missing.json"], db)
    contrib.privacy_checklist(good())
    m = subs[0].measurement
    contrib.write_submission(contrib.build_submission(m), tmp_path / "s" / "x.json")
    copy.deepcopy(rows)
    shutil.rmtree(tmp_path / "s")
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_contrib_workflow_is_read_only_and_uses_base_tool():
    import yaml

    root = Path(__file__).parent.parent
    doc = yaml.safe_load((root / ".github" / "workflows" / "contrib.yml").read_text(encoding="utf-8"))
    assert set(doc[True]) == {"pull_request"}  # pull_request_target は使わない(YAML 1.1 では on が True)
    assert doc[True]["pull_request"]["paths"] == ["data/contrib/**"]
    assert doc["permissions"] == {"contents": "read"}
    job = doc["jobs"]["validate"]
    assert job["permissions"] == {"contents": "read"}
    text = json.dumps(doc)
    assert "secrets." not in text
    checkouts = [s for s in job["steps"] if s.get("uses", "").startswith("actions/checkout")]
    assert all(s["with"]["persist-credentials"] is False for s in checkouts)
    tool = next(s for s in checkouts if s["with"].get("path") == "tool")
    assert (
        tool["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    )  # 検査の道具は PR の側から入れない
    runs = "\n".join(s.get("run", "") for s in job["steps"])
    assert "contrib validate pr/data/contrib --markdown" in runs and "GITHUB_STEP_SUMMARY" in runs
    assert "contrib aggregate pr/data/contrib --report" in runs
