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
from genmodelfit.recommend import score_components, total_score

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


def test_scrub_text_drops_whole_sentences(monkeypatch):
    monkeypatch.setattr(contrib, "_identity_words", lambda: ["MYHOST-1234", "alice42"])
    text = (
        "ComfyUI の実行時間 3.45 秒。"
        r"失敗: FileNotFoundError: C:\Users\John Smith\ComfyUI\models\x.safetensors が無い。"
        "MYHOST-1234 で測った。計測 17 回。"
        "/home/alice42/comfy/x.py を見た。"
        " GPU-0f3e1a2b-1111-2222-3333-444455556666 0000:01:00.0 http://192.168.1.20:8188/prompt"
    )
    out = contrib.scrub_text(text)
    # 空白を含むパスの続き(Smith\…)も残らないよう、当たった文ごと落とす
    assert out == "ComfyUI の実行時間 3.45 秒。計測 17 回。(個人の情報らしい文を 4 個省いた)", out
    assert not contrib.privacy_findings({"note": out})


# 攻撃の台本(priv.py / priv2.py)の文。どれも検査で誤りになり、scrub すると検査を通る
LEAKY_NOTES = [
    r"saved in C:\Users\John Smith\AppData\x.json",
    r"saved in C:\Users\山田 太郎\Documents\x",
    r"see \fileserver\share\bob",
    "see \\\\fileserver\\share\\bob",
    "see //fileserver/share/bob",
    r"in %USERPROFILE%\Documents\comfy",
    r"in $HOME/comfy ok",
    r"in ~/comfy",
    "Ｃ：＼Ｕｓｅｒｓ＼ｊｏｈｎ",
    "C" + chr(0x200B) + ":" + chr(92) + "Users" + chr(92) + "john",
    "C:¥Users¥john",
    r"\\?\C:\Users\john",
    r"Users\john\comfy",
    "/home/john/comfy",
    "/Users/john",
    "/mnt/d/ai/comfy",
    "E:/My Stuff/Taro Yamada/comfy",
    r"C:\Program Files\ComfyUI\models\johnsmith_lora.safetensors",
    "host my-gaming-rig.local",
    "PC name is KOYO-PC",
    "PC 名は taro",
    "mac aa:bb:cc:dd:ee:ff",
    "ip 192.168.1.10",
    "ip fe80::1ff:fe23:4567:890a",
    "GPU-1a2b3c4d-1111-2222-3333-444455556666",
    "SN: 1324021012345",
    "at 14:05 on 2026-10-20",
    "at 2pm",
    "user john.doe@example.com",
    "github @KoyoYeager",
    "PCI 0000:01:00.0",
    "PCI 01:00.0",
]


@pytest.mark.parametrize("note", LEAKY_NOTES)
def test_leaky_notes_are_rejected_and_scrubbed(note, db):
    d = good()
    d["measurement"]["note"] = note
    r = contrib.validate_submission(d, db, today=TODAY, check_filename=False)
    assert "privacy" in {e["code"] for e in r.errors}, (note, r.errors)
    scrubbed = contrib.scrub_text(note)
    assert scrubbed == "(個人の情報らしい文を 1 個省いた)", scrubbed
    d["measurement"]["note"] = scrubbed
    r = contrib.validate_submission(d, db, today=TODAY, check_filename=False)
    assert r.ok, r.errors


@pytest.mark.parametrize(
    "note",
    [
        "used 1024x1024 at 20 steps, 1.5.2 version",
        "ran at 2026-10-20",
        "ComfyUI の実行時間 3.45 秒。計測 17 回",
    ],
)
def test_ordinary_notes_pass(note, db):
    d = good()
    d["measurement"]["note"] = note
    assert contrib.validate_submission(d, db, today=TODAY, check_filename=False).ok


def test_patterns_are_fast_on_long_adversarial_strings():
    import time

    for s in ("a" * 2000, "a@" * 1000, "a." * 1000, "1:" * 1000, "/a" * 1000, "\\a" * 600, "-" * 2000):
        t0 = time.perf_counter()
        contrib.privacy_findings({"note": s})
        contrib.scrub_text(s)
        assert time.perf_counter() - t0 < 0.5, s[:10]


# --- 個人の情報(番兵の値): bench --submission ----------------------------------------------------

from test_privacy_pc import SENT_HOST, SENT_UUID, outbound, sentinel_pc  # noqa: E402, F401


def test_bench_submission_has_no_pc_identity(sentinel_pc, outbound, tmp_path, capsys, db):  # noqa: F811
    from test_bench import FakeComfy

    forbidden, home = sentinel_pc
    secret_note = (
        f"{SENT_HOST} で {forbidden[4]} が {home}\\ComfyUI を使い {SENT_UUID} 10.0.0.7 a@b.example 12:34:56。"
        f"{home} の下"
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
    # 偽の ComfyUI はすぐ終わる。時計の細かさで 0.0 にも 0.1 にもなる(Windows)ので、0.1 未満なら時間の値だけが
    # 範囲の外になり、0.1 以上なら誤りは無い。どちらでも名前・個人の情報などは通る
    secs = sub["measurement"]["seconds"]
    assert secs < 1.0
    if secs < 0.1:
        assert [e["code"] for e in r.errors] == ["value"] and "seconds" in r.errors[0]["message"], r.errors
    else:
        assert r.errors == [], r.errors
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
    ("tool.version", "0.3.0 by john", "schema"),
    ("system.gpu.name", "NVIDIA GeForce RTX 3070 (john's rig)", "value"),
    ("system.gpu.name", "RTX 3070 | <img src=x>\n::warning::gpu", "string"),
    ("comfyui_version", "0.40.0+john", "value"),
    ("comfyui_version", "0.40", "value"),
    ("measurement.settings.precision", "johnpc", "settings"),
    ("measurement.model_id", "x\n::error title=pwn::injected", "string"),
    ("measurement.model_id", "a" * 201, "string"),
    ("measurement.model_id", "image <b>", "value"),
    ("measurement.note", "x" * 2001, "string"),
    ("measurement.note", "line1\nline2", "string"),
    ("measurement.note", "bidi \u202e txt", "string"),
    ("measurement.offload", ["none"], "string"),
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
        ("measurement.standard", False, "nonstandard"),
        ("system.ram_total_gb", 31.4, "ram_not_rounded"),
        ("system.gpu.driver", None, "driver_missing"),
        ("measurement.settings.frames", 49, "settings_modality"),
    ],
)
def test_validate_warnings_do_not_fail(key, value, code, db):
    d = good()
    _set(d, key, value)
    r = contrib.validate_submission(d, db, today=TODAY)
    assert r.ok and r.accepted(), (r.errors, r.review)
    assert code in {w["code"] for w in r.warnings}


@pytest.mark.parametrize(
    ("key", "value", "code"),
    [
        ("measurement.model_id", "image_not_in_db", "unknown_model"),
        ("system.gpu.name", "AMD Radeon RX 7900 XTX", "unknown_gpu"),
        ("measurement.peak_vram_gb", 2.0, "deviation"),  # 式の見積もり(約 8 GB)から 40% より外れる
        ("measurement.offload", "none", "below_weights"),  # 全部載せなのに重み + 固定分より小さい
    ],
)
def test_validate_review_required(key, value, code, db, tmp_path, capsys):
    d = good()
    _set(d, key, value)
    if key == "system.gpu.name":  # 表に無い GPU は世代・compute の照合もしない
        d["system"]["gpu"]["arch"] = d["system"]["gpu"]["compute_capability"] = None
    r = contrib.validate_submission(d, db, today=TODAY)
    assert r.ok and code in {v["code"] for v in r.review}, r.review
    sha8 = r.sha256[:8]
    assert not r.accepted() and not r.accepted({"00000000"}) and r.accepted({sha8})  # 確かめるのは提出ごと
    p = tmp_path / contrib.suggested_path(d).split("/", 2)[2]
    p.parent.mkdir(parents=True)
    p.write_text(contrib.dumps(d), encoding="utf-8")
    assert cli.main(["contrib", "validate", str(p)]) == 1  # 人が確かめるまで通さない
    assert cli.main(["contrib", "validate", str(p), "--allow-review", "00000000"]) == 1
    assert cli.main(["contrib", "validate", str(p), "--allow-review", f"00000000,{sha8}"]) == 0
    capsys.readouterr()
    subs, _ = contrib.load_dir(tmp_path, db)
    assert subs == []  # まとめにも使わない
    (tmp_path / "REVIEWED").write_text(f"# 作者が確かめた提出\n{sha8} PR #99 で確かめた\n", encoding="utf-8")
    subs, _ = contrib.load_dir(tmp_path, db)  # <DIR>/REVIEWED を読む
    assert len(subs) == 1 and subs[0].approved
    assert contrib.aggregate(subs)[0]["approved"] is True
    assert cli.main(["contrib", "validate", str(p), "--reviewed", str(tmp_path / "REVIEWED")]) == 0
    capsys.readouterr()


def test_reviewed_list_and_allow_review_need_sha8(tmp_path, capsys):
    bad = tmp_path / "REVIEWED"
    bad.write_text("deadbeef\n", encoding="utf-8")  # 理由の無い承認は作らない
    with pytest.raises(ValueError):
        contrib.read_reviewed(bad)
    assert contrib.read_reviewed(tmp_path / "none") == {}
    with pytest.raises(SystemExit) as e:  # 全部をまとめて許す書き方は無い
        cli.main(["contrib", "validate", str(FIX), "--allow-review"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        cli.main(["contrib", "validate", str(FIX), "--allow-review", "all"])
    assert e.value.code == 2
    assert cli.main(["contrib", "validate", str(FIX), "--reviewed", str(bad)]) == 1
    capsys.readouterr()


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


def test_file_limits(tmp_path, db):
    big = tmp_path / "rtx-3070" / "2026-10-20-00000000.json"
    big.parent.mkdir()
    d = good()
    big.write_text(json.dumps(d) + " " * (contrib.MAX_FILE_BYTES + 1), encoding="utf-8")
    deep = tmp_path / "rtx-3070" / "2026-10-20-11111111.json"
    deep.write_text("[" * 30000 + "]" * 30000, encoding="utf-8")  # 64 KB 未満で入れ子が深い
    nested = tmp_path / "rtx-3070" / "2026-10-20-22222222.json"
    d2 = good()
    d2["measurement"]["settings"] = {"width": {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}}
    nested.write_text(json.dumps(d2), encoding="utf-8")
    reports = contrib.validate_files([big, deep, nested], db, today=TODAY)
    assert [{e["code"] for e in r.errors} >= {"read"} for r in reports[:2]] == [True, True]
    assert "大きすぎます" in reports[0].errors[0]["message"]
    assert "入れ子が深すぎます" in reports[1].errors[0]["message"]
    assert not reports[2].ok


def test_symlinks_are_rejected(tmp_path, db):
    target = tmp_path / "outside.json"
    target.write_text(contrib.dumps(good()), encoding="utf-8")
    root = tmp_path / "contrib" / "rtx-3070"
    root.mkdir(parents=True)
    link = root / "2026-10-20-33333333.json"
    try:
        link.symlink_to(target)
        (tmp_path / "contrib" / "linkdir").symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        pytest.skip("シンボリックリンクを作れない環境")
    files = contrib.find_files(tmp_path / "contrib")
    assert files == [link]  # シンボリックリンクのフォルダはたどらない
    r = contrib.validate_files(files, db, today=TODAY)[0]
    assert {e["code"] for e in r.errors} == {"read"} and "シンボリックリンク" in r.errors[0]["message"]


def test_markdown_cells_are_neutralized(tmp_path, capsys):
    """攻撃の台本 md.py: モデル ID・GPU 名・フォルダ名に Markdown / HTML / ワークフローのコマンドを入れる。"""
    d = good()
    d["measurement"]["model_id"] = (
        "x\n::error title=pwn::injected\n| a | <a href=https://evil.example>click</a> |"
    )
    d["system"]["gpu"]["name"] = "RTX 3070 | <img src=x> \n::warning::gpu"
    p = tmp_path / "evil`name" / "2026-10-20-deadbeef.json"
    p.parent.mkdir()
    p.write_text(json.dumps(d), encoding="utf-8")
    deep = tmp_path / "deep" / "2026-10-20-00000000.json"
    deep.parent.mkdir()
    deep.write_text("[" * 100000 + "]" * 100000, encoding="utf-8")
    assert cli.main(["contrib", "validate", str(tmp_path), "--markdown"]) == 1
    out = capsys.readouterr().out
    for bad in ("<a ", "<img", "\n::", "`name", "](", "| a |"):
        assert bad not in out, bad
    body = [line for line in out.splitlines() if line.startswith("| **NG**")]
    assert len(body) == 2 and all(line.count(" | ") == 4 for line in body)  # 列は増えない
    assert all(not line.startswith("::") for line in out.splitlines())
    assert contrib.md_cell("a|b`c<d>[e](f)\nx") == "a&#124;b&#96;c&lt;d&gt;&#91;e&#93;(f) x"
    # GFM の自動リンク(scheme://・www.・メール)も切る
    cell = contrib.md_cell("https://evil.example/p.png www.evil.example a@b.example")
    assert "://" not in cell and "www." not in cell and "@" not in cell


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
    subs = [_sub(6.9), _sub(7.02), _sub(7.05), _sub(3.1), _sub(6.0, standard=False)]
    subs += [_sub(5.0, gpu="NVIDIA GeForce RTX 4060")]  # 別の GPU は別の組
    subs += [_sub(7.3, outcome="oom", w=1024), _sub(7.4, outcome="oom", w=1024)]  # OOM だけの組
    rows = contrib.aggregate(subs)
    assert rows == contrib.aggregate(list(reversed(subs)))  # 並びによらない
    assert json.dumps(rows) == json.dumps(contrib.aggregate(subs[3:] + subs[:3]))
    by = {(r["gpu_name"], r["settings"]["width"], r["outcome"]): r for r in rows}
    main = by[("NVIDIA GeForce RTX 3070", 832, "ok")]
    assert main["n_submissions"] == 4  # 標準でない 1 件は数えない
    assert (main["n_used"], main["n_outliers"], main["n_oom"]) == (3, 1, 0)
    assert main["peak_vram_gb"] == 7.02 and main["consistent"] is True
    assert main["outlier_submissions"] == [_sub(3.1).sha256[:8]]
    assert main["origin"] == "community" and main["source"].startswith("community")
    one = by[("NVIDIA GeForce RTX 4060", 832, "ok")]
    assert one["n_used"] == 1 and one["consistent"] is False  # 1 件だけは使わない
    oom = by[("NVIDIA GeForce RTX 3070", 1024, "oom")]
    assert oom["n_oom"] == 2 and oom["consistent"] is False
    ms = parse_measurements(contrib.community_document(rows))
    assert [usable(m) for m in ms].count(True) == 1  # まとめた行のうち見積もりに使うのは 1 行


def test_group_with_any_oom_is_not_consistent():
    rows = contrib.aggregate([_sub(6.9), _sub(7.0), _sub(7.9, outcome="oom")])
    assert rows[0]["n_oom"] == 1 and rows[0]["consistent"] is False


def test_grouping_normalizes_gpu_names_and_precision():
    a, b = _sub(6.9), _sub(7.0, gpu="GeForce RTX 3070")
    for s, prec in ((a, "BF16"), (b, "bfloat16")):
        s.data["measurement"]["settings"]["precision"] = prec
        s.sha256 = contrib.content_hash(s.data)
    rows = contrib.aggregate([a, b])
    assert len(rows) == 1 and rows[0]["n_used"] == 2 and rows[0]["settings"]["precision"] == "bf16"


def test_grouping_keeps_vram_variants_apart():
    a, b = _sub(6.9, gpu="NVIDIA GeForce RTX 4060 Ti"), _sub(7.0, gpu="NVIDIA GeForce RTX 4060 Ti")
    b.data["system"]["gpu"]["vram_gb"] = 15.99
    b.sha256 = contrib.content_hash(b.data)
    rows = contrib.aggregate([a, b])
    assert sorted(r["gpu_vram_gb"] for r in rows) == [8.0, 15.99]  # 8 GB と 16 GB は混ぜない


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
    system = pc(12)

    def est(ms, s=s640):
        return estimate(entry, system, s, ms)

    base = est(maint)
    assert (est([]).basis, est([]).basis_detail) == ("estimated", "estimated")
    assert (base.basis, base.basis_detail) == ("estimated", "calibrated")
    # 正直に大きい値: 必要量を上げ、値は実測そのもの(basis measured、点数は推定と同じ 0)
    high = _community(
        maint[0],
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=12.0,
        peak_vram_gb=base.vram_gb + 2,
        approved=True,  # 人が確かめた(+15% を超えて上げられる)
    )
    high.settings = Settings(width=640, height=640, steps=20)
    e = est([*maint, high])
    assert (e.basis, e.basis_detail) == ("measured", "measured_community")
    assert e.vram_gb == round(high.peak_vram_gb, 2) and e.vram_gb > base.vram_gb
    assert e.measured_on.startswith("NVIDIA GeForce RTX 3070") and any("中央値" in r for r in e.reasons)
    assert score_components(entry, e, system)["basis"] == 0.0
    # 小さい値: 使わない(値・段階・点数は、ほかの利用者の実測が無いときと同じ)
    low = _community(high, peak_vram_gb=1.0)
    e = est([*maint, low])
    assert (e.basis, e.basis_detail) == ("estimated", "measured_community")
    for k in ("vram_gb", "vram_sequential_gb", "vram_offload_gb", "ram_gb", "fit", "run_mode"):
        assert getattr(e, k) == getattr(base, k), k
    assert score_components(entry, e, system) == score_components(entry, base, system)
    # 同じ設定なら作者の実測が先
    same_maint = Measurement(**{**vars(maint[0]), "settings": Settings(width=640, height=640)})
    e = est([high, same_maint])
    assert (e.basis, e.basis_detail) == ("measured", "measured")
    assert score_components(entry, e, system)["basis"] == 100.0
    loose = _community(high, consistent=False)
    e = est([*maint, loose])  # そろっていない提出は使わない
    assert e.basis_detail == "calibrated" and e.measured_on is None
    assert pick_measurement([loose], entry, s640) is None
    assert set(BASIS_DETAILS) == {"measured", "measured_community", "calibrated", "estimated"}
    d = e.to_dict()  # JSON は後方互換: basis は 2 値のまま、basis_detail を足しただけ
    assert d["basis"] in ("measured", "estimated") and d["basis_detail"] in BASIS_DETAILS


def test_torch_reserved_is_a_lower_bound(db):
    """torch_reserved(CUDA の文脈を含まない)は「これ以上」としてだけ使う。作者の RTX 3070 の 2 件も同じ。"""
    for mid in ("utility_birefnet_remove_background", "3d_moge_perspective_to_mesh"):
        entry = db.get(mid)
        ms = [m for m in db.measurements if m.model_id == mid]
        assert ms and all(m.vram_metric == "torch_reserved" for m in ms)
        with_m = estimate(entry, pc(8), None, ms)
        without = estimate(entry, pc(8), Settings(**{k: v for k, v in with_m.settings.items()}), [])
        assert with_m.vram_gb >= without.vram_gb > ms[0].peak_vram_gb
        assert with_m.basis == "estimated" and with_m.basis_detail == "calibrated"


ORDER = ["too_tight", "marginal", "good", "perfect", "unknown"]


def _poison(db, peaks, *, offload="none", gpu=("NVIDIA GeForce RTX 4090", 24.0, "ada", "8.9")):
    """攻撃の台本 poison2.py: 全モデルに、判定の設定で小さな値の提出を 2 件ずつ作る(検査は見ない)。"""
    out = []
    for t in db.models:
        e0 = estimate(t, pc(24), None, db.measurements)
        if e0.unknown_size:
            continue
        sett = {k: v for k, v in e0.settings.items() if k != "batch" and v is not None}
        subs = []
        for i, v in enumerate(peaks):
            d = good()
            d["system"]["gpu"].update(name=gpu[0], vram_gb=gpu[1], arch=gpu[2], compute_capability=gpu[3])
            d["measurement"].update(
                model_id=t.id, settings=sett, peak_vram_gb=v, offload=offload, note=f"x{i}"
            )
            subs.append(contrib.Submission(path=f"p{i}", sha256=contrib.content_hash(d), data=d))
        out.append((t, subs))
    return out


def test_poisoned_submissions_cannot_lower_needs_or_raise_levels_and_scores(db):
    poisoned = _poison(db, (0.5, 0.505))
    assert len(poisoned) > 250
    flagged = 0
    for t, subs in poisoned:
        r = contrib.validate_submission(subs[0].data, db, today=TODAY, check_filename=False)
        flagged += not r.accepted()
        cm = [Measurement.from_dict(x) for x in contrib.aggregate(subs)]
        assert cm[0].consistent  # 検査を素通りさせても(--allow-review)、見積もりは動かない
        for v in (8, 12, 16, 24):
            a = estimate(t, pc(v), None, db.measurements)
            b = estimate(t, pc(v), None, [*db.measurements, *cm])
            for k in ("vram_gb", "vram_sequential_gb", "vram_offload_gb", "ram_gb"):
                assert (getattr(b, k) or 0) >= (getattr(a, k) or 0) - 1e-9, (t.id, v, k)
            assert ORDER.index(b.fit) <= ORDER.index(a.fit), (t.id, v, a.fit, b.fit)
            assert total_score(score_components(t, b, pc(v))) <= total_score(score_components(t, a, pc(v)))
    assert flagged == len(poisoned)  # どれも人が確かめる(式から 40% より外れ、重み + 固定分より小さい)


def _upward(db, factor, models):
    out = []
    for t in models:
        e0 = estimate(t, pc(24), None, db.measurements)
        sett = {k: v for k, v in e0.settings.items() if k != "batch" and v is not None}
        subs = []
        for i, eps in enumerate((0, 0.005)):
            d = good()
            d["system"]["gpu"].update(
                name="NVIDIA A100-SXM4-80GB", vram_gb=80.0, arch="ampere", compute_capability="8.0"
            )
            v = min(80.0, round(e0.vram_gb * factor * (1 + eps), 2))
            d["measurement"].update(model_id=t.id, settings=sett, peak_vram_gb=v, note=f"x{i}")
            subs.append(contrib.Submission(path=f"p{i}", sha256=contrib.content_hash(d), data=d))
        out.append((t, subs))
    return out


def _sized(db):
    return [t for t in db.models if not estimate(t, pc(24), None, db.measurements).unknown_size]


def test_upward_poisoning_needs_per_sha_approval(db):
    """攻撃の台本 poison3.py: 検査の閾値 40% の内側(+39%)の提出 2 件。確かめていなければ、検査を素通りさせても
    段階を変えず、必要量は +15% までしか上げない。"""
    models = _sized(db)
    for t, subs in _upward(db, 1.39, models):
        cm = [Measurement.from_dict(x) for x in contrib.aggregate(subs)]
        assert cm[0].consistent and not cm[0].approved
        for v in (4, 6, 8, 12, 16, 24, 32, 48):
            for ram in (16, 64):
                sysi = pc(v)
                sysi.ram_total_gb = sysi.ram_available_gb = ram
                a = estimate(t, sysi, None, db.measurements)
                b = estimate(t, sysi, None, [*db.measurements, *cm])
                assert (b.fit, b.run_mode) == (a.fit, a.run_mode), (t.id, v, ram)
                assert b.vram_gb <= a.vram_gb * 1.15 + 0.011, (t.id, v)
                assert total_score(score_components(t, b, sysi)) <= total_score(score_components(t, a, sysi))


def test_upward_submissions_are_review_required(db):
    """validate は、段階を変える・+15% を超えて上げる提出を「要確認」にする
    (確かめたものとして足したときの効き目で見る)。"""
    models = _sized(db)[::12]
    accepted = 0
    for t, subs in _upward(db, 1.39, models):
        r = contrib.validate_submission(subs[0].data, db, today=TODAY, check_filename=False)
        assert r.ok
        if r.accepted():
            accepted += 1
            approved = [Measurement.from_dict({**x, "approved": True}) for x in contrib.aggregate(subs)]
            rise, changes = contrib.community_effect(t, contrib._own_measurements(db), approved)
            assert rise <= 0.15 + 1e-9 and changes == [], (t.id, rise, changes)
        else:
            assert {v["code"] for v in r.review} & {"raise_too_large", "fit_change", "deviation"}
    assert accepted < len(models)


def test_honest_higher_submissions_raise_needs(db):
    measured = {m.model_id for m in db.measurements}
    entry = next(
        e for e in db.models if e.modality == "image" and e.id not in measured and e.total_size_bytes
    )
    a = estimate(entry, pc(16), None, db.measurements)
    sett = {k: v for k, v in a.settings.items() if k != "batch" and v is not None}
    subs = []
    for i, v in enumerate((a.vram_gb * 1.3, a.vram_gb * 1.31)):
        d = good()
        d["system"]["gpu"].update(
            name="NVIDIA GeForce RTX 4090", vram_gb=24.0, arch="ada", compute_capability="8.9"
        )
        d["measurement"].update(model_id=entry.id, settings=sett, peak_vram_gb=round(v, 2), note=f"x{i}")
        subs.append(contrib.Submission(path=f"p{i}", sha256=contrib.content_hash(d), data=d))
    for s in subs:
        s.approved = True  # 人が確かめた(REVIEWED にある)提出
    cm = [Measurement.from_dict(x) for x in contrib.aggregate(subs)]
    assert cm[0].approved
    b = estimate(entry, pc(16), None, [*db.measurements, *cm])
    assert b.vram_gb > a.vram_gb and (b.basis, b.basis_detail) == ("measured", "measured_community")
    assert b.vram_gb == cm[0].peak_vram_gb  # measured のときの値は実測そのもの
    # 確かめていなければ +15% まで(値は実測ではないので basis は estimated)
    for s in subs:
        s.approved = False
    cm = [Measurement.from_dict(x) for x in contrib.aggregate(subs)]
    c = estimate(entry, pc(48), None, [*db.measurements, *cm])
    a48 = estimate(entry, pc(48), None, db.measurements)
    assert a48.vram_gb < c.vram_gb <= round(a48.vram_gb * 1.15, 2) + 0.01
    assert (c.basis, c.basis_detail) == ("estimated", "measured_community")
    assert any("+15%" in r for r in c.reasons)


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
        "review",
        "warnings",
        "reports",
    }
    assert d["command"] == "contrib" and d["action"] == "validate" and d["files"] == 4 and d["ok"] is True
    assert set(d["reports"][0]) == {
        "path",
        "ok",
        "errors",
        "warnings",
        "review",
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
    keys = {
        "action",
        "dir",
        "files",
        "valid",
        "invalid",
        "review",
        "submissions_used",
        "groups",
        "consistent",
    }
    assert keys <= set(d)
    assert (d["files"], d["groups"], d["consistent"]) == (4, 2, 1)
    assert json.loads(out.read_text(encoding="utf-8")) == d["community"]
    assert "report" not in d
    assert (
        cli.main(["contrib", "aggregate", str(FIX), "--report", "--threshold", "10", "--fail-on-breach"]) == 3
    )
    text = capsys.readouterr().out
    assert "警告: 誤差が閾値 10% を超えた" in text and "| 種類 | LOO 件数 |" in text
    assert "Fit-level changes caused by community data" in text
    assert cli.main(["contrib", "aggregate", str(FIX), "--report", "--json"]) == 0  # 既定の 25% は超えない
    rep = json.loads(capsys.readouterr().out)["report"]
    assert rep["breach"] is False and rep["threshold_pct"] == 25.0
    assert {"breach", "breaches", "by_modality", "summary", "loo", "check", "n_fit", "fit_changes"} <= set(
        rep
    )
    assert rep["fit_changes"] == []  # 試験用の提出は段階を変えない


def test_cli_aggregate_missing_dir_and_empty_output(tmp_path, capsys):
    out = tmp_path / "community.json"
    out.write_text("keep", encoding="utf-8")
    assert cli.main(["contrib", "aggregate", str(tmp_path / "none"), "--output", str(out)]) == 1
    assert "フォルダがありません" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["contrib", "aggregate", str(empty), "--output", str(out)]) == 1
    assert out.read_text(encoding="utf-8") == "keep"  # 空のまとめで上書きしない
    assert (
        cli.main(["contrib", "aggregate", str(empty), "--output", str(out), "--allow-empty", "--json"]) == 0
    )
    d = json.loads(capsys.readouterr().out)
    assert d["groups"] == 0 and json.loads(out.read_text(encoding="utf-8"))["measurements"] == []


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


def outside_text(job) -> str:
    return next(s for s in job["steps"] if "data/contrib の外" in s.get("name", ""))["run"]


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
    # actions はコミットの SHA で固定する
    import re

    for s in job["steps"]:
        if "uses" in s:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", s["uses"]), s["uses"]
    # 提出から来る文字はログに出さない(まとめにだけ書く)。genmodelfit の間はワークフローのコマンドを止める
    for s in job["steps"]:
        if "genmodelfit contrib" in s.get("run", ""):
            assert "::stop-commands::" in s["run"] and "cat " not in s["run"]
            assert "--reviewed tool/data/contrib/REVIEWED" in s["run"]  # 確かめた一覧は PR の向き先のもの
            assert "SHOW_ERR" in s["run"]  # 標準エラーも無害にしてまとめに書く
    assert "md_cell" in doc["env"]["SHOW_ERR"]
    assert "data/contrib/REVIEWED" in outside_text(job)
    outside = next(s for s in job["steps"] if "data/contrib の外" in s.get("name", ""))
    assert "exit 1" in outside["run"]  # data/contrib の外を変えた PR は失敗
