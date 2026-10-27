# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""いろいろな提出への強さ(#90)。分析で作った約 60 種の提出(案件 ID は A1・F26 などの名前で引く)のうち、
提出の流れ(検査・まとめ・データとして正しいかの決まり)で扱うものを試す。段階の決まり(物理的な下限・
2 つ以上の出どころ)は PR #95 の範囲。"""

from __future__ import annotations

import base64
import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from genmodelfit import cli, contrib
from genmodelfit.db import Database, load
from genmodelfit.estimate import files_fingerprint, pick_measurement
from genmodelfit.models import Gpu, Measurement, SystemInfo

TEMPLATE = {
    "contrib_schema": 2,
    "kind": "genmodelfit-measurement",
    "tool": {"name": "genmodelfit", "version": "0.10.1"},
    "date": "2026-10-27",
    "system": {
        "os_family": "windows",
        "ram_total_gb": 32,
        "gpu": {
            "name": "NVIDIA GeForce RTX 3070",
            "vram_gb": 8.0,
            "arch": "ampere",
            "compute_capability": "8.6",
            "driver": "610.88",
            "vendor": "nvidia",
            "backend": "cuda",
        },
    },
    "comfyui_version": "0.40.0",
    "measurement": {
        "model_id": "image_sdxl_simple",
        "settings": {"width": 832, "height": 832, "steps": 15},
        "outcome": "ok",
        "standard": True,
        "offload": "dynamic",
        "peak_vram_gb": 6.9,
        "peak_ram_gb": 7.4,
        "seconds": 40.0,
        "vram_metric": "torch_reserved",
    },
}


@pytest.fixture(scope="module")
def db() -> Database:
    return load(use_cache=False)


def sub(gpu: dict | None = None, **m) -> dict:
    d = copy.deepcopy(TEMPLATE)
    if gpu is not None:
        d["system"]["gpu"] = gpu
    d["measurement"].update(m)
    return d


def as_sub(d: dict, path: str | None = None) -> contrib.Submission:
    h = contrib.content_hash(d)
    return contrib.Submission(path=path or h, sha256=h, data=d)


def write(root: Path, d: dict, rel: str | None = None, raw: bytes | None = None) -> Path:
    p = root / (rel or contrib.suggested_path(d).split("/", 2)[2])
    p.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        p.write_bytes(raw)
    else:
        p.write_text(contrib.dumps(d), encoding="utf-8", newline="\n")
    return p


def pc(name: str = "NVIDIA GeForce RTX 3070", vram: float = 8.0) -> SystemInfo:
    g = Gpu(name=name, vendor="nvidia", vram_total_gb=vram, backend="cuda")
    return SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g])


def codes(r, kind: str = "errors") -> set[str]:
    return {x["code"] for x in getattr(r, kind)}


def rows_of(subs, sources=None) -> list[Measurement]:
    return [Measurement.from_dict(r) for r in contrib.aggregate(subs, sources)]


# --- 1. 足すだけ・置き場所(H3・F21b・F22・F26・F27・E1・empty) -----------------------------------------


def test_layout_errors_and_empty_folder(tmp_path, capsys):
    """置けるのは <GPU>/<日付>-<sha8>.json(小文字の .json)だけ。ほかは誤り layout。提出が 0 件なら失敗。"""
    root = tmp_path / "contrib"
    d = sub()
    good = write(root, d)
    name = good.name
    write(root, d, name)  # F21b: フォルダの直下
    write(root, d, f"rtx-3070/sub/{name}")  # F22: 深いフォルダ
    write(root, d, f"rtx-3070/{name.replace('.json', '.txt')}")  # F26
    write(root, d, "rtx-3070/2026-10-27-00000000.JSON")  # F27(Windows は大文字小文字を区別しないので別の名前)
    reports = contrib.validate_dirs([root], today=None)
    by = {Path(r.path).relative_to(root).as_posix(): r for r in reports}
    assert by[f"rtx-3070/{name}"].ok
    for rel in (
        name,
        f"rtx-3070/sub/{name}",
        f"rtx-3070/{name[:-5]}.txt",
        "rtx-3070/2026-10-27-00000000.JSON",
    ):
        assert "layout" in codes(by[rel]), rel
    assert cli.main(["contrib", "validate", str(root)]) == 1
    capsys.readouterr()
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--lang", "ja", "contrib", "validate", str(empty)]) == 1
    assert "検査する提出がありません" in capsys.readouterr().err
    # 置けない場所のファイルはまとめにも使わない
    subs, reps = contrib.load_dir(root)
    assert [Path(s.path).name for s in subs] == [name] and sum(not r.ok for r in reps) == 4


def test_exact_duplicate_in_two_places_is_an_error(tmp_path):
    """E1: 同じ中身を 2 か所に置いても 1 件(2 件目は duplicate)。"""
    root = tmp_path / "contrib"
    d = sub()
    write(root, d)
    write(root, d, "rtx-3070/" + contrib.suggested_path(d).rsplit("/", 1)[1].replace("2026", "2026", 1))
    other = sub(peak_vram_gb=7.0)
    write(root, other)
    reports = contrib.validate_dirs([root])
    assert sum(r.ok for r in reports) == 2


def test_contrib_workflow_requires_add_only_and_the_exact_layout():
    import yaml

    wf = yaml.safe_load(Path(".github/workflows/contrib.yml").read_text(encoding="utf-8"))
    steps = wf["jobs"]["validate"]["steps"]
    step = next(s for s in steps if "足すだけ" in s.get("name", ""))
    run = step["run"]
    assert '--diff-filter=DMRTC --name-only "$BASE_SHA" HEAD -- data/contrib' in run
    assert (
        "--diff-filter=A" in run
        and r"^data/contrib/[a-z0-9-]+/[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{8}\.json$" in run
    )
    assert step["working-directory"] == "pr" and "$added" not in run.replace('"$added"', "").split("echo")[0]
    # 安全の性質は変えない: pull_request・読むだけのトークン・向き先の道具
    assert set(wf[True]) == {"pull_request"} and wf["permissions"] == {"contents": "read"}
    assert any(s.get("with", {}).get("ref") == "${{ github.event.pull_request.base.sha }}" for s in steps)


# --- 4. 個人の情報(F1〜F12・F15・F24) --------------------------------------------------------------

LEAKS = {
    "F1_b64_path": base64.b64encode(rb"C:\Users\taro\ComfyUI").decode(),
    "F1b_b64_email": base64.b64encode(b"taro.yamada@example.com").decode(),
    "F2b_fullwidth_at": "taro＠example.com",
    "F3_url_noscheme": "comfy at example.com/u/taro",
    "F3b_domain": "ran on my-server.example.net port 8188",
    "F4_email_obf": "taro [at] example [dot] com",
    "F4b_email_obf2": "taro(アット)example.com",
    "F7_phone": "090-1234-5678",
    "F7b_phone_intl": "+81 90 1234 5678",
    "F10_mac_dot": "aa.bb.cc.dd.ee.ff",
    "F10b_mac_cisco": "001a.2b3c.4d5e",
    "F11_gpu_uuid_nodash": "GPU 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d",
    "F12_serial_jp": "製品番号 ABC123456",
}


@pytest.mark.parametrize("case", sorted(LEAKS))
def test_leaky_notes_are_errors(case, db):
    r = contrib.validate_submission(sub(note=LEAKS[case]), db, check_filename=False)
    assert "privacy" in codes(r), (case, r.errors)
    assert not contrib._hits(contrib.scrub_text(LEAKS[case]) or ""), case  # scrub は文ごと落とす


def test_new_patterns_do_not_hit_real_data(db):
    """作者の実測の note(bench の文)・一覧のモデル ID・名前・GPU 名の表は、新しい形に当たらない。"""
    raw = json.loads(Path("src/genmodelfit/data/measurements.json").read_text(encoding="utf-8"))
    notes = [m.get("note") or "" for m in raw["measurements"]]
    assert any("nvidia-smi" in n for n in notes)
    for n in notes:
        assert not contrib._hits(n), n
    for e in db.models:
        r = contrib.validate_submission(sub(model_id=e.id), db, check_filename=False)
        assert "privacy" not in codes(r), (e.id, r.errors)
    for name in (
        "NVIDIA GeForce RTX 4070 Ti SUPER",
        "Intel(R) Arc(TM) 140V GPU (16GB)",
        "AMD Radeon RX 7900 XTX",
    ):
        assert not contrib._hits(name), name


def test_note_length_limit():
    assert contrib.MAX_NOTE_CHARS == 500
    r = contrib.validate_submission(sub(note="計測 13 回。" * 80), check_filename=False)
    assert "string" in codes(r)
    m = Measurement(
        model_id="x", gpu_name="NVIDIA GeForce RTX 3070", gpu_vram_gb=8.0, peak_vram_gb=1, note="a。" * 400
    )
    assert len(contrib.build_submission(m)["measurement"]["note"]) <= 500  # bench の提出は切り詰める


def test_bom_and_duplicate_keys_are_clear_read_errors(tmp_path):
    """F15: BOM は消さずに断る(理由を出す)。F24: 同じキーが 2 回ある JSON は断る(後の値が黙って勝つため)。"""
    root = tmp_path / "contrib"
    d = sub()
    txt = contrib.dumps(d)
    bom = write(root, d, raw=("\ufeff" + txt).encode("utf-8"))
    r = contrib.validate_files([bom])[0]
    assert "read" in codes(r) and "BOM" in r.errors[0]["message"]
    dup = tmp_path / "dup.json"
    dup.write_text(txt.replace('"outcome": "ok"', '"outcome": "oom", "outcome": "ok"'), encoding="utf-8")
    r = contrib.validate_files([dup])[0]
    assert "read" in codes(r) and "outcome" in r.errors[0]["message"]


# --- 3. GPU の名前(A7・A7b・F14・A9・A4・A8) -----------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["NVIDIA Graphics Device", "GPU", "NVIDIA GeForce RTX 3070 taro yamada"],
    ids=["A7_odd_name", "A7b_fake_name_GPU", "F14_gpu_name_hidden"],
)
def test_unknown_nvidia_names_are_kept_but_never_used(name, db):
    gpu = {"name": name, "vram_gb": 8.0, "driver": "610.88", "vendor": "nvidia", "backend": "cuda"}
    d = sub(gpu=gpu, peak_vram_gb=6.0)
    r = contrib.validate_submission(d, db, check_filename=False)
    assert r.ok and "unknown_gpu" in codes(r, "notices"), (r.errors, r.notices)
    rows = contrib.aggregate([as_sub(d)])
    assert rows[0]["gpu_known"] is False
    entry = db.get("image_sdxl_simple")
    system = pc(name, 8.0)
    m = pick_measurement([Measurement.from_dict(rows[0])], entry, None, system.best_gpu)
    assert m is None  # 名前の表に無い GPU の行は判定に使わない


@pytest.mark.parametrize(
    ("name", "vram", "ok"),
    [
        ("NVIDIA GeForce RTX 3070", 16.0, False),  # A9
        ("NVIDIA GeForce RTX 3070", 7.99, True),
        ("NVIDIA GeForce RTX 4060 Ti", 8.0, True),  # A8
        ("NVIDIA GeForce RTX 4060 Ti", 15.99, True),
        ("NVIDIA GeForce RTX 4060 Ti", 12.0, False),
        ("NVIDIA GeForce RTX 4070 Laptop GPU", 8.0, True),  # A4: ノート用は量の表に入れない
    ],
)
def test_vram_must_match_the_name(name, vram, ok, db):
    gpu = {"name": name, "vram_gb": vram, "driver": "610.88", "vendor": "nvidia", "backend": "cuda"}
    r = contrib.validate_submission(sub(gpu=gpu, peak_vram_gb=5.0), db, check_filename=False)
    assert ("gpu_inconsistent" not in codes(r)) is ok, r.errors
    if ok:
        assert contrib.aggregate([as_sub(sub(gpu=gpu, peak_vram_gb=5.0))])[0]["gpu_known"] is True


def test_non_nvidia_gpus_keep_their_behaviour(db):
    """A1・A2: AMD・Intel は PCI の型番 ID で照合する(名前の表は NVIDIA だけ)。"""
    amd = {
        "name": "AMD Radeon RX 7900 XTX",
        "vram_gb": 24.0,
        "vendor": "amd",
        "backend": "rocm",
        "driver": "24.10.1",
        "pci_id": "1002:744c",
    }
    row = contrib.aggregate([as_sub(sub(gpu=amd, peak_vram_gb=7.0))])[0]
    assert row["gpu_known"] is True


# --- 2. 測ったときの一覧のファイル(G3) ------------------------------------------------------------


def test_files_fp_is_recorded_and_stale_rows_are_unused(db):
    entry = db.get("flux1_krea_dev")
    fp = files_fingerprint(entry)
    assert fp and len(fp) == 64
    shas = sorted(f.sha256.lower() for f in entry.files)
    import hashlib

    assert fp == hashlib.sha256("\n".join(shas).encode()).hexdigest()  # 決め方(docs/schema.md「contrib」)
    d = sub(
        model_id=entry.id,
        settings={"width": 1024, "height": 1024, "steps": 20},
        peak_vram_gb=7.6,
        files_fp=fp,
    )
    assert contrib.validate_submission(d, db, check_filename=False).ok
    rows = rows_of([as_sub(d)])
    assert rows[0].files_fp == fp
    system = pc()
    assert pick_measurement(rows, entry, None, system.best_gpu) is not None
    # 上流がファイルを差し替えた(一覧の sha256 が変わった)ら、前の実測は使わない
    swapped = copy.deepcopy(entry)
    swapped.files[0].sha256 = "0" * 64
    assert pick_measurement(rows, swapped, None, system.best_gpu) is None
    # 指紋の無い行(contrib_schema 1・作者の実測)は今までどおり使う
    old = copy.deepcopy(d)
    old["contrib_schema"] = 1
    del old["measurement"]["files_fp"]
    assert pick_measurement(rows_of([as_sub(old)]), swapped, None, system.best_gpu) is not None
    # schema 1 に files_fp は誤り・形の違う files_fp も誤り
    bad = copy.deepcopy(d)
    bad["contrib_schema"] = 1
    assert "schema" in codes(contrib.validate_submission(bad, db, check_filename=False))
    bad = copy.deepcopy(d)
    bad["measurement"]["files_fp"] = "XYZ"
    assert "value" in codes(contrib.validate_submission(bad, db, check_filename=False))
    # 違う指紋の提出は別の行になる
    other = copy.deepcopy(d)
    other["measurement"]["files_fp"] = "1" * 64
    assert len(contrib.aggregate([as_sub(d), as_sub(other)])) == 2


def test_bench_submission_carries_files_fp(db):
    from genmodelfit import bench

    args = type(
        "A",
        (),
        {
            "template": "flux1_krea_dev",
            "url": "http://127.0.0.1:1",
            "workflow": None,
            "frames": None,
            "length": None,
            "steps": None,
            "timeout": None,
        },
    )()
    opts = bench.options_from_args(args)
    assert opts.files_fp == files_fingerprint(db.get("flux1_krea_dev"))
    m = Measurement(
        model_id="flux1_krea_dev",
        gpu_name="NVIDIA GeForce RTX 3070",
        gpu_vram_gb=8.0,
        peak_vram_gb=7.0,
        files_fp=opts.files_fp,
    )
    s = contrib.build_submission(m)
    assert s["contrib_schema"] == 2 and s["measurement"]["files_fp"] == opts.files_fp


# --- 5. まとめ(D2・D3・C5・E2・n_sources) ----------------------------------------------------------


def test_error_runs_are_counted_not_turned_into_oom():
    """D2: error で終わった提出は行の値にしない(error だけなら行を作らない)。"""
    err = sub(outcome="error", peak_vram_gb=2.0, seconds=None)
    assert contrib.aggregate([as_sub(err)]) == []
    rows = contrib.aggregate([as_sub(err), as_sub(sub(peak_vram_gb=6.9))])
    assert len(rows) == 1 and rows[0]["outcome"] == "ok" and rows[0]["n_error"] == 1
    assert rows[0]["peak_vram_gb"] == 6.9


def test_oom_heavy_rows_are_not_runs_evidence():
    """D3・D3b: OOM の出どころが ok の出どころ以上なら runs_evidence は false。"""
    oom = sub(outcome="oom", offload="oom", peak_vram_gb=7.9, seconds=None)
    oom2 = sub(outcome="oom", offload="oom", peak_vram_gb=7.8, seconds=None, note="x")
    ok1 = sub(peak_vram_gb=6.9)
    assert contrib.aggregate([as_sub(ok1), as_sub(oom)])[0]["runs_evidence"] is False  # 1 対 1
    assert contrib.aggregate([as_sub(ok1), as_sub(oom), as_sub(oom2)])[0]["runs_evidence"] is False
    ok2 = sub(peak_vram_gb=7.0)
    assert contrib.aggregate([as_sub(ok1), as_sub(ok2), as_sub(oom)])[0]["runs_evidence"] is True
    assert contrib.aggregate([as_sub(ok1)])[0]["runs_evidence"] is True


def test_steps_are_not_part_of_the_key():
    """C5: ステップ数の違い・欠けは同じ組(判定の照合 estimate._settings_match もステップ数を見ない)。"""
    a = sub(settings={"width": 832, "height": 832}, peak_vram_gb=7.0)
    b = sub(settings={"width": 832, "height": 832, "steps": 30}, peak_vram_gb=7.2)
    rows = contrib.aggregate([as_sub(a), as_sub(b)])
    assert len(rows) == 1 and rows[0]["n_used"] == 2 and rows[0]["settings"]["steps"] == 30


def test_one_source_counts_once():
    """E2・E2b: 1 つの出どころ(同じ PR)の 20 件の小さな値は 1 つの値(その中央値)としてだけ数える。"""
    honest = [as_sub(sub(peak_vram_gb=v), f"h{i}") for i, v in enumerate((6.9, 7.0, 7.1))]
    spam = [as_sub(sub(peak_vram_gb=1.0, seconds=40 + i * 0.1), f"s{i}") for i in range(20)]
    sources = {s.path: f"pr-honest-{i}" for i, s in enumerate(honest)} | {s.path: "pr-spam" for s in spam}
    row = contrib.aggregate(honest + spam, sources)[0]
    assert row["n_sources"] == 3 and row["peak_vram_gb"] == 7.0 and row["n_outliers"] == 20
    # 出どころが分からない(git の外)ときは提出ごとに数えるので、数の多い側が勝つ(出どころを git で数える理由)
    row = contrib.aggregate(honest + spam)[0]
    assert row["n_sources"] == 20 and row["peak_vram_gb"] == 1.0


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.mark.skipif(shutil.which("git") is None, reason="git が無い")
def test_n_sources_from_git_history(tmp_path, capsys):
    """出どころ = そのファイルを足したコミット(main の first-parent)。同じ PR(マージ)で足した 2 件は 1 つ。"""
    repo = tmp_path / "repo"
    root = repo / "data" / "contrib"
    root.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    a, b, c = (sub(peak_vram_gb=v) for v in (6.9, 7.0, 7.1))
    write(root, a)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "first")
    _git(repo, "switch", "-q", "-c", "pr")
    write(root, b)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pr 1")
    write(root, c)
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "pr 2")
    _git(repo, "switch", "-q", "main")
    _git(repo, "merge", "-q", "--no-ff", "pr", "-m", "merge pr")
    subs, _ = contrib.load_dir(root)
    sources, via_git = contrib.submission_sources(root, subs)
    assert via_git and len(set(sources.values())) == 2
    assert contrib.aggregate(subs, sources)[0]["n_sources"] == 2
    # まだコミットしていない提出は 1 件で 1 つ
    write(root, sub(peak_vram_gb=7.05))
    subs, _ = contrib.load_dir(root)
    sources, _ = contrib.submission_sources(root, subs)
    assert len(set(sources.values())) == 3
    assert cli.main(["contrib", "aggregate", str(root), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["sources_from"] == "git"
    # git の外では提出ごと(出力にそう書く)
    plain = tmp_path / "plain"
    shutil.copytree(root, plain)
    assert contrib.submission_sources(plain, contrib.load_dir(plain)[0]) == ({}, False)
    assert cli.main(["--lang", "ja", "contrib", "aggregate", str(plain)]) == 0
    assert "出どころは提出ごとに数えた" in capsys.readouterr().out


# --- 6. 版(B1・B4c) --------------------------------------------------------------------------------


def test_version_ranges_are_notices(db):
    r = contrib.validate_submission(sub(), db, check_filename=False)
    assert not {"tool_newer", "comfyui_newer", "comfyui_old"} & codes(r, "notices")
    d = sub()
    d["tool"]["version"] = "99.0.0"  # B4c
    d["comfyui_version"] = "9.99.999"  # B1b
    r = contrib.validate_submission(d, db, check_filename=False)
    assert r.ok and {"tool_newer", "comfyui_newer"} <= codes(r, "notices")
    d = sub(model_id="image_z_image_turbo_int8", settings={"width": 768, "height": 768, "steps": 6})
    d["comfyui_version"] = "0.3.0"  # テンプレートは 0.11.0 以上
    r = contrib.validate_submission(d, db, check_filename=False)
    assert r.ok and "comfyui_old" in codes(r, "notices")
