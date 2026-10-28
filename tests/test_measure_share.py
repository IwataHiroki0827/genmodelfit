# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""measure・share・実測の Issue の受け取り(#113)。

本物の GPU・ComfyUI・ブラウザ・クリップボード・通信には触らない。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from genmodelfit import bench, cli, contrib, measure
from genmodelfit.db import cache_dir
from genmodelfit.db import load as load_db
from genmodelfit.models import Gpu, Measurement, ModelEntry, ModelFile, SystemInfo

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).parent / "fixtures" / "contrib"
GOOD = sorted(p for p in FIX.rglob("*.json") if p.parent.name == "rtx-3070")
GPU0 = {"index": 0, "uuid": "GPU-aaaa", "name": "NVIDIA GeForce RTX 3070", "driver": "566.03"}


def pc(vram: float = 8.0, name: str = "NVIDIA GeForce RTX 3070") -> SystemInfo:
    g = Gpu(name=name, vendor="nvidia", vram_total_gb=vram, backend="cuda")
    return SystemInfo(
        os="Windows 11", cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g]
    )


def no_gpu() -> SystemInfo:
    return SystemInfo(os="Windows 11", cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[])


@pytest.fixture(scope="module")
def db():
    return load_db(use_cache=False)


@pytest.fixture
def this_pc(monkeypatch):
    """この PC は RTX 3070 8 GB(本物の検出は呼ばない)。"""
    s = pc()
    monkeypatch.setattr(cli, "_system", lambda args: s)
    monkeypatch.setattr(cli, "detect", lambda *a, **k: s)
    return s


@pytest.fixture
def fake_gpu(monkeypatch):
    """bench の GPU の確かめ(tests/test_bench.py と同じ偽物。空いている RTX 3070 が 1 枚)。"""
    monkeypatch.setattr(bench, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(bench, "smi_gpus", lambda: [dict(GPU0)])
    monkeypatch.setattr(bench, "smi_compute_apps", lambda: [])
    monkeypatch.setattr(bench, "gpu_used_now", lambda gpu_id: (1024.0, 0.0))
    monkeypatch.setattr(bench, "find_server_process", lambda url: None)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    real_run = bench.run

    def run(*a, **k):  # 偽の ComfyUI はすぐ終わるので、検査に通る時間にする
        res = real_run(*a, **k)
        res.measurement.seconds = 2.0
        return res

    monkeypatch.setattr(bench, "run", run)


@pytest.fixture
def browser(monkeypatch):
    opened: list[str] = []
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
    return opened


@pytest.fixture
def clipboard(monkeypatch):
    copied: list[str] = []
    monkeypatch.setattr(cli, "_copy_to_clipboard", lambda text: copied.append(text) or True)
    return copied


def answers(monkeypatch, *values: str | None):
    """input() の答えを順に返す(None は EOF)。尋ねた文も残す。"""
    asked: list[str] = []
    it = iter(values)

    def ask(prompt=""):
        asked.append(prompt)
        v = next(it)
        if v is None:
            raise EOFError
        return v

    monkeypatch.setattr("builtins.input", ask)
    return asked


def entry(**kw) -> ModelEntry:
    files = kw.pop(
        "files",
        [
            ModelFile(name="m.safetensors", directory="diffusion_models", size_bytes=10),
            ModelFile(name="t.safetensors", directory="text_encoders", size_bytes=20),
        ],
    )
    base = {"id": "image_x", "name": "X", "modality": "image", "files": files}
    return ModelEntry(**{**base, **kw})


def meas(model_id: str = "image_x", **kw) -> Measurement:
    base = {
        "model_id": model_id,
        "gpu_name": "NVIDIA GeForce RTX 3070",
        "gpu_vram_gb": 8.0,
        "peak_vram_gb": 5.0,
    }
    return Measurement(**{**base, **kw})


# --- 入っているか・時間・この GPU の実測 -----------------------------------------------------------


def test_installed_state(tmp_path):
    e = entry()
    assert measure.installed_state(e, []).state == "unknown"
    assert measure.installed_state(entry(files=[]), [tmp_path]).state == "unknown"
    root = tmp_path / "models"
    st = measure.installed_state(e, [root])
    assert st.state == "no" and st.missing_bytes == 30 and st.missing_files == 2 and st.total_bytes == 30
    (root / "diffusion_models").mkdir(parents=True)
    (root / "diffusion_models" / "m.safetensors").write_bytes(b"x" * 10)
    # ComfyUI の古い名前(text_encoders → clip)のフォルダにあっても入っているとみなす
    (root / "clip").mkdir()
    (root / "clip" / "t.safetensors").write_bytes(b"y" * 19)  # 大きさが違えば入っていない
    st = measure.installed_state(e, [tmp_path / "other", root])
    assert st.state == "no" and st.missing_bytes == 20 and st.missing_files == 1
    (root / "clip" / "t.safetensors").write_bytes(b"y" * 20)
    assert measure.installed_state(e, [root]).state == "yes"


def test_installed_state_never_leaves_the_folder(tmp_path):
    e = entry(files=[ModelFile(name="../../evil.bin", directory="vae", size_bytes=1)])
    (tmp_path / "evil.bin").write_bytes(b"x")
    assert measure.installed_state(e, [tmp_path / "models"]).state == "no"


def test_time_hint():
    e = entry()
    assert measure.time_hint(e, []) == (60, "typical")
    assert measure.time_hint(entry(modality="video"), []) == (300, "typical")
    ms = [meas(seconds=18.0), meas(seconds=41.0), meas(seconds=23.0), meas(seconds=999.0, outcome="oom")]
    assert measure.time_hint(e, ms) == (30, "measured")  # 中央値 23 → 10 秒単位に切り上げ。oom は使わない
    assert measure.time_hint(e, [meas(seconds=0.4)]) == (10, "measured")


def test_has_measurement_here():
    e = entry()
    g = pc().best_gpu
    assert not measure.has_measurement_here(e, [], g)
    assert measure.has_measurement_here(e, [meas()], g)
    assert measure.has_measurement_here(e, [meas(outcome="oom")], g)
    # 作業用メモリだけの実測も数える(同じ GPU で測り直しても材料が増えない)
    assert measure.has_measurement_here(e, [meas(vram_metric="torch_reserved", offload="dynamic")], g)
    assert not measure.has_measurement_here(e, [meas(outcome="error")], g)
    assert not measure.has_measurement_here(e, [meas(standard=False)], g)
    assert not measure.has_measurement_here(e, [meas(gpu_name="NVIDIA GeForce RTX 4090", gpu_vram_gb=24)], g)
    assert not measure.has_measurement_here(e, [meas(gpu_vram_gb=16.0)], g)  # 同じ名前でも VRAM が違う
    assert not measure.has_measurement_here(e, [meas(model_id="image_y")], g)


# --- 提案 -----------------------------------------------------------------------------------------


def test_suggestions_rules(db):
    s = pc()
    sugg = measure.suggestions(db, s)
    assert sugg, "RTX 3070 に載るテンプレートはある"
    by_id = {m.id: m for m in db.models}
    from genmodelfit.recommend import judge

    keys = set()
    for x in sugg:
        m = by_id[x.model_id]
        assert m.source == "comfyui-template" and not x.model_id.startswith("hf:")
        f = judge(m, s, db)
        assert f.estimate.fit in ("perfect", "good") and f.estimate.run_mode == "gpu"
        assert not measure.has_measurement_here(m, db.measurements, s.best_gpu)
        assert x.group_key not in keys  # 同じモデルは 1 件
        keys.add(x.group_key)
        assert x.installed.state == "unknown" and x.seconds % 10 == 0
    order = [(x.seconds, x.installed.missing_bytes) for x in sugg]
    assert order == sorted(order)
    assert measure.suggestions(db, no_gpu()) == []


def test_suggestions_put_installed_first_and_drop_measured(db, tmp_path):
    s = pc()
    first = measure.suggestions(db, s)
    last = first[-1]
    m = db.get(last.model_id)
    root = tmp_path / "models"
    from genmodelfit.downloader import directory_refs, infer_directory

    refs = directory_refs(db)
    for f in m.files:
        d = f.directory or infer_directory(f, refs)[0]
        p = root / d / f.name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            fh.truncate(f.size_bytes or 0)  # 大きさだけ同じ(中身は見ない)
    again = measure.suggestions(db, s, roots=[root])
    assert again[0].model_id == last.model_id and again[0].installed.state == "yes"
    # この GPU で測ったら出さない
    db2 = load_db(use_cache=False)
    db2.measurements.append(meas(model_id=first[0].model_id))
    assert first[0].model_id not in [x.model_id for x in measure.suggestions(db2, s)]


def test_resolve_keeps_template_workflows(db):
    s = pc()
    f = measure.resolve(db, s, "image_sdxl_simple")
    assert f.model.id == "image_sdxl_simple"
    hf = next(m for m in db.models if m.source == "huggingface")
    from genmodelfit.downloader import ModelNotFound

    with pytest.raises(ModelNotFound, match="bench --workflow"):
        measure.resolve(db, s, hf.id)


def test_library_writes_nothing(db, capfd, tmp_path):
    measure.suggestions(db, pc(), roots=[tmp_path])
    measure.resolve(db, pc(), "image_sdxl_simple")
    measure.saved_submissions()
    contrib.intake_issue("x", "a", 1, db=db)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


# --- measure の CLI --------------------------------------------------------------------------------


def test_measure_lists_suggestions_without_a_terminal(capsys, monkeypatch, this_pc):
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("端末でなければ尋ねない"))
    assert cli.main(["--lang", "ja", "measure", "-n", "3"]) == 0
    cap = capsys.readouterr()
    lines = [x for x in cap.out.splitlines() if re.match(r"\s+\d+\. ", x)]
    assert len(lines) == 3 and "目安" in cap.out and "genmodelfit measure <ID>" in cap.err


def test_measure_suggestions_json(capsys, this_pc):
    assert cli.main(["measure", "--json", "--limit", "2"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["command"] == "measure" and d["action"] == "suggest" and len(d["suggestions"]) == 2
    assert d["comfyui"] == {"url": None, "found_by": None}  # 探すのを止めてある(conftest)
    s = d["suggestions"][0]
    assert set(s) >= {"model_id", "installed", "download_bytes", "seconds", "seconds_basis", "fit_level"}


def test_measure_choice_q_does_nothing(capsys, monkeypatch, this_pc):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "q")
    monkeypatch.setattr(bench, "run", lambda *a, **k: pytest.fail("測らない"))
    assert cli.main(["--lang", "ja", "measure"]) == 0
    assert "測りませんでした" in capsys.readouterr().err


def test_measure_without_comfyui_explains_how_to_start(capsys, monkeypatch, this_pc):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("起動していなければ尋ねない"))
    assert cli.main(["--lang", "ja", "measure", "image_sdxl_simple"]) == 1
    cap = capsys.readouterr()
    assert "=== 測る前に ===" in cap.out and "見つかりません(起動していない)" in cap.out
    assert "ComfyUI を起動してから" in cap.err and "--url" in cap.err and "Traceback" not in cap.err


def test_measure_finds_a_running_comfyui(monkeypatch, this_pc, tmp_path):
    from genmodelfit import comfyfind

    base = tmp_path / "ComfyUI"
    (base / "models").mkdir(parents=True)
    (base / "main.py").write_text("")
    cand = comfyfind.Candidate(base, "running", "127.0.0.1:8000")
    monkeypatch.setattr(comfyfind, "search_disabled", lambda: False)
    monkeypatch.setattr(comfyfind, "running_candidates", lambda *a, **k: [cand])
    args = cli.build_parser().parse_args(["measure"])
    url, how, running = cli._measure_comfyui(args)
    assert (url, how, running) == ("http://127.0.0.1:8000", "running", cand)
    assert cli._measure_roots(args, running) == [base / "models"]
    args = cli.build_parser().parse_args(["measure", "--url", "http://10.0.0.2:8188"])
    assert cli._measure_comfyui(args) == ("http://10.0.0.2:8188", "option", None)


def test_measure_runs_and_points_to_share(capsys, monkeypatch, this_pc, fake_gpu):
    from test_bench import FakeComfy

    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "")  # Enter = はい
    with FakeComfy() as fake:
        assert cli.main(["--lang", "ja", "measure", "image_sdxl_simple", "--url", fake.url]) == 0
    out = capsys.readouterr().out
    assert asked == ["測りますか? [Y/n] "]  # 測る前に 1 回だけ。送るかどうかは尋ねない
    assert "目安" in out and "何も送りません" in out
    assert "(--url で指定)" in out
    subs = list((cache_dir() / "submissions" / "rtx-3070").glob("*.json"))
    assert len(subs) == 1 and f"送る用:    {subs[0]}" in out
    assert out.rstrip().splitlines()[-1].startswith("genmodelfit share で送ると、")
    assert list((cache_dir() / "measurements").glob("*.json"))


def test_measure_no_means_no(capsys, monkeypatch, this_pc, fake_gpu):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "n")
    monkeypatch.setattr(bench, "run", lambda *a, **k: pytest.fail("測らない"))
    # やめただけなので誤りではない(share の q と同じ)
    assert cli.main(["--lang", "ja", "measure", "image_sdxl_simple", "--url", "http://127.0.0.1:9"]) == 0
    assert "測りませんでした" in capsys.readouterr().err


def test_measure_json_needs_yes_and_reports(capsys, monkeypatch, this_pc, fake_gpu):
    from test_bench import FakeComfy

    assert cli.main(["measure", "image_sdxl_simple", "--json", "--url", "http://127.0.0.1:9"]) == 1
    assert "--yes" in json.loads(capsys.readouterr().out)["error"]
    with FakeComfy() as fake:
        assert cli.main(["measure", "image_sdxl_simple", "--json", "--yes", "--url", fake.url]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["action"] == "measure" and d["model_id"] == "image_sdxl_simple"
    assert d["comfyui"] == {"url": fake.url, "found_by": "option"} and d["installed"]["state"] == "unknown"
    assert d["time_hint"]["basis"] in ("measured", "typical") and d["measurement"]["outcome"] == "ok"
    assert Path(d["submission"]["saved_to"]).is_file() and "genmodelfit share" in d["hint"]


def test_measure_busy_gpu_is_not_measured(capsys, monkeypatch, this_pc, fake_gpu):
    from test_bench import FakeComfy

    monkeypatch.setattr(bench, "gpu_used_now", lambda gpu_id: (6000.0, 90.0))
    with FakeComfy() as fake:
        assert cli.main(["--lang", "ja", "measure", "image_sdxl_simple", "--yes", "--url", fake.url]) == 1
    err = capsys.readouterr().err
    assert "エラー" in err and "次にすること: ほかの生成" in err  # measure に無い --force だけを案内しない
    assert not (cache_dir() / "submissions").exists()


def test_measure_offers_download_when_not_installed(capsys, monkeypatch, this_pc, tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "models").mkdir(parents=True)
    (comfy / "main.py").write_text("")
    calls: list[str] = []

    def fake_download(args, model_id):
        calls.append(model_id)
        return 1  # 利用者が落とすのを断った

    monkeypatch.setattr(cli, "_measure_download", fake_download)
    monkeypatch.setattr(bench, "run", lambda *a, **k: pytest.fail("落とせなければ測らない"))
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "y")
    argv = ["--lang", "ja", "measure", "image_sdxl_simple", "--url", "http://127.0.0.1:9"]
    assert cli.main([*argv, "--comfyui-dir", str(comfy)]) == 1
    out = capsys.readouterr().out
    assert calls == ["image_sdxl_simple"] and "入っていない(落とす量" in out and "先に落とします" in out
    # --json では落とさず、download を案内する
    calls.clear()
    assert (
        cli.main(
            ["measure", "image_sdxl_simple", "--json", "--yes", "--comfyui-dir", str(comfy), "--url", "u"]
        )
        == 1
    )
    d = json.loads(capsys.readouterr().out)
    assert calls == [] and "genmodelfit download image_sdxl_simple" in d["error"]


def test_measure_download_uses_the_download_command(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "cmd_download", lambda ns: seen.append(ns) or 0)
    args = cli.build_parser().parse_args(["measure", "x", "--yes", "--comfyui-dir", "D:/C"])
    assert cli._measure_download(args, "image_sdxl_simple") == 0
    ns = seen[0]
    assert ns.model == "image_sdxl_simple" and ns.yes is True and ns.comfyui_dir == "D:/C"
    # 同意の決まりは download のまま: 無検閲・ライセンス不明の同意は足さない
    assert ns.content is None and ns.include_unknown_license is False


def test_measure_unknown_model_lists_candidates(capsys, this_pc):
    assert cli.main(["measure", "sdxl", "--json", "--yes"]) == 1
    d = json.loads(capsys.readouterr().out)
    assert d["candidates"] and "error" in d


def test_measure_without_gpu(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_system", lambda args: no_gpu())
    assert cli.main(["--lang", "ja", "measure"]) == 1
    assert "GPU が見つかりません" in capsys.readouterr().err


# --- share の CLI ---------------------------------------------------------------------------------


def _saved(n: int = 2) -> list[Path]:
    out = []
    for i in range(n):
        d = json.loads(GOOD[0].read_text(encoding="utf-8"))
        d["measurement"]["peak_vram_gb"] = 6.0 + i
        d["measurement"]["note"] = None
        out.append(contrib.write_submission(d, str(measure.submissions_dir()) + "/"))
    return out


def test_share_lists_unsent_and_marks_after_opening(capsys, monkeypatch, browser, clipboard):
    paths = _saved(2)
    assert cli.main(["share", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["command"] == "share" and d["listed"] is True and len(d["submissions"]) == 2
    assert all(s["shared"] is False for s in d["submissions"])
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "2")
    assert cli.main(["--lang", "ja", "share"]) == 0
    out = capsys.readouterr().out
    assert (
        "=== 送っていない実測 ===" in out
        and len(browser) == 1
        and "issues/new?template=measurement.yml" in browser[0]
    )
    marks = json.loads(measure.shared_path().read_text(encoding="utf-8"))["shared"]
    assert len(marks) == 1
    assert cli.main(["share", "--json"]) == 0
    left = json.loads(capsys.readouterr().out)["submissions"]
    assert len(left) == 1 and Path(left[0]["file"]) in paths
    assert cli.main(["share", "--json", "--all"]) == 0
    assert len(json.loads(capsys.readouterr().out)["submissions"]) == 2


def test_share_nothing_to_send(capsys, monkeypatch, browser):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["--lang", "ja", "share"]) == 0
    assert "送っていない実測はありません" in capsys.readouterr().out and browser == []


def test_share_print_url(capsys, monkeypatch, browser):
    p = _saved(1)[0]
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["share", str(p), "--print-url"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert (
        len(out) == 1
        and out[0].startswith("https://github.com/laxshmate/genmodelfit/issues/new?")
        and browser == []
    )


def test_share_not_marked_when_the_browser_fails(capsys, monkeypatch):
    p = _saved(1)[0]
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda *a, **k: False)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["--lang", "ja", "share", str(p)]) == 0
    assert "次のページ" in capsys.readouterr().out  # 開けなければ URL を見せる
    assert measure.load_shared() == {}


def test_contrib_share_is_share(capsys):
    p = _saved(1)[0]
    assert cli.main(["contrib", "share", str(p), "--json"]) == 0
    a = json.loads(capsys.readouterr().out)
    assert cli.main(["share", str(p), "--json"]) == 0
    b = json.loads(capsys.readouterr().out)
    assert a["command"] == "contrib" and a["action"] == "share" and b["command"] == "share"
    assert a["share"] == b["share"]


# --- クリップボード -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("platform", "env", "tools", "argv", "encoding"),
    [
        ("win32", {}, {"clip"}, ["clip"], "utf-16-le"),
        ("darwin", {}, {"pbcopy"}, ["pbcopy"], "utf-8"),
        ("linux", {"WAYLAND_DISPLAY": "w"}, {"wl-copy", "xclip"}, ["wl-copy"], "utf-8"),
        ("linux", {}, {"xclip", "xsel"}, ["xclip", "-selection", "clipboard"], "utf-8"),
        ("linux", {}, {"xsel"}, ["xsel", "--clipboard", "--input"], "utf-8"),
    ],
)
def test_copy_to_clipboard_uses_os_tools(monkeypatch, platform, env, tools, argv, encoding):
    import shutil

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(shutil, "which", lambda name: f"/bin/{name}" if name in tools else None)
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", run)
    assert cli._copy_to_clipboard('{"note": "測った"}') is True
    cmd, kw = calls[0]
    assert cmd == [f"/bin/{argv[0]}", *argv[1:]] and "shell" not in kw
    data = kw["input"]
    text = data.decode(encoding)
    assert text.lstrip("\ufeff") == '{"note": "測った"}'


def test_copy_to_clipboard_without_a_tool(monkeypatch):
    import shutil

    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("道具が無ければ呼ばない"))
    assert cli._copy_to_clipboard("x") is False


def test_copy_to_clipboard_failure_is_false(monkeypatch):
    import shutil

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(shutil, "which", lambda name: "clip")

    def boom(*a, **k):
        raise subprocess.TimeoutExpired("clip", 10)

    monkeypatch.setattr(subprocess, "run", boom)
    assert cli._copy_to_clipboard("x") is False


# --- Issue の受け取り -----------------------------------------------------------------------------


def _body(text: str) -> str:
    return f"### Measurement (JSON)\n\n```json\n{text}\n```\n"


def test_extract_issue_json():
    j = '{"a": 1}'
    assert contrib.extract_issue_json(_body(j)) == j
    assert contrib.extract_issue_json(_body(j).replace("\n", "\r\n")) == j
    assert contrib.extract_issue_json("```\n" + j + "\n```") == j
    assert contrib.extract_issue_json(j) == j
    assert contrib.extract_issue_json("```sh\nrm -rf /\n```\n" + _body(j)) == j  # { で始まる囲みだけ
    assert contrib.extract_issue_json("_No response_") is None
    assert contrib.extract_issue_json(None) is None


def test_intake_accepts_a_good_submission(db, tmp_path):
    text = GOOD[0].read_text(encoding="utf-8")
    data = json.loads(text)
    res = contrib.intake_issue(_body(text), "alice", 12, db=db, existing=tmp_path / "none")
    assert res.ok and res.reasons == []
    assert res.path_in_repo == contrib.suggested_path(data) and res.sha256 == contrib.content_hash(data)
    assert res.content == contrib.dumps(data)  # 書き直した中身(本文の文字そのままではない)
    assert (
        "Closes #12" in res.pr_body and "@alice" in res.pr_body and "issue #12 by alice" in res.commit_message
    )
    assert res.title == f"measurement: {data['measurement']['model_id']} on {data['system']['gpu']['name']}"


def test_intake_refuses_duplicates_and_bad_input(db, tmp_path):
    text = GOOD[0].read_text(encoding="utf-8")
    data = json.loads(text)
    existing = tmp_path / "data" / "contrib"
    contrib.write_submission(data, str(existing) + "/")
    res = contrib.intake_issue(_body(text), "alice", 1, db=db, existing=existing)
    assert not res.ok and any("already" in r or "既に" in r for r in res.reasons)
    for body in ("hello", _body("{not json"), _body('{"a": 1, "a": 2}'), _body("{" + " " * 70000 + "}")):
        r = contrib.intake_issue(body, "alice", 1, db=db)
        assert not r.ok and r.reasons and r.content is None and r.pr_body == ""


def test_intake_comment_is_harmless(db):
    d = json.loads(GOOD[0].read_text(encoding="utf-8"))
    d["measurement"]["note"] = "see https://evil.example/x @everyone <img src=x> C:\\Users\\John Smith\\x"
    res = contrib.intake_issue(_body(json.dumps(d)), "mallory", 3, db=db)
    assert not res.ok
    body = "\n".join(line for line in res.comment.splitlines() if line.startswith("- "))
    assert "@everyone" not in body and "<img" not in body and "https://" not in body


def test_intake_author_is_checked(db):
    text = GOOD[0].read_text(encoding="utf-8")
    res = contrib.intake_issue(_body(text), "bad name; rm -rf /", 4, db=db)
    assert res.ok and "@unknown" in res.pr_body and "rm -rf" not in res.pr_body + res.commit_message


def test_cli_intake_writes_files_and_prints_nothing_from_the_body(tmp_path, capsys):
    text = GOOD[0].read_text(encoding="utf-8")
    ev = tmp_path / "event.json"
    ev.write_text(
        json.dumps({"issue": {"number": 9, "body": _body(text), "user": {"login": "bob"}}}), "utf-8"
    )
    out = tmp_path / "out"
    assert (
        cli.main(["contrib", "intake", str(ev), "--existing", str(tmp_path / "none"), "--out", str(out)]) == 0
    )
    assert capsys.readouterr().out == "accepted\n"
    res = json.loads((out / "result.json").read_text(encoding="utf-8"))
    path = (out / "path.txt").read_text(encoding="utf-8").strip()
    assert res["ok"] and path == res["path_in_repo"]
    assert re.fullmatch(r"data/contrib/[a-z0-9-]+/[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{8}\.json", path)
    assert json.loads((out / "submission.json").read_text(encoding="utf-8")) == json.loads(text)
    comment = (out / "comment.md").read_text(encoding="utf-8")
    assert "Accepted" in comment and "受け付けました" in comment  # 英語と日本語
    pr = (out / "pr_body.md").read_text(encoding="utf-8")
    assert pr.count("Closes #9") == 1 and "@bob" in pr
    # 断るとき
    ev.write_text(
        json.dumps({"issue": {"number": 9, "body": "::error::x", "user": {"login": "bob"}}}), "utf-8"
    )
    out2 = tmp_path / "out2"
    assert cli.main(["contrib", "intake", str(ev), "--out", str(out2)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("refused") and "::" not in printed
    assert not (out2 / "submission.json").exists() and (out2 / "comment.md").is_file()
    ev.write_text("{}", "utf-8")
    assert cli.main(["contrib", "intake", str(ev), "--out", str(out2)]) == 1


# --- ワークフロー --------------------------------------------------------------------------------


def _wf() -> tuple[dict, str]:
    yaml = pytest.importorskip("yaml")
    path = ROOT / ".github" / "workflows" / "measurement-intake.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8")), path.read_text(encoding="utf-8")


def test_intake_workflow_treats_the_body_as_data():
    wf, text = _wf()
    on = wf.get("on", wf.get(True))
    # 印を付けたとき(labeled)は動かない: フォームの Issue は作ったときに印が付いていて、2 回動くため
    assert set(on) == {"issues"} and set(on["issues"]["types"]) == {"opened", "edited"}
    assert wf["permissions"] == {"contents": "read"}
    check, publish = wf["jobs"]["check"], wf["jobs"]["publish"]
    assert check["permissions"] == {"contents": "read"}  # パッケージを入れるジョブは読むだけ
    assert publish["permissions"] == {"contents": "write", "pull-requests": "write", "issues": "write"}
    assert publish["needs"] == "check"
    assert "'measurement'" in check["if"] and "github.event.issue.state == 'open'" in check["if"]
    assert "github.event.changes.body != null" in check["if"]  # 題だけの編集では動かない
    sha = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
    for job in (check, publish):
        for st in job["steps"]:
            if "uses" in st:
                assert sha.match(st["uses"]), st["uses"]
                if st["uses"].startswith("actions/checkout@"):
                    assert st["with"]["persist-credentials"] is False
            assert "${{" not in st.get("run", "")  # 式は run に埋め込まない(env だけ)
            for value in st.get("env", {}).values():
                assert not re.search(r"issue\.(body|title|user)|comment\.body", str(value))
    # 書くジョブはほかの人のパッケージを入れない(git と gh だけ)
    pub_runs = "\n".join(st.get("run", "") for st in publish["steps"])
    pub_uses = [st["uses"].split("@")[0] for st in publish["steps"] if "uses" in st]
    assert set(pub_uses) == {"actions/checkout", "actions/download-artifact"}
    assert "pip install" not in pub_runs and "uv " not in pub_runs and "genmodelfit" not in pub_runs
    code = "\n".join(x for x in text.splitlines() if not x.lstrip().startswith("#"))
    assert "secrets." not in code and "pull_request_target" not in code
    chk_runs = "\n".join(st.get("run", "") for st in check["steps"])
    assert 'contrib intake "$GITHUB_EVENT_PATH"' in chk_runs and "::stop-commands::" in chk_runs
    assert (
        next(st for st in check["steps"] if st.get("uses", "").startswith("actions/checkout"))["with"][
            "fetch-depth"
        ]
        == 0
    )  # 同じ人の前の提出を数えるため
    assert r"^data/contrib/[a-z0-9-]+/[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9a-f]{8}\.json$" in pub_runs
    assert "--body-file" in pub_runs and "gh pr merge" not in pub_runs


def test_intake_workflow_is_idempotent_and_follows_edits():
    wf, _ = _wf()
    run = "\n".join(st.get("run", "") for st in wf["jobs"]["publish"]["steps"])
    # 同じ中身の PR があれば push もコメントもしない
    i_same = run.index('[ "$have" = "$want" ]')
    assert "exit 0" in run[i_same : i_same + 80] and run.index("git push") > i_same
    # 別の Issue の PR に同じ中身があれば知らせて止める
    assert "measurement/issue-[0-9]+$" in run and "dup.md" in run
    # 通らない中身なら印を付けて PR を閉じ、通る中身なら PR を直して知らせる。閉じた PR は push の前に開き直す
    assert 'close_invalid "$pr" "$OUT/comment_pr_closed.md"' in run and 'gh pr close "$1"' in run
    assert '--add-label "$invalid"' in run and 'invalid="measurement-invalid"' in run
    assert "gh pr edit" in run and "comment_updated.md" in run
    assert run.index("gh pr reopen") < run.index("git push")
    # 開き直すのはボットが閉じた(印のある)PR だけ。メンテナが閉じた PR は開き直さない
    i_reopen = run.index("gh pr reopen")
    assert 'index(\\"${invalid}\\")' in run[:i_reopen] and "--remove-label" in run[i_reopen : i_reopen + 200]
    assert "not reopening" in run
    # 別の Issue と同じ中身に変わったら、この Issue の古い PR も閉じる
    i_dup = run.index("dup.md")
    assert 'close_invalid "$pr" "$OUT/dup.md"' in run[i_dup:]
    # 見るのはボットの PR だけ(フォークの同じ名前のブランチを除く)。gh pr list はすべて bot_prs を通す
    assert "(.isCrossRepository | not)" in run and '.author.login == "app/github-actions"' in run
    assert run.count("gh pr list") == 1 and run.count("bot_prs ") >= 5
    # 同じ理由で断るコメントは 1 回だけ
    assert "marker.txt" in run and "same reasons as before" in run


def test_intake_comments_for_edits(db):
    text = GOOD[0].read_text(encoding="utf-8")
    ok = contrib.intake_issue(_body(text), "alice", 5, db=db, prior=2)
    assert ok.ok and "Submitted-by: github:alice" in ok.commit_message
    assert "already sent for this GPU and model: 2" in ok.pr_body or "既に送った提出: 2 件" in ok.pr_body
    assert "#5" in ok.comment_updated and ok.path_in_repo in ok.comment_updated
    bad = contrib.intake_issue("nothing", "alice", 5, db=db)
    assert not bad.ok and "#5" in bad.comment_pr_closed
    unknown = contrib.intake_issue(_body(text), "bad name", 5, db=db)
    assert "Submitted-by" not in unknown.commit_message  # アカウント名が分からなければ書かない


def test_intake_ignores_the_runner_identity(db, monkeypatch):
    """intake は GitHub の実行機の名前・ユーザ名(runner など)を「この PC の名前」として断らない。"""
    d = json.loads(GOOD[0].read_text(encoding="utf-8"))
    d["measurement"]["note"] = "runner measured with bench"
    text = json.dumps(d)
    monkeypatch.setattr(contrib, "_identity_words", lambda: ["runner"])
    checks = contrib.privacy_checklist(d, text)
    assert any(c["id"] == "host_user" and c["ok"] is False for c in checks)  # share では断る(手元の PC)
    assert contrib.intake_issue(_body(text), "alice", 1, db=db).ok


def test_share_print_url_without_a_file(capsys, monkeypatch, browser):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["share", "--print-url"]) == 1  # 送るものが無い
    assert "genmodelfit measure" in capsys.readouterr().err
    p = _saved(1)[0]
    assert cli.main(["share", "--print-url"]) == 0  # 1 つだけならそれ
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("https://github.com/") and browser == []
    assert measure.load_shared()  # URL を出したので印を付けた
    _saved(3)
    assert cli.main(["--lang", "ja", "share", "--print-url"]) == 1  # 2 つ以上なら選ぶように言う
    err = capsys.readouterr().err
    assert "送るファイルを指定してください" in err and str(p) not in err  # 送ったものは出さない


def test_share_json_preview_does_not_mark(capsys):
    p = _saved(1)[0]
    assert cli.main(["share", str(p), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["marked_shared"] is False and measure.load_shared() == {}
    assert cli.main(["share", str(p), "--json", "--mark-sent"]) == 0
    assert json.loads(capsys.readouterr().out)["marked_shared"] is True and measure.load_shared()


def test_share_without_a_terminal_does_not_mark(capsys, monkeypatch, browser):
    p = _saved(1)[0]
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    assert cli.main(["share", str(p)]) == 0
    assert "issues/new" in capsys.readouterr().out and browser == [] and measure.load_shared() == {}


@pytest.mark.parametrize(
    ("outcome", "metric", "offload", "word"),
    [
        ("ok", "torch_reserved", "dynamic", "照合"),
        ("ok", "torch_reserved", "none", "段階も実測で決まります"),
        ("oom", "torch_reserved", "oom", "段階を決めなくなります"),
    ],
)
def test_share_hint_matches_how_it_is_used(outcome, metric, offload, word):
    m = meas(outcome=outcome, vram_metric=metric, offload=offload)
    assert word in cli._share_hint(m) and "次の版が出てから" in cli._share_hint(m)


def test_measure_next_step_never_only_says_force():
    busy = "GPU の使用率が 90% です。空いてから測るか、--force を付けてください"
    assert "ほかの生成" in cli._measure_next_step(busy, "image_x")
    assert "genmodelfit download image_x" in cli._measure_next_step(
        "必要なファイルがサーバーにありません", "image_x"
    )
    assert "ComfyUI" in cli._measure_next_step("connection refused", "image_x")


def test_intake_marker_is_stable_per_reason_set(db, tmp_path):
    a = contrib.intake_issue("nothing", "x", 1, db=db)
    b = contrib.intake_issue("still nothing", "x", 1, db=db)
    c = contrib.intake_issue(_body("{not json"), "x", 1, db=db)
    assert a.marker == b.marker != c.marker and a.marker.startswith("<!-- measurement-intake:reasons=")
    assert a.comment.rstrip().endswith(a.marker)
    ev = tmp_path / "e.json"
    ev.write_text(json.dumps({"issue": {"number": 1, "body": "nothing", "user": {"login": "x"}}}), "utf-8")
    assert cli.main(["contrib", "intake", str(ev), "--out", str(tmp_path / "o")]) == 0
    from genmodelfit import i18n

    with i18n.forced("en"):  # marker.txt は英語の理由の組の印
        en = contrib.intake_issue("nothing", "x", 1, db=db).marker
    assert (tmp_path / "o" / "marker.txt").read_text(encoding="utf-8").strip() == en
