# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""download の置き場所の決め方と genmodelfit config(#111。docs/download.ja.md「置き場所の決め方」)。

本物の ComfyUI・設定・ネットワークは使わない(conftest の _no_real_comfyui)。探す場所は一時フォルダの
comfyfind.Places、起動中の ComfyUI は偽のセッション・偽の答えで差し替える。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from test_download import HW, _Handler, make_db, server  # noqa: F401  (server は fixture)

from genmodelfit import cli, comfyfind, config, fetch, i18n
from genmodelfit.comfyfind import running_candidates as REAL_RUNNING

# --- 部品 -----------------------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _search_on(monkeypatch):
    """conftest は全体で探すのを止めている(GENMODELFIT_COMFYUI_SEARCH=0)。
    ここの試験は差し替えた場所で探す。"""
    monkeypatch.delenv(comfyfind.SEARCH_ENV, raising=False)


def test_search_env_turns_detection_off(monkeypatch, tmp_path):
    """GENMODELFIT_COMFYUI_SEARCH=0 なら、今いるフォルダが ComfyUI でも探さない(#111)。"""
    comfy = make_comfy(tmp_path / "ComfyUI")
    assert comfyfind.find_candidates(
        probe=False, cwd=comfy, places=comfyfind.Places(platform=sys.platform, home=tmp_path / "h")
    )
    monkeypatch.setenv(comfyfind.SEARCH_ENV, "0")
    assert comfyfind.find_candidates(probe=True, cwd=comfy) == []


def make_comfy(path: Path, markers: int = 5) -> Path:
    """ComfyUI に見えるフォルダ(印を markers 個)。models も作る。"""
    path.mkdir(parents=True, exist_ok=True)
    for name in comfyfind.MARKERS[:markers]:
        if "." in name:
            (path / name).write_text("", encoding="utf-8")
        else:
            (path / name).mkdir(exist_ok=True)
    (path / "models").mkdir(exist_ok=True)
    return path


def make_desktop_base(path: Path) -> Path:
    for d in comfyfind.DESKTOP_DIRS:
        (path / d).mkdir(parents=True, exist_ok=True)
    return path


def use_places(monkeypatch, places: comfyfind.Places) -> None:
    monkeypatch.setattr(comfyfind, "current_places", lambda: places)


def _run(args: list[str], capsys) -> tuple[int, str, str]:
    code = cli.main(args)
    out, err = capsys.readouterr()
    return code, out, err


def _flat(text: str) -> str:
    return "".join(text.split())


@pytest.fixture
def db_path(server, tmp_path) -> Path:  # noqa: F811
    path = tmp_path / "db.json"
    make_db(server).save(path)
    return path


# --- 見分け方 -------------------------------------------------------------------------------------


def test_four_of_five_markers(tmp_path):
    assert comfyfind.looks_like_comfyui(make_comfy(tmp_path / "a", markers=5))
    assert comfyfind.looks_like_comfyui(make_comfy(tmp_path / "b", markers=4))
    assert not comfyfind.looks_like_comfyui(make_comfy(tmp_path / "c", markers=3))
    assert not comfyfind.looks_like_comfyui(tmp_path / "missing")
    # デスクトップ版の置き場所(本体のコードは無く、5 つのフォルダがある)
    assert comfyfind.looks_like_comfyui(make_desktop_base(tmp_path / "desk"))
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "models").mkdir()
    assert not comfyfind.looks_like_comfyui(tmp_path / "half")


def test_desktop_config_per_os(tmp_path):
    base = make_desktop_base(tmp_path / "Documents" / "ComfyUI")
    for platform, ud in (
        ("win32", tmp_path / "Roaming" / "ComfyUI"),
        ("darwin", tmp_path / "home" / "Library" / "Application Support" / "ComfyUI"),
        ("linux", tmp_path / "xdg" / "ComfyUI"),
    ):
        ud.mkdir(parents=True, exist_ok=True)
        (ud / "config.json").write_text(json.dumps({"basePath": str(base)}), encoding="utf-8")
        places = comfyfind.Places(
            platform=platform,
            home=tmp_path / "home",
            appdata=tmp_path / "Roaming",
            xdg_config=tmp_path / "xdg",
        )
        [c] = comfyfind.desktop_candidates(places)
        assert c.path == base and c.source == "desktop"
        assert c.extra_configs == [ud / "extra_models_config.yaml"]
    # Linux で XDG_CONFIG_HOME が無ければ ~/.config
    p = comfyfind.Places(platform="linux", home=tmp_path / "h2")
    assert comfyfind.desktop_user_data(p) == tmp_path / "h2" / ".config" / "ComfyUI"
    # 壊れた・basePath の無い設定は黙って無視
    (tmp_path / "Roaming" / "ComfyUI" / "config.json").write_text("{bad", encoding="utf-8")
    assert (
        comfyfind.desktop_candidates(
            comfyfind.Places(platform="win32", home=tmp_path, appdata=tmp_path / "Roaming")
        )
        == []
    )


def test_comfy_cli_config_and_default(tmp_path):
    home = tmp_path / "home"
    ws = make_comfy(tmp_path / "ws")
    conf = home / "AppData" / "Local" / "comfy-cli"
    conf.mkdir(parents=True)
    (conf / "config.ini").write_text(f"[DEFAULT]\ndefault_workspace = {ws}\n", encoding="utf-8")
    cands = comfyfind.comfy_cli_candidates(comfyfind.Places(platform="win32", home=home))
    assert [c.path for c in cands] == [ws, home / "Documents" / "comfy" / "ComfyUI"]
    assert {c.source for c in cands} == {"comfy_cli"}
    linux = comfyfind.comfy_cli_candidates(comfyfind.Places(platform="linux", home=home))
    assert linux[-1].path == home / "comfy" / "ComfyUI"
    (conf / "config.ini").write_text("not an ini [[[", encoding="utf-8")  # 壊れていても落ちない
    assert len(comfyfind.comfy_cli_candidates(comfyfind.Places(platform="win32", home=home))) == 1


def test_find_candidates_order_dedupe_and_check(tmp_path, monkeypatch):
    home = tmp_path / "home"
    drive = tmp_path / "D"
    portable = make_comfy(drive / "ComfyUI_windows_portable" / "ComfyUI")
    in_docs = make_comfy(tmp_path / "Docs" / "ComfyUI")
    make_comfy(home / "ComfyUI", markers=3)  # ComfyUI に見えないものは出さない
    places = comfyfind.Places(platform="win32", home=home, drives=[drive], folders=[tmp_path / "Docs"])
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    found = comfyfind.find_candidates(probe=False, cwd=cwd, places=places)
    assert [(c.path, c.source) for c in found] == [(portable, "common"), (in_docs, "common")]
    # 今いるフォルダが同じ ComfyUI なら 1 つにまとめ、先に見つけた見つけ方(cwd)を残す
    found = comfyfind.find_candidates(probe=False, cwd=in_docs, places=places)
    assert [(c.path, c.source) for c in found] == [(in_docs, "cwd"), (portable, "common")]


# --- 起動中の ComfyUI -------------------------------------------------------------------------------


def test_folder_paths_answer(tmp_path):
    base = make_comfy(tmp_path / "C")
    data = {
        "checkpoints": str(base / "models" / "checkpoints"),
        "diffusion_models": str(base / "models" / "unet"),
        "custom_nodes": str(base / "custom_nodes"),
    }
    c = comfyfind.from_folder_paths(data, 8188)
    assert c.path == base and c.source == "running" and c.detail == "127.0.0.1:8188" and c.models_dir is None
    # is_default などで models が別の場所: その場所を覚える
    other = tmp_path / "big" / "models"
    data2 = {
        **data,
        "checkpoints": str(other / "checkpoints"),
        "vae": str(other / "vae"),
        "loras": str(other / "loras"),
    }
    assert comfyfind.from_folder_paths(data2, 8188).models_dir is None  # 無いフォルダには書かない(L11a)
    other.mkdir(parents=True)
    assert comfyfind.from_folder_paths(data2, 8188).models_dir == other
    # デスクトップ版(download_model_base)
    desk = make_desktop_base(tmp_path / "desk")
    c = comfyfind.from_folder_paths({"download_model_base": str(desk / "models")}, 8000)
    assert c.path == desk
    # ComfyUI に見えない・相対パスは使わない
    assert comfyfind.from_folder_paths({"custom_nodes": str(tmp_path / "x" / "custom_nodes")}, 8188) is None
    assert comfyfind.from_folder_paths({"custom_nodes": "custom_nodes"}, 8188) is None


class _FakeResp:
    def __init__(self, status: int, body):
        self.status_code, self._body = status, body

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _FakeSession:
    calls: ClassVar[list[dict]] = []

    def __init__(self, answers: dict):
        self.answers = answers

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, **kw):
        type(self).calls.append({"url": url, **kw})
        port = int(url.split(":")[2].split("/")[0])
        a = self.answers.get(port)
        if a is None:
            raise OSError("refused")
        return a


def test_probe_asks_only_loopback_without_redirects_or_proxy(tmp_path, monkeypatch):
    base = make_comfy(tmp_path / "C")
    good = {"custom_nodes": str(base / "custom_nodes"), "vae": str(base / "models" / "vae")}
    answers = {8000: _FakeResp(200, good)}
    _FakeSession.calls = []
    monkeypatch.setattr(fetch, "make_session", lambda: _FakeSession(answers))
    found = REAL_RUNNING()
    assert [(c.path, c.detail) for c in found] == [(base, "127.0.0.1:8000")]
    assert sorted(c["url"] for c in _FakeSession.calls) == [
        "http://127.0.0.1:8000/internal/folder_paths",
        "http://127.0.0.1:8188/internal/folder_paths",
    ]
    for c in _FakeSession.calls:
        assert c["allow_redirects"] is False and c["timeout"] == 2.0
        assert c["proxies"] == {"http": None, "https": None, "all": None}
    # 答えが無い・形が違う・200 でない → 黙って次へ
    for bad in (
        _FakeResp(404, good),
        _FakeResp(200, ValueError("no json")),
        _FakeResp(200, ["x"]),
        _FakeResp(200, {"a": 1}),
    ):
        monkeypatch.setattr(fetch, "make_session", lambda bad=bad: _FakeSession({8188: bad, 8000: bad}))
        assert REAL_RUNNING() == []


# --- extra_model_paths.yaml -------------------------------------------------------------------------

EXAMPLE = """\
#config for comfyui
comfyui:
     base_path: {base}
     # comment
     is_default: true
     checkpoints: models/checkpoints/
     text_encoders: |
          models/text_encoders/
          models/clip/
     diffusion_models: |
          models/unet/
          models/diffusion_models/
     vae: models/vae/
"""


def test_is_default_section_comfyui_style(tmp_path):
    big = tmp_path / "big"
    (big / "models").mkdir(parents=True)
    y = tmp_path / "extra_model_paths.yaml"
    y.write_text(EXAMPLE.format(base=big), encoding="utf-8")
    sec, notes = comfyfind.default_section([y])
    assert notes == [] and sec.root == big / "models" and sec.section == "comfyui"
    assert sec.folders == {"checkpoints", "text_encoders", "diffusion_models", "vae"}
    # 相対の base_path は YAML のフォルダから
    y.write_text(EXAMPLE.format(base="big"), encoding="utf-8")
    assert comfyfind.default_section([y])[0].root == big / "models"
    # is_default でない節は使わない
    y.write_text(EXAMPLE.format(base=big).replace("is_default: true", "is_default: false"), encoding="utf-8")
    assert comfyfind.default_section([y]) == (None, [])


def test_is_default_desktop_style_and_last_wins(tmp_path):
    desk = make_desktop_base(tmp_path / "desk")
    y1 = tmp_path / "a.yaml"
    y1.write_text(EXAMPLE.format(base=tmp_path / "nope"), encoding="utf-8")  # フォルダが無い → 注意
    y2 = tmp_path / "extra_models_config.yaml"
    y2.write_text(
        f'# ComfyUI extra_model_paths.yaml for win32\ncomfyui_desktop:\n  is_default: "true"\n'
        f"  custom_nodes: custom_nodes/\n  download_model_base: models\n  base_path: {desk}\n",
        encoding="utf-8",
    )
    sec, notes = comfyfind.default_section([y1, y2])
    assert sec.root == desk / "models" and sec.folders is None and sec.section == "comfyui_desktop"
    assert len(notes) == 1 and str(tmp_path / "nope") in notes[0]


def test_is_default_other_layout_or_unreadable_is_not_used(tmp_path):
    y = tmp_path / "extra_model_paths.yaml"
    y.write_text(
        f"a111:\n  base_path: {tmp_path}\n  is_default: yes\n  checkpoints: models/Stable-diffusion\n",
        encoding="utf-8",
    )
    sec, notes = comfyfind.default_section([y])
    assert sec is None and "a111" in notes[0]
    y.write_text("comfyui: {base_path: x, is_default: true}\n", encoding="utf-8")
    sec, notes = comfyfind.default_section([y])
    assert sec is None and "is_default" in notes[0]
    with pytest.raises(comfyfind.YamlUnreadable):
        comfyfind.parse_extra_yaml("a:\n  b:\n    c: d\n")
    with pytest.raises(comfyfind.YamlUnreadable):
        comfyfind.parse_extra_yaml("a:\n  - x\n")


# --- CLI: 置き場所の決め方 --------------------------------------------------------------------------


def _list_json(args: list[str], capsys) -> dict:
    code, out, _ = _run(["download", "test_tiny", "--list", "--json", *args, *HW], capsys)
    assert code == 0, out
    return json.loads(out)


def test_resolution_order(db_path, tmp_path, capsys, monkeypatch):
    flag, env, saved = (make_comfy(tmp_path / n) for n in ("flag", "env", "saved"))
    use_places(
        monkeypatch,
        comfyfind.Places(platform=sys.platform, home=tmp_path / "h", folders=[tmp_path / "auto_parent"]),
    )
    make_comfy(tmp_path / "auto_parent" / "ComfyUI")
    db = ["--db", str(db_path)]
    j = _list_json(db, capsys)  # 自動で 1 つだけ
    assert j["destination_from"] == "common" and j["comfyui_dir"] == str(tmp_path / "auto_parent" / "ComfyUI")
    assert "見つけ方:よくある場所" in _flat(_run(["download", "test_tiny", "--list", *db, *HW], capsys)[1])
    config.set_value("comfyui-dir", str(saved))
    assert _list_json(db, capsys)["destination_from"] == "config"
    monkeypatch.setenv(cli.COMFYUI_DIR_ENV, str(env))
    assert _list_json(db, capsys)["comfyui_dir"] == str(env)
    j = _list_json([*db, "--comfyui-dir", str(flag)], capsys)
    assert j["destination_from"] == "comfyui_dir_option" and j["plan"]["root"] == str(flag / "models")
    j = _list_json([*db, "--output-dir", str(tmp_path / "out")], capsys)  # --output-dir は今までどおり
    assert j["destination_from"] == "output_dir_option" and j["plan"]["layout"] == "output"
    assert j["comfyui_dir"] is None and j["plan"]["root"] == str(tmp_path / "out")


def test_saved_folder_that_disappeared_is_an_error(db_path, tmp_path, capsys):
    config.set_value("comfyui-dir", str(tmp_path / "gone"))
    code, _out, err = _run(["download", "test_tiny", "--list", "--db", str(db_path), *HW], capsys)
    assert code == 1 and "genmodelfit config unset comfyui-dir" in err


def test_list_never_probes(db_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("--list で起動中の ComfyUI に尋ねた")

    monkeypatch.setattr(comfyfind, "running_candidates", boom)
    assert _run(["download", "test_tiny", "--list", "--db", str(db_path), *HW], capsys)[0] == 0


def test_running_comfyui_is_used_when_downloading(db_path, tmp_path, capsys, monkeypatch):
    base = make_comfy(tmp_path / "Running")
    asked = []
    monkeypatch.setattr(
        comfyfind,
        "running_candidates",
        lambda *a, **k: asked.append(1) or [comfyfind.Candidate(base, "running", "127.0.0.1:8188")],
    )
    code, out, _ = _run(["download", "test_tiny", "--db", str(db_path), "--yes", "--json", *HW], capsys)
    j = json.loads(out)
    assert code == 0 and asked and j["destination_from"] == "running" and j["comfyui_dir"] == str(base)
    assert (base / "models" / "vae" / "tiny_vae.safetensors").exists()


def _two(tmp_path, monkeypatch) -> tuple[Path, Path]:
    a = make_comfy(tmp_path / "A" / "ComfyUI")
    b = make_comfy(tmp_path / "B" / "ComfyUI")
    use_places(
        monkeypatch,
        comfyfind.Places(
            platform=sys.platform, home=tmp_path / "h", folders=[tmp_path / "A", tmp_path / "B"]
        ),
    )
    return a, b


def test_several_candidates_without_terminal_is_an_error(db_path, tmp_path, capsys, monkeypatch):
    a, b = _two(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    monkeypatch.setattr("builtins.input", lambda p="": pytest.fail("尋ねた"))
    code, out, err = _run(["download", "test_tiny", "--db", str(db_path), "--yes", *HW], capsys)
    assert code == 1 and "置き場所が決まりません" in err and str(a) in err and str(b) in err
    assert "genmodelfit config set comfyui-dir" in err and _Handler.seen == []
    code, out, _ = _run(["download", "test_tiny", "--db", str(db_path), "--yes", "--json", *HW], capsys)
    j = json.loads(out)
    assert code == 1 and j["plan"] is None and j["destination_from"] is None
    assert [c["comfyui_dir"] for c in j["destination_candidates"]] == [str(a), str(b)]
    # --list は落とさないので計画を出す(未定と候補)
    j = _list_json(["--db", str(db_path)], capsys)
    assert j["destination_from"] is None and len(j["destination_candidates"]) == 2
    assert not any(a.rglob("*.safetensors*")) and not any(b.rglob("*.safetensors*"))


def test_terminal_chooses_a_candidate_and_remembers(db_path, tmp_path, capsys, monkeypatch):
    a, b = _two(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers = iter(["2", "y", ""])  # 2 番 → 落とす → 次からも(既定 Y)
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or next(answers))
    code, _out, _ = _run(["download", "test_tiny", "--db", str(db_path), *HW], capsys)
    assert (
        code == 0
        and (b / "models" / "vae" / "tiny_vae.safetensors").exists()
        and not any(a.rglob("*.safetensors"))
    )
    assert prompts[1].startswith("[y/N/c]") and "次からも" in prompts[2]
    assert config.load()[0]["comfyui-dir"] == str(b)
    # 保存したので、次は尋ねずにそこ(決め方: 設定)
    j = _list_json(["--db", str(db_path)], capsys)
    assert j["destination_from"] == "config" and j["comfyui_dir"] == str(b)


def test_terminal_types_a_path_and_declines_remember(db_path, tmp_path, capsys, monkeypatch):
    _two(tmp_path, monkeypatch)
    typed = make_comfy(tmp_path / "Typed")
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers = iter([str(tmp_path / "not-comfy"), str(typed), "y", "n"])
    monkeypatch.setattr("builtins.input", lambda p="": next(answers))
    code, _, err = _run(["download", "test_tiny", "--db", str(db_path), *HW], capsys)
    assert code == 0 and "ComfyUI のフォルダに見えません" in err
    assert (typed / "models" / "vae" / "tiny_vae.safetensors").exists()
    assert config.load()[0] == {}  # n なので保存しない


def test_c_changes_the_place_and_asks_again(db_path, tmp_path, capsys, monkeypatch):
    a, _b = _two(tmp_path, monkeypatch)
    flag = make_comfy(tmp_path / "Flag")
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers = iter(["c", "1", "y", "n"])
    monkeypatch.setattr("builtins.input", lambda p="": next(answers))
    code, out, _ = _run(
        ["download", "test_tiny", "--db", str(db_path), "--comfyui-dir", str(flag), *HW], capsys
    )
    assert code == 0 and (a / "models" / "vae" / "tiny_vae.safetensors").exists()
    assert not any(flag.rglob("*.safetensors"))
    assert out.count("同意して落としますか") == 2  # 計画を出し直して聞き直した
    # 取りやめ(空)なら、元の場所のまま聞き直す
    answers = iter(["c", "", "n"])
    monkeypatch.setattr("builtins.input", lambda p="": next(answers))
    code, out, _ = _run(
        ["download", "test_tiny", "--db", str(db_path), "--comfyui-dir", str(flag), *HW], capsys
    )
    assert code == 1 and not any(flag.rglob("*.safetensors"))


def test_flag_place_is_not_remembered(db_path, tmp_path, capsys, monkeypatch):
    flag = make_comfy(tmp_path / "Flag")
    prompts: list[str] = []
    monkeypatch.setattr("builtins.input", lambda p="": prompts.append(p) or "y")
    assert (
        _run(["download", "test_tiny", "--db", str(db_path), "--comfyui-dir", str(flag), *HW], capsys)[0] == 0
    )
    assert len(prompts) == 1 and config.load()[0] == {}


def test_is_default_section_is_used(db_path, tmp_path, capsys):
    comfy = make_comfy(tmp_path / "C")
    big = tmp_path / "big"
    (big / "models").mkdir(parents=True)
    (comfy / "extra_model_paths.yaml").write_text(EXAMPLE.format(base=big), encoding="utf-8")
    j = _list_json(["--db", str(db_path), "--comfyui-dir", str(comfy)], capsys)
    assert j["plan"]["root"] == str(big / "models") and j["extra_model_paths"]["section"] == "comfyui"
    assert not any("extra_model_paths.yaml にこのフォルダを足して" in w for w in j["plan"]["warnings"])
    out = _run(
        ["download", "test_tiny", "--list", "--db", str(db_path), "--comfyui-dir", str(comfy), *HW], capsys
    )[1]
    assert "is_default" in out and str(comfy) in out


def test_next_step_command_names_the_found_folder(tmp_path, capsys, monkeypatch):
    from test_download import _cli_db

    found = make_comfy(tmp_path / "home" / "ComfyUI")
    use_places(monkeypatch, comfyfind.Places(platform=sys.platform, home=tmp_path / "home"))
    db = _cli_db(tmp_path, content="uncensored")
    code, out, _ = _run(["download", "test_tiny", "--list", "--db", str(db), *HW], capsys)
    assert code == 0 and f"--comfyui-dir {cli.shell_quote(str(found))}" in out


# --- genmodelfit config ---------------------------------------------------------------------------


def test_config_set_show_unset_path(tmp_path, capsys):
    comfy = make_comfy(tmp_path / "C")
    code, out, _ = _run(["config", "path"], capsys)
    assert code == 0 and out.strip() == str(config.config_path())
    assert config.config_path().parent == tmp_path / "gmf-config"  # GENMODELFIT_CONFIG_DIR(conftest)
    assert _run(["config", "set", "comfyui-dir", str(comfy)], capsys)[0] == 0
    assert _run(["config", "set", "lang", "en"], capsys)[0] == 0
    j = json.loads(_run(["config", "--json"], capsys)[1])
    assert j["settings"]["comfyui-dir"] == {"value": str(comfy), "from": "file"}
    assert j["settings"]["lang"] == {"value": "ja", "from": "env"}  # GENMODELFIT_LANG(conftest)が勝つ
    j = json.loads(_run(["config", "show", "--json", "--lang", "en"], capsys)[1])
    assert j["settings"]["lang"] == {"value": "en", "from": "option"}
    j = json.loads(_run(["config", "unset", "comfyui-dir", "--json"], capsys)[1])
    assert j["removed"] is True
    j = json.loads(_run(["config", "--json"], capsys)[1])
    assert j["settings"]["comfyui-dir"] == {"value": None, "from": "default"}
    out = _run(["config"], capsys)[1]
    assert "未設定" in out and "設定ファイル" in out
    assert json.loads(config.config_path().read_text(encoding="utf-8")) == {"lang": "en"}


def test_config_refuses_bad_input(tmp_path, capsys):
    code, _, err = _run(["config", "set", "foo", "x"], capsys)
    assert code == 1 and "知らないキー" in err
    code, _, err = _run(["config", "set", "comfyui-dir", str(make_comfy(tmp_path / "x", markers=3))], capsys)
    assert code == 1 and "ComfyUI のフォルダに見えません" in err
    code, _, err = _run(["config", "set", "lang", "fr"], capsys)
    assert code == 1
    assert not config.config_path().exists()
    # デスクトップ版の置き場所は保存できる
    assert _run(["config", "set", "comfyui-dir", str(make_desktop_base(tmp_path / "desk"))], capsys)[0] == 0
    with pytest.raises(SystemExit) as e:
        cli.main(["config", "set", "lang"])
    assert e.value.code == 2


def test_broken_config_is_ignored_with_a_warning(db_path, tmp_path, capsys):
    path = config.config_path()
    path.parent.mkdir(parents=True)
    path.write_text("{not json", encoding="utf-8")
    code, out, _ = _run(["config"], capsys)
    assert code == 0 and "読めないので使いません" in out
    code, _, _err = _run(["config", "set", "lang", "ja"], capsys)
    assert code == 1 and path.read_text(encoding="utf-8") == "{not json"  # 上書きしない
    code, out, _ = _run(["download", "test_tiny", "--list", "--db", str(db_path), *HW], capsys)
    assert code == 0 and "読めないので使いません" in out
    path.write_text('{"lang": 3, "comfyui-dir": ""}', encoding="utf-8")
    values, notes = config.load()
    assert values == {} and len(notes) == 2


def test_lang_default_from_config(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)
    out = _run(["config"], capsys)[1]
    assert "config file" in out  # 既定は en
    config.set_value("lang", "ja")
    out = _run(["config"], capsys)[1]
    assert "設定ファイル" in out
    assert "config file" in _run(["config", "--lang", "en"], capsys)[1]  # --lang が勝つ
    monkeypatch.setenv(i18n.ENV_VAR, "en")
    assert "config file" in _run(["config"], capsys)[1]  # 環境変数も勝つ


def test_library_never_reads_config(tmp_path):
    """Python のライブラリは設定ファイルを読まない(置き場所の既定は前と同じ)。"""
    import genmodelfit as gmf
    from genmodelfit import downloader

    comfy = make_comfy(tmp_path / "C")
    config.set_value("comfyui-dir", str(comfy))
    p = gmf.plan_download("test_tiny", db=make_db("https://huggingface.co/x"), system_info=_pc())
    assert p.root == str(downloader.default_output_dir())


def _pc():
    from test_download import pc

    return pc()
