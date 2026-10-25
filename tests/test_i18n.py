# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""表示の言語(#73): 決め方の順(既定は en)と、en で日本語の文字が 1 つも出ないこと、ja で今の文が出ること。

一覧に保存された自由記述(review.note・license_evidence・description.ja・実測の source / note)は訳さないので、
JSON ではその項目を外してから、表では一覧にある文そのものを外してから調べる。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from clihelp import run_cli

import genmodelfit as gmf
from genmodelfit import cli, i18n

# 日本語の文字(かな・漢字・全角)と和文の約物(U+3000–303F)
JA = re.compile(r"[　-〿぀-ヿ一-鿿＀-￯]")
HW = ["--vram", "8", "--ram", "32", "--no-cache"]
FIXTURES = Path(__file__).parent / "fixtures"
DIFF_PAIR = [str(FIXTURES / "diff" / "0.6.0-0.7.0" / name) for name in ("old.json", "new.json")]


@pytest.fixture
def no_pin(monkeypatch):
    """conftest の ja の固定(環境変数)を外す。"""
    monkeypatch.delenv(i18n.ENV_VAR, raising=False)


# --- 決め方の順 -----------------------------------------------------------------


def test_default_is_en_even_on_a_japanese_os(no_pin, monkeypatch):
    """既定はいつも英語。OS の言語(LANG・LC_ALL など)は見ない(2026-10-25 にオーナーが決めた)。"""
    monkeypatch.setenv("LANG", "ja_JP.UTF-8")
    monkeypatch.setenv("LC_ALL", "ja_JP.UTF-8")
    monkeypatch.setenv("LC_MESSAGES", "ja_JP.UTF-8")
    assert gmf.get_language() == "en"


def test_set_language_beats_default(no_pin):
    gmf.set_language("ja")
    assert gmf.get_language() == "ja"
    gmf.set_language(None)  # 取り消すと既定の en に戻る
    assert gmf.get_language() == "en"


def test_env_beats_set_language(no_pin, monkeypatch):
    gmf.set_language("en")
    monkeypatch.setenv(i18n.ENV_VAR, "ja")
    assert gmf.get_language() == "ja"
    monkeypatch.setenv(i18n.ENV_VAR, "xx")  # 知らない値は無視して次へ
    assert gmf.get_language() == "en"


def test_cli_flag_beats_env(monkeypatch, capsys):
    monkeypatch.setenv(i18n.ENV_VAR, "ja")
    gmf.set_language("ja")
    assert cli.main(["types", "--no-cache", "--lang", "en"]) == 0
    assert not JA.search(capsys.readouterr().out)
    assert gmf.get_language() == "ja"  # main が終われば --lang は残らない


def test_cli_flag_does_not_leak_to_other_threads(no_pin):
    """--lang の固定は ContextVar。ほかのスレッドのライブラリの呼び出しには効かない。"""
    import threading

    seen: list[str] = []
    with i18n.forced("ja"):
        th = threading.Thread(target=lambda: seen.append(gmf.get_language()))
        th.start()
        th.join()
        assert gmf.get_language() == "ja"
    assert seen == ["en"]


def test_set_language_rejects_unknown():
    with pytest.raises(ValueError):
        gmf.set_language("fr")
    with pytest.raises(ValueError):
        gmf.set_language("ja_JP")


def test_tr_and_pick(no_pin):
    gmf.set_language("ja")
    assert i18n.tr("日本語", "English") == "日本語"
    assert i18n.pick({"en": "E", "ja": "J"}) == "J"
    gmf.set_language("en")
    assert i18n.tr("日本語", "English") == "English"
    assert i18n.tr("日本語", "English").ja == "日本語"  # 保存する文のために日本語も持つ
    assert i18n.pick({"en": "E", "ja": "J"}) == "E"
    assert i18n.pick({"ja": "J"}) == "J"  # 無ければもう片方


def test_text_survives_copy_and_pickle(no_pin):
    import copy
    import pickle

    gmf.set_language("en")
    t = i18n.tr("日本語", "English")
    for c in (copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
        assert c == "English" and c.ja == "日本語"


def test_env_reaches_child_process(tmp_path):
    r = run_cli(["types", "--no-cache"], tmp_path, GENMODELFIT_LANG="en")
    assert r.returncode == 0
    assert not JA.search(r.stdout.decode("utf-8"))


def test_child_process_defaults_to_en(tmp_path):
    """環境変数も --lang も無ければ、日本語の OS(LANG=ja_JP)でも英語。"""
    import subprocess
    import sys

    from clihelp import child_env

    env = child_env(tmp_path)
    env.pop(i18n.ENV_VAR, None)
    env["LANG"] = "ja_JP.UTF-8"
    r = subprocess.run(
        [sys.executable, "-m", "genmodelfit", "types", "--no-cache"],
        capture_output=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert r.returncode == 0
    assert not JA.search(r.stdout.decode("utf-8"))


# --- 一覧に保存する文は言語によらず日本語 ---------------------------------------------


def test_stored_fetch_errors_are_japanese_in_en(no_pin):
    """update が一覧(sources.warnings)に残す取得の失敗の文は、表示の言語が en でも日本語(#73)。"""
    from genmodelfit import update
    from genmodelfit.fetch import FetchError

    gmf.set_language("en")

    class _Broken:
        def get_json(self, url):
            raise FetchError(url, i18n.tr("JSON ではありません", "not JSON"))

    _info, warn = update.latest_comfyui(_Broken())
    assert warn is not None and "JSON ではありません" in warn and not re.search("not JSON", warn)
    e = FetchError("https://example.invalid/x", i18n.tr("JSON ではありません", "not JSON"))
    assert str(e).startswith("not JSON") and i18n.stored_message(e).startswith("JSON ではありません")


def test_worker_threads_follow_the_caller_language(no_pin):
    """HfClient.map の仕事は、呼んだ側の言語の固定(weekly の ja など)を引き継ぐ。"""
    from genmodelfit.hf import HfClient

    client = HfClient(object(), workers=4)
    with i18n.forced("ja"):
        got = client.map(lambda _: gmf.get_language(), range(8))
    assert got == ["ja"] * 8


# --- en: 日本語の文字が出ない -----------------------------------------------------


def _stored_free_text() -> list[str]:
    """一覧・実測に保存された自由記述(訳さないもの)。長いものから外す。"""
    texts: set[str] = set()
    db = gmf.load_db()
    for m in db.models:
        d = m.to_dict()
        if isinstance(d.get("review"), dict) and d["review"].get("note"):
            texts.add(d["review"]["note"])
        texts.update(d.get("license_evidence") or [])
        for f in d.get("files") or []:
            texts.update(f.get("license_evidence") or [])
        if isinstance(d.get("description"), dict) and d["description"].get("ja"):
            texts.add(d["description"]["ja"])
    for ms in db.measurements:
        for key in ("source", "note"):
            v = getattr(ms, key, None)
            if v:
                texts.add(v)
    for f in (FIXTURES / "contrib").rglob("*.json"):  # 提出の自由記述(試験用の提出の note など)
        sub = json.loads(f.read_text(encoding="utf-8"))
        for row in [sub, sub.get("measurement") or {}, *(sub.get("measurements") or [])]:
            for key in ("source", "note"):
                if isinstance(row.get(key), str):
                    texts.add(row[key])
    return sorted(texts, key=len, reverse=True)


FREE_TEXT = _stored_free_text()


def _strip_json(obj):
    """JSON から訳さない項目を外す: 保存された自由記述(一覧にある文そのもの)、description.ja、
    label_ja(types の日本語のラベル。docs/schema.md「Uses」でキーの名前が日本語のラベルと決まっている)。"""
    free = set(FREE_TEXT)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "label_ja" or (isinstance(v, str) and v in free):
                continue
            if k == "description" and isinstance(v, dict):
                v = {kk: vv for kk, vv in v.items() if kk != "ja"}
            out[k] = _strip_json(v)
        return out
    if isinstance(obj, list):
        return [_strip_json(v) for v in obj if not (isinstance(v, str) and v in free)]
    return obj


def _run(capsys, args: list[str]) -> tuple[int, str]:
    try:
        code = cli.main([*args, "--lang", "en"])
    except SystemExit as e:  # --help・使い方の誤り
        code = e.code if isinstance(e.code, int) else 1
    out = capsys.readouterr()
    return code, out.out + out.err


def _assert_no_japanese(text: str, label: str) -> None:
    hits = [line for line in text.splitlines() if JA.search(line)]
    assert not hits, f"{label}: {hits[:5]}"


TABLE_CASES = [
    ["system", *HW],
    ["recommend", *HW],
    ["recommend", "-m", "image", *HW],
    ["recommend", "--content", "all", "--task", "edit", *HW],
    ["fit", "-m", "image", "-n", "10", *HW],
    ["fit", "--all", "--content", "all", *HW],
    ["search", "flux", *HW],
    ["search", "zzzz-no-such-model", *HW],
    ["types", "--no-cache"],
    ["info", "Z-Image-Turbo", *HW],
    ["info", "image_sdxl_simple", *HW],
    ["info", "no-such-model-xyz", *HW],
    ["download", "x", "--list", *HW],
    ["download", "Z-Image-Turbo", "--list", *HW],
    ["gate", "--no-cache"],
    ["gate", "--no-cache", "--only-held"],
    ["review", "list", "--no-cache"],
    ["diff"],
    ["diff", *DIFF_PAIR],
    ["contrib", "validate", str(FIXTURES / "contrib")],
    ["contrib", "show", str(FIXTURES / "contrib" / "rtx-3070" / "2026-10-20-5bedb2da.json")],
    ["contrib", "aggregate", str(FIXTURES / "contrib"), "--report"],
    ["recommend", "--min-fit", "good", "--fit", "good", *HW],  # 使い方の誤り
    ["recommend", "--limit", "0", *HW],  # 型の誤り
]


@pytest.mark.parametrize("args", TABLE_CASES, ids=lambda a: " ".join(a[:3]))
def test_en_table_has_no_japanese(capsys, monkeypatch, tmp_path, args):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path))
    _code, text = _run(capsys, args)
    for t in FREE_TEXT:
        text = text.replace(t, "")
    _assert_no_japanese(text, " ".join(args))


JSON_CASES = [
    ["system", *HW],
    ["recommend", "--content", "all", "--all-workflows", "--limits", *HW],
    ["fit", "--all", "--content", "all", *HW],
    ["search", "wan", *HW],
    ["types", "--no-cache"],
    ["info", "Z-Image-Turbo", *HW],
    ["info", "x", *HW],  # 候補が複数
    ["info", "no-such-model-xyz", *HW],
    ["download", "x", "--list", *HW],
    ["download", "Z-Image-Turbo", "--list", *HW],
    ["gate", "--no-cache"],
    ["review", "list", "--no-cache", "--all"],
    ["diff", *DIFF_PAIR],
    ["contrib", "validate", str(FIXTURES / "contrib")],
]


@pytest.mark.parametrize("args", JSON_CASES, ids=lambda a: " ".join(a[:3]))
def test_en_json_has_no_japanese(capsys, monkeypatch, tmp_path, args):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path))
    _code, text = _run(capsys, [*args, "--json"])
    data = json.loads(text)
    _assert_no_japanese(json.dumps(_strip_json(data), ensure_ascii=False, indent=1), " ".join(args))


HELP_CASES = [
    [],
    ["system"],
    ["recommend"],
    ["fit"],
    ["search"],
    ["types"],
    ["info"],
    ["update"],
    ["download"],
    ["gate"],
    ["diff"],
    ["review"],
    ["review", "list"],
    ["review", "approve"],
    ["review", "reject"],
    ["bench"],
    ["contrib"],
    ["contrib", "make"],
    ["contrib", "show"],
    ["contrib", "validate"],
    ["contrib", "aggregate"],
]


@pytest.mark.parametrize("args", HELP_CASES, ids=lambda a: " ".join(a) or "top")
def test_en_help_has_no_japanese(capsys, args):
    code, text = _run(capsys, [*args, "--help"])
    assert code == 0 and text.startswith("usage: genmodelfit")
    _assert_no_japanese(text, " ".join([*args, "--help"]))


def test_en_library_texts(no_pin):
    """Python から: set_language("en") で理由・注意・gate の detail が英語になる。"""
    gmf.set_language("en")
    s = gmf.system(vram_gb=8, ram_gb=32)
    for f in gmf.recommend(system_info=s, content="all", min_fit="too_tight", limit=None, all_workflows=True):
        d = f.to_dict()
        texts = [*d.get("notes", []), *(d.get("held_reasons") or [])]
        texts += [c["detail"] for c in (d.get("gate") or {}).get("checks", [])]
        texts += list((d.get("estimate") or {}).get("reasons", []))
        for t in texts:
            assert not JA.search(t), (f.model.id, t)


# --- ja: 今と同じ文 ---------------------------------------------------------------


def test_ja_keeps_current_text(capsys):
    assert cli.main(["recommend", *HW]) == 0
    out = capsys.readouterr().out
    assert "== image (画像)" in out
    assert cli.main(["types", "--no-cache"]) == 0
    assert "画像" in capsys.readouterr().out
    assert cli.main(["info", "no-such-model-xyz", *HW]) == 1
    assert JA.search(capsys.readouterr().err)


def test_ja_and_en_json_differ_only_in_text(capsys):
    """言語で変わるのは文だけ: ID・段階・点数・並びは同じ。"""
    args = ["recommend", "--content", "all", "--json", *HW]
    cli.main([*args, "--lang", "ja"])
    ja = json.loads(capsys.readouterr().out)
    cli.main([*args, "--lang", "en"])
    en = json.loads(capsys.readouterr().out)
    assert [(m["id"], m["fit_level"], m["score"], m["run_mode"]) for m in ja["models"]] == [
        (m["id"], m["fit_level"], m["score"], m["run_mode"]) for m in en["models"]
    ]
    assert set(ja) == set(en)
    assert [sorted(m) for m in ja["models"]] == [sorted(m) for m in en["models"]]


# --- 説明は両方の言語で持つ -------------------------------------------------------


def _descriptions():
    data = Path(gmf.__file__).parent / "data"
    for m in json.loads((data / "db.json").read_text(encoding="utf-8"))["models"]:
        yield "db.json", m["id"], m.get("description")
    for m in json.loads((data / "manual.json").read_text(encoding="utf-8"))["models"]:
        yield "manual.json", m["id"], m.get("description")
    for mid, m in json.loads((data / "overrides.json").read_text(encoding="utf-8"))["models"].items():
        yield "overrides.json", mid, m.get("description")


def test_every_description_has_both_languages():
    """description を持つ項目は ja と en の両方が空でない(片方の言語の利用者だけ欠けないように。#73)。"""
    checked = 0
    for where, mid, d in _descriptions():
        if not d:  # 説明なし(None・空の {})
            continue
        checked += 1
        assert isinstance(d, dict), (where, mid)
        assert (d.get("ja") or "").strip() and (d.get("en") or "").strip(), (where, mid, d)
        assert JA.search(d["ja"]) and not JA.search(d["en"]), (where, mid)
    assert checked > 0
