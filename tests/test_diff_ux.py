# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit diff` の表示を、利用者が使える形で・変化を黙って落とさずに出すことの試験(#59)。

本物の一覧は tests/fixtures/diff の、0.6.0 と 0.7.0 の同梱の一覧を縮めたもの
(tools/make_diff_fixture.py で作る。元は 1 つ約 1.8 MB)。変わったモデル(全部)・変わっていないモデル・
前から HEAD が失敗しているモデルを残してある。
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pytest
from rich.cells import cell_len

from genmodelfit import cli, weekly
from genmodelfit.db import Database, load
from genmodelfit.diff import (
    SAFE_ENTRY_FIELDS,
    SAFE_FILE_FIELDS,
    SAFE_TEMPLATE_FIELDS,
    diff_db,
    entry_changes,
    is_safe_path,
)
from genmodelfit.models import ModelEntry, ModelFile, Review, Template
from genmodelfit.table import wrap_line

FIX = Path(__file__).parent / "fixtures" / "diff"
OLD, NEW = FIX / "db_0.6.0.json", FIX / "db_0.7.0.json"
SFX = "hf:stabilityai/stable-audio-3-small-sfx"
MUSIC = ("audio_ace_step_1_t2a_instrumentals", "audio_ace_step_1_t2a_song", "audio_yue2_text2music")
MAX_COLS = 100


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """手元の update のキャッシュを使わない(引数なしの diff はキャッシュを読む)。"""
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("COLUMNS", raising=False)


def _run(capsys, *args: str) -> tuple[int, str]:
    code = cli.main(["diff", *args])
    return code, capsys.readouterr().out


def _width_ok(text: str) -> None:
    for line in text.splitlines():
        assert cell_len(line) <= MAX_COLS, line


# ---- 本物の一覧(0.6.0 → 0.7.0) ----


def test_real_default_shows_languages_and_component(capsys):
    code, out = _run(capsys, str(OLD), str(NEW))
    assert code == 0
    _width_ok(out)
    # 音楽 3 件の言語(前 → 後)
    assert "ACE-Step v1 Text to Instrumentals Music (music) — audio_ace_step_1_t2a_instrumentals" in out
    assert "言語(指示) en, zh, de, fr, es, it, pt, pl, tr, ru, cs, nl, ar, ja, hu, ko, hi → en" in out
    assert "言語(読み上げ) なし → en, zh, ru, es, ja, de, fr, pt, it, ko" in out
    assert "YuE2: Text to Music (music) — audio_yue2_text2music" in out
    assert "言語(指示) zh, en → en" in out
    # 効果音の部品(T5Gemma)とそのライセンス、一式の大きさ
    assert f"stable-audio-3-small-sfx (sfx) — {SFX}" in out
    assert "部品 + text_encoders/t5gemma_b_b_ul2.safetensors(1.11 GiB、ライセンス gemma)" in out
    assert "大きさ 2.11 → 3.22 GiB" in out
    assert "変わった 4" in out
    # 前からの HEAD の失敗と sources の数は既定では出さない(数だけ --all へ案内する)
    assert "huggingface.co/sesame/csm-1b" not in out and "401 " not in out
    assert "huggingface-search" not in out and "rate_limit" not in out
    assert "前からの HEAD の失敗" in out and "--all で" in out
    assert "変化なし" not in out


def test_real_all_shows_maintenance(capsys):
    _, default = _run(capsys, str(OLD), str(NEW))
    code, out = _run(capsys, str(OLD), str(NEW), "--all")
    assert code == 0
    _width_ok(out)
    assert "HEAD が失敗した URL(! = 新しく):" in out and "401 https://huggingface.co/sesame/csm-1b" in out
    assert "sources の数の変化:" in out and "huggingface.rate_limited: 8 -> 7" in out
    assert "人気の数・更新日・HEAD の結果などだけの変化:" in out
    assert "言語(指示) zh, en → en" in out  # 既定の内容も全部入る
    assert len(out.splitlines()) > len(default.splitlines())


def test_real_merge_check_and_diff_agree(capsys):
    """merge-check(classify)が判断を求めるモデルと、diff が「変わったモデル」に出すモデルが同じ。"""
    r = weekly.merge_check(OLD, NEW)
    raws = tuple(json.loads(p.read_text(encoding="utf-8")) for p in (OLD, NEW))
    d = diff_db(load(OLD), load(NEW), raw=raws)
    judged = {x["id"] for x in r["reasons"]}
    assert judged == {*MUSIC, SFX}
    assert {c["id"] for c in d["model_changes"]} == judged
    assert d["field_changes"][SFX][0] == "files" and "languages" in d["field_changes"][MUSIC[0]]
    # Database から作った dict でも同じ
    assert {c["id"] for c in diff_db(load(OLD), load(NEW))["model_changes"]} == judged


def test_real_json_has_new_keys(capsys):
    code, out = _run(capsys, str(OLD), str(NEW), "--json")
    d = json.loads(out)
    assert code == 0 and d["old_label"] == str(OLD) and d["new_label"] == str(NEW)
    by = {c["id"]: c for c in d["model_changes"]}
    added = [it for it in by[SFX]["items"] if it["kind"] == "file_added"]
    assert added == [
        {
            "kind": "file_added",
            "name": "t5gemma_b_b_ul2.safetensors",
            "directory": "text_encoders",
            "size_bytes": 1187264003,
            "license": "gemma",
            "role": "text_encoder",
        }
    ]
    song = {it["key"]: it for it in by[MUSIC[1]]["items"] if it["kind"] == "languages"}
    assert song["prompt"]["new"] == ["en"] and song["speech"]["old"] is None


# ---- 黙って落とさない: ModelEntry の項目を 1 つずつ変える ----

# 専用の行に出る項目と、その行の言葉(ほかの項目は「その他の変化: <項目>」に出る)
LABELS = {
    "name": "名前",
    "modality": "種類",
    "tasks": "用途",
    "runtime": "動かし方",
    "content": "内容",
    "precision": "精度",
    "license": "ライセンス",
    "license_name": "ライセンス",
    "languages": "言語(",
    "files": "部品 ",
    "review": "確認の印",
    "files.license": "のライセンス",
    "files.license_name": "のライセンス",
    "files.name": "部品 ",
    "files.size_bytes": "大きさ",
}


def _changed(v: Any, ann: str) -> Any:
    """v と違う値(型は注釈に合わせる)。知らない型は試験を落とす(新しい項目に試験を足すため)。"""
    if isinstance(v, bool) or (v is None and ann.startswith("bool")):
        return not v
    if isinstance(v, int) or (v is None and ann.startswith("int")):
        return (v or 0) + 123_456_789
    if isinstance(v, float) or (v is None and ann.startswith("float")):
        return (v or 0.0) + 1.5
    if isinstance(v, str) or (v is None and ann.startswith("str")):
        return (v or "") + "-changed"
    if ann.startswith("list[list"):
        return [*(v or []), ["x"]]
    if isinstance(v, list) or (v is None and ann.startswith("list")):
        return [*(v or []), "x-changed"]
    if isinstance(v, dict) or (v is None and ann.startswith("dict")):
        return {**(v or {}), "zz": "x-changed" if ann.startswith("dict[str, str]") else ["x-changed"]}
    raise AssertionError(f"変え方を知らない項目の型です(試験を足してください): {ann} = {v!r}")


def _base_pair() -> tuple[Database, Database]:
    db = load(NEW)
    return copy.deepcopy(db), copy.deepcopy(db)


def _entry_with_template(db: Database) -> ModelEntry:
    return next(m for m in db.models if m.template is not None and m.files and m.files[0].size_bytes)


def _cases() -> list[tuple[str, str]]:
    """(どこを, 項目名)。ModelEntry・ModelFile・Template・Review の全部の項目。"""
    out = [("entry", f.name) for f in dataclasses.fields(ModelEntry) if f.name != "id"]
    out += [("file", f.name) for f in dataclasses.fields(ModelFile)]
    out += [("template", f.name) for f in dataclasses.fields(Template)]
    out += [("review", f.name) for f in dataclasses.fields(Review)]
    return out


def _mutate(where: str, name: str, m: ModelEntry) -> str:
    """m の 1 つの項目を変え、entry_changes での名前を返す。"""
    if where == "entry":
        ann = str(next(f.type for f in dataclasses.fields(ModelEntry) if f.name == name))
        if name == "files":
            m.files = [*m.files, ModelFile(name="extra-x.safetensors", size_bytes=2**30, license="mit")]
        elif name == "template":
            m.template = None
        elif name == "review":
            m.review = Review(status="pending", reasons=["new"])
        else:
            setattr(m, name, _changed(getattr(m, name), ann))
        return name
    if where == "file":
        f = m.files[0]
        ann = str(next(x.type for x in dataclasses.fields(ModelFile) if x.name == name))
        setattr(f, name, _changed(getattr(f, name), ann))
        return f"files.{name}"
    if where == "template":
        t = m.template
        assert t is not None
        ann = str(next(x.type for x in dataclasses.fields(Template) if x.name == name))
        setattr(t, name, _changed(getattr(t, name), ann))
        return f"template.{name}"
    rv = m.review = m.review or Review(status="ok", reasons=["baseline"], decided_by="x")
    ann = str(next(x.type for x in dataclasses.fields(Review) if x.name == name))
    if name == "status":
        rv.status = "pending" if rv.status == "ok" else "ok"
    else:
        setattr(rv, name, _changed(getattr(rv, name), ann))
    return "review"


def _safe(path: str) -> bool:
    return is_safe_path(path)


@pytest.mark.parametrize(("where", "name"), _cases())
def test_no_silent_change(where, name):
    old, new = _base_pair()
    i = _entry_with_template(new).id
    if where == "review":  # 確認の印の中身を変えるには、両方に同じ印を付けてから
        for db in (old, new):
            db.get(i).review = Review(status="ok", reasons=["baseline"], decided_by="x")
    path = _mutate(where, name, new.get(i))
    d = diff_db(old, new)
    paths = d["field_changes"].get(i, [])
    assert path in paths or (where == "file" and name == "name" and "files" in paths), (path, paths)
    out, out_all = _render(old, new), _render(old, new, show_all=True)
    _width_ok(out)
    _width_ok(out_all)
    if all(_safe(p) for p in paths):
        # 人気の数・更新日・HEAD の結果など: 既定では数だけ、--all で ID と項目名
        assert "人気の数などだけの変化 1 件" in out, out
        assert f"{i}: {', '.join(paths)}" in out_all
        return
    block = _block(out, i)
    assert block, f"{i} が「変わったモデル」に出ていません:\n{out}"
    label = LABELS.get(path) or LABELS.get(path.split(".")[0])
    if label is None or (path.startswith("template.") and "大きさ" not in block):
        assert "その他の変化:" in block and path in block, (path, block)
    else:
        assert label in block or path in block, (path, block)
    # 分類(merge-check と同じ)が判断を求めるモデルは、必ず「変わったモデル」に出る
    judged = {x["id"] for x in d["classification"]["judgement_needed"]}
    assert judged <= {c["id"] for c in d["model_changes"]}


def test_every_field_has_a_case():
    """ModelEntry などに項目を足したら、この試験のどこかで扱う(黙って落とさない試験を広げる)。"""
    names = {(w, n) for w, n in _cases()}
    assert ("entry", "languages") in names and ("file", "license") in names
    assert {f.name for f in dataclasses.fields(ModelEntry)} - {"id"} <= {n for w, n in names if w == "entry"}
    # 安全の表に、知らない項目の名前が無い(綴りの間違いで、変化を「安全」として隠さない)
    assert {f.name for f in dataclasses.fields(ModelEntry)} >= SAFE_ENTRY_FIELDS
    assert {f.name for f in dataclasses.fields(ModelFile)} >= SAFE_FILE_FIELDS
    assert {f.name for f in dataclasses.fields(Template)} >= SAFE_TEMPLATE_FIELDS


def _block(out: str, model_id: str) -> str:
    """「変わったモデル」の節で、そのモデルの見出しから次の見出しまで。"""
    lines = out.splitlines()
    for k, line in enumerate(lines):
        if line.startswith("  ") and not line.startswith("   ") and line.endswith(f"— {model_id}"):
            body = [line]
            for nxt in lines[k + 1 :]:
                if not nxt.startswith("    "):
                    break
                body.append(nxt)
            return "\n".join(body)
    return ""


def _run_dbs(capsys, old: Database, new: Database, *extra: str) -> tuple[int, str]:
    """2 つの一覧をファイルに書いて CLI で比べる(書いて読める値だけで変えたとき)。"""
    tmp = Path(tempfile.mkdtemp())
    try:
        a, b = tmp / "old.json", tmp / "new.json"
        old.save(a)
        new.save(b)
        return _run(capsys, str(a), str(b), *extra)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _render(old: Database, new: Database, *, show_all: bool = False, limit: int = 20) -> str:
    """CLI と同じ表示(幅 100 で折り返す)。読むときの型の検査を通らない値に変えた一覧にも使う。"""
    lines = cli._diff_lines(diff_db(old, new), limit, show_all=show_all)
    return "".join(part + "\n" for line in lines for part in wrap_line(line, MAX_COLS))


def test_entry_changes_duplicate_file_names():
    """同じ名前のファイルが 2 つある項目(URL だけ違う)でも、取り違えて変化を作らない。"""
    f1 = {"name": "t5.safetensors", "url": "https://a/x?download=true"}
    f2 = {"name": "t5.safetensors", "url": "https://a/x"}
    a = {"id": "m", "files": [f1, f2]}
    assert entry_changes(a, copy.deepcopy(a)) == []
    b = copy.deepcopy(a)
    b["files"][1]["url"] = "https://a/y"
    assert entry_changes(a, b) == ["files.url"]


def test_removed_duplicate_file_is_shown(capsys):
    old, new = _base_pair()
    m = _entry_with_template(new)
    m.files.append(ModelFile(name=m.files[0].name, url="https://example.invalid/dup", size_bytes=1))
    old_m = old.get(m.id)
    old_m.files.append(ModelFile(name=m.files[0].name, url="https://example.invalid/dup", size_bytes=1))
    m.files.pop()  # 同じ名前の 2 つ目だけ消す
    d = diff_db(old, new)
    items = next(c for c in d["model_changes"] if c["id"] == m.id)["items"]
    assert [it["kind"] for it in items if it["kind"].startswith("file")] == ["file_removed"]


# ---- 引数の形 ----


def test_no_args_without_cache_prints_howto(capsys, tmp_path):
    code, out = _run(capsys)
    assert code == 0
    assert "genmodelfit update --from-release" in out and "Traceback" not in out
    _width_ok(out)
    code, out = _run(capsys, "--json")
    d = json.loads(out)
    assert code == 0 and d["command"] == "diff" and "update --from-release" in d["note"]
    assert "error" not in d


def test_no_args_uses_cache(capsys, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    bundled = load(None, use_cache=False)
    fetched = copy.deepcopy(bundled)
    m = fetched.models[0]
    m.languages = {**m.languages, "prompt": ["xx"]}
    fetched.save(cache / "db.json")
    code, out = _run(capsys)
    assert code == 0
    _width_ok(out)
    lines = out.splitlines()
    assert lines[0].startswith("前: 同梱の一覧(genmodelfit ")
    assert lines[1].startswith("後: 取ってきた一覧(") and str(cache / "db.json") in out.replace("\n    ", "")
    assert f"{bundled.generated_at}  {len(bundled.models)} 件" in lines[0]
    assert _block(out, m.id) and "言語(指示)" in _block(out, m.id)
    d = json.loads(_run(capsys, "--json")[1])
    assert d["old_label"].startswith("同梱の一覧") and d["new_label"].startswith("取ってきた一覧")


def test_one_arg_compares_with_bundled(capsys):
    code, out = _run(capsys, str(NEW))
    assert code == 0
    _width_ok(out)
    assert out.splitlines()[0].startswith("前: 同梱の一覧(genmodelfit ")
    d = json.loads(_run(capsys, str(NEW), "--json")[1])
    assert d["old_label"].startswith("同梱の一覧") and d["new_label"] == str(NEW)
    assert d["new"]["models"] == len(load(NEW).models)


def test_two_args_header(capsys):
    _, out = _run(capsys, str(OLD), str(NEW))
    head, _ = out.split("\n増えた ", 1)
    old_part, new_part = head.split("\n後: ", 1)  # 長いパスは折り返されるので、空白を除いて比べる

    def flat(t: str) -> str:
        return "".join(t.split())

    assert flat(old_part) == flat(f"前: {OLD} 2026-10-20T22:40:19Z 10 件 ComfyUI v0.42.1")
    assert flat(new_part) == flat(f"{NEW} 2026-10-21T06:20:18Z 10 件 ComfyUI v0.42.1")


def test_unchanged_lists_say_no_change(capsys):
    code, out = _run(capsys, str(NEW), str(NEW))
    assert code == 0
    assert "変化なし(利用者に関係のある変化はありません)" in out
    assert "変わったモデル" not in out
    code, out = _run(capsys, str(NEW), str(NEW), "--all")
    assert code == 0 and "変化なし" not in out and "HEAD が失敗した URL" in out


def test_limit_applies_to_changed_models(capsys):
    old, new = _base_pair()
    for m in new.models:
        m.runtime = "x-changed"
    code, out = _run_dbs(capsys, old, new, "-n", "2")
    assert code == 0 and "…ほか" in out
    assert out.count("動かし方") == 2


def test_long_values_fit_100_columns(capsys):
    old, new = _base_pair()
    m = new.models[0]
    m.name = "とても長い名前のモデル" * 8
    m.languages = {"prompt": [f"l{k}" for k in range(80)]}
    m.files[0].license, m.files[0].license_name = "other", "a-very-long-license-name-" + "x" * 120
    for args in ((), ("--all",)):
        _, out = _run_dbs(capsys, old, new, *args)
        _width_ok(out)
        assert "とても長い名前のモデル" in out


# ---- JSON の互換と毎週の PR の本文 ----

OLD_KEYS = {
    "old": dict,
    "new": dict,
    "policy": dict,
    "summary": dict,
    "added": list,
    "removed": list,
    "license_changes": list,
    "auto_changes": list,
    "size_changes": list,
    "precision_changes": list,
    "newly_uncensored": list,
    "head_failures": list,
    "unauthorized": dict,
    "partial_counts": (dict, type(None)),
    "stats": list,
    "classification": dict,
}


def test_json_keeps_old_keys_and_types():
    d = diff_db(load(OLD), load(NEW))
    for k, t in OLD_KEYS.items():
        assert isinstance(d[k], t), k
    assert set(d) == set(OLD_KEYS) | {"old_label", "new_label", "field_changes", "model_changes"}
    assert d["summary"]["size_changed"] == 1 and d["size_changes"][0]["id"] == SFX
    assert d["summary"]["head_failures"] > 0 and d["summary"]["head_failures_new"] == 0
    assert set(d["classification"]) == {"safe", "judgement_needed", "safe_changes"}
    assert isinstance(d["field_changes"], dict) and isinstance(d["model_changes"], list)


def test_weekly_pr_body_ignores_new_keys():
    """毎週の PR の本文は今までの鍵だけから作る(足した鍵で中身が変わらない)。"""
    d = diff_db(load(OLD), load(NEW))
    result = {"status": {}, "ok": True, "problems": []}
    body = weekly.pr_body(d, result)
    trimmed = {k: v for k, v in d.items() if k in OLD_KEYS}
    assert weekly.pr_body(trimmed, result) == body
    assert "判断が要る変化: 4 件" in body  # 分類は今までどおり(音楽 3 件の言語・効果音の部品)
