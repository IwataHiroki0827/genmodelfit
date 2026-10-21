# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit diff` の表示を、利用者が使える形で・変化を黙って落とさずに出すことの試験(#59)。

本物の一覧は tests/fixtures/diff/<前>-<後>/ の、過去の版の同梱の一覧を縮めたもの
(tools/make_diff_fixture.py で作る。元は 1 つ約 1.7 MB)。大事な変化のモデル(全部)・確認の印などだけ
変わったモデル(既定の --limit より多く)・変わっていないモデル・前から HEAD が失敗しているモデルを残してある。
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
from genmodelfit.db import Database, bundled_text, load
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
OLD, NEW = FIX / "0.6.0-0.7.0" / "old.json", FIX / "0.6.0-0.7.0" / "new.json"
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


def _has(out: str, text: str) -> bool:
    """折り返しを気にせずに含むか(空白を除いて比べる)。"""
    return "".join(text.split()) in "".join(out.split())


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
    part = "部品 + text_encoders/t5gemma_b_b_ul2.safetensors"
    assert _has(out, part + "(1.11 GiB、ライセンス gemma、Comfy-Org/stable-audio-3)")
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
            "source": "Comfy-Org/stable-audio-3",
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
    "comfy_family": "ComfyUI の系統",
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
    "files.directory": "部品 ",
    "files.url": "部品 ",
    "files.sha256": "部品 ",
    "files.repo_id": "部品 ",
    "files.gated": "部品 ",
    "files.role": "部品 ",
    "license_url": "ライセンスの出どころ",
    "files.license_url": "ライセンスの出どころ",
    "files.license_from": "ライセンスの出どころ",
    "files.license_path": "ライセンスの出どころ",
    "files.repo_license": "ライセンスの出どころ",
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
    # 名前・置き場所はファイルを組む鍵なので、変えると「消えた + 増えた」になる
    assert path in paths or (where == "file" and name in ("name", "directory") and "files" in paths), (
        path,
        paths,
    )
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
    assert set(d) == set(OLD_KEYS) | {
        "old_label",
        "new_label",
        "field_changes",
        "model_changes",
        "top_changes",
    }
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


# ---- 大事な変化の順・まとめ(本物の 0.5.0 / 0.4.0 → 0.7.0) ----


def _pair_paths(name: str) -> tuple[Path, Path]:
    return FIX / name / "old.json", FIX / name / "new.json"


@pytest.mark.parametrize("pair", ["0.5.0-0.7.0", "0.4.0-0.7.0"])
def test_real_important_change_is_not_buried(pair, capsys):
    """持ち主の指摘(#59): ID の順だと T5Gemma(Gemma 規約)の追加が 20 件の後ろに隠れていた。"""
    old, new = _pair_paths(pair)
    d = diff_db(load(old), load(new))
    before = [c["id"] for c in d["model_changes"] if c["id"] < SFX]
    assert len(before) > cli.DIFF_DEFAULT_LIMIT  # ID の順に並べると隠れる一覧であること(試験の前提)
    code, out = _run(capsys, str(old), str(new))
    assert code == 0
    _width_ok(out)
    assert _has(out, "部品 + text_encoders/t5gemma_b_b_ul2.safetensors(1.11 GiB、ライセンス gemma")
    # 大事な変化(部品)のモデルが、変わったモデルの節の先頭
    section = out.split("変わったモデル(大事な変化の順):\n", 1)[1]
    assert section.splitlines()[0].endswith(f"— {SFX}")
    # 上限 1 でも、部品の変化は隠れない。隠れたことを示す行は -n と --all を案内する
    _, one = _run(capsys, str(old), str(new), "-n", "1")
    assert "t5gemma" in one and "…ほか" in one and "--all で全部" in one and "-n " in one


def test_real_review_marks_are_collapsed(capsys):
    old, new = _pair_paths("0.5.0-0.7.0")
    _, out = _run(capsys, str(old), str(new))
    n = sum(1 for m in load(new).models if m.review is not None)
    assert f"確認の印が付いた(ok): {n} 件(ID は --all)" in out
    assert "のまま" not in out  # 印が無かったものに「ok のまま」とは書かない
    _, out_all = _run(capsys, str(old), str(new), "--all")
    assert f"確認の印が付いた(ok): {n} 件\n" in out_all and "3d_hunyuan3d-v2.1" in out_all
    assert "…ほか" not in out_all  # --all は上限なし


def test_limit_never_hides_more_important():
    """どの上限でも、出した変化より大事な変化を隠さない(並べる鍵の順に切る)。"""
    for pair in ("0.4.0-0.7.0", "0.5.0-0.7.0"):
        old, new = _pair_paths(pair)
        entries = cli._diff_changed_entries(diff_db(load(old), load(new)), False)
        ranks = [k[0] for k, _ in entries]
        assert ranks == sorted(ranks) and ranks[0] == 0


# ---- CLI が JSON をそのまま比べる(知らない鍵も merge-check と同じに拾う) ----


def _raw(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_raw(tmp_path: Path, name: str, data: dict[str, Any]) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return p


def _cli_and_merge(
    capsys, tmp_path, old: dict[str, Any], new: dict[str, Any]
) -> tuple[str, dict[str, Any], dict]:
    a, b = _write_raw(tmp_path, "a.json", old), _write_raw(tmp_path, "b.json", new)
    code, out = _run(capsys, str(a), str(b))
    assert code == 0
    _, js = _run(capsys, str(a), str(b), "--json")
    return out, json.loads(js), weekly.merge_check(a, b)


def test_cli_sees_unknown_keys_like_merge_check(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    m = new["models"][0]
    m["zz_future"] = "x"  # この版が知らない項目
    new["models"][1]["files"][0]["zz_file"] = 1
    new["zz_top"] = {"a": 1}
    out, d, mc = _cli_and_merge(capsys, tmp_path, old, new)
    assert _block(out, m["id"]) and "その他の変化: zz_future" in _block(out, m["id"])
    assert "files.zz_file" in _block(out, new["models"][1]["id"])
    assert "一覧の上の段で変わった鍵: zz_top" in out and d["top_changes"] == ["zz_top"]
    judged = {x["id"] for x in mc["reasons"]}
    assert judged == {c["id"] for c in d["model_changes"]} | {"(database)"}
    assert {x["id"] for x in d["classification"]["judgement_needed"]} == judged


# ---- 同じ名前のファイル(URL・ライセンスだけ違う)を取り違えない ----


def _dup_files() -> list[dict[str, Any]]:
    def f(k: int, lic: str) -> dict[str, Any]:
        return {
            "name": "t5.safetensors",
            "url": f"https://huggingface.co/org{k}/repo/resolve/main/t5.safetensors",
            "sha256": f"{k:064x}",
            "size_bytes": 2**30,
            "directory": "text_encoders",
            "repo_id": f"org{k}/repo",
            "license": lic,
        }

    return [f(1, "mit"), f(2, "apache-2.0"), f(3, "bsd-3-clause")]


def _dup_items(capsys, tmp_path, change) -> tuple[list[dict[str, Any]], str, dict]:
    old = _raw(NEW)
    i = old["models"][0]["id"]
    old["models"][0]["files"] = _dup_files()
    new = copy.deepcopy(old)
    change(new["models"][0]["files"])
    out, d, mc = _cli_and_merge(capsys, tmp_path, old, new)
    items = next(c for c in d["model_changes"] if c["id"] == i)["items"]
    return items, _block(out, i), mc


@pytest.mark.parametrize("op", ["swap", "reverse"])
def test_duplicate_names_reorder_is_only_order(op, capsys, tmp_path):
    def change(fs):
        if op == "swap":
            fs[0], fs[1] = fs[1], fs[0]
        else:
            fs.reverse()

    items, block, mc = _dup_items(capsys, tmp_path, change)
    assert [it["kind"] for it in items] == ["files_order"], items
    assert "部品の並び・形が変わった" in block and " url " not in block and "ライセンス" not in block
    assert mc["safe"] is False  # 並びの変化も判断が要る(今までどおり)


def test_duplicate_names_delete_first(capsys, tmp_path):
    items, block, _ = _dup_items(capsys, tmp_path, lambda fs: fs.pop(0))
    kinds = [it["kind"] for it in items if it["kind"] != "size"]
    assert kinds == ["file_removed"], items
    gone = next(it for it in items if it["kind"] == "file_removed")
    assert gone["license"] == "mit" and gone["source"] == "org1/repo"
    assert _has(block, "部品 - text_encoders/t5.safetensors(1.00 GiB、ライセンス mit、org1/repo)")


def test_duplicate_names_insert_in_middle(capsys, tmp_path):
    def change(fs):
        extra = copy.deepcopy(fs[0])
        extra.update(
            url="https://example.org/x/t5.safetensors", sha256="f" * 64, repo_id=None, license="gemma"
        )
        fs.insert(1, extra)

    items, block, _ = _dup_items(capsys, tmp_path, change)
    kinds = [it["kind"] for it in items if it["kind"] != "size"]
    assert kinds == ["file_added"], items
    added = items[0]
    assert added["license"] == "gemma" and added["source"] == "example.org/x/t5.safetensors"
    assert "ライセンス gemma" in block and "のライセンス" not in block and " url " not in block


def test_url_and_sha256_changes_show_values(capsys, tmp_path):
    def change(fs):
        fs[1]["url"] = "https://huggingface.co/neworg/repo/resolve/main/t5.safetensors"

    def change_sha(fs):
        fs[2]["sha256"] = "a" * 64

    _, block, mc = _dup_items(capsys, tmp_path, change)
    assert _has(block, "部品 t5.safetensors の url huggingface.co/org2/repo/resolve/main/t5.safetensors →")
    assert "neworg" in block and mc["safe"] is False
    _, block, _ = _dup_items(capsys, tmp_path, change_sha)
    assert f"の sha256 {3:064x}"[:20] in block and "aaaaaaaaaaaa…" in block


# ---- 安全の表を守る(識別・ライセンスの項目を「安全」に入れたら落ちる) ----

# 安全の表(変わっても判断が要らない項目)。増やすときは、利用者に関係しないことを確かめてこの表も直す
EXPECTED_SAFE_ENTRY = {"downloads", "downloads_all_time", "likes", "updated", "discovered_from"}
EXPECTED_SAFE_FILE = {"head_status", "head_checked_at"}
EXPECTED_SAFE_TEMPLATE = {"usage"}
# ファイルがどれか・何を落とすか・どの条件で使えるかを決める項目。安全の表に入れてはいけない
# (size_bytes も入れない: 小さくなるのは need_decreased で判断が要る。大きくなるのは分類では安全だが、
#  表示には必ず出す)
NEVER_SAFE_FILE = {
    "name",
    "url",
    "sha256",
    "directory",
    "repo_id",
    "license",
    "license_name",
    "gated",
    "size_bytes",
}
NEVER_SAFE_ENTRY = {"files", "license", "license_name", "content", "tasks", "modality", "runtime", "review"}


def test_safe_tables_are_exactly_the_allowlist():
    assert set(SAFE_ENTRY_FIELDS) == EXPECTED_SAFE_ENTRY
    assert set(SAFE_FILE_FIELDS) == EXPECTED_SAFE_FILE
    assert set(SAFE_TEMPLATE_FIELDS) == EXPECTED_SAFE_TEMPLATE
    assert not NEVER_SAFE_FILE & set(SAFE_FILE_FIELDS)
    assert not NEVER_SAFE_ENTRY & set(SAFE_ENTRY_FIELDS)


_FILE_CHANGES = {
    "name": "renamed.safetensors",
    "url": "https://huggingface.co/other/repo/resolve/main/x.safetensors",
    "sha256": "b" * 64,
    "directory": "zz_other_folder",
    "repo_id": "other/repo",
    "license": "apache-2.0",
    "license_name": "some-other-name",
    "gated": True,
    "size_bytes": 1,  # 小さくなる
}


@pytest.mark.parametrize("field", sorted(_FILE_CHANGES))
def test_identity_fields_show_and_are_not_safe(field, capsys, tmp_path):
    old = _raw(NEW)
    m = next(x for x in old["models"] if x["files"] and x["files"][0].get("license") == "other")
    new = copy.deepcopy(old)
    nm = next(x for x in new["models"] if x["id"] == m["id"])
    nm["files"][0][field] = _FILE_CHANGES[field]
    out, _, mc = _cli_and_merge(capsys, tmp_path, old, new)
    assert _block(out, m["id"]), (field, out)
    assert mc["safe"] is False and m["id"] in {x["id"] for x in mc["reasons"]}


def test_size_growth_is_shown_but_safe(capsys, tmp_path):
    old = _raw(NEW)
    m = next(x for x in old["models"] if x["files"] and x["files"][0].get("size_bytes"))
    new = copy.deepcopy(old)
    next(x for x in new["models"] if x["id"] == m["id"])["files"][0]["size_bytes"] += 3 * 2**30
    out, _, mc = _cli_and_merge(capsys, tmp_path, old, new)
    assert "大きさ" in _block(out, m["id"]) and mc["safe"] is True


# ---- 表示の細かいところ ----


def test_review_mark_wording(capsys, tmp_path):
    old = _raw(NEW)
    for x in old["models"]:
        x.pop("review", None)
    new = copy.deepcopy(old)
    new["models"][0]["review"] = {"status": "pending", "reasons": ["new"], "evidence": []}
    new["models"][1]["review"] = {"status": "ok", "reasons": ["baseline"], "evidence": []}
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert "確認の印が付いた(pending)(理由: new)" in _block(out, new["models"][0]["id"])
    assert "確認の印が付いた(ok)" in _block(out, new["models"][1]["id"])
    old2, new2 = copy.deepcopy(new), copy.deepcopy(new)
    new2["models"][1].pop("review")
    new2["models"][0]["review"]["evidence"] = ["https://example.org"]
    out, _, _ = _cli_and_merge(capsys, tmp_path, old2, new2)
    assert "確認の印が外れた(前は ok)" in out and "確認の印(pending)の中身が変わった: evidence" in out


def test_empty_vs_missing_languages_not_shown(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    m = new["models"][0]
    m["languages"] = {**(m.get("languages") or {}), "zzz": []}
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert "なし → なし" not in out


def test_type_not_repeated_when_name_has_it():
    assert cli._diff_model({"id": "x", "name": "Foo (music)", "modality": "music"}) == "Foo (music) — x"
    assert cli._diff_model({"id": "x", "name": "Foo", "modality": "music"}) == "Foo (music) — x"


def test_old_cache_warns(capsys, tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    bundled = json.loads(bundled_text("db.json"))
    bundled["generated_at"] = "2000-01-01T00:00:00Z"
    (cache / "db.json").write_text(json.dumps(bundled, ensure_ascii=False), encoding="utf-8")
    code, out = _run(capsys)
    assert code == 0 and "注意:" in out and "古いです" in out and _has(out, "update --from-release")
    _width_ok(out)
    d = json.loads(_run(capsys, "--json")[1])
    assert d["warnings"] and "古いです" in d["warnings"][0]


def test_no_cache_json_has_usual_shape(capsys):
    d = json.loads(_run(capsys, "--json")[1])
    assert set(OLD_KEYS) <= set(d) and {"note", "warnings", "field_changes", "model_changes"} <= set(d)
    assert d["model_changes"] == [] and d["old"] is None and "error" not in d
    # summary は普段と同じ鍵(数は 0)
    normal = diff_db(load(NEW), load(NEW))["summary"]
    assert set(d["summary"]) == set(normal) and all(v in (0, None) for v in d["summary"].values())


def test_non_string_license_name_is_rejected(capsys, tmp_path):
    """license_name が文字でない一覧は読むときに断る(以前は diff の途中で AttributeError で落ちた)。"""
    for where in ("entry", "file"):
        bad = _raw(NEW)
        target = bad["models"][0] if where == "entry" else bad["models"][0]["files"][0]
        target["license_name"] = 123
        p = _write_raw(tmp_path, f"bad_{where}.json", bad)
        code, out = _run(capsys, str(NEW), str(p), "--json")
        assert code == 1 and "license_name" in json.loads(out)["error"]


# ---- 2 回目の検証の指摘(#59) ----


def _model(data: dict[str, Any], i: str) -> dict[str, Any]:
    return next(m for m in data["models"] if m["id"] == i)


@pytest.mark.parametrize(
    ("field", "old", "new"),
    [
        (
            "url",
            "https://huggingface.co/org/repo/resolve/main/split_files/diffusion_models/model_fp16.safetensors",
            "https://huggingface.co/org/repo/resolve/0123abc/split_files/diffusion_models/model_fp16.safetensors",
        ),
        ("sha256", "a" * 30 + "b" + "c" * 33, "a" * 30 + "d" + "c" * 33),
        ("sha256", "1" * 64, "2" * 64),
        (
            "license_path",
            ["org/a-very-long-repository-name-x", "base/model-" + "y" * 50],
            ["org/a-very-long-repository-name-x", "base/model-" + "z" * 50],
        ),
    ],
)
def test_shortened_values_differ(field, old, new):
    """縮めた前と後が同じ文字列にならない(真ん中だけ違う URL・sha256 でも)。"""
    a, b = cli._diff_pair(field, old, new)
    assert a != b and len(a) < 90 and len(b) < 90


def test_url_middle_change_via_cli(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    f = _model(new, SFX)["files"][0]
    f["url"] = f["url"].replace("/resolve/main/", "/resolve/0123abcd/")
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    flat = " ".join(x.strip() for x in _block(out, SFX).splitlines())  # 折り返しを戻す
    a, b = flat.split(" の url ", 1)[1].split(" ")[0], flat.split(" → ", 1)[1].split(" ")[0]
    assert a != b and "main" in a and "0123abcd" in b


def test_permissive_and_uncensored_marks(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    m = _model(new, "audio_yue2_text2music")  # cc-by-nc-4.0 → apache-2.0(寛容な側)
    m["license"] = "apache-2.0"
    for f in m["files"]:
        f["license"] = "apache-2.0"
    _model(new, MUSIC[0])["content"] = "uncensored"
    _model(new, MUSIC[0])["content_evidence"] = ["name:nsfw"]
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert "ライセンス cc-by-nc-4.0 → apache-2.0  ! 寛容な側(原文で確かめる)" in _block(
        out, "audio_yue2_text2music"
    )
    assert "内容 sfw → uncensored  ! 新しく無検閲(name:nsfw)" in _block(out, MUSIC[0])


def _first_block_ids(out: str) -> list[str]:
    section = out.split("変わったモデル(大事な変化の順):\n", 1)[1]
    return [ln.rsplit("— ", 1)[1] for ln in section.splitlines() if ln.startswith("  ") and "— " in ln]


@pytest.mark.parametrize("kind", ["license", "files_order", "gated", "license_source"])
def test_important_kinds_come_before_languages(kind, capsys, tmp_path):
    """ライセンス・部品の変化は、ID が後ろでも、言語だけ変わったモデル(ID が前)より先に出る。"""
    old = _raw(NEW)
    new = copy.deepcopy(old)
    _model(new, "3d_hunyuan3d-v2.1")["languages"] = {"prompt": ["xx"]}
    m = _model(new, "hf:sesame/csm-1b")
    if kind == "license":
        m["license"] = "mit"
    elif kind == "files_order":
        m["files"].reverse()
    elif kind == "gated":
        m["files"][0]["gated"] = not m["files"][0]["gated"]
    else:
        m["files"][0]["license_from"] = "someone/else"
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    ids = _first_block_ids(out)
    assert ids.index("hf:sesame/csm-1b") < ids.index("3d_hunyuan3d-v2.1"), out
    if kind == "gated":
        assert "部品 transformers-00001-of-00002.safetensors の gated True → False" in out
    if kind == "license_source":
        assert "ライセンスの出どころ 部品" in out and "license_from" in out and "someone/else" in out


def test_license_and_file_lines_are_never_collapsed(capsys, tmp_path):
    """同じライセンスの変化・部品の追加が多くのモデルにあっても、モデルごとに出す(ID を隠さない)。"""
    old = _raw(NEW)
    new = copy.deepcopy(old)
    targets = [m["id"] for m in new["models"]][:5]
    for i in targets:
        m = _model(new, i)
        m["license"] = "mit-zz"
        m["files"].append({"name": "extra.safetensors", "size_bytes": 10, "license": "mit"})
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    for i in targets:
        block = _block(out, i)
        assert "ライセンス" in block and "部品 + extra.safetensors" in block, (i, out)


def test_collapsed_small_group_lists_ids(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    targets = [m["id"] for m in new["models"]][:4]
    for i in targets:
        m = _model(new, i)
        m["languages"] = {**(m.get("languages") or {}), "zz": ["xx"]}
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    lines = out.splitlines()
    k = lines.index("  言語(zz) なし → xx: 4 件")  # 「(ID は --all)」にしない(8 件まではその場に出す)
    listed = " ".join(lines[k + 1 : k + 3])
    assert all(i in listed for i in targets), out


def test_long_language_list_is_truncated_by_default(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    _model(new, MUSIC[0])["languages"] = {"prompt": [f"l{k}" for k in range(40)]}
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert _has(out, "…ほか 24(全 40)") and "l39" not in out
    a, b = _write_raw(tmp_path, "a.json", old), _write_raw(tmp_path, "b.json", new)
    _, out_all = _run(capsys, str(a), str(b), "--all")
    assert "l39" in out_all and "…ほか 24" not in out_all


def test_auto_group_lists_ids_without_all(capsys):
    """自動のおすすめの変化(判断が要る)は、まとめても ID を --all に隠さない(-n で切る)。"""
    for a, b in (("0.1.0", "0.7.1"), ("0.3.0", "0.4.0")):
        d = diff_db(load(_pair_paths(f"{a}-{b}")[0]), load(_pair_paths(f"{a}-{b}")[1]))
        gained = [
            c["id"] for c in d["auto_changes"] if c["new"] and not c["old"] and c["id"] in d["field_changes"]
        ]
        lines = cli._diff_lines(d, 20)
        text = "\n".join(lines)
        group = [ln for ln in lines if ln.startswith("  自動のおすすめ") and "件" in ln]
        for g in group:
            assert "(ID は --all)" not in g
        assert all(i in text for i in gained[:3]), text
    # -n で切ると「…ほか N 件(-n で増やす)」
    d = diff_db(load(_pair_paths("0.1.0-0.7.1")[0]), load(_pair_paths("0.1.0-0.7.1")[1]))
    entries = cli._diff_changed_entries(d, False, limit=2)
    group = next(b for _, b in entries if b[0].startswith("  自動のおすすめ 保留 → 出す"))
    assert "(ID は --all)" not in group[0] and group[-1].startswith("    …ほか 14 件(-n 16 で増やす)"), group


def test_other_changes_grouped_per_field():
    d = diff_db(load(_pair_paths("0.1.0-0.7.1")[0]), load(_pair_paths("0.1.0-0.7.1")[1]))
    lines = cli._diff_lines(d, None)
    groups = [ln for ln in lines if ln.startswith("  その他の変化: ") and " 件" in ln]
    assert groups and all(", " not in ln.split(": ", 1)[1].split(":")[0] for ln in groups), groups


def test_null_valued_new_file_key_is_no_change(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    for f in _model(new, SFX)["files"]:
        f["zz_new_key"] = None
    out, d, mc = _cli_and_merge(capsys, tmp_path, old, new)
    assert "部品の並び・形が変わった" not in out and SFX not in {c["id"] for c in d["model_changes"]}
    assert mc["safe"] is True


def test_long_id_is_never_split(capsys, tmp_path):
    old = _raw(NEW)
    new = copy.deepcopy(old)
    long_id = "hf:darknight9121/FLUX.2-klein-base-9B-bucket-uncensored-with-a-much-longer-name"
    for d in (old, new):
        _model(d, MUSIC[0])["id"] = long_id
        _model(d, long_id)["name"] = "とても長い名前のモデル" * 3
    _model(new, long_id)["runtime"] = "x-changed"
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert any(ln.strip() == long_id for ln in out.splitlines()), out
    huge = "hf:" + "x" * 120  # 幅より長い ID は 1 行にはみ出す(切らない)
    for d in (old, new):
        _model(d, long_id)["id"] = huge
    out, _, _ = _cli_and_merge(capsys, tmp_path, old, new)
    assert any(ln.strip() == huge for ln in out.splitlines())


def test_older_new_list_warns_for_one_and_two_args(capsys, tmp_path):
    old = _raw(NEW)
    older = copy.deepcopy(old)
    older["generated_at"] = "2000-01-01T00:00:00Z"
    p = _write_raw(tmp_path, "older.json", older)
    _, out = _run(capsys, str(p))  # 同梱 → 古いファイル
    assert "注意:" in out and "古いです" in out
    _, out = _run(capsys, str(NEW), str(p))
    assert "注意:" in out and "向きが逆" in out
    _, out = _run(capsys, str(p), str(NEW))
    assert "注意:" not in out
