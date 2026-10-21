# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit diff` / `gmf.diff_db` の試験(#19)。同梱の一覧を写して書き換えたものと比べる。"""

from __future__ import annotations

import copy
import json

import pytest
from rich.cells import cell_len

import genmodelfit as gmf
from genmodelfit import cli
from genmodelfit.db import load
from genmodelfit.diff import diff_db
from genmodelfit.gate import GatePolicy, evaluate_offline
from genmodelfit.models import ModelEntry, ModelFile

KEYS = {
    "old",
    "new",
    "policy",
    "summary",
    "added",
    "removed",
    "license_changes",
    "auto_changes",
    "size_changes",
    "precision_changes",
    "newly_uncensored",
    "head_failures",
    "unauthorized",
    "partial_counts",
    "stats",
}


@pytest.fixture(scope="module")
def base():
    return load(use_cache=False)


@pytest.fixture
def pair(base):
    return copy.deepcopy(base), copy.deepcopy(base)


def _auto_id(db) -> str:
    p = GatePolicy(max_head_age_days=None)
    return next(m.id for m in db.models if evaluate_offline(m, p, sources=db.sources).auto)


def test_same_list_has_no_changes(pair):
    old, new = pair
    d = diff_db(old, new)
    assert set(d) == KEYS
    s = d["summary"]
    for k in (
        "added",
        "removed",
        "license_changed",
        "auto_gained",
        "auto_lost",
        "size_changed",
        "precision_changed",
        "newly_uncensored",
        "head_failures_new",
        "unauthorized_added",
        "unauthorized_removed",
    ):
        assert s[k] == 0, k
    assert d["stats"] == [] and s["auto_old"] == s["auto_new"] > 0
    assert d["policy"]["max_head_age_days"] is None
    json.dumps(d)


def test_added_removed_and_uncensored(pair):
    old, new = pair
    gone = new.models.pop(0)
    nsfw = ModelEntry(
        id="hf:x/new-nsfw", name="new-nsfw", modality="image", source="huggingface", content="uncensored",
        content_evidence=["hf-tag:not-for-all-audiences"],
    )  # fmt: skip
    new.models.append(nsfw)
    flip = next(m for m in new.models if m.content == "sfw")
    flip.content, flip.content_evidence = "uncensored", ["name:nsfw"]
    d = diff_db(old, new)
    assert [m["id"] for m in d["removed"]] == [gone.id]
    assert [m["id"] for m in d["added"]] == [nsfw.id]
    assert d["added"][0]["license"] == "不明" and d["added"][0]["content"] == "uncensored"
    by = {m["id"]: m for m in d["newly_uncensored"]}
    assert by[nsfw.id]["added"] is True and by[flip.id]["added"] is False
    assert by[flip.id]["evidence"] == ["name:nsfw"]


def test_license_change_toward_permissive_is_flagged(pair):
    old, new = pair
    a = next(m for m in new.models if m.license is None)
    a.license = "mit"
    b = next(m for m in new.models if m.license == "apache-2.0")
    b.license = "cc-by-nc-4.0"
    d = diff_db(old, new)
    by = {c["id"]: c for c in d["license_changes"]}
    assert by[a.id]["old"] == "不明" and by[a.id]["new"] == "mit"
    assert by[a.id]["permissive_side"] is True and by[a.id]["allowed_new"] is True
    assert by[b.id]["permissive_side"] is False and by[b.id]["allowed_old"] is True
    assert by[b.id]["allowed_new"] is False
    assert d["summary"]["license_permissive_side"] == 1 and d["summary"]["license_changed"] == 2


def test_auto_change_from_head_failure(pair):
    old, new = pair
    mid = _auto_id(new)
    m = new.get(mid)
    url = m.files[0].url
    for e in new.models:  # 同じ URL を使う項目はすべて
        for f in e.files:
            if f.url == url:
                f.head_status = 401
    d = diff_db(old, new)
    lost = {c["id"]: c for c in d["auto_changes"]}
    assert lost[mid]["old"] is True and lost[mid]["new"] is False
    assert "url_reachable" in lost[mid]["held_by"]
    fail = next(h for h in d["head_failures"] if h["url"] == url)
    assert fail["status"] == 401 and fail["new"] is True and mid in fail["ids"]
    assert d["summary"]["auto_lost"] >= 1 and d["summary"]["head_failures_new"] >= 1
    # 前から失敗している URL は new ではない
    again = diff_db(new, copy.deepcopy(new))
    assert all(not h["new"] for h in again["head_failures"]) and again["auto_changes"] == []


def test_head_age_is_ignored(pair):
    old, new = pair
    for m in old.models:
        for f in m.files:
            if f.head_checked_at:
                f.head_checked_at = "2020-01-01T00:00:00Z"  # 古い確認でも、同じ方針で比べる
    assert diff_db(old, new)["auto_changes"] == []


def test_size_and_precision_thresholds(pair):
    old, new = pair
    big = next(m for m in new.models if (m.total_size_bytes or 0) > 4 * 2**30 and m.files[0].size_bytes)
    big.files[0].size_bytes += 2 * 2**30
    small = next(m for m in new.models if m.id != big.id and m.files and m.files[0].size_bytes)
    small.files[0].size_bytes += 1  # しきい値の下
    prec = next(m for m in new.models if m.precision and m.id not in (big.id, small.id))
    prec.precision = "nf4"
    d = diff_db(old, new)
    assert [c["id"] for c in d["size_changes"]] == [big.id]
    assert d["size_changes"][0]["new_bytes"] - d["size_changes"][0]["old_bytes"] == 2 * 2**30
    assert [(c["id"], c["new"]) for c in d["precision_changes"]] == [(prec.id, "nf4")]


def test_unauthorized_and_stats(pair):
    old, new = pair
    new.sources["huggingface"]["unauthorized"] = [*new.sources["huggingface"]["unauthorized"], "x/private"]
    new.sources["counts"]["models"] += 5
    d = diff_db(old, new)
    assert d["unauthorized"] == {"added": ["x/private"], "removed": []}
    keys = {x["key"]: x for x in d["stats"]}
    assert keys["counts.models"]["new"] == keys["counts.models"]["old"] + 5
    assert "huggingface.unauthorized" in keys  # リストは数で比べる


def test_fixed_comfyui_version(pair):
    old, new = pair
    d = diff_db(old, new, comfyui_version="0.1.0")
    assert (
        d["policy"]["comfyui_version"] == "0.1.0"
        and d["summary"]["auto_new"] < diff_db(old, new)["summary"]["auto_new"]
    )


def test_python_api_writes_nothing(pair, capfd):
    old, new = pair
    new.models.pop()
    assert gmf.diff_db(old, new)["summary"]["removed"] == 1
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def _write(db, path):
    db.save(path)
    return str(path)


def test_cli_json_and_text(pair, tmp_path, capsys):
    old, new = pair
    gone = new.models.pop(3)
    for m in new.models[:30]:  # 長い行と多い節(省略の行)を出す
        m.license, m.license_name = "other", "a-very-long-license-name-" + "x" * 80
    a, b = _write(old, tmp_path / "old.json"), _write(new, tmp_path / "new.json")
    assert cli.main(["diff", a, b, "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["command"] == "diff" and set(d) == KEYS | {"genmodelfit", "command"}
    assert d["removed"][0]["id"] == gone.id
    assert cli.main(["diff", a, b, "-n", "5"]) == 0
    text = capsys.readouterr().out
    assert gone.id in text and "--json で全部" in text
    assert all(cell_len(line) <= 100 for line in text.splitlines()), max(text.splitlines(), key=cell_len)


def test_cli_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["diff", str(bad), str(bad), "--json"]) == 1
    assert "error" in json.loads(capsys.readouterr().out)


def test_entry_without_files_is_fine():
    from genmodelfit.db import Database

    e = ModelEntry(id="a", name="a", modality="image", files=[ModelFile(name="x.safetensors")])
    d = diff_db(Database(models=[]), Database(models=[e]))
    assert d["summary"]["added"] == 1 and d["summary"]["auto_new"] == 0
