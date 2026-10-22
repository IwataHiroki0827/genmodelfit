# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""一覧の読み込みでの型の検査の試験(#63)。

型を変えた一覧は、読むときに断るか、通るなら diff・gate・estimate が例外を出さない。
ネットワークには触らない。"""

from __future__ import annotations

import copy
import json
import random
from dataclasses import fields

import pytest

import genmodelfit as gmf
from genmodelfit.db import DatabaseError, parse_db
from genmodelfit.diff import diff_db
from genmodelfit.gate import evaluate_offline
from genmodelfit.models import Gpu, ModelEntry, ModelFile, SystemInfo, Template
from genmodelfit.recommend import evaluate

WRONG = [123, 1.5, True, ["x"], {"a": 1}, [["x"]], [1], None, "x", ""]
PROBE = [
    SystemInfo(
        os="t",
        cpu="t",
        cpu_cores=8,
        ram_total_gb=ram,
        ram_available_gb=ram,
        gpus=[Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=v, backend="cuda", bf16=False)],
    )
    for v, ram in ((8.0, 32.0), (24.0, 64.0))
]


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def raw() -> dict:
    db = gmf.load_db(use_cache=False)
    d = db.to_dict()
    d["models"] = [m for m in d["models"] if m.get("files")][:40]
    return json.loads(json.dumps(d))


def _use(data: dict) -> None:
    """読めた一覧を判定に通す(例外が出ないこと)。"""
    db = parse_db(copy.deepcopy(data), "path")
    diff_db(parse_db(copy.deepcopy(data), "path"), db)
    diff_db(db, parse_db(copy.deepcopy(data), "path"))
    for m in db.models:
        evaluate_offline(m)
    for pc in PROBE:
        evaluate(db, pc, content="all", with_limits=True)


@pytest.mark.parametrize(
    "key",
    ["repo_id", "license_from", "license_path", "license_url", "repo_license", "license_evidence"],
)
def test_file_fields_reject_wrong_types(raw, key):
    for bad in (123, ["x", 1], {"a": 1}, True):
        data = copy.deepcopy(raw)
        data["models"][0]["files"][0][key] = bad
        with pytest.raises(DatabaseError, match=key):
            parse_db(data, "path")


@pytest.mark.parametrize("key", ["comfy_family", "comfy_vae", "sampler_stages", "license_url", "base_model"])
def test_entry_fields_reject_wrong_types(raw, key):
    for bad in (123, "x" if key == "sampler_stages" else 5, {"a": 1}, [[1]], True):
        data = copy.deepcopy(raw)
        data["models"][0][key] = bad
        with pytest.raises(DatabaseError, match=key):
            parse_db(data, "path")


def test_template_fields_checked(raw):
    data = copy.deepcopy(raw)
    m = next(m for m in data["models"] if m.get("template"))
    m["template"]["category"] = 5
    with pytest.raises(DatabaseError, match="template"):
        parse_db(data, "path")


def test_valid_values_still_load(raw):
    data = copy.deepcopy(raw)
    data["models"][0]["sampler_stages"] = [["a", "b"], ["c"]]
    data["models"][0]["comfy_family"] = None
    assert parse_db(data, "path").models[0].sampler_stages == [["a", "b"], ["c"]]


def _paths(raw: dict) -> list[tuple]:
    out = []
    for i, m in enumerate(raw["models"]):
        out += [(i, None, None, f.name) for f in fields(ModelEntry) if f.name not in ("id",)]
        out += [(i, j, None, f.name) for j in range(len(m["files"])) for f in fields(ModelFile)]
        if m.get("template"):
            out += [(i, None, "t", f.name) for f in fields(Template)]
    return out


def test_random_wrong_types_never_crash_downstream(raw):
    """項目を無作為に別の型へ変えた一覧は、読むときに断るか、通るなら判定が例外を出さない(シード固定)。"""
    rng = random.Random(63)
    paths = _paths(raw)
    accepted = 0
    for _ in range(150):
        data = copy.deepcopy(raw)
        for _k in range(rng.choice((1, 1, 2))):
            i, j, t, name = rng.choice(paths)
            m = data["models"][i]
            target = m["files"][j] if j is not None else m["template"] if t else m
            target[name] = copy.deepcopy(rng.choice(WRONG))
        try:
            parse_db(copy.deepcopy(data), "path")
        except DatabaseError:
            continue
        accepted += 1
        _use(data)
    assert accepted > 0
