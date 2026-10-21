# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""一覧(データベース)の読み書き。

読む順: 1) 明示したパス 2) 利用者のキャッシュ(`genmodelfit update` が書く) 3) 同梱のデータ。
実測(measurements)は同梱のもの(作者の measurements.json と、ほかの利用者の実測のまとめ
measurements_community.json)とキャッシュのものを合わせて使う。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

from platformdirs import user_cache_dir

from .models import SCHEMA_VERSION, Measurement, ModelEntry


class DatabaseError(RuntimeError):
    pass


def cache_dir() -> Path:
    env = os.environ.get("GENMODELFIT_CACHE_DIR")
    return Path(env) if env else Path(user_cache_dir("genmodelfit", appauthor=False))


def cached_db_path() -> Path:
    return cache_dir() / "db.json"


def bundled_text(name: str) -> str:
    return resources.files("genmodelfit.data").joinpath(name).read_text(encoding="utf-8")


@dataclass
class Database:
    models: list[ModelEntry] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    generated_at: str | None = None
    sources: dict[str, Any] = field(default_factory=dict)
    origin: str = "bundled"  # bundled / cache / path

    def get(self, model_id: str) -> ModelEntry | None:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def find(self, query: str) -> list[ModelEntry]:
        q = query.lower()
        exact = self.get(query)
        if exact:
            return [exact]
        return [m for m in self.models if q in m.id.lower() or q in m.name.lower()]

    def measurements_for(self, model_id: str) -> list[Measurement]:
        return [m for m in self.measurements if m.model_id == model_id]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_at": self.generated_at,
            "sources": self.sources,
            "models": [m.to_dict() for m in self.models],
        }

    def save(self, path: Path) -> None:
        dup = duplicate_ids(self.models)
        if dup:
            raise DatabaseError(f"同じモデル ID が 2 回以上あります: {', '.join(dup[:5])}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8", newline="\n"
        )
        tmp.replace(path)


_STR = (str,)
_OPT_STR = (str, type(None))
_OPT_NUM = (int, float, type(None))
# 項目の型(読めない値で後から落ちないように、読むときに確かめる。docs/schema.md)
_ENTRY_TYPES = {
    "id": _STR,
    "name": _STR,
    "modality": _STR,
    "tasks": (list,),
    "files": (list,),
    "template": (dict, type(None)),
    "params_b": _OPT_NUM,
    "precision": _OPT_STR,
    "license": _OPT_STR,
    "license_name": _OPT_STR,  # 文字でないと、ライセンスの段階の判定(diff・gate)が落ちる(#59)
    "content": _STR,
    "downloads": (int, type(None)),
    "downloads_all_time": (int, type(None)),
    "files_complete": (bool, type(None)),
    "size_floor_bytes": (int, type(None)),
    # 後から足した任意の項目(0.5.0、#36)。古い版はこのキーを見ずに捨てる
    "languages": (dict, type(None)),
    "runtime": _OPT_STR,
    "description": (dict, type(None)),
    "card_languages": (list, type(None)),
    # 確認の印(0.6.0、#42)。無い・null は確認済みとして扱う
    "review": (dict, type(None)),
}
_REVIEW_TYPES = {
    "status": _STR,
    "reasons": (list,),
    "decided_by": _OPT_STR,
    "decided_at": _OPT_STR,
    "evidence": (list,),
    "note": _OPT_STR,
}
_FILE_TYPES = {
    "name": _STR,
    "url": _OPT_STR,
    "sha256": _OPT_STR,
    "size_bytes": (int, type(None)),
    "directory": _OPT_STR,
    "license": _OPT_STR,
    "license_name": _OPT_STR,
    "gated": (bool, type(None)),
    "role": _OPT_STR,
}


def _check_types(d: Any, types: dict[str, tuple], where: str) -> None:
    if not isinstance(d, dict):
        raise DatabaseError(f"{where} の形が正しくありません(オブジェクトではない)")
    for key, ok in types.items():
        if key not in d:
            if ok is _STR:
                raise DatabaseError(f"{where} に {key} がありません")
            continue
        v = d[key]
        bad = not isinstance(v, ok) or (isinstance(v, bool) and bool not in ok)
        if bad:
            raise DatabaseError(f"{where} の {key} の型が正しくありません: {v!r}")


def check_entry(d: Any, i: int) -> None:
    """一覧の 1 項目の型を確かめる。違えば DatabaseError。"""
    where = f"models[{i}]"
    _check_types(d, _ENTRY_TYPES, where)
    if not all(isinstance(t, str) for t in d.get("tasks", [])):
        raise DatabaseError(f"{where} の tasks は文字列の並びです")
    langs = d.get("languages")
    for k, v in (langs or {}).items():
        if not isinstance(k, str) or not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            raise DatabaseError(f"{where} の languages は {{種類: 言語コードの並び}} です: {langs!r}")
    cl = d.get("card_languages")
    if cl is not None and not all(isinstance(x, str) for x in cl):
        raise DatabaseError(f"{where} の card_languages は言語コードの並びです: {cl!r}")
    desc = d.get("description")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in (desc or {}).items()):
        raise DatabaseError(f"{where} の description は {{言語: 文}} です: {desc!r}")
    rv = d.get("review")
    if rv is not None:
        # status の値は確かめない(知らない値は ok でないので、確認待ちと同じく自動のおすすめに出ない)
        _check_types(rv, _REVIEW_TYPES, f"{where}.review")
        for key in ("reasons", "evidence"):
            if not all(isinstance(x, str) for x in rv.get(key, [])):
                raise DatabaseError(f"{where}.review の {key} は文字列の並びです")
    for j, f in enumerate(d.get("files", [])):
        _check_types(f, _FILE_TYPES, f"{where}.files[{j}]")


def duplicate_ids(models: list[Any]) -> list[str]:
    """2 回以上出てくる ID(並びは最初に出た順)。"""
    seen: set[str] = set()
    dup: list[str] = []
    for m in models:
        i = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
        if i in seen and i not in dup:
            dup.append(i)
        seen.add(i)
    return dup


def parse_db(data: dict[str, Any], origin: str) -> Database:
    if not isinstance(data, dict):
        raise DatabaseError("一覧の形が正しくありません(JSON のオブジェクトではない)")
    ver = data.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise DatabaseError(f"未対応のデータ形式です(schema_version={ver}、対応={SCHEMA_VERSION})")
    if not isinstance(data.get("models"), list):
        raise DatabaseError("models がありません")
    for i, m in enumerate(data["models"]):
        check_entry(m, i)
    dup = duplicate_ids(data["models"])
    if dup:
        # 同じ ID が 2 つあると get() が片方だけを黙って選ぶ(差し替えた項目を紛れ込ませられる)。#42
        raise DatabaseError(f"同じモデル ID が 2 回以上あります: {', '.join(dup[:5])}")
    try:
        models = [ModelEntry.from_dict(m) for m in data["models"]]
    except (TypeError, AttributeError, ValueError) as e:
        raise DatabaseError(f"一覧のモデルの形が正しくありません: {e}") from e
    return Database(
        models=models,
        generated_at=data.get("generated_at"),
        sources=data.get("sources") or {},
        origin=origin,
    )


def parse_measurements(data: Any) -> list[Measurement]:
    items = data.get("measurements", []) if isinstance(data, dict) else data
    return [Measurement.from_dict(m) for m in items]


# ほかの利用者の実測をまとめたもの(#34。`genmodelfit contrib aggregate --output` が data/contrib から作る)
COMMUNITY_FILE = "measurements_community.json"


def load_measurements(extra_dirs: list[Path] | None = None) -> list[Measurement]:
    """同梱の実測(作者)+ 同梱のほかの利用者の実測のまとめ + この PC の bench の実測(キャッシュ)。"""
    out = parse_measurements(json.loads(bundled_text("measurements.json")))
    out.extend(parse_measurements(json.loads(bundled_text(COMMUNITY_FILE))))
    for d in [cache_dir() / "measurements", *(extra_dirs or [])]:
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                try:
                    out.extend(parse_measurements(json.loads(p.read_text(encoding="utf-8"))))
                except (OSError, ValueError, TypeError) as e:
                    raise DatabaseError(f"実測のファイルを読めません: {p}: {e}") from e
    return out


def load(path: str | os.PathLike[str] | None = None, *, use_cache: bool = True) -> Database:
    if path:
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:  # 無い・読めない・JSON でない(UnicodeDecodeError も ValueError)
            raise DatabaseError(f"一覧のファイルを読めません: {p}: {e}") from e
        db = parse_db(data, "path")
    else:
        db = None
        cp = cached_db_path()
        if use_cache and cp.exists():
            try:
                db = parse_db(json.loads(cp.read_text(encoding="utf-8")), "cache")
            except (OSError, ValueError, DatabaseError):
                db = None  # 壊れたキャッシュは使わず、同梱のデータへ戻る
        if db is None:
            db = parse_db(json.loads(bundled_text("db.json")), "bundled")
    db.measurements = load_measurements()
    return db
