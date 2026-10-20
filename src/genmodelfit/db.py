"""一覧(データベース)の読み書き。

読む順: 1) 明示したパス 2) 利用者のキャッシュ(`genmodelfit update` が書く) 3) 同梱のデータ。
実測(measurements)は同梱のものとキャッシュのものを合わせて使う。
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
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)


def parse_db(data: dict[str, Any], origin: str) -> Database:
    if not isinstance(data, dict):
        raise DatabaseError("一覧の形が正しくありません(JSON のオブジェクトではない)")
    ver = data.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise DatabaseError(f"未対応のデータ形式です(schema_version={ver}、対応={SCHEMA_VERSION})")
    if not isinstance(data.get("models"), list):
        raise DatabaseError("models がありません")
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


def load_measurements(extra_dirs: list[Path] | None = None) -> list[Measurement]:
    out = parse_measurements(json.loads(bundled_text("measurements.json")))
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
