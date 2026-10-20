"""おすすめ順に並べる。

並べ方: 段階(perfect が先)→ 実測を優先 → 人気(使われた回数・HF のダウンロード数)→ 新しさ。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .db import Database
from .estimate import Coefficients, estimate, load_coefficients
from .models import FIT_LEVELS, Estimate, ModelEntry, Settings, SystemInfo


@dataclass
class Fit:
    model: ModelEntry
    estimate: Estimate
    score: float
    gate: dict[str, Any] | None = None  # 自動でおすすめに出せるか(#3 の gate.evaluate の結果)

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        m = self.model
        d: dict[str, Any] = {
            "id": m.id,
            "name": m.name,
            "modality": m.modality,
            "tasks": m.tasks,
            "content": m.content,
            "license": m.license,
            "source": m.source,
            "total_size_gb": round((m.total_size_bytes or 0) / 1e9, 2) or None,
            "score": round(self.score, 1),
            "fit": self.estimate.fit,
            "estimate": self.estimate.to_dict(),
            "gate": self.gate,
        }
        if full:
            d["model"] = m.to_dict()
        return d


def score(m: ModelEntry, e: Estimate) -> float:
    """0〜100。段階が大半を決め、同じ段階の中を人気・新しさ・実測で並べる。"""
    level = {"perfect": 70, "good": 55, "marginal": 30, "too_tight": 0}[e.fit]
    usage = (m.template.usage or 0) if m.template else 0
    if not m.template and m.downloads:
        usage = m.downloads / 100  # HF のダウンロード数を、テンプレートの使用回数と同じくらいの目盛りに直す
    pop = min(15.0, 3 * math.log10(1 + usage))
    date = (m.template.date if m.template else None) or m.updated or ""
    year = int(date[:4]) if date[:4].isdigit() else 2023
    fresh = max(0.0, min(10.0, (year - 2022) * 2.5))
    measured = 5.0 if e.basis == "measured" else 0.0
    return level + pop + fresh + measured


def evaluate(
    db: Database,
    system: SystemInfo,
    *,
    modality: str | None = None,
    content: str = "all",
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    models: Iterable[ModelEntry] | None = None,
) -> list[Fit]:
    c = coefficients or load_coefficients()
    out: list[Fit] = []
    for m in models if models is not None else db.models:
        if modality and m.modality != modality:
            continue
        if content != "all" and m.content != content:
            continue
        e = estimate(m, system, settings, db.measurements, c)
        out.append(Fit(m, e, score(m, e)))
    out.sort(key=lambda f: (-f.score, f.model.id))
    return out


def recommend(
    db: Database,
    system: SystemInfo,
    *,
    modality: str | None = None,
    content: str = "all",
    min_fit: str = "marginal",
    limit: int | None = 10,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
) -> list[Fit]:
    allowed = FIT_LEVELS[: FIT_LEVELS.index(min_fit) + 1]
    fits = [
        f
        for f in evaluate(
            db, system, modality=modality, content=content, settings=settings, coefficients=coefficients
        )
        if f.estimate.fit in allowed
    ]
    return fits[:limit] if limit else fits
