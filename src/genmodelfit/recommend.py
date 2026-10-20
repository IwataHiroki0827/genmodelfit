"""おすすめ順に並べる。

並べ方: 段階(perfect が先)→ 実測を優先 → 人気(使われた回数・HF のダウンロード数)→ 新しさ。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .db import Database
from .estimate import GIB, Coefficients, estimate, load_coefficients
from .models import CONTENT_LEVELS, FIT_LEVELS, MODALITIES, Estimate, ModelEntry, Settings, SystemInfo

# 内容の絞り込み。既定は sfw(無検閲は all / uncensored を指定したときだけ出す)
CONTENT_FILTERS = ("all", *CONTENT_LEVELS)
DEFAULT_CONTENT = "sfw"


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
            "total_size_gb": round((m.total_size_bytes or 0) / GIB, 2) or None,
            "score": round(self.score, 1),
            "fit": self.estimate.fit,
            "estimate": self.estimate.to_dict(),
            "gate": self.gate,
        }
        if full:
            d["model"] = m.to_dict()
        return d


def check_filters(
    *, modality: str | None = None, content: str = DEFAULT_CONTENT, min_fit: str | None = None
) -> None:
    """絞り込みの値を確かめる。知らない値は ValueError(組み込み側が誤りに気づけるように)。"""
    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"modality が正しくありません: {modality!r}(使えるのは {', '.join(MODALITIES)})")
    if content not in CONTENT_FILTERS:
        raise ValueError(f"content が正しくありません: {content!r}(使えるのは {', '.join(CONTENT_FILTERS)})")
    # too_tight は動かないので recommend の下限には使えない
    if min_fit is not None and min_fit not in FIT_LEVELS[:3]:
        raise ValueError(f"min_fit が正しくありません: {min_fit!r}(使えるのは {', '.join(FIT_LEVELS[:3])})")


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
    content: str = DEFAULT_CONTENT,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    models: Iterable[ModelEntry] | None = None,
    with_limits: bool = False,
) -> list[Fit]:
    check_filters(modality=modality, content=content)
    c = coefficients or load_coefficients()
    out: list[Fit] = []
    for m in models if models is not None else db.models:
        if modality and m.modality != modality:
            continue
        if content != "all" and m.content != content:
            continue
        e = estimate(m, system, settings, db.measurements, c, with_limits=with_limits)
        out.append(Fit(m, e, score(m, e)))
    # 段階を最優先にする(実測や人気の加点で、下の段階が上に来ないように)
    out.sort(key=lambda f: (FIT_LEVELS.index(f.estimate.fit), -f.score, f.model.id))
    return out


# 「つくる」モデルではない補助ツールの作業(recommend は既定で外す。fit・info には出る)
TOOL_TASKS = frozenset({"tool", "estimate", "matting", "upscale", "interp", "segment"})


def is_tool(m: ModelEntry) -> bool:
    return bool(m.tasks) and all(t in TOOL_TASKS for t in m.tasks)


def recommend(
    db: Database,
    system: SystemInfo,
    *,
    modality: str | None = None,
    content: str = DEFAULT_CONTENT,
    min_fit: str = "marginal",
    limit: int | None = 10,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    with_limits: bool = False,
) -> list[Fit]:
    check_filters(modality=modality, content=content, min_fit=min_fit)
    allowed = FIT_LEVELS[: FIT_LEVELS.index(min_fit) + 1]
    fits = [
        f
        for f in evaluate(
            db,
            system,
            modality=modality,
            content=content,
            settings=settings,
            coefficients=coefficients,
            with_limits=with_limits,
        )
        if f.estimate.fit in allowed and not f.estimate.unknown_size  # 大きさが分からないものは出さない
    ]
    return fits[:limit] if limit else fits
