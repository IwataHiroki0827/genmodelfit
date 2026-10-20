"""genmodelfit: 画像・動画・音声・3D の生成モデルが、その PC で動くかを判定する。

Python から:
    import genmodelfit as gmf
    info = gmf.system()
    for fit in gmf.recommend(modality="image", limit=5):
        print(fit.model.id, fit.score, fit.auto, fit.estimate.fit, fit.estimate.vram_gb)
        fit.to_dict()  # `recommend --json` の models の 1 件と同じ形
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from .db import Database
from .db import load as load_db
from .estimate import estimate
from .hardware import detect
from .models import (
    CONTENT_LEVELS,
    FIT_LEVELS,
    MODALITIES,
    Estimate,
    Measurement,
    ModelEntry,
    ModelFile,
    Settings,
    SystemInfo,
)
from .recommend import Fit, evaluate

__version__ = "0.1.0.dev0"

# 途中経過は logging に出す。組み込み側が設定しなければ何も出ない
logging.getLogger("genmodelfit").addHandler(logging.NullHandler())


def system(vram_gb: float | None = None, ram_gb: float | None = None) -> SystemInfo:
    """この PC を調べる。vram_gb / ram_gb で上書きできる。"""
    return detect(vram_gb=vram_gb, ram_gb=ram_gb)


def _policy(comfyui_version: str | None):
    from .gate import GatePolicy

    return GatePolicy(comfyui_version=comfyui_version)


def recommend(
    *,
    modality: str | Iterable[str] | None = None,
    content: str = "sfw",
    min_fit: str = "marginal",
    license: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    include_tools: bool = False,
    auto_only: bool = False,
    all_workflows: bool = False,
    limit: int | None = 10,
    per_modality: int | None = None,
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
) -> list[Fit]:
    """おすすめ順の一覧(先頭が一番のおすすめ)。CLI の `recommend` と同じ並べ方。

    1 件 = 1 モデル(同じ重みの系統と大きさ。精度・使い方だけ違うワークフローはまとめ、段階が良く精度が
    高いものを出す。ほかは fit.workflows / fit.variants)。all_workflows=True ならワークフローごと。
    並べ方: 動くもの → 自動でおすすめに出せるもの(fit.auto)→ fit.score の高い順 → ID。
    `fit.to_dict()` は `recommend --json` の models の 1 件と同じ形(docs/schema.md)。

    modality: "image"、"image,video"、["image", "video"] のどれでもよい。
    task: 用途("edit"、"gen,i2v"、["i2v"]。tasks のコード t2i なども可。一覧は tasks.USES)。
    content の既定は sfw。無検閲のモデルも含めるなら "all"、無検閲だけなら "uncensored"。
    license: "apache-2.0,mit" か ["apache-2.0", "mit"](すべてのファイルのライセンスがこの中にあるものだけ。
    不明は合わない)。
    limit / per_modality: 1 以上か None。per_modality を渡すと種類ごとに上位その数(CLI の既定の節と同じ。
    CLI の `recommend`(-m も --task も無し)は per_modality=3, limit=None)。
    comfyui_version: 使う ComfyUI の版(自動おすすめの判定。アプリは自分の版を渡す)。
    知らない modality / content / min_fit / task、空の license、0 以下の limit は ValueError。
    """
    from .recommend import recommend as _recommend

    return _recommend(
        db or load_db(),
        system_info or detect(),
        modality=modality,
        content=content,
        min_fit=min_fit,
        license=license,
        task=task,
        include_tools=include_tools,
        auto_only=auto_only,
        group=not all_workflows,
        limit=limit,
        per_modality=per_modality,
        settings=settings,
        policy=_policy(comfyui_version),
    )


def search(
    query: str,
    *,
    modality: str | Iterable[str] | None = None,
    content: str = "sfw",
    min_fit: str | None = None,
    license: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    all_workflows: bool = False,
    limit: int | None = None,
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
) -> list[Fit]:
    """ID・名前の一部(大文字小文字を区別しない)で探す。CLI の `search` と同じ。合わなければ空の一覧。
    空の query は ValueError。"""
    from .recommend import select

    return select(
        db or load_db(),
        system_info or detect(),
        query=query,
        modality=modality,
        content=content,
        min_fit=min_fit,
        license=license,
        task=task,
        group=not all_workflows,
        limit=limit,
        settings=settings,
        policy=_policy(comfyui_version),
    ).fits


def info(model_id: str, *, db: Database | None = None) -> ModelEntry | None:
    """ID がそのまま合うモデル(内容によらず)。一部だけの一致では探さない。無ければ None。"""
    return (db or load_db()).get(model_id)


__all__ = [
    "CONTENT_LEVELS",
    "FIT_LEVELS",
    "MODALITIES",
    "Database",
    "Estimate",
    "Fit",
    "Measurement",
    "ModelEntry",
    "ModelFile",
    "Settings",
    "SystemInfo",
    "__version__",
    "detect",
    "estimate",
    "evaluate",
    "info",
    "load_db",
    "recommend",
    "search",
    "system",
]
