"""genmodelfit: 画像・動画・音声・3D の生成モデルが、その PC で動くかを判定する。

Python から:
    import genmodelfit as gmf
    info = gmf.system()
    for fit in gmf.recommend(modality="image", limit=5):
        print(fit.model.name, fit.estimate.fit, fit.estimate.vram_gb)
"""

from __future__ import annotations

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


def system(vram_gb: float | None = None, ram_gb: float | None = None) -> SystemInfo:
    """この PC を調べる。vram_gb / ram_gb で上書きできる。"""
    return detect(vram_gb=vram_gb, ram_gb=ram_gb)


def recommend(
    *,
    modality: str | None = None,
    content: str = "all",
    min_fit: str = "marginal",
    limit: int | None = 10,
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
) -> list[Fit]:
    """おすすめ順の一覧(先頭が一番のおすすめ)。"""
    from .recommend import recommend as _recommend

    return _recommend(
        db or load_db(),
        system_info or detect(),
        modality=modality,
        content=content,
        min_fit=min_fit,
        limit=limit,
        settings=settings,
    )


def info(model_id: str, *, db: Database | None = None) -> ModelEntry | None:
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
    "system",
]
