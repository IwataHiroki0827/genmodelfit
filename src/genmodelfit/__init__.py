# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""genmodelfit: 画像・動画・音声・3D の生成モデルが、その PC で動くかを判定する。

Python から:
    import genmodelfit as gmf
    info = gmf.system()
    for fit in gmf.recommend(modality="image", limit=5):
        print(fit.model.id, fit.score, fit.auto, fit.estimate.fit, fit.estimate.vram_gb)
        fit.to_dict()  # `recommend --json` の models の 1 件と同じ形
    if gmf.data_status().expired:  # 同梱・取り込んだ一覧の HEAD の確認が古い(30 日)
        ...
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from ._version import __version__
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
from .status import DataStatus

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
    include_unknown_license: bool = False,
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
    include_unknown_license: ライセンスの分からないファイルを含むものも出す(既定は出さない。#28)。
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
        include_unknown_license=include_unknown_license,
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
    fit: str | None = None,
    license: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    include_unknown_license: bool = False,
    all_workflows: bool = False,
    limit: int | None = None,
    sort: str = "type",
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
) -> list[Fit]:
    """ID・名前の一部(大文字小文字を区別しない)で探す。CLI の `search` と同じ。合わなければ空の一覧。
    ライセンスの分からないものは include_unknown_license=True のときだけ出す。fit / sort は fit_list と同じ
    (既定は全部の段階、種類 → 段階 → 点数の順)。空の query は ValueError。"""
    from .recommend import select

    return select(
        db or load_db(),
        system_info or detect(),
        query=query,
        modality=modality,
        content=content,
        min_fit=min_fit,
        fit=fit,
        license=license,
        task=task,
        include_unknown_license=include_unknown_license,
        group=not all_workflows,
        limit=limit,
        sort=sort,
        auto_first=False,
        settings=settings,
        policy=_policy(comfyui_version),
    ).fits


def fit_list(
    *,
    modality: str | Iterable[str] | None = None,
    content: str = "sfw",
    fit: str | None = "runnable",
    min_fit: str | None = None,
    license: str | Iterable[str] | None = None,
    task: str | Iterable[str] | None = None,
    include_tools: bool = False,
    include_unknown_license: bool = False,
    show_all: bool = False,
    sort: str = "type",
    all_workflows: bool = False,
    limit: int | None = None,
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
) -> list[Fit]:
    """CLI の `fit` と同じ一覧(0.1.1)。既定は動くもの(perfect / good / marginal)だけで、補助ツール・
    ライセンス不明・無検閲は出さない。並びは種類 → 段階 → 点数(sort="type")。

    fit: "perfect" / "good" / "marginal"(その段階だけ)、"runnable"、"all"(too_tight・判定できないものも)。
    min_fit を渡すときは fit=None にする(両方は ValueError)。show_all=True は CLI の --all
    (fit="all"・補助ツール・ライセンス不明も。無検閲は content="all" で)。
    sort: "type" / "score"(種類をまたいだ順位)/ "vram" / "disk" / "name"。"""
    from .recommend import select

    if show_all:
        fit = "all" if min_fit is None else None
        include_tools = include_unknown_license = True
    return select(
        db or load_db(),
        system_info or detect(),
        modality=modality,
        content=content,
        min_fit=min_fit,
        fit=fit,
        license=license,
        task=task,
        include_tools=include_tools,
        include_unknown_license=include_unknown_license,
        group=not all_workflows,
        limit=limit,
        sort=sort,
        auto_first=False,
        settings=settings,
        policy=_policy(comfyui_version),
    ).fits


def data_status(
    db: Database | None = None,
    *,
    now: datetime | None = None,
    comfyui_version: str | None = None,
) -> DataStatus:
    """一覧の出どころと新しさ。`expired` が True なら、HEAD の確認が古いせいで自動のおすすめから
    落ちたモデルがある(30 日。`genmodelfit update --from-release` か `update --heads-only` で取り直す)。

    判定は自動のおすすめ(gate)と同じ関数で行う。auto_count がいま自動で出せる数、expires_at が次に
    期限が切れる時刻。now は試験用(既定は今の UTC。タイムゾーンが無ければ UTC とみなす)。
    comfyui_version は recommend と同じ(アプリは自分の版を渡す)。ネットには触らない。
    """
    from .status import compute

    return compute(db or load_db(), _policy(comfyui_version), now=now)


def info(model_id: str, *, db: Database | None = None) -> ModelEntry | None:
    """ID がそのまま合うモデル(内容によらず)。一部だけの一致では探さない。無ければ None。"""
    return (db or load_db()).get(model_id)


__all__ = [
    "CONTENT_LEVELS",
    "FIT_LEVELS",
    "MODALITIES",
    "DataStatus",
    "Database",
    "Estimate",
    "Fit",
    "Measurement",
    "ModelEntry",
    "ModelFile",
    "Settings",
    "SystemInfo",
    "__version__",
    "data_status",
    "detect",
    "estimate",
    "evaluate",
    "fit_list",
    "info",
    "load_db",
    "recommend",
    "search",
    "system",
]
