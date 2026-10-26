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
    plan = gmf.plan_download("Z-Image-Turbo", comfyui_dir="C:/ComfyUI")  # 落とさずに計画だけ(通信しない)
    gmf.download(plan, confirm=ask_user, progress=show)  # 同意を取ってから落とす(sha256 で確かめる)
    gmf.diff_db(gmf.load_db("old.json"), gmf.load_db("new.json"))  # 2 つの一覧の違い(`genmodelfit diff`)
    gmf.review_list()  # 確認待ちの項目(`genmodelfit review list`)。アプリは entry.reviewed で絞れる
    # アプリが調べた GPU で 1 モデルを判定する(#27)。世代と精度は genmodelfit が決める
    fit = gmf.judge_model(
        "image_z_image_turbo", gpu_name="NVIDIA GeForce RTX 5080", vram_gb=16, driver="581.57"
    )
    fit.to_dict()["fit_level"], fit.to_dict()["memory_required_gb"]
    gmf.system_from([{"name": "NVIDIA GeForce RTX 3060", "vram_gb": 12}], ram_gb=32)  # SystemInfo

互換の対象は __all__ に入る名前だけ(docs/schema.md「Python API (stable)」)。サブモジュールの関数は内部のもの。
注意: `genmodelfit.recommend` は関数で、同じ名前のモジュール(genmodelfit/recommend.py)を隠す
(`import genmodelfit.recommend as r` は関数になる。モジュールが要るなら
`importlib.import_module("genmodelfit.recommend")`。内部のもので互換の対象ではない)。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import datetime

from ._version import __version__
from .appapi import judge_model, system_from
from .db import Database
from .db import load as load_db
from .diff import diff_db
from .downloader import (
    DownloadError,
    DownloadPlan,
    DownloadProgress,
    DownloadRefused,
    DownloadResult,
    ModelNotFound,
    plan_download,
)
from .estimate import estimate
from .hardware import detect
from .i18n import get_language, set_language
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
from .remote import RemoteError, update_from_release
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
    """おすすめ順の一覧(先頭が一番のおすすめ)。`recommend --json` と同じ並べ方
    (CLI の表は自動のおすすめ(Verified)を使わずに並べる。#84)。

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


def download(
    plan_or_model, *, progress=None, confirm=None, yes: bool = False, **plan_kwargs
) -> DownloadResult:
    """モデルを落とす(CLI の `download`)。何も表示しない。

    plan_or_model: plan_download() の計画、またはモデルの名前・ID
    (そのときは plan_kwargs を plan_download に渡す。
    precision / budget_gb / output_dir / comfyui_dir / content / include_unknown_license など)。
    confirm(plan) -> bool で利用者の同意を取る(ライセンス・利用条件の URL・大きさ・空きを見せる)。
    confirm を渡さないときは yes=True で同意済みを示す(どちらも無ければ DownloadRefused)。
    progress(DownloadProgress) で進み具合を受け取る。止める理由(gated・無検閲・ライセンス不明・空き不足・
    置き場所の外・許していない取得先)は DownloadRefused / DownloadError(.plan に計画)。
    中断(KeyboardInterrupt)は .part を残すので、もう一度呼ぶと続きから落とす。
    """
    from .downloader import download as _download

    return _download(plan_or_model, progress=progress, confirm=confirm, yes=yes, **plan_kwargs)


def review_list(db: Database | None = None, *, include_ok: bool = False) -> list[dict]:
    """確認待ち(review.status が ok でない)の項目と理由(`genmodelfit review list --json` の models)。
    include_ok なら全部。ネットに触らず、何も表示しない。"""
    from .review import review_list as _list

    return _list(db or load_db(), include_ok=include_ok)


def review_approve(
    db_path: str, model_id: str, evidence: Iterable[str], *, note: str | None = None, by: str = "review"
) -> ModelEntry:
    """`genmodelfit review approve`: db_path の一覧の項目を、根拠(https の URL。1 つ以上)を付けて
    確認済みにし、ファイルを書き戻す。根拠が無い・ID が無い・確認待ちでないときは ReviewError(ValueError)。"""
    from .review import approve_file

    return approve_file(db_path, model_id, list(evidence), note=note, by=by)


def review_reject(
    db_path: str, model_id: str, note: str, *, rejected_path: str | None = None, hide_name: bool = False
) -> dict:
    """`genmodelfit review reject`: db_path の一覧から外して書き戻し、外した記録(既定は同じフォルダの
    review_rejected.txt)に 1 行足す。update は記録にある項目を入れない。"""
    from .review import reject_file

    return reject_file(db_path, model_id, note, rejected_path=rejected_path, hide_name=hide_name)


__all__ = [
    "CONTENT_LEVELS",
    "FIT_LEVELS",
    "MODALITIES",
    "DataStatus",
    "Database",
    "DownloadError",
    "DownloadPlan",
    "DownloadProgress",
    "DownloadRefused",
    "DownloadResult",
    "Estimate",
    "Fit",
    "Measurement",
    "ModelEntry",
    "ModelFile",
    "ModelNotFound",
    "RemoteError",
    "Settings",
    "SystemInfo",
    "__version__",
    "data_status",
    "detect",
    "diff_db",
    "download",
    "estimate",
    "evaluate",
    "fit_list",
    "get_language",
    "info",
    "judge_model",
    "load_db",
    "plan_download",
    "recommend",
    "review_approve",
    "review_list",
    "review_reject",
    "search",
    "set_language",
    "system",
    "system_from",
    "update_from_release",
]
