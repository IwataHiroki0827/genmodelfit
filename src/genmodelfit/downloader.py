# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""モデルを落とす(Issue #31。`genmodelfit download` と `gmf.plan_download()` / `gmf.download()`)。

- plan_download: 落とすファイルの一覧・大きさ・ライセンス・置き場所・止める理由を、通信せずに決める
  (精度は recommend と同じ選び方: この PC で逃がさずに載る中で一番品質の高いもの。recommend.variant_key)
- download: 計画どおりに落とす。同意(confirm / yes)が無ければ始めない

安全のための約束(README「Download」):
- 一覧の URL だけを使い、転送先は 1 回ごとに許したホスト(HOST_POLICY)か確かめる。https だけ
- ~/.netrc もトークンも使わない(fetch.make_session)。PC の情報は送らない(送るのは User-Agent と Range だけ)
- 書き込みは置き場所(root)の中だけ(`..`・絶対パス・ドライブ名・シンボリックリンクで外へ出るものは拒む)
- 書きながら sha256 を計り、合わなければ消して失敗にする。`.part` から HTTP Range で続ける
  (相手が Range に応じなければ最初から)。既にあるファイルは sha256 が合えば飛ばし、合わなければ上書きしない
- 空きは「落とす残り + 1 GiB」を見る。gated・無検閲(content を指定しないとき)・ライセンス不明
  (include_unknown_license を指定しないとき)は落とさない

ライブラリの関数は何も表示しない。進み具合は progress に渡す(DownloadProgress)。
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin, urlsplit

import requests

from .estimate import GIB
from .fetch import TIMEOUT, make_session

if TYPE_CHECKING:
    from .db import Database
    from .gate import GatePolicy
    from .models import ModelEntry, ModelFile, Settings, SystemInfo
    from .recommend import Fit

log = logging.getLogger("genmodelfit")

MARGIN_BYTES = GIB  # 空きの余裕(落とす残り + 1 GiB)
CHUNK = 1024 * 1024
MAX_REDIRECTS = 10
MAX_STALLED = 4  # 進まなかった失敗(接続・時間切れ・5xx・429)をこの回数まで続けてやり直す
MAX_RETRY_AFTER = 300.0
OTHER_DIR = "other"  # ComfyUI の置き場所(directory)が一覧に無いファイルの置き場所

# 計画のファイルの状態
STATUSES = ("download", "resume", "present", "conflict", "unavailable")


class DownloadError(RuntimeError):
    """落とせなかった(通信・中身の不一致・空き不足など)。plan があれば添える。"""

    def __init__(self, message: str, plan: DownloadPlan | None = None):
        super().__init__(message)
        self.plan = plan


class DownloadRefused(DownloadError):
    """安全のための約束で落とさない(gated・無検閲・ライセンス不明・同意が無い・置き場所の外など)。"""


class DownloadDeclined(DownloadRefused):
    """同意が得られなかった(confirm が False を返した)。"""


class ModelNotFound(DownloadError):
    """名前・ID に合うモデルが無い、または 2 件以上ある(candidates に候補)。"""

    def __init__(self, message: str, candidates: list[ModelEntry] | None = None):
        super().__init__(message)
        self.candidates = candidates or []


class HostNotAllowed(DownloadError):
    """許していないホスト・https でない URL(転送先も含む)。"""


class ChecksumMismatch(DownloadError):
    """落としたファイルの sha256 が一覧と合わない(ファイルは消した)。"""


class DiskFull(DownloadError):
    """空きが足りない(落とす残り + 1 GiB)。"""


# --- 通信先 ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HostPolicy:
    """落とすときに話してよい相手。最初の URL は start_hosts、転送先は hosts か suffixes に合うもの。"""

    start_hosts: frozenset[str]
    hosts: frozenset[str]
    suffixes: tuple[str, ...]
    schemes: tuple[str, ...] = ("https",)
    any_port: bool = False  # 試験用(手元のサーバ)。既定は決まったポート(443)だけ

    def allowed(self, url: str, *, start: bool = False) -> bool:
        try:
            u = urlsplit(url)
            port = u.port
        except ValueError:
            return False
        host = (u.hostname or "").lower()
        if u.scheme not in self.schemes or not host or u.username or u.password:
            return False
        if port is not None and port != {"https": 443, "http": 80}.get(u.scheme) and not self.any_port:
            return False
        if start:
            return host in self.start_hosts
        return host in self.hosts or any(host.endswith(s) for s in self.suffixes)

    def check(self, url: str, *, start: bool = False) -> None:
        if not self.allowed(url, start=start):
            host = urlsplit(url).hostname or url
            raise HostNotAllowed(f"許していない転送先です(https の Hugging Face・GitHub だけ): {host}")


# Hugging Face(LFS は cdn-lfs*.hf.co などの CDN、Xet は cas-bridge.xethub.hf.co へ転送する)と
# GitHub(リリースの添付は objects / release-assets.githubusercontent.com へ転送する)
HOST_POLICY = HostPolicy(
    start_hosts=frozenset({"huggingface.co", "github.com"}),
    hosts=frozenset(
        {
            "huggingface.co",
            "hf.co",
            "github.com",
            "objects.githubusercontent.com",
            "release-assets.githubusercontent.com",
        }
    ),
    suffixes=(".huggingface.co", ".hf.co"),
)


# --- 置き場所 -------------------------------------------------------------------------------------

_DIR_OK = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
_BAD_CHARS = re.compile(r'[<>:"|?*\\\x00-\x1f]')
_RESERVED = re.compile(r"^(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?$", re.IGNORECASE)


def safe_relpath(directory: str | None, name: str) -> tuple[str, ...] | None:
    """置き場所の中の相対パス(部分の並び)。外へ出る・Windows で危ない名前なら None。

    name は ComfyUI のサブフォルダを含んでよい("split/vae/x.safetensors")。`..`・`.`・空の部分・
    絶対パス・ドライブ名(`:`)・`\\`・制御文字・予約名(CON など)・末尾の点や空白は拒む。"""
    folder = directory or OTHER_DIR
    if not _DIR_OK.match(folder) or folder in (".", ".."):
        return None
    parts = name.split("/")
    for p in parts:
        if (
            not p
            or p in (".", "..")
            or _BAD_CHARS.search(p)
            or _RESERVED.match(p)
            or p.endswith((".", " "))
            or len(p) > 200
        ):
            return None
    return (folder, *parts)


def _inside(path: Path, root: Path) -> bool:
    try:
        return path.resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):
        return False


def default_output_dir() -> Path:
    """--output-dir も --comfyui-dir も無いときの置き場所(利用者のデータの場所の genmodelfit/models)。"""
    from platformdirs import user_data_dir

    return Path(user_data_dir("genmodelfit", appauthor=False)) / "models"


def comfyui_models_dir(comfyui_dir: str | os.PathLike[str]) -> Path:
    """ComfyUI のフォルダ → models。ComfyUI に見えない(models も main.py も無い)なら DownloadRefused。"""
    base = Path(comfyui_dir).expanduser()
    if not (base / "models").is_dir() and not (base / "main.py").is_file():
        raise DownloadRefused(
            f"ComfyUI のフォルダに見えません(models も main.py もありません): {base}"
            "(ComfyUI を入れたフォルダを指定してください。別の場所なら --output-dir)"
        )
    return base / "models"


def _free_bytes(path: Path) -> int | None:
    p = path
    while not p.exists():
        if p.parent == p:
            return None
        p = p.parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None


# --- 計画 -----------------------------------------------------------------------------------------


@dataclass
class PlannedFile:
    """落とす 1 ファイル。status は STATUSES(unavailable / conflict があれば落とさない)。"""

    name: str
    url: str | None
    sha256: str | None
    size_bytes: int | None
    directory: str | None
    role: str | None
    license: str | None  # 判定に使うライセンス(gate.resolve_file_license。不明は None)
    license_name: str | None
    license_url: str | None
    license_note: str | None  # 一覧の値と違う読み方をした理由・確かめられない理由
    license_allowed: bool  # 許可リスト(gate の既定か --license-allowlist)にあるか
    gated: bool | None
    dest: str  # 置く場所(絶対パス)。置けないときは ""
    relpath: str  # 置き場所の中の相対パス(/ 区切り)
    status: str = "download"
    part_bytes: int = 0  # 途中まで落とした .part の大きさ
    reason: str | None = None  # unavailable / conflict の理由

    @property
    def remaining_bytes(self) -> int:
        if self.status not in ("download", "resume"):
            return 0
        return max(0, (self.size_bytes or 0) - self.part_bytes)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["remaining_bytes"] = self.remaining_bytes
        return d


@dataclass
class DownloadPlan:
    """落とす計画(plan_download の結果)。blocked が空でなければ download は始めない。"""

    model_id: str
    name: str
    model_name: str
    modality: str
    content: str
    precision: str | None
    fit_level: str
    run_mode_label: str
    auto: bool
    layout: str  # comfyui(ComfyUI の models)/ output(--output-dir か既定の場所。同じ並び)
    root: str  # 置き場所(models に当たるフォルダ)
    files: list[PlannedFile] = field(default_factory=list)
    disk_free_bytes: int | None = None
    budget_bytes: int | None = None
    blocked: list[str] = field(default_factory=list)  # 落とさない理由
    warnings: list[str] = field(default_factory=list)  # 落とせるが知らせること
    alternatives: list[dict[str, Any]] = field(default_factory=list)  # 同じモデルのほかの精度・使い方
    license_url: str | None = None  # モデル一式のライセンスの場所
    chosen: str = ""  # なぜこのワークフローを選んだか

    @property
    def total_bytes(self) -> int:
        return sum(f.size_bytes or 0 for f in self.files)

    @property
    def download_bytes(self) -> int:
        """これから落とす残り(既にある・.part の分を除く)。"""
        return sum(f.remaining_bytes for f in self.files)

    @property
    def present_bytes(self) -> int:
        return sum(f.size_bytes or 0 for f in self.files if f.status == "present")

    @property
    def disk_needed_bytes(self) -> int:
        return self.download_bytes + MARGIN_BYTES if self.download_bytes else 0

    @property
    def ok(self) -> bool:
        return not self.blocked

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["files"] = [f.to_dict() for f in self.files]
        d.update(
            total_bytes=self.total_bytes,
            download_bytes=self.download_bytes,
            present_bytes=self.present_bytes,
            disk_needed_bytes=self.disk_needed_bytes,
            margin_bytes=MARGIN_BYTES,
            ok=self.ok,
        )
        return d


def _same_use(m: ModelEntry, members: Iterable[ModelEntry]) -> list[ModelEntry]:
    from .tasks import main_use

    use = main_use(m)
    return [x for x in members if main_use(x) == use]


def resolve_candidates(db: Database, query: str, content: str = "sfw") -> tuple[list[ModelEntry], bool]:
    """名前か ID → 候補のワークフロー(同じモデルの仲間)と、ID がそのまま合ったか。

    info と同じ順: ID がそのまま合えばそれ(内容によらず)/ モデルの表示名(大文字小文字を区別しない)
    ならその仲間 / ID・名前の一部(content で絞る)。一部の一致が 2 件以上で、同じモデルの仲間でなければ
    ModelNotFound(candidates に候補。先頭を黙って選ばない)。"""
    from .groups import index_for

    q = query.strip()
    if not q:
        raise ModelNotFound("モデルの名前か ID が空です")
    exact = db.get(q)
    if exact is not None:
        return [exact], True
    idx = index_for(db)
    key = idx.find(q)
    if key is not None:
        return list(idx.members[key]), False
    ql = q.lower()
    hits = [m for m in db.models if ql in m.id.lower() or ql in m.name.lower()]
    shown = [m for m in hits if content == "all" or m.content == content]
    if shown:
        keys = {idx.key_of.get(m.id) for m in shown}
        if len(keys) == 1:
            return list(idx.members[keys.pop()]), False
        if len(shown) == 1:
            return shown, False
        raise ModelNotFound(
            f"「{q}」に合うモデルが {len(shown)} 件あります。ID かモデル名で指定してください", shown
        )
    if hits:
        kinds = "・".join(sorted({m.content for m in hits}))
        raise ModelNotFound(
            f"モデルが見つかりません: {q}(content {content} で外れた {kinds} のモデルが "
            f"{len(hits)} 件あります。"
            "--content all を付けるか ID で指定してください)"
        )
    raise ModelNotFound(f"モデルが見つかりません: {q}")


def _hash_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
            if progress:
                progress(len(b))
    return h.hexdigest()


def _plan_file(f: ModelFile, root: Path, allow: frozenset[str], verify_existing: bool) -> PlannedFile:
    from .gate import license_allowed, resolve_file_license

    r = resolve_file_license(f)
    rel = safe_relpath(f.directory, f.name)
    dest = root.joinpath(*rel) if rel else None
    pf = PlannedFile(
        name=f.name,
        url=f.url,
        sha256=(f.sha256 or "").lower() or None,
        size_bytes=f.size_bytes,
        directory=f.directory,
        role=f.role,
        license=r.license,
        license_name=r.license_name,
        license_url=f.license_url,
        license_note=r.note,
        license_allowed=license_allowed(r.license, r.license_name, allow),
        gated=True if (f.gated or r.gated) else f.gated,
        dest=str(dest) if dest else "",
        relpath="/".join(rel) if rel else "",
    )
    reason = None
    if rel is None or dest is None or not _inside(dest, root):
        reason = f"置き場所の外を指す名前です({f.directory or '-'}/{f.name})"
    elif not f.url or not HOST_POLICY.allowed(f.url, start=True):
        host = urlsplit(f.url).hostname if f.url else "-"
        reason = f"許していない取得先です({host}。https の Hugging Face・GitHub だけ)"
    elif not pf.sha256 or not re.fullmatch(r"[0-9a-f]{64}", pf.sha256):
        reason = "一覧に sha256 が無く、中身を確かめられません"
    elif not f.size_bytes or f.size_bytes <= 0:
        reason = "一覧に大きさが無く、空きを確かめられません"
    if reason:
        pf.status, pf.reason = "unavailable", reason
        return pf
    assert dest is not None
    part = dest.with_name(dest.name + ".part")
    for p in (dest, part):
        if p.is_symlink():
            pf.status, pf.reason = "conflict", f"シンボリックリンクには書きません: {p}"
            return pf
    if dest.exists():
        if not dest.is_file() or dest.stat().st_size != f.size_bytes:
            pf.status, pf.reason = "conflict", "同じ名前の別のファイルがあります(上書きしません)"
        elif not verify_existing or _hash_file(dest) == pf.sha256:
            pf.status = "present"
        else:
            pf.status, pf.reason = "conflict", "同じ名前のファイルの sha256 が違います(上書きしません)"
        return pf
    if part.is_file():
        size = part.stat().st_size
        if 0 < size < (f.size_bytes or 0) or size == f.size_bytes:
            pf.status, pf.part_bytes = "resume", size
    return pf


def _alt_dict(fit: Fit) -> dict[str, Any]:
    from .recommend import memory_use

    _, need, _ = memory_use(fit.estimate, fit.system)
    return {
        "id": fit.model.id,
        "precision": fit.model.precision,
        "use": fit.use,
        "fit_level": fit.estimate.fit,
        "memory_required_gb": round(need, 2) if need is not None else None,
        "size_bytes": fit.model.total_size_bytes,
        "auto": fit.auto,
    }


def _model_blocks(
    m: ModelEntry, content: str, include_unknown_license: bool, files: list[PlannedFile]
) -> list[str]:
    out = []
    if m.content == "uncensored" and content not in ("all", "uncensored"):
        out.append("無検閲のモデルです。落とすなら --content all か --content uncensored を明示してください")
    gated = [f.name for f in files if f.gated]
    if gated:
        out.append(
            "利用申請(gated)が要るファイルがあります: "
            + ", ".join(gated[:3])
            + "。genmodelfit はログイン情報を扱わないので落としません"
            "(配布元のページで申請して自分で落としてください)"
        )
    unknown = [f.name for f in files if f.license is None]
    if unknown and not include_unknown_license:
        out.append(
            "ライセンスが分からないファイルがあります: "
            + ", ".join(unknown[:3])
            + "。中身の条件を自分で確かめたうえで落とすなら --include-unknown-license"
        )
    if not m.files:
        out.append("一覧にファイルの記載がありません")
    return out


def plan_download(
    model: str,
    *,
    precision: str | None = None,
    budget_gb: float | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    comfyui_dir: str | os.PathLike[str] | None = None,
    content: str = "sfw",
    include_unknown_license: bool = False,
    license_allowlist: Iterable[str] | None = None,
    verify_existing: bool = True,
    settings: Settings | None = None,
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
    policy: GatePolicy | None = None,
) -> DownloadPlan:
    """落とす計画を作る(通信しない・何も書かない)。CLI の `download --list` と同じ。

    model: モデルの表示名・ワークフローの ID・名前の一部(info と同じ探し方)。ID がそのまま合えばその
    ワークフロー、表示名・一部なら同じモデルの仲間から、この PC で逃がさずに載る中で一番精度の高いもの
    (recommend と同じ選び方)。precision: その精度の仲間だけ(ID を指定したときも、同じ使い方の仲間から選ぶ)。
    budget_gb: 落とす量(既にあるファイルを除く。GiB)の上限。超えない中で一番良いものを選ぶ。
    output_dir / comfyui_dir: 置き場所(両方は ValueError)。comfyui_dir なら <dir>/models/<directory>/<名前>、
    output_dir(既定は default_output_dir())も同じ並び。content: 無検閲は "all" / "uncensored" のときだけ
    落とせる。include_unknown_license: ライセンスの分からないファイルを含んでも落とせる。
    verify_existing: 既に置いてあるファイルの sha256 を計る(False なら大きさだけ)。
    止める理由は plan.blocked、知らせることは plan.warnings(例外にはしない)。名前が合わない・
    2 件以上合う・置き場所が ComfyUI に見えないときは ModelNotFound / DownloadRefused。
    """
    from .db import load as load_db
    from .estimate import load_coefficients
    from .gate import DEFAULT_LICENSE_ALLOWLIST, GatePolicy
    from .groups import index_for, precision_label
    from .hardware import detect
    from .recommend import CONTENT_FILTERS, judge, variant_key

    if content not in CONTENT_FILTERS:
        raise ValueError(f"content が正しくありません: {content!r}(使えるのは {', '.join(CONTENT_FILTERS)})")
    if output_dir is not None and comfyui_dir is not None:
        raise ValueError("output_dir と comfyui_dir は同時に指定できません")
    if budget_gb is not None and (isinstance(budget_gb, bool) or not budget_gb > 0):
        raise ValueError(f"budget_gb は 0 より大きい数です: {budget_gb!r}")
    db = db or load_db()
    system = system_info or detect()
    pol = policy or GatePolicy(comfyui_version=comfyui_version)
    allow = frozenset(a.lower() for a in license_allowlist) if license_allowlist else pol.license_allowlist
    allow = allow or DEFAULT_LICENSE_ALLOWLIST
    if comfyui_dir is not None:
        root, layout = comfyui_models_dir(comfyui_dir), "comfyui"
    else:
        root, layout = (Path(output_dir).expanduser() if output_dir else default_output_dir()), "output"
    root = Path(os.path.abspath(root))

    cands, exact = resolve_candidates(db, model, content)
    idx = index_for(db)
    if precision:
        want = precision.strip().lower()
        pool = cands
        if exact:  # ID を指定して精度も指定: 同じモデル・同じ使い方の仲間から
            pool = _same_use(cands[0], idx.members[idx.key_of[cands[0].id]])
        cands = [m for m in pool if (m.precision or "").lower() == want]
        if not cands:
            have = sorted({precision_label(m.precision) for m in pool})
            raise ModelNotFound(f"精度 {precision} のワークフローがありません(あるのは {', '.join(have)})")
    c = load_coefficients()
    fits = [judge(m, system, db, settings=settings, coefficients=c, policy=pol) for m in cands]
    budget = int(budget_gb * GIB) if budget_gb is not None else None

    plans: list[tuple[tuple, DownloadPlan, Fit]] = []
    for fit in fits:
        m = fit.model
        files = [_plan_file(f, root, allow, verify_existing and len(fits) == 1) for f in m.files]
        p = DownloadPlan(
            model_id=m.id,
            name=m.name,
            model_name=fit.model_name,
            modality=m.modality,
            content=m.content,
            precision=m.precision,
            fit_level=fit.estimate.fit,
            run_mode_label=fit.run_mode_label,
            auto=fit.auto,
            layout=layout,
            root=str(root),
            files=files,
            budget_bytes=budget,
            license_url=m.license_url,
        )
        p.blocked = _model_blocks(m, content, include_unknown_license, files)
        over = budget is not None and p.download_bytes > budget
        # 候補の順: 止める理由が無い → 上限に収まる → recommend と同じ選び方(variant_key)
        plans.append(((bool(p.blocked), over, *variant_key(fit)), p, fit))
    plans.sort(key=lambda x: x[0])
    _, plan, fit = plans[0]
    if not exact and len(plans) > 1 and verify_existing:
        plan.files = [_plan_file(f, root, allow, True) for f in fit.model.files]  # 選んだものだけ計り直す
    plan.alternatives = [_alt_dict(x[2]) for x in plans[1:]]
    plan.chosen = (
        "ID で指定"
        if exact and not precision
        else f"この PC で逃がさずに載る中で一番精度の高いもの({len(plans)} 件から。recommend と同じ選び方)"
    )
    _finish(plan, fit, allow)
    return plan


def _finish(plan: DownloadPlan, fit: Fit, allow: frozenset[str]) -> None:
    """計画の止める理由・知らせることを足す(ファイルの状態と空き)。"""
    root = Path(plan.root)
    for f in plan.files:
        if f.status == "unavailable":
            plan.blocked.append(f"{f.name}: {f.reason}")
        elif f.status == "conflict":
            plan.blocked.append(f"{f.relpath}: {f.reason}(別の場所へ移すか、--output-dir で別の置き場所を)")
    if plan.budget_bytes is not None and plan.download_bytes > plan.budget_bytes:
        plan.blocked.append(
            f"落とす量 {plan.download_bytes / GIB:.2f} GB が上限 "
            f"{plan.budget_bytes / GIB:.2f} GB を超えます(--budget)"
        )
    plan.disk_free_bytes = _free_bytes(root)
    if plan.disk_free_bytes is not None and plan.disk_needed_bytes > plan.disk_free_bytes:
        plan.blocked.append(
            f"空きが足りません: 要る {plan.disk_needed_bytes / GIB:.2f} GB(落とす残り + 1 GB)、"
            f"空き {plan.disk_free_bytes / GIB:.2f} GB"
        )
    names = [f"{f.name}({_lic(f)})" for f in plan.files if f.license is not None and not f.license_allowed]
    if names:
        plan.warnings.append(
            "許可リスト(商用で使えて使い方の制限が無いライセンス)に無いライセンスのファイルがあります: "
            + ", ".join(names[:4])
            + "。それぞれの条件を読んでから使ってください"
        )
    unknown = [f.name for f in plan.files if f.license is None]
    if unknown and not plan.blocked:
        plan.warnings.append(
            "ライセンスが分からないファイルを含みます(--include-unknown-license): " + ", ".join(unknown[:4])
        )
    if fit.estimate.fit in ("too_tight",):
        plan.warnings.append("この PC では動かない見込みです(fit too_tight)")
    elif fit.estimate.unknown_size:
        plan.warnings.append("大きさが分からず、この PC で動くか判定できていません")
    if plan.layout == "output":
        plan.warnings.append(
            "ComfyUI の models と同じ並びで置きます。ComfyUI で使うなら --comfyui-dir を指定するか、"
            "extra_model_paths.yaml にこのフォルダを足してください"
        )


def _lic(f: PlannedFile) -> str:
    if f.license is None:
        return "不明"
    return f"other: {f.license_name}" if f.license == "other" and f.license_name else f.license


license_label = _lic


# --- 落とす ---------------------------------------------------------------------------------------


@dataclass
class DownloadProgress:
    """progress に渡す進み具合。phase: check(既にあるファイル・.part を計る)/ download / restart
    (相手が Range に応じず最初から)/ retry(つながらず待つ)/ verify / skip(既にある)/
    done(1 ファイル終わり)。"""

    phase: str
    file_index: int  # 1 から
    file_count: int
    name: str
    file_bytes: int  # このファイルの済んだ量
    file_total: int | None
    done_bytes: int  # 全体の済んだ量(今回落とした分 + .part + 既にある分)
    total_bytes: int
    message: str | None = None


@dataclass
class DownloadResult:
    plan: DownloadPlan
    downloaded: list[str] = field(default_factory=list)  # 落とした(相対パス)
    resumed: list[str] = field(default_factory=list)  # .part から続けた
    restarted: list[str] = field(default_factory=list)  # Range に応じず最初から落とし直した
    skipped: list[str] = field(default_factory=list)  # 既にあって sha256 が合った
    bytes_downloaded: int = 0  # 今回の通信で受け取った量
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.plan.model_id,
            "root": self.plan.root,
            "downloaded": list(self.downloaded),
            "resumed": list(self.resumed),
            "restarted": list(self.restarted),
            "skipped": list(self.skipped),
            "bytes_downloaded": self.bytes_downloaded,
            "seconds": round(self.seconds, 1),
            "files": [{"relpath": f.relpath, "dest": f.dest, "sha256": f.sha256} for f in self.plan.files],
        }


def _retry_after(resp: requests.Response, n: int) -> float:
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(resp.headers.get("Retry-After", ""))))
    except ValueError:
        return float(min(60, 2**n))


def _open(
    session: requests.Session, url: str, headers: dict[str, str], hosts: HostPolicy, timeout
) -> requests.Response:
    """GET(転送は 1 回ごとに許したホストか確かめて自分で追う)。"""
    hosts.check(url, start=True)
    for _ in range(MAX_REDIRECTS + 1):
        resp = session.get(url, headers=headers, stream=True, allow_redirects=False, timeout=timeout)
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("location")
            resp.close()
            if not loc:
                raise DownloadError(f"転送先がありません: {url}")
            url = urljoin(url, loc)
            hosts.check(url)
            log.debug("download: 転送 -> %s", urlsplit(url).hostname)
            continue
        return resp
    raise DownloadError(f"転送が多すぎます: {url}")


def _content_range_start(resp: requests.Response) -> int | None:
    m = re.match(r"bytes (\d+)-\d+/(\d+|\*)", resp.headers.get("content-range", ""))
    return int(m.group(1)) if m else None


def _fetch_one(
    pf: PlannedFile,
    *,
    session: requests.Session,
    hosts: HostPolicy,
    root: Path,
    sleep: Callable[[float], None],
    timeout,
    report: Callable[[str, int, str | None], None],
    result: DownloadResult,
) -> None:
    dest = Path(pf.dest)
    part = dest.with_name(dest.name + ".part")
    size = pf.size_bytes or 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _inside(dest, root) or dest.is_symlink() or part.is_symlink():
        raise DownloadRefused(f"置き場所の外には書きません: {dest}")
    hasher, hashed = None, -1  # .part のどこまでを hasher に入れたか
    stalled = 0
    resumed = restarted = False
    while True:
        have = part.stat().st_size if part.is_file() else 0
        if have > size:
            part.unlink()
            have = 0
        if hashed != have:  # .part の中身を計り直す(初回・やり直しのあと)
            report("check", have, None)
            hasher = hashlib.sha256()
            if have:
                with part.open("rb") as fh:
                    for b in iter(lambda: fh.read(CHUNK), b""):
                        hasher.update(b)
            hashed = have
        assert hasher is not None
        if have == size:
            break
        headers = {"Accept-Encoding": "identity"}
        if have:
            headers["Range"] = f"bytes={have}-"
        got = 0
        try:
            resp = _open(session, pf.url or "", headers, hosts, timeout)
            try:
                code = resp.status_code
                if code == 416 and have:  # 範囲の外: .part を捨てて最初から
                    part.unlink()
                    hashed = -1
                    stalled += 1
                    if stalled >= MAX_STALLED:
                        raise DownloadError(f"{pf.name}: 範囲の要求が通りません(HTTP 416)")
                    continue
                if code in (429, 500, 502, 503, 504):
                    stalled += 1
                    if stalled >= MAX_STALLED:
                        raise DownloadError(f"{pf.name}: HTTP {code} が続きました")
                    wait = _retry_after(resp, stalled)
                    report("retry", have, f"HTTP {code}。{wait:.0f} 秒待ってやり直します")
                    sleep(wait)
                    continue
                if code in (401, 403):
                    raise DownloadRefused(
                        f"{pf.name}: HTTP {code}(ログインか利用申請が要るファイルです。"
                        "genmodelfit はログイン情報を扱わないので落としません)"
                    )
                if code >= 400:
                    raise DownloadError(f"{pf.name}: HTTP {code}")
                if code == 206 and have and _content_range_start(resp) == have:
                    mode = "ab"
                    resumed = True
                elif code == 206:  # 頼んだ位置と違う続き: .part を捨てて最初から
                    part.unlink(missing_ok=True)
                    hashed = -1
                    stalled += 1
                    if stalled >= MAX_STALLED:
                        raise DownloadError(f"{pf.name}: 続きの位置が合いません(HTTP 206)")
                    continue
                elif code == 200:
                    if have:  # 相手が Range に応じない: 最初から
                        restarted = True
                        report("restart", 0, "相手が続きからの取得(Range)に応じないので最初から落とします")
                    have, hasher, hashed, mode = 0, hashlib.sha256(), 0, "wb"
                else:
                    raise DownloadError(f"{pf.name}: 思わない応答です(HTTP {code})")
                with part.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=CHUNK):
                        if not chunk:
                            continue
                        if have + got + len(chunk) > size:
                            raise DownloadError(
                                f"{pf.name}: 一覧の大きさ({size} バイト)より大きいので止めました"
                            )
                        fh.write(chunk)
                        hasher.update(chunk)
                        got += len(chunk)
                        hashed = have + got
                        result.bytes_downloaded += len(chunk)
                        report("download", have + got, None)
            finally:
                resp.close()
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as e:
            stalled = 0 if got else stalled + 1
            if stalled >= MAX_STALLED:
                raise DownloadError(
                    f"{pf.name}: つながりません({type(e).__name__})。もう一度実行すると続きから落とします"
                ) from e
            wait = float(min(60, 2**stalled))
            report("retry", have + got, f"つながりません({type(e).__name__})。{wait:.0f} 秒待って続きから")
            sleep(wait)
            continue
        if got:
            stalled = 0
        elif have + got < size:
            stalled += 1
            if stalled >= MAX_STALLED:
                raise DownloadError(f"{pf.name}: 途中で切れ、続きを受け取れません")
    report("verify", size, None)
    if hasher.hexdigest() != pf.sha256:
        part.unlink(missing_ok=True)
        raise ChecksumMismatch(
            f"{pf.name}: sha256 が一覧と合いません(消しました)。一覧 {pf.sha256}、"
            f"落としたもの {hasher.hexdigest()}"
        )
    os.replace(part, dest)
    result.downloaded.append(pf.relpath)
    if resumed:
        result.resumed.append(pf.relpath)
    if restarted:
        result.restarted.append(pf.relpath)


def download(
    plan_or_model: DownloadPlan | str,
    *,
    progress: Callable[[DownloadProgress], None] | None = None,
    confirm: Callable[[DownloadPlan], bool] | None = None,
    yes: bool = False,
    sleep: Callable[[float], None] = time.sleep,
    timeout=TIMEOUT,
    **plan_kwargs: Any,
) -> DownloadResult:
    """計画どおりに落とす(何も表示しない)。plan_or_model が文字列なら
    plan_download(plan_or_model, **plan_kwargs)。

    同意: confirm(plan) が True を返したときだけ始める(ライセンス・大きさ・置き場所を見せて尋ねる関数)。
    confirm を渡さないなら yes=True で同意済みを示す(どちらも無ければ DownloadRefused)。
    plan.blocked があれば DownloadRefused、空きが足りなければ DiskFull、sha256 が合わなければ ChecksumMismatch
    (そのファイルは消す)、許していない転送先なら HostNotAllowed。中断(KeyboardInterrupt)は .part を残して
    そのまま上げる(もう一度呼ぶと続きから)。
    """
    if isinstance(plan_or_model, DownloadPlan):
        if plan_kwargs:
            raise TypeError("計画を渡すときは plan_download の引数を渡せません: " + ", ".join(plan_kwargs))
        plan = plan_or_model
    else:
        plan = plan_download(plan_or_model, **plan_kwargs)
    if plan.blocked:
        raise DownloadRefused("落とせません: " + " / ".join(plan.blocked), plan)
    if confirm is not None:
        if not confirm(plan):
            raise DownloadDeclined("同意が得られなかったので落としません", plan)
    elif not yes:
        raise DownloadRefused("同意がありません(confirm を渡すか yes=True)", plan)
    root = Path(plan.root)
    hosts = HOST_POLICY
    todo = [f for f in plan.files if f.status in ("download", "resume")]
    # 空き(計画のあとで変わっていることがあるので、始める前に見直す)
    need = sum(max(0, (f.size_bytes or 0) - _part_size(f)) for f in todo)
    free = _free_bytes(root)
    if todo and free is not None and need + MARGIN_BYTES > free:
        raise DiskFull(
            f"空きが足りません: 要る {(need + MARGIN_BYTES) / GIB:.2f} GB(落とす残り + 1 GB)、"
            f"空き {free / GIB:.2f} GB",
            plan,
        )
    result = DownloadResult(plan)
    start = time.monotonic()
    total = plan.total_bytes
    base = [0]  # 済んだファイルの合計

    def emit(i: int, f: PlannedFile, phase: str, n: int, msg: str | None) -> None:
        if progress:
            progress(
                DownloadProgress(
                    phase, i, len(plan.files), f.relpath, n, f.size_bytes, base[0] + n, total, msg
                )
            )

    root.mkdir(parents=True, exist_ok=True)
    session = make_session()
    try:
        for i, f in enumerate(plan.files, 1):
            dest = Path(f.dest)
            if f.status == "present" or (
                dest.is_file() and dest.stat().st_size == f.size_bytes and _verified(dest, f)
            ):
                result.skipped.append(f.relpath)
                emit(i, f, "skip", f.size_bytes or 0, "既にあり sha256 が合うので飛ばします")
            else:
                if dest.exists() or dest.is_symlink():
                    raise DownloadRefused(
                        f"{f.relpath}: 同じ名前の別のファイルがあります(上書きしません)", plan
                    )
                _fetch_one(
                    f,
                    session=session,
                    hosts=hosts,
                    root=root,
                    sleep=sleep,
                    timeout=timeout,
                    report=lambda phase, n, msg, i=i, f=f: emit(i, f, phase, n, msg),
                    result=result,
                )
            base[0] += f.size_bytes or 0
            emit(i, f, "done", 0, None)
    finally:
        session.close()
        result.seconds = time.monotonic() - start
    log.info(
        "download %s: %d files, %d bytes", plan.model_id, len(result.downloaded), result.bytes_downloaded
    )
    return result


def _part_size(f: PlannedFile) -> int:
    p = Path(f.dest + ".part") if f.dest else None
    try:
        return p.stat().st_size if p and p.is_file() else 0
    except OSError:
        return 0


def _verified(dest: Path, f: PlannedFile) -> bool:
    return _hash_file(dest) == f.sha256
