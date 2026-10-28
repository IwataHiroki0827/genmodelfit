# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`measure` と `share` の材料(#113)。何を測ると役に立つか・入っているか・時間の目安・送っていない実測。

ライブラリの関数は何も表示しない(表示は cli.py)。通信しない(ComfyUI を探すのは cli.py の comfyfind)。
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .db import Database, cache_dir
from .i18n import tr
from .models import Measurement, ModelEntry, SystemInfo

# テンプレートのある(bench で測れる)モデルの出どころ
TEMPLATE_SOURCE = "comfyui-template"
# 提案に出す段階と動かし方(#113「GPU に載り(good 以上)」)
SUGGEST_LEVELS = ("perfect", "good")
SUGGEST_MODE = "gpu"
# 時間の目安(秒)。そのモデルの実測が無いときの種類ごとのおおよその値。読み込みを含む 1 回の生成の長さで、
# 同梱の実測(RTX 5080・3070 で 2〜56 秒)より長めにした。目安であって約束ではない
TYPICAL_SECONDS = {"image": 60, "video": 300, "music": 90, "sfx": 60, "tts": 60, "3d": 120}
DEFAULT_TYPICAL_SECONDS = 120
SUBMISSIONS_DIR = "submissions"  # キャッシュの中の、送る前の提出の置き場所(bench --share と同じ)
SHARED_FILE = "submissions_shared.json"  # 送った印(内容の sha256 → 日付)


@dataclass
class Installed:
    """モデルのファイルが ComfyUI の models に入っているか。
    state は yes / no / unknown(置き場所が分からない)。"""

    state: str
    missing_bytes: int = 0
    total_bytes: int = 0
    missing_files: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def installed_state(entry: ModelEntry, roots: list[Path], refs: Any = ()) -> Installed:
    """roots(ComfyUI の models に当たるフォルダ)のどれかに、一覧のファイルが同じ大きさであるか。
    sha256 は計らない(大きさだけ。download が落とすときに計る)。
    roots が空・一覧にファイルが無いなら unknown。"""
    from .downloader import DIRECTORY_ALIASES, infer_directory, safe_relpath

    total = sum(f.size_bytes or 0 for f in entry.files)
    if not roots or not entry.files:  # 一覧にファイルが無いもの(カスタムノードが落とすなど)は分からない
        return Installed("unknown", total, total, len(entry.files))
    reverse = {v: k for k, v in DIRECTORY_ALIASES.items()}
    missing = missing_files = 0
    for f in entry.files:
        directory = f.directory or infer_directory(f, refs)[0]
        names = [directory, DIRECTORY_ALIASES.get(directory or ""), reverse.get(directory or "")]
        found = False
        for d in dict.fromkeys(n for n in names if n):
            rel = safe_relpath(d, f.name)
            if rel is None:
                continue
            for root in roots:
                p = root.joinpath(*rel)
                with contextlib.suppress(OSError):
                    if p.is_file() and (not f.size_bytes or p.stat().st_size == f.size_bytes):
                        found = True
                        break
            if found:
                break
        if not found:
            missing += f.size_bytes or 0
            missing_files += 1
    return Installed("no" if missing_files else "yes", missing, total, missing_files)


def has_measurement_here(entry: ModelEntry, measurements: list[Measurement], gpu: Any) -> bool:
    """この GPU(estimate.same_gpu)で、今の一覧のファイル(files_match)・標準の条件で最後まで測った実測
    (ok か oom)があるか(作者・この PC・ほかの利用者)。作業用メモリだけの実測(estimate.working_only。
    今の NVIDIA の bench)も数える: 同じ GPU でもう一度測っても照合の材料が増えないため。"""
    from .estimate import files_match, measured_on_this_gpu

    return any(
        m.model_id == entry.id
        and (m.outcome or "ok") in ("ok", "oom")
        and m.standard is not False
        and files_match(m, entry)
        and measured_on_this_gpu(m, gpu)
        for m in measurements
    )


def time_hint(entry: ModelEntry, measurements: list[Measurement]) -> tuple[int, str]:
    """測るのにかかる時間の目安(秒, 根拠)。根拠は measured(そのモデルのほかの GPU の実測の中央値)か
    typical(種類ごとのおおよその値 TYPICAL_SECONDS)。10 秒単位に切り上げる。モデルを落とす時間は入れない。"""
    secs = [
        float(m.seconds)
        for m in measurements
        if m.model_id == entry.id and m.seconds and m.seconds > 0 and (m.outcome or "ok") == "ok"
    ]
    if secs:
        value, basis = statistics.median(secs), "measured"
    else:
        value, basis = float(TYPICAL_SECONDS.get(entry.modality, DEFAULT_TYPICAL_SECONDS)), "typical"
    return max(10, int(-(-value // 10) * 10)), basis


@dataclass
class Suggestion:
    """測ると役に立つモデル 1 件(measure の引数なしの一覧)。"""

    model_id: str
    model_name: str
    name: str
    modality: str
    precision: str | None
    fit_level: str
    installed: Installed
    seconds: int
    seconds_basis: str
    group_key: str = ""
    fit: Any = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "name": self.name,
            "modality": self.modality,
            "precision": self.precision,
            "fit_level": self.fit_level,
            "installed": self.installed.state,
            "download_bytes": self.installed.missing_bytes,
            "total_bytes": self.installed.total_bytes,
            "seconds": self.seconds,
            "seconds_basis": self.seconds_basis,
        }


def _order(s: Suggestion) -> tuple:
    """並び: 入っているもの → 時間が短いもの → 落とす量が少ないもの → 名前。"""
    inst = s.installed
    return (inst.state != "yes", s.seconds, inst.missing_bytes, s.model_name.lower(), s.model_id)


def suggestions(
    db: Database,
    system: SystemInfo,
    *,
    roots: list[Path] | None = None,
    content: str = "sfw",
    policy: Any = None,
) -> list[Suggestion]:
    """測ると役に立つモデル(#113): テンプレートがあり、この PC の GPU に載り(perfect / good で GPU)、
    この GPU の見積もりに使える実測が無いもの。同じモデル(groups の鍵)は 1 件(入っているもの、
    無ければ recommend と同じ選び方の一番良いワークフロー)。
    並びは入っているもの → 時間 → 落とす量。GPU が無ければ空。"""
    from .downloader import directory_refs
    from .recommend import evaluate, variant_key

    gpu = system.best_gpu
    if gpu is None:
        return []
    roots = roots or []
    refs = directory_refs(db)
    per_group: dict[str, list[Suggestion]] = {}
    for f in evaluate(db, system, content=content, policy=policy):
        m, e = f.model, f.estimate
        if m.source != TEMPLATE_SOURCE or e.fit not in SUGGEST_LEVELS or e.run_mode != SUGGEST_MODE:
            continue
        if has_measurement_here(m, db.measurements, gpu):
            continue
        secs, basis = time_hint(m, db.measurements)
        s = Suggestion(
            model_id=m.id,
            model_name=f.model_name or m.name,
            name=m.name,
            modality=m.modality,
            precision=m.precision,
            fit_level=e.fit,
            installed=installed_state(m, roots, refs),
            seconds=secs,
            seconds_basis=basis,
            group_key=f.group_key or m.id,
            fit=f,
        )
        per_group.setdefault(s.group_key, []).append(s)
    out = [
        min(group, key=lambda s: (s.installed.state != "yes", variant_key(s.fit), s.model_id))
        for group in per_group.values()
    ]
    return sorted(out, key=_order)


def resolve(db: Database, system: SystemInfo, query: str, *, content: str = "sfw", policy: Any = None):
    """measure <名前か ID> → 測るワークフローの Fit。download と同じ探し方(downloader.resolve_candidates)で、
    テンプレートのある仲間だけから、ID がそのまま合えばそれ、名前なら recommend と同じ選び方の一番良いもの。
    合うものが無ければ downloader.ModelNotFound。"""
    from .downloader import ModelNotFound, resolve_candidates
    from .estimate import load_coefficients
    from .recommend import judge, variant_key

    cands, exact = resolve_candidates(db, query, content)
    usable = [m for m in cands if m.source == TEMPLATE_SOURCE]
    if not usable:
        raise ModelNotFound(
            tr(
                f"{cands[0].id} には ComfyUI のテンプレートが無いので、measure では測れません"
                "(Hugging Face のモデルは genmodelfit bench --workflow で自分のワークフローを渡して測る)",
                f"{cands[0].id} has no ComfyUI template, so measure cannot measure it"
                " (measure a Hugging Face model with genmodelfit bench --workflow and your own workflow)",
            )
        )
    c = load_coefficients()
    fits = [judge(m, system, db, coefficients=c, policy=policy) for m in usable]
    if exact:
        return fits[0]
    return min(fits, key=lambda f: (variant_key(f), f.model.id))


# --- 送る前の提出と、送った印 ---------------------------------------------------------------------


def submissions_dir() -> Path:
    return cache_dir() / SUBMISSIONS_DIR


def shared_path() -> Path:
    return cache_dir() / SHARED_FILE


def load_shared() -> dict[str, dict[str, Any]]:
    """送った印(内容の sha256 → {date, model_id})。読めなければ空。"""
    try:
        data = json.loads(shared_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    shared = data.get("shared") if isinstance(data, dict) else None
    return {k: v for k, v in shared.items() if isinstance(v, dict)} if isinstance(shared, dict) else {}


def mark_shared(data: dict[str, Any]) -> str:
    """提出に送った印を付ける(手元のファイルだけ。何も送らない)。内容の sha256 を返す。"""
    from .contrib import content_hash

    sha = content_hash(data)
    marks = load_shared()
    mm = data.get("measurement") if isinstance(data.get("measurement"), dict) else {}
    marks[sha] = {
        "date": datetime.now(timezone.utc).date().isoformat(),
        "model_id": mm.get("model_id"),
    }
    p = shared_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"shared": marks}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return sha


@dataclass
class SavedSubmission:
    """キャッシュの submissions にある提出 1 件。"""

    path: Path
    data: dict[str, Any]
    sha256: str
    shared: bool

    def to_dict(self) -> dict[str, Any]:
        mm = self.data.get("measurement") or {}
        gpu = (self.data.get("system") or {}).get("gpu") or {}
        return {
            "file": str(self.path),
            "sha256": self.sha256,
            "shared": self.shared,
            "date": self.data.get("date"),
            "model_id": mm.get("model_id"),
            "gpu": gpu.get("name"),
            "outcome": mm.get("outcome"),
            "peak_vram_gb": mm.get("peak_vram_gb"),
            "seconds": mm.get("seconds"),
        }


def saved_submissions(root: Path | None = None, *, include_shared: bool = False) -> list[SavedSubmission]:
    """送る前の提出(<キャッシュ>/submissions/<GPU>/<日付>-<sha8>.json)。新しいものから。読めないものは飛ばす。
    include_shared が偽なら送った印のあるものを除く。"""
    from .contrib import content_hash, find_files, read_submission

    root = root or submissions_dir()
    marks = load_shared()
    out: list[tuple[tuple, SavedSubmission]] = []
    for p in find_files(root):
        try:
            data, _raw = read_submission(p)
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict) or not isinstance(data.get("measurement"), dict):
            continue
        sha = content_hash(data)
        s = SavedSubmission(p, data, sha, sha in marks)
        if s.shared and not include_shared:
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            mtime = 0.0
        out.append(((str(data.get("date") or ""), mtime, p.as_posix()), s))
    return [s for _, s in sorted(out, key=lambda x: x[0], reverse=True)]
