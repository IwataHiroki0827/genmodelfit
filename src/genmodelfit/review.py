# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""確認の印(ModelEntry.review)を付ける・確かめる・外す(#42)。

毎週の更新(Claude Code のスキル genmodelfit-weekly)で、新しいモデルやライセンスの変化を一覧に入れても、
人(またはスキル)が一次資料で確かめるまで自動のおすすめに出さないための印。

- update は前の一覧(引き継ぎの元)と比べて、次の項目を `pending` にする(理由は `reasons`):
  - `new`                 前の一覧に無い項目
  - `license_permissive`  ライセンスが寛容な側に変わった(段階が下がった・既定の許可リストに入るようになった・
                          gate の license が通るようになった)
  - `need_decreased`      見積もりの必要量が減った(REFERENCE_PCS のどれかで、段階が上がった・大きさが分かる
                          ようになった・全部載せの VRAM か RAM が 1% を超えて減った。精度・大きさ・
                          ファイルの変化による)
  - `uncensored_new`      新しく無検閲になった(新しい項目が無検閲のときも付ける)
  それ以外は前の印を引き継ぐ(前の一覧に印が無ければ、確認済みとして引き継いだことを note に書く)。
- 外した項目(`review reject`)は data/review_rejected.txt に残し、update は作り直しても入れない。
- gate の `reviewed` が pending を自動のおすすめに出さない。

ライブラリとして何も表示しない(表示は cli.py)。ネットには触らない。
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import REVIEW_REASONS, Measurement, ModelEntry, Review, SystemInfo

# 同梱の今の一覧に付けた印の decided_by(すべて ok で始めた。2026-10-21)
BASELINE = "baseline-2026-10-21"
# 外した項目のファイル(同梱のデータ。リポジトリでは src/genmodelfit/data/review_rejected.txt)
REJECTED_FILE = "review_rejected.txt"
# 必要量が「減った」とみなす割合(見積もりの丸めの揺れを拾わない)
NEED_DECREASE_RATIO = 0.01
# 必要量を比べる PC(VRAM GB, RAM GB)。0 は GPU なし(CPU で動かす判定)
REFERENCE_PCS: tuple[tuple[float, float], ...] = ((0, 16), (6, 16), (8, 16), (12, 32), (16, 32), (24, 64))
# 印が無い前の一覧から引き継いだときの note
LEGACY_NOTE = "前の一覧に確認の印が無い(0.6.0 より前)ため、確認済みとして引き継いだ"
_EVIDENCE_PREFIX = "https://"
_HASH_PREFIX = "sha256:"


class ReviewError(ValueError):
    """review の操作ができない(ID が無い・根拠が無い・確認待ちでない など)。文は利用者に見せる。"""


# --- 比べる -----------------------------------------------------------------------------------------

# 必要量に関わらない項目(違っても見積もりを比べ直さない)。ここに無い項目が変われば比べ直す(安全側)
_ENTRY_SKIP = frozenset(
    {
        "review",
        "name",
        "downloads",
        "likes",
        "downloads_all_time",
        "discovered_from",
        "updated",
        "homepage",
        "languages",
        "description",
        "card_languages",
        "license",
        "license_name",
        "license_url",
        "license_evidence",
        "content",
        "content_evidence",
        "base_model",
        "total_size_bytes",
    }
)
_FILE_SKIP = frozenset(
    {
        "url",
        "sha256",
        "head_status",
        "head_checked_at",
        "license",
        "license_name",
        "license_from",
        "license_url",
        "repo_license",
        "license_path",
        "license_evidence",
        "gated",
    }
)
_TEMPLATE_SKIP = frozenset({"title", "usage", "date", "username", "open_source"})


def _need_key(m: ModelEntry) -> dict[str, Any]:
    d = {k: v for k, v in asdict(m).items() if k not in _ENTRY_SKIP}
    d["files"] = [{k: v for k, v in f.items() if k not in _FILE_SKIP} for f in d.get("files") or []]
    if isinstance(d.get("template"), dict):
        d["template"] = {k: v for k, v in d["template"].items() if k not in _TEMPLATE_SKIP}
    return d


def _pc(vram: float, ram: float) -> SystemInfo:
    from .models import Gpu

    gpus = [Gpu(name="(review)", vendor="nvidia", vram_total_gb=vram, backend="cuda")] if vram else []
    return SystemInfo(os="-", cpu="-", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=gpus)


def need_decreased(old: ModelEntry, new: ModelEntry, measurements: Iterable[Measurement] = ()) -> bool:
    """old → new で見積もりの必要量が減ったか。REFERENCE_PCS のどれか 1 台で、段階が上がった・大きさが
    分からなかったのが分かるようになった、または全部載せの VRAM(vram_gb)かメインメモリ(ram_gb)が
    NEED_DECREASE_RATIO を超えて減った。逃がす・載せ替えのときの VRAM は比べない(部品の大きさの並びで
    上下し、効く変化は段階に出る)。必要量に関わる項目が同じなら見積もらない。"""
    if _need_key(old) == _need_key(new):
        return False
    from .estimate import estimate
    from .models import fit_rank

    ms = list(measurements)
    for vram, ram in REFERENCE_PCS:
        pc = _pc(vram, ram)
        a, b = estimate(old, pc, None, ms), estimate(new, pc, None, ms)
        if b.unknown_size:
            continue
        if a.unknown_size or fit_rank(b.fit) < fit_rank(a.fit):
            return True
        for va, vb in ((a.vram_gb, b.vram_gb), (a.ram_gb, b.ram_gb)):
            if va > 0 and vb < va * (1 - NEED_DECREASE_RATIO):
                return True
    return False


def license_permissive(old: ModelEntry, new: ModelEntry, allow: frozenset[str] | None = None) -> bool:
    """ライセンスが寛容な側に変わったか: 一式の段階(hf.license_tier)が下がった、一式のライセンスが許可リストに
    入るようになった、gate の license(ファイルごとに本家で読む)が通るようになった。"""
    from .gate import DEFAULT_LICENSE_ALLOWLIST, check_license, license_allowed
    from .hf import license_tier

    allow = DEFAULT_LICENSE_ALLOWLIST if allow is None else allow
    if license_tier(new.license, new.license_name) < license_tier(old.license, old.license_name):
        return True
    if license_allowed(new.license, new.license_name, allow) and not license_allowed(
        old.license, old.license_name, allow
    ):
        return True
    return check_license(new, allow).passed and not check_license(old, allow).passed


def review_reasons(
    old: ModelEntry | None,
    new: ModelEntry,
    measurements: Iterable[Measurement] = (),
    allow: frozenset[str] | None = None,
) -> list[str]:
    """old(前の一覧の同じ ID。無ければ None)→ new で、確認待ちにする理由(REVIEW_REASONS の順)。"""
    if old is None:
        return ["new"] + (["uncensored_new"] if new.content == "uncensored" else [])
    out = []
    if license_permissive(old, new, allow):
        out.append("license_permissive")
    if need_decreased(old, new, measurements):
        out.append("need_decreased")
    if old.content != "uncensored" and new.content == "uncensored":
        out.append("uncensored_new")
    return out


def _ordered(reasons: Iterable[str]) -> list[str]:
    rs = set(reasons)
    return [r for r in REVIEW_REASONS if r in rs] + sorted(rs - set(REVIEW_REASONS))


def apply_reviews(
    entries: list[ModelEntry],
    previous: Iterable[ModelEntry] | None,
    measurements: Iterable[Measurement] = (),
) -> dict[str, Any]:
    """entries のすべてに review を付ける(前の一覧 previous と比べて。None は前の一覧が無い = 全部 new)。

    理由があれば pending(前も pending なら理由を足す)。理由が無ければ前の印を引き継ぐ(前の一覧に印が
    無ければ ok として引き継ぎ、LEGACY_NOTE を書く)。返す値は sources["review"] の集計。"""
    prev_by = {m.id: m for m in previous} if previous is not None else {}
    ms = list(measurements)
    marked: dict[str, int] = {}
    for e in entries:
        old = prev_by.get(e.id)
        reasons = review_reasons(old, e, ms)
        prev_review = copy.deepcopy(old.review) if old is not None else None
        if reasons:
            for r in reasons:
                marked[r] = marked.get(r, 0) + 1
            if prev_review is not None and prev_review.status != "ok":
                reasons = _ordered([*prev_review.reasons, *reasons])
                e.review = Review("pending", reasons, note=prev_review.note)
            else:
                e.review = Review("pending", _ordered(reasons))
        elif old is not None:
            e.review = prev_review if prev_review is not None else Review("ok", note=LEGACY_NOTE)
        # old が無く理由も無いことは無い(old が無ければ new)
    pending = sum(1 for e in entries if not e.reviewed)
    return {"ok": len(entries) - pending, "pending": pending, "marked": dict(sorted(marked.items()))}


# --- 外した項目(data/review_rejected.txt) ----------------------------------------------------------
#
# 1 行に 1 件: `<鍵>\t<YYYY-MM-DD>\t<理由>`。# から始まる行と空の行は読まない。鍵は ID そのものか、
# 名前を残さない外し方(--hide-name。既存の除外と同じく名前を残さない)のときは `sha256:<ID の sha256>`。


def id_hash(model_id: str) -> str:
    return _HASH_PREFIX + hashlib.sha256(model_id.encode("utf-8")).hexdigest()


def parse_rejected(text: str) -> dict[str, dict[str, str | None]]:
    """review_rejected.txt の中身 → {鍵: {date, note}}。"""
    out: dict[str, dict[str, str | None]] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        key = parts[0].strip()
        if key:
            out[key] = {
                "date": parts[1].strip() if len(parts) > 1 and parts[1].strip() else None,
                "note": parts[2].strip() if len(parts) > 2 and parts[2].strip() else None,
            }
    return out


def bundled_rejected() -> dict[str, dict[str, str | None]]:
    """同梱の review_rejected.txt(無ければ空)。"""
    from .db import bundled_text

    try:
        return parse_rejected(bundled_text(REJECTED_FILE))
    except (OSError, ValueError):
        return {}


def is_rejected(model_id: str, rejected: dict[str, Any]) -> bool:
    return model_id in rejected or id_hash(model_id) in rejected


def drop_rejected(
    entries: list[ModelEntry], rejected: dict[str, Any] | None = None
) -> tuple[list[ModelEntry], int]:
    """外した項目を除く。(残り, 除いた数)。"""
    rej = bundled_rejected() if rejected is None else rejected
    kept = [e for e in entries if not is_rejected(e.id, rej)]
    return kept, len(entries) - len(kept)


def finish(
    entries: list[ModelEntry],
    previous: Iterable[ModelEntry] | None,
    measurements: Iterable[Measurement] | None = None,
) -> tuple[list[ModelEntry], dict[str, Any]]:
    """update の最後に呼ぶ: 外した項目を除き、印を付ける。(残り, sources["review"] の集計)。"""
    if measurements is None:
        from .db import load_measurements

        measurements = load_measurements()
    kept, dropped = drop_rejected(entries)
    stats = apply_reviews(kept, previous, measurements)
    stats["rejected_skipped"] = dropped
    return kept, stats


# --- 一覧・確かめる・外す ------------------------------------------------------------------------


def review_list(db: Any, *, include_ok: bool = False) -> list[dict[str, Any]]:
    """確認待ち(include_ok なら全部)の項目: id・name・modality・content・source・license と review の中身。"""
    from .gate import _license_label

    out = []
    for m in db.models:
        if m.reviewed and not include_ok:
            continue
        r = m.review
        out.append(
            {
                "id": m.id,
                "name": m.name,
                "modality": m.modality,
                "content": m.content,
                "source": m.source,
                "license": _license_label(m.license, m.license_name),
                "license_url": m.license_url,
                "homepage": m.homepage,
                "status": r.status if r else "ok",
                "reasons": list(r.reasons) if r else [],
                "decided_by": r.decided_by if r else None,
                "decided_at": r.decided_at if r else None,
                "evidence": list(r.evidence) if r else [],
                "note": r.note if r else None,
            }
        )
    return out


def _today(today: str | None) -> str:
    return today or datetime.now(timezone.utc).date().isoformat()


def check_evidence(evidence: Iterable[str]) -> list[str]:
    ev = [e.strip() for e in evidence if e and e.strip()]
    if not ev:
        raise ReviewError("根拠(--evidence の URL。一次資料)が 1 つも無いため確かめられません")
    bad = [e for e in ev if not e.startswith(_EVIDENCE_PREFIX) or any(c.isspace() for c in e)]
    if bad:
        raise ReviewError(f"根拠は https:// で始まる URL にしてください: {', '.join(bad)}")
    return ev


def approve(
    db: Any,
    model_id: str,
    evidence: Iterable[str],
    *,
    note: str | None = None,
    by: str = "review",
    today: str | None = None,
) -> ModelEntry:
    """確認待ちの項目を確認済みにする(db を書き換える。保存はしない)。理由は残す(何を確かめたか)。"""
    ev = check_evidence(evidence)
    m = db.get(model_id)
    if m is None:
        raise ReviewError(f"一覧に無い ID です: {model_id}")
    if m.reviewed:
        raise ReviewError(f"確認待ちではありません: {model_id}")
    if not by or not by.strip():
        raise ReviewError("確かめた人(--by)が空です")
    assert m.review is not None
    m.review = Review(
        "ok", list(m.review.reasons), by.strip(), _today(today), ev, (note or "").strip() or None
    )
    return m


def reject(
    db: Any,
    model_id: str,
    note: str,
    *,
    hide_name: bool = False,
    today: str | None = None,
) -> str:
    """項目を一覧から外す(db を書き換える。保存はしない)。返す値は review_rejected.txt に足す 1 行。"""
    if not note or not note.strip():
        raise ReviewError("外す理由(--note)が空です")
    m = db.get(model_id)
    if m is None:
        raise ReviewError(f"一覧に無い ID です: {model_id}")
    db.models = [x for x in db.models if x.id != model_id]
    key = id_hash(model_id) if hide_name else model_id
    text = " ".join(note.split())
    return f"{key}\t{_today(today)}\t{text}\n"


def _load_file(db_path: str | Path):
    from .db import DatabaseError
    from .db import load as load_db

    try:
        return load_db(db_path)
    except DatabaseError as e:
        raise ReviewError(str(e)) from e


def _save(db: Any, path: Path) -> None:
    """一覧を書き戻す(db.json と同じ形。sources の review の数も数え直す)。"""
    pending = sum(1 for m in db.models if not m.reviewed)
    rv = db.sources.get("review") if isinstance(db.sources.get("review"), dict) else {}
    db.sources["review"] = {**rv, "ok": len(db.models) - pending, "pending": pending}
    db.save(path)


def approve_file(
    db_path: str | Path,
    model_id: str,
    evidence: Iterable[str],
    *,
    note: str | None = None,
    by: str = "review",
    today: str | None = None,
) -> ModelEntry:
    """`review approve`: db_path の一覧を読み、確認済みにして書き戻す。"""
    p = Path(db_path)
    db = _load_file(p)
    m = approve(db, model_id, evidence, note=note, by=by, today=today)
    _save(db, p)
    return m


def default_rejected_path(db_path: str | Path) -> Path:
    """外した項目のファイルの既定: 一覧と同じフォルダの review_rejected.txt
    (`--db src/genmodelfit/data/db.json` なら同梱のもの)。"""
    return Path(db_path).parent / REJECTED_FILE


def reject_file(
    db_path: str | Path,
    model_id: str,
    note: str,
    *,
    rejected_path: str | Path | None = None,
    hide_name: bool = False,
    today: str | None = None,
) -> dict[str, Any]:
    """`review reject`: db_path の一覧から外して書き戻し、review_rejected.txt に 1 行足す。"""
    p = Path(db_path)
    db = _load_file(p)
    line = reject(db, model_id, note, hide_name=hide_name, today=today)
    rp = Path(rejected_path) if rejected_path else default_rejected_path(p)
    old = rp.read_text(encoding="utf-8") if rp.is_file() else REJECTED_HEADER
    if old and not old.endswith("\n"):
        old += "\n"
    db.sources["counts"] = _counts(db)
    _save(db, p)
    rp.write_text(old + line, encoding="utf-8")
    key, date, text = line.rstrip("\n").split("\t", 2)
    return {"key": key, "date": date, "note": text, "rejected_file": str(rp), "db": str(p)}


def _counts(db: Any) -> dict[str, Any]:
    from .update import count_entries

    return count_entries(db.models)


REJECTED_HEADER = """\
# genmodelfit review reject で一覧から外した項目(#42)。update は作り直してもこれらを入れない。
# 1 行に 1 件: <ID か sha256:<ID の sha256>>\\t<YYYY-MM-DD>\\t<理由>
# 名前を残さない外し方(review reject --hide-name)は sha256: の鍵だけを残す
"""


def queue(db: Any, classification: dict[str, Any] | None = None) -> dict[str, Any]:
    """毎週の review_queue.json の中身。"""
    from ._version import __version__

    return {
        "genmodelfit": __version__,
        "generated_at": db.generated_at,
        "pending": review_list(db),
        "classification": classification,
    }


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1) + "\n"
