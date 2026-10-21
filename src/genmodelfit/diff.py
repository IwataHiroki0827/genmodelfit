# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""2 つの一覧(db.json)の違いをまとめる(`genmodelfit diff`、`gmf.diff_db`)。#19

毎週の自動 PR(.github/workflows/weekly-data.yml、`python -m genmodelfit.weekly`)の本文と、人のレビューに使う。
ネットには触らず、何も表示しない(表示は cli.py と weekly.py)。返す値の形は docs/schema.md「diff」。

見るもの:
- 増えた・消えた項目(id・名前・種類・内容の印・出どころ・ライセンス)
- 項目のライセンスの変化。**寛容な側への変化**(段階が下がった、または既定の許可リストに入るようになった)は
  自動のおすすめに出る側の変化なので `permissive_side: true` を付け、要約でも別に数える
- 自動のおすすめの可否の変化。方針は両方に同じもの(ComfyUI の版は新しい一覧の記録、
  HEAD の古さは見ない)を当てる。
  HEAD の失敗(応答コード)は見るので、URL が取れなくなった項目はここにも出る
- 一式の大きさ(しきい値を超えたもの)・精度の変化、新しく無検閲になった項目
- HEAD が失敗した URL(200 / 302 / 307 以外の応答)。前の一覧では取れていた(または無かった)ものに `new: true`
- 匿名の API が 401 だったリポジトリの増減(sources.huggingface.unauthorized)
- sources の数の変化(件数・種類・ライセンスの段階・HEAD の応答コードの内訳・失敗の数など)
- 分類(classification。#42): 変化を「安全な向き」と「判断が要る」に分ける。毎週のスキル(genmodelfit-weekly)が
  自分でマージしてよい週か(safe)、根拠付きの PR をオーナーに出す週かを決めるのに使う。規則は classify と
  docs/schema.md「diff」
"""

from __future__ import annotations

import json
from typing import Any

from . import hf
from .db import Database
from .gate import (
    DEFAULT_LICENSE_ALLOWLIST,
    REACHABLE_STATUSES,
    GatePolicy,
    _license_label,
    evaluate_offline,
    license_allowed,
    recorded_comfyui_version,
)
from .models import ModelEntry
from .review import review_reasons

# 一式の大きさの変化を出すしきい値(両方を超えたとき): 比で 10%、量で 100 MiB
SIZE_REL_THRESHOLD = 0.10
SIZE_ABS_THRESHOLD = 100 * 2**20

# sources の数の変化で見ない鍵(毎回変わる・人が見ても意味が無い)
_SKIP_STAT_KEYS = frozenset({"checked_at", "url", "api", "generator", "failures", "errors", "warnings"})


def _entry_brief(m: ModelEntry) -> dict[str, Any]:
    return {
        "id": m.id,
        "name": m.name,
        "modality": m.modality,
        "content": m.content,
        "source": m.source,
        "license": _license_label(m.license, m.license_name),
    }


def _license_allowed(m: ModelEntry, allow: frozenset[str]) -> bool:
    """一式のライセンス(最も厳しいもの)が許可リストにあるか(gate の license と同じ照合)。"""
    return license_allowed(m.license, m.license_name, allow)


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    """sources の数・文字・真偽の値を「a.b.c」の鍵に平らにする(リストは数だけ)。"""
    out: dict[str, Any] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k in _SKIP_STAT_KEYS:
                if k == "warnings" and isinstance(v, list):
                    out[f"{prefix}{k}"] = len(v)
                continue
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key + "."))
            elif isinstance(v, list):
                out[key] = len(v)
            elif v is None or isinstance(v, (int, float, str, bool)):
                out[key] = v
    return out


def _head_failures(db: Database) -> dict[str, dict[str, Any]]:
    """HEAD の応答が 200 / 302 / 307 以外の URL → {status, ids}。"""
    out: dict[str, dict[str, Any]] = {}
    for m in db.models:
        for f in m.files:
            if f.url and f.head_status is not None and f.head_status not in REACHABLE_STATUSES:
                item = out.setdefault(f.url, {"status": f.head_status, "ids": []})
                if m.id not in item["ids"]:
                    item["ids"].append(m.id)
    return out


def _head_ok(db: Database) -> set[str]:
    return {f.url for m in db.models for f in m.files if f.url and f.head_status in REACHABLE_STATUSES}


def _unauthorized(db: Database) -> set[str]:
    hf_stats = db.sources.get("huggingface")
    items = hf_stats.get("unauthorized") if isinstance(hf_stats, dict) else None
    return {r for r in items or [] if isinstance(r, str)}


def _side(db: Database) -> dict[str, Any]:
    comfy = db.sources.get("comfyui")
    return {
        "generated_at": db.generated_at,
        "models": len(db.models),
        "license_policy": db.sources.get("license_policy"),
        "comfyui_latest_release": recorded_comfyui_version(db.sources),
        "partial": bool(db.sources.get("partial")),
        "comfyui_checked_at": comfy.get("checked_at") if isinstance(comfy, dict) else None,
    }


# 自動のおすすめに入った(auto_gained)理由が、前の一覧ではこの条件だけで保留だったなら安全な向き
# (HEAD の一時的な失敗が直った)。ただし、どのファイルの URL・sha256 も変わっていないときだけ
SAFE_AUTO_GAIN = frozenset({"url_reachable"})
# 判断が要る理由(classification.judgement_needed[].reasons)の並び。
# 最初の 4 つは review の pending の理由と同じ。
# このほかに other_change:<項目>(安全の表に無い項目の変化。知らない項目も含む)が後ろに付く
JUDGEMENT_REASONS = (
    "new",
    "license_permissive",
    "need_decreased",
    "uncensored_new",
    # ファイルの増減・URL(置き場所・パス)・sha256・置き場所のフォルダ・大きさが分かった・分からなくなった
    "files_changed",
    "use_changed",  # tasks・modality の変化
    "auto_gained",  # 自動のおすすめに入った(HEAD が直っただけのものを除く)
    "review_approved",  # 確認待ち → 確認済み(review approve)。根拠をオーナーが見る
    "review_pending",  # 確認済み → 確認待ち(review の理由を付ける)
)
OTHER_CHANGE = "other_change"

# 安全の表(#42 の Checker の指摘: 許すものだけを並べ、表に無い項目の変化はすべて判断が要る)。
# 項目の値が変わっても安全なもの。人気の数(downloads など)・更新日・テンプレートの使用回数・HEAD の結果
SAFE_ENTRY_FIELDS = frozenset({"downloads", "downloads_all_time", "likes", "updated"})
SAFE_TEMPLATE_FIELDS = frozenset({"usage"})
SAFE_FILE_FIELDS = frozenset({"head_status", "head_checked_at"})
# 一覧の項目から計算する値(比べない。元の項目の変化で見る)
_DERIVED_FIELDS = frozenset({"total_size_bytes"})
_FILE_IDENTITY = ("name", "url", "sha256", "repo_id", "directory")
_LICENSE = ("license", "license_name")
# 一覧の上の段で、変わってよい鍵(sources の数は安全。schema_version などほかの鍵の変化は判断が要る)
SAFE_TOP_KEYS = frozenset({"generated_at", "sources", "models"})


def _license_move(a: dict[str, Any], b: dict[str, Any], allow: frozenset[str]) -> str | None:
    """(license, license_name) の動き: None = 同じ、stricter = 厳しい側、permissive = 寛容な側、
    other = 同じ段階で名前だけ違う(判断が要る)。"""
    la, lb = (a.get("license"), a.get("license_name")), (b.get("license"), b.get("license_name"))
    if la == lb:
        return None
    ta, tb = hf.license_tier(*la), hf.license_tier(*lb)
    if tb < ta or (license_allowed(*lb, allow) and not license_allowed(*la, allow)):
        return "permissive"
    return "stricter" if tb > ta else "other"


def _file_reasons(fa: list[Any], fb: list[Any], allow: frozenset[str]) -> list[str]:
    na = [f.get("name") if isinstance(f, dict) else f for f in fa]
    nb = [f.get("name") if isinstance(f, dict) else f for f in fb]
    if na != nb or not all(isinstance(f, dict) for f in [*fa, *fb]):
        return ["files_changed"]  # ファイルの増減・並びの変化
    out: list[str] = []
    for a, b in zip(fa, fb, strict=True):
        for k in sorted((set(a) | set(b)) - SAFE_FILE_FIELDS):
            va, vb = a.get(k), b.get(k)
            if va == vb or k in _LICENSE:
                continue
            if k in _FILE_IDENTITY:
                out.append("files_changed")
            elif k == "size_bytes":
                if isinstance(va, int) and isinstance(vb, int) and vb < va:
                    out.append("need_decreased")
                elif not (isinstance(va, int) and isinstance(vb, int) and vb > va):
                    out.append("files_changed")  # 大きさが分かった・分からなくなった
            else:
                out.append(f"{OTHER_CHANGE}:files.{k}")
        move = _license_move(a, b, allow)
        if move == "permissive":
            out.append("license_permissive")
        elif move == "other":
            out.append(f"{OTHER_CHANGE}:files.license")
    return out


def field_change_reasons(
    a: dict[str, Any], b: dict[str, Any], allow: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST
) -> list[str]:
    """一覧の 1 項目(JSON の dict)の old → new の変化のうち、安全の表に無いものの理由。review は別に見る。"""
    out: list[str] = []
    for k in sorted((set(a) | set(b)) - {"id", "review"} - _DERIVED_FIELDS):
        va, vb = a.get(k), b.get(k)
        if va == vb or k in SAFE_ENTRY_FIELDS or k in _LICENSE:
            continue
        if k in ("tasks", "modality"):
            out.append("use_changed")
        elif k == "files":
            out += _file_reasons(va or [], vb or [], allow)
        elif k == "template" and isinstance(va, dict) and isinstance(vb, dict):
            out += [
                f"{OTHER_CHANGE}:template.{t}"
                for t in sorted((set(va) | set(vb)) - SAFE_TEMPLATE_FIELDS)
                if va.get(t) != vb.get(t)
            ]
        elif k == "content" and vb == "uncensored":
            out.append("uncensored_new")
        else:
            out.append(f"{OTHER_CHANGE}:{k}")
    move = _license_move(a, b, allow)
    if move == "permissive":
        out.append("license_permissive")
    elif move == "other":
        out.append(f"{OTHER_CHANGE}:license")
    return out


def _review_ok(r: Any) -> bool:
    """JSON の review が確認済みか(無い = 確認済み。status は小文字の ok だけ)。"""
    return r is None or (isinstance(r, dict) and r.get("status") == "ok")


def _same_files(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """どのファイルの名前・URL・sha256 も変わっていない。"""

    def key(d: dict[str, Any]) -> list[tuple[Any, ...]]:
        return [
            (f.get("name"), f.get("url"), f.get("sha256")) if isinstance(f, dict) else (f,)
            for f in d.get("files") or []
        ]

    return key(a) == key(b)


def _raw_models(db: Database) -> dict[str, dict[str, Any]]:
    return {m.id: json.loads(json.dumps(m.to_dict())) for m in db.models}


def classify(
    old: Database,
    new: Database,
    *,
    auto_changes: list[dict[str, Any]],
    license_allowlist: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST,
    counts: dict[str, int],
    raw: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """変化を「安全な向き」と「判断が要る」に分ける(docs/schema.md「diff」の classification)。

    許すものの表(allowlist)で決める。共通の項目は、すべての項目・ファイルの値を比べ、安全の表
    (SAFE_ENTRY_FIELDS・SAFE_TEMPLATE_FIELDS・SAFE_FILE_FIELDS、ファイルの大きさが増える、ライセンスが
    厳しい側、確認待ちのまま変わらない review)に無い変化はすべて判断が要る(知らない項目も)。
    増えた項目は new、自動のおすすめに入った項目は auto_gained(前の保留が url_reachable だけで、
    ファイルの URL・sha256 が同じものを除く)。消えた項目・自動のおすすめから外れた項目・sources の数は安全。

    raw: (old, new) の一覧の JSON の dict(`merge-check` がファイルから読んだもの。知らない鍵も比べる)。
    None なら Database から作る(知らない鍵は読むときに捨てられている)。
    """
    measurements = new.measurements or old.measurements
    old_by = {m.id: m for m in old.models}
    new_by = {m.id: m for m in new.models}
    if raw is None:
        ra, rb = _raw_models(old), _raw_models(new)
        top: list[str] = []
    else:
        ra = {m.get("id"): m for m in raw[0].get("models") or [] if isinstance(m, dict)}
        rb = {m.get("id"): m for m in raw[1].get("models") or [] if isinstance(m, dict)}
        top = sorted(k for k in (set(raw[0]) | set(raw[1])) - SAFE_TOP_KEYS if raw[0].get(k) != raw[1].get(k))
    found: dict[str, list[str]] = {}

    def add(i: str, reasons: list[str]) -> None:
        if reasons:
            cur = found.setdefault(i, [])
            cur += [r for r in reasons if r not in cur]

    for i in sorted(new_by.keys() - old_by.keys()):
        add(i, review_reasons(None, new_by[i]))
    common = sorted(old_by.keys() & new_by.keys())
    pending_carried = 0
    for i in common:
        a, b = ra.get(i, {}), rb.get(i, {})
        add(i, field_change_reasons(a, b, license_allowlist))
        add(i, review_reasons(old_by[i], new_by[i], measurements, license_allowlist))
        rva, rvb = a.get("review"), b.get("review")
        if rva == rvb:
            pending_carried += 0 if _review_ok(rvb) else 1
        elif not _review_ok(rva) and _review_ok(rvb):
            add(i, ["review_approved"])
        elif _review_ok(rva) and not _review_ok(rvb):
            reasons = rvb.get("reasons") if isinstance(rvb, dict) else None
            add(i, ["review_pending", *(r for r in reasons or [] if isinstance(r, str))])
        else:
            add(i, [f"{OTHER_CHANGE}:review"])  # 確認済みのまま根拠が変わった・確認待ちの理由が変わった など
    auto_gained_safe = 0
    for c in auto_changes:
        if c["new"] and not c["old"] and c["id"] in old_by:
            i = c["id"]
            if set(c["held_by_old"]) <= SAFE_AUTO_GAIN and _same_files(ra.get(i, {}), rb.get(i, {})):
                auto_gained_safe += 1
            else:
                add(i, ["auto_gained"])
    order = {r: k for k, r in enumerate(JUDGEMENT_REASONS)}
    judgement = [
        {
            "id": i,
            "name": new_by[i].name,
            "reasons": sorted(rs, key=lambda r: (order.get(r, len(order)), r)),
        }
        for i, rs in sorted(found.items())
    ]
    judgement += [{"id": "(database)", "name": None, "reasons": [f"{OTHER_CHANGE}:{k}"]} for k in top]
    safe_changes = {
        **counts,
        "auto_gained_head_only": auto_gained_safe,
        "auto_lost": sum(1 for c in auto_changes if c["old"] and not c["new"]),
        "pending_carried": pending_carried,
    }
    return {"safe": not judgement, "judgement_needed": judgement, "safe_changes": safe_changes}


def diff_db(
    old: Database,
    new: Database,
    *,
    comfyui_version: str | None = None,
    license_allowlist: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST,
    size_rel_threshold: float = SIZE_REL_THRESHOLD,
    size_abs_threshold: int = SIZE_ABS_THRESHOLD,
    raw: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """old → new の違い(docs/schema.md「diff」の形の dict)。ネットに触らず、何も表示しない。

    comfyui_version: 自動のおすすめを比べるときの ComfyUI の版。None なら新しい一覧が記録した最新の版
    (無ければ古い一覧の記録)。両方の一覧に同じ版を当てる(版の違いで可否が揺れないように)。
    raw: 2 つの一覧の JSON の dict(分類で知らない鍵も比べる。classify)。
    """
    target = comfyui_version or recorded_comfyui_version(new.sources) or recorded_comfyui_version(old.sources)
    policy = GatePolicy(license_allowlist=license_allowlist, comfyui_version=target, max_head_age_days=None)
    old_by = {m.id: m for m in old.models}
    new_by = {m.id: m for m in new.models}

    added = [_entry_brief(new_by[i]) for i in sorted(new_by.keys() - old_by.keys())]
    removed = [_entry_brief(old_by[i]) for i in sorted(old_by.keys() - new_by.keys())]
    common = sorted(old_by.keys() & new_by.keys())

    license_changes: list[dict[str, Any]] = []
    size_changes: list[dict[str, Any]] = []
    precision_changes: list[dict[str, Any]] = []
    newly_uncensored: list[dict[str, Any]] = []
    for i in common:
        a, b = old_by[i], new_by[i]
        if (a.license, a.license_name) != (b.license, b.license_name):
            ta, tb = hf.license_tier(a.license, a.license_name), hf.license_tier(b.license, b.license_name)
            aa, ab = _license_allowed(a, license_allowlist), _license_allowed(b, license_allowlist)
            license_changes.append(
                {
                    "id": i,
                    "name": b.name,
                    "old": _license_label(a.license, a.license_name),
                    "new": _license_label(b.license, b.license_name),
                    "old_tier": ta,
                    "new_tier": tb,
                    "allowed_old": aa,
                    "allowed_new": ab,
                    "permissive_side": tb < ta or (ab and not aa),
                }
            )
        sa, sb = a.total_size_bytes, b.total_size_bytes
        if sa != sb:
            if not sa or not sb:
                big = True
            else:
                big = abs(sb - sa) >= size_abs_threshold and abs(sb - sa) / sa >= size_rel_threshold
            if big:
                size_changes.append(
                    {
                        "id": i,
                        "old_bytes": sa,
                        "new_bytes": sb,
                        "ratio": round(sb / sa, 3) if sa and sb else None,
                    }
                )
        if a.precision != b.precision:
            precision_changes.append(
                {
                    "id": i,
                    "old": a.precision,
                    "new": b.precision,
                    "old_source": a.precision_source,
                    "new_source": b.precision_source,
                }
            )
        if a.content != "uncensored" and b.content == "uncensored":
            newly_uncensored.append({**_entry_brief(b), "evidence": list(b.content_evidence), "added": False})
    for item in added:
        if item["content"] == "uncensored":
            newly_uncensored.append(
                {**item, "evidence": list(new_by[item["id"]].content_evidence), "added": True}
            )

    # 自動のおすすめ(同じ方針・HEAD の古さは見ない)
    def _auto(db: Database, by: dict[str, ModelEntry]) -> dict[str, Any]:
        return {i: evaluate_offline(m, policy, sources=db.sources) for i, m in by.items()}

    ga, gb = _auto(old, old_by), _auto(new, new_by)
    auto_changes: list[dict[str, Any]] = []
    for i in sorted(old_by.keys() | new_by.keys()):
        oa = ga[i].auto if i in ga else None
        na = gb[i].auto if i in gb else None
        if bool(oa) != bool(na):
            auto_changes.append(
                {
                    "id": i,
                    "name": (new_by.get(i) or old_by[i]).name,
                    "old": oa,
                    "new": na,
                    "held_by": gb[i].held_by if i in gb else [],
                    "held_by_old": ga[i].held_by if i in ga else [],
                }
            )
    auto_old = sum(1 for r in ga.values() if r.auto)
    auto_new = sum(1 for r in gb.values() if r.auto)

    fail_old, fail_new, ok_old = _head_failures(old), _head_failures(new), _head_ok(old)
    head_failures = [
        {
            "url": url,
            "status": v["status"],
            "ids": sorted(v["ids"]),
            # 前の一覧で取れていた URL・前の一覧に無かった URL が今回失敗した
            "new": url not in fail_old or url in ok_old,
        }
        for url, v in sorted(fail_new.items())
    ]
    head_new = new.sources.get("head_check") if isinstance(new.sources.get("head_check"), dict) else {}

    ua, ub = _unauthorized(old), _unauthorized(new)

    fa, fb = _flatten(old.sources), _flatten(new.sources)
    stats = [
        {"key": k, "old": fa.get(k), "new": fb.get(k)}
        for k in sorted(fa.keys() | fb.keys())
        if fa.get(k) != fb.get(k)
    ]

    gained = sum(1 for c in auto_changes if c["new"])
    classification = classify(
        old,
        new,
        auto_changes=auto_changes,
        license_allowlist=license_allowlist,
        raw=raw,
        counts={
            "removed": len(removed),
            "license_stricter": sum(1 for c in license_changes if not c["permissive_side"]),
            "size_changed": len(size_changes),
            "precision_changed": len(precision_changes),
            "head_failures_new": sum(1 for h in head_failures if h["new"]),
            "unauthorized_added": len(ub - ua),
            "unauthorized_removed": len(ua - ub),
            "stats": len(stats),
        },
    )
    summary = {
        "models_old": len(old.models),
        "models_new": len(new.models),
        "added": len(added),
        "removed": len(removed),
        "license_changed": len(license_changes),
        "license_permissive_side": sum(1 for c in license_changes if c["permissive_side"]),
        "auto_old": auto_old,
        "auto_new": auto_new,
        "auto_gained": gained,
        "auto_lost": len(auto_changes) - gained,
        "size_changed": len(size_changes),
        "precision_changed": len(precision_changes),
        "newly_uncensored": len(newly_uncensored),
        "head_failures": len(head_failures),
        "head_failures_new": sum(1 for h in head_failures if h["new"]),
        "head_unchecked": head_new.get("unchecked") if isinstance(head_new, dict) else None,
        "unauthorized_added": len(ub - ua),
        "unauthorized_removed": len(ua - ub),
    }
    return {
        "old": _side(old),
        "new": _side(new),
        "policy": policy.to_dict(),
        "summary": summary,
        "added": added,
        "removed": removed,
        "license_changes": license_changes,
        "auto_changes": auto_changes,
        "size_changes": size_changes,
        "precision_changes": precision_changes,
        "newly_uncensored": newly_uncensored,
        "head_failures": head_failures,
        "unauthorized": {"added": sorted(ub - ua), "removed": sorted(ua - ub)},
        "partial_counts": new.sources.get("partial_counts"),
        "stats": stats,
        "classification": classification,
    }
