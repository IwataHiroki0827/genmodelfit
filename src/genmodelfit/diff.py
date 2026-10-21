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
# (HEAD の一時的な失敗が直った)。ほかの条件が外れて入ったものは判断が要る
SAFE_AUTO_GAIN = frozenset({"url_reachable"})
# 判断が要る理由(classification.judgement_needed[].reasons)。上の 4 つは review の pending の理由と同じ
JUDGEMENT_REASONS = (
    "new",
    "license_permissive",
    "need_decreased",
    "uncensored_new",
    "auto_gained",  # 自動のおすすめに入った(HEAD が直っただけのものを除く)
    "review_approved",  # 確認待ち → 確認済み(review approve)。根拠をオーナーが見る
    "review_pending",  # 確認済み → 確認待ちになった(上の理由で説明できない分。review の理由を付ける)
)


def classify(
    old: Database,
    new: Database,
    *,
    auto_changes: list[dict[str, Any]],
    license_allowlist: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST,
    counts: dict[str, int],
) -> dict[str, Any]:
    """変化を「安全な向き」と「判断が要る」に分ける(docs/schema.md「diff」の classification)。

    判断が要る(judgement_needed に ID と理由): 増えた項目(new。無検閲なら uncensored_new も)、
    共通の項目のうち review の pending の理由に当たるもの(license_permissive・need_decreased・uncensored_new。
    update と同じ review.review_reasons で決める)、自動のおすすめに入ったもの(前の保留が url_reachable
    だけのものを除く)、確認待ちから確認済みになったもの、確認済みから確認待ちになったもの。
    それ以外(消えた項目・厳しい側へのライセンスの変化・必要量が増える・変わらない大きさと精度の変化・
    自動のおすすめから外れた・HEAD の日時と失敗・401 の増減・sources の数・前から確認待ちの項目)は安全な向き。
    safe は judgement_needed が空のとき True。
    """
    measurements = new.measurements or old.measurements
    old_by = {m.id: m for m in old.models}
    new_by = {m.id: m for m in new.models}
    found: dict[str, list[str]] = {}

    def add(i: str, reasons: list[str]) -> None:
        if reasons:
            cur = found.setdefault(i, [])
            cur += [r for r in reasons if r not in cur]

    for i in sorted(new_by.keys() - old_by.keys()):
        add(i, review_reasons(None, new_by[i]))
    common = sorted(old_by.keys() & new_by.keys())
    for i in common:
        add(i, review_reasons(old_by[i], new_by[i], measurements, license_allowlist))
    auto_gained_safe = 0
    for c in auto_changes:
        if c["new"] and not c["old"] and c["id"] in old_by:
            if set(c["held_by_old"]) <= SAFE_AUTO_GAIN:
                auto_gained_safe += 1
            else:
                add(c["id"], ["auto_gained"])
    pending_carried = 0
    for i in common:
        a, b = old_by[i], new_by[i]
        if not a.reviewed and b.reviewed:
            add(i, ["review_approved"])
        elif a.reviewed and not b.reviewed and i not in found:
            add(i, ["review_pending", *(b.review.reasons if b.review else [])])
        elif not a.reviewed and not b.reviewed:
            pending_carried += 1
    order = {r: k for k, r in enumerate(JUDGEMENT_REASONS)}
    judgement = [
        {"id": i, "name": new_by[i].name, "reasons": sorted(rs, key=lambda r: order.get(r, len(order)))}
        for i, rs in sorted(found.items())
    ]
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
) -> dict[str, Any]:
    """old → new の違い(docs/schema.md「diff」の形の dict)。ネットに触らず、何も表示しない。

    comfyui_version: 自動のおすすめを比べるときの ComfyUI の版。None なら新しい一覧が記録した最新の版
    (無ければ古い一覧の記録)。両方の一覧に同じ版を当てる(版の違いで可否が揺れないように)。
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
