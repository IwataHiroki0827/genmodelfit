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
- 項目ごとの変化(field_changes・model_changes。#59): 共通の項目の全項目・ファイルの項目を entry_changes で
  比べる(分類と同じ関数)。`genmodelfit diff` の既定の表示はこれを「名前 (種類) — ID」ごとに出し、
  表に無い項目も「その他の変化」に名前を出す(黙って落とさない。tests/test_diff_ux.py)
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
# 項目の値が変わっても安全なもの。人気の数(downloads など)・更新日・テンプレートの使用回数・HEAD の結果・
# どの検索で見つかったかの記録(discovered_from。検索の順番で毎週変わりうる。おすすめの判定には使わない)
SAFE_ENTRY_FIELDS = frozenset({"downloads", "downloads_all_time", "likes", "updated", "discovered_from"})
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
    if not all(isinstance(f, dict) for f in [*fa, *fb]):
        return ["files_changed"]
    pairs, gone, new = _file_match(fa, fb)  # entry_changes と同じ組み方(#59)
    if gone or new or _order_changed(fa, pairs):
        return ["files_changed"]  # ファイルの増減・並びの変化
    out: list[str] = []
    for a, b in pairs:
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


def _file_key(f: Any) -> Any:
    return f.get("name") if isinstance(f, dict) else f


# 同じ名前のファイルが 1 つの項目に何度も出ることがある(URL・置き場所だけ違う)。組むときは確かな鍵から順に
# 組み、残りだけを (置き場所, 名前) の出てくる順で組む(順番だけで組むと、入れ替え・途中への追加で
# 「URL が変わった」「ライセンスが変わった」という無い変化を作る)。#59
_PAIR_TIERS = (
    ("name", "url", "sha256", "directory", "repo_id"),
    ("directory", "name", "sha256"),
    ("directory", "name", "url"),
    ("directory", "name"),
)


def _file_match(
    fa: list[Any], fb: list[Any]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    """ファイルの組(old, new)・消えたファイル・増えたファイル。並びは new(消えたものは old)の順。"""
    rest_a = [f for f in fa if isinstance(f, dict)]
    rest_b = [f for f in fb if isinstance(f, dict)]
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for keys in _PAIR_TIERS:
        if keys != _PAIR_TIERS[-1]:
            # sha256・url が無い(None)ファイルはこの段では組まない(None どうしで取り違えない)
            usable = [k for k in keys if k not in ("name", "directory", "repo_id")]
        else:
            usable = []
        pool: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for f in rest_a:
            if all(f.get(k) is not None for k in usable):
                pool.setdefault(tuple(f.get(k) for k in keys), []).append(f)
        left_b: list[dict[str, Any]] = []
        for f in rest_b:
            cand = (
                pool.get(tuple(f.get(k) for k in keys)) if all(f.get(k) is not None for k in usable) else None
            )
            if cand:
                pairs.append((cand.pop(0), f))
            else:
                left_b.append(f)
        used = {id(a) for a, _ in pairs}
        rest_a = [f for f in rest_a if id(f) not in used]
        rest_b = left_b
    order = {id(f): k for k, f in enumerate(fb)}
    pairs.sort(key=lambda p: order.get(id(p[1]), 0))
    return pairs, rest_a, rest_b


def _order_changed(fa: list[Any], pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    """組んだファイルの並び(new の順)が、old の並びと違うか。"""
    pos = {id(f): k for k, f in enumerate(fa)}
    idx = [pos[id(a)] for a, _ in pairs]
    return idx != sorted(idx)


def _file_pairs(fa: list[Any], fb: list[Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return _file_match(fa, fb)[0]


def entry_changes(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    """一覧の 1 項目(JSON の dict)の old → new で変わった項目の名前(安全の表のものも含む)。#59

    diff の表示(field_changes・model_changes)と分類(field_change_reasons)は、どちらもこの関数で
    変化を見つける(片方だけが変化を拾う・落とすことが無いように)。名前の形:
    - 一番上の項目はそのまま(例 languages・license・review)。id と計算で出す値(total_size_bytes)は見ない
    - files: ファイルの増減・並びの変化は "files"。組んだファイルどうしの項目の変化は "files.<項目>"
      (組み方は _file_match。同じ名前のファイルが何度も出ても取り違えない)
    - template: 両方が dict なら "template.<項目>"、それ以外は "template"
    """
    out: list[str] = []
    for k in sorted((set(a) | set(b)) - {"id"} - _DERIVED_FIELDS):
        va, vb = a.get(k), b.get(k)
        if k == "files":
            fa, fb = va or [], vb or []
            if fa == fb:
                continue
            pairs, gone, new = _file_match(fa, fb)
            found: list[str] = []
            if gone or new or not all(isinstance(f, dict) for f in [*fa, *fb]):
                found.append("files")
            for fa_, fb_ in pairs:
                found += [f"files.{t}" for t in sorted(set(fa_) | set(fb_)) if fa_.get(t) != fb_.get(t)]
            if not found and _order_changed(fa, pairs):
                found.append("files")  # 並びだけの変化(値が null の鍵が増えただけ、は変化にしない)
            out += found
        elif va == vb:
            continue
        elif k == "template" and isinstance(va, dict) and isinstance(vb, dict):
            out += [f"template.{t}" for t in sorted(set(va) | set(vb)) if va.get(t) != vb.get(t)]
        else:
            out.append(k)
    return list(dict.fromkeys(out))


def is_safe_path(path: str) -> bool:
    """entry_changes の名前が、安全の表(人気の数・更新日・HEAD の結果・テンプレートの使用回数)のものか。"""
    if path.startswith("files."):
        return path[6:] in SAFE_FILE_FIELDS
    if path.startswith("template."):
        return path[9:] in SAFE_TEMPLATE_FIELDS
    return path in SAFE_ENTRY_FIELDS


def field_change_reasons(
    a: dict[str, Any], b: dict[str, Any], allow: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST
) -> list[str]:
    """一覧の 1 項目(JSON の dict)の old → new の変化のうち、安全の表に無いものの理由。review は別に見る。

    変化は entry_changes で見つけ、ここでは理由に振り分けるだけ(diff の表示と同じ変化を見る)。"""
    out: list[str] = []
    files_done = False
    for p in entry_changes(a, b):
        if is_safe_path(p) or p in _LICENSE or p == "review":
            continue
        if p in ("tasks", "modality"):
            out.append("use_changed")
        elif p == "files" or p.startswith("files."):
            if not files_done:  # ファイルの理由はまとめて 1 回(大きさの向き・ライセンスの動きは値で見る)
                out += _file_reasons(a.get("files") or [], b.get("files") or [], allow)
                files_done = True
        elif p.startswith("template."):
            out.append(f"{OTHER_CHANGE}:{p}")
        elif p == "content" and b.get("content") == "uncensored":
            out.append("uncensored_new")
        else:
            out.append(f"{OTHER_CHANGE}:{p}")
    move = _license_move(a, b, allow)
    if move == "permissive":
        out.append("license_permissive")
    elif move == "other":
        out.append(f"{OTHER_CHANGE}:license")
    return list(dict.fromkeys(out))  # 同じ理由はまとめる(ファイルが何本変わっても files_changed は 1 つ)


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


def _raw_pair(
    old: Database, new: Database, raw: tuple[dict[str, Any], dict[str, Any]] | None
) -> tuple[dict[Any, dict[str, Any]], dict[Any, dict[str, Any]], list[str]]:
    """比べる 2 つの項目の dict(id → JSON の dict)と、変わった一番上の鍵(安全なものを除く)。
    classify と diff の表示(field_changes・model_changes)は、どちらもこれを比べる。"""
    if raw is None:
        return _raw_models(old), _raw_models(new), []
    ra = {m.get("id"): m for m in raw[0].get("models") or [] if isinstance(m, dict)}
    rb = {m.get("id"): m for m in raw[1].get("models") or [] if isinstance(m, dict)}
    top = sorted(k for k in (set(raw[0]) | set(raw[1])) - SAFE_TOP_KEYS if raw[0].get(k) != raw[1].get(k))
    return ra, rb, top


# model_changes の項目で、専用の行にする一番上の項目(ほかの安全でない項目は other に名前だけ出す)
_VALUE_FIELDS = ("name", "modality", "tasks", "runtime", "comfy_family", "content", "precision")
# ファイルの「どれか」を決める項目。変わったら old → new を出す(file_changed)
_FILE_IDENTITY_SHOWN = ("url", "sha256", "repo_id", "directory", "gated", "role")
# ライセンスの出どころ(どこで確かめたか)。ライセンスと同じ大事さで、値の前 → 後を出す
_ENTRY_LICENSE_SOURCE = ("license_url",)
_FILE_LICENSE_SOURCE = ("license_url", "license_from", "license_path", "repo_license")


def _file_source(f: dict[str, Any]) -> str | None:
    """ファイルの出どころ(同じ名前のファイルを見分ける): HF のリポジトリ、無ければ URL のホストとパス。"""
    if f.get("repo_id"):
        return str(f["repo_id"])
    url = f.get("url")
    if isinstance(url, str) and url:
        return url.split("://", 1)[-1].split("?", 1)[0]
    return None


def _file_brief(f: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": f.get("name"),
        "directory": f.get("directory"),
        "size_bytes": f.get("size_bytes"),
        "license": _license_label(f.get("license"), f.get("license_name")),
        "role": f.get("role"),
        "source": _file_source(f),
    }


def _no_lang(v: Any) -> bool:
    return v is None or v == []


def _model_items(
    a: dict[str, Any], b: dict[str, Any], paths: list[str], sizes: tuple[int | None, int | None]
) -> tuple[list[dict[str, Any]], list[str]]:
    """1 つのモデルの変化を、表示の行(items)と、名前だけ出す項目(other)に分ける。#59

    paths は entry_changes の結果。安全の表のものは見ない。どの項目も items か other のどちらかに入る
    (例外: 言語の「無い」と「空の並び」の違いは、利用者には同じなので行にしない)。
    sizes は一式の大きさ(ModelEntry.total_size_bytes)の old・new。"""
    items: list[dict[str, Any]] = []
    used: set[str] = set()
    todo = [p for p in paths if not is_safe_path(p)]
    for k in _VALUE_FIELDS:
        if k in todo:
            items.append({"kind": "value", "field": k, "old": a.get(k), "new": b.get(k)})
            used.add(k)
    if "license" in todo or "license_name" in todo:
        items.append(
            {
                "kind": "license",
                "old": _license_label(a.get("license"), a.get("license_name")),
                "new": _license_label(b.get("license"), b.get("license_name")),
            }
        )
        used |= {"license", "license_name"}
    if "languages" in todo:
        la, lb = a.get("languages"), b.get("languages")
        la, lb = (la if isinstance(la, dict) else {}), (lb if isinstance(lb, dict) else {})
        for key in sorted(set(la) | set(lb)):
            va, vb = la.get(key), lb.get(key)
            if va != vb and not (_no_lang(va) and _no_lang(vb)):
                items.append({"kind": "languages", "key": key, "old": va, "new": vb})
        used.add("languages")
    pairs, gone, new = _file_match(a.get("files") or [], b.get("files") or [])
    if "files" in todo:
        items += [{"kind": "file_added", **_file_brief(f)} for f in new]
        items += [{"kind": "file_removed", **_file_brief(f)} for f in gone]
        if not new and not gone:
            items.append({"kind": "files_order"})  # 並び・形だけの変化
        used.add("files")
    for g, f in pairs:
        for k in _FILE_IDENTITY_SHOWN:
            if f"files.{k}" in todo and g.get(k) != f.get(k):
                items.append(
                    {
                        "kind": "file_changed",
                        "name": f.get("name"),
                        "field": k,
                        "old": g.get(k),
                        "new": f.get(k),
                    }
                )
        la_, lb_ = (g.get("license"), g.get("license_name")), (f.get("license"), f.get("license_name"))
        if la_ != lb_ and ("files.license" in todo or "files.license_name" in todo):
            items.append(
                {
                    "kind": "file_license",
                    "name": f.get("name"),
                    "old": _license_label(*la_),
                    "new": _license_label(*lb_),
                }
            )
    for k in _ENTRY_LICENSE_SOURCE:
        if k in todo:
            items.append(
                {"kind": "license_source", "name": None, "field": k, "old": a.get(k), "new": b.get(k)}
            )
    for g, f in pairs:
        for k in _FILE_LICENSE_SOURCE:
            if f"files.{k}" in todo and g.get(k) != f.get(k):
                items.append(
                    {
                        "kind": "license_source",
                        "name": f.get("name"),
                        "field": k,
                        "old": g.get(k),
                        "new": f.get(k),
                    }
                )
    used |= {f"files.{k}" for k in (*_FILE_IDENTITY_SHOWN, *_FILE_LICENSE_SOURCE)}
    used |= {"files.license", "files.license_name", *_ENTRY_LICENSE_SOURCE}
    size_paths = {"files", "files.size_bytes", "template", "template.size_bytes"}
    if sizes[0] != sizes[1] and size_paths & set(todo):
        items.append({"kind": "size", "old": sizes[0], "new": sizes[1]})
        used |= {"files.size_bytes", "template.size_bytes"}
    if "review" in todo:
        ra_, rb_ = a.get("review"), b.get("review")
        da, db_ = (ra_ if isinstance(ra_, dict) else {}), (rb_ if isinstance(rb_, dict) else {})
        items.append(
            {
                "kind": "review",
                # null = 印が無い(確認済みとして扱う)
                "old": da.get("status") if isinstance(ra_, dict) else ra_,
                "new": db_.get("status") if isinstance(rb_, dict) else rb_,
                "reasons": [r for r in db_.get("reasons") or [] if isinstance(r, str)],
                # 変わった中身(status が同じでも根拠・理由だけ変わることがある)
                "fields": sorted(k for k in set(da) | set(db_) if da.get(k) != db_.get(k)),
            }
        )
        used.add("review")
    return items, [p for p in todo if p not in used]


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
    from .review import review_reasons  # 遅く読む(`python -m genmodelfit.review` の runpy の警告を出さない)

    measurements = new.measurements or old.measurements
    old_by = {m.id: m for m in old.models}
    new_by = {m.id: m for m in new.models}
    ra, rb, top = _raw_pair(old, new, raw)
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
    old_label: str | None = None,
    new_label: str | None = None,
) -> dict[str, Any]:
    """old → new の違い(docs/schema.md「diff」の形の dict)。ネットに触らず、何も表示しない。

    comfyui_version: 自動のおすすめを比べるときの ComfyUI の版。None なら新しい一覧が記録した最新の版
    (無ければ古い一覧の記録)。両方の一覧に同じ版を当てる(版の違いで可否が揺れないように)。
    raw: 2 つの一覧の JSON の dict(分類で知らない鍵も比べる。classify)。
    old_label / new_label: 比べた一覧の説明(表示用。例 "同梱の一覧"・ファイルのパス)。そのまま返す。
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

    # 項目ごとの変化(#59)。分類(classify)と同じ dict を、同じ entry_changes で比べる
    ra, rb, top_changes = _raw_pair(old, new, raw)
    auto_by = {c["id"]: c for c in auto_changes}
    field_changes: dict[str, list[str]] = {}
    model_changes: list[dict[str, Any]] = []
    for i in common:
        paths = entry_changes(ra.get(i, {}), rb.get(i, {}))
        if paths:
            field_changes[i] = paths
        items: list[dict[str, Any]] = []
        other: list[str] = []
        if any(not is_safe_path(x) for x in paths):
            a, b = old_by[i], new_by[i]
            items, other = _model_items(
                ra.get(i, {}), rb.get(i, {}), paths, (a.total_size_bytes, b.total_size_bytes)
            )
        if i in auto_by:  # 自動のおすすめの変化(HEAD が直っただけ・取れなくなっただけでも出す)
            c = auto_by[i]
            items.append({"kind": "auto", "old": c["old"], "new": c["new"], "held_by": c["held_by"]})
        if items or other:
            b = new_by[i]
            model_changes.append(
                {"id": i, "name": b.name, "modality": b.modality, "items": items, "other": other}
            )

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
        "old_label": old_label,
        "new_label": new_label,
        "field_changes": field_changes,
        "model_changes": model_changes,
        "top_changes": top_changes,
    }


# diff_db の summary の鍵(empty_diff も同じ鍵を返す)
SUMMARY_KEYS = (
    "models_old",
    "models_new",
    "added",
    "removed",
    "license_changed",
    "license_permissive_side",
    "auto_old",
    "auto_new",
    "auto_gained",
    "auto_lost",
    "size_changed",
    "precision_changed",
    "newly_uncensored",
    "head_failures",
    "head_failures_new",
    "head_unchecked",
    "unauthorized_added",
    "unauthorized_removed",
)


def empty_diff(note: str) -> dict[str, Any]:
    """比べる一覧が無いとき(`genmodelfit diff` で取ってきた一覧がまだ無い)の、diff_db と同じ鍵の dict。#59

    鍵の形は diff_db と同じにし、値は空(old / new / policy / classification は null、summary の数は 0)。
    理由は note。"""
    return {
        "old": None,
        "new": None,
        "policy": None,
        "summary": dict.fromkeys(SUMMARY_KEYS, 0) | {"head_unchecked": None},
        "added": [],
        "removed": [],
        "license_changes": [],
        "auto_changes": [],
        "size_changes": [],
        "precision_changes": [],
        "newly_uncensored": [],
        "head_failures": [],
        "unauthorized": {"added": [], "removed": []},
        "partial_counts": None,
        "stats": [],
        "classification": None,
        "old_label": None,
        "new_label": None,
        "field_changes": {},
        "model_changes": [],
        "top_changes": [],
        "note": note,
    }
