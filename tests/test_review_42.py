# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""確認の印(review)と、毎週の変化の分類(diff の classification)の試験(#42)。ネットワークには触らない。"""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import genmodelfit as gmf
from genmodelfit import cli, review, weekly
from genmodelfit.db import Database, DatabaseError, load, parse_db
from genmodelfit.diff import diff_db
from genmodelfit.gate import CHECK_IDS, GatePolicy, evaluate_offline
from genmodelfit.hf import TIER_PERMISSIVE, license_tier
from genmodelfit.models import ModelEntry, Review
from genmodelfit.table import CHECK_LABEL

NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)
URL = "https://huggingface.co/org/model/blob/main/LICENSE"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))
    for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"):
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(scope="module")
def bundled() -> Database:
    return load(use_cache=False)


def _auto(db: Database) -> ModelEntry:
    p = GatePolicy(max_head_age_days=None)
    return next(m for m in db.models if evaluate_offline(m, p, sources=db.sources).auto)


def _sized(db: Database) -> ModelEntry:
    """大きさの分かる ComfyUI のモデルで、実測の無いもの(実測は大きさを変えても動かない)。"""
    measured = {m.model_id for m in db.measurements}
    return next(
        m
        for m in db.models
        if m.id not in measured
        and m.runtime in (None, "comfyui")
        and m.files
        and all(f.size_bytes for f in m.files)
        and m.files_complete
    )


def _restricted(db: Database) -> ModelEntry:
    return next(m for m in db.models if license_tier(m.license, m.license_name) > TIER_PERMISSIVE and m.files)


def _scale(m: ModelEntry, k: float) -> ModelEntry:
    m = copy.deepcopy(m)
    for f in m.files:
        f.size_bytes = int(f.size_bytes * k)
    return m


def _loosen(m: ModelEntry) -> ModelEntry:
    m = copy.deepcopy(m)
    m.license, m.license_name = "apache-2.0", None
    for f in m.files:
        f.license, f.license_name = "apache-2.0", None
    return m


# --- 形・後方互換 ------------------------------------------------------------------------------------


def test_bundled_list_starts_all_ok(bundled):
    assert bundled.models
    for m in bundled.models:
        assert m.review is not None and m.review.status == "ok", m.id
        assert m.review.decided_by == review.BASELINE and m.review.decided_at == "2026-10-21"
    assert bundled.sources["review"]["pending"] == 0


def test_old_list_without_review_reads_as_ok(bundled):
    """review の無い古い一覧も読め、確認済みとして扱う(アプリの絞り込み `review is None or status == ok`)。"""
    data = bundled.to_dict()
    for m in data["models"]:
        m.pop("review")
    old = parse_db(json.loads(json.dumps(data)), "path")
    assert all(m.review is None and m.reviewed for m in old.models)
    kept = [e for e in old.models if e.review is None or e.review.status == "ok"]
    assert len(kept) == len(old.models)
    r = evaluate_offline(old.models[0], sources=old.sources)
    check = next(c for c in r.checks if c.id == "reviewed")
    assert check.passed and "印が無い" in check.detail


def test_review_round_trips_and_type_checked(bundled):
    data = json.loads(json.dumps(bundled.to_dict()))
    data["models"][0]["review"] = {"status": "pending", "reasons": ["new"], "evidence": []}
    db = parse_db(data, "path")
    r = db.models[0].review
    assert r == Review("pending", ["new"]) and not db.models[0].reviewed
    data["models"][0]["review"] = {"status": "ok", "reasons": "new"}
    with pytest.raises(DatabaseError, match="reasons"):
        parse_db(data, "path")
    data["models"][0]["review"] = {"status": "rejected-later", "reasons": []}  # 知らない値は ok でない
    assert not parse_db(data, "path").models[0].reviewed


# --- update が付ける印 --------------------------------------------------------------------------------


def test_new_entries_are_pending(bundled):
    prev = copy.deepcopy(bundled.models[1:])
    new = copy.deepcopy(bundled.models[:3])
    new[1].content = "uncensored"
    stats = review.apply_reviews(new, prev)
    assert new[0].review.status == "pending" and new[0].review.reasons == ["new"]
    assert new[1].review.reasons == ["uncensored_new"]  # 前からある sfw → uncensored
    assert new[2].review.status == "ok"
    assert stats["pending"] >= 1 and stats["marked"]["new"] == 1


def test_uncensored_change_and_new_uncensored(bundled):
    a = copy.deepcopy(bundled.models[0])
    b = copy.deepcopy(a)
    b.content = "uncensored"
    assert review.review_reasons(a, b) == ["uncensored_new"]
    assert review.review_reasons(None, b) == ["new", "uncensored_new"]


def test_license_toward_permissive_is_pending(bundled):
    a = _restricted(bundled)
    b = _loosen(a)
    assert review.review_reasons(a, b) == ["license_permissive"]
    assert review.review_reasons(b, a) == []  # 厳しい側は安全な向き


def test_need_decrease_is_pending_and_increase_is_not(bundled):
    a = _sized(bundled)
    assert review.review_reasons(a, _scale(a, 0.5)) == ["need_decreased"]
    assert review.review_reasons(a, _scale(a, 1.5)) == []
    assert review.review_reasons(a, _scale(a, 0.999)) == []  # 1% 以内の揺れは拾わない
    head_only = copy.deepcopy(a)
    for f in head_only.files:
        f.head_status, f.head_checked_at = 404, "2026-10-01T00:00:00Z"
    assert not review.need_decreased(a, head_only)  # HEAD だけの変化は見積もらない


def test_carry_over_and_union_of_reasons(bundled):
    a = _sized(bundled)
    prev = copy.deepcopy(a)
    prev.review = Review("pending", ["new"], note="先週から")
    cur = _scale(a, 0.5)
    review.apply_reviews([cur], [prev])
    assert cur.review.status == "pending" and cur.review.reasons == ["new", "need_decreased"]
    assert cur.review.note == "先週から"
    same = copy.deepcopy(a)
    review.apply_reviews([same], [prev])
    assert same.review == prev.review and same.review is not prev.review
    ok_prev = copy.deepcopy(a)
    ok_prev.review = Review("ok", ["new"], "owner", "2026-10-14", [URL])
    same = copy.deepcopy(a)
    review.apply_reviews([same], [ok_prev])
    assert same.review == ok_prev.review
    legacy = copy.deepcopy(a)
    legacy.review = None
    same = copy.deepcopy(a)
    review.apply_reviews([same], [legacy])
    assert same.review.status == "ok" and same.review.note == review.LEGACY_NOTE


def test_no_previous_list_means_all_new(bundled):
    es = copy.deepcopy(bundled.models[:2])
    review.apply_reviews(es, None)
    assert all(e.review.reasons[0] == "new" and not e.reviewed for e in es)


def test_rejected_ids_are_dropped(bundled):
    es = copy.deepcopy(bundled.models[:3])
    rej = review.parse_rejected(
        f"# 注釈\n{es[0].id}\t2026-10-21\tnon-commercial\n{review.id_hash(es[1].id)}\t2026-10-21\tsafety\n\n"
    )
    kept, n = review.drop_rejected(es, rej)
    assert [e.id for e in kept] == [es[2].id] and n == 2
    assert review.bundled_rejected() == {}  # 同梱の記録は注釈だけで始める


def test_partial_refresh_marks_new_manual_models(tmp_path, bundled, monkeypatch):
    """--heads-only などの一部のやり直しでも、増えた項目(手のモデルなど)は pending になる。"""
    from genmodelfit import update

    db = copy.deepcopy(bundled)
    before = copy.deepcopy(db.models[1:])
    update._finish_reviews(db, before)
    assert db.models[0].review.reasons == ["new"] and db.sources["review"]["pending"] == 1


# --- gate ------------------------------------------------------------------------------------------


def test_gate_holds_pending(bundled):
    m = copy.deepcopy(_auto(bundled))
    p = GatePolicy(max_head_age_days=None)
    assert evaluate_offline(m, p, sources=bundled.sources).auto
    m.review = Review("pending", ["new"])
    r = evaluate_offline(m, p, sources=bundled.sources)
    assert not r.auto and r.held_by == ["reviewed"]
    assert "new" in r.checks[-1].detail
    assert CHECK_IDS[-1] == "reviewed" and CHECK_LABEL["reviewed"] == "review"


# --- CLI と Python ---------------------------------------------------------------------------------


@pytest.fixture
def dbfile(tmp_path, bundled):
    db = copy.deepcopy(bundled)
    db.models[0].review = Review("pending", ["new"])
    db.models[1].review = Review("pending", ["license_permissive"])
    p = tmp_path / "data" / "db.json"
    db.save(p)
    return p, db.models[0].id, db.models[1].id, db.models[2].id


def test_review_list_json(dbfile, capsys):
    p, a, b, _ = dbfile
    assert cli.main(["review", "list", "--json", "--db", str(p)]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["command"] == "review" and d["pending"] == 2
    assert [m["id"] for m in d["models"]] == [a, b]
    assert d["models"][0]["reasons"] == ["new"] and d["models"][0]["status"] == "pending"
    assert cli.main(["review", "list", "--db", str(p)]) == 0
    assert "確認待ち 2 件" in capsys.readouterr().out


def test_approve_needs_https_evidence(dbfile, capsys):
    p, a, _, c = dbfile
    before = p.read_bytes()
    assert cli.main(["review", "approve", a, "--db", str(p)]) == 1
    assert "根拠" in capsys.readouterr().err
    assert cli.main(["review", "approve", a, "--db", str(p), "--evidence", "http://x.example/"]) == 1
    assert cli.main(["review", "approve", c, "--db", str(p), "--evidence", URL]) == 1  # 確認待ちでない
    capsys.readouterr()
    assert cli.main(["review", "approve", "nope", "--db", str(p), "--evidence", URL, "--json"]) == 1
    assert "一覧に無い" in json.loads(capsys.readouterr().out)["error"]
    assert p.read_bytes() == before
    with pytest.raises(SystemExit):  # --db は必須(書き換える先を明示する)
        cli.main(["review", "approve", a, "--evidence", URL])


def test_approve_writes_in_place(dbfile, capsys):
    p, a, _, _ = dbfile
    args = ["review", "approve", a, "--db", str(p), "--evidence", URL, "--evidence", URL + "2"]
    assert cli.main([*args, "--note", "原文を確かめた", "--by", "owner", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["review"]["status"] == "ok" and out["review"]["evidence"] == [URL, URL + "2"]
    m = load(p).get(a)
    assert m.review.status == "ok" and m.review.reasons == ["new"] and m.review.decided_by == "owner"
    assert m.review.note == "原文を確かめた" and len(m.review.decided_at) == 10
    assert load(p).sources["review"]["pending"] == 1


def test_reject_removes_and_records(dbfile, capsys):
    p, a, b, _ = dbfile
    with pytest.raises(SystemExit):  # 理由(--note)は必須
        cli.main(["review", "reject", a, "--db", str(p)])
    assert cli.main(["review", "reject", a, "--db", str(p), "--note", "非商用だけ"]) == 0
    rej = p.parent / review.REJECTED_FILE
    text = rej.read_text(encoding="utf-8")
    assert text.startswith("#") and f"\n{a}\t" in text and "非商用だけ" in text
    assert load(p).get(a) is None
    assert cli.main(["review", "reject", b, "--db", str(p), "--note", "safety", "--hide-name", "--json"]) == 0
    capsys.readouterr()
    text = rej.read_text(encoding="utf-8")
    assert b not in text and "sha256:" + hashlib.sha256(b.encode()).hexdigest() in text
    parsed = review.parse_rejected(text)
    assert review.is_rejected(a, parsed) and review.is_rejected(b, parsed)


def test_python_api_prints_nothing(dbfile, capfd):
    p, a, b, _ = dbfile
    db = load(p)
    assert [x["id"] for x in gmf.review_list(db)] == [a, b]
    m = gmf.review_approve(str(p), a, [URL], note="ok")
    assert m.reviewed
    with pytest.raises(ValueError, match="根拠"):
        gmf.review_approve(str(p), b, [])
    res = gmf.review_reject(str(p), b, "safety", hide_name=True)
    assert res["key"].startswith("sha256:")
    out, err = capfd.readouterr()
    assert out == "" and err == ""


# --- diff の分類 -------------------------------------------------------------------------------------


def _cl(old, new):
    return diff_db(old, new)["classification"]


def test_classification_safe_changes(bundled):
    old, new = copy.deepcopy(bundled), copy.deepcopy(bundled)
    assert _cl(old, new) == {
        "safe": True,
        "judgement_needed": [],
        "safe_changes": _cl(old, new)["safe_changes"],
    }
    new.models.pop(0)  # 消えた
    a = _sized(new)
    i = new.models.index(a)
    new.models[i] = _scale(a, 1.5)  # 必要量が増える
    r = _restricted(new)
    j = new.models.index(r)
    stricter = copy.deepcopy(r)
    stricter.license, stricter.license_name = "cc-by-nc-4.0", None  # 厳しい側
    new.models[j] = stricter
    for f in new.models[-1].files:  # HEAD の失敗
        f.head_status = 404
    new.sources = {**new.sources, "counts": {"models": 1}}  # sources の数
    c = _cl(old, new)
    assert c["safe"] is True and c["judgement_needed"] == [], c["judgement_needed"]
    sc = c["safe_changes"]
    assert sc["removed"] == 1 and sc["license_stricter"] >= 1 and sc["stats"] >= 1


def test_classification_judgement_needed(bundled):
    old, new = copy.deepcopy(bundled), copy.deepcopy(bundled)
    gone = old.models.pop(0)  # new には有る = 増えた
    a = _sized(old)
    new.models[new.models.index(new.get(a.id))] = _scale(a, 0.5)
    r = _restricted(old)
    new.models[new.models.index(new.get(r.id))] = _loosen(r)
    u = next(m for m in new.models if m.content == "sfw" and m.id not in (a.id, r.id, gone.id))
    u.content = "uncensored"
    c = _cl(old, new)
    got = {j["id"]: j["reasons"] for j in c["judgement_needed"]}
    assert c["safe"] is False
    assert got[gone.id][0] == "new"
    assert "need_decreased" in got[a.id] and "license_permissive" in got[r.id]
    assert got[u.id] == ["uncensored_new"]


def test_classification_review_transitions(bundled):
    old, new = copy.deepcopy(bundled), copy.deepcopy(bundled)
    old.models[0].review = Review("pending", ["new"])
    new.models[0].review = Review("ok", ["new"], "claude", "2026-10-21", [URL])
    old.models[1].review = Review("pending", ["new"])
    new.models[1].review = Review("pending", ["new"])  # 前から確認待ち(安全)
    new.models[2].review = Review("pending", ["need_decreased"])
    c = _cl(old, new)
    got = {j["id"]: j["reasons"] for j in c["judgement_needed"]}
    assert got == {
        new.models[0].id: ["review_approved"],
        new.models[2].id: ["need_decreased", "review_pending"],
    }
    assert c["safe_changes"]["pending_carried"] == 1


def test_auto_gained_by_head_recovery_is_safe(bundled):
    old, new = copy.deepcopy(bundled), copy.deepcopy(bundled)
    m = _auto(old)
    for f in m.files:
        f.head_status = 503
    c = _cl(old, new)
    assert c["safe"] is True and c["safe_changes"]["auto_gained_head_only"] == 1


# --- weekly ----------------------------------------------------------------------------------------


def test_weekly_writes_review_queue(tmp_path, monkeypatch, bundled):
    stamp = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def fake(fetch, out_path, *, allow_partial, progress, previous):
        db = copy.deepcopy(previous)
        for m in db.models:
            for f in m.files:
                if f.head_checked_at:
                    f.head_checked_at = stamp
        db.generated_at = stamp
        db.models[0].review = Review("pending", ["new"])
        db.models[0].name = "evil <!-- @someone [x](y) |"
        db.save(out_path)
        return db

    monkeypatch.setattr(weekly, "update_database", fake)
    monkeypatch.setattr(weekly, "inspect_database", lambda *a, **k: None)
    base = tmp_path / "base.json"
    bundled.save(base)
    r = weekly.run(tmp_path / "out", base=base, fetch=object(), now=NOW)
    assert r["ok"], r["problems"]
    q = json.loads((tmp_path / "out" / "review_queue.json").read_text(encoding="utf-8"))
    assert [p["id"] for p in q["pending"]] == [bundled.models[0].id]
    assert q["classification"]["safe"] is False
    assert r["classification"] == {"safe": False, "judgement_needed": 1}
    assert r["files"]["review_queue"].endswith("review_queue.json")
    body = (tmp_path / "out" / "pr_body.md").read_text(encoding="utf-8")
    assert "### 確認待ち(review: pending)の項目: 1 件" in body and "判断が要る変化" in body
    assert "<!--" not in body and "@someone" not in body  # md() で無害にする


def test_weekly_safe_week_says_so(tmp_path, monkeypatch, bundled):
    stamp = (NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z")

    def fake(fetch, out_path, *, allow_partial, progress, previous):
        db = copy.deepcopy(previous)
        for m in db.models:
            for f in m.files:
                if f.head_checked_at:
                    f.head_checked_at = stamp
        db.generated_at = stamp
        db.save(out_path)
        return db

    monkeypatch.setattr(weekly, "update_database", fake)
    monkeypatch.setattr(weekly, "inspect_database", lambda *a, **k: None)
    base = tmp_path / "base.json"
    bundled.save(base)
    r = weekly.run(tmp_path / "out", base=base, fetch=object(), now=NOW)
    assert r["ok"] and r["classification"]["safe"] is True
    body = (tmp_path / "out" / "pr_body.md").read_text(encoding="utf-8")
    assert "安全な向きの変化だけ" in body and "の項目: 0 件" in body


def test_update_from_fixtures_marks_against_previous(tmp_path):
    """本物の作り直し(記録の応答): 前の一覧が無ければ全部 new、同じ一覧を前にすれば確認待ちは増えない。"""
    from genmodelfit.fetch import DirFetcher
    from genmodelfit.update import build_database

    fix = Path(__file__).parent / "fixtures" / "update"
    first = build_database(DirFetcher(fix), head=False, carry_over=False)
    assert first.models and all(m.review.reasons[0] == "new" for m in first.models)
    assert first.sources["review"]["pending"] == len(first.models)
    for m in first.models:
        m.review = Review("ok", [], "test", "2026-10-21")
    again = build_database(DirFetcher(fix), head=False, previous=first)
    assert again.sources["review"]["pending"] == 0, [m.id for m in again.models if not m.reviewed]


# --- Checker の指摘(許すものの表・根拠のホスト・配る一覧・merge-check) -------------------------------


def _pair(bundled):
    return copy.deepcopy(bundled), copy.deepcopy(bundled)


def _reasons_for(old, new, mid):
    got = {j["id"]: j["reasons"] for j in _cl(old, new)["judgement_needed"]}
    return got.get(mid)


def test_url_and_sha_swap_to_another_repo_is_judgement(bundled):
    old, new = _pair(bundled)
    m = new.get(_auto(old).id)
    for f in m.files:  # 大きさは同じのまま、別のリポジトリのファイルに差し替える
        f.url = (f.url or "").replace("huggingface.co/", "huggingface.co/evil-") + "?x"
        f.sha256 = "f" * 64
        f.repo_id = "evil/" + (f.repo_id or "x")
    assert "files_changed" in _reasons_for(old, new, m.id)


@pytest.mark.parametrize(
    "change,reason",
    [
        (lambda m: m.tasks.append("zzz"), "use_changed"),
        (lambda m: setattr(m, "modality", "video" if m.modality != "video" else "image"), "use_changed"),
        (
            lambda m: m.languages.update({"prompt": [*m.languages.get("prompt", []), "ja"]}),
            "other_change:languages",
        ),
        (lambda m: m.description.update({"ja": "説明を変えた"}), "other_change:description"),
        (lambda m: setattr(m, "license_url", "https://example.com/other"), "other_change:license_url"),
        (lambda m: setattr(m, "name", m.name + " v2"), "other_change:name"),
        (lambda m: setattr(m.review, "status", "OK"), "review_pending"),
    ],
)
def test_field_changes_outside_the_allowlist_are_judgement(bundled, change, reason):
    old, new = _pair(bundled)
    m = next(x for x in new.models if "ja" not in x.languages.get("prompt", []))
    change(m)
    assert reason in (_reasons_for(old, new, m.id) or []), _cl(old, new)["judgement_needed"][:3]


def test_pure_head_and_popularity_changes_are_safe(bundled):
    old, new = _pair(bundled)
    for m in new.models:
        m.downloads = (m.downloads or 0) + 7
        m.likes = (m.likes or 0) + 1
        m.downloads_all_time = (m.downloads_all_time or 0) + 9
        m.updated = "2026-10-28T00:00:00Z"
        if m.template:
            m.template.usage = (m.template.usage or 0) + 3
        for f in m.files:
            f.head_checked_at = "2026-10-28T00:00:00Z"
    c = _cl(old, new)
    assert c["safe"] is True, c["judgement_needed"][:3]


def test_head_recovery_is_safe_only_with_the_same_files(bundled):
    old, new = _pair(bundled)
    mid = _auto(old).id
    for f in old.get(mid).files:
        f.head_status = 503
    assert _cl(old, new)["safe"] is True
    for f in new.get(mid).files:
        f.sha256 = "e" * 64
    assert "auto_gained" in _reasons_for(old, new, mid)


def test_unknown_keys_count_in_merge_check(tmp_path, bundled, capsys):
    base, head = tmp_path / "base.json", tmp_path / "head.json"
    bundled.save(base)
    data = json.loads(base.read_text(encoding="utf-8"))
    for f in data["models"][0]["files"]:
        f["head_checked_at"] = "2026-10-28T00:00:00Z"
    head.write_text(json.dumps(data), encoding="utf-8")
    assert weekly.main(["merge-check", "--base", str(base), "--head", str(head), "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["safe"] is True and r["reasons"] == []
    data["models"][1]["future_field"] = {"x": 1}  # この版が知らない鍵
    data["schema_extra"] = 1
    head.write_text(json.dumps(data), encoding="utf-8")
    assert (
        weekly.main(["merge-check", "--base", str(base), "--head", str(head), "--json"])
        == weekly.MERGE_NOT_SAFE
    )
    r = json.loads(capsys.readouterr().out)
    got = {j["id"]: j["reasons"] for j in r["reasons"]}
    assert r["safe"] is False and got[data["models"][1]["id"]] == ["other_change:future_field"]
    assert got["(database)"] == ["other_change:schema_extra"]
    assert weekly.main(["merge-check", "--base", str(base), "--head", str(tmp_path / "nope.json")]) == 1
    capsys.readouterr()


@pytest.mark.parametrize(
    "bad", ["https://", "https:///x", "https:// x", "http://example.com", "ftp://x.example/"]
)
def test_evidence_needs_a_host(bad):
    with pytest.raises(review.ReviewError):
        review.check_evidence([bad])
    assert review.check_evidence(["https://huggingface.co/x"]) == ["https://huggingface.co/x"]


def test_published_list_has_no_pending(tmp_path, bundled):
    """0.5.x は review を知らないので、配る一覧(data-latest)には確認待ちを入れない。元の一覧は変えない。"""
    db = copy.deepcopy(bundled)
    db.models[0].review = Review("pending", ["new"])
    out = review.publishable(db)
    assert db.models[0].review.status == "pending" and len(db.models) == len(bundled.models)
    assert len(out.models) == len(db.models) - 1 and all(m.reviewed for m in out.models)
    assert out.sources["review"]["withheld_pending"] == 1 and out.sources["counts"]["models"] == len(
        out.models
    )
    src, dst = tmp_path / "src.json", tmp_path / "dst.json"
    db.save(src)
    assert review.main(["publish", str(src), str(dst)]) == 0
    pub = json.loads(dst.read_text(encoding="utf-8"))
    assert db.models[0].id not in {m["id"] for m in pub["models"]}
    # 0.5.x の読み方(知らない鍵 review を捨てる)でも確認待ちが出てこない
    assert all(m.get("review", {}).get("status") == "ok" for m in pub["models"])
