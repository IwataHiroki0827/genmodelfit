"""同梱の一覧全体で、PC の段階ごとに崩れてはいけない性質。

CPU のみ、VRAM 4/6/8/12/16/24/32 GB × メモリ 16/32/64 GB で全モデルを判定する。
"""

from __future__ import annotations

import json
from itertools import pairwise

import pytest

from genmodelfit import cli, load_db
from genmodelfit.models import FIT_LEVELS, Gpu, SystemInfo
from genmodelfit.recommend import evaluate, recommend

VRAMS = [
    0,
    0.5,
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
]  # 0 = GPU なし(CPU のみ)。小さな GPU でも CPU より悪くならない
RAMS = [16, 32, 64]


def pc(vram: float, ram: float) -> SystemInfo:
    gpus = [Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")] if vram else []
    return SystemInfo(os="test", cpu="test", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=gpus)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def db():
    return load_db(use_cache=False)


@pytest.fixture(scope="module")
def levels(db):
    """(vram, ram) → {model id: 段階の番号(0 = perfect)}。"""
    out = {}
    for v in VRAMS:
        for r in RAMS:
            fits = evaluate(db, pc(v, r), content="all")
            assert len(fits) == len(db.models)
            for f in fits:
                e = f.estimate
                assert e.fit in FIT_LEVELS, (f.model.id, e.fit)
                if e.unknown_size:  # 「?」の印: 判定しない
                    assert e.fit == "too_tight" and e.run_mode == "none", f.model.id
                else:
                    assert e.run_mode in ("gpu", "offload", "cpu", "none"), f.model.id
                    assert e.vram_gb > 0 or e.ram_gb > 0, f.model.id
                if not v:
                    assert e.run_mode in ("cpu", "none"), f.model.id
            out[(v, r)] = {f.model.id: FIT_LEVELS.index(f.estimate.fit) for f in fits}
            scores[(v, r)] = {f.model.id: f.score for f in fits}
            for f in fits:
                assert 0 <= f.score <= 100 and all(0 <= x <= 100 for x in f.score_components.values())
    return out


scores: dict[tuple[float, float], dict[str, float]] = {}  # levels と一緒に作る(判定を 2 回しない)


def _worse(a: dict[str, int], b: dict[str, int]) -> list[str]:
    return sorted(mid for mid, la in a.items() if b[mid] > la)


def test_more_vram_never_worse(levels):
    bad = {
        (lo, hi, r): w
        for r in RAMS
        for lo, hi in pairwise(VRAMS)
        if (w := _worse(levels[(lo, r)], levels[(hi, r)]))
    }
    assert not bad, f"VRAM を増やすと段階が下がる: {bad}"


def test_more_ram_never_worse(levels):
    bad = {
        (v, lo, hi): w
        for v in VRAMS
        for lo, hi in pairwise(RAMS)
        if (w := _worse(levels[(v, lo)], levels[(v, hi)]))
    }
    assert not bad, f"メモリを増やすと段階が下がる: {bad}"


def test_score_never_drops_with_more_vram_or_ram(levels):
    """点数は VRAM・メモリを増やしても下がらない(docs/schema.md「Score」)。"""
    bad = []
    for r in RAMS:
        for lo, hi in pairwise(VRAMS):
            a, b = scores[(lo, r)], scores[(hi, r)]
            bad += [("vram", lo, hi, r, mid) for mid in a if b[mid] < a[mid]]
    for v in VRAMS:
        for lo, hi in pairwise(RAMS):
            a, b = scores[(v, lo)], scores[(v, hi)]
            bad += [("ram", v, lo, hi, mid) for mid in a if b[mid] < a[mid]]
    assert not bad, bad[:10]


def test_order_is_runs_then_auto_then_score(db, levels):
    for v in (0, 8, 24):
        fits = evaluate(db, pc(v, 32), content="all")
        keys = [(f.estimate.fit == "too_tight", not f.auto, -f.score, f.model.id) for f in fits]
        assert keys == sorted(keys)


@pytest.mark.parametrize("vram", VRAMS)
def test_recommend_never_too_tight_and_sfw_by_default(db, vram):
    for r in RAMS:
        fits = recommend(db, pc(vram, r), limit=None)
        assert all(f.estimate.fit != "too_tight" and not f.estimate.unknown_size for f in fits)
        assert all(f.model.content == "sfw" for f in fits)
        every = recommend(db, pc(vram, r), limit=None, content="all")
        assert all(f.estimate.fit != "too_tight" for f in every)
        assert {f.model.id for f in fits} <= {f.model.id for f in every}


@pytest.mark.parametrize("vram", [0, 8, 16, 32])
def test_cli_recommend_auto_only(capsys, vram):
    """--auto-only はすべて auto。既定(sfw)では無検閲を出さない(時計は conftest で固定)。"""
    hw = ["--vram", str(vram), "--ram", "32", "--no-cache", "--json", "-n", "500"]
    assert cli.main(["recommend", "--auto-only", *hw]) == 0
    models = json.loads(capsys.readouterr().out)["models"]
    assert all(m["auto"] and m["gate"]["auto"] for m in models)
    assert all(m["content"] == "sfw" and m["fit_level"] != "too_tight" for m in models)
    if vram >= 8:
        assert models, "8 GB 以上なら自動でおすすめに出せるものがある"
