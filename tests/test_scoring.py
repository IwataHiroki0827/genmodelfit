"""判定の公平さ・並び・Score の試験(#95。決まりは docs/scoring.md)。"""

from __future__ import annotations

import pytest
from test_contrib import _sub

from genmodelfit import contrib, gate
from genmodelfit.db import load
from genmodelfit.estimate import (
    DEFAULT_SETTINGS,
    _prepare,
    estimate,
    gpu_precision,
    load_coefficients,
    measured_level,
    no_size_input,
)
from genmodelfit.gate import GatePolicy
from genmodelfit.hardware import classify_gpu
from genmodelfit.models import (
    Estimate,
    Gpu,
    Measurement,
    ModelEntry,
    ModelFile,
    Settings,
    SystemInfo,
    Template,
)
from genmodelfit.popularity import compute, percentile_ranks, source_count, spearman, vram_bias
from genmodelfit.recommend import (
    Fit,
    list_key,
    memory_fit,
    score_components,
    select,
    usage_ratio,
)


@pytest.fixture(scope="module")
def db():
    return load(use_cache=False)


def nvidia(name: str, vram: float, ram: float = 32.0) -> SystemInfo:
    g = Gpu(name=name, vendor="nvidia", vram_total_gb=vram, driver="610.88", backend="cuda")
    classify_gpu(g, windows_driver=True, comfy_windows=True)
    return SystemInfo(os="Windows", cpu="", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[g])


def plain(vram: float, ram: float = 32.0) -> SystemInfo:
    g = Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")
    return SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[g])


RTX_5080 = "NVIDIA GeForce RTX 5080"
RTX_3070 = "NVIDIA GeForce RTX 3070"


# --- §4.2 Score = 収まり ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("r", "want"),
    [(0.50, 100.0), (0.60, 100.0), (0.70, 100.0), (0.80, 88.2), (0.85, 75.5), (0.90, 60.7), (0.98, 37.5)],
)
def test_memory_fit_fixed_values(r, want):
    assert round(memory_fit(r, "good"), 1) == want


def test_memory_fit_shape_floor_and_not_runnable():
    assert abs(memory_fit(0.69, "good") - memory_fit(0.71, "good")) < 1  # 崖が無い(llmfit PR #100)
    for r in (0.99, 1.2):  # 実測で段階を決めた行・ページファイルで動く CPU の行
        assert memory_fit(r, "marginal") == 37.5
    for level in ("too_tight", "unknown"):
        assert memory_fit(0.3, level) == 0.0
    assert memory_fit(None, "good") == 0.0


def _est(**kw) -> Estimate:
    base = {
        "vram_gb": 6.0,
        "vram_offload_gb": 3.0,
        "ram_gb": 8.0,
        "disk_gb": 5.0,
        "basis": "estimated",
        "fit": "good",
        "run_mode": "gpu",
        "vram_sequential_gb": 6.0,
    }
    base.update(kw)
    return Estimate(**base)


@pytest.mark.parametrize(
    ("kw", "label", "want"),
    [
        ({"fit": "perfect"}, "GPU", 0.6),  # GPU: 全部載せ ÷ VRAM
        ({"vram_gb": 12.0, "vram_sequential_gb": 7.0}, "GPU~", 0.7),  # 順に載せ替えの最大 ÷ VRAM
        (  # GPU+RAM(VRAM がきつい): 逃がしたときの VRAM ÷ VRAM
            {"vram_gb": 20.0, "vram_sequential_gb": 15.0, "vram_offload_gb": 9.0, "run_mode": "offload"},
            "GPU+RAM",
            0.9,
        ),
        (  # GPU+RAM(メモリがきつい): 要るメモリ ÷ メインメモリ
            {
                "vram_gb": 20.0,
                "vram_sequential_gb": 15.0,
                "vram_offload_gb": 5.0,
                "run_mode": "offload",
                "ram_gb": 28.0,
            },
            "GPU+RAM",
            0.875,
        ),
        ({"run_mode": "cpu", "fit": "marginal", "ram_gb": 8.0}, "CPU", 0.25),  # 要るメモリ ÷ メインメモリ
        (  # 実測で段階を決めた行: 段階に使った量 ÷ VRAM
            {"fit_from": "measurement", "measured_peak_vram_gb": 8.5, "measured_level_vram_gb": 9.0},
            "GPU",
            0.9,
        ),
        ({"fit": "marginal", "ram_gb": 40.0}, "GPU", 1.25),  # メモリ不足の行: max(VRAM の割合, メモリの割合)
    ],
)
def test_usage_ratio_per_run_mode(kw, label, want):
    from genmodelfit.recommend import mode_label

    system = plain(10.0, ram=32.0)
    e = _est(**kw)
    assert mode_label(e, system) == label
    assert usage_ratio(e, system) == pytest.approx(want)
    comps = score_components(e, system)
    assert comps == {"memory_fit": round(memory_fit(want, e.fit), 1)}


def test_score_can_drop_with_more_vram(db):
    """約束は段階だけ: VRAM が増えて代表がより重い精度・動かし方に替わると、使用率が上がり Score は
    下がることがある(Z-Image Turbo: 12 GB は int8 で 100、16 GB は bf16 の GPU~ で 82.6。llmfit も同じ。
    docs/scoring.md「Score」)。段階は下がらない。"""
    from genmodelfit.models import fit_rank

    rows = {}
    for v in (12, 16):
        sel = select(db, plain(v, 32), modality="image")
        rows[v] = next(f for f in sel.fits if f.group_key == "image:sfw:image turbo z")
    assert rows[12].model.id == "image_z_image_turbo_int8" and rows[12].score == 100.0
    assert rows[16].model.id == "image_z_image_turbo" and rows[16].score == pytest.approx(82.6, abs=0.1)
    assert fit_rank(rows[16].estimate.fit) <= fit_rank(rows[12].estimate.fit)


# --- §4.3 人気 -------------------------------------------------------------------------------------


def test_popularity_percentile_example():
    ranks = percentile_ranks({"a": 10, "b": 20, "c": 20, "d": 40, "e": 0})
    assert ranks == {"a": 30.0, "b": 60.0, "c": 60.0, "d": 90.0, "e": 0.0}
    small = percentile_ranks({"a": 10, "b": 20, "c": 30, "d": 0})  # 仲間 4 件: 数のあるものは 50
    assert small == {"a": 50.0, "b": 50.0, "c": 50.0, "d": 0.0}


def _tmpl(mid: str, usage: int | None, modality: str = "image") -> ModelEntry:
    return ModelEntry(id=mid, name=mid, modality=modality, template=Template(name=mid, usage=usage))


def _hf(mid: str, downloads: int | None, modality: str = "image", source: str = "huggingface") -> ModelEntry:
    files = [ModelFile(name="m.safetensors", repo_id=mid.split(":")[-1])]
    return ModelEntry(
        id=mid, name=mid, modality=modality, source=source, downloads_all_time=downloads, files=files
    )


def test_popularity_sources_are_not_mixed():
    t = _tmpl("t", 5)
    h = _hf("hf:a/b", 10**6)
    assert source_count([t, h]) == ("template", 5)  # 仲間に両方あればテンプレートの数
    manual = _hf("manual:x", None, source="manual")  # 手で足したモデルも HF のリポジトリがあれば HF の仲間
    assert source_count([manual]) == ("hf", 0)
    assert source_count([ModelEntry(id="m", name="m", modality="image", source="manual")]) == (None, 0)
    models = [_tmpl(f"t{i}", u) for i, u in enumerate((10, 20, 20, 40, 0))]
    models += [_hf(f"hf:o/h{i}", d) for i, d in enumerate((100, 200, 300, 400, 500))]
    models.append(manual)
    key_of = {m.id: m.id for m in models}
    ranks = compute(models, key_of)
    assert [ranks[f"t{i}"] for i in range(5)] == [30.0, 60.0, 60.0, 90.0, 0.0]
    # HF の仲間は 6 件(手のモデルは数 0 として入る)
    assert ranks["manual:x"] == 0.0 and ranks["hf:o/h0"] == pytest.approx(100 * 1.5 / 6)
    # 種類が違えば別の仲間
    video = [_tmpl("v1", 1, "video")]
    assert compute(video, {"v1": "v1"}) == {"v1": 50.0}


def test_popularity_is_per_model_and_in_json(db):
    sel = select(db, plain(16), modality="image", include_unknown_license=True)
    for f in sel.fits:
        d = f.to_dict()
        assert 0 <= d["popularity"] <= 100 and "popularity" not in d["score_components"]
    flows = select(db, plain(16), modality="image", include_unknown_license=True, group=False).fits
    by_key: dict[str, set[float]] = {}
    for f in flows:
        by_key.setdefault(f.group_key, set()).add(f.popularity)
    assert all(len(v) == 1 for v in by_key.values())  # --all-workflows の行もモデルの値


def test_spearman_and_bias_check(db):
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1], [1, 2]) is None
    rows = {r["modality"]: r for r in vram_bias(db)}
    assert rows["image"]["checked"] and rows["image"]["models"] >= 20
    assert all(not r["checked"] for r in rows.values() if r["models"] < 20)
    assert all(
        r["flagged"] == (r["checked"] and r["rho"] < -0.3) for r in rows.values() if r["rho"] is not None
    )


# --- §5 並び --------------------------------------------------------------------------------------


def _fit(
    mid: str, *, fit="good", mode="gpu", seq=None, full=6.0, pop=0.0, score=0.0, community=False, auto=False
):
    username = "someone" if community else "ComfyUI"
    m = ModelEntry(id=mid, name=mid, modality="image", template=Template(name=mid, username=username))
    off = 3.0 if mode != "offload" else 9.5
    e = _est(fit=fit, run_mode=mode, vram_gb=full, vram_sequential_gb=seq or full, vram_offload_gb=off)
    return Fit(m, e, score, {"auto": auto, "checks": []}, {}, plain(10.0), popularity=pop)


def _order(fits, **kw):
    return [f.model.id for f in sorted(fits, key=lambda f: list_key(f, **kw))]


def test_recommend_order_keys():
    fits = [
        _fit("tight", fit="too_tight", mode="none"),
        _fit("unknown", fit="unknown", mode="unknown"),
        _fit("marg-cpu", fit="marginal", mode="cpu", pop=100),
        _fit("marg-offload", fit="marginal", mode="offload", full=20.0, seq=15.0, pop=90),
        _fit("marg-gpu", fit="marginal", full=9.5, pop=0),
        _fit("good-seq", fit="good", full=12.0, seq=7.0, pop=50),  # 逃がさずに載る中では GPU~ も同じ段階
        _fit("good-gpu", fit="good", full=7.0, pop=40),
        _fit("good-comm", fit="good", full=7.0, pop=99, community=True),
        _fit("good-b", fit="good", full=7.0, pop=40, score=90),
        _fit("good-a", fit="good", full=7.0, pop=40, score=90),
    ]
    want = [
        "good-seq",  # 人気 50
        "good-a",  # 人気 40・Score 90・ID
        "good-b",
        "good-gpu",  # 人気 40・Score 0
        "good-comm",  # 利用者のテンプレートは同じ段階の公式の後(人気より先の鍵)
        "marg-gpu",  # Marginal の中は動かし方の速い順(人気より先の鍵)
        "marg-offload",
        "marg-cpu",
        "unknown",  # 動く → 判定できない → 動かない
        "tight",
    ]
    assert _order(fits, auto_first=True, verified_first=False) == want
    # --json・ライブラリは鍵 1 の次に自動のおすすめ
    fits.append(_fit("auto-marg", fit="marginal", mode="cpu", auto=True))
    assert _order(fits, auto_first=True, verified_first=True)[0] == "auto-marg"


def test_fit_sort_type_and_score():
    fits = [
        _fit("p-low", fit="perfect", full=5.0, pop=10, score=100),
        _fit("g-high", fit="good", full=7.0, pop=90, score=88),
        _fit("m-gpu", fit="marginal", full=9.5, pop=100, score=50),
        _fit("g-seq", fit="good", full=12.0, seq=8.0, pop=20, score=99),
    ]
    # type: 種類 → 段階(perfect → good → marginal)→ 公式 → 人気 → Score → ID
    assert _order(fits, sort="type", auto_first=False) == ["p-low", "g-high", "g-seq", "m-gpu"]
    # score: 鍵 1 → 逃がさずに載る / Marginal → Score → 人気 → ID
    assert _order(fits, sort="score", auto_first=False) == ["p-low", "g-seq", "g-high", "m-gpu"]


# --- §8 公平さ ------------------------------------------------------------------------------------


def _measured_ids(db) -> list[str]:
    return sorted({m.model_id for m in db.measurements if db.get(m.model_id) is not None})


def test_a_settings_do_not_depend_on_measurements(db):
    """(a) 実測のあるモデルの判定の設定は、実測の有無で同じ(種類ごとの既定。
    大きさの入力が無いものは大きさなし)。"""
    for mid in _measured_ids(db):
        entry = db.get(mid)
        for system in (nvidia(RTX_5080, 15.9), nvidia(RTX_3070, 8.0)):
            a = estimate(entry, system, None, db.measurements).settings
            b = estimate(entry, system, None, []).settings
            assert a == b, mid
            want = {k: v for k, v in vars(DEFAULT_SETTINGS[entry.modality]).items() if v is not None}
            if no_size_input(entry):
                want = {k: v for k, v in want.items() if k not in ("width", "height")}
            assert a == {**want, "batch": 1}, mid


@pytest.mark.parametrize(
    ("name", "vram", "ram"),
    [("NVIDIA GeForce RTX 4060 Ti", 16.0, 32.0), ("NVIDIA GeForce RTX 4090", 24.0, 64.0)],
)
def test_b_no_measured_level_on_other_gpus(db, name, vram, ram):
    """(b) 測った GPU と同じでない PC では、どの行も段階を実測で決めない
    (作者の実測もほかの利用者と同じ規則)。"""
    system = nvidia(name, vram, ram)
    for mid in _measured_ids(db):
        entry = db.get(mid)
        for s in [None, *(m.settings for m in db.measurements if m.model_id == mid)]:
            e = estimate(entry, system, s, db.measurements)
            assert e.fit_from != "measurement", (name, mid, s)


def test_c_measured_level_on_the_same_gpu(db):
    """(c) 測った GPU と同じ PC(RTX 5080)で、実測と同じ設定を指定し、別々の出どころ 2 つがそろえば、
    段階は実測で決まる(SDXL 832px: 作者の実測 1 行 + ほかの利用者の提出 1 つ)。
    作者の 1 行だけでは決めない。"""
    entry = db.get("image_sdxl_simple")
    system = nvidia(RTX_5080, 15.9)
    s = Settings(width=832, height=832)
    prec = gpu_precision(entry, system, s)
    own = [m for m in db.measurements if m.model_id == entry.id]
    lvl, why = measured_level(own, entry, s, system, prec, 0.0)
    assert lvl is None and "出どころが 1 つだけ" in why
    other = Measurement(
        model_id=entry.id,
        gpu_name=RTX_5080,
        gpu_vram_gb=16.0,
        peak_vram_gb=8.2,
        settings=s,
        origin="community",
        n_submissions=1,
        n_used=1,
        n_sources=1,
        n_oom=0,
        consistent=False,
    )
    lvl, _ = measured_level([*own, other], entry, s, system, prec, 0.0)
    assert lvl is not None and lvl.level == "perfect" and lvl.sources == 2 and lvl.amount_gb == 8.2
    e = estimate(entry, system, s, [*own, other])
    assert e.fit == "perfect" and e.run_mode == "gpu"
    # Wan 2.2 5B(作者の実測 2 行)は、重み + 固定分(約 18.4 GB)が 15.9 GB に収まらないので決めない
    wan = estimate(
        db.get("video_wan2_2_5B_ti2v"), system, Settings(width=640, height=640, frames=73), db.measurements
    )
    assert wan.basis == "measured" and wan.fit_from == "estimate"


def test_d_values_shift_by_at_most_the_delta(db):
    """(d) 実測の有無による要る VRAM の差は、ずらした量(_prepare の delta)以内(どの動かし方の値も)。"""
    c = load_coefficients()
    system = nvidia("NVIDIA GeForce RTX 4060 Ti", 16.0)
    for mid in _measured_ids(db):
        entry = db.get(mid)
        a = estimate(entry, system, None, db.measurements, c)
        b = estimate(entry, system, None, [], c)
        if a.unknown_size:
            continue
        from genmodelfit.estimate import settings_for

        s = settings_for(entry, None, sized=not no_size_input(entry))
        delta = abs(_prepare(entry, db.measurements, c, s, system.best_gpu).delta)
        for k in ("vram_gb", "vram_sequential_gb", "vram_offload_gb"):
            assert abs(getattr(a, k) - getattr(b, k)) <= delta + 0.011, (mid, k)
        ra, rb = usage_ratio(a, system), usage_ratio(b, system)
        if a.run_mode == b.run_mode and ra is not None and rb is not None:
            from genmodelfit.recommend import mode_label

            if mode_label(a, system) == mode_label(b, system):
                assert abs(ra - rb) <= delta / system.vram_gb + 0.002, mid


def _table_view(sel):
    return [(f.model.id, f.score) for f in sel.fits]


def test_e_run_date_does_not_change_table_order_or_score(db, monkeypatch):
    """(e) 実行した日を変えても、表の並びと Score は同じ(--json は gate の HEAD の期限で変わりうる)。"""
    from datetime import datetime, timezone

    views = []
    for day in (datetime(2026, 10, 21, tzinfo=timezone.utc), datetime(2027, 3, 1, tzinfo=timezone.utc)):
        monkeypatch.setattr(gate, "_now", lambda day=day: day)
        sel = select(db, plain(12), verified_first=False, include_unknown_license=True)
        views.append(_table_view(sel))
    assert views[0] == views[1]


def test_f_allowlist_does_not_change_table_order_or_score(db):
    """(f) 許可リストを変えても、表の並びと Score は同じ(--json の並びだけ変わる)。"""
    a = select(db, plain(12), verified_first=False, include_unknown_license=True)
    strict = GatePolicy(license_allowlist=frozenset({"mit"}))
    b = select(db, plain(12), verified_first=False, include_unknown_license=True, policy=strict)
    assert _table_view(a) == _table_view(b)
    ja = select(db, plain(12), include_unknown_license=True)
    jb = select(db, plain(12), include_unknown_license=True, policy=strict)
    assert {f.model.id: f.score for f in ja.fits} == {f.model.id: f.score for f in jb.fits}


def test_g_measurements_without_size_are_used(db):
    """(g) 大きさの入力の無いテンプレートの実測(BiRefNet・MoGe)は、その GPU の PC の判定に使われる(§3.4)。"""
    system = nvidia(RTX_3070, 8.0)
    for mid in ("utility_birefnet_remove_background", "3d_moge_perspective_to_mesh"):
        e = estimate(db.get(mid), system, None, db.measurements)
        assert e.basis_detail != "estimated" and e.sample_count == 1, mid
        assert any("RTX 3070" in r for r in e.reasons), mid


# --- §3.5 実測で段階を決める条件 -------------------------------------------------------------------


def test_fake_small_submission_does_not_make_flux_perfect(db):
    """22 GB 要る Flux に「1.0 GB で動いた」の作り物を足しても、8 GB の RTX 3070 で Perfect にならない。
    出どころ 1 つなら段階を決めず、そろった出どころが 3 つでも段階に使う量は重み + 固定分より下げない。"""
    entry = db.get("flux1_krea_dev")
    system = nvidia(RTX_3070, 8.0)
    base = estimate(entry, system, None, [])
    s = Settings(**{k: v for k, v in base.settings.items() if k != "batch"})
    fake = Measurement(
        model_id=entry.id,
        gpu_name=RTX_3070,
        gpu_vram_gb=8.0,
        peak_vram_gb=1.0,
        settings=s,
        origin="community",
        n_submissions=1,
        n_used=1,
        n_sources=1,
        n_oom=0,
        consistent=False,
    )
    for m in (fake, Measurement(**{**vars(fake), "n_sources": 3, "n_used": 3, "consistent": True})):
        e = estimate(entry, system, None, [m])
        assert e.fit != "perfect" and e.fit_from == "estimate"
        assert e.vram_gb >= 8.0  # 表の VRAM は物理的な下限(重み + 固定分)より下がらない


def test_one_source_with_many_files_does_not_set_the_level(db):
    """1 人(同じ PR)が 4 件のほぼ同じ提出を出しても、出どころは 1 つ。段階は決めない。"""
    subs = [_sub(p) for p in (6.0, 6.01, 6.02, 6.03)]
    for i, sub in enumerate(subs):
        sub.path = f"p{i}.json"
    rows = contrib.aggregate(subs, {s.path: "commit-a" for s in subs})
    assert len(rows) == 1 and rows[0]["n_used"] == 4 and rows[0]["n_sources"] == 1
    m = Measurement.from_dict(rows[0])
    entry = db.get(m.model_id)
    system = nvidia(RTX_3070, 8.0)
    prec = gpu_precision(entry, system, m.settings)
    lvl, why = measured_level([m], entry, m.settings, system, prec, 0.0)
    assert lvl is None and "出どころが 1 つだけ" in why
    e = estimate(entry, system, m.settings, [m])
    assert e.sample_count == 1 and e.fit_from == "estimate"
    # 別々の 2 つの PR からなら決まる
    rows = contrib.aggregate(subs, {s.path: f"commit-{i % 2}" for i, s in enumerate(subs)})
    m2 = Measurement.from_dict(rows[0])
    assert m2.n_sources == 2 and m2.consistent
    assert measured_level([m2], entry, m2.settings, system, prec, 0.0)[0] is not None


def test_author_one_and_community_one_agree(db):
    """作者の実測 1 行 + ほかの利用者の提出 1 つがそろえば段階を決める。
    そろわなければ(幅 15% 超・OOM)決めない。"""
    entry = db.get("utility_birefnet_remove_background")
    system = nvidia(RTX_3070, 8.0)
    own = [m for m in db.measurements if m.model_id == entry.id]
    assert len(own) == 1
    s = own[0].settings
    prec = gpu_precision(entry, system, s)
    other = Measurement(
        **{
            **vars(own[0]),
            "peak_vram_gb": 1.3,
            "origin": "community",
            "n_submissions": 1,
            "n_used": 1,
            "n_sources": 1,
            "n_oom": 0,
            "consistent": False,
        }
    )
    lvl, _ = measured_level([*own, other], entry, s, system, prec, 0.0)
    assert lvl is not None and lvl.sources == 2 and lvl.level == "perfect"
    far = Measurement(**{**vars(other), "peak_vram_gb": 2.5})  # 幅が中央値の 15% を超える
    assert measured_level([*own, far], entry, s, system, prec, 0.0)[0] is None
    oom = Measurement(**{**vars(other), "n_oom": 1})  # 同じ構成の OOM の報告
    assert measured_level([*own, oom], entry, s, system, prec, 0.0)[0] is None
    no_run = Measurement(**{**vars(other), "runs_evidence": False})
    assert measured_level([*own, no_run], entry, s, system, prec, 0.0)[0] is None
