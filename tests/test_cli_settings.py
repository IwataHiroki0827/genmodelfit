# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""設定の指定(--width / --height / --frames / --length)が、CLI の --json の判定にきちんと効くか(#99)。
値を弾く検査は test_estimate.py にある。ここは「大きくすれば必要な VRAM は増え、段階は良くならない」を
CLI の入口から確かめる。具体的な値は決め打ちせず、出てきた見積もりどうしの関係で見る。"""

from __future__ import annotations

import importlib
import json

import pytest

from genmodelfit import cli

estimate = importlib.import_module("genmodelfit.estimate")  # パッケージの estimate は関数で隠れる

LEVEL_RANK = {"perfect": 3, "good": 2, "marginal": 1, "too_tight": 0}


@pytest.fixture
def run(capsys):
    """CLI を --json で 1 回動かして結果を読む。VRAM / RAM は指定の値、一覧はキャッシュを使わず同梱のもの。"""

    def _run(*argv: str, vram: float = 16, ram: float = 32) -> dict:
        capsys.readouterr()
        assert cli.main([*argv, "--json", "--vram", str(vram), "--ram", str(ram), "--no-cache"]) == 0
        return json.loads(capsys.readouterr().out)

    return _run


def _est(run, model: str, *flags: str, vram: float = 16) -> dict:
    return run("info", model, *flags, vram=vram)["estimate"]


def _rank(est: dict) -> int:
    return LEVEL_RANK[est["fit"]]


# --- 画像 ------------------------------------------------------------------------------------------


def test_image_size_is_reflected_and_vram_grows(run):
    ests = [_est(run, "image_sdxl_simple", "--width", str(s), "--height", str(s)) for s in (512, 1024, 4096)]
    for s, e in zip((512, 1024, 4096), ests, strict=True):
        assert e["settings"]["width"] == s and e["settings"]["height"] == s
    # #100: 全部載せの作業用メモリは一番重い段。SDXL は 512px から 2000px ほどまで VAE のタイル
    # (VRAM によらない 1.06 GB)が一番重いので同じ値で、それより大きくすると生成の段が効いて増える
    vram = [e["vram_gb"] for e in ests]
    assert vram[0] <= vram[1] < vram[2]
    seq = [e["vram_sequential_gb"] for e in ests]
    assert seq[0] <= seq[1] < seq[2]


def test_image_width_and_height_are_independent(run):
    wide = _est(run, "image_sdxl_simple", "--width", "4096", "--height", "512")
    sq = _est(run, "image_sdxl_simple", "--width", "4096", "--height", "4096")
    assert (wide["settings"]["width"], wide["settings"]["height"]) == (4096, 512)
    assert wide["vram_gb"] < sq["vram_gb"]  # #100: 上の試験と同じ理由で大きな画像で比べる


@pytest.mark.parametrize("vram", [8, 12, 16, 24])
def test_image_level_never_improves_with_size(run, vram):
    sizes = (256, 512, 1024, 2048, 4096, 8192)
    ranks = [
        _rank(_est(run, "image_sdxl_simple", "--width", str(s), "--height", str(s), vram=vram)) for s in sizes
    ]
    assert ranks == sorted(ranks, reverse=True), ranks


@pytest.mark.parametrize("vram", [8, 12])
def test_image_much_larger_size_is_worse_than_a_good_default(run, vram):
    default = _est(run, "image_sdxl_simple", vram=vram)
    assert default["fit"] == "good"  # この VRAM では既定が Good(段階が変わったらこの VRAM を選び直す)
    huge = _est(run, "image_sdxl_simple", "--width", "8192", "--height", "8192", vram=vram)
    assert _rank(huge) < _rank(default)
    assert huge["vram_gb"] > default["vram_gb"]


# --- 動画 ------------------------------------------------------------------------------------------


def test_video_settings_reflect_inputs_and_vram_grows_with_frames(run):
    ests = [
        _est(run, "text_to_video_wan", "--width", "832", "--height", "480", "--frames", str(n))
        for n in (17, 49, 81)
    ]
    for n, e in zip((17, 49, 81), ests, strict=True):
        assert (e["settings"]["width"], e["settings"]["height"], e["settings"]["frames"]) == (832, 480, n)
    vram = [e["vram_gb"] for e in ests]
    assert vram[0] < vram[1] < vram[2]
    ranks = [_rank(e) for e in ests]
    assert ranks == sorted(ranks, reverse=True)


def test_video_vram_grows_with_resolution(run):
    ests = [
        _est(run, "text_to_video_wan", "--width", str(w), "--height", str(h), "--frames", "49")
        for w, h in ((640, 360), (832, 480), (1280, 720), (1920, 1080))
    ]
    vram = [e["vram_gb"] for e in ests]
    assert vram == sorted(vram) and len(set(vram)) == len(vram)
    ranks = [_rank(e) for e in ests]
    assert ranks == sorted(ranks, reverse=True)
    assert ests[2]["settings"]["width"] == 1280 and ests[2]["settings"]["height"] == 720


# --- 音声 ------------------------------------------------------------------------------------------


def test_audio_length_is_recorded_and_vram_does_not_shrink(run):
    ests = [_est(run, "audio_ace_step_1_5_checkpoint", "--length", str(s)) for s in (10, 30, 120)]
    assert [e["settings"]["seconds"] for e in ests] == [10, 30, 120]
    vram = [e["vram_gb"] for e in ests]
    assert vram == sorted(vram)  # 長さの効きは小さい(docs/accuracy)ので、減らないことだけ見る
    ranks = [_rank(e) for e in ests]
    assert ranks == sorted(ranks, reverse=True)


# --- 一覧(recommend / fit)--------------------------------------------------------------------------


def _best_by_group(out: dict) -> dict[str, int]:
    """同じモデル(group_key)の中の最良の段階。設定が変わると、表に出す代表の項目が入れ替わることがあるので、
    ID でなくグループで比べる。"""
    best: dict[str, int] = {}
    for m in out["models"]:
        k = m["group_key"]
        best[k] = max(best.get(k, 0), LEVEL_RANK[m["fit_level"]])
    return best


def _runnable_groups(out: dict) -> set[str]:
    return {k for k, r in _best_by_group(out).items() if r >= LEVEL_RANK["marginal"]}


def test_recommend_video_settings_carry_through_and_bigger_job_runs_fewer(run):
    base = run("recommend", "-m", "video", "--limit", "100000", vram=16, ram=32)
    big = run(
        "recommend",
        "-m",
        "video",
        "--width",
        "1280",
        "--height",
        "720",
        "--frames",
        "81",
        "--limit",
        "100000",
    )
    assert big["models"]
    for m in big["models"]:
        s = m["estimate"]["settings"]
        if m["id"] in estimate.NO_SIZE_INPUT:
            continue
        assert (s["width"], s["height"], s["frames"]) == (1280, 720, 81), m["id"]
    assert _runnable_groups(big) <= _runnable_groups(base)


def test_fit_image_bigger_size_never_runs_more_models(run):
    base = run("fit", "-m", "image", "--limit", "100000")
    big = run("fit", "-m", "image", "--width", "2048", "--height", "2048", "--limit", "100000")
    assert big["models"]
    for m in big["models"]:
        s = m["estimate"]["settings"]
        if m["id"] in estimate.NO_SIZE_INPUT:
            continue
        assert (s["width"], s["height"]) == (2048, 2048), m["id"]
    assert _runnable_groups(big) <= _runnable_groups(base)
    base_best = _best_by_group(base)
    for k, r in _best_by_group(big).items():
        if k in base_best:
            assert r <= base_best[k], k


# --- 大きさの入力の無いテンプレート ----------------------------------------------------------------


@pytest.mark.parametrize("model", sorted(estimate.NO_SIZE_INPUT))
def test_size_less_template_has_no_default_size_but_takes_a_given_one(run, model):
    plain = _est(run, model)
    assert "width" not in plain["settings"] and "height" not in plain["settings"]
    given = _est(run, model, "--width", "1024", "--height", "768")
    assert (given["settings"]["width"], given["settings"]["height"]) == (1024, 768)


# --- そのモデルに関係の無い指定 ----------------------------------------------------------------------


def test_irrelevant_flags_are_recorded_but_do_not_change_the_estimate(run):
    """今の動き(#99 の報告)。画像に --frames、動画に --length を付けても、エラーも注意も出ず、
    設定に値がそのまま載る。見積もり(VRAM・段階)は変わらない。"""
    img = _est(run, "image_sdxl_simple")
    img_f = _est(run, "image_sdxl_simple", "--frames", "49")
    assert img_f["settings"]["frames"] == 49 and "frames" not in img["settings"]
    assert (img_f["vram_gb"], img_f["fit"]) == (img["vram_gb"], img["fit"])
    assert not any("frames" in r or "コマ" in r for r in img_f["reasons"])

    vid = _est(run, "text_to_video_wan")
    vid_l = _est(run, "text_to_video_wan", "--length", "30")
    assert vid_l["settings"]["seconds"] == 30 and "seconds" not in vid["settings"]
    assert (vid_l["vram_gb"], vid_l["fit"]) == (vid["vram_gb"], vid["fit"])
