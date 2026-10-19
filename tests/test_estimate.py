import json

import pytest

from genmodelfit import cli
from genmodelfit.calibrate import calibrate
from genmodelfit.db import bundled_text, load
from genmodelfit.estimate import (
    Coefficients,
    components,
    estimate,
    is_censored,
    limits_for,
    load_coefficients,
)
from genmodelfit.models import Gpu, Measurement, ModelEntry, ModelFile, Settings, SystemInfo, Template


def pc(vram: float, ram: float = 32.0) -> SystemInfo:
    gpus = [Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")] if vram else []
    return SystemInfo(os="test", cpu="test", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=gpus)


def f(name: str, gb: float, directory: str | None = None, role: str | None = None) -> ModelFile:
    return ModelFile(name=name, size_bytes=int(gb * 1e9), directory=directory, role=role)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def db():
    return load(use_cache=False)


# --- 部品の分け方 -------------------------------------------------------------------------------


def test_split_sdxl_checkpoint(db):
    p = components(db.get("image_sdxl_simple"))
    assert p.denoise == pytest.approx(5.13, abs=0.02)  # UNet
    assert p.encode == pytest.approx(1.64, abs=0.02)  # CLIP-L + bigG
    assert p.decode == pytest.approx(0.17, abs=0.01)
    assert p.total == pytest.approx(6.94, abs=0.01)
    assert any("SDXL" in n for n in p.notes)


def test_split_by_directory_and_generic_checkpoint():
    e = ModelEntry(
        id="x",
        name="x",
        modality="image",
        files=[
            f("dit.safetensors", 6, "diffusion_models"),
            f("te.safetensors", 4, "text_encoders"),
            f("ae", 0.3, "vae"),
        ],
    )
    p = components(e)
    assert (p.denoise, p.encode, p.decode) == pytest.approx((6, 4, 0.3))
    unknown = ModelEntry(
        id="y", name="y", modality="image", files=[f("mystery.safetensors", 10, "checkpoints")]
    )
    p = components(unknown)
    assert (p.denoise, p.encode, p.decode) == pytest.approx((7.5, 2.0, 0.5))
    assert any("一般的な割合" in n for n in p.notes)


def test_split_hf_names_without_directory():
    e = ModelEntry(
        id="hf:a/b",
        name="b",
        modality="image",
        source="huggingface",
        files=[
            f("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 5),
            f("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 3),
            f("text_encoder/model.safetensors", 2),
            f("vae/diffusion_pytorch_model.safetensors", 0.2),
        ],
    )
    p = components(e)
    assert (p.denoise, p.encode, p.decode) == pytest.approx((8, 2, 0.2))


def test_no_files_falls_back_to_template_size():
    e = ModelEntry(id="t", name="t", modality="image", template=Template(name="t", size_bytes=int(8e9)))
    p = components(e)
    assert p.total == pytest.approx(8.0) and p.denoise == pytest.approx(6.0)
    est = estimate(e, pc(16), coefficients=Coefficients())
    assert est.basis == "estimated" and est.fit in ("perfect", "good")
    empty = ModelEntry(id="z", name="z", modality="image")
    assert estimate(empty, pc(8), coefficients=Coefficients()).fit == "perfect"


def test_precision_override_scales_main_only(db):
    e = db.get("image_z_image_turbo")  # 本体 bf16 12.3 GB
    base = estimate(e, pc(16), Settings(width=1024, height=1024), coefficients=Coefficients())
    fp8 = estimate(e, pc(16), Settings(width=1024, height=1024, precision="fp8"), coefficients=Coefficients())
    assert base.vram_gb - fp8.vram_gb == pytest.approx(12.31 / 2, abs=0.02)
    assert any("bf16 → fp8" in r for r in fp8.reasons)
    # 精度を変えたら実測(bf16)は使わない
    fp8m = estimate(e, pc(16), Settings(width=768, height=768, precision="fp8"), db.measurements)
    assert fp8m.basis == "estimated"


# --- 段階 ----------------------------------------------------------------------------------------


def big_model() -> ModelEntry:
    return ModelEntry(
        id="big",
        name="big",
        modality="image",
        files=[f("dit.safetensors", 12, "diffusion_models"), f("te.safetensors", 8, "text_encoders")],
    )


def test_levels_full_sequential_offload():
    c = Coefficients()
    e = big_model()
    s = Settings(width=1024, height=1024)
    assert estimate(e, pc(32), s, coefficients=c).fit == "perfect"
    seq = estimate(e, pc(16), s, coefficients=c)  # 全部(21 GB)は載らないが、本体の段(13 GB)は載る
    assert seq.fit == "good" and seq.run_mode == "gpu" and "順に載せ替え" in seq.reasons[0]
    off = estimate(e, pc(8), s, coefficients=c)  # 本体も載らない → 一部を逃がす
    assert off.fit == "marginal" and off.run_mode == "offload"
    assert off.vram_offload_gb <= 8 < off.vram_sequential_gb <= off.vram_gb
    assert estimate(e, pc(4), s, coefficients=c).fit == "too_tight"


def test_ram_is_hard_limit_even_when_vram_is_plenty():
    e = big_model()  # 重み 20 GB + 固定分
    assert estimate(e, pc(48, ram=64), coefficients=Coefficients()).fit == "perfect"
    tight = estimate(e, pc(48, ram=16), coefficients=Coefficients())
    assert tight.fit == "too_tight" and "メモリ不足" in tight.reasons[0]


# --- 実測の扱い ----------------------------------------------------------------------------------


def test_measured_on_16gb_falls_through_on_smaller_gpu(db):
    e = db.get("image_z_image_turbo_int8")  # 16 GB で 13.2 GB(動的 VRAM)。アプリでは 6 GB から動く
    on8 = estimate(e, pc(8), None, db.measurements)
    assert on8.basis == "measured" and on8.vram_gb == pytest.approx(13.2)
    assert on8.fit == "good" and on8.vram_sequential_gb <= 8
    assert estimate(e, pc(6), None, db.measurements).fit == "marginal"


def test_censored_measurement_is_a_lower_bound(db):
    m = next(x for x in db.measurements if x.model_id == "image_z_image_turbo")
    assert is_censored(m)  # 16 GB の GPU で 15.0 GB = 上限に張り付き
    e = estimate(db.get("image_z_image_turbo"), pc(16), None, db.measurements)
    assert e.basis == "measured" and e.vram_gb > 15.0
    assert any("張り付いた" in r for r in e.reasons)
    ok = Measurement(model_id="x", gpu_name="g", gpu_vram_gb=16, peak_vram_gb=13.2)
    assert not is_censored(ok)


def test_measurement_corrects_other_settings(db):
    e = db.get("image_sdxl_simple")
    other = estimate(e, pc(16), Settings(width=1024, height=1024), db.measurements)
    assert other.basis == "estimated" and any("実測" in r and "補正" in r for r in other.reasons)
    assert other.vram_gb > 8.0  # 832px の実測 8.0 GB より大きい


# --- 設定と単調さ --------------------------------------------------------------------------------


@pytest.mark.parametrize("coef", [Coefficients(), load_coefficients()])
def test_settings_scaling_is_monotonic(db, coef):
    img, vid, aud = (
        db.get("image_sdxl_simple"),
        db.get("text_to_video_wan"),
        db.get("audio_ace_step_1_5_checkpoint"),
    )

    def v(e, **kw):
        return estimate(e, pc(16), Settings(**kw), db.measurements, coef).vram_gb

    sides = [v(img, width=px, height=px) for px in (512, 768, 1024, 1536)]
    assert sides == sorted(sides) and sides[0] < sides[-1]
    assert v(img, width=1024, height=1024, batch=1) < v(img, width=1024, height=1024, batch=4)
    frames = [v(vid, width=512, height=512, frames=n) for n in (17, 49, 81, 121)]
    assert frames == sorted(frames) and frames[0] < frames[-1]
    secs = [v(aud, seconds=t) for t in (10, 30, 120)]
    assert secs == sorted(secs)
    if coef.act_audio_gb_per_min > 0:
        assert secs[0] < secs[-1]


def test_limits_are_the_largest_good_settings(db):
    e = db.get("text_to_video_wan")
    lim = limits_for(e, pc(8), None, db.measurements)
    assert lim["level"] == "good" and lim["max_side_px"] and lim["max_frames"]
    at = Settings(width=lim["max_side_px"], height=lim["max_side_px"], frames=49)
    over = Settings(width=lim["max_side_px"] + 32, height=lim["max_side_px"] + 32, frames=49)
    assert estimate(e, pc(8), at, db.measurements).fit in ("perfect", "good")
    assert estimate(e, pc(8), over, db.measurements).fit not in ("perfect", "good")
    more = estimate(e, pc(8), Settings(width=512, height=512, frames=lim["max_frames"] + 4), db.measurements)
    assert more.fit not in ("perfect", "good")
    # 大きな GPU ほど限界は上がる
    lim16 = limits_for(e, pc(16), None, db.measurements)
    assert lim16["max_side_px"] >= lim["max_side_px"] and lim16["max_frames"] >= lim["max_frames"]
    assert limits_for(db.get("3d_hunyuan3d_image_to_model"), pc(8), None, db.measurements) is None
    big = limits_for(db.get("image_sdxl_simple"), pc(24), None, db.measurements)
    assert big["at_cap"] == ["max_side_px"]


# --- 人が決めた目安(AI生成キットのカタログの最低 / 推奨 VRAM)との突き合わせ ------------------------
# 係数を合わせるのには使っていない(docs/accuracy.md)。最低の VRAM で too_tight にならないこと、
# 推奨の VRAM で good 以上になることを見る。

CURATED = {  # id: (最低, 推奨)
    "image_z_image_turbo_int8": (6, 8),
    "image_z_image_turbo": (12, 16),
    "image_sdxl_simple": (4, 8),
    "video_wan2_2_5B_ti2v": (8, 12),
    "text_to_video_wan": (6, 8),
    "audio_ace_step_1_5_checkpoint": (4, 8),
    "3d_hunyuan3d_image_to_model": (6, 8),
    "3d_hunyuan3d-v2.1": (8, 12),
}
# 動画の作業用メモリを Wan2.1 の 1 件から伸ばしているため厳しめに出る(直ったら strict の xfail が知らせる)
KNOWN_MISS = pytest.mark.xfail(
    strict=True, reason="Wan2.2 5B は厳しめに出る(docs/accuracy.md の「分かっている外れ」)"
)
CURATED_IDS = [
    pytest.param(i, marks=KNOWN_MISS) if i == "video_wan2_2_5B_ti2v" else i for i in sorted(CURATED)
]


@pytest.mark.parametrize("model_id", CURATED_IDS)
def test_curated_min_is_not_too_tight(db, model_id):
    vmin, _ = CURATED[model_id]
    assert estimate(db.get(model_id), pc(vmin), None, db.measurements).fit != "too_tight"


@pytest.mark.parametrize("model_id", CURATED_IDS)
def test_curated_recommended_is_good(db, model_id):
    _, vrec = CURATED[model_id]
    assert estimate(db.get(model_id), pc(vrec), None, db.measurements).fit in ("perfect", "good")


def test_curated_8gb_and_4gb(db):
    on8 = {m.id: estimate(m, pc(8), None, db.measurements).fit for m in db.models}
    for mid in (
        "image_sdxl_simple",
        "image_z_image_turbo_int8",
        "text_to_video_wan",
        "audio_ace_step_1_5_checkpoint",
        "3d_hunyuan3d_image_to_model",
    ):
        assert on8[mid] != "too_tight", mid
    assert estimate(db.get("image_z_image_turbo"), pc(16), None, db.measurements).fit != "too_tight"
    # 4 GB で bf16 の Z-Image(重み 20.7 GB)は、逃がしても生成の段に 6 GB ほど要るので too_tight
    assert estimate(db.get("image_z_image_turbo"), pc(4), None, db.measurements).fit == "too_tight"


# --- 合わせ直しの再現 ----------------------------------------------------------------------------


def test_calibration_is_reproducible():
    committed = json.loads(bundled_text("calibration.json"))
    again = calibrate(date=committed["date"])
    assert again["coefficients"] == committed["coefficients"]
    assert again["summary"] == committed["summary"]
    assert again["n"] == committed["n"] == 9
    assert load_coefficients().overhead_gb == committed["coefficients"]["overhead_gb"]


def test_cli_info_shows_limits(capsys):
    assert cli.main(["info", "text_to_video_wan", "--vram", "8", "--ram", "32", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "この PC なら 長辺" in out and "コマまで good" in out
    assert cli.main(["fit", "--json", "--limits", "--vram", "8", "--ram", "32", "--no-cache"]) == 0
    data = json.loads(capsys.readouterr().out)
    wan = next(m for m in data["models"] if m["id"] == "text_to_video_wan")
    assert wan["estimate"]["limits"]["max_side_px"] and wan["estimate"]["vram_sequential_gb"]
