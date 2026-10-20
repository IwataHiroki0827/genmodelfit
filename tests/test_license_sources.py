"""出どころの表(license_sources.py、#28)の検査。

- 実在するファイル名(同梱の一覧に出てくるもの)で、表が当てるライセンスを人が確かめた値と比べる
- 表に無いファイルには何も当てない(置き場所が違う・HF 以外・すでに決まっている・確かめられない部品)
- 寛容なライセンスは、根拠の文にそのライセンスが書いてある行からしか出ない
- 同梱の一覧に当てて、新しく auto になる項目が、人が確かめた一覧の中にだけあること
"""

from __future__ import annotations

import copy

import pytest

from genmodelfit import license_sources as ls
from genmodelfit.db import load
from genmodelfit.gate import GatePolicy, evaluate_offline, resolve_file_license
from genmodelfit.hf import (
    PERMISSIVE,
    TIER_PERMISSIVE,
    RepoInfo,
    Resolved,
    is_repackager,
    license_tier,
    lineage_for,
    parse_hf_url,
)
from genmodelfit.models import ModelEntry, ModelFile
from genmodelfit.update import apply_license

NC = "cc-by-nc-4.0"
HUNYUAN = ("other", "tencent-hunyuan-community")
KREA = ("other", "krea-2-community-license")

# (置き場所, ファイル名, ライセンス, license_name)。2026-10-20 に一次資料で確かめた値(根拠は表の evidence)
CASES = [
    ("Kijai/WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32.safetensors", NC, None),
    ("Kijai/WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32_v2.safetensors", NC, None),
    ("Kijai/WanVideo_comfy", "Wan21_CausVid_bidirect2_T2V_1_3B_lora_rank32.safetensors", NC, None),
    (
        "Kijai/WanVideo_comfy",
        "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        "apache-2.0",
        None,
    ),
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "apache-2.0",
        None,
    ),
    (
        "Comfy-Org/Wan-Animate-2",
        "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "apache-2.0",
        None,
    ),
    (
        "Kijai/WanVideo_comfy",
        "Lightx2v/lightx2v_T2V_14B_cfg_step_distill_v2_lora_rank64_bf16.safetensors",
        "apache-2.0",
        None,
    ),
    (
        "Kijai/WanVideo_comfy",
        "LoRAs/Wan22_relight/WanAnimate_relight_lora_fp16.safetensors",
        "apache-2.0",
        None,
    ),
    (
        "Kijai/WanVideo_comfy_fp8_scaled",
        "WanMove/Wan21-WanMove_fp8_scaled_e4m3fn_KJ.safetensors",
        "apache-2.0",
        None,
    ),
    ("Comfy-Org/Wan-Animate-2", "wan_animate_2_int8_convrot.safetensors", "apache-2.0", None),
    ("Comfy-Org/Wan-Animate-2", "wan_animate_2_distill_int8_convrot.safetensors", "apache-2.0", None),
    (
        "Comfy-Org/MiniMax-Music-3",
        "minimax_music3_dav.safetensors",
        "other",
        "minimax-music3-community-license",
    ),
    (
        "Comfy-Org/MiniMax-Music-3",
        "minimax_music3_dit_fp16.safetensors",
        "other",
        "minimax-music3-community-license",
    ),
    (
        "Comfy-Org/MiniMax-Music-3",
        "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
        "other",
        "minimax-music3-community-license",
    ),
    ("Comfy-Org/HunyuanVideo_repackaged", "hunyuan_video_t2v_720p_bf16.safetensors", *HUNYUAN),
    ("Comfy-Org/HunyuanVideo_repackaged", "hunyuan_video_vae_bf16.safetensors", *HUNYUAN),
    ("Kijai/HunyuanVideo_comfy", "hunyuan_video_vae_bf16.safetensors", *HUNYUAN),
    ("Comfy-Org/NewBie-image-Exp0.1_repackaged", "gemma_3_4b_it_bf16.safetensors", "gemma", None),
    ("Comfy-Org/NewBie-image-Exp0.1_repackaged", "jina_clip_v2_bf16.safetensors", NC, None),
    (
        "Kijai/LTXV2_comfy",
        "ltx-2-19b-distilled_transformer_only_bf16.safetensors",
        "other",
        "ltx-2-community-license-agreement",
    ),
    (
        "Kijai/LTXV2_comfy",
        "ltx-2-19b-embeddings_connector_distill_bf16.safetensors",
        "other",
        "ltx-2-community-license-agreement",
    ),
    ("Comfy-Org/ltx-2", "gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors", "gemma", None),
    ("Comfy-Org/HunyuanImage_2.1_ComfyUI", "qwen_2.5_vl_7b.safetensors", "apache-2.0", None),
    ("circlestone-labs/Anima", "qwen_3_06b_base.safetensors", "apache-2.0", None),
    (
        "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        "chrono_edit_14B_fp16.safetensors",
        "other",
        "nvidia-open-model-license",
    ),
    ("Comfy-Org/ERNIE-Image", "ministral-3-3b.safetensors", "apache-2.0", None),
    (
        "ByteZSzn/Flux.2-Turbo-ComfyUI",
        "Flux_2-Turbo-LoRA_comfyui.safetensors",
        "other",
        "flux-dev-non-commercial-license",
    ),
    ("Comfy-Org/flux2-dev", "mistral_3_small_flux2_bf16.safetensors", "apache-2.0", None),
    ("Comfy-Org/flux2-dev", "mistral_3_small_flux2_fp8.safetensors", "apache-2.0", None),
    ("Comfy-Org/flux2-klein-9B", "qwen_3_8b_fp8mixed.safetensors", "apache-2.0", None),
    ("Comfy-Org/vae-text-encorder-for-flux-klein-9b", "qwen_3_8b.safetensors", "apache-2.0", None),
    (
        "black-forest-labs/flux-3-image-klein-comfy",
        "qwen3.5_9b_flux3_prompt_upsampler_w4a8.safetensors",
        "other",
        "flux-kommunity-license",
    ),
    ("Comfy-Org/MOSS-Image", "qwen3.5_9b_w4a8_gptq.safetensors", "apache-2.0", None),
    ("Kijai/SUPIR_pruned", "SUPIR-v0Q_fp16.safetensors", "other", "supir-non-commercial"),
    ("Comfy-Org/stable-audio-3", "t5gemma_b_b_ul2.safetensors", "gemma", None),
    ("Comfy-Org/void-model", "vae/cogvideox_vae.safetensors", "apache-2.0", None),
    ("Comfy-Org/HuMo_ComfyUI", "whisper_large_v3_fp16.safetensors", "apache-2.0", None),
    ("Comfy-Org/PixelDiT", "gemma_2_2b_it_elm_bf16.safetensors", "gemma", None),
    ("Comfy-Org/PixelDiT", "pid_flux1_1024_to_4096_4step_bf16.safetensors", "other", "nsclv1"),
    ("Comfy-Org/Wan-Dancer", "wan2.2_dancer_14b_global_fp8_scaled.safetensors", "apache-2.0", None),
    ("Comfy-Org/Wan-Dancer", "wan2.2_dancer_14b_local_fp8_scaled.safetensors", "apache-2.0", None),
    ("Comfy-Org/Krea-2", "krea2_style_reference.safetensors", *KREA),
    ("Comfy-Org/Krea-2", "krea2_darkbrush.safetensors", *KREA),
    ("Comfy-Org/Krea-2", "qwen3vl_4b_fp8_scaled.safetensors", *KREA),
    ("k2-fsa/OmniVoice", "model.safetensors", "other", "cc-by-nc"),
]

# 表に当たらない(不明のまま)ことを確かめたもの。(置き場所, ファイル名)。理由は注に書く
NOT_COVERED = [
    # 本家 microsoft/Mage-Flow・microsoft/Lens は匿名で読めない(401)。
    # Lens の GitHub は MIT だが重みの記載が無い
    ("Comfy-Org/Mage-Flow", "mage_flow_int8_convrot.safetensors"),
    ("Comfy-Org/Lens", "lens_bf16.safetensors"),
    # Lens のテキストエンコーダは microsoft/Lens の text_encoder から読む(gpt-oss そのままとは確かめられない)
    ("Comfy-Org/Lens", "gpt_oss_20b_nvfp4.safetensors"),
    # 本家を確かめられない部品・元を読めない系統
    ("Comfy-Org/SDPose", "rt_detr_v4-x-hgnet_fp16.safetensors"),
    ("Comfy-Org/SDPose", "sdpose_wholebody_fp16.safetensors"),
    ("Comfy-Org/lotus", "lotus-depth-d-v1-1.safetensors"),
    ("Comfy-Org/void-model", "void_pass1.safetensors"),
    # ByteZSzn の v2 は fal の LoRA と sha256 が違い、ByteZSzn にライセンスの記載が無い
    ("Comfy-Org/flux2-dev", "Flux2TurboComfyv2.safetensors"),
    # Mistral の FLUX.2 [dev] 用でも、置き場所が違えば当てない
    ("Comfy-Org/flux2-klein-9B", "mistral_3_small_flux2_bf16.safetensors"),
    # LoRA の置き場所にライセンスの記載が無い
    ("circlestone-labs/Anima-Official-LoRAs", "anima-turbo-lora-v0.2.safetensors"),
    # lightx2v の GitHub は Apache-2.0 だが、重みは MiniMax-H3(独自のライセンス)の派生
    ("lightx2v/Minimax-h3-Turbo", "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors"),
    # HunyuanVideo のテキストエンコーダ(xtuner/llava-llama-3-8b にライセンスの記載が無い)
    ("Comfy-Org/HunyuanVideo_repackaged", "llava_llama3_fp8_scaled.safetensors"),
    # FILM の重みは GitHub の外(Google Drive)で、重みのライセンスの記載が無い
    ("Comfy-Org/frame_interpolation", "film_net_fp16.safetensors"),
    # 同じ名前でも、表に無い置き場所
    ("someone/WanVideo_copy", "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"),
    ("someone/Anima", "qwen_3_06b_base.safetensors"),
]


def _url(repo: str, name: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{name}"


def test_cases_are_many():
    assert len(CASES) >= 30


@pytest.mark.parametrize("repo,name,lic,lic_name", CASES, ids=[f"{c[0]}:{c[1]}" for c in CASES])
def test_table_license_for_real_files(repo, name, lic, lic_name):
    row = ls.source_for(repo, name)
    assert row is not None, (repo, name)
    assert (row.license, row.license_name) == (lic, lic_name)
    f = ModelFile(name=name.rsplit("/", 1)[-1], url=_url(repo, name), repo_id=repo)
    assert ls.apply_to_file(f) is row
    assert (f.license, f.license_name) == (lic, lic_name)
    assert f.license_from == row.source and f.license_url == row.url
    assert f.license_path[0] == repo and f.license_path[-1] == row.source
    assert f"{ls.EVIDENCE_KEY}:{row.key}" in f.license_evidence
    assert any(ls.CHECKED in e for e in f.license_evidence)


@pytest.mark.parametrize("repo,name", NOT_COVERED, ids=[f"{c[0]}:{c[1]}" for c in NOT_COVERED])
def test_not_covered_stays_unknown(repo, name):
    assert ls.source_for(repo, name) is None
    f = ModelFile(name=name, url=_url(repo, name), repo_id=repo)
    assert ls.apply_to_file(f) is None
    assert f.license is None and f.license_from is None and f.license_evidence == []


def test_never_overwrites_known_or_non_hf():
    repo, name = "Comfy-Org/Wan-Animate-2", "wan_animate_2_int8_convrot.safetensors"
    known = ModelFile(name=name, url=_url(repo, name), repo_id=repo, license="other", license_name="x")
    assert ls.apply_to_file(known) is None and (known.license, known.license_name) == ("other", "x")
    civitai = ModelFile(name=name, url="https://civitai.com/api/download/models/1", repo_id=None)
    assert ls.apply_to_file(civitai) is None and civitai.license is None
    no_url = ModelFile(name=name)
    assert ls.apply_to_file(no_url) is None and no_url.license is None


def test_permissive_only_from_evidence_naming_the_license():
    names = {
        "apache-2.0": ("apache-2.0", "apache license 2.0", "apache 2.0"),
        "mit": ("mit license", "license: mit"),
    }
    for row in ls.SOURCES:
        if license_tier(row.license, row.license_name) != TIER_PERMISSIVE:
            continue
        assert row.license in PERMISSIVE
        text = row.evidence.lower()
        assert any(n in text for n in names[row.license]), row.key


def test_rows_are_well_formed():
    keys = [r.key for r in ls.SOURCES]
    assert len(keys) == len(set(keys))
    for row in ls.SOURCES:
        # 元は再配布のリポジトリにしない
        # (gate が「再配布側の表記だけ」として落とすため。元をたどったことが要る)
        assert not is_repackager(row.source), row.key
        assert row.repos and all("/" in r for r in row.repos), row.key
        assert row.evidence and "http" in row.evidence + row.url, row.key
        assert row.license == "other" or row.license_name is None, row.key
        if row.license == "other":
            assert row.license_name, row.key
    # 表に載っている行が、表に出てくるすべての行を少なくとも 1 つの実例で確かめている
    covered = {ls.source_for(r, n).key for r, n, _, _ in CASES}
    assert covered == set(keys)


def test_lineage_and_unverified_never_match():
    # 元を読めない系統の名前は、表に行があっても当てない(VOID・SDPose・Lotus)
    assert lineage_for(["void_pass1.safetensors"])
    assert lineage_for(["netflix/void-model"])
    assert lineage_for(["cogvideox_vae.safetensors"]) is None


def test_update_uses_table_only_when_unknown():
    repo, path = "Comfy-Org/Wan-Animate-2", "diffusion_models/wan_animate_2_int8_convrot.safetensors"

    class Client:
        def __init__(self, res):
            self.res = res

        def resolve_file(self, info, p, *, discovered_uncensored=False):
            return self.res

    info = RepoInfo(repo_id=repo, license="apache-2.0")
    f = ModelFile(name=path.rsplit("/", 1)[-1], url=_url(repo, path), repo_id=repo)
    unknown = Resolved(None, None, repo, None, [repo], ["license_unverified: x"])
    apply_license(f, info, Client(unknown), path)
    assert f.license == "apache-2.0" and f.license_from == "Wan-AI/Wan2.2-Animate-2-14B"
    # 自動で決まったものは表で上書きしない
    g = ModelFile(name=f.name, url=f.url, repo_id=repo)
    known = Resolved("other", "y", "a/b", None, [repo, "a/b"], [])
    apply_license(g, info, Client(known), path)
    assert (g.license, g.license_name, g.license_from) == ("other", "y", "a/b")


# --- 同梱の一覧に当てる -------------------------------------------------------------------------

# 表を当てて新しく auto になってよい項目(2026-10-20 に全ファイルのライセンスを人が確かめた)。
# ここに無い項目が auto になったら、表の行を見直す(寛容の側に間違えていないか)
JUSTIFIED_AUTO = {
    # baidu/ERNIE-Image・-Turbo(apache-2.0)、ministral-3-3b(表)、FLUX.2 VAE(apache-2.0)
    "image_ernie_image",
    "image_ernie_image_turbo",
    # OpenMOSS-Team/MOSS-Image-Base・-SFT(apache-2.0)、Qwen3.5-9B(表)、FLUX.2 small decoder(apache-2.0)
    "image_moss_image_base_t2i",
    "image_moss_image_base_t2i_bboxes",
    "image_moss_image_t2i",
    "image_moss_image_t2i_bboxes",
    # ByteDance/Bernini-R(apache-2.0)、lightx2v の T2V v2 LoRA(表)、UMT5・Wan 2.1 VAE
    "video_bernini_r_image_editing",
    "video_bernini_r_video_editing",
    # bytedance-research/HuMo(apache-2.0)、whisper-large-v3(表)、lightx2v の I2V LoRA(表)
    "video_humo",
    # htdong/Wan-Alpha(apache-2.0)、Wan2.1-T2V-14B、lightx2v の T2V v2 LoRA(表)
    "video_wan2.1_alpha_t2v_14B",
    # MeiGen-AI/InfiniteTalk(apache-2.0)、chinese-wav2vec2-base(mit)、lightx2v の I2V LoRA(表)
    "video_wan2_1_infinitetalk",
    # Wan-AI/Wan2.2-Animate-2-14B(表)、lightx2v の I2V LoRA(表)、CLIP-ViT-H(mit)
    "video_wan_animate2",
    "video_wan_animate2_distilled",
    # Wan-AI/Wan-Dancer-14B(表)
    "video_wan_dancer",
    # Wan-AI/Wan2.1-VACE-14B(apache-2.0)、lightx2v の I2V LoRA(表)
    "video_wan_vace_flf2v",
    # Ruihang/Wan-Move-14B-480P(表)、lightx2v の I2V LoRA(表)
    "video_wanmove_480p",
}


@pytest.fixture(scope="module")
def bundled():
    return load(use_cache=False)


def test_apply_to_bundled_db_only_touches_unknown_files_in_table(bundled):
    after = copy.deepcopy(bundled)
    applied = ls.apply_license_sources(after)
    for m0, m1 in zip(bundled.models, after.models, strict=True):
        for f0, f1 in zip(m0.files, m1.files, strict=True):
            loc = parse_hf_url(f0.url)
            row = ls.source_for(loc[0], loc[2]) if loc else None
            if f0.license is None and row is not None:
                assert (f1.license, f1.license_name) == (row.license, row.license_name)
            else:
                assert f1 == f0, (m0.id, f0.name)
    # 同梱の一覧には当て済みなので、当て直しても何も変わらない(冪等)
    assert applied == []


def test_reapplying_restores_cleared_licenses(bundled):
    """表が受け持つファイルのライセンスを消してから当て直すと、同梱の一覧と同じ値に戻る。"""
    cleared = copy.deepcopy(bundled)
    n = 0
    for m in cleared.models:
        for f in m.files:
            loc = parse_hf_url(f.url)
            if loc and ls.source_for(loc[0], loc[2]) is not None and f.license is not None:
                f.license = f.license_name = f.license_url = None
                n += 1
    assert n > 0
    ls.apply_license_sources(cleared)
    for m0, m1 in zip(bundled.models, cleared.models, strict=True):
        for f0, f1 in zip(m0.files, m1.files, strict=True):
            assert (f1.license, f1.license_name) == (f0.license, f0.license_name), (m0.id, f0.name)


def test_report_on_bundled_db(bundled):
    r = ls.report(bundled)
    assert r["unknown_after"] <= r["unknown_before"]
    assert r["unknown_files_after"] <= r["unknown_files_before"]
    assert r["no_longer_auto"] == []
    assert set(r["newly_auto"]) <= JUSTIFIED_AUTO, set(r["newly_auto"]) - JUSTIFIED_AUTO


def test_justified_auto_have_only_verified_files(bundled):
    after = copy.deepcopy(bundled)
    ls.apply_license_sources(after)
    policy = GatePolicy(max_head_age_days=None)
    for mid in JUSTIFIED_AUTO:
        m = after.get(mid)
        assert m is not None, mid
        assert evaluate_offline(m, policy, sources=after.sources).auto, mid
        for f in m.files:
            assert resolve_file_license(f).license in PERMISSIVE, (mid, f.name)


def test_void_model_is_held_by_lineage(bundled):
    m = bundled.get("utility_void_video_inpainting")
    if m is None:
        pytest.skip("一覧に VOID が無い")
    mains = [f for f in m.files if "void_pass" in f.name]
    assert mains and all(resolve_file_license(f).license is None for f in mains)


def test_entries_api_accepts_list():
    f = ModelFile(
        name="ministral-3-3b.safetensors",
        url=_url("Comfy-Org/ERNIE-Image", "text_encoders/ministral-3-3b.safetensors"),
        repo_id="Comfy-Org/ERNIE-Image",
    )
    e = ModelEntry(id="x", name="x", modality="image", files=[f])
    changed = ls.apply_license_sources([e])
    assert changed == [("x", "ministral-3-3b.safetensors", "ernie-image-ministral-3-3b")]
    assert e.license == "apache-2.0"
