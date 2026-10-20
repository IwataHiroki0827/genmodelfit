"""モデルのまとめ方(大きさ = 別のモデル、精度・使い方 = 同じモデル)と、用途・補助ツールの見分け方(#17)。"""

from __future__ import annotations

import json
from collections import defaultdict

import pytest
from clihelp import run_cli

from genmodelfit import cli, load_db
from genmodelfit.groups import family_tokens, group_key, precision_label, precision_rank
from genmodelfit.models import Gpu, ModelEntry, SystemInfo
from genmodelfit.recommend import recommend, select
from genmodelfit.tasks import is_tool, main_use, task_codes, uses_of
from genmodelfit.update import template_tasks


@pytest.fixture(scope="module")
def db():
    return load_db(use_cache=False)


def pc(vram: float, ram: float = 32) -> SystemInfo:
    gpus = [Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")] if vram else []
    return SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=gpus)


# 同梱の一覧の本体のファイル名 → 精度などを外した語(目で確かめられるように書き出す)
NAMES = [
    ("z_image_turbo_bf16.safetensors", "image turbo z"),
    ("z_image_turbo_int8_convrot.safetensors", "image turbo z"),
    ("wan2.1_i2v_480p_14B_fp16.safetensors", "14b 2.1 480p i2v wan"),
    ("Wan2_1-I2V-14B-480p_fp8_e4m3fn_scaled_KJ.safetensors", "14b 2.1 480p i2v wan"),
    ("wan2.1_t2v_1.3B_fp16.safetensors", "1.3b 2.1 t2v wan"),
    ("wan2.1_t2v_14B_fp8_scaled.safetensors", "14b 2.1 t2v wan"),
    ("wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "14b 2.2 i2v wan"),
    ("wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "14b 2.2 i2v wan"),
    ("qwen_image_2512_fp8_e4m3fn.safetensors", "2512 image qwen"),
    ("qwen_image_2512_bf16.safetensors", "2512 image qwen"),
    ("qwen_image_edit_2511_fp8mixed.safetensors", "2511 edit image qwen"),
    ("qwen_image_edit_2511_int8_convrot.safetensors", "2511 edit image qwen"),
    ("flux-2-klein-4b-fp8.safetensors", "2 4b flux klein"),
    ("flux-2-klein-9b-fp8.safetensors", "2 9b flux klein"),
    ("flux-2-klein-9b.safetensors", "2 9b flux klein"),
    ("flux1-dev-fp8.safetensors", "1 dev flux"),
    ("flux1-dev.safetensors", "1 dev flux"),
    ("ace_step_v1_3.5b.safetensors", "1 3.5b acestep"),
    ("acestep_v1.5_turbo.safetensors", "1.5 acestep turbo"),
    ("ace_step_1.5_turbo_aio.safetensors", "1.5 acestep turbo"),
    ("hunyuan3d-dit-v2_fp16.safetensors", "2 dit hunyuan3d"),
    ("hunyuan_3d_v2.1.safetensors", "2.1 3d hunyuan"),
    ("ltx-2.3-22b-dev-fp8.safetensors", "2.3 22b dev ltx"),
    ("ltx-2.3-22b-dev_transformer_only_bf16.safetensors", "2.3 22b dev ltx"),
    ("ltx-2-19b-dev-fp8.safetensors", "19b 2 dev ltx"),
    ("ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors", "2.5 22b distilled ltx"),
    ("hunyuanvideo1.5_720p_t2v_fp16.safetensors", "1.5 720p hunyuanvideo t2v"),
    ("qwen-image-2.1-Q8_0.gguf", "2.1 image qwen"),
    ("flux-abliterated-v2-Q4_K_M.gguf", "2 abliterated flux"),
    ("Sakura-Iris-3B-Q8_0-4.15GiB.gguf", "3b iris sakura"),
    ("10Eros_v1.5_bf16_w4a8_audioint8.safetensors", "1.5 10eros"),
    ("10Eros_v1.5_nvfp4.safetensors", "1.5 10eros"),
    ("transformer/diffusion_pytorch_model-00001-of-00003.safetensors", "diffusion pytorch"),
]


@pytest.mark.parametrize(("name", "expected"), NAMES)
def test_family_tokens(name, expected):
    assert " ".join(family_tokens(name)) == expected


def _key(db, mid):
    return group_key(db.get(mid))


SAME = [  # 同じモデル(精度か使い方だけが違う)
    ("image_z_image_turbo", "image_z_image_turbo_int8"),
    ("image_to_video_wan", "video_wan2_1_infinitetalk"),  # Wan 2.1 I2V 14B の fp16 と fp8
    ("image_qwen_Image_2512", "templates_text_prompt_to_360hdr.app"),  # Qwen-Image 2512 の fp8 と bf16
    ("image_qwen_image_edit_2511", "image_qwen_image_edit_2511_int8"),
    ("video_wan2_2_14B_i2v", "video_wan2_2_14B_flf2v"),
    ("video_ltx2_3_t2v", "template_ltx2_3_obscura_remova_lora_remove_object_from_video"),
    ("video_wan21_scail2_character_replacement", "video_wan21_scail2_character_replacement_int8"),
    ("image_ideogram4_t2i", "image_ideogram4_t2i_int8"),
    ("video_wan_vace_14B_v2v", "video_wan_vace_inpainting"),  # 1.3B の切り替え用も載っているが本体は 14B
]
DIFFERENT = [  # 大きさ・版・種類が違う
    ("text_to_video_wan", "video_wan2.1_alpha_t2v_14B"),  # Wan 2.1 1.3B と 14B
    ("image_flux2_klein_image_edit_4b_distilled", "image_flux2_klein_image_edit_9b_distilled"),
    ("audio_ace_step_1_t2a_song", "audio_ace_step_1_5_split"),  # ACE-Step v1 3.5B と 1.5
    ("3d_hunyuan3d_image_to_model", "3d_hunyuan3d-v2.1"),
    ("video_ltx2_3_t2v", "video_ltx2_t2v"),  # LTX 2.3 22B と LTX-2 19B
    ("video_ltx2_3_t2v", "video_ltx2_5_t2v"),  # LTX 2.3 と 2.5(どちらも 22B)
    ("image_z_image", "image_z_image_turbo"),  # 素の版と turbo
    ("image_qwen_image", "image_qwen_image_edit"),
    ("video_wan2_2_14B_t2v", "video_wan2_2_14B_i2v"),  # テキストエンコーダ(umt5)は同じでも別のモデル
    ("image_qwen_image_2_1_t2i", "hf:0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF"),  # 内容(無検閲)が違う
]


@pytest.mark.parametrize(("a", "b"), SAME)
def test_same_model(db, a, b):
    assert _key(db, a) == _key(db, b)


@pytest.mark.parametrize(("a", "b"), DIFFERENT)
def test_different_model(db, a, b):
    assert _key(db, a) != _key(db, b)


def test_text_encoder_is_never_the_key(db):
    for m in db.models:
        assert "umt5" not in group_key(m) and "t5xxl" not in group_key(m), m.id


def test_groups_never_mix_type_or_content(db):
    groups = defaultdict(set)
    for m in db.models:
        groups[group_key(m)].add((m.modality, m.content))
    assert all(len(v) == 1 for v in groups.values())
    assert 200 <= len(groups) < len(db.models)


def test_precision_rank_and_label():
    order = ["bf16", "fp8", "int8", "gguf-q8_0", "gguf-q6_k", "gguf-q5_k_m", "gguf-q4_k_m", "nvfp4"]
    assert [precision_rank(p) for p in order] == sorted(precision_rank(p) for p in order)
    assert precision_rank(None) == precision_rank("fp16") == precision_rank("gguf-bf16") == 0
    assert [precision_label(p) for p in ("gguf-q4_k_m", "fp8", None)] == ["q4", "fp8", "-"]


def test_precision_auto_selection(db):
    """良い段階が先、同じ段階なら高い精度。8 GB では Z-Image Turbo は int8(good)、24 GB では bf16。"""
    ids = {f.model.id for f in recommend(db, pc(8), modality="image", limit=None)}
    assert "image_z_image_turbo_int8" in ids and "image_z_image_turbo" not in ids
    big = {f.model.id: f for f in recommend(db, pc(24, 64), modality="image", limit=None)}
    assert "image_z_image_turbo" in big and "image_z_image_turbo_int8" not in big
    f = big["image_z_image_turbo"]
    assert [v["precision"] for v in f.variants] == ["bf16", "int8"]
    assert {w["id"] for w in f.workflows} >= {
        "image_z_image_turbo_int8",
        "image_z_image_turbo_fun_union_controlnet",
    }


def test_all_workflows_and_counts(db):
    grouped = select(db, pc(8), include_tools=False)
    every = select(db, pc(8), include_tools=False, group=False)
    assert len(every.fits) == len(grouped.fits) + grouped.hidden["same_model"]
    for sel in (grouped, every):
        assert sel.matched + sum(sel.hidden.values()) == len(db.models)
    assert len({f.group_key for f in grouped.fits}) == len(grouped.fits)


def test_task_filter_picks_the_matching_workflow(db):
    edit = recommend(db, pc(24, 64), modality="image", task="edit", limit=None)
    assert edit and all(f.use == "edit" for f in edit)
    z = next(f for f in edit if f.group_key == _key(db, "image_z_image_turbo"))
    assert z.model.id == "image_z_image_turbo_fun_union_controlnet"  # 代表は編集のワークフロー
    assert task_codes(["generate"]) == task_codes(["gen"]) and "t2i" in task_codes(["gen"])
    with pytest.raises(ValueError):
        task_codes(["dance"])


# A3: 既定の recommend に出てはいけない補助ツール(深度・線画の前処理、音の分離)
TOOLS = [
    "utility_depth_anything3_image_depth_estimation",
    "utility_depth_anything3_video_depth_estimation",
    "utility_moge_depth_estimation",
    "utility-depthAnything-v2-relative-video",
    "utility-lineart-video",
    "image_lotus_depth_v1_1",
    "audio_melbandroformer_audio_separation",
]


@pytest.mark.parametrize("mid", TOOLS)
def test_utility_tools_are_tools(db, mid):
    m = db.get(mid)
    assert is_tool(m) and uses_of(m) == ["tool"] and main_use(m) == "tool"
    assert set(m.tasks) <= {"estimate", "separate"}  # 同梱の一覧のタグも直した
    # update が同じ規則で作る(タグが controlnet / text to image でも)
    t = m.template
    assert (
        template_tasks({"title": t.title, "tags": ["ControlNet", "Text to Image"]}, m.modality, t.category)
        == m.tasks
    )


@pytest.mark.parametrize("vram", [0, 8, 24])
def test_default_recommend_has_no_tools(db, vram):
    for sel in (
        select(db, pc(vram), include_tools=False, per_modality=3),
        select(db, pc(vram), include_tools=False),
    ):
        ids = {f.model.id for f in sel.fits} | {
            w["id"] for f in sel.fits for w in f.workflows if w["matched"]
        }
        assert not ids & set(TOOLS)
        assert all(not is_tool(f.model) for f in sel.fits)


def test_generative_models_are_not_tools(db):
    for mid in (
        "sd3.5_large_depth",
        "video_ltx2_depth_to_video",
        "flux_depth_lora_example",
        "image_qwen_image_layered",
    ):
        assert not is_tool(db.get(mid)), mid
    assert not is_tool(
        ModelEntry(id="x", name="Depth Map: X", modality="image", tasks=["control"])
    )  # 分類が道具でない


# A2: 壊れた・無い --db


@pytest.mark.parametrize("content", [None, "{not json", "[1, 2]", '{"schema_version": 1, "models": [1]}'])
def test_broken_db_is_a_clean_error(tmp_path, capsys, content):
    p = tmp_path / "db.json"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    assert cli.main(["fit", "--db", str(p), "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"genmodelfit", "command", "error"} and out["error"]
    assert cli.main(["recommend", "--db", str(p)]) == 1
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err.startswith("error: ")


def test_broken_db_in_subprocess_has_no_traceback(tmp_path):
    p = tmp_path / "db.json"
    p.write_text("{oops", encoding="utf-8")
    r = run_cli(["recommend", "--db", str(p), "--json"], tmp_path / "cache")
    assert r.returncode == 1 and b"Traceback" not in r.stderr
    assert "error" in json.loads(r.stdout.decode("utf-8"))
