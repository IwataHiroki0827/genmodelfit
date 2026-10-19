"""実在するモデルで、gate の結果を人が確かめたテスト(Issue #3 の受け入れ基準)。

同梱の一覧(src/genmodelfit/data/db.json。HEAD の結果つき)を読む。ライセンスは一覧の値ではなく、
2026-10-20 に各モデルカード(HF の cardData / LICENSE)を人が読んで確かめたものを根拠に、期待する結果を書いた。
一覧の値がカードと違ったものは、その旨を注に書いた(#13 のレビューで直ったものを含む)。

期待する結果の書き方: auto=True なら保留の理由は無い。False なら held_by の先頭(表の held:<id>)と、
落ちるべき条件の全部を書く。
"""

from __future__ import annotations

import pytest

from genmodelfit.db import load
from genmodelfit.gate import evaluate_offline


@pytest.fixture(scope="module")
def db():
    return load(use_cache=False)


# (モデル ID, auto, 先頭の保留理由, 落ちるべき条件の全部)
CASES = [
    # Z-Image Turbo: apache-2.0 https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    #   テキストエンコーダ Qwen3-4B: apache-2.0 https://huggingface.co/Qwen/Qwen3-4B
    ("image_z_image_turbo", True, None, set()),
    # FLUX.1 [schnell]: apache-2.0 https://huggingface.co/black-forest-labs/FLUX.1-schnell
    #   (ファイルは 1 つにまとめた checkpoint)
    ("flux_schnell", True, None, set()),
    # Wan2.1 T2V 1.3B: apache-2.0 https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
    #   UMT5-XXL: apache-2.0 https://huggingface.co/google/umt5-xxl
    ("text_to_video_wan", True, None, set()),
    # Wan2.2 TI2V 5B: apache-2.0 https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B
    ("video_wan2_2_5B_ti2v", True, None, set()),
    # Wan2.2 T2V A14B: apache-2.0 https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B
    #   4 ステップの LoRA: apache-2.0 https://huggingface.co/lightx2v/Wan2.2-Lightning
    #   Wan2.1 VAE: apache-2.0 https://huggingface.co/Wan-AI/Wan2.1-T2V-14B
    ("video_wan2_2_14B_t2v", True, None, set()),
    # ACE-Step 1.5: mit https://huggingface.co/ACE-Step/Ace-Step1.5
    ("audio_ace_step_1_5_checkpoint", True, None, set()),
    # FLUX.1 [dev]: other / flux-1-dev-non-commercial-license https://huggingface.co/black-forest-labs/FLUX.1-dev
    ("flux_dev_full_text_to_image", False, "license", {"license"}),
    # SDXL base / refiner: openrail++(使い方の制限)
    #   https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
    #   https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0
    ("sdxl_simple_example", False, "license", {"license"}),
    # Hunyuan3D 2.0: other / tencent-hunyuan-community(地域・利用者数の条件)
    #   https://huggingface.co/tencent/Hunyuan3D-2mv
    ("3d_hunyuan3d_image_to_model", False, "license", {"license"}),
    # Stable Audio Open 1.0: other / stable-audio-community(売上の基準)
    #   https://huggingface.co/stabilityai/stable-audio-open-1.0
    ("audio_stable_audio_example", False, "license", {"license"}),
    # HiDream-I1 Full: 本体は mit https://huggingface.co/HiDream-ai/HiDream-I1-Full だが、
    #   テキストエンコーダ llama_3.1_8b_instruct は Llama 3.1 Community License
    #   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
    #   (#13 の修正前の一覧では再配布 Comfy-Org/HiDream-I1_ComfyUI の表記 mit だった。今は llama3.1)
    ("hidream_i1_full", False, "license", {"license"}),
    # MiniMax Music 3: 再配布 https://huggingface.co/Comfy-Org/MiniMax-Music-3 の表記は apache-2.0 だが、
    #   本家 https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/main/LICENSE は
    #   MiniMax-Music3 COMMUNITY LICENSE
    #   (商用の画面に名前の表示、年の売上 2,000 万ドル超は許可が要る、AUP)
    #   (#13 の修正前の一覧では再配布側の apache-2.0 だった。今は本家に記載が無いため null)
    ("audio_minimax_music_3", False, "license", {"license"}),
    # Qwen-Image 2512(apache-2.0 https://huggingface.co/Qwen/Qwen-Image-2512)と
    #   360 度の LoRA(mit https://huggingface.co/ProGamerGov/qwen-360-diffusion)。ライセンスは通るが
    #   テンプレートがカスタムノード(comfyui_essentials・seedvr2_videoupscaler)を要る
    ("templates_text_prompt_to_360hdr.app", False, "custom_nodes", {"custom_nodes"}),
    # HF の無検閲のモデル。カードの表記は mit https://huggingface.co/Heartsync/NSFW-Uncensored だが、
    #   構成(unet + text_encoder_2)は SDXL で base_model の記載が無い。SDXL の派生なら openrail++ のはずで、
    #   カードの表記は確かめられない(#13 の一覧は license=null・license_unverified)。
    #   ライセンスでも、公式テンプレートでないことでも、無検閲でも保留
    (
        "hf:Heartsync/NSFW-Uncensored",
        False,
        "license",
        {"license", "official_workflow", "custom_nodes", "comfyui_version", "content"},
    ),
]


@pytest.mark.parametrize("model_id,auto,first,failing", CASES, ids=[c[0] for c in CASES])
def test_real_model_gate(db, model_id, auto, first, failing):
    m = db.get(model_id)
    assert m is not None, f"一覧に無い: {model_id}"
    r = evaluate_offline(m, sources=db.sources)
    assert r.auto is auto, r.to_dict()
    assert (r.held_by[0] if r.held_by else None) == first, r.held_by
    # ライセンスの判定がカードと合っていることは必ず確かめる。それ以外の条件は、上に書いたものだけが落ちる
    assert set(r.held_by) == failing, r.to_dict()


def test_at_least_ten_real_models():
    assert len(CASES) >= 10
    assert sum(1 for c in CASES if c[1]) >= 4 and sum(1 for c in CASES if not c[1]) >= 4


def test_bundled_db_has_head_results_and_comfyui_version(db):
    assert db.sources.get("comfyui", {}).get("latest_release")
    hf = [f for m in db.models for f in m.files if f.url and "huggingface.co" in f.url]
    assert hf and sum(1 for f in hf if f.head_status is not None) / len(hf) > 0.95
