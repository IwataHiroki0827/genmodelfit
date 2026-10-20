"""PR #13 のレビューの指摘(ライセンス・一式のそろい具合・除外・一部の失敗・写し)のテスト。

ネットワークには触らない。HF の API は _Api(repo_id → cardData・tags)で置き換える。
ライセンスの値は 2026-10-20 に HF の API で確かめた実際のもの(hf.py の表のコメントに URL)。
"""

import json
import re
import threading
import time
from collections import Counter
from pathlib import Path

import pytest

from genmodelfit import safety
from genmodelfit.db import Database
from genmodelfit.fetch import DirFetcher, FetchError, HttpFetcher, RateLimitExhausted
from genmodelfit.hf import HfClient, RepoInfo, component_for, families_for, license_tier
from genmodelfit.models import ModelEntry, ModelFile, Template
from genmodelfit.update import (
    LICENSE_POLICY,
    Candidate,
    build_database,
    hf_entry,
    is_duplicate,
    set_completeness,
    set_entry_license,
    template_tasks,
)

FIX = Path(__file__).parent / "fixtures" / "update"


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


class _Api:
    """HF の API の代わり。repos に無いリポジトリは 401(非公開・削除)。"""

    def __init__(self, repos: dict[str, dict], files: dict[str, dict[str, int]] | None = None):
        self.repos = repos
        self.files = files or {}
        self.calls: Counter[str] = Counter()
        self._lock = threading.Lock()

    def get_json(self, url):
        m = re.match(r"https://huggingface\.co/api/models/(.+?)(\?|/revision/)", url)
        rid = m.group(1)
        with self._lock:
            self.calls[rid] += 1
        if rid not in self.repos:
            raise FetchError(url, "HTTP 401", 401)
        d = self.repos[rid]
        card = {k: v for k, v in d.items() if k in ("license", "license_name", "license_link", "base_model")}
        sib = [{"rfilename": p, "size": n} for p, n in self.files.get(rid, {}).items()]
        return {"id": rid, "cardData": card, "tags": d.get("tags", []), "siblings": sib}

    def get_text(self, url):
        raise FetchError(url, "HTTP 404", 404)

    def head_size(self, url):
        return None


# 実際の上流(2026-10-20 の HF の API の値)
UPSTREAM = {
    "Qwen/Qwen-Image-2.1": {"license": "other", "license_name": "qwen-research"},
    "Qwen/Qwen3-VL-8B-Instruct": {"license": "apache-2.0"},
    "black-forest-labs/FLUX.1-dev": {"license": "other", "license_name": "flux-1-dev-non-commercial-license"},
    "krea/Krea-2-Turbo": {
        "license": "other",
        "license_name": "krea-2-community-license",
        "base_model": ["krea/Krea-2-Raw"],
    },
    "krea/Krea-2-Raw": {"license": "other", "license_name": "krea-2-community-license"},
    "meta-llama/Llama-3.1-8B-Instruct": {"license": "llama3.1"},
    "HiDream-ai/HiDream-I1-Full": {"license": "mit"},
    "black-forest-labs/FLUX.2-dev": {"license": "other", "license_name": "flux-non-commercial-license"},
    "tencent/HunyuanVideo-1.5": {"license": "other", "license_name": "tencent-hunyuan-community"},
    "Wan-AI/Wan2.1-VACE-14B": {"license": "apache-2.0"},
    "Wan-AI/Wan2.1-VACE-1.3B": {"license": "apache-2.0"},
}


def _client(**repos):
    return HfClient(_Api({**UPSTREAM, **repos}))


def _resolve(client, repo_id, path, *, uncensored=False):
    return client.resolve_file(client.repo(repo_id), path, discovered_uncensored=uncensored)


# --- 1. HF で見つけたモデルの派生のライセンス ----------


def test_uploader_license_is_not_trusted_over_family():
    # 実例: AtomicChat は apache-2.0 と書き、base_model は Qwen3-VL(apache-2.0)だけ。
    # 名前は Qwen-Image-2.1(qwen-research)
    c = _client(
        **{
            "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF": {
                "license": "apache-2.0",
                "base_model": ["Qwen/Qwen3-VL-8B-Instruct"],
                "tags": ["base_model:finetune:Qwen/Qwen3-VL-8B-Instruct"],
            }
        }
    )
    r = _resolve(c, "AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF", "q.gguf", uncensored=True)
    assert r.license is None and "qwen-image-2.1" in r.evidence[0] and r.source == "Qwen/Qwen-Image-2.1"


def test_uploader_license_vs_base_chain():
    c = _client(
        **{
            # 実例: raidenzeke → aoxo/flux.1dev-abliterated(apache-2.0、base 無し)。名前で FLUX.1-dev
            "raidenzeke/flux.1dev-abliterated-gguf": {
                "license": "apache-2.0",
                "tags": ["base_model:quantized:aoxo/flux.1dev-abliterated"],
            },
            "aoxo/flux.1dev-abliterated": {"license": "apache-2.0"},
            # 実例: Quiho は apache-2.0 だが base_model は krea/Krea-2-Turbo(other)
            "Quiho/Krea2_NSFW_Uncensored_Complete": {
                "license": "apache-2.0",
                "base_model": "krea/Krea-2-Turbo",
            },
            # 実例: Heartsync は mit、base_model 無し(SDXL の diffusers の形)
            "Heartsync/NSFW-Uncensored": {"license": "mit"},
            # 元と同じライセンスなら、そのまま(道を記録する)
            "Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1": {
                "license": "other",
                "license_name": "qwen-research",
                "base_model": "Qwen/Qwen-Image-2.1",
            },
            # 3 段: A(apache)→ B(apache)→ FLUX.1-dev(非商用)
            "x/a": {"license": "apache-2.0", "base_model": "x/b"},
            "x/b": {"license": "apache-2.0", "base_model": "black-forest-labs/FLUX.1-dev"},
            # base_model が読めない(401)
            "x/orphan": {"license": "mit", "base_model": "x/private"},
        }
    )
    r = _resolve(c, "raidenzeke/flux.1dev-abliterated-gguf", "f-Q8_0.gguf", uncensored=True)
    assert r.license is None and "flux.1-dev" in r.evidence[0]
    assert r.path[:2] == ["raidenzeke/flux.1dev-abliterated-gguf", "aoxo/flux.1dev-abliterated"]

    r = _resolve(c, "Quiho/Krea2_NSFW_Uncensored_Complete", "k.safetensors", uncensored=True)
    assert r.license is None and r.source.startswith("krea/Krea-2")
    assert r.path == ["Quiho/Krea2_NSFW_Uncensored_Complete", "krea/Krea-2-Turbo", "krea/Krea-2-Raw"]

    r = _resolve(c, "Heartsync/NSFW-Uncensored", "unet/diffusion_pytorch_model.safetensors", uncensored=True)
    assert r.license is None and "base_model" in r.evidence[0]
    # 無検閲でない(読み上げなど)ものは、元が分からなくても本人の表記
    assert _resolve(c, "Heartsync/NSFW-Uncensored", "m.safetensors").license == "mit"

    r = _resolve(c, "Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1", "n.safetensors", uncensored=True)
    assert (r.license, r.license_name) == ("other", "qwen-research")
    assert r.path == ["Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1", "Qwen/Qwen-Image-2.1"]

    r = _resolve(c, "x/a", "m.safetensors")
    assert r.license is None and r.path == ["x/a", "x/b", "black-forest-labs/FLUX.1-dev"]
    r = _resolve(c, "x/orphan", "m.safetensors")
    assert r.license is None and "x/private" in r.evidence[0] and "401" in r.evidence[0]


def test_base_model_from_tags_and_lists():
    info = RepoInfo.from_api(
        {
            "id": "o/r",
            "cardData": {"license": "mit", "base_model": ["a/one", "b/two"]},
            "tags": ["base_model:quantized:c/three", "base_model:a/one", "base_model:adapter:d/four"],
        }
    )
    assert info.base_models == ["a/one", "b/two", "c/three", "d/four"]


def test_families_table_matches_names():
    def keys(*names):
        return {f.key for f in families_for(names)}

    assert "flux.1-dev" in keys("flux1-dev-kontext_fp8_scaled.safetensors")
    assert "flux.1-dev" in keys("flux.1dev-abliterated")
    assert "flux.1-dev" not in keys("flux1-schnell.safetensors") and "flux.1-dev" not in keys("flux2_dev")
    assert "flux.2-dev" in keys("flux2_dev_fp8mixed.safetensors")
    assert "flux.2-klein-9b" in keys("PornMaster_Klein-9b")
    assert keys("Qwen-Image-2.1-Turbo-Uncensored") >= {"qwen-image-2.1", "qwen-image"}
    assert "qwen-image-2.1" not in keys("qwen_image_2512_bf16.safetensors")
    assert "sdxl" in keys("sd_xl_base_1.0.safetensors") and "sdxl" not in keys("acestep_v1.5_xl_base_bf16")
    assert "ltx-2" in keys("ltx-2.3-22b-dev.safetensors") and "ltx-2" not in keys(
        "ltx-video-2b-v0.9.safetensors"
    )
    assert "wan" in keys("wan2.2_t2v_low_noise_14B") and "krea-2" in keys("krea2_turbo_fp8_scaled")


# --- 2. 一式のライセンスはすべてのファイルで ----------


def _f(name, role, lic, lname=None, repo="o/r"):
    return ModelFile(name=name, role=role, license=lic, license_name=lname, repo_id=repo, license_from=repo)


def test_entry_license_includes_loras_and_upscalers():
    e = ModelEntry(id="templates_mjm_airt_machIne", name="x", modality="image")
    e.files = [
        _f("main.safetensors", "main", "apache-2.0"),
        _f("4x-UltraSharp.safetensors", "other", "cc-by-nc-sa-4.0", repo="Kim2091/UltraSharp"),
        _f("clip_vision_h.safetensors", "clip_vision", "mit"),
    ]
    set_entry_license(e)
    assert e.license == "cc-by-nc-sa-4.0"
    e.files.append(_f("relight_lora.safetensors", "lora", None))
    set_entry_license(e)
    assert e.license is None
    e2 = ModelEntry(id="utility_video_upscale", name="x", modality="video")
    set_entry_license(e2)
    assert e2.license is None and e2.license_evidence


# --- 3・8. 再配布の中の部品(テキストエンコーダ・VAE)は本家の表で ----------


def test_components_in_repackaged_repos():
    c = _client(
        **{
            "Comfy-Org/HiDream-I1_ComfyUI": {"license": "mit", "base_model": ["HiDream-ai/HiDream-I1-Full"]},
            "Comfy-Org/flux2-dev": {
                "license": "other",
                "license_name": "flux-1-dev-non-commercial-license",
                "base_model": ["black-forest-labs/FLUX.2-dev"],
            },
            "Comfy-Org/HunyuanVideo_1.5_repackaged": {
                "license": "other",
                "license_name": "tencent-hunyuan-community",
                "base_model": "tencent/HunyuanVideo-1.5",
            },
            "Comfy-Org/Wan_2.1_ComfyUI_repackaged": {
                "license": "apache-2.0",
                "base_model": ["htdong/Wan-Alpha"],
            },
        }
    )
    r = _resolve(
        c,
        "Comfy-Org/HiDream-I1_ComfyUI",
        "split_files/text_encoders/llama_3.1_8b_instruct_fp8_scaled.safetensors",
    )
    assert (r.license, r.source) == ("llama3.1", "meta-llama/Llama-3.1-8B-Instruct")
    assert r.evidence == ["component:llama-3.1-8b-instruct"]
    assert _resolve(c, "Comfy-Org/HiDream-I1_ComfyUI", "hidream_i1_full_fp8.safetensors").license == "mit"
    r = _resolve(c, "Comfy-Org/flux2-dev", "split_files/vae/flux2-vae.safetensors")
    assert (r.license, r.source) == ("apache-2.0", "black-forest-labs/FLUX.2-VAE")
    r = _resolve(c, "Comfy-Org/flux2-dev", "flux2_dev_fp8mixed.safetensors")
    assert (r.license, r.license_name) == ("other", "flux-non-commercial-license")
    r = _resolve(c, "Comfy-Org/HunyuanVideo_1.5_repackaged", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    assert (r.license, r.source) == ("apache-2.0", "Qwen/Qwen2.5-VL-7B-Instruct")
    r = _resolve(c, "Comfy-Org/Wan_2.1_ComfyUI_repackaged", "split_files/vae/wan_2.1_vae.safetensors")
    assert r.source == "Wan-AI/Wan2.1-T2V-14B" and r.url == "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B"
    # OmniGen2 の Qwen2.5-VL-3B は qwen-research(再配布側の apache-2.0 より厳しい)
    assert component_for("qwen_2.5_vl_fp16.safetensors").license_name == "qwen-research"
    # 部品の表は再配布のリポジトリの中だけ(ほかの持ち主の同名のファイルは改変されていることがある)
    c2 = _client(**{"someone/flux-thing": {"license": "mit"}})
    assert _resolve(c2, "someone/flux-thing", "flux2-vae.safetensors").source == "someone/flux-thing"


# --- 4. 本家が読めない再配布は不明 ----------


def test_unreadable_upstream_is_unknown():
    c = _client(
        **{
            "Comfy-Org/Mage-Flow": {"license": "mit", "base_model": ["microsoft/Mage-Flow"]},
            "Comfy-Org/Wan-Dancer": {"license": "apache-2.0", "base_model": ["Wan-AI/Wan2.2-Dancer-14B"]},
            "Comfy-Org/Wan-Animate-2": {"license": "apache-2.0"},
            "Kijai/WanVideo_comfy": {"base_model": ["Wan-AI/Wan2.1-VACE-14B", "Wan-AI/Wan2.1-VACE-1.3B"]},
        }
    )
    for repo, path in [
        ("Comfy-Org/Mage-Flow", "mage_flow_int8_convrot.safetensors"),
        ("Comfy-Org/Wan-Dancer", "wan2.2_dancer_14b_global_fp8_scaled.safetensors"),
        ("Comfy-Org/Wan-Animate-2", "wan_animate_2_int8_convrot.safetensors"),  # base_model が無い再配布
        ("Kijai/WanVideo_comfy", "Wan21_CausVid_14B_T2V_lora_rank32.safetensors"),  # 本家を選べない
    ]:
        r = _resolve(c, repo, path)
        assert r.license is None and r.evidence[0].startswith("license_unverified"), repo
    assert "401" in _resolve(c, "Comfy-Org/Mage-Flow", "mage_flow_int8_convrot.safetensors").evidence[0]
    # 部品の表に当たるものは、本家が読めなくても表の本家
    assert (
        _resolve(c, "Comfy-Org/Wan-Animate-2", "umt5_xxl_fp8_e4m3fn_scaled.safetensors").license
        == "apache-2.0"
    )
    assert (
        _resolve(c, "Kijai/WanVideo_comfy", "Wan2_1-VACE_module_14B_bf16.safetensors").license == "apache-2.0"
    )


def test_tiers_order():
    order = ["mit", "cc-by-4.0", "openrail++", "other", "cc-by-nc-4.0", None]
    assert [license_tier(x) for x in order] == sorted(license_tier(x) for x in order)
    assert license_tier("other", "qwen-research") > license_tier("other", "stable-audio-community")
    assert license_tier("other", "sai-nc-community") == license_tier("cc-by-nc-sa-4.0")
    assert license_tier("gemma") == license_tier("llama3.1") == license_tier("openrail++")


# --- 6・12. 一式がそろっていないもの ----------


def test_files_complete_and_size_floor():
    gb = 1_000_000_000
    t = Template(name="t", size_bytes=int(10.7 * gb))
    e = ModelEntry(id="templates-wan2_1_infinitetalk_music", name="x", modality="video", template=t)
    e.files = [ModelFile(name="clip_vision_h.safetensors", role="clip_vision", size_bytes=int(1.26 * gb))]
    set_completeness(e)
    assert e.files_complete is False and e.size_floor_bytes == t.size_bytes
    assert e.estimate_size_bytes == t.size_bytes and e.total_size_bytes == int(1.26 * gb)
    # 本体はあるが合計が index の半分未満
    e.files = [ModelFile(name="m.safetensors", role="main", size_bytes=4 * gb)]
    set_completeness(e)
    assert e.files_complete is False
    e.files = [ModelFile(name="m.safetensors", role="main", size_bytes=10 * gb)]
    set_completeness(e)
    assert e.files_complete is True and e.size_floor_bytes is None
    # ファイルが 1 つも無い
    z = ModelEntry(
        id="utility_video_upscale",
        name="x",
        modality="video",
        template=Template(name="u", size_bytes=46 * gb),
    )
    set_completeness(z)
    assert z.files_complete is False and z.size_floor_bytes == 46 * gb
    # HF の GGUF の本体だけ(テキストエンコーダ・VAE が無い)
    g = ModelEntry(
        id="hf:thepatch/stable-audio-3-small-sfx-GGUF", name="g", modality="sfx", source="huggingface"
    )
    g.files = [ModelFile(name="m-Q8_0.gguf", role="main", size_bytes=100_000_000)]
    set_completeness(g)
    assert g.files_complete is False and g.size_floor_bytes is None
    g.files.append(ModelFile(name="text_encoders/t5.gguf", role="text_encoder", size_bytes=1))
    set_completeness(g)
    assert g.files_complete is True


def test_estimate_does_not_look_small_when_incomplete():
    from genmodelfit.estimate import estimate
    from genmodelfit.models import Gpu, SystemInfo

    gb = 1_000_000_000
    pc = SystemInfo(
        os="x",
        cpu="x",
        cpu_cores=8,
        ram_total_gb=64,
        ram_available_gb=60,
        gpus=[Gpu(name="g", vendor="nvidia", vram_total_gb=8, backend="cuda")],
    )
    e = ModelEntry(id="x", name="x", modality="video", template=Template(name="x", size_bytes=40 * gb))
    e.files = [ModelFile(name="c.safetensors", role="clip_vision", size_bytes=1 * gb)]
    small = estimate(e, pc)
    set_completeness(e)
    big = estimate(e, pc)
    assert big.vram_gb > small.vram_gb + 30 and big.disk_gb >= 40


# --- 7. 除外 ----------


def test_negation_only_covers_the_governed_term():
    unc = {"uncensored": True}
    assert safety.blocked_terms([], "Uncensored, no filters: generates nude teen girls.", **unc) == ["teen"]
    assert safety.blocked_terms([], "No filters:\n- nude teen girls", **unc) == ["teen"]
    assert safety.blocked_terms([], "No limits loli art.", **unc) == ["loli"]
    assert safety.blocked_terms([], "Do not generate sexual content involving minors.", **unc) == []
    assert safety.blocked_terms([], "Sexual content involving minors is strictly prohibited.", **unc) == []
    assert (
        safety.blocked_terms([], "We never depict teens; sexual content features adults only.", **unc) == []
    )
    assert safety.blocked_terms([], "No loli content.", **unc) == []
    # 2026-10-20 の実データで誤って外していた形(成人向けモデルの注意書き・安全対策の説明)
    card = (
        "Do not use this model to generate illegal content, non-consensual intimate imagery, "
        "or sexual content involving minors."
    )
    assert safety.blocked_terms([], card, **unc) == []
    card = "We filtered pre-training data for NSFW content and known child sexual abuse material (CSAM)."
    assert safety.blocked_terms([], card, **unc) == []
    # 禁止の語が生成を受けていなければ、列挙の先まで効かせない
    assert safety.blocked_terms([], "Do not worry, it makes nude teen pictures.", **unc) == ["teen"]


def test_new_exclusion_terms():
    unc = {"uncensored": True}
    assert safety.blocked_terms(["lolita-style-xl"], uncensored=False) == ["lolita"]
    assert safety.blocked_terms(["shotacon_v1"], uncensored=False) == ["shotacon"]
    assert safety.blocked_terms(["schoolgirl-nsfw"], **unc) == ["schoolgirl"]
    assert safety.blocked_terms([], "A sexy school girl outfit.", **unc) == ["school girl"]
    assert safety.blocked_terms(["jk-nsfw"], **unc) == ["jk"]
    assert safety.blocked_terms(["jk-nsfw"], uncensored=False) == []
    assert safety.blocked_terms(["transformers.js"], **unc) == []  # js は大文字の JS だけ
    assert safety.blocked_terms([], "nsfw JS model", **unc) == ["js"]
    assert safety.blocked_terms([], "ロリ系のイラスト", uncensored=False) == ["ロリ"]
    assert safety.blocked_terms([], "ショタコン向け", uncensored=False) == ["ショタコン"]
    assert safety.blocked_terms([], "カロリーとロリポップ", uncensored=False) == []
    assert safety.blocked_terms([], "エロい女子中学生", **unc) == ["女子中学生"]
    assert safety.blocked_terms([], "エロい JC の画像", **unc) == ["jc"]
    assert safety.blocked_terms([], "児童の性的な描写は禁止です。", **unc) == []
    assert safety.blocked_terms([], "児童向けの絵本のイラスト", **unc) == []


def test_legit_adult_models_stay(tmp_path):
    db = build_database(DirFetcher(FIX), head=False, carry_over=False)
    ids = {e.id for e in db.models}
    assert {
        "hf:Noctaluna/Noct-Q-Uncensored-Qwen-Image-2.1",
        "hf:AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF",
    } <= ids
    assert db.sources["excluded"] == 2


# --- 9. 一部が取れなかったとき ----------


def test_partial_carries_over_only_from_same_policy():
    old = ModelEntry(id="image_missing_json_example", name="old", modality="image", license="mit")
    prev = Database(models=[old], sources={"license_policy": LICENSE_POLICY})
    db = build_database(DirFetcher(FIX), head=False, previous=prev)
    assert db.sources["partial"] is True and db.sources["partial_counts"]["carried_over"] == 1
    assert db.get("image_missing_json_example").name == "old"
    stale = Database(models=[old], sources={"license_policy": 1})
    db = build_database(DirFetcher(FIX), head=False, previous=stale)
    assert db.get("image_missing_json_example") is None and db.sources["partial_counts"]["carried_over"] == 0


class _Resp:
    def __init__(self, status, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body

    def json(self):
        return self._body


def test_rate_limit_budget():
    now = [0.0]

    def sleep(s):
        now[0] += s

    f = HttpFetcher(sleep=sleep, clock=lambda: now[0], rate_limit_budget=100)

    class S:
        def request(self, *a, **k):
            return _Resp(429, {"Retry-After": "60"})

    f._local.session = S()
    with pytest.raises(RateLimitExhausted):
        f.get_json("https://huggingface.co/api/models/x")
    assert f.budget_exhausted and f.rate_limit_waited > 100
    with pytest.raises(RateLimitExhausted):  # 使い切ったら、すぐ止める(さらに待たない)
        f.get_json("https://huggingface.co/api/models/y")


def test_transient_failures_mark_partial(tmp_path):
    class Flaky(DirFetcher):
        def get_json(self, url):
            if "/api/models/Tongyi-MAI" in url:
                raise FetchError(url, "HTTP 503", 503)
            return super().get_json(url)

    db = build_database(Flaky(FIX), head=False, carry_over=False)
    pc = db.sources["partial_counts"]
    assert db.sources["partial"] and pc["hf_repos_failed_transient"] == 1 and pc["entries_affected"] >= 1


# --- 10. 道具の分類の task ----------


def test_tool_categories_default_to_tool():
    assert template_tasks({"tags": ["Depth"]}, "image", "Image Tools") == ["tool"]
    assert template_tasks({"tags": ["Pose Map"]}, "video", "Video Tools") == ["tool"]
    assert template_tasks({"tags": ["Image Upscale"]}, "image", "Image Tools") == ["upscale"]
    assert template_tasks({"tags": []}, "image", "Image") == ["t2i"]


# --- 11. 写しの重複 ----------


def test_duplicate_weights():
    def e(rid, sha, size, base):
        f = ModelFile(
            name="m", role="main", sha256=sha, size_bytes=size, repo_id=rid, license_path=[rid, base]
        )
        return ModelEntry(id=f"hf:{rid}", name=rid, modality="video", files=[f], source="huggingface")

    a = e("Runware/x", "aa", 58_900_000_000, "10Eros/base")
    assert is_duplicate(e("TenStrip/y", "aa", 58_900_000_000, "other/base"), [a])  # sha256 が同じ
    assert is_duplicate(e("z/q", "bb", 59_200_000_000, "10Eros/base"), [a])  # 同じ元・±1%
    assert not is_duplicate(e("z/q", "bb", 30_000_000_000, "10Eros/base"), [a])  # 別の量子化
    assert not is_duplicate(e("z/q", "bb", 58_900_000_000, "another/base"), [a])
    # 元の記載が無くても、名前の系統(LTX-2)と大きさで写しとみなす(10Eros の v1.4 と v1.5 の 58.9 GB)
    r = e("Runware/LTX2.3-10Eros_1.4", "cc", 58_918_452_976, "Runware/LTX2.3-10Eros_1.4")
    t = e("TenStrip/LTX2.3-10Eros", "dd", 58_918_453_144, "TenStrip/LTX2.3-10Eros")
    assert is_duplicate(t, [r])


# --- 12. HF の 1 件(GGUF だけ) ----------


def test_hf_entry_gguf_only_is_incomplete():
    api = _Api(
        {
            "thepatch/stable-audio-3-small-sfx-GGUF": {
                "license": "other",
                "license_name": "stable-audio-community",
                "base_model": "stabilityai/stable-audio-3-small-sfx",
            },
            "stabilityai/stable-audio-3-small-sfx": {
                "license": "other",
                "license_name": "stable-audio-community",
            },
        },
        files={"thepatch/stable-audio-3-small-sfx-GGUF": {"sa3-small-sfx-Q8_0.gguf": 120_000_000}},
    )
    c = HfClient(api)
    info = c.repo("thepatch/stable-audio-3-small-sfx-GGUF")
    cand = Candidate("thepatch/stable-audio-3-small-sfx-GGUF", "sfx", "t2a", 500, 1, [], False)
    ent = hf_entry(cand, info, c)
    assert ent.files_complete is False and ent.license_name == "stable-audio-community"
    assert ent.files[0].license_path == [
        "thepatch/stable-audio-3-small-sfx-GGUF",
        "stabilityai/stable-audio-3-small-sfx",
    ]


# --- 13. 同じリポジトリを同時に頼まれても 1 回だけ取る ----------


def test_repo_is_fetched_once_under_concurrency():
    class Slow(_Api):
        def get_json(self, url):
            time.sleep(0.05)
            return super().get_json(url)

    api = Slow({"o/r": {"license": "mit"}})
    c = HfClient(api)
    threads = [threading.Thread(target=c.repo, args=("o/r",)) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert api.calls["o/r"] == 1
    c.prefetch([("o/r", "main"), ("o/missing", "main")])
    assert api.calls["o/r"] == 1 and api.calls["o/missing"] == 1


def test_db_json_round_trip_keeps_new_fields():
    e = ModelEntry(
        id="x", name="x", modality="image", files_complete=False, size_floor_bytes=5, license_name="n"
    )
    e.files = [ModelFile(name="f", license_path=["a", "b"], license_evidence=["family:wan"], license_url="u")]
    back = ModelEntry.from_dict(json.loads(json.dumps(e.to_dict())))
    assert (
        back.files_complete is False
        and back.size_floor_bytes == 5
        and back.files[0].license_path == ["a", "b"]
    )
