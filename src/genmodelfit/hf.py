# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Hugging Face Hub API から、ライセンス・利用申請(gated)・ファイルの大きさ・sha256 を取る。

- 1 リポジトリにつき `GET /api/models/{repo_id}?blobs=true` を 1 回だけ呼ぶ(結果はこの実行の中で使い回す)
- ライセンスは「寛容の側に間違えない」ことを最優先にする(使う側はソース非公開の商用アプリ)。
  分からないとき・食い違うときは不明(None)にする。規則は docs/sources.md の「ライセンス」:
  1. 部品の表(COMPONENTS。テキストエンコーダ・VAE など、ファイル名で本家が決まるもの)
  2. 再配布(Comfy-Org など)は、ファイルに対応する本家を base_model から選ぶ(選べなければ不明)
  3. base_model を N 段までたどり、上流の一番厳しいものより寛容な表記は不明にする
  4. 系統の表(FAMILIES。名前で分かる元のモデル)より寛容な表記も不明にする
- 一式のライセンスは「最も厳しいもの」(license_tier の段階)
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urlencode, urlsplit

from .fetch import Fetcher, FetchError, NetworkError

HF = "https://huggingface.co"

# 寛容なライセンス(商用で使え、配布物のソースを公開しなくてよく、表示の義務も軽いもの)。
# HF の license タグの表記
PERMISSIVE = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "isc",
    "zlib",
    "unlicense",
    "cc0-1.0",
    "pddl",
    "bsl-1.0",
    "afl-3.0",
    "ecl-2.0",
    "postgresql",
    "wtfpl",
}
# 表示(クレジット)の義務だけがあるもの
ATTRIBUTION = {"cc-by-2.0", "cc-by-2.5", "cc-by-3.0", "cc-by-4.0", "odc-by"}

# 再配布(リパッケージ)をしている持ち主。ここのリポジトリは base_model をたどって本家のライセンスを見る
REPACKAGERS = ("Comfy-Org", "comfyanonymous", "Kijai")

# base_model を書いていない再配布リポジトリの本家(上流を確かめて手で書いたもの。docs/sources.md に同じ表)
REPACKAGED_UPSTREAM: dict[str, list[str]] = {
    # OpenCLIP ViT-bigG/14(SDXL の clip_vision)。HF の laion のリポジトリは license: mit
    "comfyanonymous/clip_vision_g": ["laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"],
}

_BASE_KINDS = {"finetune", "quantized", "adapter", "merge"}

WEIGHT_EXT = (".safetensors", ".gguf", ".ckpt", ".bin", ".pth", ".pt", ".sft")

# 名前の比較で無視する語(精度・形式などモデルの種類を表さないもの)
_GENERIC_TOKENS = {
    "safetensors",
    "gguf",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "fp4",
    "nvfp4",
    "int8",
    "int4",
    "e4m3fn",
    "e5m2",
    "scaled",
    "mixed",
    "comfy",
    "comfyui",
    "repackaged",
    "split",
    "files",
    "model",
    "models",
    "diffusion",
    "org",
    "ai",
    "convrot",
    "the",
}


# --- URL ---------------------------------------------------------------------------------------


def parse_hf_url(url: str | None) -> tuple[str, str, str] | None:
    """`https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>` → (repo_id, revision, path)。"""
    if not url:
        return None
    u = urlsplit(url)
    if u.netloc not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = u.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] not in ("resolve", "blob", "raw"):
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    return repo_id, unquote(parts[3]), unquote("/".join(parts[4:]))


def api_url(repo_id: str, revision: str = "main") -> str:
    if revision in ("main", ""):
        return f"{HF}/api/models/{repo_id}?blobs=true"
    return f"{HF}/api/models/{repo_id}/revision/{quote(revision, safe='')}?blobs=true"


def readme_url(repo_id: str) -> str:
    return f"{HF}/{repo_id}/raw/main/README.md"


def search_url(**params: Any) -> str:
    q = {"sort": "downloads", "direction": "-1", "limit": "100"}
    q.update({k: str(v) for k, v in params.items() if v is not None})
    return f"{HF}/api/models?{urlencode(sorted(q.items()))}"


# --- ライセンス ------------------------------------------------------------------------------


def normalize_license(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().lower()
    return None if v in ("unknown", "none", "n/a", "unlicensed") else v


# ライセンスの段階(大きいほど厳しい)。docs/sources.md に同じ表
TIER_UNKNOWN = 6  # 不明(None)
TIER_NONCOMMERCIAL = 5  # 非商用・研究用(cc-by-nc*・*non-commercial*・qwen-research・flux-1-dev など)
TIER_OTHER = (
    4  # 独自・利用者数などの条件付き(other・tencent-hunyuan-community・gpl など、下に当たらないもの全部)
)
TIER_USE_RESTRICTED = 3  # 用途の制限付き(openrail 系・llama・gemma)
TIER_ATTRIBUTION = 2  # 表示の義務(cc-by-4.0 など)
TIER_PERMISSIVE = 1  # 寛容(apache-2.0・mit・bsd など)

_NONCOMMERCIAL = re.compile(
    r"(?<![a-z0-9])nc(?![a-z0-9])|non[-_ ]?commercial|research|flux-1-dev|nscl", re.IGNORECASE
)
_USE_RESTRICTED = re.compile(
    r"^(openrail[\w+.-]*|creativeml-openrail-m|bigscience[\w.-]*|llama[\d.]*|gemma)$"
)


def license_tier(lic: str | None, name: str | None = None) -> int:
    """ライセンスの段階(TIER_*)。other は license_name(独自ライセンスの名前)も見る。"""
    if lic is None:
        return TIER_UNKNOWN
    text = f"{lic} {name or ''}".lower()
    if _NONCOMMERCIAL.search(text):
        return TIER_NONCOMMERCIAL
    if lic in PERMISSIVE:
        return TIER_PERMISSIVE
    if lic in ATTRIBUTION:
        return TIER_ATTRIBUTION
    if _USE_RESTRICTED.match(lic):
        return TIER_USE_RESTRICTED
    return TIER_OTHER


def license_rank(lic: str | None, name: str | None = None) -> int:
    """license_tier と同じ(大きいほど厳しい)。"""
    return license_tier(lic, name)


LicenseLike = str | None | tuple[str | None, str | None]


def _pair(x: LicenseLike) -> tuple[str | None, str | None]:
    return x if isinstance(x, tuple) else (x, None)


def strictest(licenses: Iterable[LicenseLike]) -> Any:
    """一番厳しいもの。要素は license か (license, license_name)。空なら None(不明)。"""
    items = list(licenses)
    if not items:
        return None
    return max(items, key=lambda x: (license_tier(*_pair(x)), str(_pair(x)[0] or ""), str(_pair(x)[1] or "")))


@dataclass
class RepoInfo:
    repo_id: str
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None
    gated: bool | None = None
    base_models: list[str] = field(default_factory=list)
    files: dict[str, tuple[int | None, str | None]] = field(default_factory=dict)  # path → (size, sha256)
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    last_modified: str | None = None
    card: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RepoInfo:
        card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
        lic = normalize_license(card.get("license"))
        if lic is None:  # cardData に無ければタグ(license:xxx)を見る
            for t in data.get("tags") or []:
                if isinstance(t, str) and t.startswith("license:"):
                    lic = normalize_license(t.split(":", 1)[1])
                    break
        base = card.get("base_model")
        bases = [base] if isinstance(base, str) else [b for b in (base or []) if isinstance(b, str)]
        # タグの base_model:<repo> / base_model:finetune:<repo> / base_model:quantized:<repo> なども見る
        for t in data.get("tags") or []:
            if isinstance(t, str) and t.startswith("base_model:"):
                rest = t.split(":", 1)[1]
                kind, _, repo = rest.partition(":")
                b = repo if repo and kind in _BASE_KINDS else rest
                if "/" in b and b not in bases:
                    bases.append(b)
        gated = data.get("gated")
        files: dict[str, tuple[int | None, str | None]] = {}
        for s in data.get("siblings") or []:
            if not isinstance(s, dict) or not s.get("rfilename"):
                continue
            lfs = s.get("lfs") if isinstance(s.get("lfs"), dict) else {}
            files[s["rfilename"]] = (s.get("size") or lfs.get("size"), lfs.get("sha256"))
        ln = card.get("license_name")
        link = card.get("license_link")
        return cls(
            repo_id=data.get("id") or data.get("modelId") or "",
            license=lic,
            license_name=ln if isinstance(ln, str) else None,
            license_link=link if isinstance(link, str) else None,
            gated=None if gated is None else bool(gated),
            base_models=[b.strip() for b in bases if b.strip()],
            files=files,
            downloads=data.get("downloads"),
            likes=data.get("likes"),
            tags=[t for t in data.get("tags") or [] if isinstance(t, str)],
            pipeline_tag=data.get("pipeline_tag"),
            library_name=data.get("library_name"),
            last_modified=data.get("lastModified"),
            card=card,
        )

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo_id}"

    def license_url(self) -> str:
        link = self.license_link
        if link and link.startswith(("http://", "https://")):
            return link
        if link:
            return f"{HF}/{self.repo_id}/blob/main/{link.lstrip('./')}"
        return self.url


def _tokens(text: str) -> set[str]:
    toks = re.split(r"[^a-z0-9]+", text.lower())
    return {
        t for t in toks if t and t not in _GENERIC_TOKENS and not t.isdigit() and not _SIZE_TOKEN.match(t)
    }


# 大きさ・版だけを表す語(14b・720p・v2 など)は、どの本家かの手がかりにしない
_SIZE_TOKEN = re.compile(r"^(\d+(\.\d+)?[bmkp]|v\d+|rank\d+)$")


def choose_bases(bases: list[str], filename: str) -> list[str]:
    """再配布のファイルが、どの本家から来たか。1 つなら確定。複数なら名前の重なりで選ぶ(同点は全部)。
    どれとも重ならなければ空(本家が分からない = ライセンスは不明)。大きさの語(14b など)は数えない。"""
    if len(bases) <= 1:
        return list(bases)
    ft = _tokens(filename.rsplit("/", 1)[-1])
    scored = [(len(ft & _tokens(b.split("/", 1)[-1])), b) for b in bases]
    best = max(s for s, _ in scored)
    return [b for s, b in scored if s == best] if best > 0 else []


def is_repackager(repo_id: str) -> bool:
    return repo_id.split("/", 1)[0] in REPACKAGERS


def upstream_candidates(info: RepoInfo) -> list[str]:
    return info.base_models or REPACKAGED_UPSTREAM.get(info.repo_id, [])


# --- ライセンスの表 ------------------------------------------------------------------------------


@dataclass(frozen=True)
class KnownLicense:
    """手で確かめたライセンス(2026-10-20 に HF の API・モデルカードで確認。コメントの URL が確かめた場所)。"""

    key: str
    pattern: re.Pattern[str]
    license: str
    license_name: str | None
    source: str  # license_from に入れる名前(HF のリポジトリ。HF に記載が無いものは GitHub のリポジトリ)
    url: str
    gated: bool = False  # 本家が利用申請(規約への同意)を求める(置き場所が gated でなくても保留にする)

    @property
    def everywhere(self) -> bool:
        """どの持ち主のリポジトリの中にあっても当てる行(寛容でない・申請が要る)。改変しても規約は軽くならない。"""
        return self.gated or license_tier(self.license, self.license_name) > TIER_PERMISSIVE


def _k(
    key: str,
    pat: str,
    lic: str,
    name: str | None,
    source: str,
    url: str | None = None,
    *,
    gated: bool = False,
) -> KnownLicense:
    return KnownLicense(
        key, re.compile(pat, re.IGNORECASE), lic, name, source, url or f"{HF}/{source}", gated
    )


# 部品の表: 再配布(Comfy-Org など)の中の、ファイル名で本家が決まる部品(テキストエンコーダ・VAE など)。
# 再配布のリポジトリの表記や base_model は本体のものなので、部品には当てない(Llama 3.1 が mit になるなど)。
# 載せるのは、本家のライセンスを確かめ、複数の再配布で sha256 が同じ(=手を加えていない写し)と確かめたもの。
# 寛容な行を当てるのは再配布のリポジトリの中のファイルだけ(ほかの持ち主の同名のファイルは改変されている
# ことがある)。寛容でない行・本家が申請を求める行(everywhere)は、どのリポジトリの中にあっても当てる
# (DINOv3 が VAST-AI/TripoSplat の中で mit になっていた。#14 のレビュー)。
COMPONENTS: tuple[KnownLicense, ...] = (
    # https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (license: llama3.1)
    _k(
        "llama-3.1-8b-instruct",
        r"^llama_3\.1_8b_instruct\w*\.safetensors$",
        "llama3.1",
        None,
        "meta-llama/Llama-3.1-8B-Instruct",
    ),
    # https://huggingface.co/google/t5-v1_1-xxl (apache-2.0)
    _k(
        "t5-v1_1-xxl",
        r"^t5xxl_(fp16|bf16|fp32|fp8_e4m3fn|fp8_e4m3fn_scaled)\.safetensors$",
        "apache-2.0",
        None,
        "google/t5-v1_1-xxl",
    ),
    # https://huggingface.co/google/umt5-xxl (apache-2.0)。Kijai の umt5-xxl-enc はエンコーダだけを抜いたもの
    _k(
        "umt5-xxl",
        r"^umt5[_-]xxl[_-](fp16|bf16|fp32|fp8_e4m3fn|fp8_e4m3fn_scaled|enc[_-]bf16|enc[_-]fp8_e4m3fn)\.safetensors$",
        "apache-2.0",
        None,
        "google/umt5-xxl",
    ),
    # OpenAI CLIP ViT-L/14。HF の openai/clip-vit-large-patch14 には license の記載が無いので、
    # GitHub の LICENSE(MIT)
    _k(
        "clip-vit-large-patch14",
        r"^clip_l\.safetensors$",
        "mit",
        None,
        "openai/clip-vit-large-patch14",
        "https://github.com/openai/CLIP/blob/main/LICENSE",
    ),
    # https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k (mit)
    _k(
        "clip-vit-bigg-14",
        r"^(clip_g|clip_vision_g)\.safetensors$",
        "mit",
        None,
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ),
    # https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K (mit)
    _k(
        "clip-vit-h-14",
        r"^(clip_vision_h|clip-vit-h-14-laion2b-s32b-b79k)\.safetensors$",
        "mit",
        None,
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
    # https://huggingface.co/google/siglip-so400m-patch14-384 (apache-2.0)
    _k(
        "siglip-so400m",
        r"^sigclip_vision_patch14_384\.safetensors$",
        "apache-2.0",
        None,
        "google/siglip-so400m-patch14-384",
    ),
    # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct (apache-2.0)。
    # Qwen-Image と HunyuanVideo 1.5 で sha256 が同じ
    _k(
        "qwen2.5-vl-7b",
        r"^qwen_2\.5_vl_7b_fp8_scaled\.safetensors$",
        "apache-2.0",
        None,
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ),
    # OmniGen2 の Qwen2.5-VL-3B(fp16 で 7.5 GB)。
    # https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct (other: qwen-research)
    _k(
        "qwen2.5-vl-3b",
        r"^qwen_2\.5_vl_fp16\.safetensors$",
        "other",
        "qwen-research",
        "Qwen/Qwen2.5-VL-3B-Instruct",
    ),
    # https://huggingface.co/Qwen/Qwen3-4B (apache-2.0)。FLUX.2 klein 4B と Z-Image で sha256 が同じ
    _k("qwen3-4b", r"^qwen_3_4b(_fp8_mixed)?\.safetensors$", "apache-2.0", None, "Qwen/Qwen3-4B"),
    # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct (apache-2.0)。Comfy-Org/Qwen3-VL と sha256 が同じもの
    _k(
        "qwen3-vl-8b",
        r"^qwen3vl_8b_fp8_scaled\.safetensors$",
        "apache-2.0",
        None,
        "Qwen/Qwen3-VL-8B-Instruct",
    ),
    # https://huggingface.co/google/gemma-3-12b-it (license: gemma)
    _k(
        "gemma-3-12b-it",
        r"^gemma_3_12b_it(_fp8_scaled|_fp4_mixed|_bf16)?\.safetensors$",
        "gemma",
        None,
        "google/gemma-3-12b-it",
    ),
    # https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english (apache-2.0。Wan2.2-S2V が使うもの)
    _k(
        "wav2vec2-large-xlsr-53-english",
        r"^wav2vec2_large_english(_fp16)?\.safetensors$",
        "apache-2.0",
        None,
        "jonatasgrosman/wav2vec2-large-xlsr-53-english",
    ),
    # FLUX.1 の VAE。https://huggingface.co/black-forest-labs/FLUX.1-schnell (apache-2.0)。
    # HiDream・Lumina 2・Z-Image の ae.safetensors は sha256 が同じ
    _k("flux1-vae", r"^ae\.safetensors$", "apache-2.0", None, "black-forest-labs/FLUX.1-schnell"),
    # https://huggingface.co/black-forest-labs/FLUX.2-VAE (apache-2.0)。
    # flux2-dev・ERNIE・Lens で sha256 が同じ
    _k("flux2-vae", r"^flux2-vae\.safetensors$", "apache-2.0", None, "black-forest-labs/FLUX.2-VAE"),
    # https://huggingface.co/black-forest-labs/FLUX.2-small-decoder (apache-2.0)
    _k(
        "flux2-small-decoder",
        r"^flux2-vae_small\.safetensors$",
        "apache-2.0",
        None,
        "black-forest-labs/FLUX.2-small-decoder",
    ),
    # Wan の VAE。https://huggingface.co/Wan-AI/Wan2.1-T2V-14B ・ Wan-AI/Wan2.2-TI2V-5B (apache-2.0)
    _k(
        "wan2.1-vae",
        r"^(wan_2\.1_vae|wan2_1_vae_(bf16|fp16|fp32))\.safetensors$",
        "apache-2.0",
        None,
        "Wan-AI/Wan2.1-T2V-14B",
    ),
    _k("wan2.2-vae", r"^wan2\.2_vae\.safetensors$", "apache-2.0", None, "Wan-AI/Wan2.2-TI2V-5B"),
    # https://huggingface.co/Qwen/Qwen-Image (apache-2.0)。5 つのリポジトリで sha256 が同じ
    _k("qwen-image-vae", r"^qwen_image_vae\.safetensors$", "apache-2.0", None, "Qwen/Qwen-Image"),
    # Real-ESRGAN。https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE (BSD 3-Clause)
    _k(
        "real-esrgan",
        r"^realesrgan_x4plus(_anime_6b)?\.(safetensors|pth)$",
        "bsd-3-clause",
        None,
        "xinntao/Real-ESRGAN",
        "https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE",
    ),
    # --- 2026-10-20 に足したもの(#14 のレビュー: リポジトリの表記が、中の別の部品に付いていた) ---
    # DINOv3。https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m (other: dinov3-license。
    # gated: manual)。原文 https://ai.meta.com/resources/models-and-libraries/dinov3-license は、
    # 配るときに規約の写しと「Built with DINOv3」の表示を求め、用途(軍事など)を制限する
    _k(
        "dinov3",
        r"dino[\W_]?v3",
        "other",
        "dinov3-license",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "https://ai.meta.com/resources/models-and-libraries/dinov3-license",
        gated=True,
    ),
    # SAM 3 / 3.1。https://huggingface.co/facebook/sam3 (other。gated: manual)
    _k("sam3", r"(^|[^a-z0-9])sam[\W_]?3(?!\d)", "other", "sam-license", "facebook/sam3", gated=True),
    # Depth Anything V2 の Base / Large / Giant は非商用。
    # https://huggingface.co/depth-anything/Depth-Anything-V2-Large (cc-by-nc-4.0)。Small は下の apache-2.0
    _k(
        "depth-anything-v2-nc",
        r"depth[\W_]?anything[\W_]?v2[\W_]*(vit)?[blg]",
        "cc-by-nc-4.0",
        None,
        "depth-anything/Depth-Anything-V2-Large",
    ),
    # Lumina Image 2.0 系の一式のファイル(all in one)は、テキストエンコーダ Gemma 2 2B を含む
    # (https://huggingface.co/neta-art/Neta-Lumina の Text Encoder/gemma_2_2b_fp16.safetensors、
    # https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0 のカード
    # 「Text Encoder: Pre-trained Gemma-2-2b」)。
    # Gemma 2 は Gemma 利用規約 https://ai.google.dev/gemma/terms
    _k(
        "lumina2-all-in-one",
        r"(lumina|neta).*(all[\W_]?in[\W_]?one|aio)|^lumina_2\.safetensors$",
        "gemma",
        None,
        "google/gemma-2-2b",
        "https://ai.google.dev/gemma/terms",
    ),
    # https://huggingface.co/depth-anything/Depth-Anything-V2-Small (apache-2.0)
    _k(
        "depth-anything-v2-small",
        r"depth[\W_]?anything[\W_]?v2[\W_]*(vit)?s",
        "apache-2.0",
        None,
        "depth-anything/Depth-Anything-V2-Small",
    ),
    # https://huggingface.co/facebook/dinov2-large (apache-2.0)
    _k("dinov2", r"dino[\W_]?v2", "apache-2.0", None, "facebook/dinov2-large"),
    # SAM / SAM 2 / SAM 2.1。https://huggingface.co/facebook/sam-vit-huge ・
    # facebook/sam2.1-hiera-large (apache-2.0)
    _k(
        "sam-sam2",
        r"(^|[^a-z0-9])sam(2(\.1)?[\W_]*hiera|[\W_]vit[\W_]?[hlb])",
        "apache-2.0",
        None,
        "facebook/sam2.1-hiera-large",
    ),
    # https://huggingface.co/google/siglip-so400m-patch14-384 ・
    # google/siglip2-so400m-patch16-512 (apache-2.0)
    _k("siglip", r"(^|[^a-z0-9])siglip2?[\W_]", "apache-2.0", None, "google/siglip-so400m-patch14-384"),
    # https://huggingface.co/ZhengPeng7/BiRefNet (mit)
    _k("birefnet", r"^birefnet(_fp16|_fp32|_bf16)?\.safetensors$", "mit", None, "ZhengPeng7/BiRefNet"),
    # MoGe。https://huggingface.co/Ruicheng/moge-vitl ・ Ruicheng/moge-2-vitl-normal (mit)
    _k("moge-1", r"^moge_1_vitl", "mit", None, "Ruicheng/moge-vitl"),
    _k("moge-2-vitl-normal", r"^moge_2_vitl_normal", "mit", None, "Ruicheng/moge-2-vitl-normal"),
    # TRELLIS.2 の VAE。https://huggingface.co/microsoft/TRELLIS.2-4B (mit)。Pixal3D はこれを土台にしている
    _k("trellis2-vae", r"^trellis_2_(shape|texture)_vae", "mit", None, "microsoft/TRELLIS.2-4B"),
    # ACE-Step 1.5 の文章側の部品(Qwen3-Embedding-0.6B と、Qwen3 から作った 5Hz の LM)。
    # https://huggingface.co/ACE-Step/Ace-Step1.5 (mit。リポジトリに Qwen3-Embedding-0.6B/・
    # acestep-5Hz-lm-1.7B/ がある)。
    # 元の Qwen3 は apache-2.0(https://huggingface.co/Qwen/Qwen3-Embedding-0.6B ・ Qwen/Qwen3-1.7B)
    _k("ace-step-1.5-lm", r"^qwen_(0\.6b|1\.7b|4b)_ace15", "mit", None, "ACE-Step/Ace-Step1.5"),
)

# 本家を確かめられない部品(ファイル名で分かるが、重みのライセンスを読めないもの)。
# どのリポジトリの中でも不明にする
UNVERIFIED_COMPONENTS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # RT-DETRv4(https://github.com/RT-DETRs/RT-DETRv4)。Comfy-Org/SDPose の中にあるが SDPose の一部ではない
    (
        "rt-detr-v4",
        re.compile(r"rt[\W_]?detr[\W_]?v4", re.IGNORECASE),
        "RT-DETRv4 の重み(https://github.com/RT-DETRs/RT-DETRv4。コードは Apache-2.0 だが、重みは DINOv3 を"
        "教師にした蒸留で、重みのライセンスの記載が無い)",
    ),
)

# 系統は分かるが、元のライセンスの原文を今は読めないもの(名前に出たら不明にする)。
# Stable Diffusion 2 の元(stabilityai/stable-diffusion-2・-2-base)は 2026-10-20 に匿名で読めない(401)。
# 写し https://huggingface.co/sd2-community/stable-diffusion-2-base の表記は openrail++(用途の制限つき)
LINEAGES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    # https://huggingface.co/teemosliang/SDPose-Wholebody のカード:
    # 「U-Net backbone initialized with Stable Diffusion v2 weights」。表記は mit
    (
        "sdpose",
        re.compile(r"sd[\W_]?pose", re.IGNORECASE),
        "SDPose は Stable Diffusion 2 の U-Net から作られている(カードの表記は元の規約より寛容でありうる)",
    ),
    # https://github.com/EnVision-Research/Lotus の学習の設定は
    # stabilityai/stable-diffusion-2-base から始める。
    # https://huggingface.co/jingheya/lotus-depth-d-v1-1 の表記は apache-2.0
    (
        "lotus",
        re.compile(r"(?<![a-z0-9])lotus[\W_]*(depth|normal|[dg][\W_]v)", re.IGNORECASE),
        "Lotus は Stable Diffusion 2 base から学習している(カードの表記は元の規約より寛容でありうる)",
    ),
    # https://huggingface.co/netflix/void-model のカード「VOID is built on CogVideoX-Fun-V1.5-5b-InP」。
    # https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.5-5b-InP の LICENSE は The CogVideoX License
    # (5B の Transformer。商用は登録が要り、月 100 万アクセスの上限)。VOID の表記は apache-2.0(#28)
    (
        "void-model",
        re.compile(r"(?<![a-z0-9])void[\W_]*(model|pass[\W_]*\d)", re.IGNORECASE),
        "VOID は CogVideoX-Fun-V1.5-5b-InP(Transformer は CogVideoX License)から微調整している"
        "(カードの apache-2.0 は元の規約より寛容でありうる)",
    ),
    (
        "sd2",
        re.compile(
            r"(?<![a-z0-9])(sd[\W_]?2([\W_.]?[01])?(?![\d.])|stable[\W_]*diffusion[\W_]*v?2(?![\d.]))",
            re.IGNORECASE,
        ),
        "Stable Diffusion 2 の系統(元の規約の原文を確かめられない)",
    ),
)

# 系統の表: 名前(リポジトリ名・ファイル名・たどった base_model)で分かる元のモデル。派生(微調整・量子化・
# 無検閲化)のライセンスは元より寛容にならないので、表記がこれより寛容なら不明にする。
_F = r"(?<![a-z0-9])"
FAMILIES: tuple[KnownLicense, ...] = (
    # https://huggingface.co/black-forest-labs/FLUX.1-dev (other: flux-1-dev-non-commercial-license)。
    # Krea-dev・Kontext-dev・Fill-dev なども同じライセンス
    _k(
        "flux.1-dev",
        _F + r"flux[\W_]*1?[\W_]*((krea|kontext|fill|canny|depth|redux)[\W_]*)?dev(?![a-z])",
        "other",
        "flux-1-dev-non-commercial-license",
        "black-forest-labs/FLUX.1-dev",
    ),
    # https://huggingface.co/black-forest-labs/FLUX.1-schnell (apache-2.0)
    _k(
        "flux.1-schnell",
        _F + r"flux[\W_]*1?[\W_]*schnell",
        "apache-2.0",
        None,
        "black-forest-labs/FLUX.1-schnell",
    ),
    # https://huggingface.co/black-forest-labs/FLUX.2-dev (other: flux-non-commercial-license)
    _k(
        "flux.2-dev",
        _F + r"flux[\W_]*2[\W_]*dev(?![a-z])",
        "other",
        "flux-non-commercial-license",
        "black-forest-labs/FLUX.2-dev",
    ),
    # https://huggingface.co/black-forest-labs/FLUX.2-klein-9B (other: flux-non-commercial-license)
    _k(
        "flux.2-klein-9b",
        _F + r"klein[\W_]*(base[\W_]*)?9b",
        "other",
        "flux-non-commercial-license",
        "black-forest-labs/FLUX.2-klein-9B",
    ),
    # https://huggingface.co/black-forest-labs/FLUX.2-klein-4B (apache-2.0)
    _k(
        "flux.2-klein-4b",
        _F + r"klein[\W_]*(base[\W_]*)?4b",
        "apache-2.0",
        None,
        "black-forest-labs/FLUX.2-klein-4B",
    ),
    # https://huggingface.co/Qwen/Qwen-Image-2.1 (other: qwen-research)
    _k(
        "qwen-image-2.1",
        _F + r"qwen[\W_]*image[\W_]*2[\W_]*1(?!\d)",
        "other",
        "qwen-research",
        "Qwen/Qwen-Image-2.1",
    ),
    # https://huggingface.co/Qwen/Qwen-Image (apache-2.0)。Edit・2512 なども apache-2.0
    _k("qwen-image", _F + r"qwen[\W_]*image", "apache-2.0", None, "Qwen/Qwen-Image"),
    # https://huggingface.co/krea/Krea-2-Turbo (other: krea-2-community-license)
    _k("krea-2", _F + r"krea[\W_]*2(?![\d.])", "other", "krea-2-community-license", "krea/Krea-2-Turbo"),
    # https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 (openrail++)
    _k(
        "sdxl",
        _F + r"(sd[\W_]?xl|stable[\W_]*diffusion[\W_]*xl)",
        "openrail++",
        None,
        "stabilityai/stable-diffusion-xl-base-1.0",
    ),
    # https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 (creativeml-openrail-m)
    _k(
        "sd1.5",
        _F + r"(sd[\W_]?1[\W_.]?5(?!\d)|stable[\W_]*diffusion[\W_]*v?1[\W_.]?5(?!\d))",
        "creativeml-openrail-m",
        None,
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
    ),
    # https://huggingface.co/stabilityai/stable-diffusion-3.5-large (other: stabilityai-ai-community)
    _k(
        "sd3.5",
        _F + r"(sd[\W_]?3[\W_.]?5|stable[\W_]*diffusion[\W_]*3[\W_.]?5)",
        "other",
        "stabilityai-ai-community",
        "stabilityai/stable-diffusion-3.5-large",
    ),
    # https://huggingface.co/OnomaAIResearch/Illustrious-xl-early-release-v0 ・ https://huggingface.co/Laxhar/noobai-XL-1.1
    # (other: fair-ai-public-license-1.0-sd)
    _k(
        "illustrious-noobai",
        _F + r"(illustrious|noob[\W_]?ai)",
        "other",
        "fair-ai-public-license-1.0-sd",
        "OnomaAIResearch/Illustrious-xl-early-release-v0",
        "https://freedevproject.org/faipl-1.0-sd/",
    ),
    # https://huggingface.co/tencent/HunyuanVideo ・ tencent/Hunyuan3D-2 ・ tencent/HunyuanVideo-1.5
    # (other: tencent-hunyuan-community)
    _k("hunyuan", r"hunyuan", "other", "tencent-hunyuan-community", "tencent/HunyuanVideo"),
    # https://huggingface.co/stabilityai/stable-audio-open-1.0 ・ stabilityai/stable-audio-3-medium-base
    # (other: stable-audio-community)
    _k(
        "stable-audio",
        _F + r"stable[\W_]*audio",
        "other",
        "stable-audio-community",
        "stabilityai/stable-audio-open-1.0",
    ),
    # https://huggingface.co/Lightricks/LTX-2 (other: ltx-2-community-license-agreement。LTX-2.x も同じ系統)
    _k(
        "ltx-2",
        _F + r"ltx[\W_]*v?[\W_]*2(?![\db])",
        "other",
        "ltx-2-community-license-agreement",
        "Lightricks/LTX-2",
    ),
    # https://huggingface.co/Lightricks/LTX-Video (other。名前の記載なし)
    _k("ltx-video", _F + r"(ltx[\W_]*video|ltxv(?!\d))", "other", None, "Lightricks/LTX-Video"),
    # https://huggingface.co/Wan-AI/Wan2.1-T2V-14B ・ Wan-AI/Wan2.2-T2V-A14B (apache-2.0)
    _k("wan", _F + r"wan[\W_]*2[\W_.]*[12](?!\d)", "apache-2.0", None, "Wan-AI/Wan2.1-T2V-14B"),
    # https://huggingface.co/HiDream-ai/HiDream-I1-Full (mit)。
    # 本体の DiT だけ。テキストエンコーダ(Llama 3.1 など)は部品の表で別に見る
    _k("hidream", r"hidream", "mit", None, "HiDream-ai/HiDream-I1-Full"),
    # https://huggingface.co/Tongyi-MAI/Z-Image-Turbo (apache-2.0)
    _k("z-image", _F + r"z[\W_]?image", "apache-2.0", None, "Tongyi-MAI/Z-Image-Turbo"),
    # https://huggingface.co/lodestones/Chroma (apache-2.0)
    _k("chroma", _F + r"chroma(?![a-z])", "apache-2.0", None, "lodestones/Chroma"),
    # Marigold v1(Stable Diffusion 2 から)。https://huggingface.co/prs-eth/marigold-depth-v1-1 (openrail++)。
    # v2(huawei-bayerlab/marigold-v2-0)は Qwen-Image-Edit-2509 から作った apache-2.0 なので当てない
    _k("marigold-v1", _F + r"marigold(?![\W_]*v?2)", "openrail++", None, "prs-eth/marigold-depth-v1-1"),
    # https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct ・ -4B ・ -2B ・ -32B (apache-2.0)
    _k("qwen3-vl", _F + r"qwen[\W_]*3[\W_]*vl", "apache-2.0", None, "Qwen/Qwen3-VL-8B-Instruct"),
)


def component_for(filename: str) -> KnownLicense | None:
    base = filename.rsplit("/", 1)[-1]
    return next((c for c in COMPONENTS if c.pattern.search(base)), None)


def unverified_component_for(filename: str) -> tuple[str, str] | None:
    """本家を確かめられない部品なら (表の名前, 理由)。"""
    base = filename.rsplit("/", 1)[-1]
    return next(((k, why) for k, pat, why in UNVERIFIED_COMPONENTS if pat.search(base)), None)


def lineage_for(names: Iterable[str]) -> tuple[str, str] | None:
    """元のライセンスを読めない系統(LINEAGES)の名前が出ていれば (系統, 理由)。"""
    text = "\n".join(n.lower() for n in names if n)
    return next(((k, why) for k, pat, why in LINEAGES if pat.search(text)), None)


# 別のモデル(系統)の部品だと分かる名前の印。ファイル名にこれが出て、置き場所のリポジトリ・ライセンスを読んだ
# リポジトリの名前に出ていなければ「別の系統の部品」とみなし、部品の表か系統の表の行(本家を確かめたもの)が
# ファイル名に当たらない限り、ライセンスは不明にする(リポジトリの表記を中の別の部品に付けない。#14 のレビュー)
FOREIGN_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (k, re.compile(p, re.IGNORECASE))
    for k, p in (
        ("dino", r"dino"),
        ("sam", r"(^|[^a-z0-9])sam(?![a-z])"),
        ("siglip", r"sig(lip|clip)"),
        ("clip", r"(?<![a-z])clip(?![a-z])"),
        ("t5", r"(?<![a-z0-9])u?m?t5(?![0-9])"),
        ("qwen", r"qwen"),
        ("llama", r"llama|llava"),
        ("gemma", r"gemma"),
        ("mistral", r"mistral"),
        ("glm", r"(?<![a-z])glm"),
        ("birefnet", r"birefnet"),
        ("depth-anything", r"depth[\W_]?anything"),
        ("moge", r"(?<![a-z])moge(?![a-z])"),
        ("detr", r"detr"),
        ("flux", r"flux"),
        ("wan", r"(?<![a-z])wan(?![a-z])"),
        ("trellis", r"trellis"),
        ("hunyuan", r"hunyuan"),
        ("stable-diffusion", r"stable[\W_]*diffusion|(?<![a-z0-9])sd[\W_]?(xl|1|2|3)"),
        ("lumina", r"lumina"),
        ("ltx", r"(?<![a-z])ltx"),
        ("cogvideo", r"cogvideo"),
        ("whisper", r"whisper"),
        ("wav2vec", r"wav2vec"),
    )
)


def foreign_marker(filename: str, own_names: Iterable[str]) -> str | None:
    """ファイル名が、置き場所・ライセンスの出どころ(own_names)と別の系統の部品を示すなら、その印の名前。"""
    base = filename.rsplit("/", 1)[-1]
    own = "\n".join(n for n in own_names if n)
    for key, pat in FOREIGN_MARKERS:
        if pat.search(base) and not pat.search(own):
            return key
    return None


def verified_row_for(filename: str) -> KnownLicense | None:
    """ファイル名そのものに当たる、本家を確かめた行(部品の表、無ければ系統の表で一番厳しいもの)。"""
    base = filename.rsplit("/", 1)[-1]
    comp = component_for(base)
    if comp:
        return comp
    fams = families_for([base])
    return max(fams, key=lambda f: license_tier(f.license, f.license_name)) if fams else None


def families_for(names: Iterable[str]) -> list[KnownLicense]:
    text = "\n".join(n.lower() for n in names if n)
    return [f for f in FAMILIES if f.pattern.search(text)]


def license_label(lic: str | None, name: str | None = None) -> str:
    if lic is None:
        return "不明"
    return f"{lic}({name})" if name else lic


# --- 取得 --------------------------------------------------------------------------------------

MAX_HOPS = 12  # base_model をたどる段数の上限(版を重ねた系統は長い。超えたら確かめられない = 不明)
UNVERIFIED = "license_unverified"  # license_evidence の印(ライセンスを確かめられず不明にした)
# license_evidence の印: 本家が利用申請(規約への同意)を求める(置き場所が gated でなくても gate は保留にする)
UPSTREAM_GATED = "upstream_gated"


@dataclass
class Failure:
    repo_id: str
    message: str
    network: bool
    status: int | None = None

    @property
    def transient(self) -> bool:
        """待てば取れるかもしれない失敗(ネットワーク・429・5xx)。401/403/404 は不明として扱い、数えない。"""
        return self.network or self.status is None or self.status == 429 or self.status >= 500


@dataclass
class Resolved:
    """ライセンスを決めた結果。license が None なら不明(evidence に理由)。"""

    license: str | None
    license_name: str | None = None
    source: str | None = None  # license_from(ライセンスを読んだリポジトリ)
    url: str | None = None  # license_url
    path: list[str] = field(default_factory=list)  # たどった道(置き場所 → base_model … → 決め手)
    evidence: list[str] = field(default_factory=list)
    families: list[str] = field(default_factory=list)

    @property
    def tier(self) -> int:
        return license_tier(self.license, self.license_name)


def _unknown(path: list[str], why: str, source: str | None = None) -> Resolved:
    return Resolved(
        None, None, source or (path[-1] if path else None), None, list(path), [f"{UNVERIFIED}: {why}"]
    )


def _worst(items: list[Resolved]) -> Resolved:
    return max(items, key=lambda r: (r.tier, r.source or ""))


class HfClient:
    """リポジトリの情報を取って覚えておく。スレッドから同時に呼んでよい(同じリポジトリは 1 回だけ取る)。"""

    def __init__(self, fetch: Fetcher, *, workers: int = 8):
        self.fetch = fetch
        self.workers = max(1, min(8, workers))  # 上流に優しく: 同時に 8 本まで
        self._repos: dict[tuple[str, str], RepoInfo | None] = {}
        self._readmes: dict[str, str] = {}
        self._inflight: dict[tuple[str, Any], Future[Any]] = {}
        self._lock = threading.Lock()
        self.failures: list[Failure] = []

    def _once(self, kind: str, cache: dict[Any, Any], key: Any, load: Callable[[], Any]) -> Any:
        """cache に無ければ load() を 1 回だけ呼ぶ(同じ key を同時に頼まれたら、先の呼び出しの結果を待つ)。"""
        with self._lock:
            if key in cache:
                return cache[key]
            fut = self._inflight.get((kind, key))
            owner = fut is None
            if owner:
                fut = Future()
                self._inflight[(kind, key)] = fut
        assert fut is not None
        if not owner:
            return fut.result()
        value = None
        try:
            value = load()
        finally:
            with self._lock:
                cache[key] = value
                self._inflight.pop((kind, key), None)
            fut.set_result(value)
        return value

    def repo(self, repo_id: str, revision: str = "main") -> RepoInfo | None:
        key = (repo_id, revision or "main")

        def load() -> RepoInfo | None:
            try:
                data = self.fetch.get_json(api_url(*key))
                info = RepoInfo.from_api(data) if isinstance(data, dict) else None
                if info is None:
                    raise FetchError(api_url(*key), "形が違います")
                info.repo_id = info.repo_id or repo_id
                return info
            except FetchError as e:
                with self._lock:
                    self.failures.append(Failure(repo_id, str(e), isinstance(e, NetworkError), e.status))
                return None

        return self._once("repo", self._repos, key, load)

    def failed(self, repo_id: str) -> Failure | None:
        with self._lock:
            return next((f for f in self.failures if f.repo_id == repo_id), None)

    def readme(self, repo_id: str) -> str:
        def load() -> str:
            try:
                return self.fetch.get_text(readme_url(repo_id))
            except FetchError:
                return ""

        return self._once("readme", self._readmes, repo_id, load)

    def map(self, func: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(func, items))

    def prefetch(self, keys: Iterable[tuple[str, str]]) -> None:
        with self._lock:
            todo = sorted({k for k in keys if k not in self._repos})
        self.map(lambda k: self.repo(*k), todo)

    def prefetch_chains(self, repo_ids: Iterable[str]) -> None:
        """base_model の道を、段ごとにまとめて先に取る(MAX_HOPS 段まで)。"""
        level = set(repo_ids)
        seen: set[str] = set()
        for _ in range(MAX_HOPS + 1):
            level -= seen
            if not level:
                return
            self.prefetch((r, "main") for r in level)
            seen |= level
            nxt: set[str] = set()
            for r in level:
                info = self.repo(r)
                if info:
                    nxt |= set(info.base_models)
            level = nxt

    def search(self, url: str) -> list[dict[str, Any]]:
        data = self.fetch.get_json(url)
        if not isinstance(data, list):
            raise FetchError(url, "検索の結果が一覧ではありません")
        return [d for d in data if isinstance(d, dict) and d.get("id")]

    # --- ライセンス -------------------------------------------------------------------------

    def resolve_repo(self, repo_id: str, depth: int = 0, seen: frozenset[str] = frozenset()) -> Resolved:
        """リポジトリのライセンスを、base_model の道ごと確かめる(resolve_info)。読めなければ不明。"""
        info = self.repo(repo_id)
        if info is None:
            f = self.failed(repo_id)
            why = f"{repo_id} を読めません" + (f"(HTTP {f.status})" if f and f.status else "")
            return _unknown([repo_id], why)
        return self.resolve_info(info, depth, seen)

    def resolve_info(self, info: RepoInfo, depth: int = 0, seen: frozenset[str] = frozenset()) -> Resolved:
        """- 記載が無い → 不明
        - base_model のどれかが不明 → 不明
        - 自分の表記が、base_model の一番厳しいものより寛容 → 不明(派生は元より寛容にならない)
        - それ以外は自分の表記(同じか、より厳しい)
        """
        rid = info.repo_id
        if info.license is None:
            return _unknown([rid], f"{rid} にライセンスの記載がありません")
        own = Resolved(info.license, info.license_name, rid, info.license_url(), [rid])
        bases = [b for b in info.base_models if b != rid and b not in seen]
        if not bases:
            return own
        if depth >= MAX_HOPS:
            return _unknown([rid], f"base_model が {MAX_HOPS} 段を超えるので確かめきれません")
        worst = _worst([self.resolve_repo(b, depth + 1, seen | {rid}) for b in bases])
        own.path = [rid, *worst.path]
        if worst.license is None:
            return Resolved(None, None, worst.source, None, own.path, list(worst.evidence))
        if own.tier < worst.tier:
            why = (
                f"{rid} の表記 {license_label(own.license, own.license_name)} は、元の {worst.source} の "
                f"{license_label(worst.license, worst.license_name)} より寛容です"
            )
            return _unknown(own.path, why, worst.source)
        return own

    def resolve_file(self, info: RepoInfo, path: str, *, discovered_uncensored: bool = False) -> Resolved:
        """1 ファイルのライセンス(docs/sources.md の「ライセンス」の順)。"""
        base = path.rsplit("/", 1)[-1]
        rid = info.repo_id
        # 1. 本家を確かめられない部品・寛容でない部品は、どのリポジトリの中にあっても先に決める
        unv = unverified_component_for(base)
        if unv:
            return _unknown([rid], f"{base} は {unv[1]}", rid)
        comp = component_for(base)
        if comp and (comp.everywhere or is_repackager(rid)):
            return _component(rid, comp)
        if is_repackager(rid):
            ups = choose_bases(upstream_candidates(info), path)
            if not ups:
                return _unknown([rid], f"再配布 {rid} の {base} の本家が分かりません", rid)
            worst = _worst([self.resolve_repo(u, 1) for u in ups])
            res = Resolved(
                worst.license,
                worst.license_name,
                worst.source,
                worst.url,
                [rid, *worst.path],
                list(worst.evidence),
            )
            names = [base, *worst.path]
        else:
            r = self.resolve_info(info)
            res = Resolved(r.license, r.license_name, r.source, r.url, list(r.path), list(r.evidence))
            names = [rid.split("/", 1)[-1], path, *res.path[1:]]
        if res.license is None:
            return res
        # 2. 元のライセンスを読めない系統(Stable Diffusion 2 から作ったものなど)
        lin = lineage_for(names)
        if lin:
            return _unknown(res.path, f"系統 {lin[0]}: {lin[1]}", res.source)
        # 3. 別の系統の部品(リポジトリの表記を、中の別のモデルの部品に付けない)
        fk = foreign_marker(base, [rid, *res.path])
        if fk:
            row = verified_row_for(base)
            if row is None:
                why = (
                    f"{base} は {rid} の本体と別の系統({fk})の部品とみえ、本家のライセンスを確かめた表に無い"
                )
                return _unknown(res.path, why, res.source)
            if row in COMPONENTS:
                # 本家を確かめた部品。置き場所の表記と比べて厳しいほう
                if license_tier(row.license, row.license_name) >= res.tier:
                    return _component(rid, row)
                res.evidence.append(f"component:{row.key}")
        # 4. 系統の表(名前で分かる元のモデル)より寛容な表記は不明
        fams = families_for(names)
        res.families = [f.key for f in fams]
        if fams:
            fw = max(fams, key=lambda f: (license_tier(f.license, f.license_name), f.key))
            if res.tier < license_tier(fw.license, fw.license_name):
                why = (
                    f"{license_label(res.license, res.license_name)} は、系統 {fw.key} の "
                    f"{license_label(fw.license, fw.license_name)}({fw.url})より寛容です"
                )
                return _unknown(res.path, why, fw.source)
            res.evidence.append(f"family:{fw.key}")
        elif discovered_uncensored and len(res.path) <= 1:
            # 無検閲化は元のモデルの派生。元(base_model)も系統も分からなければ、表記を確かめられない
            return _unknown(res.path, f"{rid} の元のモデル(base_model)も系統も分かりません", rid)
        return res


def _component(rid: str, comp: KnownLicense) -> Resolved:
    ev = [f"component:{comp.key}"]
    if comp.gated:
        ev.append(f"{UPSTREAM_GATED}: {comp.source}")
    return Resolved(comp.license, comp.license_name, comp.source, comp.url, [rid, comp.source], ev)


# --- HF のリポジトリから、使うファイルを選ぶ -----------------------------------------------------

_SHARD = re.compile(r"[-_.]?(\d{5})-of-(\d{5})")
_VARIANT = re.compile(r"\.(fp16|bf16|fp32|fp8|non_ema|ema)(?=\.[a-z]+$)")
_GGUF_PREF = ("q8_0", "q4_k_m", "q5_k_m", "q6_k", "q4_k_s", "q5_k_s")
_SKIP_PATH = re.compile(
    r"(^|/)(\.|optimizer|scheduler|training_args|rng_state|safety_checker|feature_extractor|onnx|openvino|"
    r"coreml|flax|tf_model|voices?/|samples?/|examples?/)",
    re.IGNORECASE,
)
_DIFFUSERS_ROLE = {
    "transformer": "main",
    "transformer_2": "main",
    "unet": "main",
    "prior": "main",
    "decoder": "main",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "text_encoder_3": "text_encoder",
    "vae": "vae",
    "image_encoder": "clip_vision",
}


def name_role(path: str) -> str:
    n = path.lower()
    if re.search(r"mmproj|tokenizer|(^|[/_\-.])(codec|asr)([/_\-.]|$)", n):
        return "other"  # 画像の読み取り部・トークナイザなど。本体の判定には使わない
    if re.search(r"(^|[/_\-.])(vae|ae)([/_\-.]|$)", n):
        return "vae"
    if re.search(
        r"text[_\-]?encoder|(^|[/_\-.])(t5|t5xxl|umt5[_a-z]*|clip_[lg]|gemma[\w.]*|qwen[\d._]*_?vl[\w.]*|qwen_?[\d.]+_\d+b)([/_\-.]|$)",
        n,
    ):
        return "text_encoder"
    if re.search(r"lora|lycoris", n):
        return "lora"
    if re.search(r"clip[_\-]?vision|image_encoder", n):
        return "clip_vision"
    return "main"


def _variant_key(path: str) -> str:
    return _SHARD.sub("", path)


def _group(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in sorted(paths):
        groups.setdefault(_variant_key(p), []).append(p)
    return groups


def _size(info: RepoInfo, paths: list[str]) -> int:
    return sum((info.files.get(p, (0, None))[0] or 0) for p in paths)


def _pick_variant(info: RepoInfo, groups: dict[str, list[str]]) -> list[str]:
    """1 つの役割の中から 1 つの版を選ぶ。

    GGUF は Q8_0 → Q4_K_M → Q5_K_M → Q6_K → Q4_K_S → Q5_K_S の順、無ければ一番小さいもの。
    それ以外は safetensors を優先し、fp32 の版は他が無いときだけ使い、
    残りから一番大きいもの(完全な本体)を選ぶ。
    """
    if not groups:
        return []
    keys = list(groups)
    gguf = [k for k in keys if k.lower().endswith(".gguf")]
    if gguf:
        for pref in _GGUF_PREF:
            hit = [k for k in gguf if pref in k.lower()]
            if hit:
                return groups[min(hit, key=lambda k: _size(info, groups[k]))]
        return groups[min(gguf, key=lambda k: _size(info, groups[k]))]
    st = [k for k in keys if k.lower().endswith((".safetensors", ".sft"))]
    pool = st or keys
    non32 = [k for k in pool if "fp32" not in k.lower()] or pool
    return groups[max(non32, key=lambda k: (_size(info, groups[k]), k))]


def select_weight_files(info: RepoInfo) -> list[tuple[str, str]]:
    """HF のリポジトリから、動かすのに使う重みを 1 組だけ (path, role) で返す(docs/sources.md の規則)。

    見積もりはファイルの合計を使うので、同じ部品の別の版(精度・形式)を重ねて入れない。

    - 単一ファイルの本体(リポジトリ直下)があれば、diffusers の形があってもそちらを使う(ComfyUI が読む形)
    - diffusers の形だけのとき: 部品のフォルダ(transformer / unet / text_encoder* / vae …)ごとに 1 版。
      精度は fp16 → bf16 → 印の無い既定の版の順。分割されたファイルはまとめて取る
    - それ以外: 役割(本体・テキストエンコーダ・VAE)ごとに 1 版を選ぶ(_pick_variant)。LoRA は取らない
    """
    weights = [
        p
        for p in info.files
        if p.lower().endswith(WEIGHT_EXT) and not _SKIP_PATH.search(p) and info.files[p][0]
    ]
    if not weights:
        return []
    out: list[tuple[str, str]] = []
    diffusers = "model_index.json" in info.files
    top_main = [p for p in weights if "/" not in p and name_role(p) == "main"]
    if diffusers and top_main:
        # 単一ファイルを使う。diffusers の部品フォルダの中のものは数えない
        weights = [p for p in weights if p.split("/", 1)[0] not in _DIFFUSERS_ROLE]
    elif diffusers:
        by_dir: dict[str, list[str]] = {}
        for p in weights:
            if "/" in p:
                by_dir.setdefault(p.split("/", 1)[0], []).append(p)
        for d, paths in sorted(by_dir.items()):
            role = _DIFFUSERS_ROLE.get(d)
            if role is None:
                continue
            groups = _group(paths)
            st = {k: v for k, v in groups.items() if k.lower().endswith(".safetensors")} or groups
            chosen = None
            for pref in ("fp16", "bf16"):
                hit = sorted(k for k in st if f".{pref}." in k)
                if hit:
                    chosen = st[hit[0]]
                    break
            if chosen is None:
                default = {k: v for k, v in st.items() if not _VARIANT.search(k)}
                chosen = (
                    default[max(default, key=lambda k: _size(info, default[k]))]
                    if default
                    else _pick_variant(info, st)
                )
            out += [(p, role) for p in chosen]
        if any(r == "main" for _, r in out):
            return out
        out = []
    by_role: dict[str, list[str]] = {}
    for p in weights:
        by_role.setdefault(name_role(p), []).append(p)
    # 別々のモデルを詰め合わせたリポジトリ(本体が役割名でないフォルダ 4 つ以上に分かれている)は選べない
    model_dirs = {_top_dir(p) for p in by_role.get("main", [])} - {""}
    if len(model_dirs - _ROLE_DIRS) >= 4 and not any(_top_dir(p) == "" for p in by_role.get("main", [])):
        return []
    main_dir: str | None = None
    for role in ("main", "text_encoder", "vae"):
        groups = _group(by_role.get(role, []))
        if role == "main":
            # リポジトリ名と語が重なるファイルがあれば、そこから選ぶ(別のモデルの同梱を避ける)
            rt = _tokens(info.repo_id.split("/", 1)[-1])
            near = {k: v for k, v in groups.items() if _tokens(k.rsplit("/", 1)[-1]) & rt}
            groups = near or groups
        else:
            # 補助の部品は、本体と同じフォルダ・直下・役割名のフォルダにあるものだけ
            ok = {"", main_dir, *_ROLE_DIRS}
            groups = {k: v for k, v in groups.items() if _top_dir(k) in ok}
        picked = _pick_variant(info, groups)
        if role == "main" and picked:
            main_dir = _top_dir(picked[0])
        out += [(p, role) for p in picked]
    if not any(r == "main" for _, r in out):
        return []
    return out


_ROLE_DIRS = {
    "text_encoder",
    "text_encoders",
    "vae",
    "clip",
    "split_files",
    "diffusion_models",
    "checkpoints",
    "unet",
    "transformer",
    "models",
    "gguf",
}


def _top_dir(path: str) -> str:
    return path.split("/", 1)[0].lower() if "/" in path else ""


# --- 名前から精度・パラメータ数を推す --------------------------------------------------------------

_PRECISION = (
    ("w4a8", r"w4a8"),  # 重み 4 bit・活性 8 bit(audioint8 などの語より先に見る)
    ("nvfp4", r"nvfp4"),
    ("fp4", r"(?<![a-z])fp4"),
    ("nf4", r"(?<![a-z])nf4"),
    ("fp8", r"fp8|e4m3|e5m2"),
    ("int8", r"int8"),
    ("int4", r"int4"),
    ("bf16", r"bf16"),
    ("fp16", r"fp16"),
    ("fp32", r"fp32"),
)


def guess_precision(filename: str | None) -> str | None:
    if not filename:
        return None
    n = filename.lower()
    if n.endswith(".gguf"):
        m = re.search(r"(?<![a-z0-9])(i?q\d[a-z0-9_]*?|f16|bf16|f32)(?=\.gguf$|[-.])", n)
        return f"gguf-{m.group(1)}" if m else "gguf"
    for name, pat in _PRECISION:
        if re.search(pat, n):
            return name
    return None


def guess_params_b(*texts: str | None) -> float | None:
    for t in texts:
        if not t:
            continue
        m = re.search(r"(?<![a-z0-9.])(\d{1,3}(?:\.\d+)?)\s?b(?![a-z0-9])", t, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0.05 <= v <= 200:
                return v
    return None
