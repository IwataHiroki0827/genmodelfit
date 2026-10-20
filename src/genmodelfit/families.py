# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""ComfyUI のモデルの系統(model family)と、その系統が使える計算の精度(docs/hardware.md「精度」)。

ComfyUI(GPL-3.0)のコードは import しない。comfy/supported_models.py の `supported_inference_dtypes` と、
comfy/sd.py の VAE の `working_dtypes` を事実として読み、系統ごとに「fp16 で計算できるか」だけを表にした
(コミット COMFYUI_COMMIT。行番号はそのコミットのもの)。

- 本体: fp16 を持たない系統(`[bfloat16, float32]`)は、bf16 の計算を持たない GPU(Turing・Volta・Pascal など)で
  ComfyUI の unet_manual_cast が float32 を選ぶ(計算は fp32、作業用メモリは 2 倍)
- VAE: `working_dtypes` に fp16 が無い VAE は、bf16 を持たない GPU で vae_dtype が float32 を選ぶ
  (VAE の重みは 2 倍)

一覧の各項目の系統は `family_of`(本体のファイル名 → ID・テンプレート名の順に照らす)で決め、
update が `ModelEntry.comfy_family` / `comfy_vae` に入れる。分からないものは None(見積もりは変えない)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

COMFYUI_COMMIT = "b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c"  # 2026-10-20 の master


@dataclass(frozen=True)
class Family:
    key: str
    comfy_class: str  # supported_models.py のクラス
    fp16: bool  # supported_inference_dtypes に float16 があるか
    source: str  # 根拠の行(supported_models.py:行 など)
    vae: str | None = None  # 既定の VAE(VAES の鍵)。VAE のファイル名で分かればそちらを使う
    note: str | None = None


# 系統 → 本体の精度(supported_models.py。BASE の既定は [float16, bfloat16, float32]:
# supported_models_base.py:47)
FAMILIES: dict[str, Family] = {
    f.key: f
    for f in (
        Family("sd", "SD15 / SD20 / SDXL / SVD など", True, "supported_models_base.py:47(既定)", "kl"),
        Family("sd3", "SD3", True, "supported_models_base.py:47(既定)", "kl"),
        Family("flux", "Flux / FluxSchnell", True, "supported_models.py:753", "kl"),
        Family("flux_fill", "FluxInpaint(in_channels 96 = Fill)", False, "supported_models.py:783", "kl"),
        Family("flux2", "Flux2", True, "supported_models.py:812", "kl"),
        Family("flux3", "Flux3", False, "supported_models.py:855", "kl"),
        Family("chroma", "Chroma / ChromaRadiance", True, "supported_models.py:1820", "kl"),
        Family("longcat", "LongCatImage", True, "supported_models.py:2433", "kl"),
        Family("krea2", "Krea2", True, "supported_models.py:2023", "wan"),
        Family("lens", "Lens", False, "supported_models.py:882", "kl"),
        Family("mochi", "GenmoMochi", False, "supported_models.py:924"),
        Family("ltxv", "LTXV / LTXAV", False, "supported_models.py:952", "ltxv"),
        Family("minimax_h3", "MiniMaxH3", False, "supported_models.py:1000", "minimax_h3"),
        Family("hunyuan_video", "HunyuanVideo", False, "supported_models.py:1027", "hunyuan_video"),
        Family("hunyuan_video15", "HunyuanVideo15", True, "supported_models.py:2241", "hunyuan_video15"),
        Family("hunyuan_image21", "HunyuanImage21", False, "supported_models.py:2203"),
        Family("cosmos", "CosmosT2V / CosmosPredict2", True, "supported_models.py:1100"),
        Family("anima", "Anima", True, "supported_models.py:1171", "wan"),
        Family("lumina2", "Lumina2(NewBie も)", False, "supported_models.py:1213", "kl"),
        Family(
            "zimage",
            "ZImage",
            False,
            "supported_models.py:1240",
            "kl",
            "重みの値によっては ComfyUI が fp16 も許す"
            "(allow_fp16: model_detection.py:624、supported_models.py:1245)",
        ),
        Family("ming_image", "MingImage", False, "supported_models.py:1240(ZImage を継ぐ)"),
        Family("pixeldit", "PixelDiTT2I", False, "supported_models.py:1298"),
        Family(
            "wan",
            "WAN21_* / WAN22_*(HuMo・Vace・S2V・Animate・SCAIL など)",
            True,
            "supported_models.py:1366",
            "wan",
        ),
        Family("trellis2", "Trellis2(Pixal3D も)", False, "supported_models.py:1541", "trellis2"),
        Family("hunyuan3d", "Hunyuan3Dv2 / 2_1", True, "supported_models_base.py:47(既定)", "hunyuan3d"),
        Family("triposplat", "TripoSplat", True, "supported_models.py:1685"),
        Family("hidream", "HiDream", False, "supported_models.py:1710", "kl"),
        Family("hidream_o1", "HiDreamO1", False, "supported_models.py:1735"),
        Family("seedvr2", "SeedVR2", True, "supported_models.py:1853", "seedvr2"),
        Family("ace", "ACEStep", False, "supported_models.py:1903", "ace"),
        Family("ace15", "ACEStep15", False, "supported_models.py:2350"),
        Family(
            "omnigen2",
            "Omnigen2 / Boogu",
            True,
            "supported_models.py:1930・1938(torch 2.7 以上で float16 を先頭に足す)",
            "kl",
        ),
        Family("ideogram4", "Ideogram4", False, "supported_models.py:1994"),
        Family("moss_image", "MOSSImage", False, "supported_models.py:2052"),
        Family("mage_flow", "MageFlow", False, "supported_models.py:2085", "mage"),
        Family("qwen_image", "QwenImage", False, "supported_models.py:2114", "wan"),
        Family("qwen_image21", "QwenImage21", False, "supported_models.py:2144", "wan"),
        Family("joyimage", "JoyImage", False, "supported_models.py:2176", "wan"),
        Family("kandinsky5", "Kandinsky5", False, "supported_models.py:2294"),
        Family("yue2", "YuE2", False, "supported_models.py:2376"),
        Family("minimax_music3", "MiniMaxMusic3", True, "supported_models.py:2400"),
        Family("ernie", "ErnieImage", False, "supported_models.py:2495"),
        Family("cogvideox", "CogVideoX(VOID も)", True, "supported_models.py:2576", "cogvideox"),
        Family("stable_audio", "StableAudio / StableAudio3", True, "supported_models_base.py:47(既定)"),
        Family("sam3", "SAM3", True, "supported_models.py:2512"),
        Family("depth_anything3", "DepthAnything3", True, "supported_models.py:2471"),
    )
}


@dataclass(frozen=True)
class Vae:
    key: str
    fp16: bool  # working_dtypes に float16 があるか
    source: str


# VAE → working_dtypes(sd.py。既定は [bfloat16, float32]: sd.py:516。2D の KL VAE は既定のまま)
VAES: dict[str, Vae] = {
    v.key: v
    for v in (
        Vae("kl", False, "sd.py:516(既定。SD・SDXL・SD3・Flux の ae・Flux2 の VAE などの 2D の KL)"),
        Vae("ltxv", False, "sd.py:780"),
        Vae("ltx_diffusion", False, "sd.py:645(LTX 2.4 以降の拡散の VAE)"),
        Vae("mage", False, "sd.py:627"),
        Vae("wan", True, "sd.py:849・857・873(Wan 2.1 / 2.2 / Qwen Image 2.1)"),
        Vae("hunyuan_video", True, "sd.py:824"),
        Vae("hunyuan_video15", True, "sd.py:791"),
        Vae("hunyuan3d", True, "sd.py:902"),
        Vae("trellis2", True, "sd.py:594・599"),
        Vae("seedvr2", True, "sd.py:610"),
        Vae("cogvideox", True, "sd.py:808"),
        Vae("minimax_h3", True, "sd.py:1049"),
        Vae("ace", True, "sd.py:917"),
        Vae("fp32", True, "sd.py:540・947・999・1083(どの GPU でも float32 なので GPU による違いが無い)"),
    )
}

# 本体のファイル名・テンプレート名・リポジトリ名 → 系統(上から順に試す)
_RULES: tuple[tuple[str, str], ...] = (
    (r"flux3", "flux3"),
    (r"flux[._-]?1[._-]?fill|fill[._-]dev|onereward", "flux_fill"),
    (r"flux[._-]?2|flux2", "flux2"),
    (r"chroma", "chroma"),
    (r"longcat", "longcat"),
    (r"flux|kontext|ovis", "flux"),
    (r"krea[._-]?2", "krea2"),
    (r"(^|[^a-z])lens[_-]", "lens"),
    (r"mochi", "mochi"),
    (r"ltx", "ltxv"),
    (r"minimax[._-]?h3|fasth3", "minimax_h3"),
    (r"minimax[._-]?music", "minimax_music3"),
    (r"hunyuan[._-]?video[._-]?1\.5|hunyuanvideo15", "hunyuan_video15"),
    (r"hunyuan[._-]?image[._-]?2\.1", "hunyuan_image21"),
    (r"hunyuan[._-]?video", "hunyuan_video"),
    (r"hunyuan[._-]?3d", "hunyuan3d"),
    (r"cosmos", "cosmos"),
    (r"anima(?!te)", "anima"),  # animate(Wan Animate)と分ける
    (r"lumina|netayume|newbie", "lumina2"),
    (r"z[._-]?image", "zimage"),
    (r"ming[._-]?image", "ming_image"),
    (r"pixeldit", "pixeldit"),
    (r"hidream[._-]?o1", "hidream_o1"),
    (r"hidream", "hidream"),
    (r"seedvr2", "seedvr2"),
    (r"ace[._-]?step[._-]?1\.5|acestep[._-]?v?1\.5|ace[._-]?1\.5", "ace15"),
    (r"ace[._-]?step", "ace"),
    (r"omnigen|boogu", "omnigen2"),
    (r"ideogram|ideo[._-]?4", "ideogram4"),
    (r"moss[._-]?image", "moss_image"),
    (r"mage[._-]?flow", "mage_flow"),
    (r"qwen[._-]?image[._-]?2\.1|image[._-]2\.1[._-]|noct[._-]?q", "qwen_image21"),
    (r"qwen[._-]?image|firered", "qwen_image"),
    (r"joyai|joyimage", "joyimage"),
    (r"kandinsky", "kandinsky5"),
    (r"yue2", "yue2"),
    (r"ernie", "ernie"),
    (r"cogvideo|void[._-]pass", "cogvideox"),
    (r"trellis|pixal3d", "trellis2"),
    (r"triposplat", "triposplat"),
    (r"(^|[^a-z])wan|humo|chrono[._-]?edit|causal[._-]?forcing|scail|infinitetalk|bernini", "wan"),
    (r"sd3|sd[._-]?3\.5", "sd3"),
    (
        r"sd[._-]?xl|sdxl|noobai|lustify|juggernaut|realismbystable|meina|hassaku|svd[._-]|v1[._-]5|lotus",
        "sd",
    ),
    (r"stable[._-]?audio", "stable_audio"),
    (r"sam3", "sam3"),
    (r"depth[._-]?anything[._-]?3", "depth_anything3"),
)

# VAE のファイル名 → VAE(上から順に試す)
_VAE_RULES: tuple[tuple[str, str], ...] = (
    (r"ltx[._-]?2\.[45].*video[._-]?vae|ltx25.*video[._-]?vae", "ltx_diffusion"),
    (r"ltx", "ltxv"),
    (r"mage[._-]?flow", "mage"),
    (r"hunyuanvideo15|hunyuan[._-]?video[._-]?1\.5", "hunyuan_video15"),
    (r"hunyuan[._-]?video", "hunyuan_video"),
    (r"wan|qwen[._-]?image", "wan"),
    (r"trellis", "trellis2"),
    (r"seedvr2", "seedvr2"),
    (r"cogvideox", "cogvideox"),
    (r"minimax[._-]?h3", "minimax_h3"),
    (
        r"(^|/)ae\.safetensors|flux2[._-]?vae|vae[._-]ft[._-]mse|full[._-]encoder[._-]small[._-]decoder|sdxl[._-]?vae",
        "kl",
    ),
)


def _match(rules: tuple[tuple[str, str], ...], text: str) -> str | None:
    for pat, key in rules:
        if re.search(pat, text):
            return key
    return None


def _main_names(entry) -> list[str]:
    from .groups import main_files

    return [f.name.split("/")[-1].lower() for f in main_files(entry)]


def family_of(entry) -> str | None:
    """一覧の 1 件の系統。本体のファイル名で決まらなければ、ID・テンプレート名で決める。"""
    names = " ".join(_main_names(entry))
    key = _match(_RULES, names) if names else None
    if key:
        return key
    # 部品の置き場所(Comfy-Org/… のリポジトリ)は別のモデルの部品も混ざるので見ない。
    # HF の項目は ID がリポジトリ
    tmpl = entry.template.name if entry.template else ""
    return _match(_RULES, f"{entry.id} {tmpl}".lower())


def vae_of(entry, family: str | None = None) -> str | None:
    """VAE の種類。VAE のファイルの名前で決め、無ければ系統の既定(全部入りのチェックポイントなど)。"""
    vaes = [
        f.name.split("/")[-1].lower()
        for f in entry.files
        if (f.role == "vae" or f.directory == "vae") and f.name
    ]
    found = {_match(_VAE_RULES, n) for n in vaes} - {None}
    if len(found) == 1:
        return found.pop()
    if len(found) > 1:  # 複数の VAE(使い分け)は、fp16 を持たないものがあればそれ(安全側)
        return next((k for k in sorted(found) if not VAES[k].fp16), sorted(found)[0])
    fam = FAMILIES.get(family or "")
    return fam.vae if fam and not vaes else None


def assign(entry) -> None:
    """entry.comfy_family と entry.comfy_vae を決めて入れる(update と同梱の一覧の作り直しで使う)。"""
    entry.comfy_family = family_of(entry)
    entry.comfy_vae = vae_of(entry, entry.comfy_family)
