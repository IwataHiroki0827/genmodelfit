"""一覧の項目を「モデル」にまとめる(Issue #17)。llmfit と同じ考え方:

- 大きさ(パラメータ数)が違えば別のモデル(Wan 2.1 1.3B と 14B、FLUX.2 klein 4B と 9B)
- 精度(bf16 / fp8 / int8 / GGUF Q8 …)が違うだけなら同じモデル。動く中で一番良い精度を自動で選ぶ
- 使い方(文章から・編集・LoRA など)ごとのワークフローは、そのモデルの下の workflows

モデルの鍵(group_key、docs/schema.md「Models」):
1. 本体のファイル = 役割が main のもの(estimate の役割の決め方)。テキストエンコーダ(umt5 など)は鍵にしない。
   本体が複数なら一番大きいものと、その 95% 以上の大きさのもの(Wan 2.2 の high / low のような 1 組)。
   本体が無ければ、テキストエンコーダ・VAE 以外で一番大きいファイル。ファイルが無ければ項目そのもの
2. ファイル名(HF の model.safetensors のような決まった名前ならリポジトリ名)から、精度・量子化・包み方の語
   (fp16 / bf16 / fp8_e4m3fn / scaled / int8 / convrot / GGUF の Q4_K_M / KJ …)と分割の番号を外す。
   版(2.1・v1.5)・大きさ(1.3B・14B)・種類(turbo・dev・distilled・base・edit)の語は残す
3. 名前に大きさの語が無ければ params_b を足す
4. 語を並べ替えてつなぎ、種類(modality)と内容(sfw / uncensored)を前に付ける
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .models import ModelEntry, ModelFile

TIE_RATIO = 0.95  # 本体が複数のとき、一番大きいものとこの割合以上の大きさなら 1 組とみなす

_EXT = re.compile(r"\.(safetensors|gguf|pth|pt|ckpt|bin|onnx)$", re.IGNORECASE)
_SHARD = re.compile(r"[-_.]?\d{5}-of-\d{5}", re.IGNORECASE)
_GIB_SUFFIX = re.compile(r"[-_.]?\d+(\.\d+)?gib", re.IGNORECASE)
# GGUF の量子化(Q4_K_M・Q8_0・IQ3_XS など)。語に分ける前に外す
_GGUF = re.compile(r"(?<![a-z0-9])i?q\d+(_k)?(_[a-z0-9]+)?(?![a-z0-9])", re.IGNORECASE)
# 精度・量子化・包み方の語(語に分けたあとで外す)
DROP_TOKENS = frozenset(
    {
        "fp32", "fp16", "bf16", "f16", "f32", "fp8", "fp4", "nvfp4", "mxfp4", "e4m3fn", "e5m2", "e4m3",
        "fp8mixed", "fp8e4m3fn", "scaled", "mixed", "int8", "int4", "nf4", "w4a8", "w8a8", "audioint8",
        "convrot", "gguf", "kj", "comfy", "comfyui", "transformer", "only", "pruned", "aio", "high", "low",
        "noise", "split", "merged", "model", "weights", "safetensors",
    }
)  # fmt: skip
# HF でよくある決まったファイル名。これだけではモデルが分からないのでリポジトリ名を使う
GENERIC_NAMES = frozenset(
    {"model", "diffusion_pytorch_model", "pytorch_model", "checkpoint", "generator", "transformers", "unet"}
)
_SIZE = re.compile(r"^\d+(\.\d+)?[bm]$")
_ALIASES = ((re.compile(r"ace[-_ ]step"), "acestep"),)


def _tokens(text: str) -> list[str]:
    s = text.lower()
    s = _EXT.sub("", s)
    s = _SHARD.sub("", s)
    s = _GIB_SUFFIX.sub("", s)
    s = _GGUF.sub("", s)
    for pat, rep in _ALIASES:
        s = pat.sub(rep, s)
    s = re.sub(r"(\d)_(\d)(?=$|[^\d.b])", r"\1.\2", s)  # wan2_1 → wan2.1(14b・1080p の前ではつながない)
    out: list[str] = []
    for tok in re.split(r"[-_ /]+", s):
        tok = re.sub(r"^v(?=\d)", "", tok.strip("."))  # v1.5 → 1.5
        if not tok or tok in DROP_TOKENS:
            continue
        m = re.fullmatch(r"([a-z]+)\.?v?(\d+(?:\.\d+)*)", tok)  # wan2.1 → wan + 2.1、flux.1 → flux + 1
        parts = [m.group(1), m.group(2)] if m and not _SIZE.match(tok) else [tok]
        for p in parts:
            p = re.sub(r"^v(?=\d)", "", p)  # v1.5 → 1.5
            if p and p not in DROP_TOKENS:
                out.append(p)
    return out


def family_tokens(name: str) -> list[str]:
    """ファイル名(またはリポジトリ名)→ 精度などを外した語(並べ替え済み、重なりなし)。"""
    return sorted(set(_tokens(name.replace("\\", "/").rsplit("/", 1)[-1])))


def _base_name(f: ModelFile, m: ModelEntry) -> str:
    base = f.name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = _SHARD.sub("", _EXT.sub("", base.lower()))
    if stem in GENERIC_NAMES or stem.startswith(("model", "diffusion_pytorch_model")):
        repo = f.repo_id or (m.id[3:] if m.id.startswith("hf:") else "")
        if repo:
            return repo.rsplit("/", 1)[-1]
    return base


def main_files(m: ModelEntry) -> list[ModelFile]:
    from .estimate import _role_of

    mains = [f for f in m.files if _role_of(f) == "main"]
    if not mains:
        rest = [f for f in m.files if _role_of(f) not in ("text_encoder", "vae", "clip_vision")]
        mains = [max(rest, key=lambda f: f.size_bytes or 0)] if rest else []
    if len(mains) > 1:
        big = max(f.size_bytes or 0 for f in mains)
        if big:
            mains = [f for f in mains if (f.size_bytes or 0) >= big * TIE_RATIO]
    return mains


def model_name_key(m: ModelEntry) -> str | None:
    """本体のファイル名から作るモデルの名前の部分(大きさを含む)。決められなければ None。"""
    mains = main_files(m)
    if not mains:
        return None
    names = sorted({" ".join(family_tokens(_base_name(f, m))) for f in mains})
    names = [n for n in names if n]
    if not names:
        return None
    key = " + ".join(names)
    if m.params_b and not any(_SIZE.match(t) for n in names for t in n.split()):
        key += f" {m.params_b:g}b"
    return key


def group_key(m: ModelEntry) -> str:
    """モデルの鍵。同じ鍵の項目は、精度か使い方だけが違う同じモデル。"""
    name = model_name_key(m)
    if name is None:
        return f"{m.modality}:{m.content}:id:{m.id}"
    return f"{m.modality}:{m.content}:{name}"


def group_keys(models: Iterable[ModelEntry]) -> dict[str, str]:
    """id → group_key。"""
    return {m.id: group_key(m) for m in models}


# --- 精度 ---------------------------------------------------------------------------------------

# 精度の順位(小さいほど良い)。不明は全精度とみなす(精度の語の無いファイルはたいてい元の精度)
_PRECISION_RANK = (
    (re.compile(r"^(gguf-)?(fp32|bf16|fp16|f16|f32)$"), 0),
    (re.compile(r"^fp8"), 1),
    (re.compile(r"^int8$|^w8a8$"), 2),
    (re.compile(r"q8"), 3),
    (re.compile(r"q6"), 4),
    (re.compile(r"q5"), 5),
    (re.compile(r"q4"), 6),
)
LOWEST_RANK = 7  # q3 以下・int4・nf4・fp4 など


def precision_rank(precision: str | None) -> int:
    if not precision:
        return 0
    p = precision.lower()
    for pat, rank in _PRECISION_RANK:
        if pat.search(p):
            return rank
    return LOWEST_RANK


def precision_label(precision: str | None) -> str:
    """表に出す短い精度(bf16 / fp8 / int8 / q8 / q4 …)。不明は -。"""
    if not precision:
        return "-"
    p = precision.lower().removeprefix("gguf-")
    m = re.match(r"(i?q\d+)", p)
    return m.group(1) if m else p[:5]
