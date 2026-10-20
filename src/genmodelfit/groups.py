# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
3. 名前がリポジトリ名(HF の model.safetensors など)で大きさの語が無ければ params_b を足す
   (テンプレートの params_b は題から推したもので、言語モデルの大きさのこともあるので使わない)
4. 語を並べ替えてつなぎ、種類(modality)と内容(sfw / uncensored)を前に付ける
5. 一覧全体で: 本体が 2 つで名前が違う項目は既にあるモデルへ寄せ、本体の sha256 が同じ項目は同じモデルにする
   (build_index)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

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


def _generic(f: ModelFile) -> bool:
    base = f.name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = _SHARD.sub("", _EXT.sub("", base.lower()))
    return stem in GENERIC_NAMES or stem.startswith(("model", "diffusion_pytorch_model"))


def _base_name(f: ModelFile, m: ModelEntry) -> str:
    base = f.name.replace("\\", "/").rsplit("/", 1)[-1]
    if _generic(f):
        repo = f.repo_id or (m.id[3:] if m.id.startswith("hf:") else "")
        if repo:
            return repo.rsplit("/", 1)[-1]
    return base


# 本体の置き場所にあっても、生成の本体ではない部品(超解像・高解像度化・リファイナ)
_AUX_MAIN = re.compile(r"(^|[_\-.])(sr|super[_-]?res\w*|upscal\w*|refiner)([_\-.]|$)")


def main_files(m: ModelEntry) -> list[ModelFile]:
    """本体のファイル(役割が main。複数なら一番大きいものと、その 95% 以上の大きさのもの)。"""
    from .estimate import _role_of

    mains = [f for f in m.files if _role_of(f) == "main"]
    if not mains:
        rest = [f for f in m.files if _role_of(f) not in ("text_encoder", "vae", "clip_vision")]
        mains = [max(rest, key=lambda f: f.size_bytes or 0)] if rest else []
    core = [f for f in mains if not _AUX_MAIN.search(f.name.replace("\\", "/").rsplit("/", 1)[-1].lower())]
    if core and len(core) < len(mains):
        # 超解像(Hunyuan Video 1.5 の 1080p SR)・高解像度化・リファイナは、ほかに本体があれば本体にしない
        mains = core
    if len(mains) > 1:
        big = max(f.size_bytes or 0 for f in mains)
        if big:
            mains = [f for f in mains if (f.size_bytes or 0) >= big * TIE_RATIO]
    return mains


def _file_key(f: ModelFile, m: ModelEntry) -> str:
    name = " ".join(family_tokens(_base_name(f, m)))
    # params_b を足すのは、名前がリポジトリ名(model.safetensors など)で大きさの語が無いときだけ。
    # テンプレートの params_b は題から推したもので、言語モデルの大きさのこともある(ACE-Step 1.5 の 4B)
    if name and m.params_b and _generic(f) and not any(_SIZE.match(t) for t in name.split()):
        name += f" {m.params_b:g}b"
    return name


def model_name_key(m: ModelEntry) -> str | None:
    """本体のファイル名から作るモデルの名前の部分(大きさを含む)。決められなければ None。
    本体が 2 つ以上で名前が違うときは「+」でつなぐ(ModelIndex で、既にあるモデルへ寄せる)。"""
    names = sorted({k for f in main_files(m) if (k := _file_key(f, m))})
    return " + ".join(names) if names else None


def group_key(m: ModelEntry) -> str:
    """1 項目だけから決める鍵(同じ sha256 や、本体が 2 つの項目の寄せ先は ModelIndex が決める)。"""
    name = model_name_key(m)
    if name is None:
        return f"{m.modality}:{m.content}:id:{m.id}"
    return f"{m.modality}:{m.content}:{name}"


# --- 一覧全体でのまとめ(sha256・本体が 2 つの項目・表示名) -------------------------------------

_NAME_PRECISION = re.compile(
    r"\b(fp32|fp16|bf16|f16|fp8|fp4|nvfp4|int8|int4|nf4|w4a8|gguf|q\d+(_k)?(_[a-z0-9]+)?|"
    r"fp8[ _-]?scaled|scaled|mixed|convrot|kj|aio)\b",
    re.IGNORECASE,
)
_NAME_SKIP_WORDS = frozenset({"diffusion", "pytorch", "model", "models", "v"})


def is_community(m: ModelEntry) -> bool:
    """利用者が作ったテンプレート(index の username が ComfyUI でないもの)。"""
    u = m.template.username if m.template else None
    return bool(u) and u.strip().lower() != "comfyui"


def _flat(text: str) -> str:
    return re.sub(r"[-_. ]+", "", text.lower())


def _word_label(w: str) -> str:
    if _SIZE.match(w):
        return w[:-1] + w[-1].upper()
    if re.fullmatch(r"[a-z]\d[a-z]+|[a-z]+\d[a-z]", w):  # t2v・i2v・ti2v
        return w.upper()
    return w if any(ch.isdigit() for ch in w) else w.capitalize()


# 表示名に足してよい鍵の語(大きさ・版・種類)。それ以外の語(ファイル名の細かい語)は足さない
_APPEND_WORDS = frozenset(
    {"turbo", "distilled", "distill", "dev", "base", "schnell", "fill", "lite", "sft", "xl", "large",
     "medium", "small", "i2v", "t2v", "ti2v", "flf2v", "s2v", "480p", "720p", "1080p"}
)  # fmt: skip
# 使い方の言い回し(モデルの名前ではない)。「LTX-2 Text to Video (Distilled)」→「LTX-2 Distilled」
_USE_PHRASE = re.compile(
    r"\b(text|image|video|audio|speech|music|single view|multi ?view)[- ]to[- ]"
    r"(image|video|audio|speech|music|3d|model)\b"
    r"|\b(depth|normal|albedo|geometry|pose)\s+(estimation|map|detection)\b"
    r"|\b(image|video)\s+(upscale|segmentation|restoration)\b|\bframe interpolation\b|\bremove background\b",
    re.IGNORECASE,
)


def _appendable(w: str) -> bool:
    return bool(_SIZE.match(w) or re.fullmatch(r"\d+(\.\d+)+", w) or w in _APPEND_WORDS)


def _span_name(text: str, words: set[str]) -> tuple[int, str]:
    """(当たった鍵の語の数, 先頭から最後に当たった語までの範囲 + 含まれない大きさ・版・種類の語)。"""
    text = re.sub(r"\s{2,}", " ", _USE_PHRASE.sub("", text).replace("(", " ").replace(")", " ")).strip(" -_,")
    pieces = text.split()
    hit_at: dict[str, int] = {}
    for w in words:
        fw = _flat(w)
        if (len(fw) < 2 and not fw.isdigit()) or w in _NAME_SKIP_WORDS:
            continue
        for i in range(len(pieces)):
            ends = [j for j in range(i, min(len(pieces), i + 3)) if fw in _flat(" ".join(pieces[i : j + 1]))]
            if ends:
                hit_at[w] = ends[0]  # 当たった範囲の最後の片
                break
    if not pieces:
        return 0, ""
    end = max(hit_at.values()) if hit_at else len(pieces) - 1
    name = " ".join(pieces[: end + 1])
    missing = sorted(
        (w for w in words if w not in hit_at and _appendable(w) and _flat(w) not in _flat(name)),
        key=lambda w: (not _SIZE.match(w), w),
    )
    if missing:
        name += " " + " ".join(_word_label(w) for w in missing)
    return len(hit_at), name.strip(" -_,")


def display_name(key: str, members: list[ModelEntry]) -> str:
    words = set(key.rsplit(":", 1)[-1].replace(" + ", " ").split())
    if key.split(":", 2)[2].startswith("id:"):
        words = set()
    cands = []
    for m in members:
        parts = m.name.split(":")
        for part in [*parts, " ".join(parts)] if len(parts) > 1 else parts:
            text = clean_name(part)
            hits, name = _span_name(text, words) if text else (0, "")
            name = name.replace(",", "").strip()
            if not name:
                continue
            unofficial = is_community(m) or m.id.startswith("hf:")
            # 「:」の前・後ろ・両方をつないだもののうち、鍵の語を一番多く含むもの → 公式 → 短いもの
            cands.append((-hits, unofficial, len(name), name))
    return min(cands)[3] if cands else members[0].id


def clean_name(name: str) -> str:
    """一覧の名前 → モデルの表示名。「:」より前で、精度の語を外す
    (「Z-Image-Turbo Int8: Text to Image」→「Z-Image-Turbo」)。"""
    head = _NAME_PRECISION.sub("", name.split(":", 1)[0])
    head = re.sub(r"\s*\(\s*\)|\[\s*\]", "", head)
    head = re.sub(r"([-_])[-_]+", r"\1", head)  # 「10Eros-v1.5--ConvRot」の「--」
    return re.sub(r"\s{2,}", " ", head).strip(" -_") or name.strip()


@dataclass
class ModelIndex:
    """一覧全体の、項目 → モデルの鍵 と、モデルの表示名(一覧の中で重ならない)。"""

    key_of: dict[str, str]  # 項目の id → 鍵
    name_of: dict[str, str]  # 鍵 → 表示名
    members: dict[str, list[ModelEntry]]  # 鍵 → 項目(一覧の順)

    def find(self, text: str) -> str | None:
        """表示名(大文字小文字を区別しない)→ 鍵。"""
        t = text.strip().lower()
        hits = [k for k, n in self.name_of.items() if n.lower() == t]
        return hits[0] if len(hits) == 1 else None


def build_index(models: Iterable[ModelEntry]) -> ModelIndex:
    """鍵の決め方(docs/schema.md「Models」):

    1. 項目ごとに group_key
    2. 本体が 2 つ以上で名前が違う項目(FLUX.2 klein の base + distilled など)は、片方の名前だけのモデルが
       一覧にあればそこへ寄せる(base を含まないほう → 名前の順)。無ければつないだまま
    3. 本体のファイルの sha256 が同じ項目は同じモデル(名前の違う置き場所・リポジトリ。鍵は名前の順で先のもの)
    """
    models = list(models)
    raw = {m.id: group_key(m) for m in models}
    singles = {k for k in raw.values() if " + " not in k}
    key_of: dict[str, str] = {}
    for m in models:
        k = raw[m.id]
        if " + " in k:
            prefix, _, names = k.rpartition(":")
            parts = names.split(" + ")
            known = [p for p in parts if f"{prefix}:{p}" in singles]
            if known:
                known.sort(key=lambda p: ("base" in p.split(), p))
                k = f"{prefix}:{known[0]}"
        key_of[m.id] = k
    # 本体の sha256 の組が全部同じ項目の鍵をつなぐ(1 つでも sha256 が無い・違う本体があればつながない。
    # 内容・種類が違ってもつながない)
    parent: dict[str, str] = {}

    def root(k: str) -> str:
        while parent.get(k, k) != k:
            k = parent[k]
        return k

    by_sha: dict[tuple[str, str, frozenset[str]], str] = {}
    for m in models:
        files = main_files(m)
        if not files or not all(f.sha256 for f in files):
            continue
        sk = (m.modality, m.content, frozenset(f.sha256.lower() for f in files))
        k = key_of[m.id]
        if sk in by_sha:
            a, b = root(by_sha[sk]), root(k)
            if a != b:
                lo, hi = sorted((a, b))
                parent[hi] = lo
        else:
            by_sha[sk] = k
    key_of = {i: root(k) for i, k in key_of.items()}
    members: dict[str, list[ModelEntry]] = {}
    for m in models:
        members.setdefault(key_of[m.id], []).append(m)
    # 表示名(docs/schema.md「Models」、display_name): 仲間の名前を「:」の前後に分けた候補のうち、
    # 鍵の語(系統・版・大きさ)を一番多く含むもの → 公式のテンプレート → 短いもの。その中の鍵の語を含む範囲に、
    # 含まれない鍵の語(大きさ・turbo など)を足す。「Image Upscale: SeedVR2 3B」は「SeedVR2 3B」になる
    name_of = {k: display_name(k, ms) for k, ms in members.items()}
    # 重なったら: 種類が違えば種類 → 鍵の語 → それでも重なれば ID
    for step in ("modality", "words", "modality", "id"):
        seen: dict[str, list[str]] = {}
        for k, n in name_of.items():
            seen.setdefault(n.lower(), []).append(k)
        for keys in seen.values():
            if len(keys) < 2:
                continue
            for k in keys:
                if step == "modality":
                    if len({x.split(":", 1)[0] for x in keys}) == len(keys):
                        name_of[k] = f"{name_of[k]} ({k.split(':', 1)[0]})"
                elif step == "words":
                    words = k.rsplit(":", 1)[-1].replace(" + ", " ").split()
                    left = [
                        w for w in words if _flat(w) not in _flat(name_of[k]) and w not in _NAME_SKIP_WORDS
                    ]
                    extra = [w for w in left if _appendable(w)] or [w for w in left if not w.isdigit()]
                    if extra:
                        name_of[k] = f"{name_of[k]} ({' '.join(extra)})"
                else:
                    name_of[k] = f"{name_of[k]} [{members[k][0].id}]"
    return ModelIndex(key_of, name_of, members)


def index_for(db) -> ModelIndex:
    """一覧(Database)ごとに 1 回だけ作る。"""
    idx = getattr(db, "_model_index", None)
    if idx is None or idx[0] != len(db.models):
        idx = (len(db.models), build_index(db.models))
        db._model_index = idx
    return idx[1]


def model_precision(m: ModelEntry) -> str | None:
    """本体のファイル名から読む精度(update と表で同じものを使う)。本体が無ければ None。"""
    from .hf import guess_precision

    mains = sorted(main_files(m), key=lambda f: (-(f.size_bytes or 0), f.name))
    return guess_precision(mains[0].name) if mains else None


def group_keys(models: Iterable[ModelEntry]) -> dict[str, str]:
    """id → モデルの鍵(一覧全体で sha256 などを見て決めたもの)。"""
    return dict(build_index(models).key_of)


# --- 精度 ---------------------------------------------------------------------------------------

# 精度の順位(小さいほど良い)。不明は全精度とみなす(精度の語の無いファイルはたいてい元の精度)
_PRECISION_RANK = (
    (re.compile(r"^(gguf-)?(fp32|bf16|fp16|f16|f32)$"), 0),
    (re.compile(r"^fp8"), 1),
    (re.compile(r"^int8$|^w8a8$"), 2),
    (re.compile(r"q8"), 3),
    (re.compile(r"q6"), 4),
    (re.compile(r"q5"), 5),
    (re.compile(r"q4|^(w4a8|int4|nf4|fp4|nvfp4|mxfp4)$"), 6),
)
LOWEST_RANK = 7  # q3 以下など


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
    if m:
        return m.group(1)
    return {"nvfp4": "fp4", "mxfp4": "fp4", "fp8_e4m3fn": "fp8"}.get(p, p[:4])
