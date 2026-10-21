# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""必要な VRAM・メモリの見積もりと、PC との相性(4 段階)の判定。

考え方(willitrunai.com が公開している方式の考え方だけを参考にした。データは使っていない):

  - 重み: 部品(本体・テキストエンコーダ・VAE・その他)ごとにファイルの大きさで数える。
    同じ部品の別の形式・精度のファイル(HF のリポジトリに並ぶ fp16 / fp32、.safetensors / .bin など)は
    1 組だけ数える(select_files。決まりは docs/accuracy.md の「ファイルの数え方」)。
    1 ファイルに全部入りのチェックポイント(`checkpoints`)は、safetensors の見出しで確かめた内訳で分ける
    (内訳が分からないものは一般的な割合で分ける)
  - ComfyUI は「文章の読み取り → 生成 → 仕上げ(VAE)」を順に行い、
    足りなければ間で部品を降ろす。そこで 3 つの量を出す
      全部載せ(vram_gb)      = 固定分 + 全部品 + 作業用メモリ
      順に載せ替え(sequential) = 固定分 + 段ごとの最大(その段の部品 + 作業用メモリ)
      逃がす(vram_offload_gb)  = 固定分 + 段ごとの最大(その段の部品の一部 + 作業用メモリ)
    最後のものは ComfyUI の lowvram / 動的 VRAM が重みの一部をメインメモリに置いたまま動く場合で、遅い
  - 本体が複数あり、サンプラーごとに別の本体を使うワークフロー(Wan 2.2 14B の high_noise → low_noise など)は、
    生成の段を本体ごとの組(Parts.experts)に分ける。ComfyUI は次のサンプラーで本体を読み込むとき、
    使っていない前の本体を必要な分だけ降ろす(出典は EXPERT_SOURCE)。そこで「順に載せ替え」「逃がす」は
    1 組ずつ数え、「全部載せ」とメインメモリは全部を数える(降ろした本体はメインメモリに残る)
  - 作業用メモリ: 画像は画素数 × 枚数、動画は画素数 × コマ数の平方根、音声は長さ、3D は一定
  - 実測があれば、そのモデルの作業用メモリを実測に合わせてずらす。設定が実測と同じなら basis="measured"
  - 大きさが分からないモデルは判定しない(fit は unknown、unknown_size。0 GB として「載る」とも
    「動かない」とも言わない。0.1.0 までは too_tight だった)
  - 単位はすべて GiB(2^30 バイト)。係数は calibration.json(python -m genmodelfit.calibrate が実測から
    合わせたもの)。詳しくは docs/accuracy.md
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from .db import bundled_text
from .models import (
    FIT_UNKNOWN,
    ORIGIN_COMMUNITY,
    RUN_UNKNOWN,
    Estimate,
    Measurement,
    ModelEntry,
    ModelFile,
    Settings,
    SystemInfo,
)

GIB = 1024**3  # 大きさはすべて GiB で扱う(nvidia-smi の MiB / 1024、psutil のバイト / 2^30 と同じ単位)

# 種類ごとの既定の設定(何も指定せず、実測も無いとき)
DEFAULT_SETTINGS: dict[str, Settings] = {
    "image": Settings(width=1024, height=1024),
    "video": Settings(width=832, height=480, frames=81),
    "music": Settings(seconds=30),
    "sfx": Settings(seconds=10),
    "tts": Settings(seconds=10),
    "3d": Settings(),
}

# 実測が GPU の容量のこの割合以上なら「上限に張り付いた」とみなす(動的 VRAM が空きを使い切った。
# 本当の全部載せの量はこれ以上なので、下限としてだけ使う)
CENSOR_RATIO = 0.9

# メインメモリが足りないとき、必要量がこの倍率 × 搭載量までなら、仮想メモリ(ページファイル)で
# 動く(遅い)とみなして marginal にする。超えたら too_tight。Windows の既定(システム管理)の
# ページファイルは搭載量と同程度以上まで伸びるが、重みをページファイルに置くと極端に遅くなるので、
# 半分までにとどめた。1.5 は判断で決めた値で、実測していない。
# ページファイルの大きさが決まっている(固定の大きさ・無し・Linux のスワップ)ときは、
# 搭載量 + その大きさ(ただしこの倍率まで)を上限にする(ram_limit。docs/hardware.md)
RAM_PAGEFILE_FACTOR = 1.5

# GPU が無いときに CPU で動くとみなす重みの上限(GiB)と種類
CPU_MAX_WEIGHTS_GB = 2.0
CPU_MODALITIES = ("tts", "sfx", "music", "image")

UNKNOWN_SIZE_REASON = "大きさが分からないため判定できません"

# サンプラーごとに別の本体を使うとき、ComfyUI が前の本体を降ろしてから次を載せる根拠
# (Comfy-Org/ComfyUI のコミット b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c。v0.42.1 の後の master、
# 2026-10-20 に読んだ):
#   comfy/samplers.py:1241 サンプラーごとに sampler_helpers.prepare_sampling を呼ぶ
#   comfy/sampler_helpers.py:201 → model_management.load_models_gpu([その本体] + 付属のモデル)
#   comfy/model_management.py:1001 load_models_gpu が free_memory(要る量 × 1.1 + 作業用, device) を呼ぶ
#   comfy/model_management.py:905 / 919 今載っているモデルのうち今回使わないもの(keep_loaded に無いもの)を、
#     足りない分(memory_required − 空き)だけ model_unload で降ろす(メインメモリへ移す)
#   comfy/model_management.py:914 動的 VRAM のモデルどうしは降ろさず、要るときに読み込む(同時に全部は載せない)
EXPERT_SOURCE = "ComfyUI b1a247c comfy/model_management.py:1001 load_models_gpu → free_memory"

# 段の名前(ComfyUI が順に動かす単位)
STAGES = ("encode", "denoise", "decode", "other")
# 部品の役割 → 段。本体と同時に GPU に載るもの(ControlNet・model_patches・LoRA)は生成の段に入れる。
# 音声エンコーダ(Wan S2V の wav2vec など)は生成の前に音声を読むので読み取りの段
ROLE_STAGE = {
    "main": "denoise",
    "lora": "denoise",
    "controlnet": "denoise",
    "model_patch": "denoise",
    "text_encoder": "encode",
    "clip_vision": "encode",
    "audio_encoder": "encode",
    "vae": "decode",
    "other": "other",
}
# ComfyUI の置き場所 → 役割(最優先で使う)
DIRECTORY_ROLE = {
    "diffusion_models": "main",
    "unet": "main",
    "checkpoints": "main",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "clip_vision": "clip_vision",
    "vae": "vae",
    "vae_approx": "vae",
    "loras": "lora",
    "controlnet": "controlnet",
    "t2i_adapter": "controlnet",
    "style_models": "controlnet",
    "gligen": "controlnet",
    "hypernetworks": "controlnet",
    "photomaker": "controlnet",
    "model_patches": "model_patch",
    "audio_encoders": "audio_encoder",
}
# Hugging Face(diffusers 形式)のフォルダ名 → 役割
FOLDER_ROLE: list[tuple[str, str]] = [
    (r"^(unet|transformer(_\d+)?|prior|decoder|dit)$", "main"),
    (r"^(text_encoder|tokenizer)", "text_encoder"),
    (r"^vae", "vae"),
    (r"^(image_encoder|feature_extractor|image_processor)$", "clip_vision"),
    (r"^controlnet", "controlnet"),
    (r"^(audio_encoder|audio_processor)", "audio_encoder"),
]
# ファイル名 → 役割(置き場所もフォルダも無いとき)。テキストエンコーダは知っている名前だけにする
# (`qwen` のような短い語で引くと、qwen_image(本体)をエンコーダにしてしまう)
NAME_ROLE: list[tuple[str, str]] = [
    (r"lora", "lora"),
    (r"controlnet|control_net|^control_", "controlnet"),
    (r"clip_vision|sigclip|siglip|image_encoder|dinov2", "clip_vision"),
    (r"wav2vec|whisper|audio_encoder", "audio_encoder"),
    (
        r"text_encoder|umt5|t5xxl|t5_xxl|t5-xxl|t5-v1|t5gemma|byt5|clip_[lg](?![a-z0-9])|clip-[lg](?![a-z0-9])"
        r"|open_clip|llava_llama|qwen_?2\.5_?vl|qwen2_5_vl|qwen_?3[_-]?\d+(\.\d+)?b|gemma|llama_?3"
        r"|mistral_?3_small",
        "text_encoder",
    ),
    # `sd_xl_base_1.0_0.9vae`(VAE を差し替えた全部入り)は VAE にしない
    (r"(^|[_.-])vae([_.-]|$)|(^|[_.-])ae\.|^ae_", "vae"),
]

# 全部入りチェックポイントの内訳(本体, テキスト/画像エンコーダ, VAE の割合)。
# 2026-10-20 に各ファイルの safetensors の見出し(テンソルの大きさ)を読んで数えた。
CHECKPOINT_SPLITS: list[tuple[str, tuple[float, float, float], str]] = [
    (r"sd_xl|sdxl", (0.740, 0.236, 0.024), "SDXL(UNet 4.79 GB / CLIP-L+bigG 1.53 GB / VAE 0.16 GB)"),
    (
        r"ace_step_1\.5",
        (0.478, 0.489, 0.034),
        "ACE-Step 1.5(DiT 4.46 GB / 言語モデル等 4.56 GB / VAE 0.32 GB)",
    ),
    (
        r"hunyuan3d-dit-v2",
        (0.452, 0.461, 0.087),
        "Hunyuan3D 2.0(DiT 2.08 GB / 画像エンコーダ 2.11 GB / VAE 0.40 GB)",
    ),
    (
        r"hunyuan_3d_v2\.1",
        (0.828, 0.083, 0.089),
        "Hunyuan3D 2.1(DiT 5.68 GB / 画像エンコーダ 0.57 GB / VAE 0.61 GB)",
    ),
]
# 内訳が分からない全部入り・ファイル情報の無いもの(テンプレートの大きさだけ)に使う割合
GENERIC_SPLIT = (0.75, 0.20, 0.05)

# 精度ごとの 1 パラメータあたりのバイト数(精度の指定で本体の大きさを換算する)
PRECISION_BYTES = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "q8": 1.06,
    "q6": 0.82,
    "q5": 0.69,
    "nf4": 0.56,
    "q4": 0.56,
}
_PRECISION_ALIASES = {
    "float32": "fp32",
    "float": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
    "half": "fp16",
    "float8": "fp8",
}
_PRECISION_IN_NAME = [
    (r"fp32", "fp32"),
    (r"bf16", "bf16"),
    (r"fp16", "fp16"),
    (r"fp8|e4m3|e5m2", "fp8"),
    (r"int8", "int8"),
    (r"nf4", "nf4"),
    (r"q8_0|\bq8\b", "q8"),
    (r"q6_k|\bq6\b", "q6"),
    (r"q5_k|q5_0|q5_1|\bq5\b", "q5"),
    (r"q4_k|q4_0|q4_1|\bq4\b", "q4"),
]


def norm_precision(p: str | None) -> str | None:
    """精度の書き方をそろえる(`BF16` → `bf16`、`gguf-Q4_K_M` → `q4`、`float16` → `fp16`)。
    知らない値はそのまま小文字で返す。"""
    if not p or not str(p).strip():
        return None
    x = str(p).strip().lower()
    x = re.sub(r"^gguf[-_ ]?", "", x)
    if x in PRECISION_BYTES:
        return x
    if x in _PRECISION_ALIASES:
        return _PRECISION_ALIASES[x]
    return _name_precision(x) or x


def _name_precision(name: str) -> str | None:
    n = name.lower()
    for pat, prec in _PRECISION_IN_NAME:
        if re.search(pat, n):
            return prec
    return None


@dataclass
class Coefficients:
    """見積もりの係数。calibration.json で上書きする(fit されるのは overhead と作業用メモリの 5 つだけ)。"""

    overhead_gb: float = 0.6  # CUDA の文脈・ComfyUI 自身など
    act_image_gb_per_mp: float = 0.5  # 画像 1 メガ画素あたりの作業用メモリ(1 枚)
    act_video_gb_per_mp: float = 2.4  # 動画 1 メガ画素あたり(act_video_ref_frames コマのとき)
    act_audio_gb_per_min: float = 1.0  # 音声 1 分あたり
    act_3d_gb: float = 2.0  # 3D(octree の解像度が一定なら一定)
    # ここから下は合わせない(実測が無い、または合わせると 9 件に寄りすぎる)
    weight_factor: float = 1.0  # ファイルの大きさ → VRAM 上の大きさ
    act_video_ref_frames: int = 49
    act_video_frame_exponent: float = 0.5  # コマ数への効き方(docs/accuracy.md)
    offload_fraction: float = 0.4  # 逃がすときに GPU に残す重みの割合
    ram_extra_gb: float = 3.0  # メインメモリの固定分(Python・ComfyUI)
    good_margin: float = 0.90  # 使わない(段階は FIT_BANDS。calibration.json の互換のため残す)
    source: str = "default"
    extra: dict = field(default_factory=dict)


FITTED = ("overhead_gb", "act_image_gb_per_mp", "act_video_gb_per_mp", "act_audio_gb_per_min", "act_3d_gb")


def load_coefficients() -> Coefficients:
    try:
        data = json.loads(bundled_text("calibration.json"))
    except (FileNotFoundError, ValueError):
        return Coefficients()
    known = Coefficients.__dataclass_fields__
    values = {**data.get("fixed", {}), **data.get("coefficients", {})}
    c = Coefficients(**{k: v for k, v in values.items() if k in known and k not in ("source", "extra")})
    c.source = data.get("source", "calibration.json")
    return c


# --- ファイルの選び方(同じ部品の別の形式・精度は 1 組だけ数える) -------------------------------

FORMAT_RANK = {".safetensors": 0, ".sft": 0, ".gguf": 1, ".ckpt": 2, ".pt": 2, ".pth": 2, ".bin": 2}
# ComfyUI では使わない別の実行環境の重み(数えない)
ALT_RUNTIME_EXTS = {".onnx", ".onnx_data", ".msgpack", ".h5", ".tflite", ".mlmodel", ".pb", ".ot", ".engine"}
ALT_RUNTIME_FOLDERS = {"onnx", "openvino", "flax", "tf", "coreml", "mlx", "tensorrt", "trt", "safety_checker"}
# 重みではないファイル(設定・トークナイザ・説明・画像)
NON_WEIGHT_EXTS = {
    ".json",
    ".txt",
    ".md",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".mp4",
    ".wav",
    ".py",
    ".yaml",
    ".yml",
    ".gitattributes",
    ".model",
    ".tiktoken",
    ".vocab",
    ".merges",
    ".csv",
    ".pdf",
}
# 精度の候補の順(小さいほど優先)。同じ部品に fp16 と fp32 があれば fp16 を数える
PREC_RANK = {"bf16": 0, "fp16": 0, None: 1, "fp8": 2, "int8": 2, "q8": 2, "q6": 3, "q5": 4, "q4": 5, "nf4": 5}
PREC_RANK_FP32 = 6
# diffusers のフォルダの中の決まった名前(同じ部品の別の形式)
DIFFUSERS_STEMS = {
    "model",
    "pytorch_model",
    "diffusion_pytorch_model",
    "diffusion_flax_model",
    "openvino_model",
}
_SHARD = re.compile(r"[-_.]?\d{5}-of-\d{5}")
_PREC_TOKENS = re.compile(
    r"[-_.](fp32|bf16|fp16|fp8|e4m3fn|e5m2|scaled|int8|nf4|mixed|q\d_k(_[sml])?|q\d_\d|q\d)(?=$|[-_.])"
)


def _path(f: ModelFile) -> tuple[str | None, str]:
    """(HF のフォルダ名, ファイル名)。フォルダの無いファイルは None。"""
    name = f.name.replace("\\", "/").strip("/")
    if "/" in name:
        folder, _, base = name.rpartition("/")
        return folder.split("/")[0], base
    return None, name


def _ext(base: str) -> str:
    i = base.rfind(".")
    return base[i:].lower() if i > 0 else ""


def _stem(base: str) -> str:
    """形式・精度・分割の番号を除いた名前(同じ部品の別のファイルかを見分ける)。"""
    n = base.lower()
    ext = _ext(n)
    if ext:
        n = n[: -len(ext)]
    n = _SHARD.sub("", n)
    prev = None
    while prev != n:
        prev, n = n, _PREC_TOKENS.sub("", n)
    return n


def _size_ok(f: ModelFile) -> bool:
    return isinstance(f.size_bytes, int) and not isinstance(f.size_bytes, bool) and f.size_bytes > 0


def _role_of(f: ModelFile) -> str:
    """部品の役割。ComfyUI の置き場所 → 明示した role → HF のフォルダ名 → 知っているファイル名 の順。"""
    if f.directory:
        role = DIRECTORY_ROLE.get(f.directory)
        if role:
            return role
    if f.role in ROLE_STAGE:
        return f.role
    if f.directory:
        return "other"  # 知らない置き場所(背景の除去・高解像度化など)
    folder, base = _path(f)
    if folder:
        for pat, role in FOLDER_ROLE:
            if re.search(pat, folder.lower()):
                return role
    n = base.lower()
    for pat, role in NAME_ROLE:
        if re.search(pat, n):
            return role
    return "main"


def _known_all_in_one(f: ModelFile) -> bool:
    return any(re.search(pat, f.name.lower()) for pat, _s, _l in CHECKPOINT_SPLITS)


def select_files(entry: ModelEntry) -> tuple[list[ModelFile], list[str]]:
    """数えるファイルを選ぶ。決まり(docs/accuracy.md の「ファイルの数え方」):

    1. 重みでないファイル(.json など)と、ComfyUI で使わない形式(onnx / flax など)は数えない
    2. 部品ごとに 1 組だけ数える。部品 = ComfyUI の置き場所 + 名前(精度・形式・分割番号を除く)、
       HF の diffusers のフォルダ(VAE は 1 つ)、フォルダの外の本体
    3. 1 組の選び方: .safetensors > .gguf > .bin / .ckpt / .pt(h)、次に fp16 / bf16 > 精度の書いていないもの >
       fp8 / int8 / q8 > … > fp32、同じなら大きいほう(安全側)
    4. フォルダの外の単一ファイル(本体)と diffusers 形式の両方があれば、単一ファイルを数える。
       単一ファイルが全部入りと分かっている(SDXL など)ときは diffusers のフォルダを全部除き、
       分からないときは diffusers の本体・VAE のフォルダのうち、フォルダの外に同じ役割のファイルが
       あるものだけ除く(単一ファイルが本体だけの場合にエンコーダを数え落とさないため。全部入りなら多めに出る)
    5. フォルダの中は、diffusers の決まった名前(model / pytorch_model / diffusion_pytorch_model)を
       1 つの部品とみなす。それ以外の名前(text_encoders/clip_l と t5xxl など)は名前ごとに別の部品
    """
    notes: list[str] = []
    cands: list[ModelFile] = []
    skipped_alt = 0
    # runtime が ComfyUI 以外と決まっているもの(piper の onnx など。#36)は、その形式が本体なので数える
    own_runtime = entry.runtime not in (None, "comfyui")
    for f in entry.files:
        folder, base = _path(f)
        ext = _ext(base)
        if not own_runtime and (
            (folder and folder.lower() in ALT_RUNTIME_FOLDERS) or ext in ALT_RUNTIME_EXTS
        ):
            skipped_alt += 1
            continue
        if ext in NON_WEIGHT_EXTS:
            continue
        cands.append(f)
    if skipped_alt:
        notes.append(f"ComfyUI で使わない形式のファイル {skipped_alt} 個(onnx など)は数えていない")

    root_main = [f for f in cands if not f.directory and _path(f)[0] is None and _role_of(f) == "main"]
    in_folder = [f for f in cands if not f.directory and _path(f)[0] is not None]
    if root_main and in_folder:
        if any(_known_all_in_one(f) for f in root_main):
            drop = set(map(id, in_folder))
            notes.append("全部入りの単一ファイルと diffusers 形式の両方があるので、単一ファイルだけ数えた")
        else:
            at_root = {_role_of(f) for f in cands if not f.directory and _path(f)[0] is None}
            same = at_root & {"main", "vae"}
            drop = {id(f) for f in in_folder if _role_of(f) in same}
            if drop:
                notes.append(
                    "単一ファイルと diffusers 形式の両方がある部品(本体・VAE)は、単一ファイルを数えた"
                )
        cands = [f for f in cands if id(f) not in drop]

    # フォルダの外の本体: 名前(精度などを除く)が同じか、一方が他方で始まるもの(sd_xl_base_1.0 と
    # sd_xl_base_1.0_0.9vae など)を同じ部品の別版とみなす。
    # 名前の違う本体(高ノイズ用と低ノイズ用など)は別の部品
    root_stems: dict[str, str] = {}
    for st in sorted({_stem(_path(f)[1]) for f in root_main}, key=len):
        root_stems[st] = next((r for r in root_stems.values() if st.startswith(r)), st)

    groups: dict[tuple, list[ModelFile]] = {}
    for f in cands:
        folder, base = _path(f)
        role = _role_of(f)
        if f.directory:
            key: tuple = ("dir", f.directory, _stem(base))
        elif folder:
            st = _stem(base)
            st = "" if st in DIFFUSERS_STEMS else st
            key = ("folder", "vae" if role == "vae" else folder.lower(), st)
        elif role == "main":
            key = ("root", "main", root_stems.get(_stem(base), _stem(base)))
        else:
            key = ("root", role, _stem(base))
        groups.setdefault(key, []).append(f)

    chosen: list[ModelFile] = []
    dropped = 0
    for files in groups.values():
        sets: dict[tuple, list[ModelFile]] = {}
        for f in files:
            _folder, base = _path(f)
            low = base.lower()
            ext = _ext(low)
            sets.setdefault((_SHARD.sub("", low[: -len(ext)] if ext else low), ext), []).append(f)

        def rank(item):
            (stem, ext), fs = item
            prec = _name_precision(stem)
            prank = PREC_RANK_FP32 if prec == "fp32" else PREC_RANK.get(prec, 1)
            size = sum(f.size_bytes for f in fs if _size_ok(f))
            return (FORMAT_RANK.get(ext, 3), prank, -size, stem)

        best = min(sets.items(), key=rank)[1]
        dropped += len(files) - len(best)
        chosen.extend(best)
    if dropped:
        notes.append(
            f"同じ部品の別の形式・精度のファイル {dropped} 個は数えていない"
            "(.safetensors・fp16 / bf16 を優先して 1 組だけ数えた)"
        )
    order = {id(f): i for i, f in enumerate(entry.files)}
    chosen.sort(key=lambda f: order[id(f)])
    return chosen, notes


# --- 部品(重み) ---------------------------------------------------------------------------------


@dataclass
class Parts:
    """段ごとの重み(GiB)と、分け方の説明。unknown=True は大きさが分からず判定できないもの。"""

    encode: float = 0.0
    denoise: float = 0.0
    decode: float = 0.0
    other: float = 0.0
    notes: list[str] = field(default_factory=list)
    unknown: bool = False
    files_gb: float | None = None  # 数えたファイルの大きさの合計(ダウンロードの量)
    # 生成の段を、サンプラーごとに順に使う本体の組に分けたときの組ごとの重み(GiB。共通の部品を含む)。
    # 空なら分けない(1 組)。denoise は全部の和のまま(全部載せ・メインメモリに使う)
    experts: list[float] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.encode + self.denoise + self.decode + self.other

    @property
    def denoise_peak(self) -> float:
        """生成の段で同時に GPU に要る重み(本体の組に分けたときは一番大きい組)。"""
        return min(self.denoise, max(self.experts)) if self.experts else self.denoise

    @property
    def largest(self) -> float:
        return max(self.stage(st) for st in STAGES)

    def stage(self, name: str) -> float:
        return getattr(self, name)


def _split_of(name: str) -> tuple[tuple[float, float, float], str | None]:
    n = name.lower()
    for pat, split, label in CHECKPOINT_SPLITS:
        if re.search(pat, n):
            return split, label
    return GENERIC_SPLIT, None


def _fallback_size(entry: ModelEntry, p: Parts) -> int:
    """ファイルの大きさが使えないときの大きさ(バイト)。
    テンプレートの大きさ → パラメータ数 × 精度。無ければ 0。"""
    t = entry.template.size_bytes if entry.template else None
    if isinstance(t, int) and t > 0:
        p.notes.append("ファイルごとの大きさが分からないので、テンプレートの大きさを使った")
        return t
    prec = norm_precision(entry.precision)
    if entry.params_b and entry.params_b > 0 and prec in PRECISION_BYTES:
        p.notes.append("ファイルの大きさが無いので、パラメータ数と精度から重みを求めた")
        return int(entry.params_b * 1e9 * PRECISION_BYTES[prec])
    return 0


def components(entry: ModelEntry) -> Parts:
    """モデル一式を段ごとの重み(GiB)に分ける。大きさが分からなければ unknown=True(重みは 0 のまま)。"""
    p = Parts()
    files, notes = select_files(entry)
    p.notes.extend(notes)
    # 本体のファイルが一覧に無い一式(#2 が files_complete=False を付ける)は、
    # ファイルの合計では小さく出るので、
    # テンプレートの大きさ(size_floor_bytes)を下限にして一般的な割合で分ける。下限も無ければ判定しない
    if getattr(entry, "files_complete", True) is False:
        listed = sum(f.size_bytes for f in files if _size_ok(f))
        floor = (
            getattr(entry, "size_floor_bytes", None)
            or (entry.template.size_bytes if entry.template else 0)
            or 0
        )
        size = max(listed, floor)
        if floor <= 0:
            p.unknown = True
            p.notes.append(UNKNOWN_SIZE_REASON + "(本体のファイルが一覧に無く、全体の大きさも無い)")
            return p
        main, enc, vae = GENERIC_SPLIT
        p.denoise, p.encode, p.decode = size * main / GIB, size * enc / GIB, size * vae / GIB
        p.files_gb = size / GIB
        p.notes.append("本体のファイルが一覧に無いので、テンプレートの大きさを下限にして一般的な割合で分けた")
        return p
    if not files or not all(_size_ok(f) for f in files):
        size = _fallback_size(entry, p)
        if size <= 0:
            p.unknown = True
            p.notes.append(UNKNOWN_SIZE_REASON + "(ファイルの大きさもテンプレートの大きさも無い)")
            return p
        main, enc, vae = GENERIC_SPLIT
        p.denoise, p.encode, p.decode = size * main / GIB, size * enc / GIB, size * vae / GIB
        p.files_gb = size / GIB
        p.notes.append("部品の内訳が無いので、一般的な割合(本体 75%・エンコーダ 20%・VAE 5%)で分けた")
        return p
    denoise_files: dict[str, float] = {}  # 生成の段の単独のファイル(名前 → GiB)。本体の組に分けるのに使う
    for f in files:
        gb = f.size_bytes / GIB
        role = _role_of(f)
        folder, _base = _path(f)
        all_in_one = f.directory == "checkpoints" or (
            not f.directory and folder is None and _known_all_in_one(f)
        )
        if ROLE_STAGE[role] == "denoise" and not (role == "main" and all_in_one):
            key = _base_lower(f.name)
            denoise_files[key] = denoise_files.get(key, 0.0) + gb
        if role == "main" and all_in_one:
            (main, enc, vae), label = _split_of(f.name)
            p.denoise += gb * main
            p.encode += gb * enc
            p.decode += gb * vae
            p.notes.append(
                f"全部入りの {f.name} を分けた: {label}"
                if label
                else f"全部入りの {f.name} を一般的な割合(本体 75%・エンコーダ 20%・VAE 5%)で分けた"
            )
            continue
        stage = ROLE_STAGE[role]
        setattr(p, stage, p.stage(stage) + gb)
    p.files_gb = sum(f.size_bytes for f in files) / GIB
    if p.denoise == 0 and p.total > 0:  # 本体が見つからない(役割の付け方が違う)ときは最大の段を本体とみなす
        biggest = max(STAGES, key=p.stage)
        p.denoise, value = p.stage(biggest), p.denoise
        setattr(p, biggest, value)
    else:
        _split_experts(entry, files, denoise_files, p)
    return p


def _base_lower(name: str) -> str:
    return name.replace("\\", "/").rsplit("/", 1)[-1].lower()


_HIGH = re.compile(r"(^|[_.-])high[_-]?noise([_.-]|$)")
_LOW = re.compile(r"(^|[_.-])low[_-]?noise([_.-]|$)")


def expert_sets(entry: ModelEntry) -> tuple[list[set[str]], str] | None:
    """サンプラーごとに順に使う本体の組(ファイル名の小文字の集まり)と、その出どころ。

    1. update がテンプレートのつながりから求めた sampler_stages(2 組以上。出どころ "workflow")
    2. それが無い(調べていない)とき、名前の決まり(出どころ "name"): 本体に `*high_noise*` と `*low_noise*`
       の両方がある(Wan 2.2 14B の流儀)。LoRA などの生成の段の部品は名前の high_noise / low_noise で
       振り分け、印の無いものは共通にする
    当てはまらなければ None(生成の段は 1 組)。"""
    stages = getattr(entry, "sampler_stages", None)
    if stages is not None:
        sets = [{_base_lower(n) for n in st} for st in stages if st]
        return (sets, "workflow") if len(sets) >= 2 else None
    mains = [_base_lower(f.name) for f in entry.files if _role_of(f) == "main"]
    high = {n for n in mains if _HIGH.search(n)}
    low = {n for n in mains if _LOW.search(n)}
    if not high or not low:
        return None
    for f in entry.files:
        n = _base_lower(f.name)
        role = _role_of(f)
        if role == "main" or ROLE_STAGE[role] != "denoise":
            continue
        if _HIGH.search(n):
            high.add(n)
        elif _LOW.search(n):
            low.add(n)
    return [high, low], "name"


def _split_experts(
    entry: ModelEntry, files: list[ModelFile], denoise_files: dict[str, float], p: Parts
) -> None:
    """生成の段を、サンプラーごとに順に使う本体の組に分ける(Parts.experts)。

    組の重み = その組のファイル + どの組にも入らない生成の段のファイル(共通)。数えたファイルの本体(main)を
    含む組が 2 つ以上あるときだけ分ける。"""
    found = expert_sets(entry)
    if not found or not denoise_files:
        return
    sets, how = found
    mains = {_base_lower(f.name) for f in files if _role_of(f) == "main"}
    sets = [st & set(denoise_files) for st in sets]
    sets = [st for st in sets if st & mains]
    if len(sets) < 2:
        return
    shared = sum(gb for n, gb in denoise_files.items() if not any(n in st for st in sets))
    p.experts = [shared + sum(denoise_files[n] for n in st) for st in sets]
    src = "ワークフローのつながり" if how == "workflow" else "ファイル名の high_noise / low_noise"
    p.notes.append(
        f"本体はサンプラーごとに {len(sets)} 組を順に使う({src})。ComfyUI は次の組を載せるとき前の組を降ろす"
        f"({EXPERT_SOURCE})ので、順に載せ替え・逃がすときは 1 組ずつ数えた"
        f"(一番大きい組 {max(p.experts):.1f} GB、生成の段の全部は {p.denoise:.1f} GB)"
    )


def source_precision(entry: ModelEntry) -> str | None:
    if norm_precision(entry.precision):
        return norm_precision(entry.precision)
    for f in entry.files:
        if _role_of(f) == "main":
            prec = _name_precision(_path(f)[1])
            if prec:
                return prec
    return None


def apply_precision(parts: Parts, entry: ModelEntry, s: Settings) -> Parts:
    """Settings.precision が指定されたら、本体の重みをその精度に換算する(エンコーダ・VAE はそのまま)。"""
    want = norm_precision(s.precision)
    if not want:
        return parts
    have = source_precision(entry)
    if want not in PRECISION_BYTES:
        parts.notes.append(f"精度 {want} は知らない値なので換算していない")
        return parts
    if have not in PRECISION_BYTES:
        parts.notes.append(f"元の精度が分からないので、精度 {want} の換算をしていない")
        return parts
    if have != want:
        ratio = PRECISION_BYTES[want] / PRECISION_BYTES[have]
        parts.denoise *= ratio
        parts.experts = [x * ratio for x in parts.experts]
        parts.notes.append(f"本体を {have} → {want} に換算した(×{ratio:.2f})")
    return parts


# --- 作業用メモリ ---------------------------------------------------------------------------------


def activation_features(modality: str, s: Settings, c: Coefficients) -> dict[str, float]:
    """作業用メモリの係数に掛ける量。calibrate も同じものを使う。負の値は 0 に切り上げる。"""
    mp = max(0.0, s.megapixels or 0.0)
    batch = max(1, s.batch or 1)
    if modality == "image":
        return {"act_image_gb_per_mp": mp * batch}
    if modality == "video":
        frames = s.frames if s.frames and s.frames > 0 else c.act_video_ref_frames
        scale = (frames / c.act_video_ref_frames) ** c.act_video_frame_exponent
        return {"act_video_gb_per_mp": mp * scale * batch}
    if modality in ("music", "sfx", "tts"):
        sec = s.seconds if s.seconds and s.seconds > 0 else 30
        return {"act_audio_gb_per_min": sec / 60}
    if modality == "3d":
        return {"act_3d_gb": 1.0}
    return {}


def activation_gb(entry: ModelEntry, s: Settings, c: Coefficients) -> float:
    feats = activation_features(entry.modality, s, c)
    if not feats:
        return 0.5
    return sum(getattr(c, k) * v for k, v in feats.items())


# --- 見積もり -------------------------------------------------------------------------------------


@dataclass
class Need:
    full: float  # 全部載せ
    sequential: float  # 順に載せ替え
    offload: float  # 一部を逃がす
    ram: float  # 逃がす・CPU のときのメインメモリ(固定分 + 全部品)
    ram_gpu: float = 0.0  # 全部載せ・順に載せ替えのときのメインメモリ(固定分 + いちばん大きい段の部品)
    # CPU で動かすときに ComfyUI が fp32 に広げる分の材料(#39): 作業用メモリ・本体・VAE の重み(GiB)
    act: float = 0.0
    main: float = 0.0
    vae: float = 0.0


def need(parts: Parts, act: float, c: Coefficients) -> Need:
    w = c.weight_factor
    act = max(0.0, act)
    full = c.overhead_gb + parts.total * w + act
    # 作業用メモリが要るのは生成と仕上げの段(文章の読み取りは小さいので数えない)
    stage_act = {"encode": 0.0, "denoise": act, "decode": act, "other": 0.0}
    # 順に載せ替える・逃がすときの生成の段は、本体の組に分けていれば一番大きい組(ComfyUI が前の組を降ろす)
    peak = {st: (parts.denoise_peak if st == "denoise" else parts.stage(st)) for st in STAGES}
    seq = c.overhead_gb + max(peak[st] * w + stage_act[st] for st in STAGES)
    off = c.overhead_gb + max(peak[st] * w * c.offload_fraction + stage_act[st] for st in STAGES)
    # メインメモリ: GPU に載せて動くときは、部品を 1 つずつ読み込んで GPU へ送る。ComfyUI は safetensors を
    # メモリに割り当てて(mmap)読むので、使い終わった部品のページは OS が捨てられる(仮定。実測は無い)。
    # 逃がすときは逃がした重みがメインメモリに居続けるので、全部品を数える(安全側)
    ram_all = c.ram_extra_gb + parts.total * w
    ram_gpu = c.ram_extra_gb + parts.largest * w
    return Need(
        full=full,
        sequential=min(seq, full),
        offload=min(off, seq, full),
        ram=ram_all,
        ram_gpu=ram_gpu,
        act=act,
        main=parts.denoise * w,
        vae=parts.decode * w,
    )


def raw_estimate(entry: ModelEntry, s: Settings, c: Coefficients) -> tuple[float, float, float]:
    """(全部載せの VRAM, 逃がしたときの VRAM, 逃がしたときのメインメモリ) を GiB で返す。
    実測を使わない式だけの値。"""
    parts = apply_precision(components(entry), entry, s)
    n = need(parts, activation_gb(entry, s, c), c)
    return n.full, n.offload, n.ram


def is_censored(m: Measurement) -> bool:
    """実測が GPU の上限に張り付いている(本当の全部載せの量はこれ以上)か。"""
    if (m.offload or "").lower() in ("lowvram", "novram", "cpu"):
        return True
    return m.peak_vram_gb >= CENSOR_RATIO * m.gpu_vram_gb


def usable(m: Measurement) -> bool:
    """見積もりに使える実測か。OOM・エラーで終わったもの、標準でない設定(独自のワークフローなど)は使わない。
    ほかの利用者の実測(origin が community)は、同じ GPU・設定の 2 件以上がそろったもの(consistent)だけ使う
    (1 件だけ・ばらつきの大きいものは、測り違いを見分けられないため。
    docs/accuracy.md「ほかの利用者の実測」)。"""
    if is_community(m) and not m.consistent:
        return False
    return (m.outcome or "ok") == "ok" and m.standard is not False


def is_community(m: Measurement) -> bool:
    return getattr(m, "origin", None) == ORIGIN_COMMUNITY


# 「これ以上」としてだけ使う測り方。torch_reserved は ComfyUI のプロセスの torch の確保量で、CUDA の文脈などを
# 含まないので、本当の必要量より小さい(#34 のレビュー、オーナーの方針 2026-10-21)
LOWER_BOUND_METRICS = ("torch_reserved",)
# ほかの利用者の実測が必要量を上げてよい上限(それが無いときの見積もりに対する割合)。人が確かめた(approved)
# まとめだけが超えられる。#34 のオーナーの方針(書き換えた提出で段階を下げる攻撃を防ぐ)
COMMUNITY_RAISE_CAP = 0.15


def _measured_ram(m: Measurement, ram_est: float) -> float:
    """実測のメインメモリ。確かめていないほかの利用者の実測は、見積もりの +COMMUNITY_RAISE_CAP まで。"""
    v = m.peak_ram_gb or 0.0
    if is_community(m) and not getattr(m, "approved", False):
        v = min(v, ram_est * (1 + COMMUNITY_RAISE_CAP))
    return v


def lower_bound_only(m: Measurement) -> bool:
    """実測を「必要量はこれ以上」としてだけ使うか(式の値より小さければ使わない)。

    ほかの利用者の実測(提出は書き換えられるので、必要量を下げる・段階を上げる・点数を上げる向きには使わない)と、
    torch_reserved の実測(CUDA の文脈を含まない)。"""
    return is_community(m) or getattr(m, "vram_metric", None) in LOWER_BOUND_METRICS


def _measured_precision(m: Measurement, entry: ModelEntry) -> str | None:
    return norm_precision(m.settings.precision) or source_precision(entry)


def _settings_match(a: Settings, b: Settings, entry: ModelEntry | None = None) -> bool:
    """実測の設定 a と判定の設定 b が同じか(解像度・コマ数・長さ・枚数・精度)。"""
    keys = ("width", "height", "frames", "seconds")
    if any(getattr(a, k) != getattr(b, k) for k in keys):
        return False
    if max(1, a.batch or 1) != max(1, b.batch or 1):
        return False
    pa, pb = norm_precision(a.precision), norm_precision(b.precision)
    if entry is not None:
        src = source_precision(entry)
        pa, pb = pa or src, pb or src
    return pa == pb


def _precision_changed(m: Measurement, s: Settings, entry: ModelEntry) -> bool:
    want = norm_precision(s.precision)
    return bool(want) and want != _measured_precision(m, entry)


def pick_measurement(
    measurements: Iterable[Measurement], entry: ModelEntry, s: Settings | None = None
) -> Measurement | None:
    """そのモデルの実測を 1 つ選ぶ。設定が同じもの → 作者・この PC の実測(ほかの利用者の実測より先)→
    張り付いていないもの → 大きい GPU → 大きい値 の順(根拠の段階は models.BASIS_DETAILS)。"""
    cands = [m for m in measurements if m.model_id == entry.id and usable(m)]
    if not cands:
        return None

    def key(m: Measurement):
        same = s is not None and _settings_match(m.settings, s, entry)
        return (same, not is_community(m), not is_censored(m), m.gpu_vram_gb, m.peak_vram_gb)

    return max(cands, key=key)


_POSITIVE = ("width", "height", "frames", "seconds", "steps")


def settings_for(entry: ModelEntry, settings: Settings | None, base: Settings | None = None) -> Settings:
    """指定の無い項目を埋める。base(実測の設定)があればそれを、無ければ種類ごとの既定を使う。
    0 以下の解像度・コマ数・長さは指定が無いものとして扱う。"""
    src = base or DEFAULT_SETTINGS.get(entry.modality, Settings())
    merged = Settings(**vars(src))
    if settings is not None:
        for k, v in vars(settings).items():
            if v is None or (k in _POSITIVE and not v > 0):
                continue
            setattr(merged, k, v)
    merged.batch = max(1, merged.batch or 1)
    return merged


def _describe(entry: ModelEntry, s: Settings) -> str:
    bits = []
    if entry.modality in ("image", "video") and s.width and s.height:
        bits.append(f"{s.width}x{s.height}")
    if entry.modality == "video" and s.frames:
        bits.append(f"{s.frames} コマ")
    if entry.modality in ("music", "sfx", "tts") and s.seconds:
        bits.append(f"{s.seconds:g} 秒")
    if s.batch and s.batch > 1:
        bits.append(f"{s.batch} 枚")
    if s.precision:
        bits.append(s.precision)
    return "・".join(bits)


@dataclass
class _Model:
    """1 モデルについて、設定を変えて何度も見積もるための下ごしらえ。"""

    entry: ModelEntry
    c: Coefficients
    measurement: Measurement | None
    delta: float  # 実測に合わせて作業用メモリをずらす量(GiB)
    base: Settings
    parts: Parts
    # 下限としてだけ使う実測(lower_bound_only)が式の値(ほかの利用者の実測なら own の見積もり)より小さく、
    # その実測ではずらさなかった
    clamped: bool = False
    # ほかの利用者の実測を使うとき、それが無いときの下ごしらえ(精度を変えたときはこちらで見積もる)
    own: _Model | None = None
    # 確かめていないほかの利用者の実測が +COMMUNITY_RAISE_CAP を超えたので、そこまでにした
    capped: bool = False

    def parts_for(self, s: Settings) -> Parts:
        return apply_precision(replace(self.parts, notes=list(self.parts.notes)), self.entry, s)

    def need(self, s: Settings, prec: GpuPrecision | None = None) -> Need:
        if (
            self.own is not None
            and self.measurement is not None
            and _precision_changed(self.measurement, s, self.entry)
        ):
            return self.own.need(s, prec)  # 精度を変えたら、ほかの利用者の実測は使わない(own のまま)
        parts = self.parts_for(s)
        act = activation_gb(self.entry, s, self.c)
        if self.measurement is not None and not _precision_changed(self.measurement, s, self.entry):
            act += self.delta
        if prec is not None and (prec.act_factor != 1 or prec.vae_factor != 1):
            parts.decode *= prec.vae_factor
            act = max(0.0, act) * prec.act_factor
        return need(parts, act, self.c)


def _prepare(
    entry: ModelEntry, measurements: Iterable[Measurement], c: Coefficients, s: Settings | None = None
) -> _Model:
    measurements = list(measurements)
    parts = components(entry)
    m = None if parts.unknown else pick_measurement(measurements, entry, s)
    if m is None:
        return _Model(entry, c, None, 0.0, settings_for(entry, None), parts)
    at = settings_for(entry, m.settings)
    est = need(parts, activation_gb(entry, at, c), c).full
    delta = m.peak_vram_gb - est
    clamped = False
    if is_community(m):
        # ほかの利用者の実測は、それが無いときの見積もり(own)より必要量を下げない。判定する既定の設定も
        # own のまま(小さな設定の提出で既定を下げられないように)。#34 のオーナーの方針
        own = _prepare(entry, [x for x in measurements if not is_community(x)], c, s)
        floor = own.delta if own.measurement is not None else 0.0
        clamped = delta < floor
        delta = max(delta, floor)
        capped = False
        if not getattr(m, "approved", False):
            ceiling = floor + COMMUNITY_RAISE_CAP * max(0.0, est + floor)
            capped = delta > ceiling
            delta = min(delta, ceiling)
        return _Model(entry, c, m, delta, own.base, parts, clamped or capped, own, capped)
    if is_censored(m) or lower_bound_only(m):
        clamped = delta < 0 and lower_bound_only(m)
        delta = max(0.0, delta)  # 張り付いた実測・下限の実測は「これ以上」としてだけ使う
    return _Model(entry, c, m, delta, at, parts, clamped)


@dataclass(frozen=True)
class GpuPrecision:
    """判定に使う GPU で ComfyUI が選ぶ精度による、見積もりの直し(docs/hardware.md「精度」)。"""

    act_factor: float = 1.0  # 作業用メモリの倍率(fp32 で計算する GPU は 2)
    vae_factor: float = 1.0  # VAE(仕上げの段)の重みの倍率(fp32 で載せる GPU は、fp32 でない VAE を 2)
    notes: tuple[str, ...] = ()


def gpu_precision(entry: ModelEntry, system: SystemInfo, s: Settings) -> GpuPrecision:
    """ComfyUI の精度の選び方(comfy/model_management.py)で、見積もりが変わるところだけを直す。

    - 本体を fp32 で計算する場合は作業用メモリを 2 倍にする。作業用メモリは計算の精度の大きさに比例する
      (ComfyUI の memory_required・VAE の memory_used_decode が dtype_size を掛ける)。fp32 で計算するのは
      (1) 半精度を使わない GPU(Maxwell・一部の Pascal・GTX 16 系など。Gpu.compute_dtype が fp32)と、
      (2) bf16 を持たない GPU(Pascal・Volta・Turing)で、fp16 で動かせない系統(families.FAMILIES の fp16 が
      False: LTX・Qwen Image・Z-Image・HunyuanVideo・ACE-Step など)を動かすとき(unet_manual_cast が float32)
    - VAE を fp32 で載せる場合は VAE の重みを 2 倍にする(ファイルが fp32 なら同じ)。半精度を使わない GPU と、
      bf16 を持たない GPU で working_dtypes に fp16 が無い VAE(2D の KL・LTX など)。仕上げの作業用メモリは
      足りなければ ComfyUI がタイルに分けるので変えない
    - 本体の重みは変えない。ComfyUI は fp32 にしても `総量 × 0.88 − 最小の作業用メモリ` に収まるときだけ
      fp32 で載せ、収まらなければ元の 2 バイト(bf16・fp16)のまま計算だけ fp32 にする(unet_dtype)。
      VRAM が増えたときだけ使う量が増える動きなので、見積もりに入れると VRAM を増やして段階が下がる。
      注意として書く
    - fp8 の計算を持たない GPU(Ada より前)の fp8 の重み: 同じ理由で変えない(注意だけ)
    系統が分からないときは (2) を見積もりに入れず、bf16 を持たない GPU ではそう書く。
    GPU が無い・世代が分からないときは何も変えない。"""
    from .families import FAMILIES, VAES, family_of, vae_of

    g = system.best_gpu
    if g is None or system.vram_gb <= 0 or g.compute_dtype is None:
        return GpuPrecision()
    prec = norm_precision(s.precision) or source_precision(entry)
    notes: list[str] = []
    fam_key = getattr(entry, "comfy_family", None) or family_of(entry)
    fam = FAMILIES.get(fam_key or "")
    vae_key = getattr(entry, "comfy_vae", None) or vae_of(entry, fam_key)
    vae_kind = VAES.get(vae_key or "")
    gpu_fp32 = g.compute_dtype == "fp32"
    no_bf16 = g.bf16 is False
    # AMD の RDNA 2 以前は bf16 の計算を持たないが、ComfyUI は fp16 を持たない系統を bf16 で計算する
    # (should_use_bf16 の manual_cast が AMD_RDNA2_AND_OLDER_ARCH で True。docs/hardware.md)
    manual_bf16 = getattr(g, "vendor", "nvidia") == "amd"
    family_fp32 = no_bf16 and not gpu_fp32 and fam is not None and not fam.fp16 and not manual_bf16
    act = 2.0 if gpu_fp32 or family_fp32 else 1.0
    vae_fp32 = (gpu_fp32 or (no_bf16 and vae_kind is not None and not vae_kind.fp16)) and (
        vae_kind is None or vae_kind.key != "fp32"
    )
    vae_files = [f for f in entry.files if _role_of(f) == "vae"]
    file_fp32 = bool(vae_files) and all(_name_precision(_path(f)[1]) == "fp32" for f in vae_files)
    vae = 2.0 if vae_fp32 and not file_fp32 else 1.0
    if gpu_fp32:
        notes.append(
            f"{g.name} では ComfyUI が半精度を使わず fp32 で計算するので、作業用メモリと VAE を 2 倍で"
            "見積もった(本体は VRAM に余裕があれば fp32 に広げて載せる。遅い)"
        )
    elif family_fp32:
        notes.append(
            f"このモデルの系統({fam.comfy_class})は ComfyUI が fp16 で動かさない(supported_models の "
            f"{fam.source})ので、bf16 を持たない {g.name} では fp32 で計算する。"
            "作業用メモリを 2 倍で見積もった"
            "(本体は VRAM に余裕があれば fp32 で載せる。遅い)"
        )
        if fam.note:
            notes.append(fam.note)
    elif no_bf16 and fam is not None and not fam.fp16 and manual_bf16:
        notes.append(
            f"{g.name} は bf16 の計算に対応しないが、ComfyUI はこの系統({fam.comfy_class})を bf16 で動かす"
            "(AMD の RDNA 2 以前の扱い。作業用メモリは変えていない)"
        )
    elif no_bf16 and fam is None:
        notes.append(
            "このモデルの系統(ComfyUI のどのモデルか)が分からないので、bf16 を持たない "
            f"{g.name} で fp32 で計算するかは見積もりに入れていない"
        )
    elif prec == "bf16" and no_bf16:
        notes.append(
            f"{g.name} は bf16 の計算に対応しないので、ComfyUI は bf16 の重みを fp16 で動かす(VRAM は同じ)"
        )
    if vae_fp32 and not gpu_fp32 and vae > 1:
        notes.append(
            f"VAE({vae_kind.source})は {g.name} では fp32 で載せるので、VAE の重みを 2 倍で見積もった"
            "(仕上げで VRAM が足りなければ ComfyUI がタイルに分けるので遅くなる)"
        )
    if prec == "fp8" and g.fp8 is False:
        notes.append(
            f"{g.name} は fp8 の計算に対応しない"
            + ("(Ada 以降が対応)" if getattr(g, "vendor", "nvidia") == "nvidia" else "")
            + "。ComfyUI は fp8 の重みを、fp16 に広げても "
            "VRAM に収まれば広げて使い(VRAM を多く使う)、収まらなければ fp8 のまま計算のたびに"
            "変換する(遅い)。見積もりは fp8 の大きさのまま"
        )
    return GpuPrecision(act, vae, tuple(notes))


def ram_limit(system: SystemInfo) -> tuple[float, float | None]:
    """(仮想メモリを使えば動くとみなすメインメモリの上限, その根拠にしたページファイル・スワップの最大)。
    ページファイルの大きさが決まっているとき(固定・無し・Linux のスワップ)は 搭載量 + その大きさ
    (ただし搭載量 × RAM_PAGEFILE_FACTOR まで)。システム管理で伸びる・分からないときは倍率だけで見る。"""
    total = system.ram_total_gb
    cap = total * RAM_PAGEFILE_FACTOR
    swap = system.swap_max_gb
    if system.swap_auto is False and swap is not None:
        return min(cap, total + max(0.0, swap)), swap
    return cap, None


def _unknown_estimate(entry: ModelEntry, parts: Parts, s: Settings) -> Estimate:
    return Estimate(
        vram_gb=0.0,
        vram_offload_gb=0.0,
        ram_gb=0.0,
        disk_gb=None,
        basis="estimated",
        fit=FIT_UNKNOWN,
        run_mode=RUN_UNKNOWN,
        reasons=[UNKNOWN_SIZE_REASON, *parts.notes],
        measured_on=None,
        vram_sequential_gb=None,
        settings={k: v for k, v in vars(s).items() if v is not None},
        limits=None,
        unknown_size=True,
    )


def estimate(
    entry: ModelEntry,
    system: SystemInfo,
    settings: Settings | None = None,
    measurements: Iterable[Measurement] = (),
    coefficients: Coefficients | None = None,
    *,
    with_limits: bool = False,
) -> Estimate:
    """1 モデル × 1 設定 × 1 PC の見積もり。settings を省くと、実測があれば実測の設定、無ければ既定の設定。"""
    c = coefficients or load_coefficients()
    measurements = list(measurements)
    unreviewed = [
        m
        for m in measurements
        if m.model_id == entry.id and is_community(m) and not getattr(m, "approved", False)
    ]
    if unreviewed:
        # 人が確かめていないほかの利用者の実測は、段階・動かし方を変えるなら使わない(#34 のオーナーの方針)
        own = [m for m in measurements if not any(m is u for u in unreviewed)]
        before = _estimate(entry, system, settings, own, c, with_limits)
        after = _estimate(entry, system, settings, measurements, c, with_limits)
        if (after.fit, after.run_mode) != (before.fit, before.run_mode):
            before.reasons.append(
                "ほかの利用者の実測(人が確かめていないもの)は、段階を変えるので使わなかった"
            )
            return before
        return after
    return _estimate(entry, system, settings, measurements, c, with_limits)


def _estimate(
    entry: ModelEntry,
    system: SystemInfo,
    settings: Settings | None,
    measurements: list[Measurement],
    c: Coefficients,
    with_limits: bool,
) -> Estimate:
    s = settings_for(entry, settings, _prepare(entry, measurements, c).base)
    prep = _prepare(entry, measurements, c, s)  # 設定が同じ実測があればそれを選び直す
    if prep.parts.unknown:
        return _unknown_estimate(entry, prep.parts, s)
    prec = gpu_precision(entry, system, s)
    n = prep.need(s, prec)
    reasons: list[str] = []
    basis, measured_on, detail = "estimated", None, "estimated"
    m = prep.measurement
    if m is not None:
        on = f"{m.gpu_name} ({m.gpu_vram_gb:g} GB)"
        if is_community(m):
            reasons.append(f"{on} の実測は、ほかの利用者の実測 {m.n_submissions or '?'} 件をまとめた中央値")
        g0 = system.best_gpu
        other_vendor = g0 is not None and g0.vendor != gpu_vendor(m.gpu_name)
        if system.vram_gb <= 0:
            reasons.append(f"GPU が無いので、{on} の実測は判定に使っていない")
        elif other_vendor and not is_community(m):
            # 別の会社の GPU の実測は、作業用メモリをずらすのに使うだけ(その GPU での実測ではない)
            detail = "calibrated"
            reasons.append(f"{on} の実測で補正した推定({_vendor_label(g0.vendor)} の GPU での実測は無い)")
        elif prec.act_factor != 1:
            detail = "calibrated"
            reasons.append(f"{on} の実測は半精度で計算する GPU のものなので、精度の違いを足した推定にした")
        elif _settings_match(m.settings, s, entry) and not _precision_changed(m, s, entry) and prep.clamped:
            # 下限としてだけ使う実測が式より小さい・上限を超えた: 値は実測ではないので basis は estimated
            detail = "measured_community" if is_community(m) else "calibrated"
            if prep.capped:
                reasons.append(
                    f"{on} の実測 {m.peak_vram_gb:.2f} GB は、それを使わない見積もりの"
                    f" +{COMMUNITY_RAISE_CAP:.0%} を超えるので、そこまでにした"
                    "(人が確かめた提出だけが超えられる)"
                )
            else:
                reasons.append(
                    f"{on} の実測 {m.peak_vram_gb:.2f} GB は下限としてだけ使い、"
                    "それを使わない見積もりのほうが大きいので、そちらの値にした"
                )
            if m.peak_ram_gb:
                n.ram_gpu = max(n.ram_gpu, _measured_ram(m, n.ram_gpu))
        elif _settings_match(m.settings, s, entry) and not _precision_changed(m, s, entry):
            basis, measured_on = "measured", on
            detail = "measured_community" if is_community(m) else "measured"
            if is_censored(m):
                reasons.append(
                    f"実測 {m.peak_vram_gb:.1f} GB は GPU の上限に張り付いた値(全部載せにはこれ以上要る)"
                )
            if m.peak_ram_gb:
                n.ram_gpu = max(n.ram_gpu, _measured_ram(m, n.ram_gpu))
        elif not _precision_changed(m, s, entry):
            detail = "calibrated"
            reasons.append(f"{on} の実測({_describe(entry, m.settings)})で補正した推定")
    desc = _describe(entry, s)
    if desc:
        reasons.append(f"設定: {desc}")
    parts = prep.parts_for(s)
    disk = parts.files_gb
    fit, mode, why, ram_need = _judge(n, system, prep.c, entry, parts.total)
    fit_from, measured_peak = "estimate", None
    promoted = _measured_level(m, entry, s, system, prec, fit)
    if promoted is not None:
        level, line = promoted
        fit_from, measured_peak = "measurement", m.peak_vram_gb
        ram_need = n.ram_gpu
        fit, mode, why = _ram_check(level, "gpu", [line], ram_need, system)
    elif m is not None and basis == "measured" and system.vram_gb > 0 and mode in ("gpu", "offload"):
        if m.gpu_vram_gb > system.vram_gb / MEASURED_VRAM_TOLERANCE:
            reasons.append(
                f"段階は見積もりで決めた(実測の GPU {m.gpu_vram_gb:g} GB は"
                f"この PC の {system.vram_gb:.1f} GB より大きい)"
            )
    g = system.best_gpu
    if g is not None and g.vendor != "nvidia" and system.vram_gb > 0 and mode in ("gpu", "offload"):
        label = _vendor_label(g.vendor)
        if fit == "perfect" and fit_from != "measurement":
            fit = "good"
            why = [*why, f"推定({label}・実測なし)なので、段階は good までにした"]
        elif fit_from != "measurement":
            why = [*why, f"推定({label}・実測なし)"]
    reasons.extend(prec.notes)
    free_short = False
    g = system.best_gpu
    if g is not None and g.vram_free_gb is not None and mode in ("gpu", "offload") and system.vram_gb > 0:
        _lvl, how = gpu_choice(
            n.full, n.sequential, n.offload, system.vram_gb, allow_offload=not shared_memory(system)
        )
        use = {"full": n.full, "seq": n.sequential, "offload": n.offload}[how]
        if fit_from == "measurement":
            use = measured_peak
        if use > g.vram_free_gb:
            free_short = True
            others = f"(ほかのアプリが {g.vram_used_gb:.1f} GB 使用中)" if g.vram_used_gb is not None else ""
            why = [
                *why,
                f"今の空き VRAM は {g.vram_free_gb:.1f} GB{others}で、要る {use:.1f} GB に足りない。"
                "ほかのアプリ(ブラウザ・ゲームなど)を閉じれば動く",
            ]
    est = Estimate(
        vram_gb=round(n.full, 2),
        vram_offload_gb=round(n.offload, 2),
        ram_gb=round(ram_need, 2),
        disk_gb=round(disk, 2) if disk else None,
        basis=basis,
        fit=fit,
        run_mode=mode,
        reasons=why + reasons + parts.notes,
        measured_on=measured_on,
        vram_sequential_gb=round(n.sequential, 2),
        settings={k: v for k, v in vars(s).items() if v is not None},
        vram_free_short=free_short,
        basis_detail=detail,
        fit_from=fit_from,
        measured_peak_vram_gb=measured_peak,
    )
    if with_limits:
        est.limits = limits(prep, system, s)
    return est


# 実測の GPU の VRAM(m.gpu_vram_gb は公称の 16 など)と nvidia-smi の総量(15.9 など)の違いを許す割合(#26)
MEASURED_VRAM_TOLERANCE = 0.97


def gpu_vendor(name: str) -> str:
    """GPU 名から会社(実測の gpu_name は名前だけなので)。分からなければ nvidia(同梱の実測はすべて NVIDIA)。"""
    n = name.lower()
    if "radeon" in n or n.startswith("amd") or "instinct" in n:
        return "amd"
    if n.startswith("intel") or " arc" in f" {n}":
        return "intel"
    if n.startswith("apple"):
        return "apple"
    return "nvidia"


def _vendor_label(vendor: str) -> str:
    return {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}.get(vendor, vendor)


def _measured_level(
    m: Measurement | None, entry: ModelEntry, s: Settings, system: SystemInfo, prec: GpuPrecision, fit: str
) -> tuple[str, str] | None:
    """同じ設定で成功した実測があり、その GPU の VRAM がこの PC の VRAM 以下なら、
    実測で決めた段階と理由(#26)。

    実測が「GPU で動いた」ことは、それ以上の VRAM の同じ会社・同じ精度の扱いの GPU でも動く根拠になる。
    段階は 実測の VRAM ÷ この PC の VRAM の帯(FIT_BANDS)で、少なくとも good。上限に張り付いた実測
    (動的 VRAM が空きを使い切った)・下限としてだけ使う実測(torch_reserved)は
    perfect の根拠にしない(good まで)。
    見積もりの段階のほうが良ければ None。ほかの利用者の実測は段階を上げる向きに使わない(#34 の方針)。"""
    if m is None or is_community(m) or not usable(m):
        return None
    g = system.best_gpu
    vram = system.vram_gb
    if g is None or vram <= 0 or g.vendor != gpu_vendor(m.gpu_name) or prec.act_factor != 1:
        return None
    if not _settings_match(m.settings, s, entry) or _precision_changed(m, s, entry):
        return None
    if vram < m.gpu_vram_gb * MEASURED_VRAM_TOLERANCE:
        return None
    band = fit_band(m.peak_vram_gb, vram)
    level = "perfect" if band == "perfect" and not (is_censored(m) or lower_bound_only(m)) else "good"
    if _LEVELS.index(level) >= _LEVELS.index(fit):
        return None
    on = f"{m.gpu_name} ({m.gpu_vram_gb:g} GB)"
    pinned = "(GPU の上限に張り付いた値: 動的 VRAM が空きを使った)" if is_censored(m) else ""
    return level, (
        f"{on} で同じ設定の実測 {m.peak_vram_gb:.1f} GB{pinned}・成功があり、この PC の VRAM {vram:.1f} GB は"
        "それ以上なので、GPU で動くと判定した(段階は実測で決めた。見積もりは目安)"
    )


def _ram_check(fit: str, mode: str, reasons: list[str], need_gb: float, system: SystemInfo):
    """メインメモリの判定。足りなければ、ram_limit(ページファイルの大きさ、分からなければ倍率
    RAM_PAGEFILE_FACTOR)までは marginal(仮想メモリで遅い)、超えたら too_tight。"""
    ram_total = system.ram_total_gb
    if not ram_total or need_gb <= ram_total:
        return fit, mode, reasons
    limit, swap = ram_limit(system)
    if need_gb > limit:
        pf = f"、ページファイルは最大 {swap:.1f} GB" if swap is not None else ""
        return (
            "too_tight",
            "none",
            [f"メモリ不足: 約 {need_gb:.1f} GB 必要(この PC は {ram_total:.1f} GB{pf})", *reasons],
        )
    worse = fit if fit in ("marginal", "too_tight") else "marginal"
    return (
        worse,
        mode,
        [
            *reasons,
            f"メモリが少し足りない: 約 {need_gb:.1f} GB 必要(この PC は {ram_total:.1f} GB)。"
            "足りない分は仮想メモリ(ページファイル)を使うので遅い",
        ],
    )


def _judge(
    n: Need, system: SystemInfo, c: Coefficients, entry: ModelEntry, weights_gb: float | None = None
) -> tuple[str, str, list[str], float]:
    """(段階, 動かし方, 理由, その動かし方で要るメインメモリ)。

    GPU に載らない(逃がしても足りない・メモリが足りない)ときは、GPU が無いときの判定(CPU)とも比べて
    良いほうを取る。小さな GPU を足すと、CPU で動くものまで動かない判定になるのを防ぐ
    (VRAM を増やして段階が下がらない)。"""
    vram = system.vram_gb
    if vram <= 0:
        return _judge_cpu(n, system, entry, weights_gb)
    gpu = _judge_gpu(n, system, c)
    if gpu[0] != "too_tight":
        return gpu
    cpu = _judge_cpu(n, system, entry, weights_gb)
    if cpu[0] != "too_tight":
        return cpu[0], cpu[1], [*cpu[2], f"GPU では動かない: {gpu[2][0]}"], cpu[3]
    return gpu


def cpu_family(entry: ModelEntry):
    """CPU で動く系統(families.CPU_FAMILIES)なら、その行。種類が合わなければ None(#39)。"""
    from .families import CPU_FAMILIES, family_of

    fam = getattr(entry, "comfy_family", None) or family_of(entry)
    run = CPU_FAMILIES.get(fam or "")
    if run is None or entry.modality not in run.modalities:
        return None
    if run.exclude and re.search(run.exclude, f"{entry.id} {entry.name}".lower()):
        return None
    return run


def _judge_cpu(
    n: Need, system: SystemInfo, entry: ModelEntry, weights_gb: float | None = None
) -> tuple[str, str, list[str], float]:
    """GPU を使わない判定。小さく CPU で動く種類と、CPU で動く系統(families.CPU_FAMILIES)だけ marginal。"""
    w = weights_gb if weights_gb is not None else (entry.total_size_bytes or 0) / GIB
    why = "GPU なし" if system.vram_gb <= 0 else "GPU に載らない"
    run = cpu_family(entry)
    if run is not None and w > 0:
        # ComfyUI は CPU では本体と VAE を fp32 で持ち、fp32 で計算する(families.CPU_FAMILIES の説明)。
        # 2 バイトの重みは 2 倍、作業用メモリも 2 倍にする
        two_byte = source_precision(entry) in ("fp16", "bf16")
        ram = n.ram + (n.main + n.vae if two_byte else n.vae) + 2 * max(0.0, n.act)
        reasons = [
            f"{why}: CPU で動く系統({run.family})なので CPU で動く見込み(遅い"
            + (f"。{run.speed}" if run.speed else "")
            + f")。ComfyUI は CPU では本体と VAE を fp32 で持つので、メインメモリを約 {ram:.1f} GB とみた",
            f"CPU で動く根拠: {run.evidence}",
        ]
        fit, mode, reasons = _ram_check("marginal", "cpu", reasons, ram, system)
        return fit, mode, reasons, ram
    if 0 < w <= CPU_MAX_WEIGHTS_GB and entry.modality in CPU_MODALITIES:
        fit, mode, reasons = _ram_check("marginal", "cpu", [f"{why}: CPU で動く見込み(遅い)"], n.ram, system)
        return fit, mode, reasons, n.ram
    return (
        "too_tight",
        "none",
        ["GPU なし: このモデルは GPU が必要(CPU で動く系統・小さなモデルではない)"],
        n.ram,
    )


# 段階は、動かし方で使う置き場所(GPU なら VRAM)の使用率で決める
# (llmfit の docs/how-it-works.md「Fit analysis」にならう。2026-10-20 に利用者が決めた):
# 60% 以下 perfect / 85% 以下 good / 98% 以下 marginal / それより上 too_tight。
# 動かし方ごとの上限: 全部 GPU は上限なし、部品を順に載せ替える(GPU~)は good まで、
# メインメモリへ逃がす(GPU+RAM)と CPU は marginal まで
FIT_BANDS = ((0.60, "perfect"), (0.85, "good"), (0.98, "marginal"))
_LEVELS = ("perfect", "good", "marginal", "too_tight")


def fit_band(used: float, available: float) -> str:
    if available <= 0:
        return "too_tight"
    u = used / available
    for limit, level in FIT_BANDS:
        if u <= limit:
            return level
    return "too_tight"


def _cap(level: str, top: str) -> str:
    return _LEVELS[max(_LEVELS.index(level), _LEVELS.index(top))]


def gpu_choice(
    full: float, sequential: float | None, offload: float, vram: float, *, allow_offload: bool = True
) -> tuple[str, str]:
    """(段階, 載せ方)。載せ方は full(全部 GPU)/ seq(部品を順に載せ替え)/ offload(逃がす)。
    一番良い段階の載せ方を選ぶ(同じなら full → seq → offload)。表の VRAM の値もこれで決める。
    allow_offload=False(メインメモリと共有する GPU。逃がす先が同じメモリ)なら逃がす載せ方は使わない。"""
    seq = full if sequential is None else sequential
    options = [
        (fit_band(full, vram), "full"),
        (_cap(fit_band(seq, vram), "good"), "seq"),
    ]
    if allow_offload:
        options.append((_cap(fit_band(offload, vram), "marginal"), "offload"))
    return min(options, key=lambda o: _LEVELS.index(o[0]))


def shared_memory(system: SystemInfo | None) -> bool:
    """判定に使う GPU がメインメモリと共有する(内蔵 GPU・APU・Apple silicon)か(0.7.0、#22)。"""
    g = system.best_gpu if system is not None else None
    return g is not None and getattr(g, "memory", "dedicated") in ("shared", "unified")


def _judge_gpu(n: Need, system: SystemInfo, c: Coefficients) -> tuple[str, str, list[str], float]:
    vram = system.vram_gb
    shared = shared_memory(system)
    fit, how = gpu_choice(n.full, n.sequential, n.offload, vram, allow_offload=not shared)
    pct = {
        k: round(100 * round(v, 2) / vram)
        for k, v in (("full", n.full), ("seq", n.sequential), ("offload", n.offload))
    }
    if fit == "too_tight":
        if shared:
            msg = (
                f"GPU のメモリ不足: 部品を順に載せ替えても {n.sequential:.1f} GB 必要"
                f"(GPU が使えるのは {vram:.1f} GB、{pct['seq']}%)"
            )
        else:
            msg = (
                f"VRAM 不足: 逃がしても {n.offload:.1f} GB 必要(この PC は {vram:.1f} GB、{pct['offload']}%)"
            )
        return "too_tight", "none", [msg], n.ram
    if how == "full":
        mode = "gpu"
        reasons = [f"VRAM {n.full:.1f} / {vram:.1f} GB({pct['full']}%): 全部載る"]
    elif how == "seq":
        mode = "gpu"
        reasons = [
            f"VRAM {n.sequential:.1f} / {vram:.1f} GB({pct['seq']}%): 部品を順に載せ替えれば載る"
            f"(全部載せは {n.full:.1f} GB。段階は good まで)"
        ]
    else:
        mode = "offload"
        reasons = [
            f"VRAM {n.full:.1f} GB 必要 > {vram:.1f} GB: 重みの一部をメインメモリに逃がせば動く"
            f"(遅い。逃がすと {n.offload:.1f} GB、{pct['offload']}%。段階は marginal まで)"
        ]
    ram_need = n.ram_gpu if mode == "gpu" else n.ram
    g = system.best_gpu
    if g is not None and getattr(g, "memory", "") == "unified":
        # Apple silicon は GPU の分も同じ統合メモリから使うので、GPU に載せる量 + 固定分(Python・ComfyUI)を
        # 搭載量と比べる。内蔵 GPU・APU(shared)は、BIOS で取り分けた VRAM が搭載量に入らないので
        # これまでどおり(固定分 + いちばん大きい段の部品)
        use = n.full if how == "full" else n.sequential
        ram_need = max(ram_need, use + c.ram_extra_gb)
    fit, mode, reasons = _ram_check(fit, mode, reasons, ram_need, system)
    return fit, mode, reasons, ram_need


def judge(
    full: float,
    offload: float,
    ram: float,
    system: SystemInfo,
    c: Coefficients,
    entry: ModelEntry,
    *,
    sequential: float | None = None,
    ram_gpu: float | None = None,
) -> tuple[str, str, list[str]]:
    """4 段階の判定。perfect=全部載って余裕あり / good=全部、または段ごとに全部載る /
    marginal=重みの一部をメインメモリに逃がせば動く(遅い)・CPU・メモリが少し足りない / too_tight=動かない。
    ram は逃がす・CPU のとき、ram_gpu は GPU に載せて動くときのメインメモリ(省くと ram)。"""
    n = Need(
        full=full,
        sequential=full if sequential is None else sequential,
        offload=offload,
        ram=ram,
        ram_gpu=ram if ram_gpu is None else ram_gpu,
    )
    fit, mode, reasons, _ram = _judge(n, system, c, entry)
    return fit, mode, reasons


# --- 限界(この PC で good になる最大の設定) ----------------------------------------------------

GOOD = ("perfect", "good")


def _ok(prep: _Model, system: SystemInfo, s: Settings) -> bool:
    n = prep.need(s, gpu_precision(prep.entry, system, s))
    fit, _mode, _, _ram = _judge(n, system, prep.c, prep.entry, prep.parts.total)
    return fit in GOOD


def _largest(candidates: Iterable, ok) -> object | None:
    best = None
    for v in candidates:  # 必要量は設定に対して単調に増えるので、最初に外れたところで止める
        if not ok(v):
            break
        best = v
    return best


def _scaled(s: Settings, long_side: int) -> Settings:
    w, h = s.width or 1024, s.height or 1024
    k = long_side / max(w, h)
    out = Settings(**vars(s))
    out.width = max(64, round(w * k / 16) * 16)
    out.height = max(64, round(h * k / 16) * 16)
    return out


# 調べる範囲の上限(これを超える設定は学習の範囲外のモデルが多い)。上限まで good なら at_cap に名前を入れる
LIMIT_CAPS = {"image_side": 2048, "video_side": 1280, "frames": 241, "seconds": 600}


def limits(prep: _Model, system: SystemInfo, s: Settings) -> dict | None:
    """この PC で good 以上になる最大の解像度(長辺)・長さ。縦横比とほかの設定は s のまま。
    どの値も、それより小さくしても good にならないときは None。大きさが分からないモデルは None。"""
    if prep.parts.unknown:
        return None
    mod = prep.entry.modality
    out: dict = {"level": "good"}
    caps: list[str] = []

    def with_(**kw) -> Settings:
        t = Settings(**vars(s))
        for k, v in kw.items():
            setattr(t, k, v)
        return t

    def put(key: str, values: range, ok) -> None:
        v = _largest(values, ok)
        out[key] = v
        if v is not None and v == values[-1]:
            caps.append(key)

    if mod in ("image", "video"):
        cap = LIMIT_CAPS["image_side" if mod == "image" else "video_side"]
        put("max_side_px", range(256, cap + 1, 32), lambda px: _ok(prep, system, _scaled(s, px)))
    if mod == "video":  # Wan などは 4n+1 コマ
        put("max_frames", range(5, LIMIT_CAPS["frames"] + 1, 4), lambda f: _ok(prep, system, with_(frames=f)))
    if mod in ("music", "sfx", "tts"):
        put(
            "max_seconds",
            range(5, LIMIT_CAPS["seconds"] + 1, 5),
            lambda sec: _ok(prep, system, with_(seconds=float(sec))),
        )
    if len(out) == 1:
        return None
    out["at_cap"] = caps
    return out


def limits_for(
    entry: ModelEntry,
    system: SystemInfo,
    settings: Settings | None = None,
    measurements: Iterable[Measurement] = (),
    coefficients: Coefficients | None = None,
) -> dict | None:
    c = coefficients or load_coefficients()
    measurements = list(measurements)
    s = settings_for(entry, settings, _prepare(entry, measurements, c).base)
    return limits(_prepare(entry, measurements, c, s), system, s)
