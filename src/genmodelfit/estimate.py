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

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from .db import bundled_text
from .i18n import tr
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

UNKNOWN_SIZE_REASON = "大きさが分からないため判定できません"  # ja の文(表示は unknown_size_reason())


def unknown_size_reason() -> str:
    return tr(UNKNOWN_SIZE_REASON, "Cannot judge: the size is unknown")


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
# 内訳の説明は (日本語, 英語)
CHECKPOINT_SPLITS: list[tuple[str, tuple[float, float, float], tuple[str, str]]] = [
    (
        r"sd_xl|sdxl",
        (0.740, 0.236, 0.024),
        (
            "SDXL(UNet 4.79 GB / CLIP-L+bigG 1.53 GB / VAE 0.16 GB)",
            "SDXL (UNet 4.79 GB / CLIP-L+bigG 1.53 GB / VAE 0.16 GB)",
        ),
    ),
    (
        r"ace_step_1\.5",
        (0.478, 0.489, 0.034),
        (
            "ACE-Step 1.5(DiT 4.46 GB / 言語モデル等 4.56 GB / VAE 0.32 GB)",
            "ACE-Step 1.5 (DiT 4.46 GB / language model etc. 4.56 GB / VAE 0.32 GB)",
        ),
    ),
    (
        r"hunyuan3d-dit-v2",
        (0.452, 0.461, 0.087),
        (
            "Hunyuan3D 2.0(DiT 2.08 GB / 画像エンコーダ 2.11 GB / VAE 0.40 GB)",
            "Hunyuan3D 2.0 (DiT 2.08 GB / image encoder 2.11 GB / VAE 0.40 GB)",
        ),
    ),
    (
        r"hunyuan_3d_v2\.1",
        (0.828, 0.083, 0.089),
        (
            "Hunyuan3D 2.1(DiT 5.68 GB / 画像エンコーダ 0.57 GB / VAE 0.61 GB)",
            "Hunyuan3D 2.1 (DiT 5.68 GB / image encoder 0.57 GB / VAE 0.61 GB)",
        ),
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
        notes.append(
            tr(
                f"ComfyUI で使わない形式のファイル {skipped_alt} 個(onnx など)は数えていない",
                f"Not counted: {skipped_alt} file(s) in formats ComfyUI does not use (onnx etc.)",
            )
        )

    root_main = [f for f in cands if not f.directory and _path(f)[0] is None and _role_of(f) == "main"]
    in_folder = [f for f in cands if not f.directory and _path(f)[0] is not None]
    if root_main and in_folder:
        if any(_known_all_in_one(f) for f in root_main):
            drop = set(map(id, in_folder))
            notes.append(
                tr(
                    "全部入りの単一ファイルと diffusers 形式の両方があるので、単一ファイルだけ数えた",
                    "Both an all-in-one single file and the diffusers format exist; counted the single file "
                    "only",
                )
            )
        else:
            at_root = {_role_of(f) for f in cands if not f.directory and _path(f)[0] is None}
            same = at_root & {"main", "vae"}
            drop = {id(f) for f in in_folder if _role_of(f) in same}
            if drop:
                notes.append(
                    tr(
                        "単一ファイルと diffusers 形式の両方がある部品(本体・VAE)は、単一ファイルを数えた",
                        "Parts with both a single file and the diffusers format (model, VAE): counted the "
                        "single file",
                    )
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
            tr(
                f"同じ部品の別の形式・精度のファイル {dropped} 個は数えていない"
                "(.safetensors・fp16 / bf16 を優先して 1 組だけ数えた)",
                f"Not counted: {dropped} file(s) of the same part in another format or precision"
                " (counted one set, preferring .safetensors and fp16 / bf16)",
            )
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
            return split, tr(*label)
    return GENERIC_SPLIT, None


def _fallback_size(entry: ModelEntry, p: Parts) -> int:
    """ファイルの大きさが使えないときの大きさ(バイト)。
    テンプレートの大きさ → パラメータ数 × 精度。無ければ 0。"""
    t = entry.template.size_bytes if entry.template else None
    if isinstance(t, int) and t > 0:
        p.notes.append(
            tr(
                "ファイルごとの大きさが分からないので、テンプレートの大きさを使った",
                "File sizes are unknown; used the template size",
            )
        )
        return t
    prec = norm_precision(entry.precision)
    if entry.params_b and entry.params_b > 0 and prec in PRECISION_BYTES:
        p.notes.append(
            tr(
                "ファイルの大きさが無いので、パラメータ数と精度から重みを求めた",
                "No file sizes; computed the weights from the parameter count and precision",
            )
        )
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
            p.notes.append(
                tr(
                    UNKNOWN_SIZE_REASON + "(本体のファイルが一覧に無く、全体の大きさも無い)",
                    "Cannot judge: the size is unknown (the main model file is not listed and there is no "
                    "total size)",
                )
            )
            return p
        main, enc, vae = GENERIC_SPLIT
        p.denoise, p.encode, p.decode = size * main / GIB, size * enc / GIB, size * vae / GIB
        p.files_gb = size / GIB
        p.notes.append(
            tr(
                "本体のファイルが一覧に無いので、テンプレートの大きさを下限にして一般的な割合で分けた",
                "The main model file is not listed; split the template size (as a lower bound) by typical "
                "ratios",
            )
        )
        return p
    if not files or not all(_size_ok(f) for f in files):
        size = _fallback_size(entry, p)
        if size <= 0:
            p.unknown = True
            p.notes.append(
                tr(
                    UNKNOWN_SIZE_REASON + "(ファイルの大きさもテンプレートの大きさも無い)",
                    "Cannot judge: the size is unknown (no file sizes and no template size)",
                )
            )
            return p
        main, enc, vae = GENERIC_SPLIT
        p.denoise, p.encode, p.decode = size * main / GIB, size * enc / GIB, size * vae / GIB
        p.files_gb = size / GIB
        p.notes.append(
            tr(
                "部品の内訳が無いので、一般的な割合(本体 75%・エンコーダ 20%・VAE 5%)で分けた",
                "No breakdown of the parts; split by typical ratios (model 75%, encoders 20%, VAE 5%)",
            )
        )
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
                tr(f"全部入りの {f.name} を分けた: {label}", f"Split the all-in-one {f.name}: {label}")
                if label
                else tr(
                    f"全部入りの {f.name} を一般的な割合(本体 75%・エンコーダ 20%・VAE 5%)で分けた",
                    f"Split the all-in-one {f.name} by typical ratios (model 75%, encoders 20%, VAE 5%)",
                )
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
    if how == "workflow":
        src = tr("ワークフローのつながり", "the workflow links")
    else:
        src = tr("ファイル名の high_noise / low_noise", "high_noise / low_noise in the file names")
    p.notes.append(
        tr(
            f"本体はサンプラーごとに {len(sets)} 組を順に使う({src})。ComfyUI は次の組を載せるとき前の組を降"
            "ろ"
            "す"
            f"({EXPERT_SOURCE})ので、順に載せ替え・逃がすときは 1 組ずつ数えた"
            f"(一番大きい組 {max(p.experts):.1f} GB、生成の段の全部は {p.denoise:.1f} GB)",
            f"The model uses {len(sets)} sets in turn, one per sampler ({src}). ComfyUI unloads the previous "
            "set"
            f" before loading the next ({EXPERT_SOURCE}), so one set at a time was counted for sequential"
            f" loading and offloading (largest set {max(p.experts):.1f} GB, whole denoising stage"
            f" {p.denoise:.1f} GB)",
        )
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
        parts.notes.append(
            tr(f"精度 {want} は知らない値なので換算していない", f"Unknown precision {want}; not converted")
        )
        return parts
    if have not in PRECISION_BYTES:
        parts.notes.append(
            tr(
                f"元の精度が分からないので、精度 {want} の換算をしていない",
                f"The original precision is unknown; not converted to {want}",
            )
        )
        return parts
    if have != want:
        ratio = PRECISION_BYTES[want] / PRECISION_BYTES[have]
        parts.denoise *= ratio
        parts.experts = [x * ratio for x in parts.experts]
        parts.notes.append(
            tr(
                f"本体を {have} → {want} に換算した(×{ratio:.2f})",
                f"Converted the model {have} → {want} (×{ratio:.2f})",
            )
        )
    return parts


# --- 作業用メモリ ---------------------------------------------------------------------------------


def activation_features(modality: str, s: Settings, c: Coefficients) -> dict[str, float]:
    """作業用メモリの係数に掛ける量。calibrate も同じものを使う。負の値は 0 に切り上げる。"""
    mp = max(0.0, s.megapixels or 0.0)
    if not s.width and not s.height and modality in ("image", "video"):
        # 大きさの入力の無いテンプレート(no_size_input。入力の画像から作る)の設定は大きさを持たない。
        # 作業用メモリは今までどおり種類の既定の大きさで見積もる(#90)
        mp = DEFAULT_SETTINGS[modality].megapixels or 0.0
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
    ほかの利用者の実測(origin が community)は、まとめた行が ok なら 1 件でも使う(llmfit の measured_community
    と同じ。2026-10-22 のオーナーの決定。docs/accuracy.md「ほかの利用者の実測」)。"""
    return (m.outcome or "ok") == "ok" and m.standard is not False


def is_community(m: Measurement) -> bool:
    return getattr(m, "origin", None) == ORIGIN_COMMUNITY


# 「これ以上」としてだけ使う測り方。torch_reserved は ComfyUI のプロセスの torch の確保量で、CUDA の文脈などを
# 含まないので、本当の必要量より小さい(#34 のレビュー、オーナーの方針 2026-10-21)。
# 作者の実測では式より小さければ式の値にする。ほかの利用者の実測では物理的な下限(重み + 固定分)まで
# 下げてよい(2026-10-22)。段階を実測で決めるときは文脈の分を足して使う(TORCH_RESERVED_CONTEXT_GB。0.9.0)
LOWER_BOUND_METRICS = ("torch_reserved",)
# 実測の最大 VRAM の上限: GPU の総量 × 1.02 + 0.25 GiB(contrib validate の「ありえない値」と同じ。
# 総量の数え方の違い(公称 16 と 15.9 など)を許す)
PEAK_VRAM_RATIO = 1.02
PEAK_VRAM_SLACK_GB = 0.25
# ほかの利用者の実測を、設定の違う推定に広げる(calibrated)ときの 実測 ÷ 式の見積もり の範囲。
# llmfit の補正の倍率の範囲 [0.05, 3.0] に当たる。VRAM は速さより式に近いので、contrib validate が「式から
# 外れている」と知らせる幅(±40%)と同じにした。作者の実測の 実測 ÷ 式 は 0.64〜1.03(docs/accuracy.md)。
# 下側は物理的な下限(重み + 固定分。need が作業用メモリを 0 で切る)のほうが先に効くことが多い
COMMUNITY_DEVIATION = 0.40
COMMUNITY_CALIBRATION_RANGE = (1 - COMMUNITY_DEVIATION, 1 + COMMUNITY_DEVIATION)


def peak_vram_cap(m: Measurement) -> float:
    """その GPU で測れる最大 VRAM(これより大きい実測はありえない)。"""
    return m.gpu_vram_gb * PEAK_VRAM_RATIO + PEAK_VRAM_SLACK_GB


def lower_bound_only(m: Measurement) -> bool:
    """作者・この PC の実測を「必要量はこれ以上」としてだけ使うか(式の値より小さければ使わない)。
    torch_reserved の実測(CUDA の文脈を含まない)。ほかの利用者の実測は両方の向きに使う
    (物理的な範囲で抑える)。"""
    return not is_community(m) and getattr(m, "vram_metric", None) in LOWER_BOUND_METRICS


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


def gpu_key(name: str | None) -> str:
    """GPU 名の照合の鍵(ここだけが正本。PC 側の GPU 名にも実測の GPU 名にも使う)。
    大文字小文字・空白・記号をそろえ、(TM)・(R)・™・® を除き、"NVIDIA" "GeForce" の語も除く。
    NVIDIA の語を除くのは、同じ GPU が nvidia-smi・PyTorch では「NVIDIA GeForce RTX 3070」、lspci では
    「GeForce RTX 3070」、提出では NVIDIA を省いた名前になりうるため。NVIDIA の製品名(RTX 3070・RTX A6000・
    Quadro RTX 6000・RTX 6000 Ada など)は語を除いても重ならない。AMD・Intel の語は残す(「AMD Device 744c」と
    「Intel Device 744c」を同じにしない)。contrib.gpu_slug(提出のフォルダ名)も同じ。"""
    n = (name or "").lower()
    for word in ("nvidia", "geforce", "(tm)", "(r)", "™", "®"):
        n = n.replace(word, " ")
    return re.sub(r"[^a-z0-9]+", "-", n).strip("-")


# 名前だけでは GPU が決まらない汎用の名前(gpu_key の形)。同じ名前でも世代・大きさの違う GPU がある
# (「AMD Radeon(TM) Graphics」は Raphael の 2 CU から Phoenix の 12 CU まで、「Intel(R) Arc(TM) Graphics」は
# Meteor Lake・Arrow Lake・Lunar Lake)。名前では同じ GPU とみなさない(PCI の型番 ID があればそれで照合する)
GENERIC_GPU_KEYS = frozenset(
    {
        "",
        "gpu",
        "unknown",
        "mps",
        "amd-gpu",
        "amd-radeon-graphics",
        "radeon-graphics",
        "amd-graphics",
        "intel-gpu",
        "intel-graphics",
        "intel-arc-graphics",
        "arc-graphics",
        "intel-uhd-graphics",
        "intel-iris-xe-graphics",
        "intel-hd-graphics",
        "apple-gpu",
        # #90: ドライバが名前を知らない GPU(「NVIDIA Graphics Device」)・汎用の表示用の名前
        "graphics-device",
        "video-controller",
        "microsoft-basic-display-adapter",
        "display-adapter",
    }
)


def generic_gpu_name(name: str | None) -> bool:
    k = gpu_key(name)
    return k in GENERIC_GPU_KEYS or bool(re.fullmatch(r"\d+(-\d+)?-gb-gpu", k))  # --vram の仮の GPU


# 同じ名前で VRAM の量が違うデスクトップの NVIDIA GPU(gpu_key の形 → VRAM の GB)。
# nvidia-smi の名前は同じなので、
# ほかの利用者の実測が来たとき、その名前の各 VRAM の PC でも段階の変化を点検する(contrib.community_effect)。
# NVIDIA の公式の製品ページで確かめたものだけ(2026-10-22 確認。
# GTX 16 系は 1650・1630 が 4 GB、1660 系が 6 GB で
# VRAM の違いは無い。ノート用 GPU は別の名前なので入れない):
#   RTX 2060      6 / 12 GB   https://www.nvidia.com/en-us/geforce/graphics-cards/compare/
#   RTX 3050      6 / 8 GB    https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3050/
#   RTX 3060      8 / 12 GB   https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/
#   RTX 3080      10 / 12 GB  https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3080-3080ti/
#   RTX 4060 Ti   8 / 16 GB   https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4060-4060ti/
#   RTX 5060 Ti   8 / 16 GB   https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5060-family/
NVIDIA_VRAM_VARIANTS: dict[str, tuple[float, ...]] = {
    "rtx-2060": (6.0, 12.0),
    "rtx-3050": (6.0, 8.0),
    "rtx-3060": (8.0, 12.0),
    "rtx-3080": (10.0, 12.0),
    "rtx-4060-ti": (8.0, 16.0),
    "rtx-5060-ti": (8.0, 16.0),
}


# デスクトップの GeForce RTX 20〜50 系の公称の VRAM(GB。gpu_key の形 → 版の VRAM)。#90: ほかの利用者の提出の
# vram_gb が名前の GPU に無い量(RTX 3070 で 16 GB など)なら誤りにする。NVIDIA の比較ページ
# https://www.nvidia.com/en-us/geforce/graphics-cards/compare/ で 2026-10-27 に確かめた値。量の違う版は
# NVIDIA_VRAM_VARIANTS と同じ。ノート用・業務用は名前と量の対応が一つに決まらないので入れない
NVIDIA_NOMINAL_VRAM: dict[str, tuple[float, ...]] = {
    **NVIDIA_VRAM_VARIANTS,
    "rtx-5090": (32.0,),
    "rtx-5080": (16.0,),
    "rtx-5070-ti": (16.0,),
    "rtx-5070": (12.0,),
    "rtx-5060": (8.0,),
    "rtx-5050": (8.0,),
    "rtx-4090": (24.0,),
    "rtx-4080-super": (16.0,),
    "rtx-4080": (16.0,),
    "rtx-4070-ti-super": (16.0,),
    "rtx-4070-ti": (12.0,),
    "rtx-4070-super": (12.0,),
    "rtx-4070": (12.0,),
    "rtx-4060": (8.0,),
    "rtx-3090-ti": (24.0,),
    "rtx-3090": (24.0,),
    "rtx-3080-ti": (12.0,),
    "rtx-3070-ti": (8.0,),
    "rtx-3070": (8.0,),
    "rtx-3060-ti": (8.0,),
    "rtx-2080-ti": (11.0,),
    "rtx-2080-super": (8.0,),
    "rtx-2080": (8.0,),
    "rtx-2070-super": (8.0,),
    "rtx-2070": (8.0,),
    "rtx-2060-super": (8.0,),
}
# 提出の vram_gb(GiB。/system_stats の総量)と公称の VRAM の比の許容(1 枚の GPU の数え方の違いだけを許す)。
# CUDA が報告する総量は公称より少ない: Linux で 6 GB の版は 5934 MiB = 5.80 GiB(比 0.967)、8 GB は 7.79、
# 12 GB は 11.77、24 GB は 23.69。Windows は 7.99・15.92 など。下を 0.95 にした(版の違いの比は 0.83 以下)
NOMINAL_VRAM_TOLERANCE = (0.95, 1.03)


def nvidia_nominal_vram(name: str | None) -> tuple[float, ...]:
    """名前の表で分かる NVIDIA の GPU の公称の VRAM(GB)。表に無ければ空。名前に量が付いている
    (「RTX 3050 6GB」)ときは、表にその量の版があればその量だけ、無ければ空(contrib.gpu_size_ok が断る)。"""
    k = gpu_key(name)
    m = re.fullmatch(r"(.+)-(\d+)gb", k)
    if m and m.group(1) in NVIDIA_NOMINAL_VRAM:
        size = float(m.group(2))
        return (size,) if size in NVIDIA_NOMINAL_VRAM[m.group(1)] else ()
    return NVIDIA_NOMINAL_VRAM.get(k, ())


def nvidia_size_ok(name: str | None) -> bool:
    """名前に付いた量(「RTX 3060 16GB」)が、名前の表のその GPU の量のどれかか(表に無い GPU は True)。"""
    m = re.fullmatch(r"(.+)-(\d+)gb", gpu_key(name))
    return not (
        m and m.group(1) in NVIDIA_NOMINAL_VRAM and float(m.group(2)) not in NVIDIA_NOMINAL_VRAM[m.group(1)]
    )


def nvidia_vram_variants(name: str | None) -> tuple[float, ...]:
    """名前の表で分かる、同じ名前の NVIDIA GPU の公称の VRAM(GB)の版。表に無ければ空。"""
    return NVIDIA_VRAM_VARIANTS.get(gpu_key(name), ())


def _pci(v: str | None) -> str | None:
    v = (v or "").strip().lower()
    return v if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{4}", v) else None


def same_vram(m_vram: float | None, gpu: Any) -> bool:
    """実測の GPU の VRAM(公称)が、この PC の GPU の VRAM と近いか。専用の VRAM の GPU にだけ当てる。
    同じ名前・同じ型番 ID でも VRAM の違う GPU がある(RTX 4060 Ti の 8 GB と 16 GB、
    RTX 3060 の 8 GB と 12 GB、
    PCI ID 744c の RX 7900 XT・XTX・GRE)。VRAM が違うと同じ設定の最大 VRAM・段階が変わるので、同じ GPU の
    実測として使わない。近いとみなす幅は 実測の VRAM ÷ この PC の総量(小さいほう ÷ 大きいほう)が
    MEASURED_VRAM_TOLERANCE(0.97。公称 16 と nvidia-smi の 15.9 などの数え方の違い)以上。VRAM の違う版の
    比は実際には 0.83 以下(RTX 3080 の 10 GB と 12 GB、
    RX 7900 GRE の 16 GB と XT の 20 GB など)なので、間に十分な余裕がある。
    共有メモリ・統合メモリの GPU(内蔵 GPU・APU・Apple)は総量が PC のメインメモリで決まり、GPU の違いを
    表さないので比べない。VRAM が分からない(None)ときも比べない(これまでどおり)。"""
    if m_vram is None or getattr(gpu, "memory", "dedicated") != "dedicated":
        return True
    mine = getattr(gpu, "vram_total_gb", None)
    if not isinstance(m_vram, (int, float)) or not m_vram > 0 or not mine or not mine > 0:
        return True
    return min(m_vram, mine) / max(m_vram, mine) >= MEASURED_VRAM_TOLERANCE


def same_gpu(m_name: str | None, m_pci: str | None, gpu: Any, m_vram: float | None = None) -> bool:
    """実測の GPU(名前・PCI の型番 ID・VRAM)が、この PC の GPU(models.Gpu)と同じか。両方に型番 ID があれば
    ID で、無ければ gpu_key の名前で比べる(汎用の名前は同じとみなさない)。Apple はチップ名(Apple M2 Pro)で
    比べる。m_vram を渡したときは、さらに VRAM が近いこと(same_vram)を求める(0.9.1)。"""
    if gpu is None:
        return False
    a, b = _pci(m_pci), _pci(getattr(gpu, "pci_id", None))
    same = a == b if a and b else same_gpu_name(m_name, getattr(gpu, "name", None))
    return same and same_vram(m_vram, gpu)


def same_gpu_name(a: str | None, b: str | None) -> bool:
    return bool(a) and bool(b) and gpu_key(a) == gpu_key(b) and not generic_gpu_name(a)


def files_fingerprint(entry: ModelEntry) -> str | None:
    """一覧のモデルのファイルの指紋(#90): ファイルの sha256(16 進の小文字)を並べ替えて改行でつないだ文字列の
    sha256。ファイルが無い・sha256 の無いファイルが 1 つでもあるときは None(比べない)。bench が実測に書き、
    一覧のファイルが変わった(上流が重みを差し替えた)ときに古い実測を使わないために比べる。"""
    shas = [(f.sha256 or "").strip().lower() for f in entry.files]
    if not shas or not all(re.fullmatch(r"[0-9a-f]{64}", s) for s in shas):
        return None
    return hashlib.sha256("\n".join(sorted(shas)).encode("ascii")).hexdigest()


def files_match(m: Measurement, entry: ModelEntry) -> bool:
    """実測が今の一覧のファイルで測ったものか(#90)。両方に指紋があって違うときだけ False(指紋の無い古い実測・
    作者の measurements.json は今までどおり使う)。"""
    fp = getattr(m, "files_fp", None)
    now = files_fingerprint(entry) if fp else None
    return not (fp and now and fp != now)


def measured_on_this_gpu(m: Measurement, gpu: Any) -> bool:
    return same_gpu(m.gpu_name, getattr(m, "gpu_pci_id", None), gpu, m.gpu_vram_gb)


def pick_measurement(
    measurements: Iterable[Measurement],
    entry: ModelEntry,
    s: Settings | None = None,
    gpu: Any = None,
) -> Measurement | None:
    """そのモデルの実測を 1 つ選ぶ。ほかの利用者の実測は、この PC と同じ GPU(same_gpu)のものだけ候補にする
    (llmfit と同じ。別の GPU の PC の必要量・段階は動かさない。0.9.0)。作者の同梱の実測・この PC の bench は
    今までどおりどの PC にも使う。並びは 設定が同じもの → この PC と同じ GPU の実測 → 作者・この PC の実測 →
    張り付いていないもの → 大きい GPU → 大きい値(根拠の段階は models.BASIS_DETAILS)。
    gpu は判定に使う models.Gpu(GPU が無いときは None)。"""
    cands = [
        m
        for m in measurements
        if m.model_id == entry.id
        and usable(m)
        and files_match(m, entry)
        and (not is_community(m) or (m.gpu_known is not False and measured_on_this_gpu(m, gpu)))
    ]
    if not cands:
        return None

    def key(m: Measurement):
        same = s is not None and _settings_match(m.settings, s, entry)
        return (
            same,
            measured_on_this_gpu(m, gpu),
            not is_community(m),
            not is_censored(m),
            m.gpu_vram_gb,
            m.peak_vram_gb,
        )

    return max(cands, key=key)


_POSITIVE = ("width", "height", "frames", "seconds", "steps")


# 大きさ(幅・高さ)の入力の無いテンプレートの手の表(#90。入力の画像から作る深度推定・背景除去など)。
# メンテナが bench の記録(workflow.size_input が偽 = 実測に幅・高さが無い)か、テンプレートのノードで
# 確かめたものだけを書く。ほかの利用者の提出では決めない(1 件の提出で、大きさの入力のあるモデルの判定を
# どの PC でも変えないため)。新しいテンプレートは、最初の bench の記録(提出の PR の settings_missing)を
# 見て足す(docs/accuracy.md「実測の使い方」)。画像・動画のモデルだけ(3d・音声は既定の設定に大きさが無い)。
# 値は根拠の 1 行
NO_SIZE_INPUT: dict[str, str] = {
    "utility_birefnet_remove_background": (
        "bench 2026-10-19 RTX 3070(measurements.json に幅・高さ無し)。ノードは LoadImage → BiRefNet だけで、"
        "幅・高さの入力が無い"
    ),
    "utility_moge_depth_estimation": (
        "bench 2026-10-27 RTX 3070・ComfyUI 0.40.0(#90 の PR #92)。大きさは ResizeImageMaskNode"
        "(scale longer dimension 2048)で入力の画像から決まり、幅・高さの入力が無い"
    ),
}


def no_size_input(entry: ModelEntry) -> bool:
    """テンプレートに大きさ(幅・高さ)の入力が無いか(手の表 NO_SIZE_INPUT だけで決める。#90)。
    判定(_prepare の設定)と contrib validate の settings_missing がこれを使う。
    誰がいつ判定しても同じ結果になる。"""
    return entry.modality in ("image", "video") and entry.id in NO_SIZE_INPUT


def settings_for(
    entry: ModelEntry, settings: Settings | None, base: Settings | None = None, *, sized: bool = True
) -> Settings:
    """指定の無い項目を埋める。base(実測の設定)があればそれを、無ければ種類ごとの既定を使う。
    0 以下の解像度・コマ数・長さは指定が無いものとして扱う。sized が False(大きさの入力の無いテンプレート。
    no_size_input)なら、既定の大きさは入れない(指定があれば使う。#90)。"""
    src = base or DEFAULT_SETTINGS.get(entry.modality, Settings())
    merged = Settings(**vars(src))
    if not sized and base is None:
        merged.width = merged.height = None
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
        bits.append(tr(f"{s.frames} コマ", f"{s.frames} frames"))
    if entry.modality in ("music", "sfx", "tts") and s.seconds:
        bits.append(tr(f"{s.seconds:g} 秒", f"{s.seconds:g} s"))
    if s.batch and s.batch > 1:
        bits.append(tr(f"{s.batch} 枚", f"batch {s.batch}"))
    if s.precision:
        bits.append(s.precision)
    return tr("・", ", ").join(bits)


@dataclass
class _Model:
    """1 モデルについて、設定を変えて何度も見積もるための下ごしらえ。"""

    entry: ModelEntry
    c: Coefficients
    measurement: Measurement | None
    delta: float  # 実測に合わせて作業用メモリをずらす量(GiB)
    base: Settings
    parts: Parts
    # 実測の値をそのまま使わなかった(下限・上限で切り詰めた)
    clamped: bool = False
    # ほかの利用者の実測を使うとき、それが無いときの下ごしらえ(精度を変えたときはこちらで見積もる)
    own: _Model | None = None
    # 切り詰めの向き: floor = 下限(作者の torch_reserved は式の値、ほかの利用者は重み + 固定分)まで上げた、
    # cap = GPU の総量まで下げた
    clamp: str | None = None
    # ほかの利用者の実測を設定の違う推定に広げるときのずらし(COMMUNITY_CALIBRATION_RANGE で抑えたもの)。
    # None は delta と同じ
    delta_other: float | None = None

    def parts_for(self, s: Settings) -> Parts:
        return apply_precision(replace(self.parts, notes=list(self.parts.notes)), self.entry, s)

    def need(self, s: Settings, prec: GpuPrecision | None = None) -> Need:
        m = self.measurement
        if self.own is not None and m is not None and _precision_changed(m, s, self.entry):
            return self.own.need(s, prec)  # 精度を変えたら、ほかの利用者の実測は使わない(own のまま)
        parts = self.parts_for(s)
        act = activation_gb(self.entry, s, self.c)
        if m is not None and not _precision_changed(m, s, self.entry):
            same = _settings_match(m.settings, s, self.entry)
            act += self.delta if same or self.delta_other is None else self.delta_other
        if prec is not None and (prec.act_factor != 1 or prec.vae_factor != 1):
            parts.decode *= prec.vae_factor
            act = max(0.0, act) * prec.act_factor
        return need(parts, act, self.c)


def _prepare(
    entry: ModelEntry,
    measurements: Iterable[Measurement],
    c: Coefficients,
    s: Settings | None = None,
    gpu: Any = None,
) -> _Model:
    measurements = list(measurements)
    sized = not no_size_input(entry)  # 大きさの入力があるか(手の表。#90)
    parts = components(entry)
    m = None if parts.unknown else pick_measurement(measurements, entry, s, gpu)
    if m is None:
        return _Model(entry, c, None, 0.0, settings_for(entry, None, sized=sized), parts)
    at = settings_for(entry, m.settings, sized=sized)
    act_m = activation_gb(entry, at, c)
    est = need(parts, act_m, c).full
    delta = m.peak_vram_gb - est
    if is_community(m):
        # ほかの利用者の実測は両方の向きに使う(2026-10-22 のオーナーの決定。llmfit に合わせた)。抑えるのは
        # 物理的な範囲だけ: 全部載せの値は重み + 固定分より下げない(作業用メモリを 0 より下げない)、
        # GPU の総量を超えた値は使わない。張り付いた実測(動的 VRAM が空きを使い切った)は
        # 「これ以上」としてだけ使う。
        # 判定する既定の設定は own のまま(小さな設定の提出で、既定の設定そのものを小さくしない)
        own = _prepare(entry, [x for x in measurements if not is_community(x)], c, s, gpu)
        lo = 0.0 if is_censored(m) else -max(0.0, act_m)
        hi = max(lo, peak_vram_cap(m) - est)
        clamp = "floor" if delta < lo else ("cap" if delta > hi else None)
        delta = min(max(delta, lo), hi)
        # 設定の違う推定へは、実測 ÷ 式 を COMMUNITY_CALIBRATION_RANGE に収めて広げる
        r_lo, r_hi = COMMUNITY_CALIBRATION_RANGE
        other = min(max(delta, est * (r_lo - 1), lo), max(lo, est * (r_hi - 1)))
        return _Model(entry, c, m, delta, own.base, parts, clamp is not None, own, clamp, other)
    clamped = False
    if is_censored(m) or lower_bound_only(m):
        clamped = delta < 0 and lower_bound_only(m)
        delta = max(0.0, delta)  # 張り付いた実測・下限の実測は「これ以上」としてだけ使う
    return _Model(entry, c, m, delta, at, parts, clamped, clamp="floor" if clamped else None)


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
    from .families import FAMILIES, VAES, family_of, shown, vae_of

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
            tr(
                f"{g.name} では ComfyUI が半精度を使わず fp32 で計算するので、作業用メモリと VAE を 2 倍で"
                "見積もった(本体は VRAM に余裕があれば fp32 に広げて載せる。遅い)",
                f"ComfyUI computes in fp32 on {g.name} (no half precision), so the working memory and VAE"
                " were doubled (the model is widened to fp32 if VRAM allows; slow)",
            )
        )
    elif family_fp32:
        notes.append(
            tr(
                f"このモデルの系統({shown(fam.comfy_class)})は ComfyUI が fp16 で動かさない(supported_models "
                "の "
                f"{shown(fam.source)})ので、bf16 を持たない {g.name} では fp32 で計算する。"
                "作業用メモリを 2 倍で見積もった"
                "(本体は VRAM に余裕があれば fp32 で載せる。遅い)",
                f"ComfyUI does not run this model family ({shown(fam.comfy_class)}) in fp16 "
                "(supported_models:"
                f" {shown(fam.source)}), so {g.name}, which lacks bf16, computes in fp32."
                " The working memory was doubled (the model is loaded in fp32 if VRAM allows; slow)",
            )
        )
        if fam.note:
            notes.append(shown(fam.note))
    elif no_bf16 and fam is not None and not fam.fp16 and manual_bf16:
        notes.append(
            tr(
                f"{g.name} は bf16 の計算に対応しないが、ComfyUI はこの系統({shown(fam.comfy_class)})を bf16 "
                "で動かす"
                "(AMD の RDNA 2 以前の扱い。作業用メモリは変えていない)",
                f"{g.name} does not support bf16 compute, but ComfyUI runs this family "
                f"({shown(fam.comfy_class)})"
                " in bf16 (the handling for AMD RDNA 2 and older; working memory unchanged)",
            )
        )
    elif no_bf16 and fam is None:
        notes.append(
            tr(
                "このモデルの系統(ComfyUI のどのモデルか)が分からないので、bf16 を持たない "
                f"{g.name} で fp32 で計算するかは見積もりに入れていない",
                "The model family (which ComfyUI model it is) is unknown, so whether "
                f"{g.name} (no bf16) computes in fp32 is not included in the estimate",
            )
        )
    elif prec == "bf16" and no_bf16:
        notes.append(
            tr(
                f"{g.name} は bf16 の計算に対応しないので、ComfyUI は bf16 の重みを fp16 で動かす(VRAM "
                "は同じ)",
                f"{g.name} does not support bf16 compute, so ComfyUI runs the bf16 weights in fp16 (same "
                "VRAM)",
            )
        )
    if vae_fp32 and not gpu_fp32 and vae > 1:
        notes.append(
            tr(
                f"VAE({shown(vae_kind.source)})は {g.name} では fp32 で載せるので、VAE の重みを 2 "
                "倍で見積もった"
                "(仕上げで VRAM が足りなければ ComfyUI がタイルに分けるので遅くなる)",
                f"The VAE ({shown(vae_kind.source)}) is loaded in fp32 on {g.name}, so its weights were "
                "doubled"
                " (if VRAM runs short when decoding, ComfyUI splits it into tiles, which is slower)",
            )
        )
    if prec == "fp8" and g.fp8 is False:
        ada = getattr(g, "vendor", "nvidia") == "nvidia"
        notes.append(
            tr(
                f"{g.name} は fp8 の計算に対応しない"
                + ("(Ada 以降が対応)" if ada else "")
                + "。ComfyUI は fp8 の重みを、fp16 に広げても "
                "VRAM に収まれば広げて使い(VRAM を多く使う)、収まらなければ fp8 のまま計算のたびに"
                "変換する(遅い)。見積もりは fp8 の大きさのまま",
                f"{g.name} does not support fp8 compute"
                + (" (Ada and later do)" if ada else "")
                + ". ComfyUI widens fp8 weights to fp16 if they still fit in VRAM (uses more VRAM),"
                " otherwise keeps fp8 and converts at every step (slow). The estimate keeps the fp8 size",
            )
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
        reasons=[unknown_size_reason(), *parts.notes],
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
    return _estimate(entry, system, settings, list(measurements), c, with_limits)


def _sample_count(m: Measurement) -> int:
    """実測の件数(ほかの利用者の実測は、まとめの中央値に使った件数。作者・この PC の実測は 1)。"""
    if not is_community(m):
        return 1
    n = getattr(m, "n_used", None)
    if n is None:  # n_used の無い古いまとめ: 提出の数から OOM を引く
        n = (m.n_submissions or 1) - (getattr(m, "n_oom", None) or 0)
    return max(1, n)


def community_label(m: Measurement) -> str:
    """理由に出す「ほかの利用者の実測 3 件(中央値 7.02 GB。OOM 1 件)」。"""
    oom = getattr(m, "n_oom", None) or 0
    return tr(
        f"ほかの利用者の実測 {_sample_count(m)} 件(中央値 {m.peak_vram_gb:.2f} GB"
        + (f"。同じ構成で OOM の報告 {oom} 件" if oom else "")
        + ")",
        f"{_sample_count(m)} measurement(s) by other users (median {m.peak_vram_gb:.2f} GB"
        + (f"; {oom} OOM report(s) with the same setup" if oom else "")
        + ")",
    )


def _estimate(
    entry: ModelEntry,
    system: SystemInfo,
    settings: Settings | None,
    measurements: list[Measurement],
    c: Coefficients,
    with_limits: bool,
) -> Estimate:
    g0 = system.best_gpu
    gname = g0 if g0 is not None and system.vram_gb > 0 else None
    s = settings_for(entry, settings, _prepare(entry, measurements, c, None, gname).base)
    prep = _prepare(entry, measurements, c, s, gname)  # 設定が同じ実測があればそれを選び直す
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
            reasons.append(
                tr(f"{on} の実測は、{community_label(m)}", f"Measured on {on}: {community_label(m)}")
            )
        other_vendor = g0 is not None and g0.vendor != measurement_vendor(m)
        same_settings = _settings_match(m.settings, s, entry) and not _precision_changed(m, s, entry)
        if system.vram_gb <= 0:
            reasons.append(
                tr(
                    f"GPU が無いので、{on} の実測は判定に使っていない",
                    f"No GPU, so the measurement on {on} is not used",
                )
            )
        elif other_vendor:
            # 別の会社の GPU の実測は、作業用メモリをずらすのに使うだけ(その GPU での実測ではない)
            detail = "calibrated"
            reasons.append(
                tr(
                    f"{on} の実測で補正した推定({_vendor_label(g0.vendor)} の GPU での実測は無い)",
                    f"Estimate calibrated with a measurement on {on} (none on a {_vendor_label(g0.vendor)} "
                    "GPU)",
                )
            )
        elif prec.act_factor != 1:
            detail = "calibrated"
            reasons.append(
                tr(
                    f"{on} の実測は半精度で計算する GPU のものなので、精度の違いを足した推定にした",
                    f"The measurement on {on} is from a GPU that computes in half precision;"
                    " the estimate adds the precision difference",
                )
            )
        elif same_settings and prep.clamped:
            # 実測の値をそのまま使わなかった(下限・上限で切り詰めた): 値は実測ではないので basis は estimated
            detail = "measured_community" if is_community(m) else "calibrated"
            if not is_community(m):
                reasons.append(
                    tr(
                        f"{on} の実測 {m.peak_vram_gb:.2f} GB は下限としてだけ使い、"
                        "それを使わない見積もりのほうが大きいので、そちらの値にした",
                        f"The measurement on {on} ({m.peak_vram_gb:.2f} GB) is used only as a lower bound;"
                        " the estimate without it is larger, so that value is used",
                    )
                )
            elif prep.clamp == "cap":
                reasons.append(
                    tr(
                        f"{on} の実測 {m.peak_vram_gb:.2f} GB は GPU の総量を超えるので、"
                        "上限(総量 × 1.02 + 0.25 GB)までにした",
                        f"The measurement on {on} ({m.peak_vram_gb:.2f} GB) exceeds the GPU total,"
                        " so it was capped (total × 1.02 + 0.25 GB)",
                    )
                )
            else:
                reasons.append(
                    tr(
                        f"{on} の実測 {m.peak_vram_gb:.2f} GB は、全部 GPU に載せたときの下限"
                        f"(重み + 固定分 {n.full:.2f} GB)より小さいので、下限の値にした"
                        "(一部の重みをメインメモリに置いていた実測)",
                        f"The measurement on {on} ({m.peak_vram_gb:.2f} GB) is below the minimum for loading"
                        f" everything on the GPU (weights + fixed {n.full:.2f} GB), so the minimum is used"
                        " (some weights were kept in main memory during the measurement)",
                    )
                    if not is_censored(m)
                    else tr(
                        f"{on} の実測 {m.peak_vram_gb:.2f} GB は GPU の上限に張り付いた値なので、"
                        "これ以上としてだけ使い、式の値にした",
                        f"The measurement on {on} ({m.peak_vram_gb:.2f} GB) hit the GPU limit,"
                        " so it is used only as a lower bound and the formula value is used",
                    )
                )
            if m.peak_ram_gb:
                n.ram_gpu = max(n.ram_gpu, m.peak_ram_gb)
        elif same_settings:
            basis, measured_on = "measured", on
            detail = "measured_community" if is_community(m) else "measured"
            if is_censored(m):
                reasons.append(
                    tr(
                        f"実測 {m.peak_vram_gb:.1f} GB は GPU の上限に張り付いた値(全部載せにはこれ以上要る)",
                        f"The measured {m.peak_vram_gb:.1f} GB hit the GPU limit (loading everything needs "
                        "at least this)",
                    )
                )
            if m.peak_ram_gb:
                n.ram_gpu = max(n.ram_gpu, m.peak_ram_gb)
        elif not _precision_changed(m, s, entry):
            detail = "calibrated"
            reasons.append(
                tr(
                    f"{on} の実測({_describe(entry, m.settings)})で補正した推定",
                    f"Estimate calibrated with a measurement on {on} ({_describe(entry, m.settings)})",
                )
            )
    desc = _describe(entry, s)
    if desc:
        reasons.append(tr(f"設定: {desc}", f"Settings: {desc}"))
    parts = prep.parts_for(s)
    disk = parts.files_gb
    fit, mode, why, ram_need = _judge(n, system, prep.c, entry, parts.total)
    fit_from, measured_peak = "estimate", None
    band = _measured_band(m, entry, s, system, prec)
    promoted = band if band is not None and _LEVELS.index(band[0]) < _LEVELS.index(fit) else None
    if promoted is not None:
        level, line = promoted
        fit_from, measured_peak = "measurement", m.peak_vram_gb
        ram_need = n.ram_gpu
        fit, mode, why = _ram_check(level, "gpu", [line], ram_need, system)
    elif m is not None and basis == "measured" and system.vram_gb > 0 and mode in ("gpu", "offload"):
        if m.gpu_vram_gb > system.vram_gb / MEASURED_VRAM_TOLERANCE:
            reasons.append(
                tr(
                    f"段階は見積もりで決めた(実測の GPU {m.gpu_vram_gb:g} GB は"
                    f"この PC の {system.vram_gb:.1f} GB より大きい)",
                    f"The level comes from the estimate (the measured GPU, {m.gpu_vram_gb:g} GB, is larger"
                    f" than this PC's {system.vram_gb:.1f} GB)",
                )
            )
    g = system.best_gpu
    if g is not None and g.vendor != "nvidia" and system.vram_gb > 0 and mode in ("gpu", "offload"):
        # NVIDIA 以外の GPU は、推定だけなら段階を good までにする。この PC と同じ名前の GPU の同じ設定の
        # 実測があれば、その実測の段階まで上げてよい(2026-10-22 のオーナーの決定)
        label = _vendor_label(g.vendor)
        own = band if band is not None and measured_on_this_gpu(m, g) else None
        if own is not None:
            if fit == "perfect" and fit_from != "measurement":
                if own[0] == "perfect":
                    why = [
                        *why,
                        tr(
                            f"{label} のこの GPU で同じ設定の実測があるので、段階の上限(good)を外した",
                            f"A measurement with the same settings exists on this {label} GPU,"
                            " so the level cap (good) was lifted",
                        ),
                    ]
                else:
                    fit = own[0]
                    why = [
                        *why,
                        tr(
                            f"{label} のこの GPU の同じ設定の実測で決まる段階 {own[0]} までにした",
                            f"Capped at {own[0]}, the level from a measurement with the same settings"
                            f" on this {label} GPU",
                        ),
                    ]
        elif fit == "perfect" and fit_from != "measurement":
            fit = "good"
            why = [
                *why,
                tr(
                    f"推定({label}・実測なし)なので、段階は good までにした",
                    f"Estimated ({label}, no measurement), so the level is capped at good",
                ),
            ]
        elif fit_from != "measurement":
            why = [*why, tr(f"推定({label}・実測なし)", f"Estimated ({label}, no measurement)")]
    reasons.extend(prec.notes)
    free_short = False
    g = system.best_gpu
    if g is not None and g.vram_free_gb is not None and mode in ("gpu", "offload") and system.vram_gb > 0:
        _lvl, how = gpu_choice(
            n.full, n.sequential, n.offload, system.vram_gb, allow_offload=not shared_memory(system)
        )
        use = {"full": n.full, "seq": n.sequential, "offload": n.offload}[how]
        if fit_from == "measurement" and m is not None:
            use = measured_total_gb(m)
        if use > g.vram_free_gb:
            free_short = True
            used = g.vram_used_gb
            why = [
                *why,
                tr(
                    f"今の空き VRAM は {g.vram_free_gb:.1f} GB"
                    + (f"(ほかのアプリが {used:.1f} GB 使用中)" if used is not None else "")
                    + f"で、要る {use:.1f} GB に足りない。"
                    "ほかのアプリ(ブラウザ・ゲームなど)を閉じれば動く",
                    f"Free VRAM is now {g.vram_free_gb:.1f} GB"
                    + (f" (other apps use {used:.1f} GB)" if used is not None else "")
                    + f", short of the {use:.1f} GB needed."
                    " Close other apps (browser, games, etc.) to run it",
                ),
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
        sample_count=_sample_count(m) if m is not None and detail != "estimated" else None,
    )
    if with_limits:
        est.limits = limits(prep, system, s)
    return est


# 実測の GPU の VRAM(m.gpu_vram_gb は公称の 16 など)と nvidia-smi の総量(15.9 など)の違いを許す割合(#26)
MEASURED_VRAM_TOLERANCE = 0.97


# 実測の backend(bench が書く。提出の system.gpu.backend)→ 会社
BACKEND_VENDOR = {"cuda": "nvidia", "rocm": "amd", "xpu": "intel", "mps": "apple"}


def measurement_vendor(m: Measurement) -> str:
    """実測の GPU の会社。backend の記録(gpu_backend)があればそれを使い、無いときだけ GPU 名から推す
    (Apple の MPS では ComfyUI の GPU 名が "mps" になり、名前からは分からない。0.9.0)。"""
    backend = getattr(m, "gpu_backend", None)
    if backend in BACKEND_VENDOR:
        return BACKEND_VENDOR[backend]
    return gpu_vendor(m.gpu_name)


# torch_reserved の実測(ComfyUI のプロセスの torch の確保量)に含まれない、CUDA / HIP の文脈などの分(GiB)。
# bench の説明(bench.py の冒頭)の「CUDA の文脈(約 0.3〜0.5 GB)」の多いほうをとる(段階を甘くしない向き)。
# 段階を実測で決める(#26)ときだけ足す。見積もりの必要量の下限・係数の合わせには使わない(0.9.0)
TORCH_RESERVED_CONTEXT_GB = 0.5


def measured_total_gb(m: Measurement) -> float:
    """段階を決めるのに使う実測の VRAM。torch_reserved なら文脈の分を足す。"""
    extra = TORCH_RESERVED_CONTEXT_GB if getattr(m, "vram_metric", None) in LOWER_BOUND_METRICS else 0.0
    return m.peak_vram_gb + extra


def gpu_vendor(name: str) -> str:
    """GPU 名から会社(実測の gpu_name は名前だけなので)。分からなければ nvidia(同梱の実測はすべて NVIDIA)。"""
    n = name.lower()
    if "radeon" in n or n.startswith("amd") or "instinct" in n:
        return "amd"
    if n.startswith("intel") or " arc" in f" {n}":
        return "intel"
    if n.startswith("apple") or n == "mps":  # ComfyUI の MPS の GPU 名(0.9.0 より前の bench が書いた)
        return "apple"
    return "nvidia"


def _vendor_label(vendor: str) -> str:
    return {"nvidia": "NVIDIA", "amd": "AMD", "intel": "Intel", "apple": "Apple"}.get(vendor, vendor)


def _measured_band(
    m: Measurement | None, entry: ModelEntry, s: Settings, system: SystemInfo, prec: GpuPrecision
) -> tuple[str, str] | None:
    """同じ設定で成功した実測があり、その GPU の VRAM がこの PC の VRAM 以下
    (97% の許容)で、実測のピークがこの PC の VRAM に収まるなら、実測で決めた段階と理由(#26)。

    実測が「GPU で動いた」ことは、それ以上の VRAM の同じ会社・同じ精度の扱いの GPU でも動く根拠になる。
    段階は 実測の VRAM ÷ この PC の VRAM の帯(FIT_BANDS)で、少なくとも good。上限に張り付いた実測
    (動的 VRAM が空きを使い切った)・同じ構成で OOM の報告もあるまとめは perfect の根拠にしない(good まで)。
    torch_reserved の実測(CUDA / HIP の文脈を含まない)は文脈の分(TORCH_RESERVED_CONTEXT_GB)を足した値で
    帯を決め、perfect の根拠にもする(0.9.0。bench の提出はすべて torch_reserved なので、そうしないと
    NVIDIA 以外の GPU が実測があっても perfect にならない)。会社は measurement_vendor(backend の記録が先)。
    ほかの利用者の実測は、この PC と同じ名前の GPU のものだけ使う(llmfit と同じく、似た GPU には広げない。
    2026-10-22 のオーナーの決定)。見積もりの段階と比べるのは呼び出し側。"""
    if m is None or not usable(m):
        return None
    g = system.best_gpu
    vram = system.vram_gb
    if g is None or vram <= 0 or g.vendor != measurement_vendor(m) or prec.act_factor != 1:
        return None
    if is_community(m) and not measured_on_this_gpu(m, g):
        return None
    if not _settings_match(m.settings, s, entry) or _precision_changed(m, s, entry):
        return None
    total = measured_total_gb(m)
    if vram < m.gpu_vram_gb * MEASURED_VRAM_TOLERANCE or total > vram:
        return None  # 実測の GPU より小さい、または実測の VRAM がこの PC の VRAM に収まらない
    band = fit_band(total, vram)
    weak = is_censored(m) or (getattr(m, "n_oom", None) or 0) > 0
    level = "perfect" if band == "perfect" and not weak else "good"
    on = f"{m.gpu_name} ({m.gpu_vram_gb:g} GB)"
    pinned = (
        tr(
            "(GPU の上限に張り付いた値: 動的 VRAM が空きを使った)",
            " (hit the GPU limit: dynamic VRAM used the free space)",
        )
        if is_censored(m)
        else ""
    )
    ctx = (
        tr(
            f"(torch の確保量。文脈などの {TORCH_RESERVED_CONTEXT_GB:g} GB を足して {total:.1f} GB とみた)",
            f" (torch reserved; counted as {total:.1f} GB with {TORCH_RESERVED_CONTEXT_GB:g} GB added for "
            "the context etc.)",
        )
        if total != m.peak_vram_gb
        else ""
    )
    who = tr(f"({community_label(m)})", f" ({community_label(m)})") if is_community(m) else ""
    return level, tr(
        f"{on} で同じ設定の実測 {m.peak_vram_gb:.1f} GB{ctx}{pinned}{who}・成功があり、"
        f"この PC の VRAM {vram:.1f} GB はそれ以上なので、"
        "GPU で動くと判定した(段階は実測で決めた。見積もりは目安)",
        f"A successful measurement on {on} with the same settings, {m.peak_vram_gb:.1f} GB{ctx}{pinned}{who},"
        f" and this PC has at least that VRAM ({vram:.1f} GB), so it runs on the GPU"
        " (level from the measurement; the estimate is a guide)",
    )


def _ram_check(fit: str, mode: str, reasons: list[str], need_gb: float, system: SystemInfo):
    """メインメモリの判定。足りなければ、ram_limit(ページファイルの大きさ、分からなければ倍率
    RAM_PAGEFILE_FACTOR)までは marginal(仮想メモリで遅い)、超えたら too_tight。"""
    ram_total = system.ram_total_gb
    if not ram_total or need_gb <= ram_total:
        return fit, mode, reasons
    limit, swap = ram_limit(system)
    if need_gb > limit:
        if swap is not None:
            pf = tr(f"、ページファイルは最大 {swap:.1f} GB", f", page file up to {swap:.1f} GB")
        else:
            pf = ""
        msg = tr(
            f"メモリ不足: 約 {need_gb:.1f} GB 必要(この PC は {ram_total:.1f} GB{pf})",
            f"Not enough memory: about {need_gb:.1f} GB needed (this PC has {ram_total:.1f} GB{pf})",
        )
        return ("too_tight", "none", [msg, *reasons])
    worse = fit if fit in ("marginal", "too_tight") else "marginal"
    return (
        worse,
        mode,
        [
            *reasons,
            tr(
                f"メモリが少し足りない: 約 {need_gb:.1f} GB 必要(この PC は {ram_total:.1f} GB)。"
                "足りない分は仮想メモリ(ページファイル)を使うので遅い",
                f"Memory is a little short: about {need_gb:.1f} GB needed (this PC has {ram_total:.1f} GB)."
                " The rest uses virtual memory (page file), which is slow",
            ),
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
        return (
            cpu[0],
            cpu[1],
            [*cpu[2], tr(f"GPU では動かない: {gpu[2][0]}", f"Does not run on the GPU: {gpu[2][0]}")],
            cpu[3],
        )
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
    why = tr("GPU なし", "No GPU") if system.vram_gb <= 0 else tr("GPU に載らない", "Does not fit on the GPU")
    run = cpu_family(entry)
    if run is not None and w > 0:
        # ComfyUI は CPU では本体と VAE を fp32 で持ち、fp32 で計算する(families.CPU_FAMILIES の説明)。
        # 2 バイトの重みは 2 倍、作業用メモリも 2 倍にする
        two_byte = source_precision(entry) in ("fp16", "bf16")
        ram = n.ram + (n.main + n.vae if two_byte else n.vae) + 2 * max(0.0, n.act)
        speed, evidence = run.speed_text(), run.evidence_text()
        reasons = [
            tr(
                f"{why}: CPU で動く系統({run.family})なので CPU で動く見込み(CPU は推定・実測なし。遅い"
                + (f"。{speed}" if speed else "")
                + f")。ComfyUI は CPU では本体と VAE を fp32 で持つので、メインメモリを約 {ram:.1f} GB "
                "とみた",
                f"{why}: a family that runs on CPU ({run.family}), so it should run on the CPU (CPU is "
                "estimated, not measured; slow"
                + (f". {speed}" if speed else "")
                + "). ComfyUI keeps the model and VAE in fp32 on CPU, so main memory is taken as "
                f"about {ram:.1f} GB",
            ),
            tr(f"CPU で動く根拠: {evidence}", f"Evidence that it runs on CPU: {evidence}"),
        ]
        fit, mode, reasons = _ram_check("marginal", "cpu", reasons, ram, system)
        return fit, mode, reasons, ram
    if 0 < w <= CPU_MAX_WEIGHTS_GB and entry.modality in CPU_MODALITIES:
        line = tr(f"{why}: CPU で動く見込み(遅い)", f"{why}: should run on the CPU (slow)")
        fit, mode, reasons = _ram_check("marginal", "cpu", [line], n.ram, system)
        return fit, mode, reasons, n.ram
    return (
        "too_tight",
        "none",
        [
            tr(
                "GPU なし: このモデルは GPU が必要(CPU で動く系統・小さなモデルではない)",
                "No GPU: this model needs a GPU (not a CPU-capable family or a small model)",
            )
        ],
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
    if shared:
        return _judge_shared(n, system, pct)
    if fit == "too_tight":
        msg = tr(
            f"VRAM 不足: 逃がしても {n.offload:.1f} GB 必要(この PC は {vram:.1f} GB、{pct['offload']}%)",
            f"Not enough VRAM: {n.offload:.1f} GB needed even with offloading"
            f" (this PC has {vram:.1f} GB, {pct['offload']}%)",
        )
        return "too_tight", "none", [msg], n.ram
    if how == "full":
        mode = "gpu"
        reasons = [
            tr(
                f"VRAM {n.full:.1f} / {vram:.1f} GB({pct['full']}%): 全部載る",
                f"VRAM {n.full:.1f} / {vram:.1f} GB ({pct['full']}%): fits entirely",
            )
        ]
    elif how == "seq":
        mode = "gpu"
        reasons = [
            tr(
                f"VRAM {n.sequential:.1f} / {vram:.1f} GB({pct['seq']}%): 部品を順に載せ替えれば載る"
                f"(全部載せは {n.full:.1f} GB。段階は good まで)",
                f"VRAM {n.sequential:.1f} / {vram:.1f} GB ({pct['seq']}%): fits when the parts are loaded in "
                "turn"
                f" (all at once needs {n.full:.1f} GB; level capped at good)",
            )
        ]
    else:
        mode = "offload"
        reasons = [
            tr(
                f"VRAM {n.full:.1f} GB 必要 > {vram:.1f} GB: 重みの一部をメインメモリに逃がせば動く"
                f"(遅い。逃がすと {n.offload:.1f} GB、{pct['offload']}%。段階は marginal まで)",
                f"VRAM {n.full:.1f} GB needed > {vram:.1f} GB: runs by offloading part of the weights to "
                "main memory"
                f" (slow; {n.offload:.1f} GB offloaded, {pct['offload']}%; level capped at marginal)",
            )
        ]
    ram_need = n.ram_gpu if mode == "gpu" else n.ram
    fit, mode, reasons = _ram_check(fit, mode, reasons, ram_need, system)
    return fit, mode, reasons, ram_need


def _judge_shared(n: Need, system: SystemInfo, pct: dict[str, int]) -> tuple[str, str, list[str], float]:
    """メインメモリと共有する GPU(内蔵 GPU・APU・Apple silicon)の判定。
    GPU に載せる分もメインメモリから使うので、メインメモリの要る量 = GPU に載せないときの量
    (固定分 + いちばん大きい段の部品)+ GPU に載せる量のうち BIOS で取り分けた VRAM(搭載量に入らない分)を
    超える分。全部載せ・順に載せ替えのそれぞれでメインメモリも確かめ、良いほうを取る
    (VRAM・メインメモリを増やしても段階が下がらない)。逃がす動かし方は無い。"""
    vram = system.vram_gb
    g = system.best_gpu
    carve = (getattr(g, "carveout_gb", None) or 0.0) if g is not None else 0.0
    best: tuple[str, str, list[str], float] | None = None
    for _how, use, top, line in (
        (
            "full",
            n.full,
            "perfect",
            tr(
                f"GPU のメモリ {n.full:.1f} / {vram:.1f} GB({pct['full']}%): 全部載る",
                f"GPU memory {n.full:.1f} / {vram:.1f} GB ({pct['full']}%): fits entirely",
            ),
        ),
        (
            "seq",
            n.sequential,
            "good",
            tr(
                f"GPU のメモリ {n.sequential:.1f} / {vram:.1f} GB({pct['seq']}%): 部品を順に載せ替えれば載る"
                f"(全部載せは {n.full:.1f} GB。段階は good まで)",
                f"GPU memory {n.sequential:.1f} / {vram:.1f} GB ({pct['seq']}%): fits when the parts are "
                "loaded"
                f" in turn (all at once needs {n.full:.1f} GB; level capped at good)",
            ),
        ),
    ):
        level = _cap(fit_band(use, vram), top)
        if level == "too_tight":
            continue
        from_ram = max(0.0, use - carve)
        ram_need = n.ram_gpu + from_ram
        note = tr(
            f"GPU の分 {from_ram:.1f} GB もメインメモリから使うので、メインメモリは約 {ram_need:.1f} GB 要る",
            f"The GPU share ({from_ram:.1f} GB) also comes from main memory, so about {ram_need:.1f} GB of "
            "main memory is needed",
        )
        fit, mode, reasons = _ram_check(level, "gpu", [line, note], ram_need, system)
        if best is None or _LEVELS.index(fit) < _LEVELS.index(best[0]):
            best = (fit, mode, reasons, ram_need)
    if best is None:
        msg = tr(
            f"GPU のメモリ不足: 部品を順に載せ替えても {n.sequential:.1f} GB 必要"
            f"(GPU が使えるのは {vram:.1f} GB、{pct['seq']}%)",
            f"Not enough GPU memory: {n.sequential:.1f} GB needed even loading the parts in turn"
            f" (the GPU can use {vram:.1f} GB, {pct['seq']}%)",
        )
        return "too_tight", "none", [msg], n.ram
    return best


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
    g = system.best_gpu
    gname = g if g is not None and system.vram_gb > 0 else None
    s = settings_for(entry, settings, _prepare(entry, measurements, c, None, gname).base)
    return limits(_prepare(entry, measurements, c, s, gname), system, s)
