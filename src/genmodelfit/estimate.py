"""必要な VRAM・メモリの見積もりと、PC との相性(4 段階)の判定。

考え方(willitrunai.com が公開している方式の考え方だけを参考にした。データは使っていない):

  - 重み: 部品(本体・テキストエンコーダ・VAE・その他)ごとにファイルの大きさで数える。
    1 ファイルに全部入りのチェックポイント(`checkpoints`)は、safetensors の見出しで確かめた内訳で分ける
    (内訳が分からないものは一般的な割合で分ける)
  - ComfyUI は「文章の読み取り → 生成 → 仕上げ(VAE)」を順に行い、
    足りなければ間で部品を降ろす。そこで 3 つの量を出す
      全部載せ(vram_gb)      = 固定分 + 全部品 + 作業用メモリ
      順に載せ替え(sequential) = 固定分 + 段ごとの最大(その段の部品 + 作業用メモリ)
      逃がす(vram_offload_gb)  = 固定分 + 段ごとの最大(その段の部品の一部 + 作業用メモリ)
    最後のものは ComfyUI の lowvram / 動的 VRAM が重みの一部をメインメモリに置いたまま動く場合で、遅い
  - 作業用メモリ: 画像は画素数 × 枚数、動画は画素数 × コマ数の平方根、音声は長さ、3D は一定
  - 実測があれば、そのモデルの作業用メモリを実測に合わせてずらす。設定が実測と同じなら basis="measured"
  - 係数は calibration.json(python -m genmodelfit.calibrate が実測から合わせたもの)。詳しくは docs/accuracy.md
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field, replace

from .db import bundled_text
from .models import Estimate, Measurement, ModelEntry, ModelFile, Settings, SystemInfo

GB = 1_000_000_000  # ファイルの大きさは 10 進の GB で扱う(VRAM の GiB との差は係数に吸収)

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

# 段の名前(ComfyUI が順に動かす単位)
STAGES = ("encode", "denoise", "decode", "other")
ROLE_STAGE = {
    "main": "denoise",
    "lora": "denoise",
    "text_encoder": "encode",
    "clip_vision": "encode",
    "vae": "decode",
    "other": "other",
}
DIRECTORY_ROLE = {
    "diffusion_models": "main",
    "unet": "main",
    "checkpoints": "main",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "clip_vision": "clip_vision",
    "vae": "vae",
    "loras": "lora",
}

# 全部入りチェックポイントの内訳(本体, テキスト/画像エンコーダ, VAE の割合)。
# 2026-10-20 に各ファイルの safetensors の見出し(テンソルの大きさ)を読んで数えた。
CHECKPOINT_SPLITS: list[tuple[str, tuple[float, float, float], str]] = [
    (r"sd_xl|sdxl", (0.740, 0.236, 0.024), "SDXL(UNet 5.14 GB / CLIP-L+bigG 1.64 GB / VAE 0.17 GB)"),
    (
        r"ace_step_1\.5",
        (0.478, 0.489, 0.034),
        "ACE-Step 1.5(DiT 4.79 GB / 言語モデル等 4.90 GB / VAE 0.34 GB)",
    ),
    (
        r"hunyuan3d-dit-v2",
        (0.452, 0.461, 0.087),
        "Hunyuan3D 2.0(DiT 2.23 GB / 画像エンコーダ 2.27 GB / VAE 0.43 GB)",
    ),
    (
        r"hunyuan_3d_v2\.1",
        (0.828, 0.083, 0.089),
        "Hunyuan3D 2.1(DiT 6.10 GB / 画像エンコーダ 0.61 GB / VAE 0.66 GB)",
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
    "gguf-q8": 1.06,
    "q6": 0.82,
    "gguf-q6": 0.82,
    "q5": 0.69,
    "gguf-q5": 0.69,
    "nf4": 0.56,
    "q4": 0.56,
    "gguf-q4": 0.56,
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
    (r"q5_k|q5_0|\bq5\b", "q5"),
    (r"q4_k|q4_0|\bq4\b", "q4"),
]


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
    good_margin: float = 0.90  # perfect はこの割合以内に収まること
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


# --- 部品(重み) ---------------------------------------------------------------------------------


@dataclass
class Parts:
    """段ごとの重み(GB)と、分け方の説明。"""

    encode: float = 0.0
    denoise: float = 0.0
    decode: float = 0.0
    other: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.encode + self.denoise + self.decode + self.other

    def stage(self, name: str) -> float:
        return getattr(self, name)


def _role_of(f: ModelFile) -> str:
    if f.role in ROLE_STAGE:
        return f.role
    if f.directory:
        return DIRECTORY_ROLE.get(f.directory, "other")
    n = f.name.lower()
    if "vae" in n or re.search(r"(^|[/_.-])ae\.", n):
        return "vae"
    if re.search(r"text_encoder|tokenizer|\bt5|umt5|clip|qwen|gemma|llama|mistral", n):
        return "text_encoder"
    if "lora" in n:
        return "lora"
    return "main"


def _split_of(name: str) -> tuple[tuple[float, float, float], str | None]:
    n = name.lower()
    for pat, split, label in CHECKPOINT_SPLITS:
        if re.search(pat, n):
            return split, label
    return GENERIC_SPLIT, None


def components(entry: ModelEntry) -> Parts:
    """モデル一式を段ごとの重み(GB)に分ける。"""
    p = Parts()
    files = [f for f in entry.files if f.size_bytes]
    if not files:
        size = entry.total_size_bytes or 0
        if not size and entry.params_b and entry.precision in PRECISION_BYTES:
            size = int(entry.params_b * 1e9 * PRECISION_BYTES[entry.precision])
            p.notes.append("ファイルの大きさが無いので、パラメータ数と精度から重みを求めた")
        main, enc, vae = GENERIC_SPLIT
        p.denoise, p.encode, p.decode = size * main / GB, size * enc / GB, size * vae / GB
        if size:
            p.notes.append("部品の内訳が無いので、一般的な割合(本体 75%・エンコーダ 20%・VAE 5%)で分けた")
        return p
    for f in files:
        gb = f.size_bytes / GB
        role = _role_of(f)
        if f.directory == "checkpoints" and role == "main":
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
    if p.denoise == 0 and p.total > 0:  # 本体が見つからない(役割の付け方が違う)ときは最大の段を本体とみなす
        biggest = max(STAGES, key=p.stage)
        p.denoise, value = p.stage(biggest), p.denoise
        setattr(p, biggest, value)
    return p


def source_precision(entry: ModelEntry) -> str | None:
    if entry.precision:
        return entry.precision.lower()
    mains = [f for f in entry.files if _role_of(f) == "main"]
    for f in mains:
        for pat, prec in _PRECISION_IN_NAME:
            if re.search(pat, f.name.lower()):
                return prec
    return None


def apply_precision(parts: Parts, entry: ModelEntry, s: Settings) -> Parts:
    """Settings.precision が指定されたら、本体の重みをその精度に換算する(エンコーダ・VAE はそのまま)。"""
    want = (s.precision or "").lower()
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
        parts.notes.append(f"本体を {have} → {want} に換算した(×{ratio:.2f})")
    return parts


# --- 作業用メモリ ---------------------------------------------------------------------------------


def activation_features(modality: str, s: Settings, c: Coefficients) -> dict[str, float]:
    """作業用メモリの係数に掛ける量。calibrate も同じものを使う。"""
    mp = s.megapixels or 0.0
    batch = max(1, s.batch or 1)
    if modality == "image":
        return {"act_image_gb_per_mp": mp * batch}
    if modality == "video":
        frames = s.frames or c.act_video_ref_frames
        scale = (frames / c.act_video_ref_frames) ** c.act_video_frame_exponent
        return {"act_video_gb_per_mp": mp * scale * batch}
    if modality in ("music", "sfx", "tts"):
        return {"act_audio_gb_per_min": (s.seconds or 30) / 60}
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
    ram: float


def need(parts: Parts, act: float, c: Coefficients) -> Need:
    w = c.weight_factor
    act = max(0.0, act)
    full = c.overhead_gb + parts.total * w + act
    # 作業用メモリが要るのは生成と仕上げの段(文章の読み取りは小さいので数えない)
    stage_act = {"encode": 0.0, "denoise": act, "decode": act, "other": 0.0}
    seq = c.overhead_gb + max(parts.stage(st) * w + stage_act[st] for st in STAGES)
    off = c.overhead_gb + max(parts.stage(st) * w * c.offload_fraction + stage_act[st] for st in STAGES)
    ram = c.ram_extra_gb + parts.total * w
    return Need(full=full, sequential=min(seq, full), offload=min(off, seq, full), ram=ram)


def raw_estimate(entry: ModelEntry, s: Settings, c: Coefficients) -> tuple[float, float, float]:
    """(全部載せの VRAM, 逃がしたときの VRAM, メインメモリ) を GB で返す(実測を使わない式だけの値)。"""
    parts = apply_precision(components(entry), entry, s)
    n = need(parts, activation_gb(entry, s, c), c)
    return n.full, n.offload, n.ram


def is_censored(m: Measurement) -> bool:
    """実測が GPU の上限に張り付いている(本当の全部載せの量はこれ以上)か。"""
    if (m.offload or "").lower() in ("lowvram", "novram", "cpu"):
        return True
    return m.peak_vram_gb >= CENSOR_RATIO * m.gpu_vram_gb


def _settings_match(a: Settings, b: Settings) -> bool:
    keys = ("width", "height", "frames", "seconds")
    if any(getattr(a, k) != getattr(b, k) for k in keys):
        return False
    pa, pb = (a.precision or "").lower(), (b.precision or "").lower()
    return not pa or not pb or pa == pb


def pick_measurement(
    measurements: Iterable[Measurement], entry: ModelEntry, s: Settings | None = None
) -> Measurement | None:
    """そのモデルの実測を 1 つ選ぶ。設定が同じもの → 張り付いていないもの → 大きい GPU → 大きい値 の順。"""
    cands = [m for m in measurements if m.model_id == entry.id]
    if not cands:
        return None

    def key(m: Measurement):
        same = s is not None and _settings_match(m.settings, s)
        return (same, not is_censored(m), m.gpu_vram_gb, m.peak_vram_gb)

    return max(cands, key=key)


def settings_for(entry: ModelEntry, settings: Settings | None, base: Settings | None = None) -> Settings:
    """指定の無い項目を埋める。base(実測の設定)があればそれを、無ければ種類ごとの既定を使う。"""
    src = base or DEFAULT_SETTINGS.get(entry.modality, Settings())
    merged = Settings(**vars(src))
    if settings is not None:
        for k, v in vars(settings).items():
            if v is not None:
                setattr(merged, k, v)
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
    delta: float  # 実測に合わせて作業用メモリをずらす量(GB)
    base: Settings
    parts: Parts

    def parts_for(self, s: Settings) -> Parts:
        return apply_precision(replace(self.parts, notes=list(self.parts.notes)), self.entry, s)

    def need(self, s: Settings) -> Need:
        parts = self.parts_for(s)
        act = activation_gb(self.entry, s, self.c)
        if self.measurement is not None and not _precision_changed(self.measurement, s):
            act += self.delta
        return need(parts, act, self.c)


def _precision_changed(m: Measurement, s: Settings) -> bool:
    want = (s.precision or "").lower()
    have = (m.settings.precision or "").lower()
    return bool(want) and want != have


def _prepare(
    entry: ModelEntry, measurements: Iterable[Measurement], c: Coefficients, s: Settings | None = None
) -> _Model:
    parts = components(entry)
    m = pick_measurement(measurements, entry, s)
    if m is None:
        return _Model(entry, c, None, 0.0, settings_for(entry, None), parts)
    base = settings_for(entry, m.settings)
    est = need(parts, activation_gb(entry, base, c), c).full
    delta = m.peak_vram_gb - est
    if is_censored(m):
        delta = max(0.0, delta)  # 張り付いた実測は下限としてだけ使う
    return _Model(entry, c, m, delta, base, parts)


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
    s = settings_for(entry, settings, _prepare(entry, measurements, c).base)
    prep = _prepare(entry, measurements, c, s)  # 設定が同じ実測があればそれを選び直す
    n = prep.need(s)
    reasons: list[str] = []
    basis, measured_on = "estimated", None
    m = prep.measurement
    if m is not None:
        on = f"{m.gpu_name} ({m.gpu_vram_gb:g} GB)"
        if _settings_match(m.settings, s) and not _precision_changed(m, s):
            basis, measured_on = "measured", on
            if is_censored(m):
                reasons.append(
                    f"実測 {m.peak_vram_gb:.1f} GB は GPU の上限に張り付いた値(全部載せにはこれ以上要る)"
                )
            if m.peak_ram_gb:
                n.ram = max(n.ram, m.peak_ram_gb)
        elif not _precision_changed(m, s):
            reasons.append(f"{on} の実測({_describe(entry, m.settings)})で補正した推定")
    desc = _describe(entry, s)
    if desc:
        reasons.append(f"設定: {desc}")
    parts = prep.parts_for(s)
    disk = (entry.total_size_bytes or 0) / GB or None
    fit, mode, why = judge(n.full, n.offload, n.ram, system, c, entry, sequential=n.sequential)
    est = Estimate(
        vram_gb=round(n.full, 2),
        vram_offload_gb=round(n.offload, 2),
        ram_gb=round(n.ram, 2),
        disk_gb=round(disk, 2) if disk else None,
        basis=basis,
        fit=fit,
        run_mode=mode,
        reasons=why + reasons + parts.notes,
        measured_on=measured_on,
        vram_sequential_gb=round(n.sequential, 2),
        settings={k: v for k, v in vars(s).items() if v is not None},
    )
    if with_limits:
        est.limits = limits(prep, system, s)
    return est


def judge(
    full: float,
    offload: float,
    ram: float,
    system: SystemInfo,
    c: Coefficients,
    entry: ModelEntry,
    *,
    sequential: float | None = None,
) -> tuple[str, str, list[str]]:
    """4 段階の判定。perfect=全部載って余裕あり / good=全部、または段ごとに全部載る /
    marginal=重みの一部をメインメモリに逃がせば動く(遅い)・CPU / too_tight=動かない。"""
    seq = full if sequential is None else sequential
    vram = system.vram_gb
    ram_total = system.ram_total_gb
    reasons: list[str] = []
    if ram_total and ram > ram_total:
        reasons.append(f"メモリ不足: 約 {ram:.1f} GB 必要(この PC は {ram_total:.1f} GB)")
        return "too_tight", "none", reasons
    if vram <= 0:
        small = (entry.total_size_bytes or 0) <= 2 * GB
        if small and entry.modality in ("tts", "sfx", "music", "image"):
            reasons.append("GPU なし: CPU で動く見込み(遅い)")
            return "marginal", "cpu", reasons
        reasons.append("GPU なし: このモデルは NVIDIA の GPU が必要")
        return "too_tight", "none", reasons
    if full <= vram * c.good_margin:
        reasons.append(f"VRAM {full:.1f} / {vram:.1f} GB: 余裕をもって全部載る")
        return "perfect", "gpu", reasons
    if full <= vram:
        reasons.append(f"VRAM {full:.1f} / {vram:.1f} GB: 全部載るが余裕は少ない")
        return "good", "gpu", reasons
    if seq <= vram:
        reasons.append(
            f"VRAM {seq:.1f} / {vram:.1f} GB: 部品を順に載せ替えれば載る(全部載せは {full:.1f} GB)"
        )
        return "good", "gpu", reasons
    if offload <= vram:
        reasons.append(
            f"VRAM {full:.1f} GB 必要 > {vram:.1f} GB: 重みの一部をメインメモリに逃がせば動く"
            f"(遅い。逃がすと {offload:.1f} GB)"
        )
        return "marginal", "offload", reasons
    reasons.append(f"VRAM 不足: 逃がしても {offload:.1f} GB 必要(この PC は {vram:.1f} GB)")
    return "too_tight", "none", reasons


# --- 限界(この PC で good になる最大の設定) ----------------------------------------------------

GOOD = ("perfect", "good")


def _ok(prep: _Model, system: SystemInfo, s: Settings) -> bool:
    n = prep.need(s)
    fit, _mode, _ = judge(n.full, n.offload, n.ram, system, prep.c, prep.entry, sequential=n.sequential)
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
    どの値も、それより小さくしても good にならないときは None。"""
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
