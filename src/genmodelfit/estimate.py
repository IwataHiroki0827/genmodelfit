"""必要な VRAM・メモリの見積もりと、PC との相性(4 段階)の判定。

考え方(willitrunai.com が公開している方式を参考にした。データは使っていない):
  - 重み: ファイルの大きさがそのまま GPU に載る量(精度はファイルに表れている)
  - 全部載せ(gpu): 全ファイル + 作業用メモリ + 固定分
  - 逃がす(offload): 最大の部品 1 つ + 作業用メモリ + 固定分(他はメインメモリに置き、順に載せ替える)
  - 作業用メモリ: 画像は画素数に比例、動画は画素数 × コマ数で増える(緩やか)
  - 実測があれば必ず実測を使い、basis="measured" と出す
係数は calibration.json(実測から合わせたもの)で上書きする(#4)。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from .db import bundled_text
from .models import Estimate, Measurement, ModelEntry, Settings, SystemInfo

GB = 1_000_000_000  # ファイルの大きさは 10 進の GB で扱う(VRAM は GiB だが差は 7% で係数に吸収)

# 種類ごとの既定の設定(何も指定しないとき)
DEFAULT_SETTINGS: dict[str, Settings] = {
    "image": Settings(width=1024, height=1024, steps=None),
    "video": Settings(width=832, height=480, frames=81),
    "music": Settings(seconds=30),
    "sfx": Settings(seconds=10),
    "tts": Settings(seconds=10),
    "3d": Settings(),
}


@dataclass
class Coefficients:
    """見積もりの係数。calibration.json で上書きする。"""

    overhead_gb: float = 0.6  # CUDA の文脈・ComfyUI 自身など
    weight_factor: float = 1.0  # ファイルの大きさ → VRAM 上の大きさ
    resident_factor: float = 1.0  # 全部載せのとき、補助の部品(エンコーダ等)が同時に載る割合
    act_image_gb_per_mp: float = 0.5  # 画像 1 メガ画素あたりの作業用メモリ
    act_video_gb_per_mp: float = 0.6  # 動画 1 メガ画素あたり(コマ数の平方根に比例させる)
    act_video_ref_frames: int = 49
    act_audio_gb_per_min: float = 1.0
    act_3d_gb: float = 2.0
    ram_extra_gb: float = 3.0  # メインメモリの固定分(Python・ComfyUI)
    good_margin: float = 0.90  # perfect はこの割合以内に収まること
    source: str = "default"
    extra: dict = field(default_factory=dict)


def load_coefficients() -> Coefficients:
    try:
        data = json.loads(bundled_text("calibration.json"))
    except (FileNotFoundError, ValueError):
        return Coefficients()
    known = Coefficients.__dataclass_fields__
    c = Coefficients(**{k: v for k, v in data.get("coefficients", {}).items() if k in known})
    c.source = data.get("source", "calibration.json")
    return c


def settings_for(entry: ModelEntry, settings: Settings | None) -> Settings:
    base = DEFAULT_SETTINGS.get(entry.modality, Settings())
    if settings is None:
        return Settings(**vars(base))
    merged = Settings(**vars(base))
    for k, v in vars(settings).items():
        if v is not None:
            setattr(merged, k, v)
    return merged


def activation_gb(entry: ModelEntry, s: Settings, c: Coefficients) -> float:
    mp = s.megapixels or 0.0
    if entry.modality == "image":
        return c.act_image_gb_per_mp * mp * max(1, s.batch)
    if entry.modality == "video":
        frames = s.frames or c.act_video_ref_frames
        return c.act_video_gb_per_mp * mp * math.sqrt(frames / c.act_video_ref_frames) * 4
    if entry.modality in ("music", "sfx", "tts"):
        return c.act_audio_gb_per_min * (s.seconds or 30) / 60
    if entry.modality == "3d":
        return c.act_3d_gb
    return 0.5


def _main_bytes(entry: ModelEntry) -> tuple[float, float]:
    """(全ファイルの合計, 最大のファイル) をバイトで返す。ファイルが無ければテンプレートの size を使う。"""
    total = entry.total_size_bytes or 0
    largest = entry.largest_file_bytes or total
    return float(total), float(largest)


def _settings_match(a: Settings, b: Settings) -> bool:
    keys = ("width", "height", "frames", "seconds")
    return all(getattr(a, k) == getattr(b, k) for k in keys if getattr(a, k) is not None)


def pick_measurement(
    measurements: Iterable[Measurement], entry: ModelEntry, s: Settings
) -> Measurement | None:
    # 失敗した実測(OOM など)と、標準でない条件の実測は「動く」根拠にしない
    cands = [
        m for m in measurements if m.model_id == entry.id and m.outcome == "ok" and m.standard is not False
    ]
    same = [m for m in cands if _settings_match(m.settings, s) or _settings_match(s, m.settings)]
    return max(same, key=lambda m: m.peak_vram_gb) if same else None


def raw_estimate(entry: ModelEntry, s: Settings, c: Coefficients) -> tuple[float, float, float]:
    """(全部載せの VRAM, 逃がしたときの VRAM, メインメモリ) を GB で返す。"""
    total, largest = _main_bytes(entry)
    act = activation_gb(entry, s, c)
    main = largest * c.weight_factor / GB
    rest = max(0.0, total - largest) * c.weight_factor / GB
    full = c.overhead_gb + main + rest * c.resident_factor + act
    offload = c.overhead_gb + main + act
    ram = c.ram_extra_gb + total / GB
    return full, offload, ram


def estimate(
    entry: ModelEntry,
    system: SystemInfo,
    settings: Settings | None = None,
    measurements: Iterable[Measurement] = (),
    coefficients: Coefficients | None = None,
) -> Estimate:
    c = coefficients or load_coefficients()
    s = settings_for(entry, settings)
    full, offload, ram = raw_estimate(entry, s, c)
    basis, measured_on = "estimated", None
    m = pick_measurement(measurements, entry, s)
    if m is not None:
        basis, measured_on = "measured", f"{m.gpu_name} ({m.gpu_vram_gb:g} GB)"
        # 実測は「その GPU で実際に使った最大」。逃がした量は推定の比で割り戻す
        ratio = offload / full if full else 1.0
        full = m.peak_vram_gb
        offload = min(full, max(full * ratio, c.overhead_gb))
        if m.peak_ram_gb:
            ram = max(ram, m.peak_ram_gb)
    disk = (entry.total_size_bytes or 0) / GB or None
    fit, mode, reasons = judge(full, offload, ram, system, c, entry)
    return Estimate(
        vram_gb=round(full, 2),
        vram_offload_gb=round(offload, 2),
        ram_gb=round(ram, 2),
        disk_gb=round(disk, 2) if disk else None,
        basis=basis,
        fit=fit,
        run_mode=mode,
        reasons=reasons,
        measured_on=measured_on,
    )


def judge(
    full: float, offload: float, ram: float, system: SystemInfo, c: Coefficients, entry: ModelEntry
) -> tuple[str, str, list[str]]:
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
    if offload <= vram:
        reasons.append(
            f"VRAM {full:.1f} GB 必要 > {vram:.1f} GB: 部品を逃がせば動く(遅い。逃がすと {offload:.1f} GB)"
        )
        return "marginal", "offload", reasons
    reasons.append(f"VRAM 不足: 逃がしても {offload:.1f} GB 必要(この PC は {vram:.1f} GB)")
    return "too_tight", "none", reasons
