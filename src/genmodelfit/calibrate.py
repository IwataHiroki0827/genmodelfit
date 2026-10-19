"""実測から見積もりの係数を合わせる。`python -m genmodelfit.calibrate`

合わせるのは「全部載せ」の VRAM(estimate の vram_gb)だけ:
    予測 = overhead_gb + 重みの合計 + Σ 作業用メモリの係数 × 設定から決まる量
係数は 5 つ(overhead と、画像・動画・音声・3D の作業用メモリ)。どれも上の式に 1 次で効くので、
「相対誤差の 2 乗和 + 事前値(#1 の仮の値)からのずれへの弱い罰」を、範囲付きの座標降下で最小にする。

- GPU の上限に張り付いた実測(estimate.is_censored)は「予測がこれ以上」という片側の条件としてだけ使う
- 精度は 1 件ずつ外して残りで合わせ、外した 1 件を当てる(leave-one-out)で測る
結果は src/genmodelfit/data/calibration.json に書き、docs/accuracy.md の表を標準出力に出す。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .db import Database
from .db import load as load_db
from .estimate import (
    FITTED,
    Coefficients,
    activation_features,
    components,
    is_censored,
    settings_for,
)
from .models import Measurement

# 範囲(物理的にありえる幅)と事前値。事前値は #1 の仮の値で、実測を見る前に決めたもの
BOUNDS: dict[str, tuple[float, float]] = {
    "overhead_gb": (0.3, 1.5),
    "act_image_gb_per_mp": (0.0, 4.0),
    "act_video_gb_per_mp": (0.0, 30.0),
    "act_audio_gb_per_min": (0.0, 5.0),
    "act_3d_gb": (0.0, 4.0),
}
PRIOR = {k: getattr(Coefficients(), k) for k in FITTED}
RIDGE = 1e-4  # 事前値への罰の重み((値 - 事前値) / 事前値 の 2 乗に掛ける)
SWEEPS = 300


@dataclass
class Row:
    """実測 1 件を「予測 = 重み + Σ 係数 × 特徴」の形にしたもの。"""

    m: Measurement
    weights_gb: float
    features: dict[str, float]  # overhead_gb は常に 1
    censored: bool

    def predict(self, theta: dict[str, float]) -> float:
        return self.weights_gb + sum(theta[k] * v for k, v in self.features.items())


def build_rows(db: Database, c: Coefficients | None = None) -> list[Row]:
    c = c or Coefficients()
    rows = []
    for m in db.measurements:
        e = db.get(m.model_id)
        if e is None:
            continue
        s = settings_for(e, m.settings)
        feats = {"overhead_gb": 1.0, **activation_features(e.modality, s, c)}
        rows.append(Row(m, components(e).total * c.weight_factor, feats, is_censored(m)))
    return rows


def _loss(rows: list[Row], theta: dict[str, float]) -> float:
    total = 0.0
    for r in rows:
        y = r.m.peak_vram_gb
        rel = (r.predict(theta) - y) / y
        if r.censored:
            rel = min(0.0, rel)  # 予測が実測以上なら罰なし
        total += rel * rel
    for k in FITTED:
        total += RIDGE * ((theta[k] - PRIOR[k]) / PRIOR[k]) ** 2
    return total


def _argmin_1d(f, lo: float, hi: float, iters: int = 100) -> float:
    """凸な 1 変数関数の最小(黄金分割)。"""
    g = (5**0.5 - 1) / 2
    a, b = lo, hi
    x1, x2 = b - g * (b - a), a + g * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f1 <= f2:
            b, x2, f2 = x2, x1, f1
            x1 = b - g * (b - a)
            f1 = f(x1)
        else:
            a, x1, f1 = x1, x2, f2
            x2 = a + g * (b - a)
            f2 = f(x2)
    return (a + b) / 2


def fit(rows: list[Row]) -> dict[str, float]:
    """範囲付きの座標降下。目的関数は凸(1 次の予測の 2 乗・片側 2 乗・罰の和)なので初期値によらない。"""
    theta = dict(PRIOR)
    for _ in range(SWEEPS):
        before = _loss(rows, theta)
        for k in FITTED:
            lo, hi = BOUNDS[k]
            theta[k] = _argmin_1d(lambda v, k=k: _loss(rows, {**theta, k: v}), lo, hi)
        if before - _loss(rows, theta) < 1e-12:
            break
    return {k: round(v, 4) for k, v in theta.items()}


def leave_one_out(rows: list[Row]) -> list[dict]:
    out = []
    full = fit(rows)
    for i, r in enumerate(rows):
        theta = fit(rows[:i] + rows[i + 1 :])
        pred = r.predict(theta)
        y = r.m.peak_vram_gb
        out.append(
            {
                "model_id": r.m.model_id,
                "settings": {k: v for k, v in vars(r.m.settings).items() if v is not None and k != "batch"},
                "note": r.m.note,
                "measured_gb": y,
                "censored": r.censored,
                "fit_predicted_gb": round(r.predict(full), 2),
                "loo_predicted_gb": round(pred, 2),
                "loo_error_pct": round((pred - y) / y * 100, 1),
                # 比べる基準: 重み(ファイルの大きさ)だけで当てたときの誤差
                "weights_only_error_pct": round((r.weights_gb - y) / y * 100, 1),
                # 張り付いた実測は「予測 ≥ 実測」なら当たり
                "loo_ok": (pred >= y) if r.censored else None,
            }
        )
    return out


def summarize(loo: list[dict], target_pct: float = 20.0) -> dict:
    errs = [abs(x["loo_error_pct"]) for x in loo if not x["censored"]]
    base = [abs(x["weights_only_error_pct"]) for x in loo if not x["censored"]]
    cens = [x for x in loo if x["censored"]]
    return {
        "target_abs_pct": target_pct,
        "n_point": len(errs),
        "n_censored": len(cens),
        "mean_abs_pct": round(sum(errs) / len(errs), 1) if errs else None,
        "max_abs_pct": round(max(errs), 1) if errs else None,
        "weights_only_mean_abs_pct": round(sum(base) / len(base), 1) if base else None,
        "censored_ok": sum(1 for x in cens if x["loo_ok"]),
        "target_met": bool(errs) and max(errs) <= target_pct and all(x["loo_ok"] for x in cens),
    }


def calibrate(db: Database | None = None, date: str | None = None) -> dict:
    db = db or load_db(use_cache=False)
    rows = build_rows(db)
    theta = fit(rows)
    loo = leave_one_out(rows)
    fixed = Coefficients()
    return {
        "source": "calibration.json",
        "date": date or _dt.date.today().isoformat(),
        "method": (
            "全部載せの VRAM = overhead + 重み + 作業用メモリ。5 係数を相対誤差の 2 乗和"
            "(張り付いた実測は片側)"
            f" + 事前値への罰(重み {RIDGE})で、範囲付きの座標降下により最小化。精度は leave-one-out"
        ),
        "n": len(rows),
        "n_censored": sum(r.censored for r in rows),
        "coefficients": theta,
        "prior": PRIOR,
        "bounds": {k: list(v) for k, v in BOUNDS.items()},
        "fixed": {
            k: getattr(fixed, k)
            for k in (
                "weight_factor",
                "act_video_ref_frames",
                "act_video_frame_exponent",
                "offload_fraction",
                "ram_extra_gb",
                "good_margin",
            )
        },
        "loo": loo,
        "summary": summarize(loo),
    }


def markdown_table(result: dict) -> str:
    lines = [
        "| モデル | 設定 | 実測 GB | 全件で合わせた予測 GB | LOO 予測 GB | LOO 誤差 | 重みだけの誤差 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for x in result["loo"]:
        s = x["settings"]
        bits = []
        if s.get("width"):
            bits.append(f"{s['width']}x{s['height']}")
        if s.get("frames"):
            bits.append(f"{s['frames']} コマ")
        if s.get("seconds"):
            bits.append(f"{s['seconds']:g} 秒")
        if "文章から" in (x.get("note") or ""):
            bits.append("文章から")
        err = (
            f"張り付き({'≥ 実測で当たり' if x['loo_ok'] else '実測より小さい: 外れ'})"
            if x["censored"]
            else f"{x['loo_error_pct']:+.1f}%"
        )
        lines.append(
            f"| {x['model_id']} | {'・'.join(bits) or '-'} | {x['measured_gb']:.1f} |"
            f" {x['fit_predicted_gb']:.2f} | {x['loo_predicted_gb']:.2f} | {err} |"
            f" {x['weights_only_error_pct']:+.1f}% |"
        )
    sm = result["summary"]
    lines.append("")
    lines.append(
        f"張り付いていない {sm['n_point']} 件: 平均絶対誤差 {sm['mean_abs_pct']}%、最大 {sm['max_abs_pct']}%"
        f"(重みだけなら平均 {sm['weights_only_mean_abs_pct']}%)。"
        f" 張り付いた {sm['n_censored']} 件のうち {sm['censored_ok']} 件で予測 ≥ 実測。"
        f" 目標 ±{sm['target_abs_pct']:g}%: {'達成' if sm['target_met'] else '未達'}"
    )
    return "\n".join(lines)


def default_output() -> Path:
    return Path(__file__).parent / "data" / "calibration.json"


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="python -m genmodelfit.calibrate", description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=default_output(), help="書き出し先")
    p.add_argument("--dry-run", action="store_true", help="書き出さずに表だけ出す")
    p.add_argument("--date", help="記録する日付(既定は今日)")
    args = p.parse_args(argv)
    result = calibrate(date=args.date)
    if not args.dry_run:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(json.dumps(result["coefficients"], ensure_ascii=False))
    print(markdown_table(result))
    return 0


__all__ = ["Row", "build_rows", "calibrate", "fit", "leave_one_out", "main", "markdown_table"]

if __name__ == "__main__":
    raise SystemExit(main())
