"""コマンドライン。`--json` を付けると、どのコマンドも決まった形の JSON を出す(docs/schema.md)。

終了コード: 0 成功 / 1 失敗 / 2 使い方の誤り。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__
from .db import Database, DatabaseError
from .db import load as load_db
from .estimate import GIB, estimate, load_coefficients
from .hardware import detect
from .models import CONTENT_LEVELS, FIT_LEVELS, MODALITIES, Settings, SystemInfo
from .recommend import evaluate, recommend


class CliError(RuntimeError):
    pass


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数を指定してください: {text}") from None
    if v <= 0:
        raise argparse.ArgumentTypeError(f"1 以上を指定してください: {text}")
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数を指定してください: {text}") from None
    if not v > 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(f"0 より大きい値を指定してください: {text}")
    return v


def _non_negative_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数を指定してください: {text}") from None
    if not v >= 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(f"0 以上を指定してください: {text}")
    return v


def _out(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _system(args: argparse.Namespace) -> SystemInfo:
    return detect(disk_path=getattr(args, "disk_path", None), vram_gb=args.vram, ram_gb=args.ram)


def _db(args: argparse.Namespace) -> Database:
    return load_db(args.db, use_cache=not args.no_cache)


def _settings(args: argparse.Namespace) -> Settings | None:
    s = Settings(
        width=args.width,
        height=args.height,
        frames=args.frames,
        seconds=args.length,
        precision=getattr(args, "precision", None),
    )
    return s if any(v is not None for v in (s.width, s.height, s.frames, s.seconds, s.precision)) else None


def _envelope(command: str, system: SystemInfo | None, db: Database | None, **payload: Any) -> dict[str, Any]:
    d: dict[str, Any] = {"genmodelfit": __version__, "command": command}
    if system is not None:
        d["system"] = system.to_dict()
    if db is not None:
        d["database"] = {
            "origin": db.origin,
            "generated_at": db.generated_at,
            "models": len(db.models),
            "measurements": len(db.measurements),
        }
    d.update(payload)
    return d


def format_limits(lim: dict) -> str:
    """「この PC なら 長辺 640px・133 コマまで good」の形にする。"""
    caps = set(lim.get("at_cap") or [])
    keys = [k for k in ("max_side_px", "max_frames", "max_seconds") if k in lim]
    if keys and all(lim[k] is None for k in keys):
        return "この PC では、解像度や長さを下げても good にならない"
    parts = []
    for key, label, unit in (
        ("max_side_px", "長辺 ", "px"),
        ("max_frames", "", " コマ"),
        ("max_seconds", "", " 秒"),
    ):
        if key not in lim:
            continue
        v = lim[key]
        if v is None:
            parts.append(f"{label or unit.strip() + ' '}は下げても good にならない")
        else:
            parts.append(f"{label}{v}{unit}{'(調べた上限)' if key in caps else ''}")
    return "この PC なら " + "・".join(parts) + "まで good(ほかの設定はそのまま)"


# --- コマンド ---------------------------------------------------------------------------------


def cmd_system(args: argparse.Namespace) -> int:
    s = _system(args)
    if args.json:
        _out(_envelope("system", s, None))
        return 0
    print("=== System ===")
    print(f"OS:   {s.os}")
    print(f"CPU:  {s.cpu} ({s.cpu_cores} threads)")
    print(f"RAM:  {s.ram_total_gb:.1f} GB (available {s.ram_available_gb:.1f} GB)")
    if s.disk_free_gb is not None:
        print(f"Disk: {s.disk_free_gb:.1f} GB free")
    if not s.gpus:
        print("GPU:  none detected (CPU only)")
    for i, g in enumerate(s.gpus, 1):
        free = f", {g.vram_free_gb:.1f} GB free" if g.vram_free_gb is not None else ""
        print(f"GPU {i}: {g.name} ({g.vram_total_gb:.1f} GB VRAM{free}, driver {g.driver}, {g.backend})")
    if s.overrides:
        print(f"(overridden: {', '.join(s.overrides)})")
    return 0


def _print_fits(fits, title: str, system: SystemInfo | None = None) -> None:
    from .table import print_fit_table

    print_fit_table(fits, title=title, system=system)


def cmd_recommend(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    fits = recommend(
        db,
        s,
        modality=args.modality,
        content=args.content,
        min_fit=args.min_fit,
        limit=args.limit,
        settings=_settings(args),
        with_limits=args.limits,
    )
    _attach_gate(fits, args)
    if args.auto_only:
        fits = [f for f in fits if f.gate and f.gate.get("auto")]
    if args.json:
        _out(
            _envelope(
                "recommend",
                s,
                db,
                filters={
                    "modality": args.modality,
                    "content": args.content,
                    "min_fit": args.min_fit,
                    "limit": args.limit,
                },
                models=[f.to_dict() for f in fits],
            )
        )
        return 0
    _print_fits(fits, "Recommended", s)
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    fits = evaluate(
        db,
        s,
        modality=args.modality,
        content=args.content,
        settings=_settings(args),
        with_limits=args.limits,
    )
    if args.perfect:
        fits = [f for f in fits if f.estimate.fit == "perfect"]
    if args.limit:
        fits = fits[: args.limit]
    _attach_gate(fits, args)
    if args.json:
        _out(_envelope("fit", s, db, models=[f.to_dict() for f in fits]))
        return 0
    _print_fits(fits, "All models", s)
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    hits = db.find(args.model)
    if not hits:
        raise CliError(f"モデルが見つかりません: {args.model}")
    if len(hits) > 1 and not args.json:
        print(f"{len(hits)} 件見つかりました。ID で指定してください:")
        for m in hits[:30]:
            print(f"  {m.id}  ({m.name})")
        return 0
    m = hits[0]
    c = load_coefficients()
    e = estimate(m, s, _settings(args), db.measurements, c, with_limits=True)
    gate = _gate_for(m, args)
    if args.json:
        _out(
            _envelope(
                "info",
                s,
                db,
                model=m.to_dict(),
                estimate=e.to_dict(),
                gate=gate,
                measurements=[vars(x) | {"settings": vars(x.settings)} for x in db.measurements_for(m.id)],
            )
        )
        return 0
    print(f"{m.name}  [{m.id}]")
    print(f"  modality: {m.modality}  tasks: {', '.join(m.tasks) or '-'}  content: {m.content}")
    print(f"  license:  {m.license or 'unknown'}  source: {m.source}")
    counted = f"{e.disk_gb:.2f} GB counted" if e.disk_gb else "size unknown"
    print(f"  size:     {counted}, {len(m.files)} file(s) listed (GB = GiB)")
    for f in m.files:
        fs = f"{f.size_bytes / GIB:.2f} GB" if f.size_bytes and f.size_bytes > 0 else "?"
        print(f"    - {f.name} ({fs}, {f.directory or '-'}, license {f.license or '?'})")
    print(f"  fit:      {e.fit}  ({e.basis}{', on ' + e.measured_on if e.measured_on else ''})")
    seq = f" / {e.vram_sequential_gb:.1f} GB swapping parts" if e.vram_sequential_gb is not None else ""
    if e.unknown_size:
        print("  VRAM:     unknown (weight sizes are not known)")
    else:
        print(
            f"  VRAM:     {e.vram_gb:.1f} GB all-in-GPU{seq} / {e.vram_offload_gb:.1f} GB with offload;"
            f" RAM {e.ram_gb:.1f} GB"
        )
    if e.limits:
        print(f"  limits:   {format_limits(e.limits)}")
    for r in e.reasons:
        print(f"  - {r}")
    if gate:
        print(f"  auto-recommend: {'yes' if gate.get('auto') else 'held'}")
        for chk in gate.get("checks", []):
            mark = "ok" if chk["passed"] else "NG"
            print(f"    [{mark}] {chk['id']}: {chk.get('detail', '')}")
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    try:
        from .update import run_update
    except ImportError as e:  # #2 で実装
        raise CliError("update はまだ実装されていません(Issue #2)") from e
    return run_update(args)


def cmd_bench(args: argparse.Namespace) -> int:
    try:
        from .bench import run_bench
    except ImportError as e:  # #5 で実装
        raise CliError("bench はまだ実装されていません(Issue #5)") from e
    return run_bench(args)


# --- 自動でおすすめに出せるか(#3 で実装。無ければ None) ---------------------------------------


def _gate_for(model, args) -> dict[str, Any] | None:
    try:
        from .gate import evaluate_offline
    except ImportError:
        return None
    return evaluate_offline(model).to_dict()


def _attach_gate(fits, args) -> None:
    for f in fits:
        f.gate = _gate_for(f.model, args)


# --- 引数 ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="JSON で出す")
    common.add_argument(
        "--vram", type=_non_negative_float, metavar="GB", help="GPU の VRAM を上書きする(例: 8)"
    )
    common.add_argument("--ram", type=_positive_float, metavar="GB", help="メインメモリを上書きする(例: 32)")
    common.add_argument("--db", metavar="PATH", help="一覧のファイルを指定する")
    common.add_argument(
        "--no-cache", action="store_true", help="update で取った一覧を使わず、同梱のデータを使う"
    )

    filt = argparse.ArgumentParser(add_help=False)
    filt.add_argument("--modality", "-m", choices=MODALITIES)
    filt.add_argument(
        "--content",
        choices=("all", *CONTENT_LEVELS),
        default="all",
        help="sfw / uncensored(無検閲)で絞る。既定は all",
    )
    filt.add_argument("--width", type=_positive_int)
    filt.add_argument("--height", type=_positive_int)
    filt.add_argument("--frames", type=_positive_int, help="動画のコマ数")
    filt.add_argument("--length", type=_positive_float, metavar="SEC", help="音声の長さ(秒)")
    filt.add_argument("--precision", help="本体の精度を換算する(例: fp8。元の精度が分かるときだけ)")
    filt.add_argument("--limit", "-n", type=int)
    filt.add_argument(
        "--limits", action="store_true", help="この PC で good になる最大の解像度・長さも出す(JSON の limits)"
    )

    p = argparse.ArgumentParser(prog="genmodelfit", description="Right-size generation models to your PC.")
    p.add_argument("--version", action="version", version=f"genmodelfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("system", parents=[common], help="この PC の GPU・メモリ・空きを出す")
    sp.add_argument("--disk-path", help="空きを調べる場所(既定はホーム)")
    sp.set_defaults(func=cmd_system)

    sp = sub.add_parser("recommend", parents=[common, filt], help="おすすめ順に出す")
    sp.add_argument("--min-fit", choices=FIT_LEVELS[:3], default="marginal")
    sp.add_argument("--auto-only", action="store_true", help="自動でおすすめに出せるものだけ")
    sp.set_defaults(func=cmd_recommend, limit=10)

    sp = sub.add_parser("fit", parents=[common, filt], help="すべてのモデルの判定を表で出す")
    sp.add_argument("--perfect", "-p", action="store_true", help="perfect だけ")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("info", parents=[common, filt], help="1 つのモデルの詳細")
    sp.add_argument("model", help="モデルの ID または名前の一部")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("update", parents=[common], help="一覧を上流から作り直す")
    sp.add_argument("--offline-sources", metavar="DIR", help="(試験用)上流の代わりに使うファイル")
    sp.add_argument("--skip-hf", action="store_true", help="Hugging Face を見ない")
    sp.add_argument("--skip-head", action="store_true", help="URL の HEAD 確認を飛ばす")
    sp.add_argument("--output", metavar="PATH", help="書き出し先(既定はキャッシュ)")
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("bench", parents=[common], help="動いている ComfyUI で実測する")
    sp.add_argument("template", help="テンプレート名(モデルの ID)")
    sp.add_argument("--url", default="http://127.0.0.1:8188", help="ComfyUI の URL")
    sp.add_argument("--workflow", metavar="PATH", help="API 形式のワークフロー(省略時はテンプレートから作る)")
    sp.add_argument("--width", type=_positive_int)
    sp.add_argument("--height", type=_positive_int)
    sp.add_argument("--frames", type=_positive_int)
    sp.add_argument("--length", type=_positive_float, metavar="SEC")
    sp.add_argument("--steps", type=_positive_int)
    sp.add_argument("--output", metavar="PATH", help="結果の JSON の書き出し先")
    sp.set_defaults(func=cmd_bench)
    return p


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows の cp932 でも落ちないように
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (CliError, DatabaseError) as e:
        if getattr(args, "json", False):
            _out({"genmodelfit": __version__, "command": args.command, "error": str(e)})
        else:
            print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
