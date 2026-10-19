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
from .estimate import estimate, load_coefficients
from .hardware import detect
from .models import CONTENT_LEVELS, FIT_LEVELS, MODALITIES, Settings, SystemInfo
from .recommend import evaluate, recommend


class CliError(RuntimeError):
    pass


def _out(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _system(args: argparse.Namespace) -> SystemInfo:
    return detect(disk_path=getattr(args, "disk_path", None), vram_gb=args.vram, ram_gb=args.ram)


def _db(args: argparse.Namespace) -> Database:
    return load_db(args.db, use_cache=not args.no_cache)


def _settings(args: argparse.Namespace) -> Settings | None:
    s = Settings(
        width=args.width, height=args.height, frames=args.frames, seconds=args.length, precision=None
    )
    return s if any(v is not None for v in (s.width, s.height, s.frames, s.seconds)) else None


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


def _print_fits(fits, title: str) -> None:
    from .table import print_fit_table

    print_fit_table(fits, title=title)


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
    _print_fits(fits, "Recommended")
    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    fits = evaluate(db, s, modality=args.modality, content=args.content, settings=_settings(args))
    if args.perfect:
        fits = [f for f in fits if f.estimate.fit == "perfect"]
    if args.limit:
        fits = fits[: args.limit]
    _attach_gate(fits, args)
    if args.json:
        _out(_envelope("fit", s, db, models=[f.to_dict() for f in fits]))
        return 0
    _print_fits(fits, "All models")
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
    e = estimate(m, s, _settings(args), db.measurements, c)
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
    size = (m.total_size_bytes or 0) / 1e9
    print(f"  size:     {size:.2f} GB in {len(m.files)} file(s)")
    for f in m.files:
        fs = f"{(f.size_bytes or 0) / 1e9:.2f} GB" if f.size_bytes else "?"
        print(f"    - {f.name} ({fs}, {f.directory or '-'}, license {f.license or '?'})")
    print(f"  fit:      {e.fit}  ({e.basis}{', on ' + e.measured_on if e.measured_on else ''})")
    print(
        f"  VRAM:     {e.vram_gb:.1f} GB all-in-GPU / {e.vram_offload_gb:.1f} GB with offload;"
        f" RAM {e.ram_gb:.1f} GB"
    )
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
    common.add_argument("--vram", type=float, metavar="GB", help="GPU の VRAM を上書きする(例: 8)")
    common.add_argument("--ram", type=float, metavar="GB", help="メインメモリを上書きする(例: 32)")
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
    filt.add_argument("--width", type=int)
    filt.add_argument("--height", type=int)
    filt.add_argument("--frames", type=int, help="動画のコマ数")
    filt.add_argument("--length", type=float, metavar="SEC", help="音声の長さ(秒)")
    filt.add_argument("--limit", "-n", type=int)

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
    sp.add_argument(
        "--url", required=True, help="測りに使う ComfyUI の URL(例 http://127.0.0.1:8188。既定は持たない)"
    )
    sp.add_argument("--workflow", metavar="PATH", help="API 形式のワークフロー(省略時はテンプレートから作る)")
    sp.add_argument("--width", type=int)
    sp.add_argument("--height", type=int)
    sp.add_argument("--frames", type=int)
    sp.add_argument("--length", type=float, metavar="SEC")
    sp.add_argument("--steps", type=int)
    sp.add_argument(
        "--output", metavar="PATH", help="結果の JSON の書き出し先(失敗・標準でない結果もここへ書く)"
    )
    sp.add_argument(
        "--save-failed",
        action="store_true",
        help="失敗(OOM など)の結果もキャッシュに保存する(既定は表示だけ)",
    )
    sp.add_argument(
        "--save-nonstandard",
        action="store_true",
        help="標準でない条件(--workflow・--set・一部だけ実行など)の結果もキャッシュに保存する",
    )
    sp.add_argument("--resolution", metavar="WxH", help="幅と高さ(例: 768x768)。--width/--height と同じ")
    sp.add_argument(
        "--set", action="append", metavar="NAME=VALUE", help="入力を名前で変える(例: octree_resolution=192)"
    )
    sp.add_argument(
        "--free",
        action="store_true",
        help="開始前に /free でその ComfyUI のモデルを解放する(自分だけが使う ComfyUI のときに)",
    )
    sp.add_argument(
        "--timeout", type=float, metavar="SEC", help="待つ時間の上限(0 より大きい秒数。既定 1800)"
    )
    sp.add_argument(
        "--force", action="store_true", help="GPU が使われている・GPU の空きを確かめられないときも測る"
    )
    sp.add_argument("--source", help="誰がどこで測ったか(結果の source に入る)")
    sp.add_argument("--note", help="結果の note に足す文")
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
