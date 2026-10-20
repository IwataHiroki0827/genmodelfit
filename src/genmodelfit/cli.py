"""コマンドライン。`--json` を付けると、どのコマンドも決まった形の JSON を出す(docs/schema.md)。

終了コード: 0 成功 / 1 失敗 / 2 使い方の誤り。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .db import Database, DatabaseError
from .db import load as load_db
from .estimate import GIB, estimate, load_coefficients
from .hardware import detect
from .models import FIT_LEVELS, MODALITIES, Settings, SystemInfo
from .recommend import CONTENT_FILTERS, DEFAULT_CONTENT, evaluate, is_tool, recommend


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


def _print_fits(fits, title: str, system: SystemInfo | None = None, notes: list[str] | None = None) -> None:
    from .table import print_fit_table

    print_fit_table(fits, title=title, system=system, notes=notes or ())


def cmd_recommend(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    limit = args.limit if args.limit is not None else 10
    fits = recommend(
        db,
        s,
        modality=args.modality,
        content=args.content,
        min_fit=args.min_fit,
        limit=None,  # 保留・補助ツールを除いてから件数を絞る
        settings=_settings(args),
        with_limits=args.limits,
    )
    if not args.include_tools:
        fits = [f for f in fits if not is_tool(f.model)]
    _attach_gate(fits, args, db)
    if args.auto_only:
        fits = [f for f in fits if f.gate and f.gate.get("auto")]
    fits = fits[:limit] if limit else fits
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
                    "limit": args.limit if args.limit is not None else 10,
                    "auto_only": args.auto_only,
                    "include_tools": args.include_tools,
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
    _attach_gate(fits, args, db)
    if args.json:
        _out(_envelope("fit", s, db, models=[f.to_dict() for f in fits]))
        return 0
    notes = []
    if args.content == "sfw":  # 既定で隠した無検閲の数を表の下で知らせる(文字の表示だけ)
        hidden = sum(
            1
            for m in db.models
            if m.content == "uncensored" and (not args.modality or m.modality == args.modality)
        )
        if hidden:
            notes.append(f"uncensored hidden: {hidden} (--content all)")
    _print_fits(fits, "All models", s, notes)
    return 0


class AmbiguousModel(CliError):
    def __init__(self, query: str, candidates: list) -> None:
        super().__init__(f"「{query}」に合うモデルが {len(candidates)} 件あります。ID で指定してください")
        self.candidates = candidates


def resolve_model(db: Database, query: str, content: str = DEFAULT_CONTENT):
    """ID がそのまま合えば(内容によらず)そのモデル。合わなければ ID・名前の一部で探す。

    一部で探すときは content で絞る(既定 sfw なので無検閲は出ない)。1 件だけならそれ、
    2 件以上なら AmbiguousModel(先頭を黙って選ばない)、0 件なら CliError。
    """
    exact = db.get(query)
    if exact is not None:
        return exact
    q = query.lower()
    hits = [m for m in db.models if q in m.id.lower() or q in m.name.lower()]
    shown = [m for m in hits if content == "all" or m.content == content]
    if len(shown) == 1:
        return shown[0]
    if shown:
        raise AmbiguousModel(query, shown)
    if hits:
        raise CliError(
            f"モデルが見つかりません: {query}(無検閲のモデルなら {len(hits)} 件あります。"
            "--content all を付けるか ID で指定してください)"
        )
    raise CliError(f"モデルが見つかりません: {query}")


def _emit(lines: list[str]) -> None:
    """info の文を幅(COLUMNS か端末、最大 100 桁)に収めて出す。"""
    from .table import output_width, wrap_line

    width = output_width()
    for line in lines:
        for part in wrap_line(line, width):
            print(part)


def cmd_info(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    m = resolve_model(db, args.model, args.content)
    c = load_coefficients()
    e = estimate(m, s, _settings(args), db.measurements, c, with_limits=True)
    gate = _gate_for(m, args, db)
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
    out = [f"{m.name}  [{m.id}]"]
    out.append(f"  modality: {m.modality}  tasks: {', '.join(m.tasks) or '-'}  content: {m.content}")
    out.append(f"  license:  {m.license or 'unknown'}  source: {m.source}")
    counted = f"{e.disk_gb:.2f} GB counted" if e.disk_gb else "size unknown"
    out.append(f"  size:     {counted}, {len(m.files)} file(s) listed (GB = GiB)")
    for f in m.files:
        fs = f"{f.size_bytes / GIB:.2f} GB" if f.size_bytes and f.size_bytes > 0 else "?"
        out.append(f"    - {f.name} ({fs}, {f.directory or '-'}, license {f.license or '?'})")
    out.append(f"  fit:      {e.fit}  ({e.basis}{', on ' + e.measured_on if e.measured_on else ''})")
    seq = f" / {e.vram_sequential_gb:.1f} GB swapping parts" if e.vram_sequential_gb is not None else ""
    if e.unknown_size:
        out.append("  VRAM:     unknown (weight sizes are not known)")
    else:
        out.append(
            f"  VRAM:     {e.vram_gb:.1f} GB all-in-GPU{seq} / {e.vram_offload_gb:.1f} GB with offload;"
            f" RAM {e.ram_gb:.1f} GB"
        )
    if e.limits:
        out.append(f"  limits:   {format_limits(e.limits)}")
    out += [f"  - {r}" for r in e.reasons]
    if gate:
        target = gate.get("comfyui_version") or "?"
        out.append(f"  auto-recommend: {'yes' if gate.get('auto') else 'held'}  (ComfyUI {target})")
        for chk in gate.get("checks", []):
            mark = "ok" if chk["passed"] else "NG"
            out.append(f"    [{mark}] {chk['id']}: {chk.get('detail', '')}")
    _emit(out)
    return 0


def _progress(args: argparse.Namespace):
    """途中経過を stderr に出す関数(--json のときは出さない)。"""

    def say(msg: str) -> None:
        if not getattr(args, "json", False):
            print(msg, file=sys.stderr, flush=True)

    return say


def cmd_update(args: argparse.Namespace) -> int:
    if getattr(args, "from_release", False):
        return _update_from_release(args)
    from .db import cached_db_path
    from .update import UpdateFailed, make_fetcher, refresh_heads, update_database

    out_path = Path(args.output) if getattr(args, "output", None) else cached_db_path()
    say = _progress(args)
    fetch = make_fetcher(getattr(args, "offline_sources", None), getattr(args, "rate_limit_budget", None))
    try:
        if getattr(args, "heads_only", False):
            db = refresh_heads(
                fetch,
                out_path,
                db_path=getattr(args, "db", None),
                use_cache=not getattr(args, "no_cache", False),
                progress=say,
            )
        else:
            db = update_database(
                fetch,
                out_path,
                hf=not getattr(args, "skip_hf", False),
                head=not getattr(args, "skip_head", False),
                allow_partial=bool(getattr(args, "allow_partial", False)),
                progress=say,
            )
    except UpdateFailed as e:
        if args.json:
            _out({"genmodelfit": __version__, "command": "update", "error": str(e)})
        else:
            print(f"エラー: {e}", file=sys.stderr)
        return 1
    if args.json:
        db.origin = "path"
        _out(_envelope("update", None, db, output=str(out_path), sources=db.sources))
        return 0
    if getattr(args, "heads_only", False):
        stats = db.sources["head_check"]
        print(f"HEAD の結果を書き出しました: {out_path}")
        print(f"  URL {stats['urls']} 件: " + ", ".join(f"{k} {v}" for k, v in stats["status"].items()))
        latest = db.sources["comfyui"]["latest_release"]
        print(f"  未確認 {stats['unchecked']} 件 / ComfyUI の最新の版: {latest}")
        return 0
    counts = db.sources["counts"]
    mods = ", ".join(f"{k} {v}" for k, v in counts["modality"].items())
    print(f"{counts['models']} 件を書き出しました: {out_path}")
    print(f"  種類: {mods}")
    print("  内容: " + ", ".join(f"{k} {v}" for k, v in counts["content"].items()))
    print("  出どころ: " + ", ".join(f"{k} {v}" for k, v in counts["source"].items()))
    print("  ライセンス: " + ", ".join(f"{k} {v}" for k, v in counts["license_tier"].items()))
    print(f"  一式がそろっていない(files_complete=false): {counts['files_complete_false']} 件")
    print(f"  除外: {db.sources['excluded']} 件 / 警告: {len(db.sources['warnings'])} 件")
    if db.sources.get("partial"):
        pc = json.dumps(db.sources["partial_counts"], ensure_ascii=False)
        print(f"  注意: 一部を取得できないまま書き出しました(--allow-partial): {pc}")
    return 0


def _update_from_release(args: argparse.Namespace) -> int:
    from .db import cached_db_path
    from .remote import update_from_release

    dest = Path(args.output) if getattr(args, "output", None) else cached_db_path()
    db = update_from_release(dest=dest, force=bool(getattr(args, "force", False)))
    if args.json:
        out = {
            "genmodelfit": __version__,
            "command": "update",
            "models": len(db.models),
            "generated_at": db.generated_at,
            "output": str(dest),
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        print(f"一覧を更新しました: {len(db.models)} 件(生成日時 {db.generated_at}) -> {dest}")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from .bench import BenchError, options_from_args, run, save, save_decision
    from .workflow import WorkflowError

    try:
        result = run(options_from_args(args), log=_progress(args))
    except (BenchError, WorkflowError) as e:
        raise CliError(str(e)) from e
    m = result.measurement
    out_path, not_saved = save_decision(
        m, args.output, getattr(args, "save_failed", False), getattr(args, "save_nonstandard", False)
    )
    if out_path is not None:
        save(m, out_path)
    code = 1 if result.failed else 0
    if args.json:
        extra = {"saved_to": str(out_path) if out_path else None}
        if not_saved:
            extra["not_saved"] = not_saved
        _out(_envelope("bench", None, None, **extra, **result.to_dict()))
        return code
    for w in result.warnings:
        print(f"注意: {w}", file=sys.stderr)
    s = m.settings
    parts = [f"{s.width}x{s.height}" if s.width else None, f"{s.frames} フレーム" if s.frames else None]
    parts += [f"{s.seconds:g} 秒の音声" if s.seconds else None, f"{s.steps} ステップ" if s.steps else None]
    print(f"=== bench: {m.model_id} ===")
    print(f"GPU:      {m.gpu_name}({m.gpu_vram_gb:g} GiB、ドライバ {m.driver or '不明'})")
    print(f"ComfyUI:  {m.comfyui_version or '不明'}")
    print(f"設定:     {'、'.join(p for p in parts if p) or '(テンプレートのまま)'}")
    print(f"結果:     {m.outcome}{'' if m.standard else '(標準でない条件)'}")
    print(f"最大 VRAM: {m.peak_vram_gb:.2f} GB(ComfyUI のプロセス)")
    if m.peak_ram_gb is not None:
        print(f"最大メモリ: {m.peak_ram_gb:.2f} GB")
    print(f"時間:     {m.seconds:g} 秒(送ってから終わるまで、モデルの読み込み込み)")
    print(f"載せ方:   {m.offload or '不明'}")
    print(f"メモ:     {m.note}")
    if result.failed:
        print(f"失敗:     {result.failed}")
    print(f"保存先:   {out_path}" if out_path else f"保存:     {not_saved}")
    return code


def cmd_gate(args: argparse.Namespace) -> int:
    from .gate import CHECK_IDS, evaluate_offline, summarize

    db = _db(args)
    policy = _policy(args)
    models = [m for m in db.models if not args.modality or m.modality == args.modality]
    results = [evaluate_offline(m, policy, sources=db.sources) for m in models]
    summary = summarize(results)
    shown = results
    if args.only_auto:
        shown = [r for r in results if r.auto]
    elif args.only_held:
        shown = [r for r in results if not r.auto]
    target = results[0].comfyui_version if results else (policy.comfyui_version or None)
    if args.json:
        _out(
            _envelope(
                "gate",
                None,
                db,
                policy=policy.to_dict() | {"comfyui_version": target},
                filters={"modality": args.modality, "only_auto": args.only_auto, "only_held": args.only_held},
                summary=summary,
                models=[{"id": r.model_id} | r.to_dict() for r in shown],
            )
        )
        return 0
    print(f"=== Auto-recommend gate (ComfyUI {target or '?'}) ===")
    print(f"models {summary['models']}: auto {summary['auto']} / held {summary['held']}")
    print("check               failing  first-failing")
    for cid in CHECK_IDS:
        print(f"  {cid:<18}{summary['failing'][cid]:>8}  {summary['first_failing'][cid]:>13}")
    if args.only_auto or args.only_held:
        print()
        for r in shown:
            print(f"  {r.model_id}  {'auto' if r.auto else 'held:' + r.held_by[0]}")
    return 0


# --- 自動でおすすめに出せるか(gate.py) --------------------------------------------------------


def _policy(args: argparse.Namespace):
    cached = getattr(args, "_gate_policy", None)
    if cached is not None:
        return cached
    from .gate import DEFAULT_LICENSE_ALLOWLIST, DEFAULT_MAX_HEAD_AGE_DAYS, GatePolicy, load_allowlist

    allow = DEFAULT_LICENSE_ALLOWLIST
    if getattr(args, "license_allowlist", None):
        allow = load_allowlist(args.license_allowlist)
        if not allow:
            raise CliError("--license-allowlist が空です")
    policy = GatePolicy(
        license_allowlist=allow,
        comfyui_version=getattr(args, "comfyui_version", None),
        allow_uncensored=bool(getattr(args, "allow_uncensored_auto", False)),
        max_head_age_days=getattr(args, "max_head_age_days", DEFAULT_MAX_HEAD_AGE_DAYS),
    )
    args._gate_policy = policy  # fit では全モデルで使うので、許可リストのファイルを読むのは 1 回だけ
    return policy


def _gate_for(model, args, db: Database | None = None) -> dict[str, Any] | None:
    from .gate import evaluate_offline

    return evaluate_offline(model, _policy(args), sources=db.sources if db else None).to_dict()


def _attach_gate(fits, args, db: Database | None = None) -> None:
    for f in fits:
        f.gate = _gate_for(f.model, args, db)


# --- 引数 ---------------------------------------------------------------------------------------


def _positive_int(v: str) -> int:
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数ではありません: {v}") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"1 以上を指定してください: {v}")
    return n


def _head_age(v: str) -> int | None:
    if v.strip().lower() == "none":
        return None
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f"日数(整数)か none を指定してください: {v}") from None
    if n < 0:
        raise argparse.ArgumentTypeError(f"0 以上を指定してください: {v}")
    return n


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
        choices=CONTENT_FILTERS,
        default=DEFAULT_CONTENT,
        help="sfw / uncensored(無検閲)/ all で絞る。既定は sfw(無検閲は all か uncensored のときだけ出す)",
    )
    filt.add_argument("--width", type=_positive_int)
    filt.add_argument("--height", type=_positive_int)
    filt.add_argument("--frames", type=_positive_int, help="動画のコマ数")
    filt.add_argument("--length", type=_positive_float, metavar="SEC", help="音声の長さ(秒)")
    filt.add_argument("--precision", help="本体の精度を換算する(例: fp8。元の精度が分かるときだけ)")
    filt.add_argument("--limit", "-n", type=_positive_int, help="出す件数(1 以上)")
    filt.add_argument(
        "--limits", action="store_true", help="この PC で good になる最大の解像度・長さも出す(JSON の limits)"
    )

    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument(
        "--license-allowlist",
        metavar="LIST|FILE",
        help="自動のおすすめに許すライセンス(「mit,apache-2.0」か、1 行に 1 つ書いたファイル)。"
        "既定は docs/gate.md",
    )
    gate.add_argument(
        "--comfyui-version",
        metavar="VER",
        help="使う ComfyUI の版。アプリは自分の版を必ず渡す(既定は update が記録した最新のリリース)",
    )
    gate.add_argument(
        "--max-head-age-days",
        type=_head_age,
        default=30,
        metavar="DAYS",
        help="HEAD の確認をこの日数まで信じる(既定 30。none で期限なし)",
    )
    gate.add_argument(
        "--allow-uncensored-auto", action="store_true", help="無検閲のモデルも自動のおすすめに出してよい"
    )

    p = argparse.ArgumentParser(prog="genmodelfit", description="Right-size generation models to your PC.")
    p.add_argument("--version", action="version", version=f"genmodelfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("system", parents=[common], help="この PC の GPU・メモリ・空きを出す")
    sp.add_argument("--disk-path", help="空きを調べる場所(既定はホーム)")
    sp.set_defaults(func=cmd_system)

    sp = sub.add_parser("recommend", parents=[common, filt, gate], help="おすすめ順に出す")
    sp.add_argument("--min-fit", choices=FIT_LEVELS[:3], default="marginal")
    sp.add_argument("--auto-only", action="store_true", help="自動でおすすめに出せるものだけ")
    sp.add_argument(
        "--include-tools",
        action="store_true",
        help="深度・ポーズ・切り抜き・拡大などの補助ツールも含める(既定は「つくる」モデルだけ)",
    )
    sp.set_defaults(func=cmd_recommend)  # 既定の 10 件は cmd_recommend で決める(親の引数は fit と共有なので)

    sp = sub.add_parser("fit", parents=[common, filt, gate], help="すべてのモデルの判定を表で出す")
    sp.add_argument("--perfect", "-p", action="store_true", help="perfect だけ")
    sp.set_defaults(func=cmd_fit)

    sp = sub.add_parser("info", parents=[common, filt, gate], help="1 つのモデルの詳細")
    sp.add_argument("model", help="モデルの ID または名前の一部")
    sp.set_defaults(func=cmd_info)

    sp = sub.add_parser("update", parents=[common], help="一覧を上流から作り直す")
    sp.add_argument("--offline-sources", metavar="DIR", help="(試験用)上流の代わりに使うファイル")
    sp.add_argument("--skip-hf", action="store_true", help="Hugging Face を見ない")
    sp.add_argument("--skip-head", action="store_true", help="URL の HEAD 確認を飛ばす")
    sp.add_argument(
        "--heads-only",
        action="store_true",
        help="今の一覧はそのままで、URL の HEAD と ComfyUI の最新の版の確認だけをやり直す",
    )
    sp.add_argument("--output", metavar="PATH", help="書き出し先(既定はキャッシュ)")
    sp.add_argument(
        "--from-release",
        action="store_true",
        help="配布された署名つきの一覧を取り込む(署名が合わなければ何も変えない)。"
        "--output があればそこへ書く",
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help="(--from-release)前に取り込んだ一覧より古くても取り込む",
    )
    sp.add_argument(
        "--allow-partial",
        action="store_true",
        help="一部(テンプレート・Hugging Face)を取得できなくても書き出す(sources.partial が true になる)",
    )
    sp.add_argument(
        "--rate-limit-budget",
        type=float,
        metavar="SEC",
        help="Hugging Face の上限(429)で待つ時間の合計の上限(秒。既定 900)。超えたら止める",
    )
    sp.set_defaults(func=cmd_update)

    sp = sub.add_parser("gate", parents=[common, gate], help="自動でおすすめに出せるかの集計")
    sp.add_argument("--modality", "-m", choices=MODALITIES)
    only = sp.add_mutually_exclusive_group()
    only.add_argument("--only-auto", action="store_true", help="自動で出せるものの ID も並べる")
    only.add_argument("--only-held", action="store_true", help="保留のものの ID と理由も並べる")
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser("bench", parents=[common], help="動いている ComfyUI で実測する")
    sp.add_argument("template", help="テンプレート名(モデルの ID)")
    sp.add_argument(
        "--url", required=True, help="測りに使う ComfyUI の URL(例 http://127.0.0.1:8188。既定は持たない)"
    )
    sp.add_argument("--workflow", metavar="PATH", help="API 形式のワークフロー(省略時はテンプレートから作る)")
    sp.add_argument("--width", type=_positive_int)
    sp.add_argument("--height", type=_positive_int)
    sp.add_argument("--frames", type=_positive_int)
    sp.add_argument("--length", type=_positive_float, metavar="SEC")
    sp.add_argument("--steps", type=_positive_int)
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


def _quiet_stdout() -> None:
    with contextlib.suppress(OSError):  # 終了時の flush で再び落ちないように
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows の cp932 でも落ちず、組み込み側が UTF-8 で読めるように
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(AttributeError, ValueError, OSError):
                stream.reconfigure(encoding="utf-8")
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    if not argv:  # 引数なしは使い方の全文を出して正常に終わる(初めて試す人向け)
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:  # | head などで出力を途中で閉じられた
        _quiet_stdout()
        return 0
    except OSError as e:
        if getattr(e, "errno", None) == 22 and not sys.stdout.isatty():  # Windows でパイプを閉じられたとき
            _quiet_stdout()
            return 0
        raise
    except (CliError, DatabaseError) as e:
        cands = getattr(e, "candidates", None)
        if getattr(args, "json", False):
            err: dict[str, Any] = {"genmodelfit": __version__, "command": args.command, "error": str(e)}
            if cands is not None:
                err["candidates"] = [{"id": m.id, "name": m.name, "content": m.content} for m in cands]
            _out(err)
        else:
            print(f"error: {e}", file=sys.stderr)
            for m in (cands or [])[:30]:
                print(f"  {m.id}  ({m.name})", file=sys.stderr)
            if cands and len(cands) > 30:
                print(f"  …ほか {len(cands) - 30} 件(--json で全部)", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
