# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""コマンドライン。`--json` を付けると、どのコマンドも決まった形の JSON を出す(docs/schema.md)。

終了コード: 0 成功 / 1 失敗 / 2 使い方の誤り。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import threading
from pathlib import Path
from typing import Any

from . import i18n
from ._version import __version__
from .db import Database, DatabaseError
from .db import load as load_db
from .estimate import GIB, estimate, load_coefficients
from .hardware import detect
from .i18n import LANGUAGES, pick, tr
from .models import FIT_LEVELS, MODALITIES, Settings, SystemInfo
from .recommend import CONTENT_FILTERS, DEFAULT_CONTENT, FIT_FILTERS, SORTS, select


class CliError(RuntimeError):
    pass


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            tr(f"整数を指定してください: {text}", f"must be an integer: {text}")
        ) from None
    if v <= 0:
        raise argparse.ArgumentTypeError(
            tr(f"1 以上を指定してください: {text}", f"must be 1 or more: {text}")
        )
    return v


def _positive_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            tr(f"数を指定してください: {text}", f"must be a number: {text}")
        ) from None
    if not v > 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(
            tr(f"0 より大きい値を指定してください: {text}", f"must be greater than 0: {text}")
        )
    return v


def _non_negative_float(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(
            tr(f"数を指定してください: {text}", f"must be a number: {text}")
        ) from None
    if not v >= 0 or v == float("inf"):
        raise argparse.ArgumentTypeError(
            tr(f"0 以上を指定してください: {text}", f"must be 0 or more: {text}")
        )
    return v


def _out(obj: Any) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _system(args: argparse.Namespace) -> SystemInfo:
    return detect(
        disk_path=getattr(args, "disk_path", None),
        vram_gb=args.vram,
        ram_gb=args.ram,
        other_gpus=True if getattr(args, "all_gpus", False) else None,
    )


def _wrapped(text: str, first: str = "  - ", rest: str = "    ", width: int = 100) -> list[str]:
    """全角を 2 桁で数えて width 桁に折り返す(system の注意の文は長い日本語になる)。
    英数字の並びの途中では切らない(table.wrap_text。#76)。"""
    from .table import wrap_text

    return wrap_text(text, width, first, rest)


def _swap_line(s: SystemInfo) -> str | None:
    """仮想メモリの 1 行(Windows はページファイルとコミット、Linux はスワップ)。分からなければ None。"""
    if s.swap_auto is None and s.swap_total_gb is None and s.commit_limit_gb is None:
        return None
    windows = s.commit_limit_gb is not None or s.os.startswith("Windows")
    label = tr("ページファイル", "Pagefile") if windows else tr("スワップ", "Swap")
    if s.swap_max_gb == 0 and s.swap_auto is False:
        text = tr("なし", "none")
    elif s.swap_auto:
        now = (
            tr(f"今 {s.swap_total_gb:.1f} GB、", f"{s.swap_total_gb:.1f} GB now, ")
            if s.swap_total_gb is not None
            else ""
        )
        text = now + tr("OS が管理(増える)", "system managed (grows)")
    elif s.swap_max_gb is not None:
        text = tr(f"最大 {s.swap_max_gb:.1f} GB", f"up to {s.swap_max_gb:.1f} GB")
    elif s.swap_total_gb is not None:
        text = tr(f"今 {s.swap_total_gb:.1f} GB", f"{s.swap_total_gb:.1f} GB now")
    else:
        text = tr("不明", "unknown")
    if s.commit_limit_gb is not None and s.commit_used_gb is not None:
        text += tr(
            f"。コミット {s.commit_used_gb:.1f} / {s.commit_limit_gb:.1f} GB 使用中",
            f"; commit {s.commit_used_gb:.1f} / {s.commit_limit_gb:.1f} GB in use",
        )
    return f"{label}: {text}"


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
        return tr(
            "この PC では、解像度や長さを下げても good にならない",
            "not good on this PC even at a lower resolution or length",
        )
    ja = i18n.get_language() == "ja"
    parts = []
    for key, label, unit, label_en, unit_en in (
        ("max_side_px", "長辺 ", "px", "long side ", " px"),
        ("max_frames", "", " コマ", "", " frames"),
        ("max_seconds", "", " 秒", "", " s"),
    ):
        if key not in lim:
            continue
        v = lim[key]
        if not ja:
            name = (label_en or unit_en).strip()
            if v is None:
                parts.append(f"{name}: not good even when lowered")
            else:
                parts.append(f"{label_en}{v}{unit_en}{' (the highest checked)' if key in caps else ''}")
        elif v is None:
            parts.append(f"{label or unit.strip() + ' '}は下げても good にならない")
        else:
            parts.append(f"{label}{v}{unit}{'(調べた上限)' if key in caps else ''}")
    if not ja:
        return "good on this PC up to " + ", ".join(parts) + " (other settings unchanged)"
    return "この PC なら " + "・".join(parts) + "まで good(ほかの設定はそのまま)"


# --- コマンド ---------------------------------------------------------------------------------


def _data_status(args: argparse.Namespace, db: Database, source: str | None = None):
    from .status import compute

    return compute(db, _policy(args), source=source)


def format_data_status(st) -> list[str]:
    """`system` と `update` の Data の欄(人が読む形)。"""
    if i18n.get_language() == "ja":
        return _format_data_status_ja(st)
    lines = [f"List: {st.source}, generated {st.generated_at or '?'}, {st.models} entries"]
    if st.max_head_age_days is None:
        lines.append(f"HEAD checks: {st.heads_checked_oldest or 'none'} (no age limit)")
    elif st.expired:
        lines.append(
            f"HEAD checks: EXPIRED ({st.heads_checked_oldest or 'none'}, limit {st.max_head_age_days} days)"
            " -> run `genmodelfit update --from-release` (or `update --heads-only`)"
        )
    else:
        lines.append(
            f"HEAD checks: {st.heads_checked_oldest or 'none'}, valid until {st.expires_at or '-'}"
            f" ({st.max_head_age_days} days)"
        )
    lines.append(f"Auto-recommendable: {st.auto_count} of {st.models} (if fresh: {st.auto_count_if_fresh})")
    return lines


# 一覧の出どころ(status.source)の日本語(#76)
_SOURCE_JA = {"bundled": "同梱", "cache": "update で取った一覧", "path": "指定のファイル", "release": "配布"}


def _format_data_status_ja(st) -> list[str]:
    """format_data_status の日本語(#76)。項目は英語と同じ。"""
    src = _SOURCE_JA.get(st.source, st.source)
    lines = [f"一覧: {src}、生成 {st.generated_at or '?'}、{st.models} 件"]
    oldest = st.heads_checked_oldest or tr("なし", "none")
    if st.max_head_age_days is None:
        lines.append(f"HEAD の確認: {oldest}(期限なし)")
    elif st.expired:
        lines.append(
            f"HEAD の確認: 期限切れ({oldest}、期限 {st.max_head_age_days} 日)"
            " → `genmodelfit update --from-release`(か `update --heads-only`)を実行してください"
        )
    else:
        lines.append(f"HEAD の確認: {oldest}、{st.expires_at or '-'} まで有効({st.max_head_age_days} 日)")
    lines.append(
        f"自動のおすすめ: {st.models} 件中 {st.auto_count} 件(確認が新しければ {st.auto_count_if_fresh} 件)"
    )
    return lines


def _print_status_after_update(st) -> None:
    for line in format_data_status(st)[1:]:
        print("  " + line)


def cmd_system(args: argparse.Namespace) -> int:
    s = _system(args)
    db = _db(args)
    st = _data_status(args, db)
    if args.json:
        env = _envelope("system", s, db)
        env["database"]["status"] = st.to_dict()
        _out(env)
        return 0
    print(tr("=== この PC ===", "=== System ==="))
    print(tr(f"OS:       {s.os}", f"OS:   {s.os}"))
    print(tr(f"CPU:      {s.cpu}({s.cpu_cores} スレッド)", f"CPU:  {s.cpu} ({s.cpu_cores} threads)"))
    print(
        tr(
            f"メモリ:   {s.ram_total_gb:.1f} GB(空き {s.ram_available_gb:.1f} GB)",
            f"RAM:  {s.ram_total_gb:.1f} GB (available {s.ram_available_gb:.1f} GB)",
        )
    )
    swap = _swap_line(s)
    if swap:
        print(swap)
    if s.disk_free_gb is not None:
        print(tr(f"ディスク: 空き {s.disk_free_gb:.1f} GB", f"Disk: {s.disk_free_gb:.1f} GB free"))
    if not s.gpus:
        print(tr("GPU:      見つからない(CPU だけ)", "GPU:  none detected (CPU only)"))
    elif s.best_gpu is None:
        print(
            tr(
                "GPU:      使える GPU が無い(CPU だけとして判定。注意を見てください)",
                "GPU:  no usable GPU (judged as CPU only; see notes)",
            )
        )
    for i, g in enumerate(s.gpus, 1):
        mem = []
        if g.vram_free_gb is not None:
            mem.append(tr(f"空き {g.vram_free_gb:.1f} GB", f"{g.vram_free_gb:.1f} GB free"))
        if g.vram_used_gb is not None:
            mem.append(
                tr(
                    f"ほかのアプリが {g.vram_used_gb:.1f} GB 使用中",
                    f"{g.vram_used_gb:.1f} GB used by other apps",
                )
            )
        extra = "".join(tr(f"、{m}", f", {m}") for m in mem)
        print(
            tr(
                f"GPU {i}: {g.name}(VRAM {g.vram_total_gb:.1f} GB{extra})",
                f"GPU {i}: {g.name} ({g.vram_total_gb:.1f} GB VRAM{extra})",
            )
        )
        cc = (
            f"compute capability {g.compute_capability}"
            if g.compute_capability
            else tr("compute capability 不明", "compute capability unknown")
        )
        if g.compute_capability_source == "name":
            cc += tr("(名前から推定)", " (guessed from the name)")
        arch = tr(f"({g.arch})", f" ({g.arch})") if g.arch else ""
        if g is s.best_gpu:
            use = tr("判定に使う", "used for judging")
        elif g.backend == "none":
            use = tr("使わない", "not used")
        else:
            use = tr("一番大きい GPU ではない", "not the largest")
        if g.vendor != "nvidia":  # AMD・Intel・Apple(0.7.0、#22。対応表で決めた推定)
            gfx = tr(f"{g.gfx_target}、", f"{g.gfx_target}, ") if g.gfx_target else ""
            kind = {
                "shared": tr("内蔵・メモリと共有、", "integrated, shared with RAM, "),
                "unified": tr("ユニファイドメモリ、", "unified memory, "),
            }.get(g.memory, "")
            print(
                tr(
                    f"       {gfx}{kind}{g.backend}(推定。実測なし): {use}",
                    f"       {gfx}{kind}{g.backend} (estimated, no measurements): {use}",
                )
            )
            continue
        print(
            tr(
                f"       {cc}{arch}、ドライバ {g.driver or '不明'}、{g.backend}: {use}",
                f"       {cc}{arch}, driver {g.driver or 'unknown'}, {g.backend}: {use}",
            )
        )
        if g.torch_cuda:
            print(
                tr(
                    f"       この PC で動く PyTorch の CUDA 版: {', '.join(g.torch_cuda)}",
                    f"       PyTorch CUDA builds that run here: {', '.join(g.torch_cuda)}",
                )
            )
        if g.compute_dtype:
            if i18n.get_language() == "ja":
                yes = {True: "対応", False: "非対応", None: "不明"}
                print(f"       ComfyUI の精度: {g.compute_dtype}(bf16 {yes[g.bf16]}、fp8 {yes[g.fp8]})")
            else:
                yes = {True: "yes", False: "no", None: "unknown"}
                print(f"       ComfyUI precision: {g.compute_dtype} (bf16 {yes[g.bf16]}, fp8 {yes[g.fp8]})")
    if s.other_gpus_checked:
        print(
            tr(
                "画面の部品(AMD / Intel / Apple は、使える NVIDIA の GPU が無いときだけ使う):",
                "Display adapters (AMD / Intel / Apple are used only when no NVIDIA GPU is usable):",
            )
        )
        if not s.other_gpus:
            print(tr("  (見つからない)", "  (none found)"))
        for a in s.other_gpus:
            vram = f", {a.vram_gb:.1f} GB ({a.vram_source})" if a.vram_gb else ""
            kind = tr("、内蔵", ", integrated") if a.integrated else ""
            state = {
                True: tr(": 使える", ": usable"),
                False: tr(": 使えない → CPU として判定", ": not usable -> judged as CPU"),
            }.get(a.usable, "")
            print("\n".join(_wrapped(f"{a.name} [{a.vendor}{vram}{kind}]{state}")))
    if s.overrides:
        print(tr(f"(上書き: {', '.join(s.overrides)})", f"(overridden: {', '.join(s.overrides)})"))
    if s.notes:
        print(tr("注意:", "Notes:"))
        for n in s.notes:
            print("\n".join(_wrapped(n)))
    print(tr("=== 一覧 ===", "=== Data ==="))
    for line in format_data_status(st):
        print(line)
    return 0


def _license_list(text: str) -> list[str]:
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            tr(
                "ライセンスを 1 つ以上、カンマ区切りで指定してください(例: apache-2.0,mit)",
                "give one or more licenses, comma-separated (e.g. apache-2.0,mit)",
            )
        )
    return items


def _modality_list(text: str) -> list[str]:
    from .tasks import parse_modalities

    try:
        return list(parse_modalities(text) or ())
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _task_list(text: str) -> list[str]:
    from .tasks import task_codes

    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    try:
        task_codes(items)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None
    return items


def _select(args: argparse.Namespace, s: SystemInfo, db: Database, **kw: Any):
    return select(
        db,
        s,
        modality=args.modality,
        content=args.content,
        license=args.license,
        task=args.task,
        include_unknown_license=args.include_unknown_license or getattr(args, "all", False),
        group=not args.all_workflows,
        settings=_settings(args),
        with_limits=args.limits,
        policy=_policy(args),
        **kw,
    )


def _filters(args: argparse.Namespace, min_fit: str | None, limit: int | None) -> dict[str, Any]:
    return {
        "modality": args.modality,
        "content": args.content,
        "min_fit": min_fit,
        "fit": args.fit,
        "license": args.license,
        "include_unknown_license": args.include_unknown_license or getattr(args, "all", False),
        "task": args.task,
        "all_workflows": args.all_workflows,
        "limit": limit,
    }


def _show(
    args, command: str, title: str, s: SystemInfo, db: Database, sel, filters: dict[str, Any], sections=False
) -> int:
    if args.json:
        _out(
            _envelope(
                command,
                s,
                db,
                filters=filters,
                matched=sel.matched,
                hidden=sel.hidden,
                groups=sel.groups,
                models=[f.to_dict() for f in sel.fits],
            )
        )
        return 0
    from .table import print_fit_table

    print_fit_table(
        sel.fits,
        title=title,
        system=s,
        matched=sel.matched,
        hidden=sel.hidden,
        groups=sel.groups if sections else None,
        grouped=not args.all_workflows,
        order=getattr(args, "sort", None) or "auto",
        content=getattr(args, "content", None),
        # --auto-only は表も JSON も auto だけなので、並びの違いを知らせない(#84)
        json_first=command == "recommend" and not getattr(args, "auto_only", False),
    )
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """-m も --task も無ければ種類ごとの節(各 -n 件、既定 3)。あれば 1 つの表(既定 10 件)。"""
    s, db = _system(args), _db(args)
    sections = not args.modality and not args.task
    per_type = (args.limit or 3) if sections else None
    limit = None if sections else (args.limit if args.limit is not None else 10)
    if args.fit is not None and args.min_fit is not None:
        raise CliError(
            tr("--fit と --min-fit は同時に指定できません", "--fit and --min-fit cannot be used together")
        )
    min_fit = None if args.fit is not None else (args.min_fit or "marginal")
    sel = _select(
        args,
        s,
        db,
        min_fit=min_fit,
        fit=args.fit,
        include_tools=args.include_tools,
        auto_only=args.auto_only,
        drop_unknown_size=True,
        limit=limit,
        per_modality=per_type,
        # 表は自動のおすすめ(Verified)を先にしない(#84)。--json は今までの並び(verified が先)のまま
        verified_first=bool(args.json),
    )
    filters = _filters(args, min_fit, limit) | {
        "per_type": per_type,
        "auto_only": args.auto_only,
        "include_tools": args.include_tools,
    }
    return _show(args, "recommend", tr("おすすめ", "Recommended"), s, db, sel, filters, sections=sections)


def _fit_choice(args: argparse.Namespace, default: str | None) -> tuple[str | None, str | None]:
    """(min_fit, fit)。--perfect は --fit perfect。--all は --fit all。どれも無ければ default。"""
    if args.perfect:
        args.fit = "perfect"
    if getattr(args, "all", False) and args.fit is None and args.min_fit is None:
        args.fit = "all"
    if args.fit is not None and args.min_fit is not None:
        raise CliError(
            tr("--fit と --min-fit は同時に指定できません", "--fit and --min-fit cannot be used together")
        )
    if args.fit is None and args.min_fit is None:
        args.fit = default
    return args.min_fit, args.fit


def cmd_fit(args: argparse.Namespace) -> int:
    """既定は動くもの(perfect / good / marginal)だけ、補助ツール・ライセンス不明・無検閲は隠す。--all で全部。
    並びは種類 → 段階 → 点数(--sort)。自動のおすすめを先にはしない(自動のおすすめかは info で分かる)。"""
    s, db = _system(args), _db(args)
    min_fit, fit = _fit_choice(args, "runnable")
    tools = args.include_tools or args.all
    sel = _select(
        args,
        s,
        db,
        min_fit=min_fit,
        fit=fit,
        limit=args.limit,
        include_tools=tools,
        sort=args.sort,
        auto_first=False,
    )
    filters = _filters(args, min_fit, args.limit) | {
        "sort": args.sort,
        "include_tools": tools,
        "all": args.all,
    }
    return _show(args, "fit", tr("全モデル", "All models"), s, db, sel, filters)


def cmd_search(args: argparse.Namespace) -> int:
    """ID・名前の一部で探す(大文字小文字を区別しない)。0 件でも終了コード 0。空の語は 1。"""
    s, db = _system(args), _db(args)
    query = args.query.strip()
    if not query:
        raise CliError(tr("探す語が空です", "the search word is empty"))
    min_fit, fit = _fit_choice(args, None)
    sel = _select(
        args, s, db, query=query, min_fit=min_fit, fit=fit, limit=args.limit, sort=args.sort, auto_first=False
    )
    filters = {"query": query} | _filters(args, min_fit, args.limit) | {"sort": args.sort, "all": args.all}
    return _show(args, "search", tr(f"「{query}」の検索", f'Search "{query}"'), s, db, sel, filters)


def types_summary(db: Database, content: str = DEFAULT_CONTENT) -> list[dict[str, Any]]:
    """大きな種類ごとの、モデルの数・ワークフローの数・用途ごとのワークフローの数。"""
    from .groups import group_key
    from .tasks import MODALITY_LABEL, USES, uses_of

    out = []
    for mod in MODALITIES:
        entries = [m for m in db.models if m.modality == mod and (content == "all" or m.content == content)]
        uses = {u: sum(1 for m in entries if u in uses_of(m)) for u in USES}
        en, ja = MODALITY_LABEL[mod]
        out.append(
            {
                "modality": mod,
                "label": en,
                "label_ja": ja,
                "models": len({group_key(m) for m in entries}),
                "workflows": len(entries),
                "uses": {u: n for u, n in uses.items() if n},
            }
        )
    return out


def cmd_types(args: argparse.Namespace) -> int:
    from .tasks import USE_ALIASES, USES

    db = _db(args)
    rows = types_summary(db, args.content)
    uses = [
        {"use": u, "label": en, "label_ja": ja, "tasks": list(codes)} for u, (en, ja, codes) in USES.items()
    ]
    if args.json:
        _out(
            _envelope(
                "types",
                None,
                db,
                filters={"content": args.content},
                types=rows,
                uses=uses,
                aliases=USE_ALIASES,
            )
        )
        return 0
    lines = [
        tr(
            f"種類(内容 {args.content}。Models = 重みの数、Workflows = 一覧の項目の数)",
            f"Types (content {args.content}; models = distinct weights, workflows = list entries)",
        )
    ]
    lines.append(f"{'Type':<6}{'Models':>7}{'Workflows':>10}  Uses (workflows)")
    for r in rows:
        used = ", ".join(f"{u} {n}" for u, n in r["uses"].items())
        lines.append(f"{r['modality']:<6}{r['models']:>7}{r['workflows']:>10}  {used}")
    lines.append("")
    lines.append(
        tr(
            "--task に使う用途(下の名前か、t2i / i2i / inpaint などの用途のコード):",
            "Uses for --task (a name below, or a task code such as t2i / i2i / inpaint):",
        )
    )
    for u in uses:
        label = tr(u["label_ja"], u["label"])  # 1 行に 2 つの言語を並べない(#76)
        lines.append(f"  {u['use']:<8}{label}  [{', '.join(u['tasks'])}]")
    lines.append(
        tr("例: ", "Examples: ") + "genmodelfit recommend -m image --task edit / -m video --task i2v,lipsync"
    )
    _emit(lines)
    return 0


class AmbiguousModel(CliError):
    def __init__(self, query: str, candidates: list) -> None:
        super().__init__(
            tr(
                f"「{query}」に合うモデルが {len(candidates)} 件あります。ID で指定してください",
                f'{len(candidates)} models match "{query}". Specify one by its id',
            )
        )
        self.candidates = candidates


def resolve_model(db: Database, query: str, content: str = DEFAULT_CONTENT):
    """ID がそのまま合えば(内容によらず)そのモデル。合わなければ ID・名前の一部で探す。

    一部で探すときは content で絞る(既定 sfw なので無検閲は出ない)。1 件だけならそれ、
    2 件以上なら AmbiguousModel(先頭を黙って選ばない)、0 件なら CliError。
    """
    query = query.strip()
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
        kinds = "・".join(sorted({m.content for m in hits}))
        kinds_en = ", ".join(sorted({m.content for m in hits}))
        raise CliError(
            tr(
                f"モデルが見つかりません: {query}(--content {content} で外れた {kinds} のモデルが "
                f"{len(hits)} 件あります。--uncensored を付けるか ID で指定してください)",
                f"no such model: {query} ({len(hits)} {kinds_en} model(s) are hidden by --content {content};"
                " add --uncensored or specify the id)",
            )
        )
    raise CliError(tr(f"モデルが見つかりません: {query}", f"no such model: {query}"))


def _emit(lines: list[str]) -> None:
    """info の文を幅(COLUMNS か端末、最大 100 桁)に収めて出す。"""
    from .table import output_width, wrap_line

    width = output_width()
    for line in lines:
        for part in wrap_line(line, width):
            print(part)


def _by_model_name(db: Database, s: SystemInfo, args: argparse.Namespace, query: str):
    """モデルの表示名(大文字小文字を区別しない)に合えば、そのモデルの中でこの PC に一番合うワークフロー。"""
    from .groups import index_for
    from .recommend import judge, variant_key

    if db.get(query.strip()) is not None:
        return None  # ID がそのまま合うなら ID を優先
    idx = index_for(db)
    key = idx.find(query)
    if key is None:
        return None
    c = load_coefficients()
    fits = [
        judge(m, s, db, settings=_settings(args), coefficients=c, policy=_policy(args))
        for m in idx.members[key]
    ]
    return min(fits, key=variant_key).model


def cmd_info(args: argparse.Namespace) -> int:
    s, db = _system(args), _db(args)
    m = _by_model_name(db, s, args, args.model) or resolve_model(db, args.model, args.content)
    c = load_coefficients()
    e = estimate(m, s, _settings(args), db.measurements, c, with_limits=True)
    gate = _gate_for(m, args, db)
    key, same = same_model(db, s, m, args)
    from .groups import index_for, is_community

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
                group_key=key,
                model_name=index_for(db).name_of.get(key),
                community=is_community(m),
                same_model=same,
            )
        )
        return 0
    from .groups import index_for, is_community

    if i18n.get_language() == "ja":
        _emit(_info_lines_ja(m, e, gate, same, index_for(db).name_of.get(key, m.name), is_community(m)))
        return 0
    n = len(same) + 1
    out = [f"{m.name}  [{m.id}]"]
    out.append(
        f"  model:    {index_for(db).name_of.get(key, m.name)}"
        f"  ({n} workflow{'s' if n > 1 else ''} use{'' if n > 1 else 's'} these weights)"
    )
    if is_community(m):
        out.append(f"  community template by {m.template.username} (not made by ComfyUI)")
    out.append(f"  modality: {m.modality}  tasks: {', '.join(m.tasks) or '?'}  content: {m.content}")
    out.append(f"  license:  {m.license or '? (unknown)'}  source: {m.source}")
    if m.runtime:
        out.append(f"  runtime:  {m.runtime}")
    if m.languages:
        out.append(
            "  languages: " + "; ".join(f"{k} {', '.join(v) or '(none)'}" for k, v in m.languages.items())
        )
    about = pick(m.description) if m.description else None
    if about:
        out.append("  about:    " + about)
    prec_from = f" (from {m.precision_source})" if m.precision and m.precision_source else ""
    out.append(f"  precision: {m.precision or '? (unknown)'}{prec_from}")
    counted = f"{e.disk_gb:.2f} GB counted" if e.disk_gb else "? (size unknown)"
    out.append(f"  size:     {counted}, {len(m.files)} file(s) listed (GB = GiB)")
    for f in m.files:
        fs = f"{f.size_bytes / GIB:.2f} GB" if f.size_bytes and f.size_bytes > 0 else "?"
        out.append(f"    - {f.name} ({fs}, {f.directory or '-'}, license {f.license or '?'})")
    basis = e.basis if e.basis_detail == e.basis else f"{e.basis}: {e.basis_detail}"
    if e.sample_count is not None and e.basis_detail != "measured":
        basis += f" n={e.sample_count}"  # ほかの利用者の実測の件数(2026-10-22)
    out.append(f"  fit:      {e.fit}  ({basis}{', on ' + e.measured_on if e.measured_on else ''})")
    seq = f" / {e.vram_sequential_gb:.1f} GB swapping parts" if e.vram_sequential_gb is not None else ""
    if e.unknown_size:
        out.append("  VRAM:     ? (unknown: weight sizes are not known)")
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
        out.append(
            f"  verified (auto-recommend): {'yes' if gate.get('auto') else 'held'}  (ComfyUI {target})"
        )
        for chk in gate.get("checks", []):
            mark = "ok" if chk["passed"] else "NG"
            out.append(f"    [{mark}] {chk['id']}: {chk.get('detail', '')}")
    if same:
        k = len(same)
        out.append(f"  same model ({k} other workflow{'s' if k > 1 else ''}; precision / use differ):")
        for w in same:
            need = w["memory_required_gb"]
            # 分からない(大きさ不明)は ?、当てはまらない(GPU を使わない)は -(#30)
            vram = f"{need:.1f} GB" if need is not None else ("?" if w["fit_level"] == "unknown" else "-")
            prec = w["precision"] or "?"
            out.append(f"    - {w['id']}  ({w['use']}, {prec}, {w['fit_level']}, {vram})")
    _emit(out)
    return 0


def _ja_label(name: str, width: int = 12) -> str:
    """info・download の日本語の見出し(「  モデル:     」)。値の頭を全角の幅でそろえる(#76)。"""
    from rich.cells import cell_len

    text = name + ":"
    return "  " + text + " " * max(1, width - cell_len(text))


def _info_lines_ja(m, e, gate, same: list[dict[str, Any]], model_name: str, community: bool) -> list[str]:
    """info の日本語(#76)。英語と同じ項目を同じ順に。機械の値(ID・段階・用途・精度・コード)は訳さない。"""
    L = _ja_label
    out = [f"{m.name}  [{m.id}]"]
    out.append(f"{L('モデル')}{model_name}(この重みを使うワークフロー {len(same) + 1} 件)")
    if community:
        out.append(f"  利用者のテンプレート(作者 {m.template.username}。ComfyUI の公式ではない)")
    out.append(f"{L('種類')}{m.modality}  用途: {', '.join(m.tasks) or '?'}  内容: {m.content}")
    out.append(f"{L('ライセンス')}{m.license or '?(不明)'}  出どころ: {m.source}")
    if m.runtime:
        out.append(f"{L('動かし方')}{m.runtime}")
    if m.languages:
        langs = "; ".join(
            f"{_diff_label(_DIFF_LANG_LABELS, k)} {', '.join(v) or '(なし)'}" for k, v in m.languages.items()
        )
        out.append(f"{L('言語')}{langs}")
    about = pick(m.description) if m.description else None
    if about:
        out.append(f"{L('説明')}{about}")
    prec_from = f"(出どころ: {m.precision_source})" if m.precision and m.precision_source else ""
    out.append(f"{L('精度')}{m.precision or '?(不明)'}{prec_from}")
    counted = f"{e.disk_gb:.2f} GB(数えた分)" if e.disk_gb else "?(大きさ不明)"
    out.append(f"{L('大きさ')}{counted}、一覧のファイル {len(m.files)} 件(GB = GiB)")
    for f in m.files:
        fs = f"{f.size_bytes / GIB:.2f} GB" if f.size_bytes and f.size_bytes > 0 else "?"
        out.append(f"    - {f.name}({fs}、{f.directory or '-'}、ライセンス {f.license or '?'})")
    basis = e.basis if e.basis_detail == e.basis else f"{e.basis}: {e.basis_detail}"
    if e.sample_count is not None and e.basis_detail != "measured":
        basis += f" n={e.sample_count}"
    where = f"、{e.measured_on} で" if e.measured_on else ""
    out.append(f"{L('段階')}{e.fit}({basis}{where})")
    if e.unknown_size:
        out.append(f"{L('VRAM')}?(重みの大きさが分からない)")
    else:
        seq = f" / 部品を載せ替えて {e.vram_sequential_gb:.1f} GB" if e.vram_sequential_gb is not None else ""
        out.append(
            f"{L('VRAM')}全部 GPU に載せて {e.vram_gb:.1f} GB{seq} / 逃がして {e.vram_offload_gb:.1f} GB、"
            f"メモリ {e.ram_gb:.1f} GB"
        )
    if e.limits:
        out.append(f"{L('限界')}{format_limits(e.limits)}")
    out += [f"  - {r}" for r in e.reasons]
    if gate:
        target = gate.get("comfyui_version") or "?"
        out.append(f"  自動のおすすめ(Verified): {'yes' if gate.get('auto') else '保留'}(ComfyUI {target})")
        for chk in gate.get("checks", []):
            mark = "ok" if chk["passed"] else "NG"
            out.append(f"    [{mark}] {chk['id']}: {chk.get('detail', '')}")
    if same:
        out.append(f"  同じモデル(ほかのワークフロー {len(same)} 件。精度・使い方が違う):")
        for w in same:
            need = w["memory_required_gb"]
            vram = f"{need:.1f} GB" if need is not None else ("?" if w["fit_level"] == "unknown" else "-")
            out.append(f"    - {w['id']}  ({w['use']}, {w['precision'] or '?'}, {w['fit_level']}, {vram})")
    return out


def same_model(db: Database, s: SystemInfo, m, args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    """同じモデル(groups.group_key)のほかのワークフロー。この PC での判定つき、良い順。"""
    from .groups import index_for
    from .recommend import _member_dict, judge, variant_key

    idx = index_for(db)
    key = idx.key_of[m.id]
    c = load_coefficients()
    others = [
        judge(o, s, db, settings=_settings(args), coefficients=c, policy=_policy(args))
        for o in idx.members[key]
        if o.id != m.id
    ]
    others.sort(key=variant_key)
    return key, [_member_dict(o, True) for o in others]


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
    from .update import (
        UpdateFailed,
        inspect_database,
        make_fetcher,
        refresh_app_fields,
        refresh_heads,
        update_database,
    )

    out_path = Path(args.output) if getattr(args, "output", None) else cached_db_path()
    say = _progress(args)
    fetch = make_fetcher(getattr(args, "offline_sources", None), getattr(args, "rate_limit_budget", None))
    try:
        if getattr(args, "inspect_only", False):
            db = inspect_database(
                fetch,
                out_path,
                db_path=getattr(args, "db", None),
                use_cache=not getattr(args, "no_cache", False),
                offline=bool(getattr(args, "offline", False)),
                progress=say,
            )
        elif getattr(args, "fields_only", False):
            db = refresh_app_fields(
                fetch,
                out_path,
                db_path=getattr(args, "db", None),
                use_cache=not getattr(args, "no_cache", False),
                offline=bool(getattr(args, "offline", False)),
                progress=say,
            )
        elif getattr(args, "heads_only", False):
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
            print(tr(f"エラー: {e}", f"error: {e}"), file=sys.stderr)
        return 1
    st = _data_status(args, db, source="cache" if out_path == cached_db_path() else "path")
    if args.json:
        db.origin = "path"
        _out(_envelope("update", None, db, output=str(out_path), sources=db.sources, status=st.to_dict()))
        return 0
    _print_status_after_update(st)
    if getattr(args, "inspect_only", False):
        fh = db.sources["file_headers"]
        print(
            tr(
                f"ファイルの見出しの結果を書き出しました: {out_path}",
                f"Wrote the file header results: {out_path}",
            )
        )
        print(
            tr(
                f"  本体のファイル {fh['urls']} 件: キャッシュ {fh['from_cache']}、読んだ {fh['fetched']}、"
                f"読めない {fh['failed']}、未確認(オフライン) {fh['skipped_offline']}",
                f"  {fh['urls']} weight files: cached {fh['from_cache']}, read {fh['fetched']},"
                f" unreadable {fh['failed']}, unchecked (offline) {fh['skipped_offline']}",
            )
        )
        rp = db.sources.get("repo_precision") or {}
        if rp:
            print(
                tr(
                    f"  リポジトリから読んだ精度: Hub API {rp.get('hf_api', 0)}、"
                    f"config.json {rp.get('config', 0)}、分からない {rp.get('unknown', 0)}"
                    f"(調べた本体 {rp.get('files', 0)} 件)",
                    f"  Precision read from repositories: Hub API {rp.get('hf_api', 0)},"
                    f" config.json {rp.get('config', 0)}, unknown {rp.get('unknown', 0)}"
                    f" ({rp.get('files', 0)} weight files checked)",
                )
            )
        unknown = rp.get("precision_unknown", fh["precision_unknown"])
        print(tr(f"  精度が分からない項目: {unknown} 件", f"  Entries with unknown precision: {unknown}"))
        hf = db.sources.get("huggingface") or {}
        missing = set(hf.get("unauthorized_missing") or [])
        unauth = [r for r in hf.get("unauthorized") or [] if r not in missing]
        if unauth:
            print(
                tr(
                    f"  匿名で読めない(401。非公開か無い)リポジトリ: {', '.join(unauth)}",
                    f"  Repositories not readable anonymously (401: private or missing): {', '.join(unauth)}",
                )
            )
        if missing:
            print(
                tr(
                    f"  401 のうち無いと確かめたもの: {', '.join(sorted(missing))}",
                    f"  Of those 401s, confirmed missing: {', '.join(sorted(missing))}",
                )
            )
        return 0
    if getattr(args, "fields_only", False):
        af = db.sources["app_fields"]
        print(tr(f"アプリ向けの項目を書き出しました: {out_path}", f"Wrote the app fields: {out_path}"))
        print(
            tr(
                f"  手のモデル {db.sources['manual']['models']} 件 / 言語(指示) {af['languages_prompt']} 件・"
                f"言語(読み上げ) {af['languages_speech']} 件 / 動かし方 {af['runtime']} 件 / "
                f"説明 {af['description']} 件",
                f"  hand-added models {db.sources['manual']['models']} / languages (prompt)"
                f" {af['languages_prompt']}, languages (speech) {af['languages_speech']} / runtime"
                f" {af['runtime']} / description {af['description']}",
            )
        )
        return 0
    if getattr(args, "heads_only", False):
        stats = db.sources["head_check"]
        print(tr(f"HEAD の結果を書き出しました: {out_path}", f"Wrote the HEAD results: {out_path}"))
        status = ", ".join(f"{k} {v}" for k, v in stats["status"].items())
        print(tr(f"  URL {stats['urls']} 件: ", f"  {stats['urls']} URLs: ") + status)
        latest = db.sources["comfyui"]["latest_release"]
        print(
            tr(
                f"  未確認 {stats['unchecked']} 件 / ComfyUI の最新の版: {latest}",
                f"  unchecked {stats['unchecked']} / latest ComfyUI release: {latest}",
            )
        )
        return 0
    counts = db.sources["counts"]
    mods = ", ".join(f"{k} {v}" for k, v in counts["modality"].items())

    def joined(key: str) -> str:
        return ", ".join(f"{k} {v}" for k, v in counts[key].items())

    print(
        tr(
            f"{counts['models']} 件を書き出しました: {out_path}",
            f"Wrote {counts['models']} entries: {out_path}",
        )
    )
    print(tr(f"  種類: {mods}", f"  types: {mods}"))
    print(tr("  内容: ", "  content: ") + joined("content"))
    print(tr("  出どころ: ", "  source: ") + joined("source"))
    print(tr("  ライセンス: ", "  license: ") + joined("license_tier"))
    print(
        tr(
            f"  一式がそろっていない(files_complete=false): {counts['files_complete_false']} 件",
            f"  incomplete file sets (files_complete=false): {counts['files_complete_false']}",
        )
    )
    print(
        tr(
            f"  除外: {db.sources['excluded']} 件 / 警告: {len(db.sources['warnings'])} 件",
            f"  excluded: {db.sources['excluded']} / warnings: {len(db.sources['warnings'])}",
        )
    )
    if db.sources.get("partial"):
        pc = json.dumps(db.sources["partial_counts"], ensure_ascii=False)
        print(
            tr(
                f"  注意: 一部を取得できないまま書き出しました(--allow-partial): {pc}",
                f"  warning: written with some sources missing (--allow-partial): {pc}",
            )
        )
    return 0


def _update_from_release(args: argparse.Namespace) -> int:
    from .db import cached_db_path
    from .remote import update_from_release

    dest = Path(args.output) if getattr(args, "output", None) else cached_db_path()
    db = update_from_release(dest=dest, force=bool(getattr(args, "force", False)))
    st = _data_status(args, db, source="release")
    if args.json:
        out = {
            "genmodelfit": __version__,
            "command": "update",
            "models": len(db.models),
            "generated_at": db.generated_at,
            "output": str(dest),
            "status": st.to_dict(),
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        print(
            tr(
                f"一覧を更新しました: {len(db.models)} 件(生成日時 {db.generated_at}) -> {dest}",
                f"Updated the model list: {len(db.models)} entries (generated {db.generated_at}) -> {dest}",
            )
        )
        _print_status_after_update(st)
    return 0


# diff の表示の言葉(#59)。言語は呼んだときに選ぶ(#73)
_DIFF_FIELD_LABELS = {
    "name": ("名前", "name"),
    "modality": ("種類", "type"),
    "tasks": ("用途", "tasks"),
    "runtime": ("動かし方", "runtime"),
    "comfy_family": ("ComfyUI の系統", "ComfyUI family"),
    "content": ("内容", "content"),
    "precision": ("精度", "precision"),
}
_DIFF_LANG_LABELS = {"prompt": ("指示", "prompt"), "speech": ("読み上げ", "speech")}


def _diff_label(table: dict[str, tuple[str, str]], key: str) -> str:
    pair = table.get(key)
    return tr(*pair) if pair else key


# 変わったモデルの並びの大事さ(小さいほど先)。--limit で切るときも、大事な変化を先に出す
_DIFF_RANK = {
    "file_added": 0,
    "file_removed": 0,
    "files_order": 0,
    "file_changed": 0,
    "license": 1,
    "file_license": 1,
    # ライセンスの出どころは、自動のおすすめの変化の後・言語や大きさの前(オーナーの決定、#59)
    "license_source": 4.5,
    "auto": 4,
    "languages": 5,
    "size": 5,
    "review": 6,
}
_DIFF_VALUE_RANK = {"content": 2, "precision": 5}  # ほかの値(名前・種類・用途・動かし方・系統)は 3
_DIFF_OTHER_RANK = 6


def _diff_other_head() -> str:
    return tr("その他の変化: ", "other changes: ")


# この大事さ以下(数が大きい)の同じ行が、この件数以上のモデルにあれば 1 行にまとめる。部品・ライセンス・
# 内容(無検閲)はまとめない(モデルごとに確かめるもの)
_DIFF_COLLAPSE_RANK = 3
_DIFF_COLLAPSE_MIN = 3
# まとめた行の ID は、この件数までならその場に出す(多ければ --all)
_DIFF_COLLAPSE_SHOW_IDS = 8
# まとめない種類(大事さが低くても、モデルごとに出す)
_DIFF_NEVER_COLLAPSE = frozenset({"license_source"})
# まとめても ID を必ず出す種類(分類で判断が要る変化: 自動のおすすめ)。数は -n で切る
_DIFF_COLLAPSE_KEEP_IDS = frozenset({"auto"})
DIFF_DEFAULT_LIMIT = 20


def _diff_sections() -> tuple[tuple[str, tuple[float, float]], ...]:
    """変わったモデルの節: (見出し, 大事さの範囲)"""
    return (
        (
            tr(
                "変わったモデル(部品・ライセンス・内容・動かし方など。大事な変化の順):",
                "Changed models (files, license, content, runtime, ...; most important first):",
            ),
            (0, 4),
        ),
        (tr("自動のおすすめの変化:", "Auto-recommendation changes:"), (4, 4.5)),
        (
            tr(
                "そのほかの変化(ライセンスの出どころ・言語・大きさ・確認の印など):",
                "Other changes (license provenance, languages, size, review marks, ...):",
            ),
            (4.5, 99),
        ),
    )


_DIFF_LIST_MAX = 20  # 既定の表示で並べる数の上限(言語など。--all では全部)


def _diff_val(v: Any, full: bool = True) -> str:
    if v is None or v == []:
        return tr("なし", "none")
    if isinstance(v, list):
        if not full and len(v) > _DIFF_LIST_MAX:
            head = ", ".join(str(x) for x in v[: _DIFF_LIST_MAX - 4])
            rest = len(v) - _DIFF_LIST_MAX + 4
            return tr(f"{head} …ほか {rest}(全 {len(v)})", f"{head} … {rest} more ({len(v)} in all)")
        return ", ".join(str(x) for x in v)
    return str(v)


def _diff_auto(v: Any) -> str:
    return tr("出す", "yes") if v else (tr("なし", "none") if v is None else tr("保留", "held"))


def _diff_text(field: str, v: Any) -> str:
    if v is None:
        return tr("なし", "none")
    if isinstance(v, list):
        return " > ".join(str(x) for x in v)
    text = str(v)
    return text.split("://", 1)[-1] if field.endswith("url") else text


def _diff_pair(field: str, a: Any, b: Any, room: int = 72) -> tuple[str, str]:
    """前と後の値を、違うところが見えるように縮める(前と後が同じ文字列にならない)。

    sha256 は違う桁まで伸ばす。長い URL などは、同じ頭と尻を「…」にして、違うところとその前後を残す。"""
    ta, tb = _diff_text(field, a), _diff_text(field, b)
    if ta == tb:
        return ta, tb
    p = 0
    while p < min(len(ta), len(tb)) and ta[p] == tb[p]:
        p += 1
    if field == "sha256" and a is not None and b is not None:
        n = max(12, p + 4)
        return ta[:n] + ("…" if n < len(ta) else ""), tb[:n] + ("…" if n < len(tb) else "")
    if a is None or b is None or (len(ta) <= room and len(tb) <= room):
        return ta, tb  # 片方が無いときは、もう片方を全部出す
    s = 0
    while s < min(len(ta), len(tb)) - p and ta[-1 - s] == tb[-1 - s]:
        s += 1
    ctx = 12
    start = max(0, p - ctx)

    def cut(t: str) -> str:
        end = len(t) - max(0, s - ctx)
        seg, tail = t[start:end], end < len(t)
        if len(seg) > room:
            seg, tail = seg[:room], True  # 違うところ(先頭から ctx 文字目)は残る
        return ("…" if start else "") + seg + ("…" if tail else "")

    return cut(ta), cut(tb)


def _diff_model(m: dict[str, Any]) -> str:
    """モデルの見出し: 名前 (種類) — ID。名前が「(種類)」で終わるときは種類を重ねない。"""
    return f"{_diff_name(m)} — {m['id']}"


def _diff_name(m: dict[str, Any]) -> str:
    name, kind = str(m.get("name") or "?"), m.get("modality")
    if kind and not name.rstrip().lower().endswith(f"({kind})".lower()):
        name = f"{name} ({kind})"
    return name


def _diff_head(m: dict[str, Any], indent: str, width: int, extra: str = "") -> list[str]:
    """モデルの見出しの行。幅に収まらなければ ID を次の行に分ける(ID の途中では折り返さない)。"""
    from rich.cells import cell_len

    line = f"{indent}{_diff_model(m)}{extra}"
    if cell_len(line) <= width:
        return [line]
    out = [f"{indent}{_diff_name(m)} —", f"{indent}    {m['id']}"]
    if extra:
        if cell_len(out[1] + extra) <= width:
            out[1] += extra
        else:
            out.append(f"{indent}    {extra.strip()}")
    return out


def _diff_ids(ids: list[str], indent: str, width: int) -> list[str]:
    """ID の並びを、ID の途中で切らずに幅に詰める(幅より長い ID は 1 行にはみ出す)。"""
    from rich.cells import cell_len

    out: list[str] = []
    cur = ""
    for k, i in enumerate(ids):
        piece = i + ("," if k < len(ids) - 1 else "")
        cand = f"{cur} {piece}" if cur else f"{indent}{piece}"
        if cur and cell_len(cand) > width:
            out.append(cur)
            cand = f"{indent}{piece}"
        cur = cand
    if cur:
        out.append(cur)
    return out


def _diff_side(tag: str, side: dict[str, Any], label: str | None) -> str:
    head = f"{tag}: {label}  " if label else f"{tag}: "
    tail = tr("  (一部欠け: sources.partial)", "  (partial: sources.partial)") if side["partial"] else ""
    n = tr(f"{side['models']} 件", f"{side['models']} entries")
    return f"{head}{side['generated_at']}  {n}  ComfyUI {side['comfyui_latest_release']}{tail}"


def _diff_item(
    c: dict[str, Any], it: dict[str, Any], lic: dict[str, Any], unc: dict[str, Any], full: bool = True
) -> tuple[int, str]:
    """変わったモデルの 1 つの変化 → (大事さ, 行)。知らない種類も黙って落とさず、先に出す。"""
    kind = it["kind"]
    rank = _DIFF_RANK.get(kind, 0)
    if kind == "value":
        rank = _DIFF_VALUE_RANK.get(it["field"], 3)
        label = _diff_label(_DIFF_FIELD_LABELS, it["field"])
        line = f"{label} {_diff_val(it['old'], full)} → {_diff_val(it['new'], full)}"
        if it["field"] == "content" and it["new"] == "uncensored":
            ev = (unc.get(c["id"]) or {}).get("evidence") or []
            line += tr("  ! 新しく無検閲", "  ! newly uncensored") + (
                tr(f"({ev[0]})", f" ({ev[0]})") if ev else ""
            )
        return rank, line
    if kind == "license":
        permissive = (lic.get(c["id"]) or {}).get("permissive_side")
        mark = (
            tr("  ! 寛容な側(原文で確かめる)", "  ! more permissive (check the original text)")
            if permissive
            else ""
        )
        return rank, tr(
            f"ライセンス {it['old']} → {it['new']}{mark}", f"license {it['old']} → {it['new']}{mark}"
        )
    if kind == "license_source":
        where = tr(f"部品 {it['name']} の ", f"file {it['name']} ") if it.get("name") else ""
        a, b = _diff_pair(it["field"], it["old"], it["new"])
        return rank, tr("ライセンスの出どころ ", "license provenance ") + f"{where}{it['field']} {a} → {b}"
    if kind == "languages":
        what = _diff_label(_DIFF_LANG_LABELS, it["key"])
        vals = f"{_diff_val(it['old'], full)} → {_diff_val(it['new'], full)}"
        return rank, tr(f"言語({what}) {vals}", f"languages ({what}) {vals}")
    if kind in ("file_added", "file_removed"):
        path = f"{it['directory']}/{it['name']}" if it.get("directory") else str(it["name"])
        sign = "+" if kind == "file_added" else "-"
        size = _gib(it["size_bytes"])
        if i18n.get_language() == "en":
            src = f", {it['source']}" if it.get("source") else ""
            return rank, f"file {sign} {path} ({size} GiB, license {it['license']}{src})"
        src = f"、{it['source']}" if it.get("source") else ""
        return rank, f"部品 {sign} {path}({size} GiB、ライセンス {it['license']}{src})"
    if kind == "files_order":
        return rank, tr("部品の並び・形が変わった", "file order / shape changed")
    if kind == "file_changed":
        a, b = _diff_pair(it["field"], it["old"], it["new"])
        return rank, tr(
            f"部品 {it['name']} の {it['field']} {a} → {b}", f"file {it['name']} {it['field']} {a} → {b}"
        )
    if kind == "file_license":
        return rank, tr(
            f"部品 {it['name']} のライセンス {it['old']} → {it['new']}",
            f"file {it['name']} license {it['old']} → {it['new']}",
        )
    if kind == "size":
        a, b = it["old"], it["new"]
        if a and b and _gib(a) == _gib(b):
            return rank, tr(
                f"大きさ {_gib(b)} GiB(わずかな変化 {b - a:+,} バイト)",
                f"size {_gib(b)} GiB (small change {b - a:+,} bytes)",
            )
        return rank, tr(f"大きさ {_gib(a)} → {_gib(b)} GiB", f"size {_gib(a)} → {_gib(b)} GiB")
    if kind == "review":
        old, new = it["old"], it["new"]
        reasons = ", ".join(it["reasons"])
        why = tr(f"(理由: {reasons})", f" (reasons: {reasons})") if it["reasons"] and new != "ok" else ""
        if old is None and new is not None:
            return rank, tr(f"確認の印が付いた({new}){why}", f"review mark added ({new}){why}")
        if old is not None and new is None:
            return rank, tr(f"確認の印が外れた(前は {old})", f"review mark removed (was {old})")
        if old != new:
            return rank, tr(f"確認の印 {old} → {new}{why}", f"review mark {old} → {new}{why}")
        fields = ", ".join(it.get("fields") or []) or "?"
        return rank, tr(
            f"確認の印({new})の中身が変わった: {fields}", f"review mark ({new}) details changed: {fields}"
        )
    if kind == "auto":
        held = ", ".join(it.get("held_by") or [])
        why = tr(f"(保留: {held})", f" (held: {held})") if held else ""
        return rank, tr(
            "自動のおすすめ ", "auto-recommend "
        ) + f"{_diff_auto(it['old'])} → {_diff_auto(it['new'])}{why}"
    return rank, f"{kind}: {json.dumps(it, ensure_ascii=False)}"


def _diff_changed_entries(
    d: dict[str, Any],
    show_all: bool,
    limit: int | None = None,
    width: int = 100,
    ranks: tuple[float, float] | None = None,
) -> list[tuple[tuple[Any, ...], list[str]]]:
    """「変わったモデル」の節の並び: (並べる鍵, 行)。大事な変化のあるモデルが先。

    大事さの低い同じ行(確認の印が付いた など。「その他の変化」は項目ごと)が多くのモデルにあれば 1 行
    (件数)にまとめる。まとめた行も同じ大事さの順に並ぶので、--limit で大事なものが隠れない。
    まとめた行の ID: 自動のおすすめの変化は必ず出す(limit まで)。ほかは少なければその場に、多ければ --all。
    ranks: (下限, 上限) の大事さの行だけを使う(節に分けるとき。None = 全部)。"""
    lic = {c["id"]: c for c in d["license_changes"]}
    unc = {c["id"]: c for c in d["newly_uncensored"]}
    per: list[tuple[dict[str, Any], list[tuple[int, str, str]]]] = []
    for c in d.get("model_changes") or []:
        rows = [(*_diff_item(c, it, lic, unc, show_all), it["kind"]) for it in c["items"]]
        rows += [(_DIFF_OTHER_RANK, _diff_other_head() + f, "other") for f in c["other"]]
        if ranks is not None:
            rows = [r for r in rows if ranks[0] <= r[0] < ranks[1]]
        per.append((c, rows))
    count: dict[tuple[int, str, str], list[str]] = {}
    for c, rows in per:
        for r in dict.fromkeys(rows):
            if r[0] >= _DIFF_COLLAPSE_RANK and r[2] not in _DIFF_NEVER_COLLAPSE:
                count.setdefault(r, []).append(c["id"])
    groups = {r: ids for r, ids in count.items() if len(ids) >= _DIFF_COLLAPSE_MIN}
    out: list[tuple[tuple[Any, ...], list[str]]] = []
    for c, rows in per:
        keep = sorted((r for r in rows if r not in groups), key=lambda r: r[0])
        if not keep:
            continue
        texts = [t for _, t, k in keep if k != "other"]
        others = [t[len(_diff_other_head()) :] for _, t, k in keep if k == "other"]
        if others:
            texts.append(_diff_other_head() + ", ".join(others))
        out.append(((keep[0][0], 0, c["id"]), [*_diff_head(c, "  ", width), *(f"    {t}" for t in texts)]))
    for (rank, text, kind), ids in groups.items():
        lines = [tr(f"  {text}: {len(ids)} 件", f"  {text}: {len(ids)} models")]
        if kind in _DIFF_COLLAPSE_KEEP_IDS:  # 判断が要る変化は ID を隠さない
            shown = ids if limit is None else ids[:limit]
            lines += _diff_ids(shown, "    ", width)
            if len(shown) < len(ids):
                rest = len(ids) - len(shown)
                lines.append(
                    tr(
                        f"    …ほか {rest} 件(-n {len(ids)} で増やす)",
                        f"    … {rest} more (-n {len(ids)} for more)",
                    )
                )
        elif show_all or len(ids) <= _DIFF_COLLAPSE_SHOW_IDS:
            lines += _diff_ids(ids, "    ", width)
        else:
            lines[0] += tr("(ID は --all)", " (ids: --all)")
        out.append(((rank, 1, text), lines))
    out.sort(key=lambda e: e[0])
    return out


def _diff_lines(d: dict[str, Any], limit: int | None, show_all: bool = False, width: int = 100) -> list[str]:
    """`genmodelfit diff` の表示(折り返し前の行)。#59

    既定は利用者に関係のある変化だけ(増えた・消えた・変わったモデルの変化(自動のおすすめを含む)・
    新しい HEAD の失敗)。show_all(--all)は保守用の、前からの HEAD の失敗・401 のリポジトリ・
    sources の数・人気の数などだけの変化と、まとめた行の ID も出す。limit は節ごとの上限(None = 全部)。
    width はモデルの ID を途中で折り返さないための幅(ID は自分の行に分ける)。"""
    s = d["summary"]

    def cut(items: list[Any]) -> list[Any]:
        return items if limit is None else items[:limit]

    def more(items: list[Any]) -> list[str]:
        if limit is None or len(items) <= limit:
            return []
        rest = len(items) - limit
        return [
            tr(
                f"    …ほか {rest} 件(-n {len(items)} か --all で全部。--json でも出ます)",
                f"    … {rest} more (-n {len(items)} or --all for all; also in --json)",
            )
        ]

    changed = d.get("model_changes") or []
    head_new = [h for h in d["head_failures"] if h["new"]]
    lines = [
        _diff_side(tr("前", "old"), d["old"], d.get("old_label")),
        _diff_side(tr("後", "new"), d["new"], d.get("new_label")),
        tr(
            f"増えた {s['added']} / 消えた {s['removed']} / 変わった {len(changed)} / 自動のおすすめ"
            f" {s['auto_old']} → {s['auto_new']}(+{s['auto_gained']} -{s['auto_lost']})"
            f" / HEAD の新しい失敗 {s['head_failures_new']}",
            f"added {s['added']} / removed {s['removed']} / changed {len(changed)} / auto-recommended"
            f" {s['auto_old']} → {s['auto_new']} (+{s['auto_gained']} -{s['auto_lost']})"
            f" / new HEAD failures {s['head_failures_new']}",
        ),
    ]
    lines += [tr(f"注意: {w}", f"warning: {w}") for w in d.get("warnings") or []]
    if show_all:
        lines += [
            tr(
                f"ライセンスの変化 {s['license_changed']}(寛容な側 {s['license_permissive_side']})"
                f" / 大きさ {s['size_changed']} / 精度 {s['precision_changed']}"
                f" / 新しく無検閲 {s['newly_uncensored']}",
                f"license changes {s['license_changed']} (more permissive {s['license_permissive_side']})"
                f" / size {s['size_changed']} / precision {s['precision_changed']}"
                f" / newly uncensored {s['newly_uncensored']}",
            ),
            tr(
                f"HEAD の失敗 {s['head_failures']}(新 {s['head_failures_new']})",
                f"HEAD failures {s['head_failures']} (new {s['head_failures_new']})",
            )
            + f" / 401 +{s['unauthorized_added']} -{s['unauthorized_removed']}",
        ]
    if d.get("top_changes"):
        lines += [
            "",
            tr("一覧の上の段で変わった鍵: ", "Changed top-level keys: ") + ", ".join(d["top_changes"]),
        ]
    auto_new = {c["id"] for c in d["auto_changes"] if c["new"]}
    for key, title in (("added", tr("増えた", "Added")), ("removed", tr("消えた", "Removed"))):
        if d[key]:
            lines += ["", f"{title}:"]
            for m in cut(d[key]):
                auto = (
                    tr("  自動のおすすめに出す", "  auto-recommended")
                    if key == "added" and m["id"] in auto_new
                    else ""
                )
                lines += _diff_head(m, "  ", width, f"  [{m['content']}/{m['license']}]{auto}")
            lines += more(d[key])
    # 変わったモデルは 3 つの節に分け、節ごとに -n で切る(どの節も先頭の大事なものは隠れない)。
    # 並び(オーナーの決定、#59): 部品・ライセンス・内容・動かし方など → 自動のおすすめ →
    # ライセンスの出どころ・言語・大きさ・確認の印など
    entries: list[tuple[tuple[Any, ...], list[str]]] = []
    for title, ranks in _diff_sections():
        part = _diff_changed_entries(d, show_all, limit, width, ranks)
        if part:
            entries += part
            lines += ["", title]
            for _, block in cut(part):
                lines += block
            lines += more(part)
    hf_ = sorted(d["head_failures"], key=lambda h: (not h["new"], h["url"])) if show_all else head_new
    if hf_:
        lines += [
            "",
            tr("HEAD が失敗した URL(! = 新しく):", "URLs whose HEAD check failed (! = new):")
            if show_all
            else tr("HEAD が新しく失敗した URL:", "URLs whose HEAD check newly failed:"),
        ]
        mark = (lambda h: "! " if h["new"] else "  ") if show_all else (lambda h: "")
        lines += [f"  {mark(h)}{h['status']} {h['url']}" for h in cut(hf_)]
        lines += more(hf_)
    big = {c["id"] for c in changed}
    quiet = {i: p for i, p in (d.get("field_changes") or {}).items() if i not in big}
    if show_all:
        ua = d["unauthorized"]
        if ua["added"] or ua["removed"]:
            lines += [
                "",
                tr("401 のリポジトリ: ", "401 repositories: ")
                + f"+{', '.join(ua['added']) or '-'} / -{', '.join(ua['removed']) or '-'}",
            ]
        if quiet:
            q = sorted(quiet.items())
            lines += [
                "",
                tr(
                    "人気の数・更新日・HEAD の結果などだけの変化:",
                    "Changes only in popularity, dates, HEAD results, ...:",
                ),
            ]
            lines += [f"  {i}: {', '.join(p)}" for i, p in cut(q)]
            lines += more(q)
        if d["stats"]:
            lines += ["", tr("sources の数の変化:", "Changed numbers in sources:")]
            lines += [f"  {x['key']}: {x['old']} -> {x['new']}" for x in cut(d["stats"])]
            lines += more(d["stats"])
        return lines
    if not (d["added"] or d["removed"] or entries or head_new or d.get("top_changes")):
        lines += ["", tr("変化なし(利用者に関係のある変化はありません)", "No changes that matter to users")]
    hidden = []
    old_fail = s["head_failures"] - len(head_new)
    if quiet:
        hidden.append(tr(f"人気の数などだけの変化 {len(quiet)} 件", f"{len(quiet)} popularity-only changes"))
    if old_fail:
        hidden.append(tr(f"前からの HEAD の失敗 {old_fail} 件", f"{old_fail} older HEAD failures"))
    if d["stats"]:
        hidden.append(tr(f"sources の数 {len(d['stats'])} 件", f"{len(d['stats'])} sources numbers"))
    if d["unauthorized"]["added"] or d["unauthorized"]["removed"]:
        hidden.append(tr("401 のリポジトリの増減", "401 repository changes"))
    if hidden:
        lines += ["", tr(f"(--all で: {'、'.join(hidden)})", f"(with --all: {', '.join(hidden)})")]
    return lines


def _diff_print(lines: list[str], width: int) -> None:
    """幅に折り返して出す。字下げのあとが 1 語だけの行(ID だけの行)は折り返さない(はみ出してよい)。"""
    from .table import wrap_line

    for line in lines:
        if line.strip() and " " not in line.strip():
            print(line)
            continue
        for part in wrap_line(line, width):
            print(part)


def _gib(b: int | None) -> str:
    return "?" if not b else f"{b / GIB:.2f}"


def _diff_howto() -> str:
    return tr(
        "取ってきた一覧がまだありません({path})。"
        "先に `genmodelfit update --from-release`"
        " で最新の一覧を取ってきてから、もう一度 `genmodelfit diff` を"
        "実行してください(2 つのファイルを比べるときは `genmodelfit diff old.json new.json`)。",
        "No fetched list yet ({path}). Run `genmodelfit update --from-release` to fetch the latest list, then"
        " run `genmodelfit diff` again (to compare two files: `genmodelfit diff old.json new.json`).",
    )


def _diff_older() -> str:
    return tr(
        "後の一覧({new_label}、{new})が前の一覧({old_label}、{old})より古いです。比べる向きが逆かもしれません"
        "(新しいモデルが「消えた」に見えます)。",
        "The new list ({new_label}, {new}) is older than the old list ({old_label}, {old}). The order may be"
        ' reversed (new models would look "removed").',
    )


def _diff_older_cache() -> str:
    return tr(
        "`genmodelfit update --from-release` で取り直してください。",
        " Fetch it again with `genmodelfit update --from-release`.",
    )


def _diff_raw(path: str | Path | None) -> dict[str, Any]:
    """一覧の JSON をそのまま読む(この版が知らない鍵の変化も比べるため。merge-check と同じ)。
    None は同梱の一覧。読めない・形が違うときは load_db が先に DatabaseError を出す。"""
    from .db import bundled_text

    text = bundled_text("db.json") if path is None else Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return data if isinstance(data, dict) else {}


def _diff_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[Database, dict[str, Any], str], tuple[Database, dict[str, Any], str], list[str]] | str:
    """比べる 2 つの一覧(Database・JSON・説明)と注意。#59

    引数なし = 同梱の一覧 ↔ 取ってきた一覧(update・update --from-release が書くキャッシュ)、
    1 つ = 同梱の一覧 ↔ そのファイル、2 つ = そのまま。取ってきた一覧が無ければ案内の文を返す。
    後の一覧が前の一覧より古い(生成日時)ときは、どの形でも注意を付ける。"""
    from .db import cached_db_path

    def side(path: str | Path | None, label: str) -> tuple[Database, dict[str, Any], str]:
        db = load_db(None, use_cache=False) if path is None else load_db(path)
        return db, _diff_raw(path), label

    bundled = tr(f"同梱の一覧(genmodelfit {__version__})", f"bundled list (genmodelfit {__version__})")
    from_cache = False
    if args.old and args.new:
        old, new = side(args.old, str(args.old)), side(args.new, str(args.new))
    elif args.old:
        old, new = side(None, bundled), side(args.old, str(args.old))
    else:
        cp = cached_db_path()
        if not cp.exists():
            return _diff_howto().format(path=cp)
        old, new = side(None, bundled), side(cp, tr(f"取ってきた一覧({cp})", f"fetched list ({cp})"))
        from_cache = True
    warn = []
    ga, gb = old[0].generated_at, new[0].generated_at
    if ga and gb and gb < ga:
        text = _diff_older().format(old_label=old[2], old=ga, new_label=new[2], new=gb)
        warn.append(text + (_diff_older_cache() if from_cache else ""))
    return old, new, warn


def cmd_diff(args: argparse.Namespace) -> int:
    from .diff import diff_db, empty_diff
    from .table import output_width

    width = output_width()
    got = _diff_inputs(args)
    if isinstance(got, str):  # 取ってきた一覧が無い: 案内だけ(失敗ではない)
        if args.json:
            _out({"genmodelfit": __version__, "command": "diff", **empty_diff(got), "warnings": []})
        else:
            _diff_print([got], width)
        return 0
    (old, raw_old, old_label), (new, raw_new, new_label), warnings = got
    d = diff_db(
        old,
        new,
        comfyui_version=args.comfyui_version,
        old_label=old_label,
        new_label=new_label,
        raw=(raw_old, raw_new),
    )
    d["warnings"] = warnings
    if args.json:
        _out({"genmodelfit": __version__, "command": "diff", **d})
        return 0
    limit = args.limit if args.limit is not None else (None if args.all else DIFF_DEFAULT_LIMIT)
    _diff_print(_diff_lines(d, limit, show_all=args.all, width=width), width)
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
    sub_path, sub_repo = _bench_submission(args, m)
    code = 1 if result.failed else 0
    if args.json:
        extra = {"saved_to": str(out_path) if out_path else None}
        if not_saved:
            extra["not_saved"] = not_saved
        if sub_path is not None:
            extra["submission"] = {"saved_to": str(sub_path), "path_in_repo": sub_repo}
        _out(_envelope("bench", None, None, **extra, **result.to_dict()))
        return code
    for w in result.warnings:
        print(tr(f"注意: {w}", f"warning: {w}"), file=sys.stderr)
    s = m.settings
    if i18n.get_language() == "en":
        unknown = "unknown"
        parts = [f"{s.width}x{s.height}" if s.width else None, f"{s.frames} frames" if s.frames else None]
        parts += [f"{s.seconds:g} s of audio" if s.seconds else None, f"{s.steps} steps" if s.steps else None]
        print(f"=== bench: {m.model_id} ===")
        print(f"GPU:       {m.gpu_name} ({m.gpu_vram_gb:g} GiB, driver {m.driver or unknown})")
        print(f"ComfyUI:   {m.comfyui_version or unknown}")
        print(f"Settings:  {', '.join(p for p in parts if p) or '(as in the template)'}")
        print(f"Result:    {m.outcome}{'' if m.standard else ' (non-standard conditions)'}")
        print(f"Peak VRAM: {m.peak_vram_gb:.2f} GB (the ComfyUI process)")
        if m.peak_ram_gb is not None:
            print(f"Peak RAM:  {m.peak_ram_gb:.2f} GB")
        print(f"Time:      {m.seconds:g} s (from sending to the end, including model loading)")
        print(f"Offload:   {m.offload or unknown}")
        print(f"Note:      {m.note}")
        if result.failed:
            print(f"Failed:    {result.failed}")
        print(f"Saved to:  {out_path}" if out_path else f"Not saved: {not_saved}")
        if sub_path is not None:
            print(f"Submission: {sub_path}")
            print(
                "           check it with genmodelfit contrib"
                f" show {sub_path}; in the PR it goes to {sub_repo}"
            )
        return code
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
    if sub_path is not None:
        print(f"提出用:   {sub_path}")
        print(f"          中身は genmodelfit contrib show {sub_path} で確かめ、PR では {sub_repo} に置きます")
    return code


def _bench_submission(args: argparse.Namespace, m) -> tuple[Path | None, str | None]:
    """bench --submission: 提出用の JSON を書く(#34)。ComfyUI が同じ PC のときだけ、この PC の
    OS の種類・メモリ・GPU の世代を入れる。"""
    target = getattr(args, "submission", None)
    if not target:
        return None, None
    from .bench import is_local
    from .contrib import build_submission, suggested_path, write_submission

    system = detect() if is_local(args.url) else None
    sub = build_submission(m, system)
    return write_submission(sub, target), suggested_path(sub)


# --- contrib(#34) --------------------------------------------------------------------------------

# 終了コード(contrib)。docs/schema.md「contrib」
EXIT_BREACH = 3  # aggregate --fail-on-breach で、誤差が閾値を超えた


def _contrib_db(args: argparse.Namespace) -> Database:
    """検査・まとめに使う一覧。既定は同梱のもの(CI と手元で同じ結果にする)。"""
    return load_db(getattr(args, "db", None), use_cache=False)


def _shown(v: Any, unit: str = "") -> str:
    return tr("不明", "unknown") if v is None else f"{v}{unit}"


def format_submission(data: dict[str, Any]) -> list[str]:
    """提出の中身を人が読める形に(contrib show)。ファイルの項目はすべて出す。"""
    from .contrib import suggested_path

    system = data.get("system") or {}
    gpu = system.get("gpu") or {}
    mm = data.get("measurement") or {}
    tool = data.get("tool") or {}
    s = mm.get("settings") or {}
    if i18n.get_language() == "en":
        return _format_submission_en(data)
    gpu_bits = [
        f"{_shown(gpu.get('vram_gb'), ' GiB')}",
        f"世代 {_shown(gpu.get('arch'))}",
        f"compute {_shown(gpu.get('compute_capability'))}",
        f"ドライバ {_shown(gpu.get('driver'))}",
        # 会社・backend(0.9.0)。入っているときだけ出す(古い提出には無い)
        *(
            [f"会社 {_shown(gpu.get('vendor'))}", f"backend {_shown(gpu.get('backend'))}"]
            if {"vendor", "backend"} & set(gpu)
            else []
        ),
        # PCI の型番 ID(0.9.0。機種の番号で、個体・差し場所は表さない)。入っているときだけ
        *([f"型番 ID {_shown(gpu.get('pci_id'))}"] if "pci_id" in gpu else []),
    ]
    peak = mm.get("peak_vram_gb")
    return [
        "=== 提出する中身 ===",
        f"置き場所:   {suggested_path(data)}(PR でこの名前で足す)",
        f"形式:       contrib_schema {data.get('contrib_schema')}"
        f"({tool.get('name')} {tool.get('version')}、{data.get('kind')})",
        f"日付:       {data.get('date')}(UTC の日付だけ。時刻は入れない)",
        f"OS:         {_shown(system.get('os_family'))}(種類だけ。版・ビルドは入れない)",
        f"メモリ:     {_shown(system.get('ram_total_gb'), ' GiB')}(整数に丸めた量)",
        f"GPU:        {_shown(gpu.get('name'))}({'、'.join(gpu_bits)})",
        f"ComfyUI:    {_shown(data.get('comfyui_version'))}",
        f"モデル:     {_shown(mm.get('model_id'))}",
        f"設定:       {', '.join(f'{k}={v}' for k, v in s.items()) or '(テンプレートのまま)'}",
        f"結果:       {_shown(mm.get('outcome'))}{'' if mm.get('standard', True) else '(標準でない条件)'}",
        f"最大 VRAM:  {_shown(peak, ' GiB')}(測り方 {_shown(mm.get('vram_metric'))})",
        f"最大メモリ: {_shown(mm.get('peak_ram_gb'), ' GiB')}",
        f"時間:       {_shown(mm.get('seconds'), ' 秒')}",
        f"載せ方:     {_shown(mm.get('offload'))}",
        f"メモ:       {mm.get('note') or '(なし)'}",
        "",
        "これがファイルの中身のすべてです。genmodelfit はこのファイルをどこにも送りません。",
        "共有するのは、あなたが自分の GitHub で Pull Request を出したときだけです(CONTRIBUTING.md)。",
    ]


def _format_submission_en(data: dict[str, Any]) -> list[str]:
    """format_submission の英語(#73)。項目は日本語と同じ。"""
    from .contrib import suggested_path

    system = data.get("system") or {}
    gpu = system.get("gpu") or {}
    mm = data.get("measurement") or {}
    tool = data.get("tool") or {}
    s = mm.get("settings") or {}
    gpu_bits = [
        f"{_shown(gpu.get('vram_gb'), ' GiB')}",
        f"arch {_shown(gpu.get('arch'))}",
        f"compute {_shown(gpu.get('compute_capability'))}",
        f"driver {_shown(gpu.get('driver'))}",
        *(
            [f"vendor {_shown(gpu.get('vendor'))}", f"backend {_shown(gpu.get('backend'))}"]
            if {"vendor", "backend"} & set(gpu)
            else []
        ),
        *([f"PCI model ID {_shown(gpu.get('pci_id'))}"] if "pci_id" in gpu else []),
    ]
    peak = mm.get("peak_vram_gb")
    return [
        "=== What will be shared ===",
        f"Path:       {suggested_path(data)} (add it under this name in the PR)",
        f"Format:     contrib_schema {data.get('contrib_schema')}"
        f" ({tool.get('name')} {tool.get('version')}, {data.get('kind')})",
        f"Date:       {data.get('date')} (UTC date only, no time)",
        f"OS:         {_shown(system.get('os_family'))} (family only, no version or build)",
        f"RAM:        {_shown(system.get('ram_total_gb'), ' GiB')} (rounded to an integer)",
        f"GPU:        {_shown(gpu.get('name'))} ({', '.join(gpu_bits)})",
        f"ComfyUI:    {_shown(data.get('comfyui_version'))}",
        f"Model:      {_shown(mm.get('model_id'))}",
        f"Settings:   {', '.join(f'{k}={v}' for k, v in s.items()) or '(as in the template)'}",
        "Result:    "
        f" {_shown(mm.get('outcome'))}{'' if mm.get('standard', True) else ' (non-standard conditions)'}",
        f"Peak VRAM:  {_shown(peak, ' GiB')} (metric {_shown(mm.get('vram_metric'))})",
        f"Peak RAM:   {_shown(mm.get('peak_ram_gb'), ' GiB')}",
        f"Time:       {_shown(mm.get('seconds'), ' s')}",
        f"Offload:    {_shown(mm.get('offload'))}",
        f"Note:       {mm.get('note') or '(none)'}",
        "",
        "This is everything in the file. genmodelfit does not send this file anywhere.",
        "It is shared only when you open a Pull Request from your own GitHub account (CONTRIBUTING.md).",
    ]


def cmd_contrib_make(args: argparse.Namespace) -> int:
    from .contrib import build_submission, measurements_from_file, suggested_path, write_submission

    try:
        ms = measurements_from_file(args.source)
    except (OSError, ValueError) as e:
        raise CliError(
            tr(
                f"bench の結果を読めません: {args.source}: {e}",
                f"cannot read the bench result: {args.source}: {e}",
            )
        ) from e
    system = None if args.no_pc_info else detect()
    out = []
    target = args.output
    for m in ms:
        sub = build_submission(m, system)
        # 1 件で、ファイル名(.json)を指定したときはそこへ。ほかはフォルダの下に決まった名前で
        dest = target if len(ms) == 1 and target.lower().endswith(".json") else str(Path(target)) + os.sep
        path = write_submission(sub, dest)
        out.append({"saved_to": str(path), "path_in_repo": suggested_path(sub), "submission": sub})
    if args.json:
        _out(_envelope("contrib", None, None, action="make", submissions=out))
        return 0
    for o in out:
        print(
            tr(f"提出用の JSON を書きました: {o['saved_to']}", f"Wrote the submission JSON: {o['saved_to']}")
        )
        print(
            tr(
                f"  PR では {o['path_in_repo']} に置きます。"
                f"送る前に genmodelfit contrib show {o['saved_to']} で確かめてください",
                f"  In the PR it goes to {o['path_in_repo']}."
                f" Check it with genmodelfit contrib show {o['saved_to']} before sharing",
            )
        )
    return 0


def cmd_contrib_show(args: argparse.Namespace) -> int:
    from .contrib import privacy_checklist, read_submission, validate_submission

    try:
        data, raw = read_submission(args.file)
    except (OSError, ValueError) as e:
        raise CliError(
            tr(
                f"提出用の JSON を読めません: {args.file}: {type(e).__name__}",
                f"cannot read the submission JSON: {args.file}: {type(e).__name__}",
            )
        ) from e
    checks = privacy_checklist(data, raw)
    report = validate_submission(data, _contrib_db(args), path=args.file, raw=raw, check_filename=False)
    good = report.ok and all(c["ok"] is not False for c in checks)
    if args.json:
        _out(
            _envelope(
                "contrib",
                None,
                None,
                action="show",
                file=str(args.file),
                ok=good,
                submission=data,
                checklist=checks,
                report=report.to_dict(),
            )
        )
        return 0 if good else 1
    if isinstance(data, dict):
        for line in format_submission(data):
            print(line)
    print()
    print(tr("=== 個人の情報の点検 ===", "=== Privacy checklist ==="))
    for c in checks:
        mark = "[  ]" if c["ok"] is None else ("[ok]" if c["ok"] else "[NG]")
        print(f"  {mark} {c['label']}" + (f": {c['detail']}" if c["detail"] else ""))
    print()
    for e in report.errors:
        print(tr("誤り", "error") + f" {e['code']}: {e['message']}")
    for w in report.warnings:
        print(tr("注意", "warning") + f" {w['code']}: {w['message']}")
    for v in report.notices:
        print(
            tr("確認", "notice")
            + f" {v['code']}: {v['message']}"
            + tr(
                "(失敗にはしません。PR をマージする人が見ます)",
                " (not a failure; the maintainer who merges checks it)",
            )
        )
    nw = len(report.warnings)
    print(
        tr("検査: ", "check: ")
        + ("ok" if good else tr("NG(直してから出してください)", "NG (fix it before submitting)"))
        + (tr(f"(注意 {nw} 件)", f" ({nw} warnings)") if report.warnings else "")
    )
    return 0 if good else 1


def _result_word(r) -> str:
    if not r.ok:
        return "NG"
    return "notice" if r.notices else "ok"


def _validate_markdown(reports: list) -> list[str]:
    """CI のまとめ用の表。提出から来た文字(パス・モデル ID・GPU 名・メッセージ)は、
    すべて md_cell で無害にする。確認(notices)は失敗にしないが、マージする人が見るので表の上に数を出す。"""
    from .contrib import md_cell

    words = [_result_word(r) for r in reports]
    lines = [
        "## genmodelfit contrib validate",
        "",
        tr(
            f"ファイル {len(reports)} 件: ok {words.count('ok')}、確認つきの ok {words.count('notice')}、"
            f"NG {words.count('NG')}",
            f"{len(reports)} files: {words.count('ok')} ok, {words.count('notice')} ok with notices,"
            f" {words.count('NG')} NG",
        ),
        "",
    ]
    if words.count("notice"):
        lines += [
            tr(
                f"**マージする人へ: 確認の項目のある提出が {words.count('notice')} 件あります**"
                "(式から大きく外れる・よくある PC"
                " で段階を変える・一覧に無いモデルなど。失敗にはしていません。"
                "マージすると見積もりに使われます)",
                f"**For the maintainer: {words.count('notice')} submissions have notices**"
                " (far from the formula, changing the fit level on common PCs, a model not in the list, ...;"
                " not failures. Once merged they are used for estimates)",
            ),
            "",
        ]
    lines += [
        tr(
            "| 結果 | ファイル | モデル | GPU | 誤り・確認・注意 |",
            "| Result | File | Model | GPU | Errors / notices / warnings |",
        ),
        "|---|---|---|---|---|",
    ]
    err_w, notice_w, warn_w = tr("誤り", "error"), tr("確認", "notice"), tr("注意", "warning")
    for r, word in zip(reports, words, strict=True):
        msgs = [f"{err_w} {e['code']}: {e['message']}" for e in r.errors]
        msgs += [f"{notice_w} {v['code']}: {v['message']}" for v in r.notices]
        msgs += [f"{warn_w} {w['code']}: {w['message']}" for w in r.warnings]
        text = "<br>".join(md_cell(m) for m in msgs) or "-"
        mark = {"ok": "ok", "notice": tr("ok(**確認**)", "ok (**notice**)"), "NG": "**NG**"}[word]
        lines.append(
            f"| {mark} | {md_cell(r.path)} | {md_cell(r.model_id or '-')} |"
            f" {md_cell(r.gpu_name or '-')} | {text} |"
        )
    return lines


def cmd_contrib_validate(args: argparse.Namespace) -> int:
    from .contrib import find_files, validate_files

    files: list[Path] = []
    for f in args.files:
        p = Path(f)
        files += find_files(p) if p.is_dir() else [p]
    reports = validate_files(files, _contrib_db(args), check_filename=not args.no_filename_check)
    bad = [r for r in reports if not r.ok or (args.strict and r.warnings)]
    code = 1 if bad else 0
    if args.json:
        _out(
            _envelope(
                "contrib",
                None,
                None,
                action="validate",
                ok=not bad,
                files=len(reports),
                valid=sum(r.ok for r in reports),
                invalid=sum(not r.ok for r in reports),
                notices=sum(1 for r in reports if r.ok and r.notices),
                warnings=sum(len(r.warnings) for r in reports),
                reports=[r.to_dict() for r in reports],
            )
        )
        return code
    if args.markdown:
        for line in _validate_markdown(reports):
            print(line)
        return code
    if not reports:
        print(tr("検査するファイルがありません", "no files to check"))
    err_w, notice_w, warn_w = tr("誤り", "error"), tr("確認", "notice"), tr("注意", "warning")
    for r in reports:
        what = ", ".join(x for x in (r.model_id, r.gpu_name) if x)
        print(f"{_result_word(r):<6}  {r.path}" + (f"  ({what})" if what else ""))
        for e in r.errors:
            print(f"      {err_w} {e['code']}: {e['message']}")
        for v in r.notices:
            print(f"      {notice_w} {v['code']}: {v['message']}")
        for w in r.warnings:
            print(f"      {warn_w} {w['code']}: {w['message']}")
    words = [_result_word(r) for r in reports]
    print(
        tr(
            f"ファイル {len(reports)} 件: ok {words.count('ok')}、確認つきの ok {words.count('notice')}"
            f"(失敗にはしない。マージする人が見る)、NG {words.count('NG')}",
            f"{len(reports)} files: {words.count('ok')} ok, {words.count('notice')} ok with notices"
            f" (not failures; the maintainer checks them), {words.count('NG')} NG",
        )
    )
    return code


def format_report(rep: dict[str, Any]) -> list[str]:
    """精度の点検の表(Markdown。毎週の自動 PR にそのまま貼れる形)。"""
    from .calibrate import check_table
    from .contrib import CHECK_RAM_GB, md_cell

    thr = rep["threshold_pct"]
    lines = [
        tr("## 見積もりの精度の点検", "## Accuracy check"),
        "",
        tr(
            f"係数を合わせた実測 {rep['n_fit']} 件。閾値: 誤差の絶対値 {thr:g}%"
            "(評価できる種類の leave-one-out と、ほかの利用者の実測の照合で見る)。",
            f"{rep['n_fit']} measurements fitted. Threshold: absolute error {thr:g}%"
            " (leave-one-out for the evaluable types, and the check against community measurements).",
        ),
        "",
        tr(
            "| 種類 | LOO 件数 | LOO 平均 | LOO 最大 | 評価できる | 照合 件数 | 照合 最大 |",
            "| Type | LOO n | LOO mean | LOO max | Evaluable | Check n | Check max |",
        ),
        "|---|---:|---:|---:|---|---:|---:|",
    ]

    def pct(v: Any) -> str:
        return "-" if v is None else f"{v:g}%"

    for mod, b in rep["by_modality"].items():
        lines.append(
            f"| {mod} | {b['loo_n']} | {pct(b['loo_mean_abs_pct'])} | {pct(b['loo_max_abs_pct'])} |"
            f" {tr('はい', 'yes') if b['evaluable'] else tr('いいえ', 'no')} | {b['check_n']} |"
            f" {pct(b['check_max_abs_pct'])} |"
        )
    lines.append("")
    if rep["breach"]:
        lines.append(
            tr(
                f"**警告: 誤差が閾値 {thr:g}% を超えたものがあります**",
                f"**Warning: some errors exceed the {thr:g}% threshold**",
            )
        )
        lines.append("")
        for b in rep["breaches"]:
            gpu_name = md_cell(b.get("gpu_name") or "?")
            lines.append(
                tr(
                    f"- {md_cell(b['model_id'])}({gpu_name}、{b['kind']}、{b.get('origin')}):",
                    f"- {md_cell(b['model_id'])} ({gpu_name}, {b['kind']}, {b.get('origin')}):",
                )
                + f" {b['error_pct']:+g}%"
            )
    else:
        lines.append(
            tr(f"閾値 {thr:g}% を超えたものはありません。", f"Nothing exceeds the {thr:g}% threshold.")
        )
    lines += [
        "",
        tr(
            "### ほかの利用者の実測で変わる段階",
            "### Fit-level changes caused by community data",
        ),
        "",
    ]
    if rep.get("fit_changes"):
        lines += [
            tr("| モデル | GPU | VRAM | メモリ | 前 | 後 |", "| Model | GPU | VRAM | RAM | Before | After |"),
            "|---|---|---:|---:|---|---|",
        ]
        for c in rep["fit_changes"]:
            lines.append(
                f"| {md_cell(c['model_id'])} | {md_cell(c.get('gpu_name') or '?')} | {c['vram_gb']:g} GB |"
                f" {c['ram_gb']:g} GB | {c['before']} | {c['after']} |"
            )
    else:
        ram = ", ".join(f"{v:g}" for v in CHECK_RAM_GB)
        lines.append(
            tr(
                f"ありません(実測と同じ GPU の PC × メモリ {ram}"
                " GB で確かめた。ほかの GPU の PC には使わない)。",
                "None (checked on PCs with the measured GPU ×"
                f" {ram} GB RAM; not used for PCs with other GPUs).",
            )
        )
    if rep.get("check"):
        lines += ["", *check_table(rep["check"])]
    return lines


def _contrib_aggregate_check(args: argparse.Namespace) -> int:
    """aggregate --check: 同梱のまとめに入っていない提出を数える(#90。まとめは版を出すときに作る)。
    提出のフォルダが無ければ提出 0 件として比べる。"""
    from .contrib import bundled_status
    from .db import COMMUNITY_FILE, bundled_text

    if args.output or args.report:
        raise CliError(
            tr(
                "--check は --output・--report と一緒に使えません",
                "--check cannot be used with --output / --report",
            )
        )
    st = bundled_status(args.dir, json.loads(bundled_text(COMMUNITY_FILE)), _contrib_db(args))
    code = 0 if st["ok"] and not st["pending"] else 1
    if args.json:
        _out(_envelope("contrib", None, None, action="aggregate_check", dir=str(args.dir), **st))
        return code
    print(
        tr(
            f"同梱のまとめ: 提出 {st['listed']} 件"
            f"({'それだけをまとめ直すと同じ' if st['matches'] else 'まとめ直すと違う'})",
            f"Bundled aggregate: {st['listed']} submissions"
            + (
                " (re-aggregating them gives the same file)"
                if st["matches"]
                else " (re-aggregating them differs)"
            ),
        )
    )
    for h in st["missing"]:
        print(
            tr(
                f"  {args.dir} に無い(または検査を通らない)提出: {h[:8]}",
                f"  missing from {args.dir} (or failing the checks): {h[:8]}",
            )
        )
    print(
        tr(
            f"{args.dir} の検査を通った提出 {st['valid']} 件のうち、同梱のまとめにまだ入っていないもの"
            f" {len(st['pending'])} 件",
            f"{len(st['pending'])} of the {st['valid']} valid submissions in {args.dir}"
            " are not in the bundled aggregate yet",
        )
    )
    for path in st["pending"]:
        print(f"  {path}")
    if st["pending"] or not st["ok"]:
        print(
            tr(
                "版を出すときに genmodelfit contrib aggregate --output"
                " src/genmodelfit/data/measurements_community.json で入れ直します(docs/release.md)",
                "Rebuild it when making a release: genmodelfit contrib aggregate --output"
                " src/genmodelfit/data/measurements_community.json (docs/release.md)",
            )
        )
    return code


def cmd_contrib_aggregate(args: argparse.Namespace) -> int:
    from .contrib import (
        accuracy_report,
        aggregate,
        community_document,
        load_dir,
        md_cell,
        report_db,
        write_community,
    )

    if args.check:
        return _contrib_aggregate_check(args)
    if not Path(args.dir).is_dir():
        raise CliError(
            tr(f"提出のフォルダがありません: {args.dir}", f"no such submissions folder: {args.dir}")
        )
    db = _contrib_db(args)
    subs, reports = load_dir(args.dir, db)
    rows = aggregate(subs)
    if args.output and not rows and not args.allow_empty:
        raise CliError(
            tr(
                "まとめが空なので書き出しません(同梱のまとめを空で上書きしないため。空にするなら"
                " --allow-empty)",
                "the result is empty, so nothing is written (so as not to overwrite the bundled data with an"
                " empty one; use --allow-empty to write it anyway)",
            )
        )
    written = str(write_community(rows, args.output, subs)) if args.output else None
    rep = accuracy_report(report_db(db, rows), args.threshold) if args.report else None
    code = EXIT_BREACH if (rep and rep["breach"] and args.fail_on_breach) else 0
    stats = {
        "dir": str(args.dir),
        "files": len(reports),
        "valid": sum(r.ok for r in reports),
        "invalid": sum(not r.ok for r in reports),
        "notices": sum(1 for r in reports if r.ok and r.notices),
        "submissions_used": len(subs),
        "groups": len(rows),
        "consistent": sum(1 for r in rows if r["consistent"]),
    }
    if args.json:
        payload: dict[str, Any] = {"action": "aggregate", **stats, "written_to": written}
        payload["community"] = community_document(rows, subs)
        if rep is not None:
            payload["report"] = rep
        _out(_envelope("contrib", None, None, **payload))
        return code
    print(
        tr(
            f"提出 {stats['files']} 件(検査を通った {stats['valid']}"
            f" 件、通らなかった {stats['invalid']} 件)→ "
            f"{stats['groups']} 組にまとめた(そろった組"
            f" {stats['consistent']}。ok の組は 1 件から見積もりに使う)",
            f"{stats['files']} submissions ({stats['valid']}"
            f" passed the checks, {stats['invalid']} failed) -> "
            f"{stats['groups']} groups ({stats['consistent']} consistent; ok groups are used for estimates"
            " from one submission)",
        )
    )
    for r in rows:
        if r["outcome"] != "ok":
            mark = tr("oom だけ", "oom only")
        else:
            mark = tr("そろった", "consistent") if r["consistent"] else tr("使う", "used")
        counts = f"n={r['n_submissions']}, used={r['n_used']}, outliers={r['n_outliers']}, oom={r['n_oom']}, "
        spread = tr(f"ばらつき {r['spread_rel_pct']:g}%", f"spread {r['spread_rel_pct']:g}%")
        print(
            f"  {md_cell(r['model_id'])}  {md_cell(r['gpu_name'])}  {r['peak_vram_gb']:g} GiB  "
            + tr(f"({counts}{spread}、{mark})", f"({counts}{spread}, {mark})")
        )
    if written:
        print(tr(f"書き出し: {written}", f"written: {written}"))
    if rep is not None:
        print()
        for line in format_report(rep):
            print(line)
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
    if i18n.get_language() == "ja":
        print(f"=== 自動のおすすめの条件(ComfyUI {target or '?'}) ===")
        print(f"モデル {summary['models']} 件: 出せる {summary['auto']} / 保留 {summary['held']}")
        print("条件                落ちた数  最初に落ちた数")
    else:
        print(f"=== Auto-recommend gate (ComfyUI {target or '?'}) ===")
        print(f"models {summary['models']}: auto {summary['auto']} / held {summary['held']}")
        print("check               failing  first-failing")
    for cid in CHECK_IDS:
        print(f"  {cid:<18}{summary['failing'][cid]:>8}  {summary['first_failing'][cid]:>13}")
    if args.only_auto or args.only_held:
        print()
        for r in shown:
            state = tr("出せる", "auto") if r.auto else tr("保留: ", "held:") + r.held_by[0]
            print(f"  {r.model_id}  {state}")
    return 0


# --- review(#42) ---------------------------------------------------------------------------------


def _review_error(args: argparse.Namespace, e: Exception) -> int:
    if args.json:
        _out({"genmodelfit": __version__, "command": "review", "error": str(e)})
    else:
        print(tr(f"エラー: {e}", f"error: {e}"), file=sys.stderr)
    return 1


def cmd_review_list(args: argparse.Namespace) -> int:
    from .review import review_list

    db = _db(args)
    items = review_list(db, include_ok=args.all)
    if args.json:
        _out(
            _envelope("review", None, db, pending=sum(1 for i in items if i["status"] != "ok"), models=items)
        )
        return 0
    pending = [i for i in items if i["status"] != "ok"]
    print(
        tr(
            f"確認待ち {len(pending)} 件 / 一覧 {len(db.models)} 件({_SOURCE_JA.get(db.origin, db.origin)})",
            f"pending {len(pending)} / {len(db.models)} entries in the list ({db.origin})",
        )
    )
    for i in items:
        why = ", ".join(i["reasons"]) or "-"
        print(f"  [{i['status']}] {i['id']}  ({i['modality']}/{i['content']}/{i['license']})  {why}")
        if i["status"] == "ok" and i["decided_by"]:
            print(f"      {i['decided_by']} {i['decided_at'] or ''}  {' '.join(i['evidence'])}".rstrip())
    return 0


def cmd_review_approve(args: argparse.Namespace) -> int:
    from .review import ReviewError, approve_file

    try:
        m = approve_file(args.db, args.id, args.evidence or [], note=args.note, by=args.by)
    except ReviewError as e:
        return _review_error(args, e)
    r = m.review
    assert r is not None
    if args.json:
        _out(
            {"genmodelfit": __version__, "command": "review", "action": "approve", "id": m.id, "db": args.db}
            | {
                "review": {
                    k: getattr(r, k)
                    for k in ("status", "reasons", "decided_by", "decided_at", "evidence", "note")
                }
            }
        )
        return 0
    reasons = ", ".join(r.reasons) or "-"
    print(
        tr(f"確認済みにしました: {m.id}({reasons})→ {args.db}", f"Approved: {m.id} ({reasons}) -> {args.db}")
    )
    return 0


def cmd_review_reject(args: argparse.Namespace) -> int:
    from .review import ReviewError, reject_file

    try:
        res = reject_file(args.db, args.id, args.note, rejected_path=args.rejected, hide_name=args.hide_name)
    except ReviewError as e:
        return _review_error(args, e)
    if args.json:
        _out({"genmodelfit": __version__, "command": "review", "action": "reject", "id": args.id, **res})
        return 0
    print(
        tr(
            f"一覧から外しました: {args.id} → {res['db']}(外した記録:"
            f" {res['rejected_file']}、鍵 {res['key']})",
            f"Removed from the list: {args.id} -> {res['db']}"
            f" (record: {res['rejected_file']}, key {res['key']})",
        )
    )
    return 0


# --- download(#31) -------------------------------------------------------------------------------


def _gb(n: int | None) -> str:
    return f"{n / GIB:.2f} GB" if n is not None else "?"


# --comfyui-dir の既定(#75。docs/download.ja.md「コマンドライン」)
COMFYUI_DIR_ENV = "GENMODELFIT_COMFYUI_DIR"


def _is_windows() -> bool:
    """案内に出すシェルの形を決める OS(試験で差し替える)。Windows は PowerShell、ほかは POSIX sh。"""
    return os.name == "nt"


def _asks_phrase(asks: list[str]) -> str:
    """尋ねる同意の種類 → 「このモデルは…」に続く文(#75)。"""
    unc, lic = "uncensored" in asks, "unknown_license" in asks
    if unc and lic:
        return tr(
            "無検閲(成人向けを含みうる)で、ライセンスの記載が無いファイルを含みます"
            "(上の確かめる場所で条件を確かめてください)",
            "is uncensored (may include adult content) and has files with no stated license"
            " (check the pages above)",
        )
    if unc:
        return tr("無検閲(成人向けを含みうる)です", "is uncensored (may include adult content)")
    return tr(
        "ライセンスの記載が無いファイルを含みます(上の確かめる場所で条件を確かめてください)",
        "has files with no stated license (check the pages above)",
    )


def _consent_line(plan, asks: list[str]) -> str:
    """端末で尋ねる同意を 1 行にまとめる(#75。無検閲・ライセンスの記載が無いファイル)。"""
    parts_ja, parts_en = [], []
    if "uncensored" in asks:
        parts_ja.append("無検閲(成人向けを含みうる)")
        parts_en.append("uncensored (may include adult content)")
    if "unknown_license" in asks:
        names = ", ".join(f.name for f in plan.files if f.license is None)
        parts_ja.append(f"ライセンスの記載が無い: {names}(上の確かめる場所で確かめる)")
        parts_en.append(f"no stated license: {names} (check the pages above)")
    return tr(
        "  同意:      " + "・".join(parts_ja) + "— 落とす前に y/N で尋ねます",
        "  consent:   " + "; ".join(parts_en) + " — asked y/N before downloading",
    )


def blocked_texts(plan) -> list[str]:
    """止める理由を CLI の言い方で(同意の 2 つは CLI の指定を書く。ほかはライブラリの文のまま。#75)。"""
    out = []
    codes = list(plan.blocked_codes) + [""] * (len(plan.blocked) - len(plan.blocked_codes))
    for text, code in zip(plan.blocked, codes, strict=False):
        if code == "uncensored":
            text = tr(
                "無検閲のモデルです。落とすなら --uncensored(端末なら y/N の質問で尋ねます)",
                "This is an uncensored model. To download it, pass --uncensored (in a terminal you are asked"
                " y/N instead)",
            )
        elif code == "unknown_license":
            names = ", ".join(f.name for f in plan.files if f.license is None)
            text = tr(
                f"ライセンスが分からないファイルがあります: {names}。"
                "中身の条件を自分で確かめたうえで落とすなら --any-license(端末なら y/N の質問で尋ねます)",
                f"Some files have an unknown license: {names}. To download them after checking the terms"
                " yourself, pass --any-license (in a terminal you are asked y/N instead)",
            )
        out.append(text)
    return out


def format_plan(
    plan, *, consent: bool = False, comfyui_from: str | None = None, asks: list[str] | None = None
) -> list[str]:
    """download --list と同意の画面の文(人が読む形。幅は _emit で 100 桁までに折り返す)。

    comfyui_from: ComfyUI のフォルダをどこで決めたか("flag" / "env" / "default" = どちらも無く既定の場所。
    None = 知らせない)。asks: 端末の質問で尋ねる同意の種類(uncensored / unknown_license。#75)。"""
    from .downloader import (
        LAYOUT_WARNING_EN,
        LAYOUT_WARNING_JA,
        UNKNOWN_WARNING_EN,
        UNKNOWN_WARNING_JA,
        license_label,
    )

    ja = i18n.get_language() == "ja"
    # 見出し(#76。日本語は全角の幅で値の頭をそろえる。英語は前からの形のまま)
    L = (lambda j, e: _ja_label(j, 14)) if ja else (lambda j, e: f"  {e + ':':<10} ")
    out = [f"{plan.model_name}  [{plan.model_id}]"]
    out.append(
        L("ワークフロー", "workflow")
        + tr(
            f"{plan.name}({plan.modality}、{plan.content}、精度 {plan.precision or '?'})",
            f"{plan.name}  ({plan.modality}, {plan.content}, precision {plan.precision or '?'})",
        )
    )
    out.append(
        L("この PC", "this PC")
        + tr(
            f"{plan.fit_level}({plan.run_mode_label})。{plan.chosen}",
            f"{plan.fit_level} ({plan.run_mode_label}); {plan.chosen}",
        )
    )
    if plan.layout == "comfyui":
        where = tr("ComfyUI の models のフォルダ", "ComfyUI models folder")
    else:
        where = tr("models のフォルダ(ComfyUI と同じ並び)", "models folder (ComfyUI layout)")
    if comfyui_from == "env":
        where += tr(f"、{COMFYUI_DIR_ENV}", f", {COMFYUI_DIR_ENV}")
    out.append(
        L("置き場所", "to")
        + tr(f"{plan.root}({where}: <folder>/<file>)", f"{plan.root}  ({where}: <folder>/<file>)")
    )
    if comfyui_from == "default":
        out += _comfyui_hint()
    out.append(tr(f"  ファイル({len(plan.files)} 件。GB = GiB):", f"  files ({len(plan.files)}; GB = GiB):"))
    lic_label = tr("        ライセンス: ", "        license: ")
    for f in plan.files:
        state = {
            "download": "",
            "resume": tr(f"  {_gb(f.part_bytes)} から続きを落とす", f"  resume from {_gb(f.part_bytes)}"),
            "present": tr("  既にある", "  already there"),
        }
        mark = state.get(f.status, f"  {f.status.upper()}")
        out.append(f"    - {f.relpath or f.name}  {_gb(f.size_bytes)}{mark}")
        # 置き場所が分からない(other/)ことは、パスと warnings で知らせる
        if f.directory_basis:
            out.append(_basis_text(f))
        lic = license_label(f)
        if f.license is None:
            lic += "" if ja else " (UNKNOWN)"  # 日本語は license_label が「不明」と書く
        elif not f.license_allowed:
            lic += tr("(許可リストに無い: 条件を読む)", " (not on the allowlist: read the terms)")
        if f.license_url:
            out.append(f"{lic_label}{lic}  {f.license_url}")
        else:
            missing = (
                tr("ライセンスの記載がありません", "no license stated")
                if f.license is None
                else tr("一覧にライセンスの原文の URL がありません", "no license URL in the list")
            )
            out.append(f"{lic_label}{lic}{tr('。', '; ')}{missing}")
            if (
                f.check_url
                and not plan.model_id.startswith("hf:")
                and (f.url or "").startswith(f.check_url + "/")
            ):
                # ファイルを置いているリポジトリ(公式テンプレートの Comfy-Org など)。作者とは限らない
                out.append(
                    tr(
                        "        確かめる場所(ファイルの置き場所のリポジトリ。作者とは限らない): "
                        + f.check_url,
                        "        check (the file's repository; may not be the original publisher): "
                        + f.check_url,
                    )
                )
            elif f.check_url:
                out.append(tr(f"        確かめる場所: {f.check_url}", f"        check:   {f.check_url}"))
        if f.license_note:
            out.append(tr(f"        メモ: {f.license_note}", f"        note: {f.license_note}"))
        if f.reason:
            out.append(f"        {f.status}: {f.reason}")
    out.append(
        L("合計", "total")
        + tr(
            f"{_gb(plan.total_bytes)}。落とす量 {_gb(plan.download_bytes)}"
            f"(既にある {_gb(plan.present_bytes)})",
            f"{_gb(plan.total_bytes)}; to download {_gb(plan.download_bytes)}"
            f" (already there {_gb(plan.present_bytes)})",
        )
    )
    free = _gb(plan.disk_free_bytes) if plan.disk_free_bytes is not None else tr("不明", "unknown")
    if plan.download_bytes:
        out.append(
            L("ディスク", "disk")
            + tr(
                f"空き {free}。要る量 {_gb(plan.disk_needed_bytes)}(落とす量 + 1 GB)",
                f"{free} free; needs {_gb(plan.disk_needed_bytes)} (to download + 1 GB)",
            )
        )
    else:
        out.append(
            L("ディスク", "disk")
            + tr(
                f"空き {free}。落とすものは無い(置いてあるファイルの sha256 をもう一度確かめる)",
                f"{free} free; nothing to download (sha256 of the files there is checked again)",
            )
        )
    if plan.budget_bytes is not None:
        out.append(
            L("上限", "budget")
            + tr(f"{_gb(plan.budget_bytes)}(--budget)", f"{_gb(plan.budget_bytes)} (--budget)")
        )
    if plan.license_url:
        out.append(L("利用条件", "terms") + plan.license_url)
    if plan.alternatives:
        alts = ", ".join(
            f"{a['precision'] or '?'}/{a['use']} {a['fit_level']} {_gb(a['size_bytes'])} [{a['id']}]"
            for a in plan.alternatives[:3]
        )
        rest = len(plan.alternatives) - 3
        more = tr(f"(ほかに {rest} 件)", f" (+{rest} more)") if rest > 0 else ""
        out.append(L("ほかの候補", "others") + f"{alts}{more}")
    for w in plan.warnings:
        if comfyui_from == "default" and w in (LAYOUT_WARNING_JA, LAYOUT_WARNING_EN):
            continue  # 上の「ComfyUI:」の案内にまとめた
        if asks and w.startswith((UNKNOWN_WARNING_JA, UNKNOWN_WARNING_EN)):
            continue  # 下の「同意」の行にまとめた(ライブラリの warnings はアプリ向けにそのまま)
        out.append(tr(f"  注意: {w}", f"  warning: {w}"))
    if asks:
        out.append(_consent_line(plan, asks))
    out += [tr(f"  落とせません: {b}", f"  cannot download: {b}") for b in blocked_texts(plan)]
    if consent and asks and not plan.blocked:
        out.append("")
        if any(not f.license_url for f in plan.files):
            out.append(
                tr(
                    f"このモデルは{_asks_phrase(asks)}。上のファイルを、それぞれのライセンス・利用条件に従って"
                    "使うこと(ライセンスの原文の URL が無いファイルは、上の確かめる場所で条件を自分で確かめた"
                    "うえで)に同意して、それでも落としますか?",
                    f"This model {_asks_phrase(asks)}. Do you agree to use the files above under their"
                    " licenses and terms (for files without a license URL, after checking the pages shown"
                    " above yourself), and download anyway?",
                )
            )
        else:
            out.append(
                tr(
                    f"このモデルは{_asks_phrase(asks)}。上のファイルを、それぞれのライセンス・利用条件(URL)に"
                    "従って使うことに同意して、それでも落としますか?",
                    f"This model {_asks_phrase(asks)}. Do you agree to use the files above under their"
                    " licenses and terms (URLs), and download anyway?",
                )
            )
    elif consent and not plan.blocked and not plan.download_bytes:
        out.append("")
        out.append(
            tr(
                "落とすものはありません(置いてあるファイルの sha256 だけ確かめます)。続けますか?",
                "Nothing to download (only the sha256 of the files there is checked). Continue?",
            )
        )
    elif consent and not plan.blocked and any(not f.license_url for f in plan.files):
        out.append("")
        out.append(
            tr(
                "上のファイルを、それぞれのライセンス・利用条件に従って使うこと(ライセンスの原文の URL が無い"
                "ファイルは、上の確かめる場所で条件を自分で確かめたうえで)に同意して落としますか?",
                "Do you agree to use the files above under their licenses and terms (for files without a"
                " license URL, after checking the pages shown above yourself), and download them?",
            )
        )
    elif consent and not plan.blocked:
        out.append("")
        out.append(
            tr(
                "上のファイルを、それぞれのライセンス・利用条件(URL)に従って使うことに同意して落としますか?",
                "Do you agree to use the files above under their"
                " licenses and terms (URLs), and download them?",
            )
        )
    return out


def _basis_text(f) -> str:
    """推定した置き場所の行(根拠 directory_basis を人が読む文に。#75)。"""
    b = f.directory_basis
    params, ref, tpl = f"{b.get('params', 0):,}", b.get("file", "?"), b.get("template", "?")
    needs = b.get("needs")
    return tr(
        f"        置き場所: {f.directory}(一覧に無いので推定: ヘッダのパラメータ数 {params} が"
        f"公式テンプレート {tpl} の {ref} と同じ"
        + (f"。読むには ComfyUI にカスタムノード {needs} が要る" if needs else "")
        + ")",
        f"        folder:  {f.directory} (not in the list; inferred: header parameter count {params} matches"
        f" {ref} in the official template {tpl}"
        + (f"; ComfyUI needs the {needs} custom nodes to load it" if needs else "")
        + ")",
    )


def _comfyui_hint() -> list[str]:
    """ComfyUI のフォルダを指定していないときの案内(#75)。ライブラリの「同じ並びで置く」知らせも
    ここにまとめる。
    環境変数の決め方は、この OS のシェルの形だけ出す。"""
    if _is_windows():
        example = f"               $env:{COMFYUI_DIR_ENV} = 'C:\\ComfyUI'"
    else:
        example = f'               export {COMFYUI_DIR_ENV}="$HOME/ComfyUI"'
    return [
        tr(
            _ja_label("ComfyUI", 14)
            + "指定なし。ここは genmodelfit のフォルダで、ComfyUI は読みません。--comfyui-dir で"
            " ComfyUI のフォルダを渡すか、この端末で環境変数を決めるか、"
            "extra_model_paths.yaml にこのフォルダを足す:",
            "  ComfyUI:   not set. This is genmodelfit's folder, which ComfyUI does not read: pass"
            " --comfyui-dir, set the folder for this terminal, or add this folder to extra_model_paths.yaml:",
        ),
        example,
    ]


# 引用符なしで両方のシェル(PowerShell・POSIX sh)にそのまま渡る文字
_SHELL_SAFE = re.compile(r"[A-Za-z0-9_./:=+%-]+")


def shell_quote(arg: str, windows: bool | None = None) -> str:
    """コマンドの 1 つの引数を、貼り付けて使える形にする(#75)。

    安全な文字だけならそのまま。ほかは ' で囲む(PowerShell も POSIX sh も ' の中は $・`・\\ を展開しない)。
    中に ' があるときだけシェルで形が分かれるので、この OS のシェルの形にする
    (PowerShell は '' と重ねる、POSIX sh は '\\'' で閉じて足して開き直す)。windows=None は _is_windows()。"""
    if arg and _SHELL_SAFE.fullmatch(arg):
        return arg
    if "'" not in arg:
        return f"'{arg}'"
    if _is_windows() if windows is None else windows:
        return "'" + arg.replace("'", "''") + "'"
    return "'" + arg.replace("'", "'\\''") + "'"


def _num(v: float | int) -> str:
    """数の引数を、解釈した値のまま文字にする(丸めない。8.0 は 8)。"""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(v)


def _prog() -> str:
    """利用者が使った起動のしかた(python -m genmodelfit か genmodelfit)。"""
    argv0 = sys.argv[0] if sys.argv else ""
    spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    by_path = (
        os.path.basename(argv0) == "__main__.py" and os.path.basename(os.path.dirname(argv0)) == "genmodelfit"
    )
    if by_path or getattr(spec, "name", "") == "genmodelfit.__main__":
        return "python -m genmodelfit"
    return "genmodelfit"


def _download_command(args: argparse.Namespace, model_id: str, add: list[str]) -> str:
    """利用者の指定(解釈した引数から。argv の字面ではない)に add の指定を足した download のコマンド(#75)。

    モデルは計画のワークフローの ID で書く(同じワークフローになる。精度も ID で決まるので
    --precision は付けない)。
    --list・--json・--yes は付けない(落とすためのコマンドなので)。環境変数で決めた ComfyUI のフォルダは
    付けない(同じ環境変数で同じ場所になる)。"""
    parts = ["download", model_id]
    if args.budget is not None:
        parts += ["--budget", _num(args.budget)]
    if args.comfyui_dir:
        parts += ["--comfyui-dir", args.comfyui_dir]
    elif args.output_dir:
        parts += ["--output-dir", args.output_dir]
    if "--uncensored" in add or args.content == "all":
        parts.append("--uncensored")
    elif args.content != DEFAULT_CONTENT:
        parts += ["--content", args.content]
    if args.include_unknown_license or "--any-license" in add:
        parts.append("--any-license")
    if "--yes" in add:
        parts.append("--yes")
    for flag, value in (
        ("--db", args.db),
        ("--vram", None if args.vram is None else _num(args.vram)),
        ("--ram", None if args.ram is None else _num(args.ram)),
        ("--license-allowlist", args.license_allowlist),
        ("--comfyui-version", args.comfyui_version),
    ):
        if value is not None:
            parts += [flag, value]
    if args.max_head_age_days != 30:
        days = args.max_head_age_days
        parts += ["--max-head-age-days", "none" if days is None else _num(days)]
    if args.no_cache:
        parts.append("--no-cache")
    if args.allow_uncensored_auto:
        parts.append("--allow-uncensored-auto")
    if getattr(args, "lang", None):
        parts += ["--lang", args.lang]
    return " ".join([_prog(), *(shell_quote(p) for p in parts)])


def _repo_page(url: str | None) -> str | None:
    """Hugging Face のファイルの URL → そのリポジトリのページ(…/resolve/… の前)。ほかは None。"""
    if url and url.startswith("https://huggingface.co/") and "/resolve/" in url:
        return url.split("/resolve/", 1)[0]
    return None


def _consent_flags(codes: list[str]) -> tuple[list[str], list[str]]:
    """同意の種類 → 足す指定と、指定ごとに何に同意するかの文(#75)。"""
    add: list[str] = []
    agree: list[str] = []
    if "uncensored" in codes:
        add.append("--uncensored")
        agree.append(
            tr(
                "  --uncensored: 無検閲(成人向けを含みうる)のモデルを、自分の判断で落とす",
                "  --uncensored: you choose to download an uncensored model (may include adult content)",
            )
        )
    if "unknown_license" in codes:
        add.append("--any-license")
        agree.append(
            tr(
                "  --any-license: ライセンスの記載が無いファイルの条件を、上の確かめる場所で"
                "自分で確かめ、使う責任を自分で負う",
                "  --any-license: you check the terms of the files with no stated license"
                " yourself (pages above) and take responsibility for using them",
            )
        )
    return add, agree


def download_next_steps(
    plan, args: argparse.Namespace, asks: list[str] | None = None
) -> tuple[list[str], str | None, list[str]]:
    """止めたときの次の一手(#75)。(前の文, 1 行のコマンド(無ければ None), 後の文)。

    同意の指定(--uncensored・--any-license)だけで外せるなら、それを全部足したコマンドを 1 つ出し、
    指定ごとに何に同意するかを 1 行ずつ添える。指定では外せない理由(gated・空き・取得先・置き場所の外など)が
    あれば、コマンドは出さずにすることを書く。勝手に指定を足して落とすことはしない。
    asks: 端末なら質問で尋ねる同意の種類(--list で、止めずに尋ねる計画を見せたとき)。尋ねずに落とす
    (--yes・スクリプト)ときのコマンドを出す。"""
    if asks:
        add, agree = _consent_flags(asks)
        cmd = "  " + _download_command(args, plan.model_id, [*add, "--yes"])
        head = tr("尋ねずに落とすなら(スクリプトなど):", "To download without the question (scripts):")
        return [head], cmd, agree
    codes = list(dict.fromkeys(plan.blocked_codes))
    add, agree = _consent_flags(codes)
    fatal = [c for c in codes if c in ("gated", "unavailable", "no_files")]
    lever = [c for c in codes if c in ("disk", "budget", "conflict")]
    if not fatal and not lever:
        if not add:
            return [], None, []
        cmd = "  " + _download_command(args, plan.model_id, add)
        if args.yes:
            head = tr(
                "落とすなら(--yes を外せば y/N で尋ねます。または下の指定を足す):",
                "To download anyway (drop --yes to be asked y/N instead, or add the flags below):",
            )
        else:
            head = tr(
                "落とすなら(端末で実行すれば、指定の代わりに y/N で尋ねます):",
                "To download anyway (run it in a terminal to be asked y/N instead of adding flags):",
            )
        return [head], cmd, agree
    if fatal:
        # gated(ログインしない)・取得できないファイル・ファイル無しは、どの指定でも落とせない。指定を足せば
        # 落とせるような言い方はしない
        head = [
            tr(
                "このワークフローは genmodelfit では落とせません(どの指定でも変わりません):",
                "genmodelfit cannot download this workflow (no option changes this):",
            )
        ]
        if "gated" in fatal:
            pages = sorted({_repo_page(f.url) or f.check_url or "" for f in plan.files if f.gated} - {""})
            head.append(
                tr(
                    "  - gated: モデルのページで利用申請し、そのファイルは自分で落とす"
                    "(genmodelfit はログインしない)",
                    "  - gated: request access on the model page and download those files yourself"
                    " (genmodelfit never logs in)",
                )
                + (": " + ", ".join(pages) if pages else "")
            )
        if "unavailable" in fatal:
            head.append(
                tr(
                    "  - genmodelfit では落とせないファイル(理由は上)。自分で落とすなら:",
                    "  - files genmodelfit cannot download (reasons above). To get them yourself:",
                )
            )
            for f in plan.files:
                if f.status == "unavailable":
                    where = f.url or tr("一覧に URL がありません", "no URL in the list")
                    check = (
                        tr(f"(確かめる場所: {f.check_url})", f" (check: {f.check_url})")
                        if f.check_url
                        else ""
                    )
                    head.append(f"      {f.name}: {where}{check}")
        if "no_files" in fatal:
            head.append(tr("  - 一覧に落とすファイルがありません", "  - the list has no files to download"))
        alt = next((a for a in plan.alternatives if a.get("ok")), None)
        if alt:
            head.append(
                tr(
                    f"  - 同じモデルの別のワークフローにはこの問題がありません: {alt['id']}"
                    f"({alt['precision'] or '?'}、{alt['fit_level']})。見るなら:"
                    f" {_prog()} download {shell_quote(alt['id'])} --list",
                    f"  - another workflow of the same model has no such problem: {alt['id']}"
                    f" ({alt['precision'] or '?'}, {alt['fit_level']}). To look at it:"
                    f" {_prog()} download {shell_quote(alt['id'])} --list",
                )
            )
        return head, None, []
    todo = {
        "disk": tr(
            f"  - 空き: ディスクを {_gb(plan.disk_needed_bytes)} 空けるか、--comfyui-dir / --output-dir で"
            "別のドライブを選ぶ",
            f"  - disk: free up space (needs {_gb(plan.disk_needed_bytes)}) or choose another drive with"
            " --comfyui-dir / --output-dir",
        ),
        "budget": tr(
            "  - 上限: --budget を上げるか外す、または --precision で小さい精度を選ぶ",
            "  - budget: raise or drop --budget, or pick a smaller precision with --precision",
        ),
        "conflict": tr(
            "  - 同じ名前の別のファイル: 移すか、--output-dir で別の置き場所を選ぶ",
            "  - a different file with the same name: move it, or choose another place with --output-dir",
        ),
    }
    head = [tr("同意の指定では外れません。先に:", "The consent flags do not lift these. First:")]
    head += [todo[c] for c in lever]
    if add:
        head.append(
            tr(
                f"  - そのうえで、同意の指定 {' '.join(add)} も足す(下の意味に同意するなら)",
                f"  - then also add the consent flags {' '.join(add)} (if you agree to what they mean below)",
            )
        )
    return head, None, agree


def _emit_next_steps(plan, args: argparse.Namespace, asks: list[str] | None = None) -> None:
    head, command, tail = download_next_steps(plan, args, asks)
    if not (head or command or tail):
        return
    print()
    _emit(head)
    if command:
        print(command)  # コピーして貼れるよう折り返さない
    _emit(tail)


class _ProgressLine:
    """進み具合を stderr の 1 行に出す(端末なら 0.5 秒に 1 回まで書き換え、
    端末でなければ 10% ごとに 1 行)。"""

    def __init__(self) -> None:
        import time

        self._time = time.monotonic
        self._last = 0.0
        self._pct = -1
        self._tty = sys.stderr.isatty()

    def __call__(self, p) -> None:
        if p.message:
            self._end()
            print(f"  {p.name}: {p.message}", file=sys.stderr, flush=True)
            return
        if p.phase not in ("download", "check", "verify"):
            return
        now = self._time()
        pct = int(100 * p.done_bytes / p.total_bytes) if p.total_bytes else 0
        phase = {
            "download": tr("取得", "download"),
            "check": tr("確認", "check"),
            "verify": tr("検証", "verify"),
        }
        text = (
            f"[{p.file_index}/{p.file_count}] {phase[p.phase]} {p.name[-40:]}  "
            f"{p.file_bytes / GIB:.2f}/{(p.file_total or 0) / GIB:.2f} GB  "
            + tr(f"(全体 {pct}%)", f"(all {pct}%)")
        )
        if self._tty:
            if now - self._last >= 0.5 or p.phase != "download":
                sys.stderr.write("\r" + text[:99].ljust(99))
                sys.stderr.flush()
                self._last = now
        elif pct // 10 != self._pct // 10:
            print(text, file=sys.stderr, flush=True)
        self._pct = pct

    def _end(self) -> None:
        if self._tty and self._last:
            sys.stderr.write("\n")
            self._last = 0.0

    def close(self) -> None:
        self._end()


def _interactive() -> bool:
    """端末で y/N を尋ねられるか(標準入力と標準出力がどちらも端末)。試験で差し替える(#75)。

    Windows で入力を NUL から読ませると isatty が True になる。そのときも _ask が EOF を受けて「同意しない」に
    なるので、勝手に落とすことはない(tests/test_download.py の EOF の試験)。"""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError, OSError):
        return False


def _ask(lines: list[str]) -> bool:
    """同意を尋ねる(y / yes だけを同意とみなす)。入力が無い(パイプ・EOF)なら同意しない。"""
    _emit(lines)
    try:
        ans = input("[y/N] ")
    except EOFError:
        return False
    return ans.strip().lower() in ("y", "yes")


def cmd_download(args: argparse.Namespace) -> int:
    from .downloader import DownloadError, ModelNotFound, download, plan_download

    if args.json and not args.list and not args.yes:
        raise CliError(
            tr(
                "--json で落とすときは --yes が要ります(同意を尋ねられないため。先に --list で確かめる)",
                "--json needs --yes to download (consent cannot be asked; check with --list first)",
            )
        )
    # ComfyUI のフォルダ: --comfyui-dir > --output-dir > 環境変数 GENMODELFIT_COMFYUI_DIR > 既定の場所(#75)
    comfyui_dir, comfyui_from = args.comfyui_dir, "flag" if args.comfyui_dir else None
    env_dir = os.environ.get(COMFYUI_DIR_ENV, "").strip()
    if not comfyui_dir and not args.output_dir:
        comfyui_dir, comfyui_from = (env_dir, "env") if env_dir else (None, "default")
    s, db = _system(args), _db(args)
    try:
        plan = plan_download(
            args.model,
            precision=args.precision,
            budget_gb=args.budget,
            output_dir=args.output_dir,
            comfyui_dir=comfyui_dir,
            content=args.content,
            include_unknown_license=args.include_unknown_license,
            system_info=s,
            db=db,
            policy=_policy(args),
        )
    except ModelNotFound as e:
        err = CliError(str(e).replace('content="all"', "--uncensored"))  # ライブラリの言い方を CLI の指定に
        err.candidates = e.candidates or None  # type: ignore[attr-defined]
        raise err from e
    except DownloadError as e:
        if comfyui_from == "env":
            src = tr(
                f"(環境変数 {COMFYUI_DIR_ENV} の値)", f" (from the environment variable {COMFYUI_DIR_ENV})"
            )
            raise CliError(str(e) + src) from e
        raise CliError(str(e)) from e

    def fail(e: Exception, code: int = 1) -> int:
        if args.json:
            _out(_envelope("download", s, db, listed=False, plan=plan.to_dict(), result=None, error=str(e)))
        else:
            print(tr(f"エラー: {e}", f"error: {e}"), file=sys.stderr)
        return code

    # 無検閲・ライセンス不明は、端末で尋ねられるなら止めずに y/N の質問で同意を取る(#75)。--yes・--json・
    # 端末でないときは今までどおり指定(--uncensored・--any-license)が要る。ライブラリの約束は変えない
    from .downloader import CONSENT_CODES

    asks = [c for c in dict.fromkeys(plan.blocked_codes) if c in CONSENT_CODES]
    # 本当に実行したら尋ねるとき(端末で --yes・--json が無い)だけ。--list も同じ条件で、尋ねることを見せる
    can_ask = not args.json and not args.yes and _interactive()
    if asks and can_ask and set(plan.blocked_codes) <= set(CONSENT_CODES):
        strict = plan
        try:
            plan = plan_download(
                plan.model_id,  # 同じワークフロー(ID で指定し直す。名前の探し方は変えない)
                budget_gb=args.budget,
                output_dir=args.output_dir,
                comfyui_dir=comfyui_dir,
                content="all" if "uncensored" in asks else args.content,
                include_unknown_license=True,
                verify_existing=False,  # sha256 は 1 回目で計った。ファイルの状態は 1 回目のものを使う
                system_info=s,
                db=db,
                policy=_policy(args),
            )
        except DownloadError as e:  # 1 回目と同じ条件なので起きないはず
            raise CliError(str(e)) from e
        if [f.name for f in plan.files] != [f.name for f in strict.files]:  # 念のため(同じ ID なので同じはず)
            raise CliError(tr("計画を作り直せませんでした", "could not re-plan the download"))
        plan.files = strict.files
        plan.chosen, plan.alternatives = strict.chosen, strict.alternatives  # 選んだ理由は 1 回目のもの
    else:
        asks = []

    if args.list:
        if args.json:
            _out(_envelope("download", s, db, listed=True, plan=plan.to_dict(), result=None))
        else:
            _emit(format_plan(plan, comfyui_from=comfyui_from, asks=asks))
            _emit_next_steps(plan, args, asks)
        return 0
    if plan.blocked:
        if args.json:
            return fail(
                DownloadError(tr("落とせません: ", "cannot download: ") + " / ".join(blocked_texts(plan)))
            )
        # 理由は計画の中に 1 回だけ出し、最後に次の一手を出す(同じ文を error: で繰り返さない。#75)
        _emit(format_plan(plan, comfyui_from=comfyui_from))
        _emit_next_steps(plan, args)
        print(
            tr("エラー: 落とせません(上を見てください)", "error: cannot download (see above)"),
            file=sys.stderr,
        )
        return 1
    if not args.yes:
        try:
            agreed = _ask(format_plan(plan, consent=True, comfyui_from=comfyui_from, asks=asks))
        except KeyboardInterrupt:  # 質問の途中の Ctrl-C: 何も落とさず 130
            print(file=sys.stderr)
            print(
                tr("中断しました(何も落としていません)", "Interrupted (nothing downloaded)"), file=sys.stderr
            )
            return 130
        if not agreed:
            print(tr("落としませんでした(同意がありません)", "Not downloaded (no consent)"), file=sys.stderr)
            return 1
    elif not args.json:
        _emit(format_plan(plan, comfyui_from=comfyui_from))
    show = None if args.json else _ProgressLine()
    try:
        result = download(plan, progress=show, yes=True)
    except KeyboardInterrupt:
        if show:
            show.close()
        return fail(
            DownloadError(
                tr(
                    "中断しました。途中までのファイル(.part)は残してあり、"
                    "同じコマンドでもう一度実行すると続きから落とします",
                    "Interrupted. The partial files (.part) are kept; run the same command again to resume",
                )
            ),
            130,
        )
    except DownloadError as e:
        if show:
            show.close()
        return fail(e)
    if show:
        show.close()
    if args.json:
        _out(_envelope("download", s, db, listed=False, plan=plan.to_dict(), result=result.to_dict()))
        return 0
    print(
        tr(
            f"落としました: {len(result.downloaded)} 件(続きから {len(result.resumed)} 件)、"
            f"既にあった {len(result.skipped)} 件、{result.bytes_downloaded / GIB:.2f} GB、"
            f"{result.seconds:.0f} 秒。"
            "sha256 はすべて一覧と一致",
            f"Downloaded: {len(result.downloaded)} files ({len(result.resumed)} resumed),"
            f" {len(result.skipped)} already there, {result.bytes_downloaded / GIB:.2f} GB,"
            f" {result.seconds:.0f} s. Every sha256 matches the list",
        )
    )
    print(tr(f"置き場所: {plan.root}", f"Location: {plan.root}"))
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
            raise CliError(tr("--license-allowlist が空です", "--license-allowlist is empty"))
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


# --- 引数 ---------------------------------------------------------------------------------------


def _positive_int(v: str) -> int:
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(tr(f"整数ではありません: {v}", f"not an integer: {v}")) from None
    if n < 1:
        raise argparse.ArgumentTypeError(tr(f"1 以上を指定してください: {v}", f"must be 1 or more: {v}"))
    return n


def _head_age(v: str) -> int | None:
    if v.strip().lower() == "none":
        return None
    try:
        n = int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(
            tr(f"日数(整数)か none を指定してください: {v}", f"give a number of days (integer) or none: {v}")
        ) from None
    if n < 0:
        raise argparse.ArgumentTypeError(tr(f"0 以上を指定してください: {v}", f"must be 0 or more: {v}"))
    return n


_JSON_FIELDS = """  JSON (--json): { genmodelfit, command, system: {...}, database: {...}, filters: {...},
  matched, hidden: {reason: count}, groups: [{modality, matched, shown}],
  models: [{ id, name, model_name,
  modality, tasks, use, content, source, runtime, languages, description, fit_level, fit_label,
  run_mode, run_mode_label, score,
  score_components { fit, headroom, readiness, basis, popularity }, memory_kind,
  memory_required_gb, memory_available_gb, utilization_pct, gpu_memory_required_gb,
  ram_required_gb, disk_size_gb, total_size_gb, license, licenses, auto,
  held_reasons, estimate_confidence, notes,
  files_count, precision, group_key, variants: [{precision, workflow_id, ...}],
  workflows: [{id, name, tasks, use, precision, fit_level, ...}], estimate, gate }] }"""
_JSON_KEYS = _JSON_FIELDS  # JSON の鍵の並び(両方の言語で同じ)
_JSON_FIELDS = (
    _JSON_KEYS
    + """
  One item per model (same weights family + size); --all-workflows gives one per workflow.
  Details: docs/schema.md"""
)

_DETECT_HELP = (
    "  None. The GPU and RAM are detected; --vram / --ram override them. GPU: NVIDIA via nvidia-smi;\n"
    "  otherwise AMD / Intel / Apple via WMI (Windows), lspci + sysfs (Linux) or system_profiler (macOS).\n"
)

_COMMON_HELP = (
    """PRECONDITIONS:
"""
    + _DETECT_HELP
    + """  The bundled model list is used unless `genmodelfit update` stored a newer one (--no-cache).

SIDE EFFECTS:
  None - read-only. No network access.

EXIT CODES:
  0  Success{zero}
  1  Error{one} (e.g. a broken --db file; with --json the error is in "error")
  2  Usage error (unknown option or value)
"""
)

_HELP_EN = {
    "recommend": """Recommend generation models for this PC, best first.

Without -m / --task: one section per type (image, video, music, sfx, tts, 3d), top -n
(default 3) each. With -m or --task: one table (default 10).
One row per model; the precision and workflow that fit best are chosen (--all-workflows: all).
Order: models that run, then Perfect/Good, then Marginal, then Score (0-100), then id
(--json puts auto-recommendable ones, gate: auto, before the levels). Uncensored models and
utility tools are hidden unless asked. The default output is a table (llmfit uses JSON by
default); add --json for apps.

"""
    + _COMMON_HELP.format(zero="", one="")
    + """
AGENT USAGE:
  genmodelfit recommend --json
  genmodelfit recommend -m image --task edit -n 5 --min-fit good --json
  genmodelfit recommend -m video,music --auto-only --comfyui-version 0.40.0 --json
  genmodelfit recommend --license apache-2.0,mit --vram 8 --ram 32 --json

"""
    + _JSON_FIELDS,
    "fit": """Judge every model for this PC (too tight ones included), in the same order as recommend.

"""
    + _COMMON_HELP.format(zero="", one="")
    + """
AGENT USAGE:
  genmodelfit fit --json
  genmodelfit fit -m video --task i2v --min-fit good -n 20
  genmodelfit fit --uncensored --all-workflows --vram 12 --ram 32 --json

"""
    + _JSON_FIELDS,
    "search": """Search models by a part of the id or name (case-insensitive) and judge them for this PC.
--content applies (default sfw). With no match the list is empty and the exit code is 0.

"""
    + _COMMON_HELP.format(zero=' (also when nothing matches: empty models / "no match")', one=": blank query")
    + """
AGENT USAGE:
  genmodelfit search flux --json
  genmodelfit search wan -m video --min-fit good
  genmodelfit search sdxl --uncensored --json

"""
    + _JSON_FIELDS.replace("filters: {...}", "filters: {query, ...}"),
    "types": """List the big types (image, video, music, sfx, tts, 3d) and the uses inside each type,
with counts. The use names are what --task accepts.

"""
    + _COMMON_HELP.format(zero="", one="").replace(
        _DETECT_HELP,
        "  None. The PC is not judged.\n",
    )
    + """
AGENT USAGE:
  genmodelfit types
  genmodelfit types --json

  JSON (--json): { genmodelfit, command, database, filters: {content}, types: [{ modality,
  label, label_ja, models, workflows, uses: {use: workflows} }], uses: [{ use, label,
  label_ja, tasks }], aliases }. Details: docs/schema.md""",
    "info": """Show one model: files, licenses, estimate, limits on this PC, the gate checks, and the
other workflows that use the same model.
An exact id always matches (uncensored included). Otherwise a part of the id or name is
searched within --content; several matches are an error that lists the candidates.

"""
    + _COMMON_HELP.format(zero="", one=": no such model, several matches (candidates listed), or")
    + """
AGENT USAGE:
  genmodelfit info image_z_image_turbo_int8 --json
  genmodelfit info sdxlturbo --vram 8 --ram 32

  JSON (--json): { genmodelfit, command, system, database, model: ModelEntry, estimate,
  gate, measurements, group_key, same_model: [{id, name, tasks, use, precision, fit_level,
  memory_required_gb, disk_size_gb, auto, matched}] } / on error: { error, candidates? }.
  Details: docs/schema.md""",
    "system": """Show this PC (GPU, RAM, free disk) and whether the model list in use is still fresh
(Data: source, HEAD check dates, expired / valid until, auto-recommendable count).

"""
    + _COMMON_HELP.format(
        zero=" (also when the list has expired: see Data / database.status.expired)", one=""
    )
    + """
AGENT USAGE:
  genmodelfit system --json
  genmodelfit system --comfyui-version 0.40.0 --json

  JSON (--json): { genmodelfit, command, system, database: { origin, generated_at, models,
  measurements, status: { source, generated_at, heads_checked_oldest, heads_checked_newest,
  max_head_age_days, expired, expires_at, models, auto_count, auto_count_if_fresh,
  checked_at } } }. Details: docs/schema.md""",
    "update": """Rebuild the model list from upstream, redo the HEAD checks (--heads-only), re-apply only the
app fields and the hand-added models (--fields-only), or take in the signed list published by this
project (--from-release).

PRECONDITIONS:
  Network access.

SIDE EFFECTS:
  Network: GET / HEAD only. update reads the ComfyUI workflow templates and releases (GitHub),
  the Hugging Face Hub API and the model download URLs (HEAD, no download of weights);
  --from-release downloads only db.json and db.json.sig from this project's GitHub release.
  Nothing about this PC is sent (User-Agent: genmodelfit/<version>; no ~/.netrc credentials).
  Writes the list to the cache (or --output). The cache is left untouched on failure.

EXIT CODES:
  0  Success
  1  Error (network, bad signature, older list, ...; with --json the error is in "error")
  2  Usage error
""",
    "diff": """Compare two model lists (db.json) and show what changed for users: added / removed models, and
for each changed model (name (type) — id) added / removed / changed files (with their license and
source), license (the permissive side is marked with !), newly uncensored, runtime / use,
auto-recommendation, prompt / speech languages, size, precision, review status, and the names of
any other changed fields ("other changes"). Changed models are in three sections, each cut by
--limit separately: files / license / uncensored / runtime and use first, then auto-recommendation
changes (ids always listed), then license provenance, languages, size, review marks and the rest;
within a section the most important change comes first, so --limit never hides it; the same
low-importance change on many models (e.g. a review mark added to all) is one line with a count.
Every changed field is shown somewhere (or counted for --all). The raw JSON is compared, so
unknown keys count (as in merge-check).

With no arguments, compares the bundled list with the list fetched by `update` (the cache).
With one argument, compares the bundled list with that file. With two, compares the two files.
--all adds the maintenance view (the content of the weekly PR): all failed HEAD checks (not only
new ones), 401 repos, the changed numbers in `sources`, models where only popularity numbers,
dates or HEAD results changed, and the ids of collapsed lines; it has no line limit unless -n is
given (default -n 20 per section otherwise).

PRECONDITIONS:
  No arguments: a fetched list in the cache (run `genmodelfit update --from-release` first; without
  it, a short how-to is printed and the exit code is 0). Otherwise one or two db.json files.
  A warning is printed when the new list is older (generated_at) than the old one.

SIDE EFFECTS:
  None - read-only. No network access.

EXIT CODES:
  0  Success (also when there are differences, and when there is no fetched list yet)
  1  Error (a file cannot be read or has another schema; with --json the error is in "error")
  2  Usage error

AGENT USAGE:
  genmodelfit diff                          # bundled list vs the fetched list
  genmodelfit diff new.json                 # bundled list vs new.json
  genmodelfit diff old.json new.json --json
  genmodelfit diff old.json new.json --all  # maintenance view

  The auto-recommendation is compared with one policy for both lists: the ComfyUI version recorded
  in the new list (or --comfyui-version) and no HEAD age limit (HEAD failures still count).
  JSON (--json): { genmodelfit, command, old, new, old_label, new_label, policy, summary, added,
  removed, license_changes, auto_changes, size_changes, precision_changes, newly_uncensored,
  head_failures, unauthorized, partial_counts, stats, classification, field_changes,
  model_changes, top_changes, warnings }. With no fetched list: the same keys, empty, plus
  "note". Details: docs/schema.md""",
    "download": """Download the files of a model for this PC (all files of one workflow), checking sha256.

MODEL is a model name (as in the tables), a workflow id, or a part of either (like info).
For a model name, the workflow and precision are chosen like recommend: the highest precision
that runs on this PC without offloading (--precision picks one; --budget caps the download).
Files go to <--comfyui-dir>/models/<folder>/<file> (the folder comes from the template:
diffusion_models, text_encoders, vae, loras, checkpoints, ...), or the same layout under
--output-dir (default: the user data folder of genmodelfit, "models").
The environment variable GENMODELFIT_COMFYUI_DIR is the default for --comfyui-dir
(--comfyui-dir and --output-dir win over it). Files whose ComfyUI folder is unknown go to
other/, which ComfyUI does not read: move them before use.
Before downloading, the files, each license and its URL (or the page to check when no license
is stated), the total size and the free disk are shown and you are asked to agree (--yes skips
the question).
Uncensored models and files with an unknown license need your consent: in a terminal the
question names them (y/N); with --yes, --json or without a terminal pass --uncensored and/or
--any-license (old forms: --content all, --include-unknown-license; consent flags must be
written in full). When only these are missing, the plan ends with one command that adds them
(it never adds them by itself).
Refused (no option lifts these): gated files (no login is used), too little disk (needs the
download + 1 GB), names that would land outside the folder, hosts other than Hugging Face /
GitHub.

PRECONDITIONS:
  Network access to huggingface.co (or github.com) for the download. --list needs none.

SIDE EFFECTS:
  Network: GET of the listed URLs only (https; every redirect must stay on Hugging Face /
  GitHub hosts). Nothing about this PC is sent (User-Agent: genmodelfit/<version>; no
  ~/.netrc credentials, no tokens). Writes only inside the destination folder: <file>.part
  while downloading (kept on Ctrl-C or failure; the next run resumes it with HTTP Range),
  renamed to <file> after the sha256 matches. A sha256 mismatch deletes the file. Existing
  files whose sha256 matches are skipped; other existing files are never overwritten.
  --list: no network access and nothing is written.

EXIT CODES:
  0  Success (also --list, even when the plan is blocked: see "blocked")
  1  Error: no such model / several matches, refused (see above), declined, network,
     sha256 mismatch (with --json the error is in "error")
  2  Usage error
  130  Interrupted (Ctrl-C); run again to resume

AGENT USAGE:
  genmodelfit download "Z-Image-Turbo" --list --json
  genmodelfit download "Z-Image-Turbo" --comfyui-dir C:/ComfyUI --yes --json
  genmodelfit download image_z_image_turbo_int8 --output-dir D:/models --budget 15 --yes

  JSON (--json): { genmodelfit, command, system, database, listed, plan: { model_id, name,
  model_name, precision, fit_level, layout, root, files: [{ name, relpath, dest, url, sha256,
  size_bytes, directory, directory_basis, license, license_url, check_url, license_allowed,
  gated, status, part_bytes, remaining_bytes, reason }], total_bytes, download_bytes,
  disk_free_bytes, disk_needed_bytes, blocked, blocked_codes, warnings, alternatives, ok },
  result: { downloaded, resumed,
  restarted, skipped, bytes_downloaded, seconds, files } | null, error? }.
  Details: docs/schema.md""",
}


# --- 日本語の説明(#76。英語の _HELP_EN と同じ内容を日本語だけで書く。JSON の鍵・コマンドは同じ) ---

_DETECT_HELP_JA = (
    "  なし。GPU とメモリは調べて決める(--vram / --ram で置き換える)。GPU は NVIDIA なら nvidia-smi、\n"
    "  ほかの AMD / Intel / Apple は WMI(Windows)・lspci と sysfs(Linux)・system_profiler(macOS)で調べる。\n"
)

_COMMON_HELP_JA = (
    "前提:\n"
    + _DETECT_HELP_JA
    + """  一覧は同梱のものを使う(`genmodelfit update` で新しい一覧を取ってあればそちら。--no-cache で同梱)。

副作用:
  なし(読むだけ。通信しない)。

終了コード:
  0  成功{zero}
  1  失敗{one}(--json なら誤りは "error" に入る)
  2  使い方の誤り(知らないオプション・値)
"""
)
_ONE_DB_JA = ": 壊れた --db のファイルなど"

_JSON_FIELDS_JA = (
    _JSON_KEYS
    + """
  1 項目に 1 モデル(同じ重みの系統 + 大きさ)。--all-workflows ならワークフローごとに 1 項目。
  詳しくは docs/schema.md"""
)

_HELP_JA = {
    "recommend": """この PC に合う生成モデルを、おすすめ順に出す。

-m も --task も無ければ、種類(image, video, music, sfx, tts, 3d)ごとに上位 -n 件(既定 3 件)。
-m か --task があれば 1 つの表(既定 10 件)。
同じモデルは 1 行にまとめ、一番合う精度とワークフローを選ぶ(--all-workflows で全部)。
並べ方: 動くもの → Perfect/Good → Marginal → Score(0〜100)の高い順 → ID
(--json は自動でおすすめに出せるもの(gate: auto)を段階より先にする)。
無検閲のモデルと補助ツールは、指定しなければ出さない。既定の出力は表(llmfit の既定は JSON)。
アプリは --json を付ける。

"""
    + _COMMON_HELP_JA.format(zero="", one=_ONE_DB_JA)
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit recommend --json
  genmodelfit recommend -m image --task edit -n 5 --min-fit good --json
  genmodelfit recommend -m video,music --auto-only --comfyui-version 0.40.0 --json
  genmodelfit recommend --license apache-2.0,mit --vram 8 --ram 32 --json

"""
    + _JSON_FIELDS_JA,
    "fit": """すべてのモデルをこの PC で判定して表にする(動かないものも。並べ方は recommend と同じ)。

"""
    + _COMMON_HELP_JA.format(zero="", one=_ONE_DB_JA)
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit fit --json
  genmodelfit fit -m video --task i2v --min-fit good -n 20
  genmodelfit fit --uncensored --all-workflows --vram 12 --ram 32 --json

"""
    + _JSON_FIELDS_JA,
    "search": """ID・名前の一部で探し(大文字小文字は区別しない)、この PC での判定を出す。
--content が効く(既定 sfw)。合うものが無ければ空の一覧で、終了コードは 0。

"""
    + _COMMON_HELP_JA.format(
        zero="(合うものが無いときも。models は空、表では「(該当なし)」)",
        one=": 探す語が空、壊れた --db のファイルなど",
    )
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit search flux --json
  genmodelfit search wan -m video --min-fit good
  genmodelfit search sdxl --uncensored --json

"""
    + _JSON_FIELDS_JA.replace("filters: {...}", "filters: {query, ...}"),
    "types": """大きな種類(image, video, music, sfx, tts, 3d)と、その中の用途と数を出す。
用途の名前は --task に使う。

"""
    + _COMMON_HELP_JA.format(zero="", one=_ONE_DB_JA).replace(_DETECT_HELP_JA, "  なし。PC は判定しない。\n")
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit types
  genmodelfit types --json

  JSON (--json): { genmodelfit, command, database, filters: {content}, types: [{ modality,
  label, label_ja, models, workflows, uses: {use: workflows} }], uses: [{ use, label,
  label_ja, tasks }], aliases }
  詳しくは docs/schema.md""",
    "info": """1 つのモデルの詳細(ファイル・ライセンス・見積もり・この PC での限界・自動のおすすめの条件・
同じモデルのほかのワークフロー)を出す。
ID がそのまま合えば必ずそのモデル(無検閲も)。合わなければ ID・名前の一部を --content の中で探す。
2 件以上合えば失敗にして候補を並べる。

"""
    + _COMMON_HELP_JA.format(
        zero="", one=": モデルが無い、2 件以上合う(候補を並べる)、壊れた --db のファイルなど"
    )
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit info image_z_image_turbo_int8 --json
  genmodelfit info sdxlturbo --vram 8 --ram 32

  JSON (--json): { genmodelfit, command, system, database, model: ModelEntry, estimate,
  gate, measurements, group_key, same_model: [{id, name, tasks, use, precision, fit_level,
  memory_required_gb, disk_size_gb, auto, matched}] }
  失敗のときは { error, candidates? }。詳しくは docs/schema.md""",
    "system": """この PC(GPU・メモリ・ディスクの空き)と、使っている一覧がまだ新しいか(一覧の欄: 出どころ・
HEAD の確認の日付・期限切れか・いつまで有効か・自動のおすすめの数)を出す。

"""
    + _COMMON_HELP_JA.format(
        zero="(一覧が期限切れのときも。一覧の欄・database.status.expired を見る)", one=_ONE_DB_JA
    )
    + """
使い方の例(エージェント・アプリ向け):
  genmodelfit system --json
  genmodelfit system --comfyui-version 0.40.0 --json

  JSON (--json): { genmodelfit, command, system, database: { origin, generated_at, models,
  measurements, status: { source, generated_at, heads_checked_oldest, heads_checked_newest,
  max_head_age_days, expired, expires_at, models, auto_count, auto_count_if_fresh,
  checked_at } } }
  詳しくは docs/schema.md""",
    "update": """一覧を上流から作り直す。HEAD の確認だけをやり直す(--heads-only)、アプリ向けの項目と手で足した
モデルだけを当て直す(--fields-only)、このプロジェクトが配る署名つきの一覧を取り込む(--from-release)
こともできる。

前提:
  通信できること。

副作用:
  通信: GET / HEAD だけ。update は ComfyUI のワークフローのテンプレートとリリース(GitHub)、
  Hugging Face Hub の API、モデルのダウンロードの URL(HEAD だけ。重みは落とさない)を読む。
  --from-release は、このプロジェクトの GitHub のリリースから db.json と db.json.sig だけを落とす。
  この PC のことは何も送らない(User-Agent: genmodelfit/<版>。~/.netrc の資格情報は送らない)。
  一覧をキャッシュ(か --output)に書く。失敗したらキャッシュは変えない。

終了コード:
  0  成功
  1  失敗(通信・署名が合わない・前より古い一覧など。--json なら誤りは "error" に入る)
  2  使い方の誤り
""",
    "diff": """2 つの一覧(db.json)を比べ、利用者に関係のある違いを出す: 増えた・消えたモデルと、変わったモデル
(名前 (種類) — ID)ごとの、増えた・消えた・変わったファイル(ライセンスと出どころつき)、ライセンス
(寛容な側に ! を付ける)、新しく無検閲になったもの、動かし方・用途、自動のおすすめ、指示・読み上げの
言語、大きさ、精度、確認の状態と、ほかに変わった項目の名前(「その他の変化」)。
変わったモデルは 3 つの節に分け、節ごとに --limit で切る: ファイル・ライセンス・無検閲・動かし方と
用途が先、次に自動のおすすめの変化(ID は必ず出す)、最後にライセンスの出どころ・言語・大きさ・
確認の印など。節の中では大事な変化を先に出すので、--limit で隠れない。大事さの低い同じ変化が多くの
モデルにあれば(全部に確認の印が付いたなど)、件数つきの 1 行にまとめる。
変わった項目はどこかに必ず出る(--all では数える)。JSON をそのまま比べるので、知らない鍵の変化も
数える(merge-check と同じ)。

引数なしなら、同梱の一覧と `update` で取ってきた一覧(キャッシュ)を比べる。1 つなら同梱の一覧と
そのファイル、2 つならその 2 つを比べる。
--all は保守用の内容(毎週の PR の中身)を足す: 新しいものに限らずすべての HEAD の失敗、401 の
リポジトリ、`sources` の変わった数、人気の数・日付・HEAD の結果だけが変わったモデル、まとめた行の
ID。-n を付けなければ行の上限は無い(--all でなければ節ごとに既定 -n 20)。

前提:
  引数なし: キャッシュに取ってきた一覧があること(先に `genmodelfit update --from-release`。無ければ
  短い案内を出して終了コード 0)。ほかは db.json のファイルを 1 つか 2 つ。
  新しい一覧(generated_at)が前の一覧より古ければ注意を出す。

副作用:
  なし(読むだけ。通信しない)。

終了コード:
  0  成功(違いがあるときも、取ってきた一覧がまだ無いときも)
  1  失敗(ファイルが読めない・形が違う。--json なら誤りは "error" に入る)
  2  使い方の誤り

使い方の例(エージェント・アプリ向け):
  genmodelfit diff                          # 同梱の一覧 ↔ 取ってきた一覧
  genmodelfit diff new.json                 # 同梱の一覧 ↔ new.json
  genmodelfit diff old.json new.json --json
  genmodelfit diff old.json new.json --all  # 保守用の内容

  自動のおすすめは、両方の一覧を 1 つの条件で比べる: 新しい一覧が記録した ComfyUI の版
  (か --comfyui-version)で、HEAD の期限は無し(HEAD の失敗は数える)。
  JSON (--json): { genmodelfit, command, old, new, old_label, new_label, policy, summary, added,
  removed, license_changes, auto_changes, size_changes, precision_changes, newly_uncensored,
  head_failures, unauthorized, partial_counts, stats, classification, field_changes,
  model_changes, top_changes, warnings }
  取ってきた一覧が無いときは、同じ鍵が空で "note" が付く。詳しくは docs/schema.md""",
    "download": """モデルのファイルをこの PC 向けに落とす(1 つのワークフローの全部のファイル。
sha256 で確かめる)。

MODEL はモデルの名前(表の Model)・ワークフローの ID・そのどちらかの一部(info と同じ)。
モデルの名前なら、ワークフローと精度は recommend と同じ選び方: この PC で逃がさずに動く中で一番高い
精度(--precision で選ぶ。--budget で落とす量の上限を決める)。
ファイルは <--comfyui-dir>/models/<folder>/<file> に置く(folder はテンプレートで決まる:
diffusion_models・text_encoders・vae・loras・checkpoints など)。--output-dir なら同じ並びでその下
(既定は genmodelfit の利用者データのフォルダの「models」)。
環境変数 GENMODELFIT_COMFYUI_DIR は --comfyui-dir の既定(--comfyui-dir と --output-dir が優先)。
ComfyUI のどのフォルダか分からないファイルは other/ に置く。ComfyUI は読まないので、使う前に移す。
落とす前に、ファイル・それぞれのライセンスと URL(記載が無ければ確かめる場所)・合計の大きさ・
ディスクの空きを出し、同意を尋ねる(--yes で尋ねない)。
無検閲のモデルとライセンスの分からないファイルには同意が要る: 端末ならそれを書いて y/N で尋ねる。
--yes・--json・端末でないときは --uncensored・--any-license を指定する(前からの書き方:
--content all、--include-unknown-license。同意の指定は略さずに書く)。足りないのがこれだけなら、
計画の最後にそれを足したコマンドを 1 つ出す(勝手には足さない)。
断るもの(どのオプションでも変わらない): gated のファイル(ログインしない)、ディスクの空きが
足りない(落とす量 + 1 GB が要る)、置き場所の外に出る名前、Hugging Face / GitHub 以外のホスト。

前提:
  落とすには huggingface.co(か github.com)と通信できること。--list は通信しない。

副作用:
  通信: 一覧の URL の GET だけ(https。転送先もすべて Hugging Face / GitHub のホストに限る)。
  この PC のことは何も送らない(User-Agent: genmodelfit/<版>。~/.netrc の資格情報もトークンも
  使わない)。書くのは置き場所のフォルダの中だけ: 落としている間は <file>.part(Ctrl-C や失敗では
  残し、次の実行で HTTP Range で続きから落とす)、sha256 が合えば <file> に名前を変える。sha256 が
  合わなければ消す。sha256 の合う既にあるファイルは飛ばし、ほかの既にあるファイルは上書きしない。
  --list: 通信せず、何も書かない。

終了コード:
  0  成功(--list も。計画が止まっていても: "blocked" を見る)
  1  失敗: モデルが無い・2 件以上合う、断った(上を見る)、同意が無い、通信、sha256 が合わない
     (--json なら誤りは "error" に入る)
  2  使い方の誤り
  130  中断(Ctrl-C)。もう一度実行すれば続きから落とす

使い方の例(エージェント・アプリ向け):
  genmodelfit download "Z-Image-Turbo" --list --json
  genmodelfit download "Z-Image-Turbo" --comfyui-dir C:/ComfyUI --yes --json
  genmodelfit download image_z_image_turbo_int8 --output-dir D:/models --budget 15 --yes

  JSON (--json): { genmodelfit, command, system, database, listed, plan: { model_id, name,
  model_name, precision, fit_level, layout, root, files: [{ name, relpath, dest, url, sha256,
  size_bytes, directory, directory_basis, license, license_url, check_url, license_allowed,
  gated, status, part_bytes, remaining_bytes, reason }], total_bytes, download_bytes,
  disk_free_bytes, disk_needed_bytes, blocked, blocked_codes, warnings, alternatives, ok },
  result: { downloaded, resumed,
  restarted, skipped, bytes_downloaded, seconds, files } | null, error? }
  詳しくは docs/schema.md""",
}


def _help(name: str) -> str:
    """サブコマンドの説明(今の言語。1 つの言語だけで書く。#76)。"""
    return (_HELP_JA if i18n.get_language() == "ja" else _HELP_EN)[name]


class _HelpFormatter(argparse.RawDescriptionHelpFormatter):
    """オプションの説明を全角 2 桁で数えて幅に折り返す。英数字の並び(--include-unknown-license など)の
    途中では切らない(table.wrap_text。#76)。説明(description)・例(epilog)は書いた形のまま。"""

    def _split_lines(self, text: str, width: int) -> list[str]:
        from .table import wrap_text

        return wrap_text(" ".join(text.split()), width, space_room=False)

    def _format_usage(self, usage: Any, actions: Any, groups: Any, prefix: Any) -> str:
        # argparse は続きの行の字下げを文字数で数える。「使い方: 」は表示の幅が文字数より広いので、
        # その差だけ続きの行を右へずらす
        from rich.cells import cell_len

        text = super()._format_usage(usage, actions, groups, prefix)
        head = prefix if prefix is not None else argparse._("usage: ")  # type: ignore[attr-defined]
        extra = cell_len(head) - len(head)
        if extra <= 0:
            return text
        lines = text.split("\n")
        return "\n".join([lines[0], *(" " * extra + ln if ln.strip() else ln for ln in lines[1:])])


class _Parser(argparse.ArgumentParser):
    """説明の折り返しを _HelpFormatter にする(サブコマンドも同じ型で作られる)。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("formatter_class", _HelpFormatter)
        super().__init__(*args, **kwargs)


# argparse の決まった文(usage:・options・-h の説明・誤り)の日本語(#76)。キーは argparse の原文
# (Python 3.10〜3.13)。知らない文は英語のまま
_ARGPARSE_JA = {
    "usage: ": "使い方: ",
    "positional arguments": "引数",
    "options": "オプション",
    "optional arguments": "オプション",
    "show this help message and exit": "この説明を出して終わる",
    "show program's version number and exit": "版を出して終わる",
    "%(prog)s: error: %(message)s\n": "%(prog)s: エラー: %(message)s\n",
    "unrecognized arguments: %s": "知らない引数です: %s",
    "the following arguments are required: %s": "次の引数が要ります: %s",
    "argument %(argument_name)s: %(message)s": "引数 %(argument_name)s: %(message)s",
    "invalid choice: %(value)r (choose from %(choices)s)": (
        "選べない値です: %(value)r(選べるのは %(choices)s)"
    ),
    "invalid %(type)s value: %(value)r": "値が正しくありません(%(type)s): %(value)r",
    "expected one argument": "値を 1 つ指定してください",
    "expected at least one argument": "値を 1 つ以上指定してください",
    "expected at most one argument": "値は 1 つまでです",
    "ambiguous option: %(option)s could match %(matches)s": (
        "略した書き方が紛らわしいオプションです: %(option)s(%(matches)s のどれか)"
    ),
    "not allowed with argument %s": "引数 %s とは同時に指定できません",
    "one of the arguments %s is required": "引数 %s のどれかを指定してください",
    "ignored explicit argument %r": "値 %r は使えません",
}
_ARGPARSE_JA_N = {"expected %s argument": "値を %s 個指定してください"}


_ARGPARSE_LOCK = threading.Lock()
_ARGPARSE_DEPTH = 0
_ARGPARSE_ORIG: tuple[Any, Any] | None = None


@contextlib.contextmanager
def _argparse_language() -> Any:
    """argparse の決まった文を今の言語にする(main の間だけ)。

    差し替えは argparse モジュールの全体に効く。main を同時に複数のスレッドで呼んでも戻せるように、
    入った数を数え、最後に出る main が元に戻す(途中で戻すと、ほかの main の文が英語になる)。
    置き換えた関数は呼ぶたびに言語を見る。言語が en なら元の関数と同じ文を返すので影響はない。
    main の外のスレッドの argparse も、main の間は今の言語で出る(日本語のとき)。"""
    global _ARGPARSE_DEPTH, _ARGPARSE_ORIG
    with _ARGPARSE_LOCK:
        if _ARGPARSE_DEPTH == 0:
            _ARGPARSE_ORIG = (argparse._, argparse.ngettext)  # type: ignore[attr-defined]
            gettext_, ngettext_ = _ARGPARSE_ORIG

            def ja_gettext(msg: str) -> str:
                return _ARGPARSE_JA.get(msg, msg) if i18n.get_language() == "ja" else gettext_(msg)

            def ja_ngettext(singular: str, plural: str, n: int) -> str:
                if i18n.get_language() == "ja" and singular in _ARGPARSE_JA_N:
                    return _ARGPARSE_JA_N[singular]
                return ngettext_(singular, plural, n)

            argparse._, argparse.ngettext = ja_gettext, ja_ngettext  # type: ignore[attr-defined]
        _ARGPARSE_DEPTH += 1
    try:
        yield
    finally:
        with _ARGPARSE_LOCK:
            _ARGPARSE_DEPTH -= 1
            if _ARGPARSE_DEPTH == 0 and _ARGPARSE_ORIG is not None:
                argparse._, argparse.ngettext = _ARGPARSE_ORIG  # type: ignore[attr-defined]
                _ARGPARSE_ORIG = None


def _help_width() -> int:
    """argparse と同じ幅(COLUMNS - 2、最大 78 ではなく argparse の既定どおり)。"""
    import shutil

    return max(shutil.get_terminal_size().columns - 2, 11)


# --help の最後に出す使い方の例(#73。オーナーの依頼)。(コマンド, 日本語の説明, 英語の説明)。
# コマンドは両方の言語で同じ。tests/test_help_examples.py が本物の引数の解釈で通るかを確かめる
EXAMPLES: dict[str, list[tuple[str, str, str]]] = {
    "": [
        (
            "genmodelfit system",
            "この PC の GPU・メモリと一覧の新しさ",
            "this PC's GPU and RAM, and how fresh the list is",
        ),
        ("genmodelfit recommend", "種類ごとのおすすめ(各 3 件)", "top 3 per type for this PC"),
        ("genmodelfit recommend -m image --task edit", "画像の編集のおすすめ", "image editing models"),
        (
            "genmodelfit recommend --uncensored",
            "無検閲のモデルも含める(既定は sfw)",
            "include uncensored models (default is sfw)",
        ),
        ("genmodelfit recommend --vram 12 --ram 32", "別の PC として判定する", "judge for another PC"),
        ('genmodelfit info "Z-Image-Turbo"', "1 つのモデルの詳細", "details of one model"),
        ("genmodelfit recommend --lang ja", "日本語で出す", "output in Japanese"),
    ],
    "recommend": [
        ("genmodelfit recommend", "種類ごとのおすすめ(各 3 件)", "top 3 per type"),
        (
            "genmodelfit recommend -m image --task edit -n 5",
            "画像の編集の上位 5 件",
            "top 5 image editing models",
        ),
        (
            "genmodelfit recommend --uncensored",
            "無検閲のモデルも含める(既定は sfw)",
            "include uncensored models (default is sfw)",
        ),
        ("genmodelfit recommend --vram 12 --ram 32", "別の PC として判定する", "judge for another PC"),
        ("genmodelfit recommend -m video --json", "アプリ向けの JSON", "JSON for apps"),
    ],
    "fit": [
        ("genmodelfit fit", "この PC で動くモデルの全部", "every model that runs on this PC"),
        (
            "genmodelfit fit --sort score --fit good",
            "good だけを種類をまたいだ順位で",
            "good only, ranked across types",
        ),
        (
            "genmodelfit fit --all",
            "動かないもの・補助ツールなども全部",
            "everything, including too tight models and utility tools",
        ),
        (
            "genmodelfit fit -m video --task i2v --vram 12 --ram 32",
            "別の PC で画像から動画",
            "image to video on another PC",
        ),
        (
            "genmodelfit fit --uncensored --json",
            "無検閲も含めて JSON で(既定は sfw)",
            "JSON, uncensored included (default is sfw)",
        ),
    ],
    "info": [
        ('genmodelfit info "Z-Image-Turbo"', "モデルの名前で", "by model name"),
        (
            "genmodelfit info image_z_image_turbo_int8 --json",
            "ワークフローの ID で、JSON",
            "by workflow id, as JSON",
        ),
    ],
    "search": [
        ("genmodelfit search flux", "名前に flux を含むモデル", "models whose name contains flux"),
        (
            "genmodelfit search wan -m video --min-fit good",
            "good 以上の動画だけ",
            "video models that are good or better",
        ),
    ],
    "download": [
        (
            'genmodelfit download "Z-Image-Turbo" --list',
            "1. まず落とすファイル・ライセンス・大きさを見る(落とさない)",
            "1. first see the files, licenses and sizes (downloads nothing)",
        ),
        (
            'genmodelfit download "Z-Image-Turbo" --comfyui-dir C:/ComfyUI',
            "2. ComfyUI のフォルダへ落とす(同意を尋ねる)",
            "2. download into your ComfyUI folder (asks first)",
        ),
        (
            "genmodelfit download MODEL --uncensored --any-license --yes",
            "尋ねずに落とす(無検閲・ライセンス不明も。条件は自分で確かめる)",
            "download without the question, also uncensored / unknown-license (check the terms yourself)",
        ),
    ],
}


def _examples(name: str) -> str:
    """EXAMPLES の節(無ければ空)。説明は今の言語。"""
    rows = EXAMPLES.get(name, [])
    if not rows:
        return ""
    width = max(len(cmd) for cmd, _, _ in rows)
    lines = [tr("例:", "EXAMPLES:")]
    lines += [f"  {cmd:<{width}}  # {tr(ja, en)}" for cmd, ja, en in rows]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """引数の解釈。説明の文は今の言語で作る(main は --lang を先に読んでから呼ぶ。#73)。"""
    lang = argparse.ArgumentParser(add_help=False)
    lang.add_argument(
        "--lang",
        choices=LANGUAGES,
        default=argparse.SUPPRESS,
        help=tr(
            "表示の言語(既定は環境変数 GENMODELFIT_LANG、無ければ en)",
            "display language (default: $GENMODELFIT_LANG, else en)",
        ),
    )
    json_help = tr("JSON で出す", "output JSON")
    common = argparse.ArgumentParser(add_help=False, parents=[lang])
    common.add_argument("--json", action="store_true", help=json_help)
    common.add_argument(
        "--vram",
        type=_non_negative_float,
        metavar="GB",
        help=tr("GPU の VRAM を上書きする(例: 8)", "override the GPU VRAM (e.g. 8)"),
    )
    common.add_argument(
        "--ram",
        type=_positive_float,
        metavar="GB",
        help=tr("メインメモリを上書きする(例: 32)", "override the system RAM (e.g. 32)"),
    )
    common.add_argument("--db", metavar="PATH", help=tr("一覧のファイルを指定する", "model list file to use"))
    common.add_argument(
        "--no-cache",
        action="store_true",
        help=tr(
            "update で取った一覧を使わず、同梱のデータを使う",
            "use the bundled list, not the one fetched by update",
        ),
    )

    filt = argparse.ArgumentParser(add_help=False)
    filt.add_argument(
        "--modality",
        "-m",
        type=_modality_list,
        metavar="TYPE[,TYPE]",
        help=tr("種類で絞る(カンマ区切り可): ", "filter by type (comma-separated): ") + ", ".join(MODALITIES),
    )
    filt.add_argument(
        "--content",
        choices=CONTENT_FILTERS,
        default=None,
        help=tr(
            "sfw / uncensored(無検閲だけ)/ all で絞る。既定は sfw(無検閲も出すなら --uncensored)",
            "sfw / uncensored (only uncensored) / all. Default sfw (--uncensored also shows uncensored)",
        ),
    )

    filt.add_argument(
        "--uncensored",
        action="store_true",
        help=tr(
            "無検閲のモデルも含める(--content all と同じ)",
            "include uncensored models (same as --content all)",
        ),
    )
    filt.add_argument("--width", type=_positive_int)
    filt.add_argument("--height", type=_positive_int)
    filt.add_argument("--frames", type=_positive_int, help=tr("動画のコマ数", "video frames"))
    filt.add_argument(
        "--length", type=_positive_float, metavar="SEC", help=tr("音声の長さ(秒)", "audio length (seconds)")
    )
    filt.add_argument(
        "--precision",
        help=tr(
            "本体の精度を換算する(例: fp8。元の精度が分かるときだけ)",
            "convert the weights to this precision (e.g. fp8; only when the original precision is known)",
        ),
    )
    filt.add_argument(
        "--limit", "-n", type=_positive_int, help=tr("出す件数(1 以上)", "number of rows to show (1 or more)")
    )
    filt.add_argument(
        "--limits",
        action="store_true",
        help=tr(
            "この PC で good になる最大の解像度・長さも出す(JSON の limits)",
            "also show the largest resolution / length that is good on this PC (limits in JSON)",
        ),
    )

    gate = argparse.ArgumentParser(add_help=False)
    gate.add_argument(
        "--license-allowlist",
        metavar="LIST|FILE",
        help=tr(
            "自動のおすすめに許すライセンス(「mit,apache-2.0」か、1"
            " 行に 1 つ書いたファイル)。既定は docs/gate.md",
            'licenses allowed for auto-recommendation ("mit,apache-2.0", or a file with one per line).'
            " Default: docs/gate.md",
        ),
    )
    gate.add_argument(
        "--comfyui-version",
        metavar="VER",
        help=tr(
            "使う ComfyUI の版。アプリは自分の版を必ず渡す(既定は update が記録した最新のリリース)",
            "the ComfyUI version in use; apps should always pass theirs"
            " (default: the latest release recorded by update)",
        ),
    )
    gate.add_argument(
        "--max-head-age-days",
        type=_head_age,
        default=30,
        metavar="DAYS",
        help=tr(
            "HEAD の確認をこの日数まで信じる(既定 30。none で期限なし)",
            "trust HEAD checks for this many days (default 30; none for no limit)",
        ),
    )
    gate.add_argument(
        "--allow-uncensored-auto",
        action="store_true",
        help=tr(
            "無検閲のモデルも自動のおすすめ(Verified)に出してよい(表示の絞り込み・download の同意とは別)",
            "allow uncensored models in automatic recommendations (Verified); not a display filter and"
            " not download consent",
        ),
    )

    epilog = tr(
        "genmodelfit コマンドが見つからない(PATH に無い)ときも、`python -m genmodelfit ...` なら動きます"
        "(README「入れ方」)。",
        "If the genmodelfit command is not found (not on PATH), "
        '`python -m genmodelfit ...` always works (README "Install").',
    )
    from .table import wrap_text

    p = _Parser(
        prog="genmodelfit",
        description=tr(
            "画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、この PC で動くかを判定する。",
            "Right-size generation models to your PC.",
        ),
        epilog="\n".join(wrap_text(epilog, _help_width(), space_room=False)) + "\n\n" + _examples(""),
        formatter_class=_HelpFormatter,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"genmodelfit {__version__}",
        # Python 3.10 の argparse はこの既定の文を訳さずに使うので、自分で渡す(#76)
        help=tr("版を出して終わる", "show program's version number and exit"),
    )
    p.add_argument("--lang", choices=LANGUAGES, default=argparse.SUPPRESS, help=lang._actions[0].help)
    sub = p.add_subparsers(dest="command", required=True)

    def sub_parser(name: str, help_text: str, parents: list) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            parents=parents,
            help=help_text,
            description=_help(name),
            epilog=_examples(name) or None,
            formatter_class=_HelpFormatter,
        )

    sp = sub_parser(
        "system",
        tr(
            "この PC の GPU・メモリ・空きと、一覧の新しさを出す",
            "show this PC's GPU, memory and free disk, and how fresh the model list is",
        ),
        [common, gate],
    )
    sp.add_argument(
        "--disk-path", help=tr("空きを調べる場所(既定はホーム)", "where to check free disk (default: home)")
    )
    sp.add_argument(
        "--all-gpus",
        action="store_true",
        help=tr(
            "NVIDIA 以外も含めて画面の部品(GPU)をすべて出す(Windows は約 1 秒かかる。判定には使わない)",
            "list all display adapters (GPUs), not only NVIDIA"
            " (takes about 1 s on Windows; not used for judging)",
        ),
    )
    sp.set_defaults(func=cmd_system)

    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument(
        "--min-fit",
        choices=FIT_LEVELS,
        help=tr(
            "この段階以上だけ出す(recommend の既定は marginal、fit / search は全部)",
            "show only this fit level or better (recommend default: marginal; fit / search: all)",
        ),
    )
    sel.add_argument(
        "--license",
        type=_license_list,
        metavar="LIST",
        help=tr(
            "すべてのファイルのライセンスがこの中にあるものだけ(例: apache-2.0,mit。不明は合わない)",
            "only models whose files all have one of these"
            " licenses (e.g. apache-2.0,mit; unknown never matches)",
        ),
    )
    sel.add_argument(
        "--fit",
        choices=list(FIT_FILTERS),
        help=tr(
            "出す段階: perfect / good / marginal(その段階だけ)、runnable(動くもの全部。fit の既定)、"
            "all(too_tight・判定できないものも)。--min-fit とは同時に使えない",
            "fit levels to show: perfect / good / marginal (that"
            " level only), runnable (all that run; fit default),"
            " all (too_tight and unknown too). Not with --min-fit",
        ),
    )
    sel.add_argument(
        "--any-license",
        "--include-unknown-license",
        dest="include_unknown_license",
        action="store_true",
        help=tr(
            "ライセンスの分からないファイルを含むモデルも出す(既定は出さない。info <ID> では常に出る。"
            "前からの書き方 --include-unknown-license も同じ)",
            "also show models with files of unknown license (hidden by default; info <ID> always shows them;"
            " old form --include-unknown-license)",
        ),
    )

    sel.add_argument(
        "--task",
        type=_task_list,
        metavar="USE[,USE]",
        help=tr(
            "用途で絞る(gen / edit / i2v / lipsync / motion / voice / upscale / tool、"
            "または t2i・i2i・inpaint などの tasks。一覧は genmodelfit types)",
            "filter by use (gen / edit / i2v / lipsync / motion / voice / upscale / tool,"
            " or task codes such as t2i, i2i, inpaint; see genmodelfit types)",
        ),
    )
    sel.add_argument(
        "--all-workflows",
        action="store_true",
        help=tr(
            "同じモデル(精度・使い方だけ違うもの)をまとめず、ワークフローごとに 1 行ずつ出す",
            "one row per workflow instead of grouping the same model (differing only in precision or use)",
        ),
    )
    tools_help = tr(
        "深度・ポーズ・切り抜き・拡大などの補助ツールも含める(既定は「つくる」モデルだけ)",
        "include utility tools such as depth, pose, background removal and upscaling"
        " (default: generation models only)",
    )

    sp = sub_parser(
        "recommend", tr("おすすめ順に出す", "recommend models, best first"), [common, filt, sel, gate]
    )
    sp.add_argument(
        "--auto-only",
        action="store_true",
        help=tr("自動でおすすめに出せるものだけ", "only auto-recommendable models"),
    )
    sp.add_argument("--include-tools", action="store_true", help=tools_help)
    sp.set_defaults(func=cmd_recommend)  # 既定の 10 件・marginal は cmd_recommend で決める(親の引数は共有)

    listing = argparse.ArgumentParser(add_help=False)
    listing.add_argument(
        "--sort",
        choices=SORTS,
        default="type",
        help=tr(
            "並べ方: type(既定。種類 → 段階 → 点数)、score(種類をまたいだ順位)、vram / disk(少ない順)、name",
            "order: type (default; type -> fit level -> score), score (ranked across types),"
            " vram / disk (smallest first), name",
        ),
    )
    listing.add_argument(
        "--all",
        action="store_true",
        help=tr(
            "全部出す(動かないもの・判定できないもの・補助ツール・ライセンス不明も。無検閲は --uncensored)",
            "show everything (too tight, unknown, utility tools,"
            " unknown license; uncensored needs --uncensored)",
        ),
    )
    listing.add_argument(
        "--perfect",
        "-p",
        action="store_true",
        help=tr("perfect だけ(--fit perfect と同じ)", "perfect only (same as --fit perfect)"),
    )

    sp = sub_parser(
        "fit",
        tr(
            "動くモデルの判定を表で出す(--all で全部)",
            "judge the models that run, as a table (--all for all)",
        ),
        [common, filt, sel, gate, listing],
    )
    sp.add_argument("--include-tools", action="store_true", help=tools_help)
    sp.set_defaults(func=cmd_fit)

    sp = sub_parser(
        "search",
        tr(
            "ID・名前の一部で探して、この PC での判定を出す",
            "search by a part of the id or name and judge the matches for this PC",
        ),
        [common, filt, sel, gate, listing],
    )
    sp.add_argument(
        "query",
        help=tr(
            "探す語(ID・名前の一部。大文字小文字は区別しない)",
            "search word (part of the id or name; case-insensitive)",
        ),
    )
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser(
        "types",
        parents=[common],
        help=tr(
            "大きな種類と、その中の用途(--task の名前)と数",
            "the big types, the uses inside each (names for --task), and counts",
        ),
        description=_help("types"),
        formatter_class=_HelpFormatter,
    )
    sp.add_argument("--content", choices=CONTENT_FILTERS, default=None)
    sp.add_argument(
        "--uncensored",
        action="store_true",
        help=tr(
            "無検閲のモデルも含める(--content all と同じ)",
            "include uncensored models (same as --content all)",
        ),
    )
    sp.set_defaults(func=cmd_types)

    sp = sub_parser("info", tr("1 つのモデルの詳細", "details of one model"), [common, filt, gate])
    sp.add_argument("model", help=tr("モデルの ID または名前の一部", "model id, or a part of the id or name"))
    sp.set_defaults(func=cmd_info)

    sp = sub_parser(
        "update", tr("一覧を上流から作り直す", "rebuild the model list from upstream"), [common, gate]
    )
    sp.add_argument(
        "--offline-sources",
        metavar="DIR",
        help=tr("(試験用)上流の代わりに使うファイル", "(for tests) files to use instead of upstream"),
    )
    sp.add_argument("--skip-hf", action="store_true", help=tr("Hugging Face を見ない", "skip Hugging Face"))
    sp.add_argument(
        "--skip-head",
        action="store_true",
        help=tr("URL の HEAD 確認を飛ばす", "skip the HEAD checks of URLs"),
    )
    sp.add_argument(
        "--heads-only",
        action="store_true",
        help=tr(
            "今の一覧はそのままで、URL の HEAD と ComfyUI の最新の版の確認だけをやり直す",
            "keep the current list; redo only the URL HEAD checks and the latest ComfyUI release check",
        ),
    )
    sp.add_argument(
        "--fields-only",
        action="store_true",
        help=tr(
            "今の一覧はそのままで、アプリ向けの項目(言語・動かし方・説明)と手のモデル(data/manual.json)だけを"
            "当て直す。通信するのは、言語を読むカードと手のモデルのファイルだけ(--offline で通信せず、"
            "保存済みのカードの言語を使う)",
            "keep the current list; re-apply only the app fields (languages, runtime, description) and the"
            " hand-added models (data/manual.json). Only the model cards (for languages) and the hand-added"
            " models' files are fetched (--offline: no network, use the stored card languages)",
        ),
    )
    sp.add_argument(
        "--inspect-only",
        action="store_true",
        help=tr(
            "今の一覧はそのままで、本体のファイルの見出し(精度)・テンプレートのサンプラーの組と読み込みノードの精度・"
            "リポジトリの精度(Hub API・config.json)だけを読み直し、401 だったリポジトリを確かめ直す"
            "(Range 要求。重み本体は落とさない。読んだ結果はキャッシュに残る)",
            "keep the current list; re-read only the weight file headers (precision), the templates' sampler"
            " settings and loader precision, and the repository precision (Hub API, config.json), and recheck"
            " repositories that returned 401 (Range requests;"
            " weights are not downloaded; results are cached)",
        ),
    )
    sp.add_argument(
        "--offline",
        action="store_true",
        help=tr(
            "--inspect-only / --fields-only で通信せず、キャッシュにある見出し・リポジトリの精度だけを使う",
            "with --inspect-only / --fields-only: no network;"
            " use only the cached headers and repository precision",
        ),
    )
    sp.add_argument(
        "--output",
        metavar="PATH",
        help=tr("書き出し先(既定はキャッシュ)", "where to write (default: the cache)"),
    )
    sp.add_argument(
        "--from-release",
        action="store_true",
        help=tr(
            "配布された署名つきの一覧を取り込む(署名が合わなければ何も変えない)。--output があればそこへ書く",
            "take in the signed list published by this project (nothing changes if the signature does not"
            " match). Writes to --output if given",
        ),
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help=tr(
            "(--from-release)前に取り込んだ一覧より古くても取り込む",
            "(--from-release) take it in even if it is older than the list taken in before",
        ),
    )
    sp.add_argument(
        "--allow-partial",
        action="store_true",
        help=tr(
            "一部(テンプレート・Hugging Face)を取得できなくても書き出す(sources.partial が true になる)",
            "write the list even if some sources (templates,"
            " Hugging Face) failed (sources.partial becomes true)",
        ),
    )
    sp.add_argument(
        "--rate-limit-budget",
        type=float,
        metavar="SEC",
        help=tr(
            "Hugging Face の上限(429)で待つ時間の合計の上限(秒。既定 900)。超えたら止める",
            "the most time to wait in total for Hugging Face rate limits (429), in seconds (default 900);"
            " stops when exceeded",
        ),
    )
    sp.set_defaults(func=cmd_update)

    sp = sub_parser(
        "download",
        tr(
            "モデルのファイルを落とす(同意・sha256・続きから・ComfyUI のフォルダ)",
            "download the files of a model (consent, sha256, resume, ComfyUI folders)",
        ),
        [common, gate],
    )
    sp.add_argument(
        "model",
        help=tr(
            "モデルの名前(表の Model)・ワークフローの ID・その一部",
            "model name (the Model column), workflow id, or a part of either",
        ),
    )
    sp.add_argument(
        "--precision",
        metavar="PREC",
        help=tr(
            "精度を指定する(例: fp8 / bf16 / int8 / gguf-q8_0)",
            "precision to use (e.g. fp8 / bf16 / int8 / gguf-q8_0)",
        ),
    )
    sp.add_argument(
        "--budget",
        type=_positive_float,
        metavar="GB",
        help=tr(
            "落とす量の上限(GB。既にあるファイルは数えない)",
            "the most to download in GB (files already there are not counted)",
        ),
    )
    sp.add_argument(
        "--list",
        action="store_true",
        help=tr(
            "落とすファイルと合計を出すだけ(落とさない・通信しない)",
            "only list the files and the total (no download, no network)",
        ),
    )
    where = sp.add_mutually_exclusive_group()
    where.add_argument(
        "--output-dir",
        metavar="DIR",
        help=tr(
            "置き場所(ComfyUI の models と同じ並びで置く)",
            "destination folder (same layout as ComfyUI's models)",
        ),
    )
    where.add_argument(
        "--comfyui-dir",
        metavar="DIR",
        help=tr(
            "ComfyUI のフォルダ。models/<diffusion_models など>/ に置く"
            "(既定は環境変数 GENMODELFIT_COMFYUI_DIR)",
            "the ComfyUI folder; files go to models/<diffusion_models etc.>/"
            " (default: $GENMODELFIT_COMFYUI_DIR)",
        ),
    )
    sp.add_argument(
        "--yes",
        "-y",
        action="store_true",
        help=tr(
            "同意の確認を省く(--json で落とすときは必須)",
            "skip the consent question (required to download with --json)",
        ),
    )
    consent = sp.add_argument_group(
        tr(
            "同意(端末では y/N で尋ねる。--yes・--json・端末でないときに指定する。略さずに書く)",
            "consent (asked y/N in a terminal; pass these with --yes, --json or no terminal;"
            " write them in full)",
        )
    )
    consent.add_argument(
        "--uncensored",
        action="store_true",
        help=tr(
            "無検閲のモデルも落とす(前からの書き方 --content all と同じ)",
            "also download uncensored models (old form --content all)",
        ),
    )
    consent.add_argument(
        "--any-license",
        "--include-unknown-license",
        dest="include_unknown_license",
        action="store_true",
        help=tr(
            "ライセンスの分からないファイルを含むモデルも落とす(条件は自分で確かめる。"
            "前からの書き方 --include-unknown-license も同じ)",
            "also download models with files of unknown license (check the terms yourself;"
            " old form --include-unknown-license)",
        ),
    )
    consent.add_argument(
        "--content",
        choices=CONTENT_FILTERS,
        default=None,
        help=tr(
            "sfw / uncensored / all(--uncensored は all と同じ。既定 sfw)",
            "sfw / uncensored / all (--uncensored is the same as all; default sfw)",
        ),
    )
    sp.set_defaults(func=cmd_download)

    sp = sub.add_parser(
        "gate",
        parents=[common, gate],
        help=tr("自動でおすすめに出せるかの集計", "summary of which models can be auto-recommended"),
    )
    sp.add_argument("--modality", "-m", choices=MODALITIES)
    only = sp.add_mutually_exclusive_group()
    only.add_argument(
        "--only-auto",
        action="store_true",
        help=tr("自動で出せるものの ID も並べる", "also list the ids of auto-recommendable models"),
    )
    only.add_argument(
        "--only-held",
        action="store_true",
        help=tr("保留のものの ID と理由も並べる", "also list the ids of held models and why"),
    )
    sp.set_defaults(func=cmd_gate)

    sp = sub.add_parser(
        "diff",
        parents=[lang],
        help=tr(
            "2 つの一覧(db.json)の違い(増減・ライセンス・自動のおすすめ・HEAD の失敗)",
            "differences between two model lists (db.json): added / removed, licenses, auto-recommendation,"
            " HEAD failures",
        ),
        description=_help("diff"),
        formatter_class=_HelpFormatter,
    )
    sp.add_argument(
        "old",
        nargs="?",
        help=tr(
            "前の一覧。省くと同梱の一覧と取ってきた一覧(update のキャッシュ)を比べる。"
            "1 つだけなら同梱の一覧とそのファイルを比べる",
            "the old list. Omitted: compare the bundled list with the fetched list (the update cache)."
            " With one file: compare the bundled list with that file",
        ),
    )
    sp.add_argument("new", nargs="?", help=tr("新しい一覧", "the new list"))
    sp.add_argument("--json", action="store_true", help=json_help)
    sp.add_argument(
        "--all",
        action="store_true",
        help=tr(
            "保守用の内容も出す(前からの HEAD の失敗・401"
            " のリポジトリ・sources の数・人気の数などだけの変化)",
            "also show the maintenance view (older HEAD failures, 401 repositories, sources numbers,"
            " popularity-only changes)",
        ),
    )
    sp.add_argument(
        "--comfyui-version",
        metavar="VER",
        help=tr(
            "自動のおすすめを比べるときの ComfyUI の版(既定は新しい一覧の記録)",
            "ComfyUI version used to compare auto-recommendation (default: recorded in the new list)",
        ),
    )
    sp.add_argument(
        "--limit",
        "-n",
        type=_positive_int,
        default=None,
        help=tr(
            "1 つの節に出す行の上限(既定 20。--all では全部)",
            "max lines per section (default 20; all with --all)",
        ),
    )
    sp.set_defaults(func=cmd_diff)

    db_cache_help = tr(
        "update で取った一覧を使わず、同梱のデータを使う",
        "use the bundled list, not the one fetched by update",
    )
    write_db_help = tr(
        "書き換える一覧(例 src/genmodelfit/data/db.json)",
        "the list file to rewrite (e.g. src/genmodelfit/data/db.json)",
    )
    sp = sub.add_parser(
        "review",
        parents=[lang],
        help=tr(
            "確認待ち(新しい項目・ライセンスの変化など)の一覧・確かめる・外す(#42)",
            "list, approve or reject models waiting for review (new entries, license changes, ...) (#42)",
        ),
        description=_review_help(),
        formatter_class=_HelpFormatter,
    )
    rsub = sp.add_subparsers(dest="review_command", required=True)
    r = rsub.add_parser(
        "list", parents=[lang], help=tr("確認待ちの項目と理由", "models waiting for review and why")
    )
    r.add_argument("--json", action="store_true", help=json_help)
    r.add_argument(
        "--db",
        metavar="PATH",
        help=tr(
            "一覧のファイル(既定はキャッシュ、無ければ同梱)",
            "the list file (default: the cache, else the bundled list)",
        ),
    )
    r.add_argument("--no-cache", action="store_true", help=db_cache_help)
    r.add_argument("--all", action="store_true", help=tr("確認済みの項目も出す", "also show approved models"))
    r.set_defaults(func=cmd_review_list)
    r = rsub.add_parser(
        "approve",
        parents=[lang],
        help=tr(
            "根拠(一次資料の URL)を付けて確認済みにする(一覧のファイルを書き換える)",
            "approve with evidence (primary-source URLs); rewrites the list file",
        ),
    )
    r.add_argument("id", help=tr("モデルの ID", "model id"))
    r.add_argument(
        "--evidence",
        action="append",
        metavar="URL",
        help=tr(
            "根拠の URL(https。1 つ以上。繰り返して足す)", "evidence URL (https; one or more; repeat to add)"
        ),
    )
    r.add_argument("--note", help=tr("確かめた内容のメモ", "a note on what was checked"))
    r.add_argument(
        "--by",
        default="review",
        help=tr(
            "確かめた人・仕組み(decided_by。既定 review)",
            "who or what checked it (decided_by; default review)",
        ),
    )
    r.add_argument("--db", required=True, metavar="PATH", help=write_db_help)
    r.add_argument("--json", action="store_true", help=json_help)
    r.set_defaults(func=cmd_review_approve)
    r = rsub.add_parser(
        "reject",
        parents=[lang],
        help=tr(
            "一覧から外し、update でも入れないように記録する",
            "remove from the list and record it so that update does not add it again",
        ),
    )
    r.add_argument("id", help=tr("モデルの ID", "model id"))
    r.add_argument("--note", required=True, help=tr("外す理由", "why it is removed"))
    r.add_argument("--db", required=True, metavar="PATH", help=write_db_help)
    r.add_argument(
        "--rejected",
        metavar="PATH",
        help=tr(
            "外した記録のファイル(既定は --db と同じフォルダの review_rejected.txt)",
            "the record file (default: review_rejected.txt in the same folder as --db)",
        ),
    )
    r.add_argument(
        "--hide-name",
        action="store_true",
        help=tr(
            "記録に ID を残さず sha256 だけ残す(安全上の理由で外すとき。既存の除外と同じく名前を残さない)",
            "record only the sha256, not the id (when removing"
            " for safety reasons; like the existing exclusions)",
        ),
    )
    r.add_argument("--json", action="store_true", help=json_help)
    r.set_defaults(func=cmd_review_reject)

    sp = sub.add_parser(
        "bench",
        parents=[common],
        help=tr("動いている ComfyUI で実測する", "measure on a running ComfyUI"),
    )
    sp.add_argument("template", help=tr("テンプレート名(モデルの ID)", "template name (model id)"))
    sp.add_argument(
        "--url",
        required=True,
        help=tr(
            "測りに使う ComfyUI の URL(例 http://127.0.0.1:8188。既定は持たない)",
            "URL of the ComfyUI to measure on (e.g. http://127.0.0.1:8188; no default)",
        ),
    )
    sp.add_argument(
        "--workflow",
        metavar="PATH",
        help=tr(
            "API 形式のワークフロー(省略時はテンプレートから作る)",
            "workflow in API format (default: built from the template)",
        ),
    )
    sp.add_argument("--width", type=_positive_int)
    sp.add_argument("--height", type=_positive_int)
    sp.add_argument("--frames", type=_positive_int)
    sp.add_argument("--length", type=_positive_float, metavar="SEC")
    sp.add_argument("--steps", type=_positive_int)
    sp.add_argument(
        "--output",
        metavar="PATH",
        help=tr(
            "結果の JSON の書き出し先(失敗・標準でない結果もここへ書く)",
            "where to write the result JSON (failed and non-standard results are written here too)",
        ),
    )
    sp.add_argument(
        "--save-failed",
        action="store_true",
        help=tr(
            "失敗(OOM など)の結果もキャッシュに保存する(既定は表示だけ)",
            "also save failed results (OOM etc.) to the cache (default: show only)",
        ),
    )
    sp.add_argument(
        "--save-nonstandard",
        action="store_true",
        help=tr(
            "標準でない条件(--workflow・--set・一部だけ実行など)の結果もキャッシュに保存する",
            "also save results under non-standard conditions"
            " (--workflow, --set, partial runs, ...) to the cache",
        ),
    )
    sp.add_argument(
        "--resolution",
        metavar="WxH",
        help=tr(
            "幅と高さ(例: 768x768)。--width/--height と同じ",
            "width and height (e.g. 768x768); same as --width/--height",
        ),
    )
    sp.add_argument(
        "--set",
        action="append",
        metavar="NAME=VALUE",
        help=tr(
            "入力を名前で変える(例: octree_resolution=192)",
            "change an input by name (e.g. octree_resolution=192)",
        ),
    )
    sp.add_argument(
        "--free",
        action="store_true",
        help=tr(
            "開始前に /free でその ComfyUI のモデルを解放する(自分だけが使う ComfyUI のときに)",
            "free that ComfyUI's models with /free before starting (for a ComfyUI only you use)",
        ),
    )
    sp.add_argument(
        "--timeout",
        type=float,
        metavar="SEC",
        help=tr(
            "待つ時間の上限(0 より大きい秒数。既定 1800)", "the most time to wait (seconds > 0; default 1800)"
        ),
    )
    sp.add_argument(
        "--force",
        action="store_true",
        help=tr(
            "GPU が使われている・GPU の空きを確かめられないときも測る",
            "measure even when the GPU is in use or its free memory cannot be checked",
        ),
    )
    sp.add_argument(
        "--source",
        help=tr("誰がどこで測ったか(結果の source に入る)", "who measured it where (goes into source)"),
    )
    sp.add_argument("--note", help=tr("結果の note に足す文", "text to add to the result's note"))
    sp.add_argument(
        "--submission",
        metavar="PATH",
        help=tr(
            "提出用の JSON(contrib_schema 1)も書く。フォルダなら <フォルダ>/<GPU>/<日付>-<sha8>.json"
            "(フォークの data/contrib を渡すとそのまま PR にできる)。何も送らない(CONTRIBUTING.md)",
            "also write a submission JSON (contrib_schema 1). For a folder: <folder>/<GPU>/<date>-<sha8>.json"
            " (pass your fork's data/contrib to open a PR as is). Nothing is sent (CONTRIBUTING.md)",
        ),
    )
    sp.set_defaults(func=cmd_bench)

    cj = argparse.ArgumentParser(add_help=False, parents=[lang])
    cj.add_argument("--json", action="store_true", help=json_help)
    cj.add_argument(
        "--db",
        metavar="PATH",
        help=tr(
            "モデル ID を確かめる一覧(既定は同梱のもの。キャッシュは使わない)",
            "the list used to check model ids (default: the bundled list; the cache is not used)",
        ),
    )
    sp = sub.add_parser(
        "contrib",
        parents=[lang],
        help=tr(
            "実測の提出用 JSON を作る・見る・検査する・まとめる(何も送らない)",
            "make, preview, validate and aggregate measurement submissions (nothing is sent)",
        ),
        description=_contrib_help(),
        formatter_class=_HelpFormatter,
    )
    csub = sp.add_subparsers(dest="contrib_command", required=True)
    c = csub.add_parser(
        "make",
        parents=[cj],
        help=tr(
            "bench が保存した結果から提出用の JSON を作る",
            "make a submission JSON from a result saved by bench",
        ),
    )
    c.add_argument(
        "--from",
        dest="source",
        required=True,
        metavar="FILE",
        help=tr("bench の結果の JSON", "the bench result JSON"),
    )
    c.add_argument(
        "--output",
        default=CONTRIB_DIR_DEFAULT,
        metavar="PATH",
        help=tr(
            f"書き出す場所(フォルダなら <GPU>/<日付>-<sha8>.json。既定 {CONTRIB_DIR_DEFAULT})",
            f"where to write (for a folder: <GPU>/<date>-<sha8>.json; default {CONTRIB_DIR_DEFAULT})",
        ),
    )
    c.add_argument(
        "--no-pc-info",
        action="store_true",
        help=tr(
            "この PC の OS の種類・メモリ・GPU の世代を入れない(ComfyUI が別の PC だったとき)",
            "do not include this PC's OS family, RAM and GPU generation (when ComfyUI ran on another PC)",
        ),
    )
    c.set_defaults(func=cmd_contrib_make)
    c = csub.add_parser(
        "show",
        parents=[cj],
        help=tr("送る中身と個人の情報の点検を見せる", "show what will be shared and the privacy checklist"),
    )
    c.add_argument("file", help=tr("提出用の JSON", "the submission JSON"))
    c.set_defaults(func=cmd_contrib_show)
    c = csub.add_parser(
        "validate",
        parents=[cj],
        help=tr(
            "形・値・個人の情報・重複を検査する(CI でも使う)",
            "check shape, values, personal information and duplicates (also used in CI)",
        ),
    )
    c.add_argument(
        "files",
        nargs="+",
        metavar="FILE|DIR",
        help=tr(
            "提出用の JSON(フォルダなら下の *.json すべて)",
            "submission JSON files (for a folder: every *.json under it)",
        ),
    )
    c.add_argument(
        "--strict",
        action="store_true",
        help=tr("注意(warnings)も失敗にする", "treat warnings as failures too"),
    )
    c.add_argument(
        "--no-filename-check",
        action="store_true",
        help=tr(
            "ファイル名(<GPU>/<日付>-<sha8>.json)を確かめない",
            "do not check the file name (<GPU>/<date>-<sha8>.json)",
        ),
    )
    c.add_argument(
        "--markdown",
        action="store_true",
        help=tr("Markdown の表で出す(CI のまとめ用)", "output a Markdown table (for the CI summary)"),
    )
    c.set_defaults(func=cmd_contrib_validate)
    c = csub.add_parser(
        "aggregate",
        parents=[cj],
        help=tr(
            "同じモデル・設定・GPU ごとにまとめ、精度を点検する",
            "aggregate by model, settings and GPU, and check the accuracy",
        ),
    )
    c.add_argument(
        "dir",
        nargs="?",
        default=CONTRIB_DIR_DEFAULT,
        help=tr(
            f"提出のフォルダ(既定 {CONTRIB_DIR_DEFAULT})",
            f"the submissions folder (default {CONTRIB_DIR_DEFAULT})",
        ),
    )
    c.add_argument(
        "--output",
        metavar="FILE",
        help=tr(
            "まとめを書き出す(同梱するなら src/genmodelfit/data/measurements_community.json)",
            "write the result (to bundle it: src/genmodelfit/data/measurements_community.json)",
        ),
    )
    c.add_argument(
        "--report",
        action="store_true",
        help=tr(
            "校正をやり直し、種類ごとの誤差の表と閾値の判定を出す",
            "redo the calibration and show the error table per type and the threshold check",
        ),
    )
    c.add_argument(
        "--threshold",
        type=_positive_float,
        default=25.0,
        metavar="PCT",
        help=tr("誤差の閾値(%%、既定 25)", "error threshold (%%, default 25)"),
    )
    c.add_argument(
        "--fail-on-breach",
        action="store_true",
        help=tr(
            f"--report で閾値を超えたら終了コード {EXIT_BREACH} にする",
            f"with --report, exit with code {EXIT_BREACH} when the threshold is exceeded",
        ),
    )
    c.add_argument(
        "--allow-empty",
        action="store_true",
        help=tr(
            "まとめが空でも --output に書き出す(既定は、空のまとめで上書きしない)",
            "write to --output even if the result is empty (by default an empty result does not overwrite)",
        ),
    )
    c.add_argument(
        "--check",
        action="store_true",
        help=tr(
            "書き出さず、同梱のまとめにまだ入っていない提出を数える(あれば終了コード 1)",
            "write nothing; count the submissions not in the bundled aggregate yet (exit code 1 if any)",
        ),
    )
    c.set_defaults(func=cmd_contrib_aggregate)
    return p


_REVIEW_HELP_JA = """確認待ち(review.status = pending)の項目を出す・根拠を付けて確かめる・一覧から外す(#42)。
pending は update が前の一覧と比べて付ける
(new / license_permissive / need_decreased / uncensored_new / files_changed)。
確かめるまで自動のおすすめに出さない(gate の reviewed)。

  list     [--json] [--all] [--db PATH]          確認待ちと理由
  approve  <id> --evidence URL [--evidence URL…] [--note TEXT] [--by NAME] --db PATH
           根拠が無ければ断る。一覧のファイルをその場で書き換える
  reject   <id> --note TEXT --db PATH [--rejected FILE] [--hide-name]
           一覧から外し、review_rejected.txt に記録する(update でも入れない)

副作用:
  通信しない。list は読むだけ。approve / reject は --db のファイルを書き換える(reject は外した
  記録のファイルにも足す)。

終了コード:
  0  成功
  1  失敗(ID が無い・根拠が無い・確認待ちでない・--db が読めない。--json なら誤りは "error" に入る)
  2  使い方の誤り

使い方の例(エージェント・アプリ向け):
  genmodelfit review list --json --db src/genmodelfit/data/db.json
  genmodelfit review approve hf:org/model --db src/genmodelfit/data/db.json \\
      --evidence https://huggingface.co/org/model/blob/main/LICENSE
  genmodelfit review reject hf:org/model --db src/genmodelfit/data/db.json --note "商用には使えない"
詳しくは docs/schema.md「review」、docs/release.md「毎週のスキル」"""

_REVIEW_HELP_EN = """List, approve or reject models waiting for review (#42).
update marks an entry pending (review.status = pending) when it differs from the previous list
(new / license_permissive / need_decreased / uncensored_new / files_changed).
Pending entries are not auto-recommended until approved (gate check: reviewed).

  list     [--json] [--all] [--db PATH]          entries waiting for review and why
  approve  <id> --evidence URL [--evidence URL…] [--note TEXT] [--by NAME] --db PATH
           refused without evidence; rewrites the list file in place
  reject   <id> --note TEXT --db PATH [--rejected FILE] [--hide-name]
           removes the entry and records it in review_rejected.txt (update does not add it again)

SIDE EFFECTS:
  No network access. list is read-only; approve / reject rewrite the --db file (reject also appends to the
  rejected file).

EXIT CODES:
  0  Success
  1  Error (no such id, no evidence, not pending, unreadable --db; with --json the error is in "error")
  2  Usage error

AGENT USAGE:
  genmodelfit review list --json --db src/genmodelfit/data/db.json
  genmodelfit review approve hf:org/model --db src/genmodelfit/data/db.json \\
      --evidence https://huggingface.co/org/model/blob/main/LICENSE
  genmodelfit review reject hf:org/model --db src/genmodelfit/data/db.json --note "non-commercial only"
Details: docs/schema.md ("review"), docs/release.md (weekly skill)"""


def _review_help() -> str:
    return tr(_REVIEW_HELP_JA, _REVIEW_HELP_EN)


CONTRIB_DIR_DEFAULT = "data/contrib"
_CONTRIB_HELP_JA = """実測の提出用 JSON(contrib_schema 1)を作る・送る前に見る・検査する・まとめる。
genmodelfit は何も送らない(共有は、利用者が自分の GitHub で Pull Request を出すときだけ。
CONTRIBUTING.md)。

  make      --from <bench の結果>  → data/contrib/<GPU>/<日付>-<sha8>.json
  show      <file>                 送る中身と個人の情報の点検(同意のため)
  validate  <files|dirs>...        形・値の範囲・個人の情報・重複・ファイル名を検査
  aggregate [dir] [--output F] [--report] [--threshold 25] [--fail-on-breach]
            同じモデル・設定・GPU ごとに中央値と MAD で外れ値を除いてまとめ、校正の誤差を点検
  aggregate [dir] --check
            同梱のまとめにまだ入っていない提出を数える(まとめは版を出すときに作る)

副作用:
  通信しない。make は提出のファイルを書く。aggregate は --output のときだけ書く。

終了コード:
  0  成功(validate: すべてのファイルが通った。show: 直すところが無い)
  1  失敗 / validate: 機械的な検査に通らないファイルがある、または --strict で注意がある
     (マージする人向けの確認では失敗にしない)/ show: 直すところがある /
     aggregate: フォルダが無い、または --output で結果が空(--allow-empty が無いとき)/
     aggregate --check: 同梱のまとめに入っていない提出がある、またはまとめ直すと違う
  2  使い方の誤り
  3  aggregate --report --fail-on-breach: 誤差が閾値を超えた

使い方の例(エージェント・アプリ向け):
  genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --submission data/contrib/
  genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json
  genmodelfit contrib validate data/contrib --json
  genmodelfit contrib aggregate --report --json --fail-on-breach
詳しくは docs/schema.md「contrib」"""

_CONTRIB_HELP_EN = """Make, preview, validate and aggregate measurement submissions (contrib_schema 1).
genmodelfit sends nothing (data is shared only when you open a Pull Request from your own GitHub
account; CONTRIBUTING.md).

  make      --from <bench result>  → data/contrib/<GPU>/<date>-<sha8>.json
  show      <file>                 what will be shared, and the privacy checklist (for consent)
  validate  <files|dirs>...        check shape, value ranges, personal information, duplicates, file names
  aggregate [dir] [--output F] [--report] [--threshold 25] [--fail-on-breach]
            group by model, settings and GPU, drop outliers by median and MAD, check the calibration error
  aggregate [dir] --check
            count the submissions not in the bundled aggregate yet (it is rebuilt for each release)

SIDE EFFECTS:
  No network access. make writes the submission files; aggregate writes only with --output.

EXIT CODES:
  0  Success (validate: every file passed; show: nothing to fix)
  1  Error / validate: a file failed the mechanical checks, or has a warning with --strict (notices for the
     maintainer who merges never fail) /
     show: something to fix / aggregate: no such folder, or an empty result with --output
     (unless --allow-empty) / aggregate --check: submissions not in the bundled aggregate yet, or
     re-aggregating it differs
  2  Usage error
  3  aggregate --report --fail-on-breach: an error exceeded the threshold

AGENT USAGE:
  genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --submission data/contrib/
  genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json
  genmodelfit contrib validate data/contrib --json
  genmodelfit contrib aggregate --report --json --fail-on-breach
Details: docs/schema.md ("contrib")"""


def _contrib_help() -> str:
    return tr(_CONTRIB_HELP_JA, _CONTRIB_HELP_EN)


def _quiet_stdout() -> None:
    # 終了時の flush で再び落ちないように、出力先(fd)ごと捨て先に向ける(Python の文書の推奨手順)
    with contextlib.suppress(OSError, ValueError, AttributeError):
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
    with contextlib.suppress(OSError):
        sys.stdout = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows の cp932 でも落ちず、組み込み側が UTF-8 で読めるように
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(AttributeError, ValueError, OSError):
                stream.reconfigure(encoding="utf-8")
    argv = sys.argv[1:] if argv is None else argv
    lang = _lang_in(argv)  # 説明の文を作る前に言語を決める(#73)
    # ContextVar で固定するので、ほかのスレッドのライブラリの呼び出しには効かず、main を出れば元に戻る
    with i18n.forced(lang) if lang else contextlib.nullcontext(), _argparse_language():
        return _main(argv)


def _is_lang_flag(a: str) -> bool:
    """--lang と、argparse が受ける略し方(--la・--lan。--l は --limit などと紛れるので数えない)。"""
    return len(a) >= 4 and "--lang".startswith(a)


def _lang_in(argv: list[str]) -> str | None:
    """--lang X / --lang=X(略した --lan X も)を先に読む(-- より後は見ない)。
    正しくない値は None(argparse が誤りにする)。"""
    found = None
    for k, a in enumerate(argv):
        if a == "--":
            break
        if _is_lang_flag(a) and k + 1 < len(argv):
            found = argv[k + 1]
        elif "=" in a and _is_lang_flag(a.split("=", 1)[0]):
            found = a.split("=", 1)[1]
    return found if found in LANGUAGES else None


# 同意の指定(年齢・ライセンスの同意)は略して書けない(1〜3 文字の打ち間違いで同意したことにならないように。
# #75)。argparse の略し方(--u → --uncensored など)を、解釈する前に止める
CONSENT_FLAGS = ("--uncensored", "--any-license", "--include-unknown-license", "--content")


def _check_consent_spelling(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    for a in argv:
        if a == "--":
            break
        name = a.split("=", 1)[0]
        if not name.startswith("--") or len(name) <= 2 or name in CONSENT_FLAGS:
            continue
        full = [f for f in CONSENT_FLAGS if f.startswith(name)]
        if full:
            parser.error(
                tr(
                    f"{name}: 同意の指定は略さずに書いてください({' / '.join(full)})",
                    f"{name}: write consent flags in full ({' / '.join(full)})",
                )
            )


def _main(argv: list[str]) -> int:
    parser = build_parser()
    if not argv:  # 引数なしは使い方の全文を出して正常に終わる(初めて試す人向け)
        parser.print_help()
        return 0
    _check_consent_spelling(parser, argv)
    args = parser.parse_args(argv)
    chosen = getattr(args, "lang", None)
    # 先に読めなかった形の --lang(念のため)。説明の文から作り直す
    if chosen and chosen != i18n.get_language():
        with i18n.forced(chosen):
            return _main(argv)
    # --uncensored は --content all と同じ(#75)。--content sfw とは同時に使えない
    if hasattr(args, "content"):
        if getattr(args, "uncensored", False):
            if args.content not in (None, "all"):
                parser.error(
                    tr(
                        f"--uncensored(= --content all)と --content {args.content} は同時に指定できません",
                        f"--uncensored (= --content all) and --content {args.content}"
                        " cannot be used together",
                    )
                )
            args.content = "all"
        elif args.content is None:
            args.content = DEFAULT_CONTENT
    # 使い方の誤り(argparse と同じく終了コード 2)
    if getattr(args, "min_fit", None) is not None and (
        getattr(args, "fit", None) is not None or getattr(args, "perfect", False)
    ):
        parser.error(
            tr(
                "--fit(--perfect)と --min-fit は同時に指定できません",
                "--fit (--perfect) and --min-fit cannot be used together",
            )
        )
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
            print(tr(f"エラー: {e}", f"error: {e}"), file=sys.stderr)
            for m in (cands or [])[:30]:
                print(f"  {m.id}  ({m.name})", file=sys.stderr)
            if cands and len(cands) > 30:
                rest = len(cands) - 30
                print(
                    tr(f"  …ほか {rest} 件(--json で全部)", f"  … {rest} more (--json for all)"),
                    file=sys.stderr,
                )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
