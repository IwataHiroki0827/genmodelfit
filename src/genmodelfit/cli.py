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
import sys
from pathlib import Path
from typing import Any

from ._version import __version__
from .db import Database, DatabaseError
from .db import load as load_db
from .estimate import GIB, estimate, load_coefficients
from .hardware import detect
from .models import FIT_LEVELS, MODALITIES, Settings, SystemInfo
from .recommend import CONTENT_FILTERS, DEFAULT_CONTENT, FIT_FILTERS, SORTS, select


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
    return detect(
        disk_path=getattr(args, "disk_path", None),
        vram_gb=args.vram,
        ram_gb=args.ram,
        other_gpus=True if getattr(args, "all_gpus", False) else None,
    )


def _wrapped(text: str, first: str = "  - ", rest: str = "    ", width: int = 100) -> list[str]:
    """全角を 2 桁で数えて width 桁に折り返す(system の注意の文は長い日本語になる)。"""
    from rich.cells import cell_len

    lines, cur, prefix = [], "", first
    for ch in text:
        if cell_len(prefix + cur + ch) > width:
            lines.append(prefix + cur)
            cur, prefix = "", rest
        cur += ch
    lines.append(prefix + cur)
    return lines


def _swap_line(s: SystemInfo) -> str | None:
    """仮想メモリの 1 行(Windows はページファイルとコミット、Linux はスワップ)。分からなければ None。"""
    if s.swap_auto is None and s.swap_total_gb is None and s.commit_limit_gb is None:
        return None
    label = "Pagefile" if s.commit_limit_gb is not None or s.os.startswith("Windows") else "Swap"
    if s.swap_max_gb == 0 and s.swap_auto is False:
        text = "none"
    elif s.swap_auto:
        now = f"{s.swap_total_gb:.1f} GB now, " if s.swap_total_gb is not None else ""
        text = f"{now}system managed (grows)"
    elif s.swap_max_gb is not None:
        text = f"up to {s.swap_max_gb:.1f} GB"
    else:
        text = f"{s.swap_total_gb:.1f} GB now" if s.swap_total_gb is not None else "unknown"
    if s.commit_limit_gb is not None and s.commit_used_gb is not None:
        text += f"; commit {s.commit_used_gb:.1f} / {s.commit_limit_gb:.1f} GB in use"
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


def _data_status(args: argparse.Namespace, db: Database, source: str | None = None):
    from .status import compute

    return compute(db, _policy(args), source=source)


def format_data_status(st) -> list[str]:
    """`system` と `update` の Data の欄(人が読む形)。"""
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
    print("=== System ===")
    print(f"OS:   {s.os}")
    print(f"CPU:  {s.cpu} ({s.cpu_cores} threads)")
    print(f"RAM:  {s.ram_total_gb:.1f} GB (available {s.ram_available_gb:.1f} GB)")
    swap = _swap_line(s)
    if swap:
        print(swap)
    if s.disk_free_gb is not None:
        print(f"Disk: {s.disk_free_gb:.1f} GB free")
    if not s.gpus:
        print("GPU:  none detected (CPU only)")
    elif s.best_gpu is None:
        print("GPU:  no usable NVIDIA GPU (judged as CPU only; see notes)")
    for i, g in enumerate(s.gpus, 1):
        mem = []
        if g.vram_free_gb is not None:
            mem.append(f"{g.vram_free_gb:.1f} GB free")
        if g.vram_used_gb is not None:
            mem.append(f"{g.vram_used_gb:.1f} GB used by other apps")
        extra = "".join(f", {m}" for m in mem)
        print(f"GPU {i}: {g.name} ({g.vram_total_gb:.1f} GB VRAM{extra})")
        cc = (
            f"compute capability {g.compute_capability}"
            if g.compute_capability
            else "compute capability unknown"
        )
        if g.compute_capability_source == "name":
            cc += " (guessed from the name)"
        arch = f" ({g.arch})" if g.arch else ""
        use = (
            "used for judging"
            if g is s.best_gpu
            else ("not used" if g.backend != "cuda" else "not the largest")
        )
        print(f"       {cc}{arch}, driver {g.driver or 'unknown'}, {g.backend}: {use}")
        if g.torch_cuda:
            print(f"       PyTorch CUDA builds that run here: {', '.join(g.torch_cuda)}")
        if g.compute_dtype:
            yes = {True: "yes", False: "no", None: "unknown"}
            print(f"       ComfyUI precision: {g.compute_dtype} (bf16 {yes[g.bf16]}, fp8 {yes[g.fp8]})")
    if s.other_gpus_checked:
        print("Display adapters (informational, not used for judging):")
        if not s.other_gpus:
            print("  (none found)")
        for a in s.other_gpus:
            vram = f", {a.vram_gb:.1f} GB ({a.vram_source})" if a.vram_gb else ""
            print("\n".join(_wrapped(f"{a.name} [{a.vendor}{vram}]")))
    if s.overrides:
        print(f"(overridden: {', '.join(s.overrides)})")
    if s.notes:
        print("Notes:")
        for n in s.notes:
            print("\n".join(_wrapped(n)))
    print("=== Data ===")
    for line in format_data_status(st):
        print(line)
    return 0


def _license_list(text: str) -> list[str]:
    items = [x.strip().lower() for x in text.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError(
            "ライセンスを 1 つ以上、カンマ区切りで指定してください(例: apache-2.0,mit)"
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
    )
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """-m も --task も無ければ種類ごとの節(各 -n 件、既定 3)。あれば 1 つの表(既定 10 件)。"""
    s, db = _system(args), _db(args)
    sections = not args.modality and not args.task
    per_type = (args.limit or 3) if sections else None
    limit = None if sections else (args.limit if args.limit is not None else 10)
    if args.fit is not None and args.min_fit is not None:
        raise CliError("--fit と --min-fit は同時に指定できません")
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
    )
    filters = _filters(args, min_fit, limit) | {
        "per_type": per_type,
        "auto_only": args.auto_only,
        "include_tools": args.include_tools,
    }
    return _show(args, "recommend", "Recommended", s, db, sel, filters, sections=sections)


def _fit_choice(args: argparse.Namespace, default: str | None) -> tuple[str | None, str | None]:
    """(min_fit, fit)。--perfect は --fit perfect。--all は --fit all。どれも無ければ default。"""
    if args.perfect:
        args.fit = "perfect"
    if getattr(args, "all", False) and args.fit is None and args.min_fit is None:
        args.fit = "all"
    if args.fit is not None and args.min_fit is not None:
        raise CliError("--fit と --min-fit は同時に指定できません")
    if args.fit is None and args.min_fit is None:
        args.fit = default
    return args.min_fit, args.fit


def cmd_fit(args: argparse.Namespace) -> int:
    """既定は動くもの(perfect / good / marginal)だけ、補助ツール・ライセンス不明・無検閲は隠す。--all で全部。
    並びは種類 → 段階 → 点数(--sort)。自動のおすすめを先にはしない(Auto の列で分かる)。"""
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
    return _show(args, "fit", "All models", s, db, sel, filters)


def cmd_search(args: argparse.Namespace) -> int:
    """ID・名前の一部で探す(大文字小文字を区別しない)。0 件でも終了コード 0。空の語は 1。"""
    s, db = _system(args), _db(args)
    query = args.query.strip()
    if not query:
        raise CliError("探す語が空です")
    min_fit, fit = _fit_choice(args, None)
    sel = _select(
        args, s, db, query=query, min_fit=min_fit, fit=fit, limit=args.limit, sort=args.sort, auto_first=False
    )
    filters = {"query": query} | _filters(args, min_fit, args.limit) | {"sort": args.sort, "all": args.all}
    return _show(args, "search", f'Search "{query}"', s, db, sel, filters)


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
    lines = [f"Types (content {args.content}; models = distinct weights, workflows = list entries)"]
    lines.append(f"{'Type':<6}{'Models':>7}{'Workflows':>10}  Uses (workflows)")
    for r in rows:
        used = ", ".join(f"{u} {n}" for u, n in r["uses"].items())
        lines.append(f"{r['modality']:<6}{r['models']:>7}{r['workflows']:>10}  {used}")
    lines.append("")
    lines.append("Uses for --task (a name below, or a task code such as t2i / i2i / inpaint):")
    for u in uses:
        lines.append(f"  {u['use']:<8}{u['label']} / {u['label_ja']}  [{', '.join(u['tasks'])}]")
    lines.append("Examples: genmodelfit recommend -m image --task edit / -m video --task i2v,lipsync")
    _emit(lines)
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
        raise CliError(
            f"モデルが見つかりません: {query}(--content {content} で外れた {kinds} のモデルが "
            f"{len(hits)} 件あります。--content all を付けるか ID で指定してください)"
        )
    raise CliError(f"モデルが見つかりません: {query}")


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

    n = len(same) + 1
    out = [f"{m.name}  [{m.id}]"]
    out.append(
        f"  model:    {index_for(db).name_of.get(key, m.name)}"
        f"  ({n} workflow{'s' if n > 1 else ''} use{'' if n > 1 else 's'} these weights)"
    )
    if is_community(m):
        out.append(f"  community template by {m.template.username} (not made by ComfyUI)")
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
    if same:
        k = len(same)
        out.append(f"  same model ({k} other workflow{'s' if k > 1 else ''}; precision / use differ):")
        for w in same:
            need = w["memory_required_gb"]
            vram = f"{need:.1f} GB" if need is not None else "-"
            prec = w["precision"] or "-"
            out.append(f"    - {w['id']}  ({w['use']}, {prec}, {w['fit_level']}, {vram})")
    _emit(out)
    return 0


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
    from .update import UpdateFailed, inspect_database, make_fetcher, refresh_heads, update_database

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
            print(f"エラー: {e}", file=sys.stderr)
        return 1
    st = _data_status(args, db, source="cache" if out_path == cached_db_path() else "path")
    if args.json:
        db.origin = "path"
        _out(_envelope("update", None, db, output=str(out_path), sources=db.sources, status=st.to_dict()))
        return 0
    _print_status_after_update(st)
    if getattr(args, "inspect_only", False):
        fh = db.sources["file_headers"]
        print(f"ファイルの見出しの結果を書き出しました: {out_path}")
        print(
            f"  本体のファイル {fh['urls']} 件: キャッシュ {fh['from_cache']}、読んだ {fh['fetched']}、"
            f"読めない {fh['failed']}、未確認(オフライン) {fh['skipped_offline']}"
        )
        print(f"  精度が分からない項目: {fh['precision_unknown']} 件")
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
        print(f"一覧を更新しました: {len(db.models)} 件(生成日時 {db.generated_at}) -> {dest}")
        _print_status_after_update(st)
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


_JSON_FIELDS = """  JSON (--json): { genmodelfit, command, system: {...}, database: {...}, filters: {...},
  matched, hidden: {reason: count}, groups: [{modality, matched, shown}],
  models: [{ id, name, model_name,
  modality, tasks, use, content, source, fit_level, fit_label, run_mode, run_mode_label, score,
  score_components { fit, headroom, readiness, basis, popularity }, memory_kind,
  memory_required_gb, memory_available_gb, utilization_pct, ram_required_gb, disk_size_gb,
  total_size_gb, license, licenses, auto, held_reasons, estimate_confidence, notes,
  files_count, precision, group_key, variants: [{precision, workflow_id, ...}],
  workflows: [{id, name, tasks, use, precision, fit_level, ...}], estimate, gate }] }
  One item per model (same weights family + size); --all-workflows gives one per workflow.
  Details: docs/schema.md"""

_COMMON_HELP = """PRECONDITIONS:
  None. The GPU (NVIDIA, via nvidia-smi) and RAM are detected; --vram / --ram override them.
  The bundled model list is used unless `genmodelfit update` stored a newer one (--no-cache).

SIDE EFFECTS:
  None - read-only. No network access.

EXIT CODES:
  0  Success{zero}
  1  Error{one} (e.g. a broken --db file; with --json the error is in "error")
  2  Usage error (unknown option or value)
"""

HELP = {
    "recommend": """Recommend generation models for this PC, best first.
この PC に合う生成モデルを、おすすめ順に出す。

Without -m / --task: one section per type (image, video, music, sfx, tts, 3d), top -n
(default 3) each. With -m or --task: one table (default 10).
One row per model; the precision and workflow that fit best are chosen (--all-workflows: all).
Order: models that run, then auto-recommendable ones (gate: Auto=yes), then Score (0-100),
then id. Uncensored models and utility tools are hidden unless asked. The default output is a
table (llmfit uses JSON by default); add --json for apps.
並べ方: 動くもの → 自動でおすすめに出せるもの → 点数の高い順 → ID。

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
すべてのモデルをこの PC で判定して表にする(動かないものも、recommend と同じ並べ方で)。

"""
    + _COMMON_HELP.format(zero="", one="")
    + """
AGENT USAGE:
  genmodelfit fit --json
  genmodelfit fit -m video --task i2v --min-fit good -n 20
  genmodelfit fit --content all --all-workflows --vram 12 --ram 32 --json

"""
    + _JSON_FIELDS,
    "search": """Search models by a part of the id or name (case-insensitive) and judge them for this PC.
ID・名前の一部で探し(大文字小文字は区別しない)、この PC での判定を出す。
--content applies (default sfw). With no match the list is empty and the exit code is 0.

"""
    + _COMMON_HELP.format(
        zero=" (also when nothing matches: empty models / 「no match」)", one=": blank query"
    )
    + """
AGENT USAGE:
  genmodelfit search flux --json
  genmodelfit search wan -m video --min-fit good
  genmodelfit search sdxl --content all --json

"""
    + _JSON_FIELDS.replace("filters: {...}", "filters: {query, ...}"),
    "types": """List the big types (image, video, music, sfx, tts, 3d) and the uses inside each type,
with counts. The use names are what --task accepts.
大きな種類と、その中の用途(--task に使う名前)と数を出す。

"""
    + _COMMON_HELP.format(zero="", one="").replace(
        "  None. The GPU (NVIDIA, via nvidia-smi) and RAM are detected; --vram / --ram override them.\n",
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
1 つのモデルの詳細(ファイル・ライセンス・見積もり・この PC での限界・自動おすすめの条件・
同じモデルのほかのワークフロー)。
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
この PC と、使っている一覧の新しさ(期限切れか)を出す。

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
    "update": """Rebuild the model list from upstream, redo the HEAD checks (--heads-only), or take in the
signed list published by this project (--from-release).
一覧を上流から作り直す・HEAD だけやり直す・署名付きの配布を取り込む。

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
}


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
    filt.add_argument(
        "--modality",
        "-m",
        type=_modality_list,
        metavar="TYPE[,TYPE]",
        help=f"種類で絞る(カンマ区切り可): {', '.join(MODALITIES)}",
    )
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

    p = argparse.ArgumentParser(
        prog="genmodelfit",
        description="Right-size generation models to your PC.",
        epilog="If the genmodelfit command is not found (not on PATH), "
        "`python -m genmodelfit ...` always works. "
        "genmodelfit コマンドが見つからないときは `python -m genmodelfit ...` で動きます"
        "(README「Install」)。",
    )
    p.add_argument("--version", action="version", version=f"genmodelfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def sub_parser(name: str, help_text: str, parents: list) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            parents=parents,
            help=help_text,
            description=HELP[name],
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    sp = sub_parser("system", "この PC の GPU・メモリ・空きと、一覧の新しさを出す", [common, gate])
    sp.add_argument("--disk-path", help="空きを調べる場所(既定はホーム)")
    sp.add_argument(
        "--all-gpus",
        action="store_true",
        help="NVIDIA 以外も含めて画面の部品(GPU)をすべて出す(Windows は約 1 秒かかる。判定には使わない)",
    )
    sp.set_defaults(func=cmd_system)

    sel = argparse.ArgumentParser(add_help=False)
    sel.add_argument(
        "--min-fit",
        choices=FIT_LEVELS,
        help="この段階以上だけ出す(recommend の既定は marginal、fit / search は全部)",
    )
    sel.add_argument(
        "--license",
        type=_license_list,
        metavar="LIST",
        help="すべてのファイルのライセンスがこの中にあるものだけ(例: apache-2.0,mit。不明は合わない)",
    )
    sel.add_argument(
        "--fit",
        choices=list(FIT_FILTERS),
        help="出す段階: perfect / good / marginal(その段階だけ)、runnable(動くもの全部。fit の既定)、"
        "all(too_tight・判定できないものも)。--min-fit とは同時に使えない",
    )
    sel.add_argument(
        "--include-unknown-license",
        action="store_true",
        help="ライセンスの分からないファイルを含むモデルも出す(既定は出さない。info <ID> では常に出る)",
    )

    sel.add_argument(
        "--task",
        type=_task_list,
        metavar="USE[,USE]",
        help="用途で絞る(gen / edit / i2v / lipsync / motion / voice / upscale / tool、"
        "または t2i・i2i・inpaint などの tasks。一覧は genmodelfit types)",
    )
    sel.add_argument(
        "--all-workflows",
        action="store_true",
        help="同じモデル(精度・使い方だけ違うもの)をまとめず、ワークフローごとに 1 行ずつ出す",
    )

    sp = sub_parser("recommend", "おすすめ順に出す", [common, filt, sel, gate])
    sp.add_argument("--auto-only", action="store_true", help="自動でおすすめに出せるものだけ")
    sp.add_argument(
        "--include-tools",
        action="store_true",
        help="深度・ポーズ・切り抜き・拡大などの補助ツールも含める(既定は「つくる」モデルだけ)",
    )
    sp.set_defaults(func=cmd_recommend)  # 既定の 10 件・marginal は cmd_recommend で決める(親の引数は共有)

    listing = argparse.ArgumentParser(add_help=False)
    listing.add_argument(
        "--sort",
        choices=SORTS,
        default="type",
        help="並べ方: type(既定。種類 → 段階 → 点数)、score(種類をまたいだ順位)、vram / disk(少ない順)、name",
    )
    listing.add_argument(
        "--all",
        action="store_true",
        help="全部出す(動かないもの・判定できないもの・補助ツール・ライセンス不明も。無検閲は --content all)",
    )
    listing.add_argument("--perfect", "-p", action="store_true", help="perfect だけ(--fit perfect と同じ)")

    sp = sub_parser("fit", "動くモデルの判定を表で出す(--all で全部)", [common, filt, sel, gate, listing])
    sp.add_argument(
        "--include-tools",
        action="store_true",
        help="深度・ポーズ・切り抜き・拡大などの補助ツールも含める(既定は「つくる」モデルだけ)",
    )
    sp.set_defaults(func=cmd_fit)

    sp = sub_parser(
        "search", "ID・名前の一部で探して、この PC での判定を出す", [common, filt, sel, gate, listing]
    )
    sp.add_argument("query", help="探す語(ID・名前の一部。大文字小文字は区別しない)")
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser(
        "types",
        parents=[common],
        help="大きな種類と、その中の用途(--task の名前)と数",
        description=HELP["types"],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sp.add_argument("--content", choices=CONTENT_FILTERS, default=DEFAULT_CONTENT)
    sp.set_defaults(func=cmd_types)

    sp = sub_parser("info", "1 つのモデルの詳細", [common, filt, gate])
    sp.add_argument("model", help="モデルの ID または名前の一部")
    sp.set_defaults(func=cmd_info)

    sp = sub_parser("update", "一覧を上流から作り直す", [common, gate])
    sp.add_argument("--offline-sources", metavar="DIR", help="(試験用)上流の代わりに使うファイル")
    sp.add_argument("--skip-hf", action="store_true", help="Hugging Face を見ない")
    sp.add_argument("--skip-head", action="store_true", help="URL の HEAD 確認を飛ばす")
    sp.add_argument(
        "--heads-only",
        action="store_true",
        help="今の一覧はそのままで、URL の HEAD と ComfyUI の最新の版の確認だけをやり直す",
    )
    sp.add_argument(
        "--inspect-only",
        action="store_true",
        help="今の一覧はそのままで、本体のファイルの見出し(精度)とテンプレートのサンプラーの組だけを"
        "読み直す(Range 要求。重み本体は落とさない。読んだ見出しはキャッシュに残る)",
    )
    sp.add_argument(
        "--offline",
        action="store_true",
        help="--inspect-only で通信せず、キャッシュにある見出しだけを使う",
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
