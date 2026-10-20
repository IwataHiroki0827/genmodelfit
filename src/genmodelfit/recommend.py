"""おすすめ順に並べる。

並べ方(Issue #17): 動くもの(too_tight でない)→ 自動でおすすめに出せるもの(gate の auto)→ 点数の高い順 → ID。
点数(0〜100)は 5 つの内訳の重み付きの和(式は docs/schema.md「Score」)。VRAM・メモリを増やしても下がらない。
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .db import Database
from .estimate import GIB, Coefficients, estimate, load_coefficients
from .models import CONTENT_LEVELS, FIT_LEVELS, MODALITIES, Estimate, ModelEntry, Settings, SystemInfo

if TYPE_CHECKING:
    from .gate import GatePolicy

# 内容の絞り込み。既定は sfw(無検閲は all / uncensored を指定したときだけ出す)
CONTENT_FILTERS = ("all", *CONTENT_LEVELS)
DEFAULT_CONTENT = "sfw"

FIT_LABEL = {"perfect": "Perfect", "good": "Good", "marginal": "Marginal", "too_tight": "Too tight"}

# 点数の内訳の重み(合計 1)。docs/schema.md「Score」と合わせる
WEIGHTS = {"fit": 0.45, "headroom": 0.15, "readiness": 0.20, "basis": 0.10, "popularity": 0.10}
FIT_POINTS = {"perfect": 100, "good": 70, "marginal": 35, "too_tight": 0}
# 準備のできぐあい: gate の条件ごとの重み(通った条件の和。全部通れば 100)
READINESS_POINTS = {
    "license": 25,
    "gated": 15,
    "official_workflow": 15,
    "custom_nodes": 15,
    "comfyui_version": 10,
    "url_reachable": 5,
    "sha256": 5,
    "content": 5,
    "size_known": 5,
}
HEADROOM_FULL = 0.5  # 全部載せで VRAM の半分以上が空けば余裕 100


def headroom(e: Estimate, system: SystemInfo) -> float:
    """VRAM の余裕(0〜100)。全部載せの VRAM(PC によらない値)と比べるので、VRAM が増えれば下がらない。"""
    v = system.vram_gb
    if v <= 0 or e.unknown_size or e.vram_gb <= 0 or e.vram_gb >= v:
        return 0.0
    return min(100.0, 100.0 * (1 - e.vram_gb / v) / HEADROOM_FULL)


def popularity(m: ModelEntry) -> float:
    """使われた回数(テンプレート)か HF のダウンロード数(÷100)。10 万で 100。"""
    usage = (m.template.usage or 0) if m.template else 0
    if not m.template and m.downloads:
        usage = m.downloads / 100  # HF のダウンロード数を、テンプレートの使用回数と同じくらいの目盛りに直す
    return min(100.0, 20 * math.log10(1 + usage))


def readiness(gate: dict[str, Any] | None) -> float:
    if not gate:
        return 0.0
    if gate.get("auto"):
        return 100.0
    return float(sum(READINESS_POINTS.get(c.get("id"), 0) for c in gate.get("checks", []) if c.get("passed")))


def score_components(
    m: ModelEntry, e: Estimate, system: SystemInfo, gate: dict[str, Any] | None = None
) -> dict[str, float]:
    return {
        "fit": float(FIT_POINTS[e.fit]),
        "headroom": round(headroom(e, system), 1),
        "readiness": readiness(gate),
        "basis": 100.0 if e.basis == "measured" else 0.0,
        "popularity": round(popularity(m), 1),
    }


def total_score(components: dict[str, float]) -> float:
    return round(sum(WEIGHTS[k] * components[k] for k in WEIGHTS), 1)


def _sum_licenses(m: ModelEntry) -> list[str | None]:
    """ファイルごとのライセンス(重ならないように。不明は None で最後)。ファイルが無ければ一式のもの。"""
    if not m.files:
        return [m.license]
    found = {f.license for f in m.files}
    return sorted((x for x in found if x is not None), key=str) + ([None] if None in found else [])


def license_matches(m: ModelEntry, allowed: Iterable[str] | None) -> bool:
    """すべてのファイルのライセンスが allowed にあるか(不明は合わない)。allowed が None なら絞らない。"""
    if allowed is None:
        return True
    allow = {a.strip().lower() for a in allowed if a.strip()}
    lics = _sum_licenses(m)
    return bool(lics) and all(x is not None and x.lower() in allow for x in lics)


def mode_label(e: Estimate, system: SystemInfo | None = None) -> str:
    """表の Mode と JSON の run_mode_label。GPU / GPU~(部品を順に載せ替え)/ GPU+RAM / CPU / -。"""
    if e.unknown_size or e.run_mode == "none":
        return "-"
    if e.run_mode == "cpu":
        return "CPU"
    if e.run_mode == "offload":
        return "GPU+RAM"
    v = system.vram_gb if system is not None else 0.0
    seq = e.vram_sequential_gb
    if v and e.vram_gb > v and seq is not None and seq <= v:
        return "GPU~"
    return "GPU"


def _ram_short(e: Estimate, system: SystemInfo | None) -> bool:
    return bool(system is not None and system.ram_total_gb and e.ram_gb > system.ram_total_gb)


def vram_need(e: Estimate, system: SystemInfo | None) -> float | None:
    """判定の根拠にした VRAM(GiB)。CPU・大きさ不明・GPU が無いときは None。"""
    vram = system.vram_gb if system is not None else 0.0
    if e.unknown_size or e.run_mode == "cpu" or vram <= 0:
        return None
    seq = e.vram_sequential_gb
    if e.fit == "too_tight":
        # VRAM は足りていてメモリが理由なら、VRAM の側は判定に使った載せ方の値を出す
        if _ram_short(e, system) and e.vram_offload_gb <= vram:
            if e.vram_gb <= vram:
                return e.vram_gb
            if seq is not None and seq <= vram:
                return seq
        return e.vram_offload_gb  # 動かないときは「最低これだけ要る」を出す
    if e.run_mode == "offload":
        return e.vram_offload_gb
    if e.vram_gb > vram and seq is not None and seq <= vram:
        return seq
    return e.vram_gb


def memory_use(e: Estimate, system: SystemInfo | None) -> tuple[str, float | None, float | None]:
    """(種類, 要る GiB, ある GiB)。GPU で動かすなら VRAM、CPU で動くならメインメモリ。
    GPU が要るのに無いときは VRAM の最低限と 0。"""
    has_gpu = system is not None and system.vram_gb > 0
    if e.unknown_size:
        return ("vram" if has_gpu else "ram"), None, None
    if e.run_mode == "cpu":
        return "ram", e.ram_gb, system.ram_total_gb if system is not None else None
    need = vram_need(e, system)
    if need is None:  # GPU が要るのに無い
        return "vram", e.vram_offload_gb, 0.0
    return "vram", need, system.vram_gb if system is not None else None


@dataclass
class Fit:
    model: ModelEntry
    estimate: Estimate
    score: float = 0.0
    gate: dict[str, Any] | None = None  # 自動でおすすめに出せるか(#3 の gate.evaluate の結果)
    score_components: dict[str, float] = field(default_factory=dict)
    system: SystemInfo | None = field(default=None, repr=False, compare=False)

    @property
    def auto(self) -> bool:
        return bool(self.gate and self.gate.get("auto"))

    @property
    def run_mode_label(self) -> str:
        return mode_label(self.estimate, self.system)

    def to_dict(self, full: bool = False) -> dict[str, Any]:
        """JSON の 1 件(CLI の recommend / fit / search の models と同じ形。docs/schema.md)。"""
        m, e = self.model, self.estimate
        kind, need, avail = memory_use(e, self.system)
        util = round(100 * need / avail, 1) if need is not None and avail else None
        d: dict[str, Any] = {
            "id": m.id,
            "name": m.name,
            "modality": m.modality,
            "tasks": m.tasks,
            "content": m.content,
            "source": m.source,
            "fit_level": e.fit,
            "fit_label": FIT_LABEL[e.fit],
            "run_mode": e.run_mode,
            "run_mode_label": self.run_mode_label,
            "score": round(self.score, 1),
            "score_components": dict(self.score_components),
            "memory_kind": kind,
            "memory_required_gb": round(need, 2) if need is not None else None,
            "memory_available_gb": round(avail, 2) if avail is not None else None,
            "utilization_pct": util,
            "ram_required_gb": None if e.unknown_size else e.ram_gb,
            "disk_size_gb": e.disk_gb,
            "total_size_gb": round((m.total_size_bytes or 0) / GIB, 2) or None,
            "license": m.license,
            "licenses": _sum_licenses(m),
            "auto": self.auto,
            "held_reasons": list((self.gate or {}).get("held_by", [])),
            "estimate_confidence": e.basis,
            "notes": list(e.reasons),
            "files_count": len(m.files),
            "estimate": e.to_dict(),
            "gate": self.gate,
        }
        if full:
            d["model"] = m.to_dict()
        return d


def check_filters(
    *, modality: str | None = None, content: str = DEFAULT_CONTENT, min_fit: str | None = None
) -> None:
    """絞り込みの値を確かめる。知らない値は ValueError(組み込み側が誤りに気づけるように)。"""
    if modality is not None and modality not in MODALITIES:
        raise ValueError(f"modality が正しくありません: {modality!r}(使えるのは {', '.join(MODALITIES)})")
    if content not in CONTENT_FILTERS:
        raise ValueError(f"content が正しくありません: {content!r}(使えるのは {', '.join(CONTENT_FILTERS)})")
    if min_fit is not None and min_fit not in FIT_LEVELS:
        raise ValueError(f"min_fit が正しくありません: {min_fit!r}(使えるのは {', '.join(FIT_LEVELS)})")


def sort_key(f: Fit) -> tuple:
    return (f.estimate.fit == "too_tight", not f.auto, -round(f.score, 1), f.model.id)


def _gate(m: ModelEntry, policy: GatePolicy | None, db: Database) -> dict[str, Any]:
    from .gate import evaluate_offline

    return evaluate_offline(m, policy, sources=db.sources).to_dict()


def judge(
    m: ModelEntry,
    system: SystemInfo,
    db: Database,
    *,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    with_limits: bool = False,
    policy: GatePolicy | None = None,
) -> Fit:
    """1 モデルの判定・gate・点数。"""
    c = coefficients or load_coefficients()
    e = estimate(m, system, settings, db.measurements, c, with_limits=with_limits)
    g = _gate(m, policy, db)
    comps = score_components(m, e, system, g)
    return Fit(m, e, total_score(comps), g, comps, system)


def evaluate(
    db: Database,
    system: SystemInfo,
    *,
    modality: str | None = None,
    content: str = DEFAULT_CONTENT,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    models: Iterable[ModelEntry] | None = None,
    with_limits: bool = False,
    policy: GatePolicy | None = None,
) -> list[Fit]:
    """種類と内容で絞って全部を判定し、おすすめ順に並べる(too_tight も含む)。"""
    check_filters(modality=modality, content=content)
    c = coefficients or load_coefficients()
    out = [
        judge(m, system, db, settings=settings, coefficients=c, with_limits=with_limits, policy=policy)
        for m in (models if models is not None else db.models)
        if (not modality or m.modality == modality) and (content == "all" or m.content == content)
    ]
    out.sort(key=sort_key)
    return out


# 「つくる」モデルではない補助ツールの作業(recommend は既定で外す。fit・search・info には出る)
TOOL_TASKS = frozenset({"tool", "estimate", "matting", "upscale", "interp", "segment"})


def is_tool(m: ModelEntry) -> bool:
    return bool(m.tasks) and all(t in TOOL_TASKS for t in m.tasks)


def matches_query(m: ModelEntry, query: str) -> bool:
    q = query.strip().lower()
    return q in m.id.lower() or q in m.name.lower()


@dataclass
class Selection:
    """絞り込みの結果。hidden は外した理由ごとの件数(1 件は最初に当たった理由で 1 回だけ数える)。"""

    fits: list[Fit]
    matched: int  # 件数(limit)で切る前の数
    hidden: dict[str, int]


# hidden のキー(数える順)。docs/schema.md
HIDDEN_KEYS = (
    "modality",
    "uncensored",
    "sfw",
    "tools",
    "license",
    "size_unknown",
    "too_tight",
    "below_min_fit",
    "held",
)


def select(
    db: Database,
    system: SystemInfo,
    *,
    query: str | None = None,
    modality: str | None = None,
    content: str = DEFAULT_CONTENT,
    min_fit: str | None = None,
    license: Iterable[str] | None = None,
    include_tools: bool = True,
    auto_only: bool = False,
    drop_unknown_size: bool = False,
    limit: int | None = None,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    with_limits: bool = False,
    policy: GatePolicy | None = None,
) -> Selection:
    """recommend / fit / search の共通の絞り込み。query に合わないものは hidden に数えない。"""
    check_filters(modality=modality, content=content, min_fit=min_fit)
    allow = list(license) if license is not None else None
    hidden = dict.fromkeys(HIDDEN_KEYS, 0)
    c = coefficients or load_coefficients()
    allowed = FIT_LEVELS[: FIT_LEVELS.index(min_fit) + 1] if min_fit else FIT_LEVELS
    out: list[Fit] = []
    for m in db.models:
        if query is not None and not matches_query(m, query):
            continue
        if modality and m.modality != modality:
            hidden["modality"] += 1
            continue
        if content != "all" and m.content != content:
            hidden[m.content] = hidden.get(m.content, 0) + 1
            continue
        if not include_tools and is_tool(m):
            hidden["tools"] += 1
            continue
        if not license_matches(m, allow):
            hidden["license"] += 1
            continue
        f = judge(m, system, db, settings=settings, coefficients=c, with_limits=with_limits, policy=policy)
        e = f.estimate
        if e.unknown_size and (drop_unknown_size or e.fit not in allowed):
            hidden["size_unknown"] += 1
            continue
        if e.fit not in allowed:
            hidden["too_tight" if e.fit == "too_tight" else "below_min_fit"] += 1
            continue
        if auto_only and not f.auto:
            hidden["held"] += 1
            continue
        out.append(f)
    out.sort(key=sort_key)
    matched = len(out)
    return Selection(out[:limit] if limit else out, matched, {k: v for k, v in hidden.items() if v})


def recommend(
    db: Database,
    system: SystemInfo,
    *,
    modality: str | None = None,
    content: str = DEFAULT_CONTENT,
    min_fit: str = "marginal",
    license: Iterable[str] | None = None,
    include_tools: bool = False,
    auto_only: bool = False,
    limit: int | None = 10,
    settings: Settings | None = None,
    coefficients: Coefficients | None = None,
    with_limits: bool = False,
    policy: GatePolicy | None = None,
) -> list[Fit]:
    """おすすめ順(大きさが分からないものは出さない)。CLI の recommend と同じ既定。"""
    return select(
        db,
        system,
        modality=modality,
        content=content,
        min_fit=min_fit,
        license=license,
        include_tools=include_tools,
        auto_only=auto_only,
        drop_unknown_size=True,
        limit=limit,
        settings=settings,
        coefficients=coefficients,
        with_limits=with_limits,
        policy=policy,
    ).fits
