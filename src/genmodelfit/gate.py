"""自動でおすすめに出してよいかの判定(Issue #3)。

使う側(AI生成キット)は、人が確かめずにおすすめを出す。そのため判定は**保守的**にする:
分からないもの(ライセンス不明・HEAD 未確認・版不明など)はすべて「保留」。全部の条件を満たしたものだけ auto。

条件(id は機械で読む名前。変えない):

| id | 通る条件 |
|---|---|
| license | すべてのファイルのライセンス(本家のもの)が許可リストにある |
| gated | どのファイルも利用申請が要らない(gated=false。不明は保留) |
| official_workflow | ComfyUI 公式テンプレート(openSource が false でない) |
| custom_nodes | テンプレートがカスタムノードを要らない |
| comfyui_version | テンプレートの minComfyUIVersion が、使う ComfyUI の版以下 |
| url_reachable | すべてのファイルの URL が、匿名の HEAD で 200 / 302 / 307 を返した(update が記録) |
| sha256 | すべてのファイルに 64 桁の sha256 がある |
| content | 無検閲(uncensored)でない(allow_uncensored で外せる) |
| size_known | すべてのファイルの大きさが分かる |

判定の結果は一覧(db.json)に持たない(毎回ここで計算する)。持つのは update が記録した HEAD の結果だけ。
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hf import is_repackager
from .models import ModelEntry, ModelFile

CHECK_IDS = (
    "license",
    "gated",
    "official_workflow",
    "custom_nodes",
    "comfyui_version",
    "url_reachable",
    "sha256",
    "content",
    "size_known",
)

# 商用で使えて、使い方の制限が無いライセンス(理由は docs/gate.md)。
# openrail 系(使い方の制限)・非商用・売上の基準つきのコミュニティライセンス・llama / gemma の規約・
# other・不明は入れない
DEFAULT_LICENSE_ALLOWLIST: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "cc0-1.0",
        "cc-by-4.0",
        "unlicense",
    }
)

# 匿名の HEAD(転送を追わない)で「取れる」とみなす応答。docs/gate.md に実測を書いた:
# LFS のファイルは 302(CDN へ)、LFS でない小さいファイルは 307(/api/resolve-cache へ)、
# リポジトリの名前が変わったものも 307。gated は 401、無いファイルは 404
REACHABLE_STATUSES = frozenset({200, 302, 307})

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GatePolicy:
    """判定の方針。使う側が差し替える。

    comfyui_version: 使う ComfyUI の版。None なら一覧に記録された最新の版(update が GitHub から取る)
    """

    license_allowlist: frozenset[str] = DEFAULT_LICENSE_ALLOWLIST
    comfyui_version: str | None = None
    allow_uncensored: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "license_allowlist": sorted(self.license_allowlist),
            "comfyui_version": self.comfyui_version,
            "allow_uncensored": self.allow_uncensored,
        }


@dataclass
class Check:
    id: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "passed": self.passed, "detail": self.detail}


@dataclass
class GateResult:
    model_id: str
    checks: list[Check] = field(default_factory=list)
    comfyui_version: str | None = None  # 比べた ComfyUI の版

    @property
    def auto(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    @property
    def held_by(self) -> list[str]:
        return [c.id for c in self.checks if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "auto": self.auto,
            "held_by": self.held_by,
            "comfyui_version": self.comfyui_version,
            "checks": [c.to_dict() for c in self.checks],
        }


# --- 版の比較 -------------------------------------------------------------------------------------


def parse_version(v: str | None) -> tuple[int, ...] | None:
    """ "v0.42.1" → (0, 42, 1)。数字で始まらない部分があれば None(分からない版)。

    "0.3.60-rc1" のような後ろの印は数字の部分だけを見る(印の付いた版を少し新しく見ることになるが、
    テンプレートの minComfyUIVersion に印が付いた例は無い)。
    """
    if not isinstance(v, str):
        return None
    s = v.strip().lstrip("vV")
    if not s:
        return None
    out: list[int] = []
    for part in s.split("."):
        m = re.match(r"\d+", part)
        if not m:
            return None
        out.append(int(m.group()))
        if m.end() != len(part):  # "1-rc1" などは、ここまでで打ち切る
            break
    return tuple(out)


def version_le(a: str | None, b: str | None) -> bool | None:
    """a <= b か。どちらかが読めなければ None。足りない桁は 0 とみなす(0.3 == 0.3.0)。"""
    pa, pb = parse_version(a), parse_version(b)
    if pa is None or pb is None:
        return None
    n = max(len(pa), len(pb))
    return pa + (0,) * (n - len(pa)) <= pb + (0,) * (n - len(pb))


# --- 許可リスト -----------------------------------------------------------------------------------


def parse_allowlist(text: str) -> frozenset[str]:
    """ "mit, apache-2.0" や、1 行に 1 つ(# から後は注釈)の文字列を許可リストにする。"""
    items: set[str] = set()
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        for part in line.split(","):
            p = part.strip().lower()
            if p:
                items.add(p)
    return frozenset(items)


def load_allowlist(value: str) -> frozenset[str]:
    """CLI の --license-allowlist。ファイルのパスならその中身、そうでなければ「a,b,c」。"""
    p = Path(value)
    if p.is_file():
        return parse_allowlist(p.read_text(encoding="utf-8"))
    return parse_allowlist(value)


def _license_label(lic: str | None, name: str | None) -> str:
    if lic is None:
        return "不明"
    if lic == "other" and name:
        return f"other({name})"
    return lic


def license_allowed(lic: str | None, name: str | None, allow: Iterable[str]) -> bool:
    """ファイルのライセンスが許可リストにあるか。other は license_name で照合する
    (例: "mit-and-apache-2.0" を許可リストに足せば通せる)。"""
    allow = {a.lower() for a in allow}
    if lic is None:
        return False
    lic = lic.lower()
    if lic == "other":
        return bool(name) and name.lower() in allow
    return lic in allow


# --- ファイルのライセンスを、判定のために保守的に読む ----------------------------------------------
#
# 一覧の license は、再配布のリポジトリ(Comfy-Org など)の表記をそのまま持つことがある。再配布の中には
# 別のモデルの部品(テキストエンコーダなど)が入っていて、その部品の本当のライセンスはリポジトリの表記と違う
# (例: HiDream の llama_3.1_8b_instruct は Llama 3.1 の規約だが、リポジトリの表記は mit)。
# そこで判定では、次の順にファイルのライセンスを決める(一覧の値は書き換えない):
#   1. ファイル名が、制限の付いた部品(Llama・Gemma 1〜3・T5Gemma)なら、その規約(一覧の値より優先)
#   2. ライセンスを再配布側の表記からしか読めていない(license_from が再配布のリポジトリ)なら、
#      ファイル名が出どころの分かる寛容な部品(T5・UMT5・CLIP)のときだけそのライセンス、
#      それ以外は「確かめられない」
#   3. それ以外は一覧の値(本家から読んだもの)

# (ファイル名の正規表現, ライセンス, 根拠)。根拠の URL は docs/gate.md にもある
RESTRICTED_COMPONENTS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"llama|llava", re.IGNORECASE),
        "llama3.1",
        "Meta Llama の規約(https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)",
    ),
    (
        re.compile(r"t5gemma|gemma[-_ ]?[123](?![0-9])", re.IGNORECASE),
        "gemma",
        "Gemma 利用規約(https://ai.google.dev/gemma/terms)",
    ),
)
PERMISSIVE_COMPONENTS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"umt5[-_]?xxl", re.IGNORECASE), "apache-2.0", "https://huggingface.co/google/umt5-xxl"),
    (
        re.compile(r"(^|[^a-z0-9])t5(xxl|[-_]v1_1|[-_]base)", re.IGNORECASE),
        "apache-2.0",
        "https://huggingface.co/google/t5-v1_1-xxl",
    ),
    (re.compile(r"(^|[^a-z0-9])clip_l(?![a-z0-9])", re.IGNORECASE), "mit", "https://github.com/openai/CLIP"),
    (
        re.compile(r"(^|[^a-z0-9])clip_g(?![a-z0-9])", re.IGNORECASE),
        "mit",
        "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
    ),
    (
        re.compile(r"(^|[^a-z0-9])clip_vision_h(?![a-z0-9])", re.IGNORECASE),
        "mit",
        "https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ),
    (
        re.compile(r"(^|[^a-z0-9])wan_?2[._]1_vae", re.IGNORECASE),
        "apache-2.0",
        "https://huggingface.co/Wan-AI/Wan2.1-T2V-14B",
    ),
)


@dataclass
class ResolvedLicense:
    license: str | None  # None = 確かめられない
    license_name: str | None
    note: str | None = None  # 一覧の値と違う読み方をしたときの理由


def resolve_file_license(f: ModelFile) -> ResolvedLicense:
    base = f.name.rsplit("/", 1)[-1]
    for pat, lic, why in RESTRICTED_COMPONENTS:
        if pat.search(base):
            return ResolvedLicense(lic, None, f"ファイル名から {why}")
    src = f.license_from or f.repo_id
    if f.license is not None and src and is_repackager(src):
        for pat, lic, why in PERMISSIVE_COMPONENTS:
            if pat.search(base):
                return ResolvedLicense(lic, None, f"ファイル名から {why}")
        label = _license_label(f.license, f.license_name)
        return ResolvedLicense(None, None, f"再配布側({src})の表記 {label} だけで、本家を確かめられない")
    return ResolvedLicense(f.license, f.license_name)


# --- 条件ごとの判定 -------------------------------------------------------------------------------

_NO_FILES = "ファイルの記載がありません"


def _first(items: list[str], more: int) -> str:
    s = "、".join(items[:3])
    return s + (f" ほか {more - 3} 件" if more > 3 else "")


def check_license(m: ModelEntry, allow: frozenset[str]) -> Check:
    if not m.files:
        return Check("license", False, _NO_FILES)
    resolved = [(f, resolve_file_license(f)) for f in m.files]
    bad = [(f, r) for f, r in resolved if not license_allowed(r.license, r.license_name, allow)]
    if bad:
        items = [
            f"{f.name}: {_license_label(r.license, r.license_name)}" + (f"({r.note})" if r.note else "")
            for f, r in bad
        ]
        return Check("license", False, "許可リストに無いライセンス: " + _first(items, len(items)))
    labels = sorted({_license_label(r.license, r.license_name) for _, r in resolved})
    return Check("license", True, "すべて許可リストにある: " + ", ".join(labels))


def check_gated(m: ModelEntry) -> Check:
    if not m.files:
        return Check("gated", False, _NO_FILES)
    gated = [f.name for f in m.files if f.gated is True]
    unknown = [f.name for f in m.files if f.gated is None]
    if gated:
        return Check("gated", False, "利用申請(gated)が要る: " + _first(gated, len(gated)))
    if unknown:
        return Check("gated", False, "利用申請が要るか不明: " + _first(unknown, len(unknown)))
    return Check("gated", True, "利用申請は要らない")


def check_official(m: ModelEntry) -> Check:
    if m.source != "comfyui-template" or m.template is None:
        return Check("official_workflow", False, f"ComfyUI 公式テンプレートではない(出どころ: {m.source})")
    if m.template.open_source is False:
        return Check("official_workflow", False, "公式テンプレートだが openSource が false")
    note = "" if m.template.open_source else "(openSource の記載なし。ローカルで動くテンプレートとして扱う)"
    return Check("official_workflow", True, f"公式テンプレート {m.template.name}{note}")


def check_custom_nodes(m: ModelEntry) -> Check:
    if m.template is None:
        return Check("custom_nodes", False, "公式テンプレートではないため確かめられない")
    nodes = m.template.requires_custom_nodes
    if nodes:
        return Check("custom_nodes", False, "カスタムノードが要る: " + _first(list(nodes), len(nodes)))
    return Check("custom_nodes", True, "カスタムノードは要らない")


def check_comfyui_version(m: ModelEntry, target: str | None) -> Check:
    if m.template is None:
        return Check("comfyui_version", False, "公式テンプレートではないため確かめられない")
    need = m.template.min_comfyui_version
    if not target:
        return Check("comfyui_version", False, "使う ComfyUI の版が分からない(update で記録するか指定する)")
    if not need:
        return Check("comfyui_version", False, "テンプレートに必要な版(minComfyUIVersion)の記載がない")
    ok = version_le(need, target)
    if ok is None:
        return Check("comfyui_version", False, f"版を読めない: 必要 {need} / 使う {target}")
    if not ok:
        return Check("comfyui_version", False, f"ComfyUI {need} 以上が要る(使う版 {target})")
    return Check("comfyui_version", True, f"必要 {need} ≦ 使う版 {target}")


def check_url(m: ModelEntry) -> Check:
    if not m.files:
        return Check("url_reachable", False, _NO_FILES)
    no_url = [f.name for f in m.files if not f.url]
    unchecked = [f.name for f in m.files if f.url and f.head_status is None]
    bad = [
        f"{f.name}: HTTP {f.head_status}"
        for f in m.files
        if f.url and f.head_status is not None and f.head_status not in REACHABLE_STATUSES
    ]
    if no_url:
        return Check("url_reachable", False, "URL が無い: " + _first(no_url, len(no_url)))
    if bad:
        return Check("url_reachable", False, "匿名で取れない: " + _first(bad, len(bad)))
    if unchecked:
        return Check("url_reachable", False, "未確認(HEAD の記録なし): " + _first(unchecked, len(unchecked)))
    codes = sorted({f.head_status for f in m.files if f.head_status is not None})
    return Check("url_reachable", True, "匿名の HEAD で取れる: " + ", ".join(str(c) for c in codes))


def check_sha256(m: ModelEntry) -> Check:
    if not m.files:
        return Check("sha256", False, _NO_FILES)
    bad = [f.name for f in m.files if not (isinstance(f.sha256, str) and _SHA256.match(f.sha256.lower()))]
    if bad:
        return Check("sha256", False, "sha256 が無い: " + _first(bad, len(bad)))
    return Check("sha256", True, "すべてのファイルに sha256 がある")


def check_content(m: ModelEntry, allow_uncensored: bool) -> Check:
    if m.content == "sfw":
        return Check("content", True, "無検閲の印は無い")
    if m.content == "uncensored" and allow_uncensored:
        return Check("content", True, "無検閲(方針で自動のおすすめを許可)")
    if m.content == "uncensored":
        return Check(
            "content", False, "無検閲のモデルは初心者向けの自動おすすめに出さない(一覧・絞り込みには出る)"
        )
    return Check("content", False, f"内容の印が分からない: {m.content}")


def check_size(m: ModelEntry) -> Check:
    if not m.files:
        return Check("size_known", False, _NO_FILES)
    if getattr(m, "files_complete", None) is False:
        # テンプレートがモデルのファイルを全部は書いていない(カスタムノードが自分で取るなど)
        return Check("size_known", False, "ファイルの一覧が完全でない(files_complete=false)")
    unknown = [f.name for f in m.files if not f.size_bytes or f.size_bytes <= 0]
    if unknown:
        return Check("size_known", False, "大きさが分からない: " + _first(unknown, len(unknown)))
    total = m.total_size_bytes or 0
    return Check("size_known", True, f"合計 {total / 1e9:.2f} GB")


# --- まとめ ---------------------------------------------------------------------------------------


def recorded_comfyui_version(sources: dict[str, Any] | None) -> str | None:
    """update が一覧に記録した ComfyUI の最新の版(無ければ None)。"""
    c = (sources or {}).get("comfyui")
    v = c.get("latest_release") if isinstance(c, dict) else None
    return v if isinstance(v, str) and v else None


def evaluate_offline(
    model: ModelEntry, policy: GatePolicy | None = None, *, sources: dict[str, Any] | None = None
) -> GateResult:
    """一覧に記録された情報だけで判定する(ネットには触らない)。

    sources: 一覧の `sources`(db.sources)。policy.comfyui_version が None のとき、ここに記録された
    最新の版を使う。どちらも無ければ comfyui_version は保留になる。
    """
    p = policy or GatePolicy()
    target = p.comfyui_version or recorded_comfyui_version(sources)
    checks = [
        check_license(model, p.license_allowlist),
        check_gated(model),
        check_official(model),
        check_custom_nodes(model),
        check_comfyui_version(model, target),
        check_url(model),
        check_sha256(model),
        check_content(model, p.allow_uncensored),
        check_size(model),
    ]
    return GateResult(model.id, checks, target)


def summarize(results: Iterable[GateResult]) -> dict[str, Any]:
    """件数のまとめ: auto / held、落ちた条件ごとの件数、最初に落ちた条件ごとの件数。"""
    res = list(results)
    failing = {cid: 0 for cid in CHECK_IDS}
    first = {cid: 0 for cid in CHECK_IDS}
    for r in res:
        held = r.held_by
        for cid in held:
            failing[cid] = failing.get(cid, 0) + 1
        if held:
            first[held[0]] = first.get(held[0], 0) + 1
    auto = sum(1 for r in res if r.auto)
    return {
        "models": len(res),
        "auto": auto,
        "held": len(res) - auto,
        "failing": failing,
        "first_failing": first,
    }
