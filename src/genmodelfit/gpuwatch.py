# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""GPU の登録の表(gputables.py)の見張り(#103。docs/release.md「毎週の一覧の更新」)。

毎週の更新(weekly.py)と、手で動かす `python -m genmodelfit.gpuwatch` からだけ呼ぶ。読むだけのコマンドと
ライブラリの関数からは呼ばない(ネットワークに触らない約束。tests/test_privacy.py)。

公式の一覧(pci.ids・ROCm の hw specs と互換表・TheRock の SUPPORTED_GPUS / RELEASES・PyTorch の
Intel GPU の表・NVIDIA の cuda-gpus / legacy・PyTorch の最新の版と wheel の索引・CUDA のリリースノートの表・
ComfyUI の最新のタグの model_management.py)を取り、同じ形(スナップショット)に直して、
- 表と照らす(公式の製品の gfx・compute capability が表と合うか、表に無い gfx・CUDA の版が無いか)
- 前回の一覧(tests/fixtures/gpu_official。表と一緒に直す)と比べる(新しい型番 ID・製品名・状態の変化)
の 2 つで「表に無い」「表と違う」候補を返す。表は書き換えない(人が一次資料で確かめて足す)。読めなかった出典は
候補とは別に返す。何が起きても例外を外に出さない(毎週の更新を止めない)。

    python -m genmodelfit.gpuwatch [--json]                     # 取って照らす(通信する)
    python -m genmodelfit.gpuwatch --save-fixtures DIR          # 取ったスナップショットを DIR に書く
    python -m genmodelfit.gpuwatch --from-dir RAW [--save-fixtures DIR]   # 保存した生の応答から(通信しない)
"""

from __future__ import annotations

import argparse
import ast
import html
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from . import hardware as hw
from .gputables import (
    KNOWN_EXCEPTIONS,
    PYTORCH_ROCM_LINUX,
    SOLO_CC,
    SOURCES,
    TORCH_CUDA_BUILDS,
    TORCH_RELEASE,
)

# 取る URL(出典の名前 → URL)。{tag} は最新の版のタグで埋める
URLS: dict[str, str] = {
    "pci_ids": SOURCES["pci-ids"].url,
    "rocm_hw_specs": SOURCES["rocm-hw-specs"].url,
    "radeon_win_compat": SOURCES["radeon-win-compat"].url,
    "ryzen_win_compat": SOURCES["ryzen-win-compat"].url,
    "radeon_linux_compat": SOURCES["radeon-linux-compat"].url,
    "ryzen_linux_compat": SOURCES["ryzen-linux-compat"].url,
    "therock_supported": "https://raw.githubusercontent.com/ROCm/TheRock/main/SUPPORTED_GPUS.md",
    "therock_releases": "https://raw.githubusercontent.com/ROCm/TheRock/main/RELEASES.md",
    "intel_xpu": "https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html",
    "nvidia_cuda_gpus": SOURCES["nvidia-cuda-gpus"].url,
    "nvidia_cuda_legacy": SOURCES["nvidia-cuda-legacy"].url,
    "cuda_rn": SOURCES["cuda-rn"].url,
    "torch_release": "https://api.github.com/repos/pytorch/pytorch/releases/latest",
    "torch_index": SOURCES["torch-whl-index"].url,
    # build_cuda.sh の TORCH_CUDA_ARCH_LIST は、2.14 から torch/cuda/__init__.py の
    # PYTORCH_RELEASES_CODE_CC に写され、試験で一致を見ている(その注記による)。v2.14.1 の build_cuda.sh は
    # 404 なのでこちらを読む
    "torch_cuda_init": "https://raw.githubusercontent.com/pytorch/pytorch/{tag}/torch/cuda/__init__.py",
    "torch_build_rocm": "https://raw.githubusercontent.com/pytorch/pytorch/{tag}/.ci/docker/manywheel/build.sh",
    "comfyui_release": "https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest",
    "comfyui_mm": "https://raw.githubusercontent.com/Comfy-Org/ComfyUI/{tag}/comfy/model_management.py",
}
# 比べるスナップショットの名前(fixtures の <名前>.json)。torch_release / comfyui_release はタグを得るだけ
SNAPSHOTS = (
    "pci_ids",
    "rocm_hw_specs",
    "radeon_win_compat",
    "ryzen_win_compat",
    "radeon_linux_compat",
    "ryzen_linux_compat",
    "therock_supported",
    "therock_releases",
    "intel_xpu",
    "nvidia",
    "cuda_rn",
    "torch",
    "torch_index",
    "comfyui",
)
TIMEOUT = (10, 60)  # (接続, 読み込み) 秒。fetch の既定と同じ
BASELINE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "gpu_official"


@dataclass
class Candidate:
    source: str  # スナップショットの名前
    kind: str  # "not in tables"(表に無い) / "differs"(表と違う)
    item: str
    detail: str


@dataclass
class WatchResult:
    candidates: list[Candidate] = field(default_factory=list)
    unreadable: list[dict[str, str]] = field(default_factory=list)  # {source, url, error}
    read: list[str] = field(default_factory=list)
    baseline: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [asdict(c) for c in self.candidates],
            "unreadable": self.unreadable,
            "read": self.read,
            "baseline": self.baseline,
        }


# --- HTML の表 ----------------------------------------------------------------------------------------


class _Tables(HTMLParser):
    """HTML の <table> を、見出し(caption)と行(セルの文字の並び)にする。セルの中の <br>・</p>・</li> で
    区切った項目も残す(1 つのセルに製品名が並ぶ表がある)。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[str, Any]] = []
        self._stack: list[dict[str, Any]] = []
        self._cell: list[str] | None = None
        self._items: list[str] = []
        self._caption: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag == "table":
            self._stack.append({"caption": "", "rows": []})
        elif not self._stack:
            return
        elif tag == "caption":
            self._caption = []
        elif tag == "tr":
            self._stack[-1]["rows"].append([])
        elif tag in ("td", "th"):
            self._cell, self._items = [], []
        elif tag in ("br", "li", "p") and self._cell is not None:
            self._split()

    def _split(self) -> None:
        if self._cell is not None:
            text = _clean("".join(self._cell))
            if text:
                self._items.append(text)
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if not self._stack:
            return
        if tag == "table":
            self.tables.append(self._stack.pop())
        elif tag == "caption" and self._caption is not None:
            self._stack[-1]["caption"] = _clean("".join(self._caption))
            self._caption = None
        elif tag in ("td", "th") and self._cell is not None:
            self._split()
            rows = self._stack[-1]["rows"]
            if not rows:
                rows.append([])
            rows[-1].append({"text": " ".join(self._items), "items": list(self._items)})
            self._cell = None
        elif tag in ("p", "li") and self._cell is not None:
            self._split()

    def handle_data(self, data: str) -> None:
        if self._caption is not None:
            self._caption.append(data)
        if self._cell is not None:
            self._cell.append(data)


def _clean(text: str) -> str:
    text = html.unescape(text).replace("®", "").replace("™", "").replace("*", "")
    return " ".join(text.split())


def html_tables(text: str) -> list[dict[str, Any]]:
    p = _Tables()
    p.feed(text)
    p.close()
    return p.tables


def _header_index(rows: list[list[dict]], *names: str) -> tuple[int, dict[str, int]] | None:
    """見出しの行(names の言葉を全部含む最初の行)の位置と、言葉 → 列の番号。"""
    for i, row in enumerate(rows[:3]):
        texts = [c["text"].lower() for c in row]
        found = {}
        for n in names:
            # 同じ言葉の列(「Name」)を、言葉を含む列(「LLVM target name」)より先に取る
            hit = [j for j, t in enumerate(texts) if t == n.lower()] or [
                j for j, t in enumerate(texts) if n.lower() in t
            ]
            if not hit:
                break
            found[n] = hit[0]
        else:
            return i, found
    return None


# --- 出典ごとの読み方(生の応答 → スナップショット) ------------------------------------------------------

_AMD_KW = re.compile(
    r"navi\s*\d+|phoenix|hawk|strix|krackan|gorgon|medusa|rembrandt|raphael|granite ridge|mendocino|"
    r"van ?gogh|sephiroth|aqua vanjaram|aldebaran|arcturus|vega 20|steam machine",
    re.I,
)
_INT_KW = re.compile(
    r"\bdg2\b|alchemist|battlemage|celestial|lunar lake|panther lake|arrow lake|meteor lake|wildcat lake|"
    r"nova lake|ponte vecchio|data center gpu max|\barc\b",
    re.I,
)
_NOT_GPU = re.compile(
    r"usb|audio|root port|controller|switch|bridge|smbus|spi|uart|i2c|sram|heci|sensor|wifi|thunderbolt|"
    r"trace|espi|lpc|ufs|gaussian|redirection|sata|xhci|ehci|fabric|iommu|dummy|ccp|asp|nhi|pmc|gna|npu|"
    r"ipu|vpu|imgu|ish|cnvi|thermal|camera|dma|psp|function",
    re.I,
)


def parse_pci_ids(text: str) -> dict[str, Any]:
    """pci.ids → {"version", "devices": {"1002:7499": "Navi 33 [Radeon RX 7400/7300/Pro W7400]"}}
    (AMD・Intel の GPU らしい行だけ。言葉で選ぶ)。"""
    m = re.search(r"^#\s*Version:\s*(\S+)", text, re.M)
    devices: dict[str, str] = {}
    vendor = None
    for line in text.splitlines():
        if line.startswith("C "):
            break
        v = re.match(r"^([0-9a-f]{4})  ", line)
        if v:
            vendor = v.group(1)
            continue
        d = re.match(r"^\t([0-9a-f]{4})  (.*)", line)
        if not d or vendor not in ("1002", "8086"):
            continue
        name = d.group(2).strip()
        kw = _AMD_KW if vendor == "1002" else _INT_KW
        if kw.search(name) and not _NOT_GPU.search(name):
            devices[f"{vendor}:{d.group(1)}"] = name
    if not devices:
        raise ValueError("no GPU lines")
    return {"version": m.group(1) if m else "", "devices": devices}


def parse_rocm_hw_specs(text: str) -> list[dict[str, str]]:
    out = []
    for t in html_tables(text):
        h = _header_index(t["rows"], "name", "llvm target")
        if h is None:
            continue
        i, cols = h
        gm = _header_index(t["rows"], "graphics model")
        for row in t["rows"][i + 1 :]:
            cells = [c["text"] for c in row]
            if len(cells) <= cols["llvm target"]:
                continue
            gfx = cells[cols["llvm target"]]
            if not re.fullmatch(r"gfx[0-9a-z]+", gfx):
                continue
            item = {"name": cells[cols["name"]], "gfx": gfx}
            if gm is not None:
                item["graphics_model"] = cells[gm[1]["graphics model"]]
            out.append(item)
    if not out:
        raise ValueError("no table with 'LLVM target name'")
    return out


def parse_compat(text: str) -> list[dict[str, Any]]:
    """Radeon / Ryzen の Windows / Linux の互換表 → [{"rocm", "gfx": [...], "products": [...]}]。"""
    out = []
    for t in html_tables(text):
        h = _header_index(t["rows"], "rocm version", "supported amd")
        if h is None:
            continue
        i, cols = h
        arch = _header_index(t["rows"], "supported architectures")
        for row in t["rows"][i + 1 :]:
            if len(row) <= cols["supported amd"]:
                continue
            cell = row[cols["supported amd"]]
            items = cell["items"]
            if len(items) <= 1:  # 1 つの段落に並べた表(「AMD Radeon RX 9070 AMD Radeon RX 9070 XT …」)
                items = [x.strip() for x in re.split(r"(?=\bAMD )", cell["text"]) if x.strip()]
            gfx = re.findall(r"gfx[0-9a-z]+", row[arch[1]["supported architectures"]]["text"]) if arch else []
            out.append({"rocm": row[cols["rocm version"]]["text"], "gfx": gfx, "products": items})
    if not out:
        raise ValueError("no table with 'Supported AMD …'")
    return out


def _md_tables(text: str) -> list[tuple[str, list[str], list[list[str]]]]:
    """Markdown の表 → (直前の見出しの道筋, 見出しの列, 行)。"""
    out = []
    heads: dict[int, str] = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = re.match(r"^(#{1,6})\s+(.*)", lines[i])
        if m:
            level = len(m.group(1))
            heads = {k: v for k, v in heads.items() if k < level}
            heads[level] = m.group(2).strip()
        if (
            lines[i].lstrip().startswith("|")
            and i + 1 < len(lines)
            and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1])
        ):
            header = [c.strip().strip("*").strip() for c in lines[i].strip().strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                rows.append([c.strip().strip("*").strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            out.append((" / ".join(heads[k] for k in sorted(heads)), header, rows))
            i = j
            continue
        i += 1
    return out


def parse_therock_supported(text: str) -> list[dict[str, Any]]:
    out = []
    for where, header, rows in _md_tables(text):
        low = [h.lower() for h in header]
        if "llvm target" not in low:
            continue
        os_name = "windows" if "windows" in where.lower() else "linux"
        col = {
            k: low.index(k)
            for k in ("architecture", "llvm target", "build passing", "sanity tested", "release ready")
            if k in low
        }
        for r in rows:
            if len(r) < len(header):
                continue
            out.append(
                {
                    "os": os_name,
                    "section": where.split(" / ")[-1],
                    "arch": r[col["architecture"]],
                    "gfx": r[col["llvm target"]],
                    "build": "✅" in r[col.get("build passing", 0)],
                    "sanity": "✅" in r[col["sanity tested"]] if "sanity tested" in col else False,
                    "release": "✅" in r[col["release ready"]] if "release ready" in col else False,
                }
            )
    if not out:
        raise ValueError("no table with 'LLVM target'")
    return out


def parse_therock_releases(text: str) -> list[dict[str, str]]:
    out = []
    for _where, header, rows in _md_tables(text):
        low = [h.lower() for h in header]
        if "gfx target" not in low or "product name" not in low:
            continue
        for r in rows:
            gfx = r[low.index("gfx target")]
            if re.fullmatch(r"gfx[0-9a-z]+", gfx):
                out.append({"product": r[low.index("product name")].strip("*"), "gfx": gfx})
    if not out:
        raise ValueError("no 'GFX target lookup table'")
    return out


def parse_intel_xpu(text: str) -> dict[str, Any]:
    client, datacenter = [], []
    for t in html_tables(text):
        h = _header_index(t["rows"], "supported os", "validated hardware")
        if h is not None:
            i, cols = h
            for row in t["rows"][i + 1 :]:
                if len(row) > cols["validated hardware"]:
                    client.append(
                        {
                            "os": row[cols["supported os"]]["text"],
                            "hardware": row[cols["validated hardware"]]["items"],
                        }
                    )
            continue
        h = _header_index(t["rows"], "device")
        if h is not None:
            i, _cols = h
            heads = [c["text"] for c in t["rows"][i]]
            for row in t["rows"][i + 1 :]:
                oses = [
                    heads[j]
                    for j, c in enumerate(row)
                    if j > 0 and c["text"].lower() == "yes" and j < len(heads)
                ]
                datacenter.append({"device": row[0]["text"], "os": oses})
    if not client:
        raise ValueError("no 'Validated Hardware' table")
    return {"client": client, "datacenter": datacenter}


def parse_nvidia(text: str, source: str) -> list[dict[str, str]]:
    out = []
    for t in html_tables(text):
        rows = t["rows"]
        h = _header_index(rows, "compute capability")
        if h is None:
            continue
        i, cols = h
        heads = [c["text"] for c in rows[i]]
        for row in rows[i + 1 :]:
            if not row:
                continue
            cc = row[cols["compute capability"]]["text"]
            if not re.fullmatch(r"\d+\.\d", cc):
                continue
            for j, cell in enumerate(row):
                if j == cols["compute capability"]:
                    continue
                for name in cell["items"]:
                    out.append(
                        {
                            "source": source,
                            "segment": heads[j] if j < len(heads) else "",
                            "cc": cc,
                            "name": name,
                        }
                    )
    if not out:
        raise ValueError("no table with 'Compute Capability'")
    return out


def parse_cuda_rn(text: str) -> dict[str, int]:
    """CUDA のリリースノートの「Minimum Required Driver Version for CUDA Minor Version Compatibility」の表 →
    {"13": 580, "12": 525, ...}。"""
    for t in html_tables(text):
        if "minimum required driver version for cuda minor version" not in t["caption"].lower():
            continue
        out = {}
        for row in t["rows"]:
            if len(row) < 2:
                continue
            m = re.fullmatch(r"(\d+)\.x", row[0]["text"])
            d = re.search(r">=\s*(\d+)", row[1]["text"])
            if m and d:
                out[m.group(1)] = int(d.group(1))
        if out:
            return out
    raise ValueError("no 'Minimum Required Driver Version' table")


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except ValueError:
        return None


def parse_torch_cuda_init(text: str) -> dict[str, Any]:
    """torch/cuda/__init__.py → {"code_cc": {"13.2": {"x86_64": [...]}}, "solo": [...], "post_jetson": bool}。
    import せず ast で読む。"""
    tree = ast.parse(text)
    code_cc, solo, post = None, [], False
    for node in ast.walk(tree):
        target = value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif (
            isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
        ):
            target, value = node.targets[0].id, node.value
        if target == "PYTORCH_RELEASES_CODE_CC" and value is not None:
            code_cc = _literal(value)
        elif target == "DEVICE_REQUIREMENT" and isinstance(value, ast.Dict):
            for k, v in zip(value.keys, value.values, strict=True):
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(v, ast.Call)
                    and isinstance(v.func, ast.Name)
                    and v.func.id == "_CompatSet"
                ):
                    solo.append(int(k.value))
        elif target == "DEVICE_REQUIREMENT_POST_JETSON_SBSA_UNIFICATION":
            post = True
    if not isinstance(code_cc, dict):
        raise ValueError("no PYTORCH_RELEASES_CODE_CC")
    return {
        "code_cc": {v: {m: sorted(c) for m, c in by.items()} for v, by in code_cc.items()},
        "solo": sorted(solo),
        "post_jetson": post,
    }


def parse_rocm_build(text: str) -> list[str]:
    """.ci/docker/manywheel/build.sh の PYTORCH_ROCM_ARCH の gfx(全部の代入を合わせたもの)。"""
    found = set()
    for m in re.finditer(r'PYTORCH_ROCM_ARCH="([^"]*)"', text):
        found |= set(re.findall(r"gfx[0-9a-z]+", m.group(1)))
    if not found:
        raise ValueError("no PYTORCH_ROCM_ARCH")
    return sorted(found)


_WHEEL = re.compile(r"torch-(\d+\.\d+\.\d+)\+(cu\d+)-cp\d+t?-cp\d+t?-(\w+)\.whl")


def parse_torch_index(text: str) -> dict[str, dict[str, list[str]]]:
    """download.pytorch.org/whl/torch/(PEP 503 の HTML。PEP 691 の JSON は返さない)→
    {"cu129": {"win_amd64": ["2.8.0", ...], "linux_x86_64": [...], "linux_aarch64": [...]}}。"""
    out: dict[str, dict[str, set[str]]] = {}
    for ver, tag, plat in _WHEEL.findall(text):
        if plat.startswith("manylinux") or plat.startswith("linux"):
            plat = (
                "linux_aarch64"
                if plat.endswith("aarch64")
                else "linux_x86_64"
                if plat.endswith("x86_64")
                else plat
            )
        out.setdefault(tag, {}).setdefault(plat, set()).add(ver)
    if not out:
        raise ValueError("no CUDA wheels")
    key = lambda v: tuple(int(x) for x in v.split("."))  # noqa: E731
    return {t: {p: sorted(vs, key=key) for p, vs in sorted(by.items())} for t, by in sorted(out.items())}


def parse_comfy_mm(text: str) -> dict[str, Any]:
    """ComfyUI の comfy/model_management.py → 名前の一覧と、出てくる gfx。import せず ast で読む。"""
    tree = ast.parse(text)
    lists: dict[str, list[str]] = {}
    gfx = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("nvidia_10_series", "nvidia_16_series", "AMD_RDNA2_AND_OLDER_ARCH"):
                v = _literal(node.value)
                if isinstance(v, list):
                    lists[name] = [str(x) for x in v]
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"gfx\d[0-9a-f]{2,3}", node.value)
        ):
            gfx.add(node.value)
    if "nvidia_10_series" not in lists:
        raise ValueError("no nvidia_10_series")
    return {"lists": lists, "gfx": sorted(gfx)}


# --- 取る ------------------------------------------------------------------------------------------------


def _follow_redirect_page(fetch: Any, url: str, text: str) -> str:
    """「Redirecting…」だけの HTML(docs.pytorch.org の stable)は、meta refresh の先を 1 回だけ取る。"""
    m = re.search(r'http-equiv="refresh" content="\d+;\s*url=([^"]+)"', text)
    if m and len(text) < 20_000:
        return fetch.get_text(urljoin(url, html.unescape(m.group(1))))
    return text


def default_fetcher() -> Any:
    from .fetch import HttpFetcher  # fetch.make_session を使う(User-Agent・~/.netrc を送らない)

    return HttpFetcher(timeout=TIMEOUT)


def collect(fetch: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """全部の出典を取ってスナップショットにする。(スナップショット, 読めなかった出典)。例外を出さない。"""
    snap: dict[str, Any] = {}
    bad: list[dict[str, str]] = []

    def get(name: str, url: str, parse: Callable[[str], Any], *, json_: bool = False) -> Any:
        try:
            data = fetch.get_json(url) if json_ else fetch.get_text(url)
            if name == "intel_xpu":
                data = _follow_redirect_page(fetch, url, data)
            return parse(data)
        except Exception as e:
            bad.append({"source": name, "url": url, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return None

    simple: dict[str, Callable[[str], Any]] = {
        "pci_ids": parse_pci_ids,
        "rocm_hw_specs": parse_rocm_hw_specs,
        "radeon_win_compat": parse_compat,
        "ryzen_win_compat": parse_compat,
        "radeon_linux_compat": parse_compat,
        "ryzen_linux_compat": parse_compat,
        "therock_supported": parse_therock_supported,
        "therock_releases": parse_therock_releases,
        "intel_xpu": parse_intel_xpu,
        "cuda_rn": parse_cuda_rn,
        "torch_index": parse_torch_index,
    }
    for name, parse in simple.items():
        v = get(name, URLS[name], parse)
        if v is not None:
            snap[name] = v
    nv = []
    for name in ("nvidia_cuda_gpus", "nvidia_cuda_legacy"):
        v = get(name, URLS[name], lambda t, n=name: parse_nvidia(t, n))
        nv += v or []
    if nv:
        snap["nvidia"] = nv
    tag = get("torch_release", URLS["torch_release"], lambda d: str(d["tag_name"]), json_=True)
    if tag:
        init = get("torch_cuda_init", URLS["torch_cuda_init"].format(tag=tag), parse_torch_cuda_init)
        rocm = get("torch_build_rocm", URLS["torch_build_rocm"].format(tag=tag), parse_rocm_build)
        if init is not None:
            snap["torch"] = {"tag": tag, **init, "rocm_arch": rocm or []}
    ctag = get("comfyui_release", URLS["comfyui_release"], lambda d: str(d["tag_name"]), json_=True)
    if ctag:
        mm = get("comfyui_mm", URLS["comfyui_mm"].format(tag=ctag), parse_comfy_mm)
        if mm is not None:
            snap["comfyui"] = {"tag": ctag, **mm}
    return snap, bad


def load_snapshot(directory: str | Path) -> dict[str, Any]:
    d = Path(directory)
    out = {}
    for name in SNAPSHOTS:
        p = d / f"{name}.json"
        if p.exists():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def save_snapshot(snap: dict[str, Any], directory: str | Path) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    for name, data in snap.items():
        (d / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


# --- 表と照らす ------------------------------------------------------------------------------------------


_LSPCI_VENDOR = {"1002": "Advanced Micro Devices, Inc. [AMD/ATI]", "8086": "Intel Corporation"}


def lspci_name(key: str, pci_desc: str) -> str:
    """pci.ids の 1 行 → 新しい pci.ids の lspci の行から hardware.parse_lspci が作る製品名。"""
    ven, dev = key.split(":")
    line = f"0000:03:00.0 VGA compatible controller [0300]: {_LSPCI_VENDOR[ven]} {pci_desc} [{ven}:{dev}]"
    found = hw.parse_lspci(line)
    return found[0].name if found else pci_desc


def judge_pci_id(key: str, pci_desc: str) -> str | None:
    """pci.ids の 1 行が、新しい pci.ids の lspci の名前でどう判定されるか(gfx か Intel の系統)。汎用の名前は
    hardware.classify_adapter と同じく型番 ID の表で引く。分からなければ None。"""
    ven, dev = key.split(":")
    name = lspci_name(key, pci_desc)
    got = judge_table_name(ven, name)
    if got is None and hw._GENERIC_NAME.fullmatch(hw._low(name)):
        known = hw.PCI_DEVICE_NAMES.get((ven, dev))
        if known:
            got = judge_table_name(ven, known)
    return got


def judge_table_name(vendor: str, known: str) -> str | None:
    if vendor == "1002":
        r = hw.amd_gfx(known)
        return r[0] if r else None
    r = hw.intel_xpu(known if known.lower().startswith("intel") else f"intel {known}")
    return r[0] if r else None


def amd_product_gfx(name: str, graphics_model: str = "") -> str | None:
    """ROCm の hw specs・TheRock の製品名 → 表の gfx(内蔵 GPU は GPU 名、無ければ CPU 名で引く)。"""
    if graphics_model:
        r = hw.amd_gfx(f"AMD {graphics_model}")
        return r[0] if r else None
    if re.search(r"\bryzen\b", name, re.I) and not re.search(r"\bradeon\b|\d{3}m\b", name, re.I):
        r = hw.amd_gfx("AMD Radeon(TM) Graphics", cpu=name if name.startswith("AMD") else f"AMD {name}")
        return r[0] if r else None
    prefix = "AMD Instinct " if re.match(r"MI\d", name) else ("" if name.startswith("AMD") else "AMD ")
    r = hw.amd_gfx(prefix + name)
    return r[0] if r else None


def check_tables(snap: dict[str, Any]) -> list[Candidate]:
    """前回と比べずに、表と公式の一覧が合うかだけを見る(同梱の fixtures では空になる。試験が見る)。"""
    out: list[Candidate] = []
    arch = hw.AMD_ARCH

    def skip(key: str) -> bool:
        return key in KNOWN_EXCEPTIONS

    for row in snap.get("rocm_hw_specs") or []:
        name, gm, gfx = row["name"], row.get("graphics_model", ""), row["gfx"]
        got = amd_product_gfx(name, gm)
        label = f"{name} ({gm})" if gm else name
        if got != gfx and not skip(f"amd:{gm or name}={gfx}"):
            kind = "not in tables" if got is None else "differs"
            out.append(Candidate("rocm_hw_specs", kind, label, f"公式 {gfx} / 表 {got or '-'}"))
    for row in snap.get("therock_releases") or []:
        prod, gfx = row["product"], row["gfx"]
        got = amd_product_gfx(prod)
        if got != gfx and not skip(f"amd:{prod}={gfx}"):
            out.append(
                Candidate(
                    "therock_releases",
                    "not in tables" if got is None else "differs",
                    prod,
                    f"公式 {gfx} / 表 {got or '-'}",
                )
            )
    targets: dict[str, set[str]] = {}
    for row in snap.get("therock_supported") or []:
        targets.setdefault(row["gfx"], set()).add("therock_supported")
    for row in snap.get("rocm_hw_specs") or []:
        targets.setdefault(row["gfx"], set()).add("rocm_hw_specs")
    for g in (snap.get("comfyui") or {}).get("gfx") or []:
        targets.setdefault(g, set()).add("comfyui")
    for g in (snap.get("torch") or {}).get("rocm_arch") or []:
        targets.setdefault(g, set()).add("torch")
    for g, where in sorted(targets.items()):
        if g.endswith("x"):  # gfx117x のような系統の名前は、当てはまる gfx が表にあればよい
            if not any(k.startswith(g[:-1]) for k in arch):
                out.append(
                    Candidate("/".join(sorted(where)), "not in tables", g, "AMD_ARCH_RULES に無い系統")
                )
        elif g not in arch:
            out.append(Candidate("/".join(sorted(where)), "not in tables", g, "AMD_ARCH_RULES に無い gfx"))
    torch = snap.get("torch")
    if torch:
        if torch.get("rocm_arch") and set(torch["rocm_arch"]) != set(PYTORCH_ROCM_LINUX.values):
            add = sorted(set(torch["rocm_arch"]) - PYTORCH_ROCM_LINUX.values)
            rem = sorted(PYTORCH_ROCM_LINUX.values - set(torch["rocm_arch"]))
            out.append(
                Candidate(
                    "torch", "differs", "PYTORCH_ROCM_ARCH", f"増えた {add} / 消えた {rem}({torch['tag']})"
                )
            )
        builds = {b.cuda: b for b in TORCH_CUDA_BUILDS}
        for ver, by in (torch.get("code_cc") or {}).items():
            b = builds.get(ver)
            if b is None:
                tag = "cu" + ver.replace(".", "")
                if tag in (snap.get("torch_index") or {}):  # wheel が出てから候補にする
                    out.append(
                        Candidate(
                            "torch", "not in tables", f"CUDA {ver} ({tag})", f"{torch['tag']} の wheel がある"
                        )
                    )
                continue
            for machine, ccs in by.items():
                if (
                    set(ccs) != set(b.ccs.get(machine, ()))
                    and f"torch:{b.tag} {machine}" not in KNOWN_EXCEPTIONS
                ):
                    out.append(
                        Candidate(
                            "torch",
                            "differs",
                            f"{b.tag} {machine}",
                            f"公式 {sorted(ccs)} / 表 {sorted(b.ccs.get(machine, ()))}",
                        )
                    )
        if torch.get("solo") and set(torch["solo"]) != set(SOLO_CC.values):
            out.append(
                Candidate(
                    "torch",
                    "differs",
                    "DEVICE_REQUIREMENT",
                    f"単独の cc: 公式 {torch['solo']} / 表 {sorted(SOLO_CC.values)}",
                )
            )
        if torch.get("tag") and torch["tag"].lstrip("v") != TORCH_RELEASE:
            out.append(Candidate("torch", "differs", "PyTorch", f"最新 {torch['tag']} / 表 {TORCH_RELEASE}"))
    rn = snap.get("cuda_rn") or {}
    for major, driver in rn.items():
        bs = [b for b in TORCH_CUDA_BUILDS if b.cuda.split(".")[0] == major]
        newest = max(int(b.cuda.split(".")[0]) for b in TORCH_CUDA_BUILDS)
        if not bs:
            if int(major) > newest:
                out.append(
                    Candidate("cuda_rn", "not in tables", f"CUDA {major}.x", f"最低のドライバ {driver}")
                )
            continue
        for b in bs:
            if (driver,) != b.min_driver_linux or (driver,) != b.min_driver_windows:
                out.append(
                    Candidate("cuda_rn", "differs", b.tag, f"公式 {driver} / 表 {b.min_driver_linux[0]}")
                )
    comfy = snap.get("comfyui")
    if comfy:
        lists = comfy.get("lists") or {}
        for name, mine in (
            ("nvidia_10_series", hw._COMFY_FP16_PASCAL),
            ("nvidia_16_series", hw._COMFY_FP16_BROKEN),
        ):
            if name in lists and list(lists[name]) != list(mine):
                out.append(Candidate("comfyui", "differs", name, f"{comfy.get('tag')}: {lists[name]}"))
    for row in snap.get("nvidia") or []:
        got = hw.cc_from_name(row["name"].rstrip("*"))
        cc = tuple(int(x) for x in row["cc"].split("."))
        if got is not None and got != cc:
            out.append(
                Candidate("nvidia", "differs", row["name"], f"公式 {row['cc']} / 表 {got[0]}.{got[1]}")
            )
        if hw.arch_of(cc) == "unknown":
            out.append(
                Candidate("nvidia", "not in tables", row["name"], f"compute capability {row['cc']} の世代")
            )
    devices = (snap.get("pci_ids") or {}).get("devices") or {}
    for (ven, dev), known in hw.PCI_DEVICE_NAMES.items():
        desc = devices.get(f"{ven}:{dev}")
        if desc is None:
            continue
        new, old = judge_pci_id(f"{ven}:{dev}", desc), judge_table_name(ven, known)
        if new != old:
            out.append(
                Candidate(
                    "pci_ids",
                    "differs",
                    f"{ven}:{dev} {desc}",
                    f"新しい pci.ids の名前で {new or '-'} / 型番 ID の表で {old or '-'}",
                )
            )
    return out


def check_changes(snap: dict[str, Any], base: dict[str, Any]) -> list[Candidate]:
    """前回の一覧(base)と比べて、新しく出た・変わったものを候補にする。"""
    out: list[Candidate] = []
    new_dev = (snap.get("pci_ids") or {}).get("devices") or {}
    old_dev = (base.get("pci_ids") or {}).get("devices") or {}
    if "pci_ids" in base:
        for key, desc in sorted(new_dev.items()):
            if old_dev.get(key) == desc:
                continue
            got = judge_pci_id(key, desc)
            what = "新しい型番 ID" if key not in old_dev else f"名前が変わった(前は {old_dev[key]})"
            out.append(
                Candidate(
                    "pci_ids",
                    "not in tables" if got is None else "differs",
                    f"{key} {desc}",
                    f"{what}。判定 {got or '-'}",
                )
            )
    if "nvidia" in base:
        seen = {(r["name"], r["cc"]) for r in base["nvidia"]}
        for r in snap.get("nvidia") or []:
            if (r["name"], r["cc"]) not in seen:
                got = hw.cc_from_name(r["name"].rstrip("*"))
                out.append(
                    Candidate(
                        "nvidia",
                        "not in tables" if got is None else "differs",
                        r["name"],
                        f"新しい名前(公式 {r['cc']}、名前の表 {f'{got[0]}.{got[1]}' if got else '-'})",
                    )
                )
    if "therock_supported" in base:
        old = {(r["os"], r["gfx"]): r for r in base["therock_supported"]}
        for r in snap.get("therock_supported") or []:
            o = old.get((r["os"], r["gfx"]))
            state = lambda x: "/".join(k for k in ("build", "sanity", "release") if x.get(k)) or "-"  # noqa: E731
            if o is None:
                out.append(
                    Candidate(
                        "therock_supported",
                        "not in tables",
                        f"{r['gfx']} ({r['os']})",
                        f"新しい行: {state(r)}",
                    )
                )
            elif state(o) != state(r):
                out.append(
                    Candidate(
                        "therock_supported", "differs", f"{r['gfx']} ({r['os']})", f"{state(o)} → {state(r)}"
                    )
                )
    for name in ("radeon_win_compat", "ryzen_win_compat", "radeon_linux_compat", "ryzen_linux_compat"):
        if name not in base or name not in snap:
            continue
        old_p = {p for r in base[name] for p in r["products"]} | {g for r in base[name] for g in r["gfx"]}
        new_p = {p for r in snap[name] for p in r["products"]} | {g for r in snap[name] for g in r["gfx"]}
        for p in sorted(new_p - old_p):
            out.append(Candidate(name, "not in tables", p, "互換表に増えた"))
        for p in sorted(old_p - new_p):
            out.append(Candidate(name, "differs", p, "互換表から消えた"))
    if "intel_xpu" in base and "intel_xpu" in snap:

        def flat(s: dict[str, Any]) -> set[tuple[str, str]]:
            return {(c["os"], h) for c in s["client"] for h in c["hardware"]} | {
                (", ".join(d["os"]), d["device"]) for d in s["datacenter"]
            }

        for os_name, h in sorted(flat(snap["intel_xpu"]) - flat(base["intel_xpu"])):
            out.append(Candidate("intel_xpu", "not in tables", h, f"検証済みに増えた({os_name})"))
        for os_name, h in sorted(flat(base["intel_xpu"]) - flat(snap["intel_xpu"])):
            out.append(Candidate("intel_xpu", "differs", h, f"検証済みから消えた({os_name})"))
    if "torch_index" in base and "torch_index" in snap:
        mine = {b.tag for b in TORCH_CUDA_BUILDS}
        for tag, by in snap["torch_index"].items():
            for plat, vers in by.items():
                was = set((base["torch_index"].get(tag) or {}).get(plat) or [])
                added = [v for v in vers if v not in was]
                if added:
                    kind = "differs" if tag in mine else "not in tables"
                    out.append(Candidate("torch_index", kind, f"{tag} {plat}", f"新しい wheel {added}"))
    return out


def watch(
    fetch: Any | None = None, *, baseline: str | Path | None = BASELINE, snap: dict[str, Any] | None = None
) -> WatchResult:
    """取って照らす(毎週の更新から呼ぶ)。何が起きても例外を出さない。"""
    res = WatchResult()
    try:
        bad: list[dict[str, str]] = []
        if snap is None:
            snap, bad = collect(fetch or default_fetcher())
        res.unreadable = bad
        res.read = sorted(snap)
        res.candidates = check_tables(snap)
        if baseline is not None and Path(baseline).is_dir():
            res.baseline = str(baseline)
            res.candidates += check_changes(snap, load_snapshot(baseline))
    except Exception as e:
        res.unreadable.append(
            {"source": "gpuwatch", "url": "", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        )
    seen = set()
    uniq = []
    for c in res.candidates:
        k = (c.source, c.kind, c.item, c.detail)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    res.candidates = uniq
    return res


class _RawDir:
    """--from-dir: 保存した生の応答(URL の最後の部分のファイル名、または出典の名前)から読む(通信しない)。"""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def _file(self, url: str) -> Path:
        for name, u in URLS.items():
            if u == url or ("{tag}" in u and re.fullmatch(re.escape(u).replace(r"\{tag\}", "[^/]+"), url)):
                for p in self.root.glob(f"{name}.*"):
                    return p
        raise FileNotFoundError(url)

    def get_text(self, url: str) -> str:
        return self._file(url).read_text(encoding="utf-8", errors="replace")

    def get_json(self, url: str) -> Any:
        return json.loads(self.get_text(url))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m genmodelfit.gpuwatch", description="GPU の表の見張り(docs/release.md)"
    )
    p.add_argument(
        "--from-dir", metavar="RAW", help="保存した生の応答から読む(<出典の名前>.<拡張子>。通信しない)"
    )
    p.add_argument(
        "--save-fixtures", metavar="DIR", help="スナップショットを DIR に書く(tests/fixtures/gpu_official)"
    )
    p.add_argument("--no-baseline", action="store_true", help="前回の一覧と比べない")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    fetch = _RawDir(args.from_dir) if args.from_dir else default_fetcher()
    snap, bad = collect(fetch)
    if args.save_fixtures:
        save_snapshot(snap, args.save_fixtures)
    res = watch(snap=snap, baseline=None if args.no_baseline else BASELINE)
    res.unreadable = bad + res.unreadable
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.json:
        sys.stdout.write(json.dumps(res.to_dict(), ensure_ascii=False, indent=1) + "\n")
    else:
        for c in res.candidates:
            print(f"[{c.kind}] {c.source}: {c.item} — {c.detail}")
        for u in res.unreadable:
            print(f"[unreadable] {u['source']}: {u['error']}")
        print(f"候補 {len(res.candidates)} 件、読めなかった出典 {len(res.unreadable)} 件")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
