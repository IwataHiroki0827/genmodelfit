# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""利用者が出す実測(提出用の JSON)の作成・検査・まとめ(#34)。

流れ(genmodelfit 自体は何も送らない。利用者が自分の GitHub で Pull Request を出す):

1. `genmodelfit bench ... --submission <path>`(または `contrib make --from <bench の結果>`)で
   提出用の JSON(`contrib_schema` 1)を作る。入れるのは GPU 名・VRAM・世代・compute capability・ドライバ・
   ComfyUI の版・OS の種類(windows / linux。版やビルドは入れない)・メインメモリ(GiB の整数に丸める)・
   モデル ID・設定・結果・最大 VRAM / メモリ・時間・測り方・日付(UTC の日付だけ)。
   PC 名・ユーザ名・パス・GPU の UUID・PCI の位置・シリアル・IP・時刻は入れない(note の中も消す)
2. `contrib show` で、送る中身をそのまま見せる(同意のため)。個人の情報の点検の結果も出す
3. `data/contrib/<GPU>/<日付>-<内容の sha256 の先頭 8 桁>.json` として PR にする
4. CI(.github/workflows/contrib.yml)が `contrib validate` で形・値・個人の情報・重複を機械的に検査する
   (llmfit の validate_community_benchmarks.py と同じ考え方。2026-10-22)。式から大きく外れる・段階を変える
   などは失敗にせず、注意(notices)としてジョブのまとめに出す。マージするメンテナが見て決める
5. `contrib aggregate` が同じモデル・設定・GPU ごとに中央値と MAD で外れ値を除いてまとめ、
   同梱の `measurements_community.json` を作る。`--report` は校正(leave-one-out)の誤差の表と閾値の判定

ライブラリの関数は何も表示しない(表示は cli.py)。
"""

from __future__ import annotations

import contextlib
import getpass
import hashlib
import json
import math
import os
import platform
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .db import Database
from .estimate import (
    COMMUNITY_DEVIATION,
    PEAK_VRAM_RATIO,
    PEAK_VRAM_SLACK_GB,
    PRECISION_BYTES,
    gpu_key,
    norm_precision,
)
from .hardware import APPLE_CHIP_NAME, arch_of, cc_from_name, intel_xpu, parse_compute_cap, pci_name_conflict
from .models import MODALITIES, ORIGIN_COMMUNITY, OUTCOMES, Measurement, Settings, SystemInfo

CONTRIB_SCHEMA = 1
CONTRIB_DIR = "data/contrib"  # リポジトリの中の置き場所(PR で足す)
KIND = "genmodelfit-measurement"

# 値の健全さの許容
VRAM_TOLERANCE_GB = PEAK_VRAM_SLACK_GB  # 最大 VRAM は GPU の総量 × 1.02 + これまで(estimate.peak_vram_cap)
RAM_TOLERANCE_GB = 1.0  # 最大メモリはメインメモリ × 1.05 + これまで
MAX_GPU_VRAM_GB = 256.0  # 1 枚の GPU の VRAM の上限(超えたら誤り。0.9.0 で 512 から下げた)
MAX_RAM_GB = 4096.0
MAX_NOTE_CHARS = 2000
# 設定の範囲(テンプレートで使うありえる幅。外れたら誤り)
SETTING_RANGES: dict[str, tuple[float, float]] = {
    "width": (64, 8192),
    "height": (64, 8192),
    "frames": (1, 2000),
    "seconds": (0.1, 3600),
    "steps": (1, 1000),
    "batch": (1, 64),
}
ARCHES = (
    "fermi",
    "kepler",
    "maxwell",
    "pascal",
    "volta",
    "turing",
    "ampere",
    "ada",
    "hopper",
    "blackwell",
)
OS_FAMILIES = ("windows", "linux", "macos")
OFFLOADS = ("none", "dynamic", "partial", "lowvram", "oom")
VRAM_METRICS = ("torch_reserved",)

# 項目(これ以外のキーは誤り。知らない項目に PC の情報が紛れ込まないように)
TOP_KEYS = {"contrib_schema", "kind", "tool", "date", "system", "comfyui_version", "measurement"}
TOOL_KEYS = {"name", "version"}
SYSTEM_KEYS = {"os_family", "ram_total_gb", "gpu"}
# vendor・backend は 0.9.0 で足した任意の項目
# (NVIDIA 以外の GPU の実測を見分けるため。docs/schema.md「contrib」)
# pci_id(PCI の型番 ID "1002:744c"。機種の番号で、個体・差し場所は表さない)も 0.9.0 の任意の項目
GPU_KEYS = {"name", "vram_gb", "arch", "compute_capability", "driver", "vendor", "backend", "pci_id"}
# PCI の vendor ID → 会社(提出の pci_id と vendor が合うかを見る)
PCI_VENDORS = {"10de": "nvidia", "1002": "amd", "8086": "intel"}
_PCI_ID = re.compile(r"[0-9a-f]{4}:[0-9a-f]{4}")
# GPU の会社 → PyTorch の backend(models.USABLE_BACKENDS)
VENDOR_BACKEND = {"nvidia": "cuda", "amd": "rocm", "intel": "xpu", "apple": "mps"}
MEASUREMENT_KEYS = {
    "model_id",
    "settings",
    "outcome",
    "standard",
    "offload",
    "peak_vram_gb",
    "peak_ram_gb",
    "seconds",
    "vram_metric",
    "note",
}
SETTINGS_KEYS = {"width", "height", "frames", "seconds", "steps", "precision", "batch"}

# --- 個人の情報の検出 ------------------------------------------------------------------------------

# 文字列の長さの上限(note 以外)。長い値に正規表現を当て続けないためでもある
MAX_STR = 100
MAX_KEY = 64
MAX_FILE_BYTES = 64 * 1024  # 提出 1 件のファイルの大きさの上限
MAX_DEPTH = 8

# 制御文字・改行・見えない文字・向きを変える文字(CI のまとめやログへの差し込み、見えない書き換えを防ぐ)
CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f-\x9f\u00ad\u200b-\u200f\u2028-\u202e\u2060-\u2069\ufeff]")
_ZERO_WIDTH = re.compile(r"[\u00ad\u200b-\u200f\u2060-\u2064\ufeff]")
_HOST_WORDS = r"host ?name|computer ?name|PC ?name|user ?name|ホスト名|コンピューター?名|PC ?名|ユーザー?名"
_POSIX_ROOTS = r"home|Users|root|mnt|media|tmp|var|opt|srv|private|Volumes|etc|usr|data|workspace|run"

# (種類, 正規表現, 説明)。値を NFKC でそろえ(全角の Ｃ：＼ → C:\)、見えない文字を除き、
# ¥ を \ とみなしてから当てる。
# どれも入れ子の繰り返しの無い形にした(長い値でも時間が線形に近い。値の長さは MAX_STR / MAX_NOTE_CHARS まで)。
# 当てはまらなければ安全、ではない(人の名前・自由な文の意味は分からない)。docs/schema.md「contrib」
PRIVACY_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("path", re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]"), "Windows のパス(C:\\… など)"),
    ("path", re.compile(r"\\"), "バックスラッシュ(Windows のパスの区切り・\\\\server\\…)"),
    ("path", re.compile(r"\w[\\/]\w+[\\/]"), "区切りが 2 つ以上あるパス(a/b/c、Users\\name\\…)"),
    (
        "path",
        re.compile(r"(?<![\w.])//?(" + _POSIX_ROOTS + r")\b", re.I),
        "Linux / macOS のパス(/home/… など)",
    ),
    ("path", re.compile(r"(?<![\w.:])//\w"), "ネットワークのパス(//server/…)"),
    (
        "path",
        re.compile(r"(?<!\w)~[\\/]|%[A-Za-z_]\w*%|\$\{?[A-Za-z_]\w*\}?[\\/]|\$HOME\b|\$env:", re.I),
        "ホーム・環境変数のパス(~/、%USERPROFILE%、$HOME)",
    ),
    ("path", re.compile(r"%(2F|5C|3A)", re.I), "URL エンコードしたパス(%2F %5C %3A)"),
    ("users_dir", re.compile(r"(?<![A-Za-z])Users[\\/]|Documents and Settings", re.I), "ユーザーのフォルダ"),
    ("at_sign", re.compile(r"@\w"), "メールアドレス・@ の名前"),
    (
        "uuid",
        re.compile(r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"),
        "UUID(GPU の UUID など)",
    ),
    ("gpu_uuid", re.compile(r"\b(GPU|MIG)-[0-9A-Za-z]{4}", re.I), "GPU の UUID"),
    (
        "pci",
        re.compile(
            r"\b[0-9A-Fa-f]{4,8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\b|\b[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]\b"
            r"|PCI\\VEN_|\bbus[ _-]?id\b",
            re.I,
        ),
        "PCI の位置",
    ),
    ("ip", re.compile(r"(?<![\d.])\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?![\d.])"), "IP アドレス"),
    ("ip", re.compile(r"(?<![\w:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![\w:])"), "IPv6 アドレス"),
    ("mac", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"), "MAC アドレス"),
    ("url", re.compile(r"\b[A-Za-z][A-Za-z0-9+.-]{1,15}://|\bwww\.", re.I), "URL(ComfyUI の場所など)"),
    (
        "hostname",
        re.compile(
            r"\b(DESKTOP|LAPTOP|WIN)-[A-Z0-9]{4,}\b|\b[A-Za-z0-9]+-PC\b"
            r"|\b[A-Za-z0-9-]{1,63}\.(local|lan|localdomain|internal|corp|home\.arpa)\b|" + _HOST_WORDS,
            re.I,
        ),
        "PC 名・ユーザ名らしい文字(DESKTOP-XXXXXXX、*-PC、*.local、「PC 名」など)",
    ),
    (
        "time",
        re.compile(
            r"\b\d{1,2}:\d{2}\b|\b\d{1,2} ?(am|pm)\b|\d{1,2}時\d{1,2}分|\d{4}-\d{2}-\d{2}[T ]\d", re.I
        ),
        "時刻(hh:mm、2pm など。日付より細かい)",
    ),
    ("serial", re.compile(r"\b(S/?N|serial)\b|シリアル|製造番号", re.I), "シリアル番号"),
)


def normalize_for_scan(text: str) -> str:
    """検出の前にそろえる: NFKC(全角 → 半角)、見えない文字を除く、
    ¥(日本語の Windows のパスの区切り)を \\ に。"""
    t = unicodedata.normalize("NFKC", text)
    t = _ZERO_WIDTH.sub("", t)
    return t.replace("¥", "\\")


def _hits(text: str) -> list[tuple[str, str]]:
    t = normalize_for_scan(text)
    return [(kind, what) for kind, pat, what in PRIVACY_PATTERNS if pat.search(t)]


def _identity_words() -> list[str]:
    """この PC の名前・ユーザ名・ホーム(短すぎるものは誤検出が多いので除く)。"""
    words: list[str] = []
    with contextlib.suppress(Exception):
        words.append(platform.node())
    with contextlib.suppress(Exception):
        words.append(getpass.getuser())
    with contextlib.suppress(Exception):
        words.append(str(Path.home()))
    words += [os.environ.get(k, "") for k in ("COMPUTERNAME", "USERNAME", "USER", "HOSTNAME")]
    return sorted({w for w in words if len(w) >= 4}, key=len, reverse=True)


_SEGMENT = re.compile(r"[。\n;]")


def scrub_text(text: str | None, extra: list[str] | None = None) -> str | None:
    """note から個人の情報らしいところを消す。

    文(「。」・改行・「;」で区切ったもの)ごとに見て、この PC の名前・ユーザ名・ホーム(と extra)を含む文と、
    PRIVACY_PATTERNS のどれかに当たる文を、**文ごと**落とす(空白を含むパスの続きが残らないように、パスの途中で
    切らない)。この PC の名前はパスより先に見る。落とした数を最後に書く。"""
    if text is None:
        return None
    t = CONTROL_CHARS.sub("。", normalize_for_scan(text))
    words = [normalize_for_scan(w).lower() for w in [*(extra or []), *_identity_words()] if w]
    kept: list[str] = []
    dropped = 0
    for seg in _SEGMENT.split(t):
        s = seg.strip()
        if not s:
            continue
        low = s.lower()
        if any(w in low for w in words) or _hits(s):
            dropped += 1
            continue
        kept.append(s)
    out = "。".join(kept)
    if dropped:
        out = (out + "。" if out else "") + f"(個人の情報らしい文を {dropped} 個省いた)"
    return out


def privacy_findings(data: Any, raw: str | None = None) -> list[dict[str, str]]:
    """個人の情報らしい文字の一覧 [{where, kind, what}]。値とキーを調べる。

    raw は互換のため受け取るが使わない(JSON として読めた時点で、中身はすべて値とキーにある)。
    長すぎる文字列は調べない(validate が長さの誤りにする)。"""
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def check(where: str, text: str) -> None:
        if len(text) > MAX_NOTE_CHARS:
            return
        for kind, what in _hits(text):
            if (where, kind) not in seen:
                seen.add((where, kind))
                found.append({"where": where, "kind": kind, "what": what})

    def walk(x: Any, where: str, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            return
        if isinstance(x, dict):
            for k, v in x.items():
                key = str(k)[:MAX_KEY]
                check(f"{where}.{key} (key)", str(k))
                walk(v, f"{where}.{key}", depth + 1)
        elif isinstance(x, list):
            for i, v in enumerate(x[:100]):
                walk(v, f"{where}[{i}]", depth + 1)
        elif isinstance(x, str):
            check(where, x)

    walk(data, "$")
    return found


def string_problems(data: Any) -> list[str]:
    """制御文字・改行・見えない文字を含む文字列、長すぎる文字列・キー、配列、深すぎる入れ子(場所の一覧)。"""
    out: list[str] = []

    def walk(x: Any, where: str, depth: int = 0) -> None:
        if depth > MAX_DEPTH:
            out.append(f"{where}: 入れ子が深すぎます")
            return
        if isinstance(x, dict):
            for k, v in x.items():
                ks = str(k)
                if len(ks) > MAX_KEY or CONTROL_CHARS.search(ks):
                    out.append(f"{where}: キーが長すぎるか、制御文字を含みます")
                    continue
                walk(v, f"{where}.{ks}", depth + 1)
        elif isinstance(x, list):
            out.append(f"{where}: 配列は使えません")
        elif isinstance(x, str):
            limit = MAX_NOTE_CHARS if where == "$.measurement.note" else MAX_STR
            if len(x) > limit:
                out.append(f"{where}: {limit} 字を超えています")
            elif CONTROL_CHARS.search(x):
                out.append(f"{where}: 改行・制御文字・見えない文字を含みます")

    walk(data, "$")
    return out


def md_cell(value: Any) -> str:
    """Markdown の表の 1 マスに入れる文字。| ` < > [ ] & @ と改行・制御文字を無害にし、
    GFM の自動リンク(scheme://・www.)も切る(CI のまとめへの差し込み・リンクや HTML の埋め込みを防ぐ)。"""
    s = CONTROL_CHARS.sub(" ", str(value))
    for a, b in (
        ("&", "&amp;"),
        ("<", "&lt;"),
        (">", "&gt;"),
        ("|", "&#124;"),
        ("`", "&#96;"),
        ("[", "&#91;"),
        ("]", "&#93;"),
        ("://", ":&#47;&#47;"),
        ("@", "&#64;"),
    ):
        s = s.replace(a, b)
    return re.sub(r"(?i)www\.", "www&#46;", s)


# 個人の情報の点検の一覧(contrib show)。(id, 見出し, 当てはまる PRIVACY_PATTERNS の種類)
CHECKLIST: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("paths", "フォルダのパス(C:\\Users\\…、/home/… など)が入っていない", ("path", "users_dir")),
    ("host_user", "PC の名前・ユーザ名らしい文字が入っていない", ("hostname",)),
    (
        "device_ids",
        "GPU の UUID・PCI の位置・MAC・シリアルが入っていない",
        ("uuid", "gpu_uuid", "pci", "mac", "serial"),
    ),
    ("network", "IP アドレス・URL・メールアドレス・@ の名前が入っていない", ("ip", "url", "at_sign")),
    ("date_only", "日時は日付だけ(時刻が入っていない)", ("time",)),
)


def privacy_checklist(data: Any, raw: str | None = None) -> list[dict[str, Any]]:
    """送る前の点検の一覧 [{id, label, ok, detail}]。
    最後の note の項目は人が読んで確かめるもの(ok は None)。"""
    found = privacy_findings(data, raw)
    text = raw if raw is not None else json.dumps(data, ensure_ascii=False)
    out: list[dict[str, Any]] = []
    unknown: list[str] = []
    if isinstance(data, dict):
        unknown = sorted(set(data) - TOP_KEYS)
        for key, allowed in (("system", SYSTEM_KEYS), ("measurement", MEASUREMENT_KEYS)):
            sub = data.get(key)
            if isinstance(sub, dict):
                unknown += [f"{key}.{k}" for k in sorted(set(sub) - allowed)]
        gpu = (data.get("system") or {}).get("gpu") if isinstance(data.get("system"), dict) else None
        if isinstance(gpu, dict):
            unknown += [f"system.gpu.{k}" for k in sorted(set(gpu) - GPU_KEYS)]
    out.append(
        {
            "id": "known_fields",
            "label": "決まった項目だけ(知らない項目が無い)",
            "ok": not unknown,
            "detail": ", ".join(unknown) or None,
        }
    )
    for cid, label, kinds in CHECKLIST:
        hits = [f"{f['where']}({f['what']})" for f in found if f["kind"] in kinds]
        if cid == "host_user":
            own = [w for w in _identity_words() if w.lower() in text.lower()]
            hits += ["この PC の名前・ユーザ名・ホーム" for _ in own[:1]]
        out.append({"id": cid, "label": label, "ok": not hits, "detail": "; ".join(hits) or None})
    out.append(
        {
            "id": "note_reviewed",
            "label": "note(メモ)を自分で読んだ(自動の点検は文の意味までは分からない)",
            "ok": None,
            "detail": None,
        }
    )
    return out


# --- 作る -----------------------------------------------------------------------------------------


def gpu_slug(name: str) -> str:
    """GPU 名 → フォルダ名(例 'NVIDIA GeForce RTX 3070' → 'rtx-3070')。見積もりで「同じ GPU」と見る比べ方
    (estimate.gpu_key)と同じ。"""
    return gpu_key(name) or "unknown-gpu"


def canonical(data: dict[str, Any]) -> bytes:
    """内容の決まった並べ方(キーの順・空白によらない)。重複と名前の hash に使う。"""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.encode("utf-8")


def content_hash(data: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(data)).hexdigest()


def suggested_path(data: dict[str, Any]) -> str:
    """リポジトリの中の置き場所 data/contrib/<GPU>/<日付>-<sha8>.json(区切りは /)。"""
    gpu = ((data.get("system") or {}).get("gpu") or {}).get("name") or "unknown"
    return f"{CONTRIB_DIR}/{gpu_slug(str(gpu))}/{data.get('date')}-{content_hash(data)[:8]}.json"


def _os_family(os_name: str | None) -> str | None:
    n = (os_name or "").lower()
    if n.startswith("windows"):
        return "windows"
    if n.startswith("linux"):
        return "linux"
    if n.startswith(("darwin", "macos", "mac os")):
        return "macos"
    return None


def _clean_settings(s: Settings | dict[str, Any]) -> dict[str, Any]:
    d = asdict(s) if isinstance(s, Settings) else dict(s)
    order = ("width", "height", "frames", "seconds", "steps", "precision", "batch")
    out = {k: d.get(k) for k in order if d.get(k) is not None}
    if out.get("batch") == 1:
        out.pop("batch")  # 1 枚は省く(既定)
    return out


def build_submission(
    m: Measurement, system: SystemInfo | None = None, *, extra_secrets: list[str] | None = None
) -> dict[str, Any]:
    """bench の実測 1 件 → 提出用の JSON(contrib_schema 1)。

    system は測った PC(ComfyUI が同じ PC のときだけ渡す)。OS の種類・メインメモリ・GPU の世代を取る。
    GPU の世代は、測った GPU と同じ名前の GPU が system にあればその値、無ければ GPU 名の表から推す。
    source(自由に書ける文)は入れない。note は個人の情報らしいところを消して入れる。"""
    arch = cc = None
    driver = m.driver
    if system is not None:
        same = [g for g in system.gpus if g.name == m.gpu_name]
        if same:
            g = same[0]
            arch, cc, driver = g.arch, g.compute_capability, driver or g.driver
    if cc is None:
        guess = cc_from_name(m.gpu_name)
        if guess:
            cc, arch = f"{guess[0]}.{guess[1]}", arch or arch_of(guess)
    elif arch is None:
        parsed = parse_compute_cap(cc)
        arch = arch_of(parsed) if parsed else None
    if arch not in ARCHES:  # NVIDIA 以外の世代(RDNA 3 など)は arch に入れない(ARCHES は NVIDIA の世代だけ)
        arch = None
    backend = m.gpu_backend if m.gpu_backend in VENDOR_BACKEND.values() else None
    vendor = next((v for v, b in VENDOR_BACKEND.items() if b == backend), None)
    if vendor not in (None, "nvidia"):
        arch = cc = None  # compute capability は NVIDIA だけ
    ram = None
    os_family = None
    if system is not None:
        os_family = _os_family(system.os)
        if "ram" not in system.overrides and system.ram_total_gb:
            ram = round(system.ram_total_gb)
    note = scrub_text(m.note, extra_secrets)
    if note and len(note) > MAX_NOTE_CHARS:
        note = note[: MAX_NOTE_CHARS - 1] + "…"
    return {
        "contrib_schema": CONTRIB_SCHEMA,
        "kind": KIND,
        "tool": {"name": "genmodelfit", "version": __version__},
        "date": m.date or datetime.now(timezone.utc).date().isoformat(),
        "system": {
            "os_family": os_family,
            "ram_total_gb": ram,
            "gpu": {
                "name": m.gpu_name,
                "vram_gb": m.gpu_vram_gb,
                "arch": arch,
                "compute_capability": cc,
                "driver": driver,
                # 会社・backend は bench が記録したときだけ入れる(古い結果から作るときは入れない)
                **({"vendor": vendor, "backend": backend} if backend else {}),
                # PCI の型番 ID は bench が ComfyUI と同じ PC の検出から書いたときだけ(機種の番号だけ)
                **({"pci_id": m.gpu_pci_id} if getattr(m, "gpu_pci_id", None) else {}),
            },
        },
        "comfyui_version": m.comfyui_version,
        "measurement": {
            "model_id": m.model_id,
            "settings": _clean_settings(m.settings),
            "outcome": m.outcome,
            "standard": bool(m.standard),
            "offload": m.offload,
            "peak_vram_gb": m.peak_vram_gb,
            "peak_ram_gb": m.peak_ram_gb,
            "seconds": m.seconds,
            "vram_metric": m.vram_metric,
            "note": note,
        },
    }


def dumps(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1, allow_nan=False) + "\n"


def write_submission(data: dict[str, Any], target: str | os.PathLike[str]) -> Path:
    """書き出す。target が既にあるフォルダ(または / で終わる)なら <target>/<GPU>/<日付>-<sha8>.json、
    それ以外はそのファイル名で書く。書いた場所を返す。"""
    t = Path(target)
    if t.is_dir() or str(target).endswith(("/", "\\")):
        rel = suggested_path(data).split("/", 2)[2]  # data/contrib/ を除いた <GPU>/<名前>
        t = t / rel
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(dumps(data), encoding="utf-8", newline="\n")
    return t


def measurements_from_file(path: str | os.PathLike[str]) -> list[Measurement]:
    """bench が保存した JSON({"measurements": [...]})か、`bench --json` の出力({"measurement": {...}})。"""
    p = Path(path)
    data = loads_strict(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("measurement"), dict):
        items = [data["measurement"]]
    elif isinstance(data, dict) and isinstance(data.get("measurements"), list):
        items = data["measurements"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("bench の結果の形ではありません(measurements / measurement がない)")
    out = []
    for d in items:
        if not isinstance(d, dict) or "model_id" not in d or "gpu_name" not in d:
            raise ValueError("bench の結果の形ではありません(model_id / gpu_name がない)")
        out.append(Measurement.from_dict(d))
    return out


def to_measurement(data: dict[str, Any]) -> Measurement:
    """提出 1 件 → Measurement(まとめる前の 1 件)。最大メモリは、提出のメインメモリ(丸めた GiB)× 1.05 + 1 GiB
    (validate の上限と同じ)で切り詰める(見積もりの ram_gpu に入るため。0.9.0)。"""
    gpu = data["system"]["gpu"]
    mm = data["measurement"]
    pram = mm.get("peak_ram_gb")
    ram = (data.get("system") or {}).get("ram_total_gb")
    if _num(pram) and _num(ram) and ram > 0:
        pram = min(pram, ram * 1.05 + RAM_TOLERANCE_GB)
    return Measurement(
        model_id=mm["model_id"],
        gpu_name=gpu["name"],
        gpu_vram_gb=float(gpu["vram_gb"]),
        peak_vram_gb=float(mm["peak_vram_gb"]),
        settings=Settings.from_dict(mm.get("settings") or {}),
        peak_ram_gb=pram,
        seconds=mm.get("seconds"),
        driver=gpu.get("driver"),
        comfyui_version=data.get("comfyui_version"),
        offload=mm.get("offload"),
        source=ORIGIN_COMMUNITY,
        date=data.get("date"),
        note=mm.get("note"),
        outcome=mm.get("outcome", "ok"),
        standard=bool(mm.get("standard", True)),
        vram_metric=mm.get("vram_metric"),
        origin=ORIGIN_COMMUNITY,
        gpu_backend=gpu.get("backend"),
        gpu_pci_id=gpu.get("pci_id"),
    )


# --- 検査 -----------------------------------------------------------------------------------------


@dataclass
class Report:
    """1 ファイルの検査の結果。errors(機械的な検査: 形・値の範囲・個人の情報・重複・ファイル名)があれば
    受け付けない。warnings は受け付けるが提出した人に知らせる。notices も受け付ける(失敗にしない)が、
    マージするメンテナが見るもの(式から大きく外れる・段階を変える・一覧に無いモデルなど。
    CI のジョブのまとめに出す)。2026-10-22 に llmfit に合わせた
    (それまでの「人が確かめるまで受け付けない」review と REVIEWED は外した)。"""

    path: str
    ok: bool = True
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    notices: list[dict[str, str]] = field(default_factory=list)
    sha256: str | None = None
    suggested_path: str | None = None
    model_id: str | None = None
    gpu_name: str | None = None

    def error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})
        self.ok = False

    def warn(self, code: str, message: str) -> None:
        self.warnings.append({"code": code, "message": message})

    def notice(self, code: str, message: str) -> None:
        self.notices.append({"code": code, "message": message})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(x: Any) -> bool:
    """有限の数か(NaN・inf は数とみなさない)。"""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def number_problems(data: Any) -> list[str]:
    """NaN・inf・負の数の場所(提出のどの数も有限で 0 以上。0.9.0)。"""
    out: list[str] = []

    def walk(x: Any, where: str) -> None:
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, f"{where}.{k}")
        elif isinstance(x, (int, float)) and not isinstance(x, bool):
            if not math.isfinite(x):
                out.append(f"{where} が有限の数ではありません(NaN・inf)")
            elif x < 0:
                out.append(f"{where} が負の数です: {x!r}")

    walk(data, "$")
    return out


def _no_constants(name: str) -> Any:
    raise ValueError(f"JSON に {name} があります(NaN・Infinity は受け付けません)")


def loads_strict(text: str) -> Any:
    """JSON を読む。NaN・Infinity・-Infinity は断る(Python の json は既定で読んでしまう)。"""
    return json.loads(text, parse_constant=_no_constants)


def name_vendor(name: str) -> str | None:
    """GPU 名から分かる会社(分からなければ None。estimate.gpu_vendor と違い NVIDIA を既定にしない)。"""
    n = (name or "").lower()
    if re.search(r"nvidia|geforce|quadro|tesla|\brtx\b|\bgtx\b", n):
        return "nvidia"
    if "radeon" in n or n.startswith("amd") or "instinct" in n:
        return "amd"
    if n.startswith("intel") or re.search(r"\barc\b", n):
        return "intel"
    if n.startswith("apple") or n == "mps":
        return "apple"
    return None


def _check_keys(r: Report, d: Any, allowed: set[str], where: str, required: set[str]) -> bool:
    if not isinstance(d, dict):
        r.error("schema", f"{where} はオブジェクトにしてください")
        return False
    for k in sorted(set(d) - allowed):
        r.error(
            "unknown_field",
            f"{where}.{k} は知らない項目です(PC の情報が紛れないよう、決まった項目だけ受け付けます)",
        )
    for k in sorted(required - set(d)):
        r.error("schema", f"{where}.{k} がありません")
    return True


_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_DRIVER = re.compile(r"\d{2,4}(\.\d{1,3}){1,3}")
_VERSION = re.compile(r"v?\d{1,3}\.\d{1,3}\.\d{1,4}")  # ComfyUI の版(厳密な x.y.z。後ろに何も付けない)
_TOOL_VERSION = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,4}((a|b|rc|\.dev|\.post)\d{1,4})?")
# GPU の名前: 英数字・空白・. - と、ComfyUI・PyTorch・Windows の名前に入る (R)・(TM)・™・® だけ
# と、Intel の内蔵 GPU の名前に付く大きさ「(16GB)」(「Intel(R) Arc(TM) 140V GPU (16GB)」)
# (ほかの括弧・記号・自由な文は入れられない。0.9.0 で (R) などを許した)
_GPU_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9 .-]|\((?:R|TM|r|tm)\)|\(\d{1,3} ?GB\)|[™®]){1,63}")
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,199}")
_PRECISION = re.compile(r"[A-Za-z0-9_-]{1,24}")
# 式の見積もりから 40% より外れた値は注意(マージするメンテナが見る)。
# 見積もりで設定の違う推定へ広げる範囲と同じ
NOTICE_DEVIATION = COMMUNITY_DEVIATION
_CC = re.compile(r"\d{1,2}\.\d")
_FILENAME = re.compile(r"(\d{4}-\d{2}-\d{2})-([0-9a-f]{8})\.json")


def validate_submission(
    data: Any,
    db: Database | None = None,
    *,
    path: str | os.PathLike[str] | None = None,
    raw: str | None = None,
    today: _date | None = None,
    check_filename: bool = True,
) -> Report:
    """提出 1 件を検査する。db を渡すと、モデル ID が一覧にあるか・設定が種類に合うかも見る。"""
    r = Report(path=Path(path).as_posix() if path is not None else "-")
    today = today or datetime.now(timezone.utc).date()
    if not isinstance(data, dict):
        r.error("schema", "JSON のオブジェクトではありません")
        return r
    for msg in string_problems(data):
        r.error("string", msg)
    numbers = number_problems(data)
    for msg in numbers:
        r.error("value", msg)
    if any("NaN" in msg for msg in numbers):
        return r  # NaN・inf は JSON に書けない(中身の hash も作れない)ので、ここで止める
    for f in privacy_findings(data, raw):
        r.error("privacy", f"{f['where']}: {f['what']}らしい文字があります。消してから出してください")
    _check_keys(r, data, TOP_KEYS, "$", TOP_KEYS - {"comfyui_version"})
    if data.get("contrib_schema") != CONTRIB_SCHEMA:
        r.error(
            "schema", f"contrib_schema が {CONTRIB_SCHEMA} ではありません: {data.get('contrib_schema')!r}"
        )
        return r
    if data.get("kind") != KIND:
        r.error("schema", f"kind は {KIND!r} にしてください")
    tool = data.get("tool")
    if _check_keys(r, tool, TOOL_KEYS, "$.tool", TOOL_KEYS) and (
        tool.get("name") != "genmodelfit"
        or not isinstance(tool.get("version"), str)
        or not _TOOL_VERSION.fullmatch(tool["version"])
    ):
        r.error("schema", "tool は {name: genmodelfit, version: 版} にしてください")
    # 日付(日付だけ。未来の日付は誤り)
    d = data.get("date")
    day = None
    if not isinstance(d, str) or not _DATE.fullmatch(d):
        r.error("date", f"date は YYYY-MM-DD(日付だけ)にしてください: {d!r}")
    else:
        try:
            day = _date.fromisoformat(d)
        except ValueError:
            r.error("date", f"date が日付として読めません: {d}")
        if day and day > today + timedelta(days=1):
            r.error("date", f"date が未来です: {d}")
        if day and day.year < 2026:
            r.warn("date_old", f"date が古すぎます(genmodelfit より前): {d}")
    cv = data.get("comfyui_version")
    if cv is not None and (not isinstance(cv, str) or not _VERSION.fullmatch(cv)):
        r.error("value", f"comfyui_version は x.y.z の版の番号だけにしてください: {cv!r}")
    if cv is None:
        r.warn("comfyui_version_missing", "comfyui_version がありません")

    system = data.get("system")
    gpu: dict[str, Any] = {}
    ram = None
    if _check_keys(r, system, SYSTEM_KEYS, "$.system", SYSTEM_KEYS):
        osf = system.get("os_family")
        if osf is not None and osf not in OS_FAMILIES:
            r.error(
                "value", f"system.os_family は {' / '.join(OS_FAMILIES)} か null です(版は入れない): {osf!r}"
            )
        ram = system.get("ram_total_gb")
        if ram is not None:
            if not _num(ram) or not 0 < ram <= MAX_RAM_GB:
                r.error("value", f"system.ram_total_gb がありえない値です: {ram!r}")
                ram = None
            elif ram != round(ram):
                r.warn(
                    "ram_not_rounded", "system.ram_total_gb は GiB の整数に丸めてください(機種を絞らないため)"
                )
        g = system.get("gpu")
        if _check_keys(r, g, GPU_KEYS, "$.system.gpu", {"name", "vram_gb"}):
            gpu = g
    name = gpu.get("name")
    vram = gpu.get("vram_gb")
    if gpu:
        if not isinstance(name, str) or not _GPU_NAME.fullmatch(name):
            r.error(
                "value",
                "system.gpu.name は GPU の名前(英数字・空白・. -・(R)・(TM)・™・®、64 字まで)"
                "だけにしてください",
            )
            name = None
        if not _num(vram) or not 0 < vram <= MAX_GPU_VRAM_GB:
            r.error("value", f"system.gpu.vram_gb がありえない値です: {vram!r}")
            vram = None
        drv = gpu.get("driver")
        if drv is not None and (not isinstance(drv, str) or not _DRIVER.fullmatch(drv)):
            r.error("value", f"system.gpu.driver はドライバの版にしてください: {drv!r}")
        if drv is None:
            r.warn("driver_missing", "system.gpu.driver がありません")
        vendor, backend = gpu.get("vendor"), gpu.get("backend")
        if vendor is not None and vendor not in VENDOR_BACKEND:
            r.error("value", f"system.gpu.vendor は {' / '.join(VENDOR_BACKEND)} か無しです: {vendor!r}")
        elif backend is not None and backend not in VENDOR_BACKEND.values():
            r.error(
                "value",
                f"system.gpu.backend は {' / '.join(VENDOR_BACKEND.values())} か無しです: {backend!r}",
            )
        elif (vendor is None) != (backend is None) or (vendor and VENDOR_BACKEND[vendor] != backend):
            r.error(
                "gpu_inconsistent", f"system.gpu の vendor({vendor!r})と backend({backend!r})が合いません"
            )
        elif vendor not in (None, "nvidia") and (gpu.get("compute_capability") or gpu.get("arch")):
            r.error("gpu_inconsistent", "compute capability・arch は NVIDIA の GPU だけに入れます")
        elif vendor and isinstance(name, str) and name_vendor(name) not in (None, vendor):
            r.error("gpu_inconsistent", f"GPU 名({name})の会社と vendor({vendor})が合いません")
        elif (
            vendor == "apple"
            and isinstance(name, str)
            and name != "mps"
            and not APPLE_CHIP_NAME.fullmatch(name)
        ):
            # Apple の GPU 名は bench が system_profiler で読んだチップ名(Apple M2 Pro など)か mps だけ
            r.error(
                "gpu_inconsistent", f"Apple の GPU 名はチップ名(Apple M2 Pro など)か mps だけです: {name}"
            )
        pci = gpu.get("pci_id")
        if pci is not None:
            if not isinstance(pci, str) or not _PCI_ID.fullmatch(pci):
                r.error("value", f"system.gpu.pci_id は PCI の型番 ID(1002:744c の形、小文字)です: {pci!r}")
            elif (vendor or "nvidia") != PCI_VENDORS.get(pci[:4]):
                r.error(
                    "gpu_inconsistent",
                    f"system.gpu.pci_id の会社({PCI_VENDORS.get(pci[:4], pci[:4])})と"
                    f" vendor({vendor})が合いません",
                )
            elif isinstance(name, str):
                conflict = pci_name_conflict(vendor or "nvidia", pci, name)
                if conflict is None:
                    r.notice(
                        "unknown_gpu",
                        f"system.gpu.pci_id {pci} が型番 ID の表(hardware.PCI_DEVICE_NAMES)にありません",
                    )
                elif conflict:
                    r.error(
                        "gpu_inconsistent", f"system.gpu.pci_id {pci} の GPU と GPU 名({name})が合いません"
                    )
        arch, cc = gpu.get("arch"), gpu.get("compute_capability")
        if arch is not None and arch not in ARCHES:
            r.error("value", f"system.gpu.arch が知らない世代です: {arch!r}")
        if cc is not None and (not isinstance(cc, str) or not _CC.fullmatch(cc)):
            r.error("value", f"system.gpu.compute_capability は 8.6 の形にしてください: {cc!r}")
            cc = None
        if name:
            r.gpu_name = name
            known = cc_from_name(name)
            if known is None:
                r.notice("unknown_gpu", f"GPU 名が知っている表にありません(NVIDIA 以外・新しい GPU?): {name}")
            elif cc and parse_compute_cap(cc) != known:
                r.error(
                    "gpu_inconsistent",
                    f"GPU 名({name})の compute capability は {known[0]}.{known[1]} ですが、{cc} とあります",
                )
            pcc = parse_compute_cap(cc) if cc else None
            if pcc and arch and arch_of(pcc) != arch:
                r.error("gpu_inconsistent", f"compute capability {cc} と世代 {arch} が合いません")
            elif known and arch and arch_of(known) != arch and not cc:
                r.error(
                    "gpu_inconsistent", f"GPU 名({name})の世代は {arch_of(known)} ですが、{arch} とあります"
                )

    mm = data.get("measurement")
    if _check_keys(
        r, mm, MEASUREMENT_KEYS, "$.measurement", {"model_id", "settings", "outcome", "peak_vram_gb"}
    ):
        _check_measurement(r, mm, vram, ram, db)

    r.sha256 = content_hash(data)
    r.suggested_path = suggested_path(data)
    if r.ok and db is not None and r.model_id:
        _check_effect(r, data, db)
    if path is not None and check_filename:
        _check_filename(r, Path(path), data)
    return r


def _check_measurement(r: Report, mm: dict[str, Any], vram: Any, ram: Any, db: Database | None) -> None:
    mid = mm.get("model_id")
    entry = None
    if not isinstance(mid, str) or not _MODEL_ID.fullmatch(mid):
        r.error("value", "measurement.model_id は一覧の ID(英数字と _ . : / + -、200 字まで)にしてください")
    else:
        r.model_id = mid
        entry = db.get(mid) if db is not None else None
        if db is not None and entry is None:
            r.notice("unknown_model", f"モデル ID が同梱の一覧にありません: {mid}")
    outcome = mm.get("outcome")
    if outcome not in OUTCOMES:
        r.error("value", f"measurement.outcome は {' / '.join(OUTCOMES)} です: {outcome!r}")
    std = mm.get("standard", True)
    if not isinstance(std, bool):
        r.error("value", "measurement.standard は true / false です")
    elif not std:
        r.warn(
            "nonstandard",
            "標準でない条件の実測なので、まとめにも見積もりにも使いません(記録としては受け付けます)",
        )
    off = mm.get("offload")
    if off is not None and off not in OFFLOADS:
        r.error("value", f"measurement.offload は {' / '.join(OFFLOADS)} か null です: {off!r}")
    if outcome == "oom" and off not in (None, "oom"):
        r.warn("offload_outcome", f"outcome が oom なのに offload が {off} です")
    metric = mm.get("vram_metric")
    if metric is not None and metric not in VRAM_METRICS:
        r.error("value", f"measurement.vram_metric は {' / '.join(VRAM_METRICS)} か null です: {metric!r}")
    if metric is None:
        r.warn("metric_missing", "vram_metric がありません(測り方が分からないので比べにくい)")
    peak = mm.get("peak_vram_gb")
    if not _num(peak) or peak < 0:
        r.error("value", f"measurement.peak_vram_gb は 0 以上の数です: {peak!r}")
    else:
        if outcome == "ok" and peak <= 0:
            r.error("value", "最後まで動いた(ok)のに peak_vram_gb が 0 です")
        if _num(vram) and peak > vram * PEAK_VRAM_RATIO + VRAM_TOLERANCE_GB:
            r.error("value", f"最大 VRAM {peak} GB が GPU の総量 {vram} GB を超えています")
    pram = mm.get("peak_ram_gb")
    if pram is not None:
        if not _num(pram) or pram <= 0:
            r.error("value", f"measurement.peak_ram_gb は 0 より大きい数か null です: {pram!r}")
        elif _num(ram) and pram > ram * 1.05 + RAM_TOLERANCE_GB:
            r.error("value", f"最大メモリ {pram} GB がメインメモリ {ram} GB を超えています")
    secs = mm.get("seconds")
    if secs is None:
        if outcome == "ok":
            r.error("value", "最後まで動いた(ok)のに seconds がありません")
    elif not _num(secs) or secs <= 0 or secs > 86400:
        r.error("value", f"measurement.seconds は 0 より大きい秒数です: {secs!r}")
    note = mm.get("note")
    if note is not None and (not isinstance(note, str) or len(note) > MAX_NOTE_CHARS):
        r.error("value", f"measurement.note は {MAX_NOTE_CHARS} 字までの文です")
    s = mm.get("settings")
    if not isinstance(s, dict):
        r.error("schema", "measurement.settings はオブジェクトにしてください")
        return
    for k in sorted(set(s) - SETTINGS_KEYS):
        r.error("unknown_field", f"measurement.settings.{k} は知らない項目です")
    for k, (lo, hi) in SETTING_RANGES.items():
        v = s.get(k)
        if v is None:
            continue
        if not _num(v) or not lo <= v <= hi or (k != "seconds" and v != int(v)):
            r.error("settings", f"settings.{k} が範囲({lo:g}〜{hi:g})の外です: {v!r}")
    prec = s.get("precision")
    if prec is not None and (
        not isinstance(prec, str)
        or not _PRECISION.fullmatch(prec)
        or norm_precision(prec) not in PRECISION_BYTES
    ):
        r.error(
            "settings", f"settings.precision は知っている精度({', '.join(PRECISION_BYTES)})にしてください"
        )
    if entry is not None and outcome == "ok" and _num(peak) and peak > 0:
        _check_against_estimate(r, entry, mm, s, peak)
    if entry is not None:
        mod = entry.modality
        if mod not in MODALITIES:
            return
        if mod in ("image", "video") and (s.get("width") is None or s.get("height") is None):
            r.warn("settings_missing", f"{mod} のモデルなのに width / height がありません")
        if mod != "video" and s.get("frames") is not None:
            r.warn("settings_modality", f"{mod} のモデルに frames があります")
        if mod in ("music", "sfx", "tts") and s.get("seconds") is None:
            r.warn("settings_missing", f"{mod} のモデルなのに seconds(長さ)がありません")


def _check_against_estimate(
    r: Report, entry: Any, mm: dict[str, Any], s: dict[str, Any], peak: float
) -> None:
    """式の見積もりと比べ、ありそうにない値を注意にする(失敗にはしない。マージするメンテナが見る)。"""
    from .estimate import activation_gb, apply_precision, components, load_coefficients, need, settings_for

    parts = components(entry)
    if parts.unknown:
        return
    c = load_coefficients()
    try:
        settings = settings_for(entry, Settings.from_dict({k: v for k, v in s.items() if k in SETTINGS_KEYS}))
        parts = apply_precision(parts, entry, settings)
        full = need(parts, activation_gb(entry, settings, c), c).full
    except (TypeError, ValueError):
        return
    if full > 0 and abs(peak - full) / full > NOTICE_DEVIATION:
        r.notice(
            "deviation",
            f"最大 VRAM {peak} GB が式の見積もり {full:.2f} GB から"
            f" {abs(peak - full) / full:.0%} 外れています(±{NOTICE_DEVIATION:.0%} を超えた。"
            f"設定の違う推定へは ±{NOTICE_DEVIATION:.0%} までしか広げない)",
        )
    # 動的 VRAM・partial で下限より小さいのは普通(一部の重みをメインメモリに置く)。全部載せ(none)と言うのに
    # 小さいものだけ知らせる。見積もりはどちらも下限(重み + 固定分)で切り詰める(estimate._prepare)
    floor = parts.total * c.weight_factor + c.overhead_gb
    if mm.get("offload") == "none" and peak < floor:
        r.notice(
            "below_weights",
            f"全部 GPU に載せた(offload: none)のに、最大 VRAM {peak} GB が全部載せの下限"
            f"(重み + 固定分 {floor:.2f} GB)より小さい。見積もりでは下限の値にする",
        )


# 段階が変わるかを確かめるメインメモリ(よくある大きさ。GPU は実測と同じ GPU。_measured_pc)
CHECK_RAM_GB = (16.0, 32.0, 64.0)


def _measured_pc(m: Measurement, ram: float) -> SystemInfo:
    """実測と同じ GPU(名前・PCI の型番 ID・会社。VRAM は実測の GPU の総量)の仮の PC。ほかの利用者の実測は
    同じ GPU の PC にだけ使われる(estimate.pick_measurement)ので、段階が変わるかはこの PC で見る(0.9.0)。"""
    from .estimate import measurement_vendor
    from .hardware import classify_gpu
    from .models import Gpu

    vendor = measurement_vendor(m)
    g = Gpu(
        name=m.gpu_name,
        vendor=vendor,
        vram_total_gb=float(m.gpu_vram_gb),
        backend=VENDOR_BACKEND.get(vendor, "cuda"),
        pci_id=getattr(m, "gpu_pci_id", None),
    )
    if vendor == "nvidia":
        g = classify_gpu(g, windows_driver=True, comfy_windows=True)
        g.backend = "cuda"
    else:
        g.compute_dtype, g.bf16, g.fp8 = "fp16", True, False
        g.memory = "unified" if vendor == "apple" else "dedicated"
    return SystemInfo(os="-", cpu="-", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=[g])


def _own_measurements(db: Database) -> list[Measurement]:
    return [m for m in db.measurements if getattr(m, "origin", None) != ORIGIN_COMMUNITY]


def community_effect(
    entry: Any, own: list[Measurement], community: list[Measurement], settings: Settings | None = None
) -> tuple[float, list[dict[str, Any]]]:
    """ほかの利用者の実測を足したときの (全部載せの VRAM が変わる割合(符号つき。絶対値の最大のもの),
    段階が変わる PC の一覧)。ほかの利用者の実測は同じ GPU の PC にだけ使われる(0.9.0)ので、PC は実測と同じ
    GPU(名前・型番 ID・会社、VRAM は実測の GPU の総量)× メインメモリ CHECK_RAM_GB。既定の設定
    (と settings を渡せばその設定)で比べる。同じ GPU で段階が上がる・NVIDIA 以外の good の上限が
    外れるのも出る。"""
    from .estimate import estimate, measurement_vendor, nvidia_vram_variants

    rise = 0.0
    changes: list[dict[str, Any]] = []
    gpus: dict[tuple, Measurement] = {}
    for m in community:
        gpus.setdefault((gpu_key(m.gpu_name), getattr(m, "gpu_pci_id", None), round(m.gpu_vram_gb or 0)), m)
        # 名前の表で公称の VRAM が分かる GPU(Intel の Arc は hardware._INTEL_XPU、NVIDIA は
        # estimate.NVIDIA_VRAM_VARIANTS)は、その VRAM の
        # PC も点検する(同じ名前で VRAM の違う版がある: A770 の 8 GB と 16 GB など。同じ GPU の実測として
        # 使われるため)
        nominal = intel_xpu(m.gpu_name) if measurement_vendor(m) == "intel" else None
        if nominal and nominal[2] and round(nominal[2]) != round(m.gpu_vram_gb or 0):
            v = Measurement(**{**vars(m), "gpu_vram_gb": float(nominal[2])})
            gpus.setdefault((gpu_key(m.gpu_name), getattr(m, "gpu_pci_id", None), round(nominal[2])), v)
        # NVIDIA は estimate.NVIDIA_VRAM_VARIANTS(公式の製品ページで確かめた、VRAM の違う版)の各 VRAM
        if measurement_vendor(m) == "nvidia":
            for nv in nvidia_vram_variants(m.gpu_name):
                if round(nv) != round(m.gpu_vram_gb or 0):
                    v = Measurement(**{**vars(m), "gpu_vram_gb": nv})
                    gpus.setdefault((gpu_key(m.gpu_name), getattr(m, "gpu_pci_id", None), round(nv)), v)
    for s in [None] if settings is None else [None, settings]:
        for m in gpus.values():
            for ram in CHECK_RAM_GB:
                sysi = _measured_pc(m, ram)
                a = estimate(entry, sysi, s, own)
                b = estimate(entry, sysi, s, [*own, *community])
                if ram == CHECK_RAM_GB[-1] and a.vram_gb and not a.unknown_size:
                    rel = (b.vram_gb - a.vram_gb) / a.vram_gb
                    rise = rel if abs(rel) > abs(rise) else rise
                v = m.gpu_vram_gb
                if a.fit != b.fit and not any(
                    c["gpu_name"] == m.gpu_name and c["vram_gb"] == v and c["ram_gb"] == ram for c in changes
                ):
                    changes.append(
                        {
                            "model_id": entry.id,
                            "gpu_name": m.gpu_name,
                            "vram_gb": v,
                            "ram_gb": ram,
                            "before": a.fit,
                            "after": b.fit,
                        }
                    )
    return rise, changes


def _check_effect(r: Report, data: dict[str, Any], db: Database) -> None:
    """この提出だけを足すと、よくある PC で段階がどう変わるか(注意として出す。マージするメンテナが見る)。"""
    mm = data["measurement"]
    entry = db.get(mm["model_id"])
    if entry is None or mm.get("outcome") != "ok" or mm.get("standard") is False:
        return
    m = to_measurement(data)
    m.n_submissions, m.n_used, m.n_oom = 1, 1, 0
    own = _own_measurements(db)
    try:
        _rel, changes = community_effect(entry, own, [m], m.settings)
    except (TypeError, ValueError, ZeroDivisionError):
        return
    if changes:
        where = "、".join(
            f"{c['gpu_name']} {c['vram_gb']:g} GB・メモリ {c['ram_gb']:g} GB: {c['before']} → {c['after']}"
            for c in changes[:6]
        )
        r.notice("fit_change", f"同じ GPU の PC で段階が変わります({where})")


def _check_filename(r: Report, p: Path, data: dict[str, Any]) -> None:
    m = _FILENAME.fullmatch(p.name)
    if not m:
        r.error("filename", f"ファイル名は <日付>-<sha8>.json にしてください(例 {r.suggested_path})")
        return
    if m.group(1) != data.get("date"):
        r.error("filename", f"ファイル名の日付 {m.group(1)} と date {data.get('date')} が違います")
    if r.sha256 and m.group(2) != r.sha256[:8]:
        r.error(
            "filename", f"ファイル名の hash {m.group(2)} が内容の sha256 の先頭 {r.sha256[:8]} と違います"
        )
    gpu = ((data.get("system") or {}).get("gpu") or {}).get("name")
    if isinstance(gpu, str) and p.parent.name != gpu_slug(gpu):
        r.error("filename", f"フォルダ名 {p.parent.name} が GPU 名から決まる {gpu_slug(gpu)} と違います")


class SubmissionReadError(ValueError):
    """読めない提出(大きすぎる・シンボリックリンク・JSON でない・入れ子が深すぎる)。"""


def read_submission(path: str | os.PathLike[str]) -> tuple[Any, str]:
    p = Path(path)
    if p.is_symlink():
        raise SubmissionReadError("シンボリックリンクは受け付けません")
    if not p.is_file():
        raise SubmissionReadError("ファイルではありません")
    size = p.stat().st_size
    if size > MAX_FILE_BYTES:
        raise SubmissionReadError(f"大きすぎます({size} バイト。{MAX_FILE_BYTES} まで)")
    try:
        raw = p.read_bytes().decode("utf-8")
        return loads_strict(raw), raw
    except RecursionError as e:
        raise SubmissionReadError("入れ子が深すぎます") from e
    except (UnicodeDecodeError, ValueError) as e:
        raise SubmissionReadError(f"JSON ではありません({type(e).__name__})") from e


def validate_files(
    paths: list[str | os.PathLike[str]],
    db: Database | None = None,
    *,
    today: _date | None = None,
    check_filename: bool = True,
) -> list[Report]:
    """ファイルをまとめて検査する。
    読めない・JSON でないものも誤りとして返す。同じ内容(hash)は 2 件目以降を誤り。"""
    reports: list[Report] = []
    first: dict[str, str] = {}
    for p in paths:
        try:
            data, raw = read_submission(p)
        except SubmissionReadError as e:
            r = Report(path=Path(p).as_posix())
            r.error("read", f"読めません: {e}")
            reports.append(r)
            continue
        except OSError as e:
            r = Report(path=Path(p).as_posix())
            r.error("read", f"読めません: {type(e).__name__}")
            reports.append(r)
            continue
        r = validate_submission(data, db, path=p, raw=raw, today=today, check_filename=check_filename)
        if r.sha256:
            if r.sha256 in first:
                r.error("duplicate", f"{first[r.sha256]} と同じ内容です")
            else:
                first[r.sha256] = str(p)
        reports.append(r)
    return reports


def find_files(root: str | os.PathLike[str]) -> list[Path]:
    """フォルダの下の *.json(並びは決まった順)。ファイルを渡せばそれだけ。"""
    p = Path(root)
    if p.is_file() or p.is_symlink():
        return [p]
    if not p.is_dir():
        return []
    out: list[Path] = []
    for d, dirs, files in os.walk(p, followlinks=False):  # シンボリックリンクのフォルダはたどらない
        dirs[:] = [x for x in dirs if not (Path(d) / x).is_symlink()]
        out += [Path(d) / f for f in files if f.endswith(".json")]
    return sorted(out, key=lambda x: x.as_posix())


# --- まとめる -------------------------------------------------------------------------------------

MAD_SCALE = 1.4826  # 正規分布の標準偏差に合わせる係数
OUTLIER_Z = 3.5  # 外れ値とみなす |x - 中央値| / (1.4826 × MAD)(Iglewicz & Hoaglin の目安)
MAD_FLOOR_REL = 0.02  # MAD の下限(中央値の 2%)。そろいすぎた値で小さな差を外れ値にしない
CONSISTENT_REL = 0.15  # 残った値の幅(最大 - 最小)が中央値の 15% 以内なら「そろった」
MIN_CONSISTENT = 2  # 「そろった」とみなす最小の件数


def _settings_key(s: Settings) -> tuple:
    """設定の鍵。精度は書き方をそろえる(BF16 と bf16、fp8_e4m3fn と fp8 は同じ)。"""
    keys = ("width", "height", "frames", "seconds", "steps")
    return (
        *((k, getattr(s, k)) for k in keys),
        ("precision", norm_precision(s.precision)),
        ("batch", max(1, s.batch or 1)),
    )


def robust_filter(values: list[float]) -> tuple[list[int], list[int], float]:
    """中央値と MAD で外れ値を除く。(残した番号, 外した番号, MAD)。3 件未満は外さない(見分けられない)。"""
    if not values:
        return [], [], 0.0
    med = statistics.median(values)
    mad = statistics.median([abs(v - med) for v in values])
    if len(values) < 3:
        return list(range(len(values))), [], mad
    scale = MAD_SCALE * max(mad, MAD_FLOOR_REL * abs(med), 1e-6)
    keep = [i for i, v in enumerate(values) if abs(v - med) / scale <= OUTLIER_Z]
    drop = [i for i in range(len(values)) if i not in keep]
    return keep, drop, mad


def _mode(values: list[Any]) -> Any:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    counts = Counter(vals)
    best = max(counts.values())
    return sorted((v for v, c in counts.items() if c == best), key=str)[0]


def _med(values: list[float | None], digits: int = 2) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), digits) if vals else None


@dataclass
class Submission:
    path: str
    sha256: str
    data: dict[str, Any]

    @property
    def measurement(self) -> Measurement:
        return to_measurement(self.data)


def load_dir(
    root: str | os.PathLike[str], db: Database | None = None
) -> tuple[list[Submission], list[Report]]:
    """フォルダの提出を読み、検査を通ったもの(誤りの無いもの。注意があっても使う)だけ返す
    (同じ内容は 1 件)。(通ったもの, 全部の検査結果)。"""
    files = find_files(root)
    reports = validate_files(files, db)
    subs: list[Submission] = []
    for p, r in zip(files, reports, strict=True):
        if r.ok and r.sha256:
            data, _ = read_submission(p)
            subs.append(Submission(path=p.as_posix(), sha256=r.sha256, data=data))
    return subs, reports


def aggregate(subs: list[Submission]) -> list[dict[str, Any]]:
    """同じモデル・設定・GPU(と測り方)ごとにまとめる。標準でない提出は使わない。

    最後まで動いた(ok)ものの最大 VRAM で外れ値を除き(robust_filter)、残りの中央値を値にする。
    OOM だけのまとまりは outcome: oom の 1 行にする(見積もりには使わない)。
    結果は決まった順(モデル ID → GPU → 設定)で、同じ入力からは同じ出力になる。"""
    groups: dict[tuple, list[Submission]] = {}
    for sub in subs:
        m = sub.measurement
        if not m.standard:
            continue
        # GPU 名は書き方の違い(NVIDIA / GeForce の有無・大文字小文字・空白)をそろえる
        # VRAM(1 GB に丸める)も鍵に入れる(RTX 4060 Ti の 8 GB と 16 GB を混ぜない)
        key = (
            m.model_id,
            # 型番 ID があれば ID で(Linux と Windows で名前の違う同じ GPU をまとめる)、無ければ名前で
            getattr(m, "gpu_pci_id", None) or gpu_slug(m.gpu_name),
            round(m.gpu_vram_gb or 0),
            m.vram_metric or "",
            _settings_key(m.settings),
        )
        groups.setdefault(key, []).append(sub)
    out: list[dict[str, Any]] = []
    for key in sorted(groups, key=lambda k: json.dumps(k, default=str)):
        members = sorted(groups[key], key=lambda s: s.sha256)
        ms = [s.measurement for s in members]
        ok = [(s, m) for s, m in zip(members, ms, strict=True) if m.outcome == "ok"]
        oom = [m for m in ms if m.outcome == "oom"]
        if ok:
            keep, drop, _ = robust_filter([m.peak_vram_gb for _, m in ok])
            used = [ok[i] for i in keep]
            outliers = [ok[i][0].sha256[:8] for i in drop]
            uv = [m.peak_vram_gb for _, m in used]
            med = statistics.median(uv)
            mad = statistics.median([abs(v - med) for v in uv])
            rel = (max(uv) - min(uv)) / med if med > 0 else 0.0
            base = [m for _, m in used]
            outcome = "ok"
        else:
            used, outliers, base = [], [], oom
            uv = [m.peak_vram_gb for m in oom]
            med = statistics.median(uv) if uv else 0.0
            mad, rel = 0.0, 0.0
            outcome = "oom"
        first = ms[0]
        # consistent は係数の合わせ直し(calibrate)に入れる条件。見積もりは ok の行を 1 件から使う
        # (estimate.usable)。OOM を含む組はそろったとみなさない(同じ構成で動かなかった人がいる)
        consistent = outcome == "ok" and not oom and len(used) >= MIN_CONSISTENT and rel <= CONSISTENT_REL
        settings = {k: v for k, v in asdict(first.settings).items() if v is not None}
        if settings.get("precision"):
            settings["precision"] = norm_precision(settings["precision"])
        out.append(
            {
                "model_id": first.model_id,
                "gpu_name": _mode([m.gpu_name for m in ms]),
                "gpu_vram_gb": _med([m.gpu_vram_gb for m in base] or [first.gpu_vram_gb]),
                # 測った GPU の backend(estimate.measurement_vendor が会社を名前より先にこれで決める。
                # Apple の MPS は GPU 名が "mps" になる)
                "gpu_backend": _mode([m.gpu_backend for m in ms]),
                "gpu_pci_id": _mode([getattr(m, "gpu_pci_id", None) for m in ms]),
                "peak_vram_gb": round(med, 2),
                "settings": settings,
                "peak_ram_gb": _med([m.peak_ram_gb for m in base]),
                "seconds": _med([m.seconds for m in base], 1),
                "driver": _mode([m.driver for m in base]),
                "comfyui_version": _mode([m.comfyui_version for m in base]),
                "offload": _mode([m.offload for m in base]),
                "source": "community(data/contrib を genmodelfit contrib aggregate でまとめたもの)",
                "date": max((m.date or "" for m in ms), default="") or None,
                "note": (
                    f"ほかの利用者の実測 {len(ms)} 件(使った {len(used)} 件・外れ値 {len(outliers)} 件・"
                    f"OOM {len(oom)} 件)の中央値"
                ),
                "outcome": outcome,
                "standard": True,
                "vram_metric": first.vram_metric,
                "origin": ORIGIN_COMMUNITY,
                "n_submissions": len(ms),
                "consistent": consistent,
                "n_used": len(used),
                "n_outliers": len(outliers),
                "n_oom": len(oom),
                "spread_mad_gb": round(mad, 3),
                "spread_rel_pct": round(rel * 100, 1),
                "peak_vram_min_gb": round(min(uv), 2) if uv else None,
                "peak_vram_max_gb": round(max(uv), 2) if uv else None,
                "outlier_submissions": outliers,
                "submissions": [s.sha256[:8] for s in members],
            }
        )
    return out


def community_document(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """同梱の measurements_community.json の中身(日付は提出の最新の日付。作り直しても同じになる)。"""
    return {
        "unit": "GiB",
        "source": "community",
        "contrib_schema": CONTRIB_SCHEMA,
        "date": max((r.get("date") or "" for r in rows), default="") or None,
        "method": (
            "data/contrib の提出を、同じモデル・GPU・測り方・設定ごとにまとめる。"
            "最後まで動いた(ok)ものの最大 VRAM で、"
            f"|x - 中央値| / ({MAD_SCALE} × MAD) > {OUTLIER_Z} を外れ値として除き"
            "(3 件以上のとき。MAD は中央値の "
            f"{MAD_FLOOR_REL:.0%} を下限)、残りの中央値を値にする。ok の行は 1 件から見積もりに使う"
            "(件数は n_used、同じ構成の OOM は n_oom)。使った件数が "
            f"{MIN_CONSISTENT} 以上で、幅が中央値の {CONSISTENT_REL:.0%} 以内で、OOM が無ければ consistent"
            "(係数の合わせ直しに入れる)"
        ),
        "measurements": rows,
    }


def write_community(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dumps(community_document(rows)), encoding="utf-8", newline="\n")
    return p


# --- 精度の点検(毎週の表) ---------------------------------------------------------------------

DEFAULT_THRESHOLD_PCT = 25.0


def accuracy_report(db: Database, threshold_pct: float = DEFAULT_THRESHOLD_PCT) -> dict[str, Any]:
    """校正(leave-one-out)をやり直し、種類ごとの誤差と閾値の判定を返す。

    閾値を超えたかは、(1) 評価できる種類(張り付いていない実測が 2 件以上)の LOO 誤差と、
    (2) ほかの利用者の実測(係数の合わせに使わない測り方のもの)を全件の係数で当てた誤差、で見る。
    作者の照合の行(RTX 3070 の 2 件など)は表に出すが、閾値の判定には入れない
    (docs/accuracy.md に既知の外れとして書いた)。"""
    from .calibrate import calibrate

    result = calibrate(db, date="-")
    sm = result["summary"]
    evaluable = set(sm["evaluable_modalities"])
    by: dict[str, dict[str, Any]] = {}
    breaches: list[dict[str, Any]] = []

    def slot(mod: str) -> dict[str, Any]:
        return by.setdefault(
            mod,
            {
                "loo_n": 0,
                "loo_errors_pct": [],
                "evaluable": mod in evaluable,
                "check_n": 0,
                "check_errors_pct": [],
            },
        )

    for x in result["loo"]:
        if x["censored"]:
            continue
        b = slot(x["modality"])
        b["loo_n"] += 1
        b["loo_errors_pct"].append(x["loo_error_pct"])
        if x["modality"] in evaluable and abs(x["loo_error_pct"]) > threshold_pct:
            breaches.append(
                {
                    "kind": "loo",
                    "model_id": x["model_id"],
                    "gpu_name": x.get("gpu_name"),
                    "origin": x.get("origin"),
                    "error_pct": x["loo_error_pct"],
                }
            )
    for x in result.get("check", []):
        if x["censored"]:
            continue
        b = slot(x["modality"])
        b["check_n"] += 1
        b["check_errors_pct"].append(x["error_pct"])
        if x.get("origin") == ORIGIN_COMMUNITY and abs(x["error_pct"]) > threshold_pct:
            breaches.append(
                {
                    "kind": "community_check",
                    "model_id": x["model_id"],
                    "gpu_name": x.get("gpu_name"),
                    "origin": x.get("origin"),
                    "error_pct": x["error_pct"],
                }
            )
    for b in by.values():
        for k in ("loo", "check"):
            errs = [abs(e) for e in b.pop(f"{k}_errors_pct")]
            b[f"{k}_mean_abs_pct"] = round(sum(errs) / len(errs), 1) if errs else None
            b[f"{k}_max_abs_pct"] = round(max(errs), 1) if errs else None
    own = _own_measurements(db)
    from .estimate import usable

    comm = [m for m in db.measurements if getattr(m, "origin", None) == ORIGIN_COMMUNITY and usable(m)]
    fit_changes: list[dict[str, Any]] = []
    for mid in sorted({m.model_id for m in comm}):
        entry = db.get(mid)
        if entry is not None:
            fit_changes += community_effect(entry, own, [m for m in comm if m.model_id == mid])[1]
    return {
        "threshold_pct": threshold_pct,
        "fit_changes": fit_changes,
        "breach": bool(breaches),
        "breaches": breaches,
        "n_fit": result["n"],
        "coefficients": result["coefficients"],
        "summary": sm,
        "by_modality": dict(sorted(by.items())),
        "loo": result["loo"],
        "check": result.get("check", []),
    }


def report_db(base: Database, community_rows: list[dict[str, Any]]) -> Database:
    """点検に使う一覧: 同梱の一覧のモデル + 同梱の作者の実測 + 今回まとめたほかの利用者の実測。
    この PC のキャッシュの実測と、同梱の古いまとめ(measurements_community.json)は使わない
    (毎回同じ結果にする)。"""
    from .db import bundled_text, parse_measurements

    maint = parse_measurements(json.loads(bundled_text("measurements.json")))
    comm = [Measurement.from_dict(r) for r in community_rows]
    return Database(
        models=base.models,
        measurements=maint + comm,
        generated_at=base.generated_at,
        sources=base.sources,
        origin=base.origin,
    )


__all__ = [
    "CONTRIB_SCHEMA",
    "Report",
    "Submission",
    "accuracy_report",
    "aggregate",
    "build_submission",
    "community_document",
    "content_hash",
    "gpu_slug",
    "load_dir",
    "measurements_from_file",
    "privacy_findings",
    "robust_filter",
    "scrub_text",
    "suggested_path",
    "validate_files",
    "validate_submission",
    "write_community",
    "write_submission",
]
