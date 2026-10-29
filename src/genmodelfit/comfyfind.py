# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""この PC の ComfyUI のフォルダを探す(#111。docs/download.ja.md「置き場所の決め方」)。

使うのは `genmodelfit download` だけ(Python のライブラリの関数は使わない)。起動中の ComfyUI に尋ねる
(127.0.0.1 だけ)のは落とすときだけで、`--list` と読むだけのコマンドは通信しない。
ComfyUI(GPL-3.0)・ComfyUI デスクトップ版(GPL-3.0)・comfy-cli(GPL-3.0)のコードは使わない。
見分け方・ファイルの場所は、それぞれのソースを読んで同じ決まりを書いた(出典は各所のコメント)。
"""

from __future__ import annotations

import configparser
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import tr

# ComfyUI に見える印(5 つのうち 4 つ以上)。comfy-cli と同じ見分け方
# (Comfy-Org/comfy-cli comfy_cli/workspace_manager.py:58-61@c272a5e)
MARKERS = ("main.py", "comfy", "nodes.py", "comfy_extras", "comfy_api")
# デスクトップ版の置き場所(basePath)の見分け方。本体のコードは別の場所にあり、
# `--base-directory basePath` で起動する(Comfy-Org/desktop src/main-process/comfyServer.ts:108@e2d964b)ので、
# 上の印は無い。デスクトップ版と同じく、この 5 つのフォルダがそろえば ComfyUI の置き場所とみなす
# (src/config/comfyConfigManager.ts:51-54@e2d964b)
DESKTOP_DIRS = ("models", "input", "user", "output", "custom_nodes")
# 起動中の ComfyUI を探すポート。8188 は ComfyUI の既定(comfyanonymous/ComfyUI comfy/cli_args.py:64@426f5ee)、
# 8000 はデスクトップ版の既定(Comfy-Org/desktop src/constants.ts:157-161@e2d964b)
PROBE_PORTS = (8188, 8000)
PROBE_TIMEOUT = 2.0
# ComfyUI の古い名前(comfyanonymous/ComfyUI folder_paths.py:112-115@426f5ee)
LEGACY = {"unet": "diffusion_models", "clip": "text_encoders"}

# 見つけ方の符号(JSON の destination_from・destination_candidates[].from)
SOURCES = ("cwd", "running", "desktop", "comfy_cli", "common")


@dataclass
class Candidate:
    """自動で見つけた ComfyUI のフォルダ。"""

    path: Path
    source: str  # SOURCES
    detail: str = ""  # 例: 起動中の ComfyUI のポート、設定ファイルの場所
    models_dir: Path | None = None  # 起動中の ComfyUI が最初に見る models(path/models と違うときだけ)
    extra_configs: list[Path] = field(default_factory=list)  # ほかに読む extra_model_paths の YAML

    def to_dict(self) -> dict:
        return {"comfyui_dir": str(self.path), "from": self.source, "detail": self.detail}


@dataclass
class Places:
    """探す場所(試験で差し替える)。platform は sys.platform の形(win32 / darwin / linux)。"""

    platform: str
    home: Path
    appdata: Path | None = None  # Windows の %APPDATA%
    xdg_config: Path | None = None  # Linux の $XDG_CONFIG_HOME
    drives: list[Path] = field(default_factory=list)  # Windows の固定ドライブの直下(C:\ など)
    folders: list[Path] = field(default_factory=list)  # ドキュメント・デスクトップ・ダウンロード


def looks_like_comfyui(path: Path) -> bool:
    """ComfyUI のフォルダに見えるか(印 5 つのうち 4 つ以上、またはデスクトップ版の置き場所)。"""
    try:
        if not path.is_dir():
            return False
        if sum((path / m).exists() for m in MARKERS) >= 4:
            return True
        return all((path / d).is_dir() for d in DESKTOP_DIRS)
    except OSError:
        return False


def _key(path: Path) -> str:
    try:
        p = path.resolve()
    except OSError:
        p = Path(os.path.abspath(path))
    return os.path.normcase(str(p))


def _fixed_drives() -> list[Path]:
    """Windows の固定ドライブ(GetDriveTypeW が DRIVE_FIXED = 3)の直下。ほかの OS は空。"""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        import string

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        mask = kernel.GetLogicalDrives()
        out = []
        for i, letter in enumerate(string.ascii_uppercase):
            if mask & (1 << i):
                root = f"{letter}:\\"
                if kernel.GetDriveTypeW(ctypes.c_wchar_p(root)) == 3:
                    out.append(Path(root))
        return out
    except (OSError, AttributeError, ValueError):
        return []


def current_places() -> Places:
    """この PC の探す場所。"""
    from platformdirs import user_desktop_dir, user_documents_dir, user_downloads_dir

    home = Path.home()
    folders = []
    for fn in (user_documents_dir, user_desktop_dir, user_downloads_dir):
        try:
            folders.append(Path(fn()))
        except Exception:
            continue
    appdata = os.environ.get("APPDATA")
    xdg = os.environ.get("XDG_CONFIG_HOME")
    return Places(
        platform=sys.platform,
        home=home,
        appdata=Path(appdata) if appdata else None,
        xdg_config=Path(xdg) if xdg else None,
        drives=_fixed_drives(),
        folders=folders,
    )


# --- 設定ファイルから -----------------------------------------------------------------------------


def desktop_user_data(places: Places) -> Path | None:
    """デスクトップ版(Electron。productName が ComfyUI: Comfy-Org/desktop package.json:3@e2d964b)の
    userData。Electron の決まり: Windows %APPDATA%\\ComfyUI、macOS ~/Library/Application Support/ComfyUI、
    Linux $XDG_CONFIG_HOME(無ければ ~/.config)/ComfyUI。"""
    if places.platform == "win32":
        return places.appdata / "ComfyUI" if places.appdata else None
    if places.platform == "darwin":
        return places.home / "Library" / "Application Support" / "ComfyUI"
    return (places.xdg_config or places.home / ".config") / "ComfyUI"


def desktop_candidates(places: Places) -> list[Candidate]:
    """デスクトップ版の設定 config.json の basePath(src/store/desktopConfig.ts:44・
    src/store/desktopSettings.ts:8@e2d964b)。モデルの置き場所の設定は userData の
    extra_models_config.yaml(src/config/comfyServerConfig.ts:80,105@e2d964b)。"""
    ud = desktop_user_data(places)
    if ud is None:
        return []
    try:
        data = json.loads((ud / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return []
    base = data.get("basePath") if isinstance(data, dict) else None
    if not isinstance(base, str) or not base.strip():
        return []
    return [
        Candidate(
            Path(base).expanduser(),
            "desktop",
            str(ud / "config.json"),
            extra_configs=[ud / "extra_models_config.yaml"],
        )
    ]


def comfy_cli_candidates(places: Places) -> list[Candidate]:
    """comfy-cli の設定(config.ini の [DEFAULT] default_workspace・recent_workspace)と既定の場所
    (Comfy-Org/comfy-cli comfy_cli/constants.py:26-36,42-44・config_manager.py:17-21@c272a5e)。"""
    if places.platform == "win32":
        conf = places.home / "AppData" / "Local" / "comfy-cli"
        default = places.home / "Documents" / "comfy" / "ComfyUI"
    elif places.platform == "darwin":
        conf = places.home / "Library" / "Application Support" / "comfy-cli"
        default = places.home / "Documents" / "comfy" / "ComfyUI"
    else:
        conf = places.home / ".config" / "comfy-cli"
        default = places.home / "comfy" / "ComfyUI"
    out = []
    ini = conf / "config.ini"
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(ini, encoding="utf-8")
    except (configparser.Error, OSError, UnicodeDecodeError):
        parser = configparser.ConfigParser(interpolation=None)
    for key in ("default_workspace", "recent_workspace"):
        value = parser.defaults().get(key, "").strip()
        if value:
            out.append(Candidate(Path(value).expanduser(), "comfy_cli", f"{ini} ({key})"))
    out.append(Candidate(default, "comfy_cli", tr("comfy-cli の既定の場所", "comfy-cli default")))
    return out


def common_candidates(places: Places) -> list[Candidate]:
    """よくある場所(深く探さない): 固定ドライブ・ドキュメント・デスクトップ・ダウンロード・ホームの直下の
    ComfyUI と、ポータブル版の ComfyUI_windows_portable/ComfyUI。"""
    roots = [*places.drives, *places.folders, places.home]
    out = []
    for r in roots:
        out.append(Candidate(r / "ComfyUI", "common"))
        out.append(Candidate(r / "ComfyUI_windows_portable" / "ComfyUI", "common"))
    return out


# --- 起動中の ComfyUI -------------------------------------------------------------------------------


def _probe_port(port: int, timeout: float) -> dict | None:
    """GET http://127.0.0.1:<port>/internal/folder_paths(ComfyUI は内部用と書いている:
    comfyanonymous/ComfyUI api_server/routes/internal/internal_routes.py:8-13,47-52@426f5ee)。
    答えが無い・形が違うときは None。転送は追わない。プロキシは使わない(127.0.0.1 だけ)。"""
    from .fetch import make_session

    url = f"http://127.0.0.1:{port}/internal/folder_paths"
    try:
        with make_session() as s:
            r = s.get(
                url,
                timeout=timeout,
                allow_redirects=False,
                proxies={"http": None, "https": None, "all": None},  # type: ignore[dict-item]
            )
            if r.status_code != 200:
                return None
            data = r.json()
    except Exception:
        return None
    return normalize_folder_paths(data)


def normalize_folder_paths(data) -> dict | None:
    """/internal/folder_paths の答えを {キー: 最初のパス} にそろえる。ComfyUI の版で形が違う: 文字列
    (古い版)・パスの並び(folder_names_and_paths[key][0]。0.19 など)・[パスの並び, 拡張子] の並び。
    読めないキーは飛ばす。何も読めなければ None。"""
    if not isinstance(data, dict):
        return None
    out: dict[str, str] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            continue
        first = v
        for _ in range(3):  # [[paths], exts] → [paths] → path
            if isinstance(first, (list, tuple)):
                first = first[0] if first else None
            else:
                break
        if isinstance(first, str) and first:
            out[k] = first
    return out or None


def from_folder_paths(data: dict, port: int) -> Candidate | None:
    """/internal/folder_paths の答え(各キーの最初のパス)→ ComfyUI のフォルダ。custom_nodes の親を使い、
    無ければ models の親。models は、名前がキーと同じフォルダの親で一番多いもの。"""
    parents: dict[str, tuple[Path, int]] = {}
    for key, value in data.items():
        p = Path(value)
        if not p.is_absolute() or key == "custom_nodes":
            continue
        names = {key, *(old for old, new in LEGACY.items() if new == key)}
        if key == "download_model_base":
            k = _key(p)
            parents[k] = (p, parents.get(k, (p, 0))[1] + 100)  # デスクトップ版の models の置き場所
        elif p.name in names:
            k = _key(p.parent)
            parents[k] = (p.parent, parents.get(k, (p.parent, 0))[1] + 1)
    models = max(parents.values(), key=lambda x: x[1])[0] if parents else None
    base = None
    cn = data.get("custom_nodes")
    if cn and Path(cn).is_absolute():
        base = Path(cn).parent
    elif models is not None:
        base = models.parent
    if base is None or not looks_like_comfyui(base):
        return None
    if models is not None and not usable_models_dir(models):
        models = None  # 書いてよい場所に見えない(L11a): 使わない
    if models is not None and _key(models) == _key(base / "models"):
        models = None
    return Candidate(base, "running", f"127.0.0.1:{port}", models_dir=models)


def usable_models_dir(p: Path) -> bool:
    """起動中のサーバーが答えた models のフォルダを、置き場所に使ってよいか(L11a): あるフォルダで、OS の
    フォルダ(C:\\Windows・Program Files・/usr・/etc など)やドライブの直下でないこと。"""
    try:
        r = p.resolve()
        if not r.is_dir():
            return False
    except OSError:
        return False
    if r.parent == r or len(r.parts) <= 1:
        return False
    low = [x.lower() for x in r.parts]
    system = {"windows", "program files", "program files (x86)", "programdata", "system32"}
    if os.name == "nt" and len(low) > 1 and low[1] in system:
        return False
    posix_system = {"usr", "etc", "bin", "sbin", "lib", "proc", "sys", "dev", "boot"}
    return not (os.name != "nt" and len(low) > 1 and low[1] in posix_system)


def comfy_port_answering(ports: Iterable[int] = PROBE_PORTS, timeout: float = PROBE_TIMEOUT) -> int | None:
    """答えたポート(comfy_port_status の 1 つ目)。"""
    found = comfy_port_status(ports, timeout)
    return found[0] if found else None


def comfy_port_status(
    ports: Iterable[int] = PROBE_PORTS, timeout: float = PROBE_TIMEOUT
) -> tuple[int, int] | None:
    """127.0.0.1 のこのポートで ComfyUI(/system_stats)が答えるか。答えたポート(L3a。フォルダを読めない
    ComfyUI でも、2 台目を同じフォルダで起動しないように)。GENMODELFIT_COMFYUI_SEARCH=0 なら見ない。"""
    from .fetch import make_session

    if search_disabled():
        return None
    for port in ports:
        try:
            with make_session() as s:
                r = s.get(
                    f"http://127.0.0.1:{port}/system_stats",
                    timeout=timeout,
                    allow_redirects=False,
                    proxies={"http": None, "https": None, "all": None},  # type: ignore[dict-item]
                )
                # 何かが HTTP で答えれば(401・403 なども)、そこにはもう何かがいる(L3d): 2 台目を起動しない
                if r.status_code:
                    return port, int(r.status_code)
        except Exception:
            continue
    return None


def running_candidates(ports: Iterable[int] = PROBE_PORTS, timeout: float = PROBE_TIMEOUT) -> list[Candidate]:
    """起動中の ComfyUI(ポートごとに並べて尋ねる。Windows は閉じたポートでも 2 秒ほどかかるため)。"""
    from concurrent.futures import ThreadPoolExecutor

    ports = list(ports)
    if not ports:
        return []
    with ThreadPoolExecutor(max_workers=len(ports)) as pool:
        answers = list(pool.map(lambda p: _probe_port(p, timeout), ports))
    out = []
    for port, data in zip(ports, answers, strict=True):
        if data is not None:
            c = from_folder_paths(data, port)
            if c is not None:
                out.append(c)
    return out


# --- まとめ -----------------------------------------------------------------------------------------


# 自動で探すのを止める(0 / off / no / false)。組み込むアプリ・スクリプト・試験用
# (docs/download.md「置き場所の決め方」)
SEARCH_ENV = "GENMODELFIT_COMFYUI_SEARCH"


def search_disabled() -> bool:
    return os.environ.get(SEARCH_ENV, "").strip().lower() in ("0", "off", "no", "false")


def find_candidates(*, probe: bool, cwd: Path | None = None, places: Places | None = None) -> list[Candidate]:
    """見つけた ComfyUI のフォルダ(ComfyUI に見えるものだけ。同じフォルダは 1 つ)。
    環境変数 GENMODELFIT_COMFYUI_SEARCH=0 なら探さない(空)。

    順: 今いるフォルダ → 起動中の ComfyUI(probe のときだけ)→ デスクトップ版 → comfy-cli → よくある場所。"""
    if search_disabled():
        return []
    places = places or current_places()
    cwd = cwd if cwd is not None else Path.cwd()
    found: list[Candidate] = []
    if looks_like_comfyui(cwd):
        found.append(Candidate(cwd, "cwd"))
    if probe:
        found += running_candidates()
    for c in [*desktop_candidates(places), *comfy_cli_candidates(places), *common_candidates(places)]:
        if looks_like_comfyui(c.path):
            found.append(c)
    out: list[Candidate] = []
    seen: dict[str, Candidate] = {}
    for c in found:
        k = _key(c.path)
        if k in seen:
            # 同じフォルダ: 先に見つけたものを残し、足りない情報(起動中の models・デスクトップ版の YAML)を足す
            first = seen[k]
            first.models_dir = first.models_dir or c.models_dir
            first.extra_configs += [p for p in c.extra_configs if p not in first.extra_configs]
            continue
        seen[k] = c
        out.append(c)
    return out


def desktop_extra_configs(comfyui_dir: Path, places: Places | None = None) -> list[Path]:
    """そのフォルダがデスクトップ版の basePath なら、デスクトップ版の extra_models_config.yaml。"""
    places = places or current_places()
    k = _key(comfyui_dir)
    out = []
    for c in desktop_candidates(places):
        if _key(c.path) == k:
            out += c.extra_configs
    return out


# --- extra_model_paths.yaml の is_default --------------------------------------------------------


class YamlUnreadable(ValueError):
    """この小さな読み方で読めない YAML(使わずに注意を出す)。"""


@dataclass
class _Scalar:
    text: str
    quoted: bool


def _strip_comment(line: str) -> str:
    quote = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i].rstrip()
    return line.rstrip()


def _scalar(raw: str) -> _Scalar:
    raw = raw.strip()
    if raw[:1] in ("{", "[", "&", "*", "!", "%", "@", "`", ">"):
        raise YamlUnreadable(raw)
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        inner = raw[1:-1]
        if raw[0] == '"':
            if "\\" in inner:
                inner = json.loads(raw)  # 二重引用符の \ の書き方(JSON と同じ範囲だけ)
        else:
            inner = inner.replace("''", "'")
        return _Scalar(inner, True)
    if raw[:1] in ("'", '"'):
        raise YamlUnreadable(raw)
    return _Scalar(raw, False)


def parse_extra_yaml(text: str) -> dict[str, dict[str, _Scalar] | None]:
    """extra_model_paths.yaml の形(節 → キー: 値、値は 1 行か `|` の複数行)だけを、依存を足さずに読む。
    ほかの書き方(流れの形・アンカー・入れ子・リスト・タブの字下げ)は YamlUnreadable。"""
    lines = text.splitlines()
    out: dict[str, dict[str, _Scalar] | None] = {}
    section: str | None = None
    i = 0
    while i < len(lines):
        line = _strip_comment(lines[i])
        i += 1
        if not line.strip() or line.strip() in ("---", "..."):
            continue
        if "\t" in line[: len(line) - len(line.lstrip())]:
            raise YamlUnreadable("tab")
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()
        if body.startswith("- ") or body == "-" or ":" not in body:
            raise YamlUnreadable(body)
        key, _, rest = body.partition(":")
        key = _scalar(key).text
        rest = rest.strip()
        if indent == 0:
            if rest and rest not in ("~", "null"):
                raise YamlUnreadable(body)
            section = key
            out[section] = None
            continue
        if section is None:
            raise YamlUnreadable(body)
        if out[section] is None:
            out[section] = {}
        sec = out[section]
        assert sec is not None
        if rest in ("|", "|-", "|+"):
            block: list[str] = []
            block_indent = None
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    i += 1
                    continue
                ni = len(nxt) - len(nxt.lstrip(" "))
                if ni <= indent:
                    break
                if block_indent is None:
                    block_indent = ni
                block.append(nxt.strip())
                i += 1
            sec[key] = _Scalar("\n".join(block), True)
        elif not rest:
            # 値が無い(null)か、入れ子(この形は読まない)
            if i < len(lines):
                nxt = _strip_comment(lines[i])
                if nxt.strip() and len(nxt) - len(nxt.lstrip(" ")) > indent:
                    raise YamlUnreadable(body)
            sec[key] = _Scalar("", False)
        else:
            sec[key] = _scalar(rest)
    return out


def _truthy(v: _Scalar) -> bool:
    """ComfyUI は is_default を Python の真偽で見る(utils/extra_config.py:21-23@426f5ee)。PyYAML の
    真偽の書き方(true / yes / on …)と数を読み、引用符の文字は空でなければ真
    (デスクトップ版は "true" と書く)。"""
    if v.quoted:
        return bool(v.text)
    t = v.text.strip().lower()
    return t not in ("", "~", "null", "false", "no", "off", "0", "0.0")


def _last_line(v: _Scalar) -> str | None:
    """1 つのキーの値の、ComfyUI が最初に見るパス。is_default の節は 1 行ずつ先頭に入れる
    (folder_paths.py:382-398@426f5ee)ので、最後の行が先頭になる。"""
    parts = [p for p in v.text.split("\n") if p]
    return parts[-1] if parts else None


@dataclass
class DefaultSection:
    """is_default の節から決めた置き場所。"""

    file: Path
    section: str
    root: Path  # 置き場所(<root>/<folder>/<file>)
    folders: frozenset[str] | None  # 節に書いてある models のフォルダ(None = download_model_base で全部)

    def to_dict(self) -> dict:
        return {
            "file": str(self.file),
            "section": self.section,
            "root": str(self.root),
            "folders": sorted(self.folders) if self.folders is not None else None,
        }


def default_section(paths: Iterable[Path]) -> tuple[DefaultSection | None, list[str]]:
    """YAML(ComfyUI が読む順)の is_default の節 → 置き場所と注意。後に読む節ほど先頭になるので、
    最後に使える節を使う(utils/extra_config.py:6-35・main.py:139-145@426f5ee)。

    置き場所: 節に download_model_base があればそこ(ComfyUI-Manager・デスクトップ版の書き方:
    Comfy-Org/desktop src/config/comfyServerConfig.ts:82-99@e2d964b)。無ければ、models のフォルダの
    キーの値が「<共通の親>/<キーの名前>」のときの共通の親。どちらでもなければ使わない(注意)。"""
    chosen: DefaultSection | None = None
    notes: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except (OSError, UnicodeDecodeError):
            notes.append(tr(f"読めません: {path}", f"cannot read: {path}"))
            continue
        try:
            doc = parse_extra_yaml(text)
        except (YamlUnreadable, ValueError):
            notes.append(
                tr(
                    f"{path} は genmodelfit の読める形ではないので、is_default を確かめていません"
                    "(置き場所が違うなら --comfyui-dir か --output-dir で決める)",
                    f"{path} is not in a form genmodelfit can read, so is_default was not checked"
                    " (choose the place with --comfyui-dir or --output-dir if it is wrong)",
                )
            )
            continue
        yaml_dir = Path(os.path.abspath(path)).parent
        for name, sec in doc.items():
            if not sec or "is_default" not in sec or not _truthy(sec["is_default"]):
                continue
            base: Path | None = None
            if "base_path" in sec and sec["base_path"].text:
                b = os.path.expandvars(os.path.expanduser(sec["base_path"].text))
                base = Path(b) if os.path.isabs(b) else yaml_dir / b
            root, folders = _section_root(sec, base, yaml_dir)
            if root is None:
                notes.append(
                    tr(
                        f"{path} の節 {name} は is_default ですが、models のフォルダの並びが"
                        " ComfyUI と違うので使いません(ComfyUI の models に置きます)",
                        f"section {name} of {path} is is_default, but its folders are not laid out like"
                        " ComfyUI's models, so it is not used (files go to ComfyUI's models)",
                    )
                )
                continue
            root = Path(os.path.normpath(root))
            if not root.is_dir():
                notes.append(
                    tr(
                        f"{path} の節 {name}(is_default)のフォルダがありません: {root}"
                        "(ComfyUI の models に置きます)",
                        f"the folder of section {name} (is_default) in {path} does not exist: {root}"
                        " (files go to ComfyUI's models)",
                    )
                )
                continue
            chosen = DefaultSection(path, name, root, folders)
    return chosen, notes


_NOT_MODEL_KEYS = {"base_path", "is_default", "custom_nodes", "download_model_base"}


def _section_root(
    sec: dict[str, _Scalar], base: Path | None, yaml_dir: Path
) -> tuple[Path | None, frozenset | None]:
    def full(rel: str) -> Path:
        if base is not None:
            return base / rel
        p = Path(rel)
        return p if p.is_absolute() else yaml_dir / rel

    if "download_model_base" in sec:
        last = _last_line(sec["download_model_base"])
        return (full(last), None) if last else (None, None)
    prefixes: set[str] = set()
    folders: set[str] = set()
    for key, value in sec.items():
        if key in _NOT_MODEL_KEYS:
            continue
        last = _last_line(value)
        if not last:
            continue
        p = Path(os.path.normpath(full(last)))
        name = LEGACY.get(key, key)
        if LEGACY.get(p.name, p.name) != name:
            return None, None
        prefixes.add(_key(p.parent))
        folders.add(name)
    if len(prefixes) != 1:
        return None, None
    first = next(v for k, v in sec.items() if k not in _NOT_MODEL_KEYS and _last_line(v))
    return Path(os.path.normpath(full(_last_line(first) or ""))).parent, frozenset(folders)
