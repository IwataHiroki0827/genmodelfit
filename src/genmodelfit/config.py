# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit config` の設定ファイル(#111。docs/download.ja.md「genmodelfit config」)。

読むのはコマンドライン(cli.py)だけ。Python のライブラリの関数は読まない(アプリの動きを利用者の PC の
設定で変えないため)。ファイルは platformdirs の user_config_dir("genmodelfit")/config.json
(環境変数 GENMODELFIT_CONFIG_DIR で変えられる)。壊れたファイルは読まずに注意を返す(落ちない)。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .i18n import LANGUAGES, tr

CONFIG_DIR_ENV = "GENMODELFIT_CONFIG_DIR"
FILE_NAME = "config.json"
# 決められるキー(`genmodelfit config set KEY VALUE`)。ほかのキーは断る
KEYS = ("comfyui-dir", "lang")


class ConfigError(RuntimeError):
    """設定ファイルを書けない・値が正しくない(CLI がそのまま誤りの文として出す)。"""


def config_dir() -> Path:
    env = os.environ.get(CONFIG_DIR_ENV, "").strip()
    if env:
        return Path(env).expanduser()
    from platformdirs import user_config_dir

    return Path(user_config_dir("genmodelfit", appauthor=False))


def config_path() -> Path:
    return config_dir() / FILE_NAME


def _broken(path: Path, why: str) -> str:
    return tr(
        f"設定ファイルを読めないので使いません({why}): {path}(直すか消してください)",
        f"The config file cannot be read and is ignored ({why}): {path} (fix or delete it)",
    )


def read_raw() -> tuple[dict, str | None]:
    """ファイルの中身(知らないキーも)と、読めなかったときの注意。無いファイルは ({}, None)。"""
    path = config_path()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}, None
    except (OSError, UnicodeDecodeError) as e:
        return {}, _broken(path, type(e).__name__)
    try:
        data = json.loads(text)
    except ValueError:
        return {}, _broken(path, "JSON")
    if not isinstance(data, dict):
        return {}, _broken(path, tr("JSON のオブジェクトではない", "not a JSON object"))
    return data, None


def load() -> tuple[dict[str, str], list[str]]:
    """使える値だけ(キー → 文字)と注意。形の正しくない値は使わずに注意を返す。"""
    data, warning = read_raw()
    notes = [warning] if warning else []
    out: dict[str, str] = {}
    for key in KEYS:
        if key not in data:
            continue
        value = data[key]
        if not isinstance(value, str) or not value.strip():
            notes.append(
                tr(
                    f"設定ファイルの {key} が文字ではないので使いません: {config_path()}",
                    f"{key} in the config file is not a string and is ignored: {config_path()}",
                )
            )
            continue
        if key == "lang" and value not in LANGUAGES:
            notes.append(
                tr(
                    f"設定ファイルの lang が正しくないので使いません({value!r}。使えるのは"
                    f" {', '.join(LANGUAGES)})",
                    f"lang in the config file is not valid and is ignored ({value!r}; choose from"
                    f" {', '.join(LANGUAGES)})",
                )
            )
            continue
        out[key] = value
    return out, notes


def _write(data: dict) -> Path:
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except OSError as e:
        raise ConfigError(
            tr(f"設定ファイルを書けません: {path}({e})", f"Cannot write the config file: {path} ({e})")
        ) from e
    return path


def _check_key(key: str) -> None:
    if key not in KEYS:
        raise ConfigError(
            tr(
                f"知らないキーです: {key}(決められるのは {', '.join(KEYS)})",
                f"Unknown key: {key} (choose from {', '.join(KEYS)})",
            )
        )


def _writable() -> dict:
    data, warning = read_raw()
    if warning:  # 壊れたファイルを黙って上書きしない
        raise ConfigError(warning)
    return data


def set_value(key: str, value: str) -> Path:
    """値を保存する(値の確かめは呼ぶ側。comfyui-dir は ComfyUI に見えること)。"""
    _check_key(key)
    data = _writable()
    data[key] = value
    return _write(data)


def unset(key: str) -> tuple[Path, bool]:
    """値を消す。(ファイル, 消したか)。"""
    _check_key(key)
    data = _writable()
    if key not in data:
        return config_path(), False
    del data[key]
    return _write(data), True
