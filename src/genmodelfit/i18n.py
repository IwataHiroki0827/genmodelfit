# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""表示の言語(en / ja)を決め、人に見せる文を 1 つの言語で出す(#73)。

決め方(上が優先):
    CLI の --lang → 環境変数 GENMODELFIT_LANG → gmf.set_language() → OS の言語(日本語なら ja)→ en

文は呼んだときに選ぶ: `tr("日本語の文", "English text")`。モジュールの定数に置かず、使うところで呼ぶ
(言語は後から変わるため)。機械が読む値(ID・fit・basis・コード)は訳さない。
OS の言語を調べるのは環境変数と(Windows では)GetUserDefaultUILanguage だけ。外のコマンドは呼ばず、
失敗しても例外を出さない(en にする)。
"""

from __future__ import annotations

import os
import sys

LANGUAGES = ("en", "ja")
ENV_VAR = "GENMODELFIT_LANG"

_cli_language: str | None = None  # CLI の --lang(main の間だけ)
_api_language: str | None = None  # gmf.set_language()
_os_cache: str | None = None


def normalize(lang: object) -> str | None:
    """'ja' / 'ja_JP.UTF-8' / 'EN' などを en / ja にする。分からなければ None。"""
    if not isinstance(lang, str):
        return None
    head = lang.strip().lower().replace("-", "_").split("_", 1)[0].split(".", 1)[0]
    return head if head in LANGUAGES else None


def set_language(lang: str | None) -> None:
    """Python から表示の言語を決める(`en` / `ja`。None で決めたのを取り消し、OS の言語に戻す)。

    CLI の --lang と環境変数 GENMODELFIT_LANG のほうが優先する。"""
    global _api_language
    if lang is None:
        _api_language = None
        return
    value = normalize(lang)
    if value is None or value != lang.strip().lower():
        raise ValueError(f"language must be one of {', '.join(LANGUAGES)}: {lang!r}")
    _api_language = value


def get_language() -> str:
    """今の表示の言語(`en` か `ja`)。"""
    if _cli_language is not None:
        return _cli_language
    env = normalize(os.environ.get(ENV_VAR))
    if env is not None:
        return env
    if _api_language is not None:
        return _api_language
    return _os_language()


def _set_cli_language(lang: str | None) -> None:
    """CLI の --lang(cli.main が使う。終わったら None に戻す)。"""
    global _cli_language
    _cli_language = normalize(lang) if lang is not None else None


def tr(ja: str, en: str) -> str:
    """今の言語の文を返す。"""
    return ja if get_language() == "ja" else en


def pick(text: dict | str | None) -> str | None:
    """{"en": ..., "ja": ...} の形の説明から今の言語のものを選ぶ(無ければもう片方)。"""
    if not isinstance(text, dict):
        return text
    lang = get_language()
    other = "en" if lang == "ja" else "ja"
    return text.get(lang) or text.get(other) or None


def _os_language() -> str:
    global _os_cache
    if _os_cache is None:
        _os_cache = _detect_os_language()
    return _os_cache


def _detect_os_language() -> str:
    try:
        for name in ("LC_ALL", "LC_MESSAGES", "LANG"):
            value = os.environ.get(name)
            if value:  # 最初に値のあるものだけを見る(POSIX の優先順)
                if value.lower().startswith("ja"):
                    return "ja"
                break
        if sys.platform == "win32":  # Windows は表示の言語(Git Bash などが LANG=en_US を置いても)
            import ctypes

            langid = ctypes.windll.kernel32.GetUserDefaultUILanguage()  # type: ignore[attr-defined]
            if int(langid) & 0x3FF == 0x11:  # 主言語が日本語(LANG_JAPANESE)
                return "ja"
    except Exception:  # 調べられなければ en
        pass
    return "en"
