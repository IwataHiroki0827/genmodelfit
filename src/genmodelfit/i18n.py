# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""表示の言語(en / ja)を決め、人に見せる文を 1 つの言語で出す(#73)。

決め方(上が優先):
    CLI の --lang → 環境変数 GENMODELFIT_LANG → gmf.set_language() → en
既定はいつも英語(OS の言語は見ない。2026-10-25 にオーナーが決めた)。

文は呼んだときに選ぶ: `tr("日本語の文", "English text")`。モジュールの定数に置かず、使うところで呼ぶ
(言語は後から変わるため)。機械が読む値(ID・fit・basis・コード)は訳さない。
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from contextvars import ContextVar

LANGUAGES = ("en", "ja")
ENV_VAR = "GENMODELFIT_LANG"

# CLI の --lang(main の間だけ)・weekly などの固定。ContextVar なので、
# ほかのスレッドのライブラリの呼び出しには効かない
_cli_language: ContextVar[str | None] = ContextVar("genmodelfit_cli_language", default=None)
_api_language: str | None = None  # gmf.set_language()


def normalize(lang: object) -> str | None:
    """'ja' / 'ja_JP.UTF-8' / 'EN' などを en / ja にする。分からなければ None。"""
    if not isinstance(lang, str):
        return None
    head = lang.strip().lower().replace("-", "_").split("_", 1)[0].split(".", 1)[0]
    return head if head in LANGUAGES else None


def set_language(lang: str | None) -> None:
    """Python から表示の言語を決める(`en` / `ja`。None で決めたのを取り消し、既定の en に戻す)。

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
    forced_lang = _cli_language.get()
    if forced_lang is not None:
        return forced_lang
    env = normalize(os.environ.get(ENV_VAR))
    if env is not None:
        return env
    if _api_language is not None:
        return _api_language
    return "en"


def _set_cli_language(lang: str | None) -> None:
    """今の文脈の言語を固定する(None で外す)。戻すときは forced() を使う。"""
    _cli_language.set(normalize(lang) if lang is not None else None)


@contextlib.contextmanager
def forced(lang: str | None) -> Iterator[None]:
    """この中だけ言語を固定する(CLI の --lang、日本語で書く PR の本文など)。None なら固定しない。"""
    token = _cli_language.set(normalize(lang) if lang is not None else None)
    try:
        yield
    finally:
        _cli_language.reset(token)


class Text(str):
    """tr の返す文。今の言語の文で、日本語の文も .ja に持つ(一覧に保存する文を日本語にするため)。"""

    ja: str

    def __new__(cls, value: str, ja: str | None = None) -> Text:
        obj = super().__new__(cls, value)
        obj.ja = value if ja is None else ja
        return obj

    def __reduce__(self) -> tuple[type[Text], tuple[str, str]]:  # copy・pickle でも .ja を残す
        return (Text, (str(self), self.ja))


def tr(ja: str, en: str) -> str:
    """今の言語の文を返す(日本語の文は .ja でも読める。Text)。"""
    return Text(ja if get_language() == "ja" else en, ja)


def ja_of(text: object) -> str:
    """日本語の文(tr で作った文なら .ja、そうでなければそのまま)。"""
    return getattr(text, "ja", None) or str(text)


def stored_message(e: BaseException) -> str:
    """一覧などに保存する例外の文。言語によらず日本語(FetchError などが ja_message を持つ)。"""
    return getattr(e, "ja_message", None) or str(e)


def pick(text: dict | str | None) -> str | None:
    """{"en": ..., "ja": ...} の形の説明から今の言語のものを選ぶ(無ければもう片方)。"""
    if not isinstance(text, dict):
        return text
    lang = get_language()
    other = "en" if lang == "ja" else "ja"
    return text.get(lang) or text.get(other) or None
