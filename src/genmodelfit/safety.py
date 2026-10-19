"""一覧から外すモデルの判定(未成年の性的表現を目的とするもの)。

CLAUDE.md の決まり: 無検閲のモデルは印を付けて残すが、未成年の性的表現を目的とするものは除外する。
外したものは一覧に載せず、件数だけを `sources.excluded` に数える(名前も残さない)。

語は 2 種類に分ける(docs/sources.md に同じ表を載せる):

- STRONG(それだけで未成年の性的表現を指す語): どのモデルにも当てる
- CONTEXT(単独では無害だが、成人向けのモデルに付くと危ない語): 無検閲(uncensored)のモデルだけに当てる

ID・名前・タグにあれば外す。説明文(モデルカード)では、否定・禁止の文("No underage content." や
"Do not use it to generate:" に続く箇条書きなど。成人向けモデルの注意書きによくある)の中にあるものは数えず、
CONTEXT の語は同じ文に性的な語があるときだけ数える。

語は単語の境界で照合する(`minor` は `minority` に当たらない、`kid` は `kidney` に当たらない)。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

STRONG_TERMS = (
    "loli",
    "lolis",
    "lolicon",
    "shota",
    "shotas",
    "shotacon",
    "jailbait",
    "preteen",
    "preteens",
    "pedo",
    "pedophile",
    "pedophilia",
    "paedophile",
    "toddlercon",
    "underage",
)

CONTEXT_TERMS = (
    "child",
    "children",
    "kid",
    "kids",
    "minor",
    "minors",
    "teen",
    "teens",
    "schoolgirl",
    "schoolgirls",
    "schoolboy",
    "little girl",
    "little girls",
    "little boy",
    "little boys",
    "young girl",
    "young girls",
    "young boy",
    "young boys",
    "toddler",
    "infant",
    "cub",
)

# 説明文で CONTEXT の語と同じ文にあると外す語
SEXUAL_TERMS = (
    "sex",
    "sexual",
    "sexy",
    "nude",
    "nudes",
    "nudity",
    "naked",
    "porn",
    "pornographic",
    "erotic",
    "erotica",
    "nsfw",
    "explicit",
    "lewd",
    "hentai",
    "fetish",
    "genitals",
    "breasts",
)

# 説明文で、この語を含む文の中の CONTEXT の語は数えない(注意書き・禁止の文)
_NEGATION = re.compile(
    r"\b(no|not|never|nor|without|prohibit\w*|forbid\w*|ban\w*|disallow\w*|illegal|must not|do not|don't|"
    r"cannot|can't|refuse\w*|avoid\w*|strictly|exclude\w*|18\+|adults? only|over 18|csam)\b",
    re.IGNORECASE,
)


def _pattern(terms: Iterable[str]) -> re.Pattern[str]:
    # 区切りは英数字以外。名前の "Loli-XL" や "shota_v2"、"kids.safetensors" も単語として当てる
    alts = "|".join(re.escape(t).replace(r"\ ", r"[\s_\-]+") for t in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])", re.IGNORECASE)


_STRONG = _pattern(STRONG_TERMS)
_CONTEXT = _pattern(CONTEXT_TERMS)
_SEXUAL = _pattern(SEXUAL_TERMS)
_SENTENCE = re.compile(r"(?<=[.!?。\n])\s*")


def _hits(pattern: re.Pattern[str], text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in pattern.finditer(text)})


def _card_sentences(card_text: str) -> list[str]:
    """説明文を、数えてよい文だけに分ける。

    否定・禁止の語を含む文は除く。"Do not use it to generate:" のように否定の行が「:」で終わるときは、
    続く箇条書き("- Sexual content involving minors" など)も禁止の一覧なので除く。
    """
    out: list[str] = []
    negated_list = False
    for line in (card_text or "").splitlines():
        stripped = line.strip()
        bullet = bool(re.match(r"([-*+•]|\d+[.)])\s", stripped))
        if negated_list and (bullet or not stripped):
            continue
        negated_list = False
        if _NEGATION.search(stripped) and stripped.endswith(":"):
            negated_list = True
            continue
        out += [x for x in _SENTENCE.split(stripped) if x and not _NEGATION.search(x)]
    return out


def blocked_terms(names: Iterable[str], card_text: str = "", *, uncensored: bool) -> list[str]:
    """外す理由になった語の一覧(空なら残してよい)。

    names: ID・名前・タグなど短い文字列。card_text: モデルカードの本文など長い文章。
    uncensored: 無検閲のモデルか(CONTEXT の語はこのときだけ当てる)。

    説明文では、STRONG の語は 1 つで、CONTEXT の語は同じ文に性的な語(SEXUAL_TERMS)があるときだけ数える
    (作例のプロンプト "a tiny boat with a child" のような無害な文で外さないため)。
    """
    short = "\n".join(n for n in names if n)
    found = set(_hits(_STRONG, short))
    if uncensored:
        found |= set(_hits(_CONTEXT, short))
    for sentence in _card_sentences(card_text):
        found |= set(_hits(_STRONG, sentence))
        if uncensored and _SEXUAL.search(sentence):
            found |= set(_hits(_CONTEXT, sentence))
    return sorted(found)


def is_blocked(names: Iterable[str], card_text: str = "", *, uncensored: bool) -> bool:
    return bool(blocked_terms(names, card_text, uncensored=uncensored))
