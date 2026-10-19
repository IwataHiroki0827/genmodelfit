"""一覧から外すモデルの判定(未成年の性的表現を目的とするもの)。

CLAUDE.md の決まり: 無検閲のモデルは印を付けて残すが、未成年の性的表現を目的とするものは除外する。
外したものは一覧に載せず、件数だけを `sources.excluded` に数える(名前も残さない)。

語は 2 種類に分ける(docs/sources.md に同じ表を載せる):

- STRONG(それだけで未成年の性的表現を指す語): どのモデルにも当てる
- CONTEXT(単独では無害だが、性的な文脈で未成年を指す語): 無検閲(uncensored)のモデルの ID・名前・タグに
  あれば外す。説明文(モデルカード)では、同じ文に性的な語(SEXUAL_TERMS)があるときだけ数える

説明文の否定・禁止(成人向けモデルの注意書き "Do not generate sexual content involving minors." など)は、
**否定の語がその語を直接受けているときだけ**数えない: 否定の語(no / not / never / do not …)の後ろ 8 語以内で、
間に句読点(, ; : . など)や「filters / limits / restrictions / censorship」などの区切りの語が無いとき。
「Uncensored, no filters: generates nude teen girls.」の no は filters を受けているだけなので、teen は数える。
否定の行が「:」で終わり、禁止の形(do not / must not / never / prohibited …)のときは、
続く箇条書きも禁止の一覧とみなす。

語は単語の境界で照合する(`minor` は `minority`、`kid` は `kidney`、`teen` は `fifteen` に当たらない)。
日本語のカタカナの語は、前後がカタカナでないときだけ当てる(ロリ は カロリー・ロリポップ に当たらない)。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

STRONG_TERMS = (
    "loli",
    "lolis",
    "lolicon",
    "lolicons",
    "loli con",
    "lolita",
    "lolitas",
    "shota",
    "shotas",
    "shotacon",
    "shotacons",
    "shota con",
    "jailbait",
    "preteen",
    "preteens",
    "pedo",
    "pedophile",
    "pedophilia",
    "paedophile",
    "paedophilia",
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
    "teenage",
    "teenager",
    "teenagers",
    "schoolgirl",
    "schoolgirls",
    "school girl",
    "school girls",
    "schoolboy",
    "schoolboys",
    "school boy",
    "school boys",
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
    "jk",
)

# 日本語(カタカナの語は前後がカタカナでないときだけ。漢字の語はそのまま)
_JA_STRONG = r"(?<![ァ-ヺー])(?:ロリ(?:コン|ータ)?|ショタ(?:コン)?)(?![ァ-ヺー])|児童ポルノ"
_JA_CONTEXT = r"幼女|児童|女子小学生|女子中学生|小学生|中学生"
# JC / JS(女子中学生・女子小学生の略)は大文字のときだけ(js は JavaScript などと紛らわしい)
_ABBR_CONTEXT = r"(?<![A-Za-z0-9])(?:JC|JS)(?![A-Za-z0-9])"

# 説明文で CONTEXT の語と同じ文にあると外す語
SEXUAL_TERMS = (
    "sex",
    "sexual",
    "sexually",
    "sexy",
    "nude",
    "nudes",
    "nudity",
    "naked",
    "porn",
    "porno",
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
    "topless",
    "seductive",
)
_JA_SEXUAL = r"エロ|ヌード|裸|性的|セックス|アダルト|18禁|R-?18"

# 否定・禁止の語(これが語を直接受けているときだけ数えない)
_NEG_BEFORE = re.compile(
    r"(?<![a-z0-9])(no|not|never|nor|without|zero|prohibit\w*|forbid\w*|ban|bans|banned|disallow\w*|"
    r"exclud\w*|refus\w*|avoid\w*|cannot|can't|don't|doesn't|won't|isn't|aren't|mustn't|against)(?![a-z0-9])",
    re.IGNORECASE,
)
# 語の後ろに来る禁止("minors are strictly prohibited"、「児童の…は禁止」)
_NEG_AFTER = re.compile(
    r"^[\s\w'-]{0,40}?\b(is|are)?\s*(strictly\s+)?(prohibited|forbidden|banned|not\s+allowed|not\s+permitted|"
    r"disallowed)\b|^[^。.!?\n]{0,15}?(禁止|不可|厳禁|しないで|ではありません)",
    re.IGNORECASE,
)
# 否定の効き目を切る区切り(句読点と、「何を」否定しているかを示す語)
_SCOPE_BREAK = re.compile(
    r"[,;:.!?()\[\]\n。、!?:]|(?<![a-z0-9])(but|filters?|filtering|limits?|limitations?|restrictions?|"
    r"restricted|censor\w*|safety|guardrails?|boundaries|taboos?|holds|rules|moderation)(?![a-z0-9])",
    re.IGNORECASE,
)
NEG_WINDOW_WORDS = 8
# 「:」で終わる禁止の行(続く箇条書きは禁止の一覧)
_PROHIBITION_HEAD = re.compile(
    r"(?<![a-z0-9])(do not|don't|must not|mustn't|may not|never|should not|cannot|prohibited|forbidden|"
    r"not (be )?(used|allowed|permitted))(?![a-z0-9])|禁止",
    re.IGNORECASE,
)


def _pattern(terms: Iterable[str], extra: str = "", *, flags: int = re.IGNORECASE) -> re.Pattern[str]:
    # 区切りは英数字以外。名前の "Loli-XL" や "shota_v2"、"kids.safetensors" も単語として当てる
    alts = "|".join(re.escape(t).replace(r"\ ", r"[\s_\-]+") for t in sorted(terms, key=len, reverse=True))
    body = rf"(?<![a-z0-9])(?:{alts})(?![a-z0-9])"
    return re.compile(body + (f"|{extra}" if extra else ""), flags)


_STRONG = _pattern(STRONG_TERMS, _JA_STRONG)
_CONTEXT = _pattern(CONTEXT_TERMS, _JA_CONTEXT)
_ABBR = re.compile(_ABBR_CONTEXT)
_SEXUAL = _pattern(SEXUAL_TERMS, _JA_SEXUAL)
_SENTENCE = re.compile(r"(?<=[.!?。\n])\s*")


# 生成の禁止("Do not use this model to generate …" "must not create …")。これは「,」で区切った列挙
# ("illegal content, non-consensual imagery, or sexual content involving minors")全体に効くので、
# 文の終わり(. ; : ! ?)までの 30 語を禁止の範囲とする
_PROHIBIT_GENERATE = re.compile(
    r"(?<![a-z0-9])(do not|don't|must not|mustn't|may not|should not|never|cannot|can't|"
    r"not (be )?(allowed|permitted) to|prohibited from|forbidden to)\s+(\S+\s+){0,5}?"
    r"(use|generate|create|produce|make|depict|output|render)(?![a-z0-9])",
    re.IGNORECASE,
)
PROHIBIT_WINDOW_WORDS = 30
# 犯罪の名前としての言及(安全対策の説明 "filter known child sexual abuse material" など)。語の当たりから除く
_CRIME_NAME = re.compile(
    r"child(ren)?['’]?s?\s+sexual\s+(abuse|exploitation)(\s+(material|imagery|content))?", re.IGNORECASE
)


def _negated(text: str, start: int, end: int) -> bool:
    """text[start:end] の語が、否定・禁止の語に直接受けられているか。"""
    prefix = text[max(0, start - 300) : start]
    last_p = None
    for m in _PROHIBIT_GENERATE.finditer(prefix):
        last_p = m
    if last_p is not None:
        between = prefix[last_p.end() :]
        if not re.search(r"[;:.!?。\n]", between) and len(between.split()) <= PROHIBIT_WINDOW_WORDS:
            return True
    prefix = prefix[-120:]
    last = None
    for m in _NEG_BEFORE.finditer(prefix):
        last = m
    if last is not None:
        between = prefix[last.end() :]
        if not _SCOPE_BREAK.search(between) and len(between.split()) <= NEG_WINDOW_WORDS:
            return True
    return bool(_NEG_AFTER.match(text[end : end + 60]))


def _live_hits(pattern: re.Pattern[str], text: str, *, negation: bool) -> list[str]:
    """否定に受けられていない当たり。"""
    out = []
    text = _CRIME_NAME.sub(lambda m: " " * len(m.group(0)), text)
    for m in pattern.finditer(text):
        if negation and _negated(text, m.start(), m.end()):
            continue
        out.append(m.group(0).lower())
    return out


def _card_sentences(card_text: str) -> list[str]:
    """説明文を文に分ける。禁止の行(「:」で終わる)に続く箇条書きは、禁止の一覧なので除く。"""
    out: list[str] = []
    negated_list = False
    for line in (card_text or "").splitlines():
        stripped = line.strip()
        bullet = bool(re.match(r"([-*+•・]|\d+[.)])\s*", stripped)) and len(stripped) > 1
        if negated_list and (bullet or not stripped):
            continue
        negated_list = False
        if stripped.endswith((":", ":")) and _PROHIBITION_HEAD.search(stripped):
            negated_list = True
            continue
        out += [x for x in _SENTENCE.split(stripped) if x]
    return out


def blocked_terms(names: Iterable[str], card_text: str = "", *, uncensored: bool) -> list[str]:
    """外す理由になった語の一覧(空なら残してよい)。

    names: ID・名前・タグなど短い文字列(否定は見ない)。card_text: モデルカードの本文など長い文章。
    uncensored: 無検閲のモデルか(名前の CONTEXT の語はこのときだけ当てる)。

    説明文では、STRONG の語は否定に受けられていなければ 1 つで、CONTEXT の語は否定に受けられておらず、
    同じ文に(否定に受けられていない)性的な語があるときに数える(作例のプロンプト "a tiny boat with a child"
    のような無害な文で外さないため)。
    """
    short = "\n".join(n for n in names if n)
    found = set(_live_hits(_STRONG, short, negation=False))
    if uncensored:
        found |= set(_live_hits(_CONTEXT, short, negation=False))
        found |= set(_live_hits(_ABBR, short, negation=False))
    for sentence in _card_sentences(card_text):
        found |= set(_live_hits(_STRONG, sentence, negation=True))
        if _live_hits(_SEXUAL, sentence, negation=True):
            found |= set(_live_hits(_CONTEXT, sentence, negation=True))
            found |= set(_live_hits(_ABBR, sentence, negation=True))
    return sorted(found)


def is_blocked(names: Iterable[str], card_text: str = "", *, uncensored: bool) -> bool:
    return bool(blocked_terms(names, card_text, uncensored=uncensored))
