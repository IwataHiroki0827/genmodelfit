# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""種類(modality)の中の使い方(用途)と、補助ツールの見分け方(Issue #17)。

大きな種類は 6 つ(image / video / music / sfx / tts / 3d)のまま。用途は種類の中の小分けで、
一覧の tasks(t2i / i2i / i2v など)から決める。表は docs/schema.md「Uses」と同じ。
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .i18n import tr
from .models import MODALITIES, ModelEntry

# 種類の表示名(英語, 日本語)
MODALITY_LABEL = {
    "image": ("image", "画像"),
    "video": ("video", "動画"),
    "music": ("music", "音楽"),
    "sfx": ("sound effects", "効果音"),
    "tts": ("text to speech", "読み上げ"),
    "3d": ("3D model", "3D"),
}

# 用途: 名前 → (英語, 日本語, tasks のコード)。並びは表示の順
USES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "gen": (
        "generate from text (3D: from an image)",
        "文章から作る(3D は画像から)",
        ("t2i", "t2v", "t2a", "tts", "t23d", "i23d"),
    ),
    "edit": (
        "edit / transform / control",
        "編集・変換・形の指定",
        ("i2i", "inpaint", "outpaint", "relight", "control", "v2v", "a2a"),
    ),
    "i2v": ("image to video", "画像から動画", ("i2v", "flf2v", "ref2v")),
    "lipsync": ("lip sync / audio to video", "口の動き・音から動画", ("lipsync", "a2v")),
    "motion": ("motion control", "動きの指定", ("motion",)),
    "voice": ("voice conversion", "声の変換", ("vc",)),
    "upscale": ("upscale / restore", "高解像度化", ("upscale",)),
    "tool": (
        "utility tool (depth, pose, mask, …)",
        "補助ツール(深度・ポーズ・切り抜きなど)",
        ("tool", "estimate", "matting", "interp", "segment", "separate"),
    ),
}
# --task で使える別名
USE_ALIASES = {"generate": "gen", "lip": "lipsync"}
# 1 つの項目に用途が複数あるとき、表に出す用途の優先順(補助ツール・高解像度化を「生成」と言わない)
USE_PRIORITY = ("tool", "upscale", "lipsync", "motion", "voice", "edit", "gen", "i2v")

# 「つくる」モデルではない補助ツールの tasks(高解像度化も含む)。
# recommend は既定で外す(fit・search・info には出る)
TOOL_TASKS = frozenset(USES["tool"][2]) | {"upscale"}

# 道具の分類(Image Tools / Video Tools)の題が、深度・線画・ポーズなどの「地図を作る前処理」のもの。
# 公式テンプレートのタグが controlnet / text to image のこともあるので、題で見分ける
PREPROCESS_TITLE = re.compile(
    r"\b(depth|normal|albedo|line ?art|pose|geometry|segmentation)\b[^:]*"
    r"\b(estimation|map|detection|preprocessor)\b|\bsegmentation\b",
    re.IGNORECASE,
)
# 音の分離(ボーカルと伴奏を分けるなど)は生成ではない
SEPARATION_TITLE = re.compile(r"\b(audio|stem|vocals?|source) separation\b", re.IGNORECASE)
# 単眼の形状推定(MoGe: Monocular Geometry Estimation)。メッシュも出すが、写真から形を推すもので生成ではない。
# 分類が 3D Model でも補助ツールにする
ESTIMATOR_TITLE = re.compile(r"^\s*moge\b|\bgeometry estimation\b", re.IGNORECASE)
TOOL_CATEGORIES = frozenset({"Image Tools", "Video Tools"})


def tool_task_from_title(title: str | None, category: str | None) -> str | None:
    """題から決める補助ツールの task(estimate / separate)。当てはまらなければ None。"""
    t = title or ""
    if SEPARATION_TITLE.search(t):
        return "separate"
    if ESTIMATOR_TITLE.search(t):
        return "estimate"
    if category in TOOL_CATEGORIES and PREPROCESS_TITLE.search(t):
        return "estimate"
    return None


def is_tool(m: ModelEntry) -> bool:
    """補助ツールか。tasks がすべて補助の作業か、題が前処理・音の分離のもの
    (古い一覧のタグでも外れるように)。"""
    if m.tasks and all(t in TOOL_TASKS for t in m.tasks):
        return True
    t = m.template
    category = t.category if t else None
    return any(tool_task_from_title(x, category) for x in (t.title if t else None, m.name) if x)


def task_codes(names: Iterable[str]) -> frozenset[str]:
    """--task の名前(用途の名前・別名・tasks のコード)→ tasks のコード。知らない名前は ValueError。"""
    known = {c for _en, _ja, codes in USES.values() for c in codes}
    out: set[str] = set()
    for raw in names:
        n = raw.strip().lower()
        if not n:
            continue
        n = USE_ALIASES.get(n, n)
        if n in USES:
            out |= set(USES[n][2])
        elif n in known:
            out.add(n)
        else:
            names_ok = ", ".join([*USES, *USE_ALIASES, *sorted(known)])
            raise ValueError(
                tr(
                    f"task が正しくありません: {raw!r}(使えるのは {names_ok})",
                    f"invalid task: {raw!r} (allowed: {names_ok})",
                )
            )
    if not out:
        raise ValueError(tr("task が空です", "task is empty"))
    return frozenset(out)


def uses_of(m: ModelEntry) -> list[str]:
    """項目の用途(USES の名前)。補助ツールなら upscale か tool だけ。"""
    if is_tool(m):
        return ["upscale"] if "upscale" in m.tasks else ["tool"]
    out = [u for u, (_en, _ja, codes) in USES.items() if any(t in codes for t in m.tasks)]
    return out or ["gen"]


def main_use(m: ModelEntry, prefer: Iterable[str] | None = None) -> str:
    """表に出す用途 1 つ。prefer(--task のコード)に当たる用途があればそれ。"""
    uses = uses_of(m)
    if prefer:
        codes = set(prefer)
        hit = [u for u in uses if codes & set(USES[u][2]) & set(m.tasks)]
        if hit:
            uses = hit
    return min(uses, key=USE_PRIORITY.index)


def parse_modalities(value: str | Iterable[str] | None) -> tuple[str, ...] | None:
    """-m / modality: 1 つ、カンマ区切り、またはその並び。知らない種類は ValueError。"""
    if value is None:
        return None
    items = value.split(",") if isinstance(value, str) else list(value)
    out = tuple(dict.fromkeys(x.strip().lower() for x in items if x and x.strip()))
    bad = [x for x in out if x not in MODALITIES]
    if bad or not out:
        ok = ", ".join(MODALITIES)
        raise ValueError(
            tr(
                f"modality が正しくありません: {value!r}(使えるのは {ok})",
                f"invalid modality: {value!r} (allowed: {ok})",
            )
        )
    return out
