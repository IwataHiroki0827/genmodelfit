# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""人気の順位(#95。決まりは docs/scoring.md「人気」)。並びの鍵だけに使い、Score には入れない。

- 単位はモデル(groups.group_key の仲間)。数は仲間の中の一番大きいもの
- 出どころを 2 つに分けて混ぜない: ComfyUI のテンプレートの使用数(index.json の usage)と、Hugging Face の
  全期間のダウンロード数(downloads_all_time)。仲間にテンプレートがあればテンプレートの数を使う。
  手で足したモデル(manual)も HF のリポジトリがあれば HF の仲間に入る
- 順位 = 同じ種類・同じ出どころの、一覧の全モデル(絞り込みの前)の中での百分位(中央の順位):
  100 × (数が少ないモデルの数 + 自分を含む同じ数のモデルの数 ÷ 2) ÷ 仲間の数
- 数が無い・0 のモデルは 0(仲間の計算には数 0 として入る)。仲間が SMALL_GROUP 件未満なら、数のあるモデルは 50
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from .models import ModelEntry

TEMPLATE = "template"
HUGGING_FACE = "hf"
SMALL_GROUP = 5  # これ未満の仲間では順位の差があてにならないので、数のあるモデルを 50 にする
SMALL_GROUP_RANK = 50.0


def source_count(members: Iterable[ModelEntry]) -> tuple[str | None, int]:
    """モデル(仲間)の (出どころ, 数)。テンプレートがあればテンプレートの使用数の一番大きいもの、
    無ければ HF の全期間のダウンロード数の一番大きいもの。どちらでもなければ (None, 0)。"""
    members = list(members)
    tmpl = [m for m in members if m.template is not None]
    if tmpl:
        return TEMPLATE, max((m.template.usage or 0) for m in tmpl)
    hf = [m for m in members if m.source == "huggingface" or any(f.repo_id for f in m.files)]
    if hf:
        return HUGGING_FACE, max((m.downloads_all_time or 0) for m in hf)
    return None, 0


def percentile_ranks(counts: dict[str, int]) -> dict[str, float]:
    """仲間(鍵 → 数)の中の順位(0〜100)。数が 0 のものは 0。仲間が SMALL_GROUP 件未満なら数のあるものは 50。"""
    n = len(counts)
    out: dict[str, float] = {}
    values = sorted(counts.values())
    for key, v in counts.items():
        if v <= 0:
            out[key] = 0.0
        elif n < SMALL_GROUP:
            out[key] = SMALL_GROUP_RANK
        else:
            less = sum(1 for x in values if x < v)
            same = sum(1 for x in values if x == v)
            out[key] = 100.0 * (less + same / 2) / n
    return out


def compute(models: Iterable[ModelEntry], key_of: dict[str, str]) -> dict[str, float]:
    """一覧の全モデルの人気の順位(group_key → 0〜100)。key_of はエントリの ID → group_key。"""
    members: dict[str, list[ModelEntry]] = defaultdict(list)
    for m in models:
        members[key_of.get(m.id, m.id)].append(m)
    pools: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    ranks: dict[str, float] = {}
    for key, ms in members.items():
        src, count = source_count(ms)
        if src is None:
            ranks[key] = 0.0
            continue
        pools[(ms[0].modality, src)][key] = count
    for pool in pools.values():
        ranks.update(percentile_ranks(pool))
    return ranks


def for_db(db: Any) -> dict[str, float]:
    """一覧(Database)ごとに 1 回だけ計算する(groups.index_for と同じ考え方)。"""
    from .groups import index_for

    cached = getattr(db, "_popularity", None)
    if cached is None or cached[0] != len(db.models):
        cached = (len(db.models), compute(db.models, index_for(db).key_of))
        db._popularity = cached
    return cached[1]


# 毎週の確認(#95。docs/scoring.md「人気」): 軽いモデルほど人気になるかたよりが出ていないか
BIAS_MIN_MODELS = 20  # テンプレートのモデルがこれ以上ある種類だけ確かめる(少ないと相関があてにならない)
BIAS_THRESHOLD = -0.3  # 要る VRAM と使用数の順位の相関(Spearman)がこれを下回ったら知らせる


def _ranks(values: list[float]) -> list[float]:
    """順位(1 から。同じ値は平均の順位)。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman の順位相関(同じ値は平均の順位)。2 件未満・どちらかが全部同じ値なら None。"""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx <= 0 or syy <= 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def vram_bias(db: Any, coefficients: Any = None) -> list[dict[str, Any]]:
    """種類ごとの、テンプレートのモデルの「全部載せの VRAM(既定の設定)」と「使用数」の Spearman の相関。
    [{modality, models, rho, checked, flagged}]。checked はモデルが BIAS_MIN_MODELS 以上の種類、flagged は
    rho が BIAS_THRESHOLD を下回った種類(毎週の PR の本文で知らせる)。VRAM は仲間の中で使用数の一番大きい
    テンプレートのもの。大きさの分からないモデルは入れない。"""
    from .estimate import components, load_coefficients, no_size_input, raw_estimate, settings_for
    from .groups import index_for
    from .models import MODALITIES

    c = coefficients or load_coefficients()
    pairs: dict[str, list[tuple[float, float]]] = {}
    for _key, ms in index_for(db).members.items():
        tmpl = [m for m in ms if m.template is not None]
        if not tmpl:
            continue
        top = max(tmpl, key=lambda m: ((m.template.usage or 0), m.id))
        if components(top).unknown:
            continue
        full, _off, _ram = raw_estimate(top, settings_for(top, None, sized=not no_size_input(top)), c)
        pairs.setdefault(top.modality, []).append((full, float(top.template.usage or 0)))
    out = []
    for mod in MODALITIES:
        ps = pairs.get(mod, [])
        if not ps:
            continue
        rho = spearman([p[0] for p in ps], [p[1] for p in ps])
        checked = len(ps) >= BIAS_MIN_MODELS
        out.append(
            {
                "modality": mod,
                "models": len(ps),
                "rho": None if rho is None else round(rho, 2),
                "checked": checked,
                "flagged": bool(checked and rho is not None and rho < BIAS_THRESHOLD),
            }
        )
    return out
