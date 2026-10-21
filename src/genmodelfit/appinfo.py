# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""アプリが一覧だけで画面を作るための項目(`languages` / `runtime` / `description`。0.5.0、#36)と、
手で足すモデル(`source: "manual"`)。

- 言語の決め方(`apply_app_fields`):
  1) 手の表 `data/overrides.json`(人が一次資料か実機で確かめたもの。あれば全部これ)
  2) HF のカードの `language`(update が取れたときだけ。指示の言語には、3 の表の言語を足す。
     読み上げはカードだけ)
  3) 文字を読む部品の表(`TEXT_ENCODERS`・`FAMILY_PROMPT`)。
  どれでも決まらなければ空にする(当て推量で ja を付けない)。
  根拠は docs/sources.md「言語・動かし方・説明」
- 手で足すモデルは `data/manual.json`(`manual_entries`)。ライセンスは書いた人が一次資料で確かめた値で、
  update が取れたときだけ上流のカードと照らす(`verify_manual_licenses`。食い違えば厳しいほうに寄せて警告)
"""

from __future__ import annotations

import json
import re
from typing import Any

from .db import bundled_text
from .models import ModelEntry

OVERRIDES_FILE = "overrides.json"
MANUAL_FILE = "manual.json"

# 文字を読む部品で「多言語」と数える言語。部品の表は「日本語を含む多言語」と言える根拠のあるものだけで、
# 全言語の一覧ではない(使う側は「ja が入っているか」を見る)。en は全部の部品が持つ
MULTILINGUAL = ["en", "ja", "zh"]
ENGLISH = ["en"]

# ファイル名(小文字)の正規表現 → (結果, 根拠)。結果 "multi" = MULTILINGUAL、"en" = ENGLISH。
# 上から当てる(umt5 を t5 より先に)。出典は docs/sources.md「言語・動かし方・説明」と同じ
TEXT_ENCODERS: tuple[tuple[str, str, str], ...] = (
    # google/umt5-xxl のカードの language に multilingual と ja・zh・en が並ぶ(mC4 の多言語で事前学習)
    (r"umt5", "multi", "https://huggingface.co/google/umt5-xxl"),
    # Qwen3 の LLM(Qwen/Qwen3-4B のカード「Support of 100+ languages and dialects」)
    (r"^qwen_3_", "multi", "https://huggingface.co/Qwen/Qwen3-4B"),
    # Qwen3.5(Qwen/Qwen3.5-9B のカード「Expanded support to 201 languages and dialects」)
    (r"^qwen3[._]5", "multi", "https://huggingface.co/Qwen/Qwen3.5-9B"),
    # Qwen2.5-VL は Qwen2.5 の LLM が元(Qwen/Qwen2.5-7B-Instruct のカード
    # 「over 29 languages, including … Japanese」)。
    # Qwen3-VL(qwen3vl_*)はカードに言語の記述が無いので表に入れない(空)
    (r"^qwen[_-]?2[._]?5[_-]?vl", "multi", "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct"),
    # OpenAI CLIP のカード「not been purposefully trained in or evaluated on any languages other than
    # English」。clip_g(OpenCLIP bigG)も同じ系統の英語の画像説明
    (r"^clip[_-][lg]\b|^clip[_-][lg][_.]", "en", "https://huggingface.co/openai/clip-vit-large-patch14"),
    # T5 v1.1(t5xxl・t5-base)は C4(英語)で事前学習(カードの language: en)
    (r"^t5[_-]?xxl|^t5[_-]base|^t5-v1", "en", "https://huggingface.co/google/t5-v1_1-xxl"),
)
# 本体に文字を読む部品が入っている系統(チェックポイントに CLIP が入った SD 系)
FAMILY_PROMPT: dict[str, tuple[str, str]] = {
    "sd": ("en", "SD 1.x / 2.x / XL はチェックポイントの中の CLIP(英語の画像説明で学習)で指示を読む"),
}
# 指示(文章)を読む作業。これが無いモデル(画像から 3D など)は prompt を決めない
PROMPT_TASKS = frozenset(
    {"t2i", "i2i", "t2v", "i2v", "v2v", "flf2v", "ref2v", "a2v", "t2a", "inpaint", "outpaint", "control"}
)

_ISO3 = {
    "eng": "en",
    "jpn": "ja",
    "zho": "zh",
    "chi": "zh",
    "kor": "ko",
    "deu": "de",
    "ger": "de",
    "fra": "fr",
    "fre": "fr",
    "spa": "es",
    "por": "pt",
    "ita": "it",
    "rus": "ru",
}


def normalize_languages(value: Any) -> list[str]:
    """HF のカードの language(文字列か並び)を ISO 639-1 の 2 文字にそろえる。multilingual・und・3 文字の
    知らないものは捨てる。順序は保ち、重ねない。"""
    items = [value] if isinstance(value, str) else (value if isinstance(value, list) else [])
    out: list[str] = []
    for x in items:
        if not isinstance(x, str):
            continue
        code = x.strip().lower().replace("_", "-").split("-")[0]
        code = _ISO3.get(code, code)
        if re.fullmatch(r"[a-z]{2}", code) and code not in out:
            out.append(code)
    return out


def card_languages(info: Any) -> list[str]:
    """RepoInfo(HF のカード)の language。無ければ空。"""
    card = getattr(info, "card", None) or {}
    return normalize_languages(card.get("language"))


def uses_prompt(e: ModelEntry) -> bool:
    return e.modality != "tts" and bool(PROMPT_TASKS & set(e.tasks))


def prompt_from_tables(e: ModelEntry) -> list[str]:
    """文字を読む部品の表から、指示の言語を決める。多言語の部品が 1 つでもあればそれ、すべて英語の部品なら
    en、表に無い部品が混じれば決めない(空)。部品が無ければ系統の表。"""
    if not uses_prompt(e):
        return []
    kinds: list[str | None] = []
    for f in e.files:
        if f.role != "text_encoder" and f.directory != "text_encoders":
            continue
        name = f.name.rsplit("/", 1)[-1].lower()
        hit = next((k for pat, k, _ in TEXT_ENCODERS if re.search(pat, name)), None)
        kinds.append(hit)
    if "multi" in kinds:
        return list(MULTILINGUAL)
    if kinds:
        return list(ENGLISH) if all(k == "en" for k in kinds) else []
    fam = FAMILY_PROMPT.get(e.comfy_family or "")
    return list(ENGLISH) if fam and fam[0] == "en" else []


def runtime_of(e: ModelEntry) -> str | None:
    """動かす仕組み。公式テンプレート由来は comfyui。HF 由来は、ComfyUI の系統に当たる画像・動画・音楽・
    効果音・3D だけ comfyui(読み上げは仕組みがモデルごとに違うので決めない)。"""
    if e.source == "comfyui-template":
        return "comfyui"
    if e.source == "huggingface" and e.modality != "tts" and e.comfy_family:
        return "comfyui"
    return None


_cache: dict[str, Any] = {}


def _bundled_json(name: str) -> Any:
    if name not in _cache:
        _cache[name] = json.loads(bundled_text(name))
    return _cache[name]


def overrides() -> dict[str, dict[str, Any]]:
    """手の表(id → languages / runtime / description)。"""
    return dict(_bundled_json(OVERRIDES_FILE).get("models", {}))


def apply_app_fields(
    entries: list[ModelEntry],
    client: Any = None,
    *,
    table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, int]:
    """languages / runtime / description を決めて入れる(update と、通信しない書き直しで使う)。

    client: HfClient。あれば HF のカードの language を使う(通信はしない。取得済みのものだけ見る)。
    通信しない書き直しでは、前の一覧の languages を残し、表と手の表だけを当て直す。
    返す値は集計(sources["app_fields"])。"""
    table = overrides() if table is None else table
    stats = {"languages_prompt": 0, "languages_speech": 0, "runtime": 0, "description": 0, "from_card": 0}
    for e in entries:
        langs = {k: list(v) for k, v in (e.languages or {}).items()}  # [] は「無い」と手で決めたもの
        table_langs = prompt_from_tables(e)
        from_card = _card_for_entry(e, client) if client is not None and e.source != "manual" else []
        if from_card and e.modality == "tts":
            langs["speech"] = from_card
            stats["from_card"] += 1
        elif from_card and uses_prompt(e):
            # カードの言語に、文字を読む部品の表の言語を足す(カードは学習・説明の言語で、部品が読める言語より
            # 狭いことが多い。Z-Image のカードは en だけだが、Qwen3 の部品で ja の指示も通じる)
            langs["prompt"] = from_card + [x for x in table_langs if x not in from_card]
            stats["from_card"] += 1
        elif "prompt" not in langs and table_langs:
            langs["prompt"] = table_langs
        row = table.get(e.id) or {}
        for k, v in (row.get("languages") or {}).items():
            langs[k] = list(v)
        e.languages = {k: langs[k] for k in ("prompt", "speech") if k in langs}
        if row.get("runtime"):
            e.runtime = row["runtime"]
        elif e.source != "manual" or not e.runtime:
            e.runtime = runtime_of(e) or e.runtime
        if row.get("description"):
            e.description = dict(row["description"])
        stats["languages_prompt"] += bool(e.languages.get("prompt"))
        stats["languages_speech"] += bool(e.languages.get("speech"))
        stats["runtime"] += e.runtime is not None
        stats["description"] += bool(e.description)
    return stats


def _card_for_entry(e: ModelEntry, client: Any) -> list[str]:
    """本体のファイルの置き場所とその元(license_from)のカードの language。最初に見つかったもの。"""
    mains = [f for f in e.files if f.role == "main"] or e.files[:1]
    repos: list[str] = []
    for f in mains:
        for r in (f.repo_id, f.license_from):
            if r and r not in repos:
                repos.append(r)
    if e.source == "huggingface" and e.id.startswith("hf:") and e.id[3:] not in repos:
        repos.insert(0, e.id[3:])
    for r in repos:
        info = client.cached_repo(r)
        langs = card_languages(info) if info else []
        if langs:
            return langs
    return []


# --- 手で足すモデル -----------------------------------------------------------------------------------


def manual_entries(data: dict[str, Any] | None = None) -> list[ModelEntry]:
    """data/manual.json から ModelEntry を作る。ライセンス・大きさ・系統などは update の関数でそろえる。"""
    from .families import assign
    from .update import set_completeness, set_entry_license, set_precision

    data = _bundled_json(MANUAL_FILE) if data is None else data
    out: list[ModelEntry] = []
    for raw in data.get("models", []):
        e = ModelEntry.from_dict({**raw, "source": "manual"})
        for f in e.files:
            if f.repo_id is None and f.url and "huggingface.co/" in f.url:
                parts = f.url.split("/")
                f.repo_id = "/".join(parts[3:5])
        set_entry_license(e)
        _prefer_main_license_url(e)
        set_completeness(e)
        set_precision(e)
        assign(e)
        out.append(e)
    return out


def _prefer_main_license_url(e: ModelEntry) -> None:
    """一式のライセンスが本体のファイルと同じなら、原文の場所も本体のもの(同じ段階のファイルが複数あるとき、
    set_entry_license はファイル名の順で選ぶので、部品の URL になってしまう)。"""
    main = next((f for f in e.files if f.role == "main"), None)
    if main and main.license_url and (main.license, main.license_name) == (e.license, e.license_name):
        e.license_url = main.license_url


_MEASURED_FILE_FIELDS = (
    "head_status",
    "head_checked_at",
    "header_precision",
    "header_params",
    "repo_precision",
    "repo_precision_source",
)


def merge_manual(entries: list[ModelEntry], manual: list[ModelEntry] | None = None) -> list[ModelEntry]:
    """手のモデルを一覧に入れる(同じ id は手のものに置き換える)。並びは update と同じ。
    置き換える前の同じ URL のファイルに、通信で確かめた値(HEAD・見出しの精度)があれば引き継ぐ
    (通信しない書き直しで消さないため)。"""
    from .update import set_precision

    manual = manual_entries() if manual is None else manual
    old = {e.id: e for e in entries}
    for m in manual:
        prev = old.get(m.id)
        if prev is None:
            continue
        by_url = {f.url: f for f in prev.files if f.url}
        for f in m.files:
            pf = by_url.get(f.url)
            if pf is None or pf.sha256 != f.sha256:
                continue
            for k in _MEASURED_FILE_FIELDS:
                if getattr(f, k) is None and getattr(pf, k) is not None:
                    setattr(f, k, getattr(pf, k))
        set_precision(m)
    ids = {m.id for m in manual}
    merged = [e for e in entries if e.id not in ids] + manual
    merged.sort(key=lambda e: (e.source != "comfyui-template", e.modality, e.id))
    return merged


def verify_manual_licenses(entries: list[ModelEntry], client: Any) -> list[str]:
    """手のモデルのファイルのライセンスを、上流のカード(license_from)と照らす。読めて食い違えば、厳しいほうを
    使って警告を返す。カードを読めなければ(非公開・401・通信の失敗)書いた値のまま、確かめられなかった旨の警告。"""
    from .hf import normalize_license, strictest
    from .update import set_entry_license

    warnings: list[str] = []
    for e in entries:
        if e.source != "manual":
            continue
        for f in e.files:
            if not f.license_from:
                continue
            info = client.repo(f.license_from)
            if info is None:
                warnings.append(
                    f"手のモデル {e.id}: {f.license_from} のカードを読めず、ライセンスを照合できません"
                )
                continue
            card = (normalize_license(info.license), info.license_name)
            mine = (f.license, f.license_name)
            if card[0] is not None and card != mine:
                worse = strictest([card, mine])
                f.license, f.license_name = worse
                f.license_evidence.append(
                    f"manual_mismatch: 書いた値 {mine} とカード {card} が違うので厳しいほうにした"
                )
                warnings.append(
                    f"手のモデル {e.id}: {f.name} のライセンスがカードと違います"
                    f"(書いた値 {mine}、カード {card})"
                )
        set_entry_license(e)
        _prefer_main_license_url(e)
    return warnings
