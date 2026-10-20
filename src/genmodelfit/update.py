"""`genmodelfit update`: モデルの一覧を上流から作り直す(llmfit の update に当たる)。

データ源(詳しくは docs/sources.md):
  1. ComfyUI 公式テンプレート(MIT)の index.json と、テンプレートごとの JSON の `properties.models`
  2. Hugging Face Hub API(ライセンス・gated・大きさ・sha256。再配布は base_model をたどる)
  3. Hugging Face の検索(無検閲のモデル、テンプレートに無い読み上げ・効果音のモデル)

上流の形が変わったら黙って空にせず UpstreamFormatError で止める。ネットにつながらないときは前のキャッシュを
そのまま残して失敗する(recommend はキャッシュか同梱のデータで動き続ける)。
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from . import __version__, safety
from .db import Database, bundled_text, cached_db_path, parse_db
from .fetch import RATE_LIMIT_BUDGET, DirFetcher, Fetcher, FetchError, HttpFetcher, NetworkError
from .hf import (
    UNVERIFIED,
    HfClient,
    RepoInfo,
    Resolved,
    choose_bases,
    families_for,
    guess_params_b,
    guess_precision,
    is_repackager,
    license_tier,
    parse_hf_url,
    search_url,
    select_weight_files,
    upstream_candidates,
)
from .models import ModelEntry, ModelFile, Template

RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates"
INDEX_URL = f"{RAW}/index.json"
INDEX_PAGE = "https://github.com/Comfy-Org/workflow_templates/blob/main/templates/index.json"

# テンプレートの失敗がこの割合を超えたら、上流の形が変わったとみなして止める
MAX_FAIL_RATIO = 0.30
# ライセンスの決め方の版。前の一覧から値を引き継ぐ(一部が取れなかったとき)のは、同じ版で作った一覧からだけ
# (古い決め方の寛容すぎる値を持ち込まないため)。決め方を変えたら上げる
LICENSE_POLICY = 2
# 一式がそろっていないとみなす割合(ファイルの合計 / index の size)
COMPLETE_RATIO = 0.5
# sources.counts.license_tier の名前(hf.license_tier の段階)
_TIER_NAMES = {
    6: "unknown",
    5: "non_commercial",
    4: "other",
    3: "use_restricted",
    2: "attribution",
    1: "permissive",
}

# --- テンプレートの分類 ---------------------------------------------------------------------------

CATEGORY_MODALITY = {
    "Image": "image",
    "Image Tools": "image",
    "Video": "video",
    "Video Tools": "video",
    "3D Model": "3d",
    "Audio": "audio",  # 音楽・効果音・読み上げに分ける(audio_modality)
}
SKIP_CATEGORIES = {"LLM", "Node Basics", "Vector"}

# 名前・タグから決められないものだけ手で決める(理由は docs/sources.md)
MODALITY_OVERRIDES = {
    # Stable Audio Open 1.0。モデルカードの用途が効果音・環境音(field recordings)中心
    "audio_stable_audio_example": "sfx",
}

# テンプレートのタグ → tasks(docs/sources.md に同じ表)
TAG_TASKS = {
    "text to image": "t2i",
    "image edit": "i2i",
    "inpainting": "inpaint",
    "outpainting": "outpaint",
    "controlnet": "control",
    "image upscale": "upscale",
    "remove background": "matting",
    "relight": "relight",
    "style reference": "i2i",
    "layer decompose": "i2i",
    "depth estimation": "estimate",
    "normal estimation": "estimate",
    "albedo estimation": "estimate",
    "element segmentation": "estimate",
    "image to video": "i2v",
    "text to video": "t2v",
    "video edit": "v2v",
    "video to video": "v2v",
    "flf2v": "flf2v",
    "reference to video": "ref2v",
    "audio to video": "a2v",
    "lip sync": "lipsync",
    "motion control": "motion",
    "character replacement": "v2v",
    "frame interpolation": "interp",
    "video upscale": "upscale",
    "image to 3d": "i23d",
    "text to 3d": "t23d",
    "text to audio": "t2a",
    "text to music": "t2a",
    "music cover": "a2a",
    "audio editing": "a2a",
    "voice cloning": "vc",
    "text to speech": "tts",
}

# HF の pipeline_tag → (modality, task)
PIPELINES = {
    "text-to-image": ("image", "t2i"),
    "image-to-image": ("image", "i2i"),
    "text-to-video": ("video", "t2v"),
    "image-to-video": ("video", "i2v"),
    "text-to-speech": ("tts", "tts"),
    "text-to-audio": ("audio", "t2a"),
    "image-to-3d": ("3d", "i23d"),
    "text-to-3d": ("3d", "t23d"),
}

DIR_ROLE = {
    "diffusion_models": "main",
    "unet": "main",
    "checkpoints": "main",
    "text_encoders": "text_encoder",
    "clip": "text_encoder",
    "vae": "vae",
    "clip_vision": "clip_vision",
    "loras": "lora",
}

_TTS = re.compile(r"\b(tts|text to speech|speech|voice clon\w*)\b", re.IGNORECASE)
_SFX = re.compile(r"\b(sound effects?|sfx|foley)\b", re.IGNORECASE)
_MUSIC = re.compile(r"\b(music|song|songs|instrumentals?)\b", re.IGNORECASE)
_UNCENSORED = re.compile(r"(?<![a-z0-9])(uncensored|nsfw|abliterated)(?![a-z0-9])", re.IGNORECASE)
_NFAA = "not-for-all-audiences"


class UpdateError(RuntimeError):
    pass


class UpstreamFormatError(UpdateError):
    """上流の形が想定と違う(黙って空の一覧を作らないため)。"""


def validate_index(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, list) or not data:
        raise UpstreamFormatError("index.json の形が変わりました: 一覧(list)ではありません")
    for i, cat in enumerate(data):
        if not isinstance(cat, dict) or not isinstance(cat.get("templates"), list):
            raise UpstreamFormatError(
                f"index.json の形が変わりました: {i} 番目の分類に templates がありません"
            )
        for j, t in enumerate(cat["templates"]):
            if not isinstance(t, dict) or not isinstance(t.get("name"), str) or not t["name"]:
                raise UpstreamFormatError(
                    f"index.json の形が変わりました: 「{cat.get('title')}」の {j} 番目に name がありません"
                )
    return data


def skip_reason(category: str | None, t: dict[str, Any]) -> str | None:
    """ローカルで動くテンプレートでなければ理由を返す。

    openSource が無いものはローカル扱い(False だけを外す)。
    """
    if t["name"].startswith("api_"):
        return "api"
    if t.get("openSource") is False:
        return "not_open_source"
    if not t.get("size"):
        return "no_size"
    if category in SKIP_CATEGORIES:
        return "category"
    return None


def audio_modality(t: dict[str, Any]) -> str:
    short = " ".join([t.get("name", "").replace("_", " "), t.get("title") or "", *(t.get("tags") or [])])
    if _TTS.search(short):
        return "tts"
    text = short + " " + (t.get("description") or "")
    if _SFX.search(text) and not _MUSIC.search(text):
        return "sfx"
    return "music"


def template_modality(category: str | None, t: dict[str, Any]) -> str | None:
    if t["name"] in MODALITY_OVERRIDES:
        return MODALITY_OVERRIDES[t["name"]]
    m = CATEGORY_MODALITY.get(category or "")
    if m == "audio":
        return audio_modality(t)
    if m:
        return m
    # Character & Fashion など: 出力の種類 → タグの順に見る
    outs = {o.get("mediaType") for o in (t.get("io") or {}).get("outputs", []) if isinstance(o, dict)}
    for kind, mod in (("video", "video"), ("audio", "music"), ("3d", "3d"), ("image", "image")):
        if kind in outs:
            return mod
    tags = [x.lower() for x in t.get("tags") or []]
    if any(x in ("image to 3d", "text to 3d") for x in tags):
        return "3d"
    if any("video" in x for x in tags):
        return "video"
    return "image"


# 道具の分類(Image Tools / Video Tools)で、当てはまるタグが無いときの task(生成ではない処理)
TOOL_CATEGORIES = {"Image Tools", "Video Tools"}


def template_tasks(t: dict[str, Any], modality: str, category: str | None = None) -> list[str]:
    out: list[str] = []
    for tag in t.get("tags") or []:
        task = TAG_TASKS.get(str(tag).lower())
        if task and task not in out:
            out.append(task)
    if not out:
        if category in TOOL_CATEGORIES:
            return ["tool"]  # Depth・Pose Map など表に無いタグの道具を、t2i / t2v と言わない
        out = [{"image": "t2i", "video": "t2v", "3d": "i23d", "tts": "tts"}.get(modality, "t2a")]
    return out


def workflow_models(wf: Any) -> list[dict[str, Any]]:
    """ワークフローの全ノード(サブグラフの中も)の properties.models を、URL で重複を除いて返す。"""
    if not isinstance(wf, dict) or not isinstance(wf.get("nodes"), list):
        raise UpstreamFormatError("テンプレートの JSON に nodes がありません")
    nodes = list(wf["nodes"])
    defs = wf.get("definitions")
    if isinstance(defs, dict):
        for sg in defs.get("subgraphs") or []:
            if isinstance(sg, dict) and isinstance(sg.get("nodes"), list):
                nodes += sg["nodes"]
    found = list(wf.get("models") or [])  # 古い形: 一番上の models
    for n in nodes:
        props = n.get("properties") if isinstance(n, dict) else None
        if isinstance(props, dict) and isinstance(props.get("models"), list):
            found += props["models"]
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in found:
        if not isinstance(m, dict) or not m.get("name"):
            continue
        key = m.get("url") or m["name"]
        if key in seen:
            continue
        seen.add(key)
        out.append(m)
    return out


def model_file(m: dict[str, Any]) -> ModelFile:
    directory = m.get("directory")
    h = m.get("hash")
    sha = h.lower() if isinstance(h, str) and str(m.get("hash_type", "")).lower() == "sha256" else None
    parsed = parse_hf_url(m.get("url"))
    return ModelFile(
        name=m["name"],
        url=m.get("url"),
        sha256=sha,
        directory=directory,
        repo_id=parsed[0] if parsed else None,
        role=DIR_ROLE.get(directory or "", "other"),
    )


def template_entry(
    category: str, t: dict[str, Any], modality: str, models: list[dict[str, Any]]
) -> ModelEntry:
    files = [model_file(m) for m in models]
    main = next((f for f in files if f.role == "main"), files[0] if files else None)
    return ModelEntry(
        id=t["name"],
        name=t.get("title") or t["name"],
        modality=modality,
        tasks=template_tasks(t, modality, category),
        source="comfyui-template",
        files=files,
        template=Template(
            name=t["name"],
            title=t.get("title"),
            category=category,
            min_comfyui_version=t.get("minComfyUIVersion"),
            open_source=t.get("openSource"),
            requires_custom_nodes=list(t.get("requiresCustomNodes") or []),
            usage=t.get("usage"),
            date=t.get("date"),
            size_bytes=t.get("size") or None,
        ),
        params_b=guess_params_b(main.name if main else None, t.get("title"), t["name"]),
        precision=guess_precision(main.name if main else None),
        discovered_from=[INDEX_PAGE],
        updated=t.get("date"),
    )


# --- HF でファイルを埋める --------------------------------------------------------------------------


def _hf_keys(entries: list[ModelEntry]) -> set[tuple[str, str]]:
    keys = set()
    for e in entries:
        for f in e.files:
            p = parse_hf_url(f.url)
            if p:
                keys.add((p[0], p[1]))
    return keys


def enrich_files(entries: list[ModelEntry], client: HfClient, fetch: Fetcher, *, head: bool) -> None:
    """ファイルに大きさ・sha256・ライセンス・gated を入れ、一式のライセンスと一式のそろい具合を決める。"""
    keys = _hf_keys(entries)
    client.prefetch(keys)
    # ライセンスの道(再配布の本家と、base_model を MAX_HOPS 段まで)も先にまとめて取る
    chain: set[str] = set()
    for e in entries:
        for f in e.files:
            p = parse_hf_url(f.url)
            info = client.repo(p[0], p[1]) if p else None
            if info is None:
                continue
            if is_repackager(info.repo_id):
                chain |= set(choose_bases(upstream_candidates(info), p[2]))
            else:
                chain |= set(info.base_models)
    client.prefetch_chains(chain)

    missing: list[ModelFile] = []
    for e in entries:
        for f in e.files:
            p = parse_hf_url(f.url)
            if not p:
                host = urlsplit(f.url).netloc if f.url else "URL なし"
                set_file_license(
                    f,
                    Resolved(None, path=[], evidence=[f"{UNVERIFIED}: Hugging Face 以外のファイル({host})"]),
                )
                continue
            info = client.repo(p[0], p[1])
            if info is None:
                fail = client.failed(p[0])
                why = f"{p[0]} を読めません" + (f"(HTTP {fail.status})" if fail and fail.status else "")
                set_file_license(
                    f, Resolved(None, source=p[0], path=[p[0]], evidence=[f"{UNVERIFIED}: {why}"])
                )
                continue
            size, sha = info.files.get(p[2], (None, None))
            f.size_bytes = size or f.size_bytes
            f.sha256 = f.sha256 or sha
            f.gated = info.gated
            apply_license(f, info, client, p[2])
            if f.size_bytes is None:
                missing.append(f)
    if head and missing:
        # API の一覧に無いファイル(別の版など)だけ HEAD で大きさを見る。HF 以外(Civitai など)には送らない
        def _head(f: ModelFile) -> None:
            with contextlib.suppress(FetchError):
                f.size_bytes = fetch.head_size(f.url)

        with ThreadPoolExecutor(client.workers) as ex:
            list(ex.map(_head, missing))
    for e in entries:
        set_entry_license(e)
        set_completeness(e)


def set_file_license(f: ModelFile, res: Resolved) -> None:
    f.license, f.license_name = res.license, res.license_name
    f.license_from = res.source
    f.license_url = res.url
    f.license_path = list(res.path)
    f.license_evidence = list(res.evidence)


def apply_license(
    f: ModelFile, info: RepoInfo, client: HfClient, path: str, *, discovered_uncensored: bool = False
) -> Resolved:
    res = client.resolve_file(info, path, discovered_uncensored=discovered_uncensored)
    set_file_license(f, res)
    f.license_url = res.url
    if res.source and res.source != info.repo_id:
        f.repo_license = info.license if info.license != "other" else (info.license_name or "other")
    return res


def set_entry_license(e: ModelEntry) -> None:
    """一式のライセンスは、すべてのファイル(LoRA・アップスケーラ・clip_vision なども)の中で一番厳しいもの。
    ファイルが無ければ不明。"""
    if not e.files:
        e.license, e.license_name, e.license_url = None, None, None
        e.license_evidence = [f"{UNVERIFIED}: 使うファイルが分かりません"]
        return
    worst = max(e.files, key=lambda f: (license_tier(f.license, f.license_name), f.license or "", f.name))
    e.license, e.license_name = worst.license, worst.license_name
    e.license_url = worst.license_url if worst.license else None
    ev: list[str] = []
    for f in e.files:
        for x in f.license_evidence:
            if x.startswith(UNVERIFIED) and x not in ev:
                ev.append(f"{x}({f.name.rsplit('/', 1)[-1]})")
    e.license_evidence = ev[:10]
    mains = [f for f in e.files if f.role == "main" and f.license_from]
    if mains and e.base_model is None and mains[0].repo_id and mains[0].license_from != mains[0].repo_id:
        e.base_model = mains[0].license_from


def set_completeness(e: ModelEntry) -> None:
    """files_complete と size_floor_bytes(docs/schema.md)。

    False: 本体(role main。diffusion_models / unet / checkpoints)が無い、ファイルの合計が index の size の
    半分未満、または本体が GGUF だけでテキストエンコーダ・VAE が無い(HF のリポジトリ)。
    False のとき、テンプレートの index の size を size_floor_bytes に入れる(見積もりが小さく見ないように)。
    """
    reasons = []
    if not any(f.role == "main" for f in e.files):
        reasons.append("no_main")
    total = sum(f.size_bytes or 0 for f in e.files)
    index_size = e.template.size_bytes if e.template else None
    if index_size and total < index_size * COMPLETE_RATIO:
        reasons.append("small_vs_index")
    mains = [f for f in e.files if f.role == "main"]
    if (
        e.source == "huggingface"
        and mains
        and all(f.name.lower().endswith(".gguf") for f in mains)
        and not any(f.role in ("text_encoder", "vae") for f in e.files)
    ):
        reasons.append("gguf_without_encoder")
    e.files_complete = not reasons
    e.size_floor_bytes = index_size if reasons and index_size else None


def mark_template_content(e: ModelEntry, t: dict[str, Any], client: HfClient) -> None:
    for word in sorted(
        {m.group(1).lower() for m in _UNCENSORED.finditer(" ".join([e.id, e.name, *t.get("tags", [])]))}
    ):
        e.content_evidence.append(f"name:{word}")
    for f in e.files:
        p = parse_hf_url(f.url)
        info = client.repo(p[0], p[1]) if p else None
        if info and _NFAA in info.tags:
            ev = f"hf-tag:{_NFAA} ({info.repo_id})"
            if ev not in e.content_evidence:
                e.content_evidence.append(ev)
    if e.content_evidence:
        e.content = "uncensored"


# --- HF の検索から足す ----------------------------------------------------------------------------

# 無検閲: pipeline_tag ごとに「not-for-all-audiences のタグ」と「名前の語」で探す
UNCENSORED_PIPELINES = tuple(PIPELINES)
UNCENSORED_SEARCH_WORDS = ("uncensored", "nsfw", "abliterated")
# テンプレートで足りない種類を HF の人気順で補う(読み上げはローカルのテンプレートが無い。効果音は 1 件だけ)
SFW_QUERIES = (
    ("tts", {"pipeline_tag": "text-to-speech"}),
    ("sfx", {"pipeline_tag": "text-to-audio", "search": "sfx"}),
    ("sfx", {"pipeline_tag": "text-to-audio", "search": "foley"}),
    ("sfx", {"pipeline_tag": "text-to-audio", "search": "sound"}),
    ("sfx", {"pipeline_tag": "text-to-audio", "search": "audioldm"}),
    ("sfx", {"pipeline_tag": "text-to-audio", "search": "stable-audio-open"}),
)
CAP_UNCENSORED = 30  # 種類ごとの上限(一覧を大きくしすぎない)
CAP_SFW = {"tts": 20, "sfx": 10}
MIN_DOWNLOADS = 100
MIN_MAIN_BYTES = 30_000_000  # 本体がこれより小さいものは、差分(LoRA など)か中身の無いリポジトリとみなす
MAX_PER_OWNER = 2
_ADAPTER = re.compile(r"(?<![a-z])(lora|loras|lycoris|locon|adapter|ip-?adapter|embedding)s?(?![a-z])", re.I)
_COMPONENT = re.compile(
    r"text[-_ ]?encoder|(?<![a-z])(vl|clip|vae|captioner|tagger|llm|gptq|awq|mlx)(?![a-z])"
    r"|qwen\d*(\.\d+)?[-_]?vl",
    re.IGNORECASE,
)


@dataclass
class Candidate:
    repo_id: str
    modality: str
    task: str
    downloads: int
    likes: int
    tags: list[str]
    uncensored: bool
    evidence: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


def hf_audio_modality(repo_id: str, tags: list[str]) -> str:
    text = repo_id.replace("-", " ").replace("_", " ") + " " + " ".join(tags)
    if _TTS.search(text):
        return "tts"
    if _SFX.search(text):
        return "sfx"
    if _MUSIC.search(text):
        return "music"
    if re.search(r"mmaudio|audioldm|stable.audio.open", text, re.I):
        return "sfx"  # 効果音が主な用途のモデル(docs/sources.md)
    return "music"


def _candidate(item: dict[str, Any], url: str, modality_hint: str | None) -> Candidate | None:
    pipe = item.get("pipeline_tag")
    if pipe not in PIPELINES:
        return None
    modality, task = PIPELINES[pipe]
    tags = [t for t in item.get("tags") or [] if isinstance(t, str)]
    if modality == "audio":
        modality = hf_audio_modality(item["id"], tags)
    if modality_hint and modality != modality_hint:
        return None
    ev = []
    if _NFAA in tags:
        ev.append(f"hf-tag:{_NFAA}")
    ev += [f"name:{w}" for w in sorted({m.group(1).lower() for m in _UNCENSORED.finditer(item["id"])})]
    return Candidate(
        repo_id=item["id"],
        modality=modality,
        task=task,
        downloads=int(item.get("downloads") or 0),
        likes=int(item.get("likes") or 0),
        tags=tags,
        uncensored=bool(ev),
        evidence=ev,
        queries=[url],
    )


def discover_hf(
    client: HfClient, *, progress: Callable[[str], None], stats: dict[str, Any]
) -> list[ModelEntry]:
    queries: list[tuple[str, str | None]] = []  # (url, modality_hint)
    for pipe in UNCENSORED_PIPELINES:
        queries.append((search_url(filter=_NFAA, pipeline_tag=pipe), None))
        for w in UNCENSORED_SEARCH_WORDS:
            queries.append((search_url(search=w, pipeline_tag=pipe), None))
    for mod, params in SFW_QUERIES:
        queries.append((search_url(**params), mod))

    def _run(q: tuple[str, str | None]) -> tuple[str, str | None, list[dict[str, Any]] | str]:
        try:
            return q[0], q[1], client.search(q[0])
        except FetchError as e:
            return q[0], q[1], str(e)

    cands: dict[str, Candidate] = {}
    warnings: list[str] = []
    network_fail = 0
    for url, hint, res in client.map(_run, queries):
        if isinstance(res, str):
            warnings.append(f"HF の検索に失敗: {res}")
            network_fail += "つながりません" in res
            continue
        for item in res:
            c = _candidate(item, url, hint)
            if c is None:
                continue
            if hint is None and not c.uncensored:
                continue  # 無検閲の検索で拾ったが根拠が無いもの
            old = cands.get(c.repo_id)
            if old:
                old.queries.append(url)
                old.evidence = sorted(set(old.evidence) | set(c.evidence))
                old.uncensored = old.uncensored or c.uncensored
            else:
                cands[c.repo_id] = c
    if network_fail and network_fail == len(queries):
        raise NetworkError(queries[0][0], "Hugging Face の検索につながりません")

    skipped: Counter[str] = Counter()
    excluded = 0
    pool: list[Candidate] = []
    for c in sorted(cands.values(), key=lambda c: (-c.downloads, c.repo_id)):
        if safety.is_blocked([c.repo_id, *c.tags], uncensored=c.uncensored):
            excluded += 1
            continue
        if c.downloads < MIN_DOWNLOADS:
            skipped["low_downloads"] += 1
        elif _ADAPTER.search(c.repo_id) or any(t.lower() in ("lora", "peft") for t in c.tags):
            skipped["adapter"] += 1
        elif _COMPONENT.search(c.repo_id):
            skipped["component"] += 1
        else:
            pool.append(c)

    # 写しの重複(同じリポジトリ名)は一番使われているものだけ。持ち主ごとに種類あたり 2 件まで
    seen_names: set[str] = set()
    per_owner: Counter[tuple[str, str, bool]] = Counter()
    picked: dict[tuple[str, bool], list[Candidate]] = {}
    for c in pool:
        base = c.repo_id.split("/", 1)[1].lower()
        owner = c.repo_id.split("/", 1)[0].lower()
        if base in seen_names:
            skipped["duplicate"] += 1
            continue
        if per_owner[(owner, c.modality, c.uncensored)] >= MAX_PER_OWNER:
            skipped["owner_cap"] += 1
            continue
        seen_names.add(base)
        per_owner[(owner, c.modality, c.uncensored)] += 1
        picked.setdefault((c.modality, c.uncensored), []).append(c)

    def cap(key: tuple[str, bool]) -> int:
        return CAP_UNCENSORED if key[1] else CAP_SFW.get(key[0], 0)

    shortlist = [c for key, cs in picked.items() for c in cs[: cap(key) * 2]]
    progress(f"Hugging Face: 候補 {len(shortlist)} 件を確かめています")
    client.prefetch((c.repo_id, "main") for c in shortlist)
    client.prefetch_chains(
        b for c in shortlist for b in ((client.repo(c.repo_id) or RepoInfo(repo_id="")).base_models)
    )
    client.map(client.readme, [c.repo_id for c in shortlist])

    out: list[ModelEntry] = []
    count: Counter[tuple[str, bool]] = Counter()
    for c in shortlist:
        key = (c.modality, c.uncensored)
        if count[key] >= cap(key):
            skipped["over_cap"] += 1
            continue
        info = client.repo(c.repo_id)
        if info is None:
            skipped["fetch_failed"] += 1
            continue
        if safety.is_blocked(
            [c.repo_id, *info.tags],
            json.dumps(info.card, ensure_ascii=False) + "\n" + client.readme(c.repo_id),
            uncensored=c.uncensored,
        ):
            excluded += 1
            continue
        entry = hf_entry(c, info, client)
        if entry is None:
            skipped["no_weights"] += 1
            continue
        # 同じ重みの写し(持ち主違い・名前違い)は、ダウンロード数の多い先のものだけ(shortlist は多い順)
        if is_duplicate(entry, out):
            skipped["duplicate_weights"] += 1
            continue
        count[key] += 1
        out.append(entry)

    stats.update(
        {
            "queries": len(queries),
            "candidates": len(cands),
            "added": len(out),
            "skipped": dict(sorted(skipped.items())),
            "excluded": excluded,
            "failed_queries": len(warnings),
            "warnings": warnings,
        }
    )
    return out


def hf_entry(c: Candidate, info: RepoInfo, client: HfClient | None = None) -> ModelEntry | None:
    """HF のリポジトリ 1 つを一覧の 1 件にする。ライセンスはファイルごとに確かめる(hf.HfClient.resolve_file:
    base_model の道・系統の表。無検閲化したもので元が分からなければ不明)。"""
    chosen = select_weight_files(info)
    main_bytes = sum(info.files[p][0] or 0 for p, r in chosen if r == "main")
    if main_bytes < MIN_MAIN_BYTES:
        return None
    files = []
    for path, role in chosen:
        size, sha = info.files.get(path, (None, None))
        f = ModelFile(
            name=path,
            url=f"https://huggingface.co/{info.repo_id}/resolve/main/{path}",
            sha256=sha,
            size_bytes=size,
            repo_id=info.repo_id,
            gated=info.gated,
            role=role,
        )
        if client is not None:
            apply_license(f, info, client, path, discovered_uncensored=c.uncensored)
        else:
            set_file_license(
                f,
                Resolved(None, source=info.repo_id, path=[info.repo_id], evidence=[f"{UNVERIFIED}: 未確認"]),
            )
        files.append(f)
    main = next((f for f in files if f.role == "main"), files[0])
    e = ModelEntry(
        id=f"hf:{info.repo_id}",
        name=info.repo_id.split("/", 1)[1],
        modality=c.modality,
        tasks=[c.task],
        source="huggingface",
        files=files,
        params_b=guess_params_b(main.name, info.repo_id.split("/", 1)[1]),
        precision=guess_precision(main.name),
        base_model=info.base_models[0] if info.base_models else None,
        content="uncensored" if c.uncensored else "sfw",
        content_evidence=list(c.evidence),
        discovered_from=sorted(set(c.queries)),
        updated=(info.last_modified or "")[:10] or None,
        homepage=info.url,
        downloads=info.downloads if info.downloads is not None else c.downloads,
        likes=info.likes if info.likes is not None else c.likes,
    )
    set_entry_license(e)
    set_completeness(e)
    return e


def lineage(e: ModelEntry) -> set[str]:
    """元のモデルの手がかり(たどった base_model と、名前で当たった系統)。写しの判定に使う。"""
    out: set[str] = set()
    for f in e.files:
        out |= {p for p in f.license_path if p != f.repo_id}
        out |= {x.split(":", 1)[1] for x in f.license_evidence if x.startswith("family:")}
    if e.base_model:
        out.add(e.base_model)
    # ライセンスが不明でも系統は名前で分かる(リポジトリ名・ファイル名)
    out |= {f.key for f in families_for([e.id, *(f.name for f in e.files)])}
    return out


def is_duplicate(e: ModelEntry, kept: list[ModelEntry]) -> bool:
    """同じ重みの写し: sha256 の組が同じ、または元のモデルが重なり合計の大きさが ±1% 以内。"""
    shas = {f.sha256 for f in e.files if f.sha256}
    size = e.total_size_bytes or 0
    lin = lineage(e)
    for k in kept:
        if k.modality != e.modality:
            continue
        kshas = {f.sha256 for f in k.files if f.sha256}
        if shas and shas == kshas:
            return True
        ksize = k.total_size_bytes or 0
        if lin & lineage(k) and size and ksize and abs(size - ksize) <= 0.01 * max(size, ksize):
            return True
    return False


# --- 全体 ----------------------------------------------------------------------------------------


def _silent(_msg: str) -> None:
    pass


def _previous_db() -> Database | None:
    """前の一覧: キャッシュ(前回の update の結果)があればそれ、無ければ同梱の一覧。"""
    cp = cached_db_path()
    if cp.exists():
        with contextlib.suppress(OSError, ValueError):
            return parse_db(json.loads(cp.read_text(encoding="utf-8")), "cache")
    try:
        return parse_db(json.loads(bundled_text("db.json")), "bundled")
    except (OSError, ValueError):
        return None


def size_check(e: ModelEntry) -> dict[str, Any] | None:
    """index の size と、ファイルの大きさの合計を比べる(docs/sources.md の表)。"""
    if not e.template or not e.template.size_bytes or not e.files:
        return None
    sizes = [f.size_bytes for f in e.files]
    if any(s is None for s in sizes):
        return None
    total = sum(sizes)  # type: ignore[arg-type]
    return {
        "id": e.id,
        "index_size": e.template.size_bytes,
        "files_total": total,
        "ratio": round(e.template.size_bytes / total, 3) if total else None,
    }


def build_database(
    fetch: Fetcher | None = None,
    *,
    hf: bool = True,
    hf_search: bool = True,
    head: bool = True,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
    previous: Database | None = None,
    carry_over: bool = True,
) -> Database:
    """上流から一覧を作る。書き出しはしない(run_update が書く)。

    fetch: 取得の仕方(既定は HttpFetcher)。試験では DirFetcher などを渡す。
    previous: 前の一覧(既定はキャッシュ、無ければ同梱の一覧)。実測のあるモデルが上流から消えたときと、
        一部が取れなかったとき(sources.partial)の、取れなかった分を引き継ぐ。後者は同じ LICENSE_POLICY で
        作った一覧からだけ(古い決め方の寛容すぎるライセンスを持ち込まない)。
    carry_over: False なら引き継がない(試験用)。
    """
    fetch = fetch or HttpFetcher()
    say = progress or _silent
    stats: dict[str, Any] = {}
    warnings: list[str] = []

    say("ComfyUI 公式テンプレートの一覧を取得しています")
    index = validate_index(fetch.get_json(INDEX_URL))
    total = sum(len(c["templates"]) for c in index)
    reasons: Counter[str] = Counter()
    chosen: list[tuple[str, dict[str, Any], str]] = []
    seen: set[str] = set()
    for cat in index:
        title = cat.get("title")
        for t in cat["templates"]:
            r = skip_reason(title, t)
            if r is None and t["name"] in seen:
                r = "duplicate"
            if r:
                reasons[r] += 1
                continue
            seen.add(t["name"])
            chosen.append((title, t, template_modality(title, t) or "image"))

    say(f"テンプレート {len(chosen)} 件の JSON を取得しています")
    client = HfClient(fetch, workers=workers)

    def _load(item: tuple[str, dict[str, Any], str]) -> tuple[Any, str | None]:
        try:
            return workflow_models(fetch.get_json(f"{RAW}/{item[1]['name']}.json")), None
        except (FetchError, UpstreamFormatError) as e:
            return None, f"{item[1]['name']}: {e}"

    entries: list[ModelEntry] = []
    tmpl_by_id: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    failed_names: list[str] = []
    for item, (models, err) in zip(chosen, client.map(_load, chosen), strict=False):
        if err:
            failed.append(err)
            failed_names.append(item[1]["name"])
            continue
        e = template_entry(item[0], item[1], item[2], models)
        entries.append(e)
        tmpl_by_id[e.id] = item[1]
    if chosen and len(failed) / len(chosen) > MAX_FAIL_RATIO:
        if all("つながりません" in f for f in failed):
            raise NetworkError(RAW, "テンプレートの JSON を取得できません")
        raise UpstreamFormatError(
            f"テンプレートの JSON の {len(failed)}/{len(chosen)} 件が読めません(上流の形が変わった可能性)。"
            f"例: {failed[0]}"
        )
    warnings += [f"テンプレートを読めません: {f}" for f in failed]

    excluded = 0
    kept: list[ModelEntry] = []
    for e in entries:
        t = tmpl_by_id[e.id]
        names = [e.id, e.name, *(t.get("tags") or [])]
        if safety.is_blocked(
            names, t.get("description") or "", uncensored=bool(_UNCENSORED.search(" ".join(names)))
        ):
            excluded += 1
            continue
        kept.append(e)
    entries = kept

    stats["comfyui-templates"] = {
        "url": INDEX_URL,
        "license": "MIT",
        "templates_total": total,
        "local": len(chosen),
        "skipped": dict(sorted(reasons.items())),
        "loaded": len(entries),
        "failed": len(failed),
    }

    if hf:
        say(f"Hugging Face: {len(_hf_keys(entries))} リポジトリの情報を取得しています")
        enrich_files(entries, client, fetch, head=head)
        net = [f for f in client.failures if f.network]
        repos = len(_hf_keys(entries))
        if repos and len(net) / repos > MAX_FAIL_RATIO:
            raise NetworkError("https://huggingface.co/api/models", "Hugging Face につながりません")
    for e in entries:
        mark_template_content(e, tmpl_by_id[e.id], client)

    hf_stats: dict[str, Any] = {}
    if hf and hf_search:
        say("Hugging Face: 無検閲のモデルと、読み上げ・効果音のモデルを探しています")
        found = discover_hf(client, progress=say, stats=hf_stats)
        ids = {e.id for e in entries}
        entries += [e for e in found if e.id not in ids]
        excluded += hf_stats.get("excluded", 0)
        warnings += hf_stats.pop("warnings", [])

    prev = (previous if previous is not None else _previous_db()) if carry_over else None

    # 一部が取れなかった(テンプレートの JSON・HF の一時的な失敗・429 の待ち時間の上限・検索の失敗)
    transient = {f.repo_id for f in client.failures if f.transient}
    affected = {
        e.id
        for e in entries
        if any(f.repo_id in transient or set(f.license_path) & transient for f in e.files)
    }
    carried = 0
    if (
        prev is not None
        and prev.sources.get("license_policy") == LICENSE_POLICY
        and (affected or failed_names)
    ):
        prev_by_id = {m.id: m for m in prev.models}
        for i, e in enumerate(entries):
            if e.id in affected and e.id in prev_by_id:
                entries[i] = prev_by_id[e.id]
                carried += 1
                warnings.append(f"一部を取得できなかったため前の一覧の値を使いました: {e.id}")
        have = {e.id for e in entries}
        for name in failed_names:
            if name in prev_by_id and name not in have:
                entries.append(prev_by_id[name])
                carried += 1
                warnings.append(f"テンプレートを読めなかったため前の一覧から引き継ぎました: {name}")
    budget = bool(getattr(fetch, "budget_exhausted", False))
    stats["partial"] = bool(failed or transient or hf_stats.get("failed_queries") or budget)
    stats["partial_counts"] = {
        "templates_failed": len(failed),
        "hf_repos_failed_transient": len(transient),
        "entries_affected": len(affected),
        "hf_search_failed": int(hf_stats.get("failed_queries", 0)),
        "rate_limit_budget_exhausted": budget,
        "carried_over": carried,
    }
    stats["license_policy"] = LICENSE_POLICY

    # 実測のあるモデルは、上流から消えても前の一覧から引き継ぐ(実測が宙に浮かないように)
    if prev is not None:
        from .db import load_measurements

        ids = {e.id for e in entries}
        measured = {m.model_id for m in load_measurements()}
        for old in prev.models:
            if old.id in measured and old.id not in ids:
                entries.append(old)
                warnings.append(f"上流に無いため前の一覧から引き継ぎました: {old.id}")

    if hf:
        stats["huggingface"] = {
            "api": "https://huggingface.co/api/models/{repo_id}?blobs=true",
            "repos": len({k for k in client._repos}),
            "failed": len(client.failures),
            "rate_limited": getattr(fetch, "rate_limited", 0),
            "rate_limit_waited_s": round(float(getattr(fetch, "rate_limit_waited", 0.0)), 1),
            "failures": [f"{f.repo_id}: {f.message}" for f in client.failures[:20]],
        }
    if hf_stats:
        stats["huggingface-search"] = hf_stats
    stats["civitai"] = "使っていません(利用規約。docs/sources.md)"
    checks = [c for c in (size_check(e) for e in entries) if c]
    stats["size_check"] = {
        "templates": len(checks),
        "equal": sum(1 for c in checks if c["index_size"] == c["files_total"]),
        "within_10pct": sum(1 for c in checks if c["ratio"] and 0.9 <= c["ratio"] <= 1.1),
    }
    entries.sort(key=lambda e: (e.source != "comfyui-template", e.modality, e.id))
    stats["counts"] = {
        "models": len(entries),
        "modality": dict(sorted(Counter(e.modality for e in entries).items())),
        "content": dict(sorted(Counter(e.content for e in entries).items())),
        "source": dict(sorted(Counter(e.source for e in entries).items())),
        "license": dict(sorted(Counter(e.license or "null" for e in entries).items())),
        "license_tier": dict(
            sorted(Counter(_TIER_NAMES[license_tier(e.license, e.license_name)] for e in entries).items())
        ),
        "files_complete_false": sum(1 for e in entries if e.files_complete is False),
    }
    stats["excluded"] = excluded
    stats["warnings"] = warnings
    stats["generator"] = f"genmodelfit {__version__}"
    if not entries:
        raise UpstreamFormatError("一覧が空になりました(上流の形が変わった可能性)")
    return Database(
        models=entries,
        generated_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        sources=stats,
        origin="path",
    )


# --- CLI ------------------------------------------------------------------------------------------


def run_update(args: argparse.Namespace) -> int:
    out_path = Path(args.output) if getattr(args, "output", None) else cached_db_path()
    quiet = bool(getattr(args, "json", False))
    with contextlib.suppress(AttributeError, ValueError, OSError):
        sys.stderr.reconfigure(encoding="utf-8")  # 途中経過も cp932 で化けないように

    def say(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr, flush=True)

    src = getattr(args, "offline_sources", None)
    budget = getattr(args, "rate_limit_budget", None)
    fetch: Fetcher = (
        DirFetcher(src)
        if src
        else HttpFetcher(rate_limit_budget=RATE_LIMIT_BUDGET if budget is None else budget)
    )
    try:
        db = build_database(
            fetch,
            hf=not getattr(args, "skip_hf", False),
            head=not getattr(args, "skip_head", False),
            progress=say,
        )
    except NetworkError as e:
        return _fail(
            args, f"ネットワークにつながらないため更新できませんでした(前の一覧はそのまま使えます): {e}"
        )
    except (UpdateError, FetchError) as e:
        return _fail(args, f"更新できませんでした(前の一覧はそのまま使えます): {e}")
    if db.sources.get("partial") and not getattr(args, "allow_partial", False):
        pc = db.sources["partial_counts"]
        detail = ", ".join(f"{k}={v}" for k, v in pc.items() if v)
        hint = "時間をおいてやり直すか、" if pc["rate_limit_budget_exhausted"] else ""
        return _fail(
            args,
            f"一部を取得できなかったため書き出しませんでした({detail})。前の一覧はそのまま使えます。"
            f"{hint}欠けたままでよければ --allow-partial を付けてください",
        )
    db.save(out_path)
    counts = db.sources["counts"]
    if quiet:
        from .cli import _envelope, _out

        db.origin = "path"
        _out(_envelope("update", None, db, output=str(out_path), sources=db.sources))
        return 0
    mods = ", ".join(f"{k} {v}" for k, v in counts["modality"].items())
    print(f"{counts['models']} 件を書き出しました: {out_path}")
    print(f"  種類: {mods}")
    print("  内容: " + ", ".join(f"{k} {v}" for k, v in counts["content"].items()))
    print("  出どころ: " + ", ".join(f"{k} {v}" for k, v in counts["source"].items()))
    print("  ライセンス: " + ", ".join(f"{k} {v}" for k, v in counts["license_tier"].items()))
    print(f"  一式がそろっていない(files_complete=false): {counts['files_complete_false']} 件")
    print(f"  除外: {db.sources['excluded']} 件 / 警告: {len(db.sources['warnings'])} 件")
    if db.sources.get("partial"):
        pc = json.dumps(db.sources["partial_counts"], ensure_ascii=False)
        print(f"  注意: 一部を取得できないまま書き出しました(--allow-partial): {pc}")
    return 0


def _fail(args: argparse.Namespace, message: str) -> int:
    if getattr(args, "json", False):
        from .cli import _out

        _out({"genmodelfit": __version__, "command": "update", "error": message})
    else:
        print(f"エラー: {message}", file=sys.stderr)
    return 1
