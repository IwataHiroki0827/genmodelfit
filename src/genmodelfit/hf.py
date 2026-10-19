"""Hugging Face Hub API から、ライセンス・利用申請(gated)・ファイルの大きさ・sha256 を取る。

- 1 リポジトリにつき `GET /api/models/{repo_id}?blobs=true` を 1 回だけ呼ぶ(結果はこの実行の中で使い回す)
- Comfy-Org などの再配布リポジトリは、ライセンスが本家と違うことがある。`cardData.base_model` をたどって
  本家のライセンスをファイルに記録し、再配布側の表記は `repo_license` に残す(docs/sources.md)
- 一式のライセンスは「最も厳しいもの」:
  不明(None) > other > 制限付き(openrail・非商用など)> 寛容(apache-2.0・mit など)
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urlencode, urlsplit

from .fetch import Fetcher, FetchError, NetworkError

HF = "https://huggingface.co"

# 寛容なライセンス(商用で使え、配布物のソースを公開しなくてよいもの)。HF の license タグの表記
PERMISSIVE = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "isc",
    "zlib",
    "unlicense",
    "cc0-1.0",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "pddl",
    "odc-by",
    "bsl-1.0",
    "afl-3.0",
    "ecl-2.0",
    "postgresql",
    "wtfpl",
}

# 再配布(リパッケージ)をしている持ち主。ここのリポジトリは base_model をたどって本家のライセンスを見る
REPACKAGERS = ("Comfy-Org", "comfyanonymous", "Kijai")

# base_model を書いていない再配布リポジトリの本家(上流を確かめて手で書いたもの。docs/sources.md に同じ表)
REPACKAGED_UPSTREAM: dict[str, list[str]] = {
    # OpenCLIP ViT-bigG/14(SDXL の clip_vision)。HF の laion のリポジトリは license: mit
    "comfyanonymous/clip_vision_g": ["laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"],
}

WEIGHT_EXT = (".safetensors", ".gguf", ".ckpt", ".bin", ".pth", ".pt", ".sft")

# 名前の比較で無視する語(精度・形式などモデルの種類を表さないもの)
_GENERIC_TOKENS = {
    "safetensors",
    "gguf",
    "fp8",
    "fp16",
    "bf16",
    "fp32",
    "fp4",
    "nvfp4",
    "int8",
    "int4",
    "e4m3fn",
    "e5m2",
    "scaled",
    "mixed",
    "comfy",
    "comfyui",
    "repackaged",
    "split",
    "files",
    "model",
    "models",
    "diffusion",
    "org",
    "ai",
    "convrot",
    "the",
}


# --- URL ---------------------------------------------------------------------------------------


def parse_hf_url(url: str | None) -> tuple[str, str, str] | None:
    """`https://huggingface.co/<owner>/<repo>/resolve/<rev>/<path>` → (repo_id, revision, path)。"""
    if not url:
        return None
    u = urlsplit(url)
    if u.netloc not in ("huggingface.co", "www.huggingface.co"):
        return None
    parts = u.path.strip("/").split("/")
    if len(parts) < 5 or parts[2] not in ("resolve", "blob", "raw"):
        return None
    repo_id = f"{parts[0]}/{parts[1]}"
    return repo_id, unquote(parts[3]), unquote("/".join(parts[4:]))


def api_url(repo_id: str, revision: str = "main") -> str:
    if revision in ("main", ""):
        return f"{HF}/api/models/{repo_id}?blobs=true"
    return f"{HF}/api/models/{repo_id}/revision/{quote(revision, safe='')}?blobs=true"


def readme_url(repo_id: str) -> str:
    return f"{HF}/{repo_id}/raw/main/README.md"


def search_url(**params: Any) -> str:
    q = {"sort": "downloads", "direction": "-1", "limit": "100"}
    q.update({k: str(v) for k, v in params.items() if v is not None})
    return f"{HF}/api/models?{urlencode(sorted(q.items()))}"


# --- ライセンス ------------------------------------------------------------------------------


def normalize_license(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip().lower()
    return None if v in ("unknown", "none", "n/a", "unlicensed") else v


def license_rank(lic: str | None) -> int:
    """大きいほど厳しい。不明 4 / other 3 / 制限付き 2 / 寛容 1。"""
    if lic is None:
        return 4
    if lic == "other":
        return 3
    if lic in PERMISSIVE:
        return 1
    return 2


def strictest(licenses: Iterable[str | None]) -> str | None:
    items = list(licenses)
    if not items:
        return None
    return max(items, key=lambda x: (license_rank(x), x or ""))


@dataclass
class RepoInfo:
    repo_id: str
    license: str | None = None
    license_name: str | None = None
    license_link: str | None = None
    gated: bool | None = None
    base_models: list[str] = field(default_factory=list)
    files: dict[str, tuple[int | None, str | None]] = field(default_factory=dict)  # path → (size, sha256)
    downloads: int | None = None
    likes: int | None = None
    tags: list[str] = field(default_factory=list)
    pipeline_tag: str | None = None
    library_name: str | None = None
    last_modified: str | None = None
    card: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> RepoInfo:
        card = data.get("cardData") if isinstance(data.get("cardData"), dict) else {}
        lic = normalize_license(card.get("license"))
        if lic is None:  # cardData に無ければタグ(license:xxx)を見る
            for t in data.get("tags") or []:
                if isinstance(t, str) and t.startswith("license:"):
                    lic = normalize_license(t.split(":", 1)[1])
                    break
        base = card.get("base_model")
        bases = [base] if isinstance(base, str) else [b for b in (base or []) if isinstance(b, str)]
        gated = data.get("gated")
        files: dict[str, tuple[int | None, str | None]] = {}
        for s in data.get("siblings") or []:
            if not isinstance(s, dict) or not s.get("rfilename"):
                continue
            lfs = s.get("lfs") if isinstance(s.get("lfs"), dict) else {}
            files[s["rfilename"]] = (s.get("size") or lfs.get("size"), lfs.get("sha256"))
        ln = card.get("license_name")
        link = card.get("license_link")
        return cls(
            repo_id=data.get("id") or data.get("modelId") or "",
            license=lic,
            license_name=ln if isinstance(ln, str) else None,
            license_link=link if isinstance(link, str) else None,
            gated=None if gated is None else bool(gated),
            base_models=[b.strip() for b in bases if b.strip()],
            files=files,
            downloads=data.get("downloads"),
            likes=data.get("likes"),
            tags=[t for t in data.get("tags") or [] if isinstance(t, str)],
            pipeline_tag=data.get("pipeline_tag"),
            library_name=data.get("library_name"),
            last_modified=data.get("lastModified"),
            card=card,
        )

    @property
    def url(self) -> str:
        return f"{HF}/{self.repo_id}"

    def license_url(self) -> str:
        link = self.license_link
        if link and link.startswith(("http://", "https://")):
            return link
        if link:
            return f"{HF}/{self.repo_id}/blob/main/{link.lstrip('./')}"
        return self.url


def _tokens(text: str) -> set[str]:
    toks = re.split(r"[^a-z0-9]+", text.lower())
    return {t for t in toks if t and t not in _GENERIC_TOKENS and not t.isdigit()}


def choose_bases(bases: list[str], filename: str) -> list[str]:
    """再配布のファイルが、どの本家から来たか。1 つなら確定。複数なら名前の重なりで選ぶ(同点は全部)。
    どれとも重ならなければ空(再配布側の表記を使う)。"""
    if len(bases) <= 1:
        return list(bases)
    ft = _tokens(filename.rsplit("/", 1)[-1])
    scored = [(len(ft & _tokens(b.split("/", 1)[-1])), b) for b in bases]
    best = max(s for s, _ in scored)
    return [b for s, b in scored if s == best] if best > 0 else []


def is_repackager(repo_id: str) -> bool:
    return repo_id.split("/", 1)[0] in REPACKAGERS


def upstream_candidates(info: RepoInfo) -> list[str]:
    return info.base_models or REPACKAGED_UPSTREAM.get(info.repo_id, [])


# --- 取得 --------------------------------------------------------------------------------------


@dataclass
class Failure:
    repo_id: str
    message: str
    network: bool


class HfClient:
    """リポジトリの情報を取って覚えておく。スレッドから同時に呼んでよい。"""

    def __init__(self, fetch: Fetcher, *, workers: int = 8):
        self.fetch = fetch
        self.workers = max(1, min(8, workers))  # 上流に優しく: 同時に 8 本まで
        self._repos: dict[tuple[str, str], RepoInfo | None] = {}
        self._readmes: dict[str, str] = {}
        self._lock = threading.Lock()
        self.failures: list[Failure] = []

    def repo(self, repo_id: str, revision: str = "main") -> RepoInfo | None:
        key = (repo_id, revision or "main")
        with self._lock:
            if key in self._repos:
                return self._repos[key]
        info: RepoInfo | None
        try:
            data = self.fetch.get_json(api_url(*key))
            info = RepoInfo.from_api(data) if isinstance(data, dict) else None
            if info is None:
                raise FetchError(api_url(*key), "形が違います")
            info.repo_id = info.repo_id or repo_id
        except FetchError as e:
            info = None
            with self._lock:
                self.failures.append(Failure(repo_id, str(e), isinstance(e, NetworkError)))
        with self._lock:
            self._repos[key] = info
        return info

    def readme(self, repo_id: str) -> str:
        with self._lock:
            if repo_id in self._readmes:
                return self._readmes[repo_id]
        try:
            text = self.fetch.get_text(readme_url(repo_id))
        except FetchError:
            text = ""
        with self._lock:
            self._readmes[repo_id] = text
        return text

    def map(self, func: Callable[[Any], Any], items: Iterable[Any]) -> list[Any]:
        items = list(items)
        if not items:
            return []
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(func, items))

    def prefetch(self, keys: Iterable[tuple[str, str]]) -> None:
        todo = sorted({k for k in keys if k not in self._repos})
        self.map(lambda k: self.repo(*k), todo)

    def search(self, url: str) -> list[dict[str, Any]]:
        data = self.fetch.get_json(url)
        if not isinstance(data, list):
            raise FetchError(url, "検索の結果が一覧ではありません")
        return [d for d in data if isinstance(d, dict) and d.get("id")]

    def file_license(self, info: RepoInfo, path: str) -> tuple[str | None, str | None, RepoInfo]:
        """(license, license_name, ライセンスを取ったリポジトリ)。再配布なら本家をたどる。"""
        if is_repackager(info.repo_id):
            picked = choose_bases(upstream_candidates(info), path)
            # 本家にライセンスの記載が無いときは、再配布側の表記を使う(どちらも無ければ不明のまま)
            ups = [u for u in (self.repo(b) for b in picked) if u is not None and u.license is not None]
            if ups:
                src = max(ups, key=lambda u: (license_rank(u.license), u.repo_id))
                return src.license, src.license_name, src
        return info.license, info.license_name, info


# --- HF のリポジトリから、使うファイルを選ぶ -----------------------------------------------------

_SHARD = re.compile(r"[-_.]?(\d{5})-of-(\d{5})")
_VARIANT = re.compile(r"\.(fp16|bf16|fp32|fp8|non_ema|ema)(?=\.[a-z]+$)")
_GGUF_PREF = ("q8_0", "q4_k_m", "q5_k_m", "q6_k", "q4_k_s", "q5_k_s")
_SKIP_PATH = re.compile(
    r"(^|/)(\.|optimizer|scheduler|training_args|rng_state|safety_checker|feature_extractor|onnx|openvino|"
    r"coreml|flax|tf_model|voices?/|samples?/|examples?/)",
    re.IGNORECASE,
)
_DIFFUSERS_ROLE = {
    "transformer": "main",
    "transformer_2": "main",
    "unet": "main",
    "prior": "main",
    "decoder": "main",
    "text_encoder": "text_encoder",
    "text_encoder_2": "text_encoder",
    "text_encoder_3": "text_encoder",
    "vae": "vae",
    "image_encoder": "clip_vision",
}


def name_role(path: str) -> str:
    n = path.lower()
    if re.search(r"mmproj|tokenizer|(^|[/_\-.])(codec|asr)([/_\-.]|$)", n):
        return "other"  # 画像の読み取り部・トークナイザなど。本体の判定には使わない
    if re.search(r"(^|[/_\-.])(vae|ae)([/_\-.]|$)", n):
        return "vae"
    if re.search(
        r"text[_\-]?encoder|(^|[/_\-.])(t5|t5xxl|umt5[_a-z]*|clip_[lg]|gemma[\w.]*|qwen[\d._]*_?vl[\w.]*|qwen_?[\d.]+_\d+b)([/_\-.]|$)",
        n,
    ):
        return "text_encoder"
    if re.search(r"lora|lycoris", n):
        return "lora"
    if re.search(r"clip[_\-]?vision|image_encoder", n):
        return "clip_vision"
    return "main"


def _variant_key(path: str) -> str:
    return _SHARD.sub("", path)


def _group(paths: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for p in sorted(paths):
        groups.setdefault(_variant_key(p), []).append(p)
    return groups


def _size(info: RepoInfo, paths: list[str]) -> int:
    return sum((info.files.get(p, (0, None))[0] or 0) for p in paths)


def _pick_variant(info: RepoInfo, groups: dict[str, list[str]]) -> list[str]:
    """1 つの役割の中から 1 つの版を選ぶ。

    GGUF は Q8_0 → Q4_K_M → Q5_K_M → Q6_K → Q4_K_S → Q5_K_S の順、無ければ一番小さいもの。
    それ以外は safetensors を優先し、fp32 の版は他が無いときだけ使い、
    残りから一番大きいもの(完全な本体)を選ぶ。
    """
    if not groups:
        return []
    keys = list(groups)
    gguf = [k for k in keys if k.lower().endswith(".gguf")]
    if gguf:
        for pref in _GGUF_PREF:
            hit = [k for k in gguf if pref in k.lower()]
            if hit:
                return groups[min(hit, key=lambda k: _size(info, groups[k]))]
        return groups[min(gguf, key=lambda k: _size(info, groups[k]))]
    st = [k for k in keys if k.lower().endswith((".safetensors", ".sft"))]
    pool = st or keys
    non32 = [k for k in pool if "fp32" not in k.lower()] or pool
    return groups[max(non32, key=lambda k: (_size(info, groups[k]), k))]


def select_weight_files(info: RepoInfo) -> list[tuple[str, str]]:
    """HF のリポジトリから、動かすのに使う重みを 1 組だけ (path, role) で返す(docs/sources.md の規則)。

    見積もりはファイルの合計を使うので、同じ部品の別の版(精度・形式)を重ねて入れない。

    - 単一ファイルの本体(リポジトリ直下)があれば、diffusers の形があってもそちらを使う(ComfyUI が読む形)
    - diffusers の形だけのとき: 部品のフォルダ(transformer / unet / text_encoder* / vae …)ごとに 1 版。
      精度は fp16 → bf16 → 印の無い既定の版の順。分割されたファイルはまとめて取る
    - それ以外: 役割(本体・テキストエンコーダ・VAE)ごとに 1 版を選ぶ(_pick_variant)。LoRA は取らない
    """
    weights = [
        p
        for p in info.files
        if p.lower().endswith(WEIGHT_EXT) and not _SKIP_PATH.search(p) and info.files[p][0]
    ]
    if not weights:
        return []
    out: list[tuple[str, str]] = []
    diffusers = "model_index.json" in info.files
    top_main = [p for p in weights if "/" not in p and name_role(p) == "main"]
    if diffusers and top_main:
        # 単一ファイルを使う。diffusers の部品フォルダの中のものは数えない
        weights = [p for p in weights if p.split("/", 1)[0] not in _DIFFUSERS_ROLE]
    elif diffusers:
        by_dir: dict[str, list[str]] = {}
        for p in weights:
            if "/" in p:
                by_dir.setdefault(p.split("/", 1)[0], []).append(p)
        for d, paths in sorted(by_dir.items()):
            role = _DIFFUSERS_ROLE.get(d)
            if role is None:
                continue
            groups = _group(paths)
            st = {k: v for k, v in groups.items() if k.lower().endswith(".safetensors")} or groups
            chosen = None
            for pref in ("fp16", "bf16"):
                hit = sorted(k for k in st if f".{pref}." in k)
                if hit:
                    chosen = st[hit[0]]
                    break
            if chosen is None:
                default = {k: v for k, v in st.items() if not _VARIANT.search(k)}
                chosen = (
                    default[max(default, key=lambda k: _size(info, default[k]))]
                    if default
                    else _pick_variant(info, st)
                )
            out += [(p, role) for p in chosen]
        if any(r == "main" for _, r in out):
            return out
        out = []
    by_role: dict[str, list[str]] = {}
    for p in weights:
        by_role.setdefault(name_role(p), []).append(p)
    # 別々のモデルを詰め合わせたリポジトリ(本体が役割名でないフォルダ 4 つ以上に分かれている)は選べない
    model_dirs = {_top_dir(p) for p in by_role.get("main", [])} - {""}
    if len(model_dirs - _ROLE_DIRS) >= 4 and not any(_top_dir(p) == "" for p in by_role.get("main", [])):
        return []
    main_dir: str | None = None
    for role in ("main", "text_encoder", "vae"):
        groups = _group(by_role.get(role, []))
        if role == "main":
            # リポジトリ名と語が重なるファイルがあれば、そこから選ぶ(別のモデルの同梱を避ける)
            rt = _tokens(info.repo_id.split("/", 1)[-1])
            near = {k: v for k, v in groups.items() if _tokens(k.rsplit("/", 1)[-1]) & rt}
            groups = near or groups
        else:
            # 補助の部品は、本体と同じフォルダ・直下・役割名のフォルダにあるものだけ
            ok = {"", main_dir, *_ROLE_DIRS}
            groups = {k: v for k, v in groups.items() if _top_dir(k) in ok}
        picked = _pick_variant(info, groups)
        if role == "main" and picked:
            main_dir = _top_dir(picked[0])
        out += [(p, role) for p in picked]
    if not any(r == "main" for _, r in out):
        return []
    return out


_ROLE_DIRS = {
    "text_encoder",
    "text_encoders",
    "vae",
    "clip",
    "split_files",
    "diffusion_models",
    "checkpoints",
    "unet",
    "transformer",
    "models",
    "gguf",
}


def _top_dir(path: str) -> str:
    return path.split("/", 1)[0].lower() if "/" in path else ""


# --- 名前から精度・パラメータ数を推す --------------------------------------------------------------

_PRECISION = (
    ("nvfp4", r"nvfp4"),
    ("fp4", r"(?<![a-z])fp4"),
    ("nf4", r"(?<![a-z])nf4"),
    ("fp8", r"fp8|e4m3|e5m2"),
    ("int8", r"int8"),
    ("int4", r"int4"),
    ("bf16", r"bf16"),
    ("fp16", r"fp16"),
    ("fp32", r"fp32"),
)


def guess_precision(filename: str | None) -> str | None:
    if not filename:
        return None
    n = filename.lower()
    if n.endswith(".gguf"):
        m = re.search(r"(?<![a-z0-9])(i?q\d[a-z0-9_]*?|f16|bf16|f32)(?=\.gguf$|[-.])", n)
        return f"gguf-{m.group(1)}" if m else "gguf"
    for name, pat in _PRECISION:
        if re.search(pat, n):
            return name
    return None


def guess_params_b(*texts: str | None) -> float | None:
    for t in texts:
        if not t:
            continue
        m = re.search(r"(?<![a-z0-9.])(\d{1,3}(?:\.\d+)?)\s?b(?![a-z0-9])", t, re.IGNORECASE)
        if m:
            v = float(m.group(1))
            if 0.05 <= v <= 200:
                return v
    return None
