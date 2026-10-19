"""上流(GitHub の raw・Hugging Face)からの取得。

- HttpFetcher: 実際の取得。User-Agent を名乗り、429 / 503 は Retry-After を守って待ち、失敗は数回やり直す
- DirFetcher: 試験用。URL を決まった規則でファイル名に直し、フォルダから読む(`update --offline-sources DIR`)

どちらも get_json / get_text / head_size を持つ(update は Fetcher の形だけを使う)。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qsl, unquote, urlsplit

import requests

from . import __version__

USER_AGENT = f"genmodelfit/{__version__} (+https://github.com/laxshmate/genmodelfit)"
TIMEOUT = (10, 60)  # (接続, 読み込み) 秒
MAX_ATTEMPTS = 4  # 接続の失敗・5xx のやり直し
MAX_RATE_LIMITED = 30  # 429 のやり直し(HF の匿名の上限は 5 分あたり 500 回。待てば通るので多めに待つ)
MAX_RETRY_AFTER = 300.0
RATE_LIMIT_BUDGET = 900.0  # 429 で待つ時間の合計の上限(秒)。超えたら RateLimitExhausted で止める


class FetchError(RuntimeError):
    """取得できなかった(HTTP のエラー、または中身が JSON でない)。"""

    def __init__(self, url: str, message: str, status: int | None = None):
        super().__init__(f"{message}: {url}")
        self.url = url
        self.status = status

    @property
    def not_found(self) -> bool:
        return self.status in (401, 403, 404, 410)


class NetworkError(FetchError):
    """ネットワークにつながらない(名前解決・接続・時間切れ)。"""


class RateLimitExhausted(FetchError):
    """429(上限)で待った時間の合計が、決めた上限(rate_limit_budget)を超えた。"""


class Fetcher(Protocol):
    def get_json(self, url: str) -> Any: ...

    def get_text(self, url: str) -> str: ...

    def head_size(self, url: str) -> int | None: ...


def _retry_after(resp: requests.Response, attempt: int) -> float:
    value = resp.headers.get("Retry-After", "")
    try:
        return min(MAX_RETRY_AFTER, max(0.0, float(value)))
    except ValueError:
        return float(2**attempt)


class HttpFetcher:
    """requests で取る。スレッドから同時に呼んでよい(セッションはスレッドごと)。"""

    def __init__(
        self, *, sleep=time.sleep, clock=time.monotonic, timeout=TIMEOUT, rate_limit_budget=RATE_LIMIT_BUDGET
    ):
        self._local = threading.local()
        self.rate_limit_budget = float(rate_limit_budget)
        self.rate_limit_waited = 0.0  # 429 で全スレッドを止めた時間の合計(秒。重なりは 1 回と数える)
        self.budget_exhausted = False
        self._sleep = sleep
        self._clock = clock
        self._timeout = timeout
        self._lock = threading.Lock()
        self._pause_until: dict[str, float] = {}  # 429 を受けた相手(ホスト)ごとに、全スレッドで待つ
        self.requests = 0
        self.rate_limited = 0

    def _session(self) -> requests.Session:
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            s.headers["User-Agent"] = USER_AGENT
            self._local.session = s
        return s

    def _wait_turn(self, host: str) -> None:
        while True:
            with self._lock:
                wait = self._pause_until.get(host, 0.0) - self._clock()
            if wait <= 0:
                return
            self._sleep(wait)

    def _request(self, method: str, url: str) -> requests.Response:
        last: Exception | None = None
        host = urlsplit(url).netloc
        attempt = limited = 0
        while attempt < MAX_ATTEMPTS:
            if self.budget_exhausted:
                raise RateLimitExhausted(url, self._budget_message(), 429)
            self._wait_turn(host)
            with self._lock:
                self.requests += 1
            try:
                resp = self._session().request(method, url, timeout=self._timeout, allow_redirects=True)
            except (requests.ConnectionError, requests.Timeout) as e:
                last = e
                self._sleep(float(2**attempt))
                attempt += 1
                continue
            if resp.status_code == 429 and limited < MAX_RATE_LIMITED:
                # 上限に当たった: Retry-After の間、同じ相手への要求を全スレッドで止める
                limited += 1
                with self._lock:
                    self.rate_limited += 1
                    now = self._clock()
                    until = now + _retry_after(resp, 1) + 1.0
                    before = max(self._pause_until.get(host, 0.0), now)
                    self.rate_limit_waited += max(0.0, until - before)
                    self._pause_until[host] = max(self._pause_until.get(host, 0.0), until)
                    if self.rate_limit_waited > self.rate_limit_budget:
                        self.budget_exhausted = True
                if self.budget_exhausted:
                    raise RateLimitExhausted(url, self._budget_message(), 429)
                last = FetchError(url, "HTTP 429", 429)
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                last = FetchError(url, f"HTTP {resp.status_code}", resp.status_code)
                self._sleep(_retry_after(resp, attempt))
                attempt += 1
                continue
            if resp.status_code >= 400:
                raise FetchError(url, f"HTTP {resp.status_code}", resp.status_code)
            return resp
        if isinstance(last, FetchError):
            raise last
        raise NetworkError(url, f"つながりません({type(last).__name__})")

    def _budget_message(self) -> str:
        return (
            f"上限(HTTP 429)で待つ時間の合計が {self.rate_limit_budget:.0f} 秒を超えました"
            "(--rate-limit-budget で変えられます)"
        )

    def get_json(self, url: str) -> Any:
        resp = self._request("GET", url)
        try:
            return resp.json()
        except ValueError as e:
            raise FetchError(url, "JSON ではありません", resp.status_code) from e

    def get_text(self, url: str) -> str:
        resp = self._request("GET", url)
        resp.encoding = resp.encoding or "utf-8"
        return resp.text

    def head_size(self, url: str) -> int | None:
        resp = self._request("HEAD", url)
        for key in ("x-linked-size", "content-length"):
            v = resp.headers.get(key)
            if v and v.isdigit():
                return int(v)
        return None


# --- 試験用: URL ↔ ファイル ---------------------------------------------------------------------


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def url_to_relpath(url: str) -> str:
    """URL をフォルダ内の相対パスに直す(人が見て分かる名前にする)。"""
    u = urlsplit(url)
    path = unquote(u.path)
    if u.netloc == "raw.githubusercontent.com" and "/templates/" in path:
        return "templates/" + _safe(path.rsplit("/templates/", 1)[1])
    if u.netloc == "huggingface.co" and path.startswith("/api/models/"):
        rest = path[len("/api/models/") :]
        return "hf/" + _safe(rest.replace("/", "__")) + ".json"
    if u.netloc == "huggingface.co" and path == "/api/models":
        q = "&".join(f"{k}={v}" for k, v in sorted(parse_qsl(u.query)))
        return "hf_search/" + _safe(q)[:120] + ".json"
    if u.netloc == "huggingface.co" and "/raw/" in path:
        repo = path.strip("/").split("/raw/")[0]
        return "hf_readme/" + _safe(repo.replace("/", "__")) + ".md"
    digest = hashlib.sha1(url.encode()).hexdigest()[:12]
    return "other/" + _safe(u.netloc) + "_" + digest


class DirFetcher:
    """フォルダに置いたファイルを上流の代わりに返す。無いファイルは 404 と同じ扱い。"""

    def __init__(self, root: str | Path, *, sizes: dict[str, int] | None = None):
        self.root = Path(root)
        self.sizes = sizes or {}
        self.requests = 0

    def _path(self, url: str) -> Path:
        self.requests += 1
        p = self.root / url_to_relpath(url)
        if not p.exists():
            raise FetchError(url, "ファイルがありません(試験用のフォルダ)", 404)
        return p

    def get_json(self, url: str) -> Any:
        p = self._path(url)
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            raise FetchError(url, "JSON ではありません") from e

    def get_text(self, url: str) -> str:
        return self._path(url).read_text(encoding="utf-8")

    def head_size(self, url: str) -> int | None:
        return self.sizes.get(url)


class RecordingFetcher:
    """本物の取得をしながら、DirFetcher で読める形でフォルダに保存する(フィクスチャ作り用)。"""

    def __init__(self, inner: Fetcher, root: str | Path):
        self.inner = inner
        self.root = Path(root)

    def _save(self, url: str, text: str) -> None:
        p = self.root / url_to_relpath(url)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def get_json(self, url: str) -> Any:
        data = self.inner.get_json(url)
        self._save(url, json.dumps(data, ensure_ascii=False))
        return data

    def get_text(self, url: str) -> str:
        text = self.inner.get_text(url)
        self._save(url, text)
        return text

    def head_size(self, url: str) -> int | None:
        return self.inner.head_size(url)
