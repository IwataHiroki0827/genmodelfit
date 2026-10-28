# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`share` が実測を PR で送るための GitHub の通信(#124。llmfit の share の作りにならう)。

- ログイン: 環境変数 GITHUB_TOKEN / GH_TOKEN → キャッシュしたトークン → デバイス認証(scope public_repo)。
  `GET /user` で確かめる
- 送る: フォーク(無ければ作り、出来るまで待つ)→ 本家の main の先頭からブランチ → Contents API で
  data/contrib/<GPU>/<日付>-<sha8>.json を置く → PR を開く。同じ利用者の開いている share の PR があれば、
  そのブランチに足す。同じ名前のファイルは飛ばす(やり直しても重ならない)

ライブラリの関数は何も表示しない(表示は cli.py)。途中のことは呼び出し側が渡す log・on_code に出す。
通信は渡された session(fetch.make_session())だけで、相手は https の github.com と api.github.com だけ。
トークンは表示・ログ・誤りの文・JSON に出さない。キャッシュは config_dir()/github_token に、利用者だけが
読める権限(POSIX は 0o600)で書く。docs/schema.md「share」
"""

from __future__ import annotations

import base64
import contextlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from . import __version__
from .contrib import REPO, md_cell
from .i18n import tr

UPSTREAM = REPO  # 送る先(laxshmate/genmodelfit)
BASE_BRANCH = "main"
API = "https://api.github.com"
DEVICE_CODE_URL = "https://github.com/login/device/code"
TOKEN_URL = "https://github.com/login/oauth/access_token"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
SCOPE = "public_repo"
ALLOWED_HOSTS = ("github.com", "api.github.com")
# OAuth アプリの client id(公開してよい値。デバイス認証は secret が要らない)。
# 2026-10-29 にオーナーが登録した。
# アプリは「Expire user access tokens」が有効なので、トークンは約 8 時間で切れる。secret を持たないので
# refresh はせず、refresh_token も保存しない(切れたらデバイス認証をやり直す)
CLIENT_ID: str | None = "Ov23liLX2TDKfSO6B955"
CLIENT_ID_ENV = "GENMODELFIT_GH_CLIENT_ID"  # 替える。空にするとデバイス認証を使わない
TOKEN_ENVS = ("GITHUB_TOKEN", "GH_TOKEN")
TOKEN_FILE = "github_token"
BRANCH_PREFIX = "genmodelfit-share/"
PR_MARKER = "<!-- genmodelfit:share -->"  # genmodelfit share で作った PR の印(本文の最後)
TIMEOUT = (10, 60)
FORK_WAIT_TRIES = 15  # フォークが出来るまで 2 秒ずつ待つ回数
FORK_WAIT_SECONDS = 2.0


class GitHubError(RuntimeError):
    """ログイン・送るのに失敗した(文はそのまま利用者に見せられる。トークンは入れない)。
    kind: no_login / invalid_token / expired / denied / scope / network / api。"""

    def __init__(self, message: str, kind: str = "api", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


# --- 資格情報 -------------------------------------------------------------------------------------


def client_id() -> str | None:
    """デバイス認証の client id。環境変数 GENMODELFIT_GH_CLIENT_ID があればそれ
    (空ならデバイス認証を使わない)。"""
    if CLIENT_ID_ENV in os.environ:
        return os.environ[CLIENT_ID_ENV].strip() or None
    return CLIENT_ID


def token_path() -> Path:
    """キャッシュしたトークンの置き場所(platformdirs の user_config_dir("genmodelfit")/github_token。
    GENMODELFIT_CONFIG_DIR で変えられる)。"""
    from .config import config_dir

    return config_dir() / TOKEN_FILE


EXPIRY_MARGIN = 60.0  # 期限の少し前から切れたものとみなす(秒)


def read_cached_token(now: Callable[[], float] = time.time) -> str | None:
    """キャッシュしたトークン({"access_token", "expires_at"} の JSON)。無い・読めない・期限が切れた
    (expires_at の EXPIRY_MARGIN 秒前を過ぎた)なら None。切れた・壊れたキャッシュは消す。"""
    import json

    try:
        text = token_path().read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        data = json.loads(text)
    except ValueError:
        data = None
    token = data.get("access_token") if isinstance(data, dict) else None
    expires = data.get("expires_at") if isinstance(data, dict) else None
    if not isinstance(token, str) or not token.strip():
        clear_cached_token()
        return None
    if isinstance(expires, (int, float)) and now() >= float(expires) - EXPIRY_MARGIN:
        clear_cached_token()
        return None
    return token.strip()


def write_cached_token(
    token: str, expires_in: float | None = None, now: Callable[[], float] = time.time
) -> Path:
    """トークンと期限(expires_in 秒後。分からなければ None)だけを、利用者だけが読める権限で書く
    (POSIX は 0o600。Windows はユーザのプロファイルの下の既定の ACL)。refresh_token は書かない。"""
    import json

    expires_at = round(now() + float(expires_in)) if expires_in else None
    p = token_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"access_token": token.strip(), "expires_at": expires_at}) + "\n")
    if os.name == "posix":
        os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    return p


def clear_cached_token() -> bool:
    """キャッシュしたトークンを消す。消したか(無ければ False)。"""
    try:
        token_path().unlink()
    except FileNotFoundError:
        return False
    return True


def env_token() -> tuple[str, str] | None:
    """(トークン, 環境変数の名前)。無ければ None。"""
    for name in TOKEN_ENVS:
        v = os.environ.get(name, "").strip()
        if v:
            return v, name
    return None


@dataclass
class Credentials:
    token: str = field(repr=False)
    login: str
    source: str  # env:<名前> / cache / device

    def to_dict(self) -> dict[str, Any]:
        """JSON に入れる中身(トークンは入れない)。"""
        return {"login": self.login, "source": self.source}


# --- 通信 -----------------------------------------------------------------------------------------


def _check_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise GitHubError(f"refused to contact {u.scheme}://{u.hostname}", kind="api")


def _request(session: Any, method: str, url: str, **kw: Any) -> tuple[int, Any]:
    import requests

    _check_url(url)
    try:
        r = session.request(method, url, timeout=TIMEOUT, allow_redirects=False, **kw)
    except requests.RequestException as e:
        raise GitHubError(
            tr(
                f"GitHub に接続できません({type(e).__name__})",
                f"cannot connect to GitHub ({type(e).__name__})",
            ),
            kind="network",
        ) from None
    try:
        body = r.json()
    except ValueError:
        body = None
    return r.status_code, body


def _api(
    session: Any, token: str, method: str, path: str, body: Any = None, params: dict | None = None
) -> tuple[int, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    kw: dict[str, Any] = {"headers": headers}
    if body is not None:
        kw["json"] = body
    if params:
        kw["params"] = params
    return _request(session, method, API + path, **kw)


def _message(status: int, body: Any) -> str:
    msg = body.get("message") if isinstance(body, dict) else None
    return f"GitHub API {status}" + (f": {msg}" if isinstance(msg, str) and msg else "")


def _write_denied(status: int, body: Any, login: str) -> GitHubError:
    """書けなかった。403・404 はほとんどトークンの scope が足りない
    (GitHub は書けないリポジトリに 404 を返す)。"""
    base = _message(status, body)
    if status not in (403, 404):
        return GitHubError(base, status=status)
    return GitHubError(
        base
        + tr(
            f"。トークンは {login} としてログインしていますが、フォークに書けません。"
            f"トークンの scope が足りません({SCOPE} が要ります)。GITHUB_TOKEN・GH_TOKEN を使っているなら、"
            f"{SCOPE} の付いたトークンにするか、外してログインし直してください",
            f". The token is signed in as {login} but cannot write to the fork. The token's scope is not"
            f" enough ({SCOPE} is needed). If you use GITHUB_TOKEN or GH_TOKEN, use a token with {SCOPE},"
            " or unset it and sign in again",
        ),
        kind="scope",
        status=status,
    )


def whoami(session: Any, token: str) -> str | None:
    """トークンのアカウント名。使えないトークン(401)なら None。"""
    status, body = _api(session, token, "GET", "/user")
    if status == 401:
        return None
    if status != 200 or not isinstance(body, dict) or not isinstance(body.get("login"), str):
        raise GitHubError(_message(status, body), status=status)
    return body["login"]


# --- デバイス認証 ---------------------------------------------------------------------------------


@dataclass
class DeviceCode:
    device_code: str = field(repr=False)
    user_code: str
    verification_uri: str
    interval: int
    expires_in: int


def start_device_flow(session: Any, cid: str) -> DeviceCode:
    status, body = _request(
        session,
        "POST",
        DEVICE_CODE_URL,
        data={"client_id": cid, "scope": SCOPE},
        headers={"Accept": "application/json"},
    )
    if status != 200 or not isinstance(body, dict) or "device_code" not in body:
        raise GitHubError(
            tr("デバイス認証を始められません: ", "cannot start the device login: ") + _message(status, body),
            status=status,
        )
    return DeviceCode(
        device_code=str(body["device_code"]),
        user_code=str(body.get("user_code", "")),
        verification_uri=str(body.get("verification_uri") or "https://github.com/login/device"),
        interval=max(1, int(body.get("interval") or 5)),
        expires_in=int(body.get("expires_in") or 900),
    )


def poll_device_flow(session: Any, cid: str, device_code: str) -> tuple[str, Any]:
    """1 回だけ尋ねる: ("token", (トークン, 期限の秒か None)) / ("pending", None) /
    ("slow_down", 新しい間隔か None)。refresh_token は受け取っても使わない(secret が無いので refresh しない)。
    期限切れ・断られた・ほかの誤りは GitHubError。"""
    status, body = _request(
        session,
        "POST",
        TOKEN_URL,
        data={"client_id": cid, "device_code": device_code, "grant_type": DEVICE_GRANT},
        headers={"Accept": "application/json"},
    )
    if not isinstance(body, dict):
        raise GitHubError(_message(status, body), status=status)
    if isinstance(body.get("access_token"), str) and body["access_token"]:
        exp = body.get("expires_in")
        return "token", (body["access_token"], float(exp) if isinstance(exp, (int, float)) else None)
    err = body.get("error")
    if err == "authorization_pending":
        return "pending", None
    if err == "slow_down":
        return "slow_down", body.get("interval")
    if err == "expired_token":
        raise GitHubError(
            tr(
                "ログインのコードの期限が切れました。もう一度やり直してください",
                "the login code expired; try again",
            ),
            kind="expired",
        )
    if err == "access_denied":
        raise GitHubError(tr("ログインが断られました", "the login was denied"), kind="denied")
    raise GitHubError(
        tr("ログインできません: ", "cannot sign in: ") + str(err or _message(status, body)), status=status
    )


def device_flow(
    session: Any,
    cid: str,
    on_code: Callable[[DeviceCode], None],
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[str, float | None]:
    """デバイス認証で (トークン, 期限の秒か None) を得る。コードと URL は on_code に渡す
    (表示とブラウザは呼び出し側)。
    間隔どおりに待ち、slow_down なら間隔を 5 秒増やす。期限を過ぎたら止める。"""
    dc = start_device_flow(session, cid)
    on_code(dc)
    interval = dc.interval
    deadline = clock() + dc.expires_in
    while True:
        sleep(interval)
        kind, value = poll_device_flow(session, cid, dc.device_code)
        if kind == "token":
            return value
        if kind == "slow_down":
            interval = max(interval + 5, int(value) if isinstance(value, int) else 0)
        if clock() > deadline:
            raise GitHubError(
                tr(
                    "ログインのコードの期限が切れました。もう一度やり直してください",
                    "the login code expired; try again",
                ),
                kind="expired",
            )


def no_login_message(device_possible: bool = False) -> str:
    """ログインできないときの文。device_possible は、端末ならデバイス認証でログインできるとき。"""
    if device_possible:
        return tr(
            "ログインできません(ブラウザでのログインは端末で --json なしのときだけ)。端末で実行するか、"
            "GITHUB_TOKEN を設定するか、--issue で送ってください",
            "Cannot sign in (signing in with the browser needs a terminal and no --json). Run it in a"
            " terminal, set GITHUB_TOKEN, or send it with --issue",
        )
    return tr(
        "ログインできません。GITHUB_TOKEN を設定するか、--issue で送ってください",
        "Cannot sign in. Set GITHUB_TOKEN, or send it with --issue",
    )


def login(
    session: Any,
    *,
    interactive: bool,
    on_code: Callable[[DeviceCode], None] | None = None,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Credentials:
    """資格情報を決めて確かめる: 環境変数 GITHUB_TOKEN / GH_TOKEN → キャッシュ → デバイス認証
    (interactive のときだけ。得たトークンはキャッシュする)。使えなければ GitHubError。"""
    say = log or (lambda msg: None)
    env = env_token()
    if env is not None:
        token, name = env
        user = whoami(session, token)
        if user is None:
            raise GitHubError(
                tr(
                    f"{name} のトークンが使えません(期限切れか、正しくない)。直すか、外してください",
                    f"the token in {name} does not work (expired or wrong); fix or unset it",
                ),
                kind="invalid_token",
                status=401,
            )
        return Credentials(token, user, f"env:{name}")
    had_cache = token_path().exists()
    cached = read_cached_token()
    if had_cache and cached is None:
        say(
            tr(
                "前にログインしたトークンの期限が切れました。ログインし直します",
                "the saved login has expired; signing in again",
            )
        )
    if cached:
        user = whoami(session, cached)
        if user is not None:
            return Credentials(cached, user, "cache")
        clear_cached_token()
        say(
            tr(
                "前にログインしたトークンが使えなくなりました。ログインし直します",
                "the saved login no longer works; signing in again",
            )
        )
    cid = client_id()
    if not cid or not interactive or on_code is None:
        raise GitHubError(no_login_message(bool(cid)), kind="no_login")
    token, expires_in = device_flow(session, cid, on_code, sleep=sleep)
    user = whoami(session, token)
    if user is None:
        raise GitHubError(tr("ログインしたトークンが使えません", "the new token does not work"), status=401)
    write_cached_token(token, expires_in)
    return Credentials(token, user, "device")


# --- 送る -----------------------------------------------------------------------------------------


@dataclass
class SubmitResult:
    pr_url: str | None
    reused: bool
    branch: str
    fork: str
    uploaded: list[str]
    skipped: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.pr_url,
            "reused": self.reused,
            "branch": self.branch,
            "fork": self.fork,
            "uploaded": self.uploaded,
            "skipped": self.skipped,
        }


def _repo_name() -> str:
    return UPSTREAM.split("/", 1)[1]


def _is_upstream_fork(body: Any) -> bool:
    if not isinstance(body, dict) or not body.get("fork"):
        return False
    for key in ("source", "parent"):
        r = body.get(key)
        if isinstance(r, dict) and str(r.get("full_name", "")).lower() == UPSTREAM.lower():
            return True
    return False


def ensure_fork(session: Any, token: str, login: str, *, sleep: Callable[[float], None] = time.sleep) -> str:
    """利用者のフォーク(owner/name)。無ければ作り、読めるようになるまで待つ。"""
    status, body = _api(session, token, "GET", f"/repos/{quote(login)}/{_repo_name()}")
    if status == 200 and _is_upstream_fork(body):
        return str(body["full_name"])
    status, body = _api(session, token, "POST", f"/repos/{UPSTREAM}/forks")
    if status not in (200, 201, 202) or not isinstance(body, dict) or "full_name" not in body:
        raise _write_denied(status, body, login)
    fork = str(body["full_name"])
    for _ in range(FORK_WAIT_TRIES):
        st, _b = _api(session, token, "GET", f"/repos/{fork}")
        if st == 200:
            return fork
        sleep(FORK_WAIT_SECONDS)
    raise GitHubError(
        tr(
            "フォークがまだ出来ていません。少し待ってから、もう一度実行してください",
            "the fork is not ready yet; wait a little and run it again",
        )
    )


def find_open_pr(session: Any, token: str, login: str, fork: str, prefix: str) -> tuple[str, str] | None:
    """利用者の開いている share の PR(ブランチの名前が prefix で始まるもの)の (ブランチ, URL)。"""
    status, body = _api(
        session, token, "GET", f"/repos/{UPSTREAM}/pulls", params={"state": "open", "per_page": "100"}
    )
    if status != 200 or not isinstance(body, list):
        return None  # 見つけられないだけなら新しい PR にする(実測を失わない)
    for pr in body:
        head = pr.get("head") if isinstance(pr, dict) else None
        if not isinstance(head, dict):
            continue
        ref = head.get("ref")
        repo = head.get("repo") if isinstance(head.get("repo"), dict) else {}
        owner = (head.get("user") or {}).get("login") if isinstance(head.get("user"), dict) else None
        if (
            isinstance(ref, str)
            and ref.startswith(prefix)
            and str(owner or "").lower() == login.lower()
            and str(repo.get("full_name", fork)).lower() == fork.lower()
            and isinstance(pr.get("html_url"), str)
        ):
            return ref, pr["html_url"]
    return None


def upstream_head(session: Any, token: str) -> str:
    status, body = _api(session, token, "GET", f"/repos/{UPSTREAM}/git/ref/heads/{BASE_BRANCH}")
    sha = (body.get("object") or {}).get("sha") if isinstance(body, dict) else None
    if status != 200 or not isinstance(sha, str):
        raise GitHubError(_message(status, body), status=status)
    return sha


def create_branch(session: Any, token: str, login: str, fork: str, branch: str, sha: str) -> None:
    status, body = _api(
        session, token, "POST", f"/repos/{fork}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha}
    )
    if status in (201, 422):  # 422 はたいてい、もうある(前のやり直し)
        return
    raise _write_denied(status, body, login)


def put_file(
    session: Any, token: str, login: str, fork: str, branch: str, path: str, text: str, message: str
) -> bool:
    """ファイルを置く。置いたら True、もうある(同じ名前)なら False。"""
    url_path = "/".join(quote(p) for p in path.split("/"))
    payload = {
        "message": message,
        "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    status, body = _api(session, token, "PUT", f"/repos/{fork}/contents/{url_path}", payload)
    if status == 422 and "sha" in str((body or {}).get("message", "") if isinstance(body, dict) else ""):
        return False
    if status not in (200, 201):
        raise _write_denied(status, body, login)
    return True


def open_pr(session: Any, token: str, fork: str, branch: str, title: str, body: str) -> str:
    owner = fork.split("/", 1)[0]
    status, resp = _api(
        session,
        token,
        "POST",
        f"/repos/{UPSTREAM}/pulls",
        {"title": title, "head": f"{owner}:{branch}", "base": BASE_BRANCH, "body": body},
    )
    url = resp.get("html_url") if isinstance(resp, dict) else None
    if status not in (200, 201) or not isinstance(url, str):
        raise GitHubError(_message(status, resp), status=status)
    return url


def submit(
    session: Any,
    creds: Credentials,
    files: list[tuple[str, str]],
    *,
    title: str,
    body: str,
    branch: str,
    match_prefix: str,
    log: Callable[[str], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> SubmitResult:
    """files((リポジトリの中のパス, 中身))を PR で送る。開いている share の PR(ブランチが match_prefix で
    始まる)があればそのブランチに足す。無ければ本家の main の先頭から branch を作って PR を開く。
    同じ名前のファイルは飛ばす。新しく置くものが無ければ PR を開かない(pr_url は None)。"""
    say = log or (lambda msg: None)
    token, login_name = creds.token, creds.login
    say(tr("フォークを確かめています", "checking your fork"))
    fork = ensure_fork(session, token, login_name, sleep=sleep)
    found = find_open_pr(session, token, login_name, fork, match_prefix)
    reused = found is not None
    if found is not None:
        branch, pr_url = found
        say(tr(f"開いている PR に足します: {pr_url}", f"adding to your open PR: {pr_url}"))
    else:
        pr_url = None
        create_branch(session, token, login_name, fork, branch, upstream_head(session, token))
    uploaded: list[str] = []
    skipped: list[str] = []
    for path, text in files:
        if put_file(session, token, login_name, fork, branch, path, text, f"measurements: add {path}"):
            uploaded.append(path)
        else:
            skipped.append(path)
    if not reused and uploaded:
        pr_url = open_pr(session, token, fork, branch, title, body)
    return SubmitResult(pr_url, reused, branch, fork, uploaded, skipped)


# --- PR の題と本文 --------------------------------------------------------------------------------


def pr_title(gpu_name: str, count: int) -> str:
    return f"measurements: {gpu_name} ({count})"


def pr_body(subs: list[dict[str, Any]]) -> str:
    """PR の本文: 何を測ったか(モデル・GPU・設定・結果)、genmodelfit の版、genmodelfit share で作った印。
    提出から来る文字は contrib.md_cell で無害にする。個人の情報は入れない(提出の中身だけ)。"""
    lines = [
        f"Measurements sent with `genmodelfit share` (genmodelfit {__version__}).",
        "",
        "| Model | GPU | Settings | Outcome | Peak VRAM (GiB) | Time (s) |",
        "|---|---|---|---|---|---|",
    ]
    for d in subs:
        mm = d.get("measurement") or {}
        gpu = (d.get("system") or {}).get("gpu") or {}
        settings = ", ".join(f"{k}={v}" for k, v in (mm.get("settings") or {}).items()) or "-"
        cells = [
            mm.get("model_id"),
            f"{gpu.get('name')} ({gpu.get('vram_gb')} GiB)",
            settings,
            mm.get("outcome"),
            mm.get("peak_vram_gb"),
            mm.get("seconds"),
        ]
        lines.append("| " + " | ".join(md_cell(str(c) if c is not None else "-") for c in cells) + " |")
    lines += [
        "",
        "Each file passed the privacy checklist and `genmodelfit contrib validate` on the sender's PC;"
        " the Contrib workflow checks them again.",
        "",
        PR_MARKER,
    ]
    return "\n".join(lines) + "\n"
