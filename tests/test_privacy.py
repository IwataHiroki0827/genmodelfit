"""送るもの・送らないもの(Issue #18): ~/.netrc の資格情報を付けない、User-Agent が決まった形。

手元の http.server(127.0.0.1、空いているポート)に、fetch(update)・remote(update --from-release)・
bench(ComfyClient)の HTTP を送り、届いたヘッダを見る。
NETRC には 127.0.0.1 の資格情報を書いた一時ファイルを置く。
素の requests.Session なら Authorization が付くこと(試験が効いていること)も確かめる。
"""

from __future__ import annotations

import http.server
import threading
from typing import ClassVar

import pytest
import requests

from genmodelfit import __version__, remote
from genmodelfit.bench import ComfyClient
from genmodelfit.fetch import USER_AGENT, HttpFetcher, make_session


class _Handler(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, str]]] = []

    def _reply(self) -> None:
        type(self).seen.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
        if self.path.startswith("/redir"):
            self.send_response(302)
            self.send_header("Location", "/final.json")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b'{"ok": true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    do_GET = _reply
    do_HEAD = _reply

    def log_message(self, *a) -> None:  # 試験の出力に混ぜない
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1\nlogin secret-user\npassword secret-pass\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(netrc))
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _Handler.seen = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Handler.seen
    httpd.shutdown()
    httpd.server_close()


def test_plain_requests_would_send_netrc(server):
    """前提の確認: 素の requests は netrc の資格情報を付ける(この試験の仕掛けが効いている)。"""
    base, seen = server
    with requests.Session() as s:
        s.get(base + "/redir").raise_for_status()
    assert all(h.get("authorization", "").startswith("Basic ") for h in seen)


def test_session_sends_no_netrc_and_fixed_user_agent(server):
    base, seen = server
    with make_session() as s:
        s.get(base + "/redir").raise_for_status()  # 転送(rebuild_auth)でも付けない
        s.head(base + "/x").raise_for_status()
    assert [h["path"] for h in seen] == ["/redir", "/final.json", "/x"]
    assert not any("authorization" in h for h in seen)
    assert {h["user-agent"] for h in seen} == {USER_AGENT}
    assert f"genmodelfit/{__version__} (+https://github.com/laxshmate/genmodelfit)" == USER_AGENT


def test_update_fetcher_sends_no_netrc(server):
    base, seen = server
    f = HttpFetcher(sleep=lambda s: None)
    assert f.get_json(base + "/redir") == {"ok": True}
    f.head_status(base + "/x")
    assert seen and not any("authorization" in h for h in seen)
    assert {h["user-agent"] for h in seen} == {USER_AGENT}


def test_from_release_download_sends_no_netrc(server):
    base, seen = server
    assert remote._get(base + "/redir", 5) == b'{"ok": true}'
    assert [h["path"] for h in seen] == ["/redir", "/final.json"]
    assert not any("authorization" in h for h in seen)
    assert {h["user-agent"] for h in seen} == {USER_AGENT}


def test_bench_client_sends_no_netrc(server):
    base, seen = server
    assert ComfyClient(base).get_json("/redir") == {"ok": True}
    assert seen and not any("authorization" in h for h in seen)


def test_proxy_environment_is_still_used(monkeypatch):
    """netrc だけを止め、プロキシ・証明書の環境変数(trust_env)は今までどおり使う。"""
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    with make_session() as s:
        assert s.trust_env is True
        settings = s.merge_environment_settings("https://huggingface.co/api/models", {}, None, None, None)
    assert settings["proxies"].get("https") == "http://proxy.example:3128"


def test_L6e_proxy(monkeypatch):
    test_proxy_environment_is_still_used(monkeypatch)  # プロキシの環境は make_session の決まりどおり使う
