"""update_from_release の公開の引数 session / state_dir と、Fit.gpu_memory_required_gb(Issue #58、0.7.1)。

手元の http.server(127.0.0.1)に署名つきの一覧を置き、呼び出し側のセッションで取り込む。
NETRC には 127.0.0.1 の資格情報を書いた一時ファイルを置き、Authorization が付かないことを確かめる。
"""

from __future__ import annotations

import http.server
import inspect
import io
import json
import os
import threading
from typing import Any, ClassVar

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import genmodelfit as gmf
from genmodelfit import remote, signing
from genmodelfit.db import bundled_text, parse_db, parse_measurements
from genmodelfit.fetch import USER_AGENT
from genmodelfit.models import SCHEMA_VERSION


def _db_bytes(generated_at: str) -> bytes:
    d = {"schema_version": SCHEMA_VERSION, "generated_at": generated_at, "sources": {}, "models": []}
    return json.dumps(d).encode()


DB = _db_bytes("2026-10-20T00:00:00Z")


def _keypair() -> tuple[bytes, bytes]:
    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pub = k.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return priv, pub


def _sig(data: bytes, priv: bytes) -> bytes:
    return signing.encode_sig(signing.sign(data, priv)).encode()


class _Handler(http.server.BaseHTTPRequestHandler):
    seen: ClassVar[list[dict[str, str]]] = []
    files: ClassVar[dict[str, bytes]] = {}

    def do_GET(self) -> None:
        type(self).seen.append({"path": self.path, **{k.lower(): v for k, v in self.headers.items()}})
        if self.path.startswith("/redir/"):
            self.send_response(302)
            self.send_header("Location", "/" + self.path.split("/", 2)[2])
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = type(self).files.get(self.path.lstrip("/"))
        if body is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a) -> None:
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    """署名つきの一覧を配る手元のサーバー。→ (base_url, seen, priv)。既定のキャッシュは tmp_path/cache。"""
    priv, pub = _keypair()
    monkeypatch.setattr(signing, "embedded_public_keys_pem", lambda: [pub])
    monkeypatch.setattr(remote, "RETRY_DELAY", 0)
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1\nlogin secret-user\npassword secret-pass\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(netrc))
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    _Handler.seen = []
    _Handler.files = {"db.json": DB, "db.json.sig": _sig(DB, priv)}
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", _Handler.seen, priv
    httpd.shutdown()
    httpd.server_close()


class RecordingSession(requests.Session):
    """呼び出しを記録するセッション(アプリが渡す CA バンドル付きのセッションの代わり)。"""

    def __init__(self) -> None:
        super().__init__()
        self.gets: list[dict[str, Any]] = []
        self.verifies: list[Any] = []

    def get(self, url, **kw):
        self.gets.append({"url": url, **kw})
        return super().get(url, **kw)

    def send(self, request, **kw):
        self.verifies.append(kw.get("verify"))
        return super().send(request, **kw)


def test_plain_session_would_send_netrc(server):
    """前提の確認: 素のセッションは netrc の資格情報を付ける(この試験の仕掛けが効いている)。"""
    base, seen, _ = server
    with requests.Session() as s:
        s.get(base + "/db.json").raise_for_status()
    assert seen[0]["authorization"].startswith("Basic ")


def test_caller_session_is_used_with_fixed_ua_and_no_netrc(server, tmp_path):
    base, seen, _ = server
    s = RecordingSession()
    s.headers["X-Extra"] = "keep"
    # 転送(GitHub の Release は別の場所へ転送する)でも netrc を読み直さない
    db = remote.update_from_release(base + "/redir", session=s, dest=tmp_path / "out.json")
    assert db.models == []
    want = ["/redir/db.json", "/db.json", "/redir/db.json.sig", "/db.json.sig"]
    assert [g["url"] for g in s.gets] == [base + p for p in want]  # すべてこのセッションで取る
    assert [h["path"] for h in seen] == want
    assert not any("authorization" in h for h in seen)
    assert {h["user-agent"] for h in seen} == {USER_AGENT}
    assert {h["x-extra"] for h in seen} == {"keep"}  # セッションのほかのヘッダーは残る
    assert (
        s.headers["X-Extra"] == "keep" and s.headers["User-Agent"] != USER_AGENT
    )  # セッションは書き換えない
    assert s.auth is None


def test_session_auth_is_the_callers_choice_but_not_sent_to_other_hosts(server, tmp_path):
    base, seen, _ = server
    s = RecordingSession()
    s.auth = ("me", "pw")
    remote.update_from_release(base, session=s, dest=tmp_path / "out.json")
    assert all(h["authorization"].startswith("Basic ") for h in seen)

    # 別のホストへ転送されたら、呼び出し側の認証もヘッダーも付けない
    calls: list[dict[str, Any]] = []

    class Redirecting(requests.Session):
        def get(self, url, **kw):
            calls.append({"url": url, **kw})
            r = requests.Response()
            r.raw = io.BytesIO()
            r.status_code = 302 if len(calls) == 1 else 200
            if len(calls) == 1:
                r.headers["Location"] = "http://other.example/db.json"
            return r

    s2 = Redirecting()
    s2.auth = ("me", "pw")
    remote._session_get(s2, "http://first.example/db.json", 5)
    assert (
        calls[0]["auth"] == ("me", "pw") and "Authorization" not in calls[0]["headers"]
    )  # session.auth を使う
    assert isinstance(calls[1]["auth"], remote._NoAuth) and calls[1]["headers"]["Authorization"] is None


def test_session_enforces_max_bytes(server, monkeypatch, tmp_path):
    base, _, _ = server
    monkeypatch.setattr(remote, "MAX_BYTES", 10)
    with pytest.raises(remote.RemoteError, match="大きすぎます"):
        remote.update_from_release(base, session=RecordingSession(), dest=tmp_path / "out.json")
    assert not (tmp_path / "out.json").exists()


def test_session_tampered_data_is_refused_and_nothing_is_written(server, tmp_path):
    base, _, _ = server
    _Handler.files["db.json"] = DB.replace(b"2026-10-20", b"2026-10-21")  # 署名と合わない
    state = tmp_path / "state"
    dest = tmp_path / "data" / "db.json"
    with pytest.raises(remote.RemoteError, match="署名が合いません"):
        remote.update_from_release(base, session=RecordingSession(), dest=dest, state_dir=state)
    assert not dest.exists() and not state.exists() and not (tmp_path / "cache").exists()


def test_state_dir_gets_the_state_and_cache_and_env_are_untouched(server, tmp_path):
    base, _, priv = server
    env_before = dict(os.environ)
    state = tmp_path / "state"
    dest = tmp_path / "data" / "db.json"
    remote.update_from_release(base, session=RecordingSession(), dest=dest, state_dir=state)
    assert dest.read_bytes() == DB
    rec = json.loads((state / remote.STATE_NAME).read_text(encoding="utf-8"))
    assert rec["generated_at"] == "2026-10-20T00:00:00+00:00"
    assert not (tmp_path / "cache").exists()  # 既定のキャッシュには触らない
    assert dict(os.environ) == env_before
    # 古い配布は state_dir の記録で断る(キャッシュの記録は見ない)
    older = _db_bytes("2026-10-19T00:00:00Z")
    _Handler.files["db.json"] = older
    _Handler.files["db.json.sig"] = _sig(older, priv)
    with pytest.raises(remote.RemoteError, match="古い一覧"):
        remote.update_from_release(base, session=RecordingSession(), dest=dest, state_dir=state)
    remote.update_from_release(base, session=RecordingSession(), dest=dest, state_dir=state, force=True)
    assert dest.read_bytes() == older


def test_http_get_monkeypatch_still_works_without_session(monkeypatch, tmp_path):
    """アプリが今やっている remote._http_get の差し替えは、session を渡さないあいだは効く。"""
    priv, pub = _keypair()
    monkeypatch.setattr(signing, "embedded_public_keys_pem", lambda: [pub])
    sig = _sig(DB, priv)
    urls: list[str] = []

    class R:
        status_code = 200
        headers: ClassVar[dict[str, str]] = {}

        def __init__(self, c: bytes) -> None:
            self.c = c

        def raise_for_status(self) -> None:
            pass

        def iter_content(self, chunk_size=1):
            yield self.c

        def close(self) -> None:
            pass

    def fake(url, **kw):
        urls.append(url)
        return R(DB if url.endswith("db.json") else sig)

    monkeypatch.setattr(remote, "_http_get", fake)
    st = tmp_path / "st"
    remote.update_from_release("https://example.invalid/r", dest=tmp_path / "db.json", state_dir=st)
    assert urls == ["https://example.invalid/r/db.json", "https://example.invalid/r/db.json.sig"]
    assert (st / remote.STATE_NAME).exists()


def test_app_usage_ca_bundle_session_and_data_folder(server, tmp_path):
    """アプリの使い方: CA バンドルを verify に付けた素のセッション + データフォルダ(dest・state_dir)。"""
    base, seen, _ = server
    ca = tmp_path / "cache" / "ca-bundle.pem"
    data = tmp_path / "data folder"
    s = RecordingSession()
    s.verify = str(ca)
    db = gmf.update_from_release(base, session=s, dest=data / "db.json", state_dir=data)
    assert db.models == [] and (data / "db.json").read_bytes() == DB
    assert (data / remote.STATE_NAME).exists()
    assert s.verifies and all(v == str(ca) for v in s.verifies)  # セッションの verify がそのまま使われる
    assert not any("authorization" in h for h in seen)
    assert gmf.RemoteError is remote.RemoteError


def test_public_signature():
    assert "update_from_release" in gmf.__all__ and "RemoteError" in gmf.__all__
    p = inspect.signature(gmf.update_from_release).parameters
    assert list(p) == ["base_url", "timeout", "dest", "force", "session", "state_dir"]
    assert p["session"].default is None and p["state_dir"].default is None


# --- Fit.gpu_memory_required_gb ---------------------------------------------------------------


@pytest.fixture(scope="module")
def app_db():
    db = parse_db(json.loads(bundled_text("db.json")), "bundled")
    db.measurements = parse_measurements(json.loads(bundled_text("measurements.json")))
    return db


def _app_style(db, mid, name, vram, driver="560.94"):
    return gmf.judge_model(
        mid, gpu_name=name, vram_gb=vram, driver=driver, ram_gb=1024.0, gpu_usable=True, windows=True, db=db
    )


def test_gpu_memory_required_for_cpu_run_mode(app_db):
    # 3 GB の GPU では SDXL は載らず CPU で動く(4 GB なら GPU+RAM の載せ替えで動く)
    fit = _app_style(app_db, "image_sdxl_simple", "NVIDIA GeForce GTX 1650", 3.0)
    d = fit.to_dict()
    assert d["run_mode"] == "cpu" and d["memory_kind"] == "ram"
    assert d["gpu_memory_required_gb"] is not None and d["gpu_memory_required_gb"] > 3.0
    assert d["memory_required_gb"] > d["gpu_memory_required_gb"]  # RAM の量
    assert d["gpu_memory_required_gb"] == round(fit.estimate.vram_offload_gb, 2)
    four = _app_style(app_db, "image_sdxl_simple", "NVIDIA GeForce GTX 1650", 4.0).to_dict()
    assert four["run_mode"] == "offload" and four["gpu_memory_required_gb"] == four["memory_required_gb"]


def test_gpu_memory_required_equals_vram_need_on_gpu(app_db):
    from genmodelfit.recommend import vram_need

    seen_gpu_mode = 0
    for name, vram in (("NVIDIA GeForce RTX 4090", 24.0), ("NVIDIA GeForce RTX 3060", 12.0)):
        for mid in ("image_sdxl_simple", "image_z_image_turbo", "video_wan2_2_5B_ti2v"):
            fit = _app_style(app_db, mid, name, vram)
            d = fit.to_dict()
            need = vram_need(fit.estimate, fit.system)
            if d["run_mode"] != "cpu":
                seen_gpu_mode += 1
                assert d["gpu_memory_required_gb"] == d["memory_required_gb"]
                assert d["gpu_memory_required_gb"] == (round(need, 2) if need is not None else None)
    assert seen_gpu_mode >= 4


def test_gpu_memory_required_when_too_tight_and_without_gpu(app_db):
    d = _app_style(app_db, "video_wan2_2_5B_ti2v", "NVIDIA GeForce GTX 1650", 4.0).to_dict()
    assert d["fit_level"] == "too_tight" and d["gpu_memory_required_gb"] == d["memory_required_gb"]
    assert d["gpu_memory_required_gb"] > 4.0
    nogpu = gmf.judge_model("image_sdxl_simple", cpu_only=True, ram_gb=64, db=app_db).to_dict()
    assert "gpu_memory_required_gb" in nogpu and nogpu["gpu_memory_required_gb"] is None


# --- 点検での指摘(auth の空の値・verify の環境変数・不正な転送先・共有メモリ)----------------


class _Redirecting(requests.Session):
    """1 回目は別のホストへ転送し、2 回目で 200 を返す。get の引数を記録する。"""

    def __init__(self, first_location: str | None) -> None:
        super().__init__()
        self.calls: list[dict[str, Any]] = []
        self.first_location = first_location

    def get(self, url, **kw):
        self.calls.append({"url": url, **kw})
        r = requests.Response()
        r.raw = io.BytesIO()
        if len(self.calls) == 1 and self.first_location:
            r.status_code = 302
            r.headers["Location"] = self.first_location
        else:
            r.status_code = 200
        return r


@pytest.mark.parametrize("auth", [None, (), "", ("me", "pw")])
@pytest.mark.parametrize("cross_host", [False, True])
def test_session_auth_values_never_fall_back_to_netrc(auth, cross_host):
    """auth が空(None・()・"")でも、requests に netrc を読ませない(要求の auth が None にならない)。"""
    loc = "http://other.example/db.json" if cross_host else "/db.json"
    s = _Redirecting(loc)
    s.auth = auth
    remote._session_get(s, "http://first.example/db.json", 5)
    for i, c in enumerate(s.calls):
        assert c["auth"] is not None
        if auth and not (cross_host and i == 1):
            assert c["auth"] == auth  # 呼び出し側の認証は、同じホストにだけ使う
        else:
            assert isinstance(c["auth"], remote._NoAuth)
    if cross_host:
        assert s.calls[1]["headers"]["Authorization"] is None


@pytest.mark.parametrize("auth", [(), ""])
def test_empty_session_auth_sends_no_netrc_end_to_end(server, tmp_path, auth):
    base, seen, _ = server
    s = RecordingSession()
    s.auth = auth
    remote.update_from_release(base + "/redir", session=s, dest=tmp_path / "out.json")
    assert len(seen) == 4 and not any("authorization" in h for h in seen)


def test_session_verify_wins_over_ca_bundle_environment(server, tmp_path, monkeypatch):
    """環境変数 REQUESTS_CA_BUNDLE / CURL_CA_BUNDLE より、セッションの verify を優先する(requests#3829)。"""
    base, _, _ = server
    ca = tmp_path / "app-ca.pem"
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(tmp_path / "env-ca.pem"))
    monkeypatch.setenv("CURL_CA_BUNDLE", str(tmp_path / "env-curl.pem"))
    s = RecordingSession()
    s.verify = str(ca)
    remote.update_from_release(base, session=s, dest=tmp_path / "out.json")
    assert s.verifies and all(v == str(ca) for v in s.verifies)


def test_session_proxies_win_over_environment(monkeypatch):
    s = _Redirecting(None)
    s.proxies["https"] = "http://app-proxy.example:3128"
    remote._session_get(s, "https://first.example/db.json", 5)
    assert s.calls[0]["proxies"] == {"https": "http://app-proxy.example:3128"}
    assert s.proxies == {"https": "http://app-proxy.example:3128"}  # セッションは書き換えない


@pytest.mark.parametrize("where", ["first", "redirect"])
def test_invalid_port_is_a_remote_error(tmp_path, where):
    bad = "https://x.example:99999/db.json"
    s = _Redirecting(bad if where == "redirect" else None)
    with pytest.raises(remote.RemoteError, match="URL が正しくありません"):
        remote._session_get(s, bad if where == "first" else "https://first.example/db.json", 5)
    with pytest.raises(remote.RemoteError):
        remote.update_from_release(
            "https://x.example:99999",
            session=_Redirecting(None),
            dest=tmp_path / "o.json",
            state_dir=tmp_path / "s",
        )
    assert not (tmp_path / "o.json").exists() and not (tmp_path / "s").exists()


def test_gpu_memory_required_cpu_equals_too_tight_rule_on_shared_memory(app_db):
    """共有メモリの GPU(APU)でも、cpu のときは vram_offload_gb(too_tight と同じ規則)。"""
    from genmodelfit.models import Gpu, SystemInfo
    from genmodelfit.recommend import judge

    found = 0
    for vram in (0.5, 1.0, 2.0):
        g = Gpu(
            name="AMD Radeon(TM) 8060S Graphics",
            vendor="amd",
            vram_total_gb=vram,
            backend="rocm",
            compute_dtype="fp16",
            bf16=True,
            fp8=False,
            memory="shared",
            gfx_target="gfx1151",
        )
        system = SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g])
        for mid in ("image_sdxl_simple", "image_z_image_turbo", "video_wan2_2_5B_ti2v"):
            f = judge(app_db.get(mid), system, app_db)
            d = f.to_dict()
            if d["run_mode"] in ("cpu", "none"):
                found += 1
                assert d["gpu_memory_required_gb"] == round(f.estimate.vram_offload_gb, 2), (vram, mid)
    assert found >= 3
