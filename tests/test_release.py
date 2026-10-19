"""署名と、署名つきの一覧の取り込み(--from-release)。ネットワークには触らない。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
import yaml
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from genmodelfit import remote, signing
from genmodelfit.cli import main
from genmodelfit.models import SCHEMA_VERSION

ROOT = Path(__file__).resolve().parent.parent


def _keypair() -> tuple[bytes, bytes]:
    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pub = k.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return priv, pub


def test_roundtrip_and_rejections() -> None:
    priv, pub = _keypair()
    sig = signing.sign(b"hello", priv)
    assert len(sig) == 64
    assert signing.verify(b"hello", sig, pub)
    assert not signing.verify(b"hellO", sig, pub)  # 改ざん
    _, other_pub = _keypair()
    assert not signing.verify(b"hello", sig, other_pub)  # 別の鍵
    assert not signing.verify(b"hello", b"short", pub)  # 壊れた署名
    assert not signing.verify(b"hello", sig, b"not a key")  # 壊れた鍵
    assert signing.decode_sig(signing.encode_sig(sig)) == sig


def test_bad_private_key() -> None:
    with pytest.raises(signing.SigningError):
        signing.sign(b"x", b"not pem")


def test_embedded_public_key_loads() -> None:
    assert signing.load_public_key() is not None
    # 同梱の鍵では、別の鍵の署名は通らない
    priv, _ = _keypair()
    assert not signing.verify(b"x", signing.sign(b"x", priv))


def test_sign_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    priv, pub = _keypair()
    f = tmp_path / "db.json"
    f.write_bytes(b"{}")
    monkeypatch.setenv("GENMODELFIT_SIGNING_KEY", priv.decode())
    assert signing.main(["sign", str(f)]) == 0
    sig = signing.decode_sig((tmp_path / "db.json.sig").read_text())
    assert signing.verify(b"{}", sig, pub)
    monkeypatch.delenv("GENMODELFIT_SIGNING_KEY")
    assert signing.main(["sign", str(f)]) == 1


# --- --from-release ---------------------------------------------------------------------------


class FakeResp:
    def __init__(self, content: bytes, status: int = 200):
        self.content, self.status_code = content, status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    def __init__(self, files: dict[str, bytes] | Exception):
        self.files = files

    def get(self, url: str, **_kw) -> FakeResp:
        if isinstance(self.files, Exception):
            raise self.files
        name = url.rsplit("/", 1)[1]
        return FakeResp(self.files[name]) if name in self.files else FakeResp(b"", 404)


def _db_bytes(generated_at: str = "2026-10-20T00:00:00Z", **over) -> bytes:
    d = {"schema_version": SCHEMA_VERSION, "generated_at": generated_at, "sources": {}, "models": []}
    d.update(over)
    return json.dumps(d).encode()


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """同梱の公開鍵の代わりにテスト用の鍵を差し込み、キャッシュを一時フォルダにする。"""
    priv, pub = _keypair()
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(signing, "embedded_public_key_pem", lambda: pub)
    return priv, tmp_path / "db.json"


def _patch_session(monkeypatch: pytest.MonkeyPatch, files) -> None:
    monkeypatch.setattr(remote.requests, "get", FakeSession(files).get)


def test_from_release_ok(env, monkeypatch, capsys) -> None:
    priv, cache = env
    cache.write_text("old")
    data = _db_bytes()
    _patch_session(
        monkeypatch, {"db.json": data, "db.json.sig": signing.encode_sig(signing.sign(data, priv)).encode()}
    )
    assert main(["update", "--from-release"]) == 0
    assert cache.read_bytes() == data
    assert "更新しました" in capsys.readouterr().out
    assert not list(cache.parent.glob("*.tmp"))


def test_from_release_bad_signature(env, monkeypatch, capsys) -> None:
    priv, cache = env
    cache.write_text("old")
    data = _db_bytes()
    sig = signing.sign(data, priv)
    _patch_session(
        monkeypatch,
        {"db.json": _db_bytes(generated_at="2099-01-01"), "db.json.sig": signing.encode_sig(sig).encode()},
    )
    assert main(["update", "--from-release"]) == 1
    assert cache.read_text() == "old"
    assert "署名が合いません" in capsys.readouterr().err


def test_from_release_bad_schema_signed(env, monkeypatch, capsys) -> None:
    priv, cache = env
    data = _db_bytes(schema_version=SCHEMA_VERSION + 1)
    _patch_session(
        monkeypatch, {"db.json": data, "db.json.sig": signing.encode_sig(signing.sign(data, priv)).encode()}
    )
    assert main(["update", "--from-release"]) == 1
    assert not cache.exists()
    assert "形が正しくありません" in capsys.readouterr().err


def test_from_release_rejects_older(env, monkeypatch) -> None:
    priv, cache = env
    cache.write_bytes(_db_bytes("2026-10-20T00:00:00Z"))
    before = cache.read_bytes()
    data = _db_bytes("2026-10-01T00:00:00Z")
    _patch_session(
        monkeypatch, {"db.json": data, "db.json.sig": signing.encode_sig(signing.sign(data, priv)).encode()}
    )
    assert main(["update", "--from-release"]) == 1
    assert cache.read_bytes() == before


def test_from_release_network_error(env, monkeypatch, capsys) -> None:
    _, cache = env
    cache.write_text("old")
    _patch_session(monkeypatch, requests.ConnectionError("down"))
    assert main(["update", "--from-release"]) == 1
    assert cache.read_text() == "old"
    assert "ダウンロードできませんでした" in capsys.readouterr().err


def test_from_release_missing_sig(env, monkeypatch) -> None:
    _, cache = env
    _patch_session(monkeypatch, {"db.json": _db_bytes()})
    assert main(["update", "--from-release"]) == 1
    assert not cache.exists()


# --- ワークフロー -------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["release.yml", "data.yml", "ci.yml"])
def test_workflow_yaml_parses(name: str) -> None:
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"))
    assert doc["jobs"]


def test_release_workflow_shape() -> None:
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8"))
    pub = doc["jobs"]["publish"]
    assert pub["environment"] == "pypi"
    assert pub["permissions"]["id-token"] == "write"
    assert "refs/tags/v" in pub["if"]
    assert doc[True]["push"]["tags"] == ["v*"]  # YAML 1.1 では on が True になる
