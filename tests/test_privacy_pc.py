"""PC の情報が外に出ないこと(Issue #18。README「Privacy」)。

この PC の情報(GPU・CPU の名前、VRAM・RAM の値、ホスト名、ユーザー名、ホームのパス、GPU の UUID)を
見分けのつく値(SENTINEL)に差し替え、update・update --from-release・bench が送る要求(URL・ヘッダ・本文)と、
bench が保存する実測の JSON に出てこないことを確かめる。本物の値(ホスト名・ユーザー名・ホーム)も見る。
外へのネットワークには出ない(127.0.0.1 の偽の ComfyUI 以外は tests/fixtures/update から答える)。
"""

from __future__ import annotations

import contextlib
import getpass
import io
import json
import os
import platform
import re
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from requests.adapters import HTTPAdapter
from requests.structures import CaseInsensitiveDict

import genmodelfit as gmf
from genmodelfit import __version__, bench, cli, hardware, signing
from genmodelfit.bench import BenchOptions
from genmodelfit.db import bundled_text
from genmodelfit.fetch import HttpFetcher, url_to_relpath
from genmodelfit.models import Gpu
from genmodelfit.update import UpdateFailed, update_database

SENT_GPU = "SENTINEL-GPU-NAME-7f3a"
SENT_CPU = "SENTINEL-CPU-NAME-91c2"
SENT_UUID = "GPU-SENTINEL-UUID-5d8e"
SENT_HOST = "SENTINEL-HOST-0b6d"
SENT_USER = "sentineluser4e1f"
SENT_VRAM = 7.3141  # GiB
SENT_RAM = 31.4159  # GiB
GIB = 1024**3
FIXTURES = Path(__file__).parent / "fixtures" / "update"


def _real_identity() -> list[str]:
    vals = [platform.node(), getpass.getuser(), str(Path.home()), os.environ.get("COMPUTERNAME", "")]
    return [v for v in vals if len(v) >= 4]


@pytest.fixture
def sentinel_pc(tmp_path, monkeypatch):
    """この PC の情報を見分けのつく値にする。送ってはいけない文字列の一覧を返す。"""
    real = _real_identity()
    home = tmp_path / SENT_USER / "home"
    home.mkdir(parents=True)
    gpu = Gpu(
        name=SENT_GPU,
        vendor="nvidia",
        vram_total_gb=SENT_VRAM,
        vram_free_gb=SENT_VRAM,
        driver="999.42",
        backend="cuda",
    )
    monkeypatch.setattr(hardware, "detect_gpus", lambda: ([gpu], []))
    monkeypatch.setattr(hardware, "_cpu_name", lambda: SENT_CPU)
    vm = SimpleNamespace(total=int(SENT_RAM * GIB), available=int(SENT_RAM * GIB))
    monkeypatch.setattr(hardware.psutil, "virtual_memory", lambda: vm)
    monkeypatch.setattr(platform, "node", lambda: SENT_HOST)
    monkeypatch.setattr(socket, "gethostname", lambda: SENT_HOST)
    monkeypatch.setattr(getpass, "getuser", lambda: SENT_USER)
    for k in ("USERNAME", "USER", "LOGNAME"):
        monkeypatch.setenv(k, SENT_USER)
    monkeypatch.setenv("COMPUTERNAME", SENT_HOST)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(home / "cache"))
    # bench が nvidia-smi から読む GPU。名前は偽の ComfyUI(tests/test_bench.py)が返す GPU と合わせる
    gpu0 = {"index": 0, "uuid": SENT_UUID, "name": "NVIDIA GeForce RTX 3070", "driver": "999.42"}
    monkeypatch.setattr(bench, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(bench, "smi_gpus", lambda: [dict(gpu0)])
    monkeypatch.setattr(bench, "smi_compute_apps", lambda: [])
    monkeypatch.setattr(bench, "gpu_used_now", lambda gpu_id: (1024.0, 0.0))
    monkeypatch.setattr(bench, "find_server_process", lambda url: None)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)
    # 前提: 差し替えが効いている(この PC の判定はこの値で行われる)
    pc = gmf.system()
    assert pc.cpu == SENT_CPU and pc.gpus[0].name == SENT_GPU
    forbidden = [
        SENT_GPU,
        SENT_CPU,
        SENT_UUID,
        SENT_HOST,
        SENT_USER,
        str(SENT_VRAM),
        str(SENT_RAM),
        str(int(SENT_RAM * GIB)),
        str(home),
        str(home).replace("\\", "/"),
        *real,
    ]
    return forbidden, home


@pytest.fixture
def outbound(monkeypatch):
    """出ていく HTTP をすべて記録する。

    127.0.0.1(偽の ComfyUI)へはそのまま送り、ほかは fixtures から答える。
    """
    real_send = HTTPAdapter.send
    log: list[str] = []
    files: dict[str, bytes] = {}

    def send(self, request, **kw):
        body = request.body or b""
        if isinstance(body, str):
            body = body.encode("utf-8")
        head = "\n".join(f"{k}: {v}" for k, v in request.headers.items())
        log.append(f"{request.method} {request.url}\n{head}\n{body.decode('utf-8', 'replace')}")
        if request.url.startswith("http://127.0.0.1"):
            return real_send(self, request, **kw)
        resp = requests.Response()
        resp.request = request
        resp.url = request.url
        name = request.url.rsplit("/", 1)[-1]
        p = FIXTURES / url_to_relpath(request.url)
        data = files.get(name) or (p.read_bytes() if p.exists() else None)
        if request.method == "HEAD":
            resp.status_code, data = 200, b""
        elif data is None:
            resp.status_code, data = 404, b"{}"
        else:
            resp.status_code = 200
        resp.headers = CaseInsensitiveDict({"Content-Length": str(len(data))})
        resp.raw = io.BytesIO(data)
        return resp

    monkeypatch.setattr(HTTPAdapter, "send", send)
    return log, files


def _assert_clean(log: list[str], forbidden: list[str]) -> None:
    assert log, "要求が 1 つも記録されていない"
    for entry in log:
        for word in forbidden:
            assert word not in entry, f"送った要求に PC の情報 {word!r} が入っている:\n{entry[:500]}"


def test_update_from_release_sends_no_pc_info(sentinel_pc, outbound, monkeypatch, capsys):
    forbidden, _ = sentinel_pc
    log, files = outbound
    k = Ed25519PrivateKey.generate()
    priv = k.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pub = k.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    monkeypatch.setattr(signing, "embedded_public_keys_pem", lambda: [pub])
    data = bundled_text("db.json").encode("utf-8")
    files["db.json"] = data
    files["db.json.sig"] = signing.encode_sig(signing.sign(data, priv)).encode()
    assert cli.main(["system", "--json"]) == 0  # PC を調べたあとで取り込む
    assert cli.main(["update", "--from-release", "--json"]) == 0
    capsys.readouterr()
    base = "https://github.com/laxshmate/genmodelfit/releases/download/data-latest/"
    assert [e.split("\n", 1)[0] for e in log] == [f"GET {base}db.json", f"GET {base}db.json.sig"]
    _assert_clean(log, forbidden)


def test_update_sends_no_pc_info(sentinel_pc, outbound, tmp_path):
    forbidden, _ = sentinel_pc
    log, _ = outbound
    gmf.recommend(limit=3)
    # 記録に無い URL は 404 にしている。失敗しても、送った要求だけを見る
    with contextlib.suppress(UpdateFailed):
        update_database(HttpFetcher(sleep=lambda s: None), tmp_path / "db.json", allow_partial=True)
    assert len(log) > 10
    assert all(e.startswith(("GET https://", "HEAD https://")) for e in log)
    _assert_clean(log, forbidden)


def test_bench_sends_no_pc_info_and_saves_none(sentinel_pc, outbound):
    from test_bench import FakeComfy

    forbidden, _ = sentinel_pc
    log, _ = outbound
    with FakeComfy() as fake:
        opts = BenchOptions(model_id="image_sdxl_simple", url=fake.url, interval=0.01)
        res = bench.run(opts, log=lambda m: None)
    assert any("/prompt" in e.split("\n", 1)[0] for e in log)
    _assert_clean(log, forbidden)
    # 保存する実測: 測った GPU の名前(ComfyUI が返したもの)と driver は実測の中身として入る
    # (送るのは利用者が自分で PR を出すときだけ)。ホスト名・ユーザー名・パス・UUID・この PC の RAM は入らない
    m = res.measurement
    assert m.gpu_name == "NVIDIA GeForce RTX 3070" and m.driver == "999.42"
    path = bench.default_output_path(m)
    bench.save(m, path)
    text = path.read_text(encoding="utf-8")
    for word in forbidden:
        assert word not in text, word
    absolute = re.compile(r"[A-Za-z]:(\\\\|/)|/(home|Users|tmp|root)/")  # JSON の中の \ は \\ になる
    assert absolute.search(json.dumps({"p": str(path)})) and absolute.search('"/home/x/"')  # 検査が効く
    assert not absolute.search(text)  # 絶対パスが無い
    assert m.source == f"genmodelfit bench {__version__}"  # --source が無ければ道具の名前だけ


def test_download_sends_no_pc_info(sentinel_pc, outbound, tmp_path):
    """download(#31): 送るのは一覧の URL への GET(User-Agent・Range)だけ。
    PC の情報も置き場所のパスも送らない。"""
    import hashlib

    from genmodelfit.db import Database
    from genmodelfit.models import ModelEntry, ModelFile, Template

    forbidden, home = sentinel_pc
    log, files = outbound
    body = b"tiny-weights" * 1000
    files["tiny_privacy.safetensors"] = body
    f = ModelFile(
        name="tiny_privacy.safetensors",
        url="https://huggingface.co/example/tiny/resolve/main/tiny_privacy.safetensors",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
        directory="vae",
        license="mit",
        gated=False,
        role="vae",
    )
    m = ModelEntry(
        id="test_privacy", name="Privacy", modality="image", files=[f], template=Template(name="x")
    )
    comfy = home / "ComfyUI"  # 置き場所のパスにユーザー名が入る
    (comfy / "models").mkdir(parents=True)
    res = gmf.download(
        "test_privacy", yes=True, db=Database(models=[m]), comfyui_dir=comfy
    )  # PC は sentinel で調べる
    assert (comfy / "models" / "vae" / "tiny_privacy.safetensors").read_bytes() == body
    assert res.downloaded == ["vae/tiny_privacy.safetensors"]
    assert [e.split("\n", 1)[0] for e in log] == [f"GET {f.url}"]
    _assert_clean(log, forbidden)
