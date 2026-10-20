"""download(#31): 計画・同意・sha256・続きから・置き場所・転送先の約束。

手元の http.server(127.0.0.1、空いているポート)から落とす。許す転送先(downloader.HOST_POLICY)は
試験のあいだだけ 127.0.0.1 の http にする(既定の約束は test_host_policy_default で見る)。
外へのネットワークには出ない(実際の Hugging Face から落とす試験は test_network_download。既定では飛ばす)。
"""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import threading
from pathlib import Path
from typing import ClassVar

import pytest
import requests

import genmodelfit as gmf
from genmodelfit import cli, downloader
from genmodelfit.db import Database
from genmodelfit.downloader import (
    ChecksumMismatch,
    DiskFull,
    DownloadDeclined,
    DownloadError,
    DownloadRefused,
    HostNotAllowed,
    HostPolicy,
    safe_relpath,
)
from genmodelfit.fetch import USER_AGENT
from genmodelfit.models import Gpu, ModelEntry, ModelFile, SystemInfo, Template

MIB = 1024 * 1024
BODY_A = bytes(range(256)) * (3 * MIB // 256 + 17)  # 3 MiB と少し(1 MiB ずつ受け取るので途中で切れる)
BODY_B = b"vae-weights-" * 5000


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def pc(vram: float = 8.0) -> SystemInfo:
    gpus = [Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")]
    return SystemInfo(os="test", cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=gpus)


class _Handler(http.server.BaseHTTPRequestHandler):
    files: ClassVar[dict[str, bytes]] = {}
    seen: ClassVar[list[dict[str, str]]] = []
    range_ok: ClassVar[bool] = True
    cut_after: ClassVar[int | None] = None  # 最初の GET をこのバイト数で切る
    status_once: ClassVar[list[tuple[int, dict[str, str]]]] = []
    port: ClassVar[int] = 0

    def do_GET(self) -> None:
        cls = type(self)
        cls.seen.append(
            {"method": "GET", "path": self.path, **{k.lower(): v for k, v in self.headers.items()}}
        )
        if cls.status_once:
            code, headers = cls.status_once.pop(0)
            self.send_response(code)
            for k, v in headers.items():
                self.send_header(k, v)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/r/"):  # 同じホストの中の転送(許す)
            self._redirect("/f/" + self.path[3:])
            return
        if self.path.startswith("/evil/"):  # 別のホストへの転送(許さない)
            self._redirect(f"http://localhost:{cls.port}/f/" + self.path[6:])
            return
        name = self.path.rsplit("/", 1)[-1]
        data = cls.files.get(name)
        if data is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        rng = self.headers.get("Range")
        start = 0
        if rng and cls.range_ok:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if cls.cut_after is not None and not rng:
            n, cls.cut_after = cls.cut_after, None
            self.wfile.write(body[:n])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(body)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *a) -> None:
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    netrc = tmp_path / "netrc"
    netrc.write_text("machine 127.0.0.1\nlogin secret-user\npassword secret-pass\n", encoding="utf-8")
    monkeypatch.setenv("NETRC", str(netrc))
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))
    _Handler.files = {"tiny_unet.safetensors": BODY_A, "tiny_vae.safetensors": BODY_B}
    _Handler.seen = []
    _Handler.range_ok = True
    _Handler.cut_after = None
    _Handler.status_once = []
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    _Handler.port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    monkeypatch.setattr(
        downloader,
        "HOST_POLICY",
        HostPolicy(
            start_hosts=frozenset({"127.0.0.1"}),
            hosts=frozenset({"127.0.0.1"}),
            suffixes=(),
            schemes=("http",),
            any_port=True,
        ),
    )
    yield f"http://127.0.0.1:{_Handler.port}"
    httpd.shutdown()
    httpd.server_close()


def _file(base: str, name: str, body: bytes, directory: str | None, path: str = "f", **kw) -> ModelFile:
    d = {
        "name": name,
        "url": f"{base}/{path}/{name.rsplit('/', 1)[-1]}",
        "sha256": sha(body),
        "size_bytes": len(body),
        "directory": directory,
        "license": "mit",
        "license_url": "https://example.invalid/LICENSE",
        "gated": False,
        "role": "main",
    }
    d.update(kw)
    return ModelFile(**d)


def make_db(base: str, *, files: list[ModelFile] | None = None, content: str = "sfw") -> Database:
    files = files or [
        _file(base, "sub/tiny_unet.safetensors", BODY_A, "diffusion_models"),
        _file(base, "tiny_vae.safetensors", BODY_B, "vae", path="r", role="vae"),  # 転送を 1 回はさむ
    ]
    m = ModelEntry(
        id="test_tiny",
        name="Tiny Test",
        modality="image",
        tasks=["t2i"],
        files=files,
        template=Template(name="test_tiny", username="ComfyUI"),
        precision="fp16",
        license="mit",
        content=content,
    )
    return Database(models=[m])


@pytest.fixture
def comfy(tmp_path) -> Path:
    d = tmp_path / "ComfyUI"
    (d / "models").mkdir(parents=True)
    return d


def plan(db: Database, comfy: Path, **kw):
    return gmf.plan_download("test_tiny", db=db, system_info=pc(), comfyui_dir=comfy, **kw)


def _gets(path_part: str) -> list[dict[str, str]]:
    return [s for s in _Handler.seen if path_part in s["path"]]


# --- 正常 ---------------------------------------------------------------------------------------


def test_download_places_files_by_comfyui_folder(server, comfy):
    db = make_db(server)
    p = plan(db, comfy)
    assert p.ok and p.layout == "comfyui" and Path(p.root) == comfy / "models"
    assert [f.relpath for f in p.files] == [
        "diffusion_models/sub/tiny_unet.safetensors",
        "vae/tiny_vae.safetensors",
    ]
    assert p.download_bytes == len(BODY_A) + len(BODY_B)
    events = []
    res = gmf.download(p, yes=True, progress=events.append)
    a = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors"
    b = comfy / "models" / "vae" / "tiny_vae.safetensors"
    assert a.read_bytes() == BODY_A and b.read_bytes() == BODY_B
    assert not list(comfy.rglob("*.part"))
    assert res.downloaded == [f.relpath for f in p.files] and res.bytes_downloaded == p.download_bytes
    assert {e.phase for e in events} >= {"download", "verify", "done"}
    assert events[-1].done_bytes == p.total_bytes
    # 送ったもの: User-Agent だけ(~/.netrc の資格情報もトークンも付けない)
    for s in _Handler.seen:
        assert s["user-agent"] == USER_AGENT
        assert "authorization" not in s and "cookie" not in s
    # 2 回目は既にあるので通信しない
    _Handler.seen = []
    p2 = plan(db, comfy)
    assert [f.status for f in p2.files] == ["present", "present"] and p2.download_bytes == 0
    res2 = gmf.download(p2, yes=True)
    assert res2.skipped == res.downloaded and _Handler.seen == []


def test_output_dir_uses_same_layout(server, tmp_path):
    db = make_db(server)
    out = tmp_path / "models out"
    res = gmf.download("test_tiny", yes=True, db=db, system_info=pc(), output_dir=out)
    assert (out / "vae" / "tiny_vae.safetensors").read_bytes() == BODY_B
    assert res.plan.layout == "output"


def test_comfyui_dir_must_look_like_comfyui(tmp_path):
    with pytest.raises(DownloadRefused, match="ComfyUI"):
        gmf.plan_download(
            "test_tiny", db=make_db("https://huggingface.co"), system_info=pc(), comfyui_dir=tmp_path
        )
    with pytest.raises(ValueError):
        gmf.plan_download("x", output_dir=tmp_path, comfyui_dir=tmp_path)


# --- 中身の確かめ -------------------------------------------------------------------------------


def test_sha256_mismatch_deletes_file(server, comfy):
    db = make_db(server)
    db.models[0].files[1].sha256 = "0" * 64
    p = plan(db, comfy)
    with pytest.raises(ChecksumMismatch, match="sha256"):
        gmf.download(p, yes=True)
    assert not (comfy / "models" / "vae" / "tiny_vae.safetensors").exists()
    assert not list(comfy.rglob("*.part"))


def test_bigger_than_listed_is_stopped(server, comfy):
    db = make_db(server)
    db.models[0].files[1].size_bytes = len(BODY_B) - 10
    with pytest.raises(DownloadError, match="大きい"):
        gmf.download(plan(db, comfy), yes=True)
    assert not (comfy / "models" / "vae" / "tiny_vae.safetensors").exists()


# --- 途中で切れる・続きから ------------------------------------------------------------------------


def test_cut_connection_resumes_with_range(server, comfy):
    _Handler.cut_after = MIB + 123
    db = make_db(server)
    res = gmf.download(plan(db, comfy), yes=True, sleep=lambda s: None)
    a = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors"
    assert a.read_bytes() == BODY_A
    ranges = [s.get("range") for s in _gets("tiny_unet")]
    # 続きは、書けた分(切れた最後の断片は捨てることがある)の次から
    assert ranges[0] is None and ranges[1].startswith("bytes=")
    assert 0 < int(ranges[1][6:-1]) <= MIB + 123
    assert res.resumed == ["diffusion_models/sub/tiny_unet.safetensors"]


def test_interrupt_keeps_part_and_next_run_resumes(server, comfy):
    db = make_db(server)

    def stop(ev):
        if ev.phase == "download" and ev.file_bytes >= MIB:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        gmf.download(plan(db, comfy), yes=True, progress=stop)
    part = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors.part"
    have = part.stat().st_size
    assert 0 < have < len(BODY_A)
    p2 = plan(db, comfy)
    assert p2.files[0].status == "resume" and p2.files[0].part_bytes == have
    assert p2.download_bytes == len(BODY_A) - have + len(BODY_B)
    _Handler.seen = []
    res = gmf.download(p2, yes=True)
    assert (comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors").read_bytes() == BODY_A
    assert _gets("tiny_unet")[0]["range"] == f"bytes={have}-"
    assert res.resumed and not part.exists()


def test_server_without_range_restarts(server, comfy):
    _Handler.range_ok = False
    db = make_db(server)
    part = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(BODY_A[: 2 * MIB])
    res = gmf.download(plan(db, comfy), yes=True)
    assert (comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors").read_bytes() == BODY_A
    assert res.restarted == ["diffusion_models/sub/tiny_unet.safetensors"] and not res.resumed


def test_corrupt_part_fails_sha_and_is_deleted(server, comfy):
    db = make_db(server)
    part = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\0" * MIB)  # 中身の違う .part から続けても、最後の sha256 で見つける
    with pytest.raises(ChecksumMismatch):
        gmf.download(plan(db, comfy), yes=True)
    assert not part.exists()
    gmf.download(plan(db, comfy), yes=True)  # 消したので次は最初から
    assert (comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors").read_bytes() == BODY_A


def test_retry_after_is_honored(server, comfy):
    _Handler.status_once = [(429, {"Retry-After": "7"}), (503, {"Retry-After": "2"})]
    slept: list[float] = []
    gmf.download(plan(make_db(server), comfy), yes=True, sleep=slept.append)
    assert slept[:2] == [7.0, 2.0]


def test_401_is_explained(server, comfy):
    _Handler.status_once = [(401, {})]
    with pytest.raises(DownloadRefused, match="ログイン"):
        gmf.download(plan(make_db(server), comfy), yes=True)


# --- 空き・上限 ---------------------------------------------------------------------------------


def test_disk_insufficient(server, comfy, monkeypatch):
    db = make_db(server)
    monkeypatch.setattr(downloader, "_free_bytes", lambda p: 1024**3)  # 1 GiB: 残り + 1 GiB に足りない
    p = plan(db, comfy)
    assert not p.ok and any("空きが足りません" in b for b in p.blocked)
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert _Handler.seen == [] and not any(comfy.rglob("*.safetensors*"))


def test_disk_rechecked_before_start(server, comfy, monkeypatch):
    p = plan(make_db(server), comfy)
    assert p.ok
    monkeypatch.setattr(downloader, "_free_bytes", lambda p: 10)
    with pytest.raises(DiskFull):
        gmf.download(p, yes=True)
    assert _Handler.seen == []


def test_budget(server, comfy):
    p = plan(make_db(server), comfy, budget_gb=1 / 1024)  # 1 MiB
    assert any("--budget" in b for b in p.blocked)
    assert plan(make_db(server), comfy, budget_gb=1).ok


# --- 落とさないもの -----------------------------------------------------------------------------


def test_gated_refused(server, comfy):
    db = make_db(server)
    db.models[0].files[0].gated = True
    p = plan(db, comfy)
    assert any("gated" in b and "ログイン情報" in b for b in p.blocked)
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert _Handler.seen == []


def test_uncensored_needs_explicit_content(server, comfy):
    db = make_db(server, content="uncensored")
    p = plan(db, comfy)
    assert any("無検閲" in b for b in p.blocked)
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert plan(db, comfy, content="all").ok and plan(db, comfy, content="uncensored").ok
    with pytest.raises(downloader.ModelNotFound):  # 名前の一部では sfw の中から探す(info と同じ)
        gmf.plan_download("Tiny Te", db=db, system_info=pc(), comfyui_dir=comfy)


def test_unknown_license_needs_flag(server, comfy):
    db = make_db(server)
    db.models[0].files[1].license = None
    p = plan(db, comfy)
    assert any("ライセンスが分からない" in b for b in p.blocked)
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    p2 = plan(db, comfy, include_unknown_license=True)
    assert p2.ok and any("ライセンスが分からない" in w for w in p2.warnings)
    gmf.download(p2, yes=True)


def test_not_allowlisted_license_is_warned(server, comfy):
    db = make_db(server)
    db.models[0].files[0].license = "cc-by-nc-4.0"
    p = plan(db, comfy)
    assert p.ok and not p.files[0].license_allowed
    assert any("許可リスト" in w and "cc-by-nc-4.0" in w for w in p.warnings)


@pytest.mark.parametrize(
    "directory,name",
    [
        ("vae", "../evil.safetensors"),
        ("vae", "a/../../evil.safetensors"),
        ("vae", "/abs/evil.safetensors"),
        ("vae", "C:/evil.safetensors"),
        ("vae", "C:evil.safetensors"),
        ("vae", "a\\..\\..\\evil.safetensors"),
        ("vae", "CON"),
        ("vae", "nul.safetensors"),
        ("vae", "evil.safetensors."),
        ("vae", "evil.safetensors:stream"),
        ("..", "evil.safetensors"),
        ("../..", "evil.safetensors"),
        ("C:", "evil.safetensors"),
        ("/tmp", "evil.safetensors"),
        # 予約名・末尾の点や空白はフォルダ名でも(#31 の確認の指摘 7)
        ("con", "x.safetensors"),
        ("nul", "x.safetensors"),
        ("COM1", "x.safetensors"),
        ("vae", "COM\u00b9"),
        ("vae", "LPT\u00b2.bin"),
        ("CONIN$", "x.safetensors"),
        ("vae", "conout$"),
        ("vae", "nul .txt"),
        ("models.", "x.safetensors"),
        ("vae", "x\u202etxt.safetensors"),  # 右から左への上書き
        ("vae", "x.safetensors.part"),  # 落とすときの一時ファイルと重なる
        ("vae", "x.lock"),
    ],
)
def test_path_attacks_refused(server, comfy, directory, name):
    assert safe_relpath(directory, name) is None
    db = make_db(server, files=[_file(server, name, BODY_B, directory)])
    p = plan(db, comfy)
    assert p.files[0].status == "unavailable" and not p.ok
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert _Handler.seen == []


def test_safe_relpath_allows_comfyui_subfolders():
    assert safe_relpath("text_encoders", "split/a.gguf") == ("text_encoders", "split", "a.gguf")
    assert safe_relpath(None, "x.safetensors") == ("other", "x.safetensors")


def test_symlink_escape_refused(server, comfy, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = comfy / "models" / "vae"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        if os.name != "nt":
            pytest.skip("シンボリックリンクを作れない")
        # Windows で権限が無ければジャンクション(管理者でなくても作れる。同じく外へ出る)
        import subprocess

        r = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True, check=False
        )
        if r.returncode != 0:
            pytest.skip("シンボリックリンクもジャンクションも作れない")
    p = plan(make_db(server), comfy)
    assert p.files[1].status == "unavailable" and "外" in (p.files[1].reason or "")
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert list(outside.iterdir()) == []


def test_redirect_to_other_host_refused(server, comfy):
    db = make_db(server, files=[_file(server, "tiny_vae.safetensors", BODY_B, "vae", path="evil")])
    with pytest.raises(HostNotAllowed, match="localhost"):
        gmf.download(plan(db, comfy), yes=True)
    assert [s["path"] for s in _Handler.seen] == ["/evil/tiny_vae.safetensors"]  # 転送先には送らない
    assert not any(comfy.rglob("tiny_vae*"))


def test_host_policy_default():
    pol = downloader.HOST_POLICY  # 既定の約束(この試験は server を使わないので差し替えていない)
    ok = [
        "https://huggingface.co/a/b/resolve/main/x.safetensors",
        "https://cdn-lfs.hf.co/repos/x",
        "https://cdn-lfs-us-1.hf.co/x",
        "https://cas-bridge.xethub.hf.co/xet-bridge-us/x",
        "https://objects.githubusercontent.com/x",
        "https://release-assets.githubusercontent.com/x",
    ]
    bad = [
        "http://huggingface.co/a",  # https だけ
        "https://huggingface.co.evil.com/a",
        "https://evilhf.co/a",
        "https://huggingface.co@evil.com/a",
        "https://user:pw@huggingface.co/a",
        "https://huggingface.co:8443/a",
        "https://civitai.com/api/download/models/1",
        "file:///etc/passwd",
        "https://raw.githubusercontent.com/x",
    ]
    for u in ok:
        assert pol.allowed(u), u
    for u in bad:
        assert not pol.allowed(u), u
    assert pol.allowed("https://huggingface.co/x", start=True)
    assert not pol.allowed("https://cdn-lfs.hf.co/x", start=True)  # 最初の URL は一覧のホストだけ
    assert pol.schemes == ("https",) and not pol.any_port


def test_civitai_url_is_unavailable(comfy):
    db = make_db("https://civitai.com/api/download/models")
    p = plan(db, comfy)
    assert all(f.status == "unavailable" for f in p.files) and not p.ok


# --- 同意 ---------------------------------------------------------------------------------------


def test_consent_declined_and_missing(server, comfy):
    p = plan(make_db(server), comfy)
    seen = []
    with pytest.raises(DownloadDeclined):
        gmf.download(p, confirm=lambda pl: seen.append(pl) or False)
    assert seen == [p] and _Handler.seen == []
    with pytest.raises(DownloadRefused, match="同意"):
        gmf.download(p)
    assert _Handler.seen == [] and not any(comfy.rglob("*.safetensors*"))
    gmf.download(p, confirm=lambda pl: True)
    assert (comfy / "models" / "vae" / "tiny_vae.safetensors").exists()


def test_existing_different_file_not_overwritten(server, comfy):
    dest = comfy / "models" / "vae" / "tiny_vae.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(BODY_B))  # 同じ大きさで中身が違う
    p = plan(make_db(server), comfy)
    assert p.files[1].status == "conflict" and not p.ok
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert dest.read_bytes() == b"x" * len(BODY_B)


# --- CLI ----------------------------------------------------------------------------------------


@pytest.fixture
def db_file(server, tmp_path) -> Path:
    path = tmp_path / "db.json"
    make_db(server).save(path)
    return path


HW = ["--vram", "8", "--ram", "32"]


def _tree(d: Path) -> list[str]:
    return sorted(str(p.relative_to(d)) for p in d.rglob("*"))


def test_cli_list_makes_no_request_and_writes_nothing(db_file, comfy, monkeypatch, capsys):
    from requests.adapters import HTTPAdapter

    def no_net(*a, **k):
        raise AssertionError("--list で通信した")

    monkeypatch.setattr(HTTPAdapter, "send", no_net)
    before = _tree(comfy)
    assert (
        cli.main(["download", "test_tiny", "--list", "--db", str(db_file), "--comfyui-dir", str(comfy), *HW])
        == 0
    )
    out = capsys.readouterr().out
    assert "diffusion_models/sub/tiny_unet.safetensors" in out and "license: mit" in out
    assert "https://example.invalid/LICENSE" in out and "free" in out
    assert max(len(line) for line in out.splitlines()) <= 100
    assert (
        cli.main(
            ["download", "test_tiny", "--list", "--json", "--db", str(db_file), "--comfyui-dir", str(comfy)]
        )
        == 0
    )
    env = json.loads(capsys.readouterr().out)
    assert env["listed"] is True and env["result"] is None and env["plan"]["ok"] is True
    assert _tree(comfy) == before and _Handler.seen == []


def test_cli_json_contract(db_file, comfy, capsys):
    args = ["download", "test_tiny", "--db", str(db_file), "--comfyui-dir", str(comfy), "--json", *HW]
    assert cli.main(args) == 1  # --json で落とすなら --yes が要る
    assert "--yes" in json.loads(capsys.readouterr().out)["error"]
    assert cli.main([*args, "--yes"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["command"] == "download" and env["listed"] is False
    assert {"genmodelfit", "system", "database", "plan", "result"} <= set(env)
    plan_keys = {
        "model_id",
        "name",
        "model_name",
        "precision",
        "fit_level",
        "layout",
        "root",
        "files",
        "total_bytes",
        "download_bytes",
        "disk_free_bytes",
        "disk_needed_bytes",
        "blocked",
        "warnings",
        "alternatives",
        "ok",
    }
    assert plan_keys <= set(env["plan"])
    f0 = env["plan"]["files"][0]
    for k in (
        "name",
        "relpath",
        "dest",
        "url",
        "sha256",
        "size_bytes",
        "directory",
        "license",
        "license_url",
    ):
        assert k in f0
    for k in ("license_allowed", "gated", "status", "part_bytes", "remaining_bytes", "reason"):
        assert k in f0
    r = env["result"]
    assert r["downloaded"] == [f["relpath"] for f in env["plan"]["files"]] and r["bytes_downloaded"] > 0
    assert {"resumed", "restarted", "skipped", "seconds", "files"} <= set(r)


def test_cli_blocked_json_error(server, tmp_path, comfy, capsys):
    db = make_db(server, content="uncensored")
    path = tmp_path / "db2.json"
    db.save(path)
    assert (
        cli.main(["download", "test_tiny", "--db", str(path), "--comfyui-dir", str(comfy), "--json", "--yes"])
        == 1
    )
    env = json.loads(capsys.readouterr().out)
    assert "無検閲" in env["error"] and env["plan"]["ok"] is False and env["result"] is None


def test_cli_asks_for_consent(db_file, comfy, monkeypatch, capsys):
    args = ["download", "test_tiny", "--db", str(db_file), "--comfyui-dir", str(comfy), *HW]
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert cli.main(args) == 1
    out = capsys.readouterr().out
    assert "同意して落としますか" in out and "license: mit" in out
    assert _Handler.seen == []
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert cli.main(args) == 0
    assert "sha256 はすべて一覧と一致" in capsys.readouterr().out
    assert (comfy / "models" / "vae" / "tiny_vae.safetensors").exists()


def test_cli_unknown_model(db_file, capsys):
    assert cli.main(["download", "no_such_model", "--db", str(db_file), "--list", *HW]) == 1
    assert "見つかりません" in capsys.readouterr().err


# --- ライブラリは何も出さない ------------------------------------------------------------------


def test_library_prints_nothing(server, comfy, capfd):
    db = make_db(server)
    p = gmf.plan_download("test_tiny", db=db, system_info=pc(), comfyui_dir=comfy)
    gmf.download(p, yes=True, progress=lambda e: None)
    gmf.download(gmf.plan_download("test_tiny", db=db, system_info=pc(), comfyui_dir=comfy), yes=True)
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_plan_on_bundled_list_is_offline(monkeypatch):
    """同梱の一覧で計画を作る(通信しない)。名前で選ぶと、逃がさずに載る中で一番精度の高いもの。"""
    from requests.adapters import HTTPAdapter

    monkeypatch.setattr(
        HTTPAdapter, "send", lambda *a, **k: (_ for _ in ()).throw(AssertionError("通信した"))
    )
    db = gmf.load_db(use_cache=False)
    big = gmf.plan_download("Z-Image-Turbo", db=db, system_info=pc(24), output_dir="unused-dir")
    assert big.model_id == "image_z_image_turbo" and big.precision == "bf16" and big.ok
    small = gmf.plan_download(
        "Z-Image-Turbo", db=db, system_info=pc(24), output_dir="unused-dir", budget_gb=15
    )
    assert small.model_id == "image_z_image_turbo_int8"
    prec = gmf.plan_download(
        "image_z_image_turbo", db=db, system_info=pc(24), output_dir="x", precision="int8"
    )
    assert prec.model_id == "image_z_image_turbo_int8"
    with pytest.raises(downloader.ModelNotFound, match="精度"):
        gmf.plan_download("Z-Image-Turbo", db=db, system_info=pc(24), output_dir="x", precision="fp4")
    assert not Path("unused-dir").exists()


def test_requests_exceptions_are_caught_in_loop():
    # 接続の失敗の種類(落とすときにやり直すもの)が requests にあること
    assert issubclass(requests.exceptions.ChunkedEncodingError, requests.RequestException)


# --- 実際の Hugging Face から(手動。GENMODELFIT_NETWORK_TESTS=1) -----------------------------------

E2E_MODEL = (
    "utility_birefnet_remove_background"  # 約 424 MiB、MIT、ComfyUI 公式テンプレート(background_removal)
)


@pytest.mark.network
def test_network_download(tmp_path):
    comfy = tmp_path / "ComfyUI"
    (comfy / "models").mkdir(parents=True)
    p = gmf.plan_download(E2E_MODEL, comfyui_dir=comfy, db=gmf.load_db(use_cache=False), system_info=pc(8))
    assert p.ok, p.blocked

    def stop(ev):  # 途中で 1 回止めて、続きから落とす
        if ev.phase == "download" and ev.file_bytes >= 64 * MIB and not stop.done:
            stop.done = True
            raise KeyboardInterrupt

    stop.done = False
    with pytest.raises(KeyboardInterrupt):
        gmf.download(p, yes=True, progress=stop)
    p2 = gmf.plan_download(E2E_MODEL, comfyui_dir=comfy, db=gmf.load_db(use_cache=False), system_info=pc(8))
    assert p2.files[0].status == "resume"
    res = gmf.download(p2, yes=True)
    dest = comfy / "models" / "background_removal" / "birefnet.safetensors"
    try:
        assert res.resumed == ["background_removal/birefnet.safetensors"]
        h = hashlib.sha256(dest.read_bytes()).hexdigest()
        assert h == p.files[0].sha256
    finally:
        dest.unlink(missing_ok=True)


# --- 確認の指摘(#31 の Security Checker)の回帰 ---------------------------------------------------

BS = chr(92)
EVIL_HOPS = [
    "https://evil.com" + BS + ".hf.co/x",  # urlsplit は *.hf.co、urllib3 は evil.com と読む
    "//evil.com" + BS + ".huggingface.co/x",  # 相対の転送先(解決すると上と同じ)
    "https://evil.com" + BS + "@huggingface.co/x",
    "https://huggingface.co" + BS + "@evil.com/",
    "https://evil.com%2f.hf.co/x",
    "https://evil.com%00.hf.co/x",
    "https://evil.com\t.hf.co/x",
    "https://evil.com#.hf.co",
    "https://evil.com?.hf.co",
    "https://huggingface.co@evil.com/x",
    "https://huggingface.co.evil.com/x",
    "//evil.com/x",
    "http://huggingface.co/x",
    "https://huggingface.co:8443/x",
    "https://huggingfacе.co/x",  # キリル文字の е
    "https://ｈf.co/x",
]


class _RedirectAdapter(requests.adapters.HTTPAdapter):
    """huggingface.co の最初の応答だけを手元で作る(302 で loc へ)。ほかはそのまま本物の接続へ。"""

    def __init__(self, loc: str):
        super().__init__()
        self.loc = loc

    def send(self, request, **kw):
        import io

        from requests.structures import CaseInsensitiveDict

        r = requests.Response()
        r.status_code, r.url, r.request = 302, request.url, request
        # 本物の応答の見出しは latin-1 で読まれる(http.client と同じ形で渡す)
        loc = self.loc.encode("utf-8").decode("latin-1")
        r.headers = CaseInsensitiveDict({"Location": loc, "Content-Length": "0"})
        r.raw = io.BytesIO(b"")
        return r


@pytest.fixture
def connections(monkeypatch):
    """接続の層(urllib3 の create_connection)で、つなごうとした先をすべて記録し、つながせない。"""
    import urllib3.util.connection as uc

    seen: list = []

    def fake(address, *a, **k):
        seen.append(address)
        raise OSError("試験: つながせない")

    monkeypatch.setattr(uc, "create_connection", fake)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.delenv(k, raising=False)
    return seen


_REAL_MAKE_SESSION = downloader.make_session


def _evil_session(loc: str) -> requests.Session:
    s = _REAL_MAKE_SESSION()
    s.mount("https://huggingface.co/", _RedirectAdapter(loc))
    return s


@pytest.mark.parametrize("loc", EVIL_HOPS)
def test_redirect_never_connects_to_other_host(connections, loc):
    with pytest.raises(HostNotAllowed):
        downloader._open(
            _evil_session(loc), "https://huggingface.co/a/resolve/main/f", {}, downloader.HOST_POLICY, 5
        )
    assert connections == []  # 許していないホストへは、つなごうとすらしない


def test_redirect_to_cdn_connects_only_there(connections):
    """対照: 許すホスト(Xet の CDN)への転送はそこへつなぎに行く(接続の記録が効いていること)。"""
    s = _evil_session("https://cas-bridge.xethub.hf.co/x")
    with pytest.raises(requests.ConnectionError):
        downloader._open(s, "https://huggingface.co/a/resolve/main/f", {}, downloader.HOST_POLICY, 5)
    assert [c[0] for c in connections] == ["cas-bridge.xethub.hf.co"]


def test_download_via_evil_redirect_writes_nothing(connections, monkeypatch, comfy):
    db = make_db("https://huggingface.co/example/tiny/resolve/main")
    p = plan(db, comfy)
    assert p.ok
    monkeypatch.setattr(
        downloader, "make_session", lambda: _evil_session("https://evil.com" + BS + ".hf.co/x")
    )
    with pytest.raises(HostNotAllowed):
        gmf.download(p, yes=True)
    assert connections == [] and not any(comfy.rglob("*.safetensors*"))


@pytest.mark.parametrize(
    "url", ["https://huggingface.co/x", "https://cdn-lfs-us-1.hf.co/x", "https://hf.co/x"]
)
def test_url_host_agrees_with_urllib3(url):
    import urllib3

    _scheme, host, _port = downloader.url_host(url)
    assert host == urllib3.util.parse_url(requests.Request("GET", url).prepare().url).host


def test_truncated_200_without_range_terminates(server, comfy, monkeypatch):
    """Range に応じず、毎回途中で切る相手: 最初からのやり直しに上限があり、止まる。"""
    _Handler.range_ok = False
    orig = _Handler.do_GET

    def do_get(self):
        name = self.path.rsplit("/", 1)[-1]
        if name != "tiny_unet.safetensors":
            return orig(self)
        type(self).seen.append({"path": self.path})
        data = type(self).files[name]
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data[: MIB + 5])
        self.wfile.flush()
        self.close_connection = True

    monkeypatch.setattr(_Handler, "do_GET", do_get)
    with pytest.raises(DownloadError, match="Range"):
        gmf.download(plan(make_db(server), comfy), yes=True, sleep=lambda s: None)
    assert len(_gets("tiny_unet")) <= downloader.MAX_RESTARTS + 2


def test_unverified_present_is_hashed_before_skip(server, comfy):
    """verify_existing=False の計画で present でも、download はその場で sha256 を計る
    (違えば上書きもしない)。"""
    dest = comfy / "models" / "vae" / "tiny_vae.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(BODY_B))  # 同じ大きさで中身が違う
    p = gmf.plan_download(
        "test_tiny", db=make_db(server), system_info=pc(), comfyui_dir=comfy, verify_existing=False
    )
    assert p.files[1].status == "present" and not p.files[1].verified
    with pytest.raises(DownloadRefused, match="sha256"):
        gmf.download(p, yes=True)
    assert dest.read_bytes() == b"x" * len(BODY_B)


def test_id_plus_precision_measures_existing(server, comfy):
    """ID と精度を指定して候補が 2 つ以上でも、選んだワークフローの既にあるファイルは sha256 まで計る。"""
    import copy

    db = make_db(server)
    m2 = copy.deepcopy(db.models[0])
    m2.id = "test_tiny_b"
    db = Database(models=[db.models[0], m2])
    dest = comfy / "models" / "vae" / "tiny_vae.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"x" * len(BODY_B))
    p = gmf.plan_download("test_tiny", precision="fp16", db=db, system_info=pc(), comfyui_dir=comfy)
    assert len(p.alternatives) == 1
    assert p.files[1].status == "conflict" and not p.ok


def test_file_replaced_after_plan_is_not_trusted(server, comfy):
    db = make_db(server)
    gmf.download(plan(db, comfy), yes=True)
    p = plan(db, comfy)
    assert [f.verified for f in p.files] == [True, True]
    dest = comfy / "models" / "vae" / "tiny_vae.safetensors"
    dest.write_bytes(b"y" * len(BODY_B))  # 計画のあとで差し替えられた
    with pytest.raises(DownloadRefused, match="sha256"):
        gmf.download(p, yes=True)
    assert dest.read_bytes()[:1] == b"y"


def test_concurrent_download_same_file_is_locked(server, comfy):
    db = make_db(server)
    lock = comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text(f"{os.getpid()} 0\n", encoding="ascii")  # 動いているプロセス(自分)の印
    with pytest.raises(DownloadError, match="別の download"):
        gmf.download(plan(db, comfy), yes=True)
    assert lock.exists()
    lock.write_text("999999999 0\n", encoding="ascii")  # もう無いプロセスの印は片付けて続ける
    gmf.download(plan(db, comfy), yes=True)
    assert not list(comfy.rglob("*.lock"))
    assert (comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors").read_bytes() == BODY_A


def test_two_threads_same_destination(server, comfy):
    db = make_db(server)
    p1, p2 = plan(db, comfy), plan(db, comfy)
    out: dict[str, object] = {}

    def run(k, p):
        try:
            out[k] = gmf.download(p, yes=True, sleep=lambda s: None).downloaded
        except DownloadError as e:
            out[k] = e

    ts = [threading.Thread(target=run, args=(k, p)) for k, p in (("a", p1), ("b", p2))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(60)
    assert (comfy / "models" / "diffusion_models" / "sub" / "tiny_unet.safetensors").read_bytes() == BODY_A
    assert not list(comfy.rglob("*.part")) and not list(comfy.rglob("*.lock"))
    errors = [v for v in out.values() if isinstance(v, DownloadError)]
    assert all("別の download" in str(e) or "上書き" in str(e) for e in errors), errors


def test_crafted_plan_outside_creates_nothing(server, comfy, tmp_path):
    p = plan(make_db(server), comfy)
    outside = tmp_path / "OUTSIDE" / "deep"
    p.files[1].dest = str(outside / "x.safetensors")
    with pytest.raises(DownloadRefused):
        gmf.download(p, yes=True)
    assert not (tmp_path / "OUTSIDE").exists()


def test_os_error_becomes_download_error(server, comfy, monkeypatch, tmp_path, capsys):
    def no_space(*a, **k):
        raise OSError(28, "No space left on device")

    path = tmp_path / "db.json"
    make_db(server).save(path)
    monkeypatch.setattr(downloader.os, "replace", no_space)
    with pytest.raises(DownloadError, match="No space"):
        gmf.download(plan(make_db(server), comfy), yes=True)
    args = ["download", "test_tiny", "--db", str(path), "--comfyui-dir", str(comfy), "--json", "--yes", *HW]
    assert cli.main(args) == 1
    env = json.loads(capsys.readouterr().out)
    assert "No space" in env["error"] and env["result"] is None


def test_other_folder_warned_with_comfyui(server, comfy):
    db = make_db(server, files=[_file(server, "tiny_vae.safetensors", BODY_B, None)])
    p = plan(db, comfy)
    assert p.files[0].relpath == "other/tiny_vae.safetensors"
    assert any("models/other" in w for w in p.warnings)
    assert any("models/other" in w for w in p.to_dict()["warnings"])  # JSON にも


def test_consent_when_nothing_to_download(db_file, comfy, monkeypatch, capsys):
    base = ["download", "test_tiny", "--db", str(db_file), "--comfyui-dir", str(comfy), *HW]
    assert cli.main([*base, "--yes"]) == 0
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert cli.main(base) == 0
    out = capsys.readouterr().out
    assert "nothing to download" in out and "needs" not in out and "落とすものはありません" in out
