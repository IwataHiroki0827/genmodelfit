"""自動でおすすめに出してよいかの判定(gate)のテスト。ネットワークには触らない。"""

from __future__ import annotations

import json

import pytest

from genmodelfit import cli
from genmodelfit.db import load
from genmodelfit.fetch import DirFetcher, FetchError, HttpFetcher
from genmodelfit.gate import (
    CHECK_IDS,
    DEFAULT_LICENSE_ALLOWLIST,
    GatePolicy,
    evaluate_offline,
    license_allowed,
    load_allowlist,
    parse_allowlist,
    parse_version,
    resolve_file_license,
    summarize,
    version_le,
)
from genmodelfit.models import ModelEntry, ModelFile, Template
from genmodelfit.update import check_heads, latest_comfyui

SHA = "a" * 64
SOURCES = {"comfyui": {"latest_release": "v0.42.1"}}


def _file(name="model.safetensors", **kw) -> ModelFile:
    d = dict(
        url=f"https://huggingface.co/Org/Repo/resolve/main/{name}",
        sha256=SHA,
        size_bytes=1_000_000_000,
        repo_id="Org/Repo",
        license="apache-2.0",
        gated=False,
        role="main",
        license_from="Org/Repo",
        head_status=302,
        head_checked_at="2026-10-20T00:00:00Z",
    )
    d.update(kw)
    return ModelFile(name=name, **d)


def _entry(files=None, **kw) -> ModelEntry:
    d = dict(
        id="t",
        name="t",
        modality="image",
        source="comfyui-template",
        files=[_file()] if files is None else files,
        template=Template(name="t", min_comfyui_version="0.3.60", open_source=True),
        content="sfw",
    )
    d.update(kw)
    return ModelEntry(**d)


def _held(entry, policy=None, sources=SOURCES) -> list[str]:
    return evaluate_offline(entry, policy, sources=sources).held_by


# --- まとめ -----------------------------------------------------------------------------------


def test_all_pass_is_auto():
    r = evaluate_offline(_entry(), sources=SOURCES)
    assert r.auto and r.held_by == []
    d = r.to_dict()
    assert d["auto"] is True and d["comfyui_version"] == "v0.42.1"
    assert [c["id"] for c in d["checks"]] == list(CHECK_IDS)
    assert all(set(c) == {"id", "passed", "detail"} and c["detail"] for c in d["checks"])


def test_no_files_is_held_everywhere():
    held = _held(_entry(files=[]))
    assert {"license", "gated", "url_reachable", "sha256", "size_known"} <= set(held)


# --- license ----------------------------------------------------------------------------------


@pytest.mark.parametrize("lic", sorted(DEFAULT_LICENSE_ALLOWLIST))
def test_license_allowlist_passes(lic):
    assert "license" not in _held(_entry(files=[_file(license=lic)]))


@pytest.mark.parametrize(
    "lic,name",
    [
        (None, None),
        ("other", None),
        ("other", "flux-1-dev-non-commercial-license"),
        ("other", "tencent-hunyuan-community"),
        ("other", "stabilityai-ai-community"),
        ("openrail++", None),
        ("creativeml-openrail-m", None),
        ("openrail", None),
        ("cc-by-nc-4.0", None),
        ("cc-by-nc-sa-4.0", None),
        ("llama3.1", None),
        ("gemma", None),
    ],
)
def test_license_held(lic, name):
    e = _entry(files=[_file(license=lic, license_name=name)])
    r = evaluate_offline(e, sources=SOURCES)
    assert r.held_by == ["license"]
    assert "許可リストに無い" in r.checks[0].detail


def test_one_bad_file_holds_the_whole_set():
    e = _entry(files=[_file("a.safetensors"), _file("b.safetensors", license="cc-by-nc-4.0", role="lora")])
    assert _held(e) == ["license"]


def test_other_matches_by_license_name():
    allow = frozenset({"mit-and-apache-2.0"})
    assert license_allowed("other", "mit-and-apache-2.0", allow)
    assert not license_allowed("other", None, allow)
    assert not license_allowed(None, None, allow)
    assert license_allowed("Apache-2.0", None, DEFAULT_LICENSE_ALLOWLIST)


def test_allowlist_replaceable():
    e = _entry(files=[_file(license="openrail++")])
    assert _held(e, GatePolicy(license_allowlist=frozenset({"openrail++"}))) == []
    assert _held(_entry(), GatePolicy(license_allowlist=frozenset({"mit"}))) == ["license"]


def test_parse_allowlist_and_file(tmp_path):
    assert parse_allowlist(" MIT, apache-2.0 ,,") == {"mit", "apache-2.0"}
    p = tmp_path / "allow.txt"
    p.write_text("# 商用で使えるもの\nmit\napache-2.0  # 注釈\n\n", encoding="utf-8")
    assert load_allowlist(str(p)) == {"mit", "apache-2.0"}
    assert load_allowlist("bsd-3-clause,mit") == {"bsd-3-clause", "mit"}


def test_restricted_components_override_repo_label():
    # HiDream の再配布(表記は mit)に入っている Llama 3.1 は Llama の規約として扱う
    f = _file("llama_3.1_8b_instruct_fp8_scaled.safetensors", license="mit", license_from="Comfy-Org/HiDream")
    assert resolve_file_license(f).license == "llama3.1"
    for name in ("gemma_2_2b_fp16.safetensors", "gemma_3_12B_it.safetensors", "t5gemma_b_b_ul2.safetensors"):
        assert resolve_file_license(_file(name, license="apache-2.0")).license == "gemma"
    # Gemma 4 は Apache-2.0(https://ai.google.dev/gemma/docs/gemma_4_license)なので当てない
    g4 = _file(
        "gemma4_e4b_it_fp8_scaled.safetensors",
        repo_id="Comfy-Org/gemma-4",
        license_from="google/gemma-4-E4B-it",
    )
    assert resolve_file_license(g4).license == "apache-2.0"
    # ただし Gemma と関係の無いリポジトリの中にあると、別の系統の部品として確かめられない
    assert resolve_file_license(_file("gemma4_e4b_it_fp8_scaled.safetensors")).license is None


def test_repackager_label_alone_is_unverified():
    f = _file("minimax_music3_dit_fp16.safetensors", license_from="Comfy-Org/MiniMax-Music-3")
    r = resolve_file_license(f)
    assert r.license is None and "Comfy-Org/MiniMax-Music-3" in r.note
    # 出どころの分かる部品は、再配布側の表記でもその部品のライセンスで読む
    for name, lic in [
        ("umt5_xxl_fp8_e4m3fn_scaled.safetensors", "apache-2.0"),
        ("t5xxl_fp16.safetensors", "apache-2.0"),
        ("clip_l.safetensors", "mit"),
        ("clip_g_hidream.safetensors", "mit"),
        ("clip_vision_h.safetensors", "mit"),
        ("wan_2.1_vae.safetensors", "apache-2.0"),
        ("Wan2_1_VAE_bf16.safetensors", "apache-2.0"),
    ]:
        assert resolve_file_license(_file(name, license_from="Comfy-Org/X")).license == lic, name
    # 本家から読めたものはそのまま
    assert resolve_file_license(_file(license_from="Wan-AI/Wan2.1-T2V-14B")).license == "apache-2.0"
    # 不明は再配布でも不明のまま
    assert resolve_file_license(_file(license=None, license_from="Comfy-Org/X")).license is None


# --- gated / official / custom nodes ----------------------------------------------------------


def test_gated_true_and_unknown():
    assert _held(_entry(files=[_file(gated=True)])) == ["gated"]
    e = _entry(files=[_file(gated=None)])
    r = evaluate_offline(e, sources=SOURCES)
    assert r.held_by == ["gated"] and "不明" in r.checks[1].detail


def test_hf_entry_is_not_official():
    e = _entry(source="huggingface", template=None, id="hf:a/b")
    assert _held(e) == ["official_workflow", "custom_nodes", "comfyui_version"]


def test_open_source_false_and_missing():
    t = Template(name="t", min_comfyui_version="0.3.0", open_source=False)
    assert _held(_entry(template=t)) == ["official_workflow"]
    t = Template(name="t", min_comfyui_version="0.3.0", open_source=None)
    r = evaluate_offline(_entry(template=t), sources=SOURCES)
    assert r.auto and "openSource の記載なし" in r.checks[2].detail


def test_custom_nodes():
    t = Template(name="t", min_comfyui_version="0.3.0", open_source=True, requires_custom_nodes=["kjnodes"])
    assert _held(_entry(template=t)) == ["custom_nodes"]


# --- comfyui_version --------------------------------------------------------------------------


def test_parse_version():
    assert parse_version("v0.42.1") == (0, 42, 1)
    assert parse_version("0.3.60") == (0, 3, 60)
    assert parse_version("0.4.0-rc1") == (0, 4, 0)
    assert parse_version("1.2") == (1, 2)
    assert parse_version("") is None and parse_version(None) is None and parse_version("latest") is None


def test_version_le_is_semantic():
    assert version_le("0.3.9", "0.3.10")  # 文字列の比較だと逆になる
    assert version_le("0.16.4", "v0.42.1")
    assert not version_le("0.43.0", "0.42.1")
    assert version_le("0.3", "0.3.0") and version_le("0.3.0", "0.3")
    assert version_le("x", "0.1") is None


def test_comfyui_version_target():
    e = _entry(template=Template(name="t", min_comfyui_version="0.43.0", open_source=True))
    assert _held(e) == ["comfyui_version"]
    assert _held(e, GatePolicy(comfyui_version="0.43.0")) == []  # 指定した版が一覧の記録より優先
    # 版が分からなければ保留
    assert _held(_entry(), sources={}) == ["comfyui_version"]
    assert _held(_entry(template=Template(name="t", open_source=True))) == ["comfyui_version"]
    t = Template(name="t", min_comfyui_version="nightly", open_source=True)
    assert _held(_entry(template=t)) == ["comfyui_version"]


# --- url / sha256 / content / size ------------------------------------------------------------


@pytest.mark.parametrize("status", [200, 302, 307])
def test_url_reachable_statuses(status):
    assert _held(_entry(files=[_file(head_status=status)])) == []


@pytest.mark.parametrize("status", [401, 403, 404, 301, 500])
def test_url_unreachable_statuses(status):
    assert _held(_entry(files=[_file(head_status=status)])) == ["url_reachable"]


def test_url_unchecked_and_missing():
    r = evaluate_offline(_entry(files=[_file(head_status=None)]), sources=SOURCES)
    assert r.held_by == ["url_reachable"] and "未確認" in r.checks[5].detail
    assert "url_reachable" in _held(_entry(files=[_file(url=None, head_status=None)]))


@pytest.mark.parametrize("sha", [None, "", "abc", "g" * 64, "a" * 63])
def test_sha256_required(sha):
    assert _held(_entry(files=[_file(sha256=sha)])) == ["sha256"]


def test_sha256_upper_case_ok():
    assert _held(_entry(files=[_file(sha256="A" * 64)])) == []


def test_content_uncensored():
    e = _entry(content="uncensored")
    r = evaluate_offline(e, sources=SOURCES)
    assert r.held_by == ["content"] and "一覧・絞り込みには出る" in r.checks[7].detail
    assert _held(e, GatePolicy(allow_uncensored=True)) == []


def test_size_known():
    assert _held(_entry(files=[_file(size_bytes=None)])) == ["size_known"]
    assert _held(_entry(files=[_file(size_bytes=0)])) == ["size_known"]
    e = _entry()
    e.files_complete = False  # #13 で足す項目。無い版でも読めるように getattr で見る
    assert _held(e) == ["size_known"]


def test_summarize_counts():
    res = [
        evaluate_offline(_entry(), sources=SOURCES),
        evaluate_offline(_entry(files=[_file(license=None, gated=True)]), sources=SOURCES),
        evaluate_offline(_entry(content="uncensored"), sources=SOURCES),
    ]
    s = summarize(res)
    assert s["models"] == 3 and s["auto"] == 1 and s["held"] == 2
    assert s["failing"]["license"] == 1 and s["failing"]["gated"] == 1 and s["failing"]["content"] == 1
    assert s["first_failing"]["license"] == 1 and s["first_failing"]["gated"] == 0
    assert s["first_failing"]["content"] == 1


# --- update: HEAD と ComfyUI の版の記録 ----------------------------------------------------------


class _Resp:
    def __init__(self, status, headers=None, body=None):
        self.status_code = status
        self.headers = headers or {}
        self._body = body

    def json(self):
        return self._body


def test_head_status_does_not_follow_and_returns_4xx():
    calls = []
    replies = [_Resp(429, {"Retry-After": "1"}), _Resp(302), _Resp(401)]

    class S:
        def request(self, method, url, **kw):
            calls.append((method, kw["allow_redirects"]))
            return replies.pop(0)

    now = [0.0]
    f = HttpFetcher(sleep=lambda s: now.__setitem__(0, now[0] + s), clock=lambda: now[0])
    f._local.session = S()
    assert f.head_status("https://huggingface.co/a/b/resolve/main/x") == 302
    assert f.head_status("https://huggingface.co/a/b/resolve/main/y") == 401
    assert calls == [("HEAD", False)] * 3 and f.rate_limited == 1


def test_check_heads_records_per_file():
    hf = "https://huggingface.co/A/B/resolve/main/"
    e1 = _entry(files=[_file(url=hf + "x", head_status=None), _file(url=hf + "y", head_status=None)])
    e2 = _entry(
        files=[_file(url=hf + "x", head_status=None), _file(url="https://civitai.com/api/download/1")]
    )
    e3 = _entry(files=[_file(url=hf + "z", head_status=999)])

    class F(DirFetcher):
        def head_status(self, url):
            if url.endswith("z"):
                raise FetchError(url, "HTTP 503", 503)
            return {"x": 302, "y": 401}[url[-1]]

    stats = check_heads([e1, e2, e3], F("."), checked_at="2026-10-20T00:00:00Z")
    assert [f.head_status for f in e1.files] == [302, 401]
    assert e2.files[0].head_status == 302 and e2.files[0].head_checked_at == "2026-10-20T00:00:00Z"
    assert e2.files[1].head_status is None  # Civitai には送らない
    assert e3.files[0].head_status is None and e3.files[0].head_checked_at is None  # 失敗は未確認
    assert stats["urls"] == 3 and stats["status"] == {"302": 1, "401": 1} and stats["unchecked"] == 1
    assert stats["not_sent"] == {"civitai.com": 1}


def test_latest_comfyui():
    class F(DirFetcher):
        def get_json(self, url):
            return {
                "tag_name": "v0.42.1",
                "html_url": "https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.42.1",
            }

    info, warn = latest_comfyui(F("."))
    assert info["latest_release"] == "v0.42.1" and warn is None
    info, warn = latest_comfyui(DirFetcher("."))  # 取れない
    assert info["latest_release"] is None and "取得できません" in warn


def test_update_heads_only(tmp_path, monkeypatch, capsys):
    from genmodelfit import update

    class Fake:
        def __init__(self, *a, **k):
            pass

        def head_status(self, url):
            return 302

        def get_json(self, url):
            assert url == update.COMFYUI_RELEASE_URL
            return {"tag_name": "v9.9.9"}

    monkeypatch.setattr(update, "HttpFetcher", Fake)
    out = tmp_path / "db.json"
    assert cli.main(["update", "--heads-only", "--no-cache", "--json", "--output", str(out)]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["sources"]["comfyui"]["latest_release"] == "v9.9.9"
    db = load(out)
    assert db.sources["head_check"]["unchecked"] == 0
    hf = [f for m in db.models for f in m.files if f.url and "huggingface.co" in f.url]
    assert hf and all(f.head_status == 302 and f.head_checked_at for f in hf)
    # 一覧の中身(件数・ライセンス)は変えない
    assert len(db.models) == len(load(use_cache=False).models)


def test_update_heads_only_all_failed_keeps_output(tmp_path, monkeypatch, capsys):
    from genmodelfit import update

    class Down:
        def __init__(self, *a, **k):
            pass

        def head_status(self, url):
            raise FetchError(url, "つながりません")

        def get_json(self, url):
            raise FetchError(url, "つながりません")

    monkeypatch.setattr(update, "HttpFetcher", Down)
    out = tmp_path / "db.json"
    assert cli.main(["update", "--heads-only", "--no-cache", "--output", str(out)]) == 1
    assert not out.exists()
    assert "HEAD" in capsys.readouterr().err


# --- CLI ----------------------------------------------------------------------------------------


@pytest.fixture
def small_db(tmp_path):
    """auto 1 件・保留 2 件の一覧。"""
    db = load(use_cache=False)
    db.models = [
        _entry(id="ok", name="ok"),
        _entry(id="nc", name="nc", files=[_file(license="cc-by-nc-4.0")]),
        _entry(id="unc", name="unc", content="uncensored"),
    ]
    db.sources = dict(SOURCES)
    p = tmp_path / "db.json"
    db.save(p)
    return str(p)


def test_cli_gate_json(small_db, capsys):
    assert cli.main(["gate", "--json", "--db", small_db, "--only-held"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["command"] == "gate"
    assert out["summary"]["auto"] == 1 and out["summary"]["held"] == 2
    assert out["summary"]["first_failing"]["license"] == 1 and out["summary"]["first_failing"]["content"] == 1
    assert {m["id"] for m in out["models"]} == {"nc", "unc"}
    assert out["policy"]["comfyui_version"] == "v0.42.1"


def test_cli_gate_policy_options(small_db, tmp_path, capsys):
    allow = tmp_path / "allow.txt"
    allow.write_text("apache-2.0\ncc-by-nc-4.0\n", encoding="utf-8")
    args = ["gate", "--json", "--db", small_db, "--license-allowlist", str(allow), "--allow-uncensored-auto"]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["summary"]["auto"] == 3
    assert cli.main(["gate", "--json", "--db", small_db, "--comfyui-version", "0.1.0"]) == 0
    s = json.loads(capsys.readouterr().out)["summary"]
    # nc はライセンスが先に落ちる。ok・unc は版が最初に落ちる
    assert s["failing"]["comfyui_version"] == 3 and s["first_failing"]["comfyui_version"] == 2
    assert cli.main(["gate", "--db", small_db, "--only-auto"]) == 0
    text = capsys.readouterr().out
    assert "auto 1 / held 2" in text and "  ok  auto" in text


def test_cli_recommend_auto_only_and_info(small_db, capsys):
    # 3 件は同じファイルを使う作り物なので、まとめずに 1 件ずつ見る
    base = ["--db", small_db, "--vram", "24", "--ram", "64", "--json"]
    assert cli.main(["recommend", "--all-workflows", "--content", "all", *base]) == 0
    assert len(json.loads(capsys.readouterr().out)["models"]) == 3
    # 既定は sfw なので無検閲の unc は出ない
    assert cli.main(["recommend", "--all-workflows", *base]) == 0
    out = json.loads(capsys.readouterr().out)
    assert {m["id"] for m in out["models"]} == {"ok", "nc"} and out["filters"]["content"] == "sfw"
    assert cli.main(["recommend", "--all-workflows", "--auto-only", *base]) == 0
    out = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in out["models"]] == ["ok"] and out["filters"]["auto_only"] is True
    assert (
        cli.main(
            [
                "recommend",
                "--all-workflows",
                "--auto-only",
                "--allow-uncensored-auto",
                "--content",
                "all",
                *base,
            ]
        )
        == 0
    )
    assert {m["id"] for m in json.loads(capsys.readouterr().out)["models"]} == {"ok", "unc"}
    # 件数の上限は保留を除いたあとにかかる(上位が保留ばかりでも auto のものが出る)
    assert cli.main(["recommend", "--all-workflows", "--auto-only", "-n", "1", *base]) == 0
    assert [m["id"] for m in json.loads(capsys.readouterr().out)["models"]] == ["ok"]
    assert cli.main(["info", "nc", *base]) == 0
    gate = json.loads(capsys.readouterr().out)["gate"]
    assert gate["auto"] is False and gate["held_by"] == ["license"]
    assert cli.main(["info", "nc", "--db", small_db]) == 0
    assert "[NG] license" in capsys.readouterr().out


def test_cli_empty_allowlist_is_error(small_db, capsys):
    assert cli.main(["gate", "--db", small_db, "--license-allowlist", ","]) == 1
    assert "空" in capsys.readouterr().err
