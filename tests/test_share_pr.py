# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""share が測って PR で送る(#124)。

本物の GitHub・GPU・ComfyUI・ブラウザには触らない(requests の session は偽物に差し替える)。"""

from __future__ import annotations

import base64
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from genmodelfit import cli, contrib, github, measure
from genmodelfit.models import Gpu, ModelEntry, SystemInfo

FIX = Path(__file__).parent / "fixtures" / "contrib"
GOOD = sorted(p for p in FIX.rglob("*.json") if p.parent.name == "rtx-3070")
TOKEN = "gho_SECRETtoken1234567890"
UP = "https://api.github.com/repos/laxshmate/genmodelfit"
FORK = "https://api.github.com/repos/alice/genmodelfit"


# --- 偽の GitHub ----------------------------------------------------------------------------------


class Resp:
    def __init__(self, status: int, body=None):
        self.status_code = status
        self._body = body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class FakeGitHub:
    """GitHub の API の偽物。呼ばれた順と中身を calls に残す。"""

    def __init__(self, *, fork_exists=False, open_prs=None, existing=(), user_ok=True, polls=None):
        self.calls: list[tuple[str, str, dict]] = []
        self.fork_exists = fork_exists
        self.open_prs = open_prs or []
        self.existing = set(existing)
        self.user_ok = user_ok
        self.polls = list(polls or [])
        self.put: dict[str, dict] = {}

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        path = url.split("?")[0]
        if url == github.DEVICE_CODE_URL:
            return Resp(
                200,
                {
                    "device_code": "dev-code",
                    "user_code": "ABCD-1234",
                    "verification_uri": "https://github.com/login/device",
                    "interval": 5,
                    "expires_in": 900,
                },
            )
        if url == github.TOKEN_URL:
            return Resp(200, self.polls.pop(0))
        if path == "https://api.github.com/user":
            if not self.user_ok:
                return Resp(401, {"message": "Bad credentials"})
            return Resp(200, {"login": "alice"})
        if method == "GET" and path == FORK:
            if self.fork_exists:
                return Resp(
                    200,
                    {
                        "full_name": "alice/genmodelfit",
                        "fork": True,
                        "parent": {"full_name": "laxshmate/genmodelfit"},
                    },
                )
            return Resp(404, {"message": "Not Found"})
        if method == "POST" and path == f"{UP}/forks":
            self.fork_exists = True
            return Resp(202, {"full_name": "alice/genmodelfit"})
        if method == "GET" and path == f"{UP}/pulls":
            return Resp(200, self.open_prs)
        if method == "GET" and path == f"{UP}/git/ref/heads/main":
            return Resp(200, {"object": {"sha": "abc123"}})
        if method == "POST" and path == f"{FORK}/git/refs":
            return Resp(201, {"ref": kw["json"]["ref"]})
        if method == "PUT" and path.startswith(f"{FORK}/contents/"):
            rel = path[len(f"{FORK}/contents/") :]
            if rel in self.existing:
                return Resp(422, {"message": 'Invalid request.\n\n"sha" wasn\'t supplied.'})
            self.put[rel] = kw["json"]
            return Resp(201, {"content": {"path": rel}})
        if method == "POST" and path == f"{UP}/pulls":
            return Resp(201, {"html_url": "https://github.com/laxshmate/genmodelfit/pull/200"})
        return Resp(500, {"message": f"unexpected {method} {url}"})

    def steps(self) -> list[str]:
        """呼び出しを短い名前の並びに(順を確かめる)。"""
        out = []
        for method, url, _kw in self.calls:
            path = url.split("?")[0]
            if path.endswith("/user"):
                out.append("user")
            elif path == FORK:
                out.append("get-fork")
            elif path.endswith("/forks"):
                out.append("fork")
            elif path == f"{UP}/pulls":
                out.append("list-prs" if method == "GET" else "open-pr")
            elif path.endswith("/git/ref/heads/main"):
                out.append("head")
            elif path.endswith("/git/refs"):
                out.append("branch")
            elif "/contents/" in path:
                out.append("put")
            else:
                out.append(f"{method} {path}")
        return out


@pytest.fixture
def gh(monkeypatch):
    fake = FakeGitHub()
    monkeypatch.setattr(cli, "_share_session", lambda: fake)
    return fake


@pytest.fixture
def no_session(monkeypatch):
    def deny():
        raise AssertionError("session を作った(通信しないはず)")

    monkeypatch.setattr(cli, "_share_session", deny)


@pytest.fixture
def this_pc(monkeypatch):
    g = Gpu(name="NVIDIA GeForce RTX 3070", vendor="nvidia", vram_total_gb=8.0, backend="cuda")
    s = SystemInfo(os="Windows 11", cpu="test", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[g])
    monkeypatch.setattr(cli, "_system", lambda args: s)
    monkeypatch.setattr(cli, "detect", lambda *a, **k: s)
    return s


def answers(monkeypatch, *values):
    asked: list[str] = []
    it = iter(values)

    def ask(prompt=""):
        asked.append(prompt)
        v = next(it)
        if v is None:
            raise EOFError
        return v

    monkeypatch.setattr("builtins.input", ask)
    return asked


def _write_sub(i: int) -> Path:
    d = json.loads(GOOD[0].read_text(encoding="utf-8"))
    d["measurement"]["peak_vram_gb"] = 5.0 + i / 10
    d["measurement"]["note"] = None
    return contrib.write_submission(d, str(measure.submissions_dir()) + "/")


def sugg(model_id: str, secs: int, state: str = "yes", missing: int = 0):
    e = ModelEntry(id=model_id, name=model_id, modality="image")
    return measure.Suggestion(
        model_id=model_id,
        model_name=model_id,
        name=model_id,
        modality="image",
        precision=None,
        fit_level="good",
        installed=measure.Installed(state, missing, missing, 1 if missing else 0),
        seconds=secs,
        seconds_basis="typical",
        group_key=model_id,
        fit=SimpleNamespace(model=e),
    )


@pytest.fixture
def plan(monkeypatch):
    """提案と測る処理を差し替える。測ると送る用の提出を書く(measure と同じ置き場所)。"""
    state = {
        "sugg": [
            sugg("a_installed_short", 60),
            sugg("b_not_installed", 60, "no", 2 * 1024**3),
            sugg("c_installed", 120),
            sugg("d_installed_too_long", 200),
            sugg("e_installed", 100),
        ],
        "ran": [],
        "urls": [],
        "roots": None,
        "events": [],
    }

    def fake_suggestions(db, system, *, roots=None, **k):
        state["roots"] = roots
        return state["sugg"]

    def fake_run(args, url, entry, settings):
        state["ran"].append(entry.id)
        state["urls"].append(url)
        state["events"].append(f"run:{entry.id}")
        p = _write_sub(len(state["ran"]))
        m = SimpleNamespace(outcome="ok", peak_vram_gb=5.0, seconds=12.0)
        return {
            "error": None,
            "result": SimpleNamespace(warnings=[]),
            "measurement": m,
            "submission": p,
            "path_in_repo": None,
            "saved_to": None,
            "not_saved": None,
        }

    monkeypatch.setattr(measure, "suggestions", fake_suggestions)
    monkeypatch.setattr(cli, "_measure_run", fake_run)
    return state


# --- 1. 計画: 入っているものだけ、目安の合計 ≤ 300 秒 ------------------------------------------------


def test_share_plan_takes_installed_within_the_budget():
    s = [sugg("a", 60), sugg("b", 60, "no", 1), sugg("c", 120), sugg("d", 200), sugg("e", 100)]
    picked, total = measure.share_plan(s, 300)
    assert [x.model_id for x in picked] == ["a", "c", "e"] and total == 280
    dl, dl_total = measure.share_plan(s, 300 - total, state="no")
    assert dl == [] and dl_total == 0  # 残り 20 秒には収まらない
    dl, _ = measure.share_plan(s, 100, state="no")
    assert [x.model_id for x in dl] == ["b"]
    assert measure.SHARE_BUDGET_SECONDS == 300


def test_share_measures_installed_plan_after_one_question(capsys, monkeypatch, this_pc, plan, gh):
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "", "")  # 測りますか → はい、送りますか → はい
    argv = ["--lang", "ja", "share", "--url", "http://127.0.0.1:8188"]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert plan["ran"] == ["a_installed_short", "c_installed", "e_installed"]  # 60 + 120 + 100 ≤ 300
    assert asked[0] == "3 件を測って送りますか? [Y/n] " and asked[1].startswith("3 件を送りますか?")
    assert len(asked) == 2  # 落とす候補は時間が残らないので尋ねない
    assert "ComfyUI:    http://127.0.0.1:8188(--url で指定)" in out  # measure と同じ画面の部品
    assert "入っている" in out and "約 60 秒" in out
    assert plan["urls"] == ["http://127.0.0.1:8188"] * 3
    assert "https://github.com/laxshmate/genmodelfit/pull/200" in out


def test_share_auto_finds_a_running_comfyui(capsys, monkeypatch, this_pc, plan, gh, tmp_path):
    from genmodelfit import comfyfind

    base = tmp_path / "ComfyUI"
    (base / "models").mkdir(parents=True)
    (base / "main.py").write_text("")
    cand = comfyfind.Candidate(base, "running", "127.0.0.1:8000")
    monkeypatch.setattr(comfyfind, "search_disabled", lambda: False)
    monkeypatch.setattr(comfyfind, "running_candidates", lambda *a, **k: [cand])
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["share", "--yes", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["comfyui"] == {"url": "http://127.0.0.1:8000", "found_by": "running"}
    assert plan["urls"][0] == "http://127.0.0.1:8000"
    assert plan["roots"] == [base / "models"]  # measure と同じ _measure_roots


# --- 2. 入っていないもの: y か --download のときだけ落とす ------------------------------------------


@pytest.fixture
def dl_plan(plan):
    plan["sugg"] = [sugg("a_installed_short", 60), sugg("b_not_installed", 60, "no", 2 * 1024**3)]
    return plan


def test_share_does_not_download_when_answered_no(capsys, monkeypatch, this_pc, dl_plan, gh):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_measure_download", lambda args, mid: calls.append(mid) or 0)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "", "", "")  # 測る → はい、落とす → Enter(既定は No)、送る → はい
    assert cli.main(["--lang", "ja", "share", "--url", "http://x:1"]) == 0
    assert asked[1] == "落として測りますか? [y/N] "
    assert calls == [] and dl_plan["ran"] == ["a_installed_short"]
    assert len(gh.put) == 1  # 測ったものだけを送る


def test_share_downloads_when_answered_yes(capsys, monkeypatch, this_pc, dl_plan, gh):
    calls: list[tuple] = []
    monkeypatch.setattr(cli, "_measure_download", lambda args, mid: calls.append((mid, args.yes)) or 0)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "", "y", "")
    assert cli.main(["--lang", "ja", "share", "--url", "http://x:1"]) == 0
    # 落とすのは measure と同じ _measure_download(--yes を付けないので download が自分で同意を尋ねる)
    assert calls == [("b_not_installed", False)]
    assert dl_plan["ran"] == ["a_installed_short", "b_not_installed"]


def test_share_yes_alone_never_downloads(capsys, monkeypatch, this_pc, dl_plan, gh):
    calls: list[str] = []
    monkeypatch.setattr(cli, "_measure_download", lambda args, mid: calls.append(mid) or 0)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("--yes では尋ねない"))
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["share", "--yes", "--json", "--url", "http://x:1"]) == 0
    assert calls == []
    d = json.loads(capsys.readouterr().out)
    assert d["downloaded"] is False and [m["model_id"] for m in d["measured"]] == ["a_installed_short"]
    # --download で落とす(尋ねない)。download の同意の決まりは download のまま(--yes・--any-license を渡す)
    seen = []
    monkeypatch.setattr(cli, "_measure_download", lambda args, mid: seen.append(args) or 0)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    assert cli.main(["share", "--yes", "--download", "--any-license", "--url", "http://x:1"]) == 0
    assert len(seen) == 1 and seen[0].yes and seen[0].include_unknown_license


def test_share_download_forwards_consent_to_download(monkeypatch):
    got = []
    monkeypatch.setattr(cli, "cmd_download", lambda ns: got.append(ns) or 0)
    args = cli.build_parser().parse_args(["share", "--yes", "--any-license", "--comfyui-dir", "D:/C"])
    assert cli._measure_download(args, "image_sdxl_simple") == 0
    ns = got[0]
    assert ns.yes and ns.include_unknown_license and ns.comfyui_dir == "D:/C" and ns.content is None


def test_share_json_refuses_to_download():
    assert cli.main(["share", "--json", "--yes", "--download"]) == 1


# --- 3. ComfyUI が無く、送っていない実測も無い ------------------------------------------------------


def test_share_without_comfyui_and_nothing_to_send(capsys, monkeypatch, this_pc, plan, no_session):
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": pytest.fail("起動していなければ尋ねない"))
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["--lang", "ja", "share"]) == 1
    cap = capsys.readouterr()
    assert "測るもの(入っているもの" in cap.out  # 計画は見せる(#123 と同じ)
    assert "エラー: ComfyUI を探していません" in cap.err and "ComfyUI を起動してから" in cap.err
    assert plan["ran"] == []


def test_share_without_comfyui_offers_the_unsent_ones(capsys, monkeypatch, this_pc, plan, gh):
    _write_sub(1)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "")
    assert cli.main(["--lang", "ja", "share"]) == 0
    cap = capsys.readouterr()
    assert asked == ["1 件を送りますか? [Y/n] "] and plan["ran"] == []
    assert "送っていない実測 1 件だけを送るか" in cap.err and len(gh.put) == 1


# --- 4. ログインは測る前 -------------------------------------------------------------------------


def test_share_login_happens_before_measuring(capsys, monkeypatch, this_pc, plan, gh):
    gh.user_ok = False
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "")  # 計画を見せて尋ねたあとでログインを確かめ、測る前に止まる
    assert cli.main(["--lang", "ja", "share", "--url", "http://x:1"]) == 1
    assert asked == ["3 件を測って送りますか? [Y/n] "]
    assert plan["ran"] == [] and "GITHUB_TOKEN のトークンが使えません" in capsys.readouterr().err


def test_share_without_any_login_says_how(capsys, monkeypatch, this_pc, plan, gh):
    assert github.client_id() == "Ov23liLX2TDKfSO6B955"  # 登録した OAuth アプリ(公開してよい値)
    # --json ではデバイス認証ができない: 測る前に止まる
    assert cli.main(["share", "--yes", "--json", "--url", "http://x:1"]) == 1
    assert "GITHUB_TOKEN" in json.loads(capsys.readouterr().out)["error"] and plan["ran"] == []
    assert gh.calls == []
    # client id を空にするとデバイス認証を使わない
    monkeypatch.setenv(github.CLIENT_ID_ENV, "")
    assert github.client_id() is None
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "")
    assert cli.main(["--lang", "ja", "share", "--url", "http://x:1"]) == 1
    err = capsys.readouterr().err
    assert "ログインできません。GITHUB_TOKEN を設定するか、--issue で送ってください" in err
    assert plan["ran"] == []
    monkeypatch.setenv(github.CLIENT_ID_ENV, "Iv1.test")
    assert github.client_id() == "Iv1.test"


# --- 5. デバイス認証 -----------------------------------------------------------------------------


def test_device_flow_pending_slow_down_then_token():
    fake = FakeGitHub(
        polls=[
            {"error": "authorization_pending"},
            {"error": "slow_down"},
            {"access_token": TOKEN, "expires_in": 28800, "refresh_token": "ghr_REFRESH"},
        ]
    )
    slept: list[float] = []
    shown = []
    tok = github.device_flow(fake, "cid", shown.append, sleep=slept.append)
    assert tok == (TOKEN, 28800.0) and slept == [5, 5, 10]  # slow_down で 5 秒増やす
    assert shown[0].user_code == "ABCD-1234"
    body = fake.calls[0][2]["data"]
    assert body == {"client_id": "cid", "scope": "public_repo"}
    assert fake.calls[1][2]["data"]["grant_type"] == github.DEVICE_GRANT


@pytest.mark.parametrize(("error", "kind"), [("expired_token", "expired"), ("access_denied", "denied")])
def test_device_flow_stops(error, kind):
    fake = FakeGitHub(polls=[{"error": "authorization_pending"}, {"error": error}])
    with pytest.raises(github.GitHubError) as e:
        github.device_flow(fake, "cid", lambda dc: None, sleep=lambda s: None)
    assert e.value.kind == kind


def test_login_by_device_flow_caches_the_token(monkeypatch):
    monkeypatch.setenv(github.CLIENT_ID_ENV, "Iv1.test")
    fake = FakeGitHub(polls=[{"access_token": TOKEN, "expires_in": 28800, "refresh_token": "ghr_REFRESH"}])
    c = github.login(fake, interactive=True, on_code=lambda dc: None, sleep=lambda s: None)
    assert (c.login, c.source) == ("alice", "device") and github.read_cached_token() == TOKEN
    saved = json.loads(github.token_path().read_text(encoding="utf-8"))
    assert set(saved) == {"access_token", "expires_at"} and "ghr_REFRESH" not in json.dumps(saved)
    if os.name == "posix":
        assert (github.token_path().stat().st_mode & 0o777) == 0o600
    assert TOKEN not in repr(c) and TOKEN not in json.dumps(c.to_dict())
    c2 = github.login(FakeGitHub(), interactive=False)
    assert c2.source == "cache"
    # 使えなくなったキャッシュは消して、端末でなければ止まる
    with pytest.raises(github.GitHubError) as e:
        github.login(FakeGitHub(user_ok=False), interactive=False)
    assert e.value.kind == "no_login" and not github.token_path().exists()  # 401 でキャッシュを消した


def test_expired_cached_token_runs_the_device_flow_again(monkeypatch):
    import time

    github.write_cached_token("gho_OLD", 28800, now=lambda: time.time() - 30000)  # 期限切れ
    assert github.read_cached_token() is None and not github.token_path().exists()
    github.write_cached_token("gho_OLD", 28800, now=lambda: time.time() - 30000)
    fake = FakeGitHub(polls=[{"access_token": TOKEN, "expires_in": 28800}])
    shown = []
    logs: list[str] = []
    c = github.login(fake, interactive=True, on_code=shown.append, log=logs.append, sleep=lambda s: None)
    assert c.source == "device" and shown and logs  # 期限切れを知らせてデバイス認証をやり直した
    assert all("gho_OLD" not in str(kw) for _m, _u, kw in fake.calls)  # 切れたトークンは送らない
    assert github.read_cached_token() == TOKEN
    # 端末でなければ、期限切れで止まる(デバイス認証はできない)
    github.write_cached_token("gho_OLD", 28800, now=lambda: time.time() - 30000)
    with pytest.raises(github.GitHubError) as e:
        github.login(FakeGitHub(), interactive=False)
    assert e.value.kind == "no_login"


def test_logout(capsys):
    github.write_cached_token(TOKEN)
    assert cli.main(["--lang", "ja", "share", "--logout"]) == 0
    assert "消しました" in capsys.readouterr().out and github.read_cached_token() is None
    assert cli.main(["share", "--logout", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["removed"] is False


def test_hosts_are_only_github():
    with pytest.raises(github.GitHubError):
        github._request(FakeGitHub(), "GET", "http://api.github.com/user")
    with pytest.raises(github.GitHubError):
        github._request(FakeGitHub(), "GET", "https://example.com/user")


# --- 6. フォーク・ブランチ・ファイル・PR の順と中身 -------------------------------------------------


def _creds():
    return github.Credentials(TOKEN, "alice", "env:GITHUB_TOKEN")


def test_submit_new_pr_order_and_content():
    fake = FakeGitHub()
    files = [("data/contrib/rtx-3070/2026-10-29-aaaaaaaa.json", '{"a": 1}\n')]
    res = github.submit(
        fake,
        _creds(),
        files,
        title="measurements: NVIDIA GeForce RTX 3070 (1)",
        body="b",
        branch="genmodelfit-share/rtx-3070-12345678",
        match_prefix="genmodelfit-share/rtx-3070-",
        sleep=lambda s: None,
    )
    assert fake.steps() == ["get-fork", "fork", "get-fork", "list-prs", "head", "branch", "put", "open-pr"]
    for _m, _u, kw in fake.calls:
        assert kw["headers"]["Authorization"] == f"Bearer {TOKEN}"
        assert kw["allow_redirects"] is False
    branch = next(kw for m, u, kw in fake.calls if u.endswith("/git/refs"))["json"]
    assert branch == {"ref": "refs/heads/genmodelfit-share/rtx-3070-12345678", "sha": "abc123"}
    put = fake.put["data/contrib/rtx-3070/2026-10-29-aaaaaaaa.json"]
    assert base64.b64decode(put["content"]).decode() == '{"a": 1}\n'
    assert put["branch"] == "genmodelfit-share/rtx-3070-12345678"
    pr = next(kw for m, u, kw in fake.calls if m == "POST" and u == f"{UP}/pulls")["json"]
    assert pr == {
        "title": "measurements: NVIDIA GeForce RTX 3070 (1)",
        "head": "alice:genmodelfit-share/rtx-3070-12345678",
        "base": "main",
        "body": "b",
    }
    assert res.pr_url.endswith("/pull/200") and not res.reused and res.uploaded == [files[0][0]]


def test_submit_adds_to_the_open_pr_and_skips_existing_files():
    open_pr = {
        "html_url": "https://github.com/laxshmate/genmodelfit/pull/150",
        "head": {
            "ref": "genmodelfit-share/rtx-3070-old00000",
            "user": {"login": "alice"},
            "repo": {"full_name": "alice/genmodelfit"},
        },
    }
    other = {  # ほかの人の PR・share でないブランチには足さない
        "html_url": "https://github.com/laxshmate/genmodelfit/pull/151",
        "head": {
            "ref": "genmodelfit-share/rtx-3070-x",
            "user": {"login": "bob"},
            "repo": {"full_name": "bob/genmodelfit"},
        },
    }
    a = "data/contrib/rtx-3070/2026-10-29-aaaaaaaa.json"
    b = "data/contrib/rtx-3070/2026-10-29-bbbbbbbb.json"
    fake = FakeGitHub(fork_exists=True, open_prs=[other, open_pr], existing={a})
    res = github.submit(
        fake,
        _creds(),
        [(a, "{}\n"), (b, "{}\n")],
        title="t",
        body="b",
        branch="genmodelfit-share/rtx-3070-new11111",
        match_prefix="genmodelfit-share/rtx-3070-",
    )
    assert fake.steps() == ["get-fork", "list-prs", "put", "put"]  # 新しいブランチ・PR は作らない
    assert res.reused and res.pr_url.endswith("/pull/150")
    assert res.uploaded == [b] and res.skipped == [a]
    assert fake.put[b]["branch"] == "genmodelfit-share/rtx-3070-old00000"


def test_submit_explains_a_missing_scope():
    class NoWrite(FakeGitHub):
        def request(self, method, url, **kw):
            if method == "PUT":
                self.calls.append((method, url, kw))
                return Resp(404, {"message": "Not Found"})
            return super().request(method, url, **kw)

    with pytest.raises(github.GitHubError) as e:
        github.submit(
            NoWrite(fork_exists=True),
            _creds(),
            [("data/contrib/x/2026-10-29-aaaaaaaa.json", "{}")],
            title="t",
            body="b",
            branch="genmodelfit-share/x-1",
            match_prefix="genmodelfit-share/x-",
        )
    assert e.value.kind == "scope" and "public_repo" in str(e.value) and TOKEN not in str(e.value)


def test_pr_body_and_title_are_harmless():
    d = json.loads(GOOD[0].read_text(encoding="utf-8"))
    d["measurement"]["model_id"] = "x | [link](https://evil.example)"
    body = github.pr_body([d])
    assert github.PR_MARKER in body and "genmodelfit share" in body and "](https://" not in body
    assert github.pr_title("NVIDIA GeForce RTX 3070", 2) == "measurements: NVIDIA GeForce RTX 3070 (2)"


# --- 7. 送ったら印が付き、次の --no-measure では出ない ----------------------------------------------


def test_sent_ones_are_marked_and_not_sent_again(capsys, monkeypatch, gh):
    _write_sub(1)
    _write_sub(2)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["share", "--no-measure", "--yes", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["sent"] is True and d["marked_shared"] == 2 and len(d["prs"]) == 1
    assert d["prs"][0]["url"].endswith("/pull/200") and len(d["prs"][0]["uploaded"]) == 2
    assert len(measure.load_shared()) == 2
    title = next(kw for m, u, kw in gh.calls if m == "POST" and u == f"{UP}/pulls")["json"]["title"]
    assert title == "measurements: NVIDIA GeForce RTX 3070 (2)"
    n = len(gh.calls)
    assert cli.main(["share", "--no-measure", "--yes", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["pending"] == [] and d["sent"] is False and len(gh.calls) == n  # 送るものが無ければ通信しない


def test_share_file_sends_one_file_by_pr(capsys, monkeypatch, gh):
    p = _write_sub(1)
    _write_sub(2)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    asked = answers(monkeypatch, "y")
    assert cli.main(["--lang", "ja", "share", str(p)]) == 0
    assert asked == ["1 件を送りますか? [Y/n] "] and len(gh.put) == 1
    assert len(measure.saved_submissions()) == 1


def test_share_refuses_personal_information(capsys, monkeypatch, gh):
    d = json.loads(GOOD[0].read_text(encoding="utf-8"))
    d["measurement"]["note"] = r"C:\Users\bob\ComfyUI"
    p = contrib.write_submission(d, str(measure.submissions_dir()) + "/")
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["share", str(p), "--yes", "--json"]) == 1
    out = json.loads(capsys.readouterr().out)
    assert out["files"][0]["ok"] is False and gh.calls == []


# --- 8. --dry-run は通信しない・--issue は今のまま --------------------------------------------------


def test_dry_run_makes_no_session(capsys, monkeypatch, this_pc, plan, no_session):
    _write_sub(1)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["share", "--dry-run", "--url", "http://x:1"]) == 0
    assert "--dry-run" in capsys.readouterr().out and plan["ran"] == []
    assert cli.main(["share", "--dry-run", "--json", "--url", "http://x:1"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["dry_run"] is True and len(d["plan"]) == 3 and len(d["files"]) == 1
    assert measure.load_shared() == {}


def test_issue_flag_rules(capsys):
    assert cli.main(["share", "--print-url"]) == 1  # --print-url は --issue のときだけ
    assert cli.main(["share", "--issue", "--dry-run"]) == 1
    assert cli.main(["share", "--issue", "--download"]) == 1


def test_json_and_no_terminal_need_yes(capsys, monkeypatch, no_session):
    assert cli.main(["share", "--json"]) == 1
    monkeypatch.setattr(cli, "_interactive", lambda: False)
    assert cli.main(["share", "--no-measure"]) == 1


# --- 9. トークンはどこにも出ない ------------------------------------------------------------------


def test_token_never_shown(capsys, caplog, monkeypatch, this_pc, plan, gh):
    caplog.set_level(logging.DEBUG)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "", "")
    assert cli.main(["share", "--url", "http://x:1"]) == 0
    cap = capsys.readouterr()
    assert cli.main(["share", "--no-measure", "--yes", "--json", "--all"]) == 0
    cap2 = capsys.readouterr()
    for text in (cap.out, cap.err, cap2.out, cap2.err, caplog.text):
        assert TOKEN not in text
    gh.user_ok = False
    assert cli.main(["share", "--no-measure", "--yes", "--json", "--all"]) == 1
    assert TOKEN not in capsys.readouterr().out


def test_library_prints_nothing(capfd, monkeypatch):
    monkeypatch.setenv(github.CLIENT_ID_ENV, "Iv1.test")
    github.login(
        FakeGitHub(polls=[{"access_token": TOKEN, "expires_in": 28800}]),
        interactive=True,
        on_code=lambda dc: None,
        sleep=lambda s: None,
    )
    github.submit(
        FakeGitHub(),
        _creds(),
        [("data/contrib/x/2026-10-29-aaaaaaaa.json", "{}")],
        title="t",
        body="b",
        branch="genmodelfit-share/x-1",
        match_prefix="genmodelfit-share/x-",
        sleep=lambda s: None,
    )
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_github_module_is_not_imported_by_read_only_commands():
    import subprocess

    code = "import genmodelfit.cli as c, sys; c.main(['types']); print('genmodelfit.github' in sys.modules)"
    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    assert r.stdout.strip().splitlines()[-1] == "False"


# --- 実機で見つかったこと(#124): 足りないノード・順・次にすること・--download の案内 ---------------------


class FakeNodes:
    def __init__(self, missing):
        self.missing_map = missing
        self.asked = []

    def missing(self, model_id):
        self.asked.append(model_id)
        return self.missing_map.get(model_id, [])


def test_share_skips_models_needing_missing_nodes_and_backfills(capsys, monkeypatch, this_pc, plan, gh):
    nodes = FakeNodes({"a_installed_short": ["LoadMoGeModel", "MoGePanoramaInference"]})
    monkeypatch.setattr(cli, "_node_check", lambda url: nodes)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["--lang", "ja", "share", "--yes", "--url", "http://x:1"]) == 0
    out = capsys.readouterr().out
    # a を外し、時間の残りで c・e を測る(d は 200 秒で収まらない)
    assert plan["ran"] == ["c_installed", "e_installed"]
    assert "この ComfyUI に無いノードが要る。ComfyUI を更新すると測れます" in out
    assert "LoadMoGeModel, MoGePanoramaInference" in out
    assert "a_installed_short" not in nodes.asked[1:]  # 1 回だけ確かめる
    out_json = []
    plan["ran"].clear()
    assert cli.main(["share", "--yes", "--json", "--url", "http://x:1", "--all"]) == 0
    out_json = json.loads(capsys.readouterr().out)
    assert out_json["skipped"] == [
        {
            "model_id": "a_installed_short",
            "name": "a_installed_short",
            "missing_nodes": ["LoadMoGeModel", "MoGePanoramaInference"],
        }
    ]


def test_share_asks_before_logging_in(capsys, monkeypatch, this_pc, plan, gh):
    events = []
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    real_login = cli._share_login
    monkeypatch.setattr(cli, "_share_login", lambda args: events.append("login") or real_login(args))
    monkeypatch.setattr("builtins.input", lambda p="": events.append("ask:" + p[:6]) or "")
    assert cli.main(["--lang", "ja", "share", "--url", "http://x:1"]) == 0
    assert events[:2] == ["ask:3 件を測っ", "login"]  # 計画 → 確認 → ログイン → 測る


def test_share_suggests_download_when_nothing_was_measured(capsys, monkeypatch, this_pc, dl_plan, gh):
    def fail_run(args, url, entry, settings):
        return {"error": "boom"}

    monkeypatch.setattr(cli, "_measure_run", fail_run)
    monkeypatch.setattr(cli, "_measure_download", lambda *a: pytest.fail("--yes だけでは落とさない"))
    monkeypatch.setattr("builtins.input", lambda p="": pytest.fail("--yes では尋ねない"))
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    monkeypatch.setenv("GITHUB_TOKEN", TOKEN)
    assert cli.main(["--lang", "ja", "share", "--yes", "--url", "http://x:1"]) == 1
    out = capsys.readouterr().out
    assert "--download を付けると落として測ります" in out and "2.0 GB" in out
    assert "落としません(--download を付けると" in out  # 計画の画面も尋ねないことを言う
    assert cli.main(["share", "--yes", "--json", "--url", "http://x:1"]) == 1
    assert json.loads(capsys.readouterr().out)["download_hint"] == {"count": 1, "bytes": 2 * 1024**3}


def test_measure_stops_early_when_nodes_are_missing(capsys, monkeypatch, this_pc):
    monkeypatch.setattr(cli, "_node_check", lambda url: FakeNodes({"image_sdxl_simple": ["NewNode"]}))
    monkeypatch.setattr(cli, "_measure_run", lambda *a, **k: pytest.fail("測らない"))
    monkeypatch.setattr(cli, "_measure_download", lambda *a: pytest.fail("落とさない"))
    assert cli.main(["--lang", "ja", "measure", "image_sdxl_simple", "--yes", "--url", "http://x:1"]) == 1
    err = capsys.readouterr().err
    assert "ComfyUI に無いノードを使っています: NewNode" in err
    assert "次にすること: ComfyUI を新しい版に更新してください" in err and "起動していて" not in err


def test_measure_list_hides_models_needing_missing_nodes(capsys, monkeypatch, this_pc):
    from genmodelfit import measure as m

    sugg_all = m.suggestions(cli._db(cli.build_parser().parse_args(["measure"])), this_pc)[:3]
    first = sugg_all[0].model_id
    monkeypatch.setattr(cli, "_node_check", lambda url: FakeNodes({first: ["NewNode"]}))
    monkeypatch.setattr(cli, "_interactive", lambda: True)
    answers(monkeypatch, "q")
    assert cli.main(["--lang", "ja", "measure", "-n", "3", "--url", "http://x:1"]) == 0
    out = capsys.readouterr().out
    rows = [x for x in out.splitlines() if x.lstrip().startswith(("1.", "2.", "3."))]
    assert len(rows) == 2 and f"[{first}]: NewNode" in out


def test_next_step_for_missing_nodes_says_update():
    from genmodelfit.workflow import missing_nodes_message

    msg = missing_nodes_message(["A", "B"])
    hint = cli._measure_next_step(msg, "x")
    assert "更新" in hint and "起動していて" not in hint
    assert "起動していて" in cli._measure_next_step("ComfyUI(http://x)に接続できません: refused", "x")


def test_missing_nodes_and_node_check():
    from genmodelfit import bench
    from genmodelfit.workflow import missing_nodes

    ui = {
        "nodes": [
            {"id": 1, "type": "KSampler", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []},
            {"id": 2, "type": "LoadMoGeModel", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []},
            {"id": 3, "type": "Gone", "mode": 4, "inputs": [], "outputs": [], "widgets_values": []},
            {"id": 4, "type": "Note", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []},
        ],
        "links": [],
    }
    assert missing_nodes(ui, {"KSampler": {}}) == ["LoadMoGeModel"]  # 素通し・メモは数えない

    class Client:
        s = None
        calls = 0

        def object_info(self):
            Client.calls += 1
            return {"KSampler": {}}

        def template(self, name):
            return ui if name == "t" else None

    nc = bench.NodeCheck("http://127.0.0.1:1", client=Client())
    assert nc.missing("t") == ["LoadMoGeModel"] and nc.missing("t") == ["LoadMoGeModel"]
    assert Client.calls == 1

    class Down(Client):
        def object_info(self):
            raise bench.BenchError("down")

    assert (
        bench.NodeCheck("http://127.0.0.1:1", client=Down()).missing("t") == []
    )  # 確かめられなければ外さない
