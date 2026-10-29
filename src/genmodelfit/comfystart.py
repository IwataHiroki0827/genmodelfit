# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""起動中の ComfyUI が無いとき、見つけた ComfyUI を別のプロセスで起動して、測り終えたら止める(#124)。

measure と share が使う(cli.py)。ComfyUI(GPL-3.0)・デスクトップ版・comfy-cli のコードは import しない
(別のプロセスとして python main.py を動かすだけ)。起動の仕方は、それぞれの決まりを読んで同じにした:

- ポータブル版: `<root>/python_embeded/python.exe -s <root>/ComfyUI/main.py`
  (ComfyUI の run_nvidia_gpu.bat と同じ。--windows-standalone-build はブラウザを開くので付けない)
- 手で入れた ComfyUI・comfy-cli: ComfyUI のフォルダの `.venv` か `venv` の python で main.py
  (comfy-cli comfy_cli/resolve_python.py:36,60@c272a5e と同じ順)。venv が無ければ起動しない
  (どの python で動くか分からないので、システムの python を当て推量で使わない)
- デスクトップ版: 本体の python は basePath/.venv
  (Comfy-Org/desktop src/virtualEnvironment.ts:232,255@e2d964b)、
  main.py はアプリの resources/ComfyUI/main.py(src/main-process/comfyServer.ts:44)。両方が見つかったときだけ、
  デスクトップ版と同じ引数(--base-directory・--user-directory など。comfyServer.ts:101-111)で起動する。
  見つからなければ起動しない(アプリの画面を開くと止められないため、アプリ自体は起動しない)

どれも `--listen 127.0.0.1 --port <空いているポート> --disable-auto-launch` を付ける。出力はキャッシュの
comfyui-<日付>.log に書く。準備ができたかは GET /system_stats(127.0.0.1 だけ)で確かめる。止めるのは
自分で起動したプロセス(とその子)だけで、前から起動していた ComfyUI には触らない。

ライブラリの関数は何も表示しない。途中のことは progress に渡す。試験では subprocess・通信を差し替える。
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .i18n import tr

START_ENV = "GENMODELFIT_COMFYUI_START"  # 0 / off / no / false で自動で起動しない
DEFAULT_START_TIMEOUT = 300.0
POLL_SECONDS = 2.0
LOG_TAIL_LINES = 20
# デスクトップ版のアプリの resources(ToDesktop が置く場所。見つからなければ起動しない)
DESKTOP_RESOURCES_WIN = ("@comfyorgcomfyui-electron", "ComfyUI")
DESKTOP_RESOURCES_MAC = Path("/Applications/ComfyUI.app/Contents/Resources")


class StartError(RuntimeError):
    """起動できなかった(文はそのまま利用者に見せられる。log はログのファイル)。"""

    def __init__(self, message: str, log: Path | None = None, tail: list[str] | None = None):
        super().__init__(message)
        self.log = log
        self.tail = tail or []


def start_disabled() -> bool:
    return os.environ.get(START_ENV, "").strip().lower() in ("0", "off", "no", "false")


@dataclass
class StartPlan:
    """起動の仕方(まだ起動していない)。kind は portable / venv / desktop。"""

    kind: str
    comfyui_dir: Path  # 見つけた ComfyUI のフォルダ(デスクトップ版は basePath)
    python: Path
    main: Path
    cwd: Path
    extra_args: list[str] = field(default_factory=list)

    def argv(self, port: int) -> list[str]:
        head = [str(self.python)] + (["-s"] if self.kind == "portable" else [])
        return [
            *head,
            str(self.main),
            "--listen",
            "127.0.0.1",
            "--port",
            str(port),
            "--disable-auto-launch",
            *self.extra_args,
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "comfyui_dir": str(self.comfyui_dir)}


def _venv_python(venv: Path) -> Path | None:
    for p in (venv / "Scripts" / "python.exe", venv / "bin" / "python"):
        if p.is_file():
            return p
    return None


def _desktop_resources(home: Path, platform: str, localappdata: Path | None) -> list[Path]:
    if platform == "win32":
        base = localappdata or home / "AppData" / "Local"
        return [base / "Programs" / name / "resources" for name in DESKTOP_RESOURCES_WIN]
    if platform == "darwin":
        return [DESKTOP_RESOURCES_MAC, home / "Applications" / "ComfyUI.app" / "Contents" / "Resources"]
    return []


def plan_start(
    comfyui_dir: Path,
    *,
    desktop: bool = False,
    user_data: Path | None = None,
    home: Path | None = None,
    platform: str | None = None,
    localappdata: Path | None = None,
) -> StartPlan:
    """そのフォルダの ComfyUI の起動の仕方。分からなければ StartError(理由)。
    desktop はデスクトップ版の basePath として見つけたとき(user_data はデスクトップ版の userData)。"""
    d = Path(comfyui_dir)
    platform = platform or sys.platform
    home = home or Path.home()
    if localappdata is None and os.environ.get("LOCALAPPDATA"):
        localappdata = Path(os.environ["LOCALAPPDATA"])
    main = d / "main.py"
    if main.is_file():
        py = None
        for name in ("python_embeded",):  # ポータブル版(綴りは ComfyUI のまま)
            for exe in (d.parent / name / "python.exe", d.parent / name / "python"):
                if exe.is_file():
                    py = exe
        if py is not None:
            return StartPlan("portable", d, py, main, d)
        for venv in (d / ".venv", d / "venv"):
            py = _venv_python(venv)
            if py is not None:
                return StartPlan("venv", d, py, main, d)
        raise StartError(
            tr(
                f"ComfyUI を起動できません: {d} に Python の仮想環境(.venv・venv)が見つかりません"
                "(どの Python で動くか分からないので起動しません)",
                f"cannot start ComfyUI: no Python virtual environment (.venv or venv) in {d} (the Python it"
                " runs on is unknown, so it is not started)",
            )
        )
    if desktop:
        py = _venv_python(d / ".venv")
        res = next(
            (
                r
                for r in _desktop_resources(home, platform, localappdata)
                if (r / "ComfyUI" / "main.py").is_file()
            ),
            None,
        )
        if py is None or res is None:
            raise StartError(
                tr(
                    "ComfyUI デスクトップ版を起動できません(本体の Python かアプリの場所が見つかりません)。"
                    "アプリを開いてから、もう一度実行してください",
                    "cannot start ComfyUI Desktop (its Python or the app was not found). Open the app, then"
                    " run it again",
                )
            )
        code = res / "ComfyUI"
        extra = [
            "--base-directory",
            str(d),
            "--user-directory",
            str(d / "user"),
            "--input-directory",
            str(d / "input"),
            "--output-directory",
            str(d / "output"),
        ]
        web = code / "web_custom_versions" / "desktop_app"
        if web.is_dir():
            extra += ["--front-end-root", str(web)]
        if user_data is not None and (user_data / "extra_models_config.yaml").is_file():
            extra += ["--extra-model-paths-config", str(user_data / "extra_models_config.yaml")]
        db = (d / "user" / "comfyui.db").as_posix()
        extra += ["--database-url", f"sqlite:///{db}"]
        return StartPlan("desktop", d, py, code / "main.py", code, extra)
    raise StartError(
        tr(
            f"ComfyUI を起動できません: {d} に main.py がありません",
            f"cannot start ComfyUI: no main.py in {d}",
        )
    )


def free_port() -> int:
    """127.0.0.1 で空いているポート(0 番に結び付けて OS に選ばせる)。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def log_path(cache: Path | None = None) -> Path:
    from .db import cache_dir

    day = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return (cache or cache_dir()) / f"comfyui-{day}.log"


@dataclass
class Started:
    """起動した ComfyUI(止めるのはこれだけ)。"""

    plan: StartPlan
    process: Any
    port: int
    log: Path
    stopped: bool = False

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def to_dict(self) -> dict[str, Any]:
        return {**self.plan.to_dict(), "url": self.url, "log": str(self.log)}


def port_in_use(port: int) -> bool:
    """127.0.0.1 のそのポートで、もう何かが待っているか(つながるか)。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def launch(
    plan: StartPlan, *, port: int | None = None, log: Path | None = None, popen: Any = None
) -> Started:
    """別のプロセスで起動する(待たない)。出力はログのファイルへ。ポートは起動の直前に空いていることを
    確かめる(/system_stats に答えるのが、ほかのプロセスでなく自分で起動したものになるように)。"""
    if port is None:
        for _ in range(5):
            port = free_port()
            if not port_in_use(port):
                break
    log = log or log_path()
    log.parent.mkdir(parents=True, exist_ok=True)
    kw: dict[str, Any] = {
        "cwd": str(plan.cwd),
        "stdin": subprocess.DEVNULL,
        "stderr": subprocess.STDOUT,
    }
    if sys.platform == "win32":
        kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "CREATE_NO_WINDOW", 0
        )
    else:
        kw["start_new_session"] = True
    env = dict(os.environ)
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):  # share のトークンを ComfyUI(とカスタムノード)に渡さない
        env.pop(name, None)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    kw["env"] = env
    out = open(log, "ab")  # noqa: SIM115 子のプロセスに渡す(閉じるのは止めるとき)
    try:
        proc = (popen or subprocess.Popen)(plan.argv(port), stdout=out, **kw)
    except OSError as e:
        out.close()
        raise StartError(
            tr(f"ComfyUI を起動できません: {type(e).__name__}", f"cannot start ComfyUI: {type(e).__name__}"),
            log,
        ) from None
    started = Started(plan, proc, port, log)
    started._log_file = out  # type: ignore[attr-defined]
    return started


def tail(log: Path, n: int = LOG_TAIL_LINES) -> list[str]:
    try:
        return log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _ready(session: Any, url: str) -> bool:
    try:
        r = session.get(
            f"{url}/system_stats",
            timeout=5,
            allow_redirects=False,
            proxies={"http": None, "https": None, "all": None},
        )
    except Exception:
        return False
    return getattr(r, "status_code", 0) == 200


def wait_ready(
    started: Started,
    *,
    timeout: float = DEFAULT_START_TIMEOUT,
    progress: Callable[[float], None] | None = None,
    session: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """GET /system_stats が 200 を返すまで待つ。プロセスが終わった・時間切れなら止めて StartError。
    progress には待った秒数を渡す。"""
    from .fetch import make_session

    s = session or make_session()
    t0 = clock()
    try:
        while True:
            if started.process.poll() is not None:
                stop(started)
                raise StartError(
                    tr(
                        f"ComfyUI が起動の途中で終わりました(ログ: {started.log})",
                        f"ComfyUI exited while starting (log: {started.log})",
                    ),
                    started.log,
                    tail(started.log),
                )
            if _ready(s, started.url):
                if started.process.poll() is not None:  # 答えたのが自分で起動したものでない
                    continue
                return
            waited = clock() - t0
            if waited >= timeout:
                stop(started)
                raise StartError(
                    tr(
                        f"ComfyUI が {timeout:g} 秒たっても準備できませんでした(ログ: {started.log})",
                        f"ComfyUI was not ready after {timeout:g} s (log: {started.log})",
                    ),
                    started.log,
                    tail(started.log),
                )
            if progress is not None:
                progress(waited)
            sleep(POLL_SECONDS)
    finally:
        if session is None:
            with contextlib.suppress(Exception):
                s.close()


def stop(started: Started, *, run: Any = None, timeout: float = 15.0) -> bool:
    """自分で起動したプロセスとその子を止める(Windows は taskkill /T /F、POSIX はプロセスグループに
    SIGTERM、待っても終わらなければ SIGKILL)。止められたか。"""
    if started.stopped:
        return True
    proc = started.process
    ok = True
    if proc.poll() is None:
        if sys.platform == "win32":
            r = (run or subprocess.run)(
                ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
            ok = getattr(r, "returncode", 1) == 0
        else:
            import signal

            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                ok = False
        try:
            proc.wait(timeout=timeout)
        except Exception:
            if sys.platform != "win32":
                import signal

                with contextlib.suppress(OSError):
                    os.killpg(proc.pid, signal.SIGKILL)
            with contextlib.suppress(Exception):
                proc.wait(timeout=timeout)
        ok = ok and proc.poll() is not None
    started.stopped = True
    f = getattr(started, "_log_file", None)
    if f is not None:
        with contextlib.suppress(OSError):
            f.close()
    return ok
