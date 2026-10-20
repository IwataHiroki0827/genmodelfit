"""入れてすぐ数回たたく人の使い方(ヘルプ・誤った ID・パイプ・端末の幅・起動の仕方)。"""

from __future__ import annotations

import subprocess
import sys

import pytest
from clihelp import child_env, console_script, run_cli
from rich.cells import cell_len

from genmodelfit import __version__, cli

SUBCOMMANDS = ["system", "recommend", "fit", "info", "update", "gate", "bench"]
HW = ["--vram", "8", "--ram", "32", "--no-cache"]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


def test_subcommand_list_is_complete():
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.dest == "command")
    assert sorted(sub.choices) == sorted(SUBCOMMANDS)


def test_no_args_prints_full_help(cache):
    r = run_cli([], cache)
    assert r.returncode == 0 and r.stderr == b""
    out = r.stdout.decode("utf-8")
    assert out.startswith("usage: genmodelfit")
    for name in SUBCOMMANDS:
        assert f"    {name} " in out, name  # 一覧に説明つきで並ぶ


@pytest.mark.parametrize("name", SUBCOMMANDS)
def test_subcommand_help(cache, name):
    r = run_cli([name, "--help"], cache)
    assert r.returncode == 0 and r.stderr == b""
    assert r.stdout.decode("utf-8").startswith(f"usage: genmodelfit {name}")


def test_version(cache):
    r = run_cli(["--version"], cache)
    assert r.returncode == 0 and r.stdout.decode("utf-8").strip() == f"genmodelfit {__version__}"


def test_unknown_id_goes_to_stderr_in_utf8(cache):
    r = run_cli(["info", "no_such_model_xyz", *HW], cache)
    assert r.returncode == 1 and r.stdout == b""
    err = r.stderr.decode("utf-8")  # 厳密に UTF-8(cp932 のままだとここで落ちるか化ける)
    assert "モデルが見つかりません" in err and "no_such_model_xyz" in err


def test_usage_error_exits_2(cache):
    r = run_cli(["recommend", "--content", "nsfw"], cache)
    assert r.returncode == 2 and r.stdout == b""
    assert "--content" in r.stderr.decode("utf-8")


@pytest.mark.parametrize("args", [["fit", *HW, "--content", "all"], ["fit", *HW, "--json"]])
def test_closed_pipe_does_not_traceback(cache, args):
    """`| head -1` のように 1 行だけ読んで閉じられても、落ちた跡を出さない。"""
    p = subprocess.Popen(
        [sys.executable, "-m", "genmodelfit", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=child_env(cache),
    )
    assert p.stdout is not None and p.stderr is not None
    assert p.stdout.readline()
    p.stdout.close()
    err = p.stderr.read().decode("utf-8", "replace")
    code = p.wait(timeout=120)
    assert "Traceback" not in err and "Exception ignored" not in err, err
    assert code == 0, err


TABLE_COMMANDS = [
    ["recommend", *HW],
    ["recommend", *HW, "--content", "all", "--include-tools", "-n", "100"],
    ["fit", *HW, "--content", "all"],
    ["fit", "--vram", "0", "--ram", "16", "--no-cache"],
    ["gate", "--no-cache", "--only-held"],
    ["system", "--vram", "8", "--ram", "32"],
]


@pytest.mark.parametrize("args", TABLE_COMMANDS, ids=lambda a: " ".join(a[:2]))
@pytest.mark.parametrize("columns", [100, 60])
def test_width(cache, args, columns):
    r = run_cli(args, cache, COLUMNS=str(columns))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    lines = r.stdout.decode("utf-8").splitlines()
    assert lines
    if args[0] in ("recommend", "fit"):  # 表は端末の幅に収める(全角は 2 桁で数える)
        widest = max(cell_len(x.rstrip()) for x in lines)
        assert widest <= columns, max(lines, key=cell_len)
    else:
        assert max(cell_len(x) for x in lines) <= 100


def test_python_m(cache):
    r = run_cli(["recommend", *HW, "-n", "3"], cache)
    assert r.returncode == 0 and "Recommended (3)" in r.stdout.decode("utf-8")


def test_console_script(cache):
    exe = console_script()
    if exe is None:
        pytest.skip("genmodelfit のコマンドが venv に無い(pip install -e . で入る)")
    env = child_env(cache)
    args = [exe, "recommend", *HW, "-n", "3"]
    r = subprocess.run(args, capture_output=True, env=env, timeout=120, check=False)
    assert r.returncode == 0 and "Recommended (3)" in r.stdout.decode("utf-8")
    r = subprocess.run([exe, "--version"], capture_output=True, env=env, timeout=60, check=False)
    assert r.stdout.decode("utf-8").strip() == f"genmodelfit {__version__}"
