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
    ["info", "image_sdxl_simple", *HW],
    ["info", "hf:Heartsync/NSFW-Uncensored", *HW],  # 理由の文が長い
    ["info", "templates_purz_animatediff_simple_weighted_ipadapters_looping_animation", *HW],  # 一番長い ID
    ["info", "template_rob_split_stack_qwen_multi_wan22", *HW],  # ファイルが一番多い(11)
]
# 文だけのコマンド(幅を見ない)。短い行しか出さない
_PLAIN = ("gate", "system")


@pytest.mark.parametrize("args", TABLE_COMMANDS, ids=lambda a: " ".join(a[:2]))
@pytest.mark.parametrize("columns", [100, 60, 250])
def test_width(cache, args, columns):
    """表と info は端末の幅(最大 100 桁)に収める。全角は 2 桁で数える。"""
    r = run_cli(args, cache, COLUMNS=str(columns))
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    lines = r.stdout.decode("utf-8").splitlines()
    assert lines
    limit = 100 if args[0] in _PLAIN else min(columns, 100)
    widest = max(lines, key=lambda x: cell_len(x.rstrip()))
    assert cell_len(widest.rstrip()) <= limit, widest


def test_fit_says_uncensored_hidden(cache):
    out = run_cli(["fit", *HW], cache).stdout.decode("utf-8")
    assert "uncensored hidden: 39 (--content all)" in out  # 同梱の無検閲は 39 件
    assert "hidden" not in run_cli(["fit", *HW, "--content", "all"], cache).stdout.decode("utf-8")
    assert "hidden" not in run_cli(["fit", *HW, "-m", "3d"], cache).stdout.decode("utf-8")  # 3D には無い


def test_info_partial_match(cache):
    # ID がそのまま合えば、無検閲でも出す
    r = run_cli(["info", "hf:Heartsync/NSFW-Uncensored", *HW], cache)
    assert r.returncode == 0 and "content: uncensored" in r.stdout.decode("utf-8")
    # 一部の一致が 1 件だけなら、それを出す
    r = run_cli(["info", "sdxlturbo", *HW], cache)
    assert r.returncode == 0 and "[sdxlturbo_example]" in r.stdout.decode("utf-8")
    # 2 件以上は先頭を黙って選ばない。候補は stderr、終了コードは 1。既定(sfw)では無検閲を候補に出さない
    r = run_cli(["info", "sdxl", *HW], cache)
    err = r.stderr.decode("utf-8")
    assert r.returncode == 1 and r.stdout == b""
    assert "ID で指定してください" in err and "image_sdxl_simple" in err
    assert "lustify" not in err.lower() and "nova-furry" not in err
    r = run_cli(["info", "sdxl", *HW, "--content", "all"], cache)
    assert r.returncode == 1 and "lustify" in r.stderr.decode("utf-8").lower()
    # 無検閲にしか合わないときは、--content all を案内する
    r = run_cli(["info", "Qwen-Image-2.1", *HW], cache)
    err = r.stderr.decode("utf-8")
    assert r.returncode == 1 and r.stdout == b"" and "--content all" in err


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
