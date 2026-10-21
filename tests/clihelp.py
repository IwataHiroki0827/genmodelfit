"""CLI を別プロセスで動かすための共通の部品(組み込み側と同じ呼び方で試す)。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# 利用者の環境の影響を受けないように外す変数(UTF-8 の指定や表の幅)
_DROP = ("PYTHONIOENCODING", "PYTHONUTF8", "COLUMNS", "LINES")


def child_env(cache_dir: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _DROP}
    env["GENMODELFIT_CACHE_DIR"] = str(cache_dir)  # 手元の update のキャッシュを使わない
    env.update(extra)
    return env


def run_cli(args: list[str], cache_dir: Path, **extra: str) -> subprocess.CompletedProcess[bytes]:
    """`python -m genmodelfit <args>`。出力はバイト列のまま返す(UTF-8 で読めるかを試すため)。"""
    return subprocess.run(
        [sys.executable, "-m", "genmodelfit", *args],
        capture_output=True,
        env=child_env(cache_dir, **extra),
        timeout=120,
        check=False,
    )


def console_script() -> str | None:
    """venv に入った genmodelfit のコマンド(無ければ None)。"""
    scripts = Path(sys.executable).parent
    return shutil.which("genmodelfit", path=str(scripts))


def bundled_uncensored() -> int:
    """同梱の一覧の無検閲の件数(毎週の更新(#19)で変わるので、決め打ちせずに数える)。"""
    from genmodelfit.db import load

    return sum(1 for m in load(use_cache=False).models if m.content == "uncensored")
