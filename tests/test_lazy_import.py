# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""読むだけのコマンドと `import genmodelfit` は requests を読み込まない(Issue #74)。

利用者の環境で urllib3 / chardet の版が合わないと、requests が読み込みのたびに警告を出す。
通信しない処理でそれを出さないため、requests は通信する関数の中でだけ読む。
"""

import subprocess
import sys
import textwrap

from clihelp import child_env


def test_read_only_paths_do_not_import_requests(tmp_path):
    script = textwrap.dedent(
        """
        import contextlib, io, sys
        import genmodelfit as gmf
        assert "requests" not in sys.modules, "import genmodelfit"
        from genmodelfit import cli
        for argv in (["system", "--json"], ["recommend", "--json"], ["fit", "--json"], ["search", "wan"],
                     ["types"], ["info", "image_sdxl_simple"], ["gate", "--json"]):
            with contextlib.redirect_stdout(io.StringIO()):
                hw = [] if argv[0] == "types" else ["--vram", "8", "--ram", "32"]
                assert cli.main([*argv, *hw]) == 0, argv
            assert "requests" not in sys.modules, argv
        sys.stdout.write("OK")
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=child_env(tmp_path / "cache"),
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert r.stdout == b"OK" and r.stderr == b""


def test_session_classes_are_built_on_first_use():
    """遅らせた `_NoAuth` / `NoNetrcSession` が、これまでどおり requests の部品として使える。"""
    import requests

    from genmodelfit import fetch, remote

    s = fetch.make_session()
    assert isinstance(s, requests.Session) and isinstance(s, fetch.NoNetrcSession)
    assert isinstance(s.auth, fetch._NoAuth) and remote._NoAuth is fetch._NoAuth
