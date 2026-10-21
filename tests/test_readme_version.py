# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""README に版の番号(v0.1・v0.1.0 の形)を書かない(Issue #48)。

版を上げるたびに README を直す手間をなくし、古い番号が README・PyPI のページに残らないようにする。
最新の版は PyPI のバッジ、変更点は CHANGELOG.md に任せる。
"""

import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_has_no_version_numbers():
    text = README.read_text(encoding="utf-8")
    found = [m.group(0) for m in re.finditer(r"\bv\d+\.\d+(?:\.\d+)?\b", text)]
    assert found == [], f"README に版の番号があります(PyPI のバッジか CHANGELOG.md に任せる): {found}"
