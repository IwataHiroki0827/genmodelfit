# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""wheel と sdist の中身を確かめる(ci.yml と tools/ci_local.py の両方が使う)。

使い方: python tools/check_dist.py <dist のフォルダ>
"""

from __future__ import annotations

import glob
import sys
import tarfile
import zipfile
from pathlib import Path

NEED_WHEEL = [
    "genmodelfit/py.typed",
    "genmodelfit/data/db.json",
    "licenses/LICENSE",
    "licenses/THIRD_PARTY_NOTICES",
]
NEED_SDIST = ["LICENSE", "THIRD_PARTY_NOTICES", "CHANGELOG.md", "README.md", "src/genmodelfit/py.typed"]
FORBIDDEN = ["CLAUDE.md", ".github", "__pycache__", ".pyc"]


def check(dist: Path) -> list[str]:
    wheels = glob.glob(str(dist / "*.whl"))
    sdists = glob.glob(str(dist / "*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [f"wheel と sdist が 1 つずつ要る(wheel {len(wheels)}・sdist {len(sdists)})"]
    with zipfile.ZipFile(wheels[0]) as z:
        whl = z.namelist()
    with tarfile.open(sdists[0]) as t:
        sdist = t.getnames()
    problems = [f"wheel に {n} がない" for n in NEED_WHEEL if not any(x.endswith(n) for x in whl)]
    problems += [f"sdist に {n} がない" for n in NEED_SDIST if not any(x.endswith("/" + n) for x in sdist)]
    problems += [f"配布物に {n} が入っている" for n in FORBIDDEN if any(n in x for x in whl + sdist)]
    if not problems:
        print(len(whl), "files in wheel,", len(sdist), "in sdist")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    problems = check(Path(argv[0]))
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
