# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit diff` の試験に使う、本物の一覧(0.6.0 と 0.7.0 の同梱の db.json)を縮めたものを作る(#59)。

元の db.json は 1 つ約 1.8 MB あるので、試験に要る項目だけを残す:
- 2 つの一覧で、安全の表(downloads・HEAD の結果など)の外の項目が変わったモデル(全部)
- 変わっていないモデル(先頭から 3 件)
- 両方で HEAD が失敗しているモデル(先頭から 3 件。前からの失敗を既定で出さないことを確かめる)
sources(内部の数)はそのまま残す。

使い方(リポジトリの一番上で。git のタグから読む):
    python tools/make_diff_fixture.py v0.6.0 v0.7.0
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from genmodelfit.diff import entry_changes, is_safe_path

OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "diff"
DB = "src/genmodelfit/data/db.json"
KEEP_UNCHANGED = 3
KEEP_HEAD_FAILED = 3


def _show(tag: str) -> dict:
    text = subprocess.run(
        ["git", "show", f"{tag}:{DB}"], check=True, capture_output=True, text=True, encoding="utf-8"
    ).stdout
    return json.loads(text)


def _failed(m: dict) -> bool:
    return any(f.get("head_status") not in (None, 200, 302, 307) for f in m.get("files") or [])


def main(old_tag: str, new_tag: str) -> None:
    a, b = _show(old_tag), _show(new_tag)
    ma = {m["id"]: m for m in a["models"]}
    mb = {m["id"]: m for m in b["models"]}
    common = sorted(ma.keys() & mb.keys())
    changed = [i for i in common if any(not is_safe_path(p) for p in entry_changes(ma[i], mb[i]))]
    rest = [i for i in common if i not in changed]
    same = [i for i in rest if not _failed(mb[i])][:KEEP_UNCHANGED]
    failed = [i for i in rest if _failed(ma[i]) and _failed(mb[i])][:KEEP_HEAD_FAILED]
    keep = set(changed) | set(same) | set(failed) | (ma.keys() ^ mb.keys())
    OUT.mkdir(parents=True, exist_ok=True)
    for tag, d in ((old_tag, a), (new_tag, b)):
        d = {**d, "models": [m for m in d["models"] if m["id"] in keep]}
        path = OUT / f"db_{tag.lstrip('v')}.json"
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{path}: {len(d['models'])} 件")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("v0.6.0", "v0.7.0")))
