# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`genmodelfit diff` の試験に使う、本物の一覧(過去の版の同梱の db.json)を縮めたものを作る(#59)。

元の db.json は 1 つ約 1.7 MB あるので、試験に要る項目だけを残す:
- 部品・ライセンス・内容(無検閲)・言語のどれかが変わったモデルのうち、大事なもの
  (部品・ライセンス・内容は全部、言語だけの変化は先頭から KEEP_LANG 件)
- 大事でない変化(確認の印・動かし方など)だけのモデルを ID の順に先頭から KEEP_FILLER 件。
  既定の --limit(20)より多く残し、ID の順に並べると大事な変化が隠れる、という問題を再現できるようにする
- 変わっていないモデル・両方で HEAD が失敗しているモデル(先頭から 3 件ずつ)
- 増えた・消えたモデル(全部)
sources(内部の数)はそのまま残す。

使い方(リポジトリの一番上で。git のタグから読む。tests/fixtures/diff/<前>-<後>/old.json・new.json に書く):
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
KEEP_FILLER = 25
KEEP_LANG = 6
# 大事な変化(部品・ライセンス・内容)
IMPORTANT = ("files", "files.", "license", "content")


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
    paths = {i: [p for p in entry_changes(ma[i], mb[i]) if not is_safe_path(p)] for i in common}
    important = [i for i in common if any(p.startswith(IMPORTANT) for p in paths[i])]
    lang = [i for i in common if i not in important and "languages" in paths[i]][:KEEP_LANG]
    filler = [i for i in common if paths[i] and i not in important and i not in lang][:KEEP_FILLER]
    rest = [i for i in common if not paths[i]]
    same = [i for i in rest if not _failed(mb[i])][:KEEP_UNCHANGED]
    both_failed = [i for i in common if _failed(ma[i]) and _failed(mb[i])]
    failed = [i for i in both_failed if i not in important][:KEEP_HEAD_FAILED]
    keep = set(important) | set(lang) | set(filler) | set(same) | set(failed) | (ma.keys() ^ mb.keys())
    out = OUT / f"{old_tag.lstrip('v')}-{new_tag.lstrip('v')}"
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("old", a), ("new", b)):
        d = {**d, "models": [m for m in d["models"] if m["id"] in keep]}
        path = out / f"{name}.json"
        path.write_text(json.dumps(d, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(f"{path}: {len(d['models'])} 件")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else ("v0.6.0", "v0.7.0")))
