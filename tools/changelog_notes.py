# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""CHANGELOG.md から 1 つの版の節を取り出す(release.yml が GitHub Release の本文にする)。

使い方: python tools/changelog_notes.py <版> [--changelog CHANGELOG.md] [--out <ファイル>]

- 節の見出しは `## [<版>] - <日付 YYYY-MM-DD>`。日付が無い(`Unreleased` のまま)・節が無い・中身が空のときは
  理由を出して終了コード 1(タグを打つ前に CHANGELOG の Unreleased を日付にする。
  docs/release.md「版を出す手順」)。
- 本文は見出しの次の行から、次の `## ` の見出しか、末尾のリンクの定義(`[x]: https://…`)の手前まで。
- 標準ライブラリだけで動く(release.yml は venv を作らずに流す)。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]\s*-\s*(?P<date>.+?)\s*$")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LINK_DEF = re.compile(r"^\[[^\]]+\]:\s")


class NotesError(ValueError):
    """節が無い・日付が無い・中身が空。"""


def section(text: str, version: str) -> tuple[str, str]:
    """`version` の節の (日付, 本文) を返す。日付は見出しの `-` の後ろの文字そのまま。"""
    lines = text.splitlines()
    start = None
    date = ""
    for i, line in enumerate(lines):
        m = _HEADING.match(line)
        if m and m.group("version") == version:
            start, date = i + 1, m.group("date")
            break
    if start is None:
        raise NotesError(f"CHANGELOG.md に版 {version} の節(## [{version}] - <日付>)がありません")
    body = []
    for line in lines[start:]:
        if line.startswith("## ") or _LINK_DEF.match(line):
            break
        body.append(line)
    return date, "\n".join(body).strip("\n")


def release_notes(text: str, version: str) -> str:
    """GitHub Release の本文。節の見出しに日付が無ければ・中身が空なら NotesError。"""
    date, body = section(text, version)
    if not _DATE.fullmatch(date):
        raise NotesError(
            f"CHANGELOG.md の版 {version} の節に日付がありません({date!r})。"
            "タグを打つ前に `## [版] - YYYY-MM-DD` にしてください"
        )
    if not body.strip():
        raise NotesError(f"CHANGELOG.md の版 {version} の節が空です")
    return body + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CHANGELOG.md から 1 つの版の節を取り出す")
    p.add_argument("version", help="版(先頭の v は外す。例 0.12.0)")
    p.add_argument("--changelog", default="CHANGELOG.md")
    p.add_argument("--out", help="書き出すファイル(無ければ標準出力)")
    args = p.parse_args(argv)
    version = args.version[1:] if args.version.startswith("v") else args.version
    try:
        notes = release_notes(Path(args.changelog).read_text(encoding="utf-8"), version)
    except (OSError, NotesError) as e:
        print(f"::error::{e}", file=sys.stderr)
        return 1
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(notes, encoding="utf-8", newline="\n")
    else:
        sys.stdout.buffer.write(notes.encode("utf-8"))  # Windows の端末の符号に頼らない
    return 0


if __name__ == "__main__":
    sys.exit(main())
