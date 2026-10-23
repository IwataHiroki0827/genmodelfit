# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""文書は英語版(X.md)と日本語版(X.ja.md)の組で持つ。

- docs/ の各ページと CONTRIBUTING は、英語版と日本語版がそろい、先頭で互いに行き来できる。
- 文書の中の相対リンクと、README から docs への GitHub のリンクは、行き先のファイルと見出しがある。
- コード・試験・ワークフロー・CLAUDE.md が `docs/X.md「節の名前」` で引く節は、日本語版にその名前がある
  (コメントは日本語なので、節の名前は日本語版の見出しで書く)。
"""

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GITHUB = "https://github.com/laxshmate/genmodelfit/blob/main/"

# 英語版と日本語版の組(英語版のパス)
PAIRS = [*sorted(p for p in DOCS.glob("*.md") if not p.name.endswith(".ja.md")), ROOT / "CONTRIBUTING.md"]


def ja(path: Path) -> Path:
    return path.with_name(path.name[: -len(".md")] + ".ja.md")


PAGES = PAIRS + [ja(p) for p in PAIRS] + [ROOT / "README.md", ROOT / "README.ja.md"]


def _strip_code(text: str) -> str:
    """囲みのコード(``` の中)を除く。見出しやリンクに見える行を拾わないため。"""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if not fenced:
            out.append(line)
    return "\n".join(out)


def slug(heading: str) -> str:
    """GitHub が見出しに付ける id(小文字・記号を除く・空白は -。日本語の文字は残す)。"""
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)  # リンクは文字だけ
    text = text.replace("`", "").replace("*", "")
    text = text.strip().lower()
    kept = []
    for ch in text:
        if ch in " -_":
            kept.append("-" if ch == " " else ch)
        elif unicodedata.category(ch)[0] in "LNM":
            kept.append(ch)
    return "".join(kept)


def anchors(path: Path) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for line in _strip_code(path.read_text(encoding="utf-8")).splitlines():
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not m:
            continue
        base = slug(m.group(1))
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def links(path: Path) -> list[str]:
    return re.findall(r"\]\(([^)\s]+)\)", _strip_code(path.read_text(encoding="utf-8")))


def test_every_page_has_both_languages():
    for en in PAIRS:
        assert en.exists(), en
        assert ja(en).exists(), f"{ja(en).name} がない(英語版と日本語版はそろえる)"
    for jp in DOCS.glob("*.ja.md"):
        assert jp.with_name(jp.name.replace(".ja.md", ".md")).exists(), f"{jp.name} の英語版がない"


@pytest.mark.parametrize("en", PAIRS, ids=lambda p: p.name)
def test_language_switch_at_the_top(en):
    head_en = en.read_text(encoding="utf-8").splitlines()[:5]
    head_ja = ja(en).read_text(encoding="utf-8").splitlines()[:5]
    assert f"**English** · [日本語]({ja(en).name})" in head_en
    assert f"[English]({en.name}) · **日本語**" in head_ja


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_utf8_lf_without_bom(path):
    raw = path.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


@pytest.mark.parametrize("path", PAGES, ids=lambda p: p.name)
def test_links_point_to_existing_files_and_headings(path):
    problems = []
    for target in links(path):
        if target.startswith(GITHUB):
            rel = target[len(GITHUB) :]
            base = ROOT
        elif target.startswith("https://github.com/laxshmate/genmodelfit#"):
            rel, base = "README.md#" + target.split("#", 1)[1], ROOT
        elif re.match(r"[a-z]+:", target):
            continue  # 外のサイト
        else:
            rel, base = target, path.parent
        file_part, _, anchor = rel.partition("#")
        dest = (base / file_part).resolve() if file_part else path
        if not dest.exists():
            problems.append(f"{target}: ファイルがない")
            continue
        if anchor and dest.suffix == ".md" and anchor not in anchors(dest):
            problems.append(f"{target}: 見出し #{anchor} がない")
    assert problems == []


def _cited_sections():
    """`docs/X.md「名前」`・`docs/X.md の「名前」` の形の参照(コード・試験・ワークフロー・CLAUDE.md)。"""
    files = [p for p in (ROOT / "CLAUDE.md",) if p.exists()]  # sdist には CLAUDE.md・.github が無い
    for sub in ("src/genmodelfit", "tests", ".github/workflows"):
        files += [p for p in (ROOT / sub).rglob("*") if p.suffix in (".py", ".yml")]
    out = []
    for f in files:
        if f == Path(__file__).resolve():
            continue  # この試験の説明文の例
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"docs/(\w+)\.md(?: の)?「([^」]+)」", text):
            out.append((f.relative_to(ROOT).as_posix(), m.group(1), m.group(2)))
    return out


def test_cited_sections_exist_in_the_japanese_page():
    cited = _cited_sections()
    assert len(cited) > 20  # 探し方が壊れていないこと
    missing = []
    for where, name, section in cited:
        page = DOCS / f"{name}.ja.md"
        if not page.exists() or section not in page.read_text(encoding="utf-8"):
            missing.append(f"{where}: docs/{name}.ja.md に「{section}」がない")
    assert missing == []
