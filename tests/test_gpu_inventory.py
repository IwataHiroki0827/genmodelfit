# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""docs/gpu-inventory.md / .ja.md は tools/gpu_inventory.py の出力そのもの(#103。表を直したら作り直す)。"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "gpu_inventory.py"

pytestmark = pytest.mark.skipif(not TOOL.exists(), reason="sdist には tools が無い")


def _tool():
    spec = importlib.util.spec_from_file_location("gpu_inventory", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_committed_inventory_equals_the_generator():
    mod = _tool()
    texts = mod.generate()
    for lang, path in mod.OUT.items():
        assert path.read_text(encoding="utf-8") == texts[lang], (
            f"{path.name} が表と合わない。python tools/gpu_inventory.py で作り直す"
        )


def test_inventory_lists_every_rule():
    from genmodelfit import gputables

    text = (ROOT / "docs" / "gpu-inventory.md").read_text(encoding="utf-8")
    for name, rules in gputables.rules_by_table().items():
        assert f"## `{name}`" in text
        assert text.count(f"| `{name}` | `") == len(rules), name
    for sid in gputables.SOURCES:
        assert f"- `{sid}`:" in text


_JA = __import__("re").compile(r"[぀-ヿ一-鿿]")


def test_each_language_page_has_its_own_language():
    """英語版に日本語の文字が無く、日本語版の注意は日本語(言語の取り違えを防ぐ。#103)。"""
    from genmodelfit import gputables

    en = (ROOT / "docs" / "gpu-inventory.md").read_text(encoding="utf-8")
    ja = (ROOT / "docs" / "gpu-inventory.ja.md").read_text(encoding="utf-8")
    bad = [line for line in en.splitlines() if _JA.search(line.replace("[日本語](gpu-inventory.ja.md)", ""))]
    assert bad == []
    notes = [vs.note for vs in gputables.value_sets() if vs.note[0]]
    notes += [r.note for rules in gputables.rules_by_table().values() for r in rules if r.note[0]]
    for note_ja, note_en in notes:
        assert note_ja in ja.replace("\\|", "|") and note_en not in ja
        assert note_en in en.replace("\\|", "|") and _JA.search(note_ja)
