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
