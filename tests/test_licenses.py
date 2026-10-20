"""実行時の依存のライセンスを許可リストと照らす(CLAUDE.md「依存は MIT / Apache-2.0 / BSD のものだけ」)。

入っている genmodelfit から requires をたどる(dev などの extra は除く)。読むのは License-Expression、
License、分類子の順。例外は certifi の MPL-2.0 だけ(改変せずに使う。CLAUDE.md)。
"""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, distribution

import pytest
from packaging.requirements import Requirement

ALLOWED = {
    "MIT",
    "MIT-0",
    "Apache-2.0",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "BSD",  # 分類子の「BSD License」(2 条項か 3 条項かは書かれていない)
    "ISC",
    "PSF-2.0",
    "Apache-2.0 OR BSD-3-Clause",
    "BSD-3-Clause OR Apache-2.0",
}
EXCEPTIONS = {"certifi": {"MPL-2.0"}}
MISSING = "NOT INSTALLED"

_CLASSIFIERS = {
    "MIT License": "MIT",
    "MIT No Attribution License (MIT-0)": "MIT-0",
    "Apache Software License": "Apache-2.0",
    "BSD License": "BSD",
    "ISC License (ISCL)": "ISC",
    "Python Software Foundation License": "PSF-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
}
_ALIASES = {
    "mit": "MIT",
    "mit license": "MIT",
    "mit-0": "MIT-0",
    "apache 2.0": "Apache-2.0",
    "apache-2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache license, version 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0",
    "bsd": "BSD",
    "bsd license": "BSD",
    "bsd-2-clause": "BSD-2-Clause",
    "bsd-3-clause": "BSD-3-Clause",
    "3-clause bsd license": "BSD-3-Clause",
    "new bsd license": "BSD-3-Clause",
    "isc": "ISC",
    "psf-2.0": "PSF-2.0",
    "mpl-2.0": "MPL-2.0",
    "mpl 2.0": "MPL-2.0",
}


def _norm(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def license_of(dist) -> str:
    md = dist.metadata
    expr = (md.get("License-Expression") or "").strip()
    if expr:
        return expr
    text = (md.get("License") or "").strip()
    if text and "\n" not in text and len(text) < 80:  # 全文を入れている物は分類子で見る
        return _ALIASES.get(text.lower(), text)
    found = sorted(
        {
            _CLASSIFIERS.get(c.split(" :: ")[-1], c)
            for c in md.get_all("Classifier") or []
            if c.startswith("License :: ")
        }
    )
    return " OR ".join(found) if found else "?"


def runtime_dependencies(root: str = "genmodelfit") -> dict[str, str]:
    """{配布名: ライセンス}。extra(dev など)の依存はたどらない。"""
    seen: dict[str, str] = {}
    todo = [root]
    while todo:
        name = _norm(todo.pop())
        if name in seen:
            continue
        try:
            dist = distribution(name)
        except PackageNotFoundError:  # この環境で要る(marker が合う)のに入っていない → 落とす
            seen[name] = MISSING
            continue
        seen[name] = license_of(dist)
        for text in dist.requires or []:
            req = Requirement(text)
            if req.marker is not None and not req.marker.evaluate({"extra": ""}):
                continue
            todo.append(req.name)
    return seen


def allowed(name: str, lic: str) -> bool:
    if lic == MISSING:
        return False
    if lic in ALLOWED or lic in EXCEPTIONS.get(name, set()):
        return True
    # 「A OR B」はどちらかが許されていればよい(二重ライセンス)
    parts = [p.strip("() ") for p in re.split(r"\s+OR\s+", lic)]
    return len(parts) > 1 and any(p in ALLOWED for p in parts)


def table(deps: dict[str, str]) -> str:
    rows = [f"  {n:<22}{lic:<32}{'ok' if allowed(n, lic) else 'NG'}" for n, lic in sorted(deps.items())]
    return "\n".join(["  package               license                         ", *rows])


def test_runtime_dependency_licenses():
    deps = runtime_dependencies()
    assert {"genmodelfit", "requests", "rich", "psutil", "platformdirs", "cryptography"} <= set(deps)
    bad = {n: lic for n, lic in deps.items() if not allowed(n, lic)}
    assert not bad, "許可リストに無いライセンスの依存があります:\n" + table(deps)


@pytest.mark.parametrize(
    ("name", "lic", "ok"),
    [
        ("certifi", "MPL-2.0", True),
        ("other", "MPL-2.0", False),  # MPL は certifi だけの例外
        ("x", "GPL-3.0-or-later", False),
        ("x", "LGPL-3.0", False),
        ("x", "Apache-2.0 OR BSD-3-Clause", True),
        ("x", "?", False),
        ("x", MISSING, False),
    ],
)
def test_allowlist_rules(name, lic, ok):
    assert allowed(name, lic) is ok


def test_missing_required_dependency_fails(monkeypatch):
    """要る依存が入っていなければ、飛ばさずに NG にする。"""
    import test_licenses as me

    real = me.distribution

    def fake(name):
        if name == "rich":
            raise PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(me, "distribution", fake)
    deps = runtime_dependencies()
    assert deps["rich"] == MISSING and not allowed("rich", deps["rich"])
