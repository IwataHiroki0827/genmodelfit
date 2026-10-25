# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""--help の最後の使い方の例(#73)が、本物の引数の解釈で通り、ヘルプに出ることを確かめる。"""

from __future__ import annotations

import shlex

import pytest

from genmodelfit import cli, i18n

CASES = [(name, cmd) for name, rows in cli.EXAMPLES.items() for cmd, _, _ in rows]


@pytest.mark.parametrize("lang", ["en", "ja"])
@pytest.mark.parametrize(("name", "cmd"), CASES, ids=[f"{n or 'top'}:{c}" for n, c in CASES])
def test_example_parses(monkeypatch, lang, name, cmd):
    monkeypatch.setattr(i18n, "_cli_language", lang)
    argv = shlex.split(cmd)
    assert argv[0] == "genmodelfit"
    if name:
        assert argv[1] == name, "例はそのサブコマンドのもの"
    args = cli.build_parser().parse_args(argv[1:])  # 知らないオプションなら SystemExit(2)
    assert callable(args.func)


@pytest.mark.parametrize("lang", ["en", "ja"])
@pytest.mark.parametrize("name", list(cli.EXAMPLES))
def test_examples_in_help(capsys, lang, name):
    argv = [name, "--help", "--lang", lang] if name else ["--help", "--lang", lang]
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert ("EXAMPLES:" if lang == "en" else "例:") in out
    for cmd, ja, en in cli.EXAMPLES[name]:
        assert cmd in out
        assert (en if lang == "en" else ja) in out


def test_examples_same_commands_in_both_languages():
    """コマンドは言語によらず同じで、説明だけが違う(説明は両方の言語で書く)。"""
    for rows in cli.EXAMPLES.values():
        for cmd, ja, en in rows:
            assert cmd and ja and en
            assert not i18n_ja(en), en
            assert i18n_ja(ja), ja


def i18n_ja(text: str) -> bool:
    return any("぀" <= ch <= "ヿ" or "一" <= ch <= "鿿" or "＀" <= ch <= "￯" for ch in text)


def test_lang_flag_restored_after_main(capsys):
    """--lang は main の間だけ効く(組み込み側の言語を変えたままにしない)。"""
    assert cli.main(["types", "--lang", "en"]) == 0
    assert "Uses for --task" in capsys.readouterr().out
    assert i18n._cli_language is None
    assert i18n.get_language() == "ja"  # conftest が固定した言語に戻る


def test_bad_lang_is_usage_error(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["types", "--lang", "fr"])
    assert e.value.code == 2
    assert "--lang" in capsys.readouterr().err
