import io
import json
import re

import pytest
from rich.cells import cell_len

from genmodelfit import cli, i18n
from genmodelfit.db import load
from genmodelfit.groups import build_index
from genmodelfit.models import Estimate, Gpu, ModelEntry, SystemInfo
from genmodelfit.recommend import Fit
from genmodelfit.table import make_console, print_fit_table

HEADER = [
    "Type",
    "Status",
    "Score",
    "Model",
    "Use",
    "Prec",
    "VRAM",
    "Mem%",
    "Mode",
    "Disk",
    "License",
]


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


def pc(vram=8.0):
    return SystemInfo(
        os="t",
        cpu="t",
        cpu_cores=8,
        ram_total_gb=32,
        ram_available_gb=32,
        gpus=[Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")],
    )


def fit(mid, level="good", content="sfw", gate=None, mode="gpu"):
    m = ModelEntry(id=mid, name=mid, modality="image", content=content, license="apache-2.0")
    e = Estimate(
        vram_gb=7.5, vram_offload_gb=4.0, ram_gb=12, disk_gb=6, basis="estimated", fit=level, run_mode=mode
    )
    return Fit(m, e, 50.0, gate)


def render(fits, width=100, system=None, content=None):
    buf = io.StringIO()
    con = make_console(buf, width=width)
    print_fit_table(fits, title="T", system=system or pc(), console=con, content=content)
    return buf.getvalue()


def header_of(out: str) -> list[str]:
    return next(x for x in out.splitlines() if x.startswith("Type")).split()


def test_fits_in_100_columns_with_long_names():
    long = "hf:someone/" + "very-long-model-name-" * 6
    out = render([fit(long), fit("image_sdxl_simple", "too_tight")])
    lines = out.splitlines()
    assert all(cell_len(line) <= 100 for line in lines), max(cell_len(x) for x in lines)
    assert "…" in out  # 長い名前は詰める
    assert "Too tight" in out and "Good" in out  # 色が無くても文字で分かる


def test_compact_when_narrow():
    out = render([fit("a")], width=70)
    assert all(cell_len(line) <= 70 for line in out.splitlines())
    assert "License" not in out


def row_of(out: str, mid: str) -> str:
    return next(line for line in out.splitlines() if f" {mid} " in line + " ")


def test_held_reason_and_uncensored_marker():
    gate = {"auto": False, "checks": [{"id": "license", "passed": False}, {"id": "gated", "passed": True}]}
    out = render([fit("x", gate=gate, content="uncensored"), fit("y", gate={"auto": True, "checks": []})])
    # Verified の列は無い(#84)。行の最後は License
    assert row_of(out, "x").rstrip().endswith("apache-2.0") and row_of(out, "y").rstrip().endswith(
        "apache-2.0"
    )
    assert "[U]" not in out  # 0.6.0 から名前の前の印ではなく Censored の列
    assert row_of(out, "x").split()[4:6] == ["N", "x"]
    assert row_of(out, "y").split()[4:6] == ["Y", "y"]  # 空にしない
    assert header_of(out)[:5] == ["Type", "Status", "Score", "Censored", "Model"]
    # 列の名前は英語のまま、説明は今の言語(#76)
    assert "Censored: N = 安全のための絞り込みを外したモデル・成人向けのモデル" in out
    with i18n.forced("en"):
        out = render([fit("x", gate=gate, content="uncensored")])
    assert "Censored: N = safety filter removed / adult model" in out


@pytest.mark.parametrize(
    "content,shown", [("sfw", False), ("all", True), ("uncensored", True), (None, False)]
)
def test_content_column_only_when_uncensored_rows_can_appear(content, shown):
    """Censored の列は --content all|uncensored のときだけ(sfw では列ごと出さない)。content を渡さない
    (Python から作る表)ときは無検閲の行があるときだけ。"""
    out = render([fit("a")], content=content)
    assert ("Censored" in header_of(out)) is shown
    assert ("Censored: N" in out) is shown
    assert (row_of(out, "a").split()[4] == "Y") is shown
    assert "Censored" in header_of(render([fit("a"), fit("u", content="uncensored")]))


def test_content_column_gives_the_name_its_width_back():
    """無検閲の行も、名前の幅を「[U] 」に取られない(sfw の行と同じ幅)。"""
    long = "hf:someone/" + "very-long-model-name-" * 6
    out = render([fit(long), fit(long + "-u", content="uncensored")], content="all")
    rows = [x for x in out.splitlines() if "●" in x]
    assert len({cell_len(r.rstrip()) for r in rows}) == 1


@pytest.mark.parametrize("width", [60, 70, 80, 90, 99, 100])
def test_content_column_kept_when_narrow(width):
    long = "hf:someone/" + "very-long-model-name-" * 6
    out = render([fit(long), fit("u", "too_tight", content="uncensored")], width=width, content="all")
    assert all(cell_len(x) <= width for x in out.splitlines()), width
    assert header_of(out)[:5] == ["Type", "Status", "Score", "Censored", "Model"]
    assert "VRAM" in header_of(out) and "Mode" in header_of(out)
    if width <= 90:  # 名前を詰めてから、Disk・Mem%・License の順に外す
        assert "Disk" not in header_of(out) and "License" not in header_of(out)


def test_offload_marked():
    row = row_of(render([fit("x", "marginal", mode="offload")]), "x")
    assert " 4.0 " in row and "GPU+RAM" in row and "50%" in row


def test_no_color_codes_when_not_terminal():
    assert "\x1b[" not in render([fit("x")])


def test_empty():
    assert "該当なし" in render([])


def test_cli_json_has_no_table(capsys):
    assert cli.main(["fit", "--json", "--vram", "8", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("{") and "Fit " not in out


def test_cli_table_real_db(capsys):
    every_license = ["--content", "all", "--all"]
    assert cli.main(["fit", "--vram", "8", "--ram", "32", "--no-cache", *every_license]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("PC: ") and lines[1].startswith("全モデル: ")
    assert "SDXL" in out and "image_sdxl_simple" not in out  # まとめた表はモデルの表示名
    db = load(use_cache=False)  # 長い ID は省略記号で詰めるので、行数で確かめる
    rows = [x for x in lines if "● " in x]
    assert len(rows) == len(build_index(db.models).members)  # 1 行 = 1 モデル
    every = ["fit", "--vram", "8", "--ram", "32", "--no-cache", *every_license, "--all-workflows"]
    assert cli.main(every) == 0
    out = capsys.readouterr().out
    rows = [x for x in out.splitlines() if "● " in x]
    assert len(rows) == len(db.models) and "image_sdxl_simple" in out  # ワークフローごとなら ID


def test_sequential_and_too_tight_show_the_figure_used():
    seq = fit("seq")
    seq.estimate.vram_gb, seq.estimate.vram_sequential_gb = 13.2, 7.2
    tight = fit("tight", "too_tight", mode="none")
    tight.estimate.vram_gb, tight.estimate.vram_offload_gb = 23.9, 9.7
    out = render([seq, tight])
    assert " 7.2 " in row_of(out, "seq") and "GPU~" in row_of(out, "seq") and "90%" in row_of(out, "seq")
    row = row_of(out, "tight")
    assert " 9.7 " in row and "121%" in row and "● Too tight" in row


def test_unknown_size_shows_question_mark():
    f = fit("nosize", "too_tight", mode="none")
    f.estimate.unknown_size = True
    f.estimate.vram_gb = f.estimate.vram_offload_gb = 0.0
    row = row_of(render([f]), "nosize")
    assert " ? " in row + " " and "0.0" not in row


def test_ram_limited_rows_say_ram():
    f = fit("ramcut", "too_tight", mode="none")
    f.estimate.vram_gb, f.estimate.vram_offload_gb, f.estimate.ram_gb = 7.0, 4.0, 40.0
    row = row_of(render([f], system=pc(12)), "ramcut")
    assert " RAM " in row and "7.0" in row


def test_no_gpu_too_tight_shows_the_minimum_vram():
    f = fit("needgpu", "too_tight", mode="none")
    nogpu = SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=32, ram_available_gb=32, gpus=[])
    out = render([f], system=nogpu)
    assert out.startswith("PC: GPU なし(CPU だけ)、メモリ 32.0 GiB")
    with i18n.forced("en"):
        assert render([f], system=nogpu).startswith("PC: no GPU (CPU only), RAM 32.0 GiB")
    row = row_of(out, "needgpu")
    assert " 4.0 " in row and "%" not in row


def test_over_100_percent_never_shows_100():
    f = fit("zero", "too_tight", mode="none")
    f.estimate.vram_offload_gb = 8.04  # 100.5%
    assert "101%" in row_of(render([f]), "zero")
    g = fit("full", "marginal")
    g.estimate.vram_gb = 7.84  # 98%: 全部 GPU に載る一番上(段階は marginal)
    assert " 98%" in row_of(render([g]), "full")


def test_measured_mark_and_status_mark():
    f = fit("m1", "perfect")
    f.estimate.basis, f.estimate.vram_gb = "measured", 3.0
    row = row_of(render([f]), "m1")
    assert row.split()[1:3] == ["●", "Perfect"] and " 3.0m " in row


@pytest.mark.parametrize(
    "lang,title,pc_line,head",
    [
        (
            "en",
            "Recommended",
            "PC: Test GPU 8.0 GiB VRAM ✓ verified GPU kind, RAM 32.0 GiB (32.0 available)",
            "Recommended: 1 of 5 (5 hidden: uncensored 3, tools 2)",
        ),
        (
            "ja",
            "おすすめ",
            "PC: Test GPU VRAM 8.0 GiB ✓ 確認済みの種類、メモリ 32.0 GiB(空き 32.0)",
            "おすすめ: 5 件中 1 件(隠した 5 件: 無検閲 3・補助ツール 2)",
        ),
    ],
)
def test_header_hidden_and_count(lang, title, pc_line, head):
    buf = io.StringIO()
    hidden = {"uncensored": 3, "tools": 2}
    with i18n.forced(lang):
        print_fit_table(
            [fit("a")], title=title, system=pc(), console=make_console(buf), matched=5, hidden=hidden
        )
    lines = buf.getvalue().splitlines()
    assert lines[0] == pc_line  # NVIDIA の専用 VRAM は同梱の実測で確認済みの種類(#108)
    assert lines[1] == head
    assert any("--uncensored" in x for x in lines)


@pytest.mark.parametrize("width", [60, 70, 80, 90, 99, 100])
def test_columns_drop_in_order_when_narrow(width):
    long = "hf:someone/" + "very-long-model-name-" * 6
    out = render([fit(long), fit("b", "too_tight")], width=width)
    assert all(cell_len(x) <= width for x in out.splitlines()), width
    header = next(x for x in out.splitlines() if x.startswith("Type"))
    # Type・Status・Score・Model は外さない(0.1.1: Type が左端、Score は Model の左)
    assert header.split()[:4] == ["Type", "Status", "Score", "Model"]
    assert "VRAM" in header and "Mode" in header
    if width <= 80:  # 狭いときは Disk・Mem%・License の順に外す
        assert "Disk" not in header and "License" not in header


def test_type_column_is_always_first():
    v = fit("v")
    v.model.modality = "video"
    header = next(x for x in render([fit("a"), v]).splitlines() if x.startswith("Type"))
    assert header.split()[:4] == ["Type", "Status", "Score", "Model"]
    # 幅 100 でも Disk を出す(Verified の列を外して空いた幅。#84)。Model は 26 桁(Use は 9 桁)
    assert "Disk" in header
    from genmodelfit.table import columns_for, model_width

    assert model_width(100, columns_for(100)) == 26


def test_middle_ellipsis_and_distinct_names():
    from genmodelfit.table import distinct_names, middle_ellipsis

    assert middle_ellipsis("abcdefghij", 10) == "abcdefghij"
    s = middle_ellipsis("utility_seedvr2_7b_int8_upscale_image", 20)
    assert len(s) == 20 and s.startswith("utility") and s.endswith("image") and "…" in s
    names = ["utility_seedvr2_7b_int8_upscale_image", "utility_seedvr2_3b_int8_upscale_image"]
    shown = distinct_names(names, 20)
    assert len(set(shown.values())) == 2 and all(len(v) <= 20 for v in shown.values())


@pytest.mark.parametrize("lang", ["en", "ja"])
def test_no_verified_column(lang):
    """#84: 表に Verified の列も、その説明も無い(gate の結果は info と gate で見る)。"""
    held = {"auto": False, "checks": [{"id": "official_workflow", "passed": False}]}
    fits = [fit("w", gate=held), fit("w2", gate={"auto": False, "checks": []}), fit("y", gate={"auto": True})]
    with i18n.forced(lang):
        out = render(fits)
    assert "Verified" not in out
    assert header_of(out)[-1] == "License"
    for mid in ("w", "w2", "y"):
        cells = row_of(out, mid).split()
        assert cells[-1] == "apache-2.0" and not {"workflow", "held", "yes"} & set(cells)
    order = (
        "Order: runs, Perfect/Good (GPU+RAM, CPU last), Marginal (faster first), official, popularity, Score."
        if lang == "en"
        else "並び: 動く → Perfect/Good(GPU+RAM・CPU は後)→ Marginal(速い Mode が先)→ 公式 → 人気 → Score。"
    )
    assert order in out


def test_recommend_excludes_tools_unless_asked(capsys):
    import json

    from genmodelfit.recommend import is_tool

    assert cli.main(["recommend", "--json", "--vram", "24", "--no-cache", "-n", "1000"]) == 0
    db = load(use_cache=False)
    ids = {m["id"] for m in json.loads(capsys.readouterr().out)["models"]}
    tools = {m.id for m in db.models if is_tool(m)}
    assert tools and not (ids & tools)
    assert (
        cli.main(["recommend", "--json", "--vram", "24", "--no-cache", "-n", "1000", "--include-tools"]) == 0
    )
    ids2 = {m["id"] for m in json.loads(capsys.readouterr().out)["models"]}
    assert ids2 & tools


def _recommend_args(*extra):
    return ["recommend", "--vram", "8", "--ram", "32", "--no-cache", "-m", "image", "-n", "1000", *extra]


def test_recommend_table_order_ignores_verified():
    """#84: recommend の表は Verified(auto)で並べない。
    JSON とライブラリ(verified_first=True)の並びは今までどおり。"""
    from dataclasses import replace

    from genmodelfit.recommend import list_key

    held = {"auto": False, "checks": [{"id": "license", "passed": False}]}
    perfect = fit("perfect_held", "perfect", gate=held)
    marginal = fit("marginal_auto", "marginal", gate={"auto": True, "checks": []}, mode="offload")
    marginal = replace(marginal, score=90.0)  # 点数が上でも、段階が先
    rows = [marginal, perfect]
    table = sorted(rows, key=lambda f: list_key(f, verified_first=False))
    library = sorted(rows, key=lambda f: list_key(f))
    assert [f.model.id for f in table] == ["perfect_held", "marginal_auto"]
    assert [f.model.id for f in library] == ["marginal_auto", "perfect_held"]


def test_recommend_table_and_json_orders_real_db(capsys):
    """同梱の一覧で: 表の行は「動く → 段階 → 点数」(auto に依らない)。
    JSON はライブラリの recommend と同じ(auto が先)。"""
    import json

    from genmodelfit.recommend import list_key, recommend, select

    assert cli.main(_recommend_args("--json")) == 0
    models = json.loads(capsys.readouterr().out)["models"]
    db = load(use_cache=False)
    system = cli._system(cli.build_parser().parse_args(_recommend_args()))
    lib = recommend(db, system, modality="image", limit=1000)  # gmf.recommend() の中身
    assert [m["id"] for m in models] == [f.model.id for f in lib]  # JSON = ライブラリの並び(変えない)
    assert [m["auto"] for m in models] == [f.auto for f in lib]
    autos = [m["auto"] for m in models]
    assert autos == sorted(autos, reverse=True)  # 動くものだけなので、auto が先

    with i18n.forced("en"):
        assert cli.main(_recommend_args()) == 0
    out = capsys.readouterr().out
    rows = [x for x in out.splitlines() if "● " in x]
    sel = select(
        db,
        system,
        modality="image",
        min_fit="marginal",
        include_tools=False,
        drop_unknown_size=True,
        limit=1000,
        verified_first=False,
    )
    assert len(rows) == len(sel.fits) == len(models)
    assert sorted(f.model.id for f in sel.fits) == sorted(m["id"] for m in models)  # 同じモデル、並びだけ違う
    want = [f.estimate.fit for f in sel.fits]
    shown = [row.split()[2].lower() for row in rows]  # 「● Good」の段階
    assert shown == [w.lower().replace("_", " ") for w in want]
    assert [f for f in sel.fits] == sorted(sel.fits, key=lambda f: list_key(f, verified_first=False))


# --- Use の列(#84: +N をやめ、まとめたワークフローの用途を並べる) ---------------------------------------


def _with_uses(mid, use, others):
    f = fit(mid)
    f.use = use
    f.workflows = [{"id": f"{mid}_{i}", "use": u} for i, u in enumerate(others)]
    return f


def _use_of(out: str, mid: str) -> str:
    cells = row_of(out, mid).split()
    return cells[cells.index(mid) + 1]


def test_use_lists_the_uses_of_the_merged_workflows():
    rows = [
        _with_uses("mixed", "gen", ["i2v", "edit", "edit"]),  # 違う用途: 行の用途、残りは重複なく USES の順
        _with_uses("own_first", "edit", ["gen", "edit", "gen"]),  # 行の用途(--task edit など)が先頭
        _with_uses("same", "gen", ["gen", "gen"]),  # 同じ用途なら 1 つ
        _with_uses("alone", "i2v", []),
        _with_uses("long", "lipsync", ["gen", "edit", "i2v"]),  # 収まらない: 行の用途を残し、境目で切って ,…
    ]
    out = render(rows)
    assert _use_of(out, "mixed") == "gen,…"  # 「gen,edit,i2v」も「gen,edit,…」も 9 桁に入らない
    assert _use_of(out, "own_first") == "edit,gen"
    assert _use_of(out, "same") == "gen"
    assert _use_of(out, "alone") == "i2v"
    assert _use_of(out, "long") == "lipsync,…"
    names = ("mixed", "own_first", "same", "alone", "long")
    assert "+" not in "".join(_use_of(out, m) for m in names)
    for lang, text in (
        ("en", "Use = workflow uses (info, --all-workflows)."),
        ("ja", "Use = ワークフローの用途"),
    ):
        with i18n.forced(lang):
            legend_out = render(rows)
        assert text in legend_out and "+N" not in legend_out


def test_use_is_the_rows_own_use_with_all_workflows():
    buf = io.StringIO()
    rows = [_with_uses("w", "edit", ["gen", "i2v"])]
    print_fit_table(rows, title="T", system=pc(), console=make_console(buf, width=100), grouped=False)
    assert _use_of(buf.getvalue(), "w") == "edit"


def test_use_text_cuts_between_uses():
    from genmodelfit.table import COLUMN_WIDTH, use_text
    from genmodelfit.tasks import USES

    w = COLUMN_WIDTH["Use"]
    assert use_text(["gen", "edit"], w) == "gen,edit"
    assert use_text(["edit", "tool"], w) == "edit,tool"
    assert use_text(["i2v", "motion"], w) == "i2v,…"
    assert use_text(["gen", "edit", "i2v", "lipsync"], 14) == "gen,edit,i2v,…"
    assert use_text(["upscale"], w) == "upscale"
    # 用途 1 つで幅を埋めるときも、区切りを残して「,…」(「lipsync…」にしない)
    assert use_text(["lipsync", "motion"], w) == "lipsync,…"
    assert use_text(["upscale", "tool"], w) == "upscale,…"
    # 一番長い用途 1 つ +「,…」が列に収まる
    longest = max(USES, key=len)
    assert len(f"{longest},…") <= w


def test_use_column_on_the_bundled_list(capsys):
    """同梱の一覧: Use に +N が無く、用途は info の材料(fit.workflows)と合う。"""
    from genmodelfit.table import row_uses

    db = load(use_cache=False)
    from genmodelfit.recommend import select

    sel = select(db, pc(24.0), content="all", include_tools=True, include_unknown_license=True)
    merged = [f for f in sel.fits if len(row_uses(f)) > 1]
    assert merged  # 用途の違うワークフローをまとめたモデルがある
    for f in sel.fits:
        want = {f.use} | {w["use"] for w in f.workflows}
        uses = row_uses(f)
        assert set(uses) == want and uses[0] == f.use and len(uses) == len(set(uses))
    assert cli.main(["fit", "--vram", "24", "--ram", "64", "--no-cache", "--lang", "en"]) == 0
    rows = [x for x in capsys.readouterr().out.splitlines() if "● " in x]
    assert rows and not any(
        re.search(r"\s(gen|edit|i2v|lipsync|motion|voice|upscale|tool)[a-z0-9,]*\+\d+\s", r) for r in rows
    )


# --- recommend の表は「並べてから -n で切る」(#84) -----------------------------------------------


def _sections(out: str) -> dict[str, list[str]]:
    """表の出力を種類ごとの行(モデルの表示)に分ける。"""
    got: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        if line.startswith("== "):
            cur = line[3:].split(":")[0].strip()
            got[cur] = []
        elif cur and "● " in line:
            got[cur].append(line)
    return got


def test_recommend_table_cuts_after_sorting(capsys):
    """-n は Verified を使わない並びで切る(選んでから並べ替えるのではない)。auto と非 auto が混ざる節では、
    表と --json の顔ぶれが違う。"""
    from genmodelfit.recommend import list_key, select
    from genmodelfit.table import columns_for, distinct_names, model_width

    db = load(use_cache=False)
    width = model_width(100, columns_for(100))
    differs = 0
    for vram in ("8", "12", "16"):
        for n in ("1", "2"):
            args = ["recommend", "--vram", vram, "--ram", "32", "--no-cache", "-n", n]
            with i18n.forced("en"):
                assert cli.main(args) == 0
            table = _sections(capsys.readouterr().out)
            assert cli.main([*args, "--json"]) == 0
            json_models = json.loads(capsys.readouterr().out)["models"]
            system = cli._system(cli.build_parser().parse_args(args))
            # 期待: 種類ごとの全部(件数で切らない)を、auto を使わない鍵で並べ、先頭の n 件
            every = select(db, system, min_fit="marginal", drop_unknown_size=True, include_tools=False).fits
            for mod, rows in table.items():
                full = sorted(
                    (f for f in every if f.model.modality == mod),
                    key=lambda f: list_key(f, verified_first=False),
                )
                top = full[: int(n)]
                shown = distinct_names([f.model_name or f.model.id for f in top], width)
                assert len(rows) == len(top), (vram, n, mod)
                for row, f in zip(rows, top, strict=True):
                    assert f" {shown[f.model_name or f.model.id]} " in row, (vram, n, mod, row)
                in_json = {m["id"] for m in json_models if m["modality"] == mod}
                in_table = {f.model.id for f in top}
                autos = [f for f in full if f.auto]
                if len(autos) >= int(n) and not all(f.auto for f in top):
                    # JSON は auto を先に n 件選ぶので、非 auto を含む表とは顔ぶれが違う
                    assert in_json != in_table, (vram, n, mod)
                    assert all(m["auto"] for m in json_models if m["modality"] == mod)
                    differs += 1
    assert differs  # auto と非 auto が混ざって顔ぶれの違う節が、少なくとも 1 つある(試験が空振りしない)


@pytest.mark.parametrize("task", ["lipsync", "edit"])
def test_use_starts_with_the_task_use(task, capsys):
    """--task で合った用途を先頭に出す(幅が足りずに「gen,…」となって、合った用途が見えなくならない)。"""
    args = [
        "recommend",
        "--vram",
        "24",
        "--ram",
        "64",
        "--no-cache",
        "--task",
        task,
        "-n",
        "20",
        "--content",
        "all",
    ]
    with i18n.forced("en"):
        assert cli.main(args) == 0
    rows = [x for x in capsys.readouterr().out.splitlines() if "● " in x]
    assert rows
    for row in rows:
        assert re.search(rf"\s{task}(,[a-z0-9]+)*(,…)?\s", row), row


# --- 表の下の「--json は auto を先に並べる」の行(#84) --------------------------------------------

JSON_FIRST = {
    "en": "--json lists auto-recommendable models first (schema.md).",
    "ja": "--json は自動でおすすめしてよいものを先に並べる(schema.md)。",
}


@pytest.mark.parametrize("lang", ["en", "ja"])
@pytest.mark.parametrize(
    "args,shown",
    [
        (["recommend", "-n", "2"], True),
        (["recommend", "-m", "image"], True),
        (["recommend", "-n", "2", "--auto-only"], False),  # 表も JSON も auto だけで、並びが同じ
        (["fit"], False),
        (["search", "sdxl"], False),
    ],
)
def test_json_first_line_only_for_recommend(lang, args, shown, capsys):
    with i18n.forced(lang):
        assert cli.main([*args, "--vram", "8", "--ram", "32", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "● " in out  # 表が出ている(説明の行は表があるときだけ)
    assert (JSON_FIRST[lang] in out) is shown
    other = JSON_FIRST["ja" if lang == "en" else "en"]
    assert other not in out
