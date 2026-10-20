import io

import pytest
from rich.cells import cell_len

from genmodelfit import cli
from genmodelfit.db import load
from genmodelfit.groups import build_index
from genmodelfit.models import Estimate, Gpu, ModelEntry, SystemInfo
from genmodelfit.recommend import Fit
from genmodelfit.table import make_console, print_fit_table

HEADER = [
    "Status",
    "Model",
    "Type",
    "Use",
    "Prec",
    "Score",
    "VRAM",
    "Mem%",
    "Mode",
    "Disk",
    "License",
    "Auto",
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


def render(fits, width=100, system=None):
    buf = io.StringIO()
    print_fit_table(fits, title="T", system=system or pc(), console=make_console(buf, width=width))
    return buf.getvalue()


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
    assert row_of(out, "x").rstrip().endswith("license") and row_of(out, "y").rstrip().endswith("yes")
    assert "[U] x" in out and "[U] = uncensored" in out


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
    assert cli.main(["fit", "--vram", "8", "--ram", "32", "--no-cache", "--content", "all"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[0].startswith("PC: ") and lines[1].startswith("All models: ")
    assert "SDXL" in out and "image_sdxl_simple" not in out  # まとめた表はモデルの表示名
    db = load(use_cache=False)  # 長い ID は省略記号で詰めるので、行数で確かめる
    rows = [x for x in lines if x.startswith("● ")]
    assert len(rows) == len(build_index(db.models).members)  # 1 行 = 1 モデル
    every = ["fit", "--vram", "8", "--ram", "32", "--no-cache", "--content", "all", "--all-workflows"]
    assert cli.main(every) == 0
    out = capsys.readouterr().out
    rows = [x for x in out.splitlines() if x.startswith("● ")]
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
    assert out.startswith("PC: no GPU (CPU only), RAM 32.0 GiB")
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
    assert row.startswith("● Perfect") and " 3.0m " in row


def test_header_hidden_and_count():
    buf = io.StringIO()
    hidden = {"uncensored": 3, "tools": 2}
    print_fit_table(
        [fit("a")], title="Recommended", system=pc(), console=make_console(buf), matched=5, hidden=hidden
    )
    lines = buf.getvalue().splitlines()
    assert lines[0] == "PC: Test GPU 8.0 GiB VRAM, RAM 32.0 GiB (32.0 available)"
    assert lines[1] == "Recommended: 1 of 5 (5 hidden: uncensored 3, tools 2)"
    assert any("--content all" in x for x in lines)


@pytest.mark.parametrize("width", [60, 70, 80, 90, 99, 100])
def test_columns_drop_in_order_when_narrow(width):
    long = "hf:someone/" + "very-long-model-name-" * 6
    out = render([fit(long), fit("b", "too_tight")], width=width)
    assert all(cell_len(x) <= width for x in out.splitlines()), width
    header = next(x for x in out.splitlines() if x.startswith("Status"))
    assert "Model" in header and "VRAM" in header and "Mode" in header
    if width <= 80:  # 狭いときは Disk・Mem%・License の順に外す
        assert "Disk" not in header and "License" not in header
    if width == 100:  # 1 種類だけの表なので Type は無い
        assert header.split() == [c for c in HEADER if c != "Type"]


def test_type_column_only_when_types_mix():
    v = fit("v")
    v.model.modality = "video"
    header = next(x for x in render([fit("a"), v]).splitlines() if x.startswith("Status"))
    assert "Type" in header and "Disk" not in header  # 名前の幅を残すため Disk を外す


def test_middle_ellipsis_and_distinct_names():
    from genmodelfit.table import distinct_names, middle_ellipsis

    assert middle_ellipsis("abcdefghij", 10) == "abcdefghij"
    s = middle_ellipsis("utility_seedvr2_7b_int8_upscale_image", 20)
    assert len(s) == 20 and s.startswith("utility") and s.endswith("image") and "…" in s
    names = ["utility_seedvr2_7b_int8_upscale_image", "utility_seedvr2_3b_int8_upscale_image"]
    shown = distinct_names(names, 20)
    assert len(set(shown.values())) == 2 and all(len(v) <= 20 for v in shown.values())


def test_held_label_is_short():
    gate = {"auto": False, "checks": [{"id": "official_workflow", "passed": False}]}
    assert row_of(render([fit("w", gate=gate)]), "w").rstrip().endswith(" workflow")
    assert row_of(render([fit("w2", gate={"auto": False, "checks": []})]), "w2").rstrip().endswith(" held")


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
