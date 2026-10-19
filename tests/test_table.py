import io

import pytest
from rich.cells import cell_len

from genmodelfit import cli
from genmodelfit.db import load
from genmodelfit.models import Estimate, Gpu, ModelEntry, SystemInfo
from genmodelfit.recommend import Fit
from genmodelfit.table import make_console, print_fit_table


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


def test_held_reason_and_uncensored_marker():
    gate = {"auto": False, "checks": [{"id": "license", "passed": False}, {"id": "gated", "passed": True}]}
    out = render([fit("x", gate=gate, content="uncensored"), fit("y", gate={"auto": True, "checks": []})])
    assert "held:license" in out and "yes" in out
    assert "[U] x" in out and "[U] = uncensored" in out


def test_offload_marked():
    out = render([fit("x", "marginal", mode="offload")])
    assert "4.0*" in out


def test_no_color_codes_when_not_terminal():
    assert "\x1b[" not in render([fit("x")])


def test_empty():
    assert "該当なし" in render([])


def test_cli_json_has_no_table(capsys):
    assert cli.main(["fit", "--json", "--vram", "8", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert out.lstrip().startswith("{") and "Fit " not in out


def test_cli_table_real_db(capsys):
    assert cli.main(["fit", "--vram", "8", "--ram", "32", "--no-cache"]) == 0
    out = capsys.readouterr().out
    assert "image_sdxl_simple" in out and "PC: " in out
    assert all(m.id in out for m in load(use_cache=False).models)


def test_sequential_and_too_tight_show_the_figure_used():
    seq = fit("seq")
    seq.estimate.vram_gb, seq.estimate.vram_sequential_gb = 13.2, 7.2
    tight = fit("tight", "too_tight", mode="none")
    tight.estimate.vram_gb, tight.estimate.vram_offload_gb = 23.9, 9.7
    out = render([seq, tight])
    assert "7.2~" in out and "+0.8" in out
    assert "9.7*" in out and "-1.7" in out
