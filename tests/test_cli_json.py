"""`--json` の形が docs/schema.md と合うか(組み込み側は別プロセスで呼んで stdout を読む)。

キーの組は docs/schema.md から写した。形を変えたら両方を直す。
"""

from __future__ import annotations

import json

import pytest
from clihelp import run_cli

from genmodelfit import load_db

HW = ["--vram", "8", "--ram", "32", "--no-cache"]

ENVELOPE = {"genmodelfit", "command"}
SYSTEM = {
    "os",
    "cpu",
    "cpu_cores",
    "ram_total_gb",
    "ram_available_gb",
    "gpus",
    "best_gpu",
    "disk_free_gb",
    "overrides",
}
GPU = {"name", "vendor", "vram_total_gb", "vram_free_gb", "driver", "backend"}
DATABASE = {"origin", "generated_at", "models", "measurements"}
FIT = {
    "id",
    "name",
    "modality",
    "tasks",
    "content",
    "source",
    "fit_level",
    "fit_label",
    "run_mode",
    "run_mode_label",
    "score",
    "score_components",
    "memory_kind",
    "memory_required_gb",
    "memory_available_gb",
    "utilization_pct",
    "ram_required_gb",
    "disk_size_gb",
    "total_size_gb",
    "license",
    "licenses",
    "auto",
    "held_reasons",
    "estimate_confidence",
    "notes",
    "files_count",
    "estimate",
    "gate",
}
SCORE_COMPONENTS = {"fit", "headroom", "readiness", "basis", "popularity"}
ESTIMATE = {
    "vram_gb",
    "vram_sequential_gb",
    "vram_offload_gb",
    "ram_gb",
    "disk_gb",
    "basis",
    "measured_on",
    "fit",
    "run_mode",
    "reasons",
    "settings",
    "limits",
    "unknown_size",
}
LIMITS = {"level", "at_cap"}  # max_side_px / max_frames / max_seconds は種類で変わる
GATE = {"auto", "held_by", "comfyui_version", "checks"}
CHECK_IDS = [
    "license",
    "gated",
    "official_workflow",
    "custom_nodes",
    "comfyui_version",
    "url_reachable",
    "sha256",
    "content",
    "size_known",
]
FIT_FILTERS = {"modality", "content", "min_fit", "license", "limit"}
RECOMMEND_FILTERS = FIT_FILTERS | {"auto_only", "include_tools"}
SEARCH_FILTERS = FIT_FILTERS | {"query"}
LIST_KEYS = ENVELOPE | {"system", "database", "filters", "matched", "hidden", "models"}
HIDDEN_KEYS = {
    "modality",
    "uncensored",
    "sfw",
    "tools",
    "license",
    "size_unknown",
    "too_tight",
    "below_min_fit",
    "held",
}
MODEL_ENTRY = {
    "id",
    "name",
    "modality",
    "tasks",
    "source",
    "files",
    "template",
    "params_b",
    "precision",
    "license",
    "license_url",
    "base_model",
    "content",
    "content_evidence",
    "discovered_from",
    "updated",
    "homepage",
    "downloads",
    "likes",
    "total_size_bytes",
    "license_name",
    "license_evidence",
    "files_complete",
    "size_floor_bytes",
}
MODEL_FILE = {
    "name",
    "url",
    "sha256",
    "size_bytes",
    "directory",
    "repo_id",
    "license",
    "license_name",
    "gated",
    "role",
    "license_from",
    "repo_license",
    "license_url",
    "license_path",
    "license_evidence",
    "head_status",
    "head_checked_at",
}
TEMPLATE = {
    "name",
    "title",
    "category",
    "min_comfyui_version",
    "open_source",
    "requires_custom_nodes",
    "usage",
    "date",
    "size_bytes",
}
MEASUREMENT = {
    "model_id",
    "gpu_name",
    "gpu_vram_gb",
    "peak_vram_gb",
    "peak_ram_gb",
    "seconds",
    "settings",
    "driver",
    "comfyui_version",
    "offload",
    "source",
    "date",
    "note",
    "outcome",
    "standard",
    "vram_metric",
}
MEASUREMENT_SETTINGS = {"width", "height", "frames", "seconds", "steps", "precision", "batch"}


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return tmp_path_factory.mktemp("cache")


def call(args: list[str], cache, code: int = 0) -> dict:
    r = run_cli([*args, "--json"], cache)
    assert r.returncode == code, r.stderr.decode("utf-8", "replace")
    assert r.stderr == b""  # 注意や途中経過も出さない
    text = r.stdout.decode("utf-8")  # 厳密に UTF-8
    d = json.loads(text)  # stdout は JSON だけ(前後に何も混ざらない)
    assert isinstance(d, dict) and set(d) >= ENVELOPE
    return d


def check_system(d: dict) -> None:
    s = d["system"]
    assert set(s) == SYSTEM
    for g in s["gpus"]:
        assert set(g) == GPU
    assert s["best_gpu"] is None or set(s["best_gpu"]) == GPU
    assert "vram" in s["overrides"] and "ram" in s["overrides"]
    assert s["best_gpu"]["vram_total_gb"] == 8 and s["ram_total_gb"] == 32


def check_fit(item: dict) -> None:
    assert set(item) == FIT
    check_estimate(item["estimate"])
    check_gate(item["gate"])
    e, g = item["estimate"], item["gate"]
    assert item["fit_level"] == e["fit"] and item["run_mode"] == e["run_mode"]
    assert item["fit_label"] == {"perfect": "Perfect", "good": "Good", "marginal": "Marginal"}.get(
        item["fit_level"], "Too tight"
    )
    assert item["run_mode_label"] in ("GPU", "GPU~", "GPU+RAM", "CPU", "-")
    assert item["estimate_confidence"] == e["basis"] and item["notes"] == e["reasons"]
    assert item["auto"] == g["auto"] and item["held_reasons"] == g["held_by"]
    assert set(item["score_components"]) == SCORE_COMPONENTS
    assert 0 <= item["score"] <= 100 and all(0 <= v <= 100 for v in item["score_components"].values())
    assert item["memory_kind"] in ("vram", "ram")
    if item["utilization_pct"] is not None:
        pct = 100 * item["memory_required_gb"] / item["memory_available_gb"]
        assert abs(item["utilization_pct"] - pct) < 0.2
    assert isinstance(item["licenses"], list) and item["files_count"] >= 0
    assert item["ram_required_gb"] == (None if e["unknown_size"] else e["ram_gb"])


def check_order(models: list[dict]) -> None:
    """動くもの → auto → 点数 → ID の順。"""
    keys = [(m["fit_level"] == "too_tight", not m["auto"], -m["score"], m["id"]) for m in models]
    assert keys == sorted(keys)


def check_estimate(e: dict) -> None:
    assert set(e) == ESTIMATE
    assert e["basis"] in ("measured", "estimated")
    assert e["fit"] in ("perfect", "good", "marginal", "too_tight")
    assert e["run_mode"] in ("gpu", "offload", "cpu", "none")
    if e["limits"] is not None:
        assert set(e["limits"]) >= LIMITS
        assert set(e["limits"]) - LIMITS <= {"max_side_px", "max_frames", "max_seconds"}


def check_gate(g: dict) -> None:
    assert set(g) == GATE
    assert [c["id"] for c in g["checks"]] == CHECK_IDS
    assert all(set(c) >= {"id", "passed", "detail"} for c in g["checks"])


def test_system(cache):
    d = call(["system", "--vram", "8", "--ram", "32"], cache)
    assert d["command"] == "system" and set(d) == ENVELOPE | {"system"}
    check_system(d)


def test_recommend(cache):
    d = call(["recommend", *HW, "-n", "5"], cache)
    assert set(d) == LIST_KEYS
    check_system(d)
    assert set(d["database"]) == DATABASE and d["database"]["origin"] == "bundled"
    assert set(d["filters"]) == RECOMMEND_FILTERS
    assert d["filters"]["content"] == "sfw" and d["filters"]["limit"] == 5  # 実際に使った値を出す
    assert d["filters"]["min_fit"] == "marginal" and d["filters"]["license"] is None
    assert 0 < len(d["models"]) <= 5 < d["matched"]
    assert set(d["hidden"]) <= HIDDEN_KEYS and d["hidden"]["uncensored"] == 39
    for m in d["models"]:
        check_fit(m)
        assert m["fit_level"] != "too_tight" and m["content"] == "sfw"
    check_order(d["models"])
    assert d["models"][0]["auto"]  # 自動でおすすめに出せるものが先
    d = call(["recommend", *HW, "-n", "500", "--content", "all", "--limits"], cache)
    assert d["filters"]["content"] == "all" and "uncensored" not in d["hidden"]
    assert any(m["content"] == "uncensored" for m in d["models"])
    assert any(m["estimate"]["limits"] for m in d["models"])
    for m in d["models"]:
        check_fit(m)
    check_order(d["models"])
    assert len(d["models"]) == d["matched"]


def test_fit(cache):
    d = call(["fit", *HW], cache)
    assert set(d) == LIST_KEYS
    assert set(d["filters"]) == FIT_FILTERS and d["filters"]["min_fit"] is None
    sfw = sum(1 for m in load_db(use_cache=False).models if m.content == "sfw")
    assert len(d["models"]) == sfw == d["matched"] < d["database"]["models"]  # 既定は sfw
    assert d["hidden"] == {"uncensored": d["database"]["models"] - sfw}
    for m in d["models"]:
        check_fit(m)
    check_order(d["models"])
    every = call(["fit", *HW, "--content", "all"], cache)
    assert len(every["models"]) == every["database"]["models"] and every["hidden"] == {}


def test_min_fit_and_license(cache):
    good = call(["fit", *HW, "--min-fit", "good"], cache)
    assert good["models"] and all(m["fit_level"] in ("perfect", "good") for m in good["models"])
    assert good["hidden"]["too_tight"] and good["hidden"]["below_min_fit"]
    tight = call(["recommend", *HW, "--min-fit", "too_tight", "-n", "500"], cache)
    assert any(m["fit_level"] == "too_tight" for m in tight["models"])
    assert all(m["estimate"]["unknown_size"] is False for m in tight["models"])  # 大きさ不明は出さない
    lic = call(["fit", *HW, "--license", "Apache-2.0, MIT"], cache)
    assert lic["filters"]["license"] == ["apache-2.0", "mit"] and lic["hidden"]["license"]
    assert lic["models"] and all(set(m["licenses"]) <= {"apache-2.0", "mit"} for m in lic["models"])
    every = call(["fit", *HW], cache)
    unknown = [m for m in every["models"] if None in m["licenses"]]
    assert unknown and not {m["id"] for m in unknown} & {m["id"] for m in lic["models"]}  # 不明は合わない
    r = run_cli(["fit", *HW, "--license", " , "], cache)
    assert r.returncode == 2


def test_search(cache):
    d = call(["search", "FLUX", *HW], cache)
    assert d["command"] == "search" and set(d) == LIST_KEYS
    assert set(d["filters"]) == SEARCH_FILTERS and d["filters"]["query"] == "FLUX"
    assert d["models"] and all("flux" in (m["id"] + m["name"]).lower() for m in d["models"])
    assert all(m["content"] == "sfw" for m in d["models"]) and d["hidden"].get("uncensored")
    for m in d["models"]:
        check_fit(m)
    check_order(d["models"])
    every = call(["search", "flux", *HW, "--content", "all"], cache)
    assert len(every["models"]) == len(d["models"]) + d["hidden"]["uncensored"]
    none = call(["search", "no_such_model_xyz", *HW], cache)  # 0 件でも終了コード 0
    assert none["models"] == [] and none["matched"] == 0 and none["hidden"] == {}


def test_info(cache):
    d = call(["info", "image_sdxl_simple", *HW], cache)
    assert set(d) == ENVELOPE | {"system", "database", "model", "estimate", "gate", "measurements"}
    m = d["model"]
    assert set(m) == MODEL_ENTRY
    assert all(set(f) == MODEL_FILE for f in m["files"])
    assert m["template"] is None or set(m["template"]) == TEMPLATE
    check_estimate(d["estimate"])
    check_gate(d["gate"])
    assert d["measurements"], "image_sdxl_simple には実測がある"
    for x in d["measurements"]:
        assert set(x) == MEASUREMENT and set(x["settings"]) == MEASUREMENT_SETTINGS


def test_info_uncensored_by_id(cache):
    fit = call(["fit", *HW, "--content", "uncensored", "-n", "1"], cache)
    mid = fit["models"][0]["id"]
    d = call(["info", mid, *HW], cache)  # 既定の sfw でも、ID を指定すれば出る
    assert d["model"]["id"] == mid and d["model"]["content"] == "uncensored"


def test_gate(cache):
    d = call(["gate", "--no-cache"], cache)
    assert set(d) == ENVELOPE | {"database", "policy", "filters", "summary", "models"}
    policy = {"license_allowlist", "comfyui_version", "allow_uncensored", "max_head_age_days"}
    assert set(d["policy"]) == policy
    assert set(d["filters"]) == {"modality", "only_auto", "only_held"}
    assert set(d["summary"]) == {"models", "auto", "held", "failing", "first_failing"}
    assert len(d["models"]) == d["summary"]["models"] == d["database"]["models"]
    for m in d["models"]:
        assert set(m) == GATE | {"id"}


def test_error_is_json(cache):
    d = call(["info", "no_such_model_xyz", *HW], cache, code=1)
    assert set(d) == ENVELOPE | {"error"} and "no_such_model_xyz" in d["error"]


def test_info_ambiguous_is_error_with_candidates(cache):
    d = call(["info", "sdxl", *HW], cache, code=1)
    assert set(d) == ENVELOPE | {"error", "candidates"}
    ids = [c["id"] for c in d["candidates"]]
    assert "image_sdxl_simple" in ids and len(ids) >= 2
    assert all(set(c) == {"id", "name", "content"} and c["content"] == "sfw" for c in d["candidates"])
    every = call(["info", "sdxl", *HW, "--content", "all"], cache, code=1)
    assert any(c["content"] == "uncensored" for c in every["candidates"])
    # 無検閲にしか合わないものは黙って選ばない
    d = call(["info", "Qwen-Image-2.1", *HW], cache, code=1)
    assert set(d) == ENVELOPE | {"error"} and "--content all" in d["error"]
