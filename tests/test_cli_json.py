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
    "license",
    "source",
    "total_size_gb",
    "score",
    "fit",
    "estimate",
    "gate",
}
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
RECOMMEND_FILTERS = {"modality", "content", "min_fit", "limit", "auto_only", "include_tools"}
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
    assert item["fit"] == item["estimate"]["fit"]
    check_gate(item["gate"])


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
    assert set(d) == ENVELOPE | {"system", "database", "filters", "models"}
    check_system(d)
    assert set(d["database"]) == DATABASE and d["database"]["origin"] == "bundled"
    assert set(d["filters"]) == RECOMMEND_FILTERS
    assert d["filters"]["content"] == "sfw" and d["filters"]["limit"] == 5  # 実際に使った値を出す
    assert 0 < len(d["models"]) <= 5
    for m in d["models"]:
        check_fit(m)
        assert m["fit"] != "too_tight" and m["content"] == "sfw"
    d = call(["recommend", *HW, "-n", "500", "--content", "all", "--limits"], cache)
    assert d["filters"]["content"] == "all"
    assert any(m["content"] == "uncensored" for m in d["models"])
    assert any(m["estimate"]["limits"] for m in d["models"])
    for m in d["models"]:
        check_fit(m)


def test_fit(cache):
    d = call(["fit", *HW], cache)
    assert set(d) == ENVELOPE | {"system", "database", "models"}
    sfw = sum(1 for m in load_db(use_cache=False).models if m.content == "sfw")
    assert len(d["models"]) == sfw < d["database"]["models"]  # 既定は sfw
    for m in d["models"]:
        check_fit(m)
    every = call(["fit", *HW, "--content", "all"], cache)
    assert len(every["models"]) == every["database"]["models"]


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
