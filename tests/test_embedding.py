"""アプリに組み込んで使うときの約束(公開 API・ネットに出ない・何も表示しない・誤った引数・決まった結果)。"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

import genmodelfit as gmf
from genmodelfit.models import FIT_LEVELS, Gpu, SystemInfo
from genmodelfit.recommend import evaluate


def pc(vram: float, ram: float = 32.0) -> SystemInfo:
    gpus = [Gpu(name="Test GPU", vendor="nvidia", vram_total_gb=vram, backend="cuda")] if vram else []
    return SystemInfo(os="test", cpu="test", cpu_cores=8, ram_total_gb=ram, ram_available_gb=ram, gpus=gpus)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(scope="module")
def db():
    return gmf.load_db(use_cache=False)


def test_all_names_import():
    for name in gmf.__all__:
        assert hasattr(gmf, name), name
    ns: dict = {}
    exec(f"from genmodelfit import {', '.join(gmf.__all__)}", ns)  # 「from … import」でも取れる
    assert all(name in ns for name in gmf.__all__)


def test_recommend_with_constructed_system(db):
    fits = gmf.recommend(system_info=pc(8), db=db, limit=5)
    assert 0 < len(fits) <= 5
    assert all(f.estimate.fit in FIT_LEVELS[:3] for f in fits)
    cpu = gmf.recommend(system_info=pc(0, 16), db=db, limit=None)  # GPU の無い PC
    assert all(f.estimate.run_mode == "cpu" for f in cpu)


def test_api_writes_nothing(db, capfd):
    gmf.recommend(system_info=pc(8), db=db)
    gmf.recommend(system_info=pc(0), db=db, content="all", limit=None)
    evaluate(db, pc(12), content="all", with_limits=True)
    gmf.info("image_sdxl_simple", db=db)
    gmf.load_db(use_cache=False)
    gmf.system(vram_gb=8, ram_gb=32)  # nvidia-smi を呼んでも、その出力を漏らさない
    out, err = capfd.readouterr()
    assert out == "" and err == ""


def test_no_network(tmp_path):
    """import と主な関数でネットに出ない(別プロセスで、つなぐ所を全部ふさいでから import する)。"""
    script = textwrap.dedent(
        """
        import socket, sys

        def deny(*a, **k):
            raise AssertionError("network used")

        socket.socket.connect = deny
        socket.socket.connect_ex = deny
        socket.create_connection = deny
        socket.getaddrinfo = deny
        import requests
        requests.Session.request = deny

        import genmodelfit as gmf
        from genmodelfit.models import Gpu, SystemInfo
        from genmodelfit.recommend import evaluate

        pc = SystemInfo(os="t", cpu="t", cpu_cores=8, ram_total_gb=32, ram_available_gb=32,
                        gpus=[Gpu(name="T", vendor="nvidia", vram_total_gb=8, backend="cuda")])
        db = gmf.load_db()
        gmf.recommend(system_info=pc)
        gmf.recommend(system_info=pc, db=db, content="all", limit=None)
        evaluate(db, pc, content="all", with_limits=True)
        gmf.info("image_sdxl_simple")
        sys.stdout.write("OK")
        """
    )
    from clihelp import child_env

    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=child_env(tmp_path / "cache"),
        timeout=120,
        check=False,
    )
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")
    assert r.stdout == b"OK" and r.stderr == b""


@pytest.mark.parametrize(
    ("kwargs", "word"),
    [
        ({"modality": "picture"}, "modality"),
        ({"content": "nsfw"}, "content"),
        ({"content": "SFW"}, "content"),
        ({"min_fit": "great"}, "min_fit"),
        ({"min_fit": "too_tight"}, "min_fit"),
    ],
)
def test_invalid_arguments_raise_value_error(db, kwargs, word):
    with pytest.raises(ValueError, match=word) as ei:
        gmf.recommend(system_info=pc(8), db=db, **kwargs)
    assert "使えるのは" in str(ei.value)  # 使える値を文に入れる


def test_evaluate_rejects_unknown_content(db):
    with pytest.raises(ValueError, match="content"):
        evaluate(db, pc(8), content="adult")


def test_default_content_is_sfw(db):
    default = gmf.recommend(system_info=pc(24, 64), db=db, limit=None)
    assert default and all(f.model.content == "sfw" for f in default)
    every = gmf.recommend(system_info=pc(24, 64), db=db, limit=None, content="all")
    assert any(f.model.content == "uncensored" for f in every)
    only = gmf.recommend(system_info=pc(24, 64), db=db, limit=None, content="uncensored")
    assert only and all(f.model.content == "uncensored" for f in only)
    assert all(f.model.content == "sfw" for f in evaluate(db, pc(8)))
    # ID を指定すれば無検閲のモデルも引ける
    assert gmf.info(only[0].model.id, db=db) is not None


def test_results_are_deterministic(db):
    a = [f.to_dict() for f in gmf.recommend(system_info=pc(8), db=db, limit=None, content="all")]
    b = [f.to_dict() for f in gmf.recommend(system_info=pc(8), db=db, limit=None, content="all")]
    assert a == b
    c = [f.to_dict() for f in gmf.recommend(system_info=pc(8), db=gmf.load_db(use_cache=False), limit=None)]
    assert c == [x for x in a if x["content"] == "sfw"]
