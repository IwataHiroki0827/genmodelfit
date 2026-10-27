import os
from datetime import datetime, timezone

import pytest

# 同梱の一覧の HEAD は 2026-10-20 に確かめた。gate の「確認が古い」を、
# テストを動かす日によらず決まった結果にする
SNAPSHOT_NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


@pytest.fixture(scope="session", autouse=True)
def _isolated_cache_session(tmp_path_factory):
    """利用者の本物のキャッシュ(この PC の bench の実測・取ってきた一覧)を読まない(#90)。
    module の fixture(同梱の一覧を読む db など)は関数の fixture より先に作られるので、
    まず全体で空のフォルダにする。"""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
        yield


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """試験ごとに空のキャッシュ(書いたものが次の試験に残らない)。"""
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))


@pytest.fixture(autouse=True)
def _no_real_hardware(monkeypatch):
    """既定の試験では、本物の nvidia-smi・PowerShell・lspci・レジストリを呼ばない
    (hardware の外への出口を塞ぐ)。
    出力を使う試験は tests/fixtures/hardware のフィクスチャで差し替える。"""
    from genmodelfit import bench, hardware

    monkeypatch.setattr(hardware, "_run", lambda args, timeout: None)
    # bench は nvidia-smi を自分で動かす(_smi_query・使用率の監視)。本物の GPU の使用率・空きを読まない(#90。
    # 共用の PC で GPU が使われていると、bench を通す試験が止まった)。使う試験は smi_gpus などを差し替える
    monkeypatch.setattr(bench, "_nvidia_smi", lambda: None)
    monkeypatch.setattr(bench, "find_server_process", lambda url: None)
    monkeypatch.setattr(hardware, "_reg_display_entries", lambda: None)
    monkeypatch.setattr(hardware, "_reg_paging_files", lambda: None)


@pytest.fixture(autouse=True)
def _pin_language(monkeypatch):
    """表示の言語を ja に固定する(機械の言語に左右されないように。#73)。
    子プロセスで CLI を動かす試験にも効くよう環境変数で渡し、set_language・--lang の状態も毎回戻す。"""
    from genmodelfit import i18n

    monkeypatch.setenv(i18n.ENV_VAR, "ja")
    monkeypatch.setattr(i18n, "_api_language", None)
    with i18n.forced(None):  # --lang・weekly の固定(ContextVar)も毎回外して始め、終われば戻す
        yield


@pytest.fixture(autouse=True)
def _fixed_gate_clock(monkeypatch):
    from genmodelfit import gate

    monkeypatch.setattr(gate, "_now", lambda: SNAPSHOT_NOW)
    monkeypatch.delenv("COLUMNS", raising=False)  # 表の幅を、テストを動かす端末によらず決める
    monkeypatch.delenv("GENMODELFIT_COMFYUI_DIR", raising=False)  # download の既定の置き場所(#75)


def pytest_collection_modifyitems(config, items):
    """network の印のテストは GENMODELFIT_NETWORK_TESTS=1 のときだけ動かす。"""
    if os.environ.get("GENMODELFIT_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="実ネットワークのテスト(GENMODELFIT_NETWORK_TESTS=1 で実行)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
