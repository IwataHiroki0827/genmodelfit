import os
from datetime import datetime, timezone

import pytest

# 同梱の一覧の HEAD は 2026-10-20 に確かめた。gate の「確認が古い」を、
# テストを動かす日によらず決まった結果にする
SNAPSHOT_NOW = datetime(2026, 10, 21, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _no_real_hardware(monkeypatch):
    """既定の試験では、本物の nvidia-smi・PowerShell・lspci・レジストリを呼ばない
    (hardware の外への出口を塞ぐ)。
    出力を使う試験は tests/fixtures/hardware のフィクスチャで差し替える。"""
    from genmodelfit import hardware

    monkeypatch.setattr(hardware, "_run", lambda args, timeout: None)
    monkeypatch.setattr(hardware, "_reg_display_entries", lambda: None)
    monkeypatch.setattr(hardware, "_reg_paging_files", lambda: None)


@pytest.fixture(autouse=True)
def _fixed_gate_clock(monkeypatch):
    from genmodelfit import gate

    monkeypatch.setattr(gate, "_now", lambda: SNAPSHOT_NOW)
    monkeypatch.delenv("COLUMNS", raising=False)  # 表の幅を、テストを動かす端末によらず決める


def pytest_collection_modifyitems(config, items):
    """network の印のテストは GENMODELFIT_NETWORK_TESTS=1 のときだけ動かす。"""
    if os.environ.get("GENMODELFIT_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="実ネットワークのテスト(GENMODELFIT_NETWORK_TESTS=1 で実行)")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
