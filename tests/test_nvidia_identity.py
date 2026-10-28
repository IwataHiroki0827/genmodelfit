# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""NVIDIA の GPU の見分け方(同じ GPU かどうか)の試験。

NVIDIA は PCI の型番 ID を読まず、名前と VRAM で比べる作り(hardware.local_pci_id は AMD・Intel だけ。
nvidia-smi の出力にも型番 ID を求めない)。その作りのまま、次が成り立つことを確かめる:
- nvidia-smi の行から作った GPU は型番 ID を持たない(持たないことが前提の照合になっている)
- 片方だけ型番 ID を持つときは名前で比べ、両方にあれば ID で比べる
- nvidia-smi・WMI(画面の部品)・Linux の名前の書き方の違いで、同じ GPU が別扱いにならない
- 報告される VRAM の量の違い(公称・Windows・Linux)で、同じ GPU が別扱いにならない
業務用(RTX Ada Generation)とノート用の名前も扱う。
"""

from __future__ import annotations

import json

import pytest

from genmodelfit import hardware
from genmodelfit.estimate import gpu_key, same_gpu, same_vram
from genmodelfit.hardware import QUERY, cc_from_name, classify_gpu, nvidia_name_known, parse_nvidia_smi
from genmodelfit.models import Gpu

# nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version,compute_cap の行(MiB)
NVSMI_RTX6000_ADA = "NVIDIA RTX 6000 Ada Generation, 49140, 47900, 1240, 576.88, 8.9\r\n"

ADA_WORKSTATION = [
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 2000E Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 4000 SFF Ada Generation",
    "NVIDIA RTX 4500 Ada Generation",
    "NVIDIA RTX 5000 Ada Generation",
    "NVIDIA RTX 5880 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA RTX 2000 Ada Generation Laptop GPU",
    "NVIDIA RTX 3000 Ada Generation Laptop GPU",
    "NVIDIA RTX 3500 Ada Generation Laptop GPU",
    "NVIDIA RTX 4000 Ada Generation Laptop GPU",
    "NVIDIA RTX 5000 Ada Generation Laptop GPU",
]


def _from_nvsmi(text: str) -> Gpu:
    gpus = parse_nvidia_smi(text, QUERY)
    assert len(gpus) == 1
    return classify_gpu(gpus[0], windows_driver=True, comfy_windows=True)


def test_nvidia_smi_gpu_has_no_pci_id_and_is_judged_by_name():
    g = _from_nvsmi(NVSMI_RTX6000_ADA)
    assert g.pci_id is None  # 作りどおり(NVIDIA は名前で比べる)
    assert (g.name, g.vram_total_gb, g.arch, g.compute_capability) == (
        "NVIDIA RTX 6000 Ada Generation",
        47.99,
        "ada",
        "8.9",
    )
    assert g.backend == "cuda" and (g.bf16, g.fp8) == (True, True)


def test_local_pci_id_is_never_read_for_nvidia(monkeypatch):
    called = []
    monkeypatch.setattr(hardware, "detect_other_gpus", lambda: called.append(1) or ([], []))
    assert hardware.local_pci_id("nvidia", "NVIDIA RTX 6000 Ada Generation") is None
    assert not called  # 画面の部品(WMI・lspci)も読まない


@pytest.mark.parametrize("name", ADA_WORKSTATION)
def test_ada_workstation_and_laptop_names_are_known(name):
    assert nvidia_name_known(name)
    assert cc_from_name(name) == (8, 9)


def test_wmi_reads_the_pci_id_of_an_nvidia_card_but_only_as_another_adapter():
    """WMI(画面の部品)は NVIDIA の部品にも型番 ID を付ける。判定に使う GPU(nvidia-smi)とは別の欄で、
    差し場所(PCI のバス番号)は持たない。"""
    wmi = json.dumps(
        [
            {
                "Name": "NVIDIA RTX 6000 Ada Generation",
                "AdapterRAM": 4293918720,
                "PNPDeviceID": "PCI\\VEN_10DE&DEV_26B1&SUBSYS_16A110DE&REV_A1\\4&00000000&0&0008",
                "ConfigManagerErrorCode": 0,
            }
        ]
    )
    adapters, _ = hardware.parse_wmi_video(wmi)
    assert len(adapters) == 1 and adapters[0].vendor == "nvidia"
    assert adapters[0].pci_id == "10de:26b1"
    assert "0008" not in json.dumps(vars(adapters[0]), default=str)  # 差し場所は持たない


def _gpu(name: str, vram: float, pci: str | None = None) -> Gpu:
    return Gpu(name=name, vendor="nvidia", vram_total_gb=vram, backend="cuda", pci_id=pci)


def test_one_side_without_pci_id_falls_back_to_the_name():
    pc = _gpu("NVIDIA RTX 6000 Ada Generation", 47.99)  # nvidia-smi の GPU(型番 ID なし)
    assert same_gpu("NVIDIA RTX 6000 Ada Generation", "10de:26b1", pc, 48.0)
    assert same_gpu("NVIDIA RTX 6000 Ada Generation", None, pc, 48.0)
    assert not same_gpu("NVIDIA RTX 5000 Ada Generation", "10de:26b2", pc, 32.0)
    assert not same_gpu("NVIDIA RTX 6000 Ada Generation Laptop GPU", None, pc, 48.0)


def test_both_pci_ids_decide_even_if_the_names_match():
    pc = _gpu("NVIDIA RTX 6000 Ada Generation", 47.99, "10de:26b1")
    assert same_gpu("NVIDIA RTX 6000 Ada Generation", "10de:26b1", pc, 48.0)
    assert not same_gpu("NVIDIA RTX 6000 Ada Generation", "10de:26b2", pc, 48.0)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("NVIDIA GeForce RTX 3070", "GeForce RTX 3070"),  # 古い Linux のドライバの書き方
        ("NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3070 "),
        ("NVIDIA RTX 6000 Ada Generation", "NVIDIA RTX 6000 Ada Generation"),  # nvidia-smi と WMI
        ("NVIDIA GeForce RTX 4060 Ti", "NVIDIA GeForce RTX 4060 Ti"),
    ],
)
def test_name_spellings_of_the_same_gpu_match(a, b):
    assert gpu_key(a) == gpu_key(b)
    assert same_gpu(a, None, _gpu(b, 8.0), 8.0)


@pytest.mark.parametrize(
    ("measured", "reported", "same"),
    [
        (24.0, 23.99, True),  # 公称と nvidia-smi(Windows)
        (6.0, 5.80, True),  # Windows と Linux(CUDA は 5934 MiB と報告)
        (16.0, 15.92, True),
        (8.0, 7.79, True),
        (8.0, 16.0, False),  # 同じ名前の別の版(RTX 4060 Ti 8 GB と 16 GB)
        (12.0, 8.0, False),
    ],
)
def test_reported_vram_differences_do_not_split_the_same_gpu(measured, reported, same):
    assert same_vram(measured, _gpu("NVIDIA GeForce RTX 4060 Ti", reported)) is same
