# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""アプリが 1 モデルを判定する公開の関数(#27)。`gmf.system_from` と `gmf.judge_model`。
#96 で `gmf.gpu_family`(AMD・Intel の GPU 名の表)と `gmf.self_check`(同梱物がそろっているか)を足した。

組み込み側(AI生成キットなど)が自分で調べた GPU の名前・VRAM・ドライバから SystemInfo を作り、1 モデルを
判定する。世代(compute capability)と ComfyUI の計算の精度は genmodelfit が GPU の名前とドライバから決める。
これまで組み込み側が使っていた内部の関数(recommend.judge / vram_need・hardware.classify_gpu /
comfy_precision / parse_compute_cap・models.Gpu)を呼ばずに済むようにする。
互換の対象は `genmodelfit.__all__` に入る名前だけ(docs/schema.md「Python API (stable)」)。

ネットに触らない。GPU は渡されたものだけを使い、nvidia-smi・PowerShell は呼ばない(GPU を渡さず cpu_only でも
ないときの judge_model の既定だけは、`gmf.system()` と同じく、この PC の GPU を調べる)。OS・CPU の名前と数・
保存先の空きは、`gmf.system()` と同じくこの PC のものを読む。何も表示しない。
"""

from __future__ import annotations

import math
import os
import platform
import re
from collections.abc import Iterable, Mapping
from typing import Any

from .db import Database
from .i18n import tr
from .models import Adapter, Gpu, Settings, SystemInfo
from .recommend import Fit

GIB = 1024**3
_PCI_ID = re.compile(r"[0-9a-f]{4}:[0-9a-f]{4}")


def _amount(name: str, value: Any) -> float:
    """GiB の量(0 以上の有限の数)。bool・NaN・無限大・負の数は ValueError。"""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            tr(
                f"{name} が正しくありません(0 以上の有限の数): {value!r}",
                f"invalid {name} (a finite number >= 0): {value!r}",
            )
        )
    return float(value)


def _pci_id(spec: Any, value: Any) -> str | None:
    """PCI の型番 ID("1002:744c")。None はそのまま。形が違えば ValueError。"""
    if value is None:
        return None
    if not isinstance(value, str) or not _PCI_ID.fullmatch(value.strip().lower()):
        raise ValueError(
            tr(
                f"GPU の pci_id が正しくありません(1002:744c の形): {spec!r}",
                f"invalid GPU pci_id (like 1002:744c): {spec!r}",
            )
        )
    return value.strip().lower()


def _vendor(spec: Any, value: Any) -> str:
    if value is None or value == "":
        return "nvidia"
    if not isinstance(value, str):
        raise ValueError(
            tr(f"GPU の vendor が文字列ではありません: {spec!r}", f"the GPU vendor is not a string: {spec!r}")
        )
    return value.strip().lower() or "nvidia"


def _gpu_from(spec: Mapping[str, Any] | Gpu) -> Gpu | Adapter:
    """渡された GPU 1 つ → NVIDIA は Gpu(nvidia-smi で確かめたものの代わり)、ほかの会社は Adapter
    (この PC で見つけた画面の部品の代わり。検出と同じ対応表で判定する。#96)。"""
    if isinstance(spec, Gpu):
        vendor = _vendor(spec, spec.vendor)
        vram = _amount("vram_total_gb", spec.vram_total_gb)
        if vendor != "nvidia":
            return Adapter(
                name=spec.name.strip(),
                vendor=vendor,
                vram_gb=vram or None,
                vram_source="app" if vram else None,
                source="app",
                pci_id=_pci_id(spec, spec.pci_id),
                driver=spec.driver,
            )
        return Gpu(
            name=spec.name,
            vendor=vendor,
            vram_total_gb=vram,
            driver=spec.driver,
            compute_capability=spec.compute_capability,
            backend="cuda",
        )
    if not isinstance(spec, Mapping):
        raise TypeError(
            tr(f"GPU は dict か Gpu で渡してください: {spec!r}", f"pass a GPU as a dict or Gpu: {spec!r}")
        )
    name = spec.get("name")
    vram = spec.get("vram_gb")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(tr(f"GPU の name がありません: {spec!r}", f"the GPU has no name: {spec!r}"))
    vendor = _vendor(spec, spec.get("vendor"))
    driver = spec.get("driver")
    if vram is not None or vendor == "nvidia":
        # NVIDIA 以外は VRAM を省ける(Intel の単体 GPU は名前の公称、内蔵 GPU はメインメモリの共有の分)
        try:
            vram = _amount("vram_gb", vram)
        except ValueError:
            raise ValueError(
                tr(
                    f"GPU の vram_gb が正しくありません(0 以上の有限の数): {spec!r}",
                    f"invalid GPU vram_gb (a finite number >= 0): {spec!r}",
                )
            ) from None
    if vendor != "nvidia":
        return Adapter(
            name=name.strip(),
            vendor=vendor,
            vram_gb=vram or None,
            vram_source="app" if vram else None,
            source="app",
            pci_id=_pci_id(spec, spec.get("pci_id")),
            driver=str(driver) if driver else None,
        )
    cc = spec.get("compute_capability")
    return Gpu(
        name=name.strip(),
        vendor=vendor,
        vram_total_gb=vram,
        driver=str(driver) if driver else None,
        compute_capability=str(cc) if cc else None,
        backend="cuda",
    )


def _host(windows: bool | None, windows_build: int | None, cpu: str | None):
    """NVIDIA 以外の GPU の対応表に使う OS と CPU(hardware.Host)。渡されなかったものはこの PC のもの。"""
    from .hardware import Host, _cpu_name

    if windows_build is not None and (
        isinstance(windows_build, bool) or not isinstance(windows_build, int) or windows_build <= 0
    ):
        raise ValueError(
            tr(
                f"windows_build は正の整数です(Windows 11 は 22000 以上): {windows_build!r}",
                f"windows_build must be a positive integer (Windows 11 is 22000 or more): {windows_build!r}",
            )
        )
    if cpu is not None and not isinstance(cpu, str):
        raise TypeError(tr("cpu は CPU の名前の文字列です", "cpu must be a CPU name string"))
    here = platform.system()
    if windows is None:
        system = here
    elif windows:
        system = "Windows"
    else:
        system = here if here not in ("Windows", "") else "Linux"
    if windows_build is not None and system != "Windows":
        raise ValueError(
            tr(
                "windows_build は Windows のとき(windows=True か、この PC が Windows)だけ渡せます",
                "windows_build can be given only for Windows (windows=True, or this PC is Windows)",
            )
        )
    build = windows_build
    mac = None
    if system == "Windows" and build is None and here == "Windows":
        parts = platform.version().split(".")
        build = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
    if system == "Darwin":
        from .hardware import host_info

        mac = host_info().mac_version
    return Host(
        system=system,
        windows_build=build,
        mac_version=mac,
        cpu=_cpu_name() if cpu is None else cpu.strip(),
    )


def system_from(
    gpus: Iterable[Mapping[str, Any] | Gpu] = (),
    *,
    ram_gb: float | None = None,
    gpu_usable: bool | None = None,
    windows: bool | None = None,
    windows_build: int | None = None,
    cpu: str | None = None,
) -> SystemInfo:
    """組み込み側が調べた GPU から SystemInfo を作る(nvidia-smi・PowerShell は呼ばない。OS・CPU・保存先の
    空きは `gmf.system()` と同じくこの PC のもの)。

    gpus: GPU の並び。1 つは `{"name": "NVIDIA GeForce RTX 5080", "vram_gb": 16.0, "driver": "581.57"}`
        (`driver`・`compute_capability`("8.9" など)・`vendor`(既定 nvidia)は任意)か Gpu。空なら GPU 無し
        (CPU で判定する)。使える NVIDIA があれば、判定に使うのはその VRAM が最大の 1 枚(CLI と同じ)。
        vendor が amd / intel / apple の GPU(#96)は、`gmf.system()` がこの PC で見つけた画面の部品と同じ対応表
        (docs/hardware.md「NVIDIA 以外の GPU」)で判定し、使える NVIDIA が無いときだけ判定に使う(段階は
        NVIDIA と同じ基準。#108)。例 `{"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vram_gb": 16}`。
        NVIDIA 以外は vram_gb を省ける(Intel の単体 GPU は名前の公称、内蔵 GPU はメインメモリの共有の分)。
        `pci_id`("1002:744c"。名前が汎用のとき型番の ID で世代を引く)も任意。部品ごとの結果は
        other_gpus(usable・reason(使う根拠か使えない理由)・integrated・needs)に入る。
    ram_gb: メインメモリ(GiB)。None はこの PC のメモリと仮想メモリを読む(`gmf.system()` と同じ)。
        渡したときは、それを搭載量にも空きにも使い、この PC の仮想メモリ(ページファイル)は使わない
        (`system --ram` と同じ)。内蔵 GPU の共有の分もこの量から決める。
    gpu_usable: None(既定)は、世代とドライバから PyTorch の CUDA 版で動くかを genmodelfit が決める
        (動かない世代・ドライバが古い GPU は使わず CPU として判定し、理由を notes に書く。docs/hardware.md)。
        True は「使う側が動くことを確かめた」(アプリが自分の入れる PyTorch で確かめた)。そのときも世代と
        ComfyUI の精度は genmodelfit が決める。False は GPU を使わない(CPU で判定)。
        True / None が効くのは NVIDIA だけ(NVIDIA 以外は対応表で決める)。False はすべての GPU に効く。
    windows: ドライバの版を Windows の表で見て、ComfyUI が Windows で動くものとして精度を決めるか
        (Pascal の fp16 の扱いが変わる)。None はこの PC が Windows か(WSL はドライバだけ Windows)。
        NVIDIA 以外の対応表の OS は、True が Windows、False が Linux(この PC が macOS なら macOS)、
        None がこの PC の OS。
    windows_build: Windows のビルド番号(22000 以上が Windows 11。AMD・Intel の GPU は Windows 11 が要る)。
        None は、この PC が Windows ならその番号、ほかは分からない(分からなければ AMD・Intel は使わない)。
        OS が Windows でないときに渡すと ValueError。
    cpu: CPU の名前(名前が「AMD Radeon(TM) Graphics」だけの内蔵 GPU の世代を決める。SystemInfo.cpu にも
        入る)。None はこの PC の CPU。
    """
    from .hardware import Memory, classify_gpu, comfy_precision, detect, is_wsl, parse_compute_cap

    if gpu_usable is not None and not isinstance(gpu_usable, bool):
        raise TypeError(
            tr("gpu_usable は None / True / False のどれかです", "gpu_usable must be None, True or False")
        )
    if ram_gb is not None:
        ram_gb = _amount("ram_gb", ram_gb)
    host = _host(windows, windows_build, cpu)
    if windows is None:
        windows_driver, comfy_windows = os.name == "nt" or is_wsl(), os.name == "nt"
    else:
        windows_driver = comfy_windows = bool(windows)
    found: list[Gpu] = []
    adapters: list[Adapter] = []
    for spec in gpus:
        g = _gpu_from(spec)
        if isinstance(g, Adapter):
            if gpu_usable is not False:
                adapters.append(g)  # 検出と同じく hardware.detect が対応表で判定する
                continue
            g = Gpu(name=g.name, vendor=g.vendor, vram_total_gb=g.vram_gb or 0.0)
        if g.vendor != "nvidia" or g.vram_total_gb <= 0 or gpu_usable is False:
            g.backend = "none"
            if gpu_usable is False:
                g.notes.append(
                    tr(
                        "使う側が GPU を使わないと決めたので、CPU として判定した",
                        "The caller chose not to use the GPU, so judged as CPU",
                    )
                )
            found.append(g)
            continue
        classify_gpu(g, windows_driver=windows_driver, comfy_windows=comfy_windows)
        if gpu_usable and g.backend != "cuda":
            # 使う側が確かめた: 動くかの判断は借りず、世代と精度だけを決める
            g.backend = "cuda"
            cc = parse_compute_cap(g.compute_capability)
            g.compute_dtype, g.bf16, g.fp8 = comfy_precision(cc, g.name, windows=comfy_windows)
            g.notes.append(
                tr(
                    "使う側が動くことを確かめたので、GPU として判定した",
                    "The caller confirmed it works, so judged as GPU",
                )
            )
        found.append(g)
    memory = None
    if ram_gb is not None:
        memory = Memory(total=int(float(ram_gb) * GIB), available=int(float(ram_gb) * GIB))
    info = detect(
        gpus=found,
        ram_gb=ram_gb,
        other_gpus=False,
        memory=memory,
        host=host,
        adapters=adapters or None,
    )
    if cpu is not None:
        info.cpu = host.cpu
    return info


def gpu_family(name: str, vendor: str, cpu: str = "") -> dict[str, Any] | None:
    """AMD・Intel の GPU 名を genmodelfit の名前の表(docs/hardware.md「NVIDIA 以外の GPU」)で引く(#96)。
    アプリが入れる PyTorch の種類(ROCm の gfx・XPU)と、単体か内蔵かを決めるのに使う(表を写さずに済む)。

    返す値: `{"vendor": "amd"|"intel", "backend": "rocm"|"xpu", "arch": "gfx1101"|None,
    "family": "RDNA 3"|"Arc B-Series (Battlemage)"|..., "integrated": bool, "vram_floor_gb": float|None}`。
    arch は AMD の LLVM の名前(gfx。Intel は None)。family は AMD の世代(表に無い gfx は "?")か Intel の系統。
    vram_floor_gb は Intel の単体 GPU の名前から推す公称の VRAM(量の違う版があれば少ないほう)、ほかは None。
    名前が表に無い(PyTorch の XPU 版の対象でない Intel の GPU・表に無い新しい GPU)ときは None。
    cpu: 名前が「AMD Radeon(TM) Graphics」だけの内蔵 GPU は CPU の名前で決める(AMD だけ)。
    表にあることは、その PC で動くことではない(RDNA 1 も表にある)。動くか・段階は system_from の判定
    (Gpu.backend と other_gpus の reason)で決める。vendor が amd / intel 以外なら ValueError。
    ネットに触らず、何も表示しない。
    """
    from .hardware import AMD_ARCH, amd_gfx, intel_xpu

    if not isinstance(name, str):
        raise TypeError(tr("name は GPU の名前の文字列です", "name must be a GPU name string"))
    if not isinstance(cpu, str):
        raise TypeError(tr("cpu は CPU の名前の文字列です", "cpu must be a CPU name string"))
    v = vendor.strip().lower() if isinstance(vendor, str) else None
    if v not in ("amd", "intel"):
        raise ValueError(
            tr(f"vendor は amd か intel です: {vendor!r}", f"vendor must be amd or intel: {vendor!r}")
        )
    if v == "amd":
        found = amd_gfx(name, cpu)
        if found is None:
            return None
        gfx, integrated = found
        return {
            "vendor": "amd",
            "backend": "rocm",
            "arch": gfx,
            "family": AMD_ARCH.get(gfx, "?"),
            "integrated": bool(integrated),
            "vram_floor_gb": None,
        }
    found_i = intel_xpu(name)
    if found_i is None:
        return None
    series, integrated, floor = found_i
    return {
        "vendor": "intel",
        "backend": "xpu",
        "arch": None,
        "family": series,
        "integrated": bool(integrated),
        "vram_floor_gb": floor,
    }


def self_check() -> dict[str, Any]:
    """同梱物がそろっているか(#96。exe に固めたアプリのスモーク試験用)。例外を出さず、ネットに触らず、
    何も表示しない。

    返す値: `{"ok": bool, "version": "0.11.0", "checks": {名前: {"ok": bool, "detail": str}}}`。checks は
    `db`(同梱の一覧が読めて 1 件以上ある)、`measurements`(同梱の実測 2 つが読める)、`calibration`(係数の表が
    読める)、`cryptography`(署名を確かめる部品が読み込める)、`public_keys`(同梱の信じる公開鍵が 1 つ以上
    読める)。detail は件数・版、または ok でない理由(例外の名前と文)。ok はすべての checks が ok のとき True。
    """
    import json

    from ._version import __version__

    def db_check() -> str:
        from .db import bundled_text, parse_db

        n = len(parse_db(json.loads(bundled_text("db.json")), "bundled").models)
        if n == 0:
            raise ValueError(tr("同梱の一覧が空", "the bundled list is empty"))
        return f"{n} models"

    def measurements_check() -> str:
        from .db import load_measurements

        return f"{len(load_measurements(include_cache=False))} measurements"

    def calibration_check() -> str:
        from .db import bundled_text

        data = json.loads(bundled_text("calibration.json"))
        if not isinstance(data, dict) or not isinstance(data.get("coefficients"), dict):
            raise ValueError(tr("係数の表の形が違う", "the coefficients table is malformed"))
        return f"{len(data['coefficients'])} coefficients"

    def crypto_check() -> str:
        import cryptography
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        Ed25519PublicKey.from_public_bytes(bytes(32))  # 部品(バイナリの拡張)まで読み込めるか
        return f"cryptography {cryptography.__version__}"

    def keys_check() -> str:
        from .signing import trusted_public_keys

        return f"{len(trusted_public_keys())} public keys"

    checks: dict[str, dict[str, Any]] = {}
    for key, fn in (
        ("db", db_check),
        ("measurements", measurements_check),
        ("calibration", calibration_check),
        ("cryptography", crypto_check),
        ("public_keys", keys_check),
    ):
        try:
            checks[key] = {"ok": True, "detail": fn()}
        except Exception as e:  # 何が欠けても例外にせず、理由を返す
            checks[key] = {"ok": False, "detail": f"{type(e).__name__}: {e}"[:300]}
    return {"ok": all(c["ok"] for c in checks.values()), "version": __version__, "checks": checks}


def judge_model(
    model: str,
    *,
    gpu_name: str | None = None,
    vram_gb: float | None = None,
    driver: str | None = None,
    compute_capability: str | None = None,
    ram_gb: float | None = None,
    cpu_only: bool = False,
    gpu_usable: bool | None = None,
    windows: bool | None = None,
    gpu_vendor: str | None = None,
    windows_build: int | None = None,
    cpu: str | None = None,
    settings: Settings | None = None,
    content: str = "sfw",
    system_info: SystemInfo | None = None,
    db: Database | None = None,
    comfyui_version: str | None = None,
) -> Fit:
    """1 モデルを判定する(`recommend` の 1 件と同じ Fit。`fit.to_dict()` が JSON の形)。

    model: ID(`image_z_image_turbo` など。内容によらず、そのまま合えばそれ)か、モデルの表示名・ID の一部
        (`gmf.download` / CLI の `info` と同じ探し方。content で絞り、同じモデルの仲間が複数あるときは
        recommend の 1 行と同じく、段階が良く精度が高いものを選び、ほかを fit.workflows に入れる)。
        合わない・2 つ以上のモデルに合うときは ModelNotFound(candidates に候補)。
    PC: system_info を渡せばそれ。無ければ gpu_name + vram_gb(+ driver・compute_capability)の 1 枚で
        system_from を呼ぶ(ram_gb・gpu_usable・windows・windows_build・cpu もそのまま渡す)。gpu_vendor
        (既定 nvidia。#96)を amd / intel / apple にすると、その GPU を system_from と同じ対応表で判定する
        (NVIDIA 以外は vram_gb=0 で VRAM が分からないことを示せる)。cpu_only=True は GPU 無し
        (gpu_name・vram_gb・gpu_usable・gpu_vendor と一緒には渡せない)。vram_gb・ram_gb は 0 以上の有限の数
        (NaN・無限大は ValueError)。
        GPU も cpu_only も無ければ、この PC を調べる(`gmf.system()` と同じ。ram_gb があればそれで上書き)。
    settings: 解像度・長さなど(None は既定)。content: "sfw"(既定)/ "all" / "uncensored"(名前で探すとき)。
    db: 一覧(既定は `gmf.load_db()`。アプリは同梱の一覧を渡すとよい)。comfyui_version: 自動のおすすめ
        (fit.auto)の判定に使う ComfyUI の版。
    """
    from .downloader import resolve_candidates
    from .gate import GatePolicy
    from .recommend import _member_dict, check_filters, judge, variant_key

    check_filters(content=content)
    if not isinstance(model, str):
        raise TypeError(tr("model は ID かモデル名の文字列です", "model must be an id or model name string"))
    if vram_gb is not None:
        vram_gb = _amount("vram_gb", vram_gb)
    if ram_gb is not None:
        ram_gb = _amount("ram_gb", ram_gb)
    if cpu_only and (
        gpu_usable is not None or gpu_name is not None or vram_gb is not None or gpu_vendor is not None
    ):
        raise ValueError(
            tr(
                "cpu_only=True のときは gpu_name / vram_gb / gpu_usable / gpu_vendor を渡せません"
                "(GPU を使わないので)",
                "gpu_name / vram_gb / gpu_usable / gpu_vendor cannot be given with cpu_only=True"
                " (no GPU is used)",
            )
        )
    if gpu_vendor is not None and gpu_name is None and not cpu_only:
        raise ValueError(
            tr(
                "gpu_vendor は gpu_name・vram_gb と一緒に渡してください",
                "give gpu_vendor together with gpu_name and vram_gb",
            )
        )
    host_args = {"windows": windows, "windows_build": windows_build, "cpu": cpu}
    if system_info is None:
        if cpu_only:
            system_info = system_from((), ram_gb=ram_gb, **host_args)
        elif gpu_name is not None or vram_gb is not None:
            if gpu_name is None or vram_gb is None:
                raise ValueError(
                    tr(
                        "gpu_name と vram_gb は両方渡してください(GPU が無いなら cpu_only=True)",
                        "give both gpu_name and vram_gb (cpu_only=True if there is no GPU)",
                    )
                )
            spec = {
                "name": gpu_name,
                "vram_gb": vram_gb,
                "driver": driver,
                "compute_capability": compute_capability,
                "vendor": gpu_vendor,
            }
            system_info = system_from([spec], ram_gb=ram_gb, gpu_usable=gpu_usable, **host_args)
        else:
            from .hardware import detect

            system_info = detect(ram_gb=ram_gb)
    elif cpu_only or gpu_name is not None or vram_gb is not None:
        raise ValueError(
            tr(
                "system_info と gpu_name / vram_gb / cpu_only は同時に渡せません",
                "system_info cannot be given with gpu_name / vram_gb / cpu_only",
            )
        )
    if db is None:
        from .db import load

        db = load()
    policy = GatePolicy(comfyui_version=comfyui_version)
    members, _exact = resolve_candidates(db, model, content)
    fits = [judge(m, system_info, db, settings=settings, policy=policy) for m in members]
    fits.sort(key=variant_key)
    rep = fits[0]
    rep.workflows = [_member_dict(o, True) for o in fits[1:]]
    return rep
