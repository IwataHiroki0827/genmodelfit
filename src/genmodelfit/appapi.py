# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""アプリが 1 モデルを判定する公開の関数(#27)。`gmf.system_from` と `gmf.judge_model`。

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

import os
from collections.abc import Iterable, Mapping
from typing import Any

from .db import Database
from .models import Gpu, Settings, SystemInfo
from .recommend import Fit

GIB = 1024**3


def _gpu_from(spec: Mapping[str, Any] | Gpu) -> Gpu:
    if isinstance(spec, Gpu):
        return Gpu(
            name=spec.name,
            vendor=spec.vendor or "nvidia",
            vram_total_gb=float(spec.vram_total_gb),
            driver=spec.driver,
            compute_capability=spec.compute_capability,
            backend="cuda",
        )
    if not isinstance(spec, Mapping):
        raise TypeError(f"GPU は dict か Gpu で渡してください: {spec!r}")
    name = spec.get("name")
    vram = spec.get("vram_gb")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"GPU の name がありません: {spec!r}")
    if isinstance(vram, bool) or not isinstance(vram, (int, float)) or vram < 0:
        raise ValueError(f"GPU の vram_gb が正しくありません(0 以上の数): {spec!r}")
    driver = spec.get("driver")
    cc = spec.get("compute_capability")
    return Gpu(
        name=name.strip(),
        vendor=str(spec.get("vendor") or "nvidia"),
        vram_total_gb=float(vram),
        driver=str(driver) if driver else None,
        compute_capability=str(cc) if cc else None,
        backend="cuda",
    )


def system_from(
    gpus: Iterable[Mapping[str, Any] | Gpu] = (),
    *,
    ram_gb: float | None = None,
    gpu_usable: bool | None = None,
    windows: bool | None = None,
) -> SystemInfo:
    """組み込み側が調べた GPU から SystemInfo を作る(nvidia-smi・PowerShell は呼ばない。OS・CPU・保存先の
    空きは `gmf.system()` と同じくこの PC のもの)。

    gpus: GPU の並び。1 つは `{"name": "NVIDIA GeForce RTX 5080", "vram_gb": 16.0, "driver": "581.57"}`
        (`driver`・`compute_capability`("8.9" など)・`vendor`(既定 nvidia)は任意)か Gpu。空なら GPU 無し
        (CPU で判定する)。判定に使うのは NVIDIA で、VRAM が最大の 1 枚(CLI と同じ)。
    ram_gb: メインメモリ(GiB)。None はこの PC のメモリと仮想メモリを読む(`gmf.system()` と同じ)。
        渡したときは、それを搭載量にも空きにも使い、この PC の仮想メモリ(ページファイル)は使わない
        (`system --ram` と同じ)。
    gpu_usable: None(既定)は、世代とドライバから PyTorch の CUDA 版で動くかを genmodelfit が決める
        (動かない世代・ドライバが古い GPU は使わず CPU として判定し、理由を notes に書く。docs/hardware.md)。
        True は「使う側が動くことを確かめた」(アプリが自分の入れる PyTorch で確かめた)。そのときも世代と
        ComfyUI の精度は genmodelfit が決める。False は GPU を使わない(CPU で判定)。
    windows: ドライバの版を Windows の表で見て、ComfyUI が Windows で動くものとして精度を決めるか
        (Pascal の fp16 の扱いが変わる)。None はこの PC が Windows か(WSL はドライバだけ Windows)。
    """
    from .hardware import Memory, classify_gpu, comfy_precision, detect, is_wsl, parse_compute_cap

    if gpu_usable is not None and not isinstance(gpu_usable, bool):
        raise TypeError("gpu_usable は None / True / False のどれかです")
    if ram_gb is not None and (
        isinstance(ram_gb, bool) or not isinstance(ram_gb, (int, float)) or ram_gb < 0
    ):
        raise ValueError(f"ram_gb が正しくありません(0 以上の数): {ram_gb!r}")
    if windows is None:
        windows_driver, comfy_windows = os.name == "nt" or is_wsl(), os.name == "nt"
    else:
        windows_driver = comfy_windows = bool(windows)
    found: list[Gpu] = []
    for spec in gpus:
        g = _gpu_from(spec)
        if g.vendor != "nvidia" or g.vram_total_gb <= 0 or gpu_usable is False:
            g.backend = "none"
            if gpu_usable is False:
                g.notes.append("使う側が GPU を使わないと決めたので、CPU として判定した")
            found.append(g)
            continue
        classify_gpu(g, windows_driver=windows_driver, comfy_windows=comfy_windows)
        if gpu_usable and g.backend != "cuda":
            # 使う側が確かめた: 動くかの判断は借りず、世代と精度だけを決める
            g.backend = "cuda"
            cc = parse_compute_cap(g.compute_capability)
            g.compute_dtype, g.bf16, g.fp8 = comfy_precision(cc, g.name, windows=comfy_windows)
            g.notes.append("使う側が動くことを確かめたので、GPU として判定した")
        found.append(g)
    memory = None
    if ram_gb is not None:
        memory = Memory(total=int(float(ram_gb) * GIB), available=int(float(ram_gb) * GIB))
    return detect(gpus=found, ram_gb=ram_gb, other_gpus=False, memory=memory)


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
        system_from を呼ぶ(ram_gb・gpu_usable・windows もそのまま渡す)。cpu_only=True は GPU 無し。
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
        raise TypeError("model は ID かモデル名の文字列です")
    if system_info is None:
        if cpu_only:
            system_info = system_from((), ram_gb=ram_gb, windows=windows)
        elif gpu_name is not None or vram_gb is not None:
            if gpu_name is None or vram_gb is None:
                raise ValueError("gpu_name と vram_gb は両方渡してください(GPU が無いなら cpu_only=True)")
            spec = {
                "name": gpu_name,
                "vram_gb": vram_gb,
                "driver": driver,
                "compute_capability": compute_capability,
            }
            system_info = system_from([spec], ram_gb=ram_gb, gpu_usable=gpu_usable, windows=windows)
        else:
            from .hardware import detect

            system_info = detect(ram_gb=ram_gb)
    elif cpu_only or gpu_name is not None or vram_gb is not None:
        raise ValueError("system_info と gpu_name / vram_gb / cpu_only は同時に渡せません")
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
