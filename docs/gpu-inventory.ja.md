# GPU の登録の表

[English](gpu-inventory.md) · **日本語**

genmodelfit が使う手書きの GPU の表の全部の行と、行に当たる公式の製品名・判定・出典・確かめた日。`tools/gpu_inventory.py` が `src/genmodelfit/gputables.py` と、`tests/fixtures/gpu_official` に固定した公式の一覧から作る(手で直さない)。決まりは [hardware.ja.md](hardware.ja.md)「GPU の登録の表と公式の一覧」。

毎週の見張りが知らせた GPU を足すとき: `gputables.py` に出典と確かめた日を付けて行を足し、`python -m genmodelfit.gpuwatch --save-fixtures tests/fixtures/gpu_official` で公式の一覧を取り直し、`python tools/gpu_inventory.py` でこのページを作り直す。

公式の名前は固定した一覧(pci.ids の GPU の行 186 件・AMD の製品 116 件・NVIDIA の名前 459 件・Intel の検証済みの部品)から取った。名前はコードと同じく、最初に当たる行に数える。

## まとめ

| 表 | 行 | 公式の名前が当たる行 |
|---|---|---|
| `NAME_CC_RULES` | 17 | 17 |
| `ARCH_MIN_DRIVER_RULES` | 1 | 0 |
| `AMD_ARCH_RULES` | 31 | 27 |
| `AMD_NAME_RULES` | 33 | 30 |
| `AMD_CPU_IGPU_RULES` | 8 | 5 |
| `INTEL_XPU_RULES` | 20 | 20 |
| `PCI_DEVICE_RULES` | 106 | 106 |

## `NAME_CC_RULES`

NVIDIA: 名前 → compute capability(古いドライバ)

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| NVIDIA | `NAME_CC_RULES` | `\brtx\s*50[5-9]0\b\|\brtx pro \d{4}\b.*\bblackwell\b` | NVIDIA RTX PRO 6000 Blackwell Workstation Edition, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, NVIDIA RTX PRO 5500 Blackwell Workstation Edition, NVIDIA RTX PRO 5000 Blackwell, NVIDIA RTX PRO 4500 Blackwell Workstation Edition, NVIDIA RTX PRO 4000 Blackwell; ほか 9 件 | 12.0 | 使う(cu132, cu130, cu129, cu128) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgb10\b` | NVIDIA GB10 (DGX Spark) | 12.1 | 使う(cu132, cu130, cu129, cu128) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bg?b300\b` | NVIDIA GB300, NVIDIA B300, NVIDIA GB300 (DGX Station) | 10.3 | 使う(cu132, cu130, cu129, cu128) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bg?b[12]00\b` | NVIDIA GB200, NVIDIA B200 | 10.0 | 使う(cu132, cu130, cu129, cu128) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bg?h(100\|200\|800\|20)\b` | NVIDIA GH200, NVIDIA H200, NVIDIA H100 | 9.0 | 使う(cu132, cu130, cu129, cu128, cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\brtx\s*40[5-9]0\b\|\brtx\s*\d{4}e?\s*(?:sff\s*)?ada\b\|\bl40s?\b\|\bl4\b\|\bl20\b` | NVIDIA L4, NVIDIA L40, NVIDIA L40S, NVIDIA RTX 6000 Ada, NVIDIA RTX 5000 Ada, NVIDIA RTX 4500 Ada; ほか 10 件 | 8.9 | 使う(cu132, cu130, cu129, cu128, cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\ba100\b\|\ba800\b\|\ba30\b` | NVIDIA A100, NVIDIA A30 | 8.0 | 使う(cu132, cu130, cu129, cu128, cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\brtx\s*30[5-9]0\b\|\brtx\s*2050\b\|\brtx\s*a\d{3,4}\b\|\ba10g?\b\|\ba16\b\|\ba40\b\|\ba2\b\|\bmx570\b` | NVIDIA A40, NVIDIA A10, NVIDIA A16, NVIDIA A2, NVIDIA RTX A6000, NVIDIA RTX A5000; ほか 13 件 | 8.6 | 使う(cu132, cu130, cu129, cu128, cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\brtx\s*20[6-8]0\b\|\btitan rtx\b\|\bquadro rtx\b\|\bgtx\s*16[3-6]0\b\|\bt4\b\|\bt(400\|500\|550\|600\|1000\|1200\|2000)\b\|\bmx(450\|550)\b` | NVIDIA T4, QUADRO RTX 8000, QUADRO RTX 6000, QUADRO RTX 5000, QUADRO RTX 4000, QUADRO RTX 3000; ほか 12 件 | 7.5 | 使う(cu132, cu130, cu129, cu128, cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bv100\b\|\btitan v\b\|\bgv100\b` | NVIDIA V100, Quadro GV100, NVIDIA TITAN V | 7.0 | 使う(cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bp100\b\|\bgp100\b` | Tesla P100, Quadro GP100 | 6.0 | 使う(cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgtx\s*10[5-8]0\b\|\bgt\s*1030\b\|\btitan xp\b\|(?<!gtx )\btitan x\b\|\bquadro p\d{3,4}\b\|\btesla p(4\|40)\b\|\bmx(150\|230\|250\|330\|350)\b` | Tesla P40, Tesla P4, Quadro P6000, Quadro P5200, Quadro P5000, Quadro P4200; ほか 18 件 | 6.1 | 使う(cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgtx\s*(745\|750)\b\|\bquadro k(620\|1200\|2200)\b\|\bquadro m(520\|1200)\b\|\bmx1[13]0\b\|\b940mx\b\|\btesla m10\b` | Quadro K2200, Quadro K1200, Quadro K620, Quadro M1200, Quadro M520, GeForce GTX 750 Ti; ほか 1 件 | 5.0 | 使う(cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgtx\s*(950\|960\|970\|980)\b\|\bgtx titan x\b\|\bquadro m(620\|2000\|2200\|4000\|5000\|6000)\b\|\btesla m(4\|6\|40\|60)\b` | Tesla M60, Tesla M40, Quadro M6000 24GB, Quadro M6000, Quadro M5000, Quadro M4000; ほか 9 件 | 5.2 | 使う(cu126) | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\btesla k80\b` | Tesla K80 | 3.7 | 断る: 対応する CUDA 版が無い | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgtx\s*780\b\|\bgtx titan( black\| z)?$\|\btesla k(20\|40)\w*\b\|\bgt\s*(710\|720)\b\|\bquadro k(5200\|6000)\b` | Tesla K40, Tesla K20, Quadro K6000, Quadro K5200, GeForce GTX TITAN Z, GeForce GTX TITAN Black; ほか 4 件 | 3.5 | 断る: 対応する CUDA 版が無い | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |
| NVIDIA | `NAME_CC_RULES` | `\bgtx\s*(6[3-9]0\|760\|770)\b\|\bquadro k\d{3,4}\b` | Quadro K5000, Quadro K4200, Quadro K4000, Quadro K2000, Quadro K600, Quadro K420; ほか 10 件 | 3.0 | 断る: 対応する CUDA 版が無い | `nvidia-cuda-gpus`, `nvidia-cuda-legacy` | 2026-10-27 | 0.10 以前 |

## `ARCH_MIN_DRIVER_RULES`

NVIDIA: 世代ごとの最低のドライバ

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| NVIDIA | `ARCH_MIN_DRIVER_RULES` | `blackwell` | - | 570 (CUDA 12.8 GA(Linux 570.26 / Windows 570.65)) | - | `cuda-rn` | 2026-10-27 | 0.10 以前 |

## `AMD_ARCH_RULES`

AMD: gfx → 世代

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| AMD | `AMD_ARCH_RULES` | `gfx1200` | Radeon RX 9060 XT LP, Radeon RX 9060 XT, Radeon RX 9060, Radeon RX 9050, Radeon RX 9050 (4GB), AMD Radeon RX 9060 / XT | RDNA 4 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1201` | Radeon AI PRO R9700S, Radeon AI PRO R9700, Radeon AI PRO R9600D, Radeon AI PRO R9600, Radeon RX 9070 XT, Radeon RX 9070 GRE; ほか 2 件 | RDNA 4 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1170` | - | RDNA 4 | 断る: TheRock で Release Ready でない | `therock-supported`, `llvm-amdgpu`, `comfyui-mm` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_ARCH_RULES` | `gfx1171` | - | RDNA 4 | 断る: TheRock で Release Ready でない | `therock-supported`, `llvm-amdgpu`, `comfyui-mm` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_ARCH_RULES` | `gfx1150` | Radeon 890M, Radeon 880M, AMD Ryzen AI 9 HX 375 | RDNA 3.5 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1151` | Radeon 8065S, Radeon 8050S, Radeon 8040S, Radeon 8060S, AMD Ryzen AI Max+ PRO 395 | RDNA 3.5 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1152` | Radeon 860M, Radeon 840M, Radeon 820M, AMD Ryzen AI 7 350 | RDNA 3.5 | 使う(Windows 11・Linux。TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1153` | AMD Radeon 820M iGPU | RDNA 3.5 | 使う(Windows 11・Linux。TheRock) | `therock-supported`, `therock-releases` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1100` | Radeon PRO W7900 Dual Slot, Radeon PRO W7900, Radeon PRO W7800 48GB, Radeon PRO W7800, Radeon RX 7900 XTX, Radeon RX 7900 XT; ほか 2 件 | RDNA 3 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1101` | Radeon PRO V710, Radeon PRO W7700, Radeon RX 7800 XT, Radeon RX 7700, Radeon RX 7700 XT, AMD Radeon RX 7800 XT / 7700 XT, PRO V710 / W7700 | RDNA 3 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1102` | Radeon RX 7600, AMD Radeon RX 7600 | RDNA 3 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1103` | Radeon 780M, Radeon 760M, Radeon 740M, AMD Ryzen 7 7840U / Ryzen 9 270 | RDNA 3 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1030` | AMD Radeon RX 6900 XT / 6800 XT, PRO W6800 / V620 | RDNA 2 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1031` | AMD Radeon RX 6750 XT / 6700 XT | RDNA 2 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1032` | AMD Radeon RX 6600 XT / 6600, PRO W6600 | RDNA 2 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1033` | AMD Van Gogh iGPU | RDNA 2 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1034` | AMD Radeon RX 6500 XT | RDNA 2 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1035` | AMD Radeon 680M iGPU | RDNA 2 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1036` | AMD Raphael iGPU | RDNA 2 | 断る: 2 CU の画面用の内蔵 GPU | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1010` | AMD Radeon RX 5700 / XT | RDNA 1 | 断る: RDNA 1(RDNA 2 より前) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1011` | AMD Radeon Pro V520 | RDNA 1 | 断る: RDNA 1(RDNA 2 より前) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1012` | AMD Radeon Pro W5500 | RDNA 1 | 断る: RDNA 1(RDNA 2 より前) | `therock-supported` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx908` | MI100, AMD Instinct MI100 | CDNA | 使う(Linux だけ) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx90a` | MI250X, MI250, MI210, AMD Instinct MI250X / MI250 / MI210 | CDNA | 使う(Linux だけ) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx942` | MI325X, MI300X, MI300A, AMD Instinct MI325X / MI300X / MI300A | CDNA | 使う(Linux だけ) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx950` | MI355X, MI350X, MI350P, AMD Instinct MI355X / MI350X | CDNA | 使う(Linux だけ) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx1250` | - | CDNA | 断る: TheRock で Release Ready でない | `therock-supported`, `torch-build-rocm`, `llvm-amdgpu` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_ARCH_RULES` | `gfx900` | MI25, AMD Instinct MI25 | GCN 5 | 断る: GCN 5(RDNA 2 より前) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx906` | MI60, MI50 (32GB), MI50 (16GB), AMD Instinct MI60 / MI50, Radeon Pro VII, Radeon VII | GCN 5 | 断る: GCN 5(RDNA 2 より前) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx90c` | - | GCN 5 | 断る: GCN 5(RDNA 2 より前) | `therock-supported`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_ARCH_RULES` | `gfx803` | MI8, MI6 | GCN 4 | 断る: GCN 4(RDNA 2 より前) | `rocm-hw-specs`, `comfyui-mm` | 2026-10-27 | 0.11.0 |

## `AMD_NAME_RULES`

AMD: GPU 名 → gfx

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| AMD | `AMD_NAME_RULES` | `rx\s*9070\|ai pro r9[67]00\|navi\s*48` | Radeon AI PRO R9700S, Radeon AI PRO R9700, Radeon AI PRO R9600D, Radeon AI PRO R9600, Radeon RX 9070 XT, Radeon RX 9070 GRE; ほか 11 件 | gfx1201 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*90[456]0\|navi\s*44` | Radeon RX 9060 XT LP, Radeon RX 9060 XT, Radeon RX 9060, Radeon RX 9050, Radeon RX 9050 (4GB), AMD Radeon RX 9060 / XT; ほか 3 件 | gfx1200 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `8065s\|8060s\|8050s\|8040s\|strix halo` | Radeon 8065S, Radeon 8050S, Radeon 8040S, Radeon 8060S, AMD Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics (1002:1586) | gfx1151, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `890m\|880m\|strix(?! halo)` | Radeon 890M, Radeon 880M, AMD Radeon 880M / 890M (1002:150e) | gfx1150, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `860m\|840m\|krackan` | Radeon 860M, Radeon 840M, AMD Radeon 840M / 860M Graphics (1002:1114), AMD Radeon 820M / 840M Graphics (1002:1902) | gfx1152, 内蔵 | 使う(Windows 11・Linux。TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `820m` | Radeon 820M, AMD Radeon 820M iGPU | gfx1153, 内蔵 | 使う(Windows 11・Linux。TheRock)(820M の gfx は公式の資料どうしで食い違う(TheRock の RELEASES.md は gfx1153、ROCm の hw specs は gfx1152)。TheRock の wheel が合わなければ gfx1152 の版を試す) | `therock-releases`, `rocm-hw-specs`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*7[67]00[sm]\b\|rx\s*7600m xt` | AMD Radeon RX 7600M XT (1002:73f0), AMD Radeon RX 7600M/7600M XT (1002:7483) | gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*7900\|pro w7[89]00\|navi\s*31` | Radeon PRO W7900 Dual Slot, Radeon PRO W7900, Radeon PRO W7800 48GB, Radeon PRO W7800, Radeon RX 7900 XTX, Radeon RX 7900 XT; ほか 15 件 | gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*7[78]00\|pro w7700\|pro v710\|navi\s*32` | Radeon PRO V710, Radeon PRO W7700, Radeon RX 7800 XT, Radeon RX 7700, Radeon RX 7700 XT, AMD Radeon RX 7800 XT / 7700 XT, PRO V710 / W7700; ほか 8 件 | gfx1101 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*7[3-5]00\b\|pro w7400` | AMD Radeon RX 7400/7300/Pro W7400 (1002:7499), AMD Radeon RX 7500 (1002:749f) | gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_NAME_RULES` | `rx\s*76[05]0\|pro w7[56]00\|navi\s*33` | Radeon RX 7600, AMD Radeon RX 7600, AMD Radeon RX 7600/7600 XT/7600M XT/7600S/7700S / PRO W7600 (1002:7480), AMD Radeon Pro W7500 (1002:7489) | gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `steam machine` | AMD Steam Machine (1002:7481) | gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_NAME_RULES` | `780m\|760m\|740m\|phoenix\|hawk ?point` | Radeon 780M, Radeon 760M, Radeon 740M, AMD Radeon 740M / 760M / 780M Graphics (1002:15bf), AMD Radeon 740M Graphics (1002:15c8), AMD Phoenix (1002:164f); ほか 2 件 | gfx1103, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*6[78][05]0m\b` | - | gfx1031 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*6[678]00s\b\|rx\s*66[05]0m\b` | - | gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*6[345][05]0m\b` | - | gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*6[89]\d0\|pro w6800\|pro w6900x\|pro v620\|navi\s*21` | AMD Radeon RX 6900 XT / 6800 XT, PRO W6800 / V620, AMD Radeon Pro V620 (1002:73a1), AMD Radeon Pro W6900X (1002:73a2), AMD Radeon PRO W6800 (1002:73a3), AMD Radeon RX 6950 XT (1002:73a5), AMD Radeon Pro W6800X/Radeon Pro W6800X Duo (1002:73ab); ほか 3 件 | gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*67\d0\|navi\s*22` | AMD Radeon RX 6750 XT / 6700 XT, AMD Navi 22 (1002:73c3), AMD Navi 22-XL SRIOV MxGPU (1002:73ce), AMD Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT (1002:73df) | gfx1031 | 使う(Windows 11・Linux。TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*66\d0\|pro w6600\|navi\s*23` | AMD Radeon RX 6600 XT / 6600, PRO W6600, AMD Navi 23 (1002:73e0), AMD Radeon PRO W6600M (1002:73e1), AMD Radeon PRO W6600 (1002:73e3), AMD Radeon RX 6650 XT / 6700S / 6800S (1002:73ef), AMD Radeon RX 6600/6600 XT/6600M (1002:73ff) | gfx1032 | 使う(Windows 11・Linux。TheRock) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*6[345]00\|pro w6[345]00\|navi\s*24` | AMD Radeon RX 6500 XT, AMD Radeon PRO W6500M (1002:7421), AMD Radeon PRO W6400 (1002:7422), AMD Radeon PRO W6300/W6300M (1002:7423), AMD Radeon RX 6300 (1002:7424), AMD Radeon RX 6400/6500 XT/6500M (1002:743f) | gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `680m\|660m\|rembrandt` | AMD Radeon 680M iGPU, AMD Rembrandt (1002:164d), AMD Radeon 660M / 680M (1002:1681) | gfx1035, 内蔵 | 使う(Windows 11・Linux。TheRock) | `therock-supported`, `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `610m\|raphael\|granite ridge\|mendocino` | AMD Raphael iGPU, AMD Radeon 610M (1002:13c0), AMD Radeon 610M (1002:1506), AMD Radeon 610M (1002:164e) | gfx1036, 内蔵 | 断る: 2 CU の画面用の内蔵 GPU | `amd-product-pages`, `pci-ids` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `van ?gogh\|sephiroth\|custom gpu 0405` | AMD Van Gogh iGPU, AMD Custom GPU 0405 (1002:1435), AMD Custom GPU 0405 (1002:163f) | gfx1033, 内蔵 | 使う(Windows 11・Linux。TheRock) | `therock-releases`, `steam-deck-spec`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `rx\s*5[67]\d0\|navi\s*10` | AMD Radeon RX 5700 / XT, AMD Ariel/Navi10Lite (1002:13e9), AMD Radeon RX 5600 OEM/5600 XT / 5700/5700 XT (1002:731f) | gfx1010 | 断る: RDNA 1(RDNA 2 より前) | `pci-ids` | 2026-10-21 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `pro v520\|navi\s*12` | AMD Radeon Pro V520, AMD Oberon/Navi12Lite (1002:13f9), AMD Radeon Pro V520/V540 (1002:7362) | gfx1011 | 断る: RDNA 1(RDNA 2 より前) | `therock-releases`, `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_NAME_RULES` | `rx\s*5[35]00\|pro w5500\|navi\s*14` | AMD Radeon Pro W5500, AMD Radeon RX 5500/5500M / Pro 5300/5300M/5500M (1002:7340), AMD Radeon Pro W5500 (1002:7341), AMD Radeon Pro W5500M (1002:7347) | gfx1012 | 断る: RDNA 1(RDNA 2 より前) | `pci-ids` | 2026-10-21 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `mi35\dx?` | MI355X, MI350X, MI350P, AMD Instinct MI355X / MI350X, AMD Instinct MI350X (1002:75a0), AMD Instinct MI355X (1002:75a3); ほか 1 件 | gfx950 | 使う(Linux だけ) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `mi3[02]\dx?\|mi300a` | MI325X, MI300X, MI300A, AMD Instinct MI325X / MI300X / MI300A, AMD Instinct MI300A (1002:74a0), AMD Instinct MI300X (1002:74a1); ほか 6 件 | gfx942 | 使う(Linux だけ) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `mi2[15]0x?` | MI250X, MI250, MI210, AMD Instinct MI250X / MI250 / MI210, AMD Instinct MI250X (1002:7408), AMD Instinct MI250X/MI250 (1002:740c); ほか 1 件 | gfx90a | 使う(Linux だけ) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `mi100` | MI100, AMD Instinct MI100, AMD Instinct MI100 (1002:738c), AMD Instinct MI100 (1002:738e) | gfx908 | 使う(Linux だけ) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `radeon vii\|radeon pro vii\|mi[56]0\b\|vega\s*20` | MI60, MI50 (32GB), MI50 (16GB), AMD Instinct MI60 / MI50, Radeon Pro VII, Radeon VII, AMD Radeon Pro VII/Radeon Instinct MI50 (1002:66a1), AMD Vega 20 (1002:66a2); ほか 3 件 | gfx906 | 断る: GCN 5(RDNA 2 より前) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_NAME_RULES` | `mi25\b` | MI25, AMD Instinct MI25 | gfx900 | 断る: GCN 5(RDNA 2 より前) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_NAME_RULES` | `\bmi[68]\b` | MI8, MI6 | gfx803 | 断る: GCN 4(RDNA 2 より前) | `rocm-hw-specs` | 2026-10-27 | 0.11.0 |

## `AMD_CPU_IGPU_RULES`

AMD: CPU 名 → 内蔵 GPU の gfx

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen ai max` | Ryzen AI Max+ PRO 495, Ryzen AI Max PRO 490, Ryzen AI Max PRO 485, Ryzen AI Max PRO 390, Ryzen AI Max PRO 385, Ryzen AI Max PRO 380; ほか 6 件 | gfx1151, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen ai 9 (hx )?(pro )?[34][67]\d` | Ryzen AI 9 HX PRO 475, Ryzen AI 9 HX PRO 470, Ryzen AI 9 PRO 465, Ryzen AI 9 HX 475, Ryzen AI 9 HX 470, Ryzen AI 9 465; ほか 5 件 | gfx1150, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen ai 5 (pro )?330\b` | Ryzen AI 5 330 | gfx1153, 内蔵 | 使う(Windows 11・Linux。TheRock)(Ryzen AI 5 330 の 820M の gfx は公式の資料どうしで食い違う(TheRock は gfx1153、ROCm の hw specs は gfx1152)) | `therock-releases`, `rocm-hw-specs` | 2026-10-27 | 0.11.0 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen ai [57] (pro )?[34][3-5]\d` | Ryzen AI 7 PRO 450, Ryzen AI 5 PRO 440, Ryzen AI 5 PRO 435, Ryzen AI 5 435, Ryzen AI 5 430, Ryzen AI 7 445; ほか 6 件 | gfx1152, 内蔵 | 使う(Windows 11・Linux。TheRock) | `rocm-hw-specs` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen [3579] (pro )?7\d4[05]hx3?d?\b\|ryzen [3579] (pro )?9\d5[05]hx3?d?\b` | - | gfx1036, 内蔵 | 断る: 2 CU の画面用の内蔵 GPU | `amd-product-pages` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen z1\b` | - | gfx1103, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `amd-product-pages`, `linux-amdkfd` | 2026-10-22 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen [3579] (pro )?[78]\d4\d(u\|hs\|h)\b\|ryzen [3579] (pro )?8[5-7]00g\|ryzen 9 (pro )?270\b` | Ryzen 7 7840U, Ryzen 9 270 | gfx1103, 内蔵 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `therock-releases`, `amd-product-pages` | 2026-10-27 | 0.10 以前 |
| AMD | `AMD_CPU_IGPU_RULES` | `ryzen [3579] (pro )?(6[89]\d0\|7[37]35)(u\|hs\|h\|hx)\b` | - | gfx1035, 内蔵 | 使う(Windows 11・Linux。TheRock) | `therock-supported` | 2026-10-22 | 0.10 以前 |

## `INTEL_XPU_RULES`

Intel: GPU 名 → XPU の系統

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| Intel | `INTEL_XPU_RULES` | `arc.*\ba770\b` | Intel Arc A770 (8086:56a0) | Arc A-Series (Alchemist) (VRAM ≥ 8 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba750\b\|arc.*\ba580\b` | Intel Arc A750 (8086:56a1), Intel Arc A580 (8086:56a2) | Arc A-Series (Alchemist) (VRAM ≥ 8 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba380\b` | Intel Arc A380 (8086:56a5) | Arc A-Series (Alchemist) (VRAM ≥ 6 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba310\b` | Intel Arc A310 (8086:56a6) | Arc A-Series (Alchemist) (VRAM ≥ 4 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba770m\b` | Intel Arc A770M (8086:5690) | Arc A-Series (Alchemist) (VRAM ≥ 16 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-mobile` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba730m\b` | Intel Arc A730M (8086:5691) | Arc A-Series (Alchemist) (VRAM ≥ 12 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-mobile` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba550m\b` | Intel Arc A550M (8086:5692) | Arc A-Series (Alchemist) (VRAM ≥ 8 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-mobile` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\ba3[57]0m\b` | Intel Arc A370M (8086:5693), Intel Arc A350M (8086:5694) | Arc A-Series (Alchemist) (VRAM ≥ 4 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-mobile` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bpro a\d{2}m?\b\|arc.*\ba\d{3}[me]\b\|\bdg2\b` | Intel Arc A570M (8086:5696), Intel Arc A530M (8086:5697), Intel Arc Pro A30M (8086:56b0), Intel Arc Pro A40/A50 (8086:56b1), Intel Arc Pro A60M (8086:56b2), Intel Arc Pro A60 (8086:56b3); ほか 6 件 | Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bb580\b` | Intel Arc B580 (8086:e20b) | Arc B-Series (Battlemage) (VRAM ≥ 12 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bb570\b` | Intel Arc B570 (8086:e20c) | Arc B-Series (Battlemage) (VRAM ≥ 10 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bpro b60\b` | Intel Arc Pro B60 (8086:e211) | Arc B-Series (Battlemage) (VRAM ≥ 24 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bpro b50\b` | Intel Arc Pro B50 (8086:e212) | Arc B-Series (Battlemage) (VRAM ≥ 16 GB) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `intel-arc-desktop` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bpro b[67][05]\b` | Intel Arc Pro B65 (8086:e222), Intel Arc Pro B70 (8086:e223) | Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.11.0 |
| Intel | `INTEL_XPU_RULES` | `battlemage\|\bbmg\b` | Intel Arc B-Series Graphics (CodeName: Battlemage) | Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\bb3\d0\b\|panther lake` | Intel Arc B390 (8086:b080), Intel Arc B370 (8086:b081), Intel Arc B390 (8086:b082), Intel Arc B370 (8086:b083), Intel Arc Pro B390 (8086:b084), Intel Arc Pro B370 (8086:b085); ほか 3 件 | Core Ultra Series 3 (Panther Lake) (内蔵) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\b1[34]0v\b\|lunar lake` | Intel Core Ultra 200V Series Processors Arc Graphics 130V/140V GPU (8086:64a0), Intel Core Ultra Mobile Processors (Series 2) with Intel Arc Graphics (CodeName: Lunar Lake) | Core Ultra Series 2 (Lunar Lake) (内蔵) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `arc.*\b1[34]0t\b\|arrow lake-[hp]` | Intel Arc Pro 130T/140T (8086:7d51), Intel Core Ultra Processors (Series 2) with Intel Arc Graphics (CodeName: Arrow Lake-H) | Core Ultra Series 2 (Arrow Lake-H) (内蔵) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `^(intel )?arc graphics$\|meteor lake` | Intel Arc Graphics (8086:7d55), Intel Core Ultra Processors with Intel Arc Graphics (CodeName: Meteor Lake-H) | Core Ultra (Meteor Lake-H) (内蔵) | 使う(Windows 11・Linux・WSL2) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.10 以前 |
| Intel | `INTEL_XPU_RULES` | `data center gpu max\|ponte vecchio\|\bpvc\b` | Intel Data Center GPU Max 1450 (8086:0b69), Intel Data Center GPU Max 1100C (8086:0b6e), Intel Ponte Vecchio XL (2 Tile) (8086:0bd0), Intel Data Center GPU Max 1550VG (8086:0bd4), Intel Data Center GPU Max 1550 (8086:0bd5), Intel Data Center GPU Max 1550 (8086:0bd6); ほか 6 件 | Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ)(Linux だけ) | `pytorch-xpu`, `pci-ids` | 2026-10-27 | 0.11.0 |

## `PCI_DEVICE_RULES`

AMD / Intel: PCI の型番 ID → 名前

| 会社 | 表 | 鍵 | 当たる公式の製品名 | 値 | 判定 | 出典 | 確かめた日 | 足した版 |
|---|---|---|---|---|---|---|---|---|
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7550` | Navi 48 [Radeon RX 9070/9070 XT/9070 GRE] | Navi 48 → gfx1201 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7551` | Navi 48 [Radeon AI PRO R9700] | Navi 48 → gfx1201 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7590` | Navi 44 [Radeon RX 9050 / 9060 XT] | Navi 44 → gfx1200 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:744c` | Navi 31 [Radeon RX 7900 XT/7900 XTX/7900 GRE/7900M] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7448` | Navi 31 [Radeon Pro W7900] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7449` | Navi 31 [Radeon Pro W7800 48GB] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:744a` | Navi 31 [Radeon Pro W7900 Dual Slot] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:744b` | Navi 31 [Radeon Pro W7900D] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:745e` | Navi 31 [Radeon Pro W7800] | Navi 31 → gfx1100 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7460` | Navi 32 GL-XL [AMD Radeon PRO V710] | Navi 32 → gfx1101 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7461` | Navi 32 [AMD Radeon PRO V710] | Navi 32 → gfx1101 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7470` | Navi 32 [Radeon PRO W7700] | Navi 32 → gfx1101 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:747e` | Navi 32 [Radeon RX 7700 XT / 7800 XT] | Navi 32 → gfx1101 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7480` | Navi 33 [Radeon RX 7600/7600 XT/7600M XT/7600S/7700S / PRO W7600] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7481` | Navi 33 [Steam Machine] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7483` | Navi 33 [Radeon RX 7600M/7600M XT] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7487` | Navi 33 [Radeon Graphics] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7489` | Navi 33 [Radeon Pro W7500] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:748b` | Navi 33 [Radeon Graphics] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7499` | Navi 33 [Radeon RX 7400/7300/Pro W7400] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:749f` | Navi 33 [Radeon RX 7500] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73f0` | Navi 33 [Radeon RX 7600M XT] | Navi 33 → gfx1102 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73a1` | Navi 21 [Radeon Pro V620] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73a2` | Navi 21 Pro-XTA [Radeon Pro W6900X] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73a3` | Navi 21 GL-XL [Radeon PRO W6800] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73a5` | Navi 21 [Radeon RX 6950 XT] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73ab` | Navi 21 Pro-XLA [Radeon Pro W6800X/Radeon Pro W6800X Duo] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73ae` | Navi 21 [Radeon Pro V620 MxGPU] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73af` | Navi 21 [Radeon RX 6900 XT] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73bf` | Navi 21 [Radeon RX 6800/6800 XT / 6900 XT] | Navi 21 → gfx1030 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73c3` | Navi 22 | Navi 22 → gfx1031 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73df` | Navi 22 [Radeon RX 6700/6700 XT/6750 XT / 6800M/6850M XT] | Navi 22 → gfx1031 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73e0` | Navi 23 | Navi 23 → gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73e1` | Navi 23 WKS-XM [Radeon PRO W6600M] | Navi 23 → gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73e3` | Navi 23 WKS-XL [Radeon PRO W6600] | Navi 23 → gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73ef` | Navi 23 [Radeon RX 6650 XT / 6700S / 6800S] | Navi 23 → gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:73ff` | Navi 23 [Radeon RX 6600/6600 XT/6600M] | Navi 23 → gfx1032 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7421` | Navi 24 [Radeon PRO W6500M] | Navi 24 → gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7422` | Navi 24 [Radeon PRO W6400] | Navi 24 → gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7423` | Navi 24 [Radeon PRO W6300/W6300M] | Navi 24 → gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:7424` | Navi 24 [Radeon RX 6300] | Navi 24 → gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:743f` | Navi 24 [Radeon RX 6400/6500 XT/6500M] | Navi 24 → gfx1034 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:15bf` | Phoenix1 [Radeon 740M / 760M / 780M Graphics] | Phoenix → gfx1103 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:15c8` | Phoenix2 [Radeon 740M Graphics] | Phoenix → gfx1103 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1900` | HawkPoint1 [Radeon 760M / 780M Graphics] | Hawk Point → gfx1103 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1901` | HawkPoint2 [Radeon 740M Graphics] | Hawk Point → gfx1103 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1681` | Rembrandt [Radeon 660M / 680M] | Rembrandt → gfx1035 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:164e` | Raphael [Radeon 610M] | Raphael → gfx1036 | 断る: 2 CU の画面用の内蔵 GPU | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:13c0` | Granite Ridge [Radeon 610M] | Granite Ridge → gfx1036 | 断る: 2 CU の画面用の内蔵 GPU | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1506` | Mendocino [Radeon 610M] | Mendocino → gfx1036 | 断る: 2 CU の画面用の内蔵 GPU | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:163f` | VanGogh [AMD Custom GPU 0405] | Van Gogh → gfx1033 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1435` | Sephiroth [AMD Custom GPU 0405] | Van Gogh (Sephiroth) → gfx1033 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:150e` | Strix [Radeon 880M / 890M] | Strix → gfx1150 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1586` | Strix Halo [Radeon Graphics / Radeon 8050S Graphics / Radeon 8060S Graphics] | Strix Halo → gfx1151 | 使う(Windows 11・Linux。pytorch.org の ROCm 版と TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `1002:1114` | Krackan [Radeon 840M / 860M Graphics] | Krackan → gfx1152 | 使う(Windows 11・Linux。TheRock) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56a0` | DG2 [Arc A770] | Arc A770 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56a1` | DG2 [Arc A750] | Arc A750 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56a2` | DG2 [Arc A580] | Arc A580 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56a5` | DG2 [Arc A380] | Arc A380 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56a6` | DG2 [Arc A310] | Arc A310 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5690` | DG2 [Arc A770M] | Arc A770M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5691` | DG2 [Arc A730M] | Arc A730M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5692` | DG2 [Arc A550M] | Arc A550M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5693` | DG2 [Arc A370M] | Arc A370M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5694` | DG2 [Arc A350M] | Arc A350M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5696` | DG2 [Arc A570M] | Arc A570M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:5697` | DG2 [Arc A530M] | Arc A530M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56b0` | DG2 [Arc Pro A30M] | Arc Pro A30M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56b1` | DG2 [Arc Pro A40/A50] | Arc Pro A40 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56b2` | DG2 [Arc Pro A60M] | Arc Pro A60M → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:56b3` | DG2 [Arc Pro A60] | Arc Pro A60 → Arc A-Series (Alchemist) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e20b` | Battlemage G21 [Arc B580] | Arc B580 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e20c` | Battlemage G21 [Arc B570] | Arc B570 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e211` | Battlemage G21 [Arc Pro B60] | Arc Pro B60 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e212` | Battlemage G21 [Arc Pro B50] | Arc Pro B50 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e202` | Battlemage G21 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e20d` | Battlemage G21 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e210` | Battlemage G21 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e215` | Battlemage G21 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e216` | Battlemage G21 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e220` | Battlemage G31 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e221` | Battlemage G31 [Intel Graphics] | Battlemage → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e222` | Battlemage G31 [Arc Pro B65] | Arc Pro B65 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:e223` | Battlemage G31 [Arc Pro B70] | Arc Pro B70 → Arc B-Series (Battlemage) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:7d55` | Meteor Lake-P [Intel Arc Graphics] | Arc Graphics → Core Ultra (Meteor Lake-H) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:7d51` | Arrow Lake-P [Arc Pro 130T/140T] | Arc 140T → Core Ultra Series 2 (Arrow Lake-H) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:64a0` | Core Ultra 200V Series Processors Arc Graphics 130V/140V GPU | Arc 140V → Core Ultra Series 2 (Lunar Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b080` | Panther Lake [Arc B390] | Arc B390 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b082` | Panther Lake [Arc B390] | Arc B390 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b084` | Panther Lake [Arc Pro B390] | Arc B390 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b086` | Panther Lake [Arc Pro B390] | Arc B390 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b081` | Panther Lake [Arc B370] | Arc B370 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b083` | Panther Lake [Arc B370] | Arc B370 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b085` | Panther Lake [Arc Pro B370] | Arc B370 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:b087` | Panther Lake [Arc Pro B370] | Arc B370 → Core Ultra Series 3 (Panther Lake) | 使う(Windows 11・Linux・WSL2) | `pci-ids` | 2026-10-27 | 0.10 以前 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0b69` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1450] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0b6e` | Ponte Vecchio XT (1 Tile) [Data Center GPU Max 1100C] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd0` | Ponte Vecchio XL (2 Tile) | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd4` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1550VG] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd5` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1550] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd6` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1550] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd7` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1350] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd8` | Ponte Vecchio XT (2 Tile) [Data Center GPU Max 1350] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bd9` | Ponte Vecchio XT (1 Tile) [Data Center GPU Max 1100] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bda` | Ponte Vecchio XT (1 Tile) [Data Center GPU Max 1100] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |
| AMD / Intel | `PCI_DEVICE_RULES` | `8086:0bdb` | Ponte Vecchio XT (1 Tile) [Data Center GPU Max 1100] | Data Center GPU Max → Data Center GPU Max (Ponte Vecchio) | 使う(Linux だけ) | `pci-ids` | 2026-10-27 | 0.11.0 |

## PyTorch の CUDA 版のビルド

| ビルド | CUDA | x86_64 | aarch64 | 最低のドライバ(Win / Linux) | PyTorch | 出典 | 確かめた日 |
|---|---|---|---|---|---|---|---|
| cu132 | 13.2 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | 8.0, 9.0, 10.0, 11.0, 12.0 | 580 / 580 | 2.12〜2.14 | `torch-cuda-init`, `torch-whl-index`, `cuda-rn` | 2026-10-27 |
| cu130 | 13.0 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | 8.0, 9.0, 10.0, 11.0, 12.0 | 580 / 580 | 2.9〜2.14 | `torch-cuda-init`, `torch-whl-index`, `cuda-rn` | 2026-10-27 |
| cu129 | 12.9 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | 8.0, 9.0, 10.0, 12.0 | 525 / 525 | 2.8〜2.13(Windows の cu129 の wheel は 2.8.0・2.9.0 だけ(2.10 以降は Linux だけ)) | `torch-build-cuda`, `torch-whl-index`, `cuda-rn` | 2026-10-27 |
| cu128 | 12.8 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | 8.0, 9.0, 10.0, 12.0 | 525 / 525 | 2.7〜2.11(aarch64 の wheel は 2.7・2.9〜2.11(2.8 は無い)) | `torch-build-cuda`, `torch-whl-index`, `cuda-rn` | 2026-10-27 |
| cu126 | 12.6 | 5.0, 6.0, 6.1, 7.0, 7.5, 8.0, 8.6, 9.0 | 8.0, 9.0 | 525 / 525 | 2.6〜2.14 | `torch-cuda-init`, `torch-win-cuda`, `torch-whl-index`, `cuda-rn` | 2026-10-27 |

## 値の集まり

| 名前 | 値 | 出典 | 確かめた日 | 注意 |
|---|---|---|---|---|
| `solo_cc` | 53, 62, 72, 87, 101, 110 | `torch-cuda-init` | 2026-10-27 | From CUDA 13.2, 8.0 / 8.6 code runs on 8.7 (PyTorch v2.14.1), but Jetson is out of scope, so unchanged |
| `nvidia_10_series` | 1080, 1070, titan x, p3000, p3200, p4000, p4200, p5000, p5200, p6000, 1060, 1050, p40, p100, p6, p4 | `comfyui-mm` | 2026-10-27 | - |
| `nvidia_16_series` | 1660, 1650, 1630, T500, T550, T600, MX550, MX450, CMP 30HX, T2000, T1000, T1200 | `comfyui-mm` | 2026-10-27 | - |
| `display_only_gfx` | gfx1036 | `amd-product-pages`, `therock-supported` | 2026-10-22 | 2-CU display-only integrated GPU (a judgment; no measurement) |
| `unverified_gfx` | gfx1170, gfx1171, gfx1250 | `therock-supported` | 2026-10-27 | Not Release Ready in TheRock (gfx117x: Build Passing only; gfx1250: up to Sanity Tested) |
| `pytorch_rocm_linux_gfx` | gfx1030, gfx1100, gfx1101, gfx1102, gfx1103, gfx1150, gfx1151, gfx1200, gfx1201, gfx900, gfx906, gfx908, gfx90a, gfx942, gfx950 | `torch-build-rocm` | 2026-10-27 | - |
| `wsl_amd_supported` | ai pro r9600d, ai pro r9700, pro w7700, pro w7800, pro w7800 48gb, pro w7900, pro w7900 dual slot, rx 7700, rx 7700 xt, rx 7800 xt, rx 7900 gre, rx 7900 xt, rx 7900 xtx, rx 9060, rx 9060 xt, rx 9070, rx 9070 gre, rx 9070 xt | `amd-wsl-compat` | 2026-10-27 | - |

## わざと合わせないもの

- `amd:Radeon 820M=gfx1152`: 820M は TheRock の gfx1153 に合わせた(公式の資料どうしの食い違い。820M の行の注意)
- `torch:cu126 x86_64`: Windows の cu126 は 6.1 も入れている(.ci/pytorch/windows/internal/cuda_config.bat)

## 出典

- `nvidia-cuda-gpus`: NVIDIA CUDA GPUs (compute capability) — https://developer.nvidia.com/cuda-gpus (2026-10-27)
- `nvidia-cuda-legacy`: NVIDIA Legacy CUDA GPUs (compute capability) — https://developer.nvidia.com/cuda-legacy-gpus (2026-10-27)
- `cuda-rn`: CUDA Toolkit release notes, Table 3 (minimum driver for minor version compatibility) and Table 4 — https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html (2026-10-27)
- `torch-cuda-init`: PyTorch v2.14.1 torch/cuda/__init__.py (PYTORCH_RELEASES_CODE_CC, DEVICE_REQUIREMENT) — https://github.com/pytorch/pytorch/blob/v2.14.1/torch/cuda/__init__.py (2026-10-27)
- `torch-build-cuda`: PyTorch v2.11.0 .ci/manywheel/build_cuda.sh (TORCH_CUDA_ARCH_LIST of cu128 / cu129) — https://github.com/pytorch/pytorch/blob/v2.11.0/.ci/manywheel/build_cuda.sh (2026-10-27)
- `torch-win-cuda`: PyTorch .ci/pytorch/windows/internal/cuda_config.bat / build_env_setup.py (Windows cu126 adds 6.1) — https://github.com/pytorch/pytorch/tree/v2.14.1/.ci/pytorch/windows/internal (2026-10-21)
- `torch-whl-index`: download.pytorch.org torch wheel index (PEP 503) — https://download.pytorch.org/whl/torch/ (2026-10-27)
- `torch-build-rocm`: PyTorch v2.14.1 .ci/docker/manywheel/build.sh (PYTORCH_ROCM_ARCH) — https://github.com/pytorch/pytorch/blob/v2.14.1/.ci/docker/manywheel/build.sh (2026-10-27)
- `comfyui-mm`: ComfyUI comfy/model_management.py (should_use_fp16, AMD_RDNA2_AND_OLDER_ARCH) — https://github.com/Comfy-Org/ComfyUI/blob/426f5ee/comfy/model_management.py (2026-10-27)
- `comfyui-readme`: ComfyUI README: AMD GPUs (Windows, ROCm 10.0) — RDNA 2, RDNA 3, RDNA 3.5, RDNA 4; Windows 11 — https://github.com/Comfy-Org/ComfyUI/blob/426f5ee/README.md (2026-10-21)
- `therock-supported`: ROCm/TheRock SUPPORTED_GPUS.md (Build Passing / Sanity Tested / Release Ready) — https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md (2026-10-27)
- `therock-releases`: ROCm/TheRock RELEASES.md (GFX target lookup table) — https://github.com/ROCm/TheRock/blob/main/RELEASES.md (2026-10-27)
- `rocm-hw-specs`: ROCm GPU hardware specifications (LLVM target name) — https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html (2026-10-27)
- `radeon-win-compat`: AMD Radeon: Windows compatibility (ROCm 7.2.1) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html (2026-10-27)
- `ryzen-win-compat`: AMD Ryzen: Windows compatibility (ROCm 7.2.1) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/windows/windows_compatibility.html (2026-10-27)
- `radeon-linux-compat`: AMD Radeon: Linux compatibility (ROCm 7.2.1) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html (2026-10-27)
- `ryzen-linux-compat`: AMD Ryzen: Linux compatibility (ROCm 7.2.1) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html (2026-10-27)
- `amd-wsl-compat`: AMD Radeon: WSL compatibility (ROCm 7.2.1) — https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html (2026-10-27)
- `amd-product-pages`: AMD product pages (Graphics Model / Graphics Core Count), e.g. Ryzen 9 7940HX, Ryzen Z1 / Z1 Extreme — https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hx.html (2026-10-22)
- `linux-amdkfd`: Linux drivers/gpu/drm/amd/amdkfd/kfd_device.c (GC IP version → gfx_target_version) — https://github.com/torvalds/linux/blob/master/drivers/gpu/drm/amd/amdkfd/kfd_device.c (2026-10-22)
- `steam-deck-spec`: Valve Steam Deck tech specs (8 RDNA 2 CUs) — https://www.steamdeck.com/en/tech (2026-10-21)
- `pci-ids`: pci.ids (pciutils/pciids, version 2026.10.27) — https://raw.githubusercontent.com/pciutils/pciids/master/pci.ids (2026-10-27)
- `llvm-amdgpu`: LLVM AMDGPUUsage, processors table (gfx1153 / gfx1170 / gfx1171 / gfx1250: products TBA) — https://llvm.org/docs/AMDGPUUsage.html (2026-10-27)
- `pytorch-xpu`: PyTorch 2.14 Getting Started on Intel GPU (Hardware Prerequisite) — https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html (2026-10-27)
- `intel-arc-mobile`: Intel support 000091109 (Arc A-series mobile graphics memory) — https://www.intel.com/content/www/us/en/support/articles/000091109.html (2026-10-21)
- `intel-arc-desktop`: Intel Arc A / B-series desktop and Arc Pro B50 / B60 product pages (graphics memory) — https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html (2026-10-21)
