# ハードウェアの検出と判定(#21)

この PC で何を調べ、どう判定に使うか。実装は `src/genmodelfit/hardware.py`(検出)と
`src/genmodelfit/estimate.py`(`gpu_precision`・`ram_limit`)。出力の形は `docs/schema.md`「System」。
一次資料は 2026-10-20 に読んだ(版・コミットは各節に書く)。llmfit の `llmfit-core/src/hardware.rs`
(コミット bffd1b6、MIT)は考え方だけを参考にし、コードは写していない。

## 何を調べるか

| 項目 | どこから | 判定に使うか |
|---|---|---|
| NVIDIA の GPU(名前・VRAM の総量・空き・使用量・ドライバ・compute capability) | `nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version,compute_cap --format=csv,noheader,nounits` | 使う(backend が `cuda` のものの最大の 1 枚) |
| NVIDIA 以外も含めた画面の部品 | Windows: PowerShell の `Get-CimInstance Win32_VideoController` + レジストリの `HardwareInformation.qwMemorySize`。Linux: `lspci -Dnn`(AMD の VRAM は sysfs の `mem_info_vram_total`) | **使わない**(知らせるだけ) |
| 物理メモリ | psutil(Linux はコンテナ(cgroup)の上限があればそれ) | 使う |
| 仮想メモリ | Windows: `GlobalMemoryStatusEx`(ctypes)とレジストリ `Memory Management\PagingFiles`。Linux: `/proc/swaps`(zram を除く)。macOS: 自動で伸びる扱い | 使う(足りないときの上限) |
| ディスクの空き | `shutil.disk_usage` | 表示だけ |

調べないもの・残さないもの: PC 名・ユーザ名・GPU の UUID・PCI の差し場所(bus id)・PNPDeviceID の後半(個体・差し場所の ID)。
WMI の PNPDeviceID は、型番を表す前半(`PCI\VEN_xxxx&DEV_xxxx&SUBSYS_…&REV_…`)をレジストリとの突き合わせにだけ使い、
出力にも SystemInfo にも残さない(`tests/test_hardware.py::test_system_json_has_no_private_values`)。
取った情報は手元で使うだけで、どこにも送らない。

## nvidia-smi

- 場所: PATH → `%SystemRoot%\System32\nvidia-smi.exe`(今の Windows のドライバ)→
  `%ProgramFiles%\NVIDIA Corporation\NVSMI\nvidia-smi.exe`(古いドライバ)→ Linux の `/usr/bin`・
  WSL の `/usr/lib/wsl/lib`・`/usr/local/bin`。
- 時間制限 10 秒(普段は 0.1〜0.5 秒)。Windows では窓を出さない(`CREATE_NO_WINDOW`。GUI のアプリから呼ばれるため)。
- 出力は UTF-8 として読み、読めない字は置き換える。`, ` で区切り、項目の数が合わない行は飛ばす
  (エラーの文言・空行・壊れた行)。`[N/A]`・`[Not Supported]`・`N/A`・空は「分からない」。
  空きが分からず総量と使用量があれば 総量 − 使用量。単位付き(`8192 MiB`)・カンマの小数点(`8,6`)も読む。
- VRAM の総量が返らない GPU(メモリを CPU と共有する GB10 など)は判定に使わない(backend `none`、理由付き)。
- `compute_cap` を知らない古い nvidia-smi は「Field "compute_cap" is not a valid field to query.」で終了コード 2 になる
  (この PC で存在しない項目を聞いて確かめた。どのドライバの版で入ったかは一次資料で確かめられなかった)。
  そのときは `compute_cap` を除いて聞き直し、GPU 名の表(`hardware._NAME_CC`)から推す。迷う名前(ノート用の
  GTX 960M など、同じ型番で世代が混ざるもの)は推さない。表は NVIDIA の一覧
  (https://developer.nvidia.com/cuda-gpus 、https://developer.nvidia.com/cuda-legacy-gpus)に合わせた。
  例: 「NVIDIA TITAN X」は Pascal(6.1)、「GeForce GTX TITAN X」は Maxwell(5.2)、Quadro K620 / K1200 / K2200 は Maxwell(5.0)、
  ノート用の RTX 2050 は Ampere(8.6)。名前から推した値は `compute_capability_source: "name"`。
- 複数の GPU: ComfyUI は 1 枚で動くので、判定に使えるもの(backend `cuda`)のうち VRAM が最大の 1 枚を使う。

## 世代と PyTorch・ドライバ

### PyTorch の CUDA 版のビルドが対応する compute capability

PyTorch の配布物は機械語(SASS)だけを入れる(`.ci/manywheel/build_env_setup.py` v2.14.1 の `_ptx_arches`:
「Release/RC wheels ship SASS-only」)。機械語は「同じ major で minor が同じか大きい GPU」でだけ動く
(CUDA Programming Guide 1.3.4.1 Binary Compatibility:
https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/cuda-platform.html)。例: 8.6 向けは 8.6・8.9 で動き、
8.0 や 9.0 では動かない。Jetson の 5.3・6.2・7.2・8.7 などは別扱い(PyTorch の `DEVICE_REQUIREMENT`)。

| ビルド | PyTorch の版 | x86_64 の compute capability | 出典 |
|---|---|---|---|
| cu132 | 2.12〜2.14 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | v2.14.1 `torch/cuda/__init__.py` `PYTORCH_RELEASES_CODE_CC`、`.ci/pytorch/windows/build_env_setup.py` |
| cu130 | 2.9〜2.14 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | 同上 |
| cu129 | 2.8〜2.13 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | v2.13.0 `PYTORCH_RELEASES_CODE_CC`、`.ci/pytorch/windows/internal/cuda_config.bat` |
| cu128 | 2.7〜2.11 | 7.5, 8.0, 8.6, 9.0, 10.0, 12.0 | v2.11.0 `.ci/manywheel/build_cuda.sh`・`cuda_config.bat` |
| cu126 | 〜2.14 | 5.0, 6.0, 6.1(Windows), 7.0, 7.5, 8.0, 8.6, 9.0 | v2.14.1 同上 |

各版の CUDA 版は `.github/scripts/generate_binary_build_matrix.py` の `CUDA_ARCHES`
(2.7: 11.8 / 12.6 / 12.8、2.8: 12.6 / 12.8 / 12.9、2.9: 12.6 / 12.8 / 13.0、2.10・2.11: 12.6 / 12.8 / 12.9 / 13.0、
2.12: 12.6 / 13.0 / 13.2、2.13: 12.6 / 12.9 / 13.0 / 13.2、2.14.1: 12.6 / 13.0 / 13.2)。
cu128・cu129 は今の版(2.14)には無いが、配布先に残っていて、ComfyUI の同梱版や AI生成キットのように版を固定して
使う側があるので数える(ドライバ 580 未満の Blackwell が動く道)。

### CUDA の最低ドライバ

NVIDIA CUDA Toolkit 13.4 Update 2 のリリースノート(https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)の
表 3「CUDA Toolkit and Minimum Required Driver Version for CUDA Minor Version Compatibility」: 13.x は 580 以上、
12.x は 525 以上 580 未満(それより新しいドライバでも動く)。表 4 の CUDA 12.0 GA: Linux 525.60.13 / Windows 527.41。
PyTorch の配布物は機械語だけで PTX の JIT を使わないので、マイナー版の互換の範囲で動くとみなした。

| ビルド | Windows | Linux |
|---|---|---|
| cu126 / cu128 / cu129 | 527.41 以上 | 525.60.13 以上 |
| cu130 / cu132 | 580 以上 | 580 以上 |

WSL の nvidia-smi は Windows のドライバの版を返すので、Windows の表で見る。

### 判定

| 世代 | compute capability | 動くビルド | ドライバの条件 | 判定 |
|---|---|---|---|---|
| Fermi 以前・Kepler | 〜3.7 | なし | — | **CPU**(ドライバを更新しても動かない) |
| Maxwell | 5.0, 5.2 | cu126 | 527.41 / 525.60.13 | GPU |
| Pascal | 6.0, 6.1 | cu126 | 同上 | GPU |
| Volta | 7.0 | cu126 | 同上 | GPU |
| Turing | 7.5 | cu126〜cu132 | 同上(cu13x は 580) | GPU |
| Ampere | 8.0, 8.6 | cu126〜cu132 | 同上 | GPU |
| Ada | 8.9 | cu126〜cu132(8.6 の機械語) | 同上 | GPU |
| Hopper | 9.0 | cu126〜cu132 | 同上 | GPU |
| Blackwell | 10.x, 12.x | cu128〜cu132 | 527.41(cu128)/ 580(cu13x) | GPU |
| それより新しい | 13.x〜 | なし | — | **CPU**(PyTorch の対応を待つ) |

- 動くビルドが 1 つも無い世代、またはドライバがどのビルドにも足りないときは backend `none` にして CPU として判定する。
  理由は `Gpu.notes` と `SystemInfo.notes`(「ドライバを更新すれば GPU で動く」かどうかも書く)。
- 世代が分からない(nvidia-smi も名前の表も分からない)・ドライバの版が読めないときは断らない(分からないと書く)。
- `torch_cuda` は動くビルドの一覧(新しい順)で、参考の情報。判定には使っていない。
- AI生成キットの知見(Kepler は CPU、Maxwell・Pascal・Volta は cu126、Quadro K620/K1200/K2200 は Maxwell)と一致する。

## 精度(ComfyUI が選ぶ dtype と VRAM)

ComfyUI(コミット b1a247c、2026-10-20)の `comfy/model_management.py` と `comfy/sd.py` を読んだ(GPL-3.0。import しない。
名前の一覧だけを事実として写した: `hardware._COMFY_FP16_PASCAL`・`_COMFY_FP16_BROKEN`)。起動の引数を付けない既定の動き:

- `should_use_fp16`: major ≥ 8 は使う。major < 6(Maxwell 以前)は使わない。名前に 1080 / 1070 / 1060 / 1050 / titan x /
  p3000 … / p6 / p4 を含む GPU(Pascal)は **Windows だけ**使う(Linux は fp32 のほうが速いとして使わない)。
  それ以外の major < 7(GT 1030 など)は使わない。1660 / 1650 / 1630 / T500 / T550 / T600 / MX550 / MX450 / CMP 30HX /
  T2000 / T1000 / T1200 を名前に含む GPU は「fp16 が壊れている」として使わない。それ以外(Volta・Turing の RTX)は使う。
- `should_use_bf16`: major ≥ 8。`supports_fp8_compute`: 8.9 以上(Windows は torch 2.4 以上)。
- 本体の重み(`unet_dtype` / `unet_manual_cast`): bf16 を持たない GPU では bf16 の重みを fp16 で載せる(同じ 2 バイト)。
  半精度を使わない GPU では、fp32 にしても `total × 0.88 − 最小の作業用メモリ` に収まるときだけ fp32 に広げ、
  収まらなければ fp16 のまま計算だけ fp32 にする。fp8 の重みは、fp8 の計算を持たない GPU では、fp16 に広げても収まれば
  広げ、収まらなければ fp8 のまま(計算のたびに変換)。fp8_scaled など量子化の設定を持つ重みは fp8 のまま。
- テキストエンコーダ: 既定は fp16。ファイルの精度が 2 バイト以下ならその精度のまま載せる(`pick_weight_dtype`)。
- VAE(`vae_dtype`): fp16 / bf16 を使えなければ fp32。
- 作業用メモリ: ComfyUI 自身の見積もり(`model_base.memory_required`、VAE の `memory_used_decode`)が
  計算の精度の大きさ(`dtype_size`)を掛けている。fp32 で計算すると 2 倍。

見積もりへの効かせ方(`estimate.gpu_precision`。コードで確かなところだけ):

| GPU | 例 | Gpu.compute_dtype | 見積もりの直し |
|---|---|---|---|
| 半精度を使わない | Maxwell 全部、GT 1030、Linux の GTX 1080、GTX 1660、T1000 | `fp32` | 作業用メモリ × 2、VAE の重み × 2(VAE のファイルが fp32 なら × 1)。本体は変えない(載らなければ fp16 のままなので)。理由に書く |
| fp16 は使うが bf16 を持たない | Windows の Pascal(1080 など)、Volta、Turing の RTX | `fp16` | 変えない。bf16 の重みには「fp16 で動かす(VRAM は同じ)」と書く |
| fp8 の計算を持たない | Ada より前 | — | 変えない。fp8 の重みには「余裕があれば fp16 に広げる・無ければ遅い」と書く |
| 上のどれでもない | Ampere・Ada・Hopper・Blackwell | `fp16` | 変えない |

本体を広げる動き(fp32・fp16)は VRAM の量で変わるので、見積もりには入れない(入れると VRAM を増やしたときに段階が下がる)。
半精度を使わない GPU では、その GPU の実測でない限り `basis` は `estimated` にする(実測は半精度の GPU のもの)。

## NVIDIA 以外の GPU(知らせるだけ)

- **判定には使わない**。v0.1 は NVIDIA の CUDA だけ。レジストリには取り外した GPU の項目も残る。
- 既定では、nvidia-smi で NVIDIA の GPU が見つからず、`--vram` で上書きしていないときだけ調べる
  (AMD・Intel だけの PC で「CPU として判定した」理由を出すため。NVIDIA のドライバが入っていない NVIDIA の GPU も分かる)。
  NVIDIA の GPU があるときは調べない(PowerShell の起動に約 1 秒かかるため)。`genmodelfit system --all-gpus` で必ず調べる。
- Windows: `powershell.exe -NoProfile -NonInteractive -Command` で `Get-CimInstance Win32_VideoController` の
  Name・AdapterRAM・PNPDeviceID・ConfigManagerErrorCode を JSON で取る(出力は UTF-8 に指定。時間制限 8 秒)。
  PNPDeviceID が `PCI\` で始まらないもの(仮想の画面・リモートデスクトップの画面)は捨てる。
  - `AdapterRAM` は uint32(Microsoft の Win32_VideoController の説明:
    https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-videocontroller)なので 4 GiB 以上は
    4293918720 前後で頭打ちになる。この値以上は使わない。
  - VRAM はレジストリ `HKLM\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\00xx` の
    `HardwareInformation.qwMemorySize`(REG_QWORD。この PC の RTX 3070 で 8589934592)を使う。Microsoft の文書
    (https://learn.microsoft.com/en-us/windows-hardware/drivers/display/registering-hardware-information)にあるのは
    `HardwareInformation.MemorySize`(ULONG、MB と書いてあるが実際のドライバはバイトで書く。llmfit の場合分けと同じ)だけで、
    qwMemorySize はドライバの慣習。64 MiB 未満は単位の違いとみなして使わない。
  - レジストリの項目は、`MatchingDeviceId` が WMI の部品の型番の ID の先頭に当たるもの、無ければ `DriverDesc` の名前
    (™・® を除いて比べる)が同じものに当てる。当たる項目が複数あって値が違う(取り外した同名の GPU が残っている)ときは使わない。
  - NVIDIA の部品(VEN_10DE)が「Microsoft Basic Display Adapter」なら「NVIDIA のドライバが入っていない」と知らせる。
    ConfigManagerErrorCode が 0 でなければ「デバイスマネージャーで問題がある状態」と知らせる。
- Linux: `lspci -Dnn` のクラス 03xx(VGA・3D・Display)を読む。名前は括弧の中の製品名(`GA104 [GeForce RTX 3070]`)。
  AMD は `/sys/bus/pci/devices/<slot>/mem_info_vram_total` の VRAM を表示に使う。
- macOS: Apple の GPU は判定しない(CPU として判定し、そう書く)。
- PowerShell・lspci が無い・失敗した・時間切れは、黙って飛ばす(空のまま)。

## メモリと仮想メモリ

- 搭載量・空きは psutil。Linux でコンテナ(cgroup v2 の `memory.max`、v1 の `memory.limit_in_bytes`)の上限が搭載量より
  小さければ上限を搭載量として使う。WSL の中では WSL に割り当てた分(`.wslconfig`)になる(そう書く)。
- Windows のコミット: `GlobalMemoryStatusEx` の `ullTotalPageFile` は「今のコミットの上限(システムかこのプロセスの小さいほう)」、
  `ullAvailPageFile` はその空き(https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/ns-sysinfoapi-memorystatusex)。
  `commit_limit_gb`・`commit_used_gb` に出し、今のページファイルの大きさは 上限 − 物理メモリ(`swap_total_gb`)。
- ページファイルの設定: レジストリ `HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Memory Management` の
  `PagingFiles`(REG_MULTI_SZ)。この PC は `?:\pagefile.sys`(すべてのドライブで自動)。`C:\pagefile.sys 0 0`・大きさの無い行は
  システム管理、`C:\pagefile.sys 1024 4096` は最小・最大 MB、空はページファイル無し(この書式は Microsoft の文書では確かめられず、
  実機の値と一般に知られた書式による)。システム管理のページファイルは最大で「RAM の 3 倍か 4 GB の大きいほう(ボリュームの 1/8 まで)」
  まで伸びる(https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows)。
- Linux のスワップ: `/proc/swaps` の合計。zram(メモリを圧縮して使うもの)は ComfyUI の `get_disk_swap_total` と同じく数えない。

判定(`estimate.ram_limit`): メインメモリが搭載量より多く要るとき、

| 仮想メモリ | 動くとみなす上限 | 段階 |
|---|---|---|
| 大きさが決まっている(固定のページファイル・無し・Linux のスワップ) | 搭載量 + 最大の大きさ(ただし搭載量 × 1.5 まで) | 上限まで marginal、超えたら too_tight |
| システム管理で伸びる・分からない・`--ram` で上書きした | 搭載量 × 1.5(これまでどおり) | 同上 |

1.5 倍は「重みをページファイルに置くと極端に遅い」ための判断の値で、実測していない(docs/accuracy.md)。
ページファイルの大きさは搭載量に比べて増えるだけなので、搭載量を増やして段階が下がることはない(`test_fit_monotonic_in_ram_with_fixed_pagefile`)。

## 画面などが使う VRAM(今の空き)

- nvidia-smi の `memory.used`・`memory.free` を `vram_used_gb`・`vram_free_gb` に出す(この PC は画面・ブラウザなどで約 3.6 GB 使用中)。
- **段階は総量で決める**(使う VRAM はほかのアプリを閉じれば空くため)。判定した動かし方で要る VRAM が今の空きより多いときは、
  見積もりの `vram_free_short: true` と「今の空き VRAM は … ほかのアプリを閉じれば動く」の理由で知らせる。
- 目安の予約: ComfyUI は Windows で 600 MB(VRAM が 15 GB を超える GPU は 700 MB)、それ以外で 400 MB を、ほかのアプリのために空けておく
  (`EXTRA_RESERVED_VRAM`)。見積もりの固定分(`overhead_gb`、実測で合わせた値)がこれと CUDA の文脈などを含む(docs/accuracy.md)。
- llmfit との違い: llmfit(bffd1b6 の `gpu_fit_pool_gb`)は空きが分かれば `min(空き, 総量)` で段階を決める。
  こちらは画像・動画の生成は 1 回ずつ動かすもので、ほかのアプリを閉じれば済むため、段階は総量で決めて注意だけ出す。

## llmfit との突き合わせ

| llmfit の場合分け | こちら |
|---|---|
| nvidia-smi の `addressing_mode`(ATS)で CPU と共有するメモリの GPU(Grace Blackwell・Tegra)を見分け、RAM を VRAM とみなす | VRAM が返らない GPU は判定に使わない(v0.1 は外付けの GPU だけ)。DGX Spark などは対象外と書く |
| nvidia-smi が VRAM を返さないとき GPU 名から推す | 推さない(VRAM を推して「載る」と言わない) |
| 同じ型番の複数枚をまとめて VRAM を足す(llama.cpp などは分けて載せられる) | 足さない(ComfyUI は 1 枚で動く)。最大の 1 枚 |
| Linux で nvidia-smi が無いとき sysfs・lspci で NVIDIA の GPU を見つける | 判定には使わない(CUDA が動くか分からない)。`--all-gpus` などの一覧に出すだけ |
| AMD(rocm-smi・sysfs)・Intel(lspci)・Apple(system_profiler)を判定に使う | 使わない(v0.1 は NVIDIA だけ)。知らせるだけ |
| WMI の AdapterRAM の頭打ちをレジストリの qwMemorySize で直す(issue #830)、DriverDesc で突き合わせ、曖昧なら使わない | 同じ考え方。加えて MatchingDeviceId(型番の ID)で先に突き合わせ、PCI でない部品を捨てる |
| 内蔵 GPU(Intel UHD など)を外付けがあれば落とす | 判定に使わないので落とさない(一覧に出すだけ) |
| AMD の APU の UMA の取り分を搭載量に足し戻す | しない(v0.1 は NVIDIA だけ。必要になったら足す) |
| `available_ram_gb` が 0 のときの代わりの読み方(macOS) | psutil の値のまま |
| compute capability・ドライバ・PyTorch のビルドの対応 | llmfit には無い。こちらで足した(上の「世代と PyTorch・ドライバ」) |
| ページファイル・スワップ | llmfit には無い。こちらで足した(上の「メモリと仮想メモリ」) |

## 時間

この PC(Windows 11、RTX 3070、i7-13700KF、ほかの作業で負荷あり)で 2026-10-20 に測った中央値:

| 調べ方 | 時間 |
|---|---|
| 以前の `detect()`(nvidia-smi 1 回) | 約 0.1〜0.25 秒 |
| 今の `detect()`(既定。NVIDIA の GPU があれば nvidia-smi 1 回 + ctypes + レジストリ) | 約 0.15〜0.4 秒(nvidia-smi のばらつきの範囲。ctypes・レジストリは 1 ms 未満) |
| `detect(other_gpus=True)` / `system --all-gpus` | 約 1.0〜1.2 秒(PowerShell の起動が約 1 秒) |

nvidia-smi だけを 15 回ずつ交互に動かすと、項目を増やした問い合わせと以前の問い合わせの差は測れなかった(どちらも 0.1〜0.25 秒)。

## 試験

`tests/test_hardware.py` と `tests/fixtures/hardware/`。既定の試験は本物の nvidia-smi・PowerShell・lspci・レジストリを呼ばない
(`tests/conftest.py` の `_no_real_hardware` が `hardware._run`・`_reg_display_entries`・`_reg_paging_files` を塞ぐ)。
フィクスチャ: この PC の nvidia-smi・WMI・レジストリの実際の出力(個体の ID は 0 に置き換えた)、複数の GPU、compute_cap の無い古い版、
`[N/A]`、壊れた出力、ドライバが動いていない・GPU が無いときの出力、AMD だけ・Intel + NVIDIA・ドライバの無い NVIDIA の WMI、
取り外した GPU が残るレジストリ、lspci。世代ごと(Kepler・Maxwell・Pascal・Volta・Turing・Ampere・Ada・Hopper・Blackwell)の判定と、
世代を固定して VRAM・RAM を増やしても段階が下がらないことを確かめる。
