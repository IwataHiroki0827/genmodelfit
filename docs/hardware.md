# ハードウェアの検出と判定(#21)

この PC で何を調べ、どう判定に使うか。実装は `src/genmodelfit/hardware.py`(検出)と
`src/genmodelfit/estimate.py`(`gpu_precision`・`ram_limit`)。出力の形は `docs/schema.md`「System」。
一次資料は 2026-10-20 に読んだ(版・コミットは各節に書く)。llmfit の `llmfit-core/src/hardware.rs`
(コミット bffd1b6、MIT)は考え方だけを参考にし、コードは写していない。

## 何を調べるか

| 項目 | どこから | 判定に使うか |
|---|---|---|
| NVIDIA の GPU(名前・VRAM の総量・空き・使用量・ドライバ・compute capability) | `nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver_version,compute_cap --format=csv,noheader,nounits` | 使う(backend が `cuda` のものの最大の 1 枚) |
| NVIDIA 以外も含めた画面の部品 | Windows: PowerShell の `Get-CimInstance Win32_VideoController` + レジストリの `HardwareInformation.qwMemorySize`。Linux: `lspci -Dnn`(AMD の VRAM・GTT は sysfs の `mem_info_vram_total`・`mem_info_gtt_total`)。macOS: `system_profiler SPDisplaysDataType -json` | NVIDIA の GPU が使えないときだけ、AMD・Intel・Apple を対応表で使う(0.7.0、下の「NVIDIA 以外の GPU」) |
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
- 終了コードが 0 でなくても、読める行は使う(1 枚だけ「Unable to determine the device handle … Unknown Error」で
  読めないときなど)。読めなかった枚数を notes に書く(エラーの文言は PCI の差し場所を含むので写さない)。
- 複数の GPU: ComfyUI は 1 枚で動くので、判定に使えるもの(backend `cuda`)のうち VRAM が最大の 1 枚を使う。
  ComfyUI は既定で CUDA の 0 番(`get_torch_device` → `torch.cuda.current_device()`)を使い、CUDA の番号の付け方の既定は
  `CUDA_DEVICE_ORDER=FASTEST_FIRST`(速い順。nvidia-smi の PCI の順とは違うことがある)。どれが 0 番かは nvidia-smi からは
  確かに決められないので、違う型番の GPU が複数あるときは「ComfyUI を `--cuda-device` で最大の GPU に向ける」と notes に書く。
- `--vram` で上書きしたとき: 判定に使える NVIDIA の GPU があればその世代・ドライバ・精度のまま VRAM だけを変える
  (この PC の GPU の VRAM が違ったら、の判定)。無い・使えない世代なら、世代の分からない仮の GPU にする(精度の直しは無し)。
  `--ram` は搭載量だけを変え、この PC のページファイル・コミットは使わない(別の PC の想定なので)。

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
12.x は 525 以上 580 未満(それより新しいドライバでも動く)。表 3 は OS で分けていないので、そのまま使う
(表 4 の 527.41 / 525.60.13 は CUDA 12.0 GA の同梱ドライバの版で、最低の条件ではない)。
PyTorch の配布物は機械語だけで PTX の JIT を使わないので、マイナー版の互換の範囲で動くとみなした。

| ビルド | Windows・Linux |
|---|---|
| cu126 / cu128 / cu129 | 525 以上 |
| cu130 / cu132 | 580 以上 |

世代ごとの最低(`hardware.ARCH_MIN_DRIVER`): Blackwell(10.x・12.x)の機械語が入るのは CUDA 12.8 から
(PyTorch の cu128 以降。上の表)で、表 4 の CUDA 12.8 GA は Linux 570.26 / Windows 570.65。570 未満では CPU として判定する
(それより古いドライバは Blackwell を知らないので、nvidia-smi に出ること自体がまれ)。ほかの世代はビルドの条件と同じ。

WSL の nvidia-smi は Windows のドライバの版を返すので、Windows の表で見る。

### 判定

| 世代 | compute capability | 動くビルド | ドライバの条件 | 判定 |
|---|---|---|---|---|
| Fermi 以前・Kepler | 〜3.7 | なし | — | **CPU**(ドライバを更新しても動かない) |
| Maxwell | 5.0, 5.2 | cu126 | 525 | GPU |
| Pascal | 6.0, 6.1 | cu126 | 同上 | GPU |
| Volta | 7.0 | cu126 | 同上 | GPU |
| Turing | 7.5 | cu126〜cu132 | 同上(cu13x は 580) | GPU |
| Ampere | 8.0, 8.6 | cu126〜cu132 | 同上 | GPU |
| Ada | 8.9 | cu126〜cu132(8.6 の機械語) | 同上 | GPU |
| Hopper | 9.0 | cu126〜cu132 | 同上 | GPU |
| Blackwell | 10.x, 12.x | cu128〜cu132 | 570(世代の最低)/ 580(cu13x) | GPU |
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
- 本体の重み(`unet_dtype` model_management.py:1136〜1187 / `unet_manual_cast` 1189〜1211): 系統ごとの
  `supported_inference_dtypes`(下の表)の中から選ぶ。
  - fp16 を持つ系統(SDXL・Flux・Wan など)は、bf16 を持たない GPU でも fp16 で載せて fp16 で計算する(bf16 の重みも同じ 2 バイト)。
  - **fp16 を持たない系統**(`[bfloat16, float32]`: LTX・Qwen Image・Z-Image・HunyuanVideo・ACE-Step など)は、bf16 を持たない
    GPU(Pascal・Volta・Turing)では `unet_manual_cast` が float32 を返す(計算は fp32、作業用メモリは 2 倍)。重みは
    `should_use_bf16(manual_cast=True)` が「fp32 にすると `total × 0.88 − 最小の作業用メモリ` に収まらない」ときだけ bf16 のまま、
    収まるなら fp32(2 倍)で載せる。
  - 半精度を使わない GPU(Maxwell など)も、fp32 にしても収まるときだけ fp32 に広げ、収まらなければ 2 バイトのまま計算だけ fp32。
  - fp8 の重みは、fp8 の計算を持たない GPU では、fp16 に広げても収まれば広げ、収まらなければ fp8 のまま(計算のたびに変換)。
    fp8_scaled・int8_convrot など量子化の設定を持つ重みは量子化のまま(`quant_config`。sd.py:2268)。
- テキストエンコーダ: 既定は fp16。ファイルの精度が 2 バイト以下ならその精度のまま載せる(`pick_weight_dtype`)。
- VAE(`vae_dtype` model_management.py:1290): VAE ごとの `working_dtypes`(sd.py。既定は `[bfloat16, float32]`: sd.py:516)の
  中から、fp16 は `should_use_fp16`、bf16 は `should_use_bf16` のときに選び、どちらも無ければ fp32。2D の KL VAE
  (SD・SDXL・SD3・Flux の ae・Flux2 の VAE。sd.py:676〜705 の既定の枝)・LTX(780)・Mage(627)は fp16 を持たないので、
  bf16 を持たない GPU では fp32。Wan(849・857・873)・HunyuanVideo(791・824)・CogVideoX(808)などは fp16 を持つ。
  仕上げで VRAM が足りなければ ComfyUI はタイルに分けて続ける(遅い)。
- 作業用メモリ: ComfyUI 自身の見積もり(`model_base.memory_required`、VAE の `memory_used_decode`)が
  計算の精度の大きさ(`dtype_size`)を掛けている。fp32 で計算すると 2 倍。

見積もりへの効かせ方(`estimate.gpu_precision`。コードで確かなところだけ):

| GPU | 例 | Gpu.compute_dtype | 系統・VAE | 見積もりの直し |
|---|---|---|---|---|
| 半精度を使わない | Maxwell 全部、GT 1030、Linux の GTX 1080、GTX 1660、T1000 | `fp32` | どれでも | 作業用メモリ × 2、VAE の重み × 2(VAE のファイルが fp32、または VAE がどの GPU でも fp32 なら × 1) |
| fp16 は使うが bf16 を持たない | Windows の Pascal(1080 など)、Volta、Turing の RTX | `fp16` | fp16 を持たない系統 | 作業用メモリ × 2 |
| 同上 | 同上 | `fp16` | fp16 を持たない VAE | VAE の重み × 2 |
| 同上 | 同上 | `fp16` | fp16 を持つ系統 | 変えない(bf16 の重みには「fp16 で動かす(VRAM は同じ)」と書く) |
| 同上 | 同上 | `fp16` | 系統が分からない | 変えない。「系統が分からないので入れていない」と書く |
| fp8 の計算を持たない | Ada より前 | — | fp8 の重み | 変えない。「余裕があれば fp16 に広げる・無ければ遅い」と書く |
| 上のどれでもない | Ampere・Ada・Hopper・Blackwell | `fp16` | どれでも | 変えない |

**本体の重みは変えない**(fp32 にしても収まるときだけ fp32 で載せる、という動きは VRAM が多いほど使う量が増えるので、
見積もりに入れると VRAM を増やして段階が下がる。重みはファイルの 2 バイトで数え、ComfyUI が広げるのは収まるときだけなので、
動くかどうかの判定は変わらない。広げたときは作業用メモリの分が足りなくなり、ComfyUI が重みの一部を逃がして遅くなることがある。
そう理由に書く)。作業用メモリ × 2・VAE × 2 は GPU の世代と系統だけで決まり VRAM の量によらないので、世代を固定して
VRAM・RAM・ページファイルを増やしても段階は下がらない(`test_fit_monotonic_in_vram_per_arch`)。
fp32 で計算する場合は、その GPU の実測でない限り `basis` は `estimated` にする(実測は半精度の GPU のもの)。

### 系統の表(`src/genmodelfit/families.py`、ComfyUI コミット b1a247c の行)

本体(`comfy/supported_models.py` の `supported_inference_dtypes`。書いていないクラスは `supported_models_base.py:47` の
`[float16, bfloat16, float32]`):

| 系統(鍵) | ComfyUI のクラス | fp16 | 行 |
|---|---|---|---|
| sd / sd3 / hunyuan3d / stable_audio | SD15・SDXL・SVD・SD3・Hunyuan3Dv2・StableAudio など | ○ | base:47 |
| flux | Flux / FluxSchnell(Ovis も Flux として読む: model_detection.py:363) | ○ | 753 |
| flux_fill | FluxInpaint(in_channels 96 = Fill・OneReward) | × | 783 |
| flux2 / flux3 | Flux2 / Flux3 | ○ / × | 812 / 855 |
| chroma / longcat / krea2 / anima / cosmos | Chroma / LongCatImage / Krea2 / Anima / Cosmos | ○ | 1820 / 2433 / 2023 / 1171 / 1100 |
| lens / mochi | Lens / GenmoMochi | × | 882 / 924 |
| ltxv | LTXV / LTXAV(LTX-Video・LTX-2.x) | × | 952 |
| minimax_h3 | MiniMaxH3(FastVideo FastH3 も) | × | 1000 |
| hunyuan_video / hunyuan_image21 | HunyuanVideo / HunyuanImage21 | × | 1027 / 2203 |
| hunyuan_video15 | HunyuanVideo15 | ○ | 2241 |
| lumina2 / zimage / ming_image / pixeldit | Lumina2(NewBie: model_detection.py:611)/ ZImage / MingImage / PixelDiTT2I | × | 1213 / 1240 / 1240 / 1298 |
| wan | WAN21_* / WAN22_*(HuMo・VACE・S2V・Animate・SCAIL・CausalAR など) | ○ | 1366 |
| trellis2 | Trellis2(Pixal3D も: model_detection.py:125) | × | 1541 |
| triposplat / seedvr2 / sam3 / depth_anything3 / minimax_music3 / cogvideox | 同名 | ○ | 1685 / 1853 / 2512 / 2471 / 2400 / 2576 |
| hidream / hidream_o1 | HiDream / HiDreamO1 | × | 1710 / 1735 |
| ace / ace15 / yue2 | ACEStep / ACEStep15 / YuE2 | × | 1903 / 2350 / 2376 |
| omnigen2 | Omnigen2 / Boogu(torch 2.7 以上で float16 を先頭に足す: 1938) | ○ | 1930 |
| ideogram4 / moss_image / mage_flow | Ideogram4 / MOSSImage / MageFlow | × | 1994 / 2052 / 2085 |
| qwen_image / qwen_image21 / joyimage | QwenImage(FireRed・Marigold v2 も)/ QwenImage21 / JoyImage | × | 2114 / 2144 / 2176 |
| kandinsky5 / ernie | Kandinsky5 / ErnieImage | × | 2294 / 2495 |

Z-Image は重みの値で `allow_fp16` が立つと fp16 も使う(model_detection.py:624、supported_models.py:1245)が、
ファイルを読まないと分からないので fp16 を持たない側(安全側)で見て、理由に書く。

VAE(`comfy/sd.py` の `working_dtypes`): `kl`(既定 516。2D の KL)× / `ltxv`(780)× / `ltx_diffusion`(645)× /
`mage`(627)× / `wan`(849・857・873)○ / `hunyuan_video`(824)○ / `hunyuan_video15`(791)○ / `hunyuan3d`(902)○ /
`trellis2`(594・599)○ / `seedvr2`(610)○ / `cogvideox`(808)○ / `minimax_h3`(1049)○ / `ace`(917)○ /
`fp32`(540・947・999・1083。どの GPU でも fp32)。

一覧の各項目の系統は `families.family_of`(本体のファイル名 → ID・テンプレート名の順に照らす。部品の置き場所のリポジトリ名は
別のモデルの部品が混ざるので見ない)、VAE は VAE のファイル名(無ければ系統の既定)で決め、`update` が
`ModelEntry.comfy_family` / `comfy_vae` に入れる(`--heads-only` でも決め直す)。同梱の一覧(323 件)では 279 件に系統が付き、
44 件は分からない(読み上げ 20 件・ComfyUI に組み込みでない補助ツール 18 件・Capybara 4 件・HF の 3 件。ComfyUI のどのクラスで
読むかを確かめられなかったもの)。分からないものは見積もりを変えず、bf16 を持たない GPU でだけ理由に書く。

## NVIDIA 以外の GPU(0.7.0、#22・#23)

NVIDIA の GPU(nvidia-smi で確かめた `cuda`)が使えないときだけ、画面の部品の一覧から AMD・Intel・Apple の GPU を
下の対応表で判定に使う(`hardware.classify_adapter`)。**NVIDIA の判定は変えない**(NVIDIA の GPU が使えれば、ほかの
GPU は `--all-gpus` で一覧に出すだけ)。実測は 0 件なので、段階は推定で、理由に「推定(<会社>・実測なし)」を書き、
**段階は good まで**にする(同じ会社の GPU の、同じ設定で成功した実測があるときだけ外れる: 下の #26。2026-10-22 から、
この GPU と同じ GPU のほかの利用者の実測もその根拠になる: docs/accuracy.md「ほかの利用者の実測」)。
NVIDIA の GPU での実測は、作業用メモリをずらす補正にだけ使い(`basis_detail: calibrated`)、`basis` は `estimated`。
対応しない組み合わせは、理由(`Adapter.reason`)を付けて CPU として判定する。

### 対応表(一次資料は 2026-10-21 に読んだ)

| 会社・部品 | OS | 判定 | 一次資料 |
|---|---|---|---|
| AMD Radeon RDNA 4(RX 9070 / 9060、gfx1200・1201) | Windows 11・Linux | **GPU**(`rocm`) | ComfyUI README「AMD GPUs (Windows, ROCm 10.0)」(Comfy-Org/ComfyUI b1a247c: 「supported architectures include RDNA 2, RDNA 3, RDNA 3.5, and RDNA 4」「Use Windows 11」)、ROCm/TheRock `SUPPORTED_GPUS.md`・`RELEASES.md`(4639f91、「GFX target lookup table」・Release Ready) |
| AMD RDNA 3.5 の APU(Ryzen AI Max = gfx1151、Ryzen AI 9 HX 37x = gfx1150、Ryzen AI 7 350 = gfx1152、Radeon 820M = gfx1153) | Windows 11・Linux | **GPU**(`rocm`、共有メモリ) | 同上 |
| AMD Radeon RDNA 3(RX 7900 = gfx1100、7800 / 7700 = gfx1101、7600 = gfx1102)、Radeon 780M / 760M / 740M・Ryzen Z1 Extreme / Z1(gfx1103) | Windows 11・Linux | **GPU**(`rocm`) | 同上。Linux の pytorch.org の ROCm 版(ComfyUI の Linux の案内 `--index-url …/rocm7.2`)も入れる |
| AMD Radeon RDNA 2(RX 6900〜6800 = gfx1030、6700 = gfx1031、6600 = gfx1032、6500 / 6400 / 6300 = gfx1034)、Radeon 680M / 660M(gfx1035)、Van Gogh(Steam Deck、8 CU、gfx1033) | Windows 11・Linux | **GPU**(`rocm`。bf16 なし) | 同上。pytorch.org の ROCm 版は gfx1030 だけ(pytorch/pytorch v2.14.1 `.ci/docker/manywheel/build.sh` の `PYTORCH_ROCM_ARCH`)なので、ほかは AMD の multi-arch 版(TheRock)を使うと notes に書く |
| AMD Instinct(MI100〜MI355X、gfx908・90a・942・950) | Linux | **GPU**(`rocm`) | pytorch.org の ROCm 版の `PYTORCH_ROCM_ARCH`、TheRock の Instinct - Linux の表 |
| AMD の 2 CU の内蔵 GPU(Raphael・Granite Ridge = Ryzen 7000 / 9000 のデスクトップ、Dragon Range・Fire Range = 7x40HX / 7x45HX / 9x5xHX、Mendocino、610M = gfx1036) | — | CPU | 世代は対象だが、画面を出すための小さな GPU で、生成では CPU より速い見込みが無い(判断。実測なし) |
| AMD RDNA 1(RX 5700 / 5500)・GCN(Vega・Polaris 以前) | — | CPU | ComfyUI の案内が RDNA 2 以降(TheRock は RDNA 1 も Release Ready にしているが、ComfyUI に合わせた) |
| AMD(どれでも) | Windows 10・版が分からない | CPU | ComfyUI の案内「Use Windows 11」(分からなければ使わない側) |
| AMD | macOS | CPU | Apple の MPS 版は Apple silicon だけ(下) |
| Intel Arc A シリーズ(Alchemist: A770 / A750 / A580 / A380 / A310、Pro A) | Windows 11・Linux | **GPU**(`xpu`) | PyTorch 2.14「Getting Started on Intel GPU」(https://docs.pytorch.org/docs/2.14/notes/get_start_xpu.html)の Client GPUs の表(「Windows 11 & Ubuntu 24.04/26.04 & WSL2」)、ComfyUI README「Intel GPUs (Windows and Linux)」 |
| Intel Arc B シリーズ(Battlemage: B580 / B570、Pro B60 / B50) | Windows 11・Linux | **GPU**(`xpu`) | 同上 |
| Intel Core Ultra の Arc Graphics(Meteor Lake-H、Arrow Lake-H の 140T / 130T、Lunar Lake の 140V / 130V、Panther Lake の B390 など) | Windows 11・Linux | **GPU**(`xpu`、共有メモリ) | 同上(Panther Lake は Ubuntu 26.04) |
| Intel UHD・Iris Xe・「Intel(R) Graphics」(Arrow Lake-S など)・HD Graphics | — | CPU | 上の表に入らない |
| Intel(どれでも) | Windows 10・版が分からない | CPU | 上の表は Windows 11 |
| Apple silicon(M1〜M4 など) | macOS 14 以降 | **GPU**(`mps`、統合メモリ) | Apple「Accelerated PyTorch training on Mac」(https://developer.apple.com/metal/pytorch/ : 「Mac computer with Apple silicon」「macOS 14.0 or later」)、ComfyUI README「Apple Mac silicon」 |
| Apple silicon | macOS 13 以前・版が分からない | CPU | 同上 |

GPU の名前 → 世代は `hardware._AMD_NAMES`(製品名と lspci の開発名 Navi 31 など)・`_INTEL_XPU` の表。名前が
「AMD Radeon(TM) Graphics」だけの内蔵 GPU は CPU の名前(`_AMD_CPU_IGPU`: Ryzen AI Max → gfx1151、Ryzen 7 8845HS → gfx1103 など)で
決め、決まらなければ CPU(世代が分からない)。Dragon Range(Ryzen 7x45HX)・Fire Range(9x55HX)は 2 CU の
gfx1036 なので CPU。ノート用の Radeon は同じ番号でも石が違う(RX 6800M・6850M XT は gfx1031、RX 6800S は gfx1032、
RX 6550M は gfx1034、RX 7700S は gfx1102)。

0.9.0 で足した名前(#56。一次資料は 2026-10-22 に読んだ):

- **Ryzen Z1 Extreme・Z1**(携帯機。WMI の名前は「AMD Radeon(TM) Graphics」なので CPU の名前で決める): AMD の製品ページ
  (https://www.amd.com/en/products/processors/handhelds/ryzen-z-series/z1-series/z1-extreme.html ・
  https://www.amd.com/en/products/processors/handhelds/ryzen-z-series/z1-series/z1.html)は Graphics Core Count 12 / 4。
  Z1 Extreme は Phoenix1(780M と同じ 12 CU)、Z1 は Zen 4 2 個 + Zen 4c 4 個の Phoenix2(740M と同じ 4 CU)とみなし
  (石の対応は推定。AMD は書いていない)、どちらも gfx1103(Linux の amdkfd `kfd_device.c` が GC 11.0.1・11.0.4 を
  gfx_target_version 110003 にする)。740M(4 CU)を GPU にしているので Z1 も GPU。2 CU の 610M だけが CPU
- **Dragon Range の 7940HX・7840HX**(型番の下 2 桁が 40): AMD の製品ページ(例
  https://www.amd.com/en/products/processors/laptop/ryzen/7000-series/amd-ryzen-9-7940hx.html)で Radeon 610M・2 CU。
  以前は名前の表が 45HX だけで「gfx が分からない」理由になっていた。理由の文は「2 CU の小さな内蔵 GPU」にした
- **デスクトップの RX 6300**: pci.ids の「7424 Navi 24 [Radeon RX 6300]」で Navi 24(gfx1034)。TheRock の SUPPORTED_GPUS.md は
  gfx1034 を Release Ready にしている。2 GB の VRAM なので、動くモデルはほぼ無い(判定は VRAM の量に任せる)
- **Linux で pci.ids が古いとき**: lspci -nn は「Advanced Micro Devices, Inc. [AMD/ATI] Device [1002:7551]」と出す
  (pciutils `lib/names.c` の format_name_pair)。名前を「AMD Device 7551」として残し、`hardware.PCI_DEVICE_NAMES`
  (pci.ids 2026.10.22、pciutils/pciids 86e7077e4b79 から、判定に関わる AMD・Intel の ID だけを写した表)で石・製品の名前に
  引き直す。理由に「型番の ID から … とみなした(update-pciids で新しくすると製品名が出る)」と書く。表に無い ID は
  `update-pciids` の案内を付けて CPU(AMD・Intel とも)。型番の ID は製品の種類だけを表し、個体・差し場所は表さないので名前に残す
- **Linux の Intel の外付け GPU の VRAM**: xe・i915 のドライバは amdgpu の `mem_info_vram_total` に当たる sysfs を出さない
  (torvalds/linux の `drivers/gpu/drm/xe/xe_tile_sysfs.c`・`xe_device_sysfs.c`・`i915_sysfs.c`・Documentation/ABI を読んだ。
  量は DRM の ioctl か debugfs(root)でだけ読める)。そこで名前の表の**公称**の量を使い、理由に「名前から公称の … GB
  (量の違う版があれば少ないほう)」と書く(`Adapter.vram_source: "name"`)。ノート用の公称は Intel の表
  (https://www.intel.com/content/www/us/en/support/articles/000091109.html: A770M 16 GB・A730M 12 GB・A550M 8 GB・
  A370M 4 GB・A350M 4 GB)。A770 は 8 GB 版と 16 GB 版があるので 8 GB。A570M・A530M・Pro の M は Intel の資料で
  確かめられなかったので値を持たず、Linux では CPU のまま

### 精度(ComfyUI b1a247c の `comfy/model_management.py`)

- AMD(`torch.version.hip`): `should_use_fp16` は常に True。`should_use_bf16` は RDNA 2 以前(`AMD_RDNA2_AND_OLDER_ARCH`)で
  False だが、`manual_cast=True`(fp16 を持たない系統の計算の精度を選ぶとき)は True を返す。そこで RDNA 2 では fp16 を持たない
  系統でも作業用メモリを 2 倍にしない(VAE は fp32 で 2 倍)。
- Intel(XPU): `should_use_fp16` は `has_fp16`、`should_use_bf16` は `torch.xpu.is_bf16_supported()`。対象の Arc はどちらも持つ。
- Apple(MPS): `should_use_fp16` は True、`should_use_bf16` は macOS 14 以降で True。
- fp8 の計算(`supports_fp8_compute`)はどれも持たないとして扱う(fp8 の重みは注意だけ)。

### 共有メモリ(内蔵 GPU・APU・Apple silicon)

- **Apple silicon(`memory: unified`)**: ComfyUI は MPS では搭載量の全部を GPU のメモリとして数え
  (`get_total_memory` が CPU と MPS で `virtual_memory_total`、`vram_state` は SHARED: model_management.py)、部品をメインメモリへ
  逃がす(lowvram)動きは使わない(SHARED では `lowvram_model_memory` を求めない)。macOS は GPU が性能を落とさずに使える量
  (Metal の `recommendedMaxWorkingSetSize`。llmfit bffd1b6 が使う値)を搭載量より小さくするので、搭載量の **3/4**
  (`hardware.UNIFIED_MEMORY_SHARE`。判断で決めた値で、実測していない)を GPU のメモリとみなす。GPU に載せる量 + 固定分
  (Python・ComfyUI、3 GB)を搭載量とも比べる(超えたら仮想メモリで遅い = marginal)。
- **内蔵 GPU・APU(`memory: shared`)**: BIOS で取り分けた VRAM(Windows はレジストリの qwMemorySize、Linux は sysfs の
  `mem_info_vram_total`)+ メインメモリから使える分。Linux の amdgpu は sysfs の `mem_info_gtt_total`(GTT)を読む。読めなければ
  見えているメインメモリの **1/2**(`hardware.SHARED_MEMORY_SHARE`): Linux の TTM が GPU に貸すページの上限が「total system memory の
  約 50%」(torvalds/linux a804c57 `drivers/gpu/drm/ttm/ttm_device.c:88〜92`。amdgpu の `gttsize` の既定は「-1 = Use value specified
  by TTM」: https://docs.kernel.org/gpu/amdgpu/module-parameters.html)、Windows の共有 GPU メモリも物理メモリの約半分
  (タスク マネージャーの「共有 GPU メモリ」。WDDM の説明: https://learn.microsoft.com/ja-jp/archive/blogs/pigscanfly/graphics-memory-reporting-in-vista 。古い文書なので目安)。
- どちらも、逃がす先が同じメモリなので `GPU+RAM`(逃がす)は使わない(全部載せか、部品を順に載せ替えるか)。
- メインメモリの判定: GPU に載せる分もメインメモリから使うので、要る量 = GPU に載せないときの量(固定分 + いちばん大きい段の部品)
  + GPU に載せる量のうち BIOS で取り分けた VRAM(`Gpu.carveout_gb`。搭載量に入らない)を超える分。全部載せと順に載せ替えの
  それぞれで確かめ、良いほうを取る(例: Radeon 780M・メモリ 16 GB で、GPU 8.3 GB + メインメモリ 9.8 GB のモデルは足りない)。
  専用の VRAM の GPU があれば、内蔵 GPU より先に使う。
- メインメモリを増やすと GPU が使える量も増えるだけなので、段階は下がらない(`tests/test_vendors.py`)。

### 検出

- 既定では、判定に使える NVIDIA の GPU が無く(見つからない・Kepler・古いドライバ)、`--vram` で上書きしていないときだけ調べる
  (NVIDIA のドライバが入っていない NVIDIA の GPU も分かる)。NVIDIA の GPU があるときは調べない(PowerShell の起動に約 1 秒
  かかるため)。`genmodelfit system --all-gpus` で必ず調べる(このときも NVIDIA が使えれば NVIDIA で判定する)。
  レジストリには取り外した GPU の項目も残る(下の突き合わせで除く)。
- Windows: `powershell.exe -NoProfile -NonInteractive -Command` で `Get-CimInstance Win32_VideoController` の
  Name・AdapterRAM・PNPDeviceID・ConfigManagerErrorCode を JSON で取る(出力は UTF-8 に指定。時間制限 8 秒)。
  PNPDeviceID が `PCI\` で始まらないもの(仮想の画面・リモートデスクトップの画面)は捨てる。OS の版はビルド番号
  (22000 以上が Windows 11)。
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
- Linux: `lspci -Dnn` のクラス 03xx(VGA・3D・Display)を読む。名前は括弧の中の製品名(`Navi 31 [Radeon RX 7900 XT/7900 XTX/…]`
  の括弧の中)。AMD は `/sys/bus/pci/devices/<slot>/mem_info_vram_total`(VRAM)と `mem_info_gtt_total`(GTT)を読む。
  rocm-smi は使わない(rocm-smi の VRAM・GTT は同じ sysfs の値で、入っていない PC も多い)。Intel の外付け GPU は
  sysfs に VRAM の量が無いので、名前の表の公称の量を使う(上)。型番の分からない部品は「AMD Device 7551」の名前にする。
- macOS: `system_profiler SPDisplaysDataType -json` の `sppci_model`(「Apple M2 Pro」)と `spdisplays_vendor`。
  統合メモリの量は psutil の搭載量(macOS では `sysctl hw.memsize` と同じ値)、OS の版は `platform.mac_ver()`。
- PowerShell・lspci・system_profiler が無い・失敗した・時間切れは、黙って飛ばす(空のまま)。

### 表示

- 表の先頭の行: 使える GPU があれば「PC: AMD Radeon RX 7900 XTX 24.0 GiB VRAM (ROCm, estimated), RAM …」、Apple は
  「PC: Apple M2 Pro (unified memory; 24.0 GiB for the GPU, MPS, estimated), RAM …」、内蔵 GPU は「(integrated; 16.1 GiB shared
  with RAM, XPU, estimated)」。使える GPU が無ければ、見つけた GPU とその扱い(「PC: Intel(R) UHD Graphics 770 (integrated; not usable
  for generation → judged as CPU), RAM 15.7 GiB …」)と、次の行に「This PC has no usable GPU: only models that run on CPU are shown
  (N tools hidden: --include-tools)」。
- Mode の列: 動かないもののうち、使える GPU が無いから動かないものは `GPU req`、メインメモリが足りないものは `RAM`
  (JSON の `run_mode_label` は `-` のまま)。

### ほかのツールとの突き合わせ(2026-10-21)

表が厳しすぎ・緩すぎにならないかを、ほかのツールの対応表と比べた。基準は「ComfyUI + PyTorch で動くか」なので、
ComfyUI の README に合わせている。

- **Windows 10**: GPU を使えるとする資料は見つからなかった(AMD HIP SDK・ROCm on Radeon/Ryzen・PyTorch XPU は
  Windows 11 だけ。Ollama・LM Studio・llmfit は記載なし)。Windows 10 の版と分かれば、表の先頭の行に
  `needs Windows 11 for generation` と出す(`Adapter.needs`)
- **DirectML・ZLUDA・Vulkan は使わない**: ComfyUI は DirectML に「barely works, is very slow, has not been updated
  in over 1 year」と警告し(comfy/model_management.py)、README からも外した。Microsoft も DirectML を
  maintenance mode とし(https://github.com/microsoft/DirectML)、SD.Next は DirectML を削除、ZLUDA を
  「unofficial and currently limited」とした(https://vladmandic.github.io/sdnext-docs/ZLUDA/)。
  Fooocus や stable-diffusion-webui-amdgpu は DirectML で古い AMD を動かすが、遅い(Fooocus の README で約 3 倍)
- **AMD 公式の Windows 版 PyTorch の表より広い**: ROCm on Radeon/Ryzen の Windows の表は gfx1200/1201/1100/1101 と
  gfx1150/1151 だけ(https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)。
  ComfyUI の README は RDNA 2〜4 とし、TheRock(ComfyUI が使う ROCm)は RDNA 2・780M(gfx1103)も Windows で
  リリースの対象にしている(https://github.com/ROCm/TheRock/blob/main/SUPPORTED_GPUS.md)。RX 6000 系と 780M は
  ComfyUI に合わせて使うが、段階は good までの推定
- **RDNA 1・Vega・Polaris**: llmfit は名前と VRAM が取れれば GPU 扱い、Ollama・llama.cpp は Vulkan で動かせるが、
  ComfyUI の README の対象外なので CPU
- **2 CU の内蔵 GPU(gfx1036)**: TheRock は対象にしているが、生成で CPU より速い見込みが無いので CPU
  (対応の有無ではなく性能の判断。理由の文もそう書く)
- **Van Gogh(Steam Deck、gfx1033)**: 0.8.0 までは gfx1036 と一緒に「2 CU の小さな内蔵 GPU」として CPU にしていたが、
  Valve の仕様(https://www.steamdeck.com/en/tech 「GPU: 8 RDNA 2 CUs, 1.6GHz (1.6 TFlops FP32)」)で 8 CU あり、
  TheRock の SUPPORTED_GPUS.md(2026-10-22 に読んだ main)は gfx1033 を Linux・Windows とも Release Ready、
  ComfyUI の README は「supported architectures include RDNA 2」。680M(12 CU)・660M(6 CU)と同じく GPU(共有メモリ、
  段階は good まで)にした。pytorch.org の ROCm 版には入っていないので TheRock の版を使う、と notes に出る

## CPU で動く系統(0.7.0、#39)

GPU が無い(または GPU に載らない)とき、CPU で動くとみなすのは、これまでの「重み 2 GB 以下の画像・音声・読み上げ」に加えて、
`families.CPU_FAMILIES` の系統:

| 系統 | 種類 | 根拠 | 除くもの |
|---|---|---|---|
| `sd`(SD 1.5・SD 2.x・SDXL・SDXL Turbo など) | 画像 | ComfyUI の組み込みの系統で CUDA 専用の部品を使わない(推定。CPU での実測なし) | SUPIR(独自のノード。確かめていない) |
| `stable_audio`(Stable Audio Open 1.0・Stable Audio 3 Small) | 音楽・効果音 | 組み込みの StableAudio の系統(推定。CPU での実測なし。AI生成キットの記録に「効果音(GPU/CPU)」があるが、モデル・時間の記録は無い) | Stable Audio 3 Medium(確かめていない) |

ComfyUI は GPU が無ければ CPU で動き(`--cpu`「To use the CPU for everything (slow)」: comfy/cli_args.py)、CPU では本体と VAE を
fp32 で持つ(`should_use_fp16`・`should_use_bf16` が CPU では False: model_management.py:1902・1969。テキストエンコーダは fp16 のまま:
1261)。そこで必要なメインメモリ = 固定分 + 全部品 + (2 バイトの本体なら本体をもう 1 つ分)+ VAE をもう 1 つ分 + 作業用メモリ × 2。
これが搭載量に収まれば marginal(CPU、遅い)、仮想メモリの範囲なら marginal(遅い)、超えたら too_tight。
段階は CPU なので marginal まで。CPU で動くこと自体が推定なので、理由に「CPU は推定・実測なし」と書く
(AI生成キットのカタログの `cpu_ok` は設計の印で、実測ではない)。

## 同じ GPU で成功した実測(0.7.0、#26)

同じ設定・標準のワークフローで成功(`outcome: ok`)した実測があり、その GPU と同じ会社で、この PC の VRAM が実測の GPU の VRAM 以上
(公称 16 と nvidia-smi の 15.9 の違いを許して 97% 以上)で、実測のピークがこの PC の VRAM に収まるなら、段階は実測で決める(`Estimate.fit_from: measurement`): 実測の VRAM ÷
この PC の VRAM の帯(60 / 85 / 98%)で、少なくとも good。上限に張り付いた実測(動的 VRAM が空きを使い切った)と、
同じ構成で OOM の報告もあるほかの利用者の実測は perfect の根拠にしない(good まで)。`torch_reserved` の実測(bench。CUDA / HIP の
文脈を含まない)は文脈の分 0.5 GiB(`estimate.TORCH_RESERVED_CONTEXT_GB`)を足した値で帯を決め、perfect の根拠にもする(0.9.0。
以前は perfect にしなかったので、bench の提出しか無い NVIDIA 以外の GPU は実測があっても good 止まりだった)。会社は実測の
`gpu_backend` があればそれで、無いときだけ GPU 名で決める(Apple の MPS は GPU 名が "mps" になる)。ほかの利用者の
実測(#34)は、この PC と同じ GPU(型番 ID か、そろえた名前。汎用の名前は除く。`estimate.same_gpu`)のものだけ使う
(2026-10-22。似た GPU には広げない。0.9.0 から必要量にも同じ GPU のものだけ)。精度の直しで作業用メモリを倍にした
GPU では使わない。メインメモリの判定はそのまま行う。理由の先頭に「…で同じ設定の実測 … GB・成功があり … GPU で動くと判定した
(段階は実測で決めた。見積もりは目安)」と書き、表の VRAM はその実測の値(m)。実測の GPU がこの PC より大きいときは
「段階は見積もりで決めた」と書く。例: Wan 2.2 5B(RTX 5080 16 GB で 15.8 GB・成功)は、RTX 5080 では good / GPU
(0.6.0 は marginal / offload)、RTX 3060 12 GB では marginal / offload のまま。

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
| nvidia-smi の `addressing_mode`(ATS)で CPU と共有するメモリの GPU(Grace Blackwell・Tegra)を見分け、RAM を VRAM とみなす | VRAM が返らない NVIDIA の GPU は判定に使わない。DGX Spark などは対象外と書く |
| Apple silicon は Metal の `recommendedMaxWorkingSetSize` を GPU のメモリにする | Python から Metal を呼ばない(依存を増やさない)ので、搭載量の 3/4 とみなす(上の「共有メモリ」) |
| nvidia-smi が VRAM を返さないとき GPU 名から推す | 推さない(VRAM を推して「載る」と言わない) |
| 同じ型番の複数枚をまとめて VRAM を足す(llama.cpp などは分けて載せられる) | 足さない(ComfyUI は 1 枚で動く)。最大の 1 枚 |
| Linux で nvidia-smi が無いとき sysfs・lspci で NVIDIA の GPU を見つける | 判定には使わない(CUDA が動くか分からない)。`--all-gpus` などの一覧に出すだけ |
| AMD(rocm-smi・sysfs)・Intel(lspci)・Apple(system_profiler)を判定に使う | 0.7.0 から、ComfyUI + PyTorch が動く組み合わせ(対応表)だけ使う。llama.cpp の Vulkan のようにどの GPU でも動くわけではないので、Vulkan では数えない |
| WMI の AdapterRAM の頭打ちをレジストリの qwMemorySize で直す(issue #830)、DriverDesc で突き合わせ、曖昧なら使わない | 同じ考え方。加えて MatchingDeviceId(型番の ID)で先に突き合わせ、PCI でない部品を捨てる |
| 内蔵 GPU(Intel UHD など)を外付けがあれば落とす | 専用の VRAM の GPU を内蔵 GPU より先に使う。UHD・Iris Xe は対応表に無いので使わない |
| AMD の APU の UMA の取り分を搭載量に足し戻す | 取り分けた VRAM + GTT(または見えている RAM の半分)を GPU のメモリにする |
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

`tests/test_hardware.py`・`tests/test_vendors.py`(AMD・Intel・Apple・APU・CPU で動く系統・#26)と `tests/fixtures/hardware/`。
bench が NVIDIA 以外の GPU の使用率・使用中のメモリを読む道具(`gpustat.py`)は `tests/test_gpustat.py`(公式の文書・ソースの出力例と、この PC の typeperf の実際の出力)と `tests/test_bench_vendors.py`。既定の試験は本物の nvidia-smi・PowerShell・lspci・レジストリを呼ばない
(`tests/conftest.py` の `_no_real_hardware` が `hardware._run`・`_reg_display_entries`・`_reg_paging_files` を塞ぐ)。
フィクスチャ: この PC の nvidia-smi・WMI・レジストリの実際の出力(個体の ID は 0 に置き換えた)、複数の GPU、compute_cap の無い古い版、
`[N/A]`、壊れた出力、ドライバが動いていない・GPU が無いときの出力、AMD だけ・Intel + NVIDIA・ドライバの無い NVIDIA の WMI、
取り外した GPU が残るレジストリ、lspci。世代ごと(Kepler・Maxwell・Pascal・Volta・Turing・Ampere・Ada・Hopper・Blackwell)の判定と、
世代を固定して VRAM・RAM を増やしても段階が下がらないことを確かめる。
