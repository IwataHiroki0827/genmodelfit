# genmodelfit

[English](https://github.com/laxshmate/genmodelfit/blob/main/README.md) · **日本語**

画像・動画・音楽・効果音・読み上げ・3D の生成モデルのうち、どれがあなたの PC で快適に動くかを判定するツールです。

[![PyPI](https://img.shields.io/pypi/v/genmodelfit)](https://pypi.org/project/genmodelfit/)

LLM 向けに同じことをする [llmfit](https://github.com/AlexsJones/llmfit) に着想を得ています。状態: アルファ版
([CHANGELOG.md](https://github.com/laxshmate/genmodelfit/blob/main/CHANGELOG.md))。

![VRAM 8 GiB・メモリ 32 GiB の PC での uvx genmodelfit search flux と recommend](https://raw.githubusercontent.com/laxshmate/genmodelfit/main/docs/images/demo.ja.gif)

genmodelfit 0.12.0 で `uvx genmodelfit search flux --lang ja` と `uvx genmodelfit recommend --lang ja` を動かしたところ(2026-10-21 時点の一覧)。一覧は毎週更新するので、並ぶモデルと件数は変わります。
動画の GPU は VRAM 8 GiB の架空の Ada 世代(bf16・fp8 とも使える)に差し替えていて、記録した PC のものではありません。
`--vram` が変えるのは VRAM の量だけで、GPU の世代は動かした PC のままです。
RTX 30 系以降(30・40・50 系)なら世代で見積もりは変わらないので、GPU を買う前に試すのにも使えます(メインメモリは `--ram`)。

## すぐ試す

**入れずに試す。** [uv](https://docs.astral.sh/uv/getting-started/installation/) があれば、コマンド 1 つで動きます。
uv が専用の一時的な環境で genmodelfit を動かします(Python が無ければ uv が用意します)。お使いの Python や PATH には何も足しません。

```
uvx genmodelfit recommend         # この PC に合うモデルを種類ごとに出す
uvx genmodelfit system            # この PC で見つけたもの
```

この README のコマンドはすべて、`genmodelfit` を `uvx genmodelfit` に替えれば同じように動きます。
`uvx genmodelfit@latest …` とすると、uv が新しい版が出ていないかを先に確かめます。

**入れて使う。** `genmodelfit` コマンドをそのまま使えるようにします([詳しく](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#入れ方))。

```
pipx install genmodelfit          # または uv tool install genmodelfit / pip install genmodelfit
genmodelfit recommend             # この PC に合うモデルを種類ごとに出す
python -m genmodelfit recommend   # 「genmodelfit が見つからない」ときはこちら(中身は同じ)
```

**実測で協力する(任意。git は要りません)。** コマンド 1 つで、あなたの GPU を数分測り、結果を GitHub の PR で
送れます。同じ GPU を使う人みんなの見積もりの確かめと改善に役立ちます([やり方](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する)):

```
genmodelfit share                 # 入っているモデルを測り(5 分ほど)、送る中身を見せて PR を開く
```

## 1. 前提

### できること・しないこと

- 一覧にある各モデルが、この PC で動くか・どのくらい余裕があるかを**判定**します(VRAM・メモリ・GPU の世代から)。
- 種類ごと(image 画像・video 動画・music 音楽・sfx 効果音・tts 読み上げ・3d)に、良いものから**おすすめ**します。
- ライセンスを見せて確認したうえで、モデルのファイルを**ダウンロード**します。
- 生成は**しません**。モデルを動かすのは [ComfyUI](https://github.com/comfyanonymous/ComfyUI)(またはモデルが指定する実行環境)です。
  genmodelfit が持つのはモデルの**情報だけ**で、モデル本体は持ちません。
- llmfit を使わない理由: llmfit は LLM だけが対象です([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41))。
  生成モデルは精度・解像度・コマ数・メモリへの逃がし方で、必要なメモリが数倍変わります。

### 動く環境

- Windows・Linux・macOS、Python 3.10 以上(`uvx` で使うときは、Python が無くても uv が用意します)。
- GPU: NVIDIA(CUDA)は、NVIDIA の実測で較正した推定で判定します。使える NVIDIA の GPU が無いときは、AMD(ROCm: RDNA 2 以降、
  Windows 11・Linux)・Intel(XPU: Arc A / B シリーズ、Core Ultra の Arc Graphics、Windows 11・Linux)・Apple silicon(MPS、
  macOS 14 以降)を各社の対応表で判定します(段階は NVIDIA と同じ基準。表の PC の行の「✓ 確認済みの種類」は、2 機種以上で動いた記録がある種類の GPU だけに付きます。今は NVIDIA の専用 VRAM の GPU)。内蔵 GPU・APU
  (Radeon 780M / 680M・Steam Deck・Ryzen Z1 など)は共有メモリで判定します。それ以外の GPU(Intel UHD・Iris Xe、2 CU の
  Radeon 610M / 820M、RDNA 1 以前)と GPU の無い PC は CPU だけとして判定します。SD 系の画像モデルと Stable Audio は CPU でも動く(遅い)と
  判定します。Linux で lspci が GPU の名前を知らないときは、`system` が `update-pciids` を案内します
  ([docs/hardware.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.ja.md))。
  NVIDIA 以外の GPU は実測が無いので、✓ が付きません。`genmodelfit measure` はこれらの GPU でも測れます。
  `genmodelfit share` で送ってもらえると、確認済みになり見積もりも詰められます
  ([実測で協力する](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する))。

### 入れ方

試すだけなら入れなくても使えます: `uvx genmodelfit …`([すぐ試す](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#すぐ試す))。
`genmodelfit` コマンドを手元に置くときは:

```
pipx install genmodelfit        # 1 回だけ pipx ensurepath を実行し、端末を開き直す
uv tool install genmodelfit     # 1 回だけ uv tool update-shell を実行し、端末を開き直す
pip install genmodelfit
```

pipx と uv は `genmodelfit` コマンドを PATH に通してくれます。**`python -m genmodelfit …` は PATH に関係なく必ず動きます**
(Windows では `py -m genmodelfit …` も可)。

pip が *「WARNING: The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH」* と出したら
(python.org の Python を「Add python.exe to PATH」なしで入れた・Microsoft Store の Python・利用者ごとの場所に入った・
仮想環境を有効にしていない、のどれかが多い)、その場所を PATH に足すか、上のどれかを使ってください。
PATH を変える前に開いた端末には反映されないので、新しい端末を開いてください。

### 通信するもの

判定には通信しません。PC の情報は手元で読みます。通信するのは `update`・`download`・`measure`・`share`・`bench` だけで、
相手は公開の情報か、指定した ComfyUI だけです(フォルダや URL を指定しないときは、`download`・`measure`・`share` が
この PC(127.0.0.1)で起動中の ComfyUI にも尋ねます。`measure` と `share` は、そこで ComfyUI を起動することもあります)。`share` は測る前にもうあるトークンを GitHub で確かめるだけで、
同意するまで何も送りません(同意すると、ログインして見せたファイルの PR を開くか、Issue のフォームを開く)。
利用状況の送信(テレメトリ)もありません。詳しくは[送るもの・送らないもの](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#送るもの送らないもの)。

## 2. 手順

1. **PC を確かめる。** GPU・VRAM・メモリ・ページファイル・ディスクの空きと、モデルの一覧が新しいかを出します。
   ```
   genmodelfit system
   ```

2. **おすすめを見る。** この PC で動くモデルを、種類ごとに良い順で出します。
   種類(`-m`)や、種類の中の用途(`--task`。一覧は `genmodelfit types`)で絞れます。
   ```
   genmodelfit recommend
   genmodelfit recommend -m image --task edit
   ```

3. **動くものを全部見る。** この PC で動くモデル(Perfect / Good / Marginal)を 1 つの表で出します。
   `--all` で補助ツール・動かないもの・判定できないものも、`--sort score` で種類をまたいだ順位、`--fit good` で 1 つの段階だけ。
   ```
   genmodelfit fit
   genmodelfit fit --sort score --fit good
   ```

4. **1 つのモデルを詳しく見る。** ファイル・ライセンス・動かし方ごとのメモリ・限界と、Verified になる/ならない理由を出します。
   ```
   genmodelfit info "Z-Image-Turbo"
   ```

5. **ダウンロードする。** `--list` は、落とさずにファイル・ライセンス・大きさ・空きだけを見せます。
   そのあと ComfyUI のフォルダへ落とします。始める前に確認があり、どのファイルも sha256 で確かめます。
   `--comfyui-dir` が無ければ ComfyUI を探し、見つけた場所を出します(見つからない・2 つ以上なら尋ねます)。
   ```
   genmodelfit download "Z-Image-Turbo" --list
   genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI   # ComfyUI を入れたフォルダ
   ```

6. **一覧を新しく保つ。** 一覧は毎週作り直しています。リンクの確認から 30 日を過ぎたモデルは Verified でなくなります。
   このコマンドで、このプロジェクトが署名して配る最新の一覧を取り込みます。
   ```
   genmodelfit update --from-release
   ```

7. **測って送る(任意)。** `share` が、測ると役に立つモデルのうち入っているものを測り(5 分ほど。要れば ComfyUI を
   起動する)、送る中身をそのまま見せて、同意すると GitHub の PR で送ります([実測で協力する](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する))。
   `measure` は選んだ 1 つを測ります。
   ```
   genmodelfit share
   ```

### アプリから使う(Python / JSON)

各コマンドは `--json` を付けると JSON で出します([docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md))。
Python では次のとおりです。

```python
import genmodelfit as gmf

for f in gmf.recommend(modality="image", task="edit", limit=5):
    row = f.to_dict()  # `recommend --json` の 1 件と同じ dict
    # memory_required_gb: 選んだ動かし方で要る量(表の VRAM。CPU ならメモリ)。
    # f.estimate.vram_gb は全部を GPU に載せたときの量で、GPU より大きいことがある(GPU~・GPU+RAM の行)
    print(f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, row["memory_required_gb"])

st = gmf.data_status()  # 一覧は Verified(auto)を出せるほど新しいか
if st.expired:
    print("run: genmodelfit update --from-release", st.expires_at)

# 一覧を自分で読むときは、確認待ちのものを除く。
# 形や型の正しくない一覧では、load_db が RuntimeError を出す(RuntimeError で受ける。派生の
# genmodelfit.db.DatabaseError は gmf.__all__ に無い)
ok = [e for e in gmf.load_db().models if e.reviewed]

# アプリが調べた GPU で 1 モデルを判定する(世代と精度は genmodelfit が決める)
fit = gmf.judge_model("image_sdxl_simple", gpu_name="NVIDIA GeForce RTX 3060", vram_gb=12, driver="560.94")
print(fit.to_dict()["fit_level"])  # 互換を守るのは gmf.__all__ の名前だけ

# AMD・Intel の GPU も検出と同じ対応表で判定する。使えない理由は other_gpus[].reason
pc = gmf.system_from(
    [{"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vram_gb": 16}],
    windows=True,
    windows_build=22631,
    cpu="AMD Ryzen 7 7700X",
)
gmf.gpu_family("Intel(R) Arc(TM) B580 Graphics", "intel")  # 名前の表から {"backend": "xpu", ...}
db = gmf.load_db(use_cache=False, user_measurements=False)  # 同梱の一覧と同梱の実測だけ

# 署名つきの一覧を取り込む(通信するのはこれと gmf.download だけ)。自分のセッション(アプリの CA バンドルなど)とデータフォルダで
db = gmf.update_from_release(session=my_session, dest=data_dir / "db.json", state_dir=data_dir)
```

互換を守るのは `gmf.__all__` の名前だけです([docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#python-api-stable--公開の範囲))。

### ほかの PC で判定する

`--vram` と `--ram`(GiB)で、検出した大きさを置き換えます。例: `genmodelfit recommend --vram 12 --ram 32`。
使える NVIDIA の GPU があれば、その世代のまま VRAM だけを変えます。NVIDIA 以外の GPU(または使える GPU が無い PC)では、`--vram` は世代の分からない仮の NVIDIA の GPU を判定します。
ほかの利用者の実測と、段階を決める実測は、`--vram` と実測した GPU の VRAM の比(小さい ÷ 大きい)が 0.95 以上のときだけ使います。
`--width`・`--height`・`--frames`・`--length`・`--precision` で設定を決めて判定できます。

### 表示の言語

出力の既定は英語。日本語にするには、どのコマンドでも `--lang ja` を付ける(例 `genmodelfit recommend --lang ja`)か、
環境変数 `GENMODELFIT_LANG=ja` を置くか、`genmodelfit config set lang ja` で保存する(コマンドラインだけ)か、
Python で `gmf.set_language("ja")` を呼ぶ(優先は `--lang`、環境変数、コマンドラインは保存した設定・Python は
`set_language` の順。OS の言語は見ない)。変わるのは文(理由・注意・エラー・表の見出しと説明・`--help`)だけで、
JSON のキー・ID・コード・判定はどちらの言語でも同じ。英語の出力には日本語が出ず、日本語の出力には英語だけの文が出ない
(1 行に 2 つの言語の文を並べない)。

どちらの言語でも英語のまま出すもの: 表の列の名前(`Type`・`Status`・`Score`・`Model`・`Use`・`Prec`・`VRAM`・`Mem%`・
`Mode`・`Disk`・`License`・`Censored`)、段階の名前(`Perfect`・`Good`・`Marginal`・`Too tight`)、動かし方の印
(`GPU`・`GPU~`・`GPU+RAM`・`CPU`)、単位、ID・ライセンスの ID・オプションの名前など機械が読む値。表の幅をそろえ、
`--fit`・`--sort` などに書く名前と同じにするためです。日本語の出力では、表の下の説明がこれらを日本語で説明します。
種類の見出しは `== 画像 (image): …` のように、`-m` に書く名前も出します。

長い行は全角を 2 桁で数えて折り返します。ファイル名・URL・オプション・単位つきの数の途中では切らず(1 つで幅より
長ければ、その行だけはみ出す)、行の頭に「、。)」を、行の終わりに「(「」を置きません。

## 3. 結果

### 例

VRAM 8 GiB・メモリ 32 GiB の PC での `genmodelfit recommend --lang ja`(種類ごとに上位 2 件。`COLUMNS=100`)。
例は架空の Ada 世代の GPU で作っています。`--vram` が変えるのは VRAM の量だけで、GPU の世代は動かした PC のものです。
[tools/screenshots.py](https://github.com/laxshmate/genmodelfit/blob/main/tools/screenshots.py) の `--readme` でこれらのブロックを作り直せます:

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2 --lang ja
PC: VRAM 8.0 GiB と仮定(--vram)、メモリ 32.0 GiB(--ram)
おすすめ: 83 件中 12 件(隠した 221 件: 無検閲 26・補助ツール 34・ライセンス不明 28・動かない 133。ほ
    かに 25 件のワークフローをモデルにまとめた)
== 画像 (image): 32 件中 上位 2 件(続きは -m image)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Good        100 Anima Base 1.0             gen,edit  bf16   5.7  71% GPU~     5.4 other
image ● Good         92 SDXL1.0 Base               gen       fp16   6.2  78% GPU~     6.5 openrail++
== 動画 (video): 12 件中 上位 2 件(続きは -m video)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
video ● Good         96 Wan 2.2 5B TI2V            gen       fp16   6.1  76% GPU+RAM 16.9 apache-2.0
video ● Good        100 Wan 2.1 1.3B T2V           gen       fp16   4.3  54% GPU+RAM  9.2 apache-2.0
== 音楽 (music): 10 件中 上位 2 件(続きは -m music)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
music ● Good         96 ACE-Step 1.5 Turbo         gen       bf16   6.0  76% GPU~     9.3 mit
music ● Good        100 YuE2 3B                    gen,edit  int8   4.3  54% GPU~     3.7 cc-by-nc-…
== 効果音 (sfx): 2 件中 上位 2 件(続きは -m sfx)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
sfx   ● Good        100 stable-audio-open-1.0      gen       fp32   4.9  61% GPU~     5.3 other
sfx   ● Perfect     100 stable-audio-3-small-sfx   gen       fp32   4.7  59% GPU      3.2 other
== 読み上げ (tts): 20 件中 上位 2 件(続きは -m tts)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
tts   ● Perfect     100 MeloTTS-English            gen       ?      1.7  21% GPU      0.2 mit
tts   ● Perfect     100 Chatterbox                 gen       fp32   3.5  44% GPU      2.0 mit
== 3D (3d): 7 件中 上位 2 件(続きは -m 3d)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
3d    ● Good         99 TripoSplat                 gen       fp16   5.8  72% GPU      3.5 other
3d    ● Good        100 HY 3D 2.0 MV               gen       fp16   4.3  54% GPU~     4.6 other
並び: 動く → Perfect/Good(GPU+RAM・CPU は後)→ Marginal(速い Mode が先)→ 公式 → 人気 → Score。
Score 0〜100 = メモリの余裕(Mem% 70% 以下で 100。速さではない)。VRAM = 要る GiB(m = 実測)。
Mode: GPU 全部 GPU、GPU~ 部品を順に載せ替え、GPU+RAM メモリへ逃がす(遅い)、CPU(遅い)、- 動かない。
段階の上限: GPU~ と GPU+RAM は Good まで、CPU は Marginal まで(CPU 向けの読み上げ・道具は Good)。
Mem% = 一番きついメモリの使用率(VRAM。CPU はメモリ、GPU+RAM は大きいほう)、RAM = メモリ不足。
Perfect は余裕があるという意味で、速さの保証ではない(帯域の狭い GPU でも使用率で決める)。
GPU+RAM の段階はメモリの量で決まることが多い。逃がす量は決めの値(重みの 4 割を GPU に残す)。
1 行に 1 モデル: Prec = 載る中で一番高い精度、Use = ワークフローの用途(info、--all-workflows)。
m の無い見積もりは目安。詳しくは genmodelfit info <名前か ID>。アプリは --json(schema.md)。
--json は自動でおすすめしてよいものを先に並べる(schema.md)。
注意: 無検閲のモデル 26 件を隠しました。出すなら --uncensored。
注意: ライセンスの分からないモデル 28 件を隠しました。出すなら --any-license。
? = 分からない(精度・ライセンス・大きさ。判定していない)、- = 当てはまらない(動かないなど)。
```

最初の行は PC と、隠したモデルの数と理由です。最後の行は印の説明です。
`fit` は 1 つの表で出します(VRAM 24 GiB・メモリ 64 GiB、画像だけ、先頭の 6 行。`--lang ja`):

```
$ genmodelfit fit --vram 24 --ram 64 -m image -n 6 --lang ja
PC: VRAM 24.0 GiB と仮定(--vram)、メモリ 64.0 GiB(--ram)
全モデル: 60 件中 6 件(隠した 218 件: ほかの種類 166・無検閲 21・補助ツール 19・ライセンス不明 12。
    ほかに 51 件のワークフローをモデルにまとめた)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Perfect     100 Anima Base 1.0             gen,edit  bf16   7.0  29% GPU      5.4 other
image ● Perfect     100 SDXL1.0 Base               gen       fp16   8.0  33% GPU      6.5 openrail++
image ● Perfect     100 Anima Preview 3 Base       gen       bf16   7.0  29% GPU      5.2 other
image ● Perfect     100 NetaYume Lumina            gen       bf16  12.4  52% GPU      9.9 gemma
image ● Perfect     100 SDXL Turbo 1.0             gen       fp16   9.0  38% GPU      6.5 other
image ● Perfect     100 PixelDiT 1300M             gen       bf16   9.6  40% GPU      7.3 other
並び: Type → Status → 公式 → 人気 → Score(--sort)。
Score 0〜100 = メモリの余裕(Mem% 70% 以下で 100。速さではない)。VRAM = 要る GiB(m = 実測)。
…
```

### 列

| 列 | 意味 |
|---|---|
| Type | image・video・music・sfx・tts・3d |
| Status | 段階(下の表) |
| Score | 0〜100。メモリの余裕。その動かし方で一番きついメモリの使用率(Mem%)が 70% 以下なら 100、超えると下がる。速さ・品質・人気は入れない([決まり](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.ja.md#score)) |
| Censored | `Y` 検閲あり / `N` 無検閲。`--uncensored`(か `--content all|uncensored`)のときだけ出る |
| Model | 同じ重みを 1 行にまとめたモデル。長い名前は真ん中を `…` で詰める |
| Use | `gen`・`edit`・`i2v` など(`genmodelfit types`)。行自身の用途(`--task` で合ったもの)を先頭に、そのモデルのほかのワークフローの用途を重複なく並べる(`gen,edit`。収まらなければ `gen,…`)。ワークフローの一覧は `info`、別々の行は `--all-workflows` |
| Prec | 載る中で一番良い精度(fp16・bf16・fp8・int8 など) |
| VRAM | その動かし方で要る GiB。`m` = 実測(作者・この PC の実測、または同じ GPU のほかの利用者の実測の中央値。出どころの数は `info`・JSON の `sample_count`) |
| Mem% | 一番きついメモリの使用率。GPU なら VRAM、CPU ならメモリ、`GPU+RAM` は大きいほう。`RAM` = PC のメモリが足りない |
| Mode | `GPU` 全部 GPU、`GPU~` 部品を順に載せ替え、`GPU+RAM` メモリへ逃がす(遅い)、`CPU`(遅い)、`-` 動かない(`GPU req` は使える GPU が無いので動かない、`RAM` はメモリ不足) |
| Disk | ダウンロードの大きさ(GiB)。幅が足りないと最初に外す(そのときは `info` か `--json` で見る) |
| License | ファイルのライセンス(`other` = 自分で原文を読む) |

### ライセンス不明・無検閲のモデルは、既定で出しません

ライセンスの分からないモデルと、無検閲のモデルは、既定では一覧に出しません。何も付けずに使えば、こうしたモデルを
知らずに選ばずに済みます。隠した件数は、結果の最後に「注意」として出ます。
出すときは、次のオプションを付けます。

- `--any-license`: ライセンスの分からないモデルも出す
- `--uncensored`: 無検閲のモデルも出す

### 段階

llmfit と同じく、動かし方で使うメモリ(GPU なら VRAM、CPU ならメモリ)の使用率で決めます。

| 段階 | 使用率 | 動かし方による上限 |
|---|---|---|
| **Perfect** | 60% 以下 | 全部 GPU に載るとき(`GPU`)だけ。余裕があるという意味で、速さの保証ではない |
| **Good** | 85% 以下 | `GPU~`・`GPU+RAM` はここまで(逃がすのは段階が上がるときだけ)。`CPU` で実時間に動くと作者が書いている読み上げ(MeloTTS・Piper・VieNeu-TTS)と 1 回の計算の道具もここまで |
| **Marginal** | 98% 以下 | `CPU` の生成・メモリが少し足りないときはここまで(Mem% が低くても) |
| **Too tight** | 98% 超 | 動かない(実測で段階を決めた行は、98% を超えても Good になる) |
| **?** | — | 判定していない(ダウンロードの大きさが分からない) |

GPU の会社による上限はありません。その種類の GPU で 2 機種以上で動いた記録があれば、PC の行に「✓ 確認済みの種類」と出ます([決まり](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.ja.md#確認済みの-gpu-の種類))。
印: VRAM の後ろの `m` = 実測(誰が測ったかは分けない)。印が無ければ式による推定です。
実測は式より優先します。`--width` などを指定しなければ、どのモデルも種類ごとに同じ既定の設定で判定するので、設定の違う実測は
見積もりをずらすだけです(印は付きません)。ほかの利用者の実測は同じ GPU(PCI の型番 ID、無ければ正規化した名前が同じで、VRAM の
比が 0.95 以上)にだけ使い、必要量を上下どちらにも動かします。段階を実測で決めるのは、同じ GPU・同じ設定で別々の 2 つ以上の
出どころがそろったときだけで、そのとき表の VRAM は段階に使った量です(重み + 固定分より小さくしない。`bench` の値
(`torch_reserved`)は 0.5 GiB を足す)。決まり: [docs/scoring.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.ja.md)。
`?` = 分からない(精度・ライセンス・大きさ。判定に使っていない)。`-` = 当てはまらない(動かない・CPU なので VRAM が無い など)。
`recommend` の並びは「動く → Perfect/Good(逃がさないものが先)→ Marginal(速い動かし方が先)→ 公式のテンプレート → 人気(同じ種類・同じ出どころの
中の使用数の順位)→ Score」、`fit` は「種類 → 段階 → 公式 → 人気 → Score」です。
(`recommend --json` と `gmf.recommend()` は「動く」の次に Verified のモデルを先にします: [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#order-of-recommend--recommend-の並べ方)。)

### 精度

0.13.0 の時点では、見積もりは 1 種類の NVIDIA の GPU の少数の実測で較正し、別の GPU の 2 件の実測で照合しています。
送られた実測のまとめ(`measurements_community.json`)は、1 つの出どころ(メンテナ。RTX 3070 の SDXL の 4 つの設定)の提出 4 件で、そろった(consistent)行はまだありません。`m` の無い値は目安として見てください。
ComfyUI は空きがあるほど VRAM を多く使うので、大きい GPU で測ったピークは最低限の必要量ではありません。
今の件数と詳しい誤差は [docs/accuracy.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.ja.md)。

## 4. 参考

文書の一覧と、それぞれ誰向けか: [docs/README.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/README.ja.md)。

### コマンド

| コマンド | 内容 |
|---|---|
| `system [--all-gpus]` | この PC と、一覧の新しさ |
| `recommend [-m TYPE] [--task USE] [-n N] [--auto-only]` | 種類ごとのおすすめ |
| `fit [-m TYPE] [--fit LEVEL] [--sort KEY] [--all]` | 動くモデルを全部、1 つの表で |
| `search <text>` | ID・名前に文字を含むモデルを、この PC 向けに判定して出す |
| `types` | 6 つの種類と、その中の用途(`--task` の名前)と件数 |
| `info <名前か ID>` | ファイル・ライセンス・見積もり・限界・Verified の条件 |
| `download <名前か ID> [--list] [--comfyui-dir DIR]` | ファイルを落とす([ダウンロード](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#ダウンロード)) |
| `update [--from-release]` | 一覧を上流から作り直す、または署名付きの一覧を取り込む |
| `gate` | Verified のモデルの数と、ほかが保留の理由 |
| `diff [OLD] [NEW] [--all]` | 2 つの一覧の違い([下](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#一覧の変化)) |
| `config [set KEY VALUE \| unset KEY \| path]` | 設定: `download` の ComfyUI のフォルダと表示の言語([docs/download.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.ja.md#genmodelfit-config)) |
| `measure [名前か ID] [--url URL]` | ComfyUI でモデルを実測する(起動中のもの、無ければ見つけたものを起動して、終わったら止める)。名前を省くと、測ると役に立つモデルを出す([下](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する)) |
| `share [ファイル]` | 数分測って、結果を GitHub の PR で送る(`--issue` は Issue のフォーム)([下](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する)) |

保守する人向け: `review`(確認待ちのモデル)・`bench`(1 つのテンプレートをすべてのオプションで測る)・`contrib`
(提出を作る・検査する・まとめる。`contrib share` は `share --issue` と同じ)。

すべてのコマンドとオプション・環境変数・終了コード・例は [docs/commands.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/commands.ja.md) にまとめてあります(各コマンドの `--help` と同じ内容)。よく使うオプション: `--json`・`--vram`・`--ram`・
`--uncensored`(無検閲も含める。前からの書き方は `--content all`)・`--any-license`(ライセンス不明も含める。前からの書き方は `--include-unknown-license`)・`--content uncensored`(無検閲だけ)・`--license apache-2.0,mit`・`--comfyui-version`。

### データ源

一覧は [ComfyUI のワークフローテンプレート](https://github.com/Comfy-Org/workflow_templates)(MIT)と
[Hugging Face Hub の API](https://huggingface.co/docs/hub/api)(上流までたどったライセンス・大きさ・sha256)から作り、
少しだけ手で足したモデルがあります。llmfit と同じく、Hugging Face のモデルは全期間のダウンロード数 10,000 以上が条件です(はっきり伸びているものは 1,000 以上)。
Civitai は使いません(規約が自動アクセスを制限しているため)。詳しくは [docs/sources.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.ja.md)。

### 内容の印

安全フィルタを外したモデルや成人向けの出力を想定したモデルは `uncensored` の印を付け、**既定では出しません**。
`--uncensored`(または `--content all|uncensored`)で `Censored` の列と一緒に出します。既定では Verified になりません。
未成年の性的表現を目的とするモデルは一覧から除外しています
([docs/sources.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.ja.md))。

### Verified(自動でおすすめに出す条件)

`Verified: yes`(JSON の `auto`)は、次を**すべて**満たすときだけです: すべてのファイルのライセンスが商用利用できる許可リストにある、
gated が無い、カスタムノードの無い公式の ComfyUI ワークフロー、対応する ComfyUI の版、すべての URL が 30 日以内の匿名の HEAD に答えた、
sha256 と大きさが分かる、内容が `sfw`、確認待ちでない。満たさない理由は `held_reasons` に入ります。
`recommend` / `fit` の表には出しません。条件ごとの結果は `info <名前か ID>`、数は `gate` で見られます。
詳しくは [docs/gate.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.ja.md)。

### ダウンロード

一覧にある https の URL だけを使い、転送先は Hugging Face・GitHub のホストだけ、書きながら sha256 を確かめ、Ctrl-C のあとは続きから、
置き場所の外には書きません。gated は落としません。無検閲のモデルとライセンス不明のファイルは、同意があるときだけ落とします。
端末では、それを名指しした同意の質問(y/N。既定は No)で尋ねます。`--yes`・`--json`・スクリプトでは `--uncensored`・
`--any-license` が要ります(足したコマンドを 1 行と、それぞれの意味を出します。勝手に足して落とすことはしません)。テンプレートが既定で使わない部品(無効・素通しのノードにしかないもの)は
`--with-optional` を付けたときだけ落とします。スイッチで切ってある部品は、ComfyUI が動かす前に確かめるので落とします。ComfyUI のどのフォルダか分からないファイルは `other/` に置きます。ComfyUI は読まないので、
計画の中でそう知らせます(使う前に移す)。置き場所は `--comfyui-dir`(か `--output-dir`)、環境変数 `GENMODELFIT_COMFYUI_DIR`、
`genmodelfit config set comfyui-dir <フォルダ>`、この PC で見つけた ComfyUI(今いるフォルダ・起動中の ComfyUI・デスクトップ版と
comfy-cli の設定・よくある場所)の順で決め、どれで決めたかを計画に必ず出します。見つからない・2 つ以上見つかったときは、
端末なら選び、スクリプトでは ComfyUI が読まないフォルダに落とさずに誤りで止めます。質問は `[y/N/c]`(`c` で置き場所を変える)。
詳しい決まりと Python の関数は [docs/download.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.ja.md)。

### 一覧の保守

一覧は毎週作り直します。新しいモデル、ファイルが変わったモデル(URL・大きさ・sha256)、危うい変化には `pending` の印を付け、人が一次資料の根拠を付けて承認する
(`genmodelfit review`)まで Verified にしません。署名して `update --from-release` に配るのは確認済みのデータだけです。
詳しくは [docs/release.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.ja.md)。

### 一覧の変化

`genmodelfit diff` は、同梱の一覧と `update --from-release` で取ってきた一覧を比べます(先に取ってきてください。
無いときは手順だけを出します)。`diff NEW.json` は同梱の一覧とそのファイル、`diff OLD.json NEW.json` は 2 つの
ファイルを比べます。モデルは「名前 (種類) — ID」で出し、増えた・消えたモデル、自動のおすすめの変化、変わったモデルごとの
ライセンス(`!` = 寛容な側)・大きさ・精度・言語(指示・読み上げ)・増えた・消えた部品とそのライセンス・新しく無検閲・
確認の印、ほかに変わった項目の名前(「その他の変化」)を、3 つの節(部品・ライセンス・内容・動かし方 → 自動のおすすめ(ID 付き) → ライセンスの出どころ・言語・大きさなど)に分け、大事な変化から順に出します。多くのモデルで同じ小さな変化は件数の 1 行にまとめます。人気の数・日付・HEAD の結果だけの変化、前からの
HEAD の失敗、`sources` の数は件数だけで、`--all` で全部を出します(毎週の PR の本文と同じ中身)。
`--json` の形は [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#diff)。

### 実測で協力する

いろいろな GPU の実測が集まるほど、見積もりが良くなります。コマンド 1 つで、git は要りません
(NVIDIA・AMD・Intel・Apple の GPU。GitHub のアカウントが要ります):

```
genmodelfit share
```

1. `measure` と同じく ComfyUI を見つけます: この PC で起動中のもの(か `--url`)、無ければ見つけた ComfyUI
   (ポータブル版・`.venv` のある手で入れたもの / comfy-cli・ComfyUI デスクトップ版)を裏で起動し、終わったら止めます
   (`--no-start` で起動しない)。
2. `measure` の提案のうち、ファイルが入っていて、この PC でまだ測っていないもの(この PC に保存したものは、送ったか
   どうかによらず数える。同じ GPU のほかの人の実測は数えない。2 つの出どころの実測がそろうと、段階を実測で決められる
   ため)を、時間の目安の合計が 5 分ほど(`--budget`)になるまで選び、計画・まだ送っていない実測・合計の時間(ComfyUI の起動・
   測る時間・落とす時間を 100 Mbps の例で)を見せて 1 回だけ尋ねます。ComfyUI に無いノードが要るものは外し
   (ComfyUI を更新すると測れます)、次の提案で埋めます。
3. 1 つずつ測り、測る前には毎回 ComfyUI を空けます(モデルを外し、前の結果を消す)。入っているものが無いか、
   時間が残れば、ほかのものを落として測るかを 1 回だけ尋ねます(`[y/N]`。量と時間を見せる。`--download` で
   尋ねない。落とすときは `download` と同じ確かめ)。番号の付いた手順を出し、端末では ComfyUI の起動・1 回ごとの
   測定(目安に対する時間・そこまでの最大 VRAM)・ダウンロードの進み具合をその場で書き換えて見せます。端末で
   なければ、10 秒ほどごとに 1 行ずつ出します。
4. 送るファイルのすべて(中身のすべてと個人の情報の点検)を見せて、送り方を尋ねます: **1) PR**(GitHub にログイン:
   `GITHUB_TOKEN`・`GH_TOKEN`、無ければ保存したログイン、無ければブラウザで GitHub のデバイス認証。出たコードを入れる。
   待つあいだは `r` でページを開き直し、`n` で新しいコード、`q` で同じファイルを Issue で送る。コードの期限が切れたら、
   もう一度ログインするかを尋ねる。権限は `public_repo`。`share --logout` で消す)、**2) Issue**(ログインなし: ブラウザで実測のフォームを 1 つの
   ファイルに 1 つ開き、「Create」を押すとボットが PR を作る)、**q) 送らない**。PR は、あなたのアカウントのフォークの
   ブランチの `data/contrib/<GPU>/` にファイルを置いて開きます(開いている PR があればそこに足す)。数分で検査が動き、
   メンテナがマージすると次の版から使われます。
5. 結果は送る前に保存し、測り直しません。q を選んだときや PR で送れなかったとき(端末なら Issue で送るかを尋ねます)は、
   `share --no-measure`(か `share --issue`)であとで送れます。測る前は、もうあるトークンを確かめるだけです。`--yes` で
   端末でないときはトークン(`GITHUB_TOKEN`)が要ります(無ければ `share --issue --yes`)。

`share --dry-run` は計画と送る中身を見せるだけ、`share --no-measure` はまだ送っていない実測だけを送ります。
`--yes` の無い `share --issue` は前からの送り方で、送っていない実測を選んでフォームを開きます。
測るのは、同時に 1 つの `measure` か `share` だけです。genmodelfit が起動した ComfyUI は、終わったら止めます
(失敗・Ctrl+C・端末を閉じたときも)。測る前は、ほかのプログラムが GPU をあまり使っていないことを確かめます(GPU の
メモリを 1.5 GiB か 20% より多く使っていれば測らない)。nvidia-smi の無い AMD・Intel・Apple の GPU では、できる
ところは ComfyUI で確かめ、確かめられなかったことは結果のメモに書きます。

`genmodelfit measure` は、選んだ 1 つを測ります。この GPU に載り、この PC でまだ測っていないモデルを出します
(入っているもの → 時間が短いもの → 落とす量が少ないものの順。ComfyUI に無いノードが要るものは出さない)。選ぶと、
何をするか(モデル・設定・どの ComfyUI か・ファイルが入っているか・時間の目安・何も送らないこと)を見せてから、
`share` と同じ ComfyUI で測ります: この PC で起動中のもの(か `--url`)、無ければ見つけた ComfyUI を起動して、
終わったら止めます(`--no-start` で起動しない)。起動した ComfyUI は測る前に空けます。自分の ComfyUI は `--free`
のときだけです。入っていないモデルは、`download` と同じ確かめをして先に落とします。
この PC でテンプレートの設定のまま測ったモデルには、予測式の確かめに役立つ「設定を変えた測定」も出ます:
画像の解像度(512・768・1024・1536 px)、動画のコマ数(大きさはテンプレートのまま 17・33・49・81)、音楽・効果音の
長さ(10・30・60・120 秒)。その設定でこの GPU に載り、この PC でその設定をまだ測っていないものだけで、1 つの
モデルに 2 件まで、時間は 5 分ほどまでです(`--resolution`・`--frames`・`--length` の付いた行。自分で選ぶなら
`genmodelfit measure <ID> --resolution 1536x1536`)。1 回は生成 1 回で数分です。保存したものは `share` が送ります。
測れなかったときは失敗の記録を保存し、次の `share` が送ります。

送るもの(送る前にすべて見せます): GPU の名前・VRAM・世代と compute capability(NVIDIA)・ドライバの版・会社と backend・
AMD・Intel の GPU では PCI の型番 ID(`1002:744c` のような機種の番号。同じ機種ならどのカードでも同じ)・OS の種類・
GiB に丸めたメモリ・ComfyUI と genmodelfit の版・日付(時刻は入れない)・モデル・設定・結果・最大 VRAM / メモリ・時間・
モデルの載せ方(`offload`)・測り方・モデルのファイルの指紋・bench が書く測り方のメモ。0.13.0 からは、それに対する
genmodelfit の見積もり(この PC の実測は使わずに作ったもの)・時間の内訳・最大 VRAM が共有メモリに逃げた印・一覧の版と、
1 回に 1 つの失敗の記録のファイル(モデル・失敗の種類・足りないノードの名前・エラーの種類の名前・失敗したノードの
種類)も送ります。送らないもの: ホスト名・ユーザー名・パス・GPU の UUID・PCI の差し場所・シリアル番号・IP・CPU 名・
正確な時刻・プロンプト・エラーの本文・自分で止めた測定やほかのプログラムに乱された測定・この PC にファイルが無い
モデル。

何に使われるか: 実測は同じ GPU を使う人の判定に使います。NVIDIA で ComfyUI の動的 VRAM(今の既定)のときは作業用メモリ
しか測れないので、見積もりの確かめ(照合)に使い、まだ段階は決めません(#100。[accuracy.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.ja.md#実測の使い方))。
ほかの実測は、その GPU の見積もりを直し、別々の 2 人の実測がそろうと段階も決めます。1 つの GitHub アカウントは何件送っても 1 回だけ数えます。メモリが足りなかった結果は「この GPU では足りない」記録として残り、その設定では実測で段階を決めなく
なります。実測が使われるのは、メンテナが受け入れ、genmodelfit の次の版が出てからです。

ボットが Pull Request を作ったあとに Issue を編集すると、新しい中身でもう一度確かめます: Pull Request を直し(通らない
中身なら閉じ)、両方のページにそう書きます。git に慣れていれば、フォークの clone で `bench --submission data/contrib/`
を使う方法も残っています。手順: [CONTRIBUTING.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.ja.md)。

### 送るもの・送らないもの

- 判定のコマンドと Python の関数(自分で呼ぶ `update_from_release` と `download` を除く)は**ネットにつなぎません**(すべてのソケットをふさいだ試験で確かめています)。
- `update` は公開の情報(GitHub・Hugging Face の API・モデルの URL への HEAD)を読むだけ、`--from-release` は `db.json` と署名を取るだけです。
  `download` はモデルのファイルを取るだけ(フォルダを指定しないときは、この PC で起動中の ComfyUI(127.0.0.1 の
  8188・8000)にフォルダの場所も尋ねる)。`measure` と `share` は同じポートに ComfyUI が起動しているかを尋ね
  (`--url` が無いとき)、その ComfyUI とだけ話します(ノードの確かめと測る仕事。ほかは、その ComfyUI に無いときの
  GitHub の公式のテンプレートと入力の画像と、同意したときのモデルのファイル)。起動した ComfyUI は 127.0.0.1 でだけ
  待ちます。`bench` は `--url` で指定した ComfyUI とだけ話します。
- genmodelfit は PC の情報・実測・プロンプトを勝手には送りません。`share` は測る前に、もうある GitHub のトークン
  (`GITHUB_TOKEN`・`GH_TOKEN`・保存したログイン)を api.github.com で確かめるだけで、ほかには何も付けません。実測が共有されるのは、`share` で同意したとき
  (先にすべてのファイルを見せる)だけで、そのとき話すのは github.com と api.github.com だけです(ログインと、
  あなたのアカウントからそのファイルの PR を開く。[実測で協力する](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測で協力する))。
  ログインのトークンは設定のフォルダに本人だけが読める形で保存し、表示しません。`share --issue`(と `bench --share`・
  `contrib share`)はブラウザでフォームを開くだけで(か URL を出す・中身をクリップボードに写す)、そこで「Create」を
  押したときに送られます。`~/.netrc` の資格情報は付けません。テレメトリもありません。
- `share` が送るもの(0.13.0、#132): 実測(GPU の名前・VRAM・世代・ドライバ・backend、GiB に丸めたメインメモリ、
  OS の種類、ComfyUI と genmodelfit の版、モデル、設定、最大 VRAM・メモリ、時間)と、それに対する genmodelfit の
  見積もり(あなたの実測と、同じ GPU・同じモデルの実測は使わずに作ったもの)・時間の内訳(キュー・読み込み・サンプラ・復元)・最大 VRAM が
  GPU の総量を超えて共有メモリに逃げた印・一覧の版。ほかに 1 回の share で 1 つ、失敗の記録のファイル(足りない
  ノードの名前・時間切れ・エラーの種類の名前。エラーの本文は送らない)。送らないもの: 生成した画像や音などの中身、
  あなたの入力、PC 名・ユーザ名、GPU の UUID、PCI の位置、パス、IP アドレス。
  詳しくは [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#contrib)(「contrib」)。
- コマンドごとの表: [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#side-effects--通信と書き込み)(「Side effects」)。

### ライセンス

[Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE)(MPL-2.0)。同梱のデータも同じです。
genmodelfit を直さずに使うだけなら、ソース非公開のアプリにも組み込めます。genmodelfit のファイルを直して配るときは、そのファイルを
MPL-2.0 で公開してください。アプリ(例: PyInstaller の exe)に同梱して配るときは、ライセンスの表示を残し、ソースの入手先
(https://github.com/laxshmate/genmodelfit、MPL-2.0 §3.2)を利用者に知らせてください。`certifi`(MPL-2.0)も同じです。
データの出典は [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES)。
モデルのライセンスは表示するだけで、利用を許諾するものではありません。
