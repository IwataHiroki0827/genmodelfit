# genmodelfit

[English](https://github.com/laxshmate/genmodelfit/blob/main/README.md) · **日本語**

画像・動画・音楽・効果音・読み上げ・3D の生成モデルのうち、どれがあなたの PC で快適に動くかを判定するツールです。

[![PyPI](https://img.shields.io/pypi/v/genmodelfit)](https://pypi.org/project/genmodelfit/)

LLM 向けに同じことをする [llmfit](https://github.com/AlexsJones/llmfit) に着想を得ています。
llmfit の公式のプロジェクトでも派生(フォーク)でもありません。状態: アルファ版
([CHANGELOG.md](https://github.com/laxshmate/genmodelfit/blob/main/CHANGELOG.md))。

![VRAM 8 GiB・メモリ 32 GiB の PC での genmodelfit recommend](https://raw.githubusercontent.com/laxshmate/genmodelfit/main/docs/images/recommend.png)

`genmodelfit recommend --vram 8 --ram 32 -n 2` の先頭(2026-10-22 時点の一覧)。一覧は毎週更新するので、並ぶモデルと件数は変わります。

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
  macOS 14 以降)を推定で判定します(段階は Good まで。同じ設定の実測で段階が決まれば外れます: 同じ会社の GPU での作者・この PC の実測か、同じ GPU のほかの利用者の実測)。内蔵 GPU・APU
  (Radeon 780M / 680M・Steam Deck・Ryzen Z1 など)は共有メモリで判定します。それ以外の GPU(Intel UHD・Iris Xe、2 CU の
  Radeon 610M、RDNA 1 以前)と GPU の無い PC は CPU だけとして判定します。SD 系の画像モデルと Stable Audio は CPU でも動く(遅い)と
  判定します。Linux で lspci が GPU の名前を知らないときは、`system` が `update-pciids` を案内します
  ([docs/hardware.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.ja.md))。
  NVIDIA 以外の GPU は実測が無いので、段階を控えめにしています。`genmodelfit bench` はこれらの GPU でも測れます。実測を
  Pull Request で出してもらえると詰められます(genmodelfit は実測を送りません。[CONTRIBUTING.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.ja.md))。

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

判定には通信しません。PC の情報は手元で読みます。通信するのは `update`・`download`・`bench` だけで、
相手は公開の情報か、指定した ComfyUI だけです。利用状況の送信(テレメトリ)もありません。詳しくは[送るもの・送らないもの](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#送るもの送らないもの)。

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
   ```
   genmodelfit download "Z-Image-Turbo" --list
   genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI   # ComfyUI を入れたフォルダ
   ```

6. **一覧を新しく保つ。** 一覧は毎週作り直しています。リンクの確認から 30 日を過ぎたモデルは Verified でなくなります。
   このコマンドで、このプロジェクトが署名して配る最新の一覧を取り込みます。
   ```
   genmodelfit update --from-release
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

# 署名つきの一覧を取り込む(通信するのはこれと gmf.download だけ)。自分のセッション(アプリの CA バンドルなど)とデータフォルダで
db = gmf.update_from_release(session=my_session, dest=data_dir / "db.json", state_dir=data_dir)
```

互換を守るのは `gmf.__all__` の名前だけです([docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#python-api-stable--公開の範囲))。

### ほかの PC で判定する

`--vram` と `--ram`(GiB)で、検出した大きさを置き換えます。例: `genmodelfit recommend --vram 12 --ram 32`。
使える NVIDIA の GPU があれば、その世代のまま VRAM だけを変えます。NVIDIA 以外の GPU(または使える GPU が無い PC)では、`--vram` は世代の分からない仮の NVIDIA の GPU を判定します。
ほかの利用者の実測は、`--vram` が実測した GPU の VRAM と 3% 以内のときだけ使います。
`--width`・`--height`・`--frames`・`--length`・`--precision` で設定を決めて判定できます。

### 表示の言語

出力の既定は英語。日本語にするには、どのコマンドでも `--lang ja` を付ける(例 `genmodelfit recommend --lang ja`)か、
環境変数 `GENMODELFIT_LANG=ja` を置くか、Python で `gmf.set_language("ja")` を呼ぶ(優先は `--lang`、環境変数、
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

VRAM 8 GiB・メモリ 32 GiB の PC での `genmodelfit recommend --lang ja`(種類ごとに上位 2 件。`COLUMNS=100`):

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2 --lang ja
PC: VRAM 8.0 GiB と仮定(--vram)、メモリ 32.0 GiB(--ram)
おすすめ: 80 件中 12 件(隠した 224 件: 無検閲 26・補助ツール 34・ライセンス不明 28・動かない 136。ほ
    かに 25 件のワークフローをモデルにまとめた)
== 画像 (image): 37 件中 上位 2 件(続きは -m image)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Good         62 SDXL1.0 Base               gen       fp16  6.3m  79% GPU~     6.5 openrail++
image ● Good         56 Anima Base 1.0             gen,edit  bf16   5.8  72% GPU~     5.4 other
== 動画 (video): 4 件中 上位 2 件(続きは -m video)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
video ● Marginal     50 Wan 2.1 1.3B T2V           gen       fp16  7.8m  97% GPU~     9.2 apache-2.0
video ● Marginal     39 Wan 2.1 ControlNet 1.3B    edit      bf16   7.7  96% GPU+RAM 10.6 apache-2.0
== 音楽 (music): 10 件中 上位 2 件(続きは -m music)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
music ● Good         65 ACE-Step 1.5 Turbo         gen       bf16  6.0m  76% GPU~     9.3 mit
music ● Perfect      64 Stable Audio 3 Small music gen       fp32   4.7  59% GPU      3.2 other
== 効果音 (sfx): 2 件中 上位 2 件(続きは -m sfx)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
sfx   ● Perfect      66 stable-audio-3-small-sfx   gen       fp32   4.7  59% GPU      3.2 other
sfx   ● Good         54 stable-audio-open-1.0      gen       fp32   4.9  61% GPU~     5.3 other
== 読み上げ (tts): 20 件中 上位 2 件(続きは -m tts)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
tts   ● Perfect      80 Chatterbox                 gen       fp32   3.5  44% GPU      2.0 mit
tts   ● Perfect      80 VieNeu-TTS-v3-Turbo        gen       q8     1.6  21% GPU      0.2 apache-2.0
== 3D (3d): 7 件中 上位 2 件(続きは -m 3d)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
3d    ● Good         64 HY 3D 2.0                  gen       fp16  6.7m  84% GPU      4.6 other
3d    ● Good         57 TripoSplat                 gen       fp16   5.8  72% GPU      3.5 other
並び: 動く → Perfect/Good → Marginal → Score。Score 0〜100。VRAM = 要る GiB(m = 実測)。
Mode: GPU 全部 GPU、GPU~ 部品を順に載せ替え、GPU+RAM メモリへ逃がす(遅い)、CPU(遅い)、- 動かない。
段階の上限: GPU~ は Good まで、GPU+RAM と CPU は Marginal まで(Mem% が低くても)。
Mem% = VRAM(CPU はメモリ)の使用率、RAM = メモリ不足。
1 行に 1 モデル: Prec = 載る中で一番高い精度、Use = ワークフローの用途(info、--all-workflows)。
m の無い見積もりは目安。詳しくは genmodelfit info <名前か ID>。アプリは --json(schema.md)。
--json は自動でおすすめしてよいものを先に並べる(schema.md)。
注意: 無検閲のモデル 26 件を隠しました。出すなら --uncensored。
注意: ライセンスの分からないモデル 28 件を隠しました。出すなら --any-license。
? = 分からない(精度・ライセンス・大きさ。判定していない)、- = 当てはまらない(動かないなど)。
```

最初の行は PC と、隠したモデルの数と理由です。最後の行は印の説明です。
`fit` は 1 つの表で出します(VRAM 24 GiB・メモリ 128 GiB、画像だけ、先頭の 6 行。`--lang ja`):

```
$ genmodelfit fit --vram 24 --ram 128 -m image -n 6 --lang ja
PC: VRAM 24.0 GiB と仮定(--vram)、メモリ 128.0 GiB(--ram)
全モデル: 60 件中 6 件(隠した 218 件: ほかの種類 166・無検閲 21・補助ツール 19・ライセンス不明 12。
    ほかに 51 件のワークフローをモデルにまとめた)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Perfect      90 SDXL1.0 Base               gen       fp16  8.0m  33% GPU      6.5 openrail++
image ● Perfect      84 MOSS-Image SFT             gen       int8  13.8  58% GPU     12.1 apache-2.0
image ● Perfect      83 FLUX.2 [Klein] 4B Base     edit      fp8   13.2  55% GPU     11.5 apache-2.0
image ● Perfect      82 MOSS-Image Base            gen       int8  13.8  58% GPU     12.1 apache-2.0
image ● Perfect      81 Anima Base 1.0             gen,edit  bf16   7.1  30% GPU      5.4 other
image ● Perfect      80 Anima Preview 3 Base       gen       bf16   7.0  29% GPU      5.2 other
並び: Type → Status → Score(--sort)。Score 0〜100。VRAM = 要る GiB(m = 実測)。
…
```

### 列

| 列 | 意味 |
|---|---|
| Type | image・video・music・sfx・tts・3d |
| Status | 段階(下の表) |
| Score | 0〜100。段階・VRAM の余裕・Verified か(条件ごとの結果は `info`)・実測か・人気から計算([式](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#score--点数)) |
| Censored | `Y` 検閲あり / `N` 無検閲。`--uncensored`(か `--content all|uncensored`)のときだけ出る |
| Model | 同じ重みを 1 行にまとめたモデル。長い名前は真ん中を `…` で詰める |
| Use | `gen`・`edit`・`i2v` など(`genmodelfit types`)。行自身の用途(`--task` で合ったもの)を先頭に、そのモデルのほかのワークフローの用途を重複なく並べる(`gen,edit`。収まらなければ `gen,…`)。ワークフローの一覧は `info`、別々の行は `--all-workflows` |
| Prec | 載る中で一番良い精度(fp16・bf16・fp8・int8 など) |
| VRAM | その動かし方で要る GiB。`m` = 作者・この PC の実測、`c` = 同じ GPU のほかの利用者の実測の中央値(件数は `info`・JSON の `sample_count`) |
| Mem% | VRAM の使用率(CPU ならメモリ)。`RAM` = PC のメモリが足りない |
| Mode | `GPU` 全部 GPU、`GPU~` 部品を順に載せ替え、`GPU+RAM` メモリへ逃がす(遅い)、`CPU`(遅い)、`-` 動かない(`GPU req` は使える GPU が無いので動かない、`RAM` はメモリ不足) |
| Disk | ダウンロードの大きさ(GiB)。幅が足りないと最初に外す(そのときは `info` か `--json` で見る) |
| License | ファイルのライセンス(`other` = 自分で原文を読む) |

### 段階

llmfit と同じく、動かし方で使うメモリ(GPU なら VRAM、CPU ならメモリ)の使用率で決めます。

| 段階 | 使用率 | 動かし方による上限 |
|---|---|---|
| **Perfect** | 60% 以下 | 全部 GPU に載るとき(`GPU`)だけ |
| **Good** | 85% 以下 | `GPU~` はここまで。AMD・Intel・Apple の GPU も、同じ設定の実測で段階が決まらないうちはここまで(同じ会社の GPU での作者・この PC の実測か、同じ GPU のほかの利用者の実測) |
| **Marginal** | 98% 以下 | `GPU+RAM`・`CPU`・メモリが少し足りないときはここまで(Mem% が低くても) |
| **Too tight** | 98% 超 | 動かない |
| **?** | — | 判定していない(ダウンロードの大きさが分からない) |

印: VRAM の後ろの `m` = 作者・この PC の実測、`c` = ほかの利用者の実測の中央値。印が無ければ式による推定です。
実測は式より優先します。ほかの利用者の実測は同じ GPU(PCI の型番 ID、無ければ正規化した名前が同じで、VRAM の差が 3% 以内)に
だけ使い、必要量を上下どちらにも動かします。`bench` の値(`torch_reserved`)は、段階を決めるときに 0.5 GiB を足します。
`?` = 分からない(精度・ライセンス・大きさ。判定に使っていない)。`-` = 当てはまらない(動かない・CPU なので VRAM が無い など)。
`recommend` の並びは「動く → Perfect/Good → Marginal → Score」、`fit` は「種類 → 段階 → Score」です。
(`recommend --json` と `gmf.recommend()` の並びは今までどおりで、Verified のモデルが先です: [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#order-of-recommend--recommend-の並べ方)。)

### 精度

0.9.1 の時点では、見積もりの較正に使った実測は少なく、GPU も 1 種類(RTX 5080 16 GB)です。照合に RTX 3070 8 GB の実測も
少し使っています。ほかの利用者の実測はまだ同梱していません。`m`・`c` の無い値は目安として見てください。
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
| `review`・`bench`・`contrib` | 保守と実測(下) |

オプション・終了コード・例は各コマンドの `--help` にあります。よく使うオプション: `--json`・`--vram`・`--ram`・
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
`--any-license` が要ります(足したコマンドを 1 行と、それぞれの意味を出します。勝手に足して落とすことはしません)。ComfyUI のどのフォルダか分からないファイルは `other/` に置きます。ComfyUI は読まないので、
計画の中でそう知らせます(使う前に移す)。環境変数 `GENMODELFIT_COMFYUI_DIR` に ComfyUI のフォルダを入れておけば `--comfyui-dir` を省けます。
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

### 実測を出す(Pull Request)

いろいろな GPU の実測が集まるほど、見積もりが良くなります。動いている ComfyUI(NVIDIA・AMD・Intel・Apple の GPU)で
`genmodelfit bench <ID> --url http://127.0.0.1:8188 --share` を実行すると、測って送る中身を見せ、ファイルを入力済みの GitHub のページを
案内します(git は要りません。ログインして「Propose changes」と「Create pull request」を押すだけです)。
`--submission data/contrib/` なら、フォークの clone に提出用のファイルを作ります
(保存済みの結果は `genmodelfit contrib make --from <ファイル>` で変換できます)。`genmodelfit contrib show <ファイル>` で中身を
そのまま確かめられます。入るのは GPU 名・VRAM・ドライバ・vendor / backend・AMD・Intel の GPU では PCI の型番 ID(例 `1002:744c`)・
ComfyUI の版・OS の種類・GiB に丸めたメモリ・モデル・設定・結果です。`bench` はホスト名・ユーザー名・パス・GPU の UUID・PCI のアドレス・
IP・CPU 名・正確な時刻を書きません。genmodelfit は実測を送りません。Pull Request はご自身で出してください。手順: [CONTRIBUTING.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.ja.md)。

### 送るもの・送らないもの

- 判定のコマンドと Python の関数(自分で呼ぶ `update_from_release` と `download` を除く)は**ネットにつなぎません**(すべてのソケットをふさいだ試験で確かめています)。
- `update` は公開の情報(GitHub・Hugging Face の API・モデルの URL への HEAD)を読むだけ、`--from-release` は `db.json` と署名を取るだけです。
  `download` はモデルのファイルを取るだけ、`bench` は `--url` で指定した ComfyUI とだけ話します。
- genmodelfit は PC の情報・実測・プロンプトを送りません。実測が共有されるのは、ご自身で Pull Request を出したときだけです
  ([実測を出す](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md#実測を出すpull-request))。
  `bench --share` / `contrib share` は GitHub のページの URL を出し、端末で y と答えたときにブラウザで開くだけです。
  genmodelfit 自身は何も送らず、ファイルが送られるのは GitHub のページでボタンを押したときだけです。
  `~/.netrc` の資格情報は付けません。テレメトリもありません。
- コマンドごとの表: [docs/schema.ja.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.ja.md#side-effects--通信と書き込み)(「Side effects」)。

### ライセンス

[Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE)(MPL-2.0)。同梱のデータも同じです。
genmodelfit を直さずに使うだけなら、ソース非公開のアプリにも組み込めます。genmodelfit のファイルを直して配るときは、そのファイルを
MPL-2.0 で公開してください。アプリ(例: PyInstaller の exe)に同梱して配るときは、ライセンスの表示を残し、ソースの入手先
(https://github.com/laxshmate/genmodelfit、MPL-2.0 §3.2)を利用者に知らせてください。`certifi`(MPL-2.0)も同じです。
データの出典は [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES)。
モデルのライセンスは表示するだけで、利用を許諾するものではありません。
