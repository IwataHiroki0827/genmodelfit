# genmodelfit

[English](https://github.com/laxshmate/genmodelfit/blob/main/README.md) · **日本語**

画像・動画・音楽・効果音・読み上げ・3D の生成モデルのうち、どれがあなたの PC で快適に動くかを判定するツールです。

[![PyPI](https://img.shields.io/pypi/v/genmodelfit)](https://pypi.org/project/genmodelfit/)

LLM 向けに同じことをする [llmfit](https://github.com/AlexsJones/llmfit) に着想を得ています。
llmfit の公式のプロジェクトでも派生(フォーク)でもありません。状態: アルファ版
([CHANGELOG.md](https://github.com/laxshmate/genmodelfit/blob/main/CHANGELOG.md))。

すぐ試す:

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

- Windows・Linux・macOS、Python 3.10 以上。
- GPU: NVIDIA(CUDA)は実測で確かめています。使える NVIDIA の GPU が無いときは、AMD(ROCm: RDNA 2 以降、Windows 11・Linux)・
  Intel(XPU: Arc A / B シリーズ、Core Ultra の Arc Graphics、Windows 11・Linux)・Apple silicon(MPS、macOS 14 以降)を
  推定で判定します(段階は Good まで)。それ以外の GPU(Intel UHD・Iris Xe、古い AMD)と GPU の無い PC は CPU だけとして
  判定します。SD 系の画像モデルと Stable Audio は CPU でも動く(遅い)と判定します
  ([docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md))。

### 入れ方

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

6. **一覧を新しく保つ。** 一覧は毎週作り直しています。リンクの確認は 30 日で切れ、切れると Verified のモデルが無くなります。
   このコマンドで、このプロジェクトが署名して配る最新の一覧を取り込みます。
   ```
   genmodelfit update --from-release
   ```

### アプリから使う(Python / JSON)

各コマンドは `--json` を付けると JSON で出します([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md))。
Python では次のとおりです。

```python
import genmodelfit as gmf

for f in gmf.recommend(modality="image", task="edit", limit=5):
    print(f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, f.estimate.vram_gb)
    row = f.to_dict()  # `recommend --json` の 1 件と同じ dict

st = gmf.data_status()  # 一覧は Verified(auto)を出せるほど新しいか
if st.expired:
    print("run: genmodelfit update --from-release", st.expires_at)

# 一覧を自分で読むときは、確認待ちのものを除く
ok = [e for e in gmf.load_db().models if e.review is None or e.review.status == "ok"]

# アプリが調べた GPU で 1 モデルを判定する(世代と精度は genmodelfit が決める)
fit = gmf.judge_model("image_sdxl_simple", gpu_name="NVIDIA GeForce RTX 3060", vram_gb=12, driver="560.94")
print(fit.to_dict()["fit_level"])  # 互換を守るのは gmf.__all__ の名前だけ

# 署名つきの一覧を取り込む(Python で通信するのはこれだけ)。自分のセッション(アプリの CA バンドルなど)とデータフォルダで
db = gmf.update_from_release(session=my_session, dest=data_dir / "db.json", state_dir=data_dir)
```

### ほかの PC で判定する

`--vram` と `--ram`(GiB)で、検出した大きさを置き換えます。例: `genmodelfit recommend --vram 12 --ram 32`。
GPU の世代はこの PC のものを使います。`--width`・`--height`・`--frames`・`--length`・`--precision` で設定を決めて判定できます。

## 3. 結果

### 例

VRAM 8 GiB・メモリ 32 GiB の PC での `genmodelfit recommend`(種類ごとに上位 2 件。`COLUMNS=100`):

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 12 of 80 (224 hidden: uncensored 26, tools 34, license unknown 28, too tight 136; 25
    more workflows merged into their models)
== image (画像): top 2 of 37  (-m image for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Marginal     54 Z-Image-Turbo            gen+5   int8  7.6m  96% GPU~    apache-2.0 yes
image ● Marginal     42 MOSS-Image SFT           gen+1   int8   4.2  53% GPU+RAM apache-2.0 yes
== video (動画): top 2 of 4  (-m video for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
video ● Marginal     50 Wan 2.1 1.3B T2V         gen     fp16  7.8m  97% GPU~    apache-2.0 yes
video ● Marginal     39 Wan 2.1 ControlNet 1.3B  edit    bf16   7.7  96% GPU+RAM apache-2.0 yes
== music (音楽): top 2 of 10  (-m music for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
music ● Good         65 ACE-Step 1.5 Turbo       gen+2   bf16  6.0m  76% GPU~    mit        yes
music ● Marginal     40 ACE-Step 1.5XL Turbo     gen     bf16   5.2  65% GPU+RAM mit        yes
== sfx (効果音): top 2 of 2  (-m sfx for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
sfx   ● Perfect      68 stable-audio-3-small-sfx gen     fp32   3.6  45% GPU     other      license
sfx   ● Good         54 stable-audio-open-1.0    gen+1   fp32   4.9  61% GPU~    other      license
== tts (読み上げ): top 2 of 20  (-m tts for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
tts   ● Perfect      80 Chatterbox               gen     fp32   3.5  44% GPU     mit        workflow
tts   ● Perfect      80 VieNeu-TTS-v3-Turbo      gen     q8     1.6  21% GPU     apache-2.0 workflow
== 3d (3D): top 2 of 7  (-m 3d for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
3d    ● Good         64 HY 3D 2.0                gen     fp16  6.7m  84% GPU     other      license
3d    ● Good         57 TripoSplat               gen     fp16   5.8  72% GPU     other      license
Order: runs, Verified=yes, Perfect/Good, Marginal, Score. Score 0-100. VRAM: GiB needed, m =
    measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Verified: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
Note: 26 uncensored model(s) hidden; add --content all to show them.
Note: 28 model(s) with an unknown license hidden; add --include-unknown-license to show them.
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
```

最初の行は PC と、隠したモデルの数と理由です。最後の行は印の説明です。
`fit` は 1 つの表で出します(VRAM 24 GiB・メモリ 128 GiB、画像だけ、先頭の 6 行):

```
$ genmodelfit fit --vram 24 --ram 128 -m image -n 6
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 6 of 60 (218 hidden: other types 166, uncensored 21, tools 19, license unknown 12; 51
    more workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Perfect      90 SDXL1.0 Base             gen+3   fp16  8.0m  33% GPU     openrail++ license
image ● Perfect      84 MOSS-Image SFT           gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      83 FLUX.2 [Klein] 4B Base   edit    fp8   13.2  55% GPU     apache-2.0 yes
image ● Perfect      82 MOSS-Image Base          gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      81 Anima Base 1.0           gen+3   bf16   7.1  30% GPU     other      license
image ● Perfect      80 Anima Preview 3 Base     gen     bf16   7.0  29% GPU     other      license
Order: Type, Status, Score (--sort). Score 0-100. VRAM: GiB needed, m = measured.
…
```

### 列

| 列 | 意味 |
|---|---|
| Type | image・video・music・sfx・tts・3d |
| Status | 段階(下の表) |
| Score | 0〜100。段階・VRAM の余裕・Verified か・実測か・人気から計算([式](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)) |
| Censored | `Y` 検閲あり / `N` 無検閲。`--content all` か `uncensored` のときだけ出る |
| Model | 同じ重みを 1 行にまとめたモデル。長い名前は真ん中を `…` で詰める |
| Use | `gen`・`edit`・`i2v` など(`genmodelfit types`)。`+N` = 同じモデルのほかのワークフローが N 個(`--all-workflows`) |
| Prec | 載る中で一番良い精度(fp16・bf16・fp8・int8 など) |
| VRAM | その動かし方で要る GiB。`m` = 実際に動かして測った値 |
| Mem% | VRAM の使用率(CPU ならメモリ)。`RAM` = PC のメモリが足りない |
| Mode | `GPU` 全部 GPU、`GPU~` 部品を順に載せ替え、`GPU+RAM` メモリへ逃がす(遅い)、`CPU`(遅い)、`-` 動かない(`GPU req` は使える GPU が無いので動かない、`RAM` はメモリ不足) |
| Disk | ダウンロードの大きさ(GiB)。幅が足りないと最初に外すので、100 桁では `info` か `--json` で見る |
| License | ファイルのライセンス(`other` = 自分で原文を読む) |
| Verified | `yes` = 自動でおすすめに出せる。それ以外は最初に落ちた条件(`license`・`workflow`・`review` など) |

### 段階

llmfit と同じく、動かし方で使うメモリ(GPU なら VRAM、CPU ならメモリ)の使用率で決めます。

| 段階 | 使用率 | 動かし方による上限 |
|---|---|---|
| **Perfect** | 60% 以下 | 全部 GPU に載るとき(`GPU`)だけ |
| **Good** | 85% 以下 | `GPU~` はここまで |
| **Marginal** | 98% 以下 | `GPU+RAM`・`CPU`・メモリが少し足りないときはここまで(Mem% が低くても) |
| **Too tight** | 98% 超 | 動かない |
| **?** | — | 判定していない(ダウンロードの大きさが分からない) |

印: VRAM の後ろの `m` = 実測(実測があれば実測を使う。`m` が無ければ式による推定)。
`?` = 分からない(精度・ライセンス・大きさ。判定に使っていない)。`-` = 当てはまらない(動かない・CPU なので VRAM が無い など)。
`recommend` の並びは「動く → Verified → Perfect/Good → Marginal → Score」、`fit` は「種類 → 段階 → Score」です。

### 精度

見積もりの較正に使った実測は少なく、GPU も 1 種類(RTX 5080 16 GB)です。`m` の無い値は目安として見てください。
ComfyUI は空きがあるほど VRAM を多く使うので、大きい GPU で測ったピークは最低限の必要量ではありません。
詳しい誤差は [docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md)。

## 4. 参考

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
| `review`・`diff`・`bench`・`contrib` | 保守と実測(下) |

オプション・終了コード・例は各コマンドの `--help` にあります。よく使うオプション: `--json`・`--vram`・`--ram`・
`--content`・`--include-unknown-license`・`--license apache-2.0,mit`・`--comfyui-version`。

### データ源

一覧は [ComfyUI のワークフローテンプレート](https://github.com/Comfy-Org/workflow_templates)(MIT)と
[Hugging Face Hub の API](https://huggingface.co/docs/hub/api)(上流までたどったライセンス・大きさ・sha256)から作り、
少しだけ手で足したモデルがあります。llmfit と同じく、Hugging Face のモデルは全期間のダウンロード数 10,000 以上が条件です。
Civitai は使いません(規約が自動アクセスを制限しているため)。詳しくは [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)。

### 内容の印

安全フィルタを外したモデルや成人向けの出力を想定したモデルは `uncensored` の印を付け、**既定では出しません**。
`--content all`(または `uncensored`)で `Censored` の列と一緒に出します。既定では Verified になりません。
未成年の性的表現を目的とするモデルは一覧から除外しています
([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md))。

### Verified(自動でおすすめに出す条件)

`Verified: yes`(JSON の `auto`)は、次を**すべて**満たすときだけです: すべてのファイルのライセンスが商用利用できる許可リストにある、
gated が無い、カスタムノードの無い公式の ComfyUI ワークフロー、対応する ComfyUI の版、すべての URL が 30 日以内の匿名の HEAD に答えた、
sha256 と大きさが分かる、内容が `sfw`、確認待ちでない。満たさない理由は `held_reasons` に入ります。
詳しくは [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md)。

### ダウンロード

一覧にある https の URL だけを使い、転送先は Hugging Face・GitHub のホストだけ、書きながら sha256 を確かめ、Ctrl-C のあとは続きから、
置き場所の外には書きません。gated・無検閲・ライセンス不明のファイルは、許可しない限り落としません。
詳しい決まりと Python の関数は [docs/download.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.md)。

### 一覧の保守

一覧は毎週作り直します。新しいモデルや危うい変化には `pending` の印を付け、人が一次資料の根拠を付けて承認する
(`genmodelfit review`)まで Verified にしません。署名して `update --from-release` に配るのは確認済みのデータだけです。
詳しくは [docs/release.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.md)。

### 実測を送る

いろいろな GPU の実測が集まるほど、見積もりが良くなります。動いている ComfyUI で `genmodelfit bench` を実行すると提出用のファイルができ、
`genmodelfit contrib show` で中身をそのまま確かめられます(ホスト名・ユーザー名・パス・GPU の UUID は入りません)。
Pull Request はご自身で出してください。手順: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)。

### 送るもの・送らないもの

- 判定のコマンドと Python の関数(自分で呼ぶ `update_from_release` を除く)は**ネットにつなぎません**(すべてのソケットをふさいだ試験で確かめています)。
- `update` は公開の情報(GitHub・Hugging Face の API・モデルの URL への HEAD)を読むだけ、`--from-release` は `db.json` と署名を取るだけです。
  `download` はモデルのファイルを取るだけ、`bench` は `--url` で指定した ComfyUI とだけ話します。
- PC の情報・実測・プロンプトは送りません。`~/.netrc` の資格情報は付けません。テレメトリもありません。
- コマンドごとの表: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)(「Side effects」)。

### ライセンス

[Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE)(MPL-2.0)。同梱のデータも同じです。
genmodelfit を直さずに使うだけなら、ソース非公開のアプリにも組み込めます。genmodelfit のファイルを直して配るときは、そのファイルを
MPL-2.0 で公開してください。アプリ(例: PyInstaller の exe)に同梱して配るときは、ライセンスの表示を残し、ソースの入手先
(https://github.com/laxshmate/genmodelfit、MPL-2.0 §3.2)を利用者に知らせてください。`certifi`(MPL-2.0)も同じです。
データの出典は [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES)。
モデルのライセンスは表示するだけで、利用を許諾するものではありません。
