# genmodelfit

[English](https://github.com/laxshmate/genmodelfit/blob/main/README.md) · **日本語**

**画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。**

genmodelfit は、LLM 向けに同じことをする [llmfit](https://github.com/AlexsJones/llmfit) に着想を得た、生成モデル向けのツールです。
llmfit の公式の派生ではありません。

> 状態: v0.1.0(アルファ版)。

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

`-m` も `--task` も無ければ、`recommend` は 6 つの種類(image, video, music, sfx, tts, 3d)ごとに出します。
1 行は 1 つの**モデル**です。同じ重み(系統と大きさが同じ)を別の精度・別の使い方で使うワークフローは 1 行にまとめ
(`+5` = ほかに 5 つ。`--all-workflows` で全部出す)、載る中で一番良い精度を選びます(`Prec`。llmfit の best quant と同じ考え方)。
種類の中は `--task` で絞ります。

```
$ genmodelfit recommend --vram 8 --ram 32 -m image --task edit -n 3
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 3 of 16 (311 hidden: other types 166, uncensored 21, tools 19, other uses 63, license
    unknown 8, too tight 34; 2 more workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Marginal     41 FLUX.2 [Klein] 4B        edit+1  fp8    4.5  56% GPU+RAM apache-2.0 yes
image ● Marginal     41 FLUX.2 [Klein] 4B Base   edit    fp8    4.5  56% GPU+RAM apache-2.0 yes
image ● Marginal     41 Z-Image-Turbo            edit+5  bf16   7.5  93% GPU+RAM apache-2.0 yes
Order: runs, Verified=yes, Perfect/Good, Marginal, Score. Score 0-100. VRAM: GiB needed, m =
    measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Verified: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
Note: 21 uncensored model(s) hidden; add --content all to show them.
Note: 8 model(s) with an unknown license hidden; add --include-unknown-license to show them.
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
```

既定の出力は人が読む表です。アプリやエージェントは `--json` を付けてください(llmfit の recommend は既定が JSON ですが、
genmodelfit は初めて試す人のために表のままにしています)。

## なぜ作るか

llmfit が扱うのは LLM だけです([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41))。拡散モデル・動画・音声のモデルは、
精度・解像度・コマ数・メインメモリへの逃がし方で、必要なメモリが数倍変わります。
そのため LLM とは別のデータと判定が要ります。

## 入れ方

おすすめは [pipx](https://pipx.pypa.io/) か [uv](https://docs.astral.sh/uv/) でコマンドとして入れる方法です。
`genmodelfit` を PATH に通してくれます(PATH を通すコマンドを 1 回だけ実行し、端末を開き直す)。

```
pipx install genmodelfit        # 1 回だけ: pipx ensurepath
uv tool install genmodelfit     # 1 回だけ: uv tool update-shell
```

pip でも入れられます。

```
pip install genmodelfit
```

**`python -m genmodelfit …` は PATH に関係なく必ず動きます**(例: `python -m genmodelfit recommend`。Windows は
`py -m genmodelfit …` も可)。`genmodelfit` が「認識されません」「command not found」になるときに使ってください。

pip が *「WARNING: The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH」* と出したら、
コマンドは入っていますが、その場所に PATH が通っていません。よくあるのは次の 4 つです。

- python.org の Python を「Add python.exe to PATH」なしで入れた → その Python の `Scripts` フォルダに PATH が通っていない。
- 書き込めないため pip が利用者ごとの場所に入れた → コマンドは `%APPDATA%\Python\Python3XX\Scripts`(Windows)か
  `~/.local/bin`(Linux)にあり、PATH が通っていないことが多い。
- Microsoft Store の Python → `Scripts` フォルダが `%LOCALAPPDATA%\Packages\…` の深いところにあり、PATH が通っていない。
- 仮想環境を有効にしていない → 有効にする(`.venv\Scripts\activate` / `source .venv/bin/activate`)か、
  `.venv\Scripts\genmodelfit` を実行する。

直し方: 警告に出た場所を PATH に足す(または上の pipx / uv か `python -m genmodelfit` を使う)。
**PATH を変える前に開いた端末には反映されないので、新しい端末を開いてください。**

ソースから:

```
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10 以上。Windows と Linux。GPU は v0.1 では NVIDIA(CUDA)のみで、ほかの PC は CPU だけとして判定します。

## 使い方

| コマンド | 内容 |
|---|---|
| `genmodelfit system [--all-gpus]` | この PC の GPU(世代・ドライバ・PyTorch の版・ComfyUI の精度)・メモリ・ページファイル・空きと、一覧の新しさ(Data)。`--all-gpus` で NVIDIA 以外のアダプタも出す |
| `genmodelfit recommend [-m image,video] [--task edit] [-n 5] [--auto-only]` | この PC に合うおすすめ順。`-m`・`--task` が無ければ種類ごと(深度・ポーズ・拡大などの補助ツールは `--include-tools` のときだけ) |
| `genmodelfit fit [-m video] [--fit good] [--sort score] [--all]` | この PC で動くモデル(Perfect / Good / Marginal)の表。種類 → 段階 → 点数の順。`--all` で補助ツール・動かないもの・判定できない(`?`)もの・ライセンス不明のものも出す |
| `genmodelfit search <text>` | ID・名前の一部で探す。この PC 向けに判定し、全段階を `fit` と同じ順で出す(見つからなくても終了コードは 0) |
| `genmodelfit types` | 6 つの種類と、その中の用途(`--task` の名前)と数 |
| `genmodelfit info <id>` | 1 つのモデルの詳細(ファイル・ライセンス・見積もり・限界・gate の条件・同じモデルのほかのワークフロー) |
| `genmodelfit gate` | 自動でおすすめに出せる数と、保留の理由 |
| `genmodelfit update` | 一覧を上流から作り直す |
| `genmodelfit update --fields-only [--offline]` | 一覧はそのままで、アプリ向けの項目(`languages` / `runtime` / `description`)と手で足したモデルだけを当て直す |
| `genmodelfit update --from-release` | このプロジェクトが配る署名付きの一覧を取り込む |
| `genmodelfit review list [--json]` / `approve` / `reject` | `update` のあと、人の確認を待っているモデル([一覧の保守](#一覧の保守)) |
| `genmodelfit download <model> [--comfyui-dir DIR] [--list]` | この PC に合う精度でモデルのファイルを全部落とす(`recommend` と同じ選び方)。ライセンス・大きさ・空きを見せて同意を取る。sha256 で確かめ、続きから再開できる([下](#モデルを落とす)) |
| `genmodelfit bench <template> --url URL` | 動いている ComfyUI で VRAM・メモリ・時間を実測する |

共通の指定:

- `--json` — アプリ・エージェント向けの JSON([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md))
- `--vram 12 --ram 32` — VRAM・メモリの量を変えて判定する(この PC に使える NVIDIA の GPU があれば、その世代は引き継ぐ。[docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md))
- `--width 1024 --height 1024`、`--frames 81`、`--length 30`、`--precision fp8` — 設定を指定して判定する
- `-m image,video` — 種類(カンマ区切り可)
- `--task edit` — 種類の中の用途: `gen`、`edit`、`i2v`、`lipsync`、`motion`、`voice`、`upscale`、`tool`、
  または `t2i` などのタスクコード(`genmodelfit types`)
- `--all-workflows` — モデルにまとめず、ワークフローごとに 1 行出す
- `--content sfw|all|uncensored` — 内容の印で絞る(下。既定は `sfw`)
- `--fit perfect|good|marginal|runnable|all` — その段階だけ・動くもの全部(`fit` の既定)・全部
- `--min-fit perfect|good|marginal|too_tight` — この段階以上だけ(recommend の既定は `marginal`。`--fit` とは一緒に使えない)
- `--sort type|score|vram|disk|name`(`fit`、`search`)— `type`(既定)は種類 → 段階 → 点数、`score` は種類をまたいだ 1 つの順位、`vram` / `disk` は小さい順
- `--include-unknown-license` — ライセンスの分からないファイルがあるものも出す(既定では隠す)
- `--license apache-2.0,mit` — すべてのファイルのライセンスがこの中にあるものだけ。不明は含めない
- `--comfyui-version 0.40.0` — アプリが使う ComfyUI の版(gate の判定用)

並べ方: `fit` と `search` は 1 つの表で、**種類**(image, video, music, sfx, tts, 3d)→ **段階**
(Perfect → Good → Marginal → `?` → Too tight)→ **点数**(Score。0〜100。段階・VRAM の余裕・自動おすすめの準備ができているか・
実測か・人気で決める。式は
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md))の順です。自動でおすすめに出せるものを
上に動かすことはしません(Verified の列を見る)。`recommend` は種類ごとの上位で、自動でおすすめに出せるものを先に
します。先頭に PC と、隠した件数とその理由(`uncensored`・`tools`・`license unknown`・`too tight` など)を出します。
各コマンドの `--help` に、前提・終了コード・例があります。

**Verified** の列は JSON の `auto` を表に出したものです(0.5.0 までは Auto という名前)。`yes`、または最初に落ちた gate の条件を出します。
JSON の項目名(`auto`・`held_by`)と gate の条件の id は変わりません。

例(`COLUMNS=100 genmodelfit fit --vram 24 --ram 128`。24 GB の GPU・128 GB のメモリを仮定。行は … で省略):

```
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 146 (91 hidden: uncensored 26, tools 34, license unknown 28, too tight 3; 92 more
    workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Perfect      90 SDXL1.0 Base             gen+3   fp16  8.0m  33% GPU     openrail++ license
image ● Perfect      84 MOSS-Image SFT           gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      83 FLUX.2 [Klein] 4B Base   edit    fp8   13.2  55% GPU     apache-2.0 yes
image ● Perfect      82 MOSS-Image Base          gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      81 Anima Base 1.0           gen+3   bf16   7.1  30% GPU     other      license
image ● Perfect      80 Anima Preview 3 Base     gen     bf16   7.0  29% GPU     other      license
…
video ● Perfect      92 Wan 2.1 1.3B T2V         gen     fp16 13.3m  55% GPU     apache-2.0 yes
video ● Good         67 Wan 2.2 5B TI2V          gen     fp16 15.7m  66% GPU~    apache-2.0 yes
video ● Good         63 Wan 2.1 ControlNet 1.3B  edit    bf16  17.1  71% GPU     apache-2.0 yes
video ● Marginal     44 Wan 2.2 14B I2V          i2v+2   fp8   21.0  88% GPU~    apache-2.0 yes
…
music ● Perfect      94 ACE-Step 1.5 Turbo       gen+2   bf16 10.8m  45% GPU     mit        yes
music ● Perfect      82 ACE Step v1 3.5B         gen+2   bf16   8.7  36% GPU     apache-2.0 yes
…
tts   ● Perfect      81 Qwen3-TTS-12…CustomVoice gen     bf16   5.0  21% GPU     apache-2.0 workflow
tts   ● Perfect      80 Chatterbox               gen     fp32   3.5  14% GPU     mit        workflow
…
3d    ● Perfect      89 HY 3D 2.1                gen     fp16  9.4m  39% GPU     other      license
3d    ● Perfect      88 HY 3D 2.0                gen     fp16  6.7m  28% GPU     other      license
3d    ● Perfect      79 HY 3D 2.0 MV             gen     fp16   6.8  29% GPU     other      license
3d    ● Perfect      77 HY 3D 2.0 MV Turbo       gen     fp16   6.8  29% GPU     other      license
Order: Type, Status, Score (--sort). Score 0-100. VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
…
```

Python:

```python
import genmodelfit as gmf

pc = gmf.system()
for f in gmf.recommend(modality="image", task="edit", limit=5, comfyui_version="0.40.0"):
    print(
        f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, f.estimate.vram_gb, f.estimate.basis
    )
    print([w["id"] for w in f.workflows])  # 同じモデルのほかのワークフロー
    f.to_dict()  # `recommend --json` の 1 件と同じ dict

# `genmodelfit fit` と同じ一覧(動くものだけ。種類 → 段階 → 点数)
rows = gmf.fit_list(sort="type")  # fit="good"、show_all=True、include_tools=True など CLI と同じ
gmf.search("flux", license=["apache-2.0", "mit"])

st = gmf.data_status()  # 一覧が自動おすすめに使えるほど新しいか
if st.expired:  # HEAD の確認が 30 日より古い → 自動おすすめが減る
    print("run: genmodelfit update --from-release", st.expires_at, st.auto_count)
```

## モデルを落とす

```
genmodelfit download "Z-Image-Turbo" --list                 # 落とすものの一覧(通信も書き込みもしない)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
```

- **落とすもの**: 1 つのワークフローの全部のファイル。モデル名(表の Model)なら、`recommend` と同じく、この PC で
  逃がさずに載る中で一番精度の高いワークフローを選びます。ID を指定すればそのまま使います。`--precision int8` で精度を、
  `--budget 15`(GB)で落とす量の上限を決められます(既にあるファイルは数えません)。
- **置き場所**: `--comfyui-dir` なら、テンプレートの置き場所(`models/diffusion_models`・`text_encoders`・`vae`・`loras`・
  `checkpoints` など。ファイル名の中のサブフォルダは保つ)に置きます。`--output-dir`(既定は genmodelfit のデータのフォルダの
  `models`)も同じ並びなので、ComfyUI の `extra_model_paths.yaml` に足せます。
- **先に同意**: 落とす前に、ファイルごとのライセンスと原文の URL を出し(利用制限の無い商用利用ができる許可リスト
  ([docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md))に無いものは印付き)、合計の大きさと空きを見せて `[y/N]` で尋ねます。`--yes` で省けます
  (`--json` で落とすときは必須)。使う前に各ファイルの条件を読んでください。
- **落とさないもの**: gated(ログインや利用申請が要る。genmodelfit はログイン情報やトークンを扱いません。モデルのページで申請して
  自分で落とす)、無検閲のモデル(`--content all` か `uncensored` を明示したときだけ)、ライセンス不明
  (`--include-unknown-license` のときだけ)、一覧に sha256・大きさの無いもの、Hugging Face・GitHub 以外の取得先。
- **安全**: 書きながら sha256 を計り、合わなければ消して失敗にします。途中は `<名前>.part` に書き、Ctrl-C や通信の失敗の
  あとで同じコマンドを実行すると HTTP Range で続きから落とします(相手が Range に応じなければ最初から。3 回まで)。
  既にあるファイルは飛ばす直前に sha256 を計り直し、同じ名前の別のファイルは上書きしません。同じファイルを
  2 つの download が同時に書かないよう `<名前>.lock` を置きます(止まったプロセスの残りは片付けます)。空きが「落とす量 + 1 GB」に足りなければ
  始めません。置き場所の外には書きません(`..`・絶対パス・ドライブ名・`\`・Windows の予約名(フォルダ名も)・制御文字・
  双方向文字・外へ出るシンボリックリンクやジャンクションを拒みます)。
- **通信**: 一覧の URL(匿名の HEAD で確かめたもの)だけを https で使います。転送先は下のホストだけです。
  2 通りのホストに読める URL(`\`・空白・制御文字・ホストの中の `%` や `@`・ASCII でないホスト)は、つなぐ前に拒みます
  (ホストは requests / urllib3 と同じ読み方で取ります)。使えるのは
  `huggingface.co` / `*.hf.co`(CDN・Xet)/ `github.com` / `objects.githubusercontent.com` /
  `release-assets.githubusercontent.com`。`~/.netrc` もトークンも使わず、PC の情報も送りません(User-Agent と
  Range だけ)。429・503 では `Retry-After` を待ちます。

Python:

```python
import genmodelfit as gmf

plan = gmf.plan_download("Z-Image-Turbo", comfyui_dir=r"C:\ComfyUI")  # 通信も書き込みもしない
for f in plan.files:  # status: download / resume / present / conflict / unavailable
    print(f.relpath, f.size_bytes, f.license, f.license_url, f.status)
print(plan.blocked, plan.warnings, plan.download_bytes, plan.disk_free_bytes)


def ask(plan) -> bool:  # ライセンス・URL・合計・空きを見せる。続けるなら True
    ...


def show(p):  # gmf.DownloadProgress
    print(p.phase, p.name, p.done_bytes, p.total_bytes)


result = gmf.download(plan, confirm=ask, progress=show)
# アプリが先に尋ねたなら gmf.download("Z-Image-Turbo", yes=True, comfyui_dir=...) でもよい
```

ライブラリは何も表示しません。`gmf.download` は `gmf.DownloadRefused`(ブロック・同意なし)と `gmf.DownloadError`
(通信・sha256 の不一致・空き)を投げ、`KeyboardInterrupt` はそのまま通して、`.part` を次の呼び出しのために残します。

## ハードウェアの検出

- 判定に使うのは nvidia-smi で確かめた NVIDIA の GPU だけです(v0.1)。その世代を動かす PyTorch の CUDA 版が無い
  (Kepler 以前)か、ドライバがどの版にも古すぎる(CUDA 12 の版は 525 以上、CUDA 13 の版は 580 以上、Blackwell は 570 以上)
  GPU は **CPU** として判定します。理由は `system` と `--json` に出します。
- ComfyUI が fp32 で計算する GPU(Maxwell・一部の Pascal・GTX 16xx・T1000 など)では、作業用のメモリと VAE が 2 倍になります。
  bf16 の無い GPU(Pascal・Volta・Turing)では、ComfyUI が fp16 で動かさないモデル系統(LTX・Qwen Image・Z-Image・
  HunyuanVideo・ACE-Step など)も fp32 で計算し、fp16 の無い VAE(SD / Flux KL・LTX)も同じです。
- 段階は全体の VRAM で決めます。今空いている VRAM が足りないときは、ほかのアプリを閉じるよう注記を出します(`vram_free_short`)。
- メモリが足りないときは、固定サイズのページファイル / スワップが「遅いが動く」範囲を決めます(システム管理なら メモリの 1.5 倍)。
- AMD・Intel の GPU は知らせる(Windows は WMI とレジストリ、Linux は `lspci`)だけで、判定には使いません。ホスト名・
  ユーザー名・GPU の UUID・PCI の id は集めません。細かい決まりと出典は
  [docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md)。

## 段階

llmfit と同じく、動かし方で使うメモリ(GPU なら VRAM、CPU ならメモリ)の使用率で段階を決めます。

| 段階 | 使用率 | 備考 |
|---|---|---|
| **Perfect** | 60% 以下 | 全部 GPU に載るときだけ |
| **Good** | 85% 以下 | 部品を順に載せ替えるとき(Mode `GPU~`)もここまで |
| **Marginal** | 98% 以下 | メモリへ逃がす(Mode `GPU+RAM`、遅い)・CPU・メモリが少し足りないときもここ |
| **Too tight** | 98% 超 | 動かない |

すべての見積もりに「**実測**」(VRAM の数字の後ろの `m`。実際に動かした値)か「**推定**」(式)かを付けます。
実測があれば実測を使います。ダウンロードの大きさが分からないモデルは判定しません(`?` と表示)。

## データ源

同梱の一覧は 329 件(image 163、video 115、TTS 22、music 16、3D 10、SFX 3)で、210 のモデルになります
(2026-10-21。毎週更新。「一覧の保守」を参照)。
llmfit と同じく、Hugging Face のモデルは全期間のダウンロード数が 10,000 以上のもの(伸びているものは 1,000 以上)だけを入れ、
マージモデルとテスト用の空のものは除きます([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md))。

- [ComfyUI のワークフローテンプレート](https://github.com/Comfy-Org/workflow_templates)(MIT): ワークフローが使うモデル、
  ダウンロード URL、ハッシュ、必要な ComfyUI の版、カスタムノード。API(クラウド)のテンプレートは除きます。
  索引の `size` は手書きの丸めた合計なので、genmodelfit はファイルの大きさの合計を使います。
- [Hugging Face Hub API](https://huggingface.co/docs/hub/api): ライセンス(`base_model` をたどる。再配布されたファイルは確認済みの
  上流の表で補う)、gated、ファイルの大きさ、sha256、コミュニティのモデルの発見。
- Civitai は使いません。規約が自動アクセスを制限し、集めたデータの再販売を禁じているためです。
- 手で足したモデル(`source: manual`、`src/genmodelfit/data/manual.json`): 人気順の収集で漏れるモデル
  (つくよみちゃんの piper-plus、Irodori-TTS、Stable Audio 3 Small の音楽)。各ファイルのライセンスは上流のカードから読み、
  `update` のたびに照らし合わせます。モデルごとの `languages`(プロンプト / 音声。ISO 639-1)、`runtime`
  (`comfyui` / `piper` / `irodori-tts`)、手書きの短い `description`(ja / en)は、アプリが一覧だけで画面を作れるように持ちます
  ([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md))。
- 実測: `src/genmodelfit/data/measurements.json`。送ってもらえると助かります([CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md))。

詳しくは [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)。genmodelfit が持つのはモデルの**情報だけ**で、モデル本体は持ちません。

## 内容の印

- `sfw` — 既定。
- `uncensored` — 安全フィルタを外したモデルや、成人向けの出力を想定したモデル
  (Hugging Face の `not-for-all-audiences` タグ、または uncensored / abliterated / nsfw などの名前)。一覧には載せ
  (同梱の一覧で 26 件)、表の `Censored` の列に `N` を付けます(`Y` = 検閲あり(内容が `sfw`)。この列は
  `--content all|uncensored` のときだけ出ます)。年齢制限があるので、`recommend` / `fit` / `search` と
  `gmf.recommend()` / `gmf.search()` は、**`--content all` か `--content uncensored` を指定したときだけ**出します
  (Python では `content="all"` / `"uncensored"`)。`info <id>` はいつもどおり使えます。根拠は
  `content_evidence` に残します。表示するかどうかは使う側のアプリが決めます。既定では自動おすすめに出しません。
- 未成年の性的表現を目的とするものは、一覧から**除外**します。

`--content all`(または `uncensored`)のときは、表の Score の右に `Censored` の列を出します(検閲ありは `Y`、無検閲は `N`)。
既定の `sfw` では、この列は出ません。`COLUMNS=100 genmodelfit fit --vram 24 --ram 128 --content all` の出力:

```
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 155 (82 hidden: tools 34, license unknown 43, size unknown 2, too tight 3; 92 more
    workflows merged into their models)
Type  Status      Score Censored Model                           Use     Prec  VRAM Mode    Verified
image ● Perfect      90 Y        SDXL1.0 Base                    gen+3   fp16  8.0m GPU     license
image ● Perfect      84 Y        MOSS-Image SFT                  gen+1   int8  13.8 GPU     yes
image ● Perfect      83 Y        FLUX.2 [Klein] 4B Base          edit    fp8   13.2 GPU     yes
…
image ● Perfect      73 N        Noct-Q-Uncensore…Image-2.1 Base gen     int8   8.5 GPU     license
image ● Perfect      70 N        LUSTIFY-SDXL-NSF…2-0-INPAINTING gen     fp16   8.2 GPU     license
…
Censored: N = safety filter removed / adult model
```

## 自動でおすすめに出す条件

機械で確かめられる条件を**すべて**満たすものだけが `auto: yes` です。すべてのファイルのライセンス(アップロードした人の表記でなく
上流までたどったもの)が商用利用できる許可リスト(apache-2.0・mit・bsd・cc0・unlicense。差し替え可)にあり、
gated が無く、カスタムノードの無い公式の ComfyUI ワークフローで、ComfyUI の版が対応していて、すべてのダウンロード URL が
30 日以内の匿名の HEAD に答えていて、sha256 と大きさが分かっていて、内容が `sfw` で、確認待ちが無い(条件 `reviewed`。
「一覧の保守」を参照)こと。満たさないものは、機械で読める理由付きで保留にします(JSON の `held_reasons`。例: `license`。
表の Verified の列には最初の 1 つを出します)。
同梱の一覧では 329 件のうち 99 件が auto です(2026-10-21)。ライセンスが確かめられないものは「不明」として保留にします。
アプリは使う ComfyUI の版を `--comfyui-version` で渡してください。
[docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md) を参照。

## 一覧の保守

llmfit と同じく、一覧は毎週作り直し、確かめたものだけを配ります。

1. 週に 1 回、オーナーの PC で実行役のスクリプトと Claude Code のスキル `genmodelfit-weekly` が動きます
   (Windows のタスク スケジューラ)。実行役が一覧を作り直し(`python -m genmodelfit.weekly`: テンプレート・Hugging Face・HEAD の確認・ファイルの先頭の見出し)、
   確かめ(スキーマ・auto の数・30 日の HEAD の期限が 7 日より先)、変化の分類(`diff.json`・`review_queue.json` の
   `classification`)を出します。Claude はファイルを書き換えず、コマンドも動かしません。`review_queue.json` の
   確認待ちのモデルを一次資料(Hugging Face・GitHub)で確かめ、判断の一覧の JSON を返すだけです(根拠の URL を付けた
   `approve`、`reject`、`hold`。安全上の理由の reject は `"safety": true`)。実行役のスクリプトがそれを
   `python -m genmodelfit.review apply decisions.json --db src/genmodelfit/data/db.json` で当て(厳しく確かめ、
   全部か無し)、コミットします。
2. 実行役のスクリプトがブランチを push し、Pull Request を作ります。**機械の確かめがすべて通ったときだけ**マージします:
   変わったファイルが `src/genmodelfit/data/db.json` だけ、2 つのファイルから計算し直した分類が安全
   (`python -m genmodelfit.weekly merge-check --base <main の db.json> --head <新しい db.json> --json` が終了コード 0。
   前から確認待ちのまま変わらないモデルはよい)、確認の印の status が変わっていない、CI が通った。
   どれかが通らなければ、オーナーが見てマージします。
3. マージされると `.github/workflows/data.yml` が main の `db.json` から確認待ちのモデルを除いて署名し、
   Release `data-latest` を置き換えます(0.5.x は `review` を知らないので、確認待ちのモデルは配りません)。
4. アプリは `genmodelfit update --from-release` で取り込みます(署名を確かめる)。

`.github/workflows/weekly-data.yml` は定期の実行をやめ、手動の予備(workflow_dispatch)として残しています。

```powershell
genmodelfit diff old_db.json new_db.json            # 2 つの一覧の差(ツール向けは --json)
python -m genmodelfit.weekly --dry-run --out weekly-out   # 同じ流れを手元で回す(書くのは weekly-out だけ)
python -m genmodelfit.weekly merge-check --base old_db.json --head new_db.json --json   # 0 = 安全 / 3 = 判断が要る
```

安全とみなすのは、許すものの表にある変化だけです: 人気の数(`downloads`・`downloads_all_time`・`likes`・
テンプレートの `usage`)、`updated`、HEAD の結果、ファイルが大きくなる、ライセンスが厳しくなる、消えたモデル、
自動おすすめから外れたモデル、`sources` の数、確認待ちのまま変わらないモデル。それ以外(新しいモデル、ファイルの
URL・sha256 の変化、用途・言語・説明・ライセンスのリンク・名前の変化、知らない項目、確認の印の変化など)は
判断が要ります。

変更の確認: `genmodelfit update` は、新しいモデル・ライセンスが緩くなったもの・必要なメモリが減ったもの・
新しく無検閲になったものに `review.status: "pending"` を付けます。pending のモデルは、一次資料の根拠を付けて
承認するまで、自動でおすすめに出しません(gate の条件 `reviewed`)。

```powershell
genmodelfit review list [--json]                                        # 確認待ちと、その理由
genmodelfit review approve <id> --evidence <https URL>... [--note TEXT] --db <path>
genmodelfit review reject <id> --note <text> --db <path> [--hide-name]
```

一覧を読むアプリは `entry.review is None or entry.review.status == "ok"` で絞れます。詳しくは
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)(「review」)と
[docs/release.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.md)。

## 実測を送る

もっと多くの GPU での実測があると、見積もりが良くなります。自分のフォークのクローンで:

```powershell
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 832x832 --submission data/contrib/
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json   # 共有する中身そのものと、プライバシーの確認リスト
```

そのファイルだけを足した Pull Request を出してください。**genmodelfit は何も送りません**。ファイルに入るのは GPU の名前・VRAM・
世代・ドライバ、ComfyUI の版、OS の種類、GiB に丸めたメモリ、結果です。ホスト名・ユーザー名・パス・GPU の UUID・
PCI アドレス・IP・正確な時刻は入りません(確認はパターンによるので、メモは自分でも読んでください)。CI は
`genmodelfit contrib validate` で確かめます。同じモデル・設定・GPU で 2 件以上の合う提出がそろうと、その条件の見積もりは
`basis_detail: "measured_community"` になります。コミュニティの値で上げられるのは必要量の見積もりだけで、下げることはありません。
手順: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)。

## 送るもの・送らないもの

- `recommend`・`fit`・`search`・`info`・`system`・`types`・`gate` と Python の関数(`gmf.recommend()`・`gmf.search()`・
  `gmf.info()`・`gmf.system()`・`gmf.data_status()`)は**ネットにつなぎません**。PC の情報は手元で読みます
  (`nvidia-smi`、psutil)。すべてのソケットをふさいだ試験で確かめています。
- `genmodelfit update --from-release` は、このプロジェクトの GitHub Release から `db.json` と `db.json.sig` を取るだけです。
- `genmodelfit update` は、GET / HEAD で公開の情報を読むだけです(GitHub の ComfyUI テンプレートと Release、Hugging Face Hub の API、
  モデルのダウンロード URL への HEAD と、本体のファイルの先頭の見出し(dtype・テンソルの形)だけの Range 要求。モデル本体は落としません)。
- `genmodelfit bench` は `--url` で指定した ComfyUI とだけ話します(ComfyUI に無い公式テンプレートは GitHub から取ります)。
- `genmodelfit download`(と `gmf.download()`)は、一覧のモデルの URL へ https の GET を送るだけです
  (転送先は Hugging Face・GitHub のホストだけ)。`--list` と `gmf.plan_download()` は通信しません。
- PC の情報(GPU の名前・VRAM・メモリ・CPU・空き)は `nvidia-smi` と psutil で手元で読むだけで、**送りません**。
  `bench` の実測は手元に保存するだけで、利用者が自分で Pull Request を出したときだけ共有されます
  ([CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md))。保存する実測に PC の名前・
  ユーザー名・パス・GPU の UUID は入りません。提出用のファイル(`bench --submission`・`contrib make`)も
  手元に書くだけで、`genmodelfit contrib` は通信しません。
- 利用状況の送信(テレメトリ)はありません。PC の情報・実測・プロンプトなどを送ることはありません。
  試験では PC の情報を目印の文字列に差し替え、`update`・`update --from-release`・`bench`・`download` の要求の URL・ヘッダ・本文に
  入っていないことを確かめています。User-Agent は `genmodelfit/<版> (+https://github.com/laxshmate/genmodelfit)` で、
  `~/.netrc` の資格情報は付けません(`HTTPS_PROXY` などのプロキシ・証明書の設定は使います)。

## 精度と限界

[docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md) を参照。要点:

- 見積もりの較正に使ったのは、1 種類の GPU(RTX 5080 16 GB)での実測 9 件だけです。1 件ずつ外して当てる方法で評価できたのは
  そのうち 4 件(画像・3D、最大誤差 6.4%、平均 4.6%)で、動画と音声はまだ評価できていません。
  `est.` の値は目安として見てください。
- ComfyUI の動的 VRAM は、空きがあればそのぶん多く使うので、大きい GPU で測ったピークは最低限の必要量ではありません。
- テンプレートには代わりのファイル(例: 1.3B と 14B の版)を並べているものがあります。genmodelfit は部品ごとに 1 つの版を数えるので、
  選んだ版と合わないことがあります。
- 同梱の一覧の HEAD の確認は 30 日で切れます。`genmodelfit update --from-release`(または `update --heads-only`)で更新してください。
  更新しないと自動おすすめは 0 件になります。使っている一覧が切れたかは `genmodelfit system` と `gmf.data_status()`
  (`expired`・`expires_at`)で分かります。
- v0.1 が数えるのは NVIDIA の GPU だけです。

## ライセンス

genmodelfit は [Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE)(MPL-2.0)で、同梱のデータ
(`db.json`・実測・較正)も同じです。MPL-2.0 はファイル単位のコピーレフトです。genmodelfit のファイルを直して配るときは、
そのファイルのソースを MPL-2.0 で公開してください。直さずに使う(import する)だけなら、ソース非公開のアプリを含む、
どんなライセンスの大きな作品にも組み込めます。アプリ(例: PyInstaller の exe)に genmodelfit を同梱して配るときは、
ライセンスの表示を残し、ソースの入手先(https://github.com/laxshmate/genmodelfit、MPL-2.0 §3.2)を利用者に知らせてください。
`certifi`(MPL-2.0)も同じです。
データの一部は [ComfyUI のワークフローテンプレート](https://github.com/Comfy-Org/workflow_templates)(MIT)と
Hugging Face Hub の API に由来します。出典は [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES) にあります。
モデルのライセンスは表示するだけで、利用の許諾はしません。各モデルのライセンスを確かめてください。

```
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```
