# genmodelfit

**Right-size image / video / audio / 3D generation models to your PC.**
画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。

genmodelfit is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork.
llmfit に着想を得た、生成モデル向けのツールです(llmfit の公式の派生ではありません)。

> Status: v0.1.0 (alpha). / 状態: v0.1.0(アルファ版)。

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 12 of 83 (215 hidden: uncensored 24, tools 34, size unknown 4, too tight 153; 25 more
    workflows merged into their models)
== image (画像): top 2 of 43  (-m image for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Marginal  Z-Image-Turbo             gen+5   int8    54  7.6m  96% GPU~    11.3 apache-2.0 yes
● Marginal  FLUX.2 [Klein] 4B Base    edit    fp8     41   4.5  56% GPU+RAM 11.5 apache-2.0 yes
== video (動画): top 2 of 5  (-m video for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Marginal  Wan 2.1 1.3B T2V          gen     fp16    50  7.8m  97% GPU~     9.2 apache-2.0 yes
● Marginal  Wan 2.1 ControlNet 1.3B   edit    bf16    39   7.7  96% GPU+RAM 10.6 mit        yes
== music (音楽): top 2 of 9  (-m music for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      ACE-Step 1.5 Turbo        gen+2   ?       65  6.0m  76% GPU~     9.3 mit        yes
● Marginal  ACE-Step 1.5XL Turbo      gen     bf16    40   5.2  65% GPU+RAM 18.5 mit        yes
== sfx (効果音): top 2 of 2  (-m sfx for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   stable-audio-3-small-sfx  gen     ?       68   3.6  45% GPU      2.1 other      license
● Good      stable-audio-open-1.0     gen+1   ?       54   4.9  61% GPU~     5.3 other      license
== tts (読み上げ): top 2 of 16  (-m tts for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   Chatterbox                gen     ?       80   3.5  44% GPU      2.0 mit        workflow
● Perfect   Qwen3-TTS-12H…CustomVoice gen     ?       80   3.2  40% GPU      1.7 apache-2.0 workflow
== 3d (3D): top 2 of 8  (-m 3d for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      HY 3D 2.0                 gen     fp16    64  6.7m  84% GPU      4.6 other      license
● Good      TripoSplat                gen     fp16    57   5.8  72% GPU      3.5 other      license
Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
Note: 24 uncensored model(s) hidden; add --content all to show them.
```

Without `-m` / `--task`, `recommend` shows each of the six types (image, video, music, sfx, tts, 3d). One row is
one **model**: workflows that use the same weights (same family and size) in another precision or for another use
are folded into it (`+5` = five more workflows; `--all-workflows` lists them), and the best precision that fits is
chosen (`Prec`; like llmfit's best quant). Narrow down inside a type with `--task`:
`-m` も `--task` も無ければ 6 つの種類ごとに出します。1 行は 1 つの**モデル**で、同じ重み(系統と大きさが同じ)を
別の精度・別の使い方で使うワークフローはまとめ(`+5` = ほかに 5 つ。`--all-workflows` で全部)、載る中で一番良い精度を
選びます。種類の中は `--task` で絞ります。

```
$ genmodelfit recommend --vram 8 --ram 32 -m image --task edit -n 3
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 3 of 18 (303 hidden: other types 162, uncensored 19, tools 19, other uses 63, too
    tight 40; 2 more workflows merged into their models)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Marginal  FLUX.2 [Klein] 4B         edit+1  fp8     41   4.5  56% GPU+RAM 11.6 apache-2.0 yes
● Marginal  FLUX.2 [Klein] 4B Base    edit    fp8     41   4.5  56% GPU+RAM 11.5 apache-2.0 yes
● Marginal  Z-Image-Turbo             edit+5  bf16    41   7.5  93% GPU+RAM 22.2 apache-2.0 yes
Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
Note: 19 uncensored model(s) hidden; add --content all to show them.
```

The default output is a table for people; add `--json` for apps and agents (llmfit prints JSON by default
for `recommend`; genmodelfit keeps the table so a first try is readable).
既定の出力は人が読む表です。アプリやエージェントは `--json` を付けてください(llmfit の recommend は既定が JSON ですが、
genmodelfit は初めて試す人のために表のままにしています)。

## Why / なぜ作るか

llmfit covers LLMs only ([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41)). For diffusion, video and
audio models the memory needed changes several times over with precision, resolution, frame count and
offloading to system RAM, so they need their own database and scoring.

生成モデルは、精度・解像度・コマ数・メインメモリへの逃がし方で、必要な VRAM が数倍変わります。
そのため LLM とは別のデータと判定が要ります。

## Install / 入れ方

Recommended: install it as a command-line tool with [pipx](https://pipx.pypa.io/) or
[uv](https://docs.astral.sh/uv/). They put `genmodelfit` on PATH for you (run the PATH command once, then open a new
terminal):

```
pipx install genmodelfit        # once: pipx ensurepath
uv tool install genmodelfit     # once: uv tool update-shell
```

Or with pip:

```
pip install genmodelfit
```

**`python -m genmodelfit …` always works**, whatever PATH says (e.g. `python -m genmodelfit recommend`; on Windows
also `py -m genmodelfit …`). Use it when `genmodelfit` is "not recognized" / "command not found".

If pip printed *"WARNING: The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH"*, the command
was installed but that folder is not on PATH. Common cases:

- Python from python.org installed without "Add python.exe to PATH" → the `Scripts` folder of that Python is not on PATH.
- pip fell back to a user install (no write access to the system Python) → the command is in
  `%APPDATA%\Python\Python3XX\Scripts` (Windows) or `~/.local/bin` (Linux), which is often not on PATH.
- Microsoft Store Python → its `Scripts` folder lives deep under `%LOCALAPPDATA%\Packages\…` and is not on PATH.
- Installed into a virtual environment that is not activated → activate it (`.venv\Scripts\activate` /
  `source .venv/bin/activate`) or run `.venv\Scripts\genmodelfit`.

Fix: add the folder named in the warning to PATH (or use pipx / uv above, or `python -m genmodelfit`).
**A terminal opened before PATH changed does not see the change: open a new one.**

おすすめは pipx か uv でコマンドとして入れる方法です(PATH を通すコマンドを 1 回だけ実行し、端末を開き直す)。
pip で入れた場合も、**`python -m genmodelfit …` は PATH に関係なく必ず動きます**(Windows は `py -m genmodelfit …` も可)。
pip が「The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH」と出したら、コマンドは入っていますが、
その場所に PATH が通っていません。よくあるのは、python.org の Python を「Add python.exe to PATH」なしで入れた、
書き込めないため利用者ごとの場所(Windows は `%APPDATA%\Python\Python3XX\Scripts`、Linux は `~/.local/bin`)に入った、
Microsoft Store の Python、仮想環境を有効にしていない、の 4 つです。警告に出た場所を PATH に足すか、pipx / uv か
`python -m genmodelfit` を使ってください。**PATH を変える前に開いた端末には反映されないので、新しい端末を開いてください。**

From source / ソースから:

```
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10+. Windows and Linux. GPUs: NVIDIA (CUDA) in v0.1; other PCs are judged as CPU-only.

## Usage / 使い方

| command | what it does / 内容 |
|---|---|
| `genmodelfit system [--all-gpus]` | detected GPU (generation, driver, PyTorch builds, ComfyUI precision), RAM, page file, disk, and whether the model list is still fresh (Data); `--all-gpus` also lists non-NVIDIA adapters / この PC の GPU(世代・ドライバ・精度)・メモリ・ページファイル・空きと、一覧の新しさ |
| `genmodelfit recommend [-m image,video] [--task edit] [-n 5] [--auto-only]` | best models for this PC: per type without `-m` / `--task` (utility tools such as depth/pose/upscale only with `--include-tools`) / おすすめ順。`-m`・`--task` が無ければ種類ごと(深度・ポーズ・拡大などの補助ツールは `--include-tools` のときだけ) |
| `genmodelfit fit [-m video] [--fit good] [--sort score] [--all]` | the models that run on this PC (Perfect / Good / Marginal) in one table: by type, then fit level, then Score; `--all` also shows tools, models that do not run, unjudged (`?`) and unknown-license ones / この PC で動くモデルの表(種類 → 段階 → 点数の順)。`--all` で全部 |
| `genmodelfit search <text>` | models whose id or name contains the text, judged for this PC, every fit level, same order as `fit` (exit 0 even with no match) / ID・名前の一部で探す |
| `genmodelfit types` | the six types, the uses inside each (`--task` names) and counts / 種類と、その中の用途と数 |
| `genmodelfit info <id>` | files, licenses, estimate, limits, gate checks, other workflows of the same model / 1 つのモデルの詳細 |
| `genmodelfit gate` | how many models can be auto-recommended, and why the rest are held / 自動でおすすめに出せる数と保留の理由 |
| `genmodelfit update` | rebuild the model list from upstream / 一覧を上流から作り直す |
| `genmodelfit update --from-release` | fetch the signed model list published by this project / 署名付きの配布の一覧を取り込む |
| `genmodelfit download <model> [--comfyui-dir DIR] [--list]` | download all files of the model for this PC (precision chosen like `recommend`), after showing licenses, sizes and free disk and asking you; sha256-checked, resumable ([below](#download--モデルを落とす)) / この PC に合う精度でモデルのファイルを落とす(ライセンス・大きさ・空きを見せて同意を取る。sha256 で確かめ、続きから再開できる) |
| `genmodelfit bench <template> --url URL` | measure VRAM / RAM / time on a running ComfyUI / 動いている ComfyUI で実測 |

Common options / 共通の指定:

- `--json` — machine-readable output for apps and agents ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)) / アプリ向けの JSON
- `--vram 12 --ram 32` — judge with other VRAM / RAM sizes; the GPU generation of this PC is kept when it has a usable NVIDIA GPU ([docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md)) / VRAM・RAM の量を変えて判定(この PC に使える NVIDIA の GPU があれば、その世代は引き継ぐ)
- `--width 1024 --height 1024`, `--frames 81`, `--length 30`, `--precision fp8` — judge a specific setting / 設定を指定して判定
- `-m image,video` — types / 種類(カンマ区切り可)
- `--task edit` — uses inside a type: `gen`, `edit`, `i2v`, `lipsync`, `motion`, `voice`, `upscale`, `tool`, or task codes
  such as `t2i` (`genmodelfit types`) / 種類の中の用途
- `--all-workflows` — one row per workflow instead of one per model / モデルにまとめず、ワークフローごとに出す
- `--content sfw|all|uncensored` — filter by content mark (below; default `sfw`) / 内容の印で絞る(既定は `sfw`)
- `--fit perfect|good|marginal|runnable|all` — only that level, every level that runs (`fit` default) or everything / その段階だけ・動くもの全部(`fit` の既定)・全部
- `--min-fit perfect|good|marginal|too_tight` — only this level or better (recommend: `marginal` by default; not together with `--fit`) / この段階以上だけ
- `--sort type|score|vram|disk|name` (`fit`, `search`) — `type` (default): type, then fit level, then Score; `score`: one ranking across types; `vram` / `disk`: smallest first / 並べ方
- `--include-unknown-license` — also show models with a file whose license is unknown (hidden by default) / ライセンスの分からないものも出す
- `--license apache-2.0,mit` — every file's license must be in the list; unknown never matches / すべてのファイルのライセンスがこの中にあるものだけ
- `--comfyui-version 0.40.0` — the ComfyUI version your app runs, for the gate / 使う ComfyUI の版(自動おすすめの判定用)

Order / 並べ方: `fit` and `search` show one table, **by type** (image, video, music, sfx, tts, 3d), then **fit level**
(Perfect → Good → Marginal → `?` → Too tight), then **Score** (0–100: fit level, VRAM headroom, readiness for
auto-recommendation, measured or not, popularity; formula in
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#score--点数)); auto-recommendable
models are not moved up (see the Auto column). `recommend` shows the top models of each type, with auto-recommendable
ones first. The first lines show this PC and how many models were hidden and why (`uncensored`, `tools`,
`license unknown`, `too tight`, …). `--help` of each command lists preconditions, exit codes and examples.
`fit`・`search` は 1 つの表で、種類 → 段階 → 点数の順。`recommend` は種類ごとの上位で、自動でおすすめに出せるものを先に
します。先頭に PC と、隠した件数とその理由を出します。

Example / 例 (`COLUMNS=100 genmodelfit fit --vram 24 --ram 128`, 24 GB GPU / 128 GB RAM assumed; rows cut with …):

```
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 141 (93 hidden: uncensored 24, tools 34, license unknown 32, too tight 3; 89 more
    workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Auto
image ● Perfect      91 SDXL1.0 Base             gen+3   fp16  8.0m  33% GPU     openrail++ license
image ● Perfect      84 FLUX.2 [Klein] 4B Base   edit    fp8   13.2  55% GPU     apache-2.0 yes
image ● Perfect      80 Anima Preview 3 Base     gen     bf16   7.0  29% GPU     other      license
image ● Perfect      80 SDXL Turbo 1.0           gen     fp16   8.2  34% GPU     other      license
image ● Perfect      79 NetaYume Lumina          gen     bf16  11.6  48% GPU     gemma      license
image ● Perfect      79 PixelDiT 1300M           gen     bf16   9.0  38% GPU     other      license
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
    print(
        [w["id"] for w in f.workflows]
    )  # other workflows of the same model / 同じモデルのほかのワークフロー
    f.to_dict()  # the same dict as one item of `recommend --json` / `--json` の 1 件と同じ

# the same list as `genmodelfit fit` (runnable only, by type → level → score) / `fit` と同じ一覧
rows = gmf.fit_list(sort="type")  # fit="good", show_all=True, include_tools=True, … like the CLI
gmf.search("flux", license=["apache-2.0", "mit"])

st = gmf.data_status()  # is the model list fresh enough for auto-recommendation? / 一覧の新しさ
if st.expired:  # HEAD checks older than 30 days → auto-recommendations drop / 期限切れ
    print("run: genmodelfit update --from-release", st.expires_at, st.auto_count)
```

## Download / モデルを落とす

```
genmodelfit download "Z-Image-Turbo" --list                 # what would be downloaded (no network, writes nothing)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
```

- **Which files**: all files of one workflow. A model name (as in the tables) picks the workflow and precision like
  `recommend`: the highest precision that runs on this PC without offloading. A workflow id is used as is.
  `--precision int8` picks a precision; `--budget 15` (GB) caps the download (files already there are not counted).
- **Where**: `--comfyui-dir` puts each file in the ComfyUI folder the template names
  (`models/diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, …; subfolders in the file name are kept).
  `--output-dir` (default: genmodelfit's user data folder, `models`) uses the same layout, so it can be added to
  ComfyUI's `extra_model_paths.yaml`.
- **Consent first**: before anything is downloaded you see every file with its license and the license URL, licenses
  that are not on the allowlist (commercial use without use restrictions; [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md))
  are marked, then the total size and the free disk, and you are asked `[y/N]`. `--yes` skips the question
  (required with `--json`). Read the terms of each file before you use the model.
- **Not downloaded**: gated files (genmodelfit never handles logins or tokens; request access on the model page and
  download them yourself), uncensored models unless you pass `--content all` (or `uncensored`), files whose license is
  unknown unless you pass `--include-unknown-license`, files without a sha256 or size in the list, hosts other than
  Hugging Face / GitHub.
- **Safety**: every file is checked with the sha256 in the list while it is written; a mismatch deletes it and fails.
  A download goes to `<file>.part` first; after Ctrl-C or a network error run the same command again and it resumes
  with an HTTP Range request (if the server ignores Range it starts over, at most 3 times). Files already there are
  hashed again right before they are skipped; a different file with the same name is never overwritten. A
  `<file>.lock` next to it stops two downloads of the same file at once (a lock left by a process that is gone is
  cleaned up). It does not start unless the free disk is
  the download + 1 GB. Nothing is written outside the destination: names with `..`, absolute paths, drive letters,
  `\`, reserved Windows names (also as folder names), control / bidi characters, and symbolic links / junctions
  that lead outside are refused.
- **Network**: only the URLs in the list (checked anonymously with HEAD), https only. Every redirect must stay on
  the hosts below; a URL that could be read as two different hosts (`\`, spaces, control characters, `%` or `@`
  in the host, non-ASCII hosts) is refused before any connection, and the host is read the way requests / urllib3
  read it. Allowed:
  `huggingface.co` / `*.hf.co` (CDN, Xet) / `github.com` / `objects.githubusercontent.com` /
  `release-assets.githubusercontent.com`. No `~/.netrc` credentials, no tokens, nothing about this PC (User-Agent and
  Range only). Waits for `Retry-After` on 429 / 503.

- **落とすもの**: 1 つのワークフローの全部のファイル。モデル名(表の Model)なら、`recommend` と同じく、この PC で
  逃がさずに載る中で一番精度の高いワークフローを選びます。ID を指定すればそのまま。`--precision` で精度を、
  `--budget`(GB)で落とす量の上限を決められます(既にあるファイルは数えません)。
- **置き場所**: `--comfyui-dir` なら、テンプレートの置き場所(`models/diffusion_models`・`text_encoders`・`vae`・`loras`・
  `checkpoints` など)に置きます。`--output-dir`(既定は genmodelfit のデータのフォルダの `models`)も同じ並びです。
- **先に同意**: 落とす前に、ファイルごとのライセンスと原文の URL(許可リストに無いものは印付き)・合計の大きさ・空きを
  出して `[y/N]` で尋ねます。`--yes` で省けます(`--json` で落とすときは必須)。使う前に各ファイルの条件を読んでください。
- **落とさないもの**: gated(ログインや利用申請が要る。genmodelfit はログイン情報を扱いません)、無検閲のモデル
  (`--content all` か `uncensored` を明示したときだけ)、ライセンス不明(`--include-unknown-license` のときだけ)、
  一覧に sha256・大きさの無いもの、Hugging Face・GitHub 以外の取得先。
- **安全**: 書きながら sha256 を計り、合わなければ消して失敗にします。途中は `<名前>.part` に書き、Ctrl-C や通信の失敗の
  あとで同じコマンドを実行すると HTTP Range で続きから落とします(相手が Range に応じなければ最初から。3 回まで)。
  既にあるファイルは飛ばす直前に sha256 を計り直し、同じ名前の別のファイルは上書きしません。同じファイルを
  2 つの download が同時に書かないよう `<名前>.lock` を置きます(止まったプロセスの残りは片付けます)。空きが「落とす量 + 1 GB」に足りなければ
  始めません。置き場所の外には書きません(`..`・絶対パス・ドライブ名・`\`・Windows の予約名・外へ出るリンクを拒みます)。
- **通信**: 一覧の URL(匿名の HEAD で確かめたもの)だけを https で使い、転送先は Hugging Face(CDN・Xet を含む)と
  GitHub の決めたホストだけ。2 通りのホストに読める URL(`\`・空白・制御文字・ホストの中の `%` や `@`・ASCII でない
  ホスト)は、つなぐ前に拒みます。`~/.netrc` もトークンも使わず、PC の情報も送りません。

Python:

```python
import genmodelfit as gmf

plan = gmf.plan_download("Z-Image-Turbo", comfyui_dir=r"C:\ComfyUI")  # no network, writes nothing
for f in plan.files:  # status: download / resume / present / conflict / unavailable
    print(f.relpath, f.size_bytes, f.license, f.license_url, f.status)
print(plan.blocked, plan.warnings, plan.download_bytes, plan.disk_free_bytes)


def ask(plan) -> bool:  # show licenses, URLs, total and free disk; True to go on
    ...


def show(p):  # gmf.DownloadProgress
    print(p.phase, p.name, p.done_bytes, p.total_bytes)


result = gmf.download(plan, confirm=ask, progress=show)
# or gmf.download("Z-Image-Turbo", yes=True, comfyui_dir=...) when the app already asked
```

The library prints nothing. `gmf.download` raises `gmf.DownloadRefused` (blocked, no consent), `gmf.DownloadError`
(network, sha256 mismatch, disk) and lets `KeyboardInterrupt` through, keeping the `.part` file for the next call.

## Hardware / ハードウェアの検出

- Only NVIDIA GPUs confirmed by `nvidia-smi` are used for judging (v0.1). A GPU is judged as **CPU** when no PyTorch
  CUDA build supports its generation (Kepler and older) or its driver is too old for all of them (CUDA 12 builds need
  525+, CUDA 13 builds need 580+, Blackwell needs 570+). The reason is shown in `system` and `--json`.
- On GPUs where ComfyUI computes in fp32 (Maxwell, some Pascal, GTX 16xx, T1000 …) working memory and the VAE are doubled.
  On GPUs without bf16 (Pascal, Volta, Turing), model families that ComfyUI never runs in fp16 (LTX, Qwen Image, Z-Image,
  HunyuanVideo, ACE-Step …) also compute in fp32, and so do VAEs without fp16 (SD / Flux KL, LTX).
- The level uses total VRAM; when the VRAM free right now is not enough, a note says to close other apps (`vram_free_short`).
- When the RAM is short, a fixed-size page file / swap sets how far "slow but runs" goes (system-managed: 1.5× RAM).
- AMD / Intel GPUs are reported (Windows: WMI + registry, Linux: `lspci`) but never used for judging. Host name,
  user name, GPU UUID and PCI ids are not collected. Details and sources: [docs/hardware.md](docs/hardware.md).

判定に使うのは nvidia-smi で確かめた NVIDIA の GPU だけ。今の PyTorch が動かない世代(Kepler 以前)・古いドライバは CPU として判定し、
理由を出します。AMD・Intel の GPU は知らせるだけです。出典と細かい決まりは [docs/hardware.md](docs/hardware.md)。

## Fit levels / 段階

Like llmfit, the level comes from the utilization of the memory the run mode uses (VRAM on the GPU, RAM on CPU):
llmfit と同じく、動かし方で使うメモリの使用率で決めます。

| level | utilization / 使用率 | note |
|---|---|---|
| **Perfect** | ≤ 60% | all on the GPU only / 全部 GPU に載るときだけ |
| **Good** | ≤ 85% | also when parts are swapped in turn (Mode `GPU~`) / 部品を順に載せ替えるときもここまで |
| **Marginal** | ≤ 98% | also offload to RAM (Mode `GPU+RAM`, slow), CPU, or slightly too little RAM / 逃がす・CPU・メモリが少し足りない |
| **Too tight** | > 98% | does not run / 動かない |

Every estimate says whether it is **measured** (`m` after the VRAM figure, a real run) or **estimated** (formula).
Measured values win. Models whose download size is unknown are not judged (shown as `?`).
すべての見積もりに「実測」か「推定」かを付けます。実測があれば実測を使います。大きさが分からないモデルは判定しません。

## Data sources / データ源

The bundled list has 323 entries (image 161, video 114, TTS 20, music 15, 3D 10, SFX 3) that make 204 models.
Like llmfit, Hugging Face models are included only with at least 10,000 all-time downloads (1,000 for trending
ones), and merges and test stubs are skipped ([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)).
llmfit と同じく、Hugging Face のモデルは全期間のダウンロード数が 10,000 以上のもの(伸びているものは 1,000 以上)だけを入れます。

- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT): which models a workflow
  uses, download URLs, hashes, required ComfyUI version, custom nodes. API (cloud) templates are excluded.
  The index's `size` field is a rounded, hand-written total, so genmodelfit uses the sum of the file sizes.
- [Hugging Face Hub API](https://huggingface.co/docs/hub/api): licenses (following `base_model` chains and a
  table of verified upstreams for repackaged files), gating, file sizes, sha256, and discovery of community models.
- Civitai is **not** used: its terms limit automated access and forbid reselling compiled data.
- Measurements: `src/genmodelfit/data/measurements.json`. Contributions welcome (see [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)).

Details: [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md). genmodelfit stores model **information only**; it never hosts weights.
genmodelfit が持つのはモデルの情報だけで、モデル本体は持ちません。

## Content marks / 内容の印

- `sfw` — default.
- `uncensored` — models whose safety filtering was removed or that are made for adult output
  (Hugging Face `not-for-all-audiences` tag, or names such as uncensored / abliterated / nsfw). They are in the list
  (24 in the bundled list) and marked `[U]` in tables, but `recommend` / `fit` / `search` and `gmf.recommend()` / `gmf.search()` show them
  **only with `--content all` or `--content uncensored`** (`content="all"` / `"uncensored"` in Python).
  `info <id>` works for them as usual. The evidence is kept in
  `content_evidence`. Whether to show them is up to the app using genmodelfit; they are never auto-recommended
  by default.
- Models aimed at sexual content involving minors are **excluded** from the list entirely.

`uncensored` は、安全フィルタを外したモデルや成人向けの出力を想定したモデルです。一覧には載せ、印を付けて
います。年齢制限があるので、`recommend` / `fit` / `gmf.recommend()` の既定(`sfw`)では出さず、
`--content all` か `--content uncensored` を指定したときだけ出します(`info <id>` はいつもどおり使えます)。表示するかどうかは使う側のアプリが決めます(既定では自動おすすめに出しません)。
未成年の性的表現を目的とするものは一覧から除外します。

## Auto-recommendation gate / 自動でおすすめに出す条件

A model is `auto: yes` only if **all** machine-checkable conditions hold: every file's license (traced to the
upstream, not the uploader's label) is on a commercial-use allowlist (apache-2.0, mit, bsd, cc0, unlicense;
replaceable), nothing is gated, an official ComfyUI workflow without custom nodes, the ComfyUI version is
supported, every download URL answered an anonymous HEAD within the last 30 days, sha256 is known, the size is
known, and the content is `sfw`. Otherwise it is held with machine-readable reasons (`held_reasons` in JSON, e.g. `license`; the table's Auto column shows the first one).
In the bundled list 83 of 323 entries are auto. When a license cannot be verified it is treated as unknown
and held. See [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md).

すべての条件を満たすものだけを自動でおすすめに出し、残りは理由付きで保留にします。ライセンスが確かめられない
ものは「不明」として保留にします。アプリは使う ComfyUI の版を `--comfyui-version` で渡してください。

## Contributing measurements / 実測を送る

Measured runs on more GPUs make the estimates better. In a clone of your fork:

```powershell
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 832x832 --submission data/contrib/
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json   # exactly what will be shared + privacy checklist
```

Then open a pull request that adds only that file. **genmodelfit sends nothing**; the file holds the GPU name / VRAM /
generation / driver, ComfyUI version, OS family, RAM rounded to GiB and the results — never host or user names, paths,
GPU UUID, PCI address, IP or exact times (the checks are pattern based, so read the note yourself). CI checks it with
`genmodelfit contrib validate`. Once two or more consistent submissions exist for the same model, settings and GPU,
estimates for that setup become `basis_detail: "measured_community"`; community values can only raise the estimated
needs, never lower them. Steps: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md).

自分のフォークで `bench --submission data/contrib/` を付けて測り、`contrib show` で中身を確かめてから、そのファイルだけを
Pull Request で出してください。genmodelfit は何も送りません(PC の名前・ユーザー名・パス・GPU の UUID などは入りません)。

## Privacy / 送るもの・送らないもの

- `recommend`, `fit`, `search`, `info`, `system`, `types`, `gate` and the Python API (`gmf.recommend()`,
  `gmf.search()`, `gmf.info()`, `gmf.system()`, `gmf.data_status()`) make **no network access**. Hardware is read
  locally (`nvidia-smi`, psutil). This is tested with every socket blocked.
- `genmodelfit update --from-release` only downloads `db.json` and `db.json.sig` from this project's GitHub release.
- `genmodelfit update` only reads public data with GET / HEAD: the ComfyUI workflow templates and releases on
  GitHub, the Hugging Face Hub API, and the model download URLs (HEAD, plus a ranged GET of the first bytes of the
  main weight files to read their header — dtype and tensor shapes; no weights are downloaded).
- `genmodelfit bench` talks only to the ComfyUI URL you give with `--url` (plus the official template files on
  GitHub when ComfyUI does not have them).
- `genmodelfit download` (and `gmf.download()`) only GETs the model URLs in the list, over https, following
  redirects only to Hugging Face / GitHub hosts; `--list` and `gmf.plan_download()` make no network access.
- Hardware information (GPU name, VRAM, RAM, CPU, disk) is read locally with `nvidia-smi` and psutil and is
  **never sent**. Measurements from `bench` are saved only on your PC; they are shared only if you open a pull
  request yourself ([CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)).
  Saved measurements hold no host name, user name, paths or GPU UUID. Submission files (`bench --submission`,
  `contrib make`) are written locally only; `genmodelfit contrib` never touches the network.
- No telemetry. No hardware information, measurements, prompts or other user content is ever sent anywhere.
  Tests replace the PC's details with marker strings and check that no request URL, header or body of `update`,
  `update --from-release`, `bench` or `download` contains them.
  Requests carry the User-Agent `genmodelfit/<version> (+https://github.com/laxshmate/genmodelfit)`, and never the
  credentials in your `~/.netrc` (proxy and certificate settings such as `HTTPS_PROXY` are still honored).

- `recommend`・`fit`・`search`・`info`・`system`・`types`・`gate` と Python の関数は**ネットにつなぎません**
  (PC の情報は手元で読みます。通信をふさいだ試験で確かめています)。
- `update --from-release` は、このプロジェクトの GitHub Release から `db.json` と署名だけを取ります。
- `update` は公開の情報を読むだけです(GitHub の ComfyUI テンプレート、Hugging Face Hub の API、モデルの URL への HEAD と、
  本体のファイルの先頭の見出し(dtype・テンソルの形)だけの Range 要求。モデル本体は落としません)。
- `bench` は `--url` で指定した ComfyUI とだけ話します(ComfyUI に無い公式テンプレートは GitHub から取ります)。
- `download` は一覧のモデルの URL へ https の GET を送るだけです(転送先は Hugging Face・GitHub の決めたホストだけ)。
  `--list` と `gmf.plan_download()` は通信しません。
- PC の情報(GPU の名前・VRAM・メモリ・CPU・空き)は `nvidia-smi` と psutil で手元で読むだけで、**送りません**。
  `bench` の実測は手元に保存するだけで、利用者が自分で Pull Request を出したときだけ共有されます
  (保存する実測に PC の名前・ユーザー名・パス・GPU の UUID は入りません)。提出用の JSON(`bench --submission`・`contrib make`)も
  手元に書くだけで、`genmodelfit contrib` は通信しません。
- 利用状況の送信(テレメトリ)はありません。PC の情報・実測・プロンプトなどを送ることはありません。User-Agent は
  `genmodelfit/<版>` で、`~/.netrc` の資格情報は付けません(プロキシ・証明書の設定は使います)。

## Accuracy and limits / 精度と限界

See [docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md). In short / 要点:

- Estimates are calibrated on only 9 real runs on one GPU (RTX 5080 16 GB). Leave-one-out error could be
  evaluated on 4 of them (image and 3D: max 6.4%, mean 4.6%); video and audio could not be evaluated yet.
  Treat `est.` values as rough guidance.
  実測は 1 種類の GPU で 9 件だけです。1 件ずつ外して当てる方法で評価できたのは 4 件(画像・3D、最大誤差 6.4%、
  平均 4.6%)で、動画と音声はまだ評価できていません。
- ComfyUI's dynamic VRAM uses more memory when it is available, so a peak measured on a large GPU is not the
  minimum requirement.
- Some templates list alternative files (e.g. 1.3B and 14B variants); genmodelfit counts one variant per part,
  which may not match the variant you pick.
- The HEAD checks in the bundled list expire after 30 days; run `genmodelfit update --from-release` (or
  `update --heads-only`) to refresh them, otherwise auto-recommendations drop to zero. `genmodelfit system` and
  `gmf.data_status()` tell whether the list in use has expired (`expired`, `expires_at`).
  同梱の一覧の HEAD の確認は 30 日で切れます。切れたかは `genmodelfit system` と `gmf.data_status()` で分かります。
- v0.1 counts NVIDIA GPUs only.

## License / ライセンス

genmodelfit is licensed under the [Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) (MPL-2.0), and so is the bundled data
(`db.json`, measurements, calibration). MPL-2.0 is a file-level copyleft: if you distribute modified genmodelfit
files, you must make those files' source available under MPL-2.0; you may use genmodelfit unmodified (or import it)
in a larger work under any license, including closed-source apps. When you distribute genmodelfit inside your
app (for example in a PyInstaller exe), keep its license notices and tell your users where its source code is
available (https://github.com/laxshmate/genmodelfit, MPL-2.0 §3.2); the same applies to `certifi` (MPL-2.0).
アプリに同梱して配るときは、ライセンスの表示を残し、genmodelfit(と certifi)のソースの入手先を利用者に知らせてください。
Parts of the data come from [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT) and
the Hugging Face Hub API; see [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES). Model licenses are reported, not granted.

genmodelfit と同梱のデータは MPL-2.0 です。genmodelfit のファイルを直して配るときは、そのファイルのソースを
MPL-2.0 で公開してください。直さずに使う(import する)だけなら、ソース非公開のアプリにも組み込めます
(ライセンスの表示は残してください)。データの出典の表示は THIRD_PARTY_NOTICES にあります。
モデルのライセンスは表示するだけで、利用の許諾はしません。各モデルのライセンスを確かめてください。

```
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```
