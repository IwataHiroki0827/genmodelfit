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
● Good      ACE-Step 1.5 Turbo        gen+2   -       65  6.0m  76% GPU~     9.3 mit        yes
● Marginal  ACE-Step 1.5XL Turbo      gen     bf16    40   5.2  65% GPU+RAM 18.5 mit        yes
== sfx (効果音): top 2 of 2  (-m sfx for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   stable-audio-3-small-sfx  gen     -       68   3.6  45% GPU      2.1 other      license
● Good      stable-audio-open-1.0     gen+1   -       54   4.9  61% GPU~     5.3 other      license
== tts (読み上げ): top 2 of 16  (-m tts for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   Chatterbox                gen     -       80   3.5  44% GPU      2.0 mit        workflow
● Perfect   Qwen3-TTS-12H…CustomVoice gen     -       80   3.2  40% GPU      1.7 apache-2.0 workflow
== 3d (3D): top 2 of 8  (-m 3d for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      HY 3D 2.0                 gen     fp16    64  6.7m  84% GPU      4.6 other      license
● Good      TripoSplat                gen     fp16    57   5.8  72% GPU      3.5 other      license
Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
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

```
pip install genmodelfit
```

From source / ソースから:

```
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10+. Windows and Linux. GPUs: NVIDIA (CUDA) in v0.1; other PCs are judged as CPU-only.

## Usage / 使い方

| command | what it does / 内容 |
|---|---|
| `genmodelfit system` | detected GPU / RAM / disk, and whether the model list is still fresh (Data) / この PC の GPU・メモリ・空きと、一覧の新しさ |
| `genmodelfit recommend [-m image,video] [--task edit] [-n 5] [--auto-only]` | best models for this PC: per type without `-m` / `--task` (utility tools such as depth/pose/upscale only with `--include-tools`) / おすすめ順。`-m`・`--task` が無ければ種類ごと(深度・ポーズ・拡大などの補助ツールは `--include-tools` のときだけ) |
| `genmodelfit fit [-m video] [--min-fit good]` | every model with its fit level, same order / すべての判定の表(同じ並べ方) |
| `genmodelfit search <text>` | models whose id or name contains the text, judged for this PC (exit 0 even with no match) / ID・名前の一部で探す |
| `genmodelfit types` | the six types, the uses inside each (`--task` names) and counts / 種類と、その中の用途と数 |
| `genmodelfit info <id>` | files, licenses, estimate, limits, gate checks, other workflows of the same model / 1 つのモデルの詳細 |
| `genmodelfit gate` | how many models can be auto-recommended, and why the rest are held / 自動でおすすめに出せる数と保留の理由 |
| `genmodelfit update` | rebuild the model list from upstream / 一覧を上流から作り直す |
| `genmodelfit update --from-release` | fetch the signed model list published by this project / 署名付きの配布の一覧を取り込む |
| `genmodelfit bench <template> --url URL` | measure VRAM / RAM / time on a running ComfyUI / 動いている ComfyUI で実測 |

Common options / 共通の指定:

- `--json` — machine-readable output for apps and agents ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)) / アプリ向けの JSON
- `--vram 12 --ram 32` — judge for another PC / 別の PC を想定して判定
- `--width 1024 --height 1024`, `--frames 81`, `--length 30`, `--precision fp8` — judge a specific setting / 設定を指定して判定
- `-m image,video` — types / 種類(カンマ区切り可)
- `--task edit` — uses inside a type: `gen`, `edit`, `i2v`, `lipsync`, `motion`, `voice`, `upscale`, `tool`, or task codes
  such as `t2i` (`genmodelfit types`) / 種類の中の用途
- `--all-workflows` — one row per workflow instead of one per model / モデルにまとめず、ワークフローごとに出す
- `--content sfw|all|uncensored` — filter by content mark (below; default `sfw`) / 内容の印で絞る(既定は `sfw`)
- `--min-fit perfect|good|marginal|too_tight` — only this level or better (recommend: `marginal` by default) / この段階以上だけ
- `--license apache-2.0,mit` — every file's license must be in the list; unknown never matches / すべてのファイルのライセンスがこの中にあるものだけ
- `--comfyui-version 0.40.0` — the ComfyUI version your app runs, for the gate / 使う ComfyUI の版(自動おすすめの判定用)

Order / 並べ方 (in each type): models that run, then auto-recommendable ones (`Auto yes`), then **Score** (0–100: fit level,
VRAM headroom, readiness for auto-recommendation, measured or not, popularity; formula in
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#score--点数)), then id. The first lines show this PC and how many models were hidden
and why (`uncensored`, `tools`, `too tight`, …). `--help` of each command lists preconditions, exit codes and examples.
動くもの → 自動でおすすめに出せるもの → 点数の高い順。先頭に PC と、隠した件数とその理由を出します。

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
gmf.search("flux", license=["apache-2.0", "mit"])

st = gmf.data_status()  # is the model list fresh enough for auto-recommendation? / 一覧の新しさ
if st.expired:  # HEAD checks older than 30 days → auto-recommendations drop / 期限切れ
    print("run: genmodelfit update --from-release", st.expires_at, st.auto_count)
```

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

## Privacy / 送るもの・送らないもの

- `recommend`, `fit`, `search`, `info`, `system`, `types`, `gate` and the Python API (`gmf.recommend()`,
  `gmf.search()`, `gmf.info()`, `gmf.system()`, `gmf.data_status()`) make **no network access**. Hardware is read
  locally (`nvidia-smi`, psutil). This is tested with every socket blocked.
- `genmodelfit update --from-release` only downloads `db.json` and `db.json.sig` from this project's GitHub release.
- `genmodelfit update` only reads public data with GET / HEAD: the ComfyUI workflow templates and releases on
  GitHub, the Hugging Face Hub API, and the model download URLs (HEAD only; no weights are downloaded).
- `genmodelfit bench` talks only to the ComfyUI URL you give with `--url` (plus the official template files on
  GitHub when ComfyUI does not have them).
- Hardware information (GPU name, VRAM, RAM, CPU, disk) is read locally with `nvidia-smi` and psutil and is
  **never sent**. Measurements from `bench` are saved only on your PC; they are shared only if you open a pull
  request yourself ([CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)).
  Saved measurements hold no host name, user name, paths or GPU UUID.
- No telemetry. No hardware information, measurements, prompts or other user content is ever sent anywhere.
  Tests replace the PC's details with marker strings and check that no request URL, header or body of `update`,
  `update --from-release` or `bench` contains them.
  Requests carry the User-Agent `genmodelfit/<version> (+https://github.com/laxshmate/genmodelfit)`, and never the
  credentials in your `~/.netrc` (proxy and certificate settings such as `HTTPS_PROXY` are still honored).

- `recommend`・`fit`・`search`・`info`・`system`・`types`・`gate` と Python の関数は**ネットにつなぎません**
  (PC の情報は手元で読みます。通信をふさいだ試験で確かめています)。
- `update --from-release` は、このプロジェクトの GitHub Release から `db.json` と署名だけを取ります。
- `update` は公開の情報を読むだけです(GitHub の ComfyUI テンプレート、Hugging Face Hub の API、モデルの URL への HEAD。
  モデル本体は落としません)。
- `bench` は `--url` で指定した ComfyUI とだけ話します(ComfyUI に無い公式テンプレートは GitHub から取ります)。
- PC の情報(GPU の名前・VRAM・メモリ・CPU・空き)は `nvidia-smi` と psutil で手元で読むだけで、**送りません**。
  `bench` の実測は手元に保存するだけで、利用者が自分で Pull Request を出したときだけ共有されます
  (保存する実測に PC の名前・ユーザー名・パス・GPU の UUID は入りません)。
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
