# genmodelfit

**Right-size image / video / audio / 3D generation models to your PC.**
画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。

genmodelfit is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork.
llmfit に着想を得た、生成モデル向けのツールです(llmfit の公式の派生ではありません)。

> Status: v0.1 in development (not on PyPI yet). / 状態: v0.1 を開発中(PyPI には未公開)。

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 12 of 89 (252 hidden: uncensored 39, tools 32, size unknown 5, too tight 152, same
    model 24)
== image (画像): top 2 of 45  (-m image for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      image_z_image_turbo_int8  gen+5   int8    66  7.6m  96% GPU~    11.3 apache-2.0 yes
● Marginal  image_flux2_k…b_distilled edit    fp8     41   4.5  56% GPU+RAM 11.6 apache-2.0 yes
== video (動画): top 2 of 5  (-m video for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      text_to_video_wan         gen     fp16    66  7.8m  97% GPU~     9.2 apache-2.0 yes
● Marginal  wan2.1_fun_control        edit    bf16    39   7.7  96% GPU+RAM 10.6 mit        yes
== music (音楽): top 2 of 10  (-m music for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Good      audio_ace_ste…_checkpoint gen+1   -       65  6.0m  76% GPU~     9.3 mit        yes
● Good      audio_ace_ste…m2m_editing edit+2  -       54   6.9  86% GPU~     7.2 apache-2.0 yes
== sfx (効果音): top 2 of 4  (-m sfx for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   hf:stabilitya…3-small-sfx gen     -       68   3.6  45% GPU      2.1 other      license
● Perfect   audio_stable_…dio_example gen+1   -       67   6.8  85% GPU      5.3 other      license
== tts (読み上げ): top 2 of 16  (-m tts for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   hf:ResembleAI/chatterbox  gen     -       80   3.5  44% GPU      2.0 mit        workflow
● Perfect   hf:Qwen/Qwen3…CustomVoice gen     -       80   3.2  40% GPU      1.7 apache-2.0 workflow
== 3d (3D): top 2 of 9  (-m 3d for more)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Perfect   3d_moge_persp…ive_to_mesh gen+1   fp16    92  2.1m  26% GPU      0.6 mit        size
● Perfect   3d_hunyuan3d_…ge_to_model gen     fp16    78  6.7m  84% GPU      4.6 other      license
Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <id>. Apps: --json (docs/schema.md).
Note: 39 uncensored model(s) hidden; add --content all to show them.
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
Recommended: 3 of 18 (323 hidden: other types 169, uncensored 30, tools 19, other uses 63, too
    tight 40, same model 2)
Status      Model                     Use     Prec Score  VRAM Mem% Mode    Disk License    Auto
● Marginal  image_flux2_k…b_distilled edit    fp8     41   4.5  56% GPU+RAM 11.6 apache-2.0 yes
● Marginal  image_flux2_k…dit_4b_base edit    fp8     41   4.5  56% GPU+RAM 11.5 apache-2.0 yes
● Marginal  image_z_image…_controlnet edit+5  bf16    41   7.5  93% GPU+RAM 22.2 apache-2.0 yes
Order: runs first, then Auto=yes, then Score (0-100). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Auto: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <id>. Apps: --json (docs/schema.md).
Note: 30 uncensored model(s) hidden; add --content all to show them.
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
# from source (PyPI release pending) / ソースから(PyPI は未公開)
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10+. Windows and Linux. GPUs: NVIDIA (CUDA) in v0.1; other PCs are judged as CPU-only.

## Usage / 使い方

| command | what it does / 内容 |
|---|---|
| `genmodelfit system` | detected GPU / RAM / disk / この PC の GPU・メモリ・空き |
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

- `--json` — machine-readable output for apps and agents ([docs/schema.md](docs/schema.md)) / アプリ向けの JSON
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
[docs/schema.md](docs/schema.md#score--点数)), then id. The first lines show this PC and how many models were hidden
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
```

## Fit levels / 段階

| level | meaning |
|---|---|
| **Perfect** | everything fits in VRAM with headroom / 余裕をもって全部 GPU に載る |
| **Good** | fits all at once, or when parts are loaded one at a time (Mode `GPU~`) / 全部、または部品を順に載せ替えれば載る |
| **Marginal** | runs by offloading weights to RAM (Mode `GPU+RAM`, slow), on CPU, or with slightly too little RAM / メインメモリへ逃がせば動く(遅い) |
| **Too tight** | does not run (VRAM or RAM) / 動かない |

Every estimate says whether it is **measured** (`m` after the VRAM figure, a real run) or **estimated** (formula).
Measured values win. Models whose download size is unknown are not judged (shown as `?`).
すべての見積もりに「実測」か「推定」かを付けます。実測があれば実測を使います。大きさが分からないモデルは判定しません。

## Data sources / データ源

The bundled list has 341 models (image 172, video 118, TTS 20, music 15, 3D 10, SFX 6).

- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT): which models a workflow
  uses, download URLs, hashes, required ComfyUI version, custom nodes. API (cloud) templates are excluded.
  The index's `size` field is a rounded, hand-written total, so genmodelfit uses the sum of the file sizes.
- [Hugging Face Hub API](https://huggingface.co/docs/hub/api): licenses (following `base_model` chains and a
  table of verified upstreams for repackaged files), gating, file sizes, sha256, and discovery of community models.
- Civitai is **not** used: its terms limit automated access and forbid reselling compiled data.
- Measurements: `src/genmodelfit/data/measurements.json`. Contributions welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)).

Details: [docs/sources.md](docs/sources.md). genmodelfit stores model **information only**; it never hosts weights.
genmodelfit が持つのはモデルの情報だけで、モデル本体は持ちません。

## Content marks / 内容の印

- `sfw` — default.
- `uncensored` — models whose safety filtering was removed or that are made for adult output
  (Hugging Face `not-for-all-audiences` tag, or names such as uncensored / abliterated / nsfw). They are in the list
  (39 in the bundled list) and marked `[U]` in tables, but `recommend` / `fit` / `search` and `gmf.recommend()` / `gmf.search()` show them
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
In the bundled list 83 of 341 models are auto. When a license cannot be verified it is treated as unknown
and held. See [docs/gate.md](docs/gate.md).

すべての条件を満たすものだけを自動でおすすめに出し、残りは理由付きで保留にします。ライセンスが確かめられない
ものは「不明」として保留にします。アプリは使う ComfyUI の版を `--comfyui-version` で渡してください。

## Accuracy and limits / 精度と限界

See [docs/accuracy.md](docs/accuracy.md). In short / 要点:

- Estimates are calibrated on only 9 real runs on one GPU (RTX 5080 16 GB). Leave-one-out error could be
  evaluated on 4 of them (image and 3D: max 6.4%, mean 4.6%); video and audio could not be evaluated yet.
  Treat `est.` values as rough guidance.
  実測は 1 種類の GPU で 9 件だけです。1 件ずつ外して当てる方法で評価できたのは 4 件(画像・3D、最大誤差 6.4%、
  平均 4.6%)で、動画と音声はまだ評価できていません。
- ComfyUI's dynamic VRAM uses more memory when it is available, so a peak measured on a large GPU is not the
  minimum requirement.
- Some templates list alternative files (e.g. 1.3B and 14B variants); genmodelfit counts one variant per part,
  which may not match the variant you pick.
- The HEAD checks in the bundled list expire after 30 days; run `genmodelfit update --heads-only` (or
  `update --from-release`) to refresh them, otherwise auto-recommendations drop to zero.
- v0.1 counts NVIDIA GPUs only.

## License

MIT. Upstream data keeps its own terms; model licenses are reported, not granted.
モデルのライセンスは表示するだけで、利用の許諾はしません。各モデルのライセンスを確かめてください。
