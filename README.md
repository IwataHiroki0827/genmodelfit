# genmodelfit

**Right-size image / video / audio / 3D generation models to your PC.**
画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。

genmodelfit is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork.
llmfit に着想を得た、生成モデル向けのツールです(llmfit の公式の派生ではありません)。

> Status: v0.1 in development (not on PyPI yet). / 状態: v0.1 を開発中(PyPI には未公開)。

```
$ genmodelfit recommend -m image --auto-only --vram 8 -n 5
Recommended (5)
Fit        Model                              Type     VRAM   Room  Basis  License     Auto
Good       image_z_image_turbo_int8           image    7.6~   +0.4  meas.  apache-2.0  yes
Marginal   image_z_image_turbo                image    6.2*   +1.8  meas.  apache-2.0  yes
Marginal   image_flux2_klein…it_4b_distilled  image    4.5*   +3.5  est.   apache-2.0  yes
Marginal   image_flux2_klein…ge_edit_4b_base  image    4.5*   +3.5  est.   apache-2.0  yes
Marginal   image_flux2_klein_text_to_image    image    7.5*   +0.5  est.   apache-2.0  yes
VRAM: GiB needed. ~ parts one at a time, * offload to RAM, ? size unknown. Basis: meas. = measured
Room: this PC's VRAM minus need (GiB). RAM = not enough system RAM
```

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
| `genmodelfit recommend [-m image] [-n 5] [--auto-only]` | best models for this PC (utility tools such as depth/pose/upscale only with `--include-tools`) / おすすめ順(深度・ポーズ・拡大などの補助ツールは `--include-tools` のときだけ) |
| `genmodelfit fit [-m video] [--perfect]` | every model with its fit level / すべての判定の表 |
| `genmodelfit info <id>` | files, licenses, estimate, limits, gate checks / 1 つのモデルの詳細 |
| `genmodelfit gate` | how many models can be auto-recommended, and why the rest are held / 自動でおすすめに出せる数と保留の理由 |
| `genmodelfit update` | rebuild the model list from upstream / 一覧を上流から作り直す |
| `genmodelfit update --from-release` | fetch the signed model list published by this project / 署名付きの配布の一覧を取り込む |
| `genmodelfit bench <template> --url URL` | measure VRAM / RAM / time on a running ComfyUI / 動いている ComfyUI で実測 |

Common options / 共通の指定:

- `--json` — machine-readable output for apps and agents ([docs/schema.md](docs/schema.md)) / アプリ向けの JSON
- `--vram 12 --ram 32` — judge for another PC / 別の PC を想定して判定
- `--width 1024 --height 1024`, `--frames 81`, `--length 30`, `--precision fp8` — judge a specific setting / 設定を指定して判定
- `--content sfw|all|uncensored` — filter by content mark (below; default `sfw`) / 内容の印で絞る(既定は `sfw`)
- `--comfyui-version 0.40.0` — the ComfyUI version your app runs, for the gate / 使う ComfyUI の版(自動おすすめの判定用)

Python:

```python
import genmodelfit as gmf

pc = gmf.system()
for f in gmf.recommend(modality="image", limit=5):
    print(f.model.id, f.estimate.fit, f.estimate.vram_gb, f.estimate.basis)
```

## Fit levels / 段階

| level | meaning |
|---|---|
| **Perfect** | everything fits in VRAM with headroom / 余裕をもって全部 GPU に載る |
| **Good** | fits all at once, or when parts are loaded one at a time (`~`) / 全部、または部品を順に載せ替えれば載る |
| **Marginal** | runs by offloading weights to RAM (`*`, slow), on CPU, or with slightly too little RAM / メインメモリへ逃がせば動く(遅い) |
| **Too tight** | does not run (VRAM or RAM) / 動かない |

Every estimate says whether it is **measured** (`meas.`, a real run) or **estimated** (`est.`, formula).
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
  (39 in the bundled list) and marked `[U]` in tables, but `recommend` / `fit` and `gmf.recommend()` show them
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
known, and the content is `sfw`. Otherwise it is held with machine-readable reasons (e.g. `held:license`).
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
