# genmodelfit

**Right-size image / video / audio / 3D generation models to your PC.**
画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。

genmodelfit is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork.
llmfit に着想を得た、生成モデル向けのツールです(llmfit の公式の派生ではありません)。

> Status: v0.1 in development. / 状態: v0.1 を開発中。

```
$ genmodelfit fit --vram 8 -n 5
All models (5)
Fit        Model                          Type   VRAM  Room  Basis     License     Auto
Perfect    3d_hunyuan3d_image_to_model    3d      6.7  +1.3  measured  other       held:license
Good       image_z_image_turbo_int8       image  7.6~  +0.4  measured  apache-2.0  yes
Good       audio_ace_step_1_5_checkpoint  music  6.0~  +2.0  measured  mit         yes
Good       3d_hunyuan3d-v2.1              3d     7.8~  +0.2  measured  other       held:license
Good       image_sdxl_simple              image   8.0  +0.0  measured  openrail++  held:license
VRAM = GiB needed (~ = parts loaded one at a time, * = offload to RAM)
Room = this PC's VRAM minus need (GiB)
```
(The Auto column is filled by the gate; values above are illustrative of its format. / Auto 列は自動おすすめの判定の結果。上の値は表示の形の例です。)

## Why / なぜ作るか

llmfit covers LLMs only ([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41)). For diffusion, video and
audio models the memory needed changes several times over with precision, resolution, frame count and
offloading to system RAM, so they need their own database and scoring.

生成モデルは、精度・解像度・コマ数・メインメモリへの逃がし方で、必要な VRAM が数倍変わります。
そのため LLM とは別のデータと判定が要ります。

## Install / 入れ方

```
pip install genmodelfit        # after the first release / 最初の公開後
# from source / ソースから
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10+. Windows and Linux. GPUs: NVIDIA (CUDA) in v0.1; others are judged as CPU-only.

## Usage / 使い方

| command | what it does / 内容 |
|---|---|
| `genmodelfit system` | detected GPU / RAM / disk / この PC の GPU・メモリ・空き |
| `genmodelfit recommend [-m image] [-n 5]` | best models for this PC / おすすめ順 |
| `genmodelfit fit [-m video] [--perfect]` | every model with its fit level / すべての判定の表 |
| `genmodelfit info <id>` | files, licenses, estimate, limits / 1 つのモデルの詳細 |
| `genmodelfit update` | rebuild the model list from upstream / 一覧を上流から作り直す |
| `genmodelfit bench <template> --url URL` | measure on a running ComfyUI / 動いている ComfyUI で実測 |

Common options / 共通の指定:

- `--json` — machine-readable output for apps and agents ([docs/schema.md](docs/schema.md)) / アプリ向けの JSON
- `--vram 12 --ram 32` — judge for another PC / 別の PC を想定して判定
- `--width 1024 --height 1024`, `--frames 81`, `--length 30` — judge a specific setting / 設定を指定して判定
- `--content all|sfw|uncensored` — filter by content mark (below) / 内容の印で絞る

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
| **Good** | everything fits, little headroom / 全部載るが余裕は少ない |
| **Marginal** | runs by offloading weights to RAM (slow), or on CPU / メインメモリへ逃がせば動く(遅い) |
| **Too tight** | does not run (VRAM or RAM) / 動かない |

Every estimate says whether it is **measured** (a real run) or **est.** (formula). Measured values win.
すべての見積もりに「実測」か「推定」かを付けます。実測があれば実測を使います。

## Data sources / データ源

- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT): which models a workflow
  uses, download URLs, hashes, required ComfyUI version, custom nodes. API (cloud) templates are excluded.
- [Hugging Face Hub API](https://huggingface.co/docs/hub/api): licenses (following `base_model` for repackaged
  repos), gating, file sizes, sha256, and discovery of community models.
- Measurements: `src/genmodelfit/data/measurements.json` (contributions welcome, see CONTRIBUTING.md).

Details: [docs/sources.md](docs/sources.md). genmodelfit stores model **information only**; it never hosts weights.
genmodelfit が持つのはモデルの情報だけで、モデル本体は持ちません。

## Content marks / 内容の印

- `sfw` — default.
- `uncensored` — models whose safety filtering was removed or that are made for adult output
  (Hugging Face `not-for-all-audiences` tag, or names such as uncensored / abliterated / nsfw). They are listed,
  marked `[U]` in tables, and can be filtered with `--content`. The evidence is kept in `content_evidence`.
  Whether to show them is up to the app using genmodelfit.
- Models aimed at sexual content involving minors are **excluded** from the list entirely.

`uncensored` は、安全フィルタを外したモデルや成人向けの出力を想定したモデルです。一覧には載せ、印を付けて
絞り込めるようにしています。表示するかどうかは使う側のアプリが決めます。未成年の性的表現を目的とするものは
一覧から除外します。

## Auto-recommendation gate / 自動でおすすめに出す条件

A model is marked `auto: yes` only if **all** machine-checkable conditions hold: license on a commercial-use
allowlist (replaceable), not gated, official workflow without custom nodes, supported ComfyUI version,
download URLs answer, sha256 known, not uncensored. Otherwise it is held with the failing reasons (e.g. `held:license`).
See [docs/gate.md](docs/gate.md) and `genmodelfit gate`.

## Accuracy and limits / 精度と限界

See [docs/accuracy.md](docs/accuracy.md). In short / 要点:

- Estimates are calibrated on only 9 real runs on one GPU (RTX 5080 16 GB). Leave-one-out error could be evaluated on 4 of them (image and 3D: max 6.5%); video and audio could not be evaluated yet. Treat `est.` values as rough guidance.
  実測は 1 種類の GPU で 9 件だけです。1 件ずつ外して当てる方法で評価できたのは 4 件(画像・3D、最大誤差 6.5%)で、動画と音声はまだ評価できていません。
- ComfyUI's dynamic VRAM uses more memory when it is available, so a peak measured on a large GPU is not the
  minimum requirement.
- Models whose download size is unknown are not judged (shown as `?`).
- v0.1 counts NVIDIA GPUs only.

## License

MIT. Upstream data keeps its own terms; model licenses are reported, not granted.
モデルのライセンスは表示するだけで、利用の許諾はしません。各モデルのライセンスを確かめてください。
