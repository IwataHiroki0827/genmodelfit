# genmodelfit

**Right-size image / video / audio / 3D generation models to your PC.**
画像・動画・音声・3D の生成モデルが、その PC で動くか・快適かを判定するツールです。

> Status: planning. Nothing is implemented yet. See the [issues](https://github.com/laxshmate/genmodelfit/issues).
> 状態: 計画中(まだ実装はありません)。進め方は Issue を見てください。

## Why / なぜ作るか

[llmfit](https://github.com/AlexsJones/llmfit) does this well for LLMs, but it does not cover
generation models ([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41)).
For diffusion / video / audio models the memory needed changes several times over with
precision, resolution, frame count and offloading, so a separate database and scoring are needed.

llmfit は LLM 向けで、画像・動画・音声の生成モデルは対象外です。生成モデルは精度・解像度・コマ数・
メモリへの逃がし方で必要な VRAM が数倍変わるので、別のデータと判定が要ります。genmodelfit は
llmfit に着想を得た、生成モデル向けの判定ツールです(llmfit の公式の派生ではありません)。

## Plan / 予定している機能

- **Hardware detection** — NVIDIA GPU (VRAM, driver), RAM, free disk.
- **Auto-updating model database** — built from the official
  [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) index and the Hugging Face Hub API
  (license, gated, file sizes, sha256).
- **Fit levels** — Perfect / Good / Marginal / Too tight, per model and per setting (resolution, length, precision).
- **Estimated vs measured** — memory is estimated from model size and settings; measured values always win
  and are labelled as such.
- **Auto-promotion gate** — a model is recommended automatically only when machine-checkable conditions hold
  (commercial-use license on an allowlist, no gating, official workflow without custom nodes, reachable URLs with hashes).
  Anything else (e.g. `license: other`) is held for human review.
- **CLI with JSON output** and a Python API, so apps can embed it.

## License

MIT. Data pulled from upstream sources keeps each source's own terms; model licenses are reported, not granted.
