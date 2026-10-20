# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
変更の記録。書き方は Keep a Changelog にならう。

## [Unreleased]

## [0.1.1] - Unreleased

Fixes from the owner's review on an RTX 4500 Ada 24 GB / 128 GB RAM PC (Issue #28).
RTX 4500 Ada 24 GB・メモリ 128 GB の PC でのレビューの指摘を直した版。

### Changed

- **Wan 2.2 14B and other workflows that run several diffusion models one sampler after another** (high_noise →
  low_noise): the swapped (`GPU~`) and offload estimates count one model (+ its LoRA) at a time, because ComfyUI
  unloads the unused model before loading the next one (`load_models_gpu` → `free_memory`, ComfyUI commit
  b1a247c, cited in `docs/schema.md`). Wan 2.2 14B I2V at 832×480×81: swapped 35.5 → 21.0 GiB, offload 18.1 → 12.3 GiB.
  On 24 GB it now runs on the GPU with parts swapped (`GPU~`, `marginal` at 87 %) instead of `GPU+RAM`; on 16 GB it is
  `marginal` with offload instead of `too_tight`. The sets come from the template graph (`sampler_stages`, new) or
  from `*high_noise*` / `*low_noise*` file names.
- **Precision choice within a model**: among the variants that run on the GPU without offload (Good or better, `GPU`
  or `GPU~`), the highest precision is shown, like llmfit; only when none does, the better level wins as before.
  Z-Image Turbo on 24 GB now shows bf16 instead of int8. The row's score is taken from the same use and the same tier
  (runs without offload = one tier), so it never drops when VRAM grows.
- **Unknown licenses are hidden by default** from `recommend`, `fit`, `search`, `gmf.recommend()` and `gmf.search()`
  (`hidden.license_unknown`); `--include-unknown-license` / `include_unknown_license=True` shows them; `info <id>`
  still shows them.
- **Models whose size is unknown show `?`** (`fit_level: "unknown"`, `fit_label: "?"`) instead of "Too tight";
  `fit` lists them last, `recommend` still hides them.
- **Community templates** come after official ones only within the same auto group and fit level (they used to come
  after every official row, so a Good community model appeared after Marginal ones).
- TTS GGUF repositories (llama.cpp-style runtimes: VieNeu-TTS v3 Turbo, Qwen3-TTS GGUF, Higgs Audio v3 Studio,
  Drama-3) are no longer marked incomplete for lacking a text encoder / VAE, so they are judged.
- OS shown as `Windows 11` on Python 3.10 / 3.11 too (build number ≥ 22000).

### Added

- Precision from the files themselves: `update` (and the new `update --inspect-only`, `--offline` for the cache only)
  reads the safetensors JSON header / GGUF header of the main weight files with HTTP Range requests (never the
  weights), caches them in `file_headers.json`, and stores `ModelFile.header_precision` / `header_params`. On the
  bundled list, entries with unknown precision went from 97 to 25 (sfw: 90 → 25; the rest are gated repos, 401, or
  have no main file on Hugging Face).
- `ModelEntry.sampler_stages`, `ModelFile.header_precision`, `ModelFile.header_params` (optional; `schema_version`
  stays 1). `sources.file_headers`, `sources.sampler_stages`.
- README: how to install so that the `genmodelfit` command is found (pipx / uv tool, the pip "not on PATH" warning,
  and `python -m genmodelfit`, which always works). Also in `genmodelfit --help`.

### Data

- Bundled list: HEADs re-checked on 2026-10-20T10:33:07Z (valid until 2026-11-19T10:33:07Z); precision from file
  headers; `sampler_stages` from the 277 official templates (19 with several sampler sets).

## [0.1.0] - 2026-10-20

First public release. / 最初の公開版。

### Added

- Hardware detection (`genmodelfit system`, `gmf.system()`): NVIDIA GPUs via `nvidia-smi`, RAM, free disk;
  `--vram` / `--ram` to judge another PC. Other PCs are judged as CPU-only.
- Bundled model list (`db.json`): 323 entries (image, video, music, sound effects, TTS, 3D) built from the official
  ComfyUI workflow templates and the Hugging Face Hub API, folded into 204 models. Hugging Face models need at least
  10,000 all-time downloads (1,000 for trending ones), like llmfit.
- Memory estimates per component and stage, with swapping in turn, offload to RAM and CPU run modes; every estimate
  says whether it is `measured` or `estimated`. Calibration on real runs (`python -m genmodelfit.calibrate`,
  `docs/accuracy.md`).
- Fit levels by memory utilization (Perfect ≤ 60 %, Good ≤ 85 %, Marginal ≤ 98 %, Too tight), like llmfit.
- `recommend`, `fit`, `search`, `types`, `info` with a readable table and `--json` for apps (`docs/schema.md`):
  one row per model (same weights family and size), best precision that fits, score (0–100), per-type sections,
  filters by type, use (`--task`), license, fit level and content.
- Auto-recommendation gate (`genmodelfit gate`, `auto` / `held_reasons`): license allowlist traced to the upstream,
  gating, official workflow without custom nodes, ComfyUI version, anonymous HEAD within 30 days, sha256, known size,
  content `sfw`.
- Content marks: uncensored models are kept in the list with evidence but hidden by default (`content="sfw"`);
  models aimed at sexual content involving minors are excluded.
- `genmodelfit update` (rebuild from upstream, `--heads-only`) and `update --from-release` (signed list from the
  `data-latest` release, Ed25519, rollback protection).
- `genmodelfit bench`: measure peak VRAM / RAM / time on a running ComfyUI and contribute measurements
  (`CONTRIBUTING.md`).
- Python API: `gmf.recommend()`, `gmf.search()`, `gmf.info()`, `gmf.system()`; library functions never print.
- Data freshness: `gmf.data_status()` (`DataStatus`: source, HEAD check dates, `expired`, `expires_at`,
  `auto_count`) and the Data lines of `genmodelfit system` / `update` (`database.status` in `system --json`).
  It uses the gate's own check, so `expired` tells apps when auto-recommendations have dropped.
- Type information (`py.typed`), Python 3.10–3.13 on Windows and Linux.
- Privacy: read-only commands and the Python API never touch the network; HTTP sends a fixed User-Agent and never
  attaches `~/.netrc` credentials. No telemetry (README "Privacy").

### Changed

- License: MPL-2.0 for the code and the bundled data (was MIT before the first release).
  Attribution for the upstream data is in `THIRD_PARTY_NOTICES`.

[Unreleased]: https://github.com/laxshmate/genmodelfit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.1.0
