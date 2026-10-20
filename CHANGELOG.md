# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
変更の記録。書き方は Keep a Changelog にならう。

## [Unreleased]

### Changed

- License tracing (#28): a hand-checked source table (`license_sources.py`) maps files in repackaging repos
  (Comfy-Org, Kijai, …) to their original models, using primary sources only (original model card / LICENSE,
  official GitHub, sha256). `update` applies it to files it could not resolve; `apply_license_sources(db)` applies it
  offline. VOID (netflix/void-model) is now unknown (fine-tuned from a CogVideoX-licensed model). `LICENSE_POLICY` 4.
  ライセンスの出どころの表を足した(再配布のファイル → 元のモデル。一次資料だけで確かめたもの)。

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
