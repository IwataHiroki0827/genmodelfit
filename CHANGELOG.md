# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
変更の記録。書き方は Keep a Changelog にならう。

## [Unreleased]

### Added

- Measurement submissions from users (Issue #34). `genmodelfit bench … --submission PATH` and
  `genmodelfit contrib make --from <bench result>` write a submission file (`contrib_schema` 1:
  `data/contrib/<gpu>/<date>-<sha8>.json`) with the GPU name / VRAM / generation / compute capability / driver, the ComfyUI
  version, the OS family, RAM rounded to GiB and the results — never host / user names, paths, GPU UUID, PCI address,
  IP, CPU name or exact times (the note is scrubbed). genmodelfit sends nothing; users open the pull request themselves.
  `contrib show` prints exactly what will be shared with a privacy checklist; `contrib validate` checks schema, value
  ranges, privacy patterns, GPU name / generation consistency, file name and duplicates (exit 0 / 1 / 2);
  `.github/workflows/contrib.yml` runs it on pull requests touching `data/contrib/**` (read-only, the tool installed
  from the base commit, results in the job summary). `contrib aggregate` groups submissions by model, settings and GPU
  with median / MAD outlier removal into the new bundled `measurements_community.json` (empty for now), and
  `--report` re-runs the calibration and flags errors above a threshold (default 25%, exit 3 with `--fail-on-breach`)
  for the weekly job of #19.
- `Estimate.basis_detail` (`measured` > `measured_community` > `calibrated` > `estimated`); `basis` stays
  `measured` / `estimated`. Community measurements are used only when at least two consistent submissions agree, and
  rank after the maintainer's measurements at the same settings; they score 80 instead of 100 in `basis`.
  `Measurement.origin` / `n_submissions` / `consistent` (optional fields).
- `python -m genmodelfit.calibrate` also writes the check rows (measurements in another `vram_metric`, such as the
  RTX 3070 runs) to `calibration.json` (`check`) and the accuracy table; LOO rows carry `gpu_name` and `origin`.
  Coefficients are unchanged.
  利用者の実測の提出(作る・見る・検査・まとめ・毎週の精度の点検)と、根拠の段階 `basis_detail` を足した。

## [0.3.0] - 2026-10-21

### Added

- `genmodelfit download <model>` and `gmf.plan_download()` / `gmf.download()` (Issue #31): download all files of a
  model's workflow, with the precision chosen like `recommend` (`--precision`, `--budget GB`), into a ComfyUI folder by
  each file's template folder (`--comfyui-dir`) or the same layout elsewhere (`--output-dir`). `--list` shows the plan
  without any network access. Before downloading it shows every file's license and URL, the total size and the free
  disk and asks for consent (`--yes` to skip). sha256 is checked while writing (a mismatch deletes the file),
  `.part` files resume with HTTP Range, files already there are skipped, and it needs the download + 1 GB free.
  Gated files, uncensored models (unless `--content all|uncensored`) and unknown licenses (unless
  `--include-unknown-license`) are refused; only https URLs on Hugging Face / GitHub hosts are followed (each
  redirect is checked, and URLs that urlsplit and urllib3 would read as different hosts are refused before
  connecting); nothing is written outside the destination. Existing files are hashed again right before they
  are skipped, restarts against servers that ignore Range are capped, and a `.lock` file stops two downloads of
  the same file. URL paths may contain spaces (they are percent-encoded as requests does); spaces, `%` and `@` are
  refused only in the scheme and host, and refusals say whether the URL is malformed or the host is not allowed.
  JSON: docs/schema.md "download".
  モデルを落とす `download` を足した(同意・sha256・続きから・ComfyUI のフォルダ)。

## [0.2.0] - 2026-10-20

Fewer unknowns in the table and a clear meaning for `?` and `-` (Issue #30). The JSON output changes in a way that
can break apps (below), so this is a minor release under 0.x semver (owner's decision).
表の「?」「-」を減らし、使い分けをそろえた版。JSON の形が変わるので 0.2.0 にした(オーナーの判断)。

### Breaking (JSON)

These only concern models whose weight size is unknown (`estimate.unknown_size: true`, `fit_level: "unknown"`):

- `run_mode` (top level and `estimate.run_mode`) is now **`"unknown"`** (was `"none"`). `"none"` now only means
  "does not run" (`fit_level: "too_tight"`). `run_mode_label` is `"?"` (was `"-"`).
- `estimate.vram_gb`, `estimate.vram_offload_gb`, `estimate.ram_gb` and `estimate.vram_sequential_gb` are **`null`**
  (were `0.0`). The Python `Estimate` attributes stay `0.0`; only `Estimate.to_dict()` / `--json` change.
- `memory_available_gb` is now this PC's VRAM (or RAM) instead of `null` (the available amount is known;
  `memory_required_gb` / `utilization_pct` stay `null`).

Migration / 移行:

- Code that treated `run_mode == "none"` as "cannot use" should also handle `"unknown"` (or check
  `fit_level` / `estimate.unknown_size` first, as before).
- Code that summed or compared `estimate.vram_gb` / `ram_gb` must skip `null` (it was a meaningless `0.0` before).
- `fit_label` / `fit_level` and `schema_version` (1) are unchanged; the db.json fields added below are optional.
大きさが分からないモデルだけ: run_mode は unknown、estimate の VRAM / RAM は null、memory_available_gb は
この PC の量。run_mode の none を「使えない」と見ていたコードは unknown も扱う。

### Changed

- **`?` means unknown, `-` means not applicable**, in the table, `info`, the legend and JSON (`docs/schema.md`,
  「Unknown and not applicable」). Prec of an unknown precision is now `?` (was `-`); Mode / Mem% of a model whose
  size is unknown are `?` (Mode `-` stays for "does not run", VRAM `-` for CPU, Mem% `-` without a GPU). A new
  legend line explains both marks.
  「?」は分からない、「-」は当てはまらない。精度が分からないときは「?」(前は「-」)。
- License table: `circlestone-labs/Anima-Official-LoRAs` has a `LICENSE.md` (CircleStone Labs Non-Commercial License
  v1.0, added 2026-05-26) although its card metadata is empty; the table now sets the LoRAs to
  `other` (`circlestone-labs-non-commercial-v1.0`). The four Anima templates are no longer "license unknown" (still not
  auto-recommended: non-commercial). Bundled list: license unknown 47 → 43 entries (no uncensored), 60 → 56 in all.
  Anima の公式 LoRA は LICENSE.md(非商用)があったので表で決めた。

### Added

- **Precision from more sources**, never overriding a precision read from the file header
  (`ModelEntry.precision_source`: `header` / `filename` / `hf_api` / `config` / `workflow`, optional, schema 1;
  also `precision_source` in `models[]` and `workflows[]` of `recommend` / `fit` / `search --json`):
  - `hf_api`: the Hub API's `safetensors.parameters` when one dtype holds ≥ 90 % **and** element count × bytes is
    within 5 % of the entry's main safetensors size in that repo (readable anonymously even for gated repos whose
    file headers return 401);
  - `config`: `torch_dtype` / `dtype` of the `config.json` next to the main file (only when listed and not gated);
  - `workflow`: the official template's main loader node (`UNETLoader` `weight_dtype`, `WanVideoModelLoader`
    `quantization`, the loaded file name, `precision` widgets, …; `workflow.loader_precision`).
  New optional fields `ModelFile.repo_precision` / `repo_precision_source` and `ModelEntry.workflow_precision`.
  Tool weights whose role is not `main` (BiRefNet, Depth Anything 3) now get their header read too.
  Bundled list (no uncensored): precision unknown 25 → 10 (header 3, Hub API 4, workflow 8; csm-1b stays unknown
  because the Hub API total does not match its files; the rest are listed in docs/sources.md with the reason).
  精度をリポジトリ(Hub API・config.json)とテンプレートの読み込みノードからも読み、出どころを残す。
- `sources.repo_precision` (counts per source, `unauthorized`, `size_mismatch`), `sources.huggingface.unauthorized`
  (repos whose Hub API returns 401 — anonymous 401 is the same for private and missing repos),
  `unauthorized_missing` (the ones confirmed not to exist, `repo_precision.KNOWN_MISSING`: `google/gemma-3-12b`) and,
  after `update --inspect-only`, `unauthorized_recheck` (`unauthorized` / `private_or_missing` / `missing` /
  `readable` / `failed`), so the weekly update (#19) shows when Mage-Flow or Lens become public.
  401 のリポジトリを記録し、`--inspect-only` で確かめ直す。無いと確かめた名前は分ける。
- `docs/license_requests.md`: drafts of polite requests to upstream owners whose weights have no stated license or
  a license that conflicts with their base model (not posted; to be sent only after the owner's review).
  ライセンスの記載の依頼の下書き(まだ出さない)。

## [0.1.1] - 2026-10-20

Fixes from the owner's review on an RTX 4500 Ada 24 GB / 128 GB RAM PC (Issue #28).
RTX 4500 Ada 24 GB・メモリ 128 GB の PC でのレビューの指摘を直した版。

### Changed

- License tracing (#28): a hand-checked source table (`license_sources.py`) maps files in repackaging repos
  (Comfy-Org, Kijai, …) to their original models, using primary sources only (original model card / LICENSE,
  official GitHub, sha256). `update` applies it to files it could not resolve; `apply_license_sources(db)` applies it
  offline. VOID (netflix/void-model) is now unknown (fine-tuned from a CogVideoX-licensed model). `LICENSE_POLICY` 4.
  ライセンスの出どころの表を足した(再配布のファイル → 元のモデル。一次資料だけで確かめたもの)。
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
- **`fit` is easier to read** (like llmfit's `fit`): it shows only the models that run (Perfect / Good / Marginal)
  and counts the rest by reason in the header; `--all` shows everything (tools, models that do not run, `?`, unknown
  licenses). One table ordered by type → fit level → Score (no auto-first; see the Auto column). New options for
  `fit` / `search`: `--fit perfect|good|marginal|runnable|all` (`--min-fit` still works, not together with `--fit`),
  `--sort type|score|vram|disk|name`, `--all`; `fit --include-tools`. JSON follows the same order. Python:
  `gmf.fit_list()`, and `fit=` / `sort=` on `gmf.search()`.
- Table columns in every command: `Type | Status | Score | Model | …` (Type leftmost, Score left of the name).
- Review fixes: `recommend` ranks "runs without offload" (Perfect / Good on `GPU` / `GPU~`) as one tier, so Z-Image
  Turbo stays first in image on 24 GB after switching to bf16; JSON `level_cap` (`sequential` / `offload` / `cpu`) and a
  legend line explain why GPU~ is at most Good; `--fit` with `--min-fit` is a usage error (exit 2); info and the table
  round Mem% the same way; malformed file headers are reported as header errors.
- Licenses: rows of the source table can override an automatic result (`override`). ChronoEdit's distill LoRA was
  apache-2.0 from the card's metadata but the card says it is governed by the NVIDIA Open Model License; InfiniteTalk
  and Wan 2.1 I2V 480P now point at their real upstreams. Files of lineages whose license cannot be read (VOID) are
  unknown in the data too (as the gate already judged). An entry mixing apache-2.0 and mit files is shown as
  apache-2.0 (more obligations), not mit. `update.LICENSE_POLICY` 4 → 5.

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
