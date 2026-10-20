# JSON schema / JSON の形

`schema_version`: **1**. The source of truth is `src/genmodelfit/models.py`; this page must match it.
正本は `src/genmodelfit/models.py`。形を変えたら `SCHEMA_VERSION` とこのページを直す。

**Units / 単位**: every `*_gb` value is **GiB** (2^30 bytes): file sizes are bytes / 2^30, VRAM is nvidia-smi MiB / 1024,
RAM is psutil bytes / 2^30. Measurements are assumed to be GiB too (`unit` in `measurements.json`).
`*_gb` はすべて GiB。CLI の表示の「GB」も GiB。

## Common envelope / すべてのコマンドに共通

Every command with `--json` prints one object:

| key | type | meaning |
|---|---|---|
| `genmodelfit` | str | tool version |
| `command` | str | `system` / `recommend` / `fit` / `info` / `update` / `bench` / `gate` |
| `system` | System | the PC that was judged (after `--vram` / `--ram` overrides) |
| `database` | object | `origin` (`bundled` / `cache` / `path`), `generated_at`, `models`, `measurements` (counts) |
| `error` | str | only on failure (exit code 1) |

## System

| key | type | meaning |
|---|---|---|
| `os`, `cpu` | str | |
| `cpu_cores` | int | logical threads |
| `ram_total_gb`, `ram_available_gb` | float | GiB |
| `gpus` | Gpu[] | only GPUs confirmed by `nvidia-smi` (v0.1 counts NVIDIA CUDA only) |
| `best_gpu` | Gpu \| null | the GPU used for judging (largest VRAM) |
| `disk_free_gb` | float \| null | free space at the home directory (or `--disk-path`) |
| `overrides` | str[] | `vram` / `ram` when overridden |

Gpu: `name`, `vendor` (`nvidia`…), `vram_total_gb`, `vram_free_gb` (null when overridden), `driver`, `backend` (`cuda` / `none`).

## recommend / fit

`models`: list of Fit, best first.

| key | type | meaning |
|---|---|---|
| `id` | str | template name (`image_z_image_turbo_int8`) or `hf:<owner>/<repo>` |
| `name`, `modality`, `tasks` | | modality ∈ `image` `video` `music` `sfx` `tts` `3d` |
| `content` | str | `sfw` or `uncensored` (see README) |
| `license` | str \| null | strictest license of the files (`apache-2.0`, `mit`, `other`, …) |
| `source` | str | `comfyui-template` / `huggingface` / `manual` |
| `total_size_gb` | float \| null | size of all listed files (GiB) |
| `score` | float | 0–100, used for ordering |
| `fit` | str | `perfect` / `good` / `marginal` / `too_tight` |
| `estimate` | Estimate | |
| `gate` | Gate | can it be auto-recommended (docs/gate.md) |

`recommend` also returns `filters` (`modality`, `content`, `min_fit`, `limit`, `auto_only`). `recommend` drops `too_tight`; `fit` keeps everything.

## Estimate

| key | type | meaning |
|---|---|---|
| `vram_gb` | float | peak VRAM when everything stays on the GPU / 全部載せ |
| `vram_sequential_gb` | float \| null | peak VRAM when ComfyUI swaps parts between stages (text encoder → model → VAE) / 順に載せ替え |
| `vram_offload_gb` | float | peak VRAM when part of the weights stays in RAM (lowvram / dynamic VRAM) / 逃がす |
| `ram_gb` | float | system RAM needed for the chosen run mode: GPU (all-in / swapping) = 3 + largest stage's weights; offload / CPU = 3 + all weights. Short by up to 1.5× → `marginal` (page file, slow); more → `too_tight` |
| `disk_gb` | float \| null | size of the files counted (one variant per part; see `docs/accuracy.md`) |
| `basis` | str | **`measured`** (from a real run) or **`estimated`** (formula). Always present |
| `measured_on` | str \| null | GPU of the measurement when `basis` is `measured` (null when judged without a GPU) |
| `fit` | str | the four levels below |
| `run_mode` | str | `gpu` / `offload` / `cpu` / `none` |
| `reasons` | str[] | human-readable reasons (Japanese) |
| `settings` | object \| null | settings actually judged (unset items filled from the measurement, else per-modality defaults) |
| `limits` | Limits \| null | largest settings that still reach `good` on this PC. Set by `info`, and by `fit` / `recommend` with `--limits` |
| `unknown_size` | bool | the weight size is unknown, so nothing was judged: `fit` is `too_tight`, `run_mode` `none`, VRAM / RAM values are 0, `limits` null, first reason 「大きさが分からないため判定できません」. `recommend` drops these |

When no settings are given, a model with a measurement is judged at the measured settings (so the measured value is
used as is); others use per-modality defaults (image 1024², video 832×480 × 81 frames, music 30 s, sfx / tts 10 s).
A measurement is used as `basis: "measured"` only when width / height / frames / seconds / batch / precision match
(precision compared case-insensitively; a measurement without precision means the model's own precision) and a GPU is
present; other settings are shifted by it and stay `estimated`. Measurements with `outcome` other than `ok` or with
`standard: false` are ignored. Note that `fit` therefore compares measured models at their measured settings with other
models at the defaults; pass `--width` etc. to compare at the same settings. How measurements and coefficients are used: `docs/accuracy.md`.

Limits / 限界 (all values are the largest that give `perfect` or `good`, other settings unchanged; `null` = even the
smallest tried value does not reach `good`):

| key | type | meaning |
|---|---|---|
| `level` | str | always `good` |
| `max_side_px` | int \| null | image / video: long side in px, keeping the aspect ratio (searched 256–2048 for image, 256–1280 for video, step 32) |
| `max_frames` | int \| null | video: frames at the judged resolution (4n+1, up to 241) |
| `max_seconds` | int \| null | music / sfx / tts: length in seconds (step 5, up to 600) |
| `at_cap` | str[] | keys that reached the end of the searched range (the real limit is higher) |

Fit levels / 段階:

| level | meaning |
|---|---|
| `perfect` | everything fits in VRAM with ≥10% headroom / 余裕をもって全部載る |
| `good` | everything fits with little headroom, or fits when parts are swapped stage by stage / 全部載る(余裕は少ない)、または順に載せ替えれば載る |
| `marginal` | runs only by offloading to RAM, on CPU, or with a small RAM shortfall (page file); slow / 逃がせば動く・CPU・メモリが少し足りない。遅い |
| `too_tight` | does not run (VRAM, or RAM short by more than 1.5×), **or cannot be judged because the size is unknown** (`unknown_size: true`) / 動かない、または大きさが分からず判定できない |

## Gate

Computed every time (never stored in `db.json`). Details and the default license allowlist: docs/gate.md.

| key | type | meaning |
|---|---|---|
| `auto` | bool | true only when every check passed |
| `held_by` | str[] | ids of the failed checks, in the order below (`held_by[0]` is the table's `held:<id>`) |
| `comfyui_version` | str \| null | the ComfyUI version compared against (`--comfyui-version`, else `sources.comfyui.latest_release`) |
| `checks` | Check[] | always the 9 checks, in this order: `license`, `gated`, `official_workflow`, `custom_nodes`, `comfyui_version`, `url_reachable`, `sha256`, `content`, `size_known` |

Check: `id` (stable, machine-readable), `passed` (bool), `detail` (Japanese, for people; wording may change).

## gate

`genmodelfit gate --json`: `policy` (`license_allowlist`, `comfyui_version`, `allow_uncensored`),
`filters` (`modality`, `only_auto`, `only_held`), `summary` (`models`, `auto`, `held`,
`failing` = models failing each check, `first_failing` = models whose first failed check is that one),
`models` (`{"id", …Gate}`; all, or only auto / held with `--only-auto` / `--only-held`).

## info

`model` (ModelEntry), `estimate`, `gate`, `measurements` (Measurement[]).

### ModelEntry

`id`, `name`, `modality`, `tasks`, `source`, `files` (ModelFile[]), `template` (Template \| null), `params_b`,
`precision`, `license`, `license_url`, `base_model`, `content`, `content_evidence` (str[]: why it is marked
uncensored), `discovered_from` (str[]: where it was found), `updated`, `homepage`, `downloads` / `likes`
(int \| null: Hugging Face counters, set for `source: huggingface`), `total_size_bytes`,
`license_name` (the name of an `other` license, from the strictest file), `license_evidence` (str[]: why the license is
unknown — entries start with `license_unverified:`), `files_complete`, `size_floor_bytes` (below).

**License contract**: `license` is the strictest license over **all** files (LoRAs, upscalers and clip_vision included).
`null` means unknown and must be treated as "not allowed". A license is never more permissive than what could be verified
upstream (docs/sources.md「ライセンス」).

**Size contract** (`files_complete` / `size_floor_bytes`):

| `files_complete` | meaning | what an estimator must do |
|---|---|---|
| `true` | the listed files make up the whole set | use `total_size_bytes` |
| `false` | the list is known to be incomplete (no main model file, files < 50% of the template's index `size`, or a GGUF main file without text encoder / VAE) | never treat the entry as small: use `max(total_size_bytes, size_floor_bytes)` (`ModelEntry.estimate_size_bytes`); if `size_floor_bytes` is null, treat the size as unknown (not judgeable) |
| `null` | not checked (older data) | treat as `true` |

`size_floor_bytes` is the template's index `size` when `files_complete` is false (null for Hugging Face entries, which
have no index size).

ModelFile: `name`, `url`, `sha256`, `size_bytes`, `directory` (ComfyUI folder), `repo_id`, `license`,
`license_name`, `gated` (bool \| null = unknown), `role` (`main` / `text_encoder` / `vae` / `clip_vision` / `lora` / `other`),
`license_from` (the repo the license was read from: the upstream repo for repackaged files, the table's upstream for
known components; a repo id, never a path), `repo_license` (the hosting repo's own label when it differs from the
upstream one), `license_url`, `license_path` (str[]: the derivation path that was checked — hosting repo → base_model … →
deciding repo), `license_evidence` (str[]: `license_unverified: …`, `component:<key>`, `family:<key>`, `upstream_gated: <repo>` — the component's upstream requires accepting terms),
`head_status` (int \| null: status of an anonymous HEAD without following redirects, recorded by `update`;
null = not checked), `head_checked_at` (UTC ISO-8601 of that HEAD).
The estimator decides a file's part from `directory` first, then `role`, then the Hugging Face folder / known file
names, and counts one variant per part (`docs/accuracy.md`, 「ファイルの数え方」).

Template: `name`, `title`, `category`, `min_comfyui_version`, `open_source`, `requires_custom_nodes`, `usage`,
`date`, `size_bytes`.

### Measurement

`model_id`, `gpu_name`, `gpu_vram_gb`, `peak_vram_gb`, `peak_ram_gb`, `seconds` (wall time), `settings`
(`width`, `height`, `frames`, `seconds` = audio length, `steps`, `precision`, `batch`), `driver`,
`comfyui_version`, `offload`, `source`, `date`, `note`, `outcome`, `standard`.

| key | type | meaning |
|---|---|---|
| `gpu_vram_gb` | float | total VRAM. bench stores `/system_stats` `vram_total` in GiB rounded to 2 decimals (e.g. `7.99`); older bundled rows hold the nominal size (`16`) |
| `offload` | str \| null | how ComfyUI placed the model: `none` (all on GPU) / `dynamic` (dynamic VRAM) / `partial` (partially loaded) / `lowvram` (`--lowvram` / `--novram`) / `oom` (ran out of memory) |
| `outcome` | str | `ok` (finished) / `oom` (out of memory) / `error` (failed for another reason). Default `"ok"` |
| `vram_metric` | str \| null | how `peak_vram_gb` was taken: `torch_reserved` (bench: peak torch reservation of the ComfyUI process, excluding the CUDA context) or null (not recorded; the bundled RTX 5080 rows). Calibration uses only rows with null (`calibrate.CALIBRATION_METRIC`) |
| `standard` | bool | `true` when the official template was used as is (only width / height / frames / length / steps / seed changed), every reachable node actually ran (no partial cache) and no other job ran in between. Default `true` |

`outcome` and `standard` were added later as optional fields with defaults, so older files without them still read
as `ok` / `true` (`schema_version` stays 1). Estimates use only measurements with `outcome == "ok"` and
`standard != false`; OOM rows are still welcome as data but must keep `outcome: "oom"`.
後から足した任意の項目。無い古い行は ok・標準として読む。見積もりは ok かつ標準の実測だけを使う。

## bench

`genmodelfit bench <template> --json` prints the envelope (without `system` / `database`) plus:

| key | type | meaning |
|---|---|---|
| `measurement` | Measurement | the result (also saved to `saved_to` when saved) |
| `saved_to` | str \| null | where the JSON was written (`--output`, default `<cache>/measurements/<model>_<gpu>_<UTC>.json`); null when not saved |
| `not_saved` | str | only when not saved: why (failed or non-standard results are not cached unless `--save-failed` / `--save-nonstandard` / `--output`) |
| `workflow_source` | str | `server-template` / `github-template` / `file:api` / `file:ui` |
| `warnings` | str[] | settings that could not be applied, uploaded inputs, … (Japanese) |
| `failed` | str | only when the run failed in ComfyUI (OOM or another execution error; exit code 1). `measurement.outcome` is `oom` / `error` |

How bench fills Measurement / 測り方:

- `peak_vram_gb`: max of `/system_stats` `devices[0].torch_vram_total` polled every 0.2 s (the ComfyUI process,
  without the CUDA context). The GPU-wide `nvidia-smi` delta from the pre-run baseline goes to `note`.
- `peak_ram_gb`: max resident memory of the process listening on the port (and its children), only when its
  command line contains `main.py` or `comfy`; null when remote or not identified.
- `seconds`: wall time from submit to completion, including model loading. `/free` is sent first only with `--free`;
  without it the pre-run torch allocation is written to `note` and a warning is printed.
- `offload`: read from the ComfyUI log: `dynamic` / `partial` / `lowvram` / `none` / `oom`, or null when unknown.
- `--set` values: only numbers are written to `note`; for other values only the input name is kept.
- The saved file is `{"measurements": [Measurement]}` and holds no paths, user names or prompts.

## Database file (`db.json`)

```json
{"schema_version": 1, "generated_at": "2026-10-20T03:17:00Z", "sources": {…}, "models": [ModelEntry, …]}
```

`generated_at` is always an aware UTC ISO-8601 string ending in `Z`.
`downloads`, `likes`, `license_from`, `repo_license`, and (PR #13 review) `ModelEntry.license_name` /
`license_evidence` / `files_complete` / `size_floor_bytes` and `ModelFile.license_url` / `license_path` /
`license_evidence`, and (#3) `ModelFile.head_status` / `head_checked_at` were added without changing `schema_version` (optional fields; older readers ignore unknown keys).
Task `tool` was added for Image Tools / Video Tools templates without a known tag.

`sources` (written by `genmodelfit update`, see docs/sources.md):

| key | meaning |
|---|---|
| `comfyui-templates` | `templates_total`, `local`, `skipped` (by reason: `api` / `not_open_source` / `no_size` / `category`), `loaded`, `failed` |
| `huggingface` | `repos` (API calls), `failed`, `rate_limited` (429 waits), `rate_limit_waited_s`, `failures` (first 20) |
| `huggingface-search` | `queries`, `candidates`, `added`, `skipped` (by reason, incl. `duplicate_weights`), `excluded`, `failed_queries` |
| `partial` | bool: some upstream data could not be fetched (then `update` exits 1 unless `--allow-partial`) |
| `partial_counts` | `templates_failed`, `hf_repos_failed_transient`, `entries_affected`, `hf_search_failed`, `rate_limit_budget_exhausted`, `carried_over` |
| `license_policy` | version of the license rules (2). Values are carried over from a previous list only when it has the same version |
| `civitai` | why it is not used |
| `size_check` | `templates`, `equal`, `within_10pct` (index `size` vs the sum of the files) |
| `counts` | `models`, `modality`, `content`, `source`, `license` (null = unknown), `license_tier`, `files_complete_false` |
| `head_check` | `checked_at`, `urls` (unique HF URLs sent a HEAD), `status` (count per status code), `unchecked` (no answer), `not_sent` (files on other hosts, per host) |
| `comfyui` | `latest_release` (tag of the latest ComfyUI release, e.g. `v0.42.1`; null if it could not be read), `url`, `checked_at` |
| `excluded` | number of models dropped by the exclusion rule (names are not kept) |
| `warnings` | str[] |

## update

`genmodelfit update --json` prints the envelope with `output` (path written) and `sources` (above).
On failure it prints `error` and exits 1; the previous cache is left untouched.
`update --heads-only` reads the current database (`--db`, cache or bundled), redoes only the HEADs and the ComfyUI
version, and writes it back (`--output`, default the cache).

Measurements live in `measurements.json` (`{"unit": "GiB", "unit_note": "…", "measurements": [Measurement, …]}`), bundled plus
`<cache>/measurements/*.json`.
