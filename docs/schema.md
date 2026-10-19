# JSON schema / JSON の形

`schema_version`: **1**. The source of truth is `src/genmodelfit/models.py`; this page must match it.
正本は `src/genmodelfit/models.py`。形を変えたら `SCHEMA_VERSION` とこのページを直す。

## Common envelope / すべてのコマンドに共通

Every command with `--json` prints one object:

| key | type | meaning |
|---|---|---|
| `genmodelfit` | str | tool version |
| `command` | str | `system` / `recommend` / `fit` / `info` / `update` / `bench` |
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
| `total_size_gb` | float \| null | download size (decimal GB) |
| `score` | float | 0–100, used for ordering |
| `fit` | str | `perfect` / `good` / `marginal` / `too_tight` |
| `estimate` | Estimate | |
| `gate` | Gate \| null | can it be auto-recommended (Issue #3) |

`recommend` also returns `filters`. `recommend` drops `too_tight`; `fit` keeps everything.

## Estimate

| key | type | meaning |
|---|---|---|
| `vram_gb` | float | peak VRAM when everything stays on the GPU |
| `vram_offload_gb` | float | peak VRAM when parts are offloaded to RAM |
| `ram_gb` | float | system RAM needed |
| `disk_gb` | float \| null | download size |
| `basis` | str | **`measured`** (from a real run) or **`estimated`** (formula). Always present |
| `measured_on` | str \| null | GPU of the measurement when `basis` is `measured` |
| `fit` | str | the four levels below |
| `run_mode` | str | `gpu` / `offload` / `cpu` / `none` |
| `reasons` | str[] | human-readable reasons (Japanese) |

Fit levels / 段階:

| level | meaning |
|---|---|
| `perfect` | everything fits in VRAM with ≥10% headroom / 余裕をもって全部載る |
| `good` | everything fits, little headroom / 全部載るが余裕は少ない |
| `marginal` | runs only by offloading to RAM, or on CPU; slow / 逃がせば動く・遅い |
| `too_tight` | does not run (VRAM or RAM) / 動かない |

## info

`model` (ModelEntry), `estimate`, `gate`, `measurements` (Measurement[]).

### ModelEntry

`id`, `name`, `modality`, `tasks`, `source`, `files` (ModelFile[]), `template` (Template \| null), `params_b`,
`precision`, `license`, `license_url`, `base_model`, `content`, `content_evidence` (str[]: why it is marked
uncensored), `discovered_from` (str[]: where it was found), `updated`, `homepage`, `downloads` / `likes`
(int \| null: Hugging Face counters, set for `source: huggingface`), `total_size_bytes`.

ModelFile: `name`, `url`, `sha256`, `size_bytes`, `directory` (ComfyUI folder), `repo_id`, `license`,
`license_name`, `gated` (bool \| null = unknown), `role` (`main` / `text_encoder` / `vae` / `clip_vision` / `lora` / `other`),
`license_from` (the HF repo the license was read from; the upstream repo for repackaged files),
`repo_license` (the hosting repo's own label when it differs from the upstream one).

Template: `name`, `title`, `category`, `min_comfyui_version`, `open_source`, `requires_custom_nodes`, `usage`,
`date`, `size_bytes`.

### Measurement

`model_id`, `gpu_name`, `gpu_vram_gb`, `peak_vram_gb`, `peak_ram_gb`, `seconds` (wall time), `settings`
(`width`, `height`, `frames`, `seconds` = audio length, `steps`, `precision`, `batch`), `driver`,
`comfyui_version`, `offload`, `source`, `date`, `note`.

## Database file (`db.json`)

```json
{"schema_version": 1, "generated_at": "2026-10-20T03:17:00Z", "sources": {…}, "models": [ModelEntry, …]}
```

`generated_at` is always an aware UTC ISO-8601 string ending in `Z`.
`downloads`, `likes`, `license_from` and `repo_license` were added without changing `schema_version`
(optional fields; older readers ignore unknown keys).

`sources` (written by `genmodelfit update`, see docs/sources.md):

| key | meaning |
|---|---|
| `comfyui-templates` | `templates_total`, `local`, `skipped` (by reason: `api` / `not_open_source` / `no_size` / `category`), `loaded`, `failed` |
| `huggingface` | `repos` (API calls), `failed`, `rate_limited` (429 waits), `failures` (first 20) |
| `huggingface-search` | `queries`, `candidates`, `added`, `skipped` (by reason), `excluded` |
| `civitai` | why it is not used |
| `size_check` | `templates`, `equal`, `within_10pct` (index `size` vs the sum of the files) |
| `counts` | `models`, `modality`, `content`, `source` |
| `excluded` | number of models dropped by the exclusion rule (names are not kept) |
| `warnings` | str[] |

## update

`genmodelfit update --json` prints the envelope with `output` (path written) and `sources` (above).
On failure it prints `error` and exits 1; the previous cache is left untouched.

Measurements live in `measurements.json` (`{"measurements": [Measurement, …]}`), bundled plus
`<cache>/measurements/*.json`.
