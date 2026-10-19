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
uncensored), `discovered_from` (str[]: where it was found), `updated`, `homepage`, `total_size_bytes`.

ModelFile: `name`, `url`, `sha256`, `size_bytes`, `directory` (ComfyUI folder), `repo_id`, `license`,
`license_name`, `gated` (bool \| null = unknown), `role` (`main` / `text_encoder` / `vae` / `clip_vision` / `lora` / `other`).

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
{"schema_version": 1, "generated_at": "…", "sources": {…}, "models": [ModelEntry, …]}
```

Measurements live in `measurements.json` (`{"measurements": [Measurement, …]}`), bundled plus
`<cache>/measurements/*.json`.
