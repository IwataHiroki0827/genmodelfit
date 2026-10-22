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
| `command` | str | `system` / `recommend` / `fit` / `search` / `types` / `info` / `update` / `bench` / `gate` / `download` / `diff` / `contrib` |
| `system` | System | the PC that was judged (after `--vram` / `--ram` overrides). Not in `gate` / `update` / `bench` / `diff` / `contrib` |
| `database` | object | `origin` (`bundled` / `cache` / `path`), `generated_at`, `models`, `measurements` (counts). In `system` also `status` (DataStatus, below). Not in `bench` / `update --from-release` |
| `error` | str | only on failure (exit code 1) |
| `candidates` | object[] | only when `info` matched several models (see info) |

## Side effects / 通信と書き込み

The same as the SIDE EFFECTS of each command's `--help` and README "Privacy":

| command | network | writes |
|---|---|---|
| `system`, `recommend`, `fit`, `search`, `types`, `info`, `gate`, `diff` (and the Python API) | none | none |
| `update` | GET / HEAD to GitHub (ComfyUI templates, releases), the Hugging Face Hub API and model URLs (HEAD; for main weight files also a ranged GET of the file header only, never the weights) | the cache `db.json` (or `--output`), `file_headers.json` in the cache |
| `update --fields-only` | GET of the Hugging Face Hub API for the cards whose `language` decides `languages` (the main files' repos and their `license_from`); for the hand-added models also HEAD, ranged GET of the file header and the upstream cards' API; nothing with `--offline` | the cache `db.json` (or `--output`), `file_headers.json` in the cache |
| `update --inspect-only` | ranged GET of the headers of the main weight files on Hugging Face, GET of the ComfyUI template JSONs; nothing with `--offline` | the cache `db.json` (or `--output`), `file_headers.json` in the cache |
| `update --from-release` | GET `db.json` and `db.json.sig` from this project's `data-latest` release | the cache `db.json` (or `--output`), `release_state.json` |
| `bench` | only the ComfyUI `--url` (+ official template files on GitHub if ComfyUI lacks them) | a measurement JSON in the cache (or `--output`); with `--submission` also the submission JSON |
| `contrib` | none (genmodelfit never sends a submission; you open the pull request yourself) | `make`: the submission JSON; `aggregate --output`: the community file; `show` / `validate`: none |
| `review` | none | `list`: none; `approve` / `reject`: the `--db` file in place; `reject` also appends to `review_rejected.txt` (next to `--db` unless `--rejected`) |
| `python -m genmodelfit.weekly merge-check` | none | none |
| `python -m genmodelfit.review publish SRC DST` | none | DST (the list without pending entries; data.yml signs it) |
| `python -m genmodelfit.review apply DECISIONS --db PATH` | none | the `--db` file; rejects also append to `review_rejected.txt` |
| `download` | GET of the model URLs in the list (https; redirects only to Hugging Face / GitHub hosts); none with `--list` | the files under the destination (`<file>.part` while downloading); nothing with `--list` |

No telemetry; nothing about the PC or the user is sent. HTTP uses `fetch.make_session()`: User-Agent
`genmodelfit/<version> (+https://github.com/laxshmate/genmodelfit)`, proxy / CA environment variables honored, `~/.netrc`
credentials never attached (tests/test_privacy.py).

## System

| key | type | meaning |
|---|---|---|
| `os`, `cpu` | str | |
| `cpu_cores` | int | logical threads |
| `ram_total_gb`, `ram_available_gb` | float | GiB |
| `gpus` | Gpu[] | NVIDIA GPUs confirmed by `nvidia-smi`, and (0.7.0, #22) AMD / Intel / Apple GPUs that the support table in `docs/hardware.md` accepts when no NVIDIA GPU is usable |
| `best_gpu` | Gpu \| null | the GPU used for judging: the largest-VRAM GPU whose `backend` is `cuda`; without one, a `rocm` / `xpu` / `mps` GPU (dedicated VRAM first, then the largest). null → judged as CPU |
| `disk_free_gb` | float \| null | free space at the home directory (or `--disk-path`) |
| `overrides` | str[] | `vram` / `ram` when overridden |
| `swap_total_gb` | float \| null | current page file (Windows) / swap (Linux, zram excluded) size |
| `swap_max_gb` | float \| null | how far it can grow: the fixed page file maximum, 0 for none, the Linux swap size. null = unknown or system managed |
| `swap_auto` | bool \| null | true = system-managed page file (grows) / macOS. false = fixed size or none. null = unknown |
| `commit_limit_gb`, `commit_used_gb` | float \| null | Windows commit limit (RAM + page file) and current commit (`GlobalMemoryStatusEx`) |
| `other_gpus` | Adapter[] | display adapters of any vendor. AMD / Intel / Apple ones are classified (`usable`, `reason`); usable ones also appear in `gpus` when no NVIDIA GPU is usable. Empty unless checked |
| `other_gpus_checked` | bool | whether `other_gpus` was looked up: by default only when `nvidia-smi` found no NVIDIA GPU and `--vram` is not given; always with `system --all-gpus` |
| `notes` | str[] | what detection found or could not find (Japanese): why a GPU is judged as CPU, old driver, non-NVIDIA GPUs, page file, WSL / container limits |

With `--ram`, the swap / commit values are null (another PC is assumed). Detection never fails: unknown values are
null and the reason is in `notes`. Host name, user name, GPU UUID, PCI bus id and device instance ids are never
read into the output. Details and sources: `docs/hardware.md`.

Gpu:

| key | type | meaning |
|---|---|---|
| `name`, `vendor` | str | `vendor` is `nvidia` / `amd` / `intel` / `apple` |
| `vram_total_gb` | float | 0 when nvidia-smi does not report it (then `backend` is `none`). For `memory` `shared` / `unified`, the memory the GPU is taken to use (`docs/hardware.md`「共有メモリ」) |
| `vram_free_gb`, `vram_used_gb` | float \| null | free now / used now by other apps (screen, browser, …). null when overridden or not reported |
| `driver` | str \| null | driver version as nvidia-smi prints it |
| `backend` | str | `cuda` / `rocm` (AMD) / `xpu` (Intel) / `mps` (Apple silicon): used for judging. `none`: not used (generation not supported by any PyTorch CUDA build, driver too old for all of them, or VRAM unknown; reason in `notes`). `rocm` / `xpu` / `mps` are new in 0.7.0 (#22) and always estimates (levels capped at Good) |
| `compute_capability` | str \| null | e.g. `"8.6"` |
| `compute_capability_source` | str \| null | `nvidia-smi` / `name` (guessed from the GPU name when nvidia-smi is too old) |
| `arch` | str \| null | `kepler` / `maxwell` / `pascal` / `volta` / `turing` / `ampere` / `ada` / `hopper` / `blackwell` (`fermi` / `unknown`) |
| `torch_cuda` | str[] | PyTorch CUDA builds (`cu132` … `cu126`) that run on this GPU with this driver, newest first. Informational |
| `compute_dtype` | str \| null | what ComfyUI computes in on this GPU by default: `fp16` / `bf16` / `fp32` (fp32 doubles working memory). null when `backend` is `none` or the generation is unknown |
| `bf16`, `fp8` | bool \| null | bf16 / fp8 compute support as ComfyUI sees it (null as above). Without bf16, model families that ComfyUI never runs in fp16 compute in fp32 (`docs/hardware.md`) |
| `notes` | str[] | reasons and cautions for this GPU |
| `memory` | str | (0.7.0) `dedicated` (own VRAM) / `shared` (integrated GPU or APU using system RAM) / `unified` (Apple silicon). Shared and unified GPUs never offload (`GPU+RAM`) |
| `gfx_target` | str \| null | (0.7.0) AMD LLVM target, e.g. `gfx1100` |
| `carveout_gb` | float \| null | (0.7.0) for `memory: shared`, the VRAM the BIOS sets aside (not part of `ram_total_gb`); the RAM check counts only GPU use above it. null otherwise |

Adapter (`other_gpus`): `name`, `vendor` (`nvidia` / `amd` / `intel` / `apple` / `unknown`), `vram_gb` (Windows: registry
`HardwareInformation.qwMemorySize`, else WMI `AdapterRAM` below 4 GiB; Linux AMD: sysfs; Intel Arc on Linux: the
smaller model's VRAM from the name; null if unknown), `vram_source` (`registry` / `wmi` / `sysfs` / `name` / null),
`source` (`wmi` / `lspci` / `system_profiler`). Added in 0.7.0 (#22): `integrated` (bool | null), `usable` (bool | null;
null for NVIDIA, which `nvidia-smi` decides), `reason` (Japanese sentence: why it is or is not used), `shared_gb`
(Linux amdgpu GTT, sysfs `mem_info_gtt_total`), `needs` (string | null: what would make the GPU usable, currently only
`"Windows 11"` when a known Windows 10 build was the only reason; the table header says `needs Windows 11 for generation`).

## Data status / 一覧の新しさ

Whether the model list in use is still fresh enough for auto-recommendation. There is no separate `status` command:
`genmodelfit system --json` puts it in `database.status`, `update --json` (and `update --from-release --json`) in
`status`, and Python returns it from `gmf.data_status(db=None, *, now=None, comfyui_version=None)` (a frozen dataclass
`DataStatus`; `.to_dict()` gives the same keys). The text output of `system` / `update` shows it as the Data lines.
Network is never touched.

使っている一覧が、自動のおすすめに使えるほど新しいか。別のコマンドは作らず、`system --json` の `database.status`、
`update --json` の `status`、Python の `gmf.data_status()` で返す。

| key | type | meaning |
|---|---|---|
| `source` | str | `bundled` (shipped in the package) / `cache` (written by a local `genmodelfit update`) / `release` (taken in by `update --from-release`: the cache whose `generated_at` equals the one recorded in `release_state.json`) / `path` (`--db`, `db=`) |
| `generated_at` | str \| null | when the list was made (as written in it) |
| `heads_checked_oldest`, `heads_checked_newest` | str \| null | oldest / newest `head_checked_at` over all files (UTC, `Z`) |
| `max_head_age_days` | int \| null | the gate's limit (`--max-head-age-days`, default 30; null = no limit) |
| `expired` | bool | **true when at least one entry lost auto-recommendation only because its HEAD checks are too old** (`auto_count < auto_count_if_fresh`). Apps should then stop showing "new models" from genmodelfit and refresh with `update --from-release` (or `update --heads-only`) |
| `expires_at` | str \| null | the earliest moment an auto entry expires: its oldest HEAD check + `max_head_age_days`. Exactly at this time it is still valid (the gate drops a check when its age is **greater** than the limit) |
| `models` | int | entries in the list (one per workflow) |
| `auto_count` | int | entries that are auto-recommendable now (default gate policy: content `sfw` only; the ComfyUI version is `--comfyui-version` / `comfyui_version`, else the one recorded by update) |
| `auto_count_if_fresh` | int | the same with the HEAD age limit removed |
| `checked_at` | str | the time used for the judgement (UTC) |

The gate's `url_reachable` check itself decides both counts (`gate.evaluate_offline`, run once with the policy and once
with `max_head_age_days=None`), so `expired` and the gate never disagree. The bundled list's HEADs were checked on
2026-10-20T10:33:07Z (0.1.1), so with no update it expires after 2026-11-19T10:33:07Z and `auto_count` becomes 0.
判定は gate と同じ関数で行う。`expired` が true なら、HEAD の確認が古いせいで自動のおすすめから落ちたものがある。

## recommend / fit / search

The three commands print the same shape (Issue #17, modeled on llmfit's `recommend --json`):

| key | type | meaning |
|---|---|---|
| `filters` | object | values actually used: `modality` (str[] \| null), `content`, `min_fit`, `license` (str[] \| null), `include_unknown_license` (bool), `task` (str[] \| null: the names given), `all_workflows`, `limit`; `recommend` adds `per_type`, `auto_only`, `include_tools`; `search` adds `query` |
| `matched` | int | rows that passed every filter, before `limit` / `per_type` (models; workflows with `--all-workflows`) |
| `hidden` | object | why the other entries are not shown: `{reason: count}`, only reasons with a count (below) |
| `groups` | object[] | per type, in the order image, video, music, sfx, tts, 3d (only the `-m` types when given): `{modality, matched, shown}` |
| `models` | Fit[] | best first (order below) |

**One item per model** (below, "Models"): entries that differ only in precision or in workflow (text-to-image, edit,
LoRA, …) are collapsed into one item; the others are listed in `workflows` / `variants`. `--all-workflows`
(`all_workflows=True` in Python) gives one item per list entry as before.

`fit` (0.1.1, Issue #28; Python `gmf.fit_list()`): by default only the models that run (`--fit runnable` = perfect /
good / marginal), without utility tools, unknown-license and uncensored models; `--all` (`show_all=True`) is `--fit all`
plus tools and unknown licenses (uncensored still needs `--content all`). `--fit perfect|good|marginal` shows exactly
that level (other runnable levels are counted as `hidden.other_fit`); `--fit` and `--min-fit` cannot be combined
(`--perfect` = `--fit perfect`). `fit` and `search` print **one table** sorted by `--sort` (`sort=`):

| sort | order |
|---|---|
| `type` (default) | type (image, video, music, sfx, tts, 3d) → fit level (perfect → good → marginal → unknown → too_tight) → official before community within the same level → `score` desc → model name |
| `score` | runs (perfect / good / marginal) → unknown → too_tight, then fit level → official first → `score` desc → name: one ranking across types |
| `vram` / `disk` | runs → unknown → too_tight, then `memory_required_gb` / `disk_size_gb` ascending (null last) |
| `name` | runs → unknown → too_tight, then model name |

The fit level always comes before the score (the score weights the level most, but the other components could invert
two levels). Auto-recommendable models are not moved up in `fit` / `search` (the Verified column shows it). JSON `models`
follows the same order; `filters` of `fit` adds `fit`, `sort`, `all`, `include_tools`, of `search` `fit`, `sort`, `all`.
Table columns (all commands): `Type | Status | Score | Censored | Model | Use | Prec | VRAM | Mem% | Mode | Disk | License | Verified`;
when the terminal is narrow, the Model column shrinks first (down to 24 characters), then Disk, Mem%, License, Verified,
Prec, Use are dropped in that order (Type, Status, Score, Censored and Model never). **Verified** (named Auto up to 0.5.0) is the JSON `auto`:
`yes`, or the first failed gate check (`held_by[0]`, shortened); the JSON fields keep their names (`auto`, `held_by`,
the check ids). **Censored** (0.6.0, in place of the `[U] ` name prefix) is `Y` (`content: sfw`) or `N`
(`content: uncensored`), never blank; the column is
shown only when uncensored rows can appear (`--content all|uncensored`; a table built from Python without a content
filter shows it when a row is uncensored) and is omitted with the default `sfw`. The legend then says
`Censored: N = safety filter removed / adult model`. JSON is unchanged (`content`).
Censored の列は無検閲の行が出うるときだけ出す(既定の sfw では出さない)。
`search` shows every fit level by default (`--fit` narrows).

**Unknown and not applicable / 「?」と「-」** (0.2.0, #30). In the table, `info` and the legend, `?` means the value
is **unknown** and `-` means it **does not apply**:

| column / line | `?` (unknown) | `-` (not applicable) |
|---|---|---|
| Status | size unknown, nothing judged (`fit_level: "unknown"`) | — |
| Prec / `info` precision | precision unknown (`precision: null`) | — |
| VRAM | size unknown | runs on the CPU (no VRAM used) |
| Mem% | size unknown | no GPU although one is needed (nothing to divide by) |
| Mode | size unknown (`run_mode: "unknown"`) | does not run (`run_mode: "none"`) |
| Disk / License | size / license unknown (`null`) | — |
| Use / `info` tasks | no task known | — |
| Verified | — | the gate was not evaluated |
| `info` file folder | — | the file has no ComfyUI folder (Hugging Face repo) |

JSON: an unknown **value** is `null` (`precision`, `license`, `memory_required_gb`, `disk_size_gb`, and the
`estimate` VRAM / RAM figures when `unknown_size`); an unknown or not-applicable **state** is a documented enum value
(`fit_level: "unknown"`, `run_mode: "unknown"` vs `"none"`), with the labels `?` / `-` in `fit_label` /
`run_mode_label`. 分からない値は null、状態は決まった値(unknown / none)で分ける。
`memory_available_gb` is this PC's VRAM (RAM for CPU) even when the size is unknown, because that amount is known.

**Breaking in 0.2.0** (JSON, size-unknown models only): `run_mode` `"none"` → `"unknown"`, `run_mode_label` `-` →
`?`, `estimate.vram_gb` / `vram_offload_gb` / `ram_gb` / `vram_sequential_gb` `0.0` → `null`, `memory_available_gb`
`null` → the PC's amount. See CHANGELOG.md「Breaking (JSON)」for migration. `schema_version` stays 1.

`recommend`: without `-m` and `--task` it shows **one section per type** (image, video, music, sfx, tts, 3d), the top
`-n` (default 3) of each (`filters.per_type`, `limit` null); a type with nothing runnable prints one line in text mode
and has `matched: 0` in `groups`. With `-m` or `--task` it is one list, `-n` default 10. In JSON `models` is ordered by
type, then by the order below. Other defaults: `min_fit` `marginal`, utility tools hidden, models with unknown size never
shown. `fit` shows the runnable levels by default and `search` every level (`--fit` / `--min-fit` to narrow), with no
limit; models with unknown size (`fit_level: "unknown"`, `?`) come after marginal ones of their type (`--fit all`). `search <query>` matches a
part of the id or name (case-insensitive); entries that do not match are not counted in `hidden`. With no match it prints
`models: []` and exits 0 (text mode prints `(no match / 該当なし)`); a blank query exits 1.
`content` defaults to **`sfw`**: uncensored models appear only with `--content all` or `--content uncensored`.
既定は sfw。無検閲は `--content all|uncensored` のときだけ出す。

**Unknown licenses are hidden by default** (0.1.1, Issue #28): an entry where any file's license (the entry's license
when it lists no files) is unknown (`null`) is not shown by `recommend` / `fit` / `search` or by `gmf.recommend()` /
`gmf.search()`; it is counted as `hidden.license_unknown`. `--include-unknown-license` (Python
`include_unknown_license=True`) shows them. `info <exact id>` always shows the entry.
ライセンスが分からないファイルを含むものは既定で出さない(`--include-unknown-license` で出す。`info <ID>` は出す)。

`hidden` reasons, each entry counted once under the first reason that applies, in this order:
`modality` (other `-m`), `uncensored` / `sfw` (by `--content`), `tools` (recommend / fit without `--include-tools` / `--all`),
`task` (other uses than `--task`), `license` (`--license`), `license_unknown` (some file's license is unknown; without
`--include-unknown-license`), `size_unknown` (recommend, or any `--min-fit` other than `too_tight`), `too_tight`,
`below_min_fit` (perfect / good / marginal below `--min-fit`), `other_fit` (a runnable level not selected by `--fit`),
`held` (`--auto-only`), `same_model` (collapsed into
another row of the same model;
the text header says "N more workflows merged into their models").
**`matched` + the sum of `hidden` = the number of entries looked at** (every entry of the list; for `search`, the entries
whose id or name matched). Tested in `tests/test_cli_json.py`.

`--license a,b`: every file's license (`ModelFile.license`; the entry's `license` when it lists no files) must be in the
list (case-insensitive). An unknown license (`null`) never matches.

**Order of `recommend` / recommend の並べ方** (`gmf.recommend()`): models that run (`fit_level` is not `too_tight`; `unknown` after `too_tight`) → `auto: true` →
**runs without offload** (perfect or good on `GPU` / `GPU~`, one tier like the row score, 0.1.1: the precision upgrade
to a `good` bf16 row must not move a model below where its `perfect` int8 row was) → marginal → official before
community templates (`community`) **within the same tier** → `score` (desc) → `id`. Until 0.1.0 community templates
came after every official row of all levels.
動くもの(判定できないものは一番後ろ)→ 自動でおすすめに出せるもの → 逃がさずに載るもの(perfect / good)→ marginal →
同じ段階の中で公式(利用者のテンプレートは後)→ 点数の高い順 → ID。

### Models / モデルのまとめ方

Like llmfit: a different **size** is a different model, a different **precision** is the same model (chosen
automatically), and **workflows** are uses of that model. `group_key` is built from the main weights
(`src/genmodelfit/groups.py`):

1. Main files = role `main` (by the estimator's rule: ComfyUI folder → `role` → HF folder → name). Text encoders
   (umt5, …) and VAEs are never the key. With several main files, the largest one and those ≥95% of its size (a pair
   such as Wan 2.2 high / low noise); without any, the largest file that is not a text encoder / VAE / clip_vision;
   without files, the entry itself. Super-resolution, upscaler and refiner files (`sr`, `upscal…`, `refiner` in the
   name) are not main weights when another main file exists (Hunyuan Video 1.5 lists its 1080p SR file in both the
   720p t2v and i2v workflows).
2. From each file name (the repo name instead for generic names such as `model.safetensors`), drop the extension,
   shard numbers, GGUF quant tags (`Q4_K_M`, `Q8_0`, …) and precision / packaging words (`fp16`, `bf16`, `fp8`,
   `e4m3fn`, `scaled`, `mixed`, `int8`, `int4`, `nf4`, `nvfp4`, `w4a8`, `convrot`, `kj`, `comfy`, `transformer`, `only`,
   `pruned`, `aio`, `high`, `low`, `noise`, …). Keep version, size and variant words (`2.1`, `1.5`, `14b`, `1.3b`,
   `turbo`, `dev`, `distilled`, `base`, `edit`). `wan2_1` is read as `wan 2.1`, `v1.5` as `1.5`, `ace_step` as
   `acestep`. If no size word is left, `params_b` is added (`4b`).
3. Words are sorted and joined; the key is `<modality>:<content>:<words>`, so different types or content marks never
   share a model. Examples: `z_image_turbo_bf16` and `z_image_turbo_int8_convrot` → `image:sfw:image turbo z`;
   `wan2.1_t2v_1.3B_fp16` → `… 1.3b 2.1 t2v wan`, `wan2.1_t2v_14B_fp8_scaled` → `… 14b 2.1 t2v wan` (different).
   The test table is `tests/test_groups.py`.

After the per-entry key, the whole list is looked at (`groups.build_index`):

- An entry with two main files of different names (FLUX.2 klein 4B text-to-image loads both `klein-base-4b` and
  `klein-4b`) joins the model of one of them if that model exists on its own: the one without `base`, then by name.
  Otherwise the names stay joined with `+` (Ideogram 4 with its unconditional pair).
- Entries whose **whole set** of main-file sha256s is equal are the same model (the same file under another name or
  repo, e.g. Stable Audio Open 1.0 in the template and on HF). Sharing only some main files is not enough. Upstream records whose file name and URL / hash disagree
  are corrected first (`update.UPSTREAM_FILE_FIXES`, checked with an anonymous HEAD).
- `params_b` is added to the key only for generic HF file names (`model.safetensors`) without a size word; template
  `params_b` comes from the title and can be the language model's size (ACE-Step 1.5 "4B LM").

Precision (`ModelEntry.precision`) is read from the main file used for the key (`groups.model_precision_source`;
`update` uses the same function). The first source that tells wins, and it is kept in `ModelEntry.precision_source`
(#30, 0.2.0):

1. `header`: the file's `header_precision` when it was read. The name wins (`filename`) for packed formats the header
   cannot tell (`w4a8`, `nvfp4`, `mxfp4`, `fp4`, `nf4`, `int4`) and for GGUF files whose name states the quantization;
2. `filename`: the file name, e.g. `w4a8` for the 10Eros W4A8 file;
3. `hf_api` / `config`: `ModelFile.repo_precision`, read from the hosting Hugging Face repo when neither the header nor
   the name tells (`repo_precision.py`): the Hub API's `safetensors.parameters` when one dtype holds ≥ 90 % of the
   parameters **and** element count × bytes per element is within 5 % of the entry's main safetensors files in that
   repo (shards added; a repo with other weights beside the main one does not describe it; safetensors main files only;
   readable for gated repos), else `torch_dtype` / `dtype` of the
   `config.json` in the main file's folder (only when the API lists it and the repo is not gated);
4. `workflow`: `ModelEntry.workflow_precision`, inferred from the official template's main loader node
   (`workflow.loader_precision`): the node that loads the main file (or, for templates without a file list, the
   first of `workflow.MAIN_LOADERS`: `UNETLoader`, `CheckpointLoaderSimple`, `WanVideoModelLoader`, `SeedVR2LoadDiTModel`,
   `DownloadAndLoadGIMMVFIModel`, …). A load-time cast to fp8 (`weight_dtype` / `quantization` = `fp8_*`) → `fp8`; else
   the precision word of the loaded file name; else an explicit precision widget (`fp32`, `bf16`, `fp16`, `fp16_fast`).
   `default` / `auto` / `disabled` say nothing. LoRA, CLIP and VAE loaders are never used.

A precision read from the header is never replaced by a later source. null = unknown (`precision_source` null).

**Display name** (`model_name`, the table's Model column, `groups.display_name`): candidates are each member's
name before `:`, after `:` and both joined, without precision / packaging words (`Int8`, `FP16`, `F16`, `ConvRot`, `KJ`,
`AIO`, …) and use phrases (`Text to Video`, `Depth Estimation`, `Image Upscale`, …). The candidate containing most words
of the key wins (official templates before community templates and HF repos, then the shorter); it is cut after the
last key word it contains, and missing size / version / variant words of the key are added (`ACE Step v1 3.5B`,
`Hunyuan Video 1.5 720p T2V`, `Image Upscale: SeedVR2 7B Int8` → `SeedVR2 7B`). A version added this way gets a `v`
when it follows a number (`Kokoro-82M v1.0`); `Wan2.1` is written `Wan 2.1`. Models whose key words are the same
except the type share one name plus the type (`SAM 3.1 (3D)` / `(image)` / `(video)`; no suffix on the only type
that has official templates). A name without digits gets its one-digit version when a sibling differs only by a
version (`MoGe 2` beside `MoGe 3`). File-less templates join the one model of the same type whose key contains all
words of their model name (`Frame Interpolation: FILM` → `FILM`); otherwise they are their own model named
`<name> (files unknown)`. Remaining clashes get key words (`fl2va` → `first/last frame`), then `#2`. A small table
`groups.NAME_OVERRIDES` (entry id → name; community templates and a few HF repos whose titles carry no model name)
overrides the result; `tests/test_groups.py` checks it. Ids never appear in names. `genmodelfit info <model name>` (case-insensitive) shows the model's best workflow for this PC and the
others.

Which entry is shown for a model (`recommend.variant_key`): among the entries that passed the filters,
0. official before community templates (`community`: the template's `username` is not ComfyUI) — a community
   workflow never stands for a model that has an official one,
1. the use: `gen` → `edit` → `i2v` → `lipsync` → `motion` → `voice` → `upscale` → `tool` (with `--task`, among the
   matching uses) — so a tool or upscale workflow that shares the weights never stands for the model,
2. **entries that run on the GPU without offload** (`fit_level` `perfect` or `good` and `run_mode` `gpu`, i.e. Mode
   `GPU` or `GPU~`; `recommend.runs_clean`) first, and **among them the higher precision first**, then the better
   level — like llmfit, the best quality that fits (0.1.1, Issue #28),
3. otherwise (no entry runs without offload): the better fit level, then no offload (`GPU` / `GPU~` before `GPU+RAM`
   before CPU — same level, faster), then the higher precision,
4. the order above.
Precision order: bf16 / fp16 / fp32 > fp8 > int8 > GGUF Q8 > Q6 > Q5 > Q4 / 4-bit > lower; unknown precision counts as
full precision. So on 24 GB Z-Image Turbo shows bf16 (`good`, `GPU~`) rather than int8 (`perfect`); on 8 GB it shows
int8 (swapped, `marginal`) rather than bf16 (offload, `marginal`); with `--task edit` the representative is an edit
workflow. **The row's `score` (and `score_components`) is the highest score among the model's entries that passed the
filters and have the same use and the same tier as the shown entry**, where the tier is the fit level, except that all
entries that run without offload (perfect / good on `GPU` / `GPU~`) are one tier (`recommend.row_tier`); everything else
on the row is the shown entry's. The order uses that score and the shown entry's `auto`.
Adding VRAM never lowers a row's score and never moves a row out of the "runs without offload" tier, and the shown
precision only goes up within it; but the row's `fit_level` can go from `perfect` to `good` when a higher precision
becomes runnable (on the bundled list only LTX 2.3 10Eros: nvfp4 `perfect` at 48 GB → bf16 `good` at 64 GB).
Tested in `tests/test_groups.py`.
逃がさずに載るものがあれば、その中で一番精度の高いものを出す。無ければ段階を優先する。

On the bundled list (329 entries, 2026-10-21) this gives 210 models; 55 of them have more than one entry.

### Uses / 用途(`--task`, `genmodelfit types`)

The six big types stay the top level. Uses are sub-divisions inside a type, from the entry's `tasks`:

| use | English | 日本語 | task codes |
|---|---|---|---|
| `gen` (alias `generate`) | generate from text (3D: from an image) | 文章から作る(3D は画像から) | t2i, t2v, t2a, tts, t23d, i23d |
| `edit` | edit / transform / control | 編集・変換・形の指定 | i2i, inpaint, outpaint, relight, control, v2v, a2a |
| `i2v` | image to video | 画像から動画 | i2v, flf2v, ref2v |
| `lipsync` | lip sync / audio to video | 口の動き・音から動画 | lipsync, a2v |
| `motion` | motion control | 動きの指定 | motion |
| `voice` | voice conversion | 声の変換 | vc |
| `upscale` | upscale / restore | 高解像度化 | upscale |
| `tool` | utility tool (depth, pose, mask, …) | 補助ツール | tool, estimate, matting, interp, segment, separate |

`--task` takes use names, aliases or task codes, comma-separated (`--task edit`, `--task i2v,lipsync`, `--task t2i`).
An entry matches when one of its tasks is in the set. `use` (JSON) / the table's Use column is the matching use, else
the first of tool, upscale, lipsync, motion, voice, edit, gen, i2v that the entry has.
Utility tools (`upscale` and `tool`, plus Image / Video Tools templates whose title is a depth / normal / line-art /
pose / geometry map or a segmentation, any audio separation, and MoGe / geometry estimation even under 3D Model) are hidden by `recommend` unless `--include-tools` or a
tool use is given in `--task`.

`genmodelfit types --json`: `filters` (`content`), `types` (`[{modality, label, label_ja, models, workflows, uses: {use:
workflows}}]`; `models` = number of models as above, `workflows` = list entries; an entry with two uses counts in both),
`uses` (`[{use, label, label_ja, tasks}]`), `aliases`.

### Fit (one item of `models`)

Python `Fit.to_dict()` returns exactly this dict (`gmf.recommend()` / `gmf.fit_list()` / `gmf.search()` give the same list as the CLI).

| key | type | meaning |
|---|---|---|
| `id` | str | template name (`image_z_image_turbo_int8`) or `hf:<owner>/<repo>` |
| `name`, `modality`, `tasks` | | modality ∈ `image` `video` `music` `sfx` `tts` `3d`; `tasks` is a list (`t2i`, `i2v`, …) |
| `content` | str | `sfw` or `uncensored` (see README) |
| `source` | str | `comfyui-template` / `huggingface` / `manual` |
| `runtime` | str \| null | how the model runs: `comfyui` / `piper` / `irodori-tts` / … (`ModelEntry.runtime`, 0.5.0); null = unknown |
| `languages` | object | `{"prompt": [...], "speech": [...]}` ISO 639-1 (`ModelEntry.languages`, 0.5.0); a missing key = unknown, `[]` = none (an image-to-3D model takes no prompt; a music model without vocals has no speech) |
| `description` | object | `{"ja": "...", "en": "..."}` hand-written, only for some models (`ModelEntry.description`, 0.5.0); `{}` = none |
| `fit_level` | str | `perfect` / `good` / `marginal` / `too_tight` (= `estimate.fit`), or `unknown` when the size is unknown and nothing was judged (0.1.1; until 0.1.0 these were `too_tight`) |
| `fit_label` | str | `Perfect` / `Good` / `Marginal` / `Too tight` / `?` |
| `run_mode` | str | `gpu` / `offload` / `cpu` / `none` (does not run) / `unknown` (size unknown, not judged; 0.2.0 — until 0.1.1 these were `none`) (= `estimate.run_mode`) |
| `run_mode_label` | str | `GPU` (all on GPU) / `GPU~` (parts swapped stage by stage) / `GPU+RAM` (offload) / `CPU` / `-` (does not run) / `?` (size unknown) |
| `level_cap` | str \| null | the run mode caps the level (0.1.1): `sequential` (`GPU~`, at most `good`), `offload` (`GPU+RAM`, at most `marginal`), `cpu` (at most `marginal`); null for all on GPU or not running. So "Good, 54 %, GPU~" is not a contradiction |
| `score` | float | 0–100 (below) |
| `score_components` | object | `fit`, `headroom`, `readiness`, `basis`, `popularity`, each 0–100 |
| `memory_kind` | str | `vram` (judged on the GPU) or `ram` (CPU) |
| `memory_required_gb` | float \| null | the figure the level was judged on: VRAM for all-in / swapping / offload (the table's VRAM column); RAM for CPU. When a GPU is needed but missing: the minimum VRAM. null when the size is unknown |
| `memory_available_gb` | float \| null | this PC's VRAM (or RAM for CPU); `0.0` when a GPU is needed but missing. Also set when the size is unknown (0.2.0; was null) |
| `utilization_pct` | float \| null | `memory_required_gb / memory_available_gb × 100` (the table's Mem%) |
| `gpu_memory_required_gb` | float \| null | (0.7.1, #58) the VRAM the model needs to run on this PC's GPU, also when `run_mode` is `cpu` (where `memory_required_gb` is RAM). Same value as `memory_required_gb` for every GPU run mode and for `too_tight`; for `cpu` it is the same figure `too_tight` uses, the smallest need among the load modes (the GPU+RAM offload figure `estimate.vram_offload_gb`; also for shared / unified memory GPUs, so the meaning does not change between `cpu` and `too_tight`), i.e. "the GPU would need at least this much". null when no GPU is used for judging (CPU-only PC, `gpu_usable=False`, unusable GPU) or the size is unknown. Additive; no other field changes |
| `ram_required_gb` | float \| null | system RAM needed (= `estimate.ram_gb`; null when the size is unknown) |
| `disk_size_gb` | float \| null | download size counted (one variant per part, = `estimate.disk_gb`) |
| `total_size_gb` | float \| null | size of all listed files |
| `license` | str \| null | strictest license of the files (`apache-2.0`, `mit`, `other`, …; null = unknown); within the same tier the one with more obligations (`apache-2.0` before `bsd-3-clause` before `mit`, `hf.OBLIGATION_RANK`) |
| `licenses` | (str \| null)[] | the distinct licenses of the files (null last = some file unknown) |
| `auto` | bool | can be auto-recommended (= `gate.auto`) |
| `held_reasons` | str[] | failed gate checks (= `gate.held_by`) |
| `community` | bool | a community template (the index's `username` is not ComfyUI; docs/sources.md) |
| `estimate_confidence` | str | `measured` / `estimated` (= `estimate.basis`; the finer level is `estimate.basis_detail`) |
| `notes` | str[] | reasons in Japanese (= `estimate.reasons`) |
| `files_count` | int | download files listed |
| `use` | str | the use shown (Uses below) |
| `precision` | str \| null | the chosen entry's precision (`ModelEntry.precision`: `bf16`, `fp8`, `int8`, `gguf-q8_0`, …; null = unknown, shown as `?`, ranked like full precision). Where it came from: `precision_source` (next row) |
| `precision_source` | str \| null | `header` / `filename` / `hf_api` / `config` / `workflow` (`ModelEntry.precision_source`, see "Models"); null = unknown (0.2.0) |
| `group_key` | str | the model this entry belongs to (Models below) |
| `model_name` | str | the model's display name (unique over the list; `info` accepts it) |
| `variants` | object[] | per precision, the best entry that passed the filters: `{precision, workflow_id, memory_required_gb, fit_level, auto, disk_size_gb}`, best precision first (always includes this entry) |
| `workflows` | object[] | **all** other entries of the same model, also those filtered out: `{id, name, tasks, use, precision, precision_source, fit_level, memory_required_gb, disk_size_gb, auto, matched}` (`matched`: passed the current filters), best first |
| `estimate` | Estimate | full estimate (below; `limits` with `--limits`) |
| `gate` | Gate | full gate result (below) |

### Score / 点数

`score = 0.45·fit + 0.15·headroom + 0.20·readiness + 0.10·basis + 0.10·popularity`, rounded to 0.1
(`recommend.WEIGHTS`). Each component is 0–100:

| component | formula |
|---|---|
| `fit` | perfect 100, good 70, marginal 35, too_tight 0, unknown 0 |
| `headroom` | `min(100, 100 × (1 − vram_gb / VRAM) / 0.5)`, 0 without a GPU, when the size is unknown or when the all-in VRAM (`estimate.vram_gb`) does not fit. Half of the VRAM free → 100 |
| `readiness` | 100 when `auto`; otherwise the sum of the passed checks' points: license 25, gated 15, official_workflow 15, custom_nodes 15, comfyui_version 10, url_reachable 5, sha256 5, content 5, size_known 5 (app_runtime 40 in place of the three ComfyUI checks; it never passes; reviewed 0) |
| `basis` | 100 when `measured`, 0 when `estimated` and for other users' submissions (`basis_detail: measured_community`; they never raise the score) |
| `popularity` | `min(100, 20 × log10(1 + usage))`; usage = template uses, or Hugging Face downloads / 100 (100 at 100k) |

The score never goes down when VRAM or RAM increases (the level never drops, `headroom` compares against the all-in
VRAM which does not depend on the PC, and `basis` only turns `measured` when a GPU is present). Tested in
`tests/test_invariants.py`. VRAM・メモリを増やしても点数は下がらない(試験あり)。

## Estimate

| key | type | meaning |
|---|---|---|
| `vram_gb` | float | peak VRAM when everything stays on the GPU / 全部載せ |
| `vram_sequential_gb` | float \| null | peak VRAM when ComfyUI swaps parts between stages (text encoder → model → VAE). When the workflow runs several diffusion models one sampler after another (Wan 2.2 14B high_noise → low_noise; `ModelEntry.sampler_stages`), the model stage counts only the largest of them (with its LoRAs) / 順に載せ替え |
| `vram_offload_gb` | float | peak VRAM when part of the weights stays in RAM (lowvram / dynamic VRAM); several diffusion models run in turn are counted one at a time as above / 逃がす |
| `ram_gb` | float | system RAM needed for the chosen run mode: GPU (all-in / swapping) = 3 + largest stage's weights; offload / CPU = 3 + all weights. Short but within RAM + page file (fixed page file / swap; capped at 1.5× RAM; 1.5× RAM when system managed or unknown) → `marginal` (page file, slow); more → `too_tight` |
| `disk_gb` | float \| null | size of the files counted (one variant per part; see `docs/accuracy.md`) |
| `basis` | str | **`measured`** (from a real run) or **`estimated`** (formula). Always present. Stays two-valued for compatibility; the finer level is `basis_detail` |
| `basis_detail` | str | (#34) `measured` (maintainer / this PC's bench measurement at the same settings) > `measured_community` (other users' submissions aggregated by `contrib aggregate`, from **one** submission, same settings; the count is `sample_count`) > `calibrated` (shifted by a measurement at other settings) > `estimated` (formula only). `models.BASIS_DETAILS`. Since 2026-10-22 (like llmfit) community measurements are used in **both directions** (they can lower or raise the needs, and `measured_community` scores like `measured`), bounded only physically: the all-in value is never below weights + fixed overhead (zero working memory), never above the measuring GPU's VRAM × 1.02 + 0.25 GiB, and a value pinned at the GPU limit is a lower bound. Spread to other settings, the ratio measured ÷ formula is clamped to [0.6, 1.4] (`estimate.COMMUNITY_CALIBRATION_RANGE`; llmfit uses [0.05, 3.0] for speed). Among measurements at the same settings, one on a GPU with this PC's name comes first, then the maintainer's / this PC's, then other users'. The maintainer's `torch_reserved` rows stay lower bounds (only raise). When a bound was applied the value is not the measurement: `basis` is `estimated` while `basis_detail` still says `measured_community` (or `calibrated`). When `basis` is `measured` and the measurement is not capped at the GPU limit, `vram_gb` equals the measured value. docs/accuracy.md「ほかの利用者の実測」 |
| `fit_from` | str | (0.7.0, #26) `estimate` (the level comes from the estimate) or `measurement`: a successful standard measurement at the same settings on a GPU of the same vendor whose VRAM is at most this PC's (within 3%: nominal 16 vs reported 15.9) sets the level to at least Good — Perfect only when its peak is within 60% of this VRAM and it was neither pinned at the GPU limit nor a community group with an OOM report; a `torch_reserved` peak gets `estimate.TORCH_RESERVED_CONTEXT_GB` (0.5 GiB, the CUDA / HIP context it leaves out) added before the bands and the fit check (0.9.0; before, it never allowed Perfect). The vendor of a measurement is its `gpu_backend` when recorded, else guessed from the GPU name. The measured peak must also fit this PC's VRAM. Other users' measurements count only when the GPU name is this PC's (since 2026-10-22; never similar GPUs), and a measurement from another vendor's GPU only calibrates (`basis: estimated`). AMD / Intel / Apple GPUs are capped at Good unless such a measurement of this GPU (same name, same settings) allows Perfect. The run mode is then `gpu` and the reasons say which decided |
| `measured_peak_vram_gb` | float \| null | (0.7.0) that measurement's peak VRAM when `fit_from` is `measurement` (the table's VRAM column shows it with `m`) |
| `sample_count` | int \| null | (2026-10-22, optional; like llmfit) number of measurements behind the value: for other users' measurements the submissions used for the median (`n_used`), 1 for the maintainer's / this PC's. null when no measurement was used (`basis_detail` `estimated`). `info` prints it as `n=3`; the table's VRAM column marks other users' measurements with `c` (the maintainer's with `m`) |
| `measured_on` | str \| null | GPU of the measurement when `basis` is `measured` (null when judged without a GPU) |
| `fit` | str | the four levels below, or `unknown` (size unknown, not judged) |
| `run_mode` | str | `gpu` / `offload` / `cpu` / `none` (does not run) / `unknown` (size unknown) |
| `reasons` | str[] | human-readable reasons (Japanese) |
| `settings` | object \| null | settings actually judged (unset items filled from the measurement, else per-modality defaults) |
| `limits` | Limits \| null | largest settings that still reach `good` on this PC. Set by `info`, and by `fit` / `recommend` with `--limits` |
| `vram_free_short` | bool | judged to run on the GPU, but the VRAM free right now (`nvidia-smi` `memory.free`) is below what the chosen run mode needs: close other apps. The level stays based on total VRAM. Always false when the free VRAM is unknown (`--vram`) |
| `unknown_size` | bool | the weight size is unknown, so nothing was judged: `fit` is `unknown` (shown as `?`, never "Too tight"), `run_mode` `unknown` (0.2.0; was `none`), the VRAM / RAM values are `null` in JSON (`Estimate.to_dict`; the Python attributes stay 0.0), `limits` null, first reason 「大きさが分からないため判定できません」. `recommend` drops these; `fit` / `search` list them last |

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

Fit levels / 段階 — by the **utilization of the memory pool of the run mode** (required / available; VRAM on the
GPU, RAM on the CPU), as in llmfit (`docs/how-it-works.md`, "Fit analysis"; adopted 2026-10-20):

| utilization | level |
|---|---|
| ≤ 60% | `perfect` |
| ≤ 85% | `good` |
| ≤ 98% | `marginal` |
| > 98% | `too_tight` |

Caps per run mode: all on the GPU (`GPU`) none; parts swapped stage by stage (`GPU~`) at most `good`; offload to RAM
(`GPU+RAM`) and CPU at most `marginal`. The run mode with the best level is used (ties: all-in → swapped → offload).
The RAM check is applied afterwards (short but within the page file limit above → at most `marginal`, more →
`too_tight`). On GPUs where ComfyUI computes in fp32 (`compute_dtype: "fp32"`: Maxwell, some Pascal, GTX 16xx …) the
working memory and the VAE weights are doubled before judging (`docs/hardware.md`, "精度"). A model
that does not fit a (small) GPU even with offloading is also judged as if there were no GPU (small CPU-capable models are
`marginal`), and the better result is used, so adding VRAM never lowers the level. Models whose size is unknown
(`unknown_size: true`) get `unknown` instead of a level. Constants: `estimate.FIT_BANDS`.

**Several diffusion models in one workflow / 本体が複数のワークフロー** (0.1.1, Issue #28): ComfyUI runs the samplers one
after another, and when the next sampler loads its model, `load_models_gpu` calls `free_memory`, which unloads models
that are loaded but not used now, as much as needed (ComfyUI commit `b1a247c9cf9dbfda4b2c0a0c1d50950667eeaa0c`:
`comfy/samplers.py:1241` → `comfy/sampler_helpers.py:201` → `comfy/model_management.py:1001`, unloading at `:905` /
`:919`; dynamic-VRAM models are loaded on demand, `:914`). So the swapped (`GPU~`) and offload figures count one
sampler's set at a time: that model + the files on its MODEL input (LoRAs) + the denoise-stage files shared by all
(ControlNet, model patches); the all-in figure and system RAM still count all of them (unloaded models stay in RAM).
The sets come from `ModelEntry.sampler_stages` (read from the template graph by `update`); when that is null (not
checked), from the rule "main files named `*high_noise*` and `*low_noise*`" (LoRAs named high_noise / low_noise go to
their set, others are shared). `estimate.expert_sets`. The official Wan 2.2 14B I2V template itself notes 84% VRAM on an
RTX 4090D 24 GB at 640×640, 81 frames.
サンプラーごとに別の本体を使うワークフローは、載せ替え・逃がすときに本体を 1 組ずつ数える。
動かし方で使う置き場所の使用率で決める(60% 以下 perfect / 85% 以下 good / 98% 以下 marginal / それより上 too_tight)。
載せ替えは good まで、逃がす・CPU は marginal まで。

## Gate

Computed every time (never stored in `db.json`). Details and the default license allowlist: docs/gate.md.

| key | type | meaning |
|---|---|---|
| `auto` | bool | true only when every check passed |
| `held_by` | str[] | ids of the failed checks, in the order below (`held_by[0]` is shown, shortened, in the table's Verified column and as `held:<id>` by `gate --only-held`) |
| `comfyui_version` | str \| null | the ComfyUI version compared against (`--comfyui-version`, else `sources.comfyui.latest_release`) |
| `checks` | Check[] | the 10 checks, in this order: `license`, `gated`, `official_workflow`, `custom_nodes`, `comfyui_version`, `url_reachable`, `sha256`, `content`, `size_known`, `reviewed` (0.6.0, #42: passes unless `review.status` is not `ok`; always last, so `held_by[0]` of the older checks is unchanged). For a model whose `runtime` is neither `comfyui` nor null (0.5.0), the three ComfyUI checks are replaced by one `app_runtime` check that never passes (8 checks) |

Check: `id` (stable, machine-readable), `passed` (bool), `detail` (Japanese, for people; wording may change).

## gate

`genmodelfit gate --json`: `policy` (`license_allowlist`, `comfyui_version`, `allow_uncensored`, `max_head_age_days`),
`filters` (`modality`, `only_auto`, `only_held`), `summary` (`models`, `auto`, `held`,
`failing` = models failing each check, `first_failing` = models whose first failed check is that one; both have
a key per check id, `app_runtime` included since 0.5.0, `reviewed` since 0.6.0),
`models` (`{"id", …Gate}`; all, or only auto / held with `--only-auto` / `--only-held`).

## diff

`genmodelfit diff [OLD] [NEW] --json` (Python: `gmf.diff_db(old_db, new_db)`, the same dict without `genmodelfit` /
`command`; #19). Compares two `db.json` files; no network, nothing written. Used for the weekly pull request
(`python -m genmodelfit.weekly`, docs/release.md). No `system` / `database`.
With no arguments the bundled list is compared with the fetched list in the cache (`update` / `update --from-release`);
with one argument the bundled list with that file (#59). The CLI compares the raw JSON (keys this version does not know count, as in `merge-check`).
When there is no fetched list yet, the output has the usual keys with empty values (`old` / `new` / `policy` / `classification` null, `summary` with the usual keys and 0 (`head_unchecked` null), lists empty) plus `note` (says to run `genmodelfit update --from-release`), and the exit code is 0.

| key | meaning |
|---|---|
| `old`, `new` | `generated_at`, `models` (count), `license_policy`, `comfyui_latest_release`, `partial`, `comfyui_checked_at` |
| `policy` | the gate policy applied to **both** lists (GatePolicy.to_dict): `comfyui_version` = `--comfyui-version`, else the version recorded in the new list (else the old one); `max_head_age_days` = null (HEAD age ignored; HEAD status codes still count) |
| `summary` | `models_old`, `models_new`, `added`, `removed`, `license_changed`, `license_permissive_side`, `auto_old`, `auto_new`, `auto_gained`, `auto_lost`, `size_changed`, `precision_changed`, `newly_uncensored`, `head_failures`, `head_failures_new`, `head_unchecked` (new `sources.head_check.unchecked`), `unauthorized_added`, `unauthorized_removed` |
| `added`, `removed` | `{id, name, modality, content, source, license}` (`license` is the label: `mit`, `other(<name>)`, `不明` for unknown) |
| `license_changes` | entries whose `(license, license_name)` changed: `{id, name, old, new, old_tier, new_tier, allowed_old, allowed_new, permissive_side}`. Tiers are `hf.license_tier` (1 permissive … 6 unknown). `permissive_side` = the tier went down (unknown → anything included) or the license newly passes the allowlist |
| `auto_changes` | entries whose auto result differs (added / removed entries too; `old` / `new` is null when absent): `{id, name, old, new, held_by, held_by_old}` |
| `size_changes` | total size changed by ≥ 10 % **and** ≥ 100 MiB (or became known / unknown): `{id, old_bytes, new_bytes, ratio}` |
| `precision_changes` | `{id, old, new, old_source, new_source}` |
| `newly_uncensored` | `content` became `uncensored`, or an added `uncensored` entry: the `added` fields + `evidence` (`content_evidence`), `added` (bool) |
| `head_failures` | URLs in the new list whose HEAD answered other than 200 / 302 / 307: `{url, status, ids, new}`; `new` = that URL did not fail (or was absent) in the old list |
| `unauthorized` | `{added, removed}`: repo ids in `sources.huggingface.unauthorized` (401) that appeared / disappeared |
| `partial_counts` | the new list's `sources.partial_counts` (null if absent) |
| `classification` | 0.6.0 (#42): `{safe, judgement_needed, safe_changes}`; see below |
| `old_label`, `new_label` | #59: what was compared, for display: `同梱の一覧(genmodelfit <version>)`, `取ってきた一覧(<cache path>)` or the file path as given; null from `gmf.diff_db` unless passed |
| `field_changes` | #59: `{id: [field, …]}` for every common entry with any change, including the safe ones. Fields are the names of `diff.entry_changes`, the same function the classification uses: top-level keys (`languages`, `license`, `review`, …), `files` (files added / removed / reordered), `files.<key>` (a key of a paired file), `template.<key>`. Files are paired in tiers so that same-named files are not mixed up: same (name, url, sha256, directory, repo_id) → same (directory, name, sha256) → same (directory, name, url) → the rest by (directory, name) in order; a changed name or directory is a removed + added file. A key that is missing on one side and null on the other is not a change |
| `model_changes` | #59: common entries with a change outside the safe list or an auto-recommendation change: `{id, name, modality, items, other}`. `items` (one per change): `{kind: "value", field, old, new}` for `name` / `modality` / `tasks` / `runtime` / `comfy_family` / `content` / `precision`; `{kind: "license", old, new}` (labels); `{kind: "languages", key, old, new}` (`key` is `prompt` / `speech` / …; null when absent; absent vs empty list is not an item); `{kind: "file_added" \| "file_removed", name, directory, size_bytes, license, role, source}` (`source` = the HF repo, else the URL host and path); `{kind: "files_order"}`; `{kind: "file_changed", name, field, old, new}` (`field` is `url` / `sha256` / `repo_id` / `directory` / `gated` / `role` of a paired file); `{kind: "file_license", name, old, new}`; `{kind: "license_source", name, field, old, new}` (where the license was checked: the entry's `license_url`, or a file's `license_url` / `license_from` / `license_path` / `repo_license`; `name` is the file, null for the entry); `{kind: "size", old, new}` (total bytes, any change); `{kind: "review", old, new, reasons, fields}` (`old` / `new` = the status, null when there is no review mark; the new reasons; changed review keys); `{kind: "auto", old, new, held_by}` (as in `auto_changes`). `other`: the remaining changed field names (shown as 「その他の変化」). Every changed field outside the safe list is in `items` or `other` |
| `top_changes` | #59: changed top-level keys of the lists other than `generated_at` / `sources` / `models` (only when the raw JSON is compared: `genmodelfit diff` and `merge-check` do; `gmf.diff_db` without `raw` gives []) |
| `warnings` | #59, CLI only: notes about the comparison: the new list is older (`generated_at`) than the old one (any argument form; with no arguments it also says to run `genmodelfit update --from-release`) |
| `stats` | changed values of `sources`, flattened to dotted keys (`counts.modality.image`, `head_check.status.302`, …): `{key, old, new}`; lists are compared by length; `checked_at` / `url` / `api` / `generator` / `failures` / `errors` are skipped |

**classification** (0.6.0, #42) splits the changes so that the weekly run can tell whether a week may be merged
without the owner (`safe: true`; the runner recomputes it with `python -m genmodelfit.weekly merge-check`) or needs an
evidence pull request for the owner. It is an **allowlist**: for every common entry every field and every file field is
compared, and only the changes listed as safe below are safe; anything else, including fields this version does not
know, is judgement-needed.

| key | meaning |
|---|---|
| `safe` | true when `judgement_needed` is empty |
| `judgement_needed` | `[{id, name, reasons}]`, sorted by id (`id` `(database)` for a changed top-level key other than `generated_at` / `sources` / `models`, merge-check only). `reasons`, in this order: `new` (added entry), `license_permissive` (entry or file license tier went down or newly on the allowlist, or the gate's license check newly passes), `need_decreased` (a file got smaller, or the estimate's needs went down: `review.review_reasons`), `uncensored_new`, `files_changed` (files added / removed / reordered, or a file's `name`, `url`, `sha256`, `repo_id`, `directory` changed, or its `size_bytes` changed in either direction or became known / unknown; a smaller file also gets `need_decreased`. Growth counts since 0.9.0: apps download the listed URL as is, so a bigger file at the same URL is different content), `use_changed` (`tasks` / `modality`), `auto_gained` (newly auto-recommendable, unless the old list held it only by `url_reachable` **and** no file's name / url / sha256 changed), `review_approved` (`review` went from not ok to ok), `review_pending` (from ok to not ok; `status` must be exactly `ok`, so `"OK"` is not ok; the review's own reasons follow), then `other_change:<field>` for every other changed field (`other_change:name`, `other_change:languages`, `other_change:description`, `other_change:license_url`, `other_change:precision`, `other_change:template.<key>`, `other_change:files.<key>`, `other_change:license` for a same-tier license rename, `other_change:review` for any other review change, …) |
| `safe_changes` | counts of changes seen in the safe direction: `removed`, `license_stricter`, `size_changed`, `precision_changed` (as in the diff; entries among them can still be judgement-needed), `head_failures_new`, `unauthorized_added`, `unauthorized_removed`, `stats` (changed `sources` values), `auto_gained_head_only`, `auto_lost`, `pending_carried` (the same pending `review` in both lists) |

Safe (the allowlist; never in `judgement_needed` by itself): entry `downloads`, `downloads_all_time`, `likes`,
`updated`; template `usage`; file `head_status`, `head_checked_at`; a
license moving to a stricter tier (entry or file); removed entries; models dropping out of auto-recommendation;
auto-recommendation regained only because HEAD recovered with the same files; `sources` statistics; a pending `review`
carried over unchanged. Everything else is judgement-needed.
安全とみなすのは上の表の変化だけ(許すものの表)。知らない項目の変化も判断が要る。

```json
"classification": {
  "safe": false,
  "judgement_needed": [
    {"id": "hf:org/new-model", "name": "New Model", "reasons": ["new"]},
    {"id": "image_example", "name": "Example", "reasons": ["license_permissive", "auto_gained"]}
  ],
  "safe_changes": {"removed": 1, "license_stricter": 0, "size_changed": 2, "precision_changed": 0,
    "head_failures_new": 3, "unauthorized_added": 0, "unauthorized_removed": 1, "stats": 12,
    "auto_gained_head_only": 1, "auto_lost": 2, "pending_carried": 0}
}
```

`python -m genmodelfit.weekly --out DIR` writes `DIR/db.json`, `DIR/diff.json` (the dict above), `DIR/pr_body.md`,
`DIR/review_queue.json` (0.6.0: `{genmodelfit, generated_at, pending: [review list items], classification}`) and
`DIR/result.json`: `genmodelfit`, `ok`, `problems` (str[], why it failed), `base`, `offline`, `run_url`, `partial`,
`status` (DataStatus of the new list), `summary` (as above), `classification` (`{safe, judgement_needed: count}`),
`review_pending` (the review list items), `files` (`db`, `diff`, `pr_body`, `review_queue`), and `applied_to` with
`--apply`. The pull request body has a 「分類」 section and a 「確認待ち」 table (names escaped like every upstream
text). Exit codes are unchanged: a judgement-needed week is still `ok` (exit 0).

`python -m genmodelfit.weekly merge-check --base <main db.json> --head <new db.json> [--json]` recomputes the
classification from the two files (raw JSON, so unknown keys count): `{genmodelfit, command: "merge-check", base, head,
safe, judgement_needed (count), reasons ([{id, name, reasons}]), safe_changes}`. Exit 0 = safe, 3 = not safe,
1 = a file cannot be read (`error` with `--json`), 2 = usage error. No network, nothing written.

## review

Review marks (0.6.0, #42). New or riskier entries go into the list but are **not auto-recommended until someone has
checked them against primary sources**. `ModelEntry.review` is an optional object (`schema_version` stays 1; readers
up to 0.5.0 drop the key):

| key | type | meaning |
|---|---|---|
| `status` | str | `ok` or `pending` (any other value is treated as not ok) |
| `reasons` | str[] | why it was marked: `new`, `license_permissive`, `need_decreased`, `uncensored_new`, `files_changed` (kept after approval: what was checked) |
| `decided_by` | str \| null | who approved (`--by`; the bundled list starts with `baseline-2026-10-21`) |
| `decided_at` | str \| null | `YYYY-MM-DD` (UTC) of the approval |
| `evidence` | str[] | https URLs of the primary sources (license text, model card, …) |
| `note` | str \| null | free text |

**A missing `review` (null) means ok** (lists made before 0.6.0). **Apps should filter with
`entry.review is None or entry.review.status == "ok"`** (Python: `entry.reviewed`); the gate already holds pending
models (`reviewed`, shown as `review` in the Verified column).
アプリは `entry.review is None or entry.review.status == "ok"` で絞る(印の無い古い一覧は確認済み)。

How `update` sets it (`review.apply_reviews`, comparing with the list it carries over from: the cache or the bundled
list, or the base list in the weekly run):

- an id absent from the previous list → `pending` with `new` (plus `uncensored_new` when uncensored). Without any
  previous list (`carry_over=False`) every entry is `new`
- a common entry → `pending` when its license moved toward permissive (`license_permissive`: `hf.license_tier` went
  down, the set's license is newly on the allowlist, or the gate's `license` check newly passes), its needs went down
  (`need_decreased`: at one of the reference PCs `review.REFERENCE_PCS` = (0 GB VRAM, 16 GB RAM), (6, 16), (8, 16),
  (12, 32), (16, 32), (24, 64) the fit level went up, the size became known, or the full-load VRAM or the RAM went down
  by more than 1 %; estimated only when a need-related field changed), or it newly became `uncensored`
  (`uncensored_new`), or its files changed (`files_changed`: a file added / removed, a file's `url` (query string and
  fragment ignored) or `size_bytes` changed in either direction or became known / unknown (since 0.9.0 growth counts too: the same URL now serves different content), or a `sha256` that had a value differs or went empty; a `sha256` that was
  empty and is now filled in, the first check, does not count. Files are paired as `diff` does). Reasons are added to an entry that was already pending
- otherwise the previous mark is carried over (`ok` with a note when the previous list had none)

`update --heads-only` / `--fields-only` / `--inspect-only` apply the same rule against the list as it was before the
run. `sources.review`: `{ok, pending, marked: {reason: count}, rejected_skipped}`.

The signed list published to `data-latest` (data.yml) leaves out pending entries
(`python -m genmodelfit.review publish`; `sources.review.withheld_pending` counts them): genmodelfit 0.5.x drops the
unknown `review` key and would auto-recommend them. The bundled `db.json` on main keeps them (0.6.0 holds them by the
`reviewed` check).

Rejected entries stay out: `data/review_rejected.txt` (bundled), one line per entry
`<id or sha256:<hex of the id>><TAB><YYYY-MM-DD><TAB><reason>`, `#` comments. `update` drops matching ids
(`rejected_skipped`). With `--hide-name` only the hash is stored (like the safety exclusions, the name is not kept).

Commands (`--db` is required for writing; in a checkout use `--db src/genmodelfit/data/db.json`):

```text
genmodelfit review list [--json] [--all] [--db PATH]
genmodelfit review approve <id> --evidence <https URL> [--evidence …] [--note TEXT] [--by NAME] --db PATH [--json]
genmodelfit review reject <id> --note TEXT --db PATH [--rejected FILE] [--hide-name] [--json]
```

`approve` refuses (exit 1) without at least one evidence URL that is `https` with a host (`https://` alone is refused), for an unknown id, or when the entry is not
pending; it keeps `reasons` and sets `status: ok`, `decided_by`, `decided_at` (today, UTC), `evidence` and `note`.
`reject` requires `--note`, removes the entry, updates `sources.counts` / `sources.review` and appends the line.
Python: `gmf.review_list(db=None, include_ok=False)`, `gmf.review_approve(db_path, id, evidence, note=, by=)`,
`gmf.review_reject(db_path, id, note, rejected_path=, hide_name=)`; they raise `review.ReviewError` (a `ValueError`)
and print nothing.

`python -m genmodelfit.review apply DECISIONS.json --db PATH [--max-approve 10] [--rejected FILE] [--json]` applies the
weekly decisions (the runner script calls it; Claude only writes the decisions). Input:
`{"decisions": [{"id", "action": "approve"|"reject"|"hold", "evidence": [https URLs with a host], "note": str,
"safety": bool}]}`. Validation is strict and all-or-nothing (exit 2, nothing written): unknown keys at either level,
unknown actions, ids that are not pending, the same id twice, `approve` without evidence or with `safety`, `reject`
without a note, a note over 500 characters or with control / format characters (newlines, NUL, ESC, bidi overrides,
zero-width characters), non-boolean `safety`, more approvals than `--max-approve`. Approvals get
`decided_by: "claude-weekly"`; a reject with `safety: true` stores only the id hash (`--hide-name`). Output (`--json`):
`{applied, db, rejected_file, counts: {approve, reject, hold}, pending_left, results: [{id, action, result, safety,
evidence, note}]}` with every string escaped by `weekly.md` (safe in a pull request body; a safety reject shows only
the start of the hash and no note). Exit 0 applied, 2 invalid decisions, 1 unreadable `--db`.

Items of `review list --json` / `review_queue.json` also have `card_url` (the Hugging Face card: the repo of an `hf:`
entry, else the main file's license source or host repo, else `homepage`) and `files`:
`[{name, repo, path, url, license, license_name, license_from, license_url, role}]` (`repo` / `path` parsed from the
Hugging Face URL). Strings are data (not escaped) in JSON.

Duplicate model ids are refused everywhere (0.6.0, #42): `parse_db` / `load` / `Database.save` raise
`DatabaseError`, and `merge-check` exits 1 for a file with a duplicate id (otherwise `get()` would silently pick one
of two entries).

`review list --json`:

```json
{
  "genmodelfit": "0.6.0",
  "command": "review",
  "database": {"origin": "path", "generated_at": "2026-10-28T03:12:00Z", "models": 331, "measurements": 21},
  "pending": 1,
  "models": [
    {"id": "hf:org/new-model", "name": "New Model", "modality": "image", "content": "sfw",
     "source": "huggingface", "license": "apache-2.0", "license_url": "https://huggingface.co/org/new-model",
     "homepage": "https://huggingface.co/org/new-model", "status": "pending", "reasons": ["new"],
     "decided_by": null, "decided_at": null, "evidence": [], "note": null}
  ]
}
```

## info

`model` (ModelEntry), `estimate`, `gate`, `measurements` (Measurement[]), `group_key`, `same_model` (the other
entries of the same model, each like a `workflows` item, judged for this PC, best first), `model_name`.
`info <query>` also accepts a model's display name (exact, case-insensitive): then the model's best workflow for
this PC is shown.

How `info <query>` picks the model / モデルの選び方:

- An exact id always matches, whatever its content (uncensored included). / ID がそのまま合えば内容によらず出す
- Otherwise the query is matched as a part of the id or name (case-insensitive), **filtered by `--content`
  (default `sfw`)**. Exactly one match → that model. / それ以外は ID・名前の一部で探し、`--content` で絞る(既定 sfw)。1 件ならそれ
- Two or more matches → never picks one: exit code 1. Text mode lists the candidates on stderr; `--json` prints
  the envelope with `error` and `candidates` (`[{"id", "name", "content"}]`, all of them). / 2 件以上は選ばずに終了コード 1
- No match (or only uncensored matches under `sfw`) → exit code 1 with `error` (the message suggests `--content all`).

Python `gmf.info(model_id)` matches the exact id only and returns `None` otherwise.

### ModelEntry

`id`, `name`, `modality`, `tasks`, `source`, `files` (ModelFile[]), `template` (Template \| null), `params_b`,
`precision`, `license`, `license_url`, `base_model`, `content`, `content_evidence` (str[]: why it is marked
uncensored), `discovered_from` (str[]: where it was found), `updated`, `homepage`, `downloads` / `likes`
(int \| null: Hugging Face counters, set for `source: huggingface`; `downloads` is the last 30 days),
`downloads_all_time` (int \| null: all-time downloads, used for the inclusion floor), `total_size_bytes`,
`license_name` (the name of an `other` license, from the strictest file), `license_evidence` (str[]: why the license is
unknown — entries start with `license_unverified:`), `files_complete`, `size_floor_bytes` (below),
`workflow_precision` (str \| null: precision inferred from the template's main loader node, see "Models" above;
optional, 0.2.0), `precision_source` (str \| null: `header` / `filename` / `hf_api` / `config` / `workflow`, where
`precision` came from; null = unknown or older data; optional, 0.2.0).

**For apps that build their screens from the list alone** (0.5.0, #36; all optional, `schema_version` stays 1 — older
readers drop unknown keys, see "Database file"):

- `languages` (`{"prompt": str[], "speech": str[]}`, ISO 639-1 codes): `prompt` = languages the text prompt works in,
  `speech` = languages spoken or sung in the output (TTS voice, lyrics). A key that is missing means unknown; `[]` means
  the model has none (image-to-3D takes no prompt text, a sound model without vocals has no speech). Rebuilt from its inputs on every
  apply (nothing carries over from the previous list; docs/sources.md「言語・動かし方・説明」): the hand table
  `data/overrides.json` (or the entry's own `languages` in `data/manual.json`) wins for the keys it sets; otherwise
  the `language` of the Hugging Face card (saved as `card_languages`) is the upper bound — `speech` is the card's
  list, `prompt` is the card's list, narrowed to `en` when every text encoder is English-only (CLIP / T5) and for
  music (a music card's `language` usually lists lyric languages, so a music entry gets at most `["en"]` from the card
  and no `speech`; 0.6.0, #41); without a
  card, a known text encoder gives `["en"]` (a multilingual encoder alone never adds `ja`). Nothing decides → the key
  is left out (no guessing). Test membership of the language you need.
- `card_languages` (str[] \| null): the Hugging Face card's `language` saved by `update` / `update --fields-only`
  (the material for `languages`, so an offline re-apply gives the same result). `[]` = the card was read and has no
  language, null = not read.
- `runtime` (str \| null): `comfyui` for official templates and for Hugging Face image / video / music / sfx / 3D entries
  whose files match a ComfyUI family; `piper`, `irodori-tts` for the hand-added speech models; null = not decided. A
  runtime other than `comfyui` (and null) makes the estimator count `.onnx` and other non-ComfyUI weight files.
- `description` (`{"ja": str, "en": str}`): a short hand-written description, only for the models that have one.
- `source: "manual"` (id `manual:<slug>`): models added by hand in `data/manual.json` (`template` is null). Their license
  per file was read from the upstream card by the person who added them (`license_evidence` starts with `manual:`); `update`
  compares it with the card of `license_from` and takes the stricter one on a mismatch (with a warning).
- Hand-added components (0.6.0, #40): `components` in `data/manual.json` adds files that the Hugging Face collection
  does not pick up to an entry of another source (first: the T5Gemma text encoder of
  `hf:stabilityai/stable-audio-3-small-sfx`, Gemma Terms of Use). Their license is never written by hand: it comes from
  the per-file source table (`license_sources.py`; no row → unknown), and `license_evidence` ends with
  `manual_component: <why this model uses the file>`. `update` (all modes) applies them on every run.

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

`sampler_stages` (str[][] \| null, optional; 0.1.1): for official templates, the model files each sampler (KSampler,
KSamplerAdvanced, SamplerCustom, CFGGuider, …) loads through its MODEL input (diffusion model, LoRAs, model patches),
one list per distinct set, in graph order (`workflow.sampler_stages`, kept only for sets that contain one of the entry's
main files). Set only when there are two or more different sets of main files (`[]` = one set, `null` = not checked; HF
entries are null). Used by the estimator as above.

`review` (object \| null, optional; 0.6.0, #42): the review mark, see "review". null = ok (older lists).

`comfy_family` / `comfy_vae` (str \| null, optional; added in #21 — older files without them still load): the ComfyUI
model family and VAE kind (keys of `families.FAMILIES` / `families.VAES`), set by `update` from the main / VAE file
names. Used only to decide whether ComfyUI computes in fp32 on GPUs without bf16 (`docs/hardware.md`, 「系統の表」).
null = unknown (the estimate is not changed; a reason says so on GPUs without bf16). When null, the estimator derives
them from the file names again.

ModelFile: `name`, `url`, `sha256`, `size_bytes`, `directory` (ComfyUI folder), `repo_id`, `license`,
`license_name`, `gated` (bool \| null = unknown), `role` (`main` / `text_encoder` / `vae` / `clip_vision` / `lora` / `other`),
`license_from` (the repo the license was read from: the upstream repo for repackaged files, the table's upstream for
known components; a repo id, never a path), `repo_license` (the hosting repo's own label when it differs from the
upstream one), `license_url`, `license_path` (str[]: the derivation path that was checked — hosting repo → base_model … →
deciding repo), `license_evidence` (str[]: `license_unverified: …`, `component:<key>`, `family:<key>`, `upstream_gated: <repo>` — the component's upstream requires accepting terms),
`head_status` (int \| null: status of an anonymous HEAD without following redirects, recorded by `update`;
null = not checked), `head_checked_at` (UTC ISO-8601 of that HEAD), `header_precision` (str \| null: precision read
from the file's own header — safetensors: the dtype holding most bytes, `F32`→`fp32`, `F16`→`fp16`, `BF16`→`bf16`,
`F8_*`→`fp8`, `I8`→`int8`, packed 4-bit `U8` → null; GGUF: `general.file_type`, else the commonest tensor type, as
`gguf-q8_0` etc.), `header_params` (int \| null: total elements of all tensors in the file). Both are read by
`update` / `update --inspect-only` for Hugging Face main weight files only (`headers.py`; ranged GET of the first 8 bytes
and the JSON header, or of the first MiB(s) of a GGUF, never the weights), null = not read. Main files are the
`main` role plus what `groups.main_files` treats as the main file (the largest non-encoder / non-VAE file when no
file has the `main` role, e.g. BiRefNet; 0.2.0).
`repo_precision` / `repo_precision_source` (str \| null, optional, 0.2.0): precision read from the hosting repo
(`hf_api` or `config`, see "Models"), set only for the largest main file when its header and name tell nothing.
The estimator decides a file's part from `directory` first, then `role`, then the Hugging Face folder / known file
names, and counts one variant per part (`docs/accuracy.md`, 「ファイルの数え方」).

Template: `name`, `title`, `category`, `min_comfyui_version`, `open_source`, `requires_custom_nodes`, `usage`,
`date`, `size_bytes`, `username` (who made it; not `ComfyUI` = community template).

### Measurement

`model_id`, `gpu_name`, `gpu_vram_gb`, `peak_vram_gb`, `peak_ram_gb`, `seconds` (wall time), `settings`
(`width`, `height`, `frames`, `seconds` = audio length, `steps`, `precision`, `batch`), `driver`,
`comfyui_version`, `offload`, `source`, `date`, `note`, `outcome`, `standard`, `vram_metric`, and (#34) `origin`,
`n_submissions`, `consistent`, and (2026-10-22, optional) `n_used`, `n_oom`, `gpu_backend`.

| key | type | meaning |
|---|---|---|
| `gpu_vram_gb` | float | total VRAM. bench stores `/system_stats` `vram_total` in GiB rounded to 2 decimals (e.g. `7.99`); older bundled rows hold the nominal size (`16`) |
| `offload` | str \| null | how ComfyUI placed the model: `none` (all on GPU) / `dynamic` (dynamic VRAM) / `partial` (partially loaded) / `lowvram` (`--lowvram` / `--novram`) / `oom` (ran out of memory) |
| `outcome` | str | `ok` (finished) / `oom` (out of memory) / `error` (failed for another reason). Default `"ok"` |
| `vram_metric` | str \| null | how `peak_vram_gb` was taken: `torch_reserved` (bench: peak torch reservation of the ComfyUI process, excluding the CUDA context) or null (not recorded; the bundled RTX 5080 rows). Calibration uses only rows with null (`calibrate.CALIBRATION_METRIC`) |
| `origin` | str \| null | (#34) `community` for rows of `measurements_community.json` (other users' submissions aggregated by `contrib aggregate`); null for the maintainer's bundled rows and this PC's bench results |
| `n_submissions` | int \| null | community rows: submissions in the group (outliers and OOM included). The file also keeps `n_outliers`, `spread_mad_gb`, `spread_rel_pct`, `peak_vram_min_gb` / `peak_vram_max_gb`, `outlier_submissions` and `submissions` (sha8 of each file), which the reader ignores |
| `n_used` | int \| null | community rows: `ok` submissions behind the median (outliers removed). Becomes `Estimate.sample_count` |
| `n_oom` | int \| null | community rows: submissions of the same group that ran out of memory. A group with `ok` and `oom` uses the `ok` median and keeps this count (shown in the reasons; such a row never makes Perfect) |
| `consistent` | bool \| null | community rows: at least 2 used submissions whose range is within 15% of the median, and no OOM in the group. Only consistent rows enter the calibration of the coefficients (`calibrate`); estimates use every `ok` row from one submission (2026-10-22) |
| `gpu_backend` | str \| null | (0.9.0) PyTorch backend of the measured GPU as bench saw it: `cuda` (NVIDIA) / `rocm` (AMD; ComfyUI reports `type: cuda`, told apart by `+rocm` in `pytorch_version` or the GPU name) / `xpu` (Intel) / `mps` (Apple). null = not recorded (all bundled rows are NVIDIA). Decides the measurement's vendor for #26 and calibration across vendors (`estimate.measurement_vendor`); the GPU name is used only when it is null (Apple's MPS reports the name `mps`). `contrib aggregate` keeps it on community rows |
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
| `submission` | object | only with `--submission`: `{saved_to, path_in_repo}` (the submission JSON, see contrib) |

How bench fills Measurement / 測り方:

- `peak_vram_gb`: max of `/system_stats` `devices[0].torch_vram_total` polled every 0.2 s (the ComfyUI process,
  without the CUDA context). The GPU-wide delta from the pre-run baseline goes to `note`: `nvidia-smi` for NVIDIA;
  for other GPUs (0.9.0, `gpustat.py`) `amd-smi` → `rocm-smi` (AMD, Linux), `xpu-smi` (Intel, Linux), the Windows
  performance counters via `typeperf` (AMD / Intel, Windows) and `ioreg` (Apple). The same tools give the
  pre-run busy check (average utilization ≥ 30% refuses without `--force`). When no tool can be read, bench does
  not stop: it checks that the ComfyUI queue is empty and that apps other than ComfyUI hold at most 50% of the
  device memory (not on Apple's unified memory), then measures, with a warning and a `note` sentence;
  `standard` is not changed (docs/accuracy.md). A remote ComfyUI still needs `--force`.
- `peak_ram_gb`: max resident memory of the process listening on the port (and its children), only when its
  command line contains `main.py` or `comfy`; null when remote or not identified.
- `seconds`: wall time from submit to completion, including model loading. `/free` is sent first only with `--free`;
  without it the pre-run torch allocation is written to `note` and a warning is printed.
- `offload`: read from the ComfyUI log: `dynamic` / `partial` / `lowvram` / `none` / `oom`, or null when unknown.
- `--set` values: only numbers are written to `note`; for other values only the input name is kept.
- The saved file is `{"measurements": [Measurement]}` and holds no paths, user names or prompts.

## contrib

Measurement submissions from users (#34). genmodelfit **never sends** them: users add the file to `data/contrib/` of
their fork and open a pull request themselves (CONTRIBUTING.md). Code: `src/genmodelfit/contrib.py`.
利用者の実測の提出。genmodelfit は送らない(利用者が自分で PR を出す)。

### Submission file (`contrib_schema` 1)

Written by `genmodelfit bench … --submission PATH` or `genmodelfit contrib make --from <bench result>`. Stored in the
repository as `data/contrib/<gpu-slug>/<date>-<sha8>.json`: `gpu-slug` is the GPU name in lower case without
`NVIDIA` / `GeForce` (`rtx-3070`), `sha8` the first 8 hex digits of the sha256 of the canonical JSON (keys sorted,
no spaces, UTF-8), so the name does not depend on indentation and two identical files collide.

```json
{
 "contrib_schema": 1,
 "kind": "genmodelfit-measurement",
 "tool": {"name": "genmodelfit", "version": "0.3.0"},
 "date": "2026-10-20",
 "system": {
  "os_family": "windows",
  "ram_total_gb": 32,
  "gpu": {"name": "NVIDIA GeForce RTX 3070", "vram_gb": 8.0, "arch": "ampere", "compute_capability": "8.6", "driver": "566.03"}
 },
 "comfyui_version": "0.40.0",
 "measurement": {
  "model_id": "image_sdxl_simple",
  "settings": {"width": 832, "height": 832, "steps": 20},
  "outcome": "ok", "standard": true, "offload": "dynamic",
  "peak_vram_gb": 6.9, "peak_ram_gb": 7.4, "seconds": 41.2,
  "vram_metric": "torch_reserved",
  "note": "…"
 }
}
```

| key | type | meaning |
|---|---|---|
| `date` | str | UTC date only (`YYYY-MM-DD`); no time |
| `system.os_family` | str \| null | `windows` / `linux` / `macos`; no version or build. null when ComfyUI ran on another PC |
| `system.ram_total_gb` | int \| null | RAM in GiB rounded to an integer; null when unknown or overridden with `--ram` |
| `system.gpu` | object | `name`, `vram_gb` (as measured), `arch` / `compute_capability` (NVIDIA only: from nvidia-smi on this PC when the names match, else from the GPU name table), `driver`, and (0.9.0, optional) `vendor` (`nvidia` / `amd` / `intel` / `apple`) with `backend` (`cuda` / `rocm` / `xpu` / `mps`), written only when bench recorded the backend (`Measurement.gpu_backend`); both or neither. Older files without them stay valid (`contrib_schema` stays 1) |
| `measurement` | object | the Measurement fields that matter (settings without `batch: 1`). `note` is split into sentences (`。`, newline, `;`); every sentence that contains this PC's host / user name / home folder or matches a privacy pattern (below) is dropped whole, and the count of dropped sentences is appended (`contrib.scrub_text`) |

Not written by genmodelfit: host name, user name, paths, GPU UUID, PCI bus id, serial numbers, IP / MAC addresses,
the measurement's free-text `source`, CPU name, exact time. **Limits (honest)**: the checks are patterns, not
understanding. **Not detected**: personal names or other identifying words in plain text (e.g. `山田太郎のPCで測った`,
`on Taro's desktop`, `github KoyoYeager`), paths written in base64 or another encoding (URL-encoded `%2F` / `%5C` /
`%3A` is detected), and paths without separators described in words. That is why `contrib show` asks the user to
read the note and a person reviews each pull request.
Free-text fields are kept small: `gpu.name` only letters, digits, space, `.` and `-` (64 chars), `comfyui_version`
strictly `x.y.z`, `precision` one of the known precisions, `model_id` up to 200 chars of `[A-Za-z0-9_.:/+-]`, other
strings 100 chars, `note` 2000 chars; no control characters, newlines or invisible characters anywhere; unknown keys,
arrays and nesting deeper than 8 are rejected; files over 64 KB and symbolic links are not read.

### `contrib show FILE`

Prints every field of the file and a privacy checklist (`known_fields`, `paths`, `host_user` — also this PC's own
host / user name —, `device_ids`, `network`, `date_only`, and `note_reviewed`, which is for the user to tick).
`--json`: envelope + `action: "show"`, `file`, `ok`, `submission`, `checklist` [{id, label, ok (null = for the user
to check), detail}], `report` (below). Exit code 1 when something must be fixed.

### `contrib validate FILE|DIR… [--json] [--strict] [--no-filename-check] [--markdown]`

For each file a report `{path, ok, errors: [{code, message}], warnings: [{code, message}], notices: [{code,
message}], sha256, suggested_path, model_id, gpu_name}`. `--json`: envelope + `action: "validate"`, `ok`, `files`,
`valid`, `invalid`, `notices` (files with notices), `warnings` (count), `reports`. `--markdown`: a table for the CI
job summary, with a line for the maintainer above it when some files have notices.

Like llmfit's `validate_community_benchmarks.py`, **only mechanical checks fail** (2026-10-22): errors below.

| error code | when |
|---|---|
| `read` | not readable, not JSON, larger than 64 KB, nested too deep, or a symbolic link (symlinked folders are not followed) |
| `string` | a string with control characters / newlines / invisible or bidi characters, too long a string or key, an array, or nesting deeper than 8 |
| `schema` | `contrib_schema` ≠ 1, wrong `kind` / `tool`, missing keys, not objects |
| `unknown_field` | a key outside the schema (anywhere, including `settings`) |
| `privacy` | a value or key (after NFKC normalization, removing zero-width characters and reading `¥` as `\`) contains: a drive path (`C:\`, `C:/`), any backslash, two or more path separators between words (`a/b/c`, `Users\name\`), `/home/`-like roots, `//server`, `~/`, `%VAR%`, `$HOME`, URL-encoded separators (`%2F` `%5C` `%3A`), `@` followed by a letter (e-mail, @handle), a UUID / GPU UUID, a PCI address (`0000:01:00.0`, `01:00.0`), an IPv4 / IPv6 / MAC address, a URL (`scheme://`, `www.`), a host-like name (`DESKTOP-XXXX`, `*-PC`, `*.local`, the words "host name" / "PC 名" …), a time (`14:05`, `2pm`, `14時05分`, ISO date-time) or a serial number (`SN:`). Some harmless text is also rejected (e.g. `12:30`, a 4-part version `1.2.3.4`) |
| `date` | not `YYYY-MM-DD`, or in the future |
| `value` | out of range: `peak_vram_gb` > GPU VRAM × 1.02 + 0.25, `peak_ram_gb` > RAM × 1.05 + 1, `seconds` ≤ 0 or missing for `ok`, unknown `outcome` / `offload` / `vram_metric` / `os_family` / `arch` / `vendor` / `backend`, malformed `driver` / `comfyui_version` / `compute_capability` |
| `settings` | width / height 64–8192, frames 1–2000, seconds 0.1–3600, steps 1–1000, batch 1–64 (integers except seconds) |
| `gpu_inconsistent` | the GPU name's known compute capability or generation differs from the file's; `vendor` and `backend` do not pair up (or only one is given); `compute_capability` / `arch` on a non-NVIDIA `vendor`; a GPU name of another vendor (AMD / Intel / Apple) than `vendor` |
| `filename` | name not `<date>-<sha8>.json`, date or sha8 not matching the content, folder not the GPU slug |
| `value` (strict forms) | `gpu.name` / `comfyui_version` / `precision` / `model_id` / `tool.version` outside the forms above |
| `duplicate` | same content hash as an earlier file in the same run |

Notices (the file is valid and is aggregated; shown in the CI summary for the maintainer who merges — merging is
the review): `unknown_model` (not in the bundled list), `unknown_gpu` (not in the GPU name table, e.g. AMD / Intel /
Apple), `deviation` (peak VRAM more than 40% away from the formula's all-in estimate for those settings),
`below_weights` (`offload: none` but peak VRAM below the weights + fixed overhead; estimates use that floor for
every submission below it — with dynamic VRAM / partial loading a smaller peak is normal), `fit_change` (this
submission alone would change the level at a common size: VRAM 4 / 6 / 8 / 12 / 16 / 24 / 32 / 48 GB × RAM 16 / 32 /
64 GB, at the default and the submitted settings). Until 2026-10-21 these were "review required" and needed a
per-sha8 approval (`--allow-review`, `data/contrib/REVIEWED`); both were removed.

Warnings (accepted, but reported; `--strict` makes them fail): `nonstandard` (kept as a record, never aggregated),
`metric_missing`, `driver_missing`, `comfyui_version_missing`, `ram_not_rounded`, `settings_missing` /
`settings_modality`, `offload_outcome`, `date_old`.

`--markdown` escapes every cell (`| \` < > [ ] & @`, newlines, and GFM autolinks: `://` and `www.`) so file names
and values cannot inject Markdown, HTML, links or workflow commands into the CI summary. CI appends genmodelfit's
stderr to the summary through the same escaping.

Exit codes: 0 every file passed (warnings and notices allowed) / 1 a file failed or has a warning with `--strict` /
2 usage error.

### `contrib aggregate [DIR] [--output FILE] [--report] [--threshold PCT] [--fail-on-breach] [--json]`

Reads `DIR` (default `data/contrib`; a missing folder is an error), keeps the files that pass `validate` (notices
allowed), and groups standard submissions by model id, GPU name (normalized: case,
`NVIDIA` / `GeForce`, spaces), GPU VRAM rounded to 1 GB (an RTX 4060 Ti 8 GB and 16 GB are different groups),
`vram_metric` and settings (precision normalized: `BF16` = `bfloat16` = `bf16`). Among the `ok` runs of a group, values with |x − median| / (1.4826 × MAD) > 3.5
are outliers (only with 3 or more runs; MAD floored at 2% of the median); the rest's median is the value, with
`n_used` and `n_oom` (a group of only OOM runs becomes one `outcome: oom` row, never used). A group is `consistent`
(enters the calibration) with at least 2 used runs whose range is within 15% of the median and no OOM in the group. Output rows are Measurement records
with `origin: "community"` (see Measurement) in a fixed order, so the same input always gives the same file.
`--output` writes `{unit, source: "community", contrib_schema, date (latest submission date), method, measurements}`
(an empty result is not written unless `--allow-empty`, so the bundled file is never wiped by mistake);
the bundled copy is `src/genmodelfit/data/measurements_community.json`, which is loaded with the other measurements.

`--report` re-runs the calibration (`calibrate.calibrate`) on the bundled maintainer rows plus the new community rows
(never the local cache) and returns per modality `loo_n`, `loo_mean_abs_pct`, `loo_max_abs_pct`, `evaluable`, `check_n`,
`check_mean_abs_pct`, `check_max_abs_pct`, plus `breaches` [{kind (`loo` / `community_check`), model_id, gpu_name,
origin, error_pct}] and `breach`. Counted against the threshold (default 25%): LOO errors of evaluable modalities and
the check errors of community rows; the maintainer's own check rows (RTX 3070) are shown but not counted.
`fit_changes` [{model_id, vram_gb, ram_gb, before, after}] lists the level changes the community rows cause at the
common sizes above (section "Fit-level changes caused by community data" in the text report).
Text output is Markdown (for the weekly pull request of #19).

`--json`: envelope + `action: "aggregate"`, `dir`, `files`, `valid`, `invalid`, `notices`, `submissions_used`, `groups`,
`consistent`, `written_to`, `community` (the document above), and `report` with `--report`.
Exit codes: 0 / 1 error (no such folder; empty result with `--output` without `--allow-empty`) / 2 usage error /
**3** a breach with `--report --fail-on-breach`.

## download

`genmodelfit download <model> --json` prints the envelope (with `system` and `database`) plus the keys below.
`--list --json` prints the plan only (`listed: true`, `result: null`; no network access, nothing written, exit 0 even
when `plan.ok` is false). Downloading with `--json` needs `--yes`. On failure (exit 1, or 130 after Ctrl-C) `error` is
set, `result` is null and `plan` is the plan (when it was made). A model that is not found or matches several
models gives `error` (and `candidates`) without `plan`.
Python: `gmf.plan_download(...)` returns `DownloadPlan` (`.to_dict()` is `plan`), `gmf.download(...)` returns
`DownloadResult` (`.to_dict()` is `result`).

| key | type | meaning |
|---|---|---|
| `listed` | bool | true for `--list` (nothing downloaded) |
| `plan` | DownloadPlan | what is (or would be) downloaded |
| `result` | DownloadResult \| null | what happened; null for `--list` and on failure |

DownloadPlan:

| key | type | meaning |
|---|---|---|
| `model_id`, `name` | str | the workflow (list entry) chosen |
| `model_name` | str | the model name shown in the tables |
| `modality`, `content`, `precision` | str / str / str \| null | as in ModelEntry |
| `fit_level`, `run_mode_label`, `auto` | str / str / bool | the judgement on this PC (as in `recommend`) |
| `chosen` | str | why this workflow was chosen (Japanese) |
| `layout` | str | `comfyui` (`--comfyui-dir`: `root` is `<dir>/models`) / `output` (`--output-dir` or the default; same layout) |
| `root` | str | absolute path of the folder files go into (`<root>/<directory>/<name>`) |
| `files` | PlannedFile[] | one per file of the workflow |
| `total_bytes` | int | sum of the file sizes |
| `download_bytes` | int | still to download (minus files already there and `.part` bytes) |
| `present_bytes` | int | files already there with the right sha256 |
| `disk_free_bytes` | int \| null | free space at `root` (or its nearest existing parent) |
| `disk_needed_bytes` | int | `download_bytes` + `margin_bytes` (0 when nothing to download) |
| `margin_bytes` | int | 1 GiB |
| `budget_bytes` | int \| null | `--budget` in bytes |
| `blocked` | str[] | why it will not download (Japanese). Empty = can download |
| `warnings` | str[] | things to know (licenses not on the allowlist, will not run on this PC, layout hint) |
| `alternatives` | object[] | the other workflows considered: `id`, `precision`, `use`, `fit_level`, `memory_required_gb`, `size_bytes`, `auto` |
| `license_url` | str \| null | the model's license page |
| `ok` | bool | `blocked` is empty |

PlannedFile:

| key | type | meaning |
|---|---|---|
| `name` | str | file name in the list (may contain ComfyUI subfolders, `/`) |
| `relpath` | str | path under `root` (`<directory>/<name>`; `other/<name>` when the list has no directory); "" when refused |
| `dest` | str | absolute destination path; "" when refused |
| `url`, `sha256`, `size_bytes`, `directory`, `role` | | from ModelFile |
| `license`, `license_name` | str \| null | the license used for the check (gate's conservative reading; null = unknown) |
| `license_url` | str \| null | where the license text is |
| `license_note` | str \| null | why the license differs from the list's value, or why it could not be confirmed |
| `license_allowed` | bool | on the license allowlist (gate default or `--license-allowlist`) |
| `gated` | bool \| null | access request needed (also when the upstream of a component asks for it) |
| `status` | str | `download` / `resume` (a `.part` exists) / `present` (already there, sha256 matches) / `conflict` (a different file has the name; never overwritten) / `unavailable` (see `reason`) |
| `part_bytes` | int | size of the existing `.part` |
| `remaining_bytes` | int | bytes still to download for this file |
| `reason` | str \| null | why `conflict` / `unavailable` (Japanese) |
| `verified` | bool | for `present`: the sha256 was measured while planning (false with `verify_existing=False`: size only). `download` always measures existing files again before skipping them |

DownloadResult:

| key | type | meaning |
|---|---|---|
| `model_id`, `root` | str | as in the plan |
| `downloaded` | str[] | `relpath` of the files downloaded (sha256 checked) |
| `resumed` | str[] | of those, continued from a `.part` with HTTP Range |
| `restarted` | str[] | of those, started over because the server ignored Range |
| `skipped` | str[] | already there; the sha256 was measured in this run and matched |
| `bytes_downloaded` | int | bytes received in this run |
| `seconds` | float | wall time |
| `files` | object[] | `relpath`, `dest`, `sha256` of every file |

## Python API (stable) / Python の公開の範囲

**Only the names in `genmodelfit.__all__` are the stable API** (tests/test_judge_model.py fixes the new ones). Functions
in submodules (`genmodelfit.recommend.judge`, `genmodelfit.hardware.classify_gpu`, …) are internal and may change in
any release. 互換の対象は `__all__` に入る名前だけ。サブモジュールの関数は内部のもの。

Judging one model with a GPU the app detected itself (0.6.0, #27). The generation (compute capability) and ComfyUI's
compute precision are decided by genmodelfit from the GPU name and driver; nothing runs nvidia-smi or PowerShell:

```python
import genmodelfit as gmf

fit = gmf.judge_model(
    "hf:stabilityai/stable-audio-3-small-sfx",  # id, or a model name / part of an id (like `info` / `download`)
    gpu_name="NVIDIA GeForce RTX 3060",
    vram_gb=12.0,
    driver="560.94",
    ram_gb=32.0,  # None = this PC's memory (as gmf.system())
    gpu_usable=None,  # None: genmodelfit decides from the driver / generation; True: the app checked it
    db=gmf.load_db(),  # optional; an app may pass its bundled list
)
row = fit.to_dict()  # the same dict as one item of `recommend --json` models ("Fit" above)
row["fit_level"], row["run_mode"], row["memory_required_gb"], row["estimate_confidence"], row["licenses"]
```

| function | returns | notes |
|---|---|---|
| `gmf.system_from(gpus=(), *, ram_gb=None, gpu_usable=None, windows=None)` | `SystemInfo` | `gpus`: dicts `{"name", "vram_gb", "driver"?, "compute_capability"?, "vendor"? (default nvidia)}` or `Gpu`. The largest usable NVIDIA GPU is judged; non-NVIDIA ones never are. `ram_gb` sets total = available and ignores this PC's page file (like `--ram`). `gpu_usable`: `None` = a GPU that no current PyTorch CUDA build supports, or whose driver is too old, is treated as CPU (reason in `notes`, docs/hardware.md); `True` = the caller verified it (generation and precision are still decided here); `False` = CPU. `windows` (default: this PC): Windows driver table and Windows ComfyUI precision (Pascal fp16). OS, CPU and free disk are this PC's. Bad GPU dicts raise `ValueError` |
| `gmf.judge_model(model, *, gpu_name=None, vram_gb=None, driver=None, compute_capability=None, ram_gb=None, cpu_only=False, gpu_usable=None, windows=None, settings=None, content="sfw", system_info=None, db=None, comfyui_version=None)` | `Fit` | `model`: an exact id is used whatever its content; otherwise the name lookup of `download` (filtered by `content`; several workflows of one model → the representative of the `recommend` row, the others in `fit.workflows`). Not found / several models → `gmf.ModelNotFound` (`.candidates`). PC: `system_info`, else `gpu_name` + `vram_gb` (both, else `ValueError`), else `cpu_only=True` (no GPU), else this PC (`gmf.system()`). `system_info` together with the GPU arguments → `ValueError` |

Fetching the distributed list from Python (0.7.1, #58). Also in `__all__`: `gmf.RemoteError` (a `DatabaseError`):

```python
db = gmf.update_from_release(
    "https://github.com/laxshmate/genmodelfit/releases/download/data-latest",  # default
    timeout=30,
    dest=data_dir / "db.json",  # default: the cache db.json
    force=False,  # True: accept a list older than the one imported before
    session=my_requests_session,  # default None: genmodelfit's own session
    state_dir=data_dir,  # default None: the cache folder
)
```

| parameter | meaning |
|---|---|
| `session` | a `requests.Session` (e.g. with `verify` set to the app's CA bundle, or proxies). All GETs of this call use it. Its `verify` / `proxies` / other headers are used as they are and the session is not modified. `~/.netrc` credentials are never sent: every request carries a no-op `auth` (requests reads netrc only when neither the request nor the session has `auth`), and redirects are followed by genmodelfit (`allow_redirects=False` per hop) because requests re-reads netrc on each redirect. The fixed `User-Agent` (`genmodelfit/<version> (+https://github.com/laxshmate/genmodelfit)`) is set per request. **Limits**: if the session itself has `auth` it is the caller's choice and is used for the first host only (never for a redirect to another host; an `Authorization` header set on the session is also dropped there). Cookies, hooks and adapters of the session are not touched by genmodelfit, but the session does what it normally does: custom non-`Authorization` headers set on it are also sent to redirect targets (requests' default), and `Set-Cookie` of the responses updates its cookie jar. An empty `auth` (`None`, `()`, `""`) counts as none. `verify` and `proxies` of the session are passed per request, so the caller's CA bundle wins over `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` (requests#3829); proxy environment variables fill only what the session lacks. Redirects are followed up to 10 times with a caller session (requests' default is 30; the session's `max_redirects` is ignored); an invalid redirect URL (e.g. port 99999) is a `RemoteError`. Signature check, `MAX_BYTES`, the one retry and the errors (`RemoteError`) are the same as without `session` |
| `state_dir` | where `release_state.json` (the `generated_at` of the list imported before) is read and written. Default: the cache folder. Environment variables and module globals are not changed |

Without `session`, the call uses `fetch.make_session()` as before (the private `remote._http_get` hook still works, but is not
stable API).

Equivalence: with `gpu_usable=True, windows=True, ram_gb=1024` it gives the same `Estimate` as the internal calls the
AI生成キット used up to genmodelfit 0.5.0 (`hardware.classify_gpu` + `comfy_precision` → `recommend.judge`, then
`recommend.vram_need` = `memory_required_gb` before rounding to 2 places).

**Name shadowing / 名前の上書き**: `genmodelfit.recommend` is the function above and hides the submodule of the same
name, so `import genmodelfit.recommend as r` binds the function. Use the functions in `__all__`; if a module is
really needed, `importlib.import_module("genmodelfit.recommend")` (internal, not stable).

## Database file (`db.json`)

```json
{"schema_version": 1, "generated_at": "2026-10-20T03:17:00Z", "sources": {…}, "models": [ModelEntry, …]}
```

`generated_at` is always an aware UTC ISO-8601 string ending in `Z`.
`downloads`, `likes`, `license_from`, `repo_license`, and (PR #13 review) `ModelEntry.license_name` /
`license_evidence` / `files_complete` / `size_floor_bytes` and `ModelFile.license_url` / `license_path` /
`license_evidence`, and (#3) `ModelFile.head_status` / `head_checked_at`, and (#30) `ModelEntry.workflow_precision` /
`precision_source` and `ModelFile.repo_precision` / `repo_precision_source`, and (#36, 0.5.0) `ModelEntry.languages` /
`runtime` / `description` / `card_languages` and `source: "manual"` were added without changing `schema_version` (optional fields; older
readers ignore unknown keys: `from_dict` drops them, `check_entry` only checks the keys it knows, and `source` is a free
string there. `tests/test_appinfo.py` reads the bundled list with a copy of the old reader).
Task `tool` was added for Image Tools / Video Tools templates without a known tag.

`sources` (written by `genmodelfit update`, see docs/sources.md):

| key | meaning |
|---|---|
| `comfyui-templates` | `templates_total`, `local`, `skipped` (by reason: `api` / `not_open_source` / `no_size` / `category`), `loaded`, `failed` |
| `huggingface` | `repos` (API calls), `failed`, `rate_limited` (429 waits), `rate_limit_waited_s`, `failures` (first 20), `unauthorized` (repo ids whose Hub API answered 401 — the anonymous API answers 401 both for private and for missing repos; their licenses stay unknown; 0.2.0), `unauthorized_missing` (the subset confirmed not to exist, `repo_precision.KNOWN_MISSING`). After `update --inspect-only`: `unauthorized_recheck` = `{unauthorized, private_or_missing, missing, readable, failed, checked_at}` (those repos asked again; `readable` ones get their license on the next full `update`) |
| `huggingface-search` | `queries`, `candidates`, `added`, `skipped` (by reason, incl. `duplicate_weights`), `excluded`, `failed_queries` |
| `partial` | bool: some upstream data could not be fetched (then `update` exits 1 unless `--allow-partial`) |
| `partial_counts` | `templates_failed`, `hf_repos_failed_transient`, `entries_affected`, `hf_search_failed`, `rate_limit_budget_exhausted`, `carried_over` |
| `license_policy` | version of the license rules (2). Values are carried over from a previous list only when it has the same version |
| `civitai` | why it is not used |
| `size_check` | `templates`, `equal`, `within_10pct` (index `size` vs the sum of the files) |
| `counts` | `models`, `modality`, `content`, `source`, `license` (null = unknown), `license_tier`, `files_complete_false` |
| `head_check` | `checked_at`, `urls` (unique HF URLs sent a HEAD), `status` (count per status code), `unchecked` (no answer), `not_sent` (files on other hosts, per host) |
| `file_headers` | `files`, `urls` (main weight files on HF), `from_cache`, `fetched`, `failed`, `skipped_offline`, `errors` (first 20), `checked_at`, `precision_unknown` (entries whose precision is still null) |
| `sampler_stages` | only after `update --inspect-only`: `templates`, `failed`, `multi_stage` (entries with two or more sampler sets), `workflow_precision` (entries with a precision inferred from the loader node), `checked_at` |
| `repo_precision` | `files` (largest main files on HF whose header and name tell no precision), `repos`, `from_cache`, `fetched` (requests), `hf_api`, `config`, `unknown` (files decided by each / by none), `skipped_offline`, `unauthorized` (repos whose API answered 401), `size_mismatch` (repos whose API dtype was not used because the size did not match), `errors` (first 20), `checked_at`, `precision_unknown` (entries whose precision is still null, final), `precision_source` (entries per source, `unknown` = null) |
| `manual` | `models` (number of hand-added models from `data/manual.json`), `ids` (0.5.0) |
| `app_fields` | `languages_prompt`, `languages_speech` (entries with a non-empty list), `runtime`, `description`, `from_card` (entries with a saved Hugging Face card language, `card_languages`) (0.5.0) |
| `comfyui` | `latest_release` (tag of the latest ComfyUI release, e.g. `v0.42.1`; null if it could not be read), `url`, `checked_at` |
| `excluded` | number of models dropped by the exclusion rule (names are not kept) |
| `warnings` | str[] |

## update

`genmodelfit update --json` prints the envelope with `output` (path written) and `sources` (above).
On failure it prints `error` and exits 1; the previous cache is left untouched.
`update --from-release --json` prints a smaller object: `genmodelfit`, `command` (`update`), `models` (**int**: number of
models taken in), `generated_at`, `output`, `status` (DataStatus with `source: release`).
`update --json` (rebuild / `--heads-only`) also has `status` (DataStatus of the list written; `source` is `cache` for the
cache, `path` for `--output` elsewhere).
`update --heads-only` reads the current database (`--db`, cache or bundled), redoes only the HEADs and the ComfyUI
version, and writes it back (`--output`, default the cache).
`update --fields-only` reads the current database the same way and rebuilds only the app fields (`languages` /
`runtime` / `description`, from `data/overrides.json`, the tables and the Hugging Face cards) and the hand-added
models (`data/manual.json`): it reads the cards again (Hub API) and, for the hand-added models only, sends HEADs,
reads the file headers and compares the licenses with the upstream cards; with `--offline` it sends nothing and
uses the saved `card_languages`. The other entries' files, licenses, precision and HEAD results are not touched,
except that the hand-added components (`components` in `data/manual.json`, below) are added to their entries (with a
HEAD of the new files only, none with `--offline`).
`update --inspect-only` reads the current database the same way and redoes only what is read from the files
themselves: the main weight files' headers (`header_precision` / `header_params`, then `precision`), the official
templates' `sampler_stages` and `workflow_precision`, the repo precision (`repo_precision`: Hub API / `config.json`,
cached in `<cache>/repo_precision.json`), and it asks the Hub API again for the repos in
`sources.huggingface.unauthorized`; nothing else changes (except the review marks, see "review"). Headers already in `<cache>/file_headers.json` (keyed by URL and
sha256) are not read again; requests are sent one at a time with a short pause and honor `Retry-After`. With
`--offline` it uses only that cache and sends nothing (the templates are then not re-read).

Measurements live in `measurements.json` (`{"unit": "GiB", "unit_note": "…", "measurements": [Measurement, …]}`), bundled plus
`<cache>/measurements/*.json`.
