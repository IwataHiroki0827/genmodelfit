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
| `command` | str | `system` / `recommend` / `fit` / `search` / `types` / `info` / `update` / `bench` / `gate` |
| `system` | System | the PC that was judged (after `--vram` / `--ram` overrides). Not in `gate` / `update` / `bench` |
| `database` | object | `origin` (`bundled` / `cache` / `path`), `generated_at`, `models`, `measurements` (counts). Not in `system` / `bench` / `update --from-release` |
| `error` | str | only on failure (exit code 1) |
| `candidates` | object[] | only when `info` matched several models (see info) |

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

## recommend / fit / search

The three commands print the same shape (Issue #17, modeled on llmfit's `recommend --json`):

| key | type | meaning |
|---|---|---|
| `filters` | object | values actually used: `modality` (str[] \| null), `content`, `min_fit`, `license` (str[] \| null), `task` (str[] \| null: the names given), `all_workflows`, `limit`; `recommend` adds `per_type`, `auto_only`, `include_tools`; `search` adds `query` |
| `matched` | int | rows that passed every filter, before `limit` / `per_type` (models; workflows with `--all-workflows`) |
| `hidden` | object | why the other entries are not shown: `{reason: count}`, only reasons with a count (below) |
| `groups` | object[] | per type, in the order image, video, music, sfx, tts, 3d (only the `-m` types when given): `{modality, matched, shown}` |
| `models` | Fit[] | best first (order below) |

**One item per model** (below, "Models"): entries that differ only in precision or in workflow (text-to-image, edit,
LoRA, …) are collapsed into one item; the others are listed in `workflows` / `variants`. `--all-workflows`
(`all_workflows=True` in Python) gives one item per list entry as before.

`recommend`: without `-m` and `--task` it shows **one section per type** (image, video, music, sfx, tts, 3d), the top
`-n` (default 3) of each (`filters.per_type`, `limit` null); a type with nothing runnable prints one line in text mode
and has `matched: 0` in `groups`. With `-m` or `--task` it is one list, `-n` default 10. In JSON `models` is ordered by
type, then by the order below. Other defaults: `min_fit` `marginal`, utility tools hidden, models with unknown size never
shown. `fit` and `search` show every level by default (`--min-fit` to narrow) and no limit. `search <query>` matches a
part of the id or name (case-insensitive); entries that do not match are not counted in `hidden`. With no match it prints
`models: []` and exits 0 (text mode prints `(no match / 該当なし)`); a blank query exits 1.
`content` defaults to **`sfw`**: uncensored models appear only with `--content all` or `--content uncensored`.
既定は sfw。無検閲は `--content all|uncensored` のときだけ出す。

`hidden` reasons, each entry counted once under the first reason that applies, in this order:
`modality` (other `-m`), `uncensored` / `sfw` (by `--content`), `tools` (recommend without `--include-tools`),
`task` (other uses than `--task`), `license` (`--license`), `size_unknown`, `too_tight`, `below_min_fit` (perfect / good /
marginal below `--min-fit`), `held` (`--auto-only`), `same_model` (collapsed into another row of the same model).
**`matched` + the sum of `hidden` = the number of entries looked at** (every entry of the list; for `search`, the entries
whose id or name matched). Tested in `tests/test_cli_json.py`.

`--license a,b`: every file's license (`ModelFile.license`; the entry's `license` when it lists no files) must be in the
list (case-insensitive). An unknown license (`null`) never matches.

**Order / 並べ方**: models that run (`fit_level` is not `too_tight`) → `auto: true` → `score` (desc) → `id`.
動くもの → 自動でおすすめに出せるもの → 点数の高い順 → ID。

### Models / モデルのまとめ方

Like llmfit: a different **size** is a different model, a different **precision** is the same model (chosen
automatically), and **workflows** are uses of that model. `group_key` is built from the main weights
(`src/genmodelfit/groups.py`):

1. Main files = role `main` (by the estimator's rule: ComfyUI folder → `role` → HF folder → name). Text encoders
   (umt5, …) and VAEs are never the key. With several main files, the largest one and those ≥95% of its size (a pair
   such as Wan 2.2 high / low noise); without any, the largest file that is not a text encoder / VAE / clip_vision;
   without files, the entry itself.
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

Which entry is shown for a model: among the entries that passed the filters, the best fit level first, then the
higher precision (bf16 / fp16 / fp32 > fp8 > int8 > GGUF Q8 > Q6 > Q5 > Q4 > lower; unknown precision counts as full
precision), then the order above. So on 8 GB Z-Image Turbo shows its int8 workflow (good) and on 24 GB the bf16 one, and
with `--task edit` the representative is an edit workflow. The shown entry's `auto`, score and VRAM are used for the
order.

On the bundled list (341 entries) this gives 225 models (188 with content `sfw`); 52 of them have more than one entry.

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
pose / geometry map or a segmentation, and any audio separation) are hidden by `recommend` unless `--include-tools` or a
tool use is given in `--task`.

`genmodelfit types --json`: `filters` (`content`), `types` (`[{modality, label, label_ja, models, workflows, uses: {use:
workflows}}]`; `models` = number of models as above, `workflows` = list entries; an entry with two uses counts in both),
`uses` (`[{use, label, label_ja, tasks}]`), `aliases`.

### Fit (one item of `models`)

Python `Fit.to_dict()` returns exactly this dict (`gmf.recommend()` / `gmf.search()` give the same list as the CLI).

| key | type | meaning |
|---|---|---|
| `id` | str | template name (`image_z_image_turbo_int8`) or `hf:<owner>/<repo>` |
| `name`, `modality`, `tasks` | | modality ∈ `image` `video` `music` `sfx` `tts` `3d`; `tasks` is a list (`t2i`, `i2v`, …) |
| `content` | str | `sfw` or `uncensored` (see README) |
| `source` | str | `comfyui-template` / `huggingface` / `manual` |
| `fit_level` | str | `perfect` / `good` / `marginal` / `too_tight` (= `estimate.fit`) |
| `fit_label` | str | `Perfect` / `Good` / `Marginal` / `Too tight` |
| `run_mode` | str | `gpu` / `offload` / `cpu` / `none` (= `estimate.run_mode`) |
| `run_mode_label` | str | `GPU` (all on GPU) / `GPU~` (parts swapped stage by stage) / `GPU+RAM` (offload) / `CPU` / `-` (does not run) |
| `score` | float | 0–100 (below) |
| `score_components` | object | `fit`, `headroom`, `readiness`, `basis`, `popularity`, each 0–100 |
| `memory_kind` | str | `vram` (judged on the GPU) or `ram` (CPU) |
| `memory_required_gb` | float \| null | the figure the level was judged on: VRAM for all-in / swapping / offload (the table's VRAM column); RAM for CPU. When a GPU is needed but missing: the minimum VRAM. null when the size is unknown |
| `memory_available_gb` | float \| null | this PC's VRAM (or RAM for CPU); `0.0` when a GPU is needed but missing |
| `utilization_pct` | float \| null | `memory_required_gb / memory_available_gb × 100` (the table's Mem%) |
| `ram_required_gb` | float \| null | system RAM needed (= `estimate.ram_gb`; null when the size is unknown) |
| `disk_size_gb` | float \| null | download size counted (one variant per part, = `estimate.disk_gb`) |
| `total_size_gb` | float \| null | size of all listed files |
| `license` | str \| null | strictest license of the files (`apache-2.0`, `mit`, `other`, …; null = unknown) |
| `licenses` | (str \| null)[] | the distinct licenses of the files (null last = some file unknown) |
| `auto` | bool | can be auto-recommended (= `gate.auto`) |
| `held_reasons` | str[] | failed gate checks (= `gate.held_by`) |
| `estimate_confidence` | str | `measured` / `estimated` (= `estimate.basis`) |
| `notes` | str[] | reasons in Japanese (= `estimate.reasons`) |
| `files_count` | int | download files listed |
| `use` | str | the use shown (Uses below) |
| `precision` | str \| null | the chosen entry's precision (`ModelEntry.precision`: `bf16`, `fp8`, `int8`, `gguf-q8_0`, …; null = not stated, usually full precision) |
| `group_key` | str | the model this entry belongs to (Models below) |
| `variants` | object[] | per precision, the best entry that passed the filters: `{precision, workflow_id, memory_required_gb, fit_level, auto, disk_size_gb}`, best precision first (always includes this entry) |
| `workflows` | object[] | **all** other entries of the same model, also those filtered out: `{id, name, tasks, use, precision, fit_level, memory_required_gb, disk_size_gb, auto, matched}` (`matched`: passed the current filters), best first |
| `estimate` | Estimate | full estimate (below; `limits` with `--limits`) |
| `gate` | Gate | full gate result (below) |

### Score / 点数

`score = 0.45·fit + 0.15·headroom + 0.20·readiness + 0.10·basis + 0.10·popularity`, rounded to 0.1
(`recommend.WEIGHTS`). Each component is 0–100:

| component | formula |
|---|---|
| `fit` | perfect 100, good 70, marginal 35, too_tight 0 |
| `headroom` | `min(100, 100 × (1 − vram_gb / VRAM) / 0.5)`, 0 without a GPU, when the size is unknown or when the all-in VRAM (`estimate.vram_gb`) does not fit. Half of the VRAM free → 100 |
| `readiness` | 100 when `auto`; otherwise the sum of the passed checks' points: license 25, gated 15, official_workflow 15, custom_nodes 15, comfyui_version 10, url_reachable 5, sha256 5, content 5, size_known 5 |
| `basis` | 100 when `measured`, 0 when `estimated` |
| `popularity` | `min(100, 20 × log10(1 + usage))`; usage = template uses, or Hugging Face downloads / 100 (100 at 100k) |

The score never goes down when VRAM or RAM increases (the level never drops, `headroom` compares against the all-in
VRAM which does not depend on the PC, and `basis` only turns `measured` when a GPU is present). Tested in
`tests/test_invariants.py`. VRAM・メモリを増やしても点数は下がらない(試験あり)。

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
| `marginal` | runs only by offloading to RAM, on CPU, or with a small RAM shortfall (page file); slow / 逃がせば動く・CPU・メモリが少し足りない。遅い. A model that does not fit a (small) GPU even with offloading is also judged as if there were no GPU, and the better result is used, so adding VRAM never lowers the level |
| `too_tight` | does not run (VRAM, or RAM short by more than 1.5×), **or cannot be judged because the size is unknown** (`unknown_size: true`) / 動かない、または大きさが分からず判定できない |

## Gate

Computed every time (never stored in `db.json`). Details and the default license allowlist: docs/gate.md.

| key | type | meaning |
|---|---|---|
| `auto` | bool | true only when every check passed |
| `held_by` | str[] | ids of the failed checks, in the order below (`held_by[0]` is shown, shortened, in the table's Auto column and as `held:<id>` by `gate --only-held`) |
| `comfyui_version` | str \| null | the ComfyUI version compared against (`--comfyui-version`, else `sources.comfyui.latest_release`) |
| `checks` | Check[] | always the 9 checks, in this order: `license`, `gated`, `official_workflow`, `custom_nodes`, `comfyui_version`, `url_reachable`, `sha256`, `content`, `size_known` |

Check: `id` (stable, machine-readable), `passed` (bool), `detail` (Japanese, for people; wording may change).

## gate

`genmodelfit gate --json`: `policy` (`license_allowlist`, `comfyui_version`, `allow_uncensored`, `max_head_age_days`),
`filters` (`modality`, `only_auto`, `only_held`), `summary` (`models`, `auto`, `held`,
`failing` = models failing each check, `first_failing` = models whose first failed check is that one),
`models` (`{"id", …Gate}`; all, or only auto / held with `--only-auto` / `--only-held`).

## info

`model` (ModelEntry), `estimate`, `gate`, `measurements` (Measurement[]), `group_key`, `same_model` (the other
entries of the same model, each like a `workflows` item, judged for this PC, best first).

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
`comfyui_version`, `offload`, `source`, `date`, `note`, `outcome`, `standard`, `vram_metric`.

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
`update --from-release --json` prints a smaller object: `genmodelfit`, `command` (`update`), `models` (**int**: number of
models taken in), `generated_at`, `output`.
`update --heads-only` reads the current database (`--db`, cache or bundled), redoes only the HEADs and the ComfyUI
version, and writes it back (`--output`, default the cache).

Measurements live in `measurements.json` (`{"unit": "GiB", "unit_note": "…", "measurements": [Measurement, …]}`), bundled plus
`<cache>/measurements/*.json`.
