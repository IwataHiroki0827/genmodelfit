# Fairness, order and Score

**English** · [日本語](scoring.ja.md)

The rules for how the `recommend` / `fit` / `search` tables, `--json` and the Python functions use measurements, order the models and set the Score (0.11.0, Issue #95).
The memory formula and its coefficients are in [accuracy.md](accuracy.md), the level bands (60 / 85 / 98% use) and the caps per run mode in [schema.md](schema.md) "Estimate", and whether a model may be recommended automatically in [gate.md](gate.md); this page does not change them.
The model is [llmfit](https://github.com/AlexsJones/llmfit) (2f35825: `llmfit-core/src/fit.rs`, `analysis.rs`, `benchmarks.rs`); where genmodelfit differs, the reason is given.

## Contents

- [Four checks](#four-checks)
- [Overall flow](#overall-flow)
- [Settings used for judging](#settings-used-for-judging)
- [When measurements set the level](#when-measurements-set-the-level)
- [Shifting values with measurements](#shifting-values-with-measurements)
- [Measurements of templates without a size input](#measurements-of-templates-without-a-size-input)
- [Score](#score)
- [Popularity](#popularity)
- [Order](#order)
- [One row per model: the representative](#one-row-per-model-the-representative)
- [Differences from llmfit](#differences-from-llmfit)
- [Tests](#tests)
- [Not decided yet](#not-decided-yet)

## Four checks

An item goes into the verdict, the score or the order only if it passes all four checks. Anything that fails goes into a mark, a note or a filter instead.

| Check | Condition |
|---|---|
| Explainable | It can be stated in one sentence in the README, and users can check it from the numbers in the table |
| Evidenced | Backed by primary sources, measurements, or llmfit's code and history. Constants without evidence are called "chosen values" and kept to a minimum |
| Fair | The author's PC, whether the author measured a model, the author's app and the date of the run do not change other users' levels, scores or order. Same list, same PC and same options give the same result for everyone |
| Testable | Inputs and expected values can be fixed in an automated test |

## Overall flow

1. PC: VRAM, GPU vendor and generation, system RAM ([hardware.md](hardware.md))
2. Settings: the per-type defaults or the user's options ("Settings used for judging" below)
3. Estimate: the VRAM and RAM each run mode needs ([accuracy.md](accuracy.md)); measurements shift the values
4. Level: pick the run mode and set Perfect / Good / Marginal / too tight from the use. Measurements set the level only when two or more independent sources agree on the same GPU
5. One row per model, then the order, then the Score

## Settings used for judging

**In one sentence**: without options, every model is judged at the same size for its type.

- Settings not given are filled with the per-type defaults (`estimate.DEFAULT_SETTINGS`) whether or not the model has measurements: image 1024x1024, video 832x480 with 81 frames, music 30 s, sound effects and speech 10 s, 3D no size.
- A measurement's settings are used only to work out how far to shift the working memory.
- Up to 0.10, a model with measurements was judged at the measurement's settings when no options were given (the author's app's "fast" quality: Z-Image 768px, SDXL 832px, Wan2.1 512x512 with 49 frames, Wan2.2 5B 640x640 with 73 frames, ACE-Step 20 s). Only the models the author measured were compared at smaller settings, which is not fair, so this was dropped.
- So most of the author's measurements (image, video, music) no longer match the default settings: their `basis` becomes `estimated` (`basis_detail: calibrated`) and the table's VRAM loses the `m`. The 3D and utility measurements match the settings and keep the `m`. The author plans to measure again at the defaults ("Not decided yet").
- An app (such as AI生成キット) that wants its own settings passes them per model: `judge_model(..., settings=...)` or `--width` and so on. No API for a per-type settings table is added (the app's settings differ per model, which a table cannot express).
- Evidence: llmfit also compares every model under the same assumption (context length).
- Test: `tests/test_scoring.py::test_a_settings_do_not_depend_on_measurements`.

## When measurements set the level

**In one sentence**: if two or more independent sources ran to the end on the same GPU as yours with the same settings and agree, the level comes from them (and the amount used for the level is the table's VRAM).

`estimate.measured_level`. The level comes from measurements (`Estimate.fit_from: measurement`) only when all of the following hold, and only when it is better than the estimated level.

1. **Same GPU**: the same rule for the author's measurements and other users' (`estimate.same_gpu`).
   - The PCI device IDs match when both have one; otherwise the normalized names match (generic names never match).
   - For dedicated-VRAM GPUs, the VRAM ratio (smaller ÷ larger) must also be 0.95 or more (`MEASURED_VRAM_TOLERANCE`; "VRAM ratio for the same GPU" below).
   - Up to 0.10, the author's measurements counted on any GPU of the same vendor with at least 0.97 × the measured VRAM (for example, Wan2.2 5B measured on an RTX 5080 was Good / GPU on an RTX 4060 Ti 16 GB). That favoured the author and was dropped.
2. **Same settings**: resolution, frames, length, batch and precision match. Same vendor (`gpu_backend`, else the GPU name). Measured with the current list's files (`files_fp`) on a GPU whose name is not a generic name or placeholder (`gpu_known`; 0.11.0 dropped the name table).
3. **Two or more independent sources** (`MIN_LEVEL_SOURCES`):
   - The author's measurements (`measurements.json`; this PC's saved bench results are treated the same): **however many rows there are for the same model, GPU and settings, together they are one source** (`estimate.count_sources`). The author alone never sets the level.
   - Other users' measurements: one per commit = pull request that added submissions (#90). A source is one PR; several PRs from the same person count separately, so the maintainer checks for that when merging. Aggregated rows carry the IDs of the sources they used (`source_ids`: the first 12 characters of the commit, or `file:<first 12 of the content sha256>`), and sources are counted as the **union of the IDs** across rows. Submissions from one pull request split into separate rows (with and without a GPU device ID, or by measuring method `vram_metric`) are still one. Older rows without IDs are not added up; only their largest `n_sources` is used.
   - The author and other users are added up for the same GPU and settings (the author + one submission = 2).
   - With only one source, the level is not set and the values are only shifted ("Shifting values with measurements" below). The reasons say "only one source".
4. **They agree** (`CONSISTENT_REL`; the same definition as `consistent` in `contrib aggregate`):
   - No OOM record: no row with `outcome: oom` for the same setup, no aggregated row with `n_oom`, and none with `runs_evidence: false` (OOM sources at least as many as ok sources).
   - Each aggregated row agrees inside (rows with two or more sources have `consistent: true`).
   - Across rows, the spread (max − min) of the values (with the context added) is within 15% of the median. Aggregated rows are compared by their median.
5. **Amount used for the level** = max(largest agreeing peak + context, weights + fixed part).
   - Context: `torch_reserved` measurements leave out the CUDA / HIP context, so 0.5 GiB is added (`TORCH_RESERVED_CONTEXT_GB`).
   - Weights + fixed part: full load with zero working memory (the same physical floor as for the values).
   - **If peak + context is below weights + fixed part, the level is not set.** A peak below the full-load floor is a record of running with part of the weights kept in system RAM, not evidence that everything fits on the GPU (the same reasoning as Wan2.2 5B below; the values are still shifted). The reasons say it is below the full-load floor.
   - This amount is what the table's VRAM and Mem% show, and JSON's `estimate.measured_level_vram_gb` (and `memory_required_gb`). The raw peak is `estimate.measured_peak_vram_gb`.
   - If the amount exceeds this PC's VRAM, the level is not set.
6. **Level** = the band of this amount ÷ VRAM, at least Good (the runs finished). If any agreeing measurement hit the GPU limit (dynamic VRAM used up the free space), at most Good. The run mode is GPU.

`Estimate.sample_count` is the number of independent sources behind the value (1 for the author's measurements however many rows, `n_sources` for other users, the sum of agreeing sources when measurements set the level). Up to 0.10 it was the number of submissions (`n_used`).

Examples (bundled measurements):
- **Example: Wan2.2 5B on an RTX 5080 16 GB** (with 640x640, 73 frames; the author's two rows, 15.8 / 15.0 GB, are one source together). Weights + fixed part is about 18.4 GB, which cannot all fit in 15.9 GB. The measurement ran to the end with ComfyUI's dynamic VRAM keeping part of the weights in system RAM (offload: dynamic, a value at the GPU limit). So it is a record of "runs by offloading", and the right level is the offload level (Marginal, GPU+RAM). Even with agreeing submissions from other users, the floor of the amount (weights + fixed part) exceeds the VRAM, so measurements never make it a GPU level. From 0.7.0 to 0.10 it was Good, GPU (#26).
- SDXL on an RTX 5080 (with 832px): only one author row, so no level from measurements. One agreeing submission from another user on the same GPU and settings would set it.

The four checks:
- Explainable: the sentence above.
- Evidenced: llmfit carries measurements over only to PCs with the same configuration (`benchmarks.rs:546-558`) and never changes the level from them. genmodelfit uses "ran to the end on the same GPU with the same settings" as stronger evidence than the formula for the level (#26, 2026-10-22), so it asks for more than llmfit: two or more sources.
- Fair: one fake submission, one person submitting many times, or the author alone do not change the level. A fake small value cannot lower the amount below weights + fixed part (for example, "ran in 1.0 GB" for a Flux that needs 22 GB does not make it Perfect on an 8 GB RTX 3070).
- Testable ("Tests").
- Of the 2026-10-22 decision "other users' measurements are used from one submission", the part that sets the level changed (values are still used from one submission).

**With `--vram`**: `--vram` keeps the real GPU's name, so once the VRAM ratio drops below 0.95 it is no longer the same GPU and measurements no longer set the level (for example `--vram 17.5` on an RTX 5080 PC). That is a judgement for a GPU with a different amount, so this is correct. The [schema.md](schema.md) promise "more VRAM never lowers the level" carries this exception.

**VRAM ratio for the same GPU** (0.97 → 0.95 in 0.11.0; `estimate.MEASURED_VRAM_TOLERANCE`)

- **In one sentence**: the same card counts as the same GPU whatever the OS, and variants with another amount (8 GB and 16 GB, for example) count as different GPUs.
- Evidence: Linux CUDA reports the total minus a reservation (5934 MiB = 5.80 GiB for a 6 GB card; an RTX 2060 report on the NVIDIA developer forum). Linux totals are 5.80 / 7.79 / 9.78 / 10.76 / 11.77 / 23.69 GiB; Windows reports 6.0 / 7.99 / 15.92 and so on. For 6 GB, 5.80 ÷ 6.0 = 0.967 missed 0.97. Variants with another amount differ by a ratio of 0.83 or less (RTX 4060 Ti 8 / 16 GB 0.5, RTX 3060 8 / 12 GB 0.67, RTX 2060 6 / 12 GB 0.5, RTX 3080 10 / 12 GB 0.83), well below 0.95.
- Fair: the same value for everyone's measurements; Linux and Windows owners of the same card share each other's measurements.
- Test: `tests/test_scoring.py::test_same_vram_ratio_linux_and_windows` (Linux 5.80 matches Windows 6.0; the 8 GB and 16 GB variants never match).

**Non-NVIDIA GPUs**: estimates are capped at Good. Only when measurements set the level as above does the level go up to theirs ([hardware.md](hardware.md)).

## Shifting values with measurements

**In one sentence**: for a model with measurements, the formula's working memory is shifted to match them, on any GPU.

- Unchanged ([accuracy.md](accuracy.md) "How measurements are used"). Other users' measurements are used from one submission on PCs with the same GPU.
- Only the models the author measured get more accurate values, and the shift can change the run mode or level (for example HY 3D 2.0 (`3d_hunyuan3d_image_to_model`) on an RTX 4060 8 GB is Good / GPU (6.7 GB) with the measurement and Good / GPU~ (full load 6.85 GB) without). This is accepted as the other side of making values accurate; the table's `m` / `c` marks values that come from measurements.
- Since #100 a negative shift is clipped at 0 per stage (never below weights + fixed part), a positive shift goes to the all-on-GPU value only, and a shift from a measurement pinned at the GPU limit goes to all three ([accuracy.md](accuracy.md) "How measurements are used" 1).
- Evidence: llmfit also corrects estimates with measurements (one ratio from a machine's measurements for every model; genmodelfit shifts per model, by about −1 to +2 GB with the per-stage formula of #100).
- Test: `tests/test_scoring.py::test_d_values_shift_by_at_most_the_delta` (the difference with and without measurements is within the shift).

## Measurements of templates without a size input

**In one sentence**: a model whose size comes from the input image is judged without a size.

- Added in #90 (`estimate.NO_SIZE_INPUT`, a table the maintainer checked). The judging settings have no size, so measurements that bench writes without a size (BiRefNet, MoGe) match and are used.
- Test: `tests/test_scoring.py::test_g_measurements_without_size_are_used`.

## Score

**In one sentence**: the Score is this PC's memory headroom: 100 while the tightest resource of the run mode is at most 70% used, lower above that (speed, quality and popularity are not in it).

### Formula

```
r = use (table below; unrounded)
memory_fit = 100                                     (r ≤ 0.70)
           = 100 × exp(−0.5 × ((r − 0.70) / 0.20)²)    (r > 0.70)
Rows that run (Perfect / Good / Marginal) never go below 37.5
Rows that do not run or are not judged are 0
Score = memory_fit rounded to one decimal (Python's round); the table shows an integer
```

r per run mode (the table's Mem% and JSON's `utilization_pct` are this r):

| Run mode | r |
|---|---|
| GPU | full-load VRAM ÷ VRAM |
| GPU~ | largest VRAM when the parts are swapped in turn ÷ VRAM |
| GPU+RAM | max(offloaded VRAM ÷ VRAM, RAM needed ÷ system RAM) |
| CPU | RAM needed ÷ system RAM |
| Level set by measurements | amount used for the level ÷ VRAM |
| Any run mode short of system RAM | max(the value above, RAM needed ÷ system RAM) |

- Examples: r = 0.60 → 100.0, 0.80 → 88.2, 0.85 → 75.5, 0.90 → 60.7, 0.98 → 37.5.
- The constants 0.70 and 0.20 are llmfit's `fit_score` (`fit.rs:2335-2349`). llmfit PR #648 made the curve one-sided (free space is not a loss) and PR #100 removed the cliff. The input r is, as in llmfit, "the share of the tightest resource of the run mode".
- The floor 37.5 is the value at r = 0.98 (the edge of Marginal). It covers rows that run although r exceeds 0.98: levels set by measurements, and rows short of system RAM that run with the page file (not only CPU but also GPU, GPU~ and GPU+RAM; r is then the RAM share and Mem% can be 110–130%; the level is Marginal).
- Relation to levels: Perfect (GPU, r ≤ 0.60) is always 100. Good (GPU) is 100 to 75.5, Marginal (GPU) 75.5 to 37.5. GPU+RAM and CPU are capped at Marginal, so a high Score does not mean fast.
- The Score barely matters for the order (levels and popularity come first). Its job is to show the memory headroom as one number. `--sort score` orders by it.
- With one row per model, the Score is **the representative's value** (it matches the table's Mem%). The "highest score among the members" of 0.10 was dropped.
- JSON: `score` (= memory_fit) and `score_components = {"memory_fit": …}`. The keys up to 0.10 (`fit`, `headroom`, `readiness`, `basis`, `popularity`) were removed (no reuse of a name with a new meaning).

### The promise changed

Up to 0.10 the promise was "more VRAM or RAM never lowers the Score". The new Score can drop when more VRAM switches the representative to a heavier precision or run mode (for example Z-Image Turbo int8: 100 on GPU~ at 14 GB, 82.3 all on GPU at 16 GB; llmfit behaves the same). The promise now covers levels only (with the `--vram` exception above). A test checks that more RAM never lowers the Score.

### Left out

| Candidate | Verdict | Reason |
|---|---|---|
| Level points (45% in 0.10) | Removed | The order already uses the level first, so it would count twice. llmfit's Score has no level either |
| Popularity (10% in 0.10) | An order key | Mixing it in needs weights (chosen values) and blurs what the score means (fit or popularity) ("Popularity" below) |
| Readiness (20% in 0.10) | Removed | It is the condition for the author's app to recommend automatically, unrelated to the fit to the PC. Handled by the `--json` order and `--auto-only` |
| Measured or not (10% in 0.10) | Removed | Models the author measured ranked higher in everyone's table. Used only for shifting values and the marks |
| Precision penalty | Not added | The order (16-bit ≳ 8-bit > 4-bit) is supported by the FLUX.1 paper (SVDQuant arXiv 2411.05007), point values are not. Only the rule that picks the higher precision within a model ("One row per model") |
| Size | Not added | No source relates size and quality across different generation model families |
| Recency | Not added | The list only has template creation dates. It changes with the run date (a weakness of llmfit) |
| Family reputation | Not added | A subjective table (llmfit's has no source either) |
| Speed (seconds) | Not now | No formula or calibration yet ("Not decided yet") |

## Popularity

**In one sentence**: popularity is the rank (0–100) of how much a model is used, counted from the same place within the same type, and it is only an order key (not in the Score).

`popularity.py`; JSON `popularity`.

- The unit is the model (a row: all workflows of the same model; `--all-workflows` rows carry the model's value). The count is the largest among the members. Community templates count too (they use the same model).
- Two sources, never mixed because their numbers mean different things:
  - ComfyUI templates: `usage` in index.json (described upstream as "Usage count"; period and unit are not published: Comfy-Org/workflow_templates `templates/index.schema.json:144`, `scripts/sync/sync_data.py:1252`).
  - Hugging Face: all-time downloads (`downloads_all_time`). Manually added models (manual) with a Hugging Face repository join this source.
  - A model with members in both (for example Stable Audio Open for sound effects) uses the template count.
- Rank = the percentile (mid-rank) among all models of the same type and source in the list used for judging (bundled or after `update`; before any filters, uncensored included):
  `100 × (models with a smaller count + models with the same count including itself ÷ 2) ÷ models in the pool`
  - Example: five models with counts 10, 20, 20, 40, 0 → 30 / 60 / 60 / 90 / 0.
- Models without a count or with 0: 0 (the bottom). They join the pool as count 0.
- Manually added models (manual; 3 today) have no count, so their rank is 0 (the bottom). No count is fetched over the network. Only the author's own additions are disadvantaged, so fairness holds.
- Sources with fewer than 5 models in a type: models with a count get 50 (rank differences would not mean much).
- Comparing template ranks and Hugging Face ranks with the same key is a chosen value ("treat the top of each source alike").
- Up to 0.10 it was 10% of the Score and Hugging Face downloads (30 days) were divided by 100 to match the template counts. That conversion was dropped.
- The four checks: explainable / evidence partial (the meaning of the counts is not published, but usage counts are more objective than llmfit's reputation table) / fairness partial (checked below) / testable.

**Bias check**: since most users have ordinary PCs, lighter models might become more popular. In the bundled list (2026-10-27), the Spearman correlation between the full-load VRAM (default settings) and the usage count of template models is −0.05 for image (78 models), +0.22 for video (63), +0.48 for music (10) and +0.59 for 3D (9). No "lighter is more popular" bias shows.

The weekly update (`python -m genmodelfit.weekly`) computes the same correlation for types with 20 or more template models ("人気のかたよりの確認" in the PR body) and adds a warning when it falls below −0.3 (a signal to consider dropping popularity as an order key; `popularity.vram_bias`). The warning also goes into the result's `warnings` and makes `merge-check` not safe (`popularity_bias:<type>`), so that week is not merged automatically and the owner reviews it. Music, 3D and speech have too few models for this check.

## Order

### The `recommend` table

Keys in order:
1. Runs → not judged → does not run
2. Level: fits without offloading (Perfect / Good with GPU or GPU~) → Marginal
3. Within Marginal only: run mode GPU → GPU~ → GPU+RAM → CPU (fastest first). Not used among rows that fit without offloading (so that the 24 GB Z-Image row whose representative is bf16 Good / GPU~ does not sink, #28)
4. Official templates → community templates
5. Popularity rank, highest first
6. Score, highest first
7. ID (fixes ties independently of the input order; llmfit uses the name)

Why (difference from llmfit): llmfit only puts models that do not run last and otherwise orders by Score. genmodelfit looks at the level first. The level is the answer to "does it run" and matters more than the Score (memory headroom) and popularity (#84).

### `fit` / `search`

- Default (`--sort type`): type → level (Perfect / Good / Marginal / not judged / does not run) → official → popularity → Score → ID
- `--sort score`: key 1 → level (fits without offloading / Marginal) → Score → popularity → ID
- `--sort vram`, `disk`, `name`: key 1, then that value (unchanged)

### `--json` and the library

- After key 1, models that may be recommended automatically (`auto: true`) come first, because the app shows recommendations without a person checking. The conditions (such as the license allowlist) can be changed with `GatePolicy` / `--license-allowlist`.
- Key 1 (runs → not judged → does not run) now matches the table (up to 0.10 the library put "does not run" before "not judged").
- The table (`recommend` text output) does not use automatic recommendation for the order, so the date (link-check age) and the allowlist do not change the table's order or the Score.

## One row per model: the representative

Unchanged ([schema.md](schema.md) "Models"): use order → among entries that fit without offloading, the highest precision → otherwise the best level (`recommend.variant_key`). The precision order agrees with "16-bit ≳ 8-bit > 4-bit".

## Differences from llmfit

| Point | llmfit | genmodelfit | Reason |
|---|---|---|---|
| Score contents | quality, speed, fit, context length | fit only | There is no evidenced way to measure quality and speed for generation models yet; added once there is |
| Order | does not run last → Score | runs → level → … → popularity → Score | "Order" |
| Caps for slow run modes | offload and CPU up to Good | GPU~ up to Good, GPU+RAM and CPU up to Marginal | Decided 2026-10-20. Generation reads the weights dozens of times per image, so offloading slows it a lot (needs measurements) |
| Measurements and levels | never change the level | set the level when two or more independent sources agree on the same GPU and settings | "When measurements set the level" |
| Same hardware | CPU name + GPU name | GPU device ID or name + VRAM ratio | The CPU is not used for judging |

## Tests

`tests/test_scoring.py` (and existing tests whose expectations were updated).

| What | Test |
|---|---|
| memory_fit fixed values (r = 0.50–0.98), no cliff, floor for rows that run, 0 for rows that do not | `test_memory_fit_fixed_values`, `test_memory_fit_shape_floor_and_not_runnable` |
| Each row of the r table (including GPU+RAM with RAM tighter and RAM short) | `test_usage_ratio_per_run_mode` |
| The Score dropping with more VRAM; the level does not | `test_score_can_drop_with_more_vram`, `tests/test_invariants.py` |
| Popularity rank example, small pools, sources not mixed, manual | `test_popularity_percentile_example`, `test_popularity_sources_are_not_mixed`, `test_popularity_is_per_model_and_in_json` |
| Bias check (Spearman) | `test_spearman_and_bias_check`, `tests/test_weekly.py` |
| Each order key (level, Marginal run mode, official, popularity, Score, ID, automatic recommendation), `--sort score` | `test_recommend_order_keys`, `test_fit_sort_type_and_score` |
| (a) Settings do not depend on measurements | `test_a_settings_do_not_depend_on_measurements` |
| (b) No level from measurements on other GPUs (RTX 4060 Ti 16 GB, RTX 4090) | `test_b_no_measured_level_on_other_gpus` |
| (c) Same GPU and settings, two agreeing sources set the level | `test_c_measured_level_on_the_same_gpu` |
| (d) Value differences within the shift | `test_d_values_shift_by_at_most_the_delta` |
| (e) The run date does not change the table order or Score | `test_e_run_date_does_not_change_table_order_or_score` |
| (f) The allowlist does not change the table order or Score | `test_f_allowlist_does_not_change_table_order_or_score` |
| (g) Measurements without a size are used | `test_g_measurements_without_size_are_used` |
| A fake 1.0 GB Flux is not Perfect; one person's four files do not set the level; the author + one submission do; two author rows are one source | `test_fake_small_submission_does_not_make_flux_perfect`, `test_one_source_with_many_files_does_not_set_the_level`, `test_author_one_and_community_one_agree`, `test_author_rows_are_one_source` |

`SCHEMA_VERSION` (the shape version of the bundled and released db.json) is unchanged. The meaning of the output changed, so the package is 0.11.0 ([CHANGELOG.md](../CHANGELOG.md)).

## Not decided yet

1. Measure the author's models again at the default settings (image 1024x1024, video 832x480 with 81 frames, music 30 s). Those get their `m` back.
2. A speed (seconds) estimate: GPU compute × size × resolution and frames × steps, calibrated with measurements (a separate spec).
3. Quality measurements: differences per precision with the same seed (LPIPS etc.) with bench. A precision penalty comes after that.
4. Measurements behind capping offload and CPU at Marginal.
5. The popularity bias check for music, 3D and speech (once there are more models).
