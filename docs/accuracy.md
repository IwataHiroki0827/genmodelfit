# Estimate accuracy

**English** · [日本語](accuracy.ja.md)

This page explains how genmodelfit's VRAM and memory estimates are built, which measurements they are fitted to, and how far they can be trusted.
It is for people who read the estimates, change the numbers, or submit measurements.

As of 2026-10-20 (how other users' measurements are handled: 2026-10-22).
The coefficients are in `src/genmodelfit/data/calibration.json`; rebuild them with `python -m genmodelfit.calibrate`
(`--dry-run` prints the tables without writing). The tables in "Results (leave-one-out)" come from that command's output.
**All units are GiB (2^30 bytes).** "GB" in the tables and the CLI also means GiB.

## Contents

- [Summary](#summary)
- [Formulas](#formulas)
  - [Main memory (RAM)](#main-memory-ram)
  - [Models of unknown size](#models-of-unknown-size)
- [How files are counted (agreement with #2)](#how-files-are-counted-agreement-with-2)
- [How measurements are used](#how-measurements-are-used)
- [Fitting](#fitting)
- [Results (leave-one-out)](#results-leave-one-out)
- [Known limits](#known-limits)
- [Other users' measurements (#34, 2026-10-21; handling aligned with llmfit on 2026-10-22)](#other-users-measurements-34-2026-10-21-handling-aligned-with-llmfit-on-2026-10-22)
  - [Bench on non-NVIDIA GPUs (0.9.0)](#bench-on-non-nvidia-gpus-090)
- [Cross-check against RTX 3070 8GB measurements (bench, 2026-10-19 UTC, not used for fitting)](#cross-check-against-rtx-3070-8gb-measurements-bench-2026-10-19-utc-not-used-for-fitting)

## Summary

- **Only 4 models could be evaluated (2 image, 2 3D). The maximum leave-one-out error is 6.4%, the mean 4.6%.**
  - Leave-one-out = fit on all but one measurement, then predict the one left out.
  - This is inside the ±20% target.
  - But it rests on 4 models and one kind of GPU, so it does not show that the target is met.
- **Video and audio cannot be evaluated: each has only one measurement that is not pinned.**
  (`by_modality.video.target_met` and `music.target_met` in `calibration.json` are null.)
  - Leaving that one out sends the coefficient back to its prior value (the placeholder from #1).
  - So Wan2.1 at -17.2% and ACE-Step at +3.5% only measure how good the prior values were.
- **The 3 pinned measurements (Z-Image bf16, Wan2.2 5B × 2) carry no information.**
  - The weights alone already exceed the measurement, so "prediction ≥ measurement" holds by construction and cannot fail.
  - They do not count as evidence of accuracy.
- As a baseline, predicting from "the sum of file sizes only" misses by 22.8% on average over the 6 unpinned measurements.
- Only the all-on-GPU VRAM (`vram_gb`) was fitted.
  - **The swapped-in-turn (`vram_sequential_gb`) and offload (`vram_offload_gb`) values and main memory (`ram_gb`) have no measurements and are not verified.**
  - They need measurements on an 8 GB GPU (the bench in #5).

## Formulas

ComfyUI runs "read the text (text encoder) → generate (main model) → finish (VAE)" in turn.
So the weights are split by stage and three quantities are computed (`estimate.py`).

| quantity | formula | meaning |
|---|---|---|
| `vram_gb` (all on GPU) | fixed + weights of all parts + working memory | what dynamic VRAM uses on a GPU with plenty free. The only one fitted to measurements |
| `vram_sequential_gb` (swapped in turn) | fixed + max(encoder, main + working, VAE + working, other) | parts are unloaded between stages. A bit slower but usable |
| `vram_offload_gb` (offload) | fixed + max(weights of each stage × 0.4 + working memory of that stage) | runs with part of the weights left in main memory (lowvram / dynamic VRAM). Slow |
| `ram_gb` (running on the GPU) | 3.0 + weights of the largest stage | for all on GPU and swapped in turn (see "Main memory (RAM)" below) |
| `ram_gb` (offload, or CPU) | 3.0 + weights of all parts | the offloaded weights stay in main memory |

**How weights are assigned to a role.** The first rule that matches decides; if none matches, the file is the main model.

1. ComfyUI folder (`directory`)
2. An explicit `role`
3. HF folder name
   - `unet` / `transformer` → main model
   - `text_encoder*` / `tokenizer*` → encoder
   - `vae*` → VAE
   - `image_encoder` → image encoder
   - `controlnet` → same stage as the main model
4. Known file names
   - Text encoders: `umt5` / `t5xxl` / `clip_l` / `clip_g` / `qwen_2.5_vl` / `qwen_3_4b` / `gemma` / `llama_3` and similar
   - VAE: `vae`

Notes:

- Short words such as `qwen` are not matched (`qwen_image_fp8_e4m3fn.safetensors` is a main model).
- Parts that sit on the GPU together with the main model (`controlnet`, `model_patches`, `loras`, …) go into the generation stage.
- The audio encoder (`audio_encoders`) runs before generation, so it goes into the reading stage.

**All-in-one `checkpoints` (one file holding everything):**

- Split by a breakdown counted from the tensor sizes in the safetensors header (SDXL / ACE-Step 1.5 / Hunyuan3D 2.0 / 2.1, read on 2026-10-20).
- These are split 75% main model, 20% encoder, 5% VAE (these shares are not verified):
  - All-in-one files with no known breakdown.
  - Files with no per-file size, only a template size.

**Working memory:**

- Image = coefficient × megapixels × batch size
- Video = coefficient × megapixels × (frames / 49)^0.5 × batch size
- Audio = coefficient × minutes
- 3D = constant
- The video exponent 0.5 is not fitted (there is only one measurement).
- Why it is gentler than linear:
  - ComfyUI's attention uses memory-saving implementations (SDPA and others), so working memory does not grow with the square of the token count.
  - The Wan VAE decodes the frames in small chunks, so the frame count does not act directly.
- The value of the exponent itself has no solid basis.

**Precision setting** (`Settings.precision`, CLI `--precision`):

- Only the main model's weights are converted (for example bf16 → fp8 halves them).
- Differences in case and spelling (`BF16`, `float16`, `gguf-Q4_K_M`) are normalized before comparing.
- The original precision is `ModelEntry.precision`; if missing, it is read from the main file name (`bf16` / `fp8` / `int8`, …).
- If it cannot be told, no conversion is done.

**Fit levels** (since 2026-10-20, following "Fit analysis" in llmfit's docs/how-it-works.md):

- The level comes from how much of the memory the mode uses (VRAM on the GPU, main memory on CPU):
  - ≤ 60%: Perfect
  - ≤ 85%: Good
  - ≤ 98%: Marginal
  - above that: Too tight
- Caps by mode:
  - All on GPU: no cap.
  - Swapped in turn (GPU~): up to Good.
  - Offload (GPU+RAM) and CPU: up to Marginal.
- The mode with the best level is taken. The main-memory check ("Main memory (RAM)" below) comes after that.
- If the size is unknown, no level is given (`unknown`, shown as `?`; "Models of unknown size" below).
- Before, it was "Perfect if all-on-GPU is within 90% of VRAM, Good if within VRAM".
- Effect on the bundled list (all 341 entries of the list at the time the levels changed, before the HF lower bound was raised; the bundled list of 0.9.1 (2026-10-21) has 329):
  - 8 GB / 32 GB: Perfect 44 → 23, Good 22 → 32, Marginal 87 → 97, Too tight 188 → 189.
  - 16 GB / 32 GB: Perfect 95 → 66, Good 67 → 76, Marginal 107 → 124, Too tight 72 → 75.
- At the VRAM that AI生成キット recommends (8 GB), Z-Image Turbo int8 and Wan 2.1 1.3B use 96–97%, so they are Marginal
  (they still run on the GPU; tests/test_estimate.py).
- If offload still does not fit on the GPU, the result is compared with the no-GPU judgment (Marginal for small types that run on CPU), and the better one is taken.
  - Before, adding a small GPU could turn a CPU-capable model into "does not run", so adding VRAM lowered the level (#17).

### Main memory (RAM)

- When running on the GPU (all on GPU, swapped in turn), main memory is **fixed 3.0 + the weights of the largest stage**.
  - This rests on an **assumption**: ComfyUI reads safetensors by memory mapping (mmap) and sends the parts to the GPU one at a time.
    So the OS can drop the pages of a part once it is used.
  - There is no measurement.
- For offload and CPU, the offloaded weights stay in main memory, so it is fixed + all parts (the safe side).
- When memory is short:
  - Up to **1.5 times** the installed memory: Marginal. The shortfall uses virtual memory (the page file), which is slow. The reason says so.
  - Above that: Too tight.
- How 1.5 was chosen:
  - On Windows, the default (system-managed) page file grows to about the installed memory or more.
  - But weights in the page file are extremely slow, so it was kept to half of that.
  - **1.5 is a judgment call and has not been measured.**
- On a PC with a fixed page file size (fixed size, no page file, Linux swap), the limit is "installed memory + that size" (but at most 1.5 times) (#21; see "Memory and virtual memory" in [hardware.md](hardware.md)).
- Before, it was "3 GB + all weights, Too tight if short".
  - That disagreed with the 16 GB minimum memory in AI生成キット's catalog.
  - On a 16 GB PC, Wan2.2 5B (16.9 GiB of weights) was Too tight regardless of the GPU.
  - Now none of the 8 models is Too tight because of memory on 16 GB (or a laptop that shows 15.3 GiB) (checked by `tests/test_estimate.py`).

### Models of unknown size

When both of these hold, the weights are not treated as 0 and the model is not said to fit:

- At least one file has no size (None, 0 or negative).
- There is also no template size and no parameter count × precision.

The output is:

- `fit = "unknown"`, `run_mode = "unknown"`, `basis = "estimated"`, `unknown_size = true` (`fit` was `too_tight` before 0.1.1, `run_mode` was `none` before 0.2.0)
- The reason starts with 「大きさが分からないため判定できません」 ("size unknown, cannot judge").
- The model is left out of `recommend` (it still shows in `fit --fit all`, `search` and `info`).
- VRAM and memory values are null in JSON (the Python attributes stay 0.0), and `limits` is null.
- Without a GPU it is also not treated as "small, so it runs on CPU".

## How files are counted (agreement with #2)

The list from #2 holds only one version per part (one precision, `.safetensors` where possible, either the single file or the diffusers layout).
Even so, to avoid overcounting when a Hugging Face repository listing is passed as is,
the estimator also counts by these rules (`estimate.select_files`):

1. **Files that are not counted:**
   - Files that are not weights (`.json`, `.txt`, `.md`, images, …).
   - Formats ComfyUI does not use (`onnx/`, `flax`, `openvino`, `safety_checker/`, `.onnx`, `.msgpack`, `.h5`, …).
2. **Count only one set per part.** A part is decided as follows:
   - ComfyUI folder + name (without precision, format and split number)
     - `t5xxl_fp16` and `t5xxl_fp8_e4m3fn` are the same part.
     - `clip_l` and `t5xxl` are different parts.
   - HF diffusers folder
     - `model` / `pytorch_model` / `diffusion_pytorch_model` inside are one part.
     - In a VAE folder, `vae` and `vae_1_0` and similar count as one.
   - Main model outside folders (same name, or one name starts with the other)
     - `sd_xl_base_1.0` and `sd_xl_base_1.0_0.9vae` are the same part.
     - Main models with different names, such as a high-noise and a low-noise one, are different parts.
3. **How one set is chosen:**
   - Format: `.safetensors` > `.gguf` > `.bin` / `.ckpt` / `.pt(h)`
   - Then precision: fp16 / bf16 > no precision written > fp8 / int8 / q8 > q6 > q5 > q4 / nf4 > fp32
   - If still equal, the larger one (the safe side).
   - Split files (`-00001-of-00003`) are all counted.
4. **When both a single file and a diffusers layout exist,** the single file is counted.
   - If the single file is known to be all-in-one (SDXL and others in the breakdown table), all diffusers folders are dropped.
   - If not known, only the diffusers folders whose role (main model, VAE) also exists as a file outside the folders are dropped.
     - This is so the encoder is not missed when the single file holds only the main model (for example FLUX's `flux1-dev.safetensors`).
     - If a file is all-in-one but not in the breakdown table, the result comes out larger (the safe side).
   - **The request said "count only the single file", but that would miss the encoder (about 9 GiB of T5 for FLUX) when the single file holds only the main model, so it was done this way.**
5. If files were dropped, the reason is shown in `reasons`. `disk_gb` is the sum of the counted files.

## How measurements are used

The RTX 5080 measurements were taken with ComfyUI's dynamic VRAM, which **uses more when more is free**.
13.2 GB on a 16 GB card (Z-Image int8) is not a lower bound of what the model needs. So:

1. A measurement is used as the all-on-GPU value (`vram_gb`).
   - The model's working memory is shifted by "measurement − formula value", and the same shift is added to the swapped-in-turn and offload values.
   - On a small GPU, even if all-on-GPU does not fit, the result **falls to swapped in turn or offload**.
     Example: on 8 GB, Z-Image int8 needs 13.2 GB all on GPU, but runs at 7.6 GB swapped in turn. Use is 96%, so the level is Marginal.
   - A measured value larger than the GPU does not by itself make a model Too tight.
2. If the measurement is 90% or more of the GPU's capacity (or offload is lowvram or similar), it counts as "pinned at the limit".
   - **It is used only as a lower bound** (all on GPU = max(formula, measurement)).
   - This applies to Z-Image bf16 (15.0 / 16) and the 2 Wan2.2 5B measurements (15.8 / 15.0).
3. `basis: "measured"` only when the settings (resolution, frames, length, **batch size**, precision) equal the measurement.
   - Precision spelling is normalized. If the measurement has no precision recorded, it is compared as the model's original precision.
   - Different settings give an estimate shifted by the measurement, with `basis: "estimated"` (the reason says 「…の実測で補正した推定」, "estimate corrected by the measurement of …").
   - If you change the precision, the measurement is not used.
4. When no settings are given (changed in 0.11.0; [scoring.md](scoring.md) "Settings used for judging"):
   - Every model, with or without measurements, is judged at the default for its type (image 1024px, video 832x480 and 81 frames, music 30 seconds, …).
   - A measurement's settings are used only to work out the shift of the working memory. The maintainer's image, video and music measurements (AI生成キット's "fast" quality:
     768–832px, video at 512–640px, music of 20 seconds) do not match the defaults, so at the defaults they are `calibrated` (no `m` in the table).
   - Up to 0.10 a model with a measurement was judged at the measured settings (so only the models the maintainer measured were compared at smaller settings).
   - To judge at the measured settings, give `--width` and the like (apps: `judge_model(..., settings=...)`).
   - A template without a size input (depth estimation, background removal and the like, which work from an input image) has no size in its judging settings (#90).
     - Which templates have no size input is decided only by a hand table the maintainer checked (`estimate.NO_SIZE_INPUT`; the evidence is a bench record or the template's nodes), never by other users' submissions (so that one submission cannot change the judgment of a model with a size input on every PC).
     - A new template is added when its first bench record is reviewed (the bench writes a width and height only when the template has a size input; a size-less submission for a model not in the table gets the `settings_missing` warning in its PR, which tells the maintainer).
     - A size-less measurement then has the same settings as a judgment without settings. With `--width` and the like, things work as before.
     - The working memory is still estimated at the default size for the type (image 1024px and so on).
5. Without a GPU (`--vram 0` and so on), GPU measurements are not used: `basis: "estimated"`, `measured_on: null`.
6. These measurements are used neither for estimates nor for refitting (written by the bench in #5):
   - `outcome` is not `ok` (OOM, error).
   - `standard: false` (custom workflows or extra overrides).
7. Measured values are taken as GiB (`unit` in `measurements.json`).
   - AI生成キット's records do not state the unit.
   - It is an **assumption** from the fact that ComfyUI and nvidia-smi count in MiB.

## Fitting

- Five coefficients are fitted: the fixed part `overhead_gb`, and the working memory `act_image_gb_per_mp` / `act_video_gb_per_mp` / `act_audio_gb_per_min` / `act_3d_gb`.
- These stay fixed: the weight factor (1.0), the offload share (0.4), the video exponent (0.5), the fixed main-memory part (3.0) and the memory factor (1.5).
- The prediction is linear in every coefficient. What is minimized is the sum of:
  - **the sum of squared relative errors** (a pinned measurement counts only when the prediction falls below it: a one-sided square)
  - a weak penalty (weight 1e-4) on the relative distance from the prior values (the placeholders from #1, set before looking at the measurements)
- The minimum is found by coordinate descent inside physically possible ranges. The objective is convex, so the result does not depend on the starting point.
- With 5 coefficients for 9 points, it would overfit without the penalty. The penalty is weak because measurements should win where they exist.
- Accuracy is measured by leave-one-out (a prediction from a fit on all the data shows goodness of fit, not accuracy).
- The target check (`target_met`) is made only for types with 2 or more unpinned measurements.

**Relation to human-set guidelines (stated honestly):**

- The minimum / recommended VRAM in AI生成キット's catalog were not used to fit the 5 coefficients.
- But **the offload share 0.4 is a hyperparameter chosen by looking at those guidelines.**
  - It cannot be set by measurement, so it was chosen from the range consistent with two statements: "SDXL runs on 4GB" and "bf16 Z-Image does not run on 4GB".
  - For the coefficients refitted in GiB (before the list from #2 was imported), that range was 0.21–0.51. 0.4 is in the upper half of it.
  - That range was the author's judgment at the time and has not been re-measured with the coefficients after the import.
- "bf16 Z-Image does not run on 4GB" is not stated in the catalog; it is **this PR's author's judgment**.
  - In the list at the time, of 19.4 GiB of weights, 40% of the 11.5 GiB main model plus about 6 GiB of fixed and working memory is needed.
  - In the bundled list of 0.9.1 the weights are 19.27 GiB.
- `tests/test_estimate.py` compares the results with these guidelines. The outcome is not fed back into the fit.

Fitted result (2026-10-20, n = 9, of which 3 pinned):

| coefficient | value | prior | range |
|---|---:|---:|---|
| overhead_gb | 1.472 | 0.6 | 0.3–1.5 |
| act_image_gb_per_mp | 0.2284 | 0.5 | 0–4 |
| act_video_gb_per_mp | 9.8704 | 2.4 | 0–30 |
| act_audio_gb_per_min | 0.0455 | 1.0 | 0–5 |
| act_3d_gb | 0.784 | 2.0 | 0–4 |

- The fixed part rose close to the top of its range (1.5).
- The fixed part absorbs the roughly 7% that the weights shrank by when everything was aligned to GiB, instead of the per-type working memory.
- With 9 points, working memory and the fixed part are hard to tell apart. **Do not read a physical meaning into each coefficient.**

## Results (leave-one-out)

The output of `python -m genmodelfit.calibrate`, which prints Japanese, translated here; the original output is in [accuracy.ja.md](accuracy.ja.md#結果leave-one-out).

| model | settings | measured GB | predicted GB (fit on all) | LOO predicted GB | LOO error | error, weights only |
|---|---|---:|---:|---:|---:|---:|
| image_z_image_turbo_int8 | 768x768 | 13.2 | 12.94 | 12.85 | -2.7% | -14.1% |
| image_z_image_turbo | 768x768 | 15.0 | 20.88 | 20.88 | pinned (the weights alone exceed the measurement, so no information) | +28.5% |
| image_sdxl_simple | 832x832 | 8.0 | 8.09 | 8.38 | +4.7% | -19.2% |
| text_to_video_wan | 512x512, 49 frames | 13.3 | 13.21 | 11.02 | -17.2% | -31.2% |
| video_wan2_2_5B_ti2v | 640x640, 73 frames | 15.8 | 23.31 | 23.31 | pinned (the weights alone exceed the measurement, so no information) | +7.0% |
| video_wan2_2_5B_ti2v | 640x640, 73 frames, from text | 15.0 | 23.31 | 23.31 | pinned (the weights alone exceed the measurement, so no information) | +12.7% |
| audio_ace_step_1_5_checkpoint | 20 s | 10.8 | 10.83 | 11.18 | +3.5% | -13.5% |
| 3d_hunyuan3d_image_to_model | - | 6.7 | 6.85 | 7.13 | +6.4% | -31.5% |
| 3d_hunyuan3d-v2.1 | - | 9.4 | 9.12 | 8.97 | -4.6% | -27.0% |

4 measurements could be evaluated (3d 2, image 2); max LOO error 6.4% (mean 4.6%). Inside the ±20% target.
music and video have 1 or fewer unpinned measurements, so they cannot be evaluated (target_met is null).
Of the 3 pinned measurements, 3 are exceeded by the weights alone and carry no information. For reference: over all 6 unpinned measurements, mean absolute LOO error 6.5%, max 17.2% (weights only: mean 22.8%)

Cross-check (measured a different way, so not used for fitting; values predicted with the coefficients fitted on all data):

| model | GPU | source | settings | measured GiB | estimated GiB (all on GPU) | error |
|---|---|---|---|---:|---:|---:|
| utility_birefnet_remove_background | NVIDIA GeForce RTX 3070 | maintainer | - | 1.25 | 2.13 | +70% |
| 3d_moge_perspective_to_mesh | NVIDIA GeForce RTX 3070 | maintainer | - | 0.72 | 2.87 | +299% |

How to read it:

- "Predicting an unknown model" is measured by only 2 image and 2 3D measurements.
  The working-memory coefficient of each type is set by just one other measurement of the same type, so the count is very small.
- Wan2.1 at -17.2% and ACE-Step at +3.5% are the result of leaving out the only measurement of their type, so they are errors of the prior values (the placeholders from #1).
  Nothing can be said about video and audio accuracy.
- The audio coefficient is small: 0.05 GiB/min (0.0455).
  - The gap between the ACE-Step measurement and the formula is almost fully explained by the fixed part, so the effect of length is not visible.
  - **The audio limit "up to how many seconds" (`limits.max_seconds`) is currently a value that assumes VRAM hardly grows with length.**
- The video coefficient (9.9 GiB per megapixel) is set by the single Wan2.1 measurement.
  That one measurement very likely includes memory that dynamic VRAM placed in the free space.

## Known limits

- **Only one kind of GPU**: all 9 measurements are RTX 5080 16GB with dynamic VRAM.
  - The swapped-in-turn and offload values on small GPUs are not verified.
  - Two bench measurements on an RTX 3070 8GB exist, but they are small models measured a different way (torch_reserved), so they are used only for the cross-check
    (see "Cross-check against RTX 3070 8GB measurements" below).
  - Once bench measurements of large models on 8GB (the bench in #5) arrive, the swapped-in-turn and offload values will be fitted.
- **Dynamic VRAM**: it uses more when more is free, so a measurement is "what that GPU used", not "what is needed".
  The all-on-GPU value may also come out larger than the real need on a GPU with little free memory.
- **Pinned measurements**: the all-on-GPU values of Z-Image bf16 and Wan2.2 5B are decided by the formula alone; the measurements do not cap them from above.
- **Wan aspect ratio**: video measurements record only the long side (512px / 640px), so a square was assumed.
  If the real shape was 16:9, the pixel count is about 56% and the video coefficient would be re-estimated about 1.8 times smaller.
- **Date unknown**: the measurement date and ComfyUI version were not recorded (the date in measurements.json is empty).
- **Unit**: measurements were assumed to be GiB (7 under "How measurements are used" above). If they were GB (decimal), the measurements are 7% smaller and the coefficients change.
- **Main memory**: neither the formula nor the factor 1.5 has been checked by measurement (see "Main memory (RAM)" above).
- **Non-NVIDIA GPUs** (0.7.0, #22): there are 0 measurements on AMD, Intel and Apple.
  - NVIDIA's coefficients are used as they are, and the level is capped at Good (lifted when other users' measurements of that GPU arrive; 2026-10-22).
  - The shares counted as GPU memory are guidelines taken from the behavior described in primary sources, and are not measured (see "Shared memory" in [hardware.md](hardware.md)).
    - 3/4 of unified memory on Apple silicon
    - 1/2 of main memory for integrated GPUs and APUs (the Linux TTM default)
- **CPU-capable families** (#39): the CPU main memory for SD-family and Stable Audio is a formula taken from ComfyUI's code, not measured.
  - The formula expands the main model and VAE to fp32 and uses working memory × 2.
  - That these run on CPU is itself an estimate; there are 0 CPU measurements (the slowness is also unknown).
- **Known misses** (cross-check with the human-set guidelines, recorded by `tests/test_estimate.py` as a strict xfail):
  - Wan2.2 5B comes out Too tight on 8GB and Marginal on 12GB (the guideline is minimum 8GB, recommended 12GB).
    - It comes out strict because the video working memory is extended from the single Wan2.1 measurement.
    - Wan2.2 compresses more in its VAE and has fewer tokens per pixel, but the same coefficient is used.
    - It will not be fixed until refitted with an 8GB measurement (moving the coefficients to match the guideline would make the numbers in these tables meaningless).
  - The other 7 models are not Too tight at the guideline's minimum VRAM and are Good or better at the recommended VRAM.
- **Models with an unknown all-in-one breakdown** (such as the HF models that #2 adds) are split by general shares, so the swapped-in-turn and offload values can be far off.
- **Role guessing**: a file with neither a folder nor a location gets its role from the list of known names.
  A text encoder not in the list is counted as the main model (the all-on-GPU value does not change, but the swapped-in-turn and offload values come out larger).

## Other users' measurements (#34, 2026-10-21; handling aligned with llmfit on 2026-10-22)

Users add the submission JSON made by `genmodelfit bench --submission` through a Pull Request to `data/contrib/<GPU>/<date>-<sha8>.json`
(steps: [CONTRIBUTING.md](../CONTRIBUTING.md); format: "contrib" in [schema.md](schema.md)).
**As of 2026-10-22, 0 submissions have been accepted**, and the bundled `src/genmodelfit/data/measurements_community.json` is empty.
The rules below describe how submissions will be handled once they arrive.

**Files measured (#90)**: other users' submissions (contrib_schema 2) and this PC's bench measurements carry `files_fp`, the fingerprint of the list's files when measured; when a weekly list update sees upstream files change (the fingerprint differs), they are no longer used. The maintainer's measurements (`measurements.json`) have no fingerprint, so they keep being used when files change (the maintainer re-measures them; the difference is that other users' rows silently stop being used while the maintainer's keep being used). `genmodelfit contrib aggregate --check` lists the bundled rows that are no longer used.

**Maintainer's decision of 2026-10-22:** follow the same approach as llmfit (main, read on 2026-10-22).

- How llmfit does it:
  - It only checks submissions (the `llmfit bench --share` PR) mechanically in CI.
    - Checked: schema, path and name, size, time, value limits, min ≤ avg ≤ max.
    - There is no signature and no human-review mechanism; merging is the maintainer's review.
  - It groups only measurements with the same CPU name and GPU name, takes the median, and keeps a count (sample_count).
  - It uses even one measurement, and overrides the estimate in both directions.
  - Priority: your own measurement > other users' measurements on the same hardware > estimate.
  - For correcting other models, it clamps the median of measurement ÷ estimate to [0.05, 3.0].
- The rules of 2026-10-21 were dropped.
  - Content: use only to raise the need, at most +15%, only with 2 or more submissions within 15% spread, only submissions a person checked per sha8.
  - Reason: they were hard for the maintainer to follow, and even with measurements they could not lift the "up to Good" cap for non-NVIDIA GPUs.

**How they are grouped** (`genmodelfit contrib aggregate`):

- Submissions are grouped by what is the same in:
  - model
  - GPU (the PCI model ID if there is one, else the name normalized with `gpu_slug`)
  - GPU VRAM (rounded to 1 GB)
  - how it was measured (`vram_metric`)
  - settings (resolution, frames, length, steps, precision, batch size)
- Non-standard submissions (`standard: false`) are not used.
- The maximum VRAM of runs that finished (ok) is used.
  - With 3 or more, values with |x − median| / (1.4826 × MAD) > 3.5 are dropped as outliers (the floor of MAD is 2% of the median).
  - The median of the rest is the value.
- The count (`n_used`) and the number of OOM runs in the same setup (`n_oom`) are kept.
- A group mixing ok and OOM uses the median of the ok runs, and the OOM count is shown in the reason.
- A group with only OOM becomes one row with `outcome: oom` (not used for estimates).

**When they are used:** only measurements on **the same GPU as this PC** are used for this PC's need and level.

- Same as llmfit (0.9.0). The judgment for a different GPU or a PC without a GPU is the same with or without other users' measurements.
- `estimate.same_gpu` decides whether it is the same GPU.
  - If both have a PCI model ID (`1002:744c`), the ID is used.
  - Otherwise the name normalized by `estimate.gpu_key` is used (dropping (R), (TM), ™, ®, case, whitespace and the words NVIDIA / GeForce).
- **In addition, the VRAM must be close** (0.9.1; `estimate.same_vram`):
  - The ratio of the measured GPU's VRAM to this PC's total (smaller ÷ larger) is 0.95 or more (0.11.0; 0.97 in 0.9.1–0.10).
  - Because GPUs with the same name and the same model ID can have different VRAM (RTX 4060 Ti 8 / 16 GB, RTX 3060 8 / 12 GB, RX 7900 XT / XTX / GRE with PCI ID `744c`).
  - 0.95 allows for how the OS and driver count the total of the same card (`MEASURED_VRAM_TOLERANCE`). Linux CUDA reports the total minus a reservation: 5.80 GiB for a 6 GB card (5934 MiB, an RTX 2060 report on the NVIDIA developer forum), 7.79 for 8 GB, 9.78 for 10 GB, 10.76 for 11 GB, 11.77 for 12 GB, 23.69 for 24 GB; Windows reports 6.0, 7.99, 15.92 and so on. For 6 GB, 5.80 ÷ 6.0 = 0.967 missed 0.97, so Linux and Windows owners of the same card were different GPUs; hence 0.95.
  - The ratio of versions with different VRAM is 0.83 or less (RTX 3080 10 / 12 GB, for example), so there is room between them.
  - Integrated GPUs, APUs and Apple (shared / unified memory, whose total is set by the PC's main memory) are not compared.
- The check of other users' measurements (`fit_change` in `contrib`) is also done for a PC with that VRAM, for GPUs whose nominal VRAM the name table knows.
  - Covered: Intel Arc, and NVIDIA's `estimate.NVIDIA_VRAM_VARIANTS` (RTX 2060, 3050, 3060, 3080, 4060 Ti, 5060 Ti).
  - Confirmed on NVIDIA's official product pages; the source is in the constant's comment.
  - Laptop GPUs and the GTX 16 series are not in the table: they have no VRAM variants, or the names differ.
- Generic names such as "AMD Radeon(TM) Graphics" and "Intel(R) Arc(TM) Graphics" are not matched by name, because GPUs of different generations share them.
- The maintainer's bundled measurements shift the values on every PC (as before). Only the same GPU uses them to set the level (0.11.0; "Level" below).
- An ok row shifts the values **from one measurement**. `consistent` (2 or more sources, within 15% spread, no OOM) is the condition for entering a refit of the coefficients, and also one of the conditions for setting the level (0.11.0).

**Direction:** a measurement is used to **lower or raise** the need. It counts the same as the maintainer's measurements. Only the physical range limits it (`estimate._prepare`).

- Lower bound: the all-on-GPU value is not lowered below **weights + fixed part** (working memory 0).
  - A smaller measurement means some weights were held in main memory (dynamic VRAM, partial) or the measurement was wrong.
  - The estimate is set to the lower bound. `basis` is estimated (`basis_detail` is measured_community). The reason says 「下限の値にした」 ("set to the lower bound").
  - The condition that validate used to flag for review (`offload: none` and smaller than weights + fixed part) is now clamped in the estimate.
- Upper bound: the measured GPU's total × 1.02 + 0.25 GiB (the same as validate's "impossible value").
  - A larger value is clamped to this bound (total × 1.02 + 0.25 GiB) (`estimate._prepare`).
  - The reason reads 「GPU の総量を超えるので、上限(総量 × 1.02 + 0.25 GB)までにした」 ("it exceeds the GPU total, so it was limited to the bound (total × 1.02 + 0.25 GB)").
- The peak main memory (`peak_ram_gb`) is limited to the submission's main memory × 1.05 + 1 GiB (clamped when read).
- A measurement pinned at the GPU limit (90% or more of the total, lowvram) is used only as "at least", like the maintainer's measurements.
- When extending to an estimate with different settings (calibrated), measurement ÷ formula is clamped to **[0.6, 1.4]** (the counterpart of llmfit's [0.05, 3.0]).
  - The range is the same as the width at which validate reports "off the formula" (±40%).
  - The maintainer's measurement ÷ formula is 0.64–1.03 (pinned Z-Image bf16 and Wan2.2 5B: 0.64–0.72; the others: 0.98–1.03).
    VRAM is closer to the formula than speed (the quantity llmfit corrects).
  - On the low side, the lower bound of weights + fixed part usually acts first (the maintainer's lower bound ÷ formula is 0.73–1.0).
- The default settings used for judging are not changed to the settings of other users' measurements (a submission with small settings does not shrink the default settings themselves).
- The maintainer's and this PC's `torch_reserved` measurements (no CUDA context included; the 2 RTX 3070 ones) are used only as "at least", as before.
- Other users' `torch_reserved` measurements may be lowered down to the lower bound (weights + fixed part), because every bench submission is torch_reserved and could otherwise never be used to lower the need.

**How a measurement is picked** (`estimate.pick_measurement`): only measurements on the same GPU are candidates for other users' measurements. The order is:

1. Same settings
2. **A measurement on the same GPU as this PC**
3. The maintainer's or this PC's measurement
4. Other users' measurement
5. Not pinned
6. Larger GPU
7. Larger value

GPU names are compared after normalizing the presence of `NVIDIA` / `GeForce`, case and symbols (`estimate.gpu_key` = `contrib.gpu_slug`).

**Level** (#26, "decide the level by measurement", `estimate.measured_level`; the conditions changed in 0.11.0, full rules in [scoring.md](scoring.md) "When measurements set the level"):

- Only successful measurements on **the same GPU** as this PC (`estimate.same_gpu`) with the same settings are a basis for the level. The same rule for the maintainer's and other users' measurements (not extended to similar GPUs; up to 0.10 the maintainer's counted on any same-vendor GPU with at least the measured VRAM).
- The level is set only when **two or more independent sources** agree. The maintainer's rows are one source together, however many (the maintainer alone never sets the level); other users' rows count their `n_sources` (one per pull request that added submissions); they are added up. With one source, the values are only shifted.
- Agree: no OOM record (no `outcome: oom`, `n_oom` or `runs_evidence: false` for the same setup), each aggregated row agrees inside (`consistent`), and the spread across rows is within 15% of the median.
- **The amount used for the level = max(largest peak + context, weights + fixed part)**, the same value as the table's VRAM (`measured_level_vram_gb`). It uses the same physical floor as the values so that a fake small value cannot lift the level.
  - If this amount does not fit this PC's VRAM, the level is not set (for example, Wan2.2 5B's weights + fixed part is about 18.4 GB, which does not fit the RTX 5080's 15.9 GB).
  - Perfect needs 60% or less. A pinned measurement is not a basis for Perfect (up to Good).
- A `torch_reserved` value gets 0.5 GiB added for the context before the band is decided, and it can also be a basis for Perfect (0.9.0).
- **The "up to Good" cap for non-NVIDIA GPUs is lifted when the level is decided by measurements.**
  - The level rises as far as the measurements decide (it can become Perfect).
  - When they do not decide it (one source, for example), it stays capped at Good as before.

**Basis levels** (`Estimate.basis_detail`) and the number of sources (`Estimate.sample_count`; the number of submissions until 0.11.0):

- `basis` stays two-valued (measured / estimated) for backward compatibility.
- `sample_count` was only added to the JSON.
- How they show:
  - `info`: `measured_community n=3`
  - the mark on the table's VRAM: `c`
  - the reason: "measurements by other users (3 source(s); median 7.02 GB)" (「ほかの利用者の実測(出どころ 3。中央値 7.02 GB)」)

| level | basis | meaning |
|---|---|---|
| `measured` | measured | the maintainer's bundled measurement, or a measurement the bench saved on this PC, with the same settings |
| `measured_community` | measured (estimated when clamped by the lower or upper bound) | other users' measurement (from one), with the same settings |
| `calibrated` | estimated | an estimate shifted by a measurement with different settings (the maintainer's or other users') |
| `estimated` | estimated | a formula-only estimate (no measurement, precision changed, or no GPU) |

**Validation** (`contrib validate`, contrib.yml in CI): **only mechanical checks fail.**

- Failing checks:
  - Format, value ranges, personal information, duplicates, file name, size.
  - Value ranges: GPU VRAM at most 256 GiB, peak memory at most main memory × 1.05 + 1 GiB, no NaN, infinity or negative numbers (0.9.0).
  - A GPU name and `system.gpu.vendor` that name different vendors.
  - A `pci_id` with the wrong format or vendor.
  - A name that does not match the GPU in `hardware.PCI_DEVICE_NAMES` (an ID not in the table is a notice).
- Shown only as a "notice" above the job summary:
  - More than 40% off the formula (`deviation`).
  - All on GPU (`offload: none`) but smaller than weights + fixed part (`below_weights`).
  - The level changes on a PC with the same GPU as the submission (name, model ID, vendor; VRAM from the submission's `vram_gb`) × 16 / 32 / 64 GB of memory (`fit_change`; it also reports a level going up on the same GPU, or the "up to Good" cap being lifted).
  - A model not in the list.
  - A GPU not in the table.
- The mechanism where a person checks each sha8 (`--allow-review`, `data/contrib/REVIEWED`) was removed.
- The safety mechanisms of contrib.yml remain:
  - It runs on pull_request.
  - Validation uses the code of the PR's target branch.
  - A PR that changes anything outside data/contrib fails.
  - Characters from a submission are not written to the log.

**Remaining risk:**

- A tampered submission, once merged, moves the estimates and levels as it is.
  - Examples found in the review of 2026-10-21: a submission with small values lowers the need by up to 76%; a +39% submission lowers a level.
  - Inside the lower and upper bounds, these are not prevented.
- What prevents them is **the merger's review.** For that, notices and the table 「ほかの利用者の実測で変わる段階」 ("levels that change with other users' measurements") of `contrib aggregate --report` are shown.
- One submission decides the value of its setup (outliers can be dropped only with 3 or more).

**Feeding calibration** (`python -m genmodelfit.calibrate`):

- Only consistent (`consistent`) measurements of other users are used. A single measurement is not enough, because the coefficients affect every model.
- If the way of measuring is the same as the coefficient fit (`calibrate.CALIBRATION_METRIC`; currently the unrecorded way of the 9 RTX 5080 measurements), they enter the fit and leave-one-out.
- Bench submissions are `torch_reserved`, so for now they enter only the **cross-check** (the table above: the error when predicted with the coefficients fitted on all data).
- Once a few `torch_reserved` measurements per type exist, the fixed part will be split by way of measuring and refitted (until then they are not mixed).

**Weekly check** (`genmodelfit contrib aggregate --report`, called by the weekly automatic PR of #19):

- It re-aggregates data/contrib as of that moment, redoes the calibration, and tabulates the LOO error per type and the cross-check error.
- It warns when a threshold is exceeded (default 25%, `--threshold`; `--fail-on-breach` gives exit code 3).
- What enters the judgment:
  - The LOO error of types that can be evaluated (2 or more unpinned measurements).
  - The cross-check error of other users' measurements.
- The maintainer's cross-check rows (the 2 RTX 3070 rows below) are known misses, so they are shown in the table but not counted in the judgment.

**Inflation:** if the same person submits the same setup many times, the count is inflated.

- It cannot be detected, because submissions carry no information that identifies a person.
- The same content (hash) is rejected as a duplicate, but a re-measured different value cannot be rejected.
- The person reviewing the PR checks whether the same setup appears several times in the same PR.

### Bench on non-NVIDIA GPUs (0.9.0)

Non-NVIDIA GPUs (AMD, Intel, Apple) have 0 measurements, and their level is capped at Good (see [hardware.md](hardware.md)).
To make measurements easier to collect, the bench's "is the GPU idle?" check also uses each vendor's tools (`gpustat.py`).

- AMD on Linux: amd-smi → rocm-smi
- Intel on Linux: xpu-smi
- AMD and Intel on Windows: performance counters
- Apple: ioreg

If the usage can be read, it refuses at 30% usage, as on NVIDIA.

When no tool exists or the usage cannot be read (for example Linux without rocm-smi), it measures without asking for `--force`.

- It checks only these two things:
  - The ComfyUI queue is empty.
  - GPU memory used by anything other than ComfyUI is at most half of the total (not checked for Apple's unified memory).
- It leaves 「使用率は確かめていない」 ("usage was not checked") in warnings and note.
- If anything other than ComfyUI uses more than half of GPU memory, it stops unless `--force` is given.

**`standard` stays true.** The reasons:

- `standard` means "the official template as is, no other work mixed in ComfyUI, and every node ran", and looks only at what happens in ComfyUI's queue.
  The queue is checked before the start, just before sending, and during the measurement.
- The value used for estimates (`peak_vram_gb`) is what the ComfyUI process's torch allocated, and it hardly changes even if other apps use the GPU
  (what changes is the time, and the note about the GPU-wide increase).
- A measurement on NVIDIA taken with `--force` and no check has not changed `standard` either (kept the same).

Other points:

- A run where contention with other apps caused a memory shortage and offloading or failure shows in `offload` and `outcome`, so it can be told apart there.
- When ComfyUI runs on another PC, `--force` is needed, as on NVIDIA (this PC's tools can check nothing).
- The measurement keeps `gpu_backend` (cuda / rocm / xpu / mps).
- The submission holds `system.gpu.vendor`, `backend`, and, for AMD and Intel, the PCI model ID (`pci_id`).
- If xpu-smi shows 2 or more Intel GPUs, it cannot tell which one is ComfyUI's, so the usage is treated as unreadable.

With Apple's MPS, ComfyUI's GPU name is `mps`.

- If ComfyUI runs on the same PC, the bench uses this PC's chip name read with system_profiler as `gpu_name` of the measurement and the submission.
  - Example: `Apple M2 Pro`. Only names matching the form of `hardware.APPLE_CHIP_NAME`; serial numbers and the like are not included.
- The judgment compares with this PC's GPU name (the same system_profiler chip name), so a measurement of the same chip is a basis for the level.
- With ComfyUI on another PC, or if the chip name cannot be read, it stays `mps` (with a warning).
- An `mps` measurement is used only as vendor apple, and never counts as a measurement of the same GPU.
- `contrib validate` limits GPU names of vendor apple to a chip name or `mps`.

## Cross-check against RTX 3070 8GB measurements (bench, 2026-10-19 UTC, not used for fitting)

`genmodelfit bench` measured 2 models on this PC (RTX 3070 8GB, AI生成キット's engine ComfyUI 0.40.0).

- The way of measuring differs from the 9 RTX 5080 measurements.
  - `vram_metric: torch_reserved` = the maximum of what the ComfyUI process's torch reserved; it does not include the CUDA context and the like.
  - What the 9 RTX 5080 measurements recorded was not kept.
- So they are not used for fitting, only for the cross-check (`calibrate.CALIBRATION_METRIC`).
- `python -m genmodelfit.calibrate` makes this cross-check under the table in "Results (leave-one-out)" above (`check` in calibration.json; since #34).

| model | GPU | measured GiB (torch_reserved) | estimated GiB (all on GPU) | error | level in the estimate (8 GiB) |
|---|---|---:|---:|---:|---|
| utility_birefnet_remove_background | NVIDIA GeForce RTX 3070 | 1.25 | 2.13 | +70% | Perfect |
| 3d_moge_perspective_to_mesh | NVIDIA GeForce RTX 3070 | 0.72 | 2.87 | +299% | Perfect |

- **For small models the estimate comes out large** (+70%, +299%).
  - The fixed part (overhead, about 1.47 GiB) was fitted on the 9 RTX 5080 measurements (models of 5–20 GiB), and it makes up most of the estimate for a model under 1 GiB.
  - But the difference in the way of measuring (no CUDA context or other memory outside the process) is also part of the gap, and how much comes from each has not been separated.
- The error is in the safe direction (the need is overestimated), and the level at 8 GiB stays Perfect for both.
- Measuring a large model (main model 5 GiB or more) on the RTX 3070 was put off because this PC's free memory (commit) was not enough (genmodelfit#5).
- Once more measurements taken the same way exist, it is better to split the fixed part by way of measuring and refit.
