# Contributing measurements

**English** · [日本語](CONTRIBUTING.ja.md)

genmodelfit prefers a value measured on a real PC (a measurement) over its estimate. The more GPUs are measured, the more accurate the estimates get.
This page is for anyone who wants to send a measurement (with `genmodelfit measure` and `genmodelfit share`, or as a pull request). It also lists the rules for other changes.
For terms, see [README.md](README.md).

## Contents

- [The easy way (no git needed)](#the-easy-way-no-git-needed)
- [Advanced: bench and pull requests](#advanced-bench-and-pull-requests)
  - [1. Measure](#1-measure)
  - [2. Create the submission JSON](#2-create-the-submission-json)
  - [3. Check the contents](#3-check-the-contents)
  - [4. Send a pull request](#4-send-a-pull-request)
- [Other changes](#other-changes)
- [License](#license)

In short: the easiest way is **two commands and one button**: `genmodelfit measure` measures, `genmodelfit share` opens a GitHub form with the result filled in, and you press "Create" ("[The easy way](#the-easy-way-no-git-needed)" below; no git and no fork needed).
If you know git, you can also run `genmodelfit bench … --submission data/contrib/` in a clone of your fork, check the file with `genmodelfit contrib show`, and open a pull request that adds only that file (from "[2. Create the submission JSON](#2-create-the-submission-json)").
**genmodelfit itself never sends anything**; sharing happens only when you press "Create" in the form, or through the pull request you open.
Today's NVIDIA bench submissions (ComfyUI's dynamic VRAM, `vram_metric: torch_reserved`) hold only the working memory, so until per-stage peaks can be measured they are used for validation only, not for estimates or levels (#100; [accuracy.md](docs/accuracy.md#how-measurements-are-used)).

## The easy way (no git needed)

`measure` and `share` are available from 0.11.0. With [uv](https://docs.astral.sh/uv/) you can run them without installing (`uvx genmodelfit measure`; otherwise `pip install -U genmodelfit`).

```powershell
genmodelfit measure
genmodelfit share
```

**What you need:** a GitHub account (free), ComfyUI running on this PC with an NVIDIA, AMD, Intel or Apple GPU,
and about 1–5 minutes per model (plus the download when the model is not in your ComfyUI yet). No git, no fork and
no programming.

1. **`genmodelfit measure`**: choose a model from the list by its number and press Enter to measure it.
2. **`genmodelfit share`**: choose the measurement; the GitHub form opens in your browser, already filled in.
3. **Press "Create".** A bot checks it within a few minutes and replies on the page.

One run measures one generation. If you have a few more minutes, run `genmodelfit measure` again a few times: each time
it offers the next model or setting that has no measurement on your GPU yet, and every one of them helps.

Step by step, in detail:

1. Start ComfyUI (the one you normally use is fine) and close other generation jobs. For the cautions before measuring, see "[1. Measure](#1-measure)".
2. `genmodelfit measure` lists models worth measuring: they fit this PC's GPU (good or better on the GPU) and have no measurement on this GPU yet. Models already in your ComfyUI come first, then the shorter time, then the smaller download. Choose one by number (`q` quits).
   - Once a model is measured on this GPU at the template's settings, the list also has runs at other settings, which help check the estimate's formula: image resolution (512 / 768 / 1024 / 1536 px, text-to-image), video frames (17 / 33 / 49 / 81 at the template's size) and music / sound length (10 / 30 / 60 / 120 s). They are shown with `--resolution`, `--frames` or `--length`, only when they fit this GPU at that setting and this GPU has no measurement at that setting, at most 2 per model. Runs over about 300 s are not listed. If the template cannot take a setting, measure stops before generating.
   - To pick the model yourself: `genmodelfit measure "Z-Image-Turbo"` (a name, a workflow id or a part of either, chosen like `download`). Add `--resolution 1536x1536`, `--frames 81` or `--length 120` to change the settings, like `bench`.
   - It finds a ComfyUI running on this PC (127.0.0.1 ports 8188 and 8000); for another one, add `--url http://<host>:<port>`. When none runs, it says how to start ComfyUI.
3. Before measuring, one screen shows the model and precision, the settings (as in the official template), which ComfyUI and how it was found, whether the model files are installed, a rough time and that nothing is sent. Answer `Y` (Enter) to measure.
   - The time is a rough guide: the median of measurements of the same model on other GPUs, else a rough value per type, scaled by the ratio of pixels, frames or length when the settings are changed. Downloading is not included.
   - A model that is not installed is downloaded first, with the same questions and consent rules as `genmodelfit download`.
   - When another program uses the GPU, it does not measure (the same as `bench`).
4. It shows the result and where it saved it (`<cache>/submissions/<GPU>/<date>-<sha8>.json`), and one line pointing to `genmodelfit share`. It never asks whether to send.
5. `genmodelfit share` lists the measurements you have not sent; choose one by number. It shows everything that will be sent and the privacy checklist (GPU name, VRAM, OS family, model, settings and values; no PC or user names). **Read it yourself, including the note.**
6. The measurement form on GitHub opens in your browser with the content filled in. Sign in to GitHub (you need an account) and press **"Create"**.
7. A bot checks it and replies on the page within a few minutes. If it is fine, it goes into the project once a maintainer accepts it (the issue then closes). If something is wrong, the bot says what to fix; edit the issue with the fixed content and it is checked again. (Technically: the same checks as `genmodelfit contrib validate`; the bot opens a pull request that adds the file to `data/contrib/`.)

**Changes after the bot made its pull request:** edit the issue body. The new content is checked again; if it passes, the pull request is updated, and if it does not, the pull request is closed until you fix it. Both the issue and the pull request say what happened. Editing only the title changes nothing.

**What your measurement is used for:** it is used for people with the same GPU. On NVIDIA with ComfyUI's dynamic VRAM (the default today) only the working memory can be measured, so it is used to cross-check the estimates and does not set levels yet ([accuracy.md](docs/accuracy.md#how-measurements-are-used)). Other measurements adjust the estimate for that GPU, and when two different people's measurements agree they also set the level. One GitHub account counts once, however many measurements it sends (an issue counts as its author's GitHub account, a pull request as the account it came from; see "[After it is accepted](#after-it-is-accepted)"). Out-of-memory results are kept as "not enough on this GPU" records, which stop measurements from setting the level for that setup. A measurement takes effect after a maintainer accepts it and the next version of genmodelfit is released.

- With an `[NG]` in the checklist or an error in the checks, `share` stops without opening anything. How to fix it is in "[3. Check the contents](#3-check-the-contents)".
- When the content is too long for a URL, `share` copies it to the clipboard (with the OS tool: clip, pbcopy, wl-copy, xclip or xsel) and opens the empty form; paste it into the "Measurement (JSON)" box and press "Create". Without a clipboard tool it prints the content between two `---` lines to copy.
- After the page is opened, the measurement is marked as sent on this PC and is not listed again (`genmodelfit share --all` lists it too). `genmodelfit share <file>` sends a given submission file (also one made by `bench --share` or `contrib make`).
- Without a terminal, or with `--json` or `--print-url`, `share` does not open a browser and only prints the URL.
- genmodelfit makes no network access for sending. It only opens the page in your browser; the content is sent only when you press "Create".
- The bot writes your GitHub account in its pull request and commit, so all your measurements from that account count once (by its numeric id, so renaming the account does not change it), however many you send ("[After it is accepted](#after-it-is-accepted)").

## Advanced: bench and pull requests

This part is for people who use git, or who want every option of `genmodelfit bench` (your own workflow, other
settings, `--force`). It also explains in detail what is measured and checked, which applies to the easy way too.

### 1. Measure

1. Start ComfyUI (the one you normally use is fine). Measure **when no other generation is running**.
   - bench does not measure while the queue has jobs (it checks twice: before starting and just before sending).
   - It also stops when GPU utilization (averaged over about 2 seconds) is 30% or more.
   - Utilization is read with `nvidia-smi` on NVIDIA, `amd-smi` or `rocm-smi` (Linux) on AMD, `xpu-smi` (Linux) on Intel, Windows performance counters for AMD and Intel on Windows, and `ioreg` on Apple.
   - If the GPU's free state cannot be checked, bench stops instead of continuing silently (`--force` lets you measure anyway). Examples: no `nvidia-smi` on NVIDIA, several GPUs with the same name so it is unclear which one, or ComfyUI on another PC.
   - If a non-NVIDIA PC has no tool to read utilization, bench does not stop. It only checks that the ComfyUI queue is empty and that no app other than ComfyUI uses more than half of the GPU memory, and leaves a warning and a `note`. Close other apps (games, video export, and so on) before measuring.
   - **Measurements on AMD, Intel and Apple GPUs are especially welcome.** They have no measurements yet, so the PC line of the tables shows no "✓ verified GPU kind" (their levels follow the same rules as NVIDIA). Once two or more GPU models of a kind have measurements, the kind gets the ✓, and the estimates can be tightened too.
2. Put the model files the template uses into ComfyUI's `models` folder.
   - If files are missing, bench shows which ones.
   - `genmodelfit download <template name> --comfyui-dir <ComfyUI folder>` downloads the files in the list into ComfyUI's models folder (checked with sha256). It does not download files whose license is unknown, or gated files.
3. Measure:

```powershell
pip install -U "genmodelfit>=0.11.0"
genmodelfit bench image_z_image_turbo_int8 --url http://127.0.0.1:8188 --resolution 1024x1024 --steps 6
genmodelfit bench text_to_video_wan --url http://127.0.0.1:8188 --resolution 832x480 --frames 81 --steps 20
genmodelfit bench audio_ace_step_1_5_checkpoint --url http://127.0.0.1:8188 --length 30 --steps 8
genmodelfit bench 3d_hunyuan3d_image_to_model --url http://127.0.0.1:8188 --steps 15
```

Always pass `--url` (there is no default URL, so you cannot send to another ComfyUI by mistake).

Template and settings:

- `<template>` is the name of an official ComfyUI template (`genmodelfit info <name>` shows the ID in the list).
- The workflow is built from the template ComfyUI has, or else from the official template on GitHub.
- To measure your own workflow, save it with "Export (API)" in ComfyUI and pass the JSON with `--workflow`.
- Change settings with `--resolution WIDTHxHEIGHT` (or `--width` and `--height`), `--frames`, `--length SECONDS` (audio) and `--steps`.
- Change other inputs with `--set INPUT_NAME=VALUE` (every input with that name changes).
- A result is recorded as "non-standard" (`standard: false`) and is not used for estimates in two cases: you changed anything other than size, length, steps and seed with `--set`, or you used `--workflow`.
- For `--set`, only numeric values are kept in `note`. For strings only the name is kept, since a string may be a prompt.
- Measure at **the default settings for the type** (image 1024x1024, video 832x480 with 81 frames, music 30 seconds, sound effects and speech 10 seconds) so the result is used as is for judgments without options (from 0.11.0 every model is judged at these settings unless options are given; [docs/scoring.md](docs/scoring.md) "Settings used for judging"). Measurements at other settings still shift the estimates.
- A template whose size comes from the input image (depth estimation, background removal and so on; it has no size input) is measured without `--resolution`; bench writes no size. If the template is in the table the maintainer checked (`estimate.NO_SIZE_INPUT`), the measurement is used for judging as is. A size-less submission for a template not in the table gets the `settings_missing` notice in its PR (the maintainer decides whether to add it to the table).

Model loading and caching:

- By default bench does not send `/free`, so it does not unload models other people loaded.
- If models are already loaded, their share can end up in the result. bench then shows a warning and leaves the amount before the start in `note`.
- **If the ComfyUI is only yours, add `--free`.** It unloads models and clears the cache of earlier results, then measures from loading.
- If you send the same content again without `--free`, ComfyUI returns the earlier result and runs nothing, so bench reports that it could not measure.
- If only some nodes reused earlier results, the memory for those nodes was not measured, so the result is "non-standard".
- The result is also "non-standard" if other jobs ran in ComfyUI during the measurement, or if the run took 10 seconds or more to start after sending.

Timeout and files:

- On timeout (`--timeout SECONDS`, default 1800) or Ctrl+C, bench cancels only its own prompt. It interrupts it with the prompt_id if running, or removes it from the queue if waiting. It does not stop other people's jobs.
- For templates that need an input image (such as image to 3D), bench uploads the official template's image to ComfyUI. It never overwrites: if a different image has the same name, ComfyUI gives a new name.
- Output of save nodes goes to `genmodelfit_bench/` in ComfyUI's output folder. Delete it if you do not need it.
- With ComfyUI's dynamic VRAM (the default), ComfyUI can stop responding at settings that barely fit the VRAM (the log stalls after aimdo reports `budget_deficit`; seen in #102 on an RTX 5080 16GB with Wan at 81 frames). bench's timeout cannot cancel it then, so stop only the ComfyUI you started, restart it with `--disable-dynamic-vram` and measure again.

#### What is measured

| Field | How it is measured |
|---|---|
| `peak_vram_gb` | Reads `/system_stats` every 0.2 seconds during generation and takes the maximum of the amount torch allocated (`torch_vram_total`). This is the ComfyUI process's share and does not include the CUDA context (about 0.3 to 0.5 GB). If `torch_vram_total` is unavailable, the increase over the whole GPU is used instead and a warning is left. |
| "GPU-wide increase" in `note` | The maximum total GPU usage (`nvidia-smi`; the tool above for non-NVIDIA, about every 1 second) minus the value before the start. Changes by other apps are included. Not written if there is no tool. |
| `peak_ram_gb` | Maximum resident memory of the ComfyUI process (and its child processes) listening on the port. Only processes whose command line contains `main.py` or `comfy` are counted. Empty if ComfyUI is on another PC or the process cannot be identified. |
| `seconds` | From sending the workflow to its end (including model loading). The run time ComfyUI counted goes into `note`. |
| `offload` | Read from the ComfyUI log: `dynamic` (dynamic VRAM) / `partial` (only part was put on the GPU) / `lowvram` (started with `--lowvram` / `--novram`) / `none` (everything fit) / `oom` (failed from out of memory). Empty if unknown. |
| `gpu_vram_gb` | Total VRAM from `/system_stats` in GiB, rounded to 2 decimals (about 8.0 for a nominal 8 GB GPU). Old bundled rows hold the nominal value (such as `16`). |
| `outcome` | `ok` (ran to the end) / `oom` (failed from out of memory) / `error` (failed for another reason) |
| `standard` | Whether the official template was kept as is (only size, length, steps and seed changed), no other job got mixed in, and all nodes were executed. |

Out-of-memory (OOM) data is welcome too: it is important data saying "this GPU cannot run it". Always send it with `outcome: "oom"` unchanged.

- bench only shows a failed result and does not save it to the cache (cached measurements are used for estimates on that PC right away).
- To save it, add `--save-failed` (to the cache) or `--output PATH`.
- "Non-standard" results are the same: save them with `--save-nonstandard` or `--output`.
- Estimates use only measurements with `outcome` `ok` and `standard` `true`.

### 2. Create the submission JSON

From here on is the way with git and a clone of your fork (if you used "[the easy way](#the-easy-way-no-git-needed)", you can skip it; the table of what goes in is the same).

Clone your fork (GitHub Fork) and run bench with `--submission` in that folder. Running `pip install -e .` in the clone lets you run `contrib validate` with the same main version as CI.

```powershell
git clone https://github.com/<your-name>/genmodelfit
cd genmodelfit
pip install -e .
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 1024x1024 --submission data/contrib/
```

- The file is a submission file (`contrib_schema` 2). If you pass a folder to `--submission`, the file is written as `data/contrib/<GPU>/<date>-<sha8>.json`. Example: `data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json`, where `sha8` is the first 8 hex digits of the content's sha256. If you pass a file name, it is written under that name.
- Pass the folder with a trailing `/` (`data/contrib/`). On a first submission the folder does not exist yet, and without the `/` a file named `data/contrib` is created.
- You can also create it from a result you measured and saved earlier (`<cache>/measurements/....json` or a file from `--output`): `genmodelfit contrib make --from <bench result> --output data/contrib/`
- If ComfyUI was on another PC, add `--no-pc-info` to leave out this PC's OS, memory and GPU generation.
- Submit failed (OOM) results as they are. Results under non-standard conditions are accepted as records but not used in aggregation.

#### What goes in and what does not

| Goes in | Example |
|---|---|
| GPU name, total VRAM, generation, compute capability, driver version (generation and compute capability only on NVIDIA) | `NVIDIA GeForce RTX 3070`, `8.0`, `ampere`, `8.6`, `566.03` |
| GPU vendor and PyTorch backend (from the 0.9.0 bench) | `amd` and `rocm`, `intel` and `xpu`, `apple` and `mps`, `nvidia` and `cuda` |
| GPU PCI model ID (0.9.0; only for AMD and Intel with ComfyUI on the same PC; a model number that does not identify the unit or the slot) | `1002:744c` (Radeon RX 7900 XTX; the same on any PC with that model) |
| Apple GPU chip name (0.9.0; only names matching the system_profiler form) | `Apple M2 Pro` |
| ComfyUI version, OS family (no version or build), main memory (rounded to an integer GiB) | `0.40.0`, `windows`, `32` |
| Model ID, settings, result, peak VRAM / memory, time, how it was loaded, how it was measured | `image_sdxl_simple`, `832x832`, `ok`, `6.9` ... |
| Date (UTC date only) | `2026-10-21` |
| Note (`note`: bench's description of how it measured) | A sentence containing something that looks like a path, URL, PC name, user name, email, UUID, IP or time is removed as a whole |

**What does not go in**: PC name, user name, folder paths, GPU UUID, PCI slot address (bus number; different from the model ID), serial numbers, IP / MAC addresses, CPU name, time, the text given to bench's `--source`, and prompts.
Fields other than the defined ones are rejected by the CI check.
However, the automatic check reads patterns, not meaning. It misses things like these, so read the `note` yourself in step 3:
- names written in ordinary words ("measured on Taro Yamada's PC", `user: yamada`, `my name is ...`)
- PC names without a fixed form (`yamada-desktop`, `yamada-ws`)
- letters separated by spaces (`t a r o @ g m a i l`), short base64 (`dGFybw==`) and base64 that looks like words
- addresses, postal codes and IDs of other services (Discord, Steam, ...)
- user names under public domains (`github.com/taro`, `hf.co/yamada`; the domains are treated as public places)

On the other hand, ComfyUI node names (such as `HunyuanVideo15ImageToVideo`), 40-digit git commits, 64-digit sha256 and public domains (huggingface.co, hf.co, github.com, comfy.org, pytorch.org, nvidia.com, python.org) are not treated as personal information.

### 3. Check the contents

Always do this before sending.

```powershell
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json
```

This prints everything in the file in readable form, and prints the result of a privacy checklist as `[ok]` / `[NG]`. The checklist looks for paths, PC name, user name, UUID, PCI, IP, URL, email, time and unknown fields, and also checks that this PC's name and user name are not in the file.

- If there is an `[NG]`, fix it before sending. If you edit the file by hand, the `sha8` in its name changes, and `contrib validate` tells you the correct name.
- **Read the `note` yourself.** The automatic check does not understand what a sentence means. You may delete any sentence you are unsure about.
- `genmodelfit contrib validate data/contrib` runs the same check as CI on your PC.
- Only mechanical checks (shape, value range, personal information, duplicates, file name) make a submission fail (the same approach as llmfit; 2026-10-22).
- Notices are shown but do not fail. They are: a value more than 40% off the estimate by the formula; a value smaller than the lower bound (weights + fixed part) although everything was put on the GPU (`offload: none`); a value that changes the level on a PC with the same GPU (checked with the submission's VRAM and with variants of the same name that have different VRAM, each with 16 / 32 / 64 GB of memory); a model not in the list; an unknown GPU or a GPU name that does not fit the NVIDIA name form (`gpu_name_form`); and VRAM that is not the name's nominal amount (`gpu_vram_nominal`, e.g. a modified card).
- A GPU name is refused only when it is a placeholder (empty, n/a, none, null, unknown, -) or a generic name (NVIDIA Graphics Device etc.) (0.11.0; new GPUs not in the table are accepted).
- Notices are shown above the job summary, and the person who merges the PR reads them. If there is a reason, write it in the PR description (you may be asked to measure again).

### 4. Send a pull request

1. Add only `data/contrib/<GPU>/<date>-<sha8>.json`.
   - Do not change other files (put code changes in a separate PR).
   - The `Contrib` workflow fails a PR that changes files outside data/contrib.
   - A submission PR **only adds files**. A PR that deletes, modifies or renames an existing file in data/contrib, adds a file whose name or place is not `<GPU>/<date>-<sha8>.json` (lowercase `.json`), or adds no submission at all also fails (#90).
   - To fix a submission you sent before, send the fixed one as a new file and write the old file's name and the reason in the PR description. The maintainer removes the old file (docs/release.md "Releasing a version").
2. Create a branch, commit, and push to your fork:

   ```powershell
   git switch -c contrib/<GPU>-<model>
   git add data/contrib/<GPU>/<date>-<sha8>.json
   git commit -m "contrib: <GPU> <model>"
   git push -u origin HEAD
   ```

   Then create a pull request on GitHub against main of laxshmate/genmodelfit.
   - It helps if the description lists the ComfyUI startup options (such as `--lowvram`; only those you changed).
   - A personal name or email address is not needed.
   - For a first-time contributor, CI may not run until a maintainer allows the workflow.
3. CI (the `Contrib` workflow) runs `genmodelfit contrib validate` to check shape, value range, personal information, duplicates and file name, and writes the result to the job summary (Summary).
   - It also writes the result of `contrib aggregate --report` to the summary.
   - The check runs with genmodelfit from the PR's target (main).
   - The merge is the maintainer's review (there is no separate mechanism where a person approves each submission).

**genmodelfit does not send measurements.** The only thing shared is what you send yourself: the measurement issue you create with "Create" (a bot turns it into a pull request) or the pull request you open on your own GitHub. Once merged, the submission is published under MPL-2.0 (see [License](#license) below).

#### After it is accepted

- After the merge, your file is in data/contrib, but it **reaches users with the next version** of genmodelfit (#90). When making a release, the maintainer groups submissions with the same model, settings and GPU using `genmodelfit contrib aggregate --output src/genmodelfit/data/measurements_community.json` (outliers are removed with the median and MAD; [docs/release.md](docs/release.md)). `genmodelfit contrib aggregate --check` shows the submissions not in it yet.
- Only **PCs with the same GPU as the measured one** use the result. Judgments for PCs with other GPUs do not change (0.9.1; the same approach as llmfit). "Same GPU" means:
  - the PCI model ID is the same (when both have it), otherwise the normalized name is the same (a generic name is not counted as a match), and
  - the ratio of total VRAM is 0.95 or more (the same card reports a slightly different total on Linux and Windows). An 8 GB and a 16 GB RTX 4060 Ti are treated as different GPUs.
- On such a PC, **from 1 submission** the basis of the estimate for that configuration becomes `measured` (a measurement, treated like the maintainer's; the table's VRAM gets an `m`), and the number of sources (`sample_count`) is shown.
- The value is used in both directions, lowering or raising the required amount. A value for everything on the GPU is never lowered below weights + fixed part, and never raised above the GPU's total.
- Measurements of the same GPU and the same settings also become grounds for deciding the level once **two or more independent sources** agree (the maintainer's rows count as one source, and each other person counts as one). Sources are counted as groups of submissions connected by the same pull request or the same person (#113; `contrib.submission_sources`): one pull request is at most one source, and one GitHub account is one source however many files, pull requests or issues it sends. The person is a **GitHub account**: for a measurement issue, the issue's author (the bot writes `Submitted-by: github:<account>` and `Submitted-by-id: <the account's numeric id>` in its commit, and the numeric id, which does not change when the account is renamed, is what is counted; older bot commits without the id count by the account name. These lines are trusted only when the merge commit is `Merge pull request #N from laxshmate/measurement/issue-M`, the bot's own branch in this repository, and the commit is the bot's); for a pull request, the account the pull request came from, read from the first line GitHub writes in the merge commit (`Merge pull request #N from <account>/<branch>`; git gives only the name here). The commit authors' e-mail addresses are not used for this, since anyone can write any address in a commit. Measurement pull requests are merged with "Create a merge commit" only; anything else (a commit directly on main, squash or rebase) falls back to the commit author's e-mail address, which is weaker. **An account counts once, but pull requests from a different account, from an organization the person owns, or (for pull requests) after a rename count as different people**, so the maintainer checks the author of each pull request or issue when merging (the bot's pull request says how many submissions the same account already sent for that GPU and model, and has a checklist item for it). Measurements only raise the level and never lower it. See [docs/scoring.md](docs/scoring.md) for details.
- For a submission PR, the `Contrib` workflow shows the result of `contrib aggregate --report` in the summary. The calibration table (calibration.json, [docs/accuracy.md](docs/accuracy.md)) is rebuilt by a maintainer with `python -m genmodelfit.calibrate` when importing measurements for a release.

#### Maintainer measurements (measurements.json)

The bundled `src/genmodelfit/data/measurements.json` holds the measurements the maintainer made. Its fields are in the [Measurement](docs/schema.md#measurement) section of docs/schema.md, and the submission format is in the [contrib](docs/schema.md#contrib) section. Send users' measurements to data/contrib (do not edit measurements.json directly).

## Other changes

Code changes are welcome too.

- When adding a dependency, use only MIT / Apache-2.0 / BSD ones (apps that embed this tool distribute without publishing their source).
- Do not import ComfyUI code (GPL-3.0); talk to it over HTTP only.
- Put the same 3-line MPL-2.0 notice as in other files at the top of every new `.py` file.
- Before sending, run `python tools/ci_local.py` to pass the same checks as CI on your PC (tests on Python 3.10 to 3.13, ruff, and the package). You only need [uv](https://docs.astral.sh/uv/). On Windows, `--linux` also runs it inside WSL.
- For PRs from forks, GitHub's CI runs as well.

## License

The code of genmodelfit and the bundled data (db.json, measurements, coefficients) are under [MPL-2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE).

- Fixes, measurements and other data you send are accepted **as released under MPL-2.0**, like the rest of the project (inbound = outbound).
- By opening a pull request or a measurement issue you agree to this.
- There is no CLA (a document assigning copyright or granting extra permission).
- Only submit work you wrote yourself or are allowed to release under MPL-2.0.
