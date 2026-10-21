# genmodelfit

**English** · [日本語](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md)

**Right-size image / video / audio / 3D generation models to your PC.**

genmodelfit is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork.

> Status: v0.1.0 (alpha).

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 12 of 80 (224 hidden: uncensored 26, tools 34, license unknown 28, too tight 136; 25
    more workflows merged into their models)
== image (画像): top 2 of 37  (-m image for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Marginal     54 Z-Image-Turbo            gen+5   int8  7.6m  96% GPU~    apache-2.0 yes
image ● Marginal     42 MOSS-Image SFT           gen+1   int8   4.2  53% GPU+RAM apache-2.0 yes
== video (動画): top 2 of 4  (-m video for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
video ● Marginal     50 Wan 2.1 1.3B T2V         gen     fp16  7.8m  97% GPU~    apache-2.0 yes
video ● Marginal     39 Wan 2.1 ControlNet 1.3B  edit    bf16   7.7  96% GPU+RAM apache-2.0 yes
== music (音楽): top 2 of 10  (-m music for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
music ● Good         65 ACE-Step 1.5 Turbo       gen+2   bf16  6.0m  76% GPU~    mit        yes
music ● Marginal     40 ACE-Step 1.5XL Turbo     gen     bf16   5.2  65% GPU+RAM mit        yes
== sfx (効果音): top 2 of 2  (-m sfx for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
sfx   ● Perfect      68 stable-audio-3-small-sfx gen     fp32   3.6  45% GPU     other      license
sfx   ● Good         54 stable-audio-open-1.0    gen+1   fp32   4.9  61% GPU~    other      license
== tts (読み上げ): top 2 of 20  (-m tts for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
tts   ● Perfect      80 Chatterbox               gen     fp32   3.5  44% GPU     mit        workflow
tts   ● Perfect      80 VieNeu-TTS-v3-Turbo      gen     q8     1.6  21% GPU     apache-2.0 workflow
== 3d (3D): top 2 of 7  (-m 3d for more)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
3d    ● Good         64 HY 3D 2.0                gen     fp16  6.7m  84% GPU     other      license
3d    ● Good         57 TripoSplat               gen     fp16   5.8  72% GPU     other      license
Order: runs, Verified=yes, Perfect/Good, Marginal, Score. Score 0-100. VRAM: GiB needed, m =
    measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Verified: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
Note: 26 uncensored model(s) hidden; add --content all to show them.
Note: 28 model(s) with an unknown license hidden; add --include-unknown-license to show them.
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
```

Without `-m` / `--task`, `recommend` shows each of the six types (image, video, music, sfx, tts, 3d). One row is
one **model**: workflows that use the same weights (same family and size) in another precision or for another use
are folded into it (`+5` = five more workflows; `--all-workflows` lists them), and the best precision that fits is
chosen (`Prec`; like llmfit's best quant). Narrow down inside a type with `--task`:

```
$ genmodelfit recommend --vram 8 --ram 32 -m image --task edit -n 3
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 3 of 16 (311 hidden: other types 166, uncensored 21, tools 19, other uses 63, license
    unknown 8, too tight 34; 2 more workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Marginal     41 FLUX.2 [Klein] 4B        edit+1  fp8    4.5  56% GPU+RAM apache-2.0 yes
image ● Marginal     41 FLUX.2 [Klein] 4B Base   edit    fp8    4.5  56% GPU+RAM apache-2.0 yes
image ● Marginal     41 Z-Image-Turbo            edit+5  bf16   7.5  93% GPU+RAM apache-2.0 yes
Order: runs, Verified=yes, Perfect/Good, Marginal, Score. Score 0-100. VRAM: GiB needed, m =
    measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
Mem%: share of VRAM (of RAM for CPU), RAM = system RAM short. Verified: yes, or first failed check.
One row per model: Prec = best precision that fits, +N = other workflows (--all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
Note: 21 uncensored model(s) hidden; add --content all to show them.
Note: 8 model(s) with an unknown license hidden; add --include-unknown-license to show them.
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
```

The default output is a table for people; add `--json` for apps and agents (llmfit prints JSON by default
for `recommend`; genmodelfit keeps the table so a first try is readable).

## Why

llmfit covers LLMs only ([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41)). For diffusion, video and
audio models the memory needed changes several times over with precision, resolution, frame count and
offloading to system RAM, so they need their own database and scoring.

## Install

Recommended: install it as a command-line tool with [pipx](https://pipx.pypa.io/) or
[uv](https://docs.astral.sh/uv/). They put `genmodelfit` on PATH for you (run the PATH command once, then open a new
terminal):

```
pipx install genmodelfit        # once: pipx ensurepath
uv tool install genmodelfit     # once: uv tool update-shell
```

Or with pip:

```
pip install genmodelfit
```

**`python -m genmodelfit …` always works**, whatever PATH says (e.g. `python -m genmodelfit recommend`; on Windows
also `py -m genmodelfit …`). Use it when `genmodelfit` is "not recognized" / "command not found".

If pip printed *"WARNING: The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH"*, the command
was installed but that folder is not on PATH. Common cases:

- Python from python.org installed without "Add python.exe to PATH" → the `Scripts` folder of that Python is not on PATH.
- pip fell back to a user install (no write access to the system Python) → the command is in
  `%APPDATA%\Python\Python3XX\Scripts` (Windows) or `~/.local/bin` (Linux), which is often not on PATH.
- Microsoft Store Python → its `Scripts` folder lives deep under `%LOCALAPPDATA%\Packages\…` and is not on PATH.
- Installed into a virtual environment that is not activated → activate it (`.venv\Scripts\activate` /
  `source .venv/bin/activate`) or run `.venv\Scripts\genmodelfit`.

Fix: add the folder named in the warning to PATH (or use pipx / uv above, or `python -m genmodelfit`).
**A terminal opened before PATH changed does not see the change: open a new one.**

From source:

```
git clone https://github.com/laxshmate/genmodelfit && cd genmodelfit && pip install -e .
```

Python 3.10+. Windows and Linux. GPUs: NVIDIA (CUDA) in v0.1; other PCs are judged as CPU-only.

## Usage

| command | what it does |
|---|---|
| `genmodelfit system [--all-gpus]` | detected GPU (generation, driver, PyTorch builds, ComfyUI precision), RAM, page file, disk, and whether the model list is still fresh (Data); `--all-gpus` also lists non-NVIDIA adapters |
| `genmodelfit recommend [-m image,video] [--task edit] [-n 5] [--auto-only]` | best models for this PC: per type without `-m` / `--task` (utility tools such as depth/pose/upscale only with `--include-tools`) |
| `genmodelfit fit [-m video] [--fit good] [--sort score] [--all]` | the models that run on this PC (Perfect / Good / Marginal) in one table: by type, then fit level, then Score; `--all` also shows tools, models that do not run, unjudged (`?`) and unknown-license ones |
| `genmodelfit search <text>` | models whose id or name contains the text, judged for this PC, every fit level, same order as `fit` (exit 0 even with no match) |
| `genmodelfit types` | the six types, the uses inside each (`--task` names) and counts |
| `genmodelfit info <id>` | files, licenses, estimate, limits, gate checks, other workflows of the same model |
| `genmodelfit gate` | how many models can be auto-recommended, and why the rest are held |
| `genmodelfit update` | rebuild the model list from upstream |
| `genmodelfit update --fields-only [--offline]` | keep the list as it is; re-apply only the app fields (`languages` / `runtime` / `description`) and the hand-added models |
| `genmodelfit update --from-release` | fetch the signed model list published by this project |
| `genmodelfit review list [--json]` / `approve` / `reject` | models waiting for a person's review after `update` ([Maintenance](#maintenance)) |
| `genmodelfit download <model> [--comfyui-dir DIR] [--list]` | download all files of the model for this PC (precision chosen like `recommend`), after showing licenses, sizes and free disk and asking you; sha256-checked, resumable ([below](#download)) |
| `genmodelfit bench <template> --url URL` | measure VRAM / RAM / time on a running ComfyUI |

Common options:

- `--json` — machine-readable output for apps and agents ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md))
- `--vram 12 --ram 32` — judge with other VRAM / RAM sizes; the GPU generation of this PC is kept when it has a usable NVIDIA GPU ([docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md))
- `--width 1024 --height 1024`, `--frames 81`, `--length 30`, `--precision fp8` — judge a specific setting
- `-m image,video` — types (comma-separated)
- `--task edit` — uses inside a type: `gen`, `edit`, `i2v`, `lipsync`, `motion`, `voice`, `upscale`, `tool`, or task codes
  such as `t2i` (`genmodelfit types`)
- `--all-workflows` — one row per workflow instead of one per model
- `--content sfw|all|uncensored` — filter by content mark (below; default `sfw`)
- `--fit perfect|good|marginal|runnable|all` — only that level, every level that runs (`fit` default) or everything
- `--min-fit perfect|good|marginal|too_tight` — only this level or better (recommend: `marginal` by default; not together with `--fit`)
- `--sort type|score|vram|disk|name` (`fit`, `search`) — `type` (default): type, then fit level, then Score; `score`: one ranking across types; `vram` / `disk`: smallest first
- `--include-unknown-license` — also show models with a file whose license is unknown (hidden by default)
- `--license apache-2.0,mit` — every file's license must be in the list; unknown never matches
- `--comfyui-version 0.40.0` — the ComfyUI version your app runs, for the gate

Order: `fit` and `search` show one table, **by type** (image, video, music, sfx, tts, 3d), then **fit level**
(Perfect → Good → Marginal → `?` → Too tight), then **Score** (0–100: fit level, VRAM headroom, readiness for
auto-recommendation, measured or not, popularity; formula in
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)); auto-recommendable
models are not moved up (see the Verified column). `recommend` shows the top models of each type, with auto-recommendable
ones first. The first lines show this PC and how many models were hidden and why (`uncensored`, `tools`,
`license unknown`, `too tight`, …). `--help` of each command lists preconditions, exit codes and examples.

The **Verified** column is the JSON field `auto` (it was called Auto up to 0.5.0): `yes`, or the first failed gate check.
The JSON field names (`auto`, `held_by`) and the gate check ids are unchanged.

Example (`COLUMNS=100 genmodelfit fit --vram 24 --ram 128`, 24 GB GPU / 128 GB RAM assumed; rows cut with …):

```
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 146 (91 hidden: uncensored 26, tools 34, license unknown 28, too tight 3; 92 more
    workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Perfect      90 SDXL1.0 Base             gen+3   fp16  8.0m  33% GPU     openrail++ license
image ● Perfect      84 MOSS-Image SFT           gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      83 FLUX.2 [Klein] 4B Base   edit    fp8   13.2  55% GPU     apache-2.0 yes
image ● Perfect      82 MOSS-Image Base          gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      81 Anima Base 1.0           gen+3   bf16   7.1  30% GPU     other      license
image ● Perfect      80 Anima Preview 3 Base     gen     bf16   7.0  29% GPU     other      license
…
video ● Perfect      92 Wan 2.1 1.3B T2V         gen     fp16 13.3m  55% GPU     apache-2.0 yes
video ● Good         67 Wan 2.2 5B TI2V          gen     fp16 15.7m  66% GPU~    apache-2.0 yes
video ● Good         63 Wan 2.1 ControlNet 1.3B  edit    bf16  17.1  71% GPU     apache-2.0 yes
video ● Marginal     44 Wan 2.2 14B I2V          i2v+2   fp8   21.0  88% GPU~    apache-2.0 yes
…
music ● Perfect      94 ACE-Step 1.5 Turbo       gen+2   bf16 10.8m  45% GPU     mit        yes
music ● Perfect      82 ACE Step v1 3.5B         gen+2   bf16   8.7  36% GPU     apache-2.0 yes
…
tts   ● Perfect      81 Qwen3-TTS-12…CustomVoice gen     bf16   5.0  21% GPU     apache-2.0 workflow
tts   ● Perfect      80 Chatterbox               gen     fp32   3.5  14% GPU     mit        workflow
…
3d    ● Perfect      89 HY 3D 2.1                gen     fp16  9.4m  39% GPU     other      license
3d    ● Perfect      88 HY 3D 2.0                gen     fp16  6.7m  28% GPU     other      license
3d    ● Perfect      79 HY 3D 2.0 MV             gen     fp16   6.8  29% GPU     other      license
3d    ● Perfect      77 HY 3D 2.0 MV Turbo       gen     fp16   6.8  29% GPU     other      license
Order: Type, Status, Score (--sort). Score 0-100. VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ is at most Good; GPU+RAM and CPU are at most Marginal (Mem% can be low).
…
```

Python:

```python
import genmodelfit as gmf

pc = gmf.system()
for f in gmf.recommend(modality="image", task="edit", limit=5, comfyui_version="0.40.0"):
    print(
        f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, f.estimate.vram_gb, f.estimate.basis
    )
    print([w["id"] for w in f.workflows])  # other workflows of the same model
    f.to_dict()  # the same dict as one item of `recommend --json`

# the same list as `genmodelfit fit` (runnable only, by type → level → score)
rows = gmf.fit_list(sort="type")  # fit="good", show_all=True, include_tools=True, … like the CLI
gmf.search("flux", license=["apache-2.0", "mit"])

st = gmf.data_status()  # is the model list fresh enough for auto-recommendation?
if st.expired:  # HEAD checks older than 30 days → auto-recommendations drop
    print("run: genmodelfit update --from-release", st.expires_at, st.auto_count)
```

## Download

```
genmodelfit download "Z-Image-Turbo" --list                 # what would be downloaded (no network, writes nothing)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
```

- **Which files**: all files of one workflow. A model name (as in the tables) picks the workflow and precision like
  `recommend`: the highest precision that runs on this PC without offloading. A workflow id is used as is.
  `--precision int8` picks a precision; `--budget 15` (GB) caps the download (files already there are not counted).
- **Where**: `--comfyui-dir` puts each file in the ComfyUI folder the template names
  (`models/diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, …; subfolders in the file name are kept).
  `--output-dir` (default: genmodelfit's user data folder, `models`) uses the same layout, so it can be added to
  ComfyUI's `extra_model_paths.yaml`.
- **Consent first**: before anything is downloaded you see every file with its license and the license URL, licenses
  that are not on the allowlist (commercial use without use restrictions; [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md))
  are marked, then the total size and the free disk, and you are asked `[y/N]`. `--yes` skips the question
  (required with `--json`). Read the terms of each file before you use the model.
- **Not downloaded**: gated files (genmodelfit never handles logins or tokens; request access on the model page and
  download them yourself), uncensored models unless you pass `--content all` (or `uncensored`), files whose license is
  unknown unless you pass `--include-unknown-license`, files without a sha256 or size in the list, hosts other than
  Hugging Face / GitHub.
- **Safety**: every file is checked with the sha256 in the list while it is written; a mismatch deletes it and fails.
  A download goes to `<file>.part` first; after Ctrl-C or a network error run the same command again and it resumes
  with an HTTP Range request (if the server ignores Range it starts over, at most 3 times). Files already there are
  hashed again right before they are skipped; a different file with the same name is never overwritten. A
  `<file>.lock` next to it stops two downloads of the same file at once (a lock left by a process that is gone is
  cleaned up). It does not start unless the free disk is
  the download + 1 GB. Nothing is written outside the destination: names with `..`, absolute paths, drive letters,
  `\`, reserved Windows names (also as folder names), control / bidi characters, and symbolic links / junctions
  that lead outside are refused.
- **Network**: only the URLs in the list (checked anonymously with HEAD), https only. Every redirect must stay on
  the hosts below; a URL that could be read as two different hosts (`\`, spaces, control characters, `%` or `@`
  in the host, non-ASCII hosts) is refused before any connection, and the host is read the way requests / urllib3
  read it. Allowed:
  `huggingface.co` / `*.hf.co` (CDN, Xet) / `github.com` / `objects.githubusercontent.com` /
  `release-assets.githubusercontent.com`. No `~/.netrc` credentials, no tokens, nothing about this PC (User-Agent and
  Range only). Waits for `Retry-After` on 429 / 503.

Python:

```python
import genmodelfit as gmf

plan = gmf.plan_download("Z-Image-Turbo", comfyui_dir=r"C:\ComfyUI")  # no network, writes nothing
for f in plan.files:  # status: download / resume / present / conflict / unavailable
    print(f.relpath, f.size_bytes, f.license, f.license_url, f.status)
print(plan.blocked, plan.warnings, plan.download_bytes, plan.disk_free_bytes)


def ask(plan) -> bool:  # show licenses, URLs, total and free disk; True to go on
    ...


def show(p):  # gmf.DownloadProgress
    print(p.phase, p.name, p.done_bytes, p.total_bytes)


result = gmf.download(plan, confirm=ask, progress=show)
# or gmf.download("Z-Image-Turbo", yes=True, comfyui_dir=...) when the app already asked
```

The library prints nothing. `gmf.download` raises `gmf.DownloadRefused` (blocked, no consent), `gmf.DownloadError`
(network, sha256 mismatch, disk) and lets `KeyboardInterrupt` through, keeping the `.part` file for the next call.

## Hardware

- Only NVIDIA GPUs confirmed by `nvidia-smi` are used for judging (v0.1). A GPU is judged as **CPU** when no PyTorch
  CUDA build supports its generation (Kepler and older) or its driver is too old for all of them (CUDA 12 builds need
  525+, CUDA 13 builds need 580+, Blackwell needs 570+). The reason is shown in `system` and `--json`.
- On GPUs where ComfyUI computes in fp32 (Maxwell, some Pascal, GTX 16xx, T1000 …) working memory and the VAE are doubled.
  On GPUs without bf16 (Pascal, Volta, Turing), model families that ComfyUI never runs in fp16 (LTX, Qwen Image, Z-Image,
  HunyuanVideo, ACE-Step …) also compute in fp32, and so do VAEs without fp16 (SD / Flux KL, LTX).
- The level uses total VRAM; when the VRAM free right now is not enough, a note says to close other apps (`vram_free_short`).
- When the RAM is short, a fixed-size page file / swap sets how far "slow but runs" goes (system-managed: 1.5× RAM).
- AMD / Intel GPUs are reported (Windows: WMI + registry, Linux: `lspci`) but never used for judging. Host name,
  user name, GPU UUID and PCI ids are not collected. Details and sources: [docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md).

## Fit levels

Like llmfit, the level comes from the utilization of the memory the run mode uses (VRAM on the GPU, RAM on CPU):

| level | utilization | note |
|---|---|---|
| **Perfect** | ≤ 60% | all on the GPU only |
| **Good** | ≤ 85% | also when parts are swapped in turn (Mode `GPU~`) |
| **Marginal** | ≤ 98% | also offload to RAM (Mode `GPU+RAM`, slow), CPU, or slightly too little RAM |
| **Too tight** | > 98% | does not run |

Every estimate says whether it is **measured** (`m` after the VRAM figure, a real run) or **estimated** (formula).
Measured values win. Models whose download size is unknown are not judged (shown as `?`).

## Data sources

The bundled list has 329 entries (image 163, video 115, TTS 22, music 16, 3D 10, SFX 3) that make 210 models
(2026-10-21; refreshed weekly, see "Maintenance").
Like llmfit, Hugging Face models are included only with at least 10,000 all-time downloads (1,000 for trending
ones), and merges and test stubs are skipped ([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)).

- [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT): which models a workflow
  uses, download URLs, hashes, required ComfyUI version, custom nodes. API (cloud) templates are excluded.
  The index's `size` field is a rounded, hand-written total, so genmodelfit uses the sum of the file sizes.
- [Hugging Face Hub API](https://huggingface.co/docs/hub/api): licenses (following `base_model` chains and a
  table of verified upstreams for repackaged files), gating, file sizes, sha256, and discovery of community models.
- Civitai is **not** used: its terms limit automated access and forbid reselling compiled data.
- Hand-added models (`source: manual`, `src/genmodelfit/data/manual.json`): models the popularity-based collection misses
  (Tsukuyomi-chan piper-plus, Irodori-TTS, Stable Audio 3 Small music). Each file's license was read from the upstream
  card and is compared with it on every `update`. Per-model `languages` (prompt / speech, ISO 639-1), `runtime`
  (`comfyui` / `piper` / `irodori-tts`) and a short hand-written `description` (ja / en) are in the list for apps that build
  their screens from it alone ([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)).
- Measurements: `src/genmodelfit/data/measurements.json`. Contributions welcome (see [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)).

Details: [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md). genmodelfit stores model **information only**; it never hosts weights.

## Content marks

- `sfw` — default.
- `uncensored` — models whose safety filtering was removed or that are made for adult output
  (Hugging Face `not-for-all-audiences` tag, or names such as uncensored / abliterated / nsfw). They are in the list
  (26 in the bundled list) and marked `N` in the `Censored` column of tables (`Y` = censored, content `sfw`; the column appears only with
  `--content all|uncensored`), but `recommend` / `fit` / `search` and `gmf.recommend()` / `gmf.search()` show them
  **only with `--content all` or `--content uncensored`** (`content="all"` / `"uncensored"` in Python).
  `info <id>` works for them as usual. The evidence is kept in
  `content_evidence`. Whether to show them is up to the app using genmodelfit; they are never auto-recommended
  by default.
- Models aimed at sexual content involving minors are **excluded** from the list entirely.

With `--content all` (or `uncensored`) the tables get a `Censored` column right after Score (`Y` = censored, `N` = uncensored);
with the default `sfw` the column is not shown. From `COLUMNS=100 genmodelfit fit --vram 24 --ram 128 --content all`:

```
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 155 (82 hidden: tools 34, license unknown 43, size unknown 2, too tight 3; 92 more
    workflows merged into their models)
Type  Status      Score Censored Model                           Use     Prec  VRAM Mode    Verified
image ● Perfect      90 Y        SDXL1.0 Base                    gen+3   fp16  8.0m GPU     license
image ● Perfect      84 Y        MOSS-Image SFT                  gen+1   int8  13.8 GPU     yes
image ● Perfect      83 Y        FLUX.2 [Klein] 4B Base          edit    fp8   13.2 GPU     yes
…
image ● Perfect      73 N        Noct-Q-Uncensore…Image-2.1 Base gen     int8   8.5 GPU     license
image ● Perfect      70 N        LUSTIFY-SDXL-NSF…2-0-INPAINTING gen     fp16   8.2 GPU     license
…
Censored: N = safety filter removed / adult model
```

## Auto-recommendation gate

A model is `auto: yes` only if **all** machine-checkable conditions hold: every file's license (traced to the
upstream, not the uploader's label) is on a commercial-use allowlist (apache-2.0, mit, bsd, cc0, unlicense;
replaceable), nothing is gated, an official ComfyUI workflow without custom nodes, the ComfyUI version is
supported, every download URL answered an anonymous HEAD within the last 30 days, sha256 is known, the size is
known, the content is `sfw`, and no review is pending (check `reviewed`; see "Maintenance"). Otherwise it is held with machine-readable reasons (`held_reasons` in JSON, e.g. `license`; the table's Verified column shows the first one).
In the bundled list 99 of 329 entries are auto (2026-10-21). When a license cannot be verified it is treated as unknown
and held. See [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md).

## Maintenance

Like llmfit, the list is rebuilt every week, and only reviewed data is published:

1. Once a week a Claude Code skill (`genmodelfit-weekly`) runs on the owner's PC (Windows Task Scheduler). It rebuilds
   the list (`python -m genmodelfit.weekly`: templates, Hugging Face, HEAD checks, file headers), checks it (schema,
   auto count, the 30-day HEAD expiry must be more than 7 days away), reads the change classification
   (`classification` in `diff.json` / `review_queue.json`) and checks the pending models against primary sources
   (`genmodelfit review approve` with evidence URLs, or `review reject`; `--hide-name` when the reason is safety).
   Claude only prepares a branch and a commit.
2. A runner script pushes the branch and opens the pull request. It merges **only if every mechanical check passes**:
   the only changed file is `src/genmodelfit/data/db.json`, the classification recomputed from the two files is safe
   (`python -m genmodelfit.weekly merge-check --base <main db.json> --head <new db.json> --json`, exit 0; models that
   stay pending unchanged are allowed), no review status changed, and CI is green. Otherwise the owner reviews and
   merges the pull request.
3. On merge, `.github/workflows/data.yml` signs main's `db.json` without the pending models and replaces the
   `data-latest` release (0.5.x clients do not know `review`, so pending models are never published).
4. Apps take it with `genmodelfit update --from-release` (signature checked).

`.github/workflows/weekly-data.yml` no longer runs on a schedule; it is kept as a manual backup (workflow_dispatch).

```powershell
genmodelfit diff old_db.json new_db.json            # what changed between two lists (--json for tools)
python -m genmodelfit.weekly --dry-run --out weekly-out   # the same pipeline locally (writes only weekly-out)
python -m genmodelfit.weekly merge-check --base old_db.json --head new_db.json --json   # 0 safe / 3 not safe
```

Safe changes are only those on an allowlist: popularity counts (`downloads`, `downloads_all_time`, `likes`,
template `usage`), `updated`, HEAD results, larger files, stricter licenses, removed models, models dropping out of
auto-recommendation, `sources` statistics and models that stay pending unchanged. Anything else (new models, other
file URLs or sha256, uses, languages, descriptions, license links, names, unknown fields, review changes, …) needs
judgement.

Review of changes: `genmodelfit update` marks new models, licenses that became more permissive, lower memory needs and
newly uncensored models as `review.status: "pending"`. A pending model is never auto-recommended (gate check `reviewed`)
until it is approved with primary-source evidence.

```powershell
genmodelfit review list [--json]                                        # models waiting for review, and why
genmodelfit review approve <id> --evidence <https URL>... [--note TEXT] --db <path>
genmodelfit review reject <id> --note <text> --db <path> [--hide-name]
```

Apps that read the list can filter with `entry.review is None or entry.review.status == "ok"`. Details:
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md) ("review") and
[docs/release.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.md).

## Contributing measurements

Measured runs on more GPUs make the estimates better. In a clone of your fork:

```powershell
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 832x832 --submission data/contrib/
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json   # exactly what will be shared + privacy checklist
```

Then open a pull request that adds only that file. **genmodelfit sends nothing**; the file holds the GPU name / VRAM /
generation / driver, ComfyUI version, OS family, RAM rounded to GiB and the results — never host or user names, paths,
GPU UUID, PCI address, IP or exact times (the checks are pattern based, so read the note yourself). CI checks it with
`genmodelfit contrib validate`. Once two or more consistent submissions exist for the same model, settings and GPU,
estimates for that setup become `basis_detail: "measured_community"`; community values can only raise the estimated
needs, never lower them. Steps: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md).

## Privacy

- `recommend`, `fit`, `search`, `info`, `system`, `types`, `gate` and the Python API (`gmf.recommend()`,
  `gmf.search()`, `gmf.info()`, `gmf.system()`, `gmf.data_status()`) make **no network access**. Hardware is read
  locally (`nvidia-smi`, psutil). This is tested with every socket blocked.
- `genmodelfit update --from-release` only downloads `db.json` and `db.json.sig` from this project's GitHub release.
- `genmodelfit update` only reads public data with GET / HEAD: the ComfyUI workflow templates and releases on
  GitHub, the Hugging Face Hub API, and the model download URLs (HEAD, plus a ranged GET of the first bytes of the
  main weight files to read their header — dtype and tensor shapes; no weights are downloaded).
- `genmodelfit bench` talks only to the ComfyUI URL you give with `--url` (plus the official template files on
  GitHub when ComfyUI does not have them).
- `genmodelfit download` (and `gmf.download()`) only GETs the model URLs in the list, over https, following
  redirects only to Hugging Face / GitHub hosts; `--list` and `gmf.plan_download()` make no network access.
- Hardware information (GPU name, VRAM, RAM, CPU, disk) is read locally with `nvidia-smi` and psutil and is
  **never sent**. Measurements from `bench` are saved only on your PC; they are shared only if you open a pull
  request yourself ([CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md)).
  Saved measurements hold no host name, user name, paths or GPU UUID. Submission files (`bench --submission`,
  `contrib make`) are written locally only; `genmodelfit contrib` never touches the network.
- No telemetry. No hardware information, measurements, prompts or other user content is ever sent anywhere.
  Tests replace the PC's details with marker strings and check that no request URL, header or body of `update`,
  `update --from-release`, `bench` or `download` contains them.
  Requests carry the User-Agent `genmodelfit/<version> (+https://github.com/laxshmate/genmodelfit)`, and never the
  credentials in your `~/.netrc` (proxy and certificate settings such as `HTTPS_PROXY` are still honored).

## Accuracy and limits

See [docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md). In short:

- Estimates are calibrated on only 9 real runs on one GPU (RTX 5080 16 GB). Leave-one-out error could be
  evaluated on 4 of them (image and 3D: max 6.4%, mean 4.6%); video and audio could not be evaluated yet.
  Treat `est.` values as rough guidance.
- ComfyUI's dynamic VRAM uses more memory when it is available, so a peak measured on a large GPU is not the
  minimum requirement.
- Some templates list alternative files (e.g. 1.3B and 14B variants); genmodelfit counts one variant per part,
  which may not match the variant you pick.
- The HEAD checks in the bundled list expire after 30 days; run `genmodelfit update --from-release` (or
  `update --heads-only`) to refresh them, otherwise auto-recommendations drop to zero. `genmodelfit system` and
  `gmf.data_status()` tell whether the list in use has expired (`expired`, `expires_at`).
- v0.1 counts NVIDIA GPUs only.

## License

genmodelfit is licensed under the [Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) (MPL-2.0), and so is the bundled data
(`db.json`, measurements, calibration). MPL-2.0 is a file-level copyleft: if you distribute modified genmodelfit
files, you must make those files' source available under MPL-2.0; you may use genmodelfit unmodified (or import it)
in a larger work under any license, including closed-source apps. When you distribute genmodelfit inside your
app (for example in a PyInstaller exe), keep its license notices and tell your users where its source code is
available (https://github.com/laxshmate/genmodelfit, MPL-2.0 §3.2); the same applies to `certifi` (MPL-2.0).
Parts of the data come from [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT) and
the Hugging Face Hub API; see [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES). Model licenses are reported, not granted.

```
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```
