# genmodelfit

**English** · [日本語](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md)

genmodelfit tells you which image, video, music, sound-effect, speech (TTS) and 3D generation models run well on your PC.

[![PyPI](https://img.shields.io/pypi/v/genmodelfit)](https://pypi.org/project/genmodelfit/)

It is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs.
It is **not** an official llmfit project or fork. Status: alpha
([CHANGELOG.md](https://github.com/laxshmate/genmodelfit/blob/main/CHANGELOG.md)).

Quick start:

```
pipx install genmodelfit          # or: uv tool install genmodelfit / pip install genmodelfit
genmodelfit recommend             # the best models for this PC, per type
python -m genmodelfit recommend   # the same, if "genmodelfit" is not found
```

## 1. Prerequisites

### What it does

- **Judges** whether each model in its list runs on this PC, and how comfortably (from VRAM, RAM, GPU generation).
- **Recommends** the best models per type: image, video, music, sfx (sound effects), tts (speech), 3d.
- **Downloads** the files of a model, after showing licenses and asking you.
- It does **not** generate anything. To run a model, use [ComfyUI](https://github.com/comfyanonymous/ComfyUI) (or the
  runtime the model names). genmodelfit holds model **information only**, never weights.
- Why not llmfit: it covers LLMs only ([llmfit#41](https://github.com/AlexsJones/llmfit/issues/41)); here memory
  changes several times over with precision, resolution, frames and offloading.

### Requirements

- Windows, Linux or macOS, Python 3.10 or newer.
- GPU: NVIDIA (CUDA) as measured. Without a usable NVIDIA GPU, AMD (ROCm: RDNA 2 or newer, Windows 11 / Linux),
  Intel (XPU: Arc A / B, Core Ultra Arc Graphics, Windows 11 / Linux) and Apple silicon (MPS, macOS 14 or newer) are
  judged as estimates (at most Good). Other GPUs (Intel UHD / Iris Xe, older AMD) and PCs without a GPU are judged as
  CPU-only; SD-family image models and Stable Audio count as running on CPU (slow)
  ([docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md)).

### Install

```
pipx install genmodelfit        # once: pipx ensurepath, then open a new terminal
uv tool install genmodelfit     # once: uv tool update-shell, then open a new terminal
pip install genmodelfit
```

pipx and uv put `genmodelfit` on PATH. **`python -m genmodelfit …` always works**, whatever PATH says
(on Windows also `py -m genmodelfit …`).

If pip printed *"WARNING: The script genmodelfit.exe is installed in '…\Scripts' which is not on PATH"* (common with
python.org Python without "Add python.exe to PATH", Microsoft Store Python, user installs, or a venv not activated),
add that folder to PATH or use one of the above. A terminal opened before PATH changed does not see it: open a new one.

### What is sent over the network

Nothing, for judging: the hardware is read locally. Only `update`, `download` and `bench` connect, and only to public
data or the ComfyUI you name. No telemetry. Details: [Privacy](https://github.com/laxshmate/genmodelfit#privacy).

## 2. Procedure

1. **Check your PC.** Shows the GPU, VRAM, RAM, page file, free disk, and whether the model list is still fresh.
   ```
   genmodelfit system
   ```

2. **Get recommendations.** The top models of each type that run here, best first.
   Narrow down with a type (`-m`) or a use inside a type (`--task`; `genmodelfit types` lists them).
   ```
   genmodelfit recommend
   genmodelfit recommend -m image --task edit
   ```

3. **See everything that runs.** One table of all models that run here (Perfect / Good / Marginal).
   `--all` adds tools, models that do not run and unjudged ones; `--sort score` ranks across types;
   `--fit good` keeps one level.
   ```
   genmodelfit fit
   genmodelfit fit --sort score --fit good
   ```

4. **Look at one model.** Files, licenses, memory in each mode, limits, and why it is or is not Verified.
   ```
   genmodelfit info "Z-Image-Turbo"
   ```

5. **Download it.** `--list` shows the files, licenses, sizes and free disk without downloading.
   Then download into your ComfyUI folders; you are asked before anything starts, and every file is sha256-checked.
   ```
   genmodelfit download "Z-Image-Turbo" --list
   genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI   # your ComfyUI folder
   ```

6. **Keep the list fresh.** The list is rebuilt weekly. Its link checks expire after 30 days, and then nothing is
   Verified. This fetches the latest signed list from this project.
   ```
   genmodelfit update --from-release
   ```

### For apps (Python / JSON)

The commands take `--json` ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)).
The same in Python:

```python
import genmodelfit as gmf

for f in gmf.recommend(modality="image", task="edit", limit=5):
    print(f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, f.estimate.vram_gb)
    row = f.to_dict()  # the same dict as one item of `recommend --json`

st = gmf.data_status()  # is the list fresh enough for Verified (auto) models?
if st.expired:
    print("run: genmodelfit update --from-release", st.expires_at)

# reading the list yourself: skip entries still waiting for review
ok = [e for e in gmf.load_db().models if e.review is None or e.review.status == "ok"]

# one model, with the GPU your app detected (generation and precision are decided here)
fit = gmf.judge_model("image_sdxl_simple", gpu_name="NVIDIA GeForce RTX 3060", vram_gb=12, driver="560.94")
print(fit.to_dict()["fit_level"])  # only names in gmf.__all__ are stable API

# fetch the signed list (the only Python call that uses the network) with your own session, e.g. the app's CA bundle
db = gmf.update_from_release(session=my_session, dest=data_dir / "db.json", state_dir=data_dir)
```

### Judge another PC

`--vram` and `--ram` (GiB) replace the detected sizes, e.g. `genmodelfit recommend --vram 12 --ram 32`.
The GPU generation of this PC is kept. `--width`, `--height`, `--frames`, `--length` and `--precision` judge a
specific setting.

## 3. Results

### Example

`genmodelfit recommend` on a PC with 8 GiB VRAM and 32 GiB RAM, top 2 per type (`COLUMNS=100`):

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

The first lines show the PC and how many models were hidden and why. The last lines explain the marks.
`fit` prints one table (here 24 GiB VRAM, 128 GiB RAM, images only, first 6 rows):

```
$ genmodelfit fit --vram 24 --ram 128 -m image -n 6
PC: assumed 24.0 GiB VRAM (--vram), RAM 128.0 GiB (--ram)
All models: 6 of 60 (218 hidden: other types 166, uncensored 21, tools 19, license unknown 12; 51
    more workflows merged into their models)
Type  Status      Score Model                    Use     Prec  VRAM Mem% Mode    License    Verified
image ● Perfect      90 SDXL1.0 Base             gen+3   fp16  8.0m  33% GPU     openrail++ license
image ● Perfect      84 MOSS-Image SFT           gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      83 FLUX.2 [Klein] 4B Base   edit    fp8   13.2  55% GPU     apache-2.0 yes
image ● Perfect      82 MOSS-Image Base          gen+1   int8  13.8  58% GPU     apache-2.0 yes
image ● Perfect      81 Anima Base 1.0           gen+3   bf16   7.1  30% GPU     other      license
image ● Perfect      80 Anima Preview 3 Base     gen     bf16   7.0  29% GPU     other      license
Order: Type, Status, Score (--sort). Score 0-100. VRAM: GiB needed, m = measured.
…
```

### Columns

| column | meaning |
|---|---|
| Type | image, video, music, sfx, tts, 3d |
| Status | fit level (below) |
| Score | 0–100: fit level, VRAM headroom, Verified, measured or not, popularity ([formula](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)) |
| Censored | `Y` censored / `N` uncensored; only with `--content all` or `uncensored` |
| Model | one row per model (same weights); long names are cut in the middle with `…` |
| Use | `gen`, `edit`, `i2v`, … (`genmodelfit types`); `+N` = N more workflows of the same model (`--all-workflows`) |
| Prec | the best precision that fits (fp16, bf16, fp8, int8, …) |
| VRAM | GiB needed in that mode; `m` = measured on a real run |
| Mem% | share of VRAM used (of RAM on CPU); `RAM` = system RAM is short |
| Mode | `GPU` all on GPU, `GPU~` parts swapped in turn, `GPU+RAM` offload (slow), `CPU` (slow), `-` does not run (`GPU req`: needs a GPU, none usable here; `RAM`: not enough system RAM) |
| Disk | download size (GiB); dropped first when the width is short, so at 100 columns see `info` or `--json` |
| License | license of the files (`other` = read it yourself) |
| Verified | `yes` = can be recommended automatically; otherwise the first failed check (`license`, `workflow`, `review`, …) |

### Fit levels

The level comes from how much of the memory the mode uses (VRAM on the GPU, RAM on CPU), as in llmfit:

| level | use of memory | cap by mode |
|---|---|---|
| **Perfect** | ≤ 60% | only when all on the GPU (`GPU`) |
| **Good** | ≤ 85% | highest level for `GPU~` |
| **Marginal** | ≤ 98% | highest level for `GPU+RAM`, `CPU` or slightly short RAM (Mem% can be low) |
| **Too tight** | > 98% | does not run |
| **?** | — | not judged: the download size is unknown |

Marks: `m` after VRAM = measured on a real run (measured values win; without `m` it is a formula estimate).
`?` = unknown (precision, license, size; not judged on it). `-` = not applicable (does not run, no VRAM on CPU, …).
`recommend` orders rows by: runs, Verified, Perfect/Good, Marginal, Score. `fit` orders by type, level, Score.

### Accuracy

Estimates are calibrated on few real runs on one GPU (RTX 5080 16 GB); treat values without `m` as rough.
ComfyUI uses more VRAM when more is free, so a peak measured on a large GPU is not the minimum.
Details and error figures: [docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md).

## 4. Reference

### Commands

| command | what it does |
|---|---|
| `system [--all-gpus]` | this PC and the freshness of the list |
| `recommend [-m TYPE] [--task USE] [-n N] [--auto-only]` | best models per type |
| `fit [-m TYPE] [--fit LEVEL] [--sort KEY] [--all]` | every model that runs, in one table |
| `search <text>` | models whose id or name contains the text, judged for this PC |
| `types` | the six types, their uses (`--task` names) and counts |
| `info <name or id>` | files, licenses, estimate, limits, Verified checks |
| `download <name or id> [--list] [--comfyui-dir DIR]` | download the files ([Download](https://github.com/laxshmate/genmodelfit#download)) |
| `update [--from-release]` | rebuild the list from upstream, or fetch the signed list |
| `gate` | how many models are Verified, and why the rest are held |
| `diff [OLD] [NEW] [--all]` | what changed between two lists ([below](https://github.com/laxshmate/genmodelfit#what-changed-in-the-list)) |
| `review`, `bench`, `contrib` | maintenance and measurements (below) |

Each command's `--help` lists options, exit codes and examples. Common options: `--json`, `--vram`, `--ram`,
`--content`, `--include-unknown-license`, `--license apache-2.0,mit`, `--comfyui-version`.

### Data sources

The list is built from the [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT) and the
[Hugging Face Hub API](https://huggingface.co/docs/hub/api) (licenses traced to the upstream, sizes, sha256), plus a few
hand-added models. Like llmfit, Hugging Face models need 10,000+ all-time downloads. Civitai is not used (its terms
limit automated access). Details: [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md).

### Content marks

Models with safety filtering removed or made for adult output are marked `uncensored` and **hidden by default**.
`--content all` (or `uncensored`) shows them, with the `Censored` column. They are never Verified by default.
Models aimed at sexual content involving minors are excluded entirely
([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)).

### Verified (auto-recommendation)

`Verified: yes` (JSON `auto`) only when **all** checks pass: every file's license is on a commercial-use allowlist,
nothing is gated, an official ComfyUI workflow without custom nodes, a supported ComfyUI version, every URL answered
an anonymous HEAD within 30 days, sha256 and size are known, content is `sfw`, and no review is pending.
Otherwise the reasons are in `held_reasons`. Details: [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md).

### Download

Only https URLs in the list, redirects only to Hugging Face / GitHub hosts, sha256-checked while writing, resumable
after Ctrl-C, never outside the destination. Gated, uncensored and unknown-license files are not downloaded unless
allowed. Details and Python API: [docs/download.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.md).

### Maintenance

The list is rebuilt every week. New models and riskier changes are marked `pending` and are not Verified until a
person approves them with primary-source evidence (`genmodelfit review`). Only reviewed data is signed and published
for `update --from-release`. Details: [docs/release.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.md).

### What changed in the list

`genmodelfit diff` compares the bundled list with the one fetched by `update --from-release` (run that first; without
it, `diff` only prints how to). `diff NEW.json` compares the bundled list with a file, `diff OLD.json NEW.json` two
files. It shows models as `name (type) — id`: added and removed models, auto-recommendation changes, and for each
changed model its license (`!` = toward the permissive side), size, precision, prompt / speech languages, added /
removed files with their license, newly uncensored, review status, and the names of any other changed fields
(「その他の変化」), in three sections (files / license / uncensored / runtime, then auto-recommendation with ids,
then license provenance, languages, size and the rest), most important first; the same minor change on many models is one
line with a count. Changes of popularity numbers, dates and HEAD results, old HEAD failures and the `sources` numbers
are only counted; `--all` lists them (the content of the weekly pull request). `--json`: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#diff).

### Contributing measurements

Measured runs on more GPUs make the estimates better: `genmodelfit bench` on a running ComfyUI writes a submission
file, `genmodelfit contrib show` shows exactly what it holds (no host or user names, paths or GPU UUID), and you open
a pull request yourself. Steps: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md).

### Privacy

- Judging commands and the Python API (except `update_from_release`, which you call on purpose) make **no network access** (tested with every socket blocked).
- `update` reads public data only (GitHub, Hugging Face API, HEAD on model URLs); `--from-release` only fetches
  `db.json` and its signature. `download` only fetches model files. `bench` talks only to your `--url`.
- Hardware details, measurements and prompts are never sent. No `~/.netrc` credentials, no telemetry.
- Per-command table: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md) ("Side effects").

### License

[Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) (MPL-2.0), including the bundled
data. You may use genmodelfit unmodified in closed-source apps. If you distribute modified genmodelfit files, publish
those files under MPL-2.0. When you bundle genmodelfit in your app (e.g. a PyInstaller exe), keep its license notices
and tell users where its source is (https://github.com/laxshmate/genmodelfit, MPL-2.0 §3.2); the same applies to
`certifi` (MPL-2.0). Third-party data: [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES).
Model licenses are reported, not granted.
