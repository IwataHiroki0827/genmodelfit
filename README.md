# genmodelfit

**English** · [日本語](https://github.com/laxshmate/genmodelfit/blob/main/README.ja.md)

genmodelfit tells you which image, video, music, sound-effect, speech (TTS) and 3D generation models run well on your PC.

[![PyPI](https://img.shields.io/pypi/v/genmodelfit)](https://pypi.org/project/genmodelfit/)

It is inspired by [llmfit](https://github.com/AlexsJones/llmfit), which does this for LLMs. Status: alpha
([CHANGELOG.md](https://github.com/laxshmate/genmodelfit/blob/main/CHANGELOG.md)).

![uvx genmodelfit search flux and recommend on a PC with 8 GiB VRAM and 32 GiB RAM](https://raw.githubusercontent.com/laxshmate/genmodelfit/main/docs/images/demo.gif)

`uvx genmodelfit search flux` and `uvx genmodelfit recommend` with genmodelfit 0.12.0 (model list as of 2026-10-21). The list is updated weekly, so the models and counts change.
In the animation the GPU is a simulated Ada-generation one with 8 GiB VRAM (bf16 and fp8 usable), not the PC that recorded it.
`--vram` changes only the amount of VRAM, and the GPU generation stays that of the PC it runs on.
From the RTX 30 series on (30, 40, 50), the generation does not change the estimate, so this is also how to check a GPU before
you buy it (add `--ram` for its system RAM).

## Quick start

**Try it without installing.** If you have [uv](https://docs.astral.sh/uv/getting-started/installation/), one command
is enough. uv runs genmodelfit in its own temporary environment (and fetches Python if needed); nothing is added to
your Python or PATH.

```
uvx genmodelfit recommend         # the best models for this PC, per type
uvx genmodelfit system            # what was detected on this PC
```

Every command in this README works this way: write `uvx genmodelfit` instead of `genmodelfit`.
`uvx genmodelfit@latest …` makes uv check for a newer version first.

**Install it** to use the `genmodelfit` command directly ([details](https://github.com/laxshmate/genmodelfit#install)):

```
pipx install genmodelfit          # or: uv tool install genmodelfit / pip install genmodelfit
genmodelfit recommend             # the best models for this PC, per type
python -m genmodelfit recommend   # the same, if "genmodelfit" is not found
```

**Help by measuring (optional, no git needed).** One command measures your GPU for a few minutes and sends the
results as a GitHub pull request, which helps check and improve the estimates for everyone with the same GPU
([how](https://github.com/laxshmate/genmodelfit#help-by-measuring)):

```
genmodelfit share                 # measure installed models (~5 min), show what will be sent, open a pull request
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

- Windows, Linux or macOS, Python 3.10 or newer (with `uvx`, uv provides Python if you have none).
- GPU: NVIDIA (CUDA); the estimates are calibrated on NVIDIA runs. Without a usable NVIDIA GPU, AMD (ROCm: RDNA 2 or
  newer, Windows 11 / Linux), Intel (XPU: Arc A / B, Core Ultra Arc Graphics, Windows 11 / Linux) and Apple silicon
  (MPS, macOS 14 or newer) are judged with the vendors' support tables (same level rules as NVIDIA; the PC line shows "✓ verified GPU kind"
  only for kinds of GPU with records of running on two or more GPU models, today NVIDIA with dedicated VRAM). Integrated GPUs and APUs (e.g. Radeon 780M / 680M, Steam Deck, Ryzen Z1) use shared memory. Other GPUs
  (Intel UHD / Iris Xe, the 2-CU Radeon 610M / 820M, RDNA 1 and older) and PCs without a GPU are judged as CPU-only;
  SD-family image models and Stable Audio count as running on CPU (slow). On Linux, `system` suggests
  `update-pciids` when lspci does not know the GPU's name
  ([docs/hardware.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/hardware.md)).
  Non-NVIDIA GPUs have no measurements yet, so they get no ✓. `genmodelfit measure` works on these GPUs too;
  sending the result with `genmodelfit share` verifies them and tightens the estimates
  ([Help by measuring](https://github.com/laxshmate/genmodelfit#help-by-measuring)).

### Install

Not needed to try it: `uvx genmodelfit …` runs it without installing ([Quick start](https://github.com/laxshmate/genmodelfit#quick-start)).
To keep the `genmodelfit` command:

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

Nothing, for judging: the hardware is read locally. Only `update`, `download`, `measure` and `bench` connect, and only to
public data or the ComfyUI you name (when you give no folder or URL, `download` and `measure` also ask a ComfyUI running on
this PC, at 127.0.0.1). `share` only checks a GitHub token you already have before measuring and sends nothing
until you agree (then it signs in and opens a pull request with the files it showed you, or opens the issue form). No telemetry. Details: [Privacy](https://github.com/laxshmate/genmodelfit#privacy).

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
   Without `--comfyui-dir` it looks for your ComfyUI and shows where it found it (if it finds none or several, it asks).
   ```
   genmodelfit download "Z-Image-Turbo" --list
   genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI   # your ComfyUI folder
   ```

6. **Keep the list fresh.** The list is rebuilt weekly. A model whose link check is older than 30 days stops being
   Verified. This fetches the latest signed list from this project.
   ```
   genmodelfit update --from-release
   ```

7. **Measure and send (optional).** `share` measures the models worth measuring that are installed (about 5 minutes,
   starting ComfyUI if needed), shows exactly what will be sent, and sends it as a GitHub pull request after you
   agree ([Help by measuring](https://github.com/laxshmate/genmodelfit#help-by-measuring)). `measure` measures one
   model you choose.
   ```
   genmodelfit share
   ```

### For apps (Python / JSON)

The commands take `--json` ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md)).
The same in Python:

```python
import genmodelfit as gmf

for f in gmf.recommend(modality="image", task="edit", limit=5):
    row = f.to_dict()  # the same dict as one item of `recommend --json`
    # memory_required_gb: what the chosen mode needs (the table's VRAM; RAM on CPU).
    # f.estimate.vram_gb is the all-on-GPU need, which can be larger than the GPU (GPU~ / GPU+RAM rows).
    print(f.model.id, f.model.precision, f.score, f.auto, f.estimate.fit, row["memory_required_gb"])

st = gmf.data_status()  # is the list fresh enough for Verified (auto) models?
if st.expired:
    print("run: genmodelfit update --from-release", st.expires_at)

# reading the list yourself: skip entries still waiting for review.
# on a malformed or wrongly typed list, load_db raises a RuntimeError (catch RuntimeError; the subclass
# genmodelfit.db.DatabaseError is not in gmf.__all__).
ok = [e for e in gmf.load_db().models if e.reviewed]

# one model, with the GPU your app detected (generation and precision are decided here)
fit = gmf.judge_model("image_sdxl_simple", gpu_name="NVIDIA GeForce RTX 3060", vram_gb=12, driver="560.94")
print(fit.to_dict()["fit_level"])  # only names in gmf.__all__ are stable API

# AMD / Intel GPUs go through the same support tables as detection; refusals say why (other_gpus[].reason)
pc = gmf.system_from(
    [{"name": "AMD Radeon RX 7800 XT", "vendor": "amd", "vram_gb": 16}],
    windows=True,
    windows_build=22631,
    cpu="AMD Ryzen 7 7700X",
)
gmf.gpu_family("Intel(R) Arc(TM) B580 Graphics", "intel")  # {"backend": "xpu", ...} from the name table
db = gmf.load_db(use_cache=False, user_measurements=False)  # bundled list + bundled measurements only

# fetch the signed list with your own session, e.g. the app's CA bundle (network: only this and gmf.download)
db = gmf.update_from_release(session=my_session, dest=data_dir / "db.json", state_dir=data_dir)
```

Only names in `gmf.__all__` are stable API ([docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#python-api-stable)).

### Judge another PC

`--vram` and `--ram` (GiB) replace the detected sizes, e.g. `genmodelfit recommend --vram 12 --ram 32`.
With a usable NVIDIA GPU, its generation is kept and only the VRAM changes. On a non-NVIDIA GPU (or without a usable
GPU), `--vram` judges a hypothetical NVIDIA GPU of unknown generation. Other users' measurements, and measurements
that set the level, apply only when `--vram` and the measured card's VRAM have a ratio (smaller ÷ larger) of 0.95 or more. `--width`, `--height`, `--frames`, `--length` and `--precision` judge a specific
setting.

### Language

Output is in English by default. For Japanese, add `--lang ja` to any command (e.g. `genmodelfit recommend --lang ja`),
set the environment variable `GENMODELFIT_LANG=ja`, save it with `genmodelfit config set lang ja` (command line only),
or call `gmf.set_language("ja")` from Python (the flag wins, then the variable, then the saved setting for the command
line or `set_language` for Python; the OS language is not used). Only the texts change (reasons, notes, errors, table
headings and footers, `--help`); JSON keys, ids, codes and judgments are the same in both languages. English output
contains no Japanese, and Japanese output contains no English-only sentences (one language per line).

Kept in English in both languages: the table column names (`Type`, `Status`, `Score`, `Model`, `Use`, `Prec`, `VRAM`,
`Mem%`, `Mode`, `Disk`, `License`, `Censored`), the fit level names (`Perfect`, `Good`, `Marginal`, `Too tight`),
the run-mode marks (`GPU`, `GPU~`, `GPU+RAM`, `CPU`), units, and machine-readable values such as ids, license ids and
option names. This keeps the table widths stable and matches what you type for `--fit`, `--sort` and so on; the
Japanese output explains them in Japanese below the table. Type headings show the Japanese type name followed by
the name for `-m` (e.g. `(image)`).

Long lines are wrapped counting East Asian wide characters as two columns. File names, URLs, options and numbers with
units are never split (a single one longer than the width stays whole on its own line), and the basic Japanese
line-breaking rules apply (no line starts with a closing mark or comma, or ends with an opening bracket).

## 3. Results

### Example

`genmodelfit recommend` on a PC with 8 GiB VRAM and 32 GiB RAM, top 2 per type (`COLUMNS=100`).
The examples use a simulated Ada-generation GPU; `--vram` changes only the amount of VRAM, and the GPU generation is
that of the PC it runs on. [tools/screenshots.py](https://github.com/laxshmate/genmodelfit/blob/main/tools/screenshots.py) `--readme` rebuilds these blocks:

```
$ genmodelfit recommend --vram 8 --ram 32 -n 2
PC: assumed 8.0 GiB VRAM (--vram), RAM 32.0 GiB (--ram)
Recommended: 12 of 83 (221 hidden: uncensored 26, tools 34, license unknown 28, too tight 133; 25
    more workflows merged into their models)
== image: top 2 of 32  (-m image for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Good        100 Anima Base 1.0             gen,edit  bf16   5.7  71% GPU~     5.4 other
image ● Good         92 SDXL1.0 Base               gen       fp16   6.2  78% GPU~     6.5 openrail++
== video: top 2 of 12  (-m video for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
video ● Good         96 Wan 2.2 5B TI2V            gen       fp16   6.1  76% GPU+RAM 16.9 apache-2.0
video ● Good        100 Wan 2.1 1.3B T2V           gen       fp16   4.3  54% GPU+RAM  9.2 apache-2.0
== music: top 2 of 10  (-m music for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
music ● Good         96 ACE-Step 1.5 Turbo         gen       bf16   6.0  76% GPU~     9.3 mit
music ● Good        100 YuE2 3B                    gen,edit  int8   4.3  54% GPU~     3.7 cc-by-nc-…
== sfx: top 2 of 2  (-m sfx for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
sfx   ● Good        100 stable-audio-open-1.0      gen       fp32   4.9  61% GPU~     5.3 other
sfx   ● Perfect     100 stable-audio-3-small-sfx   gen       fp32   4.7  59% GPU      3.2 other
== tts: top 2 of 20  (-m tts for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
tts   ● Perfect     100 MeloTTS-English            gen       ?      1.7  21% GPU      0.2 mit
tts   ● Perfect     100 Chatterbox                 gen       fp32   3.5  44% GPU      2.0 mit
== 3d: top 2 of 7  (-m 3d for more)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
3d    ● Good         99 TripoSplat                 gen       fp16   5.8  72% GPU      3.5 other
3d    ● Good        100 HY 3D 2.0 MV               gen       fp16   4.3  54% GPU~     4.6 other
Order: runs, Perfect/Good (GPU+RAM, CPU last), Marginal (faster first), official, popularity, Score.
Score 0-100 = memory headroom (Mem% <= 70% is 100; not speed). VRAM: GiB needed, m = measured.
Mode: GPU all on GPU, GPU~ parts swapped in turn, GPU+RAM offload (slow), CPU (slow), - no run.
Level caps: GPU~ and GPU+RAM at most Good; CPU at most Marginal (CPU-ready TTS, tools: Good).
Mem%: share of the tightest memory (VRAM; RAM for CPU, or if larger for GPU+RAM), RAM = RAM short.
Perfect = fits with room to spare, not a speed guarantee (also on low-bandwidth GPUs).
GPU+RAM: the level is often set by RAM; the offloaded share is assumed (40% of weights on GPU).
One row per model: Prec = best precision that fits, Use = workflow uses (info, --all-workflows).
Estimates without m are rough. Details: genmodelfit info <name or id>. Apps: --json (schema.md).
--json lists auto-recommendable models first (schema.md).
Note: 26 uncensored model(s) hidden; add --uncensored to show them.
Note: 28 model(s) with an unknown license hidden; add --any-license to show them.
? = unknown (precision, license, size: not judged), - = not applicable (e.g. no run).
```

The first lines show the PC and how many models were hidden and why. The last lines explain the marks.
`fit` prints one table (here 24 GiB VRAM, 64 GiB RAM, images only, first 6 rows):

```
$ genmodelfit fit --vram 24 --ram 64 -m image -n 6
PC: assumed 24.0 GiB VRAM (--vram), RAM 64.0 GiB (--ram)
All models: 6 of 60 (218 hidden: other types 166, uncensored 21, tools 19, license unknown 12; 51
    more workflows merged into their models)
Type  Status      Score Model                      Use       Prec  VRAM Mem% Mode    Disk License
image ● Perfect     100 Anima Base 1.0             gen,edit  bf16   7.0  29% GPU      5.4 other
image ● Perfect     100 SDXL1.0 Base               gen       fp16   8.0  33% GPU      6.5 openrail++
image ● Perfect     100 Anima Preview 3 Base       gen       bf16   7.0  29% GPU      5.2 other
image ● Perfect     100 NetaYume Lumina            gen       bf16  12.4  52% GPU      9.9 gemma
image ● Perfect     100 SDXL Turbo 1.0             gen       fp16   9.0  38% GPU      6.5 other
image ● Perfect     100 PixelDiT 1300M             gen       bf16   9.6  40% GPU      7.3 other
Order: Type, Status, official, popularity, Score (--sort).
Score 0-100 = memory headroom (Mem% <= 70% is 100; not speed). VRAM: GiB needed, m = measured.
…
```

### Columns

| column | meaning |
|---|---|
| Type | image, video, music, sfx, tts, 3d |
| Status | fit level (below) |
| Score | 0–100: memory headroom. 100 while the tightest memory of the mode is at most 70% used (Mem%), lower above that; not speed, quality or popularity ([rules](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.md#score)) |
| Censored | `Y` censored / `N` uncensored; only with `--uncensored` (or `--content all|uncensored`) |
| Model | one row per model (same weights); long names are cut in the middle with `…` |
| Use | `gen`, `edit`, `i2v`, … (`genmodelfit types`); the row's own use (the one matching `--task`) first, then the other uses of the model's workflows, without repeats (`gen,edit`; `gen,…` when they do not fit). The workflows: `info`; one row each: `--all-workflows` |
| Prec | the best precision that fits (fp16, bf16, fp8, int8, …) |
| VRAM | GiB needed in that mode; `m` = measured (by the maintainer, on this PC, or the median of other users' measurements of the same GPU; number of sources: `info`, JSON `sample_count`) |
| Mem% | share of the tightest memory used: VRAM on the GPU, RAM on CPU, the larger of the two for `GPU+RAM`; `RAM` = system RAM is short |
| Mode | `GPU` all on GPU, `GPU~` parts swapped in turn, `GPU+RAM` offload (slow), `CPU` (slow), `-` does not run (`GPU req`: needs a GPU, none usable here; `RAM`: not enough system RAM) |
| Disk | download size (GiB); dropped first when the width is short (then see `info` or `--json`) |
| License | license of the files (`other` = read it yourself) |

### Fit levels

The level comes from how much of the memory the mode uses (VRAM on the GPU, RAM on CPU), as in llmfit:

| level | use of memory | cap by mode |
|---|---|---|
| **Perfect** | ≤ 60% | only when all on the GPU (`GPU`); room to spare, not a speed guarantee |
| **Good** | ≤ 85% | highest level for `GPU~` and `GPU+RAM` (offloading is chosen only when it raises the level), and on the `CPU` for speech whose authors state real-time CPU use (MeloTTS, Piper, VieNeu-TTS) and one-pass tools |
| **Marginal** | ≤ 98% | highest level for generation on the `CPU` or slightly short RAM (Mem% can be low) |
| **Too tight** | > 98% | does not run (a level set by measurements can be Good even above 98%) |
| **?** | — | not judged: the download size is unknown |

There is no cap by GPU vendor; the PC line shows "✓ verified GPU kind" when that kind of GPU has records of
running on two or more GPU models ([rules](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.md#verified-gpu-kinds)).
Marks: `m` after VRAM = measured (whoever measured it); without a mark it is a formula estimate. Measurements win over the formula. Without `--width` etc. every model is judged at
the same default settings for its type, so measurements taken at other settings only shift the estimate (no mark).
Other users' measurements apply only to the same GPU (same PCI model ID, else the same normalized name, and a VRAM
ratio of 0.95 or more) and can move the needs either way. Measurements set the level only when two or more independent sources agree on
the same GPU with the same settings, and then the table's VRAM is the amount used for the level (at least weights +
fixed part; `bench` values (`torch_reserved`) get +0.5 GiB). Rules: [docs/scoring.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/scoring.md).
`?` = unknown (precision, license, size; not judged on it). `-` = not applicable (does not run, no VRAM on CPU, …).
`recommend` orders rows by: runs, Perfect/Good (without offloading first), Marginal (faster mode first), official templates, popularity (the
rank of the usage count among models of the same type and source), Score. `fit` orders by type, level, official,
popularity, Score. (`recommend --json` and `gmf.recommend()` put Verified models first after "runs":
[docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#order-of-recommend).)

### Accuracy

As of 0.12.0, the estimates are calibrated on a few runs on one NVIDIA GPU and cross-checked against two runs on
another; no measurements from other users are bundled yet (`measurements_community.json`: 0 submissions). Treat
values without `m` as rough.
ComfyUI uses more VRAM when more is free, so a peak measured on a large GPU is not the minimum.
Current counts and error figures: [docs/accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md).

## 4. Reference

Every page of the documentation, with who each one is for: [docs/README.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/README.md).

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
| `config [set KEY VALUE \| unset KEY \| path]` | settings: the ComfyUI folder for `download` and the display language ([docs/download.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.md#genmodelfit-config)) |
| `measure [name or id] [--url URL]` | measure a model on your running ComfyUI; without a name, suggests models worth measuring ([below](https://github.com/laxshmate/genmodelfit#help-by-measuring)) |
| `share [file]` | measure for a few minutes and send the results as a GitHub pull request (`--issue`: the issue form) ([below](https://github.com/laxshmate/genmodelfit#help-by-measuring)) |

For maintainers: `review` (models waiting for review), `bench` (measure one template with every option) and `contrib`
(make, validate and aggregate submissions; `contrib share` is the same as `share --issue`).

Every command and option, the environment variables, exit codes and examples are on one page: [docs/commands.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/commands.md) (the same text as each command's `--help`). Common options: `--json`, `--vram`, `--ram`,
`--uncensored` (include uncensored models; old form `--content all`), `--any-license` (include unknown licenses; old form `--include-unknown-license`), `--content uncensored` (only uncensored), `--license apache-2.0,mit`, `--comfyui-version`.

### Data sources

The list is built from the [ComfyUI workflow templates](https://github.com/Comfy-Org/workflow_templates) (MIT) and the
[Hugging Face Hub API](https://huggingface.co/docs/hub/api) (licenses traced to the upstream, sizes, sha256), plus a few
hand-added models. Like llmfit, Hugging Face models need 10,000+ all-time downloads (1,000+ for models that are clearly trending). Civitai is not used (its terms
limit automated access). Details: [docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md).

### Content marks

Models with safety filtering removed or made for adult output are marked `uncensored` and **hidden by default**.
`--uncensored` (or `--content all|uncensored`) shows them, with the `Censored` column. They are never Verified by default.
Models aimed at sexual content involving minors are excluded entirely
([docs/sources.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/sources.md)).

### Verified (auto-recommendation)

`Verified: yes` (JSON `auto`) only when **all** checks pass: every file's license is on a commercial-use allowlist,
nothing is gated, an official ComfyUI workflow without custom nodes, a supported ComfyUI version, every URL answered
an anonymous HEAD within 30 days, sha256 and size are known, content is `sfw`, and no review is pending.
Otherwise the reasons are in `held_reasons`. The `recommend` / `fit` tables do not show it; `info <name or id>`
shows each check and `gate` counts them. Details: [docs/gate.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/gate.md).

### Download

Only https URLs in the list, redirects only to Hugging Face / GitHub hosts, sha256-checked while writing, resumable
after Ctrl-C, never outside the destination. Gated files are never downloaded. Uncensored models and files with an
unknown license are downloaded only with your consent: in a terminal the consent question names them (y/N, default
No); with `--yes`, `--json` or in scripts pass `--uncensored` / `--any-license` (the plan then shows one ready command
with them added and what each means; it never adds them by itself). Parts the template does not use by default (only on muted or bypassed nodes) are not downloaded unless you pass
`--with-optional`; parts behind a switch that is off are downloaded, because ComfyUI checks them before running.
Files whose ComfyUI folder is unknown go to `other/`, which
ComfyUI does not read (the plan says so; move them before use). The place is `--comfyui-dir` (or `--output-dir`),
then `GENMODELFIT_COMFYUI_DIR`, then `genmodelfit config set comfyui-dir <folder>`, then a ComfyUI found on this PC
(the current folder, a running ComfyUI, ComfyUI Desktop / comfy-cli settings, usual places); the plan always says
which. When none or several are found, a terminal lets you choose, and scripts stop with an error instead of
downloading into a folder ComfyUI does not read. The question is `[y/N/c]` (`c` changes the place). Details and Python API: [docs/download.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/download.md).

### Maintenance

The list is rebuilt every week. New models, models whose files changed (URL, size, sha256) and riskier changes are
marked `pending` and are not Verified until a person approves them with primary-source evidence (`genmodelfit review`). Only reviewed data is signed and published
for `update --from-release`. Details: [docs/release.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/release.md).

### What changed in the list

`genmodelfit diff` compares the bundled list with the one fetched by `update --from-release` (run that first; without
it, `diff` only prints how to). `diff NEW.json` compares the bundled list with a file, `diff OLD.json NEW.json` two
files. It shows models as `name (type) — id`: added and removed models, auto-recommendation changes, and for each
changed model its license (`!` = toward the permissive side), size, precision, prompt / speech languages, added /
removed files with their license, newly uncensored, review status, and the names of any other changed fields
("other changes:"), in three sections (files / license / uncensored / runtime, then auto-recommendation with ids,
then license provenance, languages, size and the rest), most important first; the same minor change on many models is one
line with a count. Changes of popularity numbers, dates and HEAD results, old HEAD failures and the `sources` numbers
are only counted; `--all` lists them (the content of the weekly pull request). `--json`: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#diff).

### Help by measuring

Measured runs on more GPUs make the estimates better. One command, no git needed (NVIDIA, AMD, Intel or Apple
GPU; a GitHub account):

```
genmodelfit share
```

1. It finds your ComfyUI like `measure`: one running on this PC (or `--url`), else the installed one it finds
   (portable, manual / comfy-cli with a `.venv`, or ComfyUI Desktop), which it starts in the background and stops
   afterwards (`--no-start` turns that off).
2. It picks the suggestions of `measure` whose files are installed and that have no result on your GPU yet (also
   none saved on this PC) until their rough times add up to about 5 minutes (`--budget`), shows the plan, the
   measurements you have not sent yet and the total time (starting ComfyUI, measuring, and downloading as an example
   at 100 Mbps), and asks once. Models that need nodes your ComfyUI does not have are left out (update ComfyUI to
   measure them) and the next suggestions fill their place.
3. It measures them one by one. If none is installed, or time is left, it asks once whether to download the others
   (`[y/N]`, with the size and time; `--download` skips the question; downloading has the same questions as
   `download`).
4. It shows every file to send (the whole content and the privacy checks) and asks how to send it:
   **1) pull request** (sign in to GitHub: `GITHUB_TOKEN` / `GH_TOKEN`, else the saved sign-in, else GitHub's device
   login in your browser; enter the code it shows; scope `public_repo`; `share --logout` removes it), **2) issues**
   (no sign-in: the measurement form opens in your browser, one per file, and you press "Create"; a bot makes the
   pull request), or **q) not now**. A pull request comes from a fork on your account, with the files under
   `data/contrib/<GPU>/` on a branch (or added to your open one). A check runs within minutes; once a maintainer
   merges it, the next version uses it.
5. Results are saved before sending and never measured again. If you choose q or the pull request fails (a terminal
   then offers issues), `share --no-measure` (or `share --issue`) sends them later. Before measuring, only a token you
   already have is checked; with `--yes` and no terminal it needs one (`GITHUB_TOKEN`), or use `share --issue --yes`.

`share --dry-run` only shows the plan and what would be sent; `share --no-measure` sends only the measurements you
have not sent. `share --issue` without `--yes` is the older way: choose an unsent measurement and open its form.

`genmodelfit measure` measures one model you choose. It lists models that fit your GPU and have no measurement on it yet (installed ones first, then
   the shorter time, then the smaller download). Choose one; it shows what it will do (model, settings, which ComfyUI,
   whether the files are installed, a rough time, that nothing is sent) and measures on the ComfyUI running on this PC
   (or `--url`). A model that is not installed is downloaded first with the same questions as `download`.
   Once a model has a measurement on your GPU at the template's settings, the list also has runs at other settings
   that help check the estimate's formula: image resolution (512 / 768 / 1024 / 1536 px), video frames (17 / 33 /
   49 / 81 at the template's size) and music / sound length (10 / 30 / 60 / 120 s), only when they fit your GPU at
   that setting and have no measurement at it on this GPU, at most 2 per model and about 5 minutes or less
   (shown with `--resolution`, `--frames` or `--length`; `genmodelfit measure <id> --resolution 1536x1536` picks
   one yourself). Each run is one generation of a few minutes; `share` sends what it saved.

What is sent (all of it is shown before you send): the GPU name, VRAM, generation and compute capability (NVIDIA),
driver version, vendor and backend, for AMD / Intel GPUs the PCI model ID (a model number such as `1002:744c`, the
same on every card of that model), the OS family, RAM rounded to GiB, the ComfyUI and genmodelfit versions, the date
(no time), the model, settings, result, peak VRAM / RAM, time, how the model was loaded (`offload`), how it was
measured, a fingerprint of the model files, and bench's note about how it measured. What is not: host or user names,
paths, GPU UUID, PCI slot, serial numbers, IP, CPU name, exact time, prompts.

How it is used: a measurement is used for people with the same GPU. On NVIDIA with ComfyUI's dynamic VRAM (the
default today) only the working memory can be measured, so it is used to cross-check the estimates and does not set
levels yet (#100; [accuracy.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/accuracy.md#how-measurements-are-used)).
Other measurements adjust the estimate for that GPU, and when two different people's measurements agree they also
set the level. One GitHub account counts once, however many measurements it sends. Out-of-memory results are kept as
"not enough on this GPU" records, which stop measurements from setting the level for that setup. A measurement takes
effect after a maintainer accepts it and the next version of genmodelfit is released.

If you edit the issue after the bot made its pull request, the new content is checked again: the pull request is
updated (or closed, if the new content does not pass) and both pages say so. For git users, `bench --submission
data/contrib/` in a clone of your fork still works. Steps: [CONTRIBUTING.md](https://github.com/laxshmate/genmodelfit/blob/main/CONTRIBUTING.md).

### Privacy

- Judging commands and the Python API (except `update_from_release` and `download`, which you call on purpose) make **no network access** (tested with every socket blocked).
- `update` reads public data only (GitHub, Hugging Face API, HEAD on model URLs); `--from-release` only fetches
  `db.json` and its signature. `download` only fetches model files, and, when no folder is given, asks a ComfyUI
  running on this PC (127.0.0.1 ports 8188 and 8000) where its folders are. `measure` asks the same ports whether
  ComfyUI runs (without `--url`) and talks only to that ComfyUI (plus the official template on GitHub and, when you
  agree, the model files); a ComfyUI it starts listens on 127.0.0.1 only. `bench` talks only to your `--url`.
- genmodelfit never sends hardware details, measurements or prompts on its own. A measurement is shared only when
  you agree in `share` (it shows every file first): then it talks to github.com and api.github.com only, to sign in
  and to open a pull request with those files from your account
  ([Help by measuring](https://github.com/laxshmate/genmodelfit#help-by-measuring)). The sign-in token is saved in
  the config folder, readable only by you, and never printed. `share --issue` (and `bench --share`, `contrib share`)
  only open the GitHub form in your browser (or print its URL, or copy the content to your clipboard); the content
  is sent when you press "Create" there. No `~/.netrc` credentials, no telemetry.
- What `share` sends (0.13.0, #132): each measurement (GPU name, VRAM, generation, driver, backend, RAM rounded to
  GiB, OS family, ComfyUI and genmodelfit versions, model, settings, peak VRAM / RAM, time) with the estimate
  genmodelfit made for it (without your measurements or any of this model on your GPU), a time breakdown (queue, load, sampling, decode), a
  mark when the peak VRAM went over the GPU total into shared memory, and the list version; plus one file of
  failure records per run (missing node names, timeouts, error type names, never the error text). Never sent:
  generated images or sounds, your inputs, host or user names, GPU UUID, PCI location, paths, IP addresses.
  Details: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#contrib) ("contrib").
- Per-command table: [docs/schema.md](https://github.com/laxshmate/genmodelfit/blob/main/docs/schema.md#side-effects) ("Side effects").

### License

[Mozilla Public License 2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) (MPL-2.0), including the bundled
data. You may use genmodelfit unmodified in closed-source apps. If you distribute modified genmodelfit files, publish
those files under MPL-2.0. When you bundle genmodelfit in your app (e.g. a PyInstaller exe), keep its license notices
and tell users where its source is (https://github.com/laxshmate/genmodelfit, MPL-2.0 §3.2); the same applies to
`certifi` (MPL-2.0). Third-party data: [THIRD_PARTY_NOTICES](https://github.com/laxshmate/genmodelfit/blob/main/THIRD_PARTY_NOTICES).
Model licenses are reported, not granted.
