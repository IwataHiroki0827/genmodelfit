# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).
変更の記録。書き方は Keep a Changelog にならう。

## [Unreleased]

### Changed

- **The `recommend` / `fit` / `search` tables no longer have a Verified column** (#84). It showed the gate's verdict
  for automatic recommendation (`yes`, or the first failed check such as `license` or `workflow`), which was hard to
  read and to check by hand; llmfit has no such column either. The legend line about it is gone, and the freed width
  goes back to the other columns (with the wider Use column below): at 100 columns without the Censored column (the
  default `sfw`), Model is 2 characters wider (24 → 26) and Disk is shown again; with the Censored column
  (`--content all|uncensored`), License is shown instead of being dropped (Model 31 → 27; Disk and Mem% are still
  dropped).
  `info <id>` still shows each gate check with [ok] / [NG], and `gate` still counts them.
- **The `recommend` table no longer puts Verified models first**: rows are ordered runs → Perfect/Good → Marginal →
  official first → Score → id (the legend says `Order: runs, Perfect/Good, Marginal, Score.`), and `-n` cuts this
  order, so the table may list different models than before. **`recommend --json` and `gmf.recommend()` are
  unchanged** (same order with `auto: true` first, same contents, `auto` / `held_by` / `gate` kept), so with `-n` the
  table and the JSON can list different models (docs/schema.md "Order of `recommend`"); the legend of `recommend`
  says `--json lists auto-recommendable models first (schema.md).` (not with `--auto-only`, where both orders are
  the same). `fit` and `search` never
  sorted by it. The README screenshots (`docs/images/*.png`) still show the old column; there is no tool to render
  them, so they are left unchanged.
- **The table's Use column no longer shows `+N`** (#84, owner's comment): `gen+3` ("the shown use plus N other
  workflows of the same model") was not understood. A row that merges the workflows of one model now lists their uses
  without repeats: first the row's own use (the one that matched `--task`, else the shown workflow's), then the others
  in the order of `genmodelfit types` (`gen`, `edit`, `i2v`, …), e.g. `gen,edit` (`edit,gen` with `--task edit`), or
  just `gen` when they are all the same; with `--all-workflows` it is the row's own use. The column is 9 characters
  wide (was 7, so `gen,edit` and `lipsync,…` fit); a longer list is cut after the last use that fits and ends with
  `,…` (`gen,…`), so the row's own use is always visible. The legend says
  `Use = workflow uses (info, --all-workflows)` instead of `+N = other workflows (--all-workflows)`. JSON (`use`,
  `workflows`) and `info` are unchanged.

## [0.10.0] - 2026-10-26

### Changed

- **Output is now English by default, in exactly one language** (#73). Japanese needs `--lang ja`,
  `GENMODELFIT_LANG=ja` or `gmf.set_language("ja")`; the OS language is not used. **Behavior change for apps**: an app
  that shows genmodelfit's texts to Japanese users (e.g. AI生成キット) must call `gmf.set_language("ja")` (or set
  `GENMODELFIT_LANG=ja`) to keep Japanese reasons, `notes`, gate `detail` and error messages.
- English output contains no Japanese: table section headers (`== image` instead of `== image (画像)`), `types`,
  `info` labels and descriptions (`description[lang]`, falling back to the other language), estimate reasons and
  `notes`, limits, gate `detail`, errors and warnings, `--help` of every command, and the messages of `download` /
  `update` / `bench` / `contrib` / `review` / `diff`. JSON keys, ids, codes (`fit_level`, `basis`, check ids,
  `vram_free_short`, …), scores, judgments and ordering do not change; only the language of the texts does. Free text
  stored in the list (`review.note`, `license_evidence`, measurement `source` / `note`) is not translated, and text
  that `update` writes into the list stays Japanese whatever the display language.
- **Japanese output (`--lang ja`) is now Japanese only** (#76), one language per line: the lines above and below
  the tables (`PC:`, the heading with hidden counts, the legend and notes), section headers
  (`== 画像 (image): 37 件中 上位 3 件(続きは -m image)`, keeping the name for `-m`), `system`, `info`, `types`,
  the `download` plan, `gate`, `review`, `contrib`, error messages (`エラー:`) and `--help` (descriptions, option help,
  section titles, and argparse's own texts such as `使い方:` and `-h`). Kept in English in both languages: table
  column names, fit level names (`Perfect` / `Good` / `Marginal` / `Too tight`), run-mode marks (`GPU` / `GPU~` /
  `GPU+RAM` / `CPU`), units, ids and other machine-readable values (README "Language"). JSON is unchanged. Tests
  that ran in Japanese and pinned the old mixed text now check the Japanese text (and the English text with
  `--lang en`); a new test fails on any English-only sentence in the Japanese output of the main commands and of
  every `--help`. The README screenshots (`docs/images/*.png`) still show the old mixed output; there is no tool to
  render them, so they are left unchanged for now.
- Line wrapping (tables, `info`, `download`, `diff`, `system` notes, option help) counts East Asian wide characters
  as two columns and never splits file names, URLs, options or numbers with units (one longer than the width stays
  whole on its own line, so it can be copied); Japanese lines do not start with 、。) or end with (「 (#76). English
  lines break where they did before, except that URLs, words with hyphens and numbers with units
  are no longer cut in the middle (as a long path no longer is).
- The Japanese descriptions of Stable Audio 3 Small (music), Irodori-TTS and Tsukuyomi-chan now carry what only the
  English ones said (English prompts; the ethical restrictions on the MIT license; the voice data's own terms).

### Added

- `--lang en|ja` on every command (abbreviations such as `--lan ja` work too), the environment variable
  `GENMODELFIT_LANG`, and `gmf.set_language()` / `gmf.get_language()` (#73). Order: `--lang`, `GENMODELFIT_LANG`,
  `set_language()`, then `en`. The weekly PR body stays Japanese.
- An EXAMPLES section in `genmodelfit --help` and in the `--help` of `recommend`, `fit`, `info`, `search` and
  `download`, in the display language.
- `download` explains what to do next (#75):
  - When it stops, each reason is shown once (the final `error:` line that repeated them is gone; the exit code stays
    1 and `--json` keeps `error`). When only `--content all` / `--include-unknown-license` are missing, it ends with
    "To download anyway:" and one ready-to-run command with all of them added (your other options kept), plus one
    line per flag on what you agree to. It never adds them by itself. For gated files, files it cannot fetch
    (their URLs are listed) and workflows without files it says genmodelfit cannot download the workflow and names a
    clean alternative workflow if there is one; for disk space, `--budget` and same-name files it names the lever.
    No command is shown in those cases. `--list` shows the same guidance; a blocked run writes one short `error:`
    line to stderr. The command uses the plan's workflow id and the program form you ran (`genmodelfit` or
    `python -m genmodelfit`), with single-quoted arguments where needed.
  - Files whose ComfyUI folder is unknown go to `other/` as before, and the plan now says clearly (in every layout)
    that ComfyUI does not read `models/other` and that they must be moved before use. For a main file without a
    folder in the list, the folder is inferred when its header parameter count matches a file of an official template
    in exactly one folder (e.g. SDXL all-in-one files → `checkpoints`); the plan shows the basis, and says when a GGUF
    file needs the ComfyUI-GGUF custom nodes. In the bundled list this places the main file of 12 of the 48 Hugging
    Face models in `checkpoints/` or `diffusion_models/` instead of `other/`: a copy downloaded with 0.9.2 into
    `other/` is downloaded again (move it to the new folder first to avoid that).
  - A file with no license URL says "no license stated" and shows the page to check (the model's Hugging Face page or
    the page in the list), and the consent question no longer refers to URLs that are not shown.
  - The environment variable `GENMODELFIT_COMFYUI_DIR` sets the default for `--comfyui-dir` (`--comfyui-dir` and
    `--output-dir` win over it; checked like the flag). Without either, the plan says the default folder is not
    ComfyUI's and shows how to set it (one hint for this OS, merged with the extra_model_paths.yaml note).
  - `download --help`: examples for `--list` first, then downloading into ComfyUI, and
    `--content all --include-unknown-license`.
  - JSON (optional keys): `plan.blocked_codes`, `alternatives[].ok`, and per file `directory_basis` (an
    object) and `check_url` (https pages only).
  - In a terminal, `download` no longer needs consent flags: uncensored models and files with no stated license are
    shown in the plan and named in the consent question (y/N, default No; one `y` covers them). Without a terminal,
    with `--yes` or `--json` the flags are still required. `--list` shows what will be asked. The Python API is
    unchanged (pass `content=` / `include_unknown_license=`). Gated files are still never downloaded.
- Shorter flags (#75): `--uncensored` (= `--content all`) on `recommend`, `fit`, `search`, `info`, `types` and
  `download`, and `--any-license` (= `--include-unknown-license`) on `recommend`, `fit`, `search` and `download`.
  The old forms still work; `--content uncensored` still means "only uncensored"; `--uncensored` with
  `--content sfw|uncensored` is a usage error. Consent flags (`--uncensored`, `--any-license`,
  `--include-unknown-license`, `--content`) must be written in full: abbreviations such as `--u` are a usage
  error. Hints, notes and `--help` examples now suggest the short forms. JSON is unchanged. The library's
  `blocked` / not-found texts name the Python arguments (`content="all"`, `include_unknown_license=True`); the CLI
  shows its own flags.

### Fixed

- Read-only commands (`system` / `recommend` / `fit` / `info` / `search` / `types`) and `import genmodelfit` no longer
  import `requests`. It is now imported only by the functions that use the network, so a mismatched urllib3 / chardet
  in the user's environment no longer prints `RequestsDependencyWarning` on every run, and start-up is about 60 ms faster
  (#74).
- `download`: when two downloads of the same file run at once, the one that gets the lock after the other has
  finished now checks the placed file (sha256) and skips it, instead of downloading it again, refusing to overwrite
  and leaving a `.part` behind. A download refused because a file appeared at the destination no longer leaves its
  `.part` either (found by a rare failure of the two-thread test in the local CI).

## [0.9.2] - 2026-10-24

### Fixed

- The `--help` of `system` / `recommend` / `fit` / `search` / `info` no longer says the GPU is detected only through nvidia-smi
  (AMD / Intel / Apple are detected since 0.7.0).
- The reason shown when another user's measurement is above the GPU's total now gives the real cap
  (total × 1.02 + 0.25 GB) instead of "the total".

### Documentation

- README, CONTRIBUTING and docs/ brought up to date with 0.9.0 / 0.9.1: other users' measurements apply only to
  the same GPU with VRAM within 3% (so adding VRAM can drop a measured value; noted as the exception to "more VRAM
  never lowers the score"); `gmf.download` is the other Python call that uses the network; PCI model IDs are kept
  in the output; submission checks and fields; `files_changed`; strict type checks; local CI (`tools/ci_local.py`)
  and the weekly update on the maintainer's PC. Counts and dates that change every week now say where to read them.
- README: a screenshot of `genmodelfit recommend` (8 GiB VRAM, 32 GiB RAM; model list as of 2026-10-22), and a
  quick start that shows `uvx genmodelfit recommend` (try it without installing) before the install step.
- Every page in docs/ and CONTRIBUTING now has an English version (`X.md`) and a Japanese version (`X.ja.md`) with
  the same content, instead of pages that were Japanese only or mixed both languages. Each page starts with what it
  covers and who it is for; the longer pages have a table of contents. New index: docs/README.md.
  `tests/test_docs.py` checks that both versions exist, that links and anchors resolve, and that sections cited from
  the code exist.
- The project's "Documentation" link on PyPI now points to the docs index.

## [0.9.1] - 2026-10-22

### Fixed

- **A list with a wrongly typed field is refused when it is read** (#63). `repo_id`, `license_from`, `license_path`,
  `sampler_stages`, `comfy_family` and the other optional fields of an entry, a file and a template (`source`,
  `license_url`, `base_model`, `content_evidence`, `discovered_from`, `updated`, `homepage`, `likes`, `license_evidence`,
  `comfy_vae`, `workflow_precision`, `precision_source`; file: `license_url`, `repo_license`, `license_evidence`,
  `head_status`, `head_checked_at`, `header_precision`, `header_params`, `repo_precision`, `repo_precision_source`;
  template: all fields) are type-checked like `license_name` was in 0.8.0. All 27 distinct versions of the bundled list in the git history (`git log --all -- src/genmodelfit/data/db.json`; the 41 commits that touched `data/` include unchanged copies) still load. Top-level `generated_at` / `sources` are checked too.
- **Other users' measurements are used only for a GPU with a close VRAM.** The same name (or PCI ID) can be a different card:
  RTX 4060 Ti 8 / 16 GB, RTX 3060 8 / 12 GB, RX 7900 XT / XTX / GRE (all `744c`; the 0.9.0 review's N2). A measurement now
  matches only when `min/max` of its VRAM and the PC's total is at least 0.97 (`estimate.same_vram`; not applied to shared /
  unified memory). The fit_change check also tries the nominal VRAMs of NVIDIA GPUs with several sizes
  (`estimate.NVIDIA_VRAM_VARIANTS`, from NVIDIA's product pages). The bundled `measurements_community.json` is empty, so
  bundled judgements do not change. `--vram` that differs from the measured card's VRAM no longer uses that measurement.

## [0.9.0] - 2026-10-22

### Changed — judgements that change

Only the PCs whose GPU #56 now recognizes change with the bundled data. These PCs go from CPU to GPU: Steam Deck
(Van Gogh / Sephiroth), Ryzen Z1 Extreme / Z1, Radeon RX 6300 / Pro W6300 / W6400, Linux PCs whose old pci.ids shows
`AMD Device …` / `Intel Device …` for a known ID, and Linux Intel Arc laptop GPUs (A770M / A730M / A550M / A370M / A350M).
Checked against 0.8.0 over every bundled model × a set of 36 PCs (NVIDIA 4 / 6 / 8 / 12 / 16 / 24 / 48 GB × RAM
16 / 32 / 64, RX 7900 XTX / 7600 / 6800 XT, 780M, Ryzen 9 7945HX, Arc A770 / B580, M2 Pro, CPU 16 / 32 GB, and five
#56 PCs): rows changed only on the five #56 PCs (rows / levels changed: Steam Deck 190 / 82, Ryzen Z1 Extreme 127 / 125,
RX 6300 2 GB 133 / 4, Linux `AMD Device 7551` 291 / 291, Linux Arc A770M 271 / 271); the 31 other PCs are identical.
PCs outside that set were not compared. The bundled RTX 3070
`torch_reserved` measurements change no level (the estimate was already Perfect where they apply), and the bundled
`measurements_community.json` is empty, so the rules for other users' measurements below change nothing today.

- **Non-NVIDIA GPUs that were judged too cautiously** (#56). Sources are in docs/hardware.md「NVIDIA 以外の GPU」.
  - Steam Deck (Van Gogh, gfx1033; also the OLED model's Sephiroth and the Windows driver name `AMD Custom GPU 0405`)
    was CPU as a "2 CU" display-only GPU. It has 8 RDNA 2 CUs (Valve's spec) and TheRock lists gfx1033 as Release
    Ready on Linux and Windows, so it is judged like the 680M / 660M (shared memory, at most Good). Only the 2 CU
    gfx1036 (610M, Raphael, Granite Ridge, Dragon Range) stays CPU.
  - Ryzen Z1 Extreme / Z1 (handhelds, 12 / 4 RDNA 3 CUs per AMD's pages) are judged as gfx1103 APUs like the
    780M / 740M instead of CPU.
  - Dragon Range 7940HX / 7840HX were CPU with the reason "gfx unknown"; they are now named as the 2 CU Radeon 610M
    (still CPU), like the 7945HX.
  - The desktop Radeon RX 6300 (and Pro W6300 / W6400) is Navi 24 (gfx1034).
  - Linux with an old pci.ids: lspci's `Device [1002:7551]` is kept as `AMD Device 7551` and looked up in a PCI ID
    table copied from pci.ids 2026.10.22 (AMD and Intel GPUs that matter for judging); the reason says it was
    resolved from the ID. Unknown IDs get the `update-pciids` hint for Intel too (`Intel Device e20b` used to get none).
  - Linux Intel Arc laptop GPUs (A770M 16 GB, A730M 12 GB, A550M 8 GB, A370M / A350M 4 GB, from Intel's spec table)
    use the nominal VRAM, marked as nominal in the reason, because the xe / i915 drivers expose no VRAM size in sysfs.
    They were CPU before.
- **Other users' measurements (#34) are used like llmfit's** (owner's decision, 2026-10-22), **only on the same GPU**.
  A community measurement moves the needs and the level only of a PC whose GPU is the measured one (same PCI model ID
  when both sides have one, else the same normalized name; generic names such as `AMD Radeon(TM) Graphics` never
  match). PCs with another GPU, or none, are judged exactly as without it. The maintainer's bundled measurements are
  still used on every PC (bundled judgements unchanged). `consistent` groups still enter the coefficient calibration.
  On the same GPU:
  - an `ok` group is used from **one** submission (median of the group; `n_used` / `n_oom` kept). A group with `ok`
    and `oom` runs uses the `ok` median and reports the OOM count;
  - values move the needs **both ways** and `measured_community` scores like `measured`. Physical bounds: the all-in
    value never goes below weights + fixed overhead (a smaller value becomes that floor; `basis` is then
    `estimated`), never above the measuring GPU's VRAM × 1.02 + 0.25 GiB, and a value pinned at the GPU limit is a
    lower bound. `peak_ram_gb` is capped at the submission's RAM × 1.05 + 1 GiB. Spread to other settings
    (`calibrated`), measured ÷ formula is clamped to [0.6, 1.4] (`estimate.COMMUNITY_CALIBRATION_RANGE`; llmfit uses
    [0.05, 3.0]). Removed: "never lower the needs", the +15% cap (`COMMUNITY_RAISE_CAP`), ignoring rows that change the
    level, and the 2-consistent-submissions rule;
  - a successful measurement at the same settings can set the level (#26) and lifts the AMD / Intel / Apple
    "at most Good" cap. Perfect needs a peak within 60% of this VRAM (for `torch_reserved`, after adding the context,
    next item), not pinned at the GPU limit and no OOM report in the group.
- **`torch_reserved` measurements can now make Perfect** (#26 levels from measurements). Every `bench` result and
  submission is `torch_reserved` (the ComfyUI process's torch reservation, without the CUDA / HIP context), and the
  old rule "torch_reserved never makes Perfect" kept AMD / Intel / Apple GPUs at Good even with a measurement of that
  GPU. The level bands (60 / 85 / 98%) and the fit check now use the peak plus `estimate.TORCH_RESERVED_CONTEXT_GB` =
  0.5 GiB (the upper end of bench's "about 0.3–0.5 GB" context; the cautious side). Needs (`vram_gb`) and calibration
  are unchanged.
- **Same-GPU matching** (`estimate.gpu_key`, `estimate.same_gpu`): one normalization for both the PC's GPU name and the
  measured one — case, spaces and symbols, `(R)` / `(TM)` / `™` / `®`, and the words "NVIDIA" / "GeForce" (the same
  card is "NVIDIA GeForce RTX 3070" in nvidia-smi and PyTorch but "GeForce RTX 3070" in lspci; NVIDIA product names do
  not collide without them; "AMD" / "Intel" are kept so `AMD Device 744c` ≠ `Intel Device 744c`). New `Gpu.pci_id` /
  `Adapter.pci_id` (PCI model ID `1002:744c` from Windows WMI `PNPDeviceID` or Linux `lspci -nn`; never the bus
  address) and `Measurement.gpu_pci_id`: when both sides have an ID, the ID decides.
- **The vendor of a measurement comes from its recorded backend** (`gpu_backend` / the submission's
  `system.gpu.backend`) and only falls back to the GPU name when missing. A name `mps` is Apple.
- **`bench` on Apple names the GPU by this Mac's chip** (`Apple M2 Pro`, read with system_profiler) instead of
  ComfyUI's `mps`, when ComfyUI runs on the same Mac. Only a plain chip name (`hardware.APPLE_CHIP_NAME`) is used,
  never serials or other text; otherwise, and for a remote ComfyUI, the name stays `mps` (with a warning) and counts
  only as Apple, never as the same GPU.

### Changed — reviews and the weekly list

- **`update` marks an entry `pending` with `files_changed` when its files changed**: files added / removed, a `url`
  changed (the query string counts; only a `#fragment` is ignored), a `repo_id` changed, a `size_bytes` changed **in
  either direction** or became known / unknown, a `sha256` that had a value changed or went empty. Apps download the
  listed URL as is, so an approved entry no longer gets different files silently. A `sha256` filled in for the first
  time does not count. `diff`'s classification agrees for every single-file change except that first `sha256` and a
  pure reordering (tests/test_review_42.py). Over the bundled lists v0.6.0 → v0.7.0 → v0.8.0 it marks 1 entry
  (stable-audio-3-small-sfx), 0 in the next step.
- **`diff` / `merge-check`: a file that only grew is no longer safe** (`files_changed`; a smaller file gets
  `need_decreased` and `files_changed`). A bigger file at the same URL is different content. Weeks where a Hugging
  Face file was overwritten with a bigger one now go to the owner instead of being merged by the weekly job
  (docs/release.md). No such change happened between v0.6.0, v0.7.0 and v0.8.0.

### Changed — submissions (`contrib`)

- `contrib validate` (and the Contrib workflow) fails only on mechanical checks, like llmfit. `deviation`,
  `below_weights`, `fit_change`, `unknown_model`, `unknown_gpu` are **notices** in the job summary for the maintainer
  who merges; `raise_too_large` is gone.
- The `fit_change` notice now checks PCs with the submitted GPU (name, PCI model ID, vendor; VRAM = the submission's
  `vram_gb`; RAM 16 / 32 / 64), since other GPUs are no longer affected. It shows a level going up on that GPU and the
  AMD / Intel / Apple Good cap being lifted. `contrib aggregate --report`'s fit-level table gains a GPU column.
- New errors: `system.gpu.vram_gb` above 256 GiB (was 512); `peak_ram_gb` above `ram_total_gb` × 1.05 + 1 GiB (now also
  capped when read); any NaN / infinity / negative number; a GPU name of another vendor than `system.gpu.vendor`
  (e.g. an NVIDIA name with `amd`); for `apple`, a name other than a chip name or `mps`; a malformed `pci_id`, one
  whose vendor ID does not match, or one of `hardware.PCI_DEVICE_NAMES` whose GPU differs from the name (AMD: gfx;
  Intel: generation and nominal VRAM; an ID outside that table is an `unknown_gpu` notice). Files with `NaN` / `Infinity` are refused when read, and submissions, community rows
  and bench results are written with `allow_nan=False`.
- GPU names may contain `(R)`, `(TM)`, `™`, `®` and a size such as `(16GB)` (as ComfyUI, PyTorch and Windows print
  them, e.g. `Intel(R) Arc(TM) 140V GPU (16GB)`).
- For GPUs whose nominal VRAM the name table knows (Intel Arc: `hardware._INTEL_XPU`), the `fit_change` check also
  uses a PC with that VRAM (A770 comes as 8 and 16 GB under one name). NVIDIA has no such table yet.
- Submissions carry the optional `system.gpu.pci_id` (AMD / Intel, written by bench when ComfyUI runs on this PC;
  `contrib_schema` stays 1, old files stay valid). `contrib show` lists it; CONTRIBUTING.md lists it under what is
  sent. Community rows are grouped by it when present and keep `gpu_pci_id`.
- The weekly workflow writes "提出 0 件" instead of a failure while `data/contrib` does not exist.

### Compatibility

- `contrib`: `Report.review` / `Report.accepted()` are gone (use `Report.notices`; `Report.ok` is the pass / fail);
  `--allow-review`, `--reviewed` and `data/contrib/REVIEWED` are removed; the JSON count `review` is now `notices`.
  `Measurement.approved` and `Submission.approved` are removed (old files with `approved` still read; the key is
  ignored).
- `estimate.pick_measurement(..., gpu=)` takes the PC's `Gpu` instead of a GPU name.
- `contrib.COMMON_VRAM_GB` is removed (the check PCs use the submitted GPU).
- `xpu-smi` is called with `-d -1` (all GPUs). With two or more Intel GPUs it cannot tell which one ComfyUI uses
  (it reports no total), so bench treats it as unreadable (queue and memory checks only) instead of combining them.

### Changed — bench and other

- **`bench` works on AMD, Intel and Apple GPUs.** The busy check (average utilization ≥ 30% stops) and the GPU-wide
  delta in `note` use `amd-smi` → `rocm-smi` (AMD, Linux), `xpu-smi` (Intel, Linux), the Windows performance counters
  through `typeperf` (AMD / Intel, Windows; checked against nvidia-smi on this PC) and `ioreg` (Apple; no sudo)
  (new `gpustat.py`, stdlib only). When no tool can be read, bench no longer stops: it checks the ComfyUI queue and
  that other apps hold at most half of the GPU memory, then measures with a warning and a note; `standard` stays true
  (docs/accuracy.md). A remote ComfyUI still needs `--force`.
- Measurements record `gpu_backend` (`cuda` / `rocm` / `xpu` / `mps`), and submissions carry the optional
  `system.gpu.vendor` / `backend` (both or neither). `contrib show` lists them. Non-NVIDIA submissions no longer put a
  non-NVIDIA generation into `arch`. `contrib aggregate` keeps `gpu_backend` on community rows.
- The table legend says non-NVIDIA GPUs are at most Good until measured and points to CONTRIBUTING.md; the README
  says the same next to the GPU requirements. The legend's first line fits in 100 columns again (`Verified=yes` →
  `Verified`).

### Added

- `Estimate.sample_count` (optional JSON key, like llmfit): measurements behind the value (community: `n_used`;
  maintainer / this PC: 1; null when estimated). `Measurement.n_used` / `n_oom`. Reasons say
  「ほかの利用者の実測 3 件(中央値 7.02 GB)」, `info` prints `measured_community n=3`, and the table marks community
  values with `c` (legend: `VRAM: c = measured by other users`).
- `Gpu.pci_id`, `Adapter.pci_id`, `Measurement.gpu_pci_id` (optional JSON keys; `schema_version` unchanged).

## [0.8.0] - 2026-10-22

### Changed

- **`diff` is usable by users and no longer drops changes silently** (#59). With no arguments it compares the bundled
  list with the list fetched by `update --from-release` (without one it prints how to and exits 0); with one argument
  the bundled list with that file; with two, the two files as before. The header names both lists. Models are shown as
  `name (type) — id`. The default output lists only user-relevant changes: added / removed, auto-recommendation, and
  per changed model the license (`!` = toward the permissive side), size, precision, prompt / speech languages (before
  → after), added / removed files with their license, newly uncensored, review status, and 「その他の変化」 with the names
  of any other changed field. Only new HEAD failures are listed; old HEAD failures, 401 repos, `sources` numbers and
  popularity / date-only changes are counted and listed with `--all`. 0.6.0 → 0.7.0 used to show only a size change;
  it now shows the three music language changes and the T5Gemma text encoder (Gemma license) of
  stable-audio-3-small-sfx.
- Changed models are listed by importance (files, then license, uncensored, runtime / use, auto-recommendation,
  languages / size / precision, then review marks and other fields), so `--limit` never hides a more important change.
  The same low-importance change on many models is one line with a count (`確認の印が付いた(ok): 329 件`; ids with
  `--all`). The `…ほか N 件` line points to `-n N` / `--all`; `--all` has no line limit unless `-n` is given.
  0.5.0 → 0.7.0 and 0.4.0 → 0.7.0 now show the T5Gemma (Gemma license) file first instead of after 300 review marks.
- `diff` and `weekly merge-check` find changes with one function (`diff.entry_changes`), so they cannot disagree, and
  the CLI compares the raw JSON like merge-check (unknown keys of entries, files and the top level are shown).
  Files are paired in tiers (same name + url + sha256 + directory + repo, then directory + name + sha256, then
  directory + name + url, then by order), so same-named files are not mixed up; added / removed files show their
  source (repo or URL). A changed url / sha256 / repo / directory shows the old → new value. Review marks read
  「確認の印が付いた(ok)」 / 「確認の印が外れた」; a missing vs empty language list is not shown; the type is not repeated
  when the name already ends with it. With no arguments, a warning is shown when the fetched list is older than the
  bundled one; without a fetched list, `--json` has the usual keys (empty) plus `note`.
- The loader rejects a non-string `license_name` (entry or file), which used to crash `diff` with AttributeError.
- Second review of `diff` (#59): before → after values are shortened around the part that differs (a URL changed only in
  the middle, or two sha256 that share a prefix, never look the same); the license provenance (`license_url`,
  `license_from`, `license_path`, `repo_license`) is shown with the license, not as 「その他の変化」; collapsed lines
  of auto-recommendation changes always list the ids (cut by `-n` with 「…ほか N 件(-n M で増やす)」); 「その他の変化」 is
  collapsed per field; a file key that is missing on one side and null on the other is no change; model ids are never
  broken by wrapping (a long id gets its own line); the "older list" warning is shown for every argument form; the
  no-cache `--json` has the usual `summary` keys with 0.
- Changed models are shown in three sections, each cut by `-n` separately (owner's decision, #59): files, license
  values (with 「! 寛容な側」), uncensored, runtime / use; then auto-recommendation changes (ids always listed, so
  0.1.0 → 0.7.1 shows all 16 newly auto-recommended ids with the default `-n 20`); then license provenance
  (not collapsed), languages, size, precision, review marks and other fields.

### Added

- `diff --json`: `old_label`, `new_label`, `field_changes`, `model_changes`, `top_changes`, `warnings` (docs/schema.md「diff」). Existing keys are
  unchanged, and the weekly pull request body is the same.
- Tests for `diff` with the real 0.6.0 / 0.7.0 lists (trimmed by `tools/make_diff_fixture.py`), a test that changes
  every field of `ModelEntry` / `ModelFile` / `Template` / `Review` one at a time and requires it in the output, the
  argument modes, 100-column width and JSON compatibility.

## [0.7.1] - 2026-10-21

JSON changes are additive; `schema_version` stays 1.

### Added

- **`gmf.update_from_release(..., session=None, state_dir=None)`** (Issue #58) is public API (`update_from_release` and
  `RemoteError` are in `__all__`). `session` is a caller's `requests.Session` (e.g. with the app's CA bundle) used for
  every GET of the call; `~/.netrc` is still never sent (a no-op `auth` per request, redirects followed by genmodelfit)
  and the fixed User-Agent is sent per request, without changing the session. `state_dir` is where
  `release_state.json` lives (default: the cache folder). Apps no longer need to patch `remote._http_get` or set
  `GENMODELFIT_CACHE_DIR`.
  The session's `verify` / `proxies` win over `REQUESTS_CA_BUNDLE` and proxy variables; an empty `session.auth` counts as none.
  Redirects are followed up to 10 times; an invalid redirect URL is a `RemoteError`.
- **`gpu_memory_required_gb`** on `Fit.to_dict()` (Issue #58): the VRAM needed to run on this PC's GPU, also when
  `run_mode` is `cpu` (where `memory_required_gb` is RAM). Same as `memory_required_gb` for GPU run modes; null without a GPU.

## [0.7.0] - 2026-10-21

Targets 0.7.0 (`_version.py` is 0.7.0). JSON changes are additive; `schema_version` stays 1.

### Added

- **AMD / Intel / Apple GPUs are judged** (Issues #22, #23) when no NVIDIA GPU is usable, from a support table built
  from primary sources (docs/hardware.md「NVIDIA 以外の GPU」): AMD ROCm for RDNA 2 / 3 / 3.5 / 4 on Windows 11 and Linux
  (and Instinct on Linux), Intel XPU for Arc A / B and Core Ultra Arc Graphics on Windows 11 and Linux, Apple MPS on
  macOS 14+. They are estimates: reasons say `推定(<vendor>・実測なし)` and levels are capped at Good. Integrated GPUs
  and APUs use dedicated VRAM + GTT (or half of RAM), Apple silicon 3/4 of unified memory; they never offload.
  Unsupported GPUs (Intel UHD / Iris Xe, RDNA 1 and older, 2-CU display iGPUs, Windows 10) are judged as CPU with a
  reason. NVIDIA judging is unchanged. macOS detection uses `system_profiler`; Linux reads amdgpu GTT from sysfs.
  NVIDIA 以外の GPU も、一次資料の対応表で動く組み合わせなら推定で判定する(段階は good まで)。
- JSON (additive): `Gpu.backend` gains `rocm` / `xpu` / `mps`; `Gpu.memory` (`dedicated` / `shared` / `unified`),
  `Gpu.gfx_target`, `Gpu.carveout_gb`; `Adapter.integrated`, `usable`, `reason`, `shared_gb`, `needs` and vendor `apple`;
  `Estimate.fit_from` (`estimate` / `measurement`) and `measured_peak_vram_gb`.
- **SD family and Stable Audio run on CPU** (Issue #39): SD 1.5 / 2.x / SDXL image models and Stable Audio Open 1.0 /
  3 Small are Marginal (`CPU`) without a GPU when RAM suffices, counting ComfyUI's fp32 weights and working memory
  on CPU (`families.CPU_FAMILIES`). Running on CPU is itself an estimate (no CPU measurements; reasons say
  `CPU は推定・実測なし`). SUPIR and Stable Audio 3 Medium are left out (not checked).
- Tables: without a usable GPU, the PC line names the detected GPUs and how they are treated (`PC: Intel(R) UHD
  Graphics 770 (integrated; not usable for generation → judged as CPU), RAM …`) and a second line says `This PC has no
  usable GPU: only models that run on CPU are shown (N tools hidden: --include-tools)`. The Mode column shows
  `GPU req` (needs a GPU) and `RAM` (RAM short) for rows that do not run; JSON `run_mode_label` is unchanged.
- Uncensored name words `heretic` and `unfiltered` (`update`: name detection and Hugging Face search words).
- Public functions to judge one model with a GPU the app detected itself (Issue #27): `gmf.judge_model(model, *,
  gpu_name, vram_gb, driver, ram_gb, cpu_only, gpu_usable, windows, settings, content, system_info, db,
  comfyui_version) -> Fit` and `gmf.system_from(gpus, *, ram_gb, gpu_usable, windows) -> SystemInfo`; genmodelfit decides
  the generation and the ComfyUI precision from the GPU name and driver. `vram_gb` / `ram_gb` must be finite and
  not negative, and `cpu_only=True` refuses GPU arguments (`ValueError`). `gmf.ModelNotFound` is exported. Only the names
  in `genmodelfit.__all__` are the stable API (docs/schema.md "Python API (stable)", which also notes that the function
  `genmodelfit.recommend` hides the submodule of the same name). With `gpu_usable=True, windows=True` the result equals
  what apps computed with the internal `recommend.judge` / `vram_need` and `hardware.classify_gpu` / `comfy_precision`.
  アプリが内部の関数を使わずに 1 モデルを判定できる公開の関数。
- Hand-added components (Issue #40): `components` in `data/manual.json` adds files the Hugging Face collection does not
  pick up to an entry, with the license taken from the per-file source table (`license_sources.py`; unknown without a
  row) and `manual_component: …` evidence; every file needs a unique https `url` (`ValueError` otherwise). First: `hf:stabilityai/stable-audio-3-small-sfx` now lists the T5Gemma text
  encoder (`Comfy-Org/stable-audio-3` `text_encoders/t5gemma_b_b_ul2.safetensors`, `gemma`), so its `licenses` are
  `other` (stable-audio-community) and `gemma`, and the size / VRAM estimate includes the 1.11 GiB encoder. The id is
  unchanged. 効果音の Stable Audio 3 に、Gemma 利用規約の T5Gemma の部品を足した。

### Changed

- **A successful measurement on the same GPU sets the level** (Issue #26): with a successful standard measurement at
  the same settings on a GPU of the same vendor whose VRAM is at most this PC's (97% tolerance) and
  whose measured peak fits in this PC's VRAM, the level is at least Good and the run
  mode GPU (Perfect only when the peak is within 60% and not pinned at the GPU limit). Wan 2.2 5B on an RTX 5080 16 GB
  is now Good / GPU (was Marginal / offload with `basis: measured`). Reasons say whether the measurement or the
  estimate decided the level.
  同じ GPU で成功した実測があれば、段階は実測で決める。
- `system`: non-NVIDIA GPUs show their backend and memory kind; display adapters show whether they are usable.
- Music `languages` (Issue #41): a music card's `language` usually lists lyric languages, so music entries now take at
  most `["en"]` for `prompt` from the card and no `speech` (the hand table still sets them). ACE-Step v1
  (`audio_ace_step_1_t2a_instrumentals` / `_song`, 17 languages before) and YuE (`zh`, `en` before) are now `["en"]`.
  The song workflow `audio_ace_step_1_t2a_song` gets `speech` from the hand table: the 10 well-performing lyric
  languages of the ACE-Step README (including `ja`).
  音楽のカードの言語(歌詞)を指示の言語にしない。
- **Read gate `checks` by `id`, not by position**: the number of checks depends on the model (10, or 8 for models whose
  `runtime` is neither `comfyui` nor null: `app_runtime` replaces the three ComfyUI checks; 9 → 7 in 0.5.0, docs/gate.md).
  `checks` の数はモデルで変わるので、並びの位置でなく id で読む。

### Fixed

- `download` no longer refuses a file inside the destination as "outside" when another download of the same file is
  removing its `.lock` / `.part` at that moment (Windows: `Path.resolve()` of a file pending deletion returned the
  `\\?\` extended path). The last path component is no longer resolved unless it is a link; parents still are
  (Issue #41; `test_two_threads_same_destination` failed about once in 100 runs under parallel load and passed
  1,200 of 1,200 after the fix).
  同じファイルを同時に落とすと、負けた側が「置き場所の外」と誤って断ることがあった。

## [0.6.0] - 2026-10-21

### Added

- Review marks (Issue #42): optional `ModelEntry.review` `{status: ok|pending, reasons, decided_by, decided_at,
  evidence, note}`. `update` compares with the list it carries over from and marks `pending` new entries (`new`),
  licenses that moved toward permissive (`license_permissive`), lower needs at reference PCs (`need_decreased`) and newly
  uncensored entries (`uncensored_new`); otherwise the previous mark is carried over. `--heads-only` / `--fields-only` /
  `--inspect-only` apply the same rule. A missing `review` is treated as ok, so older lists still load and apps can
  filter with `entry.review is None or entry.review.status == "ok"` (`entry.reviewed`). The bundled list starts all ok
  (`decided_by: baseline-2026-10-21`). `schema_version` stays 1.
  確認の印。新しい項目などは確かめるまで自動のおすすめに出さない。
- Gate check `reviewed` (always last; readiness weight 0): pending models are not auto-recommended
  (`held:reviewed`, `review` in the Verified column).
- `genmodelfit review list [--json]`, `review approve <id> --evidence URL… [--note] [--by] --db PATH` (refuses without
  an https evidence URL), `review reject <id> --note TEXT --db PATH [--hide-name]` (removes the entry and records it in
  `data/review_rejected.txt`, which `update` honors). Python: `gmf.review_list()`, `gmf.review_approve()`,
  `gmf.review_reject()`.
- `diff` adds `classification` `{safe, judgement_needed: [{id, name, reasons}], safe_changes}` for the weekly run
  (safe-only week → the runner may merge; otherwise an evidence pull request for the owner). It is an allowlist: only
  popularity counts, `updated`, template `usage`, HEAD results, growing files, stricter licenses, removed entries,
  auto-recommendation lost (or regained by HEAD with the same files), `sources` statistics and unchanged pending
  marks are safe; any other field change (`files_changed`, `use_changed`, `other_change:<field>`, unknown fields
  included) needs judgement. The weekly run writes `review_queue.json` and adds 「分類」/「確認待ち」 sections to the pull
  request body; exit codes are unchanged.
- `python -m genmodelfit.weekly merge-check --base A --head B [--json]`: recompute the classification from two files
  (exit 0 safe / 3 not safe / 1 unreadable) for the runner script that decides the merge.
- `python -m genmodelfit.review publish SRC DST`: the list to sign without pending entries.
- `python -m genmodelfit.review apply DECISIONS.json --db PATH [--max-approve 10] [--json]`: apply the weekly
  decisions (`approve` / `reject` / `hold`) returned by Claude, strictly validated and all or nothing (exit 2);
  approvals are `decided_by: claude-weekly`, safety rejects keep only the id hash, and the summary is escaped for
  the pull request body. `review_queue.json` / `review list --json` items add `card_url` and `files`
  (repo, path, license, license_url, …).
- Duplicate model ids are refused (`parse_db` / `load` / `save` raise `DatabaseError`; `merge-check` exits 1).

### Changed

- **Published data (data-latest) leaves out pending entries.** genmodelfit 0.5.x does not know `review` (it drops the
  key), so a pending entry that passes the other checks would be auto-recommended there. data.yml now runs
  `python -m genmodelfit.review publish` before signing; pending entries reach data-latest only after approval.
  The bundled `db.json` keeps them (0.6.0 holds them with the `reviewed` check).
  0.5.x は review を知らないので、確認待ちの項目は data-latest に配らない。
- The weekly update now runs as a Claude Code skill on the owner's PC; `weekly-data.yml` no longer runs on a schedule
  (manual `workflow_dispatch` backup only). See docs/release.md.
- `review approve` requires evidence URLs that are https with a host (`https://` alone is refused).
- Tables: the "Auto" column is now **"Verified"** (values unchanged: `yes` or the first failed check). JSON keeps
  `auto`, `held_by` and the check ids. `info` says `verified (auto-recommend)`.
- Tables: the `[U] ` prefix on uncensored model names is replaced by a **"Censored"** column right after Score
  (`Y` = censored / `N` = uncensored), shown only when uncensored rows can appear (`--content all|uncensored`; from
  Python, when a row is uncensored). Legend: `Censored: N = safety filter removed / adult model`. JSON unchanged.
- README split into English `README.md` (also the PyPI page) and Japanese `README.ja.md`.

## [0.5.0] - 2026-10-21

### Added

- App fields for apps that build their screens from the list alone (Issue #36): optional `ModelEntry.languages`
  (`{"prompt": [...], "speech": [...]}`, ISO 639-1), `runtime` (`comfyui` / `piper` / `irodori-tts`) and `description`
  (`{"ja", "en"}`), also in `recommend` / `fit` / `search` JSON and `info`. They are rebuilt on every apply (nothing
  carries over). Languages: the hand table `data/overrides.json` wins; otherwise the Hugging Face card's `language`
  (saved as `card_languages`) is the upper bound, narrowed to `en` when every text encoder is English-only (CLIP / T5);
  without a card a known text encoder gives `["en"]` — a multilingual encoder alone never adds `ja`. Undecided stays
  empty, see docs/sources.md. `schema_version` stays 1: readers up to 0.4.0 drop the unknown keys (`tests/test_appinfo.py` reads the
  bundled list with a copy of the old reader).
- Hand-added models (`source: "manual"`, `data/manual.json`, ids `manual:<name>`) that `update` merges, HEAD-checks and
  header-inspects like the others and whose per-file licenses it compares with the upstream cards (stricter wins, with a
  warning). First three: Tsukuyomi-chan (piper-plus), Irodori-TTS v4.1 Small, Stable Audio 3 Small (music).
- `update --fields-only [--offline]`: re-apply only the app fields and the hand-added models to the current list
  (re-reads the Hugging Face cards for the languages; `--offline` sends nothing and uses the saved `card_languages`).
  アプリが一覧だけで画面を作れるように、言語・動かし方・説明の項目と、手で足すモデル(最初の 3 件)を足した。
- The estimator counts `.onnx` and other non-ComfyUI weight files for models whose `runtime` is neither `comfyui` nor
  unknown (piper). Other models are counted as before. **Versions up to 0.4.0 do not count them, so with the new list
  they show Tsukuyomi-chan (onnx) as size unknown and hide it** — apps that list it need 0.5.0.
  0.4.0 以前の版ではつくよみちゃん(onnx)が大きさ不明で出ない。
- Gate check `app_runtime` (in place of `official_workflow` / `custom_nodes` / `comfyui_version`, same readiness weight
  40) for models whose `runtime` is neither `comfyui` nor null: always held, the app checks its own runtime.
- License names with `ethical-restrictions` / `use-restrictions` are in the use-restricted tier. Irodori-TTS v4.1 Small
  is `other` / `mit-with-ethical-restrictions` (MIT plus the card's Ethical Restrictions), not permissive.
- `update` drops an `hf:` entry for the same repository as a hand-added model (its main file's `license_from` or
  `homepage`), so a model is not listed twice.

## [0.4.0] - 2026-10-21

### Added

- `genmodelfit diff OLD NEW [--json]` and `gmf.diff_db(old, new)` (Issue #19): what changed between two lists —
  added / removed models, license changes (changes toward the permissive side are marked), auto-recommendation
  changes under one policy for both lists (HEAD age ignored), size / precision changes beyond a threshold, newly
  uncensored models, failed HEAD checks, 401 repos and changed `sources` numbers. JSON: docs/schema.md "diff".
  2 つの一覧の違いをまとめる `diff` を足した。
- Weekly data refresh (Issue #19): `.github/workflows/weekly-data.yml` rebuilds the list every Monday with
  `python -m genmodelfit.weekly` (caches kept between runs), checks it (auto count, the HEAD expiry must be more than
  7 days away, ruff and pytest with the new list) and opens a pull request `automated/weekly-data` whose body is the
  diff. It is never merged automatically; publishing stays in data.yml after the owner merges. A partial rebuild
  fails unless run by hand with `allow_partial` and only HF repos failed transiently. docs/release.md.
  毎週、作り直した一覧を PR にする(自動ではマージしない。配るのはマージした一覧だけ)。

- Measurement submissions from users (Issue #34). `genmodelfit bench … --submission PATH` and
  `genmodelfit contrib make --from <bench result>` write a submission file (`contrib_schema` 1:
  `data/contrib/<gpu>/<date>-<sha8>.json`) with the GPU name / VRAM / generation / compute capability / driver, the ComfyUI
  version, the OS family, RAM rounded to GiB and the results — never host / user names, paths, GPU UUID, PCI address,
  IP, CPU name or exact times (the note is scrubbed). genmodelfit sends nothing; users open the pull request themselves.
  `contrib show` prints exactly what will be shared with a privacy checklist; `contrib validate` checks schema, value
  ranges, privacy patterns, GPU name / generation consistency, file name and duplicates (exit 0 / 1 / 2);
  `.github/workflows/contrib.yml` runs it on pull requests touching `data/contrib/**` (read-only, the tool installed
  from the base commit, results in the job summary). `contrib aggregate` groups submissions by model, settings and GPU
  with median / MAD outlier removal into the new bundled `measurements_community.json` (empty for now), and
  `--report` re-runs the calibration and flags errors above a threshold (default 25%, exit 3 with `--fail-on-breach`)
  for the weekly job of #19.
- `Estimate.basis_detail` (`measured` > `measured_community` > `calibrated` > `estimated`); `basis` stays
  `measured` / `estimated`. Community measurements are used only when at least two consistent submissions agree (and
  no OOM), rank after the maintainer's measurements at the same settings, and can only raise the needs (never lower
  VRAM / RAM, never improve a level, never raise the score: `basis` points 0 like estimates). `validate` marks
  implausible values (more than 40% from the formula, below the weights with `offload: none`), values that would raise
  the estimate by more than 15% or change a level at common VRAM / RAM sizes, unknown models and unknown GPUs as
  review-required; approval is per submission (`--allow-review <sha8>` or the maintainers' `data/contrib/REVIEWED`).
  Unapproved community rows raise needs by at most 15% and are ignored where they would change the level.
  `aggregate --report` lists the level changes caused by community data.
- `torch_reserved` measurements (bench; without the CUDA context) are now lower bounds: below the formula they no
  longer lower the estimate (the bundled RTX 3070 rows for BiRefNet and MoGe now use the formula value).
  `Measurement.origin` / `n_submissions` / `consistent` (optional fields).
- `python -m genmodelfit.calibrate` also writes the check rows (measurements in another `vram_metric`, such as the
  RTX 3070 runs) to `calibration.json` (`check`) and the accuracy table; LOO rows carry `gpu_name` and `origin`.
  Coefficients are unchanged.
  利用者の実測の提出(作る・見る・検査・まとめ・毎週の精度の点検)と、根拠の段階 `basis_detail` を足した。

### Changed

- Hugging Face selection (Issue #19): the 1,000 all-time floor for models found only by the trending / likes30d
  searches now needs a trendingScore of at least 0.5 and at least 100 downloads in the last 30 days (otherwise
  10,000 applies); repo names containing `merge` / `mergekit` are skipped like the tag; quantized / format copies
  are recognized by name (`-GGUF`, `-Q8_0`, `_K_M`, `-fp8`, `-nf4`, `-int8`, `_diffusers`, `-Quantized`, …) and
  only the most downloaded is kept; uncensored models also need 100 downloads in the last 30 days.
  HF の選び方を締めた(伸びている条件・名前の merge・量子化の写し・無検閲は直近 30 日も使われているもの)。
- The bundled list was rebuilt with these rules (326 entries: HF 48, uncensored 26; auto 99 unchanged).
  Added: 6 uncensored HF models that meet the rules and the new official template `video_fasth3_ominiref_r2v`;
  removed: `hf:0xSojalSec/Qwen-Image-2.1-Uncensored-GGUF` (a quantized copy of
  `hf:abenzerps/Qwen-Image-2.1-Uncensored-Quantized`) and `hf:TheImposterImposters/lustifySDXLNSFW_v20-inpainting`
  (the same weights, by sha256, as the more downloaded `hf:andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING`).
  同梱の一覧を新しい規則で作り直した。
- Weekly PR body: every upstream string is HTML-escaped and Markdown-neutralized (`weekly.md`), mentions are
  broken, truncation keeps `<details>` balanced; a scheduled run fails when more than 5 entries become uncensored
  or when HF / template / uncensored / auto counts fall below half.

### Fixed

- Exclusion (safety): uncensored models whose ID, name, tags or card mention a dataset known to include a category
  of sexual content involving minors (`DATASET_TERMS`: e621) are excluded, even when that is not their purpose
  (owner decision, 2026-10-21; `hf:hearmeneigh/sd21-e621-rising-v2` and `hf:Laxhar/noobai-XL-1.1` — its card says it was trained on the full Danbooru and e621 datasets — removed from the bundled list).
  学習データに未成年の性的表現の分類を含む画像の集まり(e621)を使う無検閲のモデルは外す(安全側)。

- When a Hugging Face repo was renamed, `update` applied the license-source table with the new name from the API
  while re-applying it uses the name in the file URL, so the two disagreed (`license_path`). Both now use the URL's
  name. Found by the first weekly rebuild (Comfy-Org/flux2-klein-9B → vae-text-encorder-for-flux-klein-9b).
  リポジトリの名前が変わったとき、出どころの表をファイルの URL の名前で当てるようにした。
- Tests no longer pin numbers of the bundled list (uncensored count, HEAD check date) so that the weekly rebuild
  can pass them.


## [0.3.0] - 2026-10-21

### Added

- `genmodelfit download <model>` and `gmf.plan_download()` / `gmf.download()` (Issue #31): download all files of a
  model's workflow, with the precision chosen like `recommend` (`--precision`, `--budget GB`), into a ComfyUI folder by
  each file's template folder (`--comfyui-dir`) or the same layout elsewhere (`--output-dir`). `--list` shows the plan
  without any network access. Before downloading it shows every file's license and URL, the total size and the free
  disk and asks for consent (`--yes` to skip). sha256 is checked while writing (a mismatch deletes the file),
  `.part` files resume with HTTP Range, files already there are skipped, and it needs the download + 1 GB free.
  Gated files, uncensored models (unless `--content all|uncensored`) and unknown licenses (unless
  `--include-unknown-license`) are refused; only https URLs on Hugging Face / GitHub hosts are followed (each
  redirect is checked, and URLs that urlsplit and urllib3 would read as different hosts are refused before
  connecting); nothing is written outside the destination. Existing files are hashed again right before they
  are skipped, restarts against servers that ignore Range are capped, and a `.lock` file stops two downloads of
  the same file. URL paths may contain spaces (they are percent-encoded as requests does); spaces, `%` and `@` are
  refused only in the scheme and host, and refusals say whether the URL is malformed or the host is not allowed.
  JSON: docs/schema.md "download".
  モデルを落とす `download` を足した(同意・sha256・続きから・ComfyUI のフォルダ)。

## [0.2.0] - 2026-10-20

Fewer unknowns in the table and a clear meaning for `?` and `-` (Issue #30). The JSON output changes in a way that
can break apps (below), so this is a minor release under 0.x semver (owner's decision).
表の「?」「-」を減らし、使い分けをそろえた版。JSON の形が変わるので 0.2.0 にした(オーナーの判断)。

### Breaking (JSON)

These only concern models whose weight size is unknown (`estimate.unknown_size: true`, `fit_level: "unknown"`):

- `run_mode` (top level and `estimate.run_mode`) is now **`"unknown"`** (was `"none"`). `"none"` now only means
  "does not run" (`fit_level: "too_tight"`). `run_mode_label` is `"?"` (was `"-"`).
- `estimate.vram_gb`, `estimate.vram_offload_gb`, `estimate.ram_gb` and `estimate.vram_sequential_gb` are **`null`**
  (were `0.0`). The Python `Estimate` attributes stay `0.0`; only `Estimate.to_dict()` / `--json` change.
- `memory_available_gb` is now this PC's VRAM (or RAM) instead of `null` (the available amount is known;
  `memory_required_gb` / `utilization_pct` stay `null`).

Migration / 移行:

- Code that treated `run_mode == "none"` as "cannot use" should also handle `"unknown"` (or check
  `fit_level` / `estimate.unknown_size` first, as before).
- Code that summed or compared `estimate.vram_gb` / `ram_gb` must skip `null` (it was a meaningless `0.0` before).
- `fit_label` / `fit_level` and `schema_version` (1) are unchanged; the db.json fields added below are optional.
大きさが分からないモデルだけ: run_mode は unknown、estimate の VRAM / RAM は null、memory_available_gb は
この PC の量。run_mode の none を「使えない」と見ていたコードは unknown も扱う。

### Changed

- **`?` means unknown, `-` means not applicable**, in the table, `info`, the legend and JSON (`docs/schema.md`,
  「Unknown and not applicable」). Prec of an unknown precision is now `?` (was `-`); Mode / Mem% of a model whose
  size is unknown are `?` (Mode `-` stays for "does not run", VRAM `-` for CPU, Mem% `-` without a GPU). A new
  legend line explains both marks.
  「?」は分からない、「-」は当てはまらない。精度が分からないときは「?」(前は「-」)。
- License table: `circlestone-labs/Anima-Official-LoRAs` has a `LICENSE.md` (CircleStone Labs Non-Commercial License
  v1.0, added 2026-05-26) although its card metadata is empty; the table now sets the LoRAs to
  `other` (`circlestone-labs-non-commercial-v1.0`). The four Anima templates are no longer "license unknown" (still not
  auto-recommended: non-commercial). Bundled list: license unknown 47 → 43 entries (no uncensored), 60 → 56 in all.
  Anima の公式 LoRA は LICENSE.md(非商用)があったので表で決めた。

### Added

- **Precision from more sources**, never overriding a precision read from the file header
  (`ModelEntry.precision_source`: `header` / `filename` / `hf_api` / `config` / `workflow`, optional, schema 1;
  also `precision_source` in `models[]` and `workflows[]` of `recommend` / `fit` / `search --json`):
  - `hf_api`: the Hub API's `safetensors.parameters` when one dtype holds ≥ 90 % **and** element count × bytes is
    within 5 % of the entry's main safetensors size in that repo (readable anonymously even for gated repos whose
    file headers return 401);
  - `config`: `torch_dtype` / `dtype` of the `config.json` next to the main file (only when listed and not gated);
  - `workflow`: the official template's main loader node (`UNETLoader` `weight_dtype`, `WanVideoModelLoader`
    `quantization`, the loaded file name, `precision` widgets, …; `workflow.loader_precision`).
  New optional fields `ModelFile.repo_precision` / `repo_precision_source` and `ModelEntry.workflow_precision`.
  Tool weights whose role is not `main` (BiRefNet, Depth Anything 3) now get their header read too.
  Bundled list (no uncensored): precision unknown 25 → 10 (header 3, Hub API 4, workflow 8; csm-1b stays unknown
  because the Hub API total does not match its files; the rest are listed in docs/sources.md with the reason).
  精度をリポジトリ(Hub API・config.json)とテンプレートの読み込みノードからも読み、出どころを残す。
- `sources.repo_precision` (counts per source, `unauthorized`, `size_mismatch`), `sources.huggingface.unauthorized`
  (repos whose Hub API returns 401 — anonymous 401 is the same for private and missing repos),
  `unauthorized_missing` (the ones confirmed not to exist, `repo_precision.KNOWN_MISSING`: `google/gemma-3-12b`) and,
  after `update --inspect-only`, `unauthorized_recheck` (`unauthorized` / `private_or_missing` / `missing` /
  `readable` / `failed`), so the weekly update (#19) shows when Mage-Flow or Lens become public.
  401 のリポジトリを記録し、`--inspect-only` で確かめ直す。無いと確かめた名前は分ける。
- `docs/license_requests.md`: drafts of polite requests to upstream owners whose weights have no stated license or
  a license that conflicts with their base model (not posted; to be sent only after the owner's review).
  ライセンスの記載の依頼の下書き(まだ出さない)。

## [0.1.1] - 2026-10-20

Fixes from the owner's review on an RTX 4500 Ada 24 GB / 128 GB RAM PC (Issue #28).
RTX 4500 Ada 24 GB・メモリ 128 GB の PC でのレビューの指摘を直した版。

### Changed

- License tracing (#28): a hand-checked source table (`license_sources.py`) maps files in repackaging repos
  (Comfy-Org, Kijai, …) to their original models, using primary sources only (original model card / LICENSE,
  official GitHub, sha256). `update` applies it to files it could not resolve; `apply_license_sources(db)` applies it
  offline. VOID (netflix/void-model) is now unknown (fine-tuned from a CogVideoX-licensed model). `LICENSE_POLICY` 4.
  ライセンスの出どころの表を足した(再配布のファイル → 元のモデル。一次資料だけで確かめたもの)。
- **Wan 2.2 14B and other workflows that run several diffusion models one sampler after another** (high_noise →
  low_noise): the swapped (`GPU~`) and offload estimates count one model (+ its LoRA) at a time, because ComfyUI
  unloads the unused model before loading the next one (`load_models_gpu` → `free_memory`, ComfyUI commit
  b1a247c, cited in `docs/schema.md`). Wan 2.2 14B I2V at 832×480×81: swapped 35.5 → 21.0 GiB, offload 18.1 → 12.3 GiB.
  On 24 GB it now runs on the GPU with parts swapped (`GPU~`, `marginal` at 87 %) instead of `GPU+RAM`; on 16 GB it is
  `marginal` with offload instead of `too_tight`. The sets come from the template graph (`sampler_stages`, new) or
  from `*high_noise*` / `*low_noise*` file names.
- **Precision choice within a model**: among the variants that run on the GPU without offload (Good or better, `GPU`
  or `GPU~`), the highest precision is shown, like llmfit; only when none does, the better level wins as before.
  Z-Image Turbo on 24 GB now shows bf16 instead of int8. The row's score is taken from the same use and the same tier
  (runs without offload = one tier), so it never drops when VRAM grows.
- **Unknown licenses are hidden by default** from `recommend`, `fit`, `search`, `gmf.recommend()` and `gmf.search()`
  (`hidden.license_unknown`); `--include-unknown-license` / `include_unknown_license=True` shows them; `info <id>`
  still shows them.
- **Models whose size is unknown show `?`** (`fit_level: "unknown"`, `fit_label: "?"`) instead of "Too tight";
  `fit` lists them last, `recommend` still hides them.
- **Community templates** come after official ones only within the same auto group and fit level (they used to come
  after every official row, so a Good community model appeared after Marginal ones).
- TTS GGUF repositories (llama.cpp-style runtimes: VieNeu-TTS v3 Turbo, Qwen3-TTS GGUF, Higgs Audio v3 Studio,
  Drama-3) are no longer marked incomplete for lacking a text encoder / VAE, so they are judged.
- OS shown as `Windows 11` on Python 3.10 / 3.11 too (build number ≥ 22000).
- **`fit` is easier to read** (like llmfit's `fit`): it shows only the models that run (Perfect / Good / Marginal)
  and counts the rest by reason in the header; `--all` shows everything (tools, models that do not run, `?`, unknown
  licenses). One table ordered by type → fit level → Score (no auto-first; see the Auto column). New options for
  `fit` / `search`: `--fit perfect|good|marginal|runnable|all` (`--min-fit` still works, not together with `--fit`),
  `--sort type|score|vram|disk|name`, `--all`; `fit --include-tools`. JSON follows the same order. Python:
  `gmf.fit_list()`, and `fit=` / `sort=` on `gmf.search()`.
- Table columns in every command: `Type | Status | Score | Model | …` (Type leftmost, Score left of the name).
- Review fixes: `recommend` ranks "runs without offload" (Perfect / Good on `GPU` / `GPU~`) as one tier, so Z-Image
  Turbo stays first in image on 24 GB after switching to bf16; JSON `level_cap` (`sequential` / `offload` / `cpu`) and a
  legend line explain why GPU~ is at most Good; `--fit` with `--min-fit` is a usage error (exit 2); info and the table
  round Mem% the same way; malformed file headers are reported as header errors.
- Licenses: rows of the source table can override an automatic result (`override`). ChronoEdit's distill LoRA was
  apache-2.0 from the card's metadata but the card says it is governed by the NVIDIA Open Model License; InfiniteTalk
  and Wan 2.1 I2V 480P now point at their real upstreams. Files of lineages whose license cannot be read (VOID) are
  unknown in the data too (as the gate already judged). An entry mixing apache-2.0 and mit files is shown as
  apache-2.0 (more obligations), not mit. `update.LICENSE_POLICY` 4 → 5.

### Added

- Precision from the files themselves: `update` (and the new `update --inspect-only`, `--offline` for the cache only)
  reads the safetensors JSON header / GGUF header of the main weight files with HTTP Range requests (never the
  weights), caches them in `file_headers.json`, and stores `ModelFile.header_precision` / `header_params`. On the
  bundled list, entries with unknown precision went from 97 to 25 (sfw: 90 → 25; the rest are gated repos, 401, or
  have no main file on Hugging Face).
- `ModelEntry.sampler_stages`, `ModelFile.header_precision`, `ModelFile.header_params` (optional; `schema_version`
  stays 1). `sources.file_headers`, `sources.sampler_stages`.
- README: how to install so that the `genmodelfit` command is found (pipx / uv tool, the pip "not on PATH" warning,
  and `python -m genmodelfit`, which always works). Also in `genmodelfit --help`.

### Data

- Bundled list: HEADs re-checked on 2026-10-20T10:33:07Z (valid until 2026-11-19T10:33:07Z); precision from file
  headers; `sampler_stages` from the 277 official templates (19 with several sampler sets).

## [0.1.0] - 2026-10-20

First public release. / 最初の公開版。

### Added

- Hardware detection (`genmodelfit system`, `gmf.system()`): NVIDIA GPUs via `nvidia-smi`, RAM, free disk;
  `--vram` / `--ram` to judge another PC. Other PCs are judged as CPU-only.
- Bundled model list (`db.json`): 323 entries (image, video, music, sound effects, TTS, 3D) built from the official
  ComfyUI workflow templates and the Hugging Face Hub API, folded into 204 models. Hugging Face models need at least
  10,000 all-time downloads (1,000 for trending ones), like llmfit.
- Memory estimates per component and stage, with swapping in turn, offload to RAM and CPU run modes; every estimate
  says whether it is `measured` or `estimated`. Calibration on real runs (`python -m genmodelfit.calibrate`,
  `docs/accuracy.md`).
- Fit levels by memory utilization (Perfect ≤ 60 %, Good ≤ 85 %, Marginal ≤ 98 %, Too tight), like llmfit.
- `recommend`, `fit`, `search`, `types`, `info` with a readable table and `--json` for apps (`docs/schema.md`):
  one row per model (same weights family and size), best precision that fits, score (0–100), per-type sections,
  filters by type, use (`--task`), license, fit level and content.
- Auto-recommendation gate (`genmodelfit gate`, `auto` / `held_reasons`): license allowlist traced to the upstream,
  gating, official workflow without custom nodes, ComfyUI version, anonymous HEAD within 30 days, sha256, known size,
  content `sfw`.
- Content marks: uncensored models are kept in the list with evidence but hidden by default (`content="sfw"`);
  models aimed at sexual content involving minors are excluded.
- `genmodelfit update` (rebuild from upstream, `--heads-only`) and `update --from-release` (signed list from the
  `data-latest` release, Ed25519, rollback protection).
- `genmodelfit bench`: measure peak VRAM / RAM / time on a running ComfyUI and contribute measurements
  (`CONTRIBUTING.md`).
- Python API: `gmf.recommend()`, `gmf.search()`, `gmf.info()`, `gmf.system()`; library functions never print.
- Data freshness: `gmf.data_status()` (`DataStatus`: source, HEAD check dates, `expired`, `expires_at`,
  `auto_count`) and the Data lines of `genmodelfit system` / `update` (`database.status` in `system --json`).
  It uses the gate's own check, so `expired` tells apps when auto-recommendations have dropped.
- Type information (`py.typed`), Python 3.10–3.13 on Windows and Linux.
- Privacy: read-only commands and the Python API never touch the network; HTTP sends a fixed User-Agent and never
  attaches `~/.netrc` credentials. No telemetry (README "Privacy").

### Changed

- License: MPL-2.0 for the code and the bundled data (was MIT before the first release).
  Attribution for the upstream data is in `THIRD_PARTY_NOTICES`.

[Unreleased]: https://github.com/laxshmate/genmodelfit/compare/v0.10.0...HEAD
[0.10.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.10.0
[0.9.2]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.9.2
[0.9.1]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.9.1
[0.9.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.9.0
[0.8.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.8.0
[0.7.1]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.7.1
[0.7.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.7.0
[0.1.0]: https://github.com/laxshmate/genmodelfit/releases/tag/v0.1.0
