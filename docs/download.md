# Downloading models (`genmodelfit download`)

**English** · [日本語](download.ja.md)

The detailed rules of step 5 ("Download it.") in "2. Procedure" of the [README](../README.md).
For CLI users and for app developers using the Python API.
The JSON shape is in [schema.md](schema.md) (section "download").
The implementation is `src/genmodelfit/downloader.py`; `tests/test_download.py` checks these promises.

## Command line

```
genmodelfit download "Z-Image-Turbo" --list                 # what would be downloaded (no network, writes nothing)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
genmodelfit download MODEL --content all --include-unknown-license   # uncensored / unknown license (your choice)
```

- **Which files**: all files of one workflow.
  - A model name (as in the tables) picks the workflow and precision like `recommend`: the highest precision that runs on this PC without offloading.
  - A workflow id is used as is.
  - `--precision int8` picks a precision.
  - `--budget 15` (GB) caps the download (files already there are not counted).
- **Where**:
  - `--comfyui-dir` puts each file in the ComfyUI folder the template names (`models/diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, ...). Subfolders in the file name are kept.
  - `--output-dir` (default: genmodelfit's user data folder, `models`) uses the same layout, so it can be added to ComfyUI's `extra_model_paths.yaml`.
  - The environment variable `GENMODELFIT_COMFYUI_DIR` is the default for `--comfyui-dir` (CLI only; in Python pass `comfyui_dir`). Order: `--comfyui-dir`, then `--output-dir`, then `GENMODELFIT_COMFYUI_DIR`, then the default folder. The value is checked like the flag (it must look like a ComfyUI folder). When none is set, the plan says the folder is genmodelfit's (ComfyUI does not read it) and shows how to set it, for this OS only: PowerShell `$env:GENMODELFIT_COMFYUI_DIR = 'C:\ComfyUI'` on Windows, bash / zsh `export GENMODELFIT_COMFYUI_DIR="$HOME/ComfyUI"` elsewhere.
  - Files whose ComfyUI folder is unknown (many Hugging Face models) go to `other/`. ComfyUI does not read `models/other`: the plan says so, and you move them (to `checkpoints`, `diffusion_models`, `loras`, ... as the model page says) before use.
  - For a file without a folder in the list (from any source) the folder is inferred only from one piece of evidence in the bundled list: the main file's header parameter count matches a file of an official ComfyUI template within 0.001%, and all matching files are in one folder (`unet` and `diffusion_models`, `clip` and `text_encoders` count as one; e.g. an SDXL all-in-one file has the same count as `sd_xl_base_1.0.safetensors`, so `checkpoints`). The plan shows the basis (`directory_basis` in JSON). Names and tags are not used to guess. A GGUF file placed in `diffusion_models` / `text_encoders` this way needs the ComfyUI-GGUF custom nodes; the plan says so.
  - Since #75 the main files of 12 Hugging Face models go to `checkpoints/` or `diffusion_models/` instead of `other/`. If you downloaded one of them with 0.9.2 into `other/`, it is downloaded again into the new folder; move the file there yourself first to skip that (the sha256 is checked again).
- **Consent first**: before anything is downloaded you see:
  - every file with its license and the license URL. A file with no license URL says "no license stated" and shows the page to check (the Hugging Face page of the model, or the page stored in the list); the consent question then asks you to check those pages yourself;
  - licenses that are not on the allowlist, marked (commercial use without use restrictions; see [gate.md](gate.md); change it with `--license-allowlist`);
  - the total size and the free disk.
  - Then you are asked `[y/N]`. `--yes` skips the question (required with `--json`).
  - Read the terms of each file before you use the model.
- **Refused**: if any file matches, nothing of that workflow is downloaded. It never downloads only part of it.
  - Each reason is shown once. When only `--content all` and / or `--include-unknown-license` are missing, the plan ends with "To download anyway:" and one command that adds all of them (keeping your other options), with one line per flag on what you agree to. It never adds them by itself. For gated files, files genmodelfit cannot download (their URLs are listed so you can get them yourself) and workflows without files it says genmodelfit cannot download the workflow, and names another workflow of the same model without that problem when there is one. For disk space, `--budget` and a different file with the same name it names the lever (free space or `--output-dir`, `--budget`, move the file), then the consent flags if they are also missing. In both cases no command is shown. The command uses the workflow id of the plan and the program form you ran (`genmodelfit` or `python -m genmodelfit`); arguments are put in single quotes when needed. A short `error:` line goes to stderr (exit code 1).
  - Gated files. genmodelfit never handles logins or tokens: request access on the model page and download them yourself.
  - Uncensored models, unless you pass `--content all` (or `uncensored`).
  - Files whose license is unknown, unless you pass `--include-unknown-license`.
  - Files without a sha256 or size in the list.
  - Hosts other than Hugging Face / GitHub.
- **Safety**:
  - Every file is checked with the sha256 in the list while it is written. A mismatch deletes it and fails.
  - A download goes to `<file>.part` first. After Ctrl-C or a network error, run the same command again and it resumes with an HTTP Range request. If the server ignores Range it starts over, at most 3 times.
  - Files already there are hashed again right before they are skipped. A different file with the same name is never overwritten.
  - A `<file>.lock` next to the file stops two downloads of the same file at once. A lock left by a process that is gone is cleaned up.
  - It does not start unless the free disk is the download + 1 GB.
  - Nothing is written outside the destination. These are refused: names with `..`, absolute paths, drive letters, `\`, reserved Windows names (also as folder names), control / bidi characters, and symbolic links / junctions that lead outside.
- **Network**:
  - Only the URLs in the list are used (checked anonymously with HEAD by `update`; `download` itself does not require a HEAD result), https only.
  - The first URL must be on `huggingface.co` or `github.com`.
  - Every redirect must stay on these hosts (https, port 443 only): `huggingface.co`, `*.huggingface.co`, `hf.co`, `*.hf.co` (CDN, Xet), `github.com`, `objects.githubusercontent.com`, `release-assets.githubusercontent.com`.
  - A URL that could be read as two different hosts is refused before any connection: `\`, spaces, control characters, `%` or `@` in the host, non-ASCII hosts. The host is read the way requests / urllib3 read it.
  - No `~/.netrc` credentials, no tokens, nothing about this PC (User-Agent and Range only).
  - On HTTP 429 / 500 / 502 / 503 / 504 it waits for `Retry-After` (or backs off) and retries. It stops after 4 failures in a row without progress.

## Python

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

- The library prints nothing.
- `gmf.download` raises `gmf.DownloadRefused` (blocked, no consent) and `gmf.DownloadError` (network, sha256 mismatch, disk).
- It lets `KeyboardInterrupt` through and keeps the `.part` file for the next call.
