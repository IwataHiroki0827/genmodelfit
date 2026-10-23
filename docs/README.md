# genmodelfit documentation

**English** · [日本語](README.ja.md)

Start with the [README](../README.md): install, the commands, and how to read the results.
The pages below go into detail. Every page has an English version and a Japanese version with the same content.

## By reader

| You want to… | Read |
|---|---|
| Know how your PC is detected and judged | [hardware.md](hardware.md) |
| Know how far to trust the memory estimates | [accuracy.md](accuracy.md) |
| Download a model's files safely | [download.md](download.md) |
| Use genmodelfit from an app (`--json` or Python) | [schema.md](schema.md), then [gate.md](gate.md) and [download.md](download.md) |
| Send your measurements | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| Know where the model list comes from | [sources.md](sources.md) |
| Release a version or update the list (maintainer) | [release.md](release.md) |

## All pages

| Page | What it covers |
|---|---|
| [hardware.md](hardware.md) | How genmodelfit detects GPUs, memory and drivers, and the primary-source rules behind each verdict. |
| [accuracy.md](accuracy.md) | How the VRAM / RAM estimates are built and fitted, how accurate they are, and how other users' measurements are used. |
| [download.md](download.md) | The rules of `genmodelfit download` (consent, refusals, safety, network) and its Python API. |
| [schema.md](schema.md) | The JSON output of every `--json` command, the stable Python API and the `db.json` format. |
| [gate.md](gate.md) | When a model may be recommended automatically (`auto: true`): checks, license allowlist, link-check age, ComfyUI version. |
| [sources.md](sources.md) | The data sources and rules `genmodelfit update` uses to build the model list: templates, licenses, languages, uncensored models, exclusions. |
| [release.md](release.md) | For the maintainer: package releases, signed data releases, the weekly list update, signing keys. |
| [license_requests.md](license_requests.md) | Drafts of requests asking upstream owners to state a license for their model weights (Issue #30). |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to measure with `genmodelfit bench`, check the file, and send it as a pull request. |
| [CHANGELOG.md](../CHANGELOG.md) | What changed in each version (English only). |
