# Requests to state a license (drafts)

**English** · [日本語](license_requests.ja.md)

Drafts of requests to upstream owners, for models that genmodelfit treats as "license unknown"
because the license of the weights is unknown (not stated, or different from the base model's terms) (Issue #30).
This page is for maintainers.
The draft texts are meant to be posted on other people's English-language repositories, so they are not translated: the same text is in the English and Japanese pages.

## Contents

- [Background](#background)
- [A. No license stated for the weights](#a-no-license-stated-for-the-weights)
- [B. The stated license is more permissive than the base model's terms (mismatch)](#b-the-stated-license-is-more-permissive-than-the-base-models-terms-mismatch)
- [B2. A license exists, but the card metadata is empty](#b2-a-license-exists-but-the-card-metadata-is-empty)
- [C. Repositories that cannot be read anonymously (401): check again once public](#c-repositories-that-cannot-be-read-anonymously-401-check-again-once-public)
- [D. Cases with no draft](#d-cases-with-no-draft)

## Background

- **Nothing has been posted anywhere yet. Posting waits for the maintainer's (laxshmate's) confirmation**, because it writes to other people's repositories.
  - When posting, use the English drafts below as they are: a Community Discussion for a Hugging Face repository, an Issue for GitHub.
- How the list was made:
  - It was built from the files with an unknown license in the bundled list (`src/genmodelfit/data/db.json`) and grouped by upstream owner.
  - It was made on 2026-10-20, and on 2026-10-21 the bundled list confirmed that all of them are still unknown.
  - The current state can be read from:
    - `model.license` in `genmodelfit info <id> --json` (null means unknown)
    - `license` of each file (null means unknown; `license_from` is the source it was read from)
  - How a license is decided: "License" in [sources.md](sources.md).
  - Once a license is stated, rebuild the list with the table in `license_sources.py` or the upstream's statement.
- Effect on users (common to all):
  - genmodelfit treats an unknown license the same as "not permitted" ("License contract" in [schema.md](schema.md)).
  - By default such models are not shown by `recommend` / `fit` / `search` (`--include-unknown-license` shows them), and not in the automatic recommendation (gate) either.
  - An app that is distributed without its source (such as AI生成キット) cannot recommend those models to its users.

## A. No license stated for the weights

| Upstream | What is missing | Affected items (uncensored ones excluded) |
|---|---|---|
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | The README says "Model assets are governed by the repository license", but the repository has no license (metadata or LICENSE) | 3 items: MiniMax H3 i2v, continuation, t2v |
| [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) | No license. It is the base_model of Kokoro-82M (apache-2.0), so Kokoro becomes unknown too | TTS Kokoro-82M |
| [xtuner/llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b) | No license (weights made from Llama 3). The origin of HunyuanVideo's text encoder `llava_llama3_*` | 1 item: HunyuanVideo text-to-video |
| [ByteZSzn/Flux.2-Turbo-ComfyUI](https://huggingface.co/ByteZSzn/Flux.2-Turbo-ComfyUI) | No license (`Flux2TurboComfyv2`, a LoRA for FLUX.2 [dev]) | 1 item: FLUX.2 fp8 |
| [google-research/frame-interpolation](https://github.com/google-research/frame-interpolation) (FILM) | The code is Apache-2.0. The weights are on Google Drive, with no statement of a weight license (`film_net_fp16` in Comfy-Org/frame_interpolation) | 1 item: frame interpolation (FILM) |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | The code is Apache-2.0. The weights are distilled with DINOv3 as the teacher, with no statement of a weight license (`rt_detr_v4-x-hgnet_*` in Comfy-Org/SDPose) | 4 items: SDPose multi-person, SAM 3D Body, MiniMax H3 ControlNet |

### Drafts (English)

**lightx2v/Minimax-h3-Turbo**

> Title: License for the Turbo weights
>
> Hello, and thank you for the Minimax-h3-Turbo release. The README says "Model assets are governed by the
> repository license", but the repository has no license in its metadata and no LICENSE file. ComfyUI's official
> MiniMax H3 templates (image-to-video, continuation, text-to-video) download
> `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` from this repo, and license-aware tools have to treat
> it as "license unknown". Could you add the repository license (a LICENSE file and `license:` in the card
> metadata), e.g. stating whether it follows the base MiniMax H3 model's terms? Thank you very much.

**yl4579/StyleTTS2-LJSpeech**

> Title: License for the LJSpeech checkpoint
>
> Hello, and thank you for StyleTTS 2. This repository has no license in its model card. hexgrad/Kokoro-82M lists it
> as its `base_model`, so tools that trace licenses through `base_model` cannot confirm Kokoro's terms either and
> show it as "license unknown". The StyleTTS 2 code on GitHub is MIT; could you add a `license:` entry (or a LICENSE
> file) to this repo saying which terms apply to the pretrained weights? Thank you!

**xtuner/llava-llama-3-8b**

> Title: License for the weights
>
> Hello, and thank you for the LLaVA-Llama-3 models. This repo does not state a license. Its weights are used (as
> `llava_llama3_*.safetensors`) as the text encoder of HunyuanVideo in ComfyUI's official template, and
> license-aware tools treat that file as "license unknown". Could you add the applicable license to the model card
> — presumably the Meta Llama 3 Community License plus any terms of your own? Thank you.

**ByteZSzn/Flux.2-Turbo-ComfyUI**

> Title: License for Flux2TurboComfyv2
>
> Hello, and thanks for sharing the FLUX.2 Turbo LoRA. The model card has no license. ComfyUI's official FLUX.2
> fp8 template downloads `Flux2TurboComfyv2.safetensors` (re-hosted in Comfy-Org/flux2-dev), and license-aware tools
> treat it as "license unknown". Could you state the license in the model card metadata (for example, the FLUX.2
> [dev] Non-Commercial License that applies to the base model)? Thank you!

**google-research/frame-interpolation (FILM)**

> Title: License of the pretrained model weights
>
> Hello, and thank you for FILM. The repository code is Apache-2.0, but the pretrained models are hosted on Google
> Drive and I could not find a statement of which license applies to the weights. They are widely used through
> ComfyUI (re-hosted as `film_net_fp16.safetensors` / `film_net_fp32.pt`), and license-aware tools have to treat them
> as "license unknown". Could you clarify in the README whether the pretrained weights are also under Apache-2.0
> (or which terms apply)? Thank you very much.

**RT-DETRs/RT-DETRv4**

> Title: License of the released weights
>
> Hello, and thank you for RT-DETRv4. The code is Apache-2.0, but the README does not say which license applies to
> the released checkpoints (e.g. `rt_detr_v4-x-hgnet`), which are distilled with DINOv3 as the teacher. ComfyUI's
> official SDPose / SAM 3D Body templates use these weights, and license-aware tools treat them as "license unknown".
> Could you state the license of the weights in the README (and whether the DINOv3 license affects them)? Thank you!

## B. The stated license is more permissive than the base model's terms (mismatch)

The upstream states a license, but it is more permissive than the base model's terms, so it cannot be taken at face value and the license was set to unknown.

| Upstream | Stated | Base model | Affected items (uncensored ones excluded) |
|---|---|---|---|
| [jingheya/lotus-depth-d-v1-1](https://huggingface.co/jingheya/lotus-depth-d-v1-1) | apache-2.0 | Stable Diffusion 2 base (CreativeML Open RAIL++-M) | 5 items that use Lotus depth (Lotus Depth, FLUX depth LoRA, Qwen Image ControlNet, SD3.5 Large depth, LTX-2 depth-to-video) |
| [teemosliang/SDPose-Wholebody](https://huggingface.co/teemosliang/SDPose-Wholebody) | mit | Stable Diffusion 2 U-Net (CreativeML Open RAIL++-M) | 4 SDPose items and 1 MiniMax H3 ControlNet item |
| [netflix/void-model](https://huggingface.co/netflix/void-model) | apache-2.0 | CogVideoX-Fun-V1.5-5b-InP (the transformer is under the CogVideoX License) | 1 item: VOID video erasing |
| [joyfox/LTX-2.3-Transition-LORA](https://huggingface.co/joyfox/LTX-2.3-Transition-LORA) | apache-2.0 | Lightricks/LTX-2.3 (LTX-2 Community License) | 1 item: LTX-2.3 Style Transition (not an official ComfyUI template, but user shane's template) |
| [lrzjason/ObjectRemovalFluxFill](https://huggingface.co/lrzjason/ObjectRemovalFluxFill) | mit | FLUX.1 Fill [dev] (FLUX.1 [dev] Non-Commercial License) | 1 item: FLUX.1 Fill dev OneReward |
| [bytedance-research/USO](https://huggingface.co/bytedance-research/USO) | apache-2.0 | FLUX.1 [dev] (FLUX.1 [dev] Non-Commercial License) | 1 item: FLUX.1 dev USO |
| [Jmica/Drama-3-GGUF](https://huggingface.co/Jmica/Drama-3-GGUF) | other (fish-audio-community-license-v1.0) | fishaudio/drama-3 (fish-audio-research-license) | TTS Drama-3 GGUF |

### Drafts (English)

**jingheya/lotus-depth-d-v1-1** (the same text can be used for SDPose: replace "Lotus" with "SDPose" and "apache-2.0" with "MIT")

> Title: Question about the license of the weights (base model terms)
>
> Hello, and thank you for Lotus. The model card says apache-2.0, and the paper describes the model as fine-tuned
> from Stable Diffusion 2 base, whose weights are released under the CreativeML Open RAIL++-M license (with use-based
> restrictions that carry over to derivatives). Could you clarify which terms apply to these weights — Apache-2.0
> alone, or Apache-2.0 together with the RAIL++-M restrictions? ComfyUI's official templates use this file, and
> license-aware tools currently show it as "license unknown" because the two disagree. Thank you!

**teemosliang/SDPose-Wholebody**

> Title: Question about the license of the weights (base model terms)
>
> Hello, and thank you for SDPose. The model card says MIT, and the model is built from the Stable Diffusion 2
> U-Net, whose weights are released under the CreativeML Open RAIL++-M license (its use restrictions carry over to
> derivatives). Could you clarify which terms apply to `sdpose_wholebody` — MIT alone, or MIT together with the
> RAIL++-M restrictions? ComfyUI's official SDPose templates use this file, and license-aware tools currently show
> it as "license unknown". Thank you!

**netflix/void-model**

> Title: Question about the license (CogVideoX-Fun base)
>
> Hello, and thank you for releasing VOID. The model card says apache-2.0, while the model is fine-tuned from
> CogVideoX-Fun-V1.5-5b-InP, whose transformer weights are under the CogVideoX License. Could you clarify whether the
> CogVideoX License terms also apply to `void_pass1` / `void_pass2`? ComfyUI's official template uses them, and
> license-aware tools show them as "license unknown" until this is clear. Thank you!

**joyfox/LTX-2.3-Transition-LORA**

> Title: License of the LoRA vs. LTX-2 Community License
>
> Hello, and thanks for the transition LoRA. The card says apache-2.0, but the LoRA is trained on Lightricks/LTX-2.3,
> which is under the LTX-2 Community License Agreement. Could you clarify whether the LTX-2 Community License also
> applies to this LoRA (and update the `license:` metadata if so)? A community "LTX-2.3 Style Transition" template
> in ComfyUI's template library uses it. Thank you!

**lrzjason/ObjectRemovalFluxFill**

> Title: License of the LoRA vs. FLUX.1 [dev] Non-Commercial License
>
> Hello, and thanks for the object-removal weights. The card says MIT, but the base model FLUX.1 Fill [dev] is under
> the FLUX.1 [dev] Non-Commercial License, which covers derivatives. Could you clarify which terms apply to
> `removal_timestep_alpha-2-1740.safetensors` and update the metadata if needed? ComfyUI's official OneReward
> template uses it. Thank you!

**bytedance-research/USO**

> Title: License of the USO weights vs. FLUX.1 [dev] Non-Commercial License
>
> Hello, and thank you for USO. The card says apache-2.0, but the DiT LoRA and projector are trained on
> black-forest-labs/FLUX.1-dev, whose license (FLUX.1 [dev] Non-Commercial License) covers derivatives. Could you
> clarify whether the FLUX.1 [dev] terms apply to `uso-flux1-dit-lora-v1` and `uso-flux1-projector-v1`? ComfyUI's
> official USO template uses them. Thank you!

**Jmica/Drama-3-GGUF**

> Title: License name differs from the original repo
>
> Hello, and thanks for the GGUF conversion. The metadata says `fish-audio-community-license-v1.0` and links to
> fishaudio/drama-3's LICENSE, but the original repo is labeled `fish-audio-research-license`. Could you check the
> original LICENSE and align the `license_name` here, so tools that compare the two can show the right terms?
> Thank you!

## B2. A license exists, but the card metadata is empty

The license text exists, so genmodelfit decides it with a table (`anima-official-loras` in `license_sources.py`).
We only ask for the metadata so that tools can read it automatically (not posting it causes no trouble; low priority).

| Upstream | State | Effect |
|---|---|---|
| [circlestone-labs/Anima-Official-LoRAs](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs) | [LICENSE.md](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs/blob/main/LICENSE.md) (CircleStone Labs Non-Commercial License v1.0, added on 2026-05-26) exists, but `license:` on the card is empty | Decided in the table as other (circlestone-labs-non-commercial-v1.0). It is non-commercial, so it is not in the automatic recommendation |

### Drafts (English)

**circlestone-labs/Anima-Official-LoRAs**

> Title: Add license metadata to the model card
>
> Hello, and thank you for the official Anima LoRAs and for adding LICENSE.md. The model card's metadata has no
> `license:` entry yet, so tools that read the Hub metadata show the LoRAs as "license unknown" until someone adds
> the license by hand. Could you add `license: other`, `license_name: circlestone-labs-non-commercial-v1.0` and
> `license_link: LICENSE.md` to the README front matter? Thank you!

## C. Repositories that cannot be read anonymously (401): check again once public

They are private (or gone), so no request can be posted. The weekly update (#19) checks whether they became public.

- `update` lists them in `sources.huggingface.unauthorized`.
- `update --inspect-only` sends the API request again and writes the result to `sources.huggingface.unauthorized_recheck` (`unauthorized` / `readable` / `failed`).
- Once one becomes public (readable), rebuilding (`update`) reads its license. If none is stated, move it to A and write a request.

| Repository | Effect | State on 2026-10-20 |
|---|---|---|
| [microsoft/Mage-Flow](https://huggingface.co/microsoft/Mage-Flow) | The upstream of Comfy-Org/Mage-Flow. The license of the 4 Mage-Flow items (t2i, turbo, edit, edit turbo) is unknown. No GitHub repository was found either | 401 |
| [microsoft/Lens](https://huggingface.co/microsoft/Lens) | The upstream of Comfy-Org/Lens. 2 items: Lens and Lens Turbo (and the gpt-oss text encoder). The GitHub code is MIT, but the weights have no statement | 401 |
| [Wan-AI/Wan2.2-Dancer-14B](https://huggingface.co/Wan-AI/Wan2.2-Dancer-14B) | The source table decides it from Wan-AI/Wan-Dancer-14B (apache-2.0), so it does not affect the unknown-license count | 401 |
| google/gemma-3-12b | **A name that does not exist** (Google's public list has only gemma-3-12b-it, -pt and the QAT versions; the anonymous API also returns 401 for names that do not exist). Split out as `unauthorized_missing`. It does not affect the unknown-license count | 401 (does not exist) |

## D. Cases with no draft

- **Civitai files** (realismByStableYogi v4 LCM, zimage-grainscape, ltx23-obscura_remova, SDXL skin and freckle LoRAs):
  - genmodelfit does not read Civitai ("Civitai: not used" in [sources.md](sources.md)), so the list cannot decide them.
  - Decide how to treat them before sending any request.
- **Items that only affect uncensored models** (not shown by the default sfw, so postponed):
  - No license stated:
    - ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8, ChrisColeTech/LTX-2.5-uncensored-v1.1-FP8 and ChrisColeTech/qwen-image-edit-uncensored-GGUF
    - Heartsync/NSFW-Uncensored
    - Meina/MeinaHentai
    - TenStrip/LTX2.3-10Eros
    - kkkkk26/lustify_v9
    - sinkinai/hassaku-hentai
  - More permissive than the base model's terms:
    - AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF
    - kpsss34/FHDR_Uncensored
  - No longer needed (checked against the list of 2026-10-22):
    - SulphurAI/Sulphur-2-base: no longer in the list (since 2026-10-20).
    - andro-flock/LUSTIFY-SDXL-NSFW-checkpoint-v2-0-INPAINTING (the copy of the same weights that the list has used since 2026-10-21): its license is now read as creativeml-openrail-m.
  - If they are posted, the A and B texts can be used.
