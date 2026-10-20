# ライセンスの記載のお願い(下書き)

重みのライセンスが分からない(記載が無い・元のモデルの規約と食い違う)ため、genmodelfit が「ライセンス不明」として
扱っているモデルの、本家の持ち主へのお願いの下書き(Issue #30)。

- **まだどこにも出していない。出すのはオーナー(laxshmate)の確認の後**(他人のリポジトリに書き込むため)。
  出すときは、Hugging Face のリポジトリなら Community の Discussion、GitHub なら Issue に、下の英文をそのまま使う
- 一覧は同梱の一覧(`src/genmodelfit/data/db.json`、2026-10-20)のライセンス不明のファイルから作り、本家ごとにまとめた。
  決め方は docs/sources.md「ライセンス」。記載が入ったら `license_sources.py` の表か本家の表記で一覧を作り直す
- 利用者への影響(共通): genmodelfit はライセンス不明を「許されていない」と同じに扱う(docs/schema.md「License
  contract」)。`recommend` / `fit` / `search` には既定で出さず(`--include-unknown-license` で出る)、自動のおすすめ
  (gate)にも出さない。ソースを公開せずに配るアプリ(AI生成キットなど)は、そのモデルを利用者に勧められない

## A. 重みにライセンスの記載が無い

| 本家 | 足りないもの | 影響する項目(無検閲を除く) |
|---|---|---|
| [circlestone-labs/Anima-Official-LoRAs](https://huggingface.co/circlestone-labs/Anima-Official-LoRAs) | モデルカードにライセンスが無い(`anima-turbo-lora-v0.2.safetensors`) | Anima の公式テンプレート 4 件(base v1、LLLite の any / depth / inpainting) |
| [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | モデルカードにライセンスが無い | MiniMax H3 の i2v・続き・t2v の 3 件 |
| [yl4579/StyleTTS2-LJSpeech](https://huggingface.co/yl4579/StyleTTS2-LJSpeech) | ライセンスが無い。Kokoro-82M(apache-2.0)の base_model なので、Kokoro も不明になる | 読み上げ Kokoro-82M |
| [xtuner/llava-llama-3-8b](https://huggingface.co/xtuner/llava-llama-3-8b) | ライセンスが無い(Llama 3 から作った重み)。HunyuanVideo のテキストエンコーダ `llava_llama3_*` の元 | HunyuanVideo の文章から動画 1 件 |
| [ByteZSzn/Flux.2-Turbo-ComfyUI](https://huggingface.co/ByteZSzn/Flux.2-Turbo-ComfyUI) | ライセンスが無い(`Flux2TurboComfyv2`。FLUX.2 [dev] の LoRA) | FLUX.2 fp8 の 1 件 |
| [google-research/frame-interpolation](https://github.com/google-research/frame-interpolation)(FILM) | コードは Apache-2.0。重みは Google Drive にあり、重みのライセンスの記載が無い(Comfy-Org/frame_interpolation の `film_net_fp16`) | フレーム補間(FILM)1 件 |
| [RT-DETRs/RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4) | コードは Apache-2.0。重みは DINOv3 を教師にした蒸留で、重みのライセンスの記載が無い(Comfy-Org/SDPose の `rt_detr_v4-x-hgnet_*`) | SDPose の複数人・SAM 3D Body・MiniMax H3 の ControlNet の 4 件 |

### 下書き(英語)

**circlestone-labs/Anima-Official-LoRAs**

> Title: License for the LoRA weights
>
> Hello, and thank you for releasing the official Anima LoRAs. The model card does not state a license for the
> weights (e.g. `anima-turbo-lora-v0.2.safetensors`). ComfyUI's official Anima templates use this file, and tools
> that check model licenses before recommending them (such as genmodelfit) currently have to treat it as
> "license unknown", so it is hidden from users. Could you add a `license:` entry to the model card (or a LICENSE
> file) saying which terms apply — for example the same license as the Anima base model? Thank you!

**lightx2v/Minimax-h3-Turbo**

> Title: License for the Turbo weights
>
> Hello, and thank you for the Minimax-h3-Turbo release. The model card does not mention a license for these
> weights. ComfyUI's official MiniMax H3 templates (image-to-video, continuation, text-to-video) download
> `minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors` from this repo, and license-aware tools have to treat
> it as "license unknown". Could you state the license in the model card metadata (`license:`), e.g. whether it
> follows the base MiniMax H3 model's terms? Thank you very much.

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

## B. 表記が、元のモデルの規約より寛容(食い違う)

本家の表記はあるが、元のモデル(base)の規約より寛容なので、表記どおりとは言えず不明にしたもの。

| 本家 | 表記 | 元のモデル | 影響する項目(無検閲を除く) |
|---|---|---|---|
| [jingheya/lotus-depth-d-v1-1](https://huggingface.co/jingheya/lotus-depth-d-v1-1) | apache-2.0 | Stable Diffusion 2 base(CreativeML Open RAIL++-M) | Lotus の深度を使う 5 件(Lotus Depth・FLUX の深度 LoRA・Qwen Image ControlNet・SD3.5 Large の深度・LTX-2 の深度から動画) |
| [teemosliang/SDPose-Wholebody](https://huggingface.co/teemosliang/SDPose-Wholebody) | mit | Stable Diffusion 2 の U-Net(CreativeML Open RAIL++-M) | SDPose の 4 件と MiniMax H3 の ControlNet 1 件 |
| [netflix/void-model](https://huggingface.co/netflix/void-model) | apache-2.0 | CogVideoX-Fun-V1.5-5b-InP(Transformer は CogVideoX License) | VOID の動画の消去 1 件 |
| [joyfox/LTX-2.3-Transition-LORA](https://huggingface.co/joyfox/LTX-2.3-Transition-LORA) | apache-2.0 | Lightricks/LTX-2.3(LTX-2 Community License) | LTX-2.3 の場面転換 1 件 |
| [lrzjason/ObjectRemovalFluxFill](https://huggingface.co/lrzjason/ObjectRemovalFluxFill) | mit | FLUX.1 Fill [dev](FLUX.1 [dev] Non-Commercial License) | FLUX.1 Fill dev OneReward 1 件 |
| [bytedance-research/USO](https://huggingface.co/bytedance-research/USO) | apache-2.0 | FLUX.1 [dev](FLUX.1 [dev] Non-Commercial License) | FLUX.1 dev USO 1 件 |
| [Jmica/Drama-3-GGUF](https://huggingface.co/Jmica/Drama-3-GGUF) | other(fish-audio-community-license-v1.0) | fishaudio/drama-3(fish-audio-research-license) | 読み上げ Drama-3 GGUF |

### 下書き(英語)

**jingheya/lotus-depth-d-v1-1**(同じ文を SDPose にも使える。「Lotus」→「SDPose」、「apache-2.0」→「MIT」)

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
> applies to this LoRA (and update the `license:` metadata if so)? ComfyUI's official style-transition template uses
> it. Thank you!

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

## C. 匿名で読めない(401)リポジトリ — 公開されたら確かめ直す

非公開(または消えた)ので、依頼は出せない。毎週の更新(#19)で公開されたかを確かめる:
`update` が `sources.huggingface.unauthorized` に並べ、`update --inspect-only` が API を送り直して
`sources.huggingface.unauthorized_recheck`(`unauthorized` / `readable` / `failed`)に書く。
公開されたら(readable)、作り直し(`update`)でライセンスを読み、記載が無ければ A に移して依頼を書く。

| リポジトリ | 影響 | 2026-10-20 の状態 |
|---|---|---|
| [microsoft/Mage-Flow](https://huggingface.co/microsoft/Mage-Flow) | Comfy-Org/Mage-Flow の本家。Mage-Flow の 4 件(t2i・turbo・edit・edit turbo)のライセンス不明。GitHub も見つからない | 401 |
| [microsoft/Lens](https://huggingface.co/microsoft/Lens) | Comfy-Org/Lens の本家。Lens・Lens Turbo の 2 件(gpt-oss のテキストエンコーダも)。GitHub のコードは MIT だが重みの記載が無い | 401 |
| [Wan-AI/Wan2.2-Dancer-14B](https://huggingface.co/Wan-AI/Wan2.2-Dancer-14B) | 出どころの表で Wan-AI/Wan-Dancer-14B(apache-2.0)から決めているので、ライセンス不明には効かない | 401 |
| [google/gemma-3-12b](https://huggingface.co/google/gemma-3-12b) | 系統をたどる途中の名前。ライセンス不明には効かない | 401 |

## D. 下書きを作らないもの

- **Civitai のファイル**(realismByStableYogi v4 LCM、zimage-grainscape、ltx23-obscura_remova、SDXL の肌・そばかすの LoRA):
  Civitai は v0.1 では読まない(docs/sources.md「Civitai」)ので、一覧の側で決められない。依頼の前に扱いを決める
- **無検閲のモデルだけに効くもの**(既定の sfw では出ないので後回し): ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8・
  LTX-2.5-uncensored-v1.1-FP8・qwen-image-edit-uncensored-GGUF、Heartsync/NSFW-Uncensored、Meina/MeinaHentai、
  SulphurAI/Sulphur-2-base、TenStrip/LTX2.3-10Eros、TheImposterImposters/lustifySDXLNSFW_v20-inpainting、
  kkkkk26/lustify_v9、sinkinai/hassaku-hentai(記載が無い)、AtomicChat/Qwen-Image-2.1-Turbo-Uncensored-GGUF・
  kpsss34/FHDR_Uncensored(元の規約より寛容)。出すなら A・B の文を使える
