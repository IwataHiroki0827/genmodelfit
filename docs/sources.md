# データ源と一覧の作り方(`genmodelfit update`)

`genmodelfit update` は、下のデータ源から一覧を作り直して、キャッシュ(`db.cached_db_path()`)か `--output` の
場所に書き出す。Python からは `genmodelfit.update.build_database(fetch=...)`(書き出しはしない)。
同梱の `src/genmodelfit/data/db.json` は、実際に `update` を走らせた結果(ネットに繋がらないときの控え)。

持つのはモデルの**情報**だけで、重みは取らない・配らない。

## データ源

| データ源 | URL | 条件 | 使うもの |
|---|---|---|---|
| ComfyUI 公式テンプレートの一覧 | `https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/index.json` | MIT | 分類・名前・タグ・`openSource`・`size`・`usage`・`date`・`minComfyUIVersion`・`requiresCustomNodes`・`io.outputs` |
| テンプレートごとの JSON | `…/templates/<name>.json` | MIT | 全ノード(サブグラフ `definitions.subgraphs[].nodes` の中も)の `properties.models`(`name`・`url`・`hash`・`hash_type`・`directory`) |
| Hugging Face Hub API | `https://huggingface.co/api/models/{repo_id}?blobs=true`(版が main 以外なら `/revision/{rev}`) | HF の利用規約(公開 API を匿名で) | `cardData.license` / `license_name` / `license_link`(無ければ `license:` タグ)、`gated`、`siblings[].size` と `lfs.sha256`、`cardData.base_model`、`downloads`、`likes`、`lastModified`、`tags` |
| Hugging Face の検索 | `https://huggingface.co/api/models?filter=…&pipeline_tag=…&search=…&sort=downloads&direction=-1&limit=100` | 同上 | 無検閲のモデル、テンプレートに無い読み上げ・効果音のモデルの候補 |
| モデルカード | `https://huggingface.co/{repo_id}/raw/main/README.md` | 同上 | 除外の判定にだけ使う(保存しない) |

使わないもの: willitrunai.com などのサイトのデータ(CLAUDE.md)、Civitai(下)。

取得は `User-Agent: genmodelfit/<版> (+https://github.com/laxshmate/genmodelfit)` を名乗り、同時に 8 本まで、
時間切れ(接続 10 秒・読み込み 60 秒)とやり直し(接続の失敗・5xx は 4 回まで)を付ける。
HF の匿名の API は **5 分あたり 500 回まで**(応答の `ratelimit-policy: "fixed window";"api";q=500;w=300`)。
1 回の update で HF に 500〜600 回問い合わせるので、429 を受けたら `Retry-After` の秒数だけ全スレッドで待つ
(`sources.huggingface.rate_limited` に回数)。このため update は数分かかることがある。トークンは使わない。

## テンプレートの選び方

ローカルで動くものだけを取る:

- 名前が `api_` で始まるもの(クラウドの API を呼ぶテンプレート)は外す
- `openSource` が **false** のものは外す。`openSource` が**無い**ものはローカル扱い(古いテンプレートは書いていない)
- `size` が 0 か無いものは外す(重みを持たない。Chatterbox の読み上げなど、カスタムノードが自分で取るものもここで外れる)
- 分類 `LLM`・`Node Basics`・`Vector` は生成モデルではないので外す

一覧の ID はテンプレート名そのまま。テンプレートの JSON が読めないものは警告にして飛ばし、
**ローカルのテンプレートの 30% を超えて読めなければ**上流の形が変わったとみなして止める。

### 種類(modality)

| 分類 | modality |
|---|---|
| Image / Image Tools | image |
| Video / Video Tools | video |
| 3D Model | 3d |
| Audio | 名前・題・タグに `tts` / `text to speech` / `speech` / `voice cloning` → **tts**。名前・題・タグ・説明に `sound effect(s)` / `sfx` / `foley` があり `music` / `song` / `instrumental` が無い → **sfx**。それ以外 → **music** |
| それ以外(Character & Fashion・Brand & Design・Product & Ads) | `io.outputs[].mediaType`(video → video、audio → music、image → image)。無ければタグ(`Image to 3D` / `Text to 3D` → 3d、`Video` を含む → video、それ以外 → image) |

手で決めたもの(名前・タグから決められないもの): `audio_stable_audio_example`(Stable Audio Open 1.0)→ sfx。
モデルカードの用途が効果音・環境音(field recordings)中心のため。

### tasks(タグ → 略号)

| タグ | task | タグ | task |
|---|---|---|---|
| Text to Image | t2i | Image to Video | i2v |
| Image Edit / Style Reference / Layer Decompose | i2i | Text to Video | t2v |
| Inpainting | inpaint | Video Edit / Video to Video / Character Replacement | v2v |
| Outpainting | outpaint | FLF2V | flf2v |
| ControlNet | control | Reference to Video | ref2v |
| Image Upscale / Video Upscale | upscale | Audio to Video | a2v |
| Remove Background | matting | Lip Sync | lipsync |
| Relight | relight | Motion Control | motion |
| Depth / Normal / Albedo Estimation, Element Segmentation | estimate | Frame Interpolation | interp |
| Image to 3D | i23d | Text to 3D | t23d |
| Text to Audio / Text to Music | t2a | Music Cover / Audio Editing | a2a |
| Voice Cloning | vc | Text to Speech | tts |

当てはまるタグが無ければ種類の既定(image → t2i、video → t2v、3d → i23d、tts → tts、音 → t2a)。

### ファイル

`properties.models` を URL で重複を除いて全部入れる。役割(role)は `directory` から:
`diffusion_models` / `unet` / `checkpoints` → main、`text_encoders` / `clip` → text_encoder、`vae` → vae、
`clip_vision` → clip_vision、`loras` → lora、それ以外(`latent_upscale_models`・`model_patches` など)→ other。

sha256 はテンプレートの `hash` が `hash_type: SHA256` ならそれ、無ければ HF の `lfs.sha256`。
大きさは HF の `siblings[].size`。API の一覧に無いファイル(HF の URL だが別の版など)だけ HEAD で
`x-linked-size` / `content-length` を見る(`--skip-head` で飛ばす)。Civitai の URL には HEAD も送らない。

精度(precision)は本体のファイル名から(nvfp4 / fp4 / nf4 / fp8 / int8 / int4 / bf16 / fp16 / fp32 / gguf-<量子化>)、
パラメータ数(params_b)はファイル名・題の `14B` `1.3B` のような表記から推す(無ければ null)。

## ライセンス

- ファイルごとに、置き場所の HF リポジトリのライセンスを入れる(`license:` は HF の表記。不明は null)
- **再配布(リパッケージ)のリポジトリ**(持ち主が `Comfy-Org` / `comfyanonymous` / `Kijai`)は、本家とライセンスが
  違うことがある。`cardData.base_model` をたどって本家のライセンスをファイルに入れ、`license_from` に本家、
  `repo_license` に再配布側の表記を残す
  - `base_model` が 1 つならそれ。複数なら、ファイル名と語(英数字の塊。精度・形式の語は除く)が一番多く重なるもの
    (同点は全部を見て一番厳しいもの)。どれとも重ならなければ(Wan の再配布の中の umt5 など)再配布側の表記を使う
  - 本家の HF にライセンスの記載が無いときも、再配布側の表記を使う(どちらも無ければ null)
  - `base_model` を書いていない再配布リポジトリは、手で確かめた表(`hf.REPACKAGED_UPSTREAM`)で本家を決める:

    | 再配布 | 本家 |
    |---|---|
    | comfyanonymous/clip_vision_g | laion/CLIP-ViT-bigG-14-laion2B-39B-b160k(OpenCLIP ViT-bigG/14、mit) |

    2026-10-20 時点で base_model の無い再配布リポジトリは、ほかに Comfy-Org/Real-ESRGAN_repackaged・
    Comfy-Org/Wan-Animate-2・Comfy-Org/frame_interpolation・Kijai/HunyuanVideo_comfy・Kijai/LTXV2_comfy・
    Kijai/SUPIR_pruned があり、本家を確かめられていないので再配布側の表記のまま

    `comfyanonymous/flux_text_encoders`(t5xxl と clip_l)は本家が 2 つで、`openai/clip-vit-large-patch14` は HF に
    license の記載が無いため、表に入れず再配布側の表記(apache-2.0)を使う
- **一式のライセンス**は、本体・テキストエンコーダ・VAE のファイル(無ければ全ファイル)の中で**最も厳しいもの**:
  不明(null) > `other` > 制限付き(openrail 系・cc-by-nc・gpl・gemma など、下の寛容の一覧に無いもの) > 寛容
  (apache-2.0・mit・bsd 系・isc・zlib・unlicense・cc0-1.0・cc-by-*・bsl-1.0 など。`hf.PERMISSIVE`)。
  LoRA・clip_vision は一式の判定に入れない(差し替えられる部品のため。ファイルごとの license には入っている)
- `license_url` は一番厳しいファイルのライセンスを取ったリポジトリの `license_link`(無ければリポジトリのページ)

HF にライセンスの記載が無いリポジトリ(Kijai/WanVideo_comfy など)や、401 で読めないリポジトリ、Civitai の
ファイルを含む一式は license が null(不明)になる。自動のおすすめ(#3)では止まる側に倒れる。

## 無検閲(uncensored)のモデル

CLAUDE.md の決まり: 無検閲のモデルも一覧から外さず、`content: "uncensored"` と根拠(`content_evidence`)を付ける。

**テンプレート**: ID・題・タグに `uncensored` / `nsfw` / `abliterated` があれば `name:<語>`、使うファイルの HF
リポジトリに `not-for-all-audiences` タグがあれば `hf-tag:not-for-all-audiences (<repo>)`。
(2026-10-20 時点で当てはまるテンプレートは 0 件)

**Hugging Face の検索**: pipeline_tag(text-to-image・image-to-image・text-to-video・image-to-video・text-to-speech・
text-to-audio・image-to-3d・text-to-3d)ごとに、`filter=not-for-all-audiences` と `search=uncensored|nsfw|abliterated`
で、ダウンロード数の多い順に 100 件ずつ取る。根拠(`hf-tag:not-for-all-audiences` か名前の `name:<語>`)の
あるものだけを候補にし、次の順に絞る:

1. 除外の判定(下)に当たるものを外す(数だけ数える)
2. ダウンロード数が 100 未満のもの(写し・試作が多い)を外す
3. LoRA・LyCORIS・アダプタ・埋め込み(名前か `lora` / `peft` タグ)を外す。本体が無いと動かず、適否は本体で決まるため
4. 部品だけのもの(名前に text-encoder・VL・CLIP・VAE・GPTQ・AWQ・MLX など)を外す
5. 同じリポジトリ名の写し(持ち主違い)は一番ダウンロード数の多いものだけ。持ち主ごとに種類あたり 2 件まで
6. 種類ごとに**上位 30 件**まで(一覧を大きくしすぎない)。リポジトリの情報とモデルカードを取り、カードで
   除外に当たるもの・重みが選べないもの・本体が 30 MB 未満のもの(差分や中身の無いリポジトリ)を外す

ID は `hf:<owner>/<repo>`、`source: "huggingface"`、`discovered_from` に見つけた検索の URL、`downloads` / `likes` を入れる。
`recommend` の並べ方では、HF のダウンロード数を 100 で割ってテンプレートの使用回数(`usage`)と同じ目盛りにして使う。

**テンプレートで足りない種類**: ローカルの読み上げのテンプレートは 0 件(Chatterbox は `size` 0)、効果音は 1 件
(Stable Audio Open)しかないので、HF の人気順で補う(`content` は根拠が無ければ `sfw`):
読み上げ `pipeline_tag=text-to-speech` の上位 20 件、効果音 `pipeline_tag=text-to-audio` で `search=sfx|foley|sound|
audioldm|stable-audio-open` のうち効果音と判定したもの(名前・タグに sound effect / sfx / foley、または
MMAudio・AudioLDM・Stable Audio Open の系統で music の語が無いもの)の上位 10 件。

### HF のリポジトリから使うファイルを選ぶ規則(`hf.select_weight_files`)

見積もり(#4)はファイルの合計を使うので、**動かすのに要る 1 組だけ**を入れる(同じ部品の別の版を重ねない)。

1. 重みの拡張子(`.safetensors` `.gguf` `.ckpt` `.bin` `.pth` `.pt` `.sft`)で、大きさの分かるものだけ。
   `optimizer` `scheduler` `safety_checker` `feature_extractor` `onnx` `flax` `voices/` `samples/` などは除く
2. リポジトリ直下に**単一ファイルの本体**があれば、diffusers の形があってもそちらだけを使う(ComfyUI が読む形)
3. diffusers の形だけ(`model_index.json` があり直下に本体が無い)なら、部品のフォルダごとに 1 版:
   `transformer` / `transformer_2` / `unet` / `prior` / `decoder` → main、`text_encoder*` → text_encoder、`vae` → vae、
   `image_encoder` → clip_vision。safetensors を優先し、精度は fp16 → bf16 → 印の無い既定の版。分割(`-00001-of-00003`)はまとめて取る
4. それ以外は、ファイル名で役割(本体・テキストエンコーダ・VAE。`mmproj`・tokenizer・codec・LoRA は使わない)に分け、
   役割ごとに 1 版を選ぶ:
   - GGUF: Q8_0 → Q4_K_M → Q5_K_M → Q6_K → Q4_K_S → Q5_K_S の順、どれも無ければ一番小さいもの
   - それ以外: safetensors を優先(.bin / .ckpt / .pth は safetensors が無いときだけ)、fp32 の版は他が無いときだけ、
     残りから一番大きいもの(完全な本体)
   - 本体は、リポジトリ名と語が重なるファイルがあればその中から選ぶ(別のモデルの同梱を避ける)
   - テキストエンコーダ・VAE は、本体と同じフォルダ・直下・役割名のフォルダ(`text_encoders` `vae` など)にあるものだけ
   - 直下に本体が無く、本体が役割名でないフォルダ 4 つ以上に分かれているリポジトリは、別々のモデルの詰め合わせと
     みなして入れない
5. 本体が選べなければ一覧に入れない。`precision` は選んだ本体のファイル名から

規則で選ぶので、複数のモデルを詰め合わせたリポジトリなどでは外れることがある。

## 除外(未成年の性的表現を目的とするもの)

`src/genmodelfit/safety.py`。外したものは一覧に載せず、`sources.excluded` に**数だけ**を残す(名前も残さない)。

| 種類 | 語 | どこに当てるか |
|---|---|---|
| STRONG | loli(s)・lolicon・shota(s)・shotacon・jailbait・preteen(s)・pedo・pedophile・pedophilia・paedophile・toddlercon・underage | すべてのモデルの ID・名前・タグ、モデルカード・説明 |
| CONTEXT | child・children・kid(s)・minor(s)・teen(s)・schoolgirl(s)・schoolboy・little girl(s) / boy(s)・young girl(s) / boy(s)・toddler・infant・cub | 無検閲のモデルだけ。ID・名前・タグにあれば外す。カードでは同じ文に性的な語(sex・nude・naked・porn・erotic・nsfw・explicit・lewd・hentai など)があるときだけ |

- 語は英数字の境界で照合する(`minor` は `minority`、`kid` は `kidney`、`teen` は `fifteen` に当たらない。
  `Loli-XL` や `shota_v2` には当たる)
- カードの否定・禁止の文(no / not / never / must not / do not / prohibited / 18+ / adults only など)の中の語は数えない。
  「Do not use it to generate:」のように否定の行が「:」で終わるときは、続く箇条書き(禁止の一覧)も数えない。
  成人向けモデルの注意書き("No underage content" など)で正しいモデルを外さないため
- 公式テンプレート(無検閲の印が無いもの)には STRONG の語だけを当てる("children's book" などで外さないため)

2026-10-20 の実行では、候補の中で外れたものは 0 件だった(作例のプロンプトに child を含むもの 1 件・注意書きに
minors を含むもの 1 件を、規則を直す前は誤って外していた。直した規則で外れないことを確かめた)。

## Civitai: v0.1 では使わない

Civitai の利用規約(https://civitai.com/content/tos 、lastmod 2026-10-06)を読んで、使わないと決めた。

- 11.4: 自動のアクセスは「公開 API などを**自分の有効な認証情報で**、上限の範囲で」使うときだけ許される。
  利用者ごとの API キーが前提で、ツールが匿名で集めて配ることは想定されていない
- 6.1: サービスの利用許諾は「個人的・非商用の利用に限る」
- 7: サービスの「compilation, information, data」は Civitai の Materials。11.10 で Materials の販売・譲渡を禁止

genmodelfit の一覧はソース非公開の商用アプリ(AI生成キット)が同梱して配る前提なので、Civitai から集めた情報を
入れることは「はっきり許されている」とは言えない。テンプレートにある Civitai の URL(5 つ、のべ 6 か所)は URL だけを
そのまま持ち、大きさ・ライセンスは不明のまま(HEAD も送らない)。

## 上流の形が変わったとき・ネットに繋がらないとき

- index.json が一覧でない・分類に `templates` が無い・テンプレートに `name` が無い → `UpstreamFormatError`
  で止まり、終了コード 1。キャッシュはそのまま
- テンプレートの JSON に `nodes` が無い・読めない → 1 件ずつ警告(`sources.warnings`)。30% を超えたら止まる
- 一覧が空になった → 止まる
- ネットに繋がらない(index.json が取れない、テンプレートの JSON が全部取れない、HF の 30% を超えて繋がらない)
  → 前のキャッシュを残して終了コード 1。`recommend` などはキャッシュ(無ければ同梱の一覧)で動き続ける
- 実測(`measurements.json`)のあるモデルが上流から消えたときは、前の一覧(同梱)から引き継いで警告を出す

## テンプレートの `size` は一式の合計か(確かめた結果)

2026-10-20 の `update` の結果で、ファイルの大きさがすべて分かるテンプレート **263 件**について、index.json の
`size` と、テンプレートが使うファイル(HF の `siblings[].size`)の合計を比べた(`sources.size_check` にも同じ数)。

| 比べた結果 | 件数 |
|---|---:|
| バイト単位で一致 | 0 |
| ±2% 以内 | 179(68%) |
| ±10% 以内 | 214(81%) |
| size が合計より 10% 以上小さい | 22 |
| size が合計より 10% 以上大きい | 27 |
| size が 0.01 GiB の倍数(手で丸めた値) | 258 |

抜き出し(上 10 件は実測のあるものと代表、下 3 件は外れの例):

| テンプレート | index の size | ファイルの合計 | 比(size / 合計) |
|---|---:|---:|---:|
| image_z_image_turbo_int8 | 12,133,282,611(11.30 GiB) | 12,168,299,735(11.33 GiB) | 0.997 |
| image_z_image_turbo | 20,830,591,386(19.40 GiB) | 20,690,152,836(19.27 GiB) | 1.007 |
| sdxl_simple_example | 13,013,750,907(12.12 GiB) | 13,014,060,264(12.12 GiB) | 1.000 |
| image_sdxl_simple | 6,979,321,856(6.50 GiB) | 6,938,078,334(6.46 GiB) | 1.006 |
| flux_schnell | 17,233,556,275(16.05 GiB) | 17,236,328,572(16.05 GiB) | 1.000 |
| text_to_video_wan | 9,824,737,690(9.15 GiB) | 9,828,025,775(9.15 GiB) | 1.000 |
| video_wan2_2_5B_ti2v | 18,146,236,826(16.90 GiB) | 18,144,966,705(16.90 GiB) | 1.000 |
| audio_ace_step_1_5_checkpoint | 10,737,418,240(10.00 GiB) | 10,025,478,736(9.34 GiB) | 1.071 |
| audio_stable_audio_example | 5,690,831,667(5.30 GiB) | 5,745,535,406(5.35 GiB) | 0.990 |
| 3d_hunyuan3d_image_to_model | 4,928,474,972(4.59 GiB) | 4,928,151,562(4.59 GiB) | 1.000 |
| 3d_hunyuan3d-v2.1 | 4,928,474,972(4.59 GiB) | 7,365,943,290(6.86 GiB) | 0.669 |
| flux_fill_outpaint_example | 10,372,346,020(9.66 GiB) | 34,174,211,972(31.83 GiB) | 0.304 |
| templates_shane_change_any_objects | 10,737,418,240(10.00 GiB) | 46,495,526,446(43.30 GiB) | 0.231 |

結論:

- `size` は**一式(テンプレートが使うモデルファイル全部)の合計を GiB で丸めて手で書いた値**とみてよい
  (大半が ±2% に入り、ほぼ全部が 0.01 GiB 刻み)。ただしバイトの一致は 0 件で、目安にしかならない
- 外れの中身を見たもの:
  - 別のテンプレートの値の写し: `3d_hunyuan3d-v2.1` は 2.0 と同じ 4.59 GiB だが、2.1 の本体は 6.86 GiB
  - size が小さすぎる: `flux_fill_outpaint_example` は本体の FLUX.1 Fill(23.8 GB)だけで size の 9.66 GiB を超える。
    `templates_shane_change_any_objects` も本体が fp16 の 34.7 GB(size は 10 GiB。小さい精度を前提にした値かもしれない)
  - size が大きすぎる: カスタムノードが自分で重みを取り、`properties.models` に本体が書かれていないもの
    (`video-wan21_scail`・`templates-wan2_1_infinitetalk_music` は clip_vision の 1.26 GB しか書かれていない)
- このため genmodelfit は**ファイルの大きさの合計を使い、`size` はファイルが 1 つも書かれていないときの控えにだけ使う**
  (`ModelEntry.total_size_bytes`)。ただし本体(role main)が書かれていないテンプレートは合計が小さく出る
  (2026-10-20 で 10 件。MoGe など `geometry_estimation` に本体を置くものを含む)。見積もり(#4)で扱いを決める
