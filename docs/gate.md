# 自動でおすすめに出してよいかの判定(gate)

使う側(AI生成キット)は、**人が確かめずに**おすすめのモデルを PC 初心者に出す。そのため判定は保守的にし、
分からないもの(ライセンス不明・HEAD 未確認・版不明など)はすべて「保留」にする。全部の条件を満たしたものだけ
`auto: true`。保留でも一覧・`recommend`(`--auto-only` を付けないとき)・`fit`・`info` には今までどおり出る。

実装は `src/genmodelfit/gate.py`。結果は一覧(db.json)に持たず、毎回計算する。一覧に持つのは `update` が
記録した HEAD の結果と ComfyUI の最新の版だけ。

## 使い方

```powershell
genmodelfit gate [--json] [--modality image] [--only-auto | --only-held]   # 条件ごとの件数
genmodelfit recommend --auto-only --vram 8                                  # 自動で出せるものだけ
genmodelfit info image_z_image_turbo                                        # 条件ごとの結果と理由
# 方針の差し替え(recommend / fit / info / gate に共通)
--license-allowlist mit,apache-2.0      # または 1 行に 1 つ書いたファイル(# から後は注釈)
--comfyui-version 0.3.60                # 使う ComfyUI の版(既定は update が記録した最新のリリース)
--allow-uncensored-auto                 # 無検閲のモデルも自動で出してよい
```

```python
from genmodelfit.db import load
from genmodelfit.gate import GatePolicy, evaluate_offline

db = load()
policy = GatePolicy(license_allowlist=frozenset({"mit", "apache-2.0"}), comfyui_version="0.3.60")
r = evaluate_offline(db.get("image_z_image_turbo"), policy, sources=db.sources)
r.auto, r.held_by, r.to_dict()
```

`to_dict()` の形: `{"auto": bool, "held_by": [落ちた条件の id], "comfyui_version": 比べた版,
"checks": [{"id", "passed", "detail"}]}`。`checks` は下の表の順にいつも 9 つ。`held_by[0]` が表の `held:<id>`。
`detail` は人が読む日本語(文言は変わりうる。機械は `id` と `passed` を見る)。

## 条件

| id | 通る条件 | 保留になる例 |
|---|---|---|
| `license` | すべてのファイル(LoRA・アップスケーラなども)のライセンスが許可リストにある。ライセンスは下の「ファイルのライセンスの読み方」で決める | `other`・openrail 系・非商用・コミュニティライセンス・llama / gemma の規約・不明 |
| `gated` | どのファイルも `gated: false` | `true`、不明(`null`) |
| `official_workflow` | ComfyUI 公式テンプレート(`source: comfyui-template`)で `openSource` が false でない | Hugging Face だけの項目 |
| `custom_nodes` | テンプレートの `requiresCustomNodes` が空 | カスタムノードが要る、テンプレートが無い |
| `comfyui_version` | テンプレートの `minComfyUIVersion` ≦ 使う版(数字で比べる。`0.3.9 < 0.3.10`) | 版が読めない・記載なし・使う版が不明 |
| `url_reachable` | すべてのファイルの URL が、匿名の HEAD(転送を追わない)で 200 / 302 / 307 を返した記録がある | 401(gated)・404・記録なし(未確認)・URL なし |
| `sha256` | すべてのファイルに 64 桁の 16 進の sha256 がある | Civitai のファイルなど |
| `content` | 無検閲(`content: uncensored`)でない | 無検閲(`--allow-uncensored-auto` で外せる) |
| `size_known` | すべてのファイルの大きさが分かり、ファイルの一覧が完全(`files_complete` が false でない) | 大きさ不明、ファイルの記載なし |

ファイルが 1 つも書かれていない項目は、ファイルについての条件がすべて保留になる(空なら通る、にはしない)。

`openSource` の記載が無いテンプレート(SDXL・FLUX.1 schnell・Wan 2.1 などの古いもの、2026-10-20 で 53 件)は
ローカルで動くテンプレートとして通す(`update` も同じ扱い。`api_` で始まる API のテンプレートと
`openSource: false` は `update` が一覧に入れていない)。

## 許可リスト(既定)

`apache-2.0`・`mit`・`bsd-2-clause`・`bsd-3-clause`・`cc0-1.0`・`cc-by-4.0`・`unlicense`。

選び方: **商用で使えて、使い方の制限(用途の禁止の一覧など)・売上や利用者数の基準・申請が無い**もの。
これらの条件(著作権表示・ライセンス文を残す)は、重みを**再配布するとき**にかかる。genmodelfit と
使う側のアプリは重みを配らず、利用者の PC が Hugging Face から直接取るので、アプリ側で守ることが残らない。

- `cc-by-4.0` を入れた理由: 条件は「共有するときの表示(帰属)」だけで、商用も改変も許され、使い方の制限が無い。
  表示の義務がかかるのは MIT の著作権表示と同じく再配布(Share)のとき。重みを自分で配るアプリは、MIT と同様に
  表示を足す必要がある(その場合は許可リストから外して使う)。2026-10-20 の一覧に cc-by-4.0 のファイルは無い
- 入れないもの:
  - `openrail` / `openrail++` / `creativeml-openrail-m` などの RAIL: 用途の制限(Attachment A)を利用者に
    引き継ぐ義務がある。人が条件を読んで、アプリの規約に入れる必要がある
  - 非商用(`cc-by-nc-*`・`flux-1-dev-non-commercial-license` など): 販売するアプリでは使えない
  - 売上・利用者数の基準つきのコミュニティライセンス(`stabilityai-ai-community`・`stable-audio-community`・
    `tencent-hunyuan-community`・`ltx-2-community-license`・MiniMax のものなど): 基準を超えたら別の契約が要り、
    表示や地域の条件が付くものもある
  - `llama*`・`gemma`: 独自の規約(利用者数の基準・用途の制限・表示)
  - `other`(上のどれでもない独自のもの)・不明: 読んでいないので通さない。`other` は `license_name` で照合するので、
    人が読んで問題ないと判断したもの(例: `mit-and-apache-2.0`)は許可リストに名前を足せば通せる

## ファイルのライセンスの読み方

一覧の `license` は、再配布のリポジトリ(Comfy-Org・Kijai など)の表記をそのまま持つことがある。再配布の中には
別のモデルの部品(テキストエンコーダなど)が入っていて、本当のライセンスはリポジトリの表記と違うことがある。
gate は一覧を書き換えずに、判定の中で次の順に読む:

1. **ファイル名が制限付きの部品**なら、一覧の値より優先してその規約で読む(保留になる):
   - `llama` / `llava` → Llama の規約(https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct の `llama3.1`)
   - `gemma` 1〜3・`t5gemma` → Gemma 利用規約(https://ai.google.dev/gemma/terms)。
     Gemma 4 は Apache-2.0(https://ai.google.dev/gemma/docs/gemma_4_license)なので当てない
2. ライセンスを**再配布側の表記からしか読めていない**(`license_from` が再配布のリポジトリ)なら、確かめられない
   として保留。ただし出どころの分かる部品は、その本家のライセンスで読む:

   | ファイル名 | ライセンス | 本家 |
   |---|---|---|
   | `umt5_xxl…` | apache-2.0 | https://huggingface.co/google/umt5-xxl |
   | `t5xxl…`・`t5-v1_1…`・`t5-base…` | apache-2.0 | https://huggingface.co/google/t5-v1_1-xxl |
   | `clip_l…` | mit | https://github.com/openai/CLIP |
   | `clip_g…` | mit | https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k |
   | `clip_vision_h…` | mit | https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K |
   | `wan_2.1_vae…`・`Wan2_1_VAE…` | apache-2.0 | https://huggingface.co/Wan-AI/Wan2.1-T2V-14B |

3. それ以外は一覧の値(本家から読んだもの)

この読み方で保留になった例(2026-10-20 の一覧):

- `hidream_i1_*`: 本体は MIT だが、テキストエンコーダ `llama_3.1_8b_instruct_fp8_scaled` は Llama 3.1 の規約
  (一覧では再配布側の表記 mit)
- `audio_minimax_music_3`: 再配布 Comfy-Org/MiniMax-Music-3 の表記は apache-2.0 だが、本家の LICENSE は
  MiniMax-Music3 COMMUNITY LICENSE(商用の画面に名前を出す・年の売上 2,000 万ドル超は許可が要る・AUP)。
  本家のカードに `license:` の記載が無いため、一覧は再配布側の表記になっていた
- ERNIE-Image・MOSS-Image・Mage-Flow・Boogu・Wan-Animate-2・Wan-Dancer など: 再配布側の表記だけで、
  部品(`ministral-3-3b`・`qwen3vl_*`・`flux2-vae` など)の本家を確かめられない

## URL の確かめ方(`url_reachable`)

`genmodelfit update` が、すべてのファイルの Hugging Face の URL に匿名の HEAD を送り(転送は追わない)、
応答コードを `files[].head_status`、日時を `head_checked_at` に記録する。同じ URL は 1 回だけ、同時に 8 本まで、
429 は `Retry-After` を守って待つ。Civitai など HF 以外には送らない(未確認として保留)。

2026-10-20 に実際に送って確かめた応答:

| ファイル | 応答 | 転送先 |
|---|---|---|
| LFS(Xet)のファイル(重みのほぼ全部。小さい safetensors も) | 302 | `https://us.aws.cdn.hf.co/xet-bridge-us/…`。`x-linked-size`・`x-linked-etag`(sha256)付き |
| LFS でない小さいファイル(`model_index.json` など) | 307 | `/api/resolve-cache/models/…`(相対) |
| リポジトリの名前が変わったもの(`runwayml/stable-diffusion-v1-5` など) | 307 | 新しいリポジトリの `/resolve/…`(相対) |
| gated のリポジトリ | 401 | なし |
| 無いファイル | 404 | なし |

このため 200・302・307 を「取れる」とみなす。一覧の 522 URL(2026-10-20)は 302 が 493・307 が 6・401 が 23。
HEAD は HF の "resolvers" の枠(応答の `ratelimit-policy: "fixed window";"resolvers";q=3000;w=300`。
API の 500 回 / 5 分とは別)で数えられるので、1 回の update で使い切ることは無い。

一覧を作り直さずに HEAD と ComfyUI の版だけをやり直すとき:

```powershell
genmodelfit update --heads-only [--db PATH] [--output PATH]   # 既定はキャッシュの一覧を読んでキャッシュに書く
```

## ComfyUI の版(`comfyui_version`)

`update` が `https://api.github.com/repos/Comfy-Org/ComfyUI/releases/latest` を認証なしで 1 回だけ取り、
`tag_name` を `sources.comfyui.latest_release` に記録する(2026-10-20 は `v0.42.1`)。取れなければ `null` と警告。
使う側のアプリが同梱する ComfyUI の版が分かっているなら、`--comfyui-version` / `GatePolicy(comfyui_version=…)`
で指定する(記録された版より優先)。

## 無検閲のモデル(`content`)

自動のおすすめは初心者向けなので、無検閲のモデルは既定で保留にする。一覧から外すわけではなく、
`recommend --content uncensored`・`fit`・`info` などの一覧・絞り込みには今までどおり出る。
