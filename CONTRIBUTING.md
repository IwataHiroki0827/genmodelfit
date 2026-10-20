# 実測を送る / Contributing measurements

genmodelfit の見積もりは、実際の PC で測った値(実測)があるとそれを優先します。いろいろな GPU の実測が
集まるほど、見積もりが正確になります。`genmodelfit bench` で測った結果を Pull Request で送ってください。

**English summary:** Run `genmodelfit bench <template> --url http://127.0.0.1:<port>` against your own ComfyUI,
check the JSON it writes, then open a pull request that adds the record to
`src/genmodelfit/data/measurements.json`. Required fields: `model_id`, `gpu_name`, `gpu_vram_gb`, `peak_vram_gb`,
`settings`, `seconds`, `driver`, `comfyui_version`, `offload`, `outcome`, `standard`, `source`, `date`. Do not include
paths, user names, host names or prompts. One model per PR is easiest to review. Out-of-memory runs are welcome too,
but they must keep `outcome: "oom"` (bench prints them and does not save them unless you pass `--save-failed` or
`--output`).

**License of contributions:** genmodelfit (code and bundled data) is under MPL-2.0. By opening a pull request you
agree that your contribution (code, measurements or other data) is licensed under MPL-2.0 like the rest of the
project. There is no CLA.

## 1. 測る

1. ComfyUI を起動します(ふだん使っているもので構いません)。**ほかの生成が動いていないとき**に測ってください。
   bench はキューに仕事があると測りません(開始前と、送る直前の 2 回見ます)。GPU の使用率(約 2 秒の平均)が
   30% 以上のときも止まります。`nvidia-smi` が無い・ComfyUI が別の PC・同じ名前の GPU が複数あってどれか
   分からない、のように GPU の空きを確かめられないときも、黙って進まずに止まります(`--force` で測れます)
2. テンプレートが使うモデルのファイルを ComfyUI の `models` フォルダに置きます。足りないときは bench が
   どのファイルが無いかを表示します
3. 測ります

```powershell
pip install genmodelfit
genmodelfit bench image_z_image_turbo_int8 --url http://127.0.0.1:8188 --resolution 768x768 --steps 6
genmodelfit bench text_to_video_wan --url http://127.0.0.1:8188 --resolution 512x512 --frames 49 --steps 20
genmodelfit bench audio_ace_step_1_5_checkpoint --url http://127.0.0.1:8188 --length 20 --steps 8
genmodelfit bench 3d_hunyuan3d_image_to_model --url http://127.0.0.1:8188 --steps 15
```

`--url` は必ず指定します(既定の URL は持ちません。間違えてほかの ComfyUI に送らないため)。

- `<template>` は ComfyUI 公式テンプレートの名前です(`genmodelfit info <名前>` で一覧の ID を確かめられます)。
  ワークフローは ComfyUI が持つテンプレート、無ければ GitHub の公式テンプレートから作ります。
  自分のワークフローで測るときは、ComfyUI の「API 形式で書き出し」で保存した JSON を `--workflow` で渡します
- 設定は `--resolution 幅x高さ`(または `--width` `--height`)、`--frames`、`--length 秒`(音声)、`--steps` で変えます。
  ほかの入力は `--set 入力の名前=値` で変えられます(同じ名前の入力すべてが変わります)。
  `--set` で大きさ・長さ・ステップ数・seed 以外を変えた結果と、`--workflow` の結果は「標準でない」
  (`standard: false`)として記録され、見積もりには使われません。`--set` の値は、数値だけを `note` に残します
  (文字列はプロンプトのことがあるので、名前だけを残します)
- 既存の実測(`src/genmodelfit/data/measurements.json`)と同じ設定で測ると比べやすくなります
- 既定では `/free` を送りません(ほかの人が読み込んだモデルを外さないため)。読み込み済みのモデルがあると、
  その分が結果に混ざることがあるので、bench は注意を出し、`note` に開始前の量を残します。
  **自分だけが使う ComfyUI なら `--free` を付けてください**(モデルを外し、前の結果のキャッシュも消してから、
  読み込みから測ります)
- `--free` を付けずに前回と同じ内容を送ると、ComfyUI が前の結果を返して何も実行しないので、bench は測れなかったと
  表示します。一部のノードだけが前の結果を使ったときは、その分のメモリを測れていないので「標準でない」になります
- 測っている間に ComfyUI でほかの仕事が動いた、送ってから実行が始まるまで 10 秒以上かかった、というときも
  「標準でない」になります
- 時間切れ(`--timeout 秒`、既定 1800)や Ctrl+C のときは、自分のプロンプトだけを取り消します
  (実行中なら prompt_id を付けて中断し、待っている間ならキューから消します。ほかの人の仕事は止めません)
- 入力の画像が要るテンプレート(画像から 3D など)は、公式テンプレートの画像を ComfyUI にアップロードします
  (上書きはしません。同じ名前の別の画像があれば ComfyUI が別の名前を付けます)
- 保存ノードの出力は ComfyUI の output フォルダの `genmodelfit_bench/` に入ります。要らなければ消してください

### 何を測っているか

| 項目 | 測り方 |
|---|---|
| `peak_vram_gb` | 生成中に `/system_stats` を 0.2 秒ごとに読み、torch が確保した量(`torch_vram_total`)の最大値。ComfyUI のプロセスの分で、CUDA の文脈(約 0.3〜0.5 GB)は含みません |
| `note` の「GPU 全体の増分」 | `nvidia-smi` の使用量の最大値から開始前の値を引いたもの。ほかのアプリの増減も入ります |
| `peak_ram_gb` | ポートで待ち受けている ComfyUI のプロセス(と子プロセス)の常駐メモリの最大値。コマンドラインに `main.py` か `comfy` を含むプロセスだけを数えます。ComfyUI が別の PC のとき・プロセスを特定できないときは空です |
| `seconds` | ワークフローを送ってから終わるまで(モデルの読み込みを含む)。ComfyUI が数えた実行時間は `note` に入ります |
| `offload` | ComfyUI のログから読みます: `dynamic`(動的 VRAM)/ `partial`(一部だけ GPU に載せた)/ `lowvram`(`--lowvram` などの起動オプション)/ `none`(全部載った)/ `oom`(メモリ不足で失敗)。分からなければ空 |
| `gpu_vram_gb` | `/system_stats` の VRAM の総量を GiB で、小数 2 桁まで丸めた値(公称 8 GB の GPU なら 8.0 前後)。同梱の古い行は公称の値(`16` など)です |
| `outcome` | `ok`(最後まで動いた)/ `oom`(メモリ不足で失敗)/ `error`(ほかの理由で失敗) |
| `standard` | 公式テンプレートのまま(大きさ・長さ・ステップ数・seed だけを変えた)で、ほかの仕事が混ざらず、全部のノードを実行したか |

メモリ不足(OOM)のデータも歓迎します(「この GPU では動かない」という大事なデータです)。ただし必ず
`outcome: "oom"` のまま送ってください。bench は失敗した結果を表示するだけで、キャッシュには保存しません
(キャッシュの実測は、その PC の見積もりにすぐ使われるため)。保存するときは `--save-failed`(キャッシュへ)か
`--output パス` を付けます。「標準でない」結果も同じで、`--save-nonstandard` か `--output` で保存できます。
見積もりは `outcome` が `ok` で `standard` が `true` の実測だけを使います。

## 2. 確かめる

bench は、最後まで動いた標準の結果を `<キャッシュ>/measurements/<モデル>_<GPU>_<日時>.json` に保存し、
保存先を表示します(`--output` で場所を変えられます)。キャッシュに置いた実測は、その PC の `genmodelfit` の見積もりにすぐ使われます。

ファイルを開いて、次を確かめてください。

- **個人の情報が入っていないこと**: フォルダのパス、ユーザー名、PC の名前、プロンプトの文を入れないでください。
  bench はこれらを書きませんが、`--source` や `--note` に自分で書いたものは確かめてください
- `gpu_name` と `driver` が自分の PC と合っていること
- `settings` が測りたかった設定になっていること

## 3. 送る

1. このリポジトリをフォークして、ブランチを作ります
2. 保存された JSON の `measurements` の中身を、`src/genmodelfit/data/measurements.json` の `measurements` の
   最後に足します(書き方は既存の行にそろえてください)
3. `source` に、測った環境が分かる短い説明を書きます(例: `"genmodelfit bench, RTX 4060 8GB, ComfyUI portable"`)。
   個人名やメールアドレスは不要です
4. `pytest -q` が通ることを確かめて、Pull Request を作ります。説明に、PC の構成(GPU・メモリ・OS)と、
   ComfyUI の起動オプション(`--lowvram` など、変えたものだけ)を書いてください

### 必要な項目

| 項目 | 必須 | 例 |
|---|---|---|
| `model_id` | ○ | `image_z_image_turbo_int8`(一覧の ID) |
| `gpu_name`, `gpu_vram_gb` | ○ | `NVIDIA GeForce RTX 3070`, `8.0` |
| `peak_vram_gb` | ○ | `5.43` |
| `settings` | ○ | `{"width": 768, "height": 768, "steps": 6}` |
| `seconds` | ○ | `41.2` |
| `driver`, `comfyui_version` | ○ | `566.03`, `0.40.0` |
| `offload` | ○ | `dynamic` |
| `outcome`, `standard` | ○ | `ok`, `true`(OOM なら `oom` のまま送る) |
| `source`, `date` | ○ | 上の例、`2026-10-20` |
| `peak_ram_gb`, `note` | | bench が書いたものをそのまま |

形の正本は `src/genmodelfit/models.py` の `Measurement` で、説明は [docs/schema.md](docs/schema.md) にあります。

## そのほかの変更

コードの変更も歓迎します。依存を足すときは MIT / Apache-2.0 / BSD のものだけにしてください
(このツールを組み込むアプリがソースを公開せずに配布するため)。ComfyUI のコード(GPL-3.0)は import せず、
HTTP だけで話します。新しい `.py` ファイルの先頭には、ほかのファイルと同じ MPL-2.0 の表示(3 行)を入れてください。

## ライセンス / License

genmodelfit のコードと同梱のデータ(db.json・実測・係数)は [MPL-2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) です。
送っていただく修正・実測・データも、**MPL-2.0 で出していただいたものとして受け入れます**
(Pull Request を出すことで、これに同意したことになります)。CLA(著作権の譲渡や追加の許諾の書類)はありません。
自分で書いたもの、または MPL-2.0 で出してよいものだけを送ってください。

Contributions are accepted under MPL-2.0 (inbound = outbound). No CLA is required. Only submit work you wrote
yourself or are allowed to release under MPL-2.0.
