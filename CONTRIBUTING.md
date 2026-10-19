# 実測を送る / Contributing measurements

genmodelfit の見積もりは、実際の PC で測った値(実測)があるとそれを優先します。いろいろな GPU の実測が
集まるほど、見積もりが正確になります。`genmodelfit bench` で測った結果を Pull Request で送ってください。

**English summary:** Run `genmodelfit bench <template> --url http://127.0.0.1:<port>` against your own ComfyUI,
check the JSON it writes, then open a pull request that adds the record to
`src/genmodelfit/data/measurements.json`. Required fields: `model_id`, `gpu_name`, `gpu_vram_gb`, `peak_vram_gb`,
`settings`, `seconds`, `driver`, `comfyui_version`, `offload`, `source`, `date`. Do not include paths, user names,
host names or prompts. One model per PR is easiest to review.

## 1. 測る

1. ComfyUI を起動します(ふだん使っているもので構いません)。**ほかの生成が動いていないとき**に測ってください。
   bench はキューに仕事があると測りません。GPU の使用率が高いときも止まります(`--force` で無視できます)
2. テンプレートが使うモデルのファイルを ComfyUI の `models` フォルダに置きます。足りないときは bench が
   どのファイルが無いかを表示します
3. 測ります

```powershell
pip install git+https://github.com/laxshmate/genmodelfit
genmodelfit bench image_z_image_turbo_int8 --url http://127.0.0.1:8188 --resolution 768x768 --steps 6
genmodelfit bench text_to_video_wan --url http://127.0.0.1:8188 --resolution 512x512 --frames 49 --steps 20
genmodelfit bench audio_ace_step_1_5_checkpoint --url http://127.0.0.1:8188 --length 20 --steps 8
genmodelfit bench 3d_hunyuan3d_image_to_model --url http://127.0.0.1:8188 --steps 15 --set octree_resolution=192
```

- `<template>` は ComfyUI 公式テンプレートの名前です(`genmodelfit info <名前>` で一覧の ID を確かめられます)。
  ワークフローは ComfyUI が持つテンプレート、無ければ GitHub の公式テンプレートから作ります。
  自分のワークフローで測るときは、ComfyUI の「API 形式で書き出し」で保存した JSON を `--workflow` で渡します
- 設定は `--resolution 幅x高さ`(または `--width` `--height`)、`--frames`、`--length 秒`(音声)、`--steps` で変えます。
  ほかの入力は `--set 入力の名前=値` で変えられます(同じ名前の入力すべてが変わります)
- 既存の実測(`src/genmodelfit/data/measurements.json`)と同じ設定で測ると比べやすくなります
- 測る前に、その ComfyUI のモデルを `/free` で解放します(毎回同じ条件で、読み込みから測るため)。
  ほかの人が使っている ComfyUI では測らないでください。`--no-free` で解放せずに測れますが、前回と同じ内容だと
  ComfyUI が前の結果を返して何も実行しないので、bench は測れなかったと表示します
- 入力の画像が要るテンプレート(画像から 3D など)は、公式テンプレートの画像を ComfyUI にアップロードします
- 保存ノードの出力は ComfyUI の output フォルダの `genmodelfit_bench/` に入ります。要らなければ消してください

### 何を測っているか

| 項目 | 測り方 |
|---|---|
| `peak_vram_gb` | 生成中に `/system_stats` を 0.2 秒ごとに読み、torch が確保した量(`torch_vram_total`)の最大値。ComfyUI のプロセスの分で、CUDA の文脈(約 0.3〜0.5 GB)は含みません |
| `note` の「GPU 全体の増分」 | `nvidia-smi` の使用量の最大値から開始前の値を引いたもの。ほかのアプリの増減も入ります |
| `peak_ram_gb` | ポートで待ち受けている ComfyUI のプロセス(と子プロセス)の常駐メモリの最大値。ComfyUI が別の PC のときは空です |
| `seconds` | ワークフローを送ってから終わるまで(モデルの読み込みを含む)。ComfyUI が数えた実行時間は `note` に入ります |
| `offload` | ComfyUI のログから読みます: `dynamic`(動的 VRAM)/ `partial`(一部だけ GPU に載せた)/ `none`(全部載った)/ `oom`(メモリ不足で失敗) |

メモリ不足(OOM)で失敗したときも、そこまでの最大値を `offload: "oom"` として保存します。これも大事なデータです。

## 2. 確かめる

bench は結果を `<キャッシュ>/measurements/<モデル>_<GPU>_<日時>.json` に保存し、保存先を表示します
(`--output` で場所を変えられます)。キャッシュに置いた実測は、その PC の `genmodelfit` の見積もりにすぐ使われます。

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
| `source`, `date` | ○ | 上の例、`2026-10-20` |
| `peak_ram_gb`, `note` | | bench が書いたものをそのまま |

形の正本は `src/genmodelfit/models.py` の `Measurement` で、説明は [docs/schema.md](docs/schema.md) にあります。

## そのほかの変更

コードの変更も歓迎します。依存を足すときは MIT / Apache-2.0 / BSD のものだけにしてください
(このツールを組み込むアプリがソースを公開せずに配布するため)。ComfyUI のコード(GPL-3.0)は import せず、
HTTP だけで話します。
