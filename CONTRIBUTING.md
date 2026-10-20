# 実測を送る / Contributing measurements

genmodelfit の見積もりは、実際の PC で測った値(実測)があるとそれを優先します。いろいろな GPU の実測が
集まるほど、見積もりが正確になります。`genmodelfit bench` で測った結果を Pull Request で送ってください。

**English summary:** Run `genmodelfit bench <template> --url http://127.0.0.1:<port> --submission data/contrib/`
against your own ComfyUI (in a clone of your fork). It writes a submission file
`data/contrib/<gpu>/<date>-<sha8>.json` (`contrib_schema` 1: GPU name / VRAM / generation / driver, ComfyUI version,
OS family, RAM rounded to GiB, the model, settings and results). Check exactly what will be shared with
`genmodelfit contrib show <file>` (it also runs a privacy checklist), then open a pull request that adds only that file.
**genmodelfit itself never sends anything**; sharing happens only through the pull request you open. The file never
contains your host name, user name, paths, GPU UUID, PCI address, IP, CPU name or exact time. CI runs
`genmodelfit contrib validate` on the pull request. Out-of-memory runs are welcome too (`outcome: "oom"`).
Already have a saved bench result? `genmodelfit contrib make --from <file>` converts it.

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

## 2. 提出用の JSON を作る

自分のフォーク(GitHub の Fork)を手元に clone し、そのフォルダで bench に `--submission` を付けて測ります。

```powershell
git clone https://github.com/<あなたの名前>/genmodelfit
cd genmodelfit
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 832x832 --submission data/contrib/
```

- `--submission` にフォルダを渡すと、`data/contrib/<GPU>/<日付>-<sha8>.json` という名前で書きます
  (例 `data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json`。`sha8` は中身の sha256 の先頭 8 桁)。ファイル名を渡せばその名前で書きます
- 前に測って保存した結果(`<キャッシュ>/measurements/….json` や `--output` のファイル)からも作れます:
  `genmodelfit contrib make --from <bench の結果> --output data/contrib/`
  (ComfyUI が別の PC だったときは `--no-pc-info` を付けると、この PC の OS・メモリ・GPU の世代を入れません)
- 失敗(OOM)の結果も、そのまま出してください。標準でない条件の結果は記録として受け付けますが、まとめには使いません

### 入るもの・入らないもの

| 入るもの | 例 |
|---|---|
| GPU の名前・VRAM の総量・世代・compute capability・ドライバの版 | `NVIDIA GeForce RTX 3070`, `8.0`, `ampere`, `8.6`, `566.03` |
| ComfyUI の版、OS の種類(版・ビルドは入れない)、メインメモリ(GiB の整数に丸める) | `0.40.0`, `windows`, `32` |
| モデル ID・設定・結果・最大 VRAM / メモリ・時間・載せ方・測り方 | `image_sdxl_simple`, `832x832`, `ok`, `6.9` … |
| 日付(UTC の日付だけ) | `2026-10-21` |
| メモ(note。bench が書いた測り方の説明) | パス・URL・PC 名・ユーザ名・メール・UUID・IP・時刻らしい文字を含む文は、文ごと消す |

**入らないもの**: PC の名前、ユーザー名、フォルダのパス、GPU の UUID、PCI の差し場所、シリアル番号、IP / MAC アドレス、
CPU の名前、時刻、bench の `--source` に書いた文、プロンプト。決まった項目以外は CI の検査で断ります。
ただし自動の点検は文字の形を見るだけで、note に普通の言葉で書いた名前などは見分けられません(下の 3 で自分で読んでください)。

## 3. 中身を確かめる(送る前に必ず)

```powershell
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json
```

ファイルの中身をすべて読める形で出し、個人の情報の点検(パス・PC 名・ユーザ名・UUID・PCI・IP・URL・メール・時刻・
知らない項目が無いか。この PC の名前・ユーザ名が入っていないかも見ます)の結果を `[ok]` / `[NG]` で出します。

- `[NG]` があれば、直してから出してください(手で書き換えたら、名前の `sha8` が変わるので `contrib validate` が
  正しい名前を教えます)
- **note(メモ)は自分で読んでください**。自動の点検は文の意味までは分かりません。気になる文は消して構いません
- `genmodelfit contrib validate data/contrib` で、CI と同じ検査を手元で動かせます
- 「要確認」(review)は、式の見積もりから 40% より外れた値・全部 GPU に載せたのに重みより小さい値・一覧に無いモデル・
  知らない GPU です。誤りではありませんが、PR を見る人が確かめるまで受け付けません(測り直すか、PR の説明に事情を書いてください)

## 4. 送る(Pull Request)

1. 足すのは `data/contrib/<GPU>/<日付>-<sha8>.json` のファイルだけにします(ほかのファイルは変えない。コードの変更は別の PR に)
2. コミットして自分のフォークに push し、Pull Request を作ります。説明に、ComfyUI の起動オプション
   (`--lowvram` など、変えたものだけ)を書いてもらえると助かります。個人名やメールアドレスは不要です
3. CI(`Contrib` のワークフロー)が `genmodelfit contrib validate` で形・値の範囲・個人の情報・重複・ファイル名を検査し、
   結果をジョブのまとめ(Summary)に出します。検査は、PR の向き先(main)の genmodelfit で動きます

**genmodelfit は何も送りません**。共有されるのは、あなたが自分の GitHub で出した Pull Request の中身だけです
(自動で PR を作る機能はありません)。マージされると、提出は MPL-2.0 で公開されます(下の「ライセンス」)。

### 受け付けたあと

- 作者が `genmodelfit contrib aggregate --output src/genmodelfit/data/measurements_community.json` で、同じモデル・設定・GPU の
  提出をまとめます(中央値と MAD で外れ値を除く)。同じ構成の提出が **2 件以上そろい、OOM が無いと**、その構成の見積もりの
  根拠が `measured_community`(ほかの利用者の実測)になります。1 件だけの提出は、もう 1 件そろうまで見積もりには使いません。
  ほかの利用者の実測は、必要量を**上げる向きにだけ**使います(見積もりより小さい値では、段階も点数も変わりません)
- 毎週の点検(`contrib aggregate --report`)が、校正の誤差の表を作り直します(docs/accuracy.md)

### 作者の実測(measurements.json)

同梱の `src/genmodelfit/data/measurements.json` は作者が測った実測です。項目は docs/schema.md の「Measurement」、
提出の形は「contrib」にあります。利用者の実測は data/contrib に出してください(measurements.json は直接変えない)。

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
