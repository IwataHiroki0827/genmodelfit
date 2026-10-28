# 実測を送る

[English](CONTRIBUTING.md) · **日本語**

genmodelfit の見積もりは、実際の PC で測った値(実測)があるとそれを優先します。いろいろな GPU の実測が集まるほど、見積もりが正確になります。
このページは、実測を送りたい人向けです(`genmodelfit measure` と `genmodelfit share` で、または Pull Request で)。そのほかの変更の決まりも載せています。
用語は [README.ja.md](README.ja.md) に合わせています。

## 目次

- [かんたんな送り方(git は要らない)](#かんたんな送り方git-は要らない)
- [慣れた人向け: bench と Pull Request](#慣れた人向け-bench-と-pull-request)
  - [1. 測る](#1-測る)
  - [2. 提出用の JSON を作る](#2-提出用の-json-を作る)
  - [3. 中身を確かめる](#3-中身を確かめる)
  - [4. Pull Request で送る](#4-pull-request-で送る)
- [そのほかの変更](#そのほかの変更)
- [ライセンス](#ライセンス)

流れの要約: いちばんかんたんなのは **コマンド 2 つとボタン 1 回** です: `genmodelfit measure` で測り、`genmodelfit share` で結果を入れた GitHub のフォームを開き、「Create」を押します(下の「[かんたんな送り方](#かんたんな送り方git-は要らない)」。git もフォークも要りません)。
git に慣れていれば、フォークの clone の中で `genmodelfit bench … --submission data/contrib/` を実行し、`genmodelfit contrib show` で中身を確かめ、そのファイルだけを足した Pull Request を出すこともできます(「[2. 提出用の JSON を作る](#2-提出用の-json-を作る)」から)。
**genmodelfit 自身は何も送りません**。共有は、あなたがフォームで「Create」を押したときか、あなたが出す Pull Request を通してだけ行われます。
今の NVIDIA の bench の提出(ComfyUI の動的 VRAM、`vram_metric: torch_reserved`)は作業用メモリだけを持つので、段ごとの山を測れるようになるまでは照合だけに使い、見積もりにも段階にも使いません(#100。[accuracy.ja.md](docs/accuracy.ja.md#実測の使い方))。

## かんたんな送り方(git は要らない)

`measure` と `share` は 0.11.0 から使えます。[uv](https://docs.astral.sh/uv/) があれば、入れずにそのまま動かせます(`uvx genmodelfit measure`。無ければ `pip install -U genmodelfit`)。

```powershell
genmodelfit measure
genmodelfit share
```

**用意するもの:** GitHub のアカウント(無料)、この PC で起動している ComfyUI(NVIDIA・AMD・Intel・Apple の GPU)、
1 つのモデルにつき 1〜5 分ほど(ComfyUI にまだ入っていないモデルは、落とす時間も)。git・フォーク・プログラミングは
要りません。

1. **`genmodelfit measure`**: 一覧からモデルを番号で選び、Enter で測ります。
2. **`genmodelfit share`**: 測った結果を選ぶと、中身を入れた GitHub のフォームがブラウザで開きます。
3. **「Create」を押す。** 数分でボットが確かめて、そのページに返事を書きます。

1 つずつ詳しく:

1. ComfyUI を起動します(ふだん使っているものでかまいません)。ほかの生成は止めておきます。測る前の注意は「[1. 測る](#1-測る)」を見てください。
2. `genmodelfit measure` が、測ると役に立つモデルを出します: この PC の GPU に載り(GPU で good 以上)、この GPU の実測がまだ無いもの。並びは、ComfyUI にもう入っているもの → 時間が短いもの → 落とす量が少ないもの。番号で選びます(`q` でやめる)。
   - 自分でモデルを決めるなら `genmodelfit measure "Z-Image-Turbo"`(名前・ワークフローの ID・その一部。`download` と同じ選び方)。
   - この PC で起動中の ComfyUI(127.0.0.1 の 8188・8000)を探します。ほかの ComfyUI なら `--url http://<ホスト>:<ポート>` を付けます。起動していなければ、起動のしかたを案内します。
3. 測る前に 1 画面で、モデルと精度、設定(公式のテンプレートのまま)、どの ComfyUI か・どう見つけたか、モデルのファイルが入っているか、時間の目安、何も送らないことが出ます。`Y`(Enter)で測ります。
   - 時間は目安です: 同じモデルのほかの GPU の実測の中央値、無ければ種類ごとのおおよその値。落とす時間は入りません。
   - 入っていないモデルは、`genmodelfit download` と同じ確かめ・同意の決まりで先に落とします。
   - ほかのプログラムが GPU を使っているときは測りません(`bench` と同じ)。
4. 結果と保存した場所(`<キャッシュ>/submissions/<GPU>/<日付>-<sha8>.json`)を出し、最後に `genmodelfit share` を 1 行だけ案内します。送るかどうかは尋ねません。
5. `genmodelfit share` が、まだ送っていない実測を出すので、番号で選びます。送る中身のすべてと、個人の情報の点検が出ます(GPU の名前・VRAM・OS の種類・モデル・設定・測った値。PC やユーザの名前は入りません)。**note(メモ)も含めて自分で読んでください**。
6. 中身を入れた GitHub の実測のフォームがブラウザで開きます。GitHub にログインし(アカウントが要ります)、**「Create」** を押します。
7. ボットが、`genmodelfit contrib validate` と同じ検査と個人の情報の点検をします。通れば `data/contrib/` にファイルを足す Pull Request を作って Issue にリンクを書き、メンテナがマージします(そのとき Issue は閉じます)。通らなければ理由をコメントするので、直した中身で Issue の本文を編集すると、もう一度確かめます。

- 点検に `[NG]` があるか、検査の誤りがあると、`share` は何も開かずに止まります。直し方は「[3. 中身を確かめる](#3-中身を確かめる)」にあります。
- 中身が長くて URL に入りきらないときは、`share` が中身をクリップボードに写し(OS の道具 clip・pbcopy・wl-copy・xclip・xsel を使います)、中身の無いフォームを開きます。「Measurement (JSON)」の欄に貼り付けて「Create」を押してください。クリップボードの道具が無ければ、写す中身を 2 つの `---` の行の間に出します。
- ページを開いたら、この PC でその実測に送った印を付け、次から一覧に出しません(`genmodelfit share --all` で出します)。`genmodelfit share <ファイル>` で、指定した提出のファイル(`bench --share`・`contrib make` で作ったものも)を送れます。
- 端末でないとき・`--json`・`--print-url` のときは、`share` はブラウザを開かず URL を出すだけです。
- 送るために genmodelfit が通信することはありません。ブラウザでページを開くだけで、中身が送られるのは「Create」を押したときだけです。
- Issue を書いた人のアカウントを Pull Request とコミットに書きます。1 つの Issue が 1 つの Pull Request なので、自分で出す Pull Request と同じく 1 つの出どころとして数えます(「[受け付けたあと](#受け付けたあと)」)。

## 慣れた人向け: bench と Pull Request

ここからは、git を使う人や、`genmodelfit bench` のすべてのオプション(自分のワークフロー・ほかの設定・`--force`)を
使いたい人向けです。何を測り、何を確かめるかの詳しい説明もここにあります(かんたんな送り方にも当てはまります)。

### 1. 測る

1. ComfyUI を起動します(ふだん使っているもので構いません)。**ほかの生成が動いていないとき**に測ってください。
   - bench はキューに仕事があると測りません(開始前と、送る直前の 2 回見ます)。
   - GPU の使用率(約 2 秒の平均)が 30% 以上のときも止まります。
   - 使用率は、NVIDIA なら `nvidia-smi`、AMD なら `amd-smi` か `rocm-smi`(Linux)、Intel なら `xpu-smi`(Linux)、Windows の AMD・Intel は Windows の性能カウンタ、Apple は `ioreg` で読みます。
   - GPU の空きを確かめられないときは、黙って進まずに止まります(`--force` で測れます)。例: NVIDIA で `nvidia-smi` が無い、同じ名前の GPU が複数あってどれか分からない、ComfyUI が別の PC。
   - NVIDIA 以外で使用率を読む道具が無いときは、止まらずに測ります。そのときは ComfyUI のキューが空なことと、ComfyUI 以外のアプリが GPU のメモリを半分より多く使っていないことだけを確かめ、注意と `note` に残します。ほかのアプリ(ゲーム・動画の書き出しなど)は閉じてから測ってください。
   - **AMD・Intel・Apple の GPU の実測は特に歓迎します**。まだ実測が無いので、表の PC の行に「✓ 確認済みの種類」が付きません(段階は NVIDIA と同じ基準で決めます)。2 機種以上の実測が集まると、その種類の GPU に ✓ が付き、見積もりも詰められます。
2. テンプレートが使うモデルのファイルを ComfyUI の `models` フォルダに置きます。
   - 足りないときは、bench がどのファイルが無いかを表示します。
   - `genmodelfit download <テンプレート名> --comfyui-dir <ComfyUI のフォルダ>` で、一覧のファイルを ComfyUI の models に落とせます(sha256 で確かめます)。ライセンスが分からないファイルと gated のファイルは落としません。
3. 測ります:

```powershell
pip install -U "genmodelfit>=0.9.1"
genmodelfit bench image_z_image_turbo_int8 --url http://127.0.0.1:8188 --resolution 1024x1024 --steps 6
genmodelfit bench text_to_video_wan --url http://127.0.0.1:8188 --resolution 832x480 --frames 81 --steps 20
genmodelfit bench audio_ace_step_1_5_checkpoint --url http://127.0.0.1:8188 --length 30 --steps 8
genmodelfit bench 3d_hunyuan3d_image_to_model --url http://127.0.0.1:8188 --steps 15
```

`--url` は必ず指定します(既定の URL は持ちません。間違えてほかの ComfyUI に送らないため)。

テンプレートと設定:

- `<template>` は ComfyUI 公式テンプレートの名前です(`genmodelfit info <名前>` で一覧の ID を確かめられます)。
- ワークフローは ComfyUI が持つテンプレート、無ければ GitHub の公式テンプレートから作ります。
- 自分のワークフローで測るときは、ComfyUI の「API 形式で書き出し」で保存した JSON を `--workflow` で渡します。
- 設定は `--resolution 幅x高さ`(または `--width` `--height`)、`--frames`、`--length 秒`(音声)、`--steps` で変えます。
- ほかの入力は `--set 入力の名前=値` で変えられます(同じ名前の入力すべてが変わります)。
- 次の 2 つの場合、結果は「標準でない」(`standard: false`)として記録され、見積もりには使われません: `--set` で大きさ・長さ・ステップ数・seed 以外を変えたとき、`--workflow` を使ったとき。
- `--set` の値は、数値だけを `note` に残します(文字列はプロンプトのことがあるので、名前だけを残します)。
- **種類ごとの既定の設定**(画像 1024x1024、動画 832x480・81 コマ、音楽 30 秒、効果音・読み上げ 10 秒)で測ると、設定を指定しない判定にそのまま使われます(0.11.0 から、指定が無ければどのモデルもこの設定で判定します。[docs/scoring.ja.md](docs/scoring.ja.md)「判定に使う設定」)。ほかの設定の実測も、見積もりをずらすのに使われます。
- 入力の画像から大きさが決まるテンプレート(深度推定・背景除去など。大きさの入力が無い)は、`--resolution` を付けずに測ります。bench は大きさを書きません。メンテナが確かめた表(`estimate.NO_SIZE_INPUT`)にあるテンプレートなら、そのまま判定に使われます。表に無いテンプレートの大きさの無い提出には、PR で `settings_missing` の注意が出ます(メンテナが表に足すかを決めます)。

モデルの読み込みとキャッシュ:

- 既定では `/free` を送りません(ほかの人が読み込んだモデルを外さないため)。
- 読み込み済みのモデルがあると、その分が結果に混ざることがあります。そのときは bench が注意を出し、`note` に開始前の量を残します。
- **自分だけが使う ComfyUI なら `--free` を付けてください**。モデルを外し、前の結果のキャッシュも消してから、読み込みから測ります。
- `--free` を付けずに前回と同じ内容を送ると、ComfyUI が前の結果を返して何も実行しないので、bench は測れなかったと表示します。
- 一部のノードだけが前の結果を使ったときは、その分のメモリを測れていないので「標準でない」になります。
- 測っている間に ComfyUI でほかの仕事が動いた、送ってから実行が始まるまで 10 秒以上かかった、というときも「標準でない」になります。

時間切れとファイル:

- 時間切れ(`--timeout 秒`、既定 1800)や Ctrl+C のときは、自分のプロンプトだけを取り消します。実行中なら prompt_id を付けて中断し、待っている間ならキューから消します。ほかの人の仕事は止めません。
- 入力の画像が要るテンプレート(画像から 3D など)は、公式テンプレートの画像を ComfyUI にアップロードします。上書きはしません。同じ名前の別の画像があれば ComfyUI が別の名前を付けます。
- 保存ノードの出力は ComfyUI の output フォルダの `genmodelfit_bench/` に入ります。要らなければ消してください。
- ComfyUI の動的 VRAM(既定)では、VRAM がぎりぎりの設定で ComfyUI が応答しなくなることがあります(ログに aimdo の `budget_deficit` が出たあと進まない。#102 の RTX 5080 16GB で Wan の 81 コマ)。そのときは bench の時間切れでも取り消せないので、自分で起動した ComfyUI だけを止め、`--disable-dynamic-vram` を付けて起動し直して測ってください。

#### 何を測っているか

| 項目 | 測り方 |
|---|---|
| `peak_vram_gb` | 生成中に `/system_stats` を 0.2 秒ごとに読み、torch が確保した量(`torch_vram_total`)の最大値。ComfyUI のプロセスの分で、CUDA の文脈(約 0.3〜0.5 GB)は含みません。`torch_vram_total` が取れないときは GPU 全体の増分で代え、警告を残します |
| `note` の「GPU 全体の増分」 | GPU 全体の使用量(`nvidia-smi`。NVIDIA 以外は上の道具、約 1 秒ごと)の最大値から開始前の値を引いたもの。ほかのアプリの増減も入ります。道具が無ければ書きません |
| `peak_ram_gb` | ポートで待ち受けている ComfyUI のプロセス(と子プロセス)の常駐メモリの最大値。コマンドラインに `main.py` か `comfy` を含むプロセスだけを数えます。ComfyUI が別の PC のとき・プロセスを特定できないときは空です |
| `seconds` | ワークフローを送ってから終わるまで(モデルの読み込みを含む)。ComfyUI が数えた実行時間は `note` に入ります |
| `offload` | ComfyUI のログから読みます: `dynamic`(動的 VRAM)/ `partial`(一部だけ GPU に載せた)/ `lowvram`(`--lowvram` / `--novram` で起動した)/ `none`(全部載った)/ `oom`(メモリ不足で失敗)。分からなければ空 |
| `gpu_vram_gb` | `/system_stats` の VRAM の総量を GiB で、小数 2 桁まで丸めた値(公称 8 GB の GPU なら 8.0 前後)。同梱の古い行は公称の値(`16` など)です |
| `outcome` | `ok`(最後まで動いた)/ `oom`(メモリ不足で失敗)/ `error`(ほかの理由で失敗) |
| `standard` | 公式テンプレートのまま(大きさ・長さ・ステップ数・seed だけを変えた)で、ほかの仕事が混ざらず、全部のノードを実行したか |

メモリ不足(OOM)のデータも歓迎します(「この GPU では動かない」という大事なデータです)。ただし必ず `outcome: "oom"` のまま送ってください。

- bench は失敗した結果を表示するだけで、キャッシュには保存しません(キャッシュの実測は、その PC の見積もりにすぐ使われるため)。
- 保存するときは `--save-failed`(キャッシュへ)か `--output パス` を付けます。
- 「標準でない」結果も同じで、`--save-nonstandard` か `--output` で保存できます。
- 見積もりは `outcome` が `ok` で `standard` が `true` の実測だけを使います。

### 2. 提出用の JSON を作る

ここからは、git とフォークの clone で送る方法です(「[かんたんな送り方](#かんたんな送り方git-は要らない)」で送ったなら、ここは読まなくて構いません。入るもの・入らないものの表は同じです)。

自分のフォーク(GitHub の Fork)を手元に clone し、そのフォルダで bench に `--submission` を付けて測ります。clone したフォルダで `pip install -e .` すると、CI と同じ main の版で `contrib validate` を流せます。

```powershell
git clone https://github.com/<あなたの名前>/genmodelfit
cd genmodelfit
pip install -e .
genmodelfit bench image_sdxl_simple --url http://127.0.0.1:8188 --resolution 1024x1024 --submission data/contrib/
```

- 書くのは提出ファイル(`contrib_schema` 1)です。`--submission` にフォルダを渡すと、`data/contrib/<GPU>/<日付>-<sha8>.json` という名前で書きます。例: `data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json`(`sha8` は中身の sha256 の先頭 8 桁(16 進))。ファイル名を渡せばその名前で書きます。
- フォルダは末尾に `/` を付けて渡してください(`data/contrib/`)。最初の提出ではまだフォルダが無く、`/` が無いと `data/contrib` という名前のファイルになります。
- 前に測って保存した結果(`<キャッシュ>/measurements/….json` や `--output` のファイル)からも作れます: `genmodelfit contrib make --from <bench の結果> --output data/contrib/`
- ComfyUI が別の PC だったときは `--no-pc-info` を付けると、この PC の OS・メモリ・GPU の世代を入れません。
- 失敗(OOM)の結果も、そのまま出してください。標準でない条件の結果は記録として受け付けますが、まとめには使いません。

#### 入るもの・入らないもの

| 入るもの | 例 |
|---|---|
| GPU の名前・VRAM の総量・世代・compute capability・ドライバの版(世代と compute capability は NVIDIA だけ) | `NVIDIA GeForce RTX 3070`, `8.0`, `ampere`, `8.6`, `566.03` |
| GPU の会社と PyTorch の backend(0.9.0 の bench から) | `amd` と `rocm`、`intel` と `xpu`、`apple` と `mps`、`nvidia` と `cuda` |
| GPU の PCI の型番 ID(0.9.0。AMD・Intel で ComfyUI が同じ PC のときだけ。機種の番号で、個体・差し場所は表さない) | `1002:744c`(Radeon RX 7900 XTX。同じ機種ならどの PC でも同じ) |
| Apple の GPU はチップ名(0.9.0。system_profiler の名前の形に合うものだけ) | `Apple M2 Pro` |
| ComfyUI の版、OS の種類(版・ビルドは入れない)、メインメモリ(GiB の整数に丸める) | `0.40.0`, `windows`, `32` |
| モデル ID・設定・結果・最大 VRAM / メモリ・時間・載せ方・測り方 | `image_sdxl_simple`, `832x832`, `ok`, `6.9` … |
| 日付(UTC の日付だけ) | `2026-10-21` |
| メモ(`note`。bench が書いた測り方の説明) | パス・URL・PC 名・ユーザ名・メール・UUID・IP・時刻らしい文字を含む文は、文ごと消す |

**入らないもの**: PC の名前、ユーザー名、フォルダのパス、GPU の UUID、PCI の差し場所(バス番号。型番 ID とは別物)、シリアル番号、IP / MAC アドレス、CPU の名前、時刻、bench の `--source` に書いた文、プロンプト。
決まった項目以外は CI の検査で断ります。
ただし自動の点検は文字の形を見るだけで、意味は分かりません。次のようなものは見逃します。手順 3 で `note` を自分で読んでください。
- 普通の言葉で書いた名前(「山田太郎の PC で測った」、`user: yamada`、`my name is ...`)
- 形の決まっていない PC の名前(`yamada-desktop`、`yamada-ws`)
- 1 字ずつ空けた文字(`t a r o @ g m a i l`)・短い base64(`dGFybw==`)・語に見える base64
- 住所・郵便番号・ほかのサービスの ID(Discord・Steam など)
- 公開のドメインの下のユーザ名(`github.com/taro`・`hf.co/yamada`。ドメインは公開の置き場として扱うため)

逆に、ComfyUI のノードの名前(`HunyuanVideo15ImageToVideo` など)、40 桁の git のコミット・64 桁の sha256、公開のドメイン(huggingface.co・hf.co・github.com・comfy.org・pytorch.org・nvidia.com・python.org)は個人の情報として扱いません。

### 3. 中身を確かめる

送る前に必ず行ってください。

```powershell
genmodelfit contrib show data/contrib/rtx-3070/2026-10-21-1a2b3c4d.json
```

ファイルの中身をすべて読める形で出し、個人の情報の点検の結果を `[ok]` / `[NG]` で出します。点検では、パス・PC 名・ユーザ名・UUID・PCI・IP・URL・メール・時刻・知らない項目が無いかを見て、この PC の名前・ユーザ名が入っていないかも見ます。

- `[NG]` があれば、直してから出してください。手で書き換えたら名前の `sha8` が変わるので、`contrib validate` が正しい名前を教えます。
- **`note`(メモ)は自分で読んでください**。自動の点検は文の意味までは分かりません。気になる文は消して構いません。
- `genmodelfit contrib validate data/contrib` で、CI と同じ検査を手元で動かせます。
- 失敗(誤り)にするのは、機械的な検査(形・値の範囲・個人の情報・重複・ファイル名)だけです(llmfit と同じ考え方。2026-10-22)。
- 「確認」(notices)は失敗にはせず、表示するだけです。対象は次のとおりです: 式の見積もりから 40% より外れた値、全部 GPU に載せた(`offload: none`)のに下限(重み + 固定分)より小さい値、同じ GPU の PC(VRAM は提出の値と、同じ名前で VRAM の違う版の値。それぞれメモリ 16 / 32 / 64 GB)で段階を変える値、一覧に無いモデル、知らない GPU・NVIDIA の名前の形に合わない GPU 名(`gpu_name_form`)、名前の公称の量に合わない VRAM(`gpu_vram_nominal`。改造品など)。
- GPU 名で断るのは、名前の代わりの値(空・n/a・none・null・unknown・-)と汎用の名前(NVIDIA Graphics Device など)だけです(0.11.0。表に無い新しい GPU も受け付けます)。
- 「確認」はジョブのまとめの上に出し、PR をマージする人が見ます。事情があれば PR の説明に書いてください(測り直しをお願いすることがあります)。

### 4. Pull Request で送る

1. 足すのは `data/contrib/<GPU>/<日付>-<sha8>.json` のファイルだけにします。
   - ほかのファイルは変えません(コードの変更は別の PR に)。
   - data/contrib の外のファイルを変えた PR は、`Contrib` のワークフローが失敗にします。
   - 提出の PR は**足すだけ**です。data/contrib の既にあるファイルを消す・書き換える・名前を変える PR、`<GPU>/<日付>-<sha8>.json`(小文字の `.json`)でない名前・場所のファイルを足す PR、提出を 1 件も足さない PR も失敗にします(#90)。
   - 前に出した提出を直したいときは、直したものを新しいファイルとして出し、PR の説明に古いファイルの名前と理由を書いてください。古いファイルはメンテナが外します(docs/release.ja.md「次からの版を出す手順」)。
2. ブランチを作ってコミットし、自分のフォークに push します:

   ```powershell
   git switch -c contrib/<GPU>-<モデル>
   git add data/contrib/<GPU>/<日付>-<sha8>.json
   git commit -m "contrib: <GPU> <モデル>"
   git push -u origin HEAD
   ```

   GitHub で laxshmate/genmodelfit の main に向けて Pull Request を作ります。
   - 説明に、ComfyUI の起動オプション(`--lowvram` など、変えたものだけ)を書いてもらえると助かります。
   - 個人名やメールアドレスは不要です。
   - 初めての人の PR は、メンテナがワークフローの実行を許すまで CI が動かないことがあります。
3. CI(`Contrib` のワークフロー)が `genmodelfit contrib validate` で形・値の範囲・個人の情報・重複・ファイル名を検査し、結果をジョブのまとめ(Summary)に出します。
   - `contrib aggregate --report` の結果もまとめに出します。
   - 検査は、PR の向き先(main)の genmodelfit で動きます。
   - マージがメンテナの確認になります(人が提出ごとに承認する別の仕組みはありません)。

**genmodelfit は実測を送りません**。共有されるのは、あなたが自分で送ったものだけです: 「Create」で作った実測の Issue(ボットが Pull Request にします)か、自分の GitHub で出した Pull Request の中身。マージされると、提出は MPL-2.0 で公開されます(下の「[ライセンス](#ライセンス)」)。

#### 受け付けたあと

- マージされると data/contrib に入りますが、利用者に**届くのは genmodelfit の次の版から**です(#90)。版を出すときに、メンテナが `genmodelfit contrib aggregate --output src/genmodelfit/data/measurements_community.json` で、同じモデル・設定・GPU の提出をまとめます(中央値と MAD で外れ値を除く。[docs/release.ja.md](docs/release.ja.md))。まだまとめに入っていない提出は `genmodelfit contrib aggregate --check` で見られます。
- 使うのは**測った GPU と同じ GPU の PC だけ**です。ほかの GPU の PC の判定は変わりません(0.9.1。llmfit と同じ考え方)。「同じ GPU」とは次のとおりです:
  - PCI の型番 ID が同じ(両方にあるとき)。無ければ、そろえた名前が同じ(汎用の名前は一致としない)。
  - かつ、VRAM の総量の比が 0.95 以上(同じカードでも Linux と Windows で総量の数え方が少し違うため)。RTX 4060 Ti の 8 GB と 16 GB は別の GPU として扱います。
- その PC では **1 件から**、その構成の見積もりの根拠が `measured`(実測。作者の実測と同じ扱いで、表の VRAM に `m` が付きます)になり、出どころの数(`sample_count`)も出ます。
- 値は必要量を下げる向きにも上げる向きにも使います。全部載せの値は重み + 固定分より下げず、GPU の総量より上げません。
- 同じ GPU の同じ設定の実測は、**別々の 2 つ以上の出どころ**(作者の実測の行、または別の PR で送られた提出)がそろうと、段階を決める根拠にもなります(1 人が何件送っても、同じ PR の提出は 1 つの出どころ)。出どころは 1 つの PR(実測の Issue は 1 つの PR になります)なので、同じ人の別々の PR は別々に数えます。メンテナはマージのときにそれを確かめます(ボットは Issue を書いた人を PR に書きます)。実測は段階を上げるだけで、下げません。詳しくは [docs/scoring.ja.md](docs/scoring.ja.md) を見てください。
- 提出の PR では、`Contrib` のワークフローが `contrib aggregate --report` の結果をまとめに出します。校正の表(calibration.json・[docs/accuracy.ja.md](docs/accuracy.ja.md))は、メンテナが版を出すときに実測を取り込み、`python -m genmodelfit.calibrate` で作り直します。

#### 作者の実測(measurements.json)

同梱の `src/genmodelfit/data/measurements.json` はメンテナが測った実測です。項目は docs/schema.ja.md の「[Measurement](docs/schema.ja.md#measurement--実測)」の節、提出の形は「[contrib](docs/schema.ja.md#contrib)」の節にあります。利用者の実測は data/contrib に出してください(measurements.json は直接変えない)。

## そのほかの変更

コードの変更も歓迎します。

- 依存を足すときは MIT / Apache-2.0 / BSD のものだけにしてください(このツールを組み込むアプリがソースを公開せずに配布するため)。
- ComfyUI のコード(GPL-3.0)は import せず、HTTP だけで話します。
- 新しい `.py` ファイルの先頭には、ほかのファイルと同じ MPL-2.0 の表示(3 行)を入れてください。
- 送る前に `python tools/ci_local.py` で、CI と同じ検査(Python 3.10〜3.13 の試験・ruff・配布物)を手元で通してください。要るのは [uv](https://docs.astral.sh/uv/) だけです。Windows からは `--linux` で WSL の中でも流せます。
- フォークからの PR では GitHub の CI も動きます。

## ライセンス

genmodelfit のコードと同梱のデータ(db.json・実測・係数)は [MPL-2.0](https://github.com/laxshmate/genmodelfit/blob/main/LICENSE) です。

- 送っていただく修正・実測・データも、**MPL-2.0 で出していただいたものとして受け入れます**(プロジェクトのほかの部分と同じ。inbound = outbound)。
- Pull Request か実測の Issue を出すことで、これに同意したことになります。
- CLA(著作権の譲渡や追加の許諾の書類)はありません。
- 自分で書いたもの、または MPL-2.0 で出してよいものだけを送ってください。
