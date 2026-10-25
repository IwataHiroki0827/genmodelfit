# モデルを落とす(`genmodelfit download`)

[English](download.md) · **日本語**

[README](../README.ja.md)の「2. 手順」の 5(ダウンロードする)の詳しい決まりです。
CLI を使う人と、Python の API を使うアプリの開発者向けです。
JSON の形は [schema.ja.md](schema.ja.md)「download」にあります。
実装は `src/genmodelfit/downloader.py` で、ここに書いた約束は `tests/test_download.py` が検査します。

## コマンドライン

```
genmodelfit download "Z-Image-Turbo" --list                 # what would be downloaded (no network, writes nothing)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
genmodelfit download MODEL --content all --include-unknown-license   # uncensored / unknown license (your choice)
```

- **落とすもの**: 1 つのワークフローの全部のファイル。
  - モデル名(表の Model)なら、`recommend` と同じく、この PC で逃がさずに載る中で一番精度の高いワークフローを選ぶ。
  - ID を指定すればそのまま使う。
  - `--precision int8` で精度を決められる。
  - `--budget 15`(GB)で落とす量の上限を決められる(既にあるファイルは数えない)。
- **置き場所**:
  - `--comfyui-dir` なら、テンプレートの置き場所(`models/diffusion_models`・`text_encoders`・`vae`・`loras`・`checkpoints` など)に置く。ファイル名の中のサブフォルダは保つ。
  - `--output-dir`(既定は genmodelfit のデータのフォルダの `models`)も同じ並びなので、ComfyUI の `extra_model_paths.yaml` に足せる。
  - 環境変数 `GENMODELFIT_COMFYUI_DIR` が `--comfyui-dir` の既定になる(CLI だけ。Python は `comfyui_dir` を渡す)。順は `--comfyui-dir` → `--output-dir` → `GENMODELFIT_COMFYUI_DIR` → 既定のフォルダ。値はフラグと同じく確かめる(ComfyUI のフォルダに見えること)。どれも無いときは、計画の中で決め方を示す(例 PowerShell `$env:GENMODELFIT_COMFYUI_DIR = "C:\ComfyUI"`、bash / zsh `export GENMODELFIT_COMFYUI_DIR="$HOME/ComfyUI"`)。
  - ComfyUI のどのフォルダか分からないファイル(Hugging Face のモデルに多い)は `other/` に置く。ComfyUI は `models/other` を読まないので、計画の中でそう知らせる。使う前に(モデルのページの説明に従って `checkpoints`・`diffusion_models`・`loras` などへ)移すこと。
  - そのようなファイルの置き場所は、同梱の一覧の 1 つの根拠だけで決める: 本体のファイルのヘッダのパラメータ数が、公式の ComfyUI テンプレートのファイルと 0.001% の幅で一致し、一致したファイルの置き場所が 1 つだけのとき(例 SDXL の全部入りのファイルは `sd_xl_base_1.0.safetensors` と同じ数なので `checkpoints`)。計画に根拠を出す(JSON は `directory_basis`)。名前やタグからの推測はしない。
- **先に同意**: 落とす前に、次を見せる。
  - ファイルごとのライセンスと原文の URL。原文の URL が無いファイルは「ライセンスの記載がありません」と書き、確かめる場所(モデルの Hugging Face のページか、一覧にあるページ)を出す。そのときの同意の質問は、そのページで自分で確かめることを求める文になる。
  - 許可リストに無いライセンスの印(利用制限の無い商用利用ができるもの。[gate.ja.md](gate.ja.md)。`--license-allowlist` で変えられる)。
  - 合計の大きさと空き。
  - そのあと `[y/N]` で尋ねる。`--yes` で省ける(`--json` で落とすときは必須)。
  - 使う前に各ファイルの条件を読むこと。
- **落とさないもの**: 1 つでも当てはまるファイルがあれば、そのワークフローは何も落とさない。一部だけ落とすことはしない。
  - 理由は 1 回ずつ出す。足りないのが `--content all`・`--include-unknown-license` だけなら、計画の最後に「落とすなら:」として、それを全部足したコマンドを 1 行で出し(ほかの指定はそのまま)、指定ごとに何に同意するかを 1 行ずつ添える。勝手に足して落とすことはしない。ほかの理由ではコマンドを出さず、代わりにすること(利用申請・ディスクを空けるなど)を書く。
  - gated(ログインや利用申請が要る)。genmodelfit はログイン情報やトークンを扱わない。モデルのページで申請して自分で落とす。
  - 無検閲のモデル(`--content all` か `uncensored` を明示したときだけ)。
  - ライセンス不明(`--include-unknown-license` のときだけ)。
  - 一覧に sha256・大きさの無いもの。
  - Hugging Face・GitHub 以外の取得先。
- **安全**:
  - 書きながら sha256 を計り、合わなければ消して失敗にする。
  - 途中は `<名前>.part` に書く。Ctrl-C や通信の失敗のあとで同じコマンドを実行すると、HTTP Range で続きから落とす。相手が Range に応じなければ最初から(3 回まで)。
  - 既にあるファイルは、飛ばす直前に sha256 を計り直す。同じ名前の別のファイルは上書きしない。
  - 同じファイルを 2 つの download が同時に書かないよう `<名前>.lock` を置く。止まったプロセスの残りは片付ける。
  - 空きが「落とす量 + 1 GB」に足りなければ始めない。
  - 置き場所の外には書かない。次を拒む: `..`・絶対パス・ドライブ名・`\`・Windows の予約名(フォルダ名も)・制御文字・双方向文字・外へ出るシンボリックリンクやジャンクション。
- **通信**:
  - 一覧の URL だけを https で使う(update が匿名の HEAD で確かめたもの。download 自体は HEAD の結果を条件にしない)。
  - 最初の URL は `huggingface.co`・`github.com` だけ。
  - 転送先は次のホストだけ(https・443 だけ): `huggingface.co`・`*.huggingface.co`・`hf.co`・`*.hf.co`(CDN、Xet)・`github.com`・`objects.githubusercontent.com`・`release-assets.githubusercontent.com`。
  - 2 通りのホストに読める URL は、つなぐ前に拒む: `\`・空白・制御文字・ホストの中の `%` や `@`・ASCII でないホスト。ホストは requests / urllib3 と同じ読み方で取る。
  - `~/.netrc` もトークンも使わず、PC の情報も送らない(User-Agent と Range だけ)。
  - HTTP 429・500・502・503・504 では `Retry-After`(無ければ間を空けて)待ってやり直す。進まない失敗が 4 回続いたら止める。

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

- ライブラリは何も表示しない。
- `gmf.download` は `gmf.DownloadRefused`(ブロック・同意なし)と `gmf.DownloadError`(通信・sha256 の不一致・空き)を投げる。
- `KeyboardInterrupt` はそのまま通し、`.part` を次の呼び出しのために残す。
