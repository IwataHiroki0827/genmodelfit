# モデルを落とす / Download (`genmodelfit download`)

README の「手順 5」の詳しい決まりです。JSON の形は [schema.md](schema.md)「download」、
実装は `src/genmodelfit/downloader.py`(約束は tests/test_download.py が検査する)。

## English

```
genmodelfit download "Z-Image-Turbo" --list                 # what would be downloaded (no network, writes nothing)
genmodelfit download "Z-Image-Turbo" --comfyui-dir C:\ComfyUI
genmodelfit download image_z_image_turbo_int8 --output-dir D:\models --budget 15 --yes
```

- **Which files**: all files of one workflow. A model name (as in the tables) picks the workflow and precision like
  `recommend`: the highest precision that runs on this PC without offloading. A workflow id is used as is.
  `--precision int8` picks a precision; `--budget 15` (GB) caps the download (files already there are not counted).
- **Where**: `--comfyui-dir` puts each file in the ComfyUI folder the template names
  (`models/diffusion_models`, `text_encoders`, `vae`, `loras`, `checkpoints`, …; subfolders in the file name are kept).
  `--output-dir` (default: genmodelfit's user data folder, `models`) uses the same layout, so it can be added to
  ComfyUI's `extra_model_paths.yaml`.
- **Consent first**: before anything is downloaded you see every file with its license and the license URL, licenses
  that are not on the allowlist (commercial use without use restrictions; [gate.md](gate.md)) are marked, then the
  total size and the free disk, and you are asked `[y/N]`. `--yes` skips the question (required with `--json`).
  Read the terms of each file before you use the model.
- **Not downloaded**: gated files (genmodelfit never handles logins or tokens; request access on the model page and
  download them yourself), uncensored models unless you pass `--content all` (or `uncensored`), files whose license is
  unknown unless you pass `--include-unknown-license`, files without a sha256 or size in the list, hosts other than
  Hugging Face / GitHub.
- **Safety**: every file is checked with the sha256 in the list while it is written; a mismatch deletes it and fails.
  A download goes to `<file>.part` first; after Ctrl-C or a network error run the same command again and it resumes
  with an HTTP Range request (if the server ignores Range it starts over, at most 3 times). Files already there are
  hashed again right before they are skipped; a different file with the same name is never overwritten. A
  `<file>.lock` next to it stops two downloads of the same file at once (a lock left by a process that is gone is
  cleaned up). It does not start unless the free disk is the download + 1 GB. Nothing is written outside the
  destination: names with `..`, absolute paths, drive letters, `\`, reserved Windows names (also as folder names),
  control / bidi characters, and symbolic links / junctions that lead outside are refused.
- **Network**: only the URLs in the list (checked anonymously with HEAD), https only. Every redirect must stay on
  the hosts below; a URL that could be read as two different hosts (`\`, spaces, control characters, `%` or `@`
  in the host, non-ASCII hosts) is refused before any connection, and the host is read the way requests / urllib3
  read it. Allowed: `huggingface.co` / `*.hf.co` (CDN, Xet) / `github.com` / `objects.githubusercontent.com` /
  `release-assets.githubusercontent.com`. No `~/.netrc` credentials, no tokens, nothing about this PC (User-Agent and
  Range only). Waits for `Retry-After` on 429 / 503.

Python:

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

The library prints nothing. `gmf.download` raises `gmf.DownloadRefused` (blocked, no consent), `gmf.DownloadError`
(network, sha256 mismatch, disk) and lets `KeyboardInterrupt` through, keeping the `.part` file for the next call.

## 日本語

- **落とすもの**: 1 つのワークフローの全部のファイル。モデル名(表の Model)なら、`recommend` と同じく、この PC で
  逃がさずに載る中で一番精度の高いワークフローを選びます。ID を指定すればそのまま使います。`--precision int8` で精度を、
  `--budget 15`(GB)で落とす量の上限を決められます(既にあるファイルは数えません)。
- **置き場所**: `--comfyui-dir` なら、テンプレートの置き場所(`models/diffusion_models`・`text_encoders`・`vae`・`loras`・
  `checkpoints` など。ファイル名の中のサブフォルダは保つ)に置きます。`--output-dir`(既定は genmodelfit のデータのフォルダの
  `models`)も同じ並びなので、ComfyUI の `extra_model_paths.yaml` に足せます。
- **先に同意**: 落とす前に、ファイルごとのライセンスと原文の URL を出し(利用制限の無い商用利用ができる許可リスト
  ([gate.md](gate.md))に無いものは印付き)、合計の大きさと空きを見せて `[y/N]` で尋ねます。`--yes` で省けます
  (`--json` で落とすときは必須)。使う前に各ファイルの条件を読んでください。
- **落とさないもの**: gated(ログインや利用申請が要る。genmodelfit はログイン情報やトークンを扱いません。モデルのページで申請して
  自分で落とす)、無検閲のモデル(`--content all` か `uncensored` を明示したときだけ)、ライセンス不明
  (`--include-unknown-license` のときだけ)、一覧に sha256・大きさの無いもの、Hugging Face・GitHub 以外の取得先。
- **安全**: 書きながら sha256 を計り、合わなければ消して失敗にします。途中は `<名前>.part` に書き、Ctrl-C や通信の失敗の
  あとで同じコマンドを実行すると HTTP Range で続きから落とします(相手が Range に応じなければ最初から。3 回まで)。
  既にあるファイルは飛ばす直前に sha256 を計り直し、同じ名前の別のファイルは上書きしません。同じファイルを
  2 つの download が同時に書かないよう `<名前>.lock` を置きます(止まったプロセスの残りは片付けます)。空きが
  「落とす量 + 1 GB」に足りなければ始めません。置き場所の外には書きません(`..`・絶対パス・ドライブ名・`\`・
  Windows の予約名(フォルダ名も)・制御文字・双方向文字・外へ出るシンボリックリンクやジャンクションを拒みます)。
- **通信**: 一覧の URL(匿名の HEAD で確かめたもの)だけを https で使います。転送先は上の英語の節のホストだけです。
  2 通りのホストに読める URL(`\`・空白・制御文字・ホストの中の `%` や `@`・ASCII でないホスト)は、つなぐ前に拒みます
  (ホストは requests / urllib3 と同じ読み方で取ります)。`~/.netrc` もトークンも使わず、PC の情報も送りません
  (User-Agent と Range だけ)。429・503 では `Retry-After` を待ちます。
- **Python**: 上の例と同じです。ライブラリは何も表示しません。`gmf.download` は `gmf.DownloadRefused`(ブロック・同意なし)と
  `gmf.DownloadError`(通信・sha256 の不一致・空き)を投げ、`KeyboardInterrupt` はそのまま通して、`.part` を次の呼び出しのために残します。
