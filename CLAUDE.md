# CLAUDE.md — genmodelfit

画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くかを判定する(llmfit の生成モデル版)。
**MPL-2.0** で公開(コードも同梱のデータも。2026-10-20 にユーザが決めた。Issue #18)。使う側の代表は AI生成キット
(laxshmate/local_model_ai_gen_kit。ソース非公開で販売するアプリ。改変せずに import して同梱する)。
新しい `.py` には MPL の表示(Exhibit A の 3 行)を先頭に入れる。外からの修正は MPL-2.0 で受け入れる(CLA なし)。
データの出典(ComfyUI workflow_templates の MIT の表示・HF の扱い・llmfit)は THIRD_PARTY_NOTICES に書く

## 決まり(2026-10-20)

- **依存は MIT / Apache-2.0 / BSD のものだけ**。GPL・AGPL・独自の制限付きライセンスの部品は使わない
  (使う側のアプリがソース非公開で配布するため)。足すときは pyproject.toml にライセンスを書く
  - 例外: `certifi`(MPL-2.0、requests の依存)。改変せずに使う。MPL-2.0 はファイル単位のコピーレフトで、
    改変しなければ、義務はライセンス表示を残すことと、実行形式で配るときにソースの入手先を知らせること(§3.2)だけ。
    アプリのソース非公開には影響しない。
    これ以外の MPL・LGPL などを足すときは先に相談する
  - tests/test_licenses.py が、入っている実行時の依存(dev などの extra は除く)のライセンスを許可リストと照らす
    (MIT-0・ISC・PSF-2.0 も MIT と同じ扱いで許す。cffi が MIT-0)
- **ComfyUI(GPL-3.0)のコードは import しない**。動いている ComfyUI には HTTP だけで話す(bench)
- データ源は ComfyUI 公式テンプレート(MIT)の index.json とテンプレートの JSON、Hugging Face Hub API。
  willitrunai.com などのサイトのデータは取り込まない(式の考え方だけ参考にする)
- 持つのはモデルの**情報**だけ。モデル本体は持たない・配らない
- 無検閲のモデルも一覧から外さず、`content: uncensored` の印と根拠(content_evidence)を付ける。
  **未成年の性的表現を目的とするものは除外**する
  年齢制限があるので、`recommend` / `fit` / `gmf.recommend()` の既定は `sfw`。無検閲は `--content all|uncensored`
  (Python は `content="all"|"uncensored"`)を指定したときだけ出す(2026-10-20)。`info <id>` は ID を指定するので出す
- 見積もりには必ず `basis`(measured / estimated)を付ける。実測があれば実測を使う
- 精度の目標に届かないとき、係数を無理に合わせてごまかさない。docs/accuracy.md に正直に書く
- 版の正本は `src/genmodelfit/_version.py`(pyproject は hatch が読む)。変えたら CHANGELOG.md も書く
- 一覧の新しさ(HEAD の 30 日の期限)は `status.py`(`gmf.data_status()`、`system` の Data の欄)。gate の
  `evaluate_offline` で判定する(期限の式を別に持たない)
- HTTP は必ず `fetch.make_session()`(User-Agent を付け、~/.netrc の資格情報を送らない。プロキシの環境変数は使う)。
  読むだけのコマンドと Python の関数はネットに触らない(README「Privacy」、tests/test_privacy.py・test_embedding.py)
- `download`(#31、`downloader.py`)は一覧の URL だけを https で落とし、転送先は `HOST_POLICY` のホストだけ。
  同意(confirm / yes)・sha256・`.part` の続き・置き場所の外に書かない・gated / 無検閲 / ライセンス不明を落とさない、
  の約束は tests/test_download.py が検査する(約束を変えたら README「Download」と docs/schema.md「download」も直す)
- REST の serve は v0.2(作るならポートは ~/.claude/PORTS.md で選んで登録する)
- 実装は Claude、レビューはユーザ。Issue ごとに 1 本の PR。依存する PR は前の PR のブランチから作り、PR の説明に書く

## コマンド

```powershell
uv venv --python 3.12; uv pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff check . ; .venv\Scripts\ruff format --check .
.venv\Scripts\genmodelfit system | recommend --json | fit | search <語> | types | info <id> | update | bench <template> | gate
.venv\Scripts\python -m genmodelfit.calibrate      # 実測から係数を合わせ直す(calibration.json と精度の表)
.venv\Scripts\genmodelfit update --heads-only --no-cache --output src\genmodelfit\data\db.json   # 同梱の一覧の HEAD だけやり直す
.venv\Scripts\genmodelfit diff <old.json> <new.json>              # 2 つの一覧の違い(毎週の PR の本文と同じ中身)
.venv\Scripts\python -m genmodelfit.weekly --dry-run --out <dir>   # 毎週の作り直し・確かめ・PR の本文を手元で(#19)
```

毎週の一覧の更新(weekly-data.yml、#19)は PR を作るだけで、自動でマージ・公開しない。配るのはマージした
main の db.json だけ(data.yml)。試験は同梱の一覧の件数・日付を決め打ちしない(毎週変わる。一覧から数える)。

ネットワークに触るテストは `GENMODELFIT_NETWORK_TESTS=1` のときだけ動く(既定では飛ばす)。
ライブラリの関数は stdout / stderr に何も書かない(組み込み側の出力に混ざるため)。表示は cli.py に寄せ、
途中経過は `logging.getLogger("genmodelfit")` か、呼び出し側が渡す progress / log に出す(tests/test_embedding.py が検査)。

## 形

- HF のモデルは全期間のダウンロード数 10,000 以上(trendingScore・likes30d で拾ったものは 1,000 以上)、merge のタグ・
  試験用の名前は入れない(llmfit の scrape_hf_models.py にならう。2026-10-20)。利用者のテンプレート(index の username が
  ComfyUI でない)は `community`。公式の仲間があれば表の代表にしない
- 段階は動かし方で使うメモリの使用率(60 / 85 / 98%。載せ替えは good まで、逃がす・CPU は marginal まで。
  llmfit にならう。2026-10-20 に利用者が決めた。`estimate.FIT_BANDS`)
- 表・JSON では同じモデル(本体の系統 + 大きさ。精度・使い方だけ違う項目)を 1 行にまとめる(`groups.py`。
  鍵の決め方を変えたら docs/schema.md「Models」と tests/test_groups.py の名前の表を直す)。用途の表は `tasks.py`
- データの形は `src/genmodelfit/models.py` が正本。変えたら `docs/schema.md` と `SCHEMA_VERSION` も直す
- 一覧のモデル ID: 公式テンプレート由来はテンプレート名そのまま(例 `image_z_image_turbo_int8`)、
  Hugging Face 由来は `hf:<owner>/<repo>`
- 同梱のデータ: `src/genmodelfit/data/db.json`(一覧)、`measurements.json`(実測)、`calibration.json`(係数)
- **実測や、実測のあるモデルのファイル情報を変えたら `python -m genmodelfit.calibrate` で合わせ直し、
  出た表で docs/accuracy.md を直す**(tests/test_estimate.py が calibration.json と合わせ直した結果の一致を検査する)
- キャッシュ: platformdirs の user_cache_dir(`GENMODELFIT_CACHE_DIR` で変えられる)

## ハードウェアの検出(2026-10-20、#21。docs/hardware.md が正本)

- 判定に使う GPU は nvidia-smi で確かめた NVIDIA の GPU だけ(backend `cuda` の最大の 1 枚)。PyTorch の CUDA 版のビルドが
  どれも対応しない世代・どれにもドライバが足りない GPU は backend `none`(CPU として判定し、理由を notes に書く)。
  世代・ドライバが分からないときは断らない。表(`hardware.TORCH_CUDA_BUILDS`・`_NAME_CC`)を変えたら docs/hardware.md の出典も直す
- NVIDIA 以外の GPU(WMI・レジストリ・lspci)は知らせるだけで**判定に使わない**。NVIDIA の GPU があるときは既定で調べない
  (PowerShell の起動に約 1 秒)。`system --all-gpus` で調べる
- 精度の直しは ComfyUI のコード(should_use_fp16・supported_inference_dtypes・VAE の working_dtypes)で確かなところだけ。
  系統の表は `families.py`(行番号は ComfyUI のコミット付き)。表・判定の規則を変えたら同梱の一覧の `comfy_family` も作り直す
  (`families.assign`。tests/test_hardware.py が一致を検査)。VRAM の量で変わる動き(fp32・fp16 に広げる)は
  見積もりに入れず注意だけにする(入れると VRAM を増やして段階が下がる)。ComfyUI の版を上げたら名前の一覧と分岐を読み直す
- 段階は VRAM の総量で決め、今の空きが足りなければ `vram_free_short` と理由で知らせるだけ
- PC 名・ユーザ名・GPU の UUID・PCI の差し場所や個体の ID を読まない・出さない
- 試験は本物の nvidia-smi・PowerShell・lspci・レジストリを呼ばない(tests/conftest.py の `_no_real_hardware`)。
  出力はフィクスチャ(tests/fixtures/hardware)で差し替え、検出は失敗しても例外を出さない

## 共用マシン

- GPU は他の作業と共用。bench の前に nvidia-smi で空きを確かめ、他のプロセスが使っていたら測らない
- 実測に使うモデルは 1 つずつ試して消し、ディスクの空きを 20GB 以上残す
