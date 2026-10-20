# CLAUDE.md — genmodelfit

画像・動画・音楽・効果音・読み上げ・3D の生成モデルが、その PC で動くかを判定する(llmfit の生成モデル版)。
MIT で公開。使う側の代表は AI生成キット(laxshmate/local_model_ai_gen_kit。ソース非公開で販売するアプリ)。

## 決まり(2026-10-20)

- **依存は MIT / Apache-2.0 / BSD のものだけ**。GPL・AGPL・独自の制限付きライセンスの部品は使わない
  (使う側のアプリがソース非公開で配布するため)。足すときは pyproject.toml にライセンスを書く
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
- REST の serve は v0.2(作るならポートは ~/.claude/PORTS.md で選んで登録する)
- 実装は Claude、レビューはユーザ。Issue ごとに 1 本の PR。依存する PR は前の PR のブランチから作り、PR の説明に書く

## コマンド

```powershell
uv venv --python 3.12; uv pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\ruff check . ; .venv\Scripts\ruff format --check .
.venv\Scripts\genmodelfit system | recommend --json | fit | info <id> | update | bench <template> | gate
.venv\Scripts\python -m genmodelfit.calibrate      # 実測から係数を合わせ直す(calibration.json と精度の表)
.venv\Scripts\genmodelfit update --heads-only --no-cache --output src\genmodelfit\data\db.json   # 同梱の一覧の HEAD だけやり直す
```

ネットワークに触るテストは `GENMODELFIT_NETWORK_TESTS=1` のときだけ動く(既定では飛ばす)。
ライブラリの関数は stdout / stderr に何も書かない(組み込み側の出力に混ざるため)。表示は cli.py に寄せ、
途中経過は `logging.getLogger("genmodelfit")` か、呼び出し側が渡す progress / log に出す(tests/test_embedding.py が検査)。

## 形

- データの形は `src/genmodelfit/models.py` が正本。変えたら `docs/schema.md` と `SCHEMA_VERSION` も直す
- 一覧のモデル ID: 公式テンプレート由来はテンプレート名そのまま(例 `image_z_image_turbo_int8`)、
  Hugging Face 由来は `hf:<owner>/<repo>`
- 同梱のデータ: `src/genmodelfit/data/db.json`(一覧)、`measurements.json`(実測)、`calibration.json`(係数)
- **実測や、実測のあるモデルのファイル情報を変えたら `python -m genmodelfit.calibrate` で合わせ直し、
  出た表で docs/accuracy.md を直す**(tests/test_estimate.py が calibration.json と合わせ直した結果の一致を検査する)
- キャッシュ: platformdirs の user_cache_dir(`GENMODELFIT_CACHE_DIR` で変えられる)

## 共用マシン

- GPU は他の作業と共用。bench の前に nvidia-smi で空きを確かめ、他のプロセスが使っていたら測らない
- 実測に使うモデルは 1 つずつ試して消し、ディスクの空きを 20GB 以上残す
