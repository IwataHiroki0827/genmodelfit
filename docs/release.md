# リリースとデータの配り方

3 つの流れがあります。

| 流れ | 何を配る | きっかけ | 配り先 |
|---|---|---|---|
| 版のリリース | パッケージ(sdist と wheel) | `v*` のタグを push(タグは管理者だけが作れる) | PyPI と GitHub Release |
| データの配布 | モデル一覧 `db.json` と署名 `db.json.sig` | main の db.json が変わった push・手動実行 | GitHub Release `data-latest` |
| 毎週の一覧の更新 | 作り直した `db.json` の PR(配らない) | オーナーの PC のタスク スケジューラ(毎週月曜 9:00 JST)。weekly-data.yml は手動の予備 | PR(安全な週だけ自動でマージ。下の「毎週のスキル」) |

データの配布は `v*` のタグではないので、PyPI には出ません。

main はブランチの保護で PR を必須にし、force push と削除を禁じている。

## GitHub の environment(2026-10-20 に設定)

| environment | 使えるところ | 承認 | シークレット |
|---|---|---|---|
| `pypi` | `v*` のタグだけ | なし(2026-10-21 にオーナーが外した。代わりにルールセットで `v*` のタグを管理者だけにする) | なし(Trusted Publishing) |
| `data-signing` | `main` ブランチだけ | なし | `GENMODELFIT_SIGNING_KEY`(署名の秘密鍵) |

リポジトリ全体のシークレットには何も置かない。秘密鍵は `data-signing` だけにあるので、main 以外のブランチや
PR のワークフローからは読めない。data.yml でも、署名する 1 ステップにだけ環境変数で渡す。

設定の確認: `gh api repos/laxshmate/genmodelfit/environments` と
`gh secret list -R laxshmate/genmodelfit`(空であること)・`gh secret list --env data-signing -R laxshmate/genmodelfit`。

## 最初の公開(v0.1.0)の手順

**公開はユーザの承認を得てから。** 2026-10-20 にユーザが v0.1.0 のマージ(Claude が行う)と公開を承認した。
環境 `pypi` の承認(Review deployments)はユーザが押す(v0.1.0 のとき。2026-10-21 に承認者を外した)。
2026-10-20 に確認: environment `pypi`(`v*` のタグだけ・承認者 `KoyoYeager`。2026-10-21 に外した)と `data-signing`(`main` だけ・
シークレット `GENMODELFIT_SIGNING_KEY` あり)はそろっている。リポジトリ全体のシークレットは空。PyPI の
`genmodelfit` はまだ無い(404)。

1. **PR をマージする**(この順。どれも前の PR のブランチから作ってある):
   PR #8 → #13 → #14 → #10 → #11 → #12 → #9 → #16 → #20(Issue #17)→ #18 の PR → #21 の PR。
   - 積み重ねた PR は、土台が前の PR のブランチになっている。そのままマージすると main ではなく前のブランチに入るので、
     **マージする前に土台を main に付け替える**(`gh pr edit <番号> --base main`)。付け替えると差分はその PR の分だけになる。
   - **squash や rebase でマージしない**(普通のマージコミット)。squash すると後ろの PR の履歴と合わなくなる。
   - 付け替えたあと CI(Python 3.10〜3.13 × Ubuntu / Windows、wheel)が全部緑であることを確かめてからマージする
     (v0.1.0 のとき。今はブランチの PR で ci.yml が動かないので、`tools/ci_local.py --linux` を手元で通し、結果を PR に書く)。
2. **一覧を配る**: #18 のマージで main の `src/genmodelfit/data/db.json` が変わっていれば、push で
   `.github/workflows/data.yml` が動く。動いていなければ Actions の「Data」→「Run workflow」(ブランチ main)で動かす。
   緑になったら、Release `data-latest` に `db.json` と `db.json.sig` があることを確かめる。

   ```powershell
   gh release view data-latest -R laxshmate/genmodelfit
   ```
3. **タグを push する**(版の正本は `src/genmodelfit/_version.py`。`0.1.0` になっていること):

   ```powershell
   git switch main; git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```
4. `.github/workflows/release.yml` が次の順に進む。
   - test: ruff と pytest(3.12)
   - build: `uv build` → タグと `_version.py`・wheel / sdist の名前の版が一致するか確認 → `twine check --strict`
   - publish: environment `pypi`(承認者なし。`v*` のタグはルールセットで管理者だけが作れる)で PyPI に公開する
     (Trusted Publishing。API トークンは使わない)
   - github-release: 同じ dist を付けた GitHub Release `v0.1.0` を作る
5. **確かめる**(まっさらな venv で。キャッシュも別の場所にする):

   ```powershell
   uv venv -p 3.12 $env:TEMP\gmf-check
   uv pip install --python $env:TEMP\gmf-check\Scripts\python.exe genmodelfit==0.1.0
   $env:GENMODELFIT_CACHE_DIR = "$env:TEMP\gmf-check-cache"
   & $env:TEMP\gmf-check\Scripts\genmodelfit --version          # genmodelfit 0.1.0
   & $env:TEMP\gmf-check\Scripts\genmodelfit system             # Data の欄: List: bundled …
   & $env:TEMP\gmf-check\Scripts\genmodelfit update --from-release   # 署名を確かめて取り込む
   & $env:TEMP\gmf-check\Scripts\genmodelfit system             # List: release、HEAD checks の期限
   & $env:TEMP\gmf-check\Scripts\genmodelfit recommend --vram 8 --ram 32 -n 2
   ```
6. AI生成キット(laxshmate/local_model_ai_gen_kit)の Issue #8 に公開を知らせる。

## 次からの版を出す手順

1. `src/genmodelfit/_version.py` の `__version__` を上げ(pyproject.toml は hatch がここから読む)、
   `CHANGELOG.md` に書いて PR で main に入れる。PR では ci.yml が動かないので、PR の説明に
   `tools/ci_local.py --linux` の結果を書く。
2. 承認後、マージの後に main で動いた ci.yml が緑なのを確かめてから、main の先頭にタグ `v<版>` を付けて push する
   (上の 3〜5 と同じ)。

手動実行(workflow_dispatch)はビルドまでで、公開しない。タグを選んで手動実行しても、publish と github-release は
`github.event_name == 'push'` の条件で飛ばす。タグを打つ前の試し用。

PyPI 側の準備は済み。アカウント `laxshmate`(2 段階認証あり)に、`genmodelfit` / `laxshmate/genmodelfit` / `release.yml` /
environment `pypi` の Trusted Publisher を事前登録してある。事前登録では名前は確保されず、最初の公開で確定する。
**PEP 541(放棄された名前の扱い)があるので、中身の無い仮の版で名前だけ押さえない。動く v0.1 を公開する。**

公開した版は消せない(取り下げ=yank だけ)。版の番号は使い回せないので、直すなら次の版を出す。

## データの配布

`.github/workflows/data.yml` は、**main にコミットされた `src/genmodelfit/data/db.json` をそのまま**署名して配る
(上流から作り直さない。作り直して PR にする流れは下の「毎週の一覧の更新」)。きっかけは、main の db.json が変わった
push と手動実行。

1. sign: environment `data-signing`(main だけ)で、コミットされた一覧を読めることを確かめ、署名し、同梱の公開鍵で
   確かめる(`contents: read`。秘密鍵はこのジョブの「署名する」ステップだけ)。main 以外のブランチを選んだ手動実行は
   `if: github.ref == 'refs/heads/main'` で動かない
2. upload: Release `data-latest`(prerelease)の `db.json` と `db.json.sig` を置き換える(ここだけ `contents: write`)
3. verify: 置いた直後に、まっさらな venv(main のコードと、`genmodelfit==0.5.0`(`review` を知らない、いちばん古い対応の版))で
   `update --from-release` を流し、source が release・期限切れでない・100 件超・自動のおすすめが 1 件以上なのを確かめる

checkout は `persist-credentials: false`。

同梱の一覧の HEAD の確認は 30 日で切れる(切れると自動のおすすめが 0 件になる)。普段は毎週の PR(下)をマージすれば
延びる。毎週の PR が止まっているときは、切れる前に
`genmodelfit update --heads-only --no-cache --output src/genmodelfit/data/db.json` で取り直して PR にし、main に入れると
data.yml が配り直す。今の一覧の期限は `genmodelfit system --no-cache` の Data の欄(`valid until`)で分かる。

使う側:

```powershell
genmodelfit update --from-release                       # キャッシュに取り込む
genmodelfit update --from-release --output db.json      # 確かめた一覧を指定の場所に書く(キャッシュは触らない)
genmodelfit update --from-release --force               # 前に取り込んだ配布より古くても取り込む(下記)
```

```python
from genmodelfit.remote import update_from_release

db = update_from_release()  # 失敗したら RemoteError(DatabaseError の仲間)
```

取り込みの手順:

1. `db.json` と `db.json.sig` を取る(それぞれ 64 MB まで。読みながら数え、超えたらそこでやめる)
2. 同梱の信じる公開鍵(`src/genmodelfit/data/signing_keys/*.pub`)の**どれか 1 つ**で署名が合うか確かめる。
   合わなければ 5 秒待って**1 回だけ**取り直す。GitHub Release の資産は 1 つずつ置き換わるので、
   data.yml の upload の途中に取ると、新しい `db.json` と古い `db.json.sig` の組になることがあるため。
   取り直しても合わなければ失敗
3. `schema_version` と形を確かめる。`generated_at` は必須で、タイムゾーン付きの ISO 8601(`Z` も可)
4. 前に取り込んだ配布より古くないか確かめる(巻き戻しの防止)。比べる相手はキャッシュの `release_state.json`
   (前に取り込んだ配布の `generated_at`)。キャッシュの `db.json` は手元の `genmodelfit update` も書くので比べない。
   日時は UTC に直して比べる(`Z` と `+00:00` は同じ時刻)。同じ日時の取り直しは通す
5. `release_state.json` を進めてから、書き出し先(既定はキャッシュの `db.json`)を原子的に置き換える

どれかが失敗したら、書き出し先(無ければ同梱のデータ)はそのままで、メッセージを出して終了コード 1(`--json` なら
`{"error": ...}` の形)。ほかのプログラムがキャッシュを開いていて置き換えられないとき(Windows)も同じ。

`release_state.json` の日時が未来になってしまった(作り直した一覧の日時が戻った等)ときは、普段の取り込みは
「古い一覧です」で止まる。正しい配布だと確かめたうえで `--force` を付けると取り込み、記録もその日時に直す。
`release_state.json` が壊れているときは無いものとして扱い、次の取り込みで書き直す。

署名は Ed25519。`db.json.sig` は 64 バイトの署名を base64 にした 1 行。
署名の確認だけ手でやるなら `python -m genmodelfit.signing verify db.json`(同梱の信じる鍵のどれかで確かめる)。

## 毎週の一覧の更新(#19)

llmfit の weekly-model-update.yml にならい、上流から作り直した一覧を毎週 PR にする。**公開するのは人が見て
マージした一覧だけ**(作り直しと公開を分ける)。

```text
weekly-data.yml(手動の予備。#42 から定期の実行はオーナーの PC の起動スクリプト)→ PR automated/weekly-data → 作者が見てマージ
  → data.yml が main の db.json に署名して data-latest に置く → 使う側は genmodelfit update --from-release
```

`.github/workflows/weekly-data.yml`:

1. build(`contents: read` だけ):
   - キャッシュを戻す(`actions/cache/restore`。`file_headers.json`・`repo_precision.json`。鍵は毎回違う run_id で、
     restore-keys で一番新しいものを取る)
   - `python -m genmodelfit.weekly --out weekly-out --apply src/genmodelfit/data/db.json --rate-limit-budget 900`
     (時間の上限 90 分)。`genmodelfit update` と同じ作り直し(テンプレート・HF・HEAD・ファイルの見出しと精度)に続けて `update --inspect-only` と同じ読み直し(同梱の一覧と同じ 2 段)。
     引き継ぎの元は main の一覧。HF の 429 は Retry-After を守って待ち、待ち時間の合計が 900 秒を超えたら止まる
     (HF_TOKEN は使わない)
   - 確かめ(どれかに落ちたら終了コード 1 で、PR を作らない。理由は `weekly-out/result.json` の `problems` と
     実行の要約): 形(読み直し)、自動のおすすめが 0 件でない、
     **区分ごとに main の半分未満に減っていない**(全体・HF・公式テンプレート・無検閲・自動のおすすめ。元が 4 件以上の
     区分だけ)、**HEAD の期限(`expires_at`)が 7 日より先**(作り直した直後なのに近ければ HEAD の取り直しが効いていない。
     `genmodelfit system --json` の側からも二重に確かめる)、**新しく無検閲の印が付いた項目が 5 件まで**(超えたら
     選び方の規則が緩んだ疑い。手動実行で `allow_many_uncensored` を選んだときだけ、失敗にせず本文の先頭に警告)
   - キャッシュを保存する(`if: always()`。失敗・時間切れでも残す。GitHub は 7 日使わないキャッシュを消す)
   - 新しい一覧を入れた状態で `ruff check`・`ruff format --check`・`pytest -q`(CI の代わり。下の注意)
   - `genmodelfit contrib aggregate --report` の結果を PR の本文に足す(この手動の予備だけ。オーナーの PC の起動スクリプトは
     流さない)。data/contrib がまだ無ければ「提出 0 件」と書く。報告には、ほかの利用者の実測で変わる段階の表(1 件の提出も含む。
     2026-10-22 から両方の向きに使うので、上がる・下がるの両方が出る)が入る
   - `weekly-out/`(db.json・diff.json・pr_body.md・result.json・gate.json・system.json)を artifact に残す(30 日)
2. pr(`contents: write`・`pull-requests: write` はこのジョブだけ。署名の鍵の environment は使わない):
   `result.json` が ok の一覧だけを入れ、`peter-evans/create-pull-request`(コミットで固定)で PR を作る・更新する
   (ブランチ `automated/weekly-data`、ラベル `automated`・`data-update`、本文は `weekly-out/pr_body.md`)

決めたこと:

- **PR の本文の上流の文字はすべて無害にする**: モデルの名前・ID・ライセンスの名前・根拠・URL・リポジトリ名・
  失敗の理由は `weekly.md()` で、HTML の文字を実体参照に(`<!--`・`<img>`・`</details>` が効かない)、Markdown の
  記号をバックスラッシュで(リンク・画像・強調・表の区切り)、`@` の後に幅の無い空白(メンションしない)。
  長い本文は行の切れ目で切り、開いたままの `<details>` を閉じる。#34 の実測の集計(利用者の出した値を含む)は
  `weekly.code_block()` の囲みに入れる。実行の要約(`$GITHUB_STEP_SUMMARY`)にも同じ本文を出す

- **自動ではマージしない**(auto-merge も使わない)。ここでは署名も公開もしない
- **GITHUB_TOKEN で作った PR では CI(ci.yml)が自動で動かない**(GitHub の仕様。ループを防ぐため)。代わりに同じ
  ワークフローの build で、新しい一覧を入れた状態の ruff と pytest を通してから PR を作る(PR の本文にもそう書く)。
  同じリポジトリのブランチの PR では ci.yml は動かない(#68)。確かめるなら手元で `python tools/ci_local.py --linux` を回すか、
  マージの後に main で動く ci.yml を見る
- **一部を取得できなかったとき**(`sources.partial`): 既定は失敗(PR を作らない)。手動実行で `allow_partial` を
  選んだときだけ、欠けたのが HF のリポジトリの一時的な失敗だけで前の一覧から引き継げたとき(429 の待ち時間の
  上限に達していない・HF の検索が失敗していない・読めないテンプレートが無い・前の一覧と `license_policy` が同じ)に
  限って「一部欠け」の警告を本文の先頭に付けて PR にする(`weekly.partial_problems`)。それ以外の欠け方は、
  消えた・古いままの項目が「変化」に見えてしまうので PR にしない
- 試験が落ちたら PR を作らない。上流の変化で試験が落ちたとき(例: 2026-10 に HF のリポジトリの名前が変わり、
  出どころの表の当て直しが合わなくなった)は、直す PR を別に作る

### PR で確かめること

本文(`genmodelfit diff` と同じ中身)の「注意」と表を見る:

- **⚠ ライセンスが寛容な側に変わった項目**(段階が下がった、または許可リストに入った。不明 → 何か、も含む)。
  自動のおすすめに出る側の変化なので、`license_url` の原文で確かめる
- **自動のおすすめに入った・外れた項目**(両方の一覧に同じ方針を当てる: ComfyUI の版は新しい一覧の記録、
  HEAD の古さは見ない)。外れた項目に AI生成キットが使うものが無いか
- **新しく HEAD が失敗した URL**(gated になった・消えた)。前から失敗している URL は畳んである
- **消えた項目**(上流から消えた・検索の上位から外れた)
- **新しく無検閲の印が付いた項目**の根拠(未成年の性的表現を目的とするものは除外する。docs/sources.md)
- 実測のあるモデルのファイルが変わっていたら、`python -m genmodelfit.calibrate` で直す PR を別に作る
- **確認待ち(review)の項目**と、`review approve` で付けた根拠(`evidence`)の URL(#42。下の「毎週のスキル」)
- 全部の違いは artifact の `diff.json`(形は docs/schema.md「diff」)

### 毎週のスキル(genmodelfit-weekly、#42)

2026-10-21 のオーナーの決定: 毎週の更新は、オーナーの PC の起動スクリプトが行い(Windows のタスク スケジューラで
週 1 回)、確認待ちの判断だけを Claude Code のスキル `genmodelfit-weekly` に任せる。**weekly-data.yml の定期の実行(schedule)はやめた**。手動の予備
(workflow_dispatch)としてだけ残す。

動かし方(オーナーの PC。起動スクリプトと設定はリポジトリの外に置く):

- 起動スクリプト: `%USERPROFILE%\.claude\scripts\genmodelfit-weekly.ps1`(PowerShell 7 が要る: `#Requires -Version 7`)
- Claude の設定: `%USERPROFILE%\.claude\scripts\genmodelfit-review.settings.json`(Read は専用のクローンの中だけ、
  WebFetch は huggingface.co・github.com・raw.githubusercontent.com だけ)。Claude は `--restricted --tools Read Grep Glob WebFetch
  --strict-mcp-config` で呼ぶ(実行系の道具・ユーザーの設定・MCP は使えない)。approve は別のモデル(opus)が同じ制限で確かめ直す
- 専用のクローン: `C:\workspace\genmodelfit-weekly`(開発用の作業フォルダとは分ける。毎回 main に合わせ、venv を作り直す)
- タスクの登録(毎週月曜 9:00、ログオン中に実行):
  `schtasks /Create /TN genmodelfit-weekly /SC WEEKLY /D MON /ST 09:00 /TR "pwsh.exe -NoProfile -File %USERPROFILE%\.claude\scripts\genmodelfit-weekly.ps1"`
- ログ: `%LOCALAPPDATA%\genmodelfit-weekly\logs`(直近 26 回)。失敗したら Issue「毎週の更新が失敗しました」が立ち、次に成功したら閉じる
- 試すとき: `pwsh -File …\genmodelfit-weekly.ps1 -NoPush`(GitHub に何も出さない)。別の版の中身で試すなら `-Ref origin/<ブランチ> -NoPush`

役割を分ける:

- **Claude(スキル)はファイルを書き換えず、コマンドも動かさない**(2026-10-21 のオーナーの決定。読むだけと、
  Hugging Face・GitHub の WebFetch だけ)。実行役が渡す `review_queue.json` の確認待ちの項目
  (id・name・modality・reasons・ファイルごとの repo・path・license・license_url・homepage・card_url)を一次資料
  (ライセンスの原文・モデルカード・上流のリポジトリ)で確かめ、判断の一覧の JSON だけを返す:

  ```json
  {"decisions": [
    {"id": "hf:org/model", "action": "approve", "evidence": ["https://huggingface.co/org/model/blob/main/LICENSE"], "note": "Apache-2.0 の原文を確かめた"},
    {"id": "hf:org/other", "action": "reject", "evidence": ["https://huggingface.co/org/other/blob/main/LICENSE"], "note": "非商用だけ", "safety": false},
    {"id": "hf:org/unclear", "action": "hold", "note": "カードにライセンスの記載なし"}
  ]}
  ```

  理由が安全(未成年の性的表現など)の reject は `"safety": true`(名前を残さず sha256 だけ記録する)。
  分からないものは hold(pending のまま。自動のおすすめにも data-latest にも出ない)
- **実行役のスクリプト**(オーナーの PC。スキルの外)が、作り直し・試験・判断の当て込み・merge-check・PR を行う:
  1. `python -m genmodelfit.weekly --out <dir> --apply src/genmodelfit/data/db.json`(引き継ぎの元と比べる元は main の
     db.json)。終了コードが 1 なら止める(`result.json` の `problems`)
  2. `<dir>/review_queue.json` を Claude に渡し、判断の一覧を受け取る
  3. `python -m genmodelfit.review apply <decisions.json> --db src/genmodelfit/data/db.json --max-approve 10 --json`。
     形を厳しく確かめ(知らない鍵・action、確認待ちでない id、同じ id、根拠の無い approve、https でホストのない根拠、
     500 文字を超える・制御文字を含む note、approve の上限超え)、1 つでもおかしければ何も当てない(終了コード 2)。
     approve は `decided_by: claude-weekly`。出る要約(件数と項目ごとの結果)の文字は `weekly.md` で無害にしてあり、
     PR の本文にそのまま置ける
  4. ブランチにコミットして push し、PR を作る。**機械の確かめがすべて通ったときだけ**マージする:
     - 変わったファイルが `src/genmodelfit/data/db.json` だけ(approve・reject を当てた週は status が変わるので、
       下の条件で必ずオーナーが見る)
     - `python -m genmodelfit.weekly merge-check --base <main の db.json> --head <ブランチの db.json> --json` が
       終了コード 0(`safe: true`)。分類はファイルの JSON から計算し直す(スキルの報告を信じない)
     - 確認の印の status が変わっていない(承認・確認待ちへの変化は分類でも判断が要るになる)
     - 手元の試験(ruff と pytest。新しい一覧を入れた状態)が通った。PR では ci.yml が動かないので(手元で通してから
       push する決まり。2026-10-22)、CI は待たない。マージの後に main で ci.yml が動く

     どれかが通らなければマージせず、**オーナーが PR を見てマージする**(本文の「分類」「確認待ち」の節と、
     `review approve` が残した `evidence`)
- マージの後の署名と公開は data.yml が行う(いままでどおり)。**配る一覧からは確認待ちの項目を除く**
  (`python -m genmodelfit.review publish`。0.5.x は `review` を知らず、確認待ちを自動のおすすめに出してしまうため)

安全な週の条件: classification が safe(前から確認待ちのまま変わらない項目は含んでよい)。分類は**許すものの表**で
決める(docs/schema.md「diff」の classification): 人気の数(`downloads`・`downloads_all_time`・`likes`・テンプレートの
`usage`)、`updated`、HEAD の結果(`head_status`・`head_checked_at`)、ライセンスが厳しくなる、
消えた項目、自動のおすすめから外れた項目、`sources` の数、確認待ちのまま変わらない項目。それ以外の変化(知らない項目も)は
すべて判断が要る。HEAD が直って自動のおすすめに戻ったものは、どのファイルの URL・sha256 も変わっていないときだけ安全。
ファイルの大きさの変化は 0.9.0 から向きによらず判断が要る(`files_changed`。`update` も確認待ちにする)。アプリは一覧の URL を
そのまま落とすので、同じ URL のファイルが大きくなった = 中身が差し替わった、を黙って通さない。HF のリポジトリでファイルが
上書きされた週は、自動でマージされずオーナーの確認になる(以前は大きくなっただけなら安全な週としてマージしていた)。

決めたこと:

- pending の項目も main の一覧には入る(0.6.0 以降の `recommend` / `fit` / `info` には出る)。自動のおすすめ(gate)は
  `reviewed` で保留にする。アプリは `entry.review is None or entry.review.status == "ok"` で絞る
- 承認には https でホストのある根拠の URL が 1 つ以上要る(無ければ断る)
- `review reject` した項目は `src/genmodelfit/data/review_rejected.txt` に残り、`update` は作り直しても入れない
- 同梱の一覧は 0.6.0 で全部 `ok`(`decided_by: baseline-2026-10-21`)から始めた

### 手元で同じ流れを試す

```powershell
.venv\Scripts\python -m genmodelfit.weekly --dry-run --out $env:TEMP\weekly-out                         # 上流から(書くのは out だけ)
.venv\Scripts\python -m genmodelfit.weekly --dry-run --out $env:TEMP\weekly-out --record $env:TEMP\weekly-rec   # 応答も残す
.venv\Scripts\python -m genmodelfit.weekly --dry-run --out $env:TEMP\weekly-out2 --offline-sources $env:TEMP\weekly-rec  # 通信しない
.venv\Scripts\genmodelfit diff src\genmodelfit\data\db.json $env:TEMP\weekly-out\db.json
```

`--offline-sources` は、`--record` で残した GET の応答と HEAD の応答コードだけで作り直す(ファイルの見出しと
リポジトリの精度はキャッシュ `GENMODELFIT_CACHE_DIR` にあるものだけ)。HEAD の日時は実行した時刻になるので、
期限の確かめは本物にならない。失敗した応答(401・つながらない)は残らないので、読み直した結果は本物の実行と
少し違う(2026-10-21 の試しでは 335 件に対して 334 件)。流れを試す・PR の本文の形を見るためのもの。手元のキャッシュを汚したくなければ `GENMODELFIT_CACHE_DIR` を別の場所にする。
終了コード: 0 = 通った、1 = 確かめに落ちた(`result.json` の `problems`)、2 = 使い方の誤り。

### 作者が一度だけ設定すること

- Settings → Actions → General → Workflow permissions の **「Allow GitHub Actions to create and approve pull
  requests」を有効にする**(無効だと create-pull-request が失敗する)。確認:
  `gh api repos/laxshmate/genmodelfit/actions/permissions/workflow`(`can_approve_pull_request_reviews` が true)。
  この設定は名前のとおり**承認(approve)も許す**(GitHub に PR の作成だけを許す設定は無い)。weekly-data.yml は
  承認しない(`gh pr review` も auto-merge も使わない。tests/test_workflows.py が検査)。レビューとマージは作者が行う
- (定期の実行をやめたので今は関係しない)**60 日リポジトリに動き(コミットなど)が無いと、GitHub は schedule のワークフローを止める**。止まったら
  Actions の「Weekly data」で「Enable workflow」を押すか、手動実行(workflow_dispatch)すると戻る
- ラベル `automated`・`data-update` を作る(ワークフローも作ろうとするが、`pull-requests: write` だけでは
  作れないことがある): `gh label create automated -R laxshmate/genmodelfit`、`gh label create data-update -R laxshmate/genmodelfit`
- 同じリポジトリの PR では ci.yml のジョブは飛ばされる(#68)。ブランチの保護の必須のチェックに ci.yml を入れない。
  手元の `tools/ci_local.py` の結果を PR の説明に書く

## 鍵

- 公開鍵: `src/genmodelfit/data/signing_keys/*.pub`(パッケージに同梱。取り込む側はこのどれか 1 つで合えば通す)。
  今は `2026-10.pub` の 1 つ
- 秘密鍵: environment `data-signing` のシークレット `GENMODELFIT_SIGNING_KEY`(laxshmate/genmodelfit)
- 秘密鍵のバックアップ: `%USERPROFILE%\.config\genmodelfit\signing_key.pem`(1 か所だけ。ACL は本人のみ)。
  コミットしない。画面や Issue に貼らない。

### 鍵を替える(定期・または念のため)

公開鍵は版に同梱されるため、**その版が信じる鍵は版を出したときに決まる**。新旧の鍵を並べて同梱し、段階を踏んで替える。

1. 新しい鍵を作る(下記)。公開鍵を `src/genmodelfit/data/signing_keys/<年-月>.pub` として**足す**(古い鍵は残す)。
   この版を出し、利用者(特に AI生成キット)に上げてもらう。この時点では署名はまだ古い鍵。
   新旧どちらの版も、古い鍵の署名を通す。
2. 新しい版が普及したら、`data-signing` のシークレットを新しい鍵に替え、バックアップも置き換える。以後の署名は新しい鍵。
   手順 1 の版以降は通す。手順 1 より前の版は新しい署名を通せなくなる(`--from-release` が失敗し、手元のデータのまま動く)。
3. さらに後の版で古い公開鍵のファイルを消す。

鍵の作り方(秘密鍵はファイルにだけ書く。画面に出さない。上書きする前に古いバックアップを別名で残す):

```powershell
python -c "
import sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization as s
k = Ed25519PrivateKey.generate()
open(sys.argv[1], 'wb').write(k.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
open(sys.argv[2], 'wb').write(k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
" "$env:USERPROFILE\.config\genmodelfit\signing_key.new.pem" src\genmodelfit\data\signing_keys\2027-04.pub
icacls "$env:USERPROFILE\.config\genmodelfit" /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"
```

手順 2 でシークレットを替えるとき(値は標準入力から渡し、画面に出さない):

```powershell
cmd /c "gh secret set GENMODELFIT_SIGNING_KEY --env data-signing -R laxshmate/genmodelfit < ""%USERPROFILE%\.config\genmodelfit\signing_key.new.pem"""
```

### 鍵が漏れたら

署名の信頼が失われるので急ぐ。`data-latest` を消しても、鍵を持つ攻撃者は通信の途中に割り込める立場
(社内の TLS 検査の装置・乗っ取った DNS など)から偽の一覧を配れる。漏れた鍵を信じている版が残る限り
被害は止まらないので、**回復は鍵の入れ替えと新しい版の配布**で行う。

1. `data-signing` のシークレット `GENMODELFIT_SIGNING_KEY` を削除し、data.yml を無効にする。
2. 新しい鍵を作り、漏れた鍵の公開鍵ファイルを**消して**新しい公開鍵だけを同梱した版を**緊急で**出す
   (段階を踏む入れ替えはしない)。利用者(特に AI生成キット)に上げてもらう。
3. 漏れた鍵を信じている古い版には、`--from-release` を使わないよう案内する(同梱のデータだけで動く)。
4. シークレットを新しい鍵で入れ直し、data.yml で新しい鍵の署名の一覧を配る。

## 使わないものの確認

- PyPI の API トークンは作らない・保存しない(Trusted Publishing だけ)。
- 秘密鍵はリポジトリに入れない。リポジトリ全体のシークレットにも置かない。CI のログに出さない(`--key-env` で環境変数から読む)。
