---
name: genmodelfit-weekly
description: genmodelfit の一覧を毎週作り直し、変化を「安全な向き」と「判断が要る」に分けて、安全な週はマージ、判断が要る週は一次資料の根拠を付けた PR にする。タスク スケジューラから `claude -p "/genmodelfit-weekly"` で起動される前提。手で「毎週の更新を回して」と言われたときも使う。
---

# genmodelfit の毎週の更新

オーナーの決定(2026-10-21): 週 1 回、オーナーの PC のタスク スケジューラがこのスキルを動かす。
**安全な向きの変化だけの週は Claude がマージする。判断が要る変化があった週は、根拠を付けた PR を作り、オーナーがマージする。**

専用のクローン(既定 `C:\workspace\genmodelfit-weekly`)で動く。開発用の作業フォルダには触らない。

## 守ること(破ったら止まる)

- 署名の鍵(`%USERPROFILE%\.config\genmodelfit\signing_key.pem`)を読まない。PyPI に出さない。タグを打たない。
- `git push --force` をしない。main に直接 push しない(必ず PR を通してマージする)。
- 試験を飛ばさない。試験が落ちたら直そうとせず、失敗として報告して止まる(直すのは人と開発のセッション)。
- ライセンスを寛容な側に変えるのは、**一次資料**(元のモデルのカード・LICENSE・公式の GitHub・sha256 の一致)で確かめたときだけ。推測しない。迷ったら `pending` のまま残す(自動おすすめに出ないので害は無い)。
- 無検閲のモデルの中身(画像・本文)は開かない。名前・タグ・カードのメタデータだけで判断する。未成年の性的表現を目的とするもの・学習データにその分類を含むもの(safety.py の規則)は reject する。
- 1 回の実行で approve するのは 10 件まで。それを超える分は次の週に回す(オーナーが PR で追いやすくするため)。

## 手順

1. **準備**(起動スクリプトが main に合わせ、venv を入れ直してある前提)
   - `git status` が空で、ブランチが main であること。違えば失敗として止まる。
   - `.venv\Scripts\genmodelfit.exe --version` を記録する。
2. **作り直す**: `.venv\Scripts\python.exe -X utf8 -m genmodelfit.weekly --out weekly-out --apply src\genmodelfit\data\db.json`
   (確かめを通った一覧を同梱の db.json に写す。上流の取得のキャッシュは genmodelfit のキャッシュのフォルダに残り、次の週に使われる)
   - 終了コード 1(検査で止まった)なら、`weekly-out\result.json` の理由を報告して止まる(手順 7 の失敗の扱い)。
   - 終了コード 2 は使い方の誤り。スキルの不具合として報告して止まる。
3. **分類を読む**: `weekly-out\diff.json` の `classification`。
   - `safe: true` かつ確認待ち(`weekly-out\review_queue.json`)が空 → 手順 5 へ(安全な週)。
   - そうでなければ手順 4 へ。
4. **確認待ちを調べる**(判断が要る週)
   - `review_queue.json` の各項目を、理由ごとに調べる。調べは Agent(model: sonnet)に、1 件ずつ「SPEC・受け入れ基準・報告形式」を渡して任せてよい。
     - `new`: 一覧に新しく入った。ファイルごとのライセンスを、元のモデルまでたどって確かめる(docs/sources.md の規則と `license_sources.py` の表のやり方)。
     - `license_permissive`: ライセンスが緩い向きに変わった。変わった根拠が一次資料で本当か。
     - `need_decreased`: 必要量が減った。精度・大きさの読み取り(ヘッダ・Hub API)が正しいか。
     - `uncensored_new`: 新しく無検閲になった。safety.py の規則に当たらないか(名前・タグ・カードのメタデータだけ)。
   - **寛容な側にする判断(approve でライセンスが許可リストに入るもの)は、必ず別の Agent(model: opus)に一次資料で確かめ直させる。** 食い違ったら approve しない。
   - 決めたものを記録する:
     - 確かめられた: `genmodelfit review approve <id> --evidence <URL> [--evidence <URL>] --note "<要点>" --by claude-weekly --db src\genmodelfit\data\db.json`
     - 一覧から外すべき: `genmodelfit review reject <id> --note "<理由>" --db src\genmodelfit\data\db.json`
     - 確かめられない: 何もしない(`pending` のまま。PR の本文に「確かめられなかった理由」を書く)
   - ライセンスの表に足すべき根拠が見つかったら、`license_sources.py` に行を足し、その試験(`tests/test_license_sources.py`)も直す。
5. **試験**: `.venv\Scripts\ruff.exe check .` と `.venv\Scripts\python.exe -m pytest -q`。落ちたら手順 7。
6. **PR とマージ**
   - ブランチ `weekly/<YYYY-MM-DD>` を切り、変更をコミットする(本文の末尾に `Co-Authored-By: Claude <noreply@anthropic.com>`)。
   - `gh pr create -R laxshmate/genmodelfit -B main -H weekly/<日付> -t "毎週の一覧の更新 <日付>" -F <本文>` で PR を作る。本文は `weekly-out\pr_body.md` に、次を足したもの:
     - 冒頭に「安全な週(Claude がマージ)」か「判断が要る週(オーナーの確認を待つ)」かを 1 行で
     - approve / reject / 保留にした項目と、それぞれの根拠の URL(上流の文字列は weekly.md と同じく無害化する)
   - CI(`gh pr checks <番号> -R laxshmate/genmodelfit --watch`)が全部緑になるのを待つ。
   - **安全な週で、approve も reject もしていない**なら、`gh pr merge <番号> -R laxshmate/genmodelfit --merge` でマージする。マージすると data.yml が署名して `data-latest` に配る。その run が成功したかも確かめる。
   - **判断が要る週**(approve・reject・license_sources の変更のどれかがある)は、マージしない。PR にラベル `data-update` を付け、オーナーの確認を待つ。
7. **失敗の扱い**: 手順のどこかで止まったら、Issue「毎週の更新が失敗しました(<日付>)」を作るか、同じ題の開いた Issue があればコメントを足す。失敗した手順・終了コード・ログの末尾 50 行(無害化して)を書く。次に成功した週に、その Issue を閉じる。
8. **最後に**: 結果を 5 行以内で出力する(安全な週 / 判断が要る週 / 失敗、PR の URL、マージしたか、確認待ちの件数)。起動スクリプトがログに残す。
