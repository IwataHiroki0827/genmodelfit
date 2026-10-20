# リリースとデータの配り方

2 つの流れがあります。

| 流れ | 何を配る | きっかけ | 配り先 |
|---|---|---|---|
| 版のリリース | パッケージ(sdist と wheel) | `v*` のタグを push → ユーザの承認 | PyPI と GitHub Release |
| データの配布 | モデル一覧 `db.json` と署名 `db.json.sig` | main の db.json が変わった push・手動実行 | GitHub Release `data-latest` |

データの配布は `v*` のタグではないので、PyPI には出ません。

## GitHub の environment(2026-10-20 に設定)

| environment | 使えるところ | 承認 | シークレット |
|---|---|---|---|
| `pypi` | `v*` のタグだけ | 必須(ユーザ `KoyoYeager`) | なし(Trusted Publishing) |
| `data-signing` | `main` ブランチだけ | なし | `GENMODELFIT_SIGNING_KEY`(署名の秘密鍵) |

リポジトリ全体のシークレットには何も置かない。秘密鍵は `data-signing` だけにあるので、main 以外のブランチや
PR のワークフローからは読めない。data.yml でも、署名する 1 ステップにだけ環境変数で渡す。

設定の確認: `gh api repos/laxshmate/genmodelfit/environments` と
`gh secret list -R laxshmate/genmodelfit`(空であること)・`gh secret list --env data-signing -R laxshmate/genmodelfit`。

## 最初の公開(v0.1.0)の手順 — ユーザが行う

**公開はユーザの承認を得てから。Claude は承認なしにタグを push しない・PR をマージしない。**
2026-10-20 に確認: environment `pypi`(`v*` のタグだけ・承認者 `KoyoYeager`)と `data-signing`(`main` だけ・
シークレット `GENMODELFIT_SIGNING_KEY` あり)はそろっている。リポジトリ全体のシークレットは空。PyPI の
`genmodelfit` はまだ無い(404)。

1. **PR をマージする**(この順。どれも前の PR のブランチから作ってある):
   #8 → #13 → #14 → #10 → #11 → #12 → #9 → #16 → #17 → #18(この Issue の PR)。
   CI(Python 3.10〜3.13 × Ubuntu / Windows、wheel)が全部緑であることを確かめてからマージする。
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
   - publish: environment `pypi` で**ユーザの承認を待つ**。Actions の画面の「Review deployments」で `pypi` を承認すると、
     PyPI に公開する(Trusted Publishing。API トークンは使わない)
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
   `CHANGELOG.md` に書いて PR で main に入れる。
2. 承認後、main の先頭にタグ `v<版>` を付けて push する(上の 3〜5 と同じ)。

手動実行(workflow_dispatch)はビルドまでで、公開しない。タグを選んで手動実行しても、publish と github-release は
`github.event_name == 'push'` の条件で飛ばす。タグを打つ前の試し用。

PyPI 側の準備は済み。アカウント `laxshmate`(2 段階認証あり)に、`genmodelfit` / `laxshmate/genmodelfit` / `release.yml` /
environment `pypi` の Trusted Publisher を事前登録してある。事前登録では名前は確保されず、最初の公開で確定する。
**PEP 541(放棄された名前の扱い)があるので、中身の無い仮の版で名前だけ押さえない。動く v0.1 を公開する。**

公開した版は消せない(取り下げ=yank だけ)。版の番号は使い回せないので、直すなら次の版を出す。

## データの配布

`.github/workflows/data.yml` は、**main にコミットされた `src/genmodelfit/data/db.json` をそのまま**署名して配る
(上流から作り直さない。作り直して PR にする流れと毎日の実行は Issue #19)。きっかけは、main の db.json が変わった
push と手動実行。

1. sign: environment `data-signing`(main だけ)で、コミットされた一覧を読めることを確かめ、署名し、同梱の公開鍵で
   確かめる(`contents: read`。秘密鍵はこのジョブの「署名する」ステップだけ)。main 以外のブランチを選んだ手動実行は
   `if: github.ref == 'refs/heads/main'` で動かない
2. upload: Release `data-latest`(prerelease)の `db.json` と `db.json.sig` を置き換える(ここだけ `contents: write`)

checkout は `persist-credentials: false`。

同梱の一覧の HEAD の確認は 30 日で切れる(切れると自動のおすすめが 0 件になる)。切れる前に
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
