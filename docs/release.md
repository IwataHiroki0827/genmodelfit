# リリースとデータの配り方

2 つの流れがあります。

| 流れ | 何を配る | きっかけ | 配り先 |
|---|---|---|---|
| 版のリリース | パッケージ(sdist と wheel) | `v*` のタグを push → ユーザの承認 | PyPI と GitHub Release |
| データの配布 | モデル一覧 `db.json` と署名 `db.json.sig` | main からの手動実行(毎日の実行は Issue #2 の後) | GitHub Release `data-latest` |

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

## 版を出す手順

**公開はユーザの承認を得てから。Claude は承認なしにタグを push しない。**

1. `pyproject.toml` の `version` と `src/genmodelfit/__init__.py` の `__version__` を同じ値にして PR で main に入れる(例 `0.1.0`)。
2. 承認後、main の先頭にタグを付けて push する。

   ```powershell
   git switch main; git pull
   git tag v0.1.0
   git push origin v0.1.0
   ```
3. `.github/workflows/release.yml` が次の順に進む。
   - test: ruff と pytest
   - build: タグと両方の版が一致するか確認 → `uv build` → `twine check --strict`
   - publish: environment `pypi` で**ユーザの承認を待つ**。承認されたら PyPI に公開(Trusted Publishing。API トークンは使わない)
   - github-release: 同じ dist を付けた GitHub Release を作る
4. Actions の画面で全部緑になったら、`pip install genmodelfit==0.1.0` で確かめる。

手動実行(workflow_dispatch)はビルドまでで、公開しない。タグを選んで手動実行しても、publish と github-release は
`github.event_name == 'push'` の条件で飛ばす。タグを打つ前の試し用。

PyPI 側の準備は済み。アカウント `laxshmate`(2 段階認証あり)に、`genmodelfit` / `laxshmate/genmodelfit` / `release.yml` /
environment `pypi` の Trusted Publisher を事前登録してある。事前登録では名前は確保されず、最初の公開で確定する。
**PEP 541(放棄された名前の扱い)があるので、中身の無い仮の版で名前だけ押さえない。動く v0.1 を公開する。**

公開した版は消せない(取り下げ=yank だけ)。版の番号は使い回せないので、直すなら次の版を出す。

## データの配布

`.github/workflows/data.yml` は 3 つのジョブに分かれる。

1. build-data: `genmodelfit update --output dist-data/db.json` で一覧を作る(`contents: read`)
2. sign: environment `data-signing` で署名し、同梱の公開鍵で確かめる(`contents: read`。秘密鍵はこのジョブの署名のステップだけ)
3. upload: Release `data-latest`(prerelease)の `db.json` と `db.json.sig` を置き換える(ここだけ `contents: write`)

checkout は `persist-credentials: false`。`genmodelfit.update` は Issue #2 で入る。それまでは「まだありません」と
明示して失敗する(動かさなくてよい)。

**毎日の実行(schedule)は今は外してある。Issue #2 が入ったら data.yml の `on:` に次を戻す。**

```yaml
  schedule:
    - cron: "17 3 * * *" # 毎日 03:17 UTC
```

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
