# リリースとデータの配り方

2 つの流れがあります。

| 流れ | 何を配る | きっかけ | 配り先 |
|---|---|---|---|
| 版のリリース | パッケージ(sdist と wheel) | `v*` のタグを push | PyPI と GitHub Release |
| データの配布 | モデル一覧 `db.json` と署名 `db.json.sig` | 毎日 03:17 UTC と手動 | GitHub Release `data-latest` |

データの配布は `v*` のタグではないので、PyPI には出ません。

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
   - publish: environment `pypi` で承認を待たず PyPI に公開(Trusted Publishing。API トークンは使わない)
   - github-release: 同じ dist を付けた GitHub Release を作る
4. Actions の画面で全部緑になったら、`pip install genmodelfit==0.1.0` で確かめる。

手動実行(workflow_dispatch)はビルドまでで、公開しない。タグを打つ前の試し用。

PyPI 側の準備は済み。アカウント `laxshmate`(2 段階認証あり)に、`genmodelfit` / `laxshmate/genmodelfit` / `release.yml` /
environment `pypi` の Trusted Publisher を事前登録してある。事前登録では名前は確保されず、最初の公開で確定する。
**PEP 541(放棄された名前の扱い)があるので、中身の無い仮の版で名前だけ押さえない。動く v0.1 を公開する。**

公開した版は消せない(取り下げ=yank だけ)。版の番号は使い回せないので、直すなら次の版を出す。

## データの配布

`.github/workflows/data.yml` が `genmodelfit update --output dist-data/db.json` で一覧を作り、署名して
Release `data-latest`(prerelease)の `db.json` と `db.json.sig` を置き換える。`genmodelfit.update` は Issue #2 で入る。
それまでは「まだありません」と明示して失敗する(動かさなくてよい)。

使う側:

```powershell
genmodelfit update --from-release
```

```python
from genmodelfit.remote import update_from_release
db = update_from_release()   # 失敗したら RemoteError(DatabaseError の仲間)
```

取り込みの手順は、取得 → 同梱の公開鍵で署名を確認 → `schema_version` を確認 → 手元より古くないか確認 → キャッシュを原子的に置き換え。
どれかが失敗したら、手元のキャッシュ(無ければ同梱のデータ)はそのままで、メッセージを出して終了コード 1。

署名は Ed25519。`db.json.sig` は 64 バイトの署名を base64 にした 1 行。
署名の確認だけ手でやるなら `python -m genmodelfit.signing verify db.json`。

## 鍵

- 公開鍵: `src/genmodelfit/data/signing_key.pub`(パッケージに同梱。取り込む側はこれで確かめる)
- 秘密鍵: GitHub Actions のシークレット `GENMODELFIT_SIGNING_KEY`(laxshmate/genmodelfit)
- 秘密鍵のバックアップ: `%USERPROFILE%\.config\genmodelfit\signing_key.pem`(1 か所だけ。ACL は本人のみ)。
  コミットしない。画面や Issue に貼らない。

### 鍵を替える(定期・または念のため)

公開鍵は版に埋め込まれているため、**古い版を使っている人は新しい鍵の署名を通せない**(古い版は古い鍵の署名だけを信じる)。
替えるときは次の順にする。

1. 新しい鍵を作る(下記)。公開鍵 `signing_key.pub` を差し替えた版を先に出し、利用者に上げてもらう。
2. 移行期間は、`data-latest` を古い鍵の署名のままにして、新しい版が普及するのを待つ(古い鍵が漏れていないなら)。
3. シークレットを新しい鍵に替え、バックアップも置き換える。以後の署名は新しい鍵。

鍵の作り方(秘密鍵はファイルにだけ書く。画面に出さない):

```powershell
python -c "
import sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization as s
k = Ed25519PrivateKey.generate()
open(sys.argv[1], 'wb').write(k.private_bytes(s.Encoding.PEM, s.PrivateFormat.PKCS8, s.NoEncryption()))
open(sys.argv[2], 'wb').write(k.public_key().public_bytes(s.Encoding.PEM, s.PublicFormat.SubjectPublicKeyInfo))
" "$env:USERPROFILE\.config\genmodelfit\signing_key.pem" src\genmodelfit\data\signing_key.pub
icacls "$env:USERPROFILE\.config\genmodelfit" /inheritance:r /grant:r "$($env:USERNAME):(OI)(CI)F"
Get-Content "$env:USERPROFILE\.config\genmodelfit\signing_key.pem" -Raw | gh secret set GENMODELFIT_SIGNING_KEY -R laxshmate/genmodelfit
```

(上書きする前に、古いバックアップを別名で残しておく。)

### 鍵が漏れたら

署名の信頼が失われるので急ぐ。

1. まず `data-latest` の Release を消す(または data.yml を無効にする)。偽の一覧を作るには、攻撃者がその置き場にも書き込む必要があるので、置き場を止めれば被害が止まる。
2. GitHub のシークレット `GENMODELFIT_SIGNING_KEY` を削除する。
3. 新しい鍵を作り、公開鍵を差し替えた版を **緊急で** 出す。利用者(特に AI生成キット)に上げてもらう。
4. 漏れた鍵の公開鍵を信じている古い版は、`--from-release` を使わないよう案内する(同梱のデータだけで動く)。
5. シークレットを新しい鍵で入れ直し、`data.yml` を再開する。

## 使わないものの確認

- PyPI の API トークンは作らない・保存しない(Trusted Publishing だけ)。
- 秘密鍵はリポジトリに入れない。CI のログに出さない(`--key-env` で環境変数から読む)。
