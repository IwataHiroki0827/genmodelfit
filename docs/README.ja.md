# genmodelfit の文書

[English](README.md) · **日本語**

まず [README](../README.ja.md) を読んでください(入れ方・コマンド・結果の読み方)。
下のページは、その詳しい決まりです。どのページも、同じ内容の英語版と日本語版があります。

## 目的から探す

| したいこと | 読むページ |
|---|---|
| この PC をどう調べて判定するかを知る | [hardware.ja.md](hardware.ja.md) |
| メモリの見積もりをどこまで信じてよいかを知る | [accuracy.ja.md](accuracy.ja.md) |
| 並び・Score の決め方と、実測で段階を決める条件を知る | [scoring.ja.md](scoring.ja.md) |
| モデルのファイルを安全にダウンロードする | [download.ja.md](download.ja.md) |
| アプリから使う(`--json` か Python) | [schema.ja.md](schema.ja.md)、続けて [gate.ja.md](gate.ja.md) と [download.ja.md](download.ja.md) |
| 実測を送る | [CONTRIBUTING.ja.md](../CONTRIBUTING.ja.md) |
| モデルの一覧がどこから来るかを知る | [sources.ja.md](sources.ja.md) |
| 版を出す・一覧を更新する(メンテナ) | [release.ja.md](release.ja.md) |

## すべてのページ

| ページ | 中身 |
|---|---|
| [hardware.ja.md](hardware.ja.md) | GPU・メモリ・ドライバの検出方法と、判定の根拠にした一次資料の規則。 |
| [accuracy.ja.md](accuracy.ja.md) | VRAM・メモリの見積もりの式・合わせ方・精度と、ほかの利用者の実測の扱い。 |
| [scoring.ja.md](scoring.ja.md) | 判定の公平さ・一覧の並び・Score(メモリの余裕)・人気の決まり、根拠、llmfit との違い、試験。 |
| [download.ja.md](download.ja.md) | `genmodelfit download` の決まり(同意・断るもの・安全・通信)と Python の関数。 |
| [schema.ja.md](schema.ja.md) | `--json` の各コマンドの出力、Python の公開の範囲、`db.json` の形。 |
| [gate.ja.md](gate.ja.md) | 自動でおすすめに出してよいか(`auto: true`)の判定: 条件・ライセンスの許可リスト・リンクの確認の新しさ・ComfyUI の版。 |
| [sources.ja.md](sources.ja.md) | `genmodelfit update` が一覧を作るときのデータ源と規則(テンプレート・ライセンス・言語・無検閲・除外)。 |
| [release.ja.md](release.ja.md) | メンテナ向け: パッケージのリリース、署名つきのデータ配布、毎週の一覧の更新、署名の鍵。 |
| [license_requests.ja.md](license_requests.ja.md) | 重みのライセンスが分からないモデルの本家に、記載をお願いするための下書き(Issue #30)。 |
| [CONTRIBUTING.ja.md](../CONTRIBUTING.ja.md) | `genmodelfit bench` で測り、中身を確かめ、Pull Request で送る手順。 |
| [CHANGELOG.md](../CHANGELOG.md) | 版ごとの変更点(英語のみ)。 |
