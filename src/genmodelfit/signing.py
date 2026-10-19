"""データの署名(Ed25519)。

配る一覧(db.json)に署名(db.json.sig)を付け、取り込む側が埋め込みの公開鍵で確かめる。
署名の鍵(秘密鍵)は GitHub Actions のシークレット `GENMODELFIT_SIGNING_KEY` にだけ置く(docs/release.md)。

コマンド(配る側の CI が使う):
    python -m genmodelfit.signing sign <ファイル> [--key-env 環境変数名]   # <ファイル>.sig を書く
    python -m genmodelfit.signing verify <ファイル> [<署名ファイル>]       # 同梱の公開鍵で確かめる
"""

from __future__ import annotations

import argparse
import base64
import binascii
import os
import sys
from importlib import resources
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


class SigningError(ValueError):
    pass


def embedded_public_key_pem() -> bytes:
    return resources.files("genmodelfit.data").joinpath("signing_key.pub").read_bytes()


def load_public_key(key: bytes | str | Ed25519PublicKey | None = None) -> Ed25519PublicKey:
    """公開鍵を読む。None なら同梱の鍵。PEM か、素の 32 バイトを受け付ける。"""
    if isinstance(key, Ed25519PublicKey):
        return key
    raw = embedded_public_key_pem() if key is None else key.encode() if isinstance(key, str) else key
    try:
        if not raw.lstrip().startswith(b"-----BEGIN"):
            return Ed25519PublicKey.from_public_bytes(raw)
        pub = serialization.load_pem_public_key(raw)
    except ValueError as e:
        raise SigningError(f"公開鍵を読めません: {e}") from e
    if not isinstance(pub, Ed25519PublicKey):
        raise SigningError("Ed25519 の公開鍵ではありません")
    return pub


def load_private_key(pem: bytes | str) -> Ed25519PrivateKey:
    raw = pem.encode() if isinstance(pem, str) else pem
    try:
        key = serialization.load_pem_private_key(raw.strip(), password=None)
    except (ValueError, TypeError) as e:
        raise SigningError("秘密鍵を読めません(PEM 形式の Ed25519 鍵が要ります)") from e
    if not isinstance(key, Ed25519PrivateKey):
        raise SigningError("Ed25519 の秘密鍵ではありません")
    return key


def sign(data: bytes, private_key_pem: bytes | str) -> bytes:
    """data に署名する。返すのは 64 バイトの署名。"""
    return load_private_key(private_key_pem).sign(data)


def verify(data: bytes, sig: bytes, public_key: bytes | str | Ed25519PublicKey | None = None) -> bool:
    """署名が正しければ True。壊れた署名・別の鍵・改ざんは False(例外は出さない)。"""
    try:
        load_public_key(public_key).verify(sig, data)
    except (InvalidSignature, SigningError):
        return False
    return True


def encode_sig(sig: bytes) -> str:
    """`.sig` ファイルに書く形(base64 の 1 行)。"""
    return base64.b64encode(sig).decode("ascii") + "\n"


def decode_sig(text: bytes | str) -> bytes:
    try:
        return base64.b64decode(text.strip(), validate=True)
    except (binascii.Error, ValueError) as e:
        raise SigningError("署名ファイルの形が正しくありません") from e


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m genmodelfit.signing")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sign", help="<ファイル>.sig を書く")
    sp.add_argument("file")
    sp.add_argument("--key-env", default="GENMODELFIT_SIGNING_KEY", help="秘密鍵(PEM)を入れた環境変数")
    sp.add_argument("--key-file", help="環境変数の代わりに鍵のファイルを使う")
    vp = sub.add_parser("verify", help="同梱の公開鍵で確かめる")
    vp.add_argument("file")
    vp.add_argument("sig", nargs="?")
    args = p.parse_args(argv)
    path = Path(args.file)
    try:
        if args.cmd == "sign":
            pem = Path(args.key_file).read_bytes() if args.key_file else os.environ.get(args.key_env, "")
            if not pem:
                print(f"error: 秘密鍵がありません(環境変数 {args.key_env})", file=sys.stderr)
                return 1
            out = path.with_name(path.name + ".sig")
            out.write_text(encode_sig(sign(path.read_bytes(), pem)), encoding="ascii")
            print(f"署名しました: {out}")
            return 0
        sig_path = Path(args.sig) if args.sig else path.with_name(path.name + ".sig")
        ok = verify(path.read_bytes(), decode_sig(sig_path.read_text(encoding="ascii")))
    except (OSError, SigningError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print("署名は正しいです" if ok else "署名が合いません")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
