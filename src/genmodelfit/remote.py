"""配布された一覧(GitHub Release の data-latest)を取り込む。`genmodelfit update --from-release`。

手順: db.json と db.json.sig を取る → 同梱の公開鍵で署名を確かめる → 形(schema_version)を確かめる →
古い一覧でないか確かめる → キャッシュを原子的に置き換える。
どこかで失敗したら、今のキャッシュ(無ければ同梱のデータ)には一切触らずに RemoteError を出す。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import requests

from . import __version__
from .db import Database, DatabaseError, cached_db_path, parse_db
from .signing import SigningError, decode_sig, verify

RELEASE_BASE_URL = "https://github.com/laxshmate/genmodelfit/releases/download/data-latest"
MAX_BYTES = 64 * 1024 * 1024  # 一覧としては十分大きい上限


class RemoteError(DatabaseError):
    pass


def _get(url: str, timeout: float) -> bytes:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": f"genmodelfit/{__version__}"})
        r.raise_for_status()
        body = r.content
    except requests.RequestException as e:
        raise RemoteError(f"ダウンロードできませんでした: {url}: {e}") from e
    if len(body) > MAX_BYTES:
        raise RemoteError(f"ファイルが大きすぎます: {url}")
    return body


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def update_from_release(
    base_url: str = RELEASE_BASE_URL, *, timeout: float = 30, dest: Path | None = None
) -> Database:
    """署名つきの一覧を取り、確かめてからキャッシュに置く。取り込んだ一覧を返す。"""
    base = base_url.rstrip("/")
    data = _get(f"{base}/db.json", timeout)
    sig_text = _get(f"{base}/db.json.sig", timeout)
    try:
        sig = decode_sig(sig_text)
    except SigningError as e:
        raise RemoteError(str(e)) from e
    if not verify(data, sig):
        raise RemoteError("署名が合いません。改ざんされたか、壊れた一覧です(取り込みません)")
    try:
        db = parse_db(json.loads(data.decode("utf-8")), "cache")
    except (ValueError, TypeError, KeyError, DatabaseError) as e:
        raise RemoteError(f"一覧の形が正しくありません: {e}") from e
    target = dest or cached_db_path()
    if target.exists():
        try:
            old = json.loads(target.read_text(encoding="utf-8")).get("generated_at")
        except (OSError, ValueError, AttributeError):
            old = None
        if old and db.generated_at and str(db.generated_at) < str(old):
            raise RemoteError(f"手元より古い一覧です({db.generated_at} < {old})。取り込みません")
    _atomic_write(target, data)
    return db


def run_from_release(args: argparse.Namespace) -> int:
    db = update_from_release()
    if getattr(args, "json", False):
        out = {
            "genmodelfit": __version__,
            "command": "update",
            "models": len(db.models),
            "generated_at": db.generated_at,
        }
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    else:
        print(f"一覧を更新しました: {len(db.models)} 件(生成日時 {db.generated_at}) -> {cached_db_path()}")
    return 0
