"""配布された一覧(GitHub Release の data-latest)を取り込む。`genmodelfit update --from-release`。

手順: db.json と db.json.sig を取る → 同梱の信じる公開鍵のどれかで署名を確かめる →
形(schema_version)を確かめる → 前に取り込んだ配布より古くないか確かめる →
書き出し先(既定はキャッシュ)を原子的に置き換える。
どこかで失敗したら、今のキャッシュ(無ければ同梱のデータ)には一切触らずに RemoteError を出す。

古さの比較は、前に取り込んだ配布の generated_at を別ファイル(キャッシュの release_state.json)に残して行う。
キャッシュの db.json は手元の `genmodelfit update` も書くので、比べる相手にしない。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import __version__
from .db import Database, DatabaseError, cache_dir, cached_db_path, parse_db
from .signing import SigningError, decode_sig, verify

RELEASE_BASE_URL = "https://github.com/laxshmate/genmodelfit/releases/download/data-latest"
MAX_BYTES = 64 * 1024 * 1024  # 一覧としては十分大きい上限
RETRY_DELAY = 5.0  # 署名が合わないときに 1 回だけ取り直すまでの秒数(資産の置き換えの途中に当たった場合)
STATE_NAME = "release_state.json"


class RemoteError(DatabaseError):
    pass


def state_path() -> Path:
    return cache_dir() / STATE_NAME


def _get(url: str, timeout: float) -> bytes:
    """url を取る。MAX_BYTES を超えたら読むのをやめて RemoteError。"""
    try:
        r = requests.get(
            url, timeout=timeout, stream=True, headers={"User-Agent": f"genmodelfit/{__version__}"}
        )
        try:
            r.raise_for_status()
            length = r.headers.get("Content-Length")
            if length and length.isdigit() and int(length) > MAX_BYTES:
                raise RemoteError(f"ファイルが大きすぎます: {url}")
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                buf.extend(chunk)
                if len(buf) > MAX_BYTES:
                    raise RemoteError(f"ファイルが大きすぎます: {url}")
        finally:
            r.close()
    except requests.RequestException as e:
        raise RemoteError(f"ダウンロードできませんでした: {url}: {e}") from e
    return bytes(buf)


def _atomic_write(path: Path, data: bytes) -> None:
    """一時ファイルに書いてから置き換える。失敗したら RemoteError(元のファイルはそのまま)。"""
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        tmp = None
    except OSError as e:
        raise RemoteError(
            f"書き込めませんでした: {path}: {e}"
            "(ほかのプログラムがこのファイルを開いていないか確かめて、もう一度実行してください)"
        ) from e
    finally:
        if tmp is not None:
            Path(tmp).unlink(missing_ok=True)


_FRACTION = re.compile(r"(\.\d+)")


def parse_time(value: Any) -> datetime:
    """ISO 8601 の日時を UTC の aware な datetime にする。末尾の Z も受け付ける。

    タイムゾーンの無い日時・形の違うものは ValueError(Python 3.10 の fromisoformat に合わせて、
    小数の秒は 6 桁にそろえてから読む)。
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"日時がありません: {value!r}")
    s = value.strip()
    if s[-1] in "Zz":
        s = s[:-1] + "+00:00"
    s = _FRACTION.sub(lambda m: "." + (m.group(1)[1:] + "000000")[:6], s, count=1)
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"日時にタイムゾーンがありません: {value!r}")
    return dt.astimezone(timezone.utc)


def _read_state() -> datetime | None:
    """前に取り込んだ配布の generated_at。無い・壊れているときは None(次に取り込むと書き直す)。"""
    try:
        return parse_time(json.loads(state_path().read_text(encoding="utf-8"))["generated_at"])
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _fetch_verified(base: str, timeout: float) -> bytes:
    """db.json と署名を取り、署名が合えば db.json を返す。合わなければ少し待って 1 回だけ取り直す。

    data.yml は db.json と db.json.sig を順に置き換えるので、その途中に取ると組み合わせがずれることがある。
    """
    for attempt in range(2):
        data = _get(f"{base}/db.json", timeout)
        sig_text = _get(f"{base}/db.json.sig", timeout)
        try:
            sig = decode_sig(sig_text)
        except SigningError:
            sig = b""
        if sig and verify(data, sig):
            return data
        if attempt == 0:
            time.sleep(RETRY_DELAY)
    if not sig:
        raise RemoteError("署名ファイルの形が正しくありません(取り込みません)")
    raise RemoteError("署名が合いません。改ざんされたか、壊れた一覧です(取り込みません)")


def update_from_release(
    base_url: str = RELEASE_BASE_URL,
    *,
    timeout: float = 30,
    dest: Path | None = None,
    force: bool = False,
) -> Database:
    """署名つきの一覧を取り、確かめてから dest(既定はキャッシュ)に置く。取り込んだ一覧を返す。

    force=True なら、前に取り込んだ配布より古くても取り込む(記録の日時がずれたときの戻し方)。
    """
    data = _fetch_verified(base_url.rstrip("/"), timeout)
    try:
        raw = json.loads(data.decode("utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("一覧が JSON のオブジェクトではありません")
        db = parse_db(raw, "cache")
        new_time = parse_time(raw.get("generated_at"))
    except (ValueError, TypeError, KeyError, AttributeError, DatabaseError) as e:
        raise RemoteError(f"一覧の形が正しくありません: {e}") from e
    last = _read_state()
    if last is not None and new_time < last and not force:
        raise RemoteError(
            f"前に取り込んだ一覧より古い一覧です({new_time.isoformat()} < {last.isoformat()})。"
            "取り込みません(わざと戻すときは --force)"
        )
    # 記録を先に進める。後の書き込みが失敗しても、同じ配布はもう一度取り込める(同じ日時は通す)
    state = {"generated_at": new_time.isoformat(), "accepted_at": datetime.now(timezone.utc).isoformat()}
    _atomic_write(state_path(), (json.dumps(state) + "\n").encode("utf-8"))
    _atomic_write(dest or cached_db_path(), data)
    return db
