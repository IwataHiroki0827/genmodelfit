"""bench: 動いている ComfyUI に HTTP でワークフローを送り、最大 VRAM・メモリ・時間を測る。

ComfyUI のコードは import しない(GPL のため)。使うのは HTTP の API だけ:
`/system_stats` `/object_info` `/queue` `/free` `/prompt` `/history/<id>` `/interrupt`
`/upload/image` `/templates/<name>.json` `/internal/logs/raw`。

測り方:
- 最大 VRAM(peak_vram_gb): 生成中に `/system_stats` を短い間隔で読み、torch が確保した量
  (`torch_vram_total`)の最大値。ComfyUI のプロセスの分で、CUDA の文脈(約 0.3〜0.5 GB)は含まない
- GPU 全体: `nvidia-smi` の使用量の最大値から、開始前の値を引いたもの(他のアプリの増減も入る)。note に書く
- メモリ(peak_ram_gb): ポートで待ち受けている ComfyUI のプロセス(と子)の常駐メモリの最大値(同じ PC のときだけ)
- 時間(seconds): ワークフローを送ってから終わるまで(モデルの読み込みを含む)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import psutil
import requests

from . import __version__
from .db import cache_dir
from .hardware import _nvidia_smi
from .models import Measurement
from .workflow import (
    WorkflowError,
    apply_settings,
    check_inputs,
    effective_settings,
    is_api_format,
    set_output_prefix,
    template_input_url,
    template_url,
    ui_to_api,
    upload_inputs,
)

GIB = 1024**3
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


class BenchError(RuntimeError):
    """測れなかった理由(利用者に見せる文)。"""


# --- ComfyUI との HTTP ------------------------------------------------------------------------


class ComfyClient:
    def __init__(self, url: str, session: requests.Session | None = None, timeout: float = 30):
        self.url = url.rstrip("/")
        self.s = session or requests.Session()
        self.timeout = timeout

    def _req(self, method: str, path: str, **kw: Any) -> requests.Response:
        kw.setdefault("timeout", self.timeout)
        try:
            return self.s.request(method, self.url + path, **kw)
        except requests.RequestException as e:
            raise BenchError(f"ComfyUI({self.url})に接続できません: {e}") from e

    def get_json(self, path: str) -> Any:
        r = self._req("GET", path)
        if r.status_code != 200:
            raise BenchError(f"ComfyUI の {path} が {r.status_code} を返しました")
        try:
            return r.json()
        except ValueError as e:
            raise BenchError(f"ComfyUI の {path} の応答が JSON ではありません") from e

    def system_stats(self) -> dict[str, Any]:
        return self.get_json("/system_stats")

    def object_info(self) -> dict[str, Any]:
        return self.get_json("/object_info")

    def queue(self) -> dict[str, Any]:
        return self.get_json("/queue")

    def free(self) -> None:
        self._req("POST", "/free", json={"unload_models": True, "free_memory": True})

    def interrupt(self) -> None:
        self._req("POST", "/interrupt", json={})

    def delete_from_queue(self, prompt_id: str) -> None:
        self._req("POST", "/queue", json={"delete": [prompt_id]})

    def submit(self, workflow: dict[str, Any], client_id: str) -> str:
        r = self._req("POST", "/prompt", json={"prompt": workflow, "client_id": client_id})
        try:
            body = r.json()
        except ValueError:
            body = {}
        if r.status_code != 200 or "prompt_id" not in body:
            raise BenchError("ComfyUI がワークフローを受け付けませんでした: " + _prompt_error_text(body, r))
        return str(body["prompt_id"])

    def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = self.get_json(f"/history/{prompt_id}")
        return data.get(prompt_id) if isinstance(data, dict) else None

    def template(self, name: str) -> dict[str, Any] | None:
        """サーバーが持つテンプレート(版がサーバーと合う)。無ければ None。"""
        r = self._req("GET", f"/templates/{name}.json")
        if r.status_code != 200:
            return None
        try:
            data = r.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) and "nodes" in data else None

    def upload_image(self, filename: str, content: bytes) -> str:
        r = self._req(
            "POST",
            "/upload/image",
            files={"image": (filename, content, "application/octet-stream")},
            data={"type": "input", "overwrite": "true"},
        )
        if r.status_code != 200:
            raise BenchError(f"入力の画像をアップロードできません({r.status_code}): {filename}")
        try:
            return str(r.json().get("name") or filename)
        except ValueError:
            return filename

    def log_entries(self) -> list[dict[str, Any]]:
        """ComfyUI のログ(新しめの版だけ。無ければ空)。"""
        try:
            r = self._req("GET", "/internal/logs/raw")
            if r.status_code == 200:
                entries = r.json().get("entries")
                return entries if isinstance(entries, list) else []
        except (BenchError, ValueError, AttributeError):
            pass
        return []


def _prompt_error_text(body: dict[str, Any], r: requests.Response) -> str:
    parts: list[str] = []
    err = body.get("error")
    if isinstance(err, dict):
        parts.append(str(err.get("message") or err.get("type") or err))
        if err.get("details"):
            parts.append(str(err["details"]))
    for nid, ne in (body.get("node_errors") or {}).items():
        for e in (ne or {}).get("errors") or []:
            parts.append(
                f"ノード {nid}({ne.get('class_type')}): {e.get('message')} {e.get('details') or ''}".strip()
            )
    return " / ".join(parts) if parts else f"HTTP {r.status_code}"


# --- 計測 -------------------------------------------------------------------------------------


@dataclass
class Peaks:
    torch_vram: int = 0  # bytes(/system_stats の torch_vram_total の最大)
    device_used: int = 0  # bytes(/system_stats の vram_total - vram_free の最大)
    gpu_used_mib: float = 0.0  # nvidia-smi の使用量の最大(MiB)
    ram: int = 0  # bytes(ComfyUI のプロセスの常駐メモリの最大)
    samples: int = 0
    smi_samples: int = 0


class Monitor:
    """生成中の最大値を裏のスレッドで集める。"""

    def __init__(
        self,
        client: ComfyClient,
        proc: psutil.Process | None,
        gpu_index: int | None,
        interval: float = 0.2,
        use_smi: bool = True,
    ):
        self.client = client
        self.proc = proc
        self.gpu_index = gpu_index
        self.interval = interval
        self.use_smi = use_smi
        self.peaks = Peaks()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._smi: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def sample_once(self) -> None:
        with contextlib.suppress(BenchError, ValueError, KeyError, TypeError):
            dev = (self.client.system_stats().get("devices") or [{}])[0]
            with self._lock:
                p = self.peaks
                p.torch_vram = max(p.torch_vram, int(dev.get("torch_vram_total") or 0))
                used = int(dev.get("vram_total") or 0) - int(dev.get("vram_free") or 0)
                p.device_used = max(p.device_used, used)
                p.samples += 1
        if self.proc is not None:
            with contextlib.suppress(psutil.Error):
                rss = self.proc.memory_info().rss
                for c in self.proc.children(recursive=True):
                    with contextlib.suppress(psutil.Error):
                        rss += c.memory_info().rss
                with self._lock:
                    self.peaks.ram = max(self.peaks.ram, rss)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)

    def _read_smi(self) -> None:
        exe = _nvidia_smi()
        if not exe:
            return
        cmd = [exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "200"]
        if self.gpu_index is not None:
            cmd += ["-i", str(self.gpu_index)]
        try:
            self._smi = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            return
        assert self._smi.stdout is not None
        for line in self._smi.stdout:
            with contextlib.suppress(ValueError):
                v = float(line.strip().split(",")[0])
                with self._lock:
                    self.peaks.gpu_used_mib = max(self.peaks.gpu_used_mib, v)
                    self.peaks.smi_samples += 1
            if self._stop.is_set():
                break

    def start(self) -> None:
        targets = [self._poll] + ([self._read_smi] if self.use_smi else [])
        for t in targets:
            th = threading.Thread(target=t, daemon=True)
            th.start()
            self._threads.append(th)

    def stop(self) -> Peaks:
        self._stop.set()
        if self._smi is not None:
            with contextlib.suppress(OSError):
                self._smi.terminate()
        for th in self._threads:
            th.join(timeout=5)
        self.sample_once()
        return self.peaks


def gpu_used_now(gpu_index: int | None) -> tuple[float, float] | None:
    """nvidia-smi の (使用量 MiB, 使用率 %)。無ければ None。"""
    exe = _nvidia_smi()
    if not exe:
        return None
    cmd = [exe, "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        cmd += ["-i", str(gpu_index)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True).stdout
        used, util = (float(x) for x in out.strip().splitlines()[0].split(",")[:2])
        return used, util
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def smi_gpus() -> list[dict[str, Any]]:
    """nvidia-smi の GPU の一覧(index, name, driver)。"""
    exe = _nvidia_smi()
    if not exe:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3 and parts[0].isdigit():
            gpus.append({"index": int(parts[0]), "name": parts[1], "driver": parts[2]})
    return gpus


def device_name(dev: dict[str, Any]) -> str:
    """'cuda:0 NVIDIA GeForce RTX 3070 : cudaMallocAsync' → 'NVIDIA GeForce RTX 3070'。"""
    name = str(dev.get("name") or "")
    name = re.sub(r"^\w+:\d+\s+", "", name)
    return name.split(" : ")[0].strip() or "unknown"


def find_server_process(url: str) -> psutil.Process | None:
    """URL のポートで待ち受けているプロセス(同じ PC のときだけ)。"""
    u = urlparse(url)
    if (u.hostname or "") not in LOCAL_HOSTS:
        return None
    port = u.port or (443 if u.scheme == "https" else 80)
    with contextlib.suppress(psutil.Error, OSError):
        for c in psutil.net_connections(kind="tcp"):
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port and c.pid:
                return psutil.Process(c.pid)
    return None


# ログからモデルの載せ方を読む(ComfyUI の版で文が変わるので、分からなければ None)
_LOG_PARTIAL = re.compile(r"loaded partially", re.I)
_LOG_FULL = re.compile(r"loaded completely", re.I)
_LOG_DYNAMIC = re.compile(r"dynamic(ally)?[ _-]?(vram|load)", re.I)
_LOG_LOWVRAM = re.compile(r"lowvram|novram", re.I)


def offload_from_logs(lines: list[str]) -> str | None:
    text = "\n".join(lines)
    if _LOG_PARTIAL.search(text):
        return "partial"
    if _LOG_DYNAMIC.search(text):
        return "dynamic"
    if _LOG_LOWVRAM.search(text):
        return "lowvram"
    if _LOG_FULL.search(text):
        return "none"
    return None


def _new_log_lines(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> list[str]:
    seen = {(e.get("t"), e.get("m")) for e in before}
    return [str(e.get("m") or "") for e in after if (e.get("t"), e.get("m")) not in seen]


def _history_error(entry: dict[str, Any]) -> tuple[str, bool] | None:
    """履歴が失敗なら (理由, メモリ不足か)。"""
    status = entry.get("status") or {}
    if status.get("status_str") != "error":
        return None
    for msg in status.get("messages") or []:
        if isinstance(msg, list) and len(msg) == 2 and msg[0] == "execution_error":
            d = msg[1] or {}
            text = (
                f"{d.get('node_type')}: {d.get('exception_type')}: {str(d.get('exception_message')).strip()}"
            )
            oom = "OutOfMemory" in str(d.get("exception_type")) or "out of memory" in text.lower()
            return text, oom
        if isinstance(msg, list) and len(msg) == 2 and msg[0] == "execution_interrupted":
            return "途中で中断されました(ComfyUI の画面か /interrupt で止められた)", False
    return "実行に失敗しました(理由は ComfyUI のログを見てください)", False


def _exec_seconds(entry: dict[str, Any]) -> float | None:
    """履歴のメッセージの時刻(ミリ秒)から、実行にかかった秒数。"""
    ts = {}
    for msg in (entry.get("status") or {}).get("messages") or []:
        if isinstance(msg, list) and len(msg) == 2 and isinstance(msg[1], dict) and "timestamp" in msg[1]:
            ts[msg[0]] = msg[1]["timestamp"]
    start = ts.get("execution_start")
    end = ts.get("execution_success") or ts.get("execution_error") or ts.get("execution_interrupted")
    if start and end:
        return round((end - start) / 1000, 2)
    return None


def _cached_nodes(entry: dict[str, Any]) -> set[str]:
    """前の実行の結果をそのまま使った(実行しなかった)ノード。"""
    out: set[str] = set()
    for msg in (entry.get("status") or {}).get("messages") or []:
        if (
            isinstance(msg, list)
            and len(msg) == 2
            and msg[0] == "execution_cached"
            and isinstance(msg[1], dict)
        ):
            out.update(str(n) for n in msg[1].get("nodes") or [])
    return out


# --- 本体 -------------------------------------------------------------------------------------


@dataclass
class BenchOptions:
    model_id: str
    url: str = "http://127.0.0.1:8188"
    workflow_path: str | None = None
    width: int | None = None
    height: int | None = None
    frames: int | None = None
    seconds: float | None = None
    steps: int | None = None
    sets: dict[str, Any] = field(default_factory=dict)  # 入力の名前 → 値(すべてのノード)
    free_before: bool = True
    timeout: float = 1800
    interval: float = 0.2
    force: bool = False
    source: str | None = None
    note: str | None = None
    precision: str | None = None
    busy_util_pct: float = 30.0


@dataclass
class BenchResult:
    measurement: Measurement
    workflow_source: str
    warnings: list[str]
    failed: str | None = None  # 失敗したときの理由(メモリ不足は測った値を残す)

    def to_dict(self) -> dict[str, Any]:
        m = asdict(self.measurement)
        return {"measurement": m, "workflow_source": self.workflow_source, "warnings": self.warnings} | (
            {"failed": self.failed} if self.failed else {}
        )


def read_workflow_file(path: str) -> dict[str, Any]:
    p = Path(path)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BenchError(f"ワークフローを読めません: {p.name}: {e}") from e
    if not isinstance(data, dict):
        raise BenchError(f"ワークフローの形が違います: {p.name}")
    return data


def load_workflow(
    client: ComfyClient,
    opts: BenchOptions,
    object_info: dict[str, Any],
    session: requests.Session,
    data: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """(API 形式のワークフロー, どこから取ったか, 注意) を返す。"""
    if opts.workflow_path:
        if data is None:
            data = read_workflow_file(opts.workflow_path)
        if is_api_format(data):
            return data, "file:api", []
        conv = ui_to_api(data, object_info)
        return conv.workflow, "file:ui", conv.warnings
    ui = client.template(opts.model_id)
    source = "server-template"
    if ui is None:
        try:
            r = session.get(template_url(opts.model_id), timeout=30)
        except requests.RequestException as e:
            raise BenchError(f"テンプレートを取れません: {e}") from e
        if r.status_code != 200:
            raise BenchError(
                f"テンプレート {opts.model_id} が見つかりません(ComfyUI にも GitHub にも無い)。"
                "名前を確かめるか、--workflow で API 形式のワークフローを渡してください"
            )
        try:
            ui = r.json()
        except ValueError as e:
            raise BenchError("テンプレートが JSON ではありません") from e
        source = "github-template"
    conv = ui_to_api(ui, object_info)
    return conv.workflow, source, conv.warnings


def _ensure_inputs(
    client: ComfyClient, wf: dict[str, Any], object_info: dict[str, Any], session: requests.Session
) -> list[str]:
    """テンプレートの入力画像がサーバーに無ければ、公式のものを取ってアップロードする。"""
    notes: list[str] = []
    for nid, name, filename in upload_inputs(wf, object_info):
        try:
            r = session.get(template_input_url(filename), timeout=60)
        except requests.RequestException as e:
            raise BenchError(f"入力の画像を取れません: {filename}: {e}") from e
        if r.status_code != 200:
            raise BenchError(
                f"入力の画像 {filename} がサーバーにも公式テンプレートにもありません。"
                "ComfyUI の input フォルダに置くか、--workflow で別の画像を指定してください"
            )
        wf[nid]["inputs"][name] = client.upload_image(filename, r.content)
        notes.append(f"入力の画像 {filename} をアップロードした")
    return notes


def _parse_set(items: list[str] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise BenchError(f"--set は 名前=値 の形で書いてください: {item}")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except ValueError:
            out[k.strip()] = v
    return out


def run(opts: BenchOptions, *, client: ComfyClient | None = None, log=print) -> BenchResult:
    """1 回測る。測れないときは BenchError。"""
    session = requests.Session()
    file_data = (
        read_workflow_file(opts.workflow_path) if opts.workflow_path else None
    )  # サーバーに触る前に読む
    client = client or ComfyClient(opts.url, session=session)
    stats = client.system_stats()
    devices = stats.get("devices") or []
    if not devices or devices[0].get("type") not in ("cuda", "rocm", "xpu", "mps"):
        raise BenchError("ComfyUI が GPU を使っていません(CPU だけでは VRAM を測れません)")
    dev0 = devices[0]
    q = client.queue()
    if q.get("queue_running") or q.get("queue_pending"):
        raise BenchError(
            "ComfyUI のキューに他の仕事があります。終わってから測ってください(他の人の生成を邪魔しないため)"
        )

    object_info = client.object_info()
    wf, source, warnings = load_workflow(client, opts, object_info, session, file_data)
    wf, notes = apply_settings(
        wf, width=opts.width, height=opts.height, frames=opts.frames, seconds=opts.seconds, steps=opts.steps
    )
    warnings += notes
    for key, value in opts.sets.items():
        hit = [n for n in wf.values() if key in n["inputs"] and not isinstance(n["inputs"][key], list)]
        if not hit:
            warnings.append(f"--set {key}: その名前の入力がありません")
        for n in hit:
            n["inputs"][key] = value
    set_output_prefix(wf, f"genmodelfit_bench/{re.sub(r'[^A-Za-z0-9_.-]', '_', opts.model_id)}")
    warnings += _ensure_inputs(client, wf, object_info, session)
    problems = check_inputs(wf, object_info)
    if problems:
        raise BenchError(
            "\n".join(["必要なファイルがサーバーにありません。先にモデルを置いてください:", *problems])
        )
    settings = effective_settings(wf, precision=opts.precision)

    # どの GPU か(nvidia-smi の番号。名前で合わせる)
    name = device_name(dev0)
    smi = smi_gpus()
    match = next((g for g in smi if g["name"] == name), None)
    gpu_index = match["index"] if match else None
    driver = match["driver"] if match else None
    local = find_server_process(opts.url)

    preloaded = int(dev0.get("torch_vram_total") or 0)
    if opts.free_before:
        client.free()
        time.sleep(2.0)
    base_stats = (client.system_stats().get("devices") or [dev0])[0]
    base_torch = int(base_stats.get("torch_vram_total") or 0)
    base_dev = int(base_stats.get("vram_total") or 0) - int(base_stats.get("vram_free") or 0)
    base_smi = gpu_used_now(gpu_index) if match else None
    if base_smi and base_smi[1] >= opts.busy_util_pct and not opts.force:
        raise BenchError(
            f"GPU が他の仕事で使われています(使用率 {base_smi[1]:.0f}%)。"
            "空いてから測るか、--force を付けてください"
        )

    log_before = client.log_entries()
    mon = Monitor(client, local, gpu_index, interval=opts.interval, use_smi=match is not None)
    client_id = uuid.uuid4().hex
    log(f"{opts.model_id} を {opts.url} で実行します …")
    mon.start()
    t0 = time.monotonic()
    failed: str | None = None
    oom = False
    entry: dict[str, Any] | None = None
    try:
        prompt_id = client.submit(wf, client_id)
        while True:
            entry = client.history(prompt_id)
            if entry and ((entry.get("status") or {}).get("completed") or _history_error(entry)):
                break
            if time.monotonic() - t0 > opts.timeout:
                with contextlib.suppress(BenchError):
                    running = [x[1] for x in client.queue().get("queue_running") or [] if len(x) > 1]
                    if prompt_id in running:
                        client.interrupt()
                    else:
                        client.delete_from_queue(prompt_id)
                raise BenchError(f"{opts.timeout:.0f} 秒で終わらなかったので中止しました")
            time.sleep(0.5)
    finally:
        peaks = mon.stop()
    wall = time.monotonic() - t0
    err = _history_error(entry) if entry else None
    if err:
        failed, oom = err
        if not oom:
            raise BenchError(f"生成に失敗しました: {failed}")

    cached = _cached_nodes(entry or {})
    if cached and cached >= set(wf):
        raise BenchError(
            "前と同じ内容だったので ComfyUI が前の結果を返し、何も実行されませんでした。"
            "--no-free を外すか、シードなどの入力を変えてください(--set seed=…)"
        )
    log_lines = _new_log_lines(log_before, client.log_entries())
    offload = "oom" if oom else offload_from_logs(log_lines)
    exec_s = _exec_seconds(entry or {})

    vram_total = int(dev0.get("vram_total") or 0)
    torch_peak = peaks.torch_vram if peaks.samples else 0
    dev_delta = max(0, peaks.device_used - base_dev)
    if torch_peak > 0:
        peak_vram = torch_peak / GIB
    else:  # torch の量が出ない版では、GPU 全体の増分で代える
        peak_vram = dev_delta / GIB
        warnings.append("torch_vram_total が取れないので、GPU 全体の増分を最大 VRAM にした")
    note_parts = []
    if base_smi and peaks.smi_samples:
        gpu_delta = max(0.0, peaks.gpu_used_mib - base_smi[0]) / 1024
        note_parts.append(
            f"GPU 全体(nvidia-smi)の増分 {gpu_delta:.2f} GB(開始前 {base_smi[0] / 1024:.2f} GB 使用、"
            f"最大 {peaks.gpu_used_mib / 1024:.2f} GB)"
        )
    note_parts.append(f"/system_stats の GPU 全体の増分 {dev_delta / GIB:.2f} GB")
    if opts.free_before:
        note_parts.append(
            "開始前に /free でモデルを解放"
            + (
                f"(解放前 torch {preloaded / GIB:.2f} GB、解放後 {base_torch / GIB:.2f} GB)"
                if preloaded
                else ""
            )
        )
    else:
        note_parts.append(f"モデルは読み込み済みのまま(開始前 torch {base_torch / GIB:.2f} GB)")
    if exec_s is not None:
        note_parts.append(f"ComfyUI の実行時間 {exec_s:g} 秒")
    if cached:
        note_parts.append(f"前の結果を使ったノード {len(cached)} 個")
    note_parts.append(f"計測 {peaks.samples} 回")
    if failed:
        note_parts.append(f"メモリ不足で失敗: {failed[:300]}")
    if opts.sets:
        note_parts.append("変えた入力: " + ", ".join(f"{k}={v}" for k, v in opts.sets.items()))
    if opts.note:
        note_parts.append(opts.note)

    m = Measurement(
        model_id=opts.model_id,
        gpu_name=name,
        gpu_vram_gb=round(vram_total / GIB, 1),
        peak_vram_gb=round(peak_vram, 2),
        settings=settings,
        peak_ram_gb=round(peaks.ram / GIB, 2) if local is not None and peaks.ram else None,
        seconds=round(wall, 1),
        driver=driver,
        comfyui_version=str((stats.get("system") or {}).get("comfyui_version") or "") or None,
        offload=offload,
        source=opts.source or f"genmodelfit bench {__version__}",
        date=datetime.now(timezone.utc).date().isoformat(),
        note="。".join(note_parts),
    )
    return BenchResult(measurement=m, workflow_source=source, warnings=warnings, failed=failed)


def default_output_path(m: Measurement) -> Path:
    def slug(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "-", s).strip("-")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    gpu = slug(m.gpu_name.replace("NVIDIA ", "").replace("GeForce ", ""))
    return cache_dir() / "measurements" / f"{slug(m.model_id)}_{gpu}_{ts}.json"


def save(m: Measurement, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"measurements": [asdict(m)]}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def _options_from_args(args: argparse.Namespace) -> BenchOptions:
    width, height = getattr(args, "width", None), getattr(args, "height", None)
    res = getattr(args, "resolution", None)
    if res:
        mres = re.fullmatch(r"\s*(\d+)\s*[xX*×]\s*(\d+)\s*", res)
        if not mres:
            raise BenchError(f"--resolution は 幅x高さ の形で書いてください(例 768x768): {res}")
        width, height = int(mres.group(1)), int(mres.group(2))
    precision = None
    with contextlib.suppress(Exception):
        from .db import load as load_db

        entry = load_db(getattr(args, "db", None), use_cache=not getattr(args, "no_cache", False)).get(
            args.template
        )
        precision = entry.precision if entry else None
    return BenchOptions(
        model_id=args.template,
        url=args.url,
        workflow_path=args.workflow,
        width=width,
        height=height,
        frames=args.frames,
        seconds=args.length,
        steps=args.steps,
        sets=_parse_set(getattr(args, "set", None)),
        free_before=not getattr(args, "no_free", False),
        timeout=getattr(args, "timeout", None) or 1800,
        force=getattr(args, "force", False),
        source=getattr(args, "source", None),
        note=getattr(args, "note", None),
        precision=precision,
    )


def run_bench(args: argparse.Namespace) -> int:
    """cli.py の `bench` から呼ばれる。"""
    from .cli import CliError, _envelope, _out

    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")  # 進み具合と注意は stderr に出す(stdout と同じく UTF-8)

    def log(msg: str) -> None:
        if not args.json:
            print(msg, file=sys.stderr)

    try:
        opts = _options_from_args(args)
        result = run(opts, log=log)
    except (BenchError, WorkflowError) as e:
        raise CliError(str(e)) from e
    m = result.measurement
    out_path = Path(args.output) if args.output else default_output_path(m)
    save(m, out_path)
    if args.json:
        _out(_envelope("bench", None, None, saved_to=str(out_path), **result.to_dict()))
        return 1 if result.failed else 0
    for w in result.warnings:
        print(f"注意: {w}", file=sys.stderr)
    s = m.settings
    parts = [f"{s.width}x{s.height}" if s.width else None, f"{s.frames} フレーム" if s.frames else None]
    parts += [f"{s.seconds:g} 秒の音声" if s.seconds else None, f"{s.steps} ステップ" if s.steps else None]
    print(f"=== bench: {m.model_id} ===")
    print(f"GPU:      {m.gpu_name}({m.gpu_vram_gb:g} GB、ドライバ {m.driver or '不明'})")
    print(f"ComfyUI:  {m.comfyui_version or '不明'}")
    print(f"設定:     {'、'.join(p for p in parts if p) or '(テンプレートのまま)'}")
    print(f"最大 VRAM: {m.peak_vram_gb:.2f} GB(ComfyUI のプロセス)")
    if m.peak_ram_gb is not None:
        print(f"最大メモリ: {m.peak_ram_gb:.2f} GB")
    print(f"時間:     {m.seconds:g} 秒(送ってから終わるまで、モデルの読み込み込み)")
    print(f"載せ方:   {m.offload or '不明'}")
    print(f"メモ:     {m.note}")
    if result.failed:
        print(f"失敗:     {result.failed}")
    print(f"保存先:   {out_path}")
    return 1 if result.failed else 0
