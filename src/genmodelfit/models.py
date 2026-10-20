# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""データの形(スキーマ)。JSON との相互変換もここで行う。docs/schema.md と必ず一致させる。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

SCHEMA_VERSION = 1

# 生成の種類。music=音楽、sfx=効果音、tts=読み上げ
MODALITIES = ("image", "video", "music", "sfx", "tts", "3d")
# 内容の印。uncensored=無検閲(安全フィルタを外した、または成人向けの出力を想定したモデル)
CONTENT_LEVELS = ("sfw", "uncensored")
# 適合の段階(良い順)
FIT_LEVELS = ("perfect", "good", "marginal", "too_tight")
# 推定の根拠
BASIS = ("measured", "estimated")
# 実測の結果。ok=最後まで動いた、oom=メモリ不足で失敗、error=ほかの理由で失敗
OUTCOMES = ("ok", "oom", "error")


def _from_dict(cls, data: dict[str, Any]):
    """未知のキーは捨てる(新しい版のデータを古い版でも読めるように)。"""
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


@dataclass
class ModelFile:
    """モデルを構成する 1 ファイル(本体・テキストエンコーダ・VAE など)。"""

    name: str
    url: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    directory: str | None = None  # ComfyUI の置き場所(diffusion_models / text_encoders / vae など)
    repo_id: str | None = None  # Hugging Face のリポジトリ(URL から求めたもの)
    license: str | None = None  # このファイルのライセンス(SPDX 風。不明は None、独自は "other")
    license_name: str | None = None  # license が other のときの名前(例: flux-1-dev-non-commercial-license)
    gated: bool | None = None  # 利用申請が要るか(None=不明)
    role: str | None = None  # main / text_encoder / vae / clip_vision / lora / other
    license_from: str | None = None  # license を取った HF リポジトリ(再配布なら本家)
    license_url: str | None = None  # license の原文・記載の場所
    repo_license: str | None = None  # 置き場所のリポジトリの表記が本家と違うとき、その表記
    # ライセンスを決めた道(置き場所 → base_model … → 決め手。部品の表・系統の表で決めたときは表の本家)
    license_path: list[str] = field(default_factory=list)
    # 決め方の根拠。"license_unverified: …"(確かめられず不明にした)、"component:<表の名前>"、"family:<系統>"
    license_evidence: list[str] = field(default_factory=list)
    head_status: int | None = None  # 匿名の HEAD(転送を追わない)の応答コード(None=未確認)
    head_checked_at: str | None = None  # その HEAD を送った日時(UTC, ISO 8601)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelFile:
        return _from_dict(cls, d)


@dataclass
class Template:
    """ComfyUI 公式テンプレートの情報(source が comfyui-template のとき)。"""

    name: str
    title: str | None = None
    category: str | None = None
    min_comfyui_version: str | None = None
    open_source: bool | None = None
    requires_custom_nodes: list[str] = field(default_factory=list)
    usage: int | None = None
    date: str | None = None
    size_bytes: int | None = None  # index.json の size
    username: str | None = None  # index.json の username(作った人。ComfyUI 以外は利用者のテンプレート)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Template:
        return _from_dict(cls, d)


@dataclass
class ModelEntry:
    """一覧の 1 件。1 つのワークフロー(または 1 つの HF リポジトリ)で使うモデル一式。"""

    id: str
    name: str
    modality: str
    tasks: list[str] = field(default_factory=list)  # t2i / i2i / t2v / i2v / t2a / tts / t23d / i23d など
    source: str = "comfyui-template"  # comfyui-template / huggingface / manual
    files: list[ModelFile] = field(default_factory=list)
    template: Template | None = None
    params_b: float | None = None  # 本体のパラメータ数(十億)
    precision: str | None = None  # fp32 / bf16 / fp16 / fp8 / int8 / nf4 / gguf-q4 など
    license: str | None = None  # モデル一式のライセンス(最も厳しいもの)
    license_url: str | None = None
    base_model: str | None = None
    content: str = "sfw"
    content_evidence: list[str] = field(default_factory=list)  # 無検閲と判断した根拠
    discovered_from: list[str] = field(default_factory=list)  # どこから拾ったか(URL)
    updated: str | None = None
    homepage: str | None = None
    downloads: int | None = None  # HF の直近 30 日のダウンロード数(source が huggingface のとき)
    likes: int | None = None
    downloads_all_time: int | None = None  # HF の全期間のダウンロード数(一覧に入れるかの下限に使う)
    license_name: str | None = None  # license が other のときの名前(一番厳しいファイルのもの)
    # 不明にした理由など(ファイルの license_evidence のうち license_unverified をまとめたもの)
    license_evidence: list[str] = field(default_factory=list)
    # ファイルの一覧で一式がそろっているか。False: 本体が無い・合計が index の size の半分未満・
    # GGUF の本体だけ(テキストエンコーダ・VAE が無い)。None は判定していない(古いデータ)。docs/schema.md
    files_complete: bool | None = None
    # 一式の大きさの下限(files_complete が False のとき、index の size)。見積もりはこれより小さく見ない
    size_floor_bytes: int | None = None
    # ComfyUI のモデルの系統と VAE の種類(families.FAMILIES / VAES の鍵。None は分からない)。
    # bf16 を持たない GPU で fp32 で計算するかを決めるのに使う(docs/hardware.md「精度」)。後から足した項目
    comfy_family: str | None = None
    comfy_vae: str | None = None

    @property
    def total_size_bytes(self) -> int | None:
        sizes = [f.size_bytes for f in self.files if f.size_bytes]
        if sizes:
            return sum(sizes)
        return self.template.size_bytes if self.template else None

    @property
    def estimate_size_bytes(self) -> int | None:
        """見積もりに使う大きさ。ファイルの合計が size_floor_bytes より小さければ下限を使う
        (一式が欠けた一覧を小さく見せない)。"""
        total = self.total_size_bytes
        if self.size_floor_bytes and (total or 0) < self.size_floor_bytes:
            return self.size_floor_bytes
        return total

    @property
    def largest_file_bytes(self) -> int | None:
        sizes = [f.size_bytes for f in self.files if f.size_bytes]
        return max(sizes) if sizes else None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["total_size_bytes"] = self.total_size_bytes
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelEntry:
        e = _from_dict(cls, d)
        e.files = [ModelFile.from_dict(f) if isinstance(f, dict) else f for f in (d.get("files") or [])]
        t = d.get("template")
        e.template = Template.from_dict(t) if isinstance(t, dict) else t
        return e


@dataclass
class Settings:
    """生成の設定。種類ごとに使う項目が違う(画像は width/height、動画は frames も、音声は seconds)。"""

    width: int | None = None
    height: int | None = None
    frames: int | None = None
    seconds: float | None = None
    steps: int | None = None
    precision: str | None = None
    batch: int = 1

    @property
    def megapixels(self) -> float | None:
        if self.width and self.height:
            return self.width * self.height / 1_000_000
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Settings:
        return _from_dict(cls, d)


@dataclass
class Measurement:
    """実機での実測 1 件。"""

    model_id: str
    gpu_name: str
    gpu_vram_gb: float
    peak_vram_gb: float
    settings: Settings = field(default_factory=Settings)
    peak_ram_gb: float | None = None
    seconds: float | None = None
    driver: str | None = None
    comfyui_version: str | None = None
    offload: str | None = None  # none / dynamic / partial / lowvram / oom(ComfyUI の動き)
    source: str = ""  # 誰がどこで測ったか
    date: str | None = None
    note: str | None = None
    # 以下は後から足した項目(無い古い行は ok・標準として読む)
    outcome: str = "ok"  # OUTCOMES。ok 以外は「動く」根拠に使わない
    standard: bool = True  # 公式テンプレートのまま(幅・高さ・フレーム数・長さ・ステップ数だけ変えた)か
    # peak_vram_gb の測り方。torch_reserved = ComfyUI のプロセスの torch の確保量の最大
    # (bench。CUDA の文脈は含まない)。
    # None = 記録が無い(AI生成キットで測った RTX 5080 の 9 件)。測り方が違う値どうしで係数を合わせない
    vram_metric: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Measurement:
        m = _from_dict(cls, d)
        s = d.get("settings")
        m.settings = Settings.from_dict(s) if isinstance(s, dict) else (s or Settings())
        return m


@dataclass
class Gpu:
    """nvidia-smi で確かめた GPU。判定に使うのは backend が cuda のものだけ(docs/hardware.md)。"""

    name: str
    vendor: str  # nvidia / amd / intel / apple / unknown
    vram_total_gb: float
    vram_free_gb: float | None = None  # 今の空き(nvidia-smi の memory.free)。上書きしたときは None
    driver: str | None = None
    backend: str = (
        "none"  # cuda / none(v0.1 は NVIDIA の CUDA だけを数える。動かない世代・古いドライバは none)
    )
    vram_used_gb: float | None = None  # 今ほかのアプリ(画面・ブラウザなど)が使っている量(memory.used)
    compute_capability: str | None = None  # 例 "8.6"。None は分からない
    compute_capability_source: str | None = None  # nvidia-smi / name(GPU 名の表から推した)/ None
    arch: str | None = None  # kepler / maxwell / pascal / volta / turing / ampere / ada / hopper / blackwell
    # 今の PyTorch(docs/hardware.md の版)の CUDA 版のうち、この GPU とドライバで動くもの(新しい順)。参考の情報
    torch_cuda: list[str] = field(default_factory=list)
    # ComfyUI がこの GPU で選ぶ計算の精度(fp16 / bf16 / fp32)と、bf16・fp8 の計算に対応するか。
    # None は分からない
    compute_dtype: str | None = None
    bf16: bool | None = None
    fp8: bool | None = None
    notes: list[str] = field(default_factory=list)  # backend が none の理由・注意


@dataclass
class Adapter:
    """NVIDIA 以外も含めて見つけた画面の部品(GPU)。知らせるためだけに使い、判定には使わない。"""

    name: str
    vendor: str  # nvidia / amd / intel / unknown
    vram_gb: float | None = None  # 表示用(Windows はレジストリの qwMemorySize、Linux の AMD は sysfs)
    vram_source: str | None = None  # registry / wmi / sysfs / None
    source: str = ""  # wmi / lspci


@dataclass
class SystemInfo:
    os: str
    cpu: str
    cpu_cores: int
    ram_total_gb: float
    ram_available_gb: float
    gpus: list[Gpu] = field(default_factory=list)
    disk_free_gb: float | None = None
    overrides: list[str] = field(default_factory=list)  # --vram などで上書きした項目
    # 仮想メモリ。swap_total_gb は今のページファイル(Windows)・スワップ(Linux)の大きさ。
    # swap_max_gb は伸びる上限(固定の大きさのページファイル・Linux のスワップ)。None は分からない・
    # システム管理で伸びる(swap_auto=True)。判定の「少し足りなければ仮想メモリで遅い」はこれで決める
    swap_total_gb: float | None = None
    swap_max_gb: float | None = None
    swap_auto: bool | None = None
    # Windows のコミット(物理メモリ + ページファイル)の今の上限と、今使っている量
    commit_limit_gb: float | None = None
    commit_used_gb: float | None = None
    other_gpus: list[Adapter] = field(default_factory=list)  # NVIDIA 以外も含めた画面の部品(知らせるだけ)
    other_gpus_checked: bool = False  # other_gpus を調べたか(NVIDIA の GPU があれば既定では調べない)
    notes: list[str] = field(default_factory=list)  # 検出で分かったこと・分からなかった理由

    @property
    def best_gpu(self) -> Gpu | None:
        usable = [g for g in self.gpus if g.backend == "cuda"]
        return max(usable, key=lambda g: g.vram_total_gb) if usable else None

    @property
    def vram_gb(self) -> float:
        g = self.best_gpu
        return g.vram_total_gb if g else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["best_gpu"] = asdict(self.best_gpu) if self.best_gpu else None
        return d


@dataclass
class Estimate:
    """1 モデル × 1 設定 × 1 PC の見積もりと判定。"""

    vram_gb: float  # 全部を GPU に載せたときの最大 VRAM
    vram_offload_gb: float  # 重みの一部(offload_fraction 以外)をメインメモリに逃がしたときの最大 VRAM
    ram_gb: float  # 判定した動かし方で要るメインメモリ(GiB)
    disk_gb: float | None
    basis: str  # measured / estimated
    fit: str  # FIT_LEVELS
    run_mode: str  # gpu / offload / cpu / none
    reasons: list[str] = field(default_factory=list)
    measured_on: str | None = None  # basis が measured のとき、どの GPU で測ったか
    vram_sequential_gb: float | None = None  # 部品を段ごとに載せ替えたときの最大 VRAM
    settings: dict[str, Any] | None = None  # 判定に使った設定(省いた項目は既定や実測の設定で埋めたもの)
    limits: dict[str, Any] | None = None  # この PC で good 以上になる最大の設定(info で出す)
    unknown_size: bool = False  # 重みの大きさが分からず判定できない(fit は too_tight、VRAM の値は 0)
    # GPU で動く判定だが、今の空き VRAM(nvidia-smi の memory.free)では足りない(ほかのアプリを閉じれば動く)。
    # 段階は総量で決めるので、これは知らせるだけ
    vram_free_short: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
