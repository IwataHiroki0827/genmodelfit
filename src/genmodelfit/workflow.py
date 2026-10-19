"""ComfyUI のワークフローを扱う(bench 用)。

- 公式テンプレート(画面用の形式)を、サーバーの `/object_info` を見て API 形式に直す
- 設定(幅・高さ・フレーム数・音声の長さ・ステップ数)を入力の名前で書き換える
- 実際に使う設定を API 形式のワークフローから読み取る

ComfyUI のコードは import しない(GPL のため)。ここでは JSON だけを扱う。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any

from .models import Settings

TEMPLATES_RAW = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main"

# 画面だけのノード(サーバーには無い)。Note は捨て、PrimitiveNode / Reroute は値・つながりに置き換える
UI_ONLY_SKIP = {"Note", "MarkdownNote", "Label (rgthree)"}
PRIMITIVE = "PrimitiveNode"
REROUTE = "Reroute"
# seed の後ろに画面だけの「生成後の動き」の値が入る
CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
DYNAMIC_COMBO = "COMFY_DYNAMICCOMBO_V3"
# ノードの mode: 0=通常、2=無効(mute)、4=素通し(bypass)
MODE_NEVER = 2
MODE_BYPASS = 4
SUBGRAPH_IN = -10
SUBGRAPH_OUT = -20


class WorkflowError(ValueError):
    """ワークフローを変換・実行できない理由(利用者に見せる文)。"""


def template_url(name: str) -> str:
    return f"{TEMPLATES_RAW}/templates/{name}.json"


def template_input_url(filename: str) -> str:
    return f"{TEMPLATES_RAW}/input/{filename}"


def is_api_format(data: Any) -> bool:
    """API 形式({"<id>": {"class_type": ..., "inputs": {...}}})か。"""
    return (
        isinstance(data, dict)
        and bool(data)
        and "nodes" not in data
        and all(isinstance(v, dict) and "class_type" in v for v in data.values())
    )


# --- 画面用の形式 → API 形式 ----------------------------------------------------------------------


@dataclass
class _Input:
    name: str
    type: str | None
    widget: bool
    src: tuple[str, int] | None  # つながっている元(ノード ID, 出力の番号)


@dataclass
class _Node:
    id: str
    type: str
    mode: int
    widgets_values: Any
    inputs: list[_Input]
    output_types: list[str | None]
    title: str | None = None
    widget_overrides: dict[str, Any] = field(default_factory=dict)  # サブグラフの外から渡された値


def _links_map(links: Any) -> dict[int, tuple[int, int, int, int]]:
    """リンクを {id: (元, 元の出力, 先, 先の入力)} にする。トップは配列、サブグラフは辞書の形。"""
    out: dict[int, tuple[int, int, int, int]] = {}
    for link in links or []:
        if isinstance(link, dict):
            out[link["id"]] = (link["origin_id"], link["origin_slot"], link["target_id"], link["target_slot"])
        elif isinstance(link, list) and len(link) >= 5:
            out[link[0]] = (link[1], link[2], link[3], link[4])
    return out


def _flatten(
    nodes: list[dict[str, Any]],
    links: dict[int, tuple[int, int, int, int]],
    defs: dict[str, dict[str, Any]],
    prefix: str,
    out: list[_Node],
    alias: dict[tuple[str, int], tuple[str, int] | None],
    depth: int = 0,
) -> None:
    """サブグラフを展開して平らなノードの並びにする。中のノードの ID は「外の ID:中の ID」。"""
    if depth > 16:
        raise WorkflowError("サブグラフの入れ子が深すぎます")

    def src_of(link_id: int | None) -> tuple[str, int] | None:
        if link_id is None or link_id not in links:
            return None
        origin, slot, _, _ = links[link_id]
        return (f"{prefix}{origin}", slot)

    for n in nodes:
        nid = f"{prefix}{n['id']}"
        ntype = str(n.get("type"))
        mode = int(n.get("mode") or 0)
        if ntype in defs and mode not in (MODE_NEVER, MODE_BYPASS):
            sg = defs[ntype]
            inner = f"{nid}:"
            sub_links = _links_map(sg.get("links"))
            outer_inputs = {i.get("name"): i for i in n.get("inputs") or []}
            for k, sin in enumerate(sg.get("inputs") or []):
                oi = outer_inputs.get(sin.get("name"))
                alias[(f"{inner}{SUBGRAPH_IN}", k)] = src_of(oi.get("link")) if oi else None
            for k, sout in enumerate(sg.get("outputs") or []):
                for lid in sout.get("linkIds") or []:
                    if lid in sub_links:
                        origin, slot, _, _ = sub_links[lid]
                        alias[(nid, k)] = (f"{inner}{origin}", slot)
            child_start = len(out)
            _flatten(sg.get("nodes") or [], sub_links, defs, inner, out, alias, depth + 1)
            _apply_proxy_widgets(n, sg, sub_links, inner, out[child_start:])
            continue
        if ntype in defs:
            raise WorkflowError(f"無効・素通しのサブグラフには対応していません(ノード {nid})")
        inputs = [
            _Input(
                name=str(i.get("name")),
                type=i.get("type"),
                widget=bool(i.get("widget")),
                src=src_of(i.get("link")),
            )
            for i in n.get("inputs") or []
        ]
        outputs = [o.get("type") for o in n.get("outputs") or []]
        out.append(
            _Node(
                id=nid,
                type=ntype,
                mode=mode,
                widgets_values=n.get("widgets_values"),
                inputs=inputs,
                output_types=outputs,
                title=n.get("title"),
            )
        )


def _apply_proxy_widgets(
    outer: dict[str, Any],
    sg: dict[str, Any],
    sub_links: dict[int, tuple[int, int, int, int]],
    inner: str,
    children: list[_Node],
) -> None:
    """サブグラフの外に出した入力欄(proxyWidgets)の値を、中のノードへ戻す。"""
    proxies = (outer.get("properties") or {}).get("proxyWidgets") or []
    values = outer.get("widgets_values")
    if not proxies or not isinstance(values, list) or not values:
        return
    by_id = {c.id: c for c in children}
    for proxy, value in zip(proxies, values, strict=False):
        if not (isinstance(proxy, list) and len(proxy) == 2):
            continue
        node_id, widget = str(proxy[0]), str(proxy[1])
        if widget == "control_after_generate":
            continue
        if node_id == "-1":  # サブグラフの入力そのもの → つながる先の入力欄すべて
            names = [i.get("name") for i in sg.get("inputs") or []]
            if widget not in names:
                continue
            k = names.index(widget)
            for origin, slot, target, tslot in sub_links.values():
                if origin == SUBGRAPH_IN and slot == k and f"{inner}{target}" in by_id:
                    t = by_id[f"{inner}{target}"]
                    if tslot < len(t.inputs):
                        t.widget_overrides[t.inputs[tslot].name] = value
        elif f"{inner}{node_id}" in by_id:
            by_id[f"{inner}{node_id}"].widget_overrides[widget] = value


def _input_specs(info: dict[str, Any]) -> list[tuple[str, Any, dict[str, Any]]]:
    """/object_info の 1 ノード分から、入力を (名前, 型, 設定) の順の並びにする。"""
    sections = info.get("input") or {}
    order = info.get("input_order") or {}
    out: list[tuple[str, Any, dict[str, Any]]] = []
    for sec in ("required", "optional"):
        specs = sections.get(sec) or {}
        names = order.get(sec) or list(specs)
        for name in names:
            spec = specs.get(name)
            if not spec:
                continue
            typ = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            out.append((name, typ, opts))
    return out


def _is_widget(typ: Any, opts: dict[str, Any]) -> bool:
    if opts.get("forceInput"):
        return False
    return isinstance(typ, list) or typ in WIDGET_TYPES or typ == DYNAMIC_COMBO


def combo_options(typ: Any, opts: dict[str, Any]) -> list[Any] | None:
    if isinstance(typ, list):
        return typ
    if typ == "COMBO" and isinstance(opts.get("options"), list):
        return opts["options"]
    return None


def map_widgets(info: dict[str, Any], values: Any) -> dict[str, Any]:
    """widgets_values(画面の入力欄の値の並び)を入力の名前に割り当てる。

    seed / noise_seed の後ろの「生成後の動き」(fixed / randomize など)は飛ばす。
    """
    specs = _input_specs(info)
    if isinstance(values, dict):  # 一部のノードは名前つきで持つ
        names = {s[0] for s in specs}
        return {k: v for k, v in values.items() if k in names}
    if not isinstance(values, list):
        return {}
    out: dict[str, Any] = {}
    idx = 0

    def take(specs_: list[tuple[str, Any, dict[str, Any]]], prefix: str) -> None:
        nonlocal idx
        for name, typ, opts in specs_:
            if not _is_widget(typ, opts):
                continue
            if idx >= len(values):
                return
            val = values[idx]
            idx += 1
            out[prefix + name] = val
            if (
                (opts.get("control_after_generate") or name in ("seed", "noise_seed"))
                and idx < len(values)
                and values[idx] in CONTROL_VALUES
            ):
                idx += 1
            if typ == DYNAMIC_COMBO:
                for opt in opts.get("options") or []:
                    if isinstance(opt, dict) and opt.get("key") == val:
                        take(_input_specs({"input": opt.get("inputs") or {}}), f"{prefix}{name}.")

    take(specs, "")
    return out


@dataclass
class Conversion:
    workflow: dict[str, dict[str, Any]]
    warnings: list[str] = field(default_factory=list)


def ui_to_api(ui: dict[str, Any], object_info: dict[str, Any]) -> Conversion:
    """画面用の形式(テンプレート)を API 形式に直す。直せないときは WorkflowError。"""
    if not isinstance(ui, dict) or not isinstance(ui.get("nodes"), list):
        raise WorkflowError("ワークフローの形が違います(nodes がありません)")
    defs = {
        str(sg.get("id")): sg
        for sg in ((ui.get("definitions") or {}).get("subgraphs") or [])
        if isinstance(sg, dict) and sg.get("id")
    }
    flat: list[_Node] = []
    alias: dict[tuple[str, int], tuple[str, int] | None] = {}
    _flatten(ui["nodes"], _links_map(ui.get("links")), defs, "", flat, alias)
    by_id = {n.id: n for n in flat}
    warnings: list[str] = []

    def resolve(src: tuple[str, int] | None, depth: int = 0) -> tuple[str, Any] | None:
        """つながりの元をたどる。("node", [id, slot]) / ("value", 値) / None(元が無い)。"""
        if src is None or depth > 64:
            return None
        if src in alias:
            return resolve(alias[src], depth + 1)
        node = by_id.get(src[0])
        if node is None:
            return None
        if node.type == PRIMITIVE:
            wv = node.widgets_values
            return ("value", wv[0]) if isinstance(wv, list) and wv else None
        if node.type == REROUTE:
            return resolve(node.inputs[0].src if node.inputs else None, depth + 1)
        if node.mode == MODE_NEVER:
            return None
        if node.mode == MODE_BYPASS:  # 素通し: 同じ型の入力をそのまま出す
            want = node.output_types[src[1]] if src[1] < len(node.output_types) else None
            for i in node.inputs:
                if i.src is not None and i.type == want:
                    return resolve(i.src, depth + 1)
            return None
        return ("node", [node.id, src[1]])

    api: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for n in flat:
        if n.type in UI_ONLY_SKIP or n.type in (PRIMITIVE, REROUTE):
            continue
        if n.mode in (MODE_NEVER, MODE_BYPASS):
            continue
        info = object_info.get(n.type)
        if info is None:
            missing.append(n.type)
            continue
        inputs = map_widgets(info, n.widgets_values)
        inputs.update(n.widget_overrides)
        for i in n.inputs:
            if i.src is None:
                continue
            r = resolve(i.src)
            if r is None:
                if not i.widget:
                    warnings.append(f"ノード {n.id}({n.type})の入力 {i.name} の元がありません")
                continue
            inputs[i.name] = r[1]
        api[n.id] = {"class_type": n.type, "inputs": inputs, "_meta": {"title": n.title or n.type}}
    if missing:
        names = ", ".join(sorted(set(missing)))
        raise WorkflowError(
            f"ComfyUI に無いノードを使っています: {names}"
            "(ComfyUI が古いか、カスタムノードが要ります。"
            "ComfyUI を更新するか --workflow で API 形式を渡してください)"
        )
    if not any(object_info.get(v["class_type"], {}).get("output_node") for v in api.values()):
        raise WorkflowError("保存・出力のノードがありません(実行しても何も動きません)")
    return Conversion(workflow=api, warnings=warnings)


# --- 確認 -------------------------------------------------------------------------------------


def check_inputs(api: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> list[str]:
    """選択肢の入力(モデルのファイル名など)がサーバーにあるか。足りないものを文で返す。

    画像の読み込み(image_upload)は bench がアップロードするので見ない。
    """
    problems: list[str] = []
    for nid, node in api.items():
        info = object_info.get(node["class_type"]) or {}
        for name, typ, opts in _input_specs(info):
            if opts.get("image_upload") or opts.get("audio_upload") or opts.get("video_upload"):
                continue
            choices = combo_options(typ, opts)
            val = node["inputs"].get(name)
            if choices is None or val is None or isinstance(val, list):
                continue
            if val not in choices:
                kind = "モデルのファイル" if name.endswith("_name") or name == "ckpt_name" else "選択肢"
                problems.append(
                    f"{kind}がサーバーにありません: ノード {nid}({node['class_type']})の {name}={val}"
                )
    return problems


def upload_inputs(api: dict[str, dict[str, Any]], object_info: dict[str, Any]) -> list[tuple[str, str, str]]:
    """アップロードが要る入力(ノード ID, 入力名, ファイル名)のうち、サーバーに無いもの。"""
    out: list[tuple[str, str, str]] = []
    for nid, node in api.items():
        info = object_info.get(node["class_type"]) or {}
        for name, typ, opts in _input_specs(info):
            if not opts.get("image_upload"):
                continue
            val = node["inputs"].get(name)
            choices = combo_options(typ, opts) or []
            if isinstance(val, str) and val and val not in choices:
                out.append((nid, name, val))
    return out


# --- 設定の書き換え -----------------------------------------------------------------------------

_SIZE_SKIP = re.compile(r"^(Save|Load|Preview)")


def _literal(node: dict[str, Any], key: str) -> bool:
    v = node["inputs"].get(key)
    return key in node["inputs"] and not isinstance(v, list)


def _size_nodes(api: dict[str, dict[str, Any]]) -> list[str]:
    """幅・高さを持つノード(空の潜在画像・動画の潜在など)。"""
    return [
        nid
        for nid, n in api.items()
        if "width" in n["inputs"] and "height" in n["inputs"] and not _SIZE_SKIP.match(n["class_type"])
    ]


FRAME_KEYS = ("length", "frames", "num_frames", "video_frames")
SECONDS_KEYS = ("seconds", "duration", "audio_length")


def apply_settings(
    api: dict[str, dict[str, Any]],
    *,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    seconds: float | None = None,
    steps: int | None = None,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """設定を入力の名前で書き換えた写しと、書き換えられなかった理由を返す。"""
    wf = copy.deepcopy(api)
    notes: list[str] = []

    def set_all(keys: tuple[str, ...], value: Any, nodes: list[str], label: str) -> None:
        hit = False
        for nid in nodes:
            n = wf[nid]
            for k in keys:
                if k not in n["inputs"]:
                    continue
                if _literal(n, k):
                    n["inputs"][k] = value
                    hit = True
                else:
                    notes.append(f"ノード {nid} の {k} は他のノードから来るので {label} を変えられません")
        if not hit:
            notes.append(f"{label} を設定できるノードがありません")

    size = _size_nodes(wf)
    if width is not None:
        set_all(("width",), int(width), size, "幅")
    if height is not None:
        set_all(("height",), int(height), size, "高さ")
    if frames is not None:
        set_all(FRAME_KEYS, int(frames), size, "フレーム数")
    if seconds is not None:
        set_all(SECONDS_KEYS, float(seconds), list(wf), "長さ(秒)")
    if steps is not None:
        set_all(("steps",), int(steps), list(wf), "ステップ数")
    return wf, notes


def set_output_prefix(api: dict[str, dict[str, Any]], prefix: str) -> None:
    """保存ノードの filename_prefix をそろえる(利用者の出力に混ざらないように)。"""
    for n in api.values():
        if n["class_type"].startswith("Save") and isinstance(n["inputs"].get("filename_prefix"), str):
            n["inputs"]["filename_prefix"] = prefix


def _first(api: dict[str, dict[str, Any]], keys: tuple[str, ...], nodes: list[str] | None = None) -> Any:
    for nid in nodes if nodes is not None else list(api):
        n = api[nid]
        for k in keys:
            if _literal(n, k):
                return n["inputs"][k]
    return None


def effective_settings(api: dict[str, dict[str, Any]], precision: str | None = None) -> Settings:
    """API 形式のワークフローから、実際に使う設定を読む。"""
    size = _size_nodes(api)
    latent = [nid for nid in size if "Latent" in api[nid]["class_type"]] or size

    def num(v: Any, cast: type) -> Any:
        try:
            return cast(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    batch = num(_first(api, ("batch_size",), latent or None), int) or 1
    return Settings(
        width=num(_first(api, ("width",), latent), int),
        height=num(_first(api, ("height",), latent), int),
        frames=num(_first(api, FRAME_KEYS, latent), int),
        seconds=num(_first(api, SECONDS_KEYS), float),
        steps=num(_first(api, ("steps",)), int),
        precision=precision,
        batch=batch,
    )
