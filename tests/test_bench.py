"""bench と workflow の試験。ComfyUI は偽の HTTP サーバーで置き換える(実物は要らない)。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from genmodelfit import bench, cli
from genmodelfit.bench import BenchError, BenchOptions, ComfyClient, offload_from_logs
from genmodelfit.db import load_measurements
from genmodelfit.workflow import (
    WorkflowError,
    apply_settings,
    check_inputs,
    effective_settings,
    is_api_format,
    map_widgets,
    ui_to_api,
)

GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("GENMODELFIT_CACHE_DIR", str(tmp_path / "cache"))
    # 実機の nvidia-smi・プロセス探しには触らない
    monkeypatch.setattr(bench, "smi_gpus", lambda: [])
    monkeypatch.setattr(bench, "gpu_used_now", lambda idx: None)
    monkeypatch.setattr(bench, "find_server_process", lambda url: None)
    monkeypatch.setattr(bench.time, "sleep", lambda s: None)


# --- /object_info の抜粋 ---------------------------------------------------------------------------


def _w(typ, **opts):
    return [typ, opts] if opts else [typ]


OBJECT_INFO: dict[str, Any] = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["model.safetensors"]]}},
        "input_order": {"required": ["ckpt_name"]},
        "output_node": False,
    },
    "UNETLoader": {
        "input": {"required": {"unet_name": [["unet.safetensors"]], "weight_dtype": [["default", "fp8"]]}},
        "input_order": {"required": ["unet_name", "weight_dtype"]},
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": _w("STRING", multiline=True), "clip": _w("CLIP")}},
        "input_order": {"required": ["text", "clip"]},
    },
    "EmptyLatentImage": {
        "input": {"required": {"width": _w("INT"), "height": _w("INT"), "batch_size": _w("INT")}},
        "input_order": {"required": ["width", "height", "batch_size"]},
    },
    "EmptyHunyuanLatentVideo": {
        "input": {
            "required": {
                "width": _w("INT"),
                "height": _w("INT"),
                "length": _w("INT"),
                "batch_size": _w("INT"),
            }
        },
    },
    "EmptyAceStep1.5LatentAudio": {
        "input": {"required": {"seconds": _w("FLOAT"), "batch_size": _w("INT")}},
    },
    "KSampler": {
        "input": {
            "required": {
                "model": _w("MODEL"),
                "seed": _w("INT", control_after_generate=True),
                "steps": _w("INT"),
                "cfg": _w("FLOAT"),
                "sampler_name": [["euler", "dpmpp_2m"]],
                "scheduler": [["normal", "karras"]],
                "positive": _w("CONDITIONING"),
                "negative": _w("CONDITIONING"),
                "latent_image": _w("LATENT"),
                "denoise": _w("FLOAT"),
            }
        },
        "input_order": {
            "required": [
                "model",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "positive",
                "negative",
                "latent_image",
                "denoise",
            ]
        },
    },
    "VAEDecode": {"input": {"required": {"samples": _w("LATENT"), "vae": _w("VAE")}}},
    "SaveImage": {
        "input": {"required": {"images": _w("IMAGE"), "filename_prefix": _w("STRING")}},
        "output_node": True,
    },
    "LoadImage": {"input": {"required": {"image": [["a.png"], {"image_upload": True}]}}},
    "SaveAudioAdvanced": {
        "input": {
            "required": {
                "audio": _w("AUDIO"),
                "filename_prefix": _w("STRING"),
                "format": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {"key": "flac", "inputs": {"required": {}}},
                            {
                                "key": "mp3",
                                "inputs": {"required": {"quality": ["COMBO", {"options": ["V0"]}]}},
                            },
                        ]
                    },
                ],
            }
        },
        "output_node": True,
    },
}


def node(nid, ntype, widgets=None, inputs=None, outputs=None, mode=0, **extra):
    return {
        "id": nid,
        "type": ntype,
        "mode": mode,
        "widgets_values": widgets if widgets is not None else [],
        "inputs": inputs or [],
        "outputs": outputs or [],
        **extra,
    }


def sdxl_ui() -> dict[str, Any]:
    """画面用の形式の小さな SDXL(Note と Reroute と素通しのノード入り)。"""
    return {
        "nodes": [
            node(1, "CheckpointLoaderSimple", ["model.safetensors"]),
            node(2, "CLIPTextEncode", ["a cat"], [{"name": "clip", "type": "CLIP", "link": 2}]),
            node(3, "CLIPTextEncode", ["blurry"], [{"name": "clip", "type": "CLIP", "link": 3}]),
            node(4, "EmptyLatentImage", [1024, 1024, 1]),
            node(
                5,
                "KSampler",
                [42, "randomize", 25, 7, "dpmpp_2m", "karras", 1],
                [
                    {"name": "model", "type": "MODEL", "link": 9},
                    {"name": "positive", "type": "CONDITIONING", "link": 5},
                    {"name": "negative", "type": "CONDITIONING", "link": 6},
                    {"name": "latent_image", "type": "LATENT", "link": 7},
                ],
            ),
            node(6, "VAEDecode", [], [{"name": "samples", "link": 8}, {"name": "vae", "link": 10}]),
            node(7, "SaveImage", ["sdxl"], [{"name": "images", "link": 11}]),
            node(8, "MarkdownNote", ["# hello"]),
            node(9, "Reroute", [], [{"name": "", "type": "*", "link": 1}], [{"type": "MODEL"}]),
            node(
                10,
                "UNETLoader",
                ["unet.safetensors", "default"],
                [{"name": "model", "type": "MODEL", "link": None}],
                [{"type": "MODEL"}],
                mode=4,
            ),
        ],
        "links": [
            [1, 1, 0, 9, 0, "MODEL"],
            [2, 1, 1, 2, 0, "CLIP"],
            [3, 1, 1, 3, 0, "CLIP"],
            [5, 2, 0, 5, 1, "CONDITIONING"],
            [6, 3, 0, 5, 2, "CONDITIONING"],
            [7, 4, 0, 5, 3, "LATENT"],
            [8, 5, 0, 6, 0, "LATENT"],
            [9, 9, 0, 5, 0, "MODEL"],
            [10, 1, 2, 6, 1, "VAE"],
            [11, 6, 0, 7, 0, "IMAGE"],
        ],
    }


def subgraph_ui() -> dict[str, Any]:
    """Z-Image のテンプレートと同じ形のサブグラフ(幅・高さ・ステップを外に出したもの)。"""
    sg_id = "f2fdebf6-0000"
    inner = [
        node(
            28,
            "UNETLoader",
            ["unet.safetensors", "default"],
            [{"name": "unet_name", "widget": {}, "link": 73}],
        ),
        node(
            27,
            "CLIPTextEncode",
            ["inside prompt"],
            [{"name": "clip", "link": None}, {"name": "text", "link": 34}],
        ),
        node(
            13,
            "EmptyLatentImage",
            [1024, 1024, 1],
            [{"name": "width", "widget": {}, "link": 35}, {"name": "height", "widget": {}, "link": 36}],
        ),
        node(
            3,
            "KSampler",
            [7, "fixed", 8, 1, "euler", "normal", 1],
            [
                {"name": "model", "link": 13},
                {"name": "positive", "link": 30},
                {"name": "negative", "link": 30},
                {"name": "latent_image", "link": 17},
                {"name": "steps", "widget": {}, "link": 72},
            ],
        ),
        node(8, "VAEDecode", [], [{"name": "samples", "link": 14}, {"name": "vae", "link": None}]),
    ]

    def lk(i, o, os_, t, ts):
        return {"id": i, "origin_id": o, "origin_slot": os_, "target_id": t, "target_slot": ts, "type": "*"}

    sg = {
        "id": sg_id,
        "inputs": [{"name": n, "linkIds": [lid]} for n, lid in [("text", 34), ("width", 35), ("height", 36)]]
        + [{"name": "steps", "linkIds": [72]}, {"name": "unet_name", "linkIds": [73]}],
        "outputs": [{"name": "IMAGE", "linkIds": [16]}],
        "nodes": inner,
        "links": [
            lk(13, 28, 0, 3, 0),
            lk(30, 27, 0, 3, 1),
            lk(17, 13, 0, 3, 3),
            lk(14, 3, 0, 8, 0),
            lk(16, 8, 0, -20, 0),
            lk(34, -10, 0, 27, 1),
            lk(35, -10, 1, 13, 0),
            lk(36, -10, 2, 13, 1),
            lk(72, -10, 3, 3, 4),
            lk(73, -10, 4, 28, 0),
        ],
    }
    outer = node(
        57,
        sg_id,
        [],
        [{"name": "text", "widget": {"name": "text"}, "link": 100}],
        [{"name": "IMAGE", "links": [62]}],
        properties={"proxyWidgets": [["13", "width"], ["13", "height"], ["3", "steps"]]},
    )
    prim = node(99, "PrimitiveNode", ["from primitive", "fixed"], [], [{"type": "STRING", "links": [100]}])
    save = node(9, "SaveImage", ["z"], [{"name": "images", "link": 62}])
    return {
        "nodes": [save, outer, prim],
        "links": [[62, 57, 0, 9, 0, "IMAGE"], [100, 99, 0, 57, 0, "STRING"]],
        "definitions": {"subgraphs": [sg]},
    }


# --- 変換 --------------------------------------------------------------------------------------


def test_map_widgets_skips_control_after_generate():
    got = map_widgets(OBJECT_INFO["KSampler"], [42, "randomize", 25, 7, "dpmpp_2m", "karras", 1])
    assert got == {
        "seed": 42,
        "steps": 25,
        "cfg": 7,
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "denoise": 1,
    }
    # 「生成後の動き」が無い古い形でもずれない
    got = map_widgets(OBJECT_INFO["KSampler"], [42, 25, 7, "euler", "normal", 1])
    assert got["steps"] == 25 and got["denoise"] == 1


def test_map_widgets_dynamic_combo_and_extra_values():
    got = map_widgets(OBJECT_INFO["SaveAudioAdvanced"], ["audio/x", "mp3", "V0"])
    assert got == {"filename_prefix": "audio/x", "format": "mp3", "format.quality": "V0"}
    # LoadImage の後ろの「upload」の値は捨てる
    assert map_widgets(OBJECT_INFO["LoadImage"], ["a.png", "image"]) == {"image": "a.png"}


def test_ui_to_api_resolves_links_reroute_bypass_and_notes():
    conv = ui_to_api(sdxl_ui(), OBJECT_INFO)
    wf = conv.workflow
    assert set(wf) == {"1", "2", "3", "4", "5", "6", "7"}  # Note・Reroute・素通しのノードは消える
    assert wf["5"]["inputs"]["model"] == ["1", 0]  # Reroute の先をたどる
    assert wf["5"]["inputs"]["seed"] == 42 and wf["5"]["inputs"]["steps"] == 25
    assert wf["6"]["inputs"]["vae"] == ["1", 2]
    assert wf["2"]["inputs"] == {"text": "a cat", "clip": ["1", 1]}
    assert is_api_format(wf) and not is_api_format(sdxl_ui())


def test_ui_to_api_flattens_subgraph_and_primitive():
    wf = ui_to_api(subgraph_ui(), OBJECT_INFO).workflow
    assert wf["9"]["inputs"]["images"] == ["57:8", 0]  # サブグラフの出力 → 中のノード
    assert wf["57:3"]["inputs"]["latent_image"] == ["57:13", 0]
    assert wf["57:27"]["inputs"]["text"] == "from primitive"  # 外の PrimitiveNode の値が中まで届く
    assert wf["57:13"]["inputs"]["width"] == 1024  # 外につながっていない入力は中の値
    assert wf["57:3"]["inputs"]["steps"] == 8


def test_ui_to_api_proxy_widget_values():
    ui = subgraph_ui()
    ui["nodes"][1]["widgets_values"] = [640, 480, 4]
    wf = ui_to_api(ui, OBJECT_INFO).workflow
    assert (wf["57:13"]["inputs"]["width"], wf["57:13"]["inputs"]["height"]) == (640, 480)
    assert wf["57:3"]["inputs"]["steps"] == 4


def test_ui_to_api_rejects_unknown_nodes_and_no_output():
    ui = sdxl_ui()
    ui["nodes"].append(node(50, "SomeCustomNode", [1]))
    with pytest.raises(WorkflowError, match="SomeCustomNode"):
        ui_to_api(ui, OBJECT_INFO)
    ui = sdxl_ui()
    ui["nodes"] = [n for n in ui["nodes"] if n["type"] != "SaveImage"]
    with pytest.raises(WorkflowError, match="出力"):
        ui_to_api(ui, OBJECT_INFO)
    with pytest.raises(WorkflowError):
        ui_to_api({"foo": 1}, OBJECT_INFO)


def test_check_inputs_reports_missing_model_files():
    wf = ui_to_api(sdxl_ui(), OBJECT_INFO).workflow
    assert check_inputs(wf, OBJECT_INFO) == []
    wf["1"]["inputs"]["ckpt_name"] = "missing.safetensors"
    (msg,) = check_inputs(wf, OBJECT_INFO)
    assert "モデルのファイル" in msg and "missing.safetensors" in msg


# --- 設定の書き換え -----------------------------------------------------------------------------


def test_apply_settings_image_video_audio():
    wf = ui_to_api(sdxl_ui(), OBJECT_INFO).workflow
    new, notes = apply_settings(wf, width=832, height=640, steps=15)
    assert new["4"]["inputs"]["width"] == 832 and new["4"]["inputs"]["height"] == 640
    assert new["5"]["inputs"]["steps"] == 15 and wf["5"]["inputs"]["steps"] == 25  # 元は変えない
    assert notes == []
    s = effective_settings(new, precision="fp16")
    assert (s.width, s.height, s.steps, s.frames, s.seconds, s.precision) == (
        832,
        640,
        15,
        None,
        None,
        "fp16",
    )

    video = {
        "1": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": 832, "height": 480, "length": 33}},
        "2": {"class_type": "KSampler", "inputs": {"steps": 30, "latent_image": ["1", 0]}},
    }
    new, _ = apply_settings(video, width=512, height=512, frames=49, steps=20)
    s = effective_settings(new)
    assert (s.width, s.height, s.frames, s.steps) == (512, 512, 49, 20)

    audio = {
        "1": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 120.0, "batch_size": 1}},
        "2": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {"duration": 120.0, "seed": 1}},
    }
    new, notes = apply_settings(audio, seconds=20, width=512)
    assert new["1"]["inputs"]["seconds"] == 20.0 and new["2"]["inputs"]["duration"] == 20.0
    assert any("幅" in n for n in notes)  # 音声には幅が無い
    assert effective_settings(new).seconds == 20.0


def test_apply_settings_linked_input_is_reported():
    wf = {"1": {"class_type": "EmptyLatentImage", "inputs": {"width": ["9", 0], "height": 512}}}
    new, notes = apply_settings(wf, width=768)
    assert new["1"]["inputs"]["width"] == ["9", 0]
    assert any("他のノード" in n for n in notes)


def test_offload_from_logs():
    assert offload_from_logs(["loaded completely; 7000 MB usable"]) == "none"
    assert offload_from_logs(["loaded completely", "loaded partially; 3000 MB loaded"]) == "partial"
    assert offload_from_logs(["nothing"]) is None


# --- 偽の ComfyUI ----------------------------------------------------------------------------------


class FakeComfy:
    """必要な API だけを持つ偽の ComfyUI。behavior で動きを変える。"""

    def __init__(self, behavior: str = "ok"):
        self.behavior = behavior
        self.submitted: list[dict[str, Any]] = []
        self.freed = 0
        self.interrupted = 0
        self.deleted: list[str] = []
        self.uploads: list[str] = []
        self.polls = 0
        self.torch = 0
        self.history_calls = 0
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *a):
        self.server.shutdown()
        self.server.server_close()

    def _handler(self):
        fake = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code: int, obj: Any) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _body(self) -> Any:
                n = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(n) if n else b""
                try:
                    return json.loads(raw or b"{}")
                except ValueError:
                    return raw

            def do_GET(self):
                p = self.path
                if p == "/system_stats":
                    fake.polls += 1
                    if fake.submitted and fake.behavior != "timeout":
                        fake.torch = 5 * GIB  # 実行中は 5 GB
                    dev = {
                        "name": "cuda:0 NVIDIA GeForce RTX 3070 : cudaMallocAsync",
                        "type": "cuda",
                        "vram_total": 8 * GIB,
                        "vram_free": 8 * GIB - fake.torch - GIB,
                        "torch_vram_total": fake.torch,
                    }
                    return self._send(200, {"system": {"comfyui_version": "0.40.0"}, "devices": [dev]})
                if p == "/object_info":
                    return self._send(200, OBJECT_INFO)
                if p == "/queue":
                    running = [[0, "other", {}]] if fake.behavior == "busy" else []
                    if fake.behavior == "timeout" and fake.submitted:
                        running = [[0, "pid-1", {}]]
                    return self._send(200, {"queue_running": running, "queue_pending": []})
                if p.startswith("/templates/"):
                    return self._send(200, sdxl_ui()) if "image_sdxl" in p else self._send(404, {})
                if p.startswith("/history/"):
                    fake.history_calls += 1
                    if fake.behavior == "timeout" or fake.history_calls < 3:
                        return self._send(200, {})
                    msgs = [["execution_start", {"timestamp": 1000}]]
                    if fake.behavior == "cached":
                        msgs.append(
                            ["execution_cached", {"nodes": list(fake.submitted[-1]), "timestamp": 1000}]
                        )
                    if fake.behavior in ("error", "oom"):
                        exc = "torch.OutOfMemoryError" if fake.behavior == "oom" else "ValueError"
                        msgs.append(
                            [
                                "execution_error",
                                {
                                    "timestamp": 3000,
                                    "node_type": "KSampler",
                                    "exception_type": exc,
                                    "exception_message": "boom",
                                },
                            ]
                        )
                        st = {"status_str": "error", "completed": False, "messages": msgs}
                    else:
                        msgs.append(["execution_success", {"timestamp": 3500}])
                        st = {"status_str": "success", "completed": True, "messages": msgs}
                    return self._send(200, {"pid-1": {"status": st, "outputs": {}}})
                if p == "/internal/logs/raw":
                    lines = [{"t": "1", "m": "start"}]
                    if fake.submitted:
                        lines.append({"t": "2", "m": "loaded completely; 7000 MB usable"})
                    return self._send(200, {"entries": lines})
                return self._send(404, {})

            def do_POST(self):
                p = self.path
                body = self._body()
                if p == "/free":
                    fake.freed += 1
                    fake.torch = 0
                    return self._send(200, {})
                if p == "/prompt":
                    fake.submitted.append(body["prompt"])
                    if fake.behavior == "reject":
                        return self._send(
                            400,
                            {
                                "error": {"type": "prompt_outputs_failed_validation", "message": "bad"},
                                "node_errors": {
                                    "5": {"class_type": "KSampler", "errors": [{"message": "Value too big"}]}
                                },
                            },
                        )
                    return self._send(200, {"prompt_id": "pid-1", "number": 1, "node_errors": {}})
                if p == "/interrupt":
                    fake.interrupted += 1
                    return self._send(200, {})
                if p == "/queue":
                    fake.deleted += body.get("delete", [])
                    return self._send(200, {})
                if p == "/upload/image":
                    fake.uploads.append("x")
                    return self._send(200, {"name": "uploaded.png"})
                return self._send(404, {})

        return H


def _opts(url: str, **kw) -> BenchOptions:
    return BenchOptions(model_id="image_sdxl_simple", url=url, interval=0.01, **kw)


def test_bench_measures_with_fake_server():
    with FakeComfy() as fake:
        res = bench.run(_opts(fake.url, width=832, height=832, steps=15), log=lambda m: None)
    m = res.measurement
    assert fake.freed == 1  # 開始前に自分のサーバーのモデルを解放
    assert m.gpu_name == "NVIDIA GeForce RTX 3070" and m.gpu_vram_gb == 8.0
    assert m.peak_vram_gb == pytest.approx(5.0, abs=0.01)
    assert m.comfyui_version == "0.40.0" and m.offload == "none"
    assert (m.settings.width, m.settings.height, m.settings.steps) == (832, 832, 15)
    assert m.seconds is not None and m.date and "/free" in m.note and "2.5 秒" in m.note
    assert res.workflow_source == "server-template"
    sent = fake.submitted[0]
    assert sent["4"]["inputs"]["width"] == 832 and sent["7"]["inputs"]["filename_prefix"].startswith(
        "genmodelfit"
    )


def test_bench_refuses_busy_queue():
    with FakeComfy("busy") as fake, pytest.raises(BenchError, match="キュー"):
        bench.run(_opts(fake.url), log=lambda m: None)
    assert fake.submitted == [] and fake.freed == 0  # 他の人の仕事がある間は何もしない


def test_bench_execution_error_and_reject():
    with FakeComfy("error") as fake, pytest.raises(BenchError, match="ValueError"):
        bench.run(_opts(fake.url), log=lambda m: None)
    with FakeComfy("reject") as fake, pytest.raises(BenchError, match="Value too big"):
        bench.run(_opts(fake.url), log=lambda m: None)


def test_bench_oom_is_recorded():
    with FakeComfy("oom") as fake:
        res = bench.run(_opts(fake.url), log=lambda m: None)
    assert res.failed and "OutOfMemory" in res.failed
    assert res.measurement.offload == "oom" and "メモリ不足" in res.measurement.note


def test_bench_all_cached_is_refused():
    with FakeComfy("cached") as fake, pytest.raises(BenchError, match="前の結果"):
        bench.run(_opts(fake.url, free_before=False), log=lambda m: None)
    assert fake.freed == 0


def test_bench_timeout_interrupts_own_prompt():
    with FakeComfy("timeout") as fake, pytest.raises(BenchError, match="中止"):
        bench.run(_opts(fake.url, timeout=0.0), log=lambda m: None)
    assert fake.interrupted == 1


def test_bench_missing_model_and_unknown_template(monkeypatch):
    with FakeComfy() as fake:
        monkeypatch.setattr(bench, "template_url", lambda name: fake.url + "/nothing")  # GitHub に行かない
        with pytest.raises(BenchError, match="見つかりません"):
            bench.run(BenchOptions(model_id="no_such_template_xyz", url=fake.url), log=lambda m: None)


def test_bench_api_workflow_file_and_set(tmp_path):
    wf = ui_to_api(sdxl_ui(), OBJECT_INFO).workflow
    wf["1"]["inputs"]["ckpt_name"] = "missing.safetensors"
    p = tmp_path / "api.json"
    p.write_text(json.dumps(wf), encoding="utf-8")
    with FakeComfy() as fake:
        with pytest.raises(BenchError, match=r"missing\.safetensors"):
            bench.run(_opts(fake.url, workflow_path=str(p)), log=lambda m: None)
        wf["1"]["inputs"]["ckpt_name"] = "model.safetensors"
        p.write_text(json.dumps(wf), encoding="utf-8")
        res = bench.run(_opts(fake.url, workflow_path=str(p), sets={"cfg": 3.5}), log=lambda m: None)
    assert res.workflow_source == "file:api" and fake.submitted[-1]["5"]["inputs"]["cfg"] == 3.5
    assert "cfg=3.5" in res.measurement.note


def test_cli_bench_json_saves_measurement(tmp_path, capsys):
    out = tmp_path / "m.json"
    with FakeComfy() as fake:
        code = cli.main(
            [
                "bench",
                "image_sdxl_simple",
                "--url",
                fake.url,
                "--resolution",
                "640x480",
                "--json",
                "--output",
                str(out),
            ]
        )
    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["command"] == "bench" and data["measurement"]["settings"]["width"] == 640
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["measurements"][0]["model_id"] == "image_sdxl_simple"
    text = out.read_text(encoding="utf-8")
    assert str(Path.home()) not in text and "\\Users\\" not in text  # パスを入れない


def test_cli_bench_default_path_is_read_back(capsys):
    with FakeComfy() as fake:
        assert cli.main(["bench", "image_sdxl_simple", "--url", fake.url]) == 0
    assert "最大 VRAM" in capsys.readouterr().out
    ms = [
        m
        for m in load_measurements()
        if m.gpu_name == "NVIDIA GeForce RTX 3070" and m.source.startswith("genmodelfit")
    ]
    assert ms and ms[0].model_id == "image_sdxl_simple"  # キャッシュに保存した実測が推定に使われる


def test_cli_bench_errors(capsys):
    with FakeComfy("busy") as fake:
        assert cli.main(["bench", "image_sdxl_simple", "--url", fake.url, "--json"]) == 1
    assert "キュー" in json.loads(capsys.readouterr().out)["error"]
    assert cli.main(["bench", "x", "--url", "http://127.0.0.1:9", "--resolution", "big"]) == 1
    assert cli.main(["bench", "x", "--url", "http://127.0.0.1:9", "--timeout", "1"]) == 1  # 接続できない


def test_client_template_404_returns_none():
    with FakeComfy() as fake:
        assert ComfyClient(fake.url).template("nothing") is None


# --- 実物のテンプレート(Comfy-Org/workflow_templates、MIT)と ComfyUI 0.40.0 の /object_info ---------

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_real_template_z_image_subgraph():
    info = _fixture("object_info_subset.json")["nodes"]
    conv = ui_to_api(_fixture("image_z_image_turbo_int8.json"), info)
    wf = conv.workflow
    assert conv.warnings == []
    types = sorted(n["class_type"] for n in wf.values())
    assert types.count("KSampler") == 1 and "MarkdownNote" not in types
    sampler = next(n for n in wf.values() if n["class_type"] == "KSampler")
    assert sampler["inputs"]["steps"] == 8 and sampler["inputs"]["sampler_name"] == "res_multistep"
    unet = next(n for n in wf.values() if n["class_type"] == "UNETLoader")
    assert unet["inputs"]["unet_name"] == "z_image_turbo_int8_convrot.safetensors"
    new, notes = apply_settings(wf, width=768, height=768, steps=6)
    s = effective_settings(new)
    assert (s.width, s.height, s.steps, notes) == (768, 768, 6, [])


def test_real_template_ace_step_primitive_and_dynamic_combo():
    info = _fixture("object_info_subset.json")["nodes"]
    wf = ui_to_api(_fixture("audio_ace_step_1_5_checkpoint.json"), info).workflow
    enc = next(n for n in wf.values() if n["class_type"] == "TextEncodeAceStepAudio1.5")
    # PrimitiveNode(120 秒・seed 31)の値が両方のノードに入る
    assert enc["inputs"]["duration"] == 120 and enc["inputs"]["seed"] == 31 and enc["inputs"]["bpm"] == 190
    assert enc["inputs"]["language"] == "en" and enc["inputs"]["top_k"] == 0
    save = next(n for n in wf.values() if n["class_type"] == "SaveAudioAdvanced")
    assert save["inputs"]["format"] == "flac"
    new, _ = apply_settings(wf, seconds=20, steps=8)
    assert effective_settings(new).seconds == 20.0
    latent = next(n for n in new.values() if n["class_type"] == "EmptyAceStep1.5LatentAudio")
    assert latent["inputs"]["seconds"] == 20.0
