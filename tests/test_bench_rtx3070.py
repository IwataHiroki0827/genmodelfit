"""2026-10-19 に RTX 3070 で測った実測と、そのとき見つけた変換の不具合の回帰。"""

from genmodelfit import workflow as w
from genmodelfit.calibrate import CALIBRATION_METRIC, build_rows
from genmodelfit.db import load

MOGE_INFO = {
    "input": {
        "required": {
            "moge_model": ["MOGE_MODEL", {}],
            "image": ["IMAGE", {}],
            "resolution_level": ["INT", {"default": 9, "min": 0, "max": 9}],
            "fov_x_degrees": ["FLOAT", {"default": 0.0}],
            "batch_size": ["INT", {"default": 4}],
            "force_projection": ["BOOLEAN", {"default": True}],
            "apply_mask": ["BOOLEAN", {"default": True}],
            "refine_steps": ["INT", {"default": 3, "min": 0, "max": 8}],
        }
    },
    "input_order": {
        "required": [
            "moge_model",
            "image",
            "resolution_level",
            "fov_x_degrees",
            "batch_size",
            "force_projection",
            "apply_mask",
            "refine_steps",
        ]
    },
    "output": ["MOGE_GEOMETRY"],
    "output_node": False,
}
SAVE_INFO = {"input": {"required": {"geometry": ["MOGE_GEOMETRY", {}]}}, "output": [], "output_node": True}


def _ui(widgets):
    return {
        "nodes": [
            {
                "id": 1,
                "type": "MoGeInference",
                "mode": 0,
                "inputs": [],
                "outputs": [{"name": "g", "type": "MOGE_GEOMETRY"}],
                "widgets_values": widgets,
            },
            {
                "id": 2,
                "type": "SaveGeom",
                "mode": 0,
                "inputs": [{"name": "geometry", "type": "MOGE_GEOMETRY", "link": 1}],
                "outputs": [],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "MOGE_GEOMETRY"]],
    }


def test_missing_required_widget_filled_with_default():
    """テンプレートが古く refine_steps の値が無くても、既定値で埋めて注意を出す(ComfyUI 0.40 で起きた)。"""
    conv = w.ui_to_api(_ui([9, 0, 4, True, True]), {"MoGeInference": MOGE_INFO, "SaveGeom": SAVE_INFO})
    node = conv.workflow["1"]["inputs"]
    assert node["refine_steps"] == 3 and node["resolution_level"] == 9
    assert any("refine_steps" in x for x in conv.warnings)


def test_present_widget_values_are_kept():
    conv = w.ui_to_api(_ui([5, 0, 2, True, False, 1]), {"MoGeInference": MOGE_INFO, "SaveGeom": SAVE_INFO})
    node = conv.workflow["1"]["inputs"]
    assert (node["resolution_level"], node["batch_size"], node["apply_mask"], node["refine_steps"]) == (
        5,
        2,
        False,
        1,
    )
    assert not conv.warnings


def test_rtx3070_rows_are_bundled_but_not_used_for_calibration():
    db = load(use_cache=False)
    # 作者の実測だけ(同梱のほかの利用者のまとめに RTX 3070 の提出が入っても変わらない)
    rows3070 = [m for m in db.measurements if "3070" in m.gpu_name and m.origin != "community"]
    assert {m.model_id for m in rows3070} == {
        "utility_birefnet_remove_background",
        "3d_moge_perspective_to_mesh",
    }
    assert all(m.vram_metric == "torch_reserved" and m.outcome == "ok" and m.standard for m in rows3070)
    assert CALIBRATION_METRIC is None
    used = {r.m.model_id for r in build_rows(db)}
    assert not used & {m.model_id for m in rows3070}
    assert len(build_rows(db)) == 9


def test_sizeless_template_records_no_size_and_batch_one():
    """大きさの入力の無いテンプレート(MoGe)は幅・高さを書かず、MoGe の batch_size(処理の単位)を枚数と
    読まない(#90)。大きさの入力のあるテンプレートは今までどおり。"""
    conv = w.ui_to_api(_ui([9, 0, 4, True, True, 3]), {"MoGeInference": MOGE_INFO, "SaveGeom": SAVE_INFO})
    assert not w.size_input(conv.workflow)
    s = w.effective_settings(conv.workflow, precision="fp16")
    assert (s.width, s.height, s.batch) == (None, None, 1)
    sized = {
        "1": {"class_type": "EmptyLatentImage", "inputs": {"width": 832, "height": 640, "batch_size": 2}},
        "2": {"class_type": "ImageScale", "inputs": {"image": ["9", 0], "width": ["7", 0], "height": 512}},
    }
    assert w.size_input(sized)
    s = w.effective_settings(sized)
    assert (s.width, s.height, s.batch) == (832, 640, 2)
    linked = {
        "2": {"class_type": "ImageScale", "inputs": {"image": ["9", 0], "width": ["7", 0], "height": 512}}
    }
    assert not w.size_input(linked) and w.effective_settings(linked).width is None
    audio = {"1": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {"seconds": 20.0, "batch_size": 2}}}
    assert w.effective_settings(audio).batch == 2  # 音声の潜在の枚数は今までどおり読む
