"""学習データの名前による除外(DATASET_TERMS)。"""

from genmodelfit.db import load
from genmodelfit.safety import blocked_terms


def test_e621_is_excluded_only_for_uncensored():
    assert blocked_terms(["hearmeneigh/sd21-e621-rising-v2"], "", uncensored=True) == ["dataset:e621"]
    assert blocked_terms(["x"], "Fine-tuned on e621 posts.", uncensored=True) == ["dataset:e621"]
    assert blocked_terms(["e621-rising"], "", uncensored=False) == []
    assert blocked_terms(["ae6210", "e6210x"], "", uncensored=True) == []


def test_bundled_list_has_no_dataset_hits():
    db = load(use_cache=False)
    assert db.get("hf:hearmeneigh/sd21-e621-rising-v2") is None
    for m in db.models:
        assert blocked_terms([m.id, m.name], "", uncensored=m.content == "uncensored") == [], m.id
