"""学習データの名前による除外(DATASET_TERMS)。"""

from genmodelfit.db import load
from genmodelfit.safety import blocked_terms


def test_e621_is_excluded_only_for_uncensored():
    assert blocked_terms(["hearmeneigh/sd21-e621-rising-v2"], "", uncensored=True) == ["dataset:e621"]
    assert blocked_terms(["x"], "Fine-tuned on e621 posts.", uncensored=True) == ["dataset:e621"]
    assert blocked_terms(["e621-rising"], "", uncensored=False) == []
    assert blocked_terms(["ae6210", "e6210x"], "", uncensored=True) == []


def test_e621_in_the_card_text_is_excluded():
    """NoobAI-XL のように、名前ではなくカードの本文にだけ書かれている場合も外す。

    update はカードと README の文を渡す。"""
    card = (
        "NoobAI XL is a text-to-image model trained on the full Danbooru and e621 datasets "
        "(https://huggingface.co/datasets/NebulaeWis/e621-2024-webp-4Mpixel)."
    )
    names = ["hf:Laxhar/noobai-XL-1.1", "noobai-XL-1.1"]
    assert blocked_terms(names, card, uncensored=True) == ["dataset:e621"]


def test_bundled_list_has_no_dataset_hits():
    db = load(use_cache=False)
    assert db.get("hf:hearmeneigh/sd21-e621-rising-v2") is None
    assert db.get("hf:Laxhar/noobai-XL-1.1") is None
    for m in db.models:
        assert blocked_terms([m.id, m.name], "", uncensored=m.content == "uncensored") == [], m.id
