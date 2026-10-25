def test_version_numbers_and_tts_do_not_pick_a_family():
    """v1.5 のような版の番号だけで SD にしない。読み上げには系統を付けない。

    animation / animatediff は Anima ではない。"""
    from genmodelfit.db import load
    from genmodelfit.families import family_of

    db = load(use_cache=False)
    assert family_of(db.get("hf:TenStrip/LTX2.3-10Eros")) == "ltxv"
    assert family_of(db.get("hf:rockerBOO/ltx2.3-10eros-nvfp4")) == "ltxv"
    assert family_of(db.get("hf:OpenMOSS-Team/MOSS-TTS-v1.5")) is None
    animatediff = "templates_purz_animatediff_simple_weighted_ipadapters_looping_animation"
    assert family_of(db.get(animatediff)) != "anima"


def test_shown_texts_have_english():
    """理由・注意に出す表の文(系統の名前・根拠の行・注意・CPU の根拠)は、
    日本語を含めば英語の訳がある(#73)。"""
    import re

    from genmodelfit import i18n
    from genmodelfit.families import _EN, CPU_FAMILIES, FAMILIES, VAES, shown

    ja = re.compile(r"[぀-ヿ一-鿿＀-￯]")
    texts = [t for f in FAMILIES.values() for t in (f.comfy_class, f.source, f.note)]
    texts += [v.source for v in VAES.values()]
    texts += [t for r in CPU_FAMILIES.values() for t in (r.evidence, r.speed)]
    missing = [t for t in texts if t and ja.search(t) and t not in _EN]
    assert missing == []
    assert not any(ja.search(v) for v in _EN.values())
    i18n._set_cli_language("en")
    try:
        assert shown("supported_models_base.py:47(既定)") == "supported_models_base.py:47 (default)"
    finally:
        i18n._set_cli_language(None)
    assert shown("supported_models_base.py:47(既定)") == "supported_models_base.py:47(既定)"
