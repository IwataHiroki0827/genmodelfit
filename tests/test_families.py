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
