"""端末向けの表。#1 では素朴な表、#7 で見やすくする。"""

from __future__ import annotations


def print_fit_table(fits, title: str = "") -> None:
    if title:
        print(f"=== {title} ({len(fits)}) ===")
    if not fits:
        print("(該当なし)")
        return
    print(f"{'fit':<10} {'model':<44} {'type':<6} {'VRAM GB':>8} {'basis':<9} {'license':<12}")
    for f in fits:
        e, m = f.estimate, f.model
        lic = (m.license or "?")[:12]
        print(f"{e.fit:<10} {m.id[:44]:<44} {m.modality:<6} {e.vram_gb:>8.1f} {e.basis:<9} {lic:<12}")
