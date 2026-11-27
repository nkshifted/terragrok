#!/usr/bin/env python3
"""
Copy the vehicle models the explorer uses into viewer/public/vehicles/.

Sources (all CC0), unzipped or saved under .vendor/:
  .vendor/kenney/car/         Kenney "Car Kit"         https://kenney.nl/assets/car-kit
  .vendor/kenney/watercraft/  Kenney "Watercraft Kit"  https://kenney.nl/assets/watercraft-kit
  .vendor/quaternius/animals/Horse.glb
      Quaternius "Ultimate Animated Animal Pack" horse (https://quaternius.com); the single
      .glb is also on https://poly.pizza/m/qvTrSG9pZF

    uv run python experiments/pack_vehicles.py

The output is tracked; rerun only to change which models are used. Each Kenney kit keeps its own
Textures/colormap.png next to its models, because the .glb files refer to it by that path.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from paths import REPO_ROOT, main_checkout_root

OUT = REPO_ROOT / "viewer" / "public" / "vehicles"
KENNEY = {"car": ["suv.glb"], "watercraft": ["boat-speed-e.glb", "boat-sail-a.glb", "boat-row-small.glb"]}


def vendor_root() -> Path:
    for root in (REPO_ROOT, main_checkout_root()):
        cand = root / ".vendor" if root else None
        if cand and (cand / "kenney" / "car").is_dir() and (cand / "quaternius" / "animals" / "Horse.glb").exists():
            return cand
    raise SystemExit("Put the Kenney kits and the Quaternius horse under .vendor/ first (see this file's docstring)")


def main() -> None:
    vendor = vendor_root()
    for kit, models in KENNEY.items():
        src = vendor / "kenney" / kit / "Models" / "GLB format"
        dst = OUT / kit
        (dst / "Textures").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / "Textures" / "colormap.png", dst / "Textures" / "colormap.png")
        for name in models:
            shutil.copyfile(src / name, dst / name)
    (OUT / "horse").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vendor / "quaternius" / "animals" / "Horse.glb", OUT / "horse" / "horse.glb")
    (OUT / "LICENSE.txt").write_text(
        "car/ and watercraft/: models from Kenney's Car Kit and Watercraft Kit (https://kenney.nl).\n"
        "horse/: the horse from Quaternius's Ultimate Animated Animal Pack (https://quaternius.com).\n"
        "All CC0 1.0 Universal (public domain): https://creativecommons.org/publicdomain/zero/1.0/\n"
    )
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} ({size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
