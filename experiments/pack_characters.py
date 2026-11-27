#!/usr/bin/env python3
"""
Pack the explorer characters for the viewer from Quaternius's free (CC0) kits.

Reads the unzipped "Universal Base Characters[Standard]", "Universal Animation
Library[Standard]" and "Modular Character Outfits - Fantasy[Standard]" downloads under
.vendor/quaternius/{ubc,ual,mco}/ and writes viewer/public/characters/: the two base bodies
(used for their heads), the hairstyles, the outfits, the animation clips the explorer uses, and
textures shrunk from 2048²-4096² PNG to 1024² JPEG (tens of MB of source per character otherwise).

    uv run python experiments/pack_characters.py

Needs macOS `sips` for the texture conversion. The output is tracked; rerun only to change it.
"""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from pathlib import Path

from paths import REPO_ROOT, main_checkout_root

OUT = REPO_ROOT / "viewer" / "public" / "characters"
TEX_SIZE = 1024

# Clips kept from the 43 in the free library (on foot, in water, and seated for vehicles)
CLIPS = [
    "Idle_Loop",
    "Walk_Loop",
    "Jog_Fwd_Loop",
    "Sprint_Loop",
    "Swim_Fwd_Loop",
    "Swim_Idle_Loop",
    "Jump_Start",
    "Jump_Loop",
    "Jump_Land",
    "Crouch_Idle_Loop",
    "Crouch_Fwd_Loop",
    "Push_Loop",
    "Sitting_Idle_Loop",
    "Driving_Loop",
]
BODIES = ["Superhero_Male_FullBody", "Superhero_Female_FullBody"]
OUTFITS = ["Male_Ranger", "Female_Ranger", "Male_Peasant", "Female_Peasant"]
HAIR = ["Hair_SimpleParted", "Hair_Buzzed", "Hair_Beard", "Hair_Long", "Hair_Buns", "Hair_BuzzedFemale"]
# Skin tones that are not referenced by the glTFs (they ship with the dark one): swapped in at runtime
EXTRA_TEXTURES = {"T_Superhero_Male_Ligh.png": "T_Superhero_Male_Light", "T_Superhero_Female_Light_BaseColor.png": "T_Superhero_Female_Light"}
# Alternative outfit colours, likewise swapped in at runtime
OUTFIT_TEXTURES = ["Ranger/T_Ranger_3_BaseColor.png", "Peasant/T_Peasant_2_BaseColor.png"]


def source_root() -> Path:
    for root in (REPO_ROOT, main_checkout_root()):
        cand = root / ".vendor" / "quaternius" if root else None
        if cand and all((cand / d).is_dir() for d in ("ubc", "ual", "mco")):
            return cand
    raise SystemExit("Unzip the three Quaternius [Standard] kits into .vendor/quaternius/{ubc,ual,mco} first")


def convert_texture(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["sips", "-Z", str(TEX_SIZE), "-s", "format", "jpeg", "-s", "formatOptions", "82", str(src), "--out", str(dst)],
        check=True,
        capture_output=True,
    )


def find_texture(dirs: list[Path], uri: str) -> Path:
    # Some glTFs name a texture "X_png.png" where the file on disk is "X.png"
    for name in (uri, uri.replace("_png.png", ".png")):
        for d in dirs:
            if (d / name).exists():
                return d / name
    raise FileNotFoundError(uri)


def pack_gltf(src: Path, tex_dirs: list[Path]) -> None:
    doc = json.loads(src.read_text())
    for img in doc.get("images", []):
        tex = find_texture(tex_dirs, img["uri"])
        stem = tex.stem
        convert_texture(tex, OUT / "tex" / f"{stem}.jpg")
        img["uri"] = f"tex/{stem}.jpg"
        img["mimeType"] = "image/jpeg"
    for buf in doc["buffers"]:
        shutil.copyfile(src.parent / buf["uri"], OUT / buf["uri"])
    (OUT / src.name).write_text(json.dumps(doc, separators=(",", ":")))


def prune_glb(src: Path, dst: Path, keep_clips: list[str]) -> None:
    """Copy a .glb keeping only the named animations, dropping the buffer data nothing else uses."""
    raw = src.read_bytes()
    json_len = struct.unpack_from("<I", raw, 12)[0]
    doc = json.loads(raw[20 : 20 + json_len])
    bin_start = 20 + json_len + 8
    blob = raw[bin_start : bin_start + struct.unpack_from("<I", raw, 20 + json_len)[0]]

    have = {a["name"] for a in doc["animations"]}
    missing = [c for c in keep_clips if c not in have]
    if missing:
        raise SystemExit(f"{src.name} has no clip(s) {missing}")
    doc["animations"] = [a for a in doc["animations"] if a["name"] in keep_clips]

    used: set[int] = set()
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            used.update(prim["attributes"].values())
            if "indices" in prim:
                used.add(prim["indices"])
            for target in prim.get("targets", []):
                used.update(target.values())
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            used.add(skin["inverseBindMatrices"])
    for anim in doc["animations"]:
        for s in anim["samplers"]:
            used.update((s["input"], s["output"]))

    acc_map = {old: new for new, old in enumerate(sorted(used))}
    accessors = [doc["accessors"][i] for i in sorted(used)]
    view_ids = sorted({a["bufferView"] for a in accessors} | {i["bufferView"] for i in doc.get("images", []) if "bufferView" in i})
    view_map = {old: new for new, old in enumerate(view_ids)}
    out = bytearray()
    views = []
    for old in view_ids:
        v = dict(doc["bufferViews"][old])
        start = v.get("byteOffset", 0)
        while len(out) % 4:
            out.append(0)
        v["byteOffset"] = len(out)
        out += blob[start : start + v["byteLength"]]
        views.append(v)
    while len(out) % 4:
        out.append(0)

    for a in accessors:
        a["bufferView"] = view_map[a["bufferView"]]
    for img in doc.get("images", []):
        if "bufferView" in img:
            img["bufferView"] = view_map[img["bufferView"]]
    for mesh in doc.get("meshes", []):
        for prim in mesh["primitives"]:
            prim["attributes"] = {k: acc_map[v] for k, v in prim["attributes"].items()}
            if "indices" in prim:
                prim["indices"] = acc_map[prim["indices"]]
            if "targets" in prim:
                prim["targets"] = [{k: acc_map[v] for k, v in t.items()} for t in prim["targets"]]
    for skin in doc.get("skins", []):
        if "inverseBindMatrices" in skin:
            skin["inverseBindMatrices"] = acc_map[skin["inverseBindMatrices"]]
    for anim in doc["animations"]:
        for s in anim["samplers"]:
            s["input"], s["output"] = acc_map[s["input"]], acc_map[s["output"]]
    doc["accessors"], doc["bufferViews"] = accessors, views
    doc["buffers"] = [{"byteLength": len(out)}]

    body = json.dumps(doc, separators=(",", ":")).encode()
    body += b" " * (-len(body) % 4)
    total = 12 + 8 + len(body) + 8 + len(out)
    dst.write_bytes(
        struct.pack("<4sII", b"glTF", 2, total) + struct.pack("<I4s", len(body), b"JSON") + body + struct.pack("<I4s", len(out), b"BIN\0") + bytes(out)
    )


def main() -> None:
    src = source_root()
    ubc = src / "ubc" / "Universal Base Characters[Standard]"
    ual = src / "ual" / "Universal Animation Library[Standard]"
    bodies = ubc / "Base Characters" / "Godot - UE"
    hair = ubc / "Hairstyles" / "Rigged to Head Bone" / "glTF (Godot -Unreal)"
    mco = src / "mco" / "Modular Character Outfits - Fantasy[Standard]"
    outfits = mco / "Exports" / "glTF (Godot-Unreal)" / "Outfits"
    tex_dirs = [bodies, hair, ubc / "Base Characters" / "Textures", outfits]
    OUT.mkdir(parents=True, exist_ok=True)

    for name in BODIES:
        pack_gltf(bodies / f"{name}.gltf", tex_dirs)
    for name in HAIR:
        pack_gltf(hair / f"{name}.gltf", tex_dirs)
    for name in OUTFITS:
        pack_gltf(outfits / f"{name}.gltf", tex_dirs)
    for rel in OUTFIT_TEXTURES:
        convert_texture(mco / "Textures" / rel, OUT / "tex" / f"{Path(rel).stem}.jpg")
    for file, stem in EXTRA_TEXTURES.items():
        convert_texture(ubc / "Base Characters" / "Textures" / file, OUT / "tex" / f"{stem}.jpg")
    prune_glb(ual / "Unreal-Godot" / "UAL1_Standard.glb", OUT / "animations.glb", CLIPS)
    (OUT / "LICENSE.txt").write_text(
        "Characters, hairstyles and animations by Quaternius (https://quaternius.com):\n"
        "Universal Base Characters, Modular Character Outfits - Fantasy and Universal Animation\n"
        "Library, free [Standard] versions.\n"
        "CC0 1.0 Universal (public domain): https://creativecommons.org/publicdomain/zero/1.0/\n"
        "Repacked by experiments/pack_characters.py (textures resized, unused clips removed).\n"
    )
    size = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"Wrote {OUT.relative_to(REPO_ROOT)} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
