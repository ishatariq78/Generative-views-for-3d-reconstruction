#!/usr/bin/env python3
"""Hand the 8-view 360 degree orbits to VGGT, as condition `mv360`.

Same 25 frames as every other condition -- these videos are already 25 frames, so
nothing is sampled away. They join experiment 3's inputs so the same scripts score
them beside gen360 (one view, same trajectory), sparse (eight views, the wearer's
arc) and real.
"""
import json
import shutil
import sys
from pathlib import Path

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

import imageio.v3 as iio
from PIL import Image

rows = []
for name in C.names():
    mp4 = PROJ / f"experiment7/outputs/{name}__orbit360/diffusion0.mp4"
    if not mp4.exists():
        print(f"{name:22s} no 8-view orbit yet"); continue
    v = iio.imread(mp4, plugin="pyav")
    out = PROJ / "experiment3/inputs" / f"{name}__mv360"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for i, fr in enumerate(v):
        Image.fromarray(fr).save(out / f"{i:03d}.png")
    rows.append({"object": name, "condition": "mv360", "frames": int(len(v)), "source": str(mp4)})
    print(f"{name:22s} {len(v)} frames")

rec = PROJ / "experiment3/inputs/manifest.json"
prev = json.loads(rec.read_text()) if rec.exists() else []
merged = {(m["object"], m["condition"]): m for m in prev}
merged.update({(m["object"], m["condition"]): m for m in rows})
rec.write_text(json.dumps(sorted(merged.values(), key=lambda m: (m["object"], m["condition"])), indent=2))
print(f"\nwrote {len(rows)} mv360 frame sets")
