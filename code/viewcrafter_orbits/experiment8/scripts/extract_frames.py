#!/usr/bin/env python3
"""Hand the three-view 360 degree orbits to VGGT, as condition `v3_360`.

Joins experiment 3's inputs so the same scripts score them beside gen360 (one
view), mv360 (eight views, same trajectory), sparse and real.
"""
import json, sys
from pathlib import Path
import imageio.v3 as iio
from PIL import Image
PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402
rows = []
for name in C.names():
    mp4 = PROJ / f"experiment8/outputs/{name}__orbit360/diffusion0.mp4"
    if not mp4.exists():
        print(f"{name:22s} missing"); continue
    v = iio.imread(mp4, plugin="pyav")
    out = PROJ / "experiment3/inputs" / f"{name}__v3_360"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for i, fr in enumerate(v):
        Image.fromarray(fr).save(out / f"{i:03d}.png")
    rows.append({"object": name, "condition": "v3_360", "frames": int(len(v)), "source": str(mp4)})
    print(f"{name:22s} {len(v)} frames")
rec = PROJ / "experiment3/inputs/manifest.json"
prev = json.loads(rec.read_text()) if rec.exists() else []
merged = {(m["object"], m["condition"]): m for m in prev}
merged.update({(m["object"], m["condition"]): m for m in rows})
rec.write_text(json.dumps(sorted(merged.values(), key=lambda m: (m["object"], m["condition"])), indent=2))
print(f"\nwrote {len(rows)} v3_360 frame sets")
