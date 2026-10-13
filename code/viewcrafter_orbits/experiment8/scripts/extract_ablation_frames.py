#!/usr/bin/env python3
"""Hand the guessed-elevation ablation orbits to VGGT, as condition `v3_360g`.

Same three views and the same 360-degree trajectory as `v3_360`; the only
difference is that the orbit runs at the per-object elevation that earlier
experiments eyeballed off the reference image, rather than the one measured from
the input cameras. Comparing the two isolates which of the two simultaneous
changes -- three views, or a measured elevation -- produced the gain.
"""
import json
import sys
from pathlib import Path

import imageio.v3 as iio
from PIL import Image

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

rows = []
for name in C.names():
    mp4 = PROJ / f"experiment8/outputs/{name}__orbit360_guessedelev/diffusion0.mp4"
    if not mp4.exists():
        print(f"{name:22s} missing"); continue
    v = iio.imread(mp4, plugin="pyav")
    out = PROJ / "experiment3/inputs" / f"{name}__v3_360g"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for i, fr in enumerate(v):
        Image.fromarray(fr).save(out / f"{i:03d}.png")
    rows.append({"object": name, "condition": "v3_360g", "frames": int(len(v)), "source": str(mp4)})
    print(f"{name:22s} {len(v)} frames")

rec = PROJ / "experiment3/inputs/manifest.json"
prev = json.loads(rec.read_text()) if rec.exists() else []
merged = {(m["object"], m["condition"]): m for m in prev}
merged.update({(m["object"], m["condition"]): m for m in rows})
rec.write_text(json.dumps(sorted(merged.values(), key=lambda m: (m["object"], m["condition"])), indent=2))
print(f"\nwrote {len(rows)} v3_360g frame sets")
