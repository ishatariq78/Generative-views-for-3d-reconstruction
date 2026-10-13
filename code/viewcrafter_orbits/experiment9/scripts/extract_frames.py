#!/usr/bin/env python3
"""Hand all six three-view camera paths to VGGT, as conditions `p3_<path>`.

Joins experiment 3's inputs so the same scoring scripts run over them beside
gen360 (one view), v3_360 (three views, level orbit) and the rest. `p3_level`
should land on `v3_360`: same views, same trajectory file, same seed.
"""
import json, sys
from pathlib import Path
import imageio.v3 as iio
from PIL import Image
PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]
rows = []
for name in C.names():
    for path in PATHS:
        mp4 = PROJ / f"experiment9/outputs/{name}__{path}/diffusion0.mp4"
        if not mp4.exists():
            print(f"{name:22s} {path:7s} missing"); continue
        v = iio.imread(mp4, plugin="pyav")
        out = PROJ / "experiment3/inputs" / f"{name}__p3_{path}"
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.png"):
            f.unlink()
        for i, fr in enumerate(v):
            Image.fromarray(fr).save(out / f"{i:03d}.png")
        rows.append({"object": name, "condition": f"p3_{path}",
                     "frames": int(len(v)), "source": str(mp4)})
        print(f"{name:22s} {path:7s} {len(v)} frames")

rec = PROJ / "experiment3/inputs/manifest.json"
prev = json.loads(rec.read_text()) if rec.exists() else []
merged = {(m["object"], m["condition"]): m for m in prev}
merged.update({(m["object"], m["condition"]): m for m in rows})
rec.write_text(json.dumps(sorted(merged.values(), key=lambda m: (m["object"], m["condition"])), indent=2))
print(f"\nwrote {len(rows)} frame sets")
