#!/usr/bin/env python3
"""Frames for VGGT out of every experiment-10 video, into frames/<obj>__<cond>/.
Orbit runs: all 25 frames of diffusion0.mp4 (as experiments 7-9 did).
Sparse run: diffusion.mp4 sampled evenly down to 25 (as experiment 6 did).
"""
import json
from pathlib import Path
import imageio.v3 as iio
import numpy as np
from PIL import Image

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment10")
rows = []
for d in sorted((PROJ / "outputs").iterdir()):
    if not d.is_dir():
        continue
    if (d / "diffusion0.mp4").exists():
        mp4, v = d / "diffusion0.mp4", iio.imread(d / "diffusion0.mp4", plugin="pyav")
        idx = np.arange(len(v))
    elif (d / "diffusion.mp4").exists():
        mp4, v = d / "diffusion.mp4", iio.imread(d / "diffusion.mp4", plugin="pyav")
        idx = np.linspace(0, len(v) - 1, 25).round().astype(int)
    else:
        print(f"{d.name}: no video yet"); continue
    out = PROJ / "frames" / d.name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    for i, k in enumerate(idx):
        Image.fromarray(v[k]).save(out / f"{i:03d}.png")
    obj, cond = d.name.rsplit("__", 1)
    rows.append({"object": obj, "condition": cond, "frames": int(len(idx)),
                 "video_frames": int(len(v)), "source": str(mp4)})
    print(f"{d.name:36s} {len(v):4d} video frames -> {len(idx)}")
(PROJ / "frames" / "manifest.json").write_text(json.dumps(rows, indent=2))
print(f"wrote {len(rows)} frame sets")
