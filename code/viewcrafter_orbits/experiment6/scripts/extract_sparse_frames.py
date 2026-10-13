#!/usr/bin/env python3
"""Sample 25 frames out of each sparse-mode video, for VGGT.

The sparse mode concatenates one 25-frame clip per consecutive input pair, so a
16-view object yields a 375-frame video. Feeding all of it to VGGT would make this
condition incomparable with the others, which are all 25 frames -- more input is
its own advantage and would confound the thing being tested. So the video is
sampled evenly down to 25.

Frames land in experiment3/inputs/<object>__sparse/, alongside the gen360, gen45
and real sets: experiment 3 owns "VGGT on a set of frames", and this is simply a
fourth source of frames. Experiment 6 owns making the video.

    python scripts/extract_sparse_frames.py
"""
import argparse
import json
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

N_FRAMES = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=PROJ / "experiment6" / "outputs")
    ap.add_argument("--dst", type=Path, default=PROJ / "experiment3" / "inputs")
    args = ap.parse_args()

    rows = []
    for name in C.names():
        mp4 = args.src / name / "diffusion.mp4"
        if not mp4.exists():
            print(f"{name:22s} no sparse video yet")
            continue
        v = iio.imread(mp4, plugin="pyav")
        idx = np.linspace(0, len(v) - 1, N_FRAMES).round().astype(int)
        out = args.dst / f"{name}__sparse"
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.png"):
            f.unlink()
        for i, k in enumerate(idx):
            Image.fromarray(v[k]).save(out / f"{i:03d}.png")
        rows.append({"object": name, "condition": "sparse", "frames": N_FRAMES,
                     "video_frames": int(len(v)), "source": str(mp4)})
        print(f"{name:22s} {len(v):4d} video frames -> {N_FRAMES} sampled")

    rec = args.dst / "manifest.json"
    prev = json.loads(rec.read_text()) if rec.exists() else []
    merged = {(m["object"], m["condition"]): m for m in prev}
    merged.update({(m["object"], m["condition"]): m for m in rows})
    rec.write_text(json.dumps(sorted(merged.values(), key=lambda m: (m["object"], m["condition"])), indent=2))
    print(f"\nwrote {len(rows)} sparse frame sets to {args.dst}")


if __name__ == "__main__":
    main()
