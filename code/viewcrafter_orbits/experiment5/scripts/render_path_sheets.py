#!/usr/bin/env python3
"""One contact sheet per object: all six camera paths, four frames each.

Rows are the paths, columns are frames sampled across the turn. Same object, same
reference image, same seed in every row -- so anything that differs down a column
is the camera path and nothing else.

    python scripts/render_path_sheets.py
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
from PIL import Image, ImageDraw

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment5")
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]
FRAMES = [0, 6, 12, 18, 24]
TW, TH, LAB = 300, 169, 22


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    args = ap.parse_args()
    meas = {(r["object"], r["path"]): r
            for r in json.loads((args.out_dir / "measurements.json").read_text())}
    objects = sorted({o for o, _ in meas})

    for obj in objects:
        sheet = Image.new("RGB", (TW * len(FRAMES), (TH + LAB) * len(PATHS)), (250, 250, 252))
        dr = ImageDraw.Draw(sheet)
        for i, p in enumerate(PATHS):
            d = args.out_dir / f"{obj}__{p}"
            m = meas.get((obj, p))
            if not (d / "diffusion0.mp4").exists() or m is None:
                continue
            v = iio.imread(d / "diffusion0.mp4", plugin="pyav")
            dr.text((8, i * (TH + LAB) + 6),
                    "%s   worst coverage %.3f   peak subject %.2fx   %d of 25 frames over 1.5x"
                    % (p, m["coverage_min"], m["subject_peak_ratio"], m["frames_over_1_5x"]),
                    fill=(40, 46, 56))
            for j, f in enumerate(FRAMES):
                sheet.paste(Image.fromarray(v[f]).resize((TW, TH), Image.LANCZOS),
                            (TW * j, i * (TH + LAB) + LAB))
        out = args.out_dir / f"PATHS__{obj}.jpg"
        sheet.save(out, quality=84)
        print(f"{obj:20s} -> {out.name}")


if __name__ == "__main__":
    main()
