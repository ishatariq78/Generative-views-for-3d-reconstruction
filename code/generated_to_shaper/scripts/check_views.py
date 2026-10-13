#!/usr/bin/env python3
"""Dump exactly what ShapeR's loader feeds the model, for a pkl.

    ~/.conda/envs/shaper/bin/python scripts/check_views.py <obj>__<cond> [...]

The camera chain is only trustworthy if the object lands where the point mask says it
does, so this renders `images` with `masks_ingest` overlaid -- the same discipline as
exp1's reprojection check. figs/ingest__<name>.jpg
"""
import sys, os
from pathlib import Path
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
REPO = "/home/isha.tariq/shapeR_testing/ShapeR"
os.chdir(REPO); sys.path.insert(0, REPO)
import omegaconf
from PIL import Image
from dataset.shaper_dataset import InferenceDataset

E = Path("/home/isha.tariq/seva_to_shaper")
cfg = omegaconf.OmegaConf.load(f"{REPO}/checkpoints/config.yaml")
for name in sys.argv[1:]:
    ds = InferenceDataset(cfg, paths=[str(E / f"outputs/pkls/{name}.pkl")], override_num_views=16)
    s = ds[0]
    imgs, msks = s["images"], s["masks_ingest"]
    print(f"{name}: images {imgs.shape} range [{imgs.min():.2f},{imgs.max():.2f}]  masks {msks.shape} "
          f"mask coverage {[round(float(m.mean()),3) for m in msks[:6]]}")
    n = min(8, len(imgs)); tiles = []
    for i in range(n):
        im = imgs[i]
        im = im.transpose(1, 2, 0) if im.ndim == 3 and im.shape[0] <= 4 else im
        im = np.squeeze(im)
        if im.ndim == 2: im = np.repeat(im[..., None], 3, -1)
        im = im[..., :3]
        lo, hi = float(im.min()), float(im.max())
        rgb = ((im - lo) / max(hi - lo, 1e-6) * 255).astype(np.uint8).copy()
        m = np.squeeze(msks[i]) > (msks[i].max() * 0.5 if msks[i].max() > 0 else 1e9)
        rgb[m] = (0.45 * rgb[m] + 0.55 * np.array([255, 60, 60])).astype(np.uint8)
        tiles.append(np.asarray(Image.fromarray(rgb).resize((224, 224))))
    sheet = np.concatenate(tiles, 1)
    Image.fromarray(sheet).save(E / f"figs/ingest__{name}.jpg", quality=88)
    print("   ->", E / f"figs/ingest__{name}.jpg")
