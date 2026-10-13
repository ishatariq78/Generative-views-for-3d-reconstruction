#!/usr/bin/env python3
"""Convert KV-Tracker's open3d PLYs to npz, so the scoring env does not need open3d.

    ~/.conda/envs/kvtracker/bin/python scripts/kv_ply_to_npz.py

The tracker writes binary PLYs with double-precision xyz that only open3d reads correctly
here, and open3d lives in the kvtracker env while the scoring runs in shaper. One cheap
conversion keeps each env doing what it can.
"""
import json
from pathlib import Path
import numpy as np
import open3d as o3d

E = Path("/home/isha.tariq/seva_to_shaper")
names = [r["name"] for r in json.loads((E / "objects_kvtest.json").read_text())]
for obj in names:
    d = E / f"outputs_sub/kv/{obj}"
    plys = sorted(d.glob("pcd_*.ply"), key=lambda p: int(p.stem.split("_")[1]))
    if not plys:
        print(f"{obj}: no ply"); continue
    pc = o3d.io.read_point_cloud(str(plys[-1]))
    pts = np.asarray(pc.points, np.float32); col = np.asarray(pc.colors, np.float32)
    np.savez_compressed(d / "cloud.npz", points=pts, colors=col)
    print(f"{obj:32s} {len(pts):8d} points -> cloud.npz")
