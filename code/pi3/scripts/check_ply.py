"""Verify Pi3 outputs: load each .ply, report point count, colour, and bounds.

    python scripts/check_ply.py outputs/*.ply
"""
import sys
import numpy as np
from plyfile import PlyData

ok = True
for path in sys.argv[1:]:
    try:
        v = PlyData.read(path)["vertex"]
        xyz = np.stack([v["x"], v["y"], v["z"]], 1)
        has_rgb = all(c in v.data.dtype.names for c in ("red", "green", "blue"))
        finite = np.isfinite(xyz).all()
        lo, hi = xyz.min(0), xyz.max(0)
        print(f"{path}: {len(xyz):,} pts, rgb={has_rgb}, finite={finite}, "
              f"extent={np.round(hi - lo, 3).tolist()}")
        ok &= len(xyz) > 0 and finite
    except Exception as e:
        print(f"{path}: FAILED to read ({e})")
        ok = False
print("ALL POINT CLOUDS OK" if ok else "SOME POINT CLOUDS BAD")
sys.exit(0 if ok else 1)
