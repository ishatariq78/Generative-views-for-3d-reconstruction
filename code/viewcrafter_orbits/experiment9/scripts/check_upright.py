#!/usr/bin/env python3
"""Fit-independent check that the SLAM alignment puts each cloud the right way up.

The fit is a similarity transform found by search; nothing in it knows which way is
up. But VGGT does: it returns the 25 camera poses along with the points, and the
orbit's cameras sit *above* the object -- so the normal of the ring of camera
positions, pointed away from the cloud's centroid, is the cloud's own "up". The
reference frame is z-up. Push that vector through the fitted rotation: +1 means
upright, -1 means the fit landed the cloud on its head.

Recovered from the saved aligned cloud rather than by re-fitting: the aligned cloud
is the deterministic subsample of the raw one in the same order, so a Kabsch fit
between the two gives back the rotation exactly.

    python scripts/check_upright.py [--fit old|new]
"""
import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "experiment3" / "scripts"))
from align_and_score import load_ply   # noqa: E402
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C                    # noqa: E402

EXP3, EXP4 = PROJ / "experiment3/outputs", PROJ / "experiment4/outputs"
OBJ = C.names()
CONDS = ["gen360", "gen45", "real", "mv360", "v3_360", "v3_360g"] + \
        ["p3_" + p for p in ("level", "high", "low", "helix", "wobble", "dolly")]
MAX_VGGT = 120000


def kabsch_rotation(src, dst):
    """Rotation part of the similarity taking src onto dst (same order)."""
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", choices=["old", "so3", "new"], default="new")
    args = ap.parse_args()
    npy_dir = {"old": EXP4 / "oldfit" / "npy", "so3": EXP4 / "so3fit" / "npy", "new": EXP4}[args.fit]
    rows = []
    print(f"{'object':14s} " + " ".join(f"{c:>8s}" for c in CONDS))
    for obj in OBJ:
        line = []
        for cond in CONDS:
            d = EXP3 / f"{obj}__{cond}"
            f = npy_dir / f"{obj}__{cond}__aligned_to_slam.npy"
            if not (d / "points_object.ply").exists() or not f.exists():
                line.append("   --   "); continue
            pts = load_ply(d / "points_object.ply")
            z = np.load(d / "vggt.npz")
            E = z["extrinsic"]
            cams = np.stack([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(E))])
            ctr = cams.mean(0)
            n = np.linalg.svd(cams - ctr)[2][-1]
            n *= np.sign((ctr - pts.mean(0)) @ n)          # away from the object
            if len(pts) > MAX_VGGT:
                sub = np.random.default_rng(zlib.crc32(np.ascontiguousarray(pts)))
                pts = pts[sub.choice(len(pts), MAX_VGGT, replace=False)]
            aligned = np.load(f).astype(np.float64)
            if len(aligned) != len(pts):
                line.append("  n?    "); continue
            R = kabsch_rotation(pts, aligned)
            up_z = float((R @ n)[2])
            rows.append({"object": obj, "condition": cond, "fit": args.fit, "up_z": round(up_z, 3)})
            line.append(f"{up_z:+8.2f}")
        print(f"{obj.split('__')[1]:14s} " + " ".join(line))
    out = PROJ / "experiment9/outputs" / f"upright_{args.fit}fit.json"
    out.write_text(json.dumps(rows, indent=2))
    flipped = [r for r in rows if r["up_z"] < 0]
    print(f"\n{len(flipped)} of {len(rows)} fits have up -> -z (flipped)  [{args.fit} fit]")
    for r in flipped:
        print(f"   {r['object'].split('__')[1]:12s} {r['condition']:8s} {r['up_z']:+.2f}")


if __name__ == "__main__":
    main()
