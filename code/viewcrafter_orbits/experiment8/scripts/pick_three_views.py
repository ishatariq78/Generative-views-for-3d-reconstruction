#!/usr/bin/env python3
"""Choose three views per object, and measure the camera angles they were taken from.

Three is ViewCrafter's actual operating point: `sparse_view_interp` ships with 2-3
image examples and its code branches specially on `len(images) == 2`. Eight views
was already past what the point rasteriser holds. So this is the mode used as
designed.

The three are taken from the ends and middle of the eight-view arc built in
experiment 6, so they span the widest baseline the capture offers while staying in
arc order (the trajectory anchor and the interpolators both assume that order).

It also records, for each view, where the camera actually was relative to the
object:

    azimuth     angle around the object in its own ground plane
    elevation   angle above that plane

The mean elevation is what the 360-degree orbit is then run at, so the turn
happens **on the same level the object was actually photographed from** rather
than at a hand-picked tilt. That is a real change from every earlier experiment,
where elevation was a per-object guess read off the reference image.
"""
import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
VIEWS8 = PROJ / "experiment6" / "inputs8"
VIEWS16 = PROJ / "experiment6" / "inputs"     # carries the source frame indices
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

N_PICK = 3


def camera_pos(T_cam_model):
    T = np.asarray(T_cam_model, dtype=np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def angles(pos, centre):
    """Azimuth around the object, and elevation above its ground plane.
    Model coordinates are z-up, which is why z is the one separated out."""
    v = pos - centre
    az = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    el = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    return az, el


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "experiment8" / "inputs")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recs = []
    for name in C.names():
        # inputs8 records each view's azimuth but not which capture frame it came
        # from; inputs (16 views) records both. Match one to the other by azimuth,
        # which is exact because inputs8 was copied from it.
        meta8 = json.loads((VIEWS8 / f"{name}.json").read_text())
        meta16 = json.loads((VIEWS16 / f"{name}.json").read_text())
        by_az = {round(f["azimuth_deg"], 1): f["rgb_frame_index"] for f in meta16["frames"]}
        files = sorted((VIEWS8 / name).glob("*.png"))
        if len(files) < N_PICK:
            print(f"{name}: only {len(files)} views"); continue
        take = np.linspace(0, len(files) - 1, N_PICK).round().astype(int)
        take = sorted(set(take.tolist()))

        d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        centre = mesh.bounds.mean(axis=0)
        poses = d["Ts_rgbCamera_model"]

        out = args.out_dir / name
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.png"):
            f.unlink()

        views = []
        for i, k in enumerate(take):
            shutil.copy(files[k], out / f"{i:03d}.png")
            src_idx = by_az[round(meta8["azimuths"][k], 1)]
            az, el = angles(camera_pos(poses[src_idx].numpy()), centre)
            views.append({"file": f"{i:03d}.png", "rgb_frame_index": int(src_idx),
                          "azimuth_deg": round(az, 1), "elevation_deg": round(el, 1)})

        az = [v["azimuth_deg"] for v in views]
        steps = [min(abs(az[i + 1] - az[i]), 360 - abs(az[i + 1] - az[i])) for i in range(len(az) - 1)]
        el_mean = float(np.mean([v["elevation_deg"] for v in views]))
        rec = {"object": name, "views": len(views), "camera_angles": views,
               "azimuth_span_deg": round(sum(steps), 1),
               "mean_elevation_deg": round(el_mean, 1),
               "elevation_range_deg": [round(min(v["elevation_deg"] for v in views), 1),
                                       round(max(v["elevation_deg"] for v in views), 1)]}
        (args.out_dir / f"{name}.json").write_text(json.dumps(rec, indent=2))
        recs.append(rec)
        az_txt = "  ".join("%6.1f" % v["azimuth_deg"] for v in views)
        el_txt = "  ".join("%5.1f" % v["elevation_deg"] for v in views)
        print(f"{name:22s} az {az_txt}   el {el_txt}   -> orbit elevation {el_mean:5.1f}°")

    (args.out_dir / "manifest.json").write_text(json.dumps(recs, indent=2))
    (args.out_dir / "elevations.json").write_text(
        json.dumps({r["object"]: r["mean_elevation_deg"] for r in recs}, indent=2))
    print(f"\nwrote {len(recs)} three-view sets to {args.out_dir}")


if __name__ == "__main__":
    main()
