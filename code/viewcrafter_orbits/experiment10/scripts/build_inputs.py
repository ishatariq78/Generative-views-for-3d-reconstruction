#!/usr/bin/env python3
"""Rebuild the multi-view input sets with the arc ordering rule done properly.

Experiment 6's builder sorts views by azimuth and then rotates the sequence to
start just after the largest gap, so the interpolator never walks across the
unobserved side. Its gap search only looked at gaps *between* consecutive sorted
views and never at the wrap-around from the last view back to the first. That is
fine when the arc crosses 0 deg (the wrap then shows up as a huge linear gap), but
for an arc that does not, any internal hole wider than 45 deg is mistaken for the
arc's end. The wooden table (45-64 deg and 136-149 deg, a 72 deg hole, nothing
near 0) was rotated to start at 135.6 deg, putting a 104 deg jump backwards over
already-covered ground in the middle of the sequence, and every downstream
experiment inherited that order: experiment 7's anchor view, experiment 8's three
views and their anchor, experiment 9's six paths.

This script applies the circular rule -- largest gap including the wrap -- to all
five objects, re-uses experiment 6's cut-out PNGs (same frames, only the order
changes), and asserts that only the wooden table differs. It then picks the three
views the way experiment 8 does (first / middle / last of the eight) and measures
their elevation the same way, so experiment 10 differs from 6-9 in exactly one
thing.

    python scripts/build_inputs.py
"""
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import trimesh

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
E6, E8, E10 = PROJ / "experiment6", PROJ / "experiment8", PROJ / "experiment10"
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C  # noqa: E402

N_PICK = 3


def camera_pos(T_cam_model):
    T = np.asarray(T_cam_model, dtype=np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def angles(pos, centre):
    v = pos - centre
    az = float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)
    el = float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))
    return az, el


def circular_order(azimuths):
    """Indices sorted by azimuth, rotated to start just after the largest gap,
    where the gap from the last view round to the first counts too."""
    order = list(np.argsort(azimuths))
    az = [azimuths[i] for i in order]
    gaps = [(az[(i + 1) % len(az)] - az[i]) % 360.0 for i in range(len(az))]
    cut = (int(np.argmax(gaps)) + 1) % len(az)
    return order[cut:] + order[:cut], max(gaps)


def circular_steps(az):
    return [min(abs(az[i + 1] - az[i]), 360 - abs(az[i + 1] - az[i])) for i in range(len(az) - 1)]


def main():
    changed = []
    recs8, recs3 = [], []
    for name in C.names():
        meta8 = json.loads((E6 / "inputs8" / f"{name}.json").read_text())
        meta16 = json.loads((E6 / "inputs" / f"{name}.json").read_text())
        by_az = {round(f["azimuth_deg"], 1): f["rgb_frame_index"] for f in meta16["frames"]}
        old_az = meta8["azimuths"]
        old_files = sorted((E6 / "inputs8" / name).glob("*.png"))
        assert len(old_files) == len(old_az) == 8, name

        order, biggest_gap = circular_order(old_az)
        new_az = [old_az[i] for i in order]
        same = new_az == old_az
        if not same:
            changed.append(name)

        d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        centre = mesh.bounds.mean(axis=0)
        poses = d["Ts_rgbCamera_model"]

        # --- eight views, corrected order
        out8 = E10 / "inputs8" / name
        out8.mkdir(parents=True, exist_ok=True)
        for f in out8.glob("*.png"):
            f.unlink()
        frames8 = []
        for i, k in enumerate(order):
            shutil.copy(old_files[k], out8 / f"{i:03d}.png")
            src = by_az[round(old_az[k], 1)]
            az, el = angles(camera_pos(poses[src].numpy()), centre)
            frames8.append({"file": f"{i:03d}.png", "from_exp6_file": old_files[k].name,
                            "rgb_frame_index": int(src), "azimuth_deg": round(az, 1),
                            "elevation_deg": round(el, 1)})
        steps = circular_steps(new_az)
        rec8 = {"object": name, "views": 8, "order_changed_vs_exp6": not same,
                "azimuths": [round(a, 1) for a in new_az],
                "azimuths_exp6": old_az,
                "arc_extent_deg": round(360.0 - biggest_gap, 1),
                "largest_step_deg": round(max(steps), 1), "frames": frames8}
        (E10 / "inputs8" / f"{name}.json").write_text(json.dumps(rec8, indent=2))
        recs8.append(rec8)

        # --- three views, experiment 8's rule on the corrected order
        take = sorted(set(np.linspace(0, 7, N_PICK).round().astype(int).tolist()))
        out3 = E10 / "inputs3" / name
        out3.mkdir(parents=True, exist_ok=True)
        for f in out3.glob("*.png"):
            f.unlink()
        views = []
        for i, k in enumerate(take):
            shutil.copy(out8 / f"{k:03d}.png", out3 / f"{i:03d}.png")
            v = dict(frames8[k]); v["file"] = f"{i:03d}.png"; v["from_inputs8_file"] = f"{k:03d}.png"
            views.append(v)
        az3 = [v["azimuth_deg"] for v in views]
        el_mean = float(np.mean([v["elevation_deg"] for v in views]))
        rec3 = {"object": name, "views": 3, "camera_angles": views,
                "azimuth_span_deg": round(sum(circular_steps(az3)), 1),
                "anchor_file": "001.png", "anchor_azimuth_deg": az3[1],
                "mean_elevation_deg": round(el_mean, 1),
                "elevation_range_deg": [round(min(v["elevation_deg"] for v in views), 1),
                                        round(max(v["elevation_deg"] for v in views), 1)]}
        (E10 / "inputs3" / f"{name}.json").write_text(json.dumps(rec3, indent=2))
        recs3.append(rec3)

        flag = "  <-- ORDER CHANGED" if not same else ""
        print(f"{name:22s} 8 views: {[round(a,1) for a in new_az]}{flag}")
        print(f"{'':22s} 3 views: {az3}  anchor {az3[1]}°  elev {el_mean:.1f}°"
              f"  (exp8 had {json.loads((E8 / 'inputs' / f'{name}.json').read_text())['mean_elevation_deg']}°)")

    (E10 / "inputs8" / "manifest.json").write_text(json.dumps(recs8, indent=2))
    (E10 / "inputs3" / "manifest.json").write_text(json.dumps(recs3, indent=2))
    (E10 / "inputs3" / "elevations.json").write_text(
        json.dumps({r["object"]: r["mean_elevation_deg"] for r in recs3}, indent=2))
    print(f"\norder changed for: {changed}")
    assert changed == ["ADT1292__woodentable"], changed


if __name__ == "__main__":
    main()
