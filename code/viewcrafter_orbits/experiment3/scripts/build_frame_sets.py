#!/usr/bin/env python3
"""Assemble the image sets that go into VGGT.

Three conditions per object, all 25 frames, all cut out the same way, so the only
thing that differs between them is where the views came from:

  gen360   the 25 frames of experiment 2's cutout 360-degree video
  gen45    the 25 frames of experiment 2's cutout +/-45-degree video
  real     25 *real* Aria RGB frames of the same object, spread around it by
           camera azimuth and cut out with the same mesh-rasterised silhouette

`real` is the control, and it is the reason the experiment can be read at all.
Without it, a poor point cloud from `gen360` could equally mean "the generated
video is geometrically inconsistent" or "VGGT cannot do this object from 25
views". With it, the two are separable: `real` is the ceiling VGGT can reach on
this object from this many views, and the generated conditions are measured
against that rather than against perfection.

The real frames are chosen by binning the camera positions by azimuth around the
object and taking the best-scoring frame in each bin, so they cover the object the
way an orbit would instead of clustering wherever the wearer happened to linger.

    python scripts/build_frame_sets.py                 # all three objects
    python scripts/build_frame_sets.py --only ADT1292__chair
"""
import argparse
import io
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment1/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment2/scripts")

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import imageio.v3 as iio  # noqa: E402
import trimesh            # noqa: E402

# Frame ranking, rectification and framing live in experiment 1's module; the
# silhouette rasteriser and the compositing live in experiment 2's. Importing
# rather than copying keeps all three experiments looking at the same frames of
# the same captures through the same code.
from extract_reference_views import (  # noqa: E402
    SHAPER_DATA, bbox_of_mask, point_mask, rank_raw_frames, rectify_one, score_rect,
)
# crop_window comes from experiment 2, not experiment 1: the two differ (1.45 vs
# 0.66 fill), and the `real` frames have to be framed exactly like the cutouts the
# generated videos came from, or the comparison is between crops, not conditions.
from build_object_only_views import (  # noqa: E402
    crop_window, feather, make_cutout, render_silhouette, rigid,
)

sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as OBJECTS_CFG  # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
EXP2 = PROJ / "experiment2" / "outputs"
OUT_W, OUT_H = 1024, 576
N_FRAMES = 25
AZIMUTH_BINS = 12
OBJECTS = OBJECTS_CFG.names()


# ---------------------------------------------------------------- generated
def frames_from_video(mp4, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    v = iio.imread(mp4, plugin="pyav")
    for i, fr in enumerate(v):
        Image.fromarray(fr).save(out_dir / f"{i:03d}.png")
    return len(v)


# --------------------------------------------------------------------- real
def camera_azimuth(T_cam_model):
    """Where the camera sits around the object, in degrees. T is camera-from-model
    (established in experiment 2), so the camera's position in model coordinates
    is -R^T t."""
    T = np.asarray(T_cam_model, dtype=np.float64)
    pos = -T[:3, :3].T @ T[:3, 3]
    return float(np.degrees(np.arctan2(pos[1], pos[0])) % 360.0)


def build_real_set(name, out_dir):
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
    cams, poses = d["rgb_camera_params"], d["Ts_rgbCamera_model"]
    rotate_cw = not bool(d["is_ariagen2"])
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    W0, H0 = Image.open(io.BytesIO(frames[0])).size

    # Bin the usable frames by where the camera stood, then keep the best in each
    # bin: coverage around the object matters more here than any single frame.
    ranked = rank_raw_frames(projs, W0, H0)
    bins = {}
    for score, idx, _, npts in ranked:
        b = int(camera_azimuth(poses[idx].numpy()) // (360 / AZIMUTH_BINS))
        bins.setdefault(b, []).append((score, idx))
    picked = []
    per_bin = max(1, N_FRAMES // max(1, len(bins)))
    for b in sorted(bins):
        picked += [i for _, i in sorted(bins[b], reverse=True)[:per_bin]]
    # Top up from the global ranking if the bins did not fill the quota.
    for _, idx, _, _ in ranked:
        if len(picked) >= N_FRAMES * 2:
            break
        if idx not in picked:
            picked.append(idx)

    out_dir.mkdir(parents=True, exist_ok=True)
    kept, meta = 0, []
    for idx in picked:
        if kept >= N_FRAMES:
            break
        rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
        pts = np.asarray(projs[idx].numpy(), np.float32).reshape(-1, 2)
        rect, rmask, K = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
        H, W = rect.shape[:2]
        T = np.asarray(poses[idx].numpy(), np.float64)
        try:
            sil = render_silhouette(mesh, K, W, H, rigid(np.linalg.inv(T)))
        except Exception:
            continue
        if sil.sum() < 2000:
            continue
        if rotate_cw:
            rect, sil = np.rot90(rect, 3).copy(), np.rot90(sil, 3).copy()
            H, W = rect.shape[:2]
        box = bbox_of_mask(sil, W, H)
        if box is None or score_rect(box, W, H) < 0:
            continue
        img = make_cutout(rect, feather(sil))
        win = crop_window(box, W, H)
        Image.fromarray(img).crop(win).resize((OUT_W, OUT_H), Image.LANCZOS) \
             .save(out_dir / f"{kept:03d}.png")
        meta.append({"frame": int(idx), "azimuth_deg": round(camera_azimuth(T), 1)})
        kept += 1
    return kept, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out_dir", type=Path, default=PROJ / "experiment3" / "inputs")
    args = ap.parse_args()
    objs = [o for o in OBJECTS if args.only is None or o in args.only]

    manifest = []
    for name in objs:
        for cond, mp4 in (("gen360", EXP2 / f"{name}__cutout__orbit360" / "diffusion0.mp4"),
                          ("gen45", EXP2 / f"{name}__cutout__orbit45" / "diffusion0.mp4")):
            n = frames_from_video(mp4, args.out_dir / f"{name}__{cond}")
            print(f"{name:20s} {cond:7s} {n:3d} frames  <- {mp4.parent.name}")
            manifest.append({"object": name, "condition": cond, "frames": n,
                             "source": str(mp4)})
        n, meta = build_real_set(name, args.out_dir / f"{name}__real")
        az = sorted({m["azimuth_deg"] // 30 * 30 for m in meta})
        print(f"{name:20s} {'real':7s} {n:3d} frames  azimuth bins covered: "
              f"{len(az)}/{AZIMUTH_BINS}")
        manifest.append({"object": name, "condition": "real", "frames": n,
                         "source": "aria rgb, mesh-rasterised cutout", "frames_meta": meta})

    # Merge, so adding objects later does not erase the record of the earlier sets.
    rec = args.out_dir / "manifest.json"
    prev = json.loads(rec.read_text()) if rec.exists() else []
    merged = {(m["object"], m["condition"]): m for m in prev}
    merged.update({(m["object"], m["condition"]): m for m in manifest})
    rec.write_text(json.dumps(sorted(merged.values(),
                                     key=lambda m: (m["object"], m["condition"])), indent=2))
    print(f"\nwrote {len(manifest)} frame sets to {args.out_dir}")


if __name__ == "__main__":
    main()
