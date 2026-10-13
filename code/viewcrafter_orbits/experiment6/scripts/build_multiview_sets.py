#!/usr/bin/env python3
"""Cut a whole set of views of one object out of a ShapeR capture.

Every earlier experiment gave ViewCrafter a single photograph. ShapeR itself does
not work that way -- at its `balance` preset it ingests 16 views of the object --
and ViewCrafter has a mode for that too (`sparse_view_interp`). This builds the
input for it: many views of the same object, each with the background removed by
the same mesh-rasterised silhouette, on the same neutral backdrop.

Two choices that matter, both forced by how the sparse mode works:

  **How many.** 16, matching the number ShapeR itself takes. The mode renders one
  25-frame clip per *consecutive pair*, so N views cost N-1 clips of diffusion;
  and DUSt3R matches a complete pair graph, which is O(N^2). 16 is the number
  ShapeR chose for the same trade, so it is the defensible one here.

  **What order.** Sorted by camera azimuth around the object. `generate_traj_interp`
  interpolates between the input views' *own* estimated poses in file order, so
  arbitrary ordering produces a camera that zigzags across the object instead of
  travelling around it. Sorting by azimuth turns the same views into a smooth arc.

That second point is also the honest limit of this experiment: the resulting video
follows the path the wearer actually walked. These captures cover 4-7 of 12
azimuth bins, so the output is a wide arc, not a full turn -- and no setting
changes that, because the views simply do not exist.

    python scripts/build_multiview_sets.py
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
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh  # noqa: E402

import objects as C  # noqa: E402
from extract_reference_views import (  # noqa: E402
    SHAPER_DATA, bbox_of_mask, point_mask, rank_raw_frames, rectify_one, score_rect,
)
from build_object_only_views import (  # noqa: E402
    MIN_CAM_DIST, camera_distance, crop_window, feather, make_cutout,
    render_silhouette, rigid,
)

OUT_W, OUT_H = 1024, 576
N_VIEWS = 16           # what ShapeR itself ingests at --config balance
N_SCAN = 260           # ranked frames examined before giving up
MIN_SEP_DEG = 4.0      # nearly-identical viewpoints add cost and no information


def camera_pos(T_cam_model):
    T = np.asarray(T_cam_model, dtype=np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def azimuth(pos, centre):
    v = pos - centre
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)


def process(name, out_root):
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
    cams, poses = d["rgb_camera_params"], d["Ts_rgbCamera_model"]
    rotate_cw = not bool(d["is_ariagen2"])
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    centre = mesh.bounds.mean(axis=0)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    W0, H0 = Image.open(io.BytesIO(frames[0])).size

    kept = []
    for _, idx, _, npts in rank_raw_frames(projs, W0, H0)[:N_SCAN]:
        if len(kept) >= N_VIEWS * 2:
            break
        T = np.asarray(poses[idx].numpy(), np.float64)
        if camera_distance(T, centre) < MIN_CAM_DIST * diag:
            continue
        rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
        pts = np.asarray(projs[idx].numpy(), np.float32).reshape(-1, 2)
        rect, _, K = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
        H, W = rect.shape[:2]
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
            continue                                   # silhouette clipped by the frame
        az = azimuth(camera_pos(T), centre)
        if any(min(abs(az - a), 360 - abs(az - a)) < MIN_SEP_DEG for _, a, _, _ in kept):
            continue                                   # a viewpoint we already have
        kept.append((idx, az, rect, sil))

    if len(kept) < 2:
        print(f"{name}: only {len(kept)} usable views — cannot build a sparse set")
        return None

    # Order along the arc, so consecutive files are consecutive camera positions --
    # that is what the interpolator assumes. Sorting by azimuth alone is not enough:
    # these captures are an arc, not a circle, so plain ascending order buries the
    # 160-220 degree jump between the arc's two ends in the middle of the sequence,
    # and one clip is then asked to interpolate across a gap nothing was observed
    # in. Rotating the order to *start* just after the largest gap puts that jump
    # between the last and first view, where it is never interpolated.
    kept.sort(key=lambda k: k[1])
    if len(kept) > 2:
        gaps = [kept[i + 1][1] - kept[i][1] for i in range(len(kept) - 1)]
        cut = int(np.argmax(gaps)) + 1
        if gaps[cut - 1] > 45.0:
            kept = kept[cut:] + kept[:cut]
            print(f"    {name}: cut the sequence at a {gaps[cut - 1]:.0f}° gap")
    if len(kept) > N_VIEWS:
        take = np.linspace(0, len(kept) - 1, N_VIEWS).round().astype(int)
        kept = [kept[i] for i in sorted(set(take.tolist()))]

    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.png"):
        f.unlink()
    meta = []
    for i, (idx, az, rect, sil) in enumerate(kept):
        H, W = rect.shape[:2]
        img = make_cutout(rect, feather(sil))
        win = crop_window(bbox_of_mask(sil, W, H), W, H)
        Image.fromarray(img).crop(win).resize((OUT_W, OUT_H), Image.LANCZOS) \
             .save(out / f"{i:03d}.png")
        meta.append({"file": f"{i:03d}.png", "rgb_frame_index": int(idx),
                     "azimuth_deg": round(az, 1)})
    # Circular differences, not linear ones: after the rotation above the sequence
    # crosses 360->0 somewhere in the middle, and |3.6 - 327.1| = 323.5 describes
    # an arithmetic gap the camera never travelled. The real step there is 36.5.
    az_seq = [a for _, a, _, _ in kept]
    steps = [min(abs(az_seq[i + 1] - az_seq[i]), 360 - abs(az_seq[i + 1] - az_seq[i]))
             for i in range(len(az_seq) - 1)]
    spread = sum(steps)
    rec = {"object": name, "views": len(kept), "rgb_frames_available": len(frames),
           "azimuth_span_deg": round(spread, 1),
           "largest_step_deg": round(max(steps), 1) if steps else 0.0,
           "azimuth_bins_of_12": len({int(a // 30) for _, a, _, _ in kept}),
           "frames": meta}
    # Beside the directory, never inside it: ViewCrafter's load_initial_dir sorts
    # the folder with int(filename), so any non-numeric file in there is fatal.
    (out.parent / f"{name}.json").write_text(json.dumps(rec, indent=2))
    print(f"{name:22s} {len(kept):3d} views  arc {spread:5.1f}°  "
          f"largest step {max(steps) if steps else 0:5.1f}°  "
          f"{rec['azimuth_bins_of_12']}/12 bins")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/viewcrafter_orbits/experiment6/inputs"))
    args = ap.parse_args()
    recs = [r for r in (process(n, args.out_dir) for n in C.names(args.only)) if r]
    (args.out_dir / "manifest.json").write_text(json.dumps(recs, indent=2))
    print(f"\nwrote {len(recs)} multi-view sets to {args.out_dir}")
