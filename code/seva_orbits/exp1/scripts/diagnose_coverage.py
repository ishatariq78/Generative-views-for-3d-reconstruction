#!/usr/bin/env python3
"""Why do views fail the coverage test? Tighten the framing and watch.

    python scripts/diagnose_coverage.py ADT1292__woodentable

If `missing` falls as `fill` rises, the frame merely needed to be tighter. If it
is flat, the object itself runs off the edge of the source photograph and no
framing can recover it -- a fact about the capture, not about our reframing.
"""
import sys

sys.argv = [sys.argv[0]] + sys.argv[1:]
sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp1/scripts")

import numpy as np  # noqa: E402
from build_scenes import (  # noqa: E402
    FILL_LADDER, OUT, gather_views, render_silhouette, virtual_view,
)

name = sys.argv[1] if len(sys.argv) > 1 else "ADT1292__woodentable"

# gather_views already filters; re-run its front half by hand instead.
import io, pickle  # noqa: E402
from PIL import Image  # noqa: E402
import trimesh  # noqa: E402
from extract_reference_views import (  # noqa: E402
    SHAPER_DATA, bbox_of_mask, point_mask, rank_raw_frames, rectify_one, score_rect,
)
from build_object_only_views import MIN_CAM_DIST, camera_distance, rigid  # noqa: E402
from build_scenes import azimuth, cam_pos, roll_image_and_camera  # noqa: E402

d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
cams, poses = d["rgb_camera_params"], d["Ts_rgbCamera_model"]
rotate_cw = not bool(d["is_ariagen2"])
mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                       faces=np.asarray(d["mesh_faces"]), process=False)
centre = mesh.bounds.mean(axis=0)
diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
verts = np.asarray(mesh.vertices, np.float64)
W0, H0 = Image.open(io.BytesIO(frames[0])).size

shown = 0
for _, idx, _, _ in rank_raw_frames(projs, W0, H0)[:60]:
    if shown >= 6:
        break
    T = np.asarray(poses[idx].numpy(), np.float64)
    if camera_distance(T, centre) < MIN_CAM_DIST * diag:
        continue
    rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
    pts = np.asarray(projs[idx].numpy(), np.float32).reshape(-1, 2)
    rect, _, K4 = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
    rect = np.asarray(rect)
    K = np.asarray(K4, np.float64)[:3, :3].copy()
    H, W = rect.shape[:2]
    c2w = rigid(np.linalg.inv(T))
    try:
        sil = render_silhouette(mesh, K, W, H, c2w)
    except Exception:
        continue
    if sil.sum() < 2000:
        continue
    if rotate_cw:
        rect, sil = np.rot90(rect, 3).copy(), np.rot90(sil, 3).copy()
        K, W, H, c2w = roll_image_and_camera(K, W, H, c2w)
    box = bbox_of_mask(sil, W, H)
    if box is None or score_rect(box, W, H) < 0:
        continue

    # Does the source silhouette touch the frame edge? If so the object is cut
    # off in the photograph itself.
    m = sil > 0.5
    edge = (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())
    row = []
    for fill in FILL_LADDER:
        vv = virtual_view(rect, sil, K, c2w, verts, centre, fill=fill)
        if vv is None:
            row.append("--")
            continue
        _, _, cov, Kf, c2w_f = vv
        got = render_silhouette(mesh, Kf, OUT, OUT, c2w_f) > 0.5
        row.append(f"{(got & ~cov).sum() / max(got.sum(), 1):.3f}")
    print(f"frame {idx:5d}  az {azimuth(cam_pos(T), centre):6.1f}  "
          f"silhouette touches frame edge: {str(edge):5s}   "
          f"missing at fill {FILL_LADDER}: {row}")
    shown += 1
