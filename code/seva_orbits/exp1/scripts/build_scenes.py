#!/usr/bin/env python3
"""Turn each ShapeR object into a Stable Virtual Camera scene: N background-removed
views with real poses, plus a 360 deg target orbit around the object.

    /home/isha.tariq/.conda/envs/shaper/bin/python scripts/build_scenes.py

Runs in the `shaper` env, not `svcam` -- it needs ShapeR's fisheye rectifier and
pyrender (EGL, so a GPU). The generation step afterwards runs in `svcam`.

Why this is different from the ViewCrafter experiments
------------------------------------------------------
Experiments 5 and 6 could not ask for a full turn: ViewCrafter's single-view mode
takes a path but only sees one image, and its sparse mode sees many images but
interpolates *their* poses, so the path is whatever the wearer walked. SEVA's
`img2trajvid` takes both -- P posed input views and target cameras we choose. So
here the target cameras are a real circle around the object, computed in the
object's own mesh frame, where the centre and the scale are exactly known rather
than inferred.

The camera chain, which is where this can silently go wrong
-----------------------------------------------------------
`Ts_rgbCamera_model` is camera-from-model, so `inv(T)` is camera-to-world with the
model as world, in *OpenCV* axes. SEVA's transforms.json wants *OpenGL* c2w --
`ReconfusionParser` does `camtoworlds[:, :, [1, 2]] *= -1` to undo it -- so every
pose is written as `inv(T) @ CV2GL`.

Intrinsics are tracked through every step (rectify, roll, re-aim) and written
**normalised** by width/height, which is what the parser expects: the upstream
examples carry `cx = 0.5`, not `cx = 448`.

The framing is a rotation, not a crop. SEVA asserts the principal point lies in
frame, so a tight crop around an off-centre object is rejected outright; turning
a virtual camera to face the object instead is an exact homography about the
unchanged optical centre. See `virtual_view`.

The roll is the subtle one. Aria Gen1 frames are stored a quarter turn off, and
rotating the *image* without rotating the *camera* would leave the poses
describing a picture we are not sending. A 90 deg clockwise image rotation is a
roll about the optical axis, so it is applied to both: K picks up
`fx,fy <- fy,fx` and `cx,cy <- H-1-cy, cx`, and the pose picks up the matching
`R_z`. Both together, or neither.

Nothing here is trusted on argument. Every view is verified by rasterising the
mesh through the *final* intrinsics and *final* pose into the *final* 576x576
image and measuring IoU against the mask that was actually cut out. A chain error
anywhere shows up as a low IoU, and the run refuses to write the scene.
"""

import argparse
import io
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment1/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment2/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh  # noqa: E402

import objects as C  # noqa: E402
import occlusion  # noqa: E402
from extract_reference_views import (  # noqa: E402
    SHAPER_DATA, bbox_of_mask, point_mask, rank_raw_frames, rectify_one, score_rect,
)
from build_object_only_views import (  # noqa: E402
    MIN_CAM_DIST, camera_distance, feather, make_cutout, render_silhouette, rigid,
)

CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])

OUT = 576              # SEVA's native side; square, so its centre crop is a no-op
N_VIEWS = 8            # SEVA's sparse-view regime is P<=8; also matches exp6/exp7
N_TARGETS = 80         # 80 targets over 360 deg = 4.5 deg per frame
N_SCAN = 260           # ranked frames examined before giving up
MIN_SEP_DEG = 4.0      # near-duplicate viewpoints cost forwards and add nothing
FILL = 0.60            # object's larger side as a fraction of the frame
FILL_LADDER = (0.60, 0.70, 0.80, 0.88)   # tighten only when the source runs out
MIN_IOU = 0.80         # a view whose reprojection misses this is not trustworthy
# Object pixels the source frame never photographed. Not zero, because for a big
# object shot close the wearer often never got all of it into one frame: the
# wooden table's silhouette touches the frame edge in most of its candidates, and
# `missing` there is flat at 8.3% across every framing in FILL_LADDER (measured by
# scripts/diagnose_coverage.py) -- the object is cut off in the photograph, so no
# reframing can recover it. 10% keeps such views, and `outside_source` in each
# scene's meta.json records exactly how truncated each one is.
MAX_MISSING = 0.10


# --------------------------------------------------------------- camera plumbing
def cam_pos(T_cam_model):
    """Camera centre in model coordinates. T is camera-from-model, so -R^T t."""
    T = np.asarray(T_cam_model, np.float64)
    return -T[:3, :3].T @ T[:3, 3]


def azimuth(pos, centre):
    v = pos - centre
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0)


def elevation(pos, centre):
    v = pos - centre
    return float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1]))))


def roll_image_and_camera(K, W, H, c2w_cv):
    """Apply the Gen1 quarter turn to intrinsics and extrinsics together.

    np.rot90(img, 3) maps a pixel (x, y) to (H-1-y, x). That is a roll about the
    optical axis, so the camera frame turns with it: X_new = R @ X_old with
    R = [[0,-1,0],[1,0,0],[0,0,1]]. Doing one without the other leaves the poses
    describing an image we are not sending.
    """
    K2 = np.eye(3)
    K2[0, 0], K2[1, 1] = K[1, 1], K[0, 0]
    K2[0, 2], K2[1, 2] = (H - 1) - K[1, 2], K[0, 2]
    R = np.eye(4)
    R[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    # X_new = R @ X_old means w2c_new = R @ w2c_old, so c2w_new = c2w_old @ R^-1.
    return K2, H, W, c2w_cv @ np.linalg.inv(R)


def virtual_view(rect, sil, K, c2w_cv, verts, centre, out=OUT, fill=FILL):
    """Re-photograph the view with a virtual camera that *looks at* the object.

    Cropping was the obvious move and it is the wrong one. A tight square around
    an off-centre object leaves the principal point outside the crop, and SEVA
    rejects that outright: `get_plucker_coordinates` asserts 0 <= cx,cy <= 1
    (seva/geometry.py:139), because its ray embedding assumes the principal point
    is in frame. 5 of our 40 views broke it.

    Rotating instead of cropping fixes it exactly. The virtual camera keeps the
    *same optical centre* and only turns to face the object, so the mapping is a
    pure homography K_new @ R_rel @ inv(K_old) with no parallax and nothing
    approximated -- a crop is just this with R = I and a shifted principal point.
    The result is centred by construction (cx = cy = 0.5), which is what the model
    wants, and the object is framed identically in every view.

    The focal is chosen so the object's own vertices fill `fill` of the frame, so
    apparent size stops depending on where the wearer happened to stand.
    """
    C = c2w_cv[:3, 3]
    R_old = c2w_cv[:3, :3]

    z = centre - C
    z /= np.linalg.norm(z)
    # OpenCV camera axes: x right, y down, z forward. Right-handed needs x = y x z.
    y_tmp = -np.array([0.0, 0.0, 1.0])          # model +z is up, so camera y is down
    x = np.cross(y_tmp, z)
    n = np.linalg.norm(x)
    if n < 1e-8:                                 # looking straight down the up axis
        x = np.cross(np.array([0.0, 1.0, 0.0]), z)
        n = np.linalg.norm(x)
    x /= n
    y = np.cross(z, x)
    R_new = np.stack([x, y, z], axis=1)          # columns are the camera axes

    # Frame the object: project its vertices into the new camera and pick f so the
    # widest one sits at `fill` of the half-frame.
    P = (verts - C) @ R_new                      # == (R_new.T @ (V - C).T).T
    front = P[:, 2] > 1e-6
    if front.sum() < 8:
        return None
    ratio = np.abs(P[front, :2] / P[front, 2:3]).max()
    f = fill * (out / 2.0) / max(ratio, 1e-6)

    K_new = np.eye(3)
    K_new[0, 0] = K_new[1, 1] = f
    K_new[0, 2] = K_new[1, 2] = out / 2.0

    R_rel = R_new.T @ R_old                      # X_new = R_rel @ X_old
    Hm = K_new @ R_rel @ np.linalg.inv(K)
    img = cv2.warpPerspective(rect, Hm, (out, out), flags=cv2.INTER_LANCZOS4,
                              borderMode=cv2.BORDER_CONSTANT)
    m = cv2.warpPerspective(sil, Hm, (out, out), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Where the re-aimed frustum falls outside what the original camera actually
    # photographed. Turning to face the object can ask for field of view the
    # source frame never had, and those pixels come back as border fill -- which
    # is a framing failure, not a geometry error, and has to be told apart from
    # one.
    cov = cv2.warpPerspective(np.ones(rect.shape[:2], np.float32), Hm, (out, out),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    c2w_new = np.eye(4)
    c2w_new[:3, :3], c2w_new[:3, 3] = R_new, C
    return img, m, cov, K_new, c2w_new


def look_at_gl(eye, target, up):
    """OpenGL camera-to-world: looks down -z, +y up.

    Basis order matters: z = (eye-target)/|.|, x = up x z, y = z x x. The other
    order gives a left-handed frame and the object tumbles in plane.
    """
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    n = np.linalg.norm(x)
    if n < 1e-8:                       # looking straight along `up`
        x = np.cross(np.array([1.0, 0.0, 0.0]), z)
        n = np.linalg.norm(x)
    x /= n
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def frame_entry(path, K, w, h, c2w_gl):
    """One transforms.json frame. Intrinsics normalised, pose OpenGL c2w."""
    return {
        "fl_x": float(K[0, 0]) / w, "fl_y": float(K[1, 1]) / h,
        "cx": float(K[0, 2]) / w, "cy": float(K[1, 2]) / h,
        "w": int(w), "h": int(h),
        "file_path": path,
        "transform_matrix": [[float(v) for v in row] for row in c2w_gl],
    }


# --------------------------------------------------------------------- per object
def gather_views(name):
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    frames, projs = d["rgb_image_data"], d["rgb_object_point_projections"]
    cams, poses = d["rgb_camera_params"], d["Ts_rgbCamera_model"]
    rotate_cw = not bool(d["is_ariagen2"])
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    centre = mesh.bounds.mean(axis=0)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    W0, H0 = Image.open(io.BytesIO(frames[0])).size
    verts = np.asarray(mesh.vertices, np.float64)

    kept, dropped = [], []
    for _, idx, _, _ in rank_raw_frames(projs, W0, H0)[:N_SCAN]:
        if len(kept) >= N_VIEWS * 3:
            break
        T = np.asarray(poses[idx].numpy(), np.float64)
        if camera_distance(T, centre) < MIN_CAM_DIST * diag:
            continue
        az = azimuth(cam_pos(T), centre)
        if any(min(abs(az - v["az"]), 360 - abs(az - v["az"])) < MIN_SEP_DEG
               for v in kept):
            continue
        rgb = np.asarray(Image.open(io.BytesIO(frames[idx])).convert("RGB"))
        pts = np.asarray(projs[idx].numpy(), np.float32).reshape(-1, 2)
        rect, _, K4 = rectify_one(rgb, point_mask(pts, W0, H0), cams[idx])
        rect = np.asarray(rect)
        K = np.asarray(K4, np.float64)[:3, :3].copy()
        H, W = rect.shape[:2]
        c2w_cv = rigid(np.linalg.inv(T))
        try:
            sil = render_silhouette(mesh, K, W, H, c2w_cv)
        except Exception:
            continue
        if sil.sum() < 2000:
            continue
        if rotate_cw:
            rect, sil = np.rot90(rect, 3).copy(), np.rot90(sil, 3).copy()
            K, W, H, c2w_cv = roll_image_and_camera(K, W, H, c2w_cv)
        box = bbox_of_mask(sil, W, H)
        if box is None or score_rect(box, W, H) < 0:
            continue                                  # silhouette clipped by frame

        # Re-aim and validate here, not after subsampling. Turning the camera to
        # face the object can ask for field of view the source frame never had,
        # and how often that happens varies wildly by object -- the wooden table
        # loses 6 views in 8 that way. Filtering before the even-spread pick means
        # those are replaced from the candidate pool instead of just lost.
        # Try the roomiest framing first and tighten only if the source frame
        # cannot cover it. Zooming in shrinks the field of view being asked for,
        # so a view that loses its edges at FILL=0.60 is often perfectly complete
        # at 0.80 -- which is the wooden table's whole problem: shot close and
        # from above, 29 of its candidates fail the loose framing and pass a
        # tighter one. Losing the object entirely is worse than framing it tight.
        best = None
        for fill in FILL_LADDER:
            vv = virtual_view(rect, sil, K, c2w_cv, verts, centre, fill=fill)
            if vv is None:
                break
            img_w, sil_w, cov, Kf, c2w_f = vv
            # Rasterise the mesh with the *final* intrinsics and *final* pose.
            # This is both the chain check and the mask composited with: taking
            # it fresh at 576 avoids the double resampling of warping an
            # already-rendered mask, so the cut-out edge is exact.
            got = render_silhouette(mesh, Kf, OUT, OUT, c2w_f) > 0.5
            if got.sum() < 500:
                break
            missing = float((got & ~cov).sum() / got.sum())
            if missing <= MAX_MISSING:
                best = (img_w, sil_w, cov, Kf, c2w_f, got, missing, fill)
                break
        if best is None:
            dropped.append(missing)
            continue
        img_w, sil_w, cov, Kf, c2w_f, got, missing, fill_used = best
        # Geometry check, scored only where the source actually covers, so border
        # loss cannot masquerade as a camera-chain error.
        ref, g = (sil_w > 0.5) & cov, got & cov
        union = (ref | g).sum()
        iou = float((ref & g).sum() / union) if union else 0.0

        # Alpha is the object AND what the camera actually saw. Using the full
        # silhouette would composite border fill as though it were object, and a
        # truncated view would gain a black wedge; this way the unphotographed
        # part reads as backdrop, which is honest about what is known.
        kept.append({"az": az, "idx": int(idx), "img": img_w, "alpha": got & cov,
                     "cov": cov, "K": Kf, "c2w": c2w_f, "iou": iou,
                     "missing": missing, "fill": fill_used})

    if dropped:
        print(f"    {name}: {len(dropped)} candidate views dropped, "
              f"{min(dropped):.1%}-{max(dropped):.1%} outside the source frame")
    return d, mesh, centre, diag, kept


def build(name, out_root, report):
    d, mesh, centre, diag, kept = gather_views(name)
    if len(kept) < 2:
        print(f"{name}: only {len(kept)} usable views")
        return None

    kept.sort(key=lambda k: k["az"])
    pool = list(kept)
    if len(kept) > N_VIEWS:                            # spread evenly around the arc
        take = np.linspace(0, len(kept) - 1, N_VIEWS).round().astype(int)
        kept = [kept[i] for i in sorted(set(take.tolist()))]

    # A silhouette has no depth test, so whatever stands in front of the object
    # comes along with it. Ask the other views what that surface really looks
    # like: occluders get repaired, views that are not the object at all (bench
    # frame 230 is a bathroom -- the wearer walked next door and the bench is
    # behind a wall) get dropped. Refill from the pool so dropping still leaves
    # N_VIEWS, then re-clean, because a replacement can be bad too.
    # Iterate to convergence, not a fixed number of rounds: a view pulled in to
    # replace a rejected one can be bad itself, and stopping early would leave it
    # in the scene unchecked.
    condemned, rejected = set(), []
    for _ in range(6):
        keep_i, repaired, unrep, stats, reasons = occlusion.clean(kept, mesh)
        for v, st in zip(kept, stats):
            v["inconsistent"] = round(float(st["frac"]), 4)
            v["cross_checked"] = round(float(st["judged"]), 4)
        for i, why in reasons.items():
            print(f"    {name}: dropped view at az {kept[i]['az']:.0f}° -- {why}")
            rejected.append({"azimuth_deg": round(kept[i]["az"], 1),
                             "rgb_frame_index": kept[i]["idx"], "reason": why})
            condemned.add(kept[i]["idx"])
        for v, img, u in zip([kept[i] for i in keep_i], repaired, unrep):
            v["img"], v["alpha"] = img, v["alpha"] & ~u
        kept = [kept[i] for i in keep_i]
        if not reasons:
            break
        spare = [p for p in pool
                 if p["idx"] not in {v["idx"] for v in kept} | condemned]
        if not spare or len(kept) >= N_VIEWS:
            break
        # refill the widest azimuth gaps first
        for p in sorted(spare, key=lambda p: -min(
                (min(abs(p["az"] - v["az"]), 360 - abs(p["az"] - v["az"]))
                 for v in kept), default=360.0)):
            if len(kept) >= N_VIEWS:
                break
            kept.append(p)
        kept.sort(key=lambda k: k["az"])

    if len(kept) < 2:
        print(f"{name}: only {len(kept)} views survived the consistency check")
        return None

    scene = out_root / name
    (scene / "images").mkdir(parents=True, exist_ok=True)
    for f in (scene / "images").glob("*.png"):
        f.unlink()

    frames_json, meta, ious = [], [], []
    for i, v in enumerate(kept):
        Image.fromarray(make_cutout(v["img"], feather(v["alpha"].astype(np.float32)))) \
             .save(scene / "images" / f"{i:03d}.png")
        ious.append(v["iou"])
        frames_json.append(frame_entry(f"./images/{i:03d}.png", v["K"], OUT, OUT,
                                       v["c2w"] @ CV2GL))
        meta.append({"file": f"images/{i:03d}.png", "rgb_frame_index": v["idx"],
                     "azimuth_deg": round(v["az"], 1),
                     "reprojection_iou": round(v["iou"], 4),
                     "outside_source": round(v["missing"], 4), "fill": v["fill"],
                     "inconsistent": v.get("inconsistent", 0.0),
                     "cross_checked": v.get("cross_checked", 0.0)})

    worst = min(ious)
    if worst < MIN_IOU:
        print(f"{name}: REFUSED -- worst reprojection IoU {worst:.3f} < {MIN_IOU}")
        return None

    # ---------------------------------------------------------- target 360 orbit
    # In the object's own frame, with +z up (the convention azimuth() already
    # uses). Radius and elevation come from the input views, so the orbit stays in
    # the distribution the model was conditioned on rather than somewhere novel.
    pos = np.array([v["c2w"][:3, 3] for v in kept])    # c2w translation = centre
    radius = float(np.median(np.linalg.norm(pos - centre, axis=1)))
    elev = float(np.median([elevation(p, centre) for p in pos]))
    f_norm = float(np.median([fr["fl_x"] for fr in frames_json]))
    start_az = float(kept[0]["az"])

    n_inputs = len(frames_json)
    up = np.array([0.0, 0.0, 1.0])
    Kt = np.eye(3)
    Kt[0, 0] = Kt[1, 1] = f_norm * OUT
    Kt[0, 2] = Kt[1, 2] = OUT / 2.0
    for j in range(N_TARGETS):
        a = np.radians(start_az + 360.0 * j / N_TARGETS)
        eye = centre + radius * np.array([
            np.cos(a) * np.cos(np.radians(elev)),
            np.sin(a) * np.cos(np.radians(elev)),
            np.sin(np.radians(elev)),
        ])
        # look_at_gl already returns OpenGL c2w, so no CV2GL here.
        frames_json.append(frame_entry(None, Kt, OUT, OUT,
                                       look_at_gl(eye, centre, up)))

    n_in = n_inputs
    (scene / "transforms.json").write_text(json.dumps(
        {"orientation_override": "none", "frames": frames_json}, indent=2))
    (scene / f"train_test_split_{n_in}.json").write_text(json.dumps(
        {"train_ids": list(range(n_in)),
         "test_ids": list(range(n_in, n_in + N_TARGETS))}, indent=2))

    az_seq = [v["az"] for v in kept]
    steps = [min(abs(az_seq[i + 1] - az_seq[i]), 360 - abs(az_seq[i + 1] - az_seq[i]))
             for i in range(len(az_seq) - 1)]
    rec = {"object": name, "views": n_in, "targets": N_TARGETS,
           "radius": round(radius, 3), "elevation_deg": round(elev, 1),
           "mesh_diag": round(diag, 3),
           "azimuth_span_deg": round(sum(steps), 1),
           "azimuth_bins_of_12": len({int(a // 30) for a in az_seq}),
           "reprojection_iou_min": round(worst, 4),
           "reprojection_iou_mean": round(float(np.mean(ious)), 4),
           "rejected_views": rejected,
           "frames": meta}
    (scene / "meta.json").write_text(json.dumps(rec, indent=2))
    report.append(rec)
    print(f"{name:22s} {n_in} views  arc {sum(steps):5.1f}°  "
          f"{rec['azimuth_bins_of_12']}/12 bins  r={radius:.2f}  elev={elev:+.0f}°  "
          f"IoU min {worst:.3f} mean {np.mean(ious):.3f}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/stable_virtual_camera/exp1/inputs"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for n in C.names(args.only):
        build(n, args.out_dir, report)

    path = args.out_dir / "manifest.json"
    old = {r["object"]: r for r in json.loads(path.read_text())} if path.exists() else {}
    old.update({r["object"]: r for r in report})       # merge, never overwrite
    path.write_text(json.dumps(list(old.values()), indent=2))
    print(f"\n{len(report)} scenes built, {len(old)} in manifest")
