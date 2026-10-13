#!/usr/bin/env python3
"""Give ShapeR the *generated views* of the object, not just generated points.

    ~/.conda/envs/shaper/bin/python scripts/make_pkls_views.py --only <obj> --check

ShapeR's image conditioning is its strongest input, and in the benchmark it is the weakest
link: the views are whatever the wearer walked past -- partial arc, clutter, occluders,
sometimes another room. The SEVA orbit is the opposite: the target object alone on a plain
backdrop, cleaned of occluders (exp1's multi-view consistency step), all the way round.

The camera chain is exact rather than approximate. Each orbit frame was rendered from a
camera we specified, so a synthetic Aria view at that same optical centre is a pure
resampling of it -- no parallax, no depth needed. ShapeR reads Fisheye624 cameras, and a
Fisheye624 with zero distortion is exactly equidistant (r = f*theta; verified numerically),
so the frame is remapped pinhole -> equidistant and declared with params [f, cx, cy, 0x12].
`rectify_images` then maps it straight back to a pinhole view.

Conditions (image source, how many orbit frames offered, which points):
  views   real + 16 generated      SLAM points   images are the only change vs `slam`
  vonly   16 generated only        SLAM points   the clean object-only 360 deg condition
  vg80    real + all 80 generated  SLAM points   let the view selector pick its own 16
  v80     all 80 generated only    SLAM points
  vgap    real + 16 generated      gap1 points   both interventions at once

`views` and the point conditions win on different objects -- `views` rescues the captures
that are occluded or one-sided, the point conditions help where the capture is already
clean -- so `vgap` runs them together.
"""
import argparse, io, json, os, pickle
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

E = Path("/home/isha.tariq/seva_to_shaper"); SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
# The subset run keeps its own scenes, orbits and pkls so it cannot disturb the 6-object study.
SCENES = Path(os.environ.get("S2S_SCENES", E / "inputs"))
ORBITS = Path(os.environ.get("S2S_ORBITS", E / "outputs/demo/img2trajvid"))
OUT = Path(os.environ.get("S2S_PKLS", E / "outputs/pkls")); OUT.mkdir(parents=True, exist_ok=True)
GL2CV = np.diag([1.0, -1.0, -1.0, 1.0])
N_GEN = 16            # generated views added; ShapeR's `balance` preset consumes 16
DOT_MIN = 60          # a view needs at least this many projected object points to be usable


def pinhole_to_equidistant(img, f, cx, cy):
    """Resample a pinhole image so that radius = f*theta instead of f*tan(theta).

    Builds the inverse map: for each output (equidistant) pixel, theta = r_out / f, so the
    source radius is f*tan(theta). Identity at the centre, compressive at the edge.
    """
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r_out = np.hypot(dx, dy)
    theta = r_out / f
    valid = theta < (np.pi / 2 - 1e-3)
    r_src = np.where(valid, f * np.tan(np.clip(theta, 0, np.pi / 2 - 1e-3)), 1e6)
    s = np.where(r_out > 1e-6, r_src / np.maximum(r_out, 1e-6), 1.0)
    map_x = (cx + dx * s).astype(np.float32); map_y = (cy + dy * s).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def project_equidistant(P_cam, f, cx, cy):
    """Project camera-frame points with the same zero-distortion Fisheye624 model."""
    z = P_cam[:, 2]
    rho = np.linalg.norm(P_cam[:, :2], axis=1)
    theta = np.arctan2(rho, z)
    scale = np.where(rho > 1e-9, f * theta / np.maximum(rho, 1e-9), 0.0)
    return np.stack([cx + P_cam[:, 0] * scale, cy + P_cam[:, 1] * scale], 1), z > 1e-6


def fit_to_canvas(img, f, cx, cy, W0, H0):
    """Put the square orbit frame, unstretched, in the middle of the capture's own frame size.

    The loader stacks every selected view into one array, so a generated view has to carry
    the capture's resolution exactly. Scaling by min(W0,H0)/side and pasting centred keeps
    the principal point at the canvas centre (it is the frame centre to begin with), so the
    camera stays exact: f scales with the image, cx/cy become the new centre.
    """
    side = img.shape[0]
    sc = min(W0, H0) / side
    small = cv2.resize(img, (int(round(side * sc)), int(round(side * sc))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((H0, W0), img.dtype)
    x0, y0 = (W0 - small.shape[1]) // 2, (H0 - small.shape[0]) // 2
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas, f * sc, x0 + (cx * sc), y0 + (cy * sc)


def orbit_views(obj, n_gen=N_GEN):
    """(image, params15, T_cam_model, uv, pts3d) per generated view, in the model frame."""
    tf = json.loads((SCENES / obj / "transforms.json").read_text())["frames"]
    ids = json.loads(next((SCENES / obj).glob("train_test_split_*.json")).read_text())["test_ids"]
    frames = sorted((ORBITS / obj / "samples-rgb").glob("*.png"))
    assert len(frames) == len(ids), (len(frames), len(ids))
    pick = [round(i * (len(frames) - 1) / (n_gen - 1)) for i in range(n_gen)]
    base = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    slam = base["points_model"].numpy().astype(np.float64)
    W0, H0 = Image.open(io.BytesIO(base["image_data"][0])).size      # the capture's own frame size
    out = []
    for k in pick:
        fr = tf[ids[k]]
        w, h = int(fr["w"]), int(fr["h"])
        f, cx, cy = fr["fl_x"] * w, fr["cx"] * w, fr["cy"] * h
        c2w = np.asarray(fr["transform_matrix"], np.float64) @ GL2CV      # OpenGL -> OpenCV c2w
        T_cam_model = np.linalg.inv(c2w)
        img = np.asarray(Image.open(frames[k]).convert("L"))
        img_eq = pinhole_to_equidistant(img, f, cx, cy)
        img_eq, f, cx, cy = fit_to_canvas(img_eq, f, cx, cy, W0, H0)
        P = (slam - c2w[:3, 3]) @ c2w[:3, :3]                            # model -> camera
        uv, front = project_equidistant(P, f, cx, cy)
        inside = front & (uv[:, 0] >= 0) & (uv[:, 0] < W0) & (uv[:, 1] >= 0) & (uv[:, 1] < H0)
        out.append({"image": img_eq, "params": np.array([f, cx, cy] + [0.0] * 12, np.float32),
                    "T": T_cam_model.astype(np.float32), "uv": uv[inside].astype(np.float32),
                    "pts": slam[inside].astype(np.float32), "frame": int(k), "n_dots": int(inside.sum())})
    return out


def encode(img):
    buf = io.BytesIO(); Image.fromarray(img).save(buf, format="PNG"); return buf.getvalue()


SPEC = {"views": ("mix", 16, None), "vonly": ("gen", 16, None), "vg80": ("mix", 80, None),
        "v80": ("gen", 80, None), "vgap": ("mix", 16, "gap1")}


def points_from(obj, cond):
    """Reuse another condition's `points_model` so images and points can be combined."""
    with open(OUT / f"{obj}__{cond}.pkl", "rb") as fh:
        return pickle.load(fh)["points_model"]


def build(obj, conds):
    base = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    n_max = max(SPEC[c][1] for c in conds)
    allv = [v for v in orbit_views(obj, n_max) if v["n_dots"] >= DOT_MIN]
    if len(allv) < 4:
        print(f"!! {obj}: only {len(allv)} generated views carry projected points -- skipped"); return None
    cache = {}
    made = {}
    for cond in conds:
        mode, n_gen, pts_from = SPEC[cond]
        if n_gen not in cache:
            take = [allv[round(i * (len(allv) - 1) / (n_gen - 1))] for i in range(min(n_gen, len(allv)))]
            cache[n_gen] = ([encode(v["image"]) for v in take],
                            torch.from_numpy(np.stack([v["params"] for v in take])),
                            torch.from_numpy(np.stack([v["T"] for v in take])),
                            [torch.from_numpy(v["pts"]) for v in take],
                            [torch.from_numpy(v["uv"]) for v in take])
        gen_imgs, gen_par, gen_T, gen_vis, gen_uv = cache[n_gen]
        d = dict(base)
        if pts_from:
            d["points_model"] = points_from(obj, pts_from)
            n = len(d["points_model"])
            d["inv_dist_std"] = torch.zeros(n, dtype=torch.float32)
            d["dist_std"] = torch.zeros(n, dtype=torch.float32)
        if mode == "mix":
            d["image_data"] = list(base["image_data"]) + gen_imgs
            d["camera_params"] = torch.cat([base["camera_params"], gen_par])
            d["Ts_camera_model"] = torch.cat([base["Ts_camera_model"], gen_T])
            d["visible_points_model"] = list(base["visible_points_model"]) + gen_vis
            d["object_point_projections"] = list(base["object_point_projections"]) + gen_uv
        else:                                        # generated views alone
            d["image_data"] = gen_imgs
            d["camera_params"] = gen_par
            d["Ts_camera_model"] = gen_T
            d["visible_points_model"] = gen_vis
            d["object_point_projections"] = gen_uv
        d["s2s_condition"] = cond
        with open(OUT / f"{obj}__{cond}.pkl", "wb") as fh:
            pickle.dump(d, fh)
        made[cond] = len(d["image_data"])
    print(f"{obj:26s} {len(allv)} orbit frames usable  " + "  ".join(f"{k}:{v}v" for k, v in made.items()))
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--conds", nargs="*", default=["vg80", "v80", "vgap"])
    args = ap.parse_args()
    objs = json.loads(Path(os.environ.get("S2S_SETS", E / "outputs/sets.json")).read_text())
    objs = [o["name"] if isinstance(o, dict) else o for o in objs]
    rows = {}
    for o in objs:
        if args.only and o not in args.only: continue
        r = build(o, args.conds)
        if r: rows[o] = r
    (OUT / "manifest_views.json").write_text(json.dumps(rows, indent=2))
