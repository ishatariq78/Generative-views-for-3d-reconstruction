#!/usr/bin/env python3
"""Run VGGT over each frame set and extract the target object's 3D points.

VGGT (CVPR 2025) is feed-forward: hand it a set of views and it returns camera
poses, depth maps and point maps in one pass, with no bundle adjustment and no
correspondence search. That makes it a clean instrument for the question here --
does experiment 2's generated orbit carry consistent 3D structure, or does it only
look like it does?

Per frame set this writes into outputs/<object>__<condition>/:
    points_object.ply   the target object's points, coloured        <- the result
    points_all.ply      everything VGGT returned, for comparison
    vggt.npz            extrinsics, intrinsics, per-point confidence and mask
    stats.json          counts and the thresholds that produced them

Points come from unprojecting the depth branch rather than from the point-map
branch: the authors note that route is usually the more accurate of the two.

Isolating the object needs no segmentation. Experiment 2's cutout put every frame
on a light, desaturated backdrop, so a pixel that is saturated or dark is the
object -- the same rule experiment 2 used to measure subject area, reused here as
a mask. Points are additionally filtered by VGGT's own confidence.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

VGGT_REPO = Path("/home/isha.tariq/viewcrafter_orbits/third_party/vggt")
sys.path.insert(0, str(VGGT_REPO))

from vggt.models.vggt import VGGT                                   # noqa: E402
from vggt.utils.geometry import unproject_depth_map_to_point_map    # noqa: E402
from vggt.utils.load_fn import load_and_preprocess_images           # noqa: E402
from vggt.utils.pose_enc import pose_encoding_to_extri_intri        # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment3")
CONF_QUANTILE = 0.35      # drop the least-confident third of surviving points
SUBJ_SAT, SUBJ_VAL = 0.15, 0.70   # same backdrop rule as experiment 2


def subject_mask(images):
    """True where a pixel is object rather than backdrop. images: (N,3,H,W) in [0,1]."""
    mx = images.max(axis=1)
    mn = images.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > SUBJ_SAT) | (mx < SUBJ_VAL)


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"""ply
format binary_little_endian 1.0
element vertex {len(xyz)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".encode())
        rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                        ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())


def run_one(model, frame_dir, out_dir, device, dtype):
    paths = sorted(frame_dir.glob("*.png"))
    if not paths:
        print(f"  !! no frames in {frame_dir}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_and_preprocess_images([str(p) for p in paths]).to(device)
    t0 = time.time()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=dtype):
            tokens, ps_idx = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth, depth_conf = model.depth_head(tokens, images[None], ps_idx)
    infer_s = time.time() - t0

    # Depth-branch unprojection: the authors note it beats the point-map branch.
    pts = unproject_depth_map_to_point_map(depth.squeeze(0).float().cpu().numpy(),
                                           extrinsic.squeeze(0).float().cpu().numpy(),
                                           intrinsic.squeeze(0).float().cpu().numpy())
    pts = np.asarray(pts, np.float32)                 # (N,H,W,3)
    conf = depth_conf.squeeze(0).float().cpu().numpy()  # (N,H,W)
    imgs = images.float().cpu().numpy()                 # (N,3,H,W)
    cols = imgs.transpose(0, 2, 3, 1)                   # (N,H,W,3)

    subj = subject_mask(imgs)
    finite = np.isfinite(pts).all(-1)

    all_m = finite & (conf > np.quantile(conf[finite], CONF_QUANTILE))
    obj_m = finite & subj
    if obj_m.sum() > 100:
        thr = np.quantile(conf[obj_m], CONF_QUANTILE)
        obj_m &= conf > thr
    else:
        thr = 0.0

    write_ply(out_dir / "points_all.ply", pts[all_m], cols[all_m])
    write_ply(out_dir / "points_object.ply", pts[obj_m], cols[obj_m])
    np.savez_compressed(out_dir / "vggt.npz",
                        extrinsic=extrinsic.squeeze(0).float().cpu().numpy(),
                        intrinsic=intrinsic.squeeze(0).float().cpu().numpy(),
                        object_points=pts[obj_m].astype(np.float32),
                        object_colors=cols[obj_m].astype(np.float32),
                        object_conf=conf[obj_m].astype(np.float32))

    stats = {
        "frames": len(paths), "resolution": list(imgs.shape[-2:]),
        "inference_seconds": round(infer_s, 2),
        "pixels_total": int(finite.size),
        "pixels_subject": int(subj.sum()),
        "subject_fraction": round(float(subj.mean()), 4),
        "points_all": int(all_m.sum()), "points_object": int(obj_m.sum()),
        "conf_quantile": CONF_QUANTILE, "object_conf_threshold": round(float(thr), 4),
        "conf_median_object": round(float(np.median(conf[obj_m])), 4) if obj_m.any() else None,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  {len(paths):3d} frames -> {stats['points_object']:8d} object points "
          f"({stats['points_all']:8d} total), conf median "
          f"{stats['conf_median_object']}, {infer_s:.1f}s")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=PROJ / "inputs")
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these set names")
    args = ap.parse_args()

    sets = sorted(p for p in args.in_dir.iterdir()
                  if p.is_dir() and (args.only is None or p.name in args.only))
    if not sets:
        raise SystemExit(f"no frame sets under {args.in_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 needs Ampere or newer; the A100s here are sm_80, so it applies.
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    print(f"=== loading VGGT-1B on {device} ({dtype}) ===", flush=True)
    t0 = time.time()
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log = []
    for s in sets:
        print(f"\n=== {s.name} ===", flush=True)
        st = run_one(model, s, args.out_dir / s.name, device, dtype)
        if st:
            log.append({"set": s.name, **st})
    (args.out_dir / "runs.json").write_text(json.dumps(log, indent=2))
    print(f"\n=== {len(log)} sets, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
