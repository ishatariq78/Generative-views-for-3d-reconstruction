#!/usr/bin/env python3
"""Pi3 over the SEVA orbits, writing exactly what the ViewCrafter-workspace VGGT runner writes.

    /home/isha.tariq/.conda/envs/kvtracker/bin/python scripts/run_pi3.py --only <set> ...

A drop-in twin of ~/viewcrafter_orbits/experiment3/scripts/run_vggt.py with the
reconstructor swapped: same input frame sets, same subject mask, same
confidence-quantile filter, same output files (points_object.ply, points_all.ply,
a .npz, stats.json) into the same outputs/ tree -- so align_and_score.py,
compare_slam_points.py and realign_to_slam.py score the Pi3 clouds on identical
terms to every VGGT cloud, ViewCrafter's and SEVA's alike.

Pi3 (Wang et al. 2025) is the backbone KV-Tracker is built on: a permutation-
equivariant feed-forward model that returns, per pixel, a world-frame point, a
confidence logit and a camera-to-world pose, with no reference view. Runs in the
kvtracker env, which has its compiled RoPE kernel.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pi3.models.pi3 import Pi3  # noqa: E402
from pi3.utils.basic import load_images_as_tensor  # noqa: E402

# Copied verbatim from ~/viewcrafter_orbits/experiment3/scripts/run_vggt.py rather than
# imported: importing that module loads the VGGT package, which this env lacks.
CONF_QUANTILE = 0.35      # drop the least-confident third of surviving points
SUBJ_SAT, SUBJ_VAL = 0.15, 0.70   # same backdrop rule as experiment 2


def subject_mask(images):
    """True where a pixel is object rather than backdrop. images: (N,3,H,W) in [0,1]."""
    mx = images.max(axis=1); mn = images.min(axis=1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return (sat > SUBJ_SAT) | (mx < SUBJ_VAL)


def write_ply(path, xyz, rgb):
    xyz = np.asarray(xyz, np.float32)
    rgb = np.clip(np.asarray(rgb) * 255, 0, 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n").encode())
        rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rec["r"], rec["g"], rec["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        f.write(rec.tobytes())

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment3")


def run_one(model, frame_dir, out_dir, device, dtype):
    if not sorted(frame_dir.glob("*.png")):
        print(f"  !! no frames in {frame_dir}")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    imgs = load_images_as_tensor(str(frame_dir), interval=1).to(device)   # (N,3,H,W) in [0,1]
    t0 = time.time()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
        res = model(imgs[None])
    infer_s = time.time() - t0

    pts = res["points"][0].float().cpu().numpy().astype(np.float32)          # (N,H,W,3) world
    conf = torch.sigmoid(res["conf"][0][..., 0]).float().cpu().numpy()       # (N,H,W) in (0,1)
    poses = res["camera_poses"][0].float().cpu().numpy()                     # (N,4,4) c2w
    im = imgs.float().cpu().numpy()                                          # (N,3,H,W)
    cols = im.transpose(0, 2, 3, 1)

    subj = subject_mask(im)
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
    np.savez_compressed(out_dir / "pi3.npz", camera_poses=poses,
                        object_points=pts[obj_m], object_colors=cols[obj_m].astype(np.float32),
                        object_conf=conf[obj_m].astype(np.float32))
    stats = {"reconstructor": "pi3", "frames": int(im.shape[0]), "resolution": list(im.shape[-2:]),
             "inference_seconds": round(infer_s, 2), "pixels_total": int(finite.size),
             "pixels_subject": int(subj.sum()), "subject_fraction": round(float(subj.mean()), 4),
             "points_all": int(all_m.sum()), "points_object": int(obj_m.sum()),
             "conf_quantile": CONF_QUANTILE, "object_conf_threshold": round(float(thr), 4),
             "conf_median_object": round(float(np.median(conf[obj_m])), 4) if obj_m.any() else None}
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  {stats['frames']:3d} frames @ {stats['resolution']} -> {stats['points_object']:8d} object points "
          f"({stats['points_all']:8d} total), conf median {stats['conf_median_object']}, {infer_s:.1f}s")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=PROJ / "inputs")
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    t = time.time(); model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
    print(f"Pi3 loaded in {time.time()-t:.0f}s")
    sets = [args.in_dir / s for s in args.only] if args.only else \
        sorted(p for p in args.in_dir.iterdir() if p.is_dir())
    log = {}
    for s in sets:
        print(s.name)
        st = run_one(model, s, args.out_dir / s.name, device, dtype)
        if st:
            log[s.name] = st
    p = Path("/home/isha.tariq/stable_virtual_camera/exp4/outputs"); p.mkdir(exist_ok=True)
    (p / "runs.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
