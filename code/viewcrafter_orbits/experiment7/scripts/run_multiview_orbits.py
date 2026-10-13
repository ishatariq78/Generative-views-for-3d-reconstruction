#!/usr/bin/env python3
"""The single-view orbit experiments, re-run with an eight-view point cloud.

Experiments 2 and 5 chose a camera path — a 360 degree orbit, a raised one, a
helix — and rendered it from a point cloud DUSt3R built out of **one** photograph.
Experiment 6 gave DUSt3R eight views but had to accept ViewCrafter's own
interpolated path, which follows the wearer's arc and cannot be a full turn.

This combines the two halves: the eight-view cloud, and a trajectory of our
choosing. It is the same experiment as the pipeline deck, with the geometry
underneath it built from eight views instead of one.

ViewCrafter has no mode for that, but it has both pieces. `nvs_sparse_view` builds
a masked cloud from every input view and renders a *specified* trajectory through
it; `nvs_single_view` reads a trajectory from a text file. This borrows the cloud
and masking from the first and the trajectory from the second.

Two choices worth stating:

  **Anchor.** The orbit pivots on the middle view of the arc, not the first. The
  views run along an arc, so the middle one looks at the object from the centre of
  what was actually observed — anchoring on an end would put half the turn behind
  the object immediately.

  **Checkpoint.** The standard `model.ckpt`, not `model_sparse.ckpt`. The sparse
  checkpoint is trained to interpolate between given views; here the diffusion job
  is the ordinary one — 25 frames along a chosen path — and only the point cloud
  underneath is richer.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

VC = Path("/home/isha.tariq/view_crafter/ViewCrafter")
sys.path.insert(0, str(VC))
os.chdir(VC)

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402
import torch.nn.functional as F                       # noqa: E402
from configs.infer_config import get_parser           # noqa: E402
from pytorch_lightning import seed_everything         # noqa: E402
from utils.pvd_utils import (generate_traj_txt, save_video,  # noqa: E402
                             save_pointcloud_with_normals, world_point_to_obj)
from dust3r.utils.device import to_numpy              # noqa: E402
from viewcrafter import ViewCrafter                   # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment7")
VIEWS = Path("/home/isha.tariq/viewcrafter_orbits/experiment6/inputs8")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as C                                   # noqa: E402


def read_traj(path):
    lines = Path(path).read_text().splitlines()
    return ([float(v) for v in lines[0].split()],
            [float(v) for v in lines[1].split()],
            [float(v) for v in lines[2].split()])


def orbit_from_multiview(pvd, traj_file):
    """nvs_sparse_view's cloud and masks, nvs_single_view's chosen trajectory."""
    o = pvd.opts
    scene = pvd.scene
    c2ws = scene.get_im_poses().detach()
    principal_points = scene.get_principal_points().detach()
    focals = scene.get_focals().detach()
    shape = pvd.images[0]["true_shape"]
    H, W = int(shape[0][0]), int(shape[0][1])
    pcd = [i.detach() for i in scene.get_pts3d(clip_thred=o.dpt_trd)]
    depth = [i.detach() for i in scene.get_depthmaps()]

    k = len(pvd.images) // 2                     # middle of the arc
    radius = depth[k][H // 2, W // 2] * o.center_scale
    c2ws, pcd = world_point_to_obj(poses=c2ws, points=torch.stack(pcd), k=k,
                                   r=radius, elevation=o.elevation, device=pvd.device)

    # Masking, exactly as nvs_sparse_view does it: drop low-confidence points and
    # the far background, or the orbit renders a cloud of stray depth.
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(o.min_conf_thr)))
    masks = scene.get_masks()
    dmaps = scene.get_depthmaps()
    bgs = [d > o.bg_trd * (torch.max(d[40:-40, :]) + torch.min(d[40:-40, :])) for d in dmaps]
    masks = to_numpy([m + b for m, b in zip(masks, bgs)])
    imgs = np.array(scene.imgs)

    phi, theta, rr = read_traj(traj_file)
    camera_traj, num_views = generate_traj_txt(
        c2ws[k:k + 1], H, W, focals[k:k + 1], principal_points[k:k + 1],
        phi, theta, rr, o.video_length, pvd.device, viz_traj=True, save_dir=o.save_dir)

    render, _ = pvd.run_render(pcd, imgs, masks, H, W, camera_traj, num_views)
    render = F.interpolate(render.permute(0, 3, 1, 2), size=(576, 1024),
                           mode="bilinear", align_corners=False).permute(0, 2, 3, 1)
    render[0] = pvd.img_ori[k]
    if phi[-1] == 0. and theta[-1] == 0. and rr[-1] == 0.:
        render[-1] = pvd.img_ori[k]

    save_video(render, os.path.join(o.save_dir, "render0.mp4"))
    save_pointcloud_with_normals(imgs, pcd, msk=masks,
                                 save_path=os.path.join(o.save_dir, "pcd0.ply"),
                                 mask_pc=True, reduce_pc=False)
    diffusion = pvd.run_diffusion(render)
    save_video((diffusion + 1.0) / 2.0, os.path.join(o.save_dir, "diffusion0.mp4"))
    return diffusion


def base_args(image_dir, traj, elevation, seed):
    return [
        "--image_dir", str(image_dir),
        "--traj_txt", str(PROJ / "trajs" / f"{traj}.txt"),
        "--mode", "single_view_txt",
        "--out_dir", str(PROJ / "outputs"),
        "--center_scale", "1.",
        "--elevation", str(elevation),
        "--seed", str(seed),
        "--bg_trd", "0.2",
        "--dpt_trd", "0.9",
        "--ckpt_path", "./checkpoints/model.ckpt",
        "--config", "configs/inference_pvd_1024.yaml",
        "--model_path", "./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "--ddim_steps", "50",
        "--video_length", "25",
        "--device", "cuda:0",
        "--height", "576", "--width", "1024",
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--trajs", nargs="*", default=["orbit360", "orbit45"])
    args = ap.parse_args()

    objs = [(n, e) for n, e in C.load(args.only) if (VIEWS / n).is_dir()]
    plan = [(n, e, t) for n, e in objs for t in args.trajs]
    if not plan:
        raise SystemExit("nothing to run")

    parser = get_parser()
    opts = parser.parse_args(base_args(VIEWS / plan[0][0], plan[0][2], plan[0][1], args.seed))
    opts.save_dir = str(PROJ / "outputs" / f"{plan[0][0]}__{plan[0][2]}")
    os.makedirs(opts.save_dir, exist_ok=True)

    t0 = time.time()
    print(f"=== building ViewCrafter for {len(plan)} runs ===", flush=True)
    pvd = ViewCrafter(opts)
    print(f"=== ready in {time.time() - t0:.0f}s ===", flush=True)

    log, last = [], None
    for name, elevation, traj in plan:
        exp = f"{name}__{traj}"
        save_dir = PROJ / "outputs" / exp
        save_dir.mkdir(parents=True, exist_ok=True)
        o = parser.parse_args(base_args(VIEWS / name, traj, elevation, args.seed))
        o.save_dir = str(save_dir)
        pvd.opts = o

        t = time.time()
        print(f"\n=== {exp}  (8 views, elevation {elevation}) ===", flush=True)
        if last != name:                       # the cloud only changes per object
            torch.cuda.empty_cache()
            pvd.images, pvd.img_ori = pvd.load_initial_dir(image_dir=str(VIEWS / name))
            pvd.run_dust3r(input_images=pvd.images, clean_pc=True)
            last = name
        seed_everything(o.seed)
        orbit_from_multiview(pvd, o.traj_txt)
        dt = time.time() - t
        print(f"=== {exp} done in {dt:.0f}s ===", flush=True)
        log.append({"experiment": exp, "object": name, "trajectory": traj,
                    "views": len(list((VIEWS / name).glob("*.png"))),
                    "elevation": elevation, "seed": args.seed,
                    "seconds": round(dt, 1), "output_dir": str(save_dir)})
        rec = PROJ / "outputs" / "runs.json"
        prev = json.loads(rec.read_text()) if rec.exists() else []
        merged = {r["experiment"]: r for r in prev}
        merged.update({r["experiment"]: r for r in log})
        rec.write_text(json.dumps(sorted(merged.values(), key=lambda r: r["experiment"]), indent=2))

    print(f"\n=== {len(log)} runs, {time.time() - t0:.0f}s total ===")


if __name__ == "__main__":
    main()
