#!/usr/bin/env python3
"""Did the model draw the ball, ignore it, or reconstruct what was behind it?

    /home/isha.tariq/.conda/envs/shaper/bin/python scripts/score_occlusion.py

For every target frame of every occluded run we know exactly where the ball
*would* appear if the model believed it was real (same ray-sphere test as the
build, from the target camera). Inside that region we count magenta pixels, and
we compare the pixels against exp1's clean run of the same object -- same seed,
same cameras, same everything except the ball -- so any difference is the ball's
doing. Outside the region is the control: it should barely differ.

Three verdicts per frame:
  copied        magenta dominates the region       -> the ball was believed
  reconstructed region differs little from clean   -> the view(s) were overruled
  invented      not magenta, but not clean either  -> something else was drawn
"""
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp2/scripts")
sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp1/scripts")
import pickle  # noqa: E402

from build_occluded import ball_mask  # noqa: E402
from observability import E1, E2, OUT, SHAPER_DATA, K_of, c2w_cv, render_depth  # noqa: E402
from score_orbits import object_mask  # noqa: E402

MAG_T = 0.35        # region magenta fraction above which the ball counts as drawn
RECON_T = 1.6       # inside/outside difference ratio below which it is "reconstructed"


def is_magenta(img):
    a = np.asarray(img, np.int16)
    return (a[..., 0] > 150) & (a[..., 1] < 110) & (a[..., 2] > 150)


def score(scene):
    obj, cond = scene.rsplit("__", 1)
    occ = json.loads((E2 / "inputs" / scene / "occluder.json").read_text())
    tf = json.loads((E2 / "inputs" / scene / "transforms.json").read_text())["frames"]
    n_in = json.loads((E2 / "inputs" / scene / "meta.json").read_text())["views"]
    gen = E2 / "outputs/demo/img2trajvid" / scene / "samples-rgb"
    clean = E1 / "outputs/demo/img2trajvid" / obj / "samples-rgb"
    if not gen.exists():
        return None
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    centre, r = np.array(occ["ball_centre"]), occ["ball_radius"]

    rows = []
    cdir = E2 / "outputs/crops" / scene; cdir.mkdir(parents=True, exist_ok=True)
    for j, fr in enumerate(tf[n_in:]):
        K, c2w = K_of(fr), c2w_cv(fr)
        depth = render_depth(mesh, K, OUT, OUT, c2w)
        region = ball_mask(K, c2w, centre, r, depth)
        if region.sum() < 200:
            continue                                  # ball not visible from here
        g = np.asarray(Image.open(gen / f"{j:03d}.png").convert("RGB"), np.float32)
        c = np.asarray(Image.open(clean / f"{j:03d}.png").convert("RGB"), np.float32)
        mag = is_magenta(g)
        diff = np.abs(g - c).mean(-1)
        obj_px = object_mask(c) & (depth > 0)
        inside, outside = region, obj_px & ~region
        m_frac = float(mag[region].mean())
        d_in = float(diff[inside].mean())
        d_out = float(diff[outside].mean()) if outside.any() else 0.0
        ratio = d_in / max(d_out, 1.0)
        verdict = ("copied" if m_frac > MAG_T else
                   "reconstructed" if ratio < RECON_T else "invented")
        rows.append({"frame": j, "region_px": int(region.sum()), "magenta_frac": round(m_frac, 4),
                     "diff_inside": round(d_in, 2), "diff_outside": round(d_out, 2),
                     "ratio": round(ratio, 2), "verdict": verdict})
        # crops for the deck: clean | occluded output, tight around the region
        ys, xs = np.nonzero(region); pad = 40
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, OUT)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, OUT)
        pair = np.concatenate([c[y0:y1, x0:x1], g[y0:y1, x0:x1]], axis=1).astype(np.uint8)
        Image.fromarray(pair).save(cdir / f"{j:03d}.png")

    v = [r["verdict"] for r in rows]
    rec = {"scene": scene, "object": obj, "condition": cond, "frames_with_ball_in_view": len(rows),
           "copied": v.count("copied"), "reconstructed": v.count("reconstructed"),
           "invented": v.count("invented"),
           "magenta_frac_mean": round(float(np.mean([r["magenta_frac"] for r in rows])), 4),
           "magenta_frac_max": round(float(np.max([r["magenta_frac"] for r in rows])), 4),
           "diff_inside_mean": round(float(np.mean([r["diff_inside"] for r in rows])), 2),
           "diff_outside_mean": round(float(np.mean([r["diff_outside"] for r in rows])), 2),
           "frames": rows}
    print(f"{obj:22s} {cond:7s} ball in view {len(rows):2d}/80 frames  "
          f"copied {rec['copied']:2d}  reconstructed {rec['reconstructed']:2d}  invented {rec['invented']:2d}  "
          f"magenta max {rec['magenta_frac_max']:.2f}  diff in/out {rec['diff_inside_mean']:.1f}/{rec['diff_outside_mean']:.1f}")
    return rec


if __name__ == "__main__":
    scenes = sys.argv[1:] or [r["scene"] for r in json.loads((E2 / "inputs/manifest.json").read_text())]
    recs = [r for r in (score(s) for s in scenes) if r]
    (E2 / "outputs" / "occlusion_scores.json").write_text(json.dumps(recs, indent=2))
    print(f"\nscored {len(recs)} runs")
