#!/usr/bin/env python3
"""A controlled version of "a lamp in front of the sofa".

    /home/isha.tariq/.conda/envs/shaper/bin/python scripts/build_occluded.py

Takes exp1's clean 8-view scenes and adds a synthetic occluder: a magenta ball
floating just in front of a well-photographed spot on the object, projected with
the correct geometry into the input views. Two conditions per object:

  occ1    the ball is painted into ONE view only (the one where it is largest).
          The other seven see the surface behind it. If the model cross-checks
          views, it should treat the ball as an inconsistency and draw the
          surface; if it copies what it is given, the ball turns up in the video.
  occall  the ball is painted into EVERY view that can see that spot. Now no
          photograph shows the surface behind it. Whatever the model draws there
          is invented -- this is the generation test with a known answer, since
          the clean run shows what is actually there.

Magenta (255, 0, 255) is used because nothing in these scenes is remotely that
colour, so "did the ball get drawn?" is a pixel count, not a judgement.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp2/scripts")
from observability import (  # noqa: E402
    E1, E2, OUT, SHAPER_DATA, K_of, c2w_cv, render_depth, vertex_seen,
)
import pickle  # noqa: E402

import trimesh  # noqa: E402

MAGENTA = np.array([255, 0, 255], np.uint8)
BALL_R = 0.11          # radius as a fraction of the mesh diagonal
BALL_LIFT = 0.6        # centre pushed off the surface by this * radius, toward the cameras


def ball_mask(K, c2w, centre, r, depth):
    """Pixels whose ray passes within r of the ball centre, with the ball in front."""
    R, C = c2w[:3, :3], c2w[:3, 3]
    ys, xs = np.mgrid[0:OUT, 0:OUT]
    dirs = np.stack([(xs - K[0, 2]) / K[0, 0], (ys - K[1, 2]) / K[1, 1], np.ones_like(xs, float)], -1)
    dirs = dirs @ R.T                                   # rays in world
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)
    oc = centre - C
    t = dirs @ oc                                       # closest approach along the ray
    closest = C + dirs * t[..., None]
    dist = np.linalg.norm(closest - centre, axis=-1)
    hit = (dist < r) & (t > 0)
    # depth of the ball's front surface along the camera z axis
    t_hit = t - np.sqrt(np.clip(r * r - dist * dist, 0, None))
    z_ball = t_hit * (dirs @ R[:, 2])
    in_front = (depth == 0) | (z_ball < depth)
    return hit & in_front


def build(name):
    src = E1 / "inputs" / name
    tf = json.loads((src / "transforms.json").read_text())["frames"]
    meta = json.loads((src / "meta.json").read_text())
    n_in = meta["views"]
    d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
    mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                           faces=np.asarray(d["mesh_faces"]), process=False)
    verts = np.asarray(mesh.vertices)
    diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))

    # The spot: seen by the most views, so hiding it is a real loss of evidence,
    # and near the object's projected centre so it is not a stray leg.
    cams = [(K_of(fr), c2w_cv(fr)) for fr in tf[:n_in]]
    count = np.zeros(len(verts), int)
    for K, c2w in cams:
        count += vertex_seen(mesh, verts, K, c2w)
    centre_obj = mesh.bounds.mean(axis=0)
    score = count - 0.5 * np.linalg.norm(verts - centre_obj, axis=1) / diag * count.max()
    vi = int(np.argmax(score))
    spot = verts[vi]
    # lift the ball toward the mean camera position so it floats in front
    cam_mean = np.mean([c[:3, 3] for _, c in cams], axis=0)
    toward = cam_mean - spot; toward /= np.linalg.norm(toward)
    r = BALL_R * diag
    centre = spot + toward * (BALL_LIFT * r)

    masks, areas = [], []
    for K, c2w in cams:
        depth = render_depth(mesh, K, OUT, OUT, c2w)
        m = ball_mask(K, c2w, centre, r, depth)
        masks.append(m); areas.append(int(m.sum()))
    biggest = int(np.argmax(areas))

    for cond, which in [("occ1", [biggest]), ("occall", [i for i, a in enumerate(areas) if a > 0])]:
        dst = E2 / "inputs" / f"{name}__{cond}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        for i in which:
            p = dst / "images" / f"{i:03d}.png"
            im = np.asarray(Image.open(p).convert("RGB")).copy()
            im[masks[i]] = MAGENTA
            Image.fromarray(im).save(p)
        (dst / "occluder.json").write_text(json.dumps({
            "object": name, "condition": cond, "ball_centre": centre.tolist(),
            "ball_radius": r, "painted_views": which,
            "ball_pixels_per_view": areas, "views_seeing_spot": int(count[vi])}, indent=2))
        print(f"{name:22s} {cond:7s} ball r={r:.3f} m painted in views {which} "
              f"(spot seen by {count[vi]}/{n_in} views)")


if __name__ == "__main__":
    names = sys.argv[1:] or [r["object"] for r in
                             json.loads((E1 / "inputs/manifest.json").read_text())]
    for n in names:
        build(n)
    man = [{"scene": p.name} for p in sorted((E2 / "inputs").glob("*__occ*"))]
    (E2 / "inputs" / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"\n{len(man)} scenes")
