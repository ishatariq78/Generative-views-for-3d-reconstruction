#!/usr/bin/env python3
"""How much of what SEVA draws was ever photographed?

    /home/isha.tariq/.conda/envs/shaper/bin/python scripts/observability.py

Answers "is it reconstructing or generating?" without generating anything new.
We have the object's true mesh and the exact pose of each of exp1's 8 input
views, so every vertex can be classified: *observed* if at least one input
camera sees it (in frame, front-facing, not hidden behind the object itself) or
*never observed*. Then, from each of the 80 target cameras of the orbit, the
mesh is rendered with those two classes as colours and we count: of the object
pixels in this frame, what fraction is surface no photograph ever contained?

Everything the model draws in that fraction cannot be reconstruction -- there is
nothing to reconstruct from. It is generation. Output per object: the surface
fraction observed, the per-frame unobserved fraction along the orbit, and a
turntable rendering (green = photographed, orange = never) to put beside SEVA's
own output for the same cameras.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, "/home/isha.tariq/shapeR_testing/ShapeR")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment1/scripts")
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment2/scripts")
sys.path.insert(0, "/home/isha.tariq/stable_virtual_camera/exp1/scripts")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pickle  # noqa: E402

import pyrender  # noqa: E402
from extract_reference_views import SHAPER_DATA  # noqa: E402
from build_object_only_views import _renderer  # noqa: E402
from occlusion import render_depth  # noqa: E402

E1 = Path("/home/isha.tariq/stable_virtual_camera/exp1")
E2 = Path("/home/isha.tariq/stable_virtual_camera/exp2")
OUT = 576
CV2GL = np.diag([1.0, -1.0, -1.0, 1.0])
DEPTH_TOL = 0.010
SEEN, UNSEEN, BG = (70, 160, 90), (230, 120, 50), (0, 0, 0)


def K_of(fr):
    K = np.eye(3)
    K[0, 0], K[1, 1] = fr["fl_x"] * fr["w"], fr["fl_y"] * fr["h"]
    K[0, 2], K[1, 2] = fr["cx"] * fr["w"], fr["cy"] * fr["h"]
    return K


def c2w_cv(fr):
    return np.array(fr["transform_matrix"]) @ CV2GL


def vertex_seen(mesh, verts, K, c2w):
    """Which vertices this camera actually sees (depth-tested against the mesh)."""
    depth = render_depth(mesh, K, OUT, OUT, c2w)
    R, C = c2w[:3, :3], c2w[:3, 3]
    cam = (verts - C) @ R
    z = cam[:, 2]
    ok = z > 1e-6
    u = np.full(len(verts), -1.0); v = np.full(len(verts), -1.0)
    u[ok] = cam[ok, 0] / z[ok] * K[0, 0] + K[0, 2]
    v[ok] = cam[ok, 1] / z[ok] * K[1, 1] + K[1, 2]
    ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    inside = ok & (ui >= 0) & (ui < OUT) & (vi >= 0) & (vi < OUT)
    seen = np.zeros(len(verts), bool)
    ii = np.nonzero(inside)[0]
    d = depth[vi[ii], ui[ii]]
    seen[ii] = (d > 0) & (np.abs(d - z[ii]) < DEPTH_TOL)
    return seen


def render_classes(mesh_col, K, c2w):
    """Flat render of a vertex-coloured mesh (ambient only, so no shading)."""
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[1.0, 1.0, 1.0])
    scene.add(pyrender.Mesh.from_trimesh(mesh_col, smooth=False))
    cam = pyrender.IntrinsicsCamera(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                                    znear=0.01, zfar=100.0)
    pose = np.array(c2w @ CV2GL, np.float64); pose[3] = [0, 0, 0, 1]
    scene.add(cam, pose=pose)
    color, depth = _renderer(OUT, OUT).render(scene)
    return np.asarray(color)[..., :3], np.asarray(depth) > 0


def main(names):
    report = []
    for name in names:
        tf = json.loads((E1 / "inputs" / name / "transforms.json").read_text())["frames"]
        meta = json.loads((E1 / "inputs" / name / "meta.json").read_text())
        n_in = meta["views"]
        d = pickle.load(open(SHAPER_DATA / f"{name}.pkl", "rb"))
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        verts = np.asarray(mesh.vertices)

        seen = np.zeros(len(verts), bool)
        per_view = []
        for fr in tf[:n_in]:
            s = vertex_seen(mesh, verts, K_of(fr), c2w_cv(fr))
            per_view.append(float(s.mean()))
            seen |= s
        # area-weighted: a face counts as observed if its vertices mostly are
        fa = mesh.area_faces
        face_seen = seen[mesh.faces].mean(axis=1) >= 0.5
        surf_obs = float((fa * face_seen).sum() / fa.sum())

        col = np.where(seen[:, None], np.array(SEEN), np.array(UNSEEN)).astype(np.uint8)
        mc = mesh.copy()
        mc.visual.vertex_colors = np.concatenate([col, np.full((len(col), 1), 255, np.uint8)], 1)

        odir = E2 / "outputs" / "observability" / name
        (odir / "frames").mkdir(parents=True, exist_ok=True)
        per_frame = []
        for j, fr in enumerate(tf[n_in:]):
            rgb, m = render_classes(mc, K_of(fr), c2w_cv(fr))
            # classify pixels by which colour they are nearer to
            du = np.linalg.norm(rgb.astype(float) - np.array(UNSEEN), axis=-1)
            ds = np.linalg.norm(rgb.astype(float) - np.array(SEEN), axis=-1)
            unseen_px = m & (du < ds)
            per_frame.append(float(unseen_px.sum() / max(m.sum(), 1)))
            out = np.full((OUT, OUT, 3), 245, np.uint8); out[m] = rgb[m]
            Image.fromarray(out).save(odir / "frames" / f"{j:03d}.png")

        rec = {"object": name, "views": n_in,
               "surface_observed_fraction": round(surf_obs, 4),
               "vertices_observed_fraction": round(float(seen.mean()), 4),
               "per_view_vertex_fraction": [round(v, 4) for v in per_view],
               "orbit_unobserved_fraction_mean": round(float(np.mean(per_frame)), 4),
               "orbit_unobserved_fraction_max": round(float(np.max(per_frame)), 4),
               "frames_mostly_unobserved": int(sum(f > 0.5 for f in per_frame)),
               "per_frame_unobserved_fraction": [round(f, 4) for f in per_frame]}
        (odir / "observability.json").write_text(json.dumps(rec, indent=2))
        report.append(rec)
        print(f"{name:22s} surface photographed {surf_obs:5.1%}   along the orbit: "
              f"{np.mean(per_frame):5.1%} of visible surface never photographed "
              f"(max {np.max(per_frame):5.1%}, {rec['frames_mostly_unobserved']}/80 frames >50%)")
    (E2 / "outputs" / "observability.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    names = sys.argv[1:] or [r["object"] for r in
                             json.loads((E1 / "inputs/manifest.json").read_text())]
    main(names)
