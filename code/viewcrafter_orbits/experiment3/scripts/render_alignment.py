#!/usr/bin/env python3
"""Render each aligned VGGT cloud against the object's reference mesh.

The scores are only trustworthy if the alignment they rest on is trustworthy, and
a similarity fit is exactly the kind of thing that can produce a plausible number
from a nonsense pose. This draws the answer: the reference mesh in green, VGGT's
aligned points in amber, from four viewpoints. If the points wrap the mesh, the
fit is real; if they sit in a knot beside it, the number means nothing.

One sheet per frame set, into outputs/<set>/alignment.jpg.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np      # noqa: E402
import pyrender         # noqa: E402
import trimesh          # noqa: E402
from PIL import Image   # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment3")
SHAPER_MESHES = Path("/home/isha.tariq/shapeR_testing/outputs/eval_balance/per_object")
RES = 520
AZIMUTHS = (0, 90, 180, 270)
ELEV = 20.0
GT_RGB = (34, 139, 34)
PT_RGB = (255, 176, 0)
MAX_POINTS = 70000


def camera_pose(azim_deg, elev_deg, radius):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    eye = np.array([radius * np.cos(e) * np.sin(a),
                    radius * np.sin(e),
                    radius * np.cos(e) * np.cos(a)])
    z = eye / np.linalg.norm(eye)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.eye(4)
    P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, eye
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    args = ap.parse_args()

    sets = sorted(p for p in args.out_dir.iterdir()
                  if p.is_dir() and (p / "points_aligned.npy").exists())
    if not sets:
        raise SystemExit(f"no aligned clouds under {args.out_dir} -- run align_and_score.py first")

    renderer = pyrender.OffscreenRenderer(RES, RES)
    rng = np.random.default_rng(0)
    for s in sets:
        obj = s.name.rsplit("__", 1)[0]
        mesh = trimesh.load(SHAPER_MESHES / f"GT__{obj}.glb", force="mesh")
        pts = np.load(s / "points_aligned.npy").astype(np.float64)
        if len(pts) > MAX_POINTS:
            pts = pts[rng.choice(len(pts), MAX_POINTS, replace=False)]

        # Both are already in the mesh's metric frame; centre the pair for framing.
        c = mesh.bounds.mean(axis=0)
        mesh = mesh.copy()
        mesh.apply_translation(-c)
        pts = pts - c
        # Z-up meshes in a Y-up scene lie on their side.
        Rz = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(Rz)
        pts = (Rz[:3, :3] @ pts.T).T
        mesh.visual.vertex_colors = np.array(list(GT_RGB) + [255], np.uint8)

        radius = float(np.linalg.norm(mesh.vertices, axis=1).max()) / (0.62 * np.tan(np.pi / 8))
        panels = []
        for az in AZIMUTHS:
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255],
                                   ambient_light=[0.35, 0.35, 0.35])
            scene.add(pyrender.Mesh.from_trimesh(mesh))
            scene.add(pyrender.Mesh.from_points(
                pts, colors=np.tile(np.array(PT_RGB, np.uint8), (len(pts), 1))))
            cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
            pose = camera_pose(az, ELEV, radius)
            scene.add(cam, pose=pose)
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.5),
                      pose=camera_pose(az + 35, 40, radius))
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.5),
                      pose=camera_pose(az - 70, 10, radius))
            color, _ = renderer.render(scene)
            panels.append(Image.fromarray(color[..., :3]))

        sheet = Image.new("RGB", (RES * len(panels), RES), (255, 255, 255))
        for i, p in enumerate(panels):
            sheet.paste(p, (RES * i, 0))
        sheet.save(s / "alignment.jpg", quality=88)
        print(f"{s.name:30s} {len(pts):7d} points rendered -> alignment.jpg")
    renderer.delete()


if __name__ == "__main__":
    main()
