#!/usr/bin/env python3
"""exp4: VGGT and Pi3 clouds from the same SEVA orbit, side by side against the SLAM points.
Draw ShapeR's SLAM points and VGGT's points over the same reference mesh.

One sheet per object: three rows (SLAM points, VGGT from real views, VGGT from the
generated 360-degree orbit) by three viewpoints, all over the green reference
mesh. The rows share a camera, so what changes down a column is only which cloud
is drawn.

Points are drawn as instanced spheres rather than GL points. A one-pixel point is
invisible at this size, and more importantly it hides the thing the figure exists
to show: the SLAM cloud is ~2-3k points and the VGGT cloud is ~200k, so the
density difference has to be legible rather than saturated. VGGT is thinned to a
fixed count and given a smaller sphere so the comparison is about *where* the
points are, not who can paint more of the screen.

    python scripts/render_comparison.py
"""
import argparse
import json
import sys
import os
import pickle
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np      # noqa: E402
import pyrender         # noqa: E402
import trimesh          # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/shared/scripts")
import objects as OBJECTS_CFG  # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
EXP4 = Path("/home/isha.tariq/viewcrafter_orbits/experiment4/outputs")
OBJECTS = OBJECTS_CFG.names()
# ASCII only: the labels are drawn with PIL's built-in bitmap font, which has no
# em-dash or degree sign and silently renders them as a missing-glyph box.
ROWS = [("slam", "ShapeR SLAM points (real)"),
        ("seva360f", "VGGT from the SEVA orbit (80 frames)"),
        ("found", "SLAM points the VGGT cloud reached (teal) and missed (red)"),
        ("seva360fp", "Pi3 from the same SEVA orbit (80 frames)"),
        ("found_pi3", "SLAM points the Pi3 cloud reached (teal) and missed (red)")]
AZIMUTHS = (25, 145, 265)
RES = 460
ELEV = 20.0
MESH_RGB = (34, 139, 34)
SLAM_RGB = (20, 128, 122)      # the teal ShapeR uses for SLAM points in its own figures
VGGT_RGB = (255, 176, 0)
VGGT_DRAW = 9000               # thinned so density does not become opacity


def camera_pose(az, el, r):
    a, e = np.radians(az), np.radians(el)
    eye = np.array([r * np.cos(e) * np.sin(a), r * np.sin(e), r * np.cos(e) * np.cos(a)])
    z = eye / np.linalg.norm(eye)
    x = np.cross([0.0, 1.0, 0.0], z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    P = np.eye(4); P[:3, 0], P[:3, 1], P[:3, 2], P[:3, 3] = x, y, z, eye
    return P


def point_mesh(points, radius, rgb):
    sphere = trimesh.creation.uv_sphere(radius=radius, count=[6, 6])
    sphere.visual.vertex_colors = np.array(list(rgb) + [255], np.uint8)
    poses = np.tile(np.eye(4), (len(points), 1, 1))
    poses[:, :3, 3] = points
    return pyrender.Mesh.from_trimesh(sphere, poses=poses, smooth=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=EXP4)
    args = ap.parse_args()
    renderer = pyrender.OffscreenRenderer(RES, RES)
    rng = np.random.default_rng(0)

    for obj in OBJECTS:
        d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        c = mesh.bounds.mean(axis=0)

        # Z-up model coordinates into a Y-up scene, once, for everything.
        Rz = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh = mesh.copy(); mesh.apply_translation(-c); mesh.apply_transform(Rz)
        mesh.visual.vertex_colors = np.array(list(MESH_RGB) + [255], np.uint8)
        to_scene = lambda p: (Rz[:3, :3] @ (p - c).T).T   # noqa: E731

        clouds = {"slam": (to_scene(slam), SLAM_RGB, diag * 0.006)}
        # SLAM points coloured by whether the SEVA cloud came within 2% of them
        for key, cond in (("found", "seva360f"), ("found_pi3", "seva360fp")):
            fnd = args.out_dir / f"{obj}__{cond}__slam_to_vggt.npy"
            if fnd.exists():
                hit = np.load(fnd) < 0.02 * diag
                clouds[key] = (to_scene(slam), None, diag * 0.006, hit)
        for cond in ("seva360f", "seva360fp"):
            f = args.out_dir / f"{obj}__{cond}__aligned_to_slam.npy"
            if not f.exists():
                continue
            p = np.load(f).astype(np.float64)
            if len(p) > VGGT_DRAW:
                p = p[rng.choice(len(p), VGGT_DRAW, replace=False)]
            clouds[cond] = (to_scene(p), VGGT_RGB, diag * 0.004)

        radius = float(np.linalg.norm(mesh.vertices, axis=1).max()) / (0.60 * np.tan(np.pi / 8))
        panels = []
        for key, _ in ROWS:
            if key not in clouds:
                panels.append([Image.new("RGB", (RES, RES), (255, 255, 255))] * len(AZIMUTHS))
                continue
            pts, rgb, rad = clouds[key][:3]
            hit = clouds[key][3] if len(clouds[key]) > 3 else None
            row = []
            for az in AZIMUTHS:
                scene = pyrender.Scene(bg_color=[255, 255, 255, 255],
                                       ambient_light=[0.42, 0.42, 0.42])
                scene.add(pyrender.Mesh.from_trimesh(mesh))
                if hit is None:
                    scene.add(point_mesh(pts, rad, rgb))
                else:
                    if hit.any(): scene.add(point_mesh(pts[hit], rad, SLAM_RGB))
                    if (~hit).any(): scene.add(point_mesh(pts[~hit], rad, (200, 40, 40)))
                pose = camera_pose(az, ELEV, radius)
                scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=pose)
                scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.2),
                          pose=camera_pose(az + 35, 40, radius))
                scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.4),
                          pose=camera_pose(az - 70, 10, radius))
                color, _ = renderer.render(scene)
                row.append(Image.fromarray(color[..., :3]))
            panels.append(row)

        LAB = 30
        sheet = Image.new("RGB", (RES * len(AZIMUTHS), (RES + LAB) * len(ROWS)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for i, (key, label) in enumerate(ROWS):
            n = len(clouds[key][0]) if key in clouds else 0
            dr.text((10, i * (RES + LAB) + 10),
                    "%s  -  %s points drawn" % (label, format(n, ",")), fill=(60, 66, 76))
            for j, im in enumerate(panels[i]):
                sheet.paste(im, (RES * j, i * (RES + LAB) + LAB))
        out = Path("/home/isha.tariq/stable_virtual_camera/exp4/figs") / f"COMPARE_PI3__{obj}.jpg"
        sheet.save(out, quality=86)
        print(f"{obj:20s} -> {out.name}   slam {len(slam)} pts, diagonal {diag:.3f} m")
    renderer.delete()


if __name__ == "__main__":
    main()
