#!/usr/bin/env python3
"""Turn ShapeR's output meshes into 360-degree turntable videos.

The point of this experiment is a comparison, and the two halves were not
comparable to look at: ViewCrafter produced video, ShapeR produced a .glb that
nobody can see without opening a 3D viewer. This renders ShapeR's predicted mesh
and the reference mesh side by side on one turntable, so both models' answers to
"show me this object from all sides" can be watched the same way.

The turntable is a real 360-degree orbit, which is the whole difference: ShapeR's
far side comes from a mesh it actually predicted, so it is the same object at
every angle by construction, with nothing invented per frame.

Colours match ShapeR's own visualisations -- amber prediction, green reference.

    python scripts/render_shaper_turntables.py ADT1292__chair BNB1045__sofa
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio.v2 as imageio  # noqa: E402
import numpy as np            # noqa: E402
import pyrender               # noqa: E402
import trimesh                # noqa: E402

SHAPER_OUT = Path("/home/isha.tariq/shapeR_testing/outputs/eval_balance/per_object")
PRED_RGB = (255, 192, 0)
GT_RGB = (34, 139, 34)
RES = 640
FRAMES = 36          # 10 degrees per frame
ELEV_DEG = 22.0      # looks down on the object, matching the Aria reference views


def load_normalised(path):
    """Load a .glb as one mesh, centred at the origin and scaled to a unit box, so
    every object turns at the same apparent size regardless of its real metres."""
    m = trimesh.load(path, force="mesh")
    m = trimesh.Trimesh(vertices=np.asarray(m.vertices, dtype=np.float64),
                        faces=np.asarray(m.faces), process=False)
    c = m.bounds.mean(axis=0)
    m.apply_translation(-c)
    scale = float(np.max(m.extents))
    if scale > 0:
        m.apply_scale(1.0 / scale)
    return m


def camera_pose(azim_deg, elev_deg, radius):
    """Standard look-at aimed at the origin, in pyrender's convention: the camera
    looks down its own -Z, so the pose's third column is the direction back to the
    eye. Building the basis in the other order gives a left-handed frame, which
    renders as the object tumbling in the image plane instead of turning."""
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    eye = np.array([radius * np.cos(e) * np.sin(a),
                    radius * np.sin(e),
                    radius * np.cos(e) * np.cos(a)])
    z = eye / np.linalg.norm(eye)
    x = np.cross(np.array([0.0, 1.0, 0.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0], pose[:3, 1], pose[:3, 2], pose[:3, 3] = x, y, z, eye
    return pose


def fit_radius(mesh, yfov=np.pi / 4.0, fill=0.72):
    """Distance at which the object's turning circle fills `fill` of the frame.
    Computed from the mesh's own radius rather than fixed, so a bed and a mug both
    arrive at the same apparent size."""
    r = float(np.linalg.norm(mesh.vertices, axis=1).max())
    return r / (fill * np.tan(yfov / 2.0))


def render_turntable(mesh, rgb, renderer, radius):
    mesh = mesh.copy()
    # Z-up meshes rendered in a Y-up scene lie on their side; rotate once here.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    # vertex_colors + smooth shading, matching ShapeR's own visualisations. Setting
    # face_colors instead renders as a flat silhouette and hides the geometry.
    mesh.visual.vertex_colors = np.array(list(rgb) + [255], dtype=np.uint8)
    frames = []
    for i in range(FRAMES):
        scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.12, 0.12, 0.12])
        scene.add(pyrender.Mesh.from_trimesh(mesh))
        cam = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
        pose = camera_pose(i * 360.0 / FRAMES, ELEV_DEG, radius)
        scene.add(cam, pose=pose)
        # Lights are placed off the camera axis and travel with it. A light sitting
        # on the camera flattens the mesh into a silhouette, which hides exactly the
        # geometry these videos exist to show.
        az = i * 360.0 / FRAMES
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=4.0),
                  pose=camera_pose(az + 38, 42, radius))
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.8),
                  pose=camera_pose(az - 65, 8, radius))
        scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=1.0),
                  pose=camera_pose(az + 175, 30, radius))
        color, _ = renderer.render(scene)
        frames.append(color[..., :3])
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+")
    ap.add_argument("--out_dir", type=Path,
                    default=Path("/home/isha.tariq/viewcrafter_orbits/shared/turntables"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    renderer = pyrender.OffscreenRenderer(RES, RES)
    for name in args.samples:
        pred = load_normalised(SHAPER_OUT / f"{name}.glb")
        gt = load_normalised(SHAPER_OUT / f"GT__{name}.glb")
        # One radius for both halves, so prediction and reference are drawn at the
        # same scale and any size disagreement between them stays visible.
        radius = max(fit_radius(pred), fit_radius(gt))
        fp = render_turntable(pred, PRED_RGB, renderer, radius)
        fg = render_turntable(gt, GT_RGB, renderer, radius)
        pair = [np.concatenate([a, b], axis=1) for a, b in zip(fp, fg)]
        out = args.out_dir / f"{name}__turntable.mp4"
        imageio.mimsave(out, pair, fps=12, quality=8, macro_block_size=1)
        print(f"{name:24s} pred {len(pred.faces):7d} faces | gt {len(gt.faces):7d} faces "
              f"-> {out.name}")
    renderer.delete()
    print(f"\nwrote {len(args.samples)} turntables to {args.out_dir}")


if __name__ == "__main__":
    main()
