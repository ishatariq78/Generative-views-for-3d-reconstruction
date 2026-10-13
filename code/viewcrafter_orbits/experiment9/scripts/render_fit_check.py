#!/usr/bin/env python3
"""The same cloud over the same SLAM points under three fits, one row per object.

  old      the two-axis grid every experiment through 2026-09-14 was scored with
  so3      the full-rotation grid applied 2026-09-16 -- can find upright, does not prefer it
  upright  the camera-anchored fit -- the cloud's own up held to +z

Cloud in orange, SLAM points in teal, the reference mesh faint underneath. The
up_z under each panel is the fit-independent check: the cloud's camera-ring
normal pushed through the fitted rotation, +1 upright, -1 on its head.

    python scripts/render_fit_check.py [--cond p3_level]
"""
import argparse
import json
import os
import pickle
import sys
import zlib
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np                              # noqa: E402
import pyrender                                 # noqa: E402
import trimesh                                  # noqa: E402
from PIL import Image, ImageDraw                # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "experiment4" / "scripts"))
sys.path.insert(0, str(PROJ / "experiment3" / "scripts"))
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
from render_comparison import camera_pose, point_mesh          # noqa: E402
from render_explainer_figures import font, font_bold, trim, place   # noqa: E402
from align_and_score import load_ply                            # noqa: E402
import objects as C                                             # noqa: E402

SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
EXP3, EXP4 = PROJ / "experiment3/outputs", PROJ / "experiment4/outputs"
FITS = [("old", EXP4 / "oldfit" / "npy", "old fit  (two-axis grid)"),
        ("so3", EXP4 / "so3fit" / "npy", "SO(3) grid  (free search)"),
        ("upright", EXP4, "upright  (camera-anchored)")]
OBJ = C.names()
RES, CELL = 460, 430
MESH_RGB, SLAM_RGB, CLOUD_RGB = (200, 210, 200), (20, 128, 122), (235, 104, 52)
INK, INK2, MUTED, GOOD, BAD = (20, 24, 30), (74, 82, 94), (140, 146, 154), (26, 175, 122), (200, 58, 44)


def kabsch_rotation(src, dst):
    a, b = src - src.mean(0), dst - dst.mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1; R = Vt.T @ U.T
    return R


def up_of(set_dir, pts_raw):
    z = np.load(set_dir / "vggt.npz"); E = z["extrinsic"]
    cams = np.stack([-E[i, :, :3].T @ E[i, :, 3] for i in range(len(E))])
    ctr = cams.mean(0); n = np.linalg.svd(cams - ctr)[2][-1]
    return n * np.sign((ctr - pts_raw.mean(0)) @ n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="p3_level")
    args = ap.parse_args()
    renderer = pyrender.OffscreenRenderer(RES, RES)
    rng = np.random.default_rng(0)
    LABW, TOP = 150, 56
    sheet = Image.new("RGB", (LABW + CELL * 3, TOP + (CELL + 26) * len(OBJ)), (255, 255, 255))
    dr = ImageDraw.Draw(sheet)
    for j, (_, _, title) in enumerate(FITS):
        dr.text((LABW + CELL * j + 12, 14), title, fill=INK, font=font_bold(19))
    report = []
    for i, obj in enumerate(OBJ):
        d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
        mesh = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64),
                               faces=np.asarray(d["mesh_faces"]), process=False)
        slam = np.asarray(d["points_model"].numpy(), np.float64)
        diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
        c = mesh.bounds.mean(axis=0)
        Rz = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
        m = mesh.copy(); m.apply_translation(-c); m.apply_transform(Rz)
        m.visual.vertex_colors = np.array(list(MESH_RGB) + [255], np.uint8)
        to_scene = lambda p: (Rz[:3, :3] @ (p - c).T).T   # noqa: E731
        radius = float(np.linalg.norm(m.vertices, axis=1).max()) / (0.60 * np.tan(np.pi / 8))
        set_dir = EXP3 / f"{obj}__{args.cond}"
        raw = load_ply(set_dir / "points_object.ply")
        up = up_of(set_dir, raw)
        if len(raw) > 120000:
            raw = raw[np.random.default_rng(zlib.crc32(np.ascontiguousarray(raw)))
                      .choice(len(raw), 120000, replace=False)]
        y = TOP + i * (CELL + 26)
        dr.text((12, y + CELL // 2 - 10), obj.split("__")[1], fill=INK, font=font_bold(18))
        for j, (key, ndir, _) in enumerate(FITS):
            f = ndir / f"{obj}__{args.cond}__aligned_to_slam.npy"
            if not f.exists():
                continue
            al = np.load(f).astype(np.float64)
            up_z = float((kabsch_rotation(raw, al) @ up)[2]) if len(al) == len(raw) else float("nan")
            from scipy.spatial import cKDTree
            found = float((cKDTree(al).query(slam)[0] < 0.02 * diag).mean())
            draw = al[rng.choice(len(al), min(9000, len(al)), replace=False)]
            scene = pyrender.Scene(bg_color=[255, 255, 255, 255], ambient_light=[0.45, 0.45, 0.45])
            scene.add(pyrender.Mesh.from_trimesh(m))
            scene.add(point_mesh(to_scene(slam), 0.006 * diag, SLAM_RGB))
            scene.add(point_mesh(to_scene(draw), 0.0035 * diag, CLOUD_RGB))
            pose = camera_pose(25, 20.0, radius)
            scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4.0), pose=pose)
            scene.add(pyrender.DirectionalLight(color=[1, 1, 1], intensity=3.2), pose=camera_pose(60, 40, radius))
            im = Image.fromarray(renderer.render(scene)[0][..., :3])
            sheet.paste(place(trim(im), CELL, CELL - 30), (LABW + CELL * j, y))
            ok = up_z > 0
            dr.text((LABW + CELL * j + 12, y + CELL - 26),
                    "up->z %+.2f   %s     found %.3f" % (up_z, "upright" if ok else "FLIPPED", found),
                    fill=GOOD if ok else BAD, font=font_bold(15))
            report.append({"object": obj, "fit": key, "up_z": round(up_z, 3), "found": round(found, 4)})
    renderer.delete()
    out = PROJ / "experiment9/outputs" / f"fit_check_{args.cond}.jpg"
    sheet.save(out, quality=88)
    (PROJ / "experiment9/outputs" / f"fit_check_{args.cond}.json").write_text(json.dumps(report, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
