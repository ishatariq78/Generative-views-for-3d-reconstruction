#!/usr/bin/env python3
"""The two levers side by side: same generated orbit, handed back as points vs as images.

    ~/.conda/envs/shaper/bin/python scripts/render_points_vs_views.py --objects <obj> ...

Row 1 is the 3D cloud reconstructed from the orbit -- the geometry route's input.
Rows 2-5 are the reference and the three meshes (baseline, points-conditioned,
views-conditioned) from three sides, at the 16-view budget where both routes were run.

figs/pv__<obj>.jpg
"""
import argparse, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from PIL import Image, ImageDraw

E = Path("/home/isha.tariq/seva_to_shaper")
MD = E / "outputs/seeded/s0/per_object"
T, PAD, LAB = 250, 8, 30
AZ = [25, 145, 265]


def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def views_of(mesh, colour):
    c = mesh.bounds.mean(0); d = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    out = []
    for az in AZ:
        a = np.radians(az)
        eye = c + d * 1.6 * np.array([np.cos(a) * 0.82, np.sin(a) * 0.82, 0.46])
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.36] * 3)
        m = mesh.copy()
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(colour) * 255), 255], np.uint8))
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        pose = look_at(eye, c)
        sc.add(pyrender.PerspectiveCamera(yfov=np.radians(42)), pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
        r = pyrender.OffscreenRenderer(T, T)
        rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        al = rgba[..., 3:4].astype(np.float32) / 255.0
        out.append((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8))
    return out


def strip(tiles, title):
    s = np.concatenate([x for t in tiles for x in (t, np.full((T, PAD, 3), 255, np.uint8))][:-1], 1)
    head = Image.new("RGB", (s.shape[1], LAB), (255, 255, 255))
    ImageDraw.Draw(head).text((2, 8), title, fill=(25, 35, 55))
    return np.concatenate([np.asarray(head), s], 0)


def sc(obj, cond):
    f = MD / f"{obj}__{cond}.json"
    if not f.exists(): return ""
    m = json.load(open(f))["metrics"]
    return f"   CD {m['chamfer']*100:.2f}  F1 {m['f-score']:.3f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", required=True)
    args = ap.parse_args()
    rowspec = [("GT__%s__slam", (0.45, 0.65, 0.4), "REFERENCE"),
               ("%s__slam", (0.80, 0.80, 0.84), "BASELINE  - the capture's own views"),
               ("%s__pi3", (0.72, 0.55, 0.78), "ORBIT AS POINTS  - the orbit reconstructed in 3D, fed as the point cloud"),
               ("%s__views", (0.93, 0.72, 0.25), "ORBIT AS IMAGES  - the same orbit fed as extra cameras")]
    for obj in args.objects:
        rows = []
        for pat, col, title in rowspec:
            p = MD / (pat % obj + ".glb")
            if not p.exists():
                print(f"  !! {obj}: missing {p.name}"); continue
            cond = pat.split("__")[-1] if "%s__" in pat else "slam"
            lbl = title + (sc(obj, cond) if "GT__" not in pat else "")
            rows.append(strip(views_of(trimesh.load(p, force="mesh"), col), lbl))
        if len(rows) < 3:
            print(f"!! {obj}: too few rows"); continue
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        sheet = np.concatenate([x for r in rows for x in (r, np.full((PAD * 2, w, 3), 255, np.uint8))][:-1], 0)
        Image.fromarray(sheet).save(E / f"figs/pv__{obj}.jpg", quality=90)
        print("wrote", E / f"figs/pv__{obj}.jpg", flush=True)
