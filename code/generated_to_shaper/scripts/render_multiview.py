#!/usr/bin/env python3
"""Each predicted mesh seen from four sides, beside the reference, plus the orbit it came from.

    ~/.conda/envs/shaper/bin/python scripts/render_multiview.py --objects <obj> ...

One camera flatters a mesh: a shape can look right from the side the capture saw and be
wrong everywhere else, which is exactly the failure this pipeline is meant to fix. So every
mesh is rendered from four azimuths 90 deg apart on the same elevation, reference above and
prediction below, with the generated orbit frames alongside for reference.

figs/multi__<obj>.jpg
"""
import argparse, json, os
from pathlib import Path
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from PIL import Image, ImageDraw

E = Path("/home/isha.tariq/seva_to_shaper")
T, PAD, LAB = 240, 8, 30
AZ = [20, 110, 200, 290]          # four sides, same elevation


def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def views_of(mesh, colour, n=T):
    c = mesh.bounds.mean(0); d = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    out = []
    for az in AZ:
        a = np.radians(az)
        eye = c + d * 1.6 * np.array([np.cos(a) * 0.82, np.sin(a) * 0.82, 0.48])
        sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.36] * 3)
        m = mesh.copy()
        m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(colour) * 255), 255], np.uint8))
        sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
        pose = look_at(eye, c)
        sc.add(pyrender.PerspectiveCamera(yfov=np.radians(42)), pose=pose)
        sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
        r = pyrender.OffscreenRenderer(n, n)
        rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
        al = rgba[..., 3:4].astype(np.float32) / 255.0
        out.append((rgba[..., :3] * al + 255 * (1 - al)).astype(np.uint8))
    return out


def strip(tiles, title):
    s = np.concatenate([x for t in tiles for x in (t, np.full((T, PAD, 3), 255, np.uint8))][:-1], 1)
    head = Image.new("RGB", (s.shape[1], LAB), (255, 255, 255))
    ImageDraw.Draw(head).text((2, 8), title, fill=(25, 35, 55))
    return np.concatenate([np.asarray(head), s], 0)


def orbit_tiles(obj, root, n=4):
    fs = sorted((root / obj / "samples-rgb").glob("*.png"))
    if not fs: return []
    idx = np.linspace(0, len(fs) - 1, n).round().astype(int)
    out = []
    for i in idx:
        im = Image.open(fs[i]).convert("RGB"); im.thumbnail((T, T), Image.LANCZOS)
        cv = Image.new("RGB", (T, T), (255, 255, 255)); cv.paste(im, ((T - im.width) // 2, (T - im.height) // 2))
        out.append(np.asarray(cv))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", required=True)
    ap.add_argument("--mesh_dir", type=Path, required=True)
    ap.add_argument("--orbit_root", type=Path, default=E / "outputs_full/demo/img2trajvid")
    args = ap.parse_args()
    for obj in args.objects:
        gt = args.mesh_dir / f"GT__{obj}__slam.glb"
        sl = args.mesh_dir / f"{obj}__slam.glb"
        vw = args.mesh_dir / f"{obj}__views.glb"
        if not (gt.exists() and sl.exists() and vw.exists()):
            print(f"!! {obj}: missing meshes"); continue
        sc = {}
        for c in ("slam", "views"):
            f = args.mesh_dir / f"{obj}__{c}.json"
            if f.exists(): sc[c] = json.load(open(f))["metrics"]
        rows = []
        ot = orbit_tiles(obj, args.orbit_root)
        if ot: rows.append(strip(ot, "GENERATED ORBIT  - what our pipeline produced and fed to ShapeR"))
        rows.append(strip(views_of(trimesh.load(gt, force="mesh"), (0.45, 0.65, 0.4)),
                          "REFERENCE MESH  - four sides, 90 deg apart"))
        t = f"BASELINE (capture views only)   CD {sc['slam']['chamfer']*100:.2f}  F1 {sc['slam']['f-score']:.3f}" if "slam" in sc else "BASELINE"
        rows.append(strip(views_of(trimesh.load(sl, force="mesh"), (0.80, 0.80, 0.84)), t))
        t = f"OURS (+ generated orbit views)   CD {sc['views']['chamfer']*100:.2f}  F1 {sc['views']['f-score']:.3f}" if "views" in sc else "OURS"
        rows.append(strip(views_of(trimesh.load(vw, force="mesh"), (0.93, 0.72, 0.25)), t))
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        sheet = np.concatenate([x for r in rows for x in (r, np.full((PAD * 2, w, 3), 255, np.uint8))][:-1], 0)
        Image.fromarray(sheet).save(E / f"figs/multi__{obj}.jpg", quality=90)
        print("wrote", E / f"figs/multi__{obj}.jpg", flush=True)
