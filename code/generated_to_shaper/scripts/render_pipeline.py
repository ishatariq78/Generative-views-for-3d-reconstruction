#!/usr/bin/env python3
"""One strip per object showing what every stage of the pipeline outputs.

    ~/.conda/envs/shaper/bin/python scripts/render_pipeline.py --objects <obj> ...

Stages, left to right, each labelled with what produced it:
  1  raw capture      the Aria frames as they are, clutter and occluders included
  2  cleaned views    build_scenes.py: background removed, re-aimed, occluders repaired
  3  generated orbit  SEVA img2trajvid: 80 frames on cameras we chose
  4  ingested views   make_pkls_views.py: the orbit re-expressed as Aria cameras, as the
                      loader actually hands them to the model (object-point mask in red)
  5  meshes           ShapeR's output, baseline vs ours, next to the reference

figs/nodes__<obj>.jpg
"""
import argparse, io, json, os, pickle, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from PIL import Image, ImageDraw

E = Path("/home/isha.tariq/seva_to_shaper"); DATA = Path("/home/isha.tariq/shapeR_testing/data")
T, PAD, LAB = 210, 8, 30


def tile(img, size=T):
    im = Image.fromarray(np.asarray(img)).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def row(tiles, title, n=4):
    tiles = tiles[:n] + [np.full((T, T, 3), 255, np.uint8)] * max(0, n - len(tiles))
    strip = np.concatenate([x for t in tiles for x in (t, np.full((T, PAD, 3), 255, np.uint8))][:-1], 1)
    head = Image.new("RGB", (strip.shape[1], LAB), (255, 255, 255))
    ImageDraw.Draw(head).text((2, 8), title, fill=(25, 35, 55))
    return np.concatenate([np.asarray(head), strip], 0)


def raw_views(obj, n=4):
    d = pickle.load(open(DATA / f"{obj}.pkl", "rb"))
    key = "rgb_image_data" if "rgb_image_data" in d else "image_data"
    idx = np.linspace(0, len(d[key]) - 1, n).round().astype(int)
    return [tile(Image.open(io.BytesIO(d[key][i])).convert("RGB")) for i in idx]


def cleaned(obj, root, n=4):
    fs = sorted((root / obj / "images").glob("*.png"))
    idx = np.linspace(0, len(fs) - 1, n).round().astype(int) if fs else []
    return [tile(Image.open(fs[i]).convert("RGB")) for i in idx]


def orbit(obj, n=4, root=None):
    fs = sorted(((root or E / "outputs/demo/img2trajvid") / obj / "samples-rgb").glob("*.png"))
    idx = np.linspace(0, len(fs) - 1, n).round().astype(int) if fs else []
    return [tile(Image.open(fs[i]).convert("RGB")) for i in idx]


def ingested(obj, n=4):
    f = E / f"figs/ingest__{obj}__vonly.jpg"
    if not f.exists(): return []
    a = np.asarray(Image.open(f).convert("RGB")); k = a.shape[1] // 224
    idx = np.linspace(0, k - 1, min(n, k)).round().astype(int)
    return [tile(a[:, i * 224:(i + 1) * 224]) for i in idx]


def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def render(mesh, pose, color):
    sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35] * 3)
    m = mesh.copy(); m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(color) * 255), 255], np.uint8))
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.radians(40)); sc.add(cam, pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
    r = pyrender.OffscreenRenderer(T, T)
    rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def meshes(obj, seed, conds, mesh_dir=None):
    """Meshes must come from the configuration the deck reports. They previously came from
    the 16-view runs while the numbers came from 24, so the figure and the caption
    disagreed; mesh_dir now names the run whose numbers are being quoted."""
    out = []
    md = mesh_dir or (E / f"outputs/seeded/s{seed}/per_object")
    gt = md / f"GT__{obj}__slam.glb"
    if not gt.exists(): return out
    g = trimesh.load(gt, force="mesh")
    d = float(np.linalg.norm(g.bounds[1] - g.bounds[0])); c = g.bounds.mean(0)
    eye = np.array([1.0, -1.2, 0.75]); eye = eye / np.linalg.norm(eye) * d * 1.7 + c
    pose = look_at(eye, c)
    out.append(render(g, pose, (0.45, 0.65, 0.4)))
    for cond in conds:
        p = md / f"{obj}__{cond}.glb"
        if p.exists(): out.append(render(trimesh.load(p, force="mesh"), pose, (0.93, 0.72, 0.25)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", nargs="*", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", nargs="*", default=["slam", "views"])
    ap.add_argument("--scene_root", type=Path, default=E / "inputs")
    ap.add_argument("--orbit_root", type=Path, default=E / "outputs/demo/img2trajvid")
    ap.add_argument("--mesh_dir", type=Path, default=None,
                    help="directory of the run whose numbers the deck quotes")
    args = ap.parse_args()
    for obj in args.objects:
        rows = [
            row(raw_views(obj), "1  RAW CAPTURE  - the Aria frames as they are (clutter, occluders, one side)"),
            row(cleaned(obj, args.scene_root), "2  CLEANED VIEWS  - build_scenes.py: background removed, re-aimed, occluders repaired"),
            row(orbit(obj, root=args.orbit_root), "3  GENERATED ORBIT  - SEVA img2trajvid: 80 frames on cameras we specify"),
            row(ingested(obj), "4  INGESTED VIEWS  - re-expressed as Aria cameras; what ShapeR actually reads (red = point mask)"),
            row(meshes(obj, args.seed, args.conds, args.mesh_dir), "5  SHAPER OUTPUT  - reference (green), then " + ", ".join(args.conds)),
        ]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        sheet = np.concatenate([x for r in rows for x in (r, np.full((PAD * 2, w, 3), 255, np.uint8))][:-1], 0)
        Image.fromarray(sheet).save(E / f"figs/nodes__{obj}.jpg", quality=90)
        print("wrote", E / f"figs/nodes__{obj}.jpg", flush=True)
