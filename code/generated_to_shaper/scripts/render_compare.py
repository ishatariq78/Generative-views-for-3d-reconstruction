#!/usr/bin/env python3
"""The visual comparison: what ShapeR was shown, and what it produced, per condition.

    ~/.conda/envs/shaper/bin/python scripts/render_compare.py [--seed 0] [--conds ...]

Row 1  the views ShapeR ingests in the baseline (real capture: clutter, occluders, partial arc)
Row 2  the views it ingests from the generated orbit (object alone, cleaned, all round)
Row 3  the predicted mesh per condition, next to the reference, from one camera

figs/compare__<obj>.jpg
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import trimesh, pyrender
from PIL import Image, ImageDraw

E = Path("/home/isha.tariq/seva_to_shaper")
TILE, PAD = 224, 6


def look_at(eye, target, up=(0, 0, 1)):
    z = np.asarray(eye, float) - target; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye
    return M


def render(mesh, pose, size=TILE, color=(0.93, 0.72, 0.25)):
    sc = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35] * 3)
    m = mesh.copy(); m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(color) * 255), 255], np.uint8))
    sc.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.radians(40)); sc.add(cam, pose=pose)
    sc.add(pyrender.DirectionalLight(intensity=4.0), pose=pose)
    r = pyrender.OffscreenRenderer(size, size)
    rgba, _ = r.render(sc, flags=pyrender.RenderFlags.RGBA); r.delete()
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def label(img, text, colour=(20, 20, 20)):
    im = Image.fromarray(img); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(255, 255, 255))
    d.text((4, 3), text, fill=colour)
    return np.asarray(im)


def strip(tiles, texts):
    out = []
    for t, x in zip(tiles, texts):
        out.append(label(np.asarray(Image.fromarray(t).resize((TILE, TILE))), x))
        out.append(np.full((TILE, PAD, 3), 255, np.uint8))
    return np.concatenate(out[:-1], 1) if out else None


def ingest_tiles(name, n=5):
    """What the loader hands the model, straight from the saved check sheet."""
    f = E / f"figs/ingest__{name}.jpg"
    if not f.exists(): return None
    a = np.asarray(Image.open(f).convert("RGB"))
    k = a.shape[1] // 224
    return [a[:, i * 224:(i + 1) * 224] for i in range(min(n, k))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conds", nargs="*", default=["slam", "gap1", "views", "vonly"])
    args = ap.parse_args()
    scores = {}
    for c in args.conds + ["slam"]:
        for o in json.loads((E / "outputs/sets.json").read_text()):
            f = E / f"outputs/seeded/s{args.seed}/per_object/{o}__{c}.json"
            if f.exists(): scores[(o, c)] = json.load(open(f))["metrics"]
    for obj in json.loads((E / "outputs/sets.json").read_text()):
        rows = []
        for cond, cap in (("slam", "what ShapeR sees now: the real capture"),
                          ("vonly", "what it sees from the generated orbit")):
            t = ingest_tiles(f"{obj}__{cond}")
            if t: rows.append(strip(t, [cap] + [""] * (len(t) - 1)))
        gt_p = E / f"outputs/seeded/s{args.seed}/per_object/GT__{obj}__slam.glb"
        tiles, texts = [], []
        if gt_p.exists():
            gt = trimesh.load(gt_p, force="mesh")
            d = float(np.linalg.norm(gt.bounds[1] - gt.bounds[0]))
            eye = np.array([1.0, -1.2, 0.75]); eye = eye / np.linalg.norm(eye) * d * 1.7
            pose = look_at(eye + gt.bounds.mean(0), gt.bounds.mean(0))
            tiles.append(render(gt, pose, color=(0.45, 0.65, 0.4))); texts.append("reference")
            for c in args.conds:
                p = E / f"outputs/seeded/s{args.seed}/per_object/{obj}__{c}.glb"
                if not p.exists(): continue
                m = trimesh.load(p, force="mesh")
                s = scores.get((obj, c), {})
                tiles.append(render(m, pose))
                texts.append(f"{c}  CD {s.get('chamfer',0)*100:.2f}  F1 {s.get('f-score',0):.3f}" if s else c)
            rows.append(strip(tiles, texts))
        rows = [r for r in rows if r is not None]
        if not rows: print(f"!! {obj}: nothing to draw"); continue
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
        sheet = np.concatenate([x for r in rows for x in (r, np.full((PAD, w, 3), 255, np.uint8))][:-1], 0)
        Image.fromarray(sheet).save(E / f"figs/compare__{obj}.jpg", quality=90)
        print("wrote", E / f"figs/compare__{obj}.jpg", flush=True)


if __name__ == "__main__":
    main()
