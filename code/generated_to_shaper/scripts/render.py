#!/usr/bin/env python3
"""Figures.  ~/.conda/envs/shaper/bin/python scripts/render.py   (EGL -> GPU node)

  figs/cloud__<obj>.png   SLAM points | Pi3 cloud over SLAM | KV cloud over SLAM  (top and side, model frame)
  figs/mesh__<obj>.png    GT | slam | pi3 | kv | best_n | union  -- ShapeR's mesh of repeat 1, same camera
  figs/scores.png         F1 and CD per object and condition
"""
import json, os, pickle, glob
from pathlib import Path
import numpy as np, trimesh
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import pyrender
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

E = Path("/home/isha.tariq/seva_to_shaper"); FIGS = E / "figs"; FIGS.mkdir(exist_ok=True)
SHAPER_DATA = Path("/home/isha.tariq/shapeR_testing/data")
CONDS = ["slam", "pi3", "kv", "pi3n", "kvn", "best_n", "union"]
rng = np.random.default_rng(0)


def scatter(ax, pts, c, s, **kw):
    ax.scatter(pts[:, 0], pts[:, 1], s=s, c=c, linewidths=0, **kw)


def cloud_fig(obj, slam, meshv, bounds):
    panels = [("SLAM points (ShapeR input)", None)]
    for rec in ("pi3", "kv"):
        f = E / f"outputs/aligned/{obj}__{rec}.npz"
        if f.exists():
            p = np.load(f)["points"]; p = p[rng.choice(len(p), min(len(p), 60000), replace=False)]
            panels.append((f"{rec} orbit cloud over SLAM", p))
    fig, axes = plt.subplots(2, len(panels), figsize=(4.2 * len(panels), 8))
    lim = 1.25 * bounds
    for j, (title, p) in enumerate(panels):
        for i, (a, b) in enumerate([(0, 1), (0, 2)]):
            ax = axes[i, j]; mv = meshv[rng.choice(len(meshv), 4000, replace=False)]
            scatter(ax, mv[:, [a, b]], "#d0d0d0", 2)
            if p is not None: scatter(ax, p[:, [a, b]], "#e07a1f", 1, alpha=0.5)
            scatter(ax, slam[:, [a, b]], "#1f4e9e", 6)
            ax.set_xlim(-lim[a], lim[a]); ax.set_ylim(-lim[b], lim[b]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if i == 0: ax.set_title(title, fontsize=10)
            ax.set_ylabel(["top view (x,y)", "side view (x,z)"][i] if j == 0 else "")
    fig.suptitle(f"{obj}: grey = GT mesh vertices, blue = SLAM points ({len(slam)}), orange = generated-orbit cloud", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / f"cloud__{obj}.png", dpi=110); plt.close(fig)


def render_mesh(mesh, K_pose, W=420, H=420, color=(0.93, 0.72, 0.25)):
    scene = pyrender.Scene(bg_color=[1, 1, 1, 1], ambient_light=[0.35] * 3)
    m = mesh.copy(); m.visual = trimesh.visual.ColorVisuals(m, face_colors=np.array([*(np.array(color) * 255), 255], np.uint8))
    scene.add(pyrender.Mesh.from_trimesh(m, smooth=False))
    cam = pyrender.PerspectiveCamera(yfov=np.radians(40)); scene.add(cam, pose=K_pose)
    scene.add(pyrender.DirectionalLight(intensity=4.0), pose=K_pose)
    r = pyrender.OffscreenRenderer(W, H)
    # Composite over white ourselves: the scene's bg_color does not reach the colour
    # buffer here, so a plain render comes back on black.
    rgba, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA); r.delete()
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


def look_at(eye, target, up=(0, 0, 1)):
    z = eye - target; z /= np.linalg.norm(z); x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    M = np.eye(4); M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = x, y, z, eye; return M


def mesh_fig(obj, gt, bounds, scores):
    diag = float(np.linalg.norm(bounds) * 2); eye = np.array([1.0, -1.2, 0.8]) / np.linalg.norm([1.0, -1.2, 0.8]) * diag * 1.6
    pose = look_at(eye, np.zeros(3))
    cols = [("GT", gt)]
    for cond in CONDS:
        f = E / f"outputs/shaper/rep1/per_object/{obj}__{cond}.glb"
        if f.exists():
            m = trimesh.load(f, force="mesh"); cols.append((cond, m))
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.6))
    for ax, (name, m) in zip(axes, cols):
        ax.imshow(render_mesh(m, pose, color=(0.45, 0.65, 0.4) if name == "GT" else (0.93, 0.72, 0.25))); ax.axis("off")
        t = name
        if name in scores: s = scores[name]; t += f"\nCD {s['chamfer_x100'][0]:.2f}  F1 {s['f1'][0]:.3f}"
        ax.set_title(t, fontsize=9)
    fig.suptitle(f"{obj}: ShapeR output per point-conditioning (repeat 1), same camera", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGS / f"mesh__{obj}.png", dpi=110); plt.close(fig)


def scores_fig(summary):
    objs = list(summary["per_object"]); conds = [c for c in CONDS if any(c in summary["per_object"][o]["conditions"] for o in objs)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2)); w = 0.8 / len(conds); x = np.arange(len(objs))
    colors = {"slam": "#1f4e9e", "pi3": "#e07a1f", "kv": "#c0392b", "pi3n": "#f0b070", "kvn": "#e08070", "best_n": "#8e44ad", "union": "#27ae60"}
    for ax, key, lab in [(axes[0], "f1", "F1 (higher is better)"), (axes[1], "chamfer_x100", "Chamfer x10^2 (lower is better)")]:
        for i, c in enumerate(conds):
            v = [summary["per_object"][o]["conditions"].get(c, {}).get(key, [np.nan, 0]) for o in objs]
            ax.bar(x + (i - len(conds) / 2 + 0.5) * w, [a for a, _ in v], w, yerr=[b for _, b in v], label=c, color=colors[c], capsize=2)
        ax.set_xticks(x); ax.set_xticklabels([o.replace("__", "\n") for o in objs], fontsize=8); ax.set_title(lab); ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8); fig.suptitle(f"ShapeR metrics by point conditioning, mean ± std over {summary['reps']} repeats")
    fig.tight_layout(); fig.savefig(FIGS / "scores.png", dpi=120); plt.close(fig)


summary = json.loads((E / "outputs/summary.json").read_text()) if (E / "outputs/summary.json").exists() else None
for obj in json.loads((E / "outputs/sets.json").read_text()):
    d = pickle.load(open(SHAPER_DATA / f"{obj}.pkl", "rb"))
    slam = d["points_model"].numpy(); bounds = d["bounds"].numpy()
    gt = trimesh.Trimesh(vertices=np.asarray(d["mesh_vertices"], np.float64), faces=np.asarray(d["mesh_faces"]), process=False)
    cloud_fig(obj, slam, np.asarray(gt.vertices), bounds)
    if summary and obj in summary["per_object"]:
        mesh_fig(obj, gt, bounds, summary["per_object"][obj]["conditions"])
    print("rendered", obj, flush=True)
if summary: scores_fig(summary)
