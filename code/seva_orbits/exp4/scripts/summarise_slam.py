#!/usr/bin/env python3
"""SLAM-referenced summary on the corrected (full-SO(3)) fit; the old two-axis fit kept as old_slam_found.

    python scripts/summarise_slam.py -> outputs/summary_slam.json, figs/slam_found.png, figs/slam_dist.png, figs/cloud_dist.png

Numbers come from experiment4's realign_to_slam.py, which Sim(3)-fits each cloud to
the SLAM points themselves (no mesh in the loop) and reports both directions:
  slam_to_cloud  -- from each real SLAM point to the nearest cloud point (coverage)
  cloud_to_slam  -- from each cloud point to the nearest SLAM point (how much of the
                    cloud is near something real; the haze shows up here)
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
P = Path("/home/isha.tariq/viewcrafter_orbits"); E4 = Path("/home/isha.tariq/stable_virtual_camera/exp4")
R = {(r["object"], r["condition"]): r for r in json.loads((E4 / "outputs/final/realigned_final.json").read_text())}   # full-SO(3) fit
OLD = {(r["object"], r["condition"]): r for r in json.loads((P / "experiment4/outputs/realigned_to_slam.json").read_text())}
OBJ = ["ADT1292__chair", "BNB1045__sofa", "BNB1723__bed", "ADT1292__woodentable", "BNB2727__bench"]
SHORT = {o: o.split("__")[1] for o in OBJ}
CONDS = [("real", "VGGT on the real photos"), ("gen360", "ViewCrafter, 1 view"), ("v3_360", "ViewCrafter, 3 views (best)"),
         ("seva360", "VGGT, SEVA 25 frames"), ("seva360f", "VGGT, SEVA 80 frames"),
         ("seva360p", "Pi3, SEVA 25 frames"), ("seva360fp", "Pi3, SEVA 80 frames")]
COL = {"real": "#77746C", "gen360": "#B5641E", "v3_360": "#0E6FB5", "seva360": "#7FB37A", "seva360f": "#2E7A34", "seva360p": "#C07AA8", "seva360fp": "#7B2D6B"}
rows = []
for o in OBJ:
    for c, _ in CONDS:
        if (o, c) in R:
            r = R[(o, c)]
            rows.append({"object": o, "condition": c,
                         "slam_found": r["slam_to_vggt"]["within_2pct"], "slam_to_cloud_x100": r["slam_to_vggt"]["mean_x100"],
                         "cloud_near_slam": r["vggt_to_slam"]["within_2pct"], "cloud_to_slam_x100": r["vggt_to_slam"]["mean_x100"],
                         "fitted_scale": r["fitted_scale"], "up_z": r["up_z"],
                         "old_slam_found": OLD[(o, c)]["slam_to_vggt"]["within_2pct"] if (o, c) in OLD else None})
KEYS = ("slam_found", "slam_to_cloud_x100", "cloud_near_slam", "cloud_to_slam_x100")
means = {c: {k: float(np.mean([r[k] for r in rows if r["condition"] == c])) for k in KEYS} for c, _ in CONDS}
(E4 / "outputs/summary_slam.json").write_text(json.dumps({"rows": rows, "means": means, "conditions": dict(CONDS)}, indent=2))

def chart(key, ylabel, fname, higher_better=True):
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=170)
    x = np.arange(len(OBJ) + 1); w = 0.115
    for i, (c, lab) in enumerate(CONDS):
        vals = [next((r[key] for r in rows if r["object"] == o and r["condition"] == c), 0) for o in OBJ] + [means[c][key]]
        ax.bar(x + (i - 3) * w, vals, w, color=COL[c], label=lab)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[o] for o in OBJ] + ["mean"]); ax.set_ylabel(ylabel)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E3E7EC", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left" if higher_better else "upper right")
    fig.tight_layout(); fig.savefig(E4 / "figs" / fname); plt.close(fig)

chart("slam_found", "SLAM points reached (within 2% of size)", "slam_found.png")
chart("slam_to_cloud_x100", "SLAM point -> nearest cloud point (% of size)", "slam_dist.png", higher_better=False)
chart("cloud_near_slam", "cloud points within 2% of a SLAM point", "cloud_near_slam.png")
chart("cloud_to_slam_x100", "cloud point -> nearest SLAM point (% of size)", "cloud_dist.png", higher_better=False)
for c, lab in CONDS:
    m = means[c]; print(f"{lab:28s} found {m['slam_found']:.3f}  slam->cloud {m['slam_to_cloud_x100']:.2f}%  cloud near slam {m['cloud_near_slam']:.3f}  cloud->slam {m['cloud_to_slam_x100']:.2f}%")
