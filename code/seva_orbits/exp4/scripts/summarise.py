#!/usr/bin/env python3
"""exp4: VGGT and Pi3 clouds from the SEVA orbits, beside every ViewCrafter condition.

    python scripts/summarise.py   -> outputs/summary.json, figs/slam_found.png, figs/distances.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

P = Path("/home/isha.tariq/viewcrafter_orbits"); E3 = Path("/home/isha.tariq/stable_virtual_camera/exp4")
R = {(r["object"], r["condition"]): r for r in json.loads((P / "experiment4/outputs/realigned_to_slam.json").read_text())}
S = {(r["object"], r["condition"]): r for r in json.loads((P / "experiment3/outputs/scores.json").read_text())}
B = {r["object"]: r for r in json.loads((P / "experiment4/outputs/baselines.json").read_text())}
OBJ = ["ADT1292__chair", "BNB1045__sofa", "BNB1723__bed", "ADT1292__woodentable", "BNB2727__bench"]
SHORT = {o: o.split("__")[1] for o in OBJ}
CONDS = [("real", "VGGT on the real photos"), ("gen360", "ViewCrafter, 1 view"), ("mv360", "ViewCrafter, 8 views"),
         ("v3_360", "ViewCrafter, 3 views (best)"), ("seva360", "VGGT, SEVA 25 frames"), ("seva360f", "VGGT, SEVA 80 frames"),
         ("seva360p", "Pi3, SEVA 25 frames"), ("seva360fp", "Pi3, SEVA 80 frames")]
COL = {"real": "#77746C", "gen360": "#B5641E", "mv360": "#D9A066", "v3_360": "#0E6FB5", "seva360": "#7FB37A", "seva360f": "#2E7A34", "seva360p": "#C07AA8", "seva360fp": "#7B2D6B"}

rows = []
for o in OBJ:
    for c, _ in CONDS:
        if (o, c) in R:
            rows.append({"object": o, "condition": c,
                         "slam_found": R[(o, c)]["slam_to_vggt"]["within_2pct"],
                         "slam_to_vggt_x100": R[(o, c)]["slam_to_vggt"]["mean_x100"],
                         "vggt_to_mesh_x100": R[(o, c)]["vggt_to_mesh"]["mean_x100"],
                         "recall_2pct": S[(o, c)]["recall_at_2pct"], "precision_2pct": S[(o, c)]["precision_at_2pct"],
                         "fitted_scale": R[(o, c)]["fitted_scale"]})
means = {c: {k: float(np.mean([r[k] for r in rows if r["condition"] == c])) for k in
             ("slam_found", "slam_to_vggt_x100", "vggt_to_mesh_x100", "recall_2pct", "precision_2pct")} for c, _ in CONDS}
slam_acc = float(np.mean([B[o]["slam_to_mesh"]["mean_x100"] for o in OBJ]))
(E3 / "outputs/summary.json").write_text(json.dumps({"rows": rows, "means": means, "slam_to_mesh_x100": slam_acc,
                                                     "conditions": dict(CONDS)}, indent=2))

def chart(key, ylabel, fname, fmt="{:.2f}", higher_better=True):
    fig, ax = plt.subplots(figsize=(11, 3.8), dpi=170)
    x = np.arange(len(OBJ) + 1); w = 0.10
    for i, (c, lab) in enumerate(CONDS):
        vals = [next((r[key] for r in rows if r["object"] == o and r["condition"] == c), 0) for o in OBJ] + [means[c][key]]
        ax.bar(x + (i - 3.5) * w, vals, w, color=COL[c], label=lab)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[o] for o in OBJ] + ["mean"])
    ax.set_ylabel(ylabel)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#E3E7EC", lw=0.8); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, fontsize=8.5, loc="upper left" if higher_better else "upper right")
    fig.tight_layout(); fig.savefig(E3 / "figs" / fname); plt.close(fig)

chart("slam_found", "SLAM points reached (within 2% of size)", "slam_found.png")
chart("slam_to_vggt_x100", "SLAM -> nearest VGGT point (% of size)", "distances.png", higher_better=False)
print(json.dumps(means, indent=1)); print("SLAM points themselves sit", round(slam_acc, 2), "% from the true surface")
