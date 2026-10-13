#!/usr/bin/env python3
"""One scatter: does the metric experiment 5 used predict the outcome?

x is worst observed coverage (experiment 5's judge), y is SLAM points reproduced
(the pipeline's actual purpose), one dot per run. Colour is the route; marker
shape is the object. If the routes experiment 5 called equivalent are equivalent
downstream, the dots should line up; if coverage is a poor proxy, they won't.

    python scripts/render_coverage_vs_found.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
import numpy as np                                    # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits")
sys.path.insert(0, str(PROJ / "shared" / "scripts"))
import objects as C                                   # noqa: E402

OBJ = C.names()
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]
# Categorical slots 1-6 of the validated reference palette, in fixed order.
COL = dict(zip(PATHS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MARK = dict(zip(OBJ, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e6e3"


def main():
    e9 = {(r["object"], r["path"]): r
          for r in json.loads((PROJ / "experiment9/outputs/measurements.json").read_text())}
    e4 = {(r["object"], r["condition"]): r
          for r in json.loads((PROJ / "experiment4/outputs/realigned_to_slam.json").read_text())}
    xs, ys = [], []
    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=200)
    for o in OBJ:
        for p in PATHS:
            x = e9[(o, p)]["coverage_min"]
            y = e4[(o, f"p3_{p}")]["slam_to_vggt"]["within_2pct"]
            xs.append(x); ys.append(y)
            ax.scatter(x, y, s=64, c=COL[p], marker=MARK[o], edgecolors="white",
                       linewidths=1.2, zorder=3)
    r = np.corrcoef(xs, ys)[0, 1]
    ax.set_xlabel("worst observed coverage  (experiment 5's measure)", fontsize=10.5, color=INK2)
    ax.set_ylabel("SLAM points reproduced  (the outcome)", fontsize=10.5, color=INK2)
    ax.set_title(f"30 runs, 5 objects × 6 routes   ·   r = {r:+.2f}", fontsize=11, color=INK2, pad=10)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.tick_params(labelsize=9.5, colors=INK2)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.set_xlim(0.2, 0.85); ax.set_ylim(0.2, 0.85)
    h1 = [Line2D([], [], marker="o", ls="", color=COL[p], markersize=8, label=p) for p in PATHS]
    h2 = [Line2D([], [], marker=MARK[o], ls="", color="#9a9994", markersize=8,
                 label=o.split("__")[1]) for o in OBJ]
    l1 = ax.legend(handles=h1, title="route", fontsize=9, title_fontsize=9, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0))
    ax.add_artist(l1)
    ax.legend(handles=h2, title="object", fontsize=9, title_fontsize=9, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 0.45))
    fig.tight_layout()
    out = PROJ / "experiment9/outputs/coverage_vs_found.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out, f"r={r:+.3f}")


if __name__ == "__main__":
    main()
