#!/usr/bin/env python3
"""One figure showing what the six routes actually are.

Each path is a trajectory file of three rows — azimuth, elevation offset, radius
offset — interpolated across 25 frames. Drawing them makes the difference legible
without watching six videos: `high` and `low` are flat offsets that never come
back, `helix` is one slow rise and fall, `wobble` is two fast ones, `dolly` leaves
elevation alone and moves the camera in.

Elevation is drawn as an offset from the object's measured capture plane, which is
what the trajectory files encode.

    python scripts/render_path_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment9")
PATHS = ["level", "high", "low", "helix", "wobble", "dolly"]
COL = {"level": "#14181E", "high": "#B5641E", "low": "#0E6FB5",
       "helix": "#0F7A75", "wobble": "#7C4DA8", "dolly": "#A33A2A"}
# Several of these sit flat on zero in one panel or the other -- level in both,
# every path but dolly in the radius panel. Without distinct dashes the last one
# drawn simply hides the rest, which would read as though they were not there.
DASH = {"level": (0, (1, 1.6)), "high": (0, ()), "low": (0, ()),
        "helix": (0, ()), "wobble": (0, (5, 1.6)), "dolly": (0, ())}
N = 25


def read(p):
    rows = Path(p).read_text().splitlines()
    return [np.array([float(v) for v in r.split()]) for r in rows[:3]]


def resample(a, n=N):
    """ViewCrafter spreads each row evenly over the frames and interpolates."""
    if len(a) == 1:
        return np.full(n, a[0])
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(a)), a)


def main():
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.5), dpi=200)
    for p in PATHS:
        phi, theta, rr = (resample(a) for a in read(PROJ / "trajs" / f"{p}.txt"))
        ax[0].plot(phi, theta, color=COL[p], lw=2.0, ls=DASH[p], label=p)
        ax[1].plot(phi, rr, color=COL[p], lw=2.0, ls=DASH[p], label=p)
    for a, ttl, yl in ((ax[0], "elevation, as an offset from the capture plane", "degrees"),
                       (ax[1], "radius, as a fraction of the starting distance", "offset")):
        a.set_title(ttl, fontsize=10.5, color="#4A525E", pad=8)
        a.set_xlabel("azimuth around the object (degrees)", fontsize=9.5, color="#4A525E")
        a.set_ylabel(yl, fontsize=9.5, color="#4A525E")
        a.axhline(0, color="#DDE2E8", lw=1.0, zorder=0)
        a.set_xlim(0, 360); a.set_xticks([0, 90, 180, 270, 360])
        a.grid(alpha=0.25, lw=0.6)
        a.tick_params(labelsize=9, colors="#4A525E")
        for sp in a.spines.values():
            sp.set_color("#DDE2E8")
    ax[0].legend(fontsize=9, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout()
    out = PROJ / "outputs" / "path_diagram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
