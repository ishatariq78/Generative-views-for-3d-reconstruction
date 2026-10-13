#!/usr/bin/env python3
"""Measure each camera path with experiment 2's two metrics, unchanged.

Same definitions as experiment 5 used on the one-view runs, so the six paths here
are directly comparable to the six there. Nothing about the metric changed; only
the cloud underneath the orbit did.

Reusing the definitions rather than inventing new ones is the point: experiment 2
established what a good full-turn run looks like on these objects, and the only
way to say whether a different camera path is better or worse is to score it the
same way.

  observed coverage   the lit fraction of render0.mp4 -- the share of each frame
                      resting on geometry DUSt3R actually recovered, before the
                      diffusion model fills anything in.
  subject area        the share of each finished frame that is not backdrop,
                      normalised by frame 0 (the reference image), which fixes
                      1.0 at the object's true silhouette area. Above 1 means the
                      model is painting the empty backdrop in.

    python scripts/measure_paths.py
"""
import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np

PROJ = Path("/home/isha.tariq/viewcrafter_orbits/experiment9")


def subject_fraction(frame):
    f = frame.astype(np.float32) / 255.0
    mx, mn = f.max(2), f.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    return float(((sat > 0.15) | (mx < 0.70)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, default=PROJ / "outputs")
    args = ap.parse_args()
    runs = json.loads((args.out_dir / "runs.json").read_text())

    cov, subj, rows = {}, {}, []
    for r in runs:
        d = Path(r["output_dir"])
        ren = iio.imread(d / "render0.mp4", plugin="pyav")
        dif = iio.imread(d / "diffusion0.mp4", plugin="pyav")
        c = [round(float((fr.max(axis=2) > 12).mean()), 4) for fr in ren]
        s = [round(subject_fraction(fr), 4) for fr in dif]
        cov[r["experiment"]], subj[r["experiment"]] = c, s
        norm = [v / s[0] for v in s]
        rows.append({
            "experiment": r["experiment"], "object": r["object"], "path": r["path"],
            "coverage_min": round(min(c), 4), "coverage_mean": round(float(np.mean(c)), 4),
            "subject_ref": s[0], "subject_peak_ratio": round(max(norm), 4),
            "subject_peak_frame": int(np.argmax(norm)),
            "frames_over_1_5x": int(sum(v > 1.5 for v in norm)),
        })

    json.dump(cov, open(args.out_dir / "render_coverage.json", "w"), indent=1)
    json.dump(subj, open(args.out_dir / "subject_fraction.json", "w"), indent=1)
    json.dump(rows, open(args.out_dir / "measurements.json", "w"), indent=2)

    paths = []
    for r in rows:
        if r["path"] not in paths:
            paths.append(r["path"])
    print(f"\n{'path':9s} {'cov min':>8} {'cov mean':>9} {'peak subj':>10} {'frames>1.5x':>12}")
    print("-" * 52)
    for p in paths:
        sel = [r for r in rows if r["path"] == p]
        print(f"{p:9s} {np.mean([r['coverage_min'] for r in sel]):8.3f} "
              f"{np.mean([r['coverage_mean'] for r in sel]):9.3f} "
              f"{np.mean([r['subject_peak_ratio'] for r in sel]):10.2f}x "
              f"{np.mean([r['frames_over_1_5x'] for r in sel]):9.1f}/25")
    print(f"\nwrote measurements for {len(rows)} runs to {args.out_dir}")


if __name__ == "__main__":
    main()
