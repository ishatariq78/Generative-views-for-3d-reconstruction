#!/usr/bin/env python3
"""The 'input arc' column of exp1's table, computed as the true angular extent.

build_scenes.py sorts the 8 input views by azimuth and reports the arc as the sum
of circular steps between consecutive sorted views. For an arc that crosses 0 deg
(chair, bench) the sort puts the wrap in the middle of the list, and the step
across it -- the short way round, back over ground the arc already covers -- is
added to the total. So the chair's 133 deg arc was recorded as 216 deg and the
bench's 102 deg as 200 deg. Extent = 360 minus the largest circular gap between
the views. Nothing about the orbits or the scores changes; this only corrects the
reported arc. Writes outputs/input_arc_corrected.json; scores.json is untouched.
"""
import json
from pathlib import Path
import numpy as np

E1 = Path("/home/isha.tariq/stable_virtual_camera/exp1")
scores = {r["object"]: r for r in json.loads((E1 / "outputs/scores.json").read_text())}
rows = []
for d in sorted((E1 / "inputs").iterdir()):
    if not (d / "meta.json").exists():
        continue
    m = json.loads((d / "meta.json").read_text())
    az = np.sort(np.array([f["azimuth_deg"] for f in m["frames"]]) % 360)
    gaps = np.diff(np.r_[az, az[0] + 360])
    extent = float(360 - gaps.max()); start = float(az[(np.argmax(gaps) + 1) % len(az)])
    s = scores[d.name]
    rows.append({"object": d.name, "input_arc_recorded_deg": s["input_arc_deg"],
                 "input_arc_true_deg": round(extent, 1), "arc_runs_deg": [round(start, 1), round((start + extent) % 360, 1)],
                 "orbit_start_deg": round(float(m["frames"][0]["azimuth_deg"]), 1),
                 "orbit_iou_mean": s["orbit_iou_mean"], "share_of_ceiling": round(s["orbit_iou_mean"] / s["masker_iou_on_inputs"], 2)})
    print(f"{d.name:22s} recorded {s['input_arc_deg']:6.1f}°  true {extent:6.1f}°  (runs {start:.1f}°→{(start+extent)%360:.1f}°)  IoU {s['orbit_iou_mean']:.3f}")
(E1 / "outputs/input_arc_corrected.json").write_text(json.dumps(rows, indent=2))
a = np.array([r["input_arc_true_deg"] for r in rows]); b = np.array([r["orbit_iou_mean"] for r in rows])
print(f"corr(true arc, IoU) = {np.corrcoef(a, b)[0,1]:+.2f}; recorded-arc corr = {np.corrcoef([r['input_arc_recorded_deg'] for r in rows], b)[0,1]:+.2f}")
