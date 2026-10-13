#!/usr/bin/env python3
"""Score a KV-Tracker trajectory on a TUM RGB-D sequence against ground truth.

    python scripts/score_tum.py <results_dir> <scene_dir>

Uses the repo's own pieces so the number means what the paper's number means:
`TUMLoader.load_gt` for the timestamp association (nearest ground-truth pose to
each RGB frame) and `evo_utils.align_pair` for the Sim(3) alignment and ATE/RPE.
Run from the repo root, under the kvtracker env.

traj.npy holds one 4x4 per processed frame in frame order. The variable is named
T_w2c upstream but is built from pred_T_wc (camera-to-world); both readings are
scored and reported, and the one that aligns better is the convention in use.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/isha.tariq/kv_tracker_testing/third_party/kv_tracker")
from kv_tracker.dataloaders.tum import TUMLoader          # noqa: E402
from kv_tracker.eval_tools.evo_utils import align_pair    # noqa: E402

res, scene = Path(sys.argv[1]), Path(sys.argv[2])
est = np.load(res / "traj.npy")
gt = TUMLoader.load_gt(str(scene))
print(f"estimated {est.shape[0]} poses, ground truth {gt.shape[0]} frames")
n = min(len(est), len(gt))
est, gt = est[:n], gt[:n]

out = {}
for name, e in [("as_c2w", est), ("as_w2c_inverted", np.linalg.inv(est))]:
    ate, rpe_t, rpe_r = align_pair({"traj_gt": gt, "traj_est": e}, ret_np=False)
    out[name] = {"ate_rmse_m": float(ate), "rpe_trans_rmse_m": float(rpe_t),
                 "rpe_rot_rmse_deg": float(np.degrees(rpe_r))}
    print(f"{name:18s} ATE {ate*100:6.2f} cm   RPE trans {rpe_t*100:5.2f} cm   "
          f"RPE rot {np.degrees(rpe_r):5.2f} deg")
best = min(out, key=lambda k: out[k]["ate_rmse_m"])
out["convention_in_use"] = best
out["frames"] = int(n)
(res / "scores.json").write_text(json.dumps(out, indent=2))
print(f"\nconvention in use: {best}  ->  {res / 'scores.json'}")
