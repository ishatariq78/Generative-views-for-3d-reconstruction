"""A fuller rotation search for experiment3's Sim(3) fit.

`align_and_score.fit` starts from Rz(az)·Rx(tilt), tilt in {0,60,120}: two of the three
rotational degrees of freedom, and nothing within 60° of the -z pole. On the SEVA clouds
ICP then settles upside down on every object (exp5/scripts/check_upright.py). This module
swaps in a grid over all of SO(3) -- az x tilt x roll at 30°, tilt to 180° -- and gives
more basins the short ICP. Same scoring, same ICP, same scale grid; only the starts change.

    import fit_so3; M, cost = fit_so3.fit(source, target)
"""
import sys
import numpy as np, trimesh
sys.path.insert(0, "/home/isha.tariq/viewcrafter_orbits/experiment3/scripts")
import align_and_score as A

def initial_rotations_so3(step_deg=30):
    out = []; s = np.radians(step_deg)
    for tilt in np.arange(0, np.pi + 1e-9, s):
        n_az = 1 if tilt < 1e-9 or abs(tilt - np.pi) < 1e-9 else int(round(2 * np.pi / s))
        for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
            for roll in np.arange(0, 2 * np.pi - 1e-9, s):
                out.append(trimesh.transformations.rotation_matrix(az, [0, 0, 1]) @ trimesh.transformations.rotation_matrix(tilt, [1, 0, 0])
                           @ trimesh.transformations.rotation_matrix(roll, [0, 0, 1]))
    return out

def fit(source, target, n_mid=48):
    old_rot, old_mid = A.initial_rotations, A.N_MID
    A.initial_rotations, A.N_MID = initial_rotations_so3, n_mid
    try:
        return A.fit(source, target)
    finally:
        A.initial_rotations, A.N_MID = old_rot, old_mid
