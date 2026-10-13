#!/bin/bash
#SBATCH --job-name=exp4-compare
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment4/logs/%x-%j.out

# =============================================================================
# [STEP 1] Compare VGGT's points with ShapeR's SLAM points.
#   sbatch slurm/compare.sbatch
# CPU only -- KD-tree work. ~3 min. Needs experiment 3's points_aligned.npy.
# =============================================================================
set -euo pipefail
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment4/scripts/compare_slam_points.py

# [STEP 2] the control: re-align to the SLAM points instead of the mesh
/home/isha.tariq/.conda/envs/shaper/bin/python \
    /home/isha.tariq/viewcrafter_orbits/experiment4/scripts/realign_to_slam.py
