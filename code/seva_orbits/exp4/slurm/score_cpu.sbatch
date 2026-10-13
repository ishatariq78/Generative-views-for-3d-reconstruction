#!/bin/bash
#SBATCH --job-name=exp4-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp4/logs/%x-%j.out
# [STEP 2] Align every cloud (VGGT and Pi3 alike) to the mesh, then compare against
#          ShapeR's SLAM points. One pass over everything: the scorer is unseeded.
set -eo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "### experiment 3 -- align to the reference mesh and score"; $PY "$P/experiment3/scripts/align_and_score.py"
echo; echo "### experiment 4 -- against the original SLAM points"
$PY "$P/experiment4/scripts/compare_slam_points.py"; $PY "$P/experiment4/scripts/realign_to_slam.py"
