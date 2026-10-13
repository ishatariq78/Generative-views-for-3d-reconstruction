#!/bin/bash
#SBATCH --job-name=exp6-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment6/logs/%x-%j.out

# [STEP 4] Align every cloud to the true mesh, then compare against the SLAM points.
#   sbatch slurm/downstream_cpu.sbatch     (after downstream_gpu finishes)
set -euo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "### experiment 3 — align to the reference mesh and score"
$PY "$P/experiment3/scripts/align_and_score.py"
echo; echo "### experiment 4 — against the original SLAM points"
$PY "$P/experiment4/scripts/compare_slam_points.py"
$PY "$P/experiment4/scripts/realign_to_slam.py"
