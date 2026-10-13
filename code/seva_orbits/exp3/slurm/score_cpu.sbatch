#!/bin/bash
#SBATCH --job-name=exp3-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp3/logs/%x-%j.out
# [STEP 2] Align every cloud to the mesh, then compare against ShapeR's SLAM points.
#   sbatch exp3/slurm/score_cpu.sbatch      (after vggt_gpu finishes)
# Rescoring every condition in one pass, because the scorer samples the mesh
# unseeded and numbers move in the third decimal between passes.
set -eo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp
mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
echo "### experiment 3 -- align to the reference mesh and score"
$PY "$P/experiment3/scripts/align_and_score.py"
echo; echo "### experiment 4 -- against the original SLAM points"
$PY "$P/experiment4/scripts/compare_slam_points.py"
$PY "$P/experiment4/scripts/realign_to_slam.py"
echo; echo "### comparison renders"
$PY "$P/experiment4/scripts/render_comparison.py"
