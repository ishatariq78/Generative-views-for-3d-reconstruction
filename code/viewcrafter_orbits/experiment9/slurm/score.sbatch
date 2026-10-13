#!/bin/bash
#SBATCH --job-name=exp9-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/isha.tariq/viewcrafter_orbits/experiment9/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/viewcrafter_orbits/experiment9/logs/%x-%j.out
set -euo pipefail
P=/home/isha.tariq/viewcrafter_orbits
export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); export TMPDIR=/tmp; mkdir -p "$XDG_RUNTIME_DIR"
PY=/home/isha.tariq/.conda/envs/shaper/bin/python
$PY "$P/experiment3/scripts/align_and_score.py"
$PY "$P/experiment4/scripts/compare_slam_points.py"
$PY "$P/experiment4/scripts/realign_to_slam.py"
