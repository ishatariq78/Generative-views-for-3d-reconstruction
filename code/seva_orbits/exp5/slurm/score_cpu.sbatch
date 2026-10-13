#!/bin/bash
#SBATCH --job-name=exp5-score
#SBATCH --partition=cscc-cpu-p
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/stable_virtual_camera/exp5/logs/%x-%j.out
# [STEP 2] collect (clouds + pose scores, kvtracker env) then fit clouds to the SLAM points (shaper env).
set -eo pipefail
export TMPDIR=/tmp; export XDG_RUNTIME_DIR=/tmp/xdg-$(id -u); mkdir -p "$XDG_RUNTIME_DIR"
cd /home/isha.tariq/stable_virtual_camera/exp5
echo "### collect"; /home/isha.tariq/.conda/envs/kvtracker/bin/python scripts/collect.py
echo; echo "### SLAM points"; /home/isha.tariq/.conda/envs/shaper/bin/python scripts/score_slam.py
