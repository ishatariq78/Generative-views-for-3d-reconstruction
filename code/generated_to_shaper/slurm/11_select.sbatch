#!/bin/bash
#SBATCH --job-name=s2s-sel
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 11] Choose each object's output by agreement with its own SLAM points (GT-free),
# and report it against the baseline and against the oracle pick.
#   sbatch slurm/11_select.sbatch
set -eo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/seva_to_shaper
python scripts/select_by_slam.py
echo "### done"
