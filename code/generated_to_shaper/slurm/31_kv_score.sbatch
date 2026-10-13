#!/bin/bash
#SBATCH --job-name=s2s-kvscore
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 31] Place each KV-Tracker cloud in the model frame, swap it in for the SLAM points,
# and score against the same baseline at the 24-view budget. Chained on the tracking job.
set -uo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/seva_to_shaper
python scripts/kv_subset_score.py
echo "### done"
