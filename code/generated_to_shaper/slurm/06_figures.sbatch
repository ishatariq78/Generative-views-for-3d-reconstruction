#!/bin/bash
#SBATCH --job-name=s2s-figs
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 6] Re-render every figure (cloud sheets, mesh comparisons, score chart).
#   sbatch slurm/06_figures.sbatch
# Needs EGL, so it runs on a GPU node rather than the login node -- a login-node
# render dies with the shell that started it.
set -eo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/seva_to_shaper
python scripts/render.py
echo "### done"
