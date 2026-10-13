#!/bin/bash
#SBATCH --job-name=s2s-fig2
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 12] The comparison sheets, now that the meshes exist: views ingested (real vs
# generated) on top, predicted meshes per condition underneath.
#   sbatch slurm/12_figs2.sbatch
set -eo pipefail
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/seva_to_shaper
python scripts/render_compare.py --seed 0 --conds slam gap1 views vgap
echo "### done"
