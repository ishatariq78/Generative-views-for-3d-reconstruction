#!/bin/bash
#SBATCH --job-name=s2s-ptmesh
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
# [STEP 28] Export the POINTS-conditioned meshes (pi3) at the same 16-view budget the
# points-vs-views comparison was run at, so that contrast can be shown and not just stated.
set -uo pipefail
E=/home/isha.tariq/seva_to_shaper
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
cd /home/isha.tariq/shapeR_testing/exp_object_aware/scripts
PKLS=""
for O in BNB1045__barstool HWD0045__woodentable BNB1182__stool; do PKLS="$PKLS $E/outputs/pkls/${O}__pi3.pkl"; done
python run_shaper_seeded.py --input_pkls $PKLS --out_dir "$E/outputs/seeded/s0" --seed 0 --save_meshes
echo "### done"
