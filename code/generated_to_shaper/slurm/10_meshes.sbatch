#!/bin/bash
#SBATCH --job-name=s2s-mesh
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 10] Re-run seed 0 only, with --save_meshes, for the conditions whose seeded runs
# were scored but not exported. Same seed => the same mesh that produced the seed-0 number,
# so the figures and the table agree.
#   sbatch slurm/10_meshes.sbatch
set -eo pipefail
E=/home/isha.tariq/seva_to_shaper
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
mapfile -t PKLS < <(find "$E/outputs/pkls" -maxdepth 1 \
  \( -name "*__slam.pkl" -o -name "*__gap1.pkl" -o -name "*__views.pkl" -o -name "*__vonly.pkl" \) | sort)
echo "${#PKLS[@]} pkls"
cd /home/isha.tariq/shapeR_testing/exp_object_aware/scripts
python run_shaper_seeded.py --input_pkls "${PKLS[@]}" \
    --out_dir "$E/outputs/seeded/s0" --seed 0 --save_meshes
echo "### done"
