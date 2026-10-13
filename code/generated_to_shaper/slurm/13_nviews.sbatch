#!/bin/bash
#SBATCH --job-name=s2s-nv
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 13] Raise the view budget so the generated views ADD to the capture's own instead of
# displacing them. `slam` is run at the same budget as the control, so any gain is the extra
# views and not the budget. N=24 first (a quick 2-object probe decides whether 32 is worth it).
#   sbatch slurm/13_nviews.sbatch
set -eo pipefail
E=/home/isha.tariq/seva_to_shaper; N=${NVIEWS:-24}
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname)  view budget $N ==="
mapfile -t PKLS < <(find "$E/outputs/pkls" -maxdepth 1 \
  \( -name "*__slam.pkl" -o -name "*__views.pkl" -o -name "*__vgap.pkl" \) | sort)
echo "${#PKLS[@]} pkls"
cd "$E"
for SEED in 0 1 2; do
  echo; echo "### seed $SEED"
  python scripts/run_shaper_nviews.py --input_pkls "${PKLS[@]}" \
      --out_dir "$E/outputs/nviews$N/s$SEED" --seed "$SEED" --num_images "$N" --skip_done
done
echo "### done"
