#!/bin/bash
#SBATCH --job-name=s2s-comb
#SBATCH --partition=cscc-gpu-p
#SBATCH --exclude=gpu-05,gpu-51,gpu-54
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out
#SBATCH --error=/home/isha.tariq/seva_to_shaper/logs/%x-%j.out

# [STEP 8] The generated-VIEWS conditions: ShapeR is given the orbit frames as extra
# every object, so the rectified-flow noise is identical across conditions and a difference
# is caused by the point conditioning alone -- the only way to see an effect smaller than
# the sampler's own spread. Conditions: slam, pi3, kv, union, gap1, gap4; seeds 0,1,2.
#   sbatch slurm/07_seeded.sbatch
set -eo pipefail
E=/home/isha.tariq/seva_to_shaper
RUNNER=/home/isha.tariq/shapeR_testing/exp_object_aware/scripts
source /home/isha.tariq/shapeR_testing/slurm/env.sh
require_env; require_assets; require_gpu; activate_shaper
echo "=== host $(hostname) ==="
mapfile -t PKLS < <(find "$E/outputs/pkls" -maxdepth 1 \
  \( -name "*__vg80.pkl" -o -name "*__v80.pkl" -o -name "*__vgap.pkl" \) | sort)
echo "${#PKLS[@]} pkls"
cd "$RUNNER"
for SEED in 0 1 2; do
  echo; echo "### seed $SEED"
  python run_shaper_seeded.py --input_pkls "${PKLS[@]}" \
      --out_dir "$E/outputs/seeded/s$SEED" --seed "$SEED" --skip_done \
      ${SAVE_MESHES:+--save_meshes}
done
cd "$E"; echo "### summarise"; python scripts/summarise_seeded.py
echo "### done"
